## arXiv:2606.09030v1[cs.LG]8Jun2026

#### TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs

Hyeongwon Jang1*, Gyouk Chu1∗, Changhun Kim2,3, Joonhyung Park1, Hangyul Yoon1, Eunho Yang1,2 1KAIST, 2AITRICS, 3University of Wisconsin-Madison {janghw0911, kyouwook, deepjoon, hangyulmd, eunhoy}@kaist.ac.kr changhun.kim@cs.wisc.edu

###### Abstract

Clinical early warning systems built on electronic health records, in which clinical observations are recorded as irregularly sampled medical time series (ISMTS), must deliver both calibrated risk scores for patient triage and interpretable rationales that clinicians can verify. Large Language Models (LLMs) have been explored for this task, yet they collapse graded clinical risk into overconfident binary predictions. This risk polarization undermines both calibration and cross-patient comparability. To address this, we propose TRIAGE, a framework that trains an LLM to generate dialectical reasoning over competing clinical outcomes by eliciting outcome-specific rationales. This dialectical formulation mitigates risk polarization, enabling a single LLM to yield continuous risk scores grounded in explicit clinical reasoning. Evaluated on three ISMTS benchmarks, TRIAGE achieves an average AUPRC improvement of 3.3% and reduces calibration error by 81% compared to the competitive baselines. An LLM-as-a-judge assessment further shows that our rationales surpass post-hoc explanations from the baseline by 20% in clinical reasoning quality. The source code is available at https:

//github.com/HyeongWon-Jang/TRIAGE.

###### 1 Introduction

Predicting adverse events such as mortality or disease onset from Electronic Health Records (EHRs) is central to early-warning systems in critical care. Clinical observations in EHRs, such as vital signs and laboratory tests, are sparsely recorded at irregular intervals with substantial missingness and thus commonly treated as irregularly sampled medical time series (ISMTS) (Sun et al., 2020). Beyond simply predicting the correct outcome, an effective clinical decision support system necessitates two properties that existing systems rarely provide together: a well-calibrated, cross-patient comparable

*Equal Contribution.

Table 1: Comparison of LLM-based approaches for ISMTS risk prediction. Cont.: continuous risk score; NL: natural language reasoning.

Methods Example Risk Cont. NL Implicit probability

Q: Sepsis≤6h? A: Yes 0.73 ✓ ✗

(HeLM, EHR-R1)

Reasoning + Answer

Q: Sepsis≤6h? A: [Rationale] → Yes 1 ✗ ✓

(KARE, OpenTSLM)

Q: Sepsis≤6h? A: [Rationale] → Yes 0.89 ✓ ✓

TRIAGE

risk score for triage and resource allocation, and a natural language explanation grounded in reasoning over clinical knowledge that clinicians can evaluate and trust.

Existing approaches address these requirements only in isolation. Specialized deep learning models for ISMTS (Che et al., 2018; Horn et al., 2020; Luo et al., 2024) achieve strong predictive performance through dedicated architectures tailored to irregular sampling and inter-variable dependencies but produce opaque predictions with no accompanying rationale. Post-hoc explainability methods, including general-purpose techniques (Lundberg and Lee, 2017; Sundararajan et al., 2017; Shrikumar et al., 2017) and those designed for time series (Crabbé and Van Der Schaar, 2021; Liu et al., 2024; Jang et al., 2025), can identify contributing variables or time points, but their explanations remain non-linguistic and limited to per-point attribution scores–not the higher-level abstractions that clinicians reason over.

To bridge this gap, large language models (LLMs) are natural candidates, as they internalize broad medical knowledge and can articulate complex reasoning in natural language. Yet existing LLM-based approaches for ISMTS reproduce this same divide. As in Table 1, two strategies have emerged, each delivering only one of the two properties. One line of work (e.g., HeLM (Belyaeva et al., 2023) and EHR-R1 (Liao et al., 2025)) extracts a continuous risk score from the model’s

implicit answer-token probability, which is known to be better than verbalized probability (Gu et al., 2024), thereby treating the LLM as a classifier without any natural language reasoning. The other (e.g., KARE (Jiang et al., 2025) and OpenTSLM (Langer et al., 2025)) elicits reasoning alongside the prediction but optimizes solely for the discrete answer, yielding risk estimates that are not comparable across patients.

As a first step, we scrutinize this challenge through a preliminary study, examining why current LLMs struggle to provide both continuous risk scores and clinically grounded explanations. We uncover a risk polarization problem: when natural language reasoning is elicited before risk scoring, the score is pushed toward the extremes. First, rationales typically pre-commit to a single outcome (e.g., “Therefore, this patient is likely to die. Answer:”), making the final prediction nearly deterministic given the preceding biased text. Second, rationales are typically one-sided, presenting only evidence that supports the committed outcome rather than evidence bearing on all possible outcomes. This issue is exacerbated in ISMTS, where trajectories contain coexisting signals of deterioration and stabilization. Consequently, simply prompting an LLM to reason and then predict is fundamentally insufficient for jointly obtaining clinically grounded explanations and cross-patient comparable risk scores.

Motivated by these observations, we propose Time Series Reasoning by Inspecting Alternative Outcomes for Grounded Risk Estimation (TRIAGE), a framework that addresses risk polarization by grounding risk estimation in dialectical reasoning over alternative outcomes. Rather than steering reasoning toward a single outcome, TRIAGE generates a dedicated rationale for each candidate outcome and derives the final risk from the LLM’s implicit probability conditioned on these rationales, jointly delivering continuous risk estimates and natural language explanations in a single pass. TRIAGE fosters this through a two-stage training pipeline: (i) Dialectical Reasoning Supervision followed by (ii) Self-Refinement, yielding well-calibrated and discriminative risk estimates.

To summarize, our contributions are threefold:

• We identify a risk polarization problem in LLM-based clinical risk prediction on ISMTS: eliciting reasoning collapses the implicit prob-

ability because (1) rationales pre-commit to a single outcome, driving the probability to near-certainty, and (2) rationales exhibit onesided confirmation bias rather than weighing alternative outcomes.

- • To address this, we propose TRIAGE, which decomposes reasoning into outcomeconditioned rationales and derives a risk from the resulting implicit probability distribution, replacing one-sided rationalization with dialectical deliberation over alternative outcomes.
- • Extensive experiments show that TRIAGE, instantiated on a small open-source LLM, improves AUPRC by 3.3% and reduces calibration error by 81% over various strong baselines, while producing more clinically reasonable rationales grounded in patient-specific evidence.

###### 2 Related Work

Irregularly sampled time series. Prior work has developed specialized architectures for irregularly sampled time series, including RNN-based models (Che et al., 2018), ODE-based continuoustime models (Chen et al., 2018; Rubanova et al., 2019), set-based encoders (Horn et al., 2020), interpolation-based models (Shukla and Marlin, 2019), attention-based models (Shukla and Marlin, 2021; Zhang et al., 2023), transformer-based models (Tipirneni and Reddy, 2022; Li et al., 2023), and graph-based models (Zhang et al., 2022; Luo et al., 2024, 2025). While these methods advance representation learning and predictive accuracy on irregular time series, they do not produce natural language explanations. In contrast, we study LLMbased clinical risk prediction from irregularly sampled medical time series, where risk estimates are coupled with textual reasoning.

LLMs for clinical time series. LLM-based methods for clinical time series, spanning both general time series modeling and EHR-based prediction, fall into two dominant categories. The first, answertoken risk scoring, treats the LLM as a classifier and extracts the risk from its answer-token probability (Belyaeva et al., 2023; Liao et al., 2025). The second, reasoning with hard-label prediction, produces a rationale with a discrete class label optimized for correctness (Jiang et al., 2025; Liu et al., 2025a; Langer et al., 2025). A separate line of work

offloads prediction to a neural model and uses the LLM only for auxiliary rationales, contextual summaries, or discrete predictions (Nguyen et al., 2024; Lee et al., 2025; Jiang et al., 2026). We instead keep reasoning and risk scoring within a single LLM that generates natural language reasoning and produces a patient-comparable risk grounded in it.

###### 3 Problem Setup

We formulate clinical risk prediction from ISMTS as a supervised classification problem. Let D = {(si,zi,yi)}Ni=1 denote a cohort of N patients. The temporal record si is a multivariate ISMTS represented as si = {(ti,k,vi,k,mi,k)}Lk=1i , where Li is the total number of observations and each tuple specifies the time ti,k, value vi,k, and variable index mi,k ∈ {1,...,D} of the k-th measurement among D clinical variables. The static descriptor zi ∈Rds encodes time-invariant patient attributes such as age and gender, and yi∈{0,...,C − 1} is the clinical outcome label among C classes.

Our objective is to learn a predictor that outputs a patient-specific class distribution pi = (pi,0,...,pi,C−1), where pi,c ≥ 0 is the probability that patient i belongs to class c. While a discrete prediction can be obtained as yˆi = arg maxc pi,c, our primary quantity of interest is the probability vector, which yields a discriminative risk estimate over the candidate outcomes for each patient.

###### 4 Preliminary Study

In this section, we identify the causes of standard LLM reasoning’s failure at clinical risk prediction. A natural approach is to have the LLM reason over the patient record before prediction so that the risk score is grounded in explicit clinical evidence. Through a preliminary study, we show that this approach instead collapses the score distribution into a degenerate extreme. We term the resulting failure the risk polarization problem and identify its two underlying causes.

Setup. We study in-hospital mortality prediction with gpt-oss-120b (Agarwal et al., 2025), reporting both discrimination (AUROC, AUPRC) and calibration (ECE, Brier score (BS)). The “risk score” refers to the model’s implicit probability: the softmax probability assigned to the positive-class answer token at the logit level, rather than any verbalized number (motivated in Appendix A.2). Full details are in Appendix A.

Reasoning causes risk polarization. We compare reasoning-then-answer, where the model generates a rationale before predicting (e.g., [Rationale] → Yes/No), against an answer-only baseline (e.g., Yes/No without reasoning). Under the answer-only baseline, the predicted-class probability averages 86.4%±18.8% across patients, providing sufficient spread to differentiate cases. Introducing reasoning completely collapses this signal: the predicted-class probability exceeds 99.98% on every patient with near-zero variance, leaving no basis for cross-patient ranking.

Reasoning is typically pre-committed. Under the reasoning-then-answer paradigm, regardless of the preceding reasoning quality, rationales tend to terminate in an explicit verdict immediately before the answer position (e.g., “Therefore, this patient is likely to die. Answer:”). To quantify this tendency, we use gpt-oss-120b itself as an LLM-as-judge to classify whether each generated rationale closes with such a committed verdict, and find that 71.7% of rationales exhibit this pattern (judge prompts are provided in Appendix G). Once such a verdict appears in the prefix, it dominates the conditioning context. Although the original clinical evidence remains part of the context, the answer token is overwhelmingly steered by the nearby verdict, concentrating the predicted-class probability on the outcome already stated in the rationale. The rationale thus acts as a self-imposed hard prior that absorbs residual uncertainty before the label is produced.

Table 2: Results on MIMIC-III in-hospital mortality prediction with gpt-oss-120b. AUROC (%), AUPRC (%), ECE, and BS are computed over 10 sampled responses.

Method AUROC ↑ AUPRC ↑ ECE ↓ BS ↓ Standard 76.1 27.8 0.236 0.211 Two-sided 75.5 30.2 0.205 0.185

Reasoning is typically one-sided. Furthermore, we uncover that beyond the final verdict, the reasoning itself lacks balanced deliberation. Inspecting individual traces reveals that the model exhibits confirmation bias, committing to one outcome and citing only supportive evidence while disregarding countervailing signals (see Appendix A.4). This conflicts with clinical trajectories, where indicators of deterioration and stabilization may coexist (Jang et al., 2025). To test whether this tendency is systematic, we append a minimal two-sided instruction: “Weigh the evidence for survival and

| | |
|---|---|
| | |

Patient Data Prompt Reasoning over Alternative Outcomes

###### Risk Estimation

[Figure 1]

Task Definition

###### Continuous Risk Score

- ● Stable BP
- ● Improving oxygenation
- ● Normal creatinine
- ● Improving GCS
- ● … Deterioration Rationale
- ● High lactate
- ● Hypernatremia
- ● Low body temperature
- ● Leukocytosis
- ● …

###### P(“mortality” | …) = 0.73

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

P( “survival” | …) = 0.27

[Figure 6]

Static Features

| |
|---|

| | | | | | |
|---|---|---|---|---|---|

LLM Stabilization Rationale

|Softmax|
|---|

[Figure 7]

0 1 Low risk High risk

0.73

ISMTS

Order Interchangeable

Stage 1. Dialectical Reasoning Supervision

###### Stage 2. Self-Refinement

Stabilization Outcome

[Figure 8]

[Figure 9]

Stabilization ONLY

[Figure 10]

Concat Rationales

Assumption

[Figure 11]

Rationale

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|Response 1<br>Response 2<br><br><br>Response G<br><br>[Figure 18]<br><br>Response 1<br><br>Response 2<br><br><br>Response G<br><br>[Figure 19]|
|---|

CE &

GRPO Update

Strong

Reward

LLM

Small

Patient Batch

Small

Modeling

LLM

Final

LLM

[Figure 20]

Dataset

Deterioration ONLY Rationale

Decision Token

Rollouts for

Deterioration

Patient 1, 2, …, B

SFT

Outcome

Assumption

Figure 1: Overview of TRIAGE. (Upper) TRIAGE represents irregular patient records as language inputs and inspects alternative outcomes to estimate patient risk. (Lower) The training pipeline consists of dialectical reasoning supervision followed by self-refinement.

for in-hospital death, then decide.” This improves AUPRC and reduces ECE and Brier score with comparable AUROC (Table 2), confirming onesided reasoning as a systematic limitation and motivating dialectical deliberation as a better inductive bias.

In summary, our preliminary study yields two design principles. (P1) Reasoning must be structured to preserve a continuous risk score rather than collapse into a single committed outcome. (P2) Reasoning should weigh the evidence for each candidate outcome separately.

###### 5 TRIAGE

Guided by the design principles from Section 4, in this section, we propose Time Series Reasoning by Inspecting Alternative Outcomes for Grounded Risk Estimation (TRIAGE). TRIAGE introduces two components: (i) a reasoning procedure in which outcome inspection addresses P2 by generating a dedicated rationale for each candidate outcome, and risk estimation addresses P1 by deriving a graded risk from the model’s implicit outcometoken distribution (Section 5.1); and (ii) a two-stage training pipeline of Dialectical Reasoning Supervision followed by Self-Refinement (Section 5.2). Figure 1 provides an overview of the framework, and the detailed algorithm is provided in Appendix B.

- 5.1 Dialectical Reasoning over Alternative Outcomes

This section details TRIAGE’s reasoning procedure, organized into three components: input repre-

sentation, outcome inspection, and risk estimation. The procedure follows the two principles from Section 4. It inspects each candidate outcome through a dedicated rationale (P2) and reads risk from the model’s implicit outcome-token distribution at a fixed answer position (P1). This design preserves a continuous risk signal rather than collapsing into a single committed outcome.

Input representation. Following set-based encoding for irregular time series (Horn et al., 2020), we serialize only the observed measurements in a variable-centric format rather than representing the full observation matrix. The input prompt xi for patient i concatenates a structured task definition P, textualized time-invariant attributes tzi (e.g., age, gender), and temporally ordered observations tsi:

xi = concat(P, tzi, tsi). (1) An example prompt is provided in Figure 6.

Outcome inspection. As shown in Section 4, standard LLM reasoning tends to align with a single outcome direction, citing only supportive evidence while disregarding countervailing signals. This one-sided tendency is especially problematic for ISMTS, where patient trajectories frequently exhibit concurrent indicators of both deterioration and stabilization (Jang et al., 2025).

To address this, TRIAGE generates a dialectical reasoning chain that separately examines each candidate outcome before producing a final prediction (Figure 1(a)). For a binary prediction task with

candidate outcomes Y = {y+,y−}, the model produces a dedicated rationale ryk for each yk ∈ Y, articulating the evidence from the patient’s observations that supports that specific outcome:

chain = [ry1, ry2, yˆ], {y1,y2} = {y−,y+},

(2) where ryk is the rationale for outcome yk, yˆ is the final predicted outcome token. The order of the two rationales is interchangeable. This structure realizes P2 by requiring the model to consider both sides of the clinical evidence before reaching a decision, surfacing signals that one-sided reasoning would discard.

Risk estimation. Even with dialectical rationales, a reasoning chain that concludes with a verdict before the answer token reintroduces the risk polarization identified under P1: rationale commits to one direction, driving the next-token probability to nearcertainty. To preserve a graded risk signal, TRIAGE terminates the reasoning chain with a “## Final Decision” header followed directly by a single outcome token, without any intermediate summary or verdict. We map each outcome yk ∈ Y to a designated token (e.g., “0” for negative, “1” for positive) and extract the corresponding logit ℓk at this fixed position. The risk score is then computed as: P(yk | x) = exp(ℓk)/ k′ exp(ℓk′).

Because the preceding rationales present evidence for both outcomes without committing to a conclusion, the model’s internal distribution at this position reflects a graded assessment, yielding continuous and cross-patient comparable risk estimates rather than the near-deterministic outputs observed with standard reasoning.

###### 5.2 Training Pipeline

The reasoning procedure described above does not emerge from standard prompting or conventional fine-tuning. Thus, we introduce a novel two-stage pipeline of TRIAGE: Stage 1 instills the structured reasoning behavior via supervised fine-tuning (SFT) on synthesized outcome-conditioned rationales, and Stage 2 refines the model through reinforcement learning (RL) on its own samples to improve both discrimination and calibration.

Stage 1: Dialectical reasoning supervision. The goal of this stage is to teach the model to produce outcome-conditioned rationales. As illustrated in the left panel of Figure 1(b), we elicit rationales from a strong LLM separately for each candidate

outcome: the LLM is prompted to assume that a given outcome holds and to identify only the patient features that support it. This process must adhere to two constraints. First, the LLM should not reference any alternative outcome, ensuring that each rationale remains strictly outcome-specific rather than contrastive. Second, the LLM should not fabricate unobserved evidence. In cases where no supporting evidence exists, the rationale is left blank. These constraints yield concise, feature-grounded rationales while discouraging hallucinated or contrastive content. Full prompts are provided in Appendix G.

The resulting rationales are concatenated with the ground-truth answer token to form reasoning traces following Equation 2. Since the rationale ordering is interchangeable, we include both orderings as data augmentation. A small LLM is then fine-tuned on these traces with standard SFT.

Stage 2: Self-refinement. After SFT on offpolicy dialectical reasoning traces built from a strong model, the model can generate high-quality rationales. However, it does not fully address the training–inference mismatch (Ranzato et al., 2015; Bengio et al., 2015), where the model must condition on its own generated prefixes at inference time rather than on reference trajectories.

To address this limitation, we further post-train the model with Group Relative Policy Optimization (GRPO; Shao et al., 2024). The cross-entropy objective is applied only to the final decision token to supervise the implicit predictive probability, whereas the preceding reasoning tokens are optimized by the GRPO objective:

L(θ) = LGRPO(θ) + λ · LCE(θ), (3)

  1

  ,

G

LGRPO(θ) = −E

(Jj,τ(θ) − βDKL(πθ ∥ πref))

Lr

j=1 τ∈Tjr

G

|Tjr|, Jj,τ(θ) = min wj,τ(θ)Aˆj,τ, w¯j,τ(θ)Aˆj,τ ,

Lr =

j=1

w¯j,τ(θ) = clip(wj,τ(θ), 1 − ϵlow, 1 + ϵhigh) ,

where Tjr denotes the reasoning-token positions excluding the final decision token, wj,τ(θ) =

πθ(rj,τ|x,rj,<τ)

πθold(rj,τ|x,rj,<τ) is the token-level importance ratio, and Aˆj,τ is the advantage obtained by normalizing the group-level rewards {Rj}Gj=1.

While intra-sample rewards provide a learning signal, they fail to encourage cross-patient discrimination. To ensure risk scores are comparable across

Table 3: Performance comparison on clinical risk prediction benchmarks. AUROC and AUPRC report mean ± standard deviation (%), while Avg. Rank denotes the mean rank across all six metric columns. The best and second-best results are indicated in bold and underline, respectively.

P12 P19 MIMIC-III

Method

Avg. Rank AUROC AUPRC AUROC AUPRC AUROC AUPRC

###### ISMTS Baselines

GRU-D 86.9±1.2 56.7±2.6 89.3±0.6 56.2±3.0 85.1±0.2 48.7±0.7 3.42 mTAND 86.9±0.5 56.6±2.3 85.2±1.1 37.7±4.0 84.0±0.5 46.4±0.8 7.67 SeFT 86.5±1.0 54.9±3.4 88.6±0.4 51.6±2.2 84.1±0.3 45.4±1.0 7.33 Raindrop 82.6±1.5 45.4±1.2 85.5±1.1 52.2±3.0 81.0±0.6 37.2±1.4 9.25 STraTS 87.2±1.0 58.8±3.9 89.3±0.8 48.7±2.0 85.1±0.4 47.8±1.1 4.08 ViTST 85.5±0.8 49.7±3.4 88.5±0.5 45.9±4.6 82.2±0.3 39.5±1.7 8.67 KEDGN 87.8±1.1 58.3±3.2 88.2±0.5 53.5±3.0 84.7±0.4 48.5±0.9 4.00 Hi-Patch 87.3±0.8 57.0±3.7 88.7±1.6 52.1±4.3 84.6±0.6 46.2±0.3 5.08

###### Zero-shot LLMs

GPT-5.1 83.9±1.0 49.3±0.5 72.2±2.9 9.3±0.9 – – 10.50 gpt-oss-120b 81.2±0.7 43.4±3.0 64.2±2.1 6.7±0.6 76.7±0.4 31.4±0.7 11.67

Ours

TRIAGESFT 86.9±1.0 56.4±1.9 88.9±1.0 52.2±3.2 86.4±0.3 51.4±0.9 4.25 TRIAGESFT+RL 87.3±1.2 59.0±3.4 89.3±1.0 53.8±1.4 86.7±0.2 54.1±0.5 1.58

patients, we define rewards at the batch level. Inspired by Yuan et al. (2021), we compare each sample’s score against the average of opposite-class samples in the batch to enhance separation:

 

1 |B−|

Lsurr(σi,j, σ¯i′), if yi = 1,

−

i′∈B−

Ri,j =

1 |B+|

Lsurr(¯σi′, σi,j), if yi = 0,

−



i′∈B+

Lsurr(σ+, σ−) = m − (σ+ − σ−) 2+ , σi,j = ℓ1 − ℓ0,

(4)

where [z]+ = max{z,0}, m is the margin, B− and B+ are the sets of negative and positive samples in the batch, σi,j is the log-odds of response ri,j, and σ¯i = G1 Gj=1 σi,j. This reward function promotes cross-patient comparability alongside the calibration benefit of on-policy training.

###### 6 Experiments

###### 6.1 Experimental Setup

Datasets and metrics. We conduct experiments on three irregular medical time series benchmarks, P12 (Silva et al., 2012), P19 (Reyna et al., 2020), and MIMIC-III (Johnson et al., 2016). P19 involves predicting sepsis onset within 6 hours, while P12 and MIMIC-III both target in-hospital mortality prediction. For discrimination, we report the area under the ROC curve (AUROC) and the area under the precision-recall curve (AUPRC). Given the severe class imbalance, we treat AUPRC as the primary discrimination metric (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015). For calibra-

Table 4: Calibration results on P12, P19, and MIMICIII. ECE and Brier score (BS); lower is better. The best results are indicated in bold.

P12 P19 MIMIC-III Method ECE BS ECE BS ECE BS

GRU-D 0.19 0.14 0.18 0.09 0.21 0.15 STraTS 0.16 0.13 0.19 0.10 0.20 0.14 KEDGN 0.17 0.13 0.21 0.10 0.22 0.15

GPT-5.1 0.09 0.10 0.16 0.08 – – gpt-oss-120b 0.26 0.19 0.32 0.19 0.23 0.16 TRIAGESFT 0.19 0.15 0.15 0.09 0.21 0.15 TRIAGESFT+RL 0.04 0.09 0.04 0.03 0.03 0.08

tion, we report the expected calibration error (ECE) and the Brier score (BS).

Baselines. We compare against representative ISMTS baselines, namely GRU-D (Che et al., 2018), mTAND (Shukla and Marlin, 2021), SeFT (Horn et al., 2020), Raindrop (Zhang et al., 2022), STraTS (Tipirneni and Reddy, 2022), ViTST (Li et al., 2023), KEDGN (Luo et al., 2024), and Hi-Patch (Luo et al., 2025). We further include GPT-5.11 (OpenAI, 2025) and gpt-oss-120b (Agarwal et al., 2025) as zero-shot LLMs.

Implementation details. To collect training data, we use GPT-5.1 (OpenAI, 2025) for P12 and P19, and Kimi K2 Thinking2 (Team et al., 2025) for MIMIC-III. Since the datasets are highly imbal-

- 1GPT-5.1 is not evaluated on MIMIC-III, as PhysioNet’s Credentialed Data Use Agreement prohibits transmitting MIMIC data to third-party LLM APIs (https://physionet. org/news/post/llm-responsible-use/).
- 2We run Kimi K2 Thinking inference locally for MIMICIII to comply with PhysioNet’s credentialing requirements.

GRU-D STraTS KEDGN TRIAGE

###### P12

###### P12

###### MIMIC-III

MIMIC-III

88

58

86

54

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

56

52

86

54

84

50

52

48

84

AUROC(%)

AUROC(%)

50

AUPRC(%)

AUPRC(%)

82

46

48

82

44

46

80

42

44

80

40

42

78

38

40

78

36

38

76

36

76

34

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

Removal Ratio

Removal Ratio

Removal Ratio

Removal Ratio

Figure 2: Performance comparison under the leave-variables-out setting on P12 and MIMIC-III.

anced, we apply oversampling following prior work (Zhang et al., 2022; Li et al., 2023; Luo et al., 2024), collecting multiple distinct rationales per minority sample through repeated LLM inference rather than duplicating identical samples. We apply a two-stage training pipeline to Qwen3-4BBase (Yang et al., 2025).

Further details on the experimental setup are in Appendix C.

Table 5: Ablation on the reasoning structure, on P12 with Qwen3-4B-Base.

Method AUROC AUPRC Zero-shot inference 69.7±1.9 26.7±0.8 SFT baselines

Answer-only 86.4±1.5 53.4±2.2 One-sided rationale† 83.8±1.0 43.1±3.6

TRIAGESFT 86.9±1.0 56.4±1.9 † Risk estimated by averaging over 10 sampled responses.

###### 6.2 Main Results

We evaluate TRIAGE on discrimination (AUROC, AUPRC; Table 3), calibration (ECE, Brier score; Table 4), and robustness under missing variables during inference (Figure 2).

Predictive performance. As shown in Table 3, zero-shot frontier LLMs (GPT-5.1 and gpt-oss120b) rank at the bottom with average ranks of 10.50 and 11.67. This indicates that generalpurpose reasoning alone does not transfer well to clinical risk prediction on ISMTS. With SFT alone, TRIAGE already reaches an average rank of 4.25, on par with the strongest ISMTS baselines, GRUD (3.42), KEDGN (4.00), and STraTS (4.08). After RL, TRIAGE achieves the best average rank of 1.58, placing first or second on every metric. Against the strongest baseline GRU-D, TRIAGE yields relative improvements of 0.8% in mean AUROC and 3.3% in mean AUPRC.

For calibration (Table 4), SFT alone brings TRIAGE to parity with the ISMTS baselines, though it still trails zero-shot GPT-5.1 by a small margin. Adding RL reduces the mean ECE by 80% and the mean Brier score by 49%, achieving the best scores on every benchmark.

Robustness under limited information. In practice, clinical risk predictors rarely observe every variable at training time due to sensor failures, protocol differences, or missing records. Following Luo et al. (2024), we randomly mask from 10% to 50% of variables in the validation and test sets of

P12 and MIMIC-III while keeping the training set unchanged.

Figure 2 shows that TRIAGE remains robust as variables are removed. On P12, TRIAGE matches the strongest baseline on AUPRC but trails slightly on AUROC. On MIMIC-III, it leads on AUPRC at every masking ratio and on AUROC at four out of five. Full results are in Appendix D.

###### 7 Further Analysis

In this section, we ablate the components of TRIAGE (Section 7.1) and examine whether the generated rationales are clinically meaningful (Section 7.2). Additional experiments and analyses are provided in Appendix E.

###### 7.1 Ablation Study

Reasoning structure. Table 5 compares TRIAGE against zero-shot inference and two SFT variants. Answer-only SFT removes reasoning entirely and trains the model to predict the label directly, treating the LLM as a classifier backbone. One-sided rationale SFT follows the common practice of supervising reasoning toward the ground-truth label only, using the correct-outcome traces from our collected data. Detailed prompts are in Appendix G.

Answer-only SFT substantially improves over zero-shot inference but provides no clinical justification. One-sided rationale SFT introduces clinical reasoning but inherits the risk-saturation problem identified in Section 4, where the answer-token

65

0.25

SFT checkpoint

SFT checkpoint

Sample-level reward

Sample-level reward

Batch-level reward

Batch-level reward

0.20

63

AUPRC(%)

0.15

Score

61

0.10

59

0.05

57

0.00

100 400 700 1000

ECE Brier

Gradient Steps

(a) Discrimination.

(b) Calibration.

Figure 3: Comparison of discrimination and calibration performance based on the reward design on P12 data.

probability collapses toward 1.0. Even after averaging predictions over 10 sampled outputs at 10× inference cost, it underperforms Answer-only SFT on both metrics, confirming that one-sided reasoning supervision can actively harm risk estimation. In contrast, TRIAGE’s dialectical reasoning achieves the best AUROC (86.9%) and AUPRC (56.4%) while retaining clinically grounded explanations.

RL reward design. We compare our batch-level reward against a conventional sample-level reward, in which each rollout receives Ri,jsample = log P(yi | xi, ri,j) followed by the same group normalization. This reward provides a direct correctness signal for each patient, akin to the standard outcome-based reward in GRPO, but does not explicitly encourage risk estimates that are comparable across patients.

Figure 3 shows that the batch-level reward yields noticeable improvements in both discrimination

- (Figure 3a), with higher AUPRC, and calibration
- (Figure 3b), with lower ECE and Brier scores. These results confirm that the inter-sample signal introduced by the batch-level reward provides a complementary learning objective that improves both the separability and calibration of predicted risk scores.

###### 7.2 Reasoning Analysis

Unlike existing ISMTS models, which require chaining a post-hoc XAI method with an LLM interpretation step to produce linguistic explanations, TRIAGE generates clinical rationales directly. To compare the quality of these explanations, we conduct an LLM-as-a-judge evaluation on 200 randomly sampled cases on the P12 dataset. As a baseline, we apply Integrated Gradients (IG; Sundararajan et al., 2017) to STraTS (the second-best model on P12) and prompt GPT-5.1 to interpret the explanation.

Following the IDEA assessment tool (Baker et al., 2015; Schaye et al., 2021), each judge evaluates reasoning traces along four dimen-

Table 6: LLM-as-a-judge evaluation of clinical reasoning quality on P12 samples. Baseline denotes STraTS + IG + GPT Interpretation. Maximum scores for I/D/E/A are 4/2/2/2, respectively (Schaye et al., 2021).

Criterion Baseline Ours

Interpretive summary (I) 2.526 3.429 (+0.902) Differential diagnosis (D) 1.234 1.218 (-0.016) Lead diagnosis exp. (E) 1.101 1.196 (+0.095) Alternative diagnosis exp. (A) 1.609 1.898 (+0.288)

Total 6.474 7.744 (+1.269)

sions: Interpretive summary, Differential diagnosis, Explanation of lead diagnosis, and Alternative diagnosis explained, with maximum subscores of 4/2/2/2, respectively, summing to a 0–10 total score. We employ three judge models: GPT-5.1 (OpenAI, 2025), Claude Sonnet 4.5 (Anthropic, 2025), and Gemini 3 Flash (Google, 2025). We query each judge three times per sample and report the mean of the nine resulting scores.

As shown in Table 6, TRIAGE achieves a higher total IDEA score (7.744 vs. 6.474). The largest gain appears on the interpretive summary criterion (+0.902), indicating that the model captures the patient’s baseline risk and temporal trajectory more effectively. The alternative diagnosis dimension also shows a notable improvement (+0.288), in line with our design choice to reason over alternative outcomes, while the remaining criteria perform comparably.

Qualitative results in Appendix F further illustrate that TRIAGE produces more patient-specific and clinically grounded reasoning traces.

###### 8 Conclusion

We have introduced TRIAGE, a framework that grounds clinical risk estimation in reasoning over alternative outcomes rather than a single committed trajectory. By inspecting each candidate outcome before deriving the final risk from the LLM’s implicit probability, our approach addresses the risk polarization problem that arises when reasoning is elicited before risk scoring. With dialectical reasoning supervision and self-refinement on a small open-source LLM, TRIAGE outperforms various competitive baselines while producing rationales that align with how clinicians weigh patient evidence. More broadly, this work signals a path forward for LLM-based clinical decision support, where predictive performance and grounded explanation are pursued together rather than as competing objectives.

###### Limitations

Our work has several limitations. First, our evaluation is restricted to binary prediction tasks, and extension to multi-class or multi-label clinical settings is left for future work. Second, the LLMbased pipeline is also considerably more expensive than lightweight baselines such as GRU-D, both in training and at inference time, since the model generates multi-step rationales before producing a prediction. This overhead is acceptable for typical clinical deployment but may be prohibitive where strict low-latency inference is required. Third, for reasoning evaluation, we use LLM-as-a-judge with the IDEA assessment tool (Baker et al., 2015; Schaye et al., 2021) in place of clinical expert assessment. We mitigate model-specific bias by aggregating across diverse judges and grounding our analyses in medical literature, but expert evaluation remains an important next step. Finally, generated rationales may contain inaccuracies or biases. TRIAGE is a research prototype, not a clinical tool, and its outputs should not substitute for qualified clinical judgment.

###### Ethical Considerations

MIMIC-III is accessed under the PhysioNet credentialed Data Use Agreement.3 Accordingly, no stage of our MIMIC-III pipeline relies on thirdparty LLM services: preprocessing, rationale generation, training, and evaluation (including all zeroshot LLM baselines) are performed locally with self-hosted, open-weight models.

###### References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1 others. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Anthropic. 2025. Claude Sonnet 4.5 System Card. System card. Accessed May 21, 2026.

Okan Arihan, Bernhard Wernly, Michael Lichtenauer, Marcus Franz, Bjoern Kabisch, Johanna Muessig, Maryna Masyuk, Alexander Lauten, Paul Christian Schulze, Uta C Hoppe, and 1 others. 2018. Blood urea nitrogen (bun) is independently associated with mortality in critically ill patients admitted to icu. PloS one, 13(1):e0191697.

3https://physionet.org/news/post/ llm-responsible-use/

Elizabeth A Baker, Cynthia H Ledford, Louis Fogg, David P Way, and Yoon Soo Park. 2015. The idea assessment tool: assessing the reporting, diagnostic reasoning, and decision-making skills demonstrated in medical students’ hospital admission notes. Teaching and learning in medicine, 27(2):163–173.

Anastasiya Belyaeva, Justin Cosentino, Farhad Hormozdiari, Krish Eswaran, Shravya Shetty, Greg Corrado, Andrew Carroll, Cory Y McLean, and Nicholas A Furlotte. 2023. Multimodal llms for health grounded in individual-specific data. In Workshop on Machine Learning for Multimodal Healthcare Data, pages 86– 102. Springer.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. 2015. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28.

Danny Castro, Sachin Patil, Muhammad Zubair, and Michael Keenaghan. 2024. Arterial blood gas. StatPearls.

Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. 2018. Recurrent neural networks for multivariate time series with missing values. Scientific reports, 8(1):6085.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. 2018. Neural ordinary differential equations. Advances in neural information processing systems, 31.

Jonathan Crabbé and Mihaela Van Der Schaar. 2021. Explaining time series predictions with dynamic masks. In International conference on machine learning, pages 2166–2177. PMLR.

Jesse Davis and Mark Goadrich. 2006. The relationship between precision-recall and roc curves. In Proceedings of the 23rd international conference on Machine learning, pages 233–240.

Google. 2025. Gemini 3 flash: frontier intelligence built for speed. https://blog.google/ products-and-platforms/products/gemini/ gemini-3-flash/.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Bowen Gu, Rishi J Desai, Kueiyu Joshua Lin, and Jie Yang. 2024. Probabilistic medical predictions of large language models. npj Digital Medicine, 7(1):367.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR.

Hrayr Harutyunyan, Hrant Khachatrian, David C Kale, Greg Ver Steeg, and Aram Galstyan. 2019. Multitask learning and benchmarking with clinical time series data. Scientific data, 6(1):96.

Max Horn, Michael Moor, Christian Bock, Bastian Rieck, and Karsten Borgwardt. 2020. Set functions for time series. In International Conference on Machine Learning, pages 4353–4363. PMLR.

Adrian O Hosten. 1990. Bun and creatinine. Clinical Methods: The History, Physical, and Laboratory Examinations. 3rd edition.

Hyeongwon Jang, Changhun Kim, and Eunho Yang. 2025. TIMING: Temporality-aware integrated gradients for time series explanation. In Forty-second International Conference on Machine Learning.

Pengcheng Jiang, Cao Danica Xiao, Minhao Jiang, Parminder Bhatia, Taha Kass-Hout, Jimeng Sun, and Jiawei Han. 2025. Reasoning-enhanced healthcare predictions with knowledge graph community retrieval. In International Conference on Learning Representations, volume 2025, pages 81785–81830.

Yushan Jiang, Wenchao Yu, Geon Lee, Dongjin Song, Kijung Shin, Wei Cheng, Yanchi Liu, and Haifeng Chen. 2026. Timexl: Explainable multi-modal time series prediction with llm-in-the-loop. Advances in Neural Information Processing Systems, 38:135431– 135467.

Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. 2016. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1):1–9.

Damjan Kalajdzievski. 2023. A rank stabilization scaling factor for fine-tuning with lora. arXiv preprint arXiv:2312.03732.

Feiyang Kang, Michael Kuchnik, Karthik Padthe, Marin Vlastelica, Ruoxi Jia, Carole-Jean Wu, and Newsha Ardalani. 2025. Quagmires in sft-rl post-training: When high sft scores mislead and what to use instead. arXiv preprint arXiv:2510.01624.

Arif Khwaja. 2012. Kdigo clinical practice guidelines for acute kidney injury. Nephron Clinical Practice, 120(4):c179–c184.

Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Patrick Langer, Thomas Kaar, Max Rosenblattl, Maxwell A Xu, Winnie Chow, Martin Maritsch, Robert Jakob, Ning Wang, Juncheng Liu, Aradhana Verma, and 1 others. 2025. Opentslm: Time-series language models for reasoning over multivariate medical text-and time-series data. arXiv preprint arXiv:2510.02410.

Geon Lee, Wenchao Yu, Kijung Shin, Wei Cheng, and Haifeng Chen. 2025. Timecap: Learning to contextualize, augment, and predict time series events with large language model agents. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 18082–18090.

Zekun Li, Shiyang Li, and Xifeng Yan. 2023. Time series as images: Vision transformer for irregularly sampled time series. Advances in Neural Information Processing Systems, 36:49187–49204.

Yusheng Liao, Chaoyi Wu, Junwei Liu, Shuyang Jiang, Pengcheng Qiu, Haowen Wang, Yun Yue, Shuai Zhen, Jian Wang, Qianrui Fan, and 1 others. 2025. Ehr-r1: A reasoning-enhanced foundational language model for electronic health record analysis. arXiv preprint arXiv:2510.25628.

Haoxin Liu, Chenghao Liu, and B Aditya Prakash. 2025a. A picture is worth a thousand numbers: Enabling llms reason about time series via visualization. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7486–7518.

Kezhao Liu, Jason Klein Liu, Mingtao Chen, and Yiming Liu. 2025b. Rethinking kl regularization in rlhf: From value estimation to gradient optimization. arXiv preprint arXiv:2510.01555.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025c. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783.

Zichuan Liu, Tianchun Wang, Jimeng Shi, Xu Zheng, Zhuomin Chen, Lei Song, Wenqian Dong, Jayantha Obeysekera, Farhad Shirani, and Dongsheng Luo. 2024. Timex++: Learning time-series explanations with information bottleneck. arXiv preprint arXiv:2405.09308.

Scott M Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

Yicheng Luo, Zhen Liu, Linghao Wang, Junhao Zheng, Binquan Wu, and Qianli Ma. 2024. Knowledgeempowered dynamic graph network for irregularly sampled medical time series. Advances in Neural Information Processing Systems, 37:67172–67199.

Yicheng Luo, Bowen Zhang, Zhen Liu, and Qianli Ma. 2025. Hi-patch: Hierarchical patch gnn for irregular multivariate time series. In Forty-second International Conference on Machine Learning.

Victoria Mank, Waqas Azhar, and Kevin Brown. 2026. Leukocytosis. In StatPearls [Internet]. StatPearls Publishing.

Sunil Munakomi, Konstantinos Margetis, and Lindsay M Iverson. 2026. Glasgow coma scale. In StatPearls [Internet]. StatPearls Publishing.

Tuan Dung Nguyen, Thanh Trung Huynh, Minh Hieu Phan, Quoc Viet Hung Nguyen, and Phi Le Nguyen. 2024. Carer-clinical reasoning-enhanced representation for temporal health risk prediction. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10392–10407.

OpenAI. 2025. Gpt-5.1 instant and gpt-5.1 thinking system card addendum. System card, OpenAI.

Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2015. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732.

Matthew A Reyna, Christopher S Josef, Russell Jeter, Supreeth P Shashikumar, M Brandon Westover, Shamim Nemati, Gari D Clifford, and Ashish Sharma. 2020. Early prediction of sepsis from clinical data: the physionet/computing in cardiology challenge 2019. Critical care medicine, 48(2):210–217.

Yulia Rubanova, Ricky TQ Chen, and David K Duvenaud. 2019. Latent ordinary differential equations for irregularly-sampled time series. Advances in neural information processing systems, 32.

Takaya Saito and Marc Rehmsmeier. 2015. The precision-recall plot is more informative than the roc plot when evaluating binary classifiers on imbalanced datasets. PloS one, 10(3):e0118432.

Verity Schaye, Louis Miller, David Kudlowitz, Jonathan Chun, Jesse Burk-Rafel, Patrick Cocks, Benedict Guzman, Yindalon Aphinyanaphongs, and Marina Marin. 2021. Development of a clinical reasoning documentation assessment tool for resident and fellow admission notes: a shared mental model for feedback. Journal of General Internal Medicine, 37(3):507.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256.

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. 2017. Learning important features through propagating activation differences. In International conference on machine learning, pages 3145–3153. PMlR.

Satya Narayan Shukla and Benjamin Marlin. 2019. Interpolation-prediction networks for irregularly sampled time series. In International Conference on Learning Representations.

Satya Narayan Shukla and Benjamin Marlin. 2021. Multi-time attention networks for irregularly sampled time series. In International Conference on Learning Representations.

Ikaro Silva, George Moody, Daniel J Scott, Leo A Celi, and Roger G Mark. 2012. Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012. In 2012 computing in cardiology, pages 245–248. IEEE.

Leslie V Simon, Muhammad F Hashmi, and Mitchell W Farrell. 2017. Hyperkalemia.

Chenxi Sun, Shenda Hong, Moxian Song, and Hongyan Li. 2020. A review of deep learning methods for irregularly sampled medical time series data. Preprint, arXiv:2010.12493.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319– 3328. PMLR.

Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

Sindhu Tipirneni and Chandan K Reddy. 2022. Selfsupervised transformer for sparse and irregularly sampled multivariate clinical time-series. ACM Transactions on Knowledge Discovery from Data (TKDD), 16(6):1–17.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2026. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244.

Zhuoning Yuan, Yan Yan, Milan Sonka, and Tianbao Yang. 2021. Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3040–3049.

Jiawen Zhang, Shun Zheng, Wei Cao, Jiang Bian, and Jia Li. 2023. Warpformer: A multi-scale modeling approach for irregular clinical time series. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3273– 3285.

Xiang Zhang, Marko Zeman, Theodoros Tsiligkaridis, and Marinka Zitnik. 2022. Graph-guided network for irregularly sampled multivariate time series. In International Conference on Learning Representations.

###### A Preliminary Study Details

###### A.1 Implementation Details

We conduct a simple experiment on in-hospital mortality prediction on MIMIC-III (Johnson et al., 2016) using gpt-oss-120b (Agarwal et al., 2025), a sparse mixture-of-experts Transformer with 117B total parameters and 5.1B active parameters per token. We load the official release weights in native MXFP4 and sample at temperature 0.7 with reasoning_effort=high. We provide the full prompt in Appendix G.

- Table 7: In-hospital mortality prediction on MIMIC-III with gpt-oss-120b in non-reasoning mode. AUROC and AUPRC are reported in %.

Risk score AUROC ↑ AUPRC ↑ ECE ↓ Brier ↓

Verbalized 69.6 20.6 0.343 0.426 Implicit 68.0 23.6 0.075 0.103

- Table 8: Concentration of risk scores on MIMIC-III. Top-k is the share (in %) of outputs falling on the k most frequent values; #Distinct counts unique values produced across all test samples.

Method Top-1 Top-3 Top-7 #Distinct gpt-oss-120b

Verbalized 37.0 55.0 75.3 55

- Implicit 2.3 6.0 10.5 1375

Qwen3-8B

Verbalized 47.5 63.5 81.5 53

- Implicit 3.3 9.6 21.1 96

###### A.2 Implicit Probabilities as Risk Scores

To choose an appropriate risk score, we compare two ways of eliciting mortality probabilities from the LLM. The verbalized probability pverb ∈ [0,1] is the numeric estimate in the model’s response text, while the implicit probability P(y=1 | prompt) is read off the next-token distribution over the answer tokens. Prior work reports better discrimination from implicit probabilities on binary medical prediction (Gu et al., 2024); we revisit this in our setting. For a controlled comparison, this preliminary study is run on MIMIC-III with reasoning disabled (reasoning_effort=None), on gpt-oss-120b and Qwen3-8B (Yang et al., 2025) for the concentration analysis.

Discrimination and calibration. Table 7 reports the comparison. The implicit probability improves AUPRC at a small cost to AUROC and reduces ECE and Brier score by roughly 4×. The cause is anchor concentration in verbalized outputs (Table 8): seven values cover 75.3% of gpt-oss-120b

35

30

###### AUPRC(%)

25

Estimated risk

20

Verbalized risk (1 sample)

1 10 50

Number of samples per question

Figure 4: AUPRC of the outcome-based and verbalized risk scores under reasoning, on MIMIC-III with gpt-oss120b.

estimates and three values cover 63.5% of Qwen38B estimates. These anchors reflect response priors rather than patient-level risk, flattening ranking and inflating calibration error. The implicit probability varies continuously over [0,1] and preserves fine-grained differences.

Trainability. A second reason favors the implicit channel: it admits a training signal. The implicit probability is a differentiable function of the next-token logits, whereas the verbalized output is produced by discrete sampling and is nondifferentiable with respect to model parameters. For any gradient-based alignment of model risk estimates with patient outcomes, the verbalized output is structurally unusable.

Taken together, we adopt the implicit probability as the risk score throughout. A.3 Verbalized Probabilities for Zero-Shot

###### Evaluation

For zero-shot LLM benchmarking under reasoning, two candidates are available for a graded risk score: (i) an outcome-based estimator that approximates the model’s underlying probability by averaging binary answers across many sampled responses, and (ii) the verbalized probability stated in a single response. Figure 4 compares the two in AUPRC on MIMIC-III with gpt-oss-120b. The outcome-based estimator surpasses the verbalized score only at around 30 sampled responses per question, at a proportional inference cost. The verbalized probability is a clean single-number readout from a single response, providing a far more efficient zero-shot signal. We therefore adopt it whenever reporting zero-shot LLM baselines in subsequent sections.

Algorithm 1 TRIAGE Inference: Dialectical Reasoning over Alternative Outcomes

Input: Policy πθ, patient i with prompt template P, timeinvariant features tzi, observations tsi, outcome set Y = {y−, y+}

- 1: xi ← concat(P, tzi, tsi)
- 2: Sample [ry1, ry2, "## Final Decision"] ∼ πθ(· | xi)
- 3: Read logits ℓ0, ℓ1 at the token position after the header
- 4: Compute P(y+ | xi) = exp(ℓ1) k exp(ℓk) Output: Risk estimate P(y+ | xi)

###### A.4 One-Sided Reasoning Traces

We observe that LLM reasoning traces typically confirm a single outcome rather than weigh evidence on both sides. MIMIC-III is a credentialed dataset, so we do not reproduce raw patient records here. We instead draw an illustrative case from the publicly available PhysioNet 2012 Challenge (P12) (Silva et al., 2012), which targets the same task as MIMIC-III (in-hospital mortality prediction from early ICU trajectories) and exhibits the same one-sidedness under our prompting protocol.

To make this behavior easy to inspect, we set reasoning_effort=None for gpt-oss-120b and prompt the model with “Provide your clinical reasoning, then state your final prediction.” This places the entire rationale in the visible response, where the one-sided pattern is straightforward to identify.

Figure 7 shows two reasoning traces out of ten sampled from gpt-oss-120b on the same patient. One trace cites only deterioration signals and predicts death; the other cites only stabilization signals and predicts survival. Each trace is internally coherent, yet neither weighs evidence from the opposite direction.

###### B Algorithm

- Algorithm 1 and Algorithm 2 outline the detailed procedure of TRIAGE used in our experiments.

###### C Experimental Setup Details C.1 Datasets

We evaluate our method on three representative medical time series classification benchmarks: P12 (Silva et al., 2012), P19 (Reyna et al., 2020), and MIMIC-III (Johnson et al., 2016). Table 9 summarizes their key statistics. All three are binary classification tasks with severe class imbalance, where the positive class forms a small minority in every case. Following prior works (Zhang et al., 2022;

Li et al., 2023), we use five random 8:1:1 splits for P12 and P19, since neither dataset provides an official partition. For MIMIC-III, we adopt the fixed 7:1.5:1.5 split defined by Harutyunyan et al. (2019) and run five random seeds. In both settings, we report the mean and standard deviation across the five runs. We provide a detailed description of each dataset below, including our preprocessing refinements over the original releases.

P12. The P12 dataset (Silva et al., 2012) is taken from the PhysioNet 2012 Challenge.4 Following Zhang et al. (2022); Li et al. (2023), we exclude 12 inappropriate samples, leaving 11,988 ICU patient records. Each record consists of irregularly sampled measurements from 36 sensors collected during the first 48 hours of an ICU stay, together with 5 static demographic features (Age, Gender, Height, ICUType, Weight). The task is binary in-hospital mortality prediction. We adopt the training, validation, and test splits released by Raindrop (Zhang et al., 2022).

In addition, we correct a parsing misalignment in the original preprocessing, where the five demographic features are extracted by row position rather than by parameter name. When a sensor reading at t=00:00 is interleaved with the demographic rows, the two streams contaminate each other. Real sensor values are then parsed as demographics, while later static fields such as Weight are pushed into the time series. This affects 207 of 12,000 patients (1.7%) and produces implausible values such as Height = 1 cm.

P19. The P19 dataset (Reyna et al., 2020) is taken from the PhysioNet 2019 Challenge on sepsis prediction.5 Each sample contains 34 irregularly sampled time series variables together with 6 static features (Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS). The task is binary classification of whether sepsis will occur within the next 6 hours. The public release used by prior work (Zhang et al., 2022; Li et al., 2023; Luo et al., 2024) comprises 38,803 ICU stays.

In addition, we exclude 65 patients whose 34 time series variables contain no non-zero finite observation within their reported sequence length, yielding 38,738 samples. Notably, all 65 excluded patients carry a positive sepsis label. Removing

- 4https://physionet.org/content/challenge-2012/

1.0.0/

- 5https://physionet.org/content/challenge-2019/

1.0.0/

- Algorithm 2 TRIAGE Training Pipeline

Input: Dataset D = {(xi,yi)}Ni=1, strong LLM πstrong, initial policy πθ, group size G, CE weight λ, margin m

- 1: Stage 1: Dialectical Reasoning Supervision
- 2: Initialize DSFT ← ∅
- 3: for each (xi,yi) ∈ D do
- 4: for each yk ∈ Y do
- 5: ryk ← πstrong RATIONALEPROMPT(xi,yk) ▷ outcome-specific, no fabrication
- 6: end for
- 7: DSFT ← DSFT ∪ [ry−,ry+,yi], [ry+,ry−,yi] ▷ order augmentation
- 8: end for
- 9: Fine-tune πθ on DSFT with token-level cross-entropy
- 10: Stage 2: Self-Refinement with GRPO
- 11: repeat
- 12: Sample minibatch B ⊂ D; B+ = {i ∈ B : yi = 1}, B− = {i ∈ B : yi = 0}
- 13: πθold ← πθ
- 14: for each i ∈ B, j = 1,...,G do
- 15: Sample chain ri,j ∼ πθold(· | xi); σi,j ← ℓ(1i,j) − ℓ(0i,j) ▷ log-odds at decision token
- 16: end for
- 17: Compute σ¯i = G1 j σi,j and rewards {Ri,j} via Equation (4)

- 18: Compute group-normalized advantages Aˆj,τ from {Ri,j}
- 19: Update πθ by minimizing L(θ) = LGRPO(θ) + λLCE(θ) via Equation (3) ▷ CE on decision token only, GRPO on reasoning tokens
- 20: until convergence Output: Trained policy πθ

- Table 9: Statistics of the three medical time series classification datasets. #Avg. obs. denotes the average number of observed measurements per sample. The missing ratio is computed with respect to each patient’s actual observation length. Pos. rate denotes the proportion of positive samples.

Dataset #Samples #Variables #Avg. obs. Missing ratio Pos. rate P12 11,988 36 401 84.9% 14.2% P19 38,738 34 238 80.5% 4.0% MIMIC-III 21,107 16 431 65.5% 13.2%

them reduces the positive count from 1,626 to 1,561, lowering the positive rate from 4.19% to 4.03%. Such records provide no measurement signal for any time series classifier. Retaining them inflates the apparent positive rate without contributing usable information, which is particularly problematic given the already low positive rate of P19. Because we deviate from the sample set used by prior work, we generate new reproducible 5 splits for all experiments on P19.

MIMIC-III. MIMIC-III (Johnson et al., 2016) is a publicly released critical care database distributed via PhysioNet.6 We adopt the benchmark prepro-

6https://physionet.org/content/mimiciii/1.4/

cessing pipeline of Harutyunyan et al. (2019) and target binary in-hospital mortality prediction over the first 48 hours after ICU admission. The resulting dataset comprises 21,107 patient stays, each described by 16 time series variables and one static feature (Height).

###### C.2 Metrics

AUROC and AUPRC follow their standard definitions. The Brier score is the mean squared error between the predicted probability and the binary outcome. ECE (Guo et al., 2017) measures the average gap between predicted confidence and empirical accuracy across M = 10 equal-width bins on [0,1]. For both Brier score and ECE, lower values

indicate better calibration.

###### C.3 Baselines

We employ two categories of baselines: zero-shot LLMs and ISMTS baselines.

For the zero-shot LLMs, we use GPT-5.1 (OpenAI, 2025) with reasoning_effort=medium and gpt-oss-120b with reasoning_effort=high. Based on the preliminary study in Appendix A.3, we use the verbalized probability produced by the model as the predicted risk. The prompt details for the zero-shot evaluation are provided in Appendix G.

For all ISMTS baselines, we use the Adam (Kingma and Ba, 2014) optimizer. To mitigate class imbalance, we follow the training protocol of (Zhang et al., 2022; Li et al., 2023; Luo et al., 2024) and upsample the minority class threefold while constraining each mini-batch to a 1:1 class ratio. AUPRC is the primary metric for irregular time series classification under class imbalance (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015). We therefore use validation AUPRC as our model selection criterion. Unless otherwise specified, we train each model for up to 50 epochs with an early stopping patience of 5. For each hyperparameter configuration, we conduct 5 runs and select the one with the highest mean validation AUPRC. Method-specific hyperparameter configurations and search spaces are described below.

GRU-D (Che et al., 2018). We reproduce the original authors’ Keras implementation7 in PyTorch with the original MLP classification head. Following the released code, we fix the input and recurrent dropout to 0.3. We tune the hidden dimension over {32,64,128,256,512,1024}, the learning rate over {5e-4,1e-3,5e-3,1e-2}, and the batch size over {32,64,128,256,512}.

mTAND (Shukla and Marlin, 2021). We follow the official implementation8 and adopt the mTANDFull classification variant. Following the released code, we fix the number of reference time points to 128, the time-embedding dimension to 128, the number of attention heads to 1, and the generator hidden size to 50. We tune the recurrent encoder hidden size over {32,64,128}, the latent dimension over {32,64,128}, the classification-loss

- 7https://github.com/PeterChe1990/GRU-D
- 8https://github.com/reml-lab/mTAN

weight over {1,5,10,50,100}, the learning rate over {5e-4,1e-3,5e-3,1e-2}, and the batch size over {32,64,128}.

SeFT (Horn et al., 2020). We reproduce the authors’ TensorFlow/Keras implementation9 in PyTorch. Following the values dominant across datasets in the original paper’s hyperparameter table, we fix the observation encoder to a 4-layer MLP, the attention aggregator to 2 layers with 4 heads, and the output readout to a 2-layer MLP of width 512. We tune the per-observation embedding dimension over {256,512}, the observation encoder MLP width over {64,128}, the shared dropout rate over {0.0,0.2,0.4}, the timeencoding dimension over {8,16}, the maximum time scale over {100,1000}, the learning rate over {1e-4,5e-4,1e-3,5e-3}, and the batch size over {32,64,128}.

Raindrop (Zhang et al., 2022). We follow the official implementation10. Following the original paper, we fix the batch size to 128. We tune the per-sensor observation embedding dimension over {2,4,8}, the number of Raindrop layers over {1,2,4}, the number of attention heads over {1,2,4}, dropout over {0.0,0.2,0.4}, and the learning rate over {1e-4,5e-4,1e-3,5e-2}.

STraTS (Tipirneni and Reddy, 2022). We follow the official PyTorch implementation11. Following the released code, we fix the learning rate to 5e-4, the batch size to 32, and the per-patient observation cap to 880. We tune the hidden dimension over {32,64,128}, the number of Transformer layers over {1,2,4}, the number of attention heads over {8,16,32}, and the shared dropout rate (used for both feature and attention dropout) over {0.0,0.2,0.4}.

ViTST (Li et al., 2023). We follow the official implementation12 and adopt the Vision-Text variant, which combines a Swin Transformer image encoder with a RoBERTa text encoder, fine-tuning both end-to-end. Following the original paper, we additionally render the inputs as a 4 × 4 grid for MIMIC-III. We fix the number of training epochs to 4 for P12 and MIMIC-III and 2 for P19, and fix

- 9https://github.com/BorgwardtLab/Set_

Functions_for_Time_Series

- 10https://github.com/mims-harvard/Raindrop
- 11https://github.com/sindhura97/STraTS
- 12https://github.com/Leezekun/ViTST

the batch size to 48. We tune the learning rate over {1e-5,2e-5,5e-5}.

###### KEDGN (Luo et al., 2024). We follow the offi-

- cial implementation13. Following the released code, we use per-dataset batch sizes of 512, 512, and 256 and learning rates of 1e-3, 5e-3, and 5e-3 for P12, P19, and MIMIC-III, respectively, and fix the textual source to ChatGPT-extracted descriptions encoded by BERT (d = 768). We tune the dimension of query vectors q over {5,7,9}, the dimension of variables’ node embeddings n over {7,9,11}, the proportion of the density score α over {1.0,2.0,3.0}, and the shared dimension of variables’ hidden state h and structured encoding representations k (with h = k) over {8,12,16}. Hi-Patch (Luo et al., 2025). We follow the offi-
- cial implementation14. Following the released code, we train each model for up to 20 epochs, fix the time-decay coefficient α to 1 and the learning rate to 1e-3, and use per-dataset batch sizes of 8, 64, and 16 for P12, P19, and MIMIC-III, respectively. We tune the hidden dimension over {16,32,64,128}, the number of attention heads over {1,2,4,8}, the number of graph-attention layers over {1,2}, and the patch size over {T/2,T/4,T/8,T/16,T/32}, where the history window T is set to 48 hours for P12 and MIMIC-III and 60 hours for P19, following the released code.

###### C.4 Implementation Details

Data collection. We collect outcome-specific rationales using GPT-5.1 for P12 and P19, and Kimi K2 Thinking (Team et al., 2025) for MIMIC-III. We first collect rationale separately for each candidate outcome. For each candidate, the LLM is prompted to assume that outcome and to identify only the patient features that support it. To keep each rationale strictly outcome-specific rather than contrastive, we prohibit the LLM from referencing any alternative outcome or fabricating unobserved evidence; when no supporting evidence appears, the rationale is left blank. These constraints suppress hallucinated and contrastive content, yielding concise, feature-grounded rationales.

To mitigate class imbalance, we follow prior work (Zhang et al., 2022; Li et al., 2023; Luo et al., 2024) and oversample the minority class during data collection. Specifically, for P12 and MIMIC-

13https://github.com/easonLuo2001/KEDGN 14https://github.com/easonLuo2001/Hi-Patch

Table 10: SFT Experimental setup details.

Parameter Value batch_size 64 (P12)

128 (P19/MIMIC-III) max_length 12,288 (P12/MIMIC-III)

11,264 (P19) num_train_epochs 3 optimizer AdamW learning rate 2 × 10−5 lr_scheduler_type cosine warmup_ratio 0.05 weight_decay 0.01 max_grad_norm 1.0 completion_only_loss true

III, we collect three rationales per candidate outcome, while for P19, which has a lower positive ratio, we collect six rationales per candidate outcome. The full set of prompts is provided in Appendix G.

SFT. Following prior work (Zhang et al., 2022; Li et al., 2023; Luo et al., 2024), we employ a custom sampler that maintains a 1:1 minority-tomajority ratio within each batch. All SFT experiments are conducted on a 1 H200 GPU. The full set of hyperparameters is listed in Table 10.

RL. For RL initialization, we select the best SFT checkpoint before 2 epochs rather than the best checkpoint over all three SFT epochs, as this empirically yields better RL performance and aligns with recent observations that high SFT scores may not predict better post-RL performance (Kang et al., 2025). Following Dr.GRPO (Liu et al., 2025c) and DAPO (Yu et al., 2026), we compute advantages without standard-deviation normalization and use a token-level policy gradient loss. For KL regularization, we adopt the k2-as-loss formulation (Liu et al., 2025b).

To address class imbalance, the advantage is scaled by a factor of 2 for minority class samples and by N2 for majority class samples, where N : 1 denotes the ratio of the majority to minority class size in the full training set. Particularly, in the case of the P19 data, the minority class sample is oversampled by a factor of three due to severe class imbalance. We also employ a custom stratified sampler to ensure that each rollout batch, rather than the PPO mini-batch, matches the class ratio of the full training set.

All RL experiments are implemented on top of the verl (Sheng et al., 2024) and performed on 2× B200 GPUs. The full set of training hyperparameters is in Table 11. We use m = 1.0 as the margin

Table 11: RL Experimental setup details.

Parameter Value max_prompt_length 10,240 (P12/MIMIC-III)

8,192 (P19) max_response_length 2,048 train_batch_size 256 ppo_mini_batch_size 32 total_training_steps 150 optimizer AdamW learning rate 1 × 10−6 lr_warmup_steps 10 lr_scheduler_type constant rollout.n 8 rollout.temperature 1.0 clip_ratio_low 0.20 clip_ratio_high 0.28 kl_loss_coef 0.001 kl_loss_type mse (k2)

in reward and λ = 0.25 as the weight of the GRPO objective and the CE loss. Note that the batch size is set to 256, which is larger than typical configurations, in order to ensure that our batch-level reward operates stably.

Inference. We use a fixed sampling temperature of 0.7 during inference. Because the model is trained on both rationale orderings in Equation 2 to improve generalization, we likewise perform inference under each. For each pass, we apply a softmax to the logits of y+ and y− at the yˆ position and take the resulting probability of y+ as the risk score. The patient’s final risk score is defined as the mean of these two probabilities. A comparison between this bidirectional averaging and using each ordering individually is provided in Appendix E.3.

###### D Leave-Variable-Out Full Results

We report the full numerical results for the leavevariable-out experiments. Table 12 and Table 13 list the AUPRC and AUROC for every method at masking ratios of 10%,20%,30%,40%, and 50% on P12 and MIMIC-III, respectively.

###### E Additional Experiments and Analyses

###### E.1 Low-Resource Training Experiments

Clinical applications frequently suffer from limited labeled data, particularly for rare conditions and newly emerging diseases. These regimes favor models that can effectively leverage pre-trained medical and reasoning priors. Unlike ISMTS baselines trained from scratch, TRIAGE builds on a pre-trained LLM and inherits such priors directly,

without requiring extensive task-specific supervision.

Setup. To probe this advantage under data scarcity, we fine-tune TRIAGE with rsLoRA (Kalajdzievski, 2023), which preserves pre-trained representations more reliably than full fine-tuning in low-data regimes. We use rank r = 64, scaling factor α = 64, and dropout 0.05, applying adapters to all attention and MLP projection layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). We omit the RL stage to isolate the data efficiency of supervised reasoning alone. We subsample only the training set of P12 at fractions of 1%, 5%, and 10%, keeping the validation and test sets intact for consistent evaluation.

Baseline tuning. For fair comparison, we start from the hyperparameters of the full-data setting and additionally sweep smaller batch sizes and a broader range of learning rates. This ensures that observed differences reflect data efficiency rather than under-tuned baselines.

Results. Figure 5 reports the results. TRIAGE’s improvement over baselines is most pronounced in the lowest-data regime. At P12-1%, our method outperforms the strongest baseline GRU-D by 4.4% AUROC and 11.1% AUPRC. The margin diminishes as labeled data becomes more abundant, and the two methods perform comparably at P12-10%. This pattern is consistent with the hypothesis that pre-trained knowledge contributes most when supervision is scarce. Full per-fraction results are provided in Table 15.

###### P12

###### P12

85

55

GRU-D STraTS

GRU-D STraTS

KEDGN TRIAGE

KEDGN TRIAGE

50

80

45

AUROC(%)

AUPRC(%)

75

40

35

70

30

65

25

1% 5% 10%

1% 5% 10%

Data Size

Data Size

Figure 5: Performance under low-resource training. We subsample P12 at 1%, 5%, and 10% training fractions and compare TRIAGE against ISMTS baselines using AUROC and AUPRC.

###### E.2 Ablation Study on the Backbone

To verify robustness to backbone choice, we apply TRIAGE to Qwen3-1.7B and 8B (Yang et al., 2025) in addition to our 4B default, and further to Llama 3.2 3B (Grattafiori et al., 2024) across architecture families. As reported in Table 14, TRIAGE

retains a consistent advantage over the corresponding baselines on every backbone, suggesting that our reasoning supervision generalizes beyond a single model.

###### E.3 Inference Direction Analysis

Since TRIAGE is trained on both rationale orderings, we examine whether averaging the risk scores from the two directions at inference offers any benefit over using a single ordering. Table 16 reports the results on P12 with Qwen3-4B-Base under the SFT setting. The three strategies perform comparably, indicating that a single-direction pass alone already yields a reliable risk estimate and can serve as a lower-cost alternative. Nonetheless, bidirectional averaging consistently provides a small additional gain (e.g., AUPRC of 56.4 vs. 56.1/56.2), and we adopt it as the default inference strategy.

###### F Qualitative Results

To qualitatively assess the clinical reasoning of our model beyond the quantitative evaluation based on IDEA (Baker et al., 2015; Schaye et al., 2021), we conduct a manual analysis of the generated reasoning paths. Table 17 and Table 18 present representative case studies from the P12 dataset, demonstrating how the model integrates clinical indicators to infer patient outcomes.

In Table 17, we examine a patient classified as survival and compare our model’s reasoning with that of the STraTS baseline, whose decision basis is analyzed through post-hoc XAI attribution followed by LLM interpretation. The STraTS attributions contain several signals that are inconsistent with established medical knowledge. For example, a Glasgow Coma Scale (GCS) score of 15, which indicates fully intact consciousness (Munakomi et al., 2026), is attributed as evidence for mortality. Similarly, a bicarbonate (HCO3) level below 22mEq/L is used to support a survival prediction, despite such values generally being considered low and potentially indicative of metabolic acidosis (Castro et al., 2024). In contrast, our model’s reasoning paths exhibit no such contradictions. The two approaches also diverge in their treatment of temporal dynamics. For urine output, STraTS relies on a cross-sectional assessment of near-normal values observed at later time steps, whereas our model captures the trajectory from initially low output through progressive recovery over time, which serves as a clinically meaningful indicator of im-

proving renal perfusion (Khwaja, 2012). This temporal perspective enables more comprehensive and grounded clinical reasoning.

Furthermore, we show another patient classified as mortality in Table 18. The STraTS attributions again contradict established clinical knowledge: a white blood cell (WBC) count of 23.3, substantially exceeding the normal range and suggestive of leukocytosis (Mank et al., 2026), is attributed as evidence supporting survival. Despite Blood urea nitrogen (BUN) being a well-established prognostic marker in critically ill populations (Arihan et al., 2018), BUN levels of 98-99 are also used to support a survival prediction, despite being severely elevated (Hosten, 1990). Furthermore, the most clinically critical indicators in this case, a potassium level of 10.0mEq/L, representing severe hyperkalemia with the risk of fatal cardiac arrhythmia (Simon et al., 2017), and elevated Troponin T, a well-recognized marker of myocardial injury 15, do not appear among the major contributors in the STraTS attribution. In contrast, our model explicitly incorporates these values into its reasoning chain. In addition, when identifying potential evidence for survival in a case dominated by mortalityassociated signals, STraTS produces attributions that lack clinical plausibility, whereas our model identifies clinically reasonable counterevidence, such as temporal recovery trends in serum potassium. These observations suggest that our approach yields more balanced reasoning.

It is important to acknowledge, however, that certain limitations remain. Specifically, in the patient case of Table 17, body temperature exhibited a complex, non-monotonic trajectory characterized by an initial hypothermic risk, partial recovery, and a subsequent decline. Neither the baseline nor our model’s generated reasoning fully captures this intricate behavior. This indicates that reasoning over variables with highly fluctuating temporal trajectories remains a challenge.

Furthermore, to assess hallucinations in our LLM’s outputs, we employ an LLM-as-a-judge to detect severe hallucinations in our reasoning traces, such as references to features or values that were not present in the patient record (see Figure 24 for the full prompt). Out of 200 samples from the P12 dataset, we identified severe hallucinations in only 3 cases (1.5%). Representative examples are

15https://www.cpllabs.com/clinicians/ client-communications/troponin-t-gen5/

- Table 12: Performance comparison on P12 across different variable removal ratios (10%–50%). The best results are indicated in bold and the second-best are underlined.

Method

P12-10 P12-20 P12-30 P12-40 P12-50 AUROC AUPRC AUROC AUPRC AUROC AUPRC AUROC AUPRC AUROC AUPRC ISMTS Methods

GRU-D 85.9±1.2 54.3±2.1 85.3±1.5 52.6±2.9 83.7±1.6 49.9±2.5 83.0±1.9 47.7±4.2 79.9±1.8 43.2±2.5 mTAND 84.6±0.6 53.2±1.4 83.1±0.7 50.0±2.8 80.5±1.1 45.5±4.1 79.2±1.6 43.4±4.0 75.0±0.9 39.0±2.5 SeFT 83.0±1.1 47.9±2.8 83.8±1.2 49.5±1.2 81.4±1.2 45.4±2.8 79.9±2.1 41.1±6.0 77.6±2.6 37.6±3.7 Raindrop 80.8±1.8 42.3±1.5 80.8±1.8 42.4±1.1 79.7±1.8 42.1±2.2 77.9±1.8 38.0±3.6 75.9±2.6 36.2±3.4 STraTS 86.2±1.3 54.3±3.3 85.9±1.2 54.6±3.2 84.2±1.7 50.9±3.9 84.0±1.4 48.8±5.4 80.4±1.6 42.8±4.4 ViTST 83.3±0.9 45.4±2.8 82.3±0.6 43.8±3.8 80.0±1.7 41.3±4.4 78.4±1.2 37.5±3.7 77.8±1.8 36.1±2.5 KEDGN 85.5±0.6 53.8±2.5 83.8±1.0 49.6±2.0 81.1±1.1 44.8±4.1 78.8±1.4 40.8±3.7 77.4±2.2 38.9±5.1 Hi-Patch 85.6±0.9 54.7±2.6 84.6±1.1 52.9±3.9 82.6±1.5 48.7±3.7 80.4±1.8 45.3±6.8 78.7±1.4 40.1±3.2

Ours

TRIAGESFT 85.5±0.8 54.2±2.4 84.8±1.1 52.3±2.0 83.4±1.9 49.8±3.5 82.0±1.4 46.8±4.9 79.5±1.6 44.4±2.2 TRIAGESFT+RL 84.8±1.3 54.4±2.2 85.1±1.5 54.2±3.1 83.5±2.5 51.2±3.2 82.1±1.3 48.2±5.2 79.4±2.0 44.6±2.7

- Table 13: Performance comparison on MIMIC-III across different variable removal ratios (10%–50%). The best results are indicated in bold and the second-best are underlined.

MIMIC-III-10 MIMIC-III-20 MIMIC-III-30 MIMIC-III-40 MIMIC-III-50 AUROC AUPRC AUROC AUPRC AUROC AUPRC AUROC AUPRC AUROC AUPRC ISMTS Methods

Method

GRU-D 82.8±0.7 44.4±1.9 83.1±0.3 45.1±0.8 81.3±0.3 41.1±0.6 81.1±0.2 41.2±0.7 78.7±0.3 38.1±0.4 mTAND 82.5±0.6 44.0±0.6 82.4±0.3 43.8±1.1 81.3±0.3 39.7±0.8 79.5±0.8 39.0±0.7 77.0±0.7 35.2±1.1 SeFT 80.9±0.6 37.7±1.7 80.2±0.9 38.1±2.4 79.3±0.8 35.1±1.1 78.3±0.3 34.8±1.2 76.7±0.7 33.3±1.7 Raindrop 78.9±1.0 34.4±1.6 78.3±0.8 34.7±0.7 76.2±1.0 31.8±0.7 74.9±0.8 30.6±1.2 72.9±1.0 28.9±0.5 STraTS 84.2±0.5 46.0±0.6 83.7±0.5 46.4±3.2 82.2±0.4 42.1±1.0 82.6±0.5 42.4±1.8 79.7±0.6 38.6±1.7 ViTST 81.6±0.5 39.2±1.6 80.6±0.2 36.5±1.2 79.4±0.4 35.1±1.3 79.1±0.5 34.4±1.1 76.5±0.6 33.1±0.8 KEDGN 82.4±0.4 44.1±1.0 81.8±0.5 44.1±1.3 80.1±0.8 39.1±2.1 79.3±0.6 38.7±1.2 77.0±1.2 35.5±1.4 Hi-Patch 81.7±0.7 40.4±1.9 80.5±0.5 38.5±1.4 77.3±0.9 34.1±2.7 76.2±1.0 33.8±0.8 72.7±1.7 30.0±1.8

Ours

TRIAGESFT 84.9±0.1 50.1±0.5 85.2±0.3 50.2±0.6 83.0±0.2 44.8±0.6 82.0±0.2 45.3±0.3 80.7±0.4 43.2±0.5 TRIAGESFT+RL 85.1±0.2 51.5±0.4 84.5±0.4 50.8±0.9 83.1±0.6 45.8±0.7 82.1±0.4 46.4±0.5 80.9±0.5 44.7±0.9

Table 14: Ablation on the backbone, on P12.

Backbone AUROC AUPRC Qwen3-4B (default) 86.9±1.0 56.4±1.9 Scale (Qwen3 family)

Qwen3-1.7B 86.4±1.0 53.8±1.6 Qwen3-8B 86.8±0.8 56.0±2.1

###### Architecture

Llama3.2 (3B) 87.0±1.0 54.5±1.3

provided in Table 19. This indicates that during dialectical reasoning, the model rarely fabricates evidence for the opposing side when the signal from one side is already clear-cut. We attribute this favorable outcome to the safeguards built into our data construction process, in particular, the explicit instruction to leave a field blank whenever no supporting evidence is available.

###### G Prompts

We provide a comprehensive list of all the prompts utilized in our study, organized by their role in the pipeline.

Preliminary study. To motivate our method, we first probe zero-shot LLM behavior on ICU mortality prediction. The default direct-answer prompt is shown in Figure 8, its verbalized-probability variant in Figure 9, and the final evaluation prompt for zero-shot LLMs in Figure 10. The LLM-as-judge prompt used to detect whether a chain-of-thought ends with an explicit verdict-closure sentence is shown in Figure 11.

Outcome-conditioned rationale generation. To construct supervision for TRIAGE, we elicit outcome-conditioned rationales from a strong LLM, instantiating one prompt per candidate outcome per patient. The prompt template for P12 and MIMIC-III (in-hospital mortality) is shown in

- Table 15: Low-resource training results on P12 with varying training data ratios (1%, 5%, 10%). The best results are indicated in bold and the second-best are underlined.

Method

P12-1% P12-5% P12-10% AUROC AUPRC AUROC AUPRC AUROC AUPRC ISMTS Methods

GRU-D 73.0±6.3 36.0±6.3 81.8±1.1 47.8±1.9 83.6±1.4 50.6±2.4 mTAND 69.9±6.0 32.2±6.4 79.1±1.8 43.9±1.8 82.3±2.5 47.1±4.2 SeFT 66.7±6.9 24.2±6.5 78.8±2.1 41.1±4.1 80.5±1.5 43.9±2.1 Raindrop 67.2±1.6 25.8±2.0 75.6±1.1 35.9±4.0 76.9±2.5 36.8±3.2 STraTS 69.1±5.6 27.4±6.7 81.0±1.9 44.0±4.3 81.8±1.5 47.5±2.7 ViTST 62.1±2.8 22.5±2.4 73.4±2.8 31.5±4.7 75.7±3.9 32.7±5.8 KEDGN 69.4±5.7 28.1±8.0 80.8±1.5 45.5±2.3 83.1±2.2 47.4±2.3 Hi-Patch 71.2±2.4 32.3±4.2 82.1±1.1 46.8±1.7 83.4±1.6 48.0±2.8

Ours TRIAGE 76.2±4.8 40.0±5.6 82.3±1.1 49.0±3.6 84.4±0.9 48.9±2.1

- Table 16: Ablation on the inference direction, on P12 with Qwen3-4B-Base under the SFT setting. Positivefirst and Negative-first denote inference using only the (y+,y−) or (y−,y+) ordering, respectively. Bidirectional averages the risk scores from both orderings.

pretation prompt is shown in Figure 19.

Evaluation: IDEA-based reasoning quality assessment. To assess the clinical quality of generated reasoning traces, we adapt the IDEA rubric into an LLM-as-judge protocol that scores Interpretive summary (I), Differential diagnosis (D), Explanation of the lead diagnosis (E), and Alternative diagnosis explained (A). The full evaluation prompt, including task setup, factuality rules, per-domain scoring anchors, overall calibration guidance, and the required JSON output schema, is shown across Figure 20, 21, 22, and 23.

Inference strategy AUROC AUPRC Positive-first only 86.8±1.0 56.1±1.0 Negative-first only 86.8±1.0 56.2±2.2 Bidirectional (ours) 86.9±1.0 56.4±1.9

Figure 12, and the corresponding template for P19 (early sepsis prediction) is shown in Figure 13.

Main SFT prompts for TRIAGE. TRIAGE is supervised to produce two outcome-conditioned rationale blocks followed by a single final decision. To control for ordering effects, we use two symmetric variants per dataset. For P12 and MIMIC-III, the negative-class-first and positive-class-first variants are shown in Figure 14 and Figure 15, respectively. For P19, the corresponding variants are shown in Figure 16 and Figure 17.

Qualitative Analysis To quantify hallucination in our reasoning traces, we adopt an LLM-as-ajudge approach that evaluates whether each trace references features or values absent from the patient record. We use GPT-5.1 as a judge, and the exact prompt is in Figure 24.

Ablation: one-sided reasoning. For the reasoning-structure ablation (Table 5), we use a single-rationale SFT prompt that retains only the rationale aligned with the ground-truth label. The P12 template is shown in Figure 18.

Reasoning analysis: XAI interpretation. To compare model-internal explanations with LLMstyle rationales, we textualize attribution-based evidence from STraTS and ask an LLM to interpret it in the same two-sided rationale format. The inter-

# ? 1) Task Definition

##### Feature Information:

- • HR: Heart rate, in bpm
- • BUN: Blood urea nitrogen, in mg/dL
- • GCS: Glasgow coma score (3–15)

###### Question:

Will the patient experience in-hospital death during this ICU stay?

###### Reasoning by the following process:

- 1. If the patient indeed survives, which features might be the cause?
- 2. If the patient indeed experiences in-hospital death, which features might be the cause?
- 3. Make a final decision: ‘0’ for survival, ‘1’ for in-hospital death.

### 2) Patient Features

###### Textualized Static Features:

A patient is 54 years old, female, stayed in surgical ICU.

###### Textualized Temporal Observations:

HR: (0.2h, 73), (0.7h, 77), (1.7h, 60), …

GCS: (0.2h, 15), (3.7h, 15), …

BUN: (10.7h, 13), (33.2h, 8), …

- Figure 6: Example prompt format used by TRIAGE. Only some of the features are shown. The input is organized into task definition and textualized patient features.

Patient A patient is 85 years old, male, stayed in surgical ICU. For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours since ICU admission. ### GCS (0.2, 15.0), (1.7, 15.0), (5.7, 15.0), (9.7, 15.0), (13.7, 15.0), (17.7, 15.0), (21.7, 15.0), (25.7, 15.0), (29.7, 15.0), (33.7, 15.0), (37.7, 15.0), (41.7, 15.0), (45.7, 15.0) ### DiasABP

- (4.7, 56.0), (5.0, 58.0), (5.5, 54.0), (5.7, 53.0), (6.7, 52.0), (7.7, 51.0), (8.7, 47.0), (9.7, 49.0), (10.7, 52.0), (11.7, 46.0), (12.7, 51.0), (13.7, 47.0), (14.7, 50.0), (15.7, 48.0), (16.7, 46.0), (17.7, 67.0), (18.7, 59.0), (19.7, 59.0), (20.7, 57.0), (21.7, 58.0), (22.7, 61.0), (23.7, 54.0), (24.7, 54.0), (25.7, 61.0), (26.7, 66.0), (27.7, 57.0), (28.7, 63.0), (29.7, 64.0), (30.7, 66.0), (32.7, 71.0), (33.7, 65.0), (35.2, 69.0), (35.7, 59.0), (37.7, 62.0), (38.7, 62.0), (39.7, 65.0), (40.7, 66.0), (41.5, 65.0), (41.7, 67.0), (42.0, 66.0), (42.2, 65.0), (42.5, 71.0), (42.7, 68.0), (43.7, 55.0), (44.7, 59.0), (45.7, 54.0), (46.7, 52.0), (47.7, 62.0) ### HCO3
- (5.7, 19.0), (12.6, 21.0), (31.7, 20.0) ### Lactate (5.9, 1.4), (31.9, 1.6) ### MechVent (35.7, 1.0), (38.2, 1.0), (40.7, 1.0), (42.7, 1.0) ### Temp (0.2, 34.9), (0.7, 35.9), (3.7, 35.6), (8.7, 35.8), (13.7, 35.8), (17.7, 36.1), (21.7, 36.2), (25.7, 36.0), (30.7, 35.7), (34.7, 35.8), (37.7, 35.6), (42.7, 35.0), (45.7, 35.3) ### Urine (0.2, 100.0), (1.7, 20.0), (2.7, 15.0), (3.7, 15.0), (4.7, 5.0), (5.7, 10.0), (6.7, 20.0), (7.7, 5.0), (8.7, 15.0), (9.7, 17.0), (10.7, 30.0), (11.7, 10.0), (12.7, 10.0), (13.7, 30.0), (14.7, 25.0), (15.7, 15.0), (16.7, 30.0), (17.7, 30.0), (18.7, 20.0), (19.7, 15.0), (20.7, 20.0), (21.7, 25.0), (22.7, 20.0), (23.7, 30.0), (24.7, 15.0), (25.7, 33.0), (26.7, 30.0), (27.7, 120.0), (28.7, 120.0), (29.7, 160.0), (30.7, 80.0), (32.7, 300.0), (35.5, 100.0), (36.7, 160.0), (37.7, 400.0), (38.7, 100.0), (39.7, 220.0), (40.7, 400.0), (41.7, 100.0), (42.7, 120.0), (43.7, 160.0), (44.7, 320.0), (45.7, 300.0), (46.7, 160.0), (47.7, 100.0)

STraTS + XAI (p1 = 0.746)

## Supporting Evidence of in-hospital mortality: Age, 85.0 ICUType, surgical ICU Temp, (0.2, 34.9), (42.7, 35.0), (45.7, 35.3), (37.7, 35.6), (3.7, 35.6), (34.7, 35.8), (30.7, 35.7), (0.7, 35.9), (8.7, 35.8), (13.7, 35.8) GCS, (0.2, 15.0), (1.7, 15.0), (45.7, 15.0), (41.7, 15.0), (5.7, 15.0) ## Counter-evidence: Urine, (40.7, 400.0), (37.7, 400.0), (44.7, 320.0), (45.7, 300.0) DiasABP, (11.7, 46.0), (16.7, 46.0), (8.7, 47.0), (13.7, 47.0) Lactate, (31.82, 1.6) HCO3, (5.68, 19.0)

STraTS + XAI (LLM interpret)

## Rationale for in-hospital death

- - The model viewed the patient’s advanced age (85 years) and being in a surgical ICU as baseline factors associated with higher in-hospital mortality risk.
- - It relied on repeatedly low body temperatures, mostly between 34.9–35.9 °C from very early (0.2 h) through about 45.7 h, as evidence pointing toward higher risk.
- - Repeatedly normal Glasgow Coma Scale scores of 15 were included by the model in the mortality-supporting set. ## Rationale for survival
- - The model identified several episodes of moderate urine output (300–400 mL at multiple times between ∼37.7 and 45.7 h) as favorable, suggesting preserved kidney output.
- - Lactate was low at 1.6 mmol/L, which the model considered a positive sign.
- - Bicarbonate at 19 mmol/L was also included in the survival-supporting set.

Ours (p1 = 0.086)

## Rationale for survival

- - Consistently intact neurological status with Glasgow Coma Score 15 throughout the ICU stay, indicating preserved consciousness and cerebral function.
- - Lactate levels low (1.4–1.6 mmol/L), indicating absence of significant global tissue hypoperfusion or shock.
- - Renal function relatively preserved: creatinine mildly elevated but stable (1.6–1.7 mg/dL) and BUN moderately elevated but stable; urine output improves substantially over time to high hourly volumes, indicating recovery or maintenance of kidney perfusion and function. ## Rationale for in-hospital death
- - Advanced age (85 years), which is independently associated with higher in-hospital mortality risk during critical illness.
- - Persistently low temperature around 35–36 °C (e.g., 34.9–36.2 °C throughout the stay), suggesting hypothermia, which in elderly ICU patients can indicate severe systemic illness, impaired thermoregulation, or sepsis and is associated with worse outcomes.
- - Mildly reduced bicarbonate (HCO3 19–21 mmol/L), suggesting a mild metabolic acidosis or reduced buffering capacity, which in the context of critical illness can be associated with increased mortality risk.
- - Need for mechanical ventilation later in the ICU course (MechVent=1 at multiple time points), indicating respiratory failure or inability to maintain adequate ventilation/oxygenation without support, a major marker of clinical deterioration and higher mortality.

###### Table 17: Case study of a survived patient. For brevity, only a subset of patient characteristics and some of the clinical reasoning parts are excerpted. Text highlighted in blue and red indicates medically favorable and unfavorable signs, respectively. The LLM-as-a-judge results of STraTS+XAI and ours are 5.0 and 8.0, respectively.

Patient A patient is 88 years old, male, 52.0 kg, stayed in medical ICU. For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours since ICU admission. ### GCS (2.0, 7.0), (4.5, 3.0), (8.5, 3.0), (13.5, 5.0), (16.5, 6.0), (20.5, 6.0), (23.5, 6.0), (28.5, 6.0), (32.5, 6.0), (40.5, 9.0), (44.5, 9.0) ### BUN (2.6, 99.0), (9.6, 98.0), (32.4, 82.0), (44.6, 80.0) ### HCO3 (2.6, 21.0), (2.6, 23.0), (9.6, 20.0), (32.4, 15.0), (44.6, 17.0) ### K (2.6, 6.8), (2.6, 10.0), (9.6, 4.3), (32.4, 4.0), (44.6, 3.8) ### TroponinT (9.6, 0.15) ### WBC (2.6, 21.4), (9.6, 17.9), (9.6, 18.9), (32.4, 23.3) ### Na (2.6, 157.0), (2.6, 160.0), (9.6, 163.0), (21.0, 164.0), (28.6, 160.0), (32.4, 159.0), (44.6, 150.0)

STraTS + XAI (p1 = 0.977)

## Supporting Evidence of in-hospital mortality: Age, 88.0 Na, (32.35, 159.0), (28.55, 160.0), (20.97, 164.0) BUN, (44.58, 80.0), (32.35, 82.0) GCS, (44.43, 9.0), (32.43, 6.0), (40.43, 9.0), (28.43, 6.0), (23.43, 6.0), (20.43, 6.0), (16.43, 6.0), (13.43, 5.0), (8.43, 3.0), (1.93, 7.0), (2.52, 41.1), (9.52, 38.2), (4.43, 3.0) ICUType, medical ICU Weight, 52.0 ## Counter-evidence: WBC, (32.35, 23.3) BUN, (2.52, 99.0), (9.52, 98.0) Na, (2.52, 160.0) Gender, male

STraTS + XAI (LLM interpret)

## Rationale for in-hospital death

- - The model places strong weight on the patient’s advanced age (88 years) and admission to a medical ICU, both of which it associates with higher in-hospital mortality.
- - It relies heavily on repeatedly very low Glasgow Coma Scores (GCS 3–9 across many time points), indicating persistently poor neurological responsiveness.
- - Multiple very high sodium values (Na 159–164 mmol/L at several times) are treated as important adverse findings.
- - Kidney-related measures support the death prediction: elevated BUN values (around 80–82 mg/dL later in the stay). ## Rationale for survival
- - The model treats being male as slightly favoring survival in its learned patterns.
- - A very high white blood cell count (23.3) is used as counter-evidence to death by the model.
- - Some extremely high early BUN values (98–99 mg/dL) and an early very high sodium level (Na 160 mmol/L at 2.52 hours) are also treated as counter-evidence.

Ours (p1 = 0.844)

## Rationale for in-hospital death

- - Advanced age (88 years), which is independently associated with high ICU mortality risk.
- - Marked acute kidney injury/renal dysfunction: very high BUN (peak 99 mg/dL, remaining 80–98 mg/dL) and elevated creatinine (2.1–2.9 mg/dL), indicating severe renal impairment.
- - Severe hyperkalemia early in the stay (K up to 10.0 mmol/L; later 6.8 mmol/L), a life-threatening electrolyte disturbance predisposing to fatal arrhythmias.
- - Hypernatremia (Na 150–164 mmol/L for much of the stay), suggesting significant fluid/electrolyte derangement and osmotic stress, associated with poor outcomes in the elderly.
- - Metabolic acidosis/low bicarbonate (HCO3 as low as 15–17 mmol/L) indicating significant systemic derangement and severity of illness.
- - Depressed and fluctuating neurologic status: very low GCS values (as low as 3; later only partial improvement to 6 then 9), indicating severe brain dysfunction and poor neurologic prognosis.
- - Leukocytosis (WBC 17.9–23.3) suggesting a significant inflammatory or infectious process, which in this context is associated with high severity of illness.
- - Elevated Troponin T (0.15 ng/mL), indicating myocardial injury, which in an elderly, critically ill patient increases the risk of cardiac complications and death. ## Rationale for survival
- - Glasgow Coma Score improves over time from very low values (3–6) to 9 by around 40–44 hours, indicating neurological recovery.
- - Renal function markers (BUN and creatinine) show a downward trend over time, consistent with partial improvement in kidney injury.
- - Serum potassium rapidly normalizes from markedly abnormal early values to within acceptable limits, indicating effective correction of a life-threatening electrolyte disturbance.
- - Sodium levels, though initially markedly elevated, show a gradual decline over time toward more normal values, suggesting progressive management of hypernatremia.

- Table 18: Case study of a patient classified as mortality. For brevity, only a subset of patient characteristics and some of the clinical reasoning parts are excerpted. Text highlighted in blue and red indicates medically favorable and unfavorable signs, respectively. The LLM-as-a-judge results of STraTS+XAI and ours are 6.1 and 8.4, respectively.

- Patient 1 A patient is 34 years old, male, 177.8 cm, 83.0 kg, stayed in surgical ICU. For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours since ICU admission. No Troponin T information in this patient.

Our reasoning 2 ## Rationale for survival

- **Hemodynamic and cardiac stability:** Heart rate fluctuates but generally remains in a tolerable range, and troponin values are not elevated (no data suggesting myocardial injury), consistent with maintained cardiovascular stability.

- Patient 2 A patient is 50 years old, male, 182.9 cm, 127.0 kg, stayed in coronary care unit. For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours since ICU admission. ### GCS (2.3, 15.0), (5.3, 15.0), (8.3, 15.0), (10.3, 15.0), (16.3, 10.0), (20.3, 10.0), (23.3, 10.0), (29.3, 8.0), (32.3, 10.0), (39.3, 8.0), (47.3, 3.0)

Our reasoning 2 ## Rationale for survival

- Glasgow Coma Score was 15 early in the stay and remained ≥8 later (with intermittent decreases), indicating preserved neurological function and the ability to protect airway at least at times.

- Table 19: Case study of patients including the hallucinated clinical rationales. For brevity, only relevant parts are excerpted. In the case of the first patient, the feature (Troponin T) not present in the patient’s information is mentioned. In the case of the second patient, a value present in the patient’s information was misidentified.

A patient is 62 years old, female, 160.0cm, 52.1kg, stayed in medical ICU.

[Figure 21]

### FiO2 (0.5h, 1.0), (3.4h, 1.0), …, (40.7h, 0.4), (44.4h, 0.4)

### GCS

(0.2h, 15.0), (12.4h, 8.0), (16.4h, 8.0), (20.4h, 10.0), ...

9x Predicted Mortality 1x Predicted Survival

[Figure 22]

Example A: Response Predicting Mortality Example B: Response Predicting Survival

|The patient has been on mechanical ventilation almost continuously, required high FiO₂ (up to 100 % early) and still needs 0.4–0.5 FiO₂ at 44 h. She has persistent tachycardia, fluctuating blood pressures with several low MAP values, low albumin (2.2 g/dL), rising lactate early, and a falling platelet count (162 → 118 ×10⁹/L).<br><br>Her GCS remains low (≈10–11) and she is still<br><br>intubated at >2 days. These are markers of severe critical illness and multi-organ dysfunction, which are associated with a high risk of in-hospital death.<br><br>**Prediction: 1**|
|---|

|The patient has been intubated and on full-FiO₂ = 1.0 early on, but rapidly weaned to FiO₂ ≈ 0.4 with stable PaO₂ ≈ 60-80 mmHg and SaO₂ ≈ 92 %. Her hemodynamics show generally adequate MAP (70-110 mmHg) with occasional low diastolic pressures; heart rate is decreasing to the 60-70 bpm range. Renal function is normal (creatinine ≈ 0.6-0.8 mg/dL,<br><br>urine output mostly >0.5 mL/kg/h). There is<br><br>mild hypo-albuminemia (2.2 g/dL) and thrombocytopenia (platelets falling to ~120 ×10⁹/L) but no severe metabolic derangement: lactate remains <2 mmol/L, pH normal, electrolytes stable. Overall the trend is improvement with decreasing ventilatory support, stable oxygenation, and no evidence<br><br>of shock or multi-organ failure.<br><br>**Prediction:** survival (no in-hospital death).<br><br>**Answer:** 0|
|---|

- Figure 7: Two reasoning traces sampled from gpt-oss-120b on the same P12 patient. The mean implicit mortality probability over 10 samples for this patient is 0.9. Each trace cites only one-sided evidence and commits to the corresponding outcome.

###### Default prompt used in the preliminary study

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Read the patient’s ICU features and answer immediately. ## Task Will the patient experience in-hospital death during this ICU stay? Reply with only one number, immediately.

- - 0 for survival
- - 1 for in-hospital death Do not provide any explanation or additional text.

## Feature of the patient {PATIENT_FEATURES}

Figure 8: Default prompt used in the preliminary study for in-hospital mortality prediction.

Prompt for verbalized probability extraction

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Read the patient’s ICU features and answer immediately. ## Task Estimate the probability that this patient will experience in-hospital death during this ICU stay. Reply with only a single number between 0 and 1, immediately.

- - 0.0 means certain survival
- - 1.0 means certain death Do not provide any explanation or additional text.

## Feature of the patient {PATIENT_FEATURES}

- Figure 9: Prompt used for verbalized probability extraction. Only the ## Task section differs from Figure 8.

###### Prompt used for zero-shot LLM evaluation

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Read the patient’s ICU features and answer immediately. ## Task Estimate the probability that this patient will experience in-hospital death during this ICU stay. Analyze the patient’s clinical state, then answer. Reply with only a single number between 0 and 1 at the end.

- - 0.0 means certain survival
- - 1.0 means certain death

## Feature of the patient {PATIENT_FEATURES}

- Figure 10: Final prompt used for zero-shot LLM evaluation of in-hospital mortality. Only the ## Task section differs from Figures 8 and 9, permitting an analysis step before the numeric answer.

Prompt used for LLM-as-judge verdict-closure detection

You will read a chain-of-thought reasoning produced by another LLM on a binary in-hospital mortality task (0 = survival, 1 = in-hospital death).

Strict criterion: The VERY LAST SENTENCE of the reasoning must, by itself, explicitly state the final binary answer — either:

- (a) the digit “0” or “1” in the context of the answer, OR
- (b) the words “survival”/“survive”/“live” or “death”/“die” presented as the final decision.

If the last sentence does anything else — discusses evidence, raises a caveat, gives a meta-instruction (e.g. “Thus output single number”), asks a question, or trails off — output 0.

Reply with exactly one digit:

- - 1 = the LAST sentence is itself an explicit confirmatory restatement of the final answer
- - 0 = it is not

Reasoning to evaluate: """ {reasoning} """

Figure 11: Prompt used for LLM-as-judge verdict-closure detection.

###### Prompt for outcome-conditioned rationale elicitation (P12, MIMIC-III)

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Based on the given features of a patient, provide possible clinical rationale for why the patient could {outcome_phrase} during this ICU stay. Assume the outcome is {outcome_phrase} and list only supporting features.

- - Use only the provided features; do not invent data.
- - Do not describe normal or healthy findings as abnormal.
- - Do not describe abnormal findings as normal or healthy.
- - Do not mention any label, class, or numeric outcome.
- - Do not discuss the opposite outcome.
- - If there is no supporting evidence, leave the rationale blank.

Your answer format must be: ``` ## Rationale for {outcome_phrase} [blank or concise bullet points] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 12: Prompt used to elicit outcome-conditioned rationales from a strong LLM on P12 and MIMIC-III. The same prompt structure is used for both datasets; only the feature-list block is replaced with the dataset-specific list. The placeholder {outcome_phrase} is instantiated separately as in-hospital death or survival, yielding one outcome-specific prompt per outcome per patient.

Prompt for outcome-conditioned rationale elicitation (P19)

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Based on the given features of a patient, provide possible clinical rationale for why the patient could experience {assumption}. Assume the outcome is ‘{assumption}’ and list only supporting features.

- - Use only the provided features; do not invent data.
- - Do not describe normal or healthy findings as abnormal.
- - Do not describe abnormal findings as normal or healthy.
- - Do not mention any label, class, or numeric outcome.
- - Do not discuss the opposite outcome.
- - If there is no supporting evidence, leave the rationale blank.

Your answer format must be: ``` ## {header} [blank or concise bullet points] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 13: Prompt used to elicit outcome-conditioned rationales from a strong LLM on P19 (early sepsis prediction). For each candidate outcome, the placeholders {assumption} and {header} are paired and substituted as either (sepsis onset within the next 6 hours, Rationale for sepsis) or (no sepsis onset within the next 6 hours, Rationale for no sepsis).

###### Main SFT prompt for TRIAGE (P12, MIMIC-III) — negative-class-first variant

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Based on the given feature of a patient, answer the question below. ## Question Will the patient experience in-hospital death during this ICU stay? Reasoning by the following process:

- 1. If the patient indeed survives, which of the patient’s given features might be the cause?
- 2. If the patient indeed experiences in-hospital death, which of the patient’s given features might be the cause?
- 3. Make a final decision: ‘0’ for survival, ‘1’ for in-hospital death.

Your answer format must be as follows: ``` ## Rationale for survival [possible justification if patient survives]

## Rationale for in-hospital death [possible justification if patient experiences in-hospital death]

## Final Decision [0 (survival) or 1 (in-hospital death); respond by single number only] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 14: Main SFT prompt used for TRIAGE on P12 and MIMIC-III (negative-class-first variant). The model is supervised to produce two outcome-conditioned rationale blocks followed by a single ## Final Decision, with the rationale for the negative class (survival, label 0) preceding the rationale for the positive class (in-hospital death, label 1).

###### Main SFT prompt for TRIAGE (P12, MIMIC-III) — positive-class-first variant

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO} Based on the given feature of a patient, answer the question below.

## Question Will the patient experience in-hospital death during this ICU stay?

Reasoning by the following process:

- 1. If the patient indeed experiences in-hospital death, which of the patient’s given features might be the cause?
- 2. If the patient indeed survives, which of the patient’s given features might be the cause?
- 3. Make a final decision: ‘0’ for survival, ‘1’ for in-hospital death. Your answer format must be as follows: ``` ## Rationale for in-hospital death [possible justification if patient experiences in-hospital death]

## Rationale for survival [possible justification if patient survives]

## Final Decision [0 (survival) or 1 (in-hospital death); respond by single number only] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 15: Main SFT prompt used for TRIAGE on P12 and MIMIC-III (positive-class-first variant): the rationale for the positive class (in-hospital death, label 1) precedes the rationale for the negative class (survival, label 0). Compared with the negative-class-first variant (Figure 14), only the ordering of the reasoning steps and rationale blocks differs; all other text is identical.

###### Main SFT prompt for TRIAGE (P19) — negative-class-first variant

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Based on the given feature of a patient, answer the question below. ## Question Will the patient experience sepsis onset within the next 6 hours? Reasoning by the following process:

- 1. If the patient indeed does not experience sepsis onset within the next 6 hours, which of the patient’s given features might be the cause?
- 2. If the patient indeed experiences sepsis onset within the next 6 hours, which of the patient’s given features might be the cause?
- 3. Make a final decision: ‘0’ for no sepsis onset within the next 6 hours, ‘1’ for sepsis onset within the next 6 hours.

Your answer format must be as follows: ``` ## Rationale for no sepsis [possible justification if patient does not experience sepsis onset within the next 6 hours]

## Rationale for sepsis [possible justification if patient experiences sepsis onset within the next 6 hours]

## Final Decision [0 (for no sepsis onset within the next 6 hours) or 1 (for sepsis onset within the next 6 hours); respond by single number only] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 16: Main SFT prompt used for TRIAGE on P19 (negative-class-first variant). The model is supervised to produce two outcome-conditioned rationale blocks followed by a single ## Final Decision, with the rationale for the negative class (no sepsis, label 0) preceding the rationale for the positive class (sepsis, label 1).

###### Main SFT prompt for TRIAGE (P19) — positive-class-first variant

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Based on the given feature of a patient, answer the question below. ## Question Will the patient experience sepsis onset within the next 6 hours? Reasoning by the following process:

- 1. If the patient indeed experiences sepsis onset within the next 6 hours, which of the patient’s given features might be the cause?
- 2. If the patient indeed does not experience sepsis onset within the next 6 hours, which of the patient’s given features might be the cause?
- 3. Make a final decision: ‘0’ for no sepsis onset within the next 6 hours, ‘1’ for sepsis onset within the next 6 hours.

Your answer format must be as follows: ``` ## Rationale for sepsis [possible justification if patient experiences sepsis onset within the next 6 hours]

## Rationale for no sepsis [possible justification if patient does not experience sepsis onset within the next 6 hours]

## Final Decision [0 (for no sepsis onset within the next 6 hours) or 1 (for sepsis onset within the next 6 hours); respond by single number only] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 17: Main SFT prompt used for TRIAGE on P19 (positive-class-first variant): the rationale for the positive class (sepsis, label 1) precedes the rationale for the negative class (no sepsis, label 0). Compared with the negative-class-first variant (Figure 16), only the ordering of the reasoning steps and rationale blocks differs; all other text is identical.

###### Prompt for One-sided rationale SFT (P12)

I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, each characterized by number of features. The list of features are as follows: {FEATURE_INFO}

Based on the given feature of a patient, answer the question below. ## Question Will the patient experience in-hospital death during this ICU stay? Reasoning by the following process:

1. Describe the clinical evidence observed in the patient’s features. 2. Make a final decision: ‘0’ for survival, ‘1’ for in-hospital death.

Your answer format must be as follows: ``` ## Rationale [clinical evidence observed in the patient’s features]

## Final Decision [0 (survival) or 1 (in-hospital death); respond by single number only] ```

## Feature of the patient {PATIENT_FEATURES}

- Figure 18: Prompt template used for the One-sided rationale SFT baseline in our reasoning-structure ablation (Table 5), conducted on P12. The supervision retains only the rationale steps that support the ground-truth label from our collected outcome-conditioned traces, so a single ## Rationale block is followed by the corresponding ## Final Decision.

###### Prompt used for interpretation of XAI result of STraTS

You are an interpreter for explanations of a trained ICU mortality prediction model. You will be given a textualized summary of the model’s prediction evidence. This summary was generated from important patient-specific observations selected from the model. Your task is to interpret what the provided evidence means about the model’s prediction.

# Input ## List of features {FEATURE_INFO} ## Evidence Summary {XAI_RESULT_TEXTUALIZE} # Your Task Explain what the provided evidence summary means about the model’s prediction. Interpretation means:

- - identify what kinds of observations the model relied on,
- - distinguish evidence for the predicted class from evidence for the opposite class,
- - group related observations when appropriate,
- - explain the model’s prediction in plain language. Interpretation does NOT mean:
- - making a new clinical judgment,
- - deciding whether the model is medically correct,
- - inferring missing diagnoses, treatments, or outcomes,
- - explaining the XAI method,
- - discussing attribution scores or importance scores. # Rules
- - Use only the information in the textualized evidence summary.
- - Do not introduce new clinical interpretations that are not directly supported by the summary.
- - Do not overstate the evidence.
- - Do not evaluate the adequacy of the evidence.
- - Do not add unnecessary concluding statements.
- - Keep the interpretation concise. # Output Format

Only write the interpretation of the evidence predicted by the model. Write a concise interpretation (≤ 10 sentences for each section) along with specific figures in the following format: ## Rationale for survival

- - Interpretation of evidences ## Rationale for in-hospital death
- - Interpretation of evidences

Figure 19: Prompt for the interpretation of the IG result of STraTS.

You are an expert in clinical reasoning assessment and ICU prognostic reasoning. Your task is to evaluate one clinical reasoning trace and score how medically reasonable it is for predicting in-hospital mortality during the ICU stay.

# Input Data [Clinical Question] """Will the patient experience in-hospital death during this ICU stay?""" [Patient Features] """[[PATIENT_FEATURES]]""" [Reasoning] """[[MODEL_REASONING]]""" # Task Instructions Evaluate whether the reasoning trace is clinically reasonable, faithful to the provided patient features, temporally aware, balanced, and appropriately calibrated for ICU in-hospital mortality prediction. Your job is to judge the quality of the medical reasoning based only on the provided patient features. Your job is NOT to determine the true outcome. Your job is NOT to infer or reward matching any hidden label. Your job is NOT to reward longer, more fluent, more confident, or more detailed writing unless the added detail is accurate, patient-specific, clinically relevant, and proportional to the evidence. Score the reasoning using the adapted Revised-IDEA rubric below. # Mandatory Factuality and Data-Use Rules Use only the provided patient features. You may use general medical knowledge to interpret the provided features, but you must not add clinical facts that are not present. Do NOT assume unstated diagnoses, medications, procedures, organ support, code status, goals of care, complications, outcomes, or information after the prediction window. Do NOT treat missing data as normal or abnormal without justification. Do NOT infer the true label. Do NOT reward the reasoning for matching any hidden label. Do NOT reward a model simply because it gives a more extreme mortality probability. Do NOT reward generic medical knowledge unless it is accurately connected to this patient. Penalize reasoning that:

- - Hallucinates unsupported major diagnoses, organ failures, treatments, procedures, or events.
- - Contradicts the provided patient data.
- - Misreads or reverses the temporal trajectory.
- - Treats time-series measurements as static when the trend is prognostically important.
- - Overemphasizes minor or isolated abnormalities while ignoring the main prognostic problem.
- - Gives very high or very low mortality risk without evidence proportional to that confidence.
- - Cherry-picks only risk-increasing or only risk-decreasing evidence when the data are mixed. # Adapted Revised-IDEA Rubric for ICU Mortality Reasoning Score the reasoning out of 10 points using four domains:

I. Interpretive Prognostic Summary: 0-4 points

- D. Differential Prognostic Assessment: 0-2 points
- E. Explanation of Lead Prognostic Judgment: 0-2 points A. Alternative Prognostic Explanation / Counterevidence: 0-2 points

The total score is I + D + E + A, from 0 to 10. The scoring should emphasize clinical reasoning quality, factuality, temporal awareness, and calibration. Do not mechanically reward length or the number of facts listed.

- Figure 20: Prompt used for the quantitative evaluation of generated reasoning trace (1/4). We adopt IDEA assessment tool as the evaluation criteria.

## I — Interpretive Prognostic Summary, 0-4 points This assesses whether the reasoning gives an accurate, patient-specific, temporally aware summary of the patient’s main prognostic problem. Prefer reasoning that accurately identifies, when supported by the patient features:

- 1. Key baseline or contextual mortality-risk factors:

- Examples: age, major comorbidities, chronic organ disease, frailty-relevant context, admission context, or other baseline risks.

- 2. Main acute ICU problem or organ dysfunction:

- Examples: respiratory failure, shock or hemodynamic instability, renal dysfunction, metabolic acidosis, neurologic depression, hepatic or coagulation dysfunction, infection signal, multi-organ dysfunction, or other acute problems explicitly supported by the data.

- 3. Illness time course or trajectory:

- - Correctly distinguishes worsening, improving, fluctuating, or stable course.
- - Gives special attention to persistent instability or late deterioration near the end of the observation window.

- 4. Clinically meaningful prognostic abstractions:

- - Examples: persistent hypotension, escalating oxygen needs, worsening renal function, severe acidosis, multiorgan dysfunction, improving hemodynamics, stable oxygenation, transient isolated abnormality.
- - These abstractions must be grounded in the provided data. Scoring anchors:
- - 4: Accurate, patient-specific, and temporally aware summary that captures the main acute problem, key relevant baseline/contextual risks if provided, trajectory, and meaningful prognostic abstractions.
- - 3: Mostly accurate summary with the main prognostic problem identified, but with a minor omission, limited temporal interpretation, or limited abstraction.
- - 2: Partially accurate summary; mentions some relevant patient-specific factors but misses an important main problem, trajectory, or severity distinction, or includes minor unsupported statements.
- - 1: Minimal or generic summary; only one relevant patient-specific element is accurately identified, or the reasoning mostly lists data without integrating the main prognostic problem.
- - 0: No meaningful prognostic summary, or the summary is mostly contradicted by the patient features or dominated by hallucinated information. Do not penalize the reasoning for not mentioning information that is not provided. Do penalize it for inventing missing baseline risks, diagnoses, treatments, or organ failures. ## D — Differential Prognostic Assessment, 0-2 points

This assesses whether the reasoning considers competing prognostic interpretations and explicitly prioritizes the most plausible mortality-risk interpretation. For this ICU mortality task, “differential” means competing mortality-risk hypotheses, not a diagnostic differential diagnosis.

Relevant competing interpretations include:

- - High vs moderate vs low mortality risk.
- - Mortality risk driven by multi-organ dysfunction vs lower risk because abnormalities are mild or transient.
- - High-risk deteriorating trajectory vs stabilization or survival trajectory.
- - Risk-increasing signals vs risk-decreasing signals.
- - Uncertainty due to mixed or incomplete evidence. Scoring anchors:
- - 2: Clearly states a lead mortality-risk interpretation and explicitly discusses at least one plausible alternative or countervailing interpretation, then prioritizes the lead interpretation using patient-specific data.
- - 1: Gives a lead mortality-risk interpretation but alternatives are implicit, generic, weakly discussed, or not clearly prioritized; or lists competing ideas without clearly weighing them.
- - 0: No meaningful prognostic assessment; only a bare conclusion, generic statement, or diagnostic laundry list that does not support mortality-risk reasoning. If the evidence is strongly one-directional, the reasoning can still receive 2 points if it explicitly explains why alternative interpretations are less supported and acknowledges remaining uncertainty appropriately.

- Figure 21: Prompt used for the quantitative evaluation of generated reasoning trace (2/4). We adopt IDEA assessment tool as the evaluation criteria.

## E — Explanation of Lead Prognostic Judgment, 0-2 points This assesses whether the reasoning explains why its lead mortality-risk judgment follows from the objective patient data. Prefer reasoning that:

- - Clearly links objective patient-specific evidence to the lead mortality-risk judgment.
- - Uses clinically relevant data such as vital signs, laboratory values, organ-support indicators, severity scores, comorbidities, or temporal trends when provided.
- - Interprets the severity and persistence of abnormalities appropriately.
- - Gives a final risk interpretation that is proportional to the evidence. Calibration requirements:
- - Very high mortality risk requires strong evidence, such as severe or persistent instability, major organ dysfunction, multiple converging high-risk features, or marked late deterioration.
- - Very low mortality risk should not ignore major organ dysfunction, severe derangements, or sustained deterioration.
- - Mild, isolated, or transient abnormalities should not be overstated as extreme risk.
- - If evidence is mixed, the reasoning should express appropriate uncertainty. Scoring anchors:
- - 2: Uses at least two objective patient-specific data points or temporal trends to support the lead judgment, interprets them clinically and accurately, and gives a calibrated final mortality-risk interpretation.
- - 1: Provides some patient-specific support, but the explanation is incomplete, weakly linked, based on only one main data point, partially miscalibrated, or contains minor unsupported claims.
- - 0: No explanation of the lead mortality-risk judgment, or the explanation is unsupported, contradicted by the data, dominated by hallucinated facts, or purely generic. Do not give full credit for simply naming abnormal values without explaining their prognostic relevance. ## A — Alternative Prognostic Explanation / Counterevidence, 0-2 points

This assesses whether the reasoning identifies and explains patient-specific evidence that could support a different, lower-risk, higher-risk, or more uncertain interpretation. For this ICU mortality task, “alternative diagnosis explained” is adapted to mean alternative prognostic explanation or counterevidence.

Relevant counterevidence may include:

- - Reassuring or risk-decreasing factors.
- - Stable or improving organ function.
- - Transient rather than sustained abnormalities.
- - Lack of support for severe conditions that the reasoning might otherwise imply.
- - Risk-increasing evidence that contradicts an overly reassuring conclusion.
- - Mixed evidence requiring uncertainty.
- - Absence of provided evidence for major interventions or complications, without treating missing data as definitely normal. Scoring anchors:
- - 2: Identifies and explains at least two patient-specific countervailing data points, alternative prognostic explanations, or sources of uncertainty, and weighs them fairly against the lead judgment.
- - 1: Mentions one patient-specific countervailing data point or gives a generic/limited discussion of uncertainty or alternative risk interpretation.
- - 0: No meaningful counterevidence or alternative prognostic explanation; cherry-picks only evidence supporting the conclusion; or treats missing information as definitely normal or definitely abnormal. Counterevidence can point in either direction:
- - If the lead judgment is high mortality risk, counterevidence may include stabilization, improvement, preserved organ function, or transient abnormalities.
- - If the lead judgment is low mortality risk, counterevidence may include persistent instability, organ dysfunction, severe derangements, or late deterioration.

- Figure 22: Prompt used for the quantitative evaluation of generated reasoning trace (3/4). We adopt IDEA assessment tool as the evaluation criteria.

Evaluation: Prompt for IDEA assessment tool (4/4) # Overall Score Calibration Use the following general score interpretation:

- - 9-10: Excellent reasoning. Accurate, patient-specific, temporally aware, balanced, well-explained, and wellcalibrated. No important hallucinations or major omissions.
- - 7-8: Good reasoning. Mostly accurate and clinically grounded, with minor omissions, limited counterevidence, or mild calibration issues.
- - 5-6: Mixed reasoning. Some accurate patient-specific evidence, but important omissions, weak temporal reasoning, one-sided assessment, or moderate calibration problems.
- - 3-4: Weak reasoning. Generic, poorly grounded, several unsupported claims, major omissions, or significant misreading of the clinical picture.
- - 0-2: Poor reasoning. Little to no patient-specific reasoning, severe hallucinations, major contradictions, hiddenlabel reasoning, or conclusion unsupported by the provided data. # Internal Deliberation Before producing the final judgment, internally check:

- 1. What objective patient-specific evidence the reasoning used correctly.
- 2. Any unsupported claims, hallucinations, contradictions, or missing-data assumptions.
- 3. Whether the reasoning correctly handled the temporal trajectory.
- 4. Whether the reasoning considered counterevidence or uncertainty. Do not output a long chain-of-thought. Output only the JSON object below. # Output Format

Return valid JSON only. Do not include markdown, comments, or extra text. The score and all subscores must be integers. The final score must equal the sum of the four subscores after applying any cap or penalty. The confidence field reflects your confidence in the reasoning-quality score, not confidence in the patient’s mortality outcome. The confidence field must be exactly one of: "High", "Medium", or "Low". Use empty arrays if there are no notable items.

Return this JSON object: {

"score": 0 - 10, "confidence": "High" | "Medium" | "Low", "subscores": {

"I_interpretive_prognostic_summary": 0,

- "D_differential_prognostic_assessment": 0,
- "E_explanation_of_lead_prognostic_judgment": 0, "A_alternative_or_counterevidence_explained": 0

}, "rationale": "Briefly explain the score. Mention the most important patient-specific evidence, temporal reasoning,

calibration, counterevidence, and any major errors or cap applied.", "key_strengths": [], "key_errors": []

}

- Figure 23: Prompt used for the quantitative evaluation of generated reasoning trace (4/4). We adopt IDEA assessment tool as the evaluation criteria.

###### Prompt used to detect severe hallucination in reasoning traces

You are a clinical evidence-grounding judge for ICU mortality/survival rationales. You will be given:

- 1. Observed patient information from the first 48 hours of ICU stay.
- 2. A rationale explaining why the patient is predicted to die or survive. Your task is to judge whether the rationale is grounded in the observed patient information, or whether it contains serious hallucination: concrete clinical facts that are completely absent from the patient data. Return True if the rationale’s substantive clinical evidence is present in, paraphrased from, summarized from, or reasonably inferred from the patient information. Return False if the rationale uses at least one clinically meaningful fabricated fact as evidence, such as a diagnosis, treatment, event, demographic detail, lab/vital value, trend, medication, procedure, or condition that is not present in the patient information or is clearly contradicted by it. Be lenient. The goal is to detect serious hallucination, not minor imprecision. Do not mark False for small wording issues, approximate values, rounding, broad ranges, mild exaggeration, or imperfect temporal alignment. If a value or feature appears anywhere in the patient data and the rationale refers to it in a broadly consistent way, consider it grounded. For example, if the patient data contains PaO2 values of 98, 100, and 125 mmHg, then a rationale saying “PaO2 around 98–125 mmHg indicating adequate oxygenation” should be considered grounded, even if the exact timing is not perfectly aligned with another feature such as mechanical ventilation. Do not require every cited feature to co-occur at the exact same timestamp unless the rationale makes a very specific timing claim that is central and clearly contradicted. Accept clinical interpretations such as “stable,” “adequate oxygenation,” “renal dysfunction,” “shock,” or “high mortality risk” if they are reasonably supported by the observed measurements. Do not evaluate whether the mortality/survival prediction is medically correct; only judge whether the rationale is based on information actually observed in the input. Mark False only when there is a clear, material unsupported claim, for example:

- - The rationale cites lactate, creatinine, vasopressors, mechanical ventilation, sepsis, pneumonia, cancer, renal failure, surgery, or another condition/treatment/event that is not present anywhere in the patient data.
- - The rationale gives a concrete numeric value or range that is grossly inconsistent with the observed values.
- - The rationale describes a trend that is clearly opposite to the observed trend.
- - The rationale relies on medical history or outcome information not included in the patient data. If the unsupported statement is only a minor non-evidentiary phrase and the main clinical evidence is grounded, return True. When uncertain whether something is a minor overstatement or a serious hallucination, prefer True.

Input: [Patient Information] """{patient_information}""" [Rationale] """{rationale}""" Output only a Python-style dict with exactly these two keys:

{

"reason": "<brief explanation in 10 sentences or fewer>", "result": True or False

}

- Figure 24: Prompt used to detect severe hallucination in reasoning traces. The judge model decides whether the rationale’s substantive clinical evidence is grounded in the input. Only clearly fabricated facts such as unobserved diagnoses, treatments, lab values, or trends contradicting the data are marked as hallucinations.

