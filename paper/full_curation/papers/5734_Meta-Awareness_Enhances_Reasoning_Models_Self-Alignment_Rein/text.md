## Verifying Meta-Awareness via Predictive Rewards in Reasoning Models

# arXiv:2510.03259v2[cs.LG]31May2026

Yoonjeon Kim*1 Doohyuk Jang*1 Eunho Yang12

### Abstract

Recent research on reasoning models explores the meta-awareness of language models, including their ability to determine optimal thinking duration, recognize knowledge boundaries, and structure concept-level thinking. While current large reasoning models depend solely on answer-based verification, we show that adding meta-awareness objectives leads to significant performance gains over models without such meta-knowledge. MAPR (Meta-Awareness via Predictive Reward) utilizes a self-generated task of predicting rollout statistics - specifically length, pass-rate, and concepts used - allowing for verification against the actual statistics. Furthermore, by leveraging this self-predictive capability, the model can regulate its reasoning behavior by i) filtering out trivial or unsolvable prompts, ii) reducing lengthy generations that tend to be incorrect, and iii) generating hints relevant to the problem. The results are inspiring: MAPR yields significant improvements in both accuracy and training efficiency on various reasoning benchmarks. More specifically, our method can speed up GRPO training by over 1.28× to reach the same performance, and achieve 83.18% gain in accuracy on AIME25, and a 13.04% average gain over six mathematics benchmarks. The code is publicly available at https://github.com /akatigre/MAPR-RL.

### 1. Introduction

Recent studies have confirmed that applying RL-based posttraining to large language models (LLMs) (Brown et al., 2020; Yang et al., 2025a; Touvron et al., 2023) can significantly enhance their reasoning ability. In particular, methods such as GRPO (Shao et al., 2024), which efficiently train large reasoning models (LRMs) (Guo et al., 2025a; Chen

*Equal contribution 1KAIST, Daejeon, South Korea 2AITRICS, Seoul, South Korea. Correspondence to: Eunho Yang <eunhoy@kaist.ac.kr>.

Preprint. June 2, 2026.

ActualDifficulty

ActualLength

Predicted Difficulty Predicted Length

- (a) Poor Alignment of GRPO Trained Model.

ActualDifficulty

ActualLength Predicted Difficulty Predicted Length

- (b) Enhanced Alignment of MAPR Trained Model.

Figure 1. Meta-Awareness of GRPO vs MAPR. Predicted difficulty and solution length are elicited from both models using the same meta-prediction prompt, and the predictions are parsed from the model outputs. Difficulty is defined by Pass@1 scores, while length refers to the model output token count. Note that jitter is applied to discrete difficulty values to aid density visualization.

et al., 2025b) without an explicit critic model, have recently attracted considerable attention.

Beyond the success of LRMs, the paradigm of metaawareness, which is the ability to recognize its own knowledge and ignorance, has drawn increasing attention from the research community (Sui et al., 2025; Ha et al., 2025; De Sabbata et al., 2024; Chen et al., 2025a; Liu et al., 2025c; Zhang et al., 2025a; Shen et al., 2025; Tu et al., 2025; Shi et al., 2025; Qu et al., 2025). However, existing approaches remain constrained by their reliance on external model, curated dataset and reasoning pipelines that require human intervention.

To this end, we propose a novel RL framework, MetaAwareness via Predictive Reward (MAPR), which formalizes meta-awareness in reasoning models by rewarding the internal consistency of self-generated signals, thereby eliminating the need for external supervision. Our method introduces a self-predictive trajectory coupled with the primary

reasoning path, enhancing the model’s meta-awareness of its computational budget, knowledge boundaries, and cognitive strategy. These improved meta-predictions, shown in Figure 1, drive training efficiency through predictive gating, which prunes zero-variance prompts by identifying those that are either trivial or unsolvable, and early cutoff, which terminates long rollouts predicted to result in incorrect outcomes. Furthermore, the model could leverage the cognitive strategy to self-generate to provide hints for primary reasoning process.

Building on this foundation, we evaluate the effectiveness of our approach by combining with GRPO and DAPO (Yu et al., 2025; Shao et al., 2024), showing that our method is not dependent on a specific policy gradient algorithm. Remarkably, MAPR achieves substantial improvements in mathematical benchmarks with the strongest performance compared under the same compute budget. Finally, predictive gating and early cutoff deliver significant efficiency gains, attaining baseline performance 1.28 times faster than the GRPO training, with a higher accuracy score.

The contributions of this paper can be summarized as follows:

- • We introduce a predictive reward signal formulated as a parallel verification prompt, which enables the model to self-evaluate meta-awareness by alignment.
- • We experimentally show that the meta-prediction directly drives performance gain through paired analysis.
- • We propose MAPR-efficient, a post-training strategy with predictive gating and early cutoff, which achieves the strongest performance with minimal training compute.

### 2. Related Works

Meta-Cognitive Learning Meta-cognition is viewed as a prerequisite for self-improving LLMs (Liu & van der Schaar, 2025). Existing methods rely on extrinsic mechanisms with fixed action loops, limiting adaptability. Self-improving agents that plan, regulate, and reflect (Dong et al., 2025; Didolkar et al., 2025) or refine prompts via past reasoning (Qiu et al., 2025; Liu et al., 2025c) entangle control with reasoning, often causing interference. In contrast, our approach disentangles the meta and solution path separately for stable training on meta-awareness.

Other works require curated datasets (Ha et al., 2025), or delegate control to external verifiers (Ma et al., 2025; He et al., 2025) or multi-agent systems (Wan et al., 2025; Yang & Thomason, 2025; Bilal et al., 2025; Khandelwal et al., 2025), reducing scalability of meta-cognitive training. Training-free heuristics such as confidence-based stopping (Yang et al., 2025b; Qiao et al., 2025; Lu et al., 2025) or correctness checks (Ma et al., 2025) offer efficiency but

lack genuine language-level meta-cognition. In contrast, our approach does not rely on human-curated reasoning pipelines, external verifiers, PRMs, or specialized datasets targeting meta-cognitive ability, but rather leverages the self-generated signals to encourage alignment between the meta-prediction and primary thinking process.

Self-Control for Efficient Training Another direction that leverages meta-cognition is to regulate reasoning efficiency by allocating budgets via difficulty assessment (Chen et al., 2025a; Tu et al., 2025; Shi et al., 2025; Qu et al., 2025; Huang et al., 2025; Ji et al., 2025; Di & JoyJiaoW, 2025; Han et al., 2024b; Fang et al., 2025; Yang et al., 2025c; Zhang et al., 2025b; Wang et al., 2025; Zhang et al., 2025a; Shen et al., 2025), constraining output length with penalties or fixed limits (Aggarwal & Welleck, 2025; Li et al., 2025; Xiang et al., 2025; Zhang & Zuo, 2025), and adaptively stopping, continuing, or reflecting for compact reasoning (Ha et al., 2025; Zhang et al., 2025c; Dai et al., 2025). While these methods improve inference-time efficiency, they focus on making reasoning shorter or faster at inference time, often at the expense of reasoning performance drop. In contrast, we target efficiency during the post-training phase, achieving both efficiency and improved performance during model training rather than the inference.

### 3. MAPR: Meta-Awareness via Predictive Reward and MAPR-efficient

We first provide background on group relative policy optimization (GRPO) (Section 3.1). Then we show our method: (i) MAPR, which endows the LLM with the capability to perform accurate meta-predictions (Section 3.2); and (ii) MAPR-efficient, an efficiency-enhanced version that accelerates MAPR through predictive gating and early cutoff. (Section 3.3).

###### 3.1. Preliminaries

We present an overview of GRPO, which is a popular RL algorithm for post-training reasoning models. The old policy model πθ

produces a group of G responses given prompt q from tasks P(Q), creating rollouts O = {o1,··· ,oG}. Each response is assigned a reward {r1,··· ,rG} based on the rule-based verification of the extracted answer against the ground truth.

old

The objective of GRPO is formulated as,

 

  1

|oi|

G

1 |oi|

Ji,tclip(θ) − βDKL(πθ||πref)

E q∼P(Q)

G

{oi}∼πθold(·|q)

t=1

i=1

where Ji,tclip(θ) = min ρi,t(θ)Aˆi,t,clip ρi,t(θ),1 − ϵ,1 + ϵ A ˆi,t .

(1)

Note that ρi,t(θ) = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) denotes the importance sampling ratio, and πθ represents the current policy model.

πref is the reference model. clip(·) restricts the importance sampling ratio between [1 − ϵ,1 + ϵ]. The advantage is cal-

Hinting

Concepts used Difficulty Length Cutoff

Predict

|Meta|
|---|

Gating

G i=1)

culated as Aˆi,t = ri−mean({ri}

std({ri}Gi=1) . Following the practice of recent GRPO variants (Liu et al., 2025a; Zhang & Zuo, 2025; Zheng et al., 2025; Yu et al., 2025), we set β = 0 to remove the KL divergence term.

Align MAPR-eff

Solve

True Pass@k

…

- 3.2. MAPR: Designing Meta-Awareness via Predictive Reward

|Solution|
|---|

Overall Pipeline of MAPR Building on the GRPO-based framework, the policy model is prompted with two distinct inputs: solution prompt qsol and meta prompt qmeta. The solution and meta rollouts are executed simultaneously, but the rewarding pipelines differ. Solution rollouts are verified against static ground truth using rule-based verification, while meta rollouts are verified against empirical statistics derived from the solution rollouts as dynamic ground truth.

True token length

Figure 2. Overall Framework of MAPR and MAPR-efficient MAPR predicts and solves in parallel from given meta and solution prompts. The predicted values are verified against true pass@k, token length, and used concepts extracted from the solution rollouts. The efficient version, MAPR-efficient, applies predictive gating and length cutoff for efficient training.

The solution prompt qsol instructs the model to solve the problem via chain-of-thought, generating a group of G solution rollouts as detailed in Section 3.1. For verification on meta-prediction, the average Pass@1 score over G rollouts (p), the range of output token length from correct rollouts ([lmin,lmax]) is extracted, and the entire responses (O) are saved for predicted notion verification.

order to strongly penalize higher errors in prediction.

Length Reward. The length alignment reward checks if the predicted length falls within the range of correct responses. Formally, we assign the reward if the predicted length ˆl falls in-between the min-max range of correct responses as

Simultaneously, the meta prompt qmeta, instructs the model to predict the expected difficulty as Pass@1 score (pˆ), the expected length of correct response (ˆl), and a set of problemsolving notions (Gˆnotion). We generate M independent metarollouts,1 and reward each by how accurately it predicts the output length, problem difficulty, and used notions from the solution rollouts. The meta rewards, {r1meta,...,rMmeta}, are then normalized within the group of M rollouts to compute advantages. The reward computation for each meta component is detailed below. For reproducibility, we provide the complete code snippet in Section C.

rlength = lmin ≤ ˆl ≤ lmax .

If no correct solution exists, then we set the reward as 0.

Notion Reward. The notion reward evaluates whether the predicted problem-solving notion emerges more frequently in correct rollouts than in incorrect ones. More formally, for a single notion n ∈ Gˆnotion, we count the number of correct responses containing n (denoted as ccorr, n) and the number of incorrect responses containing n (denoted as cwrong, n).

Then, the notion reward is defined as

rnotion = En∼Gˆ

[c corr, n > cwrong, n] .

Difficulty Reward. The difficulty alignment reward measures the proximity between the predicted pass-rate pˆ and the actual pass-rate p. This is the proportion of correct answers among G rollouts for question q. This allows the model to learn how hard the given question is for the current knowledge boundary of the model.

notion

Notions present in the problem statement are excluded to prevent reward hacking, and lemma-based matching is used for counting.

Then, the meta reward is defined as the average of three componenets,

We compute the accuracy score as an exponential decay function of the normalized prediction error, given by

rlength + rdifficulty + rnotion 3

rmeta =

. (2)

###### rdifficulty = 0.01|p−pˆ|.

3.3. MAPR-efficient: Meta-based Active Control for Efficient Post-Training

A deviation of a single unit in difficulty prediction approximately halves the reward with the base number 0.01, in

MAPR-efficient is a variant of MAPR that can further boost training efficiency by leveraging the length and difficulty predictions.

1The full meta-prediction prompt template is deferred to Section A.

- Table 1. Performance of GRPO and MAPR for Math benchmarks. Pass@1 and Pass@8 scores are reported with standard deviations over 32 samplings. The overall performance of our method MAPR surpasses baseline GRPO method by large margin.

GRPO GRPO w/ MAPR Pass@1 Pass@8 Pass@1 Pass@8

Benchmark

Qwen3-4B Base Model

|AIME’24<br>AIME’25 AMC23 MATH500 Minerva Olympiad<br>|17.50±4.00 33.60±5.96 11.77±4.56 25.56±4.40 59.30±6.40 84.93±3.90 79.61±0.91 90.12±0.59 42.27±1.53 59.70±0.91 44.47±1.04 61.99±0.61<br><br>|26.15±3.32 (+ 49.43%) 48.82±5.32 (+ 45.30%) 21.56±4.40 (+ 83.18%) 37.17±3.63 (+ 45.42%) 70.16±4.78 (+ 18.11%) 93.18±1.90 (+ 9.71%)<br><br>84.52±0.74 (+ 6.17%) 93.74±0.42 (+ 4.02%) 41.12±2.00 (- 3.18%) 63.78±1.35 (+ 6.83%) 53.38±0.96 (+ 20.04%) 69.74±0.69 (+ 12.50%)<br><br>|
|---|---|---|
|Average|42.49±3.07 59.31±2.73<br><br>|49.48±2.70 (+ 13.04%) 67.73±2.22(+ 14.20%)|

Qwen3-8B Base Model

|AIME’24<br><br>AIME’25 AMC23 MATH500 Minerva Olympiad<br><br><br>|28.54±4.12 53.96±4.07 22.19±3.63 38.74±4.05 73.67±5.60 92.77±2.43 85.75±0.66 94.31±0.49 43.21±2.12 64.00±1.14 54.03±1.22 70.04±0.70<br><br>|34.17±5.54 (+ 19.72%) 63.80±4.98 (+ 18.24%) 28.44±5.41 (+ 28.17%) 45.96±4.41 (+ 18.64%) 79.53±4.26 (+ 7.95%) 94.39±1.80 (+ 1.75%) 88.05±0.82 (+ 2.68%) 95.35±0.49 (+ 1.1%) 47.21±1.74 (+ 9.26%) 68.21±1.23 (+ 6.58%) 56.86±0.85 (+ 5.24%) 71.87±0.51 (+ 2.61%)|
|---|---|---|
|Average<br><br>|51.23±2.89 68.97±2.15|55.71±3.10 (+ 8.74%) 73.26±2.24 (+ 6.22%)|

Qwen3-14B Base Model

|AIME’24<br><br>AIME’25 AMC23 MATH500 Minerva OlympiadMath<br><br><br>|38.54±4.30 58.55±4.07 27.92±4.69 45.56±3.87 81.56±4.98 96.20±1.66 88.73±1.03 96.02±0.36 45.03±1.73 66.42±1.06 59.04±0.90 73.03±0.58<br><br>|44.27±5.64 (+ 14.87%) 68.30±3.57 (+ 16.65%) 31.25±5.12 (+ 11.93%) 53.57±5.51 (+ 17.58%) 86.02±4.16 (+ 5.47%) 95.12±1.48 (- 1.12%) 89.93±0.88 (+ 1.35%) 96.39±0.34 (+ 0.38%)<br><br>50.36±1.53 (+ 11.84%) 69.20±0.96 (+ 4.19%) 61.59±0.89 (+ 4.32%) 74.37±0.65 (+ 1.83%)|
|---|---|---|
|Average<br><br>|56.80±2.94 72.63±1.93|60.57±3.04 (+ 6.63%) 76.15±2.09 (+ 4.85%)|

Overall Pipeline of MAPR-efficient To encourage metaawareness before accelerating the training phase, we first perform self-alignment based policy updates for the early k steps of update with self-predictive alignment reward, until the policy model shows stable meta-prediction alignment with the true solution rollouts. After k-th step, we alter into non-parallel pipeline that executes meta-predictions first, for predictive gating, followed by solution rollouts, applying early length cutoff. We may also utilize the predicted notions to provide additional hint for the model in solving the questions.

Predictive gating acts as a pre-computation filter for tasks that are deemed either trivial or impossible to solve. For a given question q, gating engages only when the standard deviation across M predicted pass-rates falls below σpg and the average prediction is 0 or 1. Distinct from methods like DAPO, which prune after expensive solution rollouts, our approach conserves computation by gating before the rollout phase. Since our primary objective is training efficiency, we employ static online gating; dynamically re-evaluating previously gated tasks would require periodic, costly metapredictions on excluded data.

Length cutoff restricts generation to the predicted length, scaled by a margin lLC. As the MAPR length reward incentivizes accurate prediction for correct rollouts, exceeding this threshold is highly unlikely to yield a correct answer, despite the cost of generating additional tokens. Therefore, MAPR-efficient utilizes the length prediction as a hard threshold to terminate rollouts once the limit is reached.

Additionally, notion feedin is implemented by appending the hint “The problem could be solved using the following math notions” to the problem statement, providing auxiliary guidance during the solution rollout phase.

### 4. Experiments

In this section, we provide the details of training and evaluation configuration in Section 4.1. Then we demonstrate the performance gain and efficiency driven by MAPR and MAPR-efficient in Section 4.2. In addition, we systematically analyze the components of our method through ablation studies in Section 4.3.

[Figure 1]

[Figure 2]

#####  Acc

 Train Step  rpred

(a) Sensitivity Analysis GRPO MAPR

(b) Alignment of Difficulty

- Figure 3. Impact of Meta-Awareness on Training Dynamics. (a) We observe a significantly steeper gradient for meta-awareness (rpred) compared to training steps, suggesting that increased meta-awareness drives performance more effectively than training duration alone. (b) The MAPR Pass@1 surge (steps 80-120) coincides precisely with the drop-then-align phase in difficulty prediction (orange), implying that predictive calibration correlates strongly with performance increase.

###### 4.1. Train and Evaluation Details

Training Details. We use VeRL with the DeepScaleR (Luo et al., 2025) dataset, batch size 128, learning rate 1e-6, 10% weight decay, maximum response length 8K, and GRPO without KL term. Training runs for one epoch (314 steps) using AdamW (Loshchilov & Hutter) with 20 warm-up steps, gradient clipping at 1.0, and clipping range for GRPO between [ϵlow = 0.2,ϵhigh = 0.28]. The rollouts use temperature 1.0 and top-p value of 1.0. The number of rollouts is 16 for the response generation, and 8 for meta prediction.

Evaluation Configuration. We use the provided math scoring function in VeRL to measure the accuracy of the predicted answer and ground truth answer, sampling 32 responses, with 16k maximum response length and temperature set at 0.6.

We evaluate the performance of our method using six widely used mathematical reasoning benchmarks, AIME24, AIME25, AMC23, MATH500 (Hendrycks et al.), Minerva, and OlympiadBench (He et al., 2024). Experiments are conducted on Qwen3 8B base model unless otherwise stated.

###### 4.2. Analysis on MAPR and MAPR-efficient

MAPR Excels in Math Benchmark MAPR excels the baseline in six math benchmarks - AIME24, AIME25, AMC23, MATH500, Minerva, and OlympiadBench (Table 1). Across all mathematical datasets, our method MAPR shows great improvement over the baseline GRPO performance, showing an average of 13.04% of improvement in Qwen3-4B model, 8.74% in Qwen3-8B model, and 6.63% in Qwen3-14B model. Among the six benchmarks, MAPR gains maximum performance on intermediate to hard level (AIME, AMC, Olympiad, Minerva), while the performance boost for MATH500 shows performance saturation especially for large scaled model of 14B. We also demonstrate

the superior ability of MAPR on out-of-domain benchmarks, ranging from logical, scientific, to coding domains in Table 8.

Meta-Awareness Directly Enhances Performance In Figure 3a, we assess whether performance gains arise from improvements in the reward metric rpred or from extended training. We conduct a paired analysis comparing marginal accuracy gains (∆Acc) with both training steps and metaawareness measured by rpred. Checkpoints are sampled every 20 steps across six mathematical benchmarks, and for each question we pair model states (t,q) from different steps to control for input variation.

We plot step differences versus accuracy gains using a jittered distribution. To isolate the effect of meta-awareness, we compute ∆rpred = |rpredt − rpredq | and plot binned values against ∆Acc. Accuracy improves more steeply with respect to rpred than with training steps, indicating that performance is more sensitive to meta-cognitive calibration than to additional training compute.

Meta-prediction Dynamics During MAPR Training As shown in Figure 3b, a critical divergence in training dynamics appears when analyzing the model’s self-prediction of problem difficulty (pˆ) versus the true difficulty (p). GRPO exhibits consistent overconfidence, predicting a Pass@1 value exceeding 0.8, despite its true score remaining significantly lower.

In contrast, while MAPR also begins with initial overconfidence, the meta-awareness objective drives a corrective drop-then-align behavior in Figure 3b. The predicted difficulty drops sharply until step 80, recalibrating to match the true Pass@1. Crucially, this coincides with the rapid ascent in the true Pass@1 score, suggesting that accurate selfassessment correlates with the performance gains observed in MAPR. Similar tendency is also observed in length pre-

###### GRPO 1 epoch MAPR-eff 1 epoch MAPR 1 epoch

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

M R

M R

Metric Value Precision 0.9417 Recall 0.8739 F1 Score 0.9065

M R

M R

M R

M R

Standard Error

(a) Predictive Gating (b) Length Cutoff

Figure 5. (a) Accuracy of Predictive Gating (b) Standard Error of Length Cutoff from MAPR-efficient.

- Figure 4. Accuracy vs Wall Clock Time. Average performance across six math benchmarks. Gray vertical lines indicate epoch milestones. Both MAPR-efficient and MAPR achieve Paretosuperiority over the GRPO baseline, showing higher accuracy for the same compute expenditure.

fectively prevents the model from generating futile extra tokens that would lead to wrong answers. The distribution of standard error values toward the positive range indicates that the length cutoff mechanism serves as a highly effective means of conserving tokens.

diction (l vs ˆl), which is shown in Section D.

###### 4.3. Ablation Studies

MAPR-efficient Achieves the Strongest Performance with Minimal Training Compute Under the same compute budget (wall clock time), we demonstrate that MAPR and MAPR-efficient surpass the performance of GRPO. In Figure 4, we demonstrate the average accuracy over six mathematical benchmarks at three different timestamps, which are 1 epoch (314 steps) duration of three model variants GRPO, MAPR-efficient, and MAPR, emphasized as gray vertical lines on the plot. Under the same train compute at three different timestamps, MAPR-efficient and MAPR consistently outperform the baseline method GRPO by a large margin. This proves the efficacy of our method in achieving a large performance gain even under same amount of training compute time.

Meta-prediction Components To attribute performance improvements to individual factors, we employ a ShapleyR2 decomposition based on linear regression and plot the value in Figure 6a. We let design matrix as the feature matrix composed of paired differences (∆rdifficulty,∆rlength,∆rnotion,∆step), and let ∆Acc be the target variable to compute the Shapley-R2 values. The details are deferred due to a spatial constraint.

Moreover, we conduct ablation on training our method MAPR with each of the three components and using all three. The results in Figure 6b show that using all three components of meta-prediction shows overall superior performance over all benchmarks.

Prediction Performance of PG and LC To evaluate the reliability of predictive gating and length cutoff in MAPRefficient, we compare gating and cutoff decisions against the ground-truth. Using unseen part of DeepScaleR train dataset, Figure 5a reports the performance of predictive gating in terms of precision, recall, and F1 score against true zero-variance questions. These metrics evaluate whether the predicted difficulty value of 0 or 1 with a standard deviation below σpg matches the true zero variance. Moreover, Figure 5b shows the standardized error value of the length cutoff decision. The standard error is calculated as

Number of Meta Rollouts In Figure 6c, we analyze the effect of reducing the number of meta-prediction path from 16, which is the default rollout number for the primary solution path. Evaluation shows that using 8 rollouts for meta-prediction, in combination with 16 rollouts for a solution path, shows that optimal result in terms of both train compute and performance.

Introducing the meta-prediction path requires the policy model to generate additional meta-predictions on the solution length, pass-rate, and high-level concepts. However, we show that the average token length and number of rollouts additionally required for meta-predictions only amount to 15.5% of total rollout compute as shown in Table 2.

(ˆli) /σ(ˆl), which quantifies the deviation of the true length from the prediction, normalized by meta-prediction uncertainty. The distribution shows the standard error from correct rollouts (green) and incorrect rollouts (red). While the distribution for correct rollouts are centered around zero error (0), incorrect rollouts are distributed in larger values. This demonstrates that the length predictions are highly accurate, and the cutoff strategy ef-

Ei∼O

###### (li) − Ei∼O

sol

meta

Hyper-parameters for MAPR-efficient (σPG, lLC) Following an initial k-step training phase for length and difficulty meta-prediction, MAPR-efficient applies predictive gating (σPG) and length cutoff (lLC).

100

| |77.9<br><br>88.5 79.5<br><br>88.0 76.9<br><br>87.4| | | | | | |
|---|---|---|---|---|---|---|---|
| |34.0<br><br>57.2<br><br>47.9 34.2<br><br>56.9<br><br>47.2 33.8<br><br>55.8<br><br>46.1| | | | | | |
| | | | | | | | |
| |24.3<br><br>28.426.1| | | | | | |
| | | | | | | | |

AIME24 AIME25

Pass@1Score

80

60

AMC23 MATH 500

rnotion rdi↵

40

20

rlength

AIME24 AIME25 AMC23 Olympiad MATH500 Minerva

Olympiad Math MinervaMath Notion Difficulty Length

Rollout 4 Rollout 8 Rollout 16

All Three GRPO

(b) Ablation of meta components (Maximum set to the score of ‘All three’ components for visualization purpose).

(c) Ablation on number of meta-prediction rollouts.

(a) Shapley R2 Analysis on component-wise contributions.

- Figure 6. Component Analysis and Ablation Studies The contribution of our meta-aware predictive reward components analyzed through Shapley values (left), meta type ablation performance (center), and the performance over different numbers of meta rollouts (right).

120000 140000 160000 Total Training Time (sec)

- 52

- 53

- 54

- 55

Accuracy(%)

Gating 0.1

Gating 0.05

Ours

(a) Gating σPG Ablation.

150000 155000 160000 165000 Total Training Time (sec)

54.0

54.5

55.0

55.5

Accuracy(%)

Cutoff x1.5 Cutoff x1.0

Cutoff x0.5

Ours

(b) Cutoff lLC Ablation.

0 40 80 120 160

Train Steps

20.0

22.5

25.0

27.5

30.0

32.5

35.0

37.5

40.0

Pass@1Score

Without Cutoff / Gating

from 0

from 40 from 80 from 120

(c) Start Step Ablation.

- Figure 7. Ablation on Hyper-parameters for MAPR-efficient. Pass@1 scores over choices of (a) predictive gating (PG), (b) length cutoff (LC) and (c) start step (k).

- Table 2. Comparison on token length between original rollouts and meta prediction rollouts. The increased token is only 15.5% of entire tokens.

Optimizing the Transition Step k The timing of these efficiency mechanisms is critical, as the MAPR-efficient requires an initial calibration period. Figure 7c compares the performance of four start-step variants (k ∈ {0,40,80,120}) across 160 training steps. We observe that while premature activation (k < 80) slightly degrades final accuracy, initiating predictive gating and length cutoff at step 80 achieves performance parity with later starts (e.g., k = 120) while providing earlier computational savings. We set k = 80 for MAPR-efficient, as it represents the optimal balance between meta-prediction calibration and resource efficiency. We show that this result is consistent across model sizes in Table 3. All configurations converge to consistent final performance even when starting from different start steps for 14B model.

| |Avg. Tokens Rollout No. Proportion<br><br>|
|---|---|
|Solution Meta-Pred<br><br>|6251 16 84.5% 2293 8 15.5%|

In Figure 7a, we evaluate the impact of the predictive gating parameter, σPG, which determines the threshold for skipping prompts based on the standard deviation of predicted difficulty. Lower values of σPG ensure that gating only occurs when the meta-predictions have low variance regarding task difficulty. Similarly, Figure 7b illustrates the effect of the length cutoff margin, lLC, a multiplier that scales the threshold for early trace termination. While both PG and LC incur a marginal performance trade-off relative to MAPR, these costs are effectively offset by substantial gains in training efficiency and reduced wall-clock time.

Base Number The choice of base number for difficulty reward rdifficulty is set as 0.01 to halve the reward per unit difference between predicted and true difficulty. We test the robustness of our method MAPR on difference base numbers 0.05, 0.01, and 0.02 on 4B and 8B scale models.

- As shown in Table 4, the results with different base numbers show consistent scores across model sizes and base numbers, except for extreme value of 0.005 on the 4B model, which degrades performance.

Meta-Predicted Notion Feedin We test whether the notions generated from meta-predictions serve as a auxiliary hint for the original solution rollouts by incorporating it into the question using prompt format: Question + ‘The problem could be solved using following math notions’. To examine the impact of such notion feed-in on performance, we conduct the following experiment which uses the predicted notions as hints to the question solving phase.

- As shown in Table 5, incorporating notion feed-in (MAPR

+ NF) yields a small amount of performance gain compared to the variant without notion feed-in (MAPR), but the gain is limited. This suggests a high degree of information overlap, implying that the model likely implicitly possesses these concepts through MAPR, which enhances metaawareness, making explicit hinting redundant. Therefore, we test whether the extracted notions boost the performance of a baseline GRPO model. The notions are extracted from the MAPR model and fed into a separately trained GRPO model using keywords with high notion reward scores. Although this setting is far from practical deployment, as it requires cross-model notion extraction and transfer, the substantial improvement achieved by GRPO + NF demonstrates that the extracted notions are highly effective in enhancing reasoning performance.

Ablation on RL Algorithm MAPR is flexibly applicable to GRPO variants. We show the superiority of MAPR combined with DAPO algorithm in Table 6. Unlike DAPO, which requires a redundant sampling phase to filter out tasks with zero-variance, our method is able to bypass the sampling for solution rollouts and preemptively gate such tasks. Even with greater efficiency, MAPR outperforms all six mathematical benchmarks by a large margin.

We train Qwen3-8B-Base with DAPO for three epochs (315 steps), which is equivalent to one epoch of GRPO (314 steps) in terms of the total number of gradient updates. Moreover, we disable the overlong reward shaping term in DAPO. In our setting, this term imposes an overly strong length constraint, which prevents the model from sufficiently increasing its reasoning depth. Empirically, we observe that keeping this term results in lower final performance. We

- Table 3. Transitioning step ablation on Qwen3-14B Base Model.

|Start Step<br><br>|AIME24 AIME25 AMC23 Avg<br><br>|
|---|---|
|0 40 80 120<br><br>|35.83 26.04 75.78 45.88<br><br>33.12 25.83 75.31 44.75<br>34.27 26.98 75.94 45.73 30.52 27.40 77.34 45.09<br>|

- Table 4. Performance comparison across different base numbers for 4B and 8B models.

(a) Qwen3-4B Base Model

Base Num AIME’24 AIME’25 AMC’23 MATH500 Minerva Olympiad Avg 0.005 16.98 14.58 61.95 79.62 43.55 45.21 43.65

- 0.01 26.15 21.56 70.16 84.52 41.12 53.38 49.48

- 0.02 26.77 23.33 70.47 84.84 43.11 53.24 50.29 (b) Qwen3-8B Base Model

Base Num AIME’24 AIME’25 AMC’23 MATH500 Minerva Olympiad Avg 0.005 34.38 25.73 78.05 88.34 48.35 57.61 55.41

- 0.01 34.17 28.44 79.53 88.05 47.21 56.86 55.71

- 0.02 33.54 24.17 79.92 87.74 46.19 56.51 54.68

- Table 5. Performance of MAPR on Qwen3-8B across six mathematical benchmarks. All metrics are Pass@1. NF denotes Notion-FeedIn.

| |GRPO|GRPO + NF<br><br>|MAPR|MAPR + NF|
|---|---|---|---|---|
|AIME’24<br><br>AIME’25 AMC’23 MATH500 Minerva Olympiad<br><br><br>|28.54±4.12 22.19±3.63 73.67±5.60 85.75±0.66 43.21±2.12 54.03±1.22|33.96±5.88 23.85±3.64 77.97±5.08 86.52±1.14 45.44±1.54 56.63±1.10<br><br>|34.17±5.54 28.44±5.41 79.53±4.26 88.05±0.82 47.21±1.74 56.86±0.85<br><br>|35.10±4.96 25.94±4.34 78.91±5.01 88.51±0.79 48.38±1.33 57.06±0.97|

therefore remove it to avoid unnecessarily restricting the model’s reasoning capacity under our training configuration.

Ablation on Different Model Our method also demonstrates consistent improvements when applied to different model families, Llama 3.1 8B Instruct (Grattafiori et al., 2024) and Gemma 2 9B IT (Team et al., 2024). Unlike Qwen3 family that are explicitly trained on long CoT reasoning datset, these two model families are relatively under-trained on mathematical reasoning dataset. Therefore, following the convention of existing works (Zhu et al., 2025; Liu et al., 2025b), we train both models with easier dataset, train split of MATH dataset, with extended training epochs of 3. All the other configurations are kept the same. In Table 7, we report the evaluation result on AMC’23, MATH500, Minerva, and OlympiadBench, excluding AIME’24 and AIME’25 for extremely low accuracy nearing 0 for both methods even after training. Overall, our method achieves consistent gains not only for Qwen3 but also for Llama 3.1 and Gemma 2 models.

### Conclusion

We present MAPR, a meta-aware reinforcement learning framework that fosters meta-cognitive ability by selfalignment. By incorporating information achieved by metathinking trajectories into training, our method enables stable and efficient optimization by integrating predictive gating and early cutoff. Empirically, MAPR accelerates RL-based post-training while improving both in-domain and out-ofdomain performance, demonstrating notable gains in accu-

- Table 6. Performance comparison of MAPR with DAPO, trained with Qwen3-8B base model.

DAPO DAPO + MAPR

|Benchmark Pass@1 Pass@8<br><br>|Pass@1 Pass@8|
|---|---|
|AIME’24 29.48±4.04 52.54±3.99<br>AIME’25 23.75±3.83 37.00±2.96 AMC’23 78.12±5.06 94.86±1.98 MATH500 87.44±0.74 94.40±0.38 Minerva 45.22±1.80 65.43±1.04 Olympiad 55.97±0.89 71.13±0.63<br>|36.56±5.97 (+ 24.02%) 66.28±3.67 (+ 26.15%) 25.94±4.39 (+ 9.22%) 42.04±2.83 (+ 13.62%)<br><br>78.52±4.56 (+ 0.51%) 95.14±1.97 (+ 0.29%) 88.96±0.85 (+ 1.74%) 95.10±0.39 (+ 0.74%) 47.97±2.10 (+ 6.08%) 68.77±0.99 (+ 5.10%) 57.53±1.08 (+ 2.79%) 73.61±0.73 (+ 3.49%)<br><br>|
|Average 53.33±2.73 69.23±1.83<br><br>|55.01±3.16 (+ 3.15%) 73.49±1.76 (+ 6.15%)|

- Table 7. Comparative performance of MAPR and GRPO across model variants.

- (a) Llama 3.1 8B Instruct (3 Epochs / 174 steps) GRPO MAPR

|Benchmark Pass@1 Pass@8<br><br>|Pass@1 Pass@8|
|---|---|
|AMC’23 25.23±4.16 42.92±3.23 Math500 52.80±1.47 71.27±0.91 Minerva 31.70±1.69 50.15±1.21 Olympiad 19.39±0.87 32.36±0.66<br><br>|31.02±3.88 56.52±3.42 53.54±1.14 71.68±0.79 31.86±1.54 50.39±1.19 19.89±0.79 35.61±0.70|

- (b) Gemma 2 9B IT (3 Epochs / 174 steps) GRPO MAPR

|Benchmark Pass@1 Pass@8<br><br>|Pass@1 Pass@8|
|---|---|
|AMC’23 26.88±4.80 46.48±3.84 Math500 54.07±0.90 71.70±0.79 Minerva 34.09±1.54 47.90±0.98 Olympiad 20.29±0.95 36.56±0.75<br><br>|29.22±4.52 58.88±3.54 57.24±1.16 76.57±0.75 33.58±1.85 49.57±1.11 21.66±0.80 38.59±0.76|

racy and generalization. These results highlight the promise of meta prediction as a principled avenue for enhancing reasoning models.

### Impact Statement

This paper presents MAPR, a framework designed to verify and utilize meta-awareness in reasoning models. The primary broader impact of our work lies in the improvement of computational efficiency for Large Language Models. By enabling models to self-regulate, such as determining optimal thinking duration and filtering out unsolvable prompts, our approach significantly reduces the computational resources required for both training and inference. This contributes to reducing the environmental footprint associated with developing and deploying large-scale reasoning systems. While advancing reasoning capabilities generally implies the need for careful consideration of dual-use risks, our work specifically focuses on internal verification and efficiency, and we do not foresee specific negative societal consequences unique to this method.

### Acknowledgement

This work was supported by Institute for Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2019II190075, Artificial Intelligence Graduate School Program(KAIST)) and National Research Foundation of Korea (NRF) grant (No.RS-2023-00209060, A Study on Optimization and Network Interpretation Method for Large-Scale Machine Learning) funded by the Korea government (MSIT).

### References

Aggarwal, P. and Welleck, S. L1: Controlling how long a reasoning model thinks with reinforcement learning. 2025. doi: 10.48550/arXiv.2503.04697. URL https: //arxiv.org/abs/2503.04697.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Bilal, A., Mohsin, M. A., Umer, M., Bangash, M. A. K., and Jamshed, M. A. Meta-thinking in llms via multiagent reinforcement learning: A survey. arXiv preprint arXiv:2504.14520, 2025.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Chen, Q., Peng, D., Liu, J., Su, H., Guan, J., Qin, L., and Che, W. Aware first, think less: Dynamic boundary self-awareness drives extreme reasoning efficiency in large language models. arXiv preprint arXiv:2508.11582, 2025a.

Chen, Y., Yang, Z., Liu, Z., Lee, C., Xu, P., Shoeybi, M., Catanzaro, B., and Ping, W. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. 2025b. doi: 10.48550/arXiv.2505.16400. URL https://arxiv.org/abs/2505.16400.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. URL https: //arxiv.org/abs/1803.05457. Use the ARCChallenge split for ARC-C results.

Dai, M., Yang, C., and Si, Q. S-grpo: Early exit via reinforcement learning in reasoning models. arXiv preprint arXiv:2505.07686, 2025.

De Sabbata, C. N., Sumers, T. R., AlKhamissi, B., Bosselut, A., and Griffiths, T. L. Rational metareasoning for large language models. arXiv preprint arXiv:2410.05563, 2024.

Di, X. and JoyJiaoW. Enhancing math reasoning in smallsized llms via preview difficulty-aware intervention. 2025. doi: 10.48550/arXiv.2508.01604. URL https://ar xiv.org/abs/2508.01604.

Didolkar, A., Balla, N., Arora, S., and Goyal, A. Metacognitive reuse: Turning recurring llm reasoning into concise behaviors. arXiv preprint arXiv:2509.13237, 2025.

Dong, H., Ye, H., Zhu, W., Jiang, K., and Song, G. Meta-r1: Empowering large reasoning models with metacognition. arXiv preprint arXiv:2508.17291, 2025.

Fang, G., Ma, X., and Wang, X. Thinkless: Llm learns when to think. arXiv preprint arXiv:2505.13379, 2025.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gu, A., Roziere, B., Leather, H. J., Solar-Lezama, A., Synnaeve, G., and Wang, S. CRUXEval: A benchmark for code reasoning, understanding and execution. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 16568–16621. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v2 35/gu24c.html.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Guo, M.-H., Xu, J., Zhang, Y., Song, J., Peng, H., Deng, Y.-X., Dong, X., Nakayama, K., Geng, Z., Wang, C., Ni, B., Yang, G.-W., Rao, Y., Peng, H., Hu, H., Wetzstein, G., and Hu, S.-m. R-bench: Graduate-level multi-disciplinary benchmarks for llm & mllm complex reasoning evaluation. arXiv preprint arXiv:2505.02018, 2025b. URL https://arxiv.org/abs/2505.02018.

Ha, R., Li, C., Pu, R., and Su, S. From ”aha moments” to controllable thinking: Toward meta-cognitive reasoning in large reasoning models via decoupled reasoning and control. 2025. doi: 10.48550/arXiv.2508.04460. URL https://arxiv.org/abs/2508.04460.

- Han, S., Schoelkopf, H., Zhao, Y., Qi, Z., Riddell, M., Zhou, W., Coady, J., Peng, D., Qiao, Y., Benson, L., et al. Folio: Natural language reasoning with first-order logic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 22017– 22031, 2024a.
- Han, T., Wang, Z., Fang, C., Zhao, S., Ma, S., and Chen, Z. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547, 2024b.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

He, T., Mu, R., Liao, L., Cao, Y., Liu, M., and Qin, B. Good learners think their thinking: Generative prm makes large reasoning model more efficient math learner. arXiv preprint arXiv:2507.23317, 2025.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Huang, S., Wang, H., Zhong, W., Su, Z., Feng, J., Cao, B., and Fung, Y. R. Adactrl: Towards adaptive and controllable reasoning via difficulty-aware budgeting. 2025. doi: 10.48550/arXiv.2505.18822. URL https://arxiv.org/abs/2505.18822.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum ?id=chfJJYC3iL.

Ji, Y., Zhao, S., Tian, X., Wang, H., Chen, S., Peng, Y., Zhao, H., and Li, X. How difficulty-aware staged reinforcement learning enhances llms’ reasoning capabilities: A preliminary experimental study. 2025. doi: 10.48550/arXiv.2504.00829. URL https://ar xiv.org/abs/2504.00829.

Khandelwal, V., Rossi, F., Murugesan, K., Miehling, E., Campbell, M., Ramamurthy, K. N., and Horesh, L. Language models coupled with metacognition can outperform reasoning models. arXiv preprint arXiv:2508.17959, 2025.

Li, G., Xia, T., Chang, Y., and Wu, Y. Length-controlled margin-based preference optimization without reference model. 2025. doi: 10.48550/arXiv.2502.14643. URL https://arxiv.org/abs/2502.14643.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558– 21572, 2023.

Liu, T. and van der Schaar, M. Position: Truly selfimproving agents require intrinsic metacognitive learning. In Forty-second International Conference on Machine Learning Position Paper Track, 2025. URL https:

//openreview.net/forum?id=4KhDd0Ozqe.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective. 2025a. doi: 10.48550/arXiv.2503. 20783. URL https://arxiv.org/abs/2503.2 0783.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective. In Conference on Language Modeling (COLM), 2025b.

Liu, Z., Gong, C., Fu, X., Liu, Y., Chen, R., Hu, S., Zhang, S., Liu, R., Zhang, Q., and Tu, D. Ghpo: Adaptive guidance for stable and efficient llm reinforcement learning. arXiv preprint arXiv:2507.10628, 2025c.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations.

Lu, J., Yu, H., Xu, S., Ran, S., Tang, G., Wang, S., Shan, B., Fu, T., Feng, H., Tang, J., et al. Prolonged reasoning is not all you need: Certainty-based adaptive routing for efficient llm/mllm reasoning. arXiv preprint arXiv:2505.15154, 2025.

Luo, M., Tan, S., Wong, J., Shi, X., Tang, W., Roongta, M., Cai, C., Luo, J., Zhang, T., Li, E., Popa, R. A., and Stoica, I. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.n otion.site/DeepScaleR-Surpassing-O1-P review-with-a-1-5B-Model-by-Scaling

-RL-19681902c1468005bed8ca303013a4e2,

2025. Notion Blog.

Ma, Z., Yuan, Q., Wang, Z., and Zhou, D. Large language models have intrinsic meta-cognition, but need a good lens. arXiv preprint arXiv:2506.08410, 2025.

Pan, L., Albalak, A., Wang, X., and Wang, W. Logiclm: Empowering large language models with symbolic solvers for faithful logical reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 3806–3824, 2023.

Qiao, Z., Deng, Y., Zeng, J., Wang, D., Wei, L., Meng, F., Zhou, J., Ren, J., and Zhang, Y. Concise: Confidenceguided compression in step-by-step efficient reasoning. arXiv preprint arXiv:2505.04881, 2025.

Qiu, Z., Chen, X., Chen, L., and Bai, R. Mela: A metacognitive llm-driven architecture for automatic heuristic design. arXiv preprint arXiv:2507.20541, 2025.

Qu, Y., Yang, M. Y., Setlur, A., Tunstall, L., Beeching, E. E., Salakhutdinov, R., and Kumar, A. Optimizing testtime compute via meta reinforcement fine-tuning. arXiv preprint arXiv:2503.07572, 2025.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https:

//openreview.net/forum?id=Ti67584b98.

Saparov, A. and He, H. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In The Eleventh International Conference on Learning Representations.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. 2024. doi: 10.48550/arXiv.2402.03300. URL https://arxiv.

org/abs/2402.03300.

Shen, Y., Zhang, J., Huang, J., Shi, S., Zhang, W., Yan, J., Wang, N., Wang, K., Liu, Z., and Lian, S. Dast: Difficultyadaptive slow-thinking for large reasoning models. arXiv preprint arXiv:2503.04472, 2025.

Shi, T., Wu, Y., Song, L., Zhou, T., and Zhao, J. Efficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520, 2025.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.

Sui, Y., He, Y., Cao, T., Han, S., Chen, Y., and Hooi, B. Meta-reasoner: Dynamic guidance for optimized inference-time reasoning in large language models. arXiv preprint arXiv:2502.19918, 2025.

Tafjord, O., Dalvi, B., and Clark, P. Proofwriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 3621–3634, 2021.

Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Ram´e, A., et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Tu, S., Lin, J., Zhang, Q., Tian, X., Li, L., Lan, X., and Zhao, D. Learning when to think: Shaping adaptive reasoning in r1-style models via multi-stage rl. arXiv preprint arXiv:2505.10832, 2025.

Wan, Z., Li, Y., Wen, X., Song, Y., Wang, H., Yang, L., Schmidt, M., Wang, J., Zhang, W., Hu, S., et al. Rema: Learning to meta-think for llms with multi-agent reinforcement learning. arXiv preprint arXiv:2503.09501, 2025.

- Wang, X., Hu, Z., Lu, P., Zhu, Y., Zhang, J., Subramaniam, S., Loomba, A. R., Zhang, S., Sun, Y., and Wang, W. Scibench: Evaluating college-level scientific problemsolving abilities of large language models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id= bq1JEgioLr.
- Wang, Y., Zhang, Y., Yu, T., Xu, C., Zhang, F., and Lian, F. Adaptive deep reasoning: Triggering deep thinking when needed. arXiv preprint arXiv:2505.20101, 2025.

Xiang, V., Blagden, C., Rafailov, R., Lile, N., Truong, S., Finn, C., and Haber, N. Just enough thinking: Efficient reasoning with adaptive length penalties reinforcement learning. 2025. doi: 10.48550/arXiv.2506.05256. URL https://arxiv.org/abs/2506.05256.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, C., Si, Q., Duan, Y., Zhu, Z., Zhu, C., Li, Q., Lin, Z., Cao, L., and Wang, W. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, 2025b.

Yang, J., Lin, K., and Yu, X. Think when you need: Self-adaptive chain-of-thought learning. arXiv preprint arXiv:2504.03234, 2025c.

Yang, W. and Thomason, J. Learning to deliberate: Metapolicy collaboration for agentic llms with multi-agent reinforcement learning. arXiv preprint arXiv:2509.03817, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zhang, J. and Zuo, C. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. 2025. doi: 10.48550/arX iv.2504.09696. URL https://arxiv.org/abs/ 2504.09696.

Zhang, J., Lin, N., Hou, L., Feng, L., and Li, J. Adaptthink: Reasoning models can learn when to think. arXiv preprint arXiv:2505.13417, 2025a.

Zhang, X., Wen, S., Wu, W., and Huang, L. Edge-grpo: Entropy-driven grpo with guided error correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025b.

Zhang, Z., Chen, Z., Li, M., Tu, Z., and Li, X. Rlvmr: Reinforcement learning with verifiable meta-reasoning rewards for robust long-horizon agents. 2025c. doi: 10.48550/arXiv.2507.22844. URL https://arxiv.

org/abs/2507.22844.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Zhong, W., Wang, S., Tang, D., Xu, Z., Guo, D., Chen, Y., Wang, J., Yin, J., Zhou, M., and Duan, N. Analytical reasoning of text. In Carpuat, M., de Marneffe, M.-C., and Meza Ruiz, I. V. (eds.), Findings of the Association for Computational Linguistics: NAACL 2022, pp. 2306– 2319, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.fin dings-naacl.177. URL https://aclanthology.o rg/2022.findings-naacl.177/.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= ftVlLG9cks.

### A. Default Meta-prediction Prompt for MAPR

Prompt [System]: You are a helpful assistant. [User]:

Think step-by-step between <meta> and </meta>, ensuring comprehensive and detailed reasoning especially for determining the pass rate and solution length values. For each component (math notion, pass rate, solution length), provide a comprehensive illustration or example during your reasoning in the <meta> section to clarify how each value is decided. When determining math notion, ensure that the notions listed do not directly include the notions already written in the problem statement. After </meta>, return a JSON object with three keys:

- - math notion (list[str])

- - pass rate (integer from 0 to 8)

- - solution length (integer from 128 to {max response length})

Problem: {problem}

In the meta-prediction prompt, math notion is predicted as a list[str], where each element denotes a mathematical notion required to solve the problem. We avoid predicting a continuous value (or an overly fine-grained scale) for pass rate, since it can introduce unnecessary variance and instability in the predicted difficulty. Instead, the prompt restricts pass rate to an integer in {0,...,8}. When computing the reward, we normalize this value by dividing it by 8. Finally, solution length is predicted as an integer between 128 and the maximum response length of the corresponding training setup.

### B. Meta-prediction Dynamics During MAPR Training

Table 8. Performance of GRPO and MAPR in Out-of-Domain benchmarks. Results are reported as pass@1 score.

Logical Reasoning Scientific Reasoning Coding

|Benchmark<br><br>|GRPO w/ MAPR<br><br>|Benchmark|GRPO w/ MAPR<br><br>|Benchmark|GRPO w/ MAPR|
|---|---|---|---|---|---|
|ProntoQA ProofWriter FOLIO Logi. Deduct AR-LSAT<br><br>|90.56 93.74 72.27 73.23 69.16 69.24 80.81 81.03 37.00 38.00|GPQA Diamond<br><br>R-Bench<br><br>ARC-Challenge<br><br>SciBench|51.72 53.72 60.69 61.68 93.10 93.13 28.33 29.64<br><br>|EvalPlus CRUX-O MBPP LiveCodeBench<br><br>|77.32 77.66 72.72 73.39 71.84 72.97 31.49 31.61|
|Avg.|69.96 71.05|Avg.<br><br>|58.46 59.54|Avg.|63.34 63.91|

MAPR Performance in Out-of-Domain Benchmarks The meta-awareness also benefits generalization ability of the reasoning model in out-of-domain logical, scientific, and coding benchmarks as shown in Table 8. For logical reasoning domain, we follow the setup of (Pan et al., 2023) and test on ProntoQA (Saparov & He), ProofWriter (Tafjord et al., 2021), FOLIO (Han et al., 2024a), LogicalDeduction (Srivastava et al.), and AR-LSAT (Zhong et al., 2022). For scientific reasoning, we use GPQA Diamond (Rein et al., 2024), R-Bench (Guo et al., 2025b), ARC-Challenge (Clark et al., 2018), and SciBench (Wang et al., 2024). For coding, we evaluate on EvalPlus (Liu et al., 2023), CRUX-O (Gu et al., 2024), MBPP (Austin et al., 2021), and LiveCodeBench (Jain et al., 2025). Although MAPR is not explicitly trained for generalization, strengthening meta-awareness consistently enhances out-of-domain performance. The base model is Qwen3-14B-Base, with the same training and evaluation configurations stated in the experiments section.

### C. Meta Reward Code Snippet

The implementation of our scoring mechanism is shown in the snippet below. We calculate a composite score based on the presence of mathematical notions, the length of the solution, and the difficulty pass rate.

- 1 def compute_score(solution_str: dict, ground_truth: dict) -> float:
- 2
- 3 # --- 1. Check Input Availability ---
- 4 has_notion = "math_notion" in solution_str
- 5 has_length = isinstance(solution_str.get("solution_length"), int)
- 6 has_diff = isinstance(solution_str.get("pass_rate"), int)
- 7
- 8 notion_score, length_score, acc_score = 0, 0, 0
- 9
- 10 # --- 2. Calculate Notion Score ---
- 11 if has_notion:
- 12 pred_notions = solution_str["math_notion"]
- 13 # Normalize to list
- 14 if isinstance(pred_notions, str):
- 15 pred_notions = [n.strip("[] ") for n in pred_notions.split(",")]
- 16
- 17 # Filter notions already in problem text
- 18 pred_notions = [n for n in pred_notions
- 19 if n not in ground_truth["problem"]]
- 20
- 21 # Count occurrences (Pos for correct resp, Neg for incorrect)
- 22 notion_counts = {n: 0 for n in pred_notions}
- 23 for resp, correct in zip(ground_truth["response"], ground_truth["score"]):
- 24 for n in pred_notions:
- 25 if n in resp:
- 26 notion_counts[n] += (1 if correct == 1 else -1)
- 27
- 28 # Score is ratio of notions with positive net utility
- 29 scores = [1 if cnt > 0 else 0 for cnt in notion_counts.values()]
- 30 if scores:
- 31 notion_score = sum(scores) / len(scores)
- 32
- 33 # --- 3. Calculate Length Score ---
- 34 if has_length:
- 35 correct_lens = [l for l, s in zip(ground_truth["length"],
- 36 ground_truth["score"]) if s == 1]
- 37 p_len = solution_str[’solution_length’]
- 38 if correct_lens:
- 39 # Check if predicted length is within range of correct answers
- 40 min_l, max_l = min(correct_lens), max(correct_lens)
- 41 length_score = int(min_l < p_len < max_l)
- 42
- 43 # --- 4. Calculate Accuracy Score ---
- 44 if has_diff:
- 45 avg_score = sum(ground_truth["score"]) / len(ground_truth["score"])
- 46 pred_score = solution_str["pass_rate"] / 8.0
- 47 acc_score = (0.01) ** abs(avg_score - pred_score)
- 48
- 49 return (notion_score + acc_score + length_score) / 3

TokenLength

TokenLength

Figure 8.

### D. Length Prediction and Training Dynamics

Similar to the observations on the difficulty prediction and training dynamics, we observe a surge in predicted length from the initial incorrect and underestimated state coincides with the rapid gain in performance during the training phase. This observation, coupled with the similar tendency in difficulty estimation, implies that calibration in the model’s meta-awareness induces performance gain in reinforcement learning.

### E. Shapley R2 Computation Details

We first fit a linear model using all p features to obtain the full-model coefficient of determination Rfull2 . To compute feature-level contributions, we consider all permutations of the feature set. For each permutation, features are added sequentially to the model, and the marginal increase in R2 upon adding feature j is recorded. The Shapley contribution of feature j is then defined as the average of its marginal R2 gains over all permutations. This decomposition yields an additive attribution of Rfull2 , providing a principled measure of each factor’s explanatory power while accounting for feature interactions and ordering effects.

#### F. Discussions Table 9. Examples of cross-domain notion prediction on coding and science tasks.

Domain Task Extracted Notions

maximum-strength-of-a-group array manipulation, mathematical operations, dynamic programming, greedy algorithms

Coding

find-the-longest-equal-subarray sliding window, hash map, two pointers

greatest-common-divisor-traversal graph traversal, GCD calculation, prime factorization

Quantum mechanics problem quantum mechanics, Heisenberg uncertainty principle

Science

Organic chemistry synthesis organic chemistry, Grignard reactions, oxidation, reaction mechanisms

Gene interaction problem epistasis, transcription factor, gene redundancy

Our notion-based reward formulation is not restricted to mathematical reasoning and can naturally generalize to other domains such as coding and science question answering. In coding tasks, notions correspond to high-level algorithmic concepts and data structure patterns, while in scientific reasoning they map to domain-specific scientific principles and terminology. To verify this, we apply a simple prompt adaptation without additional domain-specific training and analyze the generated notion predictions across domains. As shown in Table 9, the model consistently extracts meaningful and task-relevant notions for both coding and science problems, suggesting that notion prediction captures transferable high-level semantic abstractions beyond mathematics.

