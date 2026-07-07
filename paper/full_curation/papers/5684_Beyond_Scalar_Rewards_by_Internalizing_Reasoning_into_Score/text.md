# arXiv:2606.09076v2[cs.CV]14Jun2026

-Image

## Beyond Scalar Rewards by Internalizing Reasoning into Score Distributions

###### Xin Jin1,2,∗ Huanqia Cai1,∗,† Zhen Li1 Zechao Zhan1 Dengyang Jiang1 Aiming Hao1 Yuming Jiang1 Chunle Guo2,‡ Peng Gao1,‡ Ming-Ming Cheng2 Steven C.H. Hoi1 1Z-Image Team, Alibaba Group 2VCIP, CS, Nankai University ∗Equal contribution †Project lead ‡Corresponding authors

Project page: https://srameo.github.io/projects/z-reward/

#### Abstract

Reward models are central to text-to-image post-training, but visual preference is subjective and better represented as a distribution over rubric scores than as a deterministic scalar. Existing scalar, score-token, and pairwise reward models over-compress uncertainty and fine-grained score differences, while reasoning-based generative rewards provide stronger judgments but are costly to deploy and difficult to use as direct optimization signals. We propose Z-Reward, a teacher-student reward modeling framework that decouples reasoning-heavy judgment from efficient reward deployment. The teacher is a large VLM that uses reasoning to infer rubric-aligned score distributions, and is trained with Group-wise Direct Score Optimization (GDSO), which combines policy-gradient rewards from distribution expectations with direct pointwise and pairwise supervision on score distributions and score gaps. The student is trained with Reasoning-Internalized Score Distillation (RISD), which transfers the teacher’s reasoning-conditioned score distribution into a compact VLM without requiring explicit reasoning chains at inference time. On our internally annotated evaluation set, the 27B GDSO teacher reaches 89.6% human preference accuracy, outperforming SFT, RewardDance, and GRPO, while the 9B RISD student reaches 88.6%, outperforming the OPD baseline and closely matching the larger teacher. We further show that Z-Reward can serve as a differentiable reward signal for text-to-image optimization, yielding a 41.3% net human-preference improvement over SFT baseline.

#### 1 Introduction

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | |27B-GDSO 9B-RISD| |
| | | | |27B-GRPO 7B-Rewar|dDance|
| | | | |27B-SFT| |
| | | | |9B-SFT| |

| | | | |89<br><br>Z-R|.6<br><br>88<br><br>eward (Final|.6<br><br>)|
|---|---|---|---|---|---|---|
| | |84|.2<br><br>86|.0<br><br>84|.0| |
| | | | | | | |
| |81|.3| | |27B-to-9B O|PD|
|74|.4 74|.6<br><br>78|.2<br><br>77|.0| | |
| | | | | | | |
|65|.6| | | |27|B|
| | | | | |9B| |

90

HumanPreferenceAccuracy(%)

90

HumanPreferenceAccuracy(%)

85

85

80

80

75

75

70

70

65

65

60

Zero-shot SFT RewardDance GRPO GDSO RISD Method

0 250 500 750 1000 1250 1500 Step

Figure 1: Human preference accuracy for teacher optimization and student distillation. Left: accuracy curves over training steps show how reward-model performance evolves against SFT and RewardDance [58] baselines. Right: final accuracy comparison shows that the 27B GDSO teacher outperforms SFT, RewardDance, and GRPO [47], while the 9B RISD student reaches comparable performance to the larger teacher.

Tech Report

- Table 1: Comparison of reward modeling paradigms for visual generation. Scalar or pairwise reward models are efficient but compress preference uncertainty, while reasoningbased generative reward models improve judgment quality at the cost of inference efficiency and direct differentiability. Z-Reward separates these roles: the teacher uses reasoning to infer score distributions, and the student internalizes this ability for efficient direct scoring and gradient backpropagation.

Modeling Paradigm

Support Gradient Backpropagation

Training Strategy

Scoring Based on Reasoning

Base Model

Score Distribution

Inference Efficiency

Methods

ImageReward [61] CLIP Regressive SFT × × ✓ High PickScore [27] CLIP Regressive SFT × × ✓ High

HPSv2 [59] CLIP Regressive SFT × × ✓ High VisionReward [60] VLM Regressive SFT × × ✓ High

VideoAlign [32] VLM Regressive SFT × × ✓ High

HPSv3 [36] VLM Regressive SFT × ✓ ✓ High WorldPM [54] VLM Regressive RL × × ✓ High

DeepSeek-GRM [34] VLM Generative SFT ✓ ✓ × Low Pairwise RM [33] VLM Generative RL × ✓ ✓ High GenRM-CoT [65] VLM Generative SFT ✓ ✓ ✓ Low

Edit-R1 [17] VLM Generative RL ✓ × × Low

UnifiedReward [55] VLM Generative SFT ✓ × × Low RewardDance [58] VLM Generative SFT × ✓ ✓ High Z-Reward-Teacher VLM Generative RL & SFT ✓ ✓ ✓ Low Z-Reward-Student VLM Generative Distillation Internalized ✓ ✓ High

Reward models are a key component of post-training, where they provide the preference signals used for model selection, data curation, and reward-guided optimization [39, 6, 61, 60, 58]. Unlike mathematics or coding rewards, however, visual preferences are inherently subjective: the same generated image can receive different judgments from different annotators, especially for aesthetics, realism, and fine-grained prompt alignment. Thus, human evaluation for visual generation is better viewed as a distribution of judgments rather than a deterministic scalar score [37, 50, 57, 64, 36, 52, 9].

As summarized in Table 1, existing reward modeling paradigms each miss part of this requirement. Scalar, score-token, and pairwise reward models compress preference into a single value or comparison, which is efficient but discards annotator uncertainty and finegrained differences among plausible scores [52, 9]. For example, two images may both collapse to score 4 under a discrete scoring scheme, even though one is slightly below the 4-point boundary and the other is slightly above it. Reasoning-based generative reward models can produce higher-quality judgments by leveraging world knowledge and explicit rationales [68, 15, 4, 5], but they are expensive at inference time and their textual reasoning or score outputs are less suitable for large-scale deployment and gradient-based optimization. Explicit distribution modeling [50, 10, 56, 64] can represent uncertainty more directly, but it typically relies on repeated annotations per sample, which is difficult to scale in production pipelines.

This creates a central tension for visual reward modeling: high-quality scoring requires reasoning and uncertainty awareness, while scalable post-training requires fast, direct, and differentiable reward signals. Building on knowledge distillation and sequence/rationale distillation [20, 26, 21, 16], recent on-policy distillation (OPD) methods [35, 67, 12, 49, 29, 8] improve compact students by applying teacher-guided dense feedback to their own sampled reasoning trajectories, making on-policy reasoning distillation an increasingly important paradigm. For reward modeling, however, the deployment target is different: a reward model is expected to provide fast, stable, calibrated, and optimization-friendly scores, rather than expose long reasoning chains at inference time. Our key insight is that reward models do not need to reproduce how a teacher reasons; they need to reproduce how a reasoning teacher judges. Therefore, Z-Reward resolves this tension by decoupling judgment quality from reward efficiency: instead of forcing the student to imitate the sequential process of reasoning, Z-Reward allows the compact model to internalize the teacher’s reasoning-conditioned judgment directly into score distributions.

We propose Z-Reward, a teacher-student framework for reasoning-internalized score distributions, as illustrated in figure 2. The teacher is a large VLM that uses reasoning and world knowledge to infer a calibrated score distribution from scalable supervision. Here,

DEQA -like

RewardDance -like

Z-Reward

[64] [58]

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Prompt⋯ Image⋯ 1 2 3 4 5 annotator

[Figure 5]

[Figure 6]

[Figure 7]

⋯

⋯

⋯

⋯

Direct Supervision

1 2 3 4 5

1 2 3 4 5 annotator

Prompt

Image

Prompt

Image

A bunch of annotators

<thinking> 1 2 3 4 5

Large VLM

⋯

⋯

###### Direct Supervision

Direct Supervision

Self-explored

Prompt

Image

Teacher Model

Distillation

1 2 3 4 5 <thinking>

VLM

VLM

⋯

⋯

⋯

⋯

1 2 3 4 5 Reward Model

distilled

Prompt

Image

Prompt

Image

Small VLM 1 2 3 4 5

⋯

⋯

Reward Model

Prompt

Image

Student Model

[Figure 8]

[Figure 9]

[Figure 10]

heavy annotation cost

Scoring is not based on reasoning

Reasoning-internalized and Efficient

- Figure 2: Overview of Z-Reward compared with existing distributional reward modeling paradigms. Left: DEQA [64] rely on dense human score distributions for direct supervision, leading to heavy annotation cost. Middle: RewardDance [58] learn score distributions from direct supervision, but their scoring is not explicitly based on reasoning. Right: Our Z-Reward first trains a reasoning-based large VLM teacher to infer calibrated score distributions, then distills this reasoning-enhanced distribution into a compact student that directly outputs scores without generating reasoning chains, enabling efficient deployment and gradient-based optimization.

reasoning is not used merely as an explanation artifact; it helps the teacher decompose visual evidence, apply rubric criteria, and allocate probability mass across neighboring score bins. The student is a compact reward model that internalizes this reasoning-enhanced distribution and directly predicts scores without generating reasoning chains at inference time, enabling efficient deployment and gradient backpropagation.

To train the teacher, we introduce Group-wise Direct Score Optimization (GDSO), which optimizes rewards computed from predicted score distributions and applies direct distribution-level supervision. Rather than requiring repeated human annotations to observe this distribution explicitly, we learn it as a latent, reasoning-conditioned distribution from scalable rubric-based supervision. To train the student, we further introduce Reasoning-Internalized Score Distillation (RISD), which distills the teacher’s reasoningconditioned score distribution into a small VLM without explicit reasoning tokens. Thus, the student does not imitate the teacher’s reasoning text; it internalizes the distributional effect of that reasoning into direct scoring behavior.

Empirically, this design leads to strong reward-modeling performance. As shown in Figure 1, our 27B GDSO teacher reaches 89.6% human preference accuracy, outperforming SFT, RewardDance-style supervision [58], and GRPO optimization [47]. More importantly, the 9B RISD student reaches 88.6%, outperforming the OPD student while closely matching the larger reasoning teacher, and remains efficient at inference time. We further validate Z-Reward as an optimizable reward signal by applying it to text-to-image post-training, where reward-guided optimization improves human preference over the SFT baseline.

Our contributions are summarized as follows:

- • We propose a reasoning-aware and uncertainty-aware teacher-student reward modeling framework that learns latent score distributions from scalable supervision.
- • We introduce Group-wise Direct Score Optimization, which trains a reasoningbased VLM teacher by optimizing score distributions directly.
- • We develop Reasoning-Internalized Score Distillation, allowing a compact student to internalize reasoning into efficient, direct, and differentiable scoring.
- • Empirically, our 27B teacher substantially improves human preference accuracy over SFT, GRPO, and RewardDance, while the 9B student outperforms an OPDbased distillation baseline, nearly matches the teacher, and serves as an efficient reward signal for text-to-image optimization.

#### 2 Annotation and Datasets

Annotation document. We build the annotation document around four user- and production-critical dimensions: Text–Image Alignment, Realism, Aesthetics, and Physical Plausibility, following recent fine-grained and multi-dimensional human-feedback settings for text-to-image generation [30, 66]. Each dimension is scored with a five-level rubric that specifies how different error patterns should affect the score, rather than relying only on abstract terms such as “minor” or “major.” Although the rubric is organized around five integer-level anchors, final annotations are recorded on a nine-level half-point scale, i.e., 𝑠ˆ ∈ {1.0,1.5, . . . ,5.0}. This half-point annotation scheme allows annotators to capture fine-grained quality differences between samples that fall into the same coarse rubric bin. For example, a score of 4 corresponds to one or two subtle defects, while a score of 3 indicates more salient subject-level errors or clearly visible quality degradation. To make these criteria operational, each score bin in each dimension is paired with 15–20 annotated examples, allowing annotators to calibrate new samples through a nearest-neighbor-style comparison. The document is updated throughout annotation by adding newly discovered corner cases and replacing less representative examples with more typical ones.

Data for annotation and evaluation. Our annotation prompts come from three sources: 1) internal captions rewritten as generation prompts; 2) real-world prompts from users or community usage; and 3) concepts sampled from our topology, composed, and LLMexpanded into diverse prompts, covering compositional phenomena emphasized by T2I evaluation benchmarks [22, 14, 23, 45]. For evaluation, we construct a held-out test set with multiple annotations per sample. To compute the ground-truth score distribution, we drop the highest and lowest scores before aggregation to reduce outliers and stabilize preference estimates [52, 9].

Annotation workflow. As shown in figure 3, annotators 1) assign pointwise scores to generated candidates according to the rubric and example document, 2) compare candidates under the same prompt and within the same coarse score bin to shift distinguishable samples by ±0.5, and 3) submit the results to quality-control annotators for final review. Only data from annotators whose audited accuracy exceeds a preset threshold is admitted into the training set.

The risk of context mismatch. This process exposes two context mismatches between annotators and reward models: 1) the full annotation document is too long to place into a reward model’s context, since four dimensions, five score bins, and 15 images per bin already require approximately 4 × 5 × 15 × 1024 = 307,200 image tokens before counting textual instructions; and 2) annotators can compare same-prompt candidates during score adjustment, while a deployable pointwise reward model only observes one text–image pair at a time. These mismatches motivate direct supervision on scores and score distributions, so the model can learn human-calibrated scoring behavior without relying on the full annotation context at inference time.

两个⼩提琴在拿着琴弦击剑…

两个⼩提琴在拿着琴弦击剑…

两个⼩提琴在拿着琴弦击剑…

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Image A seems better than Image C, so we adjust the score

Instruct Follow: -

Instruct Follow: 3

Instruct Follow: 3

###### Based on the annotation document, two 3 one 5.

- A
- B
- C

- A
- B
- C

- A
- B
- C

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Instruct Follow: -

Instruct Follow: 5

Instruct Follow: 5

[Figure 21]

[Figure 22]

[Figure 23]

Pointwise Annotation

Score Adjustment

Instruct Follow: -

Instruct Follow: 3

Instruct Follow: 2.5

Prompt & Generated Candidates

Prompt & Generated Candidates

Prompt & Generated Candidates

Quality Check

- Figure 3: Annotation workflow. For each prompt, annotators 1) assign pointwise scores to generated candidates according to the annotation document, 2) compare candidates under the same prompt to refine scores within the same coarse bin, and 3) send the resulting annotations to quality check before they are admitted into the training set.

#### 3 Method

The annotation process described above provides calibrated human scores, but the calibration context available to annotators cannot be directly supplied to a deployable reward model. Annotators can consult a long, evolving document and compare candidates under the same prompt, whereas a pointwise reward model must usually judge one text–image pair at a time. We therefore decouple reward-model training into a reasoning-intensive teacher stage and an efficient student stage. The teacher learns a reliable reasoningaugmented score distribution, while the student internalizes this distribution into a compact model for direct scoring and gradient backpropagation.

Given a prompt 𝑝, an image 𝐼, and a reward dimension 𝑑 ∈ D, the teacher first generates a reasoning trace 𝜌 and then predicts a distribution over rubric-aligned score bins 𝑠 ∈ S:

𝑞𝜃(𝑠 | 𝑝, 𝐼, 𝑑, 𝜌), 𝑠 ∈ S. (1)

We treat 𝑞𝜃 as a predictive distribution over rubric-aligned score bins, rather than as a directly observed empirical annotator distribution for each training sample. Since scalable annotation usually provides one rubric-calibrated score per text–image pair, the distribution is learned implicitly from large-scale score supervision, same-prompt score-gap constraints, and the teacher’s reasoning-conditioned score-token probabilities. This follows the ordinal and soft-label view that neighboring score bins should carry structured uncertainty rather than being treated as unrelated classes [10, 56]. Thus, 𝑞𝜃 captures the model’s uncertainty over plausible neighboring bins while its expectation is calibrated to human rubric scores.

We decode this distribution from the teacher’s score tokens using a Q-Align-style [57] score decoder. The expected scalar score is then obtained from the decoded distribution:

𝜇𝜃(𝑝, 𝐼, 𝑑, 𝜌) = ∑︁ 𝑠∈S

𝑠 𝑞𝜃(𝑠 | 𝑝, 𝐼, 𝑑, 𝜌). (2)

These reasoning traces help decompose visual evidence, apply fine-grained rubric rules, and handle corner cases that would otherwise be compressed into one-hot labels. However, explicit reasoning is expensive and unsuitable for deployment. We therefore train a compact student to predict the teacher’s reasoning-conditioned distribution directly as 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑), without producing reasoning chains at inference time. The following subsections describe teacher optimization and student distillation separately.

###### 3.1 Training Teacher Model via Group-wise Direct Score Optimization

We begin from Group Relative Policy Optimization (GRPO) [47], which samples a group of responses for the same input and optimizes the policy using group-normalized advan-

tages. Given an input 𝑥, sampled responses {𝑜𝑖}𝐺𝑖=1, and rewards {𝑟𝑖}𝐺𝑖=1, the advantage of each response is normalized within the group:

𝑟𝑖 − mean({𝑟𝑘}𝐺𝑘=1) std({𝑟𝑘}𝐺𝑘=1) + 𝜖

. (3)

𝐴𝑖 =

The GRPO objective combines a policy-gradient term with KL regularization to a reference policy:

𝑖∼𝜋𝜃(·|𝑥) 𝐴𝑖 ∑︁

log𝜋𝜃(𝑜𝑖,𝑡 | 𝑜𝑖,<𝑡, 𝑥) − 𝛽𝐷KL(𝜋𝜃(· | 𝑥) ∥ 𝜋ref(· | 𝑥)) . (4)

LGRPO = −E𝑜

𝑡

GRPO alone treats the parsed score as a scalar reward, which can be slow to calibrate under the context mismatch discussed above. We therefore introduce Group-wise Direct Score Optimization (GDSO), which augments policy-gradient optimization with direct supervised gradients on score distributions and score gaps. Each training instance contains a winning sample 𝑥𝑤 = (𝑝, 𝐼𝑤, 𝑑) and a losing sample 𝑥𝑙 = (𝑝, 𝐼𝑙, 𝑑) with ground-truth rubric

- Algorithm 1 Iterative Group-wise Direct Score Optimization (GDSO)

Input initial teacher policy 𝜋𝜃init; annotated preference data Dann score bins S; score range [𝑆min,𝑆max]; group size 𝐺 hyperparameters 𝛽,𝜖, 𝜆pt, 𝜆pw,𝛼pt,𝛼pw

- 1: policy model 𝜋𝜃 ← 𝜋𝜃init
- 2: for iteration = 1, . . . , 𝑁 do
- 3: reference model 𝜋ref ← 𝜋𝜃
- 4: for step = 1, . . . , 𝑀 do
- 5: Sample a batch B from Dann
- 6: Update the old policy model 𝜋𝜃old ← 𝜋𝜃
- 7: for all (𝑝, 𝐼𝑤, 𝐼𝑙, 𝑠ˆ𝑤, 𝑠ˆ𝑙, 𝑑) ∈ B do
- 8: Set 𝑥𝑤 = (𝑝, 𝐼𝑤, 𝑑) and 𝑥𝑙 = (𝑝, 𝐼𝑙, 𝑑)
- 9: for 𝑗 ∈ {𝑤, 𝑙} do
- 10: Sample 𝐺 outputs {𝑜𝑗,𝑖}𝐺𝑖=1 ∼ 𝜋𝜃old(· | 𝑥𝑗)
- 11: Decode 𝜌𝑗,𝑖, 𝑞𝑗,𝑖(𝑠), and 𝜇𝑗,𝑖 from each 𝑜𝑗,𝑖
- 12: end for
- 13: Compute rewards 𝑟pt𝑗,𝑖, 𝑟pw𝑗,𝑖 , and 𝑟𝑗,𝑖 using Eqs. (7), (10), and (13)
- 14: Compute group-relative advantages 𝐴𝑗,𝑖 using Eq. (3)
- 15: Compute direct losses LCEpt and Lpw using Eqs. (8) and (11)
- 16: end for
- 17: for GDSO update = 1, . . . , 𝐾gdso do
- 18: Update 𝜋𝜃 by minimizing LGDSO in Eq. (14)
- 19: end for
- 20: end for
- 21: end for Output optimized teacher policy 𝜋𝜃

scores 𝑠ˆ𝑤 and 𝑠ˆ𝑙. For each side 𝑗 ∈ {𝑤, 𝑙}, the teacher samples 𝐺 reasoning-and-score outputs 𝑜𝑗,𝑖 = (𝜌𝑗,𝑖, 𝑎𝑗,𝑖):

𝑜𝑗,𝑖 ∼ 𝜋𝜃(· | 𝑥𝑗), 𝑖 = 1, . . . ,𝐺. (5) Following the Q-Align-style score decoder introduced above, each output is converted into a predicted score distribution 𝑞𝑗,𝑖(𝑠) = 𝑞𝜃(𝑠 | 𝑥𝑗, 𝜌𝑗,𝑖) and an expected score:

###### 𝜇𝑗,𝑖 = ∑︁ 𝑠∈S

𝑠 𝑞𝑗,𝑖(𝑠). (6)

Unlike most generative reward methods that parse a single textual score from the model response and then treat the parsed value as the reward, GDSO treats the decoded score distribution as the optimization target. Rewards are computed from the expectation of 𝑞𝑗,𝑖, while direct losses supervise the score-bin distribution and its induced score gaps.

Importantly, GDSO is group-wise not only in the GRPO-style advantage normalization, but also in its direct score supervision. For each candidate, pointwise supervision is applied to all 𝐺 sampled score distributions in its group. For each same-prompt candidate pair, pairwise supervision is applied across all 𝐺 × 𝐺 cross-side sampled output pairs. Thus, the sampled group provides multiple reasoning-conditioned distributional views for both policy optimization and direct score calibration.

Pointwise score supervision. For each sampled output, the pointwise reward measures how close its decoded expected score is to the annotated score:

𝜇𝑗,𝑖 − 𝑠ˆ𝑗 𝑆max − 𝑆min

𝑟pt𝑗,𝑖 = 1 −

. (7)

As a policy reward, this term favors outputs whose decoded scores stay close to the rubriccalibrated human score, encouraging the teacher to learn an absolute score scale rather than only a relative preference direction. In addition to using this value as a policy-gradient

reward, we directly supervise the decoded score distribution with a cross-entropy loss:

∑︁

###### ∑︁𝐺

- 1

- 2𝐺

LCEpt = −

log𝑞𝑗,𝑖(𝑠ˆ𝑗). (8)

𝑖=1

𝑗∈{𝑤,𝑙}

This supervised loss anchors the score-bin probability to the annotated bin, boosting scorescale convergence so the policy does not need to discover the scoring convention only through sampled rewards. The soft distribution around ordinal score bins also provides a more informative target than a one-hot nominal label [10, 56].

Pairwise score-gap supervision. Pointwise supervision calibrates absolute scores, while pairwise supervision preserves the relative score gap between samples under the same prompt. Let 𝑤¯ = 𝑙, 𝑙¯ = 𝑤, and

Δ𝑠ˆ𝑗,¯𝑗 = 𝑠ˆ𝑗 − 𝑠ˆ¯𝑗. (9)

For a sampled output 𝑜𝑗,𝑖, the pairwise reward compares its score gap to every sampled output from the opposite side:

###### ∑︁𝐺

1 𝐺(𝑆max − 𝑆min)

𝑟pw𝑗,𝑖 = 1 −

(𝜇𝑗,𝑖 − 𝜇¯𝑗,𝑘) − Δ𝑠ˆ𝑗,¯𝑗 . (10)

𝑘=1

As a policy reward, this term favors outputs whose score differences match the annotated gap across same-prompt candidates, encouraging the teacher to learn both the preference direction and the magnitude of visual quality differences. The corresponding direct pairwise loss is

###### ∑︁

###### ∑︁𝐺

###### ∑︁𝐺

1 2𝐺2(𝑆max − 𝑆min)

Lpw =

(𝜇𝑗,𝑖 − 𝜇¯𝑗,𝑘) − Δ𝑠ˆ𝑗,¯𝑗 . (11)

𝑖=1

𝑘=1

𝑗∈{𝑤,𝑙}

This supervised gap loss boosts within-prompt discrimination while keeping score margins calibrated to the annotation scale, rather than allowing the policy to separate pairs with arbitrary large margins.

We use score-gap supervision instead of a Bradley–Terry objective [3] or a binary preference-optimization objective such as DPO [44]. A Bradley–Terry model estimates the preference probability as

𝑃(𝑥𝑤 ≻ 𝑥𝑙) = 𝜎(𝜇𝑤 − 𝜇𝑙), LBT = −log𝜎(𝜇𝑤 − 𝜇𝑙). (12)

This objective only requires the winner score to exceed the loser score and can keep enlarging the margin, even when both absolute scores should stay close to the annotation rubric. In contrast, our pairwise term matches the annotated score gap, making it consistent with pointwise calibration.

The final GDSO reward combines pointwise and pairwise rewards:

𝑟𝑗,𝑖 = 𝜆pt𝑟pt𝑗,𝑖 + 𝜆pw𝑟pw𝑗,𝑖 . (13) The overall teacher-training objective is

LGDSO = LGRPO({𝑟𝑗,𝑖}) + 𝛼ptLCEpt + 𝛼pwLpw. (14)

Thus, GDSO does not rely on policy-gradient reward alone: the score distribution and score gap both receive supervised gradients, which accelerates score-scale calibration and score-distribution convergence. Algorithm 1 summarizes GDSO in a GRPO-style iterative procedure.

###### 3.2 Teaching Student Model via Reasoning-Internalized Score Distillation

After teacher training, the large reasoning model generates a reasoning trace 𝜌𝑇 and produces a calibrated distribution 𝑞𝑇(𝑠 | 𝑝, 𝐼, 𝑑, 𝜌𝑇) for each text–image pair and reward dimension. The student model 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑) is trained to predict this distribution directly, without generating the teacher’s reasoning chain. Unlike sequence-level or rationale distillation, which transfers generated trajectories or explanatory traces [26, 21], RISD uses the

- Algorithm 2 Reasoning-Internalized Score Distillation (RISD)

Input trained teacher policy 𝜋𝜃𝑇; distillation data Ddist = {(𝑝, 𝐼, 𝑑)}; initial student 𝑞𝜙init

score bins S; hyperparameters for student optimization

- 1: student reward model 𝑞𝜙 ← 𝑞𝜙init
- 2: Freeze the teacher policy 𝜋𝜃𝑇
- 3: for step = 1, . . . , 𝑀dist do
- 4: Sample a batch B from Ddist
- 5: for all (𝑝, 𝐼, 𝑑) ∈ B do
- 6: Query 𝜋𝜃𝑇 to generate reasoning 𝜌𝑇 and decode 𝑞𝑇(𝑠 | 𝑝, 𝐼, 𝑑, 𝜌𝑇)
- 7: Predict student distribution 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑) without reasoning tokens
- 8: end for
- 9: Update 𝑞𝜙 by minimizing LRISD in Eq. (15)
- 10: end for Output deployable student 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑) and score 𝜇𝜙(𝑝, 𝐼, 𝑑) in Eq. (16)

teacher’s reasoning-conditioned score distribution as a soft target in the spirit of knowledge distillation [20]. We distill the teacher distribution with a KL loss:

LRISD = E(𝑝,𝐼,𝑑) 𝐷KL 𝑞𝑇(𝑠 | 𝑝, 𝐼, 𝑑, 𝜌𝑇) ∥ 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑) . (15) The student score used for deployment is the expectation of the distilled distribution:

###### 𝜇𝜙(𝑝, 𝐼, 𝑑) = ∑︁ 𝑠∈S

𝑠 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑). (16)

Algorithm 2 summarizes the RISD distillation procedure. By internalizing the teacher’s reasoning-conditioned distribution, the student preserves much of the teacher’s rewardmodeling ability while avoiding explicit reasoning at inference time. This yields a compact reward model that supports efficient pointwise scoring and differentiable reward-guided optimization.

#### 4 Experiment

###### 4.1 Experimental Setup

Model choices. We instantiate the teacher with Qwen3.5-27B [42] and the student with Qwen3.5-9B [42]. This setting follows the design goal of Z-Reward: the larger teacher provides stronger reasoning and score-distribution estimation, while the smaller student is used to test whether the reasoning-conditioned distribution can be internalized into an efficient reward model.

Evaluation data and metrics. We evaluate all reward models on the held-out test set from section 2. We report PLCC and SRCC for score calibration, and human preference accuracy (HPA) and margin HPA for preference ranking. Margin HPA uses only pairs with a human score gap above 0.5.

For preference-ranking evaluation, we compute HPA over same-prompt candidate pairs with non-tied human scores. Given a pair (𝐼𝑎, 𝐼𝑏), HPA is defined as

∑︁

1 |P|

HPA =

[(𝜇𝑎 − 𝜇𝑏)(𝑠ˆ𝑎 − 𝑠ˆ𝑏) > 0] , (17)

(𝑎,𝑏)∈P

where P denotes all evaluated pairs with 𝑠ˆ𝑎 ≠ 𝑠ˆ𝑏, 𝑠ˆ is the aggregated human score, and 𝜇 is the reward model’s predicted expected score. Margin HPA is computed on the subset

satisfying |𝑠ˆ𝑎 − 𝑠ˆ𝑏| > 0.5.

Compared methods. We first include a zero-shot baseline, which evaluates the base model before SFT using only the scoring system prompt. We then compare against standard SFT, which fine-tunes the same backbone on annotated score outputs without reasoning chains and serves as the no-reasoning baseline for teacher-side comparison; RewardDance [58],

- Table 2: Reward-model evaluation on the internally annotated test set. We compare score calibration and preference-ranking quality using PLCC, SRCC, human preference accuracy, and margin human preference accuracy. Values in parentheses report absolute gains over the zero-shot baseline within the same model size. The best and second-best results within each model size are highlighted. Margin human preference accuracy is computed only on pairs whose human score gap is larger than 0.5.

Margin Human Preference Accuracy Qwen3.5-27B

Scoring Based on Reasoning

Human Preference Accuracy

Method

PLCC SRCC

Zero-shot × 0.6301 (+.0000) 0.5816 (+.0000) 0.7438 (+.0000) 0.9538 (+.0000) SFT × 0.6458 (+.0157) 0.5914 (+.0098) 0.8135 (+.0697) 0.9644 (+.0106) RewardDance × 0.6667 (+.0366) 0.6207 (+.0391) 0.8425 (+.0987) 0.9706 (+.0168) GRPO ✓ 0.7200 (+.0899) 0.6832 (+.1016) 0.8604 (+.1166) 0.9827 (+.0289) GDSO ✓ 0.7620 (+.1319) 0.7132 (+.1316) 0.8956 (+.1518) 0.9885 (+.0347)

Qwen3.5-9B

Zero-shot × 0.3411 (+.0000) 0.3167 (+.0000) 0.6563 (+.0000) 0.7501 (+.0000) SFT × 0.5296 (+.1885) 0.4942 (+.1775) 0.7459 (+.0896) 0.8401 (+.0900) RewardDance × 0.5182 (+.1771) 0.4338 (+.1171) 0.7817 (+.1254) 0.8972 (+.1471) GRPO ✓ 0.5340 (+.1929) 0.5072 (+.1905) 0.7703 (+.1140) 0.9076 (+.1575) GDSO ✓ 0.6341 (+.2930) 0.5665 (+.2498) 0.8395 (+.1832) 0.9599 (+.2098) RISD Internalized 0.7391 (+.3980) 0.6882 (+.3715) 0.8864 (+.2301) 0.9801 (+.2300)

which uses post-hoc pseudo reasoning chains distilled from Qwen-3.6-Max; and GRPO, which computes rewards from the mean of the predicted score distribution as the final output score. We also evaluate our GDSO, which directly optimizes score distributions with pointwise and pairwise supervision. For a clean comparison of reinforcement learning strategies, all GRPO and GDSO runs start from the base model; their reasoning and scoring behaviors are driven only by the system prompt and learned through pure selfexploration. These comparisons are conducted for both 27B and 9B models. For the 9B setting, we additionally evaluate RISD, which distills the 27B reasoning-based score distribution into the smaller student model.

###### 4.2 Main Results

- Table 2 summarizes reward-model performance. On the 27B teacher, GDSO achieves the best results on all metrics, improving over GRPO in both score calibration (PLCC/SRCC) and pairwise preference accuracy. On the 9B model, RISD is also consistently best and reaches 0.8864 HPA and 0.9801 margin HPA, close to the 27B GDSO teacher. This suggests that the teacher’s reasoning-conditioned score distribution can be effectively internalized into a smaller direct-scoring model.

RewardDance shows a useful contrast on 9B: compared with SFT, it improves HPA from 0.7459 to 0.7817, but decreases PLCC from 0.5296 to 0.5182 and SRCC from 0.4942 to 0.4338. This suggests that post-hoc pseudo reasoning helps the small model recognize coarse pairwise preference directions, but does not guarantee calibrated rubric scores. GRPO also trails RewardDance on 9B HPA, likely because pure self-exploration is bounded by the weaker reasoning ability of the 9B model. In contrast, GDSO provides direct distribution and score-gap supervision during exploration, giving the policy clearer optimization directions, while RISD further uses KL supervision from the 27B teacher distribution to internalize fine-grained reasoning-based scoring behavior.

###### 4.3 Ablation Studies

Effect of decoding from score distribution instead of parsing text to compute rewards. We compare two ways of extracting rewards from a generative reward model. The first follows common generative reward modeling practice: parse the final textual score from the model response and use it to compute the RL reward. The second uses our score decoder to obtain the full score distribution and computes the reward from its expectation. All other training settings are kept the same for GRPO and GDSO.

100

100

90

HumanPreferenceAccuracy(%)

HumanPreferenceAccuracy(%)

85

95

95

85

80

90

90

80

MarginHPA(%)

MarginHPA(%)

85

85

75

75

80

80

70

75

75

70

70

70

65

65

65

65

GRPO

GDSO

GRPO

GDSO

60

60

GRPO (Parsing Text)

GDSO (Parsing Text)

GRPO (Parsing Text)

GDSO (Parsing Text)

60

60

0 250 500 750 100012501500 Step

0 250 500 750 100012501500 Step

0 250 500 750 100012501500 Step

0 250 500 750 100012501500 Step

(a) GRPO HPA

(b) GRPO Margin HPA

(c) GDSO HPA

(d) GDSO Margin HPA

- Figure 4: Effect of reward computation from score distributions. We compare human preference accuracy and margin human preference accuracy when rewards are computed from decoded score distributions instead of parsed score text. “Parsing Text” denotes computing the reward from the score parsed from the generated text, rather than from the expectation of the score distribution.

As shown in figure 4, using the distribution expectation consistently improves both HPA and margin HPA for GRPO and GDSO. Parsing text scores effectively quantizes the reward signal: predictions such as 3.8 and 4.2 may both be emitted as the score token 4, so they receive the same reward and the same normalized advantage in GRPO. This removes finegrained scoring signals and slows reward-model learning. In contrast, the expectation over the score distribution preserves uncertainty across neighboring bins, providing denser supervision and better teaching the model how to score.

Distill Reasoning Chains from Teacher to Student via On-Policy Distillation. We compare RISD with a direct on-policy distillation (OPD) [35] baseline, following the per-token reverse-KL formulation used by Thinking Machines Lab and related reverse-KL distillation objectives for generative LMs [16]. In OPD, the 9B student first samples its own reasoningand-score trajectory 𝑦 = (𝑦1, . . . , 𝑦𝑇) ∼ 𝜋𝜙old(· | 𝑥) for an input 𝑥 = (𝑝, 𝐼, 𝑑), and the 27B GDSO teacher only provides token log-probabilities on the student’s visited prefixes. The per-token reverse-KL term is used as an on-policy advantage:

𝐴OPD𝑡 = log𝜋𝜃𝑇 (𝑦𝑡 | 𝑦<𝑡, 𝑥) − log𝜋𝜙old(𝑦𝑡 | 𝑦<𝑡, 𝑥). (18)

The student is then updated by an on-policy policy-gradient objective, with 𝐴OPD𝑡 treated as a stop-gradient advantage:

###### ∑︁𝑇

1

sg 𝐴OPD𝑡 log𝜋𝜙(𝑦𝑡 | 𝑦<𝑡, 𝑥) . (19)

LOPD = −E𝑥, 𝑦∼𝜋

𝜙old (·|𝑥)

𝑇

𝑡=1

- Table 3 shows that OPD improves over SFT and reaches a similar level to the 9B GDSO model, but it still does not approach the 27B teacher or the 9B RISD student. The output-token column further reveals an efficiency gap: OPD and GDSO require long autoregressive reasoning traces, averaging about 750 output tokens, while RISD returns the score in a single output token, matching SFT’s decoding cost. Since reward inference is repeatedly called during candidate filtering or optimization, this reduction directly lowers latency and serving cost. Thus, RISD does not merely improve HPA; it transfers the teacher’s reasoning benefit into a deployment-efficient outcome-level scorer. The remaining accuracy gap suggests that teacher-derived token advantages are not sufficient when the 9B student cannot explore strong reasoning trajectories by itself: OPD can reinforce better tokens on the student’s on-policy prefixes, but its learning signal is still

Table 3: Trajectory- vs. outcome-level distillation. HPA denotes human preference accuracy, and output tokens denote generated output length. Best and second-best results are highlighted.

Margin Output

Method HPA

HPA Tokens 9B OPD 0.8311 0.9643 ~750

9B SFT 0.7459 0.8401 1 9B GDSO 0.8395 0.9599 ~750

9B RISD 0.8864 0.9801 1

###### Text Image Alignment

4.4

4.2

RewardScore

4.0

3.8

3.6

3.4

3.2

0 2.5k 5k 7.5k 10k

RL Iteration

(a) Text–Image Alignment

###### Realism

4.2

RewardScore

4.0

3.8

3.6

3.4

0 2.5k 5k 7.5k 10k

RL Iteration

(b) Realism

###### Aesthetic

4.4

RewardScore

4.2

4.0

3.8

3.6

0 2.5k 5k 7.5k 10k

RL Iteration

(c) Aesthetics

###### Physical Plausibility

4.8

4.6

RewardScore

4.4

4.2

4.0

0 2.5k 5k 7.5k 10k

RL Iteration

(d) Physical Plausibility

- Figure 5: Validation reward trajectories during RL-based text-to-image optimization using Z-Reward. We report reward score trends over 10k RL iterations for four optimized dimensions: text–image alignment, realism, aesthetics, and physical plausibility. All scores are reported on the five-level rubric scale, where higher is better.

bounded by the states the student visits. OPD and GDSO almost reach a similar 9B ceiling by giving direct score-distribution and score-gap guidance during exploration. In contrast, RISD provides a finer-grained supervision signal by directly matching the teacher’s reasoning-conditioned score distribution over the score vocabulary. This allows the student to internalize the teacher’s scoring behavior without having to reproduce the full reasoning trajectory through on-policy exploration.

#### 5 Validating Z-Reward as an Optimizable Reward Signal

To demonstrate the practical utility of Z-Reward, we apply it to the Reinforcement Learning (RL) stage of text-to-image generation, a setting where prior work has explored policygradient fine-tuning, direct preference optimization, and differentiable reward backpropagation [11, 53, 41, 7]. Unlike traditional discrete scalar rewards that provide sparse guidance, the score distributions predicted by Z-Reward offer dense and informative gradient signals. We leverage these gradients to directly optimize the baseline SFT model [51], steering the generation toward human preferences.

Prompt: Focused young woman with elaborate auburn braided crown. Blue morpho butterfly perched on outstretched finger. Blurred foliage background. Close-up.

##### SFT RL

[Figure 24]

[Figure 25]

Prompt: A cozy living room with a framed picture that says 'Home Sweet Home' in elegant cursive, a cushion on the couch displaying 'Rest Now' in medium regular.

### SFT RL

[Figure 26]

[Figure 27]

- Figure 6: Qualitative comparisons between the SFT baseline and Z-Reward-guided optimization. Each row shows one held-out prompt and compares the baseline generation with the optimized model.

###### 5.1 Multi-Dimensional Reward Gradient Backpropagation

We adopt a ReFL-style [61] direct reward backpropagation scheme, extended to earlier denoising steps and multi-dimensional reward optimization, which is closely related to differentiable reward fine-tuning and dense reward views of the diffusion trajectory [7, 41, 62]. Given a prompt 𝑝 and a generated image 𝐼 = 𝐺𝜓(𝑝), the deployed student reward model predicts 𝑞𝜙(𝑠 | 𝑝, 𝐼, 𝑑) for each reward dimension 𝑑 ∈ D, where D includes text– image alignment, realism, aesthetics, and physical plausibility. We use the expected score 𝜇𝜙(𝑝, 𝐼, 𝑑) defined in Eq. 16 as the reward for dimension 𝑑, and aggregate the multidimensional rewards as:

𝑅(𝑝, 𝐼) = A {𝜇𝜙(𝑝, 𝐼, 𝑑)}𝑑∈D , (20)

where A(·) denotes a task-dependent aggregation function. We then backpropagate ∇𝜓𝑅(𝑝,𝐺𝜓(𝑝)) through the denoising process to update the generator.

###### 5.2 Reward Curve Analysis

Figure 5 shows that reward-guided optimization steadily improves validation rewards across text–image alignment, realism, aesthetics, and physical plausibility. Realism and aesthetics improve faster in the early stage, while text–image alignment and physical plausibility increase more gradually due to their stronger dependence on semantic and structural correctness. These trends suggest that Z-Reward provides stable and fine-grained optimization signals across multiple aspects of visual generation.

###### 5.3 Human Evaluation

To examine whether the reward improvements translate into human-perceived quality gains, we conduct blind human evaluation on the same held-out prompt set used in the validation reward analysis, following the broader emphasis on reproducible human evaluation for text-to-image generation [38]. The set contains 400 prompts covering compositional descriptions, attribute binding, spatial relations, and physically challenging scenes, matching the compositional coverage studied in recent T2I and text-to-visual evaluation benchmarks [22, 14, 23, 31, 28]. Professional annotators perform pairwise comparisons between images generated by the SFT baseline and those generated by the model optimized with Z-Reward.

We report results using the human-preference-based Good-Same-Bad (GSB) metric. For each prompt, annotators judge whether the optimized image is better than, comparable to, or worse than the baseline image, corresponding to Good, Same, and Bad, respectively. Let 𝐺, 𝑆, and 𝐵 denote the counts of these three outcomes. The final GSB score is defined as

𝐺 − 𝐵 𝐺 + 𝑆 + 𝐵

. (21) where a higher value indicates stronger net human preference for the optimized model.

𝐺𝑆𝐵 =

Compared with the strong SFT baseline, the model optimized using Z-Reward achieves a net GSB improvement of 41.3%. This result confirms that the improvements measured by our reward model are reflected in human judgments, which is important because rewardguided optimization can otherwise overfit proxy rewards [13, 43]. Qualitative comparisons in Figure 6 further illustrate that Z-Reward-guided optimization improves text-image alignment, visual realism, aesthetics, and physical plausibility across diverse prompts.

#### 6 Discussions and Future Works

Reasoning-score coupling. One limitation of the current teacher training objective is that it combines policy-gradient rewards with direct SFT-style losses, 𝛼ptLCEpt + 𝛼pwLpw in Eq. (14). These supervised terms substantially improve score calibration and preference metrics, but they may occasionally make the final score depend more on direct score supervision than on the generated reasoning trace itself. In our experiments, such weak coupling between reasoning and the final score appears to be a minority case, while the metric gains from these losses are clear. Future work could add explicit reasoning-score consistency checks or contrastive supervision so that the teacher preserves the calibration benefits of direct losses while keeping the score more tightly grounded in its rationale.

Potential Generalization to All Sequence-to-Score Tasks. Although this paper focuses on reward modeling for image generation, the proposed formulation is not tied to a specific visual domain. As VLMs continue to improve, a VLM-to-score model can naturally take image, video, or text-centered inputs and convert arbitrary model outputs into rubricaligned score distributions. In our framework, the score can be decoded during the reasoning process or from any scoring-oriented output of the VLM, and the distribution expectation remains directly supervised and differentiable. This makes the reward useful not only as an evaluator, but also as a continuous optimization signal that can be backpropagated to generated images or videos through a differentiable generator. Beyond visual generation, the same decoupled teacher-student design can serve as a reward model for LLMs and VLMs, or as a general sequence-to-score evaluator for tasks such as image/video

quality assessment and caption evaluation, connecting to broader evaluation lines such as reference-free caption metrics and fine-grained video-generation benchmarks [19, 24]. Our experiments instantiate this idea on text-to-image generation, while broader modalities and downstream reward-modeling settings remain promising future directions.

Possible Extension to Unified Reward Modeling. The same formulation also points toward unified reward modeling. Our annotation pipeline already provides more than isolated pointwise labels: when annotators score multiple candidates under the same prompt, their adjustments implicitly encode comparison signals among candidates. This enables comparison training, but with richer supervision than binary preferences. Because each candidate has a calibrated rubric score, the model can learn not only whether one sample is better than another, but also by how many score levels they differ. Such score-gap supervision is naturally compatible with our pointwise distribution objective and can be extended across dimensions, modalities, and task types. A future unified reward model could therefore combine pointwise score distributions, pairwise preferences, and calibrated score gaps within one teacher-student system, using the teacher for reasoning-heavy judgment and the student for efficient direct scoring.

#### 7 Related Work

###### 7.1 Reward Models

Reward models are widely used to align generative models with human preferences. Early visual reward models, such as ImageReward [61], PickScore [27], and HPSv2 [59], are mostly built on CLIP-style encoders and trained to output scalar preference scores. Recent VLM-based reward models further improve visual understanding and task coverage by replacing CLIP encoders with stronger multimodal backbones and attaching regressive reward heads, including VisionReward [60], VideoAlign [32], HPSv3 [36], and WorldPM [54]. Scalar or regressive reward models are efficient and convenient for deployment, but they can over-compress subjective preferences and may be vulnerable to reward hacking or reward overoptimization [64, 58, 13, 43]. This motivates reward modeling paradigms that preserve richer judgment information while remaining usable for optimization.

Generative reward models aim to better exploit the native next-token prediction and reasoning capabilities of VLMs, following the broader trend of LLM/VLM-as-a-judge systems [68, 15, 4, 5]. Representative works include DeepSeek-GRM [34], GenRM-CoT [65], UnifiedReward [55], RewardDance [58], and Edit-R1 [17]. RewardDance formulates reward prediction as a generative comparison task and studies scaling along model and context dimensions. Edit-R1 further shows that verifier-style reasoning, which decomposes editing instructions into explicit principles and verifies outputs with CoT, can provide stronger feedback for image editing. Orthogonally, score-distribution modeling has been explored in quality assessment and ordinal label learning [37, 50, 10, 56]. Q-Align [57] discretizes continuous scores into level tokens, while DeQA [64] shows that distributionbased soft labels better preserve uncertainty and inter-image relationships than one-hot labels. Different from these works, Z-Reward uses a reasoning VLM to infer rubric-aligned score distributions and further distills them into an efficient direct-scoring student, thereby combining reasoning-aware judgment with deployable distributional rewards.

###### 7.2 Reinforcement Learning for Visual Generation

Reinforcement learning from human feedback has been increasingly used to align visual generators with human preferences. Existing methods either adapt policy-gradient algorithms to diffusion or flow-based generators, such as DDPO [2], DPOK [11], and recent GRPO-style visual RL methods [47, 32, 58, 17], optimize diffusion models from pairwise preferences [44, 53], or directly backpropagate reward gradients through the sampling process, like ReFL [61], DRaFT [7], AlignProp [41], and dense reward formulations [62]. Related preference-optimization and online RL methods have further been explored for textto-image generation, video generation, and image editing [32, 58, 17]. These works show that reward-guided optimization can substantially improve visual generation quality, but

its effectiveness depends heavily on the reward signal. Scalar or regressive rewards are efficient and convenient for optimization, but they can over-compress subjective preferences and may be vulnerable to reward hacking [64, 17, 13, 43]. Reasoning-based rewards provide richer semantic verification, but explicit reasoning traces can introduce additional inference overhead or become incompatible with direct reward backpropagation [63, 17]. In contrast, Z-Reward distills reasoning-enhanced judgments into score distributions whose expectations provide dense and differentiable rewards, enabling efficient reward-guided optimization of text-to-image generators.

###### 7.3 On-policy Distillation

A practical reward model should be efficient enough for large-scale scoring and sufficiently stable for reward-guided optimization, since visual reward models are commonly used as automatic evaluators or optimization signals for improving generated samples [61, 58, 63]. On-policy distillation (OPD) has recently emerged as a relevant paradigm for transferring reasoning behaviors from stronger models to weaker ones. Instead of relying only on fixed offline trajectories, OPD trains the student on its own sampled trajectories and uses a teacher to provide dense supervision on the states visited by the student [1, 35]. This makes the learning signal better matched to the student’s inference-time distribution and is particularly relevant for long-horizon reasoning, where deviations from offline traces can accumulate over multiple steps. Recent studies have extended this idea to selfdistillation, privileged-information distillation, reasoning compression, continual learning, and reward-to-supervision conversion [67, 40, 48, 46, 18]. Follow-up work further analyzes the failure modes, stability issues, and practical design choices of OPD in reasoning distillation [49, 12, 29, 25, 69].

While OPD provides a natural reference for transferring a large reasoning teacher into a compact student, its trajectory-centric objective differs from the deployment goal of visual reward modeling: in reward-guided generation, the reward model is commonly used as an efficient scorer or a differentiable optimization signal for selecting or directly improving generated samples [61, 7, 63], whereas reasoning-based reward models that generate discrete multi-step traces can be costly or incompatible with direct reward backpropagation [17]. These observations suggest that directly applying OPD to distill reasoning trajectories is not the most natural objective for visual reward modeling: it teaches the student how the teacher reasons, while deployment only requires how the teacher judges. In contrast, Z-Reward distills the outcome of reasoning. The teacher first uses reasoning to produce a calibrated score distribution, and the student learns to predict this distribution directly. This reasoning-internalized distillation transfers the teacher’s judgment behavior while avoiding explicit reasoning at inference time.

#### 8 Conclusion

We presented Z-Reward, a teacher-student framework for visual reward modeling that represents human preference as a reasoning-conditioned score distribution rather than a single deterministic scalar. By training a large VLM teacher with Group-wise Direct Score Optimization, Z-Reward combines policy-gradient learning with direct supervision on score distributions and score gaps, improving both calibrated scoring and pairwise preference ranking. Through Reasoning-Internalized Score Distillation, a compact student internalizes the teacher’s reasoning-based distribution and provides efficient, direct, and differentiable scoring without generating explicit reasoning chains.

Experiments on our internally annotated benchmark show that the 27B GDSO teacher outperforms SFT, RewardDance, and GRPO, while the 9B RISD student closely matches the larger teacher and serves as an effective reward signal for text-to-image optimization. Beyond the current image-generation setting, we view Z-Reward as a general sequence-toscore modeling paradigm: future work can extend the same decoupled teacher-student design to unified reward modeling across image, video, text, and multimodal generation tasks, combining pointwise score distributions, pairwise comparisons, and calibrated score-gap supervision in a single reward model.

#### References

- [1] Agarwal, R., Vieillard, N., Zhou, Y., Stan´czyk, P., Ramos, S., Geist, M., Bachem, O.: On-policy distillation of language models: Learning from self-generated mistakes. In: International Conference on Learning Representations (2023), https:// api.semanticscholar.org/CorpusID:263610088
- [2] Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning (2024), https://arxiv.org/abs/2305.13301
- [3] Bradley, R.A., Terry, M.E.: Rank analysis of incomplete block designs the method of paired comparisons. Biometrika 39, 324–345 (1952), https://api. semanticscholar.org/CorpusID:121987403
- [4] Chen, D., Chen, R., Zhang, S., Wang, Y., Liu, Y., Zhou, H., Zhang, Q., Wan, Y., Zhou, P., Sun, L.: MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge with visionlanguage benchmark. In: Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 6562–6595. PMLR

(2024), https://proceedings.mlr.press/v235/chen24h.html

- [5] Chen, Z., Du, Y., Wen, Z., Zhou, Y., Cui, C., Weng, Z., Tu, H., Wang, C., Tong, Z., Huang, Q., Chen, C., Ye, Q., Zhu, Z., Zhang, Y., Zhou, J., Zhao, Z., Rafailov, R., Finn, C., Yao, H.: MJ-Bench: Is your multimodal reward model really a good judge for textto-image generation? (2024), https://arxiv.org/abs/2407.04842
- [6] Christiano, P.F., Leike, J., Brown, T.B., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences (2017), https://arxiv.org/abs/ 1706.03741
- [7] Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning diffusion models on differentiable rewards (2024), https://arxiv.org/abs/2309.17400
- [8] Cui, F., Li, S., Li, J.: A brief overview: On-policy self-distillation in large language models (2026), https://arxiv.org/abs/2605.18141
- [9] Davani, A.M., Díaz, M., Prabhakaran, V.: Dealing with disagreements: Looking beyond the majority vote in subjective annotations. Transactions of the Association for Computational Linguistics 10, 92–110 (2022). https://doi.org/10.1162/tacl_a_00449, https://aclanthology.org/2022.tacl-1.6
- [10] Diaz, R., Marathe, A.: Soft labels for ordinal regression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)
- [11] Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In: Advances in Neural Information Processing Systems. vol. 36 (2023)
- [12] Fu, Y., Huang, H., Jiang, K., Liu, J., Jiang, Z., Zhu, Y., Zhao, D.: Revisiting on-policy distillation: Empirical failure modes and simple fixes (2026), https://arxiv.org/ abs/2603.25562
- [13] Gao, L., Schulman, J., Hilton, J.: Scaling laws for reward model overoptimization. In: Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 10835–10866. PMLR (2023), https:// proceedings.mlr.press/v202/gao23h.html
- [14] Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. In: Advances in Neural Information Processing Systems. vol. 36 (2023)
- [15] Gu, J., Jiang, X., Shi, Z., Tan, H., Zhai, X., Xu, C., Li, W., Shen, Y., Ma, S., Liu, H., Wang, S., Zhang, K., Wang, Y., Gao, W., Ni, L., Guo, J.: A survey on llm-as-a-judge (2024), https://arxiv.org/abs/2411.15594

- [16] Gu, Y., Dong, L., Wei, F., Huang, M.: Minillm: Knowledge distillation of large language models. In: The Twelfth International Conference on Learning Representations

(2024), https://openreview.net/forum?id=5h0qf7IBZZ

- [17] Guo, H., Wu, J., Liu, J., Gao, Y., Ye, Z., Yuan, L., Wang, X., Yu, Y., Huang, W.: Leveraging verifier-based reinforcement learning in image editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 34343–34352

(2026)

- [18] He, Y., Kaur, S., Bhaskar, A., Yang, Y., Liu, J., Ri, N., Fowl, L., Panigrahi, A., Chen, D., Arora, S.: Self-distillation zero: Self-revision turns binary rewards into dense supervision (2026), https://arxiv.org/abs/2604.12002
- [19] Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: CLIPScore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 7514–7528. Association for Computational Linguistics (2021). https://doi.org/10.18653/v1/2021.emnlpmain.595, https://aclanthology.org/2021.emnlp-main.595
- [20] Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network (2015), https://arxiv.org/abs/1503.02531
- [21] Hsieh, C.Y., Li, C.L., Yeh, C.K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.Y., Pfister, T.: Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In: Findings of the Association for Computational Linguistics: ACL 2023. pp. 8003–8017. Association for Computational Linguistics (2023). https://doi.org/10.18653/v1/2023.findings-acl.507, https: //aclanthology.org/2023.findings-acl.507
- [22] Hu, Y., Liu, B., Kasai, J., Wang, Y., Ostendorf, M., Krishna, R., Smith, N.A.: Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 20406–20417 (October 2023)
- [23] Huang, K., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. In: Advances in Neural Information Processing Systems. vol. 36 (2023)
- [24] Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21807–21818

(2024)

- [25] Jang, I., Yeom, J., Yeo, J., Lim, H., Kim, T.: Stable on-policy distillation through adaptive target reformulation (2026), https://arxiv.org/abs/2601.07155
- [26] Kim, Y., Rush, A.M.: Sequence-level knowledge distillation. In: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. pp. 1317–1327. Association for Computational Linguistics, Austin, Texas (2016). https://doi.org/10.18653/v1/D16-1139, https://aclanthology.org/D16-1139
- [27] Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation (2023), https: //arxiv.org/abs/2305.01569
- [28] Li, B., Lin, Z., Pathak, D., Li, J., Fei, Y., Wu, K., Xia, X., Zhang, P., Neubig, G., Ramanan, D.: Evaluating and improving compositional text-to-visual generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 5290–5301 (June 2024)
- [29] Li, Y., Zuo, Y., He, B., Zhang, J., Xiao, C., Qian, C., Yu, T., ang Gao, H., Yang, W., Liu, Z., Ding, N.: Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe (2026), https://arxiv.org/abs/2604.13016

- [30] Liang, Y., He, J., Li, G., Li, P., Klimovskiy, A., Carolan, N., Sun, J., Pont-Tuset, J., Young, S., Yang, F., Ke, J., Dvijotham, K.D., Collins, K.M., Luo, Y., Li, Y., Kohlhoff, K.J., Ramachandran, D., Navalpakkam, V.: Rich human feedback for text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19401–19411 (June 2024)
- [31] Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., Ramanan, D.: Evaluating text-to-visual generation with image-to-text generation (2024), https://arxiv. org/abs/2404.01291
- [32] Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., Zheng, M., Wu, X., Wang, Q., Xia, M., Wang, X., Liu, X., Yang, F., Wan, P., Zhang, D., Gai, K., Yang, Y., Ouyang, W.: Improving video generation with human feedback (2025), https://arxiv.org/abs/2501.13918
- [33] Liu, Y., Yao, Z., Min, R., Cao, Y., Hou, L., Li, J.: Pairjudge rm: Perform best-of-n sampling with knockout tournament (2025), https://arxiv.org/abs/2501.13007
- [34] Liu, Z., Wang, P., Xu, R., Ma, S., Ruan, C., Li, P., Liu, Y., Wu, Y.: Inference-time scaling for generalist reward modeling (2025), https://arxiv.org/abs/2504.02495
- [35] Lu, K., Thinking Machines Lab: On-policy distillation. Thinking Machines Lab: Connectionism (2025), https://thinkingmachines.ai/blog/ on-policy-distillation, accessed: 2026-06-03
- [36] Ma, Y., Shui, Y., Wu, X., Sun, K., Li, H.: Hpsv3: Towards wide-spectrum human preference score (2025), https://arxiv.org/abs/2508.03789
- [37] Murray, N., Marchesotti, L., Perronnin, F.: Ava: A large-scale database for aesthetic visual analysis. In: 2012 IEEE Conference on Computer Vision and Pattern Recognition. pp. 2408–2415 (2012). https://doi.org/10.1109/CVPR.2012.6247954
- [38] Otani, M., Togashi, R., Sawai, Y., Ishigami, R., Nakashima, Y., Rahtu, E., Heikkilä, J., Satoh, S.: Toward verifiable and reproducible human evaluation for text-to-image generation (2023), https://arxiv.org/abs/2304.01816
- [39] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L.E., Simens, M., Askell, A., Welinder, P., Christiano, P.F., Leike, J., Lowe, R.J.: Training language models to follow instructions with human feedback (2022), https://arxiv.org/ abs/2203.02155
- [40] Penaloza, E., Vattikonda, D., Gontier, N., Lacoste, A., Charlin, L., Caccia, M.: Privileged information distillation for language models (2026), https://arxiv.org/ abs/2602.04942
- [41] Prabhudesai, M., Goyal, A., Pathak, D., Fragkiadaki, K.: Aligning text-to-image diffusion models with reward backpropagation. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/forum?id= Vaf4sIrRUC
- [42] Qwen Team: Qwen3.5: Towards native multimodal agents (February 2026), https: //qwen.ai/blog?id=qwen3.5
- [43] Rafailov, R., Chittepu, Y., Park, R., Sikchi, H., Hejna, J., Knox, W.B., Finn, C., Niekum, S.: Scaling laws for reward model overoptimization in direct alignment algorithms. In: Advances in Neural Information Processing Systems. vol. 37, pp. 126207–126242

(2024)

- [44] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. In: Advances in Neural Information Processing Systems. vol. 36, pp. 53728–53741 (2023)

- [45] Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Ayan, B.K., Salimans, T., Ho, J., Fleet, D.J., Norouzi, M.: Photorealistic text-to-image diffusion models with deep language understanding. In: Advances in Neural Information Processing Systems. vol. 35, pp. 36479–36494 (2022)
- [46] Sang, H., Xu, Y., Zhou, Z., He, R., Wang, Z., Sun, J.: Crisp: Compressed reasoning via iterative self-policy distillation (2026), https://api.semanticscholar.org/ CorpusID:286255699
- [47] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y.K., Wu, Y., Guo, D.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models (2024), https://arxiv.org/abs/2402.03300
- [48] Shenfeld, I., Damani, M., Hübotter, J., Agrawal, P.: Self-distillation enables continual learning (2026), https://arxiv.org/abs/2601.19897
- [49] Song, M., Zheng, M.: A survey of on-policy distillation for large language models

(2026), https://arxiv.org/abs/2604.00626

- [50] Talebi, H., Milanfar, P.: Nima: Neural image assessment. IEEE Transactions on Image Processing 27(8), 3998–4011 (2018). https://doi.org/10.1109/TIP.2018.2831899
- [51] Team, Z.I., Cai, H., Cao, S., Du, R., Gao, P., Hoi, S., Hou, Z., Huang, S., Jiang, D., Jin, X., Li, L., Li, Z., Li, Z.Y., Liu, D., Liu, D., Shi, J., Wu, Q., Yu, F., Zhang, C., Zhang, S., Zhou, S.: Z-image: An efficient image generation foundation model with singlestream diffusion transformer (2025), https://arxiv.org/abs/2511.22699
- [52] Uma, A.N., Fornaciari, T., Hovy, D., Paun, S., Plank, B., Poesio, M.: Learning from disagreement: A survey. Journal of Artificial Intelligence Research 72, 1385–1470 (2021). https://doi.org/10.1613/jair.1.12752
- [53] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Diffusion model alignment using direct preference optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8228–8238 (June 2024)
- [54] Wang, B., Lin, R., Lu, K., Yu, L., Zhang, Z., Huang, F., Zheng, C., Dang, K., Fan, Y., Ren, X., Yang, A., Hui, B., Liu, D., Gui, T., Zhang, Q., Huang, X., Jiang, Y.G., Yu, B., Zhou, J., Lin, J.: Worldpm: Scaling human preference modeling (2025), https: //arxiv.org/abs/2505.10527
- [55] Wang, Y., Zang, Y., Li, H., Jin, C., Wang, J.: Unified reward model for multimodal understanding and generation (2026), https://arxiv.org/abs/2503.05236
- [56] Wen, C., Zhang, X., Yao, X., Yang, J.: Ordinal label distribution learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 23424–23434 (October 2023)
- [57] Wu, H., Zhang, Z., Zhang, W., Chen, C., Li, C., Liao, L., Wang, A., Zhang, E., Sun, W., Yan, Q., Min, X., Zhai, G., Lin, W.: Q-align: Teaching lmms for visual scoring via discrete text-defined levels (2023), https://arxiv.org/abs/2312.17090
- [58] Wu, J., Gao, Y., Ye, Z., Li, M., Li, L., Guo, H., Liu, J., Xue, Z., Hou, X., Liu, W., Zeng, Y., Huang, W.: Rewarddance: Reward scaling in visual generation (2025), https: //arxiv.org/abs/2509.08826
- [59] Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis

(2023), https://arxiv.org/abs/2306.09341

- [60] Xu, J., Huang, Y., Cheng, J., Yang, Y., Xu, J., Wang, Y., Duan, W., Yang, S., Jin, Q., Li, S., Teng, J., Yang, Z., Zheng, W., Liu, X., Zhang, D., Ding, M., Zhang, X., Gu, X., Huang, S., Huang, M., Tang, J., Dong, Y.: Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation (2026), https://arxiv.org/ abs/2412.21059

- [61] Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)
- [62] Yang, S., Chen, T., Zhou, M.: A dense reward view on aligning text-to-image diffusion with preference. In: Forty-first International Conference on Machine Learning (2024), https://openreview.net/forum?id=xVXnXk9I3I
- [63] Yang, Y., Long, Y., Wei, H., Chen, W., Zhang, T., Jiang, K., Fan, H., Liu, C., Chen, J., Tang, K., et al.: Joint reward modeling: Internalizing chain-of-thought for efficient visual reward models (2026), https://arxiv.org/abs/2602.07533
- [64] You, Z., Cai, X., Gu, J., Xue, T., Dong, C.: Teaching large language models to regress accurate image quality scores using score distribution (2025), https://arxiv.org/ abs/2501.11561
- [65] Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., Agarwal, R.: Generative verifiers: Reward modeling as next-token prediction (2025), https://arxiv.org/ abs/2408.15240
- [66] Zhang, S., Wang, B., Wu, J., Li, Y., Gao, T., Zhang, D., Wang, Z.: Learning multidimensional human preference for text-to-image generation (2024), https://arxiv. org/abs/2405.14705
- [67] Zhao, S., Xie, Z., Liu, M., Huang, J., Pang, G., Chen, F., Grover, A.: Self-distilled reasoner: On-policy self-distillation for large language models (2026), https://arxiv. org/abs/2601.18734
- [68] Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., Stoica, I.: Judging llm-as-a-judge with mt-bench and chatbot arena. In: Advances in Neural Information Processing Systems. vol. 36, pp. 46595–46623 (2023)
- [69] Zhu, S., Ye, X., Lu, H., Shi, W., Liu, G.: The many faces of on-policy distillation: Pitfalls, mechanisms, and fixes (2026), https://arxiv.org/abs/2605.11182

