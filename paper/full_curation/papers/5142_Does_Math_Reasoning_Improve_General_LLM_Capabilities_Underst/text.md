# arXiv:2507.00432v2[cs.AI]20Oct2025

DOES MATH REASONING IMPROVE GENERAL LLM CAPABILITIES? UNDERSTANDING TRANSFERABILITY OF LLM REASONING

Maggie Huan1,2∗ Yuetai Li3∗ Tuney Zheng4∗ Xiaoyu Xu5 Seungone Kim1 Minxin Du5 Radha Poovendran3 Graham Neubig1 Xiang Yue1†

1Carnegie Mellon University 2University of Pennsylvania 3University of Washington 4M-A-P 5The Hong Kong Polytechnic University

ziyuh@seas.upenn.edu, yuetaili@uw.edu, xyue2@andrew.cmu.edu

ABSTRACT

Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.

###### Math Reasoning

###### Other Reasoning

###### Non-Reasoning

50

60

+58.6

20

###### PerformanceGain(%)

RL on Math

RL on Math SFT on Math

RL on Math SFT on Math

+18.2

+42.7

+54.7

40

SFT on Math

| |
|---|

| |
|---|

10

+34.3

+7.5

+6.0

+5.0

30

40

+28.7

+27.4

+26.2

+0.8

0

+24.0

+23.4

+19.9

20

-4.7

10

+20.8

20

+15.1

+15.1

10

+10.9

+5.9

+9.3

20

+2.6

0

-20.9

-22.4

0

30

-26.9

-2.0

-2.0

-29.5

10

-5.6

AIME24 AIME2S MATH-500 Olympiad Avg.

GPAQ-D ACPBench LCB2 HeadQA Avg.

CoQA MC-TACO IFEval HalluIEval Avg.

Task

Task

Task

Figure 1: Impact of SFT and RL using math-only training queries on the same base model, Qwen314B-Base. Performance improvements are measured relative to the base model. SFT-trained models show limited transfer to non-reasoning tasks. In contrast, RL-trained models exhibit broader generalization across both reasoning and non-reasoning scenarios.

1 INTRODUCTION

Over the past years, the community has raced to push large language models (LLMs) to new heights on math-centric reasoning benchmarks such as MATH (Hendrycks et al., 2021b) and AIME. A

∗Equal contribution. Work done while Maggie interning at CMU. †Corresponding author.

###### Transferability Index on Other Reasoning Tasks

125

General-Reasoner-RL-14B

SimpleRL-14B

SimpleRL-1.5B

100

UniReason-RL-14B

TransferabilityIndex

75

UniReason-SFT-14B-think

OpenThinker3-7B

50

LIMO-32B DAPO-32B

Deepseek-R1-Distill-Qwen-7B

SimpleRL-Llama3.1-8B

25

OpenThinker2-32B S1.1-32B

Mistral-Reason-24B

SimpleRL-7B

0

Reasoning (RL)

Reasoning (SFT)

25

RL Mean

S1.1-7B

50

SFT Mean

75

###### Transferability Index on Non-reasoning Tasks

100

UniReason-RL-14B

SimpleRL-14B

SimpleRL-7B

50

General-Reasoner-RL-14B

TransferabilityIndex

SimpleRL-1.5B

LIMO-32B

0

DAPO-32B

SimpleRL-Llama3.1-8B

Deepseek-R1-Distill-Qwen-7B

OpenThinker2-32B

50

S1.1-32B

UniReason-SFT-14B-think

OpenThinker3-7B

100

Reasoning (RL)

Mistral-Reason-24B

Reasoning (SFT)

150

RL Mean

S1.1-7B

SFT Mean

200

Qwen2.5-1.5BQwen2.5-Math-7B Qwen2.5-7BQwen2.5-7B-Ins. Llama3.1-8B Qwen2.5-14BQwen3-14B-BaseMistral-24B-Ins.Qwen2.5-32B-Ins. Qwen2.5-32B

- Figure 2: Transferability of mathematical reasoning to other reasoning and non-reasoning tasks. The Transferability Index measures a model’s ability to transfer performance from mathematics to other domains, with positive values indicating successful transfer and negative values indicating performance degradation. Details of this metric can be found in Section 2.1. RL models consistently outperform SFT models, regardless of model size, architecture, or training data, demonstrating superior transferability.

steady stream of reasoning-tuned models (Muennighoff et al., 2025; Guha et al., 2025) now advances the state of the art on math benchmarks nearly every week, with some even surpassing the average performance of human experts (Team, 2025a; OpenAI, 2024).

The appealing performance on math reasoning is understandable: problems are well-posed, solutions are unambiguous, and evaluation is easily verifiable, often just a single number or expression (Luo et al., 2025). This clarity has made math a popular proxy task of LLM reasoning, and researchers have developed increasingly sophisticated training recipes to maximize model performance on mathematical reasoning (Wang et al., 2024a; Yue et al., 2024a; Luo et al., 2023; Shao et al., 2024; Wei et al., 2023). This trend, on one hand, should be encouraged. Mathematics is often considered the foundational language of science, and enabling machines to reason precisely over math is central to the long-term vision of automated scientific discovery (Mishra et al., 2022). On the other hand, real-world tasks extend far beyond math. The majority of user-facing applications, question answering, dialogue, instruction following, require broader linguistic and commonsense competence that math alone does not test (Ma et al., 2025).

This raises a natural question: Do improved math reasoning abilities transfer to general LLM capabilities? Specifically, can gains in solving math problems transfer to other reasoning domains (e.g., scientific QA (Welbl et al., 2017), coding (Jain et al., 2025), agent planning (Xie et al., 2024), logical deduction (Dziri et al., 2024)) and to tasks (e.g., conversational QA (Reddy et al., 2019), instruction following (Zhou et al., 2023)) that do not require extensive reasoning? To investigate, we evaluate over 20 representative open-weight reasoning models, all of which exhibit impressive performance on recent math benchmarks across a suite of other reasoning and non-reasoning tasks. To quantitatively analyze this problem, we propose Transferability Index, a metric to measure how reasoning models can transfer their capabilities from one domain to another. Surprisingly, as shown in Figure 2, we find that some of these models fail to transfer their improved mathematical reasoning capabilities to other domains while others succeed.

What drives this divergence? Model recipes vary widely in size, data distribution, and architecture. Yet among various parts, we identify one factor that consistently predicts transferability: the

fine-tuning paradigm. Across families and sizes, models fine-tuned using reinforcement learning (RL) (Su et al., 2025; Yeo et al., 2025) exhibit much stronger generalization to non-math tasks than those trained with supervised fine-tuning (SFT) (Yue et al., 2024a;b), which often show signs of catastrophic forgetting over a wide range of non-math tasks.

To validate this observation, we conduct a controlled study. We fine-tune Qwen3-14B (Team, 2025b) on the high-quality math dataset derived from MATH and DeepScaler (Luo et al., 2025). For SFT, we construct targets via rejection sampling using Qwen3-32B, keeping only teacher responses that yield correct final answers. For RL, we apply a standard GRPO (Shao et al., 2024) setup using answer correctness as the reward. As shown in Figure 1, the results mirror our large-scale audit: RL-tuned models generalize well to non-math domains, despite being trained solely on math queries, while SFT-tuned models do not.

To better understand why this occurs, we use two diagnostic tools: (1) latent-space principal component analysis (PCA) on hidden states across layers, and (2) KL-divergence on token distributions before and after fine-tuning. These methods allow us to quantify how much the model’s internal representations and output space shift during training. We find that SFT induces significant drift in both latent and output spaces, especially for non-reasoning inputs, whereas RL better preserves the geometry of internal features and the stability of the token distributions. We then conduct an ablation study to learn which RL components contribute to the significant transferability. Results show that the sampling distribution, credit assignment and negative gradient contribute to generalization, while KL regularization play subtle roles in the transferability of LLM reasoning.

2 PHENOMENA: PERFORMANCE DISCREPANCIES OF REASONING MODELS

- 2.1 OBSERVED PERFORMANCE DISCREPANCIES BETWEEN RL AND SFT

Setup. We evaluate over 20 off-the-shelf reasoning models on different benchmarks. Namely, we select benchmarks from the following three categories: (1) math reasoning tasks: MATH500 (Hendrycks et al., 2021b), AIME24, AIME25, OlympiadBench (He et al., 2024), which contain mathematical problems only; (2) other reasoning tasks: LiveCodeBench (Jain et al., 2025), GPQA-Diamond (Rein et al., 2024), ACPBench (Kokel et al., 2025), HeadQA (Vilares & G´omezRodr´ıguez, 2019), which contain more general reasoning questions, such as medical reasoning, code generation, and language-based agent planning tasks; (3) non-reasoning tasks: CoQA (Reddy et al., 2019), IFEval (Zhou et al., 2023), HaluEval (Li et al., 2023), MC-TACO (Zhou et al., 2019), which contain factual, alignment, or conversational problems such as commonsense question answering and instruction-following. We used accuracy to evaluate the models’ performance. Detailed explanation about experiment setup, benchmarks, and evaluation metrics can be found in Appendix A.3.

To better evaluate the model’s transferability across a wide range of task groups, we define Transferability Index (TI) as follows:

- (1) Per–benchmark gain and normalization. z-normalization makes gains comparable within each group. For each group g ∈ {math,other,non} and benchmark b ∈ Bg,

∆Rb = Rbmodel − Rbbase, σg = Std{∆Rb : b ∈ Bg}, δb =

∆Rb σg

.

- (2) Robust gain score and difficulty weighting. We then compute the domain index to normalize benchmark differences. We control extremes with a signed square-root and up-weight harder tasks, and used a group-level Domain Index aggregates these weighted, robust gains:

sb = sign(δb)|δb|1/2, wb = 100 − Rbbase, wˆb =

wb u∈Bg wu

, DIg =

b∈Bg

wˆb sb.

- (3) Transferability relative to math. Finally, TI is the ratio of the group’s improvement to the math group’s improvement:

TIg(%) =

DIg DImath × 100, g ∈ {other,non}.

- Table 1: Performance of our model and baseline models on math reasoning, other reasoning and non-reasoning tasks. UniReason-Qwen3-14B-think (SFT) denotes the model is trained with the dataset distilled from Qwen3-32B thinking mode, UniReason-Qwen3-14B-no-think (SFT) denotes it is trained with Qwen3-32B non-thinking mode. The grey-colored row represents the performance difference between RL and the highest SFT score.

Math Reasoning Model AIME24 AIME25 MATH500 Olympiad Average

Qwen3-14B-Base 13.0 9.3 60.4 27.9 27.7 – UniReason-Qwen3-14B-think (SFT) 52.0 37.0 85.0 25.0 49.8 – UniReason-Qwen3-14B-no-think (SFT) 16.0 13.0 77.2 22.7 32.3 – UniReason-Qwen3-14B (RL) 55.7 38.0 87.8 33.8 53.8 – ∆ (RL - SFT) +3.7 +1.0 +2.8 +8.8 +4.1 –

Other Reasoning Model GPQA LiveCodeBench2 ACPBench HeadQA Average TIother Qwen3-14B-Base 42.6 29.7 10.7 37.6 30.2 – UniReason-Qwen3-14B-think (SFT) 55.9 21.9 68.6 34.8 45.3 +52.2 UniReason-Qwen3-14B-no-think (SFT) 48.7 23.5 69.3 35.0 45.2 +165.4 UniReason-Qwen3-14B (RL) 57.7 40.6 65.4 40.2 60.0 +82.3 ∆ (RL - SFT) +1.8 +17.1 –3.9 +5.2 +5.1 Non-Reasoning

Model CoQA MC-TACO IFEval HalluEval Average TInon Qwen3-14B-Base 10.0 67.7 69.2 35.7 45.7 – UniReason-Qwen3-14B-think (SFT) 1.7 38.2 42.3 2.3 21.1 -104.1 UniReason-Qwen3-14B-no-think (SFT) 5.3 66.1 41.4 3.3 29.0 –278.9 UniReason-Qwen3-14B (RL) 28.2 74.0 70.0 40.7 53.2 +52.2 ∆ (RL - SFT) +22.9 +7.9 +27.7 +37.4 +24.0 -

A positive TIg indicates positive transfer in group g, and higher TIg indicates the finetuned model achieves stronger and more consistent improvements in transferring math capabilities to other domains.

Results. Figure 2 and Table 5 compare transferability indices across models. Transferability varies with finetuning method, model size, and architecture; the dominant factor is the method: RL–tuned models consistently achieve higher indices on both other–reasoning and non–reasoning tasks, whereas SFT–trained models often yield negative TInon. This suggests that on–policy RL reinforces reasoning while minimally perturbing general–domain representations, whereas off–policy SFT on large static reasoning corpora can over–specialize the latent space, degrading non–reasoning performance.

- 2.2 CONTROL STUDY

Motivated by our findings in Section 2.1, we design a light-weight controlled study to directly compare SFT and RL on an identical dataset. Concretely, we start from a small, high-quality mathematics dataset (see Appendix A.3.2 for details), then query a strong teacher model (Qwen3-32BInstruct) to extract complete chain-of-thought (CoT) reasoning traces with reject sampling. These CoT traces become our SFT training targets, while the original answer labels serve as the rewards for RL. This alignment ensures both paradigms learn from the same data samples. Then, we take the Qwen3-14B-Base model and fine-tune it in two ways: (i) SFT on the teacher-generated CoT traces; (ii) RL using only the groundtruth. We name our model UniReason. We compare against the Qwen3-14B-Base model. Evaluation is conducted on three benchmark groups mentioned above using accuracy. Details about training datasets, baseline models, and hyperparameters could also be found in Appendix A.3.

Our experimental results on three groups of benchmarks (see Table 1) reveal a consistent pattern:

- • On math reasoning (Table 1), our UniReason-Qwen3-14B(RL) model climbs to 55.7% on AIME24, 87.8% on MATH500, and 33.8% on OlympiadBench, outperforming corresponding SFT-based models.

- • For other reasoning tasks, SFT-based models make uneven progress (e.g. UniReason-Qwen314B-SFT-think scores 55.9% on GPQA), whereas RL fine-tuning yields significant lifts: UniReason-Qwen3-14B(RL) gains 1.8% on GPQA, and 17.1% on LiveCodeBench2 over SFT.
- • Crucially, in non-reasoning evaluations, SFT models stagnate or decline, while the RL model recover and exceed the base in nearly all the benchmarks.
- • From Table 1, our RL-tuned model achieve positive TIother and TInon, meaning its math improvements also transfer to other reasoning and even boost non-reasoning performance. In contrast, SFT-based models often have negative TInon, indicating poor transfer beyond reasoning tasks.

These results show that RL-tuned reasoning models perform generally better than SFT-based models on both reasoning and non-reasoning tasks when carefully controlling other factors. Especially, our UniReason model is trained on a single distilled math dataset, but it still preserves and even improves general-domain performance while showing strong reasoning gains.

- 3 LATENT REPRESENTATION SHIFTS: INSIGHTS FROM PCA ANALYSIS

As discussed in Section 2.1, applying SFT to the Qwen model improves reasoning abilities such as mathematical problem-solving and code generation, but impairs general-domain performance. We observe that most SFT models fail to transfer their improved mathematical reasoning capabilities to other domains. In contrast, our controlled study shows that RL models generalize well to non-math domains, despite being trained solely on math queries, whereas SFT models do not.

To understand the underlying cause of this transferability gap, we employ PCA shift analysis to examine how the internal feature geometry of the model evolves under different training paradigms, model sizes, and model families across diverse query distributions. Recent studies (Xu et al., 2025a; Zheng et al., 2025) demonstrate that PCA shift analysis provides a sensitive and interpretable measure of representational changes relevant to task performance. Importantly, changes in model parameters do not always correspond to functional differences: large weight updates may leave outputs unchanged, while subtle parameter modifications can lead to significant shifts in the activation distribution. By focusing on hidden representations, PCA shift directly captures how the model encodes

PCA Shift across Models and Tasks

Math

Other-Reasoning

Other-Reasoning

Non-Reasoning

Non-Reasoning

UniReason-Qwen3-14B-no think (SFT); Base: Qwen3-14B-Base

MATH500

Head_QA

Livecodebench

COQA

HalluEval

Base

1000

1000

1000

1000

1000

SFT (no think)

(PC2)

d(*) = 17.23

d(*) = 25.35

d(*) = 15.62

d(*) = 10.66

d(*) = 109.30

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

100 0 100 200

100 0 100 200

100 0 100 200

100 0 100 200

100 0 100 200

UniReason-Qwen3-14B-think (SFT); Base: Qwen3-14B-Base

Base

1000

1000

1000

1000

1000

SFT (think)

(PC2)

d(*) = 10.89

d(*) = 7.45

d(*) = 17.06

d(*) = 9.47

d(*) = 29.58

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

100 0 100 200

100 0 100 200

100 0 100 200

100 0 100 200

100 0 100 200

UniReason-Qwen3-14B (RL); Base: Qwen3-14B-Base

Base

1000

1000

1000

1000

1000

RL

(PC2)

d(*) = 8.31

d(*) = 7.36

d(*) = 2.94

d(*) = 3.61

d(*) = 26.00

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

100 0 100 200

100 0 100 200

100 0 100 200

100 0 100 200

100 0 100 200

(PC1 )

(PC1 )

(PC1 )

(PC1 )

(PC1 )

- Figure 3: PCA shift of Qwen3-14B-Base across different training methods and tasks. d(∗) is the Euclidean distance between representation centroids before and after training. The first two rows show models trained with SFT , and the last row shows models trained with RL . RL training results in the smallest PCA shift for all task types, suggesting more stable latent representations.

- Table 2: Mean PCA shift magnitudes on math, reasoning, and non-reasoning tasks for Qwen3-14B and our finetuned models. RL models exhibit lower shifts than SFT models.

Model Math Other-Reasoning Non-Reasoning Qwen3-14B-no-think 40.4 13.9 129.8 Qwen3-14B-think 76.5 38.8 152.0 UniReason-Qwen3-14B-no-think (SFT) 21.4 10.9 113.7 UniReason-Qwen3-14B-think (SFT) 19.2 6.7 38.2 UniReason-Qwen3-14B (RL) 8.5 3.5 36.9

and processes information, offering a more faithful account of its internal knowledge state than parameter-based metrics. This perspective allows us to distinguish between true knowledge erasure and parameterization changes that leave the underlying feature space intact. Furthermore, since transferability fundamentally relies on the alignment and stability of learned representations across tasks or domains, PCA shift is particularly effective for diagnosing changes that may impact crossdomain generalization. Shifts in principal components reveal whether the model’s internal feature space remains suitable for knowledge transfer or has been disrupted by post-training interventions.

- 3.1 EXPERIMENTAL SETUP

Models and Tasks. In Section 2.1, we observe that models trained on math datasets show moderate transferability on other reasoning tasks. We perform PCA shift analyses on the corresponding models and tasks, aiming to critically assess these phenomena from a feature-space perspective.

Evaluation. Given input queries X, we extract hidden states H(i∗) at each layer i for each model state (∗) ∈ {orig,updated}. Applying PCA (n = 2) to H(i∗), we compute the mean projection m(i,∗1) onto the first principal direction (PC1) and m(i,∗2) onto the second (PC2). The PCA shift is defined as ∆m(i,∗1) = m(i,∗1) − morigi,1 for PC1, while for PC2, we directly report m(i,∗2) as an indicator of change.

- 3.2 INVESTIGATING LATENT SPACE SHIFT To quantify the overall latent shift, we define a representation center for each model state (∗) as

the mean of PCA-projected coordinates across all layers: z(∗) = L1 Li=1 z(i∗), where L denotes the total number of layers and z(i∗) = (∆m(i,∗1),m(i,∗2)) is the vector of PCA coordinates for layer i in state (∗). The latent shift between two model states, such as the original (base) and an updated model, is then measured by the Euclidean distance: d(∗) = z(∗) − z(orig) 2 .

Based on the analyses in Appendix A.4, RL-based training proves essential for developing generalizable language models that maintain a strong balance between general-domain and reasoning capabilities. Motivated by this, we further analyze our proposed models in the controlled study. As shown in Table 2, RL models achieve the lowest PCA shift magnitudes across math, other-reasoning, and non-reasoning tasks. Figure 3 further supports these findings, illustrating that the RL model consistently yields minimal and tightly clustered latent shifts across diverse benchmarks. In contrast, SFT models, particularly those without explicit reasoning signals, exhibit more scattered shifts.

These results, together with the evaluations in Section 2.1, highlight the advantage of RL over SFT. They underscore the importance of a holistic, well-balanced optimization objective—rather than isolated interventions, for mitigating catastrophic forgetting while preserving performance in LLMs.

###### SFT (no think)

###### SFT (think)

###### RL

0.490

0.5

0.5

0.5

###### KLDivergence

###### KLDivergence

###### KLDivergence

0.4

0.4

0.4

0.372

0.353

0.338

0.311

0.282 0.283

0.3

0.3

0.3

0.247 0.252 0.248

0.2

0.2

0.2

0.084

0.1

0.1

0.1

0.057

0.048

0.009 0.019

0.0

0.0

0.0

MATH-500OlympiadGPQA-Q IFEval CoQA

MATH-500OlympiadGPQA-Q IFEval CoQA

MATH-500OlympiadGPQA-Q IFEval CoQA

Tasks

Tasks

Tasks

- Figure 4: KL divergence analysis of RL and SFT models. Higher KL divergence indicates greater distribution shifts from the original backbone model. We observe that RL models consistently exhibit significantly lower KL divergence compared to SFT models across different tasks, suggesting less distribution shift during training.

- 4 TOKEN DISTRIBUTION SHIFTS: INSIGHTS FROM KL DIVERGENCE AND RANK ANALYSES

In this section, we conduct token-level analyses to further examine the distribution shift of RL and SFT models trained on mathematical reasoning data.

- 4.1 SETUP

KL-divergence serves as a standard metric for measuring differences between probability distributions. For token rank shift analysis, we first generate tokens using the fine-tuned model, then decode these same tokens using the backbone model to determine their original ranking positions. The rank shift is calculated as the difference in token rankings between the fine-tuned model and the backbone model for each token (Li et al., 2025c; Lin et al., 2023). Following the observations in Section 2.1, we perform additional token-distribution analyses on the corresponding models and tasks to assess the model distribution shift from a token-space perspective. Specifically, we employ KL-divergence and token rank shift metrics to analyze distribution shifts between models.

- 4.2 RESULT

RL models exhibit lower KL-divergence from backbone models. In Figure 4, we observe that the KL divergence of SFT models on both reasoning and non-reasoning tasks is significantly larger than that of RL models.

Logical Structure vs Content Words (Red = Logical Structure, Blue = Content-specific)

This indicates that RL models exhibit substantially less distribution shift from the token distribution level compared to SFT models. For instance, UniReasonQwen3-14B-SFT-no-think demonstrates KL divergences of 0.372 and 0.283 on MATH-500 and IFEval respectively compared to the backbone model, whereas UniReasonQwen3-14B(RL) achieves considerably lower KL divergences of only 0.084 and 0.019 on the corresponding tasks.

RL Model SFT Model

[Figure 1]

[Figure 2]

Figure 5: Word clouds showing significantly shifted tokens between UniReason-Qwen3-14B-RL (left) and UniReason-Qwen3-14B-SFTthink (right). Tokens are extracted based on frequency and rank shifts compared with base model then categorized as logical-structural words (in red) or content-specific words (in blue). The RL model promptly shifts logic-related tokens such as But and So while the SFT model shifts including many irrelevant tokens.

RL models demonstrate reduced token rank shifts. Across both reasoning and non-reasoning tasks, RL models show markedly smaller average token-rank shifts than SFT models as shown in Figure 17 in Appendix A.6. For example, UniReason-Qwen3-14B (RL) averages only 0.98 positions of shift, whereas the SFT no-think variant reaches 10.6. This indicates RL tends to preserve the base token distribution while SFT induces larger reordering. Detailed position-wise trajectories are provided in Figure 20 in Appendix A.6.

RL models selectively shift task-relevant tokens, while SFT models shift numerous irrelevant tokens. A case study in Table 8 shows RL moving a small set of task-relevant tokens (e.g., “define”, “add”, “second”, “number” for reasoning; “<<”, “>>”, “write”, “formally” for non-reasoning), while SFT shifts far more tokens (e.g., 390 for reasoning / 158 for non-reasoning), including many that are query-irrelevant and often inject reasoning markers into non-reasoning prompts. The tokenfrequency word cloud in Figure 5 further visualizes this selectivity gap. More detailed information could be found in Appendix A.6.

- 5 ABLATION: WHICH COMPONENTS OF RL DRIVE GENERALIZATION?

Earlier sections observed that reinforcement-style fine-tuning generalizes more effectively than standard supervised fine-tuning. Latent and output space analyses revealed that RL updates produce small, targeted “marginal shifts” around the base model, whereas SFT often causes substantial drift. This section sharpens the inquiry: Which specific components of RL updates contribute to improved generalization?

We contrast SFT and RL by aligning their objectives and gradients, yielding a unified surrogate loss that makes their differences explicit along four levers: (i) sampling distribution: SFT trains offpolicy on a fixed dataset, whereas RL samples on-policy from the live model; (ii) credit assignment: uniform weighting versus advantage-weighted updates that emphasize task-salient tokens; (iii) KL regularization: penalizing deviations from a reference policy such as PPO or GRPO; (iv) negative gradient: When sampling from π, RL learn from negative examples, while SFT only imitates positive ones.

- 5.1 CONTRASTING OBJECTIVES

For prompts x and completions y, let πθ(y | x) denote the current policy, and πref(y | x) denote a fixed reference policy (e.g., the initialization).

Supervised Fine-tuning. With reference completions y⋆, the objective is:

LSFT(θ) = −E(x,y⋆)∼D log πθ(y⋆ | x) . (1)

Reinforcement Learning. For the same prompts, we sample y ∼ πθ and weight each sample by an advantage A(x,y):

θ(·|x) A(x,y) · log πθ(y | x) . (2)

LRL(θ) = −Ex∼DEy∼π

Usually, a KL term is added to prevent the policy model from being too far away from the initialized model. We generalize these objectives using:

Lq,w,β(θ) = −Ex∼DEy∼q(·|x) w(x,y)·log πθ(y | x) +β Ex∼D KL(πθ(· | x)∥πref(· | x)) . (3)

- Table 3: Ablation settings and their corresponding sampling distributions, credit weights, and KL–regularization coefficients. The two shaded rows mark the standard baselines: Off-policy SFT and Online RL.

Setting Sampling q Weights w KL Reg. β

Off-policy SFT δy=y⋆ 1 0 On-policy SFT πθ 1 (reject sample) 0

Off-policy RL δy=y⋆ Advantage At 0 On-policy RL (no KL) πθ Advantage At 0 On-policy RL πθ Advantage At > 0

- Table 4: Performance of all the ablation models. Scores are avgerage scores on math reasoning, other reasoning and non-reasoning tasks.

Model Math Avg. Other Reasoning Avg. Non-Reasoning Avg. TIother TInon

Qwen3-8B-Base 27.6 23.6 33.6 Qwen3-8B-Off-policy-SFT 41.9 34.4 26.6 18.3 -40.5 Qwen3-8B-On-policy-SFT 33.7 35.7 35.0 68.6 30.2

Qwen3-8B-Off-policy-RL 45.5 35.9 31.7 36.4 4.5 Qwen3-8B-On-policy-RL (no KL) 37.1 38.2 35.8 65.6 39.3 Qwen3-8B-On-policy-RL 38.6 39.9 35.0 63.7 32.4

- 5.2 ABLATION RESULTS

Setup. For ablation study, we used 5 settings from Off-policy and On-policy SFT, Off-policy and On-policy RL with and without KL. We used the same set of math training queries and a Qwen38B-Base model. For on-policy SFT: we sample a batch of training queries and let the policy model generate the responses as in standard RL training. For off-policy RL, we use the Qwen3-32B model in think mode to generate n = 8 responses for each query. We utilized GRPO’s objective with KL divergence and used the response with the highest reward for gradient computation. For Off-policy SFT, we used data generated from Qwen3-32B model in think mode with reject sampling. On-policy RL used the same math dataset as our main models. The results are in Table 4.

Sampling distribution is critical. The results in Table 4 show that on-policy methods outperform off-policy methods across both evaluation categories and training paradigms. Specifically, on-policy SFT achieves higher average scores on Other Reasoning tasks and Non-Reasoning tasks compared to off-policy SFT. A similar trend is observed when KL regularization is applied to RL: on-policy RL continues to yield superior transferability on Other Reasoning and Non-Reasoning tasks. These results demonstrate that on-policy methods consistently transfer better than off-policy methods. One possible explanation is that on-policy training exposes models to data generated under their own evolving policy, allowing updates to directly align with the distribution the model will face at inference. This reduces distribution mismatch and mitigates overfitting to fixed offline samples, leading to stronger generalization. Gradient Norm during training also shows that on-policy methods update the gradient in small and conservative steps compared to off-policy methods which aggressively update its gradient especially from the start. Further details can be found in Figure 6 and AppendixA.5.

##### Grad Norm Comparison

Credit assignments and negative examples matter. Experiments on credit assignment and negative gradients show that adding these mechanisms not only improves transferability but also increases response length during training. In off-policy settings, RL achieves higher average scores on both Other Reasoning and Non-Reasoning tasks. Under on-policy training, RL outperforms SFT on Math and Other Reasoning tasks, while showing similar performance on Non-Reasoning tasks. Credit assignment helps the model focus updates on the tokens that matter most for solving the task, while penalizing uninformative gradients reduces the chance of reinforcing wrong or irrelevant patterns. Together, these mechanisms encourage more useful exploration and lead to better generalization. Further details are provided in Figure 14.

8B-online-RL

- 0

- 1

- 2

8B-online-SFT 8B-offline-SFT 8B-offline-RL

GradNorm

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0 20 40 60 80 Step

Figure 6: Grad norm comparison across online and offline method. Online methods perform steadier updates. Offline updates show larger gradient norms in early steps, reflecting more abrupt shifts that influent generalization.

KL regularization play subtle roles. On-policy RL performance remains largely unchanged with or without KL regularization. This suggests that, although KL regularization constrains models to stay closer to their original policy during updates, its overall effect on transferability is limited. One possible reason is that on-policy RL already aligns updates with the on-policy distribution, making the additional constraint from KL regularization less impactful.

- 5.3 RELATED WORKS

Reasoning Fine-Tuning of LLMs. Recent advancements in large language models have notably emphasized specialized fine-tuning methods to enhance reasoning capabilities (Wong et al., 2025; Chen et al., 2023; Ziegler et al., 2019; Liu et al., 2025; Wang et al., 2024c; Li et al., 2025b; Feng et al., 2025; Xu et al., 2025b; Yang et al., 2025; Li et al., 2025a; Yeo et al., 2025). The chain-of-thought prompting strategy introduced by Wei et al. (2022) encourages models to produce step-by-step explanations, significantly boosting performance in symbolic reasoning tasks (Lambert et al., 2025; Wei et al., 2022; Longpre et al., 2023; Yu et al., 2024). Subsequent extensions, such as DeepSeekR1 (Team, 2025a), have integrated reinforcement learning approaches alongside CoT, optimizing models through reward-driven policy improvements. Such RL-enhanced fine-tuning has achieved state-of-the-art results on benchmarks and competitive programming challenges (Hendrycks et al.,

- 2021b; Team et al., 2025; Team, 2025a; Lambert et al., 2025).

Supervised Fine-Tuning vs. Reinforcement Learning for LLMs. Fine-tuning methods for reasoning typically fall into two major categories: supervised fine-tuning and reinforcement learning (Chen et al., 2024). SFT methods predominantly utilize annotated reasoning trajectories or solution traces, directly training models to replicate explicit reasoning sequences from datasets (Wei et al.,

- 2022; Wang et al., 2023). RL-based fine-tuning, however, guides models by rewarding accurate and logically coherent reasoning steps without explicit step-by-step supervision, allowing exploration and optimization of reasoning pathways through feedback loops (Ziegler et al., 2019; Liu et al., 2025; Wang et al., 2024c; Chu et al., 2025).

Generalization in Reasoning Models. Interestingly, models heavily fine-tuned for formal reasoning sometimes falter on more general language tasks (Kumar et al., 2022). For example, OpenAI’s

- o1, while excelling in STEM benchmarks, raised concerns about its versatility on other tasks (OpenAI, 2024). Follow-up research introduced reinforcement fine-tuning precisely to address this gap, aiming to adapt a generalist model’s reasoning to new domains with limited data (Zhang et al., 2024). Indeed, o1 and similar reasoning models are built on strong general-purpose bases to retain broad knowledge (OpenAI, 2024; Hendrycks et al., 2021a). Nonetheless, trade-offs have been observed. Wang et al. (2024b) found that fine-tuning on a narrow set of instruction types can degrade a model’s performance on other skills. Recent works have also stepped into analyzing the cross-domain performance of reasoning models (Sun et al., 2025), especially for RL-based approaches (Cheng et al., 2025; Hu et al., 2025).

Representation-Level Analysis. Fine-tuning for reasoning models not only boosts task performance but also alters the model’s internal representations (Sheng et al., 2024). Recent studies have begun to probe how CoT-based fine-tuning changes the latent space of LLMs (Xu et al., 2025a; Wang et al., 2025). Lobo et al. (2024) find that task-specific fine-tuning can reduce the faithfulness

- of a model’s generated reasoning chains, indicating shifts in its underlying inference mechanisms. Complementary analyses of hidden states provide insight into such shifts. Xu et al. (2024) proposes a quantitative framework for assessing ideas that leverages hidden representations from LLMs to predict the merit of scientific ideas. Techniques like principal component analysis further reveal that fine-tuning can carve out new directions in representation space that correspond to reasoning-related features (Xu et al., 2025a; Zhou et al., 2025).

- 6 CONCLUSION

We studied what drives transferability of reasoning models across reasoning and non-reasoning benchmarks. Our evidence shows that, beyond size and architecture, the fine-tuning paradigm is decisive: RL-tuned models improve math reasoning while preserving positive transfer to other reasoning and non-reasoning tasks, whereas SFT often induces negative transfer on non-reasoning benchmarks. Latent-space PCA further indicates that RL stays close to the backbone (stable features), while SFT yields larger shifts, especially on non-reasoning data. Token–distribution analysis aligns with this: RL selectively adjusts a small set of task-relevant tokens, whereas SFT perturbs many irrelevant ones. Notably, UniReason-Qwen3-14B-RL trained on 47K math examples achieves the strongest balance of reasoning gains and general-domain retention among compared models, supporting these conclusions. We then studied which factor in RL benefits generalization by a set of controlled study, and learned that samply distribution is the most important factor.

Going forward, we plan to extend this approach in three directions: (1) exploring hybrid recipes to further optimize the trade-off between reasoning and non-reasoning tasks, spanning our current math-based approach to more reasoning domains; (2) developing adaptive regularization strategies that account for backbone architecture and size to enhance transferability; and (3) proposing more comprehensive tools to evaluate transferability in multi-modal, long-context, and interactive generation settings to ensure robust performance evaluation across an even wider range of real-world applications.

REFERENCES

Jie Chen, Xintian Han, Yu Ma, Xun Zhou, and Liang Xiang. Unlock the correlation between supervised fine-tuning and reinforcement learning in training code large language models. ArXiv preprint, abs/2406.10305, 2024. URL https://arxiv.org/abs/2406.10305.

Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. TheoremQA: A theorem-driven question answering dataset. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7889–7901, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.489. URL https: //aclanthology.org/2023.emnlp-main.489/.

Zhoujun Cheng, Shibo Hao, Tianyang Liu, Fan Zhou, Yutao Xie, Feng Yao, Yuexin Bian, Yonghao Zhuang, Nilabjo Dey, Yuheng Zha, Yi Gu, Kun Zhou, Yuqi Wang, Yuan Li, Richard Fan, Jianshu She, Chengqian Gao, Abulhair Saparov, Haonan Li, Taylor W. Killian, Mikhail Yurochkin, Zhengzhong Liu, Eric P. Xing, and Zhiting Hu. Revisiting reinforcement learning for llm reasoning from a cross-domain perspective, 2025. URL https://arxiv.org/abs/2506. 14965.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. ArXiv preprint, abs/2501.17161, 2025. URL https://arxiv.org/ abs/2501.17161.

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Za¨ıd Harchaoui, and Yejin Choi. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36, 2024. URL https: //openreview.net/forum?id=Fkckkr3ya8.

Yichen Feng, Zhangchen Xu, Fengqing Jiang, Yuetai Li, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. Visualsphinx: Large-scale synthetic vision logic puzzles for rl, 2025. URL https://arxiv.org/abs/2505.23977.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 2024. URL https://zenodo.org/records/12608602.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. ArXiv preprint, abs/2506.04178, 2025. URL https://arxiv.org/abs/ 2506.04178.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In ACL (1), pp. 3828–3850, 2024. URL https://doi.org/10. 18653/v1/2024.acl-long.211.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. URL https://openreview.net/forum?id=7Bywt2mQsCe.

Chuxuan Hu, Yuxuan Zhu, Antony Kellermann, Caleb Biddulph, Suppakit Waiwitlikhit, Jason Benn, and Daniel Kang. Breaking barriers: Do reinforcement post training gains transfer to unseen domains?, 2025. URL https://arxiv.org/abs/2506.19733.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= chfJJYC3iL.

Harsha Kokel, Michael Katz, Kavitha Srinivas, and Shirin Sohrabi. Acpbench: Reasoning about action, change, and planning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 26559–26568, 2025. URL https://ojs.aaai.org/index.php/AAAI/ article/view/34857.

Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution, 2022. URL https://arxiv. org/abs/2202.10054.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training, 2025. URL https: //arxiv.org/abs/2411.15124.

Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview.net/ forum?id=bxsrykzSnq.

Xiaomin Li, Zhou Yu, Zhiwei Zhang, Xupeng Chen, Ziji Zhang, Yingying Zhuang, Narayanan Sadagopan, and Anurag Beniwal. When thinking fails: The pitfalls of reasoning for instructionfollowing in llms, 2025a. URL https://arxiv.org/abs/2505.11423.

Yuetai Li, Zhangchen Xu, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Xiang Yue, and Radha Poovendran. Temporal sampling for forgotten reasoning in llms, 2025b. URL https://arxiv.org/abs/2505.20196.

Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, and Radha Poovendran. Small models struggle to learn from strong reasoners. CoRR, abs/2502.12143, February 2025c. URL https://doi.org/10.48550/arXiv. 2502.12143.

Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. The unlocking spell on base llms: Rethinking alignment via in-context learning, 2023. URL https://arxiv.org/abs/2312.01552.

Zeyuan Liu, Ziyu Huan, Xiyao Wang, Jiafei Lyu, Jian Tao, Xiu Li, Furong Huang, and Huazhe Xu. World models with hints of large language models for goal achieving. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings of the 2025 Conference of the Nations of the Americas

Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 50–72, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.3. URL https://aclanthology.org/2025.naacl-long.3/.

Elita Lobo, Chirag Agarwal, and Himabindu Lakkaraju. On the impact of fine-tuning on chain-ofthought reasoning. ArXiv preprint, abs/2411.15382, 2024. URL https://arxiv.org/abs/ 2411.15382.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 22631–22648. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/longpre23a.html.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. ArXiv preprint, abs/2308.09583, 2023. URL https://arxiv.org/abs/2308.09583.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. URL https://huggingface.co/ agentica-org/DeepScaleR-1.5B-Preview.

Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. ArXiv preprint, abs/2505.14652, 2025. URL https://arxiv.org/abs/2505.14652.

Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. LILA: A unified benchmark for mathematical reasoning. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pp. 5807–5832. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.392. URL https://doi.org/10.18653/v1/2022.emnlp-main.392.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.

Negin Raoof, Etash Kumar Guha, Ryan Marten, Jean Mercat, Eric Frankel, Sedrick Keh, Hritik Bansal, Georgios Smyrnis, Marianna Nezhurina, Trung Vu, Zayne Rea Sprague, Mike A Merrill, Liangyu Chen, Caroline Choi, Zaid Khan, Sachin Grover, Benjamin Feuer, Ashima Suvarna, Shiye Su, Wanjia Zhao, Kartik Sharma, Charlie Cheng-Jie Ji, Kushal Arora, Jeffrey Li, Aaron Gokaslan, Sarah M Pratt, Niklas Muennighoff, Jon Saad-Falcon, John Yang, Asad Aali, Shreyas Pimpalgaonkar, Alon Albalak, Achal Dave, Hadi Pouransari, Greg Durrett, Sewoong Oh, Tatsunori Hashimoto, Vaishaal Shankar, Yejin Choi, Mohit Bansal, Chinmay Hegde, Reinhard Heckel, Jenia Jitsev, Maheswaran Sathiamoorthy, Alex Dimakis, and Ludwig Schmidt. Automatic evals for llms, 2025. URL https://github.com/mlfoundations/evalchemy.

Siva Reddy, Danqi Chen, and Christopher D. Manning. CoQA: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266, 2019. doi: 10.1162/tacl a 00266. URL https://aclanthology.org/Q19-1016.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a

benchmark. In First Conference on Language Modeling, 2024. URL https://openreview. net/forum?id=Ti67584b98.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, pp. 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400711961. doi: 10. 1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

Shuqian Sheng, Yi Xu, Tianhang Zhang, Zanwei Shen, Luoyi Fu, Jiaxin Ding, Lei Zhou, Xiaoying Gan, Xinbing Wang, and Chenghu Zhou. RepEval: Effective text evaluation with LLM representation. pp. 7019–7033, November 2024. doi: 10.18653/v1/2024.emnlp-main.398. URL https://aclanthology.org/2024.emnlp-main.398/.

Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains, 2025. URL https://arxiv.org/abs/2503.23829.

Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, and Dawn Song. Omega: Can llms reason outside the box in math? evaluating exploratory, compositional, and transformative generalization, 2025. URL https://arxiv.org/abs/2506.18880.

DeepSeek-AI Team. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning, 2025a. URL https://arxiv.org/abs/2501.12948.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Weixin Xu, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, Zonghan Yang, and Zongyu Lin. Kimi k1.5: Scaling reinforcement learning with llms, 2025. URL https://arxiv.org/abs/2501.12599.

Qwen Team. Qwen2.5 technical report, 2025b. URL https://arxiv.org/abs/2412. 15115.

David Vilares and Carlos G´omez-Rodr´ıguez. HEAD-QA: A healthcare dataset for complex reasoning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 960–966, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1092. URL https://aclanthology.org/P19-1092.

Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. 2024a. URL https://arxiv.org/abs/2312.08935.

Renxi Wang, Haonan Li, Minghao Wu, Yuxia Wang, Xudong Han, Chiyu Zhang, and Timothy Baldwin. Demystifying instruction mixing for fine-tuning large language models. In ACL (Student Research Workshop), pp. 162–169, 2024b. URL https://aclanthology.org/2024. acl-srw.15.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.

Yiming Wang, Pei Zhang, Baosong Yang, Derek F. Wong, and Rui Wang. Latent space chain-ofembedding enables output-free LLM self-evaluation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= jxo70B9fQo.

Zihan Wang, Yunxuan Li, Yuexin Wu, Liangchen Luo, Le Hou, Hongkun Yu, and Jingbo Shang. Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 7309–7319, Miami, Florida, USA, November 2024c. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-emnlp.429. URL https://aclanthology.org/2024.findings-emnlp. 429/.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J.

Tianwen Wei, Jian Luan, Wei Liu, Shuang Dong, and Bin Wang. Cmath: Can your language model pass chinese elementary school math test?, 2023. URL https://arxiv.org/abs/2306. 16636.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106, Copenhagen, Denmark, 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4413. URL https://aclanthology.org/W17-4413.

Annie Wong, Thomas B¨ack, Aske Plaat, Niki van Stein, and Anna V. Kononova. Towards a deeper understanding of reasoning capabilities in large language models, 2025. URL https: //arxiv.org/abs/2505.10543.

Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, Zeyu Liu, Yiheng Xu, Hongjin SU, Dongchan Shin, Caiming Xiong, and Tao Yu. Openagents: An open platform for language agents in the wild. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= sKATR2O1Y0.

Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Haibo Hu, and Minxin Du. Unlearning isn’t deletion: Investigating reversibility of machine unlearning in llms, 2025a. URL https: //arxiv.org/abs/2505.16831.

Yi Xu, Bo Xue, Shuqian Sheng, Cheng Deng, Jiaxin Ding, Zanwei Shen, Luoyi Fu, Xinbing Wang, and Chenghu Zhou. Good idea or not, representation of llm could tell. ArXiv preprint, abs/2409.13712, 2024. URL https://arxiv.org/abs/2409.13712.

Zhangchen Xu, Yuetai Li, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. Tinyv: Reducing false negatives in verification improves rl for llm reasoning, 2025b. URL https://arxiv.org/abs/2505.14625.

Junxiao Yang, Jinzhe Tu, Haoran Liu, Xiaoce Wang, Chujie Zheng, Zhexin Zhang, Shiyao Cui, Caishun Chen, Tiantian He, Hongning Wang, Yew-Soon Ong, and Minlie Huang. Barrel: Boundary-aware reasoning for factual and reliable lrms, 2025. URL https://arxiv.org/ abs/2505.13529.

Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chainof-thought reasoning in llms. In ICML 2025, 2025. URL https://arxiv.org/abs/2502. 03373.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=N8N0hgNDRt.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MAmmoTH: Building math generalist models through hybrid instruction tuning. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview. net/forum?id=yLClGs770I.

Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen. MAmmoTH2: Scaling instructions from the web. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b. URL https://openreview.net/forum?id=yVu5dnPlqA.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/abs/2503.18892.

Yuxiang Zhang, Yuqi Yang, Jiangming Shu, Yuhang Wang, Jinlin Xiao, and Jitao Sang. Openrft: Adapting reasoning foundation model for domain-specific tasks with reinforcement fine-tuning. ArXiv preprint, abs/2412.16849, 2024. URL https://arxiv.org/abs/2412.16849.

Junhao Zheng, Xidi Cai, Shengjie Qiu, and Qianli Ma. Spurious forgetting in continual learning of language models. In The Thirteenth International Conference on Learning Representations,

2025. URL https://openreview.net/forum?id=ScI7IlKGdI.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Yixin Cao, Yang Feng, and Deyi Xiong (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 400–410, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-demos.38. URL https: //aclanthology.org/2024.acl-demos.38/.

Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. “going on a vacation” takes longer than “going for a walk”: A study of temporal commonsense understanding. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3363–3369, Hong Kong, China, 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1332. URL https://aclanthology.org/D19-1332.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. URL https://arxiv.org/abs/2311.07911.

Lexin Zhou, Lorenzo Pacchiardi, Fernando Mart´ınez-Plumed, Katherine M Collins, Yael MorosDaval, Seraphina Zhang, Qinlin Zhao, Yitian Huang, Luning Sun, Jonathan E Prunty, et al. General scales unlock ai evaluation with explanatory and predictive power. ArXiv preprint, abs/2503.06378, 2025. URL https://arxiv.org/abs/2503.06378.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. ArXiv preprint, abs/1909.08593, 2019. URL https://arxiv.org/abs/1909.08593.

A APPENDIX

CONTENTS

- A.1 The Use of Large Language Models . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.2 Detailed Evidence of Our Observed Phenomena . . . . . . . . . . . . . . . . . . . 18
- A.3 Full Evaluation Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- A.3.1 Post-training Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.3.2 Training Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.3.3 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.3.4 Evaluation Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.3.5 Evaluation metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- A.4 PCA Analysis under Varying Settings . . . . . . . . . . . . . . . . . . . . . . . . 21
- A.5 Details on Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- A.5.1 Gradient Norm of SFT and RL . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.5.2 Response Length of online RL and online SFT . . . . . . . . . . . . . . . 25

- A.6 Token Distribution Shifts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- A.6.1 Case study and details . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.6.2 Details about the result analysis . . . . . . . . . . . . . . . . . . . . . . . 30

- A.7 Breakdown Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- A.1 THE USE OF LARGE LANGUAGE MODELS

We employ large language models for language editing, which is limited to refining text to enhance readability. No language models contributed to the development of research ideas, analysis, models, or interpretation of results. We also used LLMs to validate part of our code to cover the edge cases.

- A.2 DETAILED EVIDENCE OF OUR OBSERVED PHENOMENA

As discussed in Section 2.1, we provided the complete evaluation for the Transferability Index for the off-the-shelf models on other reasoning and non-reasoning tasks in Table 5.

- A.3 FULL EVALUATION SETUP

- A.3.1 POST-TRAINING METHODS

Reinforcement Learning has recently proven effective at steering large language models toward complex, multi-step objectives by optimizing policies with scalar reward signals (Zeng et al., 2025). We used the Verl framework(Sheng et al., 2025) and GRPO (Shao et al., 2024) setup to fine-tune the Qwen-3-14B-Base model, utilizing answer correctness as the reward signal. Our RL uses a learning rate of 1×10−6 with an overall train batch size of 512 and clipping thresholds set between 0.22 and 0.28. We generate sequences up to 16k tokens long and perform 16 rollouts per prompt, then update the model in mini-batches of 128 samples. Both KL-divergence and entropy penalties are turned off (coefficients set to zero). We train the model for 140 steps and used the corresponding checkpoint.

Supervised Fine-Tuning remains a fundamental technique for adapting large pre-trained models by directly minimizing cross-entropy on high-quality datasets (Chu et al., 2025). We use the LLaMAFactory framework (Zheng et al., 2024), which is an extensible and user-friendly framework supporting multiple architectures and advanced optimization algorithms, to fine-tune our model on teachergenerated chain-of-thought traces. We use 5 × 10−5 as learning rate, the batch size is 512 and we train for 1.5 epoch to align with our RL settings.

- A.3.2 TRAINING DATASETS

As briefed in Section 2.2, our base training dataset is a curated set of 47K high-quality mathematics problems. We stratified the examples using two complementary sources: low-difficulty problems drawn from the DeepScaler dataset (Luo et al., 2025), and high-difficulty (levels 3–5) problems extracted from SimpleRL (Zeng et al., 2025). To generate CoT annotations, we prompt each problem into the Qwen3-32B-Instruct model (Team, 2025b) and use reject sampling to generate our dataset.

To further explore the effect of training data distribution for SFT-based reasoning models, we also distill a larger and more comprehensive dataset collected from General-Reasoner (Ma et al., 2025), which contains 232K examples across reasoning and non-reasoning tasks (e.g., Math, Chemistry, Business). This additional distilled set is used to train the General-Reasoner model using supervised fine-tuning.

- A.3.3 BASELINES

In our experiments, we compare against Qwen3-14B-Base model (Team, 2025b), which is the original Qwen3-14B model without any further adaptation. This serves as the unmodified backbone for all fine-tuning models. Also, we report the results of Qwen3-14B-Instruct model under our tested benchmarks. It is an instruction-tuned version of Qwen3-14B model trained on a large, generalpurpose instruction-following dataset. We evaluate it under two prompting modes:

- • think: prompts include a special <think> token to explicitly elicit chain-of-thought reasoning.
- • no-think: prompts are provided without the <think> token, relying solely on direct instruction-following behavior.

Due to the wide range and enormous training data, this model is considered to give the optimal outputs across tasks in the current 14B-series models.

To further validate our observation, we apply our controlled study pipeline also for GeneralReasoner (Ma et al., 2025), a reasoning model tuned by cold RL with training data containing not only math but more diverse domains. We distilled the dataset for SFT finetuning using the same rejection sampling method with their proposed dataset. The dataset contains 232K samples covering various reasoning and non-reasoning tasks. Then we finetune the Qwen3-14B base model using the distilled dataset and name the model as General-Reasoner-Qwen3-14B(SFT) to directly compare with the RL-based General-Reasoner for a fairer and more comprehensive controlled study towards SFT and RL. The Transferability Index results could be retrieved in Table 5, the observation also confirms our initial hypothesis.

- A.3.4 EVALUATION BENCHMARKS

In the experiment, we evaluated our model across a wide range of benchmarks. Notably, to explicitly reveal the transferability of reasoning models, we grouped them into three categories by their content:

Math Reasoning Datasets We collected the following datasets that are composed of mathematical problems, which means that they typically need a mathematical reasoning process to get the answer:

- • MATH500 (Hendrycks et al., 2021b): A curated subset of 500 problems sampled from the broader MATH dataset, covering topics like algebra, combinatorics, geometry, and number theory.
- • AIME: Problems drawn from the American Invitational Mathematics Examination (AIME) 2024 and 2025, each with 30 challenging short-answer questions requiring multistep reasoning.
- • OlympiadBench (He et al., 2024): Problems sourced from international olympiads (e.g., IMO and regional contests).

Other Reasoning Datasets We collected the following datasets that are mainly composed of general reasoning problems containing a wider range of subjects:

- • LiveCodeBench (Jain et al., 2025): It is a continuously updated, contamination-free coding benchmark. We used its second version.
- • GPQA-Diamond(Rein et al., 2024): It is a graduate-level question-answering dataset that contains multiple-choice questions in biology, physics, and chemistry. We followed its diamond split.
- • ACPBench (Kokel et al., 2025): It has 7 atomic reasoning tasks around 13 classical planning domains. We only used the multiple-choice problems.
- • HeadQA (Vilares & G´omez-Rodr´ıguez, 2019): Multiple-choice QA from healthcarespecialist certification exams, including questions across pharmacology, chemistry, nursery, psychology, biology, and medicine.

Non-reasoning Datasets We collected the following datasets that are mainly composed of problems with factual answers, which means that they do not need a reasoning process to give the answer:

- • CoQA(Reddy et al., 2019): It has 127K questions in dialogues over passages, focusing on maintaining context and coreference across turns.
- • IFEval (Zhou et al., 2023): It contains over 500 prompts, each embedding verifiable instructions. Evaluates strict vs. loose adherence to instructions.
- • HaluEval (Li et al., 2023): It contains human-annotated samples where models must distinguish factual content from hallucinations.
- • MC-TACO (Zhou et al., 2019): It is a multiple-choice benchmark designed to evaluate models’ temporal commonsense, covering duration, ordering, typical time, frequency, and stationarity.

- Table 5: Comparison of model transferability indices on other reasoning and non-reasoning tasks across different training paradigms (SFT vs. RL), model sizes (7B vs. 32B), and model families (Qwen vs. Llama).

Model Base Model Size Method TIother TInon SFT vs RL Deepseek-R1-Distill-Qwen-7B Qwen2.5-Math-7B-Base 7B SFT 60.4 -4.2

- OpenThinker2-7B Qwen2.5-7B-Instruct 7B SFT 52.0 -76.2

- OpenThinker3-7B Qwen2.5-7B-Instruct 7B SFT 44.5 -70.0 S1.1-7B Qwen2.5-7B-Instruct 7B SFT -65.5 -213.3 General-Reasoner-Qwen3-14B (SFT) Qwen3-14B-Base 14B SFT 77.8 -127.6 Mistral-Small-24B-Instruct-2501-Reasoning Mistral-Small-24B-Instruct-2501 24B SFT 22.6 -89.5 OpenThinker2-32B Qwen2.5-32B-Instruct 32B SFT 39.1 -53.5 S1.1-32B Qwen2.5-32B-Instruct 32B SFT 33.8 -62.2 LIMO-32B Qwen2.5-32B-Instruct 32B SFT 40.5 -27.3 Qwen2.5-1.5B-SimpleRL Qwen2.5-1.5B-Base 1.5B RL 85.6 9.5 SimpleRL-7B Qwen2.5-7B-Base 7B RL 15.4 34.4 Qwen2.5-14B-SimpleRL Qwen2.5-14B-Base 14B RL 95.9 101.5 General-Reasoner-Qwen3-14B (RL) Qwen3-14B-Base 14B RL 121.6 66.5 DAPO-Qwen-32B Qwen2.5-32B 32B RL 60.7 -39.5

7B vs 32B S1.1-7B Qwen2.5-7B-Instruct 7B SFT -65.5 -213.3 S1.1-32B Qwen2.5-32B-Instruct 32B SFT 33.8 -62.2 OpenThinker2-7B Qwen2.5-7B-Instruct 7B SFT 52.0 -76.2 OpenThinker2-32B Qwen2.5-32B-Instruct 32B SFT 39.1 -53.5

Qwen vs Llama Qwen2.5-7B-SimpleRL Qwen2.5-7B-Base 7B RL 15.4 34.4 Llama3.1-8B-SimpleRL Llama3.1-8B 8B RL 72.6 7.0

- A.3.5 EVALUATION METRICS

We used LLM-Harness (Gao et al., 2024) to evaluate the models’ performance on OlympiadBench, ACPBench, HeadQA, CoQA, HaluEval, MC-TACO and used Eval-Chemy (Raoof et al., 2025) MATH500, AIME24, AIME25, GPQA-Diamond, LiveCodeBench, IFEval. On MATH500, AIME24, AIME25, GPQA-Diamond, and LiveCodeBench, we used 0.6 as temperature, and 0.95 as top-p value. In our experiments, we used accuracy to evaluate the models’ performance. Specifically, for AIME24 and AIME 25, we averaged accuracy on 10 samples. For GPQA-Diamond, LiveCodeBench and MATH 500, our score is the average accuracy over 3 samples. Specifically, we used version 2 and overall accuracy on LiveCodeBench. For ACPBench, we only used multiple choices, and averaged the score for all 10 tasks as the final score. For OlympiadBench, we only used math queries in English, and thus categorized Olympiad as a math benchmark. For HaluEval, the performance is the accuracy averaged on 3 tasks with zero-shot. And for IFEval, we used strict instruction accuracy as the score. For OlympiadBench, ACPBench, HeadQA, CoQA, HaluEval, and IFEval, we used greedy sampling and sampled only once.

- Table 6: Comparison of mean PCA shift magnitudes on math, reasoning, and non-reasoning tasks across different training paradigms (SFT vs. RL), model sizes (7B vs. 32B), and model families (Qwen vs. Llama). Red-highlighted rows indicate models trained with RL, non-Qwen baselines, or models with larger size.

Model Base Model Size Method Math Other Non SFT vs RL

Deepseek-R1-Distill-Qwen-7B Qwen2.5-Math-7B 7B SFT 38.8 27.4 42.2

- OpenThinker2-7B Qwen2.5-7B-Instruct 7B SFT 388.8 444.0 5486.2
- OpenThinker3-7B Qwen2.5-7B-Instruct 7B SFT 409.9 484.2 5228.2 S1.1-7B Qwen2.5-7B-Instruct 7B SFT 9.8 15.5 205.7 Qwen3-14B Qwen3-14B-Base 14B SFT 40.4 13.9 129.8 General-Reasoner-Qwen3-14B (SFT) Qwen3-14B-Base 14B SFT 7.9 10.5 118.2 OpenThinker2-32B Qwen2.5-32B-Instruct 32B SFT 25.4 72.0 726.8 S1.1-32B Qwen2.5-32B-Instruct 32B SFT 2.4 1.8 7.5 LIMO-32B Qwen2.5-32B-Instruct 32B SFT 15.7 9.0 7.5 Qwen2.5-1.5B-SimpleRL-Zoo Qwen2.5-1.5B 1.5B RL 0.5 0.3 0.6 Qwen2.5-7B-SimpleRL-Zoo Qwen2.5-7B 7B RL 0.6 0.2 0.6 Llama-3.1-8B-SimpleRL-Zoo Llama-3.1-8B 8B RL 0.3 0.1 0.0 Qwen2.5-14B-SimpleRL-Zoo Qwen2.5-14B 14B RL 0.9 1.0 2.7 General-Reasoner-Qwen3-14B(RL) Qwen3-14B-Base 14B RL 2.9 4.6 64.8 DAPO-Qwen-32B Qwen2.5-32B 32B RL 2.6 1.1 1.1

###### 7B vs 32B

S1.1-7B Qwen2.5-7B-Instruct 7B SFT 9.8 15.5 205.7 S1.1-32B Qwen2.5-32B-Instruct 32B SFT 2.4 1.8 7.5 OpenThinker2-7B Qwen2.5-7B-Instruct 7B SFT 388.8 444.0 5486.2 OpenThinker2-32B Qwen2.5-32B-Instruct 32B SFT 25.4 72.0 726.8

###### Qwen vs Llama

Qwen2.5-7B-SimpleRL-Zoo Qwen2.5-7B 7B RL 0.6 0.2 0.6 Qwen3-14B-Base-SimpleRL-Zoo Qwen3-14B-Base 14B RL 0.9 1.0 2.7 Llama-3.1-8B-SimpleRL-Zoo Llama-3.1-8B 8B RL 0.3 0.1 0.0

- A.4 PCA ANALYSIS UNDER VARYING SETTINGS

Table 6 summarizes the mean d(∗) across math, other-reasoning, and non-reasoning tasks, providing an overall assessment of latent-space shifts under different training paradigms. Figure 7, 8, 9, 10

PCA Shift across Models and Tasks

Math

Other-Reasoning

Other-Reasoning

Non-Reasoning

Non-Reasoning

UniReason-Qwen2.5-7B (SFT); Base: Qwen2.5-7B

MATH500

Head_QA

Livecodebench

COQA

HalluEval

Base

1000

1000

1000

1000

1000

UniReason-7B SFT

d(*) = 32.98

d(*) = 102.17

d(*) = 1.92

d(*) = 3.23

d(*) = 386.45

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

0 250 500

0 250 500

0 250 500

0 250 500

0 250 500

UniReason-Qwen2.5-7B (RL); Base: Qwen2.5-7B

Base

1000

1000

1000

1000

1000

UniReason-7B RL

d(*) = 0.50

d(*) = 1.50

d(*) = 0.90

d(*) = 0.45

d(*) = 5.66

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

0 250 500

0 250 500

0 250 500

0 250 500

0 250 500

(PC1 )

(PC1 )

(PC1 )

(PC1 )

(PC1 )

- Figure 7: PCA shift of Qwen2.5-7B across different training methods and tasks. d(∗) is the Euclidean distance between representation centroids before and after training. The first row shows models trained with SFT , and the last row shows models trained with RL . RL training results in the smallest PCA shift for all task types, suggesting more stable latent representations.

illustrates the paradigm comparison between SFT and RL, Figure 11 contrasts model sizes (7B versus 32B), and Figure 12 compares model families (Qwen vs. Llama).

Impact of Training Paradigm. RL-based fine-tuning consistently results in lower PCA shifts than SFT across math, other-reasoning, and non-reasoning tasks. As shown in Table 6, models such as SimpleRL-7B and SimpleRL-14B exhibit significantly smaller feature shifts compared to their SFT-

PCA Shift across Models and Tasks

Math

Other-Reasoning

Other-Reasoning

Non-Reasoning

Non-Reasoning

s1.1-7B (SFT); Base: Qwen-2.5-7B-Instruct

MATH500

Head_QA

Livecodebench

COQA

HalluEval

Base

1000

1000

1000

1000

1000

s1.1-7B

d(*) = 17.03

d(*) = 53.75

d(*) = 1.95

d(*) = 1.19

d(*) = 204.22

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

200 0 200

200 0 200

200 0 200

200 0 200

200 0 200

Qwen2.5-7B-SimpleRL (RL); Base: Qwen-2.5-7B

Base

1000

1000

1000

1000

1000

SimpleRL-7B

d(*) = 0.43

d(*) = 0.11

d(*) = 0.63

d(*) = 0.09

d(*) = 0.52

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

200 0 200

200 0 200

200 0 200

200 0 200

200 0 200

(PC1 )

(PC1 )

(PC1 )

(PC1 )

(PC1 )

- Figure 8: PCA shift for Qwen-2.5-7B-Instruct and Qwen2.5-7B under different training paradigms and benchmarks. The first row corresponds to models trained with SFT , while the second row corresponds to the RL . The RL-based training paradigm achieves the lowest PCA shift across all task categories, indicating enhanced stability in the model’s latent representation space for math, other reasoning, and non-reasoning tasks. d(∗) denotes the Euclidean distance between the centroids of representations before and after training.

0 200

500

0

500

1000

(PC2)

d(*) = 12.00

Math

MATH500

Base

SFT

0 200

500

0

500

1000

d(*) = 25.81

Other-Reasoning

Head_QA

0 200

500

0

500

1000

d(*) = 13.77

Other-Reasoning

Livecodebench

0 200

500

0

500

1000

d(*) = 5.11

Non-Reasoning

COQA

0 200

500

0

500

1000

d(*) = 112.61

Non-Reasoning

HalluEval

0 200

(PC1 )

500

0

500

1000

(PC2)

d(*) = 1.04

Base

RL

0 200

(PC1 )

500

0

500

1000

d(*) = 10.81

0 200

(PC1 )

500

0

500

1000

d(*) = 3.40

0 200

(PC1 )

500

0

500

1000

d(*) = 4.07

0 200

(PC1 )

500

0

500

1000

d(*) = 59.84

PCA Shift across Models and Tasks

General-Reasoner-Qwen3-14B (SFT); Base: Qwen3-14B-Base

General-Reasoner-Qwen3-14B (RL); Base: Qwen3-14B-Base

- Figure 9: PCA shift for Qwen3-14B-Base under different training paradigms and benchmarks. The first row corresponds to models trained with SFT , while the second row corresponds to the RL . The RL-based training paradigm achieves the lowest PCA shift across all task categories, indicating enhanced stability in the model’s latent representation space for math, other reasoning, and nonreasoning tasks. d(∗) denotes the Euclidean distance between the centroids of representations before and after training.

trained counterparts. This observation is further visualized in Figure 7, 8, 9, 10, which demonstrates more concentrated representation shifts under RL. These findings are consistent with the phenomena discussed in Section 2.1, reinforcing that RL enhances generalization by better preserving internal representations. Overall, these results suggest that RL is substantially more effective at preserving general-domain representations and mitigating catastrophic forgetting.

Effect of Model Size. Increasing model size from 7B to 32B substantially reduces PCA shifts on math, other-reasoning, and non-reasoning tasks, as shown in Table 6. Figure 11 further confirms that larger models exhibit more concentrated and stable latent representations in non-reasoning domains,

PCA Shift across Models and Tasks

Math

Other-Reasoning

Other-Reasoning

Non-Reasoning

Non-Reasoning

LIMO-32B (SFT); Base: Qwen2.5-32B-Instruct

MATH500

Head_QA

Livecodebench

COQA

HalluEval

| | | | | |
|---|---|---|---|---|
| | | |(*) = 6.68| |
| | | |d| |
| | | | | |
| | | | | |
| | | | | |

Base

1000

1000

1000

1000

1000

LIMO-32B SFT

d(*) = 12.81

d(*) = 7.02

d(*) = 7.36

d(*) = 5.18

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

200 0 200

200 0 200

200 0 200

200 0 200

200 0 200

DAPO-Qwen-32B (RL); Base: Qwen2.5-32B

Base

1000

1000

1000

1000

1000

DAPO-32B RL

d(*) = 1.71

d(*) = 0.52

d(*) = 0.35

d(*) = 1.83

d(*) = 0.33

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

200 0 200

200 0 200

200 0 200

200 0 200

200 0 200

(PC1 )

(PC1 )

(PC1 )

(PC1 )

(PC1 )

- Figure 10: PCA shift for Qwen2.5-32B-Instruct and Qwen2.5-32B under different training paradigms and benchmarks. The first row corresponds to models trained with SFT , while the second row corresponds to the RL . The RL-based training paradigm achieves the lowest PCA shift across all task categories, indicating enhanced stability in the model’s latent representation space for math, other reasoning, and non-reasoning tasks. d(∗) denotes the Euclidean distance between the centroids of representations before and after training.

200 0 200

500

0

500

1000

(PC2)

d(*) = 17.03

Math

MATH500

Base

s1.1-7B

200 0 200

500

0

500

1000

d(*) = 53.75

Other-Reasoning

Head_QA

200 0 200

500

0

500

1000

d(*) = 1.95

Other-Reasoning

Livecodebench

200 0 200

500

0

500

1000

d(*) = 1.19

Non-Reasoning

COQA

200 0 200

500

0

500

1000

d(*) = 204.22

Non-Reasoning

HalluEval

200 0 200

(PC1 )

500

0

500

1000

(PC2)

d(*) = 1.96

Base

s1.1-32B

200 0 200

(PC1 )

500

0

500

1000

d(*) = 0.73

200 0 200

(PC1 )

500

0

500

1000

d(*) = 0.16

200 0 200

(PC1 )

500

0

500

1000

d(*) = 1.51

200 0 200

(PC1 )

500

0

500

1000

d(*) = 6.33

PCA Shift across Models and Tasks

s1.1-7B; Base: Qwen-2.5-7B-Instruct

s1.1-32B; Base: Qwen-2.5-32B-Instruct

- Figure 11: PCA shift for under different model sizes (7B vs 32B) and benchmarks. The first row corresponds to the 7B model, while the second row corresponds to the 32B model. d(∗) denotes the Euclidean distance between the centroids of representations before and after training.

PCA Shift across Models and Tasks

Math

Other-Reasoning

Other-Reasoning

Non-Reasoning

Non-Reasoning

Qwen2.5-7B-SimpleRL (RL); Base: Qwen2.5-7B

MATH500

Head_QA

Livecodebench

COQA

HalluEval

Base

1000

1000

1000

1000

1000

Qwen RL

d(*) = 0.43

d(*) = 0.11

d(*) = 0.63

d(*) = 0.09

d(*) = 0.52

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

25 0 25

25 0 25

25 0 25

25 0 25

25 0 25

Llama-3.1-8B-SimpleRL (RL); Base: Llama-3.1-8B

Base

1000

1000

1000

1000

1000

Llama RL

d(*) = 0.38

d(*) = 0.04

d(*) = 0.07

d(*) = 0.03

d(*) = 0.03

(PC2)

500

500

500

500

500

0

0

0

0

0

500

500

500

500

500

25 0 25

25 0 25

25 0 25

25 0 25

25 0 25

(PC1 )

(PC1 )

(PC1 )

(PC1 )

(PC1 )

- Figure 12: PCA shift under different model families (Qwen vs Llama) and benchmarks. The first row corresponds to the Qwen model, while the second row corresponds to the Llama model. d(∗) denotes the Euclidean distance between the centroids of representations before and after training.

indicating reduced feature drift. These findings suggest that scaling up model size is particularly effective for enhancing stability and resistance to latent space drift in general-domain tasks.

Effect of Model Family. The results in Table 6 and Figure 12 indicate that both RL-tuned Llama and Qwen models exhibit small PCA shifts. While Qwen displays similar shifts as Llama on reasoning tasks, it also has higher drift on other-reasoning and non-reasoning tasks. In contrast, Llama models show greater resilience to drift in general domains. These differences stem from the distinct architectures and pre-training data, suggesting that RL strategies should be tailored to each model’s characteristics.

Summary. Our analyses show that latent space drift is mainly determined by training paradigm, model size, and model family. RL-based fine-tuning is the most effective at preserving generaldomain representations and mitigating catastrophic forgetting. While larger models offer greater feature stability, especially on other-reasoning and non-reasoning tasks, scaling alone is insufficient. Qwen models remain sensitive to reasoning-related pre-training data, underscoring the need for tailored strategies. Overall, RL-based training is critical for developing robust language models that balance general-domain coverage with reasoning ability.

###### Model Type Early Slope t1/2 Total Grad. Late Avg. Late Var. Decay Rate Oscillation

14B-RL RL −0.005 22.0 48.190 0.448 0.041 −0.262 0.034 14B-SFT-no-think SFT −0.019 80.0 64.906 0.405 0.098 −0.592 0.368 14B-SFT-think SFT −0.030 80.0 64.187 0.383 0.112 −0.625 0.382

- Table 7: Gradient norm indicators in early training (first 100 steps). Smaller Early Slope values (closer to zero) show more gradual early updates. Lower Total Grad. suggests smaller overall movement in parameter space. Higher Late Avg. with lower Late Var. reflects a stable learning signal in later steps. A less negative Decay Rate shows slower reduction of gradient strength. Finally, smaller Oscillation values mean smoother training, with fewer rapid shifts. Across all metrics, RL shows more stable and generalizable training behavior than SFT.

- A.5 DETAILS ON ABLATION STUDY

- A.5.1 GRADIENT NORM OF SFT AND RL

As shown in Figure 13 and Table 7, RL and SFT exhibit distinct gradient dynamics. RL performs gentler early updates (Early Slope closer to zero) without the sharp spikes observed in SFT, and accumulates smaller overall movement in parameter space (lower Total Grad.). In later steps, RL maintains a stronger and more stable gradient signal (higher Late Avg., lower Late Var.), with a slower decay rate and much less oscillation. In contrast, SFT shows large initial fluctuations followed by rapid collapse toward near-zero gradients. Overall, RL demonstrates a more stable and persistent optimization process, whereas SFT tends to quickly saturate and lose effective gradient signals.

- A.5.2 RESPONSE LENGTH OF ONLINE RL AND ONLINE SFT

Compared to SFT, online RL produces a marked increase in response length as training progresses. This pattern reflects RL’s stronger exploration ability: negative examples discourage uninformative outputs, while credit assignment amplifies signals from task-relevant tokens. Together, these mech-

###### Grad Norm Comparison

2.5

2.0

1.5

GradNorm

1.0

0.5

| |
|---|

| |
|---|
| |

| |
|---|

0.0

0 20 40 60 80 100

Step

14B-RL 14B-SFT-no-think 14B-SFT-think

- Figure 13: Grad norm comparison across different fine-tuning strategies for the first 100 steps. RL exhibits consistently lower gradient norms compared to both SFT-no-think and SFT-think models, suggesting more stable optimization dynamics during fine-tuning.

#### Response Length Comparison

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| || |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
|
| | | | | | | | | |
| | | | | | | | | |

1600

1400

ResponseLength

1200

1000

800

0 10 20 30 40 50 60 70 80

Step

8B-online-RL 8B-online-SFT

- Figure 14: Response length comparison across online RL and SFT fine-tuning (first 80 steps). Online RL gradually increases response length during training, while SFT remains stable. This indicates that RL promotes stronger exploration, supported by mechanisms such as negative examples and credit assignment that guide the model to generate more diverse and extended outputs. Longer responses reflect broader policy exploration, which is often linked to better generalization and transfer across tasks.

anisms encourage the model to explore a wider solution space, leading to longer and more varied responses. Such exploration is important for improving transferability, as it enables the model to adapt its reasoning strategies beyond the narrow supervision provided in SFT.

- A.6 TOKEN DISTRIBUTION SHIFTS

- A.6.1 CASE STUDY AND DETAILS

This part provides the complete queries and responses in our case study in Section 4. Table 8 enumerates all shifted tokens for both reasoning and non-reasoning queries. RL modifies a compact set of task-salient tokens, whereas SFT shifts a much larger set including many off-topic tokens.

Case Study:

Reasoning Task Query:

Solve the following math problem. Present the final answer in the format: Final Answer: boxed{your answer}. Problem: Ten treeks weigh as much as three squigs and one goolee. Two treeks and one goolee are equal in weight to one squig. The combined weight of how many treeks equals the weight of one squig?

RL-tuned Models: Let’s define the variables for the weights of the treeks, squigs, and goolees. Let: - T be the weight of one treek, - S be the weight of one squig, - G be the weight of one goolee. We are given two pieces of information: 1. Ten treeks weigh as much as three squigs and one goolee: 10T = 3S+G 2. Two treeks and one goolee are equal in weight to one squig: 2T +G = S We need to find how many treeks combined equal the weight of one squig. In other words, we need to find a number x such that xT = S. First, let’s solve the second equation for G: 2T + G = S ⇒ G = S − 2T Now, substitute G = S − 2T into the first equation: 10T = 3S + (S − 2T) 10T = 3S + S − 2T 10T = 4S − 2T Now, add 2T to both sides to isolate the terms involving S: 10T + 2T = 4S 12T = 4S Now, divide both sides by 4 to solve for S: 3T = S This tells us that the weight of one squig is equal to the weight of three treeks. Therefore, the combined weight of three treeks equals the weight of one squig. Final Answer:

|3|
|---|

SFT-tuned Models: Okay, let’s see. I need to solve this problem where they talk about treeks, squigs, and goolees. The question is asking how many treeks combined equal the weight of one squig. First, let me parse the problem again. The first statement says ten treeks weigh as much as three squigs and one goolee. So, 10T = 3S + G, where T is treek, S is squig, G is goolee. The second statement is two treeks and one goolee equal one squig. So, 2T + G = S. The question is asking for how many treeks equal one squig. So, find x such that xT = S. Alright, so I have two equations:

- 1) 10T = 3S + G
- 2) 2T + G = S I need to solve these equations to find x where xT = S. Let me see. Maybe I can express G from the second equation and substitute into the first. From equation 2: G = S − 2T. Let me check that. If 2T + G = S, then subtract 2T from both sides: G = S − 2T. Yes, that’s right. Now substitute this into equation 1. So equation 1 becomes: 10T = 3S + (S − 2T) Simplify the right side: 3S + S − 2T = 4S − 2T So now the equation is 10T = 4S − 2T Let me add 2T to both sides: 10T + 2T = 4S ⇒ 12T = 4S Divide both sides by 4: 3T = S. Wait, so S = 3T? Wait, hold on. The question is asking for how many treeks equal one squig. If S = 3T, then 1 squig equals 3 treeks. Where did I go wrong? Let me check my equations again. The problem says: First statement: Ten treeks weigh as much as three squigs and one goolee. So 10T = 3S + G. Correct. Second statement: Two treeks and one goolee are equal in weight to one squig. So 2T + G = S. Correct. Then solving for G from the second equation: G = S − 2T. Substitute into the first equation: 10T = 3S + (S − 2T) ⇒ 10T = 4S − 2T ⇒ 12T = 4S ⇒ 3T = S. So S = 3T. Therefore, 1 squig equals 3 treeks. Maybe I misread the problem? Wait, maybe the problem is asking for how many treeks equal one squig, but in the equations, maybe I have to express T in terms of S. Let me try solving for T. From S = 3T, then T = S3 .

So one treek is 13 of a squig. Therefore, to get one squig, you need 3 treeks. So answer is 3. Am I wrong?

Rewrite equations:

- Equation 1: 10T = 3S + G
- Equation 2: 2T + G = S ⇒ G = S − 2T Substitute into equation 1: 10T = 3S + S − 2T = 4S − 2T ⇒ 12T = 4S ⇒ S = 3T Still the same. Try elimination form:

- Equation 1: 10T − 3S − G = 0
- Equation 2: 2T + G − S = 0 Add equations: 12T − 4S = 0 ⇒ S = 3T Still the same. Try plugging in values: assume T = 1, then S = 3, and from equation 2: 2(1) + G = 3 ⇒ G = 1 Check equation 1: 10 = 3(3) + 1 = 10, consistent.

|3|
|---|

.

Conclusion: equations are consistent, and answer is

## Non-Reasoning Task

Query: Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. ¡¡title¿¿. First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence).

RL-tuned Models: Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. ¡¡title¿¿.First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence).

—**Subject:**Resignation Notice Dear [Boss’s Name],

I hope this message finds you well. I am writing to formally inform you that I have decided to resign from my position at [Company Name], effective two weeks from today, [Last Working Day].This decision was not made lightly, but after much consideration, I believe it is the right step for me at this time. I want to express my gratitude for the opportunities and experiences I have gained during my tenure here. I have learned a great deal and have grown both professionally and personally. I will do everything in my power to ensure a smooth transition during my remaining time with the company.

Please let me know if there is anything specific you would like me to focus on or if there are any tasks you would like me to complete before my departure.Thank you once again for your support and understanding. I look forward to staying in touch and wish you and the team continued success.

Best regards, [Your Full Name] [Your Position] [Your Contact Information]”

SFT-tuned Models: Okay, so I need to write an email to my boss telling him I’m quitting. The email has to have a title wrapped in double angular brackets like ¡¡title¿¿. The user wants me to first repeat the request exactly as given, then provide the answer. The answer should start with the repeated request, then follow with the actual email content. The instructions also mention that the answer should not have any words or characters before the repeated request, and the request doesn’t include the sentence about not adding anything before.

Let me think. The original request is to write the email. So I need to make sure the email is professional, concise, and to the point. Since it’s a resignation, it’s important to be polite and express gratitude for the opportunities. But also, I need to state the decision clearly. The title should be something like ”Resignation Notice” or similar.

The structure should be straightforward. Start with the title, then the body. The body should include the date, a polite greeting, the resignation statement, the effective date, a thank-you note, and maybe a willingness to help with the transition. Then a closing.

I should avoid any negative comments or reasons for leaving, unless the user wants to include them. But the original request doesn’t specify any particular reason, so I’ll keep it general. The tone should be respectful and appreciative. Let me check the example again. The user says the answer should start with the repeated request. So the first part is the exact request, then the answer. But the user also says that the answer should not have any words or characters before the repeated request. So the answer starts right after the repeated request.

(3120 repetitive tokens)

So the answer should start with the exact request, then the email. The request is: ”Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. ¡¡title¿¿.” So the answer is:

Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. ¡¡title¿¿. [Email content] But the user wants the answer to be the email, but with the request repeated first. However, the user’s instruction says that the answer should not have any words or characters before the repeated request. So the answer starts with the repeated request, then the email.

But the user is the one who is asking me to generate the answer. So the answer is the email, but the first part is the request. This is a bit of a loop. Let me try to structure it properly. The user’s request is to write the email. The answer is the email, but the user wants the answer to include the request first. So the answer is:

First, the exact request: ”Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. ¡¡title¿¿.” Then, the generated email. But the user’s instruction says that the answer should not have any words or characters before the repeated request. So the answer starts with the repeated request, then the email.

But the user is the one who is asking me to generate the answer. So the answer is the email, but the first part is the request. This is a bit confusing. Let me try to follow the user’s instructions step by step.

The user says: ”First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)”. So the answer should”

- A.6.2 DETAILS ABOUT THE RESULT ANALYSIS

As mentioned in Section A.6, we provide additional results and visualizations on token ranks shifts across tasks, models and positions as shown in Figure 16, Figure 17, Figure 18, Figure 19 and Figure 20.

MATH-500

Olympiad

IFEval

###### GPQA-D

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

40

40

40

40

RankShift

S1.1-7B

20

20

20

20

0

0

0

0

0 2500 5000 7500

0 2500 5000 7500

0 2500 5000 7500

0 2500 5000 7500

Position Index

Position Index

Position Index

Position Index

MATH-500

Olympiad

IFEval

###### GPQA-D

OpenThinker2-7B

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 2000 4000 6000

0 2500 5000 7500

0 100 200 300

0 2500 5000 7500

Position Index

Position Index

Position Index

Position Index

MATH-500

Olympiad

IFEval

###### GPQA-D

OpenThinker3-7B

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 2000 4000

0 2500 5000 7500

0 2000 4000

0 2500 5000 7500

Position Index

Position Index

Position Index

Position Index

OpenThinker2-32B

MATH-500

Olympiad

IFEval

###### GPQA-D

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 2000 4000 6000

0 2000 4000 6000

0 1000 2000

0 2500 5000 7500

Position Index

Position Index

Position Index

Position Index

MATH-500

Olympiad

IFEval

###### GPQA-D

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

40

40

40

40

S1.1-32B

RankShift

20

20

20

20

0

0

0

0

0 1000 2000

0 2500 5000 7500

0 200 400

0 2500 5000 7500

Position Index

Position Index

Position Index

Position Index

MATH-500

Olympiad

IFEval

GPQA-D

R1-Distill-7B

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 500 1000 1500

0 2000 4000

0 500

0 500 1000

Position Index

Position Index

Position Index

Position Index

- Figure 15: Visualization of token rank shifts across different position indices for off-the-sheld SFT models.

- Table 8: Case study of shifted tokens for RL and SFT models. Completed queries and answers could be found in Appendix A.6. RL models selectively shift task-relevant or logic-token tokens (labeled in red). In contrast, SFT models exhibit extensive token shifting, including numerous queryirrelevant tokens. For example, non-reasoning queries inappropriately introduce reasoning-related tokens, leading to unnecessary overthinking that detracts from performance.

Domain Query Model Shifted Tokens

(Only 15 tokens experienced rank shift when decoded in the base model)

RL Model

In a Now Now define for number second 2 Now , add This

Ten treeks weigh as much as three squigs and one goolee. Two treeks and one goolee are equal in weight to one squig. How many treeks’ weight equals one squig?

(390 tokens experienced rank shift)

they The again conflicting but m Alternatively make have Hmm hold equations Wait For find check Let maybe using written Original pl contrad So There Wait solve I ’s Alternatively Alright so First solving a either check conflicting write Correct here another Like where ? Still From where The question / . The where here where equations Therefore problem check if was the ? equations together . answer I For or For Wait matrices this about m either and solve combined 1 problem ten Let . equation That If...

Reasoning Task

SFT Model

(Only 14 tokens experienced rank shift when decoded in the base model)

RL Model

Write << but >> Res formally much step grown will once Full

(158 tokens experienced rank shift)

Hmm Alright Wait Wait Wait try Another Maybe Another Alternatively Wait but Wait Wait Diamond On A check Who Starting user generate original ( original example make structure So follow The instructions user Let ( First ( check says doesn . to But willingness generated ’s : but says wants so has follow . . The structure the the first is But is structured with However who step like given repeated then also mention answer adding Let the . concise Since like straightforward . effective maybe wants But particular The answer the answer that would << The which original instruction which with the )". the first context . the providing Email of The The I first exactly then provide ...

Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets

Nonreasoning Task

SFT Model

- A.7 BREAKDOWN EVALUATION RESULTS

As discussed in the paper, we evaluated recent off-the-shelf models and our fine-tuned models across math reasoning, other reasoning and non-reasoning tasks. The detailed breakdown results are presented in Table 9, 10 and 11:

MATH-500

Olympiad

IFEval

###### GPQA-D

DAPO-Qwen-32B

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 500 1000

0 500 1000 1500

0 500 1000

0 500 1000

Position Index

Position Index

Position Index

Position Index

MATH-500

Olympiad

IFEval

###### GPQA-D

SimpleRL-7B

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 200 400 600

0 100 200

0 250 500 750

0 200 400

Position Index

Position Index

Position Index

Position Index

MATH-500

Olympiad

IFEval

###### GPQA-D

SimpleRL-14B

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 250 500 750

0 50

0 100 200

0 200 400

Position Index

Position Index

Position Index

Position Index

SimpleRL-llama-8B

MATH-500

Olympiad

IFEval

GPQA-D

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

40

40

40

40

RankShift

20

20

20

20

0

0

0

0

0 2500 5000 7500

0 2500 5000 7500

0 2500 5000 7500

0 100 200 300

Position Index

Position Index

Position Index

Position Index

- Figure 16: Visualization of token rank shifts across different position indices for off-the-sheld RL models.

MATH-500OlympiadGPQA-Q IFEval CoQA

Tasks

0

5

10

15

20

AvgTokenRankShift

8.25

11

7.18

20

6.43

SFT (no think)

MATH-500OlympiadGPQA-Q IFEval CoQA

Tasks

0

2

4

6

8

10

AvgTokenRankShift

7.13

8.99

5.88

4.38

1.94

SFT (think)

MATH-500OlympiadGPQA-Q IFEval CoQA

Tasks

0

5

10

15

20

AvgTokenRankShift

1.10

1.80 1.43

0.38 0.20

RL

- Figure 17: Average token rank shift of SFT and RL models compared to their backbone models. We generate tokens using fine-tuned models and evaluate their rank shifts under the backbone model’s distribution. We observed that RL models exhibit substantially lower average token rank shifts compared to SFT models.

###### UniReason-7B(SFT)

###### UniReason-7B(RL)

0.433

0.4

0.4

0.376

0.361

KLDivergence

###### KLDivergence

0.304

0.3

0.3

0.259

0.2

0.2

0.181

0.143

0.128 0.138

0.094

0.1

0.1

0.0

0.0

MATH-500OlympiadGPQA-Q IFEval CoQA

MATH-500OlympiadGPQA-Q IFEval CoQA

Tasks

Tasks

- Figure 18: KL divergence analysis of RL and SFT models. Higher KL divergence indicates greater distribution shifts from the original backbone model. We observe that RL models consistently exhibit significantly lower KL divergence compared to SFT models across different tasks, suggesting less distribution shift during training.

MATH-500Olympiad IFEval CoQA GPQA-D

Tasks

0.0

2.5

5.0

7.5

10.0

12.5

15.0

AvgTokenRankShift

12.9

13.9

9.4

13.4 13.3

UniReason-7B(SFT)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
|0.|2 0.|5 0.|8<br><br>0.|2 0.|5|

MATH-500Olympiad IFEval CoQA GPQA-D

Tasks

0.0

2.5

5.0

7.5

10.0

12.5

15.0

AvgTokenRankShift

UniReason-7B(RL)

- Figure 19: Average token rank shift of SFT and RL models compared to their backbone models. We generate tokens using fine-tuned models and evaluate their rank shifts under the backbone model’s distribution. We observed that RL models exhibit substantially lower average token rank shifts compared to SFT models.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 2000 4000

Position Index

0

20

40

RankShift

SFT(nothink)

MATH-500

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 2000 4000

Position Index

0

20

40

Olympiad

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 500 1000

Position Index

0

20

40

IFEval

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

0 2000

Position Index

0

20

40

GPQA-D

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 1000 2000

Position Index

0

20

40

RankShift

SFT(think)

MATH-500

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 1000 2000

Position Index

0

20

40

Olympiad

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 1000 2000

Position Index

0

20

40

IFEval

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 2000 4000

Position Index

0

20

40

GPQA-D

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0 500 1000 1500

Position Index

0

20

40

RankShift

RL

MATH-500

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 500 1000

Position Index

0

20

40

Olympiad

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0 250 500 750

Position Index

0

20

40

IFEval

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 1000 2000

Position Index

0

20

40

GPQA-D

- Figure 20: Visualization of token rank shifts across different position indices for both reasoning and non-reasoning tasks. We observe that RL models exhibit less token rank shifts while SFT models demonstrate substantial rank shifts across numerous positions throughout the sequence.

Table 9: Performance of off-the-shelf models on Math Reasoning Benchmarks

Model AIME24 AIME25 MATH500 Olympiad

Mistral-Small-24B-Instruct-2501 8.3 5.3 71.2 23.1 Mistral-Small-24B-Instruct-2501-Reasoning 63.7 43 89.6 41.9 Qwen2.5-1.5B-Base 0.0 0.0 27.4 6.6 Qwen2.5-1.5B-SimpleRL 0.3 0.0 60.1 11.9 Qwen2.5-Math-7B-Base 8.9 6.8 62.1 21.6 Deepseek-R1-Distill-Qwen-7B 57.3 40 87 37.2 Qwen2.5-7B-Base 3.3 1.6 59.1 11.6 SimpleRL-7B 15.7 7.6 69.8 24.1 Qwen2.5-7B-Instruct 12.0 9.7 72.8 29.6

- OpenThinker2-7B 56.3 39.6 88.4 39.1

- OpenThinker3-7B 67.8 56.6 90.0 44.1 S1.1-7B 23.3 13.3 75.8 28.7 Llama3.1-8B 0.0 0.0 13.1 2.1 Llama3.1-8B-SimpleRL 0.0 0.0 24.3 3.3 Qwen2.5-14B-Base 8.0 2.7 64.2 23.0 SimpleRL-14B 11.3 10.7 75.0 29.9 Qwen2.5-32B-Instruct 16.7 16.7 80.0 30.1 OpenThinker2-32B 76.3 57.7 94.2 43.0 S1.1-32B 59.0 44.0 92.8 41.6 LIMO-32B 56.7 46.0 86.4 42.2 Qwen2.5-32B 10.7 3.7 42.8 15.0 DAPO-Qwen-32B 57.0 34.7 88.4 39.3 Qwen3-14B-Base 13.0 9.3 60.4 27.9 Qwen3-14B (think) 79.0 67.7 92.0 44.7 Qwen3-14B (no-think) 27.3 21.3 82.2 36.1 General-Reasoner-Qwen3-14B (SFT) 35.0 22.6 78.4 30.6 General-Reasoner-Qwen3-14B (RL) 24.4 19.2 83.0 33.5 UniReason-Qwen3-14B-think (SFT) 52.0 37.0 85.0 25.0 UniReason-Qwen3-14B-no-think (SFT) 16.0 13.0 77.2 22.7 UniReason-Qwen3-14B (RL) 55.7 38.0 87.8 33.8

Table 10: Performance of models on Other Reasoning Benchmarks

Model GPQA LiveCodeBench2 ACPBench HeadQA

Mistral-Small-24B-Instruct-2501 49.3 26.4 62.9 45.6 Mistral-Small-24B-Instruct-2501-Reasoning 66.8 34.6 66.4 33.9 Qwen2.5-1.5B-Base 2.5 0.4 8.2 28.2 Qwen2.5-1.5B-SimpleRL 4.5 0.6 28.2 28.3 Qwen2.5-Math-7B-Base 29.8 14.7 42.7 27.9 Deepseek-R1-Distill-Qwen-7B 53.0 55.3 56.4 26.8 Qwen2.5-7B-Base 37.5 7.8 26.8 38.4 SimpleRL-7B 29.3 14.7 35.4 34.1 Qwen2.5-7B-Instruct 23.4 32.9 57.5 33.7 OpenThinker2-7B 47.6 55.6 66.1 31.0 OpenThinker3-7B 66.8 65.2 57.5 26.3 S1.1-7B 41.4 10.7 36.7 31.4 Llama3.1-8B 2.4 0.2 0.3 32.9 Llama3.1-8B-SimpleRL 5.6 0.5 15.4 33.7 Qwen2.5-14B-Base 49.2 15.3 30.2 37.3 SimpleRL-14B 39.1 37.9 60.4 38.4 Qwen2.5-32B-Instruct 40.4 49.0 72.1 38.6 OpenThinker2-32B 63.5 71.6 83.2 33.8 S1.1-32B 59.9 58.2 74.3 36.0 LIMO-32B 62.3 58.7 77.9 37.0 Qwen2.5-32B 33.8 28.6 42.1 38.4 DAPO-Qwen-32B 52.5 52.6 86.1 38.0 Qwen3-14B-Base 42.6 29.7 10.7 37.6 Qwen3-14B (think) 65.0 81.0 85.7 35.2 Qwen3-14B (no-think) 50.8 51.8 64.3 36.3 General-Reasoner-Qwen3-14B (SFT) 43.9 19.7 64.3 39.9 General-Reasoner-Qwen3-14B (RL) 56.1 32.9 75.0 44.0 UniReason-Qwen3-14B-think (SFT) 55.9 21.9 68.6 34.8 UniReason-Qwen3-14B-no-think (SFT) 48.7 23.5 69.3 35.0 UniReason-Qwen3-14B (RL) 57.7 40.6 65.4 40.2

Table 11: Performance of off-the-shelf models on Non-Reasoning Benchmarks

Model CoQA MC-TACO IFEval HalluEval

Mistral-Small-24B-Instruct-2501 31.5 76.2 81.8 72.4 Mistral-Small-24B-Instruct-2501-Reasoning 0.5 66.2 24.7 4.2 Qwen2.5-1.5B-Base 0.3 38.8 30.7 1.1 Qwen2.5-1.5B-SimpleRL 1.3 38.3 28.9 2.1 Qwen2.5-Math-7B-Base 0.6 46.6 32.2 5.1 Deepseek-R1-Distill-Qwen-7B 0.7 33.9 46.3 1.6 Qwen2.5-7B-Base 0.9 61.7 47.4 17.2 SimpleRL-7B 1.2 59.7 48.4 28.9 Qwen2.5-7B-Instruct 8.1 74.6 80.0 64.6 OpenThinker2-7B 1.3 53.0 47.7 0.1 OpenThinker3-7B 1.5 33.9 41.4 0.9 S1.1-7B 1.1 70.0 40.2 14.3 Llama3.1-8B 0.2 62.6 33.1 2.9 Llama3.1-8B-SimpleRL 0.7 60.3 36.0 1.8 Qwen2.5-14B-Base 0.5 66.7 57.2 24.5 SimpleRL-14B 1.0 68.9 63.1 68.8 Qwen2.5-32B-Instruct 9.4 75.3 79.1 77.5 OpenThinker2-32B 4.2 43.8 45.5 59.6 S1.1-32B 0.2 69.4 52.8 50.8 LIMO-32B 7.9 73.0 75.9 71.2 Qwen2.5-32B 7.3 76.0 83.8 52.1 DAPO-Qwen-32B 4.2 72.2 59.8 25.5 Qwen3-14B-Base 10.0 67.7 69.2 35.7 Qwen3-14B (think) 2.6 66.1 42.9 5.5 Qwen3-14B (no-think) 44.1 74.9 90.5 70.7 General-Reasoner-Qwen3-14B (SFT) 1.4 49.4 31.7 10.8 General-Reasoner-Qwen3-14B (RL) 4.8 51.19 72.0 55.7 UniReason-Qwen3-14B-think (SFT) 1.7 38.2 42.3 2.3 UniReason-Qwen3-14B-no-think (SFT) 5.3 66.1 41.4 3.3 UniReason-Qwen3-14B (RL) 28.2 74.0 70.0 40.7

