# arXiv:2510.00492v2[cs.AI]2Oct2025

## RETHINKING REWARD MODELS FOR MULTI-DOMAIN TEST-TIME SCALING

#### Dong Bok Lee1,2∗† Seanie Lee1* Sangwoo Park1 Minki Kang1 Jinheon Baek1 Dongki Kim1 Dominik Wagner3 Jiongdao Jin1 Heejun Lee1,4 Tobias Bocklet3 Jinyu Wang2 Jingjing Fu2 Sung Ju Hwang1,4 Jiang Bian2 Lei Song2 1KAIST 2Microsoft Research Asia 3TH N¨urnberg 4DeepAuto.ai

- 1{markhi,lsnfamily02,sjhwang82}@kaist.ac.kr
- 2{wang.jinyu,jifu,jiang.bian,lei.song}@microsoft.com

ABSTRACT

The reliability of large language models (LLMs) during test-time scaling is often assessed with external verifiers or reward models that distinguish correct reasoning from flawed logic. Prior work generally assumes that process reward models (PRMs), which score every intermediate reasoning step, outperform outcome reward models (ORMs) that assess only the final answer. This view is based mainly on evidence from narrow, math-adjacent domains. We present the first unified evaluation of four reward model variants, discriminative ORM and PRM (dORM, dPRM) and generative ORM and PRM (gORM, gPRM), across 14 diverse domains. Contrary to conventional wisdom, we find that (i) dORM performs on par with dPRM, (ii) gPRM is not competitive, and (iii) overall, gORM is the most robust, yielding significant and consistent gains across every tested domain. We attribute this to PRM-style stepwise scoring, which inherits label noise from LLM autolabeling and has difficulty evaluating long reasoning trajectories, including those involving self-correcting reasoning. Our theoretical analysis shows that step-wise aggregation compounds errors as reasoning length grows, and our empirical observations confirm this effect. These findings challenge the prevailing assumption that fine-grained supervision is always better and support generative outcome verification for multi-domain deployment. We publicly release our code, datasets, and checkpoints at https://github.com/db-Lee/Multi-RM to facilitate future research in multi-domain settings.

1 INTRODUCTION

Test-time scaling (TTS) enables large language models (LLMs) to generate diverse, reliable solutions, i.e., chain-of-thought (CoT; Wei et al., 2022; Kojima et al., 2022; Yao et al., 2023b; Madaan et al., 2023) and has shown impressive results on challenging reasoning tasks (Yao et al., 2023a; Snell et al., 2025; Wu et al., 2024). A widely adopted TTS approach uses external verifiers that select the best among the candidates (Snell et al., 2025): these verifiers, outcome reward models (ORMs), are typically implemented as discriminative classifiers that assign a scalar reward to a CoT (Cobbe et al., 2021; Uesato et al., 2022; Yu et al., 2024). ORMs are trained only on outcome-level signals, which are often coarse. Recent work has introduced process reward models (PRMs; Lightman et al., 2024; Wang et al., 2024a; Setlur et al., 2025; Zheng et al., 2024) that score each step of a CoT and aggregate the scores into a trajectory-level reward. Supervised with high-quality, carefully constructed process labels, e.g., manual annotation (Lightman et al., 2024) or Monte Carlo rollouts (Wang et al., 2024a), PRMs have been shown to outperform ORMs when combined with TTS.

Beyond discriminative verifiers, several studies have shown that the generative ability of LLMs can improve CoT verification, such as LLM-as-a-judge (Wang et al., 2023; Liu et al., 2023; Zheng et al., 2023). Based on this idea, other works fine-tune LLMs to generate a verification rationale for a CoT and compute the final reward from token probabilities (Zhang et al., 2025a; Khalifa et al., 2025; Zhao et al., 2025). To obtain verification CoTs for training, most previous work adopts consensusfiltering: (i) generate verification CoTs, and (ii) retain the verification CoT if its parsed verdict

∗Equal contribution. †Work done during an internship at Microsoft Research Asia.

aligns with outcome or process labels. After training, these generative verifiers have shown strong performance in math-adjacent reasoning tasks, outperforming discriminative verifiers.

However, most of the research efforts on TTS with external verifiers have been devoted primarily to math-adjacent domains. This narrow scope limits the potential for LLM deployment in highstakes real-world applications, such as legal (Guha et al., 2023; Cui et al., 2023; Fei et al., 2024) and medical (Singhal et al., 2023; Kung et al., 2023; Singhal et al., 2025) domains, where trustworthiness is paramount and rigorous verification of LLM outputs is especially important. Recently, Zeng et al. (2025) proposed multi-domain PRMs trained on the graduate level benchmark (MMLU-Pro; Wang et al., 2024c), covering 14 diverse domains, and showed that multi-domain training for PRMs significantly improves TTS performance across diverse domains. However, the study is limited to discriminative PRMs and the broader potential of different verifier types (e.g., ORMs vs. PRMs, discriminative vs. generative) in the multi-domain setting still remains underexplored.

To this end, we present the first unified evaluation of four verifier variants, discriminative ORM and PRM (dORM, dPRM), and generative ORM and PRM (gORM, gPRM), across 14 diverse domains. We rigorously review these variants in §2 and, under controlled conditions, evaluate them on math and multi-domain benchmarks (PRM800K, ProcessBench, and MMLU-Pro; Lightman et al., 2024; Zheng et al., 2024; Wang et al., 2024c) in §3. In the math domain, trends across the four variants are consistent with previous works (Lightman et al., 2024; Zhang et al., 2025a; Khalifa et al., 2025). dPRM outperforms dORM, and generative variants outperform discriminative ones. In the multi-domain, however, we observe contrasting results. dORM performs on par with dPRM, gPRM is not competitive, and overall, gORM delivers consistent and significant gains over the others.

Question: The function f(x) satisfies f(2x + 3) = 2f(x) + 3 for all x. If f(0) = 6, what is the value of f(9)?

###### "Aha" Moment

###### ⚠ Wrong Correct

Let's start from the know value f(0) = 6.

First, finding f(3): Set x = 0 ... f(3) = 15.

Second, find f(6): Set x = 0 ... f(6) = 33.

Wait, to find f(9), all we need to do is using f(3) ...

###### gORMCoT

<think> Step 1: Starting from ... Step 3: No evidence for f(6) ... </think>

###### gPRM

- Step 1: Yes
- Step 2: Yes
- Step 3: No

Not Evaluated

Is the solution correct? No

<think> Starting from f(0) ... But is f(6) = 33? ... However, f(9) is correct. </think>

Is the answer correct? Yes

Figure 1: Evaluating CoTs using gORM and gPRM.

In §4, we identify two factors underlying the failure of gPRM. First, on more difficult multi-domain problems, LLMs tend to produce longer CoTs which PRMs struggle to evaluate. As illustrated in Fig. 1, stepwise aggregation in PRMs often fails to reward long CoTs that recover from earlier errors (“aha” moments; Guo et al., 2025), because verification stops at the first mistake. In §4.1, we analyze how this PRM-style aggregation compounds errors as the chain length increases, and confirm this effect with our empirical results. Second, label noise is prevalent in multi-domain datasets. Given that step annotation in specialized domains is costly, prior work such as Zeng et al. (2025) depends on LLM-based auto-labeling, which can introduce noise. In §4.2, we show that, under a simulated label-noise analysis in the math domain, dORM is particularly sensitive to noisy step labels, whereas gORM remains robust. We further attribute the degradation of gPRM in the multi-domain setting, despite its robustness to noise in the math domain, to a severe shift in its CoT-length distribution induced by consensus filtering. Based on this analysis, we present practical guidelines for selecting among the four variants and discuss limitations and future work in §5.

Our contributions and findings are summarized as follows:

- • We present the first unified and controlled evaluation of four verifier variants, dORM, dPRM, gORM, and gPRM, across 14 diverse domains.
- • In contrast to conventional wisdom in math, we observe that (i) dORM performs similarly to dPRM, (ii) gPRM is not competitive; and (iii) overall, gORM delivers consistent gains over the others.
- • To explain the empirical observations, we provide two perspectives: (i) a theoretical analysis, with empirical support, showing that PRM risk increases with CoT length; and (ii) evidence of process label risk in the multi-domain setting with length-distribution shift induced by consensus filtering.
- • To facilitate future research in multi-domain settings, we publicly release our (i) code, (ii) datasets, and (iii) model checkpoints at https://github.com/db-Lee/Multi-RM.

2 BACKGROUND AND RELATED WORK

In this section, we review background and related work. We first formalize notation and test-time scaling in §2.1, and then discuss reward-model variants in §2.2, summarized in Fig. 2.

[Figure 1]

Figure 2: Conceptual illustration of reward models: r2 is the first incorrect step; the final answer is correct.

- 2.1 PROBLEM FORMULATION

Notation. For a given question q with the corresponding ground-truth (GT) answer a, we leverage the reasoning ability of large language models (LLMs) to reliably predict a by generating a chainof-thought (CoT), i.e., r1:T := (r1,...,rT) ∼ pLLM(· | q). Following Zeng et al. (2025), we segment the reasoning steps r1:T using the delimiter “\n\n”, where T is the number of reasoning steps. Let x := (q,r1:T) ∈ X, where X denotes the space of questions and reasoning chains, and let x1:t := (q,r1:t) be the prefix up to the t-th step. We consider two types of labels: (1) the outcome label y = (ˆa(r T) = a) ∈ {0,1}, where aˆ(rT) is the predicted answer parsed from the last reasoning step rT and is the indicator function; and (2) the process labels z 1:T = (z1,...,zT) ∈ {0,1}T, where each zt indicates whether the corresponding reasoning step rt is correct. Note that y represents the correctness label for the last reasoning step, so y = zT.

Test-time scaling (TTS) with reward models. Reward models have many applications, including LLM training via reinforcement learning (Ziegler et al., 2019; Ouyang et al., 2022; Achiam et al.,

- 2023; Dubey et al., 2024; Team et al., 2024; Yang et al., 2025), preference labeling (Dong et al.,
- 2024; Meng et al., 2024; Adler et al., 2024), rejection sampling (Gulcehre et al., 2023; Dong et al.,

- 2023), and data filtering (Dubey et al., 2024; Albalak et al., 2024; Yang et al., 2025). In this work, we focus on parallel or sampling-based (Wu et al., 2024) TTS with reward models, such as Best-ofN (BoN; Charniak & Johnson, 2005; Khalifa et al., 2023; Snell et al., 2025), which allocates more compute at test time (i.e., generates N CoTs) and selects the candidate aˆ(rT(i⋆)) with the the reward:

i⋆ = arg max

i∈{1,...,N}

f(x(i)), where x(i) := (q,r1:(i)T), and r1:(i)T i.i.d.∼ pLLM(· | q). (1)

Here, f : X → [0,1] is the true (unknown) reward function that assigns higher scores to CoTs that yield more reasonable and correct answers. However, f is unknown, so we train an external verifier fˆ: X → [0,1] to approximate f and use fˆas a surrogate in Eq. 1, which are detailed in §2.2.

- 2.2 REWARD MODELS Discriminative outcome reward model (dORM). Early studies on reward models (Cobbe et al.,

2021; Uesato et al., 2022; Yu et al., 2024) train a binary classifier fˆdORM : X  → [0,1] on outcome labels y ∈ {0,1} only, without requiring the intermediate process labels (z1,...,zT−1). Specifically, they sample CoTs and answers for given questions, construct a training dataset DdORM := {(x,y)}, and train fˆdORM with the binary cross-entropy (BCE) loss to approximate true p(y = 1 | x):

1 |DdORM|

ℓBCE f ˆdORM(x),y , (2)

LdORM :=

(x,y)∈DdORM

with ℓBCE(x,y) = − y log x + (1 − y)log(1 − x) . dORM considers only outcome correctness and ignores step-wise accuracy, making its reward signal potentially less faithful than f(x) in Eq. 1.

Discriminative process reward model (dPRM). dPRM seeks to improve the reward signal by training on fine-grained feedback for intermediate reasoning steps, i.e., process labels z1:T. For dPRM, the quality of these labels is the primary factor. Accordingly, prior work has proposed collecting process labels for sampled CoTs via manual annotation (Lightman et al., 2024), Monte Carlo (MC) rollouts (Wang et al., 2024a), automatically generated labels from LLMs (Zeng et al., 2025), or combinations thereof (Zhang et al., 2025b). After collecting the process labels, we construct the training set DdPRM := {(x,z1:T)} and train fˆdPRM using the BCE loss at each step:

T′

1 |DdPRM|

1 T′

ℓBCE f ˆdPRM(x1:t),zt , (3)

LdPRM :=

t=1

(x,z1:T )∈DdPRM

where T′ is the first incorrect reasoning step, i.e., T′ := min({t ∈ {1,...,T} : zt = 0} ∪ {T}). Training up to the T′-th step reflects a common assumption in the literature (Lightman et al., 2024; Wang et al., 2024a; Zheng et al., 2024; Zeng et al., 2025): once a reasoning step is incorrect, subsequent steps are also incorrect, i.e., if zt = 0 then zt′ = 0 for all t′ ∈ {t + 1,...,T}. At test time, we approximate f in Eq. 1 by aggregating the step rewards with the minimum (Zeng et al., 2025).

LLM-as-a-judge (Wang et al., 2023; Liu et al., 2023; Zheng et al., 2023) shows that the taskgeneralization ability of LLMs can extend to verification (i.e., zero-shot CoT verification). However, LLMs often overthink (Bavaresco et al., 2025) and, without additional training, remain practically limited (Zheng et al., 2024), implying the need of LLMs explicitly trained for verification.

Generative outcome reward model (gORM). Zhang et al. (2025a) proposed gORM, trained to generate a verification CoT together with a binary verdict, e.g., “Verification: Is the answer correct? Yes” or “No”. Because GT verification CoTs are unavailable, they synthesize training data via a consensus-filtering mechanism (Wang et al., 2024b; Zhu et al., 2025). We first sample a verification CoT and verdict from an LLM-as-a-judge, i.e., v1:L ∼ pLLM-j(· | x) using the prompt format in Fig. 20. Here, v1:L ∈ VL denotes the verification-CoT token sequence (including the verdict tokens), V is the vocabulary, and let yˆ ∈ {0,1} be the parsed verdict (1 for “Yes”, 0 for “No”). We then include (x,v1:L) in the training set DgORM only if yˆ agrees with the known outcome label y. We train pgORM with the next-token prediction over verification CoTs v1:L:

L

1 |DgORM|

1 L

−log pgORM(vi | x,v<i). (4)

LgORM :=

i=1

(x,v1:L)∈DgORM

−log pgORM is implemented as the cross-entropy loss over V. At test time, we approximate f with:

M

1 M

pgORM y = 1 | x,v1:(iL) , (5)

fˆgORM(x) := Ev

1:L∼pgORM(·|x) pgORM(y = 1 | x,v1:L) ≈

i=1

where v1:(iL) i.i.d.∼ pgORM(· | x). Here, the expectation is approximated with M MC samples and the model’s normalized probability of predicting the verdict “Yes” at the last verdict position:

pgORM(“Yes” | x,v1:(L−1)) pgORM(“Yes” | x,v1:(L−1)) + pgORM(“No” | x,v1:(L−1))

. (6)

pgORM(y = 1 | v1:L,x) :=

Generative Process Reward Model (gPRM). Beyond gORM, Khalifa et al. (2025) proposed gPRM, which is trained to generate verification CoTs v1:L with stepwise process verdicts, e.g., “Step t: The step is \boxed{correct}” or “\boxed{incorrect}”. Let the predicted verdict sequence be zˆ1:T′ ∈ {0,1}T

′

, defined up to the first predicted incorrect step T′1. Fol-

lowing Khalifa et al. (2025), we append a final verdict prompt, yielding the token sequence v1:L+ by concatenating either “Is the solution correct? Yes” or “No”—“Yes” if all predicted

process labels are 1 (zˆ1:T′ = 1T′), and “No” otherwise. We then construct DgPRM := {(x,v1:L+)} only when the predicted prefix agrees with the GT (zˆ1:T′ = z1:T′). We train pgPRM with v1:L+:

L+

1 L+

1 |DgPRM|

−log pgPRM(vi | x1:T′,v<i). (7)

LgPRM :=

i=1

(x,v1:L+)∈DgPRM

We condition on x1:T′ rather than the full input x for training (Khalifa et al., 2025), since the model pgPRM is prompted to stop verification once it reaches the first incorrect step, analogous to the datageneration process (Fig. 22). At test time, consistent with Eqs. 5 and 6, we approximate f in Eq. 1 by sampling from pgPRM and computing the normalized probability of a positive final verdict:

M

1 M

pgPRM y = 1 | x,v1:(iL)

fˆgPRM(x) := Ev

, (8)

1:L+∼pgPRM(·|x)[pgPRM(y = 1 | x,v1:L+)] ≈

+

i=1

pgPRM(“Yes” | x,v1:(L+−1)) pgPRM(“Yes” | x,v1:(L+−1)) + pgPRM(“No” | x,v1:(L+−1))

, (9)

pgPRM(y = 1 | x,v1:L+) :=

1As shown in Fig. 21, when generating verification CoTs for gPRM (i.e., v1:L ∼ pLLM-j(· | q)), Khalifa et al.

(2025) instruct the LLM-as-a-judge pLLM-j to stop once it detects the first incorrect step.

Majority Voting dORM dPRM gORM gPRM

###### 1.5B - GSM8K

1.5B - Math

1.5B - Omni

1.5B - Olympiad

###### 1.5B - Overall

84

| |
|---|
| |
| |
| |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

85

88

72

80

78

80

80

66

72

72

75

72

60

66

64

F1score(%)

70

###### 7B - GSM8K

7B - Math

7B - Omni

7B - Olympiad

###### 90 7B - Overall

96

80

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

88

90

84

88

72

84

80

78

80

64

78

72

72

72

56

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of verification CoTs (M)

Figure 3: Outcome verification results on ProcessBench in the math domain.

1.5B - GSM8K

1.5B - Math

###### 7B - GSM8K

7B - Math

Taksaccuracy(%)

84

84

90

| | | | | |
|---|---|---|---|---|
| | | | | |
| || |
|---|
<br><br>| | | |
| | | | | |
| | | | | |

85

78

78

85

72

72

80

| |
|---|

80

| |
|---|

| |
|---|

66

66

75

75

| |
|---|

| |
|---|

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

Figure 4: Best-of-N results using Qwen2.5-7B-Instruct on GSM8K and Math in the math domain.

i.i.d.∼ pgPRM(· | x) and we now condition on the full input x at test time (Khalifa et al., 2025). Byeond Khalifa et al. (2025), Zhao et al. (2025) also proposed a gPRM with code verification and more advanced training; however, it does not directly extend to multi-domain data (e.g., legal or medical domains), so we follow the approach of Khalifa et al. (2025) in this work.

where v1:(iL)

+

- 3 EXPERIMENTS

In this section, we evaluate dORM, dPRM, gORM, and gPRM in the math domain and the multidomain setting. We detail experimental setups (§3.1), and present experimental results (§3.2).

- 3.1 EXPERIMENTAL SETUPS

Math Datasets. For the math domain, we use PRM800K (Lightman et al., 2024) for training, where the process labels z1:T are human-annotated. As a testbed, we use ProcessBench (Zheng

- et al., 2024) with four splits: GSM8K/Math/Omni-Math/OlympiadBench. We generate N=16 CoTs per question in GSM8K and Math with Qwen2.5-7B-Instruct (Team, 2024a) for TTS; since we only seek to verify that a controlled evaluation reproduces prior findings, we restrict TTS to this setting.

Multi-domain datasets. Following Zeng et al. (2025), we adopt MMLU-Pro (Wang et al., 2024c), a 10-choice benchmark spanning 14 domains. For training/evaluation of reward models, each question is paired with 16/128 CoTs generated by Llama-3.1-8B-Instruct (Dubey et al., 2024), where process labels z1:T are automatically annotated by Llama-3.1-70B-Instruct. To assess generalization across different pLLM, we generate N=16 CoTs per question using SmolLM3-3B (Bakouch

- et al., 2025), Qwen2.5-7B-Instruct, gemma-2-9b-it (Team et al., 2024), and Llama-3.1-70B-Instruct. We defer further details, including prompts and dataset statistics to §C.

Implementation details. For reward-model backbones, we use R1-Distill models (Guo et al.,

- 2025): Qwen-1.5B and Qwen-7B/Llama-8B and Qwen-14B in the math/multi-domain settings. For prompt templates of gORM/gPRM, we follow Zhang et al. (2025a)/Khalifa et al. (2025) (Figs. 20 and 22). We optimize reward models using AdamW (Loshchilov & Hutter, 2019) with LoRA (Hu et al., 2022). Following Zeng et al. (2025), we limit the sequence length to 750 and 650 tokens of gORM/gPRM for the math and multi-domain, respectively. For gORM and gPRM, we sample M=16/10 verification CoTs (cf. Eqs. 5 and 8) in the math/multidomain settings, using vLLM (Kwon et al., 2023). See §D and Tab. 2 for more details and https://github.com/db-Lee/Multi-RM for all relevant artifacts, such as (i) code, (ii) datasets, and (iii) model checkpoints.

Verification CoTs. Following Zhang et al. (2025a) and Khalifa et al. (2025), we construct verification–CoT datasets for gORM and gPRM by prompting QwQ-32B (Qwen Team, 2025) with the

Law

Psychology

Chemistry

Biology

###### Overall

63

96 93.2

84

89.3

61.1

81.6

93.1

81.9 81.7 81.3

90

90.9

79.4

82

92

79.8

60

80

80

56.7

88

75.4

80

78.7

57

67.6

55.4

76

70

54.8

73.0

83.0

63.0

84

78

54

Physics

History

Economics

Math

Business

96

F1score(%)

90.3

74.4

94.5

88.6

88.1

96

90

75

88

88

71.8 72.5

84.6

90

84

83.5 83.2

72

76.9

80

84

72.2

76.1

68.5

81.5

84

78

80.2

69.3

72

79.9 80.1

74.1

69

80

Philosophy

Health

Engineering

Computer Science

Other

90 80.8

79.2

84.3

77.5

69.0

80

70 68.4

74.1

85

78 76.8

76.6

78.2

75

80.8

77.6

68

78

66.2

80

76

56.0

60

74.1

75.4

66

74.7

64.5

73.9

76

42.9

75

74

45

64

dO dP gO gP

dO dP gO gP

dO dP gO gP

dO dP gO gP

dO dP gO gP

Figure 5: Outcome verification results on MMLU-Pro in the multi-domain setting.

Majority Voting dORM dPRM gORM gPRM

- 54

60

66

72

| |
|---|

Overall

35

40

45

Law

60

65

70

Psychology

50

60

70

| |
|---|

Chemistry

72

78

84

| |
|---|

Biology

50

60

70

80

| |
|---|

Physics

48

52

56

History

60

66

72

78

Economics

50

60

70

80

| |
|---|

Math

56

64

72

| |
|---|

Business

1 2 4 8 16

45

50

- 55

Taskaccuracy(%)

Philosophy

Health

Engineering

Computer Science

Other

64

60

68

72

65

56

| |
|---|

| |
|---|

64

64

60

| |
|---|

| |
|---|

| |
|---|

48

60

56

55

40

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

Figure 6: Best-of-N results using Llama-3.1-8B-Instruct on MMLU-Pro in the multi-domain setting.

formats in Figs. 20 and 21. We discard any verification CoT whose parsed labels are inconsistent with the targets (e.g., y or z1:T), corresponding to the consensus filtering in §2.2. The training sets of gORM/gPRM contain 34,286/35,666 and 171,780/94,156 verification CoTs for the math and multi-domain settings. See §D and Figs. 23 and 24 for more details and examples.

- 3.2 EXPERIMENTAL RESULTS

Math-domain results. First, we evaluate the four verifier variants in the math domain. We compare outcome-verification performance with a 0.5 decision threshold, i.e., yˆ := ( fˆ(x) > 0.5). Fig. 3 reports F1 score (%) on ProcessBench splits. dPRM outperforms dORM overall, consistent with prior findings (Lightman et al., 2024), and shows a slight drop in Omni-Math/OlympiadBench with 7B backbones. For gORM/gPRM, the overall performance improves with M. At small M, gPRM may lag behind discriminative models (e.g., OlympiadBench). gORM generally outperforms gPRM (except 7B-GSM8K), and the gap widens on Omni-Math/OlympiadBench.

Although TTS has been well studied in the math domain, evaluations are not fully controlled: (i) models are rarely compared with a shared backbone, and (ii) gORM and gPRM have not been directly compared. We therefore evaluate the reward models with BoN under controlled conditions. As shown in Fig. 4, and consistent with the findings of Lightman et al. (2024), dPRM outperforms dORM. Notably, dORM even underperforms majority voting (MV) with 1.5B backbones, demonstrating the limitations of coarse outcome-level supervision. In line with Zhang et al. (2025a) and Khalifa et al. (2025), generative models outperform discriminative ones, with gORM slightly surpassing gPRM.

Multi-domain results. Next, we compare the four variants in the multi-domain setting. Fig. 5 reports F1 scores (%) for outcome-verification, with a 0.5 decision threshold, using R1-Distill-Qwen14B as the reward model backbone. dO/dP/gO/gP denote dORM/dPRM/gORM/gPRM. In contrast to the math domain results in Fig. 3, ORM variants achieve higher F1 scores than PRM variants.

Majority Voting dORM dPRM gORM gPRM

SmolLM3-3B

Qwen2.5-7B-Instruct

gemma2-9b-it

Llama3.1-70B-Instruct

Taskaccuracy(%)

76

| | | | | | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| | | | |
| || |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| | | | |
| || |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |

64

64

68

60

56

72

| |
|---|

64

| |
|---|

56

48

60

68

40

52

56

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

Figure 7: Overall Best-of-N results using four different pLLM on MMLU-Pro in the multi-domain setting.

Law

Psychology

Chemistry

Biology

###### Overall

0

0

5

+0.1

+0.2 +0.2

+2.5

-0.3 -0.9

0

-1.2

+0.5

0

-0.6

3

0

3

-3.9

5

3

5

6

-4.7

-7.6

-6.6

6

-3.8

10

-8.6 -10.1

BoNscore(%)difference

10

9

-6.6

-8.4

-5.2

-10.9

6

- 8

4

0

-9.2

-6.0

+0.4

-1.2

Physics

4

2

0

-3.1

+0.3

-0.4 -0.6

History

6

3

0

3

-4.8

-2.9

-0.3

+1.3

Economics

6

4

2

0

-5.9

-1.1

-0.4

-1.1

Math

8

4

0

-7.8

-2.9

+0.2

-0.8

Business

dO dP gO gP

- 9

12

Philosophy

Health

Engineering

Computer Science

Other

0

0

0

0

-0.1 -0.3

-0.1 -0.5

+0.4

+0.3

-0.4

-0.5

0

1

-1.0

3

5

2

-1.5 -1.5

2

-2.2

6

3

-2.7

10

-6.8

4

3

-14.2-13.2

-4.3

15

-5.1

-3.1

-8.5

6

dO dP gO gP

dO dP gO gP

dO dP gO gP

dO dP gO gP

Figure 8: Best-of-N performance gap between all-domain and single-domain training on MMLU-pro.

Fig. 6 shows the BoN performance using Llama-3.1-8B-Instruct as pLLM and R1-Distill-Qwen-14B as the reward model backbone. In this setting, dORM performs comparably to dPRM, while gPRM is not competitive, which is contrary to previous work (Lightman et al., 2024; Khalifa et al., 2025) and our math-domain results in Fig. 4. Overall, gORM outperforms dORM/dPRM/gPRM, without notable degradation in any domain relative to the others. As shown in Fig. 7, this trend holds across different pLLM. Using 8B reward backbones, we observe the same pattern (Figs. 28 and 29). Detailed per-domain and weighted majority voting results are provided in §F.

Effect of multi-domain training. To assess the effect of multi-domain training, we train and evaluate all four variants only on each MMLU-Pro domain and compare each variant to its multidomain counterpart. Fig. 8 reports the degradation of BoN performance with N=16 under domainspecialized training. We observe severe drops for dORM/dPRM, with a slightly larger decline for dORM, likely because outcome-only supervision is sparser than step-level supervision and both are relatively data-hungry. In contrast, gORM and gPRM appear more sample-efficient: even without multi-domain training, their performance decreases only modestly (or even increases), demonstrating the efficiency of generative reward models. We defer complete results of single-domain training for four reward models to Figs. 26 and 27 of §F.

- 4 ANALYSIS ON WHY PRMS FAIL IN MULTI-DOMAIN In this section, we analyze the failure modes of PRMs observed in the multi-domain setting of §3.

- 4.1 RISK OF PRMS WITH COT LENGTH

1.5B - Overall

7B - Overall

“Aha” CoTs. As noted in §2.2, PRMs typically assume that once a reasoning step is incorrect, all subsequent steps are incorrect. However, recent reasoning models can recover from earlier mistakes and still arrive at the correct answer (an “aha” moment; Guo et al., 2025). In such cases, PRMs can miss the recovery due to a monotonicity bias induced by their training data. To demonstrate this, we evaluate on “aha” CoTs from ProcessBench that contain at least one incorrect step (∃t ∈ {1,...,T} : zt = 0) but a correct outcome (y = 1). Overall, “aha” CoTs account for 15.3% of the cases. In Fig. 9, we report F1 scores (%) for the “aha” CoTs using M=16 for gORM/gPRM. We observe that PRM variants perform particularly poorly on “aha”

81.1

85.6

F1Score(%)

80

75

69.2

67.4

60

60

50.2

39.0

47.1

35.8

40

45

dORM dPRM gORM gPRM

dORM dPRM gORM gPRM

Figure 9: Results on “aha” CoTs.

CoTs. Moreover, scaling the backbone from 1.5B to 7B improves ORM performance, whereas PRM performance degrades with larger backbones, possibly because larger PRMs are more likely to follow the PRM assumption inherent in their training data and objective (cf. Eqs. 3 and 7).

1.5B - Overall

7B - Overall

Do ORMs overfit on “aha” CoTs? A natural concern about ORM results on “aha” CoTs in Fig. 9 is overfitting: ORMs might only memorize questions and their answers, thereby correctly verifying “aha” CoTs without checking the correctness of intermediate reasoning steps. This memorization issue in the math domain has recently been studied by Wu et al.

100

(-68.4)

(-49.5)

F1Score(%)

83.0

88.1

90

(-51.9)

80

(-18.0)

68.3

63.8

60

60

45.8

30

40

16.4 19.7

33.4

dORM gORM

dORM gORM

Figure 10: Results on randomly shuffled CoTs.

(2025). To investigate this, we conduct the following test: (i) replace the intermediate reasoning steps r1:(T−1) with r1:(′ T−1) taken from other CoTs, and (ii) evaluate ORMs on these perturbed CoTs. If ORMs only memorize the answer in the final reasoning step rT, their performance should remain largely unaffected. However, Fig. 10 shows a significant drop for ORMs (dashed), indicating the reliance on intermediate steps. Interestingly, the degradation is greater with the 7B backbone than with the 1.5B backbone for both dORM and gORM. This suggests that larger models rely more heavily on intermediate reasoning steps during verification.

ProcessBench

MMLU-pro (train)

Risk increases with CoT length. “Aha” moments can also lengthen CoTs, an effect especially pronounced in the multi-domain setting (Fig. 11), where LLMs struggle more than in math. As shown in Fig. 38, majority voting results degrade significantly with increasing CoT length in the multi-domain setting. Consistent with the outcome-verification failures of PRMs on “aha” CoTs, we argue that the error of PRM variants grows with CoT length (T). Intuitively, as a CoT grows longer, the chance that a PRM misclassifies at least one intermediate step rises, making it more likely to prematurely conclude the CoT is incorrect. Longer CoTs also create more opportunities for “aha” recoveries that PRMs systematically miss. We formalize this as follows:

0.2

| | | |
|---|---|---|
| |Aha (avg.: Non-a (avg.:<br><br>|12.49) ha<br><br>7.40)|
| | | |
| | | |
| | | |

Aha (avg.: 8.00)

0.15

Density

Non-aha (avg.: 7.48)

0.10

0.1

0.05

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

0.00

0.0

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

20 40

50 100 150

Number of CoT steps (T)

Figure 11: Length dist. of “aha” CoTs.

- Theorem 4.1 (Log-error bound of dORM and gORM). Let ϵ ∈ {ϵd,ϵg} with ϵd := log fˆdORM(x) − log f(x) and ϵg := log fˆgORM(x) − log f(x). Define m¯ := E[ϵ | x],ξ¯ := ϵ − m,β¯ orm2 := E[ ¯m2]. Then ϵ = m¯ + ξ¯with E[ξ¯ | x] = 0. If Var(ξ¯ | x) ≤ τorm2 for some constant τorm2 independent of T, then E[ϵ2] = E Var(ξ¯| x) + E[ ¯m2] ≤ τorm2 + βorm2 .
- Theorem 4.2 (Log-error lower bound of dPRM). Let ∆dPRM := log fˆdPRM(x) − log f(x). Under the assumptions in §A.1, E ∆2dPRM ≥ (σ2 − 2γ)T.
- Theorem 4.3 (Log-error lower bound of gPRM). Let ∆gPRM := log f˜gPRM(x) − log f(x), where f˜gPRM(x) is approximation of fˆgPRM in Eq. 8 with a single MC sample. Under the assumptions in §A.1, E ∆2gPRM ≥ (σ2 + τ2 − 2γ)T.

All details and proofs are deferred to §A. Theorem. 4.1, 4.2 and 4.3 establish log-error bounds for the four reward-model variants. The dORM/gORM bounds are independent of T, whereas the dPRM/gPRM lower bounds grow linearly with T. In Theorem. A.4, we also show that for gPRM with Monte Carlo estimation (cf. Eq. 5), the log-error lower bound increases linearly with T.

Empirical support. To empirically support Theorems. 4.1 to 4.3, we plot the F1 score (%) for outcome-verification in the multi-domain setting as a function of the number of reasoning CoT steps (T) in Fig. 12-(Left). We bin CoTs into eight categories. As T increases, dPRM/gPRM degrade considerably relative to dORM/gORM. Fig. 12-(Right) shows the performance improvements over majority voting of different categories w.r.t the average number of CoT steps. We observe negative correlations for dPRM (-0.058) and gPRM (-0.608), while dORM (0.316) and gORM (0.399) shows rather positive correlations. These results not only provide empir-

dORM (0.316) dPRM (-0.058) gORM (0.399) gPRM (-0.608)

90

20

| |
|---|

| |
|---|

| |
|---|

| |
|---|

75

| |
|---|

F1score(%)

:BoN-MV

15

60

| |
|---|

10

| |
|---|

| |
|---|

| |
|---|

45

| |
|---|

5

| |
|---|

| |
|---|

30

0

6 9 12 15 18

1-2 3 4 5 6 7-8 9-1314-162

Number of CoT steps (T)

Average CoT steps (T)

Figure 12: (Left): Outcome verification vs. CoT length; (Right): TTS improvement vs. average CoT length.

ical support for Theorems. 4.1 to 4.3 but also demonstrate that increasing CoT length can degrade TTS performance for dPRM and gPRM in the multi-domain setting.

- 4.2 LABEL NOISE OF PRMS

62.5 65.0 67.5 70.0 72.5 75.0 77.5 80.0

Label noise risk. Beyond CoT-length effects, label noise poses an additional risk, especially in multi-domain settings. Since human annotation of long CoTs is more costly in specialized domains such as law and medicine than in math, prior work often relies on LLMs to auto-label process steps (Zeng et al., 2025), which introduces noise that can degrade PRM performance. We study this by injecting synthetic noise into the process labels of PRM800K. We vary the level of noise along two axes: (i) process-noise ratio (the per-step probability of flipping a process label) and (ii) data-noise ratio (the fraction of examples to which noise is applied). We report the outcome-verification F1 score (%) in Fig. 13 using 1.5B backbones, using greedy decoding for generative variants (full results with 7B backbones are provided in Fig. 40). dPRM is highly sensitive to label noise, demonstrating its potential vulnerability in multi-domain data. In contrast, gPRM is more robust, aligning with findings that LLM memorization can cause random-label noise to act as a mild regularizer on mathematical reasoning tasks and improve performance (Wu et al., 2025).

[Figure 2]

dPRM (67.23-81.20)

gPRM (77.78-80.39)

Process-noiseratio(%)

50%

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

40%

30%

20%

10%

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

Data-noise ratio (%)

Figure 13: Effect of labe noise on GSM8K.

Train vs Test (dist.: 0.202)

gORM vs Test (dist.: 0.532)

gPRM vs Test (dist.: 3.083)

Avg. CoT steps vs Dist.

WassersteinDistance

- 0.20

12

Train

gORM gPRM

CoTdensity

- 0.15

9

6

0.10

| |
|---|
| |

| |
|---|

| |
|---|
| |

3

| |
|---|
| |

0.05

0 5 10 15 20 25 0 5 10 15 20 25

0 5 10 15 20 25

| |
|---|

| |
|---|

0

0.00

0 50 100 150

0 50 100 150

0 50 100 150

5 10 15

Number of CoT steps (T)

Number of CoT steps (T)

Number of CoT steps (T)

Average CoT Steps

- Figure 14: Length distribution shift on MMLU-Pro (overall/per-domain) measured by Wasserstein distance.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.5 1.0

4

6

8

10

:BoN-MJ

dORM (r = 0.345)

0.5 1.0

4

6

8

10

12

dPRM (r = 0.048)

0 1 2 3

5

10

15

20

gORM (r = 0.082)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 5 10

0

4

8

12

gPRM (r = -0.625)

Wasserstein Distance

- Figure 15: Per-domain BoN improvement over majority voting vs. Wasserstein distance on MMLU-Pro.

Length shift hurts gPRM. We further analyze why gPRM degrades in the multi-domain setting, despite its robustness to label noise in math. As CoTs lengthen, aligning stepwise verification rationales with process labels becomes increasingly difficult. Consequently, consensus filtering prunes long CoTs, skewing the training CoT-length distribution relative to the test set (Fig. 14).

We quantify the above length distribution shift with the Wasserstein distance (Kantorovich, 1960), reporting distances from the test set to the unfiltered pool (Train), the gORM training set, and the gPRM training set. In the math domain (Tab. 4), gPRM has the smallest distance (e.g., overall: 2.760/2.430/1.600 for Train/gORM/gPRM), whereas in the multi-domain setting (Fig. 14 and Tab. 3) it has the largest distance (e.g., overall: 0.202/0.532/3.083 for Train/gORM/gPRM).

The distribution shift of gPRM also corresponds to its degradation (), evaluated in the multi-domain setting. Fig. 15 shows a strong negative correlation between the Wasserstein distance and perdomain improvement over majority voting (N=16) for gPRM (−0.625), whereas correlations are weak for the other methods (0.345/0.048/0.082 for dORM/dPRM/gORM). Together, these results suggest that consensus filtering induces a length-distribution shift that disproportionately affects gPRM in the multi-domain setting, despite its robustness to label noise.

- 5 PRACTICAL GUIDELINES, LIMITATIONS, AND FUTURE WORK This section suggests practical guidance, clarifies limitations, and outlines future directions:

- (i) Short CoTs, clean labels, tight latency dPRM

- (ii) Long CoTs / frequent error recoveries gORM if compute permits; else dORM

- (iii) Mixed/shifting domains gORM

- (iv) High label noise

ORM

PRMs amplify early errors

- (v) Strict compute/latency

dORM/dPRM

gORM and gPRM add sampling overhead

- (vi) Limited training data

gORM/gPRM

Higher sample efficiency

Limitations. While we present a thorough analysis of four reward model variants, our study has several limitations: (i) we evaluate only tasks with verifiable outcomes (e.g., math and non-math multiple–choice), which may not generalize to open-ended generation. (ii) All models are trained via supervised fine–tuning. One could instead use a generative verifier to roll out rationales and treat agreement between their verdict and the GT label as a reward signal for reinforcement learning (RL). Because using RL to train verifiers/reward models is uncommon and introduces additional confounders, we exclude RL-based training from our analysis. (iii) Owing to computational constraints, we adopt LoRA adapters rather than full-parameter fine-tuning. This choice may affect performance and scaling behavior, however, we expect the qualitative trends to hold. (iv) Following most of the PRM literature (Lightman et al., 2024; Zeng et al., 2025), we do not consider tool use, however, Gou et al. (2024) showed that tool use can help reduce auto-label noise.

Future work. In future work, we plan to extend our analysis to open-ended generation, broader task domains, and a wider range of model families and training regimes, and to explicitly study tool-augmented verification and inference pipelines to assess their impact on label quality.

ETHICS STATEMENT

This work evaluates verification strategies for test-time scaling of LLMs across multiple domains. It does not involve human subjects, user studies, or the collection of personally identifiable information. All datasets used are publicly available benchmarks and were accessed under their respective licenses. To the best of our knowledge, they do not contain sensitive personal data.

A natural direction for future work is to increase the trustworthiness of LLM outputs in real systems by verifying them, thereby reducing reasoning errors and hallucinations. Although our experiments include legal and medical themed datasets (e.g., law and health), the models and methods are research artifacts and are not intended for real-world legal, medical, or other high-stakes decisionmaking. They should not substitute professional judgment, and any deployment in such settings would require additional domain-specific validation, safety auditing, and regulatory compliance.

REPRODUCIBILITY STATEMENT

For reproducibility, we believe that we provide sufficient materials, including prompts, hyperparameters, model backbones, training details, and the synthetic data generation process, throughout the main paper (Sections 2 and 3.1). Additional details are deferred to Sections B to E due to space constraints. We publicly release all relevant artifacts for reproducibility: (i) code, (ii) datasets (including any we generate), and (iii) model checkpoints at https://github.com/db-Lee/Multi-RM as follows:

#### Code.

- • data generation/: code for generating verification CoTs of gORM/gPRM.

- • discriminative/: code for training/inference of dORM/dPRM (mostly adapted from VersaPRM (Zeng et al., 2025)).
- • generative/: code for training/inference of gORM/gPRM.

#### Training datasets.

- • train: multi-domain training dataset for dORM/dPRM (mostly adapted from VersaPRM (Zeng et al., 2025)).
- • train gORM: multi-domain training dataset for gORM generated by QwQ-32B.

- • train gPRM: multi-domain training dataset for gPRM generated by QwQ-32B. Test datasets.

- • test: multi-domain test dataset with CoTs (N=128) generated by Llama-3.1-8B-Instruct (mostly adapted from VersaPRM (Zeng et al., 2025)).
- • test smollm: multi-domain test dataset with CoTs (N=16) generated by SmolLM3-3B.

- • test qwen: multi-domain test dataset with CoTs (N=16) generated by Qwen2.5-7B-Instruct.

- • test gemma: multi-domain test dataset with CoTs (N=16) generated by gemma-2-9b-it.

- • test llama: multi-domain test dataset with CoTs (N=16) generated by Llama-3.1-70BInstruct. Model checkpoints.
- • dORM-14B: dORM with DeepSeek-R1-Distill-Qwen-14B backbone, trained on train.
- • dPRM-14B: dPRM with DeepSeek-R1-Distill-Qwen-14B backbone, trained on train.
- • gORM-14B: gORM with DeepSeek-R1-Distill-Qwen-14B backbone, trained on train gORM.

- • gORM-14B-merged: LoRA-merged version of gORM-14B for vLLM inference.
- • gPRM-14B: gPRM with DeepSeek-R1-Distill-Qwen-14B backbone, trained on train gPRM.

- • gPRM-14B-merged: LoRA-merged version of gPRM-14B for vLLM inference.
- • dORM-8B: dORM with DeepSeek-R1-Distill-Llama-8B backbone, trained on train.
- • dPRM-8B: dPRM with DeepSeek-R1-Distill-Llama-8B backbone, trained on train.
- • gORM-8B: gORM with DeepSeek-R1-Distill-Llama-8B backbone, trained on train gORM.

- • gORM-8B-merged: LoRA-merged version of gORM-8B for vLLM inference.
- • gPRM-8B: gPRM with DeepSeek-R1-Distill-Llama-8B backbone, trained on train gPRM.

- • gPRM-8B-merged: LoRA-merged version of gPRM-8B for vLLM inference.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Bo Adler, Niket Agarwal, Ashwath Aithal, Dong H Anh, Pallab Bhattacharya, Annika Brundyn, Jared Casper, Bryan Catanzaro, Sharon Clay, Jonathan Cohen, et al. Nemotron-4 340b technical report. arXiv preprint arXiv:2406.11704, 2024.

Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. A survey on data selection for language models. arXiv preprint arXiv:2402.16827, 2024.

Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Pati˜no, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallou´edec, Kashif Rasul, Nathan Habib, Cl´ementine Fourrier, Hynek Kydlicek, Guilherme Penedo, Hugo Larcher, Mathieu Morlon, Vaibhav Srivastav, Joshua Lochner, Xuan-Son Nguyen, Colin Raffel, Leandro von Werra, and Thomas Wolf. SmolLM3: smol, multilingual, long-context reasoner. https: //huggingface.co/blog/smollm3, 2025.

Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fern´andez, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael Hanna, Alexander Koller, Andre Martins, Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle, Barbara Plank, David Schlangen, Alessandro Suglia, Aditya K Surikuchi, Ece Takmaz, and Alberto Testoni. LLMs instead of human

judges? a large scale empirical study across 20 NLP evaluation tasks. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 238–255, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-889176-252-7. doi: 10.18653/v1/2025.acl-short.20. URL https://aclanthology.org/ 2025.acl-short.20/.

Eugene Charniak and Mark Johnson. Coarse-to-fine n-best parsing and maxent discriminative reranking. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics, pp. 173–180. Association for Computational Linguistics, 2005.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen, Yang Yan, Hao Li, Bin Ling, Yonghong Tian, and Li Yuan. ChatLaw: A multi-agent collaborative legal assistant with knowledge graph enhanced mixture-of-experts large language model. arXiv preprint arXiv:2306.16092, 2023.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research (TMLR), 2023.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. RLHF workflow: From reward modeling to online RLHF. Transactions on Machine Learning Research (TMLR), 2024. ISSN 2835-8856.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. LawBench: Benchmarking legal knowledge of large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7933–7962, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.452. URL https://aclanthology.org/ 2024.emnlp-main.452/.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. OmniMATH: A universal olympiad level mathematic benchmark for large language models. International Conference on Learning Representations (ICLR), 2025.

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Nan Duan, and Weizhu Chen. CRITIC: Large language models can self-correct with tool-interactive critiquing. International Conference on Learning Representations (ICLR), 2024.

Neel Guha, Julian Nyarko, Daniel Ho, Christopher R´e, Adam Chilton, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, et al. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems (NeurIPS), 2023.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024. acl-long.211/.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. International Conference on Learning Representations (ICLR), 2022.

Leonid V Kantorovich. Mathematical methods of organizing and planning production. Management science, 6(4):366–422, 1960.

Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang. Grace: Discriminator-guided chain-of-thought reasoning. arXiv preprint arXiv:2305.14934, 2023.

Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, Honglak Lee, and Lu Wang. Process reward models that think. arXiv preprint arXiv:2504.16828, 2025.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems (NeurIPS), 2022.

Tiffany H Kung, Morgan Cheatham, Arielle Medenilla, Czarina Sillos, Lorie De Leon, Camille Elepa˜no, Maria Madriaga, Rimel Aggabao, Giezel Diaz-Candido, James Maningo, et al. Performance of chatgpt on usmle: potential for ai-assisted medical education using large language models. PLoS digital health, 2(2):e0000198, 2023.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. International Conference Learning Representations (ICLR), 2024.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: NLG evaluation using gpt-4 with better human alignment. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2511–2522, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.153. URL https://aclanthology.org/2023. emnlp-main.153/.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. International Conference on Learning Representations (ICLR), 2019.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems (NeurIPS), 36:46534– 46594, 2023.

Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems (NeurIPS), 2022.

Qwen Team. QwQ-32B: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for LLM reasoning. International Conference on Learning Representations (ICLR), 2025.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language models. Nature Medicine, 31(3):943–950, 2025.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. International Conference on Learning Representations (ICLR), 2025.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´e, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Qwen Team. Qwen2.5: A party of foundation models, September 2024a. URL https: //qwenlm.github.io/blog/qwen2.5/.

Qwen Team. QwQ: Reflect deeply on the boundaries of the unknown, November 2024b. URL https://qwenlm.github.io/blog/qwq-32b-preview/.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. Is ChatGPT a good NLG evaluator? a preliminary study. In Yue Dong, Wen Xiao, Lu Wang, Fei Liu, and Giuseppe Carenini (eds.), Proceedings of the 4th New Frontiers in Summarization Workshop, pp. 1–11, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.newsum-1.1. URL https: //aclanthology.org/2023.newsum-1.1/.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), pp. 9426–9439, Bangkok, Thailand, 2024a. doi: 10.18653/v1/2024.acl-long.510.

Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. Self-taught evaluators. arXiv preprint arXiv:2408.02666, 2024b.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems (NeurIPS), 2024c.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems (NeurIPS), 35:24824–24837, 2022.

Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Huijie Lv, Ming Zhang, et al. Reasoning or memorization? unreliable results of reinforcement learning due to data contamination. arXiv preprint arXiv:2507.10532, 2025.

Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Scaling inference computation: Compute-optimal inference for problem-solving with language models. The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems (NeurIPS), 36:11809–11822, 2023a.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023b.

Fei Yu, Anningzhe Gao, and Benyou Wang. OVM, outcome-supervised value models for planning in mathematical reasoning. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Findings of the Association for Computational Linguistics: NAACL 2024, pp. 858–875, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.55. URL https://aclanthology.org/2024.findings-naacl.55/.

Thomas Zeng, Shuibai Zhang, Shutong Wu, Christian Classen, Daewon Chae, Ethan Ewer, Minjae Lee, Heeju Kim, Wonjun Kang, Jackson Kunde, et al. VersaPRM: Multi-domain process reward model via synthetic reasoning data. International Conference on Machine Learning (ICML), 2025.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. International Conference on Learning Representations (ICLR), 2025a.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 10495–10516, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-2565. doi: 10.18653/v1/2025.findings-acl.547. URL https://aclanthology.org/2025. findings-acl.547/.

Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li, et al. GenPRM: Scaling test-time compute of process reward models via generative reasoning. arXiv preprint arXiv:2504.00891, 2025.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. PROCESSBENCH: Identifying process errors in mathematical reasoning. arXiv preprint arXiv:2412.06559, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena. Advances in neural information processing systems (NeurIPS), 2023.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. JudgeLM: Fine-tuned large language models are scalable judges. International Conference on Learning Representations (ICLR), 2025.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul F. Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

APPENDIX OVERVIEW

This appendix provides supplementary materials to support the main paper as follows:

- • Theoretical Analysis (§A): details notations, assumptions, and proofs for Theorems. 4.1 to 4.3.
- • Prompts (§B): presents the detailed prompt formats.
- • Datasets (§C): describes the datasets used in our experiments.
- • Implementation Details (§D): provides implementation details, such as (i) backbones for reward models, (ii) hyperparameters, and (iii) verification CoTs for gORM and gPRM.
- • Training Examples (§E): contains training examples including verification CoTs of gORM and gPRM in the law domain of MMLU-pro.
- • Additional Results (§F): includes the complete results of §3.2 (omitted in the main paper due to the space limit), such as results on MMLU-pro using weighted majority voting.
- • Additional Analysis (§G): includes the complete results of §4.
- • Use of LLMs (§H): outlines our use of LLMs in accordance with the ICLR 2026 submission policy.

- A THEORETICAL ANALYSIS

- A.1 ANALYSIS ON LOG-ERROR BOUND

Notation. We assume that a correct final step, y = zT = 1, implies all previous steps are correct. Define the stepwise conditional probabilities ut := Pr(zt = 1 | x,z1 = 1,...,zt−1 = 1) for t ∈ [T]. By the chain rule and the assumption, the true reward function,

T

f(x) = p(y = 1 | x) = p(zT = 1 | x) = p(z1:T = 1 | x) =

ut(x)

t=1

and we write ζ(x) := log f(x) = Tt=1 log ut(x). For dPRM, we define the stepwise conditional distribution uˆt(x) := fˆdPRM(x1:t) and use product for the aggregation, i.e.fˆdPRM(x) :=

T t=1 uˆt(x). Similarly, we define the conditional distribution Ft(x,v≤t) ∈ [0,1] to be the

gPRM’s normalized probability that step t is correct given the verification prefix, i.e.fˆgPRM(x) := Ev

[ Tt=1 Ft(x,v≤t)]. To bound log probability, we assume there is ς ∈ (0,1/2] such that all probabilities/predictors appearing inside logarithms are clipped into [ς,1 − ς]. Hence all logs are finite and |log(·)| ≤ log(1/ς).

1:L+

#### Error terms.

- 1. dPRM. Define δt := log uˆt − log ut (evaluated at the appropriate prefixes), and mt := E[δt | x], ξt := δt − mt,

so E[ξt | x] = 0.

- 2. dORM or gORM. Let ϵ ∈ {ϵd,ϵg},

ϵd := log fˆdORM(x) − log f(x), ϵg := log fˆgORM(x) − log f(x), and decompose

m¯ := E[ϵ | x], ξ¯:= ϵ − m,¯ βorm2 := E[ ¯m2], so that E[ξ¯| x] = 0.

- 3. gPRM. For a single rollout v1:L+ ∼ pgPRM(· | x), define

T

u˜t := Ft(x,v≤t), f˜gPRM(x) :=

u˜t. The sampled gPRM log-error is

t=1

∆gPRM := log f˜gPRM(x) − ζ(x) =

T

δt(g), δt(g) := log u˜t − log ut.

t=1

Let

m(tg) := E[δt(g) | x], ξt(g) := δt(g) − m(tg), so that E[ξt(g) | x] = 0.

Assumptions. There exist constants σ2 > 0 and γ ≥ 0 (independent of T) such that for all x,

- 1. (Variance floors) Var(ξt | x) ≥ σ2, Var(ξt(g) | x) ≥ σ2 + τ2
- 2. (Weak anti-correlation)

1≤s<t≤T

Cov(ξs,ξt | x) ≥ −γT,

1≤s<t≤T

Cov(ξs(g),ξt(g) | x) ≥ −γT

- 3. (Positive slope) σ2 > 2γ.

For gPRM with sampled verification CoTs, sampling contributes per-step noise: Var(ξt(g) | x) ≥ σ2 + τ2 for some τ2 > 0. For ORMs, assume Var(ξ¯| x) ≤ τorm2 < ∞ (no T-dependence).

- Theorem A.1 (Log-error lower bound of dPRM). Let ∆dPRM := log fˆdPRM(x) − ζ(x). Under the assumptions above,

E ∆2dPRM ≥ (σ2 − 2γ)T.

- Theorem A.2 (Log-error bound of dORM or gORM). Let ϵ ∈ {ϵd,ϵg} and write ϵ = m¯ + ξ¯ with E[ξ¯| x] = 0. If Var(ξ¯| x) ≤ τorm2 (independent of T), then

E[ϵ2] = E[Var(ξ¯| x)] + E[ ¯m2] ≤ τorm2 + βorm2 , a bound that does not depend on the CoT length T.

- Theorem A.3 (Log-error lower bound of gPRM). Under the assumptions above, E (∆gPRM)2 ≥ (σ2 + τ2 − 2γ)T.

Jensen-gap representation (mean predictor). Let L(x,v) := Tt=1 log Ft(x,v≤t) and Kx(θ) := log E[eθL | x]. Define the mean predictor µ(x) := E[eL | x] and ∆mean(x) := log µ(x) − ζ(x). Then with B(g)(x) := E[L | x] − ζ(x), we have the exact decomposition

∆mean(x) = B(g)(x) + δJ(x), δJ(x) = Kx(1) − Kx′ (0) =

1

0

(1 − θ)Varθ(L | x)dθ ≥ 0,

where Varθ denotes variance under the exponentially tilted law dPθ ∝ eθLdP, i.e., dPθ(v) = {M(θ) > 0}e θL(x,v) M(θ)−1dP(v) with M(θ) := E[eθL | x].

- Theorem A.4 (Log-error lower bound of mean-gPRM). Assume the conditions of Theorem A.3. In addition, suppose there exists κ ∈ (0,1] such that for all θ ∈ [0,1],

Varθ(L | x) ≥ κVar(L | x). Then, for every x,

κ 2

κ 2

(σ2 + τ2 − 2γ)T . Consequently,

∆mean(x) ≥ B(g)(x) +

Var(L | x) ≥ B(g)(x) +

E[∆mean] ≥

κ 2

(σ2 + τ2 − 2γ)T − E[B(g)(x)2], E[∆2mean] ≥ (max{0,E[∆mean]})2 .

Takeaways. Under mild anti-correlation and variance-floor assumptions, dPRM and sampled gPRM incur log-error that grows at least linearly in the CoT length T, and the additional sampling noise τ2 makes gPRM strictly worse. In contrast, ORM estimators admit error bounds that are independent of T provided the conditional noise is bounded, which makes them preferable for long CoTs. For mean-gPRM, the Jensen gap introduces a strictly nonnegative bias that scales with the variance of L and hence with T, so even a calibrated predictor (B(g) = 0) exhibits error that increases with chain length. All proofs are deferred to §A.2.

- A.2 PROOFS

#### Proof of Theorem. A.1.

Proof. Let

T

T

B :=

mt, N :=

ξt,

t=1

t=1

so ∆dPRM = B + N with E[N | x] = 0. By the tower property,

E ∆2dPRM = E E[(B + N)2 | x] = E B2 + 2BE[N | x] + E[N2 | x]

= E Var(N | x) + E[B2] ≥ E Var(N | x) . Expanding,

T

Var(N | x) =

Var(ξt | x) + 2

Cov(ξs,ξt | x).

t=1

1≤s<t≤T

Apply the variance floors and weak anti-correlation to get Var(N | x) ≥ +Tσ2 − 2γT. Taking expectations preserves the bound.

| |
|---|

#### Proof of Theorem. A.2

Proof. By the conditional bias–variance decomposition (law of total variance),

E[ϵ2] = E Var(ϵ | x) + E (E[ϵ | x])2 = E Var(ξ¯| x) + E m ¯ 2 .

The assumption Var(ξ¯ | x) ≤ τorm2 for all x gives E[Var(ξ¯ | x)] ≤ τorm2 , and by definition βorm2 = E[ ¯m2].

| |
|---|

#### Proof of Theorem. A.3

Proof. Decompose

T

T

T

δt(g) =

m(tg)

ξt(g)

∆gPRM =

+

.

t=1

t=1

t=1

=: B(g)

=: N(g)

Conditional mean-zero E[N(g) | x] = 0 implies

E (∆gPRM)2 = E Var N(g) | x + E (B(g))2 ≥ E Var N(g) | x . Now expand Var(N(g) | x):

T

Var(ξt(g) | x) + 2

Cov(ξs(g),ξt(g) | x) ≥ T(σ2 + τ2) − 2γT.

Var(N(g) | x) =

t=1

1≤s<t≤T

Taking expectations in x gives the stated bound.

| |
|---|

#### Proof of Theorem. A.4

Proof. 1) Exponential tilting and log-mgf. Define M(θ) := E[eθL | X] and KX(θ) := log M(θ). Since eθL ∈ (0,1] for θ ∈ [0,1] and E[|L|2] < ∞, dominated convergence yields M′(θ) =

E[LeθL | X] and M′′(θ) = E[L2eθL | X]. Let dPθ(C) := eθL(X,C)M(θ)−1dP(C) and Eθ[·] := E[·eθL]/M(θ). Then

M′′(θ)M(θ) − (M′(θ))2 M(θ)2

M′(θ) M(θ)

= Eθ[L | X], KX′′ (θ) =

KX′ (θ) =

= Varθ(L | X).

- 2) Jensen-gap identity. Taylor with integral remainder at θ = 0 gives

KX(1) = KX(0) + KX′ (0) +

1

0

(1 − θ)KX′′ (θ)dθ.

Since KX(0) = 0 and KX′ (0) = E[L | X], we obtain

log µ(X) = E[L | X] +

1

0

(1 − θ)Varθ(L | X)dθ. By definition of the mean predictor,

∆mean(X) = log µ(X) − ζA(X), where µ(X) = E[eL | X].

Plugging log µ(X) = ∆mean(X) + ζA(X) with B(g)(X) := E[L | X] − ζA(X), this yields

∆mean(X) = B(g)(X) + δJ(X), δJ(X) :=

1

0

(1 − θ)Varθ(L | X)dθ ≥ 0.

- 3) Lower bound on δJ and variance linkage. By tilt-stability,

δJ(X) ≥

κ 2

Var(L | X).

Moreover, since L = ζA + ∆g-prm = ζA + B(g) + N(g) with E[N(g) | X] = 0, and since ζA and B(g)(X) are constants when conditioning on X, we have

Var(L | X) = Var(N(g) | X). Expanding and using the variance floors and weak anti-correlation conditions (as in Theorem A.3),

Var(N(g) | X) ≥ σA2 + T(σ2 + τ2) − 2γT − 2γA. Combining this gives the pointwise bound

∆mean(X) ≥ B(g)(X) +

κ 2

σA2 + T(σ2 + τ2) − 2γT − 2γA .

- 4) Expectations and MSE. Taking expectations over X and applying Cauchy–Schwarz to E[B(g)(X)] yields

κ 2

(σ2 + τ2 − 2γ)T + (σA2 − 2γA) − E[B(g)(X)2].

E[∆mean] ≥

Finally, Jensen’s inequality gives (max{0,E[∆mean]})2 ≤ E[∆2mean], so the MSE bound follows. In the calibrated case B(g) ≡ 0, the stated simplified bounds hold.

| |
|---|

- B PROMPTS In this section, we present prompt formats used in this work:

- • Fig. 16: User prompt format for generating CoTs on GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021).
- • Fig. 17: User prompt format for generating CoTs on MMLU-Pro (Wang et al., 2024c) proposed by Zeng et al. (2025).
- • Fig. 18: System prompt format for auto-labeling process labels on MMLU-Pro (Wang et al., 2024c) proposed by Zeng et al. (2025).
- • Fig. 19: User prompt format for auto-labeling process labels on MMLU-Pro (Wang et al., 2024c) proposed by Zeng et al. (2025).
- • Fig. 20: Prompt format of gORM (Zhang et al., 2025a). We use this format for both generating synthetic verification-CoTs and training/evaluation of gORM.

#### • Fig. 21: Prompt format for generating verification-CoTs for gPRM following Khalifa et al.

(2025).

##### • Fig. 22: Prompt format of gPRM for training and evaluation.

[user] Solve the following math problem efficiently and clearly:

- - For simple problems (2 steps or fewer): Provide a concise solution with minimal explanation.
- - For complex problems (3 steps or more): Use this step-by-step format:

- ## Step 1: [Concise description] [Brief explanation and calculations]
- ## Step 2: [Concise description] [Brief explanation and calculations] [OMITTED...]

Regardless of the approach, always conclude with: Therefore, the final answer is: $\\boxed{answer}$.

I hope it is correct. Where [answer] is just the final number or expression that solves the problem.

[Problem] {problem} [/user] [assistant]

- Figure 16: User prompt format for generating CoTs on GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021).

[user] Given the following question and candidate answers, choose the best answer. [Question]

- {question #1} [/user] [assistant]

- {assistant response #1} [/assistant]

[user] Given the following question and candidate answers, choose the best answer. [Question] {question #2} [/user] [assistant]

- {assistant response #2} [/assistant] [OMITTED...]

[user] Given the following question and candidate answers, choose the best answer.

[Question] {question} [/user] [assistant]

- Figure 17: User prompt format for generating CoTs on MMLU-Pro (Wang et al., 2024c) proposed by Zeng et al. (2025)

[system] You are an experienced evaluator specializing in assessing the quality of reasoning steps in problem-solving. Your task is to find the first BAD step in a student’s solution to a multiple choice question. You will judge steps as GOOD, OK, or BAD based on the following criteria:

- 1. GOOD Step A step is classified as GOOD if it meets all of these criteria:

- • Correct: Everything stated is accurate and aligns with known principles or the given problem.
- • Verifiable: The step can be verified using common knowledge, simple calculations, or a quick reference (e.g., recalling a basic theorem). If verifying requires extensive effort (e.g., detailed calculations or obscure references), mark it BAD instead.
- • Appropriate: The step fits logically within the context of the preceding steps. If a prior mistake exists, a GOOD step can correct it.
- • Insightful: The step demonstrates reasonable problem-solving direction. Even if ultimately progressing in the wrong direction, it is acceptable as long as it represents a logical approach.

- 2. OK Step A step is classified as OK if it is:

- • Correct and Verifiable: Contains no errors and can be verified.
- • Unnecessary or Redundant: Adds little value, such as restating prior information or providing basic encouragement (e.g., “Good job!”).
- • Partially Progressing: Makes some progress toward the solution but lacks decisive or significant advancement.

- 3. BAD Step A step is classified as BAD if it:

- • Is Incorrect: Contains factual errors, misapplies concepts, derives an incorrect result, or contradicts the ground truth answer.
- • Is Hard to Verify: Requires significant effort to confirm due to poor explanation.
- • Is Off-Topic: Includes irrelevant or nonsensical information.
- • Derails: Leads to dead ends, circular reasoning, or unreasonable approaches.

Task Description You will be provided with:

- 1. A Multiple Choice Question
- 2. A Ground Truth Answer
- 3. A Student’s Step-by-Step Solution, where each step is enclosed with tags and indexed from 0.

Once you identify a BAD step, return the index of the earliest BAD step. Otherwise, return the index of -1 (which denotes all steps are GOOD or OK). Please put your final answer (i.e., the index) in \boxed{}. [/system]

- Figure 18: System prompt format for auto-labeling process labels on MMLU-Pro (Wang et al., 2024c) proposed by Zeng et al. (2025)

[user] The following is a multiple choice question and its ground truth answer. You are also given a student’s solution (split into steps, enclosed with tags and indexed from 0): [Multiple Choice Question] {problem} [Ground Truth Answer] {answer} [Student Solution] {solution} [/user] [assistant] The first BAD step index is:

- Figure 19: User prompt format for auto-labeling process labels on MMLU-Pro (Wang et al., 2024c) proposed by Zeng et al. (2025)

[user] You are a {category} teacher. Grade the solution, verifying correctness step by step. At the end of Solution verification, when you give your final grade, write it in the form “Verification: Is the answer correct (Yes/No)? X”, where X is either Yes or No.

[{Category} Problem] {problem}

[Solution] {solution} [/user] [assistant] [think] Let’s verify step by step:

- Figure 20: Prompt format of gORM (Zhang et al., 2025a). We use this format for both generating synthetic verification-CoTs and training/evaluation of gORM.

[user] You are given a {category} problem and a proposed multiple-step solution (with a step on each line): [{Category} Problem] {problem} [Solution] {solution}

Review and critique the proposed solution steps and determine whether each step is correct. If the solution is incomplete, only critique the steps that are provided. Your output must be in the following format:

- Step 1: The step is \boxed{correct/incorrect}
- Step 2: The step is \boxed{correct/incorrect}

. Step n: The step is \boxed{correct/incorrect}

Once you find an incorrect step, you should stop since you do not need to analyze the remaining steps. If the solution is incomplete, only verify the provided steps. [/user] [assistant] [think] Let’s verify step by step:

###### Figure 21: Prompt format for generating verification-CoTs for gPRM following Khalifa et al. (2025).

[user] You are given a {category} problem and a proposed step-by-step solution: [{category} Problem] {problem} [Solution] {solution}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps. [/user] [assistant] [think] Let’s verify step by step:

Figure 22: Prompt format of gPRM for training and evaluation following Khalifa et al. (2025).

- C DATASET

In this section, we provide more details on the datasets used in this paper.

Math Datasets. For the math domain, we use the widely adopted PRM800K (Lightman et al., 2024) for training, where the process labels z1:T are human-annotated. For training ORMs, we set the outcome label y = z1:T = 1T (rather than y = (ˆa(r T) = a)), since PRM800K provides high-quality ground-truth process labels. As a testbed, we use ProcessBench (Zheng

et al., 2024), which comprises four splits: 400 CoTs from GSM8K (Cobbe et al., 2021), 1K from Math (Hendrycks et al., 2021), 1K from Omni-Math (Gao et al., 2025), and 1K from OlympiadBench (He et al., 2024). We evaluate outcome verification by predicting y ∈ {0,1} using the final answer correct field. We also generate N = 16 CoTs per question with Qwen2.57B-Instruct (Team, 2024a) to assess test-time scaling (TTS).

Table 1: Dataset statistics for each domain of MMLU-pro (Wang et al., 2024c). We report the number of questions, the number of CoTs, and the average number of CoTs per question for both training and test splits.

Training Set Test Set # Questions # CoTs Avg. CoTs / Q # Questions # CoTs Avg. CoTs / Q

Domain

Law 500 7,806 15.61 145 18,537 127.84 Psychology 498 7,901 15.87 150 19,164 127.76 Chemistry 500 6,537 13.07 150 15,981 106.54 Biology 417 6,420 15.40 130 16,441 126.47 Physics 500 6,680 13.36 150 16,460 109.73 History 81 1,275 15.74 150 19,159 127.73 Economics 500 7,749 15.50 150 18,911 126.07 Math 500 6,940 13.88 150 17,014 113.43 Business 489 6,969 14.25 149 17,344 116.40 Philosophy 199 3,125 15.70 149 18,844 126.47 Health 456 7,202 15.79 140 17,862 127.59 Engineering 500 6,032 12.06 150 15,708 104.72 Computer Science 110 1,638 14.89 150 18,429 122.86 Other 500 7,824 15.65 150 18,982 126.55

Total 5,750 84,098 14.63 2,063 248,836 120.62

Multi-domain datasets. For the multi-domain setting, we adopt MMLU-Pro (Wang et al., 2024c), a 10-choice benchmark spanning 14 domains: law, psychology, chemistry, biology, physics, history, economics, math, business, philosophy, health, engineering, computer science, and other. As shown in Tab. 1, the corpus includes 5,750 training and 2,063 evaluation questions. For each question, Zeng et al. (2025) generate 16/128 CoTs for training/evaluation with Llama-3.1-8BInstruct (Dubey et al., 2024), and auto-label reasoning steps (i.e., process labels) using Llama3.1-70B-Instruct with prompts in Figs. 18 and 19; please see Zeng et al. (2025) for more details. To assess generalization across CoTs from different pLLM, we also generate 16 CoTs per evaluation question using SmolLM3-3B (Bakouch et al., 2025), Qwen2.5-7B-Instruct, gemma-2-9b-it (Team et al., 2024), and Llama-3.1-70B-Instruct, spanning diverse model sizes and families.

- D IMPLEMENTATION DETAILS

In this section, we provide implementation details omitted from the main paper due to space limits. Backbones for reward models. Following Zhang et al. (2025a) and Khalifa et al. (2025), we use R1-Distill-Qwen-1.5B and R1-Distill-Qwen-7B (Guo et al., 2025) for the math domain, and R1-Distill-Llama-8B and R1-Distill-Qwen-14B for the multi-domain setting, as reward-model backbones. Note that VersaPRM (Zeng et al., 2025) originally used Llama-3.1-8B-Instruct as the rewardmodel backbone for dPRM; for a fair comparison, we use R1-Distill models for both dORM and dPRM.

Table 2: Summary of hyperparameters.

LoRA Training Inference

Method

| |Rank r α Dropout p|Batch Optim. Epochs LR Decay Scheduler|Package Temp. τ M<br><br>|
|---|---|---|---|
|dORM & dPRM gORM & gPRM|16 32 0.1 32 16 0.1|16 AdamW 1 1e-4 1e-2 Cosine 16 AdamW 1 1e-4 1e-2 Linear|- - vLLM 0.6 10 or 16|

Hyperparameters. We apply LoRA (Hu et al., 2022) for parameter-efficient fine-tuning, optimize with AdamW (Loshchilov & Hutter, 2019), and use vLLM (Kwon et al., 2023) for fast inference. At inference, we sample M=16 verification CoTs for the math domain and M=10 for the multi-domain setting. Hyperparameters are summarized in Tab. 2: for dORM/dPRM we adopt those of Zeng et al. (2025), and for gORM/gPRM we follow Khalifa et al. (2025). Note that in preliminary experiments we set r=32 and α=16 for dORM/dPRM to compare fairly with gORM/gPRM (also using r=32 and

α=16). However, we observed an overall performance degradation (e.g., ≈2%), so we follow the settings of Zeng et al. (2025). The hyperparameters in Tab. 2 are shared across all experiments and we do not perform exhaustive tuning. We report means over three independent runs for the math domain and a single run for the multi-domain setting due to resource constraints.

Verification CoTs for gORM and gPRM. Following Khalifa et al. (2025), we sample 4 different verification CoTs for each question q and CoT r1:T pair in the training dataset by prompting QwQ-32B (Qwen Team, 2025) with temperature=0.6, top k=20, top p=0.95, and min p=0 using the formats in Figs. 20 and 21. Note that Khalifa et al. (2025) originally used QwQ-32BPreview (Team, 2024b). In preliminary, we found QwQ-32B more likely to follow instructions and produce more parsable verification CoTs (e.g., 1K vs. 7K for gPRM in the law domain), so we use it throughout.

For the math domain we set category as math; for the multi-domain setting we use category ∈{law, ..., computer science} and leave it blank for other. We discard any verification CoT that (i) has unparsable labels; (ii) contains Chinese characters; (iii) exceeds the token limit—4,096 for math (Khalifa et al., 2025) or 8,192 for multi-domain; or (iv) whose parsed labels are inconsistent with the targets (e.g., y or z1:T), corresponding to the consensus filtering in §2.2. We also balance the counts of Yes/No examples. The resulting training sets contain 34,286 CoTs for gORM and 35,666 CoTs for gPRM in the math domain, and 171,780 (gORM) and 94,156 (gPRM) in the multi-domain setting.

- E TRAINING EXAMPLES In this section, we present training examples including verification CoTs:

- • Fig. 23: Training example of gORM on the law domain of MMLU-pro (Wang et al., 2024c).
- • Fig. 24: Training example of gPRM on the law domain of MMLU-pro (Wang et al., 2024c).

[user] You are a {category} teacher. Grade the solution, verifying correctness step by step. At the end of Solution verification, when you give your final grade, write it in the form “Verification: Is the answer correct (Yes/No)? X”, where X is either Yes or No.

[Law Problem] Question: A shopper in a store sees that a display case is about to fall on a second shopper’s head. The shopper does nothing to warn or assist the second shopper, who is a stranger to her. She could have taken action to protect the second shopper without harming herself. Did the shopper have a duty to take action to try and protect the second shopper from injury?

- A. No, the shopper is not required to intervene unless she is a store employee or security personnel.
- B. Yes, the shopper must take action under the Good¨ Samaritan¨law.
- C. Yes, the fact that a person sees that action is needed for another’s aid or protection does impose a legal duty to act. [OMITTED...] [Solution]

The shopper could have taken action to protect the second shopper without harming herself, so the issue is whether there is a legal duty to intervene.

The “Good Samaritan” law protects individuals from liability for failing to provide assistance, so B is incorrect.

There is no special relationship between the shopper and the second shopper, so I is incorrect. [OMITTED...] Therefore, the shopper is not required to intervene, and the correct answer is D. The answer is (D). [/user] [assistant] [think]

Let’s verify step by step: The question is about whether a shopper has a duty to act to prevent harm to another stranger when she can do so without risk. The solution concludes D is correct.

First, the solution correctly dismisses B because Good Samaritan laws typically protect those who do help, not impose a duty to act. That seems right.

For option I, the solution says no special relationship exists. Since the shopper and victim are strangers, that’s accurate. So I is wrong.

[OMITTED...]

The solution mentions “duty to rescue” in certain circumstances like special relationships or being first to notice. However, in the problem, there’s no indication the shopper was first. So D holds.

Wait, but the problem states “she could have taken action without harming herself,” which is a key point. Some jurisdictions might have a duty if it’s safe, but traditionally, common law doesn’t. The solution’s conclusion aligns with that.

Therefore, the answer D is correct. The solution’s reasoning is sound, dismissing other options properly. The final answer D matches the analysis. So the solution is correct.

[/think] Verification: Is the answer correct (Yes/No)? Yes [assistant]

- Figure 23: Training example of gORM on the law domain of MMLU-pro (Wang et al., 2024c).

[user] You are given a law problem and a proposed step-by-step solution:

[Law Problem] Question: A plaintiff sued a defendant for injuries that the plaintiff suffered when he was struck by the defendant’s car. At trial, the plaintiff testified that he was walking across the street inside a crosswalk at the time the defendant’s car hit him. [OMITTED...] In support of the defendant’s testimony, the defendant calls a neighbor to testify that she has observed the plaintiff frequently during the years that she has lived on that street, and that the plaintiff ”always crosses the street outside the crosswalk.” The neighbor’s testimony is likely to be A. admitted, because it tends to show that the plaintiff was careless. [OMITTED...]

- G. admitted, because it is probative of the plaintiff’s habit of crossing the street outside the crosswalk.
- H. excluded, because it is irrelevant to the specific incident.
- I. admitted, because it provides context to the plaintiff’s behavior. [Solution]

- Step 1: The neighbor’s testimony is about the plaintiff’s habit of crossing the street outside the crosswalk, which is relevant to the case.
- Step 2: Habit evidence is admissible to prove that a person regularly acts in conformity with the habit.
- Step 3: The neighbor’s testimony is an example of habit evidence, as it describes the plaintiff’s regular behavior of crossing the street outside the crosswalk.
- Step 4: The answer is (G).

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps. [/user] [assistant] [think] Let’s verify step by step:

First, the problem is about whether the neighbor’s testimony about the plaintiff’s habit is admissible. The proposed solution says step 1 is correct because the testimony is about habit and relevant. That seems right since the case hinges on where the plaintiff was crossing.

- Step 2 states habit evidence is admissible to show conformity. Under Federal Rules of Evidence, habit (as opposed to character) is admissible. So step 2 is correct.
- Step 3 says the neighbor’s testimony is habit evidence. But wait, habit requires a regular, routine practice. The neighbor says “frequently” over years, which might qualify as a habit. So step 3 is correct.
- Step 4 concludes G is correct. So all steps seem okay. Wait, but maybe “frequently” isn’t enough? Some jurisdictions require more specific regularity. But generally, habit evidence can be admissible if it’s a settled pattern. So steps are correct. Hmm, but let me check again. The options: G says “probative of plaintiff’s habit”. So yes, the steps are correct. The answer should be G. So all steps are correct. [/think]

- Step 1: The step is \boxed{correct}
- Step 2: The step is \boxed{correct}
- Step 3: The step is \boxed{correct}
- Step 4: The step is \boxed{correct} Is the solution correct? Yes [assistant]

- Figure 24: Training example of gPRM on the law domain of MMLU-pro (Wang et al., 2024c).

- F ADDITIONAL RESULTS

In this section, we present additional results as follows:

- • Fig. 25: Weighted majority voting results using Llama-3.1-8B-Instruct on MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.
- • Fig. 26: Best-of-N results using Llama-3.1-8B-Instruct when trained and evaluated on each domain of MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.
- • Fig. 27: Weighted majority voting results using Llama-3.1-8B-Instruct when trained and evaluated on each domain of MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.

- • Fig. 28: Best-of-N results using Llama-3.1-8B-Instruct on MMLU-Pro with R1-Distill-Llama8B backbone for reward models.
- • Fig. 29: Weighted majority voting results using Llama-3.1-8B-Instruct on MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.
- • Fig. 30: Best-of-N results using SmolLM3-3B on MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.
- • Fig. 31: Weighted majority voting results using SmolLM3-3B on MMLU-Pro with R1-DistillLlama-8B backbone for reward models.
- • Fig. 32: Best-of-N results using Qwen2.5-7B-Instruct on MMLU-Pro with R1-Distill-Llama8B backbone for reward models.
- • Fig. 33: Weighted majority voting results using Qwen2.5-7B-Instruct on MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.
- • Fig. 34: Best-of-N results using gemma-2-9b-it on MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.
- • Fig. 35: Weighted majority voting results using gemma-2-9b-it on MMLU-Pro with R1-DistillLlama-8B backbone for reward models.
- • Fig. 36: Best-of-N results using Llama-3.1-70B-Instruct on MMLU-Pro with R1-DistillLlama-8B backbone for reward models.
- • Fig. 37: Weighted majority voting results using Llama-3.1-8B-Instruct on MMLU-Pro with R1-Distill-Llama-8B backbone for reward models.

Majority Voting dORM dPRM gORM gPRM

| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>|
| |
| |

- 54

60

66

Overall

36

42

48

| |
|---|

| |
|---|

| |
|---|

Law

| | |
|---|---|
| || |
|---|
<br><br>|
| | |
| | |

60

65

70

Psychology

50

60

70

| |
|---|

| |
|---|

Chemistry

72

78

84

| |
|---|

| |
|---|

Biology

50

60

70

80

| |
|---|

| |
|---|

| |
|---|

Physics

48

52

56

| |
|---|

| |
|---|

| |
|---|

History

| | |
|---|---|
| || |
|---|
<br><br>|
| | |
| | |

60

66

72

Economics

50

60

70

80

| |
|---|

| |
|---|

Math

54

60

66

72

| |
|---|

Business

1 2 4 8 16

45

50

- 55

Taskaccuracy(%)

Philosophy

Health

Engineering

Computer Science

Other

72

64

68

| | | | | | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| | | | |
| || |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |

65

66

| |
|---|

| |
|---|

| |
|---|

| |
|---|

56

64

| |
|---|

60

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

| |
|---|

48

60

55

| |
|---|

54

40

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

- Figure 25: Weighted majority voting results using Llama-3.1-8B-Instruct on MMLU-Pro with R1-distillLlama-8B backbone for reward models.

Majority Voting dORM dPRM gORM gPRM

- 54

60

66

72

| |
|---|

| |
|---|

Overall

| | |
|---|---|
| | |
| | |
| || |
|---|
<br><br>|

36

42

48

Law

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

60

64

68

72

Psychology

| | | |
|---|---|---|
| | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
| | | |

50

60

70

Chemistry

72

78

84

| |
|---|

| |
|---|

| |
|---|

Biology

50

60

70

80

| |
|---|

| |
|---|

| |
|---|

Physics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

48

52

56

History

60

66

72

78

| |
|---|

| |
|---|

| |
|---|

Economics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|

50

60

70

80

Math

56

64

72

| |
|---|

| |
|---|

| |
|---|

Business

1 2 4 8 16

45

50

- 55

Taskaccuracy(%)

Philosophy

Health

Engineering

Computer Science

Other

64

60

72

| | | | | |
|---|---|---|---|---|
| | | | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| | | |
| | | | | |

64

66

56

64

60

63

| |
|---|

| |
|---|

| |
|---|

56

| |
|---|

48

60

| |
|---|

56

| |
|---|

| |
|---|

| |
|---|

52

57

40

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

###### Figure 26: Best-of-N performance using Llama-3.1-8B-Instruct when trained and evaluated on each domain of MMLU-Pro with R1-distilled-Llama-8B backbone for reward models.

| |
|---|
| |
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| |

- 54

60

66

72 Overall

| | |
|---|---|
| | |
| | |
| || |
|---|
<br><br>|

36

42

48

Law

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

60

64

68

72

Psychology

| | | |
|---|---|---|
| | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
| | | |

50

60

70

Chemistry

72

78

84

| |
|---|

| |
|---|

| |
|---|

Biology

50

60

70

80

| |
|---|

| |
|---|

| |
|---|

Physics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

48

52

56

History

60

66

72

78

| |
|---|

| |
|---|

| |
|---|

Economics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|

50

60

70

80

Math

56

64

72

| |
|---|

| |
|---|

| |
|---|

Business

1 2 4 8 16

45

50

- 55

Taskaccuracy(%)

Philosophy

Health

Engineering

Computer Science

Other

64

60

72

| | | | | |
|---|---|---|---|---|
| | | | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| | | |
| | | | | |

64

66

56

64

60

63

| |
|---|

| |
|---|

| |
|---|

56

| |
|---|

48

60

| |
|---|

56

| |
|---|

| |
|---|

| |
|---|

52

57

40

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

###### Figure 27: Weighted majority voting performance using Llama-3.1-8B-Instruct when trained and evaluated on each domain of MMLU-Pro with R1-distilled-Llama-8B backbone for reward models.

Law

Psychology

Chemistry

Biology

Overall

85

68

| | | | |
|---|---|---|---|
| | | | |
|| |
|---|
<br><br>| | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

65

70

45

80

| |
|---|

| |
|---|

64

60

| |
|---|

40

60

75

55

60

35

50

70

50

Physics

History

Economics

Math

Business

75

Taskaccuracy(%)

70

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

80

70

70

51

| |
|---|

65

| |
|---|

70

65

60

| |
|---|

| |
|---|

| |
|---|

48

60

| |
|---|

| |
|---|

60

60

50

55

45

50

Philosophy

Health

Engineering

Computer Science

Other

64

55

| |
|---|

65

60

| |
|---|

54

| |
|---|

60

60

57

50

| |
|---|

| |
|---|

48

| |
|---|

| |
|---|

55

56

54

| |
|---|

45

| |
|---|

| |
|---|

| |
|---|

42

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

51

52

| |
|---|

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

###### Figure 28: Best-of-N results using Llama-3.1-8B-Instruct on MMLU-Pro (Wang et al., 2024c) with R1distill-Llama-8B backbone for reward models.

Majority Voting dORM dPRM gORM gPRM

| | | | |
|---|---|---|---|
| | | | |
|| |
|---|
<br><br>| | | |
| | | | |
| | | | |

- 55

60

65

Overall

32

36

40

44

| |
|---|

| |
|---|

Law

60

64

68

| |
|---|

Psychology

48

56

64

72

| |
|---|

| |
|---|

Chemistry

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

70

75

80

Biology

| | | | | |
|---|---|---|---|---|
| | | | | |
| || |
|---|
<br><br>| | | |
| | | | | |
| | | | | |

48

- 56

50

Physics

History

Economics

Math

Business

Taskaccuracy(%)

70

54

80

72

68

| |
|---|

65

51

70

64

64

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

48

| |
|---|

60

60

55

45

56

50

Philosophy

Health

Engineering

Computer Science

Other

65

60

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| || |
|---|
<br><br>| | | |
| | | | | |
| | | | | |

52

60

64

| |
|---|

60

| |
|---|

54

57

48

60

| |
|---|

| |
|---|

| |
|---|

48

55

54

44

| |
|---|

| |
|---|

56

| |
|---|

42

50

51

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

- Figure 29: Weighted majority voting results using Llama-3.1-8B-Instruct on MMLU-Pro (Wang et al., 2024c) with R1-distill-Llama-8B backbone for reward models.

| |
|---|
|| |
|---|
<br><br>|
|| |
|---|
<br><br>|
| |

40

48

56

64

Overall

20

25

30

35

| |
|---|

| |
|---|

| |
|---|

Law

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

48

56

64

Psychology

50

60

70

| |
|---|

| |
|---|

| |
|---|

Chemistry

64

72

80

88

| |
|---|

| |
|---|

Biology

50

60

70

| |
|---|

| |
|---|

Physics

| | |
|---|---|
| || |
|---|
<br><br>|
| | |
| | |

36

42

48

54

History

| | |
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |
| | |

48

56

64

72

Economics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

60

70

80

Math

| | |
|---|---|
| | |
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|

48

56

64

72

Business

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

1 2 4 8 16

32

40

48

56

Philosophy

| || |
|---|
<br><br>|
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

1 2 4 8 16

32

40

48

56

Health

1 2 4 8 16

40

50

60

| |
|---|

| |
|---|

Engineering

1 2 4 8 16

50

60

70

| |
|---|

Computer Science

1 2 4 8 16

40

48

56

| |
|---|

| |
|---|

Other

Number of CoTs (N)

Taskaccuracy(%)

- Figure 30: Best-of-N results using SmolLM3-3B on MMLU-Pro with R1-distill-Llama-14B backbone for reward models.

| |
|---|
|| |
|---|
<br><br>|
|| |
|---|
<br><br>|
| |

40

48

56

64

Overall

20

25

30

35

| |
|---|

| |
|---|

| |
|---|

Law

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

48

56

64

Psychology

50

60

70

| |
|---|

| |
|---|

| |
|---|

Chemistry

64

72

80

88

| |
|---|

| |
|---|

Biology

50

60

70

| |
|---|

| |
|---|

Physics

| | |
|---|---|
| || |
|---|
<br><br>|
| | |
| | |

36

42

48

54

History

| | |
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |
| | |

48

56

64

72

Economics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

60

70

80

Math

| | |
|---|---|
| | |
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|

48

56

64

72

Business

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

1 2 4 8 16

32

40

48

56

Philosophy

| || |
|---|
<br><br>|
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

1 2 4 8 16

32

40

48

56

Health

1 2 4 8 16

40

50

60

| |
|---|

| |
|---|

Engineering

1 2 4 8 16

50

60

70

| |
|---|

Computer Science

1 2 4 8 16

40

48

56

| |
|---|

| |
|---|

Other

Number of CoTs (N)

Taskaccuracy(%)

- Figure 31: Weighted majority voting results using SmolLM3-3B on MMLU-Pro with R1-distill-Llama-14B backbone for reward models.

Majority Voting dORM dPRM gORM gPRM

Law

Psychology

Chemistry

Biology

###### Overall

72

78

| |
|---|
|| |
|---|
<br><br>|
| |
| |

| | |
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

70

48

84

69

72

| |
|---|

65

42

| |
|---|

| |
|---|

80

66

66

36

60

63

76

60

30

Physics

History

Economics

Math

Business

80

Taskaccuracy(%)

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

90

| |
|---|

80

75

75

| |
|---|

56

| |
|---|

84

| |
|---|

72

70

| |
|---|

| |
|---|

70

52

78

| |
|---|

65

64

65

48

72

Philosophy

Health

Engineering

Computer Science

Other

75

| || |
|---|
<br><br>| | | | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| | | | |
| || |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |

64

64

60

64

70

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

55

60

56

| |
|---|

| |
|---|

65

| |
|---|

| |
|---|

| |
|---|

| |
|---|

56

50

48

56

60

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

- Figure 32: Best-of-N results using Qwen2.5-7B-Instruct on MMLU-Pro with R1-distilled-Llama-14B backbone for reward models.

| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>|
| |
| |

56

60

64

68

Overall

30

36

42

48

| |
|---|

Law

63

66

69

72

| |
|---|

| |
|---|

Psychology

60

66

72

| |
|---|

| |
|---|

Chemistry

76

80

84

88

| |
|---|

| |
|---|

| |
|---|

Biology

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

64

72

80

Physics

48

51

54

57

| |
|---|

History

64

68

72

76

| |
|---|

Economics

72

78

84

90

| |
|---|

| |
|---|

| |
|---|

Math

65

70

75

| |
|---|

Business

1 2 4 8 16

48

52

56

60

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Philosophy

1 2 4 8 16

56

60

64

| |
|---|

| |
|---|

Health

1 2 4 8 16

48

56

64

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Engineering

1 2 4 8 16

60

65

70

| |
|---|

| |
|---|

| |
|---|

Computer Science

1 2 4 8 16

56

60

64

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Other

Number of CoTs (N)

Taskaccuracy(%)

- Figure 33: Weighted majority voting results using Qwen2.5-7B-Instruct on MMLU-Pro with R1-distillLlama-14B backbone for reward models.

| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>|
|| |
|---|
<br><br>|
| |

52

56

60

64

Overall

| | |
|---|---|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |
| | |

36

40

44

48

Law

57

60

63

66

| |
|---|

| |
|---|

Psychology

| || |
|---|
<br><br>|
|---|---|
| | |
| || |
|---|
<br><br>|
| | |

48

54

60

66

Chemistry

81

84

87

90

| |
|---|

Biology

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
|

48

56

64

72

Physics

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

51

54

57

History

| || |
|---|
<br><br>| |
|---|
|
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |

64

68

72

Economics

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>|

56

64

72

Math

55

60

65

| |
|---|

| |
|---|

| |
|---|

Business

1 2 4 8 16

44

48

52

| |
|---|

| |
|---|

| |
|---|

Philosophy

1 2 4 8 16

56

60

64

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Health

1 2 4 8 16

36

42

48

54

| |
|---|

Engineering

1 2 4 8 16

50

55

60

65

| |
|---|

Computer Science

1 2 4 8 16

56

60

64

| |
|---|

| |
|---|

| |
|---|

Other

Number of CoTs (N)

Taskaccuracy(%)

- Figure 34: Best-of-N results using gemma-2-9b-it on MMLU-Pro with R1-distill-Llama-14B backbone for reward models.

Majority Voting dORM dPRM gORM gPRM

Law

Psychology

Chemistry

Biology

###### Overall

64

48

| |
|---|
| |
| |

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |
| | |

| | |
|---|---|
| || |
|---|
<br><br>|
| | |
| | |
| | |

90

66

| |
|---|

64

| |
|---|

60

44

| |
|---|

87

63

40

| |
|---|

56

56

84

60

36

81

52

48

57

Physics

History

Economics

Math

Business

Taskaccuracy(%)

72

72

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
|

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| | |

57

72

65

| |
|---|

64

| |
|---|

| |
|---|

68

| |
|---|

54

66

60

| |
|---|

56

64

60

51

55

| |
|---|

48

54

Philosophy

Health

Engineering

Computer Science

Other

64

64

| || |
|---|
<br><br>| |
|---|
<br><br>| | | | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| | | | |
| || |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |

63

50

| |
|---|

48

| |
|---|

60

60

| |
|---|

45

60

| |
|---|

| |
|---|

| |
|---|

56

45

40

56

| |
|---|

57

52

42

35

54

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

1 2 4 8 16

Number of CoTs (N)

- Figure 35: Weighted majority voting results using gemma-2-9b-it on MMLU-Pro with R1-distill-Llama14B backbone for reward models.

| |
|---|
| |
| |
| |

64

- 68

72

76

Overall

| | |
|---|---|
| | |
| || |
|---|
<br><br>|
| | |
| | |

48

52

56

60

Law

72

75

78

81

| |
|---|

| |
|---|

Psychology

64

72

80

| |
|---|

Chemistry

84

88

92

| |
|---|

| |
|---|

| |
|---|

Biology

| | |
|---|---|
| | |
| | |
| | |

72

80

88

Physics

63

66

69

| |
|---|

| |
|---|

History

75

78

81

84

| |
|---|

Economics

64

72

80

88

| |
|---|

| |
|---|

| |
|---|

Math

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| | |
| | |

70

75

80

85 Business

1 2 4 8 16

60

63

66

- 69

| |
|---|

Philosophy

1 2 4 8 16

68

70

72

74

| |
|---|

| |
|---|

| |
|---|

Health

1 2 4 8 16

54

60

66

| |
|---|

| |
|---|

Engineering

1 2 4 8 16

68

72

76

80

| |
|---|

| |
|---|

Computer Science

1 2 4 8 16

68

72

76

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Other

Number of CoTs (N)

Taskaccuracy(%)

- Figure 36: Best-of-N results using Llama-3.1-70B-Instruct on MMLU-Pro with R1-distill-Llama-14B backbone for reward models.

| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>|
| |
| |

64

68

72

76

Overall

48

52

56

60

| |
|---|

| |
|---|

Law

| | |
|---|---|
| | |
| | |
| | |
| | |

72

75

78

81

Psychology

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

64

72

80

Chemistry

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |
| | |

81

84

87

90

Biology

| | |
|---|---|
| | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| | |
| | |

66

72

78

84

Physics

| || |
|---|
<br><br>|
|---|---|
| || |
|---|
<br><br>|
| | |

66

69

72

History

| | |
|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| | |
| | |

76

80

84

Economics

64

72

80

88

| |
|---|

| |
|---|

Math

70

75

80

| |
|---|

Business

1 2 4 8 16

60

64

68

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Philosophy

1 2 4 8 16

70

72

74

| |
|---|

| |
|---|

Health

1 2 4 8 16

54

60

66

| |
|---|

Engineering

1 2 4 8 16

68

72

76

80

| |
|---|

Computer Science

1 2 4 8 16

66

69

72

Other

Number of CoTs (N)

Taskaccuracy(%)

- Figure 37: Weighted majority voting results using Llama-3.1-70B-Instruct on MMLU-Pro with R1-distillLlama-14B backbone for reward models.

- G ADDITIONAL ANALYSIS

In this section, we present additional analysis on the failure of PRMs.

- • Fig. 38: Majority voting results of Llama-3.1-8B-Instruct vs. CoT length on MMLU-pro.
- • Fig. 39 (the full version of Fig. 9): results on “aha” CoTs in ProcessBench using R1-DistillQwen-1.5B and -7B.
- • Fig. 40 (the full version of Fig. 13): Outcome-verification results of PRMs vs. label noise on GSM8K.
- • Tab. 3: Wasserstein distance in the multi-domain setting before and after filtering for gORM and gPRM.

- • Tab. 4: Wasserstein distance in the math domain before and after filtering for gORM and gPRM.

64.3

66

61.5

61.5

MajorityVoting(%)

58.1 57.8

60

56.1

54

48.1

48

41.9

42

[2,3) [3,4) [4,5) [5,5) [5,7) [7,9) [9,15) [15,63)

###### Figure 38: Majority voting results of Llama-3.1-8B-Instruct vs. CoT length on MMLU-pro.

1.5B - Overall (15.3% = 521/3400)

###### 1.5B - GSM8K (1.8% = 7/400)

1.5B - Math (9.4% = 94/1000)

1.5B - Omni-Math (25.9 = 259/100)

1.5B - OlympiadBench (16.1% = 161/1000)

90

80

95 92.3

92.3

85.1

81.8

81.1

71.2

80

80

75

90

57.5

67.4

62.4

60

86.3

57.4

60

60

83.3

85

60

39.8

40

50.2

39.0

38.9

F1Score(%)

47.1

40

28.9

32.9

34.6

40

80

45

7B - Overall (15.3% = 521/3400)

###### 7B - GSM8K (1.8% = 7/400)

7B - Math (9.4% = 94/1000)

7B - Omni-Math (25.9 = 259/100)

7B - OlympiadBench (16.1% = 161/1000)

100

100

87.6

87.3

100 92.3

78.1

85.6

89.3

80

80

72.7

61.5

75

75

65.4

75

69.2

60

57.5

60

50

50

50

40

32.7

27.0

39.0

23.7

35.8

19.5

18.8

18.8

18.6

40

25

25

25

20

dORMdPRMgORMgPRM

dORMdPRMgORMgPRM

dORMdPRMgORMgPRM

dORMdPRMgORMgPRM

dORMdPRMgORMgPRM

Figure 39: Performance on “aha” CoTs in ProcessBench using R1-distill-Qwen-1.5B and -7B.

62.5 65.0 67.5 70.0 72.5 75.0 77.5 80.0

65 70 75 80 85

| | | | | | |
|---|---|---|---|---|---|
|[Figure 3]| | | | | |

[Figure 4]

1.5B dPRM (Range: 67.23-81.20)

1.5B gPRM (Range: 77.78-80.39)

7B dPRM (Range: 71.51-88.31)

7B gPRM (Range: 86.59-88.59)

Process-noiseratio(%)

50%

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

40%

30%

20%

10%

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

10% 20% 30% 40% 50%

Data-noise ratio (%)

###### Figure 40: Outcome-verification results of PRMs vs. label noise on GSM8K.

###### Table 3: Wasserstein distance in the multi-domain setting before and after filtering for gORM and gPRM.

###### Overall Law Psychology Chemistry Biology Physics History Economics

Train 0.202 0.090 0.203 0.393 0.264 0.628 0.069 0.311 gORM 0.532 0.089 0.218 1.128 0.506 1.201 0.154 0.564

- gPRM 3.083 1.284 0.742 6.922 2.039 5.952 0.581 1.782 Math Business Philosophy Health Engineering CS Other

Train 0.167 0.322 0.129 0.105 1.234 0.353 0.093 gORM 0.282 0.491 0.545 0.213 3.611 0.338 0.312

- gPRM 4.655 4.267 1.235 0.979 12.735 3.459 0.927

###### Table 4: Wasserstein distance in the math domain before and after filtering for gORM and gPRM.

###### Overall GSM8K Math Omni-Math OlympiadBench

Train (PRM800K) 2.760 5.113 3.813 2.027 1.514 gORM 2.430 4.780 3.480 1.695 1.194 gPRM 1.600 3.680 2.348 1.448 1.203

### H THE USE OF LLMS

We used LLMs solely for light editing such as correcting grammatical errors and polishing some words. They did not contribute to research ideation, experiments, analysis, or substantive writing.

