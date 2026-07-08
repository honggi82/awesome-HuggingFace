### LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?

Jingyuan Wang1* Yankai Chen2* Zhonghang Li1 Chao Huang1† 1University of Hong Kong 2University of Chicago {jameswangjingyuan,xiaoshu0712,bjdwh.zzh,chaohuang75}@gmail.com

# arXiv:2510.07962v2[cs.CL]21May2026

#### Abstract

Large language models (LLMs) have demonstrated remarkable progress in reasoning, often through supervised fine-tuning (SFT). However, SFT is resource-intensive, relying on large curated datasets, rejection-sampled demonstrations, and uniform optimization across all tokens—even though only a fraction carry meaningful learning value. In this work, we explore a counterintuitive idea: can smaller language models (SLMs) teach larger language models (LLMs) by revealing high-value reasoning moments that reflect the latter’s unique strength? We propose LightReasoner1, a novel framework that leverages the behavioral divergence between a stronger expert model (LLM) and a weaker amateur model (SLM). LightReasoner operates in two stages: (1) a sampling stage that pinpoints critical reasoning moments and constructs supervision examples capturing the expert’s advantage through expert–amateur contrast, and (2) a fine-tuning stage that aligns the expert model with these distilled examples, amplifying its reasoning strengths. Across seven benchmarks, LightReasoner improves accuracy by up to 28.1%, while reducing time consumption by 90%, sampled problems by 80%, and tuned token usage by 99%, all without relying on ground-truth labels. By turning weaker SLMs into effective teaching signals, LightReasoner offers a scalable and resource-efficient approach for advancing LLM reasoning.

#### 1 Introduction

Large language models (LLMs) have achieved remarkable progress in language understanding and generation (Kaplan et al., 2020; Touvron et al., 2023). However, they continue to struggle with systematic reasoning tasks that demand step-by-step logical precision. A prevalent strategy to address this gap is to align models with demonstrations of

*Equal contribution. †Corresponding author. 1https://github.com/HKUDS/LightReasoner

Sampled Problems

Sampled Problems

###### Qwen2.5-Math-1.5B

###### Qwen2.5-Math-7B

4K probs

6K Probs

+ LightReasoner

+ LightReasoner

| |
|---|

| |
|---|

+ SFT

+ SFT

| |
|---|

| |
|---|

1K probs

1K Probs

4 hours 0.5 hours

9.5 Hours 0.75 Hours

Total Time

Total Time

+7.7%

+4.5%

+11.8%

+4.7%

Performance Gain

Performance Gain

(a) Qwen2.5-Math-1.5B

(b) Qwen2.5-Math-7B

Sampled Problems

Sampled Problems

###### Qwen2.5-Math-1.5B-Ins

###### DeepSeek-R1-Distill-1.5B

6K Probs

7K Probs

+ LightReasoner

| |
|---|

+ LightReasoner

| |
|---|

+ SFT

| |
|---|

+ SFT

| |
|---|

1K Probs

1K Probs

3.6 Hours 0.5 Hours

3.4 Hours 0.4 Hours

Total Time

Total Time

+3.0%

+5.6%

+0.1%+0.1%

Performance Gain

Performance Gain

(c) DeepSeek-R1-1.5B

(d) Qwen2.5-Math-1.5B-Instruct

Figure 1: LightReasoner achieves competitive or superior performance compared to SFT while substantially reducing resource consumption.

correct problem-solving trajectories. To this end, recent works (Yang et al., 2024; Guo et al., 2025) often integrate rejection sampling (Yuan et al., 2023) into supervised fine-tuning (SFT). While effective, this strategy is resource-intensive: it requires generating multiple candidate solutions, filtering them against ground-truth answers to retain only correct trajectories, and then fine-tuning on every token of those trajectories, treating trivial and crucial reasoning steps as equally valuable.

These limitations have motivated more targeted strategies for improving LLM reasoning. Early as the Chain-of-Thought paper (Wei et al., 2022), it was shown that LLMs acquire latent reasoning abilities during pre-training, which can be elicited through appropriate prompting. Recent works (Zhao et al., 2025; Prabhudesai et al., 2025) show that models can refine themselves using internal feedback signals, such as self-certainty, without relying on external rewards or labeled data. Another

line of research (Lin et al., 2024a) finds that during pre-training, learning progress is concentrated on a small subset of tokens, suggesting that selective optimization on these high-impact tokens can yield significant gains in efficiency and performance. Together, these insights highlight the promise of harnessing underexplored internal learning dynamics to improve LLM reasoning. For completeness, a detailed discussion of related work is in Appendix A.

Motivated by these insights, we pose a counterintuitive possibility: can smaller, weaker models help guide the training of their larger, stronger counterparts? Specifically, we ask whether the divergent behaviors between a strong expert model and a weaker amateur model can pinpoint high-impact reasoning moments, and whether these moments can be transformed into effective training signals. In §2.2, we analyze this Expert-Amateur dynamics through the Kullback–Leibler (KL) divergence between their next-token predictions: when models of different capabilities disagree strongly, those moments often mark critical steps in problem-solving. Our analysis shows that such steps, where the Expert confidently predicts the correct token while the Amateur diverges or shows uncertainty, are signaled by pronounced KL values. Exploiting these moments yields targeted supervision signals that reinforce the Expert’s reasoning strengths.

Building on these analyses, we introduce LightReasoner, a novel framework that leverages expert-amateur divergence to enhance LLM reasoning. LightReasoner operates in two stages: (1) Sampling stage. For each reasoning trajectory, Expert and Amateur models generate next-token predictions under identical prefixes. Steps where the KL divergence between their distributions exceeds a threshold are retained (§2.3.1). From these selected steps, we construct supervision examples that encode Expert’s relative advantage by contrasting the two models’ next-token distributions (§2.3.2). (2) Fine-tuning stage. The same Expert model is then trained to align with these contrastive signals, increasing probability of tokens where its advantage over Amateur is most pronounced, thereby reinforcing its strengths and avoiding amateur-like tendencies (§2.3.3).

Our experimental evaluation highlights four key advantages of LightReasoner: • Strong Performance Gains (§3.2). LightReasoner matches or surpasses SFT under equal experimental settings across 5 models and 7 benchmarks. • Orderof-Magnitude Efficiency (§3.3). LightReasoner

delivers substantial savings: reducing total time cost by 90%, requiring 80% fewer sampled problems, and tuning 99% fewer tokens compared to SFT2 with rejection sampling, while entirely eliminating the need for ground-truth labels. • Domain Expertise Drives Effective Contrasts (§3.4). Our analysis shows that the most effective Expert–Amateur contrasts stem from domain-specific expertise differences, reinforcing our insight that weaker models can provide meaningful contrast signals for teaching stronger ones. • Synergistic Design (§3.5). Ablation studies confirm the critical roles of step selection and contrastive supervision, along with their mutually-reinforcing effect, in enabling LightReasoner to strengthen the expert model’s reasoning efficiently and reliably.

#### 2 Methodology

##### 2.1 Preliminaries

Autoregressive Language Model Generation. Given vocabulary A and input a0, a language model generates a response a1:T = [a1,...,aT] autoregressively. At each step t, the model predicts the next token based on the prefix st = [a0,...,at−1] and outputs distribution πLM(· | st) over A. The joint likelihood factorizes as:

T

πLM(at | st). (1)

P(a1:T | a0) =

t=1

The quality of reasoning emerges from the accumulation of individual token-level decisions. Thus, improving a model’s reasoning ability fundamentally amounts to refining its policy πLM. A central challenge, then, is to determine which token-level decisions truly matter, and how to provide learning signals that target them effectively.

Learning from Behavioral Divergence. Existing approaches often depend on human annotations or external verification mechanisms, which suffer from resource limitations, hindering their practicality for continuous improvement. We observe that models of different capabilities exhibit systematic differences in their token-level decision patterns. This insight motivates our approach: leveraging behavioral divergence between models to automatically identify critical decision points and extract effective learning signals.

2Unless otherwise stated, SFT in this paper refers to supervised fine-tuning on rejection-sampled model trajectories (also known as rejection sampling fine-tuning, RFT).

KL Divergence Histogram

59.1%

Tail Detail: KL 2.0

1.0

60

PercentageofTokens(%)

0.8

50

0.43%

0.6

0.34%

40

0.23%

0.4

0.14%

0.12%

0.12%

0.11%

0.11%

0.09%

0.09%

0.07%

0.07%

0.07%

0.05%

0.2

30

0.0

...

[2.0,2.2)

[2.2,2.4)

[2.4,2.6)

[2.6,2.8)

[2.8,3.0)

[3.0,3.2)

[3.2,3.4)

[3.4,3.6)

[3.6,3.8)

[3.8,4.0)

[4.0,4.2)

[4.2,4.4)

[4.4,4.6)

[4.6,4.8)

20

10.5%

6.1%

10

3.8%

3.7%

3.3%

2.1%

1.8%

1.6%

1.3%

0

[2.0,2.0+]

[0.0,0.1)

[0.1,0.2)

[0.2,0.3)

[0.3,0.4)

[0.4,0.5)

[0.5,0.6)

[0.6,0.7)

[0.7,0.8)

[0.8,0.9)

[0.9,1.0)

[1.0,1.1)

[1.1,1.2)

[1.2,1.3)

[1.3,1.4)

[1.4,1.5)

[1.5,1.6)

[1.6,1.7)

[1.7,1.8)

[1.8,1.9)

[1.9,2.0)

KL Divergence Bin

Figure 2: Most tokens show minimal KL divergence, only few exhibiting elevated values.

Critical Decision Points in Reasoning. The foundation of our approach lies in the observation that reasoning ability is shaped not by uniform performance across all tokens, but by a handful of critical decision points. Cognitive science shows that certain reasoning steps exert disproportionate influence on final outcomes, creating natural bottlenecks in reasoning chains (Chi et al., 1981; Ericsson and Smith, 1991). Recent studies confirm that language models benefit most from training on such critical token subsets (Lin et al., 2024a,b).

We therefore propose that targeting these highstakes decision points creates a leverage effect: small improvements at bottlenecks can yield large overall gains. Our method operationalizes this intuition by exploiting expert–amateur differences to identify reasoning bottlenecks and focus enhancement where it matters most. To this end, we leverage two models with distinct reasoning capabilities: (1) an Expert model πE, which we aim to improve, and (2) an Amateur model πA, serving as a weaker baseline. As πE generates a response a1:T through a sequence of prefixes s1:T, we evaluate both models at each step t on the same prefix st. This produces paired distributions (πE(· | st),πA(· | st)), whose divergences reveal exactly where the Expert departs from amateur-level reasoning, providing targeted supervisory signals that concentrate on consequential decisions.

##### 2.2 Token Informativeness

To pinpoint critical decision points, we quantify the disagreement between Expert and Amateur models at each generation step t using Kullback–Leibler (KL) divergence:

DKL(πE(· | st)∥πA(· | st))

πE(a | st) πA(a | st)

. (2)

πE(a | st) log

=

a∈A

- 0.0

0.5

1.0

- 1.5

- 2.0

2.5

3.0

1.11

0.75

2.98

0.44

- Reasoning segment 1 (bottom x-axis)

- Reasoning segment 2 (top x-axis)

- x-axis: Tokens from Expert Model
- y-axis: Expert-Amateur KLD

Thetotal

number

of

people

consumed

overthree

hundred years is thesum

ofthe

people onthe

threeships ,

which

is \( x +

0.86

0.45 0.45

0.80

2.18

1.00 0.95

Figure 3: Predictable tokens yield near-zero KL divergence, while critical steps trigger notable spikes.

Large KL values signals reasoning bottlenecks where the Expert departs sharply from amateurlevel choices. In mathematical reasoning, these often recur at key transitional steps (Ji et al., 2025; Prabhudesai et al., 2025). Our analysis highlights three patterns that validate this perspective:

- • Critical Tokens (Figure 2): 60% of tokens exhibit negligible divergence KLD ∈ [0.0,0.1), while only 20% exceed 0.4, confirming that reasoning bottlenecks arise at specific moments.
- • Reasoning Complexity (Figure 3): Case studies show that KL divergence stays near zero for routine tokens but spikes at demanding steps such as arithmetic operations and logical transitions, tracking underlying complexity.
- • Hidden Disagreements: When the Expert and Amateur disagree on top-1 tokens, the average KL jumps to 1.99, versus 0.166 under agreement. Even among top-1 agreements, 10% of cases still show KLD > 0.4, exposing hidden misalignments beneath apparent consensus.

2.3 LightReasoner Framework

We present LightReasoner (Figure 4), a novel self-distillation framework for enhancing LLM reasoning. It is built on three core components: (1) an informative step-selection mechanism that pinpoints critical reasoning steps via divergence-based metrics (§2.3.1); (2) a contrastive method for constructing supervision samples that capture the Expert model’s reasoning advantage (§2.3.2); (3) a training objective that distills the learning signal back into the Expert (§2.3.3). Full procedure in Algorithm 1.

- 2.3.1 Informative Step Selection Reasoning trajectories consist of tokens with uneven learning value. Routine steps often yield close

First ,convert 5 0minutestohours : 5 0 \text {minutes } = \frac { 5 0}{ 6 0 }

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### To find out…we need to convert…First, convert 50 minutes to hours: \\[50 \\text{minutes}=…

[Figure 5]

Expert

|E-A KLD|
|---|

|↑ KLD|
|---|

|same prefixes|
|---|

1K Problems (Weng earns $12 ... did she earn?)

[Figure 6]

KLD-based Selection

[Figure 7]

###### How find out…we need to multiply…5, we 50 minutes to hours: 5 50 \\text{minutes}=…

From the trajectory, filters out

Amateur

[Figure 8]

|Contrastive Samples<br><br>KLD Loss<br><br>LoRA<br><br>[Figure 9]<br><br>[Figure 10]<br><br>Expert<br><br>E’s output<br><br>Sample Label<br><br>20K samples<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

Expert

Encode E’s advantage

[Figure 13]

Prediction π

Fine Tuning

[Figure 14]

|Informative Reasoning steps|
|---|

log𝜋 − log𝜋

E-A Distribution Contrast

[Figure 15]

query

[Figure 16]

[Figure 17]

both

Contrastive Samples

Enhanced Expert Model

[Figure 18]

Prediction π

Amateur

- Figure 4: Overview of the LightReasoner framework. Sampling Stage: Expert and Amateur models generate distributions πE and πA. Informative step selection (§2.3.1) retains steps with DKL(πE ∥ πA) > β, and contrastive supervision (§2.3.2) constructs soft labels vC capturing the Expert’s advantage through Expert–Amateur contrast. Fine-tuning Stage: The Expert model is enhanced by minimizing the KLD between its output and vC (§2.3.3).

Expert–Amateur agreement, while valuable supervision arises at critical decision points where expert reasoning is decisive.

noisy tail probabilities from distorting the supervision signal. For each a ∈ Amask, we compute the unnormalized contrast score (Li et al., 2022):

To capture these moments, we introduce informative step selection, which uses model KL divergence to quantify reasoning criticality: large divergences mark bottlenecks where expert knowledge can be deterministic. We implement this via β-filtering, retaining only steps where the Expert’s advantage is pronounced. For a given prefix st, a step is selected if

πE(a | st) πA(a | st)

vC′ (a | st) = log

, (5) quantifying the Expert’s advantage margin.

We apply softmax to obtain a normalized distribution v˜C, then extend it over all tokens a ∈ A:

vC(a | st) = v˜C(a | st) · [a ∈ A mask], (6)

The resulting vC(· | st) encodes the expert superiority as probabilistic supervision, enabling targeted refinement that reinforces expert-like decisions.

DKL(πE(· | st) ∥ πA(· | st)) > β. (3)

- 2.3.2 Contrastive Distributional Supervision

2.3.3 Self-Distillation Training Objective

After identifying informative steps, we convert Expert–Amateur disagreement into effective training signals. Instead of one-hot targets, which discard distributional information, we construct soft targets that encode the Expert’s relative advantage. For a step st passing the β-filter, we first define the masked support set of model vocabulary A:

The framework is completed with a self-distillation objective that transfers the signal encoded in vC back into the Expert model. Specifically, the Expert distribution πE is trained to match vC by minimizing the KL divergence:

L(st) = DKL vC(· | st) πE(· | st)

(7)

vC(a | st) πE(a | st)

vC(a | st)log

=

.

Amask = a ∈ A : πE(a | st) ≥ α · max

a∈A

(4)

Since the first term is constant with respect to πE,

πE(b | st) ,

(7) is equivalent to cross-entropy minimization:

b∈A

L˜(st) = −

vC(a | st) log πE(a | st), (8)

where α ∈ (0,1] (Li et al., 2022) removes lowconfidence tokens from the vocabulary, preventing

a∈A

Qwen2.5-Math-1.5B (Base)

Qwen2.5-Math-1.5B + LightReasoner

| |
|---|

| |
|---|

100

+11.7

+7.2

80

Pass@1Accuracy(%)

+28.1

+25.1

60

+5.1

40

+3.4

20

+1.5

0

GSM8K MATH Minerva Math

Olympiad Bench

SVAMP ASDiv MMLU STEM

Qwen2.5-Math-7B (Base)

Qwen2.5-Math-7B + LightReasoner

| |
|---|

| |
|---|

100

+7.9

+9.3

80

Pass@1Accuracy(%)

+0.7

+10.4

+6.0

60

40

+0.9

20

-1.9

0

GSM8K MATH Minerva Math

Olympiad Bench

SVAMP ASDiv MMLU STEM

- Figure 5: LightReasoner consistently improves accuracy across 7 math benchmarks for baseline models.

which reinforces the Expert’s probability mass on tokens where its advantage over the Amateur is most pronounced.

Detailed math derivations, analysis of component synergy, and an intuitive explanation of how the method enhances reasoning are provided in Appendix C and Appendix D.

#### 3 Experiment

To assess the effectiveness of LightReasoner, we structure our experiments around four research questions: • RQ1: How does LightReasoner improve performance across different baseline models? • RQ2: How much does LightReasoner reduce computational costs? • RQ3: Which factors drive successful Expert–Amateur collaboration? • RQ4: How do the core components of LightReasoner contribute to its effectiveness?

- 3.1 Experimental Setup

- 3.1.1 Models and Data

LightReasoner leverages Expert–Amateur model pairing to generate supervision signals. The Expert models include Qwen2.5-Math-1.5B and 7B, their Instruct versions, and the DeepSeek-R1-Distill variant (Yang et al., 2024; Guo et al., 2025). The Amateur is fixed as Qwen2.5-0.5B (Yang et al., 2024), a base model with general linguistic reasoning ability but without specialized math training.

For generating supervision samples, we use the GSM8K training set (Cobbe et al., 2021), selected for its emphasis on step-by-step, broadly applicable logical reasoning. CoT prompting (Wei et al., 2022) is employed to elicit reasoning trajectories. Comprehensive model specifications and dataset descriptions are provided in Appendix H.1 and Appendix H.2.

- 3.1.2 Training Configuration We fine-tune the same Expert model on the curated LightReasoner supervision set using LoRA (Hu et al., 2022) for parameter-efficient training. Key hyperparameters are as follows: a masking threshold of α = 0.2 to filter out low-probability Expert tokens (Li et al., 2022); a KL divergence threshold of β = 0.4 to isolate informative reasoning steps; and the sampling rollout length capped at 128 tokens (ablation on the sampling length is provided in Appendix G.1), as early steps are typically less error-prone (Ji et al., 2025). Models are finetuned for 1000 steps, with each step comprising 16 contrastive supervision samples, which we find sufficient for convergence. Full configuration details are provided in Appendix H.3.
- 3.1.3 Design Rationale Despite the effectiveness of the step selection mechanism (§2.3.1), one potential concern arises when both models follow an incorrect reasoning path. In such cases, the resulting KL divergence may still be large, allowing these steps to pass the filtering and enter the supervision set as false positives. This reflects shared errors rather than genuine expertise differences, potentially reinforcing erroneous patterns in the Expert model.

We mitigate this risk through two design choices. First, we use GSM8K to generate supervision samples, as its step-wise reasoning and basic arithmetic largely fall within the Expert model’s capability, reducing the likelihood of systematic failure. At the same time, it ensures that the Amateur, despite lacking domain-specific training, can still produce coherent outputs, enabling meaningful contrast with the Expert. Second, we restrict sampling to short rollout prefixes rather than full trajectories, as prior work (Ji et al., 2025) shows that early reasoning steps are significantly more stable, while later steps are more prone to cascading errors.

- 3.1.4 Evaluation For competition, we implement SFT on the baseline models (Appendix H.4), LoRA fine-tuning on

Olympiad Bench

Model GSM8K MATH SVAMP ASDiv Minerva Math

MMLU

STEM AVG. Qwen2.5-Math-1.5B

Baseline 42.5 34.2 68.8 68.1 9.9 23.7 49.8 42.4 + SFT 69.2 57.1 64.1 70.2 15.1 27.6 47.7 50.1 + LightR 70.6 59.3 76.0 79.8 11.4 27.1 54.9 54.2

##### Qwen2.5-Math-1.5B-Instruct

Baseline 84.8 75.8 94.2 94.7 29.4 37.5 57.4 67.7 + SFT 85.4 75.8 93.5 94.7 31.6 37.5 56.2 67.8 + LightR 86.7 75.5 93.0 94.1 32.0 37.8 55.2 67.8

##### DeepSeek-R1-Distill-Qwen-1.5B

Baseline 75.2 54.2 79.9 84.9 16.2 19.1 22.3 50.3 + SFT 78.2 60.3 81.5 87.4 18.4 21.2 26.2 53.3 + LightR 79.5 60.2 83.5 87.5 18.0 36.5 26.2 55.9

##### Qwen2.5-Math-7B

Baseline 57.5 51.8 67.9 72.7 14.0 16.0 69.8 50.0 + SFT 64.4 63.3 76.2 76.6 12.1 20.5 68.5 54.5 + LightR 67.9 57.8 77.2 80.6 12.1 16.9 70.5 54.7

##### Qwen2.5-Math-7B-Instruct

Baseline 95.2 83.2 93.9 95.3 33.8 41.5 69.3 73.2 + SFT 95.4 83.1 94.1 95.2 38.2 40.7 68.2 73.6 + LightR 95.8 83.6 93.1 95.2 34.2 39.0 67.8 72.7

- Table 1: Zero-shot pass@14 accuracy (%) across evaluation benchmarks. LightReasoner consistently achieves comparable or superior performance over SFT across 5 baseline models and 7 mathematical benchmarks.

Method Time Probs. Tokens Gain Qwen2.5-Math-1.5B

+ SFT 4.0h 3952 1.77M +7.7%

+ LightR 0.5h 1000 0.02M +11.8% Qwen2.5-Math-7B

+ SFT 9.5h 6029 2.20M +4.5% + LightR 0.75h 1000 0.02M +4.7%

DeepSeek-R1-Distill-Qwen-1.5B

+ SFT 3.6h 6023 5.95M +3.0% + LightR 0.5h 1000 0.02M +5.6%

Qwen2.5-Math-1.5B-Instruct

+ SFT 3.4h 7153 2.08M +0.1% + LightR 0.4h 1000 0.02M +0.1%

- Table 2: Efficiency comparison across time, sampled problems, tuned tokens, and average performance gain.

###### Attribute Efficiency SFT LightR

Full sampling ↓ ✓ ✗ All-token tuning ↓ ✓ ✗ GT verification ↓ ✓ ✗ Prefix sampling ↑ ✗ ✓ Selective tuning ↑ ✗ ✓ Verification-free ↑ ✗ ✓

Table 3: SFT vs. LightReasoner – efficiency comparison at a glance. ↑ and ↓ indicate whether each aspect improves or reduces efficiency.

zero-shot pass@14as the primary metric. 3.2 Performance Improvements (RQ1)

Table 1 presents the zero-shot pass@14 accuracy across a diverse suite of math reasoning benchmarks. Across 5 models and 7 datasets, LightReasoner delivers consistent performance improvements and demonstrates fundamental advances in math reasoning capability. These results highlight LightReasoner’s strong generalization, which we analyze through three perspectives below.

demonstrations of correct reasoning trajectories collected via rejection sampling (Yuan et al., 2023). We evaluate the baseline models, the SFT-trained variants, and the LightReasoner fine-tuned counterparts on a diverse suite of benchmarks ranging from basic arithmetic to expert-level math reasoning (Appendix H.1). All evaluations are conducted with the Qwen2.5-Math toolkit3, and we report

##### • Cross-Dataset Reasoning Enhancement. De-

3https://github.com/QwenLM/Qwen2.5-Math

4MMLU STEM is 5-shot; all others are zero-shot.

Expertise Gap vs. Performance Gain

72.5

0.5B 1.5B

LightR (Expert: Math-1.5B , varied Amateur)

LightR (Expert: Math-7B , varied Amateur)

70.0

0.5B

Avg.AccuracyAcrossTasks(%)

67.5

1.5B

65.0

Expert-only Baseline (Math-7B)

Math-1.5B-Ins

62.5

Math-1.5B

60.0

57.5

55.0

Math-1.5B-Ins

Expert-only Baseline (Math-1.5B)

52.5

###### 50 40 30 20 10 0 10 20 30 40 50 60

Expertise Gap Between Expert & Amateur (%)

- Figure 6: Performance gains from LightReasoner diminish as the Expert-Amateur expertise gap narrows. Label above each data point denotes the Amateur model paired with the Expert.

Attribute Utility CD LightR Contrast usage / Inference Training Size-based contrast ↓ ✓ ✗ Expertise contrast ↑ ✗ ✓ Persistent improvement ↑ ✗ ✓ Independent inference ↑ ✗ ✓

- Table 4: Key differences between Contrastive Decoding (CD) and LightReasoner. ↑ and ↓ indicate whether each attribute improves or reduces the method’s practical applicability.

spite being trained exclusively on GSM8K, LightReasoner achieves consistent gains across diverse benchmarks including MATH, SVAMP, and ASDiv. This shows that LightReasoner cultivates foundational reasoning enhancement rather than memorizing dataset-specific heuristics. By focusing on reasoning bottlenecks where the expert diverges from amateur patterns, the method captures transferable logical structures that extend beyond the training domain.

- • Adaptive Enhancement across Models. Our approach delivers consistent improvements across models of different capacities. For non-instruct models (e.g., Qwen2.5-Math-1.5B), we observe dramatic gains (+28.1% on GSM8K; +25.1% on MATH), showing that contrastive supervision can activate latent reasoning circuits previously dormant. For heavily optimized instruct models (e.g., 1.5B-Instruct), the improvements are modest but steady (+1.9% on GSM8K; +2.6% on Minerva Math), suggesting our method refines existing reasoning pathways. This differential effectiveness highlights how the method adapts flexibly to varying model capabilities.
- • Superior Efficiency over SFT. Direct compar-

isons show that LightReasoner matches or surpasses SFT performance while consuming 90% less total time, 80% fewer sampled problems, and 99% fewer tuned tokens (§3.3). These efficiency gains stem from two synergistic components: step selection (§2.3.1), which concentrates learning on the ∼20% of tokens that drive reasoning (§2.2), and contrastive supervision (§2.3.2), which constructs training signals that encode the expert’s advantage over amateurish tendencies.

##### 3.3 Efficiency Study (RQ2)

We assess efficiency along three key dimensions: (1) Time budget: the total sampling and finetuning time on a single NVIDIA H200 GPU without inference accelerators; (2) Training instances: the number of distinct training set problems to generate the supervision dataset; (3) Tuned tokens: the computational overhead at the token level, where LightReasoner trains on selective nexttoken predictions while SFT optimizes over full reasoning trajectories. As shown in Table 2 and Figure 1, LightReasoner consistently outperforms SFT with only a fraction of the resources. These efficiency improvements stem from three core design principles of LightReasoner that directly address SFT’s computational bottlenecks:

- • Sampling efficiency via prefix termination. During rejection sampling, SFT must generate complete reasoning trajectories. LightReasoner, on the other hand, halts each sampling rollout at 128 tokens, sharply reducing the cost. This design leverages the observation that early reasoning steps provide more reliable signals with fewer cascading errors (Ji et al., 2025), avoiding computational expense on error-prone later steps.
- • Training efficiency through selective tokens. LightReasoner concentrates learning on highvalue reasoning moments rather than low-return tokens. This selectivity accounts for the large gap in token usage between SFT and LightReasoner. By avoiding SFT’s indiscriminate tuning over full trajectories, LightReasoner achieves faster and more focused reasoning improvement.
- • Data efficiency via verifier-free supervision. SFT relies on “generate-and-verify” loops that check against ground-truth answers, posing a major limitation. LightReasoner avoids this by using the Amateur as a contrastive baseline, turning relative performance gaps into supervision

Amateur Model ∆ Perf. GSM8K MATH SVAMP ASDiv MMLU

STEM AVG. Expert: Qwen2.5-Math-1.5B

- Qwen2.5-0.5B 38.2 70.6 59.3 76.0 79.8 54.9 68.1

- Qwen2.5-1.5B 35.1 63.4 57.1 69.7 75.7 54.8 64.1 Qwen2.5-Math-1.5B-Ins -42.3 41.4 35.5 67.5 66.4 55.0 53.2 Expert Only (Baseline) / 42.5 34.2 68.8 68.1 49.8 52.7

##### Expert: Qwen2.5-Math-7B

- Qwen2.5-0.5B 53.2 67.9 57.8 77.2 80.6 70.5 70.8

- Qwen2.5-1.5B 50.1 69.0 56.0 77.6 78.9 69.5 70.2 Qwen2.5-Math-1.5B 15.0 56.9 50.2 63.5 63.4 70.7 60.9 Qwen2.5-Math-1.5B-Ins -27.3 59.4 49.0 68.3 69.6 70.3 63.3 Expert Only (Baseline) / 57.5 51.8 67.9 72.7 69.8 63.9

- Table 5: Impact of expertise-driven contrast. We fix the Expert model and vary the Amateur; ∆Perf. denotes the Expert-Amateur performance difference on GSM8K; each group ends with the Expert baseline.

signals. This design removes the dependency on ground-truth verification and extends to domains where definitive solutions are unavailable. By decoupling learning from outcome validation, LightReasoner focuses on strengthening reasoning processes rather than merely outcomes.

##### 3.4 Domain Expertise Drives Contrast (RQ3)

Prior contrastive methods rely on rigid parametersize disparities to create Expert–Amateur contrast. For instance, OPT-13B vs. OPT-125M in Contrastive Decoding (CD) (Li et al., 2022) and LLaMA-65B vs. LLaMA-1.5B in its follow-up study (O’Brien and Lewis, 2023). Such setups restrict applicability and introducing substantial computational demands. Table 4 summarizes the key differences between contrastive decoding methods and LightReasoner. To overcome this limitation, we hypothesize that domain-specific expertise provides a more applicable axis of contrast than the raw model scale. To validate this hypothesis, we fixed the Expert model and systematically varied the Amateur, progressively narrowing their expertise gap. Table 5 and Figure 6 demonstrate how Expert–Amateur expertise relationships determine LightReasoner’s effectiveness and offers valuable insights into optimal model contrast pairings.

• Domain expertise as contrast axis. Our results demonstrate that domain-specific knowledge, rather than parameter count, is the primary driver of effective contrastive supervision. The clearest evidence comes from pairing the Qwen2.5-Math-1.5B expert with the generalist

Qwen2.5-1.5B amateur, which yields striking performance gains (+12.1% average gain) despite identical model sizes. This finding frees LightReasoner from rigid scale requirements and extends it to a broader range of models.

• Effectiveness depends on the expertise gap. As illustrated in Figure 6, performance gains are closely correlated with the size of the expertise gap. When Amateur capabilities approach those of the Expert, contrastive signals weaken and benefits diminish. In the extreme case of pairing the Math-1.5B or Math-7B Expert with the stronger Math-1.5B-Instruct model—where the expertise gap is effectively negative—fine-tuning yields negligible gains or even degradation, further proving that expertise differentials are essential for model contrastive learning.

##### 3.5 Ablation Study (RQ4)

To assess the contribution of each core component in the LightReasoner framework, we conducted a systematic ablation study. By progressively removing individual mechanisms and measuring the impacts, we isolate how each design contributes to overall performance. The results, presented in Table 6, highlight the role of every component and provide key insights into the design of LightReasoner. We summarize the findings below:

• Impact of Informative Step Selection. Removing step selection turns LightReasoner into a full-sampling pipeline without KL-based pruning. This ablation led to a clear performance decline (e.g., –3.0% on GSM8K), indicating that

Olympiad Bench

Ablation Setting GSM8K MATH SVAMP ASDiv Minerva Math

AVG.

Qwen2.5-Math-1.5B 42.5 34.2 68.8 68.1 9.9 23.7 41.2 + Rejection SFT 69.2 57.1 64.1 70.2 15.1 27.6 50.6 + GT Supervision 43.4 34.8 70.4 69.7 10.2 19.8 41.4 + Full LightReasoner 70.6 59.3 76.0 79.8 11.4 27.1 54.0

- ✗ Select ✓ Contrast 67.6 58.8 78.7 80.5 11.0 26.4 53.8

✓ Select ✗ Contrast 62.0 53.1 56.6 61.0 10.7 25.5 44.8

- ✗ Select ✗ Contrast 55.5 50.2 50.0 65.4 10.4 24.0 42.6

- Table 6: Ablation study on the LightReasoner framework. We progressively remove key components, step selection and contrastive supervision, to isolate their contributions.

MATH

Performance Gain Full Method w/o Select w/o Contrast w/o Select + Contrast

| |
|---|

+25.1

| |
|---|

+24.6

| |
|---|

| |
|---|

+18.9

| |
|---|

+16.0

Minerva

GSM8K

+28.1

+1.5

+25.1

+1.1

+19.5

+0.8

+13.0

+0.5

+0.3 +1.4

+3.6

+1.8

+2.7

+12.6

+3.4

+12.8

Olympiad AVG.

- Figure 7: Impact of ablation. Removing key components from LightReasoner consistently degrades performance, emphasizing their critical roles.

many steps contribute noise rather than meaningful learning value. The step-selection filter addresses this by discarding trivial cases, enabling more targeted training.

- • Impact of Contrastive Supervision. Removing contrastive supervision reduces LightReasoner to fine-tuning the Expert on its own paths (filtered by step selection). Without the Amateur’s contrast to capture the Expert’s relative margin, average performance drops by 9.2%. This demonstrates the central role of contrastive supervision in amplifying the Expert’s strengths while steering it away from amateur-like tendencies.
- • Synergy Between Contrast and Selection. When both mechanisms are removed, average performance drops by a staggering 12.4%—even greater than the sum of the individual ablations (–9.2% and –0.2%). This superadditive decline reveals their mutual dependence: without step selection, the contrastive signal is diluted by trivial steps; without contrastive supervision, the highvalue steps cannot be transformed into effective

learning signals. Together, these mechanisms form a tightly coupled system, explaining the amplified benefits when used jointly.

• Insights from Competitive Approaches. To further validate our method, we compared LightReasoner against alternative strategies (Table 6). Fine-tuning on human-curated solutions (GT Supervision) yielded weak results, while SFT on correct self-generated trajectories provided clear gains but still lagged behind the ablation variant with contrastive supervision alone. These findings highlight a key principle: pretrained models learn most effectively from signals grounded in their own behavior. By selectively amplifying the Expert’s advantage on the most informative reasoning steps, LightReasoner achieves faster and more robust improvements than either humancurated supervision or rejection-sampled SFT.

#### 4 Conclusion

In this work, we introduced LightReasoner, a novel framework for advancing LLM reasoning by exploiting the behavioral divergence between expert and amateur models. Grounded in the insight that critical reasoning steps carry disproportionate learning value, LightReasoner integrates targeted step selection with contrastive supervision to amplify the expert’s strengths while minimizing resource demands and training complexity. Experiments across diverse benchmarks demonstrate that LightReasoner not only outperforms traditional SFT under equal settings, but also delivers orderof-magnitude efficiency and eliminates reliance on labeled ground truth. These results highlight LightReasoner as a practical efficient solution for building more capable reasoning models.

#### Limitations

While LightReasoner demonstrates strong performance, several limitations remain:

- (1) Scope of evaluation. Our experiments focus primarily on mathematical reasoning benchmarks (e.g., GSM8K, MATH, Minerva Math), leaving the generality of the framework to other domains, such as code reasoning, for future exploration.
- (2) Pairing strategy. The Expert–Amateur contrast relies on model pairs with a balanced capability gap; developing more adaptive or data-driven pairing strategies could yield stronger and more stable supervision signals.
- (3) Hyperparameter sensitivity. Although the proposed step-selection and contrastive supervision mechanisms effectively reduce resource consumption while maintaining competitive performance, they introduce additional hyperparameters (e.g., α-pruning, βfiltering) that may require careful tuning across different tasks.
- (4) Model scalability. Our experiments are conducted on small and mid-scale open-weight models; extending the approach to larger proprietary models would further demonstrate its scalability and practical applicability.

#### References

Haw-Shiuan Chang, Nanyun Peng, Mohit Bansal, Anil Ramakrishna, and Tagyoung Chung. 2024. Explaining and improving contrastive decoding by extrapolating the probabilities of a huge and hypothetical lm. arXiv preprint arXiv:2411.01610.

Michelene TH Chi, Paul J Feltovich, and Robert Glaser. 1981. Categorization and representation of physics problems by experts and novices. Cognitive science, 5(2):121–152.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and

Ning Ding. 2025. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617.

K Anders Ericsson and Jacqui Smith. 1991. Toward a general theory of expertise: Prospects and limits. Cambridge University Press.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, and 1 others. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and 1 others. 2026. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802.

Ke Ji, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang, Junying Chen, Benyou Wang, and 1 others. 2025. The first few tokens are all you need: An efficient and effective unsupervised prefix fine-tuning method for reasoning models. arXiv preprint arXiv:2503.02875.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo

Gutman-Solo, and 1 others. 2022. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2022. Contrastive decoding: Open-ended text generation as optimization. arXiv preprint arXiv:2210.15097.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and 1 others. 2024a. Rho-1: Not all tokens are what you need. arXiv preprint arXiv:2404.07965.

Zicheng Lin, Tian Liang, Jiahao Xu, Qiuzhi Lin, Xing Wang, Ruilin Luo, Chufan Shi, Siheng Li, Yujiu Yang, and Zhaopeng Tu. 2024b. Critical tokens matter: Token-level contrastive estimation enhances llm’s reasoning capability. arXiv preprint arXiv:2411.19943.

Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. 2026. Sparse but critical: A token-level analysis of distributional shifts in rlvr finetuning of llms. arXiv preprint arXiv:2603.22446.

Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2021. A diverse corpus for evaluating and developing english math word problem solvers. arXiv preprint arXiv:2106.15772.

Sean O’Brien and Mike Lewis. 2023. Contrastive decoding improves reasoning in large language models. arXiv preprint arXiv:2309.09117.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal.

- 2021. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191.

Phuc Phan, Hieu Tran, and Long Phan. 2024. Distillation contrastive decoding: Improving llms reasoning with contrastive decoding and distillation. arXiv preprint arXiv:2402.14874.

Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. 2025. Maximizing confidence alone improves reasoning. arXiv preprint arXiv:2505.22660.

Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. 2026. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897.

Charlie Snell, Dan Klein, and Ruiqi Zhong. 2022. Learning by distilling context. arXiv preprint arXiv:2209.15189.

Yiliu Sun, Zicheng Zhao, Yang Wei, Yanfang Zhang, and Chen Gong. 2026. Well begun, half done: Reinforcement learning with prefix optimization for llm reasoning. In Proceedings of the AAAI Conference

on Artificial Intelligence, volume 40, pages 33144– 33152.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, and 1 others. 2025. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, and 1 others. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. 2026. Self-distilled rlvr. Preprint, arXiv:2604.03128.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. Preprint, arXiv:2308.01825.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv preprint arXiv:2601.18734.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. 2025. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590.

## Appendix

#### A Related Work

##### A.1 Contrastive Decoding

Contrastive decoding (CD) has emerged as a novel technique for improving inference-time decoding in language models. CD introduces a smaller “amateur” model alongside a larger “expert” model from the same family, and contrasts their nexttoken distributions at each decoding step (Li et al.,

- 2022). Instead of directly sampling from the expert’s distribution, CD selects tokens based on the difference between the expert and amateur logprobabilities, effectively reweighting the expert’s predictions. The key intuition is that the amateur model serves as a proxy for undesirable behaviors (e.g., repetition or incoherence). By subtracting these tendencies, CD isolates the “net expertise” of the expert model, yielding a more reliable decoding signal. Empirically, this contrastive objective reduces hallucination and improves coherence in open-ended generation.

Subsequent work extended CD beyond openended generation to reasoning tasks, where it promotes less noisy token predictions, leading to improved reasoning performance (O’Brien and Lewis,

- 2023). To reduce the overhead of maintaining two separate models, Phan et al. (2024) propose constructing the amateur model by applying distillation techniques (e.g., dropout or quantization) to the expert, enabling a single model to play both roles. More recently, Chang et al. (2024) provide a theoretical perspective, interpreting the contrastive scores as approximations to an idealized target distribution.

Despite their promise, CD approaches are subject to several inherent limitations. First, CD incurs substantial computational overhead at inference time, as it requires evaluating both the expert and amateur models at every decoding step. This dualmodel dependency increases memory usage and introduces additional latency. Furthermore, CD relies on a sufficiently large capability gap between the expert and amateur models to produce meaningful contrast. In practice, this often necessitates pairing models with a significant size disparity within

the same model family, which imposes restrictive design constraints and limits the flexibility of CD.

LightReasoner builds upon the core intuition of contrastive decoding while introducing a fundamental paradigm shift. Instead of applying contrast at inference time, LightReasoner leverages expert–amateur divergence during training, distilling it into persistent supervision signals that eliminates concurrent model execution at inference. This removes the memory and latency overhead associated with CD. Moreover, by exploiting domain-specific capability gaps rather than relying solely on model size differences, LightReasoner relaxes the strict size disparity requirements of traditional CD setups. Altogether, LightReasoner extends the contrastive paradigm beyond decoding, enabling efficient and robust improvements in reasoning performance.

##### A.2 Training on Selective Tokens

Conventional training paradigms for language models uniformly optimize over all tokens in a trajectory, implicitly treating them as equally informative. Recent work has begun to challenge this convention, recognizing that different tokens carry varying learning value. In this view, uniformly training on all tokens may dilute learning signals by allocating substantial optimization effort to “low-return” tokens, while limiting the impact of more informative “high-return” tokens (Wang et al., 2025; Sun et al., 2026; Meng et al., 2026). Focusing on a subset of informative tokens can improve training efficiency by reducing redundant updates, while potentially enhancing robustness by filtering out noise and trivial patterns.

A central challenge, however, lies in identifying such informative tokens a priori. Existing approaches broadly fall into two categories. The first adopts simple, heuristic-based selection strategies that focus on prefix segments of reasoning trajectories, motivated by the observation that early-stage reasoning often shapes downstream outcomes. For example, UPFT (Ji et al., 2025) performs SFT exclusively on initial prefixes, while PPPO (Sun et al., 2026) restricts policy optimization to earlystage reasoning tokens. The second category employs adaptive, data-driven criteria to identify highvalue tokens. RHO-1 (Lin et al., 2024a) analyzes token-level training dynamics and trains a reference model to score token importance, selecting only tokens with higher learning utility for large-scale pretraining. Similarly, Wang et al. (2025) observe an 80/20 rule, where a small subset of high-entropy

tokens drives most of the learning signal; restricting updates to these tokens yields comparable or improved performance over full-trajectory training.

LightReasoner introduces a fundamentally different perspective on selective token training. Rather than relying on heuristic truncation or external scoring mechanisms, it leverages the domain-specific expertise gap between an expert and an amateur model within the same family. This capability gap induces systematic differences in their next-token distributions, where the resulting divergence signals distinctive reasoning behavior exhibited by the expert. These high-divergence tokens naturally correspond to steps where the expert demonstrates reasoning patterns absent in the amateur.

##### A.3 Self-Distillation

Knowledge distillation (Hinton et al., 2015) is a widely used technique for improving language model performance by transferring knowledge from a teacher model to a student model. It offers several practical advantages, including token-level supervision, algorithmic simplicity, and stable optimization. To reduce reliance on external teacher models, self-distillation has been proposed as an extension of this paradigm, where a model improves by learning from targets derived from its own generations. In this setting, the model first produces candidate outputs, which are then filtered or refined to construct supervision signals for further training. Rejection sampling fine-tuning (Yuan et al., 2023), widely adopted as post-training SFT, can be viewed

- as a form of self-distillation, where the model is trained to imitate higher-quality versions of its own generated trajectories.

Recently, self-distillation has gained renewed attention with the emergence of on-policy selfdistillation (OPSD)5. Building on the framework of Snell et al. (2022), several concurrent works explore this paradigm. SDFT (Shenfeld et al., 2026) and Self-Distilled Reasoner (Zhao et al., 2026) propose using a single model as both teacher and student by conditioning on different contexts, where the teacher has access to privileged information unavailable to the student. SDPO (Hübotter et al., 2026) further extends this framework by incorporating additional feedback signals, such as execution errors or evaluation outcomes, into the privileged context. Despite differences in implementation,

5OPSD emerged after the initial version of our work; we include it in this revised version for completeness and discuss its relation to our method.

these methods share a common principle: leveraging the model’s in-context learning ability to construct a stronger target from enriched inputs.

LightReasoner adopts a self-distillation perspective while taking a different approach from OPSDstyle methods. Instead of relying on auxiliary privileged information to induce a teacher–student asymmetry, LightReasoner leverages the native capability gap between an expert and an amateur model. Unlike OPSD, which follows a “strong-toweak” paradigm, LightReasoner does not treat the amateur as a teacher. Rather, the amateur serves as a reference signal that highlights the expert’s relative strengths. This avoids the need for artificially constructed signals, which can be brittle when the privileged information is noisy (Hübotter et al., 2026), and also prevents the train–inference mismatch observed in OPSD (Yang et al., 2026). In addition, LightReasoner introduces two key mechanisms into the self-distillation framework: selective training and a contrastive objective. The former improves efficiency by focusing updates on high-value tokens, while the latter reduces target noise and improves supervision quality. While LightReasoner introduces these novel components, it remains largely off-policy. Exploring on-policy extensions of LightReasoner is an important direction for future work.

#### B The Use of Large Language Models

We used OpenAI’s ChatGPT solely as a writing assistant to polish grammar, phrasing, and readability of the paper. All research content is entirely the work of the authors.

#### C From KL Divergence to Contrast Score

In §2.2, we analyzed Expert–Amateur disagreement via KL divergence and observed that reasoning steps where the Expert holds a clear advantage correspond to higher KL divergence values. A natural idea, therefore, is to train the Expert model to maximize this divergence, which is equivalent to minimizing the loss:

###### LKL(st) = −DKL(πE(· | st)∥πA(· | st)). (9)

Algorithm 1 LightReasoner: An efficient self-distillation framework for reasoning enhancement. Input: Expert model πE, Amateur model πA, dataset Q Output: Enhanced Expert model πE′

- 1: // Phase 1: Contrastive Sampling
- 2: for all problem q ∈ Q do
- 3: Generate a CoT trajectory using πE: {(st,πE(· | st))}Tt=1
- 4: for t = 1 to T do
- 5: Compute πA(· | st)
- 6: Compute DKL(πE(· | st) ∥ πA(· | st))
- 7: if DKL(πE(· | st) ∥ πA(· | st)) > β then
- 8: for all ai ∈ A do
- 9: if πE(ai | st) ≥ α · max a

πE(a | st) then

- 10: Add ai to Amask
- 11: Compute vC′ (ai | st) ← log ππE(ai|st)

A(ai|st)

- 12: Normalize: v˜C(· | st) ← softmax(vC′ (· | st)) over Amask
- 13: for all ai ∈ A do
- 14: if ai ∈ Amask then
- 15: vC(ai | st) ← v˜C(ai | st)
- 16: else
- 17: vC(ai | st) ← 0
- 18: Store (st,vC(· | st)) for training
- 19: // Phase 2: Contrastive Fine-tuning
- 20: for all stored (st,vC(· | st)) do
- 21: Compute current output πE(· | st)
- 22: Compute loss L(st) ← DKL vC(· | st) πE(· | st)
- 23: Update πE using gradient ∇L(st) to obtain the fine-tuned model πE′

Treating πA as fixed, the gradient with respect to the Expert parameters θE is:

∇θE LKL(st) (10)

=∇θE − DKL πE(· | st)∥πA(· | st)

πE(a | st) πA(a | st)

πE(a | st) log

= − ∇θE

a∈A

∇θE πE(a | st) log ππE(a|st)

= −

A(a|st) ,

a∈A

By product rule:

∇θE LKL(st)

log ππE(a|st)

A(a|st) ∇θEπE(a | st)

= −

a∈A

+ πE(a | st)∇θE log ππE(a|st)

A(a|st) , (11)

Note that by chain rule:

πE(a | st) πA(a | st)

∇θE log

=∇θE log πE(a | st) − log πA(a | st)

=∇θE log πE(a | st)

=∇θEπE(a | st) πE(a | st)

, (12) Substituting (12) into (11) yields:

∇θE LKL(st) (13)

log ππE(a|st)

A(a|st) + 1 ∇θEπE(a | st). At this point, the log-ratio naturally emerges:

= −

a∈A

πE(a | st) πA(a | st)

vC′ (a | st) = log

(14)

###### = log πE(a | st) − log πA(a | st),

which coincides exactly with the contrast score proposed in Li et al. (2022). This observation motivates our choice: rather than directly maximizing

KL divergence, we adopt the contrast score as the central mechanism for constructing the supervision signal to guide the Expert model.

#### D Connection between Selection, Contrast, and Training

As shown in §C, a central quantity in our framework is the contrast score (Li et al., 2022):

vC′ (a | st) = log πE(a | st) (15) −log πA(a | st),

This log-ratio underlies step selection, contrastive supervision, and the training objective, unifying the three components into a mutually reinforcing and coherent framework.

Step Selection. Expert–Amateur disagreement is quantified by the KL divergence:

DKL(πE(· | st)∥πA(· | st)) (16)

πE(a | st)vC′ (a | st).

=

a∈A

=Ea∼πE(·|st)[vC′ (a | st)]

Contrastive Distributional Supervision. Restricting to a masked vocabulary Amask, we normalize vC′ (a | st),a ∈ Amask via:

exp(vC′ (a | st)) b∈Amask exp(v′

, (17)

v˜C(a | st) =

C(b | st))

and extend to the full vocabulary a ∈ A by:

vC(a | st) = v˜C(a | st) · [a ∈ A mask], (18) Training Objective. The Expert is distilled toward vC(a | st) by minimizing forward KL:

L(st) =DKL(vC(· | st)∥πE(· | st)) (19)

vC(a | st) πE(a | st)

vC(a | st) log

=

.

a∈A

Unification. All three components are governed by the same log-ratio vC′ (a | st):

- 1. Step selection quantifies divergence through

expected value of vC′ (a | st) and retains informative reasoning steps.

- 2. Contrastive supervision transforms vC′ (a | st) into a learning signal, capturing the Expert model’s difference margin from the Amateur.
- 3. Training propagates gradients implicitly

weighted by vC′ (a | st), guiding toward high-contrast tokens.

Intuitive explanation. The expert and amateur models are drawn from the same model family (e.g., the expert is Qwen2.5-Math while the amateur is Qwen2.5), sharing identical architectures and pretraining. The expert can be viewed as the amateur model after additional domain-specific posttraining. As a result, the two models exhibit similar syntactic and low-level statistical structure, and their divergence primarily reflects differences in domain expertise. In particular, this divergence tends to concentrate on directions where the expert demonstrates stronger reasoning ability shaped by its additional training.

From this perspective, the difference between the two policies can be interpreted as a noisy estimate of a performance improvement direction. It highlights regions where the expert assigns higher probability than the amateur, which often correspond to more accurate reasoning steps. The noise arises because not all differences reflect genuine improvements, but it is largely unstructured and tends to average out.

The KL divergence DKL(πE(· | s)∥πA(· | s)) provides a natural mechanism to aggregate this signal. Its gradient emphasizes directions where the expert deviates from the amateur, and serves as a good estimation of the performance improvement direction. Consequently, maximizing this divergence encourages updates that further reinforce regions where πE outperforms πA, while placing less emphasis on regions where the two models agree or where the signal is weak.

Although individual differences may be noisy, repeated sampling combined with filtering acts as a form of variance reduction. In expectation, this process nudges updates toward consistent advantageous directions. Intuitively, the method reinforces regions where the expert is reliably stronger, rather than indiscriminately shifting the entire distribution, thereby helping to avoid pathological behavior collapse.

#### E Relation to Reinforcement Learning

Policy gradient methods. In reinforcement learning (RL), the objective of policy gradient methods is to maximize the expected return

J(θ) = Eπθ

T−1

γt rt , (20)

t=0

where rt is the reward at step t and γ ∈ [0,1] is a discount factor. Under the actor–critic framework,

the policy gradient theorem states

∇θJ(θ)

= Eπθ

T−1

∇θ log πθ(at | st)Aθ(at,st) (21)

t=0

where Aθ(at,st) denotes the advantage of action

- at at state st.

Parallel with our framework. In our setting, the training objective is the KL divergence:

L(st) =DKL vC(· | st)∥πE(· | st)

vC(a | st) πE(a | st)

vC(a | st) log

=

a∈A

vC(a | st) log vC(a | st)

=

a∈A

vC(a | st) log πE(a | st). (22)

−

a∈A

The first term in (22) is constant with respect to πE. Therefore, the gradient of the loss with respect to πE reduces to:

∇θEL(st)

vC(a | st)∇θE log πE(a | st). (23)

= −

a∈A

This update has the same structure as a policy gradient step. In particular, it can be written as:

∇θEL(st) (24)

ALightReasoner(a,st)∇θE log πE(a | st),

= −

a∈A

where ALightReasoner(a,st) = vC(a | st).

Inner connection. In standard RL, this weight corresponds to the advantage Aπθ(st,at). In our framework, the contrastive target vC(a | st) plays an analogous role as an advantage signal. The key distinction lies in the optimization direction: policy gradient methods perform gradient ascent to maximize J(θ), whereas we perform gradient descent to minimize L(st).

Moreover, the contrastive score vC(a | st) is a masked and renormalized version of the log-ratio vC′ (a | st) defined in (15). As a result, the training objective is closely aligned with the heuristic of maximizing the KL divergence between the expert and amateur models in (9).

This perspective suggests that our framework can be viewed as a variant of policy gradient training, where the advantage signal arises from Expert– Amateur divergence rather than from environment rewards or human feedback.

Recent work on on-policy self-distillation (OPSD) (Shenfeld et al., 2026; Hübotter et al., 2026) similarly explores deriving advantage-like signals from the model itself, for instance through log-ratio-based comparisons under different conditioning contexts. Exploring how LightReasoner can be extended into a fully on-policy framework with explicit policy updates and reward signals remains an interesting direction for future work.

#### F Entropy Dynamics

Policy updates instantiate an exploration– exploitation trade-off. Recent work shows that this trade-off is closely reflected in policy entropy (Cui et al., 2025): without explicit control, entropy often collapses early, leading to overconfident policies and premature performance saturation. Empirically, pretrained models with higher initial entropy tend to achieve better downstream performance after RL, while the RL process itself largely converts entropy into reward (Yue et al., 2025). This makes entropy preservation an important consideration in LLM post-training.

Policy entropy. Formally, the entropy of a policy π relative to a dataset D is

H(π,D) (25)

= −Ey<t∼D

yt

π(yt | y<t)log π(yt | y<t) .

Entropy change under policy gradient. For a single state s, the intrinsic policy entropy is

H(π | s) := −

a

π(a | s) log π(a | s). (26)

For a tabular softmax policy updated by one step of vanilla policy gradient with step size η, the stepwise entropy change satisfies the first-order approximation

H(πk+1| s) − H(πk | s) ≈ −η (27) Cova∼πk(·|s) log πk(a | s), πk(a | s)Ak(s,a) ,

where Ak(s,a) is the advantage, satisfying Ea∼πk[Ak(s,a)] = 0. Thus, when highprobability actions tend to carry positive advantage,

the covariance is positive and entropy decreases; when advantage is concentrated on low-probability actions, the covariance can become negative and entropy can increase (Cui et al., 2025).

Entropy change with contrast score. As established in §E, our framework has the same update structure as policy gradient when the contrast score is interpreted as the advantage signal. Consequently, the first-order change in policy entropy can be written as

H(πEk+1| s) − H(πEk | s) ≈ −η (28) Cova∼πk

E(·|s) log πEk (a | s), πEk (a | s)vCk (s,a) .

Let Amask(s) ⊆ A denote the masked action set, and let [a ∈ A mask(s)] be its indicator function. Define the masked, temperature-scaled contrast as

1 τ

[a ∈ A mask(s)]vC′k(s,a),

vCk (s,a) :=

(29)

πEk (a | s) πAk (a | s)

vC′k(s,a) := log

,

where τ > 0 is a temperature scaling parameter. Substituting (29) into (28) yields

H(πEk+1| s) − H(πEk | s) ≈ (30) −

η τ

E(·|s) log πEk (a | s), πEk (a | s) [a ∈ A mask(s)] log

Cova∼πk

πEk (a | s) πAk (a | s)

.

Intuitively, (30) shows that the entropy change is driven by terms of the form

πEk (a | s) πAk (a | s)

πEk (a | s) [a ∈ A mask(s)] log

. (31)

The α-mask restricts updates to actions with nonnegligible probability under the Expert. If the Expert and Amateur assign similar probabilities to an action, then the log-ratio is close to zero, and the corresponding contribution to entropy change is negligible even when πE is large. In contrast, when the Expert places substantial probability mass on actions for which it strongly outperforms the Amateur, the log-ratio becomes large and positive, making the covariance term positive and causing entropy to decrease more substantially. Therefore, the contrastive signal spends entropy selectively on high-value, high-contrast actions rather than on regions where the two models already agree. This

makes entropy reduction more targeted, helping avoid premature entropy collapse while preserving the Expert’s exploration capacity in low-signal regions.

#### G Additional Experiments

- G.1 Effect of Truncation Length

In the LightReasoner framework, supervision examples are constructed from short sampling rollouts (e.g., the first 128 tokens of the model’s reasoning trajectory). Importantly, generation is paused once the maximum token limit is reached, rather than forcing the model to compress full solutions into shorter forms. Each supervision example is constructed at the next-token level, which avoids introducing length-dependent biases or fragmented reasoning patterns.

To verify that the model is not biased toward shallow reasoning, we include qualitative case studies in Appendix I, showing that LightReasoner-trained models produce full, coherent CoT solutions.

We further conduct a controlled experiment by varying the sampling rollout length (i.e., the number of generated tokens before pausing) during supervision construction, as shown in Table 7. Across benchmarks, performance does not follow a monotonic trend with respect to the rollout length, with different tasks favoring different prefix lengths. This suggests that LightReasoner is not sensitive to the exact sampling rollout length, and that performance gains primarily stem from the contrastive signal derived from informative steps rather than from length-dependent factors.

- G.2 Generalization Beyond Math

To evaluate whether our method transfers beyond the math domain, we evaluate the base model Qwen2.5-Math-1.5B and its LightReasoner-trained counterpart on three standard commonsense reasoning benchmarks. The results are shown in Table 8.

Although training is conducted solely on supervision samples derived from GSM8K, a math dataset, LightReasoner consistently improves performance across all three commonsense benchmarks. This suggests that our method captures and enhances fundamental reasoning capabilities that generalize beyond the training domain.

#### H Supplementary Details

We provide additional details to complement the descriptions in §3.1.1 and §3.1.2, covering both our

###### Sampling Length GSM8K MATH SVAMP ASDiv Minerva Olympiad

64 74.3 61.5 81.5 81.4 18.8 24.6 128 70.6 59.3 76.0 79.8 11.4 27.1 256 72.3 62.2 78.5 79.4 18.4 21.3

- Table 7: Performance under different sampling rollout lengths. Results do not exhibit a monotonic trend, suggesting that LightReasoner is largely insensitive to this parameter.

Model CommonsenseQA HellaSwag ARC-Challenge

Baseline 62.6 46.2 42.8 + LightReasoner 64.2 47.4 45.6

- Table 8: Generalization performance on non-math reasoning benchmarks. Despite training solely on GSM8K, LightReasoner demonstrates effective transfer beyond the training domain.

proposed method, LightReasoner, and the competitive supervised fine-tuning (SFT) baseline.

##### H.1 Datasets

We exclusively use the GSM8K (Cobbe et al., 2021) training set, a collection of grade-school math problems emphasizing step-by-step reasoning, to generate contrastive samples. To evaluate the transferability of the learned skills, we assess our models on a diverse suite of benchmarks: MATH (Hendrycks et al., 2021), a collection of high school competition problems; SVAMP (Patel

- et al., 2021) and ASDiv (Miao et al., 2021), testing numerical reasoning through linguistically varied arithmetic problems; Minerva Math (Lewkowycz
- et al., 2022), quantitative problems from advanced STEM courses; OlympiadBench (He et al., 2024), challenging problems from international math olympiads; and MMLU-STEM (Hendrycks et al., 2020), which evaluates broad knowledge across science and math. This range spans from foundational arithmetic to expert-level reasoning, enabling a thorough assessment of both generalization and specialization.

##### H.2 Baseline Models

Our method leverages pairing between Expert and Amateur models to generate training signals:

• Expert Models span varying capabilities and sizes to ensure robust evaluation: (1) Qwen2.5Math-1.5B and (2) Qwen2.5-Math-7B are derived from Qwen2.5 base models via pretraining on a 1T-token math corpus (Yang et al., 2024). (3) Qwen2.5-Math-1.5B-Instruct and (4) Qwen2.5Math-7B-Instruct receive additional multi-stage post-training, including supervised fine-tuning and GRPO-based reinforcement learning (Yang

et al., 2024). (5) DeepSeek-R1-Distill-Qwen1.5B is fine-tuned based on the corresponding Qwen2.5-Math model, using teacher-curated examples generated by DeepSeek-R1 (Guo et al., 2025).

• Amateur Model is fixed as Qwen2.5-0.5B, a base model without specialized mathematical pretraining but with general linguistic reasoning ability (Yang et al., 2024). Being in the same model family as the Experts ensures that performance differences reflect mathematical expertise rather than architectural discrepancies.

##### H.3 LightReasoner

For LightReasoner, hyperparameters were selected to ensure training efficiency and final performance. Below we report further details of the experimental setup.

H.3.1 α-masking Following Li et al. (2022), we apply α-masking for a ∈ A to truncate the Expert distribution:

Amask = πE(a | st) ≥ α · max b∈A

πE(b | st) .

(32)

This operation trades off quality and diversity: larger α keeps only top-probability tokens, yielding higher quality but reduced coverage, while smaller α admits more diverse candidates at the cost of reliability. In our framework, the Expert and Amateur share the same vocabulary and tokenizer, and α ∈ [0,1] controls how aggressively the Expert’s next-token distribution is truncated. Tokens below the threshold are excluded, and contrastive supervision is applied only over the surviving set.

We set α = 0.2 throughout, deviating from the α = 0.1 commonly used in Li et al. (2022) and O’Brien and Lewis (2023). This choice was motivated by manual inspection across five different models: we compared α ∈ {0.1,0.2,0.4} on a variety of examples and found that α = 0.2 strikes the best balance. It preserves high-quality candidates while avoiding distributional collapse into a near one-hot target, which can occur under more aggressive α-truncation.

- H.3.2 β-filtering As discussed in §2.2 and §2.3.1, not all tokens in a reasoning trajectory are equally informative. We proposed β-filtering to address this, retaining only those steps where the Expert–Amateur divergence exceeds a threshold:

DKL πE(· | st)∥πA(· | st) > β. (33)

Here, we provide additional insight into the mechanism of β-filtering. Empirically, we observe two types of contrastive supervision signals:

- • Single-token labels: When the Expert’s distribution is sharply peaked, α-masking retains only the top-1 token, yielding a degenerate label of [1.0].
- • Multi-token labels: When the Expert’s distribution is more spread out, multiple tokens survive α-masking, producing a distributed label.

In practice, β effectively regulates the balance between these two cases. Single-token labels almost always arise from low-KL steps: a highly peaked Expert distribution indicates strong confidence, which typically occurs on easier steps where Expert and Amateur agree, resulting in low divergence. Such cases contribute little useful contrast.

Our ablation confirms this effect: removing β increases the proportion of single-token labels by +35.6% for Qwen2.5-Math-1.5B, +33.5% for Qwen2.5-Math-7B, and +27.4% for Qwen2.5Math-1.5B-Instruct. β-filtering governs the proportion of contrast-rich samples and ensures that fine-tuning focuses on genuine Expert–Amateur disagreements. We fixed β = 0.4 throughout, chosen via manual inspection of the collected examples, and leave the exploration of potentially more optimal choices to future work.

- H.3.3 Prompting Recall the two central stages of LightReasoner:

- • Sampling. For a given problem q, we prompt the

Expert πE to generate a reasoning trajectory and record the sequence of prefix–distribution pairs:

TE = st, πE(· | st) Tt=1. (34)

For each prefix st in (34), we query the Amateur πA with the same input (q,st) to obtain πA(· | st) and construct the contrastive target vC(· | st) from the pair πE(· | st), πA(· | st) .

- • Fine-tuning. For each supervision example (st, vC(· | st)), we re-query the current Expert to obtain πE(· | st) and minimize the forward KL divergence:

###### L(st) = DKL(vC(· | st)∥πE(· | st)) (35)

Throughout both stages we attach a fixed Chain-ofThought (CoT) cue,

Please reason step by step, and put your final answer within \boxed{}. (36)

to every query. Using a single, shared prompt template for all calls (Expert/Amateur during sampling and Expert during training) is critical: it ensures that (i) Expert–Amateur differences arise from the models rather than prompt mismatch, and (ii) the context used to form vC(· | st) is exactly the context against which the training KL is evaluated. This prompt consistency eliminates confounding from input formatting and yields stable, comparable distributions πE(· | st) and πA(· | st) across the pipeline.

H.3.4 LoRA Configuration For LightReasoner fine-tuning, we adopt a LoRA setup (Hu et al., 2022) for efficiency while maintaining stability. The configuration is summarized in Table 9. Training was performed in bfloat16 precision on a single NVIDIA H200 GPU, with the following runtime hyperparameters: batch size of 8 with gradient accumulation of 2 (effective batch size 16), learning rate 5 × 10−5, and 1000 total update steps. The same configuration was applied across all five backbone models studied in this paper to ensure comparability, while avoiding model-specific hyperparameter tuning. The choice of 1000 steps is further justified by the perplexity curves in Figure 8, which show convergence within this horizon for multiple representative models.

##### LoRA Parameter Value

Rank (r) 8 Scaling factor (α) 16 Target modules q_proj, v_proj Dropout 0.05 Bias None Task type Causal LM

- Table 9: LoRA configuration used in LightReasoner, shared across all models.

LightReasoner: Perplexity over Fine-tuning Steps

Qwen2.5-Math-1.5B

1.0

Qwen2.5-Math-7B

Qwen2.5-Math-1.5B-Ins

0.8

NormalizedPPL

0.6

0.4

0.2

0.0

100 200 300 400 500 600 700 800 900 1000

Fine-tuning Steps

Figure 8: Perplexity convergence. PPL curves show training stabilizes around 1000 steps, supporting our choice of tuning horizon.

##### H.4 Supervised Fine-tuning (SFT)

We provide additional details on the SFT configuration, which serves as the competitive baseline against our method LightReasoner.

##### H.4.1 Rejection Sampling

Recent works (Yang et al., 2024; Guo et al., 2025) commonly employ rejection sampling (Yuan et al.,

- 2023) in SFT, where models are aligned with demonstrations of correct problem-solving trajectories. This involves generating multiple reasoning traces and retaining only those whose final answers match the ground truth.

In our setting, we adopt a simplified form. For each GSM8K training problem, the base model produces a single reasoning trajectory under the CoT prompt shown in (36), ensuring fairness and comparability with the LightReasoner pipeline. We then filter by checking whether the final answer matches the ground truth, retaining only correct trajectories. Depending on the capability of the base model, this yields between 4000 and 7000 problems (see Table 2 in §3.3). The resulting set of verified trajectories is used to fine-tune each model.

##### H.4.2 Model Fine-tuning

In canonical SFT, each training instance consists of a complete reasoning trajectory generated by the base model that concludes with the correct final answer. For each training instance, the model is reprompted with the fixed CoT prompt shown in (36) along with the problem statement, and is trained under teacher forcing to predict the next token along the gold trajectory until completion. The training objective is the standard cross-entropy loss, computed over the gold trajectory tokens against the model’s predicted distributions. For comparability with LightReasoner, we conducted SFT fine-tuning using the same LoRA configuration, as detailed in Table 10.

##### LoRA Parameter Value

Rank (r) 8 Scaling factor (α) 16 Target modules q_proj, v_proj Dropout 0.05 Bias None Task type Causal LM

Table 10: LoRA adapter configuration used for SFT, applied across all models.

SFT Baseline: Cross-Entropy Loss over Fine-tuning Steps

Qwen2.5-Math-1.5B

0.12

Qwen2.5-Math-7B

Qwen2.5-Math-1.5B-Ins

0.10

TrainingLoss

0.08

0.06

0.04

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

0.02

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

0 100 200 300 400

Fine-tuning Steps

Figure 9: SFT training loss. Curve lengths vary with the number of correct demonstrations, but all runs reach convergence.

Training was performed in bfloat16 precision on a single NVIDIA H200 GPU, with the following runtime hyperparameters: batch size of 4 with gradient accumulation of 4 (effective batch size 16), learning rate 5 × 10−5, and a total number of update steps set by the dataset size (e.g., 4K samples correspond to 250 steps). The same configuration was applied to all five backbone models to ensure fair comparison and avoid model-specific tuning. As shown in Figure 9, the training curves vary in length because each model yields a different number of correct demonstrations, but all runs are

trained to convergence.

#### I Case Study

Here, we present case studies illustrating how LightReasoner enhances the base model’s reasoning. Specifically, we show GSM8K test problems where Qwen2.5-Math-1.5B fails but its LightReasoner-enhanced counterpart succeeds. Both are evaluated under the zero-shot pass@1 setting:

- • Base model. We adopt the prompt in (37):

"Question: {input} Answer: ", "{output}"

(37)

This choice is motivated by two factors. First, it yields the best base-model accuracy (42.5% on GSM8K with (37) vs. 38.4% with (36)). Second, under (36), the base model often degenerates into repeated fragments of the question and partial solutions until the output limit, producing corrupted traces that obscure rather than clarify its reasoning process.

- • Enhanced model. For the LightReasonerenhanced model, we use the prompt in (36), under which it achieves its best accuracy and demonstrates more systematic CoT reasoning.

We emphasize that the use of different prompts is not a bias but a fairness criterion: each model is paired with the prompt under which it performs best. For example, base models in the Qwen2.5 family cannot effectively exploit CoT prompts such as (36), whereas instruct models require them to achieve their full potential. For instance, on the GSM8K test problems, Qwen2.5-Math-1.5B achieves 42.5% with (37) compared to 38.4% with (36), whereas Qwen2.5-Math-1.5B-Instruct obtains 84.8% with (36) but only 56.5% with (37).

Accordingly, all performance numbers in this paper are reported under each model’s best-fitting prompt, in line with the common practice of evaluating models under conditions that best reflect their intended usage, thereby ensuring that comparisons capture true model capability.

- Case Study 1 ? Question.

Toulouse has twice as many sheep as Charleston . Charleston has 4 times as many sheep as S e a t t l e . How many sheep do Toulouse , Charleston , and S e a t t l e have toge ther i f S e a t t l e has 20 sheep ?

★ Ground-truth Answer. 260

✗ Base Model Output (Incorrect).

200\ n \ nExplanation : I f S e a t t l e has 20 sheep , then Charleston has 4 times as many , which i s 80 sheep . Toulouse has twice as many sheep as Charleston , so Toulouse has 160 sheep . Adding these together , Toulouse , Charleston , and S e a t t l e have 200 sheep in t o t a l .

✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure our c a l c u l a t i o n s are accurate . \ n \ n1 . We know t h a t S e a t t l e has 20 sheep .

- \ n2 . Charleston has 4 times as many sheep as S e a t t l e . So , Charleston has \ \ ( 4 \ \ times 20 = 8 0 \ \ ) sheep .
- \ n3 . Toulouse has twice as many sheep as Charleston . So , Toulouse has \ \ ( 2 \ \ times 80 = 1 6 0 \ \ ) sheep .
- \ n4 . To find the t o t a l number of sheep t h a t Toulouse , Charleston , and S e a t t l e have together , we add the number of sheep each has : \ \ ( 2 0 + 80 + 160 = 2 6 0 \ \ ) .

- Case Study 2

? Question.

In a dance c l a s s of 20 students , 20% en rolle d in contemporary dance , 25% of the remaining enrolled in jazz dance , and the r e s t enrolled in hip −hop dance . What percentage of the

- e n t i r e students enrolled in hip −hop dance ?

★ Ground-truth Answer. 60

✗ Base Model Output (Incorrect).

40% of the students enrolled in hip −hop dance . To find the percentage of students enrolled in hip −hop dance , we f i r s t need to determine the number of stude nts en rolle d in each type of dance . 20% of 20 students enr olled in contemporary dance , which i s 4 students . The remaining stude nts are 20 − 4 = 16 students . 25% of the remaining

16 students enrolled in jazz dance , which i s 4 student s . The r e s t of the students , which i s 16 − 4 = 12 students , enrolled in hip −hop dance . Therefore , 12 out of 20 students , or 60%, enrolled in hip −hop dance . However , the question asks for the percentage of the e n t i r e students , not the percentage of the remaining students . Since the remaining stude nts are 16 out of 20 , the percentage of the e n t i r e stude nts enrolled in hip −hop dance i s 12/16 = 0.75 or 75%.

##### ✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure the accuracy of our c a l c u l a t i o n s . \ n \ n1 . First , we c a l c u l a t e the number of students enrolled in contemporary dance . Since 20% of the 20 students enrolle d in contemporary dance , t h i s i s \ ( 0 . 2 0 \ times 20 = 4 \ ) studen ts . \ n2 . Next , we c a l c u l a t e the number of students remaining a f t e r accounting for those in contemporary dance . This i s \(20 − 4 = 16\) stude nts . \ n3 . Then , we c a l c u l a t e the number of stu dents enrolled in jazz dance . Since 25% of the remaining 16 stude nts enrolled in jazz dance , t h i s i s \ ( 0 . 2 5 \ times 16 = 4 \ ) students . \ n4 . Finally , we c a l c u l a t e the number of students enrolled in hip −hop dance . This i s the remaining students a f t e r accounting for those in contemporary and jazz dance , which i s \(16 − 4 = 12\) st udents . \ n5 . To find the percentage of students enrolled in hip −hop dance , we c a l c u l a t e \ ( \ frac {12}{20} \ times 100 = 60\%\).

##### Case Study 3 ? Question.

Henry made two stops during his 60− mile bike t r i p . He f i r s t stopped a f t e r 20 miles . His second stop was 15 miles before the end of the t r i p . How many miles did he t r a v e l between his f i r s t and second stops ?

★ Ground-truth Answer. 25

✗ Base Model Output (Incorrect).

- 5 miles \ n \ nExplanation : Henry ' s f i r s t stop was a f t e r 20 miles , and his second stop was 15 miles before the end of the trip , which i s 45 miles . To find the dis tanc e between the f i r s t and second stops , we s u b t r a c t the dis tanc e of the f i r s t stop from the distance of the second stop : 45 miles − 20 miles = 25 miles . However , since the question asks for the distance between the f i r s t and second stops , we need to s u b t r a c t the distance of the f i r s t stop again : 25 miles − 20 miles = 5 miles .

##### ✓ LightR-tuned Model Output (Correct).

To determine how many miles Henry t r a v e l e d between his f i r s t and second stops , we need to follow these steps : \ n \ n1 . Determine the p o s i t i o n of the f i r s t stop . \ n2 . Determine the p o s i t i o n of the second stop . \ n3 . Calculate the di stan ce between the f i r s t and second stops . \ n \ nLet ' s break i t down : \ n \ n1 . Henry ' s f i r s t stop i s a f t e r 20 miles . \ n2 . His second stop i s 15 miles before the end of the trip , which means i t

- i s at \ \ ( 6 0 − 15 = 4 5 \ \ ) miles . \ n3 . The dist ance between the

- f i r s t and second stops i s \ \ ( 4 5 − 20 = 2 5 \ \ ) miles .

Case Study 4 ? Question.

In a room , there are various toys : 5 red cars , 3 action figures , and a doll . The doll cost as much as 3 action

- f i g u r e s and one red car cost $4 . How much are a l l toys worth , of an action f i g u r e costs $5?

##### ★ Ground-truth Answer. 50

✗ Base Model Output (Incorrect). The t o t a l value of a l l toys i s $33 . Here ' s the reasoning : \ n \ n1 . The doll costs as much as 3 action figures , and since each action f i g u r e costs $5 , the doll costs 3 * $5

= $15 . \ n2 . Each red car costs $4 , so 5 red cars cost 5 * $4 = $20 . \ n3 . The t o t a l value of the toys i s the sum of the values of the red cars , the action figures , and the doll : $20

( red cars ) + $15 ( doll ) + 3 * $5 ( action f i g u r e s ) = $33 .

##### ✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure our c a l c u l a t i o n s are accurate . \ n \ n1 . We know the cost of one red car i s $4 . \ n2 . We know the cost of one doll i s equal to the cost of 3 action f i g u r e s . \ n3 . We know the cost of one action f i g u r e i s $5 . \ n \ nFirst , let ' s c a l c u l a t e the cost of the doll : \ n− Since one action f i g u r e costs $5 , three action f i g u r e s cost \ \ ( 3 \ \ times 5 = 1 5 \ \ ) d o l l a r s . \ n− Therefore , the cost of one doll i s $15 . \ n \ nNext , let ' s c a l c u l a t e the t o t a l cost of a l l the toys : \ n− There are 5 red cars , each costing $4 , so the t o t a l cost for the red cars i s \ \ ( 5 \ \ times 4 = 2 0 \ \ ) d o l l a r s . \ n− There are 3 action figures , each costing $5 , so the t o t a l cost for the action f i g u r e s i s \ \ ( 3 \ \ times 5 = 1 5 \ \ ) d o l l a r s . \ n− There i s 1 doll , costing $15 . \ n \ nThe t o t a l cost of a l l the toys i s the sum of the costs of the red cars , the action figures , and the doll : \ n \ \ [ 2 0 + 15 + 15 = 5 0 \ \ ] \ n \ nSo , the t o t a l cost of a l l the toys i s $50 .

##### Case Study 5 ? Question.

Luke i s spending time at the beach building s a n d c a s t l e s . He eventually no t i c e s t h a t each l e v e l of a s a n d c a s t l e will have half the square footage as the l e v e l below i t . I f he makes a 4 leveled s a n d c a s t l e where the top l e v e l has a square footage of 16 , what i s the average square footage of a l e v e l ?

★ Ground-truth Answer. 60

✗ Base Model Output (Incorrect). To find the average square footage of a level , we f i r s t need to determine the square footage of each l e v e l . Since each l e v e l has half the square footage of the l e v e l below it , we can c a l c u l a t e the square footage of each l e v e l as follows : \ n \ n− Top l e v e l : 16 square f e e t \ n− Second l e v e l : 16 * 2 = 32 square f e e t \ n− Third l e v e l : 32 * 2 = 64 square f e e t \ n− Bottom l e v e l : 64 * 2 = 128 square f e e t \ n \nNow, to find the average square footage of a level , we add up the square footage of a l l l e v e l s and divide by the number of l e v e l s : \ n \ nAverage square footage

= (16 + 32 + 64 + 128) / 4 = 64 square f e e t \ n \ nTherefore , the average square footage of a l e v e l i s 64 square f e e t .

✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step . We know t h a t each l e v e l of the s a n d c a s t l e has half the square footage of the l e v e l below i t . The top l e v e l has a square footage of 16. Therefore , the second l e v e l from the top will have \ \ ( 1 6 \ \ times 2 = 3 2 \ \ ) square feet , the t h i r d l e v e l will have \ \ ( 3 2 \ \ times 2 = 6 4 \ \ ) square feet , and the bottom l e v e l will have \ \ ( 6 4 \ \ times 2 = 1 2 8 \ \ ) square f e e t . \ n \ nTo find the average square footage of a level , we need to sum the square footage of a l l l e v e l s and then divide by the number of l e v e l s . The t o t a l square footage i s \ \ ( 1 6 + 32 + 64 + 128 = 2 4 0 \ \ ) . Since there are 4 levels , the average square footage i s \ \ ( \ \ fra c {240}{4} = 6 0 \ \ ) .

##### Case Study 6 ? Question.

Two g i r l s each got 1/6 of the 24 l i t e r s of water . Then a boy got 6 l i t e r s of water . How many l i t e r s of water were l e f t ?

★ Ground-truth Answer. 10

✗ Base Model Output (Incorrect).

- 1 l i t e r \ n \ nExplanation : Each g i r l got 1/6 of the 24 l i t e r s of water , which i s 4 l i t e r s each . So , the t o t a l amount

of water given to the g i r l s i s 4 l i t e r s * 2 = 8 l i t e r s . The boy got 6 l i t e r s of water . Therefore , the t o t a l amount of water given away i s 8 l i t e r s + 6 l i t e r s =

- 14 l i t e r s . The remaining water i s 24 l i t e r s − 14 l i t e r s

= 1 l i t e r .

##### ✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure the accuracy of our c a l c u l a t i o n s . \ n \ n1 . First , c a l c u l a t e how much water each g i r l got . Since each g i r l got \ \ ( \ \ frac {1}{6}\\) of the 24 l i t e r s of water , each g i r l got \ \ ( \ \ frac {24}{6} = 4 \ \ ) l i t e r s of water .

- \ n2 . Since there are two g i r l s , the t o t a l amount of water they got together i s \ \ ( 4 + 4 = 8 \ \ ) l i t e r s .
- \ n3 . The boy then got 6 l i t e r s of water . \ n4 . The t o t a l amount of water used i s \ \ ( 8 + 6 = 14 \ \ ) l i t e r s . \ n5 . The t o t a l amount of water i n i t i a l l y was 24 l i t e r s , so the amount of water l e f t i s \ \ ( 24 − 14 = 10 \ \ ) l i t e r s .

- Case Study 7

? Question.

Gene i s sewing a q u i l t out of old souvenir t − s h i r t s . He has one s h i r t from each vacation he has been on . Every s h i r t i s

- i t s own q u i l t block . Each row i s made of blocks from a

- d i f f e r e n t year of vacations . He goes on four vacations a year and has been vacationing since he was 23 years old . He i s now 34. How many q u i l t blocks does he have in t o t a l ?

★ Ground-truth Answer. 44

✗ Base Model Output (Incorrect).

Gene has been on 12 vacations ( from age 23 to 34 , i n c l u s i v e ) . Since he goes on four vacations a year , he has 12 / 4 =

- 3 years of vacations . Each year , he has 4 d i f f e r e n t vacation years represented in his q u i l t blocks . Therefore , the t o t a l

number of q u i l t blocks i s 3 years * 4 vacation years / year

= 12 q u i l t blocks .

✓ LightR-tuned Model Output (Correct). To determine the t o t a l number of q u i l t blocks Gene has ,

we need to follow these steps : \ n \ n1 . Calculate the number of years Gene has been vacationing . \ n2 . Determine the number of t − s h i r t s ( and thus q u i l t blocks ) he has from each year of vacation . \ n3 . Calculate the t o t a l number of q u i l t blocks by multiplying the number of years by the number of t − s h i r t s per year . \ n \ nLet ' s break i t down : \ n \ n1 . Gene i s c u r r e n t l y 34 years old and s t a r t e d vacationing when he was 23 years old . Therefore , he has been vacationing for \ \ ( 3 4 − 23 =

- 1 1 \ \ ) years . \ n2 . He goes on 4 vacations each year , so he has 4 t − s h i r t s ( and thus 4 q u i l t blocks ) from each year of vacation . \ n3 . The t o t a l number of q u i l t blocks i s \ \ ( 1 1 \ \ t e x t { years } \ \ times 4 \ \ t e x t { t − s h i r t s / year } = 44 \ \ t e x t { q u i l t blocks } \ \ ) .

Case Study 8 ? Question.

Emily has 4 kids named Amy, Jackson , Corey , and James . Amy i s 5 years older than Jackson and 2 years younger than Corey . I f James i s 10 and i s 1 year younger than Corey , how old i s Jackson ?

★ Ground-truth Answer. 4

✗ Base Model Output (Incorrect).

Jackson i s 7 years old . To find this , we can s t a r t by determining the age of Corey . Since James i s 10 and 1 year younger than Corey , Corey i s 11 years old . Given t h a t Amy i s 2 years younger than Corey , Amy i s 9 years old . Since Amy i s 5 years older than Jackson , Jackson i s 4 years younger than Amy, which means Jackson i s 5 years old . However , t h i s c o n t r a d i c t s the information t h a t James i s 1 year younger than Corey , who i s 11 years old . Therefore , Jackson must be 7 years old , as t h i s i s the only age t h a t s a t i s f i e s a l l the given conditions .

✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure our c a l c u l a t i o n s are accurate . \ n \ n1 . We know t h a t James i s 10 years old . \ n2 . James i s 1 year younger than Corey , so Corey i s \ \ ( 1 0 + 1 = 1 1 \ \ ) years old . \ n3 . Amy i s

- 2 years younger than Corey , so Amy i s \ \ ( 1 1 − 2 = 9 \ \ ) years old . \ n4 . Amy i s 5 years older than Jackson , so Jackson i s \ \ ( 9 − 5 = 4 \ \ ) years old .

##### Case Study 9 ? Question.

In a neighborhood , the number of r a b b i t s pets i s twelve l e s s than the combined number of pet dogs and cats . I f there are two cats for every dog , and the number of dogs i s 60 , how many pets in t o t a l are in the neighborhood ?

★ Ground-truth Answer. 348

✗ Base Model Output (Incorrect). 180\ n \ nExplanation : Let ' s denote the number of cats as C and the number of dogs as D. According to the problem , we have two cats for every dog , so C = 2D. We are given t h a t D = 60 , so C = 2 * 60 = 120. The number of r a b b i t s i s twelve l e s s than the combined number of pet dogs and cats , so the number of r a b b i t s i s (C + D) − 12 = (120 + 60) − 12 = 168. Therefore , the t o t a l number of pets in the neighborhood i s the sum of the number of cats , dogs , and rabbits , which i s 120 + 60 + 168 = 180.

✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure our c a l c u l a t i o n s are accurate . \ n \ n1 . Let ' s denote the number of dogs as \ \ (D\ \ ) , the number of cats as \ \ ( C \ \ ) , and the number of r a b b i t s as \ \ ( R \ \ ) . \ n2 . According to the problem , the number of dogs \ \ (D\ \ ) i s 6 0 . \ n3 . The number of cats \ \ ( C \ \ ) i s two times the number of dogs , so \ \ ( C =

- 2 \ \ times D = 2 \ \ times 60 = 1 2 0 \ \ ) . \ n4 . The number of r a b b i t s \ \ ( R \ \ ) i s twelve l e s s than the combined number of pet dogsand cats , so \ \ ( R = D + C − 12 = 60 + 120 − 12

= 1 6 8 \ \ ) . \ n5 . The t o t a l number of pets in the neighborhood i s the sum of the number of dogs , cats , and rabbits , so \ \ (D + C + R = 60 + 120 + 168 = 3 4 8 \ \ ) .

##### Case Study 10 ? Question.

Cars have lined up on the motorway . Some of the cars drive through in the f i r s t 15 minutes of the t r a f f i c jam , then 20 more cars drive through in the remaining 15 minutes of the jam . 5 cars from the l i n e take an e x i t so they don ' t have to drive through the t r a f f i c jam . I f there were o r i g i n a l l y 30 cars on the motorway , how many cars drove through the t r a f f i c jam in the f i r s t 15 minutes ?

##### ★ Ground-truth Answer.

- 5

##### ✗ Base Model Output (Incorrect).

To find out how many cars drove through the t r a f f i c jam in the f i r s t 15 minutes , we can use the information given . I n i t i a l l y , there were 30 cars on the motorway . After

- 15 minutes , 20 more cars drove through , and 5 cars took an

- e x i t . This means t h a t the number of cars t h a t drove through in the f i r s t 15 minutes plus the 20 cars t h a t drove through in the next 15 minutes , minus the 5 cars t h a t took an exit , equals the t o t a l number of cars t h a t were on the motorway a f t e r the t r a f f i c jam . Let ' s denote the number of cars t h a t drove through in the f i r s t 15 minutes as \ \ ( x \ \ ) . Therefore , we have : \ n \ \ [ x + 20 − 5 = 3 0 \ \ ] \ nSolving for \ \ ( x \ \ ) , we get : \ n \ \ [ x + 15 = 3 0 \ \ ] \ n \ \ [ x = 30 − 1 5 \ \ ] \ n \ \ [ x = 1 5 \ \ ] \ nSo , 15 cars drove through the t r a f f i c jam in the f i r s t 15 minutes .

✓ LightR-tuned Model Output (Correct).

Let ' s break down the problem step by step and use Python to ensure our c a l c u l a t i o n s are accurate . \ n \ n1 . We s t a r t with 30 cars on the motorway . \ n2 . Let \ \ ( x \ \ ) be the number of cars t h a t drove through the t r a f f i c jam in the

- f i r s t 15 minutes . \ n3 . In the remaining 15 minutes , 20 more cars drove through the t r a f f i c jam . \ n4 . After the 30 minutes , 5 cars took an e x i t . \ n \ nThe t o t a l number of cars t h a t drove through the t r a f f i c jam i s \ \ ( x + 20 \ \ ) . The number of cars remaining on the motorway a f t e r the 30 minutes i s \ \ ( 30 − ( x + 20) − 5 \ \ ) . Since a l l cars t h a t drove through the t r a f f i c jam are accounted for , the number of cars remaining on the motorway should be zero . \ n \ nSo , we can s e t up the equation : \ n \ \ [ 30 − ( x + 20) − 5 = 0 \ \ ] \ n \ nSimplifying the equation : \ n \ \ [ 30 − x − 20 − 5

= 0 \ \ ] \ n \ \ [ 5 − x = 0 \ \ ] \ n \ \ [ x = 5 \ \ ] \ n \ nSo , the number of cars t h a t drove through the t r a f f i c jam in the f i r s t 15 minutes i s 5.

