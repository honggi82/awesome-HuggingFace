# arXiv:2510.00263v1[cs.CL]30Sep2025

[Figure 1]

## Judging with Confidence: Calibrating Autoraters to Preference Distributions

###### Zhuohang Li1,2, Xiaowei Li1, Chengyu Huang1,3, Guowang Li1, Katayoon Goshvadi4, Bo Dai4, Dale Schuurmans4,5, Paul Zhou1, Hamid Palangi1, Yiwen Song1, Palash Goyal1, Murat Kantarcioglu6, Bradley A. Malin2 and Yuan Xue7

1Google, 2Vanderbilt University, 3Cornell University, 4Google DeepMind, 5University of Alberta, 6Virginia Tech, 7Scale AI

The alignment of large language models (LLMs) with human values increasingly relies on using other LLMs as automated judges, or “autoraters”. However, their reliability is limited by a foundational issue: they are trained on discrete preference labels, forcing a single ground truth onto tasks that are often subjective, ambiguous, or nuanced. We argue that a reliable autorater must learn to model the full distribution of preferences defined by a target population. In this paper, we propose a general framework for calibrating probabilistic autoraters to any given preference distribution. We formalize the problem and present two learning methods tailored to different data conditions: 1) a direct supervised fine-tuning for dense, probabilistic labels, and 2) a reinforcement learning approach for sparse, binary labels. Our empirical results show that finetuning autoraters with a distribution-matching objective leads to verbalized probability predictions that are better aligned with the target preference distribution, with improved calibration and significantly lower positional bias, all while preserving performance on objective tasks.

### 1. Introduction

The alignment of large language models (LLMs) with human values (Ouyang et al., 2022) increasingly relies on using other powerful LLMs as automated judges, or “autoraters”, to score model responses. This LLM-as-a-Judge paradigm (Zheng et al., 2023) is now a cornerstone of evaluating and developing safer AI systems, particularly through methods such as reinforcement learning from AI feedback (RLAIF) (Bai et al., 2022).

Currently, autoraters are typically trained on discrete preference labels (Kim et al., 2024b; Li et al., 2024; Wang et al., 2024b), which leads to a fundamental limitation: (collective) human judgment does not correspond to a single label, but rather a distribution (Nie et al., 2020; Pavlick and Kwiatkowski, 2019), especially in complex situations that involve uncertainty or balancing multiple criteria (Arora et al., 2025). Even among qualified annotators, disagreement is common, not simply due to noise but because of systematic differences in how individuals define problems, interpret evidence, or apply values and decision strategies (Mumpower and Stewart, 1996). Current autoraters are trained with a mode-seeking objective that collapses this rich distributional information into a single verdict (e.g., the majority label), which discards crucial uncertainty signals and erases minority viewpoints by construction.

We argue that for an autorater to be reliable, it must be calibrated to model the full distribution of human preferences. An ideal judge should recognize when a topic is contentious (e.g., a 50/50 split), when a preference is clear but not unanimous (e.g., 80/20), and when a judgment is objectively certain. Modeling this distribution is essential for effective risk management, fairness, and building robust alignment systems.

This paper introduces a general and scalable framework for calibrating autoraters’ verbal probability predictions to any target preference distribution, while preserving their ability to generate

The work of ZL, CH, and YX was done at Google, when ZL and CH worked as student researchers. Correspondence to: zhuohang.li@vanderbilt.edu, yuan.xue@scale.com

[Figure 2]

Figure 1 | Overview of discrete vs. probabilistic autoraters. Left: Given a user query and two candidate responses, a discrete autorater returns a single preference (e.g., “B is better”), collapsing annotator variability. A probabilistic autorater predicts the full preference distribution and is finetuned via SFT/RL to match the target preference distribution. Right: Our finetuned autorater vs. zero-shot probabilistic conversions of discrete autoraters, including Verbalized Confidence (VC), Self-Consistency (SC), and Logits (Lo), evaluated using Gemma-2-9B on JudgeLM val set. Alignment error is measured by MSE, calibration error by ECE, and agreement by percentage.

natural-language reasoning traces. We introduce two finetuning methods, each tailored to a set of different data conditions. First, when dense, probabilistic labels from multiple annotators are available, we use direct supervised finetuning (SFT). Second, when only sparse, binary labels are available, we employ a reinforcement learning (RL) approach with rewards based on proper scoring rules. Our empirical results validate this distribution-matching objective. Autoraters finetuned with our methods show significant improvements in performance, calibration, and reliability. Notably, our methods achieve an 18-51% reduction in Mean Squared Error (MSE), a 4-45% reduction in Expected Calibration Error (ECE), and a 7-81% gain in consistency against positional bias. Notably, our findings offer guidance on annotation strategy: for a fixed budget, RL with many sparse, binary labels is more data-efficient than SFT with fewer dense, probabilistic labels, highlighting the benefits of prompt diversity. Our method also enhances alignment with human judgment on out-of-distribution tasks. On the PandaLM dataset, our finetuned Gemma-2-9B model achieves 73.17% agreement with human annotations, outperforming all baselines including GPT-4. Moreover, this improved calibration on subjective tasks does not compromise performance on objective ones, as the same model achieves an overall accuracy of 46.57% on JudgeBench, on par with Gemini-1.5-pro.

### 2. A Probabilistic Framework for Calibrating Autoraters

##### 2.1. Problem Formulation

We consider the scenario of pairwise judgements where an input 𝑿 specifies a prompt that is associated with two responses (𝐴, 𝐵). The population’s ground-truth preference is modeled as a Bernoulli random variable

𝑌 ∈ {0, 1}, 𝑌 | 𝑿 = 𝒙 ∼ Bernoulli(𝑝∗(𝒙)) ,

where 𝑌=1 indicates 𝐵 ≻ 𝐴 (i.e., 𝐵 is preferred to 𝐴), and 𝑝∗(𝒙) = Pr[𝑌=1 | 𝑿=𝒙] is the (unknown) preference distribution for the pair (𝐴, 𝐵). Let ℎ index a human annotator drawn from the population 𝑝(ℎ), then conceptually 𝑝∗(𝒙) represents the true population-level human preference 𝑝∗(𝒙) = Prℎ∼𝑝(ℎ)[𝐵 ≻ 𝐴|𝒙, ℎ], i.e., the probability that a randomly chosen annotator would prefer 𝐵 given the context 𝒙.

Discrete Autorater. An autorater is a language model (LM) prompted to act as a judge. In the discrete setting, the LM produces a single decision (e.g., via greedy decoding)

𝑑𝜽(𝒙) ∈ {0, 1},

or an uncalibrated scalar margin 𝑚𝜽(𝒙) ∈ ℝ (e.g., a parsed rubric rating or a logit difference), with decision {𝑚 𝜽(𝒙) ≥ 0}. Such outputs collapse the rater distribution at 𝒙 to a point estimate (typically the majority choice).

Probabilistic Autorater. A probabilistic judge instead predicts the full preference distribution through its Bernoulli parameter,

𝑝𝜽(𝒙) ∈ [0, 1] ≈ 𝑝∗(𝒙).

A definitive decision can be recovered by thresholding 𝑝𝜽(𝒙) if needed, but the primary output is the (conditional) probability itself, which is optimized to match the population preference rather than merely to choose a label. Crucially, this formulation does not rely on the Bradley-Terry assumption (Bradley and Terry, 1952) typically seen in reward modeling, thereby allowing richer representations of uncertainty.

##### 2.2. Benefits of Probabilistic Autoraters

Informativeness. In contrast to the mode-seeking behavior of discrete autoraters, probabilistic autoraters are optimized to match the full preference distribution. This probability prediction provides more information for cost-sensitive decision-making by revealing the aleatoric ambiguity within the task. Probabilistic reporting in autoraters also improves fairness and auditability by revealing annotator disagreement, whereas discrete reporting collapses the minority viewpoints.

Alignment. Probabilistic autoraters can be better aligned with the target preference distribution. In particular, it is straightforward to show that if Pr[0 < 𝑝∗(𝒙) < 1] > 0, then any discrete autorater or any single human annotator who effectively reports a degenerate distribution 𝑑(𝒙) ∈ {0, 1} is strictly worse than reporting 𝑝(𝒙) = 𝑝∗(𝒙) under a strictly proper scoring rule.

Calibration. Additionally, this distribution-matching objective of probabilistic autoraters implies calibration: if 𝑝𝜃(𝒙) = 𝑝∗(𝒙) almost surely, then 𝔼[𝑌 | 𝑝𝜃(𝒙) = 𝑐] = 𝑐 for all 𝑐 ∈ [0, 1]. In practice, as 𝑝𝜃 approaches 𝑝∗, calibration error (e.g., ECE) shrinks. By contrast, any probability prediction obtained from a discrete autorater post hoc (e.g., vote fractions from self-consistency or logits passed through softmax) is not trained to recover 𝑝∗(𝒙) and thus is generally not calibrated.

### 3. Finetuning Autoraters to Match the Preference Distribution

We introduce two distribution-matching finetuning paradigms for calibrating the autorater’s probabilistic prediction 𝑝𝜽 to the ground truth preference distribution 𝑝∗(𝒙). We focus on verbalized

probability because it is (1) more flexible and interpretable than training a dedicated classification head by preserving the model’s ability to generate natural-language rationales, and (2) more efficient than sampling-based approaches, as it only requires a single decoding pass.

- Setting 1: Direct Supervised Finetuning with Probabilistic Labels. When multiple annotations (𝒙, 𝑦(1), ..., 𝑦(𝑚)) are available for each prompt 𝒙 (the pair to be judged), we estimate the population preference by the multi-annotator mean ˆ𝑝(𝒙) = 𝑚1 𝑚𝑗=1 𝑦(𝑗) ≈ 𝑝∗(𝒙). We then instruction-tune the autorater in a text-to-text fashion: given prompt 𝒙 (the pair), the target sequence includes optional CoT reasoning and a structured numeric field encoding the probability that B is better. We apply standard autoregressive supervised finetuning (SFT) to maximize the likelihood of the target sequence 𝝉1:𝑆 composed using ˆ𝑝(𝒙):

LSFT = −𝔼(𝒙,ˆ𝑝)

∑︁ 𝑆

𝑡=1

log 𝑃𝜽(𝝉𝑡 | 𝒙, 𝝉1:𝑡−1) ,

where 𝝉𝑖 is the 𝑖th token of the sequence 𝝉 and 𝑆 is its sequence length. In practice, we parse the generated string to recover the numeric form of 𝑝𝜽(𝒙); training encourages

the model to produce 𝑝𝜽(𝒙) ≈ ˆ𝑝(𝒙) while retaining the ability to perform free-form reasoning.

- Setting 2: Reinforcement Learning from Binary Labels via Piecewise Proper Rewards. When only single-sample binary labels (𝒙𝑖, 𝑦𝑖) are available (e.g., via crowdsourced platforms), we treat the autorater as a sequence policy 𝝅𝜽(𝝉 | 𝒙) that produces a token sequence 𝝉 containing a numeric probability 𝑝. A deterministic parser 𝑔 maps 𝝉 to either a valid probability prediction in [0, 1] or ⊥ (unparsable), with the probability of producing a parsable response denoted as 𝑠𝜽(𝒙):

###### 𝑔 : T → [0, 1] ∪ {⊥}, 𝝉 ↦→ 𝑝 or ⊥, 𝑠𝜽(𝒙) = Pr

[ 𝑔(𝝉) ≠ ⊥].

𝝉∼𝝅𝜽(·|𝒙)

We use piecewise strictly proper scoring rules as rewards. Let 𝑦 ∈ {0, 1} (1 = B better), we have:

##### • Brier reward:

1 − (𝑝 − 𝑦)2, if 𝑔(𝝉) = 𝑝 ∈ [0, 1], 0, if 𝑔(𝝉) = ⊥.

𝑅Brier(𝝉; 𝑦) =

(1)

###### • Logarithmic reward (with clipping): For numerical stability, we consider a fixed small 𝜖 ∈ (0, 12) and define 𝑝′ = clip(𝑝, 𝜖, 1 − 𝜖). Then

𝑦 log 𝑝′ + (1 − 𝑦) log(1 − 𝑝′), if 𝑔(𝝉) = 𝑝 ∈ [0, 1], log𝜖, if 𝑔(𝝉) = ⊥.

𝑅Log(𝝉; 𝑦) =

(2)

For either reward 𝑅 ∈ {𝑅Brier, 𝑅Log}, the goal is to maximize the population objective

𝜽(·|𝒙)[ 𝑅(𝜏; 𝑦) ] . (3)

𝐽𝑅(𝜽) = 𝔼(𝒙,𝑦) 𝔼𝝉∼𝝅

Compared to the Brier reward, the Log reward heavily penalizes overconfident yet incorrect predictions. Both objectives can be optimized with policy-gradient-based reinforcement learning methods using the parsed numeric probability.

##### 3.1. Consistency Analysis

In Setting 1, the multi-annotator mean ˆ𝑝(𝒙) is an unbiased estimate of the true preference distribution 𝑝∗(𝒙) with variance decreasing as 1/𝑚 (Appendix B.1, Lemma 2), and thereby provides a high fidelity target for learning.

In Setting 2, at the population level, the optimal autorater policy under either the piecewise Brier reward or the clipped Log reward is (i) parsable everywhere (i.e., 𝑠𝜽(𝒙) = 1), (ii) deterministic in the numeric value it reports, and (iii) reports the truthful probability (i.e., recovers the target preference distribution, up to clipping for the Log reward). The proof is included in Appendix B.2.

Proposition 1 (Fisher Consistency of Brier and Log Rewards). Assume the policy class can realize, for each 𝒙, a deterministic numeric output 𝑝𝜽(𝒙) ∈ [0, 1] with 𝑠𝜽(𝒙) = 1. Then any global maximizer of 𝐽𝑅 in equation 3 satisfies:

- (a) Brier: 𝑝𝜽(𝒙) = 𝑝∗(𝒙) for almost all 𝒙.
- (b) Log (with clipping): 𝑝′

𝜽(𝒙) = clip 𝑝∗(𝒙), 𝜖, 1 − 𝜖 for almost all 𝒙.

Moreover, for both rewards, any stochasticity in the reported numeric value or any non-zero density associated with unparsable outputs strictly reduces the expected reward; thus, an optimizer is deterministic and fully parsable for almost every 𝑥.

### 4. Experimental Setup

Here, we describe the datasets used for calibration and evaluation, our scalable preference-annotation pipeline, the autorater output format and reward instantiation, and the finetuning setup.

Calibration Data. We build on a subset of prompts from the JudgeLM corpus (Zhu et al., 2025), which aggregates instruction-following tasks (e.g., Alpaca-GPT4 (Peng et al., 2023), Dolly-15K (Conover

- et al., 2023)) paired with responses from 11 open-source LLMs (including LLaMA (Touvron et al., 2023), Alpaca (Taori et al., 2023), Vicuna (Chiang et al., 2023)). The source corpus contains 105K prompts. To ensure the same total annotation budget across the two finetuning paradigms, we construct two calibration splits: (i) SFT uses 5K prompts with 10 annotations each; (ii) RL uses 50K prompts with a single annotation each. We also apply swap augmentation (Li et al., 2024) by duplicating each pair with A/B swapped and the label flipped. For evaluation, we sample 1K prompts from the original validation set, each with 10 annotations to form probabilistic labels.

Preference Annotation. Since most existing datasets lack sufficient multi-rater annotation for reliable probability estimates, to evaluate our method at scale, we employ Gemini-2.5-Flash (Comanici et al., 2025) as an advanced teacher to generate pairwise preference labels with brief rationales. We set the temperature to 1.0 and condition on a randomly sampled persona (Appendix F) to increase coverage and reduce prompt-induced bias. Across calibration and evaluation splits, this yields ∼110K total annotations. For each comparison instance 𝒙, we convert 𝑚 independent teacher votes into a probabilistic target ˆ𝑝(𝒙) = 𝑚1 𝑚𝑗=1 {𝐵 ≻ 𝐴} . For the SFT + CoT setting, we additionally elicit teacher reasoning traces as expert demonstrations by conditioning on the obtained ˆ𝑝(𝒙).

Response Format and Reward Instantiation. The autorater is prompted to compare responses A vs.

- B and emit (optionally) a chain-of-thought enclosed in <think> tags, followed by a single probability

within a <prob_rb_better> tag representing 𝑝𝜽(𝒙) ≈ Pr[𝐵 ≻ 𝐴 | 𝒙]. Our prompts are provided in Appendix F. In SFT, we maximize the likelihood of the target token sequence that encodes ˆ𝑝(𝒙). In RL, we parse the numeric probability via a deterministic parser 𝑔 (implemented by rule-based string-matching) and optimize either 𝑅Brier in equation 1 or 𝑅Log in equation 2. Unparsable outputs receive a default reward (0 for Brier; log𝜖 for Log), which empirically drives the parsability rate 𝑠𝜽(𝒙) toward 1. When dense labels are available, the SFT objective provides a low-variance target for 𝑝𝜽; with single-label supervision, the RL objectives remain consistent and, being strictly proper, recover 𝑝∗(𝒙) in expectation.

Base Models and Finetuning Protocol. We finetune the instruction-tuned Gemma-2-9B (Gemma Team, 2024) and Qwen-2.5-7B (Qwen Team, 2024) models with full-parameter updates for both SFT and RL. For RL, we use GRPO (Shao et al., 2024). For 𝑅Log we set 𝜖 = 10−3 to avoid degenerate rewards. Full hyperparameters and training details are reported in Appendix C.

### 5. Experimental Results

##### 5.1. Main Evaluation

In this section, we empirically demonstrate that our distribution-matching finetuning approaches can lead to better performing and calibrated autoraters.

Baselines. We consider the following four types of zero-shot baselines that can be immediately applied to any existing discrete autoraters to obtain probabilistic predictions:

- (1) Verbalized Confidence (Tian et al., 2023): The autorater is directly prompted to provide a confidence score without intermediate reasoning.
- (2) Verbalized Confidence w/ CoT (Wei et al., 2022): The autorater first generates a step-by-step chain-of-thought explanation before providing its confidence score.
- (3) Self-Consistency (Wang et al., 2023): The autorater aggregates preferences over 𝑁 independent CoT samples. The confidence for a response is the fraction of samples that voted for it.
- (4) Logit-based Confidence: Confidence is derived by applying a softmax function to the model’s output

logits 𝑧𝜏𝑖 for the verbalized preference tokens 𝜏𝑖 (“A” or “B”), i.e., 𝑝(𝑦 = 𝑖 | 𝒙) = 𝑒𝑧𝜏𝑖/ 𝑖 𝑒𝑧𝜏𝑖, 𝑖 ∈ {0, 1}.

Additionally, we consider the following calibration baselines that extend the logit-based confidence:

- (1) Temperature Scaling (Guo et al., 2017) is a post-hoc calibration method that rescales pre-softmax logits 𝒛 by a single scalar temperature 𝑇 > 0 learned on a held-out calibration set by minimizing negative log likelihood, producing calibrated confidences 𝒑ˆTS(𝒚 | 𝒙) = softmax(𝒛/𝑇).
- (2) Contextual Calibration (Zhao et al., 2021) is a test-time debiasing method that estimates the prompt-induced prior using a content-free probe (e.g., “N/A”), then corrects predictions by

subtracting this bias in logit space (or dividing probabilities): 𝒑ˆCC(𝒚 | 𝒙) = 𝑾𝒑(𝒚 | 𝒙), where 𝑾 = diag(𝒑(𝒚 | [N/A]))−1 makes the content-free prediction uniform and reduces bias.

- (3) Batch Calibration (Zhou et al., 2024) is a zero-shot, inference-only correction that estimates the

contextual bias 𝒃 from the current test batch {𝒙𝑖}𝑖𝐵=0 via 𝒃 = 𝔼𝒙∼𝑃(𝒙) 𝒑(𝒚 | 𝒙) ≈ 1𝐵

𝐵 𝑖=1 𝒑(𝒚 | 𝒙𝑖). Each example is then calibrated by dividing by this bias term (or equivalently, subtracting log 𝒃 from logits): 𝒑ˆBC(𝒚 | 𝒙) ∝ 𝒑(𝒚 | 𝒙)/𝒃. To ensure estimation accuracy, we use the entire test set in our experiments to estimate the bias.

- Table 1 | Main experiment results comparing our methods against zero-shot and calibration baselines on two models. We evaluate alignment (MSE), performance (Agreement, F1 Score), and calibration (ECE, Brier).

Alignment Performance Calibration Model Method MSE↓ Agr.↑ F1↑ ECE↓ Brier↓

Zero-shot Baselines

Verbalized Confidence 0.1255 0.7773 0.5260 0.1183 0.1615 Verbalized Confidence w/ CoT 0.1065 0.7893 0.5345 0.0869 0.1445 Self-consistency (N=10) 0.1248 0.7853 0.5482 0.1397 0.1551 Self-consistency (N=30) 0.1217 0.7921 0.5361 0.1374 0.1514 Logits 0.1162 0.8074 0.5665 0.1285 0.1416

Gemma-2-9B

Calibration Baselines

Temperature Scaling 0.0839 0.8074 0.5665 0.0827 0.1224 Contextual Calibration 0.1384 0.7753 0.5226 0.1598 0.1728 Batch Calibration 0.1153 0.8104 0.5482 0.1255 0.1406

Ours

SFT 0.0972 0.8314 0.5623 0.0972 0.1257 SFT w/ CoT 0.1033 0.8214 0.5575 0.1111 0.1332 RL (Brier) 0.0764 0.8706 0.5895 0.0879 0.0946 RL (Log) 0.0934 0.8545 0.5780 0.1141 0.1173

Zero-shot Baselines

Verbalized Confidence 0.1823 0.6723 0.4486 0.1846 0.2276 Verbalized Confidence w/ CoT 0.1571 0.7241 0.4866 0.1693 0.1965 Self-consistency (N=10) 0.1916 0.7091 0.4765 0.2168 0.2314 Self-consistency (N=30) 0.1840 0.7251 0.4861 0.2075 0.2212 Logits 0.1775 0.7382 0.4982 0.2102 0.2133

Qwen-2.5-7B

Calibration Baselines

Temperature Scaling 0.1173 0.7402 0.4982 0.1529 0.1646 Contextual Calibration 0.1551 0.7632 0.5159 0.1888 0.1893 Batch Calibration 0.1796 0.7402 0.4978 0.2129 0.2162

Ours

SFT 0.1143 0.8264 0.5590 0.1341 0.1394 SFT w/ CoT 0.1033 0.8122 0.6075 0.1095 0.1324 RL (Brier) 0.0893 0.8575 0.5804 0.1015 0.1103 RL (Log) 0.1192 0.8244 0.5580 0.1472 0.1474

Metrics. We assess the following three aspects of the autorater: (1) its alignment to the target preference distribution, as measured by the Mean Squared Error (MSE) between the predicted 𝑝𝜽(𝒙) and the high fidelity estimate ˆ𝑝(𝒙) of the true preference distribution 𝑝∗(𝒙), (2) its performance, in terms of agreement (Agr.) with the majority label (i.e., the mode of the target distribution) and the resulting F1 score, and (3) its calibration, as measured by Expected Calibration Error (ECE) (Guo et al., 2017) and Brier score. A more detailed discriptions of these metrics are provided in Appendix C.

Distribution-Matching Tuning Improves Preference Calibration. As shown in Table 1, our distribution-matching finetuning methods consistently outperform both zero-shot and calibration baselines across all metrics. While prompting strategies like Chain-of-Thought and self-consistency improve upon simple verbalized confidence, they still result in high alignment errors. In contrast, our finetuning approach drastically reduces this error. For instance, RL with a Brier reward achieves an MSE of just 0.0764 on Gemma-2-9B. This superior alignment translates directly into stronger performance and better calibration. Notably, on Gemma-2-9B, the RL-Brier model attains the highest agreement (0.8706), F1 score (0.5895), and lowest Brier score (0.0946) among all methods. The benefits are even more pronounced on Qwen-2.5-7B, where the same model achieves an agreement of 0.8575—a 12.4% improvement over Contextual Calibration, the best-performing baseline—while simultaneously achieving best calibration, as measured by both ECE and Brier score.

- Table 2 | Evaluation of positional bias. We report Consistency (higher is better) and expected Absolute Symmetry Deviation (𝔼[|Δ𝑆𝐷|], lower is better).

Gemma-2-9B Qwen-2.5-7B Consistency ↑ Abs. Dev. ↓ Consistency ↑ Abs. Dev. ↓ Zero-shot Baselines

Method

Verbalized Confidence 0.7301 0.2242 0.4964 0.3461 Verbalized Confidence w/ CoT 0.8094 0.1709 0.6399 0.3120 Logits 0.7963 0.1912 0.6529 0.3388

Calibration Baselines

Temperature Scaling 0.7963 0.1239 0.6489 0.1953 Contextual Calibration 0.7021 0.3202 0.7422 0.2473 Batch Calibration 0.7973 0.1893 0.6549 0.2130

Ours

SFT 0.8375 0.1875 0.8284 0.1827 SFT w/ CoT 0.7803 0.2291 0.8335 0.1654 RL (Brier) 0.8926 0.1026 0.9007 0.0964 RL (Log) 0.8776 0.1231 0.8726 0.1259

RL on Binary Labels is More Annotation-Efficient than SFT on Probabilistic Labels. A key finding is that for a fixed annotation budget, RL is a more annotation-efficient training paradigm than SFT. As seen in Table 1, RL-tuned autoraters, trained on 50K prompts with a single binary label each, consistently outperform their SFT counterparts, which were trained on 5K prompts with 10 aggregated labels each. We attribute this to the benefits of data diversity: the performance boost from seeing a 10× larger set of unique prompts appears to outweigh the advantage of learning from a less

noisy, aggregated target on a smaller dataset. Within the RL framework, the Brier reward consistently yields better results than the Log reward. This is likely because the Log reward’s heavy penalties for tail miscalibrations can introduce training instability, whereas the Brier reward provides a smoother optimization landscape.

A Unbiased B

VC

VC+CoT

Logits

TS

##### 5.2. Evaluation of Positional Bias

CC

LM-based autoraters, even those based on powerful proprietary models such as GPT-4, are known to be susceptible to positional bias (Wang et al., 2024a; Zheng et al., 2023), which causes their final verdict to be dependent on the order of the responses and thus undermines the reliability of their judgment result. To evaluate the positional bias of the probabilistic autoraters, for each prompt 𝒙, we perform inference twice by swapping the order of the responses to obtain two predicted probabilities: 𝑝orig that predicts Pr[𝐵 ≻ 𝐴 | 𝒙] and 𝑝swap that predicts Pr[𝐴 ≻ 𝐵 | 𝒙].

BC

SFT+CoT

RL-Brier

-1 -0.5 0 0.5 1 Symmetry Deviation

Figure 2 | Positional bias by method for Qwen-2.5-7B. Each horizontal box shows the distribution of Symmetry Deviation (Δ𝑆𝐷): 0 is swap-symmetric, −1 indicates bias toward A, and +1 toward B. The black solid line marks the median, while the green dashed line marks the mean.

Metrics. Following Zheng et al. (2023), we measure consistency, i.e., the ratio of cases where the autorater gives consistent verdicts when swapping the order of the two responses. Additionally, we measure Symmetry Deviation (Δ𝑆𝐷) as Δ𝑆𝐷 := 𝑝orig + 𝑝swap − 1. Ideally, an unbiased autorater should produce 𝑝orig and 𝑝swap that sum to 1, and thus the estimated Δ𝑆𝐷 would approximate 0. A positive Δ𝑆𝐷 indicates bias toward re-

sponse B, and vice versa. We report the expected Absolute Symmetry Deviation across the dataset as 𝔼[|Δ𝑆𝐷|] ≈ 𝑁1

𝑁 𝑖=1|𝑝orig,𝑖 + 𝑝swap,𝑖 − 1|.

Distribution-Matching Tuning Reduces Positional Bias. Probabilistic autoraters, like their discrete counterparts, are susceptible to positional bias, as shown in Table 2. This bias can be severe. For example, the zero-shot verbalized confidence method on Qwen-2.5-7B yields a poor consistency of just 0.4964. While other baselines, including CoT prompting and post-hoc calibration, can mitigate this issue, significant bias remains. For instance, on Qwen-2.5-7B, no single baseline excels at both metrics, with Contextual Calibration achieving the highest consistency (0.7422) and Temperature Scaling achieving the lowest deviation (0.1953). By contrast, our distribution-matching finetuning nearly eliminates this bias. The RL-Brier model, in particular, achieves a consistency of 0.9007 and a near-perfect absolute symmetry deviation of 0.0964. This dramatic improvement is visualized in Figure 2, which shows that baseline methods exhibit heavily skewed deviation distributions, indicating a systematic preference for one response position (in this case, response B). Conversely, our finetuned models center the distribution tightly around zero, demonstrating robust swap-symmetry and verifying their effectiveness at debiasing autoraters.

##### 5.3. Out-of-Distribution Evaluation on Human-Annotated Data

To validate our approach against genuine human judgments, we conduct an out-of-distribution evaluation on the PandaLM test set (Wang et al., 2024b) using our autoraters finetuned on JudgeLM

subset. This benchmark contains 1K samples, each independently annotated by three human experts. Following the standard protocol for this dataset, we treat the majority vote as the ground truth and report agreement, precision, recall, and F1 score.

Table 3 | Comparison of autorater performance on the PandaLM (Wang et al., 2024b) test set based on humanannotated data. We report Agreement, Precision, Recall, and F1 Score. Results marked by * are reported by Zhu et al. (2025).

Method Agreement↑ Precision↑ Recall↑ F1↑ Zero-shot Baselines

GPT-3.5* 0.6296 0.6195 0.6359 0.5820 GPT-4* 0.6647 0.6620 0.6815 0.6180

Calibrated Autoraters Are Better Aligned with Human Preference. As shown in Table 3, our models demonstrate superior alignment with human judgments compared to both powerful zero-shot models like GPT-4 and specialized, finetuned judges, including PandaLM-7B (Wang

Finetuned Baselines

PandaLM-7B* 0.5926 0.5728 0.5923 0.5456 JudgeLM-7B* 0.6507 0.6689 0.7195 0.6192

Ours (Gemma-2-9B)

SFT 0.6856 0.7103 0.5196 0.4998 SFT w/ CoT 0.7247 0.6533 0.6166 0.6266 RL (Brier) 0.7317 0.6983 0.6048 0.6220 RL (Log) 0.7357 0.4923 0.5487 0.5176

- et al., 2024b) and JudgeLM-7B (Zhu et al.,
- 2025). The results are particularly compelling given the data efficiency of our method. For example, our Qwen-2.5-7B model, after SFT with CoT, achieves a state-of-the-art F1 score of 0.6417. This performance surpasses JudgeLM7B, a model trained on the full 100K JudgeLM training set, i.e., 20× more data than what’s used by our SFT model. Even without CoT, our SFT model achieves an agreement of 0.7027, outperforming all baselines, including GPT-4. These findings confirm that our distribution-matching framework is a highly data-efficient method for aligning autoraters with nuanced human preferences.

Ours (Qwen-2.5-7B)

SFT 0.7027 0.4720 0.5240 0.4947 SFT w/ CoT 0.7187 0.6358 0.6522 0.6417 RL (Brier) 0.7297 0.8185 0.5617 0.5564 RL (Log) 0.7157 0.8129 0.5454 0.5361

##### 5.4. Out-of-Distribution Evaluation on Objective Tasks

To assess performance on tasks with a single ground truth (i.e., the target preference distribution degenerates to a single point), we evaluate our models on JudgeBench (Tan et al., 2025), a

benchmark comprising four objective tasks (Knowledge, Reasoning, Mathematics, and Coding) with binary, verifiable labels. This benchmark allows for comparison against a diverse set of models, including stateof-the-art proprietary APIs such as GPT-4o and Gemini-1.5-pro, multiagent judges such as ChatEval (Chan et al., 2024), and several specialized finetuned judges, including PandaLM7B (Wang et al., 2024b), Prometheus27B (Kim et al., 2024b), JudgeLM-

Table 4 | Evaluation of autoraters on JudgeBench (Tan et al., 2025) across four objectively labeled tasks: Knowledge, Reasoning, Mathematics, and Coding. We report evaluation accuracy in percentage. Results marked by * are reported by Tan et al. (2025).

Method Knowledge Reasoning Math Coding Overall Zero-shot Baselines

GPT-4o* 44.16 47.96 66.07 61.90 50.86 Gemini-1.5-pro* 45.45 44.90 53.57 28.57 44.57

Multi-Agent Baseline

ChatEval* 32.47 31.63 44.64 30.95 34.00 Finetuned Baselines

PandaLM-7B* 9.09 21.43 7.14 16.67 13.14 Prometheus2-7B* 38.31 25.51 35.71 42.86 34.86 JudgeLM-7B* 23.38 29.59 32.14 11.90 25.14 AutoJ-13B* 40.26 29.59 44.64 28.57 36.57 Skywork-Critic-8B* 51.30 54.08 73.21 33.33 53.43

- 7B (Zhu et al., 2025), AutoJ-13B (Li et al., 2024), and Skyuwork-Critic-
- 8B (Shiwen et al., 2024). Following the official protocol (Tan et al., 2025), we mitigate positional bias by evaluating each response pair twice, with swapped response order, and aggregating the results to obtain the final verdict.

Ours (RL w/ Brier)

Qwen-2.5-7B 39.61 46.94 60.71 38.10 44.86 Gemma-2-9B 39.61 55.10 58.93 35.71 46.57

Calibrated Autoraters Remain Performant on Objective Tasks. As shown in Table 4, training our autoraters to model preference distributions does not compromise their performance on objective tasks. Our RL-Brier tuned Gemma-2-9B model, for instance, achieves the highest accuracy of any model on the reasoning task (55.10%). Its overall accuracy of 46.57% surpasses strong baselines like Gemini-1.5-pro and all other finetuned judges except for Skywork-Critic-8B, which was trained on a substantially larger dataset1. Our Qwen-2.5-7B model is also highly competitive, achieving an overall accuracy of 44.86%. These results demonstrate that our calibration framework produces versatile probabilistic autoraters that excel at judging subjective tasks without sacrificing their effectiveness on objective, fact-based evaluations.

### 6. Related Work

Uncertainty of Human Annotations. There is a growing recognition that human-annotated data is not monolithic. Researchers have highlighted the importance of modeling label ambiguity and disagreement in standard classification tasks (Baan et al., 2022; Nie et al., 2020; Zhou et al., 2022). Most relevant to our work, Elangovan et al. (2025) argue that standard correlation metrics for evaluating LLM judges are insufficient and propose new metrics that stratify data based on human label uncertainty. We take the next step by not only evaluating with respect to this uncertainty, but also proposing methods to directly train models to capture it.

1While the exact size of the training data is not disclosed, the Skywork-Critic-8B model is described as having been finetuned on an array of high-quality datasets, including the Skywork-Reward-Preference dataset (80K samples), the Open-Critic-GPT dataset (55K samples), and other human-annotated and synthetic data.

Calibration of LLMs. The calibration of LLMs is a well-studied problem. Early work focused on post-hoc calibration methods or prompting strategies to elicit confidence (Tian et al., 2023; Xiong et al., 2024). Other approaches use supervised fine-tuning to teach models to express uncertainty, for example by using a model’s own empirical accuracy as a target label (Lin et al., 2022). More recently, reinforcement learning has been used to improve calibration. Xu et al. (2024) use a quadratic reward function with PPO to calibrate a model for question answering. Tao et al. (2024) combine a ranking loss with an order-preserving reward to align a model’s outputs. In the context of reward modeling, Leng et al. (2025) address the overconfidence of reward models directly by proposing PPO variants to align quality with verbalized confidence. In concurrent work, Damani et al. (2025); Stangel et al. (2025) propose designing rewards with proper scoring rules to improve confidence calibration on question-answering tasks. Our work differs by focusing specifically on the autorater calibration problem and by proposing a framework grounded in modeling the true distribution of human preferences, rather than a single notion of correctness.

LLM-as-a-Judge. The use of powerful large language models as automated evaluators has been explored extensively. This includes the creation of benchmarks (Tan et al., 2025; Zeng et al., 2024; Zheng et al., 2023), analyses of various biases (Wang et al., 2024a; Ye et al., 2024), and methods for training specialized judge models (Kim et al., 2024a; Li et al., 2024; Saha et al., 2025; Wang et al., 2024b; Zhu et al., 2025). Additionally, recent work has also investigated the role of model uncertainty (Xie et al., 2025) and non-transitivity (Xu et al., 2025) in LLM judge evaluations. Our work contributes to this line by addressing a fundamental aspect of judge reliability: its calibration to the inherent subjectivity of the evaluation task.

### 7. Conclusion

In this paper, we address the core limitation of training autoraters on discrete preference labels, a practice that overlooks the subjective and distributional nature of human judgment. We introduced a general probabilistic framework to calibrate autoraters to model the full preference distribution. Our empirical results show that finetuning with distribution-matching objective leads to autoraters that are better aligned with target preference distributions, with significant improvements in calibration and a substantial reduction in positional bias. By shifting the goal from predicting a single verdict to modeling the spectrum of human opinion, this work contributes to building more reliable, fair, and robust AI alignment systems.

### Acknowledgments

The authors would like to thank Kai Chen from Google DeepMind for his helpful discussion and constructive feedback during the early stage of this project.

### Author Contributions

ZL and YX conceptualized the project and designed the experiments. ZL conducted the experiments and analysis. ZL and YX summarized the major findings and drafted the manuscript with inputs from XL, CH, GL, KG, BD, DS, PZ, HP, YS, and PG. DS, MK, and BAM provided intellectual input and contributed to manuscript revisions. All authors approved the final version.

### References

- R. K. Arora, J. Wei, R. S. Hicks, P. Bowman, J. Quiñonero-Candela, F. Tsimpourlas, M. Sharman, M. Shah, A. Vallone, A. Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

J. Baan, W. Aziz, B. Plank, and R. Fernández. Stop measuring calibration when humans disagree. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1892–1915, 2022.

Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

- R. A. Bradley and M. E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

- B. Brehmer. Social judgment theory and the analysis of interpersonal conflict. Psychological bulletin, 83(6):985, 1976.
- C.-M. Chan, W. Chen, Y. Su, J. Yu, W. Xue, S. Zhang, J. Fu, and Z. Liu. Chateval: Towards better llmbased evaluators through multi-agent debate. In The Twelfth International Conference on Learning Representations, 2024.

- W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.

- G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram,

- D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, and R. Xin. Free dolly: Introducing the world’s first truly open instruction-tuned llm, 2023. URL https://www.databricks.com/blog/2023/04/12/ dolly-first-open-commercially-viable-instruction-tuned-llm.

M. Damani, I. Puri, S. Slocum, I. Shenfeld, L. Choshen, Y. Kim, and J. Andreas. Beyond binary rewards: Training lms to reason about their uncertainty. arXiv preprint arXiv:2507.16806, 2025.

A. P. Dawid and A. M. Skene. Maximum likelihood estimation of observer error-rates using the em algorithm. Journal of the Royal Statistical Society: Series C (Applied Statistics), 28(1):20–28, 1979.

- A. Elangovan, L. Xu, J. Ko, M. Elyasi, L. Liu, S. B. Bodapati, and D. Roth. Beyond correlation: The impact of human uncertainty in measuring the effectiveness of automatic evaluation and llm-as-a-judge. In The Thirteenth International Conference on Learning Representations, 2025.

L. Ge, D. Halpern, E. Micha, A. D. Procaccia, I. Shapira, Y. Vorobeychik, and J. Wu. Axioms for ai alignment from human feedback. Advances in Neural Information Processing Systems, 37:80439– 80465, 2024.

Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

- C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

M. Kahng, I. Tenney, M. Pushkarna, M. X. Liu, J. Wexler, E. Reif, K. Kallarackal, M. Chang, M. Terry, and L. Dixon. LLM Comparator: Interactive analysis of side-by-side evaluation of large language models. IEEE Transactions on Visualization and Computer Graphics, 31(1), 2025. doi: 10.1109/ TVCG.2024.3456354.

- S. Kim, J. Shin, Y. Cho, J. Jang, S. Longpre, H. Lee, S. Yun, S. Shin, S. Kim, J. Thorne, et al. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, 2024a.

- S. Kim, J. Suk, S. Longpre, B. Y. Lin, J. Shin, S. Welleck, G. Neubig, M. Lee, K. Lee, and M. Seo. Prometheus 2: An open source language model specialized in evaluating other language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4334–4353, 2024b.

J. Leng, C. Huang, B. Zhu, and J. Huang. Taming overconfidence in llms: Reward calibration in rlhf. In The Thirteenth International Conference on Learning Representations, 2025.

J. Li, S. Sun, W. Yuan, R.-Z. Fan, P. Liu, et al. Generative judge for evaluating alignment. In The Twelfth International Conference on Learning Representations, 2024.

S. Lin, J. Hilton, and O. Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.
- J. L. Mumpower and T. R. Stewart. Expert judgement and expert disagreement. Thinking & Reasoning, 2(2-3):191–212, 1996.

- R. Munos, M. Valko, D. Calandriello, M. G. Azar, M. Rowland, Z. D. Guo, Y. Tang, M. Geist, T. Mesnard, C. Fiegel, et al. Nash learning from human feedback. In International Conference on Machine Learning, pages 36743–36768. PMLR, 2024.

- Y. Nie, X. Zhou, and M. Bansal. What can we learn from collective human opinions on natural language inference data? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9131–9143, 2020.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama,

- A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- R. J. Passonneau and B. Carpenter. The benefits of a model of annotation. Transactions of the Association for Computational Linguistics, 2:311–326, 2014.
- S. Paun, J. Chamberlain, U. Kruschwitz, J. Yu, and M. Poesio. A probabilistic annotation model for crowdsourcing coreference. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1926–1937, 2018.

- E. Pavlick and T. Kwiatkowski. Inherent disagreements in human textual inferences. Transactions of the Association for Computational Linguistics, 7:677–694, 2019.

- B. Peng, C. Li, P. He, M. Galley, and J. Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.

Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm. github.io/blog/qwen2.5/.

- S. Saha, X. Li, M. Ghazvininejad, J. E. Weston, and T. Wang. Learning to plan & reason for evaluation with thinking-llm-as-a-judge. In Forty-second International Conference on Machine Learning, 2025.

- J. Schulman. Approximating kl divergence, 2020. URL http://joschu. net/blog/kl-approx. html, 2020.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

T. Shiwen, Z. Liang, C. Y. Liu, L. Zeng, and Y. Liu. Skywork critic model series. https: //huggingface.co/Skywork, September 2024. URL https://huggingface.co/Skywork.

- A. Siththaranjan, C. Laidlaw, and D. Hadfield-Menell. Distributional preference learning: Understanding and accounting for hidden context in rlhf. In The Twelfth International Conference on Learning Representations, 2024.

P. Stangel, D. Bani-Harouni, C. Pellegrini, E. Özsoy, K. Zaripova, M. Keicher, and N. Navab. Rewarding doubt: A reinforcement learning approach to calibrated confidence expression of large language models. arXiv preprint arXiv:2503.02623, 2025.

- S. Tan, S. Zhuang, K. Montgomery, W. Y. Tang, A. Cuadron, C. Wang, R. Popa, and I. Stoica. Judgebench: A benchmark for evaluating llm-based judges. In The Thirteenth International Conference on Learning Representations, 2025.
- S. Tao, L. Yao, H. Ding, Y. Xie, Q. Cao, F. Sun, J. Gao, H. Shen, and B. Ding. When to trust llms: Aligning confidence with response quality. In Findings of the Association for Computational Linguistics ACL 2024, pages 5984–5996, 2024.

R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023.

- K. Tian, E. Mitchell, A. Zhou, A. Sharma, R. Rafailov, H. Yao, C. Finn, and C. D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442, 2023.

- H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- P. Wang, L. Li, L. Chen, Z. Cai, D. Zhu, B. Lin, Y. Cao, L. Kong, Q. Liu, T. Liu, et al. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, 2024a.

- X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023.

- Y. Wang, Z. Yu, W. Yao, Z. Zeng, L. Yang, C. Wang, H. Chen, C. Jiang, R. Xie, J. Wang, et al. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. In The Twelfth International Conference on Learning Representations, 2024b.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- Q. Xie, Q. Li, Z. Yu, Y. Zhang, Y. Zhang, and L. Yang. An empirical analysis of uncertainty in large language model evaluations. In The Thirteenth International Conference on Learning Representations, 2025.

M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, and B. Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In The Twelfth International Conference on Learning Representations, 2024.

- T. Xu, S. Wu, S. Diao, X. Liu, X. Wang, Y. Chen, and J. Gao. Sayself: Teaching llms to express confidence with self-reflective rationales. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5985–5998, 2024.

- Y. Xu, L. Ruis, T. Rocktäschel, and R. Kirk. Investigating non-transitivity in llm-as-a-judge. In Forty-second International Conference on Machine Learning, 2025.

J. Ye, Y. Wang, Y. Huang, D. Chen, Q. Zhang, N. Moniz, T. Gao, W. Geyer, C. Huang, P.-Y. Chen, et al. Justice or prejudice? quantifying biases in llm-as-a-judge. In Neurips Safe Generative AI Workshop, 2024.

- Z. Zeng, J. Yu, T. Gao, Y. Meng, T. Goyal, and D. Chen. Evaluating large language models at evaluating instruction following. In The Twelfth International Conference on Learning Representations, 2024.

- Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh. Calibrate before use: Improving few-shot performance of language models. In International conference on machine learning, pages 12697–

12706. PMLR, 2021.

- L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

H. Zhou, X. Wan, L. Proleev, D. Mincu, J. Chen, K. A. Heller, and S. Roy. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. In The Twelfth International Conference on Learning Representations, 2024.

X. Zhou, Y. Nie, and M. Bansal. Distributed nli: Learning to predict human opinion distributions for language reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 972–987, 2022.

L. Zhu, X. Wang, and X. Wang. Judgelm: Fine-tuned large language models are scalable judges. In The Thirteenth International Conference on Learning Representations, 2025.

### A. Extended Related Work

Disagreement in Human Judgments. Psychology studies have shown that even when individuals are presented with the same evidence, they can rationally arrive at different choices for what is the “best” because of systematic differences in their judgmental processes (Mumpower and Stewart, 1996), including (i) different problem definitions, where disagreement stems from judging different environmental criteria or a confusion between scientific facts and personal value, (ii) different organizing principles, where individuals may apply different cue weights, function forms, or overall biases when integrating the same set of information, as explained by Social Judgment Theory (Brehmer, 1976), and (iii) different mental models about how the evidence was generated. In our context, this explains why human annotators can disagree about LLM outputs even under identical prompts or instructions: they face different value trade-offs (e.g., safety vs. helpfulness) or apply different thresholds for judgment, which may result in distinct yet internally coherent choices of what is preferred.

Probabilistic Models of Agreement. Probabilistic modeling of agreement dates back to the 1950s. Classic models such as the Dawid and Skene (1979) model aim to infer a single “gold standard” label from multiple, often noisy, annotators. Studies (Passonneau and Carpenter, 2014; Paun et al., 2018) have shown that by modeling annotator reliability, these methods can produce high-quality data even from non-expert crowd workers and outperform simpler aggregation techniques such as majority vote. In contrast to this approach, a recent line of work shows that, for many complex and subjective tasks, disagreement is not simply noise but a valid and reproducible signal that reflects legitimate differences in human interpretation (Nie et al., 2020; Pavlick and Kwiatkowski, 2019). Our probabilistic autorater aligns with this view by treating the annotator heterogeneity as the prediction target to capture the full distribution of human judgments.

Learning from Human Feedback. Canonical Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022) involves learning a reward model from pairwise human preferences, often by fitting a Bradley-Terry model via maximum likelihood estimation. This approach implicitly assumes that heterogeneous feedback from different humans is merely a noisy estimate of a single ground-truth preference. However, a growing body of work (Ge et al., 2024; Munos et al., 2024; Siththaranjan et al., 2024) suggests that, in the context of AI alignment, preference heterogeneity often reflects legitimate differences in individual values and should be modeled directly rather than averaged away. Our research complements this line of work as our finetuned probabilistic autoraters can be employed in such preference-based RL to better align models with the diversity of human preferences.

### B. Proofs

##### B.1. Preference Distribution and Estimation

Let ℎ index an annotator drawn from a population distribution 𝑝(ℎ). For a pairwise input 𝑿 = 𝒙, define the annotator-specific preference probability

𝑝ℎ(𝒙) = Pr[𝑌 = 1 | 𝑿 = 𝒙, ℎ], 𝑌 ∈ {0, 1} (1 indicates 𝐵 ≻ 𝐴). The population (ground-truth) preference is the annotator-average:

𝑝∗(𝒙) = 𝔼ℎ∼𝑝(ℎ)[ 𝑝ℎ(𝒙) ].

A common special case here is when annotators hold stable preferences, i.e., where 𝑝ℎ(𝒙) ∈ {0, 1} (each annotator has a fixed judgment per 𝒙). In that case, 𝑝ℎ(𝒙) is the indicator of “ℎ prefers B” and 𝑝∗(𝒙) is the population fraction preferring B.

Given 𝑚 i.i.d. labels {𝑦(𝑗)}𝑚𝑗=1 collected by sampling annotators ℎ𝑗 i.i.d.∼ 𝑝(ℎ) and then 𝑦(𝑗) ∼ Bernoulli(𝑝ℎ𝑗(𝒙)), the Monte Carlo estimator

###### ∑︁𝑚

1

ˆ𝑝𝑚(𝒙) =

𝑦(𝑗)

𝑚

𝑗=1

is an unbiased estimate of 𝑝∗(𝒙) with variance decreasing as 1/𝑚. Lemma 2 (Unbiasedness and variance of the multi-annotator estimate). For any fixed 𝒙 and i.i.d. sampling as above,

𝔼[ˆ𝑝𝑚(𝒙)] = 𝑝∗(𝒙), Var[ˆ𝑝𝑚(𝒙)] =

𝑝∗(𝒙) 1 − 𝑝∗(𝒙) 𝑚

.

Proof. By the law of total expectation, 𝔼[𝑦(𝑗) | 𝒙] = 𝔼ℎ[𝑝ℎ(𝒙)] = 𝑝∗(𝒙), so 𝔼[ˆ𝑝𝑚(𝒙)] = 𝑝∗(𝒙). Since the 𝑦(𝑗) are i.i.d. Bernoulli with mean 𝑝∗(𝒙) (marginalizing over ℎ), Var(ˆ𝑝𝑚(𝒙)) = Var(𝑦(1))/𝑚 = 𝑝∗(𝒙)(1 − 𝑝∗(𝒙))/𝑚. □

##### B.2. Proof of Proposition 1

We first recall the setup. For (𝑿,𝑌) ∼ D with 𝑌 ∈ {0, 1} and 𝑌 | 𝑿 = 𝒙 ∼ Bernoulli(𝑝∗(𝒙)), the policy 𝝅𝜽(𝝉 | 𝒙) emits a token sequence 𝝉 intended to encode a numeric probability. A deterministic parser 𝑔 : T → [0, 1] ∪ {⊥} returns either a number 𝑝 ∈ [0, 1] or the unparsable symbol ⊥. Let 𝑠𝜽(𝒙) = Pr𝝉∼𝝅

𝜽(·|𝒙)[𝑔(𝝉) ≠ ⊥]. The piecewise rewards are:

1 − (𝑝 − 𝑦)2, 𝑔(𝝉) = 𝑝 ∈ [0, 1], 0, 𝑔(𝝉) = ⊥,

𝑅Brier(𝝉; 𝑦) =

𝑦 log 𝑝′ + (1 − 𝑦) log(1 − 𝑝′), 𝑔(𝝉) = 𝑝 ∈ [0, 1], log𝜖, 𝑔(𝝉) = ⊥,

𝑅Log(𝝉; 𝑦) =

with 𝑝′ = clip(𝑝, 𝜖, 1−𝜖) and 𝜖 ∈ (0, 12). The population objective is 𝐽𝑅(𝜽) = 𝔼(𝑿,𝑌) 𝔼𝝉∼𝝅

𝜽(·|𝑿)[𝑅(𝝉;𝑌)]. Fix 𝒙 and abbreviate 𝑝∗ = 𝑝∗(𝒙). All statements below are conditional on 𝑿 = 𝒙 and the conclusion

holds for almost every 𝒙 (w.r.t. the marginal of 𝑿).

This proof utilizes the following observations: (i) for any random variable 𝑍 with finite variance and any constant 𝑎, 𝔼[(𝑍 −𝑎)2] = (𝔼[𝑍] −𝑎)2 +Var(𝑍); (ii) the function 𝜙(𝑝) = 𝑝∗ log 𝑝+ (1− 𝑝∗) log(1− 𝑝) is strictly concave on 𝑝 ∈ (𝜖, 1 − 𝜖) with unique maximizer at 𝑝 = 𝑝∗ (and at the boundary when 𝑝∗ ∉ (𝜖, 1 − 𝜖)).

Brier. Let 𝑃 denote the random numeric report 𝑔(𝝉) conditional on 𝑔(𝝉) ≠ ⊥ (so 𝑃 is defined with probability 𝑠𝜽(𝒙)). Then

𝔼𝝉,𝑌 [𝑅Brier(𝝉;𝑌) | 𝒙] = 𝑠𝜽(𝒙) 𝔼𝝉 𝔼𝑌|𝒙 1 − (𝑃 − 𝑌)2 + 1 − 𝑠𝜽(𝒙) · 0

= 𝑠𝜽(𝒙) 𝔼𝝉 1 − 𝔼𝑌|𝒙 (𝑌 − 𝑃)2

= 𝑠𝜽(𝒙) 𝔼𝝉 1 − 𝔼𝑌|𝒙[𝑌] − 𝑃 2 − Var𝑌|𝒙(𝑌) applying (i)

= 𝑠𝜽(𝒙) 𝔼𝝉 1 − (𝑝∗ − 𝑃)2 − 𝑝∗(1 − 𝑝∗)

= 𝑠𝜽(𝒙) 1 − 𝔼𝝉 (𝑃 − 𝑝∗)2 − 𝑝∗(1 − 𝑝∗)

= 𝑠𝜽(𝒙) 1 − (𝔼𝝉[𝑃] − 𝑝∗)2 + Var𝝉(𝑃) − 𝑝∗(1 − 𝑝∗) . applying (i)

For fixed 𝑠𝜽(𝒙) this is maximized by setting Var(𝑃) = 0 (deterministic numeric output) and 𝔼[𝑃] = 𝑝∗ (truthful reporting). Moreover, since 1−𝑝∗(1−𝑝∗) > 0, the expectation is positive when Var(𝑃) = 0 and 𝔼[𝑃] = 𝑝∗. Increasing 𝑠𝜽(𝒙) strictly increases the expectation; hence an optimizer satisfies 𝑠𝜽(𝒙) = 1. Therefore, at any global maximizer, 𝑃 is almost surely constant and equals 𝑝∗, i.e., 𝑝𝜽(𝒙) = 𝑝∗.

Log with clipping. Write 𝑃′ for the clipped numeric report when parsable. Then

𝔼𝝉,𝑌 𝑅Log(𝝉;𝑌) | 𝒙 = 𝑠𝜽(𝒙) 𝔼𝝉 𝔼𝑌|𝒙 [𝑌 log 𝑃′ + (1 − 𝑌) log(1 − 𝑃′)] + (1 − 𝑠𝜽(𝒙)) log𝜖

= 𝑠𝜽(𝒙) 𝔼𝝉[ 𝑝∗ log 𝑃′ + (1 − 𝑝∗) log(1 − 𝑃′) ] + (1 − 𝑠𝜽(𝒙)) log𝜖

= 𝑠𝜽(𝒙) 𝔼𝝉[𝜙(𝑃′)] + (1 − 𝑠𝜽(𝒙)) log𝜖.

By strict concavity of 𝜙 and Jensen’s inequality, 𝔼[𝜙(𝑃′)] ≤ 𝜙(𝔼[𝑃′]) with equality iff 𝑃′ is almost surely constant (deterministic numeric output). The maximizer over 𝑃′ ∈ [𝜖, 1 − 𝜖] is uniquely 𝑃′ ≡ clip(𝑝∗, 𝜖, 1 − 𝜖) (truthful reporting). Finally, at this maximizer 𝜙 clip(𝑝∗, 𝜖, 1 − 𝜖) > log𝜖, so allocating any mass to unparsable outputs (which yields log𝜖) strictly reduces the expectation; hence 𝑠𝜽(𝒙) = 1. Therefore, the optimal report is deterministic and equals the clipped truth clip(𝑝∗, 𝜖, 1 − 𝜖).

Combining the two cases completes the proof. □

### C. Implementation Details

##### C.1. Evaluation Details

We measure the probabilistic autorater’s performance by comparing its judgment to human annotation or discretized probabilistic labels annotated by Gemini-2.5-Flash. Following prior work (Wang et al., 2024b; Zhu et al., 2025), we formulate the pairwise judgment task as a three-class classification problem with labels 𝐴 ≻ 𝐵, Tie, and 𝐵 ≻ 𝐴. Let 𝑇𝑃𝑖, 𝐹𝑃𝑖, and 𝐹𝑁𝑖 denote the true positives, false positives, and false negatives for class 𝑖, respectively. We report macro-averaged metrics by computing

each score per class and then averaging over all 𝐶 classes:

###### ∑︁𝐶

1

𝑇𝑃𝑖 𝑇𝑃𝑖 + 𝐹𝑃𝑖

Precisionmacro =

,

𝐶

𝑖=1

###### ∑︁𝐶

1

𝑇𝑃𝑖 𝑇𝑃𝑖 + 𝐹𝑁𝑖

Recallmacro =

,

𝐶

𝑖=1

###### ∑︁𝐶

1

2 · 𝑇𝑃𝑖 2 · 𝑇𝑃𝑖 + 𝐹𝑃𝑖 + 𝐹𝑁𝑖

F1-scoremacro =

,

𝐶

𝑖=1

𝐶 𝑖=1 𝑇𝑃𝑖

Agreement =

.

𝐶 𝑖=1(𝑇𝑃𝑖 + 𝐹𝑁𝑖)

By convention, calibration is measured with respect to a set of discrete (binary) labels. To evaluate model calibration, we first binarize the preference distribution to obtain ground truth preference labels and then measure the Expected Calibration Error (ECE) and Brier Score. Test samples with ground truth label being “Tie” are skipped for calibration evaluation.

The ECE is calculated by dividing the confidence into 𝐾 equal-sized bins (each of size 𝐾1), and then calculating the accuracy and average confidence within each bin:

∑︁𝐾

|𝐵𝑘| 𝑁

Acc(𝐵𝑘) − Conf(𝐵𝑘) ,

ECE =

𝑘=1

###### ∑︁

###### ∑︁

1 |𝐵𝑘|

1 |𝐵𝑘|

Acc(𝐵𝑘) =

1(ˆ𝑦𝑖 = 𝑦𝑖) , Conf(𝐵𝑘) =

ˆ𝑝𝑖,

𝑖∈𝐵𝑘

𝑖∈𝐵𝑘

where 𝐵𝑘 is the number of samples whose prediction confidence falls into the interval (𝑘−𝐾1, 𝐾𝑘 ], ˆ𝑦𝑖 and 𝑦𝑖 are the predicted and true preference labels, and ˆ𝑝𝑖 is the predicted probability. By default, we set 𝐾 = 10 in our experiments.

The Brier score is calculated as 𝑁1 𝑖 𝑁=1(𝑦𝑖 − ˆ𝑝𝑖)2.

- C.2. Finetuning Details In Setting 2, we optimize the following GRPO objective (Shao et al., 2024)

𝝅𝜽(𝝉|𝒙) 𝝅𝜽old(𝝉|𝒙)

𝝅𝜽(𝝉|𝒙) 𝝅𝜽old(𝝉|𝒙)

𝐴𝑡, clip

min

###### J(𝜽) = 𝔼𝒙∼𝑝(𝒙),𝝉∼𝝅

𝜽old

with the unbiased KL estimator (Schulman, 2020)

, 1 − 𝜀, 1 + 𝜀 𝐴𝑡 − 𝛽 𝐷KL(𝝅𝜽 ∥ 𝝅ref) ,

𝝅ref(𝝉|𝒙) 𝝅𝜽(𝝉|𝒙)

𝝅ref(𝝉|𝒙) 𝝅𝜽(𝝉|𝒙)

− log

− 1,

𝐷KL(𝝅𝜽 ∥ 𝝅ref) ≈

where 𝝅𝜽 is the policy model being optimized, 𝝅𝜽old is the previous policy model, 𝝅ref is the reference policy, 𝐴𝑡 is the advantage estimate, 𝜀 is the clipping hyperparameter, and 𝛽 is the KL penalty coefficient. Detailed parameter settings for our experiments are presented in Table 5.

Table 5 | Detailed finetuning settings.

- Setting 1 — Supervised Finetuning (SFT) General

max total sequence length 2048 precision bf16

Optimization optimizer AdamW (Loshchilov and Hutter, 2019) optimizer hyperparameters 𝛽1, 𝛽2, 𝜖 = 0.9, 0.98, 10−5 weight decay 0.1 batch size 32 training epochs 3 learning rate 1 × 10−6

- Setting 2 — Reinforcement Learning (RL)

General max total sequence length 2048 precision bf16

GRPO Setting hyperparameters 𝛽 = 0.01, 𝜀 = 0.2 number of prompts per step 32 (Gemma) / 64 (Qwen) number of generations per prompt 32

Optimization optimizer AdamW (Loshchilov and Hutter, 2019) optimizer hyperparameters 𝛽1, 𝛽2, 𝜖 = 0.9, 0.999, 10−8 weight decay 0.001 batch size 512 training epochs 1 learning rate 3 × 10−7 learning rate warm-up linear warm-up ratio / steps 0.01 / 50

### D. Additional Results

##### D.1. Win Rate Prediction

We use a subset of the LMSys Chatbot Arena Conversation dataset (Kahng et al., 2025; Zheng et al., 2023) to evaluate the autorater’s ability to predict the LM’s win rate. This subset contains a total of 900 prompts for comparing responses from Gemma 1.0 and Gemma 1.1. For this evaluation, we use Qwen-2.5-7B as the base model for the autorater to avoid potential self-enhancement bias (i.e., favoring responses generated by LMs from the same family). Specifically, given two LMs 𝝅𝐴 and 𝝅𝐵, the win rate (of 𝝅𝐵) is defined as

Pr[𝝅𝐵 ≻ 𝝅𝐴] := 𝔼𝒙∼𝑝(𝒙)𝔼𝝉

𝐴∼𝝅𝐴,𝝉𝐵∼𝝅𝐵 Pr[𝝉𝐵 ≻ 𝝉𝐴|𝒙].

From the results shown in Table 6, we observe that the predicted win rates from the finetuned autoraters are more aligned with the true win rate voted by human judges.

##### D.2. RL with Probabilistic Labels

While our main experiments utilized sparse binary labels for RL, our reward function is also compatible with dense probabilistic labels. To explore how this data format affects performance, we finetuned

Table 6 | Evaluation on the Chatbot Arena Conversations dataset for comparing Gemma 1.0 to Gemma 1.1. We compare the autorater’s predicted win rate for Gemma 1.0 against the true win rate from human annotators.

Win Rate Absolute Error Method of Gemma 1.0 to Human Human 0.4344 —

Qwen-2.5-7B Verbal 0.7397 0.3053 Qwen-2.5-7B Verbal w/ CoT 0.5951 0.1607 Qwen-2.5-7B SFT 0.3146 0.1198 Qwen-2.5-7B SFT w/ CoT 0.3082 0.1262 Qwen-2.5-7B RL (Brier) 0.3640 0.0704 Qwen-2.5-7B RL (Log) 0.3662 0.0682

Qwen-2.5-7B using the Brier score reward on the same 5K prompts (each with 10 annotations) used for SFT.

As shown in Figure 3, we compared this model against several baselines: the zero-shot model with direct verbalized confidence, the SFT model trained on probabilistic labels, and our primary model trained with RL on binary labels. The results indicate that RL with probabilistic labels improves both performance (agreement) and calibration (ECE) over the zero-shot baseline. However, it underperformed compared to the autoraters trained with a larger set of binary labels, highlighting the crucial role of data diversity in achieving optimal results.

1.0

Agreement

0.8575

0.8264

ECE

0.7913

0.8

0.6723

0.6

0.4

0.1846

0.2

0.1341

0.1321

0.1015

0.0

VerbalizedconfidenceSFTw/prob.labels RLw/binarylabels RLw/prob.labels

Figure 3 | Result of RL finetuning with probabilistic labels.

### E. Discussions

We provide a discussion on the extensions and limitations of our approach and results.

First, our work initiates the study on more reliable autoraters by predicting the full preference distribution. While our empirical results focused on the pairwise evaluation setting, the general framework and analysis also apply to the pointwise evaluation setting, where the preference takes nominal or ordinal values (e.g., Likert scale). Specifically, a direction extension involves employing multi-class strictly proper scoring rules, such as the multi-class Brier score or cross-entropy, as the reward function for RL. We leave a thorough empirical study in this space as future work.

Second, our finetuning objective aims to capture the distribution of human preferences, which stems from the human-level uncertainty on the subject. As such, the uncertainty communicated by the finetuned autorater in the form of verbalized confidence is rather aleatoric than epistemic, whereas the latter may require predicting a second-order distribution (i.e., a distribution over the

predicted probability 𝑝𝜽(𝒙)). However, such an approach may significantly complicate training; instead, we demonstrate empirically that simply providing a point estimation of the probability is sufficient to improve the autorater’s alignment to the preference distribution while maintaining good out-of-distribution generalizability.

### F. Prompts

##### Prompt Set for Persona-Based Preference Annotation Base Prompt Structure:

{Persona-specific instruction here} Your output must be a single letter, either ’A’ or ’B’, enclosed in <winner> and </winner> tags. [Question] {question}

- [The Start of Response A] {responseA} [The End of Response A]
- [The Start of Response B] {responseB} [The End of Response B] Considering these factors, which response is better?

##### Persona-Specific Instructions:

- 1. Factual & Technical Specialist Your task is to evaluate two responses based on their helpfulness, relevance, accuracy, and level of detail. As a technical specialist, you would additionally consider factual correctness and precision in your judgment.
- 2. Safety & Privacy Advocate Your task is to evaluate two responses based on their helpfulness, relevance, accuracy, and level of detail. As a safety advocate, you would additionally consider safety and ethical responsibility. An unsafe response is a poor response.
- 3. Empathetic Communicator Judge the overall quality of the responses based on their helpfulness, accuracy, and detail. As an empathetic communicator, you would also pay close attention to the tone and respectfulness of the language.
- 4. Concision Champion Assess which response is better, considering its helpfulness, accuracy, and relevance. As a champion of concision, you would additionally prefer the response that is more direct and to-the-point, penalizing unnecessary verbosity.
- 5. Nuance & Depth Seeker Evaluate the helpfulness, relevance, and accuracy of both responses. As an academic researcher, you would also give importance to the depth and nuance of the explanation.
- 6. Creative & Arts Critic Assess which response is more helpful, relevant, and accurate. As a creative critic, you would also weigh the originality, style, and engagement of the writing.
- 7. Everyday User (4x Weight) From the perspective of a regular person, not an expert, evaluate which response is better overall. Your decision should be based on which one is most helpful, relevant to the question, accurate, and provides a good level of detail. The best response is the one that is most satisfying and clear.

Prompts for Eliciting Probabilistic Prediction

- 1. Verbalized Confidence

Your task is to evaluate two responses , A and B, based on the provided question. Compare the responses and determine the probability that ** Response B is better than Response A**.

Your output must be a single float between 0.0 and 1.0, enclosed in ‘< prob_rb_better >‘ and ‘</prob_rb_better >‘ tags. Do not output any other text , reasoning , or explanations.

- - A probability of 0.0 means Response A is clearly better.
- - A probability of 1.0 means Response B is clearly better.
- - A probability of 0.5 indicates they are of similar quality.

Example Output Format: <prob_rb_better >0.85</ prob_rb_better >

[Question] {question}

- [The Start of Response A]

- {responseA}

- [The End of Response A]

[The Start of Response B] {responseB}

- [The End of Response B] Probability Output:

- 2. Verbalized Confidence w/ CoT

Your task is to evaluate two responses , A and B, based on the provided

question. Compare the responses on helpfulness , accuracy , and clarity. Follow these steps for your output:

- 1. First , provide a step -by-step reasoning of your evaluation inside ‘<think >‘ and ‘</think >‘ tags. Explain which response is better and why.
- 2. Immediately after the closing ‘</think >‘ tag , provide a single float between 0.0 and 1.0 representing the probability that **Response B is better than Response A**. This float must be enclosed in ‘<prob_rb_better >‘

and ‘</prob_rb_better >‘ tags. A probability of 0.0 means Response A is clearly better , 1.0 means Response B is clearly better , and 0.5 indicates they are of similar quality.

Example Output Format: <think >Your detailed reasoning comparing A and B goes here.</think ><

prob_rb_better >0.85</ prob_rb_better >

[Question] {question}

- [The Start of Response A]

- {responseA}

- [The End of Response A]

[The Start of Response B] {responseB}

- [The End of Response B] Your Evaluation:

- 3. Self-Consistency

Your task is to evaluate two responses , A and B, based on the provided question. You must decide which response is superior. First , provide a step -by-step thinking process that justifies your decision. This reasoning should be enclosed within <think ></think > tags.

After your reasoning , provide your final decision , which must be a single character (’A’ or ’B’), enclosed within <answer ></answer > tags.

Example Output Format: <think >Response A directly answers the user ’s question about X, while Response

B is more general. Response A also provides a concrete example which is helpful. Therefore , A is the better response.</think ><answer >A</answer >

[Question] {question}

- [The Start of Response A]

- {responseA}

- [The End of Response A]

[The Start of Response B] {responseB}

- [The End of Response B] Final Evaluation:

#### 4. Logit-Based Confidence

Your task is to evaluate two responses , A and B, based on the provided question. You must decide which response is superior. Your output must be a single

character: ’A’ or ’B’. Do not output any other text , reasoning , or explanations. Example Output Format: B

[Question] {question}

- [The Start of Response A]

- {responseA}

- [The End of Response A]

[The Start of Response B] {responseB}

- [The End of Response B] Final Decision (A or B):

### G. Response Examples

- G.1. JudgeLM Dataset Gemma-2-9B (JudgeLM val set - QID 827)

- • Question Compose a haiku with the provided subject. Spring
- • Model A (vicuna-13b-1.1) Response

Fresh new start, green leaves grow Raindrops pitter-patter, flowers bloom.

- • Model B (mix-instruct-output) Response

Amidst blooming flowers Gentle breeze moves budding trees Spring awakens life.

- • Ground Truth 𝑃(B is better) = 1.00
- • Autoraters Outputs Verbalized Confidence w/ CoT (Prediction: 0.60)

<think>Both responses follow the 5-7-5 syllable structure of a haiku. Response B is slightly better because it paints a more vivid picture with

its imagery. "Amidst blooming flowers" and "Gentle breeze moves budding trees" create a stronger sense of the season’s atmosphere compared to Response A’s more general descriptions. Response B also uses more evocative language like "awakens life," which adds a deeper layer of meaning. </think><prob_rb_better>0.6</prob_rb_better>

##### SFT w/ CoT (Prediction: 1.00)

<think>1. **Analyze Response A:** Response A attempts to follow the 5-7-5 syllable structure but fails. "Fresh new start, green leaves grow" has 7 syllables, "Raindrops pitter-patter, flowers bloom" has 9 syllables. It also includes a line break within the second line, which is not a haiku characteristic.

- 2. **Analyze Response B:** Response B correctly follows the 5-7-5 syllable structure for all three lines. It also uses evocative imagery and language appropriate for the subject "Spring."
- 3. **Compare and Justify Score:** Response B is a perfect haiku, adhering to the strict structural and thematic requirements. Response A completely fails on the syllable count and line structure, making it not a haiku at all. Therefore, Response B is unequivocally better than Response A. A confidence score of 1.00 is warranted as there is no scenario where Response A would be preferred for this task.</think><prob_rb_better>1.0</prob_rb_better>

##### RL (Brier) (Prediction: 0.99)

<think>Response B is a better haiku because it follows the traditional structure of a haiku (5-7-5 syllables per line) while Response A does not. Response B also captures the essence of spring with its imagery of blooming flowers, gentle breezes, and awakening life. Response A’s lines are too long and don’t have the same poetic flow. </think><prob_rb_better>0.99</prob_rb_better>

##### RL (Log) (Prediction: 1.00)

<think>Response B is better because it adheres to the 5-7-5 syllable structure of a haiku, while Response A does not. Response B also uses more evocative language and imagery to capture the essence of spring. Response A is more of a descriptive paragraph than a haiku.</think><prob_rb_better>1.0</prob_rb_better>

Gemma-2-9B (JudgeLM val set - QID 3960)

- • Question

Tell me what are aspects of polytopes from given text In geometry, a simplicial polytope is a polytope whose facets are all simplices. For example, a simplicial polyhedron in three dimensions contains only triangular faces and corresponds via Steinitz’s theorem to a maximal planar graph. They are topologically dual to simple polytopes. Polytopes which are both simple and simplicial are either simplices or two-dimensional polygons.

- • Model A (mix-instruct-output) Response

- 1. A simplicial polytope is a polytope in geometry where each facet is a simplice.
- 2. As an illustration, a simplicial polyhedron in three dimensions with only triangular faces corresponds to a maximum planar network according to Steinitz’s theorem.
- 3. They are simple polytopes topological dual.
- 4. Two-dimensional polygons or simplices are polytopes that are both simple and simplicial.

- • Model B (alpaca-native) Response

The aspects of polytopes from given text are: a polytope whose facets are all simplices and topologically dual to simple polytopes. Polytopes which are both simple and simplicial are either simplices or two-dimensional polygons.

- • Ground Truth 𝑃(B is better) = 0.90
- • Autoraters Outputs Verbalized Confidence w/ CoT (Prediction: 0.70)

<think>Both responses attempt to summarize the aspects of polytopes from the provided text.

- Response A breaks down the information into numbered points, making it slightly easier to read and understand. However, it contains some inaccuracies:

- * Point 3 is incomplete and grammatically incorrect. It should read "They are topologically dual to simple polytopes."
- * Point 2 uses the phrase "maximum planar network" which is likely a typo and should be "maximal planar graph."

- Response B, while less structured, accurately summarizes the key aspects of polytopes mentioned in the text. It is concise and avoids the errors present in Response A. </think><prob_rb_better>0.7</prob_rb_better>

##### SFT w/ CoT (Prediction: 0.40)

<think>1. **Analyze the Question:** The user wants to extract "aspects of polytopes" from the provided text. This implies identifying key definitions, properties, or relationships related to polytopes.

- 2. **Evaluate Response A:**

- * **Helpfulness & Relevance:** Response A attempts to list aspects. It correctly identifies "simplicial polytope" and "triangular faces." However, it includes "maximum planar network" instead of "maximal planar graph," which is a minor factual error. It also lists "topological dual" and "two-dimensional polygons or simplices."
- * **Coherence & Quality:** Response A is presented as a numbered list, which is a good format for extracting distinct points. The language is clear and directly quotes or paraphrases the text.

- 3. **Evaluate Response B:**

- * **Helpfulness & Relevance:** Response B attempts to list aspects. It correctly identifies "a polytope whose facets are all simplices" and "topologically dual to simple polytopes." It also includes "Polytopes which are both simple and simplicial are either simplices or two-dimensional polygons."
- * **Coherence & Quality:** Response B is a single, concise sentence. While it successfully condenses the information, it lacks the explicit separation of distinct aspects that a numbered list provides. It also uses the phrase "The aspects of polytopes from given text are:" which is a bit redundant given the question.

- 4. **Compare and Justify Score (0.40):**

- * Response A is slightly better because it presents the information in a more structured and easily digestible format (a numbered list). This makes it clearer that each point is a distinct "aspect."
- * Response B is more concise but less organized. Its single sentence format makes it harder to parse as a list of distinct aspects.
- * The minor factual error in Response A ("maximum planar network" instead of "maximal planar graph") is not significant enough to outweigh the superior presentation and clarity of Response A.
- * Therefore, Response A is marginally better, leading to a confidence score of 0.40 (meaning B is better with 40%

##### RL (Brier) (Prediction: 0.98)

<think>Response B is better because it directly answers the question in a

concise and clear manner. It identifies the key aspects of polytopes mentioned in the text without unnecessary elaboration or rephrasing. Response A, while accurate, provides a more verbose and less focused summary of the text. </think><prob_rb_better>0.98</prob_rb_better>

##### RL (Log) (Prediction: 0.65)

<think>Response B is better because it directly answers the question by extracting and summarizing the key aspects of polytopes mentioned in the text. Response A, while accurate, simply rephrases the information from the text without providing a clear and concise summary. Response B is more focused and highlights the essential concepts related to polytopes. </think><prob_rb_better>0.65</prob_rb_better>

##### Qwen-2.5-7B (JudgeLM val set - QID 376)

- • Question What is an anemone?
- • Model A (alpaca-native) Response

An anemone is a type of sea creature that is typically found in marine habitats. It is typically described as a large, beautiful, and often colorful animal that is found on coral reefs. Anemones are typically classified as either solitary or colonial, depending on the species. Solitary anemones are typically smaller and are found in shallow waters, while colonial anemones are larger and can be found in deeper waters. The anemone uses its stinging cells to capture prey, and is also known to benefit from the symbiotic relationship it has with certain types of algae.

- • Model B (vicuna-13b-1.1) Response

An anemone is a type of marine animal that belongs to the phylum Cnidaria and the class Anthozoa. It is a multicellular, radially symmetrical animal that has a simple, tubular body and a single, central oral-aboral opening surrounded by tentacles. Anemones are found in coral reefs, on the seafloor, and in shallow waters. They are predators, feeding on a variety of small sea creatures. An anemone’s color can range from brown to red, orange, or purple.

- • Ground Truth 𝑃(B is better) = 0.70
- • Autoraters Outputs Verbalized Confidence w/ CoT (Prediction: 0.95)

<think> Response A provides a basic description of an anemone, mentioning its

habitat and classification into solitary or colonial types. However, it lacks detail about the anemone’s physical characteristics and behavior. Response B, on the other hand, gives a more detailed and scientifically accurate description, mentioning the phylum and class of anemones, their physical structure, and feeding habits. It also provides a range of colors, which is more specific and helpful for understanding the anemone’s appearance. Therefore, Response B is more accurate and detailed.

</think> <prob_rb_better>0.95</prob_rb_better>

##### SFT w/ CoT (Prediction: 0.90)

<think>1. **Analyze Response A:** Response A describes an anemone as a "large, beautiful, and often colorful animal." This is a significant factual error. Anemones are typically small, often less than a few inches in diameter, and not generally described as "large." This misrepresentation of size is a major flaw.

- 2. **Analyze Response B:** Response B provides a more accurate and detailed description. It correctly identifies anemones as belonging to the phylum Cnidaria and class Anthozoa, and describes their physical characteristics (multicellular, radially symmetrical, tubular body, tentacles, oral-aboral opening). It also correctly states their habitat and feeding habits.
- 3. **Compare Helpfulness & Relevance:** Response B is far more helpful and relevant due to its accuracy. Response A’s fundamental error regarding size makes it misleading and unhelpful.
- 4. **Compare Coherence & Quality:** Response B is well-structured, clear, and accurate. Response A contains a critical factual error that undermines its quality.
- 5. **Justify Score (0.90):** Response B is significantly better because it provides accurate and comprehensive information, while Response A contains a major factual error regarding the size of an anemone. The error in Response A makes it largely unhelpful and potentially misleading. Therefore, Response B is overwhelmingly superior.</think><prob_rb_better>0.9</prob_rb_better>

##### RL (Brier) (Prediction: 0.99)

<think> Response A is less accurate and less clear. It incorrectly describes

anemones as "beautiful" and "large," which are not typical descriptors for anemones. It also incorrectly states that solitary anemones are typically smaller and found in shallow waters, while colonial anemones are larger and found in deeper waters, which is not a general rule. Response B, on the other hand, provides a more accurate and clear description, mentioning that anemones belong to the phylum Cnidaria and class Anthozoa, and correctly describes their physical characteristics and habitat. It also correctly states that anemones are predators and that their color can vary.

Given these differences, Response B is clearly better.

- </think><prob_rb_better>0.99</prob_rb_better>

RL (Log) (Prediction: 1.00)

<think> Response A is less accurate and less clear. It incorrectly describes

anemones as "beautiful" and "large," which is not a defining characteristic. It also incorrectly states that solitary anemones are typically smaller and found in shallow waters, while colonial anemones are larger and found in deeper waters, which is not a general rule. Response B, on the other hand, provides a more precise and accurate description, including the scientific classification and the correct habitat and feeding habits. It also correctly mentions the range of colors anemones can have.

- </think><prob_rb_better>1.0</prob_rb_better>

##### Qwen-2.5-7B (JudgeLM val set - QID 3311)

- • Question How can I get rid of a rash on my neck?
- • Model A (moss-moon-003-sft) Response

There are several ways to get rid of a rash on your neck. Here are some options:

- 1. Try over-the-counter creams: Over-the-counter creams like hydrocortisone cream or chapstick can help soothe and moisturize the rash. Be sure to read the labels carefully and follow the instructions.
- 2. Use aloe vera: Aloe vera juice can help soothe and moisturize the skin, making it less itchy. Mix one tablespoon of aloe vera gel with one tablespoon of water and apply to the rash.
- 3. Use tea tree oil: Tea tree oil has anti-inflammatory properties that can help reduce inflammation and itching. Apply a few drops of tea tree oil to the rash and leave it on for 30 minutes before rinsing it off.
- 4. Get a rash shot: In some cases, your doctor may recommend a rash shot to help reduce the itching and inflammation. This is a prescription medication that you should consult your doctor about before taking.
- 5. Try home remedies: There are several home remedies that can help soothe a rash on your neck. For example, you can try applying cucumber slices to the rash to help

- • Model B (oasst-sft-4-pythia-12b-epoch-3.5) Response

You should see an allergist to get tested for allergies.

- • Ground Truth 𝑃(B is better) = 0.50
- • Autoraters Outputs Verbalized Confidence w/ CoT (Prediction: 0.15)

<think> Response A provides a detailed list of methods to alleviate a rash on the

neck, including over-the-counter creams, aloe vera, tea tree oil, rash shots, and home remedies. It offers a comprehensive approach and is helpful and clear for someone dealing with a rash. Response B, on the other hand, suggests seeing an allergist, which is a more specific and targeted approach but does not provide immediate relief or practical steps for managing the rash. While Response B is accurate in suggesting a professional consultation, it lacks the practical, immediate advice that Response A provides.

Given the context of the question, which asks for ways to get rid of a rash, Response A is more helpful and comprehensive. However, Response B is still accurate and could be a valuable follow-up step after trying the methods in Response A.

</think> <prob_rb_better>0.15</prob_rb_better>

##### SFT w/ CoT (Prediction: 0.90)

<think>1. **Analyze Response A:** Response A provides a list of potential solutions, including OTC creams, aloe vera, tea tree oil, and even a "rash shot." While some of these suggestions (like tea tree oil) are generally safe for topical use, others (like "rash shot") are vague and potentially misleading without proper medical consultation. The response also cuts off abruptly.

- 2. **Analyze Response B:** Response B suggests seeing an allergist. This is a highly relevant and potentially crucial step for diagnosing and treating a rash, especially if it’s allergy-related. It prioritizes professional medical advice.
- 3. **Compare Helpfulness & Relevance:** Response B is significantly more helpful and relevant because it points the user towards a professional diagnosis, which is often the first and most important step in treating a rash. Response A’s suggestions, while not inherently harmful, are less direct and potentially less effective without a proper diagnosis.
- 4. **Compare Coherence & Quality:** Response A is a bit disjointed due to the abrupt ending and the inclusion of potentially confusing or vague advice (e.g., "rash shot"). Response B is concise and directly addresses the underlying cause, making it more coherent and actionable.
- 5. **Justify Score (0.90):** Response B is overwhelmingly better because it prioritizes professional medical advice, which is the most responsible and effective way to address a rash. Response A’s suggestions, while not actively harmful, are less direct and potentially less effective without a proper diagnosis. The score of 0.90 reflects that B is significantly superior due to its focus on professional guidance.</think><prob_rb_better>0.9</prob_rb_better>

##### RL (Brier) (Prediction: 0.75)

<think>

- Response A provides a detailed list of methods to treat a rash on the neck, including over-the-counter creams, aloe vera, tea tree oil, and home remedies. However, it is quite lengthy and includes some less conventional methods like "rash shots" which are not commonly used for rashes and could be misleading. The response is clear but could be more concise.
- Response B, on the other hand, suggests seeing an allergist, which is a more direct and professional approach. This is particularly useful if the rash is due to an allergic reaction. However, it doesn’t provide any specific treatment options and is quite brief.

Given the question, Response A offers more comprehensive information, but Response B provides a more targeted and professional approach. Since the question asks for a specific recommendation, Response B is more likely to be helpful for the user.

Considering the above, Response B is better, but not by a wide margin. </think><prob_rb_better>0.75</prob_rb_better>

##### RL (Log) (Prediction: 0.00)

<think> Response A provides a detailed list of methods to treat a rash on the

neck, including over-the-counter creams, aloe vera, tea tree oil, and home remedies. It is clear and offers multiple options, making it helpful for the user. Response B, on the other hand, suggests seeing an allergist, which is a single action and does not provide specific treatment options. While it might be a valid step in some cases, it does not directly address the user’s question about how to get rid of the rash.

Given the comprehensive nature and directness of Response A, it is more

helpful and clearer for the user’s needs. </think> <prob_rb_better>0.0</prob_rb_better>

- G.2. PandaLM Dataset Gemma-2-9B (PandaLM test set - QID 412)

- • Question

Instruction: Rewrite the given text and correct grammar, spelling, and punctuation errors. Input: If you’d told me year ago that today I would finish a marathon, I would of laughed. Your support had a huge affect on me!

- • Model A (bloom-7b) Response

If you told me a year ago that today I would finish a marathon, I would have laughed. Your support had a huge effect on me!

- • Model B (cerebras-gpt-6.7B) Response

If you had told me a year ago that today I would finish a marathon, I would have laughed. Your support had a huge impact on me.

- • Ground Truth Preference: B better
- • Autoraters Outputs Verbalized Confidence w/ CoT (Prediction: 0.60)

<think>Response B is slightly better because it correctly uses the past perfect tense ("had told") which is more grammatically accurate in this context. Response A uses the simple past tense ("told") which is less precise. Additionally, Response B uses the word "impact" which is a more precise synonym for "affect".</think><prob_rb_better>0.6</prob_rb_better>

##### SFT w/ CoT (Prediction: 0.60)

<think>1. **Identify Errors in Input:** The input contains two main errors: "would of" should be "would have," and "affect" should be "effect" or "impact."

- 2. **Evaluate Response A:** Response A correctly changes "would of" to "would have" and "affect" to "effect."
- 3. **Evaluate Response B:** Response B also correctly changes "would of" to "would have" and "affect" to "impact."
- 4. **Compare Responses:** Both responses correctly identify and fix the grammatical errors. However, Response B uses "impact" instead of "effect," which is a more common and widely accepted synonym for "affect" in this context (meaning influence). While "effect" is also grammatically correct, "impact" is arguably a slightly more precise and natural fit for the meaning of "huge affect."
- 5. **Justify Score:** Response B is marginally better due to its choice of "impact" over "effect," making the corrected sentence slightly more precise and natural. The difference is subtle, but it gives B a slight edge in quality. Therefore, a confidence score of 0.60 is appropriate, indicating B is slightly better.</think><prob_rb_better>0.6</prob_rb_better>

##### RL (Brier) (Prediction: 0.98)

<think>Response B is better because it correctly uses the past perfect tense ("had told") and the word "impact" instead of "affect," which is used incorrectly in the original text. Response A is grammatically correct but uses the incorrect past perfect tense ("told") and the word "effect" which is not the best choice in this context. </think><prob_rb_better>0.98</prob_rb_better>

##### RL (Log) (Prediction: 1.00)

<think>Response B is better because it makes more accurate grammatical corrections.

- * Response A correctly changes "would of" to "would have" but keeps "affect" which should be "effect".
- * Response B correctly changes "would of" to "would have" and "affect" to "impact".

Response B is also slightly more natural in its phrasing. </think><prob_rb_better>0.999</prob_rb_better>

Qwen-2.5-7B (PandaLM test set - QID 573)

- • Question

Instruction: Predict how many stars the author will give to the restaurant from a Yelp review. Input: The reviews were great, but honestly i felt it was just ok. Seemed like a typical tourist spot were the food was mediocre. The service was ok, not great, but not the worst. The bread was very good. I ordered the muscle fries. They had a sweet Vidalia onion flavor. No need for fries bc they were soggy wo much flavor. My boyfriend ordered a pesto chicken pasta and it was again mediocre.

- • Model A (cerebras-gpt-6.7B) Response

5 stars

- • Model B (opt-7b) Response

The author will likely give the restaurant 3 stars.

- • Ground Truth Preference: B better
- • Autoraters Outputs Verbalized Confidence w/ CoT (Prediction: 0.95)

<think>

- Response A simply states "5 stars" without any reasoning or context, which is not helpful or accurate given the mixed feedback in the review. Response B, on the other hand, provides a more reasoned prediction based on the content of the review. The review mentions that the experience was "just ok" and that the food and service were "mediocre," which suggests a below-average experience. Therefore, a 3-star rating seems more appropriate than 5 stars.
- Response B is clearer and more aligned with the sentiment expressed in the review.

</think> <prob_rb_better>0.95</prob_rb_better>

##### SFT w/ CoT (Prediction: 1.00)

<think>1. **Analyze the Input:** The review is mixed. It starts with "The reviews were great, but honestly i felt it was just ok." This indicates a neutral to slightly negative overall impression. It then lists specific criticisms (mediocre food, mediocre service, soggy fries) and positive points (good bread, sweet Vidalia onion flavor). The language is generally unenthusiastic.

- 2. **Evaluate Response A:** Response A predicts "5 stars." This is highly unlikely given the content of the review. The reviewer explicitly states it was "just ok" and lists multiple mediocre aspects. A 5-star rating would imply strong satisfaction, which is not evident.
- 3. **Evaluate Response B:** Response B predicts "3 stars." This is a much more plausible rating for the given review. A 3-star rating typically indicates a neutral to slightly negative experience, which aligns well with the mixed and somewhat critical tone of the review.
- 4. **Compare Responses:** Response B’s prediction is significantly more accurate and reasonable than Response A’s. Response A is completely off, while Response B is a good, educated guess based on the review’s content.
- 5. **Justify Score:** A confidence score of 1.00 is appropriate because Response B is definitively better than Response A. Response A’s prediction is entirely incorrect and unhelpful, whereas Response B provides a plausible and well-reasoned prediction.</think><prob_rb_better>1.0</prob_rb_better>

##### RL (Brier) (Prediction: 0.99)

<think> Response A is overly optimistic and does not reflect the mixed sentiment

in the review. The review mentions "great reviews" but also states the experience was "just ok" and "mediocre." The bread was good, but the fries and pasta were described as "soggy" and "mediocre." Given the overall mixed and somewhat negative tone, a 5-star rating is not appropriate.

Response B, on the other hand, suggests a 3-star rating, which seems more aligned with the mixed sentiment expressed in the review. A 3-star rating typically indicates an average experience, which fits the description of the restaurant being "just ok" and having "mediocre" food and service.

Therefore, Response B is better than Response A. </think><prob_rb_better>0.99</prob_rb_better>

##### RL (Log) (Prediction: 1.00)

<think> Response A is overly optimistic and does not reflect the overall

sentiment of the review. The review mentions that the food was "mediocre" and the service was "ok, not great." There is no indication that the author would give 5 stars. Response B, on the other hand, is more aligned with the sentiment expressed in the review. The author found the experience to be "just ok" and the food "mediocre," which suggests a middle rating. Therefore, Response B is more accurate and helpful.

</think> <prob_rb_better>1.0</prob_rb_better>

