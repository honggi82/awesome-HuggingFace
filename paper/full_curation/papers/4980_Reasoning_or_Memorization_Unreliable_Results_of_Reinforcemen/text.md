# arXiv:2507.10532v3[cs.LG]17Dec2025

## Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination

### Mingqi Wu1*, Zhihao Zhang12*, Qiaole Dong1*, Zhiheng Xi1, Jun Zhao1, Senjie Jin1, Xiaoran Fan1, Yuhao Zhou1, Huijie Lv12, Ming Zhang1, Yanwei Fu1, Qin Liu4, Songyang Zhang2, Qi Zhang123†

1Fudan University 2Shanghai Artificial Intelligence Laboratory 3Shanghai Key Lab of Intelligent Information Processing 4University of California, Davis qz@fudan.edu.cn, qinli@ucdavis.edu

##### Abstract

Reasoning in large language models has long been a central research focus, and recent studies employing reinforcement learning (RL) have introduced diverse methods that yield substantial performance gains with minimal or even no external supervision. Surprisingly, some studies even suggest that random or incorrect reward signals can enhance performance. However, these breakthroughs are predominantly observed for the mathematically strong Qwen2.5 series on benchmarks such as MATH-500, AMC, and AIME, and seldom transfer to models like Llama, which warrants a more in-depth investigation. In this work, our empirical analysis reveals that pre-training on massive web-scale corpora leaves Qwen2.5 susceptible to data contamination in widely used benchmarks. Consequently, conclusions derived from contaminated benchmarks on Qwen2.5 series may be unreliable. To obtain trustworthy evaluation results, we introduce a generator that creates fully clean arithmetic problems of arbitrary length and difficulty, dubbed RandomCalculation. Using this leakagefree dataset, we show that only accurate reward signals yield steady improvements that surpass the base model’s performance boundary in mathematical reasoning, whereas random or incorrect rewards do not. Moreover, we conduct more finegrained analyses to elucidate the factors underlying the different performance observed on the MATH-500 and RandomCalculation benchmarks. Consequently, we recommend that future studies evaluate models on uncontaminated benchmarks and, when feasible, test various model series to ensure trustworthy conclusions about RL and related methods.

Code: https://github.com/wumingqi/LLM-Math-Evaluation

### 1 Introduction

In recent years, advances in reinforcement learning (RL) have markedly strengthened the reasoning abilities of large language models (LLMs). Flagship systems, including OpenAI’s o1 (OpenAI 2024b, 2025, 2024a), DeepSeekR1 (DeepSeek-AI et al. 2025), and QwQ (Qwen Team 2024,

*These authors contributed equally. †Corresponding author.

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

2025), already match or exceed human-level accuracy on a variety of challenging benchmarks. Among open-source contenders that excel in these mathematical benchmarks, the Qwen family (Yang et al. 2024a,b, 2025), ranging from 0.5B to 72B and pre-trained on up to 36T high-quality tokens, produces state-of-the-art results in language understanding, mathematics, programming, and preference alignment.

Within this landscape, mathematical reasoning emerges as a particularly discriminative test bed because it demands precise symbolic manipulation and multi-step logical deduction. Standard suites such as MATH-500 (Hendrycks et al. 2021), AIME (Li et al. 2024), AMC (Li et al. 2024), and Minerva Math (Lewkowycz et al. 2022) require models to parse natural-language problem statements, uncover the latent mathematical structure, and generate exact numeric answers. Recent work has further enhanced this capability by reinforcement learning with verifiable rewards (RLVR) (DeepSeek-AI et al. 2025): rule-based reward that returns 1 when the predicted answer equals ground truth and 0 otherwise. Because the reward is computed analytically, RLVR removes the need for a separate learned reward model, lowering computational cost while providing a precise training signal, especially attractive for domains like mathematics, where solutions are unambiguous.

Although RL nominally depends on accurate reward signals to guide training, recent studies (Zuo et al. 2025; Shao et al. 2025) find that even random or incorrect rewards can improve Qwen’s performance on standard math benchmarks, while the same procedures offer little or no benefit to Llama3.1-8B (Dubey et al. 2024). To understand why these seemingly problem-agnostic rewards help Qwen but not Llama, we undertake a systematic comparison of two model families under identical training protocols. We consider two working hypotheses to explain this phenomenon. (i) Data Contamination: Considering Qwen2.5 is pre-trained on massive web-scale corpora from the Internet, including GitHub repositories that store benchmark problems alongside their official solutions. If segments of the evaluation benchmarks leaked into the pre-training corpus, spurious rewards could cue the model to retrieve memorized answers

Example from MATH-500 Problem: For how many positive integers n > 1 is it true that 224 is a perfect nth power? Answer:

|7|
|---|

Partial Prompt Prompt: For how many positive integers n > 1 is

Qwen2.5-7B Output it true that 224 is a perfect nth power? To determine for how many positive integers n > 1 the number 224 is a perfect n-th power... Let’s count these values. There are 7 such values... The final answer is

###### Llama3.1-8B Output

the set of positive divisors of n (including 1 and n) less than 50 ?

(A) 15 (B) 16 (C) 17 ... The number of positive divisors of n is given by... The number of factors of each of these numbers is... The answer is

|(B) 16|
|---|

|7|
|---|

.

Figure 1: Memorization Example of Qwen2.5 on MATH500. In this case, the first 40% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration (see Table 1). The Qwen2.5-7B model accurately reproduces the original question verbatim and, moreover, generates a complete and precise chain of reasoning that yields the correct answer. In contrast, Llama3.1-8B produces an incorrect completion and ultimately arrives at an incorrect answer.

rather than acquire new reasoning skills. (ii) Strong Math Capacity: Qwen’s pre-training endows it with better mathematical capacity than Llama, so even noisy policy-gradient updates appear to help on MATH-500. However, if strong capacity is the real driver, the same spurious rewards should still work on a clean benchmark. Distinguishing between these possibilities requires both a leakage audit and a rigorously out-of-distribution RLVR evaluation.

To assess the extent of potential data contamination in popular mathematical benchmarks, we propose two metrics: partial-prompt completion rate (can the model reconstruct the tail of a problem?) and partial-prompt answer accuracy (can the model give the correct answer with an incomplete problem?). As shown in Fig. 1, Qwen can indeed complete the problem accurately and provide the correct answer, whereas Llama fails. Furthermore, prompting with the first 60% of each MATH-500 problem, Qwen2.5-Math-7B regenerates the remaining 40% with a 54.60% exact-match rate and answers 53.6% of these incomplete problems correctly. In contrast, Llama3.1-8B scores 3.8% and 2.4% on both metrics. Crucially, on the newly released LiveMathBench (version 202505) (Liu et al. 2024), Qwen’s completion rate drops sharply to 0.0%, consistent with Llama’s 0.0%. Its partial-prompt answer accuracy also falls to just 2.0%, comparable to Llama’s 1.0%. These results confirm that Qwen’s pre-training corpus suffers from test data con-

tamination. So, results derived from MATH-500 and similar datasets for Qwen should be interpreted with caution.

Based on this, we attribute that data contamination is the main factor behind the ‘magical’ success of spurious rewards on Qwen. To test this claim, we first create a clean benchmark (i.e., RandomCalculation, example shown in Fig. 2): We use automatic generator to construct arithmetic expressions of arbitrary length with random operands and operators, guaranteeing that every instance post-dates the public release of Qwen. Zero-shot evaluation on this benchmark shows no memorization: the accuracy of Qwen2.5 declines monotonically with the number of computation steps. To isolate the effect of rewards, we next trained Qwen2.5Math-7B under the standard RLVR protocol on two subsets. The outcome is unambiguous: Correct rewards deliver consistent performance gains, surpassing the model’s performance ceiling. In contrast, random rewards make training highly unstable, yielding no reliable improvement, while inverse rewards rapidly erode the model’s mathematicalreasoning ability. These results rule out the ‘Strong Math Capacity’ hypothesis and directly imply ‘Data Contamination’: once leakage is removed, spurious gains evaporate.

To further test this hypothesis, we measure the similarity between model outputs before and after RL. In MATH500, the pre- and post-RL responses exhibited substantially higher lexical overlap than in RandomCalculation, indicating that Qwen inadvertently retrieves its memory and answers during RL with spurious rewards. A more detailed token-level analysis also supports this: the token-level KL divergence between the pre- and post-RL models is significantly lower for MATH-500. These results strengthen our hypothesis that data contamination leads to successful RL through spurious rewards on Qwen series. Based on these findings, we recommend that future work should test on uncontaminated benchmarks or more diverse model series to draw trustworthy conclusions about RL-related methods.

Contributions of our work can be summarized as follows:

- • We conduct a systematic leakage audit of math benchmarks with two novel metrics and demonstrate that Qwen suffers from data contamination on public benchmarks.
- • We propose an automatic generator that creates arbitrarily long arithmetic expressions. Zero-shot evaluation on this dataset exposes the absence of memorization, enabling fair assessment of RL methods.
- • Using this clean dataset, we conduct RL experiments and demonstrate that only correct reward yields stable improvement, whereas spurious rewards provide no benefit.
- • We reveal that spurious rewards solely enable the retrieval of memory from pre-training, leading to spurious performance improvement on MATH-500.

### 2 Related Works

#### 2.1 RL on Qwen2.5 for Mathematical Reasoning

A growing body of work investigates how reinforcement learning (RL) can amplify the mathematical-reasoning capacity of the open-source Qwen2.5 family. Early studies use verifiable rewards that score an answer as 1 / 0 by exact numerical agreement. Test-time RL (Zuo et al. 2025) applies

10-Step Calculation Problem: Evaluate this LaTeX numerical expression step-bystep and give the final value within \boxed{}:





732 · (62 − 10)

41 6

94 2

12 7

·

+

+

 

 

65 9 +47 49

7 ·81 622

|6490.42220471333|
|---|

###### Answer:

Figure 2: Examples of RandomCalculation dataset.

this signal on-the-fly during inference and yields sizeable gains on MATH-500 and AIME2024. Subsequent efforts pursue extreme data efficiency: few labeled (Wang et al. 2025a; Li et al. 2025) or even unlabeled examples (Gao et al. 2025; Zhao et al. 2025a) can suffice to boost performance. A parallel research replaces external supervision with intrinsic signals derived from the model itself: Prabhudesai et al. (2025); Agarwal et al. (2025) reward low-entropy output distributions, while Zhao et al. (2025b); Shafayat et al. (2025) rely on self-consistency or self-certainty as feedback. These approaches report large jumps on Qwen2.5-Math-7B, occasionally matching or surpassing stronger supervised baseline. Other variants explore noisy (Lv et al. 2025) or even random rewards (Shao et al. 2025) for Qwen2.5. However, these methods fails to transfer to Llama or OLMo (OLMo et al. 2025), suggesting model-specific idiosyncrasies.

#### 2.2 Factors Influencing Performance on Math

The choice of pretraining corpora plays a crucial role in shaping the reasoning abilities of LLMs, particularly in mathematical domains. Several math-specific datasets (Paster et al. 2024; Han et al. 2024; Wang et al. 2024; Allal et al. 2025) have been proposed and shown to significantly enhance performance on relevant benchmarks. Further, Wang et al. (2025b) finds that mid-training Llama models on high-quality mathematical corpora substantially improve their capacity, both at the base level and after reinforcement learning. However, evaluation of math capability can be misleading due to potential test data contamination. Xu et al. (2024) observed that certain widely-used benchmarks may be partially included in the pretraining corpus of LLMs, including early versions of Qwen. On the other hand, Liu et al. (2025) found that omitting dialogue-style prompting can lead to improved mathematical reasoning. These observations underscore the presence of multiple confounding factors in evaluating model performance, motivating the need for rigorous and systematic analysis.

3 Experimental Setup

#### 3.1 Model Selection

Prior researches on mathematical reasoning with LLMs focus predominantly on the Qwen-2.5 (Yang et al. 2024a,b).

Accordingly, we center our study on four representative checkpoints from this series: Qwen2.5-7B, Qwen2.57B-Instruct, Qwen2.5-Math-7B, and Qwen2.5-Math-7BInstruct. For a controlled comparison, we also evaluate Llama3.1-8B and Llama3.1-8B-Instruct (Dubey et al. 2024), which possess comparable parameter counts and thus help isolate model-specific differences in behavior.

#### 3.2 Evaluation of Memorization Capability

We assess the model’s memorization of benchmark data, i.e., data contamination, using two metrics: Partial-Prompt Completion Rate and Partial-Prompt Answer Accuracy.

Specifically, we prompt the model to complete the remaining parts of a problem based on partial prefixes. To evaluate its performance, we use the Partial-Prompt Completion Rate measured by ROUGE-L (Lin 2004), which calculates the overlap of the longest common subsequence between the generated and reference text, capturing fluency and sentence-level structure. Additionally, we utilize Exact Match (EM) accuracy, which checks if the model’s output exactly matches the reference. The final EM score is the average across all instances. Higher EM indicates a greater proportion of partial-prompts that model can recall exactly.

Besides, for each question, we supply the model with only a truncated prompt (e.g., the first 60% of the original problem) and allow it to generate an unconstrained continuation. After generation, we check whether the completion contains the ground-truth answer; if so, the instance is scored as correct. Partial-Prompt Answer Accuracy is defined as the fraction of prompts for which the model’s continuation embeds the correct answer. A high accuracy indicates that the model frequently ‘recovers’ the answer even from a partial problem, which in turn may signal data contamination.

#### 3.3 RLVR-Based Evaluation

Group Relative Policy Optimization (GRPO) (Shao et al. 2024) is adopted as our RLVR algorithm. Formally, for each question q, GRPO samples a group of outputs {o1,··· ,oG} from the old policy πθ

and then optimizes the policy model by maximizing the following objective:

old

|oi|

G

1 G

1 |oi|

min ri,tAˆi,t,

J GRPO(θ) = Eq,{o

i}Gi=1

t=1

i=1

clip(ri,t,1 − ϵ,1 + ϵ)Aˆi,t − βDKL [πθ||πref] ,

(1) where ri,t = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t), ϵ and β are hyperparameters, and Aˆi,t is the advantage calculated based on the relative rewards of the outputs inside each group only.

Spurious Reward. Following Shao et al. (2025), we consider following spurious reward types for RLVR:

- • Random: assigns 1 with probability γ and 0 otherwise (γ = 0.5 in our experiments).
- • Inverted: flips the correct signal, i.e., 1−correct, so that correct solutions receive 0 and incorrect ones 1.

Configuration # Sample Temp. Top-P Top-K

Correct

Random

Mv-Incorrect

Greedy (w/o Template)

Pass@16

Greedy (w/o Template) 1 1.0 1.0 1 Avg@16 (w/o template) 16 0.7 0.8 20 Greedy (w/ Template) 1 1.0 1.0 1 Avg@16 (w/ template) 16 0.7 0.8 20

Qwen2.5-Math-7B

Qwen2.5-Math-7B-Instruct

Llama3.1-8B-Instruct

1.0

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
|290<br><br>| | |
|---|---|
| | |
| | |
<br><br>0.81<br><br>0.85| | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.8

Accuracy

Accuracy

Accuracy

0.6

0.4

Table 1: The sampling parameters used under different generation configurations. Greedy sampling is performed using the default model.generate(...) function, while random sampling is implemented using vLLM. w/ Template indicates the usage of official chat template.

0.2

0.0

0 100 200 300 Training Step

0 100 200 300 Training Step

0 100 200 300 Training Step

Figure 3: Accuracy on the MATH-500 for Qwen2.5-Math-

- 7B, Qwen2.5-Math-7B-Instruct, and Llama3.1-8B-Instruct trained with RLVR under various reward signals. Greedy and pass@16 scores are reported without template.

Greedy (w/o Template)

81.8

83.6

-1.4

+0.2

Greedy (w/ Template)

74.9

72.8

-1.2

72.2

-0.6

Avg@16 (w/o Template)

68.5

67.6 66.2

Avg@16 (w/ Template)

MATH-500Acc.

-21.6

-19.7

43.6 39.6

-27.6

-28.0

-4.8 -1.4

• Mv-incorrect: uses the majority-voted incorrect labels from the model, assigning a reward of 1 when the model output matches an incorrect label, and 0 otherwise.

Qwen2.5-Math-7B-Instruct

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

Qwen2.5-Math-7B

Qwen2.5-7B

### 4 Results & Analysis

#### 4.1 Spurious Rewards on MATH-500

Figure 4: Accuracy (%) of Qwen and Llama models on the MATH-500 dataset under different generation configurations, using original questions as prompts. More detailed results can be found in Tab. 4 of Appendix.

Following the work of Shao et al. (2025), we replicate the performance of Qwen2.5-Math-7B and Llama3.1-8BInstruct on the MATH-500 benchmark under various reward signal configurations, using the same experimental setup. The accuracy curves of MATH-500 are shown in Fig. 3. Interestingly, the results demonstrate that while random rewards and mv-incorrect rewards noticeably boost accuracy for Qwen2.5-Math-7B, they have little or even adverse impacts on the performance of Llama3.1-8B-Instruct. Additionally, we apply the same RLVR procedure to Qwen2.5Math-7B-Instruct and discover that the resulting gains are marginal when compared with those of Qwen2.5-Math-7B, indicating that the two Qwen variants exhibit differential sensitivity to RLVR under spurious rewards.

gains’ of Qwen2.5-Math-7B largely reflect adaptation to the template format and merely converge to the Greedy (w/o Template) baseline, indicative of memory recall rather than genuine mathematical generalization. However, spurious rewards, e.g., random and mv-incorrect, still boost the accuracy of Qwen base and maintain the performance of Qwen instruct, while degrading Llama eventually. This is the point we need to further explore in the following sections.

Considering the base and instruct variants of Qwen are trained under different paradigms: the former is pretrained as a general language model without exposure to any dialogue-specific templates, whereas the latter undergoes an additional instruction-tuning stage on data wrapped in a fixed dialogue template. This mismatch creates a training–testing gap for the base model at the start point of RLVR, and its initial accuracy is therefore likely underestimated. Consequently, to obtain a fair estimate of each model’s starting point, we next measure performance under four decoding configurations as shown in Tab. 1. The corresponding results are summarized in Fig. 4. Surprisingly, we discover that applying the official chat template substantially degrades performance for the Qwen base model: both Qwen2.5-7B and Qwen2.5-Math-7B suffer pronounced drops once the template is enabled.

#### 4.2 Analysis of Memorization Capability

Considering the Qwen series is trained on massive web-scale corpora, we hypothesize that its divergent RLVR behavior from Llama is because the evaluation set MATH-500 may be inadvertently contaminated in Qwen’s large-scale training data, which is hard to eliminate completely. To verify our hypothesis, we probe memorization on several widely used mathematical-reasoning benchmarks. Concretely, we truncate the original questions at 40%, 60%, and 80% of their lengths and feed these partial questions as prompts into the model, and then evaluate the model’s partial-prompt completion rate by computing ROUGE and EM scores between the generated completion and ground-truth continuations. In addition, we evaluate the model’s partial-prompt answer accuracy by checking if the continuation contains the correct answer, across both partial and full question settings.

Building on this observation, we report two additional initial accuracy for reference in Fig. 3: (i) Greedy (w/o Template) corresponds to the best performance of the initial model, and (ii) pass@16, adopted from Yue et al. (2025), serves as a plausible performance upper bound for the initial model. Viewed against these baselines, the seeming ‘RL

The detailed results are presented in Tab. 2, revealing strong signs of data contamination in the Qwen2.5 series models when evaluated on commonly used benchmarks, such as MATH-500, AMC, and AIME2024. For instance, when only the first 60% of the questions are provided,

80%-Problem 60%-Problem 40%-Problem ROUGE-L EM ROUGE-L EM ROUGE-L EM

Model Dataset Size

MATH-500 500 81.25 65.80 78.06 54.60 69.01 39.20 AMC 83 77.38 55.42 70.25 42.17 75.17 36.14

- AIME2024 30 74.04 56.67 55.31 20.00 57.72 16.67

- AIME2025 30 54.71 16.67 34.88 0.00 27.43 0.00 MinervaMath 272 36.08 2.94 31.22 0.37 29.35 0.00 LiveMathBench 100 42.76 5.00 32.78 0.00 29.97 0.00

Qwen2.5-Math-7B

MATH-500 500 66.42 40.20 60.98 21.20 50.36 8.20 AMC 83 73.24 49.40 64.42 33.73 63.79 28.92

- AIME2024 30 59.80 30.00 48.69 13.33 44.65 10.00

- AIME2025 30 54.61 10.00 37.59 0.00 30.30 0.00 MinervaMath 272 35.24 2.94 32.35 0.37 27.89 0.00 LiveMathBench 100 41.15 4.00 32.74 0.00 27.95 0.00

Qwen2.5-7B

MATH-500 500 48.33 17.80 40.55 3.80 32.07 0.60 AMC 83 44.54 4.82 30.62 0.00 27.10 0.00

- AIME2024 30 50.50 13.33 30.80 0.00 26.08 0.00

- AIME2025 30 47.04 10.00 33.49 0.00 25.20 0.00 MinervaMath 272 36.24 2.21 29.52 0.00 27.11 0.00 LiveMathBench 100 35.55 5.00 31.93 0.00 26.88 0.00

Llama3.1-8B

Table 2: Accuracy (EM) and ROUGE-L on several datasets (lower scores in gray) under different prompt prefix ratios with Greedy (w/o Template) configuration. Suspicious accuracy is highlighted in bold.

Qwen2.5-Math-7B is able to accurately reconstruct more than half of the remaining problems on MATH-500. Even when just 40% proportion of the questions are shown, the model still manages to recover 39.2% of the problems on MATH-500. Similar patterns are observed on AMC and AIME2024. These results indicate that the evaluation benchmarks for Qwen2.5 may suffer from data contamination. Although pre-training on massive web-scale corpora brings strong capacity on mathematical reasoning, e.g., superior performance on recently introduced mathematical tasks like LiveMathBench and AIME2025, those large-scale corpora also include publicly available benchmark problems inevitably, leading to less convincing results of old benchmarks, while removing such instances during large-scale crawling is notoriously difficult.

Meanwhile, we summarize the answer accuracy of the model under different ratios of prefix in Fig. 5. The Qwen2.5 models achieve remarkably high accuracy on MATH-500 even with partial questions. For instance, with 80% proportion of the questions, Qwen2.5-Math-7B reaches an accuracy of 63.8% on MATH-500. Even with only 40% proportion of the questions, the model still achieves an accuracy of 41.2%. Because our evaluation matches only the final numeric answer, the model can sometimes output the correct value by accident, which explains the anomalous accuracy of Llama on AIME2025 (solving exactly one problem with a faulty reasoning process). Besides, we also inspect several questions that Qwen solves correctly, as shown in Fig. 11 to 15 in Appendix. We find that the responses of Qwen contain coherent reasoning chains and even syntactically valid Python code, which, however, is not executed. The emergence of such structured solutions indicates that the train-

Qwen2.5-Math-7B Qwen2.5-7B Llama3.1-8B

MATH-500

###### AMC

###### AIME2024

0.8

0.8

0.8

0.6

0.6

0.6

Accuracy

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

40 60 80 100

40 60 80 100

40 60 80 100

Prompt Proportion (%)

Prompt Proportion (%)

Prompt Proportion (%)

AIME2025

MinervaMath

LiveMathBench

0.20

0.20

0.20

0.15

0.15

0.15

Accuracy

0.10

0.10

0.10

0.05

0.05

0.05

0.00

0.00

0.00

40 60 80 100

40 60 80 100

40 60 80 100

Prompt Proportion (%)

Prompt Proportion (%)

Prompt Proportion (%)

Figure 5: Accuracy (%) of LLMs on various math datasets under Greedy (w/o Template) configuration. More detailed experimental results can be found in Tab. 5 of Appendix.

ing corpora may have included publicly available resources where benchmark problems are accompanied by detailed solutions. Besides, we provide more results for Qwen2.5 and Qwen3 under various sampling configurations in Appendix. The memorization of Qwen2.5 on LiveCodeBench (Jain et al. 2024) and Qwen3 series on math is also analyzed in Appendix, showing similar results as Tab. 2.

#### 4.3 Spurious Rewards on RandomCalculation

To further support our hypothesis that the anomalous performance surge of the Qwen2.5-Math-7B on the MATH-500 benchmark is primarily caused by data contamination rather than the model’s intrinsic mathematical reasoning ability, we replicate this experiment on a newly constructed dataset that

Greedy (w/o Template)

Greedy (w/ Template)

Avg@16 (w/o Template)

Avg@16 (w/ Template)

Qwen2.5-Math-7B

Qwen2.5-Math-7B-Instruct

1.0

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Accuracy

Accuracy

0.5

0.5

0.0

0.0

0 5 10 15 20 Calculation Step

0 5 10 15 20 Calculation Step

Qwen2.5-7B

Qwen2.5-7B-Instruct

1.0

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Accuracy

Accuracy

0.5

0.5

0.0

0.0

0 5 10 15 20 Calculation Step

0 5 10 15 20 Calculation Step

Figure 6: Performance of the Qwen2.5 series models on the RandomCalculation datasets under different configurations.

the model has never encountered before. We hypothesize that for math problems free from contamination, the model’s reasoning ability still requires properly aligned reward signals to yield meaningful performance improvements.

Dataset Construction of RandomCalculation. To obtain an uncontaminated evaluation benchmark, we employ Algorithm 1 (shown in Appendix) to construct a suite of challenging yet verifiable datasets. These datasets are composed of expressions built from basic numerical elements, including integers from 0 to 100, as well as fractions, squares, and cubes derived from them. Using these components, we randomly generate mathematical expressions that involve between 1 and 20 steps, using the four fundamental arithmetic operations: addition, subtraction, multiplication, and division. To construct the final datasets, we append a standardized problem prefix to each generated expression, resulting in 20 sub-datasets, each containing 1,000 unique problems. We refer to this suite of datasets as RandomCalculation. Examples from the datasets can be found in Fig. 2.

Zero-shot Performance on RandomCalculation. We first test the zero-shot performance of the Qwen2.5 series on the RandomCalculation dataset in Fig. 6. We find that when using chat templates, the models’ performance degrades gradually as the number of computation steps increases, leaving ample room for improvement in multi-step calculation problems. When chat templates are removed, the reasoning performance peaks on problems of three computation steps, and then gradually declines.

Correct Reward Function for RandomCalculation. The ground-truth answers to our randomly generated arithmetic problems often contain high-precision decimals. When using the standard RLVR framework, which only provides binary feedback, the model rarely receives positive reinforcement, making training unstable and prone to divergence. To overcome this, we design a continuous reward function that ranges from 0 to 1 and penalizes both absolute and relative errors between the model’s prediction and the reference

Correct

Random Trial 1

Random Trial 2

Mv-Incorrect

Inverted

Max@16

Greedy (w/o Template)

Qwen2.5-Math-7B 5 Steps

Qwen2.5-Math-7B 10 Steps

Llama3.1-8B-Instruct 10 Steps

1.0

1.0

1.0

Reward

Reward

Reward

0.5

0.5

0.5

0.0

0.0

0.0

0 100 200 300 Training Step

0 100 200 300 Training Step

0 100 200 300 Training Step

Figure 7: Reward of Qwen2.5-Math-7B and Llama3.1-8BInstruct on RandomCalculation. Results are presented for datasets with 5-step and 10-step calculations.

answer. This richer feedback helps stabilize reinforcement learning. Let a be the model output, b be the reference answer, and ϵ = 10−6 be a small constant for numerical stability. The reward r is computed as:

−0.5 · min |a − b| |b| + ϵ

###### r = 1 − 0.5 · min(|a − b|, 1)

, 1

absolute distance

relative distance

(2)

RLVR on RandomCalculation. We also perform RLVR training on Qwen2.5-Math-7B using the RandomCalculation datasets. Specifically, experiments are conducted on two sub-datasets comprising 5-step and 10-step calculation problems. Each dataset contains 1,000 problems, with 700 used for training and the remaining 300 reserved for validation. As shown in Fig. 7, the performance improves steadily throughout training under correct rewards. However, training becomes unstable and inconsistent with random or incorrect rewards. Under inverted rewards, the performance collapses rapidly. These findings suggest that for problems not leaked during pretraining, only correct reward signals can effectively guide the model toward improved performance. For comparison, we also evaluate Llama3.1-8B-Instruct and observe similar findings.

Qwen v.s. Llama on Clean Benchmark. Considering obtaining a reward of 1 on a RandomCalculation instance is virtually impossible, we report Max@16, the highest reward among 16 samples of initial model, in Fig. 7. We observe that on RandomCalculation datasets, Qwen2.5-Math-7B can surpass Max@16 when provided with correct reward signals. This finding indicates that reward-aligned RLVR effectively transfers the high-accuracy single-step arithmetic skills (as shown in Fig. 6) to more complex multi-step calculations, as illustrated by one reasoning trace in Fig. 16 of Appendix. In contrast, under the incorrect and random reward configurations, the Qwen model either maintains its base performance or exhibits only marginal and unstable improvements, due to the learning of format as explained in Shao et al. (2025). This improvements gradually disappear when calculation steps increased from 5 to 10. This further highlights the critical role of correct and well-

aligned reward signals in enhancing model performance on uncontaminated and difficult datasets. Notably, Llama3.1-

- 8B-Instruct fails to surpass the Max@16 even when trained with correct reward signals, and its accuracy falls below the greedy-decoding baseline when exposed to spurious signals. This discrepancy indirectly suggests that Qwen2.5 exhibits stronger mathematical capabilities than Llama3.1 before mid-training. However, such inherent strength is not the root cause of its performance boosting under spurious reward on contaminated datasets.

#### 4.4 More Evidence for Memorization

Here, we provide more detailed analyses of Qwen’s sudden performance gains on MATH-500 under random reward. Let

min ri,tAˆi,t,clip(ri,t,1 − ϵ,1 + ϵ)Aˆi,t ,

JCLIP = EAˆ

i,t

where Aˆi,t is a random variable under the setup of random reward. Referring to Appendix B of Shao et al. (2025), the gradient of the clipped policy has the following format:

∇θJCLIP = ∇θri,t · G(ri,t), (3)

 

µ, ri,t < 1 − ϵ, 0, 1 − ϵ ≤ ri,t ≤ 1 + ϵ, −µ, ri,t > 1 + ϵ,

(4)

G(ri,t) =



where µ > 0 is a positive coefficient, ri,t = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t). Memory Retrieval due to Exploitation Bias. Assume a high-probability token with πold = 0.85 and ϵ = 0.20, the upper clipping boundary is 1.02 for πθ, which exceeds the probability ceiling of 1.0 and therefore is never reached. Consequently, the gradient is non-negative for this token, leading to a net positive gradient bias on the policy model, In general, for high-probability token, we have ∇θJCLIP(θ) ∝ ∇θri,t, due to G(ri,t) ≥ 0 hold almost surely. So high-probability tokens continue to be upweighted without penalty. For MATH-500, correct answers typically have a high probability due to data contamination in the initial model (results shown in Fig.9 of Appendix), except for the low-probability answer format. Therefore, GRPO with random reward can retrieve these answers after learning format and leads to sharp accuracy jump in Fig 3.

On the other hand, assume another token with pre-update

likelihood πold = 0.5 (which is a typical value for our 10step RandomCalculation as shown in Fig. 9 of Appendix).

The corresponding clipping boundary is [0.4,0.6] for πθ. Gradient update with random reward perturbs πθ around this narrow band, so that G(ri,t) ≈ 0 in most cases. Consequently, ∇θJCLIP(θ) ≈ 0. Therefore, no meaningful performance improvement observed in 10-step RandomCalculation with random reward in Fig. 7. Overall, clipped objective introduces systematic exploitation bias for high-probability tokens, whereas mid-probability tokens are less optimized.

Response Similarity Before and After RL. We further compare the responses of the model before and after RL, with ROUGE-L and KL distance1 as the similarity score. As

1For each generated answer we compute KL(PBase ∥ PFT) over the full vocabulary, where PBase and PFT denote probability distributions produced by the fine-tuned and base models, respectively.

###### Dataset Reward Signal ROUGE-L

Correct 0.555 Random 0.601

MATH-500

Mv-Incorrect 0.563

Correct 0.225 Random 0.247

RandomCalculation 5 Steps

Mv-Incorrect 0.251

Correct 0.193 Random 0.251

RandomCalculation 10 Steps

Mv-Incorrect 0.279

Table 3: Similarity of model outputs before and after RL.

MATH-500

RandomCalculation 5 Steps

10 Steps

| |
|---|

| |
|---|

| |
|---|

Correct

Random

Mv-Incorrect

0.03

0.03

0.03

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

0.02

0.02

0.02

KL

KL

KL

0.01

0.01

0.01

0.00

0.00

0.00

Figure 8: KL distance of model outputs before and after RL.

shown in Tab. 3 and Fig. 8, the similarity in MATH-500 is substantially higher than in RandomCalculation, further implying that MATH-500 suffers from data contamination. Additionally, spurious rewards achieve even higher ROUGE-L than correct reward after RL. Therefore, performance surge under spurious rewards arises because GRPO inadvertently triggers Qwen to retrieve memorized answers, rather than stimulating Qwen’s existing reasoning patterns like codes as explained in Shao et al. (2025). This is due to the exploitation bias of GRPO. However, RL with correct reward can still stimulate the model to find new reasoning paths, as verified with a smaller ROUGE-L of response.

### 5 Conclusion

In this work, we investigate the unexpected performance improvements of Qwen on mathematical reasoning with spurious rewards. Our analysis reveals that these gains were primarily due to data contamination rather than Qwen’s inherent mathematical capabilities. By auditing the MATH-500 dataset and introducing a clean benchmark, we demonstrate that Qwen’s successes with spurious reward were driven by memorization of benchmark problems rather than genuine reasoning skills. Additionally, we show that only correctly aligned rewards lead to consistent performance improvements, while spurious rewards fail to provide meaningful benefits. These findings underscore the importance of using uncontaminated benchmarks in evaluating RL-based methods and call for caution when interpreting results from datasets that may suffer from data leakage. Our work highlights the need for rigorous evaluation protocols in future research to ensure that performance gains reflect true advancements in ability, rather than data contamination.

### Acknowledgments

The authors wish to thank the anonymous reviewers for their helpful comments. This work was partially funded by National Natural Science Foundation of China (No.62476061, 62376061, 62206057), Shanghai RisingStar Program (23QA1400200), Natural Science Foundation of Shanghai (23ZR1403500). The computations were partially performed using Ascend AI Accelerators. The authors would like to thank Ascend Cloud Ecological Development Project for the support of Ascend 910 processors. Qin Liu is supported by the Amazon Nova Trusted AI Prize.

### References

Agarwal, S.; Zhang, Z.; Yuan, L.; Han, J.; and Peng, H. 2025. The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning. CoRR, abs/2505.15134.

Allal, L. B.; Lozhkov, A.; Bakouch, E.; Bl´azquez, G. M.; Penedo, G.; Tunstall, L.; Marafioti, A.; Kydl´ıcek, H.; Lajar´ın, A. P.; Srivastav, V.; Lochner, J.; Fahlgren, C.; Nguyen,

- X.; Fourrier, C.; Burtenshaw, B.; Larcher, H.; Zhao, H.; Zakka, C.; Morlon, M.; Raffel, C.; von Werra, L.; and Wolf, T. 2025. SmolLM2: When Smol Goes Big - DataCentric Training of a Small Language Model. CoRR, abs/2502.02737.

DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; Zhang,

- X.; Yu, X.; Wu, Y.; Wu, Z. F.; Gou, Z.; Shao, Z.; Li, Z.; Gao, Z.; Liu, A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li, G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin, H.; Gao, H.; Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Cai, J. L.; Ni, J.; Liang, J.; Chen, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Zhao, L.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang, M.; Li, M.; Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen,

- Q.; Du, Q.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen,
- R. J.; Jin, R. L.; Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye,
- S.; Wang, S.; Yu, S.; Zhou, S.; Pan, S.; and Li, S. S. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. CoRR, abs/2501.12948.

Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; Goyal, A.; Hartshorn, A.; Yang, A.; Mitra, A.; Sravankumar, A.; Korenev, A.; Hinsvark, A.; Rao, A.; Zhang, A.; Rodriguez, A.; Gregerson, A.; Spataru, A.; Rozi`ere, B.; Biron, B.; Tang, B.; Chern, B.; Caucheteux, C.; Nayak, C.; Bi, C.; Marra, C.; McConnell, C.; Keller, C.; Touret, C.; Wu, C.; Wong, C.; Ferrer, C. C.; Nikolaidis, C.; Allonsius,

- D.; Song, D.; Pintz, D.; Livshits, D.; Esiobu, D.; Choudhary, D.; Mahajan, D.; Garcia-Olano, D.; Perino, D.; Hupkes, D.; Lakomkin, E.; AlBadawy, E.; Lobanova, E.; Dinan,
- E.; Smith, E. M.; Radenovic, F.; Zhang, F.; Synnaeve, G.; Lee, G.; Anderson, G. L.; Nail, G.; Mialon, G.; Pang, G.; Cucurell, G.; Nguyen, H.; Korevaar, H.; Xu, H.; Touvron, H.; Zarov, I.; Ibarra, I. A.; Kloumann, I. M.; Misra, I.; Evtimov,

- I.; Copet, J.; Lee, J.; Geffert, J.; Vranes, J.; Park, J.; Mahadeokar, J.; Shah, J.; van der Linde, J.; Billock, J.; Hong,
- J.; Lee, J.; Fu, J.; Chi, J.; Huang, J.; Liu, J.; Wang, J.; Yu, J.; Bitton, J.; Spisak, J.; Park, J.; Rocca, J.; Johnstun, J.; Saxe, J.; Jia, J.; Alwala, K. V.; Upasani, K.; Plawiak, K.; Li, K.; Heafield, K.; Stone, K.; and et al. 2024. The Llama 3 Herd of Models. CoRR, abs/2407.21783.

Gao, Z.; Chen, L.; Zhou, J.; and Dai, B. 2025. One-shot Entropy Minimization. arXiv:2505.20282.

Han, X.; Jian, Y.; Hu, X.; Liu, H.; Wang, Y.; Fan, Q.; Ai, Y.; Huang, H.; He, R.; Yang, Z.; and You, Q. 2024. InfiMMWebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning. CoRR, abs/2409.12568.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart,

- S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring Mathematical Problem Solving With the MATH Dataset. In Vanschoren, J.; and Yeung, S., eds., Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual. Jain, N.; Han, K.; Gu, A.; Li, W.-D.; Yan, F.; Zhang,
- T.; Wang, S.; Solar-Lezama, A.; Sen, K.; and Stoica,

I. 2024. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv:2403.07974.

Lewkowycz, A.; Andreassen, A.; Dohan, D.; Dyer, E.; Michalewski, H.; Ramasesh, V. V.; Slone, A.; Anil, C.; Schlag, I.; Gutman-Solo, T.; Wu, Y.; Neyshabur, B.; GurAri, G.; and Misra, V. 2022. Solving Quantitative Reasoning Problems with Language Models. In Koyejo, S.; Mohamed, S.; Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Li, J.; Beeching, E.; Tunstall, L.; Lipkin, B.; Soletskyi, R.; Huang, S.; Rasul, K.; Yu, L.; Jiang, A. Q.; Shen, Z.; et al. 2024. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13: 9.

Li, P.; Skripkin, M.; Zubrey, A.; Kuznetsov, A.; and Oseledets, I. 2025. Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models. arXiv:2506.06395.

Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, 74–81. Barcelona, Spain: Association for Computational Linguistics.

Liu, J.; Liu, H.; Xiao, L.; Wang, Z.; Liu, K.; Gao, S.; Zhang, W.; Zhang, S.; and Chen, K. 2024. Are Your LLMs Capable of Stable Reasoning? CoRR, abs/2412.13147.

Liu, Z.; Chen, C.; Li, W.; Qi, P.; Pang, T.; Du, C.; Lee, W. S.; and Lin, M. 2025. Understanding R1-Zero-Like Training: A Critical Perspective. CoRR, abs/2503.20783.

Lv, A.; Xie, R.; Sun, X.; Kang, Z.; and Yan, R. 2025. The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason. arXiv:2505.22653.

OLMo, T.; Walsh, P.; Soldaini, L.; Groeneveld, D.; Lo, K.; Arora, S.; Bhagia, A.; Gu, Y.; Huang, S.; Jordan, M.; Lambert, N.; Schwenk, D.; Tafjord, O.; Anderson, T.; Atkinson, D.; Brahman, F.; Clark, C.; Dasigi, P.; Dziri, N.; Guerquin, M.; Ivison, H.; Koh, P. W.; Liu, J.; Malik, S.; Merrill, W.; Miranda, L. J. V.; Morrison, J.; Murray, T.; Nam, C.; Pyatkin, V.; Rangapur, A.; Schmitz, M.; Skjonsberg, S.; Wadden, D.; Wilhelm, C.; Wilson, M.; Zettlemoyer, L.; Farhadi, A.; Smith, N. A.; and Hajishirzi, H. 2025. 2 OLMo 2 Furious. CoRR, abs/2501.00656.

- OpenAI. 2024a. Hello GPT-4o.
- OpenAI. 2024b. Learning to Reason with LLMs. OpenAI. 2025. Introducing OpenAI o3 and o4-mini.

Paster, K.; Santos, M. D.; Azerbayev, Z.; and Ba, J. 2024. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Prabhudesai, M.; Chen, L.; Ippoliti, A.; Fragkiadaki, K.; Liu, H.; and Pathak, D. 2025. Maximizing Confidence Alone Improves Reasoning. arXiv:2505.22660.

- Qwen Team. 2024. QwQ: Reflect deeply on the boundaries of the unknown.
- Qwen Team. 2025. QwQ-32B: Embracing the Power of Reinforcement Learning.

Shafayat, S.; Tajwar, F.; Salakhutdinov, R.; Schneider, J.; and Zanette, A. 2025. Can Large Reasoning Models SelfTrain? arXiv:2505.21444.

Shao, R.; Li, S. S.; Xin, R.; Geng, S.; Wang, Y.; Oh, S.; Du, S. S.; Lambert, N.; Min, S.; Krishna, R.; Tsvetkov,

- Y.; Hajishirzi, H.; Koh, P. W.; and Zettlemoyer, L. 2025. Spurious Rewards: Rethinking Training Signals in RLVR. arXiv:2506.10947.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Zhang, M.; Li,

- Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. CoRR, abs/2402.03300.

- Wang, Y.; Yang, Q.; Zeng, Z.; Ren, L.; Liu, L.; Peng, B.; Cheng, H.; He, X.; Wang, K.; Gao, J.; Chen, W.; Wang, S.; Du, S. S.; and Shen, Y. 2025a. Reinforcement Learning for Reasoning in Large Language Models with One Training Example. CoRR, abs/2504.20571.
- Wang, Z.; Li, X.; Xia, R.; and Liu, P. 2024. MathPile: A Billion-Token-Scale Pretraining Corpus for Math. In Globersons, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J. M.; and Zhang, C., eds., Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Wang, Z.; Zhou, F.; Li, X.; and Liu, P. 2025b. OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling. arXiv:2506.20512.

Xu, R.; Wang, Z.; Fan, R.; and Liu, P. 2024. Benchmarking Benchmark Leakage in Large Language Models. CoRR, abs/2404.18824.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou, F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou,

- J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.; Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men,

- R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,

- X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang,
- Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Wang, Z.; Cui, Z.; Zhang,
- Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report. CoRR, abs/2505.09388.

Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Xia, T.; Ren, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Cui, Z.; Zhang, Z.; and Qiu, Z. 2024a. Qwen2.5 Technical Report. CoRR, abs/2412.15115.

Yang, A.; Zhang, B.; Hui, B.; Gao, B.; Yu, B.; Li, C.; Liu, D.; Tu, J.; Zhou, J.; Lin, J.; Lu, K.; Xue, M.; Lin, R.; Liu, T.; Ren, X.; and Zhang, Z. 2024b. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via SelfImprovement. CoRR, abs/2409.12122.

Yue, Y.; Chen, Z.; Lu, R.; Zhao, A.; Wang, Z.; Yue, Y.; Song,

- S.; and Huang, G. 2025. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? CoRR, abs/2504.13837.

Zhao, A.; Wu, Y.; Yue, Y.; Wu, T.; Xu, Q.; Yue, Y.; Lin, M.; Wang, S.; Wu, Q.; Zheng, Z.; and Huang, G. 2025a. Absolute Zero: Reinforced Self-play Reasoning with Zero Data. CoRR, abs/2505.03335.

Zhao, X.; Kang, Z.; Feng, A.; Levine, S.; and Song, D. 2025b. Learning to Reason without External Rewards. arXiv:2505.19590.

Zuo, Y.; Zhang, K.; Qu, S.; Sheng, L.; Zhu, X.; Qi, B.; Sun, Y.; Cui, G.; Ding, N.; and Zhou, B. 2025. TTRL: Test-Time Reinforcement Learning. CoRR, abs/2504.16084.

### A Discussion and Limitation

Due to limited computational resources, our experiments were restricted to a subset of commonly used Qwen2.5 and Qwen3 series. Besides, given the rapid development of various RL algorithms, it is infeasible to conduct a comprehensive evaluation of all these methods in the short term. In future work, our efforts will focus on comprehensive evaluation on more diverse benchmarks, reinforcement learning methods, and model families.

### B Details of RandomCalculation Construction

We provide the specific algorithm for the construction of our clean RandomCalculation benchmark here:

Algorithm 1: Construction of RandomCalculation Dataset Require: Maximum computation steps: N = 20

Initialize dataset S0 with basic mathematical expressions Initialize dataset list: DL ← {S0} Define operator set: OPSET ← {+,−,×,÷}

- for i = 1 to N do Di ← ∅
- for j = 0 to ⌈i/2⌉ do Randomly select Left ∈ DL[j] Randomly select Right ∈ DL[i − 1 − j] Randomly select op ∈ OPSET Randomly swap Left and Right expr ← Left op Right

Add expr to Di end for Append Di to DL

end for Save DL as the RandomCalculation dataset

### C Quantitative Analysis of Answer-Related Numeric Token Probabilities

Fig. 9 illustrates the change in probabilities of numeric tokens associated with final answers before and after reinforcement learning (RL), evaluated on 100 samples from the MATH-500 and 10-step RandomCalculation datasets using random reward. The MATH-500 results exhibit consistently high probabilities both before and after RL, suggesting that the model retains a strong memory of the answers. In contrast, the results on the clean 10-step RandomCalculation dataset are more dispersed, showing that random reward signals are ineffective.

### D Quantitative Results on Partial-Prompt Answer Accuracy

Tab. 4 presents an expanded comparison of partial-prompt answer accuracy between Qwen-2.5 and Llama-3.1 on the MATH-500 benchmark. To complement these findings, Tab. 5–8 reports analogous results across several additional mathematics datasets, evaluated under four generation configurations: Greedy decoding with and without templates, and Avg@16 decoding with and without templates.

### E Quantitative Results for Qwen3

Qwen3 is the closest open-source model within the Qwen family, and we subject it to the same memorization diagnostics. Table 9 reports its partial-prompt completion rate, whereas Tables 10–13 present the corresponding partialprompt answer accuracy under all generation settings. The results mirror those of Qwen2.5: despite the model’s increased capacity, it still exhibits pronounced evidence of data contamination, likely attributable to pre-training on large-scale web corpora.

### F RLVR on LiveMathBench

In the early exploratory phase, we conduct RLVR training on Qwen2.5-Math-7B with the clean LiveMathBench dataset. The experimental results are shown in Fig. 10. Under correct reward signal, the model achieves a limited improvement, which is mainly due to the relatively small amount of training data. In contrast, when using random reward signal, the model fails to obtain stable performance gains and ultimately exhibits a declining trend.

### G Other Reasoning Domains

Beyond the mathematics domain, we also conduct preliminary memory tests on LiveCodeBench (Jain et al. 2024), a commonly used code evaluation benchmark. The results are shown in Tab. 14. When 80% proportion of problems provided as prompts, Qwen2.5-Math-7B is able to accurately reproduce 56.59% of the problems, whereas Llama3.1-8B only reproduces 4.40%. We also observe that the outputs of Qwen2.5-Math-7B frequently included complete test cases, as illustrated in Fig. 17.

### H Illustrative Examples of Model Output

Figures 11 and 12 present concrete instances of memorization by Qwen2.5-7B on the AMC and AIME2024 benchmarks, respectively. Figures 13–15 provide analogous examples for Qwen2.5-Math-7B on MATH-500, AMC, and AIME2024. Figure 16 depicts a representative response produced by the RLVR-fine-tuned Qwen2.5-Math-7B on the 5step RandomCalculation benchmark.

### I Hardware and Software Requirements

All experiments in this work were conducted on NVIDIA A800 80G GPUs, with Ubuntu 20.04.6 LTS as the operating system and CUDA driver version 12.4. Reinforcement learning (RL) training is performed using 8 A800 GPUs and 512 GB of RAM.

### J RLVR Training Details

We follow the standard TTRL experimental configuration. Specifically, during training, the learning rate is set to 5e7 and the temperature is set to 1.0. For each prompt, we generate 16 samples. The training batch size is set to 128. To improve efficiency and reduce memory usage, we enable FlashAttention to accelerate attention computation in the Transformer model.

MATH-500

RandomCalculation 10 Steps

1.0

1.0

Probability

Probability

0.5

0.5

0.0

0.0

Post-RL Before-RL

Post-RL Before-RL

0 20 40 60 80 100 Sample Index

0 20 40 60 80 100 Sample Index

- Figure 9: Token-level comparison of the probabilities assigned to answer-relevant numeric tokens before and after reinforcement learning (RL) on the MATH-500 and RandomCalculation (10-step) benchmarks under a random-reward setting. For each benchmark, 100 problems are randomly selected; the post-RL model generates an answer for every problem, and the probabilities that both the pre- and post-RL models assign to each numeric token appearing in these answers are subsequently evaluated.

| | | | |
|---|---|---|---|
| | | | |

0 100 200 300 Training Step

0.0

0.1

0.2

0.3

0.4

Accuracy

Qwen2.5-Math-7B

Correct Random

- Figure 10: RLVR training is performed on Qwen2.5-Math-7B with the LiveMathBench (v202412 + v202505) dataset under correct and random reward signals.

Model Configuration 100% 80% 60% 40%

Greedy (w/o Template) 72.20 63.80 53.60 41.20 Avg@16 (w/o Template) 68.53 57.25 45.51 31.03 Greedy (w/ Template) 50.60 29.20 20.20 10.00 Avg@16 (w/ Template) 48.84 28.01 18.99 10.36

Qwen2.5-Math-7B

Greedy (w/o Template) 83.60 53.80 31.80 16.00 Avg@16 (w/o Template) 81.76 50.80 31.54 15.43 Greedy (w/ Template) 82.20 3.80 24.40 13.20 Avg@16 (w/ Template) 82.01 43.66 25.14 12.78

Qwen2.5-Math-7B-Instruct

Greedy (w/o Template) 67.60 49.60 30.80 18.40 Avg@16 (w/o Template) 66.20 46.04 30.01 16.10 Greedy (w/ Template) 40.00 22.20 13.40 6.80 Avg@16 (w/ Template) 38.15 21.75 11.89 5.25

Qwen2.5-7B

Greedy (w/o Template) 72.80 50.00 31.20 15.20 Avg@16 (w/o Template) 74.90 50.18 30.88 15.79 Greedy (w/ Template) 72.20 36.00 20.80 10.00 Avg@16 (w/ Template) 73.69 37.16 20.67 8.75

Qwen2.5-7B-Instruct

Greedy (w/o Template) 1.60 2.00 2.40 2.00 Avg@16 (w/o Template) 3.12 2.81 2.80 2.30 Greedy (w/ Template) – – – – Avg@16 (w/ Template) – – – –

Llama3.1-8B

Greedy (w/o Template) 43.60 24.80 15.00 7.40 Avg@16 (w/o Template) 39.61 24.76 14.21 6.84 Greedy (w/ Template) 38.80 17.20 10.60 5.00 Avg@16 (w/ Template) 38.24 18.46 9.95 4.25

Llama3.1-8B-Instruct

- Table 4: Accuracy (%) of Qwen and Llama models on the MATH-500 dataset under different generation configurations, using varying proportions of questions as prompts. The configuration parameters can be found in Table 1. Note that Llama3.1-8B lacks an official chat template, so template-dependent experiments are omitted.

MATH-500 500 72.20 63.80 53.60 41.20 AMC 83 53.01 53.01 43.37 43.37

- AIME2024 30 13.33 13.33 13.33 13.33

- AIME2025 30 16.67 3.33 0.00 0.00 MinervaMath 272 14.71 6.25 2.94 2.57 LiveMathBench 100 7.00 9.00 2.00 3.00

Qwen2.5-Math-7B

MATH-500 500 83.60 53.80 31.80 16.00 AMC 83 57.83 27.71 7.23 4.82

- AIME2024 30 16.67 3.33 3.33 0.00

- AIME2025 30 6.67 0.00 0.00 0.00 MinervaMath 272 20.96 7.72 3.68 1.84 LiveMathBench 100 9.00 4.00 4.00 1.00

Qwen2.5-Math-7B-Instruct

MATH-500 500 67.60 49.60 30.80 18.40 AMC 83 43.37 32.53 24.10 22.89

- AIME2024 30 6.67 6.67 6.67 3.33

- AIME2025 30 3.33 0.00 0.00 0.00 MinervaMath 272 12.87 4.41 2.57 2.21 LiveMathBench 100 9.00 8.00 3.00 4.00

Qwen2.5-7B

MATH-500 500 72.80 50.00 31.20 15.20 AMC 83 44.58 34.94 22.89 14.46

- AIME2024 30 16.67 6.67 0.00 0.00

- AIME2025 30 6.67 3.33 0.00 0.00 MinervaMath 272 18.01 5.51 3.68 2.21 LiveMathBench 100 9.00 6.00 2.00 4.00

Qwen2.5-7B-Instruct

MATH-500 500 1.60 2.00 2.40 2.00 AMC 83 4.82 2.41 1.20 2.41

- AIME2024 30 0.00 0.00 0.00 0.00

- AIME2025 30 0.00 0.00 0.00 3.33 MinervaMath 272 2.57 1.10 1.10 0.00 LiveMathBench 100 0.00 2.00 1.00 1.00

Llama3.1-8B

MATH-500 500 43.60 24.80 15.00 7.40 AMC 83 21.69 10.84 3.61 2.41

- AIME2024 30 10.00 0.00 0.00 0.00

- AIME2025 30 0.00 0.00 0.00 0.00 MinervaMath 272 10.29 4.04 2.57 1.84 LiveMathBench 100 3.00 3.00 2.00 0.00

Llama3.1-8B-Instruct

- Table 5: Accuracy (%) of different models on various math datasets under Greedy (w/o Template) configuration with varying

MATH-500 500 50.60 29.20 20.20 10.00 AMC 83 37.35 20.48 14.46 7.23 AIME2024 30 10.00 6.67 0.00 0.00

Qwen2.5-Math-7B

AIME2025 30 6.67 3.33 0.00 0.00 MinervaMath 272 9.19 4.78 2.94 2.21 LiveMathBench 100 5.00 8.00 2.00 2.00

MATH-500 500 82.20 43.80 24.40 13.20 AMC 83 55.42 14.46 7.23 4.82 AIME2024 30 20.00 0.00 3.33 0.00

Qwen2.5-Math-7B-Instruct

AIME2025 30 16.67 0.00 0.00 0.00 MinervaMath 272 26.47 6.62 4.78 1.84 LiveMathBench 100 8.00 12.00 6.00 5.00

MATH-500 500 40.00 22.20 13.40 6.80 AMC 83 27.71 9.64 3.61 3.61 AIME2024 30 6.67 0.00 0.00 0.00

Qwen2.5-7B

AIME2025 30 6.67 0.00 0.00 0.00 MinervaMath 272 8.09 4.41 2.21 1.47 LiveMathBench 100 8.00 6.00 2.00 1.00

MATH-500 500 72.20 36.00 20.80 10.00 AMC 83 48.19 10.84 4.82 2.41 AIME2024 30 6.67 3.33 0.00 0.00

Qwen2.5-7B-Instruct

AIME2025 30 6.67 0.00 0.00 0.00 MinervaMath 272 23.53 6.25 3.68 2.94 LiveMathBench 100 10.00 10.00 3.00 2.00

MATH-500 500 – – – – AMC 83 – – – – AIME2024 30 – – – –

Llama3.1-8B

AIME2025 30 – – – – MinervaMath 272 – – – – LiveMathBench 100 – – – –

MATH-500 500 38.80 17.20 10.60 5.00 AMC 83 25.30 6.02 2.41 1.20 AIME2024 30 6.67 0.00 0.00 0.00

Llama3.1-8B-Instruct

AIME2025 30 0.00 0.00 3.33 0.00 MinervaMath 272 15.81 2.94 3.68 2.21 LiveMathBench 100 2.00 2.00 4.00 1.00

- Table 6: Accuracy (%) of different models on various math datasets under Greedy (w/ Template) configuration with varying

MATH-500 500 68.53 57.25 45.51 31.03 AMC 83 49.47 48.57 40.06 37.27 AIME2024 30 18.33 13.75 14.58 14.17

Qwen2.5-Math-7B

AIME2025 30 6.88 1.67 0.83 0.00 MinervaMath 272 11.40 5.12 3.29 1.77

- LiveMathBench 100 9.50 7.56 3.38 3.06

Qwen2.5-Math-7B-Instruct

MATH-500 500 81.76 50.80 31.54 15.43 AMC 83 51.28 28.54 9.41 4.52 AIME2024 30 12.29 4.58 1.88 0.00

AIME2025 30 10.42 1.46 1.04 0.42 MinervaMath 272 18.50 7.08 4.07 1.75

- LiveMathBench 100 10.88 7.81 5.38 2.56

MATH-500 500 66.20 46.04 30.01 16.10 AMC 83 40.06 32.23 27.71 22.52 AIME2024 30 11.04 7.50 4.58 4.58 AIME2025 30 7.92 0.83 0.42 0.21 MinervaMath 272 10.66 5.51 3.26 1.38 LiveMathBench 100 7.75 6.50 4.06 3.19

Qwen2.5-7B

MATH-500 500 74.90 50.18 30.88 15.79 AMC 83 43.07 33.51 17.62 11.75 AIME2024 30 11.67 5.42 1.04 0.21 AIME2025 30 6.67 1.04 0.42 0.00 MinervaMath 272 18.18 6.82 4.11 1.59 LiveMathBench 100 10.06 7.75 5.12 3.31

Qwen2.5-7B-Instruct

MATH-500 500 3.12 2.81 2.80 2.30 AMC 83 0.98 1.58 1.66 1.88 AIME2024 30 0.00 0.00 0.00 0.21

Llama3.1-8B

AIME2025 30 0.00 0.00 0.21 0.21 MinervaMath 272 1.93 1.24 0.92 1.06 LiveMathBench 100 0.25 1.38 0.44 1.06

MATH-500 500 39.61 24.76 14.21 6.84 AMC 83 21.84 9.26 3.31 2.48 AIME2024 30 5.42 1.46 0.42 0.21

Llama3.1-8B-Instruct

AIME2025 30 0.42 0.42 0.00 0.00 MinervaMath 272 8.89 5.08 3.19 1.45 LiveMathBench 100 3.25 2.00 1.38 1.88

- Table 7: Accuracy (%) of different models on various math datasets under Avg@16 (w/o Template) configuration with varying

MATH-500 500 48.84 28.01 18.99 10.36 AMC 83 35.32 14.83 10.99 8.36 AIME2024 30 11.46 5.83 0.83 1.25

Qwen2.5-Math-7B

AIME2025 30 3.54 1.46 0.42 0.00 MinervaMath 272 8.94 4.83 2.96 1.70 LiveMathBench 100 5.75 4.56 3.06 2.06

MATH-500 500 82.01 43.66 25.14 12.78 AMC 83 50.53 14.68 6.70 4.07 AIME2024 30 12.50 0.83 0.21 0.00

Qwen2.5-Math-7B-Instruct

AIME2025 30 10.62 0.00 0.42 0.00 MinervaMath 272 27.07 7.01 4.76 1.93 LiveMathBench 100 11.19 10.94 6.56 4.44

MATH-500 500 38.15 21.75 11.89 5.25 AMC 83 20.78 8.89 4.22 2.86 AIME2024 30 5.42 0.00 0.00 0.21

Qwen2.5-7B

AIME2025 30 0.62 0.21 0.00 0.00 MinervaMath 272 8.18 3.52 2.27 1.77 LiveMathBench 100 5.00 3.69 2.81 1.12

MATH-500 500 73.69 37.16 20.67 8.75 AMC 83 44.80 10.54 5.12 1.88 AIME2024 30 12.92 1.25 1.25 0.00

Qwen2.5-7B-Instruct

AIME2025 30 5.21 0.00 0.00 0.00 MinervaMath 272 24.95 6.20 3.56 1.79 LiveMathBench 100 10.25 8.75 4.31 3.56

MATH-500 500 – – – – AMC 83 – – – – AIME2024 30 – – – –

Llama3.1-8B

AIME2025 30 – – – – MinervaMath 272 – – – – LiveMathBench 100 – – – –

MATH-500 500 38.24 18.46 9.95 4.25 AMC 83 19.43 6.10 3.24 2.48 AIME2024 30 6.25 0.42 0.00 0.00

Llama3.1-8B-Instruct

AIME2025 30 0.42 0.00 0.21 0.21 MinervaMath 272 14.38 3.15 2.44 1.47 LiveMathBench 100 3.00 2.56 1.69 2.06

- Table 8: Accuracy (%) of different models on various math datasets under Avg@16 (w/ Template) configuration with varying

80%-Problem 60%-Problem 40%-Problem ROUGE-L EM ROUGE-L EM ROUGE-L EM

Model Dataset Size

MATH-500 500 67.03 40.00 58.31 21.80 48.74 9.40 AMC 83 74.57 50.60 64.87 33.73 67.77 30.12 AIME2024 30 69.72 43.33 52.27 16.67 50.93 16.67

Qwen3-4B-Base

AIME2025 30 53.61 10.00 34.94 0.00 30.18 0.00 MinervaMath 272 35.05 3.31 32.94 0.74 29.25 0.00 LiveMathBench 100 41.42 4.00 31.56 0.00 27.86 0.00

MATH-500 500 51.41 19.00 44.42 5.20 35.78 0.40 AMC 83 43.22 2.41 32.23 0.00 33.42 0.00 AIME2024 30 48.54 0.00 33.52 0.00 29.56 0.00

Qwen3-4B

AIME2025 30 47.15 3.33 30.71 0.00 27.26 0.00 MinervaMath 272 36.50 2.57 31.48 0.37 27.05 0.00 LiveMathBench 100 35.91 3.00 30.44 0.00 28.80 0.00

MATH-500 500 72.43 48.00 66.15 32.00 56.61 18.60 AMC 83 79.22 56.63 70.82 40.96 73.81 34.94 AIME2024 30 72.36 53.33 58.88 23.33 56.42 16.67

Qwen3-8B-Base

AIME2025 30 53.57 10.00 34.15 0.00 29.41 0.00 MinervaMath 272 37.50 2.94 33.24 0.00 29.56 0.00 LiveMathBench 100 43.26 7.00 34.10 0.00 30.37 0.00

MATH-500 500 53.66 22.00 44.91 5.80 36.30 0.60 AMC 83 45.33 4.82 36.38 0.00 32.48 0.00

- AIME2024 30 54.43 3.33 32.21 0.00 26.55 0.00

- AIME2025 30 48.88 10.00 33.21 0.00 31.76 0.00 MinervaMath 272 38.04 3.68 32.64 0.74 28.52 0.00 LiveMathBench 100 39.17 4.00 30.57 0.00 29.81 0.00

Qwen3-8B

MATH-500 500 75.40 56.40 72.61 43.60 62.88 27.40 AMC 83 80.49 60.24 74.40 48.19 77.39 42.17

- AIME2024 30 76.19 53.33 61.01 33.33 58.83 23.33

- AIME2025 30 56.12 10.00 38.55 0.00 30.80 0.00 MinervaMath 272 38.79 2.94 34.32 0.37 30.51 0.00 LiveMathBench 100 43.35 3.00 34.24 0.00 30.51 0.00

Qwen3-14B-Base

MATH-500 500 56.96 24.80 48.13 7.40 38.30 0.80 AMC 83 48.29 7.23 35.85 0.00 36.37 0.00

- AIME2024 30 55.85 6.67 27.03 0.00 28.30 0.00

- AIME2025 30 55.44 13.33 36.99 0.00 32.92 0.00 MinervaMath 272 37.42 3.31 32.67 0.00 28.79 0.00 LiveMathBench 100 39.15 5.00 31.26 0.00 29.49 0.00

Qwen3-14B

- Table 9: Accuracy (Exact Match, EM) and ROUGE-L scores on several datasets (lower scores in gray) under different prompt prefix ratios in greedy decoding mode (Greedy (w/o Template) configuration).

MATH-500 500 68.00 47.80 31.40 16.60 AMC 83 38.55 38.55 31.33 25.30 AIME2024 30 6.67 13.33 10.00 6.67 AIME2025 30 6.67 6.67 0.00 0.00 MinervaMath 272 10.29 4.04 2.94 1.10 LiveMathBench 100 6.00 5.00 3.00 3.00

Qwen3-4B-Base

MATH-500 500 58.40 36.20 21.60 8.60 AMC 83 48.19 18.07 8.43 4.82 AIME2024 30 16.67 6.67 6.67 0.00

Qwen3-4B

AIME2025 30 13.33 3.33 0.00 0.00 MinervaMath 272 8.82 6.62 2.94 1.84 LiveMathBench 100 6.00 5.00 2.00 3.00

MATH-500 500 70.80 53.80 42.60 26.20 AMC 83 42.17 36.14 30.12 26.51 AIME2024 30 20.00 13.33 10.00 10.00

Qwen3-8B-Base

AIME2025 30 10.00 0.00 0.00 0.00 MinervaMath 272 11.03 4.41 1.84 1.47 LiveMathBench 100 9.00 4.00 4.00 4.00

MATH-500 500 62.60 38.00 21.40 8.20 AMC 83 49.40 28.92 8.43 0.00 AIME2024 30 20.00 10.00 3.33 3.33

Qwen3-8B

AIME2025 30 16.67 0.00 0.00 3.33 MinervaMath 272 12.50 6.25 5.15 0.00 LiveMathBench 100 9.00 7.00 0.00 3.00

MATH-500 500 74.00 62.20 49.60 32.80 AMC 83 54.22 44.58 37.35 32.53 AIME2024 30 23.33 16.67 10.00 16.67

Qwen3-14B-Base

AIME2025 30 3.33 0.00 0.00 0.00 MinervaMath 272 8.46 5.88 2.57 1.47 LiveMathBench 100 11.00 12.00 4.00 4.00

MATH-500 500 74.00 44.60 24.80 11.40 AMC 83 54.22 18.07 9.64 6.02 AIME2024 30 13.33 3.33 0.00 3.33

Qwen3-14B

AIME2025 30 26.67 3.33 3.33 0.00 MinervaMath 272 13.97 8.46 4.41 1.47 LiveMathBench 100 9.00 6.00 2.00 1.00

- Table 10: Accuracy (%) of different models on various math datasets under Greedy (w/o Template) configuration with varying

MATH-500 500 67.31 46.11 30.94 16.44 AMC 83 39.91 33.51 25.23 20.63 AIME2024 30 12.50 7.29 6.46 6.88 AIME2025 30 8.54 0.83 0.83 0.00 MinervaMath 272 9.28 3.70 2.16 1.15 LiveMathBench 100 8.00 4.44 4.12 2.94

Qwen3-4B-Base

MATH-500 500 59.08 33.81 19.80 8.92 AMC 83 42.70 18.98 8.43 4.37 AIME2024 30 15.83 6.25 2.08 1.46

Qwen3-4B

AIME2025 30 11.67 1.67 1.04 0.00 MinervaMath 272 10.34 5.63 3.68 1.61 LiveMathBench 100 6.50 4.88 2.62 3.06

MATH-500 500 69.51 52.28 39.57 24.41 AMC 83 47.67 39.76 29.59 27.48 AIME2024 30 16.67 13.33 14.17 12.50

Qwen3-8B-Base

AIME2025 30 10.21 0.42 0.21 0.00 MinervaMath 272 9.88 4.71 2.41 1.19 LiveMathBench 100 8.12 5.19 3.50 3.25

MATH-500 500 64.28 37.72 20.74 9.01 AMC 83 45.33 22.59 10.02 1.88 AIME2024 30 23.33 8.33 3.96 1.67

Qwen3-8B

AIME2025 30 16.67 3.54 2.29 0.42 MinervaMath 272 8.92 6.32 4.18 1.40

- LiveMathBench 100 8.50 6.56 3.25 2.69

Qwen3-14B-Base

MATH-500 500 72.10 59.60 47.40 32.46 AMC 83 55.12 45.11 36.07 34.71 AIME2024 30 16.88 17.29 13.12 15.62

AIME2025 30 10.83 1.04 1.25 0.42 MinervaMath 272 8.48 4.73 2.34 1.65

- LiveMathBench 100 9.31 6.62 4.44 3.44

MATH-500 500 73.70 43.53 25.00 11.44 AMC 83 49.40 20.41 6.70 3.39 AIME2024 30 25.83 7.50 1.04 0.83

Qwen3-14B

AIME2025 30 22.50 3.12 2.08 0.42 MinervaMath 272 12.48 7.10 4.23 1.88 LiveMathBench 100 9.69 6.56 3.50 3.19

- Table 11: Accuracy (%) of different models on various math datasets under Avg@16 (w/o Template) configuration with varying

MATH-500 500 37.60 23.00 12.80 5.80 AMC 83 32.53 14.46 7.23 0.00 AIME2024 30 10.00 0.00 0.00 0.00

Qwen3-4B-Base

AIME2025 30 10.00 0.00 0.00 0.00 MinervaMath 272 10.29 4.78 1.84 2.57 LiveMathBench 100 11.00 3.00 2.00 1.00

MATH-500 500 64.20 31.40 19.60 7.80 AMC 83 31.33 8.43 8.43 2.41 AIME2024 30 6.67 0.00 0.00 0.00

Qwen3-4B

AIME2025 30 3.33 0.00 0.00 0.00 MinervaMath 272 24.63 7.35 2.94 1.84 LiveMathBench 100 3.00 6.00 2.00 3.00

MATH-500 500 67.20 38.80 20.60 11.00 AMC 83 40.96 12.05 13.25 4.82 AIME2024 30 23.33 3.33 0.00 0.00

Qwen3-8B-Base

AIME2025 30 6.67 0.00 0.00 0.00 MinervaMath 272 20.59 7.35 4.04 1.84 LiveMathBench 100 13.00 9.00 3.00 2.00

MATH-500 500 63.20 31.40 20.20 10.20 AMC 83 26.51 7.23 2.41 3.61 AIME2024 30 6.67 0.00 0.00 0.00

Qwen3-8B

AIME2025 30 6.67 0.00 0.00 0.00 MinervaMath 272 23.53 7.35 5.88 2.57 LiveMathBench 100 3.00 1.00 0.00 4.00

MATH-500 500 73.40 39.60 25.20 12.80 AMC 83 53.01 15.66 9.64 7.23 AIME2024 30 16.67 6.67 0.00 0.00

Qwen3-14B-Base

AIME2025 30 6.67 0.00 0.00 0.00 MinervaMath 272 21.32 8.46 5.88 1.84 LiveMathBench 100 15.00 9.00 4.00 2.00

MATH-500 500 68.20 35.00 19.80 10.20 AMC 83 30.12 6.02 7.23 1.20 AIME2024 30 6.67 0.00 0.00 0.00

Qwen3-14B

AIME2025 30 10.00 0.00 0.00 0.00 MinervaMath 272 26.47 7.72 4.41 3.31 LiveMathBench 100 4.00 8.00 2.00 5.00

- Table 12: Accuracy (%) of different models on various math datasets under Greedy (w/ Template) configuration with varying

Model Dataset Size 100% 80% 60% 40%

MATH-500 500 37.91 21.10 11.91 6.02 AMC 83 21.16 9.04 6.10 3.09 AIME2024 30 5.21 0.83 0.21 0.00

Qwen3-4B-Base

AIME2025 30 4.79 0.42 0.62 0.21 MinervaMath 272 12.48 3.86 1.61 1.10 LiveMathBench 100 6.00 3.88 3.38 1.81

MATH-500 500 63.04 32.02 19.76 8.97 AMC 83 31.10 8.96 5.27 2.41 AIME2024 30 5.42 0.21 0.00 0.42

Qwen3-4B

AIME2025 30 7.50 0.62 0.21 0.21 MinervaMath 272 23.18 7.28 4.11 2.16 LiveMathBench 100 2.44 3.31 2.50 3.25

MATH-500 500 63.36 34.84 20.59 10.47 AMC 83 41.04 13.18 7.83 3.99 AIME2024 30 11.88 1.46 0.42 0.62

Qwen3-8B-Base

AIME2025 30 9.58 0.42 0.21 0.42 MinervaMath 272 17.90 6.53 3.63 1.95 LiveMathBench 100 7.44 6.06 3.50 2.62

MATH-500 500 61.74 32.35 20.12 9.00 AMC 83 28.54 8.28 4.22 2.56 AIME2024 30 4.58 0.42 1.04 0.00

Qwen3-8B

AIME2025 30 5.83 0.42 0.00 0.00 MinervaMath 272 22.79 7.67 4.80 2.69 LiveMathBench 100 2.31 2.31 2.69 4.00

MATH-500 500 69.53 39.45 23.31 11.29 AMC 83 48.19 17.39 9.04 5.20 AIME2024 30 16.25 1.67 1.04 0.21

Qwen3-14B-Base

AIME2025 30 10.42 0.83 0.42 0.83 MinervaMath 272 19.37 7.79 3.95 2.04 LiveMathBench 100 11.06 6.69 3.69 2.12

MATH-500 500 67.76 36.51 21.75 10.27 AMC 83 33.21 9.71 5.20 3.01 AIME2024 30 7.50 1.25 0.42 0.42

Qwen3-14B

AIME2025 30 10.83 0.42 0.21 0.62 MinervaMath 272 27.00 8.39 5.06 2.78 LiveMathBench 100 1.65 5.19 4.44 4.94

- Table 13: Accuracy (%) of different models on various math datasets under Avg@16 (w/ Template) configuration with varying proportions of problem prefixes used as prompts.

Model

80%-Problem 60%-Problem 40%-Problem ROUGE-L EM ROUGE-L EM ROUGE-L EM

Qwen2.5-Math-7B 83.47 56.59 75.34 35.16 66.66 20.33 Qwen2.5-Math-7B-Instruct 46.60 4.95 36.49 0.00 32.62 0.00 Qwen2.5-7B 94.54 85.71 93.43 81.32 91.46 70.88 Qwen2.5-7B-Instruct 77.50 48.90 70.79 26.37 67.03 16.48 Llama3.1-8B 45.28 4.40 37.53 1.10 29.78 0.00 Llama3.1-8B-Instruct 36.20 1.65 32.77 0.00 28.33 0.00

- Table 14: Accuracy (Exact Match, EM) and ROUGE-L scores of different models on LiveCodeBench (test generation) under different prompt prefix ratios in greedy decoding mode (Greedy (w/o Template) configuration).

Example from AMC Problem: The taxicab distance between points (x1,y1) and (x2,y2) in the coordinate plane is given by

|x1 − x2| + |y1 − y2|. For how many points P with integer coordinates is the taxicab distance between P and the origin less than or equal to 20? Answer:

|841|
|---|

##### Partial Prompt

Prompt: The taxicab distance between points (x1,y1) and (x2,y2) in the coordinate plane is given

##### Qwen2.5-7B Output by

|x1 − x2| + |y1 − y2|. For how many points P with integer coordinates is the taxicab distance between P and the origin less than or equal to 20 ? To solve the problem, we need to find the number of points P with integer coordinates such that the taxicab distance between P and the origin (0,0) is less than or equal to 20. The taxicab distance between P = (x,y) and the origin is given by:

|x| + |y| ≤ 20

We need to count all integer coordinate pairs (x,y) that satisfy this inequality. Let’s break down the problem into manageable steps:

- 1. Understand the Constraint: The constraint |x| + |y| ≤ 20 means that the sum of the absolute values of x and y must be less than or equal to 20.
- 2. Iterate Over Possible Values: We need to iterate over all possible integer values of x and y and count the pairs that satisfy the constraint.
- 3. Count Valid Points: For each valid pair (x,y), increment a counter to keep track of the number of such points. Let’s implement this in Python to find the solution.

- 1 # Initialize the counter for valid points
- 2 count = 0
- 3 # Iterate over possible values of x and y
- 4 for x in range(-20, 21): # x ranges from -20 to 20
- 5 for y in range(-20, 21): # y ranges from -20 to 20
- 6 if abs(x) + abs(y) <= 20:
- 7 count += 1
- 8 print(count) output 1 841

The number of points P with integer coordinates such that the taxicab distance between P and the origin is less than or equal to 20 is

|841|
|---|

.

- Figure 11: Memorization Example of Qwen2.5-7B on AMC. In this case, the first 40% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration.

Example from AIME2024 Problem: Find the largest possible real part of

96 + 144i z

(75 + 117i)z +

where z is a complex number with |z| = 4. Answer:

|540|
|---|

##### Partial Prompt

Prompt: Find the largest possible real part of (75 + 117i

Qwen2.5-7B Output

)z + 96+144z i where z is a complex number with |z| = 4.

To find the largest possible real part of the expression (75+117i)z+ 96+144z i where z is a complex number with |z| = 4, we can proceed as follows:

- 1. Let z = 4eiθ, where θ is a real number. This is because |z| = 4.
- 2. Substitute z = 4eiθ into the expression.
- 3. Simplify the expression and find the real part.
- 4. Maximize the real part with respect to θ. Let’s start by substituting z = 4eiθ into the expression:

... Combining the real and imaginary parts, we get:

(300cosθ + 24cosθ + 36sinθ − 468sinθ) + i(300sinθ + 468cosθ − 24sinθ + 36cosθ) This simplifies to: (324cosθ − 432sinθ) + i(336cosθ + 276sinθ)

... Let’s calculate this using Python to ensure accuracy.

- 1 import math
- 2
- 3 # Coefficients
- 4 a = 324
- 5 b = -432
- 6
- 7 # Maximum value of a*cos(theta) + b*sin(theta)
- 8 max_real_part = math.sqrt(a**2 + b**2)
- 9 print(max_real_part) output 1 540.0

The largest possible real part of the expression (75 + 117i)z + 96+144z i where z is a complex number with |z| = 4 is

|540|
|---|

.

- Figure 12: Memorization Example of Qwen2.5-7B on AIME2024. In this case, the first 40% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration.

Example from MATH-500 Problem: Suppose a and b are different prime numbers greater than 2. How many whole-number divisors are there for the integer a(2a + b) − 2a2 + ab? Answer:

|8|
|---|

Partial Prompt Prompt: Suppose a and b are different prime numbers greater than

##### Qwen2.5-Math-7B Output

2. How many whole-number divisors are there for the integer a(2a + b) − 2a2 + ab? To determine the number of whole-number divisors of the integer a(2a + b) − 2a2 + ab, we first need to simplify the expression. Given the expression:

a(2a + b) − 2a2 + ab Let’s simplify it step-by-step:

a(2a + b) − 2a2 + ab = 2a2 + ab − 2a2 + ab = 2ab

So, the expression simplifies to 2ab. Since a and b are different prime numbers greater than 2, 2ab is a product of three distinct prime factors: 2, a, and b. The number of divisors of a number n with the prime factorization n = pe

- 1

1 pe

2

- 2 ···pe

k is given by (e1 +

k

1)(e2 + 1)···(ek + 1). For 2ab, the prime factorization is 21 ·a1 ·b1. Therefore, the number of divisors is (1+1)(1+1)(1+1) = 2 · 2 · 2 = 8. Let’s confirm this with Python code.

- 1 # Function to calculate the number of divisors of a number
- 2 def number_of_divisors(n):
- 3 count = 0
- 4 for i in range(1, int(n**0.5) + 1):
- 5 if n % i == 0:
- 6 if i * i == n:
- 7 count += 1
- 8 else:
- 9 count += 2
- 10 return count
- 11 # Let’s choose two different prime numbers greater than 2, ...
- 12 a = 3
- 13 b = 5
- 14 expression_value = 2 * a * b
- 15 # Calculate the number of divisors of the expression value
- 16 divisors_count = number_of_divisors(expression_value)
- 17 print(divisors_count) output 1 8

|8|
|---|

The number of whole-number divisors of the integer a(2a + b) − 2a2 + ab is

.

- Figure 13: Memorization Example of Qwen2.5-Math-7B on MATH-500. In this case, the first 40% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration.

Example from AMC Problem: What is the value of

23 − 13 + 43 − 33 + 63 − 53 + ··· + 183 − 173?

|3159.0|
|---|

Answer: Partial Prompt Prompt: What is the value of

23 − 13

##### Qwen2.5-Math-7B Output

+43 − 33 + 63 − 53 + ··· + 183 − 173?

To solve the given problem, we need to evaluate the alternating sum of cubes: 23 − 13 + 43 − 33 + 63 − 53 + ··· + 183 − 173.

We can break this down into pairs of terms and compute the difference for each pair. Specifically, we need to compute:

(23 − 13) + (43 − 33) + (63 − 53) + ··· + (183 − 173). Let’s write a Python code to compute this sum.

- 1 # Initialize the sum
- 2 result = 0
- 3
- 4 # Iterate through the pairs (2,1), (4,3), (6,5), ..., (18,17)
- 5 for i in range(1, 18, 2):
- 6 result += ( (i+1)**3 - i**3 )
- 7
- 8 print(result) output 1 3159

|3159|
|---|

The value of the given expression 23 − 13 + 43 − 33 + 63 − 53 + ··· + 183 − 173 is

.

- Figure 14: Memorization Example of Qwen2.5-Math-7B on AMC. In this case, the first 40% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration.

Example from AIME2024 Problem: Find the number of triples of nonnegative integers (a,b,c) satisfying a + b + c = 300 and

a2b + a2c + b2a + b2c + c2a + c2b = 6,000,000. Answer:

|601|
|---|

##### Partial Prompt

Prompt: Find the number of triples of nonnegative integers (a,b,c) satisfying

##### Qwen2.5-Math-7B Output a + b + c = 300 and

a2b + a2c + b2a + b2c + c2a + c2b = 6,000,000.

To solve the problem, we need to find the number of triples (a,b,c) of nonnegative integers that satisfy the following conditions:

- 1. a + b + c = 300
- 2. a2b + a2c + b2a + b2c + c2a + c2b = 6,000,000 Let’s start by analyzing the given equations. We can use Python to iterate through possible values of a,b, and c to find the valid triples that satisfy both conditions. First, we can express c in terms of a and b using the first equation:

c = 300 − a − b

Next, we substitute c into the second equation and check if the equation holds for different values of a and b. Let’s write the Python code to iterate through possible values of a and b and check the conditions.

- 1 # Initialize the count of valid triples
- 2 count = 0
- 3
- 4 # Iterate through possible values of a and b
- 5 for a in range(301):
- 6 for b in range(301 - a):
- 7 c = 300 - a - b
- 8 if a**2 * b + a**2 * c + b**2 * a + b**2 * c + c**2 * a + c**2 * b

== 6000000:

- 9 count += 1
- 10
- 11 print(count) output 1 601

|601|
|---|

The number of triples (a,b,c) of nonnegative integers that satisfy the given conditions is

.

- Figure 15: Memorization Example of Qwen2.5-Math-7B on AIME2024. In this case, the first 40% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration.

Example from RandomCalculation, 5 Calculation Steps Problem: Evaluate this LaTeX numerical expression step-by-step and give the final value within \boxed{}:

34 8

89 4 −

49 9 ∗

56 4

##### + 622 Answer:

182/

+

|3866.26307189542|
|---|

##### Qwen2.5-Math-7B Output

To evaluate the given LaTeX numerical expression step-by-step, we will follow the order of operations (PEMDAS/BODMAS): Parentheses/Brackets, Exponents/Orders, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). The expression is:

34 8

89 4 −

49 9 ∗

56 4

182/

+ 622

+

- Step 1: Evaluate the exponents. 182 = 324

622 = 3844

...

- Step 2: Simplify the fractions and perform the division. 34

8

= 4.25

...

- Step 3: Simplify the remaining fractions and perform the multiplication. 89

4

= 22.25 49

9 ≈ 5.444444444444444

...

- Step 4: Perform the addition and subtraction from left to right. 76.23529411764706 + 22.25 ≈ 98.48529411764706

98.48529411764706 − 76.22222222222223 ≈ 22.26307189542483 22.26307189542483 + 3844 ≈ 3866.263071895425 So the final value of the expression is:

|3866.263071895425|
|---|

- Figure 16: Example of a correct output produced by RLVR on the 5-Step RandomCalculation dataset. The result is obtained using the Qwen2.5-Math-7B model, trained for 300 steps under the correct reward signal.

##### Example from LiveCodeBench

Problem: You are given a 0-indexed integer array nums. You have to find the maximum sum of a pair of numbers from nums such that the maximum digit in both numbers are equal. Return the maximum sum or -1 if no such pair exists.

Test Cases: Input: [51, 71, 17, 24, 42] Output: 88 Input: [1, 2, 3, 4] Output: -1

Partial Prompt Prompt: You are given a 0-indexed integer array nums. You have to find the maximum sum of a pair of numbers from nums such that the maximum digit in both numbers are equal. Return the maximum

##### Qwen2.5-Math-7B Output sum or -1 if no such pair exists.

- Example 1: Input: nums = [51,71,17,24,42] Output: 88 Explanation: For i = 1 and j = 2, nums[i] and nums[j] have equal maximum digits with a pair sum of 71 + 17 = 88. For i = 3 and j = 4, nums[i] and nums[j] have equal maximum digits with a pair sum of 24 + 42 =

66. It can be shown that there are no other pairs with equal maximum digits, so the answer is 88.

- Example 2: Input: nums = [1,2,3,4] Output: -1 Explanation: No pair exists in nums with equal maximum digits. Constraints: 2 <= nums.length <= 100 1 <= nums[i] <= 104

##### ... Code

- 1 class Solution {
- 2 public int maxSum(int[] nums) {
- 3 int max = -1;
- 4 for(int i=0;i<nums.length-1;i++){
- 5 for(int j=i+1;j<nums.length;j++){
- 6 if(maxDigit(nums[i])==maxDigit(nums[j])){
- 7 max = Math.max(max,nums[i]+nums[j]);
- 8 }
- 9 }
- 10 }
- 11 return max;
- 12 }
- 13 public int maxDigit(int n){
- 14 int max = 0;
- 15 while(n>0){
- 16 int rem = n%10;
- 17 max = Math.max(max,rem);
- 18 n = n/10;
- 19 }
- 20 return max;
- 21 }
- 22 }

- Figure 17: Memorization Example of Qwen2.5-Math-7B on LiveCodeBench. In this case, the first 80% of the original problem is used as the prompt, and the generation is performed under the Greedy (w/o Template) configuration. Notably, the model even included the test cases in its output.

