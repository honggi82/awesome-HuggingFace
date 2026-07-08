# arXiv:2504.21659v2[cs.AI]21May2025

## Ada-R1: Hybrid CoT via Bi-Level Adaptive Reasoning Optimization

#### Haotian Luo1∗, Haiying He2∗, Yibo Wang3, Jinluan Yang4, Rui Liu5

Naiqiang Tan5, Xiaochun Cao1, Dacheng Tao6, Li Shen1† 1 Sun Yat-sen University; 2 China Agricultural University; 3 Tsinghua University; 4 Zhejiang University; 5 Didichuxing Co. Ltd; 6 Nanyang Technological University

### Abstract

Recently, long-thought reasoning models achieve strong performance on complex reasoning tasks, but often incur substantial inference overhead, making efficiency a critical concern. Our empirical analysis reveals that the benefit of using Long-CoT varies across problems: while some problems require elaborate reasoning, others show no improvement—or even degraded accuracy. This motivates adaptive reasoning strategies that tailor reasoning depth to the input. However, prior work primarily reduces redundancy within long reasoning paths, limiting exploration of more efficient strategies beyond the Long-CoT paradigm. To address this, we propose a novel two-stage framework for adaptive and efficient reasoning. First, we construct a hybrid reasoning model by merging long and short CoT models to enable diverse reasoning styles. Second, we apply bi-level preference training to guide the model to select suitable reasoning styles (group-level), and prefer concise and correct reasoning within each style group (instance-level). Experiments demonstrate that our method significantly reduces inference costs compared to other baseline approaches, while maintaining performance. Notably, on five mathematical datasets, the average length of reasoning is reduced by more than 50%, highlighting the potential of adaptive strategies to optimize reasoning efficiency in large language models. Our code is coming soon at https://github.com/StarDewXXX/AdaR1

### 1 Introduction

Recent large language models (LLMs) such as OpenAI’s O1[1] and Deepseek’s R1[2] adopt extended and structured reasoning processes (Long-CoT) to enhance problem-solving, achieving strong performance through human-like deliberation. However, the improved reasoning comes with high inference costs, including increased latency and resource consumption[3–5], which limits deployment in real-time or resource-constrained scenarios. Existing efficiency-oriented methods operate within the Long-CoT distribution, aiming to reduce redundancy through pruning or compression[6, 7, 4]. While effective to some extent, these approaches do not question whether long reasoning is necessary, overlooking potential gains from fundamentally shorter reasoning strategies. CoT-Valve[8] enables both long and short outputs but lacks adaptive selection based on input complexity, leading to suboptimal results.

Our investigation (presented in Section 3) about the benefit of Long-CoT reasoning reveals a crucial insight: the utility of long, elaborate reasoning chains is highly problem-dependent. While complex problems genuinely benefit from detailed, step-by-step derivations, many other problems can be solved accurately and more efficiently with shorter, more direct reasoning paths. In fact, for simpler

*Equal contribution †Corresponding Author: Li Shen (shenli6@mail.sysu.edu.cn)

Preprint. Under review.

problems, forcing a Long-CoT process might not only be wasteful but can sometimes even introduce errors or degrade performance. This observation strongly motivates the need for adaptive reasoning strategies – systems that can tailor the depth and style of their reasoning process to the specific demands of the input problem.

Inspired by these limitations, we propose a two-stage framework for efficient and adaptive reasoning by enabling models to choose between distinct reasoning strategies. The first stage constructs a hybrid model capable of generating both Long-CoT and Short-CoT outputs. The second introduces BiLevel Adaptive Reasoning Optimization , a training method comprising: (i) Group-Level Preference, guiding the model to select an appropriate reasoning style based on input complexity, and (ii) InstanceLevel Preference, encouraging concise yet accurate reasoning within the chosen style. This dual-level adaptation allows dynamic allocation of computational resources, yielding substantial efficiency gains without sacrificing performance. On MATH[9], our method reduces reasoning length by 58% with no accuracy loss, and on GSM8K[10], by 74% with improved accuracy. These results highlight the effectiveness of adaptive reasoning in balancing quality and efficiency in large-scale models.

Our contributions can be summarized as follows:

- • We conduct an empirical analysis investigating the benefits of long Chain-of-Thought (CoT) reasoning relative to shorter CoT approaches, identifying the conditions under which extended reasoning paths offer tangible advantages.
- • We propose using Adaptive Hybrid Reasoning Model to enhance inference efficiency, accompanied by a novel training pipeline (Ada-R1). Comprehensive experiments demonstrate that our proposed method achieves excellent performance, significantly improving efficiency while maintaining high accuracy.
- • We perform further analyses on the resulting Adaptive Hybrid Reasoning Model to gain deeper insights into its characteristics and operational behavior. And we will release the model weights of the Adaptive Hybrid Reasoning Model to the public to encourage further research and application by the community.

### 2 Related Work

Model Merging Model merging [11] is an emerging technique that fuses parameters from multiple trained models into one without access to original training data. Recent methods include parameter interpolation [12] and alignment-based strategies [13], with applications in LLMs, multimodal models, and other machine learning subfields. Beyond simple linear averaging, advanced methods such as DARE [14], TIES-Merging [15], and AdaMerging [16] have been proposed. DARE reduces redundancy by dropping and rescaling delta parameters. TIES-Merging mitigates interference by trimming and aligning parameter signs. AdaMerging improves performance via entropy-based layer or task weighting on unlabeled data. In contrast to traditional model merging that consolidates capabilities from multiple models, our work enables a single model to adaptively choose between Long-CoT and Short-CoT reasoning for each instance, aiming to optimize computational efficiency rather than multi-task performance.

Efficient Reasoning A variety of methods have been proposed for improved reasoning efficiency. Several techniques apply post-training strategies to shorten reasoning paths. [4] constructs preference datasets using DPO and SimPO, guiding models toward concise reasoning through preferencebased fine-tuning. O1-Pruner[6] samples CoTs to build baselines for length and accuracy, then applies offline optimization to reduce reasoning length without harming performance. Similarly, [17] leverages simple fine-tuning on self-generated concise CoTs obtained via best-of-N sampling and few-shot prompting. Some approaches focus on token-level compression. TokenSkip[18], for instance, removes tokens selectively based on their estimated importance within the CoT. CoTValve[8], in contrast, manipulates the parameter space to produce CoTs with varying degrees of compression. Besides, various methods adopt different reasoning paradigms for efficiency. For instance, COCONUT[19] and CCOT[3] enable reasoning within the latent space, reducing the need for explicit token-level generation. Speculative Thinking[20] enhances small model inference by allowing large models to guide them during reasoning. Similarly, LightThinker[21] achieves efficiency by dynamically compressing intermediate thoughts throughout the reasoning process. Also, some works ([22],[23], [24], [25], [26], [27]) design novel reasoning paradigms for efficiency. [28]

also explores model merging technical for reasoning efficiency. Different from most works, our work solves reasoning efficiency in a novel adaptive reasoning perspective.

### 3 Motivation

#### 3.1 Problem Setup

Chain-of-Thought (CoT) prompting has emerged as a powerful technique for enhancing the reasoning capabilities of large language models. Within the CoT paradigm, a distinction can be made between Long-CoT, which involves generating detailed and extensive thinking steps, and Short-CoT, which directly generate solving steps.

#### 3.2 When Do We Need Long-CoT?

Simply applying Long-CoT to all problems introduces unnecessary overhead, especially for easier tasks where detailed reasoning brings little or no benefit. To understand when Long-CoT is truly needed, we empirically analyze its effectiveness across different problem types. We compare LongCoT and Short-CoT on a mixed dataset (MixMathematics) composed of samples from AIME[29], MATH, and GSM8K (details in Section 5.1). We use DeepSeek-R1-Distill-Qwen-7B for Long-CoT, and fine-tune it with 2,000 Short-CoT samples from Qwen2.5-Math-7B-Instruct[30] to create a consistent Short-CoT model. We avoid using Qwen2.5 directly due to its differing training format, which may affect later merging and sampling. From 2,500 problems, we generate 12 responses per model per question and remove cases where both models fail completely. We then calculate accuracy gains (Long-CoT accuracy minus Short-CoT accuracy).

As shown in Figure 1 (left), nearly half the samples show no improvement from Long-CoT, and some even suffer performance drops. Further analysis (Figure 1, right) groups samples by the average length of their Long-CoT outputs—longer CoTs tend to correspond to harder problems. We find that Long-CoT significantly improves accuracy on complex questions but provides little or no benefit for simpler ones.

|Gain|Ratio|
|---|---|
|>0|48%|
|=0|45%|
|<0|7%|

|-1<br>-0.8<br>-0.6<br>-0.4<br>-0.2<br><br><br>0<br><br>0.2<br><br>0.4<br><br>0.6<br><br>0.8<br>1<br><br><br>0k-1k 1k-2k 2k-4k 4k-6k 6k-8k 8k-16k<br><br>Proportion<br><br>Proportion of Accuracy Diff by Length Bins<br><br>gain>0 gain≤0<br><br>|
|---|

- Figure 1: The proportion of gain in the data (left) and the relationship between CoT length and accuracy improvement (right), Long-CoT reasoning improves accuracy on difficult problems but has little effect or harms performance on easy ones.

#### 3.3 A New Perspective on CoT Efficiency

Prior methods (Table 1), such as Overthinking [4], kimi-1.5 [31], and O1-Pruner, typically operate within a limited optimization scope but generally maintain performance stability or incur only a slight drop, with O1-Pruner notably achieving no performance decrease. In contrast, methods designed for a broad optimization scope, including Model Merge and CoT-Valve, did not consider how to tackle easy and different problems, rendering the model incapable of determining its reasoning depth according to the inherent difficulty of the task. Thus they frequently result in significant performance degradation. In a nutshell, methods with a restricted optimization can generally preserve performance but lose the chance to utilize shorter CoT. However, approaches capable of utilize broader CoT

Method CoT Optimization Scope Performance (Accuracy) Overthinking[4] Limited × Slightly Dropped ✓

kimi-1.5[31] Limited × Slightly Dropped ✓ O1-Pruner Limited × Not Dropped ✓

Naive Merge Broad ✓ (mostly) Dropped × CoT-Valve Broad ✓ Dropped ×

Ada-R1(Ours) Broad ✓ Slightly Dropped ✓

- Table 1: Comparison of Different Methods. "Limited" indicates optimization within the Long-CoT distribution, restricting efficiency. "Broader" covers both Long- and Short-CoT, enabling shorter, more efficient responses. "Slightly dropped" means accuracy decreased by less than 3%, while "dropped" refers to a decrease greater than 3%.

distribution have struggled to maintain accuracy due to their inability to adapt adequate reasoning depth to problem complexity.

The finding mentioned in last section motivates us to address the efficiency challenge of Long-CoT models from a novel perspective: enabling the reasoning model to adaptively select an appropriate reasoning mode (long or short CoT) for different problems, and then generate a correct and concise CoT in the determined mode. Our proposed method (Ada-R1) differentiates itself by successfully achieving a broad optimization scope while incurring only a marginal performance decrement. This demonstrates a more favorable trade-off between efficiency and accuracy compared to existing broad-scope optimization techniques.

### 4 Bi-Level Adaptive Reasoning Optimization

#### 4.1 Problem Setup

We consider a LLM parameterized by θ and denoted as πθ. In the context of math problem solving, the LLM accepts a sequence x = [x1,...,xn], commonly termed as the problem, and then generate a corresponding solution y = [y1,...,ym]. Hence, the solution y is construed as a sample drawn from the conditional probability distribution πθ(·|x). The conditional probability distribution πθ(y|x) can be decomposed as follows:

m

πθ(yj|x,y<j). (1)

πθ(y|x) =

j=1

We consider two LLMs: one trained to generate long, reflective Chain-of-Thought (CoT) reasoning (Long-CoT model, denoted as θL) and the other trained for short and concise reasoning paths (ShortCoT model, denoted as θS). These two models are typically fine-tuned with different CoT and demonstrate distinct reasoning patterns.

#### 4.2 Method Overview

Our method consists of two stages, shown in Figure 2. First, we merge a Long-CoT model and a Short-CoT model to obtain a unified reasoning model capable of generating both types of reasoning paths. This allows exploration over a broader CoT distribution. In the second stage, we apply Bi-Level Preference Training: for group-level preference, the model learns to choose between long and short reasoning group based on the input; for instance-level preference, it learns to compress the reasoning path to improve efficiency within the chosen group determined by group-level preference.

#### 4.3 Stage I: Long-and-Short Reasoning Merge

To enable flexible reasoning behaviors within a single model, we first perform model merging with long and short models. We adopt a simple yet effective strategy of linearly merging their parameters. Given two models with parameters θL and θS, we compute the merged model as:

θH = αθL + (1 − α)θS, (2)

| |
|---|

[Figure 1]

[Figure 2]

Sample

𝑃𝑔𝑟𝑜𝑢𝑝

|𝜋𝜃𝐿|
|---|

|𝜋𝜃𝐿|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

Merge

Train

Bi-Level Optimization

|𝜋𝜃𝐻|
|---|

|𝜋𝜃𝐻∗|
|---|

|𝜋𝜃𝐻|
|---|

[Figure 6]

[Figure 7]

Sample

𝑃𝑖𝑛𝑠𝑡𝑎𝑛𝑐𝑒

|𝜋𝜃𝑆|
|---|

|𝜋𝜃𝑆|
|---|

Stage I

Stage II

- Figure 2: Pipeline of Ada-R1. At Stage I, we fused the models to obtain πθ

. In Stage II, we sample from both long and short models and then elicit the group-level and instance-level preference. After this, we optimize πθ

H

at both group and instance level to obtain a hybrid adaptive reasoning model.

H

where α ∈ [0,1] is a merging coefficient that balances the contribution from each model. The resulting hybrid reasoning model, πθ

, inherits the capacity to generate both long and short CoT depending on the input.

H

This merged model expands the diversity of the CoT distribution it can produce, laying the foundation for adaptive reasoning. By combining the strengths of both reasoning styles, it enables the model to potentially match different problem types with suitable reasoning strategies, which is key to improving efficiency in the next stage.

#### 4.4 Stage II: Bi-Level Preference Training

In this stage, we introduce a Bi-Level Preference Training strategy to fine-tune the model toward efficient reasoning. The core idea is to train the model to: (1) select the appropriate reasoning style (long or short) for each problem (group-level preference) and (2) further compress the reasoning within the determined chosen group (instance-level preference).

Group Labels. We define a group label g to denote the reasoning style of a response group. Let gL denote the long reasoning group and gS denote the short reasoning group. For a given input problem x, a generated resposne (solution) y belongs to one of the two groups. We use {yi}g=gL

to denote the set of K Long-CoT responses generated by the Long-CoT model θL, and {yj}g=gS

for the corresponding short responses from the Short-CoT model θS.

Group-Level Preference. For each math problem x in the dataset D, we sample K solutions from both the long and short reasoning models. Let {yiL}Ki=1 and {yjS}Kj=1 be the respective sample sets. We define the approximated accuracy expectation for each group as:

K

1 K

Eˆ[CL(x)] =

i=1

K

1 K

[Correct(yiL)], Eˆ[CS(x)] =

j=1

[Correct(yjS)], (3)

where [·] is the indicator function. Then we introduce a preference margin threshold ϵ > 0. The group-level preference for x is then determined as:

gL ≻ gS | x if Eˆ[CL(x)] − Eˆ[CS(x)] > ϵ, gS ≻ gL | x if Eˆ[CL(x)] − Eˆ[CS(x)] ≤ ϵ.

Given the group-level preference for an input x, we form training pairs from the Cartesian product of the two groups. For example, if gL ≻ gS | x, we construct the preference pairs as:

Pgroup(x) = (x,yiL,yjS) i ∈ [1,K],j ∈ [1,K] . (4)

From this set of pairs, we randomly sample a subset contain M1 pairs to construct DPO training tuples (x,yw,yl), where yw is the preferred (chosen) response and yl is the less preferred (rejected).

For all x ∈ D, we perform group-level preference assignment by comparing the sampled long and short responses as described above. These tuples are then aggregated into a new dataset Dgroup = {(x,yw,yl)}, which serves as supervision for optimizing the DPO objective at the group level.

Instance-Level Preference. Once the preferred group g∗ ∈ {gL,gS} is determined for a given x, we further construct instance-level preferences within that group to encourage more concise reasoning.

We compare response pairs (ya,yb) such that both belong to the same group (e.g., ya,yb ∈ {yiL}), and prefer the shortest correct response. For dispreferred samples, we select M2 longest responses. Formally, for each x ∈ D with preferred group g∗, we first identify the subset of correct responses {yi}correct ⊆ {yi}g=g∗. Among these, we select the shortest correct response as the preferred instance:

|y|.

yw = arg min

y∈{yi}correct

To construct instance-level preference pairs, we then select the M2 longest responses from the entire group {yi}g=g∗. Denote these as {yl

j}M

j=1. This yields a dataset of instance-level training tuples:

2

i}correctg=g∗|y|,yl ∈ argmaxy(M∈{2y)

Dinstance = (x,yw,yl) yw = argminy∈{y

|y|

i}g=g∗

These instance-level preferences encourage the model not only to reason correctly, but also to do so concisely within the preferred reasoning style.

We sample such intra-group pairs and use them as additional training data for DPO to encourage the model to favor more concise reasoning within each group.

Objective. Given collelcted preference datasets Dgroup and Dinstance sampled from p∗ which contains N preference pairs (x,yw,yl). With a parameter β controlling the deviation from the reference model pref, DPO optimize the model by:

(yl | x) πθ

(yw | x) πθ

πθ

πθ

E(x,y

H

H

(yw | x) − β log

w,yl)∼Dgroup∪Dinstance log σ β log

max

(yl | x)

πθH

ref

ref

### 5 Experiments

#### 5.1 Setup

Long-CoT Models. The long thought models we chosen for our experiment are DeepSeek-R1-DistillQwen-7B and DeepSeek-R1-Distill-Qwen-1.5B, which have demonstrated excellent performance on most math problem-solving tasks. For both models, we utilize full-parameter fine-tuning.

Short CoT Models. Since model merging requires Shot-CoT models, we face two issues with existing Shot-CoT models: (1) they often employ templates that differ from those used in LongCoT models; (2) they tend to exhibit substantial parameter deviations from the base model, which introduces instability during the merging process[16, 32]. To address these challenges, we fine-tune the Long-CoT models using a small number of short CoT examples to obtain the corresponding Shot-CoT models. This approach ensures consistency in template usage and maintains a closer parameter proximity between the two models.

Dataset. Following s1[33] and Light-R1[34], we construct a mixed training dataset to ensure coverage across mathematical problems of varying difficulty levels. Specifically, we combine GSM8K, MATH, and AIME datasets in a ratio of 1:3:1, resulting in a total of 2,500 diverse math problems.

Evaluation. We use the GSM8K test set, the MATH test set, and AIME25 as in-distribution evaluation data, while Olympiad[35] and Minerva[36] are employed as out-of-distribution test sets. For evaluation metrics, we consider both accuracy and sequence length. Additionally, we report the average accuracy degrade rate and the average length reduction rate across all test sets.

#### 5.2 Competitive Methods

DPO. DPO are widely used baselines in reasoning optimization area. Follwoing the setting of [31, 4], we choose shortest sample as chosen samples and longest sample as rejected sample.

- Table 2: Accuracy (shown above) and length (shown below) of models and methods on different benchmarks. Avg represents the change in length and accuracy compared to the Long model (+ for increase, - for decrease).

Bench

AIME25 MATH500 GSM8K Olympiad Minerva Avg.(%)

Model

7B Models Long(R1-distill)

38.3 90.2 88.9 54.4 35.7 -

(11005) (3534) (1014) (7492) (4533) Short

10.0 78.6 89.5 39.4 28.6 −19.97%

(957) (591) (272) (910) (579) (-84.57%) Merge

21.7 79.4 88.4 41.2 25.7 −18.63%

(9079) (916) (236) (3743) (1734) (-56.02%) DPO

35.8 89.4 86.0 55.2 35.6 -3.56% (9976) (2334) (360) (5309) (3281) (-33.26%) O1-Pruner

40.0 92.4 89.4 55.3 35.3 +2.48% (9353) (2212) (377) (5295) (3259) (-34.53%) CoT-Valve

22.5 78.6 87.9 39.6 29.4 −18.41%

(5024) (747) (235) (2313) (629) (-73.06%) Ada-R1(Ours)

35.8 90.2 90.3 52.4 34.1 -1.65% (8426) (1468) (260) (4889) (1647) (-50.93%)

1.5B Models Long(R1-distill)

23.3 81.0 80.9 41.6 26.1 -

(12307) (4416) (1481) (7687) (5789) Short

9.0 69.4 78.2 30.7 22.4 −26.34%

(1098) (740) (269) (1373) (725) (-85.15%) Merge

20.8 71.8 74.2 28.6 20.0 −10.12%

(9226) (1740) (251) (3767) (1399) (-59.10%) DPO

20.8 81.4 74.8 42.8 24.3 −5.93%

(10224) (3055) (374) (6319) (3905) (-34.57%) O1-Pruner

23.3 82.6 84.6 44.7 28.3 +2.18% (9496) (2782) (726) (5658) (3964) (-33.75%) CoT-Valve

14.2 69.6 76.3 28.7 19.5 −19.61%

(7744) (1299) (205) (3169) (867) (-67.52%) Ada-R1(Ours)

23.0 80.8 79.2 42.1 23.5 -1.21% (9516) (2455) (341) (5802) (3021) (-43.28%)

CoT-Valve. CoT-Valve enables dynamic control of Chain-of-Thought length using a single model by identifying and leveraging a controllable direction in the model’s parameter space to generate compressed CoT.

O1-Pruner. O1-Pruner is a method designed to reduce reasoning overhead while maintaining model accuracy. It begins by establishing a baseline through pre-sampling, and then applies reinforcement learning-based finetuning.

#### 5.3 Main Results

We can be seen from the Table 2 that: the Short and Merge models achieve the most significant length reduction compared to the Long Model. However, this efficiency gain is accompanied by a notable degradation in accuracy, exceeding 10 percentage points. Among the models that do not suffer significant accuracy degradation, our method achieves the best length reduction performance, reaching 50.93% for the 7B model and 43.28% for the 1.5B model. Compared to DPO, our approach demonstrates both more substantial length reduction and significantly less accuracy degradation. While O1-Pruner maintains high accuracy, its length reduction effect is considerably weaker than that of our method.

- Table 3: Ablation study of each component on several benchmarks, showing that the Merge + bi-level achieves the best trade-off, with a 52.08% average length reduction and a minimal 0.51% accuracy degradation compared to others.

Bench

AIME25 MATH500 GSM8K Avg.(%)

Model

38.3 90.2 88.9 -

Long(R1-distill)

(11005) (3534) (1014) Merge

21.7 79.4 88.4 −12.83%

(9079) (916) (236) (-56.10%) Merge + SFT

35.8 84.6 88.7 -3.82% (11222) (2314) (375) (-31.86%) Merge + group level

30.8 87.8 91.6 -3.31% (9049) (1565) (359) (-46.03%) Merge + bi level

35.8 90.2 90.3 -0.51% (8426) (1468) (260) (-52.08%)

#### 5.4 Ablation Study

To assess each component’s impact in our framework, we conduct an ablation study on AIME25, MATH500, and GSM8K. As shown in Table 3, the Merge model reduces average output length by 56.10%, but with a notable 12.83% drop in accuracy.

Supervised Fine-Tuning (SFT) on the merged model (using the chosen sample in our group level preference dataset), helps recover a significant portion of the lost accuracy, bringing the average degradation down to 3.82%. However, its average length reduction is less pronounced (31.86%) compared to the Merge model without further training.

Introducing the group-level preference training after merging (Merge + group level) yields better results than SFT. It achieves a higher average length reduction (46.03%) and a slightly better accuracy recovery, with only a 3.31% average degradation relative to the baseline. This indicates that training the model to select the appropriate reasoning style is effective in balancing efficiency and accuracy.

The full method (Merge + bi level), combining group and instance level preference training, offers the best trade-off: 52.08% length reduction with only 0.51% accuracy loss. This result highlights the complementary benefits of the bi-level training approach: the group level guides the model towards suitable reasoning styles, and the instance level further refines the chosen style by favoring concise and correct responses, leading to a highly efficient and accurate hybrid reasoning model.

### 6 Further Evaluation

#### 6.1 Thinking Ratio Study

To investigate the thinking characteristics of different models, we propose the "Thinking Ratio" metric. This metric is designed to detect whether a response constitutes a deep thinking (Long-CoT) sample. Long-CoT responses typically include unique keywords (e.g., ‘wait’, ‘recheck’). By detecting the presence of these keywords in a response, we can determine if it is a deep thinking sample. This detection method is more generalizable than relying solely on response length. We use a subset of Math Testset. Using the method described above, we analyzed the proportion of deep thinking samples for each model. Furthermore, for each category (thinking/non-thinking samples), we also calculated their accuracy.

The results are shown in Figure 3. The baseline Long-CoT model predominantly employs deep thinking (0.98), yielding high accuracy. In contrast, the Naive Merge model drastically shifts towards non-thinking responses (0.94) but suffers significant accuracy degradation on both thinking (0.68) and non-thinking (0.81) paths. DPO shows a moderate shift to non-thinking (0.34) while preserving accuracy. Our Ada-R1 model achieves a more significant shift towards non-thinking (0.72) than DPO, yet crucially maintains high accuracy for these dominant non-thinking responses (0.96), unlike

###### the Naive Merge. This demonstrates Ada-R1’s effective adaptation, utilizing efficient shorter paths without substantial accuracy loss.

Non-thinking COT Thinking COT Non-thinking COT ACC Thinking COT ACC

- 0.8
- 1

Proportions

0.6

0.4

0.2

0

AdaR1 R1-7B 7B-DPO Merging

- Figure 3: The proportion and accuracy of thinking and non-thinking in different methods, Ada-R1 can achieve a good balance and accuracy between thinking and non-thinking.

- 6.2 Adaptive Reasoning Study

This section evaluates the adaptive reasoning ability of Ada-R1 (7B) on the MATH dataset, which is divided into five difficulty levels (Level 1–5). We analyze both the model’s thinking ratio (Long-CoT usage) and its average accuracy across these levels. As shown in the left part of Figure 4, the thinking ratio increases significantly with task difficulty. Level 1 problems have the lowest Long-CoT usage, while Level 5 shows the highest, indicating that Ada-R1 adaptively chooses to think more on harder problems. In terms of accuracy (Figure 4, right), Ada-R1 achieves strong performance across difficulty levels. Its accuracy is comparable to that of a full Long-CoT model (Deepseek-R1-Qwen-7B-Distill) and consistently higher than the Short-CoT model, especially on Levels 3 to 5. These results support our hypothesis from Section 3: Ada-R1 can selectively apply Long-CoT when needed, achieving a better balance between accuracy and efficiency.

|0<br><br>0.2<br><br>0.4<br><br>0.6<br><br>0.8<br>1<br><br><br>Level 1 Level 2 Level 3 Level 4 Level 5<br><br>Proportions<br><br>Different Difficulty Levels<br><br>Thinking COT Non-Thinking COT<br><br>|
|---|

ons

|0.5<br><br>0.75<br><br>1<br><br><br>Level 1 Level 2 Level 3 Level 4 Level 5<br><br>Accuracy<br><br>Different Difficulty Levels<br><br>AdaR1 Long Short<br><br>|
|---|

Accuracy

- Figure 4: The ratio of thinking and non-thinking CoTs of Ada-R1-7B on different MATH levels (left) and the accuracy on different MATH levels of different models (right). As the difficulty increases, Ada-R1 is able to think more on harder problems and maintain higher accuracy.

### 7 Conclusion

In this paper, we demonstrate through empirical analysis that the benefits of Long-CoT reasoning vary significantly depending on the problem. Motivated by this, we propose a novel two-stage training framework for adaptive reasoning. Experiments show that model trained with our method can reason adaptively to different problems. And our method significantly reduces inference costs while preserving performance, highlighting the promise of adaptive strategies for optimizing reasoning efficiency in large language models.

### References

- [1] OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, 2024. [Accessed 19-09-2024].
- [2] DeepSeek-AI team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [3] Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through dense representations, 2024.
- [4] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2025.
- [5] Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Hu. Stop overthinking: A survey on efficient reasoning for large language models, 2025.
- [6] Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning, 2025.
- [7] Daman Arora and Andrea Zanette. Training language models to reason efficiently, 2025.
- [8] Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Length-compressible chain-of-thought tuning, 2025.
- [9] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.
- [10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.
- [11] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities, 2024.
- [12] Yue Zhou, Yi Chang, and Yuan Wu. Mixup model merge: Enhancing model merging performance through randomized linear interpolation, 2025.
- [13] Rishabh Bhardwaj, Do Duc Anh, and Soujanya Poria. Language models are homer simpson! safety re-alignment of fine-tuned language models through task arithmetic, 2024.
- [14] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch, 2024.
- [15] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models, 2023.
- [16] Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao. Adamerging: Adaptive model merging for multi-task learning, 2024.
- [17] Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young Yun. Self-training elicits concise reasoning in large language models, 2025.
- [18] Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms, 2025.
- [19] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2024.
- [20] Wang Yang, Xiang Yue, Vipin Chaudhary, and Xiaotian Han. Speculative thinking: Enhancing small-model reasoning with large model guidance at inference time, 2025.

- [21] Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. Lightthinker: Thinking step-by-step compression, 2025.
- [22] Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models, 2025.
- [23] Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, and Alane Suhr. Learning adaptive parallel reasoning with language models, 2025.
- [24] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking, 2025.
- [25] Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, and Yaoxue Zhang. Concise: Confidence-guided compression in step-by-step efficient reasoning, 2025.
- [26] Ren Zhuang, Ben Wang, and Shuifa Sun. Accelerating chain-of-thought reasoning: When goal-gradient importance meets dynamic skipping, 2025.
- [27] Junjie Yang, Ke Lin, and Xing Yu. Think when you need: Self-adaptive chain-of-thought learning, 2025.
- [28] Han Wu, Yuxuan Yao, Shuqi Liu, Zehua Liu, Xiaojin Fu, Xiongwei Han, Xing Li, Hui-Ling Zhen, Tao Zhong, and Mingxuan Yuan. Unlocking efficient long-to-short llm reasoning with model merging, 2025.
- [29] MAA. American invitational mathematics examination - aime. In American Invitational Mathematics Examination - AIME 2024, February 2024.
- [30] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.
- [31] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms, 2025.
- [32] Yuyan Zhou, Liang Song, Bingning Wang, and Weipeng Chen. Metagpt: Merging large language models using model exclusive task arithmetic, 2024.
- [33] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025.
- [34] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond, 2025.

- [35] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- [36] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022.

### A Training Details

For both models, we selected 2,500 problems from the mixed Mathematics as training data. For each problem, we sample 12 times. From each set of solutions, we randomly selected 2 solutions for training. After computing the rewards, we normalized the reward values. Both models are trained with 8 * A800-80G GPUs. The other hyperparameters used in the training process are presented in the table below.

Table 4: Hyperparameters for the Deepseek-Distill-1.5B and Deepseek-Distill-7B. Hyperparameter Deepseek-Distill-1.5B Deepseek-Distill-7B. cutoff_len 4096 4096 batch_size 32 32 learning_rate 5.0e-7 5.0e-7 num_train_epochs 2.0 2.0 lr_scheduler_type constant constant

M1 4 4 M2 2 2 beta 0.05 0.1

### B Further Evaluation of Different Methods

We further evaluate the performance and efficiency of different methods (Ada-R1, DPO, O1-Pruner) across varying levels of problem difficulty, as illustrated in Figure 5 and Figure 6. Figure Figure 5 presents the accuracy ratio of each method relative to a baseline model across different difficulty levels within the MATH dataset. The results indicate that while performance trends may vary, our proposed Ada-R1 method demonstrates strong robustness. Specifically, as the inherent difficulty of the mathematical problems increases, Ada-R1 is able to consistently maintain a high accuracy ratio.

Figure 6 show the ratio of average tokens consumed by each method to solve problems across the same difficulty spectrum. As expected, solving more difficult problems generally requires more reasoning steps and thus more tokens. However, Figure 2 reveals that Ada-R1 exhibits favorable token efficiency. Critically, when faced with increasing problem difficulty, Ada-R1 manages to solve these complex problems while utilizing relatively fewer tokens compared to other evaluated methods, showcasing its ability to achieve efficient reasoning even for demanding tasks.

Collectively, these figures highlight Ada-R1’s ability to strike a beneficial balance between accuracy and efficiency. It not only maintains high performance on challenging problems (Figure 5) but also does so in a computationally efficient manner, particularly evident in its lower token usage for difficult instances (Figure 6), addressing limitations observed in prior methods.

### C How Does Ada-R1 Work?

#### C.1 Early Mode Selection Assumption

While Ada-R1 significantly reduces inference cost by adaptively selecting a reasoning strategy during the inference stage, its design relies on an important assumption: the model determines the reasoning mode (Long-CoT or Short-CoT) immediately after receiving the problem input, without relying on any intermediate computation or external signals. In other words, the model is expected to assess the complexity of the problem and select an appropriate reasoning path before beginning the actual problem-solving process.

#### C.2 Visualization Setup

To investigate this question and better understand how Ada-R1 works, we design an experiment. We randomly select 500 problems from the training data and evaluate them using the 7B models (R1, and Ada-R1). For each problem, we extract the hidden states of the final token in the input sequence and use the last layer’s hidden states as the internal representation of the problem. Based on previously

|0.5<br><br>0.75<br><br>1<br><br><br>Level 1 Level 2 Level 3 Level 4 Level 5<br><br>Accuracy<br><br>Different Difficulty Levels<br><br>AdaR1 DPO O1-Pruner Long Short<br><br>|
|---|

Accuracy

- Figure 5: The ratio of accuracy at different MATH levels on different models. As the difficulty increases, Ada-R1 is able to maintain high accuracy.

|200<br><br>850<br><br>1500<br><br>2150<br><br>2800<br><br>3450<br><br>4100<br><br>4750<br><br>5400<br><br>Level 1 Level 2 Level 3 Level 4 Level 5<br><br>Tokens<br><br>Different Difficulty Levels<br><br>AdaR1 DPO O1-Pruner Long Short<br><br>|
|---|

- Figure 6: The ratio of average tokens on different models. As the difficulty increases, Ada-R1 is able to use relatively fewer tokens to solve difficult problems.

[Figure 8]

[Figure 9]

Figure 7: Visualization of R1 model Figure 8: Visualization of Ada-R1 model

computed group-level preferences (i.e., whether the problem should be solved using Long-CoT or Short-CoT), we assign a color label to each sample—red for problems requiring Long-CoT and blue

for those suitable for Short-CoT. We then apply t-SNE to project the high-dimensional hidden states into a two-dimensional space for visualization.

#### C.3 Ada-R1 Learns an Implicit Problem Classifier

From the visualization, we observe that after preference-based training, Ada-R1 is able to partially separate problems that require Long-CoT from those that do not, based solely on their internal representations. This suggests that the model learns to encode problem complexity in representation space, enabling early and efficient reasoning mode selection. Such a capability underpins the effectiveness of Ada-R1: by making an informed decision on the reasoning strategy at the problem stage, the model avoids unnecessary computation for simpler problems while retaining full reasoning capacity for more complex ones.

- D Case Study We select two cases (9 and 10) of R1 and Ada-R1 for better understanding of our proposed method.
- E Limitations

While our proposed adaptive hybrid reasoning framework shows strong performance on benchmark datasets, its applicability in real-world scenarios presents new challenges. In particular, real-world tasks often involve diverse input distributions, domain-specific knowledge, and evolving requirements that differ from curated academic datasets such as MATH or GSM8K. The current design assumes clear patterns in problem complexity that can guide reasoning strategy selection; however, in practical applications, this complexity may be harder to estimate reliably without additional context or metadata. We view this as a promising direction for future research to bridge the gap between controlled evaluation settings and real-world reasoning applications.

Question: Convert the point (0,3) in rectangular coordinates to polar coordinates. Enter your answer in the form (r,θ), where r > 0 and 0 ≤ θ < 2π. Let’s think step by

step and output the final answer within boxed{}. Ground Truth: (3, π2)

Ada-R1: To convert the rectangular coordinate (0,3) to polar coordinates, we use the formulas: r = x2 + y2 and θ = arctan x y . For the point (0,3): - r = √02 + 32 = √9 = 3 - Since the point lies on the positive y-axis, the angle θ is π2. Thus, the polar coordinates are (3, π2). Final Answer:

|(3,<br><br>π 2<br><br>)|
|---|

##### ✓

R1: Okay, so I need to convert the rectangular coordinate (0,3) to polar coordinates. Hmm, polar coordinates are represented as (r,θ), where r is the radius (distance from the origin) and θ is the angle made with the positive x-axis. Let me recall the formulas to convert from rectangular to polar coordinates. First, I remember that the radius r can be found using the Pythagorean theorem. Since in rectangular coordinates, the point is (x,y), the distance from the origin is r = x2 + y2. So

in√0this+ 9case,= √x9is=03and. Okay,y is that3. Pluggingseems straightforward.that into the formula, r should be √02 + 32 which is Now, the angle θ. I know that θ is calculated using the arctangent function because tanθ = xy. But wait, in this case, x is 0. Hmm, arctangent of y0. But dividing by zero is undefined. I need to think about what angle θ corresponds to when x is 0. In rectangular coordinates, when x is 0, the point lies on the y-axis. Since y is positive here (y = 3), the point is on the positive y-axis. So, in terms of polar coordinates, the

angle θ should be the angle that points directly upwards along the y-axis. I think that’s π2 radians, which is 90 degrees. Because starting from the positive x-axis, if we rotate π2 radians counterclockwise, we end up pointing straight up along the positive y-axis.

Let me double-check that. If θ is π2, then converting back to rectangular coordinates, x should be r cosθ = 3cos(π2) and y should be r sinθ = 3sin(π2). cos(π2) is 0 and sin(π2) is 1, so that gives x = 0 and y = 3, which is exactly the point we started with. So that makes sense.

Wait, but could θ also be 32π? No, because 32π is pointing downward along the negative y-axis, which would give us y = −3, not 3. So θ has to be in the range where the point is on the

positive y-axis, which is π2. Just to make sure, let’s think about the unit circle. At θ = 0, we’re on the positive x-axis; at θ = π2, positive y-axis; at θ = π, negative x-axis; at θ = 32π, negative y-axis. So yeah, θ = π2 is correct for the positive y-axis. So, putting it all together, r is 3 and θ is π2. Therefore, the polar coordinates should be (3, π2). Let me just write that out to make sure I didn’t make any mistakes. Wait, another thought: sometimes people use different ranges for θ, like from −π to π. But in this case, the problem specifies that θ should be between 0 and 2π, so π2 is within that range. So, I don’t need to adjust θ or anything. Just to recap: the point is (0,3) in rectangular, so x = 0, y = 3. Calculating r: √02 + 32 = 3. Calculating θ: since x = 0 and y is positive, θ is π2. So, (3, π2) is the polar coordinate. I think that’s solid. Let me visualize it too. If I plot (0,3) on the rectangular coordinate system, it’s on the y-axis, three units up. In polar coordinates, that’s a radius of 3 and an angle of π2 from the positive x-axis. Yep, that lines up perfectly. Just to add another check, let me compute the sine and cosine with θ = π2. cos(π2) is 0, so x = 3 ∗ 0 = 0, which matches. sin(π2) is 1, so y = 3 ∗ 1 = 3, which also matches. Perfect, no discrepancies there. So, I think I’m confident that the polar coordinates are (3, π2). I don’t see any mistakes in my reasoning, and all the checks seem to confirm that this is the correct answer. Final Answer: The polar coordinates are

|(3,<br><br>π 2<br><br>)|
|---|

##### . ✓

- Figure 9: Ada-R1 can infer answers to simple problems with a small amount of CoT, whereas the R1 model requires a long CoT.

#### Question: Define p = ∞k=1 k12 and q = ∞k=1 k13 . Find a way to write ∞ j=1

∞ k=1

#### (j+k)3 in terms of p and q. Let’s think step by step and output the final answer within boxed{}. Ground Truth: p − q

1

Ada-R1: Okay, so I have this problem where I need to express a double sum in terms of two known series, p and q. Let me first write down what’s given. We have: p = ∞k=1 k12

and q = ∞k=1 k13. And the expression we need to evaluate is ∞j=1 ∞k=1 (j+1k)3 . Hmm, so this is a double sum over j and k starting from 1 to infinity, and each term is 1 divided by (j + k) cubed. I need to find a way to write this in terms of p and q. Let me think about how to approach this. Maybe I can change variables to simplify the expression. If I let n = j + k, then for each fixed n, j can range from 1 to n-1, and k would be n - j. So, for each n, the number of terms where j + k = n is n - 1. So, maybe I can rewrite the double sum as a single sum over n, and for each n, sum over j from 1 to n-1. Let me write that

down: ∞j=1 ∞k=1 (j+1k)3 = ∞n=2 j n=1−1 n13 . Yes, that seems right. Because when j + k = n, j can go from 1 to n-1, and for each j, k is determined as n - j. So, the inner sum is just

adding n13 (n-1) times. So, the double sum becomes: ∞n=2 n−1

n3 . Simplify that expression:

∞ n=2

n3 = ∞n=2 n 12 − n13 . Yes, because n−1

n−1

n3 = nn3 − n13 = n12 − n13 . So now, the sum can be split into two separate sums: ∞n=2 n12 − ∞n=2

n3 . But notice that these sums are just

1

- p and q minus their first term. Let’s write that: The first sum is: ∞n=2 n12 = p − 112 = p − 1. Similarly, the second sum is: ∞n=2 n13 = q − 113 = q − 1. Therefore, the original ... [OMITTED] Thus, the original double sum becomes: (p−1)−(q−1) = p−1−q+1 = p−q. Therefore, the final answer is:

|p − q|
|---|

. ✓

R1: Okay, so I have this problem where I need to express a double sum in terms of two known series, p and q. Let me first write down what’s given. We have: p = ∞k=1 k12 and

- q = ∞k=1 k13. And the expression we need to evaluate is ∞j=1 ∞k=1 (j+1k)3. Hmm, so this is a double infinite sum where both j and k start from 1 and go to infinity, and each term is 1 over (j + k) cubed. I need to find a way to write this in terms of p and q. Let me think about how to approach this. Maybe I can find a substitution or change variables to simplify the expression. Since the denominator is (j +k)3, maybe if I let n = j +k, then I can rewrite the sum in terms of n and another variable. Let’s try that. Let me set n = j + k. Then, for each fixed n, j can range from 1 to n - 1, because k would be n - j, which also needs to be at least 1.

So, for each n, j goes from 1 to n - 1, and k = n - j. So, substituting, the double sum becomes:

∞ n=2

n−1 j=1

n3 . Wait, that seems promising. Because now for each n, we have a term n13 multiplied by the number of j’s, which is (n - 1). So, the inner sum is just n−1

1

n3 . Therefore, the entire expression becomes: ∞n=2 n−1

n3 . Simplify that: ∞n=2 n 12 − n13 . Which is:

∞ n=2

n3 . But I know that: ∞n=1 n12 = p and ∞n=1 n13 = q. So, subtracting

n2 − ∞n=2

1

1

the first term from both series: ∞n=2 n12 = p − 1 and ∞n=2 n13 = q − 1. Therefore, substituting back into the expression: (p − 1) − (q − 1) = p − 1 − q + 1 = p − q. Wait, so

the entire double sum simplifies to p - q? Let me verify my steps to make sure I didn’t make a mistake. 1. I started by changing variables, setting n = j + k. Then for each n, j goes from 1 to n - 1, so k = n - j also goes from 1 to n - 1. That seems correct. 2. Then, I rewrote the double sum as a single sum over n, with each term being n−1

n3 . That makes sense because for each n,

there are (n - 1) terms contributing n13 each. 3. Then, I split the fraction into n12 − n13 . That’s a valid step because n−1

n3 = nn3 − n13 = n12 − n13. 4. Then, I separated the sums into two infinite series starting from n = 2, which is correct because the original sum starts from n = 2. 5. Finally, I expressed the sums in terms of p and q by subtracting the n=1 term from each series. That’s correct because the original p and q start from n=1. So, putting it all together... [OMITTED]

✓

Therefore, the final answer is:

|p − q|
|---|

- Figure 10: Ada-R1 can infer answers to difficult problems with relatively few long CoT, whereas the R1 model requires longer CoT.

