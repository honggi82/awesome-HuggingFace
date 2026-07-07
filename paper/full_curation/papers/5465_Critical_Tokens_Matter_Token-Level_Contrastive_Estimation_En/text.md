## Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability

Zicheng Lin∗,1,2 , Tian Liang∗1 , Jiahao Xu∗1 , Qiuzhi Liu1 , Xing Wang1 , Ruilin Luo2 , Chufan Shi2 , Siheng Li2 , Yujiu Yang†2 , and Zhaopeng Tu†1

- 1Tencent AI Lab
- 2Tsinghua University

# arXiv:2411.19943v3[cs.CL]13Jan2025

✘ ✘ ✘

$125 initially. She still owes $125 * 0.75 = $93.75. The answer is $93.75.

owed

125 dollars. 75% is 3/4. 3/4 of 125 is 3/4 x 125 = 93.75. The answer is 93.75.

Question: Mathilda is determined to pay back the money she owes a friend so she decides to pay an initial installment of $125. If she still has 75% left to pay, how much did she owe originally?

125 dollars initially. If she still owes 75% of that, that is 0.75 x 125 = 93.75 dollars. The answer is 93.75.

Response: Let’s think step by step. She owed $125 initially, She still has 75% left to pay, So she owes $125 * 0.75 = $93.75. The answer is $93.75. paid

an initial installment of $125, which represents 25% of the total amount she owed. So the total amount she owes is: 125 / 0.25 = 500. The answer is 500.

✔

125 dollars. She owes 75% of the original amount. 100% minus 75% is 25%. 125 / 0.25 is 500. The answer is 500.

✔ ✔

$125 as an initial installment with 75% left to pay. So, the total owed is 125 wes is: 125 / 0.25 = 500. The answer is 500.

Figure 1: An illustration of the critical token “owed” shows that it fails to lead to the correct answer in any case. Replacing it with an alternative can significantly increase model accuracy.

#### Abstract

Mathematical reasoning tasks pose significant challenges for large language models (LLMs) because they require precise logical deduction and sequence analysis. In this work, we introduce the concept of critical tokens – elements within reasoning trajectories that significantly influence incorrect outcomes. We present a novel framework for identifying these tokens through rollout sampling and demonstrate their substantial divergence from traditional error tokens. Through extensive experiments on datasets such as GSM8K and MATH500, we show that identifying and replacing critical tokens significantly improves model accuracy. We propose an efficient methodology for pinpointing these tokens in large-scale datasets using contrastive estimation and extend this framework to enhance model training processes with direct preference optimization (DPO). Experimental results on GSM8K and MATH500 benchmarks with the widely used models Llama-3 (8B and 70B) and Deepseek-math (7B) demonstrate the effectiveness of the proposed approach, cDPO. Our results underscore the potential of leveraging critical tokens to reduce errors in reasoning tasks, advancing the development of AI systems capable of robust logical deduction. Our code, annotated datasets, and trained models are available at https://github.com/chenzhiling9954/Critical-Tokens-Matter to support and encourage future research in this promising field.

∗Equal Contribution. The work was done when Zicheng was interning at Tencent AI Lab. †Correspondence to: Zhaopeng Tu <zptu@tencent.com> and Yujiu Yang <yang.yujiu@sz.tsinghua.edu.cn>.

#### 1 Introduction

In the domain of artificial intelligence, mathematical reasoning tasks are seen as a crucible for evaluating the proficiency of large language models (LLMs) (Cobbe et al., 2021; Hendrycks et al., 2021; Yuan et al., 2023; Ahn et al., 2024; Yu et al., 2023; Collins et al., 2024). These tasks necessitate logical and sequential deduction to derive solutions, making them challenging for models trained primarily on general language processing. The chain of thought (COT) method (Wei et al., 2022) has significantly improved the reasoning capability of LLMs by employing a series of intermediate reasoning steps, or reasoning trajectories. Prior research has categorized trajectory errors based on modifications required to correct the COT, such as calculator errors, with the objective of identifying avenues for model improvement (Wei et al., 2022; Wang et al., 2023).

Despite these advancements, the token-level discrepancies within mathematical reasoning contexts has not been systematically explored. Our study seeks to bridge this gap by introducing a novel framework for identifying and quantifying the impact of critical tokens on model accuracy. We define critical tokens in mathematical reasoning as crucial components within an incorrect trajectory that significantly alter the final outcome. We utilize rollout sampling to identify tokens that substantially influence the correctness of reasoning trajectories. Our findings reveal that critical tokens often diverge from human-annotated error tokens, highlighting their unique role in disrupting logical coherence and computational accuracy. By analyzing the characteristics of critical tokens through word type and positional analysis, we provide novel insights into their nature and influence mechanisms. Furthermore, altering or manipulating a single critical token in incorrect trajectories can significantly enhance accuracy, underscoring their pivotal role in error mitigation.

Building on these insights, we illustrate how critical tokens can enhance reasoning capabilities within Direct Preference Optimization (DPO), a commonly used reinforcement learning algorithm. Although DPO proves effective for general tasks, it encounters difficulties in mathematical reasoning because it may reduce the generation likelihood of positive examples due to lexical similarities with negative examples. Our proposed method, cDPO, addresses this issue by targeting critical tokens unique to negative examples, thereby improving the model’s ability to differentiate between positive and negative instances. cDPO involves the efficient identification and penalization of critical tokens predominantly found in negative examples, refining the model’s learning process and enhancing its understanding of positive outcomes. Experimental results on GSM8K and MATH500 benchmarks using widely recognized models like Llama-3 (8B and 70B) and Deepseek-math (7B) demonstrate that our approach surpasses strong DPO baselines, such as TokenDPO (Zeng et al., 2024) and RPO (Liu et al., 2024; Pang et al., 2024), across all evaluated scenarios.

In summary, our contributions are three-fold:

- • We introduce the concept of critical tokens in mathematical reasoning tasks and empirically validate their existence through extensive rollout sampling, distinguishing them from traditional error tokens.
- • We propose an efficient approach using contrastive estimation to practically identify critical tokens in large-scale training data, requiring only as little as 0.002% of the computational cost of rollout sampling for GSM8K.
- • We develop cDPO, an innovative approach that leverages critical tokens within DPO, enhancing the algorithm’s ability to distinguish between positive and negative examples in mathematical reasoning.

#### 2 Critical Tokens in Mathematic Reasoning

In this section, we explore the concept of “critical tokens” in mathematical reasoning tasks and their impact on model accuracy. We begin by defining critical tokens as pivotal points in incorrect reasoning trajectories that significantly influence outcomes using the example in Figure 1. To validate the presence of critical tokens, we perform rollout sampling, identifying tokens with zero correctness

scores that meet specific conditions in sequence analysis. Our findings reveal that critical tokens differ from human-annotated error tokens in a substantial proportion of cases, emphasizing their unique role in reasoning failure (Table 1). Further experimentation shows that replacing critical tokens boosts model accuracy significantly, highlighting their importance in error reduction (Figure 2). We conclude with an analysis of critical tokens based on word types and positional attributes (Table 2), providing insights into their characteristics.

Intuition Mathematical reasoning tasks require logical and sequential deduction to find solutions. We have observed that within incorrect reasoning trajectories, certain tokens are pivotal in leading to incorrect outcomes. These tokens disrupt the logical flow, misrepresent relationships, or introduce computational errors, thus significantly affecting the final result. Unlike other tokens that may have negligible effects on the reasoning process, these “critical tokens” are crucial points of failure. Identifying these tokens is essential because avoiding or correcting them can often result in correct outcomes, even within an incorrect trajectory. As illustrated in Figure 1, the token “owed” is predominantly responsible for incorrect reasoning trajectories as it misguides the logical deduction process. In contrast, prompting the model to decode alternative tokens like “paid” significantly increases the likelihood of producing a correct final result.

Empirical Validation with Rollout Sampling To empirically validate the existence of critical tokens, we conducted 64 rollout samplings for each token within an incorrect trajectory. We then calculated a score for each token, based on the correctness ratio of the generated completions, to quantify its influence on the trajectory. The first token that meets the following two conditions is identified as the critical token:

- • The token’s correctness score is 0;
- • The correctness scores of all subsequent tokens are below 5%.

We analyzed 100 incorrect trajectories generated by Llama3-8B-base, randomly selected from the MATH training dataset, and successfully identified the critical token in each trajectory. In addition, by examining 100 random incorrect trajectories from the GSM8K training data, we identified the critical token in 99 cases out of 100. For the one outlier case, we identified a critical token that only satisfied the first condition. These results demonstrate the existence of critical tokens.

Table 1: The ratio of incorrect trajectories where critical tokens are different from error tokens across to the error types.

GSM8K Training Data MATH500 Training Data #Count Critical != Error #Count Critical != Error

Error Types

Calculation Error 60 56.7% 54 79.6% One step Missing 17 72.7% 8 87.5% Semantic Misund. 22 82.3% 17 100.0%

Degeneration 1 100.0% 21 95.2% Total 100 65.0% 100 87.0%

Critical Tokens Are Not Necessarily Error Tokens Researchers might hypothesize that critical tokens tend to coincide with error tokens, given their definition (i.e., the correctness ratio of rollout samplings is 0). However, Table 1 demonstrates that critical tokens frequently differ from humanannotated error tokens across various error types (Wei et al., 2022; Wang et al., 2023).

In the GSM8K training dataset, 65% of the critical tokens do not match the error tokens, while this disparity increases to 87% in the MATH500 training dataset. This variance is further nuanced when examining specific error categories. For example, in the GSM8K data, calculation errors show a 56.7% mismatch, which suggests that nearly half the time the critical tokens identified do not

correspond to the actual error tokens. Errors due to one-step missing demonstrate a higher 72.7% discrepancy, and semantic misunderstandings show an even greater divergence of 82.3%. Notably, degeneration errors—though based on a single occurrence—exhibit a complete 100% discrepancy with error tokens.

In the MATH500 dataset, a similar pattern is observed. Calculation errors exhibit a 79.6% discrepancy, one-step missing errors an 87.5% mismatch, semantic misunderstandings a 100% divergence, and degeneration errors a significant 95.2% discrepancy. The results in MATH500, particularly the large differences in degeneration errors, underscore the complexity involved in high-precision domains.

These findings underscore a critical insight: while critical tokens are valuable for flagging potential issues in a trajectory, there isn’t always a direct correlation with human-annotated error tokens. This divergence emphasizes the intricate nature of error detection and correction in algorithmic analyses, suggesting that critical tokens capture a broader context of underlying issues, possibly before an error becomes evident.

Table 2: Analysis of identified critical tokens.

Word Types Relative Positions Function Content Number Operator Punct. Before After

Error Types

100 Instances from GSM8K Training Data Calculation Error 10 14 19 10 7 17 17 One step Missing 5 2 9 0 1 9 5

- Semantic Misund. 3 6 3 3 7 7 9 Degeneration 0 0 1 0 0 0 1 Total 18 22 32 13 15 33 32

100 Instances from MATH500 Training Data Calculation Error 10 10 15 17 2 30 13 One step Missing 3 0 2 3 0 6 1

- Semantic Misund. 4 3 2 6 2 10 7 Degeneration 6 3 6 5 1 10 10 Total 23 16 25 31 5 56 31

Analysis of Critical Tokens We analyze critical tokens from the following perspectives, as detailed in Table 2:

- • Word Types: Tokens are categorized into five types based on their linguistic roles: function words, content words, and numbers. Additionally, operators (e.g., mathematical symbols) and punctuation marks are identified. Our analysis reveals differing patterns across error types and datasets, highlighting the complexity of math reasoning tasks. In the GSM8K dataset, calculation errors predominantly occur with numbers (19 instances) and are fairly distributed among other word types. On the other hand, one step missing errors primarily involve numbers and function words. This suggests an emphasis on numerical manipulation where precision in number usage and function words indicating operations are critical. The MATH500 dataset follows a somewhat similar trend, though it exhibits a higher occurrence of operator-related calculation errors (17 instances), indicating a more frequent use of complex mathematical operations in this dataset. This accentuates the need for careful handling of operators in mathematical computations.
- • Relative Positions: The position of critical tokens is analyzed relative to the corresponding error tokens, categorizing them as occurring before or after the error tokens. If the critical token is the error token itself, it is not counted. In GSM8K, critical tokens are almost evenly distributed: 33 occur before and 32 after the error. However, in MATH500, more critical tokens occur before error tokens, suggesting that critical tokens capture a broader context of underlying issues in the complex MATH500 problems, potentially before an error becomes evident.

1000

Tokens We investigate tokens by replacing them

100%

|Impact of Critical 901 the effect of critical|
|---|
|with alternative tokens|
|Specifically, let ti 451be a trajectory, and let T<i =|
|the preceding226tokens.218|
|7 7 16 50 45 7 7 5 7<br><br>ples based on two late the Pass@k metric:|

|91.0% 85.3%<br><br>76.3%<br><br>64.7%<br><br>52.3%<br><br>40.5%<br><br>29.9%<br><br>0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0%<br><br>77.8%<br><br>66.2%<br><br>54.1%<br><br>42.0%<br><br>30.5% MATH500: w/o Critical Tokens<br><br><br>MATH500: w/ Critical Tokens<br><br>GSM8K: w/o Critical Tokens<br><br>GSM8k: w/ Critical Tokens|
|---|

GeneratedTokens

800

%

during model decoding. critical token in a given

80%

600

{t1, . . . , ti−1} represent

400

60%

Pass@k

We perform k rollout samdifferent prefixes and calcu-

200

40%

- 29

0.

95.0% 88.0%

- 30

0

GPT-4o-mini

- o1-Preview
- o1-mini

GPT-4o

Qwen2.5-72B

Qwen2.5-Math-72B

Qwen2.5-Math-7B

GeminiPro

DeepSeek-R1

Claude-3.5

LLaMa3.3-70B

DeepSeek-Math-7B

QwQ-32B-Preview

20%

###### • w/ critical tokens: The prefix is

{t1, . . . , ti−1, ti}. Based on the definition of a critical token, all Pass@k scores are zero.

0%

0.0%1 0.0%2 0.0%4 0.0%8 0.0%16 0.0%32 0.0%64

###### • w/o critical tokens: The prefix is {t1, . . . , ti−1}, excluding the critical to-

Number of Samples (k)

ken ti. We force the model to decode an alternative token at the same position, using the model’s probability distribution while masking out the critical token. This substitution allows us to explore if it leads to improved outcomes in model predictions.

Figure 2: Impact of critical tokens on reasoning accuracy. Replacing critical tokens with alternatives (“w/o Critical Tokens”) can significantly increase model accuracy on both GSM8K and MATH500, highlighting the importance of these tokens.

- Figure 2 displays the results on 100 instances sampled individually from the GSM8K and MATH500 training datasets. By replacing critical tokens with alternative tokens, we observed a significant improvement in the model’s performance on these datasets, with Pass@1 accuracy reaching approximately 30% and Pass@64 increasing to over 90%. These results underscore the crucial role that critical tokens play as potential stumbling blocks in the reasoning process. By preventing the model from proceeding with these critical tokens and suggesting alternatives, we enable a higher likelihood of reaching accurate conclusions.

The findings emphasize the importance of understanding and manipulating critical tokens to enhance model performance, especially in complex reasoning tasks. By identifying these critical tokens in reasoning trajectories, we can mitigate the risk of errors, thereby significantly improving the model’s effectiveness. Such insights could be instrumental in refining training processes and increasing the reliability of AI systems in real-world applications.

#### 3 Enhancing Reasoning Capability with Critical Tokens

In this section, we demonstrate how reasoning capabilities, trained with commonly-used DPO, can be enhanced by using critical tokens.

Despite the success of DPO in general instruction tuning tasks, challenges persist when it is applied to reasoning and mathematical tasks. Studies have delved into this issue from an optimization perspective, identifying that these algorithms often diminish the generation likelihood of positive examples in reasoning tasks due to the lexical similarity between positive and negative examples. This overlap can lead to a situation where the model struggles to effectively prioritize and generate the correct trajectory during reasoning or mathematical problem-solving tasks. As a result, the adopted approach is to optimize preferences while ensuring that high generation likelihoods are maintained exclusively for positive examples (Liu et al., 2024; Pang et al., 2024; Pal et al., 2024; Feng et al., 2024). However, these approaches still attribute high likelihood to tokens that appear in both negative and positive examples, thereby failing to adequately distinguish between truly beneficial features and those that are merely ubiquitous across positive and negative examples.

In this work, we mitigate this problem by leveraging critical tokens that occur only in negative examples. By focusing on these critical tokens, we aim to enhance the model’s ability to effectively differentiate between positive and negative examples. Our approach involves the identification and penalization of critical tokens that are prevalent exclusively in negative examples. This allows us

|Base Model<br><br>[Figure 7]|
|---|

|𝑫 = {(𝒙𝒊, 𝒚𝒊)}𝒊=𝟏𝑴|
|---|

[Figure 8]

Sampling with Correctness Validation

She owed $125 initially…… The answer is $93.75.

|𝑫𝒑 = {(𝒙𝒊, {𝒚𝒊,𝒋𝒑 }𝒋=𝟏𝒌 )}𝒊=𝟏𝑴|
|---|

|𝑫𝒏 = {(𝒙𝒊, {𝒚𝒊,𝒋𝒏 }𝒋=𝒌+𝟏𝑵 )}𝒊=𝟏𝑴|
|---|

|Negative Model<br><br>[Figure 9]|
|---|

|Positive Model<br><br>[Figure 10]|
|---|

SFT SFT

Prediction

|Positive Model<br><br>[Figure 11]|
|---|

|Negative Model<br><br>[Figure 12]|
|---|

[Figure 13]

[Figure 14]

Contrastive Estimation

[Figure 15]

[Figure 16]

|Base Model<br><br>[Figure 17]|
|---|

|𝑫 = {(𝒙𝒊,𝒚𝒊p, 𝒚𝒊𝒏, 𝒔𝒊𝒏)}𝒊=𝟏𝑴|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

cDPO

|Trained Model<br><br>[Figure 21]|
|---|

- Figure 3: Pipeline of the proposed cDPO that involves the efficient identification and penalization of critical tokens that are prevalent exclusively in negative examples.

to adjust the model’s learning process by reducing the likelihood of these negative-specific tokens, thereby refining the model’s understanding of what constitutes a positive outcome. By explicitly incorporating a mechanism that penalizes frequent negative-example tokens, we ensure that the model learns to prioritize features that truly contribute to successful task completion.

###### 3.1 Efficient Identification of Critical Tokens

While it is straightforward to identify critical tokens using rollout sampling as described in Section 2, such methods incur prohibitively high sampling costs and face significant scalability challenges. Moreover, existing methods (Guo et al., 2023; Yoon et al., 2024) depend on external models for token-level annotations, which, although providing effective supervision signals, are costly and limited by the capabilities of the external models.

To efficiently identify critical tokens, we propose a method called contrastive estimation, which leverages models trained to learn patterns from both correct and incorrect reasoning trajectories.

- Figure 3 depicts the framework. By comparing the token-level likelihoods produced by two separately trained models, contrastive estimation can effectively pinpoint critical tokens contributing to incorrect outcomes. The contrastive estimation probability naturally highlights tokens (e.g., “owed”) that lead to incorrect reasoning outcomes. We provide additional details throughout the remainder of this section.

Training Positive and Negative Models To implement the contrastive estimation, we need to develop models that can effectively estimate a wide range of both correct and incorrect reasoning distributions. To this end, we collect a wide range of reasoning trajectories based on the sampling

strategy: given a dataset of M instances D = {(xi, yi)}iM=1, we utilize a pre-trained LLM to decode reasoning trajectories with N times sampling. Then, we verify the outcome results based on the

golden labels yi, which yields ki positive reasoning trajectories and N − ki negative reasoning

trajectories, which is denoted as:

##### Dp = {(xi, {yip,j}kj=1)}iM=1 Dn = {(xi, {yin,j}Nj=k+1)}iM=1

For training the positive model, we randomly selected a single correct trajectory because we expect the model to develop decisiveness using its own accurate reasoning paths. For training the negative model, we chose the incorrect trajectories that most frequently occur and account for 50% of all incorrect cases. This approach ensures both variety and representativeness, allowing us to accurately identify critical tokens. For instance, if there are 10 incorrect trajectories comprising 3 cases with the incorrect answer a, 2 cases with the incorrect answer b, and other cases with answers {c, d, e, f, g} occurring only once each, we would randomly select one incorrect trajectory from those with answers a and b, as they appear in 5 cases in total. Finally, we train the negative model on this example using two incorrect trajectories: one with answer a and the other with b.

Contrastive Estimation With both the positive model and the negative model available, we can automatically annotate the likelihood of each token in an incorrect trajectory being a critical token using contrastive estimation. Let x be a query, and yn = {y1, . . . , yt, . . . , yT} be a negative example of length T used in DPO training. We compute the likelihood of token yt being a critical token, denoted as st, with the following equation:

log st = (1 + β) log Pp(yt|x, y<t) − β log Pn(yt|x, y<t) − log Z (1) Here, β is a scaling hyperparameter, while Pp(·) and Pn(·) represent the probabilities from the positive and negative models, respectively. The term log Z is the partition function used in the softmax computation. A low st indicates a low likelihood under the correct pattern and a high likelihood under the incorrect pattern, signaling the presence of critical tokens.

Distribution Analysis of Contrastive Estimation We demonstrate that contrastive estimation does not fundamentally alter the nature of the trajectory distribution. According to (Guo et al., 2023; Lambert et al., 2024), the trajectory distribution can be modeled as Gaussian distributions based on correctness. Consequently, we define the probability density functions for the correct and incorrect distributions, denoted as Pp and Pn respectively, as follows:

(x − µp)2 2σ2

1 √2πσ

Pp(x) =

, (2)

exp −

(x − µn)2 2σ2

1 √2πσ

Pn(x) =

, (3)

exp −

where the means satisfy µp > µn, and both distributions share the same standard deviation σ, facilitating a straightforward comparison between them.

Therefore, according to Eq. 1, the probability density function Pce of the CE distribution can be calculated as follows:

log(Pce(x)) = (1 + β) log(Pp(x)) − β log(Pn(x)) − log(Z1), (4) where Z1 is the partition function. Substituting the definitions of Pp(x) and Pn(x), we obtain:

##### (1 + β)(x − µp)2 − β(x − µn)2

1

log(Pce(x)) = log

2σ2 − log(Z1). (5) Thus, the CE distribution Pce is:

√2πσ −

(1 + β)(x − µp)2 − β(x − µn)2 2σ2

1 √2πσ

1 Z1

Pce(x) =

. (6)

exp −

The term (1 + β)(x − µp)2 − β(x − µn)2 can be expressed as:

(1 + β)(x − µp)2 − β(x − µn)2 = (x − µce)2 + Z3, (7) where µce = µp + β(µp − µn), and Z3 is a constant independent of x. Substituting this result into (6) gives:

(x − µce)2 2σ2

1 Z1

1 √2πσ

Z3 2σ2

Pce(x) =

exp −

exp −

, (8)

where Z1 = exp −2Zσ32 . Finally, the CE distribution can be written as: Pce(x) =

(x − µce)2 2σ2

1 √2πσ

exp −

. (9)

Hence, Pce(x) is also a Gaussian distribution with mean µce = µp + β(µp − µn) and variance σ2.

Efficiency Analysis of Contrastive Estimation To evaluate the computational efficiency of contrastive estimation compared to rollout sampling, we estimate the number of forward passes required. Rollout sampling, used to identify critical tokens, incurs substantial inference costs as it relies on sampling from the base model. For GSM8K, obtaining critical tokens through rollout sampling for 100 incorrect examples (64 samples per token) results in an average of 581,425 additional tokens per response (7,613,942 tokens for MATH). Therefore, for n examples, rollout sampling requires approximately 581,425 × n forward passes for GSM8K.

In contrast, contrastive estimation involves both training and inference costs. On GSM8K, the dataset for training the positive and negative models contains 26,131 examples (68,391 examples for MATH), and SFT on this dataset requires approximately 3 × 26,131 = 78,393 forward passes, assuming a batch size of 1. For inference on n examples, contrastive estimation only requires each of the positive and negative models to perform one forward pass per example, totaling 2n. Consequently, the total cost for contrastive estimation is 78, 393 + 2n forward passes for GSM8K, which is significantly lower than the cost of rollout sampling. For example, for GSM8K that consists of 7,500 examples, contrastive estimation requires only as little as (78393+2*7500)/(581425*7500)=0.002% of the computational cost of rollout sampling.

###### 3.2 cDPO: Explicitly Penalizing Critical Tokens in DPO

Intuition Critical tokens in incorrect trajectories significantly contribute to errors, even when other tokens may be correctly placed. By assigning token-level scores to incorrect trajectories, we can specifically penalize critical tokens without adversely affecting correct ones. Conversely, scoring correct trajectories to encourage certain tokens can inadvertently penalize other valid tokens, resulting in undesired distribution shifts in DPO (Rafailov et al., 2024; Xu et al., 2024). Therefore, we focus exclusively on scoring tokens within incorrect trajectories and extend DPO from the example level to the token level by utilizing token-level rewards for preference optimization.

Formulation Given the pairwise preference dataset D = {(xi, yip, yin)}iM=1, the original DPO loss is formulated as:

M

log σ(ϕ(xi, yip) − ϕ(xi, yin)) Here, ϕ(x, y) is an implicit reward function, given by:

### ∑

ℓDPO = −

i=1

πθ(y | x) πref(y | x)

ϕ(x, y) = γ log

where πθ(·|x) and πref(y|x) represent the policy model and the reference model, respectively, and γ is the coefficient for the KL divergence penalty.

We extend the sample-level DPO to token-level DPO with critical rewards (i.e., cDPO). First, we modify the reward function ϕ(x, y) to include token-level scores sin (Equation 1) as follows:

T

πθ(yt|x, y<t) πref(yt|x, y<t)

### ∑

ϕs(x, y, s) = γ

(1 − st) log

t=1

where T is the total length of the response y, and st represents the token-level reward score in cDPO for the t-th token. Accordingly, the objective of cDPO is formulated as:

M

log σ(ϕ(xi, yip) − ϕs(xi, yin, sin))

### ∑

ℓcDPO = −

i=1

Note that only the reward function for the negative example yn is modified. Intuitively, lower values of st suggest a higher likelihood of being critical tokens, which are more prone to result in incorrect outcomes. By weighting each token’s contribution with 1 − st, the model effectively penalizes generating these critical tokens. This token-level approach helps ensure that the model reduces the likelihood of generating critical tokens, thus improving the overall accuracy of the responses.

###### 3.3 Experimental Results

Experimental Setup We used two widely recognized math reasoning datasets for training and evaluation: GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). For training, we sampled from all questions in the training set to generate the data. For evaluation, we utilized the MATH500 subset, which is uniformly sampled and has a distribution of difficulty levels and subjects that matches the full MATH test set, as demonstrated in Lightman et al. (2023). Additionally, for both training sampling and evaluation, we applied the few-shot prompt approach from Fu et al. (2023).

We conducted experiments on a range of models, including the general-purpose models Llama-3-8Bbase and Llama-3-70B-base (Dubey et al., 2024), as well as the domain-specific model DeepSeekmath-7B-base (Shao et al., 2024). For comparison, we evaluated multiple baseline methods using the data generated from the process described in Section 3.1. For Supervised Fine-Tuning (SFT), we fine-tuned the model using the positive response set Dp. For preference optimization (PO) methods, we utilized the token-level annotated pair-wise preference dataset D. The baselines we compared include:

- • DPO (Rafailov et al., 2024): We tested two different starting points for training: based on the base model and on the SFT model.
- • TokenDPO (Zeng et al., 2024), which is a token-level approach that enhances Kullback-Leibler (KL) divergence regulation by incorporating forward KL divergence constraints at the token level. The SFT model is used as the starting point for training. We implemented TDPO using the publicly available implementation.
- • RPO (Liu et al., 2024; Pang et al., 2024) introduces an additional negative log-likelihood term to improve performance on reasoning tasks. We implemented it using HuggingFace’s implementation and starting with the base model.

For our proposed cDPO, in the contrastive estimation setup, each problem was sampled N = 64 times, selecting the top-p = 50% of incorrect trajectories to train the negative model q(·). During estimation, the hyperparameter β was set to 1.0.

We used LoRA adapters (Hu et al., 2022) to train all the models. We trained both positive and negative models for 1 epoch with a learning rate of 3e-4. For preference optimization training, we set γ = 1.0 and trained for 3 epochs with a learning rate of 2e-5 for all baseline methods. For our cDPO approach, since the token-level scores range between 0 and 1 (whereas in DPO, the scores were all 1), we simply increased the learning rate to 4e-5.

- Figure 4: Log probabilities of chosen and rejected sequences during training on the GSM8K dataset using DPO, RPO, and cDPO. Solid lines represent chosen sequences, while dashed lines denote rejected sequences. The figure illustrates how cDPO achieves a better separation between chosen and rejected sequences compared to DPO and RPO.

GSM8K MATH500 Llama-3 DeepSeek

Method

Llama-3 DeepSeek

Avg. 8B 70B math-7B 8B 70B math-7B

Avg.

Baseline 56.4 80.4 64.1 67.0 16.8 42.2 31.4 30.1

+ SFT 61.2 82.1 67.1 70.1 17.2 43.0 32.6 30.9 + DPO 59.7 87.8 66.5 71.3 17.0 41.2 33.4 30.5 + TokenDPO 62.3 83.3 69.6 71.7 17.8 42.2 32.4 30.8

+ DPO 59.6 88.9 63.1 70.5 15.4 39.8 33.0 29.4 + RPO 67.5 89.7 68.9 75.4 18.4 43.8 34.8 32.3 + cDPO (Ours) 67.9* 90.8* 72.9* 77.2* 19.6* 45.6* 35.0* 33.4*

Table 3: Experimental results on GSM8K and MATH500 datasets. Our proposed method surpasses all the strong baselines at a large margin on individual settings and average performance. * denotes the significance test where p < 0.005.

Learning Curves We begin by examining the impact of cDPO on training dynamics. Figure 4 shows the log probability trends for selected and non-selected sequences over training steps on the GSM8K dataset using the Llama-3-8B model with DPO, RPO, and cDPO.

The proposed cDPO method successfully differentiates between chosen and rejected sequences by significantly increasing the log probability of correct sequences while sharply decreasing that of incorrect ones. In comparison, RPO (DPO with an additional NLL term) increases the probability of correct sequences, but its impact on reducing incorrect response probabilities is less significant. On the other hand, DPO notably decreases the probability of generating incorrect sequences, but also reduces the probability of correct sequences. This suggests that cDPO strikes a balanced approach, effectively enhancing the probability of correct outputs while minimizing critical errors, exceeding the performance of both DPO and RPO.

Main Results Table 3 presents the experimental results for various methods across the GSM8K and MATH500. Our proposed method consistently outperforms all baselines and other methods, achieving the highest scores across both datasets.

For the GSM8K dataset, our method achieves a remarkable average score of 77.2, surpassing the Baseline and notable improvements such as those incorporating SFT and DPO. Specifically, our approach reaches the highest scores with Llama-3 (90.8 for 70B) and DeepSeek (72.9). These results highlight the effectiveness of our method in leveraging the strengths of both large-scale models (Llama-3) and task-specific models (DeepSeek).

Similarly, on the MATH500 dataset, our method attains an average score of 33.4, marking a significant improvement over the baseline (30.1) and other enhanced methods such as SFT and RPO. Notably, our approach yields the highest individual score with Llama-3 (45.6 for 70B) and performs robustly across all model configurations.

The consistent performance improvements observed across various settings underscore the superiority of our method compared to existing techniques. The significance tests, which were conducted to verify the statistical reliability of these results, confirm the competitive advantage of our proposed approach.

#### 4 Related Work

Contrastive Estimation Contrastive estimation is a method commonly employed in statistical modeling and machine learning to estimate model parameters by contrasting observed data with artificially constructed “noise” data. The primary concept is to enhance parameter estimation by comparing the likelihood of the observed data against that of less plausible data. Notable works have refined contrastive estimation techniques (Gutmann & Hyv¨arinen, 2010; Bose et al., 2018; He et al., 2020; Denize et al., 2023). Specifically, our work is closely related to contrastive decoding (CD), an application of contrastive estimation in downstream tasks. CD (Li et al., 2023) involves contrasting token distribution likelihoods between expert and amateur models during decoding. As described by O’Brien & Lewis (2023), this technique avoids high-probability but low-quality tokens, ensuring text fluency and coherence.

Subsequent research has emphasized CD’s potential to improve factuality (Zhang et al., 2023; Yang et al., 2024), knowledge editing (Bi et al., 2024), safety (Zhao et al., 2024), and reasoning (O’Brien & Lewis, 2023; Shi et al., 2024). Notably, O’Brien & Lewis (2023) showed that CD enhances reasoning tasks and mitigates typical errors like missing steps or semantic misunderstandings (Wang et al., 2023). Shi et al. (2024) demonstrated that unchosen experts in Mixture-of-Experts models could be applied for CD, thereby improving model reasoning capacities. Different from those works that focus on the inference of contrastive decoding, our research primarily uses contrastive estimation to identify “critical tokens” that significantly affect the correctness of the reasoning process.

Reinforcement Learning from Human Feedback Aligning LLMs with human preferences is a significant research challenge. This process involves fine-tuning pretrained LLMs on diverse instruction datasets so that the models align with human values, preferences, and instructions. This alignment strategy has shown significant progress and is widely applied across various LLM applications. Among various alignment algorithms (Christiano et al., 2017; Schulman et al., 2017; Ziegler et al., 2019; Ouyang et al., 2022; Bai et al., 2022), DPO (Rafailov et al., 2024) is one of the most representative algorithms. DPO uses the LLM itself as a secret reward model and conducts preference optimization on a preference pair of positive and negative examples. Since then, various contributions have been made to further advance the DPO development of LLM alignment (Pal et al., 2024; Amini et al., 2024; Azar et al., 2024; Lai et al., 2024; Liu et al., 2024; Pang et al., 2024; Tang et al., 2024; Zeng et al., 2024).

Despite the success of DPO in general instructional tasks, challenges remain when applying it to reasoning and mathematical problems. Research has highlighted these challenges from an optimization

perspective, demonstrating that these algorithms often decrease the generation likelihood of positive examples in reasoning. The evolved approach aims to prefer optimization while maintaining high generation likelihoods for positive examples (Liu et al., 2024; Pang et al., 2024; Pal et al., 2024; Feng et al., 2024). Further, Lai et al. (2024) leverage human or GPT-4 validation to pinpoint incorrect reasoning steps; Guo et al. (2023); Yoon et al. (2024) utilize external LLMs to refine responses, deriving token-level preferences from pre- and post-revision comparisons. In contrast, our study seeks to establish an automatic process supervision strategy devoid of human annotation and easy to scale. Specifically, we harness contrastive estimation to identify critical tokens, providing token-level signals for preference optimization that significantly enhance LLM reasoning capabilities.

#### 5 Conclusion

Our work contributes a significant framework for understanding and enhancing mathematical reasoning in LLMs through critical token analysis. By defining and identifying critical tokens, we provide valuable insights into token-level discrepancies that disrupt logical reasoning. Our approach, cDPO, successfully integrates this analysis into DPO, improving model performance on mathematical tasks by improving the model’s differentiation between positive and negative examples. Experimental results from GSM8K and MATH500 benchmarks have shown that our method outperforms existing DPO baselines, underscoring the potential of critical token interventions in enhancing model accuracy. Our research opens new doors for further exploration of token-level influences in complex reasoning tasks, which could lead to more refined and effective LLMs. Future work should explore the integration of cDPO with other reasoning frameworks and extend its application to diverse logical reasoning domains, contributing towards the broader aim of developing more robust and reliable LLMs.

#### References

Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop, pp. 225–237, 2024.

Afra Amini, Tim Vieira, and Ryan Cotterell. Direct preference optimization with an offset. arXiv preprint arXiv:2402.10571, 2024.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Baolong Bi, Shenghua Liu, Lingrui Mei, Yiwei Wang, Pengliang Ji, and Xueqi Cheng. Decoding by contrasting knowledge: Enhancing llms’ confidence on edited facts. arXiv preprint arXiv:2405.11613, 2024.

Avishek Joey Bose, Huan Ling, and Yanshuai Cao. Adversarial contrastive estimation. arXiv preprint arXiv:1805.03642, 2018.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Katherine M Collins, Albert Q Jiang, Simon Frieder, Lionel Wong, Miri Zilka, Umang Bhatt, Thomas Lukasiewicz, Yuhuai Wu, Joshua B Tenenbaum, William Hart, et al. Evaluating language models for mathematics through interactions. Proceedings of the National Academy of Sciences, 121(24): e2318124121, 2024.

Julien Denize, Jaonary Rabarisoa, Astrid Orcesi, Romain H´erault, and St´ephane Canu. Similarity contrastive estimation for self-supervised soft contrastive learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2706–2716, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Duanyu Feng, Bowen Qin, Chen Huang, Zheng Zhang, and Wenqiang Lei. Towards analyzing and understanding the limitations of dpo: A theoretical perspective. arXiv preprint arXiv:2404.04626, 2024.

Yao Fu, Litu Ou, Mingyu Chen, Yuhao Wan, Hao Peng, and Tushar Khot. Chain-of-thought hub: A continuous effort to measure large language models’ reasoning performance. arXiv preprint arXiv:2305.17306, 2023.

Geyang Guo, Ranchi Zhao, Tianyi Tang, Wayne Xin Zhao, and Ji-Rong Wen. Beyond imitation: Leveraging fine-grained quality signals for alignment. arXiv preprint arXiv:2311.04072, 2023.

Michael Gutmann and Aapo Hyv¨arinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 297–304. JMLR Workshop and Conference Proceedings, 2010.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12286–12312, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer. arXiv preprint arXiv:2405.16436, 2024.

Sean O’Brien and Mike Lewis. Contrastive decoding improves reasoning in large language models. arXiv preprint arXiv:2309.09117, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228, 2024.

Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Chufan Shi, Cheng Yang, Xinyu Zhu, Jiahao Wang, Taiqiang Wu, Siheng Li, Deng Cai, Yujiu Yang, and Yu Meng. Unchosen experts can contribute too: Unleashing moe models’ power by selfcontrast. arXiv preprint arXiv:2405.14507, 2024.

Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, R´emi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Avila´ Pires, and Bilal Piot. Generalized preference optimization: A unified approach to offline alignment. arXiv preprint arXiv:2402.05749, 2024.

Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2717–2739, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, and Yi Wu. Is dpo superior to ppo for llm alignment? a comprehensive study. In Forty-first International Conference on Machine Learning, 2024.

Dingkang Yang, Dongling Xiao, Jinjie Wei, Mingcheng Li, Zhaoyu Chen, Ke Li, and Lihua Zhang. Improving factuality in large language models via decoding-time hallucinatory and truthful comparators. arXiv preprint arXiv:2408.12325, 2024.

Eunseop Yoon, Hee Suk Yoon, SooHwan Eom, Gunsoo Han, Daniel Nam, Daejin Jo, Kyoung-Woon On, Mark Hasegawa-Johnson, Sungwoong Kim, and Chang Yoo. Tlcr: Token-level continuous reward for fine-grained reinforcement learning from human feedback. In Findings of the Association for Computational Linguistics ACL 2024, pp. 14969–14981, 2024.

Longhui Yu, Weisen Jiang, Han Shi, YU Jincheng, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2023.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.

Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Token-level direct preference optimization. arXiv preprint arXiv:2404.11999, 2024.

Yue Zhang, Leyang Cui, Wei Bi, and Shuming Shi. Alleviating hallucinations of large language models through induced hallucinations. arXiv preprint arXiv:2312.15710, 2023.

Zhengyue Zhao, Xiaoyun Zhang, Kaidi Xu, Xing Hu, Rui Zhang, Zidong Du, Qi Guo, and Yunji Chen. Adversarial contrastive decoding: Boosting safety alignment of large language models via opposite prompt optimization. arXiv preprint arXiv:2406.16743, 2024.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

