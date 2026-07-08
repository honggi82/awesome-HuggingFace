# arXiv:2507.09104v1[cs.CL]12Jul2025

## CompassJudger-2: Towards Generalist Judge Model via Verifiable Rewards

Taolin Zhang1,2,∗, Maosong Cao1,∗, Alexander Lam1, Songyang Zhang1,†,‡, Kai Chen1,† 1Shanghai AI Laboratory 2Tsinghua University Github:https://github.com/open-compass/CompassJudger

#### Abstract

Recently, the role of LLM-as-judge in evaluating large language models has gained prominence. However, current judge models suffer from narrow specialization and limited robustness, undermining their capacity for comprehensive evaluations. In this work, we present CompassJudger-2, a novel generalist judge model that overcomes these limitations via a task-driven, multi-domain data curation strategy. Central to our approach is supervising judgment tasks with verifiable rewards, guiding intrinsic critical reasoning through rejection sampling to foster robust, generalizable judgment capabilities. We introduce a refined learning objective with margin policy gradient loss to enhance performance. Empirically, CompassJudger2 achieves superior results across multiple judge and reward benchmarks, and our 7B model demonstrates competitive judgment accuracy with significantly larger models like DeepSeek-V3 and Qwen3-235B-A22B. Additionally, we propose JudgerBenchV2, a comprehensive benchmark evaluating cross-domain judgment accuracy and rank consistency to standardize judge model evaluation. These contributions advance robust, scalable LLM judgment and establish new performance and evaluation standards. 1

#### 1 Introduction

In recent years, large language models (LLMs) have advanced rapidly with the development of new foundation models such as DeepSeek-R1 [13], OpenAI-o1 [14], and the Qwen series [29]. Innovations in architecture and data scaling have enabled LLMs to achieve state-of-the-art performance across diverse tasks, including natural language understanding, code generation, creative writing, and complex reasoning [19, 15, 11, 16].

As LLMs are deployed in real-world applications, accurate evaluation of response quality has become increasingly critical. Rule-based benchmarks [24, 34, 15, 6, 8, 10] excel at evaluating standardized tasks but struggle with LLM output variability, often failing to handle edge cases due to reliance on complex regex designs. Model-based approaches like Reward Models and LLM-as-Judge [35, 4, 30, 32] reduce evaluation efforts by leveraging the reasoning ability of LLMs. However, these approaches introduce some new challenges in that restricted generalization ability of existing judge models confines them to specific prompts or datasets. Moreover, some inadequate world knowledge of these LLMs may lead to inaccurate judgments on knowledge-intensive queries, limiting their application for iterative model improvement.

To address these limitations, we propose a unified training paradigm for judge models. First, we define a series of potential application scenarios for judge models and collect a wide range of judge-related public datasets. Subsequently, we curate and synthesize data from different sources to obtain a diverse

1This work is done when Taolin Zhang is on internship at Shanghai AI Laboratory, * means equal contribution, † means corresponding author, ‡ means project lead.

Preprint. Under review.

training dataset. Second, we employ judgment-oriented chain-of-thought (CoT) data generation to improve judgment accuracy, combined with rejection sampling to select high-quality training examples. Finally, we introduce a margin policy gradient loss with verifiable reward signals for better optimization. The resulting CompassJudger-2 series achieves superior performance on judge benchmarks, with our 7B model demonstrating competitive accuracy against significantly larger models like DeepSeek-V3-0324 [21] and Qwen3-235B-A22B [27].

To advance the evaluation of judge models, we also present JudgerBenchV2, a standardized benchmark comprising 10,000 questions across 10 scenarios to evaluate judging capabilities. For the first time, it establishes category-specific judging standards and uses Mix-of-Judgers (MoJ) consensus as ground truth, paired with novel metrics that assess both sample-level accuracy and model-level rank consistency, providing a more robust evaluation.

To summarize, our contributions are as follows:

- • We develop a versatile, multi-styled judge data composition scheme with data curation and synthesis, enhancing CompassJudger-2’s robustness and domain adaptability at the data level.
- • We significantly improve judge performance of CompassJudger-2 by generating high-quality chain-of-thought judge data, selecting optimal training trajectories via rejection sampling, and applying policy gradient loss.
- • We introduce JudgerBenchV2, which treats a Mix-of-Judgers as ground truth and deploys new metrics that jointly assess accuracy and rank fidelity, enabling more reliable evaluation.

#### 2 Related Works

LLM Judgers as Generative Verifiers. LLM-as-judge represents a novel approach where LLMs are fine-tuned to evaluate and provide judgment on model responses, offering not only a reward but also an analysis of the reasoning behind the decision. Unlike traditional reward models that assign a single reward value, LLMs can deliver more valuable feedback by explaining the logic and rationale of their judgments. However, many existing judge models [35, 18] are trained for specific prompts, show poor generalization and cannot adapt to the diverse model evaluation needs. Therefore, all-in-one generative models have emerged, with CompassJudger-1 [3] being the first to incorporate a wide range of judge tasks into model training, greatly enhancing the generalization ability. Con-J [30] and RISE [32] have also conducted all-in-one Judge model training and achieved better Judge performance through the DPO strategy. Although these models have greatly ensured the generalization of prompts, they have not yet verified on other judge tasks such as critique generation and stylized judge.

LLM Judging Evaluation. Despite the rapid evolution of judge models, there is a notable lack of benchmarks for their evaluation. Rewardbench [16] focuses on assessing a model’s reward capability across four categories: Chat, Chat Hard, Reasoning, and Safety. However, it faces issues with outdated data and a limited number of evaluation scenarios, leading to overfitting in many models on Rewardbench. JudgeBench [26], by contrast, evaluates judge models based on their ability to determine the correctness of answers in datasets like MMLU-Pro [28] and LiveCodeBench [15], thus testing their knowledge base to answer factual questions. RMB [33] introduces a method using the Best of N (BoN) and involves a comparative model making multiple judgments to assess the consistency of the model’s judging. Nonetheless, these benchmarks only offer a limited view of the judging ability and do not encompass a wide enough range of evaluation scenarios.

#### 3 Methodology

In this section, we first outline the training data pipeline for CompassJudger-2, covering data curation and data synthesis. We then explain how to apply rejection sampling and policy gradient optimization to incorporate verified rewards into the judge task. We apply the training data and the training strategy to the Qwen2.5-Instruct series of models, yielding our CompassJudger-2.

###### Data Synthesis

[Figure 1]

###### Data Curation

Outdated Data

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Knowledgebased Datasets

Filter by time Judgement Rectification

Public Judge Data

Ground Truth

Judgment Generation

[Figure 6]

Response Generation

Response

Data Pool

Up-todate Data

Data Pool

Diversity Enhance

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Chat-based Datasets

[Figure 12]

[Figure 13]

Public Reward Data

Judgment Generation

Detailed Response

Concise Response

Style-controled Response Generation

Judgment Generation

Multiple Judgment Rejection Sampling

- Figure 1: The data construction pipeline of CompassJudger-2, including data curation and synthesis. The Data Curation stage include reconstruction of public judge and reward data, while the data synthesis stage contains response generation over knowledge-based and chat based datasets.

##### 3.1 Overall Data Pipeline

Data Curation. We begin by collecting open-source judge-related datasets, including Public Judge Data and Public Reward Data. Public Judge Data contain critiques and explanations while Public Reward Data only contain ground truth labels.

For Public Judge Data, we observe that many judgments were generated by outdated models such as ChatGPT, which may introduce misjudgments and implicit errors. To address this issue, we split the data into outdated and up-to-date subsets based on the cutoff date of October 2024. For outdated data, we use Qwen2.5-72B-Instruct to reconstruct outdated judgment and further verify correctness by comparing the predictions with human-labeled ground truth, ensuring that only accurate judgments are preserved. For up-to-date data, we leverage a large number of subjective evaluation datasets available in the community, such as ArenaHard [19], WildBench [20], MTBench [1], etc., to collect their judgment prompt templates, which are then used to replace the original prompt templates in the existing judgment data, thereby enhancing their diversity.

For Public Reward Data, such data lacks critique annotations, making it suboptimal for training generative judge models. To leverage these data effectively, we prompt Qwen2.5-72B-Instruct to generate multiple judgments for each data instance and further refine the quality through rejection sampling. A detailed description of our construction pipeline is provided in Section 3.2.

Data Synthesis. To enhance the robustness and versatility, we systematically design and synthesize data from Knowledge-based Datasets and Chat-based Datasets, aiming to enrich world knowledge and improve stylistic adaptability, respectively.

For Knowledge-based Datasets, we aggregate model outputs from standardized benchmarks (e.g., MMLU[28], CMMLU [17], GSM8K [7]) and employ Qwen2.5-72B-Instruct to evaluate their correctness while providing detailed rationales. These judgments are subsequently validated against ground truth answers, with only verified correct evaluations retained in the training corpus.

For Chat-based Datasets, we generate response pairs exhibiting contrasting characteristics and instruct Qwen2.5-72B to select the superior response according to specified style requirements, thereby creating style-sensitive judgment data.

Overall Training Data Construction. Prior studies [3, 23] have also demonstrated that incorporating general instruction data helps maintain a model’s generalization capability while preserving its judge performance. Therefore, we also include general instruction data collected from CompassJudger-1 in our training dataset. The final training data for CompassJudger-2 consists of four components: (1) publicly available judge data that undergo diversity enhancement and quality rectification, (2) publicly available reward data process through rejection sampling (RFT data), (3) synthetic data generated from knowledge-based and chat-based datasets, and (4) general instruction data (G-SFT data).

##### 3.2 Incorporating Verified Reward

To enhance the judge model’s accuracy and generalization, we propose a training paradigm that integrates verifiable rewards through policy gradient optimization and rejection sampling. Specifically, we first guide the model to generate judgments through critical thinking, then reinforce the reward for final judgment outcomes using policy gradient loss, and further enhance judgment performance by incorporating a rejection sampling strategy. We elaborate on these steps in detail below.

Verifiable Reward

Instruction Response A

Response B

Demand Strengthen Weakness Reason A/B

COT

Answer

Prompt

- Figure 2: Illustration of the reasoning path in the judge task. The reasoning path involves critical analysis of the instruction and responses from various models. The final answer prediction can be treated as a classification task, which is further guided by a verified reward for supervision.

Prompt

[Figure 14]

[Figure 15]

Ground Truth

Reasoning Path

Answer Logit

CoT Answer B

- CoT Answer A

- CoT Answer B

- Figure 3: Training framework of CompassJudger-2. CompassJudger-2 utilize rejection sampling to choose correct reasoning paths for SFT training and apply policy gradient loss over the answer logit to incorporate verifiable reward.

Critical Thinking. The SFT training of judge models requires high-quality instruction-response data, which can be costly to obtain. To tackle this challenge, we introduce an innovative chain-of-thought methodology aimed at producing high-quality instruction-response data specifically for the judge task. Following the reasoning pipeline in DeepSeek-R1 [13], we craft a critical thinking prompt specifically for judge models, as shown in Figure 2. We divide the judge task into several important steps and require the model in making predictions through comprehensive thinking. Formally, the model is required to dissect the problem by evaluating: (1) User’s Demand: The model need to analysis the specific requirements of the user’s instruction. (2) Strengths of Model A/B. (3) Weaknesses of Model A/B. (4) Reasoning: Perform reasoning based on the aforementioned analysis. (5) Prediction: Output the final prediction. In practice, we employ Qwen2.5-72B-Instruct [29] as backbone for data synthesis.

Judge Reward. In the judge task, the model performs binary classification by outputting its prediction at designated positions. This structured output enables us to utilize the ground truth labels as explicit guidance signals for optimization. Inspired by DeepSeek-R1 [13], given an instructionresponse pair (x,y), a prediction position kx, and the corresponding ground truth label yk∗

, we apply a rule-based reward r(x,y) defined as 1 if the model’s prediction at position kx matches the ground truth label yk∗

x

, and 0 otherwise.

x

Policy Gradient Optimization. We formulate the learning objective as maximizing the expected reward over the response distribution and the gradient of this objective can be derived as follows:

∇θJ(θ) = ∇θ [Ex∼DEy∼π

[r(x,y)]]

θ

= Ex∼DEy∼π

[r(x,y)∇θ log πθ(y|x)]

θ

(1)

n

= Ex∼DEy∼π

∇θ log πθ(yt|x,y<t)

r(x,y)

θ

t=1

This decomposition shows how the gradient propagates through all sequence positions in autoregressive models. Given that reward function only depends on the prediction at position kx, the policy gradient loss can be further simplified as follows:

###### LPG = − E

###### E

y∼πθ

x∼D

x|x,y<k

log πθ(yk

x

)

ykx=yk∗x

. (2)

We observe that SFT loss computes the conditional probability under fixed prefixes, while the policy gradient loss approximates the marginal probability by aggregating over diverse prefix. This

- Table 1: Total loss and mapping functions. We discuss three mapping functions to approximate g in the total loss.

Loss Loss Function Total Loss Ltotal = −NM1 Ni=1

log πθ(yt(i,j)|x(j),y<t(i,j)) + g log πθ(yk(i,∗)

|x(j),y<k(i,j)

M j=1[ t̸=k

) ]

ij

ij

ij

πθ(yk(i,∗)

|x(j),y<k(i,j)

) πθ(yk(i,−)

DPO Loss LDPO = −NM1 Ni=1

M j=1 log σ β log

ij

ij

|x(j),y<k(i,j)

) Temperature Loss LTemp = −NM1 Ni=1

ij

ij

exp(log πθ(yk(i,∗)

|x(j),y<k(i,j)

)/τ) y′ exp(log πθ(y′|x(j),y<k(i,j)

M j=1 log

ij

ij

)/τ) Margin Loss LMargin = NM1 Ni=1 Mj=1 max 0,γ − log πθ(yk(i,∗)

ij

) + log πθ(yk(i,−)

|x(j),y<k(i,j)

|x(j),y<k(i,j)

)

ij

ij

ij

ij

distinction arises because SFT employs teacher forcing with deterministic prefixes, whereas policy gradient optimization explores various response trajectories to maximize expected rewards.

Rejection Sampling for RL Generalization. While policy gradient optimization directly maximizes expected rewards, it suffers from limited exploration during the standard SFT stage, where fixed prefixes constrain the diversity of generated responses. To address this exploration bottleneck, we leverage rejection sampling to enhance model generalization through diversified prefix generation. Our approach systematically generates and filters diverse response candidates based on quality metrics and reject the samples that do not match the ground truth label. Formally, for the ith instruction (i ∈ {1..,N}) in the dataset, we generate M response samples that satisfy the ground truth label y(i,∗) to approximate the policy gradient loss:

N

M

1 NM

log πθ(yk(i,∗)

|x(i),y<k(i,∗)

) (3)

LPG = −

x

x

i=1

j=1

Similarly, we apply the SFT loss to the sampled response candidates. We further combine the SFT loss and policy gradient loss:

Ltotal =LSFT + LPG . (4)

Mapping Function. The total loss can be decomposed with SFT loss over the prefix and a mapping function g over the prediction position, as shown in Table 1. We also design three different mapping

loss function on the prediction position as g for optimization over the ground truth answer yk(i,∗)

and the wrong answer yk(i,−)

ij

.

ij

- • DPO Loss w/o Reference Model encourages the model to increase the probability of true answer while decreasing the probability of wrong answer.
- • Temperature Loss performs temperature scaling to the logits before softmax, effectively sharpening the probability distribution around the ground truth token with τ as the temperature.
- • Margin Loss introduces a margin γ between the ground truth token and other answer, ensuring that the ground truth probability is sufficiently higher.

#### 4 JudgerBenchV2: A More Robust Benchmark for Judge Models

Existing benchmarks for judge models have numerous limitations, such as insufficient coverage of judge scenarios and a lack of sufficiently accurate ground truth (GT). To address these issues, we propose JudgerBenchV2, aiming to improve the evaluation landscape for judge models and provide a more comprehensive and accurate benchmark.

Data Construction. We first collect real-world user queries in Chinese and English through CompassArena [8], and cluster them via K-means. We then utilize an LLM to classifies each query by difficulty level and manually select 100 queries per scenario, ensuring a balanced distribution of languages and difficulty level. Next, we select 10 high-performing models of comparable capability and generate their responses to these queries. We then use GPT-4o-mini as the policy model and pair it with each of the 10 models to form response pairs. A judge model evaluates these pairs in a pairwise manner to obtain judge results. By comparing with the GT, we derive the performance scores of the judge model.

- Table 2: Main results on judge benchmarks. CompassJudger-2 achieves state-of-the-art performance on both 7B and 32B+ variants.

Model JudgerBenchV2 JudgeBench RMB RewardBench Average General Models

Qwen2.5-7B-Instruct [29] 57.14 23.23 69.03 79.69 57.27 Llama3.1-8B-Instruct [12] 57.64 33.23 66.01 73.64 57.63 InternLM3-8B-Instruct [2] 57.71 24.19 72.02 80.62 58.64 Qwen2.5-32B-Instruct [29] 62.97 59.84 74.99 85.61 70.85 DeepSeek-V3-0324 [21] 64.43 59.68 78.16 85.17 71.86 Qwen3-235B-A22B [27] 61.40 65.97 75.59 84.68 71.91

Reward Models

InternLM2-20B-reward [2] - - 62.90 90.20 Deepseek-GRM-27B [23] - - 69.00 86.00 RM-R1-Qwen-Instruct-32B [5] - - 73.00 92.90 -

7B Judge Models

- CompassJudger-1-7B-Instruct [3] 57.96 46.00 38.18 80.74 55.72 Con-J-7B-Instruct [31] 52.35 38.06 71.50 87.10 62.25 RISE-Judge-Qwen2.5-7B [32] 46.12 40.48 72.64 88.20 61.61

- CompassJudger-2-7B-Instruct 60.52 63.06 73.90 90.96 72.11 32B+ Jugde Models

- CompassJudger-1-32B-Instruct [3] 60.33 62.29 77.63 86.17 71.61 Skywork-Critic-Llama-3.1-70B [25] 52.41 50.65 65.50 93.30 65.47 RISE-Judge-Qwen2.5-32B [32] 56.42 63.87 73.70 92.70 71.67

- CompassJudger-2-32B-Instruct 62.21 65.48 72.98 92.62 73.32

- Table 3: Results on general benchmarks. CompassJudger-2 maintains strong performance on both objective and subjective datasets.

Model MMLU Pro GPQA Diamond AIME2025 LiveCodeBench v5 IFEval ArenaHard 7B Judge Models

Qwen2.5-7B-Instruct [29] 55.43 34.85 6.67 12.57 73.20 47.86 Con-J-7B-Instruct [31] 44.74 27.27 3.33 6.59 54.90 23.49 RISE-Judge-Qwen2.5-7B [32] 51.56 32.32 6.67 12.57 44.18 35.99 CompassJudger-2-7B-Instruct 52.55 39.39 6.67 14.37 74.49 53.49

32B Judge Models

Qwen2.5-32B-Instruct [29] 68.92 42.93 16.67 30.54 79.85 70.16 RISE-Judge-Qwen2.5-32B [32] 67.88 42.93 6.67 27.54 62.85 61.52 CompassJudger-2-32B-Instruct 69.22 50.51 16.67 25.15 79.48 83.31

Mixture of Judges. Evaluating open-ended questions is highly subjective since different individuals may produce varying judgments, and different models also exhibit judge biases. Relying solely on the judgments from a single human or a single model as GT thus risks introducing bias. To address this, we introduce the mixture of judgers (MoJ) strategy, leveraging the judgments of DeepSeek-R1, DeepSeek-v3-0324, and Qwen3-235B-A22B and their majority consensus is considered as GT.

Robust Judge Performance Metrics. Traditional judge evaluation metrics primarily focus on sample-level accuracy and fail to capture essential dimensions like ranking consistency. For example, human raters often converge on overall model rankings although they may disagree on individual samples. A comprehensive evaluation framework should therefore incorporate both fine-grained judgment accuracy and high-level ranking fidelity.

In JudgerBenchV2, we conduct pairwise comparisons between a candidate model and GPT-4o-mini to determine which delivers superior responses. Each comparison is evaluated by both a ground truth judge model and a test judge model. A sample is considered correct if both judges agree on the better-performing model. For each sample, the model deemed superior earns a score increment of 1. The total number of pairwise samples is denoted by N and C represents the number of samples where the GT and test judge models agree on the superior model. For a set of M candidate models, let the GT judge model and the test judge model generate score lists S1 = {s1,m}m∈M and S2 = {s2,m}m∈M, respectively, where si,m represents the cumulative score for model m based on pairwise wins. Additionally, let R1 = {r1,m}m∈M and R2 = {r2,m}m∈M denote the rank lists, where ri,m is the rank of model m according to judge i. The performance of the test judge model is evaluated using the following metric:

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

90

75

60

Score

45

30

15

0

JudgerBenchV2 JudgeBench RMB RewardBenchAverage

w/o RFT Data w/o G-SFT Data Full Data

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

90

75

60

Score

45

30

15

0

MMLU Pro GPQA AIME2025 LCBv5 IFEval ArenaHard

w/o RFT Data w/o G-SFT Data Full Data

(a) Judge benchmarks.

(b) Subjective and objective benchmarks.

Figure 4: Data ablation results on different benchmarks.

- Table 4: Ablation results with policy gradient loss on CompassJudger-2-7B-Instruct. Margin loss provides a significant boost compared to other forms of loss.

Loss JudgerBenchV2 JudgeBench RMB RewardBench Average

Baseline 60.20 61.77 68.13 89.50 69.90 DPO 60.56 61.13 66.35 90.07 69.53 Temperature 59.43 62.42 67.77 90.25 69.97 Margin 60.52 63.06 73.90 90.96 72.11

P = 100 ·

C N

Sample-level accuracy

−

100 |M| m∈M

(∗

|r1,m − r2,m| |M| − 1

Normalized rank difference

+ |s1,m − s2,m|

maxm′∈M |s1,m′ − s2,m′| Normalized score difference

).

(5)

The first term captures the sample-level accuracy by measuring the agreements between the judges. The second term penalizes discrepancies in rankings and scores, with normalization to ensure equitable comparisons across different models.

- 5 Experiments

- 5.1 Experimental Setup

Evaluation Datasets. We evaluate the performance of CompassJudger-2 on leading judge benchmarks, including RewardBench [16], JudgeBench [26], RMB [33], and our JudgerBenchV2. Besides, we compare our method with other judge models over popular objective and subjective benchmarks, including MMLU Pro [28], GPQA Diamond [24], AIME2025, LiveCodeBench v5 [15], IFEval [9] and ArenaHard [19]. We further conduct extensive experiments on AlignBench [22] and AlpacaEval [11], showcasing the critique ability of CompassJudger-2 for model improvement.

Training Settings. In practice we generate 8 candidate responses for filtering during rejection sampling. For model training, we utilize Qwen-2.5 series as the checkpoint and adopt 6e-5 as the learning rate. For policy gradient loss parameter, we set β = 0.1 in DPO loss, τ = 5 in temperature loss and γ = 10 in margin loss. We apply DPO loss on only the candidate answer and margin loss on the top 10 logits. We train the model for 1 epoch with batch size equal to 512.

- 5.2 Main Results

Judge Ability Analysis. To verify the judge ability of our method, we conduct evaluation across multiple benchmarks and compare our method with general models, reward models and specialized judge models including the Skywork [25] and RISE [32] series. As presented in Table 2, CompassJudger-2 consistently surpasses all baselines in average performance, demonstrating significant advancements in the generalization ability. Notably, CompassJudger-2-7B-Instruct outperforms RISE-Judge-Qwen2.5-7B by 22.58% on JudgeBench and by 10.5% on average. Compared to the

Table 5: Model improvement with generated critique on chat-based datasets. AlignBench scores range from 0 to 10 and other datasets score range from 0-100. To standardize the scale, we normalize all the scores to a 0–100 range and then compute the average.

Model AlignBench AlpacaEval ArenaHard Average Policy Model: LLama3.1-8B-Instruct

Base 4.90 27.95 29.11 35.35 RISE-Judge-Qwen2.5-7B 4.99 28.03 28.64 35.52

- CompassJudger-2-7B-Instruct 5.20 30.68 32.76 38.48 Policy Model: Qwen2.5-7B-Instruct

Base 6.65 36.65 47.86 50.34 RISE-Judge-Qwen2.5-7B 6.43 35.12 45.07 48.16

- CompassJudger-2-7B-Instruct 6.76 38.14 51.15 52.30 Policy Model: InternLM3-8B-Instruct

Base 6.46 64.84 46.27 58.57 RISE-Judge-Qwen2.5-7B 6.47 62.17 43.89 56.92 CompassJudger-2-7B-Instruct 6.50 65.85 47.76 59.54

100.0%

1.60

0.07

1.14

2.60

2.62

1.16

0.98

95.0%

0.77 0.05

3.98

2.31

1.89

0.78

0.27

90.0%

0.65

1.81

2.59

Score

3.02

85.0%

10.67

80.0%

2.79

75.0%

70.0% Chat Chat Hard Safety Reasoning Average

CJ2-7B Standard Prompt CJ2-7B Style Prompt

CJ2-32B Standard Prompt

RISE-7B Standard Prompt

RISE-32B Standard Prompt

CJ2-32B Style Prompt

RISE-7B Style Prompt

RISE-32B Style Prompt

| |
|---|

##### Figure 5: Comparison results over style judge of CompassJudger-2 and RISE.

- CompassJudger-1 series, CompassJudger-2 enhances judge performance by 16.39% for the 7B model and 1.71% for the 32B model, on average. General Ability Analysis. We further highlight the improvements in general capabilities of
- CompassJudger-2 compared to other judge models across objective and subjective benchmarks, as shown in Table 3. CompassJudger-2 achieves markedly superior performance over other judge models on both objective and subjective datasets, demonstrating its generalization ability. Remarkably, CompassJudger-2 surpasses general models like Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct on specific datasets, revealing a strong correlation between judge ability and general ability in LLMs and their potential to enhance each other.

##### 5.3 Ablation Study

Policy Gradient Loss. To evaluate the impact of incorporating policy gradient loss, we conduct a thorough ablation study to determine the most effective type of policy gradient loss for improving model performance. We compare a baseline model without policy gradient loss to models using various policy gradient losses, including DPO, Temperature, and Margin loss on 7B level, as presented in Table 4. Our findings reveal that the verified reward serves as a critical supervised signal, significantly enhancing the model’s performance in judge tasks. All models with policy gradient loss surpass the baseline model on RewardBench, achieving performance improvements ranging from 0.5% to 1.4%. Notably, the model with margin loss demonstrate the best generalization across 3 out of 4 datasets, delivering an 2.21% performance on average boost compared to the baseline model. As a result, we select margin loss as the default choice for our study.

Data Ablation. To investigate how general instruction data (G-SFT Data) and rejection sampling (RFT Data) impact judge ability and general ability, we perform ablation studies by separately removing each data type from the training set. As illustrated in Figure 4, the results highlight several key findings. Removing RFT data causes a significant decline in judge performance, mainly due to lower judge consistency and result in poor results on the RMB dataset. In addition, including RFT data enhances performance across specific datasets, such as GPQA-Diamond and ArenaHard, underscoring its role in boosting general ability. In contrast, General SFT data primarily maintain the general ability of the model, with minimal impact on judge ability.

##### 5.4 Discussions

Critique Ability for Model Improvement. An effective all-in-one judge model should be capable to produce high-quality critiques that offer insightful analysis and explanations. To evaluate the critique ability of CompassJudger-2, we task it with generating analyses of responses from various policy models on subjective datasets. We then permit the policy models to revise their initial responses based on these critiques. For comparison, we present the initial scores of the policy models (Base) alongside the results of using RISE-Judge-Qwen2.5-7B as the critique model, as shown in Table 5. The results reveal a striking insight that all policy models improve when guided by critiques from CompassJudger2, whereas low-quality critiques from RISE-Judge-Qwen2.5-7B often result in performance drop. This suggest the superior critique quality of CompassJudger-2 and highlights its potential to enhance training performance during model iterations. We also provide some case study for comparison in the Appendix.

Style Judge. An effective all-in-one judge model should also maintain consistent performance with various prompts. Therefore, we conduct style judge experiment with modifying judging prompts by adding following sentences: "Beyond this, users prefer a more detailed response; therefore, you need to determine which model’s answer provides more comprehensive and useful information when both responses are correct and have completed the user’s request". We present the results on different subset of RewardBench. As can be seen from the results in Figure 5, RISE-32B suffers from significant performance drop by 10.67% in the Chat Hard subset. Compared with RISE, CompassJudger-2 are less sensitive of judging prompts and show better consistency and generalization ability, indicating the superiority of our method.

#### 6 Conclusions

In this work, we present CompassJudger-2, an series of all-in-one judge models that advance LLM-as-judge performance through a unified training paradigm combining diverse task-driven data composition, high-quality chain-of-thought supervision, and verifiable reward-guided optimization. Furthermore, we introduce JudgerBenchV2, a comprehensive benchmark with mixed-of-judgers and novel ranking-aware metrics, to enable more nuanced and reliable evaluation of judge models. Looking forward, CompassJudger-2 paves the way for more adaptable, interpretable, and efficient judge services in real-world LLM deployments, and we anticipate that extending this work to multi-modal and interactive evaluation scenarios will further enhance its applicability and impact.

#### 7 Limitations

Despite the superior performance, there are still some limitations of CompassJudger-2. Rejection sampling incurs relatively higher inference costs, and the hallucinations produced by the LLM when synthesizing data may pose potential risks. These issues need to be further addressed, which will in turn enhance the performance of the judge models.

#### References

- [1] Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su, Tiezheng Ge, Bo Zheng, et al. Mt-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. arXiv preprint arXiv:2402.14762, 2024.

- [2] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye

- Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report, 2024.
- [3] Maosong Cao, Alexander Lam, Haodong Duan, Hongwei Liu, Songyang Zhang, and Kai Chen. Compassjudger-1: All-in-one judge model helps model evaluation and evolution. arXiv preprint arXiv:2410.16256, 2024.

- [4] Ding Chen, Qingchen Yu, Pengyuan Wang, Wentao Zhang, Bo Tang, Feiyu Xiong, Xinchi Li, Minchuan Yang, and Zhiyu Li. xverify: Efficient answer verifier for reasoning model evaluations. arXiv preprint arXiv:2504.10481, 2025.

- [5] Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, et al. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025.

- [6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions, 2019. URL https://arxiv.org/abs/1905.10044.
- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [8] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.
- [9] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.

- [10] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs,

2019. URL https://arxiv.org/abs/1903.00161.

- [11] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

- [12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [14] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

- [16] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024.

- [17] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.

- [18] Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. Generative judge for evaluating alignment. arXiv preprint arXiv:2310.05470, 2023.

- [19] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024.

- [20] Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. Wildbench: Benchmarking llms with challenging tasks from real users in the wild. arXiv preprint arXiv:2406.04770, 2024.

- [21] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

- [22] Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, et al. Alignbench: Benchmarking chinese alignment of large language models. arXiv preprint arXiv:2311.18743, 2023.

- [23] Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.

- [24] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

- [25] Tu Shiwen, Zhao Liang, Chris Yuhao Liu, Liang Zeng, and Yang Liu. Skywork critic model series. https://huggingface.co/Skywork, September 2024. URL https://huggingface. co/Skywork.
- [26] Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges. arXiv preprint arXiv:2410.12784, 2024.

- [27] Qwen team. Qwen3: Think deeper, act faster. https://qwenlm.github.io/blog/qwen3/, 2025.
- [28] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

- [29] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [30] Ziyi Ye, Xiangsheng Li, Qiuchi Li, Qingyao Ai, Yujia Zhou, Wei Shen, Dong Yan, and Yiqun Liu. Beyond scalar reward model: Learning generative judge from preference data, 2024. URL https://arxiv.org/abs/2410.03742.
- [31] Ziyi Ye, Xiangsheng Li, Qiuchi Li, Qingyao Ai, Yujia Zhou, Wei Shen, Dong Yan, and Yiqun Liu. Learning llm-as-a-judge for preference alignment. In The Thirteenth International Conference on Learning Representations, 2025.

- [32] Jiachen Yu, Shaoning Sun, Xiaohui Hu, Jiaxu Yan, Kaidong Yu, and Xuelong Li. Improve llm-as-a-judge ability as a general ability. arXiv preprint arXiv:2502.11689, 2025.

- [33] Enyu Zhou, Guodong Zheng, Binghai Wang, Zhiheng Xi, Shihan Dou, Rong Bao, Wei Shen, Limao Xiong, Jessica Fan, Yurong Mou, et al. Rmb: Comprehensively benchmarking reward models in llm alignment. arXiv preprint arXiv:2410.09893, 2024.

- [34] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

- [35] Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631, 2023.

### Appendix

#### A Deriving the Loss Function

Judge Reward. In the judge task, given a instruction-response pair (x,y), prediction position kx and ground truth label yk∗, we apply a rule-based reward defined as:

r(x,y) =

1 if yk

= yk∗

x

0 otherwise

. (6)

x

Policy Gradient Optimization. To optimize the judge model’s performance, we formulate the learning objective as maximizing the expected reward over the response distribution:

[r(x,y)] (7) The gradient of this objective can be derived using the policy gradient theorem:

J(θ) = Ex∼DEy∼π

θ

∇θJ(θ) = Ex∼DEy∼π

θ

= Ex∼DEy∼π

θ

[r(x,y)∇θ log πθ(y|x)]

n

∇θ log πθ(yt|x,y<t)

r(x,y)

t=1

(8)

This decomposition shows how the gradient propagates through all sequence positions in autoregressive models. The corresponding policy gradient loss is:

LPG = −Ex∼DEy∼π

θ

r(x,y)

n

log πθ(yt|x,y<t) (9)

t=1

Given our binary reward function that only depends on the prediction at position kx, we can simplify:

LPG = −Ex∼DEy∼π

θ

= −Ex∼DEy∼π

θ

= −Ex∼DEy∼π

θ

x|x,y<k

[r(x,y)log πθ(yk

)]

x

= yk∗

I(yk

x|x,y<k

)log πθ(yk

)

x

x

x

x|x,y<k

log πθ(yk

)

x

ykx=yk∗x

(10)

Rejection Sampling for RL Generalization. We further apply rejection sampling to approximate the policy gradient loss. Formally, for the ith instruction (i ∈ {1..,N}) in the dataset, we generate M response samples that satisfy the ground truth label y(i,∗) and obtain the following loss:

LPG = −Ex∼DEy∼π

θ

N

M

1 N

1 M

= −

i=1

j=1

N

M

1 NM

= −

i=1

j=1

x|x,y<k

log πθ(yk

x

)

ykx=yk∗x

log πθ(yk(i,j)

|x(i),y<k(i,j)

)

x

x

yk(i,jx )=yk(i,x∗)

log πθ(yk(i,∗)

|x(i),y<k(i,∗)

)

x

x

(11)

Similarly, we apply the SFT loss to the sampled response candidates. To balance the standard sequence modeling objective with reward optimization, we combine the SFT loss and policy gradient loss through a mapping function f and derive another mapping function g:

L = LSFT + LPG(f)

N

M

1 NM

log πθ(yt(i,j)|x(j),y<t(i,j))

= −

i=1

j=1 t̸=kij

N

M

, (12)

1 NM

log πθ(yk(i,∗)

) + f log πθ(yk(i,∗)

|x(j),y<k(i,j)

|x(j),y<k(i,j)

−

)

ij

ij

ij

ij

i=1

j=1

 

 

N

M

1 NM

log πθ(yt(i,j)|x(j),y<t(i,j)) + g log πθ(yk(i,∗)

|x(j),y<k(i,j)

= −

)

ij

ij

i=1

j=1

t̸=kij

where g is a composite function that combines the original mapping function f with the log probability term to provide a more flexible optimization objective. In our method, the mapping function g is approximate by DPO loss, Temperture Loss and Margin Loss.

- B Detailed Results on the Judge Benchmarks We list the detailed results of judge models on the Judge Benchmarks in Table 6, 7 and 8.

##### Table 6: Detailed results on JudgerBenchV2 benchmarks.

Model Accuracy Normalized Diff Rank Diff Score Diff Final Score 7B Judge Models

- CompassJudger-1-7B-Instruct [3] 77.41 61.48 11.40 83.40 57.96 Con-J-Qwen2-7B [31] 71.30 66.61 17.60 85.20 52.35 RISE-Judge-Qwen2.5-7B [32] 70.08 77.85 14.00 202.50 46.12

- CompassJudger-2-7B-Instruct 78.04 57.00 10.80 76.90 60.52 32B+ Judge Models

- CompassJudger-1-32B-Instruct [3] 80.99 60.32 11.40 62.90 60.33 Skywork-Critic-Llama-3.1-70B [25] 70.27 65.44 15.20 97.30 52.41 RISE-Judge-Qwen2.5-32B [32] 74.00 61.15 10.60 88.80 54.42

- CompassJudger-2-32B-Instruct 80.90 56.47 8.60 64.10 62.21

##### Table 7: Detailed results on RMB benchmarks.

Model Pair Accuracy BoN Accuracy Final Score 7B Judge Models

- CompassJudger-1-7B-Instruct [3] 47.40 28.96 38.18 Con-J-Qwen2-7B [31] 84.80 74.20 79.50 RISE-Judge-Qwen2.5-7B [32] 78.79 66.50 72.64

- CompassJudger-2-7B-Instruct 80.58 67.23 73.90 32B+ Judge Models

- CompassJudger-1-32B-Instruct [3] 82.73 72.53 77.63 Skywork-Critic-Llama-3.1-70B [25] 68.35 62.50 65.50 RISE-Judge-Qwen2.5-32B [32] 79.99 67.42 73.70

- CompassJudger-2-32B-Instruct 79.61 66.35 72.98

##### Table 8: Detailed results on RewardBench benchmarks.

Model Chat Chat Hard Safety Reasoning Final Score 7B Judge Models

- CompassJudger-1-7B-Instruct [3] 97.80 61.00 84.50 89.50 83.20 Con-J-Qwen2-7B [31] 91.90 80.30 88.20 88.10 87.10 RISE-Judge-Qwen2.5-7B [32] 92.20 76.50 88.00 96.10 88.20

- CompassJudger-2-7B-Instruct 92.36 85.99 91.08 94.41 90.96 32B+ Judge Models

- CompassJudger-1-32B-Instruct [3] 98.00 65.10 85.30 92.40 85.20 Skywork-Critic-Llama-3.1-70B [25] 96.60 87.90 93.10 95.50 93.30 RISE-Judge-Qwen2.5-32B [32] 96.60 83.30 91.90 98.80 92.70

- CompassJudger-2-32B-Instruct 93.37 88.58 90.68 97.00 92.40

CoT Synthesizing Prompt

Now we are reviewing a user’s interaction with two models. Your task is to evaluate the responses from Model A and Model B by carefully analyzing the dialogue step by step, following a clear and structured thought process:

- 1. User’s Demand:

- Carefully analyze the user’s request. What is the user specifically asking for? What are the key aspects of the request that need to be fulfilled? Identify any constraints (e.g., time, format, quantity) the user has provided.

- 2. Strengths of Model A:

- - Identify the strengths of Model A’s response. Consider how well it addresses the user’s demand, meets the user’s constraints, and how well it serves the intended purpose.

3. Weaknesses of Model A:

- - Identify the weaknesses of Model A’s response. What aspects of the response fail to meet the user’s request or constraints? What could have been improved?

4. Strengths of Model B:

- Identify the strengths of Model B’s response. Consider how well it addresses the user’s demand, meets the user’s constraints, and how well it serves the intended purpose.

5. Weaknesses of Model B:

- - Identify the weaknesses of Model B’s response. What aspects of the response fail to meet the user’s request or constraints? What could have been improved?

- 6. Reasoning:

- Based on your analysis of both responses, explain which model better addresses the user’s needs. Discuss which model’s response is more suitable given the user’s request and constraints.

- 7. Choice:

- Conclude with a choice between Model A and Model B based on your reasoning. Indicate which model provides the more appropriate and useful response for the user’s request. Your final reply must be structured in the following format: { "User’s Demand": "[The user’s request or need]",

- "Strengths of Model A": "[Summary of the strengths of Model A]",

- "Weaknesses of Model A": "[Summary of the weaknesses of Model A]",

"Strengths of Model B": "[Summary of the strengths of Model B]",

- "Weaknesses of Model B": "[Summary of the weaknesses of Model B]", "Reasoning": "[Explanation of which model is more suitable for the user’s demand]", "Choice": "[Model A or Model B]" }

Critique Example-Prompt

Below is a user’s question and a model’s response. Please help identify the strengths and weaknesses of the model’s reply so that it can better address the user’s question. The user’s question is as follows: Write a poem about winter, requiring each sentence to be exactly five words long, and without using the words "winter" or "snow," yet still evoking a sense of the season. The model’s response is as follows: Frost paints silent trees. Chill lingers in the air. Footsteps fade on paths. Windows gleam with light. Hushed, the world awaits. Now, please provide your feedback, highlighting the strengths and areas for improvement in the model’s response.

Critique Example-RISE-Judge-Qwen2.5-7B Answer Strengths of the Model’s Response:

- 1. Conciseness and Adherence to Word Count: Each sentence is exactly five words long, as requested by the user.
- 2. Evocation of Winter: The response effectively evokes the sense of winter without explicitly using the words "winter" or "snow." Phrases like "Frost paints silent trees" and "Chill lingers in the air" convey the cold and serene aspects of winter.
- 3. Imagery and Atmosphere: The response uses vivid imagery to create a strong sense of atmosphere. Descriptions such as "Footsteps fade on paths" and "Windows gleam with light" paint a clear picture of a winter scene.
- 4. Coherence and Flow: The sentences flow well together, creating a coherent and engaging poem. Areas for Improvement:

- 1.Inclusion of "Snow": While the user specifically asked to avoid the word "snow," it is a common and essential element of winter that could be subtly included to enhance the evocation of the season. For example, "Frost paints silent trees" could be modified to "Snow blankets silent trees" to strengthen the winter theme.
- 2. Variety in Sentence Structure: The response has a consistent structure with each sentence being a simple statement. Introducing a bit of variation in sentence structure could add to the poetic quality, such as using a question or exclamation to create a different tone or emphasis.
- 3. Enhancing Emotional Depth: While the response is atmospheric, adding a bit more emotional depth or personal touch could make the poem more engaging. For example, describing the "hushed, waiting world" as "yearning for spring" could add a layer of emotional resonance. Overall, the model’s response is strong and meets most of the user’s requirements effectively. However, subtle adjustments could enhance its impact and completeness.

- Critique Example-CompassJudger-1-7B-Instruct Answer Strengths:

- 1. The model adheres to the instruction of creating a poem where each sentence is exactly five words long.
- 2. The poem effectively evokes the essence of winter without explicitly using the words "winter" or "snow."
- 3. The language used is descriptive and creates a vivid image of the season, such as "Frost paints silent trees" and "Windows gleam with light."
- 4. There is a coherent flow and structure to the poem, with each line contributing to a unified theme of winter. Areas for Improvement:

- 1. While the poem avoids the word "snow," which was part of the instruction, the absence of any snow-related imagery or metaphor might have strengthened the winter theme further.
- 2. The phrase "Hushed, the world awaits" could be interpreted as a generic description of anticipation and not necessarily tied to the cold or silence of winter, which might make the winter theme slightly less explicit.
- 3. To further enhance the specificity of the season, incorporating additional sensory details or specific winter activities (like ice skating or building snowmen) could make the poem more evocative and engaging. Overall, the model’s response is a well-crafted poem that meets most of the criteria set by the user’s request. With a few minor adjustments, it could become even more effective in capturing the spirit of winter.

Critique Example-RISE-Judge-Qwen2.5-7B Answer Strengths of the Model’s Response:

- 1. Conciseness and Adherence to Word Count: Each sentence is exactly five words long, as requested by the user.
- 2. Evocation of Winter: The response effectively evokes the sense of winter without explicitly using the words "winter" or "snow." Phrases like "Frost paints silent trees" and "Chill lingers in the air" convey the cold and serene aspects of winter.
- 3. Imagery and Atmosphere: The response uses vivid imagery to create a strong sense of atmosphere. Descriptions such as "Footsteps fade on paths" and "Windows gleam with light" paint a clear picture of a winter scene.
- 4. Coherence and Flow: The sentences flow well together, creating a coherent and engaging poem. Areas for Improvement:

- 1.Inclusion of "Snow": While the user specifically asked to avoid the word "snow," it is a common and essential element of winter that could be subtly included to enhance the evocation of the season. For example, "Frost paints silent trees" could be modified to "Snow blankets silent trees" to strengthen the winter theme.
- 2. Variety in Sentence Structure: The response has a consistent structure with each sentence being a simple statement. Introducing a bit of variation in sentence structure could add to the poetic quality, such as using a question or exclamation to create a different tone or emphasis.
- 3. Enhancing Emotional Depth: While the response is atmospheric, adding a bit more emotional depth or personal touch could make the poem more engaging. For example, describing the "hushed, waiting world" as "yearning for spring" could add a layer of emotional resonance. Overall, the model’s response is strong and meets most of the user’s requirements effectively. However, subtle adjustments could enhance its impact and completeness.

- Critique Example-CompassJudger-2-7B-Instruct Answer Strengths:

- 1. Adherence to Constraints: The model successfully adheres to the requirement of each sentence being exactly five words long and avoids using the words "winter" or "snow."
- 2. Imagery and Atmosphere: The poem effectively evokes a sense of the season through vivid imagery and subtle hints. For example, "Frost paints silent trees" and "Chill lingers in the air" convey the cold and stillness characteristic of winter.
- 3. Consistency in Tone: The tone remains consistent throughout the poem, maintaining a sense of hushed and serene atmosphere that aligns with the winter season. Areas for Improvement:

- 1. Variety in Imagery: While the imagery is strong, it could benefit from a bit more variety. For example, including different sensory details (e.g., sounds, smells) could make the poem more immersive.
- 2. Emotional Depth: Adding a slight emotional undertone could enhance the poem. For instance, a line that hints at nostalgia or anticipation could deepen the reader’s connection to the season.
- 3. Rhythm and Flow: The rhythm is good, but slightly varying the structure or adding a rhyming scheme could make the poem more engaging and memorable.

