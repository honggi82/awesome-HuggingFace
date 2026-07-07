# arXiv:2508.02150v1[cs.AI]4Aug2025

### Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models’ Instruction Following

Qingyu Ren1*, Qianyu He1*, Bowei Zhang2, Jie Zeng1, Jiaqing Liang2†, Yanghua Xiao1† Weikang Zhou3, Zeye Sun3, Fei Yu3 1Shanghai Key Laboratory of Data Science, College of Computer Science and Artificial Intelligence, Fudan University, 2School of Data Science, Fudan University, 3Ant Group {qyren24,qyhe21,bwzhang24, jzeng23}@m.fudan.edu.cn, {liangjiaqing, shawyh}@fudan.edu.cn

#### Abstract

Reasoning models excel in complex problem solving but exhibit a concerning tradeoff between reasoning capabilities and instruction following abilities. Existing approaches for improving instruction following rely on stronger external models, creating methodological bottlenecks and practical limitations including increased costs and accessibility constraints. We propose a self-supervised RL framework that leverages reasoning models’ own internal signals to improve instruction following capabilities without external supervision. Extensive experiments demonstrate that our framework significantly improves instruction following capabilities while maintaining reasoning performance, offering a scalable and cost-effective approach to enhance instruction following in reasoning models. The data and code are publicly available at https://github.com/Rainier-rq/verl-if.

#### 1 Introduction

Reasoning models have excelled in various reasoning domains (OpenAI, 2024; Guo et al., 2025; Seed et al., 2025). Their instruction following capabilities (i.e., the ability to follow multiple constraints simultaneously) are crucial

* Equal contribution. † Corresponding author.

R1-Distill-Qwen-1.5B

R1-Distill-Qwen-1.5B-IF

MMLU-Pro

R1-Distill-Qwen-7B

R1-Distill-Qwen-7B-IF

Qwen2.5-7B-Instruct

Qwen2.5-7B-Instruct-IF

IFEval

GPQA

80

60

40

20

CFBench

AIME2024

ComplexBench

Figure 1: Performance comparison of distilled, instruct models and their variants optimized by our method (-IF) on reasoning tasks (AIME2024, GPQA, MMLU-Pro) and instruction following tasks (IFEval, CFBench, ComplexBench).

to ensure practical use in real-world applications. On one hand, real-world conversations with human users often contain multiple constraints in the instructions (Deshpande et al., 2025). On the other hand, reliable instruction following is essential for reasoning models to excel in complex agentic tasks (Qi et al., 2025).

However, reasoning models exhibit a concerning trade-off between reasoning capabilities and instruction following abilities. Fig. 1 illustrates this phenomenon. Current approaches show a

[Figure 1]

Reward Model Training

Training data Collection for Reward Model

:output i :constraint :reward

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|1 2 3 4<br><br>|
|---|
|✘ ✘ ✘ ✘|
|✘ ✘ ✘<br><br>[Figure 6]|
|✘ ✘<br><br>[Figure 7]<br><br>[Figure 8]|
|✘<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]|
|[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]|

## ⊕ ⊕ ⊕ ⊕

Seed Ins

Policy Model

###### 1 2 3 4

Seed Ins Seed Ins Seed Ins Seed Ins

＋ 1

[Figure 16]

whether the response adheres to this constraint?

＋ 1 ＋ 2

Reward Model

＋ 1 ＋ 2 ＋ 3

＋ 1 ＋ 2 ＋ 3＋ 4

RL Training

hard constraint

Reward

Rollout

[Figure 17]

[Figure 18]

2 4

... what strategies to quickly enhance my project portfolio?Include a case study that highlights a successful product launch. Your entire response should include exactly 3 sentences. Adopt a persuasive tone as if presenting to a board of directors. The whole response should be wrapped in double quotes.

[Figure 19]

- 1

- 2

- 3

[Figure 20]

Code

= .

[Figure 21]

Prob( , )

1

Policy Model

 .  .

... Prob( , )

...

Reward Model

3

4

soft constraint

- Figure 2: Overview of our RL framework. We first self-supervisedly construct reward model training data through curriculum decomposition for constraint-wise binary classification. Then, we efficiently train the policy model with both hard constraints via rule-based verification and soft constraints via the reward model.

clear bias: instruction-tuned models excel in instruction following while inferior in reasoning capabilities (Team, 2024), while reasoning models prioritize reasoning performance but underperform in complex instruction following tasks (Guo et al., 2025). This trade-off poses challenges for reasoning models in real-world applications that require both capabilities.

lems. Methodologically, the teaching model’s capabilities limit the student model’s potential improvement (Xu et al., 2024). Practically, this approach introduces increased accessibility constraints, as stronger models are often proprietary or computationally expensive (Sun et al., 2024).

Hence, we leverage the reasoning model’s own internal signals to improve its instruction following capabilities. To achieve this goal, we choose RL as our training paradigm for several reasons. First, RL does not require high-quality outputs from external models, eliminating the dependency bottleneck (Shao et al., 2024; Schulman et al., 2017). Second, RL only requires instructions with reward signals, and constraints in complex instructions are inherently verifiable (Yang et al., 2025), making RL naturally suited for our framework. However, applying RL without external supervision presents three

Existing approaches for improving instruction following abilities have a key limitation: they rely on stronger external models. Some works distill training data from stronger models for supervised fine-tuning (Sun et al., 2024) or collect pairwise preference data for direct preference optimization (He et al., 2024). Others adopt reinforcement learning (RL), but require stronger models as reward models or for distilling reward model training data (Qin et al., 2025). This dependency creates several prob-

key challenges: (1) Obtaining effective training data and reward signals, as samples that are too hard or too easy can lead to ineffective optimization (Yu et al., 2025). (2) Modeling rewards for different constraint types, as most works only model hard constraints (Pyatkin et al., 2025), which can be verified using explicit rules, while many instructions contain soft constraints requiring human judgment (Jiang et al., 2023; Zhang et al., 2024). (3) Training efficiently with RL, as generative reward models are computationally expensive (Qin et al., 2025).

To address these challenges, we propose an efficient self-supervised RL framework that improves reasoning models’ instruction following capabilities without external supervision. First, to address sparse learning signals from challenging multi-constraint instructions (Yu et al., 2025), we decompose multi-constraint instructions into simpler instructions with incrementally increasing constraint numbers. Second, to address soft constraints that require semantic understanding (Ren et al., 2025), we establish reward signals for soft constraints without any external supervision. Third, we design an efficient constraint-wise binary classification approach that scores each constraint individually before aggregating the results, achieving computational efficiency while maintaining effectiveness.

Overall, our contributions are as follows: (1) We propose a self-supervised RL framework that improves instruction following capabilities of reasoning models using only internal signals, eliminating reliance on stronger external models. (2) We design an efficient reward modeling approach that handles soft constraints in complex instructions through constraint-wise binary classification. (3) Extensive experiments demonstrate that our framework significantly improves instruction following capabilities while maintaining reasoning performance.

#### 2 Related Work

##### 2.1 Complex Instruction Following Improvement

Some works distill responses from stronger models through supervised fine-tuning (Sun et al., 2024; Qin et al., 2025) or collecting pairwise preference data for direct preference optimization (He et al., 2024; Qi et al., 2024). Others adopt self-play approaches that enhance model capabilities through code generation verification (Dong et al., 2024) or training additional refiner models (Cheng et al., 2024). Unlike these approaches, we do not rely on stronger models and instead train exclusively through reinforcement learning based on the model’s own capabilities.

##### 2.2 Reinforcement Learning for Complex Instruction Following

Some works use rule-based rewards for hard constraints (Lambert et al., 2024; Pyatkin et al., 2025), but these methods cannot generalize to soft constraints. Other works employ stronger reasoning models (Peng et al., 2025) or closedsource models (Liu et al., 2025) as judges for soft constraint reward modeling. However, they either rely on a stronger model as the reward model or utilize the stronger model to distill data for reward model training. Additionally, existing approaches often suffer from computational inefficiency. Different from these works, our reward model is independent of stronger reasoning models, requires no distillation, and operates more efficiently.

#### 3 Method

Our framework consists of three main stages. First, we construct a multi-constraint instruction dataset and decompose complex instructions into incremental constraint curricula to provide

dense training signals. Second, as illustrated in Fig. 2, we train constraint-wise binary classification reward models using the self-supervised data, combining them with rule-based verification for comprehensive reward modeling. Finally, we apply GRPO algorithm (Shao et al., 2024) to optimize the policy model using the composite reward signals.

##### 3.1 Dataset Construction

Our training dataset consists of two main components: (1) synthetically generated multiconstraint instructions, and (2) general reasoning data from math and science domains to maintain the model’s overall capabilities.

Complex Instruction Synthesis. To ensure diversity in our multi-constraint instruction dataset, we cover both hard and soft constraint types (Ren et al., 2025). We begin by collecting a diverse set of 3,000 seed instructions from different sources (Köpf et al., 2024; Wang et al., 2022a,b; Li et al., 2025). Subsequently, we systematically add multiple constraints to these seed instructions, generating comprehensive multi-constraint instruction datasets. For hard constraints, we include 23 types, such as JSON format and frequency of all-capital words. For soft constraints, we include 25 types, such as role-based constraints and emulating specific authors’ styles. Details of the constraint types can be found in Appx. A.1.2.

General Reasoning Data Integration. To maintain the general abilities of the model, we integrate reasoning tasks from math and science domains into our training dataset. Specifically, we select 4,501 math problems from the DeepScaleR-Preview-Dataset (Luo et al., 2025) and 1,929 science questions from SciKnowEval (Feng et al., 2024).

Curriculum # Instruct. # Cons. # Soft. #Hard.

L1 2806 2806 1578 1228 L2 2745 5490 3203 2287 L3 2700 8100 4833 3267 L4 2700 10800 6481 4319 L5 2619 13095 8101 4994

Table 1: Statistics of curricula. #Instruct, #Constraints, #Soft, and #Hard refer to the number of instruction constraints, total constraints, soft constraints, and hard constraints, respectively.

Incremental Constraint Curriculum. Complex instruction-following is challenging (Zhang et al., 2024), leading to sparse reward signals during RL training (Yu et al., 2025). We decompose complex instructions for progressive learning. Given a multi-constraint instruction x with constraints {c1 ⊕ c2 ⊕ ... ⊕ cn}, we create curriculum levels Lk where level k contains sub-instruction xk with the first k constraints: xk = x with constraints {c1,c2,...,ck}. This creates a progressive curriculum from singleconstraint (L1) to full multi-constraint instructions (Ln). Statistical details are in Tab. 1.

##### 3.2 Reward Modeling

To model constraint satisfaction, we design different reward mechanisms for hard and soft constraints to produce constraint-level rewards.

Hard Constraint Modeling. For hard constraints that can be directly verified using explicit rules (Pyatkin et al., 2025), we adopt programmatic verification. For an input example (o,c), we define a binary constraint-level reward function:

Rh(o,c) =

1, if o satisfies constraint c 0, otherwise

Soft Constraint Modeling. To model soft constraints that cannot be verified through rules,

Reward Model Kendall Tau Coefficient Position Consistency Our Dataset 94.0 97.0

- Table 2: Agreement Between constructed dataset and human annotation. The detailed evaluation setup is provided in Appx. A.1.3

avoid external supervision, and achieve efficiency, we train a binary classification reward model using self-supervised data from §3.1 without external labels.

During constraint decomposition, a natural relationship emerges: for constraint ck, the response ok (generated for instruction with constraint ck) is likely to satisfy it, while ok−1 (generated for instruction without ck) does not. This allows us to construct training samples: (1) Positive sample (ok,ck,label = 1): response satisfies the constraint, (2) Negative sample (ok−1,ck,label = 0): response does not satisfy the constraint. We define a constraint-level reward function f(o,c) → [0,1] that estimates the probability that response o satisfies constraint c. The model is trained to minimize the Binary Cross-Entropy loss:

n

L = −

k=1

log f(ok, ck) + log 1 − f(ok−1, ck) .

- As shown in Tab. 2, our self-supervised dataset demonstrates high consistency with humans.

##### 3.3 RL Training

With the constraint-level reward signals established, we introduce how to utilize these models during reinforcement learning training to predict sample-level rewards for policy optimization.

Reward Model Usage During Training. For soft constraints, the trained reward model f(o,c) takes a response o and constraint c as input, producing logits over two classes (satisfy/not sat-

isfy). These logits are converted to probabilities using softmax:

exp(logits[1]) exp(logits[0]) + exp(logits[1])

Rs(o, c) =

This gives a scalar reward value Rs(o,c) ∈ [0,1] representing the probability that response

- o satisfies soft constraint c.

Sample-Level Reward Prediction. We aggregate constraint-level rewards into sample-level rewards. For instruction xk with constraints {c1⊕c2,...,ck} and policy-generated response

- ok, the sample-level reward is:

k

Rs(ok, ci), if ci is soft Rh(ok, ci), if ci is hard

1 k

Rf =

ri, ri =

i=1

For reasoning tasks, we assign Rf = 1 for correct answers and Rf = 0 otherwise. This composite reward Rf ∈ [0,1] captures the overall constraint satisfaction of the response and serves as the reward signal for GRPO optimization.

#### 4 Experiment

##### 4.1 Experiment Setup

We experiment on reasoning LLMs distilled from R1 (R1-Distill-Qwen-1.5B/7B, R1-0528Qwen3-8B (Guo et al., 2025)). To investigate the effect of incorporating instruction following reasoning data during the cold-start phase of reasoning model training, we also trained a checkpoint Qwen2.5-7B-Instruct-R based on Qwen2.5-7B-Instruct (Team, 2024) (details in Appx. A.3.4). IF denotes models trained with our method1. We evaluate instruction-following ability on multi-constraint benchmarks and general ability on scientific/mathematical reasoning benchmarks2.

- 1Baselines include instruction-following optimized models detailed in Appx. A.3.2.
- 2Details in Appx. A.3.3

|In-Domain<br><br>|Out-of-Domain|
|---|---|
|IFEval CFBench FollowBench|ComplexBench WritingBench Collie|
|Pr.(L) CSR ISR PSR HSR|Overall Avg. Avg.|

Models Base Model

IR-1.5B∗ Qwen-1.5B 57.7 N/A N/A N/A 37.8 N/A N/A N/A Conifer-7B-DPO Mistral-7B 52.3 55.0 18.0 26.0 50.0 48.1 3.2 17.8 Crab-7B-DPO Mistral-7B 57.7 62.0 25.0 32.0 49.4 59.0 4.5 19.6 TULU 3 Persona IF LLaMA-8B 69.9 67.0 28.0 38.0 44.2 53.9 4.4 16.4 UltraIF-8B-DPO∗ LLaMA-8B 75.4 N/A N/A N/A N/A N/A N/A N/A RECAST-30K-RLVC∗ LLaMA-8B 77.4 N/A N/A N/A 63.2 N/A N/A N/A SPAR-8B-DPO∗ LLaMA-8B 82.4 N/A N/A N/A N/A N/A N/A N/A

- VERIF-7B∗ Qwen-7B 79.5 N/A 68.0 N/A N/A N/A N/A N/A

- VERIF-8B∗ LLaMA-8B 87.1 N/A 72.0 N/A N/A N/A N/A N/A

Qwen2.5-7B-Instruct Qwen-7B 73.9 72.0 38.0 48.0 55.1 66.1 5.7 36.3 Qwen2.5-7B-Instruct-R Qwen-7B 72.3 72.0 36.0 46.0 49.4 63.3 5.8 35.7 Qwen2.5-7B-Instruct-IF Qwen-7B 83.9 76.0 40.0 52.0 52.7 66.7 5.9 38.0 Distill-Qwen-1.5B-7B Qwen-1.5B 42.3 53.0 17.0 24.0 22.7 39.8 3.9 14.0 Distill-Qwen-1.5B-IF Qwen-1.5B 58.8 59.0 20.0 28.0 26.4 43.3 4.1 14.5 Distill-Qwen-7B Qwen-7B 61.7 72.0 36.0 48.0 41.7 55.2 5.3 25.2 Distill-Qwen-7B-IF Qwen-7B 71.7 77.0 42.0 52.0 49.1 60.7 5.6 27.0 0528-Qwen3-8B Qwen-8B 79.7 89.0 66.0 75.0 60.4 68.5 7.6 36.9 0528-Qwen3-8B-IF Qwen-8B 87.1 90.0 68.0 76.0 63.8 71.1 7.1 37.1

- Table 3: The overall performance on In-Domain and Out-of-Domain benchmarks. We use bold for the best results and underlined for the second-best results. ∗ denotes results are sourced from the original paper. N/A represents datasets not evaluated in the original papers.

##### 4.2 Overall Performance

- As shown in Tab. 3, our method effectively improves the model’s instruction-following ability under both hard and soft constraints compared to the backbone models. Moreover, our method is effective across models with different architectures and sizes. Specifically, the performance of the instruct and distilled models shows that

- our method is applicable to reasoning models with different training backgrounds. Detailed results of the model’s performance under hard and soft constraints are provided in Appx. A.3.7.

##### 4.3 Generalizability

Out-of-Domain Generalization. We select additional benchmarks which contain constraints that are entirely different from training data to evaluate the model’s out-ofdomain generalization ability. As shown in Tab. 3, our method improves the model’s

instruction-following ability on out-of-domain tasks, demonstrating effective generalization.

General Abilities. Besides instructionfollowing ability, we also evaluate the model’s general ability. As shown in Tab. 4, our method maintains the model’s general abilities while enhancing its instruction-following ability. Moreover, our method can also improve the model’s general abilities on some benchmarks, such as FOLIO and MMLU-Pro.

##### 4.4 Ablation Studies

Reward Modeling. We conduct ablation studies for reward modeling under the following experimental settings: (1) w/o rule_based reward, which uses only the reward model to provide rewards. (2) w/o probability reward, which directly uses binary rewards: 0 for constraint unfollowing and 1 for constraint following instead of using probability values. As shown

Method GPQA Diamond BBEH AIME24 AIME25 FOLIO MMLU-Pro Avg. Qwen2.5-7B-Instruct 34.9 44.5 7.2 2.6 57.6 49.9 32.8 Qwen2.5-7B-Instruct-R 33.8 45.0 8.9 6.0 64.5 51.4 34.9 Qwen2.5-7B-Instruct-IF 34.9 48.5 8.2 5.2 62.1 52.7 35.3 R1-Distill-Qwen-7B 49.1 47.5 51.2 38.7 69.5 56.1 52.0 R1-Distill-Qwen-7B-IF 50.0 48.0 49.9 36.0 71.9 56.2 52.0

- Table 4: Model performance on general capability benchmarks. Avg@30 refers to the average score across 30 sampled responses per question. The max tokens for inference is set to 32k.

IFEval CFBench FollowBench ComplexBench WritingBench Collie Pr. (L) CSR ISR PSR HSR Overall Avg. Avg.

Method

R1-Distill-Qwen-7B-IF 71.7 77.0 42.0 52.0 49.1 60.7 5.6 27.0 Ablation Study w/o rule_based reward 67.5 76.0 40.0 52.0 46.7 59.0 5.6 20.1 w/o probability reward 69.9 77.0 41.0 52.0 47.1 58.7 5.5 26.9 w/o incremental constraint curriculum 67.7 74.0 40.0 49.0 45.5 57.7 5.4 25.1

Table 5: Ablation study results for Reward Modeling and Incremental Constraint Curriculum.

Reward Model Kendall’s Tau PC Time Per Group

LLM-as-a-judge QwQ-32B 73.2 87.2 35.7s IF-Verifier-7B 61.2 82.0 7.4s

Reward Model Training

BT Training Model 48.8 78.8 0.4s Our Reward Model 61.2 83.4 0.3s

Table 6: Comparison of reward modeling methods.

in Tab. 5, both settings lead to degraded model performance, indicating the necessity of combining rule-based reward models to prevent reward hacking and probability-based rewards to provide denser supervision. (3) Comparison of different modeling methods. We manually label 50 groups of preference data, each consisting of an instruction with five constraints and five corresponding responses that satisfy 1 to 5 constraints respectively. We then compare the correlation between human-labeled rankings and the rankings produced by three modeling methods: (a) LLM-as-a-judge, (b) the reward

model trained with Bradley-Terry (BT) Loss, and (c) our proposed reward model. The LLMas-a-judge method refers to directly prompting a strong model to assess whether the response satisfies the given constraint, assigning the score of 1 if it does and 0 if it does not. The BT Loss can be defined as:

###### L = −log σ(ra − rb)

Here, ra denotes the reward of the preferred sample, and rb denotes the reward of the less preferred sample. As shown in Tab. 6, our reward model demonstrates better alignment with humans and achieves faster inference speed.

Incremental Constraint Curriculum. To validate the effectiveness of the Incremental Constraint Curriculum, we conduct ablation studies under the following setting: w/o incremental constraint curriculum, which does not decompose multi-constraint instructions step by step, but instead directly trains on multi-constraint

Overall

Instruction Following

Math

Science

ResponseLength

ResponseLength

ResponseLength

650

0.65

Reward

1000

1250

600

0.60

0.55

900

550

1000

0 100 200 Step

0 100 200 Step

0 100 200 Step

0 100 200 Step

Overall

Instruction Following

Math

Science

ResponseLength

ResponseLength

ResponseLength

4500

975

2200

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Reward

0.525

4400

950

0.500

4300

2000

925

0 20 40 Step

0 20 40 Step

0 20 40 Step

0 20 40 Step

- Figure 3: The reward and response length of each domain during RL training of Qwen2.5-7B-R (top row) and R1-Distill-Qwen-7B (bottom row).

0.62

0.60

0.58

Reward

0.56

0.54

0.52

0.50

w/ incremental constraint curriculum

w/o incremental constraint curriculum

0.48

10 20 30 40 50

Steps

Figure 4: Reward dynamics comparison.

instructions. As shown in Tab. 5, directly training the model with multi-constraint instructions leads to the performance drop. As illustrated in Fig. 4, the model trained w/o incremental constraint curriculum receives sparser rewards during training, which hinders its ability to learn to follow multi-constraint instructions.

##### 4.5 Training Dynamics

As shown in Fig. 3, we analyze the dynamics during training on Qwen2.5-7B-R and R1Distill-Qwen-7B across different task categories. First, rewards consistently increase during training for both models until convergence. Sec-

ond, response length dynamics reveal critical insights about incorporating reasoning data during cold-start training. For Qwen2.5-7B-R, response length increases across all domains, with instruction-following tasks showing the most significant growth. This demonstrates that mixing instruction-following reasoning data during cold-start enables the model to generate increasingly longer, more detailed responses. In contrast, the distilled model shows a different pattern: response length first increases then decreases, with science tasks showing the largest drop. Since the distilled reasoning model has been extensively trained on mathematical, programming, and reasoning tasks, its search space has been significantly constrained. This finding highlights the importance of incorporating reasoning data during the cold-start phase rather than relying solely on distilled models.

#### 5 Conclusion

We propose a self-supervised RL framework that enhances instruction following without external models. Our approach features: (1) curriculum decomposition for dense training signals, (2) self-supervised reward modeling using internal

capabilities, and (3) constraint-wise binary classification for efficiency. Experiments show significant improvements in instruction following while preserving reasoning capabilities.

#### 6 Limitations

Due to computational resource limitations, we have not validated our method on larger-scale models (e.g., 32B parameters), though our experiments on smaller models provide strong evidence of the method’s effectiveness and scalability potential. Additionally, as our primary focus centers on reward modeling design, the construction of multi-constraint datasets remains relatively limited in diversity, and the current dataset construction process could benefit from incorporating a broader range of constraint types, domains, and complexity levels. Despite these limitations, our results provide compelling evidence for the effectiveness of the proposed approach, establishing a solid foundation for future scaling and enhancement efforts.

#### References

Kaikai An, Li Sheng, Ganqu Cui, Shuzheng Si, Ning Ding, Yu Cheng, and Baobao Chang. 2025. Ultraif: Advancing instruction following from the wild. arXiv preprint arXiv:2502.04153.

Jiale Cheng, Xiao Liu, Cunxiang Wang, Xiaotao Gu, Yida Lu, Dan Zhang, Yuxiao Dong, Jie Tang, Hongning Wang, and Minlie Huang. 2024. Spar: Self-play with tree-search refinement to improve instruction-following in large language models. arXiv preprint arXiv:2412.11605.

Kaustubh Deshpande, Ved Sirdeshmukh, Johannes Baptist Mols, Lifeng Jin, Ed-Yeremai Hernandez-Cardona, Dean Lee, Jeremy Kritz, Willow E Primack, Summer Yue, and Chen Xing. 2025. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. In Findings of the Association

for Computational Linguistics: ACL 2025, pages 18632–18702.

Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. 2024. Self-play with execution feedback: Improving instruction-following capabilities of large language models. arXiv preprint arXiv:2406.13542.

Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen. 2024. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. arXiv preprint arXiv:2406.09098.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James Coady, David Peng, Yujie Qiao, Luke Benson, and 1 others. 2022. Folio: Natural language reasoning with first-order logic. arXiv preprint arXiv:2209.00840.

Qianyu He, Jie Zeng, Qianxi He, Jiaqing Liang, and Yanghua Xiao. 2024. From complex to simple: Enhancing multi-constraint complex instruction following ability of large language models. arXiv preprint arXiv:2404.15846.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2023. Followbench: A multi-level fine-grained constraints following benchmark for large language models. arXiv preprint arXiv:2310.20410.

Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K Jain, Virginia Aglietti, Disha Jindal, Peter Chen, and 1 others. 2025. Big-bench extra hard. arXiv preprint arXiv:2502.19187.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens,

Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, and 1 others. 2024. Openassistant conversations-democratizing large language model alignment. Advances in Neural Information Processing Systems, 36.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, and 1 others. 2024. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124.

Jijie Li, Li Du, Hanyu Zhao, Bo-wen Zhang, Liangdong Wang, Boyan Gao, Guang Liu, and Yonghua Lin. 2025. Infinity instruct: Scaling instruction selection and synthesis to enhance language models. arXiv preprint arXiv:2506.11116.

Wenhao Liu, Zhengkang Guo, Mingchen Xie, Jingwen Xu, Zisu Huang, Muzhao Tian, Jianhan Xu, Muling Wu, Xiaohua Wang, Changze Lv, and 1 others. 2025. Recast: Strengthening llms’ complex instruction following with constraintverifiable data. arXiv preprint arXiv:2505.19030.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, and 1 others. 2025. Deepscaler: Surpassing o1-preview with a 1.5 b model by scaling rl. Notion Blog.

- MAA. 2024. American invitational mathematics

- examination - aime. Accessed in February 2024.

MAA. 2025. American invitational mathematics

- examination - aime. Accessed in February 2025.

OpenAI. 2024. Learning to reason with llms.

Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, and Juanzi Li. 2025. Verif: Verification engineering for reinforcement learning in instruction following. arXiv preprint arXiv:2506.09942.

Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. 2025. Generalizing verifiable instruction following. arXiv preprint arXiv:2507.02833.

Yunjia Qi, Hao Peng, Xiaozhi Wang, Amy Xin, Youfeng Liu, Bin Xu, Lei Hou, and Juanzi Li. 2025. Agentif: Benchmarking instruction following of large language models in agentic scenarios. arXiv preprint arXiv:2505.16944.

Yunjia Qi, Hao Peng, Xiaozhi Wang, Bin Xu, Lei Hou, and Juanzi Li. 2024. Constraint backtranslation improves complex instruction following of large language models. arXiv preprint arXiv:2410.24175.

Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, and Xing Sun. 2025. Incentivizing reasoning for advanced instruction-following of large language models. arXiv preprint arXiv:2506.01413.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Qingyu Ren, Jie Zeng, Qianyu He, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, and Fei Yu. 2025. Step-by-step mastery: Enhancing soft constraint following ability of large language models. Preprint, arXiv:2501.04945.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, and 1 others. 2025. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Haoran Sun, Lixin Liu, Junjie Li, Fengyu Wang, Baohua Dong, Ran Lin, and Ruohui Huang.

2024. Conifer: Improving complex constrained instruction-following ability of large language models. arXiv preprint arXiv:2404.02823.

Qwen Team. 2024. Qwen2 technical report. arXiv preprint arXiv:2412.15115.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022a. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, and 1 others. 2022b. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxing Xu, and 1 others. 2024. Benchmarking complex instructionfollowing with multiple constraints composition. Advances in Neural Information Processing Systems, 37:137610–137645.

Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, and 1 others. 2025. Writingbench: A comprehensive benchmark for generative writing. arXiv preprint arXiv:2503.05244.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. 2024. Stronger models are not stronger teachers for instruction tuning. arXiv preprint arXiv:2411.07133.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang

Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Shunyu Yao, Howard Chen, Austin W Hanjie, Runzhe Yang, and Karthik Narasimhan. 2023. Collie: Systematic construction of constrained text generation tasks. arXiv preprint arXiv:2307.08689.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Tao Zhang, Chenglin Zhu, Yanjun Shen, Wenjing Luo, Yan Zhang, Hao Liang, Fan Yang, Mingan Lin, Yujing Qiao, Weipeng Chen, and 1 others. 2024. Cfbench: A comprehensive constraintsfollowing benchmark for llms. arXiv preprint arXiv:2408.01122.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

#### A Appendix

A.1 Dataset Analysis A.1.1 Constraint Distribution

Our dataset includes instruction-following, mathematical, and scientific tasks. For instruction-following tasks, the constraints added in the instructions can be classified into soft constraints and hard constraints. Each instruction contains 1 to 5 constraints. As shown in Fig. 5, we present statistics on the amount of data across different domains, the number of soft and hard constraints, and the distribution of instructions with varying numbers of constraints.

Task Type

Constraint Category

Constraint Number Distribution

| |4501<br><br>1929<br><br>13570| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |24196<br><br>16095| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |2806<br><br>2745 2700 2700<br><br>2619| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

25000

14000

3000

12000

2500

20000

10000

2000

15000

8000

1500

6000

10000

1000

4000

5000

500

2000

0

0

0

Math Science Instruction following

1constraint2constraints3constraints4constraints5constraints

Hard constraints

Soft constraints

Figure 5: Sample Distribution across Task Types, Constraint Categories, and Number of Constraints

As shown in Tab. 7 and Tab. 8, we present examples of soft and hard constraints in the dataset.

- A.1.2 Constraint Types As shown in Tab. 10, our dataset includes vari-

ous types of constraints, covering multiple domains and tasks. For each seed instruction, we use the prompt shown in Tab. 11 with GPT4o to construct constraints, resulting in multiconstraint instructions.

- A.1.3 Evaluation Setup We manually label 50 groups of preference data, each consisting of an instruction with five constraints and five corresponding responses that satisfy 1 to 5 constraints respectively. We then compare the correlation between the humanlabeled ranking and the ranking.

##### A.2 Reward Model Training

We perform full fine-tuning on two models, Qwen2.5-1.5B-Instruct and Qwen2.5-7BInstruct, for a binary classification task aimed at determining whether a response satisfies a given constraint. Each training example consists of a prompt that concatenates the response and its associated constraint. Fine-tuning is conducted using the HuggingFace Trainer framework with

FP16 precision enabled and Deepspeed optimization (Stage 2 ZeRO) configured via a JSON file specifying automatic batch size and gradient accumulation steps, as well as support for the adamw_torch optimizer. Training uses a batch size of 1, a learning rate of 5e-6, and runs for 3 epochs. Accuracy is employed as the evaluation metric. The Qwen2.5-1.5B reward model provides reward signals for reinforcement learning (RL) training of Distill-Qwen-1.5B, while the Qwen2.5-7B reward model serves as a general reward function for other base models. All training is performed on 8 NVIDIA A100 80GB GPUs. As shown in Tab. 9, we present examples of the reward model training data.

A.3 RL Training A.3.1 Implementation Details

We apply the GRPO training using the VeRL framework. We use a distributed training setup across 3 nodes, each equipped with 8 NVIDIA A100 80GB GPUs, for a total of 24 NVIDIA A100 80GB GPUs. Prompts and responses are truncated to a maximum length of 8192 tokens. The RL algorithm employs the GRPO advantage estimator with no KL penalty (kl_coef set to 0.0). The optimizer uses a learning rate of 1e-6 with a

/* Seed instruction */

Assign a category to each text snippet identifying the main emotion expressed. Categories may include “joy,” “anger,” “sadness,” or “fear.” Sentence: NAME_2 was overwhelmed with tears of happiness as she accepted the award on stage.

/* Constraints */ Must include the word “exuberant” in at least one category description. Incorporate a scenario where a character is experiencing mixed emotions. Limit the response to a maximum of 25 words. Write the response in the style of a psychological evaluation report. Tailor the answer for an audience of high school psychology students.

Table 7: Examples of instructions with soft constraints.

/* Seed instruction */

Given a question, generate a paraphrase of that question without changing the meaning of it. Your answer should reword the given sentence, but not add information to it or remove information from it. The answer to your question should be the same as the answer to the original question. Input: Question: who does the voice of carl in phineas and ferb?

/* Constraints */ Answer with the letter "i" appearing at least 4 times. Highlight at least 2 sections of your response in markdown such as *highlighted section*. Make sure your reply is in English and all capital letters. Put your entire response inside double quotation marks.

Table 8: Examples of instructions with hard constraints.

/* Positive Example */ Question: Introduce Time Constraints: Impose a time limit for the role-play to unfold or for certain actions to be completed.

Answer: Characters identified in the story: 1. Kathy 2. Sue. The main conflict between the characters is the tension and argument that arises due to Kathy spreading rumors about Sue and Sue discovering and reacting to these rumors. To introduce a time constraint for the role-play, we can say: The events in the story unfold over the course of a week. Kathy starts the rumors on Monday, Sue finds out by Wednesday, they have the argument on Thursday, and by the end of the week, Sue stops speaking to Kathy. This time constraint adds a sense of urgency and progression to the narrative.

Label: 1 /* Negative Example */ Question: Introduce Time Constraints: Impose a time limit for the role-play to unfold or for certain actions to be completed.

Answer: The characters available in the given story are Kathy and Sue. The main conflict between the characters is the tension and argument that arises due to Kathy’s rumors about Sue, which leads to Sue finding out and subsequently never talking to Kathy again.

Label: 0

- Table 9: Examples of reward model training data.

|Constraint Type<br><br>|Definition<br><br>|Example|
|---|---|---|
|Lexical content constraint<br><br>|Must include specific terms or symbols with precise placement.<br><br>|"...must include the word ’beautiful.’"|
|Element constraint<br><br>|Include specific entities or scenarios.|"...highlights the Great Wall."|
|Semantic constraint|Focus on themes, tone, or stance.<br><br>|"Write a poem about London."|
|Word Count<br><br>|Limit the number of words.<br><br>|"A 50-word poem."|
|Sentence Count<br><br>|Limit the number of sentences.|"...three sentences."|
|Paragraph Count<br><br>|Limit the number of paragraphs.|"...divided into 3 sections."|
|Document Count|Limit the number of documents.|"...list 3 articles."|
|Tone and emotion<br><br>|Conform to specific emotional tone.|"Write a letter in an angry and sarcastic tone."|
|Form and style<br><br>|Use specified stylistic form and perception.|"Write a passage in an encyclopedic style."|
|Audience-specific|Tailored to a specific audience group.|"Write a poem for a 6-year-old."|
|Authorial style<br><br>|Emulate specific authors’ styles.|"Write a passage in the style of Shakespeare."|
|Fundamental format<br><br>|Follow standard formats like JSON, HTML, etc.<br><br>|"Output in JSON format."|
|Bespoke format|Use custom formatting protocols.|"Bold the main idea and output in unordered list."|
|Specialized format|Tailored for specific applications or domains.<br><br>|"Convert to electronic medical record format."|
|Pragmatic constraint|Adapt to context like dialects or language policy.<br><br>|"Output in English, classical Chinese, etc."|
|Syntactic constraint|Follow specific phrase and clause structures.<br><br>|"Use imperatives with nouns and verb phrases."|
|Morphological constraint<br><br>|Control over affixes, roots, and word formation.|"Output all content in lowercase English."|
|Phonological constraint|Focus on sounds, tone, and intonation.|"Single-syllable tongue twisters."|
|Role-based constraint<br><br>|Respond with specific role identity.<br><br>|"You are Confucius, how do you decide?"|
|Task-specific constraint|Address a defined situational task.<br><br>|"Work from home, how to report?"|
|Complex context constraint<br><br>|Involve multi-faceted and nested reasoning.|"On the left, 10 total, what to do?"|
|Example constraint<br><br>|Conform to patterns from example pairs.|"input:x..., output:...; input:y..., output?"|
|Inverse constraint|Narrow response space via exclusions.<br><br>|"No responses about political topics."|
|Contradictory constraint|Combine requirements that are hard to satisfy simultaneously.|"A five-character quotation, 1000 words."|
|Rule constraint<br><br>|Follow symbolic or logical operation rules.|"Each answer adds 1+1=3, then 2+2=5."|

###### Table 10: Constraint Types (Zhang et al., 2024).

|[Task Description]<br><br>1. I currently have a seed question, but the seed questions are relatively simple. To make the instructions more complex, I want you to identify and return five atomic constraints that can be added to the seed question.<br><br>2. I will provide [Seed Question] and [Constraint References], and you can use these references to propose five constraints that would increase the difficulty of the seed question.<br><br>3. [Constraint References] are just suggestions. You may choose one or more constraints from the list or propose new ones if needed.<br><br>4. Do not modify or rewrite the seed question. Your task is only to generate new constraints that can be added to it.<br><br>5. Return the added constraints in the following JSON format: json {<br><br>"c1": "<first constraint>", "c2": "<second constraint>", "c3": "<third constraint>", "c4": "<fourth constraint>", "c5": "<fifth constraint>"<br><br>}<br><br>6. Do not return anything else. No explanation, no reformulated question, no analysis—only the JSON structure.<br><br><br>[Constraint References]<br><br>1. Lexical content constraint : {Definition} {Example}<br><br>2. Word Count : {Definition} {Example}<br><br><br>. . .<br><br>25. Rule Constraint : {Definition} {Example}<br><br>[Seed Question] {raw_question}<br><br>|
|---|

###### Table 11: Prompt for generating constraints.

You are a meticulous assistant who precisely adheres to all explicit and implicit constraints in user instructions. When presented with complex, multi-constraint tasks, you follow a structured approach:

FORMAT: Your response must be structured with two distinct sections: <think> Your detailed analytical process and strategy development </think> <answer> Your final solution that adheres to all requirements </answer>

In your <think> section, employ these strategic approaches:

- 1. Constraint Analysis: Thoroughly examine each explicit and implicit constraint, identifying their interdependencies and potential conflicts.
- 2. Validation Strategy: Draft preliminary answers, then methodically verify compliance with each constraint, making iterative refinements as needed.
- 3. Sequencing Logic: Determine the optimal order for addressing constraints, prioritizing foundational requirements before tackling dependent ones.
- 4. Edge Case Consideration: Proactively identify boundary conditions and exceptions that might challenge constraint compliance.
- 5. Coherence Check: Ensure your solution maintains internal consistency while satisfying all requirements simultaneously.
- 6. Format Verification: Confirm your response adheres to all specified formatting and structural guidelines.

In your <answer> section, deliver a solution that precisely implements all requirements while maintaining natural flow and coherence. Your final response must satisfy all constraints without drawing attention to the mechanics of constraint management itself.

Here is an example: {example}

Table 12: Prompt used to construct data for cold start.

weight decay of 1e-2, and no learning rate warmup. We leverage Fully Sharded Data Parallel (FSDP) with full sharding enabled, rank0 parameter initialization, and parameter offloading to optimize GPU memory usage. The training batches are organized with a global batch size of 96, micro-batches of size 2 for updates, and micro-batches of size 16 for experience generation. Gradient clipping is applied with a max norm of 1.0. Rollouts are performed with a temperature of 1.0 and a group size of 5. Tensor parallelism of size 2 is applied. We train for a total of 260 steps.

- A.3.2 Baselines We include IR-1.5B (Qin et al., 2025), Conifer-

- 7B-DPO (Sun et al., 2024), Crab-7B-DPO (Qi et al., 2024), TULU 3 Persona IF (Lambert

- et al., 2024), UltraIF-8B-DPO (An et al., 2025), RECAST-30K-RLVC (Liu et al., 2025), SPAR-

8B-DPO (Cheng et al., 2024), and VERIF (Peng

- et al., 2025) as our baselines.

- A.3.3 Benchmarks

We evaluate instruction-following ability using benchmarks such as IFEval (Zhou et al., 2023), CFBench (Zhang et al., 2024), FollowBench (Jiang et al., 2023), ComplexBench (Wen

- et al., 2024), WritingBench (Wu et al., 2025), and Collie (Yao et al., 2023). In contrast, we assess general reasoning and knowledge capabilities with datasets including GPQADiamond (Rein et al., 2024), BBEH (Kazemi
- et al., 2025), AIME2024 (MAA, 2024), AIME2025 (MAA, 2025), FOLIO (Han et al., 2022), and MMLU-Pro (Wang et al., 2024).

- A.3.4 Cold Start

Effective instruction following requires systematic thinking patterns that can handle complex multi-constraint scenarios. We identify that

thinking patterns guided by Constraint Satisfaction Problems (CSP) are particularly beneficial for instruction following, including: (1) Constraint Decomposition - breaking down complex instructions into individual constraint components; (2) Constraint Prioritization - identifying and ordering constraints by importance and feasibility; (3) Incremental Consistency Checking - verifying constraint satisfaction step-by-step during response generation; and (4) Backtracking Search - revising responses when constraint conflicts are detected. To incorporate these CSP-guided thinking patterns, we design a specialized prompt that encourages models to explicitly reason through constraint satisfaction processes. We construct 1,361 instructions containing 1 to 7 constraints, generate responses using Claude-Sonnet-3.7-thinking as instruction following reasoning data, and then use these data for supervised fine-tuning (SFT) training. The prompt we use is detailed in Tab. 12.

We fine-tune the Qwen2.5-7B-Instruct model using full-parameter training with DeepSpeed ZeRO-3 optimization using LLaMAFactory (Zheng et al., 2024) on three NVIDIA A100 80GB GPUs. The training runs for 2 epochs with a batch size of 2 per device and a gradient accumulation step of 2. We set the learning rate to 1e-5 with a cosine scheduler and a warmup ratio of 10%. The training data is processed using 16 preprocessing workers, with a maximum sequence length of 4096 tokens. Mixed-precision training is enabled using bfloat16 (bf16).

A.3.5 Analysis of Model Response Improvement

Tab. 13 demonstrates the model’s improvement after training. Before training, the model mainly generates a detailed thought process instead of

directly producing the poem, resulting in an output that does not meet the specified constraints and has a low score. After training, the model outputs a concise poem that strictly follows all requirements—including the title format, sentence limit, and punctuation rules—with fluent and vivid content. This clear difference in output quality and adherence to constraints, along with the improved score, shows that the training effectively enhances the model’s performance.

- A.3.6 Full Training Dynamics Analysis As shown in Fig. 6, we present the model’s reward and output length changes during training under the setting without incremental constraint curriculum (w/o incremental constraint curriculum).
- A.3.7 Full Performance on Soft and Hard Constraints

As shown in Tab. 14 and Tab. 15, we present the model’s performance on the soft and hard constraint benchmarks.

|Case<br><br>|Content|
|---|---|
|Before Training<br><br>|Input: Write a poem about a curious cat. The poem must have a title wrapped in double angular brackets, i.e. «title», contain less than 13 sentences, and no commas. Don’t forget to add other punctuations.<think><br><br>Think: Alright, so the user has asked me to write a poem about a curious cat. They’ve given me specific guidelines: the title needs to be in double angular brackets, the poem should have less than 13 sentences, no commas, and include other punctuations. Hmm, okay, let me break this down. First, I need a title. It should be catchy and reflect the curiosity aspect. Maybe something like "Whisker’s Quest" or "Paws of Curiosity." I’ll go with "Curious Cat’s Journey" since it clearly states the theme. Next, the structure. The poem should have fewer than 13 sentences, so maybe around 11 to keep it concise. I’ll aim for a consistent rhyme scheme to make it flow smoothly. Let’s try AABB or ABAB, but I’ll see as I go. No commas, so I’ll have to be careful with sentence structure. Instead, I’ll use other punctuation like periods, exclamation points, and question marks to add rhythm and emphasis. Maybe some internal rhymes to make it catchy. Thinking about the content, I want to highlight the cat’s curiosity and adventure. Maybe describe the cat exploring places, meeting new things, and the excitement of discovery. Including sensory details like sights, sounds, smells can make the poem vivid. I’ll start each line with a different image related to curiosity—like whiskers, paws, exploring, meeting new friends, etc. Then, gradually build up the story of the cat’s journey. Each stanza can focus on a different aspect: meeting a friend, discovering a new place, the thrill of exploration. I should also include a question to engage the reader, something like "What’s next for our curious friend?" to leave a lasting impression. Ending with a playful line about the cat chasing its own shadow adds a nice touch and a bit of whimsy. Now, putting it all together, I’ll draft each line, ensuring it flows without commas and uses the required punctuation effectively. I’ll count the sentences to make sure it’s under 13. Let me check each line for rhythm and how well it contributes to the overall theme of curiosity and adventure. Finally, I’ll review the poem to ensure it meets all the user’s guidelines and conveys the intended emotions and imagery. Making sure it’s engaging and captures the essence of a curious cat’s journey. </think><br><br>Output: Model responds with a detailed plan and thought process instead of directly generating the poem. It includes reasoning about how to choose the title, avoid commas, structure the stanzas, and ensure vivid imagery. The final poem is long, significantly exceeding the 13-sentence limit.<br><br>Score: 0.333|
|After Training|Input: Write a poem about a curious cat. The poem must have a title wrapped in double angular brackets, i.e. «title», contain less than 13 sentences, and no commas. Don’t forget to add other punctuations.<think><br><br>Think: Alright, let’s break down the user’s query. They want a poem about a curious cat with specific constraints. The title needs to be in double angular brackets, less than 13 sentences, no commas, and include other punctuations. First, I’ll think about the structure. The poem should be concise, so each line should pack a punch. I’ll start with a title that reflects curiosity, maybe something like "The Curious Cat’s Tail" to fit the format. Next, I need to ensure each line flows well without commas. I’ll use periods or other punctuation like exclamation marks or question marks to maintain rhythm. I’ll describe the cat’s actions and environment to make it vivid. I’ll consider the imagery. A curious cat might explore, interact with other animals, and show personality. Including elements like yarn, yarn balls, and sun can add depth. I’ll make sure each line is short and impactful. Now, checking the sentence count. I’ll aim for around 10-12 lines to stay under 13. I’ll also ensure that the poem doesn’t use any commas, so I’ll structure the sentences to avoid them naturally. Finally, I’ll review the poem to make sure it meets all the criteria: title format, sentence count, no commas, and appropriate punctuation. This should create a engaging and concise poem that highlights the cat’s curiosity. </think> Output:<br><br>«The Curious Cat’s Tail» A curious cat purred soft and high Curious as a yarn ball on a hook It meandered through the sunlit yard Exploring every nook and every pore Meeting friends old and new along the way With curiosity in its bright eyes It spun tales of adventure untold Of yarn and games and endless joy The cat’s tail wagged like a flag Pointing to wonders both near and far In each glance and in each sigh The curious cat’s spirit did thrive A poem of paws and a heart full of glee Wrapped in warmth and filled with glee<br><br>Score: 1.0|

###### Table 13: Comparison of the outputs of DeepSeek-R1-Distill-Qwen-7B before and after training for the same instruction.

Overall

Instruction Following

Math

Science

ResponseLength

ResponseLength

ResponseLength

4500

975

2200

0.525

Reward

4400

950

2100

0.500

4300

2000

925

0 20 40 Step

0 20 40 Step

0 20 40 Step

0 20 40 Step

Figure 6: The reward and response length of w/o incremental constraint curriculum.

Easy Set Hard Set Full Set CSR ISR PSR CSR ISR PSR CSR ISR PSR

Method

Qwen2.5-7B-Instruct 79.0 52.0 58.0 66.0 24.0 37.0 72.0 38.0 48.0 Qwen2.5-7B-Instruct-R 80.0 52.0 58.0 65.0 20.0 34.0 72.0 36.0 46.0 Qwen2.5-7B-Instruct-IF 81.0 55.0 63.0 71.0 24.0 41.0 76.0 40.0 52.0

R1-Distill-Qwen-1.5B 58.0 24.0 29.0 48.0 10.0 18.0 53.0 17.0 24.0 R1-Distill-Qwen-1.5B-IF 64.0 30.0 35.0 53.0 11.0 20.0 59.0 20.0 28.0 R1-Distill-Qwen-7B 79.0 50.0 57.0 66.0 22.0 38.0 72.0 36.0 48.0 R1-Distill-Qwen-7B-IF 83.0 56.0 62.0 71.0 27.0 41.0 77.0 42.0 52.0 R1-0528-Qwen3-8B 95.0 83.0 86.0 84.0 50.0 64.0 89.0 66.0 75.0 R1-0528-Qwen3-8B-IF 94.0 83.0 86.0 85.0 54.0 66.0 90.0 68.0 76.0

- Table 14: Full results on CFBench. CSR, ISR, and PSR refer to Constraint Satisfaction Rate, Instruction Satisfaction Rate, and Priority Satisfaction Rate, respectively.

Method Pr. (S) Ins. (S) Pr. (L) Ins. (L) Avg. Qwen2.5-7B-Instruct 71.5 79.1 73.9 81.3 76.5 Qwen2.5-7B-Instruct-R 71.9 78.7 72.3 79.0 75.5 Qwen2.5-7B-Instruct-IF 83.7 88.7 83.9 88.8 86.3 R1-Distill-Qwen-1.5B 38.1 50.4 42.3 54.3 46.3 R1-Distill-Qwen-1.5B-IF 54.0 65.8 58.8 69.8 62.1 R1-Distill-Qwen-7B 56.7 68.3 61.7 72.5 64.8 R1-Distill-Qwen-7B-IF 65.8 75.2 71.7 80.2 73.2 R1-0528-Qwen3-8B 74.9 82.4 79.7 85.9 80.7 R1-0528-Qwen3-8B-IF 83.5 88.7 87.1 91.2 87.6

- Table 15: Full results on IFEval. “Pr.” and “Ins.” refer to prompt-level and instruction-level metrics, respectively, while “S” and “L” denote strict and loose evaluation.

