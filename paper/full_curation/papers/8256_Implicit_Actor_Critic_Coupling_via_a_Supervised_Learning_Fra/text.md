# arXiv:2509.02522v2[cs.CL]16Feb2026

## IMPLICIT ACTOR CRITIC COUPLING VIA A SUPERVISED LEARNING FRAMEWORK FOR RLVR

Jiaming Li1,2∗ Longze Chen1,2∗ Ze Gong1† Yukun Chen1,2 Lu Wang3 Wanwei He1,2 Run Luo1,2 Min Yang1†

- 1 Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences
- 2 University of Chinese Academy of Sciences 3 Ritzz-AI {jm.li4, lz.chen2, ze.gong, min.yang}@siat.ac.cn

ABSTRACT

Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have empowered large language models (LLMs) to tackle challenging reasoning tasks such as mathematics and programming. Despite its promise, the RLVR paradigm poses significant challenges, as existing methods often suffer from sparse reward signals and unstable policy gradient updates, inherent to RL-based approaches. To address the challenges, we propose PACS, a novel RLVR framework that achieves imPlicit Actor Critic coupling via a Supervised learning framework. By treating the outcome reward as a predictable label, we reformulate the RLVR problem into a supervised learning task over a score function parameterized by the policy model and optimized using cross-entropy loss. A detailed gradient analysis shows that this supervised formulation inherently recovers the classical policy gradient update while providing more stable and efficient training. Extensive experiments demonstrate that PACS significantly outperforms strong opensource models and RLVR baselines, yielding substantial average gains of +8.26% (4B) and +9.57% (8B) over base models offering a promising avenue for LLMs post-training with verifiable rewards. Our code and data are available as open source at https://github.com/ritzz-ai/PACS.

1 INTRODUCTION

Recent advancements like OpenAI-o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025) have revolutionized complex reasoning by scaling test-time compute to enable longer Chains of Thought (CoT). This progress significantly improves performance in mathematics (Shao et al., 2024; Liu et al., 2025a; Zuo et al., 2025) and programming (Lambert et al., 2024; Cui et al., 2025; Zhao et al., 2025). These achievements are largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which empowers LLMs to self-improve using verifiable outcome rewards.

Existing RLVR methods generally fall into two categories: value-model-based (e.g., PPO (Schulman et al., 2017), VAPO (Yue et al., 2025)) and value-model-free (e.g., GRPO (Shao et al., 2024; Guo et al., 2025), DAPO (Yu et al., 2025)). While both classes have demonstrated remarkable success, they face fundamental challenges stemming from the sparse nature of outcome rewards in RLVR settings, where only a single reward signal is provided after the entire response is generated.. Valuemodel-based approaches mitigate sparsity via explicit value modeling but incur high computational overhead. Conversely, value-model-free methods leverage Monte Carlo estimation to avoid this cost but suffer from high variance, often leading to advantage collapse and training instability. This challenge, rooted in the inherently sparse nature of RLVR feedback, highlights fundamental limitations of the existing RL-based paradigm. It motivates the development of alternative policy optimization strategies that can leverage direct supervision to enable more stable and efficient learning in RLVR settings.

To address the limitations of the RL-based paradigm rooted in sparse feedback, we propose PACS (imPlicit Actor Critic coupling via a Supervised learning framework). We recast RLVR as a super-

∗Equal contribution. † Ze Gong and Min Yang are corresponding authors.

[Figure 1]

- Figure 1: Comparison between RLVR and the supervised learning reformulation, where the query and output are input, and the outcome reward is treated as a predictable label.

vised learning task (as illustrated in fig. 1) by treating outcome rewards as labels to train a policyparameterized score function via cross-entropy loss. Gradient analysis reveals that this formulation inherently recovers the standard policy gradient while implicitly coupling the actor and critic through shared parameterization. This design eliminates the temporal mismatch introduced by separate value estimators and mitigates the high-variance issues associated with Monte Carlo estimates by providing stable prediction-error signals for optimization. Moreover, it reduces compute overhead compared to value-model-based training, and attenuates extreme updates that cause entropy collapse via bounded supervised residual. Overall, PACS offers a principled and efficient training paradigm that unifies policy learning and reward estimation within a coherent supervised learning framework.

Experimental results on five benchmarks demonstrate PACS’s superiority. It outperforms strong open-source models like LIMO and Eurus-2-7B with a 63.32% pass@1 accuracy. Notably, PACS 4B model (45.63% on AIME 2025) surpasses the significantly larger Qwen3-14B (36.22%). Against RLVR baselines, PACS establishes a decisive lead, particularly on complex tasks: on BeyondAIME, PACS-8B achieves 64.02% (pass@64), exceeding GRPO (57.64%) and PPO (53.80%) by approximately 6.4 and 10.2 points, respectively. Furthermore, PACS mitigates entropy collapse, achieving significantly higher solution diversity than GRPO (0.0392 vs. 0.0249).

The main contributions are summarized as follows:

- • We propose PACS, a novel RLVR framework that recasts sparse outcome rewards as supervised signals, effectively training the policy via cross-entropy loss.
- • We theoretically demonstrate that the proposed loss inherently captures policy gradient updates and implicitly unifies actor and critic through shared parameterization, enabling efficient and stable policy optimization.
- • Extensive experiments on challenging mathematical reasoning benchmarks show that PACS consistently outperforms state-of-the-art RLVR methods, achieving superior reasoning performance.
- • Generalization and diversity analyses demonstrate that PACS possesses strong transfer capabilities on out-of-domain tasks and effectively mitigates entropy collapse, fostering diverse reasoning paths that are instrumental to the model’s superior performance.

2 RELATED WORK

Reasoning Models. The emergence of reasoning models was catalyzed by OpenAI-o1 (Jaech et al.,

- 2024), which demonstrated that reasoning performance can scale significantly via test-time compute. This paradigm was further advanced by DeepSeek-R1 (Guo et al., 2025), which utilized GRPO (Shao et al., 2024) to show that rule-based rewards can elicit self-correction and reflection without dense human supervision. This paradigm has been extended through the Qwen family (Team, 2025b; Qwen et al., 2025; Yang et al., 2025) and the Kimi series (Team et al., 2025c;b), which have brought high-performance reasoning to the ecosystem. Most recently, Gemini family (Comanici et al., 2025) and the Llama-Nemotron series (Grattafiori et al., 2024; Bercovich et al.,
- 2025) have integrated multimodal perception and instruction-following into the reasoning loop, establishing a new state-of-the-art for general-purpose agents.

[Figure 2]

- Figure 2: An illustration of the PACS framework. The framework consists of three main components: (1) Reward Proxy Computation, which calculates a reward proxy rˆ based on the logprobability ratio. (2) Group Computation, which computes RLOO-based advantage scores ψ from the reward proxies. (3) Cross-Entropy Loss, which converts the RLVR problem into a supervised learning task, optimizing a scoring function parameterized by the policy with a cross-entropy loss.

Reinforcement Learning with Verifiable Reward (RLVR). RLVR has become a widely adopted strategy for enhancing the reasoning capabilities of LLMs in domains with clearly verifiable correctness, such as mathematics and programming (Guo et al., 2025; Ma et al., 2025; Chu et al., 2025; Yan et al., 2025; Zheng et al., 2025; Hao et al., 2025). Recent work has explored a wide range of RLVR techniques, which can be broadly categorized as value-model-based or value-model-free. Value-model-based methods, such as PPO (Schulman et al., 2017), VinePPO (Kazemnejad et al.,

- 2024), and VAPO (Yue et al., 2025), explicitly learn a value function to estimate the expected cumulative reward, providing stable training signals at the cost of additional computational overhead. On the other hand, value-model-free methods, such as GRPO (Shao et al., 2024), REINFORCE++ (Hu et al., 2025), and DAPO (Yu et al., 2025), avoid explicit value modeling by relying on Monte Carlo advantage estimation. While these approaches reduce modeling complexity, they often suffer from high gradient variance, which undermines training stability and performance.

- 3 METHODOLOGY

In this section, we first establish the preliminaries for RLVR before introducing PACS, a novel framework achieving imPlicit Actor–Critic coupling via Supervised learning. We then detail the overall framework, validate its effectiveness through gradient analysis, and conclude with practical implementation designs.

- 3.1 PRELIMINARIES

Let θ denote the parameters of an LLM, and let P(Q) be a distribution over queries from which a query q is sampled. The model generates an output o according to the policy πθ(·|q). A verifiable outcome reward R(q,o) ∈ {0,1} is provided by either a reward model or an external verifier which determines whether the model’s output o to the query q is correct (1) or incorrect (0). The objective of RLVR is to learn a policy πθ that maximizes the expected reward: LRLVR(θ) = Eq∼P(Q),o∼π

θ(·|q)[R(q,o)]. Various RLVR approaches (Schulman et al., 2017; Kool et al., 2019; Ahmadian et al., 2024; Shao et al., 2024; Yu et al., 2025; Yue et al., 2025) have been proposed to optimize this objective function, each seeking to effectively learn the policy πθ under the supervision of outcome rewards.

- 3.2 RECASTING RLVR VIA SUPERVISED LEARNING

In the RLVR framework, LLMs receive verifiable outcome rewards only after generating a complete response. Inspired by this observation, we propose an alternative to conventional RL-based optimization: rather than policy learning through RL from sparse reward signals, we recast the problem as a supervised learning task in which the model is trained to directly predict the outcome reward.

Concretely, instead of using the reward R(q,o) ∈ {0,1} associated with a query-output pair (q,o) to guide policy optimization via RL, we consider (q,o) as input data and treat R(q,o) as the corresponding label. The objective is to learn a mapping f(q,o) → R that accurately predicts the

reward. Specifically, this problem can be cast as a classification task where rewards serve as binary labels (e.g., correct vs. incorrect). To this end, we define the learning objective via a standard binary cross-entropy loss as follows:

L(θ) = −Eq∼P(Q),o∼π

θ(·|q) R(q,o)log σ ψ(q,o;πθ) + 1 − R(q,o) log 1 − σ ψ(q,o;πθ) ,

(1)

where σ(z) = 1+1e−z is the sigmoid function that maps real-valued scores to the [0,1] interval, representing the predicted probability of correctness. The score function ψ(q,o;πθ) is parameterized in terms of the policy model πθ, which is trained to estimate the quality of the generated outputs directly. The specific form and implementation of ψ(q,o;πθ) will be discussed in detail in section 3.4.

- 3.3 GRADIENT ANALYSIS OF OBJECTIVE FUNCTION

To gain a deeper understanding of how our objective guides learning, we analyze the gradient of the loss function L(θ) with respect to the model parameters θ. We begin by introducing the per-sample loss term:

#### l(q,o;πθ) := R(q,o)log σ ψ(q,o;πθ) + 1 − R(q,o) log 1 − σ ψ(q,o;πθ) (2)

which corresponds to the cross-entropy between the ground-truth reward R(q,o) and the prediction σ ψ(q,o;πθ) . Using this definition, the gradient of the full loss can be expressed as:

∇θL(θ) = − Eq ∇θEo∼π

θ(·|q) l(q,o;πθ)

(3)

= − Eq,o∼π

θ(·|q) ∇θ log πθ(o|q) l(q,o;πθ) + ∇θl(q,o;πθ)

We now derive the inner gradient term ∇θℓ(q,o;πθ). Applying the chain rule and recalling that the derivative of the sigmoid function is σ′(x) = σ(x)(1 − σ(x)), we simplify the gradient term as follows:

∇θℓ(q,o;πθ) = R(q,o) − σ ψ(q,o;πθ) ∇θψ(q,o;πθ). (4) Substituting eq. (4) into the expression for ∇θL(θ), we obtain the final form of the gradient:

∇θL(θ) = − Eq,o∼π

θ(·|q)

. (5)

+ R(q,o) − σ ψ(q,o;πθ) ∇θψ(q,o;πθ)

ℓ(q,o;πθ) ∇θ log πθ(o|q) ACTOR: policy improvement

CRITIC: reward estimation

In the above gradient expression, the first term corresponds to a standard policy gradient update, weighted by the per-sample cross-entropy loss. The second term serves as a reward estimation correction, adjusting the score function ψ to align the predicted reward σ ψ(q,o;πθ) with the ground-truth outcome. Notably, this formulation unifies the ACTOR and CRITIC updates within a single gradient step and shared parameter space.

Implicit Actor Critic Coupling. As shown in eq. (5), our formulation enables the learned model πθ to simultaneously fulfill two roles:

- • ACTOR: It samples outputs o conditioned on the query q, according to the policy distribution πθ(o|q).
- • CRITIC: It estimates output quality using the score function ψ(q,o;πθ), where the

sigmoid-transformed value σ ψ(q,o;πθ) represents the predicted probability of correctness and serves as the predicted reward.

The ACTOR update is modulated by the per-sample cross-entropy loss ℓ(q,o;πθ), which weights the policy gradient according to the alignment between predicted and ground-truth rewards.

The CRITIC update is driven by the residual R(q,o)−σ ψ(q,o;πθ) , providing a supervised learning signal that improves reward prediction. Improving the ACTOR changes the score ψ and thus yields a tighter supervised target for the CRITIC; in turn, the CRITIC improvements feed back to directly affect future ACTOR updates through the shared weighting term. This unified design enables implicit Actor Critic coupling without separate networks or alternating update schedules and facilitates efficient training while leveraging the strengths of both policy gradient and supervised learning paradigms.

- 3.4 SCORE FUNCTION INSTANTIATION

To instantiate the score function ψ(q,o;πθ), we require a mechanism to quantify the quality of a sampled response o for a given query q. We interpret ψ(q,o;πθ) as an advantage-like function that measures the relative quality of a response within a group of samples.

Such a formulation is particularly appropriate for our setting because it centers the score distribution with respect to the set of samples while preserving its full range. As a result, it is well suited with a sigmoid mapping, yielding outputs confined to the [0,1] interval.

Unlike value-model-free methods that estimate advantages using ground-truth rewards, we define ψ(q,o;πθ) in terms of the policy model πθ, enabling direct policy optimization. To compute this advantage efficiently, we adopt the REINFORCE Leave-One-Out (RLOO) estimator (Kool et al., 2019; Ahmadian et al., 2024). RLOO provides an unbiased estimate of the relative advantage by comparing each sampled output to others generated for the same query. For each query q, given a set of G candidate responses {o1,o2,...,ok}, the RLOO-based advantage-like score function for each output oi is defined as:

1 G − 1 j̸=i

rˆ(q,oj;πθ), (6)

ψ(q,oi;πθ) = rˆ(q,oi;πθ) −

following prior work (Rafailov et al., 2023; Cui et al., 2025), where rˆ(q,oi;πθ) is a reward proxy based on log-probability ratios:

|oi|

πθ(oi|q) πref(oi|q)

log πθ(oi,t|q,oi,<t) − log πref(oi,t|q,oi,<t) , (7)

rˆ(q,oi;πθ) = β log

= β

t=1

where πref is a fixed reference policy used to regularize the learned policy, and β is a scaling hyperparameter. However, in practical implementation, the reward proxy rˆ(q,oi;πθ) may grow progressively larger over time due to the fixed reference policy, leading to high variance and instability during training. To address this, we follow the strategy proposed in (Liu et al., 2025a), periodically hard-resetting the reference policy πref to a recent snapshot of the online policy πθ, and reinitializing optimizer states to maintain stable training dynamics. With the RLOO-estimated advantage serving as the score function, the training objective becomes:

LPACS(θ) = − Eq∼P(Q),{o

i}Gi=1∼πθ(·|q)

G

1 G

[R(q,oi)log (σ(ψ(q,oi;πθ))) + (1 − R(q,oi))log (1 − σ(ψ(q,oi;πθ)))].

i=1

(8)

Loss Function Analysis. For a correct output oi (i.e., R(q,oi) = 1), the loss reduces to log(σ(ψ(q,oi;πθ))). Minimizing this term maximizes ψ(q,oi;πθ), thereby increases the reward proxy rˆ(q,oi;πθ) and, equivalently, the probability πθ(oi|q) of generating the correct output. Conversely, for an incorrect output oi (i.e., R(q,oi) = 0), the loss becomes log(1 − σ(ψ(q,oi;πθ))). Minimizing it decreases ψ(q,oi;πθ), which in turn lowers rˆ(q,oi;πθ) and reduces the probability πθ(oi|q) of generating the incorrect output. Thus, the model learns not only to generate correct outputs, but also to avoid producing incorrect ones.

Model MATH 500 AMC 23 AIME 2024 AIME 2025 BeyondAIME Avg. Closed-source Models

Claude Opus 4.1 - 87.5 - 78.0 - Gemini 2.5 pro 96.7 100.0 88.7 86.7 58.8 86.18

Open-Source Model

Qwen3-4B 91.01 82.30 46.48 34.71 17.33 54.37 Qwen3-8B 89.94 81.74 46.98 33.83 16.28 53.75 Qwen3-14B 92.07 84.88 49.06 36.22 18.59 56.16 Deepthought-8B 45.07 16.58 1.82 0.44 0.76 12.93 Eurus-2-7B-PRIME 82.07 63.65 18.39 13.88 6.67 36.93 LIMO 91.00 80.61 40.86 31.82 15.98 52.05 Marco-o1 70.48 45.12 9.22 6.95 2.77 26.91 OpenThinker3-7B 88.90 72.44 36.95 26.98 12.33 47.52 Sky-T1-7B 85.38 69.59 20.86 20.47 8.54 40.97 PACS-4B 94.80 90.45 55.10 45.63 27.16 62.63

∆(Qwen3-4B) ↑3.79 ↑8.15 ↑8.62 ↑10.92 ↑9.83 ↑8.26 PACS-8B 95.09 88.69 57.58 46.38 28.86 63.32 ∆(Qwen3-8B) ↑5.15 ↑6.95 ↑10.6 ↑12.55 ↑12.58 ↑9.57

- Table 1: Comparison of pass@1 accuracy (%) on mathematical reasoning benchmarks. “Avg.” indicates the average performance across all tasks. Bold numbers indicate the best performance. Underlined numbers indicate the second best. The values in green represent the performance gain (∆) over the model.

Overall, this formulation preserves the supervised learning structure of our original objective while incorporating relative comparisons between sampled outputs, yielding more informative learning signals1. The overall PACS framework is illustrated in fig. 2.

Mitigating Entropy Collapse. With the score function instantiated in a policy-dependent form, the gradient of the loss can be expressed as an advantage-weighted policy gradient (Appendix A.1). Unlike conventional RL, where positive rewards can push the policy toward overly deterministic behavior, our gradient (eq. (5)) is modulated by the residual R(q,o)−σ(ψ(q,o;πθ)). As predictions approach the ground truth, the residual vanishes, naturally attenuating updates and preventing overconcentration of probability mass, thereby preserving a diverse set of responses. We provide a formal proof in Appendix A.2 showing that this mechanism implicitly imposes a structural constraint that regularizes the policy towards a Gibbs distribution, fundamentally preventing mode collapse.

- 3.5 PRACTICAL CONSIDERATIONS: HANDLING DATA IMBALANCE

To address the potential distributional imbalance in generated outputs oi for q, we adopt the class imbalance treatment methodology proposed by (King & Zeng, 2001), whereby differential weights are assigned to correct and incorrect samples respectively. Specifically, when the proportion of correct and incorrect samples in the training data exhibits imbalance, we mitigate the adverse effects of sample distribution bias on model performance by adjusting weight parameters to balance the model’s attention across different categorical samples.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

Datasets & Models. We utilize the DeepScaleR dataset as the training corpus (Luo et al., 2025), which constitutes a high-quality mathematical problem-solving collection. For evaluation, we conduct extensive evaluations on five representative benchmarks: MATH 500, AMC 23, AIME 2024, AIME 2025 and BeyondAIME. The experiments utilize Qwen3-4B and Qwen3-8B (Qwen et al.,

- 2025) as base models, assessing PACS across different model scales.

1A rule-based reward function is employed, assigning a reward of 1 to correct answers and 0 to incorrect ones.

AMC 23 (pass@k) AIME 2024 (pass@k) Model k = 1 4 8 16 32 64 k = 1 4 8 16 32 64

Base 81.74 90.53 92.31 94.00 95.65 96.86 46.98 59.78 65.10 69.28 72.12 74.99 PPO 86.27 92.36 93.91 95.46 96.70 98.13 54.61 71.46 75.86 79.11 81.75 83.12 GRPO 87.83 93.17 94.71 95.92 97.91 99.69 54.56 74.52 80.08 82.42 83.21 83.33 PACS 88.69 94.28 95.36 96.03 96.72 97.35 57.58 76.24 81.21 83.30 84.14 85.00

- w/o weight 89.63 94.80 95.53 96.02 96.50 97.32 58.93 76.64 81.13 83.21 83.59 84.73

AIME 2025 (pass@k) BeyondAIME (pass@k) Model k = 1 4 8 16 32 64 k = 1 4 8 16 32 64

Base 33.83 45.86 51.19 56.12 60.54 65.74 16.28 23.00 26.75 31.36 36.65 41.74 PPO 42.42 55.67 61.70 68.25 73.99 77.43 23.38 33.60 38.83 44.10 49.11 53.80 GRPO 45.29 58.80 64.04 69.12 73.84 77.39 25.79 37.61 43.08 48.40 53.32 57.64 PACS 46.38 61.93 67.78 72.20 75.71 79.12 28.86 43.15 49.58 55.39 60.19 64.02

- w/o weight 46.15 60.07 66.29 71.98 76.10 78.23 28.52 40.43 45.55 50.10 54.35 58.97

- Table 2: Results of Qwen3-8B trained with PPO, GRPO and PACS. Bold numbers indicate the best performance. Underlined numbers indicate the second best. Due to space constraints, results on MATH 500 are shown in Table 8.

Baselines & Hyperparameters. To comprehensively evaluate the effectiveness of PACS, we compare PACS with a diverse set of baselines categorized into three groups: 1). Closed-source models: including state-of-the art models such as Claude Opus 4.1 and Gemini 2.5 Pro (Comanici et al., 2025), whose results are directly cited from prior works (Team et al., 2025a; Li et al., 2025); 2). Open-source models: including recent models utilizing advanced post-training of RL techniques, such as Qwen3 models (Yang et al., 2025), Deepthought-8B (Ruliad, 2024), Eurus-2-7BPRIME (Cui et al., 2025), LIMO (Ye et al., 2025), Marco-o1 (Zhao et al., 2024), OpenThinker3-

- 7B (Guha et al., 2025) and Sky-T1-7B (Team, 2025a); and 3). Representative RL algorithms: we implement and train PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024) on the same base models to ensure a fair comparison. The training objectives of these algorithms can be found in Appendix B. We also report the performance of base models for reference. The train batch size is set to 1,024. For each query during training, 8 responses are sampled. For PACS, β is set to 1.0. During inference, we configure the sampling parameters with a temperature of 0.6. Prompt template and detailed hyperparameter configuration can be found in the Appendix C.

Hardware. All experiments are conducted on NVIDIA H200 GPUs. We employ verl (Sheng et al., 2024), an reinforcement learning library specifically designed for LLMs. For inference, we utilize vllm (Kwon et al., 2023).

Evaluation Metrics. To mitigate sampling bias and evaluate solution diversity, we employ the pass@k metric. Instead of single trials, we calculate the unbiased estimator (Chen et al.,

(n−c

) (n

, where c is the number of correct solutions among n sam-

2021):pass@k = Ex∼D 1 −

k

)

k

ples. We generate n = 128 candidates for MATH 500, AMC 23, AIME series, and BeyondAIME to ensure statistically stable evaluations.

- 4.2 MAIN RESULTS

Comparative Analysis with Existing Models. table 1 shows that PACS significantly enhances the reasoning capabilities of base models across all benchmarks. Specifically, PACS-8B achieves an average pass@1 of 63.32%, surpassing Qwen3-8B by 9.57 points and outperforming strong baselines like LIMO (52.05%). Notably, PACS exhibits exceptional parameter efficiency: PACS-4B (62.63%) not only beats 7B-scale competitors but also outperforms the significantly larger Qwen3-14B on challenging tasks like AIME 2025 (45.63% vs. 36.22%). The gains are particularly pronounced on complex benchmarks; for instance, on AIME 2025 and BeyondAIME, PACS-8B improves upon the base model by over 12 points, effectively unlocking deep chain-of-thought capabilities.

Superior Performance against RL Baselines. Comparing algorithmic efficacy in tables 2 and 6, PACS consistently outperforms PPO and establishes a decisive advantage over GRPO on complex reasoning tasks. This superiority is most evident on the hardest benchmark, BeyondAIME. Here, PACS-8B achieves a pass@64 of 64.02%, surpassing GRPO (57.64%) by over 6 points, while PACS-

- 4B similarly improves the metric from 57.58% to 62.10%. Furthermore, PACS-8B maintains a

###### AMC 23 BeyondAIME

###### AIME 2024 AIME 2025

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

89 88 87 86 85 84 83 82

46 44 42 40 38 36 34

58 56 54 52 50 48

28 26 24 22 20 18 16

pass@1

50 100 150 200 250 300 50 100 150 200 250 300 50 100 150 200 250 300 50 100 150 200 250 300

Training Steps

Figure 3: Training dynamics of Qwen3-8B. The curves illustrate the evolution of pass@1 on math benchmarks throughout the training process.

comprehensive lead over GRPO on AIME 2024 and 2025 in both single- and multi-sample settings. These results indicate that PACS’s supervised learning formulation, which implicitly couples actor and critic updates, provides more effective optimization signals than pure RL approaches for navigating complex solution spaces.

Training Dynamics and Performance Analysis. To further investigate the optimization efficiency, we visualize the evaluation trajectories of Qwen3-8B throughout the training process in Figure 3. The curves report pass@1 on benchmarks, evaluated every 20 training steps. The results highlight PACS’s robustness across diverse problem distributions. On benchmarks like AMC 23 and the highly challenging BeyondAIME, PACS exhibits high training efficiency, rapidly establishing a decisive performance lead over RL baselines. Notably, on AIME 2025, the performance gap narrows as GRPO also demonstrates competitive learning trajectories; yet, PACS remains resilient, matching the strong baseline’s pace and securing the best final performance.

- 5 ABLATION STUDY

- 5.1 DIFFERENT β

[Figure 7]

β

k

pass@k

95 90 85 80

10

2

1

0.5

0.1

100

1

2

4

8

16

32

64

[Figure 8]

β

k

90

80 70 60 50

pass@k

10

2

1

0.5

0.1 1

2

4

8

16

32

64

[Figure 9]

β

k

pass@k

80 70 60 50 40

10

2

1

0.5

0.1 1

2

4

8

16

32

64

[Figure 10]

β

k

pass@k

60 50 40 30 20

70

10

2

1

0.5

0.1 1

2

4

8

16

32

64

AIME 2025 BeyondAIME

AMC 23 AIME 2024

Figure 4: Performance analysis of PACS with varying β. The 3D heatmaps show pass@k scores for different combinations of β values (0.1, 0.5, 1, 2, 10) and k values on AMC23, AIME-2024, AIME-2025 and BeyondAIME.

To validate the impact of β on the model’s performance in eq. (7), we design and conduct ablation experiments. Models are trained with different β ∈ {0.1,0.5,1,2,10} and their performance is evaluated in fig. 4. The results demonstrate that PACS is highly robust to variations in β. While extreme values (e.g., β = 10) maintain competitive performance on easier tasks like AMC 23, a distinct ”sweet pot” emerges in β ∈ [0.5,1.0] for maximizing pass@1 on complex reasoning tasks. Given that pass@1 is often the primary objective for reasoning models, we select β = 1 as the optimal setting, effectively balancing the reward signal strength to achieve superior reasoning performance.

- 5.2 DIFFERENT SCORE FUNCTION

To validate the effectiveness of our scoring mechanism, we compare the RLOO score function against GRPO (Shao et al., 2024) and Dr. GRPO (Liu et al., 2025c). Detailed results are shown in Appendix D.3. Our analysis reveals that PACS with RLOO generally yields superior performance, particularly on complex reasoning tasks. For instance, on BeyondAIME benchmark, PACS achieves a pass@1 of 27.16%, consistently outperforming both GRPO (26.41%) and Dr. GRPO (26.67%). We attribute this robustness to the leave-one-out mechanism of RLOO, which provides unbiased advantage estimates with lower variance compared to the group-mean baselines used in GRPO variants.

###### Entropy Loss

- 0

- 0.1
- 0.2
- 0.3
- 0.4

1 30 60 90 120

PPO GRPO PACS

steps

AIME 2024 AIME 2025

Qwen3-4B

[Figure 11]

AIME 2024 AIME 2025

Qwen3-8B

[Figure 12]

0

14

[Figure 13]

[Figure 14]

2

16 16

[Figure 15]

[Figure 16]

2 15

[Figure 17]

[Figure 18]

18

Entropy Collapse

Entropy Collapse

Qwen3-8B

Qwen3-4B

-0.30 0.00 -0.10 -0.15 0.00 0.10

-0.10

0.075

0.00 0.00

- -0.10

0.15

0.10

0.00

- -0.10

-0.15 0.00 0.15 -0.15 0.00 0.15

-0.15

0.15

0.00

-0.10

- -0.20 0.00 0.15

0.15

0.00

-0.10 0.00 0.15

- -0.10

0.10

0.00

-0.10 0.00 0.10

- -0.10

0.00

0.10

-0.15 0.00 0.15

- -0.15

0.10

0.00

Figure 5: Exploration and Diversity Analysis. (a) Entropy loss dynamics for Qwen3-4B (top) and 8B (bottom). Unlike baselines that suffer from entropy collapse, PACS maintains higher entropy, enabling sustained exploration.(b) Centered PCA projection of correct solutions for sampled problems (ID shown in bottom-right). The broader semantic coverage of PACS (Blue) compared to GRPO (Red) visually confirms superior diversity.

- 5.3 EFFECT OF WEIGHT INTEGRATION

To validate the effectiveness of the weight mechanism in PACS, we conduct ablation experiments by comparing the performance of PACS with and without the weight component as shown in tables 2 and 6 to 8. For Qwen3-4B, PACS systematically outperforms the unweighted variant across benchmarks; notably, it achieves a pass@1 of 90.45% on AMC 23 and 55.10% on AIME 2024, surpassing the w/o weight PACS scores of 89.18% and 54.56% respectively. This advantage is further amplified on complex tasks with the larger Qwen3-8B model. While performance on simpler tasks remains comparable, PACS establishes a clear lead on the hardest benchmark, BeyondAIME. It not only improves the pass@1 accuracy to 28.86% compared to 28.52% , but more importantly, achieves a remarkable 64.02% at pass@64, exceeding the unweighted variant’s 58.97%.

6 GENERALIZATION AND DIVERSITY ANALYSIS

In this section, we analyze the broader capabilities of PACS by first evaluating its generalization to out-of-domain tasks to rule out overfitting, and then exploring the solution diversity that underpin this mechanism.

- 6.1 GENERALIZATION OF OUT-OF-DOMAIN TASKS

PPO GRPO PACS

- 0

- 0.1
- 0.2
- 0.3
- 0.4

- 1 60 120 180 240 300

(a) (b)

A key challenge in RLVR is preventing the policy from over-specializing to the training domain, which can lead to a degradation of general capabilities. To evaluate whether PACS learns fundamental reasoning skills that transfer across domains, we assess the Qwen3-8B model trained on the DeepScaleR dataset (math) on two out-of-domain benchmarks: LiveCodeBench (Jain et al., 2024)2 and GPQA (Rein et al., 2023). For these assessments, we utilize the toolkit provided by GLM-4.5, adhering to the default settings for all experimental configurations. As shown in Figure 6, PACS demonstrates superior crossdomain generalization compared to PPO and GRPO. On LiveCodeBench, PACS achieves 45.96, significantly outperforming the Vanilla model

Vanilla PPO GRPO PACS

60

52.46 48.04 48.48 48.55

50

45.96 35.87 38.04

40

Score

31.06

30

20

10

0

LiveCodeBench

###### GPQA

Figure 6: Performance on out-of-domain tasks. The chart reports the evaluation scores of Qwen3-8B on LiveCodeBench and GPQA. PACS achieves substantial gains over PPO and GRPO.

2Evaluations are performed on the LiveCodeBench dynamic subset spanning the period from 7/1/2024, through 1/1/2025.

(31.06) as well as PPO (35.87) and GRPO (38.01), which suggests that the reasoning chains optimized by PACS effectively transfer to code logic generation while baseline RL methods yield diminishing returns. This advantage is further corroborated on GPQA, where PACS attains 52.46, surpassing both GRPO (48.55) and PPO (48.48). These results indicate that PACS does not merely memorize domain-specific templates; instead, by treating outcome rewards as supervised signals within a stable optimization framework, it enhances the model’s intrinsic reasoning capabilities.

6.2 EXPLORATION AND DIVERSITY

We observe that PACS effectively mitigates the entropy collapse issue commonly found in RL baselines, leading to semantically richer and more diverse solutions as shown in fig. 5(a). To quantify the diversity of the generated reasoning paths, we utilize the average pairwise consine distance between the embeddings of responses. A higher diversity score (Sdiv) indicates greater semantic variation. The definition of Sdiv is detailed in Appendix E. We assess the response diversity on two representative benchmarks: AIME 2024 and AIME 2025. We specifically compare PACS with GRPO us-

Model Method AIME2024 AIME2025 Qwen3-4B

PACS 0.0278 0.0300 GRPO 0.0260 0.0266

PACS 0.0272 0.0392 GRPO 0.0217 0.0249

Qwen3-8B

Table 3: Comparison of diversity scores on AIME 2024 and AIME 2025. PACS consistently exhibits higher diversity than GRPO across different model scales.

ing the Qwen3-4B and Qwen3-8B. Note that we compute Sdiv on correct responses, as our goal is to achieve diverse yet valid solutions. As shown in table 3, PACS consistently surpasses GRPO in diversity (0.0392 vs. 0.0249). While the rigid nature of mathematical solutions limits absolute scores, PACS achieves a substantial relative improvement, effectively mitigating mode collapse and fostering semantically distinct reasoning paths.

To intuitively compare the diversity of reasoning paths, centered PCA approach is employed. We randomly sample some problems. fig. 5(b) displays the 2D PCA projection of the embeddings for correct solutions generated by PACS (Blue) and GRPO (Red). The data is mean-centered to align the centroids of both distributions at the origin (0,0). The shaded regions (KDE contours) and scatter points illustrate that PACS covers a significantly larger area in the semantic space, indicating a more diverse set of reasoning paths compared to the more concentrated distribution of GRPO.

- 7 CONCLUSION

In this work, we propose PACS, a novel RLVR algorithm that achieves implicit actor-critic coupling via a supervised learning framework. By treating outcome rewards as supervised targets and parameterizing the score function with policy log probabilities, PACS mitigates the sparse reward and training instability challenges which are inherent to existing RL-based methods. Extensive experiments on various benchmarks demonstrate that PACS significantly outperforms strong baselines including PPO and GRPO, achieving superior task performance while maintaining healthier policy entropy for sustained exploration. These results highlight PACS as a promising approach for advancing RLVR.

REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨¨ un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosallanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi, Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek,

Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin, Matvei Novikov, Daria Gitman, Evelina Bakhturina, Prasoon Varshney, Makesh Narsimhan, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi Kong, Markus Kliegl, Rabeeh Karimi Mahabadi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha, Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Abhinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor-Yiing Su, Guyue Huang, Terry Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru, Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf, Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya, Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam, Smita Ithape, Karthik Ramamoorthy, Yuting Wu, Suguna Varshini Velury, Omri Almog, Joyjit Daw, Denys Fridman, Erick Galinkin, Michael Evans, Shaona Ghosh, Katherine Luna, Leon Derczynski, Nikki Pope, Eileen Long, Seth Schneider, Guillermo Siman, Tomasz Grzegorzek, Pablo Ribalta, Monika Katariya, Chris Alexiuk, Joey Conway, Trisha Saar, Ann Guan, Krzysztof Pawelec, Shyamala Prayaga, Oleksii Kuchaiev, Boris Ginsburg, Oluwatobi Olabiyi, Kari Briski, Jonathan Cohen, Bryan Catanzaro, Jonah Alben, Yonatan Geifman, and Eric Chung. Llama-nemotron: Efficient reasoning models, 2025. URL https://arxiv.org/abs/2505.00949.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala,

Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur ¸Celebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Ro-

driguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yaru Hao, Li Dong, Xun Wu, Shaohan Huang, Zewen Chi, and Furu Wei. On-policy rl with optimal reward baseline. arXiv preprint arXiv:2505.23585, 2025.

Jian Hu, Jason Klein Liu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Refining credit assignment in rl training of llms. arXiv preprint arXiv:2410.01679, 2024.

Gary King and Langche Zeng. Logistic regression in rare events data. Political Analysis, 9(2): 137–163, 2001. doi: 10.1093/oxfordjournals.pan.a004868.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! drlStructPred@ICLR, 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Xinyuan Li, Murong Xu, Wenbiao Tao, Hanlun Zhu, Yike Zhao, Jipeng Zhang, and Yunshi Lan. Ride: Difficulty evolving perturbation with item response theory for mathematical reasoning,

2025. URL https://arxiv.org/abs/2511.04120.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025c. URL https: //arxiv.org/abs/2503.20783.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, and Jia Li. S2r: Teaching llms to self-verify and self-correct via reinforcement learning. arXiv preprint arXiv:2502.12853, 2025.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Ruliad. Deepthought-8b: A small and capable reasoning model, 2024. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

- 5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang, Zilin Zhu, Bohan Zhang, Bosi Wen, Bowen Wu, Bowen Xu, Can Huang, Casey Zhao, Changpeng Cai, Chao Yu, Chen Li, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li, Congfeng Yin,

Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji Li, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jianyang Yan, Jiaqi Wang, Jiayi Gui, Jiayue Zhao, Jie Liu, Jijie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan, Jingxuan Li, Jingzhao Du, Jinhua Du, Jinxin Liu, Junkai Zhi, Junli Gao, Ke Wang, Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, Mingshu Zhai, Pengfan Du, Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang, Shaoyou Lu, Shijie Li, Shuang Li, Shuang-Li, Shuxun Yang, Sibo Yi, Tianshu Yu, Wei Tian, Weihan Wang, Wenbo Yu, Weng Lam Tam, Wenjie Liang, Wentao Liu, Xiao Wang, Xiaohan Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang, Yuhang Li, Yuhao Wu, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu, Zhen Yang, Zhengda Zhou, Zhongpei Qiao, Zhuoer Feng, Zhuorui Liu, Zichen Zhang, Zihan Wang, Zijun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zixuan Li, Zuodong Zhao, Wenguang Chen, Jidong Zhai, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models, 2025a. URL https://arxiv.org/abs/2508.06471.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Xiaofei Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yangkun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian Zhao, Yikai Zhao, Huabin Zheng, Shaojie Zheng, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence, 2025b. URL https://arxiv.org/abs/2507.20534.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, C Chen, C Li, C Xiao, C Du, C Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms, 2025. URL https://arxiv. org/abs/2501.12599, 2025c.

NovaSky Team. Sky-t1: Train your own o1 preview model within $450. https://novaskyai.github.io/posts/sky-t1, 2025a.

Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025b. URL https://qwenlm.github.io/blog/qwq-32b/.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu,

Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning, 2025. URL https://arxiv.org/abs/2502.03387.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. Marco-o1: Towards open reasoning models for open-ended solutions,

2024. URL https://arxiv.org/abs/2411.14405.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

A PROOFS AND THEORETICAL DISCUSSIONS

In this section, we provide a rigorous theoretical analysis of the PACS framework. We first derive the exact gradient update of the proposed objective function, demonstrating that it inherently recovers an advantage-weighted policy gradient. Subsequently, we analyze the global optimality conditions to explain why PACS maintains solution diversity and resists entropy collapse, a property achieved through implicit regularization rather than explicit penalty terms.

- A.1 DERIVATION OF THE PACS GRADIENT UPDATE

We first formally establish the gradient dynamics of PACS. We demonstrate that minimizing the supervised cross-entropy loss over the score function ψ(q,o;πθ) results in a policy update mathematically equivalent to the policy gradient. Specifically, the effective advantage function in this update is dynamically composed of the cross-entropy loss term and the reward prediction error.

- Proposition A.1 (Gradient Derivation). The gradient of the PACS objective function LPACS(θ) with respect to the policy parameters θ can be formulated as a policy gradient update weighted by an effective advantage function derived from the prediction error and the cross-entropy loss.

Proof. Consider the PACS objective defined as the expected binary cross-entropy loss:

LPACS(θ) = −Eq∼P(Q),o∼π

θ(·|q) [ℓBCE(R(q,o),σ(ψθ(q,o)))]

where ℓBCE(y,yˆ) = y log yˆ + (1 − y)log(1 − yˆ) is the standard binary cross-entropy function. Without loss of generality (i.e., abstracting from group-level variance reduction baselines), the score

function is structurally parameterized as ψθ(q,o) = β(log πθ(o|q) − log πref(o|q)). To compute the gradient ∇θLPACS(θ), we apply the log-derivative trick to the expectation term and the chain rule to the integrand:

∇θLPACS = −Eq,o∼π

θ(·|q) [∇θ log πθ(o|q) · ℓBCE(q,o;πθ) + ∇θℓBCE(q,o;πθ)]

For the second term, ∇θℓBCE(q,o;πθ), we apply the chain rule with respect to the score ψθ. Noting that the derivative of the sigmoid function is σ′ = σ(1 − σ) and ∂ℓ

∂yˆ = yˆ(1yˆ−−yyˆ), we obtain:

BCE

∂ℓBCE ∂σ

∂σ ∂ψ∇θψθ

∇θℓBCE =

R − σ(ψθ) σ(ψθ)(1 − σ(ψθ)) · σ(ψθ)(1 − σ(ψθ)) · ∇θψθ

=

= (R(q,o) − σ(ψθ))∇θψθ

Given the definition of ψθ, its gradient is ∇θψθ = β∇θ log πθ(o|q) (since πref is fixed). Substituting this back yields:

∇θℓBCE = β(R(q,o) − σ(ψθ))∇θ log πθ(o|q) Combining both terms, we obtain the final gradient expression:

  

  (ℓBCE(q,o;πθ) + β(R(q,o) − σ(ψθ)))

∇θLPACS = −Eq,o∼π

∇θ log πθ(o|q)

θ(·|q)

Effective Advantage APACS

This result confirms that PACS performs a valid policy gradient update. The effective advantage APACS naturally incorporates a “critic” signal (the residual R−σ(ψθ)) which vanishes as the model’s prediction aligns with the ground truth, providing a variance-reduced update signal compared to standard policy gradients with statistical baselines.

| |
|---|

- A.2 IMPLICIT REGULARIZATION AND ROBUSTNESS TO ENTROPY COLLAPSE

- A key empirical finding of this work is that PACS effectively prevents entropy collapse (the degeneration of the policy into a deterministic distribution), a common failure mode in standard RLVR methods like GRPO when explicit KL penalties are omitted. In this section, we provide a theoretical justification for this robustness. We prove that the structural parameterization of PACS imposes an implicit regularization that constrains the optimal policy to a Gibbs distribution, thereby preserving exploration.

Proposition A.2 (Implicit Gibbs Regularization). Assuming sufficient model capacity, the global minimum of the PACS objective function is achieved if and only if the policy πθ follows a Gibbs distribution anchored to the reference policy πref. This stands in contrast to standard unregularized RL objectives, where the global minimum is a deterministic Dirac distribution.

Proof. We first analyze the optimality condition for the PACS objective. Since LPACS is a strictly convex binary cross-entropy loss with respect to the logits ψθ, for any query-response pair (q,o), the loss is minimized when the predicted probability matches the expected target reward:

σ(ψθ∗(q,o)) = Edata[R(q,o)] In the context of RLVR with deterministic ground-truth verification, this implies the model attempts to predict the correctness of o. Inverting the sigmoid function, the optimal score function ψθ∗ must satisfy:

ψθ∗(q,o) = σ−1(E[R(q,o)]) Crucially, in PACS, the score function is not a free parameter but is structurally coupled to the policy via the definition ψθ(q,o) = β(log πθ(o|q) − log πref(o|q)). Substituting this definition into the optimality condition yields:

β log

πθ∗(o|q) πref(o|q)

= σ−1(E[R(q,o)])

Exponentiating both sides and rearranging terms, we derive the form of the optimal policy πθ∗:

πθ∗(o|q) = πref(o|q)exp

σ−1(E[R(q,o)]) β

Comparison with Standard RL: Consider a standard RL objective J = Eo∼π[R(o)] (often optimized by baselines without explicit KL). The global maximum of J occurs at a deterministic Dirac

delta distribution: πstd∗ = δ(o − arg maxo R(o)), which represents total entropy collapse.

In contrast, the optimal policy for PACS derived above is a Gibbs distribution. The term πref(o|q) acts as a base measure, ensuring that the optimized policy retains the support and distributional characteristics of the pre-trained model. Even without an explicit KL loss term in the training objective, the specific parameterization of ψθ enforces this structural constraint. This guarantees that PACS inherently balances reward maximization with adherence to the reference distribution, theoretically explaining the superior diversity and resistance to mode collapse observed in our experiments.

| |
|---|

- B BASELINES

- B.1 PROXIMAL POLICY OPTIMIZATION ALGORITHMS (PPO) The objective function of PPO (Schulman et al., 2017) is formulated as follows:

JPPO(θ) =Eq∼P(Q),o∼π

θold(·|q)

|o|

πθ(ot|q,o<t) πθ

πθ(ot|q,o<t) πθ

1 |o|

At,clip

,1 − ε,1 + ε At

min

(ot|q,o<t)

(ot|q,o<t)

old

old

t=1

where A represents the advantage function estimated using Generalized Advantage Estimation (GAE), and ε denotes the clipping hyperparameter that stabilizes the training process in PPO.

- B.2 GROUP RELATIVE POLICY OPTIMIZATION (GRPO) The objective function of GRPO (Shao et al., 2024) is defined as:

|oi|

G

1 G

1 |oi|

JGRPO(θ) = Eq∼P(Q),{o

i}Gi=1∼πθold(·|q)

t=1

i=1

πθ(oi,t|q,oi,<t) πθ

πθ(oi,t|q,oi,<t) πθ

Ai,t,clip

,1 − ε,1 + ε Ai,t − βDKL[πθ∥πref]

min

(oi,t|q,oi,<t)

(oi,t|q,oi,<t)

old

old

where the advantage function is computed as:

R(q,oi) − mean({R(q,o1),...,R(q,oG)}) std({R(q,o1),...,R(q,oG)})

Ai,t =

The KL divergence term is estimated using:

πref(oi,t|q,oi,<t) πθ(oi,t|q,oi,<t) − 1.

πref(oi,t|q,oi,<t) πθ(oi,t|q,oi,<t) − log

DKL[πθ∥πref] =

Following the approaches in DAPO (Yu et al., 2025) and Dr. GRPO (Liu et al., 2025b), the KL divergence term may be optionally omitted from the original GRPO objective (i.e., setting β = 0), as empirical evidence suggests it may not be essential for performance.

- C HYPERPARAMETERS SETTINGS

To ensure Experimental efficiency and effectiveness, specialized optimized frameworks are used for training and inference phases. This section details the key hyperparameters used in each phase.

- C.1 TEMPLATE

The chat template employed in this study is developed by modifying the official model template as shown below. We incorporate the instruction “Please reason step by step, and put your final answer within \\boxed{}” into the user input to elicit the model’s reasoning capabilities and standardized output formatting of final answers, thereby facilitating automated evaluation.

Chat Template

<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user {input} Please reason step by step, and put your final answer within \\boxed{}.<|im_end|> <|im_start|>assistant

- C.2 TRAINING HYPERPARAMETERS

During the model training phase, the verl (Sheng et al., 2024) framework is employed, which is a powerful toolkit designed for large-scale model training. The relevant hyperparameter settings during training are as follows:

### Hyperparameters Configuration

Train batch size 1,024 Max prompt length 1024 Max response length 8,192 Filter overlong prompts True Mini batch size 256 Micro batch size per GPU 8 Learning rate 1 × 10−6 for actor, 1 × 10−5 for critic in PPO Rollout number 8 Use KL loss False Total training steps 120 for Qwen3-4B, 320 for Qwen3-8B

Table 4: Training hyperparameters.

- C.3 INFERENCE HYPERPARAMETERS

During the model inference and evaluation phase, the high-performance inference serving framework vllm (Kwon et al., 2023) is employed. Through vllm, high-throughout and low-latency text generation is achieved. Consistent inference configurations are adopted across all experiments to ensure fairness and comparability of evaluation results.

Hyperparameters Configuration Enable prefix caching True GPU memory utilization 0.9 Max tokens 8,192 Temperature 0.6 Top-p 0.95 Tensor parallel size 1

Table 5: Inference hyperparameters.

- D DETAILED EXPERIMENTAL RESULTS

This section contains additional experimental results that are omitted from the main text due to space constraints.

- D.1 RESULTS OF QWEN3-4B

Detailed performance metrics for Qwen3-4B on the AMC 23, AIME 2024, AIME 2025, and BeyondAIME benchmarks are presented below.

AMC 23 (pass@k) AIME 2024 (pass@k) Model k = 1 4 8 16 32 64 k = 1 4 8 16 32 64

Base 82.30 91.15 93.33 95.34 97.38 99.20 46.48 61.00 66.15 70.21 73.77 77.21 PPO 86.33 93.93 95.48 96.37 97.07 97.47 52.21 70.64 77.03 81.12 82.94 83.32 GRPO 86.95 94.69 95.98 97.08 98.44 99.70 52.84 70.71 76.31 79.97 82.05 83.13 PACS 90.45 96.01 96.96 97.40 97.50 97.50 55.10 72.91 77.93 80.74 82.25 83.13

- w/o weight 89.18 94.91 95.80 96.54 97.18 97.48 54.56 72.60 78.32 81.68 83.13 83.33

AIME 2025 (pass@k) BeyondAIME (pass@k) Model k = 1 4 8 16 32 64 k = 1 4 8 16 32 64

Base 34.71 49.27 54.57 58.40 62.41 66.81 17.33 24.92 29.31 34.37 39.91 45.29 PPO 40.34 54.87 60.52 66.36 71.58 75.24 21.82 32.97 39.06 45.46 51.28 56.01 GRPO 40.18 54.87 61.23 67.69 73.48 77.34 24.39 37.90 44.31 49.52 53.79 57.58 PACS 45.63 61.00 66.18 70.14 72.91 74.80 27.16 41.66 47.96 53.41 57.91 62.10

- w/o weight 44.90 59.10 64.08 68.11 71.37 73.09 25.89 39.11 45.28 51.18 56.53 61.12

- Table 6: Results of Qwen3-4B trained with PPO, GRPO and PACS. Bold numbers indicate the best performance. Underlined numbers indicate the second best. Due to space constraints, results on MATH 500 are shown in table 7.

D.2 MATH500

MATH500 (pass@k) Model k = 1 4 8 16 32 64

Base 91.00 95.75 96.86 97.68 98.21 98.50 PPO 93.25 97.06 97.86 98.40 98.80 99.13 GRPO 93.91 97.48 98.20 98.74 99.17 99.43 PACS 94.80 97.87 98.47 98.93 99.30 99.49

- w/o weight 94.35 97.70 98.41 98.92 99.23 99.35

- Table 7: Results of Qwen3-4B on MATH500 trained with PPO, GRPO and PACS. Bold numbers indicate the best performance. Underlined numbers indicate the second best.

MATH500 (pass@k) Model k = 1 4 8 16 32 64

Base 89.94 95.20 96.54 97.54 98.27 98.75 PPO 93.78 97.16 97.88 98.38 98.90 99.40 GRPO 95.15 98.08 98.67 99.13 99.50 99.69 PACS 95.09 98.09 98.71 99.16 99.50 99.72

- w/o weight 95.32 98.06 98.66 99.13 99.46 99.65

- Table 8: Results of Qwen3-8B on MATH500 trained with PPO, GRPO and PACS. Bold numbers

We further analyze the performance on the MATH500, as detailed in tables 7 and 8. On the Qwen3-

- 4B scale, PACS demonstrates a consistent and clear advantage over the baselines. Specifically, it achieves a pass@1 score of 94.80%, surpassing GRPO (93.91%) by roughly 0.9 points, and maintains this lead across all k values, reaching 99.49% at pass@64. On the Qwen3-8B scale, performance on this benchmark approaches saturation, with base models already scoring near 90%. Despite this limited room for improvement, PACS remains highly competitive. It significantly outperforms PPO and matches the strong performance of GRPO at pass@1 (95.09% vs. 95.15%), while marginally surpassing it at higher sample counts (e.g., k ≥ 4). This confirms that PACS retains its stability and effectiveness even on relatively simpler tasks where the solution space is well-explored.

- D.3 ABLATION OF DIFFERENT ADVANTAGE ESTIMATORS

To conduct a comparative analysis of how different advantage estimators affect PACS performance, we benchmark our default RLOO method against two alternatives: GRPO (Shao et al., 2024) and Dr. GRPO (Liu et al., 2025c). While RLOO utilizes a leave-one-out mechanism, the advantage functions for GRPO and Dr. GRPO are defined as follows. Dr. GRPO introduces simple yet significant modifications to address the biases in GRPO by removing the std normalization terms.

rˆ(q,oi;πθ) − mean({rˆ(q,o;πθ)}) std({rˆ(q,o;πθ)})

, (9)

ψGRPO(q,oi;πθ) =

ψDr. GRPO(q,oi;πθ) = rˆ(q,oi;πθ) − mean({rˆ(q,o;πθ)}), (10)

MATH500 (pass@k) Model k = 1 4 8 16 32 64 PACS 94.80 97.87 98.47 98.93 99.30 99.49

- - GRPO 94.58 97.78 98.49 99.00 99.27 99.38
- - Dr. GRPO 94.82 97.86 98.48 98.93 99.27 99.47

- Table 9: Results of different advantage estimators of Qwen3-4B on MATH 500. Bold numbers indicate the best performance. Underlined numbers indicate the second best.

AMC23 (pass@k) Model k = 1 4 8 16 32 64 PACS 90.45 96.01 96.96 97.40 97.50 97.50

- - GRPO 89.55 93.72 95.60 96.79 97.59 98.11

- - Dr. GRPO 90.04 95.50 96.49 97.15 97.47 97.50

- Table 10: Results of different advantage estimators of Qwen3-4B on AMC23. Bold numbers indicate the best performance. Underlined numbers indicate the second best.

AIME2024 (pass@k) Model k = 1 4 8 16 32 64 PACS 55.10 72.91 77.93 80.74 82.25 83.13

- - GRPO 56.02 67.74 77.01 81.32 83.02 83.33

- - Dr. GRPO 55.78 71.54 76.28 81.27 82.89 83.32

- Table 11: Results of different advantage estimators of Qwen3-4B on AIME2024. Bold numbers

AIME2025 (pass@k) Model k = 1 4 8 16 32 64 PACS 45.63 61.00 66.18 70.14 72.91 74.80

- - GRPO 44.01 59.66 65.92 70.94 75.04 78.96

- - Dr. GRPO 44.95 60.17 65.90 70.85 75.06 79.03

- Table 12: Results of different advantage estimators of Qwen3-4B on AIME2025. Bold numbers indicate the best performance. Underlined numbers indicate the second best.

BeyondAIME (pass@k) Model k = 1 4 8 16 32 64 PACS 27.16 41.66 47.96 53.41 57.91 62.10

- - GRPO 26.41 39.95 45.73 50.54 54.03 56.30
- - Dr. GRPO 26.67 40.70 46.87 52.33 56.73 60.36

- Table 13: Results of different advantage estimators of Qwen3-4B on BeyondAIME. Bold numbers indicate the best performance. Underlined numbers indicate the second best.

Our findings indicate that PACS with RLOO exhibits superior performance, particularly on challenging benchamarks. As shown in table 13, on the hardest dataset BeyondAIME, PACS achieves a pass@1 of 27.16% and pass@64 of 62.10%, consistently outperforming Dr. GRPO (26.67% / 60.36%) and GRPO (26.41% / 56.30%) across all k values.

While Dr. GRPO demonstrates competitive performance on standard benchmarks like MATH 500 and at higher sample counts on AIME 2025 , PACS maintains a critical edge in the low-sample regime (pass@1) across most tasks (e.g., 90.45% vs. 90.04% on AMC 23 ). We posit that this performance advantage stems from the algorithmic property of RLOO: by contrasting a single sample with the leave-one-out average rather than the group mean including itself, RLOO minimizes the bias in advantage estimation, yielding more precise signals for credit assignment in complex reasoning paths.

- E DIVERSITY SCORE DEFINITION

To rigorously quantify the semantic diversity of the generated reasoning paths, we utilize an embedding-based metric. Let Q = {q1,q2,...,qN} denote the evaluation dataset consisting of N queries. For each query qi, the model generates a set of K candidate outputs, denoted as Oi = {oi,1,oi,2,...,oi,K}.

We employ a pre-trained embedding model E(·) (specifically Qwen3-0.6B-embedding (Zhang et al., 2025) in our experiments) to map each textual output oi,j into a d-dimensional dense vector representation:

vi,j = E(oi,j) ∈ Rd (11)

For a specific query qi, the semantic similarity between two output vectors vi,m and vi,n is measured using cosine similarity:

vi,m · vi,n ∥vi,m∥∥vi,n∥

(12)

sim(vi,m,vi,n) =

The diversity score for query qi, denoted as Div(qi), is defined as the average pairwise cosine distance among the generated responses. To ensure an unbiased estimate, we compute the average over all unique pairs (m,n) where 1 ≤ m < n ≤ K. Since cosine distance is defined as 1 − similarity, the formulation is as follows:

- 1 K

- 2 1≤m<n≤K

##### (13)

sim(vi,m,vi,n)

Div(qi) = 1 −

Average Pairwise Similarity

Finally, the global diversity metric Sdiv for the model is calculated by aggregating the diversity scores across all queries in the dataset:

N

1 N

Div(qi) (14)

Sdiv =

i=1

A higher Sdiv value indicates greater semantic variation among the sampled responses, reflecting a broader exploration of the solution space.

