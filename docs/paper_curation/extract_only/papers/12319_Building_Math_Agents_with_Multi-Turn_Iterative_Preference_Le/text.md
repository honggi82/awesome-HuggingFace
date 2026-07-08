## arXiv:2409.02392v2[cs.LG]27Feb2025

[Figure 1]

# Building Math Agents with Multi-Turn Iterative Preference Learning

###### Wei Xiong1,*, Chengshuai Shi2, Jiaming Shen3, Aviv Rosenberg4, Zhen Qin3, Daniele Calandriello3, Misha Khalman3, Rishabh Joshi3, Bilal Piot3, Mohammad Saleh3, Chi Jin5, Tong Zhang1 and Tianqi Liu3

1University of Illinois Urbana-Champaign, 2University of Virginia, 3Google Deepmind, 4Google Research, 5Princeton University

Recent studies have demonstrated the potential to enhance the mathematical problem-solving capabilities of large language models (LLMs) by integrating external tools such as code interpreters and employing multi-turn Chain-of-Thought (CoT) reasoning. While existing approaches primarily focus on synthetic data generation and Supervised Fine-Tuning (SFT), this paper explores complementary preference learning to further improve model performance. However, existing direct preference learning algorithms are originally designed for the single-turn chat task, and do not fully address the complexities of multiturn reasoning and external tool integration required for tool-integrated mathematical reasoning tasks. To fill in this gap, we introduce a multi-turn online iterative direct preference learning framework tailored to this unique context, which incorporates feedback from code interpreters and optimizes trajectory-level preferences. The effectiveness of our framework is validated through training of various language models using an augmented prompt set derived from GSM8K and MATH datasets. Our results show significant improvements even with only final result checking: for instance, the performance of a supervised fine-tuned Gemma-1.1-it-7B model increased from 77.5% to 83.9% on GSM8K and from 46.1% to 51.2% on MATH. Similarly, a Gemma-2-it-9B model improved from 84.1% to 86.3% on GSM8K and from 51.0% to 54.5% on MATH.

Keywords: RLHF, Agent learning, Mathematical reasoning

### 1. Introduction

Large language models (LLMs) have demonstrated remarkable capacities across a variety of language tasks, showcasing their broad-ranging capabilities in natural language processing. Notable models include ChatGPT (OpenAI, 2023), Claude (Anthropic, 2023), and Gemini (Team et al., 2023). However, despite these advances, even the most advanced closed-source LLMs still struggle with complex reasoning tasks that require multi-rounds of decision making. In particular, for the representative task of mathematical problem solving, LLMs often fail with basic arithmetic and symbolic computations (Cobbe et al., 2021b; Hendrycks et al., 2021; Zheng et al., 2021). To address this issue, recent studies recommend the integration of external tools (e.g., calculators, computational Python libraries and symbolic solvers) to augment the LLMs’ mathematical problem-solving capabilities (Cobbe et al., 2021b; Mishra et al., 2022; Shao et al., 2022; Zhang et al., 2024a). Specifically, by integrating natural language reasoning with the use of these external tools, these enhanced LLMs can receive external messages from tool interactions and reason based on both previously generated tokens and external messages, which significantly improves their performance in mathematical tasks (Gou et al., 2023b; Shao et al., 2024; Toshniwal et al., 2024).

These successes of tool-integrated LLMs lead to a natural research question: how can we better train LLMs to combine tool usage with intrinsic reasoning to tackle complex reasoning tasks? For the mathematical problem solving task, existing works primarily focus on synthetic data generation (by a strong teacher model) and supervised fine-tuning (SFT), as seen in ToRA (Gou et al., 2023b), Meta-

∗ Work done during an internship at Google DeepMind. A preliminary draft without the results of Gemma-2 had been circulated internally in early July. Correspondence to: wx13@illinois.edu, tongzhang@tongzhang-ml.org, tianqiliu@google.com.

MathQA (Yu et al., 2023), MAmmoTH (Yue et al., 2023, 2024), and Open-MathInstruct (Toshniwal et al., 2024). These methods and synthetic datasets have yielded significant improvements in test accuracy on standard benchmarks like MATH (Hendrycks et al., 2021) and GSM8K (Cobbe et al., 2021a).

Building on strong SFT models, Reinforcement Learning from Human Feedback (RLHF) has proven to be a key technique to elicit LLMs’ knowledge during the post-training stage and has become a standard practice in the LLM training pipeline (Bai et al., 2022; Ouyang et al., 2022; Team et al., 2023; Touvron et al., 2023). Broadly speaking, the RLHF learning paradigm, which was originally designed for aligning large language models (LLMs) with human values and preferences (Bai et al.,

- 2022; Ouyang et al., 2022), is distinct from SFT as it learns from relative feedback (Christiano et al., 2017; Ziegler et al., 2019). It has notably enhanced the capabilities of models like ChatGPT, Claude, and Gemini, enabling them to generate responses that are more helpful, harmless, and honest (Bai

- et al., 2022). Inspired by RLHF’s success in general chat applications, in this paper, we explore RLHF for improving LLMs’ mathematical problem-solving abilities when equipped with external tools.

In particular, since deep RL methods (e.g., the proximal policy optimization, PPO algorithm (Schulman et al., 2017)) are often sample inefficient and unstable (Choshen et al., 2019), our goal is to derive direct preference learning algorithms that directly learn from the preference dataset (Azar

- et al., 2023; Rafailov et al., 2023; Zhao et al., 2023). We begin by formulating the learning process as a Markov decision process (MDP), distinct from the contextual bandit approach typically used in RLHF for making general chatbots without external environment interactions (Rafailov et al., 2023; Xiong et al.). Then, we derive the optimality condition of the optimization problem and develop multi-turn variants of direct preference learning algorithms that incorporate external messages, where the primary modification is to mask out irrelevant tokens during training. Furthermore, we extend our approach to its online iterative variants, which recent works demonstrated to be promising (Guo
- et al., 2024b; Xiong et al.).

We evaluate our approach through case studies using augmented training sets from MATH and GSM8K benchmarks, employing various base models such as Gemma (Team et al., 2024), CodeGemma (Team, 2024), and Mistral (Jiang et al., 2023). For instance, the performance of a supervised finetuned Gemma-1.1-it-7B model increased from 77.5% to 83.9% on GSM8K and from 46.1% to 51.2% on MATH. Similarly, a Gemma-2-it-9B model improved from 84.1% to 86.3% on GSM8K and from 51.0% to 54.5% on MATH. These empirical results indicate a significant improvement in performance over standard SFT models, demonstrating the potential of RLHF in complex reasoning task. We also provide a comprehensive recipe for the practical implementation of our online iterative multiturn methods, and make our models, datasets, and code publicly available for further research and development.

##### 1.1. Problem Formulation

We denote prompt as 𝑥 ∈ X and assume that the interactions run for up to 𝐻 rounds. At the first step, a prompt 𝑥 is sampled from some distribution 𝑑0 as the initial state 𝑠1 (We use the terminology “state” instead of “context” because we are concerning about an MDP instead of a contextual bandit here). Then, at each step ℎ ∈ [𝐻],

- • Action: the agent observes the current state 𝑠ℎ, which is the history of the first ℎ−1 interactions with the external environment, and takes an action 𝑎ℎ according to some policy 𝜋ℎ(·|𝑠ℎ) ∈ Δ(A). Typically, the action is in the ReAct manner, which consist of a reasoning step 𝑓ℎ and an execution step 𝑒ℎ (e.g., writing python code) (Yao et al., 2022).
- • Observation: in response to the agent’s action, the environment then returns an observation 𝑜ℎ

based on the history 𝑠ℎ and current action 𝑎ℎ.

Then, we transit to a new state, which is the history up to the step ℎ + 1: 𝑠ℎ+1 = (𝑠ℎ, 𝑎ℎ, 𝑜ℎ) = (𝑥, 𝑎1, 𝑜1, · · · , 𝑎ℎ−1, 𝑜ℎ−1),

and a new step begins. This process repeats for 𝐻 rounds in total and eventually, we collect a trajectory:

𝜏 = (𝑥, 𝑎1, 𝑜1, · · · , 𝑜𝐻−1, 𝑎𝐻).

See Table ?? for an example. The framework presented here is a Markov decision process (MDP), which offers a distinct approach from the contextual bandit model discussed in Xiong et al.. Formally, we define the following MDP.

- Definition 1. An MDP is specified by a tuple (S, A, 𝐻, ℙ∗, 𝑑0), where A is the action space, 𝐻 is the episode length1, ℙ∗ = {ℙ∗ℎ}ℎ𝐻=1 are the state transition kernels, and 𝑑0 denotes the distribution of prompt 𝑠1 = 𝑥. For each ℎ ∈ [𝐻], ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ) is the distribution of the next state given the state-action pair (𝑠ℎ, 𝑎ℎ) at step ℎ. In our setup, a trajectory 𝜏 = (𝑥, 𝑎1, 𝑜1, · · · , 𝑜𝐻−1, 𝑎𝐻) is generated by: 𝑠1 = 𝑥 ∼ 𝑑0 and for all ℎ ∈ [𝐻], 𝑎ℎ ∼ 𝜋ℎ(·|𝑠ℎ), 𝑜ℎ ∼ ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ) where 𝑠ℎ+1 = (𝑠ℎ, 𝑎ℎ, 𝑜ℎ). When there is no ambiguity, the abbreviation 𝑠ℎ+1 ∼ ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ) is also adopted.

The MDP formulation of preference learning was recently studied in Rafailov et al. (2024); Xie et al. (2024a); Zhong et al. (2024) but with a focus on the single-turn chat task and without explicitly considering the external messages. A unique feature of RLHF, as opposed to traditional RL studies, is the relative feedback obtained through comparisons between two trajectories that share the same initial state (prompt). We follow Bai et al. (2022); Ouyang et al. (2022); Ziegler et al. (2019) to assume that the preference signal is generated by the so-called Bradley-Terry model.

- Definition 2 (Bradley-Terry model). We denote 𝜏/𝑥 = 𝑦, where the prompt is excluded from the trajectory. We assume that there exists a utility function of the trajectory 𝑢∗ such that given (𝑥, 𝑦1, 𝑦2), one response 𝑦1 is preferred over another response 𝑦2, denoted as 𝑦1 ≻ 𝑦2, with probability

Prob 𝑦1 ≻ 𝑦2 | 𝑥, 𝑦1, 𝑦2 = 𝜎 𝑢∗(𝑥, 𝑦1) − 𝑢∗(𝑥, 𝑦2) , (1)

where 𝜎 is the sigmoid function 𝜎(𝑧) = 1/(1 + exp(−𝑧)). Also, given (𝑥, 𝑦1, 𝑦2) we denote the sampled preference signal as 𝑧 with 𝑧 = 1 indicating 𝑦1 ≻ 𝑦2 while 𝑧 = 0 indicating 𝑦2 ≻ 𝑦1.

Under this definition, we only assume access to the trajectory-level preference, but not an actionlevel one. This should distinguish our approach from a straightforward extension of the single-turn RLHF (Christiano et al., 2017; Ziegler et al., 2019), which fixes a prompt that may include midtrajectory steps such as (𝑥, 𝑎1, 𝑜1, 𝑎2, 𝑜2) and look into the next single step 𝑎3. However, we remark that the utility function itself, can be defined in a step-wise manner. To further illustrate the notion of the BT model in trajectory-level comparisons, we provide some examples of the utility function here.

- Example 1 (Result Checking in Math). Since the math reasoning datasets GSM8K (Cobbe et al., 2021a) and MATH (Hendrycks et al., 2021) have the gold answer, we can check the final answer to determine the reward. In this case, 𝑢∗(𝑥, 𝑦) = 𝕀(𝑎𝐻 = gold answer).

1In practice, the episode length can vary across the trajectories. We may additionally define that the shorter trajectories that output the final answer are in an absorbing state. We consider the fixed episode length to simplify the subsequent mathematical analysis.

- Example 2 (Outcome-supervised Reward Models (ORMs)). Final result checking is not perfectly reliable because we can encounter false positive solutions that have the correct answer but incorrect reasoning trajectory. Instead, as shown in Cobbe et al. (2021b); Lightman et al. (2023), we can uniformly sample 𝑛 trajectories per prompt and train an ORM to predict whether each solution is correct or not. Then, we can take the ORM prediction at the final token as the utility function.
- Example 3 (Process-supervised Reward Model (PRM) and PRM without Human Annotation.). Lightman et al. (2023) argues that if we can provide step-by-step supervision signal, the utility function is more effective. However, this requires more fine-grained human labels to give rating for each step of the trajectory. Wang et al. (2023a) studies how to automatically construct the process-labeled data for math

problems with gold answers. Specifically, for 𝑠ℎ, 𝑎ℎ, we generate 𝑁 trajectories with final answers [𝑎𝐻𝑗 ]𝑁𝑗=1. We can define the proxy reward value:

We may also use a hard version

𝑁 𝑗=1 𝕀(𝑎𝐻𝑗 = gold answer)

. (2)

𝑟(𝑠ℎ, 𝑎ℎ) :=

𝑁

𝑟(𝑠ℎ, 𝑎ℎ) := 𝕀(There exists a 𝑗0 : 𝑎𝐻𝑗0 = gold answer). (3) Then, we can train the PRM by

###### L𝑃𝑅𝑀(𝜃) = 𝔼𝜏∼D ∑︁𝐻

𝑟(𝑠ℎ, 𝑎ℎ) log𝑟𝜃 + (1 − 𝑟(𝑠ℎ, 𝑎ℎ)) log(1 − 𝑟𝜃) . (4)

ℎ=1

In this case, we can use 𝑢∗(𝑥, 𝑦) = minℎ∈[𝐻] 𝑟𝜃(𝑠ℎ, 𝑎ℎ) (Lightman et al., 2023), where 𝑟𝜃 is the constructed step-wise reward function.

Notations. To improve the readability of this work, we provide a notable table in Table 6.

##### 1.2. Related Work

LLMs for Mathematical Problem Solving. A line of works proposes to prompt LLMs to solve the complex reasoning task in a step-by-step manner, known as the Chain-of-Thought (CoT) prompting (Tong et al., 2024; Wei et al., 2022; Zhou et al., 2022; Zhu et al., 2022), which has been a standard practice in reasoning task. However, LLMs often struggle with basic arithmetic and symbolic manipulations when relying solely on internal knowledge and natural language reasoning, as measured by standard benchmarks (Cobbe et al., 2021a; Hendrycks et al., 2021). To overcome these limitations, several studies have explored the use of external tools to enhance the LLMs’ problem-solving abilities. This includes calculators (Cobbe et al., 2021b; Shao et al., 2022), symbolic solvers (Zhang, 2023), and code interpreters (Mishra et al., 2022; OpenAI, 2023). A particularly effective approach is the Program-based method (PoT), which performs CoT reasoning by writing code and using the output of the written code as the final answer (Chen et al., 2022; Gao et al., 2023a). This method significantly outperforms traditional CoT-based techniques in mathematical problem solving. However, PoT also faces challenges in planning and error handling, where natural language reasoning is more suitable (Gou et al., 2023a). In view of this, tool-integrated reasoning is proposed to combine the naturallanguage-based intrinsic reasoning with the external tools (Gou et al., 2023b) and has achieved great progresses in recent studies (Gou et al., 2023b; Shao et al., 2024; Toshniwal et al., 2024; Yu et al.,

- 2023; Yue et al., 2023). While these efforts have primarily focused on synthetic data generation for tool-integrated reasoning, our work aims to further boost the performance of tool-integrated LLMs by RLHF.

RLHF and RLHF Algorithms. The predominant approach in RLHF is the deep RL method, Proximal Policy Optimization Algorithms (PPO) (Schulman et al., 2017), which leads to the great successes in Chat-GPT (OpenAI, 2023), Gemini (Team et al., 2023), and Claude (Anthropic, 2023). However, applying PPO requires extensive efforts and resources (Choshen et al., 2019; Engstrom et al., 2020), often beyond the scope of open-source capabilities. In view of this, alternative approaches have been developed. The rejection sampling fine-tuning was first proposed with the name RAFT (reward ranked fine-tuning) in RLHF (Dong et al., 2023) and was later extended to machine translation (Gulcehre

- et al., 2023) and mathematical problem solving (Yuan et al., 2023a). Its theoretical advantage was explored in Gui et al. (2024). Subsequently, another long line of works proposes direct preference learning algorithms, including Slic (Zhao et al., 2023), DPO (Rafailov et al., 2023), IPO (Azar et al.,

- 2023), KTO (Ethayarajh et al., 2024), and GPO (Tang et al., 2024). These algorithms bypass the reward modeling step and optimize carefully designed loss objectives directly on the preference dataset, hence the name direct preference learning. There are also some works focusing on more general preference structure Munos et al. (2023); Rosset et al. (2024); Swamy et al. (2024); Ye et al.

(2024) beyond the reward-based framework or post-processing of the model (Lin et al., 2023; Zheng

- et al., 2024).

The newly proposed direct preference learning algorithms have largely advanced the RLHF area, particularly the post-training of open-source models, with the Zephyr project as a notable example (Tunstall et al., 2023). After this, a long line of work (e.g., Guo et al., 2024b; Liu et al., 2023b, 2024a,b; Meng et al., 2024; Tajwar et al., 2024; Xie et al., 2024a; Xiong et al.; Xu et al., 2023; Zhang et al., 2024b) demonstrates the effectiveness of on-policy sampling (the samples are generated by the policy to be trained) and online exploration in enhancing direct preference learning. In particular, the online iterative DPO (Hoang Tran, 2024; Xiong et al.; Xu et al., 2023) and its variants (e.g., Cen et al., 2024; Chen et al., 2024b; Rosset et al., 2024; Zhang et al., 2024c) have made state-of-the-art open-source models (Dong et al., 2024), or even the industry models (qwe, 2024; Meta, 2024). Despite these advancements, most algorithms are proposed and designed for single-turn interactions and chat. The scenarios beyond single-turn chat remain largely unexplored in the existing literature. One exception is the very recent work by Shani et al. (2024), which studies multi-turn chat task under general preferences. In contrast, in this paper, we aim to explore the use of RLHF in multi-turn tasks that incorporate interactions with external tools. Meanwhile, they derive a mirror-descent-based policy optimization algorithm, which is also different from ours.

RLHF for Math Problem Solving. Algorithms traditionally used in general chatbot applications have been adapted to enhance the reasoning capabilities of LLMs in mathematical contexts. For instance, RAFT (Reward-rAnked Fine-Tuning) (Dong et al., 2023; Touvron et al., 2023; Yuan et al., 2023b) is extensively employed for synthetic data generation, whether through on-policy (self-improving) (Yuan et al., 2023a) or off-policy (knowledge distillation) methods (Gou et al., 2023b; Singh et al., 2023; Tong et al., 2024; Toshniwal et al., 2024; Yu et al., 2023). The reward signal in these scenarios is typically derived from either final result checking or Outcome-supervised Reward Models (ORMs) (Uesato et al., 2022; Zelikman et al., 2022). A novel approach by Lightman et al. (2023) introduces Process-supervised Reward Models (PRMs), which provide feedback at each step of the Chain-ofThought, demonstrating significant improvements over ORMs when combined with rejection sampling (Lightman et al., 2023; Wang et al., 2023a).

In addition to the RAFT, the GRPO algorithm proposed in Shao et al. (2024) studies multi-turn math problem solving but focuses on the CoT format without external inputs and the resulting model achieves the state-of-the-art performance in its class. The GRPO is a variant of Reinforce (Williams, 1992) thus falling into the scope of deep RL methods.

Further advancements include adapting direct preference learning algorithms to mathematical problem solving. For instance, Jiao et al. (2024); Yuan et al. (2024) have applied the original DPO or KTO by taking the trajectory completion as a “meta” action. Pang et al. (2024); Xie et al. (2024b) further adapt the online iterative DPO originally designed for chat (Hoang Tran, 2024; Xiong et al.; Xu et al., 2023) and achieve better performance for CoT reasoning. Inspired by the success of PRMs, recent studies have explored generating proxy step-wise labels for the intermediate steps of the reasoning trajectories. For instance, Chen et al. (2024a); Lai et al. (2024); Xie et al. (2024b) leverage Monte Carlo Tree Search (MCTS) and use the estimated Q value to generate the proxy labels for the intermediate steps. Lai et al. (2024) proposes to use AI feedback like GPT-4 (Lai et al., 2024) to find the first error step in the trajectory. Meanwhile, Lu et al. (2024) identifies a trajectory with the correct final answer and no errors as preferable, and prompts the SFT model with a high temperature, starting from some intermediate step to collect a rejected trajectory with errors (Pi et al., 2024). Finally, a very recent study by Chen et al. (2024a) proposes to use MCTS with a backward iteration from the final leaf node to compute the proxy unregularized value of each node. Preference pairs are then extracted from the tree by fixing the prefix and comparing the next single reasoning step. Then, they run the original DPO on these intermediate actions with the proxy labels from MCTS. To summarize, these works present different ways of preference data collection and apply the original DPO algorithm (with some additional marginal loss and regularization adapted from the literature), thereby differing from our work in both algorithmic concepts and application scope. In contrast, we study preference learning in the context of trajectory-level comparison, where we derive the optimality condition and introduce a multi-turn DPO within an online iterative framework, specifically for tool-integrated mathematical problem solving. However, we remark that while we focus on the trajectory-level comparison, the preference signal itself can be generated in a step-by-step supervision (see Section 1.1 for the detailed examples). When preference signals for partial trajectories with shared prefixes are available, our method can also adapt to learn these step-level signals (see the optimality condition in (13)). In particular, the algorithmic design presented in this paper can be readily combined with the MCTS-based data collection strategy outlined in recent literature, which we leave for future work.

### 2. Algorithms Development

We develop the main algorithms of this paper in this section. We proceed to handle the general MDP formulation presented in Section 1.1, which subsumes the tool-integrated mathematical reasoning problem as a special example. Therefore, the algorithms may also be applied to more general scenarios with external messages..

##### 2.1. Planning with a Fixed Model: Optimality Condition

Following Rafailov et al. (2023), we first establish the connection between any model M = (S, A, 𝐻, ℙ, 𝑑0, 𝑢) and its associated optimal policy. In particular, we are interested in the following KL-regularized

planning problem with respect to a reference policy 𝜋ref: argmax

###### ∑︁𝐻

𝐷KL 𝜋ℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) . (5)

𝐽(𝜋;M, 𝜋ref) = 𝔼𝑥∼𝑑0𝔼𝑎

ℎ∼𝜋ℎ(·|𝑠ℎ),𝑜ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ) 𝑢(𝑥, 𝑦) − 𝜂

𝜋

ℎ=1

In the single-turn case (i.e., 𝐻 = 1 and without transitions ℙ), Azar et al. (2023); Rafailov et al. (2023) show that the optimal solution with respect to a utility function 𝑢 admits a closed-form solution, which is the Gibbs distribution (see Lemma 3):

𝑢(𝑥, 𝑎1) 𝜂

𝜋M(𝑎1|𝑥) ∝ 𝜋ref(𝑎1|𝑥) exp

.

Moving from the single-step to multi-turn scenario, we first show that we are still concerning about the Gibbs distribution, but in a dynamic programming manner. We summarize the results into the following proposition.

- Proposition 1. We can recursively define the following optimal value functions and optimal policies for a KL-regularized MDP with horizon 𝐻 and external observation 𝑜ℎ. For Q value, we have

𝑢(𝑠𝐻, 𝑎𝐻), if ℎ = 𝐻, 𝔼𝑜

𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) =

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)[𝑉M,ℎ+1(𝑠ℎ+1)], if ℎ ≤ 𝐻 − 1.

Also, for all ℎ ∈ [𝐻], we have:

𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) 𝜂

𝑉M,ℎ(𝑠ℎ) = 𝜂 log𝔼𝑎ℎ∼𝜋ref,ℎ(·|𝑠ℎ) exp

,

(6)

=:𝑍ℎ(𝑠ℎ)

𝜋ref,ℎ(𝑎ℎ | 𝑠ℎ) 𝑍ℎ(𝑠ℎ)

𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) 𝜂

. (7)

· exp

𝜋M,ℎ(𝑎ℎ | 𝑠ℎ) =

We have a few interesting observations that may be of independent interests.

- 1. The optimal value function is characterized by the expectation with respect to the initial reference policy due to the additional KL constraint.
- 2. For a fixed step ℎ and state-action pair (𝑠ℎ, 𝑎ℎ), we can treat the future as a bandit (with only one step), then, we have 𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) = 𝔼𝑧𝑢(𝑠ℎ, 𝑎ℎ, 𝑧), where 𝑧 is a completion staring from (𝑠ℎ, 𝑎ℎ). One can use the Monte-Carlo estimation to estimate this value by multiple roll-outs. We notice that the non-regularized version of this process, is commonly referred to as the process-supervised reward (PRM) in the literature (Wang et al., 2023a). In other words, the PRM constructed in Wang et al. (2023a) is essentially a Q learning process.

The results are essentially from the study of entropy-regularized MDPs (Williams and Peng, 1991; Ziebart, 2010).

To illustrate the idea, we first consider the simplest case of 𝐻 = 2, where the model is allowed to call the tool only once. Then, our goal is to maximize the following target:

𝔼𝑥∼𝑑0 𝔼𝑎1∼𝜋1(·|𝑥) 𝔼𝑜1∼ℙ1(·|𝑥,𝑎1) 𝔼𝑎2∼𝜋2(·|𝑠2)𝑢(𝑠2, 𝑎2) − 𝜂𝐷KL 𝜋2(·|𝑠2), 𝜋ref,2(·|𝑠2) Inner Loop

− 𝜂𝐷KL 𝜋1(·|𝑠1), 𝜋ref,1(·|𝑠1) .

The idea is to take a backward iteration from ℎ = 𝐻 = 2 to ℎ = 1. Specifically, when we fix 𝑠2 and consider the inner loop, we can leverage Lemma 3 to solve

𝑢(𝑠2, ·) 𝜂

𝜋M,2(·|𝑠2) = argmax

𝔼𝑎2∼𝜋2(·|𝑠2) 𝑢(𝑠2, 𝑎2) − 𝜂 · 𝐷KL 𝜋2(·|𝑠2), 𝜋ref,2(·|𝑠2) ∝ 𝜋ref,2(·|𝑠2) · exp

.

𝜋2

Then, we can define the value of the inner loop associated with 𝜋M,2 as 𝑉M,2(𝑠2) := 𝔼𝑎2∼𝜋M

,2(·|𝑠2) 𝑢(𝑠2, 𝑎2) − 𝜂𝐷KL 𝜋M,2(·|𝑠2), 𝜋ref,2(·|𝑠2) 𝑄M,1(𝑠1, 𝑎1) := 𝔼𝑜1∼ℙ1(·|𝑠1,𝑎1) 𝑉M,2(𝑠2) .

Then, for step ℎ = 𝐻 − 1 = 1, we are concerning the following KL-regularized optimization problem:

𝑄M,1(𝑠1, ·) 𝜂

𝜋M,1(·|𝑠1) = argmax

𝔼𝑎1∼𝜋1(·|𝑥) 𝑄M,1(𝑠1, 𝑎1) − 𝜂𝐷KL 𝜋1(·|𝑠1), 𝜋ref,1(·|𝑠1) ∝ 𝜋ref,1(·|𝑠1) · exp

𝜋1

.

By construction, it can be observed that {𝜋M,ℎ}2ℎ=1 is optimal as it maximizes the KL-regularized target.

For general 𝐻-step MDP, we can repeat the above process for 𝐻 times starting with 𝑉M,𝐻+1 = 0 where we recursively define

𝑢(𝑠𝐻, 𝑎𝐻), if ℎ = 𝐻, 𝔼𝑜

𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) =

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)[𝑉M,ℎ+1(𝑠ℎ+1)], if ℎ ≤ 𝐻 − 1,

Here the optimal policy and the 𝑉-values are given by

1 𝑍ℎ(𝑠ℎ)

𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) 𝜂

(Gibbs distribution of 𝑄M,ℎ) 𝑉M,ℎ(𝑠ℎ) := 𝔼𝑎

𝜋ref,ℎ(𝑎ℎ|𝑠ℎ) · exp

𝜋M,ℎ(𝑎ℎ|𝑠ℎ) :=

ℎ∼𝜋M,ℎ(·|𝑠ℎ) 𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) − 𝜂 · 𝐷KL 𝜋M,ℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ)

𝑄M,ℎ(𝑠ℎ, 𝑎′

ℎ)

= 𝜂 log𝔼𝜋ref

ℎ|𝑠ℎ) exp

,

,ℎ(𝑎′

𝜂

(8)

(9)

where 𝑍ℎ(𝑠ℎ) = 𝑎

ℎ∈A 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ) ·exp 𝑄M,ℎ(𝑠ℎ,𝑎ℎ)

𝜂 is the normalization constant. The second equality

in the definition of the 𝑉-value is from Lemma 3. Then, by definition, [𝜋M,ℎ]ℎ𝐻=1 is the optimal policy. Essentially, we solve 𝐻 Gibbs distributions in terms of the 𝑄-values2.

##### 2.2. Planning with a Fixed Model: Practical Algorithm

While (9) can be approximately solved with standard deep RL methods, here we are interested in the implementation in a direct preference learning manner like Slic (Zhao et al., 2023), DPO (Rafailov et al., 2023) or IPO (Azar et al., 2023). The existing attempts (e.g., Yuan et al., 2024) take the completion 𝑦 as a “meta action” and plug it into the single-step DPO loss. In other words, they treat the external messages as the regular texts generated by the model itself. Another natural idea is to plug the probability of the trajectory into the single-step DPO loss. To be specific, for a pair (𝑥, 𝜏𝑤, 𝜏𝑙), where 𝜏𝑤 refers to the preferred (i.e., winning) trajectory, we have

Prob𝜋(𝜏𝑙|𝑥) Prob𝜋ref (𝜏𝑙|𝑥)

Prob𝜋(𝜏𝑤|𝑥) Prob𝜋ref (𝜏𝑤|𝑥)

− log𝜎 𝜂 log

− log

𝐻

𝐻

𝜋ℎ(𝑎ℎ𝑙 |𝑠ℎ𝑙 ) ℙℎ(𝑜ℎ𝑙 |𝑠ℎ𝑙 , 𝑎ℎ𝑙 ) 𝜋ref,ℎ(𝑎ℎ𝑙 |𝑠ℎ𝑙 ) ℙℎ(𝑜ℎ𝑙 |𝑠ℎ𝑙 , 𝑎ℎ𝑙 )

𝜋ℎ(𝑎𝑤ℎ |𝑠ℎ𝑤) ℙℎ(𝑜𝑤ℎ |𝑠ℎ𝑤, 𝑎𝑤ℎ ) 𝜋ref,ℎ(𝑎𝑤ℎ |𝑠ℎ𝑤) ℙℎ(𝑜𝑤ℎ |𝑠ℎ𝑤, 𝑎𝑤ℎ )

= −log𝜎 𝜂 log

− log

(10)

ℎ=1

ℎ=1

∑︁𝐻

𝜋ℎ(𝑎ℎ𝑙 |𝑠ℎ𝑙 ) 𝜋ref,ℎ(𝑎ℎ𝑙 |𝑠ℎ𝑙 )

𝜋ℎ(𝑎𝑤ℎ |𝑠ℎ𝑤) 𝜋ref,ℎ(𝑎𝑤ℎ |𝑠ℎ𝑤)

= −log𝜎 𝜂

log

− log

.

ℎ=1

Unfortunately, the resulting algorithm does not always lead to the optimal policy as we explain next. In particular, we can solve the 𝑄-values as

𝑄M,ℎ(𝑠ℎ, 𝑎′

ℎ)

𝜋M,ℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

𝑄M,ℎ(𝑠ℎ, 𝑎ℎ) = log

+ 𝜂 log𝔼𝜋ref

ℎ|𝑠ℎ) exp

,ℎ(𝑎′

𝜂

(11)

𝜋M,ℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

= log

+ 𝑉M,ℎ(𝑠ℎ),

2The definitions of 𝑄-values are different from that of Ziebart (2010) so that the optimal policy can be interpreted as the Gibbs distribution of 𝑄-values.

where two equalities uses the definition of the optimal policy 𝜋M,ℎ and 𝑉-value 𝑉M,ℎ in (9), respectively. Furthermore, by the definition of 𝑄-values 𝑄M,ℎ in (8), we have

𝜋M,ℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)𝑉M,ℎ+1(𝑠ℎ+1) = log

+ 𝑉M,ℎ(𝑠ℎ), if ℎ ≤ 𝐻 − 1

𝔼𝑜

(12)

𝜋M,𝐻(𝑎𝐻|𝑠𝐻) 𝜋ref,𝐻(𝑎𝐻|𝑠𝐻)

𝑢(𝑠𝐻, 𝑎𝐻) = log

+ 𝑉M,𝐻(𝑠𝐻).

Summing over ℎ ∈ [𝐻], we have

∑︁𝐻

###### ∑︁𝐻

𝜋M,ℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

log

𝑢(𝑠𝐻, 𝑎𝐻) = 𝜂

𝑉M,ℎ(𝑠ℎ) − 𝔼𝑜

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)𝑉M,ℎ+1(𝑠ℎ+1)

+

ℎ=1

ℎ=1

∑︁𝐻

𝐻∑︁−1

(13)

𝜋M,ℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

log

= 𝜂

𝑉M,ℎ+1(𝑠ℎ+1) − 𝔼𝑜

###### +𝑉M,1(𝑠1)

+

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)𝑉M,ℎ+1(𝑠ℎ+1)

.

ℎ=1

ℎ=1

term (𝐵)

term (𝐴)

term (𝐶)

Here, term (𝐴) is the counterpart of 𝜂 log 𝜋𝜋(𝑎1|𝑠1)

ref(𝑎1|𝑠1) in the single-step DPO derivation and term (𝐵)

will be cancelled if we consider the reward difference of two trajectories with the same prompt 𝑠1 = 𝑥. Unfortunately, in practice, term (𝐶) is typically not feasible to directly compute. Especially, some simple math leads to that with probability at least 0.9,

###### 𝐻∑︁−1

1/2

𝜎ℎ2

|𝐶| ≤ 4

,

ℎ=1

where 𝜎ℎ2 is the conditional variance of 𝑉M,ℎ+1(𝑠ℎ+1) − 𝔼𝑜

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)𝑉M,ℎ+1(𝑠ℎ+1). Therefore, the bias term (𝐶) is related to the randomness of the external environment.

For most cases of tool-integrated LLMs for mathematical reasoning, i.e., the focus of this work, luckily the code execution result is determined by the history (the codes written by the LLMs). In other words, given the history 𝑠ℎ, the external observation is deterministic, which leads to term (𝐶) = 0. Thus, with a dataset D consisting of (𝑥, 𝜏𝑤, 𝜏𝑙), the following multi-turn DPO (M-DPO) loss can be adopted:

###### LM-DPO(𝜃) = − ∑︁

###### ∑︁𝐻

𝜋𝜃,ℎ(𝑎ℎ𝑙 |𝑠ℎ𝑙 ) 𝜋ref,ℎ(𝑎ℎ𝑙 |𝑠ℎ𝑙 )

𝜋𝜃,ℎ(𝑎𝑤ℎ |𝑠ℎ𝑤) 𝜋ref,ℎ(𝑎𝑤ℎ |𝑠ℎ𝑤)

, (14)

log𝜎 𝜂

log

− log

ℎ=1

(𝑥,𝜏𝑤,𝜏𝑙)∈D

We emphasize again that although the loss presented in (14) is identical to the one in (10), a rigorous derivation procedure (rather than a direct plug-in) is provided. To the best of our knowledge, (14) is new in the context of multi-turn reasoning task with external messages. In particular, it is noted that such a M-DPO loss is only valid upon deterministic transitions, i.e., term (𝐶) = 0.

Moreover, with (13) implying that with term (𝐶) = 0, the implicit reward is given by 𝐴 = 𝜂 ℎ 𝐻=1 log 𝜋

∗

ℎ(𝑎ℎ|𝑠ℎ)

𝜋ref,ℎ(𝑎ℎ|𝑠ℎ), a multi-turn version of KTO (Ethayarajh et al., 2024), denoted as M-KTO, can also be naturally derived:

LM-KTO(𝜃) = 𝔼𝑥,𝑦∼D 𝜆𝑦 − 𝑣(𝑥, 𝑦) , (15) where

###### ∑︁𝐻

𝜋𝑢,ℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

log

𝑢𝜃(𝑥, 𝑦) = 𝜂

,

ℎ=1

∑︁𝐻

𝑧0 = 𝔼𝑥′∼D,𝜏′∼𝜋𝜃(·|𝑥′)

𝐷KL 𝜋𝜃(·|𝑠ℎ), 𝜋ref(·|𝑠ℎ) ,

ℎ=1

and

𝜆+𝜎 𝜂(𝑢𝜃(𝑥, 𝑦) − 𝑧0) if 𝑦 ∼ 𝑦𝑑𝑒𝑠𝑖𝑟𝑎𝑏𝑙𝑒|𝑥 𝜆−𝜎 𝜂(𝑧0 − 𝑢𝜃(𝑥, 𝑦)) if 𝑦 ∼ 𝑦𝑢𝑛𝑑𝑒𝑠𝑖𝑟𝑎𝑏𝑙𝑒|𝑥

𝑣(𝑥, 𝑦) =

.

Here 𝜆+ and 𝜆− are two hyper-parameters. We notice that Mitra et al. (2024) developed an online iterative version of KTO for the CoT format reasoning task. Here we extend it to build the toolintegrated reasoning agent.

The above discussions, in particular, M-DPO and M-KTO losses provided in (14) and (15), are focused on deterministic observations due to the deterministic nature of tool-integrated LLMs for mathematical reasoning. In contrast, some other applications may encounter stochastic observations, e.g., multi-turn chats with the external message provided by a human or another LLM (Shani et al.,

- 2024). In these scenarios, (14) is biased and cannot lead to the optimal policy since term (𝐶) ≠ 0. Instead, one should first construct a value network based on the Bellman equations provided in (8) and (9), similar to the approach in Richemond et al. (2024). Subsequently, term (𝐶) can be estimated using Monte-Carlo methods and serve as an adaptive margin in the preference training. Particularly, the distinctions between direct preference learning algorithms and classical deep RL methods become less clear. The exploration of this more complex algorithm and its application to general multi-turn learning scenarios is left for future research.

We note that the MDP formulation above and related discussions have been previously derived by Rafailov et al. (2024); Xie et al. (2024a); Zhong et al. (2024) in the context of either token-wise MDP or more general MDP with deterministic transition but their focuses are all on the single-turn chat tasks. Although the mathematical formulations appear similar, our primary focus lies on tool-integrated reasoning tasks that incorporate additional external messages {𝑜ℎ}ℎ𝐻=−11.

##### 2.3. Learning with Online Iterative Training

In the literature of direct preference learning, a long line of work shows that the online single-turn RLHF significantly outperforms their offline counterpart, both in the literature of direct preference learning (Dong et al., 2024; Guo et al., 2024b; Rosset et al., 2024; Tajwar et al., 2024; Xiong et al.; Ye et al., 2024) and DRL-based approach or rejection sampling fine-tuning (Bai et al., 2022; Ouyang et al., 2022; Touvron et al., 2023). Motivated by these successes, we propose to further incorporate online interactive learning to the multi-turn RLHF studied in this work. In the following, we illustrate the proposed ideas from mainly two aspects: two learning objectives and one unified algorithmic framework.

Learning objective. We consider two different learning objectives. The first one is the KL-regularized target:

###### ∑︁𝐻

𝐷KL 𝜋(·|𝑠ℎ), 𝜋0(·|𝑠ℎ) , (16)

max

ℎ∼𝜋(·|𝑠ℎ),𝑜ℎ∼ℙ∗ℎ(·|𝑠ℎ,𝑎ℎ) 𝑢∗(𝑥, 𝑦) − 𝜂

𝔼𝑥∼𝑑0𝔼𝑎

𝜋

ℎ=1

i.e., max𝜋 𝐽(𝜋;M∗, 𝜋0) where M∗ = (S, A, 𝐻, ℙ∗, 𝑑0, 𝑢∗) is the groundtruth environment and 𝜋0 is the initial policy (e.g., from SFT) that RLHF starts from. This target is widely adopted in practice (Bai

- et al., 2022; Christiano et al., 2017; Dong et al., 2024; Ouyang et al., 2022; Rafailov et al., 2023)

and requires us to search for the optimal policy only at a fixed KL ball centered at the SFT policy 𝜋0 (Xie et al., 2024a; Xiong et al.; Ye et al., 2024).

In contrast, the second one is the non-regularized target, i.e., directly optimizing the reward: max

ℎ∼𝜋(·|𝑠ℎ),𝑜ℎ∼ℙ∗ℎ(·|𝑠ℎ,𝑎ℎ) 𝑢∗(𝑥, 𝑦) . (17)

𝔼𝑥∼𝑑0𝔼𝑎

𝜋

This target is the standard one in canonical RL studies (Sutton and Barto, 2018). One motivation for this target is that in the reasoning task, the reward function is more interpretable (e.g. final result checking) compared to the chat task.

Additionally, we note that a stronger KL regularization in the target (16) is known to be beneficial for mitigating over-fitting issue and forgetting on the out-of-domain tasks (Coste et al., 2023; Gao

- et al., 2023b; Lin et al., 2023). On the other hand, (17) allows the model to move more far away, thus achieving a better in-domain performance. Thus, from one perspective, the choice between the above two targets can be viewed as a tradeoff between out-of-domain and in-domain performances. This intuition is also verified by later experiments, where optimizing the second target in (17) leads to better performance on in-domain test sets. In the rest of this section, we discuss two learning objectives to fully develop the multi-turn preference learning framework. We also conduct an ablation study on these objectives in the experimental section.

Algorithmic framework. We present a general online iterative algorithmic framework in Algorithm 1. Specifically, starting from 𝜋0, at each iteration, we first collect a pair of trajectories by the current policy pair, where the preference signal is also revealed according to Definition 1. Then, we update our policy pair given the data collected so far and the next iteration begins. We now discuss some features of the framework as follows.

Policy choice for exploration-exploitation trade-off. We update our behavior policies in a nonsymmetric way. The first agent, which aims to extract the historical information we have gathered so far, planning with respect to the empirically best model on the historical dataset D to get 𝜋1𝑡 , where the planning algorithms have been discussed in Section 2.2, e.g., optimizing the M-DPO or M-KTO loss in (14) or (15). However, it is widely recognized in RL studies (Auer et al., 2002; Sutton and Barto, 2018) that simply exploiting the historical data via following the empirically best model is not sufficient to obtain a good final policy, while it is also required to explore the environment so that new information can be collected to facilitate subsequent learning, i.e., the exploration-exploitation tradeoff. While the main agent targeting exploitation, we design the second agent, in contrast, to strategically incorporate the uncertainty of the future relative to 𝜋1𝑡 given the historical information we collect so far into its policy choice. We call the policy of the second agent 𝜋2𝑡 as an exploration policy because it serves to explore the underlying environment and facilitate the first agent’s learning. In practice, this principle of exploration is generally interpreted as maximizing the difference between the two behavior policies or increasing the diversity of the collected data. We summarize some popular heuristic exploration policy adopted in the online iterative RLHF practice:

- • Mixture sampling: in the Claude project (Anthropic, 2023), the authors choose to use the checkpoints from different training steps to collect data;
- • Inference parameters tuning: in the LLaMA project (Touvron et al., 2023), the authors carefully tune the sampling temperature to balance data diversity and data quality;
- • West-of-n sampling: Dong et al. (2024); Hoang Tran (2024); Pace et al. (2024); Xu et al. (2023) samples n responses per prompt and extract the best one and the worst one (based on some ranking criteria) to construct a preference pair.

We will explore the mixture sampling in the experimental section and also provide a theoretical justification in the next subsection.

Reference model choice for controlling regularization level. Despite two different learning targets are discussed in (16) and (17) seperately, we note that one general algorithmic framework can be adopted with the reference model choice taking as a hyper-parameter to control the regularization level and account for the two targets:

- • KL-regularized target in (16): if we fix the reference model as the initial policy, i.e., 𝜋𝑡,ref = 𝜋0,∀𝑡 ∈ [𝑇], we always search the optimal policy within the KL ball centered at 𝜋0, and thus optimize the KL-regularized target.
- • Non-regularized target in (17): in contrast, inspired by the mirror descent (Nemirovskij and Yudin, 1983), if we update the reference policy every iteration to be the policy learned in the

last iteration, i.e., 𝜋𝑡,ref = 𝜋1𝑡−1,∀𝑡 ∈ [𝑇], the cumulative update can make the model to move away from the original 𝜋0 (while a constraint is made on the per-iteration update magnitude) and we thus optimize the non-regularized target.

A graphical illustration is provided in Figure 1 to facilitate the understanding.

[Figure 2]

- Figure 1 | Illustration of the difference between the two learning objectives. The left-hand figure corresponds to the KL-regularized target where we do not update the reference model. The right-hand figure corresponds to the non-regularized target where we always update the reference model as the last-iteration one.

##### 2.4. Theoretical Result

In this section, we show that the multi-turn RLHF problem can be solved in a statistically efficient manner under standard assumptions in learning theory literature. In particular, for generality, we target the most challenging scenario with stochastic and unknown transitions, while as aforementioned, multi-turn mathematical reasoning with external tools falls into an relatively easier regime with deterministic transitions. Here we mostly studies the KL-regularized target due to the lack of theoretical research on it. The other target of optimizing the rewards has been theoretically studied in Wang et al. (2023b) while the techniques of analyzing mirror-descent-style algorithm and corresponding guarantees have also be developed in Cai et al. (2020), which can be migrated to considering preference feedbacks. Also, to ease the presentation, we consider the scenario with batch size 𝑚 = 1, while the results can be easily generalized to large batches.

First, to measure the online learning process, we define the optimal policy as 𝜋∗ := argmax

𝐽(𝜋) := 𝐽(𝜋;M∗, 𝜋0), (18)

𝜋

and introduce the standard notion of regret as Reg(𝑇) := ∑︁

𝑡∈[𝑇]

𝐽(𝜋∗) − 𝐽(𝜋1𝑡 ), (19)

Algorithm 1 Online Iterative M-GSHF

- 1: Input: KL coefficient 𝜂 > 0, horizon 𝑇 > 0, initial policy 𝜋0, batch size 𝑚 > 0.
- 2: Initialize D ← ∅ and 𝜋11 = 𝜋21 = 𝜋1,ref ← 𝜋0.
- 3: for 𝑡 = 1, 2, · · · ,𝑇 do
- 4: Sample 𝑚 pairs (𝑥, 𝜏1, 𝜏2, 𝑧) as D𝑡 by 𝑥 ∼ 𝑑0, 𝜏1 ∼ 𝜋1𝑡 , 𝜏2 ∼ 𝜋2𝑡 , receive the 𝑚 preference signals 𝑧 following the Bradley-Terry model from Definition 1 and update the preference dataset D ← D ∪ D𝑡.
- 5: ⊲ Extract the empirically optimal policy from historical data
- 6: Practical: Perform the planning algorithms on D to get 𝜋1𝑡 (e.g., using the M-DPO loss in (14) or the M-KTO loss in (15))
- 7: Theoretical: Perform MLE on D to obtain model estimation Mˆ𝑡 = (𝑢ˆ𝑡, ℙˆ𝑡) as in (20) and (21); call Oracle 3 with Mˆ𝑡, 𝜂, 𝜋𝑡,ref to get 𝜋1𝑡
- 8: ⊲ Select the exploration policy to facilitate learning
- 9: Practical: Given 𝜋1𝑡 , select 𝜋2𝑡 as an exploration policy using heuristic methods (such as mixture sampling, inference parameters tuning and west-of-n sampling listed in Section 2.3)
- 10: Theoretical: Given 𝜋1𝑡 , choose 𝜋2𝑡 as an exploration policy following (22)
- 11: ⊲ Choose the reference model to control regularization level
- 12: if KL-regularized target in (16) then
- 13: Keep 𝜋𝑡+1,ref ← 𝜋0
- 14: else if Non-regularized target in (17) then
- 15: Update 𝜋𝑡+1,ref ← 𝜋1𝑡
- 16: end if
- 17: end for
- 18: Output: the best model in 𝜋11:𝑇 by a validation set.

which represents the cumulative performance loss over 𝑇 steps comparing the learned policies [𝜋1𝑡 ]𝑇𝑡=1 against the optimal policy 𝜋∗. In addition, we consider that a bounded 𝑢∗(𝑥, 𝑦) ∈ [0, 𝐵] for all (𝑥, 𝑦)

to maintain a reasonable utillity regime. Also, it is assumed that we have accesses to the following policy improvement oracle, that is analogue to the one considered in Xiong et al..

- Definition 3 (Policy Improvement Oracle). For any model M = (S, A, 𝐻, ℙ, 𝑑0, 𝑢) and a reference

function 𝜋ref, we can compute the optimal policy associated with the model [𝜋M,ℎ]ℎ𝐻=1 iteratively as in

(9).

The overall algorithm, i.e., the theoretical version of online iterative M-GSHF, is also summarized

in Algorithm 1. At each round 𝑡, with D = ∪𝑡𝑖=−11D𝑖 as the aggregated dataset, it starts with performing a maximum likelihood estimation (MLE) of the reward function 𝑢∗ over a set U, whose elements are

bounded in [0, 𝐵], as

𝐿𝑡(𝑢ˆ) := ∑︁

𝑧 log(𝜎(𝑢ˆ(𝜏1) − 𝑢ˆ(𝜏2))) + (1 − 𝑧) log(𝜎(𝑢ˆ(𝜏2) − 𝑢ˆ(𝜏1))) , (20)

𝑢ˆ𝑡 = argmax

𝑢ˆ∈U

(𝑥,𝜏1,𝜏2,𝑧)∈∪𝑡𝑖=−11D𝑖

and also an MLE of the transition kernel ℙ∗ over a set P as ℙˆ𝑡 = argmax

𝐿𝑡(ℙˆ) := ∑︁

logℙˆ𝜋(𝜏), (21)

ℙˆ∈P

(𝜋,𝜏)∈∪𝑡𝑖=−11D𝑖

where ℙ𝜋(𝜏) denotes the probability of trajectory 𝜏 under policy 𝜋 and transition kernel ℙ. With the obtained model Mˆ𝑡 = (𝑢ˆ𝑡, ℙˆ𝑡), the Oracle defined in Definition 3 is called with the reference policy 𝜋ref set as the initial policy 𝜋0, whose output is adopted as the main policy 𝜋1𝑡 .

Then, we specify how to choose a theoretically sound exploration policy 𝜋2𝑡 . The previous work of Xiong et al. on single-turn RLHF has demonstrated the intuition that the exploration policy should be in charge of collecting information of the uncertain parts of the environment M, which is thus often selected to maximize one uncertainty measurement. In the multi-turn RLHF setup considered in this work, the following proposition serves as the cornerstone to find a suitable uncertainty measurement to decide the exploration policy. In particular, we can observe that the optimal policy is parameterized by the optimal 𝑄-function. If a different set of 𝑄-function is adopted for policy parameterization, we can bound its performance as follows.

- Proposition 2 (Value Decomposition Lemma for KL-regularized MDP). If considering a set of 𝑄functions [𝑄ˆℎ]ℎ𝐻=1 and a reference policy 𝜋ref with the induced policy 𝜋ˆ as

𝜋ˆℎ(𝑎ℎ|𝑠ℎ) ∝ 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ) · exp 𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ)/𝜂 ,

and the corresponding set of 𝑉-functions [𝑉ˆℎ]ℎ𝐻=1 as 𝑉ˆℎ(𝑠ℎ) = 𝔼𝑎

ℎ∼𝜋ˆℎ(·|𝑠ℎ) 𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝜂𝐷KL(𝜋ˆℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ)), 𝑉ˆ𝐻+1(𝑠𝐻+1) = 0, for any comparator policy 𝜋, it holds that

𝐽(𝜋) − 𝐽(𝜋ˆ) = 𝔼𝑑0,𝜋,ℙ∗[𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑0,𝜋,ˆ ℙ∗[𝑢∗(𝑠𝐻, 𝑎𝐻)]

###### + ∑︁

𝔼𝑑0,𝜋,ℙ∗ 𝑉 ˆℎ+1(𝑠ℎ+1) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − ∑︁

𝔼𝑑0,𝜋,ˆ ℙ∗ 𝑉 ˆℎ+1(𝑠ℎ+1) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ)

− 𝜂 · ∑︁

ℎ∈[𝐻]

ℎ∈[𝐻]

𝔼𝑑0,𝜋,ℙ∗ [𝐷KL(𝜋ℎ(·|𝑠ℎ), 𝜋ˆℎ(·|𝑠ℎ))] ,

ℎ∈[𝐻]

where the expectation 𝔼𝑑0,𝜋,ℙ∗ is with respect to the prompt and response (i.e., the trajectory) generated following 𝑑0, ℙ∗ and 𝜋.

Based on Proposition 2, the exploration policy 𝜋2𝑡 is selected as 𝜋2𝑡 = argmax

0,𝜋, ℙ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

max

0,𝜋1𝑡 , ℙ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

0,𝜋1𝑡 , ℙ [ 𝑢(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋, ℙ [ 𝑢(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

𝔼𝑑

𝜋

𝑢∈ U𝑡, ℙ∈ P𝑡

uncertainty measurement of reward estimation

###### + ∑︁

0,𝜋, ℙ 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ)

, (22)

𝔼𝑑

ℎ∈[𝐻]

uncertainty measurement of transition estimation

where U𝑡 and P𝑡 are two confidence sets defined as

U𝑡 = {𝑢 ∈ U : 𝐿𝑡(𝑢) ≥ 𝐿𝑡(𝑢ˆ𝑡) − 𝑐1 log(|U|𝑇/𝛿)}, P𝑡 = {ℙ ∈ P : 𝐿𝑡(ℙ) ≥ 𝐿𝑡(ℙˆ𝑡) − 𝑐1 log(|P|𝑇/𝛿)}

(23)

with 𝑐1 denoting an absolute constant here. Note that for the theoretical convenience, we have assumed U and P are finite here, which can be extended to the infinite case using standard discretization

techniques. It can be observed that 𝜋2𝑡 is selected to maximize a combination of uncertainty from estimations of both rewards and transitions. If considering known transitions (i.e., without the need to estimate ℙ), the uncertainty from the estimation of transitions dimimishes, which leads to a similar uncertainty measurement adopted in Xiong et al..

The following theorem establishes a rigorous guarantee for the regret incurred.

Theorem 1. Assuming 𝑢∗ ∈ U and ℙ∗ ∈ P, with probability at least 1 − 𝛿, we have that

Reg(𝑇) ≲𝜅−1𝐵√︁𝑑U𝑇 log(|U|𝑇/𝛿) + 𝐵2𝐻𝜉(𝑑P,𝑇, 𝑐2 log(|P|𝐻𝑇/𝛿))

− 𝜂 · ∑︁

𝔼𝑑0,𝜋∗,ℙ∗ 𝐷KL(𝜋∗ℎ(·|𝑠ℎ), 𝜋1𝑡,ℎ(·|𝑠ℎ)) ,

ℎ∈[𝐻]

where 𝜅 := 1/(2 + exp(−𝐵) + exp(𝐵)), 𝑐2 is an absolute constant, 𝑑U is the Eluder coefficient defined in

- Definition 4 while 𝑑P and 𝜉(·) are from the generalized Eluder-type condition defined in Definition 5.

We note that the Eluder coefficient and the generalized Eluder-type condition are standard and well-adopted conditions in the theoretical studies on RL (Agarwal et al., 2023; Liu et al., 2023a; Xie et al., 2022; Zhang, 2023; Zhong et al., 2022) and also RLHF (Wang et al., 2023b; Ye et al., 2024; Zhan et al., 2023). Moreover, for a board class of RL problems (see Liu et al. (2023a); Zhang (2023) for more details), the Eluder coefficient 𝑑U is small and the condition is satisfied with 𝜉(𝑑P,𝑇, 𝑐2 log(|P|𝐻𝑇/𝛿)) ≲ √︁𝑑P𝑇 log(|P|𝐻𝑇/𝛿), which implies that the regret of theoretical version of Algorithm 1 is sublinear in 𝑇, further evidencing its statistical efficiency.

3. Experiments

3.1. Experiment Setup

Task, and datasets. We use the test sets of MATH (Hendrycks et al., 2021) and GSM8K (Cobbe et al., 2021a) to measure the model’s ability to solve the mathematical problems. The MATH dataset includes

- 5K problems across diverse mathematical fields such as algebra, geometry, probability, number theory, and calculus. The GSM8K test set consists of 1319 grade-school math word problems, which are generally simpler than those in the MATH dataset. Examples from each dataset are as follows:

- • GSM8K: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
- • MATH: Find the center of the circle with equation 𝑥2 − 6𝑥 + 𝑦2 + 2𝑦 = 9.

To effectively solve these problems, the model needs to perform multi-turn reasoning and arithmetic operations before getting the final answer. To construct the training prompt set, we follow Gou et al.

- (2023b); Liu and Yao (2024); Toshniwal et al. (2024); Yu et al. (2023); Yue et al. (2023) to use an augmented prompt set from the 7.5K training problems of MATH and 7.47K training problems of GSM8K. In particular, we use the prompts from MetaMathQA (Yu et al., 2023) and MMIQC (Liu and Yao, 2024). The new questions include rephrasing question, backward question (starting with the final answer and thinking backward to determine an unknown variable in the original question), and bootstrapping questions by in-context learning and iterative question composing (Liu and Yao, 2024). We delete the duplicate questions and also ensure that none from the test sets of MATH and GSM8K were used. Eventually, we have 60K training prompts in total for training and randomly split them into three disjoint sets for iterative training. We also reserve a set of 1K prompts for model selection during the training.

Base models. We train with a range of base models, including Gemma-1.1-it-7B (Team et al., 2024), CodeGemma-1.1-it-7B (Team, 2024), Mistral-7B-v0.3 (Jiang et al., 2023), and Gemma2-it-9B. We use the pre-trained version of Mistral instead of the instruction version because the chat template of its huggingface checkpoint and that of their own code base are not consistent so we start from the pre-trained model and fine-tune it by ourselves.

Data format and generation. We format the data into a multi-turn chat where the user initially ask the LLMs a question, and provide the messages returned by the Python interpreter in the subsequent user rounds of chat. In each model turn, the model reasons based the history gathered so far and can output a final answer enclosed in \boxed, or call the Python interpreter by writing a code wrapped in “‘python and “‘. After receiving the response of the model, we return the execution result of the code if the model calls the tool, and stop if the model outputs the final answer or reaches the maximal number of rounds H (6 in our setting). See Table ?? for an illustration. We generated N=30 samples per prompt for each iteration using a temperature setting of 1.0, without employing top-K or top-p sampling. We employ a mixture sampling strategy, where the up-to-date model generates only 20 trajectories, and the remainder (10 trajectories) are collected using the model from the last iteration. For the initial iteration, we employed models fine-tuned for 3 epochs and 1 epoch, respectively, to conduct mixture sampling. Intuitively, the mixture sampling helps to improve the diversity of the collected samples, and have been employed in previous RLHF practices (Bai et al., 2022; Dong et al., 2024). For all the data generation process, we adopt the following constraints: (1) for each turn, the model can generate up to 512 tokens; (2) the maximal number of steps is H=6; (3) the maximal number of token for each trajectory is 2048.

Supervised fine-tuning (SFT). We first fine-tune the model for the tool-integrated reasoning task (Gou et al., 2023b), using a subset of the Open-MathInstruct dataset, which was generated by the permissively licensed Mixtral-8x7B model through in-context learning. The problems are from the training sets of MATH and GSM8K datasets. We restrict the number of samples for each question to be 50 and remove the nearly duplicate responses. Eventually we get 510K samples in the SFT dataset. We train the models for 4 epochs at most with a learning rate of 5e-6 for Gemma instruct models (Team et al., 2024) and a learning rate of 1e-5 for Mistral-v0.3 model (Jiang et al., 2023). The learning rates are determined by searching {2e-6, 5e-6, 1e-5}. We use the pretrained model of Mistral because the chat template of Mistral instruct models are not consistent in different code bases (huggingface and the official one) at the time of our experiments. We use a cosine learning rate scheduler and set the warm-up steps as 100. The samples are packed into blocks with length 4096 to accelerate training and a global batch size of 64 is used. We also mask all the user messages (i.e., the prompt and the messages returned by the Python interpreter) in the training. It takes roughly 10-15 hours when training with 8xA100 80G GPUs. The checkpoint at the end of the third epoch is used for Gemma and the checkpoint of the end of the second epoch is used for Mistral as the starting point for RLHF. This is because these models outperform the last-iteration one with considerable margin and is very close to the next one. An ablation study on the SFT epochs is also included.

Data Annotation. For each prompt, we first divide the responses into the winning set 𝐺𝑤 and the losing set 𝐺𝑙 by checking the final answer. In practice, we observe that the model can memorize the final answer and output it even though the reasoning path itself is incorrect. To mitigate this issue, we include some heuristic filtering process. First, we delete all the trajectories in the winning set where the returned messages in the second last round indicate the code is with some bugs, but the models just ignore it and predict the ground-truth answer. Then, we delete the responses in both the winning set 𝐺𝑤 and losing set 𝐺𝑙 if they are longer than 2048 tokens. Finally, we randomly sample a trajectory from the 𝐺𝑤 and a trajectory from 𝐺𝑙 to construct a pair or to add them into the training set of KTO algorithm. For each iteration, we typically get 15K-20K samples because some of the prompts may not have any correct answer. We notice that it is possible to leverage AI feedback like Gemini (Team et al., 2023) or GPT4 (OpenAI, 2023) to further verify the correctness of the trajectory step by step or construct a PRM (Lightman et al., 2023; Wang et al., 2023a) to rank the trajectories, which we leave for future work.

Implementation of M-DPO and M-KTO. To implement the M-DPO, we simply set the labels of all the user-turn tokens to be -100 and mask the log-probability in the subsequent loss computation. We train the model for 1 epoch at most and tune the learning rate in {2e-7, 4e-7, 7e-7, 1e-6} with the first iteration of iterative training. Eventually, the learning rate of 4e-7 is used for Gemma-1.1 models and 2e-7 is used for Gemma-2 model and Mistral model. The global batch size is 32 with a warm-up step of 40. We evaluate the model every 50 training steps by the split prompt set and the best model is typically obtained between 150 steps to 600 steps, which is expected because the prompts for SFT and prompts for RLHF are overlapped. This has also been observed in previous work of RLHF for making general chatbot (Lin et al., 2023). Further exploration of prompt scaling is also left for future work. The hyper-parameters are of M-KTO are mostly the same as the M-DPO. We also set the 𝜆+ = 𝜆− = 1 following the original KTO paper (Ethayarajh et al., 2024). The RLHF experiments of this paper are run with 8xA100 80G GPUs, where an additional machine with 8xA100 40G GPUs is also used for data collection and model evaluation. The main experiment of this paper can be reproduced by 24 - 48 hours with this setup. We defer some other implementation details to Appendix B due to space constraint.

- 3.2. Main Results We evaluate the models in the zero-shot setting and report the main results in Table 1.

Competitors. The existing literature mainly focuses on the synthetic data generation and teach the models to use the external tool by supervised fine-tuning on the collected data. We use the results from Toshniwal et al. (2024) as baselines because we use the same SFT dataset so the results are generally comparable. For the CoT baselines, we use the Wizardmath models from Luo et al. (2023). We also include the reward ranked fine-tuning (RAFT) as a baseline (Dong et al., 2023), which is also known as rejection sampling fine-tuning in the literature (Touvron et al., 2023). RAFT first collects N trajectories per prompt, filters the low-quality data (by reward function), and fine-tune on the selected trajectories. Another baseline is the single-turn online iterative DPO and KTO (Ethayarajh et al., 2024; Rafailov et al., 2023), which ignores the problem structure (i.e., the external messages) and treats the trajectory as a whole. In implementation, it means that we do not mask the user turn and the tokens of external messages also contribute to the loss.

From the first two sections in Table 1, we first observe that the tool-integrated LLMs significantly outperform their CoT counterparts with only SFT, demonstrating the benefits of leveraging external tools. In the subsequent discussions, we focus on the comparison within the scope of tool-integrated LLMs.

Iterative M-DPO and M-KTO considerably improve the SFT models. We observe that for all the four base models, after the iterative training with M-DPO or M-KTO, the resulting model outperforms their starting SFT checkpoint with considerable margins on both GSM8K and MATH. In particular, with M-DPO, the aligned Gemma-1.1-it-7B model attains accuracies of 83.9% and 51.2% on GSM8K and MATH, respectively, and is comparable to the open-source Open-MathInstruct-finetuned CodeLLaMA-270B (slightly worse on GSM8K but also slightly better on MATH). Moreover, the aligned Gemma-2-it-9B model achieves accuracies of 86.3% and 54.5% on GSM8K and MATH, surpassing all of the opensource models trained with Open-MathInstruct in the 7B to 70B range. Overall, our framework can robustly further boost the tool-integrated models’ ability on the top of supervised fine-tuning.

- Table 1 | Main results of different methods on the test sets of GSM8K and MATH. The SFT training with external tool is based on (a subset of) Open-MathInstruct so the results are generally comparable to the previous SFT models. †: the model also serves as the starting checkpoint of other methods except for prompting and CoT without tool use. All the models are allowed to use code interpreter except for the CoT without tool use. The results of the CoT methods are borrowed from the technical reports (Gou et al., 2023b; Toshniwal et al., 2024). The gains relative to the SFT starting checkpoint are marked by ↑.

Base Model Method with Tool GSM8K MATH AVG

WizardMath-7B SFT for CoT ✗ 54.9 10.7 32.8 WizardMath-13B SFT for CoT ✗ 63.9 14.0 39.0 WizardMath-70B SFT for CoT ✗ 81.6 22.7 52.2

CodeLLaMA-2-7B SFT ✓ 75.9 43.6 59.8 CodeLLaMA-2-13B SFT ✓ 78.8 45.5 62.2 CodeLLaMA-2-34B SFT ✓ 80.7 48.3 64.5

LLaMA-2-70B SFT ✓ 84.7 46.3 65.5 CodeLLaMA-2-70B SFT ✓ 84.6 50.7 67.7

Gemma-1.1-it-7B SFT† ✓ 77.5 46.1 61.8 Gemma-1.1-it-7B RAFT ✓ 79.2 47.3 63.3 Gemma-1.1-it-7B Iterative Single-turn DPO ✓ 81.7 48.9 65.3 Gemma-1.1-it-7B Iterative Single-turn KTO ✓ 80.6 49.0 64.8 Gemma-1.1-it-7B Iterative M-DPO + fixed reference ✓ 79.9 48.0 64.0

- Gemma-1.1-it-7B M-DPO Iteration 1 ✓ 81.5 49.1 65.3

- Gemma-1.1-it-7B M-DPO Iteration 2 ✓ 82.5 49.7 66.1

- Gemma-1.1-it-7B M-DPO Iteration 3 ✓ 83.9 ↑6.4 51.2 ↑5.1 67.6 ↑5.8 Gemma-1.1-it-7B Iterative M-KTO ✓ 82.1 ↑4.6 49.5 ↑3.4 65.8 ↑4.0

CodeGemma-1.1-it-7B SFT† ✓ 77.3 46.4 61.9 CodeGemma-1.1-it-7B RAFT ✓ 78.8 48.4 63.6 CodeGemma-1.1-it-7B Iterative Single-turn DPO ✓ 79.1 48.9 64.0 CodeGemma-1.1-it-7B Iterative Single-turn KTO ✓ 80.2 48.6 64.4 CodeGemma-1.1-it-7B Iterative M-DPO ✓ 81.5 ↑4.2 50.1 ↑3.7 65.8 ↑4.0 CodeGemma-1.1-it-7B Iterative M-KTO ✓ 81.6 ↑4.3 49.6 ↑3.2 65.6 ↑3.8

Mistral-7B-v0.3 SFT† ✓ 77.8 42.7 60.3 Mistral-7B-v0.3 RAFT ✓ 79.8 43.7 61.8 Mistral-7B-v0.3 Iterative Single-turn DPO ✓ 79.8 45.1 62.5 Mistral-7B-v0.3 Iterative Single-turn KTO ✓ 81.3 46.3 63.8 Mistral-7B-v0.3 Iterative M-DPO ✓ 82.3 ↑4.5 47.5 ↑4.8 64.9 ↑4.7 Mistral-7B-v0.3 Iterative M-KTO ✓ 81.7 ↑3.9 46.7 ↑4.0 64.2 ↑4.0

Gemma-2-it-9B SFT† ✓ 84.1 51.0 67.6 Gemma-2-it-9B RAFT ✓ 84.2 52.6 68.4 Gemma-2-it-9B Iterative Single-turn DPO ✓ 85.2 53.1 69.2 Gemma-2-it-9B Iterative Single-turn KTO ✓ 85.4 52.9 69.2 Gemma-2-it-9B Iterative M-DPO ✓ 86.3 ↑2.2 54.5 ↑3.5 70.4 ↑2.9 Gemma-2-it-9B Iterative M-KTO ✓ 86.1 ↑2.0 54.5 ↑3.5 70.3 ↑2.8

Iterative M-DPO and M-KTO surpass existing RLHF baselines. We also observe that the iterative M-DPO and M-KTO surpass other existing RLHF baselines. First, they consistently and significantly outperform the RAFT algorithm across all four base models, which is known to be a robust and competitive baseline in the literature (Dong et al., 2023; Yuan et al., 2023a). This is because the RAFT algorithm only utilizes the positive signal by imitating the correct trajectories, while the DPO-based and KTO-based algorithms further leverage the negative signal from those incorrect trajectories. We note that the SFT stage in our pipeline can also be viewed as an application of RAFT, an idea that further dates back to expert iteration (Anthony et al., 2017). Consequently, our results should be interpreted to be that on the top of the first stage of SFT, algorithms with negative signal are more sample efficient. Moreover, while the online iterative single-turn DPO (KTO) (Xiong et al.; Xu et al., 2023) also gives a boost performance, it is generally worse than the multi-turn version. This suggests that learning to predict the off-policy external messages returned by the code interpreter usually has a negative impact on the reasoning ability improvement. Essentially, this corresponds to the fact that when deriving the optimality condition of the KL-regularized optimization problem, we are not allowed to optimize the external messages. Meanwhile, we present a representative example we encounter in Table ??, where LLMs generate poorly constructed code resulting in anomalous and lengthy external messages. Forcing LLMs to learn to predict these messages can significantly hurt the model’s reasoning abilities.

Iterative training and reference update lead to better performance. We use the Gemma-1.1-it-7B and M-DPO as a representative example and observe that the model benefits from online iterative training, where the test accuracy of GSM8K improves from 77.5% (SFT) to 81.5% (iter 1) to 82.5% (iter2) to 83.9% (iter3), and the test accuracy of MATH improves from 46.1% (SFT) to 49.1% (iter 1) to 49.7% (iter2) to 51.2% (iter3). This is consistent with our theoretical insight that iterative training allows the models to explore the underlying space and learn the optimal policy progressively. Moreover, we observe that if we fix the reference model as the SFT policy, the final model performance is much worse compared to the case where we update the reference model as the current model at every iteration. We suspect that this is because this version of algorithm essentially optimizes the non-regularized reward and the reward in the mathematical reasoning task is more accurate than those in the general chat task, leading to the superior in-domain performance. We defer a more detailed ablation study on the impact of KL regularization to next subsection.

GSM8K Test Accuracy

MATH Test Accuracy

90

95

80

Pass@n(%)

Pass@n(%)

90

70

60

85

Gemma-7B SFT

Gemma-7B SFT

50

80

Gemma-7B M-DPO + Fixed Reference

Gemma-7B M-DPO + Fixed Reference

Gemma-7B M-DPO + Update Reference

Gemma-7B M-DPO + Update Reference

40

1 2 4 8 16 32 64

1 2 4 8 16 32 64

n: the number of candidates

n: the number of candidates

- Figure 2 | The pass@n rate with respect to the number of candidates n. We evaluate the models using temperature 0.7 following the previous works Shao et al. (2024); Toshniwal et al. (2024). We notice that preference learning only improves the metric pass@n when n is relatively small.

Preference learning improves pass@n only when n is relatively small. We plot the pass@n accuracy in terms of the number of candidate trajectories n in Figure 2. To evaluate the pass@n, for each question, we independently sample n trajectories, and the question is considered to be solved if there exists at least one trajectory with the correct final answer. We observe that the preference learning only improves the pass@n when n is relatively small. In particular, when 𝑛 > 16, all the models perform similarly on both GSM8K and MATH. In other words, the iterative M-DPO does not inject new knowledge but elicits the models’ knowledge acquired in pre-training and SFT stages by boosting the quality of Top n responses. The observation is consistent with that of Shao et al. (2024), which studies the DRL-based GRPO method for the CoT mathematical reasoning task. Therefore, the success of preference learning is on top of a well-trained SFT model. We expect that the final model performance can be further improved with more high-quality SFT data.

##### 3.3. Ablation Study and Discussion

We conduct ablation studies in this subsection for a more comprehensive understanding of the proposed algorithm.

A moderate level of KL regularization balances the per-iteration improvement and exploration efficiency. The effectiveness of (iterative) DPO is significantly influenced by the choice of reference model and the KL coefficient. Previous research by Tunstall et al. (2023) on offline DPO for general chatbot applications suggests that a lower KL coefficient, specifically 0.01, yields superior performance by allowing the resulting model to move more far away from the SFT model 𝜋0. Meanwhile, for online iterative training, Dong et al. (2024); Xiong et al. adopt a fixed reference model of 𝜋0, and achieves continuous improvements as the training iterates. In our ablation study, we consider two different choices: (1) using the fixed reference model 𝜋0; (2) updating the reference model to the last iteration’s model at each round. Moreover, we search the KL coefficient 𝜂 ∈ {0.01, 0.1, 0.5}. The results are summarized in Table 2. First, we notice that if we update the reference model at each iteration, the final model outperforms the one with a fixed reference model 𝜋0 with a large margin. Essentially, this dynamic approach optimizes the non-regularized reward, while the one with a fixed reference model 𝜋0 aims to maximize the KL-regularized reward. This can be viewed as a trade-off between the generation diversity and reward optimization. We suspect this performance difference is because for reasoning task, the correct reasoning paths are highly concentrated on a small subset of the generation space, and the diversity is less important in this case.

We also find that the strongest model is obtained by a moderate KL coefficient of 0.1, outperforming both 0.01 and 0.5. To understand this phenomena, we plot the test accuracy of GSM8K in Figure 3 along the way of iterative training. As we can see, for the first iteration, the results align with Tunstall

- et al. (2023)’s findings, where a smaller KL coefficient leads to a larger model improvement. However, the resulting intermediate model is further used to collect trajectories for subsequent iterative training. The models trained with very low KL coefficients tend to lose diversity rapidly, potentially reducing their capacity to collect diverse trajectories for subsequent training, leading to diminishing gains in the second and third iterations. In contrast, a higher KL coefficient of 0.5 imposes strong regularization between the resulting model and the reference model, and the model improvement is less compared to that of 0.1 for each iteration. To summarize, for online iterative training, we need to strike a balance between the per-iteration improvement and exploration efficiency to optimize the overall performance. We will see that such an intuition also extends to the choices of sampling strategy choice and other experimental tricks.

- Table 2 | Ablation study of the impact of KL regularization. The SFT policy is the starting checkpoint for all other experiments.

##### Model Method GSM8K MATH

Gemma-1.1-it-7B SFT 3 epoch 77.5 46.1 Gemma-1.1-it-7B Iterative M-DPO + 𝜂 = 0.01 81.7 50.1 Gemma-1.1-it-7B Iterative M-DPO + 𝜂 = 0.1 83.9 51.2 Gemma-1.1-it-7B Iterative M-DPO + 𝜂 = 0.5 82.8 49.7

Gemma-1.1-it-7B Iterative M-DPO + fixed reference + 𝜂 = 0.1 79.9 48.0

GSM8K Test Accuracy

- 76

- 77

- 78

- 79

- 80

- 81

- 82

- 83

- 84

Accuracy

SFT

eta = 0.01

eta = 0.1 eta = 0.5 eta = 0.1 + fixed reference

0 1 2 3

Iteration

- Figure 3 | The plot of test accuracy on GSM8K dataset and iterations with different levels of KL regularization.

The impact of sampling strategy: data diversity and coverage are crucial. Throughout our iterative training process of the Gemma-1.1-it-7B, we observed a steady increase in the percentage of correct trajectories—from 47% in the first iteration to 76% in last iteration. Moreover, since we update the reference model at each iteration, the diversity of the generated trajectories also decrease rapidly. However, the diversity of the collected data is critical for DPO/KTO training due to their contrastive nature. Prior studies in online iterative DPO for general chatbots (Dong et al., 2024) recommend employing model variants with different sampling temperatures or training steps to enhance trajectory diversity. Motivated by this, we explored two data collection strategies: (1) on-policy sampling, where all trajectories are sampled using the current policy, and (2) mixture sampling, where 20 trajectories are collected using the current model and 10 from the last iteration’s model. We report the results in Table 5, where with mixture sampling, the final model performance considerably outperform the one with only on-policy sampling. To understand this phenomena, we plot the MATH test accuracy in terms of the iteration in Figure 4. We observe that on-policy sampling fails to improve MATH test accuracy in the third iteration, while we achieve considerable gain with the mixture sampling. This again demonstrates the importance of the diversity of the collected responses in the iterative training and also aligns with previous findings that advanced exploration strategies, which prevent diversity

collapse, provide more meaningful signals for iterative preference learning (Bai et al., 2022; Dong et al., 2024; Pace et al., 2024; Touvron et al., 2023; Xiong et al.). It would be interested to explore more advanced exploration strategy like Monte Carlo tree search (MCTS) in the future study.

In our experiments, we collected N trajectories per prompt to ensure the presence of both correct and incorrect reasoning paths for constructing the comparison pair. A larger N generally leads to a better coverage of the prompt set because for some difficult problem, we need to sample more responses to find a correct reasoning path. For instance, in iteration 1, with N=30, 92.5% of the prompts are covered, compared to 83.0% for N=12 and 60% for N=6. See Figure 2 for an illustration of the relationship between pass@1 and N. However, increasing N also incurs higher computational costs. To understand the impact of the parameter N, we conduct an ablation study with 𝑁 ∈ {6, 12, 30} and summarize the results in Table 3. We observe a substantial performance boost when increasing N from 6 to 12, reflecting a better coverage of the complex problems that require more attempts to find a correct path. In contrast, from N=12 to N=30, we only get very minor improvement in the test accuracy, suggesting that the incremental benefits of increasing N in best-of-N sampling diminish rapidly.

- Table 3 | Ablation study of the impact of sampling strategy. The SFT policy is the starting checkpoint for all other experiments. Mixture sampling is adopted for the iterative M-DPO training by default and we run for three iterations in total.

##### Model Method GSM8K MATH

Gemma-1.1-it-7B SFT 3 epoch 77.5 46.1 Gemma-1.1-it-7B Iterative M-DPO with N=30 83.9 51.2 Gemma-1.1-it-7B Iterative M-DPO with N=12 83.5 51.2 Gemma-1.1-it-7B Iterative M-DPO with N=6 82.0 49.2 Gemma-1.1-it-7B Iterative M-DPO with N=30 + On-policy sampling 83.1 49.5

MATH Test Accuracy

SFT 3 epochs

- 45

- 46

- 47

- 48

- 49

- 50

- 51

Mixture Sampling

On-policy Sampling

Accuracy

0 1 2 3

Iteration

- Figure 4 | The plot of test accuracy on MATH dataset in terms of training iterations with different sampling strategies.

The best model is obtained with starting checkpoint fine-tuned with more than 1 epochs. Tunstall et al. (2023) finds that if the SFT model is trained for more than one epoch, the subsequent DPO training will lead to performance regression with longer training in terms of instruction-following ability and benchmark for a general chatbot. In other words, there exists a trade-off between the SFT training epochs and the DPO training steps. Moreover, the best model is obtained by SFT for one epoch in their practice. We also conduct an ablation study on the impact of the SFT epoch and summarize the results in Table 4. Consistently across all tested scenarios, the subsequent iterative M-DPO training leads to considerable model improvement compared to the SFT model. Meanwhile, we also observe a similar trade-off between SFT and RLHF training because with more SFT epochs, the gains from the RLHF stage decrease. However, in our case, the strongest model is obtained with three epochs of SFT, followed by fine-tuning through iterative M-DPO, which is different from the offline DPO training (Tunstall et al., 2023) or the iterative DPO for general chatbot (Dong et al., 2024) with only one epoch of SFT.

- Table 4 | Ablation study of the impact of SFT epoch. Mixture sampling is adopted for the iterative M-DPO training and we run for three iterations in total. The gains relative to their starting SFT checkpoints are marked by ↑.

##### Model Method GSM8K MATH

- Gemma-1.1-it-7B SFT 1 epoch 75.1 41.1

- Gemma-1.1-it-7B SFT 1 epoch + Iterative M-DPO 80.6 ↑5.5 46.7 ↑5.6

- Gemma-1.1-it-7B SFT 2 epoch 75.3 44.0

Gemma-1.1-it-7B SFT 2 epoch + Iterative M-DPO 82.4 ↑7.1 49.8 ↑5.8 Gemma-1.1-it-7B SFT 3 epoch 77.5 46.1

- Gemma-1.1-it-7B SFT 3 epoch + Iterative M-DPO 83.9 ↑6.4 51.2 ↑5.1

##### NLL loss helps when the SFT model is substantially underfitting. The recent work Pang et al.

- (2024) has introduced iterative RPO, specifically aimed at enhancing Chain of Thought (CoT) capabilities for solving mathematical problems. A key feature of this approach is the inclusion of an additional negative log-likelihood (NLL) loss for the preferred response. The main intuition for adding the NLL loss is that the original DPO algorithm (Rafailov et al., 2023) tends to reduce the likelihood of the preferred responses, and this is believed to hurt the reasoning ability (Wang et al., 2024). Motivated by their results, we explored the applicability of this idea to our setup. We conduct an ablation study by adding the NLL loss into the iterative M-DPO training and observe performance regression as reported in Table 5. We observe that the best model is obtained in the second iteration if we add the additional NLL loss even though we use the mixture sampling to increase the diversity of the collected data. With time-weighted exponential moving average for smoothing training record, we observe that the log probability of the chosen responses and rejected responses are (-126, -222) at the 200th step of the third iteration training when we add the NLL loss, as compared to (-166, -350) in the case without the NLL loss. This is consistent with the result of Pang et al. (2024) where with the additional NLL loss, both the log probability of chosen responses and that of rejected responses increase. These evidences indicate that the NLL loss further contributes to the model distribution collapse and eventually hurt the overall performance of online iterative learning. Finally, we notice that the additional NLL loss can be viewed as an implementation of the pessimistic principle (Liu et al., 2024b). This also explains its inferior in-domain performance though it may be helpful to stable the training, which requires more in-depth studies.

However, one distinct feature between our setup and Pang et al. (2024) is whether we first

fine-tune the initialized SFT model with in-domain data. To further understand the phenomena, we fine-tune the Gemma-1.1-it-7B with only 100 steps (so that the model knows to leverage Python code to solve the problem) as the starting checkpoint of preference learning and conduct an ablation study with the NLL loss using this model. We observe when the SFT model is substantially underfitting, the addition of NLL loss actually enhances performance. This scenario mirrors the findings of Pang

- et al. (2024), who utilized a general LLaMA2-70B-chat model (Touvron et al., 2023) without firstly fine-tuning on the in-domain data. Our observations align with prior research in the context of developing general chatbots (Lin et al., 2023), which suggests that RLHF is less effective without preliminary SFT.

- Table 5 | Other ablation studies. Mixture sampling is adopted for the iterative M-DPO training and we run for three iterations in total. The gains relative to the iterative M-DPO are marked by ↑.

##### Model Method GSM8K MATH

Gemma-1.1-it-7B SFT 3 epoch 77.5 46.1 Gemma-1.1-it-7B SFT 3 epoch + Iterative M-DPO 83.9 51.2 Gemma-1.1-it-7B Iterative M-DPO with NLL loss 81.7 ↓2.2 49.5 ↓1.7

Gemma-1.1-it-7B SFT 100 steps 50.8 23.7 Gemma-1.1-it-7B + M-DPO Iteration 1 57.8 27.9 Gemma-1.1-it-7B + M-DPO and NLL loss Iteration 1 61.0 ↑3.2 30.1 ↑2.2

On-policy sampling and small learning rate mitigate the probability drops in preferred responses. In the literature, the Direct Preference Optimization (DPO) algorithm is often reported to diminish reasoning capabilities by reducing the likelihood of preferred responses (Hong et al., 2024; Meng et al., 2024; Yuan et al., 2024). In our preliminary experiments, we also observe similar phenomena with a large learning rate (1e-6), where the model’s reasoning ability collapses after only a few training steps, preventing convergence to good reasoning performance. In contrast, we find that using on-policy sampling within our online iterative training framework, coupled with a smaller learning rate (2e-7 or 4e-7), the DPO algorithm enhances the model’s reasoning abilities. To interpret our observation, we can first write down the gradient of the DPO as follows:

1 𝜋𝜃(𝑦𝑤|𝑥)

1 𝜋𝜃(𝑦𝑙|𝑥)

∇𝜃L𝐷𝑃𝑂(𝜋𝜃, 𝜋ref) = −𝜂 · 𝜎 𝑟𝜃(𝑥, 𝑦𝑙) − 𝑟𝜃(𝑥, 𝑦𝑤)

∇𝜃𝜋𝜃(𝑦𝑤|𝑥) −

∇𝜃𝜋𝜃(𝑦𝑙|𝑥) ,

where 𝑟𝜃(𝑥, 𝑦) = 𝜂 log 𝜋𝜋𝜃(𝑥,𝑦)

ref(𝑥,𝑦) is the implicit reward and we use the single-turn one for simplicity. In practice, the probability of the rejected responses typically decrease, and their gradient quickly dominates when 𝜋𝜃(𝑦𝑙|𝑥) << 𝜋𝜃(𝑦𝑤|𝑥) and the optimization becomes unlearning of the rejected responses. In this case, the probability of the chosen responses cannot increase. This phenomenon was also discussed in the blog Guo et al. (2024a). When we adopt on-policy sampling, it leads to a relatively large probability for both rejected and chosen responses at the initial stage, ensuring that both gradients remain valid and effective. Moreover, a small learning rate prevents the model from deviating too significantly, maintaining the effectiveness of both gradients. We also notice that for the KTO algorithm, the preferred responses and the rejected responses do not appear in pairs. We suspect that the probability of the preferred response increases because the gradients of the rejected response do not dominate in every mini-batch of data. A more comprehensive understanding of the training dynamic of the direct preference learning algorithms remains largely open and we leave a more detailed study of this phenomena to future study.

### 4. Conclusion, Limitation, and Future Research Direction

We demonstrate that preference learning, as an alternative learning paradigm to supervised finetuning, can further boost the performance of the tool-integrated reasoning LLMs on top of iterative best-of-n fine-tuning. We introduce an online iterative multi-turn direct preference optimization algorithm and validate its effectiveness through extensive experimentation across various base models. Our results indicate substantial improvements in the pass@1 metric over the SFT policy, as evidenced by performance gains on standard benchmarks such as GSM8K (Cobbe et al., 2021a) and MATH (Hendrycks et al., 2021). Additionally, we also conduct various ablation studies to show that achieving optimal performance requires a careful balance between per-iteration improvement and exploration, facilitated by moderate levels of KL regularization and strategic exploration choices.

There are also several potential directions to further improve the model performance that we have not explored in this paper. Currently, our experiments only use final result check as the preference signal, so we cannot effectively compare trajectories that both end with correct or incorrect answers. Although our algorithm is designed for trajectory-level preference learning, the reward signal in the Bradley-Terry model could be adapted to a step-wise level. In particular, we may leverage AI feedback to verify trajectories step by step or train a process-supervised reward model (Lightman et al., 2023) to provide learning signals. Additionally, with more fine-grained reward signals, it is also possible to adopt more advanced heuristic exploration policy like west-of-n sampling, which prove to be effective in the practice of making general chatbot (Dong et al., 2024; Hoang Tran, 2024; Pace et al., 2024; Xu et al., 2023) and Monte Carlo tree search (MCTS) (Xie et al., 2024b). Furthermore, it is also possible to leverage some well-established tricks like adaptive margin and length regularization for DPO training (Hong et al., 2024; Meng et al., 2024). These techniques have proven to be effective for achieving a better in-domain performance for the chat task. We expect that these more fine-grained preference signals and algorithmic designs can largely improve the models’ performance.

Finally, while the direct preference learning algorithms show promising gains for the mathematical reasoning tasks with code interpreter, it is not directly applicable to the general agent learning with more complex and stochastic external environments or against dynamic opponents. In particular, it requires to construct a value network for involving an adaptive margin in the optimization target and take the randomness of the external environment into consideration. We leave the study of this more involved algorithm to the future work. Moving beyond the framework presented this paper, it is also possible to explore more general preference structures beyond the BT model (Munos et al., 2023; Ye et al., 2024). We hope that the insights from this paper will inspire further research in this direction, extending the utility of preference learning beyond the general structured chat tasks.

### Acknowledgements

Wei Xiong and Tong Zhang are partially supported by an NSF IIS grant No. 2416897

### References

Qwen2 technical report. 2024.

- A. Agarwal, Y. Jin, and T. Zhang. VO𝑄L: Towards optimal regret in model-free rl with nonlinear function approximation. In The Thirty Sixth Annual Conference on Learning Theory, pages 987–1063. PMLR, 2023.

T. Anthony, Z. Tian, and D. Barber. Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems, 30, 2017.

#### Anthropic. Introducing claude. 2023. URL https://www.anthropic.com/index/ introducing-claude.

- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47:235–256, 2002.

M. G. Azar, M. Rowland, B. Piot, D. Guo, D. Calandriello, M. Valko, and R. Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036, 2023.

- Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Q. Cai, Z. Yang, C. Jin, and Z. Wang. Provably efficient exploration in policy optimization. In International Conference on Machine Learning, pages 1283–1294. PMLR, 2020.

- S. Cen, J. Mei, K. Goshvadi, H. Dai, T. Yang, S. Yang, D. Schuurmans, Y. Chi, and B. Dai. Valueincentivized preference optimization: A unified approach to online and offline rlhf. arXiv preprint arXiv:2405.19320, 2024.

- G. Chen, M. Liao, C. Li, and K. Fan. Step-level value preference optimization for mathematical reasoning. arXiv preprint arXiv:2406.10858, 2024a.

- W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

Z. Chen, Y. Deng, H. Yuan, K. Ji, and Q. Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024b.

L. Choshen, L. Fox, Z. Aizenbud, and O. Abend. On the weaknesses of reinforcement learning for neural machine translation. arXiv preprint arXiv:1907.01752, 2019.

- P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton,

- R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021a.

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton,

- R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021b.

T. Coste, U. Anwar, R. Kirk, and D. Krueger. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

- H. Dong, W. Xiong, D. Goyal, Y. Zhang, W. Chow, R. Pan, S. Diao, J. Zhang, K. SHUM, and T. Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum? id=m7p5O7zblY.

H. Dong, W. Xiong, B. Pang, H. Wang, H. Zhao, Y. Zhou, N. Jiang, D. Sahoo, C. Xiong, and T. Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

- L. Engstrom, A. Ilyas, S. Santurkar, D. Tsipras, F. Janoos, L. Rudolph, and A. Madry. Implementation matters in deep policy gradients: A case study on ppo and trpo. arXiv preprint arXiv:2005.12729, 2020.

- K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.
- L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023a.

- L. Gao, J. Schulman, and J. Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023b.

Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, N. Duan, and W. Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023a.

Z. Gou, Z. Shao, Y. Gong, Y. Yang, M. Huang, N. Duan, W. Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023b.

L. Gui, C. Gârbacea, and V. Veitch. Bonbon alignment for large language models and the sweetness of best-of-n sampling. arXiv preprint arXiv:2406.00832, 2024.

C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, A. Sharma, A. Siddhant, A. Ahern, M. Wang, C. Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

- S. Guo, W. Xiong, and C. Wang. "alignment guidebook. Notion Blog, 2024a.

- S. Guo, B. Zhang, T. Liu, T. Liu, M. Khalman, F. Llinares, A. Rame, T. Mesnard, Y. Zhao, B. Piot, et al. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792, 2024b.

- D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

#### B. H. Hoang Tran, Chris Glaze. Snorkel-mistral-pairrm-dpo. https://huggingface.co/ snorkelai/Snorkel-Mistral-PairRM-DPO, 2024. URL https://huggingface.co/ snorkelai/Snorkel-Mistral-PairRM-DPO.

J. Hong, N. Lee, and J. Thorne. Orpo: Monolithic preference optimization without reference model. arXiv preprint arXiv:2403.07691, 2(4):5, 2024.

A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand,

- G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

- F. Jiao, C. Qin, Z. Liu, N. F. Chen, and S. Joty. Learning planning-based reasoning by trajectories collection and process reward synthesizing. arXiv preprint arXiv:2402.00658, 2024.

X. Lai, Z. Tian, Y. Chen, S. Yang, X. Peng, and J. Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024.

- H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

- Y. Lin, L. Tan, H. Lin, Z. Zheng, R. Pi, J. Zhang, S. Diao, H. Wang, H. Zhao, Y. Yao, et al. Speciality vs generality: An empirical study on catastrophic forgetting in fine-tuning foundation models. arXiv preprint arXiv:2309.06256, 2023.

H. Liu and A. C.-C. Yao. Augmenting math word problems via iterative question composing. arXiv preprint arXiv:2401.09003, 2024.

- Q. Liu, P. Netrapalli, C. Szepesvari, and C. Jin. Optimistic mle: A generic model-based algorithm for partially observable sequential decision making. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 363–376, 2023a.

- T. Liu, Y. Zhao, R. Joshi, M. Khalman, M. Saleh, P. J. Liu, and J. Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023b.

- T. Liu, Z. Qin, J. Wu, J. Shen, M. Khalman, R. Joshi, Y. Zhao, M. Saleh, S. Baumgartner, J. Liu, et al. Lipo: Listwise preference optimization through learning-to-rank. arXiv preprint arXiv:2402.01878, 2024a.

Z. Liu, M. Lu, S. Zhang, B. Liu, H. Guo, Y. Yang, J. Blanchet, and Z. Wang. Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer. arXiv preprint arXiv:2405.16436, 2024b.

Z. Lu, A. Zhou, K. Wang, H. Ren, W. Shi, J. Pan, and M. Zhan. Step-controlled dpo: Leveraging stepwise error for enhanced mathematical reasoning. arXiv preprint arXiv:2407.00782, 2024.

H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.

- Y. Meng, M. Xia, and D. Chen. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

Meta. Introducing meta llama 3: The most capable openly available llm to date. Meta AI Blog, 2024.

https://ai.meta.com/blog/meta-llama-3/.

S. Mishra, M. Finlayson, P. Lu, L. Tang, S. Welleck, C. Baral, T. Rajpurohit, O. Tafjord, A. Sabharwal, P. Clark, et al. Lila: A unified benchmark for mathematical reasoning. arXiv preprint arXiv:2210.17517, 2022.

A. Mitra, H. Khanpour, C. Rosset, and A. Awadallah. Orca-math: Unlocking the potential of slms in grade school math. arXiv preprint arXiv:2402.14830, 2024.

- R. Munos, M. Valko, D. Calandriello, M. G. Azar, M. Rowland, Z. D. Guo, Y. Tang, M. Geist, T. Mesnard, A. Michi, et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.

A. S. Nemirovskij and D. B. Yudin. Problem complexity and method efficiency in optimization. 1983. OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023. L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama,

A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

- A. Pace, J. Mallinson, E. Malmi, S. Krause, and A. Severyn. West-of-n: Synthetic preference generation for improved reward modeling. arXiv preprint arXiv:2401.12086, 2024.

- R. Y. Pang, W. Yuan, K. Cho, H. He, S. Sukhbaatar, and J. Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733, 2024.

R. Pi, T. Han, W. Xiong, J. Zhang, R. Liu, R. Pan, and T. Zhang. Strengthening multimodal large language model with bootstrapped preference optimization. arXiv preprint arXiv:2403.08730, 2024.

R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

- R. Rafailov, J. Hejna, R. Park, and C. Finn. From r to q*: Your language model is secretly a q-function. arXiv preprint arXiv:2404.12358, 2024.

P. H. Richemond, Y. Tang, D. Guo, D. Calandriello, M. G. Azar, R. Rafailov, B. A. Pires, E. Tarassov, L. Spangher, W. Ellsworth, et al. Offline regularised reinforcement learning for large language models alignment. arXiv preprint arXiv:2405.19107, 2024.

C. Rosset, C.-A. Cheng, A. Mitra, M. Santacroce, A. Awadallah, and T. Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715, 2024.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

L. Shani, A. Rosenberg, A. Cassel, O. Lang, D. Calandriello, A. Zipori, H. Noga, O. Keller, B. Piot,

- I. Szpektor, et al. Multi-turn reinforcement learning from preference human feedback. arXiv preprint arXiv:2405.14655, 2024.

- Z. Shao, F. Huang, and M. Huang. Chaining simultaneous thoughts for numerical reasoning. arXiv preprint arXiv:2211.16482, 2022.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, P. J. Liu, J. Harrison, J. Lee, K. Xu, A. Parisi, et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

- R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

- G. Swamy, C. Dann, R. Kidambi, Z. S. Wu, and A. Agarwal. A minimaximalist approach to reinforcement learning from human feedback. arXiv preprint arXiv:2401.04056, 2024.

- F. Tajwar, A. Singh, A. Sharma, R. Rafailov, J. Schneider, T. Xie, S. Ermon, C. Finn, and A. Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367, 2024.

Y. Tang, Z. D. Guo, Z. Zheng, D. Calandriello, R. Munos, M. Rowland, P. H. Richemond, M. Valko, B. Á. Pires, and B. Piot. Generalized preference optimization: A unified approach to offline alignment. arXiv preprint arXiv:2402.05749, 2024.

C. Team. Codegemma: Open code models based on gemma. arXiv preprint arXiv:2406.11409, 2024.

- G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Y. Tong, X. Zhang, R. Wang, R. Wu, and J. He. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. 2024.

S. Toshniwal, I. Moshkov, S. Narenthiran, D. Gitman, F. Jia, and I. Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. arXiv preprint arXiv:2402.10176, 2024.

- H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

L. Tunstall, E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, S. Huang, L. von Werra, C. Fourrier, N. Habib, et al. Zephyr: Direct distillation of lm alignment. arXiv preprint arXiv:2310.16944, 2023.

- J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. CoRR, abs/2312.08935, 2023a.

- X. Wang, Z. Wang, J. Liu, Y. Chen, L. Yuan, H. Peng, and H. Ji. Mint: Multi-turn interactive evaluation for tool-augmented llms with language feedback. In Proc. The Twelfth International Conference on Learning Representations (ICLR2024), 2024.
- Y. Wang, Q. Liu, and C. Jin. Is rlhf more difficult than standard rl? arXiv preprint arXiv:2306.14111, 2023b.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

- R. J. Williams and J. Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268, 1991.

- T. Xie, D. J. Foster, Y. Bai, N. Jiang, and S. M. Kakade. The role of coverage in online reinforcement learning. arXiv preprint arXiv:2210.04157, 2022.

- T. Xie, D. J. Foster, A. Krishnamurthy, C. Rosset, A. Awadallah, and A. Rakhlin. Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf. arXiv preprint arXiv:2405.21046, 2024a.

- Y. Xie, A. Goyal, W. Zheng, M.-Y. Kan, T. P. Lillicrap, K. Kawaguchi, and M. Shieh. Monte carlo tree search boosts reasoning via iterative preference learning. arXiv preprint arXiv:2405.00451, 2024b.

- W. Xiong, H. Dong, C. Ye, Z. Wang, H. Zhong, H. Ji, N. Jiang, and T. Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In Forty-first International Conference on Machine Learning.

- J. Xu, A. Lee, S. Sukhbaatar, and J. Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

C. Ye, W. Xiong, Y. Zhang, N. Jiang, and T. Zhang. A theoretical analysis of nash learning from human feedback under general kl-regularized preference. arXiv preprint arXiv:2402.07314, 2024.

L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

L. Yuan, G. Cui, H. Wang, N. Ding, X. Wang, J. Deng, B. Shan, H. Chen, R. Xie, Y. Lin, et al. Advancing llm reasoning generalists with preference trees. arXiv preprint arXiv:2404.02078, 2024.

Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023a.

Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023b.

- X. Yue, G. Z. Xingwei Qu, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.

- X. Yue, T. Zheng, G. Zhang, and W. Chen. Mammoth2: Scaling instructions from the web. arXiv preprint arXiv:2405.03548, 2024.

E. Zelikman, Y. Wu, J. Mu, and N. Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

- W. Zhan, M. Uehara, N. Kallus, J. D. Lee, and W. Sun. Provable offline reinforcement learning with human feedback. arXiv preprint arXiv:2305.14816, 2023.

B. Zhang, K. Zhou, X. Wei, X. Zhao, J. Sha, S. Wang, and J.-R. Wen. Evaluating and improving tool-augmented computation-intensive math reasoning. Advances in Neural Information Processing Systems, 36, 2024a.

S. Zhang, D. Yu, H. Sharma, Z. Yang, S. Wang, H. Hassan, and Z. Wang. Self-exploring language

models: Active preference elicitation for online alignment. arXiv preprint arXiv:2405.19332, 2024b. T. Zhang. Mathematical analysis of machine learning algorithms. Cambridge University Press, 2023.

- Y. Zhang, D. Yu, B. Peng, L. Song, Y. Tian, M. Huo, N. Jiang, H. Mi, and D. Yu. Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning. arXiv preprint arXiv:2407.00617, 2024c.

- Y. Zhao, R. Joshi, T. Liu, M. Khalman, M. Saleh, and P. J. Liu. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

- C. Zheng, Z. Wang, H. Ji, M. Huang, and N. Peng. Weak-to-strong extrapolation expedites alignment. arXiv preprint arXiv:2404.16792, 2024.

- K. Zheng, J. M. Han, and S. Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021.

- H. Zhong, W. Xiong, S. Zheng, L. Wang, Z. Wang, Z. Yang, and T. Zhang. Gec: A unified framework for interactive decision making in mdp, pomdp, and beyond. arXiv preprint arXiv:2211.01962, 2022.

- H. Zhong, G. Feng, W. Xiong, L. Zhao, D. He, J. Bian, and L. Wang. Dpo meets ppo: Reinforced token optimization for rlhf. arXiv preprint arXiv:2404.18922, 2024.

- D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. Le, et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022.

- X. Zhu, J. Wang, L. Zhang, Y. Zhang, Y. Huang, R. Gan, J. Zhang, and Y. Yang. Solving math word problems via cooperative reasoning induced language models. arXiv preprint arXiv:2210.16257, 2022.

- B. D. Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy. Carnegie Mellon University, 2010.

- D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

### A. Notation Table

|Notation|Description|
|---|---|

|𝑥, X 𝑑0 𝑠ℎ ∈ S, 𝑎ℎ ∈ A, 𝑜ℎ 𝐻 ℙ∗ = [ℙ∗ℎ]ℎ𝐻=1 𝜏 = (𝑥, 𝑦) 𝑢∗ M∗ = (S, A, 𝐻, ℙ∗, 𝑑0, 𝑢∗) 𝜎(·) 𝑧 ∈ {0, 1}|The prompt and the prompt space. The distribution of initial state (prompt). The state, action, and observation. Episode length, e.g., the maximal number of tool calls. The true observation kernel. 𝜏 is a trajectory and 𝑦 is the completion part, i.e., we exclude 𝑥 from 𝜏. The true utility function associated with the BT model defined in Definition 1. The true model with observation kernel ℙ∗ and utility function 𝑢∗ 𝜎(𝑧) = 1/(1 + exp(−𝑧)) is the sigmoid function. Preference signal.<br><br>|
|---|---|
|𝜋 = [𝜋ℎ]ℎ𝐻=1 M = (S, A, 𝐻, ℙ, 𝑑0, 𝑢)<br><br>𝜋ref = [𝜋ref,ℎ]ℎ𝐻=1 𝐽(𝜋;M, 𝜋ref) 𝜂 𝑄M = [𝑄M,ℎ]ℎ𝐻=1 𝑉M = [𝑉M,ℎ]ℎ𝐻=1 𝜋M = [𝜋M,ℎ]ℎ𝐻=1<br><br>|The policy, which is parameterized by the LLM. One arbitrary environment with observation kernel ℙ and utility function 𝑢. One arbitrary reference policy. The KL-regularized target ((5)) with environment M and reference 𝜋ref. The coefficient of KL penalty, defined in (5).<br><br>The optimal 𝑄-values associated with 𝐽(𝜋;M, 𝜋ref), defined in (8). The optimal 𝑉-values associated with 𝐽(𝜋;M, 𝜋ref), defined in (9). The optimal policy associated with 𝐽(𝜋;M, 𝜋ref), defined in (9).|
|LM-DPO(·) LM-KTO(·)<br><br>|M-DPO loss, defined in (14). M-KTO loss, defined in (15).|
|𝐽(𝜋) 𝜋∗ = [𝜋∗<br><br>ℎ]ℎ𝐻=1 𝜋1𝑡 , 𝜋2𝑡 Reg(𝑇) U, P 𝐵<br><br>𝑢ˆ𝑡, ℙˆ𝑡 U𝑡, P𝑡 𝑐1, 𝑐2, 𝑐 𝜅 𝑑U<br><br>𝑑P, 𝜉(·) TV(·, ·)|The abbreviation of 𝐽(𝜋;M∗, 𝜋0), defined in (18). The optimal policy associated with 𝐽(𝜋). The main and exploration policy at round 𝑡 Regret over horizon 𝑇, defined in (19). Known sets such that 𝑢∗ ∈ U and ℙ∗ ∈ P Assuming 𝑢∗(𝑥, 𝑦) ∈ [0, 𝐵],∀(𝑥, 𝑦). MLE of 𝑢∗ and ℙ∗ at round 𝑡, defined in (20) and (21). Confidences sets of 𝑢∗ and ℙ∗ at round 𝑡, defined in (23). Absolute constants. 1/(2 + exp(−𝐵) + exp(𝐵)). Eluder coefficient from Definition 4. Generalized Eluder-type condition from Definition 5. Total variation distance between two distributions.<br><br>|

Table 6 | The table of notations used in this paper.

### B. Implementation Detail

Tools in Math Problem Solving. Following Gou et al. (2023b); Toshniwal et al. (2024), the LLM agent is allowed to call the python interpreter when it decodes a python code starting with “‘python and ending with “‘. For each step ℎ, to generate the observation 𝑜ℎ, we leverage the python package IPython, and run all the codes in the history one by one and treat each code snippet as a Jupyter cell. We only return the standard output or the error message from the last snippet. When there exists some bug in the code, we only return the error message which is typically less than 20 tokens as in Toshniwal et al. (2024). We notice that some works (e.g. Shao et al. (2024)) also returns the first and the last 50 tokens of the trackback information.

Data Generation. All the models are evaluated in the zero-shot setting. For all the data generation process, we adopt the following constraints: (1) for each turn, the model can generate up to 512

tokens; (2) the maximal number of steps is H=6; (3) the maximal number of generated token for each trajectory is 2048. When collecting new data for online iterative M-DPO, we set temperature to be 1.0 and decode without top-K or top-p sampling. For evaluation, greedy decoding is employed so that the results are generally comparable with previous works Gou et al. (2023b); Toshniwal et al. (2024). For evaluating the models with pass@n rate, we follow Toshniwal et al. (2024) to adopt a temperature of 0.7.

Data Annotation. For each prompt, we first divide the responses into the winning set 𝐺𝑤 and the losing set 𝐺𝑙 by checking the final answer. In practice, we observe that the model can memorize the final answer and output it even though the reasoning path itself is incorrect. To mitigate this issue, we include some heuristic filtering process. First, we delete all the trajectories in the winning set where the returned messages in the second last round indicate the code is with some bugs, but the models just ignore it and predict the ground-truth answer. Then, we delete the responses in both the winning set 𝐺𝑤 and losing set 𝐺𝑙 if they are longer than 2048 tokens. Finally, we randomly sample a trajectory from the 𝐺𝑤 and a trajectory from 𝐺𝑙 to construct a pair or to add them into the training set of KTO algorithm. For each iteration, we typically get 15K-20K samples because some of the prompts may not have any correct answer. We notice that it is possible to leverage AI feedback like Gemini (Team et al., 2023) or GPT4 (OpenAI, 2023) to further verify the correctness of the trajectory step by step or construct a PRM (Lightman et al., 2023; Wang et al., 2023a) to rank the trajectories, which we leave for future work.

Python Experiment Environment. We find that the evaluation can be influenced by the python environment, the precision (especially for the Gemma-1.1 models), and even the virtual machine we use. This does not affect the overall trend and conclusion because the magnitude of oscillation is relatively small compared to the overall improvement. For completeness, however, we specify some of the key package versions here. We use transformers 4.42.4, torch 2.3.0, sympy 1.2, antlr4-python3runtime 4.11.0, IPython 8.26.0 for all models. We evaluate the models using torch.float and use vllm 0.5.0.post1 for most the experiments except for Gemma-2 where vllm 0.5.1 is required. The inconsistency of vllm version is because Gemma-2 model was not released when we performed the main experiments of this project. We fix the python environment and machine for our evaluation throughout the experiment. For SFT, we use the open-source axolotl project with version 0.4.1 and for online iterative preference learning and RAFT, we use the code base from RLHF Workflow (Dong et al., 2024).

RAFT implementation. The data generation step is similar to the online iterative M-DPO training, except that we only keep the trajectories with correct final answer. For each prompt, we sample at most 𝑘 trajectories where we search 𝑘 ∈ {1, 3, 8} and use 𝑘 = 1 eventually because we do not see improvement by leveraging more data. We run the algorithm for three iterations in total. The training parameters are similar to the SFT stage, but we use a smaller batch size of 32 so that there are enough optimization steps. For Gemma models, we use a learning rate of 5e-6. For each training stage, we train the models for two epochs in total according to our parameter search. For Mistral model, we find that a smaller learning rate of 1e-6 and training for 1 epoch give us much better performance.

Prompt template. We do not tune the prompt though we do observe that the prompt engineering can further improve the performance. For all the experiments, we simply adopt the chat template of the models as in Table ??.

### C. Omitted Theoretical Proofs

##### C.1. Proof of Proposition 2

Proof of Proposition 2. For one policy 𝜋, starting with 𝑉M𝜋 ,𝐻+1 = 0, we recursively define its 𝑉-value and 𝑄-value functions on one model M = (S, A, 𝐻, ℙ, 𝑑0, 𝑢) and the reference policy 𝜋ref as

𝑢(𝑠𝐻, 𝑎𝐻), if ℎ = 𝐻, 𝔼𝑜

𝑄𝜋M,ℎ(𝑠ℎ, 𝑎ℎ) :=

ℎ∼ℙℎ(·|𝑠ℎ,𝑎ℎ)[𝑉M𝜋 ,ℎ+1(𝑠ℎ+1)], if ℎ ≤ 𝐻 − 1, 𝑉M𝜋 ,ℎ(𝑠ℎ) := 𝔼𝑎

ℎ∼𝜋ℎ(·|𝑠ℎ) 𝑄𝜋M,ℎ(𝑠ℎ, 𝑎ℎ) − 𝜂 · 𝐷KL 𝜋ℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) . It is noted that with the optimal policy 𝜋M, 𝑄M,ℎ = 𝑄𝜋M

M,ℎ. In the following discussions, we exclusively focus on the model M∗ = (S, A, 𝐻, ℙ∗, 𝑑0, 𝑢∗) with abbreviations 𝑄𝜋ℎ = 𝑄𝜋M∗,ℎ and 𝑉ℎ𝜋 = 𝑉M𝜋 ∗,ℎ.

M,ℎ and 𝑉M,ℎ = 𝑉𝜋M

For any comparator policy 𝜋, it holds that

𝐽(𝜋) − 𝐽(𝜋ˆ) = 𝔼𝑑0 𝑉1𝜋(𝑠1) − 𝑉ˆ1(𝑠1) − 𝔼𝑑0 𝑉1𝜋ˆ(𝑠1) − 𝑉ˆ1(𝑠1) , For any ℎ ∈ [𝐻], we can obtain that

ℎ−1,ℙ1:∗ ℎ−1 𝑉ℎ𝜋ˆ(𝑠ℎ) − 𝑉ˆℎ(𝑠ℎ) (=𝑎) 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 𝑉ℎ𝜋(𝑠ℎ) − 𝑉ˆℎ(𝑠ℎ) − 𝔼𝑑0,𝜋ˆ1:

𝔼𝑑0,𝜋1:

ℎ 𝑄𝜋ℎ(𝑠ℎ, 𝑎ℎ) − 𝜂𝐷KL 𝜋ℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) − 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 𝔼𝜋

𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝜂𝐷KL 𝜋 ˆℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) − 𝔼𝑑0,𝜋ˆ1:

ℎ−1,ℙ1:∗ ℎ−1 𝔼𝜋ˆ

ℎ

ℎ 𝑄𝜋ℎˆ(𝑠ℎ, 𝑎ℎ) − 𝜂𝐷KL(𝜋ˆℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ))

ℎ−1,ℙ1:∗ ℎ−1 𝔼𝜋ˆ

𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝜂𝐷KL(𝜋ˆℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ))

+ 𝔼𝑑0,𝜋ˆ1:

ℎ−1,ℙ1:∗ ℎ−1 𝔼𝜋ˆ

ℎ

ℎ,ℙ1:∗ ℎ−1 𝑄𝜋ℎˆ(𝑠ℎ, 𝑎ℎ) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ)

ℎ,ℙ1:∗ ℎ−1 𝑄𝜋ℎ(𝑠ℎ, 𝑎ℎ) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝔼𝑑0,𝜋ˆ1:

= 𝔼𝑑0,𝜋1:

𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝔼𝜋ˆ

𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ)

+ 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 𝔼𝜋

ℎ

ℎ

term (I)

ℎ−1,ℙ1:∗ ℎ−1 𝐷KL 𝜋 ˆℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) (=𝑏) 𝔼𝑑0,𝜋1:

− 𝜂 · 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 𝐷KL 𝜋ℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) + 𝜂 · 𝔼𝑑0,𝜋1:

ℎ,ℙ1:∗ ℎ−1 𝑄𝜋ℎˆ(𝑠ℎ, 𝑎ℎ) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝜂 · 𝔼𝑑0,𝜋1:

ℎ,ℙ1:∗ ℎ−1 𝑄𝜋ℎ(𝑠ℎ, 𝑎ℎ) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝔼𝑑0,𝜋ˆ1:

ℎ−1,ℙ1:∗ ℎ−1 [𝐷KL (𝜋ℎ(·|𝑠ℎ), 𝜋ˆℎ(·|𝑠ℎ))] .

In the above derivation, equation (a) is from the definitions of 𝑄𝜋 and 𝑉𝜋, and the relationship between 𝑄ˆ and 𝑉ˆ. The equation (b) is because

𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝔼𝜋ˆ

𝑄 ˆℎ(𝑠ℎ, 𝑎ℎ)

(term I) := 𝔼𝜋

ℎ

ℎ

𝜋ˆℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

𝜋ˆℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

log

log

= 𝜂 · 𝔼𝜋

− 𝜂 · 𝔼𝜋ˆ

ℎ

ℎ

= 𝜂 · 𝐷KL 𝜋ℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) − 𝜂 · 𝐷KL (𝜋ℎ(·|𝑠ℎ), 𝜋ˆℎ(·|𝑠ℎ)) − 𝜂 · 𝐷KL 𝜋 ˆℎ(·|𝑠ℎ), 𝜋ref,ℎ(·|𝑠ℎ) . where the second equation is from the relationship that

𝜋ˆℎ(𝑎ℎ|𝑠ℎ) 𝜋ref,ℎ(𝑎ℎ|𝑠ℎ)

− 𝜂 · log 𝑍ˆℎ(𝑠ℎ).

𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) = 𝜂 · log

Furthermore, if ℎ = 𝐻, we can obtain that

𝐻−1,ℙ1:∗ 𝐻−1 𝑉𝐻𝜋(𝑠𝐻) − 𝑉ˆ𝐻(𝑠𝐻) − 𝔼𝑑0,𝜋ˆ1:

𝐻−1,ℙ1:∗ 𝐻−1 𝑉𝐻𝜋ˆ(𝑠𝐻) − 𝑉ˆ𝐻(𝑠𝐻)

𝔼𝑑0,𝜋1:

𝐻,ℙ1:∗ 𝐻−1 𝑢∗(𝑠𝐻, 𝑎𝐻) − 𝑄ˆ𝐻(𝑠𝐻, 𝑎𝐻) − 𝜂 · 𝔼𝑑0,𝜋1:

𝐻,ℙ1:∗ 𝐻−1 𝑢∗(𝑠𝐻, 𝑎𝐻) − 𝑄ˆ𝐻(𝑠𝐻, 𝑎𝐻) − 𝔼𝑑0,𝜋ˆ1:

= 𝔼𝑑0,𝜋1:

𝐻−1,ℙ1:∗ 𝐻−1 [𝐷KL (𝜋𝐻(·|𝑠𝐻), 𝜋ˆ𝐻(·|𝑠𝐻))]

𝐻,ℙ1:∗ 𝐻−1 [𝑢∗(𝑠𝐻, 𝑎𝐻)]

𝐻,ℙ1:∗ 𝐻−1 [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑0,𝜋ˆ1:

= 𝔼𝑑0,𝜋1:

𝐻,ℙ1:∗ 𝐻 𝑉 ˆ𝐻+1(𝑠𝐻+1) − 𝑄ˆ𝐻(𝑠𝐻, 𝑎𝐻) − 𝜂 · 𝔼𝑑0,𝜋1:

𝐻,ℙ1:∗ 𝐻 𝑉 ˆ𝐻+1(𝑠𝐻+1) − 𝑄ˆ𝐻(𝑠𝐻, 𝑎𝐻) − 𝔼𝑑0,𝜋ˆ1:

+ 𝔼𝑑0,𝜋1:

𝐻−1,ℙ1:∗ 𝐻−1 [𝐷KL (𝜋𝐻(·|𝑠𝐻)||𝜋ˆ𝐻(·|𝑠𝐻))] , where the second equality leverages that 𝑉ˆ𝐻+1(𝑠𝐻+1) = 0; otherwise, for all ℎ ≤ 𝐻 − 1, it holds that 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 𝑉ℎ𝜋(𝑠ℎ) − 𝑉ˆℎ(𝑠ℎ) − 𝔼𝑑0,𝜋ˆ1:

ℎ−1,ℙ1:∗ ℎ−1 𝑉ℎ𝜋ˆ(𝑠ℎ) − 𝑉ˆℎ(𝑠ℎ)

ℎ,ℙ1:∗ ℎ−1 𝑄𝜋ℎˆ(𝑠ℎ, 𝑎ℎ) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝜂 · 𝔼𝑑0,𝜋1:

ℎ,ℙ1:∗ ℎ−1 𝑄𝜋ℎ(𝑠ℎ, 𝑎ℎ) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝔼𝑑0,𝜋ˆ1:

= 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 [𝐷KL (𝜋ℎ(·|𝑠ℎ)||𝜋ˆℎ(·|𝑠ℎ))]

ℎ,ℙ1:∗ ℎ 𝑉 ˆℎ+1(𝑠ℎ+1) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝜂 · 𝔼𝑑0,𝜋1:

ℎ,ℙ1:∗ ℎ 𝑉 ˆℎ+1(𝑠ℎ+1) − 𝑄ˆℎ(𝑠ℎ, 𝑎ℎ) − 𝔼𝑑0,𝜋ˆ1:

= 𝔼𝑑0,𝜋1:

ℎ−1,ℙ1:∗ ℎ−1 [𝐷KL (𝜋ℎ(·|𝑠ℎ)||𝜋ˆℎ(·|𝑠ℎ))]

ℎ,ℙ1:∗ ℎ 𝑉ℎ𝜋ˆ+1(𝑠ℎ+1) − 𝑉ˆℎ+1(𝑠ℎ+1) . The proposition can be obtained by iteratively using the above relationship for ℎ ∈ [𝐻]. □

ℎ,ℙ1:∗ ℎ 𝑉ℎ𝜋+1(𝑠ℎ+1) − 𝑉ˆℎ+1(𝑠ℎ+1) − 𝔼𝑑0,𝜋1:

+ 𝔼𝑑0,𝜋1:

##### C.2. Proof of Theorem 1

First, with the assumption 𝑢∗ ∈ U and ℙ∗ ∈ P, the following lemma demonstrates that U𝑡 and P𝑡 are valid confidence sets.

- Lemma 1 (Proposition B.1 from Liu et al. (2023a)). There exists an absolute constant 𝑐1 such that for any 𝛿 ∈ (0, 1], with probability at least 1 − 𝛿, for all 𝑡 ∈ [𝑇], 𝑢ˆ ∈ U, and ℙˆ ∈ P, it holds that

𝐿𝑡(𝑢ˆ) − 𝐿𝑡(𝑢∗) ≤ 𝑐1 log(|U|𝑇/𝛿), 𝐿𝑡(ℙˆ) − 𝐿𝑡(ℙ∗) ≤ 𝑐1 log(|P|𝑇/𝛿), which implies that 𝑢∗ ∈ U𝑡 and ℙ∗ ∈ P𝑡.

Then, we provide an additional lemma demonstrating the in-sample error of the MLE and optimistic estimators.

- Lemma 2. There exists an absolute constant 𝑐2 such that for any 𝛿 ∈ (0, 1], with probability at least 1 − 𝛿, for all 𝑡 ∈ [𝑇], we have

###### ∑︁

2

𝜎 𝑢 ˆ𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢ˆ𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝜎 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻)

###### ≤ 𝑐2 log(|U|𝑇/𝛿); ∑︁

𝑖<𝑡

2

𝜎 𝑢𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝜎 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻)

≤ 𝑐2 log(|U|𝑇/𝛿),

𝑖<𝑡

and for all 𝑡 ∈ [𝑇], ℎ ∈ [𝐻], we have

∑︁

∑︁

∑︁

2

TV {𝑑0, 𝜋𝑖𝑗, [ℙ1:∗ ℎ−1, ℙˆ𝑡,ℎ, ℙ∗ℎ+1:𝐻]}, {𝑑0, 𝜋𝑖𝑗, ℙ1:∗ 𝐻}

≤ 𝑐2 log(|P|𝐻𝑇/𝛿);

𝑗∈{1,2}

𝑖<𝑡

ℎ∈[𝐻]

###### ∑︁

∑︁

∑︁

2

TV {𝑑0, 𝜋𝑖𝑗, [ℙ1:∗ ℎ−1, ℙ𝑡,ℎ, ℙ∗ℎ+1:𝐻]}, {𝑑0, 𝜋𝑖𝑗, ℙ1:∗ 𝐻}

≤ 𝑐2 log(|P|𝐻𝑇/𝛿),

𝑗∈{1,2}

𝑖<𝑡

ℎ∈[𝐻]

where TV({𝑑0, 𝜋, ℙ}, {𝑑0, 𝜋′, ℙ′}) denotes the TV distance between the probability distributions over the trajectories induced by 𝑑0, 𝜋, ℙ and 𝑑0, 𝜋′, ℙ′.

Proof of Lemma 2. First, for 𝑢𝑡, we can obtain that with probability at least 1 − 𝛿, there exists an absolute constant 𝑐 such that for all 𝑡 ∈ [𝑇],

###### ∑︁

2

𝜎 𝑢𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝜎 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻)

𝑖<𝑡

###### ≤ 𝑐 ∑︁

𝑧𝑖 · 𝜎 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) + (1 − 𝑧𝑖) · 𝜎 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) 𝑧𝑖 · 𝜎 𝑢𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝑢𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) + (1 − 𝑧𝑖) · 𝜎 𝑢𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻)

log

+ log(|U|𝑇/𝛿)

𝑖<𝑡

= 𝑐 (𝐿𝑡(𝑢∗) − 𝐿𝑡( 𝑢𝑡) + log(|U|𝑇/𝛿)) ≤ 𝑐 (𝐿𝑡(𝑢∗) − 𝐿𝑡(𝑢ˆ𝑡) + 𝑐1 log(|U|𝑇/𝛿) + log(|U|𝑇/𝛿)) ≤ 𝑐2 log(|U|𝑇/𝛿).

where the first inequality is from Proposition B.2 from Liu et al. (2023a) and the second inequality uses Lemma 1. The result for 𝑢ˆ𝑡 can be similarly established.

Then, following similar steps, for ℙ𝑡, we can obtain that with probability at least 1−𝛿, there exists an absolute constant 𝑐 such that for all 𝑡 ∈ [𝑇],

###### ∑︁

∑︁

∑︁

2

TV {𝑑0, 𝜋𝑖𝑗, [ℙ1:∗ ℎ−1, ℙ𝑡,ℎ, ℙ∗ℎ+1:𝐻]}, {𝑑0, 𝜋𝑖𝑗, ℙ1:∗ 𝐻}

𝑗∈{1,2}

𝑖<𝑡

ℎ∈[𝐻]

𝑐 · ∑︁

###### ≤ ∑︁

∑︁

ℙ∗ℎ(𝑠𝑖,ℎ𝑗 +1|𝑠𝑖,ℎ𝑗 , 𝑎𝑖,ℎ𝑗 ) ℙ𝑡,ℎ(𝑠𝑖,ℎ𝑗 +1|𝑠𝑖,ℎ𝑗 , 𝑎𝑖,ℎ𝑗 )

log

+ log(|Pℎ|𝐻𝑇/𝛿)

𝑗∈{1,2}

𝑖<𝑡

ℎ∈[𝐻]

###### = 𝑐 · ∑︁

###### ∑︁

𝑗

###### 𝑖 (𝜏𝑖𝑗) ℙ𝜋

ℙ∗,𝜋

log

+ 2log(|P|𝐻𝑇/𝛿)

𝑗 𝑖

𝑡 (𝜏𝑖𝑗)

𝑗∈{1,2}

𝑖<𝑡

= 𝑐 · 𝐿𝑡(ℙ∗) − 𝐿𝑡( ℙ𝑡) + 2log(|P|𝐻𝑇/𝛿) ≤ 𝑐 · 𝐿𝑡(ℙ∗) − 𝐿𝑡(ℙˆ𝑡) + 𝑐1 log(|P|𝑇/𝛿) + 2log(|P|𝐻𝑇/𝛿) ≤ 𝑐2 log(|P|𝐻𝑇/𝛿).

The result for ℙˆ𝑡 can also be similarly established. □

Proof of Theorem 1. In the following proofs, we omit the KL term in the decomposition to ease the presentation. Then, with probability at least 1 − 𝛿, for all 𝑡 ∈ [𝑇], we can obtain that

𝐽(𝜋∗) − 𝐽(𝜋1𝑡 )

0,𝜋1𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑0,𝜋∗,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

= 𝔼𝑑0,𝜋∗,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

𝔼𝑑0,𝜋∗,ℙ∗ 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ) − ∑︁

###### + ∑︁

0,𝜋1𝑡 ,ℙ∗ 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ)

𝔼𝑑

ℎ∈[𝐻]

ℎ∈[𝐻]

0,𝜋2𝑡 , ℙ𝑡 [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 , ℙ𝑡 [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

0,𝜋2𝑡 , ℙ𝑡 [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 , ℙ𝑡 [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

≤ 𝔼𝑑

###### + ∑︁

ℎ∈[𝐻]

term (I)𝑡

0,𝜋2𝑡 , ℙ𝑡 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ) + ∑︁

𝔼𝑑

ℎ∈[𝐻]

term (II)𝑡

0,𝜋1𝑡 ,ℙ∗ ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ) − 𝑉ˆ𝑡,ℎ+1(𝑠ℎ+1)

𝔼𝑑

,

where the inequality is from the definition of 𝜋2𝑡 and the fact that (𝑢∗, ℙ∗) ∈ U𝑡 × P𝑡 from Lemma 1. We define the following terms:

- term (A)𝑡 := 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] ,

- term (B)𝑡 := 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] ,

- term (C)𝑡 := ∑︁ 𝑗∈{1,2}

∑︁

ℎ∈[𝐻]

𝔼𝑑

0,𝜋𝑡𝑗,ℙ∗ TV ℙ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ) ,

- term (D)𝑡 := ∑︁ 𝑗∈{1,2}

∑︁

ℎ∈[𝐻]

0,𝜋𝑡𝑗,ℙ∗ TV ℙ ˆ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ) .

𝔼𝑑

- For term (I)𝑡, we have that

term (I)𝑡 := 𝔼𝑑

0,𝜋2𝑡 , ℙ𝑡 [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 , ℙ𝑡 [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 , ℙ𝑡 [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 , ℙ𝑡 [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

= 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻)

+ 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

+ 𝔼𝑑

0,𝜋2𝑡 , ℙ𝑡 [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 , ℙ𝑡 [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)]

+ 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 , ℙ𝑡 [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 , ℙ𝑡 [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] ≤ 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻)

+ 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

+ 4𝐵 · TV {𝑑0, 𝜋1𝑡 , ℙ𝑡}, {𝑑0, 𝜋1𝑡 , ℙ∗} + 4𝐵 · TV {𝑑0, 𝜋2𝑡 , ℙ𝑡}, {𝑑0, 𝜋2𝑡 , ℙ} ≤ 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻)

term (A)𝑡

+ 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ 𝑢∗𝑡 (𝑠𝐻, 𝑎𝐻) − 𝔼𝑑

0,𝜋2𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢ˆ𝑡(𝑠𝐻, 𝑎𝐻)]

term (B)𝑡

+ 4𝐵 · ∑︁

𝑗∈{1,2}

∑︁

ℎ∈[𝐻]

𝔼𝑑0𝔼𝜋𝑗

𝑡,ℙ∗ TV ℙ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ)

term (C)𝑡

.

- For term (II)𝑡, we have that

term (II)𝑡 = ∑︁

ℎ∈[𝐻]

0,𝜋2𝑡 , ℙ𝑡 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ)

𝔼𝑑

###### + ∑︁

0,𝜋1𝑡 ,ℙ∗ ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ) − 𝑉ˆ𝑡,ℎ+1(𝑠ℎ+1)

𝔼𝑑

###### = ∑︁

ℎ∈[𝐻]

0,𝜋2𝑡 ,ℙ∗ 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ)

𝔼𝑑

###### + ∑︁

ℎ∈[𝐻]

0,𝜋2𝑡 , ℙ𝑡 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ) − ∑︁

𝔼𝑑

ℎ∈[𝐻]

0,𝜋2𝑡 ,ℙ∗ 𝑉 ˆ𝑡,ℎ+1(𝑠ℎ+1) − ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ)

𝔼𝑑

###### + ∑︁

ℎ∈[𝐻]

0,𝜋1𝑡 ,ℙ∗ ℙ ˆ𝑡,ℎ𝑉ˆ𝑡,ℎ+1 (𝑠ℎ, 𝑎ℎ) − 𝑉ˆ𝑡,ℎ+1(𝑠ℎ+1) ≤ 2𝐵 · ∑︁

𝔼𝑑

∑︁

ℎ∈[𝐻]

0,𝜋𝑡𝑗,ℙ∗ TV(ℙˆ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ)), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ)

𝔼𝑑

𝑗∈{1,2}

ℎ∈[𝐻]

+ 2𝐵𝐻 · TV({𝑑0, 𝜋2𝑡 , ℙ𝑡}, {𝑑0, 𝜋2𝑡 , ℙ∗}) ≤ 2𝐵 · ∑︁

∑︁

0,𝜋𝑡𝑗,ℙ∗ TV(ℙˆ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ)), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ)

𝔼𝑑

𝑗∈{1,2}

ℎ∈[𝐻]

term (D)𝑡

###### + 2𝐵𝐻 · ∑︁

∑︁

0,𝜋𝑡𝑗,ℙ∗ TV( ℙ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ)), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ)

.

𝔼𝑑

𝑗∈{1,2}

ℎ∈[𝐻]

term (C)𝑡

In the above derivations, we have repeatedly used similar relationships as follows:

TV({𝑑0, 𝜋2𝑡 , ℙ𝑡}, {𝑑0, 𝜋2𝑡 , ℙ∗}) ≤ ∑︁

0,𝜋2𝑡 ,ℙ∗ TV ℙ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ) ,

𝔼𝑑

ℎ∈[𝐻]

which can be derived as

TV({𝑑0, 𝜋2𝑡 , ℙ𝑡}, {𝑑0, 𝜋2𝑡 , ℙ∗}) ≤ ∑︁

TV {𝑑0, 𝜋2𝑡 , ℙ1:∗ ℎ−1, ℙ𝑡,ℎ:𝐻}, {𝑑0, 𝜋2𝑡 , ℙ1:∗ ℎ, ℙ𝑡,ℎ+1:𝐻}

###### = ∑︁

ℎ∈[𝐻]

0,𝜋2𝑡 ,ℙ∗ TV ℙ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ)} .

𝔼𝑑

ℎ∈[𝐻]

Then, we can obtain that ∑︁

𝐽(𝜋∗) − 𝐽(𝜋ˆ1𝑡 ) ≤ ∑︁ 𝑡∈[𝑇]

term (A)𝑡 + ∑︁ 𝑡∈[𝑇]

term (B)𝑡 + (4𝐵 + 2𝐵𝐻) ∑︁ 𝑡∈[𝑇]

term (C)𝑡 + 2𝐵 ∑︁ 𝑡∈[𝑇]

term (D)𝑡.

𝑡∈[𝑇]

Then, we control the sum of each individual term in the following. First, for term (A)𝑡, with probability at least 1 − 𝛿, we have that

∑︁

term (A)𝑡

###### = ∑︁

𝑡∈[𝑇]

0,𝜋2𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [𝑢∗(𝑠𝐻, 𝑎𝐻)]

0,𝜋2𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

0,𝜋1𝑡 ,ℙ∗ [ 𝑢𝑡(𝑠𝐻, 𝑎𝐻)] − 𝔼𝑑

𝔼𝑑

≤ ∑︁

𝑡∈[𝑇]

𝑢𝑡(𝑠𝑡,𝐻2 , 𝑎2𝑡,𝐻) − 𝑢𝑡(𝑠𝑡,𝐻1 , 𝑎1𝑡,𝐻) − 𝑢∗(𝑠𝑡,𝐻2 , 𝑎2𝑡,𝐻) − 𝑢∗(𝑠𝑡,𝐻1 , 𝑎1𝑡,𝐻) + 𝑂(𝐵√︁𝑇 log(1/𝛿))

𝑡∈[𝑇]

∑︁𝑇

∑︁𝑡−1

+ 𝑂(𝐵√︁𝑇 log(1/𝛿))

2

𝑢𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻)

1 +

≤ 𝑑U

𝑡=2

𝑖=1

∑︁𝑇

∑︁𝑡−1

+ 𝑂(𝐵√︁𝑇 log(1/𝛿))

2

𝜎 𝑢𝑡(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢𝑡(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻) − 𝜎 𝑢∗(𝑠2𝑖,𝐻, 𝑎2𝑖,𝐻) − 𝑢∗(𝑠1𝑖,𝐻, 𝑎1𝑖,𝐻)

1 + 𝜅−2

≤ 𝑑U

𝑡=2

𝑖=1

≲ 𝜅−1𝐵√︁𝑑U𝑇 log(|U|𝑇/𝛿),

where the first inequality is from the Hoeffding inequality, the second inequality uses the Eluder coefficient 𝑑U := EC(1,U − U,𝑇) from Definition 4, the third inequality leverages the mean value theorem with 𝜅 := 1/(2+exp(−𝐵)+exp(𝐵)) representing the minimum derivative of 𝜎(·) in the regime of [0, 𝐵], and the last inequality incorporates Lemma 2. A similar result can be obtained for term (B)𝑡.

For term (C)𝑡, we have that

∑︁

term (C)𝑡 = ∑︁

∑︁

∑︁

0,𝜋𝑡𝑗,ℙ∗ TV ℙ𝑡,ℎ(·|𝑠ℎ, 𝑎ℎ), ℙ∗ℎ(·|𝑠ℎ, 𝑎ℎ)

𝔼𝑑

𝑗∈{1,2}

= ∑︁

∑︁

∑︁

𝑡∈[𝑇]

𝑡∈[𝑇]

ℎ∈[𝐻]

TV {𝑑0, 𝜋𝑡𝑗, [ℙ1:∗ ℎ−1, ℙ𝑡,ℎ, ℙ∗ℎ+1:𝐻]}, {𝑑0, 𝜋𝑡𝑗, ℙ1:∗ 𝐻}

𝑗∈{1,2}

𝑡∈[𝑇]

ℎ∈[𝐻]

≤ 2𝐻 · 𝜉(𝑑P,𝑇, 𝑐2 log(|P|𝐻𝑇/𝛿)),

where the last step is from the generalized Eluder-type condition in Definition 5 and Lemma 2. A similar result can be obtained for term (D)𝑡.

Finally, we obtain that Reg(𝑇) ≲𝜅−1𝐵√︁𝑑U𝑇 log(|U|𝑇/𝛿) + 𝐵2𝐻𝜉(𝑑P,𝑇, 𝑐2 log(|P|𝐻𝑇/𝛿)

− 𝜂 · ∑︁

𝔼𝑑0,𝜋∗,ℙ∗ 𝐷KL(𝜋∗ℎ(·|𝑠ℎ), 𝜋1𝑡,ℎ(·|𝑠ℎ)) ,

ℎ∈[𝐻]

which concludes the proof. □

### D. Technical Lemmas

- Lemma 3 (Solution of KL-regularized Optimization (Proposition 7.16 and Theorem 15.3 of Zhang

(2023))). Given a loss functional with respect to 𝑝(·|𝑥), written as

1

1

𝔼𝑤∼𝑝(·) − 𝑈(𝑤) + 𝜂𝐷KL 𝑝(·), 𝑝0(·) = 𝜂𝐷KL 𝑝(·), 𝑝0(·) exp

𝑈(·) − 𝜂 · log𝔼𝑤∼𝑝0(·) exp

𝑈(𝑤)

,

𝜂

𝜂

𝐶𝑟

𝑝0(𝑤) exp 1𝜂𝑈(𝑤) , also known as Gibbs distribution.

where the minimizer of the loss functional is 𝑝∗(𝑤) = 𝐶1

𝑟

- Definition 4 (Eluder Coefficient, Definition 17.17 in Zhang (2023)). Given a function class F, its Eluder coefficient EC(𝜆, F,𝑇) is defined as the smallest number 𝑑 so that for any sequence {𝑥𝑡 : 𝑡 ∈ [𝑇]} and { 𝑓𝑡 : 𝑡 ∈ [𝑇]} ∈ F,

∑︁𝑇

∑︁𝑇

∑︁𝑡−1

###### ( 𝑓𝑡(𝑥𝑖) − 𝑓∗(𝑥𝑖))2 .

| 𝑓𝑡(𝑥𝑡) − 𝑓∗(𝑥𝑡)| ≤ 𝑑

𝜆 +

𝑡=2

𝑡=2

𝑖=1

- Definition 5 (Generalized Eluder-type Condition, Condition 3.1 in Liu et al. (2023a)). There exists a

real number 𝑑P ∈ ℝ+ and a function 𝜉 such that for any (𝑇, Δ) ∈ ℕ × ℝ+, transitions {ℙ𝑡′ : 𝑡 ∈ [𝑇]} and policies {𝜋𝑡 : 𝑡 ∈ [𝑇]}, we have

###### ∀𝑡 ∈ [𝑇], ∑︁

TV({𝑑0, ℙ′𝑖, 𝜋𝑖}, {𝑑0, ℙ, 𝜋𝑖})2 ≤ Δ ⇒ ∑︁ 𝑡∈[𝑇]

TV({𝑑0, ℙ𝑡′, 𝜋𝑡}, {𝑑0, ℙ, 𝜋𝑡}) ≤ 𝜉(𝑑P,𝑇, Δ).

𝑖<𝑡

