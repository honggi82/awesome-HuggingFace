# arXiv:2503.15478v1[cs.LG]19Mar2025

## SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks

Yifei Zhou1,2,†, Song Jiang1, Yuandong Tian1, Jason Weston1, Sergey Levine2, Sainbayar Sukhbaatar1,∗, Xian Li1,∗

1FAIR at Meta, 2UC Berkeley †Work done at Meta, ∗Equal advising

Large language model (LLM) agents need to perform multi-turn interactions in real-world tasks. However, existing multi-turn RL algorithms for optimizing LLM agents fail to perform effective credit assignment over multiple turns while leveraging the generalization capabilities of LLMs and it remains unclear how to develop such algorithms. To study this, we first introduce a new benchmark, ColBench, where an LLM agent interacts with a human collaborator over multiple turns to solve realistic tasks in backend programming and frontend design. Building on this benchmark, we propose a novel RL algorithm, SWEET-RL (RL with Step-WisE Evaluation from Training-time information), that uses a carefully designed optimization objective to train a critic model with access to additional training-time information. The critic provides step-level rewards for improving the policy model. Our experiments demonstrate that SWEET-RL achieves a 6% absolute improvement in success and win rates on ColBench compared to other state-of-the-art multi-turn RL algorithms, enabling Llama-3.1-8B to match or exceed the performance of GPT4-o in realistic collaborative content creation.

Date: March 20, 2025 Correspondence: Yifei Zhou at yifei_zhou@berkeley Code: https://github.com/facebookresearch/sweet_rl Data: https://huggingface.co/datasets/facebook/collaborative_agent_bench

1 Introduction

🤝CollaborativeAgentBench🤝

[Figure 1]

[Figure 2]

SWEET-RL

Backend Programming

Chosen/Rejected trajectories

Frontend Design

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Ref Ref

Training-TimeHistory Information

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### I would like a website for my shop. Would love a green palette.

Write a python function that processes events and interactions in Renesmee's life and return a summary of these interactions.

[Figure 13]

[Figure 14]

SWEET Advantage from BT objective

[Figure 15]

[Figure 16]

How do you like this design?

[Figure 17]

What is the interaction log, is it a

Would like to have a search bar and featured items.

[Figure 18]

Chosen/Rejected actions

[Figure 19]

The interaction log is a list of dictionary.

[Figure 20]

. . . History

[Figure 21]

. . .

[Figure 22]

[Figure 23]

I still don’t know the edge case

. . .

[Figure 24]

It seems that this is the web page that you look for

[Figure 25]

Now here is my solution: ……

……

Policy from RLHF optimization

[Figure 26]

[Figure 27]

Unit Tests Image Similarity

Figure 1 (Left) Overview of our ColBench including Backend Programming and Frontend Design that supports cheap and reliable evaluation of multi-turn RL algorithms for agents in realistic settings. (Right) The high-level motivation behind SWEET-RL that uses additional training-time information along with appropriate Bradley-Terry (BT) objective to perform effective credit assignment.

Large language models (LLMs) have the potential of serving as decision-making agents capable of executing complex tasks autonomously, such as navigating the web and controling devices (Zhou et al., 2024a,b; Xu

et al., 2024; Gur et al., 2021; Bai et al., 2024), writing and maintaining code bases (Jimenez et al., 2024), and serving as personal assistants (Xie et al., 2024a; Jiang et al., 2024), thanks to advances in the reasoning and generalization capabilities of LLMs (OpenAI, 2024; GeminiTeam, 2024; Llama3Team, 2024). However, to achieve the best performance on tasks that involve making a sequence of decisions, the agent needs to directly optimize for the multi-turn objective of interest such as success rates, which is more challenging than only imitating the most probable action at each turn as learnt in the next-token-prediction pre-training objective.

While a natural approach to directly optimize for multi-turn objective is to apply successful algorithms from single-turn Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022; Ziegler et al., 2020; Christiano et al., 2023), such as RAFT (Dong et al., 2023), DPO (Rafailov et al., 2024b) and PPO (Schulman et al., 2017), these methods do not perform explicit credit assignment across turns. Consequently, they may suffer from high variance and poor sample complexity due to the long-horizon nature of complex sequential decision-making tasks (Zhou et al., 2024c). Another alternative is to apply value function learning methods, such as TD-learning (Mnih et al., 2013; Zhou et al., 2024c; Snell et al., 2023). Yet that requires training a new task-specific value head on top of LLM representations, which may not generalize well with limited fine-tuning data (Figure 3). As a result, it is still unclear what is the most effective multi-turn RL algorithm that fully takes advantage of the reasoning capabilities of LLMs for training general, capable, and goal-directed agents.

To begin to address this challenge, we first notice that a benchmark for validating multi-turn RL algorithms for realistic LLM agents is required to meet the following three criteria: 1) sufficient task diversity for RL training without overfitting, 2) sufficient task complexity that challenges the reasoning and generalization capability of agents, and 3) minimum engineering overhead for fast research prototyping. However, as shown in Table 1, none of the existing benchmarks satisfy all of the three necessary criteria. To address this gap, our first contribution in this work is to develop a benchmark, Collaborative Agent Benchmark (ColBench), designed to support research on multi-turn RL algorithms for realistic LLM agent scenarios. Our benchmark focuses on the realistic domain of artifact creation, where the goal for agents is to interact with humans to produce a final artifact (e.g., code, web pages, or slides) that fulfills human expectations. To solve such tasks, the agent must act to understand the intent of the human collaborator and reason with the missing pieces, as only limited information is provided in each turn for complex and potentially multi-modal artifacts like code and web pages. To facilitate rapid iteration and cost-effective evaluation, we employ LLMs as human “simulators”, where we crucially also provide the ground-truth artifacts to ensure faithful simulations in their responses. For reliable evaluations, we have developed a series of functional evaluators that measure the similarity between the agent-produced artifact and the ground-truth. Examples of tasks in ColBench are shown in Figure 1(left) and full trajectories in Appendix E.

For a multi-turn RL algorithm to perform effective credit assignments in such LLM agent settings, it needs to incorporate solutions to the following realistic challenges. Firstly, the agent is acting in a partially observable environment where some of the important task-relevant information is not directly revealed to the agent. In such cases, the agent needs to be properly awarded for information-seeking behaviors in the stochastic environment. Moreover, there is only limited amount of data available during fine-tuning for a large diverse set of tasks that may show up at test time. Therefore, the learning objective of the algorithm needs to effectively take advantage of the reasoning capabilities of LLMs for the best generalization performance. For the first challenge, we observe that additional training-time information, such as the final outcome and the reference solution, may be available during training. The knowledge of additional training-time information can offer a shortcut for credit assignments for an agent performing information-seeking behaviors without such knowledge. A natural approach to leveraging this training-time information is to train a value function that predicts the expected utility of each action as a scalar value. However, this introduces a fine-tuning objective that significantly differs from the next-token prediction pre-training objective of LLMs, which leads to inferior reasoning and generalization performance (Figure 3).

With these observations, the second contribution of this work is an easy-to-implement yet highly effective RL algorithm, SWEET-RL (RL with Step-WisE Evaluation from Training-Time Information) as depicted in Figure 1(right). SWEET-RL improves credit assignments by providing the critic with training-time information that is inaccessible to the actor. Our novel turn-level critic takes advantage of this asymmetric observation spaces for the critic and actor. Furthermore, we propose directly learning the advantage function, which characterizes the effectiveness of each action at the current state, avoiding the need of first training a value function that predicts the expected utility of the current state and action. Finally, we also propose

Table1 ComparisonsbetweenColBenchandexistingbenchmarksformulti-turnLLMagents. As shown in the table, no existing benchmarks satisfy all of the three criterions necessary for developing efficient RL algorithms for fine-tuning LLM agents: 1) sufficient task diversity for RL training without overfitting, 2) sufficient task complexity that challenges the reasoning and generalization capability of agents, and 3) minimum engineering overhead for fast research prototyping.

RL Complex Min Training Reasoning Overhead

Web/Device-Control(Zhou et al., 2024a;AgentsXie et al., 2024b) Yes Yes No SWEBench(Jimenez et al., 2024; Pan et al., 2024) Yes Yes No Travel Planner (Xie et al., 2024a) No Yes Yes LLF Benchmark (Cheng et al., 2023) No No Yes AgentBench (Liu et al., 2023) No Yes Yes Mint (Wang et al., 2024b) No Yes Yes Dialop (Lin et al., 2024) No Yes Yes LMRL Gym (Abdulhai et al., 2023) Yes No Yes RL4VLM (Zhai et al., 2024) Yes No Yes ColBench (ours) Yes Yes Yes

parameterizing the advantage function by the mean log probability of the action at each turn and training this advantage function through the Bradley-Terry objective at the trajectory level. We find such an objective aligns better with pre-trained LLMs compared to the common practice of training a value head on top of the hidden states of LLMs, leading to superior generalization results. In our experiments, we find that the use of asymmetric information during training and appropriate learning objectives result in a superior multi-turn agent on both realistic Backend Programming and Frontend Design tasks from ColBench, with 6% absolute success and win rates gains compared to other SOTA algorithms. As a result, the performance of Llama-3.1-8B (Llama3Team, 2024) can match or even surpass the performance of SOTA proprietary models including GPT-4o and o1-mini (OpenAI, 2024).

- 2 Related Work

Benchmarks for LLM Agents. While many recent benchmarks have been proposed to evaluate the capabilities of LLM agents in various settings, such as software engineering (Jimenez et al., 2024; Liu et al., 2023), web navigation (Zhou et al., 2024a; Koh et al., 2024; Deng et al., 2023; Yao et al., 2023), device control (Rawles et al., 2023, 2024; Xie et al., 2024b), and travel planning (Xie et al., 2024a), most of them tend to focus on evaluation of state-of-the-art generalist LLMs without providing a research-friendly interactive environment and a set of training tasks to study multi-turn RL algorithm. While LMRL Gym (Abdulhai et al., 2023) and RL4VLM (Zhai et al., 2024) offers this flexibility for comparing different multi-turn RL algorithms, the task settings focus on narrower domains and do not require the model to have strong reasoning capabilities. As shown in Table 1, there is no existing LLM agent benchmark that provides the flexibility for testing multi-turn RL algorithms on reasoning-intensive tasks with minimum engineering overhead. In contrast, ColBench is the first benchmark designed to support research efforts in multi-turn RL algorithms on reasoning-intensive tasks, focusing on the realistic domain of artifact creation with reliable functional verifiers.

Multi-turn RL algorithms for LLM Agents. Unlike single-turn scenarios such as single-turn preference optimization (Christiano et al., 2023; Ziegler et al., 2020; Casper et al., 2023; Xu et al., 2023) where it suffices for LLMs to produce a single response without further interactions with the environment, multi-turn RL (Zhou et al., 2024c; Kumar et al., 2024; Abdulhai et al., 2023) captures realistic agent scenarios where LLM agents need to make a sequence of actions to complete the task, such as operating a unix terminal (Liu et al., 2023) and navigating through the web (Zhou et al., 2024a). While some early works directly applied successful methods from single-turn RL, such as REINFORCE (Sutton et al., 1999; Wu and Hu, 2018), DPO (Xiong et al., 2024; Song et al., 2024), and PPO (Schulman et al., 2017; Szot et al., 2024), they often suffer from high variance when the horizon gets longer, resulting in poor performance. While recent works have applied more advanced techniques from the deep RL literature such as Bellman bootstrapping (Zhou et al., 2024c; Snell et al., 2023) and Path Consistency (Wang et al., 2024a; Liu et al., 2024; Nachum et al., 2017) to reduce

long-horizon variance, our work makes an important advancement to take advantage of the oft-neglected additional training-time information and make corresponding adjustments to the optimization objective for improved assignment. Finally, while some prior works apply an asymmetric actor-critic structure to perform sim-to-real transfer in robotics where the critic observes the latent state and the actor observes RGB inputs (Pinto et al., 2017; Wilson and Hermans, 2020; Salter et al., 2019), less has been studied in terms of how such techniques can be applied in reasoning-intensive LLM tasks.

Process reward models. The use of a step-wise critic resembles the notion of a “process reward model” (PRM) in the reasoning literature (Lightman et al., 2023; Uesato et al., 2022). PRMs evaluate the “correctness” of each reasoning step and can be trained from automated supervision (Luo et al., 2024; Setlur et al., 2024; Yuan et al., 2024; Hwang et al., 2024) without costly human-annotated process labels. Once a PRM is trained, it can be used for searching with more test-time compute (Snell et al., 2024; Yuan et al., 2024) or accelerating the explorations in on-policy RL (Setlur et al., 2024; Shao et al., 2024; Lin et al., 2025) (i.e. when the policy is trained on the online collected trajectories by itself). In contrast, in our work, the step-wise critic is mainly used to perform credit assignment as an intermediate “reward proxy” to directly optimize the policy without the need to collect additional interaction data. This benefit is important in LLM agent tasks where collecting on-policy data involves expensive interactions with an external environment.

#### 3 Collaborative Agent Benchmark (ColBench)

In this section, we first outline the foundational design principles of our ColBench to address challenges in developing multi-turn RL algorithms for LLM agents, followed by a detailed explanation of the two specific tasks: Backend Programming and Frontend Design.

- 3.1 Design Principles

- (1) Sufficient task complexity that challenges the rea- soning and generalization capability of agents. As the ultimate goal is to enable the LLM agents to complete tasks on our behalf in the real world, where the agents need to address complicated challenges in out-of-distribution scenarios, it is essential that the benchmark reflects such realistic reasoning and generalization challenges. To achieve this, ColBench is designed to align with realistic artifact creation tasks, where the objective of collaboration is to produce tangible outputs such as code or web pages. In order to do well, it is necessary for agents to dive deep into the structure of the code and nuanced differences in visual design for unseen requests and a potentially sub-optimal collaborator.
- (2) Minimum overhead for fast research prototyping. To achieve this, we ground each collaboration task with the goal to re-create the exact same product as the reference artifact. In this way, the human collaborator can be easily simulated by an LLM with access to the reference artifact to faithfully answer the uncertainties from the LLM agent. Furthermore, the presence of a reference artifact allows ColBench to employ objective and functional evaluation metrics that assess the similarity between the final collaborative product and the reference artifact. As a result, the only requirement needed for setting up ColBench is API access for some LLM calls and some Python packages for running code and rendering HTML web pages. Although in real-world scenarios the human collaborator might only have a general idea of the desired final product, having access to a clear reference artifact is a reasonable assumption that significantly enhances reliability.
- (3) Sufficient task diversity for RL training without overfitting. As the best LLMs are often trained on a huge amount of data, it is essential that our benchmark contains enough tasks to understand the scalability of different multi-turn RL algorithms while at the same time ensuring the reliability of simulation and evaluation. Therefore, we designed ColBench to be highly scalable, with more than 10k different procedurally-generated tasks, which also can be easily expanded as needed by incorporating more existing artifacts such as code and web pages. The difficulty of ColBench can be easily adjusted by simply creating more collaboration tasks with more complicated code bases and web pages. Next, we will describe detailed setups of ColBench, including Backend Programming and Frontend Design.

- 3.2 Backend Programming Collaborations

Task description. In this task, the agent is required to collaborate with the human simulator to write a custom Python function (up to 50 lines). In the beginning of the collaboration, the agent is provided with a high-level

description and the signature of the function. However, many concrete details, such as what conditions should be considered and what to do at edge cases, are not provided, and the agent has to reason and decide what clarifications are needed from the human simulator. The human simulator will provide a brief explanation in natural language to each clarification question based on the reference code visible only to the human simulator, but it will not write code. The interactions are limited to 10 back-and-forth rounds between the agent and the human simulator. The interaction ends either when the agent has decided to give the final solution or the maximum number of rounds has been reached. The success of the agent is evaluated by 10 hidden unit tests for each function for 0/1 reward only at the end of each collaboration.

Data generation. Python functions, high-level descriptions, and unit tests are generated by prompting Llama-

- 3.1-70B-Instruct (Llama3Team, 2024) to extract a python function as inspired by an Internet excerpt from DCLM (Li et al., 2024). Only the tasks where the generated python functions can pass their corresponding unit tests are kept. We generate 10k such tasks in the train set and 1k tasks in the test set where tasks in the test set are manually inspected by authors. 15k offline train trajectories are generated by zero-shot prompting Llama-3.1-8B-Instruct as agent and Llama-3.1-70B-Instruct as human simulator.

- 3.3 Frontend Design Collaborations

Task description. In this task, the agent is required to collaborate with the human simulator to design a web page by writing an HTML snippet (around 100 lines). At the beginning of the collaboration, the agent is provided with a high-level description of the web page. Again, many concrete details such as the layout and color palette of the web page are missing and only visible to the human simulator. At each round, the agent has a chance to write an HTML solution and it will be rendered by the web browser. The human simulator will be able to examine the rendered web page from the agent and the reference web page, then describe their differences to the agent. Similar to the backend programming collaboration, the interaction ends either when the agent has decided to give the final solution or the maximum number of 10 rounds has been reached. The performance of the agent is evaluated by the cosine similarity of CLIP (Radford et al., 2021) embeddings between the agent solution and reference web page, the best metric found in prior works (Si et al., 2024) for frontend design. It serves as a reward within the range of 0 to 1 only at the end of the collaboration.

Data generation. The tasks containing reference web pages and high-level descriptions are from WebSight (Laurençon et al., 2024). We include 10k such tasks for training and 500 for the test set. The test set tasks are manually inspected by the authors. We generate 6k offline train trajectories by zero-shot prompting Llama-3.1-8B as agent and Qwen2-VL-72B (Yang et al., 2024) as human simulator.

- 4 SWEET-RL: RL with Step-WisE Evaluation from Training-Time Information

To introduce our method, we will begin by first defining the terminology for multi-turn RL on ColBench. Then, we will motivate the two-stage training procedure of SWEET-RL for first training a step-wise critic with additional training-time information and using it as a per-step reward model to train the actor with careful algorithmic choices. An overview of the two-stage training procedure is presented in Figure 2.

Step-WisE Evaluation from Training-Time Information

Policy Improvement

Probability Distributions

Probability Distributions

a1t aLt

ot

τ+,τ−

−log( )/L

Aθ(ot,at,c) = log( )/L

× ×

× ×

[BOS] …… [EOS]

[Figure 28]

🔥 Actor πϕ

[Figure 29]

Chosen/Rejected trajectories

c

a1t aLt

a1t aLt

[Figure 30]

[Figure 31]

[Figure 32]

…… [EOS]

[BOS]

. . .

r(τ+) > r(τ−)

[BOS] …… [EOS]

[BOS] …… [EOS]

[Figure 33]

at

ot

[Figure 34]

Step t

❄ Reference πref

🔥 Critic πθ

[Figure 35]

[Figure 36]

Chosen/Rejected actions

I still don’t know the edge case

Aθ(ot,at+,c) > Aθ(ot,at−,c)

Bradley-Terry Optimization RLHF Optimization

…… [EOS]

…… [EOS]

[BOS]

[BOS]

……

at

at

ot

ot

c

c

###### Figure 2 textbfAn overview of the training procedure of SWEET-RL. At a high level, we first apply Bradley-Terry objective to directly train a step-wise advantage function with access to additional training-time information. Once the advantage function is trained, we perform policy improvement by using the advantage function as a reward model for each turn.

- 4.1 Problem Setup We frame the problem of collaboration between humans and agents as a finite-horizon partially observable Markov decision process (POMDP) M = {O,C,A,T ,µ1,R,N}. Here, O and C are the observable and hidden parts of the state space. In the beginning of each episode an initial instruction o1 ∈ O and some hidden training-time information c ∈ C (e.g., the reference solution) are drawn from the initial state distribution µ1. The hidden training-time information c remains unchanged during the episode.

At the t-th turn, the observation ot for the agent includes all the interaction history, and the agent needs to take an action at ∈ A by outputting a response consisting of tokens a1:t L. After taking an action, the user will respond to the agent, and the new state for the agent (represented by the transition function T ) is derived by appending the latest interaction to the interaction history. At each step, the agent has a chance to receive a scalar reward r(ot,at,c) ∈ R. The episode ends either when the agent outputs a termination token or when a maximum number of t = N rounds of interactions has been reached. The objective of RL is to train a policy that can generate token sequences maximizing the cumulative rewards Nt=1 r(ot,at,c) throughout a rollout (we assume no reward decay for simplicity). We consider the offline setting of learning from a past interaction dataset as online interactions with humans may be costly to obtain.

The Q-function for a policy π represents the expected cumulative reward of a specific action at the current step, followed by adherence to policy π: Qπ(ot,at,c) = Eπ Nt′=t r(ot′,at′,c) . The value function of π, V π(ot,c), is defined as the expected Q-value, Ea

t∼π[Qπ(ot,at,c)], where actions at are sampled from π. The advantage function Aπ(ot,at,c) indicates the relative benefit of taking action at in state (ot,c) and is calculated as the difference between the Q-value and the state’s value under the policy: Aπ(ot,at,c) = Qπ(ot,at,c) − V π(ot,c). We directly model the turn-wise advantage function A using parameters θ, and use that advantage Aθ to train the policy parameterized by ϕ to generate tokens a1:t L within each turn.

- 4.2 Learning turn-wise advantage functions

To perform explicit credit assignments in reasoning-intensive tasks, some prior works have explored learning an explicit value function first and derive the advantage of each individual action from the learnt value function (Bai et al., 2024; Zhou et al., 2024c; Snell et al., 2023). However, in our experiments we found that such value functions do not generalize well when only a limited number of samples for fine-tuning are available as shown in subsection 5.4. We hypothesize this is because learning an accurate value function in reasoning-intensive tasks is itself a hard task and does not effectively take advantage of the reasoning and generalization capability of pre-trained LLMs.

Since the ultimate goal of performing credit assignment is to derive the advantages for each action which may be a easier task for LLMs compared to estimating the expected future returns, we propose to directly learn the advantage function for each turn-wise action at. Inspired by the success of preference optimization in funetuning LLMs (Christiano et al., 2023; Ziegler et al., 2020), we propose to train the the turn-wise advantage function from preference pairs of trajectories. Given two trajectories under the same task with additional training-time information c, we label them chosen τ+ and rejected τ− judged by their cumulative rewards. This allows us employ the Bradley-Terry objective (Bradley and Terry, 1952; Rafailov et al., 2024a) for fine-tuning

JBT = −log σ

t

βr(o+t ,a+t ,c) −

t

βr(o−t ,a−t ,c) , (1)

where o+t ,a+t and o−t ,a−t are from τ+ and τ− respectively, and β is a hyperparameter. We can rewrite this objective using the advantage function:

JA(θ) = −log σ

t

βAθ(o+t ,a+t ,c) −

t

βAθ(o−t ,a−t ,c) . (2)

Intuitively, similar to the objective of single-turn RLHF to learn a high reward for each chosen response and a low reward for each rejected response, the effect of Equation 2 is to increase the advantage for each action in the chosen trajectory and lower the advantage for each action in the rejected trajectory. For

completeness, we provide a theoretical derivation in Appendix B. To further align the learning objective with next-token-prediction pre-training, we parameterize the advantage function by re-purposing the existing language model head of the LLM:

L

πθ(alt|ot,a1:t l−1,c) πref(alt|ot,a1:t l−1,c)

1 L

(3)

log

Aθ(ot,at,h) =

l=1

where πθ is the LLM model that we train to act as the advantage function, while πref is a frozen initial seed model. We find it important to include L1 to normalize the advantage by the length of the response to stabilize training.

- 4.3 Optimizing the agent by turn-wise advantage

Our key observation is that while our final policy πϕ cannot condition on the hidden information h, such information is available during training time. Since the advantage LLM πθ will only be used during training, it can take c as input as in Equation 3. Intuitively, many realistic problems such as collaboration and math reasoning have some hidden training-time information like reference solutions. If the turn-wise advantage function has access to such training-time information, it should be in a better position to judge whether the action taken by the policy is on the right track.

Therefore, we provide additional training-time information c to the turn-wise advantage function while only the interaction history ot is provided to the policy, resulting in an asymmetric actor-critic structure. In principle, any successful algorithm from the RLHF literature can be used to optimize the per-turn policy πϕ by treating the interaction histories as prompts and the turn-wise advantage function Aθ as the reward model. In this stage of training the policy, no interaction from human collaborators is needed.

For simplicity we choose to use DPO (Rafailov et al., 2024b) for training. For each turn t we first sample candidate actions from the current policy given interaction history ot, and rank them by the learnt turn-wise advantage function to obtain chosen and rejected actions. We then optimize the policy for each turn using the standard DPO loss:

log πϕ(a−|ot) log πref(a−|ot)

log πϕ(a+|ot) log πref(a+|ot) − β′

Jπ(ϕ) = −log σ β′

. (4)

In practice, for each turn we sample 16 candidate actions and take random actions from top-50% quantile as chosen and from the bottom-50% quantile as rejected.

#### 5 Experiments

The purpose of our experiments is to validate the effectiveness of SWEET-RL as a multi-turn RL algorithm that trains LLM agents for complex collaborative tasks. Specifically, they are designed to answer the following questions: (1) How do existing generalist models and multi-turn RL algorithms perform for collaborative tasks on ColBench? (2) How does SWEET-RL’s performance compare with other SOTA multi-turn RL algorithms for training LLM agents on reasoning-heavy tasks? (3) How does the use of asymmetric information help with credit assignments? (4) What are the best algorithmic choices for effectively taking advantage of LLM’s reasoning and generalization capability to perform credit assignments? (5) How does SWEET-RL scale as the number of training samples increase compared to baselines?

- 5.1 Experimental Setup

Baseline comparisons. We compare SWEET-RL with a variety of SOTA LLMs and multi-turn RL algorithms on ColBench. We consider Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct (Llama3Team, 2024) as representatives of SOTA open-source LLMs and GPT4-O and O1-Mini as representatives of SOTA proprietary LLMs. We test these models in both a single-turn and a collaborative setting to understand how LLM agents can benefit from multi-turn collaborative interactions on ColBench. We compare different RL algorithms based on Llama-3.1-8B-Instruct. We first consider a simple yet effective baseline Rejection Fine-Tuning widely used for LLM agent fine-tuning (Zhou et al., 2024b; Dong et al., 2023), where Supervised Fine-Tuning (SFT) is

Table 2 Comparisons of different LLMs and multi-turn RL algorithms on ColBench. SWEET-RL is able to achieve more than

###### 6% performance gain over other multi-turn RL algorithms, enabling Llama-3.1-8B-Instruct to be on par with larger proprietary models.

Backend Programming Frontend Design

% Tests Passed Success Rate Cosine Similarity Win Rate

Llama-3.1-8B-Instruct 11.8 6.9 63.1 13.6 Llama-3.1-70B-Instruct 24.2 14.8 61.8 13.2 O1-Mini 22.4 13.1 70.2 23.8 GPT-4O 27.6 16.2 68.6 23.8

Single-Turn

Llama-3.1-70B-Instruct 48.0 35.0 73.7 39.8 GPT-4O 54.6 40.4 78.1 50.0 O1-Mini 43.2 30.3 77.5 47.2

SOTA LLMs

Zero-Shot 34.2 22.4 72.4 33.8 Rejection Fine-Tuning 40.9 28.2 75.2 38.6 Multi-Turn DPO 48.0 34.4 76.9 42.8 SWEET-RL (ours) 56.8 40.4 77.7 48.2

Llama-3.1-8B-Instruct

performed on successful trajectories to minimize the negative log-likelihood loss. Furthermore, we consider a recent effective baseline Multi-Turn DPO that applies a variant of DPO (Rafailov et al., 2024b) to the multi-turn setting (Xiong et al., 2024; Song et al., 2024). Multi-Turn DPO first constructs contrastive trajectory pairs where the chosen trajectory achieves a higher trajectory reward compared to the rejected trajectory, and uses the DPO loss to directly optimize the policy without using a critic.

SWEET-RL first trains a turn-wise advantage model using the same model architecture with access to trainingtime information and then optimizes the policy with respect to rewards given by the turn-level advantage model as described in section 4. We use the reference code and web page as training-time information for Backend Programming and Frontend Design respectively. Due to the multi-modal nature of the reference web page, we instantiate the advantage LLM with a similar sized VLM Qwen2-VL-7B-Instruct (Yang et al., 2024) with a regression head on top of the mean representations of all visual and text tokens.

Note that our experiments focus on the setting of learning from historically collected data (offline setting), and thus RL algorithms like PPO (Schulman et al., 2017) and REINFORCE (Williams, 2004) that require on-policy data collection do not apply. This is because in the real world, online human collaboration data requires extensive human annotations and can be costly to obtain.

Evaluation metrics. Each task on Backend Programming comes with 10 unit tests, and we report the average percentage of tests passed and task success rate where all 10 unit tests for a task are passed. We report the average cosine similarity of the final web page and the reference web page measured by the image representations from Clip-vit-base-patch32 (Radford et al., 2021) for Frontend Design. To provide a more interpretable metric, we also include the win rate with respect to GPT-4O, where the model that lands on a web page closer to the reference web page as measured by cosine similarity wins for each task.

- 5.2 Comparisons on ColBench

- Table 2 shows the performance comparison of different SOTA LLMs and multi-turn RL algorithms across different tasks on ColBench. First, comparing “single-turn” results and the other collaborative results, we note that multi-turn collaborations can greatly enhance the performance of LLM agents for artifact creations by more closely aligning the final product with the reference “expectations” of human simulators. If the agent has to directly produce the final product in a single turn (top rows), even the best-performing GPT-4O can only achieve 16.2%. In contrast, the success rates for all models are doubled (e.g., the success rate for Llama-3.1-8B-Instruct increases from 6.9% to 22.4%) if they are given the chance to interact with human simulators for several turns and gather more information. Nonetheless, multi-turn collaboration remains a challenging task even for proprietary LLMs like GPT-4O and O1-Mini, which can only achieve a success rate of 40.4% and 30.3%, respectively. Despite the improvements of O1-Mini on symbolic reasoning tasks such as math and coding, we observe that these improvements do not directly result in taking a better strategy for multi-turn collaborative agents, indicating that downstream fine-tuning is still necessary for LLMs to optimize

collaboration with humans.

- 5.3 Comparing other algorithms with SWEET-RL

After fine-tuning with downstream data, we find that even the most naive RL algorithm, Rejection Fine-Tuning, can improve the performance on both tasks, with 5.8% and 4.8% improvements on Backend Programming success rate and Frontend Design win rate, respectively. However, we observe that oftentimes Rejection Fine-Tuning simply teaches the LLM to “memorize” the solution for each training task without learning a generalizable strategy to tackle a novel test tasks. While this issue is mitigated for Multi-Turn DPO by introducing “negative gradients” for the rejected trajectories, the improvement is still limited without proper credit assignments over a long horizon. By explicitly training a turn-level reward model to perform credit assignments through the use of training-time information, we observe a significant gain of SWEET-RL over Multi-Turn DPO on both tasks (6% improvement in success rate for Backend Programming and 5.4% improvement in win rate for Frontend Design). In fact, the resulting model of SWEET-RL using Llama-3.18B-Instruct not only matches the performance of Llama-3.1-70B-Instruct with more than 8 times parameters but also achieves competitive performance with SOTA proprietary models like GPT-4O and O1-Mini.

- 5.4 Analysis

With the advantage of SWEET-RL over baselines presented in Table 2, this section presents analytical results to understand this advantage and compare alternative designs.

| | |SWEET-RL| | |44.3|0%|
|---|---|---|---|---|---|---|
| | |SWEET-RL<br><br>LLM-as-a-Jud|w/o Training-Tim ge|e Information| | |
| | |SWEET w/ V|34.2<br><br>alue Function|0%<br><br>37.3|0%| |
| | | | | | | |
| | |28.5|0%| | | |
| | | | | | | |
| | | | | | | |
| | | |Zeroshot su|ccess rate| | |
| | | | | | | |

45

SWEET-RL

Multi-Turn DPO

41.60%

40.40%

SuccessRate(%)

SFT

40

34.40%

35

30.60% 30.90%

30

28.20%

27.60%

26.40%

27.00%

25

Zeroshot success rate

20

0 1 2 3 5

0 3k 9k 15k

(a) Best-of-N Sampling (2^n)

(b) Training Samples

- Figure 3 (a) Scaling curve of Best-of-N sampling with respect to different step reward model on Backend Programming. Results show that SWEET can best tell good actions on a turn-wise basis, resulting in the best scaling with respect to Best-of-N sampling. Note that this curve is different from test-time scaling curve because SWEET exploits additional training-time information. (b) Performance scaling of different multi-turn RL algorithms with respect to the amount of fine-tuning data on Backend Programming. While SWEET-RL takes more data initially to learn a reliable critic, it quickly catches up and achieves a better converging performance.

How should we use training-time information to help credit assignment? We carry out Best-of-N sampling experiments on Backend Programming to directly compare the capability of different methods to perform credit assignments. At each turn, N candidate actions are sampled from a fixed actor Llama-3.1-8B-Instruct and different methods are used to choose the best action to be executed. We compare our reward model in SWEET-RL with three other natural choices: 1) SWEET-RL w/o Training-Time Information – the same as SWEET-RL except that it only has the interaction history as inputs without access to the reference solution, 2) LLM-as-a-Judge uses a Llama-3.1-8B-Instruct to compare pairwise the quality of each action based on the interaction history and reference solution, and 3) SWEET-RL w/ Value Function where a regular classification head on top of s Llama-3.1-8B-Instruct backbone is trained to predict the expected success rate given the interaction history and the reference solution with a binary classification loss. Performance comparisons are presented in Figure 3(a) and qualitative comparisons are presented in Appendix D. First of all, we observe that the use of training-time information can significantly enhance the capability to perform credit assignment, evidenced by the huge performance gap between SWEET-RL and SWEET-RL w/o Training-Time Information. While Best-of-N sampling with respect to a fixed LLM-as-a-Judge can result in some improvements over the zeroshot success rate, this improvement is limited. Qualitatively, we found that a fixed LLM-as-a-Judge can easily get distracted by the length and format of the response without actually attending to its utility for task success. Finally, while being standard practice in the deep RL literature, the use of a value function fails to achieve comparable scaling performance compared to SWEET-RL. This shows the importance of the careful

RL algorithmic choices of SWEET-RL and that the go-to practice of training a value function may generalize poorly in unseen tasks.

What are the best parameterization choices for the critic to perform credit assignment? In Table 3, we perform ablation experiments to understand the effect of different parameterizations for the advantage function of the critic. In particular, we consider two alternative parameterizations: 1) SWEET-RL w/ Regression Head uses a regression head on top of the mean pooled representations of the last hidden state across all tokens, 2) SWEET-RL w/o Training-Time Information only uses the interaction history as inputs without access to the reference solution, and 3) SWEET-RL w/o Normalization where we do not perform the normalization step of division by the number of tokens in Equation 3. As shown in Table 3, similar to the conclusion from the previous section, the use of a regression head does not generalize well compared to SWEET-RL and training-time information significantly improves the performance. Without the additional normalization step, we find that the trained actor quickly collapses by generating shorter and shorter responses, showing the importance of carefully chosen parameterizations for multi-turn RL algorithms.

- Table 3 Ablation study on different parameterizations of the critic. Results show that the parameterization of using the mean log probability significantly outperform the other natural choices.

% Tests Passed Success Rate

Rejection Fine-Tuning 40.9 28.2 Multi-Turn DPO 48.0 34.4 SWEET-RL 56.8 40.4

SWEET-RL w/ Regression Head 45.3 36.2 SWEET-RL w/o Train-Time Info. 44.0 31.2 SWEET-RL w/o Normalization 4.2 3.6

HowdoesSWEET-RLscalewiththeamountoffine-tuningdata? Additionally, we compare the scaling performance of SWEET-RL compared to the multi-turn RL baselines Rejection Fine-Tuning and Multi-Turn DPO. Results are presented in Figure 3(b). Although SWEET-RL requires more data to train a reliable critic for performing credit assignment where it under-performs Multi-Turn DPO with 3k fine-tuning samples available, SWEET-RL quickly catches up with more samples once the turn-wise critic is trained and results in a significantly improved converging performance.

How does SWEET-RL work for different model architectures and off-policy data? Finally, Table 4 presents additional comparison experiments on Backend Programming using a stronger base model Llama-3.1-70BInstruct to study how SWEET-RL works across different model architectures. We use the same offline data generated by Llama-3.1-8B as Table 2 to understand the effectiveness of different methods for taking advantage of off-policy data (i.e. offline generated by a different model). We first observe that Rejection Fine-Tuning fails to learn from data generated by a worse model, with the success rate dropping from 35.0% to 31.9%. This is potentially because the objective of Rejection Fine-Tuning forces Llama-3.1-70B-Instruct to imitate sub-optimal trajectories generated from the worse Llama-3.1-8B-Instruct word-by-word. While Multi-turn DPO is able to achieve a big improvement even using off-policy data generated from an inferior model (35.0% to 41.8% in success rate), SWEET-RL still maintains a similar gap of 3.8% through performing explicit credit assignments with training-time information (41.8% compared to 45.6% in success rate).

- Table 4 Comparison results on Backend Programming using Llama-3.1-70B-Instruct as the base model. Results show that SWEET-RL achieves a similar gain over the baselines when using stronger Llama-3.1-70B-Instruct as the base model.

Llama-3.1-70B-Instruct % Tests Passed Success Rate

Zeroshot 48.0 35.0 Rejection Fine-Tuning 45.5 31.9 Multi-Turn DPO 56.7 41.8

SWEET-RL 60.2 45.6

6 Conclusion

To advance the development of effective multi-turn RL algorithms that perform effective credit assignments, this paper first introduces a benchmark, ColBench, focusing on the realistic domain of collaborative artifact creation. ColBench is the first LLM agent benchmark designed to validate multi-turn RL algorithms for reasoning-intensive tasks with minimum engineering overhead. Building upon ColBench, we develop a novel multi-turn RL algorithm, SWEET-RL, leveraging additional training-time information and appropriate algorithmic choices, achieving significantly improved performance over SOTA baselines in this domain. Our experiment results on ColBench show that off-the-shelf deep RL methods for multi-turn LLM agents can lead to poor generalization performance. While SWEET-RL serves as a preliminary step for closing this gap, there are a lot of future research opportunities to develop a better multi-turn RL algorithm in this important area of LLM agents.

Impact Statement

This paper advances the development of more effective multi-turn RL algorithms and better human-agent collaborations. An effective collaborative LLM may significantly improve human productivity in many areas such as content creation. However, various safety concerns may arise as LLM agents take over more tasks from humans where they might be subject to malicious attacks or conduct unexpected behaviors. We leave this important direction for future research.

References

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models, 2023. https://arxiv.org/abs/ 2311.18232.

Alekh Agarwal, Nan Jiang, and Sham M Kakade. Reinforcement learning: Theory and algorithms. 2019. Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-the-wild

device-control agents with autonomous reinforcement learning, 2024. https://arxiv.org/abs/2406.11896. Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39:324, 1952. https://api.semanticscholar.org/CorpusID:125209808.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Wang, Samuel Marks, Charbel-Raphaël Segerie, Micah Carroll, Andi Peng, Phillip Christoffersen, Mehul Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J. Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter Hase, Erdem Bıyık, Anca Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. Open problems and fundamental limitations of reinforcement learning from human feedback, 2023. https://arxiv.org/abs/2307.15217.

Ching-An Cheng, Andrey Kolobov, Dipendra Misra, Allen Nie, and Adith Swaminathan. Llf-bench: Benchmark for interactive learning from language feedback, 2023. https://arxiv.org/abs/2312.06853.

Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences, 2023. https://arxiv.org/abs/1706.03741.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web, 2023. https://arxiv.org/abs/2306.06070.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment, 2023. https://arxiv.org/abs/2304.06767.

GeminiTeam. Gemini: A family of highly capable multimodal models, 2024. https://arxiv.org/abs/2312.11805. Izzeddin Gur, Hiroki Furuta, Austin V. Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust.

A real-world webagent with planning, long context understanding, and program synthesis, 2021.

Hyeonbin Hwang, Doyoung Kim, Seungone Kim, Seonghyeon Ye, and Minjoon Seo. Self-explore: Enhancing mathematical reasoning in language models with fine-grained rewards, 2024. https://arxiv.org/abs/2404.10346.

Song Jiang, Da JU, Andrew Cohen, Sasha Mitts, Aaron Foss, Justine T Kao, Xian Li, and Yuandong Tian. Towards full delegation: Designing ideal agentic behaviors for travel planning, 2024. https://arxiv.org/abs/2411.13904.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024. https://arxiv.org/abs/2310.06770.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks, 2024. https://arxiv.org/abs/2401.13649.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra Faust. Training language models to self-correct via reinforcement learning, 2024. https://arxiv.org/abs/2409.12917.

Hugo Laurençon, Léo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset, 2024. https://arxiv.org/abs/2403.09029.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024. https://arxiv.org/abs/2406.11794.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. https://arxiv.org/abs/2305.20050.

Jessy Lin, Nicholas Tomlin, Jacob Andreas, and Jason Eisner. Decision-oriented dialogue for human-ai collaboration,

2024. https://arxiv.org/abs/2305.20076.

Yen-Ting Lin, Di Jin, Tengyu Xu, Tianhao Wu, Sainbayar Sukhbaatar, Chen Zhu, Yun He, Yun-Nung Chen, Jason Weston, Yuandong Tian, et al. Step-KTO: Optimizing mathematical reasoning through stepwise binary feedback. arXiv preprint arXiv:2501.10799, 2025.

Guanlin Liu, Kaixuan Ji, Renjie Zheng, Zheng Wu, Chen Dun, Quanquan Gu, and Lin Yan. Enhancing multi-step reasoning abilities of language models through direct q-function optimization, 2024. https://arxiv.org/abs/2410. 09302.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating llms as agents, 2023. https://arxiv.org/abs/2308.03688.

Llama3Team. The llama 3 herd of models, 2024. https://arxiv.org/abs/2407.21783. Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu,

Lei Meng, Jiao Sun, and Abhinav Rastogi. Improve mathematical reasoning in language models by automated process supervision, 2024. https://arxiv.org/abs/2406.06592.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning, 2013. https://arxiv.org/abs/1312.5602.

Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between value and policy

based reinforcement learning, 2017. https://arxiv.org/abs/1702.08892. OpenAI. Gpt-4 technical report, 2024. https://arxiv.org/abs/2303.08774. Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini

Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens,

Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. https://arxiv.org/abs/2203.02155.

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with swe-gym, 2024. https://arxiv.org/abs/2412.21139.

Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization, 2024. https://arxiv.org/abs/2404.19733.

Lerrel Pinto, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, and Pieter Abbeel. Asymmetric actor critic for image-based robot learning, 2017. https://arxiv.org/abs/1710.06542.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. https://arxiv.org/abs/2103.00020.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to q∗: Your language model is secretly a q-function, 2024a. https://arxiv.org/abs/2404.12358.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct

preference optimization: Your language model is secretly a reward model, 2024b. https://arxiv.org/abs/2305.18290. Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Android in the wild: A large-scale

dataset for android device control, 2023. https://arxiv.org/abs/2307.10088.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents, 2024. https://arxiv.org/abs/2405.14573.

Sasha Salter, Dushyant Rao, Markus Wulfmeier, Raia Hadsell, and Ingmar Posner. Attention privileged reinforcement

learning for domain transfer. ArXiv, abs/1911.08363, 2019. https://api.semanticscholar.org/CorpusID:215835372. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization

algorithms, 2017. https://arxiv.org/abs/1707.06347.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning,

2024. https://arxiv.org/abs/2410.08146.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models,

2024. https://arxiv.org/abs/2402.03300. Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering?, 2024. https://arxiv.org/abs/2403.03163. Charlie Snell, Ilya Kostrikov, Yi Su, Mengjiao Yang, and Sergey Levine. Offline rl for natural language generation with implicit language q learning, 2023. https://arxiv.org/abs/2206.11871. Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024. https://arxiv.org/abs/2408.03314. Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. Trial and error: Exploration-based trajectory optimization for llm agents, 2024. https://arxiv.org/abs/2403.02502.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. Solla, T. Leen, and K. Müller, editors, Advances in Neural Information Processing Systems, volume 12. MIT Press, 1999. https://proceedings.neurips.cc/paper_files/paper/1999/file/ 464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.

Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Walter Talbott, Katherine Metcalf, Natalie Mackraz, Devon Hjelm, and Alexander Toshev. Large language models as generalizable policies for embodied tasks, 2024. https://arxiv.org/abs/2310.17722.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback, 2022. https: //arxiv.org/abs/2211.14275.

Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, and Yi Wu. Offline reinforcement learning for llm multi-step reasoning, 2024a. https://arxiv.org/abs/2412.16145.

Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. Mint: Evaluating llms in multi-turn interaction with tools and language feedback, 2024b. https://arxiv.org/abs/2309.10691.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 2004. https://api.semanticscholar.org/CorpusID:2332513.

Matthew Wilson and Tucker Hermans. Learning to manipulate object collections using grounded state representations,

2020. https://arxiv.org/abs/1909.07876. Yuxiang Wu and Baotian Hu. Learning to extract coherent summary via deep reinforcement learning, 2018. https: //arxiv.org/abs/1804.07036. Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. Travelplanner: A benchmark for real-world planning with language agents, 2024a. https://arxiv.org/abs/2402.01622.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024b. https://arxiv.org/abs/2404.07972.

Wei Xiong, Chengshuai Shi, Jiaming Shen, Aviv Rosenberg, Zhen Qin, Daniele Calandriello, Misha Khalman, Rishabh Joshi, Bilal Piot, Mohammad Saleh, Chi Jin, Tong Zhang, and Tianqi Liu. Building math agents with multi-turn iterative preference learning, 2024. https://arxiv.org/abs/2409.02392.

Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction, 2024. https://arxiv.org/abs/2412.04454.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. https://arxiv.org/abs/2407.10671.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents, 2023. https://arxiv.org/abs/2207.01206.

Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao Peng. Free process rewards without process labels, 2024. https://arxiv.org/abs/2412.01981.

Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. Fine-tuning large vision-language models as decision-making agents via reinforcement learning, 2024. https://arxiv.org/abs/2405.10292.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents, 2024a. https://arxiv.org/abs/2307.13854.

Yifei Zhou, Qianlan Yang, Kaixiang Lin, Min Bai, Xiong Zhou, Yu-Xiong Wang, Sergey Levine, and Erran Li. Proposer-agent-evaluator(pae): Autonomous skill discovery for foundation model internet agents, 2024b. https: //arxiv.org/abs/2412.13194.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi-turn rl, 2024c. https://arxiv.org/abs/2402.19446.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2020. https://arxiv.org/abs/1909.08593.

- A Hyperparameters

For completeness and reproducibility, we present all hyperparameters used in SWEET-RL and all baselines in Table 5. In general, we found that the performances of both Multi-Turn DPO and SWEET-RL are consistent with respect to the hyperparameters in the DPO loss objective such as learning rate and beta, and the inclusion of a negative-log-likelihood coefficient of 0.01 helps in most cases (Pang et al. (2024) also found this to be useful).

Table 5 Hyperparameters for SWEET-RL and baseline methods for all experiments.

| | |Backend Programming Frontend Design|
|---|---|---|
|Rejection Fine-Tuning<br><br>|learning rate batch size epochs|2e-7 2e-7 32 32 4 8|
|Multi-Turn DPO<br><br>|learning rate beta negative-log-likelihood loss coefficient batch size epochs|2e-7 2e-7 0.1 0.1 0.01 0.01 8 8 4 8|
|SWEET-RL|critic learning rate critic beta critic negative-log-likelihood loss coefficient critic batch size critic epochs actor lr actor beta actor negative-log-likelihood loss coefficient actor batch size actor epochs|2e-7 2e-7 0.1 0.1 0.01 0.01 8 8 4 8 2e-7 2e-7 0.1 0.1 0.01 0.01 8 8 1 1|

- B Theoretical Justifications

In this section, we will provide some theoretical justifications for the derivation of our method. We first show that the trajectory-level Bradley-Terry objective can be similarly used for learning the advantage function:

- Lemma B.1. For an MDP with each state (o,c) ∈ O × C and for any π : O  → A, assume the transition function T (o,a,c) is deterministic for any (o,a,c) then the following holds ∀τ+,τ−:

−log σ

t

βr(o+t ,a+t ,c) −

t

βr(o−t ,a−t ,c) = −log σ

t

βAπ(o+t ,a+t ,c) −

t

βAπ(o−t ,a−t ,c)

Proof. To show this, we would like to prove:

Aπ(ot,at,c) We will prove this by telescoping:

r(ot,at,c) =

t

t

Aπ(ot,at,c)

t

[Qπ(ot,at,c) − V π(ot,c)]

=

t

N−1

t+1∼T (·|ot,at,c)V π(o′t+1,c) − V π(ot,c) + r(oN,aN,c)

r(ot,at,c) + Eo′

=

t=1

N−1

N

t+1∼T (·|ot,at,c)V π(o′t+1,c) − V π(ot,c)

Eo′

=

r(ot,at,c) +

t=1

t=1

N

=

r(ot,at,c),

t=1

where the last equality follows by the assumption of deterministic transition.

| |
|---|

Additionally, we would like to also provide a theoretical justification for using an asymmetric critic for optimizing the actor with a different observation space. Intuitively, although for each sample such a trainingtime advantage function may give a different judgement compared to its regular counterpart, the following lemma shows that the policy gradient (Williams, 2004) estimated from an advantage function with training-time information is unbiased averaged over all samples.

- Lemma B.2. For an MDP with each state (o,c) ∈ O × C and for any π : O  → A, let dπt (ot,at,c) to be joint state-action occupancy distribution at step t, the following two estimators are both unbiased estimators of the policy gradient of π:

∇Eτ∼π

H

N

r(ot,at,c) =

t=1

t=1

N

t,atAπ(ot,at)∇log π(at|ot) =

Eo

t=1

t,at,cAπ(ot,at,c)∇log π(at|ot)

Eo

Proof. The proof of this lemma is similar to the standard policy gradient analysis (Agarwal et al., 2019).

∇Eτ∼π

N

V π(o1)

r(ot,at,c) =∇EcEo

1

t=1

1∇V π(o1)

=EcEo

=EcEo

1∇

a1

π(a1|o1)Qπ(o1,a1)

=EcEo

1

(∇π(a1|o1))Qπ(o1,a1) +

a1

a1

π(a1|o1)∇Qπ(o1,a1)

=EcEo

1

a1

π(a1|o1)(∇log π(a1|o1))Qπ(o1,a1) +

a1

π(a1|o1)∇Qπ(o1,a1)

=EcEo

1

=EcEo

1

N

=Ec

t=1

(∇log π(a1|o1))Qπ(o1,a1) +

###### Ea

1

a1

###### (∇log π(a1|o1))Qπ(o1,a1) + Ea

###### [Ea

1

1

t,atQπ(ot,at)∇log π(at|ot).

Eo

To proceed, we need to first show a useful equality:

π(a1|o1)∇Qπ(o1,a1)

2∇V π(o2)]

Eo

t,atV π(ot)∇log π(at|ot) =Eo

π(at|ot)V π(ot)∇log π(at|ot)

Eo

t

a

V π(ot)∇π(at|ot)

=Eo

t

a

V π(ot)∇

=Eo

π(at|ot)

t

a

V π(ot)∇1

=Eo

t

=0

|Backend Programming Prompt<br><br>You are a helpful LLM agent. Your task is to help a human user to resolve their problem, in particular python programming.<br><br>1) Note that the problem is highly personalized so you need to explicitly gather information by asking questions to the human user about some hidden information and implicit constraints. YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.<br>2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.<br>3) When you have gathered enough information to answer, say "I WANT TO ANSWER:" in the beginning of your response and provide your final answer.<br>4) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.<br>5) You should be as concise as possible in your response to human. "I WANT TO ANSWER:" should be included in your response to human if you think that you have gathered enough information for addressing this problem. Directly output the raw python code after "I WANT TO ANSWER:". Complete only the immediate agent response in this dialogue:<br>|
|---|

- Figure 4 The prompt used for testing different models on Backend Programming task.

Therefore, we can use the advantage function instead of the Q-function in the expression of policy gradients:

∇Eτ∼π

N

r(ot,at,c) =Ec

t=1

=Ec

N

N

=Ec

N

N

=

t=1

N

=

t=1

N

=Ec

t,atQπ(ot,at)∇log π(at|ot)

Eo

t=1

t,at(Qπ(ot,at) − V π(ot,at))∇log π(at|ot)

Eo

t=1

t,atAπ(ot,at)∇log π(at|ot)

Eo

t=1

t,at (∇log π(at|ot))EcAπ(ot,at)

Eo

t,at (∇log π(at|ot))EcAπ(ot,at,c)

Eo

t,at (∇log π(at|ot))Aπ(ot,at,c),

Eo

t=1

where the second last equation follows from the fact that Ec∼dπ

t (·|ot,at)Aπ(ot,at,c) = Aπ(ot,at).

| |
|---|

- C Prompts

For completeness, we have included the prompt that we used for testing different models on Backend Programming in Figure 4 and on Frontend Programming in Figure 5.

- D Qualitative Comparisons of Different Credit Assignment Methods

We present qualitative comparison results of different credit assignment in Figure 6. First of all, we observe that LLM-as-a-Judge can easily get distracted by the length and format of the response without actually attending to its utility for task success. Furthermore, while being a natural practice in deep RL literature, the use of a value function fails to reasonably predit the expected future utility in unseen tasks. In particular, it

|Frontend Design Prompt<br><br>You are a helpful LLM agent. Your task is to help a human user to code a complete website with a good design in HTML and Tailwind CSS. Write the code inside a tag <html>. Write real and long sentences about the business. You don’t have to include images, but if you do, use only this source https://picsum.photos/id/48/W/H, by replacing W and H with the width and height of the image. Keep the id the same to only use id 48 image.<br><br>1) Note that the problem is highly personalized so you need to go through a few rounds of revisions.<br>2) When you have gathered enough information to answer, say "I WANT TO ANSWER:" in the beginning of your response and provide your final answer.<br>3) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.<br>4) You will be judged both by the quality of the final answer and the efficiency of the conversation.<br>5) You can include ONLY ONE snippet raw html and Tailwind css code (wrapped in html tag)in your response to human user to ask how is the proposed design different from what the human user wants. This snippet of raw html and Tailwind css code (WRAPPED IN html TAG) will be rendered for the human to see a screenshot of the webpage. The human user will respond by comparing your rendered webpage with the webpage that the human user has in mind.<br>6) You need to make sure that your html webpage looks exactly as the human user wants, including the overall layout, navigation bars, background color etc.<br>7) The human user can only see your rendered image and provide suggestions based on the rendered image, and not any text questions. First output your thought on your remaining uncertainties about the understanding of the problem and user preferences such as name of the function, input format, output format, and etc. Then say "OUTPUT: n" followed by your proposal html.<br>|
|---|

###### Figure 5 The prompt used for testing different models on Frontend Design task.

predicts that the first candidate response has a probability of 97% to lead to the final success of the agent despite being only the second turn out of 10 turns and this candidate response being phrased in a very confusing way. In contrast, SWEET-RL is able to tell the advantage of the second response with a higher score because it is important in this task for the agent to figure out that the returned list can contain duplicate objects. This shows the importance of the RL algorithmic choices of SWEET-RL and that the go-to practice of training a value function may generalize arbitrarily poorly in unseen tasks.

Interaction History

Training-Time Reference

[Figure 37]

I'd like you to write a Python function that can identify symmetric objects within a set. The function should take in a list of objects, where each object is a tuple of coordinates, and a list of axes to check for symmetry. The function should return a list of objects that are symmetric across one of the given axes.

Possible Next Actions

[Figure 38]

- 1. Can an object be symmetric across mul ple axes at the same  me? If not, do we consider an object symmetric if it's symmetric across any of the given axes (inclusive or exclusive of the respec ve axis)?
- 2. What if an object is point (0, 0) or any object that can be scaled the same value across all axes (for example an object at point (2,2,2) an object (4, 4, 4))? Is this considered as symmetric across mul ple axes?

Should the returned list contain duplicate objects (i.e., objects that are reﬂec ons of each other across the same axis), or should the list contain each symmetric pair only once?

[Figure 39]

1. What type of coordinates are in the tuple for an object (e.g., 2D, 3D, etc.)?2. Are the axes provided as tuples or some other format (e.g., (x, y), (x, y, z), or some numerical index for an nth axis, etc.)?

For example, if an object is a tuple of (x, y, z), then the axes would be provided as a list of integers, such as [0, 1, 2], where 0 corresponds to the x-axis, 1 corresponds to the y-axis, and 2 corresponds to the z-axis.

SWEET

- 80.8 - 32.4 0.97 0.78

[Figure 40]

Value Function

Agent 2's response is a good start. However, it doesn't explicitly ask for some crucial information about the symmetry definition. Agent 1, on the other hand, has relevance of questions is important to arrive at an accurate implementation as shown in the provided reference. AGENTS 1 WINS

[Figure 41]

LLM-as-a-Judge

……

###### Figure 6 Qualitative comparisons between different credit assignment methods. A fixed LLM-as-a-Judge can be easily distracted by length and formats of the actions without considering their actual utility. A value function generalizes poorly to unseen tasks. In contrast, SWEET can attend to the actual utility of the action for task success and generalize well.

- E Full Qualitative Examples

To demonstrate the level of difficulty of tasks in ColBench and provide a qualitative comparisons of different models, we have included examples of full trajectories in this section.

In particular, in Figure 7, Figure 8, Figure 9, Figure 10, we have provided full trajectories on Backend Programming for SWEET-RL Llama-3.1-8B-Instruct, Zeroshot Llama-3.1-8B-Instruct, and Zeroshot GPT4-O. While zeroshot baselines do try to propose some critical questions to seek more information from the human collaborator, they quickly jump into conclusions without gathering enough information, thus resulting in a wrong answer. Such failure modes exist even for stronger general-purpose LLMs like GPT4-O, indicating that task-specific tuning may always be necessary despite the improvement in the capability of the base model. In contrast, SWEET-RL Llama-3.1-8B-Instruct learnt back-and-forth information-seeking behaviors and only answered the question once all information has been collected. Surprisingly, we found that RL training also results in some emergent behaviors such as reasoning with longer chain-of-thought and even self-corrections as shown in the last response from the agent in Figure 8.

We also include a full trajectory example on Frontend Design with SWEET-RL Llama-3.1-8B-Instruct in Figure 11, Figure 12, Figure 13, Figure 14, Figure 15, Figure 16. We would like to note the significant complexity of this task where the agent needs to reason about HTML code over an extended horizon (up to 16k tokens), as a HTML code snippet is included in the response of each turn. After SWEET-RL training, the LLM agent has learnt nuanced collaborative and reward-maximizing behaviors where it first proposes a scratch solution to gather coarse-grained feedback and only perform fine-grained edits in the end.

Backend Programming - SWEET-RL Llama3.1-8B-Instruct Reference: test set #84

[Figure 42]

[Figure 43]

I need a python function that determines whether the government should grant recognition to a tribe based on their ancestral and demographic data.The signature of the function is: def recognize_tribe(ancestral_data, demographic_stats)

[Figure 44]

[Figure 45]

To create the function recognize_tribe, I need more information about the inputs and the criteria for recognition. Can you please tell me what the ancestral_data and demographic_stats variables represent in this function? For example, are they lists, dictionaries, or something else? Also, what are the key factors that the function should consider when deciding whether to grant recognition to the tribe? Are there any specific rules or conditions that I should know about?

The ancestral_data is a dictionary containing information about the tribe's ancestry, and the demographic_stats is a list of dictionaries, each representing a demographic statistic with keys such as 'ancestry' and 'documents'. The key factors for recognition include the tribe's ancestry ratio and the presence of certain documents, but I'm not sure what the exact required ancestry ratio and document

[Figure 46]

[Figure 47]

So the ancestral_data is a dictionary that contains information about the tribe's ancestry, and demographic_stats is a list of dictionaries where each dictionary contains information about a specific demographic statistic. You mentioned that the ancestry ratio and the presence of certain documents are key factors for recognition. That's a good starting point. Can you please clarify what you mean by "ancestry ratio"? Is it a proportion of the tribe's population that belongs to a specific ancestry, or is it a ratio of some other kind (e.g. a ratio of the tribe's size to the size of the surrounding population)? Additionally, what do you mean by "certain documents"? Are there specific documents that are required to be present in the demographic_stats list for the tribe to be recognized? If so, which ones are they?

I'm not entirely sure about the exact required ancestry ratio, but I think it's a proportion of the tribe's population that belongs to a specific ancestry. The required documents are an ancestral certificate, identity proof, and tribal affiliation, but I'm not sure if these are the only ones or if there are any other conditions that need to be met.

[Figure 48]

[Figure 49]

So the ancestry ratio is a proportion of the tribe's population that belongs to a specific ancestry, and the required documents for recognition are an ancestral certificate, identity proof, and tribal affiliation. To clarify, are all of these documents (ancestral certificate, identity proof, and tribal affiliation) expected to be present in the demographic_stats list for the tribe to be recognized, or are some of them optional? Also, is the ancestry ratio that we're interested in the proportion of the tribe's population that belongs to any specific ancestry that is mentioned in the ancestral_data, or is it a specific proportion of a particular ancestry that is predefined? For example, is it a proportion of the total population that belongs to the tribe's "primary" or "main" ancestry, as per the ancestral_data?

I think all the documents (ancestral certificate, identity proof, and tribal affiliation) are expected to be present in the demographic_stats list for the tribe to be recognized. I'm not sure about the specific proportion of the ancestry ratio, but I think it's a predefined proportion of a particular ancestry.

[Figure 50]

- Figure 7 Example full trajectory for Backend Programming with SWEET-RL Llama-3.1-8B-Instruct. After training, the LLM agent has learnt back-and-forth information seeking behaviors before giving the final answer.

(Continued) Backend Programming - SWEET-RL Llama3.1-8B-Instruct Reference: test set #84

[Figure 51]

[Figure 52]

So all three documents (ancestral certificate, identity proof, and tribal affiliation) must be present in the demographic_stats list for the tribe to be recognized. And you mentioned that the ancestry ratio is a predefined proportion of a particular ancestry, but you're not sure what the specific proportion is. Let me try to summarize the information we have so far: ancestral_data is a dictionary with information about the tribe's ancestry demographic_stats is a list of dictionaries, one of which must contain an ancestral certificate, identity proof, and tribal affiliation The tribe must meet a predefined ancestry ratio for a particular ancestry. I'm going to take a guess that the function will return True if the tribe meets both of these conditions, and False otherwise. Is this a correct assumption, or are there any other conditions that need to be met for the tribe to be recognized? If I understand the problem correctly, I'll try to provide a code solution in the next message.

I think the function will return True if the tribe meets both of these conditions, and False otherwise. I'm not sure about the specific proportion of the ancestry ratio, but I think it's a predefined proportion of a particular ancestry, and I also think that if the tribe is already a recognised tribe according to the ancestral_data, then it should be recognized.

[Figure 53]

[Figure 54]

So the demographic_stats lists will be non-empty lists of dictionaries, and the function should check if the required documents (ancestral certificate, identity proof, and tribal affiliation) are present in the list. Since the lists will not be empty, we can simply check if the documents are in the list. Here's a code solution based on the conditions we discussed: def recognize_tribe(ancestral_data, demographic_stats):

# Check if the tribe is already recognized in the ancestral_data is_recognized = ancestral_data.get('recognized', False) # Assuming 'recognized' is the key in ancestral_data

# Check if the required documents are in the demographic_stats list has_required_documents = any('ancestral certificate' in stats for stats in demographic_stats) and any('identity proof' in stats for stats in demographic_stats) and

any('tribal affiliation' in stats for stats in demographic_stats)

# Check if the tribe meets the ancestry ratio (let's assume it's 0.5 for now, replace with the actual value if provided) # meets_ancestry_ratio = ... # This will be True if the tribe meets the ratio, False otherwise # For the sake of completion, I'll leave it as a placeholder for now, but it's not used in the final function.

# Return True if the tribe is recognized, has the required documents, and meeting the ancestry ratio (which we assumed as 0.5) return is_recognized or (has_required_documents and 0.5) # I used a placeholder value for the ratio, it should be replaced with the actual ratio.

However, the above solution will always return True when the tribe has the required documents, so the solution might be adjusted to return is_recognized or (has_required_documents and meets_ancestry_ratio) where meets_ancestry_ratio is the actual ratio you're looking for. I'll proceed as if this ratio will be used in the logical OR operation to return True if the tribe is recognized or if it has the documents and meets the ratio. Please note that meets_ancestry_ratio is a comment for now and should be replaced with the actual variable checking the ratio. Also, the key in ancestral_data is assumed to be 'recognized', but it should be replaced with the actual key if different. Given the conditions you provided, the meets_ancestry_ratio isn't actually used in the function as the function now will return True if the tribe is recognized or if it has the required documents.

Let's proceed to the final answer. I WANT TO ANSWER: def recognize_tribe(ancestral_data, demographic_stats):

# Check if the tribe is already recognized in the ancestral_data is_recognized = ancestral_data.get('recognized', False)

# Check if the required documents are in the demographic_stats list has_required_documents = any('ancestral certificate' in stats for stats in demographic_stats) and any('identity proof' in stats for stats in demographic_stats) and

any('tribal affiliation' in stats for stats in demographic_stats)

# Return True if the tribe is recognized or if it has the required documents return is_recognized or has_required_documents

[Figure 55]

### Unit Tests Passed! ✅

[Figure 56]

- Figure 8 Example full trajectory for Backend Programming with SWEET-RL Llama-3.1-8B-Instruct (Continued). After training, the LLM agent has learnt back-and-forth information seeking behaviors before giving the final answer.

###### 21

Backend Programming - Zeroshot Llama3.1-8B-Instruct Reference: test set #84

[Figure 58]

[Figure 59]

I need a python function that determines whether the government should grant recognition to a tribe based on their ancestral and demographic data.The signature of the function is: def recognize_tribe(ancestral_data, demographic_stats)

[Figure 60]

[Figure 61]

To provide a helpful function, can you please clarify what specific information is contained in the ancestral_data and demographic_stats arguments. For example:

What kind of ancestral data is it (e.g., percentage of indigenous population, historical records, tribe's timeline)? What are the demographic stats (e.g., population size, percentage of population, ethnic diversity)? Is there any specific condition or threshold that the tribe must meet to be recognized?

The ancestral data contains information such as whether the tribe is already recognized, and the demographic stats include data points like the percentage of ancestry and the documents provided by the tribe members, such as ancestral certificates, identity proofs, and tribal affiliations. The tribe must meet a certain ancestry ratio and provide all the required documents to be recognized.

[Figure 62]

[Figure 63]

How is the ancestry ratio defined and determined? Is it a fixed percentage of the total population, or does it have to be a mix of specific ancestral groups?

Also, are the "required documents" a fixed set, or is it open-ended?

The ancestry ratio is a fixed percentage, and the tribe must have at least 25% of its population with the required ancestry to be recognized. The required documents are a fixed set, consisting of ancestral certificates, identity proofs, and tribal affiliations, and all of these must be provided by the tribe members.

[Figure 64]

[Figure 65]

So the function would expect a tribe to have at least 25% of its population with the required ancestry, and all tribe members must provide specific documents.

I WANT TO ANSWER:

def recognize_tribe(ancestral_data, demographic_stats): required_ancestry_percentage = 0.25 required_documents = ["ancestral certificates", "identity proofs", "tribal affiliations"]

if demographic_stats["ancestry_ratio"] >= required_ancestry_percentage and \ all(document in demographic_stats["documents"] for document in required_documents) and \ ancestral_data["recognized"] is False:

return True else:

return False

[Figure 66]

### Unit Tests Failed! ❌

[Figure 67]

- Figure 9 Example full trajectory for Backend Programming with Zeroshot Llama-3.1-8B-Instruct. While the agent has asked a few questions, it quickly jumps into conclusions, resulting in a wrong final answer.

Backend Programming - Zeroshot GPT-4o Reference: test set #84

[Figure 68]

[Figure 69]

I need a python function that determines whether the government should grant recognition to a tribe based on their ancestral and demographic data.The signature of the function is: def recognize_tribe(ancestral_data, demographic_stats)

[Figure 70]

[Figure 71]

Could you please specify what criteria or conditions need to be checked within the ancestral and demographic data to determine recognition?

The function should check if the tribe has a recognized ancestral history or if a certain percentage of its demographic statistics meet specific ancestry and documentation requirements. I'm not sure what the exact percentage or documentation requirements are, but I know that the function should return True if these conditions are met and False otherwise.

[Figure 72]

[Figure 73]

Could you clarify what constitutes a "recognized ancestral history"? Additionally, if you have any default percentage or documentation requirements in mind, please share those so I can tailor the function to your needs.

I'm not entirely sure what constitutes a "recognized ancestral history", but I think it might be indicated by a specific key in the ancestral data, such as 'recognised_tribe' being set to True. As for the percentage and documentation requirements, I believe the function should check for an ancestry ratio of at least 0.25 and the presence of certain documents like 'ancestral_certificate', 'identity

[Figure 74]

[Figure 75]

Could you confirm if the 'demographic_stats' includes both the 'ancestry_ratio' and a list of documents? Additionally, should the function check both conditions independently, or can one suffice for recognition?

Yes, the 'demographic_stats' includes both the 'ancestry_ratio' and a list of documents, and the function should check both conditions independently, but if the 'recognised_tribe' key in the ancestral data is True, then the function should return True regardless of the demographic stats. The function should also check for the presence of 'ancestral_certificate', 'identity_proof', and 'tribal_affil

[Figure 76]

[Figure 77]

I WANT TO ANSWER: def recognize_tribe(ancestral_data, demographic_stats):

if ancestral_data.get('recognised_tribe', False): return True

ancestry_ratio = demographic_stats.get('ancestry_ratio', 0) documents = demographic_stats.get('documents', [])

required_documents = {'ancestral_certificate', 'identity_proof', 'tribal_affiliation'}

if ancestry_ratio >= 0.25 and required_documents.issubset(documents): return True

return False

[Figure 78]

Unit Tests Failed! ❌

[Figure 79]

- Figure 10 Example full trajectory for Backend Programming with Zeroshot GPT-4O. While the agent does propose critical questions to the human collaborator, it also has the issue of jumping into conclusions.

FrontEnd Design - SWEET-RL Llama3.1-8B-Instruct (1) Reference: test set #22

[Figure 80]

Creative Agency: A playful and visually stimulating design with a diagonal grid pattern, a logo in the top left corner, and a call-to-action button prominently displayed. Bright and bold color palette.

[Figure 81]

[Figure 82]

It seems like you'd like me to design a visually appealing website for a creative agency. I'm a bit unclear about some aspects, so I'd like to clarify:

Do you have a preferred name for the creative agency? If not, I'll suggest a default name. Do you have a specific logo in mind, or would you like me to generate a placeholder logo for demonstration purposes? Would you like the call-to-action (CTA) button to lead to a specific page (e.g., homepage, about, contact) or should it be a general "learn more" type of CTA? Are there any specific colors you'd like to use or would you like me to suggest a bold and bright color palette? Are there any specific pages you'd like to include in the design (e.g., services, portfolio, team)? Given these uncertainties, I'll provide a basic design proposal to get feedback. Here's the code:

[Figure 83]

The agent-provided image has a different layout and color scheme compared to the ground truth design. The text in the agent-provided image is aligned to the left, while in the ground truth design, the text is centered. Additionally, the agentprovided image has a white background with a blue header, whereas the ground truth design has a light gray background with a purple header. The "Get Started" button in the agent-provided image is blue, while in the ground truth design, it is purple. The overall design and color scheme of the ground truth image are more cohesive and visually appealing.

[Figure 84]

[Figure 85]

It seems like the agent-provided design had some deviations from the ground truth design. To address the issues mentioned, I will make the following adjustments:

Align the text to the center. Change the background and header colors to match the ground truth design. Change the button color to match the ground truth design. Improve the overall design and color scheme to match the ground truth design. To confirm, I'll try to include a diagonal grid pattern and a more cohesive color scheme. Here's another code snippet to review: This updated design includes a gray background with a white header section, centered text, and a purple "Get Started" button. Please let me know if this revised design meets your vision. If not, please provide more specific feedback so I can make further adjustments.

[Figure 86]

- Figure 11 Example full trajectory for Frontend Design with SWEET-RL Llama-3.1-8B-Instruct (1). After training, the LLM agent has learnt sophisticated task-specific strategies to optimize the final reward.

[Figure 87]

The agent-provided image has a different layout and design compared to the ground truth image. Here are the main differences:

Text Alignment and Positioning: In the agent-provided image, the text is aligned to the left and positioned below the image. In the ground truth image, the text is aligned to the right and positioned next to the image. Text Color and Style: In the agent-provided image, the text is in black and has a standard font style. In the ground truth image, the text is in a bold, purple font. Button Color and Position: In the agent-provided image, the "Get Started" button is white and positioned below the text. In the ground truth image, the "Get Started" button is purple and positioned below the text. Image Position: In the agent-provided image, the image is positioned on the left side. In the ground truth image, the image is positioned on the right side. Text Content: The text content in

[Figure 88]

[Figure 89]

It seems like there are several discrepancies between the agent-provided design and the ground truth design. To address these differences, I'll make the following adjustments:

Align the text to the right and position it next to the image. Change the font style to bold and text color to purple. Change the "Get Started" button color to purple Move the image to the right side. Here's the updated code:

[Figure 90]

[SEE RENDERED HTML] This updated design meets all of your requirements: aligned text to the right, next to the image, with a bold, purple font style and color. The "Get Started" button is purple and positioned below the text, and the image is on the right side.

However, I still didn't address the issue of the grid pattern mentioned in the original specification. I'm assuming that this is a feature that might be requested in the future, but I want to confirm it is still a requirement.

Please review this updated design and let me know if it meets your expectations or if I need to make further adjustments.

The agent-provided image has a different layout and color scheme compared to the ground truth design. The text in the agent-provided image is aligned to the left, while in the ground truth design, the text is centered. Additionally, the agentprovided image has a white background with a single image on the right side, whereas the ground truth design has a light gray background with the image and text side by side. The text color in the agent-provided image is black, while in the ground truth design, the text is blue. The "Get Started" button in the agent-provided image is white with black text, while in the ground truth design, it is blue with white text.

[Figure 91]

- Figure 12 Example full trajectory for Frontend Design with SWEET-RL Llama-3.1-8B-Instruct (2). After training, the LLM agent has learnt sophisticated task-specific strategies to optimize the final reward.

Frontend Design - SWEET-RL Llama3.1-8B-Instruct (3) Reference: test set #22

[Figure 92]

[Figure 93]

To achieve the desired layout and color scheme, I will make the following adjustments:

[Figure 94]

Center the text. Change the background color to a light gray. Display the image and text side by side. Change the text color to blue. Change the "Get Started" button color to blue with white text. Here's an updated code snippet:

[SEE RENDERED HTML] This updated design should meet all the requirements mentioned: centered text, light gray background, image and text side by side, blue text, and blue "Get Started" button with white text.

Please verify that this is in line with the ground truth design, or if there's anything else that needs to be adjusted.

The agent-provided image has a different layout and color scheme compared to the ground truth design. The text in the agent-provided image is aligned to the left and uses a different font style, while the ground truth design has the text centered and uses a larger, bolder font. The "Get Started" button in the agent-provided image is gray and rectangular, whereas in the ground truth design, it is blue and has rounded corners. Additionally, the background color of the agentprovided image is a solid gray, while the ground truth design has a white background.

[Figure 95]

[Figure 96]

To address these differences, I'll make the following adjustments:

Center the text and use a larger, bolder font. Change the background color to white. Change the "Get Started" button color to blue and add rounded corners. Update the font style to match the ground truth design. Here's the updated code:

[Figure 97]

[SEE RENDERED HTML] This updated design includes a white background, centered text with a larger, bolder font, and a blue "Get Started" button with rounded corners. The font style has also been updated to match the ground truth design. Please review this design and let me know if it meets your requirements.

- Figure 13 Example full trajectory for Frontend Design with SWEET-RL Llama-3.1-8B-Instruct (3). After training, the LLM agent has learnt sophisticated task-specific strategies to optimize the final reward.

[Figure 98]

The agent-provided image has a different layout and content compared to the ground truth image. Here are the main differences:

Text Content and Positioning: In the agent-provided image, the text "Welcome to Our Creative Agency!" is at the top left, while in the ground truth image, the text "Creative Agency" is centered at the top. The agent-provided image has a smaller text block with a "Get Started" button on the right side, while the ground truth image has a larger text block with a "Get Started" button at the bottom center. Image Positioning: In the agent-provided image, the image of the laptop is on the left side, while in the ground truth image, the same image is on the right side. Text Alignment and Size: The text in the agent-provided image is smaller and less prominent compared to the text in the ground truth image, which is larger and more visually striking. Button Positioning: The "Get Started" button in the agent-provi

[Figure 99]

[Figure 100]

Based on the differences mentioned, I will adjust the design to match the ground truth image. Here is the updated code: [SEE RENDERED HTML] This design now includes the correct text positioning, font size, and alignment, as well as the correct positioning of the "Get Started" button and the image. I have also centered the text and image horizontally using the align-items and justify-content properties in the grid. Please review this updated design and let me know if it meets your requirements.

[Figure 101]

The agent-provided image has a white background with a centered text block and a centered image on the right side. The text is in black, and there is a blue "Get Started" button below the text. The image on the right is a photo of a laptop on a wooden table.

The ground truth design that the human user wants has a light gray background with a centered text block and a centered image on the right side. The text is in a darker shade of blue, and there is a purple "Get Started" button below the text. The image on the right is the same photo of a laptop on a wooden table.

[Figure 102]

[Figure 103]

Here is the updated code to match the ground truth design: [SEE RENDERED HTML] I've made the following changes:

[Figure 104]

Changed the background color to a light gray (#F0F0F7) Changed the text color to a darker shade of blue (#2F343A) Changed the color of the "Get Started" button to purple (#6c5ce7) Updated the hover effect for the "Get Started" button This should match the ground truth design you described.

- Figure 14 Example full trajectory for Frontend Design with SWEET-RL Llama-3.1-8B-Instruct (4). After training, the LLM agent has learnt sophisticated task-specific strategies to optimize the final reward.

[Figure 105]

The agent-provided image has a different layout and color scheme compared to the ground truth design. The text in the agent-provided image is aligned to the left, while in the ground truth design, the text is centered. Additionally, the text color in the agent-provided image is black, whereas in the ground truth design, the text is in a blue color. The "Get Started" button in the agent-provided image is blue with white text, while in the ground truth design, the button is also blue but with a different shade and white text. The overall layout of the text and the button is also different, with the ground truth design having a more balanced and centered appearance.

[Figure 106]

[Figure 107]

It seems like the agent-provided design has some alignment and color issues. To fix these issues, I will make the following adjustments:

Align the text to the center. Change the text color to the desired blue color. Update the button color and text color to match the ground truth design. Adjust the layout to have a more balanced and centered appearance. Here's the updated code:

[Figure 108]

[SEE RENDERED HTML] This updated code should have a centered text block, a more balanced layout, and the correct colors. Let me know if there are any further adjustments needed.

The agent-provided image has a different layout and content compared to the ground truth design. Here are the main differences:

Text Alignment and Positioning: In the agent-provided image, the text is centered and placed above the image. In the ground truth image, the text is aligned to the left and placed to the left of the image. Text Content: The agent-provided image has a brief description: "We help you grow your brand through innovative design and creative solutions." The ground truth image has a more detailed description: "At Creative Agency, we believe in the power of creativity to transform businesses and brands. We are a team of passionate designers, developers, and strategists who are dedicated to creating visually stunning and user-friendly websites. Our approach is playful, innovative, and results-driven. We work closely with our clients to understand their business, their target audience, and their unique brand voice. We believe in the

[Figure 109]

[Figure 110]

To address the differences in the layout and content, I will make the following adjustments: Align the text to the left. Move the image to the left side, next to the text. Update the text content to match the ground truth design. Format the text content to make it more readable. Here's the updated code: [SEE RENDERED HTML] This updated code should have the text aligned to the left, the image moved to the right, and the text content updated to match the ground truth design. Let me know if there are any further adjustments needed. However, please note that the formatting of the text content has been updated but the scroll width in tailwind-css needs to be adjusted to let the text align properly

[Figure 111]

- Figure 15 Example full trajectory for Frontend Design with SWEET-RL Llama-3.1-8B-Instruct (5). After training, the LLM agent has learnt sophisticated task-specific strategies to optimize the final reward.

Frontend Design - SWEET-RL Llama3.1-8B-Instruct (6) Reference: test set #22

[Figure 112]

The agent-provided image has the text aligned to the left, while the ground truth image has the text centered. Additionally, the "Get Started" button in the agent-provided image is blue, whereas in the ground truth image, it is purple.

[Figure 113]

[Figure 114]

To fix the issues, I will make the following adjustments:

[Figure 115]

Center the text. Change the color of the "Get Started" button to purple. Here's the updated code:

[SEE RENDERED HTML] This updated code should center the text and change the color of the "Get Started" button to purple.

[Figure 116]

Image Similarity

### 0.883

- Figure 16 Example full trajectory for Frontend Design with SWEET-RL Llama-3.1-8B-Instruct (6). After training, the LLM agent has learnt sophisticated task-specific strategies to optimize the final reward.

