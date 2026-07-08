## arXiv:2512.10534v3[cs.AI]5Mar2026

[Figure 1]

2025-12-15

# Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning

#### Haiteng Zhao*,1, Junhao Shen*,1,2, Yiming Zhang*,1, Songyang Gao1, Kuikun Liu1

Tianyou Ma1,3, Fan Zheng4, Dahua Lin1,5, Wenwei Zhang1,†, Kai Chen1,† 1Shanghai AI Laboratory 2Shanghai Jiao Tong University 3Peking University 4ICMAT, Spanish National Research Council 5MMLab, The Chinese University of Hong Kong

{zhaohaiteng,zhangwenwei,chenkai}@pjlab.org.cn

Large language model (LLM) agents exhibit strong mathematical problem-solving abilities and can even solve International Mathematical Olympiad (IMO) level problems with the assistance of formal proof systems. However, due to weak heuristics for auxiliary constructions, AI for geometry problem solving remains dominated by expert models such as AlphaGeometry 2, which rely heavily on large-scale data synthesis and search for both training and evaluation. In this work, we make the first attempt to build a medalist-level LLM agent for geometry and present InternGeometry. InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine’s feedback to guide subsequent proposals. A dynamic memory mechanism enables InternGeometry to conduct more than two hundred interactions with the symbolic engine per problem. To further accelerate learning, we introduce Complexity-Boosting Reinforcement Learning (CBRL), which gradually increases the complexity of synthesized problems across training stages. Built on InternThinker-32B, InternGeometry solves 44 of 50 IMO geometry problems (2000-2024), exceeding the average gold medalist score (40.9), using only 13K training examples, just 0.004% of the data used by AlphaGeometry 2, demonstrating the potential of LLM agents on expert-level geometry tasks. InternGeometry can also propose novel auxiliary constructions for IMO problems that do not appear in human solutions.

### 1. Introduction

Large language model (LLM) agents have demonstrated general problem-solving ability across domains such as mathematics and programming. By interacting with tools such as code interpreters and LEAN [17] multiple times, LLM agents can reason and reflect on the tool execution feedback and progressively solve complex problems. Such a universal paradigm can obtain medalist-level performance on International Mathematical Olympiad (IMO) level problems and is believed to have better generalization ability [11, 14].

However, when faced with geometry problems, the potential of LLM agents is still underexplored. The IMO-level geometry problems usually require extra-long proving steps, whose solution not only combines various geometry theorems but also contains creative auxiliary constructions that have weak heuristics and require multiple trials, as shown in Figure 1. Consequently, current state-of-the-art approaches [3, 5, 29] mainly adopt expert models learned from large-scale synthesized data to guide the large-scale search with a symbolic engine for finding the geometry proof. Given the success of LLM agents in other mathematical domains, this raises a natural question: Can we adopt LLM agents in solving geometry problems for achieving higher efficiency and generalization?

To answer this question, this paper makes the first attempt to investigate LLM agents for solving IMO-level geometry problems and proposes InternGeometry, an LLM agent that obtains medalist-level performance for

* Equal contribution, † Corresponding author

A convex quadrilateral ABCD satisfies AB · CD = BC · DA. Point X lies inside ABCD so that

A

T4

∠XAB = ∠XCD and ∠XBC = ∠XDA. Prove that ∠BXA + ∠DXC = 180 .

◦

A

T

M

D

T1

X

K

W

B

C

D

X

L

T2 T3

C

B

(b) Auxiliary Constructions Proposed by InternGeometry

(a) Original Geometry Sutrcutre in IMO 2018 P6

- Figure 1: An example of IMO-level geometry problems. (a) The configuration in IMO 2018 Problem 6 appears simple, but it is difficult to prove. Its solution relies on sophisticated constructions, as illustrated in (b).

solving geometry problems. We first identify the limitations in the current open-sourced symbolic engines for geometry problems and build InternGeometry-DDAR, which contains a rich theorem library whose search space theoretically covers the complete solution of most of the IMO geometry problems. Taking InternGeometry-DDAR

- as a tool, InternGeometry solves the geometry problem through long-term LLM-tool interactions, where the LLMs continuously propose propositions or auxiliary constructions after thinking with natural language, verify those ideas in the symbolic engine with formal language, and reflect on the feedback from the symbolic engine
- at each interaction. In the long-term reasoning process, InternGeometry adopts dynamic memory to maintain the exploration history and the observed geometry properties in a compact form, which not only reduces the context without losing key information but also guides diverse explorations in future interactions. By extra-long-horizon LLM-tool interactions with memory, InternGeometry can conquer the weak heuristics of geometry proof and progressively find the feasible solution based on the accumulated geometry properties observed during explorations. Such a design also aligns with human experts who obtain insights into auxiliary construction by exploratory probing [5, 29].

To train InternGeometry, we first apply cold start training using 7K examples created by formalizing existing geometry problems and constructing trajectory data. After the cold start, we introduce a complexity-boosting reinforcement learning (CBRL) framework, a multi-stage curriculum RL pipeline [4, 18, 33, 37], to further improve training efficiency. Specifically, we build a data synthesis pipeline that can generate geometry tasks with a specified complexity (e.g., required proof steps). At each stage, we first synthesize problems at the current complexity level, then perform RL training on the current InternGeometry model, and update the target complexity based on the results to best fit the current model. Over iterations, the synthesized problems become highly challenging, providing the foundation for acquiring expert-level capabilities on high-difficulty tasks.

We conduct extensive experiments to verify the effectiveness of InternGeometry. InternGeometry solves 44 out of 50 geometry problems from 2000 to 2024, surpassing the average score of IMO gold medalists (40.9 points) and the score of AlphaGeometry2 (42) and SeedGeometry (43), and it also solves the geometry problem in IMO 2025. Notably, the model attains this performance with only approximately 13K training examples, 0.004% of AlphaGeometry2 and 0.006% of SeedGeometry. Our ablation studies further demonstrate that long-horizon proof interaction is critical to the agent’s ability: removing proposition proving steps and only allowing the agent to add auxiliary constructions significantly degrades performance, substantiating the importance of long-horizon trial-and-error for the weak-to-strong heuristic transition. Complexity escalation plays a pivotal role in RL convergence: directly training on high-difficulty data leads to low task completion rates and poor convergence, whereas using data below a certain difficulty threshold substantially impairs generalization to IMO-level tasks. In addition, our case studies show that the model can devise novel auxiliary constructions compared to human solutions, exhibiting creativity in geometric reasoning.

### 2. Method

#### 2.1. Geometry Proof Language and Environment

In previous work such as AlphaGeometry, geometric structures are defined point by point through domainspecific language (DSL). Once the construction is complete, a deductive database arithmetic reasoning (DDAR) system is employed to exhaustively search for theorems, deriving all conclusions reachable from the known

Query

###### InternGeometry

A

###### ......

###### Answer

###### Query Think Action Feedback Think Action Feedback

D

X

B

C

###### Dynamic Memory Dynamic Memory

Answer

Domain Specific Language as Action

Proof steps:

000. ABKX are cyclic (Arc determines internal angles)=> ∠(AB,BX) = ∠(AK,KX)

Alright, with the initial setup, I now have a solid foundation

- action 1 <build>: successful
- action 2 <propose>: successful
- action 3 <propose>: fail

Successfully build the InternGeometry-DDAR Environment. The available premises are listed below:

Historical Memory

... So, I will add k on both circle a b x and circle c d x.

Compress

...

eqratio a b a d b c c d

......

- eqangle a x a b c x c d
- eqangle b x b c d x d a

<action> <add> k = on_circum k a b x, on_circum k c d x

action M <add>: successful Current feedback:

Keep Key Information

=> 1/1 * ∠(BX,AX) + 1/1 * ∠(DX,CX) = 1pi/1

......

...

Training

Complexity Boosting Reinforcement Learning

Generate complexityconditioned training batch Reinforcement learning

Training Batch

Synthesize complexityconditional data based on rewards

Reward Data Synthesis 0 Agent Policy 1

Compute rewards

- Figure 2: An overview of InternGeometry and Complexity-Boosting Reinforcement Learning (CBRL). (a) InternGeometry performs natural-language reasoning (Think), outputs a structured action in a domain-specific language (Action), and receives execution results (Feedback) in each turn. A dynamic memory module W compresses the multi-turn interaction history to preserve essential actions and outcomes. (b) CBRL optimizes the agent policy by generating synthetic training data with controllable difficulty, assigning binary rewards to effective steps and successful outcomes, and optimizing policy through iterative reinforcement learning.

facts. In this work, we build InternGeometry-DDAR, an interactive geometric proof engine based on the open-source DDAR system Newclid [25]. To support more complex geometric structures, we introduce several advanced definition strategies, such as globally optimizing point placements to satisfy constraints. During interaction, an agent not only employs the DSL for both problem specification and auxiliary point construction, but also proposes sub-proof goals that will be subsequently verified by the engine. As an interactive proof engine, InternGeometry-DDAR maintains state across steps, including the geometric configuration, constructed auxiliary points, and all proven preliminaries and propositions. Further details are provided in the Appendix B.

#### 2.2. Geometric Proof Agent

The agent is allowed to perform natural-language reasoning at each step and then mark its final action output using an action-separating token. Let the agent be denoted by G and the interactive proof engine by E. Given a geometric problem 𝑋, at step 𝑡 the agent’s output is

##### [𝑃𝑡,𝐴𝑡] = G(𝑋,W(𝐻𝑡−1)) (1)

where 𝑃𝑡 denotes the slow chain-of-thought reasoning, 𝐴𝑡 the final formalized code, and 𝐻𝑡−1 the interaction history, which includes each round’s thoughts, actions, and the feedback observations obtained from the environment prior to step 𝑡. The module W is a memory manager that returns a compressed long history to improve the agent’s long-horizon capacity.

The code 𝐴𝑡 is executed by the proof engine E, which is in state 𝑡 − 1. The execution result 𝑂𝑡 is appended together with the corresponding thoughts and actions to the interaction history as feedback to the agent, guiding its next step of reasoning and action.

##### 𝑂𝑡,E𝑡 = E𝑡−1(𝐴𝑡) 𝐻𝑡 = 𝐻𝑡−1 + [𝑃𝑡,𝐴𝑡,𝑂𝑡]

(2)

At each reasoning step, the agent may summarize progress, analyze the problem, or plan future proof strategies. In its action code, the agent can choose specific operations—such as constructing geometric objects, adding auxiliary constructions, or verifying whether a proposition holds.

When all targets in the problem have been proven, the geometric reasoning tool determines that the problem is fully solved, aggregates the entire proof process, and produces a complete proof of the problem, thereby concluding the reasoning session.

We posit that long-horizon capability is key to addressing the weak-heuristic challenge of auxiliary construction in geometric proofs. To this end, we introduce a dynamic memory management strategy W for the agent, as well as a prior-guided rejection sampling method for G.

To shorten the long interaction history 𝐻, which can span hundreds of turns, W summarizes earlier exchanges, including their thoughts and detailed environment feedback, while retaining only core action outputs and key environment feedback to improve the agent’s context efficiency, as illustrated in Figure 2. Through W, the agent obtains a concise overview of its action history and their outcomes—e.g., whether an auxiliary construction was successfully added and whether a proposed proposition holds. The last turn feedback remains unchanged, informing the agent of the currently known propositions.

Another major challenge for long-horizon agents is action collapse, where the model falls into poor patterns, such as producing highly repeated outputs or outputs similar to previous rounds [28]. To address this challenge, we use an intuitive rejection sampling method during agent inference to avoid such patterns. Denote naive LLM inference as [𝑃ˆ𝑡,𝐴ˆ𝑡] = 𝐺(𝑋,W(𝐻𝑡−1)). Then, for the sampled value [𝑃ˆ𝑡,𝐴ˆ𝑡],

If PassCheck([𝑃ˆ𝑡,𝐴ˆ𝑡]) : G(𝑋,W(𝐻𝑡−1)) = [𝑃ˆ𝑡,𝐴ˆ𝑡] Else: reject the value of [𝑃ˆ𝑡,𝐴ˆ𝑡] and return to the sampling step

(3)

Here, PassCheck is a rule-based multi-condition checking policy that enforces no repeated actions relative to the history, no excessively long thinking without a stop, no turn without a valid action or with formatting issues, and no use of the same action type for too many consecutive rounds.

#### 2.3. Complexity Boosting Reinforcement Learning

Before reinforcement learning begins, there is a cold-start phase in which we perform supervised fine-tuning on a small amount of synthetic data to help the agent quickly adapt to the task paradigm.

)︀𝑁𝑖=1, where 𝑋 denotes the input, ℎ = W(𝐻) denotes the compressed history (aligned with the agent’s input), and 𝑦 = [𝑃,𝐴] denotes the output, including the thinking content 𝑃 and the action content 𝐴. Let the agent model 𝐺 have parameters 𝜃. The supervised fine-tuning objective is:

(︀

Denote the training dataset as 𝒟 =

𝑋𝑖,ℎ𝑖,𝑦𝑖

[︃

]︃ (4)

∑︁𝑁

∑︁𝑇

(︀

)︀

1 𝑁

𝑦𝑡𝑖 | 𝑥𝑖,ℎ𝑖𝑡

−

log 𝐺𝜃

𝐿𝑠𝑡 =

𝑡=1

𝑖=1

Based on the supervised fine-tuning result 𝜃𝑠𝑡, the agent exhibits the basic behavior patterns expected in geometric reasoning tasks, such as slow thinking and proactively invoking tools.

The subsequent CBRL phase is an iterative interaction–training loop. In each iteration, the agent first attempts a proof on the training task and then performs online learning using reward signals from its trajectories. Following GRPO [23], given task 𝑋, the training gradient is:

∇𝐽𝑟𝑙(𝑋,𝜃) = E𝑦,ℎ∼𝐺

𝜃(|𝑋) ∑︁𝑇

,1 − 𝜖,1 + 𝜖))︂𝐴(𝑋,𝑦𝑡)∇𝐺𝜃 (𝑦𝑡 | 𝑋,ℎ𝑡)

min(︂

𝐺𝜃 (𝑦𝑡 | 𝑋,ℎ𝑡) 𝐺𝜃

𝐺𝜃 (𝑦𝑡 | 𝑋,ℎ𝑡) 𝐺𝜃

(5)

,clip(

(𝑦𝑡 | 𝑋,ℎ𝑡)

(𝑦𝑡 | 𝑋,ℎ𝑡)

old

old

𝑡=1

− 𝛽∇D𝐾𝐿 (𝐺𝜃‖𝐺ref )

where

}︀)︀ std

(︀{︀

𝑟11,𝑟21,··· ,𝑟𝑇1 ,𝑟12,··· ,𝑟𝑇𝐾

𝑟𝑡𝑖 − mean

}︀)︀ , (6)

𝐴𝑖 (𝑋,𝑦𝑡) =

(︀{︀

𝑟11,𝑟21,··· ,𝑟𝑇1 ,𝑟12,··· ,𝑟𝑇𝐾

represents the advantage at step 𝑡 of the 𝑖-th trajectory within a batch of 𝐾 samples. It measures the quality improvement at step 𝑡 of the 𝑖-th generated trajectory relative to the average policy. Here, 𝑟𝑡𝑖 denotes the reward at step 𝑡 of the 𝑖-th trajectory. 𝜖 is a hyperparameter that constrains the policy ratio. 𝐺𝜃

is the policy

old

model from the previous iteration. 𝐺ref is the initial model. D𝐾𝐿 is the KL divergence, used as a regularizer to constrain optimization of the agent model.

Here, the reward is a binary value, computed as the conjunction of the outcome reward and the step effectiveness reward:

𝑟 = 𝑟𝑜 ∧ 𝑟𝑠 (7) The outcome reward 𝑟𝑜 is 1 if the proof is complete; otherwise, it is 0. The step effectiveness reward is defined by whether the step’s action succeeds. For proposition-proposing steps, 𝑟𝑠 is 1 if the proposed proposition is successfully proven by the engine. For auxiliary-construction steps, 𝑟𝑠 is 1 if the construction is successfully added and used in the final proof of the overall question; otherwise, it is 0.

Note that the reward in our work is deliberately simple and can be computed by rules automatically. It rewards effective steps in trajectories that succeed while penalizing all steps in failed trajectories and ineffective steps in successful trajectories.

Next, we focus on the curriculum algorithm for CBRL in geometric tasks. One major advantage of our data synthesis pipeline is the fully controllable difficulty of the problems. As illustrated by AlphaGeometry [30], the difficulty of IMO geometry problems for humans is positively correlated with the number of DDAR proof steps. Therefore, we choose the DDAR proof step count as the measure of task complexity, denoted as 𝜅. Denote the data synthesis pipeline as X. We implement CBRL as follows:

E𝑋∼X(𝜅)𝐽𝑟𝑙(𝑋,𝜃) (8) 𝜅* = arg max

𝜃* = arg max

𝜃

𝜃|𝐴(𝑋,𝑦)| (9)

E𝑋∼X(𝜅)E𝑦∼𝐺

𝜅

where the goal of 𝜅 optimization is to maximize the average absolute advantage during learning, and 𝐴(𝑋,𝑦) is the advantage of outcome reward. We then present properties of CBRL. As shown in [33] and [4], maximum absolute advantage has the following properties:

- Theorem 2.1. Given model parameter 𝜃, the 𝜅* obtained from Equation 9 approximately optimally accelerates the learning progress.
- Theorem 2.2. For binary rewards, the maximum average absolute advantage is 0.5, which indicates that the task is of moderate difficulty for the model—neither too difficult nor too trivial.

In practice, in each CBRL round, we sample data conditioned on complexity 𝜅, perform RL training to the agent, and finally update 𝜅 according to learning rate 𝛼. See Appendix D for details.

2.4. Data Synthesis Pipeline

The proposed data pipeline targets to synthesize geometry problems with adaptable levels of difficulty for InternGeometry. Specifically, it comprises two stages: cold start and expert-level problem synthesis for CBRL.

First, due to the scarcity of data in DSL form, we fine-tuned InternThinker-32B [15] as InternGeometryFormalizer through expert iteration [1] and then exploit large-scale natural language problem data from diverse sources. This process produced a total of 7K formal problem and solution trajectory pairs, which provide a cold start for InternGeometry.

However, these paired data are constrained by the imbalanced difficulty distribution with relatively few problems at the expert level. To endow LLM with expert-level problem-solving abilities, we further synthesize problems dynamically during reinforcement learning. We first add auxiliary constructions into randomly constructed problems with statistical prior of given complexity, then leveraging the InternGeometry-DDAR to filter valid constructions and goals to form new problems. Finally, a total of 6K problems are constructed by difficulty based on proof steps during CBRL. See Appendix D for details.

- 3. Experiment

#### 3.1. Experiment Setup

Implementation. We use InternThinker-32B [15] as the backbone model for our method. For the agent model, we set the maximum number of steps to 200 by default, with inference hyperparameters of temperature 0.9

- Table 1: Comparison of overall performance on IMO 50 between InternGeometry and SOTA geometry expert models.

Model Model Type Training Data Sampling Setting IMO 50 Pass@K AlphaGeometry 2 Expert Model 300M Ensemble of search trees 42/50

SeedGeometry Expert Model 230M N/A 43/50 InternGeometry LLM Agent 13K Pass@256 44/50

- Table 2: A problem-by-problem comparison on IMO 50 between InternGeometry and SOTA geometry expert models.

Year ID Split AG2 SG IG Year ID Split AG2 SG IG Year ID Split AG2 SG IG

- 2000 P1 ✓ ✓ ✓ 2007 P2 ✓ ✓ ✓ 2015 P3 ✓ ✓ ✓

- 2000 P6 ✓ ✓ ✓ 2007 P4 ✓ ✓ ✓ 2015 P4 ✓ ✓ ✓

- 2001 P1 ✗ ✗ ✗ 2008 P1 a ✓ ✓ ✓ 2016 P1 ✓ ✓ ✓

2001 P5 ✓ ✗ ✓ 2008 P1 b ✓ ✓ ✓ 2017 P4 ✓ ✓ ✓ 2002 P2 a ✓ ✓ ✓ 2008 P6 ✓ ✓ ✓ 2018 P1 ✓ ✓ ✓

- 2002 P2 b ✓ ✓ ✓ 2009 P2 ✓ ✓ ✓ 2018 P6 ✗ ✓ ✓

- 2002 P6 ✗ ✗ ✗ 2009 P4 a ✓ ✓ ✓ 2019 P2 ✓ ✓ ✓

- 2003 P3 ✗ ✗ ✗ 2009 P4 b ✓ ✗ ✓ 2019 P6 ✓ ✓ ✓

2003 P4 a ✓ ✓ ✓ 2010 P2 ✓ ✓ ✓ 2020 P1 ✓ ✓ ✓ 2003 P4 b ✓ ✓ ✓ 2010 P4 ✓ ✓ ✓ 2020 P6 ✗ ✗ ✗ 2004 P1 ✓ ✓ ✓ 2011 P6 ✓ ✓ ✓ 2021 P3 ✓ ✓ ✓ 2004 P5 a ✓ ✓ ✓ 2012 P1 ✓ ✓ ✓ 2021 P4 ✓ ✓ ✓

- 2004 P5 b ✓ ✓ ✓ 2012 P5 ✓ ✓ ✓ 2022 P4 ✓ ✓ ✓

- 2005 P1 ✓ ✓ ✓ 2013 P3 ✓ ✓ ✓ 2023 P2 ✓ ✓ ✓

2005 P5 ✓ ✓ ✓ 2013 P4 ✓ ✓ ✓ 2023 P6 ✗ ✓ ✓ 2006 P1 ✗ ✓ ✗ 2014 P3 ✓ ✓ ✓ 2024 P4 ✓ ✓ ✓

- 2006 P6 ✗ ✗ ✗ 2014 P4 ✓ ✓ ✓ 2025 P2 N/A ✓ ✓

and top-p 0.9. During the test, the pass@K is set to 256.

Dataset and Baselines. We use IMO 50 [5] as the test set, which includes all geometry problems from IMO 2000 to IMO 2024. We additionally evaluate InternGeometry on the geometry problem from IMO 2025, reported separately in Table 2. We use AlphaGeometry 2 [5] and SeedGeometry [5] as our baselines, both of which are state-of-the-art geometry proving methods based on expert models. The performance of these baselines is taken directly from the results reported in their respective papers.

#### 3.2. Overall Results

We compare the performance of InternGeometry with baselines in Table 1. InternGeometry solved 44 problems in IMO 50, surpassing AlphaGeometry 2 and SeedGeometry. The “split” in the table refers to subproblems of questions that contain multiple subquestions. Notably, InternGeometry used only 13K data points—just 0.004% of AlphaGeometry 2 and 0.006% of SeedGeometry. Furthermore, its test-time scaling budget was also far lower than AlphaGeometry 2, which use ensembles of beam search, and the reported optimal single beam tree configuration is beam size 128, branching number (samples) 32, and beam depth 4. See Appendix G for more discussion. These comparisons clearly demonstrate the potential of LLM-based agent approaches on expert-level tasks.

We list the individual results on IMO 50 in Table 2. We additionally include the geometry problem of IMO 2025 in the table. InternGeometry solved 45 out of 51 problems, covering all problems solved by AlphaGeometry 2, and additionally solving 2018 P6 and 2023 P6. Compared to SeedGeometry, it additionally solved 2001 P5 and 2009 P4b, but missed 2006 P1. Notably, the remaining unsolved problems largely involve computations that go beyond the scope of pure geometric proof, and thus fall outside the current expressive range of geometric DDAR systems. See Appendix F for cases.

###### 3.3. Analysis for Long Horizon Agent To analyze the effect of long-horizon interaction to the proof, we compare the pass@K on IMO 50 under different max step setting, and the result is in Figure 3 (left). It is evident that as interaction steps increase, the proving

- Figure 3: Left: The effect of long-horizon interaction on the proof. As the interaction steps increase, the proving success rate improves significantly, which holds for different sampling times. As sampling times increase, Pass@K also rises, indicating the test-time scalability of InternGeometry. Right: Extending the agent’s trajectory length is more effective than repeated sampling for scaling. The total inference budget is defined as the sampling number K multiplied by the agent’s steps. When the maximum length is capped (the blue lines), performance improves with inference budget at a slower rate for shorter trajectories. On the other hand, when the sampling size is fixed (the green lines), increasing the budget by lengthening the trajectory yields efficient scaling.

- Table 3: Ablation study on long-horizon agents in InternGeometry.

Propositions Slow Thinking Context Compression Reject Sampling IMO 50 Pass@256 ✓ ✓ ✓ ✓ 44/50

- ✗ ✓ ✓ ✓ 35/50
- ✗ ✗ ✓ ✓ 23/50

✓ ✓ ✗ ✓ 20/50 ✓ ✓ ✓ ✗ 38/50

success rate improves significantly, which holds for different sampling times. Shorter interactions significantly limit the success rate of agent proofs. As the interaction trajectory grows, the agent can continually explore to develop heuristics about the problem, enabling it to better leverage its reasoning ability for generalization. Additionally, the figure illustrates the trend of test-time scaling. As shown, as sampling times increase, Pass@K also rises, indicating the test-time scalability of InternGeometry.

We emphasize that extending an agent’s trajectory length scales performance more effectively than repeated sampling. As shown in Figure 3 (right), with total inference budget defined as 𝐾 times the number of steps, performance grows slowly when the step cap is 64 but improves much faster when it is 200. For fixed 𝐾, increasing trajectory length consistently yields higher efficiency, supporting our hypothesis that long-horizon interactions enhance heuristics more effectively in geometric proofs.

An ablation on IMO 50 (Table 3) further confirms this. InternGeometry solves fewer problems when any long-horizon component is removed. Slow thinking and context compression increase solved problems from 20 and 23 to 44, underscoring their key roles in enabling IMO-level reasoning.

#### 3.4. Analysis on Complexity Boosting Reinforcement Learning

In this section, we analyze the effectiveness of the CBRL method. We first present the distribution of the synthetic data obtained during the CBRL process in Figure 4 (left). As noted earlier, we use the length of a problem’s proof steps as an indicator of its difficulty. Accordingly, we compile statistics on the distribution of proof lengths in the synthetic data generated during model training, as shown in the figure. The figure shows that the difficulty distribution of the synthetic data exhibits a fairly uniform improving trend, indicating that our agent training provides a well-structured curriculum from simple to difficult tasks, which helps the agent master complex combinations of proof skills.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- Figure 4: Left: The distribution of proof lengths in the synthetic data generated during model training, indicating task complexity. The figure shows that the difficulty distribution of the synthetic data exhibits a fairly uniform improving trend, providing a well-structured curriculum from simple to difficult tasks. Right: Agent’s generalization performance on IMO 50 during training. The agent’s overall performance on IMO 50 shows a steady upward trend. Notably, there is a significant performance jump in the sixth training round.

- Table 4: Ablation study on CBRL in InternGeometry.

Training Setting IMO 50 Pass@256

With CBRL 44/50 SFT Cold Start 22/50 Easy Data Only 29/50 Challenging Data Only 24/50 Same Data without Schedule 38/50

We further demonstrate the agent’s generalization performance on IMO 50 during training in Figure 4 (right). As the training data is continually updated, the agent’s overall performance on IMO 50 shows a steady upward trend. Notably, there is a significant performance jump in the sixth training round, indicating breakthrough progress as task complexity is progressively increased in the reinforcement learning process.

We then conduct ablation studies on the key components of CBRL. The result is in Table 4. Notably, after SFT cold start, the model’s baseline performance on IMO 50 is 22/50. We first analyze the impact of data difficulty on reinforcement learning, revealing the importance of allowing the learning algorithm to autonomously control the difficulty of synthesized data. To strictly control variables, we run experiments with the same data scale and training steps as in InternGeometry. Specifically, we modify the data distribution in the RL phase to: using only low-difficulty data, and using only high-difficulty data. The corresponding results are shown in the rows Easy Data Only and Challenging Data Only in the table. The results indicate that using only low-difficulty data limits the agent’s generalization ability to IMO-difficulty problems; on the other hand, using only high-difficulty data also leads to suboptimal training outcomes. The latter occurs because learning becomes slow and fails to converge within the same training budget: the agent remains in an early trial-and-error stage with extremely sparse learning signals. This highlights the efficiency of CBRL.

Finally, we ablate the dynamic difficulty curriculum by uniformly sampling from all synthesized data throughout the InternGeometry training phase. The result (Same Data without Schedule) shows that removing the curriculum again degrades performance. Without progressive difficulty, the agent struggles with sparse or absent rewards on sampled hard problems, preventing it from learning effective strategies under the same training budget. This confirms that CBRL significantly improves data efficiency and training effectiveness in reinforcement learning.

#### 3.5. Case Study

During our manual checking, we find that InternGeometry shows remarkable creativity on certain problems. As shown in Figure 1, while most human solvers relied on inversion, trigonometry or complex numbers, the agent solves this problem via an elegant geometric construction using classical angle chasing and basic theorems.

Specifically, InternGeometry first places a point 𝑇 on segment 𝐴𝐶 such that ∠𝐵𝐷𝐴 = ∠𝑇𝐷𝐶, and defines point 𝐾 as the intersection of two circles. These two points form an isogonal conjugate pair in quadrilateral 𝐴𝐵𝐶𝐷, revealing that the agent can discover this implicit structure through exploration. To further exploit the isogonal property, it then constructs the symmetric points of 𝑇 with respect to each side of the quadrilateral, showing both an understanding of rotational symmetry and an ability to generalize the use of auxiliary points for handling isogonal conjugates from triangles to quadrilaterals. Overall, this case study highlights how InternGeometry generate creative constructions that differ fundamentally from human solutions.

### 4. Related Work

Reinforcement learning agents in the field of mathematics Currently, large language model agents have achieved remarkable performance in tasks such as Code, Search, GUI, and also Mathematics through reinforcement learning (RL). In mathematics, RL agents based on informal proofs solve problems using general-purpose tools like Python compilers. Examples include OR (Open Reasoning) approaches Li et al. [13], Mai et al. [16], Prabhudesai et al. [19], Shang et al. [22], Shen et al. [24], Singh et al. [27], Zuo et al. [39] and PR (Proof Reasoning) approaches Goldie et al. [8], Hao et al. [10], Li et al. [12], Simonds & Yoshiyama [26]. Alternatively, agents built on Interactive Theorem Provers (ITPs) specialized for mathematics can handle more complex problems. Works like Ren et al. [21], Wang et al. [32], Xin et al. [34], Zhang et al. [36] achieve strong results on benchmarks such as miniF2F [38] and ProofNet [2]. However, few agents address geometry problems. Our work targets this gap, developing an interactive geometric prover and showing the potential of data synthesis and difficulty scaling.

Rurriculum learning for agents While several research [18, 33, 37] study curriculum reinforcement learning, curriculum agents learning remains limited, and most approaches rely on highly structured task types. For example, Voyager [31] uses manually designed curricula in MineDojo [7] to teach agents complex skills. WebRL [20] iteratively generates increasingly complex task instructions and employs a reward model for automatic success evaluation. In scaling RL for agents and task synthesis, Envgen [35] trains large language models to generate formal parameters for Crafter [9] and Heist [6], enabling dynamic environment and task generation. Unlike these approaches, our method fully automates large-scale synthesis at specified difficulty levels, allowing unrestricted and unbiased curriculum adjustment, especially at higher complexity.

Automatic geometry theorem proving Current AI-based approaches to automated geometric proofs remain largely expert-model driven. State-of-the-art systems like AlphaGeometry [5, 29] and SeedGeometry [3] typically decompose a problem into two tasks: (1) auxiliary construction prediction, where the model proposes additional geometric elements (e.g., lines, points); and (2) formal reasoning, where a search algorithm assembles a complete proof using geometric theorems (e.g., angle bisector, triangle similarity) and logical inference rules. Following this paradigm, most methods train specialized models on large datasets to predict constructions and then combine them with a formal search engine for proof generation. Recent work has also explored using large language models for geometric reasoning, but these efforts mainly target elementary-level problems and computational task formats.

### 5. Conclusion

LLM-based agents can solve tough math problems, even IMO-level with formal provers, but geometry remains dominated by expert systems like AlphaGeometry2 that depend on massive synthetic data and search.We introduce InternGeometry, a medalist-level LLM agent for geometry. It overcomes weak auxiliary-construction heuristics by iteratively proposing propositions and constructions, verifying them with a symbolic engine, and refining based on feedback. A dynamic memory enables over 200 interaction steps. To speed learning, we use CBRL, which gradually increases synthesized problem difficulty during training. Built on InternThinker-32B, InternGeometry solves 44/50 IMO geometry problems (2000–2024), surpassing the average gold medalist score (40.9), using only 13K training examples—about 0.004% of AlphaGeometry2’s data. It can also generate novel auxiliary constructions unseen in human solutions.

### References

- [1] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems, 30, 2017. 2.4
- [2] Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W Ayers, Dragomir Radev, and Jeremy Avigad. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. arXiv preprint arXiv:2302.12433, 2023. 4
- [3] Luoxin Chen, Jinming Gu, Liankai Huang, Wenhao Huang, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Kaijing Ma, et al. Seed-prover: Deep and broad reasoning for automated theorem proving. arXiv preprint arXiv:2507.23726, 2025. 1, 4
- [4] Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970,

2025. 1, 2.3, D

- [5] Yuri Chervonyi, Trieu H Trinh, Miroslav Olšák, Xiaomeng Yang, Hoang Nguyen, Marcelo Menegali, Junehyuk Jung, Vikas Verma, Quoc V Le, and Thang Luong. Gold-medalist performance in solving olympiad geometry with alphageometry2. arXiv preprint arXiv:2502.03544, 2025. 1, 1, 3.1, 4
- [6] Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning, pp. 2048–2056. PMLR, 2020. 4
- [7] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35:18343–18362, 2022. 4
- [8] Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, and Christopher D Manning. Synthetic data generation & multi-step rl for reasoning & tool use. arXiv preprint arXiv:2504.04736, 2025. 4
- [9] Danijar Hafner. Benchmarking the spectrum of agent capabilities. arXiv preprint arXiv:2109.06780, 2021. 4
- [10] Qianyue Hao, Sibo Li, Jian Yuan, and Yong Li. Rl of thoughts: Navigating llm reasoning with inference-time reinforcement learning. arXiv preprint arXiv:2505.14140, 2025. 4
- [11] Yichen Huang and Lin F Yang. Winning gold at imo 2025 with a model-agnostic verification-and-refinement pipeline. arXiv preprint arXiv:2507.15855, 2025. 1
- [12] Chengpeng Li, Mingfeng Xue, Zhenru Zhang, Jiaxi Yang, Beichen Zhang, Xiang Wang, Bowen Yu, Binyuan Hui, Junyang Lin, and Dayiheng Liu. Start: Self-taught reasoner with tools. arXiv preprint arXiv:2503.04625, 2025. 4
- [13] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl. arXiv preprint arXiv:2503.23383, 2025. 4
- [14] Thang Luong and Edward Lockhart. Advanced version of gemini with deep think officially achieves gold-medal standard at the international mathematical olympiad, july 2025. URL https://deepmind. google/discover/blog/advanced-version-ofgemini-with-deep-think-officially-achieves-goldmedal-standardat-the-international-mathematical-olympiad, 2025. 1
- [15] Chengqi Lyu, Songyang Gao, Yuzhe Gu, Wenwei Zhang, Jianfei Gao, Kuikun Liu, Ziyi Wang, Shuaibin Li, Qian Zhao, and Haian Huang. Exploring the limit of outcome reward for learning mathematical reasoning. 2025. 2.4, 3.1
- [16] Xinji Mai, Haotian Xu, Weinong Wang, Jian Hu, Yingying Zhang, Wenqiang Zhang, et al. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving. arXiv preprint arXiv:2505.07773, 2025. 4

- [17] Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In International Conference on Automated Deduction, pp. 625–635. Springer, 2021. 1
- [18] Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, et al. Curriculum reinforcement learning from easy to hard tasks improves llm reasoning. arXiv preprint arXiv:2506.06632, 2025. 1, 4
- [19] Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. Maximizing confidence alone improves reasoning. arXiv preprint arXiv:2505.22660, 2025. 4
- [20] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337, 2024. 4
- [21] ZZ Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, et al. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025. 4
- [22] Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, et al. rstar2-agent: Agentic reasoning technical report. arXiv preprint arXiv:2508.20722, 2025. 4
- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, and Y. Wu. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. 2024. 2.3
- [24] Maohao Shen, Guangtao Zeng, Zhenting Qi, Zhang-Wei Hong, Zhenfang Chen, Wei Lu, Gregory Wornell, Subhro Das, David Cox, and Chuang Gan. Satori: Reinforcement learning with chain-of-action-thought enhances llm reasoning via autoregressive search. arXiv preprint arXiv:2502.02508, 2025. 4
- [25] Vladmir Sicca, Tianxiang Xia, Mathïs Fédérico, Philip John Gorinski, Simon Frieder, and Shangling Jui. Newclid: A user-friendly replacement for alphageometry. arXiv preprint arXiv:2411.11938, 2024. 2.1, B
- [26] Toby Simonds and Akira Yoshiyama. Ladder: Self-improving llms through recursive problem decomposition. arXiv preprint arXiv:2503.00735, 2025. 4
- [27] Joykirat Singh, Raghav Magazine, Yash Pandya, and Akshay Nambi. Agentic reasoning and tool integration for llms via reinforcement learning. arXiv preprint arXiv:2505.01441, 2025. 4
- [28] Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, and Jonas Geiping. The illusion of diminishing returns: Measuring long horizon execution in llms. arXiv preprint arXiv:2509.09677, 2025. 2.2
- [29] Trieu Trinh and Thang Luong. Alphageometry: An olympiad-level ai system for geometry. Google DeepMind, 17, 2024. 1, 1, 4, B, D
- [30] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024. 2.3
- [31] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. 4
- [32] Haiming Wang, Mert Unsal, Xiaohan Lin, Mantas Baksys, Junqi Liu, Marco Dos Santos, Flood Sung, Marina Vinyes, Zhenzhe Ying, Zekai Zhu, et al. Kimina-prover preview: Towards large formal reasoning models with reinforcement learning. arXiv preprint arXiv:2504.11354, 2025. 4
- [33] Zhenting Wang, Guofeng Cui, Yu-Jhe Li, Kun Wan, and Wentian Zhao. Dump: Automated distributionlevel curriculum learning for rl-based llm post-training. arXiv preprint arXiv:2504.09710, 2025. 1, 2.3, 4, D
- [34] Huajian Xin, ZZ Ren, Junxiao Song, Zhihong Shao, Wanjia Zhao, Haocheng Wang, Bo Liu, Liyue Zhang, Xuan Lu, Qiushi Du, et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152, 2024. 4

- [35] Abhay Zala, Jaemin Cho, Han Lin, Jaehong Yoon, and Mohit Bansal. Envgen: Generating and adapting environments via llms for training embodied agents. arXiv preprint arXiv:2403.12014, 2024. 4
- [36] Jingyuan Zhang, Qi Wang, Xingguang Ji, Yahui Liu, Yang Yue, Fuzheng Zhang, Di Zhang, Guorui Zhou, and Kun Gai. Leanabell-prover: Posttraining scaling in formal reasoning. arXiv preprint arXiv:2504.06122,

2025. 4

- [37] Xuemiao Zhang, Liangyu Xu, Feiyu Duan, Yongwei Zhou, Sirui Wang, Rongxiang Weng, Jingang Wang, and Xunliang Cai. Preference curriculum: Llms should always be pretrained on their preferred data. arXiv preprint arXiv:2501.13126, 2025. 1, 4
- [38] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021. 4
- [39] Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084,

2025. 4

### A. The Use of Large Language Models (LLMs)

Beyond these technical roles, LLMs did not play a significant part in research ideation, experimental design, or manuscript writing, the authors conceived the study, designed/evaluated experiments, analyzed results, and wrote the paper. Any automated assistance, if present, was limited to non-substantive copy-editing/LaTeX linting. The authors take full responsibility for all contents, including verifying any machine-generated intermediate artifacts, and acknowledge that LLMs are not eligible for authorship. This disclosure follows the ICLR policy on LLM usage and research integrity.

### B. Improvements in InternGeometry-DDAR

InternGeometry-DDAR builds upon open-sourced symbolic engines (i.e., Newclid [25] and AlphaGeometry [29]) and mainly consist of two components: a deductive database and algebraic reasoning. The former expands the current set of premises toward the proof goal based on geometry rules, while the latter performs angle, length, and ratio chasing using Gaussian elimination. We introduce three main improvements: dynamic diagram adjustment, the incorporation of syntax and rules for handling double points, and the addition of new predicates and rules.

First, open-sourced symbolic engines can only create points one by one based on existing construction definitions, with each point constrained by at most two construction definitions. However, in IMO geometry problem, it is often necessary to make global adjustments to previously constructed points so they satisfy more specific requirements (e.g., a line defined as two existing points may also need to be tangent to a existing circle). Consider IMO 2003 P4a as an example: "Let 𝐴𝐵𝐶𝐷 be a cyclic quadrilateral. Let 𝑃, 𝑄 and 𝑅 be the feet of the perpendiculars from 𝐷 to the lines 𝐵𝐶, 𝐶𝐴 and 𝐴𝐵, respectively. Show that 𝑃𝑄 = 𝑄𝑅 if the bisectors of angles ∠𝐴𝐵𝐶 and ∠𝐴𝐷𝐶 meet on segment 𝐴𝐶". The condition "the bisectors of angles ∠𝐴𝐵𝐶 and ∠𝐴𝐷𝐶 meet on segment 𝐴𝐶" cannot be enforced automatically during point-by-point construction; it is a specialized condition that holds only for certain point placements. Such hard configurations require globally adjusting previously constructed points so that multiple geometric constraints are satisfied simultaneously. To address this issue, we use gradient descent to adjust certain specific points so that they simultaneously satisfy multiple requirements.

Second, we address the issue of double points (i.e., distinctly named points with identical coordinates) by proving that they represent the same geometric point, which is an important technique. Difficult geometry proofs often rely on reformulating intersections or handling degeneracies that naturally create such overlaps—a common trick used by human solvers. To this end, we introduce new syntax: prefixing a construction statement with ! allows the system to create a point even if it shares coordinates with an existing one. In addition, we update the inference module of the symbolic engine to support this extended behavior. We also introduce new predicates and rules for handling double points. Specifically, we define the predicate idc x y to indicate that points 𝑋 and 𝑌 are geometrically considered the same, and we provide rules for determining this relationship.

Additionally, we add several common geometry theorems, such as the Power of a Point and Menelaus’ theorem into InternGeometry-DDAR.

InternGeometry-DDAR serves both as an automated geometry problem solver and as a powerful symbolic tool for InternGeometry.

### C. Interaction between InternGeometry and InternGeometry-DDAR

InternGeometry has three interaction with InternGeometry-DDAR, which includs obtain initial state in symbolic engine, adding auxiliary construction and proposing proof steps. Specifically, when giving a formal geometry problem, InternGeometry first output ‘<build>’ tag to construct this problem and retrieve the initial geometric relationships from the symbolic engine. Then, in the following turn, InternGeometry performs thinking and then automatically decides whether to add an auxiliary construction using ‘<add>’ tag or propose a proof step using ‘<propose>’. InternGeometry-DDAR executes the instructions from InternGeometry and returns feedback, such as successfully proving the proposed proposition or reporting a failure when adding a new point. After each step, InternGeometry summarizes and compresses the current proof state to support long-horizon interaction. When the final goal is proven, InternGeometry reviews the entire proof and briefly summarizes the

reasoning process, extracting key steps and auxiliary constructions.

### D. Details of Complexity Boosting Reinforcement Learning (CBRL)

We provide a detailed introduction to CBRL in this section, as the main paper is space-constrained. As outlined in Subsection 2.3, CBRL adapts task complexity to maximize the expected absolute advantage during reinforcement learning. The core intuition is to present tasks that are neither too difficult nor too easy for the policy model—i.e., tasks that best match the model’s current capability. Following AlphaGeometry [29], which observed that human-perceived difficulty of IMO geometry problems (measured by average IMO scores) correlates positively with the number of proof steps taken by the DDAR solver, we quantify task complexity using DDAR proof length.

We first introduce the geometry question generation algorithm, then describe the data synthesis pipeline. Finally, we present the CBRL algorithm and explain its motivation.

The question generation procedure, Generate Data (Algorithm 1), aims to sample nontrivial geometry questions that cannot be solved by exhaustive search using the InternGeometry-DDAR engine alone. The algorithm iteratively samples a raw geometric structure, 𝑋𝑟𝑎𝑤, by randomly instantiating DDAR predicates and points. It then augments this structure with auxiliary constructions to obtain 𝑋𝑎𝑑𝑑. Both stages are conditioned on a user-specified complexity parameter 𝜅, for which we design distinct priors and construction patterns to improve the hit rate of valid problems. We perform exhaustive search on both 𝑋𝑟𝑎𝑤 and 𝑋𝑎𝑑𝑑. Conclusions that (i) involve only points in 𝑋𝑟𝑎𝑤, (ii) are provable in 𝑋𝑎𝑑𝑑, and (iii) are not provable in 𝑋𝑟𝑎𝑤, are deemed nontrivial problems rooted in the raw structure. Among these candidates, we select the most complex one—primarily by proof length, while also considering priors such as predicate distributions. The algorithm repeats to continuously generate questions targeting the specified complexity prior 𝜅. Because the actual complexity of generated items is only loosely controlled by 𝜅 via stochastic priors, we apply additional post-sampling to better align the dataset with the desired complexity.

- Algorithm 1 Generate Data Require: Complexity 𝜅

- 1: for _ in range(MaxSample) do
- 2: 𝑋raw ← RandDDARConstruction(𝜅)
- 3: 𝑋add ← AddAuxConstructions(𝑋raw,𝜅)
- 4: 𝒫raw ← ExhaustSearch(𝑋raw)
- 5: 𝒫add ← ExhaustSearch(𝑋add)
- 6: 𝒞 ← ,𝑐 ∈ 𝒫add ∖ 𝒫raw & conclusion of 𝑐 uses only points in 𝑋raw,
- 7: if 𝒞 ̸= ∅ then
- 8: data⋆,pl⋆ ← SelectMostComplex(𝑋raw,𝒞) ◁ primarily by proof length
- 9: yield data⋆,pl⋆
- 10: end if
- 11: end for

- Algorithm 2 Data Synthesis Pipeline X Require: Complexity 𝜅, Data Number 𝐾

- 1: global 𝐶 ◁ cache: list of (data,proof length)
- 2: ok,dataset ← SelectAroundRange(𝐶,𝜅,𝐾)
- 3: while not ok do
- 4: for (data,pl) in GenerateData(𝜅) do
- 5: 𝐶.append((data,pl))
- 6: end for
- 7: ok,dataset ← SelectAroundRange(𝐶,𝜅,𝐾)
- 8: end while
- 9: return dataset ◁ exactly 𝐾 items around target complexity

Next, Algorithm 2 presents the complete Data Synthesis Pipeline. We maintain a global cache of all generated

questions. Given a target complexity 𝜅 and a required sample count, we first check whether the cache already contains enough items within a tolerance range around 𝜅. If not, we invoke Generate Data to enrich the cache until we can retrieve a sufficient number of questions at the desired complexity.

Finally, we introduce Complexity Boosting Reinforcement Learning (CBRL) in Algorithm 3. The idea is straightforward: in each iteration, we sample a batch of data from the Data Synthesis Pipeline, run RL on that batch, and record rewards. After processing the batch, we compute the average reward and update 𝜅 by comparing this average to 0.5: increase 𝜅 if the average reward exceeds 0.5, otherwise decrease it. Below, we justify why using 0.5 as the target lead to optimized expected absolute advantage, as indicated by Theorem 2.2.

For a binary reward with 𝑃(𝑟 = 1) = 𝑝 and 𝑃(𝑟 = 0) = 1−𝑝, we have mean(𝑟) = 𝑝 and std(𝑟) = √︀𝑝(1 − 𝑝). The expected absolute advantage is

]︃ = 𝑝 ·

E[|𝐴𝑖|] = E[︃

= 2√︀𝑝(1 − 𝑝). (10)

𝑟𝑖 − 𝑝 √︀𝑝(1 − 𝑝)⃒

1 − 𝑝 √︀𝑝(1 − 𝑝)

𝑝 √︀𝑝(1 − 𝑝)

+ (1 − 𝑝) ·

⃒

This quantity is concave in 𝑝 on [0,1] and is maximized at 𝑝 = 0.5. Therefore, we can maximize the expected absolute advantage by steering the average reward toward 0.5 via 𝜅-adjustment.

- Algorithm 3 Complexity Boosting Reinforcement Learning (CBRL) Require: Initial complexity 𝜅, Initial policy parameter 𝜃

- 1: for 𝑡 in range(MaxIter) do
- 2: batchdata ← X(𝜅,DataNumEachIter)
- 3: batchrewards ← [,]
- 4: for minibatch in SplitMiniBatch(batchdata) do
- 5: rewards,𝜃 ← EvaluateAndUpdate(𝜃,minibatch)
- 6: batchrewards.extend(rewards)
- 7: end for
- 8: if Average(batchrewards) > 0.5 then
- 9: 𝜅 ← 𝜅 + 𝛼
- 10: else
- 11: 𝜅 ← 𝜅 − 𝛼
- 12: end if
- 13: end for
- 14: return 𝜃

We further explain why maximizing the expected absolute advantage benefits policy learning [4, 33], as stated in Theorem 2.1. The key idea is that the expected absolute advantage serves as the primary learning signal that scales the gradient. The simplified gradient of the GRPO objective is

[︀

]︀

(11) Thus, the gradient norm is

∇𝜃𝒥 (𝜃) = E𝑥∼X(𝜅)

E𝑦

𝑖∼𝐺𝜃(·|𝑥) [𝐴𝑖 · ∇𝜃 log 𝐺𝜃 (𝑦𝑖 | 𝑥)]

[︀

]︀

⃦ (12)

‖∇𝜃𝒥 (𝜃)‖ = E𝑥∼X(𝜅) ⃦

E𝑦

𝑖∼𝐺𝜃(·|𝑥) [𝐴𝑖 · ∇𝜃 log 𝐺𝜃 (𝑦𝑖 | 𝑥)]

Assuming that the gradient term ∇𝜃 log 𝐺𝜃 (𝑦𝑖 | 𝑥) is bounded and approximately random in direction, the gradient norm is approximately maximized by maximizing E𝑋∼X(𝜅)E𝑦∼𝐺

𝜃|𝐴(𝑋,𝑦)|. In other words, increasing the expected absolute advantage yields larger gradients during optimization and thus can accelerate learning.

### E. Details of Training Token

We report the total number of training tokens and a token-based comparison to prior work. Our model totally trains approximately 1.91 × 109 tokens. For reference, AlphaGeometry 2 reports training on up to 1 × 1012 tokens. Framed in terms of training tokens, InternGeometry is therefore substantially more data-efficient.

### F. Failed Cases

We illustrate failure cases of InternGeometry on IMO 50 in this section. Notably, the unsolved problems are largely beyond pure geometry, and they primarily rely on numerical or non-geometric analysis.

- 2001 P1

Let 𝐴𝐵𝐶 be an acute-angled triangle with 𝑂 as its circumcenter. Let 𝑃 on line 𝐵𝐶 be the foot of the altitude from 𝐴. Assume that ∠𝐵𝐶𝐴 ≥ ∠𝐴𝐵𝐶 + 30∘. Prove that ∠𝐶𝐴𝐵 + ∠𝐶𝑂𝑃 < 90∘.

- 2002 P6

Let 𝑛 ≥ 3 be a positive integer. Let 𝐶1,𝐶2,...,𝐶𝑛 be unit circles in the plane, with centers 𝑂1,𝑂2,...,𝑂𝑛 respectively. If no line meets more than two of the circles, prove that

∑︁

1≤𝑖<𝑗≤𝑛

1 𝑂𝑖𝑂𝑗 ≤

(𝑛 − 1)𝜋 4

.

- 2003 P3

Each pair of opposite sides of a convex hexagon has the property that the distance between their midpoints is

√3 2

times the sum of their lengths. Prove that the hexagon is equiangular.

2006 P1

Let 𝐴𝐵𝐶 be a triangle with incenter 𝐼. A point 𝑃 in the interior of the triangle satisfies

∠𝑃𝐵𝐴 + ∠𝑃𝐶𝐴 = ∠𝑃𝐵𝐶 + ∠𝑃𝐶𝐵. Show that 𝐴𝑃 ≥ 𝐴𝐼 and that equality holds if and only if 𝑃 = 𝐼.

2006 P6

Assign to each side 𝑏 of a convex polygon 𝑃 the maximum area of a triangle that has 𝑏 as a side and is contained in 𝑃. Show that the sum of the areas assigned to the sides of 𝑃 is at least twice the area of 𝑃.

2020 P6

Consider an integer 𝑛 > 1, and a set 𝑆 of 𝑛 points in the plane such that the distance between any two different points in 𝑆 is at least 1. Prove there is a line ℓ separating 𝑆 such that the distance from any point of 𝑆 to ℓ is at least Ω(𝑛−1/3). (A line ℓ separates a set of points 𝑆 if some segment joining two points in 𝑆 crosses ℓ.)

### G. Discussion of Inference Cost

Because neither AlphaGeometry 2 nor SeedGeometry has released open-source code or models, we cannot perform a direct, controlled comparison. Therefore, we rely on the configurations reported in their respective papers to provide a rough estimate and comparison. Since the SeedGeometry paper does not detail its inference budget, we primarily base our equivalent estimates on AlphaGeometry 2.

According to the AlphaGeometry 2 paper, the method uses a Shared Knowledge Ensemble of Search Trees (SKEST) that integrates classical beam search with several tree-search variants (e.g., predicting multiple auxiliary points at each node) and runs multiple trees with different search configurations and models in parallel. Consequently, the total inference budget scales with the number of configurations × the number of instantiated trees × the number of models. The reported optimal single-tree configuration uses a beam size of 128, a branching number (samples) of 32, and a beam depth of 4. However, within SKEST there are configurations that incur higher budgets, such as beam size 64 with depth 10 or beam size 512 with depth 4. The AlphaGeometry 2 model size is 3.3B parameters.

We compare inference efficiency from four perspectives: the equivalent number of solutions explored during inference, the overall inference steps, the environment execution cost, and the overall computation cost.

- 1. Equivalent number of solutions explored. For a single beam search, the equivalent number of explored

solutions can be approximated as beam size × branching number. Under the optimal single-tree configuration, this is 128 × 32 = 4,096, and it can be larger for other configurations (e.g., beam size 512). The total number further scales with the number of configurations × the number of instantiated trees × the number of models. By contrast, InternGeometry’s best-of-K inference explores only K solutions (256 in our experiments), addressing its solution-efficiency.

- 2. Inference steps. The optimal beam-tree configuration results in 128 × 32 × 4 = 16,384 steps, and the

same number of symbolic-engine executions per tree. Other configurations, such as beam size 64 and depth 10 (64 × 32 × 10 = 20,480) or beam size 512 and depth 4 (512 × 32 × 4 = 65,536), can require more steps. InternGeometry uses a simple pass@256 parallel-inference setting with up to 200 turns of agentic interaction per pass, totaling 51,200 steps. Overall, the per-tree inference budget of AlphaGeometry 2 is on the same order as InternGeometry. However, the total inference cost for AlphaGeometry 2 scales with the number of configurations × the number of instantiated trees × the number of models, leading to a larger overall inference steps.

- 3. Environment execution cost. Each step in both AlphaGeometry 2 and InternGeometry is executed in an engine. Because InternGeometry’s total number of steps is smaller than the full SKEST budget of AlphaGeometry 2, it yields fewer total executions. Furthermore, each AlphaGeometry 2 step requires the DDAR system to attempt solving the entire problem, whereas each InternGeometry step either adds an auxiliary construction or attempts to prove a subgoal—operations that are less expensive than attempting to solve the whole problem.
- 4. Overall computation cost. InternGeometry’s reasoning style and larger model size indeed increase computation. Each InternGeometry step involves natural-language reasoning, resulting in more tokens and higher compute: for IMO-50, the average number of output tokens per trajectory is 89.6K. The InternGeometry model (32B) is also larger than AlphaGeometry 2 (3.3B). Due to the unknown total cost of AlphaGeometry 2, a direct comparison is difficult. However, we emphasize that the increased computation from deeper reasoning and larger models should not be viewed as a drawback. Instead, it represents a feasible new scaling dimension—alongside training data size and the number of searched solutions—one that aligns more naturally with LLM-based approaches than with expert-model diagram.

