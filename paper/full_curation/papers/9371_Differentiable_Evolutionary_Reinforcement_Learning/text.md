# arXiv:2512.13399v2[cs.AI]13May2026

## Differentiable Evolutionary Reinforcement Learning

Sitao Cheng1∗, Tianle Li2∗, Xuhan Huang3∗, Xunjian Yin4, Difan Zou2 1University of Waterloo 2The University of Hong Kong 3The Chinese University of Hong Kong, Shenzhen 4Duke University sitao.cheng@uwaterloo.ca tianleli@connect.hku.hk xuhanhuang@link.cuhk.edu.cn

### Abstract

Crafting effective reward signals remains a central challenge in Reinforcement Learning (RL), especially for complex reasoning tasks. Existing automated reward optimization methods typically rely on derivative-free search heuristics that treat the reward function as a black box, failing to exploit the causal dynamics between reward structure modifications and policy performance. We introduce Differentiable Evolutionary Reinforcement Learning (DERL), a bi-level framework for the autonomous discovery of optimal reward structures. DERL employs a MetaOptimizer that evolves a reward function through the composition of structured atomic primitives to guide an inner-loop policy. Unlike prior black-box methods, DERL introduces differentiability into the meta-optimization process by updating the Meta-Optimizer using policy gradients derived from inner-loop validation performance. This allows for the progressive learning of a “meta-gradient” for task success, providing the system with dense, actionable feedback. We validate DERL across diverse reasoning domains: embodied agent (ALFWorld), scientific simulation (ScienceWorld), and mathematical reasoning (GSM8K, MATH). Results show that DERL achieves state-of-the-art performance on agent benchmarks, substantially outperforming non-differentiable baselines—especially in out-of-distribution generalization. Trajectory analyses confirm that DERL captures the intrinsic causal structure of tasks, enabling fully autonomous, self-improving agent alignment.

### 1 Introduction

The efficacy of reinforcement learning (RL) hinges fundamentally on the quality of the reward signal, the lens through which an agent perceives its environment and learns desirable behaviors [1, 2]. However, crafting optimal rewards remains a persistent bottleneck. In complex reasoning tasks, outcome signals are often too sparse to drive learning over long horizons, while manual reward design is highly susceptible to “reward hacking” [3, 4]. Although dense reward models can mitigate these issues [5], they incur prohibitive human annotation costs.

To bypass manual engineering, recent studies have pivoted toward the automatic evolution of agent configurations (e.g., rewards, prompts) using genetic algorithms via stochastic mutations [6, 7] or prompted LLM agents [8–10]. A critical limitation of these approaches is that they treat the configuration as a black box. By relying on derivative-free perturbations, they blindly traverse the optimization landscape and fail to systematically capture a non-arbitrary structure—the causal relationship between structural reward modifications and the resulting shifts in agent performance.

When tuning a system, human experts intuitively maintain a meta-gradient—the consciousness that a specific reward configuration change will yield a specific behavior shift [11]. We hypothesize that language models are able to capture the meta-gradient to update their own parameters, generating progressively better reward functions. To this end, we propose Differentiable Evolutionary Reinforcement Learning (DERL) for autonomous discovery of optimal objectives (Figure 1). Rather

Preprint.

###### Traditional Reward Design Black-box Evolution Bi-level Differentiable Evolution (DERL)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Heuristic Reward or

[Figure 6]

Reward Model

Prompted Agent

Meta-Optimizer Evolve

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Policy Model Evolve

Policy Model Evolve

Policy Model Evolve

Decreasing Human Labor; Increasing Automation; Increasing Scalability

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Fixing parameters Tuning parameters

Reward Function

- Figure 1: Illustration of DERL. Comparison: Traditional reward involves human labor to design a function or train a reward model. With LLMs, specific instruction is designed to evolve the agent configuration based on execution feedback. In our DERL, a Meta-Optimizer generates a parameterized Meta-Reward to guide the policy model evolution via policy gradients derived from validation performance, establishing a differentiable, closed-loop optimization and eliminating heuristic design or expensive annotation. Performance: Performance comparison of our Meta-Reward with outcome reward, average reward over atomic primitives. Meta-Reward consistently outperforms all baselines in different tasks, demonstrating the effectiveness of DERL.

than relying on heuristic mutations or assuming analytic end-to-end differentiability, DERL formulates reward generation as a continuous, higher-level RL problem. It features a bi-level evolutionary training framework: an inner-loop where the policy evolves based on a generated Meta-Reward, and an outer-loop where a parameterized Meta-Optimizer evolves by learning from the inner policy’s validation performance (Figure 2). By policy gradients, the Meta-Optimizer captures an estimated “meta-gradient” of task success, allowing the model to update its own parameters to synthesize progressively denser and more actionable feedback.

The instantiation of DERL addresses two key challenges: a tractable search space and a robust outerloop evolution signal. First, rather than generating arbitrary code, our Meta-Optimizer constructs rewards by mathematically composing atomic primitives–modular, executable blocks like format checkers or partial goal verifiers. This structural constraint enables the system to focus on logical reward composition rather than text parsing. Second, by using the inner-loop policy’s validation performance as the direct feedback signal, the Meta-Optimizer approximates the gradient of task success, learning to refine Meta-Rewards via policy gradient [12].

We validate DERL across diverse reasoning domains: Robotic Agent (ALFWorld [13]), Scientific Simulation (ScienceWorld [14]), and Mathematical Reasoning (GSM8k [15], MATH [16]). Results demonstrate that DERL generalizes effectively across diverse tasks, consistently outperforming other methods relying on human-designed heuristics. Notably, DERL achieves state-of-the-art performance on ALFWorld and ScienceWorld, showing superior robustness in Out-of-Distribution (OOD) problems. Further analysis of evolutionary process reveals that the Meta-Optimizer successfully captures the meta-gradient: as training progresses, our Meta-Rewards evolve to encode the intrinsic logic of the tasks, demonstrating a self-exploratory capability that aligns with the true gradient of optimization. Our contributions are summarized as follows:

- • We introduce DERL, a bi-level training framework that moves beyond black-box search for automatic reward modeling. By framing meta-optimization as an RL problem, the Meta-Optimizer captures an estimated meta-gradient between reward structural changes and task success, allowing it to update its own parameters to refine the reward generation.
- • We present a novel reward parameterization strategy by composing atomic primitives and a robust evolution signal for Meta-Optimizer from inner-loop validation performance. This design alleviates manual annotation while ensuring a highly structural and expressive reward search space.
- • Empirical Results across three domains demonstrate that DERL achieves strong performance, state-of-the-art on ALFWorld and ScienceWorld, exhibiting superior robustness in OOD scenarios. Evolutionary trajectory analysis confirms that the Meta-Optimizer evolves and learns to generate numerically stable and intrinsically sound reward structures without human intervention.

### 2 Differentiable Evolutionary Reinforcement Learning (DERL)

We first introduce the formulation of the DERL framework, modeling reward design as a bi-level optimization process. Then, we detail the reward parameterization and implementation of training.

|Outer-Loop Evolution with Meta-Optimizer 𝝍|
|---|

𝑇𝑎𝑠𝑘 𝐼𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛

𝑹𝝓𝟏

|Inner-Loop for Policy Model 𝜽𝒊 with Meta-Reward 𝑹𝝓𝒊|
|---|

𝜱

𝑹𝝓𝟐

Meta-Optimizer 𝜓

…

###### KL

𝑹𝝓𝒏 Meta-Reward

- 𝐼𝑛𝑛𝑒𝑟𝑙𝑜𝑜𝑝(𝑅𝜙1)

- 𝐼𝑛𝑛𝑒𝑟𝑙𝑜𝑜𝑝(𝑅𝜙2)

𝑞

Reference Model for 𝜽𝒊

Group Computation

- 𝑟1

…

- 𝑟2 𝑟𝑛

- 𝐴1

…

- 𝐴2 𝐴𝑛

…

- 𝑜1

…

- 𝑜2 𝑜𝑛

Reward Function 𝓡𝒊

Policy Model 𝜽𝒊

- 𝐴𝑑𝑣1

…

- 𝐴𝑑𝑣2

- 𝑣1

…

- 𝑣2

𝐼𝑛𝑛𝑒𝑟𝑙𝑜𝑜𝑝(𝑅𝜙𝑛)

Group Computation

Evaluation

𝑣𝑛

𝐴𝑑𝑣𝑛

Validation Performance

Advantage of Meta-Reward

- Figure 2: Bi-level evolutionary training framework for DERL. Blue Block: Evolution of MetaOptimizer ψ with n generated Meta-Rewards R (i.e., rollouts). Taking a fixed task instruction as input, ψ updates the parameter Φ of R with the signal from validation performance v. Green Block:

The inner-loop training for policy model θi with Meta-Reward Rϕ

by GRPO. We evaluate the validation performance for each θi as the reward of Rϕ

i

, making it a differentiable signal for ψ to evolve through reinforcement learning.

i

#### 2.1 Bi-Level Evolutionary Training

Recent efforts to automate reward design predominantly treat the reward function as a static configuration optimized via discrete evolutionary search [17], primarily relying on genetic algorithms by stochastic mutations [18, 19] or heuristic prompt engineering via LLM agents [20, 21]. Functioning essentially as zero-order optimizers, these methods navigate the search space blindly; they fail to capture the causal relationship between reward modifications and policy performance, resulting in sample inefficiency akin to grid search.

Mitigating this requires reformulating the discrete search into a continuous, differentiable optimization process, thereby enabling gradient-guided evolution to capture the meta-gradients to guide searching. Inspired by RL-based discovery of optimization algorithms [22], we introduce DERL, a bi-level evolutionary training framework (Figure 2). In DERL, the outer-loop (level) optimizes a MetaOptimizer ψ via RL to generate a configuration ϕ (which parameterizes the reward function Rϕ, detailed in Section 2.2), while the inner-loop (level) optimizes a policy model θ under the generated reward function Rϕ. Specifically, the bi-level optimization problem is formulated as follows:

Inner-loop (Policy Model Evolution) Given a configuration ϕ from the outer-loop, the inner-loop policy θ is optimized to maximize the expected parameterized reward:

Jϕinner(θ) = Ex∈D,τ∼π

θ(·|x)[Rϕ(τ)], (1) where D represents the training dataset and τ denotes the trajectories sampled from the policy model. Outer-loop (Meta-optimizer Evolution) The objective is to train the Meta-Optimizer ψ to generate configurations ϕ that yield high-performing inner policies. We postulate that the data-driven policy gradient can automate the optimization of reward functions. Formally, the Meta-Optimizer acts as a generator policy πψ(·|ins), taking a fixed task instruction ins as input and outputting the configuration ϕ that instantiates the Meta-Reward Rϕ (Equation (1)). Once the inner policy converges to an optimal θ∗ under Rϕ, it is evaluated using the performance metric Perf(·) (e.g., validation accuracy). This metric serves as the outer feedback signal (i.e., the meta-reward) for the MetaOptimizer ψ. Consequently, the Meta-Optimizer aims to maximize the following bi-level objective:

J outer(ψ) = Eϕ∼π

Jϕinner(θ). (2) The concrete estimator of Perf(·) and its data protocol are specified in Section 2.2.

ψ(·|ins)[Perf(θ∗)], s.t. θ∗ = arg max

θ

This bi-level training framework transforms the discrete evolutionary search of reward functions into a continuous, differentiable optimization over the meta-policy parameters ψ. Unlike genetic algorithms that rely on stochastic mutations (blindly searching the space), DERL learns the “search direction”, transforming the zero-order reward search into a first-order gradient-guided evolution.

#### 2.2 Instantiation of DERL

Reward Parameterization A key challenge in Meta-Optimizer evolution lies in designing a tractable structure of the generated configuration to parameterize the reward function for the inner-

loop. Traditional approaches rely on heuristic scalar functions, which often yield sparse signals [12], or trained reward models, which incur prohibitive annotation costs [1]. To circumvent these issues without introducing the intractable search space of arbitrary function generation, our Meta-Optimizer generates a configuration ϕ—a symbolic formulation dictating the structure and weights for composing multiple atomic primitives to evaluate the inner-loop agent.

Let G = {g1,g2,...,gk} denote a set of atomic primitives that evaluate specific aspects of a policy’s output o given context C (e.g., the query q and ground-truth a∗). Example primitives include binary outcome correctness (i.e., comparing o to a∗), adherence to formatting rules in C, or process heuristics like step counts. The Meta-Optimizer ψ predicts the structure and weights (ϕ) that mathematically compose these signals instead of generating an arbitrary function. The reward function is therefore defined as:

##### Rϕ(o,C) = Func(g1(o,C),...,gk(o,C);ϕ) (3)

where Func(·) applies the weights and mathematical operators (e.g., summation, conditional logic) specified by ϕ. This parameterization ensures an expressive, continuous search space while focusing the Meta-Optimizer on structural reasoning rather than raw text parsing. Furthermore, by decoupling the definition of atomic signals from their utilization, the framework generalizes seamlessly to new tasks; the evolutionary process autonomously filters and weights components, discarding detrimental heuristics without requiring manual validation. See more details and examples in Appendix A.

Inner-loop (Policy Model Evolution) We utilize GRPO for policy model optimization of the inner-loop (Green block in Figure 2). The objective is defined as:

G

πθ(oi | q) πθ

πθ(oi | q) πθ

1 G

###### E

, 1 − ϵ, 1 + ϵ Ai − β DKL(πθ∥πref)

min

Ai, clip

(oi | q)

(oi | q)

q∼P(D), {oi}Gi=1∼πθold(·|q)

old

old

i=1

(4) where we sample a group of G outputs {oi}Gi=1 from the old policy πθ

for each question q from the training set D. πθ and πθ

old

denotes the current and previous policy. ϵ and β denotes hyperparameters for the clipping range and KL-divergence penalty against a reference policy πref, with DKL detailed in Shao et al. [12]. Crucially, Ai represents the group-wise advantage derived from our parameterized reward function Rϕ. For a given context C and computed reward ri = Rϕ(oi,C), the advantage is defined as:

old

ri − mean({rj}Gj=1) std({rj}Gj=1)

, where ri = Rϕ(oi,C). (5)

Ai =

To investigate model plasticity and optimization efficiency, we implement two initialization strategies for inner-loop: 1) Standard Init (denoted as DERL): The policy θ resets to the original base model for each outer-loop iteration, ensuring that performance improvements are strictly attributable to the current reward configuration ϕ. 2) Population-based Variant (denoted as DERL-Pop.): In the first inner-loop, we initialize the policy model θ from the base model. Subsequently, the policy initializes from the highest-performing checkpoint of the previous iteration. This is similar to population-based evolution where the policy model is evolved from different training configurations [7, 8], but our Meta-Optimizer captures the meta-gradient in a differentiable dynamic. For fair comparison, we ensure the total inner-loop training step of DERL-pop. remains the same as standard DERL, saving computation while allowing the outer-loop to evolve more frequently and dynamically.

Outer-loop (Meta-optimizer Evolution) We similarly employ GRPO to optimize the outer-loop (Blue block in Figure 2). In each iteration, we sample a group of n (i.e., rollout size) reward configurations Φ = {ϕ1,ϕ2,...,ϕn}. To evaluate these configurations, we partition the original training data into an inner-training split Din and a validation split V; the D in Eq. (1) is instantiated as Din for each inner-loop update. Each configuration ϕi instantiates a reward function Rϕ

to train a corresponding inner policy θi on Din. Upon completion of the inner training, we evaluate each policy θi on V to compute the performance score vi = Perf(θi;V) using the pass@1 accuracy:

i

1 |V|

I(fθ(q) = a∗), (6)

Perf(θ;V) =

(q,a∗)∈V

where fθ(q), a∗ and I(·) denote output of the trained policy πθ with deterministic decoding, ground truth, and indicator function, respectively. The validation scores {v1,...,vn} act as outer feedback signals for the configurations {ϕ1,...,ϕn}, allowing us to compute the group-wise advantage to

update the Meta-Optimizer parameters ψ. Notably, V is the feedback signal of the outer-loop RL environment, not for final reporting; final evaluation is conducted strictly on held-out test data.

To guarantee the mathematical validity of the generated symbolic configurations, we apply supervised fine-tuning (SFT) on a small set of valid examples to cold-start the Meta-Optimizer, enforce tokenlevel constrained decoding during generation, and assign a penalty reward of vi = 0 to any ill-defined ϕi that causes execution errors.

Crucially, DERL establishes a closed-loop computation graph that propagates non-differentiable validation signals back to the Meta-Optimizer via estimated meta-gradients (detailed in Appendix I.1). This RL-driven feedback loop replaces manual heuristics with an autonomous optimization path, enabling the composition of dense, actionable signals from sparse outcomes, effectively resolving scalability challenges in complex domains.

### 3 Experiments

We experiment on three reasoning domains: Robotic agents, Scientific Simulation and Mathematical Reasoning. We address two research questions: 1) Does DERL discover reward functions better than heuristic signals? 2) Does the learned reward generalize better to OOD scenarios?

#### 3.1 Experiment Setups

Robotic Agent We adopt a multi-round robotic task–ALFWorld [13]–requiring the agent to complete embodied household tasks with natural language or visual observations. To rigorously assess the generalization capability, similar to [23], we evaluate on three difficulty levels based on the distribution shift of training and testing data: L0 (in-distribution, seen): trained on all 6 task types and evaluated on seen variants; L1 (in-distribution, unseen): trained on all 6 task types but evaluated on unseen variants; L2 (out-of-distribution), trained only on 4 task types and evaluated on the remaining 2 unseen types. Appendix B explains the settings in detail with examples.

We compare DERL with standard RL baselines: 1) GRPO + Out.: GRPO with binary outcome rewards. 2) GRPO + Avg.: As we introduce the atomic primitives, a common practice is to calculate the weighted sum over all functions. To demonstrate DERL’s exploration of an optimal reward structure over the search space, we compare with the average weighted sum. 3) GiGPO [24] with a two-level structure for finer-grained credit assignment. 4) RLVMR [23] with structured verifiable process reward, which is the previous state-of-the-art method. We do not compare with LLM-based reward models due to their high resource intensity, particularly since ground-truth outcome signals are readily available for the target domains.

Scientific Simulation We adopt ScienceWorld [14], an interactive text environment at the level of a standard elementary school science curriculum, testing the agents’ scientific reasoning abilities. To ensure consistent evaluation of generalization, we evaluate on the three difficulty levels (i.e., L0, L1, and L2) and compare against the same set of baselines as in the Robotic Agent tasks.

Mathematical Reasoning We evaluate DERL on two established benchmarks: GSM8K [15] for grade-school math and MATH [16] for advanced competition-level problems. We vary the training data by using either the MATH training set, which contains more difficult math problems, or combining the MATH and GSM8K, which contains both easy and hard problems.

We benchmark against a set of baselines varying in reward structure: 1) Outcome: a standard binary outcome-based reward; 2) Outcome + Format: the outcome rewards augmented with format reward; and 3) Avg Reward: the average reward over all individual atomic primitives. Beyond these main baselines, Appendices F.1 and F.2 compare with RLAIF (Reinforcement Learning from AI Feedback) and random-search. Appendix B summarizes the shared experimental protocol, default training settings, and additional setup details across domains.

#### 3.2 Implementation Details

Data split and evaluation protocol. In all experiments, the outer-loop tuning split in Section 2.2 uses an 8:2 ratio for Din and V. After tuning, we train a fresh final policy on the full Dtrainorig with the same step budget as baselines and report only on the held-out test set Dtest; thus V is not part of

the final evaluation. Especially, in the final comparison, DERL does not receive extra final-policy training data or steps beyond the compared baselines. The additional outer-loop computation is used only to tune the reward configuration and is accounted for separately in Appendix H.

Robotic Agent and Scientific Simulation We introduce four atomic primitives to construct the reward search space. The first is binary outcome reward. Others are captured from three stages of the interaction trajectory, inspired by [23]. Specifically, we compute the average reward over the first, middle and last third of stages of the interaction trajectory, respectively. For instance, given a six-step interaction with a step-wise reward sequence of [1,0,1,1,0,0], the atomic primitives corresponding to the three temporal stages yield values of 0.5, 1, and 0, respectively. This straightforward design incentivizes the model to attend to distinct temporal phases of the task. We describe how to obtain this step-wise reward in Appendix A.2.

We implement with GRPO via VeRL [25]. For the outer loop, we use Qwen-2.5-0.5B-Instruct as the Meta-Optimizer with a rollout size of 8. The ablation experiments on the number of rollouts in the Appendix F.4 demonstrate that DERL is not sensitive to this parameter. Other hyperparameters use the default settings. For the inner loop, we employ Qwen2.5-1.5B-Instruct as the base policy with a cold start (same as other baselines). We set the number of epochs to 40 for ALFWorld and 80 for ScienceWorld, respectively. After obtaining the optimal Meta-Reward, we train the policy model (from scratch) using this reward function for 100 steps, the same as RLVMR, whereas other baselines are trained for 150 steps. The Meta-Optimizer achieves convergence in approximately ten and five outer-loop iterations for ALFWorld and ScienceWorld, respectively. For DERL-pop., we train the outer-loop for 10 rounds and set the training epochs for inner-loops to 10 on ALFWorld. On ScienceWorld, we train the outer-loop for 3 rounds and the inner-loop for 33 rounds, to ensure the consistency of the total training epochs for the inner-loop. We report the success rate on the test set, i.e., whether the model can ultimately complete the task.

Mathematical Reasoning We construct the reward space with four straightforward atomic primitives: 1) Binary outcome reward; 2) Format reward which verifies if the answer is enclosed in “boxed{}”;

- 3) Step-by-step reward, identifying whether the output contains CoT tokens e.g., “step 1”; 4) Soft outcome reward, which credits the presence of ground truth anywhere in the output, helpful when the model indeed knows the answer but generates in a wrong format.

We want to emphasize that while DERL relies on artificially constructed atomic primitives, these do not need to be elaborately designed. Very simple and intuitive atomic primitives are sufficient to make the DERL framework effective. In the Appendix F.3, we empirically demonstrate that DERL is not sensitive to primitive vocabulary.

For the outer loop, we keep all configurations the same as other tasks. For the inner loop, we adopt Qwen-2.5-3B as the base policy model. We train for 10 epochs and enforce a time limit of 3.5 hours for inner-loop training. With these settings, the Meta-Optimizer achieves convergence in approximately 8 outer-loop iterations. We then take the optimal Meta-Reward to train the base policy for 15 epochs for fair comparison with baselines. For DERL-pop., we train each inner-loop for 2 epochs and report the testing result after 7th outer-loop iteration for fair comparison. We evaluate the exact match of the ground truth answer in the test set.

#### 3.3 Results

Robotic Agent and Scientific Simulation Table 1a presents the performance on Robotic Agent and Scientific Simulation benchmarks across three generalization levels. We observe two key findings:

- 1) State-of-the-Art (sota) Performance. DERL achieves sota success rates across all difficulty levels on both benchmarks. This indicates that the Meta-Optimizer effectively explores the function space to discover Meta-Rewards that drive policy improvement beyond standard outcome signals. Notably, our population-based instantiation, DERL-pop., further demonstrates exceptional performance, achieving 91.8% on ALFWorld (L0) and 98.2% on Science World (L0). This shows that by initializing the inner-loop policy with the best-performing model from previous generations, the Meta-Optimizer can effectively adapt the reward signal dynamically as the policy model evolves, creating a curriculumlike effect that accelerates convergence and elevates the performance ceiling. It is worth noting that DERL-pop. consumes significantly less computation by optimizing the outer-loop more frequently. Appendix D details the training dynamics of DERL-pop.

Table 1: Performance of DERL on three distinct domains. Bold and underline denote the best and second best performance.

(b) Accuracy on GSM8K and MATH (Qwen-2.5-3B) with different training data and reward functions. Our DERL outperforms all baselines, including the outcome reward, outcome + format reward, and the average reward over atomic primitives. All results are run under the same configuration.

(a) Success rates on ALFWorld and ScienceWorld (Qwen2.5-1.5B-Instruct) across three levels of generalization difficulty (Section 3.1). Out. and Avg. denotes outcome reward and average reward over all atomic primitives. GRPO baselines are run by ourselves. Other results are from [23]. Our DERL outperforms all baselines in all difficulty levels, achieving state-of-the-art performance. Appendix G reports detailed statistics.

###### Reward Train Data GSM8k MATH

ALFWorld ScienceWorld

Outcome MATH+GSM8k 82.6 58.8 Out.+Format MATH+GSM8k 86.4 55.9 Avg. MATH+GSM8k 86.5 55.8 Outcome MATH 82.9 59.1 Out.+Format MATH 83.9 56.8 Avg. MATH 83.6 54.9

Method

L0 L1 L2 L0 L1 L2 GRPO + Out. 76.6 71.1 29.7 21.1 13.7 10.9 GRPO + Avg. 88.1 85.4 30.5 37.9 31.3 18.0 GiGPO 86.7 83.2 48.0 25.8 15.2 4.7 RLVMR 89.1 87.9 56.3 46.9 34.4 26.5 DERL 91.0 89.1 65.0 47.7 43.0 30.1 DERL-pop. 91.8 88.3 76.4 98.2 95.3 31.3

DERL MATH+GSM8k 87.0 60.2 DERL-pop. MATH+GSM8k 87.6 60.2 DERL MATH 83.2 60.5 DERL-pop. MATH 84.1 60.9

- 2) Robustness in OOD Scenarios. A critical limitation of heuristic rewards is the brittleness under distribution shifts. As shown in L2 (OOD) columns, standard baselines falter significantly. For instance, while “GRPO + Avg.” improves in-distribution (L0) performance by approximately 10% over “GRPO + Out.”, it fails to translate this gain to the OOD setting (showing only a 0.8% improvement). This suggests that the straightforward reward summation encourages overfitting rather than genuine reasoning. In contrast, our DERL substantially improves the OOD robustness, achieving 65.0% and 30.1% success rates on ALFWorld and ScienceWorld, respectively. This more than doubles the performance of the outcome reward baseline.

- Take-away 1. The Meta-Reward captures the intrinsic structure of the task, enabling generalization to unseen scenarios where heuristic combinations fail.

Mathematical Reasoning presents a unique challenge where the outcome reward is already strong, but sparse for hard question exploration. Table 1b illustrates the performance. We observe that naively incorporating auxiliary signals (e.g., GRPO with Outcome + Format or Avg Reward) often degrades performance on the more difficult MATH dataset (dropping from 58.8% to 55.8%), likely due to “reward hacking” or distraction from the core reasoning path (e.g., prioritizing formatting over correct reasoning). However, our DERL successfully navigates this pitfall. By autonomously optimizing the reward structure without any human effort, DERL outperforms all baseline reward functions, including the strong outcome reward (e.g., 60.2% vs. 58.8% on MATH), with the population-based instantiation DERL-pop. further improving the performance. This demonstrates DERL ’s ability to navigate the delicate trade-off between signal density and signal fidelity.

- Take-away 2. Even in domains with strong ground-truth signals, DERL discovers non-trivial reward composition that provides denser feedback.

### 4 Analysis

Having shown the empirical superiority, we now investigate the mechanisms driving the improvements: the outer-loop optimization dynamics and the structural evolution of the generated rewards.

#### 4.1 Optimization Dynamics

A critical question is whether the Meta-Optimizer genuinely learns a progressive optimization strategy or merely performs a random search over the function space. To investigate this, we visualize the training dynamics (i.e., how the Meta-Optimizer evolves over training steps) on ALFWorld, GSM8K and MATH benchmarks in Figure 3. Results on ScienceWorld are not shown

###### Train Acc Test Acc Trend of Train Acc Trend of Test Acc

ALFWorld

GSM8k

MATH Benchmark

0.75

1.0

0.70

0.8

AvgAccuracy

AvgAccuracy

SuccessRate

0.65

0.9

0.7

0.60

0.6

0.8

0.55

0.5

0.50

0.7

0.4

0.45

0.40

0.6

0.3

0 1 2 3 4 5 6 7 8 9 101112131415

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7

Step

Step

Step

- Figure 3: Training dynamics of the Meta-Optimizer on ALFWorld, GSM8K and MATH Benchmarks. The x-axis represents the training steps of outer-loop. The blue and orange line denotes the average validation and testing performance over Meta-Reward (i.e., “rollouts”), respectively. The results show that DERL gradually converges as the training progresses, without overfitting.

because the Meta-Optimizer converges faster. Specifically, we demonstrate the average validation accuracy of inner-loop policies θ1,θ2,...,θn trained with n Meta-Rewards.

We observe a consistent, monotonic upward trend in average performance of both validation and testing as the outer-loop progresses. Specifically, we find that the trend in mathematical reasoning is more stable as the Meta-Optimizer recognizes that the verifiable task is mainly driven by the outcome reward. For agent tasks, outer-loop optimization involves more exploration than exploitation, ultimately showcasing robust evolution. Crucially, the concurrent rise in validation and testing performance provides empirical verification that the Meta-Optimizer is not overfitting to the specific instances in outer-loop training. Instead, it successfully approximates the “meta-gradient” of task success. By leveraging the validation performance as a supervisory signal, the Meta-Optimizer gradually refines the reward function, generating increasingly effective signals that guide the innerloop agent toward higher performance. Appendix E provides concrete examples of Meta-Rewards explored across outer-loop iterations.

- Take-away 3. The Meta-Optimizer drives a gradient-guided evolution rather than a stochastic random search, validating that DERL captures the intrinsic meta-gradient of the task, enabling the refinement of generalizable reward structures without overfitting to the outer-loop training instances.
- 4.2 Evolution Dynamics of Reward Structures

Beyond performance gains, we ask whether gradient feedback alone induces stable reward structures in learned Meta-Rewards. To this end, we qualitatively analyze the structural composition of the generated Meta-Reward throughout the evolution, i.e., the evolution dynamics.

We categorize the combinations of atomic primitives (denoted as g1, g2, g3, and g4) into three distinct structure types based on mathematical properties: 1) Stable Structure: Linear combinations or normalization mechanisms. For example, linear additions (e.g., 0.5 · g1 + 0.8 · g2) or division operations (e.g., g

0.8

0.8

UnstableStructure

###### StableStructure

0.7

0.7

0.6

0.6

Stable Structure (ss.)

0.5

0.5

Unstable Structure (us.)

0.4

0.4

Trend of ss. Trend of us.

g2+1) that act similarly to sigmoid functions bound the output range. This structure mirrors robust designs in deep learning, preventing numerical explosion while retaining sufficient expressivity to guide the agent. 2) Unstable Structure: Predominantly unbounded products without normalization. A typical example is a chain of sequential multiplications (e.g., g1 · (g2 + 0.2) · g3). This structure creates a severe “veto” mechanism: if any single atomic signal approaches zero, the entire reward vanishes, leading to high variance and unstable gradient updates. 3) Invalid Structure: Mathematically adversarial forms, such as assigning negative coefficients to positive signals (e.g., −(g1 + 0.5 · g2)), which penalize desirable behaviors and offer no optimization utility.

1

0.3

0.3

0.2

0.2

0.1

0.1

0.0

0.0

0 1 2 3 4 5 6 7 8

Step

Figure 4: Evolution dynamics of reward structures on ALFWorld. Stable Structures increasingly dominate while Unstable Structures decline, revealing the Meta-Optimizer’s preference for mathematical robustness.

- Figure 4 tracks the distribution of these structural types over the course of training on ALFWorld. The optimization trajectory reveals a distinct gradient-guided refinement process. In the early exploration phase, Unstable Structures appear frequently as the optimizer explores the search space. However, as training progresses, we observe a sharp decline in their prevalence. Simultaneously, the proportion of Stable Structures exhibits a strong upward trend, eventually becoming dominant. This dynamic suggests that the Meta-Optimizer effectively acts as an evolutionary filter for mathematical robustness. Without explicit human programming or constraints, DERL implicitly discovers that consistent, bounded, and numerically stable rewards are critical prerequisites for effective policy optimization.

Take-away 4. The Meta-Optimizer implicitly learns to prioritize numerical stability and boundedness. This demonstrates that DERL discovers essential reward design principles solely through gradient feedback, without requiring manual constraints.

- 5 Related Work

Automatic Agent Evolution LLMs increasingly power agent systems with planning, tool use, and self-correction [26, 27]. However, their deployment remains bottlenecked by brittle humanengineered configurations (e.g., prompts, workflow, codes, reward functions, etc) [28, 29]. While evolutionary algorithms optimize agent configurations, they operate as discrete black-box search through permutation [7, 30, 31] or prompted agents [8–10, 32], relying on sparse final fitness scores and ignoring training dynamics [33]. DERL instead introduces a parameterized MetaOptimizer for gradient-guided configuration search. Using validation performance as the reward signal, DERL captures the meta-gradient through evolution, moving beyond human priors toward scalable, computation-driven optimization.

Learning to Learn Meta-learning develops models or algorithms that adapt by leveraging experience from related tasks [34, 35], including learned optimizers and hyper-parameter tuning [36]. In RL, meta-learners have been used to optimize the inner-loop learning process of an agent [22, 37–40]. Our DERL instead tunes an LLM Meta-Optimizer ψ that proposes discrete symbolic configurations ϕ for composing Rϕ, using validation performance of independently optimized inner policies as outer feedback. Unlike meta-gradient RL, DERL does not backpropagate through the full inner-loop optimizer or directly optimize continuous reward parameters. This design targets the demanding reward-tuning problem directly, while avoiding both preference-labeled reward-model training and differentiating through the full inner-loop optimization.

Reward Modeling LLM alignment relies on reward functions that provide feedback for RL-based agent evolution [1], but faces a tradeoff between sparse objective outcome rewards [12, 41] and dense but costly human-annotated rewards such as RLHF [5, 42]. Recent studies train reward models on large-scale LLM-annotated web data [20, 43, 44], while others design heuristic rewards that require manual coordination and may degrade performance if naively combined [4, 23, 45, 46]. To address these challenges, our DERL employs a Meta-Optimizer to automatically generate reward functions without external human preference data.

- 6 Conclusions and Limitations

We introduced Differentiable Evolutionary Reinforcement Learning (DERL), a bi-level framework that automates reward discovery by composing atomic primitives and training a Meta-Optimizer from inner-loop validation performance. This turns black-box reward search into gradient-guided meta-optimization, allowing the Meta-Optimizer to progressively capture the “meta-gradient” of task success without relying on expensive human annotation. Across robotic, scientific, and mathematical reasoning tasks, DERL consistently outperforms standard RL baselines and human-designed heuristics, with particularly strong OOD generalization on ALFWorld and ScienceWorld. Analysis of the evolution dynamics further shows that the Meta-Optimizer converges toward stable and robust reward structures, suggesting that DERL discovers task-relevant reward logic rather than merely performing unguided search. A major limitation is that the framework relies on a predefined set of atomic primitives. However, we empirically demonstrate that DERL is not sensitive to atomic primitives (Appendix F.3). Therefore, DERL can be easily adapted to different downstream tasks by simply defining a set of atomic primitives.

### References

- [1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [2] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.
- [3] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.
- [4] Chuanhao Yan, Fengdi Che, Xuhan Huang, Xu Xu, Xin Li, Yizhi Li, Xingwei Qu, Jingzhe Shi, Chenghua Lin, Yaodong Yang, Binhang Yuan, Hang Zhao, Yu Qiao, Bowen Zhou, and Jie Fu. Re:form – reducing human priors in scalable formal software verification with rl in llms: A preliminary study on dafny, 2025. URL https://arxiv.org/abs/2507.16331.
- [5] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [6] Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. arXiv preprint arXiv:1712.06567, 2017.
- [7] Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, et al. Population based training of neural networks. arXiv preprint arXiv:1711.09846, 2017.
- [8] Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G Finlayson, David Sontag, et al. Dr tulu: Reinforcement learning with evolving rubrics for deep research. arXiv preprint arXiv:2511.19399, 2025.
- [9] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin godel machine: Open-ended evolution of self-improving agents. arXiv preprint arXiv:2505.22954, 2025.
- [10] Alexander Novikov, Ngân V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.
- [11] W Bradley Knox and Peter Stone. Interactively shaping agents via human reinforcement: The tamer framework. In Proceedings of the fifth international conference on Knowledge capture, pages 9–16, 2009.

- [12] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [13] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.
- [14] Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? arXiv preprint arXiv:2203.07540, 2022.
- [15] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [17] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nat., 625(7995):468–475, 2024. doi: 10.1038/S41586-023-06924-6. URL https://doi.org/ 10.1038/s41586-023-06924-6.
- [18] Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. Population based training of neural networks, 2017. URL https://arxiv.org/abs/1711.09846.
- [19] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36:49205–49233, 2023.
- [20] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=IEduRUO55F.
- [21] Michel Ma, Takuma Seno, Kaushik Subramanian, Peter R. Wurman, Peter Stone, and Craig Sherstan. Automated reward design for gran turismo, 2025. URL https://arxiv.org/abs/2511.02094.
- [22] Irwan Bello, Barret Zoph, Vijay Vasudevan, and Quoc V. Le. Neural optimizer search with reinforcement learning. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 459–468. PMLR, 2017. URL http://proceedings.mlr.press/ v70/bello17a.html.
- [23] Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, and Xiaolong Li. Rlvmr: Reinforcement learning with verifiable meta-reasoning rewards for robust long-horizon agents. arXiv preprint arXiv:2507.22844, 2025.
- [24] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.
- [25] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [26] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.
- [27] Sitao Cheng, Ziyuan Zhuang, Yong Xu, Fangkai Yang, Chaoyun Zhang, Xiaoting Qin, Xiang Huang, Ling Chen, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Call me when necessary: LLMs can efficiently and faithfully reason over structured environments. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 4275–4295, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.254. URL https://aclanthology.org/2024.findings-acl.254/.

- [28] Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 13(1):38, 2019.
- [29] Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, Anya Sims, Dylan Cope, Jarek Liesen, Lukas Seier, Theo Wolf, et al. Evolution strategies at the hyperscale. arXiv preprint arXiv:2511.16652, 2025.
- [30] Xingwu Chen, Tianle Li, and Difan Zou. Reshaping reasoning in llms: A theoretical analysis of rl training dynamics through pattern selection, 2025. URL https://arxiv.org/abs/2506.04695.
- [31] Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, et al. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. arXiv preprint arXiv:2508.07407, 2025.
- [32] Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, and William Yang Wang. G\" odel agent: A self-referential agent framework for recursive self-improvement. arXiv preprint arXiv:2410.04444, 2024.
- [33] Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, et al. A survey of self-evolving agents: On path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 2025.
- [34] Ricardo Vilalta and Youssef Drissi. A perspective view and survey of meta-learning. Artificial intelligence review, 18(2):77–95, 2002.
- [35] Zhongwen Xu, Hado P van Hasselt, and David Silver. Meta-gradient reinforcement learning. Advances in neural information processing systems, 31, 2018.
- [36] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando de Freitas. Learning to learn by gradient descent by gradient descent,

2016. URL https://arxiv.org/abs/1606.04474.

- [37] Rishabh Agarwal, Chen Liang, Dale Schuurmans, and Mohammad Norouzi. Learning to generalize from sparse and underspecified rewards. In International conference on machine learning, pages 130–140. PMLR, 2019.
- [38] Zhongwen Xu, Hado P van Hasselt, Matteo Hessel, Junhyuk Oh, Satinder Singh, and David Silver. Metagradient reinforcement learning with an objective discovered online. Advances in Neural Information Processing Systems, 33:15254–15264, 2020.
- [39] Junhyuk Oh, Matteo Hessel, Wojciech M Czarnecki, Zhongwen Xu, Hado P van Hasselt, Satinder Singh, and David Silver. Discovering reinforcement learning algorithms. Advances in Neural Information Processing Systems, 33:1060–1070, 2020.
- [40] Anonymous. Temperature as a meta-policy: Adaptive temperature in LLM reinforcement learning. In Submitted to The Fourteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=AoTHU2OmS6. under review.
- [41] Zhengyang Tang, Zihan Ye, Chenyu Huang, Xuhan Huang, Chengpeng Li, Sihang Li, Guanhua Chen, Ming Yan, Zizhuo Wang, Hongyuan Zha, Dayiheng Liu, and Benyou Wang. Calm before the storm: Unlocking native reasoning for optimization modeling, 2025. URL https://arxiv.org/abs/2510.04204.
- [42] Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Reinforcement learning for reasoning in large language models with one training example, 2025. URL https://arxiv.org/abs/2504.20571.
- [43] Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. arXiv preprint arXiv:2505.14652, 2025.
- [44] Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024.
- [45] Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Sean Chen, Mohammad Kachuee, Teja Gollapudi, Tony Liao, Nicolas Scheffer, et al. Truthrl: Incentivizing truthful llms via reinforcement learning. arXiv preprint arXiv:2509.25760, 2025.
- [46] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

- [47] Xiang Huang, Sitao Cheng, Yuheng Bao, Shanshan Huang, and Yuzhong Qu. Markqa: A large scale kbqa dataset with numerical reasoning. arXiv preprint arXiv:2310.15517, 2023.
- [48] Sitao Cheng, Liangming Pan, Xunjian Yin, Xinyi Wang, and William Yang Wang. Understanding the interplay between parametric and contextual knowledge for large language models. arXiv preprint arXiv:2410.08414, 2024.
- [49] Sitao Cheng, Xunjian Yin, Ruiwen Zhou, Yuxuan Li, Xinyi Wang, Liangming Pan, William Yang Wang, and Victor Zhong. From atomic to composite: Reinforcement learning enables generalization in complementary reasoning. arXiv preprint arXiv:2512.01970, 2025.
- [50] Xiang Huang, Jiayu Shen, Shanshan Huang, Sitao Cheng, Xiaxia Wang, and Yuzhong Qu. TARGA: Targeted synthetic data generation for practical reasoning over structured data. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2704–2726, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.137. URL https://aclanthology.org/2025.acl-long.137/.

### A Detailed Primitive Design with Concrete Examples

To clarify how the reward search space is constructed and how primitives are evaluated, we detail the atomic primitive definitions with concrete examples.

#### A.1 The merits of our reward parameterization design.

- • Coverage of an informative and continuous search space. Equation (3) takes abundant factors into account, ensuring an expressive space covering the standard outcome signal.
- • Structural reasoning. The Meta-Optimizer can focus on considering different aspects of the problem instead of processing tedious textual output.
- • Extensibility, decoupling the definition of atomic signals from their utilization. This renders the system agnostic to the specific choice of G, thereby facilitating seamless generalization to new tasks. One can incorporate diverse potential discriminators—ranging from rigorous constraints to taskspecific heuristics, or even potentially detrimental signals. The evolutionary process automatically filters and weights these components, circumventing the need for manual validation of their individual utility.

#### A.2 Concrete Examples

All primitives are simple and intuitive, providing the fundamental “tools” for the Meta-Optimizer to explore the functional space without relying on heavy human engineering or struggling with textual parsing. Moreover, we clarify that the primitives are easy to obtain and enjoy robustness in Appendix F.3.

Robotic Agent and Scientific Simulation. For embodied and scientific agents, outcome rewards are often delayed until the end of a long trajectory. To provide the Meta-Optimizer with building blocks for dense credit assignment, we design primitives based on temporal trajectory phases. Specifically, inspired by RLVMR [23], the model is instructed to use tags to assign per-step scores for exploration (e.g., reaching a new state), planning (e.g., breaking down tasks), and formatting. By simply computing the atomic primitives for each stage of this process reward, our method significantly outperforms baselines, including RLVMR.

First, we assign a step-wise heuristic score by verifying if the agent’s textual action aligns with the underlying meta-tag of the task (inspired by prior work [23]) using simple string pattern matching. We then construct our primitives by evenly dividing the trajectory steps into three temporal phases:

- • g1 = 1 if the task is ultimately successful, otherwise 0.
- • g2 = The average step score across the first third of the trajectory.
- • g3 = The average step score across the second third of the trajectory.
- • g4 = The average step score across the final third of the trajectory.

Example Generated Func(): g1 + (g2 · (g3/2)) − (g4/3)

Mathematical Reasoning For mathematical reasoning (i.e., GSM8K and MATH), the ground-truth outcome is highly reliable, but sparse. We define four simple primitives based on the policy’s raw output o, the extracted predicted answer yˆ, and the ground truth y∗:

- • g1 = 1 if yˆ == y∗ (Strict accuracy match), otherwise 0.
- • g2 = 1 if yˆ is properly formatted and enclosed within \boxed{}, otherwise 0.
- • g3 = 1 if structured reasoning step indicators (e.g., “Step 1”, “Step a”) are explicitly present in o, otherwise 0.
- • g4 = 1 if y∗ appears anywhere within the raw output o (Soft match, capturing instances where the model derives the correct number but fails formatting), otherwise 0.

##### Example Generated Func(): g1 + 0.5 · g2 + 0.1 · (g3 + g4)

- B Experimental Protocol and Detailed Training Setups This section summarizes the common experimental protocol and default settings used across domains.

Data split and validation usage. Let Dtrainorig denote the original training set. During outer-loop tuning, we randomly split Dtrainorig into an inner-training split Din and a validation split V with an 8:2 ratio. For each generated configuration ϕi, the corresponding reward Rϕ

trains an inner-loop policy θi on Din, and V is used only to compute the outer feedback vi = Perf(θi;V) for updating the Meta-Optimizer and selecting the reward configuration. The held-out test set Dtest is never used during outer-loop tuning.

i

Final training and evaluation. After selecting the reward configuration, we discard the inner-loop policies used during tuning and train a fresh final policy on the full original training set Dtrainorig . The final policy is evaluated only on the held-out test set Dtest. Thus, V provides the outer-loop reward during Meta-Reward tuning but is not part of the final reported evaluation.

Training-budget fairness. For final-policy comparison, DERL uses the same original training data and no more final-policy training steps than the compared baselines. The extra cost of DERL comes from outer-loop reward tuning, which is reported separately in Appendix H. We also include compute-controlled random search and rollout-sensitivity experiments to distinguish the effect of learned outer-loop optimization from the effect of additional search budget.

Default outer-loop settings. Unless otherwise stated, the Meta-Optimizer is initialized from Qwen-2.5-0.5B-Instruct and optimized with GRPO. The default outer-loop rollout size is n = 8. We use a brief cold-start SFT stage on valid reward-expression examples to initialize the output grammar, constrained decoding to restrict generations to valid primitives and mathematical operators, and a validity penalty vi = 0 for reward configurations that are mathematically ill-defined or fail to execute.

Default inner-loop settings. For ALFWorld and ScienceWorld, we implement GRPO via VeRL and use Qwen2.5-1.5B-Instruct as the base policy model. We train inner-loop policies for 40 epochs on ALFWorld and 80 epochs on ScienceWorld. After the reward configuration is selected, the final policy is trained from scratch with the selected reward function. For DERL-pop., we run 10 outer-loop rounds with 10 inner-loop epochs on ALFWorld and 3 outer-loop rounds with 33 inner-loop epochs on ScienceWorld.

For mathematical reasoning, we use Qwen-2.5-3B as the base policy model. Inner-loop training runs for 10 epochs with a 3.5-hour time limit, and the selected Meta-Reward is then used to train the final base policy for 15 epochs. For DERL-pop., each inner-loop trains for 2 epochs, and we report the testing result after the seventh outer-loop iteration.

Detailed Definition of Generalization Levels for Robotic Agent and Scientific Simulation. To rigorously evaluate the robustness of our learned Meta-Rewards, we define distinct task levels based on combinations of data distributions (in-distribution vs. out-of-distribution) and environment novelty (seen vs. unseen), inspired by RLVMR [23]. We use the ALFWorld benchmark [13] partitions to illustrate this setup.

ALFWorld comprises six broad task categories (e.g., Pick & Place, Examine in Light). We employ two training configurations: training on all six categories (in-distribution) or training on only four categories (out-of-distribution). Furthermore, ALFWorld officially partitions its test set into two subsets:

- • Seen Test Set: Contains the exact same target instances and concepts encountered during training, but with altered environmental variables (e.g., if “interacting with a pencil” was in the training set, the test set might feature a pencil in a novel location or quantity).
- • Unseen Test Set: Contains completely novel object variants and instructions never encountered during training (e.g., “put a watch in the safe,” where neither the watch nor the safe appeared in the training set).

- (a) Inner Loop

- (b) Outer Loop on Trajectory 𝜏′

|GRPO on Trajectory 𝜏|
|---|

KL

|𝑹|
|---|

- 𝑜1

- 𝑜2

- 𝑟1

- 𝑟2

- 𝐴1

- 𝐴2

Reference Model

Policy Model 𝜃

Group Computation

𝑞

Reward Model

…

…

…

𝑜𝐺

𝑟𝐺

𝐴𝐺

Node 1.1

Meta Optimizer parameterized by 𝒘𝒂𝒅𝒅,𝒘𝒔𝒖𝒃… on Trajectory 𝝉′

𝑤𝑎𝑑𝑑 (𝑔1+𝑔2) + 𝑤𝑠𝑢𝑏 (𝑔1−𝑔2) + 𝑤𝑚𝑢𝑙 (𝑔1∗ 𝑔2)

###### Node 2.1

Final Reward

𝑤𝑎𝑑𝑑 (…) + 𝑤𝑠𝑢𝑏 (…) + 𝑤𝑚𝑢𝑙 (…) + 𝑤𝑑𝑖𝑣 (…)

- 𝑔1 + 𝑤𝑑𝑖𝑣 (𝑔1 /𝑔2)

- 𝑔2

- 𝑔3

- 𝑔4

𝑅𝑡

###### Node 1.2

𝑤𝑎𝑑𝑑 (𝑔1+𝑔2) + 𝑤𝑠𝑢𝑏 (𝑔1−𝑔2) + 𝑤𝑚𝑢𝑙 (𝑔1∗ 𝑔2) + 𝑤𝑑𝑖𝑣 (𝑔1 /𝑔2)

Trainable (Differentiable)

Frozen

- Figure 5: Demonstration of the preliminary training loop. We adopt GRPO as an example RL algorithm for the inner-loop. In the offline reward-fitting stage, we fit a compact computation graph over atomic reward primitives and obtain the final reward function Φt parameterized by ϕt = {wadd,wsub,...} through differentiable optimization.

By combining these partitions, we establish the three difficulty levels reported in the main text: L0 (In-distribution, Seen), L1 (In-distribution, Unseen), and L2 (Out-of-distribution, Unseen).

### C Preliminary Experiments – Feasibility and Generality

We present a preliminary experiment to investigate the capacity of our proposed bi-level training framework (DERL). We aim to address two research questions: (1) Is the Meta-Optimizer indeed able to learn a dedicated and concrete reward function better than standard outcome rewards? (2) Is our DERL generalizable to other model architecture and training frameworks? We utilize a computational graph as the Meta-Optimizer, parameterized by a small set of weights (i.e., 12 parameters in total) to derive a reward function, which is then utilized to train the inner-loop model (Figure 5).

#### C.1 Experimental Setup

Purpose and scope. This experiment studies a simplified offline setting rather than the full bi-level DERL framework. The goal is to test whether a compact computation graph over the same atomic primitives can represent useful reward functions (research question 1). During this preliminary optimization, the Meta-Reward is fitted on fixed trajectories generated by Qwen2.5-3B-Instruct; there is no closed-loop update from inner-loop validation performance to the Meta-Optimizer. After the graph reward is fitted, we freeze it and use it as the reward function for a downstream policytraining run. Therefore, this experiment is a proof-of-concept for the expressiveness of the primitive space, not a substitute for the main outer-loop optimization (research question 2).

Architecture. As illustrated in Figure 5, this preliminary study uses a compact computation graph over atomic reward primitives to parameterize the Meta-Reward. The Meta-Reward function, parameterized by ϕt, is represented by a set of twelve learnable weights (e.g., wadd,wsub,wmul,...)

Table 2: Performance comparison on mathematical reasoning benchmarks. We compare the preliminary results of Meta-Reward with different baselines. The Meta-Optimizer utilizes a 12-parameter graph optimizer to shape the reward signal.

#### Reward Function GSM8K MATH

Outcome 82.6 58.8 Avg Reward 86.5 55.8 Meta-Reward (Regression) 85.5 62.9 Meta-Reward (RL) 83.2 59.9

distributed across the computation nodes. The set of atomic primitives used in this graph remains consistent with those described in the main body of this paper (Section 2).

Training Protocols. We implement the training of computation graph in the offline preliminary setting in two ways:

- • Supervised regression: We fit the compact graph reward toward the standard outcome reward on fixed trajectories. This is feasible because the target is manually specified as the binary outcome signal. Note that this supervised objective differs from the cold-start SFT in the main DERL pipeline which only teaches the LLM Meta-Optimizer the grammar of valid reward expressions before RL.
- • Reinforcement learning (RL): We also optimize the graph weights with an RL-style objective over the same fixed trajectories. For each computation step, we sample operations based on the distribution of w within each node. If the resulting Meta-Reward aligns with the ground truth outcome reward (i.e., I(Meta-Reward = Outcome)), a reward of 1.0 is assigned to the current configuration of w; otherwise, the reward is 0.0. This tests whether the compact graph can be optimized without direct supervised regression, but it still does not use the closed-loop validation feedback of the main DERL framework.

Model and Data. We first generate offline trajectories on the GSM8K and MATH training sets using Qwen2.5-3B-Instruct. These fixed trajectories provide the inputs for fitting the 12-parameter graph reward. Once the graph reward is fitted, it is frozen and used to train a downstream policy model (Qwen2.5-3B) under the same inner-loop settings as the corresponding baselines. This separates the offline reward-fitting stage from the downstream policy-training stage.

#### C.2 Results and Analysis

These results should be interpreted as evidence that the primitive-composition space contains useful dense reward signals. The strong regression result does not imply that supervised learning can replace the outer loop in DERL: in this preliminary setting, the target reward is fixed in advance as the binary outcome signal, whereas in the main DERL setting the optimal reward formulation is unknown. DERL therefore uses validation performance from independently optimized inner policies as outer feedback for tuning the Meta-Optimizer, allowing the reward generator to adapt to downstream policy learning rather than merely mimic a predefined outcome reward.

- Table 2 presents the performance comparison between the standard outcome reward, an average reward baseline (same as the main experiments in Section 3), and our proposed Meta-Reward (regression and RL). The results indicate that the Meta-Reward, despite being parameterized by only 12 weights, effectively discovers a reward function that outperforms the sparse outcome reward. Notably, on the challenging MATH dataset, the Meta-Reward (Regression) achieves a significant improvement over the Outcome baseline (62.9% vs. 58.8%).

These findings suggest that the Meta-Reward mechanism avoids overfitting to the rigid binary outcome signal. Analogous to an educational setting, using a binary outcome reward is akin to instructing a student solely to "score 100 points," which provides a sparse and high-variance signal. In contrast, our approach—constrained by the computational graph structure—encourages the model to learn a generalized heuristic. Although this is not an explicit process reward annotated by humans,

DERL pop.

GRPO w/ Avg Reward

0.8

| |
|---|

| |
|---|

| |
|---|

0.6

| |
|---|

Success Rate

| |
|---|

| |
|---|

| |
|---|

0.4

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.2

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.0

0 10 20 30 40 50 60 70 80 90

Step

- Figure 6: Training dynamics of DERL-population. We present the training dynamics of DERL-pop. and GRPO w/ Avg Reward to demonstrate the superiority of the population method.

the optimization process discovers an implicit, dense reward function that guides the model more effectively toward the correct reasoning path than the raw outcome signal alone.

### D Training Dynamics of DERL-pop.

Does DERL-pop yield consistent progress throughout the bi-level training process? As established in the main text, DERL-pop achieves significantly superior performance across all evaluated domains, indicating that the Meta-Optimizer successfully and dynamically explores more effective reward configurations. Notably, policies trained via RL with these evolved Meta-Rewards exhibit strong generalization to out-of-distribution (OOD) environments, extending beyond the parametric skills explicitly learned from the training data [47–50]. In this section, we investigate exactly when this generalization capacity emerges during the training dynamics of DERL-pop.

- Figure 6 illustrates these training dynamics, comparing DERL-pop against the GRPO w/ Avg Reward baseline on the ScienceWorld L0 task. To ensure a fair comparison of computational budgets, the baseline is trained continuously for a full 100 steps. Conversely, the DERL-pop inner-loop is trained for only 33 steps per outer-loop iteration. Crucially, each subsequent inner-loop phase initializes from the highest-performing checkpoint of the previous generation, rather than training from scratch. The empirical trajectory reveals three distinct phases:

- • Initial Phase (0–33 steps): Both models perform on par with one another, as the MetaOptimizer is still exploring the reward space and the policy has not yet benefited from a refined signal.
- • First Adaptation (34–66 steps): Beginning in the second outer-loop iteration, DERL-pop injects an updated reward structure. Driven by this denser feedback, the policy rapidly diverges from and outperforms the static baseline.
- • Sustained Generalization (>66 steps): In the final stages, DERL-pop exhibits significantly accelerated performance gains, establishing a substantially higher performance ceiling.

Ultimately, this trajectory highlights how the dynamic, curriculum-like nature of the DERL-pop reward formulation bypasses the performance plateaus inherent to static fixed reward functions.

### E Examples of Outer-loop Evolution

We demonstrate the detailed training dynamics of DERL by showcasing each Meta-Reward explored in the outer-loop on ALFWorld (L2) in Table 3. We show four outer-loop iterations. We observe that there may be a lot of low-quality meta-rewards in the early stages, but DERL can quickly learn high-quality rewards.

- Table 3: Evolution of Meta-Reward structures and their corresponding reward across outer-loop training steps on ALFWorld.

#### Step Meta Reward Reward

- 0

g1 ∗ (g2 − 1)/2 + (g3 + 1) ∗ (g4 − 1) ∗ 2/3 0

g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4)))) − 0.5 ∗ (g2 + 0.5 ∗ (g3 + ··· 0 g1 + 0.01 ∗ (g2 − 0.001) + 0.0001 ∗ (g3 − 0.0001) + 0.000005 ∗ (g4 − ··· 0.8496

g1 ∗ 0.5 + 0.2 ∗ (g2 + 0.1) − 0.3/2 + 0.4 ∗ (g3 ∗ 0.1) + 0.1 ∗ (g4 ∗ ··· 0.8789 g1 ∗ (g2 + (g3 − 1.0) ∗ (g4 − 0.0)) 0.0234

g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4 + 0.5))) 0.8848 g1 + 0.5 ∗ (g2 + 0.2 ∗ (g3 + 0.1 ∗ (g4 + 0.05)))) 0

g1 ∗ (g2 + 2 ∗ (g3/3)) + (g4 − 4 ∗ (g2 − 1)) − 2.0 0.8945

- 1

(0.5 ∗ (g1 − 0.1)) + (0.5 ∗ (g2 − 0.1)) + (0.5 ∗ (g3 − 0.1)) + (0.5 ∗ ··· 0.7793 g1 ∗ (g2 + (g3 ∗ (g4 − (g2 + (g3 ∗ (g4 − (g3 ∗ (g4 − (g3 ∗ (g4 − (g3 ∗ ··· 0

−(g1 + 0.5 ∗ (g2 + 0.3 ∗ (g3 − 0.2 ∗ (g4 + 0.1 ∗ 1))) + 0.1 ∗ 1)/1.2 0.0020 g1 ∗ (g2 − 1) ∗ (1 − 0.5) + 0.5 ∗ ∗2 ∗ (g3 − 1) ∗ (1 − 0.25) + 0.25 ∗ ∗··· 0

g1 + (g2 ∗ (g3/2)) − (g4/3) 0.8594 g1 ∗ (g2 + 0.5) + 0.2 ∗ (g3 ∗ 0.3 + 0.1) − 0.1 ∗ (g4 ∗ 0.2 + 0.5) 0.8477 g1 + 2 ∗ (g2 + 3 ∗ (g3 − 2)) ∗ 0.1 + 4 ∗ 0.3 − 5 ∗ (g4 + 1) 0.8984 g1 ∗ (g2 − 1)/2 + (g3 − 1)/3 + (g4 − 1)/4 + 3.0 0.0098

- 2

g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4)))) 0 g1 + 0.5 ∗ (g2 − 0.5) + 0.3 ∗ (g3 − 0.05) + 0.2 ∗ (g4 + 0.05) − 0.2 ∗ ··· 0

g1 + (g2 ∗ (g3/2)) − (g4/4) 0.8438 g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4 − 1)))) 0

g1 + 0.01 + 0.0001 ∗ (g2) + 0.000001 ∗ (g3) + 0.00000001 ∗ (g4) 0.8652

g1 ∗ (g2 + (g3 ∗ (g4/2))) + (g1 − (g2 + (g3 ∗ (g4/2)))) ∗ 0.5 0.8867 g1 ∗ 0.99 + 0.01 ∗ (g2 + 0.99) + 0.005 ∗ (g3 + 0.99) + 0.0005 ∗ (g4 + ··· 0.8477

g1 + 0.5 ∗ (g2 − 0.5) + 0.2 ∗ (g3 − 0.5) + 0.1 ∗ (g4 − 0.5) 0.8906

- 3

g1 + (g2/2.0) − (g3 ∗ 0.1) + (g4 ∗ 0.05) 0.8438 g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4 + 0.5))) 0.875

g1 + 0.05 ∗ (g2 + 0.05 ∗ (g3 + 0.05 ∗ (g4 + 0.05))) 0.8242 g1 + 0.5 ∗ (g2/2.0) + 0.1 ∗ (g3 ∗ 2.0) + 0.25 ∗ (g4/4.0) 0.8632

- g1 + 0.5 ∗ (g2 + 0.4 ∗ (g3 + 0.2 ∗ (g4 + 0.1))) 0.8496
- g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4 + 0.5))) 0.8926 g1 + 0.5 ∗ (g2 + 0.5 ∗ (g3 + 0.5 ∗ (g4 + 0.5))) 0.8789

g1 + (g2 ∗ (g3/2)) − (g4/3) 0.8984

### F Additional Experiments

To further emphasize the differences between our DERL and other frameworks, highlighting DERL’s robustness, we conduct a series of additional experiments. These include comparisons with Reinforcement Learning from AI Feedback (RLAIF) [10], computation-controlled random search methods, sensitivity analyses of the primitive vocabulary and ablation study over number of outer-loop rollouts. Our experiments are trained and tested on the GSM8K dataset using Qwen2.5-1.5B. Experimental results show that DERL outperforms black-box methods using a significantly more powerful closed-source model (GPT-4o) and compute-controlled random search.

#### F.1 Comparison with RLAIF Method

Is DERL better than LLM-based single-level evolutionary method? A typical black-box evolution is not compared in the main paper because we can not find any comparable methods in literature. Here we implement a comparable baseline with RLAIF (Reinforcement Learning from AI Feedback) [10], which uses an LLM refine the reward signal based on its rollouts. The difference between RLAIF and our DERL is that it relies solely on the prompt for interaction where outer-loop does not update any parameters. Therefore, we replace the outer-loop of our DERL framework with a powerful closed-source model, GPT-4o, whose model size is significantly larger than the mere

Table 4: Comparison with RLAIF method.

(a) Comparison of performance. Our outer-loop uses the GPT-4o API, a powerful closed-source model. We fix the training method for the inner-loop. For the outerloop, we require the API to generate multiple reward functions to pass to the inner-loop. Experimental results show that the RLAIF method cannot surpass DERL. Bold denotes the best performance.

#### Method Train Test

GRPO + Out. 89.45 75.82 GRPO + Avg. 89.74 76.12 RLAIF 89.23 77.18 DERL 91.01 79.22

(b) Comparison of training dynamics. We show the average test set performance across multiple rollouts generated by the model in each outer-loop training step. DERL significantly optimizes at each step, but RLAIF does not optimize progressively.

#### Step DERL RLAIF

- 0 75.13 76.88
- 1 76.60 76.67
- 2 77.69 76.40
- 3 77.13 76.85
- 4 77.29 76.45
- 5 78.28 76.17
- 6 78.30 76.95

0.5B Meta-Optimizer in DERL. We fix the training method for the inner-loop as the same. For the outer-loop, we require the API to generate multiple reward functions to pass to the inner-loop. Except for the first round, we prompt the API model with the reward functions it generated in each round and the corresponding rewards (i.e., validation set performance).

- Table 4a shows that the RLAIF method can find reward functions that are better than the naive baselines, but cannot surpass our DERL.

We further demonstrate the average test set performance across multiple rollouts generated by the model in each training step, as shown in Table 4b. DERL significantly optimizes at each step (consistent with Section 4.1). However, RLAIF does not optimize progressively, even when using a strong closed-source model.

F.2 Comparison with Compute-controlled Random Search Method

Is the superiority of DERL comes from increased computation or from genuine optimization of the outer-loop? An important baseline is compute-controlled random search, which uses the same amount of computation but for random search over the reward function search space. For implementation, we freeze the outer-loop parameters. We keep all other settings completely consistent as DERL, such as the reward space and the number of rollouts, to ensure that the computational load is exactly the same.

- Table 5a shows that compute-controlled random search, due to the randomness in a high-coverage search space (validated by Appendix C), can indeed find a relatively good reward function, but it cannot surpass DERL. Table 5b further illustrates that DERL’s performance comes from stepwise optimization, while random search methods cannot be optimized at all.

This means that the superiority of DERL does not solely stem from increased computational cost, but rather from the fact that the outer-loop has indeed learned to optimize the reward function. Using methods with the same computational cost alone will not yield the same performance.

#### F.3 Ablation and Perturbation on the Primitive Vocabulary

Is DERL’s performance sensitive or robust to different atomic primitives? As stated in the main paper, while our primitive vocabulary relies on human design, all atomic primitives are very simple and intuitive, requiring no elaborate design. To further demonstrate that DERL’s high performance is independent of human-designed primitive vocabulary, we introduce four baselines.

Compared to the vanilla GRPO, we have three more atomic primitives. We first delete these three atomic primitives individually. Then we try flipping one of the atomic primitives (g2), i.e., taking its opposite. This is a toxic signal, because higher reward lead to smaller value.

Table 5: Comparison with compute-controlled random search method.

(a) Comparison of performance. We fix the same reward space and introduce a compute-controlled random search baseline. It might be able to randomly sample a reward function that is better than the vanilla GRPO, but it still cannot exceed DERL. Bold denotes the best performance. For the random search baseline, due to its randomness, we report the highest performance across all steps to ensure fairness.

(b) Comparison of training dynamics. We show the average test set performance across multiple rollouts generated by the model in each training step. DERL significantly optimizes at each step (consistent with Section 4.1). However, even when sharing the same reward space, random search method clearly cannot optimize during the iteration process to learn a beneficial reward function.

#### Method Train Test

GRPO + Out. 89.45 75.82 GRPO + Avg. 89.74 76.12 Random Search 89.59 76.50 DERL 91.01 79.22

#### Step DERL Random Search

- 0 75.13 76.08
- 1 76.60 74.90
- 2 77.69 75.97
- 3 77.13 54.58
- 4 77.29 76.34
- 5 78.28 75.16
- 6 78.30 74.83

Table 6: Ablation and perturb on the primitive vocabulary. To examine whether the performance gains come from DERL truly learning useful components, we introduce four baselines. Even with arbitrary removal of atomic primitive, or even the introduction of toxic atomic primitive, DERL can still find excellent reward functions.

#### Method Train Test

GRPO + Out. 89.45 75.82 GRPO + Avg. 89.74 76.12 Reverse g2 88.52 77.79

- Remove g2 89.47 78.39
- Remove g3 89.28 78.92
- Remove g4 89.86 77.94 DERL 91.01 79.22

- Table 6 shows that removing each additional atomic primitive has no impact on the model learning a

beneficial reward function. Furthermore, we find that for the Reverse g2, the model correctly utilizes the signal by learning a negative coefficient from this flipped atomic primitive. This means that the model may have captured the signal that the atomic primitive was flipped during the outer loop update process, i.e., negative coefficients would bring better results.

The results strongly support our view that DERL’s high performance does not depend on carefully designed atomic primitives. In other words, DERL is not sensitive to atomic primitives. Even with arbitrary removal of atomic primitive, or even the introduction of toxic atomic primitive, DERL can still find excellent reward functions within a limited or toxic primitive vocabulary space during optimization.

#### F.4 Sensitivity to Outer-loop Rollouts

DERL generates multiple rollouts in each step at the outer-loop and passes them to the inner-loop. We conduct ablation experiments on the number of rollouts. Table 7 shows that DERL is not sensitive to the number of rollouts. It does not affect the robustness or performance of the model. Furthermore, fewer rollouts (e.g., 4) can significantly reduce computational cost.

- Table 7: Sensitivity to outer-loop rollouts. We compare the impact of different numbers of outer-loop rollouts. With different rollout numbers (4, 6, 8), DERL achieves similar performance, indicating that DERL is not sensitive to the number of rollouts.

Method Train Test

GRPO + Out. 89.45 75.82 GRPO + Avg. 89.74 76.12 DERL, rollout=4 91.01 79.22 DERL, rollout=6 88.10 78.92 DERL, rollout=8 89.98 79.23

- Table 8: Mean, standard deviation and confidence interval under 4 random seeds (ALFWorld). In each data point, the first row represents mean ± std. The second row represents the confidence interval (t-distribution).

ALFWorld L0 L1 L2 GRPO + Out.

Method

76.56 ± 1.91 [73.52, 79.61]

71.09 ± 2.89 [66.50, 75.69]

29.69 ± 2.12 [26.32, 33.05]

88.09 ± 1.33 [85.96, 90.21]

85.35 ± 1.33 [83.23, 87.47]

30.47 ± 2.63 [26.28, 34.65]

GRPO + Avg.

89.06 ± 1.42 [86.80, 91.32]

91.02 ± 2.07 [87.73, 94.30]

65.04 ± 2.64 [60.83, 69.25]

DERL

91.80 ± 1.35 [89.64, 93.95]

76.37 ± 4.05 [69.92, 82.82]

88.28 ± 1.78 [85.46, 91.11]

DERL-pop.

### G Detailed Statistical Results in Table 1a

In our experiments on Robotic Agent and Scientific Simulation, we notice a degree of randomness in the results (Table 1a), which is due to the tasks themselves. Therefore, we report the specific results for these two tasks, which provide detailed statistics, including the mean, standard deviation and confidence interval under 4 random seeds.

- Table 8 and Table 9 report the statistical results for 4 random seeds (seed = 42, 43, 44, 45). In each data point, the first row represents mean ± std. The second row represents the confidence interval (t-distribution). The results show that our methods, DERL and DERL-pop, statistically outperform the baselines.

### H Computational Cost Analysis

We provide a detailed breakdown of the computational costs associated with the bi-level evolutionary training framework (DERL) and discuss potential strategies for efficiency improvements.

#### H.1 Computational Breakdown

The training process consists of an outer-loop (Meta-Optimizer evolution) and an inner-loop (Policy Model evolution). We incorporate parallelism in most parts of DERL to improve hardware utilization. The computational cost is summarized as follows:

Inner-Loop Latency (The Bottleneck). The primary computational bottleneck lies in the innerloop, where the policy model θi evolves (e.g., interacts with the environment) using the MetaReward Ri. Since our outer-loop utilizes the GRPO algorithm [12], the Meta-Optimizer generates n

- Table 9: Mean, standard deviation and confidence interval under 4 random seeds (ScienceWorld). In each data point, the first row represents mean ± std. The second row represents the confidence interval (t-distribution).

ScienceWorld L0 L1 L2 GRPO + Out.

Method

21.09 ± 3.19 [16.02, 26.17]

13.67 ± 1.86 [10.71, 16.63]

10.94 ± 1.10 [9.18, 12.70]

37.89 ± 4.53 [30.68, 45.10]

31.25 ± 3.72 [25.33, 37.17]

17.97 ± 5.45 [9.30, 26.64]

GRPO + Avg.

47.66 ± 2.76 [43.26, 52.05]

42.97 ± 2.31 [39.29, 46.65]

30.08 ± 1.58 [27.56, 32.60]

DERL

98.24 ± 1.61 [95.68, 100.0]

95.31 ± 1.28 [93.28, 97.34]

31.25 ± 2.18 [27.78, 34.72]

DERL-pop.

distinct Meta-Rewards (as rollouts) per step. To mitigate latency, we implement a fully parallelized architecture similar to standard GRPO:

- • Parallel Execution: All n rollouts (where n = 8 in our experiments) are evaluated simultaneously, with each inner-loop allocated a dedicated set of compute resources. Each inner-loop for each task finally consumes similar computation resources.
- • Malformed Rewards: Meta-Rewards that fail to compile or produce valid computation graphs are immediately terminated, assigned a reward of 0.0, and incur zero training cost (though the system waits for concurrent inner-loops to complete before updating the outerloop).
- • Time Budgeting: A significant challenge in Meta-Reward discovery is that certain reward functions may incentivize excessively long reasoning chains, increasing training costs unpredictably. To address this, we impose a strict computational budget. For mathematical reasoning tasks, we set a maximum floating-point operation cap approximately 1.3× the cost of standard binary-outcome training. If training exceeds this threshold, the process is halted, and the latest checkpoint is saved for evaluation. For other tasks, we rely on a fixed number of inner epochs, as the action space is more controllable.

Evaluation Cost. Following the inner-loop, we evaluate the validation performance to calculate the advantage for the Meta-Optimizer. We utilize vLLM for high-throughput inference. By parallelizing the evaluation across all n rollouts, the validation phase incurs negligible latency compared to training.

Outer-Loop Update. The Meta-Optimizer utilizes a lightweight 0.5B parameter model. Updating this model using the collected rollout data (n = 8) is computationally negligible, taking only minutes to complete.

Based on the parallelization strategy described above, the wall-clock time for one complete outer-loop iteration is determined by the slowest successful inner-loop trial plus evaluation and update overhead.

Ttotal ≈ max(Tinner) + Teval + Tupdate

#### H.2 Resource Estimation

To contextualize the resource requirements of DERL, we compare its cost against the baseline of training a single inner-loop policy model. Let Cinner denote the computational cost required to train one standard inner-loop.

The total computational cost for standard DERL, which runs for Eouter outer-loop epochs with n parallel rollouts per step, can be estimated as:

CDERL ≈ n × Eouter × Cinner

𝜃𝑡−1 = 𝜃𝑡−2 + ∇𝐽𝑡−2𝑖𝑛𝑛𝑒𝑟(𝜃𝑡−2) 𝜃𝑡 = 𝜃𝑡−1 + ∇𝐽𝑡−2𝑖𝑛𝑛𝑒𝑟(𝜃𝑡−1) 𝜃𝑡+1 = 𝜃𝑡 + ∇𝐽𝑡−2𝑖𝑛𝑛𝑒𝑟(𝜃𝑡)

Optimizee 𝜃𝑡−2

𝜃𝑡+1

𝜃𝑡−1

𝜃𝑡

𝑅𝜙𝑡+1

𝑅𝜙𝑡−1

𝑅𝜙𝑡

|𝑣𝑡−2|
|---|

|𝑣𝑡|
|---|

|𝑣𝑡+1|
|---|

𝑣𝑡−1

| |∇𝐽𝑜𝑢𝑡𝑒𝑟(𝜓𝑡+1)<br><br>|
|---|---|

∇𝐽𝑜𝑢𝑡𝑒𝑟 (𝜓𝑡−2)

∇𝐽𝑜𝑢𝑡𝑒𝑟(𝜓𝑡−1)

∇𝐽𝑜𝑢𝑡𝑒𝑟(𝜓𝑡)

Optimizer 𝜓𝑡−2

+ 𝜓𝑡−1

+ 𝜓𝑡

+ 𝜓𝑡+1

+

- Figure 7: Illustration of Gradient Propagation in bi-level evolutionary training loop. The top row (Orange) represents the trajectory of the optimizee θ, updated via instructions ϕ derived from the optimizer. The bottom row (Green) represents the evolution of the Meta-Optimizer ψ. The blue nodes vt denote the validation performance evaluation. Unlike static methods, our framework computes the meta-gradient ∇J outer(ψt) (vertical arrows), allowing the optimizer to update its own parameters ψ to explicitly maximize the optimizee’s performance.

This cost scales linearly with the number of outer-loop iterations required for the Meta-Optimizer to converge. In contrast, for DERL-pop, we simplify the process selecting the best reward function from a single population generation. In this setting, the total cost is significantly reduced to:

CDERL-pop ≈ n × Cinner

Consequently, DERL-pop offers a more efficient alternative, consuming way less wall-clock time while still benefiting from population-based exploration. We utilized high-memory data center accelerators to accommodate the memory requirements of the parallel inner-loop training.

#### H.3 Efficiency Improvements and Future Work

In our current implementation, we utilized a relatively large number of rollouts (n = 8) and a full inner-loop training protocol to empirically verify that LLMs can effectively learn meta-gradients through Reinforcement Learning. However, our preliminary experiments (demonstrated in Section C) suggest that simple parameterizations (e.g., 12 parameters) can also yield competitive results.

This observation points toward a promising direction for future work: reducing the heavy computational burden of the inner-loop by adopting lightweight RL algorithms, such as REINFORCE++. By simplifying the inner-loop requirements or using proxy tasks, the reliance on massive parallel resources can be significantly reduced, making the evolution of Meta-Rewards accessible to a broader range of computational budgets.

Evaluation Cost. Following the inner-loop, we evaluate the validation performance to calculate the advantage for the Meta-Optimizer. We utilize vLLM for high-throughput inference. By parallelizing the evaluation across all n rollouts, the validation phase typically requires only a few minutes.

Outer-Loop Update. The Meta-Optimizer itself utilizes a lightweight 0.5B parameter model. Updating this model using the collected rollout data (n = 8) is computationally negligible, taking only minutes to complete.

### I Gradient Propagation on DERL

We elaborate on the information flow between the meta-optimizer (parameterized by ψ) and the innerloop policy model (the optimizee, parameterized by θ). A distinct feature of our DERL framework is the preservation and utilization of meta-gradient information, which allows the optimizer to explicitly learn from the validation performance of the optimizee.

#### I.1 Gradient Propagation Flow

The interaction between the optimizer and the optimizee unfolds as a bi-level optimization process, as illustrated in Figure 7. Let vt denote the validation performance (or evaluation metric) of the policy model θt at step t. The optimization process consists of two coupled loops:

- • Inner Loop (Optimizee): The policy model updates its parameters θt−1 → θt based on the

guidance of the Meta-Reward Rϕt

provided by the optimizer. The optimizer then generates the update instructions (parameterized by ϕt) conditioned on its current state ψt.

- • Outer Loop (Optimizer): The meta-optimizer evolves ψt−1 → ψt by maximizing the expected future validation performance of the optimizee.

Crucially, the update of the optimizer ψ is driven by the gradient of the validation performance, denoted as ∇J outer(ψt). This term represents the meta-gradient: it quantifies the sensitivity of the optimizee’s performance with respect to the optimizer’s parameters. By backpropagating the signal from the validation performance v through the update step to ψ, our DERL establishes a direct feedback loop.

#### I.2 Meta-Gradient Propagation Compared with Previous Evolution

The core innovation of our framework lies in the end-to-end differentiability of the optimization trajectory, or how the feedback signal vt is utilized. In traditional reinforcement learning or promptbased optimization methods, the optimizer ψ is often treated as a static entity (e.g., a fixed prompted agent or a random perturbation generator). In such cases, the dependency chain is broken, and the “meta-gradient”—the gradient of the validation performance with respect to the optimizer’s parameters—is lost.

In contrast, our approach treats ψ as a learnable entity. We explicitly compute the gradient flow from the evaluation metric back to the optimizer parameters. As depicted in the bottom flow of Figure 7, the optimizer updates its own parameters ψ to maximize the expected future validation performance of the optimizee:

ψt = ψt−1 + η · ∇J outer(ψt−1) (7) This derivation highlights that updating ψ is fundamentally learning the meta-gradient. Unlike prior works where the optimizer is fixed (resulting in ∂ψ∂ϕ = 0 or undefined), our DERL framework maintains a differentiable (or estimable) path. This enables the Meta-Optimizer to iteratively improve the reward structure ϕ driven by direct performance feedback. In doing so, DERL serves as a foundational proof-of-concept for completely autonomous, self-improving frameworks.

### J Broader Impacts

This paper presents Differentiable Evolutionary Reinforcement Learning (DERL), a framework designed to automate the discovery of optimal reward functions for autonomous agents. By parameterizing reward structures and optimizing them via meta-gradients, this work has the potential to significantly impact the field of reinforcement learning in several ways:

Reducing Human Labor and Bias: A primary contribution of DERL is the reduction of reliance on expensive human annotation and heuristic reward engineering. By autonomously exploring the structured reward search space, this framework democratizes access to high-performance agent training in complex domains like robotics and scientific reasoning, where manual reward design is often a prohibitive bottleneck.

Agent Alignment and Safety: The design of reward functions is central to AI alignment. Our empirical analysis demonstrates that DERL evolves “stable structures” that are mathematically robust, potentially offering a more reliable path toward alignment than brittle manual heuristics. However, as with all systems that autonomously evolve objectives, there remains a need for careful interpretability analysis to ensure the generated meta-rewards do not incentivize unintended behaviors in long-horizon tasks.

Computational Considerations: We acknowledge that the bi-level optimization process of DERL is computationally resource-intensive compared to standard single-level reinforcement learning, as it requires multiple inner-loop rollouts for meta-updates. This increases the energy consumption and carbon footprint of training. We have addressed this in part through our population-based variant (DERL-pop), which improves efficiency, and we encourage future research into more sample-efficient outer-loop algorithms to further mitigate these environmental costs.

