# arXiv:2411.14251v3[cs.LG]28May2025

#### Natural Language Reinforcement Learning

Xidong Feng1∗†, Bo Liu2∗†, Yan Song1†, Haotian Fu3†, Ziyu Wan4†, Girish A. Koushik5, Zhiyuan Hu2, Mengyue Yang6, Ying Wen4, Jun Wang1 1University College London, 2National University of Singapore, 3Brown University, 4Shanghai Jiao Tong University, 5University of Surrey, 6University of Bristol {fengxidongwh}@gmail.com

###### Abstract

Artificial intelligence progresses towards the "Era of Experience," where agents are expected to learn from continuous, grounded interaction. We argue that traditional Reinforcement Learning (RL), which typically represents value as a scalar, can restrict agent’s deep understanding of environments and hinders the active, deliberative learning crucial for navigating this new paradigm. To address the issue, we introduce Natural Language Reinforcement Learning (NLRL), a framework that extends RL principles into natural language counterparts. Central to NLRL is the Language Value Function (LVF), which redefines value as an interpretable linguistic narrative articulating the rationale behind an evaluation. NLRL further extends this concept to core RL components, including policy, the Bellman equation, and policy iteration. Leveraging recent advancements in Large Language Models (LLMs), NLRL can be practically implemented to achieve RL-like policy and value training through unsupervised environment interactions. Experiments over 4 multi-step agentic tasks demonstrate NLRL’s effectiveness, efficiency, and its potential to foster deeper understanding and more active learning strategies.

###### 1 Introduction

The pursuit of artificial intelligence is rapidly advancing towards the "Era of Experience," where agents are expected to learn and master complex tasks through continuous, grounded interaction with their environments, thereby moving beyond the inherent limitations of pre-existing human data [40]. Central to this paradigm shift is the ability of agents to not only act but also to deeply understand and evaluate their experiences to inform future decisions. Traditional Reinforcement Learning (RL) [43] has laid a robust mathematical foundation for such decision-making, primarily through the concept of a value function, which estimates the cumulative reward of states or state-action pairs.

The conventional representation of value is scalar quantity [44], acting as highly compressed summaries of future trajectories. While computationally convenient for optimization, it fundamentally limits an agent’s capacity for a deep, structural understanding of its experiences. This limitation becomes particularly salient when considering the vast, nuanced information embedded within an agent’s interaction stream – information that a single numerical value struggles to encapsulate. For instance, why is a particular state valuable? Is it due to an immediate tactical advantage, a long-term strategic opportunity, a successful mitigation of a threat, or a combination thereof? A scalar value provides the "what" (how good) but not the "why." Without this crucial "why," the agent operates with a superficial grasp of its environment and the true drivers of long-term success.

This superficial understanding, in turn, shapes the very nature of the learning process. While policy optimization methods like policy gradients [51, 44] effectively reinforce successful behaviors, their

*Equal contribution, correspondence to fengxidongwh@gmail.com †Core contributors.

Preprint. Under review.

efficacy is largely predicated on a "passive" discovery through sampling: the agent can only refine its strategy by first observing the outcomes of sampled actions, essentially waiting to stumble upon and then amplify beneficial action sequences. Without a richer, more interpretable grasp of why certain actions yield better outcomes, it is challenging to shift from passive sampling. It cannot engage in "active" or "deliberative" reasoning over the root causes of success or failure to proactively and systematically adjust its strategies. To truly unlock the potential of experiential learning and build agents capable of nuanced understanding, transparent reasoning, and more active, deliberative learning, we argue for a richer, more expressive representation of value function.

In this paper, we propose a novel approach, Natural Language Reinforcement Learning (NLRL), that directly addresses this challenge by re-envisioning the core components of RL within the domain of natural language. At the heart of NLRL is the concept of the Language Value Function (LVF). Instead of a simple scalar, the LVF redefines value as a rich, interpretable linguistic narrative. LVF allows an agent to articulate why a state or action is valuable, drawing justifications from textualized experiences and observations. This richer representation of value is crucial for learning actively from the diverse and often nuanced feedback characteristic of the "Era of Experience."

Building upon the Language Value Function, our NLRL framework further extends other key RL elements – including policy, Bellman equation, and policy iteration – into their language counterparts. We demonstrate how these components can be practically implemented with the recent success of large language models (LLMs). Our experiments in 4 multi-step agentic environments showcase the effectiveness, efficiency, and particularly potentials for more active learning within the NLRL framework compared to traditional RL approaches, paving the way for agents that learn not just to act, but to understand, reason, and actively improve. The codebase is available.

###### 2 Preliminary of Reinforcement Learning

RL models the decision-making problem as a Markov Decision Process (MDP), defined by the state space S, action space A, probabilistic transition function 𝑃 : S × A × S → [0, 1], discount factor 𝛾 ∈ [0, 1) and reward function 𝑟 : S × A → [−𝑅max, 𝑅max]. The goal of RL aims to learn a policy 𝜋 : S × A → [0, 1], which measures the action 𝑎’s probability given the state 𝑠: 𝜋(𝑎|𝑠) = Pr (𝐴𝑡 = 𝑎 | 𝑆𝑡 = 𝑠). In decision-making tasks, the optimal policy tends to maximize the expected discounted cumulative reward: 𝜋∗(𝑎|𝑠) = argmax𝜋 𝔼𝜋 ∞𝑡=0 𝛾𝑡𝑟 (s𝑡, a𝑡) . The state-action and state value functions are two key concepts that evaluate states or state-action pairs by measuring the cumulative reward starting from them: 𝑄𝜋(𝑠𝑡, 𝑎𝑡) = 𝔼(𝑠,𝑎)𝑡+1:∞∼𝑃𝜋 ∞𝑖=𝑡 𝛾𝑖−𝑡𝑟 (𝑠𝑖, 𝑎𝑖) | 𝑠𝑡, 𝑎𝑡 , 𝑉𝜋(𝑠𝑡) = 𝔼𝑎𝑡,(𝑠,𝑎)𝑡+1:∞∼𝑃𝜋 ∞𝑖=𝑡 𝛾𝑖−𝑡𝑟 (𝑠𝑖, 𝑎𝑖) | 𝑠𝑡 , where 𝑃𝜋 is the trajectory distribution given the policy 𝜋 and dynamic transition 𝑃. Given the definition of 𝑉𝜋(𝑠𝑡), the relationship between temporally adjacent state’s value (e.g., 𝑉(𝑠𝑡) and 𝑉(𝑠𝑡+1)) forms the Bellman expectation equation [5]. For all states 𝑠𝑡 ∈ S, the one-step Bellman expectation equation is:

𝑉𝜋(𝑠𝑡) = 𝔼𝑎𝑡∼𝜋𝜃 [𝑟(𝑠𝑡, 𝑎𝑡) + 𝛾𝔼𝑠𝑡+1∼𝑝(𝑠𝑡+1|𝑠𝑡,𝑎𝑡)[𝑉𝜋(𝑠𝑡+1)]],∀𝑠𝑡 ∈ S, (1)

Given these basic RL definitions and equations, many RL algorithms fall in the scope of generalized policy iteration (GPI). GPI is an iterative process over policy evaluation and policy improvement.

Policy Evaluation. The target of the policy evaluation process is to estimate state value function 𝑉𝜋(𝑠) or state-action value function 𝑄𝜋(𝑠, 𝑎) for a given policy 𝜋. For simplicity, we only utilize 𝑉𝜋(𝑠) in the following illustration. Two common value function estimation methods are the Monte Carlo (MC) estimate and the Temporal Difference (TD) estimate [42]. Starting from the definition of 𝑉𝜋(𝑠𝑡), the Monte Carlo (MC) estimate uses sampling over complete trajectories to calculate an unbiased estimate: 𝑉𝜋(𝑠𝑡) ≈ 𝐾1 𝑛 𝐾=1

∞

𝑖=𝑡 𝛾𝑖−𝑡𝑟(𝑠𝑖𝑛, 𝑎𝑖𝑛) where we average the cumulative rewards over multiple full paths starting from 𝑠𝑡. The Temporal Difference (TD) estimate, on the other hand, builds on the temporal relationship between states without requiring complete

trajectories. It estimates 𝑉𝜋(𝑠𝑡) using the immediate reward and the estimated value of the next state: 𝑉𝜋(𝑠𝑡) ≈ 𝐾1 𝑛 𝐾=1 𝑟(𝑠𝑡, 𝑎𝑡𝑛) + 𝛾𝑉𝜋(𝑠𝑡𝑛+1) . This approach effectively uses a "bootstrap" by depending on the estimated value of the next state to approximate 𝑉𝜋(𝑠𝑡).

Policy Improvement. The policy improvement process aims to update and improve policy according to the result from the policy evaluation process. Specifically, it replaces the old policy 𝜋old with the new one 𝜋new to increase the expected return: 𝑉𝜋new (𝑠𝑡) ≥ 𝑉𝜋old (𝑠𝑡). In the environment with

small, discrete action spaces, such improvements can be achieved by greedily choosing the action that maximizes 𝑄𝜋old(𝑠𝑡, 𝑎):

𝜋new(· | 𝑠𝑡) = argmax

𝔼𝑎∼𝜋¯ 𝑄𝜋old(𝑠𝑡, 𝑎) , (2)

𝜋¯ (·|𝑠𝑡 )∈P(A)

Another improvement method involves applying policy gradient ascent [44]. It parameterizes the policy 𝜋𝜃 with 𝜃 and we can derive the analytical policy gradient:

∇𝜃𝑉𝜋𝜃 (𝑠𝑡) = 𝔼𝑎∼𝜋𝜃 ∇𝜃 log 𝜋𝜃(𝑎|𝑠𝑡)𝑄𝜋𝜃 (𝑠𝑡, 𝑎) , (3) By choosing a small step-size 𝛼 > 0 to conduct gradient ascent: 𝜃new = 𝜃 + 𝛼∇𝜃𝑉𝜋𝜃 (𝑠𝑡) 𝜃=𝜃

, we can guarantee the policy improvement: 𝑉𝜋new (𝑠𝑡) ≥ 𝑉𝜋old (𝑠𝑡).

old

###### 3 Motivation and Methodology

###### 3.1 Discussion: policy gradient is a bit ’passive’ because of scalar 𝑄𝜋𝜃 (𝑠𝑡, 𝑎)

Policy gradient (Equation (3)) is widely used in RL training, especially in recent LLM post-training trends. RL algorithms like PPO [37] leads to significant success in reasoning [21, 18] and agent [24, 48]. These successes, however, often rely heavily on the pre-training model’s prior. For example, smaller models may fail with RL training [18]; the capacity for test-time reasoning "reflection" may not emerge from RL, but already existed in the pre-training model [30, 38]. Moreover, [48] highlights that directly employing RL to optimize Chain-of-Thought (CoT) enhanced actions in multi-turn agentic environments can lead to CoT degradation.

The underlying cause of these challenges, we argue, stems from the inherent passive nature of policy gradient. Policy gradient algorithms improve policies by incrementally adjusting sampled action probabilities based on the scalar 𝑄𝜋𝜃 (𝑠𝑡, 𝑎). Effective policy improvement, therefore, becomes contingent on the agent’s ability to sample high-quality reasoning chains and actions. Since 𝑄𝜋𝜃 (𝑠𝑡, 𝑎) only provides scalar feedback without semantic information, the agent has to passively ’wait’ for beneficial actions to arise through chance, rather than actively leveraging reasoning to anticipate and strategically guide its choices. This inherent reliance on effective sampling explains why a strong pre-training model is a prerequisite for successful RL: it increases the likelihood of encountering high-reward trajectories, providing a more fertile ground for subsequent policy refinement.

The left block in Figure 1 shows an illustration of RAGEN [48] that trains CoT-based LLM agent with RL. RAGEN samples thinking and action from the LLM, calculate the value by taking MC estimate from future rollout, then update the LLM with policy gradient. Such MC estimate presents an extremely aggressive compression of future rollouts: it averages over the entire distribution of future outcomes, and discarding all concrete state, action or reward information, let alone any analysis. This brings in passivity: CoT may degrade if the LLM prior cannot generate diverse, beneficial thinking/action. An example of such degradation will be presented in Figure 2.

###### 3.2 Language value function brings in active and deliberative learning

To overcome these limitations, we argue for a different approach to compressing future rollouts. Instead of collapsing the future into a single number, we propose Language Value Function, that leverages the expressive power of natural language, processed by the underlying LLM, to create a richer, more informative summary. This approach offers several potential advantages. First, it utilizes the LLM’s inherent language understanding, reasoning and summarization capabilities, potentially drawing upon its pre-trained knowledge to generate high-quality, semantically meaningful

Traditional RL NLRL

LLM

LLM

Thought + Action samping Future rollout

Thought + Action samping Future rollout

...

rt st+1 at+1 ... sK

Thinking Process

Thinking Process

at

Language-based reasoning Passive learning

Only keep reward

Active reasoning

MC estimate

E.g., LPI

E.g., Policy gradient

Figure 1: Comparison between NLRL and traditional RL in agentic LLM

RAGEN

[Figure 1]

THINK: My thoughts are to go to the goal. Action: Up

THINK: Based on the evaluations of the next board positions, moving up results in the highest final evaluation of

NLRL

+5, indicating that P is one block closer to the goal position G after the move. Although there is a hole one block away from P after moving up, the potential strategy of moving up and then moving right can help P avoid the hole and arrive at the goal position. In contrast, moving left does not immediately change P's position, and moving down moves P away from the goal position G. Therefore, moving up is the most favourable move. Action: Up

- Figure 2: Comparing after-train reasoning on FrozenLake between RAGEN [48] and NLRL (Ours). Traditional RL results in a meaningless CoT, while NLRL keeps informative CoT and active reasoning.

descriptions of future possibilities. Second, it retains significantly more information than a scalar value – capturing aspects of the strategy, reasoning steps, potential pitfalls, or key events – without the impracticality of storing or processing entire raw trajectory distributions. This strikes a balance, achieving useful compression without excessive information loss.

The language state value function 𝑉𝜋𝐿 and language state-action value function 𝑄𝐿𝜋 are defined as:

𝑄𝐿𝜋(𝑠𝑡, 𝑎𝑡) = 𝐷𝐿 (s, a, r)𝑡+1:∞ ∼ 𝑃𝜋 | 𝑠𝑡, 𝑎𝑡 ,𝑉𝜋𝐿(𝑠𝑡) = 𝐷𝐿 at, (s, a)𝑡+1:∞ ∼ 𝑃𝜋 | 𝑠𝑡 (4)

where 𝐷𝐿 is a language descriptor that can transform future trajectory distribution (𝑠, 𝑎,𝑟)𝑡+1:∞ ∼ 𝑃𝜋 into its corresponding language description.

The right block in Figure 1 encapsulates the features of LVF in the agentic LLM example. The LVF utilizes the LLM to perform language-based reasoning on the future rollout, generating an evaluation that captures nuanced details and offers an informative evaluation of potential outcomes. This semantically rich feedback facilitates active reasoning and strategic analysis for policy adjustment—a stark contrast to passive learning. We argue that this active reasoning path can be viewed as the agent’s deliberative thinking process over task’s structure. Figure 2 shows an example of after-train policy CoT comparison between traditional value and LVF guidance. The next section will detail how language policy improvement (LPI) can be one way to implement such active reasoning process.

###### 3.3 Natural language reinforcement learning

In addition to LVF, we take one-step further: verbalizing all RL components presented in Section 2, resulting in the NLRL framework. This includes language policy, language policy evaluation and improvement. Next, we demonstrate how LLMs can function in various roles.

- ① LLMs as language policy (𝜋𝐿). Many works adopted LLMs as the decision-making agent [46, 15, 10, 54] with Chain-of-thought process [50]. We represent the policy as 𝜋𝐿(𝑎, 𝑐|𝑠) = 𝜋𝐿(𝑐|𝑠)𝜋𝐿(𝑎|𝑐, 𝑠), including thought process 𝜋𝐿(𝑐|𝑠), and the final action probability 𝜋(𝑎|𝑐, 𝑠).
- ② LLMs as language value function (𝑄𝐿,𝑉𝐿). As mentioned in Section 3.2, we leverage LLMs, to serve as the language value function 𝑄𝐿,𝑉𝐿. This exactly corresponds to what (multi-modal) LLMs are capable of – they are designed to take in the features from the task state, such as low-dimension statistics, text, or images, and output the corresponding language understanding. By further prompting or fine-tuning over evaluation dataset, LLMs can generate language assessment.
- ③ LLMs as language Monte Carlo and TD operator (𝐺1, 𝐺2). One core issue for language value function is: how can we generate language value estimate to train the language value function? Inspired by traditional policy evaluation methods, we propose to verbalize traditional MC and TD operations, resulting in language MC or TD estimate. Check Appendix E for more about how language MC/TD corresponds to the traditional MC/TD.

Language MC estimate is conducted over text rollouts (i.e. 𝐾𝑀𝐶 full trajectories {𝑎𝑡, (𝑠, 𝑎)𝑡+1:∞}) given the policy 𝜋𝐿. Specifically, we have 𝐺1 that summarizes and extract the core information from different trajectories, generating language MC estimate:

𝑉𝜋𝐿(𝑠𝑡) ≈ 𝐺1 ant , (s, a)𝑡𝑛+1:∞

𝐾𝑀𝐶 𝑛=1 , (5)

Language TD is over 𝐾𝑇𝐷 few-step transitions. By drawing an analogy to Equ. 1, we have: 𝑉𝜋𝐿(𝑠𝑡) ≈ 𝐺1 𝐺2 𝑑(𝑠𝑡, 𝑎𝑡𝑛,𝑟(𝑠𝑡, 𝑎𝑡𝑛), 𝑠𝑡𝑛+1),𝑉𝜋𝐿(𝑠𝑡𝑛+1) 𝑛 𝐾=𝑇𝐷1 ,∀𝑠𝑡 ∈ S, (6)

where 𝑑 (𝑎𝑡,𝑟 (𝑠𝑡, 𝑎𝑡) , 𝑠𝑡+1)) depicts the language description of intermediate transition, while 𝐺1 and 𝐺2 serves as two language aggregation functions. Looking back to Equ. 1, 𝐺2 mimics the add

|You are the player 'O' in tic-tac-toe, given current board state,<br><br>to win the game, what's your next move?<br><br>1 LLMs as Language Policy<br><br>Response:<br><br>{<br><br>"Thought": The 'O' should block xxxxx, "Action": 1 (top middle square)<br><br>}<br><br>Query:<br><br>You are the player 'O' in tic-tac-toe, given current board state,<br><br>You next move is action 2, what's your assessment about this move?<br><br>LLMs as Language Value Function Approximator<br><br>2<br><br>Response:<br><br>{<br><br>"Thought": The action 2 (top right) xxxxx, "Final Evaluation": xxxxx<br><br>}<br><br>board board<br><br>Query:<br><br>You are the player 'O' in tic-tac-toe, given variations and subsequent evals, generate the evaluation for current state-action pair:<br><br>Response:<br><br>{<br><br>"Thought": By reviewing and aggregating these variations and subsequent evals, xxxx<br><br>"Final Evaluation": action 2 seems good xxxx }<br><br>......<br><br>Move<br><br>One-step Lookahead<br><br>{ "Variation": 'X' plays 7. "Thought": xxxxx, "Final Evaluation":<br><br>'O' can win in one step. }<br><br>{ "Variation": 'X' plays 0. "Thought": xxxxx, "Final Evaluation":<br><br>'O' needs to block 'x'. }<br><br>3<br><br>3<br><br>LLMs as Language Monte-Carlo Or Temporal Difference operator<br><br>Query:<br><br>Query:<br><br>You are the player 'O' in tic-tac-toe, given current board state and move, evaluate the state-action by aggregating the trajectories:<br><br>Response:<br><br>......<br><br>Move<br><br>Rollout to the End<br><br>{<br><br>"Thought": By reviewing and aggregating these trajectories after taking move 2, xxxx<br><br>"Final Evaluation": action 2 is good xxx }<br><br>You are the player 'O' in tic-tac-toe, given current board state, candidate moves and their evaluations, what is the best move?<br><br>5<br><br>Response:<br><br>Query:<br><br>{ "Thought":<br><br>xxxxx, "Final Evaluation":<br><br>good }<br><br>{ "Thought":<br><br>xxxxx, "Final Evaluation":<br><br>bad }<br><br>{ "Thought":<br><br>xxxxx, "Final Evaluation":<br><br>bad }<br><br>LLMs as Policy Improvement Operator<br><br>{<br><br>"Thought": By comparing these action candidates, xxxx "Final action": 2 (top right square)<br><br>}<br><br>4 Distilling to Train Language<br><br>Value function and Policy<br><br>6<br><br>Language Policy<br><br>Language Value Function<br><br>Response:<br><br>{<br><br>"Thought": The 'O' should block 'X' xxxx "Action": 2 (top right square) "Evaluation": action 2 is a good move while<br><br>in other cases 'X' wins }<br><br>3<br><br>Buffer D<br><br>5|
|---|

- Figure 3: Practical pipeline for implementing NLRL in the Tic-tac-toe game. LLMs can serve as the language policy ①, the language-based value function approximator ②, the language Monte Carlo or Temporal Difference operator ③, and the language policy improvement operator ⑤. By distilling (④, ⑥) the improved evaluations from ② and the enhanced actions from ⑤, the NLRL agent can iteratively refine its language policy and evaluation capabilities.

the summation operation ‘+’, aggregating information from intermediate transition’s description 𝑑 and future evaluation 𝑉𝜋𝐿(𝑠𝑡+1), while 𝐺1 serves the role of the expectation operator 𝔼, aggregating information accross different (𝑎𝑡, 𝑠𝑡+1) pairs. By proper promoting, LLMs can serve as such language aggregator 𝐺1, 𝐺2 to fuse language information [58, 52].

- ④ Training LVF approximator with language value estimation. ③ provides an unsupervised and scalable way to generate language evaluation data through environment interaction, which can be leveraged to train our LVF approximator in ②. This corresponds to traditional critic training for value-based RL (e.g, DQN [32]) or actor-critic algorithms (e.g, PPO [37]), but happens in natural language representation space. The only difference is the training objective – the scalar-based value function regresses the value mean with mean squared error while LVF tries to imitate the language aggregation result with supervised-finetuning’s cross-entropy loss.
- ⑤ LLMs as policy improvement operator (𝐼). As mentioned in Section 3.2, LVFs offer potentials to deliberative learning. Inspired by policy improvement (Equ. 2 and policy gradient) in traditional RL, we propose language policy improvement (𝐼) to achieve as one way for deliberative learning. Specifically, for a given state 𝑠, we prompt the LLMs with several action candidates and their corresponding language evaluations 𝑄𝐿, by which we can obtain the improved action with a chainof-thought process analyzing different actions’ evaluation and proposing better action. With CoT and prior knowledge about the world, LLMs are good at determining the most promising action 𝜋new(· | 𝑠) by reasoning on different language evaluation results 𝑄𝐿𝜋old(𝑠, 𝑎).
- ⑥ Training language policy with improved policy Similar to ④, we can train our language policy LLM in ① by supervised-finetuning over the improved chain-of-thought based language policy data from ⑤. This corresponds to the policy training in traditional RL.

We also provide example visualizations in Fig 3 to illustrate these components and procedures. Building upon these components and procedures, we can create various NLRL applications. Here we illustrate three use cases, though many more possibilities exist beyond these examples.

###### 3.4 Language GPI boosts LLM’s critic and policy by prompting (①, ②, ③, ⑤)

Our first case utilizes language GPI to enhance LLM’s critic and policy solely through prompting, which can be particularly beneficial for improving proprietary models such as GPT-4 [33] or Gemini [45]. Specifically, we first combine ①, ②, ③ to build a language policy evaluation pipeline. Take

language TD shown in Equ. 6 as an example. We prompt LLMs to (1) evaluate the subsequent state’s value 𝑉𝜋𝐿(𝑠𝑡+1) (②) (2) serve as the TD operator 𝐺1, 𝐺2 (③). By performing a one-step look-ahead with 𝑎𝑡 ∼ 𝜋 (①) and leveraging 𝐺1, 𝐺2 to aggregate information from intermediate transition 𝑑 alongside the subsequent evaluation 𝑉𝜋𝐿(𝑠𝑡+1), language TD can generate new and improved language evaluation, 𝑉𝜋𝐿(𝑠𝑡)new. Building upon this new evaluation, LLM-based policy improvement operator 𝐼 (⑤) converts the evaluation into a better policy 𝜋new. Refer to Algorithm 1 for detailed procedures.

###### 3.5 Training natural language value function for a given policy (②, ③, ④)

Our second case aims to train an LLM critic capable of evaluating any given state with natural language explanations, similar to a chess annotator who provides insightful commentary on boards, moves, and strategies. For example, we can build an iterative language TD pipeline by combining ②, ③, and ④. First, we leverage a tunable LLM 𝐴 and prompt it to become a language value function approximator ②. Combined with the look-ahead transitions (𝑎𝑡, 𝑠𝑡+1) ∼ 𝑃𝜋 by taking rollouts with policy 𝜋, and the subsequent state evaluation 𝑉𝜋𝐿(𝑠𝑡+1) generated by 𝐴, we prompt LLM 𝐵 for the language TD estimate (③), similar to Sec 3.4. The model 𝐴 is further finetuned by such language TD estimates (④) and will be plugged back for 𝑉𝜋𝐿(𝑠𝑡+1)new in a new iteration. Iteratively, we can obtain the final converged natural language value function. Check Algorithm 2 for more details.

###### 3.6 Natural Language Actor-Critic Learning (①, ②, ③, ④, ⑤, ⑥)

Last, we combine all these procedures and build the full natural language actor-critic pipeline. Similar to traditional actor-critic [4, 44], our natural language actor-critic framework simultaneously learns both a language policy and a language critic through environment interactions. For each iteration, we first use language policy ① to take rollouts in the environment. With new trajectories, we update our language value model ② by language MC or TD ③ and train it with supervised-finetuning loss (④). For language policy improvement, we query our updated language value to evaluate action candidates for states from the rollout trajectories. Further LLM-based improvement operator ⑤ provides improved CoT data to train our language policy (⑥). Check Algorithm 3 for more details.

###### 4 Experiments

###### 4.1 Language GPI by prompting (Sec 3.4)

Our first experiments explore Sec 3.4 and leverage language GPI to improve LLM capability with pure prompting. Specifically, we choose the maze games of LMRL [1], aiming to validate that Language TD Estimate and Language Policy Improvement can benefit the evaluation and further improve policy. We use the original settings of LMRL Gym Maze, where the agent is required to navigate to a goal position in a “Double T” or “Medium” maze. We consider the fully-observable setting, where the agent’s observation (described by text) includes the agent’s current position in the maze, the agent’s action history, the walls’ position around the agent (if any), and the goal position. The action space is discrete, including moving up / down / right / left. We evaluate the performance on 30 different initial positions, each with 3 random seeds. Refer to Appendix C.1.2 for more details.

For language TD estimate in Equ. 6, we prompt gpt-4o-mini-2024-07-18 as the language aggregator 𝐺1, language state-action value aggregator 𝐺2, and language state value function 𝑉𝜋𝐿 respectively. Specifically, given an environment state, for each candidate action, we use a fully random policy to rollout 𝑁 steps into the future and use the language state value function 𝑉𝜋𝐿 to evaluate the look-ahead state. For each state we repeat this process for 𝐾 times and use 𝐺2 to aggregate into a state-action value estimation. The same GPI-4o-mini is leveraged as the policy improvement operator 𝐼.

We compare Language GPI with a few different baselines, including prompt-based language policy 𝜋𝐿 (①), as well as prompt-based language value function 𝑄𝐿(𝑠, 𝑎) + language policy improvement 𝐼. As shown in Table 1, Language TD produces better performance than the prompt-based language value in both mazes, and more variations & look ahead steps can have even better results. Language policy improvement can only benefit stably for language-TD-enhanced language value (② and ③).

Table 1: Language GPI results with ablations on look-ahead steps 𝑁 and variations number 𝐾.

Avg Reward Double T Maze Medium Maze

Language policy 𝜋𝐿(𝑠) (①) −27.29 ± 4.43 −27.05 ± 5.27 Language value function 𝑄𝐿(𝑠, 𝑎) + improvement 𝐼 (②,⑤) −18.33 ± 6.11 −33.57 ± 14.41 Language GPI (1 variation, 3 look ahead steps) (②, ③, ⑤) −17.85 ± 3.68 −20.85 ± 7.59 Language GPI (4 variations, 1 look ahead steps) (②, ③, ⑤) −17.48 ± 4.53 −12.65 ± 4.72 Language GPI (4 variations, 3 look ahead steps) (②, ③, ⑤) −12.74 ± 4.47 −15.09 ± 4.44 Language GPI (6 variations, 3 look ahead steps) (②, ③, ⑤) −12.15 ± 2.96 −15.44 ± 4.97 Language GPI (8 variations, 3 look ahead steps) (②, ③, ⑤) −11.19 ± 2.86 −12.23 ± 4.49

(a) Performance vs Baselines

(b.1) Variation Number (Train set)

(b.2) Variation Number (Test set)

0.85

0.80

0.85

0.80

0.80

EvaluationAccuracy

EvaluationAccuracy

EvaluationAccuracy

0.75

0.75

0.75

Best LVF

GPT4o

0.70

LLaMA3.1-70B

0.70

0.70

LLaMA3.1-8B

0.65

0.60

0.65

0.65

- Variation=1

- Variation=2

- Variation=1

- Variation=2

0.55

0.60

Variation=4

Variation=4

0.60

0 10 20 30 40 50 Iteration

0 10 20 30 40 50 Iteration

0 10 20 30 40 50 Iteration

(c) State Scaling Law (Test set)

(d.1) Look-ahead Step Number (Train set)

(d.2) Look-ahead Step Number (Test set)

0.90

0.85

0.80

0.85

0.80

EvaluationAccuracy

EvaluationAccuracy

EvaluationAccuracy

0.75

0.80

0.75

0.75

0.70

0.70

0.70

0.65

Size=0.4k

0.65

0.65

Size=2k

Step=2 Step=4 Step=8

Step=2 Step=4 Step=8

0.60

Size=10k Size=45k

0.60

0.60

0 10 20 30 40 50 Iteration

0 10 20 30 40 50 Iteration

0 10 20 30 40 50 Iteration

- Figure 4: Breakthrough experiment results. (a) Performance comparison with baselines. (b,d) Ablation study over look-ahead step number and variation number. (c) Results for state scaling law.

###### 4.2 Training Natural Language Value Function with language TD (Sec 3.5)

Our second experiment is an implementation of Sec 3.5. Specifically, we aim to train a language value function with language TD in the 5x5 breakthrough board game (Fig 8). Breakthrough is a turn-based board game for two players, white and black. Each player has a few lines of pawns. Black moves first and in each turn, the player chooses one piece to move one square forward (straight or diagonal) onto empty squares or capture diagonally. The goal is to break through the opponent’s line and reach their home row. As mentioned in Sec 3.5, this value function can serve as a reliable board evaluator/annotator. However, evaluating board states in this game is challenging for LLMs. First, the 5x5 breakthrough variant is niche, leading to sparse pre-training data on board evaluations. Most LLMs have minimal knowledge of it. Second, despite the small board size, its state space complexity can reach 108 [36], making natural language assessments significantly more difficult. Refer to Appendix C.2 for more experimental details and result visualizations.

###### 4.2.1 Experimental Results

Experimental setups. To train a reliable board annotator, we select a strong or near-optimal policy—Monte-Carlo tree search (MCTS) algorithm [26] from OpenSpiel [28], with a high number of simulations and rollouts to ensure an extensive search. For the language TD operator (𝐺1, 𝐺2), we prompt the large LLaMA-3.1-70B-Instruct [14] model since it preserves stronger aggregation and reasoning ability. The language value function 𝑉𝜋𝐿(𝑠) is trained from the small LLaMA-3.1-8BInstruct model. For evaluation, we adopt a coarser metric—judging which side has the advantage. For any given state, we estimate win rates using Monte Carlo simulations to the game outcome. The ground-truth label is assigned to the side with a win rate exceeding a predefined threshold.

Performance vs Baselines. Our fig 4.a shows the evaluation accuracy comparison on 𝐷𝑡𝑒𝑠𝑡𝑠 , between our best language value function’s training curve and prompting-based LLMs (LLaMA-3.1-70B, LLaMA-3.1-8B and GPT-4o-08-06), which clearly demonstrates the necessity of language TD training. Due to the lack of domain knowledge, all prompting-based LLMs perform poorly. The best accuracy (0.61) is only slightly higher than the random guess (0.5), while our trained language value function dominates the task and the accuracy reaches more than 0.85 on the test set 𝐷𝑡𝑒𝑠𝑡𝑠 .

###### (a) Random-Move Opponent

(b) Ablations on Components

(c) Ablations on Number of Training Epochs

(d) Ablations on KMC

(e) Ablations on Rollout

0.9

0.9

0.9

0.9

+0.50

0.8

0.8

0.8

0.8

0.8

0.7

WinRate

WinRate

WinRate

WinRate

WinRate

0.7

0.7

0.7

0.7

0.6

0.6

0.6

0.6

0.6

Natural Language Actor Critic

w/o KMC w/o Kbuffer w/o Action Selection Mask

PPO

0.5

0.5

0.5

0.5

w/o KMC w/o Kbuffer w Action Selection Mask

GPT-4o as Policy

- Train Epoch = 1

- Train Epoch = 2

- Train Epoch = 3

KMC = 1 KMC = 5 KMC = 10

Rollout = 128 Rollout = 256 Rollout = 512

0.5

w/o KMC w Kbuffer w Action Selection Mask

LLaMA-3.1-70B-Instruct as Policy

w KMC w Kbuffer w Action Selection Mask

LLaMA-3.1-8B-Instruct as Policy

0.4

0.4

0.4

0.4

0 5 10 15 20 25 30 Iteration

0 5 10 15 20 25 30 Iteration

0 5 10 15 20 25 30 Iteration

0 5 10 15 20 25 30 Iteration

0 5 10 15 20 25 30 Iteration

- Figure 5: Natural Language Actor Critic Pipeline training results. (a) Training results against the

Random-Move Opponent. (b) Ablation study on components (𝐾𝑀𝐶, 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟, and Action Selection Mask). These results demonstrate that our proposed Natural Language Actor Critic pipeline can stably improve under stochastic dynamics. (c) - (e) Ablation studies on number of training epochs, Monte Carlo sample size 𝐾𝑀𝐶, and number of rollout trajectories.

Look-ahead Ablation. Fig 4.b and Fig 4 shows further ablation study over variation number (𝑚) and look-ahead steps 𝑘. The test set performance is shown in Fig 4.b.2 and Fig 4.d.2, while Fig 4.b.1 and Fig 4.d.1 present a subset of in-training state evaluations. The ablation study indicates that determining the appropriate 𝑚 and 𝑘 is crucial for optimal performance. Notably, in Fig 4.d.1 and Fig 4.d.2, using 8 steps resulted in significantly higher in-training accuracy but slightly lower test set accuracy compared to 4 steps. This suggests that longer look-ahead steps may lead to overfitting, potentially hindering generalization to novel states.

State Scaling Law. We also examine how scaling the training state size affects performance in Fig 4.c, with datasets sizes 0.4k, 2k, 10k, and 45k. The results demonstrate robust scalability of our algorithm, with consistent improvements in test set performance as the training set size increases.

###### 4.3 Natural Language Actor-Critic Learning (Sec 3.6)

For validating Sec 3.6, we implement the natural language actor-critic pipeline in a model-free setting [43] on the Tic-tac-toe game [25] and FrozenLake (used in [48]). As a supplement to language TD in Sec. 4.1 and 4.2, we utilize the language MC (Equ. 5) for the language policy evaluation.

###### 4.3.1 Experimental Results on tic-tac-toe

Experiment setup. Our implementation uses three language models: one LLaMA-3.1-70B-Instruct and two LLaMA-3.1-8B-Instruct models. The 70B model serves as the language aggregator (𝐺1) and policy improvement operator 𝐼. The two 8B models implement our trainable components: the language policy 𝜋𝐿 that generates actions through CoT reasoning, and the LVF 𝑄𝐿𝜋 that evaluates state-action pairs. To thoroughly evaluate our approach, we test against two different types of opponents. The results against a deterministic opponent that always takes the first available action can be found in Appendix C.3.8. For our main experiments, we focus on a stochastic opponent that plays randomly, presenting a more challenging case for policy evaluation. We measure performance through game win-rate. To stabilize training and address value function instability, we implement two key techniques (More details in Appendix C.3.3): (1) action selection masking that restricts policy improvement to the top-𝑚 moves from 𝑁𝑠𝑎𝑚𝑝𝑙𝑒 samples, preventing hallucination on unseen state-action pairs; and (2) buffer merging that combines the last 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟 value function training sets to mitigate catastrophic forgetting. Refer to Appendix C.3 for more experimental details.

Performance vs Baselines. The learning curves in Fig. 5(a) demonstrate our system’s superiority in the random-move opponent setting. Our algorithm outperforms all other baselines (promptingbased method using GPT-4o-08-06, LLaMA-3.1-8B/70B-Instruct with CoT reasoning), including the PPO [37] fine-tuned LLaMA-3.1-8B-Instruct baseline. The improvement is substantial (0.4 to 0.9), demonstrating NLRL’s robustness despite the stochastic dynamics. As shown in Fig. 5(b), each component (Monte Carlo sampling number 𝐾𝑀𝐶, buffer merging size 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟, and action selection mask) enhances the system’s performance, with the full system achieving the highest win rate.

Ablation Studies We conduct comprehensive ablation studies to investigate key hyperparameters. First, training with 128 trajectories per iteration shows that increasing from 1 to 3 epochs significantly improves stability and adaptation, as shown in Fig. 5(c). Second, our investigation of Monte Carlo sampling with 𝐾𝑀𝐶 values of 1, 5, and 10 shows in Fig. 5(d) that increased sampling with 512 trajectories leads to more stable training. Finally, Fig. 5(e) shows how the number of rollout trajectories affects training, with 512 trajectories per iteration yielding the most robust learning curve. Additional ablation studies on action selection mask parameter top-𝑚 and effect of experience

(a) FrozenLake (Deterministic)

(b) FrozenLake (Stochastic)

0.35

0.7

EvaluationSuccessRate

EvaluationSuccessRate

0.30

LLaMA3.1-70B GPT4o

GPT4o

0.6

0.25

0.5

LLaMA3.1-70B

0.20

LLaMA3.1-8B

NLRL

0.4

RAGEN (Qwen0.5B)

0.15

RAGEN (LLaMA3.1-8B)

0.3

0.10

0.2

LLaMA3.1-8B

NLRL

0.05

0.1

RAGEN (Best Practice) RAGEN (LLaMA3.1-8B)

0.00

0.0

0.0 0.5 1.0 1.5 2.0

0 2 4 6 8

Steps 1e4

Steps 1e4

- Figure 6: FrozenLake experiment results. (a) Performance comparison with baselines on FrozenLake with deterministic transition dynamics. (b) Performance comparison with baselines on FrozenLake with stochastic transition dynamics (slippery). These results demonstrate that our proposed pipeline can more efficiently and stably improve compared to direct reinforcement learning.

buffer size 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟 can be found in Appendix C.3.9, where we demonstrate that larger 𝑚 values enhance exploration while maintaining stable learning and 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟 = 3 helps maintain consistent performance by preserving past experiences.

###### 4.3.2 Experimental Results on FrozenLake

FrozenLake environment is a grid-based navigation task (e.g., 4×4 grids) where a player must reach a goal while avoiding holes. Following RAGEN’s original implementation [48], we conduct similar experiments under both deterministic and stochastic transition dynamics. To test NLRL’s robustness, we directly transfer our system and hyperparameters, previously tested on Tic-Tac-Toe, to FrozenLake with minimal changes. Surprisingly, even without any tuning, NLRL in in Figure 6 demonstrates better performance on both scenarios, whereas well-tuned RAGEN suffers from inefficiency and occasional performance degradation, a phenomenon mentioned in its original paper. Notably, RAGEN’s "best practice" has slow convergence even in deterministic settings, suggesting that its inefficiency may stem not from environmental stochasticity but from its passive learning paradigm, as discussed in Section 3.1. Such differences can be further visualized through the progression of reasoning trajectory (Table 2). Refer to Appendix C.4 for more experimental details and result visualizations.

###### 5 Related work

Language Model Based Autonomous Agent. We are inspired by the strong emergent capabilities of LLMs in complex reasoning and planning scenarios [6, 2, 33, 16, 54]. The field of language model-based autonomous agents [15, 10, 57] has witnessed a growing trend of leveraging LLMs for high-level planning purposes. Reflexion [39] is built upon ReAct [54] with self-reflection, to generate tips given online feedback. [59] proposes to distill such reflection ability from the large model to train a small policy feedback model that can guide the policy. Their underlying ideas share similarities with language MC, while NLRL focuses on the training framework–we extract core information by sampling multiple trajectories and leverage them to train our language value function.

Learning from Language Feedback. Our work is also related to research on learning from language feedback. While [8] focuses on benchmarking algorithms, we aim to propose a new algorithmic framework. Studies such as [53, 56, 9] introduce an LLM-based optimization paradigm that leverages natural language to represent optimization operators like gradients and backpropagation, achieving end-to-end generative optimization. NLRL, on the other hand, represents a parallel approach, offering a generative framework specifically designed for RL optimization problems.

LLMs as evaluation function. Our language value function aligns with recent efforts in NLP that leverage LLMs as generative evaluators or verifiers, as seen in [47, 29, 23, 17, 57, 31], or that utilize LLMs’ self-evaluation capabilities to enhance task-solving performance [35, 55, 19]. These approaches often rely on the LLM’s self-assessment (which can be unreliable), or distill on stable supervision signals like human annotations or guidance from stronger LLMs. In contrast, NLRL’s training of a language value function offers a novel unsupervised approach.

Interpretable Reinforcement Learning. Our work is also related to interpretable RL, such as methods of concept-based explanations for sequential decision-making tasks. [41, 20, 12] formulates concept-based explanations upon state preconditions and action costs.

###### 6 Conclusion

In this work, we propose a new paradigm NLRL, which reformulates the RL process with natural language representation. We believe NLRL opens a new door for understanding and implementing RL algorithms in natural language space, improving agent’s effectiveness, efficiency and enabling active learning. NLRL also serves as a potential way for generating high-quality language synthetic data for policy and critic, which could be crucial in training more advanced language models and agents. We leave further discussion and limitation in Appendix F.

###### References

- [1] M. Abdulhai, I. White, C. Snell, C. Sun, J. Hong, Y. Zhai, K. Xu, and S. Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models. arXiv preprint arXiv:2311.18232, 2023.
- [2] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [3] A. Barreto, W. Dabney, R. Munos, J. J. Hunt, T. Schaul, H. P. van Hasselt, and D. Silver. Successor features for transfer in reinforcement learning. Advances in neural information processing systems, 30, 2017.
- [4] A. G. Barto, R. S. Sutton, and C. W. Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics, (5): 834–846, 1983.
- [5] R. Bellman, R. E. Kalaba, et al. Dynamic programming and modern control theory, volume 81. Citeseer, 1965.
- [6] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [7] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [8] C.-A. Cheng, A. Kolobov, D. Misra, A. Nie, and A. Swaminathan. Llf-bench: Benchmark for interactive learning from language feedback. arXiv preprint arXiv:2312.06853, 2023.
- [9] C.-A. Cheng, A. Nie, and A. Swaminathan. Trace is the next autodiff: Generative optimization with rich feedback, execution traces, and llms. arXiv preprint arXiv:2406.16218, 2024.
- [10] F. Christianos, G. Papoudakis, M. Zimmer, T. Coste, Z. Wu, J. Chen, K. Khandelwal, J. Doran, X. Feng, J. Liu, et al. Pangu-agent: A fine-tunable generalist agent with structured reasoning. arXiv preprint arXiv:2312.14878, 2023.
- [11] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [12] D. Das, S. Chernova, and B. Kim. State2explanation: Concept-based explanations to benefit agent learning and user understanding. arXiv preprint arXiv:2309.12482, 2023.
- [13] P. Dayan. Improving generalization for temporal difference learning: The successor representation. Neural computation, 5(4):613–624, 1993.
- [14] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [15] X. Feng, Y. Luo, Z. Wang, H. Tang, M. Yang, K. Shao, D. Mguni, Y. Du, and J. Wang. Chessgpt: Bridging policy learning and language modeling. arXiv preprint arXiv:2306.09200, 2023.
- [16] X. Feng, Z. Wan, M. Wen, Y. Wen, W. Zhang, and J. Wang. Alphazero-like tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179, 2023.
- [17] M. Gao, X. Hu, J. Ruan, X. Pu, and X. Wan. Llm-based nlg evaluation: Current status and challenges. arXiv preprint arXiv:2402.01383, 2024.
- [18] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [19] S. Hao, Y. Gu, H. Ma, J. J. Hong, Z. Wang, D. Z. Wang, and Z. Hu. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.
- [20] B. Hayes and J. A. Shah. Improving robot controller transparency through autonomous policy explanation. In Proceedings of the 2017 ACM/IEEE international conference on human-robot interaction, pages 303–312, 2017.
- [21] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [22] Y. Ji, Y. Wang, and J. Kato. Spatial-temporal concept based explanation of 3d convnets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15444–15453, 2023.
- [23] D. Jiang, Y. Li, G. Zhang, W. Huang, B. Y. Lin, and W. Chen. Tigerscore: Towards building explainable metric for all text generation tasks. arXiv preprint arXiv:2310.00752, 2023.
- [24] B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang, H. Zamani, and J. Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [25] K. J. Ju. gym-tictactoe: A tic tac toe environment for openai gym, 2024. URL https: //github.com/haje01/gym-tictactoe. GitHub repository.
- [26] L. Kocsis and C. Szepesvári. Bandit based monte-carlo planning. In European conference on machine learning, pages 282–293. Springer, 2006.
- [27] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.
- [28] M. Lanctot, E. Lockhart, J.-B. Lespiau, V. Zambaldi, S. Upadhyay, J. Pérolat, S. Srinivasan, F. Timbers, K. Tuyls, S. Omidshafiei, D. Hennes, D. Morrill, P. Muller, T. Ewalds, R. Faulkner, J. Kramár, B. D. Vylder, B. Saeta, J. Bradbury, D. Ding, S. Borgeaud, M. Lai, J. Schrittwieser, T. Anthony, E. Hughes, I. Danihelka, and J. Ryan-Davis. OpenSpiel: A framework for reinforcement learning in games. CoRR, abs/1908.09453, 2019. URL http://arxiv.org/abs/1908.09453.
- [29] J. Li, S. Sun, W. Yuan, R.-Z. Fan, H. Zhao, and P. Liu. Generative judge for evaluating alignment. arXiv preprint arXiv:2310.05470, 2023.
- [30] Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [31] D. Mahan, D. Van Phung, R. Rafailov, C. Blagden, N. Lile, L. Castricato, J.-P. Fränken, C. Finn, and A. Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024.
- [32] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.
- [33] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.
- [34] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
- [35] P. Putta, E. Mills, N. Garg, S. Motwani, C. Finn, D. Garg, and R. Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199, 2024.
- [36] A. Saffidine, N. Jouandeau, and T. Cazenave. Solving breakthrough with race patterns and job-level proof number search. In Advances in Computer Games: 13th International Conference, ACG 2011, Tilburg, The Netherlands, November 20-22, 2011, Revised Selected Papers 13, pages 196–207. Springer, 2012.

- [37] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [38] D. J. Shah, P. Rushton, S. Singla, M. Parmar, K. Smith, Y. Vanjani, A. Vaswani, A. Chaluvaraju, A. Hojel, A. Ma, et al. Rethinking reflection in pre-training. arXiv preprint arXiv:2504.04022, 2025.
- [39] N. Shinn, B. Labash, and A. Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.
- [40] D. Silver and R. S. Sutton. Welcome to the era of experience. Google AI, 2025.
- [41] S. Sreedharan, U. Soni, M. Verma, S. Srivastava, and S. Kambhampati. Bridging the gap: Providing post-hoc symbolic explanations for sequential decision-making problems with inscrutable representations. arXiv preprint arXiv:2002.01080, 2020.
- [42] R. S. Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3: 9–44, 1988.
- [43] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.
- [44] R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [45] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [46] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.
- [47] Y. Wang, Z. Yu, Z. Zeng, L. Yang, C. Wang, H. Chen, C. Jiang, R. Xie, J. Wang, X. Xie, et al. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. arXiv preprint arXiv:2306.05087, 2023.
- [48] Z. Wang, K. Wang, Q. Wang, P. Zhang, L. Li, Z. Yang, K. Yu, M. N. Nguyen, L. Liu, E. Gottlieb, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025.
- [49] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.
- [50] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.
- [51] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.
- [52] D. Xu, W. Chen, W. Peng, C. Zhang, T. Xu, X. Zhao, X. Wu, Y. Zheng, and E. Chen. Large language models for generative information extraction: A survey. arXiv preprint arXiv:2312.17617, 2023.
- [53] C. Yang, X. Wang, Y. Lu, H. Liu, Q. V. Le, D. Zhou, and X. Chen. Large language models as optimizers. arXiv preprint arXiv:2309.03409, 2023.
- [54] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.
- [55] S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

- [56] M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, Z. Huang, C. Guestrin, and J. Zou. Textgrad: Automatic" differentiation" via text. arXiv preprint arXiv:2406.07496, 2024.
- [57] L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, and R. Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024.
- [58] T. Zhang, F. Ladhak, E. Durmus, P. Liang, K. McKeown, and T. B. Hashimoto. Benchmarking large language models for news summarization. arXiv preprint arXiv:2301.13848, 2023.
- [59] V. Zhong, D. Misra, X. Yuan, and M.-A. Côté. Policy improvement using language feedback models. arXiv preprint arXiv:2402.07876, 2024.

###### Contents

- A Detailed Related work 17
- B Algorithms 17

- B.1 Language GPI by prompting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 Train natural language language value function with language TD . . . . . . . . . 17
- B.3 Natural language actor critic . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- C Experimental details 19

- C.1 Maze Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- C.1.1 Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.1.2 Language GPI Visualization . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.1.3 The effect of Language TD on the Value function . . . . . . . . . . . . . . 22

- C.2 Breakthrough Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- C.2.1 Experiment Setup and Methods . . . . . . . . . . . . . . . . . . . . . . . 22
- C.2.2 More Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- C.2.3 Language TD Visualization . . . . . . . . . . . . . . . . . . . . . . . . . 24
- C.2.4 Comparison of language value function . . . . . . . . . . . . . . . . . . . 28

- C.3 Tic-tac-toe Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- C.3.1 Experiment Setup and Methods . . . . . . . . . . . . . . . . . . . . . . . 29
- C.3.2 More details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.3.3 Stabilizing language value function . . . . . . . . . . . . . . . . . . . . . 30
- C.3.4 Training Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- C.3.5 Algorithm Implementation Details . . . . . . . . . . . . . . . . . . . . . . 31
- C.3.6 Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- C.3.7 Baseline Implementation Details . . . . . . . . . . . . . . . . . . . . . . . 41
- C.3.8 Additional Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- C.3.9 Additional Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . 42

- C.4 FrozenLake Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

- C.4.1 Comparing Detailed Reasoning . . . . . . . . . . . . . . . . . . . . . . . 43
- C.4.2 Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

- D Prompts 53

- D.1 Maze experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
- D.2 Breakthrough Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
- D.3 Tic-tac-toe Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56

- E Analogy 59

- E.1 Language MC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- E.2 Language TD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59

###### F Limitation and future work 59

###### A Detailed Related work

Language Model Based Autonomous Agent. Inspired by the strong emergent capabilities of LLMs in complex reasoning and planning scenarios [6, 49, 2, 33, 16]. The field of language model-based autonomous agents [15, 10, 57] has witnessed a growing trend of leveraging LLMs for high-level planning purposes. For text agents, ReAct[54] leverages chain-of-thought planning about the intermediate goals and task-specific generation via few-shot prompting. Reflexion [39] is built upon ReAct [54] with self-reflection, named verbal reinforcement learning, to generate tips given online feedback. Such tips strengthen agent’s capability through in-context learning. [59] proposes to distill such reflection ability from the large model to train a small policy feedback model that can guide the policy. Their underlying ideas share similarities with NLRL’s language Monte Carlo estimate, while NLRL formulates this process more formally and incorporates it into training–we extract core information/concepts by sampling multiple trajectories and leverage them to train our natural language value function. Another line of work like [58, 52] leverages Large Language Models to summarize and extract information to help fuse intermediate changes. They are related to the "aggregation" idea of language value function training.

Interpretable Reinforcement Learning. One of the major purposes of interpretable RL is to automatically seek explanations from non-AI experts. For instance, methods of concept-based explanations for sequential decision-making tasks. [22] provide a concept-based explanation for 3D action recognition CovNets by clustering learned human interpretable features. [41] formulates concept-based explanations upon state preconditions and action costs, representing any factual statement a user associates with a given state. Similarly, [20] uses logical formulas to summarize policies. Additionally, [12] trains a joint embedding model for state-action pairs and concept-based explanations.

Learning from Language Feedback. Our work is also related to research on learning from language feedback. While [8] focuses on benchmarking algorithms, we aim to propose a new algorithmic framework. Studies such as [53, 56, 9] introduce an LLM-based optimization paradigm that leverages natural language to represent optimization operators like gradients and backpropagation, achieving end-to-end generative optimization. NLRL, on the other hand, represents a parallel approach, offering a generative framework specifically designed for RL optimization problems.

LLMs as evaluation function. Our language value function aligns with recent efforts in NLP that leverage LLMs as generative evaluators or verifiers, as seen in [47, 29, 23, 17, 57, 31], or that utilize LLMs’ self-evaluation capabilities to enhance task-solving performance [35, 55, 19]. These approaches often rely on the LLM’s self-assessment (which can be unreliable), or distill on stable supervision signals like human annotations or guidance from stronger LLMs. In contrast, NLRL’s training of a language value function offers a novel unsupervised approach: we can train a generative evaluator solely by aggregating language feedback from the environment, without distilling any strong model–experimental results in Sec. 4.2 show that our language value function’s performance is much better than any LLMs we utilize during the training.

Successor Features. Successor features [13, 3] aim to learn a high-dimensional value function representation, where each component captures the expected future occurrence of all state representations under a fixed policy. As shown in [3], successor features help decouple the environment dynamics from the rewards, facilitating transfer learning in RL. The language value function in NLRL is conceptually similar to successor features since it also represents state values in a high-dimensional space and satisfies the language Bellman equation.

###### B Algorithms In this section, we provide the full pseudo-code for three cases mentioned in Sec. 3.4, 3.5, and 3.6.

- B.1 Language GPI by prompting
- B.2 Train natural language language value function with language TD

- Algorithm 1 Language GPI by Prompting

Require: Initial language policy 𝜋, language value function 𝑉𝜋𝐿, LLM implementing operators 𝐺1,

𝐺2, policy improvement operator 𝐼, number of lookahead steps 𝑁, number of variations 𝐾

- 1: for each state 𝑠𝑡 do
- 2: // Language TD estimation
- 3: for 𝑖 = 1 to 𝐾 do
- 4: Sample action 𝑎𝑡(𝑖) (①)
- 5: Simulate 𝑁 steps ahead to get trajectory 𝜏(𝑖)
- 6: Prompt LLM to evaluate 𝑉𝜋𝐿(𝑠𝑡(+𝑖)𝑁) (②)
- 7: Generate intermediate description 𝑑(𝑖) = 𝑑(𝑎𝑡(𝑖),𝑟𝑡(𝑖), 𝑠𝑡(+𝑖)1)
- 8: Compute value estimate 𝑉(𝑖) = 𝐺2(𝑑(𝑖),𝑉𝜋𝐿(𝑠𝑡(+𝑖)𝑁)) (③)
- 9: end for
- 10: Aggregate value estimates 𝑉𝜋𝐿(𝑠𝑡) = 𝐺1({𝑉(𝑖)}𝑖𝐾=1) (③)
- 11: // Policy improvement
- 12: Use 𝑉𝜋𝐿(𝑠𝑡) and prompt LLM as operator 𝐼 to select the optimal action 𝑎𝑡 (⑤)
- 13: end for

- Algorithm 2 Train natural language language value function with language TD

Require: Pre-defined policy 𝜋, LLM-based language value model 𝑉𝜋𝐿, language descriptor 𝑑, language aggregator 𝐺1, 𝐺2, state buffer 𝐷𝑠, number of iterations 𝑇, number of trajectories 𝑁, number of MC trajectories 𝐾

- 1: // Collect few-step rollout data using fixed policy for TD data buffer
- 2: Initialize TD data buffer B ← {}
- 3: for state 𝑠𝑡 in state buffer 𝐷𝑠 do
- 4: Starting from 𝑠𝑡, run policy 𝜋 for a few 𝑙-step look-ahead rollouts and collect 𝐾 distinct variations: 𝑉𝑠𝑡 = {(𝑠𝑡+𝑖, 𝑎𝑡+𝑖,𝑟𝑡+𝑖, 𝑠𝑡+𝑖+1)𝑖𝑙=−01}𝐾
- 5: B ← B ∪ 𝑉𝑠𝑡
- 6: end for
- 7: // Language TD learning
- 8: for iteration 𝑡 = 1 to 𝑇 do
- 9: // Language TD estimate ②, ③
- 10: D𝑉 ← {} {Value training data}
- 11: for each look-ahead data (𝑠𝑡,𝑉𝑠𝑡) in B do
- 12: For each variation 𝑛, generate the final state evaluation 𝑉𝜋𝐿(𝑠𝑡𝑛+𝑙) with 𝑉𝜋𝐿, and utilize 𝑑 to describe intermediate transitions 𝑑𝑛 = 𝑑(𝑠𝑡, 𝑎𝑡𝑛,𝑟𝑡𝑛, 𝑠𝑡𝑛+1, ..., 𝑠𝑡𝑛+𝑙−1, 𝑎𝑡𝑛+𝑙−1,𝑟𝑡𝑛+𝑙−1)
- 13: 𝑉𝜋𝐿(𝑠𝑡)𝑛𝑒𝑤 ← 𝐺1 𝐺2 𝑑𝑛,𝑉𝜋𝐿(𝑠𝑡𝑛+1) 𝑛 𝐾=1 {l-step, k-variation Language TD}
- 14: D𝑉 ← D𝑉 ∪ 𝑉𝜋𝐿(𝑠𝑡)𝑛𝑒𝑤
- 15: end for
- 16: // Update value function ④
- 17: Train 𝑉𝜋𝐿 using language TD estimate dataset D𝑉 with supervised loss
- 18: end for

###### B.3 Natural language actor critic

- Algorithm 3 Natural Language Actor-Critic Learning

Require: Initial language policy model 𝜋𝐿, language value model 𝑉𝜋𝐿, large language model for

evaluation 𝐺1/𝐺1,𝐺2, improvement operator 𝐼

Require: Number of iterations 𝑇, number of trajectories 𝑁, number of MC trajectories 𝐾

- 1: Initialize replay buffer history B ← {}
- 2: for iteration 𝑡 = 1 to 𝑇 do
- 3: // Collect trajectories using language policy ①
- 4: 𝜏 ← {} {Initialize trajectory buffer}
- 5: for 𝑖 = 1 to 𝑁 do
- 6: Run policy 𝜋𝐿 to collect trajectory: 𝜏𝑖 ← {(𝑠𝑗, 𝑎𝑗,𝑟𝑗, 𝑠𝑗+1)𝐻𝑗=𝑖0}
- 7: 𝜏 ← 𝜏 ∪ {𝜏𝑖}
- 8: end for
- 9: // Language value estimation ②, ③
- 10: D𝑉 ← {} {Value training data}
- 11: for each state-action pair (𝑠, 𝑎) in 𝜏 do
- 12: if using MC estimate then
- 13: Sample 𝐾𝑀𝐶 trajectories starting from (𝑠, 𝑎): {𝜏𝑘}𝐾𝑘=𝑀𝐶1
- 14: 𝑣 ← 𝐺1({𝜏𝑘}𝐾𝑘=𝑀𝐶1 ) {Language MC estimate}
- 15: else if using TD estimate then
- 16: Sample 𝐾𝑇𝐷 transitions starting from (𝑠, 𝑎): {(𝑠𝑘, 𝑎𝑘,𝑟𝑘, 𝑠′𝑘)}𝐾𝑘=𝑇𝐷1
- 17: 𝑣 ← 𝐺1({𝐺2(𝑑(𝑠𝑘, 𝑎𝑘,𝑟𝑘, 𝑠′𝑘),𝑉𝜋𝐿(𝑠′𝑘))}𝐾𝑘=𝑇𝐷1 ) {Language TD estimate}
- 18: end if
- 19: D𝑉 ← D𝑉 ∪ {(𝑠, 𝑎, 𝑣)}
- 20: end for
- 21: // Update value function ④
- 22: B ← B ∪ {D𝑉} {Add to buffer history}
- 23: Train 𝑉𝜋𝐿 on merged data from last 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟 buffers in B with supervised loss
- 24: // Language policy improvement ⑤
- 25: D𝜋 ← {} {Policy training data}
- 26: for each state 𝑠 in 𝜏 do
- 27: Sample actions {𝑎𝑖}𝑖𝑁=𝑠𝑎𝑚𝑝𝑙𝑒1 from 𝜋𝐿(𝑠)
- 28: Select top-𝑚 actions A𝑘 based on sampling frequency
- 29: Query value estimates: 𝑄𝐿 ← {𝑉𝜋𝐿(𝑠, 𝑎)|𝑎 ∈ A𝑘}
- 30: 𝜋𝑛𝑒𝑤, 𝑐 ← 𝐼(𝑄𝐿) {Language improvement operator}
- 31: D𝜋 ← D𝜋 ∪ {(𝑠, 𝜋𝑛𝑒𝑤, 𝑐)}
- 32: end for
- 33: // Update policy ⑥
- 34: Train 𝜋𝐿 on D𝜋 with supervised loss
- 35: end for

###### C Experimental details

- C.1 Maze Experiment

- C.1.1 Experiment Details

Our Maze experiment is conducted using LMRL Gym [1]. We use the original codebase for the environment. We consider the fully-observable setting, where the agent’s observation (described by text) includes the agent’s current position in the maze, the agent’s action history, the walls’ position around the agent (if any), and the goal position. The action space is discrete, including moving up / down / right / left. We evaluate the performance on 30 different initial positions, each with 3 random seeds.

[Figure 2]

[Figure 3]

Figure 7: The “double T” maze and the “medium” maze. At the beginning of each test, the agent randomly respawns in white space.

We prompt gpt-4o-mini-2024-07-18 (temperature = 0.7) as the language aggregator 𝐺1, language state-action value aggregator 𝐺2, and language state value function 𝑉𝜋𝐿 respectively.

###### C.1.2 Language GPI Visualization

We present the language GPI’s prompt and response visualization for language state value function 𝑉𝜋𝐿 and language state-action value aggregator 𝐺2 in the LMRL Maze experiment.

###### Maze prompts

|Role<br><br>|Content|
|---|---|
|SYSTEM<br><br>|You are an expert maze solver. You only respond in json.|
|USER (𝑉𝜋𝐿)<br><br>|You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. The possible actions are "move up", "move down", "move left", "move right". You are a highly skilled evaluator in this game and is expected to function similar to state value function in reinforcement learning. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent. Your output evaluation should be a json array that includes the following concepts:<br><br>- "thoughts": Let’s think step by step. Generate your detailed thought process and evaluation of the agent’s position and the distance towards the goal.<br>- "final evaluation": Concisely judge how good the agent’s current position is compared to all the possible positions in the maze, in terms of reaching the goal. Let’s start a new game. Now, please give your evaluation of the current state given the move history of the agent: The goal is at position 8, 6. Your current position is at position 5, 7. There are walls above you, below you. move up The goal is at position 8, 6. Your current position is at position 5, 7. There are walls above you, below you. move down The goal is at position 8, 6. Your current position is at<br><br><br>position 5, 7. There are walls above you, below you. move right The goal is at position 8, 6. Your current position is at<br>position 5, 8. There are walls above you, below you. move right The goal is at position 8, 6. Your current position is at<br>position 5, 9. There are walls to your right, below you.<br>|
|ASSISTANT|’thoughts’: ["The goal is located at (8, 6) and the agent’s current position is (5, 9).", ’To evaluate the distance to the goal, I will calculate the Manhattan distance, which is the sum of the absolute differences of the coordinates: |8 - 5|<br><br>+ |6 - 9| = 3 + 3 = 6.’, "The agent’s current position (5, 9) has walls directly below (6, 9) and to the right (5, 10). This limits movement options.", ’The possible actions for the agent are to move left to (5, 8) or move up to (4, 9). Moving down or right is not possible due to walls.’, ’Moving left to (5, 8) would bring the agent back to a previous position, while moving up to (4, 9) opens potential new pathways.’, ’Overall, the agent is relatively far from the goal, with limited movement options due to surrounding walls.’],|

###### ’final evaluation’: "The agent’s current position (5, 9) is not ideal as it is 6 steps away from the goal and has limited movement options. Moving up to (4, 9) is the best immediate option to explore new paths."

|Role|Content|
|---|---|
|SYSTEM|You are an expert maze solver. You only respond in json.|
|USER (𝐺2)|You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. The possible actions are "move up", "move down", "move left", "move right". You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead of the current maze position after taking the given action. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent, the action that the agent is going to take, *along with several key variations of trajectory pieces after taking this action (and the corresponding natural language evaluations of the trajectory pieces)*. Your task is to understand these look-ahead information and summarize, derive non-trivial analysis and understanding the *the agent’s position after taking the action*. Your output evaluation should be a json array with the following *two* concepts: "thoughts": Let’s think step by step. Summarize the look-ahead information of the variations after taking action move down. "final evaluation": Now Concisely judge how good the chosen action is, in terms of reaching the goal. Now, please give your evaluation given action move down, the<br><br>*current environment state*: The goal is at position 8, 6. Your current position is at position 5, 7. There are walls above you, below you and the look-ahead information of different variations after taking action move down: AGGREGATION OF VALUE ESTIMATES FOR ALL VARIATIONS|
|ASSISTANT|’thoughts’: ["The goal is located at (8, 6), which is three rows down and one column to the left of the current position (5, 7).", "The agent attempts to move up from (5, 7) but cannot because there is a wall above.", "Each variation shows the agent getting stuck in a loop of ineffective moves after trying to move up, consistently returning to (5, 7) or moving to (5, 9), which is still not closer to the goal.", "The walls surrounding the agent restrict movement, making it impossible to progress toward the goal efficiently.", "In all variations, the agent finds itself either returning to previous positions or moving to areas blocked by walls.", "The agentś progression towards the goal becomes increasingly limited as it cannot move up, down, or directly toward the goal from its current position."], ’final evaluation’: "{ "current position": "(5, 7)", "goal position": "(8, 6)", "manhattan distance": 6, "possible moves": ["move down", "move left"], "blocked moves": ["move up", "move right"], "evaluation": "The chosen action m´ove upís ineffective as it does not change the agentś position, and the surrounding walls create a dead end, making it impossible to reach the goal efficiently."}"<br><br>|

###### C.1.3 The effect of Language TD on the Value function

Below we show a comparison of the Language state-action values with/without Language TD. Comparing the two Language State-action Value, we can see that by analyzing various potential outcomes, Language TD identifies both immediate and future benefits or drawbacks of actions, which is reflected in its more detailed thoughts and final evaluation. The inclusion of varied trajectory information allows Language TD to provide a more balanced and informed assessment, mitigating over-reliance on immediate next-step evaluations.

|Language State-action Value (Language TD)<br><br>For action "move down", {"thoughts": ["The agent’s current position is (5, 7) with the goal at (8, 6).", "After attempting to move down, the agent remains at (5, 7) due to walls above and below.", "In Variation 1, the agent eventually moves to (6, 6), which is one step away from the goal at (8, 6).", "Variation 2 shows that moving down leads to blocked paths, necessitating lateral movement, but ultimately keeps the agent away from the goal.", "Variation 3 allows the agent to successfully reach (7, 6), positioning it just one step away from the goal.", "Variation 4 also confirms that moving down to (6, 6) is effective, placing the agent two steps away from the goal.", "Overall, the evaluations indicate that while the agent faces wall constraints, moving down leads to favorable outcomes that progress toward the goal."], "final_evaluation": {"current_position": "(5, 7)", "distance_to_goal": 3, "proposed_action": "move down", "evaluation": "Although the move down is initially blocked, subsequent evaluations indicate that this action can lead to optimal positions in the following moves. The agent can reach the goal within two more steps if it navigates correctly."}}|
|---|

|Language State-action Value (without Language TD)<br><br>For action "move down", {<br><br>"thoughts": [ "The agent’s current position is (5, 7) and the goal is at (8, 6).", "The possible action is to move down to (6, 7).", "There are walls above (4, 7) and below (6, 7), which means the agent can<br><br>only move down if it is not blocked.", "Moving down to (6, 7) is valid since there is no wall at that position.", "After moving down, the new position will be (6, 7).", "The distance to the goal can be calculated as follows: the horizontal<br><br>distance is |8 - 6| = 2 and the vertical distance is |6 - 7| = 1.", "The total distance to the goal after moving down would be 2 + 1 = 3."<br><br>], "final_evaluation": { "current_position": "(6, 7)", "distance_to_goal": 3, "evaluation": "The agent’s position after moving down is better than the<br><br>previous one, as it moves closer to the goal. There are no walls blocking further movement towards the goal from this position."<br><br>} }|
|---|

- C.2 Breakthrough Experiment

- C.2.1 Experiment Setup and Methods

Text-Breakthrough. As mentioned in the text-based MDP, we textualize the game, including board, pieces, moves, positions, and information about capturing or termination.

Policy 𝜋 in 𝑉𝜋𝐿(𝑠). Our first task is to determine the 𝜋 in 𝑉𝜋𝐿(𝑠) since 𝑉𝜋𝐿(𝑠) measures the value for a specific policy, according to Equ 4. To train a reliable board annotator, we select a strong or near-optimal policy—analogous to the way we place greater trust in a grandmaster’s annotations. In our experiment, our 𝜋 is the Monte-Carlo tree search (MCTS) algorithm [26] from OpenSpiel [28], with a high number of simulations and rollouts to ensure an extensive search.

Building State Dataset 𝑠 ∼ 𝑃(𝑠). Since 𝑉𝜋𝐿(𝑠) is over state 𝑠, our second step is to build a state dataset 𝐷𝑠–the distribution 𝑃(𝑠) that our 𝑉𝜋𝐿 works on. To ensure that 𝑉𝜋𝐿 can assess positions across diverse levels of gameplay, we build a mixed-state dataset by collecting rollouts between different MCTs configurations. Then we can easily split 𝐷𝑠 to build training state set 𝐷𝑡𝑟𝑎𝑖𝑛𝑠 and test state set 𝐷𝑡𝑒𝑠𝑡𝑠 .

Building TD training dataset. The TD training dataset is built on 𝐷𝑡𝑟𝑎𝑖𝑛𝑠 by conducting look-ahead expansion with rollout policy 𝜋. For each state 𝑠𝑡, we take a few times 𝑙-step look-ahead rollout and

deduplicate to 𝐾 distinct variations: {(𝑠𝑡+𝑖, 𝑎𝑡+𝑖,𝑟𝑡+𝑖, 𝑠𝑡+𝑖+1)𝑖𝑙=−01}𝐾.

Models. For the language TD operator (𝐺1, 𝐺2), we prompt the large LLaMA-3.1-70B-Instruct [14] model since it preserves stronger aggregation and reasoning ability. The language value function

𝑉𝜋𝐿(𝑠) is trained from the small LLaMA-3.1-8B-Instruct model.

Evaluation. An automatic evaluation metric is challenging even for advanced LLMs like GPT-4o, given their limited domain knowledge. We adopt a coarser metric—judging which side has the advantage. For any given state, we estimate win rates using Monte Carlo simulations to the game outcome. The ground-truth label is assigned to the side with a win rate exceeding a predefined threshold.

###### C.2.2 More Details

Our breakthrough experiment is conducted with Openspiel [28] 5x5 breakthrough game and MCTS policy. We slightly modified Openspiel’s codebase so the 5x5 breakthrough game supports two lines of pieces for each side.

For the policy 𝜋, as mentioned in the paper, we choose an MCTS policy with 𝑢𝑐𝑡𝑐 = 1, 1000 simulations for search, and 100 rollouts for the random rollout evaluator. For the state dataset, we define the MCTS policies using a grid of configurations based on the number of simulations and rollouts. Specifically, we use the following values for these parameters:

- • Simulation numbers: 2, 10, 100, 1000
- • Rollout numbers: 1, 10, 100, 1000

[Figure 4]

Figure 8: 5x5 Breakthrough.

This results in 4 × 4 = 16 unique MCTS policies, each characterized by a specific combination of simulation and rollout numbers. To generate the mixed state dataset, we pair every possible combination of these 16 policies (including self-pairing), leading to 16 × 16 = 256 policy pairs.

For each pair of policies, we perform the same number of rollouts where both policies interact within the environment. The states encountered during these rollouts are recorded, ensuring that the dataset captures a diverse distribution of positions. By merging the data from all policy pairs, we create the final mixed state dataset 𝐷𝑠, which is designed to represent states arising from a wide range of gameplay strategies and skill levels.

We use vLLM [27] for any LLM inference used in the experiment, including language value function inference and aggregation inference.

We sample 3000 states from the state dataset for evaluation to serve as the 𝐷𝑡𝑒𝑠𝑡𝑠 . When computing prediction accuracy, we only count state evaluation which can pass our rule-based parser to extract

the advantageous side judgment. Here we present the hyperparameters used in our experiment: Rollout

###### Parameter Value

Parallel Environments 192 Lookahead Step 4 Lookahead Rollout Number 4 Deduplicate State True

Table 4: Rollout Parameters

Prompting

LLM Sampling Parameter Value

Temperature 1.0 Top K 50 Top P 0.95 Max Tokens 512

Table 5: LLM sampling parameters for prompting.

Training

Parameter Value

Max Sequence Length 1024 Warmup Ratio 0.03 Learning Rate 2e-5 Learning Rate Scheduler Constant Dtype bfloat16 Per Device Train Batch Size 4 Gradient Accumulation Step 8 Training Epoch 2 Number of GPUs 4 Distributed Framework FSDP

Table 6: Data Collection Parameters

Evaluation. We use the same LLM sampling parameters as the prompting process.

- C.2.3 Language TD Visualization Here we present an example of how language TD works for query and response.

###### Language State-action Value (Before)

|Role<br><br>|Content|
|---|---|
|SYSTEM<br><br>|Here is the rule for the Breakthrough board game: The game is played on an 5x5 board for 2 players (white and black), with each player starting with 10 pawns. white pawns are on the first two rows and black pawns are on the last two rows. Black moves first. In each turn, players can move one of their pieces one square forward, diagonally forward if the target square is empty. Or it can capture an opponent’s piece if that square is one step diagonally forward.|

| |The game ends when one player successfully break through the opponent lines – either move a piece to the opposite last row of the board or captures all of the opponent’s pieces.<br><br>For board representation, we use b for black pieces, w for white pieces, and . for empty squares. (1-5) and (a-e) are used to show the rows and columns index respectively.<br><br>You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead analysis of the current board position. I will provide you with current board representation, *along with several key variations starting from this position (and their corresponding natural language evaluations of the subsequent positions)*.<br><br>Your task is to aggregate and compare these look-ahead information, to summarize, derive non-trivial analysis about the *current board*. Your output should include the following concepts:<br><br>1. *Tactical Considerations*: Any immediate threats, potential tactics, or vulnerabilities in the position.<br>2. *Positional Evaluation*: Consideration of pawn structure, piece activity, control of key squares, and game safety.<br>3. *Suggested Moves*: One or two strong candidate moves for the side to move, along with a brief rationale for comparing different moves.<br>4. *Advantage*: Based on all previous rationale, determine if white or black takes advantage. Use <white> or <black> to represent the winning side. Your response should be informative and concise.<br>|
|---|---|
|USER|"*The board you need to evaluate:*<br><br>5bb.b. 4b..b. 3..bw. 2w.w.. 1wwwww abcde<br><br>It is White’s turn. White pieces are at: d3, a2, c2, a1, b1, c1, d1, e1. Black pieces are at: a5, b5, d5, a4, d4, c3.<br><br>Here are the look-ahead variations from the current board position:<br><br>*Key Variations and Subsequent Evaluation:*:<br><br>*Variation 1:* Description of variation’s move sequence: The action sequence is: d3e4,d5e4*,a2b3,a4b3*.<br><br><br>Move 1:White moves piece from d3 (Column d, Row 3) to e4 (Column e, Row 4).<br><br>Move 2:Black moves piece from d5 (Column d, Row 5) to e4 (Column e, Row 4), capturing White piece.<br><br>Move 3:White moves piece from a2 (Column a, Row 2) to b3 (Column b, Row 3).<br><br>Move 4:Black moves piece from a4 (Column a, Row 4) to b3 (Column b, Row 3), capturing White piece. Subsequent position evaluation:<br><br><br>|

The subsequent board is:

5bb... 4...bb 3.bb.. 2..w.. 1wwwww abcde

It is White’s turn. White pieces are at: c2, a1, b1, c1, d1, e1. Black pieces are at: a5, b5, d4, e4, b3, c3.

The evaluation of this subsequent board is:

**Current Board Analysis:**

**Tactical Considerations:** White’s most significant tactical consideration is the potential to capture Black’s pieces on d4 and e4, which would gain a significant advantage. White’s pieces are well-positioned to launch a decisive attack. Black’s pieces on a5 and b5 are somewhat isolated and vulnerable to potential attacks.

**Positional Evaluation:** The current position slightly favors White due to their piece activity, control of key squares, and potential to create a strong pawn center. White’s pawns on a1 and d1 provide potential support for the central pawns. Black’s pawns are somewhat isolated, but they still have a chance to reorganize.

**Suggested Moves:** Based on the look-ahead analysis, two strong candidate moves for White are:

- 1. **e1-d2**: This move captures Black’s potential piece on d2 (in case Black plays d4-d2) and creates a strong initiative.

- 2. **c2-b3**: This move attacks Black’s position and prepares to break through to the opposite side, gaining a strong initiative.

**Advantage:** <white>

Overall, White has a slight advantage in this position, with multiple ways to break through Black’s lines and gain a significant advantage. However, Black’s position is not yet critical, and they can still reorganize their pieces to defend. White’s advantage is mainly due to their piece activity, control of key squares, and potential to create a strong pawn center.

*Variation 2:* Description of variation’s move sequence: The action sequence is: c1b2,c3d2,e1d2*,d4e3.

- Move 1:White moves piece from c1 (Column c, Row 1) to b2 (Column b, Row 2).

- Move 2:Black moves piece from c3 (Column c, Row 3) to d2 (Column d, Row 2).

- Move 3:White moves piece from e1 (Column e, Row 1) to d2 (Column d, Row 2), capturing Black piece.

| |Move 4:Black moves piece from d4 (Column d, Row 4) to e3 (Column e, Row 3).<br><br>Subsequent position evaluation: The subsequent board is:<br><br>5bb.b. 4b.... 3...wb 2wwww. 1ww.w. abcde<br><br>It is White’s turn. White pieces are at: d3, a2, b2, c2, d2, a1, b1, d1. Black pieces are at: a5, b5, d5, a4, e3.<br><br>The evaluation of this subsequent board is:<br><br>**Tactical Considerations:** The current position presents a strong opportunity for White to create a breakthrough on the ’d’ file, as demonstrated by the two variations. Black’s pawn structure is somewhat disjointed, making it difficult to respond to White’s advances on the ’d’ and ’e’ files.<br><br>**Positional Evaluation:** White’s pawn structure is more compact and cohesive, controlling key central squares. The pawns on ’a’ and ’b’ files provide additional support for the potential breakthrough on ’d’ and ’e’. Black’s pieces on ’a5’ and ’d5’ are somewhat isolated, making it difficult to coordinate a defense.<br><br>**Suggested Moves:** Considering the look-ahead variations, the strongest candidate move for White is **d3d4**. This move puts pressure on Black’s position and sets up the potential breakthrough on the ’d’ and ’e’ files. The rationale for this move is that it creates an immediate threat and forces Black to respond, potentially weakening their position.<br><br>Another possible move could be **d2d3**, but it is less aggressive and does not put as much pressure on Black.<br><br>**Advantage:** Based on the analysis, the advantage is **<white>**. The two variations demonstrate a clear path to victory for White, and Black’s position appears to be under significant pressure.<br><br>Please provide your analysis and understanding of the current board position based on the provided look-ahead information. Your response should be informative and concise.<br><br>|
|---|---|
|ASSISTANT|"**Current Board Analysis:** **Tactical Considerations:** The primary tactical consideration for White is to utilize the pawns on the ’d’ file to create a breakthrough, potentially capturing Black’s pieces on d4 and e4. Additionally, White should focus on controlling the key squares and putting pressure on Black’s somewhat isolated pieces on a5 and b5. Black needs to reorganize their pawns and pieces to defend against White’s central pawn advance.<br><br>|

**Positional Evaluation:** The position slightly favors White due to their more compact and cohesive pawn structure, piece activity, and control of key central squares. White’s pawns on the ’a’ and ’b’ files provide additional support for the potential breakthrough on the ’d’ and ’e’ files. Black’s pawns are somewhat isolated, making it challenging to coordinate a defense.

**Suggested Moves:** Based on the look-ahead analysis, two strong candidate moves for White are:

- 1. **d3d4**: This move puts pressure on Black’s position, sets up a potential breakthrough on the ’d’ and ’e’ files, and forces Black to respond.

- 2. **e1-d2**: This move captures Black’s potential piece on d2 and creates a strong initiative.

**Advantage:** <white>

Overall, White has a slight advantage in this position, with multiple ways to break through Black’s lines and gain a significant advantage. The two variations demonstrate a clear path to victory for White, and Black’s position appears to be under significant pressure. White’s advantage is mainly due to their piece activity, control of key squares, and potential to create a strong pawn center.

The provided look-ahead information suggests that White should prioritize controlling the central squares, putting pressure on Black’s position, and utilizing the pawns on the ’d’ file to create a breakthrough."

###### C.2.4 Comparison of language value function

Here we present another visualization comparison between the initial LLaMA-3.1-8B-Instruct Model and our trained model. The evaluation is conducted on the board 5..b.b 4.bb.. 3.w... 2.w... 1ww..w abcde, where white takes large advantage in this place. By our language TD training, the language value function can accurately evaluate this board and generate insightful annotation, largely decreasing the hallucination.

Language State Value (Before) Board: 5..b.b 4.bb.. 3.w... 2.w... 1ww..w abcde, white turn

Based on the current board position, here’s the evaluation for White’s turn:

- *Tactical Considerations:*

- - Black has an opportunity to attack White’s pawn at e1 with a pawn from e5 or c5, which might put White in a difficult position.
- - White’s pawn at b3 and b2 is a target for Black’s pawns at b4 and c4.
- - White needs to focus on safely advancing their pawns and creating a barrier for Black’s pieces.

*Positional Evaluation:*

- - White has a relatively weak pawn structure, with some pawns on the first row.
- - The center is somewhat controlled by White, but Black has a solid presence on the fifth row.

- - Black has a strategic advantage in pawn structure, but White has the opportunity to influence the game with proper planning.

- *Suggested Moves:*

- 1. Move the pawn from b2 to b3. This move has the goal of preparing to defend and possibly create a barrier against Black’s pieces.
- 2. Move the pawn from a1 to a2. This move aims to create more space and put pressure on Black’s pieces, which will make it harder for them to maneuver.

*Advantage:* Based on the current board position, <black> is favored.

|Language State Value (After training) Board: 5..b.b 4.bb.. 3.w... 2.w... 1ww..w abcde, white turn<br><br>**Analysis and Evaluation of the Current Board Position**<br><br>**Tactical Considerations:** The key variation shows an immediate threat of White breaking through Black’s lines. White’s piece on b3 is strategically positioned to capture Black’s piece on c4, which creates an opportunity for a decisive push.<br><br>**Positional Evaluation:** White has a significant advantage in terms of piece activity and control of key squares. With pieces on the third and second rows, White is poised to push forward and break through Black’s lines. Black’s pieces are concentrated on the fifth and fourth rows, which limits their mobility and ability to respond to White’s attack.<br><br>**Suggested Moves:** Two strong candidate moves for White are:<br><br>1. b3-c4*: This move captures Black’s piece and creates a weakness in Black’s position, allowing White to potentially break through.<br>2. b2-c3: This move develops another White piece and prepares to support the potential breakthrough on the c-file.<br><br><br>However, given the look-ahead variations, the most promising move is b3-c4*, as it leads to a terminal state where White wins.<br><br>**Advantage:** Based on the analysis, White has a decisive advantage. The ability to capture Black’s piece on c4 and break through Black’s lines gives White a strong initiative and a clear path to victory. Therefore, <white> is the winning side.|
|---|

- C.3 Tic-tac-toe Experiment

- C.3.1 Experiment Setup and Methods

Text-Tic-tac-toe. Similar to text-breakthrough, we textualize all information in the Tic-tac-toe.

Model Architecture. Our implementation uses three language models: one LLaMA-3.1-70B-Instruct and two LLaMA-3.1-8B-Instruct models. The 70B model is for language aggregator (𝐺1) and policy improvement operator 𝐼. The two 8B models implement our trainable components: language policy 𝜋𝐿 that generates actions through chain-of-thought reasoning, and language value function 𝑄𝐿𝜋 that evaluates state-action pairs.

Evaluation. To thoroughly evaluate our approach, we test against two different types of opponents. The results against a deterministic opponent that always takes the first available action can be found in Appendix C.3.8. For our main experiments, we focus on a stochastic opponent that plays randomly,

presenting a more challenging case for policy evaluation. We measure performance through metrics including win/loss/tie rates.

Baselines. We compare our algorithm with prompting-based method using GPT-4o-08-06, LLaMA-

- 3.1-8B/70B-Instruct with step-by-step reasoning. We also include a traditional RL baseline by fine-tuning LLaMA-3.1-8B-Instruct with standard PPO [37].

###### C.3.2 More details

Our experiments are conducted on the Tic-tac-toe environment [25], a standard two-player game environment that provides a clear testbed for evaluating sequential decision-making. The implementation uses several key components: https://github.com/haje01/gym-tictactoe (MIT License) for the Tic-tac-toe environment, FSDP (Fully Sharded Data Parallel) for distributed model training, vLLM [27] for efficient language model inference, and a custom parallel framework for trajectory collection. The environment is wrapped with a parallel execution layer to enable efficient batch processing of multiple games simultaneously.

For policy evaluation, we employ Monte Carlo sampling to estimate state-action values, requiring complete game trajectories until terminal states. Policy improvement utilizes a structured sampling approach with temperature-controlled exploration and frequency-based action selection.

To manage computational resources efficiently, we employ gradient checkpointing and use the FSDP strategy across our GPU cluster. This configuration allows us to train larger language models while maintaining reasonable memory requirements and training speed.

The parallel data collection framework is designed to maximize GPU utilization during training. It maintains a queue of active environments and processes state-action transitions in batches, significantly reducing the overall training time compared to sequential execution.

Model Architecture. Our implementation uses three language models: one LLaMA-3.1-70B-Instruct and two LLaMA-3.1-8B-Instruct models. The 70B model implements the language aggregator (𝐺1) and policy improvement operator 𝐼 - notably, we restrict this model to use only its general language processing and information aggregation capabilities, rather than any game-specific knowledge it may have learned during pre-training. This restriction is enforced through carefully designed prompts (see Appendix D.3 for details). The two 8B models implement our trainable components: one serves as the language policy 𝜋𝐿, generating actions through chain-of-thought reasoning, while the other implements the language value function 𝑄𝐿𝜋, evaluating state-action pairs.

Dataset Construction. For each iteration, we collect 512 complete game trajectories with 𝜋𝐿. For Monte Carlo value estimation, we collect 𝐾𝑀𝐶 = 5 complete trajectories for each state-action pair. For policy improvement, 𝜋𝐿 proposes 𝑁𝑠𝑎𝑚𝑝𝑙𝑒 = 10 candidate actions per state and we select the top-𝑚 (𝑚 = 10) actions from the candidate action list.

Evaluation. To thoroughly evaluate our approach, we test against two different types of opponents. The first is a deterministic opponent that always takes the first available action, while the second is a stochastic opponent that plays randomly, presenting a more challenging case for policy evaluation. We measure performance through metrics including win/loss/tie rates.

###### C.3.3 Stabilizing language value function

Despite the soundness of Sec 3.6, we identify training instability resulting from the language value function, mainly from two perspectives. First, we observed that the language value function 𝑉𝜋𝐿 can easily hallucinate when evaluating unseen state-action pairs, leading to incorrect language policy improvement. To address this, we add an action selection mask to restrict language policy improvement only to moves that are likely to be chosen by the language policy 𝜋𝐿 during rollout. Instead of prompting LLM to select from all available moves, we sample the language policy 𝑁𝑠𝑎𝑚𝑝𝑙𝑒 times and build an action candidate list. The language policy improvement is conducted from the list’s top-𝑚 moves. Such selection mask can help constrain policy updates within a limited range, akin to how KL divergence is used in RLHF objective function [34].

Second, we encountered the challenge of catastrophic forgetting for value training. Specifically, LLM can rapidly overfit to the state-action pairs encountered in the current iteration while forgetting past experiences, e.g. state-action pairs encountered only a few iterations ago. To address this, we enlarge

the language value function training set by merging the last 𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟 value function buffers, which help stabilize the training of the language value function by preserving past knowledge.

###### C.3.4 Training Infrastructure

Data Collection Pipeline For each training iteration, we collect data using the following configuration:

Parameter Value Parallel Environments 64 Trajectories per Iteration 512 Monte Carlo Samples (𝐾𝑀𝐶) 5

Policy Samples per State (𝑁𝑠𝑎𝑚𝑝𝑙𝑒) 10 Top-k Actions 10

Table 8: Data Collection Parameters

Model Architecture and Training

Both policy and value networks are trained from LLaMA-3.1-8B-Instruct using the following parameters:

Hyperparameter Value

Learning Rate 1e-5 Learning Rate Schedule Constant Training Epochs per Iteration 2 FSDP Configuration Full Sharding Gradient Checkpointing Enabled Batch Size 8 Max Sequence Length 1024 Training Hardware 4 × H100 GPUs Table 9: Model Training Hyperparameters

Value Function Buffer Management To prevent catastrophic forgetting, we maintain experience buffers with the following configuration:

Parameter Value

Buffer History Length (𝐾𝑏𝑢 𝑓 𝑓 𝑒𝑟) 3 iterations Merging Strategy Equal sampling Buffer Content State-action pairs with MC estimates Table 10: Buffer Management Configuration

###### C.3.5 Algorithm Implementation Details

Policy Evaluation Details For Monte Carlo evaluation, we use the following configuration:

###### Parameter Value

MC Trajectories per State-Action 5 Trajectory Completion Full game Value Aggregation Average over returns Sampling Temperature 0.7 Action Space Size 9 positions (0-8)

Table 11: Policy Evaluation Configuration

###### C.3.6 Visualization

Below we show a comparison of the Language state-action values before and after Language MC estimation. The evaluation is conducted on the following board:

|O|X<br><br>|3|
|---|---|---|
|4|O<br><br>|O|
|7<br><br>|X<br><br>|9|

Through analyzing multiple complete game trajectories, Language MC helps the value function develop a comprehensive understanding of long-term consequences of actions, rather than just immediate rewards. This is particularly evident in the enhanced evaluation quality after MC estimation, where the model provides more strategic insights and nuanced assessment of game positions.

###### Language State-action Value (Before)

|Role<br><br>|Content|
|---|---|
|SYSTEM|You are a player of the game of Tic Tac Toe. The game goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The board consists of "O", "X" and grid numbers. The grid number indicates an empty grid. You are learning how to evaluate a (board, action) pair in the Tic Tac Toe game by playing the game given the (board, action) pair and reflecting on the playing history. The playing history depicts a game in progress with a final result. Your answer consists of two elements:<br><br>- "thought": Let’s think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. You should only rely on the playing history as context and shouldn’t evaluate the game with your own judgment. Your evaluation should contain the following elements: Win probability, Threat, and Potential strategies.<br><br>- "final_evaluation": After all of your thoughts, judge which side takes advantage. 1 means you think ’O’ takes advantage,<br>-1 means you think ’X’ takes advantage, and 0 means you think the game will be a draw. You should output your answer in the JSON format. EXAMPLE: user: The (board, action) to evaluate is O’s turn: Board: O | O | X<br><br><br>––––4 | X | 6<br><br>––––7 | 8 | 9 Action: The O’s move is 4.<br><br>|

| |Below is the rollout sequence after this (board, action): After O’s move 4, the board position is: O | O | X<br><br>––––O | X | 6<br><br>––––7 | 8 | 9 After X’s move 7, the board position is: O | O | X<br><br>––––O | X | 6<br><br>––––X | 8 | 9 The game is over. X wins. X wins by occupying the positions 3, 5, 7. assistant: { "thought": { "Reflection": "It appears that the initial board position O | O | X<br><br>––––4 | X | 6<br><br>––––7 | 8 | 9 and action 4 were not favorable for O, as X was able to block on O’s moves and ultimately win the game.", "Win probability": "The win probability for X is large, while the win probability for O is low.", "Threat": "X has played center 5 and corner 3. X can win by playing corner 7. O was able to occupy 1, 4, 7 and create a potential winning opportunity.", "Potential strategies": "Potential strategies for O include playing the corner 7 to block X as opposite corner and win by occupying 1, 4, 7. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and there is 1 step to complete the plan." }, "final_evaluation": -0.9 }<br><br>|
|---|---|
|USER|The board to evaluate is O’s turn: Board: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 Action: The O’s move is 3. Below is the rollout sequence 1 after this (board, action): After O taking action 3, the board position is: O | X | O<br><br>––––4 | O | 6<br><br>––––7 | X | 9. After X taking action 4, the board position is: O | X | O<br><br>––––X | O | 6<br><br>––––7 | X | 9. After O taking action 7, the board position is: O | X | O<br><br>––––-|

| |X | O | 6<br><br>––––O | X | 9. The game is over. O wins. O wins by occupying the positions [3, 5, 7]. Now generate your evaluation for the (board, action) pair.<br><br>|
|---|---|
|ASSISTANT<br><br>|{ "thought": { "Reflection": "The initial board position O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 and action 3 were favorable for O, as O was able to occupy a strategic position and ultimately win the game.", "Win probability": "The win probability for O is high, while the win probability for X is low.", "Threat": "O has played center 5 and corner 3. O can win by playing corner 7. X has played center 5’s opposite corner 9 but failed to block O’s winning line.", "Potential strategies": "Potential strategies for O include playing the corner 7 to win by occupying 3, 5, 7. X could have occupied 4, 6, 9 to block O’s winning line but failed to do so." }, "final_evaluation": 0.9 }|

###### Language State-action Value (After)

|Role<br><br>|Content|
|---|---|
|SYSTEM<br><br>|You are a player of the game of Tic Tac Toe. The game goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The board consists of "O", "X" and grid numbers. The grid number indicates an empty grid. You are learning how to evaluate a (board, action) pair in the Tic Tac Toe game by playing the game given the (board, action) pair and reflecting on the playing history. The playing history depicts a game in progress with a final result. Your answer consists of two elements:<br><br>- "thought": let’s think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. You should only rely on the playing history as context and don’t evaluate the game with your own judgment. Your evaluation should contain the following elements: Win probability, Threat, and Potential strategies.<br>- "final_evaluation": After all of your thoughts, judge which side takes advantage. 1 means you think ’O’ takes advantage,<br>-1 means you think ’X’ takes advantage, and 0 means you think the game will be a draw. You should output your answer in the JSON format. EXAMPLE: user: The (board, action) to evaluate is O’s turn: Board: O | O | X<br><br><br>––––4 | X | 6|

| |––––7 | 8 | 9 Action: The O’s move is 4. Below is the rollout sequence after this (board, action): After O’s move 4, the board position is: O | O | X<br><br>––––O | X | 6<br><br>––––7 | 8 | 9 After X’s move 7, the board position is: O | O | X<br><br>––––O | X | 6<br><br>––––X | 8 | 9 The game is over. X wins. X wins by occupying the positions 3, 5, 7. assistant: { "thought": { "Reflection": "It appears that the initial board position O | O | X<br><br>––––4 | X | 6<br><br>––––7 | 8 | 9 and action 4 were not favorable for O, as X was able to block on O’s moves and ultimately win the game.", "Win probability": "The win probability for X is large, while the win probability for O is low.", "Threat": "X has played center 5 and corner 3. X can win by playing corner 7. O was able to occupy 1, 4, 7 and create a potential winning opportunity.", "Potential strategies": "Potential strategies for O include playing the corner 7 to block X as opposite corner and win by occupying 1, 4, 7. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and there is 1 step to complete the plan." }, "final_evaluation": -0.9 }|
|---|---|
|USER<br><br>|The board to evaluate is O’s turn: Board: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 Action: The O’s move is 9.<br><br>Below is the rollout sequence 1 after this (board, action): After O taking action 9, the board position is: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | O. The game is over. O wins. O wins by occupying the positions [1, 5, 9].<br><br>Below is the rollout sequence 2 after this (board, action): After O taking action 9, the board position is: O | X | 3<br>|

| |––––4 | O | 6<br><br>––––7 | X | O. The game is over. O wins. O wins by occupying the positions [1, 5, 9].<br><br>Below is the rollout sequence 3 after this (board, action): After O taking action 9, the board position is: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | O. The game is over. O wins. O wins by occupying the positions [1, 5, 9].<br><br>Below is the rollout sequence 4 after this (board, action): After O taking action 9, the board position is: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | O. The game is over. O wins. O wins by occupying the positions [1, 5, 9].<br><br>Below is the rollout sequence 5 after this (board, action): After O taking action 9, the board position is: O | X | 3<br><br><br>––––4 | O | 6<br><br>––––7 | X | O. The game is over. O wins. O wins by occupying the positions [1, 5, 9].<br><br>Now generate your evaluation for the (board, action) pair.<br><br>|
|---|---|
|ASSISTANT|{ "thought": { "Reflection": "It appears that the initial board position O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 and action 9 were highly favorable for O, as O was able to win the game immediately after taking action 9 in all rollout sequences.", "Win probability": "The win probability for O is extremely high, while the win probability for X is extremely low.", "Threat": "There were no threats from X in any of the rollout sequences, as O was able to win the game immediately after taking action 9.", "Potential strategies": "The potential strategy for O was to occupy the corner 9 to win the game by occupying the positions [1, 5, 9]. This strategy was successful in all rollout sequences." }, "final_evaluation": 1 }<br><br>|

###### Policy Improvement Details

Below we show a comparison of the Language Policy Improvement procedure at the beginning of training and at the end. The evaluation is conducted on the following board:

|O<br><br>|X|3|
|---|---|---|
|4|O<br><br>|6|
|7<br><br>|X|9|

The evolution in policy quality demonstrates how our framework effectively leverages value function assessments to enhance decision-making. The improved policy shows more sophisticated strategic reasoning, considering not just immediate moves but also their implications for future game states. This transformation is achieved while maintaining our restriction against using the model’s pre-trained game knowledge, instead relying purely on learned evaluation capabilities.

###### Language Policy Improvement (Beginning)

|Role<br><br>|Content|
|---|---|
|SYSTEM<br><br>|You are playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The board consists of "O", "X" and grid number. The grid number indicates an empty grid. Your task is to determine the best move for the next player based on the given board position and the next player. The evaluations of (board, action) pairs after possible moves are given. DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move. The evaluation for the next board is in the format of a JSON, consisting of two elements:<br><br>- "thought": Evaluation of the board and action pair.<br>- "final_evaluation": Judge which side takes advantage. 1 means ’O’ takes advantage, -1 means ’X’ takes advantage, and 0 means the game will be a draw.<br>|
|USER<br><br>|Here is the board position and the next player is O: Board: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9. The possible moves are [3, 7, 4, 9]. The following are the boards after each possible move: ### Evaluation for taking action 3: { "thought": { "Reflection": "The initial board position O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 and action 3 were favorable for O, as O was able to win the game in all rollout sequences.", "Win probability": "The win probability for O is high, while the win probability for X is low.", "Threat": "O has played corner 1 and center 5. O can win by playing corner 3, 4, 7 or 9. X has played corner 2 and 8 but failed to block O’s winning lines.",|

"Potential strategies": "Potential strategies for O include playing the corner 3 to create a potential winning opportunity. O could have occupied 1, 4, 7 or 1, 5, 9 or 3, 5, 7 to win the game. X could have blocked O’s winning lines by playing 4 or 7 but failed to do so." }, "final_evaluation": 0.9 } ### Evaluation for taking action 7: { "thought": { "Reflection": "The initial board position O | X | 3

––––4 | O | 6

––––7 | X | 9 and action 7 were favorable for O, as O was able to win the game in all rollout sequences.", "Win probability": "The win probability for O is high, while the win probability for X is low.", "Threat": "O has played corner 1 and center 5. O can win by playing corner 7 and then 9, or 4 and then 7. X has played corner 3 and 8 but failed to block O’s winning lines.", "Potential strategies": "Potential strategies for O include playing the corner 7 to create a potential winning opportunity and then playing 9 or 4 to win the game. X could have occupied

- 3, 6, 9 to block O’s winning lines but failed to do so." }, "final_evaluation": 0.9 } ### Evaluation for taking action 4: { "thought": { "Reflection": "The initial board position O | X | 3

––––-

- 4 | O | 6

––––7 | X | 9 and action 4 were favorable for O, as O was able to win the game in all rollout sequences.", "Win probability": "The win probability for O is high, while the win probability for X is low.", "Threat": "O has played center 5 and corner 1. O can win by playing corner 7 or 9. X has played center 6 and corner 3 but failed to block O’s winning lines.", "Potential strategies": "Potential strategies for O include playing the corner 7 or 9 to win the game. X could have occupied 3, 6, 9 to block O’s winning lines but failed to do so." }, "final_evaluation": 0.9 } ### Evaluation for taking action 9: { "thought": { "Reflection": "It appears that the initial board position O | X | 3

––––4 | O | 6

––––7 | X | 9

| |and action 9 were favorable for O, as O was able to win the game in both rollout sequences.", "Win probability": "The win probability for O is high, while the win probability for X is low.", "Threat": "O has played center 5 and corner 1. O can win by playing corner 9. X has played corner 3 and center 6 is empty, but X was not able to block O’s moves and ultimately lost the game.", "Potential strategies": "Potential strategies for O include playing the corner 9 to create a potential winning opportunity and then playing 3 or 7 to win the game. X could have occupied<br><br>3, 6, 7 to block O’s winning opportunity, but X was not able to do so." }, "final_evaluation": 0.9 } Now, please give your evaluation and the best move for O based on the given board position: O | X | 3<br><br>––––-<br><br>4 | O | 6<br><br><br>––––7 | X | 9. You should output your answer in the JSON format. Your answer consists of two elements:<br><br>- "thought": Let’s think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.<br><br>- "best_move": The best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions [3, 7, 4, 9]. Don’t output extra information except for the JSON format.<br><br><br>|
|---|---|
|ASSISTANT<br><br>|{ "thought": "Analyzing the given board position O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 and the evaluations of the possible moves, we can see that all four moves (3, 7, 4, 9) have the same final evaluation of 0.9, indicating that O has a high chance of winning with any of these moves. However, we should consider the potential strategies and threats for each move. Move 3 and 7 allow O to create a potential winning opportunity by playing the corner, while move 4 allows O to play the center and create a potential winning opportunity by playing the corner 7 or 9. Move 9 allows O to play the corner and create a potential winning opportunity by playing 3 or 7. Considering the potential strategies, all four moves seem to have similar advantages. However, move 7 seems to have a slight advantage as it allows O to create a potential winning opportunity and then play 9 to win the game, while also putting pressure on X to block the winning line. Therefore, the best move for O is 7.", "best_move": 7 }|

###### Language Policy Improvement (End)

|Role<br><br>|Content|
|---|---|

|SYSTEM<br><br>|You are playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The board consists of "O", "X" and grid numbers. The grid number indicates an empty grid. Your task is to determine the best move for the next player based on the given board position and the next player. The evaluations of (board, action) pairs after possible moves are given. DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move. The evaluation for the next board is in the format of a JSON, consisting of two elements:<br><br>- "thought": Evaluation of the board and action pair.<br>- "final_evaluation": Judge which side takes advantage. 1 means ’O’ takes advantage, -1 means ’X’ takes advantage, and 0 means the game will be a draw.<br>|
|---|---|
|USER<br><br>|Here is the board position and the next player is O: Board: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9. The possible moves are [9]. The following are the boards after each possible move: ### Evaluation for taking action 9: { "thought": { "Reflection": "It appears that the initial board position O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9 and action 9 were highly favorable for O, as O was able to win the game immediately after taking action 9 in all rollout sequences.", "Win probability": "The win probability for O is extremely high, while the win probability for X is extremely low.", "Threat": "There were no threats from X in any of the rollout sequences, as O was able to win the game immediately after taking action 9.", "Potential strategies": "The potential strategy for O was to occupy the corner 9 to win the game by occupying the positions [1, 5, 9]. This strategy was successful in all rollout sequences." }, "final_evaluation": 1 } Now, please give your evaluation and the best move for O based on the given board position: O | X | 3<br><br>––––4 | O | 6<br><br>––––7 | X | 9. You should output your answer in the JSON format. Your answer consists of two elements:|

| |- "thought": Let’s think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.<br>- "best_move": The best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions [9]. Don’t output extra information except for the JSON format.<br>|
|---|---|
|ASSISTANT<br><br>|{ "thought": "Based on the evaluation of the board position after taking action 9, it appears that this move is highly favorable for O. The win probability for O is extremely high, while the win probability for X is extremely low. There were no threats from X in any of the rollout sequences, as O was able to win the game immediately after taking action 9. The potential strategy for O was to occupy the corner 9 to win the game by occupying the positions [1, 5, 9]. This strategy was successful in all rollout sequences. Therefore, taking action 9 is the best move for O.", "best_move": 9 }|

###### C.3.7 Baseline Implementation Details

For our PPO baseline, we implement a custom PPO agent adapted for language models. The agent consists of two key components: an actor network based on LLaMA-3.1-8B-Instruct for action selection, and a critic network for value estimation.

Architecture Details The actor network uses LLaMA-3.1-8B-Instruct’s architecture with added special tokens for moves 1-9. The critic network consists of a two-layer MLP (Multi-Layer Perceptron) that takes the last hidden state from the actor as input:

- • Input dimension: LLaMA-3.1-8B hidden size
- • Hidden layer: ReLU activation
- • Output layer: Single value with Tanh activation

Training Configuration The PPO training uses the following hyperparameters:

###### Parameter Value

Learning Rate (Actor) 1e-5 Learning Rate (Critic) 1e-5 PPO Epochs 1 Batch Size 16 Gradient Accumulation Steps 2 Value Loss Coefficient 0.5 Entropy Coefficient 0.01 PPO Clip Range 0.2 Discount Factor (𝛾) 0.99 GAE Parameter (𝜆) 0.95

Table 16: PPO Training Hyperparameters

Implementation Details For each training iteration:

- • We collect 512 complete game trajectories using 8 parallel environments
- • Actions are selected using a temperature-controlled sampling (𝑇 = 0.7) over the logits
- • The policy proposes 𝑁𝑠𝑎𝑚𝑝𝑙𝑒 = 1 candidate actions per state and selects top-𝑘 (𝑘 = 1) based on sampling frequency

- • We use 1 A100 GPU for training
- • Gradient checkpointing is enabled to optimize memory usage

Both networks are trained using AdamW optimizer with weight decay of 0.01. We maintain experience buffers with a history length of 1 iterations for on-policy optimization.

###### C.3.8 Additional Experiments

(a) First-Move Opponent

1.0

+0.46

0.9

0.8

WinRate

0.7

0.6

Natural Language Actor Critic

0.5

Natural Language Actor Critic w/o Action Selection Mask

PPO

0.4

GPT-4o as Policy

LLaMA-3.1-70B-Instruct as Policy

0.3

LLaMA-3.1-8B-Instruct as Policy

0 2 4 6 8 10 Iteration

Figure 9: Training results against the deterministic first-move opponent.

###### C.3.9 Additional Ablation Studies

Action Selection Parameter Study To complement the main ablation studies presented in Section 4.3.1, we conduct additional experiments on the action selection parameter top-𝑚. Our results demonstrate that increasing 𝑚 from 2 to 10 enhances exploration capabilities while maintaining training stability. Specifically, larger 𝑚 value (10) show more stable training compared to smaller values (2-5).

(a) top-m

0.8

WinRate

0.7

0.6

0.5

top-m=2 top-m=5 top-m=10

0 5 10 15 20 25 30 Iteration

Figure 10: Ablation studies on action selection parameter top-𝑚.

###### C.4 FrozenLake Experiment

FrozenLake environment is a grid-based navigation task (e.g., 4×4 grids) where a player must reach a goal while avoiding holes. The board looks like this:

|_ G O _ _ _ _ O P O _ _ _ _ _ _<br><br>|
|---|

[Figure 5]

Since it has similar properties to the tic-tac-toe game, we borrow most of the implementation and hyperparameter settings introduced in Appendix C.3. In FrozenLake experiments, we specifically compare NLRL with RAGEN [48]. The implementation of RAGEN follows its original paper and open codebase: https://github.com/RAGEN-AI/RAGEN (MIT License) with minimal modification. The original RAGEN paper has only tried learning on Qwen2.5-0.5B-Instruction model, whereas we also extend it to other backbone language models to investigate the CoT degradation phenomenon.

###### C.4.1 Comparing Detailed Reasoning

We have tried RAGEN-like training on multiple model series including Qwen2.5-0.5B, Qwen2.5-7B and Llama3.1-8B. All of them have experienced a degradation in reasoning, as illustrated in Figure 17.

###### C.4.2 Visualization Language State-action Value (Before)

|Role<br><br>|Content|
|---|---|
|SYSTEM<br><br>|You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid. The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.<br><br>The meaning of each symbol in the state is: P: player, _: empty, O: hole, G: goal, X: player in hole, √ : player on goal<br><br>The possible actions are: 1:Left, 2:Down, 3:Right, 4:Up<br><br>You are learning how to evaluate a (board, action) pair in the FrozenLake by playing the game given the (board, action) pair and reflect the playing history. The playing history depicts a game in progress with a final result. Your answer consists of two elements:<br><br>- "thought": let’s think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. You should only rely on the playing history as context and don’t evaluate the game with your own judgment. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies.|

|Methods<br><br>|Map|Initial Stage|Late Training|
|---|---|---|---|
|RAGEN (Qwen2.50.5BInstruct)<br><br>|_G0_ ___O PO__ ____<br><br>|(THINK): Let’s start by considering the initial state: <MAP>. The goal is to reach the target ’PO___’ from the hole ’O__O__O__O’. I’ll start by moving to the left to avoid the hole.<br><br>(Action): Left || Up<br><br>|(THINK): Forbid the goal and go left then up. (Action): Left || Up|
|RAGEN (Qwen2.57BInstruct)| |(THINK): Move left to avoid the hole and then go up to reach the goal.<br><br>(Action): Left || Up|(THINK): Up may lead to the hole or the target. (Action): Up || Up|
|RAGEN (Llama3.18BInstruct)| |(THINK): To avoid the hole and go to the target, I should go up to stay safe and then move to the right to approach the goal from a safe distance.<br><br>(Action): Up || Right|(THINK): My thoughts are to go to the goal (Action): Up || Right|
|NLRL (Llama3.18BInstruct)| |(THINK): The current board position is <MAP>. The goal is on the top left, and the player is on the bottom left. The player is not in the hole. The available moves are 1, 2, 3, 4. Since the player is already on the leftmost position, moving left is not an option. The player can move down to get closer to the goal. Therefore, the best next move for P is to play action 2 to move down.<br><br>(Action): Down|(THINK): Based on the evaluations of the next board positions, moving up results in the highest final evaluation of +5, indicating that P is one block closer to the goal position G after the move. Although there is a hole one block away from P after moving up, the potential strategy of moving up and then moving right can help P avoid the hole and arrive at the goal position. In contrast, moving left does not immediately change P’s position, and moving down moves P away from the goal position G. Therefore, moving up is the most favourable move.<br><br>(Action): Up|

Table 17: Comparing detailed reasoning on FrozenLake environment between RAGEN [48] and NLRL (Ours). No matter what backbone model we used, RAGEN style of training will downgrade the model only to output the simplest reasoning. Our method encourages the model to develop rigorous reasoning as the training progresses.

"final_evaluation": After all your thought, finally judge how good the agent’s current position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position after taking this action. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.

### EXAMPLE: user: The (board, action) to evaluate: Board: P___ _O__ ___O GO__ Action: 2 (Down).

| |For your reference, below is the rollout sequence 1 after this (board, action):<br><br>After move 2 (Down), the board position is: ____ PO__ ___O GO__<br>After move 3 (Right), the board position is: ____ _X__ ___O GO__ The game is over. Player fall into the hole and therefore fails.<br><br><br>assistant: {"thought": {"Reflection": "It appears that the initial board position P___ _O__ ___O GO__ and move 2 (Down) was favorable for P, as P is one block closer to the goal position G.", "Win probability": "The win/success probability for P is larger.", "Threat": "There are three holes on the board. There is a hole only one block away from P after action has been taken, which can be dangerous if O choose to move right.", "Potential strategies": "Potential strategies for P include moving 2 (Down) twice to arrive at goal position."} "final_evaluation": -0.3}|
|---|---|
|USER|The board to evaluate is: Board: _OO_ G___ ____ _OOP Action: The next move is 2.<br><br>For your reference, below is the rollout sequence 1 after this (board, action): After taking action 2, the board position is _OO_ G___ ____ _OOP.<br><br>After taking action 2, the board position is _OO_ G___ ____ _OOP.<br><br>...... The game is over. Player fall into the hole and therefore fails. Now generate your evaluation for the (board, action) pair.<br><br>|
|ASSISTANT|{"thought": {"Reflection": "It appears that the initial board position|

G___ ____ _OOP and move 2 (Down) was not favorable for P, as P did not move after taking this action in both rollout sequences.", "Win probability": "The win/success probability for P is smaller.", "Threat": "There are four holes on the board. There are holes both above and below P after action has been taken, which can be very dangerous if P choose to move up or down.", "Potential strategies": "Potential strategies for P include moving 3 (Right) or 1 (Left) to avoid the holes above and below P."} "final_evaluation": -0.5}

###### Language State-action Value (After)

|Role<br><br>|Content|
|---|---|
|SYSTEM<br><br>|You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid. The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.<br><br>The meaning of each symbol in the state is: P: player, _: empty, O: hole, G: goal, X: player in hole, √ : player on goal<br><br>The possible actions are: 1:Left, 2:Down, 3:Right, 4:Up<br><br>You are learning how to evaluate a (board, action) pair in the FrozenLake by playing the game given the (board, action) pair and reflect the playing history. The playing history depicts a game in progress with a final result. Your answer consists of two elements:<br><br>- "thought": let’s think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. You should only rely on the playing history as context and don’t evaluate the game with your own judgment. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies. "final_evaluation": After all your thought, finally judge how good the agent’s current position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position after taking this action. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.<br><br>### EXAMPLE: user: The (board, action) to evaluate: Board: P___ _O__ ___O GO__ Action: 2 (Down).|

| |For your reference, below is the rollout sequence 1 after this (board, action):<br><br>After move 2 (Down), the board position is: ____ PO__ ___O GO__<br>After move 3 (Right), the board position is: ____ _X__ ___O GO__ The game is over. Player fall into the hole and therefore fails.<br><br><br>assistant: {"thought": {"Reflection": "It appears that the initial board position P___ _O__ ___O GO__ and move 2 (Down) was favorable for P, as P is one block closer to the goal position G.", "Win probability": "The win/success probability for P is larger.", "Threat": "There are three holes on the board. There is a hole only one block away from P after action has been taken, which can be dangerous if O choose to move right.", "Potential strategies": "Potential strategies for P include moving 2 (Down) twice to arrive at goal position."} "final_evaluation": -0.3}|
|---|---|
|USER Board:<br><br>|The board to evaluate is:<br><br>___P _OO_ _O__ __G_ Action: The next move is 3.<br><br>For your reference, below is the rollout sequence 1 after this (board, action): After taking action 3, the board position is ___P _OO_ _O__ __G_.<br><br>After taking action 3, the board position is ___P _OO_ _O__ __G_. The game is over. Player has reach maximum number of move and therefore fails.<br><br>...... Now generate your evaluation for the (board, action) pair.<br><br>|
|ASSISTANT|{"thought": {"Reflection": "It appears that the initial board position ___P|

_O__ __G_ and move 3 (Right) resulted in no change in the board position in all rollout sequences.", "Win probability": "The win/success probability for P is low, as P did not move closer to the goal position G in any rollout sequence.", "Threat": "There are four holes on the board. However, none of the rollout sequences resulted in P falling into a hole.", "Potential strategies": "Potential strategies for P include moving down to avoid the holes and get closer to the goal position."} "final_evaluation": -0.4}

###### Language Policy Improvement (Beginning)

|Role|Content|
|---|---|
|SYSTEM|You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid. The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.<br><br>Your task is to evaluate a given board position. The meaning of each symbol in the state is: P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal<br><br>The possible actions are: 1:Left, 2:Down, 3:Right, 4:Up<br><br>Your task is to determine the best next move based on the given board position. The evaluations of (board, action) pairs after possible moves are given. DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move. The evaluation for the next board is in the format of a json format, consisting of two elements:<br><br>- "thought": evaluation of the the board and action pair.<br><br>- "final_evaluation": After all your thought, finally judge how good the agent’s position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.<br><br><br>|
|USER<br><br>|Here is the board position: __P_ _G__ ____ ____. The possible moves are [2, 3, 4]. The following are the boards after each possible move:<br><br>### Evaluation for taking action 2: {"thought": {"Reflection": "It appears that the initial board position __P_ _G__ ____ ____ and move 2 (Down)|

| |was favorable for P, as P is one block closer to the goal position G after taking this action.", "Win probability": "The win/success probability for P is larger.", "Threat": "There are no holes on the board that are immediately threatening P after the action has been taken.", "Potential strategies": "Potential strategies for P include moving 2 (Down) to arrive at goal position, and then moving left to reach the goal."} "final_evaluation": 0.1}<br><br>### Evaluation for taking action 3: {"thought": {"Reflection": "It appears that the initial board position __P_ _G__ ____ ____<br><br>and move 3 (Right) was not favorable for P, as P is now further away from the goal position G.", "Win probability": "The win/success probability for P is smaller.", "Threat": "There are no holes on the board that are immediately threatening P after the action has been taken.", "Potential strategies": "Potential strategies for P could have been to move 2 (Down) to arrive at goal position, but the chosen action 3 (Right) did not lead to this outcome."} "final_evaluation": -0.4}<br><br>### Evaluation for taking action 4: {"thought": {"Reflection": "It appears that the initial board position __P_ _G__ ____ ____<br><br>and move 4 (Up) resulted in no change in the board position, which indicates that the move was not effective in making progress towards the goal.", "Win probability": "The win/success probability for P is lower.", "Threat": "There are no holes on the board that are immediately threatening P after the action has been taken.", "Potential strategies": "Potential strategies for P include moving 2 (Down) to move closer to the goal position G, but the rollout sequence shows that P did not take this opportunity and instead moved in other directions, which did not lead to a successful outcome."} "final_evaluation": -0.4}<br><br><br><br><br>Now, please give your evaluation and the best next move based on the given board position __P_ _G__ ____ ____. You should output your answer in the json format. Your answer consists of two elements:<br><br>- "thought": let’s think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.<br>- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 4, indicating the moving direction. Don’t output extra information except for the json format.<br>|
|---|---|
|ASSISTANT|{|

"thought" : "Analyzing the given board position __P_ _G__ ____ ____ and the evaluations of the possible moves, it appears that taking action 2 (Down) is the most favorable move. This is because the evaluation for this move indicates that P is one block closer to the goal position G and the win/success probability for P is larger. In contrast, taking action 3 (Right) or action 4 (Up) would not be effective in making progress towards the goal, as P would be further away from the goal position G or the board position would remain unchanged. Therefore, the best next move is to take action 2 (Down) to move closer to the goal position G.", "best_move" : 2 }

###### Language Policy Improvement (End)

|Role|Content|
|---|---|
|SYSTEM|You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid. The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.<br><br>Your task is to evaluate a given board position. The meaning of each symbol in the state is: P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal<br><br>The possible actions are: 1:Left, 2:Down, 3:Right, 4:Up<br><br>Your task is to determine the best next move based on the given board position. The evaluations of (board, action) pairs after possible moves are given. DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move. The evaluation for the next board is in the format of a json format, consisting of two elements:<br><br>- "thought": evaluation of the the board and action pair.<br><br>- "final_evaluation": After all your thought, finally judge how good the agent’s position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.<br><br><br>|
|USER<br><br>|Here is the board position: __G_ ___O _O__ P_O_. The possible moves are [4, 1, 3]. The following are the boards after each possible move:<br><br>### Evaluation for taking action 4: {"thought": {"Reflection": "It appears that the initial board position __G_ ___O _O__ P_O_ and move 4 (Up)|

was not favorable for P, as P is still two blocks away from the goal position G and there are holes around P.", "Win probability": "The win/success probability for P is smaller.", "Threat": "There are three holes on the board. There are two holes only one block away from P after action has been taken, which can be very dangerous if P choose to move right or left.", "Potential strategies": "Potential strategies for P include moving 4 (Up) again to arrive at the middle row and then move 3 (Right) twice to arrive at goal position, but the rollout sequence shows that this strategy is not successful."} "final_evaluation": -0.5}

### Evaluation for taking action 1: {"thought": {"Reflection": "It appears that the initial board position __G_ ___O _O__ P_O_ and move 1 (Left) resulted in no change in the board position, which indicates that the move was not successful due to the slippery ice.", "Win probability": "The win/success probability for P is not affected.", "Threat": "There are three holes on the board. There are two holes close to P, which can be dangerous if P moves right or up.", "Potential strategies": "Potential strategies for P include moving 4 (Up) to avoid the holes and move towards the goal position G."}

- "final_evaluation": -0.4} ### Evaluation for taking action 3: {"thought": {"Reflection": "It appears that the initial board position __G_ ___O _O__ P_O_ and move 3 (Right) was not favorable for P, as P is still far from the goal position G and there are holes nearby.", "Win probability": "The win/success probability for P is lower.", "Threat": "There are three holes on the board. There is a hole only one block away from P after action has been taken, which is very dangerous.", "Potential strategies": "Potential strategies for P include moving 4 (Up) to move closer to the goal position G." }

- "final_evaluation": -0.5}

Now, please give your evaluation and the best next move based on the given board position __G_ ___O _O__ P_O_. You should output your answer in the json format. Your answer consists of two elements:

- - "thought": let’s think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.

- - "best_move": the best move for the next player. The move should be in the format of a number from 1 to 4, indicating the moving direction. Don’t output extra information except for the json format.

|ASSISTANT|{ "thought" : "Analyzing the given board position __G_ ___O _O__ P_O_, we can see that the goal is located at the top left corner, and there are three holes on the board. The player is currently at the bottom right corner. We have three possible moves: 4 (Up), 1 (Left), and 3 (Right). After evaluating the next board positions, we can see that moving 4 (Up) and 3 (Right) both result in a final evaluation of -0.5, indicating that the player is still far from the goal and there are holes nearby. Moving 1 (Left) results in no change in the board position due to the slippery ice, but the final evaluation is -0.4, which is slightly better than the other two options. However, considering the potential strategies, moving 4 (Up) seems to be a better option as it allows the player to move closer to the goal position G.", "best_move" : 4 }<br><br>|
|---|---|

- D Prompts Here we provide all the prompt templates in three experiments.

- D.1 Maze experiment

|Language Value/Evaluation Prompt<br><br>EVAL_USER_PROMPT_S_V = f"""\ You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible.<br><br>The possible actions are "move up\n", "move down\n", "move left\n", "move right\n".<br><br>You are a highly skilled evaluator in this game and is expected to function similar to state value function in reinforcement learning. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent. Your output evaluation should be a json array that includes the following concepts:<br><br>- "thoughts": Let’s think step by step. Generate your detailed thought process and evaluation of the agent’s position and the distance towards the goal.<br>- "final_evaluation": Concisely judge how good the agent’s current position is compared to all the possible positions in the maze, in terms of reaching the goal.<br><br><br>Let’s start a new game. Now, please give your evaluation of the current state given the move history of the agent: ‘‘‘ {{game_content}} ‘‘‘ """|
|---|

Language TD Prompt

EVAL_USER_PROMPT_S_TD_G2_new = f"""\ You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible.

The possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead of the current maze position after taking the given action. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent, the action that the agent is going to take, *along with several key variations of trajectory pieces after taking this action (and the corresponding natural language evaluations of the trajectory pieces)*. Your task is to understand these look-ahead information and summarize, derive non-trivial analysis and understanding the *the agent’s position after taking the action*. Your output evaluation should be a json array with the following *two* concepts:

- - "thoughts": Let’s think step by step. Summarize the look-ahead information of the variations after taking action {{chosen_action}}.
- - "final_evaluation": Now Concisely judge how good the chosen action is, in terms of reaching the goal.

Now, please give your evaluation given action {{chosen_action}}, the *current environment state*:

‘‘‘ {{game_content}} ‘‘‘

and the look-ahead information of different variations after taking action {{chosen_action}}: """

|Policy Improvement Prompt<br><br>POLICY_IMPROVEMENT_PROMPT_TD = f"""\ You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible.<br><br>Your task is to determine the best action for the next time step given the current state (the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent).<br><br>Your possible actions are "move up\n", "move down\n", "move left\n", "move right\n".<br><br>The evaluations of the agent after possible actions are given. Each of them consists of two elements:<br><br>- "thoughts": Summarization of the look-ahead information of the variations after taking the chosen action.<br>- "final_evaluation": Judge how good the chosen action is, in terms of reaching the goal.<br><br><br>DO NOT judge the action based on your exterior knowledge, only use the given evaluations to determine the best move.<br><br>Here are the evaluations of each possible action: For action "move up", {{evaluations_up}} , For action "move down", {{evaluations_down}} , For action "move left", {{evaluations_left}} , For action "move right", {{evaluations_right}} Return the best action (choose only one from the possible actions) given the evaluations in a json array with a key "action". """|
|---|

###### D.2 Breakthrough Experiment Language Value/Evaluation Prompt

GAME_RULE_PROMPT = """Here is the rule for the Breakthrough board game: The game is played on an 5x5 board for 2 players (white and black), with each player starting with 10 pawns. white pawns are on the first two rows and black pawns are on the last two rows. Black moves first. In each turn, players can move one of their pieces one square forward, diagonally forward if the target square is empty. Or it can capture an opponent’s piece if that square is one step diagonally forward. The game ends when one player successfully break through the opponent lines -either move a piece to the opposite last row of the board or captures all of the opponent’s pieces.

For board representation, we use b for black pieces, w for white pieces, and . for empty squares. (1-5) and (a-e) are used to show the rows and columns index respectively."""

EVAL_SYSTEM_PROMPT = f"""{GAME_RULE_PROMPT} You are a highly skilled evaluator in this game. I will provide you with specific board information representing the current board. Your output should include the following concepts:

- 1. *Tactical Considerations*: Any immediate threats, potential tactics, or vulnerabilities in the position.
- 2. *Positional Evaluation*: Consideration of pawn structure, piece activity, control of key squares, and game safety.
- 3. *Suggested Moves*: One or two strong candidate moves for the side to move, along with a brief rationale for comparing different moves.
- 4. *Advantage*: Based on all previous rationale, determine if white or black takes advantage. Use <white> or <black> to represent the winning side. Your response should be informative and concise."""

EVAL_USER_PROMPT = """*The board you need to evaluate:* {board}"""

###### Language TD Prompt

TD_SYSTEM_PROMPT = f"""{GAME_RULE_PROMPT} You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead analysis of the current board position. I will provide you with current board representation, *along with several key variations starting from this position (and their corresponding natural language evaluations of the subsequent positions)*. Your task is to aggregate and compare these look-ahead information, to summarize, derive non-trivial analysis about the *current board*. Your output should include the following concepts:

- 1. *Tactical Considerations*: Any immediate threats, potential tactics, or vulnerabilities in the position.
- 2. *Positional Evaluation*: Consideration of pawn structure, piece activity, control of key squares, and game safety.
- 3. *Suggested Moves*: One or two strong candidate moves for the side to move, along with a brief rationale for comparing different moves.
- 4. *Advantage*: Based on all previous rationale, determine if white or black takes advantage. Use <white> or <black> to represent the winning side. Your response should be informative and concise.""" TD_USER_PROMPT = """*The board you need to evaluate:* {board}

Here are the look-ahead variations from the current board position:

*Key Variations and Subsequent Evaluation:*: {variations} Please provide your analysis and understanding of the current board position based on the provided look-ahead information. Your response should be informative and concise.""" VARIATION_PROMPT = """*Variation {i}:* Description of variation’s move sequence: {move_desc} Subsequent position evaluation: {subsequent_eval}"""

SUBSEQUENT_PROMPT = """The subsequent board is: {sub_board} The evaluation of this subsequent board is: {sub_eval}"""

- D.3 Tic-tac-toe Experiment

|Policy Inference Prompt<br><br>POLICY_INFERENCE_SYSTEM_PROMPT = """\ You are an expert agent at playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.<br><br>Your task is to choose the best move given board position. The board consists of "O", "X" and grid number. The grid number indicates empty grid. You should output your answer in the json format. Your answer consists of two elements:<br><br>- "thought": let’s think step by step. Generate your detailed thought process and plan for the next move.<br>- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions."""<br><br>POLICY_INFERENCE_EXAMPLE_USER_PROMPT = """\ Here is the board position and the next player is O: O | O | X<br><br>--------4 | X | 6<br><br><br>--------7 | 8 | 9 The available move positions are 4, 6, 7, 8, 9. """<br><br>POLICY_INFERENCE_EXAMPLE_ASSISTANT_PROMPT = """\ {"thought": "It appears that the initial board position O | O | X --------4 | X | 6 --------7 | 8 | 9 was favorable for X, as X has occupied the positions 3, 5 and X can win by occupying the position 7. O has occupied the positions 1, 2. Therefore, the best move for O is to occupy the position 7 to block X and create a potential winning opportunity by occupying the positions 1, 4, 7.", "best_move": 7} """<br><br>POLICY_INFERENCE_USER_PROMPT = """\ Here is the board position and the next player is {next_player}: {state}. The available move positions are {available_positions}. """|
|---|

Policy Improvement Prompt

POLICY_IMPROVEMENT_SYSTEM_PROMPT = """\

You are playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The board consists of "O", "X" and grid number. The grid number indicates empty grid. Your task is to determine the best move for the next player based on the given board position and the next player. The evaluations of (board, action) pairs after possible moves are given. DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move. The evaluation for the next board is in the format of a json format, consisting of two elements:

- - "thought": evaluation of the the board and action pair.
- - "final_evaluation": Judge which side takes advantage. 1 means ’O’ takes advantage, -1 means ’X’ takes advantage, and 0 means the game will be a draw. """

POLICY_IMPROVEMENT_USER_PROMPT = """\ Here is the board position and the next player is {next_player}: {state}. The possible moves are {available_positions}. The following are the boards after each possible move: {next_states}

Now, please give your evaluation and the best move for {next_player} based on the given board position {state}. You should output your answer in the json format. Your answer consists of two elements:

- - "thought": let’s think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.
- - "best_move": the best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions {available_positions}. Don’t output extra information except for the json format. """

Policy Evaluation Prompt

POLICY_EVALUATION_SYSTEM_PROMPT = """You are a player of the game of Tic Tac Toe. \nThe game goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid. \nYou are learning how to evaluate a (board, action) pair in the tic tac toe by playing the game given the (board, action) pair and reflect the playing history. \nThe playing history depicts a game in progress with a final result. Your answer consists of two elements:

- - "thought": let’s think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. You should only rely on the playing history

- as context and don’t evaluate game with your own judgement. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies.

- - "final_evaluation": After all of your thoughts, judge which side takes advantage. 1 means you think ’O’ takes advantage, -1 means you think ’X’ takes advantage, and 0 means you think the game will be a draw. You should output your answer in the json format.""" POLICY_EVALUATION_EXAMPLE_USER_PROMPT = """The (board, action) to evaluate is O’s turn: Board: O | O | X
- ---------

- 4 | X | 6

--------7 | 8 | 9 Action: The O’s move is 4.

Below is the rollout sequence after this (board, action): After O’s move 4, the board position is: O | O | X --------O | X | 6 --------7 | 8 | 9 After X’s move 7, the board position is: O | O | X --------O | X | 6 --------X | 8 | 9 The game is over. X wins. X wins by occupying the positions 3, 5, 7."""

POLICY_EVALUATION_EXAMPLE_ASSISTANT_PROMPT = """ {"thought": {"Reflection": "It appears that the initial board position O | O | X --------4 | X | 6 --------7 | 8 | 9 and O’s move 4 were not favorable for O, as X was able to block on O’s move at 7 and ultimately win the game.", "Win probability": "The win probability for X is large, while the win probability for O is low.", "Threat": "X has played

- at 5 and 3. X can win by move 7. O can occupy 1, 4, 7, and create a potential winning opportunity. X occupies 5, which is a key position to win the game.", "Potential strategies": "Potential strategies for O include playing at 7 to block X and create a potential win by occupying 1, 4, 7. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and needs only 1 move to complete the win."} "final_evaluation": -0.8}"""

|Policy Evaluation Prompt (Continue)<br><br>POLICY_EVALUATION_USER_PROMPT = """The board to evaluate is {player}’s turn: Board: {board} Action: The {player}’s move is {action}.<br><br>Below is the rollout sequence 1 after this (board, action): After {player} taking action {action}, the board position is: {rollout_board_1} The game is over. {winner_1} wins. {winner_1} wins by occupying the positions {positions_1}.<br><br>... Below is the rollout sequence N after this (board, action): After {player} taking action {action}, the board position is: {rollout_board_N} The game is over. {winner_N} wins. {winner_N} wins by occupying the positions {positions_N}. Now generate your evaluation for the (board, action) pair."""|
|---|

## Natural

## Reinforcement Learning (RL)

###### Policy: Hu

[ T

- 0.5

Get Reward Values

- 1

Traditional Agent

World

State [0.5]

## Value function in RL

## Human-

“Not ab

Go s

Yo

|Monte-Carlo Estimate<br><br>Aggregating evaluation from multi trajectories<br><br>Aggregating all steps in one trajectory<br><br>Intermediate changes over action, reward and next state<br><br>| |
|---|
<br><br>Reward decay|Temporal Difference Estimate<br><br>| |
|---|
<br><br>Aggregating evaluation from multi one-step samples<br><br>Aggregating intermediate changes and future evaluation<br><br>Intermediate changes over action, reward and next state<br><br>Future evaluation over next state<br><br>Future evaluation decay<br><br>“Not ab<br><br>Go s<br><br>Yo|
|---|---|

- Figure 11: Comparison between traditional MC and language MC

- 0.5

Get Reward Values

- 1

Reinforcement Learning (RL)

Traditional Agent

World

Natural

Value function in RL

State [0.5]

[ T

Human-

Policy: Hu

|Monte-Carlo Estimate<br><br>Aggregating evaluation from multi trajectories<br><br>Aggregating all steps in one trajectory<br><br>Intermediate changes over action, reward and next state<br><br>| |
|---|
<br><br>Reward decay|
|---|

|Temporal Difference Estimate<br><br>| |
|---|
<br><br>Aggregating evaluation from multi one-step samples<br><br>Aggregating intermediate changes and future evaluation<br><br>Intermediate changes over action, reward and next state<br><br>Future evaluation over next state<br><br>Future evaluation decay|
|---|

- Figure 12: Comparison between traditional TD and language TD

###### E Analogy

- E.1 Language MC

As mentioned in the main paper and in Figure 11, language MC is an analogy of traditional MC. Specifically, the mean and sum operation in traditional MC estimate can be replaced by the new lew language aggregator 𝐺1. 𝑟 can correspond to the intermediate transition for action, reward, and next state. The discount factor 𝛾 can also have an interesting analogy in 𝐺1 if 𝐺1 acts as a lossy information compressor. Specifically, just as the discount factor reduces the weight of future rewards, 𝐺1 can perform a lossy compression of future information, selectively retaining only the recent and most relevant aspects.

- E.2 Language TD

Most Analogies between language TD and traditional TD share similarities with that in the language MC setting, as shown in Figure 12. 𝐺1 is a language aggregator over multiple lookahead variations, corresponding to the trajectory mean operation, while 𝐺2 combines immediate change/intermediate transition and future value evaluation, akin to the sum operation.

###### F Limitation and future work

Limitations. Currently, NLRL still faces several limitations. First, our experiments demonstrate its effectiveness primarily in environments with discrete action spaces and low-dimensional (textualisable) states. Its performance in environments involving continuous action spaces and high-

dimensional states (e.g., low-level robotic control tasks) remains largely unexplored. Second, our experiments are primarily conducted in small-scale settings due to the substantial computational cost associated with invoking Large Language Models. The time efficiency of the current NLRL approach is significantly lower than that of traditional RL methods using smaller networks.

Future work. Given NLRL’s potential, there are several promising directions for future research. First, a theoretical foundation for the entire framework in the language space is needed to move beyond analogical reasoning towards a more robust, principled basis. This would help formalise the framework and enhance its generalisability. Another important direction is exploring how NLRL can be integrated more organically with existing research on self-evaluation, self-improvement, LLM agent framework, and LLM’s planning. Such integration could unlock new capabilities and efficiency improvements. Additionally, in terms of application domains, there is significant potential to extend NLRL beyond the current setups–we are actively exploring extending NLRL to general domains beyond games, such as reasoning tasks [11] and code [7].

###### NeurIPS Paper Checklist

###### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Section 4 and Appendix C have discussed related experiments claimed in the abstract and introduction in full detail. The conclusion supports the paper’s contributions and scope.

Guidelines:

- • The answer NA means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

###### 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Appendix F has discussed potential limitation of our works. Guidelines:

- • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate "Limitations" section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

###### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA] Justification: The paper does not include theoretical results. Guidelines:

- • The answer NA means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

###### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] Justification: Section 4, Appendix C and D include detailed experiments setup and hyperparameter settings used in our work. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

###### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: The paper provides open access to the code anonymously. Guidelines:

- • The answer NA means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

###### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes] Justification: Section 4, Appendix C and D include detailed experiments setup and hyperparameter settings used in our work. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

###### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports error bars on some of the experiments, such as Table 1. Due to high computational burdens, the paper does not include error bars for all of our experiments, but the paper contains every detail necessary for experiment reproduction.

Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

###### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: The paper includes the computational resources required to run the experiments in Appendix C. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

###### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: The paper conforms with the Code of Ethics. Guidelines:

- • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

###### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] Justification: The paper mainly focuses on foundational research and and not tied to particular applications. Guidelines:

- • The answer NA means that there is no societal impact of the work performed.

- • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

###### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] Justification: The paper does not release models or datasets. Guidelines:

- • The answer NA means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

###### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: Existing assets used in the paper are credited and the necessary licenses are mentioned. Guidelines:

- • The answer NA means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

###### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes] Justification: The new assets introduced is well documented. Guidelines:

- • The answer NA means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

###### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

###### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

###### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes] Justification: The paper is about foundational research on LLM and has described the usage of LLMs. Guidelines:

- • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

