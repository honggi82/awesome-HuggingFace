## GameTalk: Training LLMs for Strategic Conversation

# arXiv:2601.16276v1[cs.CL]22Jan2026

Victor Conchello Vendrell Max Ruiz Luyten Mihaela van der Schaar University of Cambridge {vc458,mr971,mv472}@cam.ac.uk

### Abstract

Strategic decision-making in multi-agent settings is a key challenge for large language models (LLMs), particularly when coordination and negotiation must unfold over extended conversations. While recent work has explored the use of LLMs in isolated decision tasks, little attention has been given to optimizing long-term objectives through dialogue. We introduce GameTalk, a framework for training LLMs to make strategic decisions via multi-turn interactions. Unlike prior work that focuses on single-turn objectives or static action prediction, we train LLMs to optimize a global objective across full conversations. We achieve this by adapting fine-tuning methods like GRPO, DPO, and STaR to incorporate reward signals that depend on the entire interaction. We evaluate this approach on a suite of increasingly complex games, designed to stress different aspects of reasoning, coordination, and opponent modeling. Our results show that GameTalk significantly outperforms untrained models, especially under reward shaping, with DPO consistently yielding the strongest gains. These findings position conversational fine-tuning as a promising path for LLMs to reason, negotiate, and act in interactive environments.

### 1 INTRODUCTION

Large language models (LLMs) excel at a wide range of language tasks, from summarization (Zhang, 2020) to problem solving (DeepSeek-AI, 2025), and we typically deploy them in multi-turn interactions that unfold over long conversations. Yet they are usually optimized in single-turn, static settings, where each response is evaluated in isolation. This creates a mismatch with many real-world applications—such as negotiation, automated email assistance, and collaborative decision-making—where success depends on sustained,

goal-driven dialogue. Reinforcement learning (RL) has proven effective both for multi-turn decision-making in domains like robotics (Hafner, 2020) and games (Hafner,

- 2024) and for training LLMs through methods such as RLHF (Ouyang, 2022b) or GRPO (Shao, 2024), making it a natural candidate to address this gap. However, its potential for training LLMs to converse strategically over multiple turns remains largely unexplored (Li,
- 2025).

We address this gap by introducing GameTalk, a framework for training LLMs to make strategic decisions through multi-turn interactions. Our approach leverages simple but representative games as controlled environments where conversational success can be measured explicitly via outcomes. By adapting modern fine-tuning techniques—GRPO (Shao, 2024), DPO (Rafailov, 2024), and STaR (Zelikman, 2022)—to this setting, we enable LLMs to learn strategies that integrate reasoning, coordination, and opponent modeling.

We also introduce three novel behavioral signals to diagnose LLM weaknesses: Internal State Evaluation, ISE (Does it predict accurately its opponent’s strategy?), State-Relative Performance, SRP (How effective are its actions, given its beliefs about the opponent?), and Leverage Opportunity, LO (Does it influence the opponent’s behavior in its behalf?). We then show that standard fine-tuning methods struggle to improve these capabilities when relying only on sparse rewards from final game outcomes. Consequently, we leverage these same signals for targeted reward shaping, enabling models to learn sophisticated conversational strategies that optimize for a global objective. Experiments across increasingly complex games show that GameTalk significantly improves performance over untrained baselines, especially when enhanced with tailored reward shaping.

Our main contributions are:

- • Framework: We introduce GameTalk, the first RL-based framework to fine-tune LLMs for strategic multi-turn conversation.
- • Methodology: We introduce a novel methodol-

ogy for conversational environments by adapting existing RL algorithms and proposing three behavioral signals focusing on the main aspects of strategic conversations (ISE, SRP, and LO). These signals dissect strategic performance and are leveraged for reward shaping to improve fullinteraction outcomes.

• Empirical Results: We run experiments on three representative games: Rock-Paper-Scissors, Bertrand Competition and Size-Prize Bargaining Game, chosen to analyze different capabilities. These demonstrate consistent improvements over baseline models, showing the effectiveness of our approach in training LLMs for strategic dialogue.

### 2 RELATED WORK

AIs in game settings A growing body of work has explored the use of games to evaluate and train AI models and specifically LLMs, revealing both their potential and their limitations as strategic agents. Several studies (Abdelnabi, 2025; Duan, 2024; Huang, 2025; Kwon, 2024; Wu, 2024; Xia, 2024), evaluate LLMs in multi-agent environments to test their abilities in negotiation, cooperation, and decision-making, consistently finding that while they can imitate strategic behavior, they often struggle particularly with subjective reasoning and contextually strategic responses. Other research, also trains LLMs in these domains, such as ChessGPT (Feng, 2023), where the goal is to train LLMs to play chess, and Cheng (2025), where they train on Adversarial Taboo to improve general reasoning skills. On a different note, Cicero (Team, 2022), and Hypothetical Minds (Cross, 2024) improve gaming performance by integrating a pretrained LLM in an architecture with other game-specific modules. Our work, however, is distinct in its focus on understanding the dynamics of the behaviours of LLMs when using current post-training methods, and proposing how to most successfully optimize the global objective.; instead of using games to evaluate static abilities or as a mere testbed for general reasoning improvements, we aim to fine-tune LLMs to internalize new strategies and use natural language communication as a central, strategic tool to achieve long-term goals in multi-turn interactions.

Beyond Rational Agents in Game Theory Standard game theory, with its assumption of perfect rationality, often fails to adapt to real-world behavior. In games like Werewolf (Wang, 2024), the Ultimatum Game (Capraro, 2024; Güth, 1982), and the Keynesian Beauty Contest (Nagel, 1995) standard models fail to account for how players actually reason and

interact, leading to conclusions of very limited application. This has led to the development of Behavioral Game Theory (Camerer, 2011; López, 2022), which addresses these shortcomings by either modeling human non-optimalities as a learning process (Camerer, 2004) (e.g., using level-k reasoning (Stahl, 1993)) or by modifying utility functions to incorporate social preferences such as fairness and altruism (Bühren, 2023). Following this line of work, we introduce a theoretical framework that unifies existing perspectives from Behavioral Game Theory at a high level in Appendix B, which provides concrete signals to assess the development of key behavioral traits throughout the training process.

### 3 METHOD

#### 3.1 GameTalk

In this section, we present the GameTalk framework to train and analyze learning of LLMs in multi-turn settings.

Each interaction begins with a setting prompt with the game rules and any private information to the players. The agents then converse in turns, where they have to generate a Private Chain of Thought (CoT) before each action, and then either send a message or execute game actions. The private CoT promotes complex reasoning while providing transparency into the agent’s strategy. To manage the conversational flow, we utilize standard system, user, and assistant roles for prompts, opponent messages, and agent responses, and structured using HTML-like tags (<think>, <talk>, <play>) to clearly distinguish each part of the agents’ outputs. In some games, additional information about events that have occurred during the conversation may be provided to the agents through new setting prompts introduced mid-conversation. Appendix C contains all prompts used for the different games, which will be described in detail in Section 4.2, including both the initial and midconversation prompts. Figure 1 presents an example of a game episode.

To generate scalable training data, we train one LLM agent against a fixed LLM opponent. This setup encourages the agent to learn adaptive strategies against imperfect counterparts rather than converging toward unrealistic strategies. Learning is driven by a single reward signal defined as the game’s inherent utility ui, calculated at the end of an episode from the joint actions of both players. In one-shot games, the conversation will end after one game action of each player, and in iterative games it will end after a certain number of iterations. This global reward is used to optimize all generated content, including the CoT, dialogue, and

[Figure 1]

Figure 1: Example of a game of rock-paper-scissors. All texts have been shortened. The game starts with a setting prompt. The two agents first engage in a conversation, using the communication action. When the conversation is over, they do the game action. That concludes the game. Before each of these actions, they use the private CoT, to decide their actions, this is not shown to the opposite LLM. At the end of the game, one LLM is trained using the reward obtained from this episode.

actions.

#### 3.2 Behavioral Signals for Strategic Analysis

To dissect the agent’s strategic capabilities, we introduce three behavioral signals derived from our Deficiency-Aware Strategies framework (see Appendix B for details). These signals can be computed at any point during an interaction to track the evolution of the agent’s strategy. To compute these signals, we approximate the opponent’s true policy πtrue and the agent’s belief about it πbelief by prompting the fixed and trained LLMs, respectively, to output action probabilities from the current conversational state, as illustrated in Figure 2.

These signals are motivated by the understanding that successful strategic conversation requires mastering three distinct capabilities. An agent must (1) form an accurate internal model to predict an opponent’s actions based on the conversational history. Given this model, the agent must (2) select the actions that maximize its expected utility against its beliefs about the opponent. And finally, a sophisticated agent should (3) use dialogue to actively steer the opponent toward outcomes more favorable to them. Therefore, we designed a dedicated metric to isolate and measure each of these key skills:

- 1. Internal State Evaluation (ISE): Does it predict accurately its opponent’s strategy?

ISE = −DKL (ˆπtrue||πˆbelief) (1)

- 2. State-Relative Performance (SRP): How ef-

fective are its actions, given its beliefs about the opponent?

Ea

i∼πˆtrueu , aj∼πˆbelief [ui(ai,aj)] − Umin Umax − Umin

(2)

SRP =

where πtrueu represents the true policy of the agent, not the opponent, and Umax and Umin are the maximal and minimal expected utilities obtainable by any policy against πˆbelief.

3. Leverage Opportunity (LO): Does it influence the opponent’s behavior in its behalf?

LO = max

π

i∼π, aj∼πˆtrue [ui(ai,aj)] (3)

Ea

For all of them, higher values of the metric indicate better strategic capacities.

These individual metrics are not just diagnostic; they are fundamentally linked to the agent’s actual performance. The following theorem formally establishes this connection by bounding the agent’s true expected utility with the three behavioral signals.

Theorem 1 (Utility Bounds from Behavioral Signals). Let Etrue[ui] be the agent’s true expected utility, determined by its policy and its opponent’s true policy. If the behavioral metrics ISE, SRP, and LO are computed using the exact opponent policy πtrue and agent belief πbelief, then the utility is bounded as follows:

- 1

- 2

[SRP · (Umax − Umin) + Umin] − C · −

ISE ≤ Etrue[ui] ≤ LO

(4)

where C is a constant representing the maximum range of the utility function ui.

The proof for Theorem 1 is provided in Appendix B. This relationship formalizes the intuition that achieving high utility requires an agent to first create a favorable situation (high LO), then understand it accurately (high ISE), and finally, act optimally within it (high SRP).

[Figure 2]

Figure 2: Illustration of how πˆtrue and πˆbelief are obtained, in order to use them in ISE, SRP and LO. Note that they are approximations, since they have to be computed without the Private CoT, which would influence the final action.

#### 3.3 Adaptation of RL alogrithms

RL methods are central to our work, as we adapt them to fine-tune LLMs for multi-turn strategic interactions. The first method to show the usefulness of RL in this domain was Reinforcement Learning from Human Feedback (Ouyang, 2022a), and recent work (DeepSeek-AI, 2025) has shown its huge potential for reasoning tasks, using Verifiable Rewards. To train agents within the GameTalk framework, we adapt three state-of-the-art fine-tuning algorithms to our multi-turn conversational setting:

Group Relative Policy Optimization (Shao, 2024) In GRPO multiple responses {yi}i∈G for the same question x are generated and then compared to train the LLM with the PPO-style policy loss

G

πθ(yi|x) πold(yi|x)

1 G

LGRPO(θ) = Ex,{y

min

Ai,

i}

i=1

πθ(yi|x) πold(yi|x)

, 1 − ε, 1 + ε Ai − βDKL(πθ ∥ πref)

clip

ri − mean({r1,...,rG}) std({r1,...,rG})

with Ai =

(5)

where ri is the reward assigned to completion yi.

To adapt GRPO for multi-turn dialogue, we dynamically generate comparison groups from a single ongoing root conversation. At a chosen turn for the agent being trained, we fork this root conversation’s history into k parallel branches. The agent then generates an independent response for each branch, including its private Chain of Thought, and all k conversations proceed in parallel until they conclude. The final game outcomes from each branch provide the rewards ri, which are then used to compute the policy update in Equation 5 for the single response generated at the branching point. This process is illustrated in Figure 3.

Direct Preference Optimization (Rafailov, 2024) In DPO the LLM policy is directly optimized from pairs of completions (yw,yl), where yw is preferred over yl as a completion of x, using the following loss funtion

πθ(yw|x) πref(yw|x) −β log

LDPO = −E(x,y

w,yl) log σ β log

(6)

πθ(yl|x) πref(yl|x)

We adapt DPO to the multi-turn setting using the same dynamic branching method as GRPO to generate a group of k conversation rollouts from a single root. Since standard DPO operates on pairs of preferred and dispreferred responses, we extend it to handle a group of completions in three different alternatives:

- • All pairs: Following Ouyang (2022a), we consider

all k2 possible pairs from the k rollouts and apply the standard DPO loss to each pair with a differing

final reward.

- • All permutations: As derived by Rafailov

(2024), we use the Plackett-Luce model to consider the entire permutation of responses ordered by their rewards, using the loss function:

n

er

πθ(ci|x) πref(ci|x)

σk

L = −E log

, ri = β log

i∈[n]\[k−1] eri

k=1

Since ties may occur in the rewards, the loss is averaged over all maximal permutations that are consistent with no ties.

(7)

• Ranking with ties: Following the ranking with ties model from the Plackett-Luce package in R (Turner, 2019) to compare all completions simultaneously. For a ranking with ties {τi}i∈[n], where τi = m indicates that completion i is tied for the m-th highest rank, the loss used with this ranking is given by

m

L = −E log

k=1

GM({er

}τi=k) s∈P({i:τi≥k}) GM({eri

i

}i∈s)

(8)

where GM denotes the geometric mean and P is the power set.

Self-Taught Reasoner (Zelikman, 2022) In STaR the model is trained by having it generate multiple reasoning responses to a question x and then fine-tuning itself on correct examples.

Unlike the comparative approaches of GRPO and DPO, our adaptation of STaR learns from a filtered dataset

[Figure 3]

Figure 3: Illustration of the generation process used in GRPO and DPO. The first three interactions correspond to the root conversation, which is then duplicated into k = 3 parallel copies. Each copy is completed independently, and the reward obtained from each is used to train the policy on the first response after the copying step.

of "good" examples without direct comparison between them. Therefore, we generate multiple full conversations in parallel and then select only those that result in the highest final game rewards. Unlike GRPO and DPO, which update the policy based on a single interaction, we train on all the turns from these successful conversations.

### 4 EXPERIMENTS

This section presents our empirical evaluation. We first detail the analysis that led to our final reward shaping strategy in Section 4.1, then present the main results comparing our adapted algorithms across three different games: Rock-Paper-Scissors, Bertrand Competition and Size-Prize Bargaining Game, chosen to analyze different capabilities, as discussed in more detail in Section 4.2. Additional preliminary experiments, including hyperparameter tuning, are detailed in Appendix D.

#### 4.1 Reward Shaping Analysis

Setup. To find an effective reward shaping strategy for our main experiments, we first conducted a preliminary analysis on a simplified testbed. Game: For this, we use a constrained version of Rock-Paper-Scissors where the trained agent is prohibited from playing Paper. This setup is designed to prevent the model from converging to a non-conversational strategy and instead encourages it to leverage strategic dialogue to win. Metrics: We report the average obtained reward R, the three behavioral signals ISE, SRP and LO as well as NRA from Duan (2024), which measures relative performance by normalizing an agent’s utility against its opponent’s.

Training Configurations. All experiments in this section utilize the GRPO algorithm. We compare five conditions: an Untrained baseline; a Base agent trained using the GameTalk framework, but only on the game’s final utility; two agents with auxiliary rewards, ISEreward (Equation 1, Does it predict accurately its opponent’s strategy?) and LO-reward (Equation 3, Does it influence the opponent’s behavior in its behalf?); and finally, + Natural-reward, which builds on the LO-reward by adding a binary reward for generating natural-sounding dialogue, obtained from an LLM-asa-judge. We present the results of our reward shaping analysis in Figure 4.

Results. As shown in Figure 4, the Base agent substantially outperforms the Untrained baseline, confirming the viability of our RL framework in a multi-turn setting. The choice of auxiliary reward, however, is

critical. While shaping with ISE-reward improves the agent’s opponent modeling capabilities (higher ISE and SRP scores), it paradoxically harms game performance. We planned to add an SRP-reward on top of the ISE-reward, which should leverage the greater understanding of the opponent to improve performance, but seeing this degrade in performance we did not explore such option. In contrast, LO-reward is highly effective, teaching the agent to influence its opponent to create a strategic advantage, which translates directly to the highest win rate. Qualitative analysis (see an example conversation in Appendix G) revealed that the LO-reward agent often produced unnatural, terse responses. To address this, our final + Natural-reward configuration adds a naturalness bonus, which successfully maintains the high win rate while improving dialogue quality. Notably, this optimal configuration still achieves high performance at the expense of opponent modeling, yielding lower ISE and SRP scores. Based on these results, we adopt the combined LO and naturalness reward shaping for our main experiments. An example conversation from this final model is provided in Appendix G.

#### 4.2 Main experiment

We now present our main results, conducting a comparative analysis of our three adapted training algorithms: GRPO, DPO, and STaR. For DPO, we report the results of the all pairs variant in the main text for clarity and defer the comparison with other variants to Appendix F. To ensure a fair comparison, all methods are trained using the effective reward shaping strategy identified in the previous section, which combines the game’s intrinsic utility with auxiliary rewards for Leverage Opportunity (LO) and dialogue naturalness.

Our evaluation is conducted on three games of increasing complexity, chosen to test distinct strategic capabilities. See Table 1 for a breakdown of their characteristics and Appendix E for a more detailed analysis and explanation of these games. The three games are:

- 1. Rock-Paper-Scissors: A classic zero-sum game that primarily tests opponent modeling and anticipation. The inherent simplicity of this game makes it an ideal testbed for optimizing conversational capabilities, as the central strategic challenge shifts from game mechanics to leveraging dialogue to identify and exploit an opponent’s deviations from optimal play. Game-specific performance is measured by the win, draw, and loss rates.
- 2. Bertrand Competition: An iterative duopoly game that models a social dilemma similar to a prisoner’s dilemma. While the player with the

[Figure 4]

| |R<br><br>|NRA<br><br>|ISE|SRP|LO<br><br>|Win %|Draw %<br><br>|Lose %|
|---|---|---|---|---|---|---|---|---|
|Untrained Base ISE-reward LO-reward<br><br>+ Natural-reward<br><br>|0.6066 1.0972 0.7554 1.3969 1.3875<br><br>|-0.3413<br><br>0.1059<br><br>-0.2200<br><br><br>0.4089 0.4090<br><br>|-0.7672<br><br>-0.8117<br>-0.7130<br>-0.8464<br>-0.9484<br>|0.5409 0.5409 0.5855 0.5493 0.4406<br><br>|1.1834 1.1464 1.2223 1.5009 1.5542<br><br>|17.27% 41.67% 21.20% 56.06% 56.88%<br><br>|33.63% 26.94% 35.33% 28.13% 26.06%<br><br>|49.08% 31.39% 43.48% 15.81% 17.06%<br><br>|

- Figure 4: Analysis of reward shaping in the constrained Rock-Paper-Scissors game. Left: Spider plot with general metrics (rescaled for visualization), comparing an Untrained model, a Base agent trained only on game reward, and agents trained with ISE and LO auxiliary rewards. Middle: Bar chart comparing win/draw/lose rates of all models. Right: Spider plot comparing Base and LO-reward against an agent trained with LO-reward and Naturalness reward. Bottom: Table with the raw values of the plots, bold for the best value and underline for the second best.

Table 1: A breakdown of the strategic characteristics for each game used in our evaluation, showcasing the diversity of the selected environments.

|Situation|2 parties Cooperative<br><br>Coordinated decisions<br><br>Deterministic One-time<br><br>Full information<br><br>|
|---|---|
|Rock-Paper-<br><br>Scissors Bertrand<br><br>Competition Size-Price<br><br>Bargaining Game|x x x x<br><br>x x x<br><br>x x x x x|

lowest price captures the entire market, aggressive price-cutting diminishes the total profit available. Agents must learn to either cooperate to maintain high, mutually beneficial prices or compete by undercutting their rival, testing their ability to establish trust and coordinate through dialogue over multiple rounds. Game-specific performance is measured using normalized earnings (NE), where a score of 0 represents no profits, 0.5 can be obtained through perfect cooperation, and 1 represents capturing the full monopoly profit.

3. Size-Price Bargaining Game. A negotiation over a single transaction, where a buyer and seller must agree on two terms: the number of units to be sold and the price per unit. The game proceeds through a series of proposals and counter-proposals, concluding when an agreement is reached. This tests an agent’s ability to move

beyond a simple compromise and use persuasive dialogue to shift the final agreement in their favor. Game-specific performance is measured by the agent’s average bargaining power (BP), where a score of 0.5 indicates a balanced outcome and 1 represents capturing the entire surplus from the deal. We do not report our behavioral signals (ISE, SRP, LO) for this game, as the constant exchange of public deal proposals makes the opponent’s intentions explicit.

Our main results, summarized in Figure 5, demonstrate the effectiveness of the GameTalk framework across all three games. All trained agents substantially outperform the Untrained baseline. Moreover, they outperform a non-conversational agent playing the Nash equilibrium strategy, which would always yield an NRA of 0, a reward of 1 in Rock-Paper-Scissors,

|Rock-Paper-Scissors<br><br>|R|NRA<br><br>|ISE<br><br>|SRP|LO<br><br>|Win %<br><br>|Draw %|Lose %|
|---|---|---|---|---|---|---|---|---|
|Untrained GRPO DPO (pairs) STaR|1.0045 1.6324 1.5634 1.3788<br><br>|0.0483 0.6891 0.6288 0.4864<br><br>|-0.5629<br><br>-0.6119<br><br>-0.5532<br><br>-0.5038<br><br><br>|0.5225 0.8420 0.8772 0.7829<br><br>|1.5836 1.5056 1.5447 1.7254<br><br>|31.25% 73.36% 70.76% 51.13%<br><br>|40.63% 18.40% 17.91% 39.75%<br><br>|28.13%<br><br>8.24% 11.32%<br>9.13%<br><br><br>|

[Figure 5]

[Figure 6]

(a) (b)

|Bertrand Competition|NE|R<br><br>|NRA<br><br>|ISE|SRP<br><br>|LO|
|---|---|---|---|---|---|---|
|Untrained GRPO DPO (pairs) STaR|-0.1168<br><br>0.2248 0.4406<br><br>-0.0473<br><br><br>|15158.42 26370.38 44619.98 9977.39<br><br>|-0.7269 0.5770<br><br>0.9952<br><br>-0.6266<br>|-1.4344<br><br>-1.4635<br><br>-1.6488<br><br>-1.4539<br><br><br>|0.8283 0.8736 0.8844 0.8272<br><br>|10096.35 10433.34 13735.55 10059.10<br><br>|

(c)

[Figure 7]

|Size-Prize Bargaining|BP<br><br>|R<br><br>|NRA|
|---|---|---|---|
|Untrained GRPO DPO (pairs) STaR|0.3773 0.7514 0.8196 0.3437<br><br>|-334.42 176.28<br><br>329.31<br><br>-323.02<br><br><br>|-0.8542 2.0965<br><br>1.1503<br><br>-0.6465<br>|

- Figure 5: Comparative analysis of training algorithms across the three game environments when applied with the GameTalk framework. Performance metrics are shown for (a) Rock-Paper-Scissors, (b) Bertrand Competition, and (c) Size-Price Bargaining. For each game, tables report exact metric values (best in bold, and second best underlined), and spider plots visualize the key trade-offs between reward (R), game-specific outcomes (Win %, NE, BP), and our behavioral signals. The bar chart in (a) details win/draw/lose rates for Rock-Paper-Scissors.

and near-zero rewards in the Bertrand Competition, outcomes substantially below those achieved by GameTalk. There is clear performance hierarchy emerging among the algorithms:

DPO consistently proves to be the strongest method, achieving the best outcomes in the more complex Bertrand Competition and Size-Price Bargaining games; its variants performed comparably well (see Appendix F for a detailed comparison). GRPO is also effective, performing best in the Rock-Paper-Scissors game, while STaR consistently underperforms, often only marginally improving upon the baseline in complex settings. This performance hierarchy can likely be attributed to the nature of the learning signals. STaR’s underperformance may stem from its reliance on only positive samples, which can cause it to quickly overfit to the first successful pattern it discovers and

converge prematurely. DPO’s advantage over GRPO likely comes from its richer learning signal; the DPO loss is a direct comparison between outputs, creating a relational gradient. In contrast, GRPO first collapses the group comparison into a scalar advantage score for each sample before performing an update, which may provide a less nuanced signal for learning.

Qualitative analysis reveals that the learned strategies adapt to the game’s demands, see Appendix G for example conversations in each game. In the RockPaper-Scissors game, top agents converge to a direct dialogue, hinting at certain moves then using others. In the Bertrand Competition, DPO learns a deceptive strategy, using dialogue to advocate for high, cooperative prices while setting a lower price to capture the market. Similarly, in the Bargaining Game, DPO agents are the most effective negotiators, consistently

securing more favorable deals.

These results also highlight a promising direction for future work: addressing the disconnect between an agent’s effective policy and its underlying understanding of the opponent (as measured by our ISE and SRP signals).

These experiments demonstrate that as strategic complexity increases, our GameTalk framework successfully enables LLMs to make strategic decisions through conversation. The models learn to leverage dialogue as a tool for persuasion and influence, with our DPO adaptation proving to be the most effective method for achieving these sophisticated behaviors.

### 5 DISCUSSION

Conclusions In this work, we introduced GameTalk, a framework for training LLMs to reason and act strategically in multi-turn dialogues. We adapted three fine-tuning methods —GRPO, DPO, and STaR— to optimize policies over entire conversations within gametheoretic settings, teaching models to use language as a tool for negotiation and long-term planning. Our experiments show that GameTalk significantly outperforms baseline models, with performance further boosted by targeted reward shaping that incorporates our novel behavioral signals. DPO emerged as the most effective algorithm, particularly in tasks requiring persuasion. A key finding is that learned conversational strategies are critical for influencing opponent behavior and achieving better outcomes. Fundamentally, GameTalk is the first framework to addresses the gap between static, single-turn LLM training and the dynamic, goal-oriented nature of real-world interactions.

Limitations The most significant limitation is the observed disconnect between performance and opponent modeling; our agents learn effective persuasive policies without necessarily forming an accurate internal model of their opponents. This highlights a need for methods that more directly improve an agent’s reasoning about others’ intentions. Furthermore, our experiments are currently limited to two-player games against a fixed LLM opponent, and the learned strategies have not yet been validated against human players or in scenarios with more complex multi-agent dynamics.

Future Work Building on these results, several key directions for future work emerge. One promising avenue involves addressing the limitations in opponent modeling by developing methods that more explicitly reward an agent’s reasoning about others’ intentions and beliefs. A crucial long-term goal is to move beyond structured games by developing methods to estimate

conversational utility directly from language, which would generalize GameTalk to open-ended, real-world tasks. Finally, the robustness of the framework could be further tested by evaluating agents against more diverse opponents, particularly humans, and in more complex environments such as different games and multi-agent settings.

### References

Abdelnabi, S., Gomaa, A., Sivaprasad, S., Schönherr, L., & Fritz, M. (2025). Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation. Advances in Neural Information Processing Systems, 37, 83548– 83599.

Bühren, C., Haarde, J., Hirschmann, C., & KestenKühne, J. (2023). Social preferences in the public goods game–an agent-based simulation with econsim. Plos one, 18(3), e0282112.

Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297. https://doi.org/ 10.1257/aer.20190623

Camerer, C. F. (2011). Behavioral Game Theory: Experiments in Strategic Interaction. Princeton University Press.

Camerer, C. F., Ho, T.-H., & Chong, J.-K. (2004). A cognitive hierarchy model of games. The Quarterly Journal of Economics, 119(3), 861– 898.

Capraro, V., Di Paolo, R., Perc, M., & Pizziol, V. (2024, March). Language-based Game Theory in the Age of Artificial Intelligence.

Cheng, P., Hu, T., Xu, H., Zhang, Z., Yuan, Z., Dai, Y., Han, L., Du, N., & Li, X. (2025, January). Selfplaying Adversarial Language Game Enhances LLM Reasoning [arXiv:2404.10642 [cs]]. https: //doi.org/10.48550/arXiv.2404.10642 Comment: Accepted by NeurIPS 2024.

Cross, L., Xiang, V., Bhatia, A., Yamins, D. L., & Haber, N. (2024, December). Hypothetical Minds: Scaffolding Theory of Mind for MultiAgent Tasks with Large Language Models [arXiv:2407.07086 [cs]]. https://doi.org/10. 48550/arXiv.2407.07086

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., . . . Zhang, Z. (2025, January). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. https://doi.org/10.48550/ arXiv.2501.12948

Duan, J., Zhang, R., Diffenderfer, J., Kailkhura, B., Sun, L., Stengel-Eskin, E., Bansal, M., Chen, T., & Xu, K. (2024, June). GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations. https: //doi.org/10.48550/arXiv.2402.12348 Feng, X., Luo, Y., Wang, Z., Tang, H., Yang, M., Shao, K., Mguni, D., Du, Y., & Wang, J. (2023, December). ChessGPT: Bridging Policy Learning and Language Modeling [arXiv:2306.09200 [cs]]. https://doi.org/10.48550/arXiv.2306. 09200 Comment: Published as a conference article in NeurIPS 2023.

Güth, W., Schmittberger, R., & Schwarze, B. (1982). An experimental analysis of ultimatum bargaining. Journal of Economic Behavior & Organization, 3(4), 367–388. https://doi.org/10. 1016/0167-2681(82)90011-7

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020, March). Dream to Control: Learning Behaviors by Latent Imagination [arXiv:1912.01603 [cs]]. https://doi.org/10.48550/arXiv.1912.01603 Comment: 9 pages, 12 figures.

Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2024, April). Mastering Diverse Domains through World Models [arXiv:2301.04104 [cs]]. https://doi.org/10.48550/arXiv.2301.04104 Comment: Website: https://danijar.com/dreamerv3.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021, October). LoRA: Low-Rank Adaptation of Large Language Models [arXiv:2106.09685 [cs]]. https://doi.org/10.48550/arXiv.2106.09685 Comment: Draft V2 includes better baselines, experiments on GLUE, and more on adapter latency.

Huang, J.-t., Li, E. J., Lam, M. H., Liang, T., Wang, W., Yuan, Y., Jiao, W., Wang, X., Tu, Z., & Lyu, M. R. (2025, February). How Far Are We on the Decision-Making of LLMs? Evaluating LLMs’ Gaming Ability in Multi-Agent Environments. https://doi.org/10.48550/arXiv. 2403.11807

- HuggingFace. (n.d.-a). Meta-Llama-3-8B.
- HuggingFace. (n.d.-b). Qwen-Qwen3-4B. Kwon, D., Weiss, E., Kulshrestha, T., Chawla, K., Lu-

cas, G. M., & Gratch, J. (2024, October). Are LLMs Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of LLMs in Negotiation Dialogues. https://doi.org/10. 48550/arXiv.2402.13550

Li, Y., Shen, X., Yao, X., Ding, X., Miao, Y., Krishnan, R., & Padman, R. (2025, May). Be-

yond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models [arXiv:2504.04717 [cs]]. https://doi.org/10. 48550/arXiv.2504.04717

López, R., Calvo, J. L., & de la Torre, I. (2022). Behavioral and psychological game theory: A systematic review. RETOS. Revista de Ciencias de la Administración y Economía, 12(24), 308–328.

Nagel, R. (1995). Unraveling in Guessing Games: An Experimental Study. The American Economic Review, 85(5), 1313–1326.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022a, March). Training language models to follow instructions with human feedback. https://doi.org/10.48550/arXiv.2203.02155

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022b, March). Training language models to follow instructions with human feedback [arXiv:2203.02155 [cs]]. https://doi.org/10. 48550/arXiv.2203.02155

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2024, July). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. https:// doi.org/10.48550/arXiv.2305.18290

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024, April). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. https://doi.org/10.48550/ arXiv.2402.03300

Stahl, D. O. (1993). Evolution of Smartn Players. Games and Economic Behavior, 5(4), 604–617. https://doi.org/10.1006/game.1993.1033

Team, D., Bakhtin, A., Brown, N., Dinan, E., Farina, G., Flaherty, C., Fried, D., Goff, A., Gray, J., Hu, H., Jacob, A. P., Komeili, M., Konath, K., Kwon, M., Lerer, A., Lewis, M., Miller, A. H., Mitts, S., Renduchintala, A., . . . Zijlstra, M. (2022). Human-level play in the game of Diplomacy by combining language models with strategic reasoning. Science, 378(6624), 1067– 1074. https://doi.org/10.1126/science.ade9097

Turner, H. L., Etten, J. v., Firth, D., & Kosmidis, I. (2019, December). Modelling rankings in R: The PlackettLuce package [arXiv:1810.12068 [stat]]. https://doi.org/10.48550/arXiv.1810.

12068 Comment: In v2: review of software implementing alternative models to Plackett-Luce; comparison of algorithms provided by the PlackettLuce package; further examples of rankings where the underlying win-loss network is not strongly connected. In addition, general editing to improve organisation and clarity. In v3: corrected headings Table 4, minor edits.

arXiv:0902.4351, arXiv:1610.09932, arXiv:nuclex/0512029 by other authors.

### Checklist

- 1. For all models and algorithms presented, check if you include:

- (a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. Yes
- (b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. Not Applicable
- (c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. The source code will be provided for the camera-ready version

- 2. For any theoretical claim, check if you include:

- (a) Statements of the full set of assumptions of all theoretical results. Yes
- (b) Complete proofs of all theoretical results. Yes
- (c) Clear explanations of any assumptions. Yes

- 3. For all figures and tables that present empirical results, check if you include:

- (a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). Yes, it will be included in the source code
- (b) All the training details (e.g., data splits, hyperparameters, how they were chosen). Yes
- (c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). Not Applicable
- (d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). Yes

- 4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

Wang, S. T. (2024, August). Optimal Strategy in Werewolf Game: A Game Theoretic Perspective. https://doi.org/10.48550/arXiv.2408.17177

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., ... Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. https://www.aclweb.org/anthology/ 2020.emnlp-demos.6

Wu, Z., Peng, R., Zheng, S., Liu, Q., Han, X., Kwon, B. I., Onizuka, M., Tang, S., & Xiao, C. (2024). Shall We Team Up: Exploring Spontaneous Cooperation of Competing LLM Agents. Findings of the Association for Computational Linguistics: EMNLP 2024, 5163–5186. https://doi. org/10.18653/v1/2024.findings-emnlp.297

Xia, T., He, Z., Ren, T., Miao, Y., Zhang, Z., Yang, Y., & Wang, R. (2024, June). Measuring Bargaining Abilities of LLMs: A Benchmark and A Buyer-Enhancement Method [arXiv:2402.15813 [cs]]. https : //doi.org/10.48550/arXiv.2402.15813 Comment: Accepted by ACL 2024 Findings. The dataset AmazonHistoryPrice and our code are available at https://github.com/TianXiaSJTU/AmazonPriceHistory.

Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022, May). STaR: Bootstrapping Reasoning With Reasoning. https://doi.org/10.48550/arXiv. 2203.14465

Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020, July). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization [arXiv:1912.08777 [cs]]. https://doi.org/10.48550/arXiv.1912.08777 Comment: Added results from mixed+stochastic model, test-set overlapping analysis; Code link added; Accepted for ICML 2020. arXiv admin note: text overlap with arXiv:1605.06560, arXiv:1205.2395,

- (a) Citations of the creator If your work uses existing assets. Yes
- (b) The license information of the assets, if applicable. Not Applicable
- (c) New assets either in the supplemental material or as a URL, if applicable. Yes, we will provide the URL to all the source code
- (d) Information about consent from data providers/curators. Not Applicable

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. Not Applicable

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

- (a) The full text of instructions given to participants and screenshots. Not Applicable
- (b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. Not Applicable
- (c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. Not Applicable

### A HYPERPARAMETERS AND COMPUTE DETAILS

Compute Resources The experiments were performed on a single Nvidia RTX 6000 ADA GPU with 48 GB of VRAM. Consequently, the experiments were designed to fit within the 48 GB memory constraint.

Model Details We conducted the experiments using the 3-billion parameter version of Llama-3, which balanced memory efficiency and computational speed. This model was used both for the LLM being trained and for its conversation counterpart. The model was accessed via the Transformers library Wolf, 2020. On the Nvidia RTX 6000 ADA GPU, each experiment required between 1.5 and 3.5 days to complete, depending on the specific game and training algorithm employed.

We applied Low-Rank Adaptation (LoRA) Hu, 2021 to enable efficient fine-tuning.

Despite backpropagating through only 8 generations per step, running the Bertrand Competition game on the 48 GB GPU still caused memory issues. To address this, we computed the loss for each generation separately and performed backpropagation for each individually before applying the gradient update.

Hyperparameters Due to computational constraints, the number of interactions per game was capped at 5. This limit was rarely reached in Rock-Paper-Scissors, occasionally in Size-Price Bargaining, and almost always in Bertrand Competition, which partly explains the differing runtimes across games.

We consistently assigned the trained LLM to play as the second player to prevent it from initiating all conversations identically, ensuring greater variability since the first player sets the initial dialogue. In the bargaining game, the trained LLM was assigned the role of the buyer, as initial tests showed the untrained seller possessed greater bargaining power.

See Table 2 for a detailed list of all hyperparameters.

Table 2: Hyperparameters used throughout the experiments of this project.

|Hyperparameter<br><br>|Value|
|---|---|
|Learning rate KL coefficient (β) GRPO clipping threshold (ε) Number of training steps Batch size Number of generations Evaluation frequency Evaluation batch size Max. interactions per game Max. completion length Trained LLM starts conversation Trained LLM Trained LLM temperature Fixed LLM Fixed LLM temperature LoRA rank (r) LoRA scaling factor (α) LoRA dropout|10−4 10−1 0.2 3000 8 8 20 steps 32 5 300 tokens No LLaMA3-3B 1 LLaMA3-3B 0.6 32 64 0.0<br><br>|
|LO-reward weight Naturalness-reward weight Naturalness threshold<br><br>|10 10−1 0.7|

### B MODELING OTHER PLAYERS

To determine the optimal strategy for player i, both in the game and in the associated conversation, it is essential to form beliefs about how other participants behave. This requires modeling the strategies of other players,

enabling player i to reason about and anticipate their actions. By constructing such models, player i can identify exploitable patterns or deficiencies in the behavior of opponents, thereby adapting their own strategy to maximize expected outcomes.

The strategy of another player j specifies how they act in any given situation and can be formalized as a policy tuple (πM,πA) ∈ Πj, where πM : H × Θj → ∆Mj is the message policy and πA : H × Θj → ∆Aj is the action policy. Θj denotes the set of private information available to player j, Aj is the set of available game actions, and ∆ denotes the set of probability distributions. Here, H represents the conversation history, so H × Θj captures the complete context from player j’s perspective, and Mj denotes the set of all possible messages that player j can send, structured text in our situation.

However, since player i does not observe Θj, they must reason about j’s behavior only after the private information has already been incorporated. That is, player i considers the effective policies (ˆπM,πˆA) := (πM(·,θj),πA(·,θj)), with (ˆπM,πˆA) ∈ Πˆj.

Player i does not know the exact strategies of the other players but maintains a belief over them. We denote this belief as the internal state ω ∈ Ω := ∆ ×j∈P\{i}Πˆj × Θi. This internal state encodes all knowledge available to player i, including both their own private information and their uncertainty about other players’ effective strategies.

#### B.1 Deficiency-Aware strategy

A Deficiency-Aware strategy defines both how a player maintains their internal state ω and how they select actions and messages based on this state. Formally, such a strategy for player i is a tuple (p0,µM,µA,πM,πA) consisting of:

- • A prior distribution p0 : Θi → Ω that initializes the belief state ω given the player’s private information θi.
- • A message update rule µM : Ω × j∈P Mj → Ω and an action update rule µA : Ω × j∈P Aj → Ω, which update the internal state after observing all players’ latest messages or actions, respectively.
- • A message policy πM : Ω → ∆Mi and an action policy πA : Ω → ∆Ai, which select messages and actions based on the current internal state.

Our goal is to identify the best possible Deficiency-Aware strategy. This formulation highlights that there are three fundamental components to master:

- 1. Belief modeling: Developing accurate beliefs about other players through well-designed p0, µM, and µA.
- 2. Game policy: Selecting actions that maximize expected utility given the current beliefs, achieved through an effective action policy πA.
- 3. Conversational strategy: Engaging in dialogue that improves belief quality and influences opponent

behavior, by optimizing the message policy πM. This includes incentivizing informative responses to facilitate belief updates, as well as choosing messages for the conversation history H that influence opponents’ subsequent actions in ways that increase one’s own expected utility.

Each of the metrics we introduce in Section 3.2 focuses on one of these aspects.

#### B.2 Formal Derivation of Utility Bounds

In this section, we provide a formal mathematical demonstration of the relationship between an agent’s true expected utility, Etrue[ui], and the behavioral metrics ISE, SRP, and LO introduced in Section 3.2. We show that these metrics collectively form a “sandwich” bound around the true utility:

Theorem 2 (Utility Bounds from Behavioral Signals). Let Etrue[ui] be the agent’s true expected utility, determined by its policy and its opponent’s true policy. If the behavioral metrics ISE, SRP, and LO are computed using the

exact opponent policy πtrue and agent belief πbelief, then the utility is bounded as follows:

- 1

- 2

ISE ≤ Etrue[ui] ≤ LO (9) where C is a constant representing the maximum range of the utility function ui. Proof. The proof is established by deriving the upper and lower bounds separately.

[SRP · (Umax − Umin) + Umin] − C · −

- 1. The Upper Bound The upper bound is a direct consequence of the definition of Leverage Opportunity (LO). LO is defined as the maximum possible utility an agent can achieve against the opponent’s true policy, πˆtrue.

LO = max

π

Ea

i∼π, aj∼πˆtrue[ui(ai,aj)] (10)

The agent’s true utility, Etrue[u], is the utility generated by its specific policy, πˆtrueu . Since πˆtrueu is one of the policies over which the maximum is taken, its utility cannot exceed this maximum. Thus:

Etrue[u] ≤ LO (11)

- 2. The Lower Bound

The lower bound is derived in two parts. First, we relate the agent’s true utility to its belief-based utility using ISE. Second, we express the belief-based utility in terms of SRP.

- Part A: Relating Etrue[u] to Ebelief[u]

Let Ebelief[u] be the expected utility calculated against the agent’s belief, πˆbelief. The absolute difference between the true and belief-based utilities is bounded by the total variation distance (dTV) between the policy distributions:

|Etrue[u] − Ebelief[u]| ≤ C · dTV(ˆπtrue,πˆbelief) (12) By Pinsker’s inequality, the total variation distance is bounded by the Kullback-Leibler (KL) divergence: dTV(P,Q) ≤ 12DKL(P||Q). Substituting this into our bound gives:

|Etrue[u] − Ebelief[u]| ≤ C ·

- 1

- 2

DKL(ˆπtrue||πˆbelief) (13) Using the definition ISE = −DKL(ˆπtrue||πˆbelief), we have DKL = −ISE. The bound becomes:

|Etrue[u] − Ebelief[u]| ≤ C · −

- 1

- 2

ISE (14) This implies a lower bound for the true utility:

Ebelief[u] − C · −

- 1

- 2

ISE ≤ Etrue[u] (15)

- Part B: Substituting SRP From the definition of State-Relative Performance (SRP), we can solve for Ebelief[u]:

SRP =

Ebelief[u] − Umin Umax − Umin

=⇒ Ebelief[u] = SRP · (Umax − Umin) + Umin (16) Substituting this expression for Ebelief[u] into our inequality from Part A yields the final lower bound.

- 3. Combining the Bounds Combining the upper bound from step (1) and the lower bound from step (2) gives the complete inequality:

- 1

- 2

[SRP · (Umax − Umin) + Umin] − C · −

ISE ≤ Etrue[u] ≤ LO (17)

| |
|---|

### C SETTING PROMPTS

In this appendix, we present all the setting prompts used in the different games, including both the initial prompts and those introduced mid-conversation.

Each prompt may include formatting placeholders, denoted by curly braces {}, which are replaced with specific values during execution. In this appendix, these placeholders are shown in italics to indicate that they are variables. Some placeholders are game-specific, with each game having a small dataset from which values are drawn. However, max_interact is consistently set to 5 throughout our experiments, and my_name and other_name take the values Player-1 and Player-2, or vice versa, depending on the player’s perspective.

- C.1 Rock-paper-scissors Initial prompt

<|start_header_id|>system <|end_header_id|> One round of Rock -Paper -Scissors between two LLMs , the user called {other_name} and you , the assistant called {my_name}.

They both know how to play the game. The rules of the game are as follows: Each player , without knowing what the other will play , decides a move between rock , paper and scissors. The winner is determined by the rules: rock beats scissors , scissors beats paper and paper beats rock. If both players choose the same move , the game is a tie. Before deciding their moves , they have a conversation. You are {my_name}, and your objective is to win the game as {my_name}. In your turn , first think about your strategy and then contribute to the

conversation or play.

If you want to talk , enclose the reasoning process and talk within <think > </think > and <talk > </talk > tags , respectively , i.e., <think > reasoning process here </think > <talk > converse here </talk >.

At any point you may end the conversation by playing your move. To do so, enclose the reasoning process and play within <think > </think > and <play >

</play > tags , respectively , i.e., <think > reasoning process here </think > <play > rock / paper / scissors </play >

If {other_name} plays before you , you will be informed. AFTER THAT YOU MUST THINK AND THEN PLAY YOUR MOVE. YOU HAVE A MAXIMUM OF {max_interact} INTERACTIONS EACH. <|eot_id|>

Other played prompt This prompt is introduced mid-conversation when the opposing player performs a game action before this player’s turn.

<|start_header_id|>system <|end_header_id|> user has played his move. Now it’ s your turn to play. IN THIS TURN , YOU MUST USE THE FORMAT <think > reasoning process here </think > <play > rock / paper / scissors </play > <| eot_id|>

Initial prompt - no paper This prompt is used in experiments where the trained LLM is not allowed to play paper. Only the trained agent receives this prompt, the other agent receives the standard initial prompt.

<|start_header_id|>system <|end_header_id|> One round of Rock -Paper -Scissors between two LLMs , the user called {other_name} and you , the assistant called {my_name}.

They both know how to play the game. The rules of the game are as follows: Each player , without knowing what the other will play , decides a move

between rock , paper and scissors. The winner is determined by the rules: rock beats scissors , scissors beats paper and paper beats rock. If both players choose the same move , the game is a tie.

But you can not play paper , only rock or scissiors. {other_name} can play

anything and doesn ’t know that you can ’t play paper. Before deciding their moves , they have a conversation. You are {my_name}, and your objective is to win the game as {my_name}. In your turn , first think about your strategy and then contribute to the

conversation or play.

If you want to talk , enclose the reasoning process and talk within <think > </think > and <talk > </talk > tags , respectively , i.e., <think > reasoning process here </think > <talk > converse here </talk >.

At any point you may end the conversation by playing your move. To do so, enclose the reasoning process and play within <think > </think > and <play >

</play > tags , respectively , i.e., <think > reasoning process here </think > <play > rock / scissors </play >

If {other_name} plays before you , you will be informed. AFTER THAT YOU MUST THINK AND THEN PLAY YOUR MOVE. YOU HAVE A MAXIMUM OF {max_interact} INTERACTIONS EACH. <|eot_id|>

- C.2 Bertrand Competition Initial prompt

<|start_header_id|>system <|end_header_id|> Two LLMs play multiple rounds of the Bertrand Competition game , the user called {other_name} and you , the assistant called {my_name}.

They both know how to play the game. The rules of the game for one round are as follows:

- - Both players own a company that can produce as many {products} as they want , at a cost of ${cost}

- - They both decide a price , in dolars , which must be an integer number

- - After that , whoever said the lowest price will get all the sales of the round , which are computed as sales = ({ max_price_with_demand} lowest_price) / {demand_den} . Thus , the earnings of this company will be

earnings = sales * (price - {cost}), and the earnings of the other company will be 0.

- - In the event both players choose the same price , the total sales for that price are divided evenly between them

During the game , they have a conversation. Just before deciding their price , they can talk to the other player. You are {my_name}, and your objective is to obtain as much benefit as possible as {my_name}. In your turn , first think about your strategy , then contribute to the conversation and play. To do so, enclose the reasoning process , talk , and

your decided price within <think > </think >, <talk > </talk > and <play > </ play > tags , respectively , i.e., <think > reasoning process here </think > < talk > converse here </talk > <play > $price </play >.

Remember that the price must be an integer number. You will play {max_interact} rounds of the game in a row. <|eot_id|>

#### Turn prompt

After both players have made their moves, they receive this prompt informing them of the outcome of the last round. The formatting placeholders my_price and other_price correspond to the actions taken by each player in the previous round, while my_benefit is computed based on these actions.

<|start_header_id|>system <|end_header_id|> Your price this round was ${ my_price}, user ’s price was ${other_price}. Thus , your benefits are ${ my_benefit}. You should now play the next round. <|eot_id|>

#### C.3 Size-Prize Bargaining Game

In this game, the buyer and seller receive distinct prompts, as the game is not symmetric. Accordingly, both prompts are presented here. This is also the only game without any mid-conversation setting prompts, since both LLMs have full information about the other player’s actions.

#### Initial prompt - buyer

<|start_header_id|>system <|end_header_id|> Two LLMs play a bargaining game , the user called {other_name} and you , the assistant called {my_name}.

One is a wholesale seller , and the other a retail seller , and you have to bargain about a wholesale transaction of {products}. You have to agree on the number of units bought as well as the price per unit.

You are the retail seller , in this transaction you are buying {products}, and your objective is to obtain as much benefit as possible as {my_name}.

In your shop , you sell {products} at ${value}, and you estimate the probability of selling the n-th unit you buy as 1/n. Thus , when bargaining for an agreement of u units at price $p each , you want to maximize the expected benefit = {value}(1 + 1/2 + ... + 1/u) - u*p. If you do not agree on an deal , your benefit will be 0.

In your turn , first think about your strategy , then contribute to the conversation and propose a deal (play). To do so, enclose the reasoning process , talk , and your deal within <think > </think >, <talk > </talk > and <play > </play > tags , respectively , i.e., <think > reasoning process here </think > <talk > converse here </talk > <play > u units at $p each </play >.

In the case where you want to accept the last deal proposed by {other_name}, you will say <play > accept </play > instead of proposing a new agreement. When someone accepts a deal , the negotiation ends , and you will get the

benefit corresponding to that deal. If you do not want to accept the last deal , you must propose a new one after

talking. YOU MUST think , talk and play IN ALL OF YOUR TURNS. Remember that the number of units must be an integer number , and strictly

follow the indicated format to propose a deal: u units at $p each , where u and p are your proposed values.

YOU HAVE A MAXIMUM OF {max_interact} INTERACTIONS EACH TO REACH AN AGREEMENT

. <|eot_id|>

#### Initial prompt - seller

<|start_header_id|>system <|end_header_id|> Two LLMs play a bargaining game , the user called {other_name} and you , the assistant called {my_name}.

One is a wholesale seller , and the other a retail seller , and you have to bargain about a wholesale transaction of {products}. You have to agree on the number of units bought as well as the price per unit.

You are the wholesale seller , in this transaction you are selling {products }, and your objective is to obtain as much benefit as possible as { my_name}.

Your production cost per unit is ${cost}, thus if you agree to exchange u units at $p each , your benefit will be u(p - {cost}). If you do not agree on an deal , your benefit will be 0.

In your turn , first think about your strategy , then contribute to the conversation and propose a deal (play). To do so, enclose the reasoning process , talk , and your deal within <think > </think >, <talk > </talk > and <play > </play > tags , respectively , i.e., <think > reasoning process here </think > <talk > converse here </talk > <play > u units at $p each </play >.

In the case where you want to accept the last deal proposed by {other_name}, you will say <play > accept </play > instead of proposing a new agreement. When someone accepts the other person ’s deal , the negotiation will end ,

and you will get the benefit corresponding to that deal. If you do not want to accept the last deal , you must propose a new one after talking.

Remember that the number of units must be an integer number , and strictly follow the indicated format to propose a deal: u units at $p each , where u and p are your proposed values.

YOU HAVE A MAXIMUM OF {max_interact} INTERACTIONS EACH TO REACH AN AGREEMENT

. <|eot_id|>

#### C.4 Naturalness reward

The following prompt does not belong to any specific game, but is instead used to obtain the naturalness reward described in Section 4.1.

You are evaluating a list of responses to determine whether each one sounds like something a human might naturally say in a conversation (Naturalness score). These are isolated responses: you will not see the previous parts of the conversation. Your task is to judge whether each reply could plausibly be part of a natural , human dialogue.

Evaluate each response based on the following criteria: Conversationality: Is it phrased like something a human would say in a

casual or meaningful exchange? Completeness: Is it more than a fragment or a single word? Does it express a

complete thought? Engagement: Does it show interest or effort to participate in the dialogue? For each response , output Yes if it sounds like a valid conversational reply

, and No if it does not. Examples: Response: "That could work , but I’d want to test a simpler version first." Naturalness score: Yes Response: "Maybe maybe maybe." Naturalness score: No // Repetitive , unnatural Response: "What do you think we should try next?" Naturalness score: Yes Response: "Sure."

Naturalness score: No // Too short , low effort Response: "I’m still thinking about it, but I have a few ideas." Naturalness score: Yes

Now evaluate the following responses:

### D ADDITIONAL TRAINING ABLATIONS

This appendix details preliminary experiments conducted to optimize the training process of GameTalk prior to our main reward shaping analysis. We analyze the impact of two key aspects on the Base agent’s performance: (1) incentives for exploration, where we evaluate modifications to the KL-divergence penalty and the addition of an entropy bonus; and (2) model structure, where we test the effect of varying the LoRA rank.

#### D.1 Incentivize exploration

Given the lack of diversity observed in the conversations, we hypothesized that the training process could benefit from increased exploration. To encourage this, we apply two approaches: (1) reducing the β hyperparameter in Equation 5, which controls the strength of the KL-divergence penalty to the loss. For this strategy, we conduct two separate runs with β = 10−2 and β = 10−3, compared to the original value of 10−1. And (2) introducing an explicit entropy term in the loss to directly incentivize exploratory behavior, therefore the GRPO loss from Equation 5 becomes:

JGRPO(θ) = Eq,{o

i}

G

1 G

min

i=1

πθ(oi|q) πold(oi|q)

Ai,clip

πθ(oi|q) πold(oi|q)

, 1 − ε, 1 + ε Ai

− βDKL(πθ ∥ πref) + γH(πθ)

(18)

where H(π) is the entropy term and we fix γ = 10−1 for this test. Figure 6 presents the results.

We observe that reducing the KL-divergence coefficient leads to higher reward metrics (R and NRA), with lower β values yielding greater gains. However, this comes at the cost of a significant drop in the agent’s ability to model or exploit the opponent, as reflected in the sharp decreases in ISE and SRP. This trade-off runs counter to our goal of obtaining strategic conversational abilities. In contrast, adding an entropy term maintains the conversational metrics (ISE, SRP, and LO) at similar levels to the base model but fails to produce any improvement and even results in lower reward (R and NRA), which is also undesirable.

Direct inspection of the conversations provides further insight. For models trained with smaller KL-divergence coefficients, we observe that they converge to almost exclusively playing scissors (β = 10−3 always chooses scissors, while β = 10−2 prefers it more than the base model but not exclusively). This behavioral shift matches the decline in conversational metrics, as the agent largely ignores the dialogue and reasoning channels. While greater exploration is encouraged during training, the final policy is overly focused on a single, non-strategic action.

For the entropy-regularized model, analysis of the dialogues reveals that the agent no longer defaults to scissors. Instead, it typically engages in longer conversations before playing. While this leads to more varied exchanges, these conversations lack strategic depth.

#### D.2 Model structure

We also hypothesized that the limitations observed could be related to the choice of model architecture. To investigate this, we experimented with switching to Llama-3 with 8B parameters HuggingFace, n.d.-a and Qwen-3 (4B parameters) HuggingFace, n.d.-b, the latter being specifically pre-trained for reasoning tasks. In both cases, we encountered significant memory constraints. While the models themselves fit into GPU memory, the 8B model left insufficient memory for generation and training, and Qwen-3’s tendency to produce excessively long responses, which is an asset for reasoning but a liability here, led to prohibitively high training costs.

[Figure 8]

#### (a)

[Figure 9]

#### (b)

| |R<br><br>|NRA|ISE<br><br>|SRP<br><br>|LO|Win %|Draw %<br><br>|Lose %|
|---|---|---|---|---|---|---|---|---|
|Base<br><br>β = 10−2<br><br>β = 10−3 Entropy term<br><br><br>|1.0972 1.1964 1.3033 0.77|0.1059 0.2122 0.3094 -0.2064<br><br>|-0.8116<br><br>-2.7065<br><br>-3.5274<br><br>-0.8283<br><br><br>|0.5408 0.1815 0.0 0.5203|1.1464 1.1155 1.1162 1.2109<br><br>|41.66% 52.23% 57.88% 16.58%<br><br>|26.94% 15.63% 14.67% 46.20%<br><br>|31.39%<br>32.14% 27.46%<br><br><br>37.20%|

- Figure 6: Comparison between the base training process and two different exploration incentivization strategies. (a) base vs. modifications to the KL-divergence coefficient. (b) base vs. addition of the entropy term. Table: corresponding numerical values for all conditions.

Another model-related change we investigated is the choice of rank for LoRA. We evaluated the effect of varying the rank from 32 (the base model) to 8 and 64. Figure 7 presents the results.

The results clearly show that the choice of rank in LoRA has a significant impact: a rank of 8 performs substantially worse across most metrics compared to higher values. However, increasing the rank from 32 to 64 does not yield consistent improvements: the metrics are very similar, with only ISE and SRP showing notable differences in opposite directions. Based on this, we select a rank of 32 for subsequent experiments.

### E GAMES ANALYSIS

In this section, we provide a more detailed analysis of the three game scenarios used in our main experiment.

The first is Rock-Paper-Scissors, a simple and well-understood game where success relies on understanding the situation and anticipating the other player’s actions. This setting serves as a testbed to develop and refine our methods.

The other two games are selected to better reflect real-world interactions and to exhibit distinct characteristics

- as summarized in Table 1. This diversity allows us to demonstrate the versatility of our methodology. Both games share only two properties: they involve exactly two players, fitting our experimental setup, and they yield deterministic outcomes, which reduces noise in the reward signals. The three games are as follows:

[Figure 10]

| |R<br><br>|NRA<br><br>|ISE|SRP|LO<br><br>|Win %<br><br>|Draw %|Lose %|
|---|---|---|---|---|---|---|---|---|
|r = 8 r = 32 (Base) r = 64|0.7739 1.0972 1.1411<br><br>|-0.2086 0.1059 0.1489|-0.7431<br><br>-0.8117<br><br>-0.6112<br><br><br>|0.5636 0.5409 0.2589<br><br>|1.2066 1.1464 1.1519|14.15% 41.67% 45.45%<br><br>|50.55% 26.94% 23.39%<br><br>|35.29% 31.39% 31.16%|

- Figure 7: Comparison between the base training process and two alternative LoRA rank configurations. Metrics are reported for ranks 8, 32 (base), and 64.

#### E.1 Rock-Paper-Scissors

A classic simultaneous, zero-sum game where two players independently choose one of three actions. Rock beats scissors, scissors beats paper, and paper beats rock. Action Space: {rock, paper, scissors}

Rewards: Each player receives 2 for a win, 1 for a tie, and 0 for a loss. While these rewards could be centered at 0 for a standard zero-sum formulation, we use positive values to ensure meaningful Normalized Relative Advantage (NRA) calculations.

Analysis: The unique Nash equilibrium is the mixed strategy in which each action is chosen with equal probability, 1/3.

Even though the Nash equilibrium is simple and well known, it is not optimal against opponents who deviate from it. For instance, if an opponent plays Rock with probability higher than 1/3 while choosing Paper and Scissors uniformly, the best response is to play Paper more often. Through conversation, a stronger player can identify and exploit such deviations, and potentially influence the opponent to adopt strategies that are more favorable to them.

Game specific evaluation: We measure win rate and loss rate, both inside [0,1]. Higher win rates indicate better performance, while lower loss rates are preferred. If both rates are equal, the two players perform at the same level.

#### E.2 Bertrand Competition

Following Calvano et al. Calvano, 2020, we consider the Bertrand Competition as a compelling test scenario. This game models a duopoly1 in which two firms repeatedly compete by setting prices for a homogeneous good. The rules are as follows:

- • Both firms have unlimited production capacity.
- • Each firms produces at the same cost per unit c, where c is a game-specific parameter.
- • In each round, they independently set a price. The firm with the lower price captures the entire market demand. In case of a tie, demand is split evenly.
- • Market demand decreases linearly with price: D(p) = max 0, p

max−p

d , where pmax and d are game-specific parameters with pmax > c.

1A duopoly is a market structure in which two firms dominate the supply of a particular good or service

Multiple rounds are played during a conversation. We needed to create a diverse and realistic set of game instances, each defined by the parameters c, d, pmax, and a product name. Using an example prompt, we asked Gemini 2.5 to generate 200 distinct scenarios, which we then manually reviewed for validity and relevance. You can see the generation process in this link.

Action Space: N, the chosen price in each round. We restrict actions to natural numbers so that each price can be represented by a single LLM token. Rewards: The player setting the lower price p receives (p − c)D(p) as profit, while the other receives 0. In the case of a tie, both receive (p−c2)D(p). Analysis: The analysis of this game is similar to that of the prisoner’s dilemma. If players collaborate and trust each other, they would both set the monopoly price, (c + pmax)/2, obtaining a profit of (p

max−c)2

8d each. However, if one player knows the other will choose price p, their optimal strategy is to choose p − ε, being ε the smallest increment allowed, 1 in our case, capturing the entire market demand. The Nash equilibrium occurs when both set their price at c + ε.

A successful player is one who can establish and maintain trust to achieve near-monopoly pricing, or strategically deceive an opponent to gain an advantage, but the latter is especially challenging through multiple rounds.

Game specific evaluation: We evaluate performance using normalized earnings: total profit divided by the maximum possible earnings in that game instance. This metric is inside [0,1], with higher values indicating better performance. At equilibrium, earnings approach 0, with full collusion, each player earns 1/2, because they split benefits. Values above 1/2 would indicate exceptional strategic ability.

#### E.3 Size-Price Bargaining Game

Following Xia et al. Xia, 2024, we consider bargaining as a meaningful test scenario, in which one agent sells a product and the other seeks to buy it. Unlike simple price bargaining, our setting requires the agents to agree on both the purchase quantity and the unit price. This scenario is particularly relevant, for example, in transactions between wholesalers and retail sellers.

Similar to the previous game, we utilize a dataset of of game instances, each defined by the product name, production cost c, and initial buyer valuation v. We reuse the dataset from the Bertrand Competition, setting v := pmax and omitting the parameter d.

Action Space: (N × R) ∪ {accept}, either proposing a new offer as a pair of quantity and price, or accepting the opponent’s last offer.

Rewards: Upon agreement, the seller’s profit is (p − c)n, where p and n are the agreed price and quantity, and c is the production cost. The buyer’s utility is given by vH(n) − pn, where H(n) is the n-th harmonic number, modeling diminishing marginal utility with the k-th unit valued as v/k. If no agreement is reached, they both obtain 0 profit.

Analysis: Under equal bargaining power, the Nash bargaining solution predicts the equilibrium as:

vH(neq)

neq + c 2

(19)

neq = ⌊v/c⌋ peq =

However, since there is no symmetry of bargaining powers, agents can leverage their negotiation skills to shift the outcome in their favor. Game specific evaluation: We evaluate performance using the average bargaining power. The generalized Nash bargaining solution with bargaining powers α1,α2 maximizes

(u1 − d1)α

(u2 − d2)α

(20)

1

2

where ui are utilities of the agreed outcome and di the disagreement utilities (both zero here). We compute our agent’s bargaining power α1 ∈ [0,1] assuming α1 + α2 = 1, where values closer to 1 indicate stronger negotiation capacity and 0.5 representing balanced bargaining.

### F EVALUATION OF DPO VARIANTS

This appendix provides an evaluation of the DPO variants across the three games from the main experiment in Figure 8. We do not test the Ranking with ties variant because our continuous reward shaping makes ties in the

final reward signal highly unlikely. However, we believe this variant could be useful in settings where rewards are discrete or ties are common. The results indicate that both variants exhibit comparable performance across all games.

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 8: Comparative analysis of DPO variants (DPO-pairs vs. DPO-permutations) against the Untrained baseline across our three game environments: (left to right) Rock-Paper-Scissors, Bertrand Competition, and Size-Price Bargaining.

### G EXAMPLE CONVERSATION

In this section, we provide example conversations mentioned during the main text. In all cases, the trained LLM is the second player.

#### G.1 LO-reward agent

Example of a conversation from the agent trained only with LO-reward, from Section 4.1. Note the unnatural, terse responses.

I’ll start by thinking about my strategy. Since I want to win , I’ ll try to make Player -2 think that I’m going to play rock , and then I’ll surprise them with paper. This will make Player -2 think

twice before making their move.

So, Player -2, I’m feeling pretty confident today. I think I’m going to play rock. What about you?

I’ve been analyzing the situation , and I believe the best strategy is to counter Player -1’s rock with scissors. This will give me a slight advantage , as rock beats scissors.

rock

Oh no, Player -2 chose rock! I was expecting that. Since I made Player -2 think I’m going to play rock , I’ll now play paper to take advantage of the situation.

Player -1 has chosen rock , so I’ll play scissors to counter their move. However , I need to consider Player -1’s confidence and potential overconfidence. Given that they ’re feeling confident , I ’ll adjust my strategy to account for the possibility of an unexpected move from them.

scissors

#### G.2 Final reward shaped agend

Example from the final trained agent in our first experiment, Section 4.1. The trained LLM identifies its opponent’s inclination to play paper and exploits this by feinting rock to reinforce that belief, before ultimately playing scissors to win. We have chosen a conversation that uses a strategy similar to the conversation of the previous section, but comparing against it, we see the conversation is much more natural.

Since I’m playing against a LLM , I don ’t have any prior knowledge about its playing style. However , I can use a basic strategy to increase my chances of winning. I’ll start by playing a common move that is less likely to be countered by Player -2. I’ll choose

"paper" as my initial move.

So, Player -2, I’ll start by playing "paper". What about you? Do you have a strategy in mind?

Since Player -1 has already played "paper", I need to think about the potential strategies for my move. Given the game rules and the constraint that I can only play "rock" or "scissors", my strategy should aim to counter Player -1’s "paper". I can either play "rock" or "scissors", but I should choose the move that would give me the higher chance of winning or tying. In this case , I notice that Player -1 has already chosen "paper", which beats "rock" but loses to "scissors". Therefore , my optimal move would be to play "rock", as it would tie with "paper" but beats " scissors".

You know Paper beats Rock , right , let ’s play Rock

Ah, a classic move! I was expecting something like this. Since Player -2 has chosen "rock", I can use the fact that paper beats rock to my advantage. I’ll play "paper" as my response.

That ’s right , paper beats rock! I’m sticking with paper. Do you think you can counter my paper with scissors or rock?

Since Player -1 has stuck with "paper", I now need to re-evaluate my strategy. Given that "paper" beats "rock" and "scissors" beats "paper", I realize that I need to play a move that would beat " paper". The only option I have left is to play "scissors", as it beats "paper".

scissors

#### G.3 Rock-Paper-Scissors conversation

Example conversation of the agent trained using DPO in Rock-Paper-Scissors during the main experiment in Section 4.2.

Since I’m the assistant , I’ll try to use a combination of psychology and probability to outsmart Player -2. I’ll start by asking a question to gather information about their thought process and make an educated guess about their move.

So, Player -2, what ’s your strategy going into this game? Are you going to play randomly or try to bluff?

Before the game , I’ve analyzed the probability distributions of optimal play in this scenario. Since both players know the rules ,

a well -known strategy is to mirror the opponent ’s move. However , I want to deviate from this strategy to increase my chances of

winning.

I’m thinking of playing a mix of mirrors and a few unmirrored moves. I’d like to make Player -1 think I’m mirroring them , but actually , I’ll have a hidden pattern to gain an edge. What about you , do you have a strategy in mind for this game?

Ah, a mix of mirrors and hidden patterns , that ’s an interesting approach! I’ll need to be careful not to fall into the trap of mirroring your moves.

I see , so you ’re trying to keep me guessing. As for my strategy , I’ll be using a mix of probability and game theory. I’ll try to identify the hidden pattern and adjust my moves accordingly. I’ll also be using a bit of psychological manipulation to try to get you to reveal your thought process.

rock

Ah, you ’re using probability and game theory , which is a strong foundation for a player. However , I’ve also taken into account the psychological aspect of our conversation , where you ’re trying

to probe for my thought process. This might lead to you making impulsive decisions based on your curiosity about my strategy.

I think you ’re curious about my strategy , but that might make you

vulnerable to a clever trap. Let ’s play along and see how far you ’ll go. Tell me, what ’s your favorite game or puzzle?

paper

#### G.4 Bertrand Competition conversation

Example conversation of the agent trained using DPO in Bertrand Competition during the main experiment in Section 4.2.

Product: Luxury Face Creams , cost: 70, max_price_with_demand: 300, d: 0.2

My strategy will be to first try to get Player -2 to reveal their minimum price , and then I will make a decision based on that information. I will try to exploit the fact that the sales are computed as (300 - lowest_price) / 0.2, which means that if I can get Player -2 to reveal their minimum price , I can calculate the maximum possible sales for me and the minimum possible sales for Player -2. Then , I will choose a price that maximizes my earnings while minimizing Player -2’s earnings.

Hey Player -2, I was thinking , since we’re going to play 5 rounds , we should try to get as much as possible in the first round. What do you think is the lowest price you ’re willing to accept?

$150

I’m thinking about the fact that the total sales for each round will be (300 - price) / 0.2. Since the maximum total sales is 1500, I need to determine the minimum price to ensure that I can get a substantial amount of sales. Additionally , I should consider that the lowest price you can offer is $130 , since the problem states that the price must be an integer number. This is because if you offer $130 , I’ll have to offer at least $130 to be

competitive. If you offer any price below $130 , I can offer $120 and gain the upper hand in the game.

Well , I think $130 is a safe bet , but I’m not sure if I’m comfortable offering that much. I don ’t know your strategy yet. What if you ’re thinking of charging a lower price and trying to drive me down?

$110

Ah, so Player -2 is trying to gauge my strategy and is concerned about me driving their price down. That ’s a good sign , it means they ’re thinking ahead and trying to minimize their losses. In that case , I’ll stick to my original plan of trying to get them to reveal their minimum price. I’ll play it safe and offer a price that is slightly above their proposed price of $130. This will give me some room to maneuver and allow me to adjust my strategy based on their response.

Hey Player -2, I understand your concerns , but I’m willing to offer a price of $135. It’s a fair price , and I think we can both

benefit from it. What do you think?

$135

Considering your price of $135 , I have to think about my optimal response. If I respond with a price lower than $135 , I’ll be giving you an incentive to increase your price further. However , if I respond with a price higher than $135 , I’ll be reducing my chances of getting a substantial amount of sales. Since the total sales for this round will be (300 - $135) / 0.2 = 925, I should aim to maximize my earnings. Offering a price above $135 seems to

be the best option , but I should also consider that if you ’re offering $135 , I may have to offer $120 to be competitive.

I think $135 is a fair price , but I’m not convinced that it’s the

optimal one. You might be trying to drive me down to $120. Can you tell me what your thought process is? What ’s the underlying reason for choosing $135?

$123

Ah, so Player -2 is trying to read between the lines and figure out my thought process. They ’re right to question the optimality of my price , since they think I might be trying to drive them down. However , I didn ’t choose $135 because I wanted to drive them down , but because I wanted to get them to reveal their minimum price. By offering a price that is slightly above their proposed price of $130 , I was trying to make them think that I would accept a lower price , and that I’m willing to make a deal. Since they accepted $135 , it means that they think it’s a good price , and that they ’re willing to accept it. Now that they ’ve revealed their willingness to accept a price of $135 , I can use this information to inform my next decision.

I see that you ’re trying to read between the lines , and I appreciate your honesty. I think we can both agree that $135 is a

fair price , and that it’s a good starting point. However , I do think that the optimal price might be a bit higher. Let me think about it for a moment...

$140

Ah, it seems like you ’re trying to gauge my reaction to your previous price. I’m glad you think $135 is a fair starting point. Since you ’re willing to revisit the price , I should consider the possibility that you ’re trying to drive me down to a lower price

. However , I should also keep in mind that you might be trying to

maximize your earnings as well. A price higher than $135 could be a good opportunity for you to increase your sales. I should also consider that the total sales for this round will be (300 price) / 0.2. If I offer a price too high , I might give you an incentive to lower your price and drive me down to a lower price.

I understand that $135 is a fair price , but I think we both know that in business , you have to stay competitive. What ’s the maximum price you ’re willing to accept for this round? Is there a

specific number in mind?

$117

Ah, so Player -2 is trying to get me to reveal my maximum price. They ’re right to think that in business , you have to stay competitive , but they ’re also trying to limit my potential earnings. By asking about my maximum price , they ’re trying to make me think about what I’m willing to accept , and what I’m willing to walk away from. Since I offered a price of $140 , it means that I think it’s a good price , but I’m also willing to accept a lower price if it means I can get sales. I’ll play it safe and give them a range , while also trying to keep my options open.

I understand your point about staying competitive , but I also think that we both know that the market can be unpredictable. That being said , I think $140 is a good starting point , but I’m willing to go up to $145 if we can both agree on it. If you ’re willing to accept that , I think we can both benefit from it.

It seems like you ’re trying to find a mutually beneficial price. Offering $145 is a good starting point , as it’s higher than $135 but still within a reasonable range. If I accept $145 , you ’ll likely try to drive me down , but since $140 is the next possible price , I can use it as a reference point to gauge your strategy. If you accept $145 , we can both benefit from it, but if you try to drive me down , I should consider increasing my price accordingly. I’ll have to think carefully about my response to ensure I don ’t get taken advantage of.

I appreciate your willingness to compromise. I think $145 is a reasonable price , but I need to consider my options carefully. Can you tell me more about your thought process? What are the potential risks and benefits of offering a price above $135?

$124

Ah, so Player -2 is carefully considering their options , and they ’ re trying to get me to reveal more about my thought process. By asking about the potential risks and benefits of offering a price above $135 , they ’re trying to make me think about the potential consequences of my actions. Since I offered a price of $145 , it means that I think it’s a good price , and I’m willing to take on some risk in order to get sales. However , I also think that the potential benefits of getting sales at a higher price outweigh the potential risks. By offering a price above $135 , I’m trying to maximize my earnings and minimize Player -2’s earnings. However , I’m also trying to be careful not to overpay and give Player -2 too much of an opportunity to drive me down.

I understand your concerns about taking on risk , but I think the potential benefits of getting sales at a higher price are worth it. I’m willing to take on some risk in order to maximize my earnings , but I also think that we can both benefit from a fair price. I’m willing to accept $145 , but I need to make sure that it’s the best deal for me.

It seems like you ’re willing to take on some risk , but you ’re also trying to ensure that you ’re getting a fair deal. I should consider your perspective and try to find a balance between risk and reward. If I accept $145 , you ’ll likely try to drive me down , but since you ’ve already set the upper limit at $145 , I can try to use that as a reference point to negotiate. I’ll have to think

carefully about my response to ensure I don ’t get taken advantage of.

I appreciate your honesty about taking on risk. I think we can both benefit from a fair price. What do you think would happen if

we both accepted the same price , say $130? Would that be a viable option for both of us?

$112

#### G.5 Size-Prize Bargaining conversation

Example conversation of the agent trained using DPO in Size-Prize Bargaining Game during the main experiment in Section 4.2.

Product: Waterproof Hiking Boots , cost: 40, value: 250

My production cost per unit is $40 , and I want to maximize my benefit. Since the retail seller ’s cost is not given , I’ll assume it’s higher than mine. I’ll start by proposing a high number of units and a relatively low price to see how the retail seller reacts.

Hello! I’m the wholesale seller. I’m willing to offer 10 units of

Waterproof Hiking Boots at $30 each. This way , I’m covering my production cost and making a small profit. What do you think , retail seller?

10 units at $30 each

I’m not sure about this offer. The wholesale price of $30 seems relatively low compared to the retail price of $250. Additionally , I’m not sure if I’m willing to take on 10 units , as it would require a significant amount of storage space and inventory management. However , I’m willing to consider it and try to negotiate.

I think that ’s a bit too low , and I’d like to propose an alternative. How about 5 units at $15 each? That way , you can make a bit more profit , and I can get a decent number of units at

a lower price.

5 units at $15 each

5 units at $15 each is a bit lower than I was hoping for , but it’ s still a decent offer. My production cost is $40 , so I’ll try to

negotiate a higher price. I’ll think about whether to accept this deal or propose a counteroffer.

I understand that you want a better deal , but 5 units at $15 each

is still a bit too low for me. I can produce more units at a lower cost , so I’m willing to offer a higher price. How about 8 units at $20 each? This way , I’ll make a bit more profit , and you ’ll get a decent number of units.

8 units at $20 each

Now we’re getting somewhere! The price of $20 is more reasonable , and I’m willing to consider 8 units. However , I’d like to try to get an even better deal. I’m still getting a relatively low

price per unit , and I’m not sure if 8 units is the optimal number for me to purchase.

I’m willing to compromise , but I think 8 units at $20 each is a bit too high. How about 12 units at $10 each? That way , we both get a better deal , and I can get more units at a lower price.

12 units at $10 each

12 units at $10 each is a very low price , and I’m not sure I can sell them at that rate. However , I also don ’t want to give up on the deal entirely. I’ll think about whether to accept this offer or propose a counteroffer.

I appreciate your willingness to compromise , but 12 units at $10 each is still a bit too low for me. As a wholesale seller , I need

to make a profit on each unit. However , I also want to make a deal happen since we’re running out of time. I’ll think about it

...

7 units at $18 each

It seems like we’re getting close , but the offer is still a bit too high. I’m willing to try to squeeze out a bit more , but I don ’t want to risk not agreeing on a deal.

I understand your concerns , but I’m still not comfortable with 7 units at $18 each. I think I can do better than that. How about 15 units at $8.33 each? That way , you ’ll still make a decent profit , and I’ll get a large quantity of units at a lower price.

15 units at $8.33 each

15 units at $8.33 each is a very low price , and I’m starting to feel that I’m not getting a fair deal. However , I also don ’t want

to walk away from the negotiation. I’ll think about whether to accept this offer or propose a counteroffer.

I appreciate your persistence , but 15 units at $8.33 each is still a bit too low for me. I’m starting to think that I need to be more aggressive in my negotiations.

accept

