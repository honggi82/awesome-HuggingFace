## MASPRM: Multi-Agent System Process Reward Model

### Milad Yazdani1, Mahdi Mostajabdaveh2*, Zirui Zhou2, Ying Xiong2

1Department of Electrical and Computer Engineering, University of British Columbia, Vancouver, BC V6T1Z4, Canada 2Huawei Technologies Canada, Burnaby, BC V5C6S7, Canada Correspondence: mahdi.mostajabdaveh1@huawei.com

# arXiv:2510.24803v2[cs.MA]12Feb2026

### Abstract

Practical deployment of multi-agent systems (MAS) demands strong performance at test time, motivating methods that guide search during inference and selectively spend compute to improve quality. We present the Multi-Agent System Process Reward Model (MASPRM). It assigns values to partial inter-agent transcripts for each action and each agent, and acts as a controller during inference. MASPRM is trained from multi-agent Monte Carlo Tree Search (MCTS) rollouts labeled only with terminal outcome rewards, without requiring human step-level annotations, by propagating returns to local targets. During inference, MASPRM guides step-level beam search (SBS) and MCTS, focusing computation on promising branches and pruning unpromising ones. We train and test MASPRM across different tasks and domains, using GSM8K, MATH, MMLU, and LOGIQA as benchmarks. Averaged across these benchmarks, MASPRM improves Hit@1 over policy likelihood by up to +13.4 points and improves ranking quality, reducing Hit@1→Hit@5 gaps by up to 10.3 points. MASPRM complements inference-time search by scoring intermediate routed transcripts to guide rollouts in MAS with fixed schedules. Code: https:

//github.com/milad1378yz/MASPRM

### 1 Introduction

Large language models (LLMs) have made multistep problem solving increasingly practical. However, reliability often drops when a single pass must simultaneously explore alternatives, justify choices, and coordinate subtasks (Zhang et al., 2024). Multiagent systems (MAS) address this limitation by running multiple agents that interact over a communication graph specifying who can message whom, often with distinct roles and capabilities. An MAS

*Corresponding Author

MCTS(10,3) + MASPRM-7B

| |
|---|

80

SBS(5,3) + MASPRM-7B

75

SBS(5,1) + MASPRM-7B

###### MCTS(10,3) + MASPRM-1.5B

Hit@1onGSM8K(%)

70

SBS(5,3) + MASPRM-1.5B

SC@5 + MASPRM-7B SC@5 + ORM-1.5B

MCTS(10,3) + ORM-1.5B

| |
|---|

65

SC@5 + MASPRM-1.5B

SBS(5,1) + MASPRM-1.5B

SC@5

MCTS(10,3) + Policy Likelihood

| |
|---|

SC@5 + Policy Likelihood

SBS(5,3) + Policy Likelihood

60

55

SBS(5,1) + Policy Likelihood

Baseline

50

SBS MCTS

Single-Pass

| |
|---|

1 2 3 4 5 6 7 8 9

TOK (×10³)

Figure 1: Trade-off between token cost (×103) and accuracy on GSM8K. Each point reports Hit@1 (yaxis) versus average tokens at inference time (x-axis).

execution produces an inter-agent transcript, an ordered sequence of routed messages: at each step, an acting agent proposes an action and routes it to recipients. This structure naturally supports inference-time procedures that allocate compute to exploration, verification, and ranking, rather than committing to one trajectory. Indeed, the community has found that how compute is spent at inference can matter as much as, or more than, parameter count: expanding and scoring partial solutions yields substantial gains in correctness and confidence (Snell et al., 2024; Cobbe et al., 2021; Lightman et al., 2023). This is especially relevant for smaller LLMs, whose lower call cost makes multi-agent search and verification feasible, and for which structured exploration can be particularly effective (Guan et al., 2025; Wang et al., 2022; Yao et al., 2023). Yet, realizing these gains in MAS depends on reliable guidance over long inter-agent transcripts. Despite this promise, search during inference in MAS remains unreliable for two reasons. First, outcome only evaluation (e.g., majority vote or final answer verifiers) is too sparse to steer long

[Figure 1]

Figure 2: Supervision from search. Left: MAS-MCTS rollouts yield backed up edge values Qˆ(s,a), which we convert into sibling preference pairs to train MASPRM. Right: the learned value head Vϕ guides MAS during inference (voting, SBS, MCTS).

inter-agent transcripts, providing little information about which message, from which agent, advanced progress (Cobbe et al., 2021). Second, errors can accumulate across steps. Without scoring partial transcripts, search may keep extending trajectories that are likely under the policy but misaligned with eventual correctness, allocating substantial compute to unproductive directions. Prior work in single-agent settings shows that feedback on intermediate reasoning can guide inference-time search toward trajectories where improvement is more likely (Setlur et al., 2024; Lightman et al., 2023; Chen et al., 2024; Guan et al., 2025; Zeng et al., 2025). However, extending these ideas to MAS is not straightforward.

We introduce MASPRM, a Process Reward Model designed for multi-agent systems. Unlike prior PRMs that score single-agent reasoning chains, MASPRM assigns values to intermediate routed transcripts and agents’ proposed actions, enabling agent-aware credit assignment in collaborative settings. These scores serve two roles. First, they act as a controller during inference and guide expansion and search budget allocation over agent rollouts under the fixed schedule. Second, they provide a process score for each agent, which is useful for analysis and for improving MAS. In this work, we study MAS with fixed schedules and outcome supervision for training, where search during inference is practical.

Developing PRMs for MAS presents challenges beyond traditional PRMs for single-agent reasoning. Score granularity. In a single-agent chain-ofthought, a "step" typically corresponds to a small reasoning act. In MAS, a step refers to an action taken by an agent, which may include tasks such as planning, routing, and tool use. Consequently, a PRM for MAS should be able to assign scores

at the level of inter-agent states rather than tokens. Schedule and graph. In MAS, the value of a transcript depends on the schedule, the order in which agents act and what interactions occur. Because this ordering matters, changing the schedule or the graph can change the expected outcome. Heterogeneous agents. Agents in a MAS can differ in role, tool access, system prompts, and even base models. As a result, intermediate inter-agent states are drawn from different distributions, unlike singleagent reasoning steps. Partial observability. Messages are routed over a communication graph, so each agent observes only part of the global transcript. This contrasts with single-agent reasoning, which typically has access to the full prior context. To address these challenges, we let each acting agent propose candidate actions from its local view, while a MASPRM scorer conditions on the full routed transcript prefix. This preserves partial observability for generation but provides a consistent global state representation for guiding inferencetime search. These design choices draw on lessons from structured exploration (Wang et al., 2022; Yao et al., 2023; Snell et al., 2024) and search guided by verifiers (Cobbe et al., 2021; Lightman et al., 2023). They also remain compatible with implicit process signals learned from outcome supervision alone (Cui et al., 2025).

#### Contributions.

- • We formulate MASPRM, a process reward model that estimates the value of intermediate inter-agent states, scores agent actions, and guides search at inference.
- • We propose MAS-MCTS supervision by backing up terminal rewards to agent-action values, enabling MASPRM training without costly manual step-level annotations.
- • We introduce MASPRM-guided search via SBS

and MCTS over agent rollouts, improving decision quality and pruning unproductive branches.

Key Results. Under matched compute budgets, MASPRM-7B improves average Hit@1 over policy likelihood by up to +13.4 points (averaged across GSM8K, MATH, MMLU, and LOGIQA; Table 1). MASPRM also improves ranking quality, reducing Hit@1→Hit@5 gaps by up to 10.3 points compared to policy likelihood.

### 2 Related Work

Recent work on MAS optimizes communication graphs, agent roles, and tool routing to improve coordination and efficiency (Zhang et al., 2024; Zhou et al., 2025). Multi-agent reinforcement fine-tuning can be unstable under sparse rewards, which highlights the credit assignment problem (Liao et al., 2025). SPA-RL addresses delayed rewards by redistributing the final reward into stepwise contributions using a learned progress estimator and improves agent performance on interactive benchmarks (Wang et al., 2025). ARMAP learns reward models from environment interaction (no human labels) and combines them with planning to evaluate trajectories and guide autonomous agents (Chen et al., 2025). Another line of work uses LLM judges, where a centralized critic assigns credit to each agent, either by generating dense rewards from natural language task descriptions (Lin et al., 2025) or by decomposing team rewards into individual signals for each agent (Nagpal et al., 2025). These approaches require repeated LLM calls, which adds overhead and limits scaling for long transcripts. In contrast, MASPRM serves as a lightweight controller (comparing to LLMs) during inference (App. C.6) by scoring messages and tool calls across the communication graph to guide rollouts and provide process scores for each agent. PRMs provide dense feedback over intermediate reasoning and have been used to guide search and prune unpromising branches, primarily in singleagent reasoning chains (Setlur et al., 2024; Lightman et al., 2023). To reduce labeling cost, several works derive process supervision from search traces. For example, ALPHAMATH ALMOST ZERO uses MCTS rollouts to construct step-level targets (Chen et al., 2024), and RSTAR-MATH pairs PRMs with MCTS to achieve state of the art results in math (Guan et al., 2025). Automated process supervision (e.g., OMEGAPRM) scales PRM data collection for math (Luo et al., 2024). Beyond do-

main specific pipelines, VERSAPRM trains PRMs that generalize across domains and remain useful at inference time (Zeng et al., 2025). Choudhury (2025) proposes AGENTPRM/INVERSEPRM for process-reward learning in a single-agent setting, where the agent interacts with an environment.

While these process supervision methods have proven effective for reasoning, they are largely developed for single-agent chains where a single linear prefix defines the intermediate state. In multiagent systems, messages are routed and agents act from different local views, so evaluating a partial transcript must be conditioned on the routed transcript and the acting agent. MASPRM makes this by scoring intermediate routed transcript prefixes, enabling guided multi-agent search at inference time.

### 3 Preliminaries

Each instance consists of a question x and a ground truth answer y⋆. We model an MAS execution as a finite routed transcript, i.e., an ordered list of interagent messages along the communication graph, ending when a message is routed to the terminal node. Correctness is assessed only at this terminal step.

We model the system as a directed graph G =

V ∪ {q,sink},E , V = {0,...,M−1}, where each i ∈ V is an agent, q is a non-acting question node that holds x, and sink is a non-acting terminal node with no outgoing edges. An edge (i → j) ∈ E means i routes a message to j. Let the adjacency A ∈ {0,1}(M+2)×(M+2) satisfy Auv = 1 iff (u→v) ∈ E, with rows/columns indexed by V ∪ {q,sink}. For u ∈ V ∪ {q,sink} define Nin(u) = {v : (v → u) ∈ E } and Nout(u) = {w : (u→w) ∈ E }.

We use a fixed schedule σ : {1,...,D} → V. At turn t ∈ {1,...,D}, if the rollout has not yet terminated, the acting agent is it = σ(t). Rollouts run for at most D turns and terminate once a message is routed to sink. If the rollout has not terminated earlier, the final scheduled action at t = D routes to sink. We use pre-turn indexing and let T ≤ D+1 denote the terminal index such that sT is terminal. In our experimental MAS configurations, termination occurs immediately after the final scheduled turn (T = D+1), but the formalism allows earlier stopping.

Let Ht−1 be the routed transcript after t−1 turns, an ordered list of triples

(speaker,message,recipient) consistent with G, with H0 = ∅. We define the state space S and represent a state as st = x, Ht−1, it ∈ S. Because routing limits information flow, agents do not observe the full routed transcript Ht−1. For each agent i ∈ V, let Ht(−i)1 denote its local view: the ordered subsequence of Ht−1 containing only tuples where i is the speaker or recipient. In what follows, the acting agent conditions action proposals on (x,Ht(−it1),it), while MASPRM scorer conditions on (x,Ht−1,it). Thus, MASPRM has access to the full routed transcript prefix, while agent policies operate under partial observability through their local views Ht(−i)1. At st, agent it chooses an action at ∈ Ait(st), a (message, recipient) pair where the recipient must be in Nout(it). Here Ai(s) is the set of valid actions for agent i at state s, yielding the deterministic successor st+1 = next(st,at). Routing affects which agents observe messages, but turn-taking is fixed by σ. If the recipient is not sink, then it+1 = σ(t+1). Otherwise, it+1 = sink, st+1 is terminal, and the rollout ends at T = t+1.

For a terminal state sT, let yˆ(sT) be the extracted final answer; define R(sT) = +1 iff yˆ(sT) = y⋆, and R(sT) = −1 otherwise. For any non-terminal state s, set R(s) = 0. For any state s and valid action a,

V (s) = E[R(sT) | s], Q(s,a) = E[R(sT) | s,a],

where the expectation is over future actions under the rollout distribution induced by the search procedure. During training, these actions are sampled via the search policy (UCT selection at expanded nodes and policy sampling at frontiers). At inference, the learned process value head guides the improved search behavior. Under deterministic transitions and terminal rewards, Q(s,a) = V next(s,a) .

Each agent i is paired with a policy π(i)(· | x,Ht(−i)1) over actions, conditioned only on its local view. During search, we form a finite candidate set Ct ⊂ Ait(st) by sampling from π(it)(· | x,Ht(−it1)).

As a baseline scorer, we use policy likelihood guidance based on a length-normalized log-likelihood of each candidate action under the acting agent (App. B.9).

A process value head Vϕ : S → R predicts the expected terminal reward from any state, whether

intermediate or terminal. Because Vϕ is defined

over states st = (x,Ht−1,it), it yields per-step, per-agent values. For scoring a decision a at state s, we use the action value Vϕ next(s,a) , which equals Q(s,a) under the relation above.

For each visited edge (s,a) we maintain a visit count N(s,a) ∈ N and a running value sum W(s,a) ∈ R with empirical estimate Qˆ(s,a) = W(s,a)/N(s,a). All successors generated during search must respect A and the schedule σ.

For search hyperparameters, we use Nsim simulations per instance and Cmax as the per-node candidate cap. Rollouts have a maximum depth D, corresponding to a single pass through the MAS schedule.

### 4 MASPRM

MASPRM is a process reward model that assigns values to intermediate MAS states via a shared head Vϕ. We train Vϕ from MAS-MCTS rollouts by backing up terminal rewards into action values Qˆ(s,a) and using them as targets for successor states with no manual annotations. At inference, MASPRM guides MCTS via upper confidence bounds applied to trees (UCT) selection (Eq. (1)) and guides SBS by value-based ranking of successors (see Fig. 2).

#### 4.1 MASPRM training

Selection uses UCT (Kocsis and Szepesvári, 2006) with constant cUCT > 0. Let Aexpit (st) = {a ∈ Ait(st) : N(st,a) > 0} denote the expanded actions at st.

Q ˆ(st, a)

a⋆ = arg max

a∈Aexpi

(st)

t

ln 1 + b∈Aexp

(st) N(st, b) 1 + N(st, a)

it

+ cUCT

(1)

Unexpanded actions are introduced in the expansion step via candidate sampling. We generate supervision by running MAS-MCTS under graph constraints A and fixed schedule σ. At a nonterminal state st, expansion samples a candidate set Ct ⊂ Ait(st) from π(it)(· | x,Ht(−it1)) and creates successors s′ = next(st,a) for a ∈ Ct. Simulations are evaluated only at terminal leaves, using vleaf(sT) = R(sT) ∈ {−1,+1}, and then backed up to update (N,W,Qˆ). We omit virtual visit initialization during training to avoid biasing the Monte Carlo targets. A concise summary appears in Algorithm 1 (App. E).

After completing rollouts, for each visited parent state st and each expanded action a ∈ Ct with N(st,a) > 0 and child s′ = next(st,a), we define a process target

yproc(s′) ≜ Qˆ(st,a) ∈ [−1,1],

a Monte Carlo estimate of Q(st,a) backed up from terminal rewards. Collecting these labeled post action states yields the process dataset Dproc = {(s′,yproc(s′))}.

We then construct a preference pair dataset by comparing sibling successors within each expanded parent state. Let Succ(st) = {s′ = next(st,a) : a ∈ Ct and N(st,a) > 0} denote the set of expanded children of st. We create preference pairs

{(s+,s−)|s+,s− ∈ Succ(st),

Dpref =

st

yproc(s+) > yproc(s−)} (2)

We train Vϕ using a Bradley-Terry ranking loss (Ouyang et al., 2022). We use ranking rather than pointwise regression because at inference the scorer is used to rank candidate successor states during search.

pϕ s+ ≻ s− =

exp(Vϕ(s+)) exp(Vϕ(s+)) + exp(Vϕ(s−))

,

1 |Dpref|

−log pϕ s+ ≻ s− .

min

ϕ

(s+,s−)∈Dpref

Because comparisons are formed among siblings, each preference pair comes from the same parent state st and differs only in the branching action. As a baseline that uses only terminal outcomes, we train an outcome reward model (ORM) Γ on the same rollouts (App. B.10) and use it as an outcome scorer in our experiments.

- 4.2 Inference time guidance For inference search, we score successor states s′ = next(st,a) with Vϕ(s′) or a policy likelihood score ψ˜pol(a | st) (App. B.9). In MCTS, newly created edges may be initialized with a virtual visit by setting N(st,a) = 1 and W(st,a) = Qˆ(st,a) = fINIT(st,a), where fINIT(st,a) = Vϕ(s′) (MASPRM) or fINIT(st,a) = ψ˜pol(a | st). MCTS (inference). We run MCTS with UCT selection (Eq. (1)). At a frontier non-terminal state

st, we sample Ct from π(it)(· | x,Ht(−it1)) and create successors s′ = next(st,a) for a ∈ Ct (with virtual

visit initialization as above). Simulation leaves are evaluated by bootstrap vleaf(s) = Vϕ(s) (App. B.9 gives the policy likelihood definition), and values are backed up to update (N,W,Qˆ). We decode by greedily following the child with maximal Qˆ from the root. Algorithm 2 (App. E) summarizes the procedure.

SBS. At depth t, expand each beam state s ∈ Bt

by sampling B2 actions and scoring each successor by the chosen scorer. Here Bt is the beam at depth t, B2 is the number of sampled actions per beam state, and B1 is the beam width. We pool all |Bt|B2 successors and keep the top-B1 to form Bt+1, repeating for t = 1,...,D, and return the highest scoring terminal state in the final beam. Algorithm 3 (App. E) summarizes SBS.

### 5 Experiments & Results

We design our experiments to answer the following questions:

- RQ1: MASPRM vs. policy likelihood at inference. Does a MASPRM improve top-1 accuracy (Hit@1) under matched compute across datasets and search algorithms?
- RQ2: MASPRM vs. ORM. How does MASPRM guidance compare to a conventional ORM under matched compute?
- RQ3: Zero-shot out of distribution (OOD) generalization. Does a MASPRM trained on one dataset and/or agent LLM transfer zero-shot to other datasets and agents and improve Hit@1 at matched compute? We also analyze Hit@k sensitivity to illustrate the robustness of MASPRM (see App. C.1, Table 5).

#### 5.1 Overall experimental setup

Benchmarks. We evaluate on GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) (free-form math reasoning) and MMLU (Hendrycks et al., 2021a) and LOGIQA (Liu et al., 2020) (more detail in App. B.1). Answer extraction and normalization are described in App. B.2. We use the dataset-specific MAS configurations presented in App. D (agent roles, schedule depth D, and communication graph) and keep them fixed within each dataset. Unless stated, agents are Qwen2.5-1.5B-Instruct. Learned scorers (MASPRM and ORM) use Qwen2.5-1.5B/7B as the base model. We compare policy likelihood guidance (App. B.9) to learned scorers. We use

MASPRM Vϕ, which scores intermediate MAS states, and an outcome reward model (ORM) Γ, which scores only terminal outputs. Both are trained from MAS-MCTS rollouts labeled with terminal reward. MASPRM uses Bradley-Terry preference ranking (Sec. 4.1), and the ORM uses supervision from terminal outcomes (App. B.10). Additional training details appear in App. B. We include an ablation on the scorer head choice in App. C.3 and training time details in App. B.8.

Inference procedures. For each dataset, we keep prompts and graphs fixed (App. D) and vary only the inference algorithm and the scorer. Scorers are ψ˜pol, Γ, or Vϕ (Sec. 4.2). Single-Pass executes the fixed MAS schedule once and returns the final extracted answer. Self-Consistency@K (SC@K) (Wang et al., 2022) runs K=5 independent MAS rollouts producing answers yˆ(i) and scalar scores w(i) (for MASPRM, the mean value after each action, averaged across turns). We sigmoid normalize {w(i)}Ki=1 to w˜(i) ∈ [0,1], compute a total score for each distinct answer S(ˆy) =

i:ˆy(i)=ˆy w˜(i), and return the answer with the largest S(ˆy). SBS(B2,B1) keeps a beam of width B1, expands each beam state with B2 sampled actions per step, and returns the highest scoring terminal candidate. MCTS(Nsim,Cmax) runs UCT search and decodes by following the child with maximal backed up value from the root (Sec. 4.2, App. E). Budgets and metrics. We report Hit@1 (exact-match after normalization), token cost (TOK (×103)), and agent calls (AC). We match inference compute primarily by AC while keeping TOK comparable. Reward model calls (RC) are deterministically recoverable from AC and the inference hyperparameters (e.g., K and Nsim). Please refer to App. B.6 for more detail. For methods that output multiple ranked terminal candidates, we also report Hit@k for k ∈ {1,3,5} (Sec. 5.5), i.e., the fraction of examples whose correct answer appears among the top-k candidates when ranked by the scorer. Unless stated, results use seed 42 and we average stochastic methods over three seeds (14, 24, 42). Hardware details are in App. B.7.

- 5.2 MASPRM improves MAS inference under matched compute budgets

To answer RQ1 and RQ2, we evaluate MASPRM under matched compute budgets (Sec. 5.1) across datasets and inference procedures (Single-Pass, SC@5, SBS(5,1), SBS(5,3), MCTS(10,3)). We compare three scoring signals: policy likeli-

hood, an ORM, and MASPRM (with 1.5B vs. 7B scorer sizes where applicable). Table 1 shows that MASPRM-guided inference improves Hit@1 over policy likelihood scoring and ORM scoring under comparable compute (similar AC and TOK). The agent is fixed across rows (Qwen2.5-1.5B-Instruct). Across GSM8K, MATH, MMLU, and LogiQA, MASPRM-7B outperforms policy likelihood scoring under matched AC and TOK. Averaged across datasets, policy likelihood → MASPRM-7B gains are +8.5 points for SC@5, +13.4 points for SBS(5,1), +12.9 points for SBS(5,3), and +13.4 points for MCTS(10,3). Under size-matched comparison (1.5B models), MASPRM-1.5B shows mixed results vs. ORM-1.5B in SC@5 (ORM ahead by 0.6–1.2 points on math benchmarks), but MASPRM-1.5B outperforms ORM-1.5B in MCTS by 2.0–3.0 points on math. Scaling scorer size from 1.5B to 7B yields +3.3 to +14.4 point gains depending on search method and dataset. TOK remains comparable across scorers. MASPRM introduces additional reward model calls (RC, App. B.6), but RC has substantially lower wall clock cost than agent generation (App. C.6). We also compare guided MAS search to singleagent baselines (a single LLM answering directly, without MAS) across several instruction models. Table 2 reports Hit@1 across all four datasets. Guided MAS search with 1.5B agents and MASPRM-7B substantially improves over the single 1.5B baseline (Table 2). In particular, it is competitive with the strongest single-agent baselines (Mistral 24B-Instruct and GPT-4o-mini) on GSM8K and MATH, and substantially outperforms them on MMLU.

#### 5.3 Zero-shot OOD: Datasets & Agents

We now address RQ3 by studying two forms of zero-shot generalization.

Transfer across datasets. We train MASPRM7B with a Bradley-Terry head on one dataset and evaluate it on a different dataset, without retraining the reward model on the target dataset, under MCTS(10,3) with Qwen2.5-1.5B-Instruct as the agent. We focus on transfers within a domain, specifically math (MATH ↔ GSM8K) and knowledge/reading (MMLU ↔ LOGIQA). Diagonal entries correspond to in distribution training. Results are reported in Table 3.

#### Cross-LLM Transfer. We swap only the agent

- Table 1: Hit@1, TOK (×103), and AC across datasets and inference methods with Qwen2.5-1.5B-Instruct agents (mean±std over 3 seeds for stochastic methods).

Search Scorer

GSM8K MATH MMLU LOGIQA

Hit@1 TOK AC Hit@1 TOK AC Hit@1 TOK AC Hit@1 TOK AC Single-Pass – 47.2(±1.9) 0.7(±0.0) 4 31.0(±1.5) 1.1(±0.1) 4 48.4(±0.6) 0.1(±0.0) 3 32.5(±1.2) 0.5(±0.0) 3

SC@5 – 63.4(±1.7) 3.6(±0.0) 20 42.4(±0.9) 5.6(±0.1) 20 52.7(±0.6) 0.3(±0.1) 15 35.4(±1.9) 2.6(±0.1) 15 SC@5 Policy Likelihood 61.2(±1.3) 3.6(±0.0) 20 41.5(±0.9) 5.6(±0.1) 20 54.5(±1.9) 0.3(±0.1) 15 35.9(±0.8) 2.6(±0.1) 15 SC@5 ORM-1.5B 65.4(±1.1) 3.6(±0.0) 20 45.4(±1.3) 5.6(±0.1) 20 58.5(±0.7) 0.3(±0.1) 15 36.1(±1.2) 2.6(±0.1) 15 SC@5 MASPRM-1.5B 64.2(±0.9) 3.6(±0.0) 20 44.8(±1.0) 5.6(±0.1) 20 58.3(±0.2) 0.3(±0.1) 15 36.1(±0.7) 2.6(±0.1) 15 SC@5 MASPRM-7B 67.8(±0.6) 3.6(±0.0) 20 49.2(±1.3) 5.6(±0.1) 20 67.2(±3.9) 0.3(±0.1) 15 43.0(±0.6) 2.6(±0.1) 15

SBS(5,1) Policy Likelihood 55.0(±1.4) 3.6(±0.1) 20 41.0(±1.2) 5.5(±0.1) 20 55.0(±2.4) 0.3(±0.0) 15 35.9(±1.6) 2.6(±0.1) 15 SBS(5,1) MASPRM-1.5B 63.4(±0.9) 3.4(±0.1) 20 38.5(±1.6) 5.6(±0.1) 20 58.7(±1.6) 0.3(±0.0) 15 36.1(±0.7) 2.7(±0.1) 15 SBS(5,1) MASPRM-7B 73.4(±0.7) 3.2(±0.1) 20 50.4(±0.9) 5.5(±0.1) 20 72.4(±1.7) 0.2(±0.0) 15 44.1(±0.9) 2.6(±0.1) 15

SBS(5,3) Policy Likelihood 60.3(±1.2) 8.4(±0.1) 50 40.4(±1.0) 12.3(±0.2) 50 59.4(±1.8) 0.6(±0.1) 35 37.0(±0.7) 6.7(±0.0) 35 SBS(5,3) MASPRM-1.5B 69.1(±0.3) 8.4(±0.0) 50 41.2(±1.2) 12.3(±0.2) 50 60.4(±0.6) 0.8(±0.1) 35 37.3(±0.2) 6.6(±0.0) 35 SBS(5,3) MASPRM-7B 77.5(±0.4) 7.9(±0.0) 50 51.3(±1.2) 12.1(±0.2) 50 74.8(±3.6) 0.7(±0.1) 35 45.0(±0.8) 6.5(±0.0) 35

MCTS(10,3) Policy Likelihood 62.5(±0.7) 5.7(±0.1) 38.2(±5.4) 42.5(±1.1) 9.8(±0.3) 34.5(±3.2) 60.5(±0.8) 0.2(±0.0) 12.4(±1.0) 36.5(±1.1) 2.4(±0.1) 13.6(±2.1) MCTS(10,3) ORM-1.5B 66.8(±0.6) 5.8(±0.1) 35.2(±3.2) 45.3(±1.1) 9.8(±0.3) 33.6(±4.1) 66.2(±0.7) 0.2(±0.0) 12.1(±1.3) 40.2(±0.9) 2.3(±0.1) 13.1(±2.6) MCTS(10,3) MASPRM-1.5B 69.8(±0.5) 5.6(±0.1) 34.7(±2.4) 47.3(±0.9) 9.5(±0.3) 33.7(±3.5) 68.8(±0.6) 0.2(±0.0) 12.4(±1.9) 42.2(±0.8) 2.3(±0.1) 13.5(±1.7) MCTS(10,3) MASPRM-7B 82.9(±3.2) 5.5(±0.1) 34.4(±1.7) 51.8(±0.8) 9.4(±0.3) 32.9(±2.7) 75.2(±0.5) 0.2(±0.0) 12.4(±0.6) 45.5(±0.7) 2.4(±0.1) 13.5(±2.4)

- Table 2: Hit@1 (%) for single-agent baselines vs. guided MAS search (Qwen2.5-1.5B-Instruct agents

+ MASPRM-7B).

Method GSM8K MATH MMLU LOGIQA Single-Agent Baselines

Qwen2.5-1.5B-Instruct 41.9 17.5 34.3 25.8 Llama3.2 3B-Instruct 54.8 24.4 41.2 35.3 Llama3.1 8B-Instruct 36.2 26.7 50.5 42.2 GPT-4o-Mini 62.0 51.0 59.0 50.1 Mistral 24b-Instruct 80.4 51.9 60.6 56.4

Guided MAS (Qwen2.5-1.5B-Instruct)

MCTS(10,3)+MASPRM 82.9 51.8 75.2 45.5 SBS(5,3)+MASPRM 77.5 51.3 74.8 45.0

- Table 3: Hit@1 (%) under MCTS(10,3): MASPRM7B train→test transfer (rows=train, cols=test), plus policy likelihood baseline.

edge cross-transfer (MMLU↔LogiQA) shows −3.7 to −6.1 point degradation vs in-domain but +5.3 to +8.6 point gain over policy likelihood. Averaged across transfers, cross-domain MASPRM degrades by 5.8 points vs in-domain training but improves by +7.6 points vs policy likelihood.

MASPRM-7B trained on Qwen2.5-1.5B rollouts transfers to Llama3.2-3B agents without retraining (Table 6). Under MCTS(10,3), MASPRM7B outperforms policy likelihood by +11.2 points on GSM8K (64.1→75.3) and +8.7 points on MATH (43.4→52.1), averaging +10.4 points across datasets. Absolute Hit@1 values differ between Qwen and Llama agents, but relative gains over policy likelihood and ORM remain consistent (App. C.2).

Test dataset (Hit@1 %) GSM8K MATH MMLU LOGIQA Policy Likelihood 62.5 42.5 60.5 36.5

Train dataset

#### 5.4 Process vs. outcome guidance

GSM8K 82.9 47.9 – – MATH 73.6 51.8 – – MMLU – – 75.2 41.8 LOGIQA – – 69.1 45.5

Table 7 compares head architectures under MCTS(10,3) with Qwen2.5-1.5B-Instruct agents. For ORM, BCE classification outperforms MSE regression by +0.6 to +0.8 points across datasets. For MASPRM, Bradley-Terry (BT) preference ranking outperforms MSE regression by +3.2 to +4.3 points on math benchmarks for the 1.5B model and by +1.2 to +9.6 points for the 7B model. Comparing process vs outcome guidance with size-matched 1.5B models and optimal heads (MASPRM-1.5B BT vs ORM-1.5B BCE), MASPRM outperforms ORM by +3.0 points on GSM8K (69.8 vs 66.8) and +2.0 points on MATH (47.3 vs 45.3) under identical MCTS(10,3) search budgets (Table 1).

LLM to Llama3.2 3B-Instruct at inference time while keeping the reward models fixed (no retraining, Sec. 5.1). We repeat the same search/scoring setups as in Sec. 5.2 and report results in App. C.2 (Table 6).

Training on one dataset and evaluating on another within the same domain yields cross-transfer performance that surpasses policy likelihood baselines, yet remains below in-domain training. Math crosstransfer (GSM8K↔MATH) shows −3.9 to −9.3 point degradation vs in-domain but +5.4 to +11.1 point gain over policy likelihood (Table 3). Knowl-

#### 5.5 Stability of MASPRM Search

Under MCTS(10,3), policy likelihood scoring shows Hit@1→Hit@5 gaps of +19.6 points (GSM8K), +18.1 points (MATH), +10.8 points (MMLU), and +15.9 points (LogiQA), averaging +16.1 points (Table 5). MASPRM-7B shows Hit@1→Hit@5 gaps of +5.3, +8.7, +2.2, and +7.0 points respectively, averaging +5.8 points, a reduction of 10.3 points. SC@5 shows similar patterns with policy likelihood averaging +19.0 point gaps and MASPRM-7B averaging +10.5 point gaps.

#### 5.6 Sensitivity to MAS communication graph

Three GSM8K MAS configurations (MAS1: planthen-solve with verification, MAS2: critique and revise, MAS3: consensus) with identical agents (Qwen2.5-1.5B) and scorer (MASPRM-7B) show Single-Pass Hit@1 ranging 47.2 to 56.2 (9.0 point range). With MASPRM guidance, ranges narrow to 1.8 points for SBS(5,1) (73.4 to 75.2), 2.2 points for SBS(5,3) (77.5 to 79.7), and 2.0 points for MCTS(10,3) (80.9 to 82.9) (Table 8).

### 6 Discussion

- For RQ1, MASPRM-guided search yields higher Hit@1 than policy likelihood under matched inference-time compute across datasets and inference procedures (Table 1). When search returns a ranked list of candidates, MASPRM also narrows Hit@1→Hit@5 gaps (Sec. 5.5, Table 5), consistent with more reliable ranking of multi-agent transcripts.
- For RQ2, dense process guidance is more effective than outcome scoring for guiding MCTS at the same model size, while SC@5 results are more mixed (Table 1). Head ablations suggest that preference ranking heads are a particularly good fit for search because they mirror the pairwise comparisons performed during selection (Table 7).
- For RQ3, the learned process signal generalizes beyond a single rollout distribution: MASPRM transfers across related datasets and across different LLMs without retraining (Table 3, Table 6). It also remains effective across different MAS communication graphs (Table 8), suggesting that MASPRM captures reusable structure in multi-agent reasoning traces rather than overfitting to a particular coordination scheme. Finally, guided MAS search with small agents and MASPRM is competitive with substantially larger single-agent baselines on GSM8K and MATH,

and outperforms them on MMLU (Table 2). This highlights a complementary scaling path through inference-time compute allocation.

### 7 Conclusion

We presented MASPRM, a process reward model for multi-agent systems. It scores routed transcript prefixes and guides inference-time search with SBS and MCTS. Across GSM8K, MATH, MMLU, and LOGIQA, MASPRM improves Hit@1 over policy likelihood scoring and an outcome reward model (ORM) under matched compute. MASPRM also transfers across related datasets and agent LLMs without retraining. Many MAS coordinate roles such as planning, solving, and verification, and search over multiple candidate transcripts. MASPRM provides process guidance in this setting. Future work includes using MASPRM as a dense signal for multi-agent reinforcement learning, with safeguards against reward hacking. Another direction is learning communication graphs and turn-taking schedules, improving training objectives and calibration, and extending to tool-augmented and multimodal tasks.

### 8 Limitations

We study scheduled MAS with outcome supervision for training; we do not claim generality to open-ended interactive environments or real-world deployments where latency, monitoring, and safety constraints may differ from our setting. Training requires verifiable terminal outcomes. At inference, MASPRM operates without ground-truth labels, but guidance quality depends on the training distribution. Extending to imperfect verifiers is an important direction. We evaluate fixed MAS configurations and do not study dynamic routing, changing agent populations, or continuous interaction. Our graph-sensitivity experiments (Table 8) show reduced cross-topology variance; broader topologies remain future work. The scorer conditions on the full routed transcript prefix, requiring centralized message access. This may not fit privacyconstrained settings, and the scoring cost grows with transcript length. Agent-local or hierarchical critics are natural extensions. We focus on benchmarks with clear ground-truth answers. We do not evaluate on open-ended tasks where quality, helpfulness, or safety is harder to measure, and we do not test adversarial pressure against MASPRM (e.g., reward hacking) or score calibration.

### References

Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. 2024. Alphamath almost zero: process supervision without process. Advances in Neural Information Processing Systems, 37:27689–27724.

Zhenfang Chen, Delin Chen, Rui Sun, Wenjun Liu, and Chuang Gan. 2025. Scaling autonomous agents via automatic reward modeling and planning. arXiv preprint arXiv:2502.12130.

Sanjiban Choudhury. 2025. Process reward models for llm agents: Practical framework and directions. arXiv preprint arXiv:2502.10325.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, and 1 others. 2025. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456.

Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. 2025. rstar-math: Small llms can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2501.04519.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR).

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Geoffrey Irving, Paul Christiano, and Dario Amodei.

2018. Ai safety via debate. arXiv preprint arXiv:1805.00899.

Levente Kocsis and Csaba Szepesvári. 2006. Bandit based monte-carlo planning. In European conference on machine learning, pages 282–293. Springer.

Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. arXiv preprint arXiv:1706.03872.

Junwei Liao, Muning Wen, Jun Wang, and Weinan Zhang. 2025. Marft: Multi-agent reinforcement finetuning. arXiv preprint arXiv:2504.16129.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Muhan Lin, Shuyang Shi, Yue Guo, Vaishnav Tadiparthi, Behdad Chalaki, Ehsan Moradi Pari, Simon Stepputtis, Woojun Kim, Joseph Campbell, and Katia Sycara. 2025. Speaking the language of teamwork: Llmguided credit assignment in multi-agent reinforcement learning. arXiv preprint arXiv:2502.03723.

Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2020. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, and 1 others. 2024. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594.

Kartik Nagpal, Dayi Dong, Jean-Baptiste Bouvier, and Negar Mehr. 2025. Leveraging large language models for effective and explainable multi-agent credit assignment. arXiv preprint arXiv:2502.16863.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. 2024. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314.

Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, and Wenjie Li. 2025. Spa-rl: Reinforcing llm agents via stepwise progress attribution. arXiv preprint arXiv:2505.20732.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Planand-solve prompting: Improving zero-shot chain-ofthought reasoning by large language models. arXiv preprint arXiv:2305.04091.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, and 1 others. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

Yilin Yang, Liang Huang, and Mingbo Ma. 2018. Breaking the beam search curse: A study of (re-) scoring methods and stopping criteria for neural machine translation. arXiv preprint arXiv:1808.09582.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. 2024. Textgrad: Automatic" differentiation" via text. arXiv preprint arXiv:2406.07496.

Thomas Zeng, Shuibai Zhang, Shutong Wu, Christian Classen, Daewon Chae, Ethan Ewer, Minjae Lee, Heeju Kim, Wonjun Kang, Jackson Kunde, and 1 others. 2025. Versaprm: Multi-domain process reward model via synthetic reasoning data. arXiv preprint arXiv:2502.06737.

Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, and Dawei Cheng. 2024. G-designer: Architecting multi-agent communication topologies via graph neural networks. arXiv preprint arXiv:2410.11782.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and 1 others. 2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625.

Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vuli´c, Anna Korhonen, and Sercan Ö Arık. 2025. Multi-agent design: Optimizing agents with better prompts and topologies. arXiv preprint arXiv:2502.02533.

### A Illustrative Examples

#### A.1 Example MAS and schedule

This example with four agents (Fig. 3) is illustrative and does not exactly match all MAS variants used in experiments. See App. D for the full set of MAS configurations used in the results and ablations. Dashed directed edges indicate allowed message routing. One agent acts per depth according to σ, producing outputs o1,...,o4.

[Figure 2]

- Figure 3: Example MAS (four agents) and schedule. Dashed directed edges define allowed routing and the

order of agent outputs o1 to o4. One agent acts per depth according to σ.

A.2 Stylized transcript for a GSM8K problem

We visualize (Fig. 4) one rollout on the ticket problem: "A group buys 14 tickets. Adult tickets cost $5 and child tickets cost $3. The total is $56. How much more was spent on adult tickets than on child tickets?" The roles are Reader & Fact Extractor (o1), Planner & Variable Setter (o2), Arithmetic Worker (o3), and Verifier & Finalizer (o4). We show two transcripts. In the green path, the system solves a+c = 14 and 5a+3c = 56 to get a=c=7, then returns 5a−3c = 35−21 = 14. In the red path, it makes an arithmetic error and outputs 15.

[Figure 3]

- Figure 4: Stylized transcript (example). A 4-agent

pipeline produces outputs o1-o4. Green is correct (a=c=7 ⇒ 5a−3c = 14), while red is incorrect (35−21 ̸= 15).

#### A.3 MCTS rollout with MASPRM values

Fig. 5 shows value-guided search over the unrolled tree. Internal annotations (e.g., 0.5, 0.25) show example empirical Q-values Qˆ backed up from MASPRM leaf evaluations. White circles denote agent outputs. Green/red leaves indicate terminal reward +1/ − 1 (correct/incorrect). Selection uses UCT (Eq. (1)).

[Figure 4]

Figure 5: MCTS over the unrolled tree (example). Numbers on nodes are illustrative empirical Q-values Qˆ. Green/red leaves denote R ∈ {+1,−1}.

B Additional Experimental Details

#### B.1 Benchmarks and Dataset Splits

We use the four benchmarks from Sec. 5.1. All datasets are loaded with the HuggingFace datasets library. When forming subsets, we shuffle with a fixed random seed (42). For each dataset, we define a training pool Dtrain and a disjoint held out evaluation set Deval used for reporting. We then create a development set Ddev (for hyperparameter selection) by holding out 5% of the (shuffled) training pool:

|Ddev| = ⌊0.05|Dtrain|⌋, Dtrain ← Dtrain\Ddev.

(3) MATH (Competition-Math). We use a held out evaluation subset Deval with |Deval|=2000 examples and use the remainder as the training pool, holding out 5% of Dtrain as a development set.

GSM8K and LogiQA. We use the train split as Dtrain and the test split as Deval, and we hold out 5% of Dtrain as a development set (we do not use any provided validation split).

MMLU. We use the larger provided partition as Dtrain and the other partition as Deval, and we hold out 5% of Dtrain as a development set. Finally, we optionally cap the number of used examples to n by taking the first n examples after shuffling. Table 4 reports the exact split sizes after applying this protocol.

Dataset Train Dev Eval MATH (Competition-Math) 9,975 525 2,000 GSM8K 7,100 373 1,319 MMLU 13,340 702 1,531 LogiQA 7,008 368 651

Table 4: Dataset split sizes (Train/Dev/Eval).

#### B.2 Answer extraction and normalization

Final answers are extracted from the last scheduled agent (e.g., Verifier, Judge, Query_Checker, or the single-agent baseline), with multiple fallback patterns to accommodate the different MAS prompt templates used in App. D. We first attempt to extract a final answer span using a case insensitive regex over common final answer templates:

(Final Answer|Answer) *: *(.*).

If a match is found, we use the captured group as the candidate answer. If no explicit final answer tag is present, we fall back to: (i) for free-form datasets (GSM8K, MATH): strip trailing units from the final message (e.g., cm, degrees, or \text{...}) and then take the last remaining number-like token. (ii) for multiple-choice datasets (MMLU, LOGIQA): the first option label in {A,B,C,D} (case insensitive) that appears as a standalone token in the final message. For free-form datasets, we parse the captured span as a number when possible and apply light normalization (lowercase, strip commas/spaces, and remove trailing unit markers). We accept numeric answers within tolerance δ = 10−3. For multiplechoice datasets, we map the extracted label to the canonical answer. Non-parsable outputs or timeouts are scored incorrect.

##### B.3 MASPRM training data generation For training data generation, we run MASMCTS(40,3) on training problems (i.e., Nsim=40 simulations and expansion cap Cmax=3 children per expanded state) with cUCT=4.0. Leaf values are set to the ground-truth terminal reward,

vleaf(sT) = R(sT) ∈ {−1,+1} (no virtual visits during training).

Process targets. From the resulting search trees, we construct process targets at the post action state level. For each visited edge (st,a) with successor s′ = next(st,a), we assign yproc(s′) ≜ Qˆ(st,a) where Qˆ(st,a) = W(st,a)/N(st,a) is the backed

up Monte Carlo estimate. Targets are clipped to [−1,1].

Preference pairs (BT head). Eq. (2) in the main text defines Dpref as all ordered sibling pairs with yproc(s+) > yproc(s−). In practice, materializing all such pairs can be quadratic in the number of expanded children and can introduce many nearduplicate comparisons (e.g., formatting variants of the same action). We therefore mine a small, highsignal subset of pairs from each MAS-MCTS tree and use it as the instantiated Dpref in our experiments. We fix the path length to D+1 nodes (root + D agent actions) and, for each prefix node at depth d (0 ≤ d < D), we compare next step candidate children that share the same prefix state. Concretely, at each such prefix we: (i) deduplicate children by normalized action text, (ii) rank unique candidates by their backed up value yproc(s′), (iii) take up to K+=4 top-valued and K−=4 bottom-valued candidates (disjoint), and (iv) emit pairs (s+,s−) between the two sets when yproc(s+) > yproc(s−) (ties ignored). For the final step (d=D−1), we restrict candidates to terminal children. We cap the total number of mined preference pairs per tree to 8.

Pointwise samples (MSE head). For the MSE head ablations, we extract fixed length root to terminal transcripts of D+1 nodes and supervise each step with the node value (clipped to [−1,1]). We keep only terminal leaves with reward ±1, and remove exact duplicate samples.

#### B.4 MASPRM model and optimization

Backbone: Qwen2.5-1.5B with a scalar head that maps each state to a single score in [−1,1] (via a tanh output). For the BT variant, this scalar is the Bradley-Terry score used for pairwise ranking, and we use its raw value as the inference score. For the MSE variant, we regress the scalar to yproc(s′). We use QLoRA 4-bit with rank 256 and dropout 0.05. Batch size is 8 (effective with grad accumulation 16), LR 1×10−5, epochs 5, optimizer ADAMW, and weight decay 0.01. Evaluation metrics on held out preference pairs include average margin, pairwise accuracy, AUC, and BT loss.

#### B.5 Search configurations

SBS(B2,B1): At each depth, we sample B2 candidates per beam state and keep the top B1 globally (we report SBS(5,1) and SBS(5,3)). Temperature = 0.7, top-p = 0.95. Tie-breaking uses the

higher MASPRM score (if applicable), otherwise the higher final answer policy likelihood.

MCTS(Nsim,Cmax): Selection uses UCT (Eq. (1)). In our main experiments we use MCTS(10,3) (Nsim=10, Cmax=3). Expansion samples up to Cmax children by top-p sampling from the current agent. Leaf evaluation uses Vϕ. Rollout depth is capped at the scheduled depth D of the active MAS configuration (App. D), so that the final scheduled agent produces the answer. On dead ends, we back up the current MASPRM leaf value. No policy priors are added to UCT unless otherwise stated.

#### B.6 Compute budgets and accounting

We match inference compute primarily by AC and keep TOK comparable across methods. An agent call is one LLM generation by one agent at one depth. TOK counts total generated tokens aggregated over all agent calls. RC counts reward model forward passes used to score candidate successor states during search.

Recovering RC from AC. In our reported setups, RC is deterministically recoverable from AC and the inference hyperparameters (and the scheduled depth D, App. D), so we omit RC from the main tables. Policy likelihood guidance uses no learned scorer, so RC = 0. When a scorer is applied to each newly generated successor state (SBS, MCTS with MASPRM, and SC@K with MASPRM), we score each successor once, giving RC = AC. For ORMs used only to score terminal candidates, RC equals the number of terminal evaluations: SC@K uses RC = K = AC/D, and MCTS(Nsim,Cmax) with an ORM at simulation leaves uses RC = Nsim (one ORM call per simulation). Let D denote the scheduled depth (number of agent turns) in a single MAS pass. D depends on the MAS configuration (App. D). Single-Pass uses AC = D, and SC@K uses AC = K D. For SBS with global beam B1 and samples per state B2, the agent call budget is AC = B2 + (D−1)B1 B2. For MCTS, AC is not deterministic due to stochasticity. We report the empirical mean ± std and tune hyperparameters to obtain comparable average AC and TOK across scorers.

#### B.7 Hardware

Experiments run on 8× NVIDIA Tesla V100-PCIE32GB (32GB VRAM, driver 535.54.03, CUDA 12.2), an Intel Xeon Gold 6140 CPU @ 2.30GHz, and 754GB RAM.

#### B.8 MASPRM training time

As a minimal reference point, GSM8K has 7,100 training instances (Table 4). With our cap of at most 8 preference pairs mined per MAS-MCTS tree (App. B.3), this yields at most 56,800 preference pairs for MASPRM fine-tuning. With 5 epochs and effective batch size 128 (batch size 8 with grad accumulation 16), this corresponds to ≈ 2.2×103 optimizer steps. On our V100 setup, fine-tuning a single scorer typically takes at most 48 hours for the 1.5B model and at most 72 hours for 7B. Other datasets are similar and scale roughly with the corresponding pair counts (56k–107k from Table 4).

#### B.9 Policy likelihood guidance

To compare against policy likelihood guidance, we score each candidate action a by its lengthnormalized log-likelihood under the acting agent’s language model (Wu et al., 2016; Yang et al., 2018; Koehn and Knowles, 2017):

L

1 L

log pθπ wℓ | w<ℓ,x,Ht(−it1) ,

ψpol(a | st) =

ℓ=1

(4) where pθπ(· | w<ℓ,x,Ht(−it1)) is the token distribution of the acting agent’s policy, and w1:L are the L generated tokens for a. Because ψpol has a different scale than learned value heads, we map it to [0,1] with a sigmoid: ψ˜pol(a | st) = sigmoid(ψpol(a | st)) where sigmoid(z) = 1/(1 + exp(−z)). In MCTS, we use ψ˜pol for virtual-visit initialization fINIT(st,a) and for frontier leaf evaluation vleaf(st) = maxa∈Ct ψ˜pol(a | st). In SBS, we rank successors by ψ˜pol.

#### B.10 ORM Γ

We compare against a standard ORM that assigns a scalar score to a completed transcript. Concretely, an ORM Γ maps a terminal state sT = (x,HT−1,sink) (i.e., the full routed transcript, the same input MASPRM receives at the final step) to a scalar score and is only defined on terminal states (it is not used on intermediate states). We train ORMs from the same MAS-MCTS rollouts used for MASPRM, but using only terminal outcomes for supervision. From each terminal state sT, we compute the terminal reward r = R(sT) ∈ {−1,+1} (via the extracted final answer yˆ(sT)) to form: Dout = (sT, r) : sT terminal, r = R(sT) .

We then fit the ORM ΓθΓ by minimizing

min

θΓ

1 |Dout|

ℓ ΓθΓ(sT), r ,

(sT ,r)∈Dout

using either a BCE classification head (treating correctness as a binary label) or an MSE regression head (regressing to r ∈ {−1,+1}). Training hyperparameters and optimizer mirror the MASPRM training. Loss/head variants (BCE vs. MSE) are detailed in App. C.3.

### C Additional Experiments

- C.1 Hit@k sensitivity under limited candidates

We study robustness when only a small number of candidate solutions can be generated and returned. Table 5 reports Hit@k for k ∈ {1,3,5} under MCTS(10,3), SC@5, and SBS(5,3) with Qwen2.5-1.5B-Instruct agents. Policy likelihood scoring shows Hit@1→Hit@5 gaps averaging +16.1 points for MCTS and +19.0 points for SC@5. MASPRM scoring reduces these gaps to +5.8 points (MCTS) and +10.5 points (SC@5), improvements of 10.3 and 8.5 points respectively.

- C.2 Cross-LLM transfer results
- C.3 Head-type ablation for MASPRM and ORM

We ablate reward-model head choices under a fixed search budget (RQ2). We focus on MCTS with Nsim=10 simulations (MCTS(10,3)) using Qwen2.5-1.5B-Instruct as the agent LLM and compare: ORM variants trained with BCE vs. MSE heads. MASPRM-1.5B and MASPRM-7B with MSE vs. BT heads. We report Hit@1 on all four datasets in Table 7.

MASPRM. We use the same MAS-MCTS rollouts and preprocessing as App. B.3. For the BT head, we train on extracted preference pairs with the Bradley-Terry ranking loss (Sec. 4.1). For the MSE head, we train pointwise by minimizing ℓMSE = Vϕ(s′) − yproc(s′) 2 to the clipped process targets.

ORM. ORM training discards intermediate states and uses only terminal outcomes from the same rollouts: Dout = {(sT,r)} with r = R(sT). The BCE ORM treats correctness as a binary label, while the MSE ORM regresses to the scalar reward r ∈ {−1,+1}.

Bradley-Terry MASPRM heads outperform MSE regression heads by +3.2 to +4.3 points (1.5B model) and +1.2 to +9.6 points (7B model) on math benchmarks. For ORM, BCE classification outperforms MSE regression by +0.6 to +0.8 points consistently across datasets. These results demonstrate that ranking-based objectives (BT for process, BCE for outcome) better align with the comparative selection task performed during search than regression-based losses.

- C.4 Graph sensitivity ablation on GSM8K We evaluate three GSM8K MAS instantiations, de-

noted MAS1-MAS3. All three share the same generator model family and reward backbone: the agents are Qwen2.5-1.5B-Instruct, and the scorer is MASPRM-7B built on Qwen2.5-7B. We keep the search/decoding hyperparameters fixed (App. B). The only differences across MAS1MAS3 are the agent role prompts and the directed communication graph (which determines what each agent can condition on at its turn).

MAS definitions. MAS1 corresponds to a planthen-solve pipeline with verification (Reader → Planner → Solver → Verifier) (Wang et al., 2023). MAS2 corresponds to the critique and revise pipeline (Solver → Evaluator → Reflector → Reviser) (Madaan et al., 2023). MAS3 corresponds to a consensus style pipeline with two independent solutions and a final judge (Reader → SolverA → SolverB → Judge), inspired by self-consistency (Wang et al., 2022). We tried to optimize our MAS2 and MAS3 with a method inspired by (Yuksekgonul et al., 2024). (Exact prompts/graphs are in App. D.) MASPRM scoring protocol. At a fixed prefix state s, candidate continuations share identical previous steps and differ only in the final (newly generated) step. Concretely, for candidate next step messages a, we form successor states s′ = next(s,a) and score each s′ with MASPRM. Search selects and expands based on these relative scores. Single-Pass performance varies by 9.0 points across MAS1 to MAS3 (47.2 to 56.2). With MASPRM guidance, all three configurations converge to within 1.8 to 2.2 points (SBS/MCTS ranges: 73.4 to 75.2, 77.5 to 79.7, 80.9 to 82.9), reducing cross-topology variance by 6.8 to 7.2 points.

- C.5 OOD transfer protocol

The GSM8K-trained MASPRM is frozen for MATH (no further tuning). We keep the same preprocessing and scoring protocol.

Table 5: Hit@k (%) for k ∈ {1,3,5} under MCTS(10,3), SC@5, and SBS(5,3).

GSM8K MATH MMLU LOGIQA Hit@1 Hit@3 Hit@5 Hit@1 Hit@3 Hit@5 Hit@1 Hit@3 Hit@5 Hit@1 Hit@3 Hit@5

Method

MCTS(10,3) (policy likelihood) 62.5 76.4 82.1 42.5 54.5 60.6 60.5 68.2 71.3 36.5 47.3 52.4 MCTS(10,3) (MASPRM-7B) 82.9 85.0 88.2 51.8 57.5 60.5 75.2 76.7 77.4 45.5 50.5 52.5 SC@5 (policy likelihood) 61.2 73.3 81.4 41.5 50.6 57.5 54.5 69.2 74.9 35.9 46.5 55.4 SC@5 (MASPRM-7B) 67.8 77.0 81.4 49.2 55.3 57.5 67.2 72.7 74.9 43.0 50.2 55.2 SBS(5,3) (policy likelihood) 60.3 66.8 – 40.4 45.6 – 59.4 59.7 – 37.0 40.5 – SBS(5,3) (MASPRM-7B) 77.5 79.5 – 51.3 55.8 – 74.8 77.6 – 45.0 47.7 –

- Table 6: Cross-LLM results with Llama3.2 3B-Instruct agents and fixed Qwen2.5-trained reward models (no retraining): Hit@1 (%), TOK (×103), and AC.

Search Scorer

GSM8K MATH MMLU LOGIQA Hit@1 TOK AC Hit@1 TOK AC Hit@1 TOK AC Hit@1 TOK AC Single-Pass – 52.6 0.8 4 35.2 1.2 4 56.0 0.1 3 36.3 0.5 3

SC@5 Policy Likelihood 64.2 3.7 20 44.1 5.6 20 56.4 0.3 15 37.2 2.6 15 SC@5 MASPRM-7B 69.8 3.7 20 48.6 5.6 20 66.7 0.3 15 43.3 2.6 15

SBS(5,1) Policy Likelihood 58.4 3.8 20 42.1 5.5 20 55.0 0.3 15 36.1 2.5 15 SBS(5,1) MASPRM-7B 74.6 3.7 20 49.1 5.5 20 71.6 0.3 15 44.1 2.5 15

SBS(5,3) Policy Likelihood 63.1 8.3 50 41.3 12.3 50 58.5 0.8 35 37.6 6.5 35 SBS(5,3) MASPRM-7B 78.2 8.5 50 52.2 12.5 50 74.3 0.8 35 45.4 6.5 35

MCTS(10,3) Policy Likelihood 64.1 5.9 38.7 43.4 9.8 34.5 61.9 0.2 12.8 37.4 2.5 13.9 MCTS(10,3) ORM-1.5B 68.6 5.9 37.3 46.7 9.5 34.5 67.6 0.2 13.4 41.5 2.2 13.0 MCTS(10,3) MASPRM-7B 75.3 5.4 35.4 52.1 9.7 37.8 74.9 0.3 13.1 46.1 2.6 14.0

- Table 7: Head-type ablation (Hit@1, %) under MCTS(10,3) with Qwen2.5-1.5B-Instruct agents.

Head GSM8K MATH MMLU LOGIQA ORM

BCE 66.8 45.3 66.2 40.2 MSE 66.1 44.5 65.4 39.6

MASPRM-1.5B

BT 69.8 47.3 68.8 42.2 MSE 65.5 44.1 67.6 41.9

MASPRM-7B

BT 82.9 51.8 75.2 45.5 MSE 73.3 50.6 74.9 44.4

- Table 8: GSM8K graph sensitivity (Hit@1, %) across MAS1-MAS3.

#### C.6 Wall-clock Comparison

Each AC takes 18.94s on average, while each RC takes 0.16s on average for GSM8K dataset. Estimated per-instance wall-clock on GSM8K: MCTS(10,3) ≈ 11min; SBS(5,3) ≈ 16min.

### D MAS Graph, Prompts, and Configuration

This work uses multiple MAS instantiations (varying in depth D, agent roles, and directed communication graphs) rather than a single fixed MAS for all settings. The overall design choices are inspired by: decomposition/plan-then-solve prompting (Wang et al., 2023), self-refinement/self-critique loops (Madaan et al., 2023), and multi-agent debate (Irving et al., 2018). Below we list the exact YAMLstyle configurations used, plus the GSM8K MAS3 consensus configuration introduced for the graph sensitivity ablation. In these listings, agent IDs are 0-indexed, and we use −1 to denote the question node (q in Sec. 3).

Method MAS1 MAS2 MAS3

Single-Pass 47.2 54.4 56.2 SBS(5,1)+MASPRM 73.4 73.7 75.2 SBS(5,3)+MASPRM 77.5 78.5 79.7 MCTS(10,3)+MASPRM 82.9 81.4 80.9

D.1 Agent roles, system prompts, and decoding

Each MAS is specified by a list of agents (name, system prompt, and a per-agent decoding cap). For readability, we present each MAS’s agents block.

#### MASSingle: single-agent baseline.

- name: Assistant system_prompt: |

You are an intelligent assistant that

. . . helps users by providing accurate

. . . and concise answers. Solve the problem carefully. Output exactly: Final Answer: <your answer>

max_new_tokens: 1024

This is a standard single-pass baseline aligned with chain-of-thought style prompting (Wei et al., 2022).

MASRPSV (GSM8K MAS1/MATH): Reader → Planner → Solver → Verifier.

- name: Reader system_prompt: "You are the Reader. Extract

. . . key facts and restate the question

. . . ." max_new_tokens: 1024

- name: Planner system_prompt: "You are the Planner. Propose

. . . a concise plan to solve the task

. . . ." max_new_tokens: 1024

- name: Solver system_prompt: "You are the Solver. Carry

. . . out the plan and compute results." max_new_tokens: 1024

- name: Verifier system_prompt: "You are the Verifier. Double

. . . -check the result and produce the

. . . final answer only." max_new_tokens: 1024

This is a plan-then-solve decomposition with an explicit verification stage (Wang et al., 2023).

MASRefine (GSM8K MAS2): Solver → Evaluator → Reflector → Reviser.

- name: Agent_Solver system_prompt: |

You are a math solver. Solve the problem

. . . step by step. First show your work, then give the final

. . . answer.

End with: FINAL ANSWER: [your answer] max_new_tokens: 512

- name: Agent_Evaluator system_prompt: |

You are a checker. Look at the solution

. . . and check if it is correct. Check:

- - Are the steps logical?
- - Is the math correct?

Reply with only one of these: CORRECT or INCORRECT: [explain the error]

max_new_tokens: 256

- name: Agent_Reflector system_prompt: |

You are an advisor. A solution was wrong. Give one short tip to fix the mistake. Do not solve the problem yourself. Reply with: TIP: [your advice]

max_new_tokens: 256

- name: Agent_Reviser system_prompt: |

You are a math solver. A previous answer

. . . was wrong. Use the tip given to solve the problem

. . . correctly. Show your work step by step. End with: FINAL ANSWER: [your answer]

max_new_tokens: 512

This configuration implements an explicit feedback→refine loop (Madaan et al., 2023).

MASConsensus (GSM8K MAS3): Reader → SolverA → SolverB → Judge.

- name: Reader system_prompt: "You are the Reader. Extract

. . . key facts and restate the question

. . . ." max_new_tokens: 512

- - name: Solver_A system_prompt: |

You are Solver A. Solve the problem step

. . . by step. Be careful with arithmetic. End with: FINAL ANSWER: <number>

max_new_tokens: 1024

- - name: Solver_B system_prompt: |

You are Solver B. Independently solve the

. . . same problem from scratch. Try a different approach if possible and

. . . be careful with arithmetic. End with: FINAL ANSWER: <number>

max_new_tokens: 1024

- name: Judge system_prompt: |

You are the Judge/Verifier. You receive:

- (i) the original question,
- (ii) the Reader summary,
- (iii) Solver_A's solution,
- (iv) Solver_B's solution.

If both solvers agree and the reasoning is

. . . sound, output exactly: Final Answer: <value> If they disagree or either contains an

. . . error, identify the mistake and

. . . recompute cleanly. Then output exactly: Final Answer: <value>

max_new_tokens: 512

This configuration is motivated by self-consistency style aggregation and final answer selection (Wang et al., 2022).

MASDebate: Angel/Devil debate with Judge aggregation (MMLU).

- name: Angel system_prompt: |

You are a helpful debater. Read the

. . . multiple-choice question with

. . . options A-D.

Reason briefly and choose the best option. max_new_tokens: 1024

- name: Devil system_prompt: |

You are a skeptical debater. Read the same

. . . question. Challenge assumptions and choose the best

. . . option. max_new_tokens: 1024

- name: Judge system_prompt: |

You are the Judge. Read the question and

. . . both debaters' letters. Decide the final answer. Respond EXACTLY with this format: Final Answer: X where X is one of A, B, C, or D.

max_new_tokens: 1024

This configuration follows debate and judge style aggregation (Irving et al., 2018).

MASLogic: constraint extraction → world building → option checking (LogiQA).

- name: Constraint_Extractor system_prompt: |

You are the Constraint Extractor. Ignore the answer options for now. Extract all explicit constraints and

. . . logical rules stated in the

. . . passage. Format them as a numbered list (e.g., "1)

. . . A != B", "2) If C then D"). max_new_tokens: 1024

- name: World_Builder system_prompt: |

You are the World Builder. You receive a list of constraints. Construct one or more concrete "valid

. . . states" that satisfy all

. . . constraints.

If multiple states are possible, describe . . . what must be true in all valid

. . . states. max_new_tokens: 1024

- name: Query_Checker system_prompt: |

You are the Query Checker. You receive the question (with options) . . . and the valid state summary.

Determine which option is entailed by (or . . . most consistent with) the valid

. . . state(s).

Output exactly: Final Answer: X where X is one of A, B, C, or D.

max_new_tokens: 1024

This configuration uses decomposition into structured intermediate artifacts (Zhou et al., 2022).

#### D.2 Communication edges and schedule

Edges are directed and −1 denotes the root (question). For each MAS, the schedule order is one turn per agent in the order listed in the agents block.

MASSingle schedule. Schedule order: [Assistant]. Edges:

Edges

edges:

- [-1, 0]

MASRPSV schedule. Schedule order: [Reader, Planner, Solver, Verifier]. Edges:

Edges

edges:

- - [-1, 0]
- - [-1, 1]
- - [0, 1]
- - [1, 2]
- - [-1, 3]
- - [2, 3]

MASRefine schedule. Schedule order: [Agent_Solver, Agent_Evaluator, Agent_Reflector, Agent_Reviser].

Edges: Edges

edges:

- - [-1, 0]
- - [0, 1]
- - [0, 2]
- - [1, 2]
- - [-1, 3]
- - [2, 3]

MASConsensus schedule. Schedule order: [Reader, Solver_A, Solver_B, Judge]. Edges:

Edges

edges:

- - [-1, 0]
- - [-1, 1]
- - [-1, 2]
- - [ 0, 1]
- - [ 0, 2]
- - [-1, 3]
- - [ 0, 3]
- - [ 1, 3]
- - [ 2, 3]

MASDebate schedule. Schedule order: [Angel, Devil, Judge]. Edges:

Edges

edges:

- - [-1, 0]
- - [-1, 1]
- - [-1, 2]
- - [0, 2]
- - [1, 2]

MASLogic schedule. Schedule order: [Constraint_Extractor, World_Builder, Query_Checker].

Edges: Edges

edges:

- - [-1, 0]
- - [0, 1]
- - [-1, 2]

- [1, 2]

### E Algorithms (Brief Overview)

This appendix summarizes the procedures used for MAS-MCTS data generation during training, MCTS(Nsim,Cmax) with MASPRM during inference, and SBS(B2,B1). The key difference between training and inference MCTS is the leaf evaluation: training backs up the ground truth terminal reward R(sT) (no virtual visits), whereas inference bootstraps with Vϕ and initializes new children with virtual visits. For the pseudocode below, when in state s = (x,Ht−1,it), the acting agent is it and π(it)(· | x,Ht(−it1)) denotes its policy conditioned on its local view as defined in Sec. 3.

- Algorithm 1 MAS-MCTS(Nsim,Cmax) Data Generation During Training Input: Instance (x,y⋆), graph G, schedule σ, simulations Nsim, candidate cap Cmax, exploration cUCT

- 1: Initialize empty search tree with statistics N(s,a)=0, W(s,a)=0
- 2: for k = 1 to Nsim do
- 3: s ← root(x)
- 4: path ← [] ▷ Selection
- 5: while s is expanded and non-terminal do
- 6: a ← arg maxb Q ˆ(s,b) + cUCT ln(1+1+N u(Ns,b()s,u))

- 7: Append (s,a) to path
- 8: s ← next(s,a)
- 9: end while ▷ Expansion (no virtual visits in training)
- 10: if s non-terminal then
- 11: Sample up to Cmax candidates C from the acting agent’s policy π(it)(· | x,Ht(−it1)) where s = (x,Ht−1,it)
- 12: For each a ∈ C, create child s′ = next(s,a)
- 13: end if ▷ Evaluation with ground truth (terminal reward)
- 14: Continue selection/expansion until reaching a terminal leaf sT
- 15: v ← R(sT) ∈ {−1,+1} ▷ Backpropagation
- 16: for each (u,b) in path do
- 17: N(u,b)←N(u,b) + 1
- 18: W(u,b)←W(u,b) + v
- 19: Qˆ(u,b)← WN((u,bu,b))

- 20: end for
- 21: end for
- 22: Process targets: For every visited edge (s,a) with child s′ = next(s,a), set yproc(s′) ← Qˆ(s,a) and add (s′,yproc(s′)) to Dproc
- 23: Fit the process head: Form preference pairs (s+,s−) with yproc(s+) > yproc(s−) and train Vϕ with a Bradley-Terry ranking loss

- Algorithm 2 MCTS(Nsim,Cmax) with MASPRM During Inference

Input: Instance x, graph G, schedule σ, simulations Nsim, candidate cap Cmax, exploration cUCT, value

model Vϕ

- 1: Initialize empty search tree with statistics N(s,a)=0, W(s,a)=0
- 2: for k = 1 to Nsim do
- 3: s ← root(x)
- 4: path ← [] ▷ Selection
- 5: while s is expanded and non-terminal do
- 6: a ← arg maxb Q ˆ(s,b) + cUCT ln(1+1+N u(Ns,b()s,u))

- 7: Append (s,a) to path
- 8: s ← next(s,a)
- 9: end while ▷ Expansion with virtual visit initialization from Vϕ
- 10: if s non-terminal then
- 11: Sample up to Cmax candidates C from π(it)(· | x,Ht(−it1)) where s = (x,Ht−1,it)
- 12: for each a ∈ C do
- 13: s′ ← next(s,a)
- 14: N(s,a)←1
- 15: W(s,a)←Vϕ(s′)
- 16: Qˆ(s,a)←Vϕ(s′)
- 17: end for
- 18: end if ▷ Leaf evaluation by bootstrap
- 19: v ← Vϕ(s) ▷ Backpropagation
- 20: for each (u,b) in path do
- 21: N(u,b)←N(u,b) + 1
- 22: W(u,b)←W(u,b) + v
- 23: Qˆ(u,b)← WN((u,bu,b))

- 24: end for
- 25: end for
- 26: Decode: From the root, greedily follow at each depth the child with maximal Qˆ

- Algorithm 3 Step-Level Beam Search (SBS(B2,B1)) with MASPRM or Policy Score Input: Instance x, graph G, schedule σ, global beam B1, samples per state B2, scorer f

- 1: Initialize beam B ← {root(x)}
- 2: for t = 1 to D do ▷ D is the schedule length (one agent per depth, it = σ(t))
- 3: Call ← []
- 4: for each s = (x,Ht−1,it) ∈ B do
- 5: Sample B2 candidate actions {a(b)}Bb=12 ∼ π(it)(· | x,Ht(−it1))
- 6: for b = 1 to B2 do
- 7: s′ ← next(s,a(b))
- 8: score(s′) ← f(s′) (e.g., f = Vϕ)
- 9: Append (s′,score(s′)) to Call
- 10: end for
- 11: end for
- 12: B ← top-B1 states in Call by score ▷ Global pool across parents
- 13: Early stop: If any s ∈ B is terminal, optionally stop and return the best terminal by score
- 14: end for
- 15: Select output: If no terminal in the final beam, select the highest scoring state and finalize with the last scheduled agent

