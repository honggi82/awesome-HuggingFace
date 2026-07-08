# arXiv:2411.01493v2[cs.LG]9Nov2024

## SAMPLE-EFFICIENT ALIGNMENT FOR LLMS

Zichen Liu1,2 Changyu Chen1,3 Chao Du1† Wee Sun Lee2 Min Lin1 1Sea AI Lab 2National University of Singapore 3Singapore Management University {liuzc,chency,duchao,linmin}@sea.com {zichen,leews}@comp.nus.edu.sg

ABSTRACT

We study methods for efficiently aligning large language models (LLMs) with human preferences given budgeted online feedback. We first formulate the LLM alignment problem in the frame of contextual dueling bandits. This formulation, subsuming recent paradigms such as online RLHF and online DPO, inherently quests for sample-efficient algorithms that incorporate online active exploration. Leveraging insights from bandit theory, we introduce a unified algorithm based on Thompson sampling and highlight its applications in two distinct LLM alignment scenarios. The practical agent that efficiently implements this algorithm, named SEA (Sample-Efficient Alignment), is empirically validated through extensive experiments across three model scales (1B, 2.8B, 6.9B) and three preference learning algorithms (DPO, IPO, SLiC). The results demonstrate that SEA achieves highly sample-efficient alignment with oracle’s preferences, outperforming recent active exploration methods for LLMs. Additionally, we release the implementation of SEA together with an efficient codebase designed for online alignment of LLMs, aiming to accelerate future research in this field.

https://github.com/sail-sg/oat

Winratev.s.referenceresponses

1.0

SFT

Online DPO

+176%

Offline DPO

SEA DPO

0.8

+142%

+147%

+205%

+124%

0.6

+110%

+113%

+89%

0.4

+84%

0.2

1B 2.8B 6.9B

Model size

QueriesrequiredbyPassive

Fewer better

50k

| |
|---|

40k

| |
|---|

| |
|---|

30k

| |
|---|

20k

Passive

APL SEA

| |
|---|

10k

XPO

0

0 10k 20k 30k 40k 50k

Queries required by alternatives

Figure 1: Win rate comparison of model responses against reference responses on the TL;DR task, judged by the preference oracle. All compared methods use the same optimization method (DPO). (Left) Performance improvements at convergence over SFT models achieved by offline (Offline DPO), passively online (Online DPO), and our active exploration (SEA DPO) methods. (Right) The number of queries required by the passively online method (Passive) versus that by different active exploration methods to attain various levels of win rates. SEA achieves the best sample efficiency for online alignment compared to XPO and APL.

1 INTRODUCTION

Aligning LLMs with human preferences is a crucial step to elicit various desirable behaviors, e.g., helpfulness and harmlessness (Bai et al., 2022). Moreover, it holds the potential to create superhuman capabilities with only human-level feedback, as verifying is believed to be easier than synthesizing novel behaviors. By iteratively generating massive new candidates and asking for human feedback, LLMs could learn to reinforce good behaviors and may eventually surpass human capabilities.

Existing methods, either via reinforcement learning from human feedback (RLHF) (Stiennon et al., 2020; Ouyang et al., 2022) or direct alignment from preferences (DAP) (Rafailov et al., 2023; Azar

†Corresponding author.

et al., 2024), typically require a large amount of human annotations to achieve effective alignment. As a result, the volume of human feedback becomes a major bottleneck in practical alignment scenarios. This poses a challenging and under-explored research question:

##### How to align LLMs sample-efficiently?

To seek answers, in Section 2, we formalize LLM alignment as a contextual dueling bandit (CDB) (Yue et al., 2012; Dudík et al., 2015), where the agent (i.e., the learner and decision maker, in our case the LLM) interacts with the environment (i.e., human) to collect experience for improving its policy. This formulation naturally calls for two key properties for alignment algorithms to be sample-efficient:

- Property 1 (online interaction). Interacting and learning online allows the agent to act with the latest learned policy and then use that experience to immediately improve the policy.
- Property 2 (active exploration). An actively exploring agent strategically selects actions such that the collected experience leads to maximal policy improvement.

Since the CDB formulation is general and almost subsumes all existing LLM alignment methods, it provides us a lens to scrutinize prior methods on the axes of Properties 1 and 2. In Section 3, we thoroughly discuss prior alignment approaches, ranging from offline learning (Rafailov et al., 2023; Azar et al., 2024) and passive learning with iterative (Christiano et al., 2017; Dong et al., 2024) or online interaction (Guo et al., 2024), to active exploration for learning preference models (Dwaracherla et al., 2024) or aligning LLMs (Muldrew et al., 2024; Zhang et al., 2024a; Xie et al., 2024). As will be revealed, most prior methods (partially) fail to satisfy the two properties, resulting in inferior sample efficiency. Moreover, through the CDB formulation, we identify two LLM alignment scenarios, namely aligning from online users’ feedback (e.g., ChatGPT (2024)) and aligning from crowdsourcing (Christiano et al., 2017; Ouyang et al., 2022), and shed light on their correspondences to two bandit settings (explore & exploit and best arm identification). Understanding their differences is important for designing efficient alignment algorithms for respective scenarios. We detail these two settings in Section 2 and discuss how prior works approach them in Section 3.

Leveraging algorithmic insights from bandit theory, our answer to the research question above is a principled alignment algorithm based on Thompson sampling (TS) (Thompson, 1933). Our method fulfills Properties 1 and 2 to enhance sample efficiency, and it solves either of the two settings depending on practical scenarios (Section 4.1). We incorporate techniques including epistemic reward model, policy-guided search and mixed preference learning to implement the proposed TS algorithm (Section 4.2), yielding a practical agent which we call SEA (Sample-Efficient Alignment). In addition, we develop and open source a highly efficient, distributed learning system for studying online LLM alignment methods (Section 5), eliminating barriers to fair empirical comparisons of different alignment algorithms. Through extensive experiments (Section 6), SEA shows strong empirical results (see Figure 1), consistently achieving higher win rates and improved sample efficiency compared to baseline approaches across three model scales. We hope our open-sourced codebase and proposed algorithm could inspire future research in sample-efficient online LLM alignment.

2 LLM ALIGNMENT AS CONTEXTUAL DUELING BANDITS

We first review the definitions and two typical objectives of Contextual Dueling Bandits (Sec-

- tion 2.1), then translate them into the language of LLM alignment (Section 2.2). The tight connection between them, as we will see, allows us to leverage insights from bandit algorithms to design efficient alignment algorithms for LLMs.

- 2.1 CONTEXTUAL DUELING BANDITS

Contextual dueling bandits (CDB) (Yue et al., 2012; Dudík et al., 2015) is proposed to study online learning problems where the feedback consists of relative pairwise comparisons. A CDB problem can be characterized by a tuple (C,A,P), where C is the context space, A is the action space, and P : A×A×C  → [0,1] denotes the unknown preference oracle. An agent learns by iteratively interacting with the environment (i.e., the preference oracle P) as follows. At each round t of the learning process, a context ct ∼ pC is presented to the agent, who needs to take two actions at,a′t ∈ A for a “dueling” comparison. The agent then receives stochastic feedback in the form of a comparison result zt ∼ Ber(P(at ≻ a′t|ct)) from the environment, where Ber(·) is the Bernoulli distribution and ≻ denotes that the first action is preferred.

###### Contextual Dueling Bandits

###### LLM Alignment Interface

x ⇠ pX

c ⇠ pC

a,a0

Agent

- y,y0 Human

- z

How to cook oats Environment f

Agent

Can im in Python?

|you implementfor llamas?it| |
|---|---|
| | |

⇡✓ r 

#### ⇡

z

Dt = {x⌧,y⌧+,y⌧ }t⌧=1

Figure 2: Illustrative comparison between CDB and LLM alignment.

Regret. The quality of the dueling actions selected by the agent is measured by the immediate regret: Rt = P(a⋆t ≻ at|ct) + P(a⋆t ≻ a′t|ct) − 1, where a⋆t is the best action1 the agent would take at round t if it had complete knowledge of P. Intuitively, if the agent has learned how to act optimally from round t onwards, it would no longer suffer any regret since its actions would be indistinguishable from the best action (P(a⋆τ ≻ aτ|cτ) = 21 hence Rτ = 0 for τ ≥ t).

Optimal policy. A policy π ∈ ∆CA2 associates each context c ∈ C with a probability distribution π(·|c) ∈ ∆A over the action space. The total preference of policy π over policy µ given a context sampling distribution pC ∈ ∆C and a preference oracle P is defined as

Ea∼π(·|c)Ea′∼µ(·|c) [P(a ≻ a′|c)] . (1) We adopt the von Neumann winner (Dudík et al., 2015) as the solution concept, which requires the optimal policy π⋆ to satisfy that

C,P(π ≻ µ) = Ec∼p

Pp

C

- 1

- 2

∀π′ ∈ ∆CA, Pp

C,P(π⋆ ≻ π′) ≥

. (2)

In words, the von Neumann winner policy should beat or tie with every policy (i.e., is zero-regret) on average.

Learning objectives. The goal of bandit agents is to learn an optimal policy through interactions with the environment. There are two subtypes of objectives that focus on different learning scenarios. The first type considers the conventional explore and exploit (E&E) setting (Robbins, 1952; Auer et al., 2002), where the agent learns fully online and tries to minimize the cumulative regret

over T rounds: Tt=1 Rt. The second type of objective concerns the best arm identification (BAI) setting (Bubeck et al., 2009; Audibert & Bubeck, 2010), where the agent is only evaluated offline on

its average performance, possibly at any round (a.k.a., anytime regret), and tries to learn the optimal policy with minimum interaction. Both settings call for effective online exploration strategies that satisfy Properties 1 and 2. Their differences will be made clearer with real scenarios in Section 2.2.

- 2.2 ALIGNMENT AS CDB

LLM alignment can be framed as a CDB problem with their correspondences illustrated in Figure 2. Specifically, at time t a text prompt (cf. context) xt ∈ X is sampled from a prompt distribution pX. Then, two distinct responses (cf. actions), yt,yt′ ∈ Y, are chosen by the agent, and presented to human annotators (cf. the environment) for preference ranking. The winning and losing responses are labeled as (yt+,yt−) based on a binary stochastic feedback zt. The agent is expected to behave optimally by pursuing either E&E or BAI objectives, with knowledge learned from the experience accumulated so far: Dt = {xτ,yτ+,yτ−}tτ=1. A standard assumption is that human preferences follow the Bradley-Terry (BT) model (Bradley & Terry, 1952):

exp(r⋆(xt,yt)) exp(r⋆(xt,yt)) + exp(r⋆(xt,y′

P(yt ≻ yt′|xt) =

= σ(r⋆(xt,yt) − r⋆(xt,yt′)), (3)

t))

where σ is the sigmoid function and r⋆ encodes human’s implicit reward. The immediate regret of LLM alignment can be rewritten as Rt = r⋆(xt,yt⋆) − (r⋆(xt,yt) + r⋆(xt,yt′))/2 with the BT assumption (Saha, 2021; Li et al., 2024), where yt⋆ is the best response for prompt xt given human’s implicit reward, i.e., r⋆(xt,yt⋆) ≥ r⋆(xt,y),∀y ∈ Y. The von Neumann winner policy is also redefined as

Ey∼π(·|x)[r⋆(x,y)] is the objective, (4)

π⋆ ∈ arg max

J(π), where J(π) = Ex∼p

X

π∈∆XY

- 1We assume that a best action a⋆ in the sense that P(a⋆ ≻ a|c) ≥ 12, ∀a ∈ A exists for all context c ∈ C.

- 2We denote by ∆CA the set of all mappings C  → ∆A, where ∆A denotes the set of all probability distribu-

tions over A.

Reinforcement Learning from Human Feedback

Direct Alignment from Preferences

Active Exploration with Reward Models

Sample-Efficient Alignment

[Figure 1]

[Figure 2]

⇡ ! ! Human

{y,y0}! Human !

{y,y0}! Human

! ! Human

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

!

[Figure 8]

z

z

z

z

[Figure 9]

!

!

!

!

sync

[Figure 10]

[Figure 11]

[Figure 12]

D

D

D

D

(a) (b) (c) (d)

Figure 3: Different paradigms for solving the LLM alignment problem in the CDB framework. Note that although all paradigms follow the LLM alignment interface (Figure 2) with the interaction loop, some are actually offline or iteratively online (i.e., loop only once or a few times). Detailed comparisons will be made in Sec-

- tion 3. We use colors to denote learnable components, RL optimizer, direct optimizer, and active exploration. rϕ denotes a point estimate of human’s implicit reward, while RΦ refers to an uncertainty-aware reward model.

by substituting Eq. (3) into Eq. (1) and maximizing Pp

X,P(π ≻ π⋆) towards 1/2.

The two settings in bandits have their respective applications in LLM alignment. (1) The E&E setting applies to the scenario of serving an LLM-based application online and aligning it continually with users’ preferences. In this setting, the agent needs to balance exploration with exploitation, thus the cumulative regret is of interest because the quality of every response matters. In fact, commercial systems like ChatGPT would strategically ask users to make a dueling comparison, while upholding the quality of both responses. Please see Figure 10 in Appendix E for an example. (2) The BAI setting corresponds to the other scenario where annotators are paid to provide human feedback (Christiano et al., 2017; Ouyang et al., 2022). The desideratum in this scenario is to align the LLM at the minimum labeling cost, while the quality of the dueling responses is not important as long as the experience helps sample-efficiently learn the von Neumann winner policy.

After formalizing LLM alignment in the framework of CDB and uncovering their tight connections, we next thoroughly discuss existing alignment methods in the CDB framework and reveal the sources of their sample inefficiencies.

- 3 HOW PRIOR WORKS (PARTIALLY) SOLVE LLM ALIGNMENT AS CDB

We first align the notations and terminology used in CDB with commonly referred ones in the LLM community. Previously, we used the term “agent” to denote the learner and decision maker, and referred to its overall behavior as the “policy” π (as in Eq. (4)), following the standard abstraction in RL (Sutton & Barto, 2018; Sutton et al., 2022). However, in the LLM literature, “policy” typically refers to the generative language model alone, excluding components like reward models (RMs) that the agent might additionally build (see Figure 2). To avoid confusion, from now on we use πθt to denote the generative language model (policy) and rϕt to denote the (optional) RM at time t, both of which are learned from preference data Dt collected up to time t. We will omit t when the time-indexing is not applicable (i.e., no online interaction) or not important in the context.

RLHF and DAP. Commonly adopted RLHF pipelines (Christiano et al., 2017; Stiennon et al., 2020; Bai et al., 2022; Ouyang et al., 2022) first learn a proxy RM with a negative log-likelihood loss:

Lr(ϕ|D) = −E(x,y+,y−)∼pD log σ rϕ x,y+ − rϕ x,y− , (5)

where D is collected by querying human annotators using a behavior policy πref (typically the supervised fine-tuned policy πsft). Afterwards, offline RL (Lange et al., 2012; Levine et al., 2020) is conducted to learn πθ with respect to the learned reward rϕ internally within the agent (Figure 3a). However, the learned model πθ might be inaccurate at regions out of the distribution (o.o.d.) of πref because little training data can be collected. An effective remedy is to incorporate a pessimistic term to combat the distributional shift, leading to a reformulation of the von Neumann winner policy objective in Eq. (4) as

πθ(y|x) πref(y|x) o.o.d. reward penalty

(6)

###### E

J(πθ) = E

−β log

rϕ(x,y)

x∼pX

y∼πθ(·|x)

estimated r⋆

[rϕ(x,y)] − βDKL(πθ(·|x)||πref(·|x)) , (7)

###### E

###### = E

x∼pX

y∼πθ(·|x)

which converts an online objective regarding the human’s implicit reward r⋆ to an offline objective regarding the proxy reward rϕ. The KL penalty in Eq. (7) is widely used for language model fine-tuning (Jaques et al., 2020; Xiong et al., 2024), and PPO (Schulman et al., 2017) has become a default RL optimizer to maximize the KL-regularized reward. However, the performance of RLHF is guaranteed only if the preference data D induced by πref adequately covers π⋆ (Zhu et al., 2023), which is often approximated by updating πref with the latest (improved) πθ for re-sampling a batch of online experience and repeating Eq. (5) and (7). Prior methods typically employ only a few iterations of online interaction with large batches (Xiong et al., 2024; Dong et al., 2024), which may compromise sample efficiency (Property 1).

Online RLHF is difficult due to the complexity and instability of RL optimizers. For example, Huang et al. (2024) openly reproduces offline RLHF scaling behaviors but requires many implementation tricks for training, highlighting the difficulties of an online counterpart. Fortunately, the introduction of DAP (or direct optimizers) largely simplifies and stabilizes the alignment process by conducting contrastive supervised learning directly on D (Figure 3b). While most DAP works focus on learning from a fixed offline preference dataset (Zhao et al., 2023; Rafailov et al., 2023; Azar et al., 2024; Meng et al., 2024; Zhang et al., 2024b), iterative DPO (Xu et al., 2023) observes improved results when allowing iterative online interaction. Guo et al. (2024) further propose OAIF to make DAP faithfully online, satisfying Property 1, and demonstrate that online learning prevents over-fitting and yields continual performance improvement. Nevertheless, it still employs passive exploration strategies (directly using y,y′ ∼ πθ), hindering sample efficiency (Property 2).

Online exploration in LLMs. A line of recent works (Mehta et al., 2023; Das et al., 2024; Melo et al., 2024; Dwaracherla et al., 2024) adopts the fully online bandit formulation and incorporates active exploration with uncertainty-aware RMs for response selection (Figure 3c). In particular, Mehta

- et al. (2023) consider the E&E setting and develop a UCB-style (Auer et al., 2002) algorithm; Das
- et al. (2024) instead select the dueling responses with the most uncertain preference estimate, targeting the BAI setting in a pure exploration way; unlike the above, Melo et al. (2024) view the problem from the angle of pool-based active learning and propose an acquisition function based on both entropy and epistemic uncertainty; finally, the work by Dwaracherla et al. (2024) is the closest to ours in the sense that they apply double Thompson sampling (DTS) (Wu & Liu, 2016) for exploration, but DTS is designed for the E&E setting while they evaluate anytime average performance as in the BAI setting. We will show in Section 6.3 that pure exploration by Das et al. (2024) is not the best choice for BAI, and the objective mismatch in Dwaracherla et al. (2024) could lead to suboptimal performance in respective settings. Meanwhile, all these works primarily focus on learning uncertainty-aware RMs online without updating LLM policies. Therefore, all responses are sam-

pled from a fixed proposal policy πβ (or even a fixed dataset), making the data coverage a critical concern.

Another line of research updates LLMs online while incorporating exploration. Zhang et al. (2024a) and Xie et al. (2024) independently propose to learn an optimistic RM to encourage exploration. They leverage the property of DPO (Rafailov et al., 2023) to reparameterize RM with policy and conclude with an extra optimistic term in the DPO loss function. Thus, their learning processes are like Figure 3b but with an optimistic direct optimizer. Muldrew et al. (2024) adopt the vanilla DPO loss but utilize the implicit reward margin to actively select dueling responses. Yet, these methods are tightly coupled with DPO and not compatible to other direct optimizers. Their experiments are also limited to a few online iterations, possibly due to the implementation difficulty of a faithfully online learning system. Given their relevance to our approach, we will reproduce them in a fully online manner for fair comparisons in Section 6.1. We summarize prior works in Table 2 in Appendix E.

- 4 SEA: SAMPLE-EFFICIENT ALIGNMENT FOR LLMS

In this section we present our online exploration agent SEA (Figure 3d). We first introduce a principled Thompson sampling algorithm inspired by bandit theory (Section 4.1), and then derive SEA as its practically efficient implementation (Section 4.2). Interestingly, SEA can also be viewed as an instantiation of a classical model-based RL architecture called Dyna (Sutton, 1990), for which we defer the discussion to Appendix B.

Algorithm 1 Thompson sampling for LLM alignment (intractable).

Input: Prompt distribution pX, unknown but queryable preference oracle P.

- 1: Initialize experience D0 ← ∅.
- 2: for t = 1, . . . , T do
- 3: Receive a prompt xt ∼ pX.
- 4: Sample r ∼ pr(·|Dt−1) and set yt ← arg maxb∈Yr(xt, b). // Select 1st response y. // E&E objective: aligning an online system.
- 5: repeat Sample r ∼ pr(·|Dt−1) and set yt′ ← arg maxb∈Yr(xt, b). // Select 2nd response y′.

until yt′ ̸= yt

// BAI objective: labeling via crowdsourcing.

- 6: Set yt′ ← arg maxb∈YV [σ (r(xt, yt) − r(xt, b))], // OR select 2nd response y′. where V [·] computes variance over the posterior pr(·|Dt−1).
- 7: Query P to label {yt, yt′}, and update experience Dt ← Dt−1 {xt, yt+, yt−}.

- 8: end for // See Algorithm 2 for a practical version.

- 4.1 THOMPSON SAMPLING FOR LLM ALIGNMENT

Thompson sampling (TS) (Thompson, 1933) is widely adopted for solving bandit problems at scale due to its efficiency and strong empirical performance in general online learning problems (Chapelle & Li, 2011; Russo et al., 2018). A bandit agent using Thompson sampling typically maintains and incrementally updates a posterior distribution of the oracle reward p(r|D). Meanwhile, the agent takes actions following a greedy policy with respect to a sampled RM: at = arg maxa r(a) with r ∼ pr(·|D). This simple yet effective algorithm naturally balances exploration and exploitation: when the agent has limited knowledge about the environment, the posterior estimate exhibits high uncertainty so that the sampled RM could guide the greedy policy to explore; after sufficient experience is gathered, the sampled RM approximates the oracle more closely, allowing the agent to exploit near-optimal policies.

In the context of LLM alignment, we leverage the BT assumption (Eq. (3)) to replace the preference oracle P with the implicit reward r⋆. This substitution enables us to model the reward posterior p(r|D) in the standard TS framework, preserving the probabilistic structure necessary for effective posterior sampling. Inspired by prior works (Wu & Liu, 2016; González et al., 2017) on noncontextual K-arm bandits and preferential Bayesian optimization problems, we generalize them for LLM alignment and develop a unified algorithm as shown in Algorithm 1. Note that we assume for now the LLM agent can be fully described by the posterior p(r|D), and we defer practical reward (rϕ) and policy (πθ) learning to Section 4.2.

As Algorithm 1 presents, the first response of the duel is always selected via standard TS (Line 4). The selection of the second response varies across different settings. Line 5 will be used for scenarios where preference feedback is collected from online users (the E&E setting). The dueling responses selected in this case will both try to maximize a sampled RM, so that the online user experience is warranted with best effort. However, such algorithm can have poor asymptotic performance for BAI problems (Russo, 2016), because sub-optimal responses with confidently high rewards might be tried for a long time at the expense of not exploring other potentially better choices. In light of this, Line 6 provides an alternative for scenarios where we could hire annotators for feedback and low-quality but exploratory responses are safe to try. Specifically, Line 6 selects the second response as the one that maximizes the variance of the preference (Eq. (3)) over the first response y. This variance quantifies the epistemic uncertainty of the RM, pointing the agent to the maximally informative direction to explore for better sample efficiency.

However, Algorithm 1 is yet to be practical for LLM alignment for three main reasons. First, computing and sampling from a reward posterior is intractable for nearly all RMs at LLM scale, which are mostly based on large transformers (Lambert et al., 2024). Second, even if we managed to approximate the reward posterior, the arg max operations for response selection are still intractable since the search space Y is discrete and massive for token sequences of arbitrary length. Last but not least, an LLM agent (Achiam et al., 2023; Touvron et al., 2023) typically consists in a generative model πθ (e.g., a transformer (Vaswani et al., 2017)), while the algorithm above is centered around a reward posterior p(r|D) that cannot be easily converted into a generative model.

- 4.2 PRACTICAL IMPLEMENTATION

- 4.2.1 EPISTEMIC REWARD MODEL FOR POSTERIOR SAMPLING

To implement active exploration with TS, we seek an efficient way to maintain and incrementally update the reward posterior p(r|D). We consider deep ensemble for our purpose, due to its capability to model epistemic uncertainty (Lakshminarayanan et al., 2017) and provable results when applied to TS in linear bandits (Qin et al., 2022). Specifically, we update a set of plausible RMs independently and online, using the preference data and a regularized negative log-likelihood loss:

LR(Φt|Dt) =

K

k=1

Lr(ϕtk|Dt) − λ||ϕtk − ϕ0k|| , (8)

where Lr is defined in Eq. (5), Φt = {ϕtk}Kk=1 contains the weights of the ensemble of size K, and λ controls the regularization towards individual initial weights ϕ0k to retain the diversity across ensemble members (Dwaracherla et al., 2024). In practice, we train K MLP heads on top of a pretrained and frozen transformer. We refer to the ensemble as the Epistemic Reward Model (ERM, denoted as RΦ), with which the posterior sampling (r ∼ pr(·|Dt)) simply amounts to randomly picking a ϕtk from Φt.

- 4.2.2 POLICY-GUIDED SEARCH TO APPROXIMATE arg max

With the ERM approximating the reward posterior, we need to further approximate the response selection steps (Lines 4 to 6) which generally take the form of arg maxb∈YU(b), where U absorbs the sampled prompt, the sampled RM, and optionally the selected first response (for BAI, Line 6). To obtain the maximum, bandit algorithms for large action spaces typically resort to an action optimization oracle (Katz-Samuels et al., 2020; Zhu et al., 2022), but they assume a linear structure of U with respect to b, which might be impractical for LLMs. Therefore, we instead replace the optimization over Y with sampling from a policy-guided distribution conditioned on U, πprior(·|x)exp(U(·)/η), which is appropriate since it favors responses y that approximately maximize U(y). In practice, for a given prompt xt, we sample M candidate responses from the prior policy πprior(·|xt) to construct a proposal set St = {yti}Mi=1. We then conduct a greedy search in St (taking η → 0) to identify the response yt (or yt′) that locally maximizes the utility function U, which is subsequently used in the duel. We also reuse the same St for different U functions at time t to save computation. The choice of πprior will be discussed in the next section.

- 4.2.3 ONLINE POLICY LEARNING FROM MIXED PREFERENCES

We finally resolve two remaining questions: (Q1) how to choose a sensible πprior at each time t and (Q2) how to get a good generative policy online. To this end, we propose a simple approach to approximately address both questions simultaneously. That is, we can utilize any direct optimizer to learn the policy πθt online with the following loss and use the latest online policy as πprior:

###### Lπ(θt|Bt,πref,F) = E(x,y+,y−)∼pBt Fθt(x,y+,y−,πref) , (9)

where Bt is a batch of preference data labeled by the oracle wherein the responses are proposed by πprior and selected by RΦt, F could be any DAP loss (see Appendix A for some examples), and πref is chosen to be πsft. Note that we use πθt as πprior at any time t, thus Bt is a batch of on-policy data. By contrastive training on these on-policy data, we leverage their orthogonal benefits to achieve maximal policy improvement (Tajwar et al., 2024; Tang et al., 2024).

Now that optimizing Eq. (9) yields a good online policy πθt (answering Q2), we need to assess whether πθt can serve as a suitable πprior for approximating the arg max in TS (Q1). If we optimize πθt with oracle preference data, St will be biased towards responses with high oracle reward r⋆. Bias towards high-r⋆ region is generally helpful because it aligns with arg maxb∈Yr(x,b) that seeks high-reward responses. However, optimizing πθt only with oracle data can average out the epistemic uncertainty of R, hindering the exploration efficiency. To mitigate this issue, we further align πθt with RΦt using the same direct optimizer to encourage πθt to propose high-rϕt

responses for individual rϕt

k

, leading to better approximation of arg maxb∈Yr(x,b) for any sampled r. To implement, we optimize Eq. (9) over a mixture distribution pBmix

k

, where

###### + (1 − γ)pBERM

###### = γpB

t

t

t

γ ∈ [0,1] controls the mixture ratio and BtERM = {xi,y˜i+,y˜i−}bi=1 consists of preference data labeled by randomly sampled individual ensemble members rϕt

. Interestingly, learning from mixed preferences further boosts sample efficiency because it utilizes the internal ERM to get pseudo labels instead of querying humans. This relates closely to model-based RL, for which we discuss further in Appendix B. We summarize our practical algorithm (Algorithm 2) in Appendix A.

k

- 5 EXPERIMENTAL SETUP

In this section, we elaborate the experimental setup employed to validate our algorithm and ensure fair comparisons with other online alignment baselines. We start by introducing the distributed learning system designed for experimenting with online LLM alignment using simulated human preferences (Section 5.1). Then, we provide key experimental details in Section 5.2, with a full description available in Appendix D.

- 5.1 DISTRIBUTED LEARNING SYSTEM

The interactive nature of LLM alignment necessitates an integrated online learning system that simulates the interface depicted on the right of Figure 2. The absence of a performant open-source online alignment system has restricted many existing works to only a few iterations of batch learning (Muldrew et al., 2024; Dong et al., 2024; Chen et al., 2024; Zhang et al., 2024a; Xie et al., 2024), which creates a mismatch with their theories that typically require a large number of online interaction rounds. Even worse, such absence also makes the comparison between different LLM exploration methods difficult, often restricting evaluations to the simplest iterative DAP baselines (Zhang et al., 2024a; Xie et al., 2024).

To fill this gap, we build a highly efficient learning system for experimenting with online LLM alignment algorithms. We notice that the computational bottleneck lies in online response sampling (i.e., autoregressive generation) and preference labeling (e.g., human, large RMs, or large LLMs), which mirrors the slow actor-environment interaction seen in RL systems. Inspired by distributed deep RL systems which spawn many actors or environments in parallel (Espeholt et al., 2018; Weng et al., 2022), we design an Actor-LearnerOracle architecture for online LLM alignment, which is depicted in Figure 4. The three types of workloads (i.e., actor, learner and oracle) are heterogeneous and require different optimization. In particular, we adopt vLLM (Kwon et al., 2023) for the actor to accelerate the autoregressive response generation. We also use DeepSpeed’s ZeRO (Rasley et al., 2020; Rajbhandari et al., 2020) strategies to enhance the memory efficiency of the learner. The updated model weights are broadcasted from the learner master to all actors after every optimizer step efficiently via NCCL, similar to Hu et al. (2024). Furthermore, to improve the scalability, we wrap the oracle RM as a service using Mosec (Yang et al., 2021b), which supports dynamic batching and parallel processing, to minimize preference query latency. Finally, we leverage DeepMind Launchpad (Yang et al., 2021a) to compose all workloads into a distributed program and adopt Plasma (Philipp & Robert, 2017) to efficiently transfer data across process boundaries.

Experience

Learner Workers

###### Actors

Parameters

vLLM

Query

DeepSpeed

Oracle RM

Learner Master

Mosec

Experience

Figure 4: The learning system for experimenting online LLM alignment algorithms.

We benchmark our system’s efficiency against a concurrent implementation of online DPO by HuggingFace3, which utilizes only DeepSpeed for memory optimization. Our system achieves up to 2.5× latency reduction compared to this counterpart, demonstrating its computational efficiency. Due to space constraints, detailed benchmarking methods and results are presented in Appendix C. Our codebase, oat (online alignment), along with the implementation of SEA, is open-sourced at https://github.com/sail-sg/oat to accelerate future research in online LLM alignment.

- 5.2 EXPERIMENT DETAILS

We adopt SFT models tuned on TL;DR (Stiennon et al., 2020) from Huang et al. (2024), which cover three scales (1B, 2.8B, 6.9B) of the Pythia family (Biderman et al., 2023), as starting points for our experiments. We choose Liu et al. (2024a) to be the oracle RM. To verify the effectiveness of SEA, we employ three direct optimizers: DPO (Rafailov et al., 2023), IPO (Azar et al., 2024),

3https://huggingface.co/docs/trl/main/en/online_dpo_trainer.

Pythia 1B

Pythia 2.8B

Pythia 6.9B

0.9

0.9

0.9

0.8

0.8

Winrate

0.8

0.7

DPO

0.7

0.7

0.6

Offline Online SEA

XPO APL

Offline Online SEA

XPO APL

Offline Online SEA

XPO APL

0.6

0.6

0.5

0.5

0.5

0.9

0.9

0.9

0.8

0.8

Winrate

0.8

0.7

IPO

0.7

0.7

0.6

0.6

0.6

Offline Online

SEA

Offline Online

SEA

Offline Online

SEA

0.5

0.5

0.5

0.9

0.9

0.9

0.8

0.8

0.8

Winrate

0.7

SLiC

0.7

0.7

0.6

0.6

0.6

Offline Online

SEA

Offline Online

SEA

Offline Online

SEA

0.5

0.5

0.5

0 10k 20k 30k 40k 50k

0 10k 20k 30k 40k 50k

0 10k 20k 30k 40k 50k

Query step

Query step

Query step

Figure 5: Win rate comparison of different algorithms against their initial SFT models across three scales and three direct optimizers.

and SLiC (Zhao et al., 2023). Besides, two LLM exploration methods built on DPO, APL (Muldrew et al., 2024) and XPO (Xie et al., 2024), are fairly compared when using DPO as the optimizer. Our experiments primarily focus on the BAI setting (crowdsourcing labeling), and we use win rate against reference responses as the metric. We refer readers to Appendix D for more details.

- 6 EMPIRICAL RESULTS

In this section, we present our empirical results and analyses, organized into four parts: (1) an overall comparison between SEA and baselines across various direct optimizers and model scales; (2) an ablation analysis to study the effects of SEA’s key components; (3) a comparison of different exploration strategies under E&E and BAI settings; (4) additional results for alignment with a human oracle simulated by GPT4o-mini.

- 6.1 OVERALL COMPARISON

We first compare SEA with all baselines across three model scales and three direct optimizers. APL and XPO are only compared when DPO is used as the direct optimizer, because they are incompatible with IPO or SLiC. Figure 5 shows the win rate curves versus the number of query steps. Across all settings, Online agents consistently improve sample efficiency over their Offline counterparts, validating the necessity of Property 1 for alignment algorithms. Focusing on the first row, we observe that among prior active exploration methods, XPO gives a small improvement in final performance over Online (passive) at the 1B scale, but falls short for larger scales. On the other hand, APL shows a significant sample efficiency boost at the 1B scale, but this advantage diminishes when scaling up and it performs almost the same as Online at 6.9B scale. Our method, SEA, outperforms both offline and online passive methods across all scales and all direct optimizers, confirming the critical role that Property 2 plays for sample-efficient alignment. Meanwhile, in the special case of using DPO as the direct optimizer, SEA also shows superior performance to prior online active exploration methods including APL and XPO. We invite readers to revisit Figure 1, where we show that SEA not only attains significantly improved final performance (Left) but also achieves 2-5× better sample efficiency (Right).

Additionally, we note that the choice of direct optimizer matters for both online learning and active exploration. When comparing different optimizers at 1B scale (the first column), all Offline agents demonstrate comparable learning efficiency and reach the same level of final performance

Table 1: Decomposition of different driving factors of online active alignment algorithms.

Variant Inference (Test) Exploration Learn Remark

- 1 πθ passive πθ Online DAP (Guo et al., 2024)
- 2 πθ active (πθ, RΦ) SEA without ERM sync (Section 4.2.3)
- 3 πθ active (πθ ↔ RΦ) SEA

- 4 BoN(πθ, RΦ) passive (πθ, RΦ) -
- 5 BoN(πθ, RΦ) active (πθ, RΦ) -
- 6 BoN(πθ, RΦ) active (πθ ↔ RΦ) SEA with Best-of-N sampling

- 7 BoN(πref, RΦ) active RΦ Not learn policy (Dwaracherla et al., 2024)

(around 70% win rate), but SLiC Online agent deliver slightly less improvement than DPO and IPO Online agents. Besides, when incorporating active exploration, the SEA agent using DPO shows much larger improvement than the other two. This suggests that selecting the most suitable policy optimizer coupled with active exploration would yield the best agent.

- 6.2 ABLATION ANALYSIS

Inference with policy

Inference with Best-of-N

Next, we decompose SEA into distinct components to evaluate their individual contributions. Table 1 shows the three axes we dissect SEA on, including inference methods, exploration strategies, and learning components. We construct seven agent variants from different combinations, which cover two closely related baselines (Guo et al., 2024; Dwaracherla et al., 2024). We show in Figure 6 the performance curves of each variant, all trained with DPO on 1B scale. The left plot compares variants that directly use the policy for inference. It clearly shows the benefits of learning ERM for active exploration (Variant-2) and aligning πθt with RΦt (Variant-3). Since a reward model is learned within the agent, we can further incorporate inference-time alignment via Best-of-N (BoN) sampling (Nakano et al., 2021; Touvron et al., 2023). This also facilitates a direct comparison between SEA and Dwaracherla et al. (2024), which learns a similar ERM for both exploration and BoN but does not align the LLM policy. Results in the right plot of Figure 6 suggest a similar trend that Variant-6 ≻ Variant-5 ≻ Variant-4. The Variant-7 (Dwaracherla et al., 2024), however, ceases to improve after ERM converges due to the limited performance of its fixed policy.

0.9

0.8

Winrate

0.7

0.6

- 4

- 5

6 7

1 2 3

0.5

0 10k 20k 30k 40k 50k

0 10k 20k 30k 40k 50k

Query step

Query step

Figure 6: Win rate comparison of different agent variants when using (Left) policy and (Right) Best-of-N sampling for inference.

- 6.3 CHOICE OF EXPLORATION STRATEGIES

Recalling that different LLM alignment scenarios (online system or crowdsourcing) require different exploration strategies to meet their respective learning objectives (Section 2.2). We investigate three strategies based on posterior sampling and compare them on both online and offline performance. The first strategy (Uncertainty) focuses on pure exploration with information maximization. It seeks the pair of dueling responses that exhibits the largest epistemic uncertainty, which is implemented by selecting the pair whose logits difference has the largest variance across ensemble members. The second (E&E-TS) and the third (BAI-TS) strategies follow the principles in Algorithm 1, and their differences are between Line 5 and Line 6. The comparison results are shown in Figure 7 (Left and Middle). Focusing on the left plot, we observe that E&E-TS strategy achieves the best online performance, which is within our expectation. In contrast, Uncertainty shows the worst online performance because it tries to maximize the information gain but does not prioritize reward maximization. On the other hand, conclusions are interestingly different when taking the offline performance as the metric. In this case, BAI-TS and Uncertainty both exhibit more efficient offline performance improvement than E&E-TS. This can be attributed to that exploration for uncertainty minimizing helps to identify more informative responses to train the LLM policy. Moreover,

GPT4o-mini-as-a-judge

Online performance (E&E)

Offline performace (BAI)

0.9

0.8

Winrate

0.7

Uncertainty

Uncertainty

0.6

E&E TS

E&E TS

Offline Online

XPO APL

SEA

BAI TS

BAI TS

0.5

0 10k 20k 30k 40k 50k

0 10k 20k 30k 40k 50k

0 10k 20k 30k 40k 50k

Query step

Query step

Query step

Figure 7: (Left and Middle) Win rate comparison of different exploration strategies measured in E&E and BAI settings. (Right) Win rate comparison of different agents when using GPT4o-mini to simulate human feedback via LLM-as-a-judge.

BAI-TS ≻ Uncertainty indicates exploration with both reward and information maximization is better than exploration with only information maximization. E&E-TS, however, always chooses two responses with similarly high quality to exploit. This can not only lead to less efficient exploration, but also result in less efficient policy learning due to smaller DAP loss gradients.

- 6.4 ALIGNING LLMS WITH A HUMAN SIMULATOR

Results presented so far are based on experimenting LLM alignment with the preference oracle being a scalar reward model, which is deterministic and does not capture the potential randomness of the choice by real humans. To test different agents in a more realistic setting, we use generative models as human simulator in an LLM-as-a-judge (Bubeck et al., 2023; Zheng et al., 2023) manner. In particular, we directly query the OpenAI API and use the gpt-4o-mini-2024-07-18 model as the judge to provide preference feedback. We follow the prompt template of Li et al. (2023). The results are shown in Figure 7 (Right). We can observe the performance curves generally exhibit higher variance, possibly due to the randomness introduced in the feedback process, which puts more stringent requirements for learning algorithms. The two active exploration methods demonstrate opposite results to those in Section 6.1—APL learns fast initially but is eventually outperformed by Online, while XPO improves over Online after stabilizing its training and delivers a better final performance. Our agent, SEA, is shown to offer the best sample efficiency as well as asymptotic performance, further validating the importance of online learning and well-designed active exploration mechanism.

- 7 CONCLUSION

In this paper, we study the problem of LLM alignment through the lens of contextual dueling bandits and propose a Thompson sampling-based algorithm to achieve sample-efficient alignment. We incorporate three techniques, including epistemic reward model, policy-guided search and mixed preference learning to yield a practically efficient online alignment method. Extensive empirical evaluation demonstrates the superior sample efficiency of our method compared to existing baselines. To our knowledge, this is the first work to study active exploration for online LLM alignment with fully online experimental verification. We hope our positive empirical results, along with the open-sourced codebase, will encourage future research in this direction, ultimately enabling LLMs to achieve superhuman intelligence with an affordable amount of human feedback.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jean-Yves Audibert and Sébastien Bubeck. Best arm identification in multi-armed bandits. In Conference on learning theory, pp. 41–53, 2010.

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47:235–256, 2002.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław De˛biak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Sébastien Bubeck, Rémi Munos, and Gilles Stoltz. Pure exploration in multi-armed bandits problems. In Algorithmic Learning Theory: 20th International Conference, ALT 2009, Porto, Portugal, October 3-5,

2009. Proceedings 20, pp. 23–37. Springer, 2009.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

Róbert Busa-Fekete, Balázs Szörényi, Paul Weng, Weiwei Cheng, and Eyke Hüllermeier. Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm. Machine learning, 97:327–351, 2014.

Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. Advances in neural informa-

tion processing systems, 24, 2011. OpenAI ChatGPT. ChatGPT. https://chatgpt.com/, 2024. Accessed: 2024-09-30. Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakantham, and Min Lin. Bootstrapping language models with dpo implicit rewards. arXiv preprint arXiv:2406.09760, 2024. Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement

learning from human preferences. Advances in neural information processing systems, 30, 2017. Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano, and Sayak Ray Chowdhury. Provably sample efficient rlhf via active preference optimization. arXiv preprint arXiv:2402.10500, 2024.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Miroslav Dudík, Katja Hofmann, Robert E Schapire, Aleksandrs Slivkins, and Masrour Zoghi. Contextual dueling bandits. In Conference on Learning Theory, pp. 563–587. PMLR, 2015.

Vikranth Dwaracherla, Seyed Mohammad Asghari, Botao Hao, and Benjamin Van Roy. Efficient exploration for llms. In International Conference on Machine Learning, 2024.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International conference on machine learning, pp. 1407–1416. PMLR, 2018.

Javier González, Zhenwen Dai, Andreas Damianou, and Neil D Lawrence. Preferential bayesian optimization. In International Conference on Machine Learning, pp. 1282–1291. PMLR, 2017.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792, 2024.

Jian Hu, Xibin Wu, Weixun Wang, Dehao Zhang, Yu Cao, et al. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

Shengyi Huang, Michael Noukhovitch, Arian Hosseini, Kashif Rasul, Weixun Wang, and Lewis Tunstall. The n+ implementation details of rlhf with ppo: A case study on tl; dr summarization. arXiv preprint arXiv:2403.17031, 2024.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. Advances in neural information processing systems, 32, 2019.

Natasha Jaques, Judy Hanwen Shen, Asma Ghandeharioun, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Shane Gu, and Rosalind Picard. Human-centric dialog training via offline reinforcement learning. arXiv preprint arXiv:2010.05848, 2020.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models with pairwise comparison and generative fusion. In Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (ACL 2023), 2023.

Julian Katz-Samuels, Lalit Jain, Kevin G Jamieson, et al. An empirical process approach to the union bound: Practical algorithms for combinatorial and linear bandits. Advances in Neural Information Processing Systems, 33:10371–10382, 2020.

Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Model-based offline reinforcement learning. Advances in neural information processing systems, 33:21810–21823, 2020.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. Rewardbench: Evaluating reward models for language modeling, 2024.

Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. In Reinforcement learning: State-of-the-art, pp. 45–73. Springer, 2012.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https: //github.com/tatsu-lab/alpaca_eval, 5 2023.

Xuheng Li, Heyang Zhao, and Quanquan Gu. Feel-good thompson sampling for contextual dueling bandits. arXiv preprint arXiv:2404.06013, 2024.

Chris Yuhao Liu, Liang Zeng, Liu Jiacai, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork reward model series. arXiv preprint arXiv:2410.18451, 2024a.

Zichen Liu, Siyi Li, Wee Sun Lee, Shuicheng Yan, and Zhongwen Xu. Efficient offline policy optimization with a learned model. In International Conference on Learning Representations, 2023.

Zichen Liu, Chao Du, Wee Sun Lee, and Min Lin. Locality sensitive sparse encoding for learning world models online. In International Conference on Learning Representations, 2024b.

Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, and Willie Neiswanger. Sample efficient reinforcement learning from human feedback via active exploration. arXiv preprint arxiv:2312.00267, 2023.

Luckeciano C Melo, Panagiotis Tigas, Alessandro Abate, and Yarin Gal. Deep bayesian active learning for preference modeling in large language models. arXiv preprint arXiv:2406.10023, 2024.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

William Muldrew, Peter Hayes, Mingtian Zhang, and David Barber. Active preference learning for large language models. In International Conference on Machine Learning, 2024.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Andrew Y Ng and Stuart Russell. Algorithms for inverse reinforcement learning. In International Conference on Machine Learning, volume 1, pp. 2, 2000.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Moritz Philipp and Nishihara Robert. Plasma: A high-performance shared-memory object store, 2017. URL https://arrow.apache.org/blog/2017/08/08/plasma-in-memory-object-store/.

Chao Qin, Zheng Wen, Xiuyuan Lu, and Benjamin Van Roy. An analysis of ensemble sampling. Advances in Neural Information Processing Systems, 35:21602–21614, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 37, 2023.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 3505–3506, 2020.

Herbert Robbins. Some aspects of the sequential design of experiments. Bulletin of the American Mathematics Society, 58:527–535, 1952.

Daniel Russo. Simple bayesian algorithms for best arm identification. In Conference on Learning Theory, pp. 1417–1418. PMLR, 2016.

Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen, et al. A tutorial on thompson sampling. Foundations and Trends® in Machine Learning, 11(1):1–96, 2018.

Aadirupa Saha. Optimal algorithms for stochastic contextual preference bandits. Advances in Neural Information Processing Systems, 34:30050–30062, 2021.

Julian Schrittwieser, Thomas Hubert, Amol Mandhane, Mohammadamin Barekatain, Ioannis Antonoglou, and David Silver. Online and offline reinforcement learning by planning with a learned model. Advances in Neural Information Processing Systems, 34:27580–27591, 2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Richard S. Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In Machine Learning Proceedings, pp. 216–224. Morgan Kaufmann, 1990.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018.

Richard S Sutton, Michael Bowling, and Patrick M Pilarski. The alberta plan for ai research. arXiv preprint arXiv:2208.11173, 2022.

Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367, 2024.

Yunhao Tang, Daniel Zhaohan Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, Rémi Munos, Bernardo Ávila Pires, Michal Valko, Yong Cheng, et al. Understanding the performance gap between online and offline alignment algorithms. arXiv preprint arXiv:2405.08448, 2024.

William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3-4):285–294, 1933.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Jiayi Weng, Min Lin, Shengyi Huang, Bo Liu, Denys Makoviichuk, Viktor Makoviychuk, Zichen Liu, Yufan Song, Ting Luo, Yukun Jiang, Zhongwen Xu, and Shuicheng Yan. EnvPool: A highly parallel reinforcement learning environment execution engine. In Advances in Neural Information Processing Systems, volume 35, pp. 22409–22421, 2022.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

Christian Wirth, Riad Akrour, Gerhard Neumann, and Johannes Fürnkranz. A survey of preference-based reinforcement learning methods. Journal of Machine Learning Research, 18(136):1–46, 2017.

Huasen Wu and Xin Liu. Double thompson sampling for dueling bandits. Advances in neural information processing systems, 29, 2016.

Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf. arXiv preprint arXiv:2405.21046, 2024.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In Forty-first International Conference on Machine Learning, 2024.

Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

Fan Yang, Gabriel Barth-Maron, Piotr Sta´nczyk, Matthew Hoffman, Siqi Liu, Manuel Kroiss, Aedan Pope, and Alban Rrustemi. Launchpad: A programming model for distributed machine learning research. arXiv preprint arXiv:2106.04516, 2021a.

Keming Yang, Zichen Liu, and Philip Cheng. MOSEC: Model Serving made Efficient in the Cloud, 2021b. URL https://github.com/mosecorg/mosec.

Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn. Combo: Conservative offline model-based policy optimization. Advances in neural information processing systems, 34:28954–28967, 2021.

Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012.

Shenao Zhang, Donghan Yu, Hiteshi Sharma, Ziyi Yang, Shuohang Wang, Hany Hassan, and Zhaoran Wang. Self-exploring language models: Active preference elicitation for online alignment. arXiv preprint arXiv:2405.19332, 2024a.

Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. Chain of preference optimization: Improving chain-of-thought reasoning in llms. Advances in Neural Information Processing Systems, 38, 2024b.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

Banghua Zhu, Michael Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In Proceedings of the 40th International Conference on Machine Learning, pp. 43037–43067. PMLR, 2023.

Yinglun Zhu, Dylan J Foster, John Langford, and Paul Mineiro. Contextual bandits with large action spaces: Made practical. In International Conference on Machine Learning, pp. 27428–27453. PMLR, 2022.

### A ALGORITHM DETAILS

While Algorithm 1 presents our Thompson sampling algorithm for LLM alignment, it is intractable and centered around the reward posterior modeling. We next present a practical sample-efficient alignment agent that learns both an LLM policy and an epistemic reward model online in Algorithm 2.

Algorithm 2 Sample-efficient alignment (SEA) for LLMs

Input: Reference policy πref, DAP loss function F, prompt distribution pX, unknown but queryable preference oracle P, mixture ratio γ.

- 1: Initialize experience D0 ← ∅, policy πθ0 ← πref, and ERM weights Φ0 = {ϕ0k}Kk=1 randomly.
- 2: for t = 1, . . . , T do
- 3: Receive a prompt xt ∼ pX.
- 4: Sample M responses yti ∼ πθt−1(·|xt) to construct St = {yti}Mi=1.
- 5: Sample ϕ ∼ Uniform(Φt−1) and set yt ← arg maxb∈Strϕ(xt, b). // Select 1st response y. // E&E objective: aligning an online system.
- 6: repeat

Sample ϕ ∼ Uniform(Φt−1) and set yt′ ← arg maxb∈Strϕ(xt, b). // Select 2nd response y′. until yt′ ̸= yt

// BAI objective: labeling via crowdsourcing.

- 7: Set yt′ ← arg maxb∈St Vϕ [σ (rϕ(xt, yt) − rϕ(xt, b))], // OR select 2nd response y′. where Vϕ [·] computes variance across ensemble members of Φt−1.
- 8: if g < γ for g ∼ Uniform(0, 1) then

Label {yt, yt′} with P to obtain Bt = {xt, yt+, yt−} and update experience Dt ← Dt−1 Bt. else

Use RΦt−1 to get synthetic labels and obtain Bt = {xi, y˜i+, y˜i−}. end if

- 9: Update ERM with the regularized NLL loss (Eq. (8)):

Φt ← Φt−1 − αR∇ΦLR(Φt−1|Dt).

// Reward learning.

- 10: Update policy with the direct optimizer (Eq. (9)):

θt ← θt−1 − απ∇θLπ(θt−1|Bt, πref, F).

// Policy learning.

- 11: end for

In Algorithm 2, we describe an online setting where a single example is processed at each time t (batch size b = 1). This is mainly for notational convenience, while in implementation we set b to be the training batch size (e.g., 128). We instantiate the reward posterior with an epistemic reward model, which allows for efficient incremental update and sampling. We also replace the global optimization (arg maxb∈Y) with a policy-guided local search among proposals sampled from the latest online policy πθt−1. At each time t, we update ERM weights Φ with m randomly sampled batches from the experience Dt. We find setting m = 5 suffices to achieve reasonable accuracy. The policy parameters θ are updated using mixed preference data, with proportion γ being the real environment experience and (1 − γ) from the ERM’s synthetic experience. Note that the synthetic experience is not added into Dt to ensure reward learning always uses ground truth environment data.

We consider the following three direct optimizers in our experiments:

- • DPO (Rafailov et al., 2023):

Fθ(x, y+, y−, πref) = − log σ β log

πθ y+|x πref y−|x πref (y+|x) πθ (y−|x)

(10)

- • IPO (Azar et al., 2024):

Fθ(x, y+, y−, πref) = log

πθ y+|x πref y−|x πref (y+|x) πθ (y−|x) −

- 1

- 2β

2

(11)

- • SLiC (Zhao et al., 2023):

πθ y+|x πref y−|x πref (y+|x) πθ (y−|x)

Fθ(x, y+, y−, πref) = max 0, 1 − β log

where β controls the rate of deviation of πθ from πref.

(12)

### B ON CONNECTIONS WITH SINGLE-STEP RL

By viewing contextual dueling bandits as single-step preference-based RL (PbRL) (Busa-Fekete et al., 2014; Wirth et al., 2017) problems, we can interpret paradigms shown in Figure 3 from the RL perspective.

RLHF approaches (Figure 3a) are instances of offline model-based RL (Kidambi et al., 2020; Yu et al., 2021; Schrittwieser et al., 2021; Liu et al., 2023; Tajwar et al., 2024), where they learn a reward model (no need for a transition model since the prompt-response interaction is single-step) of the environment from a batch of offline collected data, and train a policy (i.e., LLM) to maximize the return (i.e., expected one-step reward) with respect to the learned reward.

In contrast, DAP methods (Figure 3b) are similar to policy-based model-free RL algorithms, e.g., REINFORCE (Williams, 1992) which conducts policy gradient update:

Ex∼XEy∼πθ(·|x) [R(x, y)∇θ log πθ(y|x)] , (13)

where R(x, y) is the return (i.e., cumulative reward) of the trajectory. To connect with DAP, we could set R as arbitrary scalar values based on the binary preference outcomes, e.g., R(x, y+) = ζ and R(x, y−) = −ζ for preference triplet {x, y+, y−}. In this way we could rewrite Eq. (13) as

Ex∼XEy,y′∼πθ(·|x)E(y+≻y−)∼P ζ ∇θ log πθ(y+|x) − ∇θ log πθ(y−|x) , (14)

by repeating action sampling twice and querying the oracle for preference labeling. This matches the gradient direction of contrastive DAP losses (e.g., see Section 4 of DPO (Rafailov et al., 2023)) if we optimize them online (Guo et al., 2024).

Additionally, active reward learning from behavior policy’s data distribution (Figure 3c) can be regarded as inverse RL (Ng & Russell, 2000), which tries to recover environment’s reward function given expert trajectories. In the context of LLM alignment, the preference data {x, y+, y−}Ni=1 directly encodes human’s implicit reward r⋆, which can be inversely learned with assumptions such as the BT model (Bradley & Terry, 1952). However, existing methods belonging to this paradigm mostly rely on a fixed (and suboptimal) behavior policy for response sampling, whose coverage inherently limits the quality of the recovered reward function.

Last but not least, SEA depicted in Figure 3d resembles a class of online model-based RL algorithms, known as Dyna (Sutton, 1990; Janner et al., 2019), that learns a world model from environment experience and trains a base agent (consisting of reactive policies and value functions) from both environment experience and model experience. Compared to model-free methods, Dyna naturally enables more sample-efficient learning by planning with the learned world model to update the base agent. In SEA, we learn the reward model online and update the LLM (i.e., the reactive policy) with model-planing experience by mixed preference learning (Section 4.2.3). Online model-based RL algorithms could suffer from catastrophic forgetting in the face of nonstationary data (Liu et al., 2024b), and we leave it for future work. Overall, this model-based RL formulation is powerful and explains popular LLM techniques, e.g., Best-of-N sampling (Touvron et al., 2023) can be viewed as planning for acting, which trades compute for performance. We believe it is a promising path leading us to unlock superhuman capabilities of LLMs.

### C SYSTEM BENCHMARKING

We conduct a rigorous benchmarking comparison on the efficiency of online DPO training using our learning system oat4, alongside the trl’s implementation5.

Settings. In alignment with the examples provided by trl, we use the TL;DR (Stiennon et al., 2020) dataset and evaluate training efficiency at three model scales: 1B, 2.8B and 6.9B parameters for both SFT-ed LLMs and exclusively trained RMs. This is similar to the settings in our experiments (see Appendix D) except that we fix the reward oracle to be a strong general-purpose RM.

Hardware & Software. All benchmarking experiments are conducted on a single machine with eight A10040G GPUs and 96 AMD EPYC 7352 CPUs. To ensure fair comparison, we align all key hyperparameters for both oat and trl. The DeepSpeed ZeRO-2 strategy is employed by default when GPU memory suffices; otherwise, ZeRO-3 or ZeRO-2-offload is utilized as applicable. Notably, the distributed architecture of oat provides flexibility in system configuration, enabling adjustments to accommodate memory and computational time constraints. Figure 8 illustrates two example configurations employed in our benchmarking experiments.

• Config 1 collocates all three workloads on each of the GPUs. Specifically, eight vLLM instances (for actors) and eight Mosec workers (for oracle RMs) are spawned to run independently on each GPU. After a batch of responses is generated (by actors) and labeled (by oracle RMs), it is sent to

- 4https://github.com/sail-sg/oat.
- 5https://github.com/huggingface/trl/blob/main/trl/trainer/online_dpo_trainer.py.

Config 1: full collocation Config 2: half collocation

- device:0
- device:1
- device:2
- device:3
- device:4
- device:5
- device:6
- device:7

- device:0
- device:1
- device:2
- device:3
- device:4
- device:5
- device:6
- device:7

###### ✓2

✓1 ✓1 ✓1

✓2

✓0

###### ✓0 ✓0

✓1 ✓1

✓2 ✓2 ✓2 ✓2 ✓2

✓0 ✓0 ✓0

✓2

###### ✓2

✓1 ✓1

✓0

✓2

✓1

- ✓0
- ✓1

✓2

✓1 ✓1 ✓1 ✓1 ✓1

✓0 ✓0 ✓0

✓2

✓1

✓3

✓2

- ✓0
- ✓1

✓2

✓0

✓3

✓2

✓2

✓0

time

time

Actor: vLLM inference

Oracle: Mosec service Learner: DeepSpeed

Weights synchronization

Figure 8: Two example configurations of oat used in benchmarking experiments.

the learner, which runs on all eight GPUs coordinated through ZeRO strategies for policy learning. The updated policy weights are then broadcasted to all actors for on-policy response sampling on subsequent prompt batch. While this configuration maximizes GPU utilization, it requires substantial GPU memory to accommodate all workloads and is thus employed only for 1B scale experiments.

• Config 2 only collocates actor and oracle workloads on half of the GPUs, reserving the remaining four GPUs exclusively for the learner. This is suited for larger-scale experiments (e.g., 2.8B or 6.9B), where additional GPU memory is allocated to the learner. However, this setup incurs idle time on half of the GPUs due to data dependency, as the learner must await new preference data, and the actor must await updated policies. An alternative is to implement asynchronous data collection, where minor data staleness is allowed by using θt−1 to generate data for updating θt+1. Although this data would not be strictly on-policy, asynchronous training could reduce idle time and enhance GPU utilization. This approach has proven effective in large-scale RL systems (Berner et al., 2019), and we leave this optimization to future work.

We provide all benchmarking scripts in our codebase6 for reproducibility.

Results. Benchmarking results for the latency of training a batch of 128 samples are presented in Figure 9. Overall, training with oat config 2 demonstrates consistently greater efficiency than trl, achieving up to a 2.5× reduction in latency at the 2.8B scale.

65.39

trl-learn

oat-learn

60

trl-oracle

oat-oracle

Batchlatency(second)

trl-generate

oat-generate

48.83

trl-other

oat-other

40

34.43

23.56

20

13.77

9.25

4.21 4.67 3.50

0

config 1 gloo

config 2 nccl

config 2 gloo

config 2 nccl

config 2 gloo

config 2 nccl

1B 2.8B 6.9B

- Figure 9: Averaged training latency (over 10 batches, equivalent to 1280 samples) comparing sail-sg/oat against huggingface/trl.

We next analyze the time costs for individual stages: generate, oracle and learn. Across all scales and configurations, oat demonstrates significantly lower generate time than trl, due to distributed actors utilizing vLLM.

6https://github.com/sail-sg/oat/tree/main/benchmark.

Additionally, at the 6.9B scale, oat requires substantially less oracle time than trl, as trl employs ZeRO-3 to prevent GPU memory overflow, thereby slowing inference. In contrast, oat config 2 allows for flexible collocation, enabling oracle RMs hosted via Mosec to operate in parallel without sharding. However, oat config 2 incurs longer learn time compared to trl due to the use of only half the available GPUs. This limitation also explains why, at the 1B scale, config 2 has higher latency than config 1 across all stages.

The other category accounts for time costs associated with data loading, tokenization, and communication. Here, inter-process communication is the primary cost, with trl showing minimal overhead as all three stages operate within the same process on identical micro-batches, avoiding weight synchronization. By contrast, oat requires considerable time to transfer updated policy weights from the learner to all actors. While NCCL is recommended for synchronization over GLOO, it requires older vLLM packages (prior to version 0.4.3), which may lack support for newer LLM architectures. Moreover, NCCL is incompatible with config 1 due to its restriction on the learner master process establishing two separate process groups (one for DeepSpeed, the other for weight synchronization). In summary, we recommend future researchers prioritize oat config 2 and employ NCCL when feasible.

### D FULL EXPERIMENTAL DETAILS

Models. We experiment three model scales (1B, 2.8B, 6.9B) from the Pythia family (Biderman et al., 2023). We take pretrained SFT models from Huang et al. (2024) as πref for the starting model in all experiments. Except in Section 6.1, we use 1B model for experiments to save computation.

Reward oracle. We simulate the process of human feedback with a strong scalar RM and refer it as reward oracle. We choose Skywork-Reward-Llama-3.1-8B7 (Liu et al., 2024a), which is top-ranked in RewardBench leaderboard (Lambert et al., 2024), as the reward oracle.

Epistemic reward model. We build ERM on top of a pretrained 0.4B transformer (Jiang et al., 2023), by removing its head and adding an ensemble of MLPs. The size of ensemble is set to K = 20, and all MLPs contain 2 hidden layers of 128 nodes. Note that the ERM is chosen to be much smaller than the reward oracle following Dwaracherla et al. (2024), which reflects the fact that human preferences can be more complex than what the agent can model. The regularization coefficient λ is fixed to be 0.5 after a coarse hyperparameter search.

Data. We employ the widely adopted TL;DR dataset (Stiennon et al., 2020) for our experiments. It consists of Reddit posts as prompts, and the agent is required to give summaries that align with human preferences. We fix 50k prompts for training and limit the query budget to 50k as well.

DAP methods. We adopt three DAP methods (direct optimizers) to thoroughly validate our algorithm, including DPO (Rafailov et al., 2023), IPO (Azar et al., 2024) and SLiC (Zhao et al., 2023).

Baselines. We include the offline and online variants of different DAP methods as baselines, which are studied by Guo et al. (2024). Additionally, we compare with two active exploration baselines built on online DPO: APL (Muldrew et al., 2024) and XPO (Xie et al., 2024). We omit the comparison with SELM (Zhang et al., 2024a) since SELM and XPO share a very similar algorithmic design.

Metrics. We use the win rate of agent’s responses against reference responses judged by the reward oracle as the performance metric. This metric can reflect both the agent’s cumulative regret and anytime regret (i.e., average performance). In the E&E setting, we measure the “online” win rate of the agent’s dueling responses that are executed during experience collection and take the average. In the BAI setting, we measure the “offline” win rate by evaluating the latest agent’s responses given a fixed set of holdout prompts periodically. We mainly focus on the BAI setting because crowdsourcing seems a major scenario for most practitioners, and present one set of experiments for comparing different exploration strategies in both settings. When the comparison is only made within a model scale, we report the relative win rate against the initial STF models. When the comparison is across scales (Figure 1 Left), we report the absolute win rate against the ground truth responses in the dataset.

Hyperparameters. We set β = 0.1 for DPO and β = 0.2 for SLiC and find they are robust for all scales. We tune β from {0.2, 0.3, 0.5, 1.0} for IPO across scales and report the best performing results. We sample M = 20 on-policy responses with a temperature η = 0.7 during training, and use greedy decoding for offline evaluation (BAI’s metric). We use the Adam optimizer with learning rate of 5e − 7 and cosine scheduling, and set the batch size to be 128. We initialize the mixture ratio γ of SEA to be 1 and adjust it to 0.7 after a burn-in period of 1k samples. To reproduce baselines (APL and XPO), we follow the recommended hyperparameters from their papers.

Statistical significance. There are various factors to introduce randomness during online learning. We thus launch 3 independent runs for every experiment with different random seeds. All the results are reported with mean and standard error to indicate their statistical significance.

7https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B.

Computational resources. Experiments at all scales are conducted on a single machine with 8 A100 GPUs to run the learner and actors. We additionally host a separate remote server with workers spawned on 16 A100 GPUs for the oracle RM8, so that it can be queried by all concurrently running experiments. All experiments conducted for this research consume about 2 A100 GPU years.

### E SUPPLEMENTARY MATERIALS

We include a comparison of prior works (Table 2) and an example of ChatGPT’s active exploration (Figure 7) in this section.

Exploration Interaction Proposal Policy Active Passive Online Iterative Offline πθ πβ

Method

Christiano et al. (2017) ✓ ✓ ✓ ✓ Stiennon et al. (2020) ✓ ✓ ✓ ✓

RL Optimizer

Bai et al. (2022) ✓ ✓ ✓ ✓ Ouyang et al. (2022) ✓ ✓ ✓ ✓

Zhao et al. (2023) ✓ ✓ ✓

Rafailov et al. (2023) ✓ ✓ ✓ Azar et al. (2024) ✓ ✓ ✓ Meng et al. (2024) ✓ ✓ ✓

Xu et al. (2023) ✓ ✓ ✓ Guo et al. (2024) ✓ ✓ ✓

Direct Optimizer

Mehta et al. (2023) ✓ ✓ ✓

Das et al. (2024) ✓ ✓ ✓ Melo et al. (2024) ✓ ✓ ✓

Dwaracherla et al. (2024) ✓ ✓ ✓ Zhang et al. (2024a) ✓ ✓ ✓

Xie et al. (2024) ✓ ✓ ✓ Muldrew et al. (2024) ✓ ✓ ✓

Table 2: A summary of prior work. πθ denotes the proposal policy that is continuously updated based on newly collected preference data, while πβ denotes a fixed proposal policy. Algorithms that encompass online interaction (Property 1), active exploration (Property 2), and learnable πθ offer the best sample efficiency. Notably, only three methods (listed at the bottom of the table) satisfy these characteristics, and we include them for comparisons in our experiments.

8We utilize the Kubernetes service for routing requests to multiple Mosec (Yang et al., 2021b) instances.

[Figure 13]

###### Figure 10: ChatGPT system asks for users’ preference feedback to strategically explore better answers. In this case, algorithms should be designed around the objective of minimizing cumulative regret (i.e., the E&E setting), because the quality of both responses generated by the system affects user experience.

