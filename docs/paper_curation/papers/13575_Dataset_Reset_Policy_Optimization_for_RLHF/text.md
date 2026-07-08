## Dataset Reset Policy Optimization for RLHF

# arXiv:2404.08495v3[cs.LG]16Apr2024

Jonathan D. Chang∗ Department of Computer Science Cornell University jdc396@cornell.edu

##### Wenhao Zhan∗

Department of Electrical and Computer Engineering Princeton University wenhao.zhan@princeton.edu

Owen Oertell Department of Computer Science Cornell University ojo2@cornell.edu

Kianté Brantley Department of Computer Science Cornell University kdb82@cornell.edu

Dipendra Misra Microsoft Research New York dimisra@microsoft.com

Jason D. Lee Department of Electrical and Computer Engineering Princeton University jasonlee@princeton.edu

Wen Sun Department of Computer Science Cornell University ws455@cornell.edu

Abstract

Reinforcement Learning (RL) from Human Preference-based feedback is a popular paradigm for fine-tuning generative models, which has produced impressive models such as GPT-4 and Claude3 Opus. This framework often consists of two steps: learning a reward model from an offline preference dataset followed by running online RL to optimize the learned reward model. In this work, leveraging the idea of reset, we propose a new RLHF algorithm with provable guarantees. Motivated by the fact that offline preference dataset provides informative states (i.e., data that is preferred by the labelers), our new algorithm, Dataset Reset Policy Optimization (DR-PO), integrates the existing offline preference dataset into the online policy training procedure via dataset reset: it directly resets the policy optimizer to the states in the offline dataset, instead of always starting from the initial state distribution. In theory, we show that DR-PO learns to perform at least as good as any policy that is covered by the offline dataset under general function approximation with finite sample complexity. In experiments, we demonstrate that on both the TL;DR summarization and the Anthropic Helpful Harmful (HH) dataset, the generation from DR-PO is better than that from Proximal Policy Optimization (PPO) and Direction Preference Optimization (DPO), under the metric of GPT4 win-rate. Code for this work can be found at https://github.com/Cornell-RL/drpo.

### 1 Introduction

Reinforcement learning aims at maximizing a cumulative reward function. However, specifying a reward function in practice can be challenging (Wirth et al., 2017). Reinforcement Learning with Human Feedback (RLHF) has become an effective approach when a reward function does not exist (Christiano et al., 2017). Operating under a setting where human labelers provide preference-based feedback (e.g., ranking of generations from an RL agent), RLHF learns a reward model and then optimizes the reward model via RL techniques. RLHF has found applications across various domains, including games (MacGlashan et al., 2017; Christiano et al., 2017; Warnell et al., 2018), large language models (LLMs) (Ziegler et al., 2019; Stiennon et al., 2020; Wu et al., 2021; Nakano et al., 2021; Ouyang et al., 2022; Glaese et al., 2022; Bai et al., 2022a; Ramamurthy et al., 2022; Liu et al., 2023), and robot learning (Brown et al., 2019; Shin et al., 2023).

RLHF typically consists of the following two steps: (1) fitting a reward model using a pre-collected offline preference-based dataset (often generated from some pre-trained models and labeled by humans), (2) and learn a policy via online RL (e.g., Proximal Policy Optimization (Schulman et al., 2017)) to optimize the learned reward model. These two steps are often done separately in the sense that once the reward model is learned, step (2) only optimizes the reward model without ever using the offline preference dataset. Is there any benefit of re-using the offline data during the procedure of optimizing the reward model

∗Equal contribution

via online RL? Prior work on hybrid RL (Song et al., 2022; Ball et al., 2023) demonstrated that combining offline data and online data can often significantly boost learning efficiency. Can we achieve a similar boost in learning efficiency for RLHF?

Towards answering this, we propose an algorithm called Dataset Reset Policy Optimization (DR-PO), operating under the assumption of being able to reset, i.e., we can go back to any state and start policy optimization and data collection from that point (as opposed to reseting to initial states). While being able to reset is certainly an assumption, it is naturally satisfied when using RL to fine-tune generative models like language models and diffusion models (Lee et al., 2023). This is because the underlying Markov transitions are simple, known, and deterministic. For instance, when using RL to optimize text generation, resetting to a state is equivalent to feeding a partial sentence (together with the initial prompt) to the transformer-based policy. Our algorithm, DR-PO, is a hybrid RL approach that integrates offline data into an online RL procedure: when collecting online data, DR-PO resets the policy optimizer to the states in the offline dataset for exploration. Algorithmically, DR-PO is simple: it iteratively collects a batch of online data by resetting the policy to states in the offline data, performs policy rollouts, and optimizes the policy using the online batch via policy optimization techniques such as Natural Policy Gradient (NPG) (Kakade, 2001) or Actor-critic methods (e.g., PPO (Schulman et al., 2017)).

While DR-PO is as simple to implement as most of the existing policy optimization algorithms, we demonstrate that DR-PO achieves strong theoretical guarantees under natural assumptions. Specifically, when optimizing a reward model learned from an offline preference dataset, DR-PO is capable of learning a policy that is at least as good as any policy which is covered by the offline data in terms of maximizing the ground truth rewards, and DR-PO achieves this result under general function approximation with finite sample complexity. DR-PO is also computationally tractable since it only requires supervised learning style oracles such as a Maximum Likelihood Estimation (MLE) oracle (for fitting reward models) and a Least Squares Regression oracle (for learning value functions). Thus DR-PO advances the status of the theoretical work on RLHF (see more detailed discussion in Section 2). Empirically, we test our approach on two standard RLHF datasets: TL;DR summarization (Stiennon et al., 2020) and Anthropic HH. In TL;DR summarization, we demonstrate that the summaries generated by DR-PO outperform those from PPO and DPO (Rafailov et al., 2023) in terms of GPT4 win-rate. We also show that when transferring the policies trained on TL;DR to the CNN/DailyMail news articles in a zero-shot manner, policies trained via DR-PO again generate summaries that outperform those from PPO and DPO, indicating that dataset reset does not make DR-PO overfit. Finally, we test how DR-PO scales on Anthropic HH (Bai et al., 2022b) across three different model scales and show that DR-PO scales just as well as PPO while still outperforming baselines.

Our key contributions can be summarized as follows.

- • We propose to use the idea of dataset reset to integrate offline data into online RLHF. Reset is a property that comes for free when optimizing generative models using RL. By leveraging dataset reset, our new algorithm DR-PO achieves strong performance guarantees and offers significant benefits in terms of computation tractability over prior theoretical RLHF works.
- • When instantiating PPO as a policy optimizer in DR-PO, we show that our approach can outperform strong baselines PPO and DPO over two standard RLHF benchmarks: TL;DR summarization and Anthropic HH. DR-PO achieves superior empirical performance over PPO without introducing any additional computation or memory overhead to PPO.

### 2 Related Work

Provably efficient RLHF. The theoretical investigation on online RLHF started in bandit setting with the notion of dueling bandits (Yue et al., 2012; Zoghi et al., 2014; Dudík et al., 2015), which aims at identifying the optimal arm with human preference feedback over action pairs. Extending this discussion to tabular MDPs, Novoseller et al. (2020) proposes a dueling posterior sampling algorithm that requires computing and sampling from the posterior of the model dynamics and reward function, leading to potential computational inefficiency. Another PAC RLHF algorithm for tabular MDPs is presented by Xu

- et al. (2020). However, this method involves computing complicated bonus terms to guide exploration. Additionally, Pacchiano
- et al. (2021); Chen et al. (2022) have designed online RLHF algorithms with provable guarantees by updating a confidence set of the policies iteratively, which, unfortunately, are not practically feasible either. In a more recent study, Zhan et al. (2023b) tackles the problem of reward-free RLHF. Nevertheless, their algorithm introduces a series of non-convex optimization problems which are challenging to solve. Notably, these works either only focus on tabular MDPs Novoseller et al. (2020); Xu et al. (2020); Pacchiano et al. (2021) or rely on specialized function approximation such as linear parametrization (Pacchiano et al., 2021; Zhan et al., 2023b) and function classes with small Eluder dimension (Chen et al., 2022; Wu and Sun, 2023), which further restricts their application in practice. In contrast, we focus on the setting where preference-labeled data is only available offline, which is more consistent with the settings considered in applications of fine-tuning language models. Also by using the idea of dataset reset, our algorithm works with function approximation that is much more general than the above prior works.

The study on theoretical offline RLHF is more limited. Li et al. (2023) focuses on learning the reward from a human’s behavior in dynamic discrete choice models rather than from human preference feedback, and thus, the setting is different. Zhu et al. (2023a) studies PAC algorithms for linear models and Zhan et al. (2023a) extends the analysis to general function approximation. However, both of their algorithms are not computationally efficient because they rely on constructing a confidence set for the reward function and solving a constrained maximin problem.

Tiapkin et al. (2023) studied the setting where high-quality expert demonstrations exist. They use behavior cloning to train a policy using expert demonstrations and then run an Upper-confidence-bound style algorithm to optimize a reward function under a KL regularization to the behavior-cloned policy. They show that for tabular and linear MDP, the expert demonstrations reduce the sample complexity of online RL. We consider preference-based offline datasets, which may not necessarily come from a high-quality expert, and function approximation that is significantly more general than linear and tabular functions. Note that UCB based algorithms can quickly become computationally intractable beyond tabular and linear settings (e.g., Jiang et al. (2016); Du et al. (2021)). Our algorithm uses the idea of dataset reset for exploration and does not involve any optimism-based exploration strategy, making it computationally tractable even when dealing with general function approximation. We think that the key idea of dataset reset can also be used in the setting from Tiapkin et al. (2023) to make their algorithm extend beyond the tabular and linear MDP settings.

Empirical RLHF algorithms. This work continues the recent literature of RLHF algorithms that perform online RL (Zhu et al., 2023b; Wu et al., 2023; Chang et al., 2023) to finetune large generative models. There have also been efforts to build on top of DPO (Rafailov et al., 2023) with algorithms such as IPO (Azar et al., 2011) and KTO (Contextual.ai, 2023). In this paper, our work is complementary to many of these efforts in augmenting RL through the incorporation of dataset resets in online generation. Ideas from this work could directly be applied to existing online RLHF algorithms such as P3O (Wu et al., 2023) and APA (Zhu et al., 2023b). Given the recent work (Yuan et al., 2024) in incorporating online generations to improve DPO, an offline RLHF method, the idea of dataset resets could also be relevant in this space of hybrid RLHF methods.

Using reset in RL The idea of reset is not new in RL (Kakade, 2003; Bagnell, 2004; Nair et al., 2018; Salimans and Chen, 2018; Yin et al., 2022; Uchendu et al., 2023; Silver et al., 2016; Agarwal et al., 2019; Daumé III and Marcu, 2005; Daumé et al., 2009). When resetting is available, it helps address exploration and credit assignment problems. In this work, we show that resetting to an offline dataset helps in RLHF. The key challenge in RLHF is that the reward model is learned purely from offline data which may not have a global coverage to the entire state space. Our algorithm incorporates KL regularization to ensure the learned policies do not deviate too much from the offline data so that we do not over-optimize the learned reward model (e.g., reward hacking). While the idea of KL-regularization was also used in prior empirical RLHF works (e.g.,Stiennon et al. (2020); Bai et al. (2022a)), we show that by combining the two key ideas, KL regularization and dataset reset, our algorithm achieves strong performance in both theory and practice. We also demonstrate the efficacy of our approach in the application of fine-tuning language models.

### 3 Preliminaries

Markov Decision Processes. In this paper we consider an episodic time-inhomogeneous Markov Decision Process (MDP) M with state space S = {Sh}Hh=1, action space A and horizon H. Here Sh is the subspace of all states at step h. We suppose the states incorporate the information of the current step and thus {Sh}Hh=1 are mutually disjoint. We assume that every episode begins at the same state s1 and ends at the dummy state sH+1, but our analysis can be extended to a random starting state easily. In each episode, at step h ∈ [H], the agent observes the current sh and executes an action ah. Then the environment generates a reward r⋆(sh,ah) (which can be unobservable to the agent), and transits to a new state sh+1, which is sampled from the transition probability P(·|sh,ah). Here we suppose the reward function r⋆ : S × A  → [0,1] is bounded, and for any possible trajectory τ = (sh,ah)Hh=1, we have Hh=1 r⋆(sh,ah) ≤ rmax. Note that when the reward is sparse, rmax can be much smaller than H.

A policy π : S → ∆A specifies the action selection probability of the agent conditioned on the current state. Given a policy π, we define its state-action visitation measure as dπh(s,a) = Pπ(sh = s,ah = a) for all s ∈ Sh,a ∈ A,h ∈ [H] where Pπ(·) denotes the distribution of the trajectory when executing policy π. We will also use dπh(s) = a∈A dπh(s,a) to denote the state visitation measure and dπ(τ) to denote the distribution of the trajectory under policy π. We can further define the associated

value functions and Q functions of policy π and reward function r as V π,r(s) = Eπ[ Ht=h r(st,at) | sh = s],Qπ,r(s,a) =

Eπ[ Ht=h r(st,at) | sh = s,ah = a] for all h ∈ [H],s ∈ Sh,a ∈ A.2 They characterize the expected cumulative reward under policy π starting from a state or a state-action pair.

We aim to find an ϵ-optimal policy π with respect to the true reward r⋆ and a target policy π⋆ which we denote as some high-quality policy (π⋆ is not necessarily the globally optimal policy), i.e., V π

⋆,r⋆(s1)−V π,r

⋆

(s1) ≤ ϵ. Particularly, we would only utilize common oracles such as Maximum Likelihood Estimator (MLE) and Least Squares Regression (LSR). We also want our algorithms to be able to leverage general function classes beyond linear functions.

RL from Human Feedback (RLHF). We consider the setting where the true reward r⋆ is unobservable. Instead, we have access to an offline trajectory-pair dataset DR = {(τm0 ,τm1 ,om)Mm=1} labeled with human preference, where the trajectories τm0 and τm1 are i.i.d. sampled from some pre-trained policy πSFT (e.g., in NLP tasks, this can be the instruction fine-tuned policy, which is also called supervised fine-tuned (SFT) policy). In this work, we do not explicitly consider the learning procedure of πSFT, and we assume it is given to us. Here om ∈ {0,1} characterizes the human preference over the trajectory pairs (τm0 ,τm1 ) and we suppose the human preference is modeled by a monotonically increasing link function Φ:

P(o = 1 | τ0,τ1) = P(τ1 ≻ τ0) = Φ(r⋆(τ1) − r⋆(τ0)),

where we use r⋆(τ) to denote Hh=1 r⋆(sh,ah) for any trajectory τ = (sh,ah)Hh=1. A widely-used model is the Bradley-TerryLuce (BTL) model (Bradley and Terry, 1952) where the link function is chosen to be the sigmoid function σ(x) = 1/{1 +

exp(−x)}. We will use κ = inf 1

x∈[−rmax,rmax] Φ′(x) to measure the non-linearity of the link function Φ, which in turn reflects the hardness of learning the reward model from the human preference. Given DR, we can learn a reward model r using MLE:

M

−log P(o = om | τm0 ,τm1 ;r), (1)

r = arg min

r∈R

m=1

With the BTL model, the above NLL becomes

(om = 1) · log 1 + exp(r(τm0 ) − r(τm1 )) + (om = 0) · log 1 + exp(r(τm1 ) − r(τm0 )) ,

which is a loss function that has been used in many prior RLHF works(Christiano et al., 2017; Stiennon et al., 2020). We also assume that we have an unlabeled dataset DTR = {τn}Nn=1 where τn is i.i.d. sampled from πSFT. Note that DTR is unlabeled, so it potentially can be much larger than the human-labeled dataset DR.

The Ability to Reset. We consider the setting where we can reset the system. More formally, given any state sh at time step h, we can reset the RL agent directly to sh and rollout a policy π. While this is certainly an assumption, it is satisfied in many important applications, e.g., fine-tuning generative models such as LLMs (Ouyang et al., 2022; Ramamurthy et al., 2022; Chang et al., 2023) and Diffusion models (Lee et al., 2023) with RL. In text generation, a state sh typically means a partial sentence. Resetting from this state would then mean that we feed the partial sentence sh to a transformer based policy and have it generate new tokens one by one starting from the given partial sentence. We emphasize that in the RL literature, prior works (e.g., PPO and many RL theoretical works (Agarwal et al., 2021; Azar et al., 2017; Jin et al., 2020; Zhan et al., 2022)) typically do not assume the ability to reset – they often assume the agent has to always start from some initial states. However, when reset is available, it is often a game changer, in both theory (Yin et al., 2022) and in practice (e.g., AlphaGo (Silver et al., 2016)).

### 4 Dataset Reset Policy Optimization

We present a meta-algorithm here to provide the details of how we leverage the idea of dataset reset to collect online batch data. We abstract away the policy optimization oracle here to emphasize the novelty of our interaction with the environment for online data collection via dataset reset. Once the online batch data is collected, we feed it to a policy optimization oracle, e.g., PG, NPG, Actor-critic methods, or a PPO-style update 3.

- 2For notation simplicity, we drop the usual subscript h in value functions, as we have assumed state s contains the information of time step h.
- 3Here we mean the specific actor-critic style policy optimization formulation where clipping is used to ensure small policy update, and critic is learned via GAE, on a given online batch data (Schulman et al., 2017).

- Algorithm 1 Dataset Reset Policy Optimization (DR-PO)

- 1: Input: Preference dataset DR, unlabeled dataset DTR, reward function class R, total number of iterations T.
- 2: Initialize: π1 = πSFT.
- 3: Learn a reward model r via MLE based on Eq. (1).
- 4: for t = 1,··· ,T do
- 5: Initialize an empty online batch Don. /* Online data collection */
- 6: for n = 1,···N do
- 7: Randomly sample a trajectory in DTR and a state sh from it where h ∈ [H].
- 8: Reset πt to sh and rollout πt to generate trajectory {sh,ah,...,sH,aH}.
- 9: Add trajectory {sh′,ah′, r(sh′,ah′),ln(πt(ah′|sh′)/πSFT(ah′|sh′))}Hh′=h to Don.
- 10: end for
- 11: Policy update: πt+1 ⇐ PO(πt,Don). {PG, NPG / TRPO, CPI, Actor-Critic, PPO}
- 12: end for

Algorithm 1 summarizes the key idea of dataset reset in DR-PO. The key difference between DR-PO and a more standard policy optimizer is that in DR-PO, for each episode, the policy collects online trajectories via resetting to a state randomly sampled from some trajectory in the offline dataset DTR. In other words, we do not rollout the policy π from the initial state s1 as typically done in standard policy optimization algorithms like PG. The online data collection procedure collects a batch of online trajectories Don. Note for each online trajectory, we record each state-action pair’s reward measured under the learned reward model r, and also the log ratio of πt and πSFT which serves as an empirical estimate of the policy KL divergence, i.e., KL(πt(sh′)||πSFT(sh′)). Such a KL divergence term can be optionally used as a reward penalty to ensure the learned policies do not deviate too far from πSFT so that the reward model r stays as a good approximation of the true reward r⋆ under learned policies’ trajectory distributions. We use this KL penalty both in theory and in practice.

Once the online data is collected, we feed it to a policy optimization oracle PO for a policy update. A PO oracle can be a PG, NPG, or PPO style update. To be more specific, for a PPO style update procedure, we use Don to fit a critic for advantage estimation A(s,a)4 (e.g., via generalized advantage estimation used in PPO), and then update the policy on Don with the clipping trick: πt+1 ⇐ arg maxπ s,a∈D

Clip π π(a|s)

t(a|s) A(s,a). This is the policy update that we use in our experiments. In our theory, we use NPG as the PO oracle. While PPO and NPG are different when it comes to exact implementation, PPO can be understood as a heuristic that approximates NPG for the purpose of being more scalable for large-scale optimization (e.g., the clipping trick induced by PPO is approximately trying to ensure that the new policy does not deviate too much from the old one – a key property that NPG methods advocated for (Kakade, 2001; Kakade and Langford, 2002; Bagnell and Schneider, 2003; Schulman et al., 2015)).

on

Implementation wise, with PPO as a PO oracle, given a standard PPO implementation, all we need to do is to feed the policy optimization and GAE oracles in PPO using the online batch of data collected in our way, i.e., Don collected via dataset reset. Our experiments on two RLHF datasets show that hyperparameters that work well for PPO also work for DR-PO.

### 5 Theoretical Analysis

In this section, we analyze the DR-PO (Alg 1) by instantiating the policy optimization oracle PO to be a Natural Policy Gradient (NPG) oracle. For completeness, we describe PO in Algorithm 2, which in high level consists of policy evaluation via least square regression, and then policy update via Mirror Descent style procedure. We leave the detailed full description of the algorithm in Appendix A.

In Alg. 2, we use the online data to fit a Q function estimate of the current policy πt. Once we learn the critic, we perform policy update via running KL-based Mirror Descent. Note that this step has a closed-form expression for πt+1:

πt+1(a|s) ∝ πSFT(a|s)

ηλ

ηλ+1 · πt(a|s)

1 ηλ+1

· exp

η ηλ + 1 · Q(s,a)

4when using KL penalty, this advantage function measures the advantage under KL regularized reward — r − λKL with λ ∈ R+ as coefficient for the KL penalty.

Note that the KL penalty to πSFT in the policy update procedure is important to ensure that πt+1 does not deviate too much from πSFT. Also this type of updates ensures that the support of πt(·|s) is always a subset of the support of πSFT(s) for all state s.

- Algorithm 2 NPG update for the PO oracle in Alg. 1

- 1: Input: Online dataset Don, the previous policy πt, Q function class F, regularization parameter λ, learning rate η
- 2: Create an empty regression dataset D.
- 3: for each (partial) trajectory τ in Don do
- 4: Take the first state-action pair (sh,ah) in τ and calculate the total reward y = Hh′=h r(sh′,ah′)
- 5: Add ((sh,ah),y) to D
- 6: end for
- 7: Learn critics:

Q = arg min

f∈F

1 |D|

(s,a,y)∈D

(f(s,a) − y)2 .

- 8: Policy update:

πt+1(s) = arg min

⟨−Q(s,·),p⟩ + λKL(p∥πSFT(s)) +

p∈∆(A)

1 η

KL(p∥πt(s)),∀s.

Remark 5.1. Though we mainly focus on the settings where we can reset, when resetting is not possible (e.g., real robotics applications), we can implement the reset by a roll-in and roll-out procedure since we have access to πSFT: we roll-in πSFT to some sh, and then continue by rolling out our policy that is being optimized. This procedure is closely related to the PPO++ algorithm proposed in Chang et al. (2023), where the authors empirically demonstrated that it outperforms vanilla PPO on some RLHF benchmarks (but no detailed theoretical investigation). When resetting is available, by directly resetting to the offline data generated by πSFT, we further reduce computation.

- 5.1 Theoretical Sample Complexity Now we introduce the required assumptions in our analysis.

Function classes. We first assume that the reward function class and Q function class are realizable and bounded:

- Assumption 5.2 (reward function classes). Suppose that we have r⋆ ∈ R. In addition, assume that 0 ≤ r(τ) ≤ rmax for all r ∈ R and trajectory τ.
- Assumption 5.3 (Q function classes). Suppose that we have Qπ

t, r ∈ F for all t ∈ [T]. In addition, assume that 0 ≤ f(s,a) ≤ rmax for all f ∈ F,s ∈ S,a ∈ A.

Realizability is a standard assumption used in the theoretical analysis of supervised learning. It is possible to extend our analysis to the setting where model-misspecification exists, and we leave this extension as a future work.

Concentrability. Then we assume that πSFT can cover the comparator policy π⋆. In addition, we know the learned policy π is close to πSFT in terms of Kl divergence due to the regularizer KL(·∥πSFT) in the mirror descent step. Thus, to deal with distribution shift, we also assume πSFT can cover the policies which are close to itself:

- Assumption 5.4 (single-policy concentrability). Suppose that we have for any BKL ≥ 0:

- (1)max τ

dπ

⋆

(τ) dπSFT(τ)

= CTR < ∞;

- (2) max h∈[H],s∈Sh,a∈A

⋆

dπ

h (s,a) dπhSFT(s,a)

= CST < ∞;

dπ(τ) dπSFT(τ)

- (3) max π∈Θ(πSFT,BKL),τ

= CSFT(BKL),

where Θ(πSFT,BKL) := {π : KL(π(s)∥πSFT(s)) ≤ BKL,∀s ∈ S}.

Note that in Assumption 5.4 we need πSFT to cover π⋆, both trajectory-wise and state-action-wise. In particular, we always have CST ≤ CTR. Assuming trajectory-wise covering is necessary in RLHF because the human feedback is also trajectory-wise, as shown by the lower bounds in Zhan et al. (2023a). Intuitively, if the offline data only covers low performance policies’ traces, then the learned reward model cannot guarantee to recognize trajectories from a high performance policy during test time (because it has never seen such things in training).

- Remark 5.5. We can indeed relax Assumption 5.4 by leveraging the information in R and F, as shown in the discussion in Appendix B.
- Remark 5.6. Note that we have CSFT(BKL) < ∞ for all BKL < ∞ naturally because π ∈ Θ(πSFT,BKL) has bounded KL diveregnce with respect to πSFT.

Under the above assumptions, we have the following theorem to characterize the suboptimality of π returned by Algorithm 3. Recall that κ = inf 1

x∈[−rmax,rmax] Φ′(x) measures the non-linearity of the link function Φ. Theorem 5.7. Suppose Assumption 5.2,5.3,5.4 hold. For any δ ∈ (0,1], let

κ2 M

rmax2 N

log |R| δ

T|F| δ

log

ϵMLE := Θ

, ϵeval := Θ

,

###### and set η = Tr12

, then with probability at least 1 − δ, we have Algorithm 1 with NPG update (Algorithm 2) returns a policy π which satisfies

max

⋆,r⋆(s1) − V π,r

⋆

V π

(s1) ≤ ( CTR + CSFT(Trmax/λ))ϵMLE

###### +2H CSTϵeval

+

(2)

(1)

2H 23 rmax log CST √

. (2)

+ λH log CST

T

(3)

Theorem 5.7 indicates that the suboptimality of π scales with M1 and N1 polynomially. More specifically, term (1) in Equation 2 measures the estimation error of the reward, (2) is the Q function estimation error and (3) is the optimization error of NPG. We can see that there exists a tradeoff between the estimation error and optimization error. With increasing T and decreasing λ, the optimization error (3) will decrease while the distirbution shift coefficient CSFT will become larger, leading to amplified estimation error. In particular, from Theorem 5.7, we can obtain the following sample complexity of DR-PO by setting T and λ appropriately:

Corollary 5.8. Suppose Assumption 5.2,5.3,5.4 hold and set

then if we have

36H3rmax2 log2 CST ϵ2

T =

, η =

1 Trmax2

ϵ 3H log CST

,

, λ =

###### M = Ω

CTR + CSFT(108H4rmax3 log3 CST/ϵ3) κ2 ϵ2

log |R| δ

,

###### N = Ω

H2rmax2 CST ϵ2

T|F| δ

log

,

we have with probability at least 1 − δ that Algorithm 1 with NPG update (Algorithm 2) returns a policy π which satisfies

⋆,r⋆(s1) − V π,r

⋆

V π

###### (s1) ≤ ϵ.

Theorem 5.7 and Corollary 5.8 indicate that DR-PO with NPG update is capable of finding an ϵ-optimal policy with polynomial sample complexity, i.e., O(1/ϵ2) labeled trajectory pairs and unlabeled trajectories. Algorithmically, our algorithm does not require pessimism and is model-free, which is much easier and more practical than the pessimistic model-based algorithm proposed in Zhan et al. (2023a).

Remark 5.9. In Theorem 5.7 and Corollary 5.8 we assume R and F are finite, but our results can be extended to infinite classes directly by replacing |R|(|F|) with their covering numbers.

### 6 Experiments

We empirically evaluate DR-PO’s ability to learn from dataset resets. First, we test how well DR-PO is able to both efficiently optimize the reward score as well as minimize the KL-divergence with the reference policy. We also test the generation quality of our resulting policies in terms of Rouge (Lin, 2004) and win rate (Rafailov et al., 2023) against human references measured by GPT4 (Achiam et al., 2023). Next, we conduct an ablation study, incrementally relaxing the the proportion of dataset resets in our online data collection to study how sensitive DR-PO is to this hyperparameter. We investigate DR-PO’s performance when transferring to another summarization task such as CNN/DailyMail (See et al., 2017). Finally, we conduct a scaling experiment on Anthropic HH by varying model sizes ranging from 1B to 7B. We find that collecting online generations with dataset resets results in a policy with a better tradeoff between reward optimization and KL-divergence, leading to improved generations over baseline RL algorithms, PPO (Schulman et al., 2017) and Direct Preference Optimizaion (DPO) (Rafailov et al., 2023).

Tasks We evaluated DR-PO on the TL;DR summarization dataset used in Stiennon et al. (2020)5 and tested scaling performance on the Anthropic Helpful Harmful (HH) task (Bai et al., 2022b). For TL;DR, a model is trained to generate summaries of online Reddit posts guided by human preference data. The task consists of two datasets: one with human reference summaries and another with preference data. Following the standards set by both Stiennon et al. (2020) and Rafailov et al. (2023), we train our reward models and DPO baseline on the preference dataset while performing online RL (for PPO and DR-PO) on the human reference dataset. We set the maximum context length to be 512 and the maximum generation length to be 53, ensuring that it is possible to generate all references in the dataset. For Anthropic HH, the model is asked to respond to a dialogue sequence in a helpful, harmless manner. We follow much of design choices from TRLx6 for dataset processing, context length, and generation length. For more details about the dataset, please see Appendix D

Evaluation To test the performance of DR-PO against our baselines we evaluate each method by its tradeoff between reward model score and KL-divergence with the reference policy, testing the effectiveness of the algorithm in optimizing the regularized RLHF objective. Furthermore, we compute the Rouge score and GPT4 win rate to evaluate the generation quality of our resulting policies. Note for our win rate calculation, we report the win rate of a randomly sampled subset (10%) of the test set for a total of 600 samples. Please see Appendix D.3 for the prompt used to query GPT4 as well as an example response. When evaluating the on CNN/DailyMail we make use of the constructed preference dataset from Stiennon et al. (2020) and for training a supervised finetuned model, we use HuggingFace’s dataset version 2.0.07.

Methods We instantiate DR-PO by using PPO style policy optimization (Schulman et al., 2017) as the policy optimizer (PO in Algorithm 1). First for TL;DR, we maintain the same pretrained LLM and supervised finetuned model for all of our experiments. For supervised finetuning, we trained a Pythia 2.8B8 (Biderman et al., 2023) parameter model for 1 epoch over the dataset with human references as labels. Similarly for the reward model, we trained a Pythia 2.8B parameter model for 1 epoch over the preference labeled dataset. Then, for DPO, PPO, and DR-PO, we trained our policy and critic with low rank adapters (LoRA) (Hu et al., 2022) on top of our supervised finetuned (SFT) model and our reward model (RM) respectively. Finally for our scaling experiments for Anthropic HH, we trained Pythia 125M, 1B, and 6.9B parameter models for 1 epoch over the HH dataset for both SFT and RM training. Please see Appendix D for details and Appendix D.2 for pseudocode to implement resets.

#### 6.1 How well can DR-PO optimize the RLHF objective?

- Table 1 compares DR-PO against PPO, DPO, and supervised finetuning. The KL-regularized reward optimization broadly used in RLHF as well as analyzed in Section 5 balances reward exploitation and deviation from a reference policy. When computing the KL-divergence, we use our SFT policy as our reference policy for all our methods. Notably, DR-PO scores a higher RM value over the test set over all baselines with a slightly larger KL discrepancy than PPO. We also see that with GPT4 win rate, DR-PO achieves the highest preference over human references showcasing the benefit of learning from resets. Figure 1 plots a

5Dataset can be obtained from https://github.com/openai/summarize-from-feedback

- 6https://github.com/CarperAI/trlx
- 7https://huggingface.co/datasets/cnn_dailymail 8HuggingFace Model Card: EleutherAI/pythia-2.8b-deduped

##### Algorithms TL;DR Summarization

Win Rate RM Score KL(π||πref) Rouge 1 Rouge 2 RougeL (↑) (↑) (↓) (↑) (↑) (↑)

SFT 31.6 ± 0.2% -0.51 ± 0.04 - 32.17 ± 1.01 12.27 ± 0.67 24.87 ± 1.22 DPO 52.6 ± 0.4% - 37.33 ± 2.01 30.03 ± 3.23 7.93 ± 1.02 22.05 ± 0.83 PPO 62.3 ± 2.5% 1.17 ± 0.13 16.32 ± 1.46 33.73 ± 2.34 11.97 ± 0.91 24.97 ± 1.03 DR-PO 70.2 ± 1.7% 1.52 ± 0.09 16.84 ± 0.83 33.68 ± 1.78 11.90 ± 0.06 25.12 ± 0.76

- Table 1: TL;DR Summarization Results: Our RM Score is under our trained preference reward model and the win rate is evaluated by GPT4. All evaluated policies except for SFT are models with LoRA adapters. We present results across 3 seeds.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 20 40

KL(π||πref) (←)

- 0
- 1
- 2

→RMScore()

TL;DR Summarization

DR-PO PPO SFT Reference

- Figure 1: Reward vs KL-Divergence Frontier: Plotting the regularized optimization tradeoff between DR-PO and our baselines over the entire test set. DR-PO is able to achieve a much better tradeoff by learning higher reward generations with lower KL. The average reference and SFT scores under the RM are shown as dashed lines.

more detailed frontier of the reward and KL tradeoff for DR-PO and PPO. We generate this plot by binning the test scores according to KL. We see that for most KL values, DR-PO is able to achieve a higher score than PPO.

- 6.2 Analysis of Dataset Reset Proportion

Next, we investigate how sensitive DR-PO is to the amount of dataset resets done during online generation. We define β as the proportion of generations in a given online batch of generations with dataset resets. More specifically, our main results are with β = 1.0 which translates to all generations during online training of DR-PO starting from a randomly sampled reset from the human references. Note that a β value of 0 recovers the baseline PPO (e.g., all generations start from initial prompts). Table 2 shows the expected RM score, KL, and win rate of DR-PO as we increase the mixing proportion from 0% (PPO) to 100% (DR-PO) after 2 epochs of training. Notably, even with a small amount of dataset resets DR-PO is able to learn higher scoring generations with a lower KL than PPO. Moreover, we see that DR-PO with any amount of reference resets leads to higher win rates than PPO. Figure 2 plots the RM score/KL-divergence frontier of our learned policies on the test set. Note that DR-PO is robust to the amount of dataset resets in optimizing the regularized RLHF objective. Finally, supporting our analysis from Section 5, DR-PO generally performs better the more online data we gather from resets with a 100% reset proportion performing the best.

- 6.3 DR-PO Transfer Performance

Finally, we investigate DR-PO’s ability to do zero-shot transfer to another summarization task, ensuring that learning a policy by reseting from human references does not diminish the generalization observed with PPO in Stiennon et al. (2020). Specifically, we investigate whether leveraging human references on TL;DR has the unintended consequence of overfitting to the specific dataset rather than learning more generally to summarize. For our baselines, we test the zero-shot capabilities of both PPO and

Algorithms Win Rate RM Score KL(π||πref)

(↑) (↑) (↓)

PPO 60.7% 1.14 15.08 DR-PO (β = 0.25) 61.7% 1.28 14.77 DR-PO (β = 0.5) 66.5% 1.28 15.63

- DR-PO (β = 0.75) 64.3% 1.25 14.32
- DR-PO (β = 1.0) 68.5% 1.47 16.65

- Table 2: DR-PO Ablation of Datset Reset Proportion: Our RM Score is under our trained preference reward model and the Win Rate is evaluated by GPT4. β represents the proportion of online data generated from dataset resets with 1.0 being all generations are from resets and 0.0 being PPO (i.e., always reset to initial prompts). The values on this table are across one seed.

- 0
- 1

ProportionofDatasetResets

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

2.5

→RMScore()

2.0

1.5

1.0

0 25

KL(π||πref) (←)

- Figure 2: Ablation of Dataset Reset: Plotting the RM score and KL-Divergence tradeoff as a function of dataset reset proportion. Blue represents no mixing while red represents every online generation starting from a reset.

DPO as well as report the performance of a supervised finetuned policy on CNN/DailyMail using the same base model, Pythia 2.8B. Table 3 demonstrates DR-PO’s zero-shot capabilities, being the only policy to outperform a supervised finetuned model on all metrics. Therefore, we see that learning from resets not only improves RLHF on the training task but also the zero-shot transfer performance to another summarization task.

- 6.4 DR-PO Scaling Performance on Anthropic HH

- Figure 3 shows DR-PO’s performance across different model scales on Anthropic HH task. Specifically we tested three model sizes: 125M, 1B, and 6.9B. We specifically trained on the Pythia models (Biderman et al., 2023) using TRLx9. We kept the decoding to be the same across all methods here with a sampling temperature of 0.01 as Rafailov et al. (2023) showed that DPO performed best with greedier sampling. We see that both SFT and DPO showed similar scaling performance gains with PPO

9https://github.com/CarperAI/trlx

##### Algorithms CNN/DM Summarization

Win Rate Rouge 1 Rouge 2 RougeL (↑) (↑) (↑) (↑)

SFT (CNN/DM) 10.5% 25.60 12.27 19.99 DPO 6.0% 20.71 9.47 15.70 PPO 8.5% 23.62 12.29 18.56 DR-PO 12.0% 29.53 15.36 22.88

Table 3: Zero-shot transfer to CNN/DM: the Win Rate is evaluated by GPT4.

and DR-PO scaling better from 1B to 6.9B parameters. Figure 3 shows that DR-PO has similar scaling improvements as PPO, but performs strictly better and produces generations that are more preferred than those from all of our baselines.

Anthropic HH Scaling

0.35

0.30

GPT4WinRate

0.25

0.20

0.15

108 109 1010 Model size

SFT DPO PPO DR-PO

Figure 3: Scaling on Anthropic HH: The GPT4 win rate of DR-PO when tested across 3 model scales: 125M, 1B, and 6.9B. Reported winrates are mean and std across 3 seeds.

### 7 Conclusion

We present DR-PO, a provably efficient algorithm that exploits a generative model’s ability to reset from offline data to enhance RLHF from preference-based feedback. Both in theory and in practice, we demonstrate the effectiveness of incorporating dataset resets into online RL. While in our experiments we specifically demonstrate dataset resets on a PPO style policy optimizer, the idea of dataset reset is both general and simple to implement into any online data collection component of other RL algorithms. We leave it to exciting future work to test the full capabilities of dataset resets in other RLHF methods.

### Acknowledgements

Wen Sun acknowledges funding from NSF IIS-2154711, NSF CAREER 2339395, and Cornell Infosys Collaboration. Jonathan Chang is supported by LinkedIn under the LinkedIn-Cornell Grant. Kiante Brantley is supported by NSF under grant No. 2127309 to the Computing Research Association for the CIFellows Project.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Agarwal, A., Jiang, N., Kakade, S. M., and Sun, W. (2019). Reinforcement learning: Theory and algorithms. Technical report. Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G. (2021). On the theory of policy gradient methods: Optimality,

approximation, and distribution shift. The Journal of Machine Learning Research, 22(1):4431–4506. Azar, M. G., Munos, R., Ghavamzadeh, M., and Kappen, H. (2011). Reinforcement learning with a near optimal rate of convergence. Technical report, INRIA. Azar, M. G., Osband, I., and Munos, R. (2017). Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 263–272. JMLR. org.

Bagnell, J. A. (2004). Learning decisions: Robustness, uncertainty, and approximation. Carnegie Mellon University. Bagnell, J. A. and Schneider, J. (2003). Covariant policy search. In Proceedings of the 18th international joint conference on

Artificial intelligence, pages 1019–1024.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022a). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. (2022b). Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Ball, P. J., Smith, L., Kostrikov, I., and Levine, S. (2023). Efficient online reinforcement learning with offline data. arXiv preprint arXiv:2302.02948.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. (2023). Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR.

Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345.

Brown, D., Goo, W., Nagarajan, P., and Niekum, S. (2019). Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations. In International conference on machine learning, pages 783–792. PMLR.

Chang, J. D., Brantley, K., Ramamurthy, R., Misra, D., and Sun, W. (2023). Learning to generate better than your llm. arXiv preprint arXiv:2306.11816.

Chen, X., Zhong, H., Yang, Z., Wang, Z., and Wang, L. (2022). Human-in-the-loop: Provably efficient preference-based reinforcement learning with general function approximation. In International Conference on Machine Learning, pages 3773–3793. PMLR.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human

preferences. Advances in neural information processing systems, 30. Contextual.ai (2023). https://contextual.ai/better-cheaper-faster-llm-alignment-with-kto/. Daumé, H., Langford, J., and Marcu, D. (2009). Search-based structured prediction. Machine learning, 75:297–325. Daumé III, H. and Marcu, D. (2005). Learning as search optimization: Approximate large margin methods for structured

prediction. In Proceedings of the 22nd international conference on Machine learning, pages 169–176. Du, S. S., Kakade, S. M., Lee, J. D., Lovett, S., Mahajan, G., Sun, W., and Wang, R. (2021). Bilinear classes: A structural framework for provable generalization in rl. Dudík, M., Hofmann, K., Schapire, R. E., Slivkins, A., and Zoghi, M. (2015). Contextual dueling bandits. In Conference on Learning Theory, pages 563–587. PMLR. Glaese, A., McAleese, N., Tre˛bacz, M., Aslanides, J., Firoiu, V., Ewalds, T., Rauh, M., Weidinger, L., Chadwick, M., Thacker,

P., et al. (2022). Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375. Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-rank adaptation

of large language models. In International Conference on Learning Representations. Jiang, N., Krishnamurthy, A., Agarwal, A., Langford, J., and Schapire, R. E. (2016). Contextual decision processes with low bellman rank are pac-learnable. arXiv preprint arXiv:1610.09512. Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. (2020). Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory, pages 2137–2143. PMLR.

Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement learning. In Proceedings of the

Nineteenth International Conference on Machine Learning, volume 2, pages 267–274. Kakade, S. M. (2001). A natural policy gradient. Advances in neural information processing systems, 14. Kakade, S. M. (2003). On the sample complexity of reinforcement learning. University of London, University College London

(United Kingdom). Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P., Ghavamzadeh, M., and Gu, S. S. (2023). Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192. Li, Z., Yang, Z., and Wang, M. (2023). Reinforcement learning with human feedback: Learning dynamic choices via pessimism. arXiv preprint arXiv:2305.18438.

Lin, C.-Y. (2004). Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81. Liu, H., Sferrazza, C., and Abbeel, P. (2023). Languages are rewards: Hindsight finetuning using human feedback. arXiv

preprint arXiv:2302.02676.

MacGlashan, J., Ho, M. K., Loftin, R., Peng, B., Wang, G., Roberts, D. L., Taylor, M. E., and Littman, M. L. (2017). Interactive learning from policy-dependent human feedback. In International Conference on Machine Learning, pages 2285–2294. PMLR.

Nair, A., McGrew, B., Andrychowicz, M., Zaremba, W., and Abbeel, P. (2018). Overcoming exploration in reinforcement learning with demonstrations. In 2018 IEEE international conference on robotics and automation (ICRA), pages 6292–6299. IEEE.

Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. (2021). Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

Novoseller, E., Wei, Y., Sui, Y., Yue, Y., and Burdick, J. (2020). Dueling posterior sampling for preference-based reinforcement learning. In Conference on Uncertainty in Artificial Intelligence, pages 1029–1038. PMLR.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.

(2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Pacchiano, A., Saha, A., and Lee, J. (2021). Dueling rl: reinforcement learning with trajectory preferences. arXiv preprint arXiv:2111.04850.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

Ramamurthy, R., Ammanabrolu, P., Brantley, K., Hessel, J., Sifa, R., Bauckhage, C., Hajishirzi, H., and Choi, Y. (2022). Is reinforcement learning (not) for natural language processing?: Benchmarks, baselines, and building blocks for natural language policy optimization. arXiv preprint arXiv:2210.01241.

Salimans, T. and Chen, R. (2018). Learning montezuma’s revenge from a single demonstration. arXiv preprint arXiv:1812.03381.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning, pages 1889–1897.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

See, A., Liu, P. J., and Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073–1083, Vancouver, Canada. Association for Computational Linguistics.

Shin, D., Dragan, A. D., and Brown, D. S. (2023). Benchmarks and algorithms for offline preference-based reward learning. arXiv preprint arXiv:2301.01392.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. (2016). Mastering the game of Go with deep neural networks and tree search. nature, 529(7587):484–489.

Song, Y., Zhou, Y., Sekhari, A., Bagnell, J. A., Krishnamurthy, A., and Sun, W. (2022). Hybrid rl: Using both offline and online data can make rl efficient. arXiv preprint arXiv:2210.06718.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. (2020). Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021.

Tiapkin, D., Belomestny, D., Calandriello, D., Moulines, E., Naumov, A., Perrault, P., Valko, M., and Menard, P. (2023). Demonstration-regularized rl.

Uchendu, I., Xiao, T., Lu, Y., Zhu, B., Yan, M., Simon, J., Bennice, M., Fu, C., Ma, C., Jiao, J., et al. (2023). Jump-start reinforcement learning. In International Conference on Machine Learning, pages 34556–34583. PMLR.

Warnell, G., Waytowich, N., Lawhern, V., and Stone, P. (2018). Deep tamer: Interactive agent shaping in high-dimensional state spaces. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

Wirth, C., Akrour, R., Neumann, G., Fürnkranz, J., et al. (2017). A survey of preference-based reinforcement learning methods. Journal of Machine Learning Research, 18(136):1–46.

Wu, J., Ouyang, L., Ziegler, D. M., Stiennon, N., Lowe, R., Leike, J., and Christiano, P. (2021). Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862.

Wu, R. and Sun, W. (2023). Making rl with preference-based feedback efficient via randomization. arXiv preprint arXiv:2310.14554.

Wu, T., Zhu, B., Zhang, R., Wen, Z., Ramchandran, K., and Jiao, J. (2023). Pairwise proximal policy optimization: Harnessing relative feedback for llm alignment. arXiv preprint arXiv:2310.00212.

Xu, Y., Wang, R., Yang, L., Singh, A., and Dubrawski, A. (2020). Preference-based reinforcement learning with finite-time guarantees. Advances in Neural Information Processing Systems, 33:18784–18794.

Yin, D., Hao, B., Abbasi-Yadkori, Y., Lazi´c, N., and Szepesvári, C. (2022). Efficient local planning with linear function approximation. In International Conference on Algorithmic Learning Theory, pages 1165–1192. PMLR.

Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J., and Weston, J. (2024). Self-rewarding language models. arXiv preprint arXiv:2401.10020.

Yue, Y., Broder, J., Kleinberg, R., and Joachims, T. (2012). The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556.

Zhan, W., Huang, B., Huang, A., Jiang, N., and Lee, J. (2022). Offline reinforcement learning with realizability and single-policy

concentrability. In Conference on Learning Theory, pages 2730–2775. PMLR. Zhan, W., Uehara, M., Kallus, N., Lee, J. D., and Sun, W. (2023a). Provable offline preference-based reinforcement learning. Zhan, W., Uehara, M., Sun, W., and Lee, J. D. (2023b). Provable reward-agnostic preference-based reinforcement learning. Zhu, B., Jiao, J., and Jordan, M. I. (2023a). Principled reinforcement learning with human feedback from pairwise or k-wise

comparisons. arXiv preprint arXiv:2301.11270. Zhu, B., Sharma, H., Frujeri, F. V., Dong, S., Zhu, C., Jordan, M. I., and Jiao, J. (2023b). Fine-tuning language models with advantage-induced policy alignment. arXiv preprint arXiv:2306.02231. Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. (2019). Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593. Zoghi, M., Whiteson, S. A., De Rijke, M., and Munos, R. (2014). Relative confidence sampling for efficient on-line ranker evaluation. In Proceedings of the 7th ACM international conference on Web search and data mining, pages 73–82.

### A DR-PO with NPG

Algorithm 3 DR-PO with NPG update

- 1: Input: labeled preference dataset DR, unlabeled dataset DTR, reward function class R, Q function class F, regularization parameter λ, stepsize η, total number of iterations T.
- 2: Initialize: π1 = πSFT.
- 3: Learn a reward model rˆ via MLE based on Eq. (1).
- 4: Let N0 ← NT . Partition DTR into {DTR,t := {τt,n}N

0

n=1}t∈[T] with an equal size. /* Policy Evaluation with Dataset Reset */

- 5: for t = 1,··· ,T do
- 6: for n = 1,··· ,N0 do
- 7: Sample h ∼ Unif([H]) and pick the state at step h from τt,n, denoted by st,nh .
- 8: Take action at,nh ∼ (12πSFT(st,nh ) + 12πt(st,nh )) and then execute πt to step H.

- 9: Denote the trajectory by (sh,ah,··· ,sH,aH) and let yht,n = Hh′=h rh′(sh′,ah′).
- 10: Add (st,nh ,at,nh ,yht,n) into Dt.
- 11: end for
- 12: Compute

Qt = arg min f∈F

LD

t

(f) :=

1 N0

(s,a,y)∈Dt

(f(s,a) − y)2 .

/* NPG Update */

- 13: Compute for all s ∈ S:

πt+1(s) = arg min

p∈∆(A)

− Qt(s,·),p + λKL(p∥πSFT(s)) +

1 η

KL(p∥πt(s)).

- 14: end for
- 15: Output: π = Unif({πt}Tt=1).

### B Proof of Theorem 5.7

First we relax the single-policy concentrability in Assumption 5.4 to the following assumptions.

- Assumption B.1 (single-policy concentrability w.r.t. the reward class). Suppose that we have:

max 0, sup

r∈R

Eτ0∼dπ⋆,τ1∼dπSFT[r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)] Eτ0∼dπSFT,τ1∼dπSFT |r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2

≤ Cr(R).

- Assumption B.2 (single-policy concentrability w.r.t. Q function class). Suppose that we have for all t ∈ [T]:

sup

h∈[H],f∈F,π∈{πt,π⋆}

Es∼dπ⋆

h ,a∼π(s) f(s,a) − Qπ

t, r(s,a)

Es∼dπSFT

h ,a∼(

- 1

- 2πSFT(s)+12πt(s)) f(s,a) − Qπt, r(s,a)

2

≤ Ceval(F)

- Assumption B.3 (single-policy concentrability w.r.t. KL divergence). Suppose that we have:

H

h∼dπh⋆ KL(π⋆(sh)∥πSFT(sh)) ≤ CKL.

Es

h=1

- Assumption B.4 (Concentrability w.r.t. bounded KL-diveregnce policies). Suppose that we have for any BKL ≥ 0:

Eτ0∼dπ,τ1∼dπSFT[r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)] Eτ0∼dπSFT,τ1∼dπSFT |r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2

≤ Cs(R,BKL).

max 0, sup

r∈R,π∈Θ(πSFT,BKL)

Note that from Cauchy-Schwartz inequality we have the following proposition: Proposition B.5. We have

dπ⋆(τ) dπSFT(τ)

dπ(τ) dπSFT(τ)

Cr(R) ≤ max

,Cs(R,BKL) ≤ max

τ

π∈Θ(πSFT,BKL),τ

dπh⋆(s,a) dπhSFT(s,a)

Ceval(F) ≤ 2 · max

,

h∈[H],s∈Sh,a∈A

⋆

dπ

h (s,a) dπhSFT(s,a)

CKL ≤ H log max

.

h∈[H],s∈Sh,a∈A

Proof. First from Cauchy-Schwartz inequality, we have

Eτ0∼dπ⋆,τ1∼dπSFT[r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)] ≤ Eτ0∼dπ⋆,τ1∼dπSFT[|r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2]. Therefore we have

Eτ0∼dπ⋆,τ1∼dπSFT[|r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2] Eτ0∼dπSFT,τ1∼dπSFT |r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2

Cr(R) ≤ sup r∈R

The bound for Cs follows the same arguments. Similarly, we have:

dπ⋆(τ) dπSFT(τ)

≤ max

.

τ

2

t, r(s,a) ≤ Es∼dπ⋆

h ,a∼π(s) f(s,a) − Qπ

Es∼dπ⋆

h ,a∼π(s) f(s,a) − Qπt, r(s,a)

###### .

Therefore we have

2

Es∼dπ⋆

h ,a∼π(s) f(s,a) − Qπt, r(s,a)

Ceval(F) ≤ sup

###### 2 .

h∈[H],f∈F,π∈{πt,π⋆}

Es∼dπSFT

- 1

- 2πSFT(s)+12πt(s)) f(s,a) − Qπt, r(s,a)

h ,a∼(

Note that we have

2

t, r(s,a)

Es∼dπ⋆

h ,a∼π⋆(s) f(s,a) − Qπ

###### 2 ≤ max

2 ·

sup

h∈[H],s∈Sh,a∈A

h∈[H],f∈F

Es∼dπSFT

- 1

- 2πSFT(s)+12πt(s)) f(s,a) − Qπt, r(s,a)

h ,a∼(

On the other hand, we know

⋆

dπ

h (s,a) dπhSFT(s,a)

.

2

t, r(s,a)

Es∼dπ⋆

h ,a∼πt(s) f(s,a) − Qπ

sup

2

h∈[H],f∈F

Es∼dπSFT

- 1

- 2πSFT(s)+12πt(s)) f(s,a) − Qπt, r(s,a)

h ,a∼(

⋆

⋆

dπ

dπ

h (s) dπhSFT(s) ≤ max

h (s,a) dπhSFT(s,a)

≤ max

2 ·

2 ·

.

h∈[H],s∈Sh

h∈[H],s∈Sh,a∈A

Therefore, we have

For CKL, we have

H

h=1

dπh⋆(s,a) dπhSFT(s,a)

Ceval(F) ≤ 2 · max

.

h∈[H],s∈Sh,a∈A

H

π⋆(a|s) πSFT(a|s)

⋆

h∼dπh⋆ KL(π⋆(sh)∥πSFT(sh)) =

dπ

π⋆(a|s)log

Es

h (s)

h=1 s∈Sh

a∈A

H

π⋆(a|s) πSFT(a|s)

⋆

dπ

=

h (s,a)log

h=1 s∈Sh,a∈A

H

H

⋆

π⋆(a|s) πSFT(a|s)

dπ

h (s) dπhSFT(s)

⋆

⋆

dπ

dπ

≤

+

h (s,a)log

h (s)log

h=1 s∈Sh,a∈A

h=1 s∈Sh

H

H

⋆

π⋆(a|s) πSFT(a|s)

dπ

h (s) dπhSFT(s)

⋆

⋆

dπ

dπ

=

+

h (s,a)log

h (s,a)log

h=1 s∈Sh,a∈A

h=1 s∈Sh,a∈A

H

⋆

⋆

dπ

dπ

h (s,a) dπhSFT(s,a) ≤ H log max

h (s,a) dπhSFT(s,a)

⋆

dπ

=

h (s,a)log

h∈[H],s∈Sh,a∈A

h=1 s∈Sh,a∈A

.

| |
|---|

With Proposition B.5, we only need to prove the following theorem to validate Theorem 5.7:

Theorem B.6. Suppose Assumption 5.2,5.3,B.1,B.2,B.4 hold. Then with probability at least 1 − δ, we have Algorithm 1 with NPG update (Algorithm 2) returns a policy π which satisfies

⋆,r⋆(s1) − V π,r

⋆

(s1) ≤ (Cs(Trmax/λ) + Cr(R))ϵMLE + ϵ′PMD, where

V π

ϵMLE := O

CKL ηT

ϵ′PMD :=

κ2 M

log |R| δ

, ϵ′eval := O

Ceval2 (F)Trmax2 N

T|F| δ

log

Hrmax2 η 2

+ λCKL + 2Hϵ′eval.

+

In this section we provide the proof of Theorem B.6. Our proof consists of three steps: we first quantify the estimation error of the Q function incurred by LSR oracles – this step only involves standard supervised learning analysis, then study the performance guarantee of NPG, and lastly investigate how to deal with the reward uncertainty and obtain the final suboptimality gap.

#### B.1 Q function Estimation Error

t, r(s,a) :

We have the following lemma to bound the estimation error Q(s,a) − Qπ

- Lemma B.7. Fix any δ1 ∈ (0,1]. With Assumption 5.3, we have with probability at least 1 − δ1 that for all t ∈ [T],

256rmax2 N0

2T|F| δ1

t, r(s,a) ≤ Ceval(F)

:= ϵ′eval,

h ,a∼π(s) Qt(s,a) − Qπ

Eh∼Unif([H]),s∼dπ⋆

log

where π ∈ {πt,π⋆}.

Proof. From the guarantee of least squares (Lemma C.1 in Appendix C), fix t ∈ [T], we have with probability at least 1 − δ1 that,

256rmax2 N0

2|F| δ1

2

t, r(s,a)

- 1

- 2πSFT(s)+12πt(s)) Qt(s,a) − Qπ

Eh∼Unif([H]),s∼dπSFT

≤

. Take union bound over t ∈ [T] and we have for all t ∈ [T] that

log

h ,a∼(

256rmax2 N0

2T|F| δ1

t, r(s,a) ≤ Ceval(F)

h ,a∼π(s) Qt(s,a) − Qπ

Eh∼Unif([H]),s∼dπ⋆

log

.

| |
|---|

#### B.2 NPG Analysis

In the following discussion we use f(s) to denote the vector f(s,·) for all functions f. We have the following lemma which indicates that NPG is able to find a near optimal policy with respect to the estimated reward r (recall that ϵeval is defined in

- Lemma B.7):

- Lemma B.8. Denote the event in Lemma B.7 by E1. Then conditioned on E1, with Assumption 5.3 and B.4, we have

V π

⋆, r(s1) − V π, r(s1) ≤

CKL ηT

+

Hrmax2 η 2

+ 2Hϵ′eval + λCKL := ϵ′PMD.

Proof. In the following proof we use g(π(s)) to denote KL(π(s)∥πSFT(s)) for any policy π. First note that from the update rule in line 8 of Algorithm 1, due to first order optimality, we know for all distribution p ∈ ∆(A) and all t ∈ [T],s ∈ S that:

−η Qt(s) + (1 + ηλ)∇g(πt+1(s)) − ∇g(πt(s)),p − πt+1(s) ≥ 0. (3) This implies that for all t ∈ [T],s ∈ S, we have

η Qt(s),π⋆(s) − πt(s) + ηλg(πt(s)) − ηλg(π⋆(s))

= η Qt(s) − (1 + ηλ)∇g(πt+1(s)) + ∇g(πt(s)),π⋆(s) − πt+1(s)

+ ∇g(πt+1(s)) − ∇g(πt(s)),π⋆(s) − πt+1(s) + η Qt(s),πt+1(s) − πt(s)

+ ηλ∇g(πt+1(s)),π⋆(s) − πt+1(s) + ηλg(πt(s)) − ηλg(π⋆(s)) ≤ ∇g(πt+1(s)) − ∇g(πt(s)),π⋆(s) − πt+1(s)

(1)

+ η Qt(s),πt+1(s) − πt(s)

(2)

+ ηλ∇g(πt+1(s)),π⋆(s) − πt+1(s) + ηλg(πt(s)) − ηλg(π⋆(s))

(3)

,

where the last step is due to Equation (3). Now we bound the term (1)(2)(3) respectively.

- Bounding term (1). Note that the KL divergence is indeed the Bregman divergence induced by g, therefore the following three point lemma holds true:

- Lemma B.9 (three point lemma). For any distributions p1(s),p2(s),p3(s) ∈ ∆(A) ,we have ⟨∇g(p1(s)) − ∇g(p2(s)),p3(s) − p1(s)⟩ = KL(p3(s)∥p2(s)) − KL(p3(s)∥p1(s)) − KL(p1(s)∥p2(s)).

Proof. From definition of g, we know ∇g(p(s)) = log p(s) − log πSFT(s) + 1. This implies that

⟨∇g(p1(s)) − ∇g(p2(s)),p3(s) − p1(s)⟩ = ⟨log p1(s) − log p2(s),p3(s) − p1(s)⟩. Substitute the definition of KL divergence and we can prove the lemma.

| |
|---|

From Lemma B.9, we can rewrite (1) as follows:

(1) = KL(π⋆(s)∥πt(s)) − KL(π⋆(s)∥πt+1(s)) − KL(πt+1(s)∥πt(s)).

- Bounding term (2). From Cauchy-Schwartz inequality, we have

(2) ≤

- 1

- 2

πt+1(s) − πt(s) 21 +

η2 2

Qt(s)

2 ∞

≤

- 1

- 2

πt+1(s) − πt(s) 21 +

η2rmax2 2

.

- Bounding term (3). Since g is convex, we know ηλ∇g(πt+1(s)),π⋆(s) − πt+1(s) ≤ ηλg(π⋆(s)) − ηλg(πt+1(s)).

This implies that

(3) ≤ ηλ g(πt(s)) − g(πt+1(s)) . In summary, we have for all t ∈ [T],s ∈ S that

η Qt(s),π⋆(s) − πt(s) + ηλg(πt(s)) − ηλg(π⋆(s)) ≤ KL(π⋆(s)∥πt(s)) − KL(π⋆(s)∥πt+1(s)) + ηλ g(πt(s)) − g(πt+1(s))

η2 2

- 1

- 2

πt+1(s) − πt(s) 21 − KL(πt+1(s)∥πt(s)) ≤ KL(π⋆(s)∥πt(s)) − KL(π⋆(s)∥πt+1(s)) + ηλ g(πt(s)) − g(πt+1(s)) +

CQ2 +

+

η2rmax2 2

, where the last step is due to Pinsker’s inequality.

This implies that

T

H

h∼dπh⋆ η Qt(sh),π⋆(sh) − πt(sh) + ηλg(πt(sh)) − ηλg(π⋆(sh))

Es

t=1

h=1

H

h∼dπh⋆ KL(π⋆(sh)∥π1(sh)) − KL(π⋆(sh)∥πT+1(sh))

Es

≤

h=1

H

HTrmax2 η2 2

h∼dπh⋆ g(π1(sh) − g(πT+1(sh))) +

Es

+ ηλ

h=1

H

HTrmax2 η2 2 ≤ CKL +

HTrmax2 η2 2

h∼dπh⋆ KL(π⋆(sh)∥πSFT(sh)) +

. (4)

Es

≤

h=1

Note that here we use the fact that we initialize the policy as π1 = πSFT and thus g(π1(s)) = 0. On the other hand, note that we have the following performance difference lemma, whose proof is deferred to Appendix C.3:

- Lemma B.10 (performance difference lemma). For any policy π,π′ and reward function r, we have:

H

′,r(s1) =

V π,r(s1) − V π

h=1

′,r(sh),π(sh) − π′(sh) .

h∼dπh Qπ

Es

Now substitute Lemma B.10 into Equation (4), and from Lemma B.7 we have

This is equivalent to

T

1 T

t=1

CKL ηT

⋆, r(s1) − V π

t, r(s1) ≤

V π

Hrmax2 η 2

+ 2Hϵ′eval + λCKL.

+

Hrmax2 η 2

CKL ηT

⋆, r(s1) − V π, r(s1) ≤

+ 2Hϵ′eval + λCKL, which concludes our proof.

V π

+

| |
|---|

We also would like to bound the KL divergence between π and πSFT as shown in the following lemma:

- Lemma B.11. We have for all t ∈ [T],s ∈ S that

rmax(t − 1) λ

KL(πt(s)∥πSFT(s)) ≤

.

Proof. From the NPG update and use the fact that πt+1 is the minimizer, we know for all t ∈ [T],s ∈ S:

rmax λ

1 λ

Qt(s),πt+1(s) − πt(s) ≤

KL(πt+1(s)∥πSFT(s)) − KL(πt(s)∥πSFT(s)) ≤

,

where we utilize Assumption 5.3 in the second step. Note that KL(π1(s)∥πSFT(s)) = 0 since π1 = πSFT. This implies that for all t ∈ [T]:

rmax(t − 1) λ

KL(πt(s)∥πSFT(s)) ≤

.

| |
|---|

#### B.3 Unregularized Suboptimality Gap w.r.t. r⋆ Now we can start to prove Theorem 5.7. First we have

⋆,r⋆(s1) − V π,r

⋆

⋆,r⋆(s1) − V π

⋆, r(s1) + V π

⋆, r(s1) − V π, r(s1) + V π, r(s1) − V π,r

⋆

V π

(s1) = V π

(s1)

⋆, r(s1) − V π, r(s1)

+ V π

= Eτ∼dπ⋆ [r⋆(τ) − r(τ)] − Eτ∼dπSFT [r⋆(τ) − r(τ)]

(1)

(2)

T

1 T

Eτ∼dπSFT [r⋆(τ) − r(τ)] − Eτ∼dπt [r⋆(τ) − r(τ)]

+

t=1

(3)

###### .

Next we bound term (1)(2)(3) respectviely.

- Bounding term (1). From the guarantee of MLE (Lemma C.2 in Appendix C) we have with probability at least 1 − δ2 that

Eτ0∼dπSFT,τ1∼dπSFT |r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2 ≤

c1κ2 log(|R|/δ2) M

:= ϵ2MLE, (5) where c1 > 0 is a universal constant. Denote the event of the above inequality by E2. Then conditioned on E2, we have

(1) = Eτ0∼dπ⋆,τ1∼dπSFT[r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)] ≤ Cr(R) Eτ0∼dπSFT,τ1∼dπSFT |r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2 ≤ Cr(R)ϵMLE.

- Bounding term (2). From Lemma B.8, conditioned on E1, we have

⋆, r(s1) − V π, r(s1) ≤ ϵ′PMD.

V π

- Bounding term (3). Note that from Lemma B.11, we have πt ∈ Θ(πSFT,(t−1)rmax/λ) for all t ∈ [T]. Therefore, following the same arguments as we have to bound term (1), we have for all t ∈ [T],

Eτ∼dπSFT [r⋆(τ) − r(τ)] − Eτ∼dπt [r⋆(τ) − r(τ)] ≤ Cs((t − 1)rmax/λ)ϵMLE, which implies that

(3) ≤ Cs(Trmax/λ)ϵMLE Overall, we have conditioned on event E1 ∩ E2,

⋆,r⋆(s1) − V π,r

⋆

(s1) ≤ (Cr(R) + Cs(Trmax/λ))ϵMLE + ϵ′PMD. We finish the proof by letting δ1 = δ2 = δ/2.

V π

### C Auxiliary Lemmas

#### C.1 Least Sqaures Guarantee

Lemma C.1 (Lemma 15 in Song et al. (2022)). Fix any R > 0, δ ∈ (0,1) and assume we have a class of real valued functions

- H : X  → [−R,R]. Suppose we have K i.i.d. samples {(xk,yk)}Kk=1 where xk ∼ ρ and yk is sampled via the conditional probability p(· | xk):

yk ∼ p(· | xk) := h∗(xk) + ϵk,

where h∗ ∈ H and {ϵk}Kk=1 are independent random variables such that E[yk | xk] = h∗(xk). Additionally, suppose that maxk |yk| ≤ R and maxx |h∗(x)| ≤ R. Then the least square solution h ← arg minh∈H Kk=1 (h(xk) − yk)2 satisfies with probability at least 1 − δ,

256R2 log(2|H|/δ) K

2

Ex∼ρ h(x) − h∗(x)

≤

.

The proof is the same as in Song et al. (2022) and thus is omitted here.

#### C.2 Maximum Likelihood Estimation Guarantee

- Lemma C.2. With Assumption 5.2, we have with probability at least 1 − δ that

Eτ0∼dπSFT,τ1∼dπSFT |r⋆(τ0) − r⋆(τ1) − r(τ0) + r(τ1)|2 ≤

c1κ2 log(|R|/δ) M

,

where c1 > 0 is a universal constant. Proof. The proof largely follows the proof of Theorem 1 in Zhan et al. (2023a). Specifically, we have the following lemma from Zhan et al. (2023a):

- Lemma C.3 (Lemma 2 in Zhan et al. (2023a)). Fix any δ ∈ (0,1]. Then with probability at least 1 − δ, we have that for all reward function r ∈ R,

M

Pr⋆(om|τm,0,τm,1) Pr(om|τm,0,τm,1)

+ log |R| δ

2 1

c1 M

Eτ0,τ1∼dπSFT Pr(·|τ0,τ1) − Pr⋆(·|τ0,τ1)

≤

log

.

m=1

Then from Lemma C.3, since Mm=1 log Pr⋆(om|τm,0,τm,1) ≤ Mm=1 log P r(om|τm,0,τm,1), we have with probability at least 1 − δ:

c1 log |R|δ M

2 1

Eτ0,τ1∼dπSFT P r(·|τ0,τ1) − Pr⋆(·|τ0,τ1)

. (6)

≤

Task Train/Val/Test Prompt Gen. Length TL;DR 117K/6.45K/6.55K "TL;DR: " 53

CNN/DailyMail 287K/13.4K/11.4K "TL;DR: " 64

Table 4: Train, val, test splits, prompts, and max generation length used for each task.

Then under Assumption 5.2, we can apply the mean value theorem between r⋆(τ1) − r⋆(τ0) and r(τ1) − r(τ0) to (6) and ensure that

c1κ2 log |R|δ M

Eτ0,τ1∼dπSFT[|(r⋆(τ1) − r⋆(τ0)) − ( r(τ1) − r(τ0))|2] ≤

,

where κ := inf 1

x∈[−rmax,rmax] Φ′(x) measures the non-linearity of the link function Φ.

| |
|---|

- C.3 Performance Difference We restate and prove Lemma B.10 as follows.

- Lemma C.4. For any policy π,π′ and reward function r, we have:

H

′,r(s1) =

V π,r(s1) − V π

h=1

′,r(sh),π(sh) − π′(sh) .

h∼dπh Qπ

Es

Proof. For any two policies π,π′ and reward r, we have that

′,r(s1)

V π,r(s1) − V π

′,r(s1)

=Eπ [r(s1,a1) + V π,r(s2)] − Eπ V π

′,r(s1,a1) − V π

′,r(s2) + V π,r(s2) − Eπ V π

′,r(s1)

=Eπ Qπ

′,r(s2) + Eπ Qπ

′,r(s1,a1) − V π

′,r(s1)

=Eπ V π,r(s2) − V π

′,r(s2) + Es

′,r(s1,·),π(·|s1) − π′(·|s1)

=Eπ V π,r(s2) − V π

1∼dπ1 Qπ

H

′,r(sh),π(sh) − π′(sh) .

h∼dπh Qπ

Es

=··· =

h=1

This concludes our proof.

| |
|---|

### D Additional Experiment Details

#### D.1 Experiment Hyperparameters and Task Details

##### D.1.1 Task Details

We present dataset specific details in table 4 For both datasets we obtained the training data from https://github.com/openai/summarize-from-feedback.

#### D.2 Dataset Reset Implementation Details

Here is a code snippet of the logit processor that handles dataset resets from references for a HuggingFace transformers model. β here represents the proportion of generations in the batch to do resets for.

import torch import numpy as np from transformers import LogitsProcessor

class ResetProcessor(LogitsProcessor):

def __init__(self, references, beta, rng, seq_lens): self.counter = 0 self.references = references self.seq_lens = seq_lens self.create_mask(beta, rng)

def create_mask(self, beta, rng): batch_size, seq_len = self.references.shape[:2] # Mixin init_mask = rng.choice(

[True, False], size=(batch_size, 1), p=[beta, 1-beta]

) init_mask = np.tile(init_mask, (1, seq_len)) # Rollin Selection length_masks = np.tril(np.ones((seq_len, seq_len))) masks = [] for length in self.seq_lens:

if length < 2:

masks.append(np.zeros((seq_len)).astype(bool)) else:

masks.append(

rng.choice(length_masks[: length - 1, :] ).astype(bool))

self.rollin_mask = np.stack(masks) self.rollin_mask[~init_mask] = False

def __call__( self, input_ids: torch.LongTensor, scores: torch.FloatTensor

) -> torch.FloatTensor: vocab_size = scores.size(-1) new_scores = one_hot(

self.references[:, self.counter], num_classes=vocab_size ).float() new_scores[new_scores == 0] = -float("inf") mask = self.rollin_mask[:, self.counter] assert scores.shape == new_scores.shape

new_scores = new_scores.to(scores.device) # Only do Teacher Forcing on the rollins scores[mask] = new_scores[mask] self.counter += 1 return scores

##### D.2.1 Computation

Note since we start with the references from the dataset, the computational requirements to generate with resets are the same as generating from the initial state distribution. For all of our experiments, we ran with the same per device batch size between PPO and DR-PO. For this work, we made use of 16 A6000 gpus with 48GB of VRAM. We used 4 gpus for each run.

- D.3 Details on GPT4 Winrate For winrate calculation, we used the following prompt:

Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post: {{post}}

- ### Summary A:

- {{summarya}}

### Summary B:

- {{summaryb}}

### Instructions: FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format: Comparison: <one-sentence comparison and explanation> Preferred: <"A" or "B">

##### D.3.1 Win Rate Example

Here is an example of getting a one sentence explanation as to why GPT4 chose certain generations for the winrate.

Prompt SUBREDDIT: r/AskReddit TITLE: How do you get someone out of your head? POST: Hi, I’m 22, and I have been with my girlfriend for 5 years now. We recently moved together. We’ve always loved each other intensely. Problem, I recently started to have feelings for an other person (a friend). This person has had a boyfriend for now 3 years, and has absolutely no ideas. Those feelings were so strong, it was hard to hide them. After 2 months of me being distant and really sad, my girlfriend forced me to say what was bothering me. I’m not a good liar, and now she knows. We decided to give us a week alone, I went to my parents. Now, I’m completely lost. I keep on thinking about this person, and I hate that. I would like for those feelings to go away, to leave me alone. But I can’t. What do I do? It’s been 3 months now, and I’m just desperate. TL;DR: \end{lstlisting}

\textbf{DR-PO Generation (Summary A)} \begin{verbatim}

- I recently started to have feelings for someone else, my girlfriend knows, we decided to give ourselves a week alone, now I’m completely lost, I hate that, what do I do

##### Reference (Summary B)

long relationship; fell in love with an other person; admitted it; would like it to disappear, though it doesn’t.

##### GPT4 Explanation for Choosing DR-PO

Summary A is more detailed and better represents the tone and content of the forum post, while Summary B is too abbreviated and omits important details such as the fact that the poster’s girlfriend now knows about their feelings.

- D.4 Examples from Test Prompt 1 SUBREDDIT: r/AskReddit

TITLE: My friend works at a California Charter School and their principal is disregarding their charter and threatening to fire anyone who questions his leadership. What can she do?

POST: I’m not an educator but my friend is having a really hard time at her job right now and I was wondering if anyone could help.

**The Issue:** She works for a charter school in California where the principal has gone off the deep end. At first it started casually with cuts and a few firings (understandable at the time because of budget cuts). It steadily got worse however as he slowly stripped the teachers of any rights/protections they once had. Now he’s threatening to fire them at any time if they refuse to work late/take personal days/attempt talks with the teachers union/show signs of disagreeing with the principal’s leadership, etc.

This guy has continually canceled meetings by any form of teacher-staffed governing boards that would normally provide the teachers with a voice. Now he just dictates policy for the school with no regard or input from the teachers. He has threatened to let anyone go if they show even the slightest dissatisfaction with his decisions.

Keep in mind this was a very good school before this guy came in. It’s not like it was a failing school in need of some tough love. As far as scholastic achievement goes, not much has changed since this guy was hired on (which is why I suspect he still has a job despite complaints to district).

Is there anything my friend can do to oust this guy or at least protect her job? TL;DR:

##### DR-PO

My friend works at a California charter school where the principal is disregarding their charter and threatening to fire anyone who questions his leadership. What can she do?

##### Prompt 2

SUBREDDIT: r/AskReddit TITLE: I am in the financial and insurance industry, but is looking to go back to school to become a vet. Any suggessions? POST: I’m not 100% being a Veternarian is something I really want to do in life. A bit of background. I’m currently 24, graduated with a finance degree 3 years ago. I’m currently a financial rep focusing on selling insurance. Even though I don’t hate my job I feel like it doesn’t fit my personality. But I stuck with it for a while because I don’t like to quit easily. The idea of becoming a vet happened when a chinchilla of mine passed because I didn’t take it to the vet in time, and I really love animals. People tell me I’m a warm person and great with kids and animals. I feel like this is something I came up with myself and not what other want me to be. So back to reality, I did a tiny bit of research on this. The closest grad school that has this program is Cornell university (I’m located in NY) I’d probably need to do an undergrad in science or medical field. I’m a little unsure of 8 more years of school, but I guess if there’s no other choice. Another concern is money, I only have about 10k in bank, I cannot touch my retirement and life insurance money. I’m not sure if I can qualify for any federal grant. I may also need to move back with my parents. One idea I have is to just brokage product with high residuals while I wait to get into a school. I’ll also appraciate any insight and experiences from a vet or a person going back to school. TL;DR:

##### DR-PO

I’m in the financial and insurance industry, but am looking to go back to school to become a vet. Any suggestions or experiences from a vet or someone going back to school.

- D.5 Hyperparameters We write the relevant hyperparameters from our experiments for DPO, PPO, SFT, and DRPO in table 5.

Setting Values model Pythia 2.8B (HuggingFace Model Card: EleutherAI/pythia-2.8b-deduped)

PPO train epochs: 1 batch size: 512 num epochs: 4 num minibatches: 1 learning rate: 3e-6 schedule: linear decay discount factor: 1 gae λ: 0.95 clip ratio: 0.2 value function coeff: 0.1 kl coefficient: 0.05

DR-PO mixing parameter (β): 1 DPO batch size: 64

β: 0.05 learning rate: 3e-6 schedule: linear decay num train epochs: 1

Reward Model batch size: 64 learning rate: 3e-6 schedule: linear decay num train epochs: 1

SFT batch size: 64 learning rate: 3e-6 schedule: linear decay num train epochs: 1

LoRA Adapter Config r: 1024 α: 2048 dropout: 0.0 bias: False

Decoding sampling: true top k: 0.0 top p: 1.0 min length: 53 max new tokens: 53 temperature: 0.1

Tokenizer padding side: left truncation side: left max length: 563

- Table 5: Hyperparameters used for TL;DR and CNN/DailyMail. Note that DP-RO and PPO share the same parameters (other than mixing proportion). All processes use the same decoding, LoRA config, and tokenizer parameters.

Algorithms RM TL;DR Accuracy RM CNN/DM Accuracy

RM 66.21% 67.48% DPO 65.92% 67.28%

RM w/ LoRA 62.87% 66.75% DPO w/ LoRA 66.14% 61.78%

- Table 6: Reward Model Transfer to CNN/DM: The accuracy of the RM and DPO’s implicit learned reward in accuracy predicting the preference. We evaluate models trained with and without LoRA on TL;DR. We also report the zero-shot performance of these models on the CNN/DailyMail preference dataset from Stiennon et al. (2020).

#### D.6 Additional Experiments

Shown in Table 6, we investigate DPO’s implicit learned reward accuracy to our RM’s accuracy on both TL;DR and CNN/DailyMail’s test sets. Furthermore, we also report the effects of LoRA on the RM and DPO performance. We see that DPO without LoRA has comparable preference accuracy on CNN/DM as our RM. Thus, we used the DPO policy without LoRA when comparing against PPO and DR-PO in Table 3.

