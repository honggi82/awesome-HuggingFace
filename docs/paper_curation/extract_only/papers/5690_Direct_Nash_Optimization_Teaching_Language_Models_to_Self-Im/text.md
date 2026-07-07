# arXiv:2404.03715v1[cs.LG]4Apr2024

## Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences

Corby Rosset∗ Ching-An Cheng Arindam Mitra Michael Santacroce Ahmed Awadallah∗ Tengyang Xie∗

Microsoft Research

Abstract

This paper studies post-training large language models (LLMs) using preference feedback from a powerful oracle to help a model iteratively improve over itself. The typical approach for post-training LLMs involves Reinforcement Learning from Human Feedback (RLHF), which traditionally separates reward learning and subsequent policy optimization. However, such a reward maximization approach is limited by the nature of “point-wise” rewards (such as that of the Bradley-Terry model), which fails to express complex intransitive or cyclic preference relations. While advances on RLHF show reward learning and policy optimization can be merged into a single contrastive objective for stability, they yet still remain tethered to the reward maximization framework. Recently, a new wave of research sidesteps the reward maximization presumptions in favor of directly optimizing over “pair-wise” or general preferences. In this paper, we introduce Direct Nash Optimization (DNO), a provable and scalable algorithm that marries the simplicity and stability of contrastive learning with theoretical generality from optimizing general preferences. Because DNO is a batched on-policy algorithm using a regression-based objective, its implementation is straightforward and efficient. Moreover, DNO enjoys monotonic improvement across iterations which helps it improve even over a strong teacher (such as GPT-4). In our experiments, a resulting 7B parameter Orca-2.5 model aligned by DNO achieves the state-of-the-art win-rate against GPT-4-Turbo of 33% on AlpacaEval 2.0 (even after controlling for response length), an absolute gain of 26% (7% → 33%) over the initializing model. It outperforms models with far more parameters, including Mistral Large, Self-Rewarding LM (70B parameters), and older versions of GPT-4. Our ablation studies analyze critical design decisions surrounding the choice of preference pairs, and the use of LLMs-as-preference-annotators. These results underscore the promise of DNO in the LLMs post-training, as well as offer actionable insights for the AI research community.

### 1 Introduction

The field of artificial intelligence is evolving towards advanced models that can understand, reason, follow complex instructions, and create nuanced content, while aligning with human values and preferences. Large Language Models (LLMs) (e.g., Brown et al., 2020; Ouyang et al., 2022; Touvron et al., 2023; OpenAI et al., 2023) have demonstrated remarkable capabilities in generating human-like text, answering questions, and coding, yet they still face challenges in tasks that require a high degree of reliability, safety, and ethical alignment. To address these challenges, fine-tuning LLMs using Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017; Bai et al., 2022a; Ouyang et al., 2022) has demonstrates strong potential for making LLMs more helpful by aligning them with human values.

The RLHF framework has been long studied in the context of preference-based reinforcement learning (RL) or RL from human preferences (e.g., Knox and Stone, 2008; Akrour et al., 2012; Griffith et al., 2013; Wirth et al., 2017; Christiano et al., 2017). The conventional methods for RLHF typically assume that the preference is determined by a scalar reward function through some model, such as the frequently used Bradley-Terry (BT) model (Bradley and Terry, 1952).1 RLHF then optimizes toward the preference in a two-step procedure: reward learning, and policy optimization (through RL) to maximize the learned reward. Under certain conditions, the two-step procedure can be

∗Correspondence to {corbyrosset,hassanam,tengyangxie}@microsoft.com 1We use “reward model” to denote a framework that translates preferences into rewards, e.g., Bradley-Terry, while “reward function” is a (possibly

learned) function that outputs reward scalars.

Win Rate

##### LC Win Rate

40%

45%

40%

35%

35%

30%

30%

25%

25%

20%

20%

15%

15%

10%

10%

5%

5%

0%

0%

Figure 1: Direct Nash Optimization achieves state-of-the-art results for a 7B parameter large language model, being the first to surpass 30% in both raw win-rate and length-controlled (LC) win-rate against GPT-4-Turbo. Win Rate and LC Win Rate have 0.93 to 0.98 correlation with ChatBot Arena scores.

streamlined into a single-step contrastive learning approach (Rafailov et al., 2023), eliminating the need for explicit reward learning. Algorithms of this kind (e.g., Rafailov et al., 2023, DPO) leverage the insight that a policy can be expressed equivalently by an “internal reward function” that the policy is optimal to, so they reduce the RLHF problem to regressing the policy’s internal reward function to that of the preference model. These algorithms are originally offline, and boast enhanced stability and ease of optimization. Nonetheless, two-step RLHF algorithms and their single-step contrastive variants still fundamentally rely on the reward maximization framework, wherein reward-based preferences are governed by, e.g., the BT model.

The reward maximization framing poses a major limitation. Reward functions, defined to output a scalar score r(x,y) for a single response y to input x, cannot express general preferences y ≻ y′ | x between a pair of outputs in all cases, e.g., intransitive or cyclic preferences (Elo, 1978). Hence, LLMs trained under reward maximization cannot always align with human preference. Furthermore, recent works show that even in settings where preferences can be perfectly expressed under the reward-based BT models, optimizing towards rewards yields problematic behaviors; we refer the reader to Bertrand et al. (2023); Azar et al. (2023); Munos et al. (2023) for more details. Lastly, reward functions in practice can quickly become “stale” as the distribution of the policy shifts under training (Ross et al., 2011; Cheng

- et al., 2023; Azar et al., 2023; Munos et al., 2023) – leaving them vulnerable to “reward hacking” (Amodei et al., 2016)

Inresponsetotheseweaknesses, anappealinglineofworkonRLHFproposestodirectlyoptimizethegeneralpreference function itself, instantiated as some oracle. These studies re-frame RLHF as finding a Nash equilibrium of a two-player game with “payoffs” from a regularized (Munos et al., 2023) or un-regularized (Swamy et al., 2024) general preference function. To solve it, they further approximate such Nash equilibrium using single-player algorithms by leveraging the symmetry of the preference function. Then, they instead define the reward of a response as the expected win-rate against the policy’s own behavior, as judged by the preference function, e.g., r(x,y) = Ey′∼π(·|x) [P(y ≻ y′ | x)]. Hence, rewards are maximized by responses that are preferred over the policy’s expected response, and a Nash equilibrium is achieved when both players deploy a π⋆ that is preferred over any competing policy. However, these proposed single-player algorithms are primarily (purely) on-policy, and they sometimes require a separately estimated preference function or a time-varying reward function. How to scale these algorithms up faithfully is still under-investigated.

We are motivated to overcome two separate challenges: the limited expressivity of reward-based RLHF, and the lack of clarity on how to scale up optimizing with respect to general preferences. Recent advances in reward-based optimization e.g., DPO, already have efficient and scalable implementations – we seek a similarly efficient solution under the framework of general preferences.

We propose a provable and scalable RLHF algorithm – Direct Nash Optimization (DNO) (Algorithm 1) that achieves the best of both worlds, combining the scalability of contrastive objectives with the theoretical soundness of general

preference optimization. DNO is designed as a batched on-policy algorithm with a regression-based learning objective; this design choice makes DNO stable and scalable, striking a balance between deployment efficiency and adaptability.

We summarize at a high level the key ingredients and insights of DNO below.

- 1. To address the issue that reward functions cannot express general preferences, we leverage recent insights that the notion of reward of ought to be expressed as expected win-rates with regard to a general preference function.2
- 2. To address the issue found in previous work that optimizing this more general objective with online algorithms is sample-inefficient or unstable, we decompose the learning procedure into a sequence of “batched on-policy” iterations, wherein each step instead optimizes a simple regression objective.
- 3. The regression objective (we choose binary cross-entropy) aligns the “internal reward function” of the policy to the expected win-rate compared with itself (as defined in Line 3 of Algorithm 1). By sampling outputs from the current policy to use for training (i.e., “self-play”), this procedure incentivizes self-improving behavior.
- 4. Our framework is general enough to admit off-policy samples into training, importantly, those from a more powerful teacher (See choice of µ1 and µ2 in Algorithm 1).
- 5. Furthermore, to ensure stability and computational efficiency, we propose a filtering scheme such that the reward regression is only performed on preference pairs with a sufficiently large margin (for theoretical explanation, see Section 4; in practice, see Section 5.2).
- 6. DNO repeats this procedure for multiple iterations to let the policy optimize toward the general preference. Since each step involves a regression problem it can be easily implemented at scale.

Theoretically, we prove DNO converges to the intended Nash equilibrium on average, and that it can improve monotonically across iterations (see Section 3.1). Furthermore, our finite-sample analysis shows that approximation error at any iteration between the learned policy and the target is tightly bounded (Theorem 1).

On the practical side, we provide a scalable implementation of DNO (Algorithm 2): an iterative self-improving algorithm with contrastive updates, which approximates Algorithm 1 under several critical design choices. Those choices include: sampling multiple online outputs from the policy being trained, using GPT-4 as the preference oracle, comparing onpolicy samples to GPT-4’s own (teacher) outputs, and training only on pairs with “large margin” (for theoretical explanation, see Section 4; in practice, see Section 5.2).

The primary distinction of our work over related works of Nash-MD (Munos et al., 2023) and SPO (Swamy et al., 2024) is that they both exhibit sample efficiency issues (two timescale updates or sample-inefficient RL steps), and both use purely on-policy samples. We resolve the efficiency issue with a sample-efficient objective that works in practice, and DNO is more flexible to incorporate off-policy samples from e.g., a powerful teacher.

Most importantly, DNO works in practice – we provide comprehensive empirical evaluations, resulting in state-of-the-art performance:

- • The resulting 7B parameter Orca-2.5 model, aligned using the practical implementation of DNO (Algorithm 2), achieves the state-of-the-art win-rate of any 7B model, exceeding 33% against GPT-4-Turbo beyond on the AlpacaEval 2.0, even after controlling for length. This is an over 26% absolute gain (7%→33%) compared to the initialized model. It outperforms several recent advanced closed-source models, including Mistral Large and GPT-4-0613, as well as open-source models with far more (10×) parameters, such as Self-Rewarding LM (Yuan et al., 2024) which has 70B parameters.
- • Our thorough ablation studies in Section 5.2 examine critical design touchpoints surrounding choice of loss function(supervisedfinetuningorcontrastive), trainingparadigm(withorwithouton-policysamples), preference annotator quality (large margin or not), and training pair construction (self-play, teacher-vs-student, etc). Our findings highlight that carefully-crafted methods encoded in Algorithm 2 lead to substantial gains.
- • We show some examples of outputs across iterations which demonstrate qualitative improvements such as better addressing nuanced issues and presumptious questions (Table 5), better organization and clarity while refraining from making misleading statements (Table 6), and higher information density in answers (Table 7).

We hope that the results presented herein will provide clarity to the community regarding the use of AI feedback for post-training LLMs.

- 2E.g., for a fixed y | x, the expected win-rate of y against the policy itself is: Ey′∼π(·|x) [P(y ≻ y′ | x)].

### 2 Preliminaries

This section provides an overview of the RL from human feedback (RLHF) pipeline. We do not differentiate between RLHF and RLAIF (e.g., Bai et al., 2022b; Lee et al., 2023), as the distinction is outside our scope of discussion. Thus, we will uniformly refer to both concepts as RLHF. However, we want to make a clear delineation between two subtle differences: RLHF maximizing point-wise reward functions, and RLHF optimizing general preferences. It should be noted that this discussion is more broadly applicable in scope to general contextual bandits setup as well.

Throughout this paper, we use x ∈ X to denote the input (i.e. the prompt) received by the LLM from a space X. In this paper, we do not consider the distribution shift over the prompts, following the standard contextual bandits setup of RLHF (e.g., Ouyang et al., 2022; Rafailov et al., 2023), and we use ρ to denote the distribution of the prompts. We use y ∈ Y to denote the response from the LLM given the prompt x (this corresponds to action in the contextual bandits setup). We also use π : X → ∆(Y) to denote the policy, which is a LLM here, and Π is the policy class.

Our discussion throughout this paper will also regularly involve the following three learning paradigms, which are originally introduced and commonly used in the RL literature:

- (1) Offline: The learning algorithm operates without any active data collection, e.g., sampling from the current policy. The algorithm relies solely on an offline dataset for training.
- (2) Purely on-policy: technically, online on-policy. In this setup, learning takes place by sampling outputs from the latest policy and immediately updating it based on the newly collected data. No data reuse or additional offline data is considered.
- (3) Batched on-policy3: This setup is the middle of the offline and purely on-policy setups, striking a balance between deployment efficiency and adaptability. It involves iterative online data collection and can use other offline data. Its distinctive feature is that here the data collection in each iteration occurs in a batched fashion (e.g., akin to a dataset scale, much larger than the size of a typical mini-batch), and the amount of policy change can be more significant (e.g., running gradient steps over multiple epochs of a dataset, as opposed to tens of updates).

#### 2.1 RLHF Based on Reward Models

One typical approach to conducting RLHF is a two-step procedure through a reward function (Christiano et al., 2017). Suppose a preference dataset Dpref := {(x,y+,y−)} is given, where (y+,y−) ∼ πref(· | x), πref is some reference policy such as the policy obtained after supervised fine-tuning (SFT), and a preference y+ ≻ y− | x is labeled by some human or AI annotator. In RLHF with reward functions, the preference is assumed to be generated based on some latent reward r⋆. The first step is to learn a reward function r ∈ R under some reward model assumption, where R is the reward class. A number of reward model assumptions have been studied, and the Bradley-Terry (BT) model (Bradley and Terry, 1952) is the most commonly used one. The BT model assumes the probability of y+ ≻ y− | x satisfies

exp(r⋆(x,y+)) exp(r⋆(x,y+)) + exp(r⋆(x,y−))

P(y+ ≻ y− | x) :=

.

This leads to the maximum-likelihood reward learning objective:

E(x,y+,y−)∼D log σ(r(x,y+) − r(x,y−)) , (1)

r ← argmax

r∈R

where σ(·) := 1+exp(exp(·)·) is the sigmoid function. After that, the LLM is finetuned using the learned r with RL,

π(x,y) πref(x,y)

, (2)

Ex∼D

π ← argmax

pref,y∼π(·|x) r(x,y) − β log

π∈Π

3We acknowledge abuse of terminology. Our algorithm is not entirely online, as it only contains batched data collection. It is also not strictly on-policy because it uses examples from other policies, like a teacher. While “offline” or “off-policy” may be technically more relevant, they might lead to misunderstanding among readers and detract from the emphasis we want to place on the collection samples from the current policy, which constitute the majority of our training data.

ref(x,y) , is used to mitigate overoptimization of the reward model (Ouyang et al., 2022), controlled by the coeffcient β. For the purposes of our discussion, we will call the objective above as “PPO objective”, and this two-step learning procedure as “PPO”.

where the KL penalty term, Ex∼D

pref,y∼π(·|x) β log ππ(x,y)

DPO. Direct Preference Optimization (DPO) is proposed by Rafailov et al. (2023) as an alternative RLHF approach for combining the two-step procedure of PPO into a single objective. It utilizes the closed form solution π in Eq. (2), so that solving π directly from Eq. (1) becomes possible via

π(y− | x) πref(y− | x)

π(y+ | x) πref(y+ | x) − β log

E(x,y+,y−)∼Dpref log σ β log

π ← argmax

π∈Π

.

#### 2.2 RLHF with General Preferences

We now introduce the setup for directly optimizing a general preference function, as well as provide an overview of existing solutions to achieve this goal (mostly by leveraging the symmetry of the preferences), especially those proposed by Munos et al. (2023); Swamy et al. (2024).

Here we assume that the learner is given query access to a general preference function P(y ≻ y′ | x) ∈ [0,1], for any (x,y,y′) ∈ X × Y × Y. This function indicates the probability that action y is preferred over y′ given the context x. In practice, this setup can be viewed as the theoretical mode of RLAIF (e.g., Bai et al., 2022b; Yuan et al., 2024), human-in-the-loop RLHF (e.g., Ouyang et al., 2022), or distillation fine-tuning (e.g., Tunstall et al., 2023).

One common difficulty in optimizing a general preference function is its intransitivity, e.g., it is possible that P(a ≻ b) = P(b ≻ c) = P(c ≻ a) = 1, for some options (a,b,c) (details see, e.g., Bertrand et al., 2023; Munos et al., 2023; Swamy et al., 2024). Therefore, the learning goal of optimizing general preferences can be the Nash equilibrium of the two-player zero-sum game with the payoffs as the general preference function P. The formal definition of such Nash equilibrium is defined by the Minimax Winner, MW (see, e.g., Kreweras, 1965; Simpson, 1969; Kramer, 1973; Fishburn, 1984), or the von Neumann Winner (see, e.g., Dudík et al., 2015),

where

MW(P) := argmax

argmin

π′∈Π

π∈Π

P(π ≻ π′) = argmax

min

π∈Π

P(π ≻ π′),argmin

max

π′∈Π

π∈Π

π′∈Π

P(π ≻ π′) , (3)

P(π ≻ π′) := Ex∼ρ,y∼π(·|x),y′∼π′(·|x) [P(y ≻ y′ | x)].

SPO. To approximate the Nash equilibrium as defined in Eq. (3), Swamy et al. (2024) proposed a single-player algorithm, SPO. This algorithm applies results from no-regret algorithms (e.g., Freund and Schapire, 1997). The SPO algorithm is executed essentially using the following two-step iterative process: for each t = 1,2,...,T,

- (i) rt(x,y) ← Ey′∼πt(·|x) [P(y ≻ y′ | x)], ∀(x,y) ∈ X × Y
- (ii) πt+1(· | x) ←

rt(x,·) η

1 Zt(x)

, ∀x ∈ X, (4)

πt(· | x)exp

where η is the learning rate, π1 is the uniform policy, i.e., π1(· | x) ← unif(Y), ∀x ∈ X, and Zt(x) := y∈Y πt(y | x)exp r

η is the partition function for iteration t. Using the no-regret update of soft policy iteration, as shown in Eq. (4), Swamy et al. (2024) proved that the uniform mixture of π1:T from SPO is an approximation of the Nash equilibrium of MW(P), as defined in Eq. (3).

t(x,y)

Nash-MD. Munos et al. (2023) proposed Nash-MD to approximate the Nash equilibrium of a KL-regularized preference function,

π(y | x) πref(y | x)

′

τ (y ≻ y′ | x) := P(y ≻ y′ | x) − τ log

Pπ,π

π′(y | x) πref(y | x)

, (5)

+ τ log

′

τ (y ≻ y′ | x) (6)

Pτ(π ≻ π′) := Ex∼ρ,y∼π(·|x),y′∼π′(·|x) Pπ,π

= P(π ≻ π′) − τEx∼ρ [DKL(π(· | x) ∥ πref(· | x))] + τEx∼ρ [DKL(π′(· | x) ∥ πref(· | x))].

Following this, Munos et al. (2023) demonstrate that the Nash Equilibrium of MW(Pτ) can be approximated using a mirror descent (Nemirovskĳ and Yudin, 1983; Bubeck, 2015; Lattimore and Szepesvári, 2020) inspired algorithm, Nash-MD, which has a last-iteration guarantee. The Nash-MD algorithm can be viewed as a two-step iterative process: for each t = 1,2,...,T,

- (i) rt(x,y) ← Ey′∼πtτ(·|x) [P(y ≻ y′ | x)], ∀(x,y) ∈ X × Y
- (ii) πt+1(· | x) ←

rt(x,·) η

1 Zt(x)

πtτ(· | x)exp

, ∀x ∈ X,

where η is the learning rate, πtτ is the geometric mixture between πt and πref,

πt(y | x)1−τ/

πref(y | x)τ/

η

η

, ∀(x,y) ∈ X × Y, (7)

πtτ(y | x) :=

y′∈Y πt(y | x)1−τ/η

πref(y | x)τ/

η

and Zt(x) := y∈Y πtτ(y | x)exp r

η is the partition function for iteration t.

t(x,y)

### 3 Direct Nash Optimization

While the no-regret update of soft policy iteration used in SPO and Nash-MD has inspired many standard (deep) reinforcement learning algorithms (e.g., NPG, Kakade, 2001; TRPO, Schulman et al., 2015; PPO, Schulman et al., 2017; SAC, Haarnoja et al., 2018), its faithful implementation still usually involves the two-timescale update. This could potentially lead to complex hyperparameter tuning and unstable performance. In this section, we propose a direct and iterative algorithm, Direct Nash Optimization (Algorithm 1), to approximate the Nash equilibrium of MW(P). This algorithm is primarily inspired by SPO. It can be readily adapted to Nash-MD for approximating the Nash equilibrium of MW(Pτ) with the last-iteration guarantee, and we will discuss this in Appendix A.

- Algorithm 1 Direct Nash Optimization (DNO) input: General preference function P, learning rate η, number of iterations T, prompt distribution ρ.

- 1: Initialize π1 ← unif(A).
- 2: for iteration t = 1,2,...,T do
- 3: Compute rt(x,y) ← Ey′∼πt(·|x) [P(y ≻ y′ | x)], ∀(x,y) ∈ X × Y.
- 4: Obtain πt+1 by,

πt+1 ← argmax

π∈Π

E(x,y

1,y2)∼Dt σ (rt(x,y1) − rt(x,y2))log σ η log

π(y1 | x) πt(y1 | x) − η log

π(y2 | x) πt(y2 | x)

+σ (rt(x,y2) − rt(x,y1))log σ η log

π(y2 | x) πt(y2 | x) − η log

π(y1 | x) πt(y1 | x)

,

(8)

where Dt is generated by x ∼ ρ,y1 ∼ µ1,t(· | x),y2 ∼ µ2,t(· | x); µ1,t and µ2,t can be either off-policy (e.g., pre-defined) or on-policy (based on πt).

- 5: end for
- 6: return π¯ = unif(π1:T).

#### 3.1 Derivation of Algorithm 1

In most practical algorithms which are inspired by soft policy iteration, including the original practical version of SPO, they typically adopt the following approach: “pushing” π towards this subsequent learning goal in each iteration (we will refer to this as the soft policy iteration target throughout the paper):

1 Zt(x)

πt⋆+1(· | x) :=

πt(· | x)exp

rt(x,·) η

, (9)

η is the partition function. It can be realized by minimizing a distance metric between πt+1 and π. For example, the PPO algorithm for RLHF (e.g., Christiano et al., 2017; Ouyang et al.,

where Zt(x) = y∈Y πt(y | x)exp r

t(x,y)

- 2022) essentially minimizes the reverse KL divergence as follows,

(πtPPO+1 ←) argmin

Ex∼ρ DKL(π(· | x) ∥ πt⋆+1(· | x))

π∈Π

πt⋆+1(x,y) πt(x,y) − η log

π(x,y) πt(x,y)

Ex∼ρ,y∼π(·|x) η log

= argmax

π∈Π

π(x,y) πt(x,y)

Ex∼ρ,y∼π(·|x) rt(x,y) − ηZt(x) − η log

= argmax

π∈Π

π(x,y) πt(x,y)

. (⇔ PPO objective, as Zt is independent of π)

Ex∼ρ,y∼π(·|x) rt(x,y) − η log

= argmax

π∈Π

However, implementing the above approach typically necessitates on-policy sampling from the current policy π. Ignoring the Zt(x) term could also lead to high variance in the empirical gradient estimation. This is a persistent issue in actor-critic style algorithms that usually suggests the need for an additional baseline (details see, e.g., Mnih et al., 2016), which also requires on-policy estimation. When rt also varies over iterations, as in SPO or Nash-MD, we then need to update all of the policy, baseline, and reward online simultaneously. These challenges have hindered the scalability of existing algorithms which are based on learning the Nash equilibrium of general preference functions.

Regressing “internal rewards” towards preference-based rewards. Different from the mentioned approaches above which are mostly focusing on the concept of “pushing” π → πt⋆+1. We now consider the following mechanism: regressing rπ,t → rt, where rπ,t is the internal reward function of a given π at iteration t:

π(y | x) πt(y | x)

+ ηZt(x). (10)

rπ,t(x,y) := η log

This can be interpreted as a reparameterization trick, where π is exactly the soft policy iteration target (refer to Eq. (9)) induced by πt and the defined rπ,t. Therefore, regressing that specifically parameterized rπ,t to rt allows us to directly optimize the soft policy iteration target with respect to rπ,t and πt. This idea is inspired by techniques from inverse RL (e.g., Finn et al., 2016b,a, Guided Cost Learning) as well as recent advances in RLHF (Rafailov et al., 2023, DPO). To avoid the issues arising from the partition function Zt(x), we consider learning from the (x,y1,y2) tuple, where y1 and y2 are both responses to textual input x. Note that, due to the offline learning nature of the regressive objective, the sampling distribution of y1 and y2 does not impact the learning objective (i.e., rπ,t → rt, but it may affect the sample complexity from the coverage reason as we will discuss later), whereas pushing π → πt⋆+1 requires sampling y on-policy, as previously discussed. Therefore, given an arbitrary (x,y1,y2) tuple, we regress the “prediction” zˆ to the “goal” z (both defined below), using binary logarithmic/cross-entropy loss to measure the prediction error (see, e.g., Foster and Krishnamurthy, 2021),

π(y2 | x) πt(y2 | x)

π(y1 | x) πt(y1 | x) − η log

z := σ (rπ,t(x, y1) − rπ,t(x, y2)) = σ η log

=:∆π,t(x,y1,y2)

, z := σ rt(x, y1) − rt(x, y2)

=:∆⋆t (x,y1,y2)

ℓπ,t(x, y1, y2) := z log(1/ z) + (1 − z) log(1/(1 − z)) = −σ ∆⋆t(x, y1, y2) log σ ∆π,t(x, y1, y2) − σ ∆⋆t(x, y2, y1) log σ ∆π,t(x, y2, y1) .

Therefore, we obtain the following objective to learn πt+1,

; (11)

LDt

(π;πt) := argmin

argmin

π∈Π

E(x,y

1,y2)∼Dt [ℓπ,t(x,y1,y2)]

π∈Π

1,y2)∼Dt σ ∆⋆t(x,y1,y2) log σ ∆π,t(x,y1,y2) + σ ∆⋆t(x,y2,y1) log σ ∆π,t(x,y2,y1)

E(x,y

= argmax

π∈Π

π(y2 | x) πt(y2 | x)

π(y1 | x) πt(y1 | x) − η log

(12)

E(x,y

1,y2)∼Dt σ (rt(x,y1) − rt(x,y2))log σ η log

= argmax

π∈Π

π(y2 | x) πt(y2 | x) − η log

π(y1 | x) πt(y1 | x)

+ σ (rt(x,y2) − rt(x,y1))log σ η log

.

Here, Dt is generated by x ∼ ρ,y1 ∼ µ1,t(· | x),y2 ∼ µ2,t(· | x) with some policies µ1,t and µ2,t. It should be noted that µ1,t and µ2,t for each t ∈ [T] are parts of our algorithm’s design decisions. We will provide choices for them in Section 3.2 to promote sample efficiency, which are informed by our finite-sample analysis.

Monotonic improvement from the batched on-policy updates. One key distinction between DNO and existing algorithms for learning Nash equilibrium (such as SPO and Nash-MD) is that those algorithms aim to approach the Nash equilibrium in a purely on-policy manner, which can be potentially unstable and may need to incorporate two-timescale updates (that change the reward function used in the inner problem more frequently). On the other hand, DNO is a batched on-policy algorithm with single-timescale updates.

From a purely theoretical perspective, it seems that DNO may require many iterations to ensure the convergence of π¯ to the Nash equilibrium, which could potentially be costly. Additionally, DNO only converges on-average, and it is unrealistic to deploy in practice that uniform mixture policy π¯ (note that, as inspired by Munos et al. (2023), DNO could be extended to regularized preferences with last-iteration convergence, which is discussed in Appendix A). However, from a practical perspective, we can leverage the following two desirable properties from LLMs scenario to eliminate these concerns and ensure monotonic improvement over the DNO iterations:

Firstly, the soft policy iteration target Eq. (9) is actually the analytical solution for maximizing the following loss, ℓt(π) := P(π ≻ πt) − ηEx∼ρ [DKL(π(· | x) ∥ πt(· | x))], and πt⋆+1 = argmaxπ ℓt(π). We can notice that ℓt(πt) = 0.5 and Ex∼ρ [DKL(π(· | x) ∥ πt(· | x))] ≥ 0. This means 0.5 ≤ ℓt(πt⋆+1) = P(πt⋆+1 ≻ πt) − ηEx∼ρ DKL(πt⋆+1(· | x) ∥ πt(· | x)) =⇒ P(πt⋆+1 ≻ πt) ≥ 0.5 + ηEx∼ρ DKL(πt⋆+1(· | x) ∥ πt(· | x)) . This means πt⋆+1 is guaranteed to be more preferred than πt with respect to the preference P, and there is even a computable lower bound of the amount of improvement—ηEx∼ρ DKL(πt⋆+1(· | x) ∥ πt(· | x)) . Therefore, if πt+1 learned from Line 4 of Algorithm 1 is a accurate enough approximation of πt⋆+1 (which is proved in Section 3.2), we could expect that the policy is monotonically improved over DNO iterations. Note that the monotonic improvement guarantee is exclusive to our design choice of batched on-policy updates in DNO, because the alternatives are either unclear or unstable: it is undefined how to perform iterative updates offline, and one gradient update from a purely online algorithm may not be able able to accurately approximate the the soft policy iteration target πt⋆+1. Secondly, in practice, we usually have validation data available, which allows us to deploy the best policy over π1:(T+1).

#### 3.2 Theoretical Analysis

One of our major proposals is to use a regression-based objective to approximate the explicit soft policy iteration; in this section we show the approximation error from this regression is tightly bounded with finite-sample analysis. The following proposition discusses how well the solution of the regression-based objective (defined in Eq. (12) or Line 4 of

- Algorithm 1) can approximate the soft policy iteration (Eq. (9)) in terms of the total variation metric at each iteration.

Theorem 1 (informal). Fix an arbitrary iteration t ∈ [T]. Suppose πt+1 is from Line 4 of Algorithm 1, and πt⋆+1 is defined in Eq. (9). Then, under mild assumptions (realizability and boundedness, formally introduced in Appendix B), we have

CtRmax2 log(|Π|/δ) N

Ex∼ρ DTV(πt+1(· | x),πt⋆+1(· | x)) 2 ≤ O

,

where the concentrability coefficient Ct is defined as below,

2

⋆ t+1(y2|x)

⋆ t+1(y1|x)

πt+1(y1|x) − log π

1∼πt⋆+1(·|x),y2∼πt+1(·|x) log π

Ex∼ρ,y

πt+1(y2|x)

Ct :=

2 .

⋆ t+1(y1|x)

⋆ t+1(y2|x)

1∼µ1,t(·|x),y2∼µ2,t(·|x) log π

πt+1(y1|x) − log π

Ex∼ρ,y

πt+1(y2|x)

If πt = πt⋆ for all t ∈ [T], the reader can refer to (Swamy et al., 2024, Section 3) for the convergence of π¯ (returned by Algorithm 1) to the Nash equilibrium. We expect the total variation difference between πt and πt⋆ provided by Theorem 1 will be additive errors on top of the guarantees from Swamy et al. (2024).

Note that, we present the concentrability coefficient Ct as data-dependent, with πt+1 (learned from data) as part of its definition. We aim to make this guiding the design choices of µ1,t and µ2,t from such Ct for the purpose of sample efficiency. The formal statement and detailed proof of Theorem 1, without involving πt+1, are deferred to Appendix B. Although it shares a similar expression to the concentrability coefficient in offline reinforcement learning (e.g., Chen and Jiang, 2019; Xie et al., 2021), the policies µ1,t and µ2,t are flexible here due to the generative nature of large language models. This flexibility allows for additional intervention, enhancing sample efficiency.

⋆ t+1(y|x)πt+1(y|x)

We can notice that the value of Ct can be always bounded by Ct ≤ max(x,y)∈X×Y π

µ1,t(y|x)µ2,t(y|x) in the worst case. However, as πt+1 is likely to be restricted within a certain region, for instance, because fine-tuning will not significantly alter the behavior of the language model, we anticipate that such a coefficient will not depend on the per-(x,y) worst case. On the other hand, as a direct observation, we notice that the ideal selection of µ1,t and µ2,t should be close to the target of soft policy iteration πt⋆+1 (assuming πt⋆+1 and πt+1 are close). Interestingly, this theoretical observation coincides with recent empirical results. Here, Liu et al. (2024b) suggests that using statistical rejection sampling to sample from the soft policy iteration target (which is almost equivalent to sampling y1 and y2 from πt⋆+1) could benefit preference tuning. However, in our case, if we use similar statistical rejection sampling techniques on πt to sample πt⋆+1 (and πt+1), the cost of rejection sampling is likely to be comparable to the concentrability coefficient Ct when choosing µ1,t and µ2,t to be πt (see, e.g., Owen, 2013). This suggests that both πt and πt⋆+1 (via rejection sampling)

- as the choices of µ1,t and µ2,t will be comparable options in terms of sample efficiency. On the other hand, as we will demonstrate in the next section, since rt is defined based on πt (as shown in Line 3 of Algorithm 1), choosing µ1,t and µ2,t to be πt can easily adapt to such a reward of rt.

Another interesting observation is that despite Eq. (12) sharing a similar form with Bradley-Terry style reward modeling with using MLE, the target distributions used to measure distribution shift appear to be quite different. This disparity is due to the different objectives: fitting soft policy iteration versus reward estimation. For the Bradley-Terry style reward modeling using MLE, the desired distribution of y1 and y2 should be two distinct distributions (see, e.g., Zhan

- et al., 2024; Xiong et al., 2023). However, in our case where the learning goal is to fit the soft policy iteration, we may

prefer y1 and y2 from two (near) on-policy distributions as discussed above, as long as we expect the learned πt+1 will be accurate enough. To the best of our knowledge, this is the first theoretical result that illustrates the importance of on-policy sampling beyond policy optimization style algorithms for RLHF.

### 4 Practical Algorithm – Iterative Contrastive Self-Improvement

In this section, we shift our focus to the algorithmic design of the practically scalable version of DNO, following the principles discussed in the last section. A primary challenge encountered in the implementation of the conceptual algorithm DNO (Algorithm 1) stems from the necessity to compute the expectation with respect to the preference function P under the current policy πt. Perhaps surprisingly, as we will show, all we need is a properly implemented iterative DPO-like contrastive learning algorithm.

We present our the practical implementation of DNO in Algorithm 2 (DNO-Prct), which is a batched on-policy algorithm that conducts self-improvement iteratively via contrastive learning. One key consideration in our algorithmic design is that we only need to implicitly use the reward function rt. This comes from the specifically designed on-policy sampling, data filtering, and pair construction. While these specific design choices make DNO-Prct seem similar to simply performing DPO iteratively, there are significant reasons for these design decisions, as we will discuss below.

Batched on-policy sampling. The use of batched on-policy sampling in Line 4 of Algorithm 2 is crucial to avoid explicit use of rt (defined as Ey′∼πt(·|x) [P(y ≻ y′ | x)] in Line 3 of Algorithm 1). This means we essentially choose µ1 and µ2 in DNO to be πt in DNO-Prct, but we are free to let them vary slightly as a mixture of other policies, e.g., from a stronger teacher. Specifically, it is unrealistic to assume in practice that we can access the exact value of P(y ≻ y′ | x) given an (x,y,y′) tuple. Based on the definition of rt and the fact of {yt1,yt2,...,ytK} are sampled from πt, DNO-Prct essentially uses the following sampled based approach to estimate rt: rt(x,y) ≈

K y′∈{yt1,yt2,...,ytK,ygold}\y 1P(Is y better than y′ on x?), foranyxandy ∈ {yt1,yt2,...,ytK,ygold}, where1P denotes one sample from P and output {0,1}. This is implemented in Line 5 of Algorithm 2, and its precise implementation on

1

this is discussed in the Section 5. On the other hand, as we discussed in the last section, the batched on-policy sampling from πt is an appropriate option due to the consideration of sample efficiency when we use Eq. (13) to approximate

- Algorithm 2 DNO-Prct: Practical Implementation of DNO via Iterative Contrastive Self-Improvement input: General preference function P, learning rate η, iterations T, reference policy πref, prompt distribution ρ.

- 1: Initialize π1 ← πref.
- 2: for iteration t = 1,2,...,T do
- 3: Construct Dt = {(x,ygold)} where x ∼ ρ and y ∼ πgold(· | x).
- 4: Sample batched on-policy responses: Sample K outputs per per prompt using the current πt: {yt1,yt2,...,ytK} ∼ πt(· | x), ∀x ∈ Dt.
- 5: Rank responses: For each x ∈ Dt, rank the corresponding {yt1,yt2,...,ytK,ygold} using the pair-wise win-rate by sampling from the general preference function P.
- 6: Filter preference pairs: Construct Dt+1 = {(x,yt+,yt−)}, for all x ∈ Dt+1, and (yt+,yt−) are large-margin pairs (based on the win-rate rank) within the responses for x from the previous step.
- 7: Contrastive learning: Obtain πt+1 by,

πt+1 ← argmax

π∈Π

E(x,y+

t ,yt−)∼Dt+1 log σ η log

π(yt+ | x) πt(yt+ | x) − η log

π(yt− | x) πt(yt− | x)

. (13)

- 8: end for
- 9: return best of π1:(T+1) on the validation data.

the soft policy iteration (see Theorem 1 and its discussion).

Preference pair construction. Another key design choice in Algorithm 2 is that Eq. (13) of Algorithm 2 only uses the purely contrastive loss, whereas Eq. (8) of Algorithm 1 also contains the regression target σ (rt(x,y) − rt(x,y′)) (for a given (x,y,y′) tuple), which is not necessarily {0,1}. As we discussed above, it is unrealistic to expect access to the exact value of P(y ≻ y′ | x), so it is also unlikely to get an accurate value of the regression target σ(rt(x,y)−rt(x,y′)). Thus, we add an additional data filtering step to address this issue as in Line 6 of Algorithm 2. Ideally, we want the selected (x,y+,y−) tuple to satisfy σ(rt(x,yt+) − rt(x,yt−)) ≈ 1, so that Eq. (8) can be approximated by Eq. (13). However, one can notice that it requires rt(x,yt+) − rt(x,yt−) → ∞, but we know rt(x,y) ∈ [0,1], ∀(x,y) ∈ X × Y. From the derivation of DNO in Section 3, it is clear that scaling up rt and η with the same absolute constant c does not affect the soft policy iteration target of Eq. (9), but it will slightly change the DNO objective (Eq. (8) in Algorithm 1) by rt → c · rt and η → c · η =: η. This scaling strategy helps us sidestep the problem of bounded rt, and in this sense, we may expect the proper η in DNO-Prct to be relatively larger (than, e.g., η in Algorithm 1). However, an enlarged η in Eq. (8) will worsen the sample complexity suggested in Theorem 1 (for details, refer to its proof in Appendix B, especially for the derivation of Eq. (18)). So, to avoid the proper η being too large, we only use pairs with large margin as in Line 6 of Algorithm 2 to make sure rt(x,yt+)−rt(x,yt−) is not too small. This decision is also supported empirically in techniques like RLCD (Yang et al., 2023) and Axiomatic Preference Models (Rosset et al., 2023) which highlight the importance of having large margin or clear directional differences between positive and negative LLM responses when training preference models.

Relationship between DNO-Prct and DPO. The reader may discern that DNO-Prct (Algorithm 2)—the practical implementation of DNO—can be described as an iterative version of the DPO algorithm. Such similarity is by design, intended to harness the simplicity and effectiveness of DPO (Rafailov et al., 2023) and build on empirical advancements from recent work that applies DPO iteratively (e.g., Yuan et al., 2024; Tran et al., 2024). Our experiments point to the importance of several design choices which help accommodate the general preferences, such as rankings derived from pair-wise win rates. More interestingly, our findings point to a surprising connection—that “a meticulously designed iterative DPO algorithm” could approach the Nash equilibrium of any given general preferences.

Our general algorithmic framework—DNO (Algorithm 1)—is broader and fundamentally different from iterative DPO. For example, the DNO framework could also be directly extended to the regularized preference case (as discussed in Appendix A) or equipped with other advanced sample techniques (e.g., Liu et al., 2024b, RSO) as suggested by Theorem 1 for sample efficiency. On the other hand, although the soft policy iteration (or the KL-regularized reward optimization) is used in both DNO and DPO, they arise from fundamentally different reasons. For DNO, KL-regularization

Alpaca Eval 2 Win-Rate vs. GPT-4 Turbo

35

30

Win-RateagainstGPT-4

DNO 7B - More Data

DNO 7B

25

DNO 7B - Restrictive

SPIN 7B

Iterative KTO 7B Offline DPO (Ep) SFT on Positives (Ep)

20

15

Self-Rewarding 70B

Initializer Orca-2.5

Mistral Large

10

GPT-4 0314

5

1 2 3 4 5 6

Iterations or Epochs

Figure 2: Comparison of various post-training techniques showing that Direct Nash Optimization (DNO) is the most effective. All methods with colorful error bands are 1) implemented by ourselves, 2) initialized with a 7B parameter Orca-2.5 LLM, and 3) are “batched on-policy” (except SFT and Offline DPO which are epochs), all else being equal.

originates from online learning, no-regret learning through mirror descent (Nemirovskĳ and Yudin, 1983) or followthe-regularized-leader (FTRL) (Kalai and Vempala, 2005; Cesa-Bianchi and Lugosi, 2006; Shalev-Shwartz et al., 2012; Hazan et al., 2016). For DPO and PPO, the KL-regularization is an approximation for the total variation penalty to ensure monotonic improvement of the policy (Kakade and Langford, 2002; Schulman et al., 2015). Later, this approach was simplified by Schulman et al. (2017, PPO), and recently used for post-training LLMs (Ouyang et al., 2022).

### 5 Experiments

- Algorithm 2 is chosen for its efficiency and simplicity from an implementation standpoint (in this section, we will use

DNO to denote Algorithm 2 or DNO-Prct for simplicity). Once the input dataset {xi ∈ X} is chosen, each iteration of DNO proceeds in three phrases: sampling outputs from the current policy, annotating outputs for preference pair generation, and then training the next policy with the new training pairs. Iteration 0 is defined to start by sampling from the initial SFT model to produce training data for iteration 1.

#### 5.1 Experimental Setup

Data: We mainly use Ultrafeedback (Cui et al., 2023), which consists of 60k prompts, several models’ outputs to those prompts, and preference annotations from GPT-4-Turbo. This dataset thus provides a source of offline preferences. For our iterative experiments, we split this dataset into three non-overlapping partitions of the inputs to be used for separate iterations of batched on-policy learning. For each input, we also collect the GPT-4-Turbo output if it was not already present in the original dataset to be reserved for ygold.

Every experiment except one in this study solely uses UltraFeedback. The exception is one “scaled up” experiment with about 10x more data sourced from a mixture of datasets aggregated including Anthropic HH (Bai et al., 2022a), UltraChat (Ding et al., 2023), MetaMathQA (Yu et al., 2023), EvolInstruct (Xu et al., 2023a), UltraFeedback (Cui et al., 2023) and Orca-2 (Mitra et al., 2023). Note that we only use the input prompts for these datasets and collect a

|Technique<br><br>Epoch or Iter|Alpaca Eval 2<br><br>Len-control. Win Rate<br><br>Win Rate vs. GPT-4<br><br>Avg. len (chars)<br><br>|MT Bench 1st Turn<br><br>2nd Turn<br><br>Avg|
|---|---|---|
|Orca-2.5 SFT Epoch 1 Orca-2.5 SFT on Positives Epoch 4 Offline DPO (ours) Epoch 4<br><br>|10.76 6.99 1174<br><br>11.62 7.96 1420 19.49 18.22 1884<br><br><br>|7.72 6.02 6.88 7.62 6.23 6.92 7.69 7.08 7.38|
|Self-Rewarding 70B Iter 3 SPIN (ours) Iter 3 DNO-Restrictive Iter 3 DNO-Lookahead Epoch 1|- 20.44 2552 16.18 16.13 1922 21.61 19.21 1795 18.58 18.28 1907<br><br>|- - 7.25<br><br>7.58 7.53 7.55<br><br>7.59 7.35 7.46<br><br><br>8.09 7.32 7.70<br><br>|
|DNO Iter 3<br><br>|22.59 24.97 2228|7.62 7.35 7.48|

Table 1: AlpacaEval 2.0 and MT-Bench results in our controlled setting after training on UltraFeedback.

GPT-4-Turbo responses for all 600k of these input prompts.

Sampling from the Policy: At the end of training, we sample 5 outputs from the resulting student policy using top p sampling with p = 0.95 and temperature 0.7. Several works have shown the benefit of sampling and comparing multiple diverse outputs from the policy (Yuan et al., 2023a; Mitra et al., 2024; Liu et al., 2024b; Dong et al., 2023; Wang et al., 2022). We implement a simple defect detection system which flags any sample that has a high amount of repeated n-grams as automatic negative.

Preference Annotation: We use GPT-4-Turbo “as a judge” to label preferences among the 5 policy samples and 1 gold sample (which is also GPT-4-Turbo) as shown in Figure 3. This prompt contains a few minor modifications from the that used in (Yuan et al., 2024). It implements an additive scoring framework on a 6-point scale where a score of 6 represents the highest quality answer according to certain dimensions like “correctness”, “expert knowledge”, “conciseness” etc. By following this rubric, GPT-4 acting as an annotator represents a best-effort general preference model because it compares multiple candidate responses side-by-side in the context window, and stratifies them along meaningful dimensions of quality.

Training Pair Construction: Adhering to Line 6 in Algorithm 2 implies that not all pairs are suitable for training. Firstly, we must enforce the positives to be high quality in an absolute sense, and secondly, the negatives are directionally worse by a large margin. On the 6 point annotation scale, only samples that score a 5 or 6 are allowed to be positives. From the positives that meet this criteria, if any, we then construct all pairs such that the negative is at least 2 points lower. If the positive happens to be from the student, we relax this constraint to 1 point margin since the GPT-4-Turbo teacher outputs rarely receive a score less than 5 (as shown by the average teacher score in Table 2).

Additionally, we are motivated to preserve the preference behavior from previous iterations so that new policies do not inadvertently regress to past bad behavior. To enforce this, we incorporate an exponentially decaying proportion of prior iterations’ training pairs into the current iteration, i.e. we sample at most 30% of training pairs from iteration t − 1, 15% from t − 2, and so on. We do not re-inference outputs for those inputs from the most recent policy. Recall that previous iterations’ inputs are non-overlapping with the splits for other iterations.

Training: To prevent overfitting, we train our batched on-policy methods for at most one epoch on newly constructed pairs. Our effective batch size is fixed to 64 for all experiments. Our learning rate, beta, and alpha are found with brief hyperparameter searches. For most experiments, the learning rate is 5E-5, beta is either 0.1 or 0.05, and alpha is 0.005. We found that at higher iterations, the learning rate needs to be lowered. In SFT (supervised fine-tuning) experiments, our learning rate is 5E-6 and we mask out loss for the inputs. We use the open-source TRL library’s implementation to run our experiments.

Evaluation: Our primary goal is to train a policy that is comparable to the most powerful state-of-the-art langauge models. Hence, AlpacaEval 2.0 (Dubois et al., 2023) is an appropriate benchmark because it computes win-rate against GPT-4-Turbo in a head-to-head fashion on a dataset of 805 input prompts that is shown to correlate with human preferences (0.93 spearman correlation with Chatbot Arena). While it is known that auto-eval methods also correlate with spurious features such as length, a new version of AlpacaEval 2.0 corrects for this with a length-controlled win-rate

| |inputs<br><br>student length (words)|Annotations of Training Data best-of-n student win-rate<br><br>Avg. # student wins<br><br>Avg. student score<br><br>Avg. teacher score<br><br>|New Training Pairs T ≻ S S ≻ T S ≻ S|
|---|---|---|---|
|DNO-Restrictive Iter 0<br><br>DNO-Restrictive Iter 1<br><br>DNO-Restrictive Iter 2<br><br><br>|19.6k 162 +/- 190 19.9k 359 +/- 350 19.8k 256 +/- 207|15.9% 0.486 3.46 4.99<br><br>34.2% 1.11 4.86 4.77<br><br>35.0% 1.31 5.21 4.87<br><br><br>|42.4k 0 0 17.5k 0 0<br><br>9.9k 0 0|
|DNO Iter 0<br><br>DNO Iter 1<br><br>DNO Iter 2<br><br><br>|19.6k 162 +/- 190 19.9k 671 +/- 546 19.8k 361 +/- 251|15.9% 0.486 3.46 4.99 34.6% 1.22 4.61 4.62 43.6% 1.90 5.25 4.59<br><br>|30.7k 4.2k 25.9k 20.3k 19.4k 62.9k<br><br>7.1k 32.4k 10.9k|

Table 2: The dynamics of how sampled outputs from a previous iteration’s policy compare to their teacher, and how many new training pairs they give rise to in the next iteration. The crucial point is that DNO constructs new pairs where the student is compared to the teacher, whereas DNO-Restrictive, SPIN, and IPO-MD do not.

that has an even higher spearman correlation (0.98) with Chatbot Arena 4.

We also evaluate on MT-Bench (Zheng et al., 2023) which allows the llm-as-a-judge to first explain its reasoning before providing a scalar score on 1-10 for the candidate response to a bank of 80 questions. One crucial difference between AlpacaEval 2.0 and MT Bench is that the former asks GPT-4-Turbo to predict which of two side-by-side responses humans would prefer, weighted by the logits to represent its uncertainty, whereas MT-Bench asks the model to first generate a justification and then output a score on 1-10, but it neither defines the ratings (e.g. how a 7 is different than a 5) nor accounts for uncertainty in the logits of the score.

WealsoevaluateontheOpenLLMleaderboard(Beechingetal.,2023), whichmeasuresreasoningabilityondownstream NLP tasks like coding and question answering by evaluating the accuracy of the multiple choice answer option with the highest logit. Since our training data is primarily instruction-following and not trained to output just the sole answer option, this benchmark is not the primary target of this study; nonetheless, DNO on instruction tuning tasks ought to show no regression on reasoning tasks.

#### 5.2 Results and Analysis

We run several head-to-head experiments that control for hyperparameters and input data. We often refer to the policy being trained as the “student” and GPT-4 as a “teacher”; GPT-4 is also used as an annotator when prompted.

SFT Baselines The first baseline is Orca-2.5 itself, which is a mistralai/Mistral-7B-v0.1 raw pretrained model fine-tuned on a new collection of Orca-2 data (Mitra et al., 2023). This model was finetuned for three epochs and achieves scores shown in the top of Table 4. All other experiments in this study are initialized with Epoch 1 of Orca-2.5. This is the solid horizontal line in Figure 2.

The second baseline is continue-SFT of Orca-2.5 training towards the positives in UltraFeedback (and masking out loss over the input prompts). If the original positive in that dataset was not from GPT-4-Turbo, we replace it with one that is. This is the red line in Figure 2. It is clear that even offline contrastive training methods are more beneficial than additional SFT, showing that the difference between the positive and negative output provides more valuable training signal than the positive in isolation.

Large Margin Filtering of Training Pairs: We ran a simple experiment of Offline DPO for one epoch on UltraFeedback data. In the control, we trained on all 63k preference pairs in the original dataset, whereas in the treatment we filtered the 42k pairs that met a large margin requirement enforcing that the positive’s scores exceeded that of the negative by

- at least 1.0 (out of 10) according to their GPT-4-Turbo annotator. All else was equal. Even though the treatment was trained for fewer steps on less data, it achieved an AlpacaEval 2.0 win rate of 11.60 vs 9.60 for the control, showing that fewer higher quality preference pairs is better than a higher quantity of noisy pairs (not shown in the tables).

On-Policy is Better than Off-Policy One of the critical questions in this study whether to sample “on-policy” outputs from the current student to use in training pairs, or whether “off-policy” outputs collected from other models different

4https://github.com/tatsu-lab/alpaca_eval

| |ARC-C (25-shot)<br><br>GSM8K (5-shot)<br><br>HellaSwag (10-shot)<br><br>MMLU (5-shot)<br><br>TruthfulQA (0-shot)<br><br>WinoGrande (5-shot)|Avg|
|---|---|---|
|Orca-2.5 Epoch 1 Orca-2.5 Epoch 3 SPIN (ours) Iter 3|0.609 0.635 0.818 0.614 0.489 0.738 0.624 0.641 0.826 0.624 0.506 0.746 0.668 0.448 0.862 0.623 0.601 0.759<br><br>|0.652 0.661 0.660|
|DNO Iter 1<br><br>DNO Iter 2<br><br>DNO Iter 3<br><br><br>|0.657 0.572 0.834 0.623 0.568 0.755 0.663 0.562 0.845 0.624 0.580 0.753 0.672 0.542 0.852 0.622 0.606 0.753|0.668 0.671 0.675<br><br>|

Table 3: Results on Open-LLM Leaderboard reasoning tasks, which we do not expect to decrease.

than the student will suffice. We ran 4 epochs of Offline DPO on UltraFeedback (filtered for large margin), and as shown in Table 1, on-policy methods especially DNO surpass the off-policy DPO, even when trained for 4 epochs while the on-policy models were granted only three iterations. Recall that each iteration of batched on-policy training sees only a third of the UltraFeedback input data, whereas an epoch of Offline DPO sees the entire dataset.

Higher Quality Annotators In our study, we use GPT-4-Turbo to provide the annotations for preference pairs. However, the Self-Rewarding Language Model uses the Llama-2-70B (Touvron et al., 2023) model trained to also give feedback as the annotator, which in their study starts off with a 65% agreement rate with human-labeled preferences improving to 80% in the last iteration (Yuan et al., 2024). While it was not reported how well GPT-4-Turbo’s annotations agree with their held-out human labels, we believe that having a higher-quality annotator to start with will lead to higher quality policies. Since both our studies use UltraFeedback data, and our annotation prompt is based on their annotation prompt, we believe there is a valid comparison.

We observe DNO initialized with a 7B base model outperforms the 70B parameter Self-Rewarding model over the same number of training iterations (24.97 win-rate vs 20.44 on AlpacaEval 2.0, and 7.46 MT-Bench vs 7.25), at least in part due to the higher quality preference annotations. See the dark blue band versus the gray line in Figure 2 and the corresponding row in Table 1. However, unlike Self-Rewarding LM, we saw a slight gain rather than a drop reasoning benchmarks like ARC-Challenge (Clark et al., 2018) and HellaSwag (Zellers et al., 2019). Granted, the evaluation of OpenLLM predicts the answer with the max logit corresponding to one of the multiple-choice options, which is not congruous with how these techniques are trained.

Training Pair Construction One of the most critical implementation questions in this study is how to construct training pairs that help the student policy exceed a strong teacher like GPT-4-Turbo. One approach, Self-Play Finetuning (SPIN), removes the preference annotation step and automatically assigns the teacher output to be the positive, and all student samples to be negative (Chen et al., 2024). We find in our re-implementation of SPIN that this is detrimental, presumably because this automatic assignment could lead to noisy training pairs in cases where the student might actually be preferred. The resulting win-rate of SPIN is only 16.13 after three epochs of iterative training compared to 24.97 for DNO as shown in Table 1, all else being equal. Similar results hold in the OpenLLM results in Table 3.

In a second experiment, which we denote DNO-Restrictive, we annotate all preference pairs with GPT-4-Turbo as usual, but only admit training pairs where the teacher’s output is the preferred one. The difference between DNO and DNO-Restrictive is illustrated in Table 2 where 0 student-vs-teacher and student-vs-student pairs are created. The same is also true for SPIN, but SPIN would admit a greater quantity of noisy teacher-vs-student examples even when they are dis-preferred: Table 2 shows that after Iteration 2 of DNO-Restrictive, only 9.9k instances exist of the teacher being preferred over the student, whereas SPIN would have automatically created about 100k (5 samples × 20k inputs).

While DNO-Restrictive is slightly better (19.21 win-rate) than SPIN, it still does not give the student a chance to compare its behavior to a powerful teacher. Absence of this signal is a major oversight, since the last row of Table 2 shows that by Iter 3, over 64% of the DNO training data (32k pairs) are cases where the student is in fact preferred over the teacher, a number which increases with iteration. We conclude it is imperative to “allow the student to become the teacher” i.e. learn from comparisons where its own outputs are preferred over a more powerful teacher.

One curious phenomenon in Table 2 is that while the teacher outputs are fixed ahead of time, the annotator gives slightly lower scores to the teacher as the student improves; we are not sure if this is an innocuous artifact of preference

|Technique<br><br>Epoch or Iter<br><br>|Alpaca Eval 2 Len-control. Win Rate<br><br>Win Rate vs. GPT-4<br><br>Avg. len (chars)<br><br>|MT Bench 1st Turn<br><br>2nd Turn<br><br>Avg|
|---|---|---|
|Orca-2.5 SFT Epoch 1<br>Orca-2.5 SFT Epoch 2<br>Orca-2.5 SFT Epoch 3<br>|10.76 6.99 1174 15.29 7.88 1060 15.90 8.17 1058<br><br>|7.72 6.02 6.88 7.56 6.38 6.98 7.53 6.73 7.13<br><br>|
|DNO-More-Data Iter 1<br><br>DNO-More-Data Iter 2<br><br>DNO-More-Data Iter 3<br><br>DNO-More-Data Iter 4<br><br>DNO-More-Data Iter 5<br><br>DNO-More-Data Iter 6<br><br><br>|8.96 10.67 2795 14.61 16.94 2782<br><br>21.81 26.74 2539<br><br>22.93 29.08 3033<br><br><br>32.06 34.98 2856<br><br>33.05 34.38 2683<br><br><br>|7.00 6.06 6.53 7.62 7.23 7.42 7.74 6.66 7.21 7.54 6.92 7.24 7.10 6.39 6.75 7.28 6.65 6.97|

Table 4: DNO-More-Data is trained on 10x more instruction data than DNO. It is still initialized with Epoch 1 of Orca-2.5 SFT, so the delta it provides in AlpacaEval 2.0 win rate is 27.39 absolute (22.29 length-controlled)

annotations, or symptomatic of a deeper problem. Also, the total quantity of new “large margin” training pairs (not counting those sampled from previous iterations) in DNO tends to decrease as the policy improves across iterations, but we do not have enough data to quantify how this relates to a change in quality.

Lookahead to Future Iterations As a curiosity, we experimented with whether a model could benefit from the knowledge of which training pairs it would generate if it could look into the future. We tested this by running three-iterations of DNO, accumulating all the preference pairs across iterations, combining and shuffling them, and then re-starting training from the initial model. In essence, this turns the batch-online DNO into an offline learning algorithm we denote as DNO-Lookahead. We trained for one epoch on the three iterations’ worth of preference data. It deteriorated more than we expected on AlpacaEval 2.0 win-rate (24.97 to 18.18), however, even more surprisingly, the MT-Bench numbers improved significantly (7.48 to 7.70). While the reasons for the relatively low correlation between MT-Bench and AlpacaEval 2.0 are not entirely clear, it is important to consider the disparity in the size of the datasets. Given that MT-Bench consists of merely 80 examples, whereas AlpacaEval 2.0 contains 10x more, we conjecture that the statistical significance and reliability of the findings from AlpacaEval 2.0 are regarded with greater confidence.

DNO Scales with More Data: One of the reasons we split UltraFeedback into three non-overlapping partitions is to avoid overfitting. Another strategy to avoid overfitting is to collect more data, so we increased by a factor of 10 the instruction data based on publicly available datasets. We split a large mixture of datasets into six nonoverlapping partitions of roughly 100k inputs each (and inference GPT-4-Turbo outputs for all inputs), and show that DNO-More-Data scales well in this expanded regime (see the purple line in Figure 2 and the last row of Table 4.

We make some notes on the behavior of this experiment: because each iteration builds on outputs of the previous iteration, if there are any anomalies or errors in critical components such as preference annotation, those errors will propagate and the only way to combat them is “roll back” to the iteration that introduced them. This can result in wasted time and cost, which are both already very high as shown in Appendix C. We suspect that the “depth” of iterations matters more than the “width” or number of samples within each iteration, and furthermore, that having equal number of inputs per iteration may not be optimal, but we did not test this thoroughly. From an efficiency standpoint, although this algorithm is “batched”, some optimizations can be made, such as starting to annotate sampled policy outputs are soon as they are ready instead of waiting for all inference jobs to finish.

“Exploding” Lengths It is known that contrastive LLM training techniques, especially DPO, lead to longer outputs from the model which is widely suspected to be a form of “reward hacking”. Curiously, Table 2 shows that the largest jump comes after the first round of contrastive training (Iteration 1), where lengths explode by at least a factor of 2 over the initializing SFT model, before inching down again in the next iteration. We interpret this “length spike” as wasted computation optimizing towards a spurious signal; we wish we were better equipped to control this phenomenon.

### 6 Related Work

We divide the space of related work into whehter or not the techniques use SFT or contrastive losses, in offline or online update settings.

Online RLHF algorithms: RLHF innovated how to align language models with human preferences (Christiano et al., 2017; Stiennon et al., 2020), but it is unstable to train and memory-intensive, requiring all three of the parameterized policy model, reward model, and advantage model to be on device for training.

Reward-model Augmented SFT: Since the introduction of RLHF, several emergent techniques apply reward models in various ways, such as to filter training data or rank responses. Reward rAnked Finetuning (RAFT) (Dong et al.,

- 2023) and RRHF (Yuan et al., 2023b) offer the conceptually simplest solution for offline preference learning, which is to sample multiple outputs from a policy, rank them with a reward model, and then finetune on the best sampled output using SFT. This resembles the iterative behavior-cloning technique DAgger (Ross et al., 2011).

Offline Contrastive Preference Learning: There exist several loss functions for contrastive preference learning, first introduced in the offline setting, namely Direct Preference Optimization (Rafailov et al., 2023, DPO) and Calibrated Sequence Likelihood Estimation a.k.a. SLiC (Zhao et al., 2023). Azar et al. (2023) make it clear that point-wise reward estimates are no substitute for pair-wise preferences, and that a policy can easily overfit to deterministic preferences without proper regularization. They derive a more general objective for RLHF, IPO, to directly optimize offline preference probabilities.

Statistical Rejection Sampling Optimization (RSO) generates multiple samples from an initial model, ranks them to create training pairs, and optimizes them under a unified framework encompassing DPO and SLiC (Liu et al.,

- 2024b). Inspired by the learning-to-rank literature, Listwise preference optimization (LIPO) extends pair-wise preference learning to list-wise (Liu et al., 2024a). Preference Ranking Optimization (PRO) also learns towards list-wise preferences (Song et al., 2024). The KTO algorithm takes a different approach from DPO and does not assume that a pair of good-vs-bad outputs for the same input exist, but rather a pool of good outputs and a pool of bad outputs for any inputs exist and optimizes an “unpaired” loss (Ethayarajh et al., 2024).

Iterative Reward-based Finetuning: Reinforced Self-Training (ReST) is one of the first methods to explore iterative self-improving training strategies framed as a two-stage “Grow” step that samples from the current policy, and a “Improve” step that uses a reward model to filter ever-higher quality samples that are then used to improve the policy with offline RL (Gulcehre et al., 2023). A follow-up work explores the use of AI feedback rather than reward ranking (Singh et al., 2023).

On-policy Contrastive Learning: Self-Rewarding Language Models (Yuan et al., 2024) is in practice very similar to DNO. They study the benefits of batched iteratively training on preferences derived from a recent policy’s sampled outputs, but in their work, they use the policy itself as the annotator, which starts off being able to provide only weak preference signals. Self-Play Fine-Tuning (Chen et al., 2024) a.k.a SPIN and Adversarial Preference Optimization a.k.a APO (Cheng et al., 2023) are both iterative LLM training techniques that are compatible with contrastive losses, but they make a very limiting assumption that the teacher is better than the student (without regard to any annotator feedback).

The Cringe Loss (Adolphs et al., 2022) is a token-level loss function that contrasts the correct next token with a hard-negative token from the vocabulary that has high logit weight but still incorrect. The Pairwise Cringe Loss (Xu et al., 2023b) applies the cringe loss to an iterative self-improving style of training.

On-Policy General Preference Optimization: Wang et al. (2023) consider finding the von Neumann winner of general preferences via multi-agent RL from the theoretical perspective. Nash-MD optimizes a policy towards the Nash equilibrium of a generalized preference model using policy gradients, showing that by sampling from a mixture of policies, one can converge to the Nash equilibrium in the last iteration (Munos et al., 2023). Self-play Preference Optimization (SPO) is another online two-player mini-max game that converges to a Nash equilibrium with no-regret guarantees (Swamy et al., 2024). However, these techniques are not as data efficient as contrastive losses and are difficult to implement faithfully without cumbersome two-timescale updates (Munos et al., 2023). A concurrent improvement, IPO-MD, mitigates these difficulties by using purely on-policy IPO updates and is empirically evaluated on an article summarization task (Calandriello et al., 2024). Guo et al. (2024) also propose to eliminate rewards in online AI-feedback (OAIF) by using another LLM to annotate which of two online-sampled outputs from the current

policy is preferred. However, all the above studies only consider training pairs constructed between self-play “student vs student” samples, and between student and initial πref. That is, there is no concept of a more powerful “teacher” to compare against in their training pairs. We showed in Table 2 that omitting these “student vs teacher” preferences may hinder performance.

### 7 Conclusion

In this paper we achieve dual goals of post-training LLMs against a more general class of preference models while providing a practical and scalable implementation with finite-sample analysis. Our strong empirical results are based on the insight that optimizing general preference functions can be reduced to finding the Nash equilibrium of a two-player game with the payoff as the preference, and further solved by a single-play algorithm. Most techniques to optimize for this objective use soft policy iteration, which is difficult to implement faithfully and may require unstable on-policy and two-timescale updates. Our contribution, Direct Nash Optimization, addresses these challenges by approximating soft policy iteration updates with a regression-based contrastive objective in a batched manner, which is a much more stable and forgiving learning objective, and we establish a concentration bound of O(1/N) on the squared total variation error between the learned policy and its target of the soft policy iteration update at any given iteration t. Theoretically, DNO converges to the Nash equilibrium on-average, but in practice enjoys monotonic improvement across iterations. Training a 7B parameter LLM with DNO achieves state-of-the-art performance on AlpacaEval 2.0, exceeding both Mistral Large and older versions of GPT-4. We illuminate many of the practical design choices that will aid future development of iterative self-improving algorithms.

### References

Leonard Adolphs, Tianyu Gao, Jing Xu, Kurt Shuster, Sainbayar Sukhbaatar, and Jason Weston. The cringe loss: Learning what language not to model. arXiv preprint arXiv:2211.05826, 2022.

Riad Akrour, Marc Schoenauer, and Michèle Sebag. April: Active preference learning-based reinforcement learning. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2012, Bristol, UK, September 24-28, 2012. Proceedings, Part II 23, pages 116–131. Springer, 2012.

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and

- Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036, 2023.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022a.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022b.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https://huggingface.co/spaces/HuggingFac eH4/open_llm_leaderboard, 2023.

Quentin Bertrand, Wojciech Marian Czarnecki, and Gauthier Gidel. On the limitations of the elo, real-world games are transitive, not additive. In International Conference on Artificial Intelligence and Statistics, pages 2905–2921. PMLR, 2023.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- Sébastien Bubeck. Convex optimization: Algorithms and complexity. Foundations and Trends® in Machine Learning, 8(3-4):231–357, 2015.

Daniele Calandriello, Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, et al. Human alignment of large language models through online preference optimisation. arXiv preprint arXiv:2403.08635, 2024.

Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge university press, 2006.

Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning, pages 1042–1051. PMLR, 2019.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Pengyu Cheng, Yifan Yang, Jian Li, Yong Dai, and Nan Du. Adversarial preference optimization. arXiv preprint arXiv:2311.08045, 2023.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377, 2023.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, SHUM KaShun, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S. Liang, and Tatsunori B. Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36, 2023.

Miroslav Dudík, Katja Hofmann, Robert E. Schapire, Aleksandrs Slivkins, and Masrour Zoghi. Contextual dueling bandits. In Conference on Learning Theory, pages 563–587. PMLR, 2015.

Arpad E. Elo. The rating of chessplayers, past and present. Arco Pub., New York, 1978. ISBN 0668047216

9780668047210. URL http://www.amazon.com/Rating-Chess-Players-Past-Present/dp/0668047216. Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024. Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv preprint arXiv:1611.03852, 2016a. Chelsea Finn, Sergey Levine, and Pieter Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. In International conference on machine learning, pages 49–58. PMLR, 2016b. Peter C. Fishburn. Probabilistic social choice based on simple voting comparisons. The Review of Economic Studies, 51(4):683–692, 1984. Dylan J. Foster and Akshay Krishnamurthy. Efficient first-order contextual bandits: Prediction, allocation, and triangular discrimination. Advances in Neural Information Processing Systems, 34:18907–18919, 2021. Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1):119–139, 1997.

Shane Griffith, Kaushik Subramanian, Jonathan Scholz, Charles L. Isbell, and Andrea L. Thomaz. Policy shaping: Integrating human feedback with reinforcement learning. Advances in neural information processing systems, 26, 2013.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, and Mathieu Blondel. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792, 2024.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.

Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4): 157–325, 2016.

Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the

Nineteenth International Conference on Machine Learning, pages 267–274, 2002. Sham M. Kakade. A natural policy gradient. Advances in neural information processing systems, 14, 2001. Adam Kalai and Santosh Vempala. Efficient algorithms for online decision problems. Journal of Computer and System

Sciences, 71(3):291–307, 2005. W Bradley Knox and Peter Stone. Tamer: Training an agent manually via evaluative reinforcement. In 2008 7th IEEE international conference on development and learning, pages 292–297. IEEE, 2008. Gerald H. Kramer. On a class of equilibrium conditions for majority rule. Econometrica: Journal of the Econometric Society, pages 285–297, 1973.

Germain Kreweras. Aggregation of preference orderings. In Mathematics and Social Sciences I: Proceedings of the seminars of Menthon-Saint-Bernard, France (1–27 July 1960) and of Gösing, Austria (3–27 July 1962), pages 73–79, 1965.

Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020. Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and

Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Tianqi Liu, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao, Mohammad Saleh, Simon Baumgartner, Jialu Liu, Peter J. Liu, and Xuanhui Wang. Lipo: Listwise preference optimization through learning-to-rank. arXiv preprint arXiv:2402.01878, 2024a.

TianqiLiu, YaoZhao, RishabhJoshi, MishaKhalman, MohammadSaleh, PeterJ.Liu, andJialuLiu. Statisticalrejection sampling improves preference optimization. In The Twelfth International Conference on Learning Representations, 2024b.

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching small language models how to reason. arXiv preprint arXiv:2311.11045, 2023.

Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math. arXiv preprint arXiv:2402.14830, 2024.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 1928–1937. PMLR, 2016.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, Marco Selvi, Sertan Girgin, Nikola Momchev, Olivier Bachem, Daniel J. Mankowitz, Doina Precup, and Bilal Piot. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.

Arkadĳ Semenovič Nemirovskĳ and David Borisovich Yudin. Problem complexity and method efficiency in optimization. Wiley-Interscience, 1983.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, GabrielGoh, Rapha Gontĳo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vĳayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Art B. Owen. Monte Carlo theory, methods and examples. https://artowen.su.domains/mc/, 2013. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct pref-

erence optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to

no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

Corby Rosset, Guoqing Zheng, Victor Dibia, Ahmed Awadallah, and Paul Bennett. Axiomatic preference modeling for longform question answering. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11445–11475, 2023.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends® in Machine Learning, 4(2):107–194, 2012.

Paul B. Simpson. On defining areas of voter choice: Professor tullock on stable voting. The Quarterly Journal of Economics, 83(3):478–490, 1969.

Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. Preference ranking optimization for human alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18990–18998, 2024.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Gokul Swamy, Christoph Dann, Rahul Kidambi, Zhiwei Steven Wu, and Alekh Agarwal. A minimaximalist approach to reinforcement learning from human feedback. arXiv preprint arXiv:2401.04056, 2024.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Hoang Tran, Chris Glaze, and Braden Hancock. snorkelai/snorkel-mistral-pairrm-dpo, 2024. https://huggingfac e.co/snorkelai/Snorkel-Mistral-PairRM-DPO.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of lm alignment. arXiv preprint arXiv:2310.16944, 2023.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

- 2022.

Yuanhao Wang, Qinghua Liu, and Chi Jin. Is rlhf more difficult than standard rl? arXiv preprint arXiv:2306.14111,

- 2023.

ChristianWirth, RiadAkrour, GerhardNeumann, andJohannesFürnkranz. Asurveyofpreference-basedreinforcement learning methods. Journal of Machine Learning Research, 18(136):1–46, 2017.

Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for offline reinforcement learning. Advances in neural information processing systems, 34:6683–6694, 2021.

Tengyang Xie, Dylan J. Foster, Yu Bai, Nan Jiang, and Sham M. Kakade. The role of coverage in online reinforcement learning. In The Eleventh International Conference on Learning Representations, 2023.

Wei Xiong, Hanze Dong, Chenlu Ye, Han Zhong, Nan Jiang, and Tong Zhang. Iterative preference learning from

human feedback: Bridging theory and practice for rlhf under kl-constraint. arXiv preprint arXiv:2312.11456, 2023. Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm:

Empowering large language models to follow complex instructions, 2023a. Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023b. Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, and Yuandong Tian. Rlcd: Reinforcement learning from contrast distillation for language model alignment. arXiv preprint arXiv:2307.12950, 2023. Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian

Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models, 2023. Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-

rewarding language models. arXiv preprint arXiv:2401.10020, 2024. Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023a. Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023b. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019. Wenhao Zhan, Masatoshi Uehara, Nathan Kallus, Jason D. Lee, and Wen Sun. Provable offline preference-based reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024.

Yao Zhao, Misha Khalman, Rishabh Joshi, Shashi Narayan, Mohammad Saleh, and Peter J. Liu. Calibrating sequence likelihood improves conditional language generation. In The Eleventh International Conference on Learning Representations, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2023.

## Appendix

### A Extension to Regularized Preferences

In this section, we discuss how to extend the DNO framework to the case of regularized preferences (defined in Eq. (5)),

π′(y | x) πref(y | x)

π(y | x) πref(y | x)

′

τ (y ≻ y′ | x) = P(y ≻ y′ | x) − τ log

Pπ,π

+ τ log

,

which was first introduced and solved by Munos et al. (2023) via Nash-MD introduced earlier.

One can notice that the only difference between SPO and Nash-MD is that SPO uses the last iteration policy πt for both constructing reward rt and performing a soft policy iteration update, whereas Nash-MD uses the smoothed version πtτ (firstly defined in Eq. (7)),

πt(y | x)1−τ/

πref(y | x)τ/

η

η

, ∀(x,y) ∈ X × Y, (14)

πtτ(y | x) :=

y′∈Y πt(y | x)1−τ/η

πref(y | x)τ/

η

for both. This allows Nash-MD to obtain a late-iteration guarantee. On the other hand, due to the symmetry of regularized preferences, if we consider on-average convergence case, it is likely that SPO can be adapted with a simpler way as follows: for each t = 1,2,...,T,

- (i) rt(x,y) ← Ey′∼πt(·|x) [P(y ≻ y′ | x)], ∀(x,y) ∈ X × Y
- (ii) πt+1(· | x) ←

rt(x,·) η

1 Zt(x)

πtτ(· | x)exp

, ∀x ∈ X,

where Zt(x) := y∈Y πtτ(y | x)exp r

η is the partition function for iteration t. Here, the smoothed policy πtτ is only used in the soft policy iteration step, and this coincides with the OMD algorithm from Munos et al. (2023).

t(x,y)

- Algorithm 3 DNO (Regularized Preferences Version) input: General preference function P, learning rate η, coefficient of KL-regularization τ, number of iterations T, prompt distribution ρ.

- 1: Initialize π1 ← unif(A).
- 2: for iteration t = 1,2,...,T do
- 3: Compute rt(x,y) by,

- Option I: ▷ for on-average convergence rt(x,y) ← Ey′∼πt(·|x) [P(y ≻ y′ | x)], ∀(x,y) ∈ X × Y.
- Option II: ▷ for last-iteration convergence

rt(x,y) ← Ey′∼πtτ(·|x) [P(y ≻ y′ | x)], ∀(x,y) ∈ X × Y, where πtτ is defined in Eq. (14).

- 4: Obtain πt+1 by,

πt+1 ← argmax

π∈Π

E(x,y

1,y2)∼Dt σ (rt(x,y1) − rt(x,y2))log σ η log

π(y1 | x) πtτ(y1 | x) − η log

π(y2 | x) πtτ(y2 | x)

+σ (rt(x,y2) − rt(x,y1))log σ η log

π(y2 | x) πtτ(y2 | x) − η log

π(y1 | x) πtτ(y1 | x)

,

where Dt is generated by x ∼ ρ,y1 ∼ µ1,t(· | x),y2 ∼ µ2,t(· | x) with some policies µ1,t and µ2,t, and πtτ(y | x) := πt(y | x)1−τ/

η

πref(y | x)τ/

η

,∀(x,y) ∈ X × Y (the unnormalized version of πtτ(y | x) defined in Eq. (14)).

- 5: end for
- 6: return π¯ = unif(π1:T).

Based on discuss above, we can then obtain the extension of DNO to the regularized preferences in Algorithm 3, and its practical implementation in Algorithm 4. Note that, similar to Nash-MD, the late-iteration option for both Algorithm 3

- Algorithm 4 DNO-Prct (Regularized Preferences Version) input: General preference function P, learning rate η, coefficient of KL-regularization τ, number of iterations T, reference policy πref, seed dataset D0 = {(x,ygold)} where x ∼ ρ and y ∼ πgold(· | x), reference model πref.

- 1: Initialize π1 ← πref.
- 2: for iteration t = 1,2,...,T do
- 3: Sample batched on-policy responses:

- Option I: ▷ for on-average convergence Sample K outputs per per prompt using the current πt: {yt1,yt2,...,ytK} ∼ πt(· | x), ∀x ∈ D0.
- Option II: ▷ for last-iteration convergence

Sample K outputs per per prompt using the smoothed current policy πtτ: {yt1,yt2,...,ytK} ∼ πtτ(· | x), ∀x ∈ D0, where πtτ is defined in Eq. (14) with accommodating η → η.

- 4: Rankresponses: Foreachx ∈ D0, rankthecorresponding{yt1,yt2,...,ytK,ygold}usingthepair-wisewin-rate by sampling from the general preference function P.
- 5: Filter and construct preference pairs: Construct Dt = {(x,yt+,yt−)}, for all x ∈ D0, and (yt+,yt−) are large-margin pairs (based on the win-rate rank) within the responses for x from the previous step.
- 6: Contrastive learning: Obtain πt+1 by,

πt+1 ← argmax

π∈Π

E(x,y+

t ,yt−)∼Dt log σ η log

π(yt+ | x) πtτ(yt+ | x) − η log

π(yt− | x) πtτ(yt− | x)

,

where πtτ(y | x) := πt(y | x)1−τ/

η

πref(y | x)τ/

η

,∀(x,y) ∈ X ×Y (the unnormalized version of πtτ(y | x) defined in Eq. (14), after accommodating η → η).

- 7: end for
- 8: return best of π1:(T+1) on the validation data.

and Algorithm 4 requires sampling from the smoothed policy πtτ (the mixture between πt and πref, defined in Eq. (14)). One solution to address this can be sampling from the token-level between πt and πref instead as suggested by Munos et al. (2023).

### B Detailed Proofs

In this section, we provide detailed proofs for our theoretical results. Note that, the definitions and assumptions presented heavily adopts the ideas related to version space and concentrability from reinforcement learning theory literature (esp., Xie et al., 2021, 2023). Nevertheless, the descriptions provided herein are intentionally simplified to elucidate the core insights into the algorithmic design. A full and exhaustive theoretical analysis falls outside the primary scope of this paper. We now make the following definitions and assumptions.

- Definition 1 (Feasible solution space). For each iteration t ∈ [T], we define Πt ⊆ Π as the feasible solution space for iteration t. The πt obtained by Algorithm 1 is always belong to Πt, regardless of the randomness of the data sampling procedure in Algorithm 1.

Here, Definition 1 follows a similar spirit as the version space in RL theory literature, where Πt only contains policies that have a small empirical loss, which can be further converted to a small population loss under standard concentration procedures.

- Definition 2 (Concentrability coefficient over the feasible solution space). For all t ∈ [T], suppose Πt is defined in

- Definition 1, and µ1,t and µ2,t are some given data generate policy. Now, for any t ∈ [T], we define Ct to be the concentrability coefficient at iteration t over its feasible solution space, where

2

⋆ t+1(y1|x)

⋆ t+1(y2|x)

1∼πt⋆+1(·|x),y2∼πt+1(·|x) log π

πt+1(y1|x) − log π

Ex∼ρ,y

πt+1(y2|x)

2 ≤ Ct,

⋆ t+1(y2|x)

⋆ t+1(y1|x)

1∼µ1,t(·|x),y2∼µ2,t(·|x) log π

πt+1(y1|x) − log π

Ex∼ρ,y

πt+1(y2|x)

η : π ∈ Πt ; and here we use the definition of rπ(x,y) := Ey′∼π(·|x) [P(y ≻ y′ | x)],∀(x,y) ∈ X × Y, and Zπ(x) = y∈Y π(y | x)exp r

for any πt+1 ∈ Πt+1 and any πt⋆+1 ∈ Z 1

π(x,·)

π(x)π(· | x)exp r

η ,∀x ∈ X.

π(x,y)

- Definition 2 can be viewed as a natural extension of concentrability from the (offline) reinforcement learning literature to our setup.

- Assumption 1 (Realizability over the feasible solution space). For any π ∈ Πt where Πt is defined in Definition 1 for all t ∈ [T], we assume the following soft-policy iteration update

πSPI(· | x) :=

1 Zπ(x)

π(· | x)exp

rπ(x,·) η

,

where rπ(x,y) := Ey′∼π(·|x) [P(y ≻ y′ | x)],∀(x,y) ∈ X ×Y, and Zπ(x) = y∈Y π(y | x)exp r

π(x,y)

η ,∀x ∈ X is the partition function.

- Assumption 2 (Boundedness over the feasible solution space). Suppose Πt is defined in Definition 1 for all t ∈ [T],

then we assume log ππ(y|x)

t(y|x) ∈ [−Rmax,Rmax] for all π ∈ Π, πt ∈ Πt, and (x,y) ∈ X × Y. Assumption 2 may appear somewhat unconventional, as it explicitly assumes boundedness on the log probabilities. Nonetheless, it is important to note that the value of log ππ(y|x)

t(y|x) is directly measurable and controllable in practice, which is different from the common use case, such as maximum likelihood problems. Theorem 2 (Formal Version of Theorem 1). Under Assumptions 1 and 2, and fix an arbitrary iteration t ∈ [T]. Suppose πt+1 is from Line 4 of Algorithm 1, and πt⋆+1 is defined in Eq. (9). Then, we have

Ex∼ρ DTV(πt+1(· | x),πt⋆+1(· | x)) 2 ≤ O

CtRmax2 log(|Π|/δ) N

,

where Ct is defined in Definition 2. Proof of Theorem 2. We will now present the proof using the following two-step procedure.

- Step 1: From regression with log loss to squared error bound. By standard results on the regression with the logarithmic loss, we know,

log(|Π|/δ) N

. (15)

E(x,y1,y2)∼Dt σ ∆⋆t (x, y1, y2) log σ ∆πt+1,t(x, y1, y2) + σ ∆⋆t (x, y2, y1) log σ ∆πt+1,t(x, y2, y1) ≲

Note that similar results could also apply beyond finite Π. For simplicity, we omit the detailed discussion in our paper. For more in-depth discussions about regression with the logarithmic loss, the reader can refer to, e.g., Foster and Krishnamurthy (2021).

Next, by the Pinsker’s inequality, we have for any z, z ∈ [0,1],

(z − z)2 2 ≤ z log

1 − z 1 − z

z z

+ (1 − z)log

.

Substituting the z and z with Eq. (11) and combining with Eq. (15), we obtain that

log(|Π|/δ) N

t+1,t(x,y2) 2 ≲

, (16)

E(x,y

1,y2)∼Dt σ (rt(x,y1) − rt(x,y2)) − σ rπ

t+1,t(x,y1) − rπ

where a ≲ b means a ≤ c · b for some absolute constant c. Then, by the standard concentration for squared loss, e.g., Lemma A.4 of Xie et al. (2021) with γ = 0, Eq. (16) implies

log(|Π|/δ) N

t+1,t(x,y2) 2 ≲

, (17)

E(x,y

1,y2)∼ρ×µ1:2,t σ (rt(x,y1) − rt(x,y2)) − σ rπ

t+1,t(x,y1) − rπ

where we use “×” as the shorthand of joint distribution for the sake of simplicity, for example, (x,y1,y2) ∼ ρ × µ1:2,t is shorthand for x ∼ ρ,y1 ∼ µ1,t(· | x),y2 ∼ µ2,t(· | x).

By the definition of rt in Line 3 of Algorithm 1, we know rt(x,y) ∈ [0,1] for all (x,y) ∈ X × Y. Thus, by a variant of mean value theorem, we know

rt(x,y1) − rt(x,y2) − rπ

t+1,t(x,y2) ≤

t+1,t(x,y1) + rπ

(18)

ηRmax 1 − σ(1)

σ (rt(x,y1) − rt(x,y2)) − σ rπ

t+1,t(x,y1) − rπ

t+1,t(x,y2) ,

for any (x,y1,y2) ∈ X × Y × Y, where Rmax is introduced from Assumption 2. This is because: let a := rt(x,y1) − rt(x,y2) ∈ [−1,1], and b := rπ

t+1,t(x,y2) ∈ [−ηRmax,ηRmax], and, then, we can directly verify that the slope we need to bound a−b / σ(a)−σ(b) reaches its maximum at a = 1 and b = ηRmax. Combining Eqs. (17) and (18), we obtain

t+1,t(x,y1) − rπ

η2Rmax2 log(|Π|/δ) N

t+1,t(x,y2) 2 ≲

. (19)

E(x,y

1,y2)∼ρ×µ1:2,t rt(x,y1) − rt(x,y2) − rπ

t+1,t(x,y1) + rπ

- Step 2: Concentration in the policy space. We now reason about the concentration of πt+1 → πt⋆+1 from Eq. (19),

where πt⋆+1 is defined in Eq. (9) and πt+1 is the policy corresponding to the learned rπ

t+1,t. By the definition of rπ,t in Eq. (10), we have

rt(x,y1) − rt(x,y2) − rπ

t+1,t(x,y1) + rπ

t+1,t(x,y2)

πt+1(y2 | x) πt(y2 | x)

πt+1(y1 | x) πt(y1 | x)

= rt(x,y1) − rt(x,y2) − η log

+ η log

πt⋆+1(y2 | x) πt+1(y2 | x)

πt⋆+1(y1 | x) πt+1(y1 | x) − η log

. This implies

= η log

2

πt⋆+1(y1 | x) πt+1(y1 | x) − η log

πt⋆+1(y2 | x) πt+1(y2 | x)

E(x,y

1,y2)∼ρ×µ1:2,t η log

πt⋆+1(y1 | x) πt+1(y1 | x) − log

πt⋆+1(y2 | x) πt+1(y2 | x)

=⇒ E(x,y

1,y2)∼ρ×πt⋆+1×πt+1 log

where the last step follows from the definition of Ct (Definition 2). On the other hand, we have

2

πt⋆+1(y1 | x) πt+1(y1 | x) − log

πt⋆+1(y2 | x) πt+1(y2 | x)

E(x,y

1,y2)∼ρ×πt⋆+1×πt+1 log

2

2

πt⋆+1(y2 | x) πt+1(y2 | x)

πt⋆+1(y1 | x) πt+1(y1 | x)

= E(x,y

− 2 log

+ log

1,y2)∼ρ×πt⋆+1×πt+1 log

2

2

πt⋆+1(y | x) πt+1(y | x)

πt⋆+1(y | x) πt+1(y | x)

+ E(x,y)∼ρ×π

= E(x,y)∼ρ×π⋆

log

log

t+1

t+1

πt⋆+1(y | x) πt+1(y | x)

πt+1(y | x) πt⋆+1(y | x)

+ 2Ex∼ρ Ey∼π⋆

·Ey∼π

t+1(·|x) log

t+1(·|x) log

=DKL(πt⋆+1(·|x) ∥ πt+1(·|x))

=DKL(πt+1(·|x) ∥ πt⋆+1(·|x))

2

2

πt⋆+1(y | x) πt+1(y | x)

πt⋆+1(y | x) πt+1(y | x)

≥ E(x,y)∼ρ×π⋆

+ E(x,y)∼ρ×π

log

log

t+1

t+1

Next, we fix an arbitrary x ∈ X, and we have

2

πt⋆+1(y | x) πt+1(y | x)

Ey∼π⋆

t+1(·| x) log

πt⋆+1(y | x) πt+1(y | x)

+ Ey∼π

t+1(·| x) log

η2Rmax2 log(|Π|/δ) N

≲

2

CtRmax2 log(|Π|/δ) N

, (20)

≲

πt⋆+1(y1 | x) πt+1(y1 | x) · log

πt⋆+1(y2 | x) πt+1(y2 | x)

. (21)

2

2

2

πt⋆+1(y | x) πt+1(y | x)

πt⋆+1(y | x) πt+1(y | x)

(by Jensen’s inequality)

+ Ey∼π

≥ Ey∼π⋆

t+1(·| x) log

t+1(·| x) log

2

πt⋆+1(y | x) πt+1(y | x)

πt⋆+1(y | x) πt+1(y | x)

, (22)

≳ Ey∼π⋆

+ Ey∼π

t+1(·| x) log

t+1(·| x) log

where a ≳ b means a ≥ c · b for some absolute constant c. We now recall the definition of f-divergence: Df(p,q) := Ey∼q[f(p(y)/q(y))] for two distributions p and q, where f : R+ → R is convex with f(1) = 0. Thus, we can notice that,

- p(y)

- q(y)

- p(y)

- q(y)

(p,q), where f1(u) := (1 + u) · |log(u)|, u ∈ R+. (23)

Ey∼p log

+ Ey∼q log

= Df

1

On the other hand, by the definition of total variation distance, we know

- 1

- 2 |u − 1|, u ∈ R+. (24)

(p,q), where f2(u) := It is easy to verify that

DTV(p,q) = Df

2

- 1

- 2 |u − 1| ≥ 0, for all u ∈ R+, (25)

f1(u) − f2(u) = (1 + u) · |log(u)| −

as it is a convex function on R+ with a minimum of 0 at u = 1. Therefore, combining Eqs. (22) to (25), we obtain

2

πt⋆+1(y | x) πt+1(y | x)

Ey∼π⋆

t+1(·| x) log

We then apply Eq. (26) to Eq. (21),

πt⋆+1(y | x) πt+1(y | x)

+ Ey∼πt+1(·| x) log

2

≳ [DTV(πt+1(· | x), πt⋆+1(· | x))]2 . (26)

Ex∼ρ DTV(πt+1(· | x),πt⋆+1(· | x)) 2

πt⋆+1(y1 | x) πt+1(y1 | x) − log

πt⋆+1(y2 | x) πt+1(y2 | x)

≲ E(x,y

1,y2)∼ρ×πt⋆+1×πt+1 log

CtRmax2 log(|Π|/δ) N

≲

,

where the last step follows from Eq. (20). This completes the proof.

2

| |
|---|

### C Additional Experimental Details

Batched Prompting: We also show in Figure 3 the prompt that we send to GPT-4 to annotate preferences. For the sake of efficiency, we “batch” requests to GPT-4, meaning that instead of sending every pair of candidate responses to be annotated, we show all candidates side-by-side and ask GPT-4 to apply the scoring rubric to each one in the context window.

Cost Analysis: We also do a brief cost analysis associated with the scaled-up experiment on 600k training inputs. The major line items are the cost of sampling outputs, annotating them with GPT-4 to construct training pairs, and then training the next iteration against those pairs. For each of the six iterations:

- 1. Sampling: it took about 18-24 hours to inference 5 outputs for all 100k examples on 10 8xA100 80GB pods, depending on the average length, costing about $6,000 based on spot pricing.
- 2. Annotation: the average number of prompt tokens sent to GPT-4 for annotation across iterations was about 450M, with an average of about 60M completion tokens, amounting to about $34,000 based on the version of the endpoint we were using.
- 3. Training: ironically, training was the cheapest step, taking only 12-24 hours on two 8xA100 80GB nodes.

|Review the user’s question and the corresponding response using the additive 6-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:<br><br>1. Give 1 point if the response is formatted as an answer, e.g. does not repeat itself over and over, is not a continuation of the question or other random question/answer pairs or examples, etc.<br>2. Add 1 more point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.<br>3. Add a third point if the response addresses a substantial portion of the user’s question without rambling, but does not completely resolve the query or provide a direct answer.<br>4. Award a fourth point if the response answers nearly all the elements of the user’s question correctly, in a useful and direct way, and is largely devoid of errors, extraneous content, or excess wordiness.<br>5. Grant a fifth point if the response is entirely correct (without any errors or made-up facts), and directly answers all aspects of the user’s question in a helpful, well-organized, focused, and succinct style.<br>6. Bestow a sixth point for a response that is entirely correct *AND* concise, impeccably tailored to the user’s question, without any unnecessary information or wordiness, reflecting expert knowledge, amounting to a superbly high-quality, focused, and insightful answer.<br><br><br>### User: <instructions> ### Candidate Responses: <candidate answers> ### Instructions: After examining the user’s instruction and the response:<br><br>1. Briefly justify your total score for EACH answer, up to 100 words.<br>2. score EACH answer using the format: "Score: <totalpoints>"<br><br><br>Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria. Remember to score EACH candidate answer separately. Format your justification and score only as a python dictionary with no extraneous output e.g. {“id1_justification”: “...”, “id1_score”: 3, “id2_justification” : ...} with “idX” corresponding to the id of the X’th answer. {“id1_justification” :|
|---|

###### Figure 3: The annotation prompt submitted to GPT-4 to collect multiple side-by-side preference scores for DNO. One of the candidates is also an output from GPT-4 itself, along with several outputs from the current policy.

|Example 1 Question: When will the earth run out of fresh water?|
|---|
|Example 1: Initial SFT model output<br><br>The earth will not run out of fresh water. There is a finite amount of fresh water on the planet, but it is constantly being replenished through the water cycle. The water cycle involves the evaporation of water from the earth’s surface, its condensation into clouds, and its precipitation back to the earth’s surface as rain, snow, or other forms of precipitation. This cycle ensures that fresh water is always available, although the distribution of fresh water may vary depending on factors such as climate, geography, and human use.|

|Example 1: DNO Iter 1<br><br>It is difficult to predict exactly when the Earth will run out of fresh water, as it depends on various factors such as climate change, population growth, water usage, and technological advancements in water management and conservation. However, currentestimatessuggestthatwearealreadyfacingaglobalwatercrisis, withmanyregionsexperiencingwaterscarcityandstress.<br><br>According to the United Nations, about 2.2 billion people worldwide lack access to safe drinking water, and this number is expected to increase due to population growth and climate change. The World Wildlife Fund (WWF) estimates that by 2025, half of the world’s population will be living in water-stressed areas.<br><br>While the Earth’s total water supply remains constant, the availability of fresh water is decreasing due to overuse, pollution, and climate change. It is important to note that fresh water is a renewable resource, but it takes time for it to replenish naturally. Therefore, it is crucial to manage and conserve water resources to ensure their sustainability for future generations.<br><br>To address the global water crisis, it is necessary to implement sustainable water management practices, improve water infrastructure, promote water conservation and efficiency, and raise awareness about the importance of water conservation.<br><br>In summary, while the Earth will not run out of water, the availability of fresh water is becoming increasingly scarce in many regions. It is important to take action to manage and conserve water resources to ensure their sustainability for future generations.<br><br>It is also important to note that the availability of fresh water varies depending on the location and the specific water sources. Some regions may face water scarcity sooner than others, and it is important to address these issues on a local level.<br><br>In conclusion, while it is difficult to predict exactly when the Earth will run out of fresh water, it is clear that we are already facing a global water crisis. It is important to take action to manage and conserve water resources to ensure their sustainability for future generations.<br><br>Reference: United Nations. (2019). World Water Development Report 2019. Retrieved from https://www.unwater.org/publications/worldwater-development-report-2019/ World Wildlife Fund. (2019). Water Scarcity. Retrieved from https://www.worldwildlife.org/issues/water-scarcity<br><br>Note: The information provided in this response is based on current scientific knowledge and available data. However, it is important to note that the field of water resources management is constantly evolving, and new research and findings may emerge in the future.<br><br>Note 2: The response provided is based on the information available at the time of writing and may not reflect the most current data or research. It is important to consult the latest scientific literature and data sources for the most up-to-date information on the topic.<br>Note 3: The response provided is based on the assumption that the question is asking about the availability of fresh water for human consumption. However, it is important to note that fresh water is also essential for ecosystems and other living organisms. The availability of fresh water for all purposes is a complex issue that requires a holistic approach to management and conservation.<br>Note 4: The response provided is based on the assumption that the question is asking about the availability of fresh water on a global scale.<br>|
|---|

|Example 1: DNO Iter 2<br><br>It is not accurate to say that the Earth will "run out" of fresh water, but rather that we could potentially deplete or contaminate fresh water resources to a point where they can no longer sustain us. Fresh water is a renewable resource, but its availability depends on various factors such as climate, geography, and human activities. Here are some key points to consider:<br><br>1. Water cycle: The water cycle, which includes evaporation, condensation, precipitation, and runoff, constantly replenishes fresh water resources. However, the rate of replenishment varies depending on the region and climate.<br>2. Groundwater: Groundwater is a significant source of fresh water, but it is not infinite. Over-extraction of groundwater can lead to depletion and contamination. It takes thousands of years for groundwater to replenish naturally.<br>3. Surface water: Surface water, such as rivers and lakes, is also a renewable resource, but it can be affected by climate change, deforestation, and other human activities. Droughts and changes in precipitation patterns can reduce the availability of surface water.<br>4. Contamination: Contamination from pollution, agricultural runoff, and other sources can render fresh water unusable. This includes both groundwater and surface water.<br>5. Population growth: As the global population continues to grow, the demand for fresh water is increasing. This can lead to over-extraction, contamination, and other issues that can impact the availability of fresh water.<br>6. Climate change: Climate change is expected to exacerbate water scarcity in many regions due to changes in precipitation patterns, increased evaporation, and melting of glaciers and snowpacks.<br><br><br>In conclusion, while fresh water is a renewable resource, its availability is not infinite and can be impacted by various factors. It is essential to manage and conserve fresh water resources sustainably to ensure their long-term availability for future generations. So, instead of asking when the Earth will "run out" of fresh water, it is more accurate to ask how we can manage and conserve fresh water resources to ensure their long-term sustainability.|
|---|
|Example 1: DNO Iter 3<br><br>The Earth as a planet does not "run out" of fresh water, as it continues to cycle through the hydrological cycle, which includes processes such as evaporation, condensation, precipitation, and infiltration. However, the availability and accessibility of fresh water for human use are significant concerns.<br><br>The freshwater crisis is primarily driven by population growth, urbanization, agricultural expansion, and climate change, which lead to increased demand, pollution, and variability in water availability. According to the United Nations, approximately 2.2 billion people lack access to safe drinking water services, and 4.2 billion people experience severe water scarcity for at least one month a year.<br><br>While the Earth will not run out of fresh water, the distribution and availability of fresh water for human use are becoming increasingly challenging. To address this issue, it is crucial to implement sustainable water management practices, reduce water waste, and promote equitable access to fresh water resources.<br><br>In summary, the Earth will not run out of fresh water, but the availability and accessibility of fresh water for human use are becoming increasingly limited, and it is essential to address this challenge through sustainable water management practices and equitable access to resources.|

- Table 5: Outputs of the various DNO models across iterations for the question: When will the earth run out of fresh water?. The output for Iter-1 is a bit too long as shown in Table 2. We believe this could be addressed with better hyperparameter tuning or preference data. We fine the initial SFT model’s response to be missing important points about how the premise of the question is best addressed by highlighting access to freshwater. The last reponse, for Iter

- 3 is more informative and specific than the initial SFT response.

|Example 2 Question: what data structure would you use to design a url lookup system to quickly determine whether a url exists, or if not, its closest prefix. Explain.|
|---|

|Example 2: Initial SFT model output To design a URL lookup system that can quickly determine whether a URL exists or find its closest prefix, we can use a Trie data structure.<br><br>A Trie is a tree-like data structure that stores strings in a hierarchical manner. Each node in the Trie represents a character in the string, and the edges represent the characters themselves. The root node represents an empty string, and each subsequent node represents a character appended to the previous node’s string.<br><br>In our URL lookup system, we can use a Trie to store all the existing URLs. Each node in the Trie will represent a URL prefix, and the edges will represent the characters in the URL.<br><br>To determine whether a URL exists, we can traverse the Trie starting from the root node. If we reach a node that represents a complete URL, we know that the URL exists. If we reach a node that represents an incomplete URL, we know that the URL does not exist.<br><br>To find the closest prefix of a URL, we can traverse the Trie starting from the root node. We can keep track of the longest prefix we have encountered so far. Whenever we encounter a node that represents a URL prefix, we compare its length with the length of the longest prefix we have encountered so far. If the length of the current prefix is greater, we update the longest prefix.<br><br>By using a Trie, we can efficiently determine whether a URL exists or find its closest prefix in constant time complexity. This is because the Trie allows us to traverse the data structure in a hierarchical manner, and we can quickly determine whether a URL exists or find its closest prefix by following the appropriate edges.|
|---|
|Example 2: DNO Iter 1<br><br>To design a URL lookup system that quickly determines whether a URL exists or finds its closest prefix, you can use a combination of a Trie (Prefix Tree) and a Bloom Filter. This approach provides efficient lookup and prefix matching while maintaining a low memory footprint.<br><br>1. Trie (Prefix Tree): A Trie is a tree-like data structure that stores strings as nodes. Each node in the Trie represents a character in the string, and the edges represent the characters themselves. The root node is empty, and each subsequent node contains a single character. This data structure allows for efficient prefix matching and string lookup.<br>2. Bloom Filter: A Bloom Filter is a probabilistic data structure that uses a bit array and a set of hash functions to determine whether an element is a member of a set. It provides a fast way to check if a URL exists in the system, but it may produce false positives.<br><br><br>Here’s how you can use these data structures together:<br><br>1. Create a Trie to store all the URLs in the system. Each URL is added as a node in the Trie, and the edges represent the characters in the URL.<br>2. Create a Bloom Filter to store the URLs in the system. The Bloom Filter will have a fixed size and a set of hash functions. When a URL is added to the system, it is hashed using the hash functions, and the resulting bits are set in the Bloom Filter.<br>3. To check if a URL exists, first check the Bloom Filter. If the URL is not present in the Bloom Filter, it does not exist in the system. If the URL is present in the Bloom Filter, look up the URL in the Trie. If the URL is found in the Trie, it exists in the system. If the URL is not found in the Trie, it may be a prefix of an existing URL.<br>4. To find the closest prefix, look up the URL in the Trie. If the URL is not found, start from the root node and traverse the Trie until you find the longest prefix that matches the input URL. This will give you the closest prefix.<br><br><br>This approach provides efficient lookup and prefix matching while maintaining a low memory footprint. The Trie allows for efficient string lookup, and the Bloom Filter provides a fast way to check if a URL exists in the system. However, the Bloom Filter may produce false positives, so it should be used in conjunction with the Trie to ensure accuracy.<br><br>Note: The Trie and Bloom Filter can be optimized further by using a compressed Trie or a Bloom Filter with a variable-size bit array. These optimizations can reduce the memory footprint and improve the performance of the system.|

|Example 2: DNO Iter 2 To design a URL lookup system that can quickly determine if a URL exists or find its closest prefix, you can use a data structure called a Trie (also known as a Prefix Tree). A Trie is a tree-like data structure that stores strings as nodes, where each edge is labeled with a single character. This data structure is particularly efficient for searching and inserting strings with common prefixes. Here’s how you can use a Trie to implement a URL lookup system:<br><br>1. Create a Trie node structure with properties like ‘is_end_of_word‘ (indicates if the node represents a complete URL) and a dictionary to store child nodes.<br>2. Insert URLs into the Trie by iterating over each character in the URL and adding it as a child node to the current node. If the character is not present in the current node’s dictionary, create a new node and add it.<br>3. To search for a URL, start at the root node and iterate over each character in the URL. If the character is present in the current node’s dictionary, move to the corresponding child node. If the character is not present, return the closest prefix by traversing the Trie upwards from the last valid node.<br><br><br>Here’s an example implementation in Python:<br><br>class TrieNode:<br><br>def __init__(self): self.is_end_of_word = False self.children = {}<br><br>class Trie: def __init__(self):<br><br>self.root = TrieNode()<br><br>def insert(self , url): current_node = self.root for char in url:<br><br>if char not in current_node.children:<br><br>current_node.children[char] = TrieNode() current_node = current_node.children[char]<br><br>current_node.is_end_of_word = True<br><br>def search(self , url): current_node = self.root for char in url:<br><br>if char in current_node.children:<br><br>current_node = current_node.children[char] else:<br><br>return self.find_closest_prefix(current_node) return url if current_node.is_end_of_word else None<br><br>def find_closest_prefix(self , current_node): prefix = [] while current_node:<br><br>prefix.append(list(current_node.children.keys())[0]) current_node = current_node.children[list(current_node.children.keys())[0]]]<br><br>return ’’.join(reversed(prefix))[::-1]<br><br>This implementation allows you to quickly determine if a URL exists or find its closest prefix by searching the Trie. The ‘insert‘ method inserts URLs into the Trie, and the ‘search‘ method searches for a URL or returns its closest prefix if it doesn’t exist.|
|---|

|Example 2: DNO Iter 3 To design a URL lookup system that can quickly determine if a URL exists or find its closest prefix, you can use a data structure called the Trie (also known as Prefix Tree). A Trie is a tree-like data structure where each node represents a character in the key (URL in this case) and each branch corresponds to a possible character in the alphabet. Here’s how you can implement a URL lookup system using a Trie:<br><br>1. Create a Trie node with an end-of-string (EOS) marker. This node will represent the end of a URL and will indicate that the URL exists.<br>2. For each URL in your dataset, insert it into the Trie. While inserting, traverse the Trie by following the corresponding characters of the URL. If a character doesn’t exist in the Trie, create a new node for it.<br>3. When inserting a URL, follow these steps:<br><br><br>(a) Traverse the Trie using the URL’s characters.<br>(b) If a node doesn’t have a child for the next character, create a new node for it.<br>(c) If a node doesn’t have a child for the next character, create a new node for it.<br>(d) Repeat step a and b until you reach the end of the URL.<br>(e) Set the EOS marker for the last node you visited.<br><br><br>To search for a URL or find its closest prefix, follow these steps:<br><br>1. Traverse the Trie using the characters of the URL you want to search.<br>2. If a node doesn’t have a child for the current character, stop the search. This URL doesn’t exist in the Trie.<br>3. If you reach the EOS marker, the URL exists in the Trie.<br>4. If you stop the search before reaching the EOS marker, you’ve found the closest prefix of the URL.<br>|
|---|

- Table 6: Outputs of the various DNO models across iterations for an interview question relating to the design of a URL lookup system.. The last response from Iter 3 is more informative and clearer, and doesn’t contain misleading information (searching a trie runs in linear complexity in the length of the strings, not constant...). The response for Iter 1 contains an implementation of a Trie, which may be unnecessary because the user did not ask for it.

|Example 3 Question: What factors led to the outbreak of ww1|
|---|
|Example 3: Initial SFT model output There were several factors that led to the outbreak of World War I:<br><br>1. Nationalism: The rise of nationalism in Europe led to a desire for territorial expansion and the desire to unite people of the same ethnicity or language under one nation. This led to tensions between countries and a desire to assert dominance over neighboring territories.<br>2. Imperialism: The scramble for colonies in Africa and Asia led to competition between European powers, which further fueled tensions and rivalries.<br>3. Militarism: The arms race between European powers led to a buildup of military forces and weapons, which created a sense of insecurity and fear among neighboring countries.<br>4. Alliances: The complex system of alliances between European powers created a web of interdependencies that made it difficult for any one country to remain neutral in the event of a conflict.<br>5. Assassination of Archduke Franz Ferdinand: The assassination of Archduke Franz Ferdinand of Austria-Hungary by a Serbian nationalist in 1914 was the immediate trigger for the war, but it was the culmination of these other factors that led to the outbreak of the war.<br>|

|Example 3: DNO-More-Data Iter 1 The outbreak of World War I was the result of a complex interplay of factors, including:<br><br>1. Imperialism and Colonialism: The competition for colonies and territories among European powers led to tensions and rivalries. The scramble for Africa and the race to acquire overseas territories created a sense of competition and mistrust among nations.<br>2. Militarism and Arms Race: The arms race among European powers, particularly between Germany and Britain, led to a buildup of military power. The development of new technologies, such as machine guns, artillery, and submarines, increased the destructive potential of warfare.<br>3. Alliances and Blocs: The complex web of alliances and blocs among European powers created a situation where conflicts in one region could escalate into a larger war. The Triple Entente (France, Russia, and Britain) and the Triple Alliance (Germany, Austria-Hungary, and Italy) were the main alliances that formed the basis of the conflict.<br>4. Nationalism and Ethnic Tensions: The rise of nationalism and ethnic tensions in various regions, such as the Balkans, contributed to the outbreak of war. The desire for self-determination and independence led to conflicts and tensions among different ethnic groups.<br>5. Assassination of Archduke Franz Ferdinand: The assassination of Archduke Franz Ferdinand of Austria-Hungary by a Serbian nationalist in 1914 was the immediate trigger for the war. The complex web of alliances and rivalries among European powers led to a series of diplomatic maneuvers and ultimatums, which ultimately resulted in the declaration of war.<br>6. Economic Factors: Economic competition and trade disputes among European powers also contributed to the tensions leading up to the war. The economic rivalry between Germany and Britain, particularly in the context of colonial competition, added to the existing tensions.<br>7. Political Instability: Political instability and power struggles within various countries, such as the assassination of King Alexander I of Serbia in 1903, created a sense of uncertainty and instability. These political upheavals contributed to the overall atmosphere of tension and conflict.<br>8. Failure of Diplomacy: The failure of diplomatic efforts to resolve conflicts and tensions among European powers led to the breakdown of communication and trust. The inability of leaders to find peaceful solutions to disputes contributed to the escalation of tensions and the eventual outbreak of war.<br>9. Miscalculation and Miscommunication: Miscalculations and miscommunications among military and political leaders also played a role in the outbreak of war. The lack of clear communication and misunderstandings between different countries led to a series of missteps and escalations.<br>10. Systemic Factors: The complex system of alliances, blocs, and rivalries among European powers created a situation where conflicts in one region could have a domino effect on other regions. The interconnectedness of these factors led to a chain reaction of events that ultimately led to the outbreak of World War I.<br>|
|---|

|Example 3: DNO-More-Data Iter 6 The outbreak of World War 1, which started on July 28, 1914, resulted from a complex web of events and circumstances that had been developing for years prior. Here are some of the key factors that led to the war:<br><br>1. **Militarism:** All major powers had been engaged in an arms race, building up their military forces and fortifications along their borders. This led to a sense of insecurity and a belief that preventive war was necessary to secure national interests.<br>2. **Alliance System:** Europe was divided into two major alliances: the Triple Entente (France, Russia, and Britain) and the Triple Alliance (Germany, Austria-Hungary, and Italy). These alliances created a complex web of obligations that meant if one member was attacked, all members were obliged to come to their aid.<br>3. **Imperialism:** The scramble for colonies had led to territorial disputes and competition for resources, leading to tensions between the major powers.<br>4. **Nationalism:** Strong feelings of national identity and pride led to a disregard for other nations’ interests. The assassination of Archduke Franz Ferdinand of Austria, which triggered the war, was a result of nationalist tensions in the Balkans.<br>5. **Security Dilemma:** The actions of one country led other countries to take defensive measures, which in turn made the first country feel more insecure, leading to a spiral of actions and reactions that escalated the situation.<br>6. **Assassination of Archduke Franz Ferdinand:** The assassination by Gavrilo Princip, a member of the Black Hand, a Serbian nationalist group, provided the immediate trigger for the war.<br>7. **Ultimatum to Serbia:** Austria-Hungary delivered an ultimatum to Serbia, which Serbia did not accept fully, leading to Austria-Hungary declaring war on Serbia.<br><br><br>These factors combined to create a powder keg that was waiting for a spark to ignite the war. Once the spark was lit, the complex web of alliances meant that the conflict quickly escalated into a global war.|
|---|

- Table 7: Outputs of first and last DNO-More-Data iterations for the third example What factors led to the outbreak of ww1?. The last response from Iter 6 has a higher information density; it recalls more key facts and entities, including the specific date of the start of WW1, Triple Entente, the Triple Alliance, Gavrilo Princip, and the Black Hand. Iteration 1 also contains some of this information, but is too wordy. The initial SFT model seems a lot more superficial.

