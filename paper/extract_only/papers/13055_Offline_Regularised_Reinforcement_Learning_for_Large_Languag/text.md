arXiv:2405.19107v1[cs.LG]29May2024

[Figure 1]

Ofﬂine Regularised Reinforcement Learning for Large Language Models Alignment

[Figure 2]

Pierre Harvey Richemond∗ Google DeepMind

Yunhao Tang∗ Google DeepMind

Daniel Guo Google DeepMind

Daniele Calandriello Google DeepMind

Mohammad Gheshlaghi Azar Cohere

Rafael Rafailov Stanford University

Bernardo Avila Pires Google DeepMind

Eugene Tarassov Google DeepMind

Lucas Spangher Google DeepMind

Will Ellsworth Google DeepMind

Aliaksei Severyn Google Research

Jonathan Mallinson Google Research

Lior Shani Google Research

Gil Shamir Google DeepMind

Rishabh Joshi Google DeepMind

Tianqi Liu Google DeepMind

Remi Munos∗ Google DeepMind

Bilal Piot∗ Google DeepMind

Abstract

The dominant framework for alignment of large language models (LLM), whether through reinforcement learning from human feedback or direct preference optimisation, is to learn from preference data. This involves building datasets where each element is a quadruplet composed of a prompt, two independent responses (completions of the prompt) and a human preference between the two independent responses, yielding a preferred and a dis-preferred response. Such data is typically scarce and expensive to collect. On the other hand, single-trajectory datasets where each element is a triplet composed of a prompt, a response and a human feedback is naturally more abundant. The canonical element of such datasets is for instance an LLM’s response to a user’s prompt followed by a user’s feedback such as a thumbs-up/down. Consequently, in this work, we propose DRO, or Direct Reward Optimisation, as a framework and associated algorithms that do not require pairwise preferences. DRO uses a simple mean-squared objective that can be implemented in various ways. We validate our ﬁndings empirically, using T5 encoder-decoder language models, and show DRO’s performance over selected baselines such as Kahneman-Tversky Optimization (KTO). Thus, we conﬁrm that DRO is a simple and empirically compelling method for single-trajectory policy optimisation.

- 1 Introduction

Aligning the behavior of artiﬁcial agents with human preferences is critical for improving quality, helpfulness and safety [Bai et al., 2022a] of agents’ responses. The most established methodology for human alignment is Reinforcement Learning from Human Feedback (RLHF) [Knox and Stone, 2008, Grifﬁth et al., 2013, Christiano et al., 2017, Warnell et al., 2018] which consists of ﬁne-tuning

[Figure 3]

∗Equal contribution. Correspondence at richemond@google.com or piot@google.com.

pre-trained Large Language Models (LLMs) [Glaese et al., 2022, OpenAI, 2022]. More precisely, it typically entails learning a reward model under the Bradley-Terry model [Bradley and Terry, 1952] of human preferences and subsequently employing reinforcement learning (RL) to optimise the LLM’s performance as judged by this reward model [Christiano et al., 2017, Ziegler et al., 2020]. This optimisation is done by generating a set of responses and their associated rewards from a set of chosen prompts. Therefore, on top of learning an additional reward model, this method requires sampling from the LLM at training time, which is costly and technically challenging.

Rafailov et al. [2023] introduced an alternative, reward-free and sampling-free method termed direct preference optimisation (DPO). This popular approach uses a supervised objective that contrasts pairs of responses to a speciﬁc prompt. DPO is able to circumvent the explicit learning of a reward signal, while remaining mathematically equivalent to the traditional RL approach, as proven by Azar et al. [2023]. Such ofﬂine preference optimisation method has been extended to a few variants in follow-up work (see, e.g., [Zhao et al., 2023a, Tang et al., 2024]) and gained popularity in practice.

Despite DPO’s widespread use, there is still an important remaining shortcoming to this approach which is the high-cost of collecting human preferences. Establishing human preferences might over time become a self-defeating endeavour: as LLMs improve in quality, the task of distinguishing between a pair of strong responses gets increasingly difﬁcult (see, e.g., arguments in [Saunders et al.,

- 2022, Bowman et al., 2022]), and would require additional efforts in improving both the quality and scale of the collected human feedback.

Second and more importantly, annotating pairwise data is more expensive and less natural than simply indicating whether a single completion is satisfactory or not, e.g., by assigning a binary thumbs up or down rating to the model completion. The former is generally carried out by paid raters whereas the latter could be produced by users at a much larger scale. Consequently, singletrajectory data is much more abundant in the wild, hence, cheaper and more easily collected than scarce preference data. Leveraging single trajectory data promises to unlock the beneﬁts associated with scale in deep learning. Analogous algorithmic advances motivated by the need to decrease supervision signals have often resulted in important step-changes in the empirical capacities of deep networks, e.g. through self-supervised systems [Devlin et al., 2019, Chen et al., 2020, Grill et al., 2020, Radford et al., 2021].

In order to exploit this single trajectory setting, we introduce Direct Reward Optimisation (DRO). DRO is a framework, derived from mathematical ﬁrst principles. DRO is designed to work in the ofﬂine single-trajectory setting with human feedback. Speciﬁcally, our contributions are as follows:

- • We introduce DRO as a generic framework performing single-trajectory RLHF optimisation thanks to a simple quadratic objective and perform theoretical analysis.
- • We propose a practical instantiation of DRO, DRO-V which combines ofﬂine policy learning with a value function learning, and hence the sufﬁx -V.
- • We compare DRO-V against Kahneman-Tversky Optimization (KTO) [Ethayarajh et al., 2024], an algorithm that has also been speciﬁcally designed for the single-trajectory setting. We ﬁnd that DRO-V signiﬁcantly outperforms KTO, when using T5 encoders [Raffel et al., 2020] with up to 3 billion parameters, on the UltraFeedback dataset [Cui et al., 2023]. We perform several ablations to investigate and understand our algorithm’s empirical performance.

- 2 Background

Here we introduce background for RLHF and a few important alignment algorithms.

Standard pairwise alignment. Ofﬂine-alignment of LLMs has mainly been achieved using preference datasets of the form (xi,yiw,yil)Ni=1, where we are given a prompt xi, and a pair of prompt completions (or generations) (yiw,yil) with yiw the preferred generation and yil the dis-preferred one. Most objectives in the RLHF literature, such as DPO [Rafailov et al., 2023], IPO [Azar et al., 2023] or SLiC [Zhao et al., 2023a], can be described and subsumed by the following loss:

L(θ) =

n

1 n

f β · log

[Figure 4]

i=1

πθ(yiw|xi) πref(yiw|xi)

[Figure 5]

− log

πθ(yil|xi) πref(yil|xi)

[Figure 6]

,

where β is a scalar, n a batch size, πθ is the parameterised policy and πref a reference policy, typically obtained after a ﬁrst step of pre-training and supervised ﬁne-tuning. f is a scalar function; each choice of function f results in a speciﬁc given algorithm: for example, f(z) = log (1 + exp(−z)) for DPO; f(z) = max(0,1 − z) for SLiC; f(z) = (z − 1)2 for IPO, among other possible alternatives as discussed in [Tang et al., 2024].

We argue that preference datasets are expensive to build and do not occur naturally in the wild. Most data coming from user logs is not collected pairwise, but instead comes in the form of a single trajectory.

Single-trajectory setting. Formally, we consider single-trajectory datasets of the form (xi,yi,ri)Ni=1 where xi is a prompt, yi a generation and ri a scalar reward, collected by some unknown behavior policy. As a simple example to model the thumbs-up vs. thumbs-down response, we can set a binary reward where ri = 1 is for thumbs-up. This formulation can be understood as a special case of the ofﬂine RL setup [Levine et al., 2020] tailored to the contextual bandit case for RLHF.

Perhaps surprisingly, few ofﬂine alignment methods consider this setting with the exception of Kahneman-Tversky Optimization (KTO) [Ethayarajh et al., 2024]. KTO is derived from principles related to utility and prospect theory [Kahneman and Tversky, 1979] that build upon the notion of human risk aversion. KTO also makes strong simplifying assumptions, which as we will show, biases the method to produce suboptimal policies. In contrast, we are interested in deriving a simple, general purpose and performantalgorithm, withoutstrong dependencyon mathematicalassumptions on risk preference or utility.

Online vs. ofﬂine algorithms. Whilst a natural idea might be to try and use online RL (like in [Calandriello et al., 2024, Guo et al., 2024]), this would require one to either generate new prompt completions online, or to correct for the distribution of the online policy by using importance sampling. The latter in turn would bring its own set of challenges, e.g., high variance in the importance sampling ratios. In order to circumvent the associated difﬁculties, we consider the ofﬂine setting instead. Additionally, ofﬂine RL brings orthogonal beneﬁts of its own, such as simplicity and computational efﬁciency. Therefore in the following, we present an ofﬂine, sound and practical method that approximates the optimal closed-form policy.

- 3 Direct Reward Optimisation (DRO)

In this section we present the main contribution of this work. We start with some theoretical background on the policy optimisation setting. We then introduce the DRO objective and discuss a few important theoretical properties, followed by the design of practical algorithms.

- 3.1 KL regularised policy optimisation with single-trajectory data

Complementary to the pairwise preference setting highlighted above, we now consider the single trajectory case. We want to approximate the optimal Kullback-Leibler (KL) regularised policy π∗, following the canonical formulation of RLHF [Christiano et al., 2017]:

π∗(x) def= arg max

π

Ex∼ρ,y∼π(·|x)[r(x,y) − τ · KL(π(·|x) || πref(·|x))], (1)

where πref is some initial reference policy, such as the policy obtained after pretraining and supervised ﬁne-tuning. We then have that, necessarily,

πref(y|x)eτ1r(x,y)

[Figure 7]

π∗(y|x) =

eτ1V ∗(x) , (2) where V ∗(x) def= τ log Ey∼π

[Figure 8]

[Figure 9]

ref(·|x)[eτ1r(x,y)] is a function that depends on the regulariser πref. Importantly, the normalisation constant, also called partition function, that normalizes the numerator in Equation (2) can be explicitly written as Z = exp τ1V ∗(x). The denominator log-sum-exp as a value function has already appeared in the literature about soft reinforcement learning [Ziebart et al., 2008, Haarnoja et al., 2017, Richemond and Maginnis, 2017, Schulman et al., 2018]. Unlike KTO, which assumes a constant partition function Z for each prompt of the batch, we do not make any

[Figure 10]

[Figure 11]

assumptions on the form of Z or V . Unlike DPO or IPO, where cancellation of the partition function happens due to the difference of rewards in the Bradley-Terry preference model, we do not assume any functional form for V . For technically minded readers, we expand on these points in Appendix D. We also note that the form of the partition function is intuitive, if we consider the LegendreFenchel conjugate [Bauschke and Combettes, 2011] of the KL regulariser in Equation (1).

The DRO objective. Now, we rearrange the optimality condition from above that holds jointly with π = π∗,V = V ∗,

π(y|x) πref(y|x)

(3)

r(x,y) − V (x) = τ log

[Figure 12]

Multiple objectives (and subsequently algorithms) can be derived in order to enforce or approximate this condition, such that we refer to DRO as a framework encompassing a plurality of algorithms. We can focus in particular on deﬁning the following DRO loss for any pair of policy and value functions (π,V ):

LDRO(π,V ) def=

- 1

[Figure 13]

- 2

π(y|x) πref(y|x)

Ex∼ρ,y∼µ(·|x) r(x,y) − V (x) − τ log

[Figure 14]

2

. (4)

We begin with an existence and unicity result for the optimum of LDRO, thereafter denoted as L.

Theorem 1. (π∗,V ∗) is a global optimum of the loss L(π,V ). In addition, assuming the supports of µ and πref coincide, (i.e., for all x ∈ suppρ, supp(µ(·|x)) = supp(πref(·|x))), then (π∗,V ∗) is the unique global optimum of the loss L(π,V ).

Proof. From the deﬁnition of π∗ and V ∗, we have that L(π∗,V ∗) = 0. Now notice that L(π,V ) is non-negative since it is a sum of quadratic terms t(x,y)2 where t(x,y) def= r(x,y) − V (x) − τ log ππ(y|x)

ref(y|x). Thus (π∗,V ∗) is a global optimum of L(π,V ).

[Figure 15]

Now let us prove that it is unique. Assume there is another global optimum (˜π,V˜) such that L(˜π,V˜) = 0. This means that for all x ∈ supp(ρ) and y ∈ suppµ(·|x), its t(x,y)-term is zero. Since the support of µ and πref coincide, we have that ∀x ∈ supp(ρ),∀y ∈ suppπref(·|x),

π˜(y|x) πref(y|x)

r(x,y) − V˜(x) − τ log

= 0,

[Figure 16]

from which we deduce that

1 τ r(x,y)

πref(y|x)e

[Figure 17]

V˜(x) .

π˜(y|x) =

[Figure 18]

eτ1

[Figure 19]

But since π˜(·|x) is a probability distribution, we must have that V˜(x) = τ log y πref(y|x)eτ1r(x,y). Thus V˜(x) = V ∗(x) and π˜(y|x) = π∗(y|x) for all x ∈ suppρ, y ∈ supp(πref(·|x)).

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Remarks. The signiﬁcance of Theorem 1 lies in the fact that, though π∗ and V ∗ are clearly related via log π∗(y|x)+V ∗(x) = log πref(y|x)+ τ1r(x,y), there is no need to account for that connection during optimisation. We can optimize π and V independently and end up ﬁnding the optimum. This can greatly simplify algorithmic design in practice, as we will see shortly.

[Figure 25]

Role and necessity of the V function. The loss function requires a joint optimisation over (π,V ), which means separately parameterising a value function V in addition to the policy π so as to be theoretically sound. Thus, DRO stands in strong contrast to alternative algorithms such as KTO [Ethayarajh et al., 2024], which uses a policy loss only without a value function. One natural question arises as to whether it is possible to convert the loss function jointly over (π,V ) into a loss over π alone. This is an idea we explore in Appendix B where we optimise V fully before updating π.

- 3.2 Approximation error

As alluded to before, we can interpret V as a value function in conjunction with the policy π. When ﬁxing π we can deﬁne the minimising value function V π = arg minV L(π,V ). This value function, when ﬁxed to V = V π, provides learning signal to policy improvement. Indeed, since (π,V ) can be independently optimised as shown before, we can see that the optimal policy π∗ is obtained by solving the optimisation problem π∗ = arg minπ L(π,V π) = arg minπ arg minV L(π,V ).

However, since in practice the learning of V ≈ V π is approximate, the optimisation of π against such an approximate value function would induce errors. Formally, we consider what happens when π is optimised against a ﬁxed V rather than a well-learned V π. The approximate nature of V induces error in π ≈ π∗, which we characterise below.

Proposition 1. Consider an approximation V (x) of the value function V π = argminV L(π,V ):

V π(x) = Ey∼µ(·|x) r(x,y) − τ log

π(y|x) πref(y|x)

[Figure 26]

and for this function kept ﬁxed, let us optimise the loss π  → L(π,V ) with respect to the policy only. Let πV = arg minπ L(π,V ). Then we have that πV satisﬁes the following equation:

πV (y|x) ∝ πref(y|x)e

1

[Figure 27]

τ r(x,y)−πµV(y(y|x|x))(VπV (x)−V(x)) . (5) In addition we have that for all x,y,

[Figure 28]

log

πV (y|x) π∗(y|x)

[Figure 29]

≤

2 τ

[Figure 30]

max

y

(V π

V

(x) − V (x)) 1 −

πV (y|x) µ(y|x)

[Figure 31]

. (6)

Proof. We delay the full technical proof to the Appendix, see section A. Two remarks are in order here:

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- • Equation (6) says that if our approximation V is close to the value function of πV (in some sense,

the "Bellman residual" V − V π

V of V is small), then πV is close to π∗.

- • In particular, if we consider the best state-independent baseline Vcπ = argminV :V(x)≡c L(π,V ) = Ex∼ρ[V π(x)] (this could be estimated using a large enough

batch size), we see that we would recover the optimal policy π∗ if V π

∗

(x) = V π

∗

c for all x.

- 3.3 Practical Implementation

We consider a parametric policy πθ and a parametric function Vϕ with neural network parameters θ,ϕ respectively. We assume access to data in the form of tuples (xi,yi,ri)1≤i≤n where xi ∼ ρ are prompts, yi ∼ µ(·|xi) are prompt-conditional generations associated with a model policy µ, and ri = r(xi,yi) are scalar reward functions for the individual prompt-completion combination (xi,yi). We perform gradient descent, both on θ and ϕ, to minimise the empirical loss:

n

2

πθ(yi|xi) πref(yi|xi)

- 1

[Figure 36]

- 2

L(θ,ϕ) def=

r(xi,yi) − Vϕ(xi) − τ log

.

[Figure 37]

i=1

We can deﬁne explicitly the gradient w.r.t. ϕ and θ: the gradient w.r.t. the parameter ϕ of the value function is:

n

πθ(yi|xi) πref(yi|xi)

∇ϕVϕ(xi), (7)

Vϕ(xi) − r(xi,yi) + τ log

∇ϕ L(θ,ϕ) =

[Figure 38]

i=1

and the policy gradient:

n

∇θ L(θ,ϕ) = −τ

i=1

∇θ log πθ(yi|xi)(r(xi,yi) − Vϕ(xi))

−

[Figure 39]

[Figure 40]

policy optimisation loss

2

τ 2

πθ(yi|xi) πref(yi|xi)

∇θ log

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

ℓ2-regularisation loss

. (8)

The policy optimisation loss is similar to a standard policy gradients RL algorithm, however it is importantto notice a few key differencesand connectionsbetween the loss above and policy gradient algorithms.

Understanding the value function V . First, when considering a ﬁxed policy π, learning the value function Vϕ correspondsto learning the function V π whose approximationerror we discussed earlier in Section 3.2.

Second, the value function Vϕ(xi) which is subtracted from the reward r(xi,yi) is not simply a baseline used for the purpose of variance reduction as is usually the case in RL [Sutton et al., 2000,

Konda and Tsitsiklis, 1999]. Indeed removing this term (or replacing it by any function V (xi)) would bias our policy gradient estimate. The reason for that being that the samples yi are off-policy, i.e., drawn from µ and not from πθ, thus in general, Ey∼µ(·|x

i)[∇θ log πθ(y|xi)V (xi)] = 0 (whereas if on-policy we would have Ey∼π

θ(·|xi)[∇θ log π(y|xi)V (xi)] = 0). Thus it is important to maintain this value function estimate in the policy gradient.

Ofﬂine regularisation. Another difference compared to a usual regularised PG algorithm is the use of a ℓ2-regularisation loss instead of a KL-regularisation loss. Notice that these two regularisation losses do not lead to equivalent gradients for the reason that the samples yi are off-policy: indeed we have (see, e.g., also [Calandriello et al., 2024, Tang et al., 2024] for discussion of similar theoretical results)

- 1

[Figure 45]

- 2

πθ(y|xi) πref(y|xi)

2

= ∇θKL(πθ(·|xi),πref(·|xi)),

i) ∇θ log

Ey∼µ(·|x

[Figure 46]

unless the sampling is on-policy µ = πθ, where both losses would would lead to the same gradient. Thus we could see the policy gradient update rule Equation (8) as the natural extension of a usual on-policy regularised PG algorithm to the off-policy case.

Policy learning rate rescaling. Finally, in practice, we rescale the policy gradient Equation (8) by multiplying the update by a factor of 1/τ. We found this works better empirically and we hypothe-

sised that the global loss L(θ,ϕ) may be ill-conditioned (as its sensitivities w.r.t. the dimensions θ and ϕ are different, leading to a high condition number). We refer to this algorithm as the Direct Reward Optimisation with Value algorithm, or DRO-V. DRO-V is described in Algorithm 1.

[Figure 47]

Algorithm 1 Direct Reward Optimisation with Value Algorithm (DRO-V)

[Figure 48]

Inputs: A single-trajectory dataset: (xi,yi,ri = r(xi,yi))Ni=1, a parameterised policy: πθ, a reference policy: πref, a parameterised value function: Vϕ, a regularisation scalar τ, a number of total steps K, a batch size B and an optimiser.

for k = 1 to K do Sample uniformly a batch: (xi,yi,ri)Bi=1 Compute gradient updates ∇θ L(θ,ϕ) and ∇ϕ L(θ,ϕ) as in Equation (8) and Equation (7):

1 B

∇θ L(θ,ϕ) = −

[Figure 49]

B

i=1

∇θ log πθ(yi|xi)(r(xi,yi) − Vϕ(xi)) −

πθ(yi|xi) πref(yi|xi)

- 1

[Figure 50]

- 2

∇θ log

[Figure 51]

2

1 B

∇ϕ L(θ,ϕ) =

B

[Figure 52]

i=1

Vϕ(xi) − r(xi,yi) + τ log

πθ(yi|xi) πref(yi|xi)

[Figure 53]

∇ϕVϕ(xi)

Update the policy parameters: θ ← UpdateOptimiser(θ,∇θL(θ,ϕ)) Update the value parameters: ϕ ← UpdateOptimiser(ϕ,∇ϕ L(θ,ϕ))

end for Outputs: πθ

[Figure 54]

Ofﬂine optimisation. The dataset of (prompt, completion, reward) triplets remains static during optimisation. This is because we do not use the current "online" policy, parameterised by the most recent θ parameters, to regenerate completions for a given prompt. In this regard, our optimisation is performed like in ofﬂine reinforcement learning, where taking new actions in the environment is structurally prohibited. We note that our method does not require the training of an additional

reward model. The distinction between ofﬂine and online procedures is important in RL(HF), since they can give rise to different gradients and correspond to very different theoretical justiﬁcations, as studied in e.g. Calandriello et al. [2024], Guo et al. [2024]. Given the additional τ regularisation term appearing in Equation (8), we describe DRO as ofﬂine, regularised reinforcement learning for large language model alignment.

Neural network implementation. It is natural to wonder whether θ and ϕ need be separate or whether parameter sharing can occur for efﬁcient learning. Perhaps counter-intuitively, we found that using two separate networks, one for πθ and one for Vϕ, was beneﬁcial empirically, compared to using policy logits as value outputs. Another implementation decision, this time strictly related to Vϕ, is whether to use a single value per batch or a value per token. We found that using a single value per batch hurts performance. In both instances parameter sharing is detrimental.

As such, by default, we implemented DRO-V using two networks, as well as multiple values across the batch. We will return to these points and ablate these design choices in the Experiments section next.

- 4 Experiments

We now present our empirical results on ﬁnetuning LLMs using DRO-V.

Models and task dataset. In all that follows, we perform our test experiments using the UltraFeedback dataset described in Cui et al. [2023]. We preprocess the data to create an ofﬂine dataset of triplets (prompt, completion, reward) where the prompt might be shared across multiple triplets. We also normalize the dataset such that the rewards have mean 0 and variance 1 across the dataset.

We use T5 large language models [Raffel et al., 2020], a family of auto-regressive transformers with an encoder-decoder architecture, in order to train all our models. The details of the models architecture and software implementation can be found in [Roberts et al., 2022]. Furthermore, our checkpoints are initialised from instruction ﬁnetuning according to the FLAN recipe [Chung et al.,

- 2022]. We denote this initialisation policy (πref as earlier), as the SFT, for supervised ﬁnetuning policy. Depending on the experiment, we use either a large (L) or an extra-large (XL) encoder-decoder model. The large model sports 770M parameters, whereas the XL model has 3B parameters.

Evaluation. Following now standard practice [Zheng et al., 2023], our evaluation pipeline consists in automated evaluation; speciﬁcally, side-by-side comparison. We use the PaLM2 [Anil et al.,

- 2023] LLM as a judge. Given a test set of prompts, for each pair of trained policies, we sample completion responses, and then ask PaLM2 to judge which one is better. The format of the evaluation prompt we use for side-by-side comparison is as follows:

In this task, you will be provided with an instruction and two responses. Your job is to assess the helpfulness and fulfillment of two responses A and B.

Instruction: article response A (left):summary1 response B (right):summary2

For each model, we record a checkpoint every 2,000 training steps, before selecting the best checkpoint across the training curve as determined by side-by-side comparison against the SFT policy. We then use this best checkpoint as representative of an algorithm or set of parameters.

Compute and hyperparameters. For compute, we use version 5 (’v5e’) Tensor Processing Units [TPUs; Jouppi et al., 2023], in the cloud. We train large encoders in conﬁgurations of 4×4 devices and XL encoders in conﬁgurations of 4 × 8 devices. With this computational setup we obtain speeds of around 0.5 training steps per second (21 hours per 40,000 steps) for the large encoders, and 0.1 training steps per second (2 days per 20,000 steps) for the XL encoders. We run our experiments with default learning rate 1e-4 both for the value and the policy networks, and a default total of 40,000 training steps for L models and 20,000 training steps for XL models, using a batch size of 32. The optimiser we use is AdaFactor [Shazeer and Stern, 2018] with a decay rate value of 0.8. For the learning rate, we employ 150 linear warmup steps.

KTO baseline. We compare our algorithm primarily to the strong Kahneman-Tversky Optimization (KTO) baseline of Ethayarajh et al. [2024]. We need a criterion to decide which sample completions are desirable (’thumbs up’) or not. We choose to do so per-batch, based on whether each attached scalar reward is greater or less than the average reward seen over the minibatch. We use

the same KL divergence estimator on the batch as they do, and do not weigh the loss in any manner, since we are already compensating for batch statistics by computing the batchwise average reward. For both algorithms KTO and DRO-V, the τ regularisation factor (denoted by β in Ethayarajh et al. [2024]) is held constant throughout training.

4.1 Results and Ablations

Here we show our empirical results as well as perform ablations and study the impact of hyperparameters on performance.

Empirical results. DRO-V outperforms KTO in side-by-side comparison, both for the T5-L and XL encoders. In the interest of fair comparison, we optimise our baseline as much as possible and present the best KTO results we obtained after search over three values of the τ regularisation parameter, τ ∈ {0.1,1.0,5.0} (τ = 5.0 being optimal). By contrast we show our DRO-V results without such search, simply setting τ to a default intuitive value of 1.0. In this section only, we give standard deviation estimates, thanks to computing averages of scores obtained by side-by-side comparison over 5 evaluation folds of 1,000 test prompts each:

[Figure 55]

Side-by-side Winrate (ﬁrst over second) DRO-V vs SFT: 78.9% ± 0.3%

[Figure 56]

[Figure 57]

[Figure 58]

KTO vs SFT: 67.5% ± 0.7% DRO-V vs KTO: 63.4% ± 1.0%

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Figure 1: Winrates with T5-L encoders.

[Figure 63]

Side-by-side Winrate (ﬁrst over second) DRO-V vs SFT: 81.5% ± 1.0%

[Figure 64]

[Figure 65]

[Figure 66]

KTO vs SFT: 78.2% ± 0.7% DRO-V vs KTO: 57.5% ± 0.8%

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Figure 2: Winrates with T5-XL encoders.

We see signiﬁcant empirical outperformance of DRO-V. Both on T5-L and T5-XL encoders, DRO-V clearly outperforms the SFT (78.9% and 81.5%, respectively). Most importantly, DRO-V also wins over the KTO baseline in direct comparison (63.4% and 57.5% respectively). We also give qualitative examples of this difference, with sample completions from both algorithms, in Appendix E where we demonstrate how these numbers correspond to more helpful and focused prompt completions.

We now study ablations over hyperparameters and architecture. In all that follows, we retain the same experimental protocol as above, and here use T5 large encoders exclusively.

Impact of the learning rate for policy and value. We begin by examining the impact of the learning rate on downstream performance. We study two cases: ﬁrst varying jointly the learning rate of the policy and the value network, and second, switching the learning rate of the value network only. We pick our learning rates to be one of 1e-5, 5e-5, 5e-4, or the default 1e-4. Results are presented in Figure 3. For the value only sweep, we observe that changing the value learning rate parameter alone yields a small impact. Thus the policy learning rate rescaling factor, 1/τ (Equation 8) is all the more important. We do also note a small yet monotonic improvement in learning the value V faster than the policy. Overall, the performance of DRO-V remains very stable within an order of magnitude change for learning rates.

[Figure 71]

DRO-V learning rate value 1e−5 5e-5 1e-4 5e-4 Joint LR : Winrate vs SFT 73.7% 78.7% 78.9% 78.4%

[Figure 72]

[Figure 73]

[Figure 74]

LR, value V only : Winrate vs SFT 76.8% 78.0% 78.9% 79.1%

[Figure 75]

[Figure 76]

Figure 3: Top line: Impact of varying jointly the π and V learning rate parameter. Bottom line: Impact of value function V learning rate parameter only.

Impact of regularisation parameter τ. Similarly, we vary the strength of regularisation parameter τ both for DRO-V and KTO. We pick between three values: 0.1, 1.0 and 5.0. Results are presented in Figure 4. We see substantial variation in performance due to this parameter. A regularisation parameter of 1.0, an intuitive value, is actually best for DRO-V. On the other hand, we found it far from optimal for KTO and picked the best value, 5.0, instead. These experiments were performed using T5-L encoders, and we re-used those τ choices for T5-XL experiments as well.

[Figure 77]

[Figure 78]

KTO τ 0.1 1.0 5.0

[Figure 79]

[Figure 80]

DRO-V τ 0.1 1.0 5.0 Winrate vs SFT 70.5% 78.9% 76.6%

[Figure 81]

[Figure 82]

Winrate vs SFT 63.5% 61.9% 67.5% Figure 4: Left: Impact of τ parameter on DRO-V. Right: Impact of τ parameter on KTO.

[Figure 83]

[Figure 84]

Impact of parameter sharing. Finally, we investigate the quantitative impact of parameter sharing for DRO-V, as exposed in Section 3.3. We jointly study the performance of the single or double network version, as well as whether to use a single value number per batch or not. For computational reasons, these experiments at performed using 10,000 steps of training only (and therefore slightly undertrained compared to our main T5-L 78.9% result). Results are presented Figure 5, with the full comparison matrix in Appendix C. The impact of full parameter sharing is material, with most of the hit coming from the single or double network choice. However, we also observe that when using two networks for π and V , there are signiﬁcant gains in not using a single value per batch (76.6% against 72.1% winrate vs SFT, a difference of 4.5%, conﬁrmed in direct side-by-side comparison of those two variants yielding 54.9% in favour of the multiple value version). These observations help explain some of the outperformance of our method.

[Figure 85]

Double Net Multiple Values Winrate vs SFT 72.1% 57.6% 55.5% 76.6%

Double Net Single Value

Single Net Single Value

Single Net Multiple Values

[Figure 86]

Parameter Sharing Variant

[Figure 87]

[Figure 88]

[Figure 89]

Figure 5: Parameter sharing variants. DRO-V Winrate vs SFT.

- 5 Related work

Human Feedback in Reinforcement Learning. Integrating human feedback into reinforcement learning, as introduced by Christiano et al. [2017], has rapidly grown to be considered essential for improving the practical utility of LLMs and mitigating their epistemic risk [Hannigan et al., 2024]. RLHF may not always yield direct improvements in benchmark performance [Touvron et al., 2023], but it signiﬁcantly enhances human-centric applications like dialogue systems [Nakano et al., 2021, Ouyang et al., 2022] and extends to non-human-centric tasks too like MuJoCo physics [Yuan et al.,

- 2024b] and robotics [Gao et al., 2024]. Precursors to RLHF were implemented with Deep Q Networks [Mnih et al., 2013] and Actor-Critic algorithms [Mnih et al., 2016, Glaese et al., 2022], and RLHF itself was debuted with proximal policy optimisation [Schulman et al., 2017]. Given the complexity of RLHF techniques and algorithms used [Casper et al., 2023], the research community is revealing surprising beneﬁts of simpler strategies like REINFORCE [Ahmadian et al., 2024], sequence likelihood [Zhao et al., 2023a], and ranking approaches [Dong et al., 2023, Yuan et al., 2023].

Advancements in Policy Optimisation. Assuming a Bradley-Terry model [Bradley and Terry, 1952] for human reward modeling enables the RLHF problem to be reformulated as a supervised learning task [Rafailov et al., 2023]; this may provide greater training stability [Zhao et al., 2023b] and data efﬁciency, although this point is debated [Xu et al., 2024]. Recent research has focused on enhancing the direct preference optimization (DPO) methods for scalability [Tunstall et al.,

- 2023, Ivison et al., 2023] and safety [Liu et al., 2024] while expanding their mathematical underpinnings [Azar et al., 2023, Wang et al., 2023, Ji et al., 2024]. Other work constrains DPO’s contextual focus [Zeng et al., 2024] or explores alternatives to preference models such as Nash equilibria [Munos et al., 2023, Rosset et al., 2024]. Despite these advancements, both DPO and RLHF encounter challenges such as reward hacking [Pang et al., 2022, Skalse et al., 2022], length bias [Park et al., 2024], and overoptimisation [Amodei et al., 2016, Pan et al., 2022], which can in turn lead to under-regularised models [Gao et al., 2022, Singhal et al., 2023] and objective mismatch Kirk et al. [2023]. While techniques like ensembling can alleviate issues of alignment [Eisenstein et al., 2023, Ramé et al., 2024] and overoptimisation[Wortsman et al., 2022, Coste et al., 2023], they may be overlooked due to their computational intensity [Zhang et al., 2024].

Different data types. RLHF classically focuses on a choice that human raters make between two outputs produced from a single input to the policy. Collecting this data type can be costly, motivating

the use of cheaper data, such as upvote/downvote point-wise data [Ethayarajh et al., 2024], or joint preferences from different questions [Bansal et al., 2024]. Such data can also be noisy, motivating factual augmentation [Sun et al., 2023], ﬁne-grained responses [Wu et al., 2023].

Online vs. Ofﬂine Learning and Alignment through Self-Play. The practical distinction between online and ofﬂine methods [Jaques et al., 2019] appears more relevant than the reinforcement versus supervised learning dichotomy. Online policies risk deviating from the original data distribution, causing shifts and potential issues [Zhuang and Hadﬁeld-Menell, 2020, Shin et al., 2023]. Alternatively, alignment can be achieved through self-play in a two-player game framework [Munos et al., 2023, Swamy et al., 2024], encompassing both online and ofﬂine settings and allowing smooth transitions between them. Similarly, iterative DPO has demonstrated alignment improvement [Yuan et al., 2024a], building upon established techniques like reinforcement learning from AI feedback [Bai et al., 2022b, Lee et al., 2023].

- 6 Conclusion and limitations

We have introduced DRO, a new framework for aligning LLMs in the setting of single-trajectory datasets, where for each prompt a single completion associated to a scalar reward is available. DRO moves away from the traditional preference setting in RLHF. In doing so it not only makes the training of an explicit reward model redundant, but also and most importantly enables leveraging potentially orders of magnitude more data coming from user feedback instead of raters’. DRO is theoretically principled since it learns both a policy and a value. Thus it does not rely on any simplifying mathematical assumptions, such as a uniform value function on the batch, or the cancellation of the partition function that typically underpins RLHF methods. Using T5 large and XL text encoders, we have shown that these properties translate into strong performance on the UltraFeedback dataset when compared to Kahneman-Tversky optimisation. However, our empirical study is limited, both in terms of number of tasks and scale. Further work is required to more broadly establish the performance gains that our approach provides when considering the largest language models, as it is able to leverage large amounts of user-generated data.

Acknowledgements

We would like to thank the whole Google DeepMind team for providing the infrastructure to make this work possible. In particular, we would like to thank Matt W. Hoffman, Bobak Shahriari, Nikola Momchev, Sertan Girgin and Piotr Stanczyk for their support in building the coding infrastructure and Doina Precup and Olivier Bachem for their guidance and continual support.

References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arxiv preprint arXiv:2402.14740, 2024.

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv, 2016.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. PaLM 2 technical report, 2023.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv, 2023.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatﬁeld-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcementlearning from human feedback. arXiv, 2022a.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, John Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, E Perez, Jamie Kerr, Jared Mueller, Jeff Ladish, J Landau, Kamal Ndousse, Kamile˙ Lukoiut¯ e,˙ Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noem’i Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, T. J. Henighan, Tristan Hume, Sam Bowman, Zac Hatﬁeld-Dodds, Benjamin Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom B. Brown, and Jared Kaplan. Constitutional AI: Harmlessness from AI feedback. arXiv, 2022b.

Hritik Bansal, Ashima Suvarna, Gantavya Bhatt, Nanyun Peng, Kai-Wei Chang, and Aditya Grover. Comparing bad apples to good oranges: Aligning large language models via joint preference optimization. arxiv preprint arXiv: 2404.00530, 2024.

Heinz H. Bauschke and Patrick L. Combettes. Convex Analysis and Monotone Operator Theory in Hilbert Spaces. Springer, 2011.

Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile˙ Lukošiut¯ e,˙ Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable oversight for large language models. arXiv preprint arXiv:2211.03540, 2022.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Daniele Calandriello, Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, Rishabh Joshi, Zeyu Zheng, and Bilal Piot. Human alignment of large language models through online preference optimisation. arxiv preprint arXiv:2403.08635, 2024.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arxiv preprint arXiv:2002.05709, 2020.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, 2017.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-ﬁnetuned language models, 2022.

Thomas Coste, Usman Anwar, Robert Kirk, and David Scott Krueger. Reward model ensembles help mitigate overoptimization. arXiv, 2023.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377, 2023. URL https://github.com/OpenBMB/UltraFeedback. MIT license.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arxiv preprint arXiv:1810.04805, 2019.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and T. Zhang. RAFT: Reward rAnked FineTuning for generative foundation model alignment. arXiv, 2023.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen R Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. Helping or herding? Rward model ensembles mitigate but do not eliminate reward hacking. arXiv, 2023.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arxiv preprint arXiv:2402.01306, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Proceedings of the International Conference on Machine Learning, 2022.

Zhaolin Gao, Jonathan D. Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J. Andrew Bagnell, Jason D. Lee, and Wen Sun. Rebel: Reinforcement learning via regressing relative rewards. arxiv preprint arXiv:2404.16767, 2024.

Amelia Glaese, Nat McAleese, Maja Trebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, Lucy Campbell-Gillingham, Jonathan Uesato, Po-Sen Huang, Ramona Comanescu, Fan Yang, Abigail See, Sumanth Dathathri, Rory Greig, Charlie Chen, Doug Fritz, Jaume Sanchez Elias, Richard Green, Sonaˇ Mokrá, Nicholas Fernando, Boxi Wu, Rachel Foley, Susannah Young, Iason Gabriel, William Isaac, John Mellor, Demis Hassabis, Koray Kavukcuoglu, Lisa Anne Hendricks, and Geoffrey Irving. Improving alignment of dialogue agents via targeted human judgements. arXiv, 2022.

Shane Grifﬁth, Kaushik Subramanian, Jonathan Scholz, Charles L Isbell, and Andrea L Thomaz. Policy shaping: Integrating human feedback with reinforcement learning. In Advances in Neural Information Processing Systems, 2013.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In Advances in Neural Information Processing Systems, 2020.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792, 2024.

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. arxiv preprint arXiv:1702.08165, 2017.

Timothy R. Hannigan, Ian P. McCarthy, and André Spicer. Beware of botshit: How to manage the epistemic risks of generative chatbots. Business Horizons, 2024. ISSN 0007-6813. doi: https://doi.org/10.1016/j.bushor.2024.03.001. URL https://www.sciencedirect.com/science/article/pii/S0007681324000272.

Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, and Hanna Hajishirzi. Camels in a changing climate: Enhancing LM adaptation with Tulu 2. arXiv, 2023.

Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Way off-policy batch deep reinforcement learning of implicit human preferences in dialog. arXiv, 2019.

Haozhe Ji, Cheng Lu, Yilin Niu, Pei Ke, Hongning Wang, Jun Zhu, Jie Tang, and Minlie Huang. Towards efﬁcient and exact optimization of language model alignment. arxiv preprint arXiv:2402.00856, 2024.

Norman P. Jouppi, George Kurian, Sheng Li, Peter C. Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, Cliff Young, Xiaoping Zhou, Zongwei Zhou, and David A. Patterson. TPU v4: An optically reconﬁgurable supercomputer for machine learning with hardware support for embeddings. In Proceedings of the Annual International Symposium on Computer Architecture, 2023.

D. Kahneman and A. Tversky. Prospect theory: An analysis of decision under risk. Econometrica, 47(2):263–292, 1979.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of rlhf on llm generalisation and diversity. arXiv preprint arXiv:2310.06452, 2023.

W Bradley Knox and Peter Stone. TAMER: Training an agent manually via evaluative reinforcement. In Proceedings of the IEEE International Conference on Development and Learning, 2008.

Vijay Konda and John Tsitsiklis. Actor-critic algorithms. Advances in neural information processing systems, 12, 1999.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. RLAIF: Scaling reinforcement learning from human feedback with AI feedback. arXiv, 2023.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Ofﬂine reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Zixuan Liu, Xiaolin Sun, and Zizhan Zheng. Enhancing llm safety via constrained direct preference optimization, 2024.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller. Playing atari with deep reinforcement learning. CoRR, abs/1312.5602, 2013.

VolodymyrMnih, Adrià PuigdomènechBadia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2016.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, Marco Selvi, Sertan Girgin, Nikola Momchev, Olivier Bachem, Daniel J. Mankowitz, Doina Precup, and Bilal Piot. Nash learning from human feedback. arXiv, 2023.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. WebGPT: Browser-assisted question-answering with human feedback. arXiv, 2021.

OpenAI. Introducing ChatGPT, 2022. URL https://openai.com/blog/chatgpt.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. arXiv, 2022.

Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspeciﬁcation: Mapping and mitigating misaligned models. arXiv, abs/2201.03544, 2022.

Richard Yuanzhe Pang, Vishakh Padmakumar, Thibault Sellam, Ankur P. Parikh, and He He. Reward gaming in conditional text generation. In Annual Meeting of the Association for Computational Linguistics, 2022.

Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling length from quality in direct preference optimization, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. arxiv preprint arXiv:2103.00020, 2021.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a uniﬁed text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.

Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. WARM: On the beneﬁts of weight averaged reward models. arXiv, 2024.

Pierre H. Richemond and Brendan Maginnis. A short variational proof of equivalence between policy gradients and soft q learning. arxiv preprint arXiv:1712.08650, 2017.

Adam Roberts, Hyung Won Chung, Anselm Levskaya, Gaurav Mishra, James Bradbury, Daniel Andor, Sharan Narang, Brian Lester, Colin Gaffney, Afroz Mohiuddin, Curtis Hawthorne, Aitor Lewkowycz, Alex Salcianu, Marc van Zee, Jacob Austin, Sebastian Goodman, Livio Baldini Soares, Haitang Hu, Sasha Tsvyashchenko, Aakanksha Chowdhery, Jasmijn Bastings, Jannis Bulian, Xavier Garcia, Jianmo Ni, Andrew Chen, Kathleen Kenealy, Jonathan H. Clark, Stephan Lee, Dan Garrette, James Lee-Thorp, Colin Raffel, Noam Shazeer, Marvin Ritter, Maarten Bosma, Alexandre Passos, Jeremy Maitin-Shepard, Noah Fiedel, Mark Omernick, Brennan Saeta, Ryan Sepassi, Alexander Spiridonov, Joshua Newlan, and Andrea Gesmundo. Scaling up models and data with t5x and seqio. arXiv, 2022. URL https://github.com/google-research/t5x. Apache-2.0 license.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences, 2024.

William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv, 2017.

John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft qlearning. arxiv preprint arXiv:1704.06440, 2018.

Noam M. Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. arXiv, 2018.

Daniel Shin, Anca D. Dragan, and Daniel S. Brown. Benchmarks and algorithms for ofﬂine preference-based reward learning. arXiv, 2023.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. ArXiv, abs/2310.03716, 2023.

Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Deﬁning and characterizing reward gaming. In Neural Information Processing Systems, 2022.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented rlhf, 2023.

R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems 12, pages 1057–1063. MIT Press, 2000.

Gokul Swamy, Christoph Dann, Rahul Kidambi, Zhiwei Steven Wu, and Alekh Agarwal. A minimaximalist approach to reinforcement learning from human feedback. arXiv, 2024.

Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Rémi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Ávila Pires, and Bilal Piot. Generalized preference optimization: A uniﬁed approach to ofﬂine alignment. arXiv preprint arXiv:2402.05749, 2024.

Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov,

Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and ﬁne-tuned chat models. arXiv, 2023.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of LM alignment. arXiv, 2023.

Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, and Yuxin Chen. Beyond reverse KL: Generalizing direct preference optimization with diverse divergence constraints. arXiv, 2023.

Garrett Warnell, Nicholas Waytowich, Vernon Lawhern, and Peter Stone. Deep TAMER: Interactive agent shaping in high-dimensional state spaces. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 2018.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple ﬁne-tuned models improves accuracy without increasing inference time. In Proceedings of the International Conference on Machine Learning, 2022.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training, 2023.

Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, and Yi Wu. Is dpo superior to ppo for llm alignment? a comprehensive study. arxiv preprint arXiv:2404.10719, 2024.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models, 2024a.

Yifu Yuan, Jianye Hao, Yi Ma, Zibin Dong, Hebin Liang, Jinyi Liu, Zhixin Feng, Kai Zhao, and Yan Zheng. Uni-rlhf: Universal platform and benchmark suite for reinforcement learning with diverse human feedback, 2024b.

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Feiran Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv, abs/2304.05302, 2023.

Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Token-level direct preference optimization, 2024.

Shun Zhang, Zhenfang Chen, Sunli Chen, Yikang Shen, Zhiqing Sun, and Chuang Gan. Improving reinforcement learning from human feedback with efﬁcient reward model ensemble, 2024.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J. Liu. SLiCHF: Sequence likelihood calibration with human feedback. arXiv, 2023a.

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839, 2023b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.

Simon Zhuang and Dylan Hadﬁeld-Menell. Consequences of misaligned AI. In Advances in Neural Information Processing Systems, 2020.

Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pages 1433–1438. Chicago, IL, USA, 2008.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2020.

APPENDICES

- A Mathematical Proofs

We recall Proposition 1 and proceed to prove it here: Proposition. Consider an approximation V (x) of the value function V π = argminV L(π,V ):

π(y|x) πref(y|x)

V π(x) = Ey∼µ(·|x) r(x,y) − τ log

[Figure 90]

and for this function kept ﬁxed, let us optimise the loss π  → L(π,V ) with respect to the policy only. Let πV = arg minπ L(π,V ). Then we have that πV satisﬁes the following equation:

τ r(x,y)−πµV(y(y|x|x))(VπV (x)−V(x)) . (9) In addition we have that for all x,y,

1

[Figure 91]

[Figure 92]

πV (y|x) ∝ πref(y|x)e

πV (y|x) π∗(y|x)

log

[Figure 93]

≤

2 τ

(V π

max

[Figure 94]

y

πV (y|x) µ(y|x)

(x) − V (x)) 1 −

V

[Figure 95]

. (10)

Proof. We proceed to prove this here. Consider the Lagrangian G(π) = L(π,V ) +

- x ρ(x)λx( y π(y|x) − 1) where λx are the Lagrange multipliers corresponding to the constraint
- y π(y|x) = 1. Notice there is no need to impose such constraint outside of the support of ρ. The

derivative of G(π) w.r.t. a variable π(y|x) is

π(y′|x′) πref(y′|x′)

π(y′|x′) πref(y′|x′)

∂π(y|x)G(π) = Ex′∼ρ,y′∼µ(·|x′) r(x′,y′) − τ log

− V (x′) ∂π(y|x) −τ log

[Figure 96]

[Figure 97]

 

 

ρ(x′)λx′

π(y′|x′)

+∂π(y|x)

x′

y′

τ π(y|x)

π(y|x) πref(y|x)

− λx .

− V (x)

= −ρ(x) µ(y|x) r(x,y) − τ log

[Figure 98]

[Figure 99]

Setting the optimality conditions ∂π(y|x)G(π) = 0 for all x and y, we deduce that for any x in the support of ρ, we have, for any y,

π(y|x) πref(y|x)

µ(y|x) π(y|x)

r(x,y) − τ log

− V (x)

λx = τ

[Figure 100]

[Figure 101]

µ(y|x) π(y|x)

π(y|x) πref(y|x)

= τ

π(y|x)

r(x,y) − τ log

− V (x)

[Figure 102]

[Figure 103]

y

π(y|x) πref(y|x)

− V (x)

= τ

µ(y|x) r(x,y) − τ log

[Figure 104]

y

= τ (V π(x) − V (x)) . Thus

µ(y|x) π(y|x)

π(y|x) πref(y|x)

− V (x) = V π(x) − V (x), or, equivalently:

r(x,y) − τ log

[Figure 105]

[Figure 106]

π(y|x) µ(y|x)

π(y|x) πref(y|x)

(V π(x) − V (x)) , (11) from which we deduce that the optimal policy satisﬁes

= r(x,y) − V (x) −

τ log

[Figure 107]

[Figure 108]

[r(x,y)−πµ((yy||xx))(Vπ(x)−V(x))−V(x)] (12) ∝ πref(y|x)e

1 τ

π(y|x) = πref(y|x)e

[Figure 109]

[Figure 110]

[r(x,y)−πµ((yy||xx))(Vπ(x)−V(x))] (13)

1 τ

[Figure 111]

[Figure 112]

which proves the ﬁrst claim Equation (9). Now from the property of the optimal policy π∗, and from Equation (11), we deduce that

π(y|x) µ(y|x)

π(y|x) π∗(y|x)

= V ∗(x) − V (x) −

τ log

[Figure 113]

[Figure 114]

Taking the expectation w.r.t. µ(·|x) on both sides, we get

(V π(x) − V (x)) . (14)

τ

y

π(y|x) π∗(y|x)

= V ∗(x) − V π(x).

µ(y|x)log

[Figure 115]

Plugging back V ∗(x) into Equation (14) we have for any x,y,

π(y|x) π∗(y|x)

τ log

[Figure 116]

= τ

= τ

y′

y′

π(y′|x) π∗(y′|x)

π(y|x) µ(y|x)

µ(y′|x)log

+ V π(x) − V (x) −

[Figure 117]

[Figure 118]

π(y′|x) π∗(y′|x)

µ(y′|x)log

+ (V π(x) − V (x)) 1 −

[Figure 119]

(V π(x) − V (x))

π(y|x) µ(y|x)

[Figure 120]

.

Let us write ǫ(x) def= τ1 maxy (V π(x) − V (x)) 1 − πµ((yy||xx)) and ρ(y|x) def= log ππ∗((yy||xx)). We have that

[Figure 121]

[Figure 122]

[Figure 123]

1 =

y

π∗(y|x)eρ(y|x)

π(y|x) =

y

π∗(y|x)e y′ µ(y′|x)ρ(y′|x)+ǫ(x)

≤

y

= e y′ µ(y′|x)ρ(y′|x)+ǫ(x),

thus y′ µ(y′|x)ρ(y′|x) ≥ −ǫ(x). Similarly we can upper bound y′ µ(y′|x)ρ(y′|x) by ǫ(x). We deduce that for all y

|ρ(y|x)| ≤

y′

µ(y′|x)ρ(y′|x) + ǫ(x)

≤ 2ǫ(x), which concludes the proof of the second claim.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- B A policy-only algorithmic variant

To reduce the computational burden incurred by DRO-V, where we use typically use two different neural networks for π and V , we can also deﬁne a loss function over policy π alone. To this end, and as per Section 3.2, we can deﬁne V π = argminV L(π,V ) as the best value function given a ﬁxed policy π. Writing down the optimality condition we recall that, as per Proposition 1:

π(y|x) πref(y|x)

V π(x) = Ey∼µ r(x,y) − τ log

[Figure 128]

.

From this we deduce and deﬁne the loss over policy only, L(π): L(π) def= L(π,V π)

=

π(y′|x) πref(y′|x)

π(y|x) πref(y|x)

- 1

[Figure 129]

- 2

− Ey′∼µ(·|x) r(x,y′) − τ log

Ex∼ρ,y∼µ(·|x) r(x,y) − τ log

[Figure 130]

[Figure 131]

=

π(y|x) πref(y|x)

- 1

[Figure 132]

- 2

Ex∼ρ Vary∼µ(·|x) r(x,y) − τ log

[Figure 133]

.

2

We notice that operationalising this loss requires evaluating an empirical estimate for the variance of a residual term, Vary∼µ(·|x), over completions y given a prompt x. Since the variance cannot be estimated from a single sample y ∼ π(·|x), we cannot practically implement the loss function in an unbiased way. However, if we allow for two samples estimation, i.e., y1,y2 ∼ π(·|x), this will deviate from the single trajectory setting but closely connect to the above loss to the pairwise preference loss. Most notably, we can rewrite

- 1

[Figure 134]

- 2

L(π) =

π(y1|x) πref(y1|x)

i∼π(·|x),x∼ρ r(x,y2) − r(x,y1) + τ log

Ey

[Figure 135]

π(y2|x) πref(y2|x)

− τ log

[Figure 136]

2

.

Note that the reward difference r(x,y2) − r(x,y1) can be understood as a form of preference of y2 over y1. Indeed, when replacing the difference by a preference indicator, the above loss simply reduces to the IPO loss [Azar et al., 2023].

However, as we are placed in the single trajectory setting rather than the preference setting (where a 2-sample estimate of the variance would be available for each prompt), we resort instead to a tractable approximation for L(π). We estimate the variance term Vary∼µ(·|x)(·) as variance on each mini-batch B instead, where each element has a different x. Thus we replace this term with empirical estimate Vary∼B, involving computation of the empirical variance of log-policy ratios along the batch axis. This gives us a tractable, approximate policy-only variant for DRO, where the loss is parameterised by θ only:

LDRO-P(θ) def=

- 1

[Figure 137]

- 2

πθ(y|x) πref(y|x)

Ex∼ρ Vary∼B r(x,y) − τ log

[Figure 138]

(15)

We present the associated Algorithm 2, that we will call DRO-P, below. Algorithm 2 DRO Policy Only Optimisation : DRO-P

[Figure 139]

[Figure 140]

Inputs: A single-trajectory dataset: (xi,yi,ri = r(xi,yi))Ni=1, a parameterised policy: πθ, a reference policy: πref, a regularisation scalar τ, a number of total steps K, a batch size B and an optimiser.

for k = 1 to K do

Sample uniformly a batch: (xi,yi,ri)Bi=1 Compute gradient updates ∇θLDRO-P(θ) for the sampled loss deﬁned by Equation (15):

1 B

∇θLDRO-P(θ) = −

[Figure 141]

B

i=1

πθ(yi|xi) πref(yi|xi)

∇θ log πθ(yi|xi)r(xi,yi) − τ∇θVary∼B log

[Figure 142]

Update the policy parameters: θ ← UpdateOptimiser(θ,∇θLDRO-P(θ)) end for Outputs: πθ

[Figure 143]

DRO-P appears similar to a standard policy gradients RL algorithm, featuring an additional variance regulariser computed on the batch. We found that DRO-P in practice does not collapse and produces non-trivial resulting policies, that can in fact beat the SFT policy in side-by-side comparison on our evaluation setting (T5-encoder on UltraFeedback task, as in Experiments section 4). Nonetheless, this algorithm, while being simpler and more parsimonious than DRO-V, is also less competitive. This speaks to the necessity to use both a policy and a value function - in a theoretically sound and principled way - in order to achieve the best empirical results, and is why we emphasised DRO-V in our presentation. Finally, this result also shows that DRO is a framework that can be instantiated in a variety of algorithmic ways.

- C Additional Empirical Results C.1 Parameter sharing ablation

Here we provide the full results of a pairwise side-by-side comparison involving all architectural modiﬁcations, namely either using a single neural network or two networks for π and V , and whether to use a state-independent function V or not (a single or multiple value number per batch). Results are presented below Figure 6, complementing those from Figure 5 in the main text, which only presented comparison against the SFT.

[Figure 144]

Parameter Sharing Variant

[Figure 145]

Double Net Single Value

Single Net Single Value

Single Net Multiple Values

Double Net Multiple Values

SFT Double net single value 50 0% 37 1% 28 9% 54 9% 27 9%

[Figure 146]

[Figure 147]

Single net single value 62 9% 50 0% 45 2% 70 4% 42 4% Single net multiple value 71 1% 54 8% 50 0% 74 7% 44 5% Double net multiple value 45 1% 29 6% 25 3% 50 0% 23 4%

[Figure 148]

[Figure 149]

[Figure 150]

SFT 72 1% 57 6% 55 5% 76 6% 50 0%

[Figure 151]

[Figure 152]

Figure 6: Parameter sharing variants. DRO-V variants’ side-by-side winrate.

- D Discussion on the cancellation of partition function

We provide additional discussion on the cancellation of partition function in the pairwise preference optimisation algorithms. Starting with the regularised policy optimisation setting in Equation (1), we can write

τ log

π∗(y|x) πref(y|x)

[Figure 153]

− V (x) = r(x,y).

The derivation of pairwise preference optimisation algorithms typically casts the ofﬂine policy optimisation problem into a reward modeling problem. That is, one can parameterise a reward function r using the equation above, and plug into a reward modeling loss.

Many of the pairwise preference optimisation algorithms make use of the paired data (x,yl,yw) where given a prompt x, the completion yw is preferred over yl by human raters. The loss function takes the form of taking the reward difference between the two completions, resulting in

r(x,yw) − r(x,yl) = τ log

π∗(yw|x) πref(yw|x)

[Figure 154]

− τ log

π∗(yl|x) πref(yl|x)

[Figure 155]

which cancels out the partition function or the value function V (x). Hence, in the pairwise preference optimisation setting, the partition function V is implicit in the derivation and we do not need to parameterise or learn it explicitly. This is unlike the single trajectory setting, where the partition function does not cancel out and hence needs to be learned explicitly.

- E Qualitative study of prompt completions

Here we show some example side-by-side prompt completions for our best DRO-V and KTO checkpoints (as evaluated in Section 4), using T5-XL text encoders. We show those under the format ’Prompt - Response 1 ( DRO-V ) - Response 2 (KTO)’. Overall, DRO-V responses appear slightly more helpful and focused, and are generally preferred by the Palm2 LLM judge. We found that it’s often a combination of helpfulness and concision, rather than just one of these dimensions, that tilts the decision one way or another. In order to be as representative as possible, we try to illustrate both some clear preference cases, as well as others that the reader might ﬁnd more subtle.

[Figure 156]

Prompt: List the components of an Ethernet cable.

[Figure 157]

- Response 1: [’Ethernet cables are made up of several components, including conductors (conductors), wires (coaxial cables), and connectors. The components of an Ethernet cable are designed to transmit data and signals from a computer over a network. The following are the components of an Ethernet cable: 1. Conductors: Conductors are the core of the Ethernet cable. They are made up of copper wires in pairs, joined together by a metallic strip. The wires in the Ethernet cable form a network. 2. Wires: These wires are the insulators, or inner shield, of the Ethernet cable. They form the outer layer of the cable, preventing the signals from being transmitted and prevent electromagnetic interference from the network. 3. Connectors: Connectors are the wires that carry data and signals from the Ethernet cable to the network. They are made up of copper and non-conductive metal wires. The types of Ethernet cables available include Cat 5, Cat 7, and Category 5.’]
- Response 2: [’Ethernet cables are made up of several components, including conductors (conductors), wires (modems), and casings to shield and protect them from the elements. They use twisted pairs to join them together, which are insulated with copper. This creates a sturdy and flexible cable that has a low impedance, making it ideal for network connections. Ethernet cables can also include other components, such as switches and hubs, but such items are uncommon in Ethernet-based networks. For more information on Ethernet, refer to "The History of Computer Networks."’]

[Figure 158]

Prompt: Q: You are given a sentence and a question in the input. The information provided in the sentence should be enough to answer the question. You’re expected to write the correct answer. Do not use any facts other than those provided in the sentence. Sentence: They were Dig the dog, Mittens the rooster, and Bub the duck. Question: Who made the noise that caused the sock monster to look at the friends? A:

[Figure 159]

- Response 1: [’Dig the dog, Mittens the rooster, and Bub the duck.’]
- Response 2: [’Dig, Mittens, and Bub the duck.’]

[Figure 160]

Prompt: You will be given a definition of a task first, then an example. Follow the example to solve a new instance of the task. You are given two sentences. You have to find if there is entailment or agreement of the Hypothesis by the Premise. From the given pair of sentences, you should identify if there is enough information in the Premise to support the claim made in the Hypothesis. The Premise may not exactly be the same as Hypothesis. Your task is to return ’entails’ if the premise supports hypothesis else return ’neutral’. Premise: Lyme Disease is caused by a bacterium that’s transmitted by tick bite, but many infected people don’t remember a bite. Hypothesis: Lyme disease is caused by bacteria. Solution: entails Why? The premise sentence agrees with the hypothesis that Lyme Disease is a bacterium. The premise also gives additional information about Lyme disease. Therefore Premise entails Hypothesis. New input: Premise: Fronts are boundaries between air masses of different temperatures. Hypothesis: In weather terms, the boundary between two air masses is called front. Solution:

[Figure 161]

- Response 1: [’entails Why? The premise states that fronts are boundaries between air masses of different temperatures which is a clear statement of the hypothesis. The premise also gives additional information about fronts by adding that they are boundaries between air masses of different temperatures. Therefore, the premise entails the hypothesis.’]
- Response 2: [’entails’]

[Figure 162]

Prompt: What are some common misconceptions about the value of a college degree in today’s job market?

[Figure 163]

- Response 1: [’Common misconceptions about the value of a college degree include:

- 1. College degrees are irrelevant to the job market: Students with low-skill job experience, no college degree, or with limited skills in a desired field of study might still be able to get a competitive job upon graduation and receive a paycheck. This might lead to a perception among some that college degrees have little value in todayś job market. 2. College degrees are required for all jobs: Many employers believe that college education is a basic requirement for finding jobs. However, this is not the case worldwide and some sectors may require additional skills, such as computer programming or engineering, which may not be covered by the typical four-year college degree. 3. College degrees are insignificant: Many believe degrees in science, mathematics, engineering, and other related fields are more important in todayś job market. These degrees may increase the marketability of individuals, but they may not lead to a higher salary. 4. College degrees are merely for the young: According to a recent study, college graduates are more likely than the general population to be unemployed or underemployed within a decade. This trend is primarily due to the rapid aging of the baby boomer generation, who has not yet entered the workforce in the traditional sense. 5. College degrees are not worth it: In some industries, especially those that compete with big business, a bachelorś degree or higher is often enough to get a job. This may be true for entry-level positions, but for senior positions, a masterś or doctoral degree may be better. 6. College degrees for older people: According to some surveys, some 65-69 percent of people over the age of 55 believe they might need a college degree to find a job, even if the job market is not favorable for their experience. This view is aimed at the older generation and the perception of college degrees as just another piece of the puzzle they must "educate" before being able to get hired. By understanding the current job market and what employers value, it is possible to create a more comprehensive and accurate picture of the value of a college degree. This can include examining industry trends, job profiles, and demographic characteristics when evaluating job offers for potential candidates.’]

- Response 2: ["Sure, there are common misconceptions about the value of a college degree in today’s job market. Here’s a list of some common misconceptions about the value of a college degree. 1. Colleges are only good for academics. Colleges are not the only source of knowledge. Technical skills and practical experience are also valued in today’s job market. 2. Colleges are expensive. The cost of a college education is often more expensive than a high school diploma. Furthermore, the cost of tuition and other expenses associated with a college education is significantly higher than the cost of going to school. 3. Colleges are not relevant for today’s job market. Colleges are considered irrelevant for today’s job market and are not as important as they were in the past. 4. Colleges are a waste of money. Colleges are not cost-effective for individuals or society as a whole. They are often viewed as a waste of time because they are seen as a waste of money by those who did not finish high school. 5. Colleges are not affordable. Colleges are seen as costly, even though they may be less expensive than high schools. To counteract these misconceptions, the U.S. Department of Education (ED) publishes the results of the annual Cost of College Survey, which provides an objective, fact-based assessment of the cost of a college education. The research is based on recent data and is widely accepted as accurate and credible. However, these stereotypes are often harmful in persuading people to forego the expense of a college education. As a result, many individuals don’t attend college as they believe that the quality of education is not worth the cost. By avoiding misconceptions and looking at the costs of various college options, individuals can make educated choices that benefit both themselves and society."]

[Figure 164]

Prompt: What do you think it feels like to be a Software Engineer? Response 1: [’"As a software engineer, you have the unique opportunity to work on some of the worldś most complex and challenging software projects. Your work can sometimes feel like a never-ending battle to come up with the best solution, but itś absolutely essential." According to a study conducted by the International Institute for Engineering Education (IIE), software engineers often experience a grueling and unpredictable work schedule, with long hours and intense responsibilities. The work may involve developing algorithms, writing code, fixing defects, testing and debugging, updating and maintaining software, and managing work-life balance. This can be especially challenging for those who are not used to a fast-paced, fast-paced work environment. It is also essential to provide a safe working environment, where teamwork, communication, and problem-solving are essential qualities for successful software engineers.’] Response 2: ["I think it’s safe to say that I know firsthand the anxiety and thrill that comes with being a software engineer. It’s not just a job, it’s an art form, and it demands great skills and an unwavering commitment to continuous learning and development. It’s also, sometimes, a lonely profession that requires constant mental and physical focus, resilience, and a willingness to work in a fast-paced and dynamic environment. The constant challenges of staying up-to-date with the latest technology and keeping up with technological advancements can make the experience even more rewarding. And, being a software engineer means that you’re constantly learning, growing, and pushing yourself to develop new skills in order to stay current and competitive in a fiercely competitive job market."]

[Figure 165]

[Figure 166]

Prompt: Define the definition of the idiomatic expression “change of pace” in the context of altering one’s rhythm or tempo of activity, which is often used to signify a shift in the pace of an ongoing event or activity.

[Figure 167]

- Response 1: [’In the context of altering oneś rhythm or tempo of activity, the idiomatic expression "change of pace" is used to express a change in direction, speed, or intensity. This can be seen in the following example: 1. "I began walking at a fast pace and then suddenly slowed down when I went into the forest."

2. "He started the race at a walking pace and then stopped. He had to stop for a few minutes to catch his breath. His heart was racing and he felt his breath. He started again, and got back to running his first run. He ran the whole race in a couple of hours." 3. "At the local beach, we got lost and only found our way after an hour of searching." In the example of the phrase you mentioned "change of pace" it means that the tempo or the speed of an activity can be altered or modified without stopping. A sudden change in the tempo of an activity can indicate that it is becoming faster or more intense. One of the reasons people use "change of pace" when they want to describe transforming their activities is because these changes can have different impacts on the outcome of those activities. They can make activities more exciting, more challenging, more relaxed, or even more engaging. So, itś important to note that these changes can often be subtle or subtle, depending on the nuances of the activity. In conclusion, the idiomatic expression "change of pace" is used in a way that communicates a change in direction, speed or intensity, which can give the impression of both excitement and calm.’]

- Response 2: [’"Change of pace" is an idiomatic expression commonly used in English to refer to a sudden or rapid change in a person or eventś pace or timing. The phrase is also widely used in technical contexts, such as the phrase "going from a fast pace to a slow pace" or "changing a pace to a slow pace" as the indication of a transition or change in performance.’]

[Figure 168]

Prompt: what gift should I take to 8 year olds in India?

[Figure 169]

- Response 1: [’A gift for an 8-year-old boy or girl living in India would be

a box of chocolates or a small gift basket with a few toys. Chocolates can be a popular gift in India, while small gift baskets with some snacks can also be a good choice. It is important to ensure that the gift is appropriate for the age of the recipient: for a young girl this may mean a toy suitable for her age and interests, or for a boy this could be a doll, an electronic toy, or a stuffed animal. Additionally, you should make sure that the box has a good quality, safe material for the recipient to open, and that it does not break easily. Also, some gift-giving customs can vary, so it is important to check this beforehand.’]

- Response 2: [’In India, 8-year-olds generally celebrate their birthday with friends, family, and teachers. Popular gifts often include school supplies, gift cards, or toys. If gifts are unavailable, parents or caregivers can offer homemade gifts, such as chocolates, fruit, or dry fruits.’]

[Figure 170]

Prompt: Given the task definition and input, reply with output. Given an English language product review, determine if it is a Good Review or a Bad Review. A good review is one where the product’s review has positive tone and Bad review is one where the tone of product’s review is negative. I really wanted to like this book. It is set in Italy yet my mother was Italian and it makes perfect sense for a woman to be there. However the story seemed disjointed at first. Did they meet before or after she disappeared? If we knew why she disappeared perhaps more insight into her character would have been appreciated. Maybe there will be another book to help with that.

[Figure 171]

- Response 1: ["This review shares a mix of factors that could be contributing to a mixed-tone review. While the reviewer expresses a desire to like the book, they also express a desire for a better understanding of the story. While the reviewer could see potential in the book, they find some parts of the story seeming disjointed, which could be a sign that the writer neglected to provide a meaningful focus on the character’s motivation. While the reviewer might not deem the book a complete waste of time, they still acknowledge that the disjointedness of the story may have been distracting or frustrating. Overall, the tone of the review is positive, showing interest and appreciation of the work, while expressing some reservations. Considering the reviewer’s initial excitement for the book, their concern about the story’s disjointedness, and their desire to see better insights into the protagonist’s character, their overall sentiment is ambivalent. This review is an attempt to provide constructive feedback and encourage better storytelling skills, which aligns with the goals of creating a well-rounded reading experience (readability, clarity, and engagement) in the genre of the book."]
- Response 2: [’Good Review’]

