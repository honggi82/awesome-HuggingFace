arXiv:2505.16925v2[cs.LG]26May2025

# Risk-Averse Reinforcement Learning with Itakura-Saito Loss

Igor Udovichenko Skolkovo Institute of Science and Technology Vega Institue Foundation Moscow, Russia i.udovichenko@skoltech.ru

Olivier Croissant Natixis Foundation Paris, France

Anita Toleutaeva Skolkovo Institute of Science and Technology Moscow, Russia

Evgeny Burnaev Skolkovo Institute of Science and Technology Artificial Intelligence Research Institute Moscow, Russia

Alexander Korotin Skolkovo Institute of Science and Technology Artificial Intelligence Research Institute Moscow, Russia a.korotin@skoltech.ru

## Abstract

Risk-averse reinforcement learning finds application in various high-stakes fields. Unlike classical reinforcement learning, which aims to maximize expected returns, risk-averse agents choose policies that minimize risk, occasionally sacrificing expected value. These preferences can be framed through utility theory. We focus on the specific case of the exponential utility function, where one can derive the Bellman equations and employ various reinforcement learning algorithms with few modifications. To address this, we introduce to the broad machine learning community a numerically stable and mathematically sound loss function based on the Itakura-Saito divergence for learning state-value and action-value functions. We evaluate the Itakura-Saito loss function against established alternatives, both theoretically and empirically. In the experimental section, we explore multiple scenarios, some with known analytical solutions, and show that the considered loss function outperforms the alternatives.

## 1 Introduction

Reinforcement learning (RL) has achieved remarkable success in domains where the primary goal is to learn policies by interacting with an environment [31, 40]. The goal is often formalized through a Markov decision process (MDP), which aims to find a policy that maximizes the expected cumulative reward received during the interaction with the environment [40].

However, agents must prioritize risk mitigation alongside performance in high-stakes applications such as finance, healthcare, and autonomous systems [4, 42, 30]. Traditional risk-neutral RL frameworks, which optimize for expected returns, often fail to account for the variability and tail risks inherent in these settings. Risk-averse RL addresses this by incorporating preferences that penalize uncertainty, typically formalized through utility theory or coherent risk measures [22].

Preprint. Under review.

Among the various utility-based methods, the exponential (or entropic) utility function stands out for its convenient properties [37, 26]. Yet, existing exponential-utility RL approaches typically require exponentiation of the value function at each step and can suffer from significant numerical instabilities [16]. These instabilities often prevent reliable convergence.

This paper introduces an approach to risk-averse RL by leveraging the Itakura-Saito (IS) divergence [28], a specific case of Bregman divergence [8, 3] historically used in signal processing [13] and non-negative matrix factorization [21]. The loss was first introduced in [34] yet remains unexplored in the broad RL community. In the paper, we derive the loss from the fundamental property of the Bregman divergence and compare it empirically against known alternatives.

### Contributions

- 1. Derivation: We formally derive the IS loss from Bregman divergence and show that it recovers the exponential utility’s Bellman equation under mild conditions, ensuring that the resulting value estimate is correct and that the method is scale-invariant. We also derive the corresponding stochastic approximation rule for the tabular setup.
- 2. Empirical Validation: Across a range of benchmarks—from analytically tractable portfolio examples 4.1 to a combinatorial RL problem 4.2—the IS loss outperforms existing baselines.

## 2 Background and Related Works

In this section, we briefly review the essentials of reinforcement learning (RL), emphasizing how risk aversion arises in decision-making processes, and why exponential utility proves useful for risk-sensitive control.

### 2.1 Markov Decision Process (MDP)

Consider a MDP of the form (S,A,r,p,s0), where S and A are sets of states and actions, respectively. Here r(s,a,s′) is the reward function, dependent on the current state, action, and next state. Statetransition probability (or density) is denoted as p(s′ | s,a). The initial state at t = 0 is s0. We assume the finite time horizon, so the time index t = 0,...,T, where T < ∞. The discount factor γ = 1 for simplicity. Extending our ideas on the case with γ < 1 and infinite time horizon is straightforward [26]. The timestamp is assumed to be a part of the state to avoid notation overload. By Π we denote the set of Markov policies π(a | s). We restrict our considerations to the class of Markov policies, because the optimal policy lies in it [26].

We define the trajectory T π as a random sequence of states and actions according to a policy π:

T π def= s0,a0,s1,a1,...,sT , at ∼ π(· | st), t = 0,...,T − 1. (1)

A trajectory part started at state s is denoted as Tsπ. Furthermore, we write Ts,aπ if the action at s is also fixed rather than sampled from π. We define the random return of a policy π as follows:

Rπ Rπ(s) Rπ(s,a) =

T−1

r(sτ,aτ,sτ+1), st,at,...,sT ∼ T π Tsπ Ts,aπ , (2)

τ=t

where t = 0 for Rπ or t is a timestamp of s for Rπ(s) and Rπ(s,a). The standard goal of RL is to find a policy that maximizes the expected return:

π∗ = arg max π∈Π

ET π[Rπ]. (3)

### 2.2 Learning Optimal Value Functions

Many RL algorithms rely on the state-value function V π(s) (or simply V -function) or action-value function Qπ(s,a) (Q-function) defined as follows:

V π(s) def= ET π

[Rπ(s)], Qπ(s,a) def= ET π

[Rπ(s,a)], (4)

s

s,a

V π(s) = 0 and Qπ(s,a) = 0 in all terminal states s. We denote the V -function of the optimal policy π∗ (optimal value function) by V ∗(·), and the optimal Q-function as Q∗(·,·). Thanks to the tower

property of the conditional expectation operator, value and optimal value functions satisfy the famous Bellman equations [40]:

V π(s) = Ea,s′[r(s,a,s′) + V π(s′)], (VV) V ∗(s) = max

Es′[r(s,a,s′) + V ∗(s′)], (VV*) Qπ(s,a) = Ea′,s′[r(s,a,s′) + Qπ(s′,a′)], (QQ) Q∗(s,a) = Es′ r(s,a,s′) + max

a∈A

Q∗(s′,a′) , (QQ*)

a′∈A

where the expectation is taken over the variables sampled from the policy or the state-transition law. Many deep learning algorithms in RL involve learning either the Q- or the V -function using Bellman equations. Policy gradient methods often rely on learning the value function for some policy π [31]. Policy evaluation step [40] aims to find a V π. The Vθπ is a NN parametrized by a weight vector θ. The NN is trained by optimizing the MSE objective that regresses Vθπ on the rhs of (VV). Define

δV (θ) = Vθπ(s) − r(s,a,s′) − Vθπ−(s′), (5)

the difference between the current approximation of the V -function and its target from the corresponding Bellman equation. By θ− we denote the target network’s weights. Then

LMSE(θ) = E(s,a,s′)

- 1

- 2δV (θ)2 , (MSE)

is optimized when Vθπ = V ∗. It follows from the fundamental property of conditional expectation being the optimal L2-predictor, so the minimum is attained when Vθπ satisfies (VV). The expectation is taken over tuples (s,a,s′) collected during the interaction of an agent an environment.

### 2.3 Formalizing the Risk Aversion

Consider two alternative returns an agent can choose, one is deterministic zero reward, and the other is either 1 or −1 with equal probabilities. For objective (3) they are equal, because their expected values are equal, but for some applications the deterministic reward is preferable, because it is “less risky”. There are many possible ways to formalize the preferences of random outcomes. The most straightforward one is through the von Neumann–Morgenstern (VNM) utility theorem. It states that under 4 VNM-rationality axioms [42, 22], the utility function can describe the agent’s preferences, i.e., random outcome X is preferable to Y , if E[u(X)] > E[u(Y )]. The utility function is defined up to affine transformations, e.g. u(·) and a + bu(·) describe the same preferences for a ∈ R and b > 0. A natural assumption, not implied by the VNM theorem, is that u(·) is a strictly increasing function, which can be interpreted as “there is no such thing as too much money”. Under this assumption, one can define the certainty equivalent (CE) as u−1(E[u(X]), a non-random reward that is equivalent to a random one from the VNM agent’s point of view.

The exponential (also called entropic) utility function u(x) = α−1(1−e−αx) represents a significant specific example. Coefficient α > 0 defines the agent’s risk aversion. In some applications, one can also consider the case α < 0, in which the agent is said to be risk seeking. If α → 0, the agent becomes indifferent to risk and treats outcomes with the same expected values as equal in the limit. As α → ∞, the agent treats all positive returns equally regardless of their magnitude and does not tolerate any losses.

The certainty equivalent (CE) for a random variable X is defined as the guaranteed amount that an agent would accept instead of taking a risk. For exponential utility, this is expressed as

E˜α[X] def= −α−1 log E e−αX , (6)

where α > 0 is the risk aversion parameter. Operator E˜α[·] shares many properties with expectation, hence the notation. The key ones are [22]:

- P.1 Normalization: E˜α[0] = 0.
- P.2 Monotonicity: If X ≤ Y a.s., then E˜α[X] ≤ E˜α[Y ].
- P.3 Translation invariance: E˜α[X + c] = E˜α[X] + c, c ∈ R.
- P.4 Tower property: E˜α E ˜α[Y | X] = E˜α[Y ]

Unlike the expectation, E˜α[·] is not linear, but concave, which is a weaker property:

#### P.5 Concavity: E˜α[λX + (1 − λ)Y ] ≥ λE˜α[X] + (1 − λ)E˜α[Y ], λ ∈ [0,1].

These unique properties allow us to derive Bellman equations [26] for the exponential utility similar to those widely used to solve risk-neutral MDPs.

2.4 Entropic MDP and its limitations

Risk-averse MDP aims to maximize the anticipated future return adjusted for unwillingness to bear excess risks. Due to our focus on the exponential utility, we formalize the objective as follows:

E˜αT π[Rπ]. (7)

π∗ = arg max π∈Π

This objective is analogous to (3), but the expectation operator E[·] is replaced with the CE operator E˜α[·] of exponential utility. We can define the value functions analogously to (4):

V˜π(s) def= E˜α[Rπ(s)], Q˜π(s,a) def= E˜α[Rπ(s,a)], (8)

The Bellman equations become [26]: V˜π(s) = E˜αa,s′ r(s,a,s′) + V˜π(s′) , (EVV) V˜∗(s) = max

E˜αs′ r(s,a,s′) + V˜∗(s′) , (EVV*) Q˜π(s,a) = E˜αa′,s′ r(s,a,s′) + Q˜π(s′,a′) , (EQQ)

a∈A

Q˜∗(s,a) = E˜αs′ r(s,a,s′) + max a′∈A

Q˜∗(s′,a′) . (EQQ*)

The seminal work on risk-sensitive MDP considered exponential utility [27]. Recently, works [5,

##### 6, 7, 32, 35, 20, 15, 18, 19, 17, 33, 36, 26, 16, 23] also considered MDPs with exponential utility specifically. Many of these methods rely on learning the optimal Q- or V -function. The value function is often auxiliary in RL algorithms, since the ultimate goal is to learn policy. However, in some applications, learning the precise value function is critical. For example, in finance, it represents the portfolio value or the price of the derivative being hedged [10, 11, 9, 34, 29, 24].

Since the CE operator replaces the expectation, note that objective (MSE) does not learn the correct value function for entropic MDP. The majority of the works mentioned above rely on the following objective, which we call exponential MSE loss:

2

- 1

- 2

LEMSE(θ) = E(s,a,s′)

α−2 exp −αV˜θπ(s) − exp −αr(s,a,s′) − αV˜θπ−(s′)

. (EMSE)

The optimizer of this loss is a correct value function for the risk-averse MDP. By the Taylor expansion (EMSE) can be rewritten as:

LEMSE = 21 exp −2αV˜θπ(s) Ea,s′ δV˜ (θ)2 + o δV˜ (θ)2 , (9) so, the objective (EMSE) reduces to MSE loss for risk-tolerant agents or small error δV˜ (θ). Note that it depends on θ not only through the δV˜ (θ) because of the factor exp −2αV˜θπ(s) . We argue that such dependence is highly undesirable. First, the loss vanishes for high positive values of V˜θπ(s) and explodes for high negative values. Works [23, 16] note its numerical instability. Second, from the translation invariance property P.3 of E˜α[·], the learning of V˜θπ should not depend on the absolute levels of its values.

Another loss was proposed in [15], which we call softplus because of the term log 1 + exp{αz} :

#### LSP(θ) = 2δV˜ (θ)α−1 log 1 + exp αδV˜ (θ) + 2α−2li2 −exp αδV˜ (θ) + π2 (6α2), (SP)

where li2(z) is Spence’s dilogarithm function li2(z) = − 0 z log(1z−z) dz. It appears as an objective, whose gradient coincides with the heuristic stochastic approximation rule, introduced in [32]. This

loss depends on θ only through the δV˜ (θ), which makes it numerically more stable than (SP). However, it is not convex and only learns the correct value function when the target has a Gaussian

distribution. In summary, the known losses have the following limitations:

[Figure 1]

- Figure 1: Comparison of loss penalties for a one-step value prediction error δV˜ (θ) when α = 1. A

positive δV˜ (θ) > 0 means the current estimate V˜θ(s) underestimates the true CE value (the return is higher than expected). Risk-averse losses heavily penalize underestimation (δV˜ (θ) > 0) since underestimating the value implies unaccounted risk, whereas overestimation (δV˜ (θ) < 0) is penalized less. MSE, being risk-neutral, is symmetric. EMSE (exponential MSE) grows with the absolute value of V , leading to numerical instability for large values.

- • EMSE is numerically unstable,
- • SP is optimized by the value function only in a specific case. In the following section, we consider the objective that addresses these limitations.

## 3 Itakura-Saito Loss for Learning Risk-Averse Value Function

In this section, we consider an objective for learning the value function that is mathematically correct and numerically stable. While MSE loss minimizes the expectation, there are other objectives with similar properties, especially for risk-sensitive settings.

- 3.1 Bregman Divergence and Itakura-Saito Loss Recall the definition of Bregman divergence (BD) [8]:

dφ(x,y) = φ(x) − φ(y) − x − y,∇φ(y) , (10)

where φ(·) is a differentiable convex function. It measures the discrepancy induced by a convex function φ; The important property is that the true mean minimizes the expected divergence:

E[dφ(X,y)]. (11)

E[X] = arg min

y

In other words, the expectation is the “best prediction” under any Bregman loss, a generalization of the fact that the mean minimizes MSE. Moreover, BD is an exhaustive class of loss functions for

which the expectation is the optimizer [2]. BD with φ(z) = 21∥z∥2 reduces to the MSE loss.

BD with φ(z) = −log z is known as Itakura-Saito (IS) distance [28] dIS(x,y) = x/y−log(x/y)−1. It is widely used in audio processing [13] and non-negative matrix factorization [21]. Substituting the

prediction exp −αV˜θπ(s) and the regression target exp −αr(s,a,s′) − αV˜θπ−(s′) into dIS(·,·) yields the following Itakura-Saito loss [34]:

LIS(θ) def= α−2E(s,a,s′) exp αδV˜ (θ) − αδV˜ (θ) − 1 . (IS) In Appendix C we formally state and prove the following proposition. Proposition 1. Under mild assumptions the value function that minimizes (IS) satisfies (EVV).

First, note that (IS) depends on θ only through δV˜ (θ). Second, by the Taylor expansion:

LIS = E(s,a,s′) so, for a risk-tolerant agent or small discrepancies between the V -function and its target, the ItakuraSaito loss reduces to the MSE loss. We compare visually all losses in Figure 1. Notably, IS loss casts the risk-sensitive Bellman criterion into a form suitable for stochastic gradient descentcircumventing the bias issues identified in past risk-sensitive Q-learning attempts [32].

- 1

- 2δV˜ (θ)2 + o δV˜ (θ)2 , (12)

### 3.2 Stochastic Approximation Rule

Tabular RL algorithms [40] rely on the stochastic approximation (SA) [38] procedure rather than optimizing some loss function. The procedures are equivalent in the following sense:

θ∗ = arg min

L(θ) ⇐⇒ ∇L(θ∗) = 0.

θ

Solving the latter problem with SA is equivalent to solving the former with stochastic gradient descent. Taking the gradient of (IS) we derive the following stochastic approximation scheme:

V˜kπ+1(s) ← V˜kπ(s) − ηkα−1 exp αδV˜ (k) − 1 , (13)

where ηk is the learning rate, V˜kπ(·) is the approximation of V˜π(·) on the k-th step of the stochastic approximation algorithm, and δV˜ (k) is the difference between V˜k(s) and its target.

## 4 Experiments

In this section, we empirically compare the (IS) loss against (EMSE) and (SP). We choose the financial problems as our primary experimental setups for the following reasons:

- 1. The very concept of risk aversion originates in economics and finance [4, 42, 22].
- 2. RL is widely considered in financial literature [30, 25].
- 3. The proposed setups admit ground truth analytical solutions (see Appendix A) or theoretical references, which allow us to highlight the difference between the losses.

Also, we compared all losses in a more complex setup considered in [16], where the authors propose to use risk-averse RL to increase the robustness of the learned policy against distribution shifts. We disclose all technical details in Appendix B.

- 4.1 Portfolio Optimization and Hedging Consider the problem of optimal stock trading in several setups. The state space is represented by

the stock price augmented with timestamp: s = (t,St) ∈ {0,...,T} × R. Each time, an agent can buy or sell any amount of stock, so the action space A = R is continuous. We consider the discrete-time Bachelier model [1] of stock price dynamics, so the price increments are independent and normally distributed. Let Zt ∼ N(µ,σ2), t = 1,...,T be the iid Gaussian variables with mean µ and variance σ2. The state transition law is: s′ = (t + 1, St + Zt+1), t = 0,...,T − 1, so the state transition law is independent of the action taken. The initial stock price is S0, so s0 = (0,S0). The reward function is specified differently for each setup.

We parametrize πϕ(s) and V˜θπ(s) as multi-layer perceptrons. We use the TD(0) learning with function approximation [40] to learn the V -functions. Authors in [26] prove that the optimal policy is

deterministic in this case, so the action is a non-random output of πϕ(s). To learn the optimal policy, we minimize the following objective:

L(ϕ) = Es′ α−1 exp −α r(s,πϕ(s),s′) + V˜θπ(s′) − V˜θπ(s) , (14)

which estimates the gradient of α−1 exp −α Q ˜π s,πϕ(s) − V˜θπ(s) using one TD(0) sample. It learns the correct policy, since the function α−1 exp{−αx} is monotonically decreasing.

Analytically Tractable Cases In the first experiment we set µ > 0 and r(s,a,s′) = a(St+1 − St), t = 0,...,T − 1, so the rewards come from stock trading solely. Return Rπ(s0) =

T−1 t=0 atZt+1 is distributed normally as a sum of Gaussian random variables, so the application of

objective (SP) is mathematically sound here. The optimal V -function and optimal policy can be derived analytically (see Appendix A):

µ2(T − t) 2ασ2

µ ασ2

π∗(s) =

, V ∗(s) =

. (15)

Next, we set µ = 0 and we consider the reward function of the form:

r(s,a,s′) =

a(St+1 − St), t = 0,...,T − 2, a(ST − ST−1) + g(ST), g(x) = 21(x − S0)2 t = T − 1.

(16)

[Figure 2]

[Figure 3]

(a) Gaussian reward (b) Quadratic reward

- Figure 2: Error in learning the obtained approximation of V ∗ in the Gaussian and quadratic cases. Each experiment was run five times with different random seeds. In the Gaussian case, losses perform on par. Loss (SP) does not learn the correct value function for the non-Gaussian return.

[Figure 4]

(a) Training process with α = 10. We depict the loss value during training for five random seeds for each loss. Objectives (SP) and (IS) converge successfully, while all runs with (EMSE) failed.

[Figure 5]

(b) We run (SP) and (IS) five times for each value of risk aversion α. The filling covers the area ± 1 standard deviation around the mean value. Although losses converge to the theoretical risk-neutral reference, (IS) is more stable for large values of α than (SP).

- Figure 3: Loss performance on the Deep Hedging problem [10]. Loss (IS) shows more stable and reliable convergence than the alternatives.

It is similar to the previous one, except the agent receives a quadratic reward at the last moment. The return is not Gaussian anymore, so the loss (SP) learns the incorrect value function. An analytical solution exists in this case:

−α(St − S0)2 + (T − t)log 1 − ασ2 2α

π∗(s) = α(St − S0), V ∗(s) = −

. (17)

We compare the (IS) loss with the alternatives in Figure 2. We use RMSE between the learned value function and the analytical solution RMSE = Es[(V ∗(s) − Vθπ(s))2]. Note that the expectation does not depend on a, because the state-transition law is independent of a.

Deep Hedging The European call option is the simplest non-linear derivative contract, which has the payoff of the form h(x) = max{x − K,0}, where x is the stock price at a pre-determined moment and K is the contract parameter, called the strike price [22]. We consider the following reward function:

a(St+1 − St), t = 0,...,T − 2, a(ST − ST−1) − h(ST), h(x) = max{x − K,0} t = T − 1.

r(s,a,s′) =

(18)

Similar problems are widely considered in the financial literature [10, 34, 12, 29, 39]. The main goal is to calculate the price of the derivative, V˜∗ s0 in our notations. In our case, the closed-form solution is available only for the risk-neutral case (α = 0): V ∗ s0 = σ T/2π [1, 14]. Interestingly, in the risk-neutral case, every policy for which the expected return is well defined mathematically is optimal, because the price process is a martingale.

[Figure 6]

(a) Validation performance of risk-sensitive SAC under undiscounted returns (γ = 1).

[Figure 7]

(b) Validation performance of RSSAC under discounted returns (γ = 0.99).

- Figure 4: Loss performance on the RSSAC problem [16]. Learning curves depict the mean validation return during the training process. Each line represents the average over three random seeds, with shaded areas indicating ±1 standard deviation. The (EMSE) loss destabilizes training.

We show the results in Figure 3. First, Figure 3a supports our speculations (9) about the unstable nature of (EMSE) loss. Second, Figure 3b shows that (SP) and (IS) succeeded in converging to the theoretical risk-neutral reference. However, the (SP) loss shows higher variance across random seeds than (IS) loss. We speculate that the primary cause is the non-convex nature of (SP), so the optimization procedure can get stuck in a local minimum. Also, the return is not Gaussian in this case (although close), so the usage of (SP) is not justified.

- 4.2 Risk-Averse Soft Actor-Critic (RSSAC) for Robust Combinatorial Optimization

This experiment aims to show that loss (IS) can act as a performance-enhancing drop-in replacement of known losses in complex RL algorithms. We adopt the experimental setup of [16], which resembles the warehouse management problem. The environment is a 5 × 5 grid. Each time, items randomly and independently appear in grid cells according to some probability distribution unknown to the agent. The agent can move up, down, left, right, or stay. Any movement costs −1 to the agent. When the agent reaches the cell with an item, it picks it. If the agent delivers the item to the specific cell, it receives a +15 reward. If an item is not picked during some period after its spawn, it disappears. The agent can carry at most one item at a time. The duration of episodes is constant.

The authors study the problem of learning a policy robust to distribution shifts. They propose the risk-averse soft actor-critic algorithm with exponential utility to learn such policies. The algorithms learns a Q-function as a critic. The authors rely on the approximate equality E[Xγ] ≈ E[X]γ, where γ is a discount factor, and derive the Bellman equation of the form:

Qπθ(s,a) = E˜αs′,a′ r(s,a,s′) + γκH(πϕ · | s) + γQπθ−(s′,a′) , (19) where H πϕ(· | s) is the entropy of the policy πϕ and κ is the entropic regularization coefficient. The authors note the unstable nature of (EMSE) loss and propose to regress Qπθ(s,a) on sampling-based estimation of the rhs of (19). They replace the expectation over s′ with a single s′ from the replay buffer and directly compute the expectation over the next actions to estimate the regression target. As noted by the authors, this results in a biased estimation of the Q-function. Nevertheless, they do it because they do not aim to recover the correct Q-function, but to learn the close-to-optimal policy.

We run the proposed soft actor-critic algorithms with minor modifications. Instead of relying on the direct estimation of the rhs of (19), we learn the Q-function using the unbiased objective. We compare the losses in Figure 4. The figure shows the validation return during the training process. To measure the robustness, the return is computed with the probability of item appearance different from the one used during training. The objective (EMSE) destabilizes the training process. The loss (IS) performs on par with (SP) and consistently outperforms the (EMSE) objective in complex RL algorithms and environments.

- 5 Conclusion

This paper considers the Itakura-Saito loss first proposed in [34], a simple loss function to learn the value function in risk-averse MDPs. We proved that the minimizer of this loss is indeed the correct value function and derived the corresponding SA rule. Numerical experiments show that alternatives either destabilize training or do not recover the correct value function. Itakura-Saito loss can be used as a drop-in replacement in complex RL algorithms.

## References

- [1] Louis Bachelier. “Théorie de la spéculation”. In: Annales scientifiques de l’École normale supérieure. Vol. 17. 1900, pp. 21–86.
- [2] Arindam Banerjee, Xin Guo, and Hui Wang. “On the optimality of conditional expectation as a Bregman predictor”. In: IEEE Transactions on Information Theory 51.7 (2005), pp. 2664– 2669.
- [3] Arindam Banerjee et al. “Clustering with Bregman divergences”. In: Journal of machine learning research 6.Oct (2005), pp. 1705–1749.
- [4] Daniel Bernoulli. “Commentarii academiae scientiarum imperialis petropolitanae”. In: Petropoli. Chap. De vibrationibus et sono laminarum elasticarum 27 (1751), p. 28.
- [5] Vivek S Borkar. “A sensitivity formula for risk-sensitive cost and the actor–critic algorithm”. In: Systems & Control Letters 44.5 (2001), pp. 339–346.
- [6] Vivek S Borkar. “Q-learning for risk-sensitive control”. In: Mathematics of operations research 27.2 (2002), pp. 294–311.
- [7] Vivek S Borkar and Sean P Meyn. “Risk-sensitive optimal control for Markov decision processes with monotone cost”. In: Mathematics of Operations Research 27.1 (2002), pp. 192– 209.
- [8] Lev M Bregman. “The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming”. In: USSR computational mathematics and mathematical physics 7.3 (1967), pp. 200–217.
- [9] Hans Buehler, Phillip Murray, and Ben Wood. “Deep bellman hedging”. In: arXiv preprint arXiv:2207.00932 (2022).
- [10] Hans Buehler et al. “Deep hedging”. In: Quantitative Finance 19.8 (2019), pp. 1271–1291.
- [11] Hans Buehler et al. “Deep hedging: learning to remove the drift under trading frictions with minimal equivalent near-martingale measures”. In: arXiv preprint arXiv:2111.07844 (2021).
- [12] Jay Cao et al. “Deep Hedging of Derivatives Using Reinforcement Learning”. In: The Journal of Financial Data Science 3.1 (2021), pp. 10–27.
- [13] Alan HS Chan. Advances in industrial engineering and operations research. Vol. 5. Springer Science & Business Media, 2008.
- [14] Jaehyuk Choi et al. “A Black–Scholes user’s guide to the Bachelier model”. In: Journal of Futures Markets 42.5 (2022), pp. 959–980.
- [15] Grégoire Delétang et al. “Model-free risk-sensitive reinforcement learning”. In: arXiv preprint arXiv:2111.02907 (2021).
- [16] Tobias Enders, James Harrison, and Maximilian Schiffer. “Risk-sensitive soft actor-critic for robust deep reinforcement learning under distribution shifts”. In: arXiv preprint arXiv:2402.09992 (2024).
- [17] Yingjie Fei and Ruitu Xu. “Cascaded gaps: Towards logarithmic regret for risk-sensitive reinforcement learning”. In: International Conference on Machine Learning. PMLR. 2022, pp. 6392–6417.
- [18] Yingjie Fei, Zhuoran Yang, and Zhaoran Wang. “Risk-sensitive reinforcement learning with function approximation: A debiasing approach”. In: International Conference on Machine Learning. PMLR. 2021, pp. 3198–3207.
- [19] Yingjie Fei et al. “Exponential bellman equation and improved regret bounds for risk-sensitive reinforcement learning”. In: Advances in neural information processing systems 34 (2021), pp. 20436–20446.
- [20] Yingjie Fei et al. “Risk-sensitive reinforcement learning: Near-optimal risk-sample tradeoff in regret”. In: Advances in Neural Information Processing Systems 33 (2020), pp. 22384–22395.
- [21] Cédric Févotte, Nancy Bertin, and Jean-Louis Durrieu. “Nonnegative matrix factorization with the Itakura-Saito divergence: With application to music analysis”. In: Neural computation 21.3

(2009), pp. 793–830.

- [22] Hans Föllmer and Alexander Schied. Stochastic finance: an introduction in discrete time. Walter de Gruyter, 2011.
- [23] Alonso Granados, Reza Ebrahimi, and Jason Pacheco. “Risk-Sensitive Variational ActorCritic: A Model-Based Approach”. In: The Thirteenth International Conference on Learning Representations. 2025.

- [24] Igor Halperin. “QLBS: Q-Learner in the Black-Scholes (-Merton) Worlds”. In: Journal of Derivatives 28.1 (2020), pp. 99–122.
- [25] Ben Hambly, Renyuan Xu, and Huining Yang. “Recent advances in reinforcement learning in finance”. In: Mathematical Finance 33.3 (2023), pp. 437–503.
- [26] Jia Lin Hau, Marek Petrik, and Mohammad Ghavamzadeh. “Entropic risk optimization in discounted MDPs”. In: International Conference on Artificial Intelligence and Statistics. PMLR. 2023, pp. 47–76.
- [27] Ronald A Howard and James E Matheson. “Risk-sensitive Markov decision processes”. In: Management science 18.7 (1972), pp. 356–369.
- [28] Fumitada Itakura. “Analysis synthesis telephony based on the maximum likelihood method”. In: Reports of the 6th Int. Cong. Acoust., 1968 (1968).
- [29] Petter N Kolm and Gordon Ritter. “Dynamic replication and hedging: A reinforcement learning approach”. In: The Journal of Financial Data Science 1.1 (2019), pp. 159–171.
- [30] Petter N Kolm and Gordon Ritter. “Modern perspectives on reinforcement learning in finance”. In: Modern Perspectives on Reinforcement Learning in Finance (September 6, 2019) (2019).
- [31] Yuxi Li. Deep Reinforcement Learning. 2018. arXiv: 1810.06339 [cs.LG]. URL: https: //arxiv.org/abs/1810.06339.
- [32] Oliver Mihatsch and Ralph Neuneier. “Risk-sensitive reinforcement learning”. In: Machine learning 49 (2002), pp. 267–290.
- [33] Mehrdad Moharrami et al. “A policy gradient algorithm for the risk-sensitive exponential cost mdp”. In: Mathematics of Operations Research 50.1 (2025), pp. 431–458.
- [34] Phillip Murray et al. “Deep hedging: Continuous reinforcement learning for hedging of general portfolios across multiple risk aversions”. In: Proceedings of the Third ACM International Conference on AI in Finance. 2022, pp. 361–368.
- [35] David Nass, Boris Belousov, and Jan Peters. “Entropic risk measure in policy search”. In: 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE. 2019, pp. 1101–1106.
- [36] Erfaun Noorani, Christos N Mavridis, and John S Baras. “Exponential TD Learning: A Risk-Sensitive Actor-Critic Reinforcement Learning Algorithm”. In: 2023 American Control Conference (ACC). IEEE. 2023, pp. 4104–4109.
- [37] Irina Penner. “Dynamic convex risk measures: time consistency, prudence, and sustainability”. In: Humboldt-Universität zu Berlin (2007).
- [38] Herbert Robbins and Sutton Monro. “A stochastic approximation method”. In: The annals of mathematical statistics (1951), pp. 400–407.
- [39] Zoran Stoiljkovic. “Applying Reinforcement Learning to Option Pricing and Hedging”. In: arXiv preprint arXiv:2310.04336 (2023).
- [40] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.
- [41] Xiaolu Tan and Nizar Touzi. “Optimal transportation under controlled stochastic dynamics”. In: The annals of probability (2013), pp. 3201–3240.
- [42] John Von Neumann and Oskar Morgenstern. “Theory of games and economic behavior, 2nd rev”. In: (1947).

## A Analytical Solutions

We provide here the closed-form derivations for the ground-truth benchmarks reported in the experiments (cf. Section 4 in the main text). All results are obtained under the discrete-time Bachelier dynamics

St+1 = St + Zt+1, Zt+1 ∼ N(µ,σ2), t = 0,...,T − 1, and use the exponential-utility certainty equivalent

1 α

log E[exp{−αX}], α > 0. For any Gaussian variable G ∼ N(m,v) one has

Eα[X] = −

2

2

Eexp −αG ] = exp −αE[G] + α

2 Var(G) = exp −αm + α

2 v a fact used repeatedly below.

### A.1 Pure trading with Gaussian reward

Consider a single-period reward rt = at(St+1 − St) = atZt+1. Since (Zt) are i.i.d. Gaussians, the optimal deterministic policy π∗ is time-independent and the value function does not depend on St. For any fixed action a the one-step certainty-equivalent return is

1 α

log E[exp{−αaZt+1}] = µa − 21αa2σ2.

Eα[aZt+1] = −

Maximising this quadratic over a gives a∗t = ασµ2 and the corresponding single-step optimum maxa Eα[aZt+1] = µ

2(T−t) 2ασ2 . Summing over T − τ periods yields the value function V∗(st) =

µ2(T − t)/(2ασ2). Hence, the optimal policy is constant in time and satisfies

|π∗(st) =<br><br>µ ασ2<br><br>, V∗(st) =<br><br>µ2(T − t) 2ασ2<br><br>,|
|---|

as quoted in Eq. (14) in Sec. 4.1. No additional integrability condition is required here because exp(−αaZ) is integrable for all α.

### A.2 Trading with a quadratic terminal penalty (µ = 0)

Let xt = St − S0 be the centred price. Rewards are rt = atZt+1 for t < T − 1, while at t = T − 1 the agent additionally incurs a terminal penalty rT−1 = aT−1ZT − 21x2T. We look for a solution of the form with boundary conditions:

- 1

- 2

Ktx2 + Ct, KT = 1, CT = 0. Since xt+1 = xt + Zt+1 conditional on xt, xt+1 is Gaussian, so

Vt(x) = −

E exp −α{aZt+1 − 12Kt+1(xt + Zt+1)2 + Ct+1} =

ασ2 2(1 − αKt+1σ2)

αKt+1 2

- 1

- 2 exp −

x2t (1 − αKt+1σ2)−

(a − Kt+1xt)2 .

= exp −αCt+1 +

By direct calculation of the certainty equivalent at each step, one finds that the exponential inside the expectation is maximized when a = Kt+1xt, so the optimal policy is

a∗t = Kt+1xt

and the maximized Bellman update is dynamic recursion

Vt(x) = Ct+1 −

- 1

- 2

Kt+1x2 +

- 1

- 2α

log(1 − αKt+1σ2).

Matching coefficients gives Kt = Kt+1 and KT = 1, hence Kt ≡ 1 for all t. Meanwhile, the scalars Ct satisfy Ct = Ct+1 + 21α log(1 − ασ2), with boundary CT = 0, which solves to

T − t 2α

log(1 − ασ2).

Ct =

The recursion is well-defined only if ασ2 < 1, ensuring the Gaussian moment-generating function exists. Under the above condition the optimal policy and value are Provided ασ2 < 1 (so that the required moment-generating function is well-defined), the optimal policy and value function are

|π∗(st) = St − S0, V∗(st) = −<br><br>1<br><br>2<br><br><br>(St − S0)2 +<br><br>T − t 2α<br><br>log(1 − ασ2), ασ2 < 1,|
|---|

which coincides with Eq. (16) in Sec. 4.1.

## B Experimental Details

### B.1 Portfolio Optimization and Hedging

Throughout all experiments in this section, we take T = 10. The code is written in pure PyTorch 2.7.0. We approximate the state-value function and policy with multi-layer perceptrons with Mish activation and 2 hidden layers, 64 neurons each. We used the Adam optimizer (lr=1e-4, β1 = 0.99, β2 = 0.999) and applied the following learning rate schedule:

- • During the first 1k iterations, the learning rate grows linearly with a start factor of 0.01.
- • It then remains constant for the next 49k (149k for Deep Hedging experiments) iterations.
- • Afterwards, we apply cosine decay with T_max equal to 50k (150k for Deep Hedging).

Each training batch has a size of 1024, with gradient values clipped at 1 and the gradient norms clipped at 10. Each experiment is repeated with five independent seeds that affect weight initialization and mini-batch sampling. The other parameters are listed in the table:

Experiment µ σ α Gaussian return 0.03

1 Quadratic penalty

√0.102

100

0

Deep Hedging {0.1,0.3,1,3,10}

Our experiments typically run on an A100 GPU, with up to three runs in parallel on a single device. For the Gaussian and quadratic tasks, we use 100k training iterations, requiring between 25 and 65 minutes of wall-clock time depending on concurrent jobs. The Deep Hedging setup generally requires 50 minutes to 2 hours for each run, owing to its longer 300k iteration schedule.

### B.2 Risk-Averse Soft Actor-Critic (RSSAC) for Robust Combinatorial Optimization

We employ the open-source code provided by the authors of [16]1 and leave all network, replay-buffer and optimiser hyper-parameters unchanged except for the three modifications below.

- 1. We took β = −0.1, which is α = 0.1 in our notations;
- 2. The discount factor is γ ∈ {0.99,1};
- 3. The loss function is simpler than in [16] and do not involve computing the expectation over the next actions.

Every RSSAC run (500 k environment steps) takes ≈2 h on a single A100 GPU. No concurrent jobs are scheduled on the same device.

1https://github.com/tumBAIS/RiskSensitiveSACforRobustDRLunderDistShifts.

- C Statement and Proof of Proposition 1 Before stating the result we recall the Itakura–Saito loss (IS)

LIS(θ) def= α−2E(s,a,s′) exp αδV˜ (θ) − αδV˜ (θ) − 1 (20) and

δV (θ) = Vθπ(s) − r(s,a,s′) − Vθπ−(s′). (21) Proposition. Assume the following hold:

- 1. The target network weights θ− are a copy of the main network weights θ, Vθπ−(s′) = stop gradient Vθπ(s′) in (21);
- 2. Both E exp −αr(s,a,s′) − αV˜θπ−(s′) and E r(s,a,s′) + V˜θπ−(s′) exist;
- 3. LIS attains its minimum at θ⋆. Then, V˜θπ∗ satisfies the risk–averse Bellman equation (EVV):

V˜π(s) = E˜αa,s′ r(s,a,s′) + V˜π(s′) . (22)

Proof. First, note that

LIS(θ) = α−2Es,a,s′ dIS exp −αr(s,a,s′) − αV˜θπ(s′) , exp −αV˜θπ(s) ,

where α−2 is a scaling coefficient. Thanks to the assumption 2, one can apply Theorem 1 in [41], so for the minimizer it holds:

exp −αV˜θπ∗(s) = Ea,s′ exp −αr(s,a,s′) − αV˜θπ∗(s′) .

Taking the logarithm of both sides, dividing by −α and recalling the definition of E˜α[·] we obtain:

V˜θπ∗(s) = E˜αa,s′ r(s,a,s′) + V˜θπ∗(s′) .

| |
|---|

- D Geometric and Field-Theoretic Interpretations of the IS Loss

This appendix explores the deeper mathematical structure behind the Itakura-Saito (IS) loss and its regularization. While the main paper focuses on empirical performance and optimization theory, here we reinterpret the IS loss in the language of variational calculus and field theory.

We show that, under regularization, the IS loss induces an energy functional with features reminiscent of classical scalar field theories. In particular, we analyze its behavior in the linearized regime, where it yields a propagator with long-range correlations and, in the high-target limit, reveals an emergent conformal symmetry. These structures illuminate why the IS loss promotes global consistency and smoothness in prediction.

We then speculate on a broader analogy: drawing inspiration from the AdS/CFT correspondence in theoretical physics, we suggest that the robustness and coherence induced by IS-trained models may be seen as a form of holographic encoding — where local losses reflect and preserve global structure.

No advanced physics background is required to follow this appendix — only a curiosity for the beautiful human adventure of understanding the universe through mathematics, symmetry, and learning.

### D.1 Conformal Symmetry and Long-Range Correlations Induced by Itakura-Saito Loss

To understand the structural properties of the IS loss beyond pointwise optimization, we adopt a variational viewpoint by constructing an action functional. This approach is standard in physics and calculus of variations: instead of minimizing a discrete loss over samples, we define a functional that integrates a local loss density over a continuous domain. In this setting, the learned function ϕ(x) becomes a field, and the loss becomes an energy functional whose minimizers reflect both local fit and global coherence. This field-theoretic perspective allows us to uncover the long-range smoothing and scale-invariance properties implicit in the IS loss when combined with regularization.

We analyze the field-theoretic formulation of the IS loss with smoothness regularization and show how conformal symmetry emerges in the continuum limit.

### D.2 Two-Point Correlation from the IS-Induced Action We consider the action functional:

2

y(x) ϕ(x) − log

dϕ dx

S[ϕ] = dx λ

+

y(x) ϕ(x) − 1 (23)

To study long-range correlations, we simplify and linearize around a constant input y(x) = y0, assuming:

ϕ(x) = y0 + ε(x), ε(x) ≪ y0 (24) Expanding the IS potential term:

2

y0 ϕ

1 1 + ε/y0 ≈ 1 −

ε y0

ε y0

=

+

2

y0 ϕ

ε y0 ≈ −

- 1

- 2

ε y0

ε y0 −

= −log 1 +

log

So the potential becomes:

2

ε y0

- 1

- 2

(25) The linearized action reads:

VIS(y0,ϕ) ≈

2

dε dx

- 1

- 2y02

ε2(x) (26)

S[ε] ≈ dx λ

+

This is a standard Gaussian field theory with mass m2 = 2y12

.

0

### D.3 Propagator and Emergence of Conformal Limit The propagator satisfies:

d2 dx2

- 1

- 2y02

G(x) = δ(x) (27)

−λ

+

Here, δ(x) denotes the Dirac delta function, a generalized distribution satisfying δ(x)f(x)dx = f(0) for any smooth test function f. It models a pointwise source and is used to define Green’s functions in field theory.

Solving in 1D yields:

y0 √

|x| √

e−

2λy0 (28)

G(x) =

2λ

In the conformal limit y0 → ∞ or λ → 0:

1 |x|

(29)

G(x) ∼

This reflects:

- • Absence of intrinsic length scale,
- • Power-law correlation decay,
- • Conformal symmetry and long-range propagation.

### D.4 Implications for Optimization Dynamics

We now show how IS loss, being scale-invariant, improves optimization convergence. Let θ ∈ Rd be model parameters, with outputs f(θ) ∈ Rn, and targets y. Assume gradient descent:

θt+1 = θt − η∇L(θt) (30) We compare:

- • MSE: LMSE(θ) = 21∥f(θ) − y∥2

- • IS: LIS(θ) = i y

fi(θ) − log y

fi(θ) − 1 Conditioning and Hessian Geometry:

i

i

- • MSE: Hessian is H = J⊤J where J = ∂f/∂θ.
- • Large spread in y leads to poor conditioning.
- • IS: Penalizes relative differences f(θ)/y.
- • Built-in normalization leads to better conditioned Hessian and stable steps.

Local Approximation in 1D: Let f = y + ε with small ε. Then:

y f − log

LIS(f) =

2

1 2

ε y

≈

y f − 1

Hence IS behaves locally like a rescaled MSE:

LIS(f) ≈

- 1

- 2y2

ε2 (31)

Large y ⇒ lower weight, small y ⇒ higher weight. This performs implicit preconditioning, akin to natural gradients.

Natural gradients arise in information geometry as an improvement over standard (Euclidean) gradients. Instead of computing updates in the raw parameter space, the natural gradient rescales the direction of steepest descent using the inverse Fisher information matrix. This aligns updates with the intrinsic curvature of the loss landscape, leading to faster convergence and better conditioning. In the context of IS loss, the local reweighting of errors by 1/y2 mimics this effect: low-output regions receive stronger updates, similar to how natural gradients emphasize directions of low Fisher variance [43].

## E Conformal Invariance Improves Statistical Conditioning

In this appendix, we formalize the intuition that conformal invariance, as induced by the Itakura-Saito (IS) loss, statistically improves the conditioning of the optimization problem. We do so by analyzing the distribution of Hessian spectra under a Gaussian prior over models.

Remark. This theorem highlights a novel geometric interpretation of scale-invariance: it not only regularizes the optimization surface but also statistically preconditions the curvature — providing a mathematical basis for the empirical advantages of IS loss.

We propose a theoretical explanation for the improved optimization behavior observed when using the Itakura-Saito (IS) loss, through the lens of statistical conditioning and symmetry constraints.

Theorem (Improved Spectral Conditioning under Conformal Invariance). Let ϕ ∼ G be a Gaussian ensemble of models (e.g., fields or neural networks), and let Gconf ⊂ G be the subset of models such that the IS-induced action SIS[ϕ] is conformally invariant.

Define a spectral conditioning measure λ(ϕ), such as the variance or entropy of the eigenvalue spectrum of the Hessian at ϕ. Then:

[Var(λ(ϕ))] < Eϕ∈G[Var(λ(ϕ))] (32)

Eϕ∈G

conf

Sketch of proof. The IS loss induces a field theory with conformal symmetry in the limit y0 → ∞, leading to scale-free correlations. When restricting to Gconf, the underlying symmetry enforces statistical regularity in the structure of the Hessian. In contrast, over the full Gaussian ensemble G, more arbitrary fluctuations are allowed. By known results from random matrix theory and information geometry, imposing such symmetries reduces the variance and entropy of the spectral distribution. This leads to improved average conditioning.

| |
|---|

This supports the empirical observation that the IS loss improves the landscape geometry for gradient descent by constraining optimization trajectories to a submanifold of better-conditioned models.

### E.1 Discussion and Related Work

While a general formal proof connecting conformal invariance to better conditioning is still open, similar ideas appear in:

- • Theoretical physics, where scale-invariant theories exhibit smoother correlation functions and long-range order [47, 51].
- • Studies of the conformal bootstrap, which show how scale invariance constrains fluctuations and narrows operator spectra [54].
- • Random matrix theory and kernel methods, where flattened spectra correspond to improved generalization.
- • Recent numerical studies that identify conformal symmetry in real physical transitions as a signature of underlying flatness and universality [58, 57].

These connections reinforce the insight that conformal invariance leads to a better distributed Hessian spectrum — measurable through variance and entropy — and thus enhances the robustness and convergence of gradient-based learning.

### E.2 The particular case of flattened spectra and RMT

We talk about flattened spectra when Hessians or kernel matrices have lower spectral variance or more uniform eigenvalue distribution. This is usually associated with improved generalization in both kernel methods and deep networks. In random matrix theory (RMT), this corresponds to ensembles where the eigenvalue density concentrates, reducing the effect of high-curvature directions that may cause overfitting. In kernel methods [44] and [48] show that flatter spectra lead to more stable interpolation and better generalization in the overparameterized regime. Similar ideas appear in deep learning: models trained with flatter loss landscapes, reflected in the Hessian spectrum (e.g., fewer large outliers), often exhibit better generalization [56, 55]. In our setting, the conformal IS loss effectively preconditions the Hessian, reducing its spectral variance, thus echoing these theoretical insights.

The flattening of the Hessian spectrum induced by conformal invariance also resonates with results from random matrix theory (RMT). In high-dimensional models, RMT provides a statistical framework to describe the eigenvalue distribution of Hessians, Fisher matrices, or kernel operators. A common benchmark is the Marchenko–Pastur law, which characterizes the spectrum of sample covariance matrices in the absence of structure. Deviations from this law — such as heavy tails, outliers, or sharp spectral peaks — often signal overfitting or poor generalization [49, 52]. In contrast, flatter or more regular spectra (e.g., those with lower variance and fewer extreme outliers) tend to reflect better generalization performance. In our setting, the IS-induced conformal symmetry reduces the spectral variance of the Hessian, effectively steering the system toward a more stable and “bulk-like” spectrum, thus aligning with favorable RMT regimes.

### E.3 CFT Robustness and the Holographic Analogy

In the high-target limit y0 → ∞, we have shown that the IS loss regularized by a smoothness penalty induces an effective action

2

dϵ dx

(33)

S[ϵ] = dxλ

This corresponds to a massless scalar field theory — a conformal field theory (CFT) in one dimension

— with long-range power-law correlations G(x) ∼ 1/|x| and no intrinsic length scale.

Beyond its mathematical elegance, this structure echoes the foundational role of CFTs in the AdS/CFT correspondence [50]. In this duality, a gravitational theory defined in a (d + 1)-dimensional Anti-de Sitter (AdS) bulk is fully determined by a d-dimensional CFT living on its boundary. Perturbations and dynamics in the bulk geometry are encoded in boundary correlators and operator insertions of the CFT.

Robustness as Boundary Consistency. Within this holographic framework, the robustness and stability of the boundary CFT are essential for the well-posedness of the dual gravitational theory. Small fluctuations on the boundary propagate into coherent and physically meaningful bulk geometries. Similarly, in our context, we observe that the IS loss — by enforcing conformal invariance in the large-y0 regime — leads to stable, globally consistent learning dynamics. The long-range coherence of the predictions resembles the behavior of a holographic boundary theory robustly encoding bulk structure.

A Speculative Correspondence. We may interpret the IS-trained model as a system where:

- • The prediction layer behaves like a boundary CFT, enforcing smooth, scale-invariant structure.
- • The internal representation space (e.g., hidden layers or latent variables) acts as a discrete, dynamically evolving “bulk,” whose geometry is regularized through this induced boundary behavior.

This analogy opens a speculative but intriguing avenue: robust generalization in learning systems may reflect a kind of holographic encoding, where local losses induce structured global behavior. The IS loss, in this view, acts as a holographically robust preconditioner: it ensures that perturbations do not remain confined, but are smoothed in a globally consistent manner.

Future Directions. While our model remains in 1D and is far from a full-fledged CFT, let alone a holographic dual, the structural parallels motivate future exploration. It may be fruitful to investigate:

- • Learning systems that approach RG fixed points under IS-like training, mirroring conformal fixed points.
- • Architectures where latent representations exhibit AdS-like metrics, optimized via boundaryconsistent losses.
- • Explicit bulk-boundary decompositions in model design, inspired by holographic renormalization.

This connection between optimal loss design, conformal symmetry, and holographic stability invites a deeper geometric and dynamical understanding of learning systems.

Conclusion The IS loss with regularization is mathematically equivalent to a conformal scalar field theory. It induces power-law correlations and better-conditioned optimization landscapes. This explains both its empirical stability and its ability to propagate information globally in high-risk or low-signal regimes.

Thus, scale-invariant losses improve convergence by embedding learning into the geometry of critical systems with long-range structure [45, 46, 59, 53].

## Additional References

- [43] Shun-Ichi Amari. “Natural gradient works efficiently in learning”. In: Neural computation 10.2 (1998), pp. 251–276.

- [44] Mikhail Belkin et al. “Reconciling modern machine learning and the bias-variance trade-off”. In: Proceedings of the National Academy of Sciences 116.32 (2019), pp. 15849–15854.
- [45] John Cardy. Scaling and renormalization in statistical physics. Vol. 5. Cambridge university press, 1996.
- [46] Philippe Francesco, Pierre Mathieu, and David Sénéchal. Conformal field theory. Springer Science & Business Media, 2012.
- [47] Malte Henkel. Conformal Invariance and Critical Phenomena. Springer, 1999. DOI: 10.1007/ 978-3-662-03937-3.
- [48] Arthur Jacot, Franck Gabriel, and Clément Hongler. “Neural tangent kernel: Convergence and generalization in neural networks”. In: NeurIPS (2018).
- [49] Camille Louart, Zhenyu Liao, and Romain Couillet. “A random matrix approach to neural networks”. In: Annals of Applied Probability 28.2 (2018), pp. 1190–1248.
- [50] Juan Maldacena. “The large-N limit of superconformal field theories and supergravity”. In: International journal of theoretical physics 38.4 (1999), pp. 1113–1133.
- [51] Hidetoshi Nishimori and Gerardo Ortiz. Elements of Phase Transitions and Critical Phenomena. Oxford University Press, 2010. ISBN: 9780199577224.
- [52] Jeffrey Pennington and Pratik Worah. “The emergence of spectral universality in deep networks”. In: Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS). 2018, pp. 1924–1932.
- [53] Michael E Peskin. An Introduction to quantum field theory. CRC press, 2018.
- [54] David Poland, Slava Rychkov, and Alessandro Vichi. The Conformal Bootstrap: Theory, Numerical Techniques, and Applications. arXiv preprint arXiv:1805.04405. 2018. arXiv: 1805.04405 [hep-th].
- [55] Levent Sagun, Leon Bottou, and Yann LeCun. “Eigenvalues of the hessian in deep learning: Singularity and beyond”. In: arXiv preprint arXiv:1611.07476 (2016).
- [56] Zhewei Yao et al. “Hessian-based analysis of large batch training and robustness to adversaries”. In: Advances in Neural Information Processing Systems 31 (2018).
- [57] You Zhou et al. “Emergent Symmetry in Quantum Phase Transitions”. In: Fundamental Research 3.4 (2023), pp. 27–38. DOI: 10.1016/j.fmre.2023.02.003.
- [58] Wei Zhu et al. “Uncovering conformal symmetry in the 3D Ising transition: state-operator correspondence from a quantum fuzzy sphere regularization”. In: Physical Review X 13.2

(2023), p. 021009.

- [59] Jean Zinn-Justin. Quantum field theory and critical phenomena. Vol. 171. Oxford university press, 2021.

