# arXiv:2509.04419v2[cs.LG]20Jan2026

## Towards a Unified View of Large Language Model Post-Training

Xingtai Lv1∗, Yuxin Zuo1∗, Youbang Sun1†, Hongyi Liu1, Yuntian Wei1, Zhekai Chen1, Xuekai Zhu1, Kaiyan Zhang1, Bingning Wang3, Ning Ding1,2†, Bowen Zhou1,2† 1Tsinghua University, 2Shanghai AI Laboratory, 3WeChat AI

Code: TsinghuaC3I/Unify-Post-Training Mail: lvxt24@mails.tsinghua.edu.cn

#### Abstract

Two major sources of training data exist for post-training modern language models: online (model-generated rollouts) data, and offline (human or other-model demonstrations) data. These two types of data are typically used by approaches like Reinforcement Learning (RL) and Supervised FineTuning (SFT), respectively. In this paper, we show that these approaches are not in contradiction, but are instances of a single optimization process. We derive a Unified Policy Gradient Estimator, and present the calculations of a wide spectrum of post-training approaches as the gradient of a common objective under different data distribution assumptions and various bias-variance tradeoffs. The gradient estimator is constructed with four interchangeable parts: stabilization mask, reference policy denominator, advantage estimate, and likelihood gradient. Motivated by our theoretical findings, we propose Hybrid Post-Training (HPT), an algorithm that dynamically selects different training signals. HPT is designed to yield both effective exploitation of demonstration and stable exploration without sacrificing learned reasoning patterns. We provide extensive experiments and ablation studies to verify the effectiveness of our unified theoretical framework and HPT. Across six mathematical reasoning benchmarks and two out-of-distribution suites, HPT consistently surpasses strong baselines across models of varying scales and families.

Stabilization Mask Reference Policy

Algorithms SFT

𝟙

Data Source Online Rollout Offline Demo

PPO

Clip Diff

𝟙

Advantage Estimate

Advantage Estimation

Likelihood Gradient

Figure 1: Illustration of the Unified Policy Gradient Estimator. The “∇” in the background of the Likelihood Gradient part refers to the calculation of the gradient with respect to the πθ.

∗ Equal Contributions. † Corresponding Authors.

#### Contents

- 1 Introduction 3
- 2 A Unified View on Post-Training Algorithms 4

- 2.1 Components of the Unified Policy Gradient Estimator . . . . . . . . . . . . . 4
- 2.2 Derivation of the Unified Policy Gradient Estimator . . . . . . . . . . . . . . 5
- 2.3 Gradient Component Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.4 Hybrid Post-Training with Performance Feedback . . . . . . . . . . . . . . . 8

- 3 Experiments 9

- 3.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 4 Empirical Analysis 11

- 4.1 Exploration and Exploitation . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.2 Training Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 4.3 Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.4 Impact of Off-policy RL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 4.5 Gate Threshold Ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 5 Conclusion 16

- A Gradient Derivation for Classical Algorithms 20

- A.1 Gradient of SFT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- A.2 Gradient of Online RL: PPO, GRPO and Beyond . . . . . . . . . . . . . . . . 20
- A.3 Gradient of Offline RL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- B Additional Theoretical Details for Section 2.2 21

- B.1 Deriving Equation 2 from Equation 1 . . . . . . . . . . . . . . . . . . . . . . . 21
- B.2 Extension: Adding a Trust-Region Regularizer . . . . . . . . . . . . . . . . . 22
- B.3 PPO Clipping and the Stabilization Mask . . . . . . . . . . . . . . . . . . . . 22

#### 1 Introduction

Reinforcement Learning has played an integral role in enhancing the reasoning capabilities of large language models (LLMs) (Jaech et al., 2024; Team et al., 2025; Guo et al., 2025). RL allows the model to freely explore the reasoning space in the post-training process and improve its performance based on the feedback provided in the environment. However, applying Reinforcement Learning directly to a base model (i.e., “Zero RL”) (Zeng et al., 2025a) presupposes a certain level of inherent capability. This method often falters when applied to weaker models or tasks of high complexity, as the exploration process may fail to explore and discover meaningful reward signals. Conversely, the classical Supervised Fine-Tuning (SFT) (Wei et al., 2021) offers a direct and efficient method to distill knowledge from high-quality, human-annotated data, enabling models to rapidly and accurately fit the target distribution. Yet this approach often curtails the model’s exploratory capabilities, potentially leading to overfitting on the demonstration data and compromising its generalization performance on out-of-distribution inputs. Consequently, a sequential “SFT-then-RL” pipeline (Yoshihara et al., 2025) has emerged as the standard, adopted by numerous state-of-the-art open-source models. While effective, this multi-stage process, which first elevates the model’s capabilities through SFT before refining them with RL, is notoriously resource-intensive and usually requires careful tuning to ensure effectiveness.

To circumvent these challenges, recent works have focused on integrating SFT or SFT-style imitation learning losses directly with RL objectives (Yan et al., 2025; Fu et al., 2025; Zhang et al., 2025). In these approaches, the model is updated using a composite loss function. The balance between the imitation and exploration components is governed by various strategies, including a fixed coefficient, a predefined schedule, a dynamic adjustment based on entropy, or a learnable parameter. These works predominantly treat the SFT and RL losses as two distinct objectives. And a detailed analysis of why these two learning signals can be effectively combined within a unified optimization process remains largely unexplored.

Despite their distinct mathematical formulations, we find that the gradient calculations from these approaches can be viewed as a single, unified form. Inspired by Generalized Advantage Estimator (Schulman et al., 2015b), we introduce Unified Policy Gradient Estimator (UPGE), a framework that formally subsumes the gradients of various post-training objectives into one generalized expression. We provide analysis to show that the various forms of gradients are, in fact, not conflicting. Instead, they act as complementary learning signals that can jointly guide the optimization process. However, these gradient estimators possess different characteristics, and there exists a bias-variance tradeoff in their respective gradient components. Building upon this unified perspective, we propose Hybrid PostTraining (HPT), a hybrid algorithm to dynamically choose more desirable training signals by adapting a mixing ratio between the SFT and RL losses. This mechanism allows HPT to be intrinsically adaptive to models of varying capabilities and data of differing complexities.

We implement a simple instance of HPT, which adaptively switches between SFT and RL based on rollout accuracy, and empirically demonstrate that it achieves strong results. Our empirical evaluations demonstrate that HPT surpasses strong baselines such as SFT→GRPO and LUFFY with Qwen2.5-Math-7B, achieving a 7-point gain over our strongest baseline on AIME 2024. Moreover, HPT also yields substantial improvements even on relatively smaller and weaker models, including Qwen2.5-Math-1.5B and Llama3.1-8B. Through detailed training dynamics and illustrative training visualizations, we clearly reveal the features and underlying mechanisms of HPT. The following are several key takeaways:

Takeaways

- 1. UPGE provides a theoretical unification of a wide spectrum of post-training algorithms, covering both SFT and RL losses within a single formulation (§ 2).
- 2. HPT is capable of outperforming previous post-training and mixed-policy algorithms across a diverse range of models (§ 3).
- 3. Dynamic integration of SFT and RL in HPT achieves the highest Pass@1024, facilitating enhanced exploration and generalization of the model (§ 4.1).

Table 1: Theoretical unified view of various post-training algorithms.

Algorithm Reference Policy Advantage Estimate Unified Policy Gradient Estimator SFT πref = πθ AˆSFT ≡ 1 ∇JSFT(θ) = ∇πθ(τ) AˆπSFT=1

θ(τ)

Online Reinforcement Learning Methods PPO

AˆPPO1Clip πref (τ)

(Schulman et al., 2017) πref = πθold AˆPPO = GAE (Schulman et al., 2015b) ∇JPPO = ∇πθ(τ)

AˆGRPO1Clip πref (τ)

GRPO

(Shao et al., 2024) πref = πθold AˆGRPO = R(τj)std−mean({R(τ({R(τj)}Gon)

j)}Gon) ∇JGRPO = ∇πθ(τ)

REINFORCE

(Ahmadian et al., 2024) πref = πθ AˆREINFORCE = ±1 ∇JREF.(θ) = ∇πθ(τ)πAˆREF.

θ(τ)

CISPO

(Chen et al., 2025) πref = πθold AˆCISPO = AˆGRPO ∇JCISPO = ∇πθ(τ) AˆCISPOπ 1CIS-Mask

ref (τ)

1/|τi,j|

AˆGSPO1Seq-Clip πref (τ)

GSPO

(Zheng et al., 2025) πref = πθ ππθold(τi,j|qi)

AˆGSPO = AˆGRPO ∇JGSPO = ∇πθ(τ)

θ(τi,j|qi)

###### Offline/Online Reinforcement Learning Methods

R(τj)−mean({R(τj)}Gon∪Gof f )

SRFT (Offline) (Fu et al., 2025) πref ≡ 1 AˆSRFT =

std({R(τj)}Gon∪Goff ) ∇JSRFT = ∇πθ(τ)π AˆSRFT

ref (τ)=1

LUFFY (Offline) (Yan et al., 2025) πref ≡ 1 AˆLUFFY = AˆSRFT ∇JLUFFY = ∇πθ(τ)πAˆLUFFY

ref(τ)=1 fshape′

#### 2 A Unified View on Post-Training Algorithms

In this section, we adopt a unified perspective to understand both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) as post-training objectives. We present the gradient calculations of various post-training approaches in Table 1, with exact derivations of classical approaches presented in the Appendix A. From the table, it can be seen that policy gradient calculations for LLM post-training can be written in a unified policy gradient form.

Takeaways

We propose a unified framework for the gradient calculation of LLM post-training, named the Unified Policy Gradient Estimator:

1 πref

Aˆ∇πθ.

gradUni = 1stable

All gradient calculations can be written in the unified form.

In the following sections, we further show that the differences between different gradient calculations can be broken down into four distinct components. We theoretically derive the Unified Policy Gradient Estimator from a common objective and provide a detailed analysis of its gradient components. Based on this unified perspective, we then propose the Hybrid Post-Training (HPT) algorithm.

###### 2.1 Components of the Unified Policy Gradient Estimator

We present the Unified Policy Gradient Estimator, our unified framework for gradient calculations. In Table 1, we list a series of fundamental and well-studied post-training methods, divided into SFT and two types of RL processes. Apart from providing the closedform policy gradients of these methods, we also present the decomposition of these methods with detailed components. It can be seen that these seemingly different methods in fact share common components and that all gradients follow our proposed unified framework.

In this paper, we divide the unified gradient into four terms: stabilization mask, reference policy, advantage estimate, and likelihood gradient. We address each of the terms below.

Stabilization Mask 1stable Starting from PPO (Schulman et al., 2017), the stabilization mask was first derived as an approximation of the TRPO Algorithm (Schulman et al., 2015a). In practice, the PPO clipping addresses the instability issue during RL training by turning off the current gradient when the current iterate is considered unsafe. In consequent works

in Table 1, many have provided their modifications on the stability mask, usually motivated by empirical evaluations.

Reference Policy Denominator πref The second term in our unified estimator is the reference policy on the denominator. We note that our notion of reference policy differs

from the commonly used rollout policy πθold, for which we provide a discussion in Section

- 2.3. This denominator denotes a token-level reweight coefficient, usually in the form of an inverse probability. There are multiple choices for this coefficient. For the case of SFT, the

policy denominator uses the current policy πθ(τ). This is a result of L = − log(πθ(τ)) as the objective function. For the case of PPO-style online RL algorithms, generally, the policy

denominator uses the rollout policy πθold(τ). Due to the unavailability of πref(τ) in the offline demonstration dataset, most offline RL algorithms simply assume πref(τ) = 1 for the denominator.

Advantage Estimate Aˆ In traditional RL, the advantage evaluates the additional benefit of taking the current action given the current state. For the context of LLMs, most of the advantage estimation is sequence-level rather than token-level, and measures the quality of the current response sequence. Similar to traditional RL literature, the post-training process seeks to maximize the likelihood of generating positive sequences with high advantage and minimize negative sequences.

Likelihood Gradient ∇πθ(τ) The policy gradient term is a general term which maps gradient information from the actions to the model parameters θ. It is crucial for backpropagating the objective signals to the network weights, and is kept the same across all gradient calculations.

###### 2.2 Derivation of the Unified Policy Gradient Estimator

We begin from a simple and common objective shared by all post-training algorithms: improve the likelihood of positive trajectories and decrease the likelihood of negative trajectories such that the total reward in expectation maxθ J (θ) := E[r(τ|q)] is maximized. From this starting point, we theoretically derive our Unified Policy Gradient Estimator. We then show that SFT and RL objectives are not in conflict, and they can be optimized jointly within a single loss.

Common Objective. We model the post-training as a process to maximize the expected success rate while keeping the model policy closely adhering to a demonstration dataset (behavior policy) πβ:

Jµ(θ) = Eτ∼πθ(·|q) r(τ | q) − µ KL πβ(· | q) ∥ πθ(· | q) , µ ≥ 0, (1)

where q∼D denotes the question from a given distribution, τ denotes a trajectory, r denotes the (binary/real) score, and πβ denotes behavior policy from demonstration.

Gradient of the Common Objective. Differentiating and rearranging Equation 1 (full derivation in Appendix B.1), we obtain

∇θJµ(θ) = Eτ∼πθ r(τ | q) ∇θ log πθ(τ | q) + µ Eτ∼πβ ∇θ log πθ(τ | q) . (2)

From gradient to the Unified Policy Gradient Estimator. Applying the measure-change identity (detailed in Appendix B.1) with the reference policy πref which we mentioned in Section 2.1 and using ∇ log πθ = (1/πθ)∇πθ yields the gradient:

1 πref(τ | q)

Auni(τ, q) ∇θπθ(τ | q) , (3)

∇θJµ(θ) = Eτ∼πref(·|q)

with the unified advantage

πβ(τ | q) πθ(τ | q) ASFT(τ,q)

Auni(τ, q) = r(τ | q) ARL(τ,q)

. (4)

+ µ

In many RL works, the raw score r(τ | q) is replaced by a more structured advantage to reduce variance, provide relative credit assignment within a rollout group, and stabilize step sizes. For example, GRPO uses group-wise normalization:

R(τj) − mean({R(τ)}Gon) std({R(τ)}Gon)

AGRPO(τj, q) =

. (5)

When trust-region stabilization masks, as induced by PPO clipping, are inserted multiplicatively without altering the target objective, we obtain our Unified Policy Gradient Estimator:

1 πref(τ | q)

graduni = Eτ∼πref(·|q) 1stable(τ, q)

Auni(τ, q) ∇θπθ(τ | q)

(6)

1 πref

Aˆ ∇πθ.

= 1stable

The trust-region surrogate that produces the mask is given in Appendix B.3.

The gradient in (2) is the sum of two terms: (i) a reward + trust-region term sampled from πθ and (ii) a data-adherence (SFT) term sampled from πβ. Both terms map to the same estimator via (3)–(6) by choosing πref accordingly (e.g., πθold for on-policy trust-region updates and πβ for SFT/offline updates). Therefore, SFT and RL optimize a single Common Objective (1) and can be trained jointly within one loss without intrinsic conflict.

###### 2.3 Gradient Component Analysis

Takeaways

- 1. While all algorithms share the same Common Objective, bias-variance trade-offs still exist across current instances for different components of the unified gradient estimator.
- 2. We can improve the post-training process by constructing a better and more suitable estimation of the policy gradient.

Across the wide spectrum of algorithms contained in our previous discussions and Table 1, it can be inferred that the four components that construct the unified gradient estimator are motivated by different procedures in the post-training process. To better illustrate the relationship between the different processes with the respective components of our unified gradient, we present Figure 1.

We divide the post-training process of LLMs into the four steps shown in Figure 1: i) First, the LLM makes the decision on its data source, either to use data from an offline demonstration dataset, from self-generated rollout data, or a mixture of both. In this process, the policy likelihood πθ of the data with respect to the current LLM is generated. ii) Given the data source used for data generation, a reference policy πref is calculated. iii) After data collection is complete, the algorithm calculates the advantage estimation Aˆ for each token/sequence. iv) Lastly, the algorithm may choose to apply an additional masking procedure 1stable to disable the gradient calculation of various tokens, which could lead to theoretical or numerical stability issues. After these four steps, the components are collected to construct the policy gradient gradUni, which is used to update the LLM in the system. Similar to GAE presented in (Schulman et al., 2015b), multiple instantiations exist to estimate the policy gradient. However, different component selections introduce various degrees of bias and variance, where a trade-off is often encountered. We provide the following discussion on key components of the unified gradient below.

Reference Policy Calculation Practically speaking, the reference policy denominator places a weight on each token-level update such that any token with a smaller probability, often implying more significance, is weighted more. SFT and REINFORCE assign weights inversely proportional to the current policy πθ, enforcing a bigger update when the model outputs a small probability. On the other hand, when the data is generated with an outdated model, algorithms such as PPO assign weights inversely proportional to the rollout policy πθold, and offline RL does not assign additional weights for tokens.

Theoretically, the reference policy is usually set given the source of the dataset and/or the rollout policy. For online RL methods that train purely with on-policy data, such as

REINFORCE (Ahmadian et al., 2024), uses π1

, which produces an unbiased estimate for gradient calculation. However, these methods usually suffer from high variance. For PPOstyle online RL algorithms, the reference policy refers to the rollout policy, which is a result of importance sampling. PPO is a numerically simplified version of TRPO (Schulman et al., 2015a). PPO makes conservative updates that effectively reduce variance. However, the important sampling ratio is in fact theoretically ill-posed and could introduce systematic bias, as discussed in GSPO (Zheng et al., 2025). GSPO has also proposed a novel calculation for πref, as shown in Table 1. On the other hand, in the offline setting, the choice for reference policy πref is limited, since the algorithm generally has no access to the rollout policy. If we are given the assumption that the offline data evenly covers the entire state-action rollout

θ

space, then the importance sampling ratio r(θ) = ππθ(τ)

ref(τ) reduces to πθ(τ) by setting constant πref(τ) = 1. Notably, it is apparent that setting πref(τ) = 1 introduces much bias at the cost of numerical stability. For the SFT case, we can consider that the domain-specific dataset is generated with respect to the expert policy π∗; therefore, no weighted sampling is required. Neither of the two approaches is entirely theoretically justified, from an RL perspective; both require a lower bound on the state-action visitation of all the possible state-action pairs (Kakade, 2003), which can not be satisfied due to the severely limited datasets in practice.

Apart from the strong connection to data source and sampling polices, some studies employ a hand-crafted reweight factor within the reference policy denominator. These works (Yan et al., 2025; Zhang et al., 2025) typically find desirable token properties and purposefully place a higher/lower weight on these desirable/undesirable tokens, respectively.

Choice of Stabilization Mask The clipping operation introduced in PPO was the first to explicitly add a stop gradient operation on LLM post-training. Clipping gradient estimation where the importance sampling strays too far from 1 is an effective approach to address high variances. However, this aggressive clipping behavior has been criticized by some to be overly conservative: Both DAPO (Yu et al., 2025) and CISPO (Chen et al., 2025) stated that the classical PPO approach drops all the tokens corresponding to large model updates, and that many such tokens are in fact crucial for stabilizing entropy and facilitating scalable RL. DAPO presented a slight modification to the clipping threshold, and CISPO further extended the notion of token-wise mask, where more granular tuning was introduced to decide whether gradients from specific tokens should be dropped. The recent work of Cui et al. (2025b) has demonstrated that many existing algorithms negatively impact the output entropy during training and introduced Clip-Cov, adding another clipping mechanism to address the entropy-collapse encountered in training. While these methods demonstrated performance enhancements in practice, they also provide additional sources of bias.

On the other hand, works such as GSPO (Zheng et al., 2025) have stated that the PPO-style clipping is inherently noisy and inefficient for sample exploitation: GSPO clips a much larger fraction of tokens and yet demonstrated superior training efficiency.

In addition, post-training algorithms using offline data have chosen to purposefully remove the clipping from training, mostly guided by performance. Though setting πref(τ) = 1 as the policy denominator does effectively reduce the instability in gradient calculations.

Advantage Estimation There are two commonly used settings for estimating the sequencelevel advantage function: the fixed advantage setting and the adaptive advantage setting.

The fixed setting considers Aˆ = ±1 given the rule-based verification, which is adapted by REINFORCE and implicitly by SFT (where all sequences are positive samples). Alternatively, recent studies have focused on using adaptive advantage estimations, performing re-centering or normalization based on the performance of the current rollout group. Notably, GRPO and its variants, such as DAPO (Yu et al., 2025) and LUFFY (Yan et al., 2025), use unit normalization such that the advantage estimation of the group has a unit standard deviation. Other approaches, such as Dr. GRPO (Liu et al., 2025), RLOO (Ahmadian et al.,

- 2024), and REINFORCE++ (Hu et al., 2025a), claim that dividing the standard deviation introduces a difficulty bias and that only recentering is adequate.

Apart from sequence-level advantage estimate Aˆi,j, recent works (Wang et al., 2025; Yang et al., 2025; Sun et al., 2025) have also adapted a more granular token-level advantage

estimate Aˆi,j,t to a varying degree of success.

A Combination of Gradient Estimators Although bias-variance trade-offs exist for the gradient estimator, we state that, given data distribution assumptions and sufficient data samples, all policy gradient estimators covered in our framework should result in an effective direction of improvement for the Common Objective. To effectively reduce the variance and bias for each policy update, we can treat instances of policy gradient as different noisy measurements of the true policy gradient, and perform a weighted average to generate a more accurate gradient estimation, similar to complementary filters (Marantos et al., 2015).

However, the complexity of LLM RLVR introduces additional challenges. The current state of the behavior policy πθ and its relationship with the respective tasks also greatly impacts the bias-variance tradeoff of each instance of the gradient estimator. For instance, RL-zero is significantly more effective for the Qwen model series compared to LlaMA, but SFT is effective for both methods (Zeng et al., 2025a); SFT → RL and RL → SFT also yield significantly different results on the same LLM (Fu et al., 2025). We argue that for constructing a post-training algorithm with better effectiveness and efficiency, a dynamic and adaptive mechanism is crucial to construct optimal gradient components.

###### 2.4 Hybrid Post-Training with Performance Feedback

Our unified perspective above shows that different post-training losses have the same optimization objective with different characteristics. Inspired by this view, we propose the Hybrid Post-Training (HPT) algorithm. We use a mixed loss L = αLRL + βLSFT, which contains the weighted on-policy RL loss LRL and SFT loss LSFT, to optimize the target LLM πθ. The weights of the two losses (α and β) are determined by the real-time sampling performance of the model.

Performance on Single Question. For any question q provided to the LLM, we first obtain both a supervising trajectory τ⋆ and the model’s performance P on the question. Specifically,

we draw n on-policy trajectories {τi}in=1 ∼ πθ(· | q) and evaluate them with a verifier v : τi → {0,1}. This verifier is the same as the rule-based reward function and the model’s performance P is defined as the mean of these n verification scores:

v(τi) = R(τi) =

1 if τi contains the correct answer of q 0 otherwise

(7)

n

1 n

### ∑

v(τi) (8) Intuitively, P indicates how well the current policy performs on q across multiple trajectories. Feedback Coefficients. Then, we obtain the coefficients of on-policy RL loss α and SFT loss β based on the performance feedback:

P =

i=1

α = f(P), β = g(P), (9)

Algorithm 1 The Hybrid Post-Training (HPT) Algorithm Input: Pretrained LLM (policy) πθ; SFT dataset DSFT = {(q, τ⋆)} with supervising trajec-

tories τ⋆; verifier v; on-policy samples number n; total training steps T; feedback functions f and g; learning rate η

Output: Fine-tuned policy πθ∗. for t = 1 to T do

##### for i = 1 to n do

Sample trajectory τi ∼ πθ(· | q) Evaluate with verifier (rule-based reward): v(τi) ← R(τi) ∈ {0,1}

end P ← n1 ∑in=1 v(τi) α ← f(P), β ← g(P) # Performance feedback on question q Compute on-policy RL loss LRL using rollouts {τi} and normalized advantages derived

from {R(τi)}. Compute SFT loss LSFT on the supervising trajectory τ⋆. L ← α LRL + β LSFT # Mixed loss with performance feedback coefficients θ ← θ − η ∇θL

end return πθ∗

where the f and g are the specific feedback functions. Experientially, when the model demonstrates strong capability, it is advantageous to emphasize on-policy RL to foster exploration; conversely, when the model’s competence is limited, SFT should take precedence to ensure correct guidance. Consequently, f ought to be positively correlated with P, whereas g should exhibit a negative correlation. In this paper, we employ a pair of simple yet empirically effective switch functions f and g:

1 if P > γ 0 if P ≤ γ

1 if P ≤ γ 0 if P > γ

, β = g(P) =

(10)

α = f(P) =

The switch gate γ enables the model to perform SFT when its performance falls below a predefined threshold, and RL otherwise.

Mixed Loss. Finally, we calculate the RL loss LRL with the already generated n on-policy trajectories τi and SFT loss LSFT with the supervising trajectory τ⋆, and we use Dr. GRPO as the on-policy RL algorithm:

|τi|

n

1 n

### ∑

### ∑

min ri,t Ai,t, clip ri,t, 1 − ϵ, 1 + ϵ Ai,t (11)

LRL = −

t=1

i=1

|τ⋆|

1 |τ⋆|

### ∑

log πθ(τt⋆ | q, τ<⋆t) (12)

LSFT = −

t=1

(τi,t|q,τi,<t)

where ri,t = πθ

is the per-token importance sampling ratio, Ai,t ≡ Ai = R(τi)−mean({ R(τi) | i=1,2,...,n})

πθold(τi,t|q,τi,<t)

std({R(τi) | i=1,2,...,n}) is the advantage and ϵ is the clip gate hyperparameter. The mixed loss is then obtained by taking a weighted average of these two losses using performance feedback coefficients α and β:

L = αLRL + βLSFT (13)

#### 3 Experiments

###### 3.1 Experimental Setup

Models To evaluate the generalizability of HPT across different backbone models, we conduct experiments using Qwen and LLaMA models of various scales. The models we experiment with are as follows:

- Table 2: In-distribution and out-of-distribution performance of HPT and baselines on Qwen2.5-Math7B. ∗ means the results are taken from the corresponding paper.

In-Distribution Out-of-Distribution

Model

AIME 24 AIME 25 AMC MATH-500 Minerva Olympiad Avg ARC-c GPQA Avg Qwen2.5-Math-7B 12.3 4.7 33.0 43.6 8.8 13.6 19.3 30.9 28.3 29.6

SFT 25.1 22.8 56.1 84.2 33.8 44.7 44.5 67.4 25.3 46.4 GRPO 19.4 13.8 59.1 81.8 38.2 46.2 43.1 81.2 36.4 58.8

SFT → GRPO 25.7 21.6 62.2 84.6 38.2 46.8 46.5 67.7 30.8 49.3 LUFFY 26.1 21.8 66.2 88.4 41.9 54.1 49.8 80.8 39.4 60.1 SRFT 18.4 15.5 55.9 83.8 42.6 48.9 44.2 80.5 36.8 58.7 HPT 33.0 21.9 69.4 89.2 46.0 56.9 52.7 81.6 42.9 62.3

Qwen2.5-Math-7B-Ins. 11.8 9.8 48.3 83.2 34.2 39.3 37.8 72.7 29.3 51.0 PRIME-Zero∗ 17.0 12.8 54.0 81.4 39.0 40.3 40.8 73.3 18.2 45.8 SimpleRL-Zero∗ 27.0 6.8 54.9 76.0 25.0 34.7 37.4 30.2 23.2 26.7 OpenReasoner-Zero∗ 16.5 15.0 52.1 82.4 33.1 47.1 41.0 66.2 29.8 48.0 Oat-Zero∗ 33.4 11.9 61.2 78.0 34.6 43.4 43.8 70.1 23.7 46.9

- • Qwen Family: Qwen2.5-Math-1.5B, Qwen2.5-Math-7B (Yang et al., 2024);
- • LLaMA Family: LLaMA-3.1-8B (Grattafiori et al., 2024);

Benchmarks We evaluate HPT on 6 mathematical reasoning benchmarks: AIME 2024 (Li

- et al., 2024), AIME 2025 (Li et al., 2024), AMC (Li et al., 2024), MATH-500 (Hendrycks et al., 2021), Minerva (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). AMC (Li et al., 2024) comprises problems drawn from the AMC12 2022 and AMC12 2023 examinations. Moreover, when employing Qwen2.5-Math-7B as the backbone, we further conduct evaluations on GPQA-Diamond (Rein et al., 2024), a challenging and high-quality subset of the Graduate-Level Google-Proof Question Answering benchmark, as well as on ARC-c (Clark et al., 2018), an open-domain reasoning benchmark.

Evaluation Setup We set the maximum generation length to 8, 192 tokens, unless otherwise specified. For the main experiments, following DeepSeek-R1 (Guo et al., 2025), we adopt the Pass@k evaluation protocol (Chen et al., 2021) and report Pass@1 using non-zero temperature sampling. To ensure a fair comparison with previous works (Yan et al., 2025; Fu et al.,

- 2025), we compute avg@32 for AIME 24, AIME 25, and AMC (avg@1 for others) using a temperature of 0.6 and a top-p value of 0.95 for accuracy calculation.

Baselines Since HPT dynamically integrates GRPO (Shao et al., 2024) and SFT, the most natural baselines are SFT and GRPO individually. Furthermore, we compare HPT against the mix-policy approach LUFFY (Yan et al., 2025). For experiments using Qwen2.5-Math7B as the backbone, we additionally include SFT→GRPO and SRFT (Fu et al., 2025) as a baseline, as well as models trained with the Zero-RL procedure on the same backbone for a more comprehensive comparison. We also use PRIME-Zero (Cui et al., 2025a), SimpleRLZero (Zeng et al., 2025b), OpenReasoner-Zero (Hu et al., 2025b) and Oat-Zero (Liu et al., 2025) as baselines.

Implementation Details We apply GRPO (Shao et al., 2024) as the RL algorithm to implement HPT. We introduce a gating mechanism that adaptively assigns the coefficients α and β to the RL loss and the SFT loss based on the rollout performance, respectively. Formally, the gating mechanism is defined as:

(α, β) =

- (0,1), if P ≤ γ,
- (1,0), if P > γ,

where P denotes model’s performance as introduced in Section 2.4 and γ is the gate threshold. We fix γ at 0 throughout all experiments on the Qwen Family models and 2/8 for LLaMA, and provide relative ablation studies in Section 4.5. For hyperparameters, we use a constant learning rate of 5×10−6 and adopt the AdamW optimizer for the policy model. For

The results of SRFT are based on our own implementation, as the official code is not public.

- Table 3: Performance of HPT and baselines on LaMA3.1-8B and Qwen2.5-Math-1.5B. ∗ means the results are taken from the LUFFY paper (Yan et al., 2025).

Model AIME 24 AIME 25 AMC MATH-500 Minerva Olympiad Avg LLaMA3.1-8B 0.4 0.1 4.7 13.8 4.8 3.9 4.6

SFT∗ 0.5 0.1 5.4 20.2 4.0 5.3 5.9 GRPO∗ 0.3 0.5 9.4 23.4 17.6 6.1 9.6 LUFFY∗ 1.9 0.1 13.5 39.0 15.1 9.6 13.2

HPT 2.1 1.2 18.6 47.8 18.8 20.4 18.2 Qwen2.5-Math-1.5B 2.8 6.1 24.5 32.8 11.0 16.4 15.6 SFT 14.7 17.6 45.4 78.4 29.4 35.7 36.9 GRPO 12.2 8.5 43.8 71.0 33.1 35.3 34.0 LUFFY 14.1 9.4 43.5 75.2 26.1 39.7 34.7 HPT 16.6 17.8 51.0 81.0 37.5 47.3 41.9

rollout, we sample 8 responses using a temperature of 1.0. The maximum generation length is set to 8,192 tokens for all other models. For other details that may not have been explicitly introduced, we have endeavored to follow previous works as closely as possible (Zhao et al., 2025; Zuo et al., 2025). All experiments were conducted on 8 x NVIDIA A800 80GB GPUs.

- 3.2 Main Results

Table 2 presents the overall performance of HPT on Qwen2.5-Math-7B. As introduced in Section 2.4, in our implementation of HPT, the coefficients of the RL and SFT loss terms are both degraded and simplified into a binary form. Despite this highly streamlined experimental setup, HPT still yields substantial performance gains. It not only significantly outperforms both SFT-only and GRPO-only baselines, but also surpasses SFT→GRPO, which requires substantially higher computational cost. This suggests that simply concatenating the two training stages is not the most effective strategy. Moreover, HPT achieves marked improvements over existing mixed-policy approaches such as LUFFY and SRFT, with particularly notable gains of 6.9 and 14.6 points on AIME 2024, respectively. Furthermore, we conduct experiments on models of different scales and families to evaluate the effectiveness of HPT, including LLaMA3.1-8B and Qwen2.5-Math-1.5B, as shown in Table 3. Compared with SFT, GRPO, and LUFFY, HPT achieves substantial performance gains.

4 Empirical Analysis

Our empirical analysis progressively reveals how HPT reconciles exploration and exploitation, stabilizes training, and ultimately enhances the reasoning ability. We begin in § 4.1 with an examination of exploration and exploitation. In § 4.2, we provide a training visualization, contrasting HPT with the conventional SFT→GRPO. Next, § 4.3 investigates fine-grained training metrics of HPT. Building on this, § 4.4 explores the role of off-policy RL, testing whether alternative strategies for utilizing offline data yield benefits. Finally, § 4.5 presents a gate threshold ablation study.

- 4.1 Exploration and Exploitation

HPT inherently achieves an adaptive switching between RL and SFT. These two paradigms naturally correspond to the learning modes of exploration and exploitation. Accordingly, we can examine whether HPT addresses the initial challenges from both perspectives.

Exploration From the exploration perspective, we want to analyze the model’s Pass@k performance after training with HPT. Recently, Limit-of-RLVR (Yue et al., 2025) demonstrated that while RLVR training yields a significant improvement in Pass@1, it does not lead to gains in large-k Pass@k. In other words, RLVR does not expand the capability boundary of the base model. This finding has sparked broad discussions regarding the relationship between a model’s exploratory capacity and its Pass@k performance. Moreover, Pass@k has increasingly been recognized as a widely accepted metric for evaluating both the upper bound of model capability and its exploration ability. We follow Yue et al. (2025) to evaluate

HPT LUFFY GRPO SFT SFT GRPO Base Model

AIME25

###### AIME24

AMC

100

80

80

Pass@k(%)

60

80

60

40

60

40

20

20

40

2 2¹ 2² 2³ 2 2 2 2 2 2 2¹

2 2¹ 2² 2³ 2 2 2 2 2 2 2¹

2 2¹ 2² 2³ 2 2 2 2 2 2 2¹

Sampling Number k

- Figure 2: Pass@k performance of HPT against baselines on Qwen2.5-Math-7B. The evaluation spans

- 3 benchmarks, with Pass@k values estimated via bootstrap sampling from a set of 2048 generated solutions per problem.

Pass@k up to 1024 for each problem of AIME25, AIME24, and AMC for Pass@k evaluation. Based on these sets of generated solutions, we apply bootstrap sampling to obtain accurate estimates of Pass@k scores for various values of k. Figure 2 illustrates the resulting Pass@k curves, comparing HPT against baselines and the base model.

- • First, we can observe that methods incorporating SFT achieve higher large-k Pass@k compared to the GRPO (purely RL). This may be attributed to the introduction of data outside the model’s own distribution during SFT, which increases output uncertainty while also providing new knowledge from offline data, thereby enhancing the model’s exploratory capacity.
- • Furthermore, we identify an interesting phenomenon: since HPT dynamically integrates RL (GRPO) with SFT, we might intuitively expect its large-k Pass@k performance to fall between that of the two individual methods. However, HPT achieves the highest large-k Pass@k performance overall. This indicates that Hybrid Post-Training not only delivers substantial improvements in Pass@1, but also maximally preserves and enhances the model’s exploratory ability.

- Table 4: Bidirectional analysis of exclusive solves on MATH-500, comparing the Qwen2.5-Math-7B trained with HPT against baseline methods (GRPO and LUFFY). The notation +X/-Y in each cell indicates the performance trade-off: +X represents the number of problems solved by the HPT but not the baseline, while -Y represents the number solved by the baseline but not by the HPT.

Methods Level 1 Level 2 Level 3 Level 4 Level 5 Overall (N=43) (N=90) (N=105) (N=128) (N=134) (N=500)

GRPO Absolute +0/-0 +5/-1 +9/-2 +17/-4 +27/-8 +58/-15 Percentage +0.0%/-0.0% +5.6%/-1.1% +8.6%/-1.9% +13.3%/-3.1% +20.1%/-6.0% +11.6%/-3.0%

LUFFY Absolute +1/-0 +5/-1 +5/-3 +10/-5 +22/-7 +43/-16 Percentage +2.3%/-0.0% +5.6%/-1.1% +4.8%/-2.9% +7.8%/-3.9% +16.4%/-5.2% +8.6%/-3.2%

Exploitation From the exploitation perspective, the key question is whether our method, by leveraging SFT, enhances the model’s initial competence and facilitates subsequent RL training. As illustrated in Figure 3, RL training alone may fail to solve many problems (white line), requiring the dynamic intervention of SFT. To investigate this, we analyze its exclusive solves against the GRPO and LUFFY, building upon the results from the evaluation on MATH-500 with Qwen2.5-Math-7B as the backbone, as shown in Table 4. The red numbers denote problems that are solved by our method but not by GRPO or LUFFY, i.e., problems newly acquired through our training procedure. Three clear trends emerge from the analysis:

• First, the red counts consistently increase with problem difficulty, suggesting that HPT improves the model’s ability to tackle more challenging problems.

Performance

0 1/8 2/8 3/8 4/8 5/8 6/8 7/8 1

Level 3

Level 5

| | | | | | | | | | | | | | | |[Figure 1]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | |[Figure 2]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0 10 20 30 40 50

0 10 20 30 40 50

Epoch

Epoch

- Figure 3: GRPO training dynamics of SFT→GRPO on Qwen2.5-Math-1.5B across 50 training epochs. We visualize the model’s per-question sampling accuracy throughout the training process.

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | |[Figure 3]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0 10 20 30 40 50

Epoch

Level 3

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | |[Figure 4]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0 10 20 30 40 50

Epoch

Level 5

1 7/8 6/8 5/8 4/8 3/8 2/8 1/8 0 +1/8 +2/8 +3/8 +4/8 +5/8 +6/8 +7/8 +1

Performance Difference

- Figure 4: Performance difference (HPT v.s. SFT→GRPO) on Qwen2.5-Math-1.5B across 50 training epochs. A diverging color scale indicates the advantage: red for HPT, blue for SFT→GRPO, and white for no difference.

- • Second, the green counts within the red boxes remain essentially unchanged across settings: this indicates that, compared with existing methods, HPT preserves performance on problems that the model could already solve, thereby mitigating the risk of catastrophic forgetting.
- • Finally, the fact that the red counts are consistently large relative to both baselines demonstrates that our method enables the model to acquire a substantial number of problems that prior approaches struggled to solve.

###### 4.2 Training Visualization

To facilitate a fine-grained examination of the training process and thereby obtain deeper insights into how HPT works, we conduct a visualization analysis comparing the SFT→GRPO approach with HPT. We sample 255 problems from the MATH dataset (Hendrycks et al., 2021) for subsequent training, with 85 problems each from Levels 3, 4, and 5. For SFT→GRPO, we perform 50 epochs of GRPO on a Qwen2.5-Math-1.5B model fine-tuned with SFT, tracking rollout accuracy across training, as shown in Figure 3. We track the rollout accuracy for each sample throughout the entire training process. To highlight difficulty effects, we focus on Levels 3 and 5 as representative cases. The left subplot shows Level 3 (easier) problems, and the right shows Level 5 (hardest). Notably, GRPO frequently produces dense white regions, and sometimes even continuous white lines, reflecting widespread rollout errors across outputs. This illustrates a core limitation of RL methods: they struggle to learn effectively when frequent rollout errors occur across all outputs.

In parallel, we train Qwen2.5-Math-1.5B from scratch for 50 epochs to visualize HPT. To compare against SFT→GRPO and enable a more intuitive comparison, we conduct a differential analysis of the training dynamics. Specifically, we calculate the accuracy difference at corresponding positions (at matched prompts and steps) in the evaluation grid between two methods: red indicates HPT is better, blue the opposite. Figure 4 presents the results of the difference plots. Notably, SFT→GRPO actually requires greater computational resources than HPT: it involves a preceding SFT phase, and our approach also reduces computational costs during the transition from GRPO to SFT, as expensive operations such as rollouts are no longer required. This unfair comparison leads to an initial dominance of the blue regions, which is expected since the SFT stage in SFT→GRPO has already incorporated substantial prior knowledge. However, in the later stages of training, HPT still surpasses and ultimately reveals the dominance of the red regions, indicating that HPT consistently outperforms SFT→GRPO by substantially enhancing learning performance on the training set. This advantage becomes even more pronounced in the Level 5 subplot, suggesting that HPT provides particular benefits for learning on more challenging problems, which may be attributed to its use of question-level rollout performance as feedback.

###### 4.3 Training Dynamics

In this section, we investigate the training dynamics of HPT, focusing on validation performance, entropy, response length, and the offline data ratio. Our analysis centers on two aspects: whether HPT enables the model to acquire knowledge from offline data when its initial capabilities are limited, and whether its performance can be further enhanced through continued exploration with reinforcement learning.

GRPO LUFFY HPT

AIME24

###### AMC

MATH-500

| || |
|---|
|| |
|---|
| || |
|---|
|
|---|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
| | | |
| | | | | |
| | | | | |

| | | | || |
|---|
|
|---|---|---|---|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
| | | |
|| |
|---|
<br><br>| |
|---|
| | | | |
| | | | | |

80

ValidationAcc.(%)

60

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

17.5

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
| |

| |
|---|
| |

| |
|---|

| |
|---|

70

| |
|---|
| |

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

15.0

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

50

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

| |
|---|

60

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

12.5

| |
|---|
| |

50

40

10.0

| |
|---|

| |
|---|

40

| |
|---|
| |

7.5

30

30

5.0

| |
|---|

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Steps

Steps

Steps

Figure 5: Validation performance comparisons on Qwen2.5-Math-1.5B across benchmarks.

Validation Performance. We track the validation performance on the Qwen2.5-Math-1.5B

- as shown in Figure 5, where HPT consistently outperforms the baselines and delivers stable improvements across multiple benchmarks.

0 100 200 300 400 500

Steps

0.00

0.05

0.10

0.15

0.20

0.25

OfflineDataRatio

HPT (Qwen2.5-Math-7B)

HPT (Qwen2.5-Math-1.5B)

LUFFY (1/8)

Figure 6: Dynamic offline data ratio dynamics during training. The offline data ratio is calculated as the proportion of offline training samples relative to the total training data at each step.

Offline Data Ratio. We begin by quantifying the fraction of prompts whose gradients update the model through the SFT loss versus the RL loss at each training step, as shown in Figure 6. The offline data ratio is defined as the proportion of offline samples relative to the total number of training samples in each batch, with online samples calculated based on the remaining batch capacity. As expected, when the model has not yet acquired competence on the target tasks, the early phase is characterized by a large proportion of SFT-driven updates. As training progresses and the model’s onpolicy reward increases, the mixture gradually shifts: the contribution of RL grows while that of SFT diminishes, eventually stabilizing

- at a small but non-zero level. This trend is observed for both Qwen2.5-Math-7B and Qwen2.5-

Math-1.5B. The weaker 1.5B model remains in the SFT-dominated regime for a longer period before transitioning, whereas the stronger 7B model shifts earlier. These results align with our technical analysis of the design of HPT, where the mixing ratio is automatically adjusted based on performance rather than fixed in advance like LUFFY.

- 0.4

0.6

0.8

1.0

1.2

1.4

Entropy

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 100 200 300 400 500

Steps

1000

2000

3000

4000

5000

6000

ResponseLength

GRPO LUFFY HPT

- Figure 7: Comparisons of training dynamics across different methods: (left) The entropy measures the diversity of model outputs, indicating exploration behavior; (right) The response length tracks the average length of generated responses.

Entropy and exploration. Figure 7 (left) tracks token-level entropy over 500 steps. HPT maintains higher entropy than GRPO throughout the training phases. This is expected as the offline SFT trajectories are derived from the external demonstration distribution, which consequently increases the diversity in the model’s outputs.

Response length and acquired reasoning patterns. Figure 7 (right) reports the average response length. Our offline SFT trajectories have a length of up to 8k tokens. Under HPT, the model’s response length increases quickly during the early steps but does not jump to the 8k ceiling. More importantly, after the method shifts toward RL and the SFT proportion plateaus at a low level, the response length does not regress. This persistence suggests that the model has internalized long-form reasoning routines from the offline data rather than merely echoing teacher outputs. In other words, the learned reasoning pattern becomes part of the policy, and RL fine-tuning refines it instead of erasing it.

4.4 Impact of Off-policy RL

Table 5: Performance of different training paradigms to evaluate the impact of Off-policy RL. SFT/ON denotes SFT/On-policy (HPT), OFF/ON denotes Off-policy/On-policy, and Mix/ON denotes Mixpolicy/On-policy.

Name AIME 24 AIME 25 AMC MATH-500 Minerva Olympiad Avg OFF/ON 16.6 11.8 47.3 76.2 35.3 41.6 38.1 Mix/ON 16.7 17.2 46.9 79.4 37.5 43.9 40.3 SFT/ON 16.6 17.8 51.0 81.0 37.5 47.3 41.9

In our work, we have only made preliminary attempts at unifying post-training by integrating RL with SFT. However, off-policy RL represents an important training paradigm that emphasizes leveraging offline data. To this end, we further conduct experiments to investigate its influence and potential role.

We compare three different training paradigms: (1) SFT/On-policy, the model alternates between SFT and on-policy RL, which corresponds to the method we introduced above (HPT); (2) Off-policy/On-policy, the model alternates between off-policy RL and on-policy RL during training; and (3) Mix-policy/On-policy, the model combines the loss from SFT and off-policy RL, and dynamically switches it with the on-policy RL objective. For the Mix setting, we performed hyperparameter search and found the optimal SFT/OFF weighting ratio to be 1/10, i.e., the coefficients of the SFT loss and the off-policy loss are set to 0.1 and

- 1.0, respectively. We replicate the off-policy RL implementation described in LUFFY (Yan

0.2

0.0

0 100 200 300 400 500

Steps

- et al., 2025), and all experiments are conducted in the same settings to ensure fairness.

We evaluate the results of three methods on six math benchmarks. Table 5 presents results. Overall, the SFT/ON method achieves the best average performance (41.9), outperforming both Mix/ON (40.3) and OFF/ON (38.1). This suggests that off-policy RL may not be essential, as SFT already serves effectively as the training method of HPT for learning from offline data.

###### 4.5 Gate Threshold Ablation

Gate ( ) = 0 Gate ( ) = 1 Gate ( ) = 2

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

0.6

OfflineDataRatio

0.8

TrainingReward

0.5

0.6

0.4

0.3

0.4

0.2

0.2

0.1

0.0

0 100 200 300 400 500

0 100 200 300 400 500

Steps

Steps

- Figure 8: Training reward (left) and offline data ratio (right) comparisons across different gate settings on Qwen2.5-Math-1.5B.

In this section, we investigate the effect of different gate thresholds γ. A value of γ = 0 indicates that the model switches to SFT only when it fails all questions. Similarly, γ = 1/8 and γ = 2/8 correspond to settings where the model remains in on-policy reinforcement learning as long as it answers at least one or two out of eight questions correctly, respectively. To visualize the impact of the gating mechanism, we conduct experiments on the Qwen2.5Math-1.5B under three different gate settings. As shown in Figure 8, we analyze the training dynamics by tracking the dynamics of rewards and the proportion of offline data utilized throughout training, thereby highlighting how different gate thresholds mediate the balance between leveraging offline demonstrations and incorporating online feedback. We observe that, under different gate thresholds, varying degrees of engagement with offline data–based SFT learning emerge. A larger gate threshold introduces a greater extent of SFT based on offline data, as expected.

Table 6: Performance of HPT with different switch gate γ on Qwen2.5-Math-1.5B.

Name AIME 24 AIME 25 AMC MATH-500 Minerva Olympiad Avg γ = 2/8 15.8 13.0 49.0 77.6 34.6 44.1 39.0 γ = 1/8 18.1 14.2 46.0 75.4 35.7 42.5 38.7

γ = 0 16.6 17.8 51.0 81.0 37.5 47.3 41.9

To further compare the performance across different gating strategies, we evaluate the three trained models on six benchmarks. Table 6 presents the results. Among the three configurations, γ = 0 achieves the best overall performance with an average score of 41.9, outperforming both γ = 1/8 (38.7) and γ = 2/8 (39.0). This observation suggests that simply incorporating more SFT does not necessarily lead to better outcomes. Instead, it is crucial to maintain a dynamic balance between the exploration of RL and the exploitation of SFT. The optimal degree of this gating mechanism should be adjusted according to the characteristics of the base model and the specific training data employed.

#### 5 Conclusion

In this paper, we introduce the Unified Policy Gradient Estimator to provide a theoretical framework for LLM post-training. We demonstrate that SFT and RL optimize a common objective, with their respective gradients representing different bias-variance tradeoffs. Motivated by this unified perspective, we propose Hybrid Post-Training (HPT), an algorithm

that dynamically adapts between SFT for exploitation and RL for exploration based on real-time performance feedback. Extensive empirical validation shows that HPT consistently outperforms strong baselines, including sequential and static mixed-policy methods, across various models and benchmarks. Our work contributes both a unifying theoretical perspective on post-training and a practical algorithm that effectively balances exploitation and exploration to enhance model capabilities.

#### References

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨ un,¨ and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards, 2025a.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. Srft: A single-stage method with supervised and reinforcement fine-tuning for reasoning. arXiv preprint arXiv:2506.19767, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jian Hu, Jason Klein Liu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025a.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025b.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Sham Machandranath Kakade. On the sample complexity of reinforcement learning. University of London, University College London (United Kingdom), 2003.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Panos Marantos, Yannis Koveos, and Kostas J Kyriakopoulos. Uav state estimation using adaptive complementary filters. IEEE Transactions on Control Systems Technology, 24(4): 1214–1226, 2015.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889–1897. PMLR, 2015a.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015b.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Wei Sun, Wen Yang, Pu Jian, Qianlong Du, Fuwei Cui, Shuo Ren, and Jiajun Zhang. Ktae: A model-free algorithm to key-tokens advantage estimation in mathematical reasoning. arXiv preprint arXiv:2505.16826, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation. arXiv preprint arXiv:1805.01954, 2018.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, and Jing Tang. Treerpo: Tree relative policy optimization. arXiv preprint arXiv:2506.05183, 2025.

Hiroshi Yoshihara, Taiki Yamaguchi, and Yuichi Inoue. A practical two-stage recipe for mathematical llms: Maximizing accuracy with sft and efficiency with reinforcement learning. arXiv preprint arXiv:2507.08267, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. In Second Conference on Language Modeling, 2025a.

Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. HKUST-NLP Blog, 2025b.

Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. On-policy rl meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. arXiv preprint arXiv:2508.11408, 2025.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

#### A Gradient Derivation for Classical Algorithms

###### A.1 Gradient of SFT

We first consider the SFT process as a warm-up. As mentioned in the previous section, SFT takes a pre-trained foundation model and further makes the model more specialized by training its output prediction distribution to align with domain-specific data. The fine-tuning process uses the same cross-entropy loss as in model pre-training, defined as follows,

|τi|

N

∑

∑

log πθ(τi,t|qi, τi,<t). (14)

LSFT(θ) = −

t=1

i=1

where DSFT = {(qi, τi)}i∈[N] denotes the SFT dataset consisting of N question and trajectory pairs. τt denotes the t-th token in the trajectory and τ<t denotes all the tokens prior to τt.

For any t, the LLM outputs the next-token prediction as a probability distribution. In the context of RL, such a probability distribution has been commonly considered as a stochastic policy. Then, the gradient calculation of SFT can be obtained by directly taking the derivative of Equation (14) and takes the following form:

N

∑

∇JSFT(θ) = −∇LSFT(θ) =

i=1

|τi|

1 πθ(τi,t|qi, τi,<t)

∑

∇πθ(τi,t|qi, τi,<t)

. (15)

t=1

In this section, we slightly abuse the notion of policy gradient and consider the SFT as a case of behavioral cloning (BC) (Torabi et al., 2018), and Equation (15) can be seen as a specific form of policy gradient.

###### A.2 Gradient of Online RL: PPO, GRPO and Beyond

For online RL, we first consider Proximal Policy Optimization (PPO) (Schulman et al., 2017) and a series of its derivations. PPO is a pivotal technique for RLVR in LLMs. Motivated by TRPO, PPO keeps the new policy close to the old policy, and perform conservative policy updates by incorporating a clipped version of its policy ratio in its objective. The clipping function was shown to stabilize the training process and avoid performance collapse during training. In this section, we omit the regularization terms, such as the KL divergence and entropy. The loss objective for PPO can be written as follows,

|τj|

N

G

1 N

1 G

1 |τj|

min(ri,j,t(θ)Aˆi,j,clip(ri,j,t(θ),1 − ϵ,1 + ϵ)Aˆi,j), (16)

### ∑

### ∑

### ∑

LPPO(πθ) = −

t=1

i=1

j=1

In this setting, we consider questions sampled from a given dataset DRL ≜ {qi}iN=1, and for each question, we consider G trajectories independently sampled using a reference

policy πref. We use ri,j,t(θ) = ππθ(τi,j,t|qi,τi,j,<t)

ref(τi,j,t|qi,τi,j,<t) to denote the policy ratio πθ/πref introduced for importance sampling, ϵ denotes the clipping factor for the importance sampling ratio, enhancing stability.

For PPO, Aˆ is estimated using the Generalized Advantage Estimation (GAE) (Schulman et al., 2015b), calculated based on the reward of the sampled trajectories. For the case of GRPO, the advantage estimate Aˆ is calculated based on a set of sampled trajectories. Given question qi, a group of sampled roll-out trajectoried {τi,j}j∈[G] with verifiable reward R(τi,j) ∈ {0,1}, Aˆi,j is calculated as the normalized reward over the group.

R(τi,j) − mean({R(τi,k)}k∈[G]) std({R(τi,k)}k∈[G])

Aˆi,j =

, (17)

Compared to PPO, the most significant difference introduced by GRPO is the group relative advantage described above. Notably, the original manuscript of GRPO has also induced a sequence-level policy gradient balancing and a KL regularization term. However, more recent works such as (Yu et al., 2025) have removed or modified these terms in general.

The clipped surrogate objective in PPO and similar algorithms enhances the stability of the RL training process by turning off gradient propagation on samples where πθ moves too far from πref. For gradient calculation, this can be represented as an indicator function 1clip.

N

1 N

1 G

### ∑

∇JPPO = −∇LPPO =

i=1

Aˆi,j1clip πref(τi,j,t|qi, τi,j,<t)

|τj|

G

1 |τj|

### ∑

### ∑

∇πθ(τi,j,t|qi, τi,j,<t)

. (18)

t=1

j=1

Apart from PPO and GRPO, many recent RL algorithms for RL post-training in LLMs can be shown to exhibit a similar form for their policy gradient calculations.

###### A.3 Gradient of Offline RL

As stated in the previous sections, many recent studies seek to leverage offline data in the online RL training process for LLMs. These methods consider expert demonstration data as trajectories sampled from a near-optimal policy, and perform RL updates on these data based on policy gradient updates. These algorithms are adapted from the online RL literature and often combine offline and online training, setting them apart from simple SFT.

Taking SRFT (Fu et al., 2025) as an instance, the offline RL objective can be written as follows

1 N

LSRFT(πθ) = −

N

1 G

### ∑

i=1

|τj|

G

1 |τj|

πθ(τi,j,t|qi, τi,j,<t)Aˆi,j, (19)

### ∑

### ∑

t=1

j=1

This objective is derived from the GRPO objective in Equation (16), while setting πref ≡ 1 and removing the clipping mechanism since it becomes imbalanced. The motivation behind setting πref ≡ 1 is that πref is typically unavailable for offline data. Under the assumption that the demonstration policy evenly covers the current policy πθ. In this case, setting πref to 1 changes the algorithm from importance sampling to rejection sampling. The policy gradient of the offline SRFT objective can be derived consequently.

N

1 N

1 G

### ∑

∇JSRFT = −∇LSRFT =

i=1

Aˆi,j πref = 1

|τj|

G

1 |τj|

### ∑

### ∑

∇πθ(τi,j,t|qi, τi,j,<t)

. (20)

t=1

j=1

#### B Additional Theoretical Details for Section 2.2

###### B.1 Deriving Equation 2 from Equation 1

- Lemma A1 (Score-function identity). For density πθ and integrable f(τ), ∇θ Eτ∼πθ[f(τ)] = Eτ∼πθ f(τ) ∇θ log πθ(τ) , Eτ∼πθ ∇θ log πθ(τ) = 0.
- Lemma A2 (Differentiating an expectation with parameterized integrand). For differentiable fθ,

∇θ Eτ∼πθ[fθ(τ)] = Eτ∼πθ ∇θ log πθ(τ) fθ(τ) + ∇θ fθ(τ) .

- Lemma A3 (Measure-change (importance reweighting) identity). Let s(τ | q) be any sampling density that is positive wherever πθ(τ | q) is. Then

1 s(τ)

πθ(τ) s(τ)

f(τ) ∇θ log πθ(τ) = Eτ∼s

f(τ) ∇θπθ(τ) .

Eτ∼πθ f(τ) ∇θ log πθ(τ) = Eτ∼s

Proof. By Lemma A1, ∇Eπθ[r(· | q)] = Eπθ[r(· | q) ∇ log πθ]. For the data-adherence term, since KL(πβ∥πθ) = Eπβ[log πβ − log πθ] and πβ does not depend on θ, we have −µ ∇KL(πβ∥πθ) = µ Eπβ[∇ log πθ]. Summing yields the claim.

| |
|---|

###### B.2 Extension: Adding a Trust-Region Regularizer

A trust region encourages conservative policy updates by penalizing the KL divergence from the current policy πθ to a fixed reference policy πref:

λ KL πθ(· | q) ∥ πref(· | q) , λ ≥ 0. It is the penalty form of the constrained problem max θ

Eτ∼πθ[r(τ | q)] s.t. KL πθ∥πref ≤ δ,

where λ acts as the Lagrange multiplier tied to the trust-region radius δ. Typical choices are πref = πθold (on-policy stability, TRPO/PPO-style). This penalty controls step sizes, dampens distribution shift, and yields clipping-style masks when optimized with PPO surrogates.

Objective and gradient with trust region. Augmenting the Common Objective with the trust-region term gives

Jλ,µ(θ) = Eτ∼πθ(·|q)[r(τ | q)] − λ KL πθ(· | q) ∥ πref(· | q) − µ KL πβ(· | q) ∥ πθ(· | q) , whose gradient is

∇θ Jλ,µ(θ) = Eτ∼πθ r(τ | q)−λ log ππθ(τ|q)

ref(τ|q) ∇θ log πθ(τ | q) + µ Eτ∼πβ ∇θ log πθ(τ | q) . In the estimator (3), this corresponds to replacing the unified advantage by

πβ(τ | q) πθ(τ | q)

πθ(τ | q) πref(τ | q)

A(uniλ)(τ, q) = r(τ | q) − λ log

.

+ µ

All other expressions, including the masked estimator in (6), remain unchanged in form (with Auni replaced by A(uniλ)).

###### B.3 PPO Clipping and the Stabilization Mask With rollout policy πθold and trust-region constraint KL(πθ∥πθold) ≤ δ, the PPO surrogate

πθ(τ) πθold(τ)

min rθ(τ) Aθold(τ), clip(rθ(τ),1 − ϵ,1 + ϵ) Aθold(τ) , rθ(τ) =

max

,

Eτ∼πθ

old

θ

has a piecewise derivative that is zero outside the trusted region in the harmful direction, yielding

1 πθold(τ)

Aθold(τ) ∇θπθ(τ) , (21)

###### ∇θ ≈ Eτ∼πθ

1stable(τ)

old

which matches the masked Unified Policy Gradient Estimator with πref = πθold and A = Aθold.

