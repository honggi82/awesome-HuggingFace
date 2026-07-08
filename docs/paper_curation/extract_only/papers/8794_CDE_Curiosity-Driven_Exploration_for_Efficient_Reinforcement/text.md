# arXiv:2509.09675v1[cs.CL]11Sep2025

## CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models

Runpeng Dai1,3:, Linfeng Song1:, Haolin Liu 1,4, Zhenwen Liang1, Dian Yu1, Haitao Mi1, Zhaopeng Tu2, Rui Liu1,5, Tong Zheng1,5, Hongtu Zhu3, Dong Yu1

1Tencent AI Lab, 2Tencent Multimodal Department,

- 3University of North Carolina at Chapel Hill,
- 4University of Virginia,
- 5University of Maryland, College Park : Core contributors runpeng@unc.edu, lfsong@global.tencent.com

### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model’s own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes.

### 1 Introduction

The reasoning ability of Large Language Models (LLMs) has achieved remarkable performance across diverse application domains such as mathematics (Shao et al., 2024) and coding (Guo et al.,

- 2024). A central challenge in this development is how to efficiently elicit high-quality Chain-ofThought (CoT) reasoning. A major breakthrough earlier this year was the introduction of reinforcement learning with verifiable rewards (RLVR), a training paradigm in which models are optimized directly using the signal of final-answer correctness. This approach removes the burden of designing and training potentially fragile reward models. Despite the emergence of various RLVR training algorithms, such as GRPO (Guo et al., 2024) and DAPO (Yu et al., 2025), key issues remain. In particular, problems such as premature convergence and phenomena like entropy collapse (Cui et al.,
- 2025) have been widely observed during training, posing fundamental challenges to the stability and effectiveness of RLVR.

These challenges stem from the classic exploration-exploitation dilemma in reinforcement learning (Sutton & Barto, 2018). Phenomena like entropy collapse reveal a critical flaw in the training process: it is heavily biased towards exploitation, causing models to converge prematurely instead of sufficiently exploring their environment for better solutions. Although the RL literature encompasses a wide range of exploration strategies, these methods exhibit significant limitations when applied to LLMs. Simple heuristics, including entropy bonuses (Haarnoja et al., 2018) and ϵ-greedy policies (Sutton & Barto, 2018), either injecting randomness to the environment or encouraging the policy

to be more stochastic. These methods are either provably suboptimal in theory (Dann et al., 2022) or demonstrate debatable effectiveness in complex environments like Deep RL (Andrychowicz

- et al., 2021) and LLM-based reasoning (Cui et al., 2025; Shen, 2025). More principled methods are count-based, which incentivize visiting rarely explored state–action pairs. Algorithms such as UCB (Lai, 1987) for multi-armed bandits, LinUCB (Li et al., 2010), LSVI-UCB (Jin et al., 2020) for linear bandits/MDPs achieve near-optimal exploration guarantees across a variety of settings. However, these methods (i) require computationally intensive operations such as matrix inversion, and (ii) heavily depend on highly expressive representations of state-action pair (reasoning paths), which become impractical for reasoning-focused LLMs with long chains of thought. Thus, developing efficient and scalable exploration methods for LLMs remains a key open challenge.

[Figure 3]

In our preliminary experiments, we investigate the direct application of count-based exploration methods to the RLVR setting. To avoid the computational burden of matrix inversion, we adopt the SimHash technique (Tang et al., 2017), which maps the embedding of a CoT response into a discrete hash code, and then uses the visitation frequency of hash cells as pseudo-counts (see Section 3.1 for details). However, as illustrated in Figure 1, this approach proves problematic: it is difficult to meaningfully characterize a complex CoT reasoning trajectory with a fixed embedding vector. In practice, most responses collapse into the same or neighboring hash grids, leading to a highly concentrated distribution of counts and thus undermining the effectiveness of count-based exploration for RLVR.

Figure 1: Distribution of number of visitations across hashcells.

In this work, we propose an intuitive approach that leverages the model’s intrinsic sense of curiosity as a guide for exploration. An LLM, having been trained on vast reasoning corpora, develops a sophisticated internal model of what constitutes a familiar versus a novel reasoning pattern. This parallels early childhood development (Chu & Schulz, 2020), where learning is not driven by a external summary and count of experiences, but is instead propelled by an intrinsic curiosity to explore novel situations. We formalize this principle in our Curiosity-Driven Exploration (CDE) framework, which considers curiosity signals from both the actor and the critic. For the actor, perplexity (PPL) over its generated response serves as the curiosity measure. For the critic, we measure curiosity via the variance of its posterior value distribution. We then approximate this posterior by extending the PPO framework with a multi-head, bootstrapped structure. The curiosity signals are served as an exploration bonus, shaping the reward and advantage functions to effectively guide exploration.

Our theoretical analysis offers further insights into the properties of our method. (i) Theorem 3.1 interprets the proposed perplexity-based bonus, showing that it intrinsically penalizes overconfident errors while encouraging diversity among correct responses. (ii) Theorem 3.2 establishes that in the linear MDP setting, our critic-based exploration bonus is theoretically equivalent to classical count-based bonuses, grounding our approach in established exploration principles.

Our empirical evaluation demonstrates consistent performance gains across four widely used mathematics benchmarks (AIME25, AIME24, AMC23, and MATH), including an approximate +3 point improvement on the challenging AIME benchmarks. Furthermore, our analysis of the training process supports our theoretical findings and reveals a phenomenon we term calibration collapse: under a naive GRPO policy, the model’s confidence progressively decouples from its correctness, while adding PPL bonus mitigates this miscalculation.

### 2 Preliminaries: RLVR, GRPO and PPO

We formulate the language generation process of LLMs as a sequential decision-making problem (Yu et al., 2025; Yue et al., 2025). Specifically, we consider two reinforcement learning algorithms: Group Relative Policy Optimization (GRPO), a critic-free method, and Proximal Policy Optimization (PPO), a

canonical actor–critic method. We adopt the training paradigm of Reinforcement Learning with Verifiable Rewards (RLVR) (Guo et al., 2025; Lambert et al., 2024) and utilize a rule-based verifier to compare the generated response with the ground truth to judge its correctness.

#### 2.1 Group Relative Policy Optimization (GRPO, Shao et al. 2024)

GRPO is an REINFORCE-style optimization algorithm. Let πθ denote the LLM policy with parameters θ. At each training step, given a prompt q sampled from the dataset D, the current policy πθ generates a group of G candidate outputs to1, o2, . . . , oGu. For each candidate oi, we compute its total reward ri “ rpoi, qq.

The advantage for each output is computed by normalizing its reward with respect to the group’s rewards:

ri ´ meanpr1, . . . ,rGq stdpr1, . . . ,rGq ` δ

,

Ai “

where δ is a small constant for numerical stability. The same advantage Ai is applied to all tokens in oi. Let πθold be the policy from the previous step and πref the original pre-trained model. GRPO maximizes:

»

fi fl ´ βDKL pπθ}πrefq,

ÿ|oi|

ÿG

– 1 G

1 |oi|

Lθpr˜i,t, Aiq

LGRPOpθq “ Eq„D,toiu„πθ

old

t“1

i“1

where the clipped objective is

πθpoi,t | q, oi,ătq πθoldpoi,t | q, oi,ătq

˘

`

, r˜i,t “

Lθpr˜i,t, Aiq “ min

r˜i,tAi,clippr˜i,t,1 ´ ε,1 ` εqAi

.

Here, ε and β control the ratio clipping threshold and the KL-penalty strength, respectively. The clipping mitigates large, unstable policy updates, while the KL term constrains deviation from πref.

#### 2.2 Proximal Policy Optimization (PPO, Schulman et al. 2017)

PPO is an actor–critic algorithm that maintains both a policy (actor) πθ and a value function (critic) Vϕ with parameters ϕ, estimating the expected total reward from a given state (prompt and sequence prefix). The advantage function in PPO leverages the critic to reduce variance. Specifically, Generalized Advantage Estimation (GAE) is applied to compute token-level advantages. For an output oi with sentence-level reward ri, the GAE at token t is:

ÿ|oi|

pγλql´tδi,l,

Ai,t “

l“t

where

δi,l “ ri,l ` γVϕpq, oi,ďl`1q ´ Vϕpq, oi,ďlq,

and in our setting ri,l “ 0 for all non-terminal tokens, with ri,|oi| “ ri. The hyperparameters γ and λ are the discount factor and GAE trace-decay, respectively. The PPO objective is:

ÿ|oi|

” 1 |oi|

“Lθpr˜i,t, Ai,tq ´ c1Lϕpq, oi,ăt,riq‰ı ´ βDKL pπθ}πrefq,

LPPOpθ, ϕq “ Eq„D,toiu„πθ

old

t“1

where Lθ is as in GRPO but with per-token Ai,t, and the value loss is:

˘2

Lϕpϕq “ `

. In practice, we alternate optimization of the actor (θ) and the critic (ϕ).

Vϕpq, oi,ătq ´ ri

### 3 CDE: Curiosity-Driven Exploration

In this section, we first explore count-based exploration and identify two of its key challenges. To overcome these challenges, we propose Curiosity-Driven Exploration (CDE), a systematic framework that considers curiosity signals from both the actor and the critic. We introduce the detailed formulations of actor and critic curiosity in Section 3.2 and Section 3.3, respectively.

#### 3.1 Challenge of Count-based exploration for RLVR

The core idea of count-based exploration is to measure the occurrence of Chain-of-Thought (CoT) patterns via sentence embeddings, and to assign an exploration bonus to rarely occurred CoTs. While conceptually appealing, this approach faces two major challenges:

• Curse of dimensionality: Classical count-based methods (e.g., LSVI-UCB (Jin et al., 2020), CFPO

(Cassel & Rosenberg, 2024)) rely on computing the inverse of the covariance matrix Λ´t 1 (see Appendix E) to construct historical visitation ellipsoids. For high-dimensional embeddings, this

operation is computationally prohibitive.

To circumvent the need for matrix inversion, we investigated hash-based counts (Tang et al., 2017) (details in Appendix C), which project sentence embeddings into discrete hash grids and treat grid visitation frequency as a proxy for counts. However, this alternative introduces a second limitation:

• Poor expressiveness of embeddings: As illustrated in Figure 1, after hash coding, most CoT embeddings collapse into neighboring hash grids. This clustering highlights the limited ability of sentence embeddings to distinguish between diverse reasoning patterns, leading to ineffective exploration.

In this work, we move beyond the paradigm of explicit state–action counts and instead utilize the model’s own measure of novelty. Our approach is motivated by the key intuition that agents, much like children in early cognitive development (Chu & Schulz, 2020), exhibit a form of curiosity. They respond with confidence when revisiting familiar states or CoT patterns, yet display uncertainty and exploratory behavior when confronted with novel situations. This learning process is not driven by a external count of experiences but is instead propelled by an intrinsic drive to explore. We first detail the implementation of the actor’s curiosity before turning to the critic’s.

#### 3.2 Exploration Guided by Actor Curiosity

We model actor curiosity as the actor’s uncertainty about its own actions. Intuitively, a response that is surprising to the actor—i.e., has a low probability under its current policy—likely resides in an underexplored region of its learned distribution.

A natural and computationally efficient measure of this surprise is the perplexity of the actor’s generation. We formalize this as a sentence-level curiosity bonus, defined as the negative average log-probability of a generated sentence o “ to1, . . . , oTu, given a prompt q:

ÿT

1 T

Bactorpq, oq “ ´

log πpot|oăt, qq (1)

t“1

where π denotes the actor policy. A higher value for Bactorpq, oq indicates greater surprise and thus a stronger intrinsic reward signal for exploration.

However, practically simply adding this bonus to the original reward can be unstable and suboptimal. Unconstrained exploration might incentivize the model to generate high-perplexity but lowquality or inaccurate responses (a behavior known as reward hacking), or lead to over-exploration where the policy fails to converge to a stable, high-quality output. To ensure that exploration remains tethered to the primary objective of maximizing the original reward signal, we integrate the bonus

using an adaptive clipping mechanism. The total sentence-level reward, r˜, is a combination of the original reward signal rpq, oq and the curiosity bonus Bactorpq, oq, where the bonus is capped relative to the original reward:

r˜pq, oq “ rpq, oq ` ωt minp

|rpq, oq| κ

, αBactorpq, oqq (2)

This formulation promotes exploration by rewarding sentences that the actor finds surprising, while constraining the bonus to remain a fraction of the original reward. In this way, the model is discouraged from trading response quality for novelty. The behavior of this reward function is controlled by three key hyperparameters:

- • The bonus weight ωt is a dynamic coefficient, typically set with an annealing schedule to decrease over the course of training. This allows for more aggressive exploration in the early stages and then gradually shifts focus towards exploitation of high-reward regions as the policy converges.
- • The clipping ratio κ governs the maximum size of the curiosity bonus relative to the original reward. By capping the bonus at |rpq, oq|{κ, it ensures the bonus remains a supplement and prevents it from dominating the learning signal. This is particularly crucial when rpq, oq is negative, as it guarantees the bonus cannot reverse the sign of the reward, maintaining the integrity of the penalty.
- • The bonus scaling factor α normalizes the curiosity bonus Bactorpq, oq before it is compared to the clipped reward. A higher α allows the curiosity bonus to reach the clipping threshold more easily, whereas a smaller α diminishes its potential impact.

Intuitions and Theoretical Foundation While the above formulation specifies how the perplexity bonus is shaped and controlled through its hyperparameters, it is equally important to understand its qualitative effect on model behavior. To build this intuition, we analyze responses along two axes: correctness and actor perplexity. Among these four categories, two require particular attention:

[Figure 7]

- 1. Incorrect responses with low PPL indicate that the model is highly confident in its answer, yet the response is wrong. This reflects overfitting and should be penalized.
- 2. Correct responses with high PPL suggest that the model is less familiar with such answers, but they nevertheless turn out to be successful. This reflects effective exploration and should be encouraged. Figure 2: Responses by correctness

and avg PPL.

As illustrated in Figure 2, we find out that the PPL bonus intrinsically penalizes confident mistakes while encouraging novel correct responses. For correct responses, those novel responses (with higher PPL) receive a larger positive reward. For incorrect responses, those confident responses (with lower PPL) receive larger penalty as it receives smaller PPL bonus. The following theorem formalizes this intuition; its precise statement and proof are deferred to Appendix D.

- Theorem 3.1. Let πt denote the policy at training step t. With PPL bonus in Equation (1), the update to πt`1 calibrates the policy’s confidence as follows:

- (i) Among correct responses, trajectories with higher perplexity receive a larger relative probability increase.
- (ii) Among incorrect responses, trajectories with lower perplexity receive a larger relative probability decrease.

[Figure 9]

Previous analyses distinguish the PPL bonus from the entropy bonus, which is sample-agnostic at the token level. The entropy at any given step depends solely on the policy’s probability distribution and is independent of the token ultimately sampled. Because the calculation considers the entire next-token distribution πθ pv | q, oătq (Equation 3), the bonus Ht remains constant for any potential outcome (Figure 3). Therefore, even when the model makes a high-confidence error by sampling token 1, the entropy bonus fails to penalize that choice.

Ht “ ´ ÿ vPV

Figure 3: An illustration of the model’s policy distribution for selecting the next token.

πθ pv | q, oătqlog πθ pv | q, oătq. (3)

#### 3.3 Exploration Guided by Critic Curiosity

In contrast to critic-free methods such as REINFORCE and GRPO, the critic (value function) in actor–critic frameworks provides a higher-level understanding of the prompt–response pair by estimating the expected reward-to-go. Since this estimate is learned directly from collected trajectories, its posterior distribution conditioned on the observed data naturally reflects the degree of coverage: regions with dense data yield concentrated (low-variance) posteriors, whereas sparsely sampled regions result in higher uncertainty. Posterior distributions are a well-established means of quantifying predictive uncertainty in deep learning models (Gal & Ghahramani, 2016; Lakshminarayanan et al., 2017). As shown in Figure 4, the orange curve exhibits lower variance—evidence of better data coverage—whereas the other curve is more dispersed, reflecting greater uncertainty.

To approximate the posterior distribution of value estimates, we adopt the classical bootstrap method (Davison & Hinkley, 1997), widely used in statistics and increasingly recognized in the RL community as an effective tool for exploration (Osband et al., 2016; Ciosek et al., 2019; Bai et al., 2021). We implement this idea through a multi-head critic (upper-left subfigure in Figure 10), where K critics

[Figure 10]

tVp1, . . . ,VpKu share a common LLM backbone. Each head is trained on a resampled subset of the collected trajectories (bottom subfigure in Figure 10), thereby producing an empirical approximation to the posterior distribution.

We then use the standard deviation across the K heads as a principled curiosity signal, guiding the policy toward regions of high disagreement where the value function remains uncertain and underexplored. In the following theorem, we establish a surprising yet intuitive result: under a Linear MDP assumption, the standard deviation of the bootstrap critics is a consistent estimator of the pseudo-count bonus.

Figure 4: An illustration of two posterior distributions of the critic and their bootstrap approximations.

- Theorem 3.2. In linear MDPs, the standard deviation across multi-head critics can serve as a consistent

estimator for the pseudo-count exploration bonus, bϕnJ,hΛ´n,h1ϕn,h, as used in LSVI-UCB (Jin et al., 2020) and CFPO (Cassel & Rosenberg, 2024), where ϕn,h “ ϕpsn,h, an,hq is the feature vector of a state-action pair and Λn,h “ řn

i“0 ϕi,hϕiJ,h ` λI is the coverage matrix.

The rigorous formulation of the linear MDP assumptions and the proof of Theorem 3.2 are provided in Appendix E, while empirical results in Section 4.4 further support this finding. Building on this foundation, we now describe the training procedure of the multi-head PPO algorithm, which follows the standard stages of vanilla PPO: (i) generating trajectories with the actor, (ii) updating the actor, and (iii) updating the critic. The key distinction is that we incorporate the multi-head variance as an

[Figure 12]

Figure 5: Illustration of the multi-head critic framework.

exploration bonus, encouraging the policy to visit under-explored regions. A visual illustration of these steps is shown in Figure 5.

- • Actor roll-out: Given a prompt q, the actor generates a set of responses to1, . . . , onu. Each

response is denoted as oi “ toi,1, . . . , oi,|oi|u. Correspondingly, we associate each response with a verifiable reward ri. For clarity, we focus on the case of a single prompt q.

- • Actor update: In this step, the advantage is estimated as

`ωt minˆ|A˜i,t| κ

, αBcriticpq, oi,ďt`1q˙. (4)

ÿ|oi|

pγλql´tδpi,l loooooomoooooon

Api,t “

l“t

«A˜i,t

The advantage consists of two components. The first term, A˜i,t, largely follows the standard advantage estimation in PPO, except that we exploit bootstrap estimators by using an ensemble of value functions rather than a single point estimate:

ÿK j“1Vpjpq, oi,ďl`1q ´

ÿK j“1Vpjpq, oi,ďlq.

1 K

γ K

δpi,l “ ri,l `

The second term of Equation 4 introduces the multi-head critic bonus (Bcritic), governed by the bonus weight ωt, clipping ratio κ, and scaling factor α (see discussion following Equation (2) for interpretation). Specifically, Bcritic is defined as the standard deviation across the K value heads, encouraging exploration by assigning higher bonus to actions leading to uncertain/less-visited regions:

˘ “ std´␣

(¯. (5)

`

Vpjpq, oi,ďt`1qˇ1 ď j ď K

q, oi,ďt`1

Bcritic

- • Critic update: We use the collected roll-outs to update the critic. For notational convenience, let the dataset be

D “ tpq, oi,ďt,riq|i P rns, t P r|oi|su, (6) consisting of (prompt, partial response, reward) triplets. For each critic head j, we sample without replacement a subset Dj Ă D of size |Dj| “ ζ|D|, where the hyperparameter ζ P p0,1s controls the fraction of data assigned per head. Smaller ζ increases head diversity, while larger ζ improves sample efficiency. The multi-head critic is then updated with the following bootstrap loss:

### 4 Experiments

1 ζK|D|

Lϕ “

ÿK

ÿ

j“1

pq,o,rqPDj

´

Vpjpq, oq ´ r¯2 .

#### 4.1 Dataset and Model

In this paper, we adopt DAPO-17K (Yu et al., 2025) for training and evaluate the performance of CDE on four challenging mathematical reasoning benchmarks: MATH (Hendrycks et al., 2021), AMC23 (MAA, b), AIME24, and AIME25 (MAA, a). These evaluations are designed to assess CDE’s effectiveness in comparison to standard PPO and GRPO algorithms. Due to computational resource constraints, we conduct training with a reduced setting. All experiments are implemented within the Verl framework using the Qwen3-4B-Base model (Yang et al., 2025). For fair comparison, all the models use the default prompt in DAPO-17K as shown in Appendix B and the implementation details in Appendix A to further elaborate on the training settings.

#### 4.2 Main Results

The main results are presented in Table 1 while the training dynamic is presented in Figure 6. Here PPL bonus denote adding Curiosity bonus on actors as in Equation 2, K Heads represents multi-head critic PPO with K head critics. We report both average Pass@1 accuracy and Pass@16 results on evaluation datasets. The key observations are as follows:

MATH AMC23 AIME24 AIME25 Avg Avg@1 Avg@16 Pass@16 Avg@16 Pass@16 Avg@16 Pass@16

Model

Qwen3-4B-Base 23.1 10.9 53.8 1.5 8.4 1.3 8.3 9.2

###### GRPO based methods

Qwen3-4B-Base-GRPO 87.3 63.6 89.1 20.8 41.9 21.0 39.2 48.2 ë w/ PPL bonus 87.7 67.8 89.5 23.3 48.5 23.5 42.5 50.6

###### PPO based methods

Qwen3-4B-Base-PPO 86.6 64.1 87.2 17.8 36.0 17.5 33.7 46.5 ë w/ PPL bonus 87.9 66.1 88.5 18.3 37.6 18.3 33.5 47.7 ë w/ 2 Heads 83.2 63.6 89.9 19.6 34.8 19.6 36.1 46.6 ë w/ 4 Heads 87.3 63.9 87.9 21.5 35.5 21.5 45.5 48.5 ë w/ 8 Heads 85.1 66.7 86.9 21.7 46.4 19.0 37.1 48.1 ë w/ 16 Heads 88.3 65.0 88.7 20.5 41.9 20.0 38.8 48.6

- Table 1: Zero-shot accuracy of different models on the validation datasets. Avg@16 denotes the mean Pass@1 accuracy over 16 sampled generations, while Avg column represents the overall average across datasets, computed as Avg@1 for MATH and Avg@16 for the remaining datasets.

• The PPL bonus further enhances the mathematical reasoning ability of the GRPO method, yielding an average improvement of approximately `2.4 points across datasets and demonstrating

[Figure 15]

[Figure 16]

Figure 6: Comparison of Avg@16 accuracy on AIME25 over training of vanilla GRPO and PPO (Baseline methods) and GRPO with PPL bonus and 16 head multi-head PPO(Our methods).

consistent superiority. In particular, our method achieves notable gains on Pass@16, surpassing the baseline GRPO by about `8 points on the AIME24 dataset.

- • Across benchmarks, multi-head PPO consistently outperforms vanilla PPO. Using K “ 4 and K “ 16 heads yields average gains of roughly `2 points, and we observe an around `10 points of increase in Pass@16 on AIME datasets in many cases.
- • The performance of multi-head PPO generally increases with the number of heads K: with K “ 2 delivers negligible gains over the baseline, and performance increase begin to plateau once K ě 4, which suggests that a modest number of heads already captures most of the epistemic uncertainty needed.
- • As shown in Figure 6, GRPO with PPL bonus and multi-head PPO increase test accuracy more slowly than baseline PPO/GRPO early in training, then catch up and ultimately surpass them. This pattern is consistent with enhanced exploration: the PPL bonus and head disagreement discourage premature exploitation of spurious high-reward trajectories. As state-action coverage expands, these signals calibrate, enabling a smoother shift to targeted exploitation and yielding higher final accuracy.

#### 4.3 Understanding the Effect of the PPL Bonus

In this subsection, we present additional experiments to investigate the role of the PPL bonus, from which we derive the following key findings.

[Figure 17]

Bonus weight decay is crucial We compare four schedules for the bonus weight ωt—No decay, Linear, Cosine, and Staircase—as illustrated in Figure 7, with the performance of models trained under each schedule summarized in Table 2. Briefly, the No decay schedule maintains strong exploration throughout training, while the Staircase schedule reduces ωt abruptly, enabling strong exploration in the early phase and then removing the bonus for final convergence. The Linear and Cosine schedules provide intermediate behaviors.

Figure 7: An illustration of different weight anneal schedules.

The results in Table 2 underscore two insights: First, decay of the bonus weight is necessary, as all decay schedules outperform the no-decay baseline by enabling a gradual shift from exploration to exploitation. Second, strong exploration in the early phase is crucial, with the staircase scheme proving most effective by sustaining high exploration initially to broaden state–action coverage and then removing the bonus abruptly to allow stable convergence, whereas the gentler cosine and linear decays weaken the signal too soon and thus yield smaller gains.

MATH AMC23 AIME24 AIME25 Avg Avg@1 Avg@16 Pass@16 Avg@16 Pass@16 Avg@16 Pass@16

Model

###### Bonus Weight Decay Schedules

Qwen3-4B-Base-GRPO 87.3 63.6 91.1 21.0 41.9 20.8 39.2 48.2 ë ωt No decay 85.1 64.5 84.6 20.8 39.0 22.3 36.2 48.2 ë ωt Linear decay 85.4 66.1 91.9 23.3 40.4 20.0 40.4 48.7 ë ωt Cosine decay 86.7 68.1 90.0 22.5 44.9 21.5 40.7 49.7 ë ωt Staircase decay 87.7 67.8 89.2 23.5 48.5 23.3 40.3 50.6

- Table 2: Zero-shot accuracy of GRPO models under different PPL bonus weight decay schedules. The schedules follow those illustrated in Figure 7.

Analysis of Entropy Dynamics As highlighted in prior work, entropy provides an important lens for understanding exploration ability (Cui et al., 2025), where a sharp decline in entropy often signals premature convergence and insufficient exploration. Figure 8 illustrates the entropy dynamics of baseline GRPO compared to our proposed methods. First, relative to the baseline, the PPL bonus alleviates entropy collapse, demonstrating its role in promoting exploration. Second, when comparing decay schemes, PPL with No Decay shows persistent fluctuations and fails to converge, whereas Staircase decay yields more stable entropy trajectories. This observation is consistent with our earlier findings that decaying the bonus weight is essential for ensuring stable convergence while still supporting effective exploration.

[Figure 19]

Figure 8: Dynamics of policy entropy over the training process. The bonus weight decay mechanism follows Figure 7. The Staircase and No Decay schedules share the same early training phase.

[Figure 20]

[Figure 21]

(a) GRPO without PPL bonus (b) GRPO with PPL bonus

Figure 9: Average response PPL per training step, stratified by correctness.

Analysis of Calibration As shown in Figure 9, we plot the batch-wise mean response perplexity (PPL), stratified by answer correctness. In subfigure (a), we observe a phenomenon we term calibration collapse: early in naive GRPO training, correct responses have lower PPL (higher confidence) than incorrect ones, but as training progresses this gap shrinks and ultimately vanishes—confidence

no longer tracks correctness. By contrast, with a PPL bonus (subfigure (b)), this separation is sustained throughout training.

This pattern is explained by Theorem 3.1: while both naive GRPO and GRPO with a PPL bonus tend to increase confidence on correct answers, the PPL bonus additionally suppresses confident errors (low-PPL incorrect trajectories), thereby improving calibration.

This finding is original and practically important. Ideally, a trained model should be faithful—confident when its answer is correct and cautious when it is not. Better calibration enhances interpretability and supports inference-time selection strategies such as self-certainty BoN (Wang

- et al., 2022) and DeepConf (Fu et al., 2025). It also connects to the growing literature on calibrating LLMs, both during training (e.g., (Shen et al., 2024)) and at test time (e.g., (Ulmer et al., 2024)).

#### 4.4 Further Analysis of the Multi-Head Critic

Analysis of Dynamics of Bcritic We further examine the dynamics of the multi-head exploration bonus Bcritic by tracking its average value over the course of training. Specifically, for each training step, given the roll-outs D defined in Equation 6, we compute the average Bcritic across (prompt, partial response, reward) triplets within D. As shown in sub-figure (a) of Figure 10, this average decreases steadily as training progresses. The decline reflects that, with more training, similar trajectories are revisited more frequently, leading to reduced disagreement among critic heads. This phenomenon provides empirical support for interpreting the multi-head bonus as analogous to count-based exploration measures.

[Figure 23]

[Figure 24]

(a) (b)

Figure 10: (a) The average Bcritic over training steps. (b) Distribution of the standard deviation of value heads across prompts from different datasets.

In sub-figure (b) of Figure 10, we present a cross-dataset analysis by calculating the average standard deviation of the value estimates across different questions. Specifically, we evaluate three datasets: the training set (DAPO-17K), the in-domain validation set (AMC23), and the out-of-domain validation set GPQA (Rein et al., 2023). We observe that the training set exhibits a smaller standard deviation compared to both the in-domain and out-of-domain validation sets. This pattern aligns with the intuition that multi-head critics tend to show stronger disagreement on data that is less frequently encountered during training.

Analysis of sub-sample fraction ζ during critic update Additionally, we examine the sensitivity of the critic update to the hyperparameter ζ (sub-sample fraction). We vary ζ under two configurations—critics with 16 heads and with 4 heads—and compare ζ P t0.5,1u. As shown in Table 3, while a larger number of heads benefits from a larger sub-sample fraction, the overall performance

is stable across settings. The model demonstrates robustness to the masking fraction ζ, achieving similar results for both values tested (0.5 and 1.0).

MATH AMC23 AIME24 AIME25 Avg Avg@1 Avg@16 Pass@16 Avg@16 Pass@16 Avg@16 Pass@16

Model

###### Mask fraction

16 Heads ; ζ “ 0.5 88.3 65.0 88.7 20.5 41.9 20.0 38.8 48.6 16 Heads ; ζ “ 1 85.4 65.3 85.3 21.0 39.2 21.7 43.2 48.4 4 Heads ; ζ “ 0.5 86.1 66.4 85.8 18.1 36.7 23.1 39.1 48.4 4 Heads ; ζ “ 1 87.3 63.9 87.9 21.5 35.5 21.5 45.5 48.5

Table 3: Ablation study on sub-sample fraction ζ.

### 5 Related Work

#### 5.1 Reinforcement Learning (RL) for LLM reasoning

Reinforcement Learning is a central technique for advancing the reasoning capabilities of LLMs. Initial approaches relied on reward models that provided either outcome-based supervision, focusing on the final answer (Cobbe et al., 2021), or process-based supervision, evaluating intermediate reasoning steps (Uesato et al., 2022). To navigate more complex problem spaces, these foundational reward strategies were often augmented with search algorithms such as MCTS (Feng et al., 2023; Tian

- et al., 2024; Chen et al., 2024; Wang et al., 2024c) and Q* (Wang et al., 2024b;a). More recently, RLVR (Lambert et al., 2024) has emerged as a powerful alternative, demonstrating significant performance on complex reasoning tasks in mathematics and coding (Guo et al., 2025). Consequently, a growing body of work seeks to apply RLVR to diverse domains, including multi-modal reasoning (Wang
- et al., 2025; Li et al., 2025), logical reasoning (Zhou et al., 2025), search engine use (Jin et al., 2025; Xiong et al., 2025), and information extraction (Dai et al., 2025b). Parallel efforts aim to improve upon the standard RLVR paradigm with techniques such as mixture-of-thought (Zheng et al., 2025a), self-evolving (Huang et al., 2025), parallel thinking (Zheng et al., 2025b). Despite these advances, persistent concerns remain regarding robustness (Dai et al., 2025a; Zhao et al., 2025), calibration (Shen et al., 2024), and a lack of exploration evidenced by entropy collapse (Cui et al., 2025; Shen, 2025), highlighting the need for more principled training approaches.

#### 5.2 Efficient exploration

Efficient exploration is a central challenge in Reinforcement Learning (RL), which aim to balance between exploration and exploitation (Sutton & Barto, 2018; Weng, 2020; Amin et al., 2021). Many foundational approaches are heuristic-based, such as Gaussian noise (Lillicrap et al., 2015) or the ϵ-greedy method (Sutton & Barto, 2018). Entropy regularization is a more principled heuristic, which encourages the policy to be more stochastic. While simple to implement, these methods are often undirected—they promote pure randomness. Consequently, they can be suboptimal (Dann et al., 2022) with no significant gains in complex Deep RL (Andrychowicz et al., 2021) or LLM training (Cui et al., 2025; Shen, 2025).

In contrast, a major class of methods incentivizes exploration by adding exploration bonus to guide the agent toward novel or uncertain parts of the environment. Count-based approaches like UCB (Lai, 1987), LinUCB (Li et al., 2010), and LSVI-UCB (Jin et al., 2020) use pseudo-counts of state-action visitations to encourage exploring rarely visited areas, achieving near-optimal theoretical guarantees in bandits and linear MDPs. Similarly, prediction-based methods such as ICM (Pathak et al., 2017) and RND (Burda et al., 2018) use the error from a predictive model as a bonus, rewarding the agent for reaching states that are difficult to predict. Applying these guided exploration principles is a growing field in LLM. For instance, Bai et al. (2025) incorporate a count-based bonus into the RLHF process by introducing a coin flipping module. Gao et al. (2025) draws inspiration from RND by adding an auxiliary noise prediction network. However, both methods rely on expressive

representations of long COT trajectories and introduce additional modules, which complicates the training framework. In contrast, CDE uses intrinsic curiosity signals from the actor and critics, requiring only minimal modifications to the framework and yielding efficient exploration both theoretically and empirically.

### 6 Conclusion and Future Work

We have presented Curiosity-Driven Exploration, an efficient technique that enhances agent learning by incorporating curiosity signals from both the actor and the critic. Our approach is notably lightweight, demanding only minor modifications to the original training architecture. Its effectiveness is demonstrated by consistent accuracy improvements over strong baselines on a suite of challenging mathematical reasoning benchmarks, with these empirical results strongly corroborating our underlying theoretical framework and intuition.

The calibration collapse revealed in our analysis aligns with recent findings on the root causes of LLM hallucination (Kalai et al., 2025), pointing to a promising avenue for future work. We hypothesize that the underlying source of this collapse is the reward design of RLVR training. Specifically, RLVR with outcome reward prioritizes correct final outcomes at the expense of rigorous intermediate reasoning. Our experiments shed light on this direction by demonstrating that an alternative multi-perspective reward design (e.g., the PPL bonus) can be valuable for guiding the RLVR process more effectively.

### References

Susan Amin, Maziar Gomrokchi, Harsh Satija, Herke Van Hoof, and Doina Precup. A survey of exploration methods in reinforcement learning. arXiv preprint arXiv:2109.00157, 2021.

Marcin Andrychowicz, Anton Raichuk, Piotr Sta´nczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What matters for on-policy deep actor-critic methods? a large-scale study. In International conference on learning representations, 2021.

Chenjia Bai, Lingxiao Wang, Lei Han, Jianye Hao, Animesh Garg, Peng Liu, and Zhaoran Wang. Principled exploration via optimistic bootstrapping and backward induction. In International Conference on Machine Learning, pp. 577–587. PMLR, 2021.

Chenjia Bai, Yang Zhang, Shuang Qiu, Qiaosheng Zhang, Kang Xu, and Xuelong Li. Online preference alignment for language models via count-based exploration. arXiv preprint arXiv:2501.12735, 2025.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.

Asaf Cassel and Aviv Rosenberg. Warm-up free policy optimization: Improved regret in linear

markov decision processes. Advances in Neural Information Processing Systems, 37:3275–3303, 2024. Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. Alphamath almost zero: process supervision

without process. Advances in Neural Information Processing Systems, 37:27689–27724, 2024. Junyi Chu and Laura E Schulz. Play, curiosity, and cognition. Annual Review of Developmental Psychology, 2(1):317–343, 2020. Kamil Ciosek, Quan Vuong, Robert Loftin, and Katja Hofmann. Better exploration with optimistic actor critic. Advances in Neural Information Processing Systems, 32, 2019.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Runpeng Dai, Run Yang, Fan Zhou, and Hongtu Zhu. Breach in the shield: Unveiling the vulnerabilities of large language models. arXiv preprint arXiv:2504.03714, 2025a.

Runpeng Dai, Tong Zheng, Run Yang, Kaixian Yu, and Hongtu Zhu. R1-re: Cross-domain relation extraction with rlvr. arXiv preprint arXiv:2507.04642, 2025b.

Chris Dann, Yishay Mansour, Mehryar Mohri, Ayush Sekhari, and Karthik Sridharan. Guarantees for epsilon-greedy reinforcement learning with function approximation. In International conference on machine learning, pp. 4666–4689. PMLR, 2022.

Anthony Christopher Davison and David Victor Hinkley. Bootstrap methods and their application. Number 1. Cambridge university press, 1997.

Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179, 2023.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016.

Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. Navigate the unknown: Enhancing llm reasoning with intrinsic motivation guided exploration. arXiv preprint arXiv:2505.17621, 2025.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. Pmlr, 2018.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-zero: Self-evolving reasoning llm from zero data. arXiv preprint arXiv:2508.05004, 2025.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on learning theory, pp. 2137–2143. PMLR, 2020.

Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, and Edwin Zhang. Why language models hallucinate, 2025. URL https://openai.com/index/why-language-models-hallucinate/.

Tze Leung Lai. Adaptive treatment allocation and the multi-armed bandit problem. The annals of statistics, pp. 1091–1114, 1987.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pp. 661–670, 2010.

Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding vision-language model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

MAA. American invitational mathematics examination (AIME). Mathematics Competition Series, n.d.a. URL https://maa.org/math-competitions/aime.

MAA. American mathematics competitions (AMC 10/12). Mathematics Competition Series, n.d.b. URL https://maa.org/math-competitions/amc.

Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. Advances in neural information processing systems, 29, 2016.

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pp. 2778–2787. PMLR, 2017.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Han Shen. On entropy control in llm-rl algorithms. arXiv preprint arXiv:2509.03493, 2025. Maohao Shen, Subhro Das, Kristjan Greenewald, Prasanna Sattigeri, Gregory Wornell, and Soumya

Ghosh. Thermometer: Towards universal calibration for large language models. arXiv preprint arXiv:2403.08819, 2024.

Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. MIT press, 2018. Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John

Schulman, Filip DeTurck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.

Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Lei Han, Haitao Mi, and Dong Yu. Toward self-improvement of llms via imagination, searching, and criticizing. Advances in Neural Information Processing Systems, 37:52723–52748, 2024.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seong Joon Oh. Calibrating large language models using their generations only. arXiv preprint arXiv:2403.05973, 2024.

Ante Wang, Linfeng Song, Ye Tian, Baolin Peng, Dian Yu, Haitao Mi, Jinsong Su, and Dong Yu. Litesearch: Efficacious tree search for llm. arXiv preprint arXiv:2407.00320, 2024a.

Chaojie Wang, Yanchen Deng, Zhiyi Lyu, Liang Zeng, Jujie He, Shuicheng Yan, and Bo An. Q*: Improving multi-step reasoning for llms with deliberative planning. arXiv preprint arXiv:2406.14283, 2024b.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.

Xiyao Wang, Linfeng Song, Ye Tian, Dian Yu, Baolin Peng, Haitao Mi, Furong Huang, and Dong Yu. Towards self-improvement of llms via mcts: Leveraging stepwise knowledge with curriculum preference learning. arXiv preprint arXiv:2410.06508, 2024c.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Lilian Weng. Exploration strategies in deep reinforcement learning. lilianweng.github.io, Jun 2020. URL https://lilianweng.github.io/posts/2020-06-07-exploration-drl/.

Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang, Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing Song, Dengyu Wang, Minjia Zhang, et al. Rag-gym: Optimizing reasoning and search agents with process supervision. arXiv preprint arXiv:2502.13957, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Yulai Zhao, Haolin Liu, Dian Yu, SY Kung, Haitao Mi, and Dong Yu. One token to fool llm-as-a-judge. arXiv preprint arXiv:2507.08794, 2025.

Tong Zheng, Lichang Chen, Simeng Han, R Thomas McCoy, and Heng Huang. Learning to reason via mixture-of-thought for logical reasoning. arXiv preprint arXiv:2505.15817, 2025a.

Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Xinyu Yang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, and Dong Yu. Parallel-r1: Towards parallel thinking via reinforcement learning. 2025b. URL https://api.semanticscholar.org/CorpusID: 281218689.

Yujun Zhou, Jiayi Ye, Zipeng Ling, Yufei Han, Yue Huang, Haomin Zhuang, Zhenwen Liang, Kehan Guo, Taicheng Guo, Xiangqi Wang, et al. Dissecting logical reasoning in llms: A fine-grained evaluation and supervision study. arXiv preprint arXiv:2506.04810, 2025.

### A Training Details

We use verl as the training framework1. Configurations for training CDE and baseline models are listed in Table 4.

Config GRPO PPO actor-lr 1e-6 1e-6 critic-lr - 1e-5 critic-warmup - 10 kl_coef 0.0 0.0 max_prompt_length 2K 2K max_response_length 3K 3K train_batch_size 256 512 ppo_mini_batch_size 256 256 clip_ratio 0.20 0.20 sample temperature 1.0 1.0 rollout.n 8 4 total_training_steps 300 300

(a)

Config PPL 2,4 Heads 8,16 Heads

κ 3 3 3 α 1 0.5 0.5 ωt Staircase No decay No decay ζ - 1 0.5

(b)

Table 4: (a) Baseline training configurations. The GRPO setup is shared across all GRPO-based methods (e.g., “Qwen3-4B-Base-GRPO” and “w/PPL bonus” in Table 1); likewise, the PPO setup is shared across all PPO-based methods. (b) CDE-specific configurations. The PPL settings are identical for both the GRPO “w/PPL bonus” and PPO “w/PPL bonus” variants.

### B Prompt

Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. {Problem} Remember to put your answer on its own line after “Answer:”.

Figure 11: The prompt for RLVR training.

### C Details on Hash-based pseudo count

The core idea is to map a full prompt–response trajectory to a compact hash that serves as a pseudostate for exploration. Given a prompt–response pair pq, oq with tokenized sequences q “ tq1, . . . , qDu and o “ to1, . . . , oTu, let the model produce last-layer hidden states “ th1, . . . , hD`Tu, hi P Rd. We form a trajectory embedding hq,o P Rd from via one of: (i) hD`T; (ii) hD`T´1; or (iii) mean pooling D`1 T

řD`T

i“1 hi. With a random projection matrix A P Rkˆd (rows drawn i.i.d. from Np0, Iq or Rademacher), we compute a k-bit SimHash code

`

˘ P t´1,`1uk,

ϕpq, oq “ sign

Ahq,o

1https://github.com/volcengine/verl

Algorithm 1: Count-based exploration for RLVR through SimHash Inputs:Policy πθ; aggregator g; random projection matrix A P Rkˆd; hash counts nr¨s Ð 0;

weights βt.

for each training iteration t “ 1,2, . . . do

Sample prompts q and generate responses o „ πθp¨ | qq for each pq, oq in the batch do

Obtain last-layer token states thiu|i“q|`|1 o|; set h Ð gpthiuq c Ð signpAhq, b Ð bucketpcq nrbs Ð nrbs ` 1

r˜pq, oq Ð rpq, oq ` βt{a

nrbs Update πθ.

and map it to a bucket index b “ bucketpϕq P t0, . . . ,2k ´ 1u. Let npbq be the visitation count of bucket b. We apply intrinsic reward shaping to encourage rarely visited trajectories:

βt a

r˜pq, oq “ rpq, oq ` ωt

,

npbq

where βt is the weight for exploration bonus a. This yields an efficient, matrix-inversion–free intrinsic bonus that scales linearly in kd per sample, following Tang et al. (2017).

### D Proof for Calibration Theorem

Define r˜tpq, oq “ rpq, oq ` btpq, oq where btpq, oq “ ω mintκ|rpq, oq|,´Tα

log πtpo|qqu is a bonus function where To is the length of response o. Note that ω is a redundant variable in theory because we can write btpq, oq “ mintκ1|rpq, oq|,´Tα

o

log πtpo|qqu with κ1 “ ωκ and α1 “ ωα. Given that rpx, yq P t1,´1u, it suffices to consider btpq, oq “ mintκ,´Tα

o

log πtpo|qqu. Thus, as long as we use κ ă 1, we have signpr˜tpq, oqq “ signprpq, oqq. The introduce of bonus does not change the sign of the original correctness reward. Consider single step policy optimization

o

+, which has closed-form solution

#

ÿ

1 η

πpo|qqr˜tpq, oq ´

πt`1p¨|qq “ argmax

KLpπp¨|qq}πtp¨|qqq

π

o

πtpo|qqexppηr˜tpq, oqq ř

. For any question q and response o. Define Zpqq “ ř

πt`1po|qq “

o1 πtpo1|qqexppηr˜tpq, o1qq

`

ηr˜tpq, o1q˘

o1 πtpo1|qqexp

, we have

log πt`1po|qq “ log πtpo|qq ` ηr˜tpq, oq ´ logpZpqqq. Define ∆tpo|qq “ log πt`1po|qq ´ log πtpo|qq as the change of likelihood of response o under question q at update step t. For two correct response o1` and o2` with length To`

, and ´Tα

and To`

2

1

log πtpo1`|qq ě ´Tα

log πtpo2`|qq (i.e. o1` has larger perplexity), we have ∆tpo1`|qq ´ ∆tpo2`|qq

o` 1

o` 2

“ r˜tpq, o1`q ´ r˜tpq, o2`q “ btpq, o1`q ´ btpq, o2`q “ mintκ,´

α To`

α To`

log πtpo2`|qqu ě 0

log πtpo1`|qqu ´ mintκ,´

2

1

Similarly, for two incorrect response o1´ and o2´ with ´Tα

log πtpo1´|qq ě ´Tα

log πtpo2´|qq (i.e. o1´

o´ 1

o´ 2

has larger perplexity), we have ∆tpo1´|qq ´ ∆tpo2´|qq ě 0. Specifically, given a question q, for any response po1, o2q that has the same correctness label and ´Tα

log πtpo1|qq ě ´Tα

log πtpo2|qq, we have

o1

o2

- • If r˜tpq, o1q ě η1 logpZpqqq and r˜tpq, o2q ě η1 logpZpqqq , then ∆tpo1|qq ě 0 and ∆tpo2|qq ě 0 but o1 has more likelihood increase.

- • If r˜tpq, o1q ě η1 logpZpqqq and r˜tpq, o2q ă η1 logpZpqqq , then ∆tpo1|qq ě 0 and ∆tpo2|qq ă 0 where o1’s likelihood increase but o2’s likelihood decrease.

- • If r˜tpq, o1q ă η1 logpZpqqq and r˜tpq, o2q ă η1 logpZpqqq , then ∆tpo1|qq ă 0 and ∆tpo1|qq ă 0 but o1 has less likelihood decrease.

- • It is impossible that r˜tpq, o1q ă η1 logpZpqqq and r˜tpq, o2q ě η1 logpZpqqq given that po1, o2q

has the same correctness label and ´Tα

log πtpo1|qq ě ´Tα

log πtpo2|qq.

o1

o2

### E Proof for Consistency of Multi-head Critic Bonus

#### Linear MDP and Assumptions

Assumption E.1 (Linear MDP). We consider finite horizon M “ pS, A, R, P, Hq with horizon H, state space S, action space A, reward function R : S ˆ A Ñ R, and transition P : S ˆ A Ñ S such that there exists a known feature ϕ P Rd and unknown features θ, ψ P Rd to ensure

Rps, aq “ ϕps, aqJθ Pps1|s, aq “ ϕps, aqJψps1q. Without loss of generality, we assume }ϕps, aq} ď 1 for all ps, aq, and }ψps1q} ď

?

?

d,}θ}2 ď

d. Lemma E.2 (Proposition 2.3 in Jin et al. (2020)). For linear MDPs that satisfy Assumption E.1, there exists w‹h P Rd such that

Qπh ps, aq :“ E” ÿH t“h

rtˇsh “ s, ah “ aı “ ϕps, aqJw‹h.

The linearity of Q-functions enables using regression technique to solve it. Consider a dataset with n observations D “ tsi,h, ai,h, Gi,huin“1 where Gi,h is the Monte-Carlo return. Let ϕi,h “ ϕpsi,h, ai,hq and denote the regression noise as εi,h “ Gi,h ´ ϕiJ,hw‹h. We impose the following assumptions.

- (A1) Erεi,h | ϕi,hs “ 0 and tpεi,hquin“1 are i.i.d. σ2–sub-Gaussian for each fixed h;
- (A2) n1

řn i“1 ϕi,hϕiJ,h ÝÑP Σt ą 0

Jin et al. (2020) shows that doing value iteration on optimistically estimated Q function can achieve near-optimal regret for linear MDP, where the optimistic Q function is the combination of linear

regression estimation and exploration bonus bn,h “ βbϕnJ,hΛ´n,h1ϕn,h, where Λn,h “ λI ` řn

i“1 ϕi,hϕiJ,h and β is some constant. Below we will formally connect our bootstrapped bonus with this term. Formulation of the bootstrap multi-head critic

We accommodate the bootstrap multi-head into the linear-MDP setting. For any time step h, we sample K mini-batches tSk Ă rnsukK“1 of size m “ ζn uniformly without replacement from D and construct the ridge estimator as follows

ÿ

wppnk,hq “ argmin

pGr,h ´ ϕrJ,hwq2 ` ζλ}w}2.

w

rPSk

For any feature ϕ P Rd, we define the bootstrap multi-head bonus as bhboot,K pϕq “ std´␣

(¯.

ϕJwppnk,hqˇ1 ď k ď K

Elliptical (“count-based”) bonus in (Jin et al., 2020). The ridge estimator is constructed using all data across n trajectories as follows

ÿn

pGi,h ´ ϕiJ,hwq2 ` ζλ}w}2.

wpn,h “ argmin

w

i“1

For any query feature ϕ P Rd, the bonus term is bhcntpϕq “ bϕJΛ´n,h1ϕ.

Formal Version and Proof of Theorem 3.2 Theorem E.3. Under Assumption E.1 and assumptions (A1)–(A2), for any fixed time-step h and query ϕ P Rd,

βbϕJΛ´n,h1ϕ, where β is some constant.

bhboot,K pϕq ÝÝÝÝÝÝÝÝÝÑP

KÑ8, nÑ8

Proof. For any time-step h and Sk Ă rns, we have the explicit solution of the ridge regression

ÿn

wpn,h “ Λ´n,h1

ϕi,hGi,h.

i“1

Conditioning on Xh “ rϕ1,Jh; . . . ; ϕnJ,hs, the conditional variance of the estimator is Var

˘ “ σ2ϕJ´Λ´n,h1 ´ λΛ´n,h2¯ϕ. From Assumption (A2), we have }Λ´n,h1}op “ Opp1{nq, therefore

`

ϕJwpn,h | Xh

ϕJΛ´n,h1ϕ ď }ϕ}2 }Λ´n,h1}op “ Opp1{nq and ϕJΛ´n,h2ϕ “ Opp1{n2q, and

nϕJ´Λ´n,h1 ´ λΛ´n,h2¯ϕ ´ nϕJΛ´n,h1ϕ “ ´nλϕJΛ´n,h2ϕ ÝÑP 0. Therefore, we have

ϕJ´Λ´n,h1 ´ λΛ´n,h2¯ϕ ÝÑP ϕJΛ´n,h1ϕ.

Before moving to bhboot,K pϕq, we define the following quantities ∆Σ “

ÿn

ÿn

1 ζ ÿ

1 ζ ÿ

ϕr,hϕrJ,h ´

ϕi,hϕiJ,h, b “

ϕr,hGr,h, ∆b “ bs ´ b.

ϕi,hGi,h, bs “

rPSk

rPSk

i“1

i“1

?nq. Use the expansion

Since Σt ą 0, matrix Bernstein for sampling without replacement yields }∆Σ}op “ Opp

pΛn,h ` ∆Σq´1 “ Λ´n,h1 ´ Λ´n,h1∆ΣΛ´n,h1 ` RΣ, }RΣ}op “ Opp}Λ´n,h1}3op}∆Σ}2opq “ Opp1{n2q. The k-th bootstrap ridge solution is

wppnk,hq “ pΛn,h ` ∆Σq´1bs. Subtracting wpn,h “ Λ´n,h1b and inserting the expansion,

`` ´ Λ´n,h1∆ΣΛ´n,h1∆b ` RΣbs

˘

wppnk,hq ´ wpn,h “ Λ´n,h1∆b ´ Λ´n,h1∆Σ wpn,h looooooooooooomooooooooooooonfirst order

.

looooooooooooooooomooooooooooooooooon“:rn

Since Gi,h “ ϕiJ,hwh ` ϵi,h, for any ϕ we have

ÿn

ϕi,hϵi,h¯ ` ϕJΛ´n,h1∆Σpw‹h ´ wpn,hq ` ϕJrn. (7)

ϕJpwppnk,hq ´ wpn,hq “ ϕJΛ´n,h1´1

ζ ÿ

ϕr,hϵr,h ´

rPSk

i“1

?nq, thus we have the second term ϕJΛ´n,h1∆Σpw‹h ´ wpn,hq “ OPp1{nq. Similarly, for the last term we have

From standard results for ridge regression, we have }w‹h ´ wpn,h}2 “ OPp1{

ϕJrn ď }ϕ}´}Λ´n,h1}2op}∆Σ}op}∆b}op ` }∆Σ}op}bs}op¯ “ OPp1{nq. Therefore, both terms are negligible at the ?

¨ scale. Condition on pXh,tϵi,huin“1q the only randomness comes from S. By finite-population sampling theory,

ÿn

Var˚´1 ζ ÿ

ϕr,hϵr,h¯ “

1 ´ ζ ζ

ϕi,hϕiJ,hσ2.

rPS

i“1

Therefore,

σ2ϕJΛ´n,h1´ ÿn i“1

Var˚´ϕJpwppnk,hq ´ wpn,hq¯ “

ϕi,hϕiJ,h¯Λ´n,h1ϕ ` oP

1 ´ ζ ζ

`

˘

1{n

σ2ϕJ´Λ´n,h1 ´ λΛ´n,h2¯ϕ ` oP

1 ´ ζ ζ

`

˘

1{n

“

1 ´ ζ ζ

`

˘

σ2ϕJΛ´n,h1ϕ ` oP

1{n

“

Finally, by the conditional strong law of large numbers, we have

d1 ´ ζ ζ

cVar˚pϕJwppnk,hqq ÝÑP

σbϕJΛ´n,h1ϕ.

bhboot,K pϕq “ std´␣

(¯ Ña.s.

ϕJwppnk,hqˇ1 ď k ď K

| |
|---|

