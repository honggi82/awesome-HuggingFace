# arXiv:2604.26326v2[cs.LG]10May2026

## Addressing Performance Saturation for LLM RL via Precise Entropy Curve Control

Bolian Li, Yifan Wang, Yi Ding, Anamika Lochab, Ananth Grama, Ruqi Zhang Purdue University, West Lafayette, IN, USA Correspondence to: li4468@purdue.edu, ruqiz@purdue.edu

### Abstract

Reinforcement learning (RL) has enabled complex reasoning abilities in large language models (LLMs). However, most RL algorithms suffer from performance saturation, preventing continued gains as RL training scales. This problem can be characterized by the collapse of entropy, a key diagnostic for exploration in RL. Existing attempts focus on preventing entropy collapse through regularization or clipping. However, their resulting entropy curves often exhibit instability in the long term, which hinders performance gains. In this paper, we introduce Entrocraft, a simple rejection-sampling approach that realizes user-customized entropy schedule by biasing the advantage distributions. Entrocraft requires no objective regularization and is advantage-estimator-agnostic. Theoretically, we relate perstep entropy change to the advantage distribution under minimal assumptions. This explains the behavior of existing RL and entropy-preserving methods. Entrocraft also enables a systematic study of entropy schedules, which reveals that linear annealing, which starts high and decays to a slightly lower target, performs best. Empirically, Entrocraft addresses performance saturation, significantly improving generalization, output diversity, and long-term training. It enables a 4B model to outperform an 8B baseline, sustains improvement for up to 4× longer before plateauing, and raises pass@K by 50% over the baseline.12

### 1 Introduction

Reinforcement learning (RL) has become the dominant approach for aligning with human preference and realizing multi-step reasoning ability in large language models (LLMs) [31, 2, 27]. Despite these successes, many RL algorithms still underperform anticipated performance limits: as training scales, performance saturates earlier than expected, leaving additional data and compute unable to translate into further improvements [13, 28, 4]. A core reason behind this saturation is the collapse of the exploration–exploitation balance, where the LLM over-commits to a narrow region of solutions and stops exploring alternative reasoning trajectories [7, 46, 22]. Empirically, this phenomenon is well captured by entropy dynamics: the frequently observed entropy collapse corresponds to a shrinking exploration ability during RL.

Recent efforts have resulted in several entropy-preserving techniques to prevent entropy drop during RL. These techniques are based on loss regularization [40], clipping [45, 7, 38], or positive-negative decoupling [52, 44]. While effectively increasing entropy, the entropy curves during RL training are still coarsely controlled. Entropy can drift too high after a few steps, which in turn makes RL unstable and thus hinders sustained performance gains. Besides, they typically control entropy indirectly

1The code is available at https://github.com/lblaoke/entrocraft. 2We also provide an interactive demo for playing with entropy curve control at https://lblaoke.github.io/

demo/entrocraft.

Preprint.

RL Flow Entrocraft Flow

[Figure 1]

[Figure 2]

[Figure 3]

Controlled ℋ Curve No Performance Saturation + Entropy Loss + Clip-Higher + Clip-Cov

Avg.pass@32over3benchmarks

79

[Figure 4]

| |
|---|
| |
|Standard RL encounters entropy collapse,|
|causing performance saturation.|
| |
| |

GRPO + Entrocraft

0.8

y1 y2 y3 y4

77

0.4

75

ℋ

y5 y6 y7 y8

GRPO

InferenceEngine

73

TrainingEngine

0.0

71

Rejection Sampling Based

0 100k 200k 300k 400k Training Samples

52 54

46 48 50

Avg. mean@32 over 3 benchmarks

on Current Entropy ℋ

16

| |
|---|
|[Figure 5]<br><br>GRPO + Entrocraft|
| |
|GRPO|
| |
| |

| |
|---|
| |
| |
| |
|[Figure 6]<br><br>Entrocraft controls entropy curve,|
|addressing performance saturation.|

0.8

AIME-25mean@32

12

y4

0.4

ℋ

8

y5 y7

0.0

4

0 100k 200k 300k 400k

0 100k 200k 300k 400k

Training Samples

Training Samples

- Figure 1: Overview of Entrocraft. It uses entropy-guided rejection sampling to filter rollouts against a target entropy, enabling precise control over the entropy curve throughout RL training. This control addresses performance saturation, a key obstacle to scaling RL. We find that a linear annealing schedule performs best, improving generalization, output diversity, and sustained training.

through the loss or update rule, making it difficult to prescribe an explicit entropy schedule over long training horizons. These drawbacks is particularly severe in long-term RL training.

To address this, we propose Entrocraft, a method for precise control over the entropy curve that allows entropy schedules to be user-customized. Fig. 1 summarizes our method, the entropy curve control, and empirical improvements. We begin with an LLM-oriented theoretical analysis of entropy change based on realistic policy assumptions. We highlight that entropy changes are negatively related to the advantage, and high model confidence amplifies such entropy changes.

Based on the theoretical results, we design a simple rejection sampling to filter out positive/negativeadvantage rollout samples when entropy is lower/higher than a threshold, biasing the advantage distribution towards the entropy-increasing/decreasing region. Since rejection sampling directly modifies the advantage distribution, it is able to move the entropy to target values within very few steps, enabling the accurate crafting of entropy curves. The method requires no entropy regularization and applies as a drop-in to existing RL algorithms.

Precise control opens a question that the field has not yet been able to ask experimentally: what entropy schedule is the best? Comparing across schedule families, we find that a simple linear annealing schedule performs best.

The main contributions of this paper can be summarized below:

- • We provide rigorous theoretical results on entropy changes grounded in realistic LLM-based policy assumptions. Entropy changes are negatively related to the advantages and high model confidence amplifies such changes.
- • We introduce a lightweight controller based on rejection sampling for entropy schedules in LLM RL. Unlike entropy regularization, clipping, or decoupling methods, Entrocraft does not modify the RL objective, and is advantage-estimator-agnostic. Entrocraft can craft the entropy curve to be user-specified entropy schedules, which is the key to addressing performance saturation.
- • Extensive experiments demonstrate the effectiveness of Entrocraft. It significantly improves generalization (a 4B model surpasses an 8B baseline), increases output diversity (AIME25 pass@32 is 50% higher than baseline), and extends the training horizon (sustaining improvement for up to 4× longer before plateauing as training scales).

### 2 Preliminaries

#### 2.1 Reinforcement Learning for LLMs

In a standard policy-gradient RL framework like Group Relative Policy Optimization (GRPO) [32] or Group Sequence Policy Optimization (GSPO) [51], the language model (or actor) we aim to train is denoted as a θ-parameterized distribution (or policy) πθ. The direct output of language

models is a softmax distribution over the entire vocabulary V, interpreted as next-token probabilities: pt = πθ(·|x,y<t). Each new token yt is drawn from pt.

Each RL step consists of rollout generation, advantage estimation, and policy update, allowing the model to explore different potential answers and learn from environment feedback. For a single question (or prompt) x, rollout generation samples a set of G responses {yi}Gi=1 from an old checkpoint πθ

(·|x). The following PPO-style objective is used by many recent RL algorithms:

sampler

|yi|

G

1 G

min rI(t) · Aˆ(x, yi), clip rI(t), 1 − ϵlow, 1 + ϵhigh A ˆ(x, yi) , (1)

J (θ) =

t=1

i=1

where rI(t) = π πθ(yi,t|x,yi,<t)

θsampler(yi,t|x,yi,<t) is the importance sampling ratio, and Aˆ is the estimated advantage. Our theoretical analysis is based on a simplified policy-gradient objective that does not consider clipping or importance sampling: J (θ) = Ey∼π

θ(·|x)Aˆ(x,y), and thus the per-step policy update is: ∆θ = η · ∇θJ (θ) = η · Ey∼π

θ(·|x)[Aˆ(x,y) · ∇θ log πθ(y|x)], (2) where η is the learning rate.

#### 2.2 Entropy of LLMs

The predictive entropy of LLMs provides a principled measurement of model uncertainty and serves as an indicator of response diversity and exploration capability. For a single question x and

|V|

answer y, the aggregated entropy is computed as: H = −|y1| |ty=1|

i=1 pt,i log pt,i. The expected entropy, averaged over all prompts in a batch and their corresponding rollout samples, serves as an indicator of how LLMs’ exploration capability evolves during RL. This evolution is known as entropy dynamics [30, 38]. In this paper, we primarily study entropy change during RL updates:

∆H = H(p + δp) − H(p), (3) to enable accurate and per-step entropy control.

### 3 Theoretical Analysis: How Entropy Evolves during LLM RL

This section presents theoretical results on entropy changes during RL training. We use these results to interpret the entropy dynamics of existing RL algorithms, particularly in long-running scenarios. Our analysis extends prior work [23, 7, 44, 38, 33] to a more realistic setting that does not require the actor to follow a tabular softmax policy3. The resulting bounds are direct and easy to interpret, avoiding the complicated covariance and expectation terms that appear in prior analyses [23, 38].

- 3.1 From Advantages to Entropy Changes The analysis begins with two fundamental questions: (i) What is the sign of entropy change ∆H, and

- (ii) what is its magnitude? These questions help us predict entropy change at each RL step, revealing how advantages affect entropy dynamics. To obtain exact analytical results, we make minimal assumptions about the actor policy and advantage distribution, only requiring that the learning rate is sufficiently low, as stated in Assumption 1.

Assumption 1 (Proximity of Policy Updates) We assume the learning rate η in Eq. (2) is small enough that the Taylor expansion approximation of policy probability updates holds (i.e., ∥δp∥22 ≪ ∥δp∥1). This is a standard assumption in continuous optimization, and is satisfied in practice by modern adaptive optimizers like Adam [19] with typical learning rates (e.g., 10−6 ≤ η ≤ 10−4).

- Theorem 1 (Token-Level Entropy Change in LLMs) Consider a single policy-gradient update step of the form in Eq. (2). Let pk be the probability that token k is sampled during rollout generation.

3The tabular softmax policy assumes θ = z, where logits are model parameters. However, in realistic LLM settings, the logits z are the functions of model parameters θ, and even a simple MLP module would make this assumption invalid.

Then the sign entropy change ∆H (Eq. (3)) triggered by token k is opposite to that of its estimated advantage Aˆk:

δpi δpk

p−

Aˆk · ∆H ≤ 0, whenever pk >

i ,

i∈V\{k}

where δpi is the probability change at this RL step.

- Theorem 2 (Sequence-Level Entropy Change in LLMs) Consider a single policy-gradient update step of the form in Eq. (2), and assume that all tokens share the same outcome reward. Let

pt,i = πθ(yt = i|x,y<t) be the probability that i is sampled as the t-th token in the sequence. The sign of the entropy change ∆H (Eq. (3)) triggered by response y is opposite to that of the estimated

advantage Aˆ(x,y):

|y|

− δpδpt,yt,i

Aˆ(x,y) · ∆H ≤ 0, whenever πθ(y|x) ≥

t,i ,

p

t

t=1 i∈V\{yt}

where δpt,i is the probability change at this RL step.

We provide theoretical guarantees in Theorem 1 and 2 for token-level entropy and sequence-level entropy4 respectively, and outline their proofs in Appendix B. Intuitively, both theorems state that entropy changes are negatively related to the advantage, provided the probability of rollout samples is high enough to be above a baseline constant:

Entropy Change ∝ − Advantage × (Log Likelihood of Rollout Sample − Output Space Baseline), (4)

where the output space baseline is: − |ty=1| i∈V\{yt}

δπθ(yt=i|x,y<t)

δπθ(yt|x,y<t) log πθ(yt = i|x,y<t).

Theorem 2 suggests that positive-advantage rollout samples lead to an entropy drop if the model confidence πθ(y|x) is above the output space baseline. We further give empirical evidence to support this condition in Fig. 2a, where we compare the log likelihoods (confidence) and output space baselines of training Qwen3-4B-Base under positive (RAFT++ [41]), zero-mean (GRPO [32]), and negative (NSR [52]) advantage estimators. Fig. 2a shows that the log likelihood is significantly higher than the output space baseline in all cases, verifying the condition for Aˆ(x,y) · ∆H ≤ 0 to hold.

Entropy collapse/explosion in RL is a predictable consequence of advantage-weighted updates. Our results show that positive-advantage updates tend to reduce entropy, while negative-advantage updates tend to increase it. As a result, entropy collapse becomes the default, once training is dominated by positive advantages. This explanation also justifies the “accuracy-entropy tradeoff” [7] in standard RL algorithms, where accuracy increase leads to negative entropy changes. However, the theoretical results also suggest that entropy changes are not directly related to the model performance. It is possible to maintain entropy while still improving rewards if the algorithms selectively choose which advantage regions contribute to the policy gradients.

#### Takeaway for Entropy Analysis

We prove the relationship between advantages and entropy changes for LLM-based policies. Entropy changes are negatively related to the advantage, and high model confidence will amplify such entropy changes.

#### 3.2 Interpreting the Entropy Dynamics of Existing Methods

The entropy dynamics of RL algorithms are important indicators of training stability and explorationexploitation balance. Our theoretical results reveal a clear relationship between entropy change and advantage, explaining the entropy behavior of existing RL algorithms and their performance limitations. We discuss why existing RL algorithms exhibit specific entropy dynamics in the following discussion.

4NOTE: These theoretical results are based on the entropy computed from the learner policy πθ.

| |
|---|

- (a) All advantage estimators lead to sufficiently high confidence (b) Positive-sample confidence is consistently higher

Figure 2: Empirical justification for entropy analysis. (a) All advantage estimators lead to sufficiently large log likelihoods on average, so that the model confidence condition in Theorem 2 always holds.

- (b) The log likelihoods of positive-advantage samples are always larger than those of negative samples, justifying why GRPO/GSPO encounters entropy collapse even with normalized advantages.

Standard RL Algorithms. Our theoretical results imply a categorization of existing RL algorithms that do not explicitly consider entropy. There are three types of algorithms based on their advantage statistics: (i) In positive-advantage RL like RAFT [8] and RAFT++ [41], most RL steps lead to entropy drop; (ii) in negative-advantage RL like NSR [52], most RL steps lead to entropy increase;

- (iii) in zero-mean-advantage RL like GRPO [32] and GSPO [51], empirical results show that the entropy still tends to decrease. We interpret this phenomenon by comparing the training dynamics of Qwen3-4B-Base, and find that this is due to the overconfidence of positive samples, as shown in Fig. 2b. Models are consistently more confident in the positive samples, allowing the negative entropy changes to dominate the training dynamics.

Clipping. Many recent efforts leverage the clipping technique to address entropy collapse, including DAPO [45], ADAPO [29], ClipB/V [38], and Clip-Cov [7]. The mechanism behind clipping is the removal of high-advantage and/or high-confidence tokens, which biases the advantage distribution toward 0-mean. Our theory explains why this works: it reduces expected |∆H| and thereby alleviates entropy drop.

Positive-Negative Decoupled RL. Recent studies also propose decoupled objectives for positive (correct) and negative (incorrect) rollout samples, respectively, inspired by the empirical finding that negative-only RL increases entropy [52]. This approach is well explained by our theoretical framework, as it explicitly enforces the sign of advantages. For example, W-Reinforce [52] modifies the coefficients of positive RL: JW-Reinforce = λ · Jpos − Jneg, to weaken the entropy drop triggered by the positive objective; EntroPIC [44] further makes the coefficients adjustable: JEntroPIC = (1 + α(H)) · Jpos − (1 − α(H)) · Jneg, and eventually converges to a targeted entropy value.

### 4 Methodology

In this section, we introduce our entropy-control framework (Entrocraft), which builds upon a simple rejection-sampling filter. We begin with rejection sampling in rollout generation (Section 4.1), and then introduce the dynamic rejection sampling filter for entropy control (Section 4.2). Finally, we discuss our insights on entropy curve annealing, highlighting that, for the first time, entropy in RL can be tuned just like learning-rate schedules (Section 4.3).

#### 4.1 Rejection Sampling as a Simple Entropy Controller

Our theoretical results suggest that entropy change is not directly tied to model performance. Entropy can remain stable or even increase while training accuracy improves, as long as the positive-advantage rollout samples are filtered out and no longer contribute to the policy gradients. This behavior can be realized by rejection sampling.

Our key observation is that entropy collapse/explosion is a consequence of uncontrolled gradient updates. From the theoretical results in Section 3.1, the subset of rollouts contributing to the gradient determines whether an update is entropy-decreasing or entropy-increasing. The sign of ∆H can be controlled by selecting which rollouts enter the policy gradient. Therefore, rather than developing

Algorithm 1: Entrocraft Inputs: Question x, original rollout samples {yi}Gi=1, current policy πθ, advantage estimator Aˆ,

and the entropy range (hlow,hhigh). Outputs: The actual rollout samples used for the RL update Sx. Sx ← ∅; Compute the current entropy H; m ← I(H > hhigh) − I(H < hlow) ; /* entropy out-of-range indicator */

for i = 1 .. G do

if m · Aˆ(x,yi) ≥ 0 then

Sx ← Sx ∪ {yi} ; /* rejection sampling */ end

end Return Sx.

new RL objectives or adding an auxiliary entropy loss, we find that a simple rejection-sampling filter at rollout generation suffices to precisely control entropy changes.

For example, to increase entropy, we apply rejection sampling to retain only the negative subset: Sx = {yi|Aˆ(x,yi) < 0}, and the RL training objective becomes:

|y|

1 |Sx| y∈S

min rI(t) · Aˆ(x, y), clip rI(t), 1 − ϵlow, 1 + ϵhigh A ˆ(x, y) , (5)

Jrej(θ) =

t=1

x

with the only difference from the standard RL objective (Eq. (1)) highlighted in color. Rejection sampling provides a simple, objective-agnostic entropy control knob, retaining the strengths of existing RL algorithms while eliminating the risk of entropy collapse or explosion. As it directly modifies the advantage distribution, the filter is responsive enough to move entropy to target values within a few steps, enabling the accurate crafting of entropy curves shown later in Section 4.3. The cost is comparable to or lower than standard RL, as only accepted samples contribute to the gradient computation. This can also be monitored by the effective rollout batch sizes as shown in Appendix C.3.

#### 4.2 Stabilizing and Crafting Entropy Dynamics

Entropy dynamics have been used to monitor the training stability of RL [50]. In a stable training run, the entropy curve should be within a reasonable range, neither low enough to trigger performance saturation [7, 25, 18], nor high enough to cause numerical overflow [50].

To realize stable entropy dynamics, we apply the rejection sampling filter to dynamically encourage or discourage the exploration of LLMs. The acceptance probability of rejection sampling depends on the current batch entropy H against a target range (hlow,hhigh), in which we use an entropy out-of-range indicator: m = I(H > hhigh) − I(H < hlow) to measure the direction of entropy drift. When entropy is too low, the filter rejects most high-advantage rollouts while retaining lower- and negative-advantage ones. When entropy is too high, the filter retains positive-advantage rollouts and rejects most negative samples, steering RL updates toward entropy reduction. The full procedure is given in Algorithm 1.

Entrocraft provides a plug-and-play entropy control framework, applicable to all policy-gradient methods. It treats the entropy curve as a controllable training hyperparameter in the same spirit as a learning-rate schedule, making the training dynamics of RL stable and customizable.

#### 4.3 Precise Entropy Curve Control and Annealing Schedules

The Long-Term RL Challenge. A growing body of work has shown that RL tends to sharpen the base policy around existing solutions rather than discover new ones [17, 47, 46, 48], a behavior consistent with the entropy collapse observed empirically. Once the policy becomes slightly more confident in a small subset of correct solutions, such solutions will be sampled more often, which

- Table 1: Overview of entropy-preserving baselines. We compare the methods along three properties: (i) Can the method reach a target entropy value? (ii) Can it control entropy curves? (iii) Does it apply to any policy-gradient method?

Method Reach Target Entropy Control Entropy Curve Algorithm-Agnostic

Entropy Loss ✓ Clip-Higher [45] ✓ Clip-Cov [7] ✓ W-Reinforce [52] EntroPIC [44] ✓ Entrocraft (ours) ✓ ✓ ✓

further increases their likelihood. The problem is exacerbated in long-term RL. As training rewards rise, the advantage distribution becomes increasingly imbalanced and heavy-tailed, leaving fewer negative-advantage samples to counteract the drift. By Theorems 1 and 2, these positive-advantage and high-likelihood solutions are mostly entropy-decreasing. The self-reinforcing feedback loop would lead to entropy collapse just within a few steps.

A Constant Entropy Target Is Not Enough. This fragility motivated us to stress-test Entrocraft under long-term RL training. As demonstrated in Appendix C.6, Entrocraft with a slightly higher constant entropy target would become unstable and fluctuate a lot eventually. We attribute this instability to the imbalance of rollouts, which makes the negative samples so scarce in the long term that Entrocraft’s entropy-increasing steps (rejecting all positive samples) rely on very few samples.

Curve Control with Annealing Schedules. To address this, we propose to anneal the entropy curves as training proceeds. For example, we set the initial entropy target to be around 0.6, and gradually lower this target toward 0.2 during RL training. This can stabilize the training dynamics

- as it reduces the unstable entropy-increasing steps in the later phase of RL. We compare different annealing schemes in Section 5.3, finding that the simple linear-decaying entropy curve achieves the best performance. This annealing design is uniquely enabled by Entrocraft. It converts entropy in RL from a passive training diagnostic into a controllable hyperparameter, extending the toolkit for tuning RL performance to any policy-gradient method.

### 5 Experiments

In this section, we present empirical results to demonstrate the effectiveness of the proposed Entrocraft algorithm. Specifically, we show a comprehensive benchmark comparison in Section 5.2, elaborate on the entropy curve annealing schemes in Section 5.3, and discuss the case of long-term RL in Section 5.4.

#### 5.1 Settings

Data and Models. The experiments described in this section focus on math reasoning tasks, using Numina-Math [3] (440K questions in total) as the training set. We hold out a 100K subset for general RL experiments, and the full-size dataset is used for long-term RL experiments. We primarily demonstrate the RL results using Qwen3-4B-Base [43], as well as the comparison with larger models (Qwen3-8B-Base and Qwen3-14B-Base) and models from different model families (Llama-3.1-8B-Instruct) [9].

RL Algorithms and Baselines. We primarily use the proposed Entrocraft algorithms to augment GRPO [32] and GSPO [51]. In comparison with Entrocraft, we also implement other entropypreserving methods on top of these RL algorithms, including loss-regularization (entropy loss), clipping (Clip-Higher [45] and Clip-Cov [7]), and positive-negative decoupled RL (W-Reinforce [52] and EntroPIC [44]).5 The implementation follows the standard verl framework [34]. Other training details and hyper-parameters are summarized in Appendix C.1.

Evaluation. The evaluation scheme consists of AMC-23 [20], and AIME-24/25/26 [36]. Following previous works [41, 52, 44], we randomly sample 32 answers per question, with temperature set to 0.6. Due to space constraints, we show results from the full AIME experiments in Appendix C.4.

5The related clipping method ADAPO [29] is not compared as it is primarily used for tool-use LLMs, incompatible with our experiments.

- Table 2: Evaluations on math reasoning benchmarks. The proposed Entrocraft consistently improves the final performance of RL algorithms, better than other entropy-preserving methods. The best scores are in bold, and the second-best scores are marked with underlines.

MATH-500 AMC-23 AIME-25 Avg. Score mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 mean@32 pass@32

Method

GRPO 75.3 89.4 57.4 92.5 8.9 40.0 47.2 74.0 + Entropy Loss 76.0 89.8 55.0 92.5 9.9 33.3 47.0 71.9 + Clip-Higher 75.8 91.4 55.5 95.0 10.8 40.0 47.4 75.5 + Clip-Cov 76.8 91.8 57.1 90.0 10.8 33.3 48.2 71.7 + Entrocraft 79.0 93.0 65.0 95.0 15.1 46.7 53.5 78.2

GSPO 75.6 90.2 57.7 92.5 8.9 33.3 47.4 72.0 + Entropy Loss 74.7 90.6 55.9 97.5 11.9 33.3 47.5 73.8 + Clip-Higher 75.5 90.4 54.7 95.0 10.0 40.0 46.7 75.1 + Clip-Cov 75.3 91.2 56.3 92.5 10.0 40.0 47.2 74.6 + Entrocraft 78.7 92.2 56.1 95.0 14.7 40.0 49.8 75.7

W-Reinforce 74.9 90.8 53.5 90.0 11.3 40.0 46.6 73.6 EntroPIC 73.7 90.2 55.3 92.5 10.9 43.3 46.6 75.3

20 21 22 23 24 25

20 21 22 23 24 25

(a) Model Size v.s. AIME-25

(b) pass@K curve on AIME-25

(c) pass@K curve on MATH-500

- Figure 3: Effectiveness of Entrocraft across model size and inference cost. (a) A 4B model can outperform an 8B model with Entrocraft; (b-c) Entrocraft improves inference-time scaling, with pass@K growing faster than under standard GRPO.

#### 5.2 Benchmark Evaluation

We first conduct general RL experiments and evaluate the final checkpoints on math reasoning benchmarks, as shown in Table 2 and Fig. 3. We demonstrate that the proposed Entrocraft outperforms all other baselines under both mean@32 and pass@32 settings. Fig. 3a highlights that Entrocaraft can enable Qwen3-4B-Base to outperform Qwen3-8B-Base trained using standard GRPO. The pass@K experiments in Fig. 3b-3c also demonstrate that Entrocraft successfully prevents the actor from collapsing to just a few solutions, which benefits inference-time scaling.

#### 5.3 Crafting Entropy Curves

We demonstrate the entropy control capability of Entrocraft. The entropy control powered by the proposed entropy-guided rejection sampling filter is powerful enough that entropy curves can be crafted directly, much like learning-rate schedules. As empirical evidence of entropy annealing introduced in Section 4.3, we comprehensively compare three entropy annealing schemes (fixed target, linear decay, and cosine decay) in Fig. 4.6 The fixed-target scheme, similar to previous entropy-control studies [44, 29], becomes unstable after the first 200k training samples. Its entropy curve fluctuates sharply, leading to a drop in performance. In contrast, decaying schemes eliminate the instability and sustain improvement even after 400k training samples.

#### 5.4 Long-Term RL

Finally, we demonstrate continual improvement in long-term RL powered by the proposed Entrocraft. As shown in Fig. 5 and Table 3, standard RL algorithms like GRPO [32] improve smoothly through the first 100K training samples. However, we observe only minimal sustained gains thereafter, a phenomenon known as performance saturation [7, 25, 18]. In contrast, entropy-preserving methods alleviate this saturation and achieve better final performance. Among all entropy-preserving methods, the proposed Entrocraft achieves the best long-term performance, surpassing all compared baselines

6We set the fixed entropy target to be 0.5. Both linear and cosine decaying use annealing entropy range schedules, starting

- at (0.6, 0.7) and ending at (0.1, 0.2)

(a) Training Reward (b) Entropy (c) KL Loss

(d) MATH-500 mean@32 (e) AIME-25 mean@32 (f) AIME-26 mean@32

- Figure 4: Long-term training dynamics of three entropy annealing schemes implemented in Entrocraft. The fixed-target scheme becomes unstable due to rollout imbalance. Both linear and cosine decay schemes remain stable and sustain improvement, and linear decay is slightly better. x-axis: the number of samples used for training.

at any of the 4 stages. We attribute this to its precise entropy control, which prevents entropy from drifting and avoids entropy instability common in uncontrolled entropy-preserving methods [45, 7]. As a concrete illustration, Clip-Cov [7] suffers a performance drop after 300K training samples, caused by entropy explosion (Fig. 5a) that destabilizes training in the later phase.

(a) Entropy (b) MATH-500 mean@32 (c) AIME-25 mean@32

- Figure 5: Long-term performance comparison. Entrocraft accurately controls the entropy dynamics to be linearly decaying, and thus prevents the performance saturation and significantly improves over GRPO. x-axis: the number of samples used for training.

- Table 3: Long-term RL comparison at the first 100-400K training samples. GRPO suffers from performance saturation after 100K samples due to entropy collapse, while entropy-preserving methods overcome this saturation. Among all, Entrocraft is the most stable and well-performing one.

Method

MATH-500 mean@32 AIME-25 mean@32 100K 200K 300K 400K 100K 200K 300K 400K GRPO 75.2 75.4 75.6 76.6 9.9 10.1 9.6 9.4

+ Clip-Higher 76.0 77.6 78.4 78.8 9.6 12.3 12.8 13.6 + Clip-Cov 76.5 78.1 79.7 77.5 10.9 12.5 14.2 15.0 + Entrocraft 76.9 78.9 79.3 79.9 13.1 14.2 14.8 16.2

- 6 Conclusion

This paper introduces Entrocraft, a simple and precise entropy-control method that addresses entropy collapse and the consequent performance saturation in RL. We first provide an LLM-oriented theoretical analysis of what drives entropy change, then design a rejection-sampling-based method that accurately controls entropy by biasing the advantage distribution. Experiments demonstrate that Entrocraft enables highly customizable entropy curves and consistently outperforms existing entropypreserving methods across benchmarks. Entrocraft integrates as a drop-in to any policy-gradient method, enabling continual improvement on more data without saturation.

### Ethics and Broader Impact Statement

The paper does not involve human-subject data collection, personally identifiable information, or deployment in safety-critical settings. Potential risks include enabling stronger reasoning models that may also be misused in broader downstream applications. However, the paper primarily contributes a training-stability technique rather than a new capability domain. We document datasets, implementation details, hyperparameters, and compute requirements, supporting external scrutiny while discouraging inappropriate use.

### References

- [1] Milton Abramowitz and Irene A Stegun. Handbook of mathematical functions with formulas, graphs, and mathematical tables, volume 55. US Government printing office, 1948.
- [2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.
- [3] Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul, Ziju Shen, Roman Soletskyi, and Lewis Tunstall. Numinamath 7b cot. https://huggingface.co/ AI-MO/NuminaMath-7B-CoT, 2024.
- [4] Michael Beukman, Khimya Khetarpal, Zeyu Zheng, Will Dabney, Jakob Foerster, Michael Dennis, and Clare Lyle. Preventing learning stagnation in ppo by scaling to 1 million parallel environments. arXiv preprint arXiv:2603.06009, 2026.
- [5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [7] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.
- [8] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023, 2023.
- [9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. In Neural Information Processing Systems. Curran Associates, 2024.
- [10] Florian Grötschla, Ahmet Solak, Luca A Lanzendörfer, and Roger Wattenhofer. Benchmarking music generation models and metrics via human preference studies. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [12] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International conference on machine learning, pages 1352–1361. PMLR, 2017.
- [13] Zhenyu Hou, Pengfan Du, Yilin Niu, Zhengxiao Du, Aohan Zeng, Xiao Liu, Minlie Huang, Hongning Wang, Jie Tang, and Yuxiao Dong. Does rlhf scale? exploring the impacts from data, model, and method. arXiv preprint arXiv:2412.06000, 2024.
- [14] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025.

- [16] Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237–285, 1996.
- [17] Aayush Karan and Yilun Du. Reasoning with sampling: Your base model is smarter than you think. arXiv preprint arXiv:2510.14901, 2025.
- [18] Minwu Kim, Safal Shrestha, and Keith Ross. Training reasoning models on saturated problems via failure-prefix conditioning. arXiv preprint arXiv:2601.20829, 2026.
- [19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [20] Knovel Engineering. AMC-23 Dataset. https://huggingface.co/datasets/knoveleng/ AMC-23, 2025. Hugging Face dataset.
- [21] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [22] Pengyi Li, Elizaveta Goncharova, Andrey Kuznetsov, and Ivan Oseledets. Back to basics: Revisiting exploration in reinforcement learning for llm reasoning via generative probabilities. arXiv preprint arXiv:2602.05281, 2026.
- [23] Jiacai Liu. How rl policy entropy converges during updates. Blog, 2025. URL https://zhuanlan. zhihu.com/p/28476703733.
- [24] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling, 2025.
- [25] TaiMing Lu, Lingfeng Shen, Xinyu Yang, Weiting Tan, Beidi Chen, and Huaxiu Yao. It takes two: On the seamlessness between reward and policy model in rlhf. In ICML Workshop on Foundation Models in the Wild, 2024.
- [26] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 1928–1937. PmLR, 2016.
- [27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [28] Seohong Park, Kevin Frans, Deepinder Mann, Benjamin Eysenbach, Aviral Kumar, and Sergey Levine. Horizon reduction makes rl scalable. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [29] Aleksei Petrenko, Ben Lipkin, Kevin Chen, Erik Wijmans, Marco Francis Cusumano-Towner, Raja Giryes, and Philipp Kraehenbuehl. Entropy-preserving reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026.
- [30] Yi Ren and Danica J Sutherland. Learning dynamics of llm finetuning. In The Thirteenth International Conference on Learning Representations, 2025.
- [31] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [32] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [33] Han Shen. On entropy control in llm-rl algorithms. arXiv preprint arXiv:2509.03493, 2025.
- [34] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.
- [35] Vaishnavi Shrivastava, Ahmed Hassan Awadallah, Vidhisha Balachandran, Shivam Garg, Harkirat Behl, and Dimitris Papailiopoulos. Sample more to think less: Group filtered policy optimization for concise reasoning. In First Workshop on Foundations of Reasoning in Language Models, 2025.

- [36] Haoxiang Sun, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Challenging the boundaries of reasoning: An olympiad-level math benchmark for large language models. arXiv preprint arXiv:2503.21380, 2025.
- [37] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.
- [38] Shumin Wang, Yuexiang Xie, Wenhao Zhang, Yuchang Sun, Yanxi Chen, Yaliang Li, and Yanyong Zhang. On the entropy dynamics in reinforcement fine-tuning of large language models. arXiv preprint arXiv:2602.03392, 2026.
- [39] Lilian Weng. Reward hacking in reinforcement learning. lilianweng.github.io, Nov 2024. URL https: //lilianweng.github.io/posts/2024-11-28-reward-hacking/.
- [40] Ronald J Williams and Jing Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268, 1991.
- [41] Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.
- [42] Wujiang Xu, Wentian Zhao, Zhenting Wang, Yu-Jhe Li, Can Jin, Mingyu Jin, Kai Mei, Kun Wan, and Dimitris N Metaxas. Epo: Entropy-regularized policy optimization for llm agents reinforcement learning. arXiv preprint arXiv:2509.22576, 2025.
- [43] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [44] Kai Yang, Xin Xu, Yangkun Chen, Weijie Liu, Jiafei Lyu, Zichuan Lin, Deheng Ye, and Saiyong Yang. Entropic: Towards stable long-term training of llms via entropy stabilization with proportional-integral control. arXiv preprint arXiv:2511.15248, 2025.
- [45] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [46] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [47] Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [48] R Zhao, A Meterez, SM Kakade, C Pehlevan, S Jelassi, and E Malach. Echo chamber: Rl post-training amplifies behaviors learned in pretraining. The Conference on Language Modeling (COLM), 2025.
- [49] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: Experiences on scaling fully sharded data parallel. Proceedings of the VLDB Endowment, 16(12):3848–3860, 2023.
- [50] Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, et al. Stabilizing reinforcement learning with llms: Formulation and practices. arXiv preprint arXiv:2512.01374, 2025.
- [51] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [52] Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

### A Discussion

- A.1 Related Works

Reinforcement Learning with Verifiable Rewards Reinforcement Learning [16] has become the dominant approach in the post-training of LLMs, in which LLMs generate a set of rollout trajectories (exploration) and then enforce the rewarded trajectories while punishing the less-rewarded ones (exploitation). Defining the reward scores for LLMs’ trajectories is a critical challenge. Reinforcement Learning from human feedback (RLHF) [27] separately trains a reward model (RM) as the proxy of human preference. However, RM-based reward has been shown to suffer from significant reward hacking problems [39], which makes the LLMs poorly generalized to new questions. Luckily, post-training of LLMs has increasingly focused on tasks that exhibit long, complex trajectories with simple verification, including math reasoning [6, 10] and code generation [5, 15]. For example, the famous o1 [14] and R1 [11] use RL with verifiable rewards to elicit the complex reasoning capability of LLMs, and the GRPO [32] has become the default RL algorithm for training many reasoning models. Recent works propose many variants of GRPO to mitigate its drawbacks, including the length inflation (Dr. GRPO [24] and GFPO [35]) and training instability (GSPO [51] and Zheng et al. [50]). The key challenge among these drawbacks is the scalability of RL algorithms: with more data and compute, will RL algorithms continually improve the model performance?

Entropy Dynamics of Reinforcement Learning The entropy has been used as a key indicator for models’ exploration during RL [40, 26, 12], which is essential for achieving the upper limit of model performance [37]. The entropy collapse phenomenon has long been regarded as the default behavior of RL algorithms, trading exploration potential for higher rewards [7, 17, 47]. However, recent synthetic theoretical analysis reveals that the entropy collapse is related to the bias of the advantage distribution, and it is possible to improve performance without an entropy drop [23, 7, 44, 38, 33]. Further, this paper (Entrocraft) and a concurrent work [29] prove the relationship between entropy changes and advantages under realistic LLM settings. Empirically, the techniques used to avoid entropy collapse include entropy loss7, clipping (Clip-Higher [45], Clip-Cov [7], and ClipB/V [38]), and positive-negative decoupled RL (W-Reinforce [52], EntroPIC [44], and ADAPO [29]). However, the entropy control implemented using the above methods is still not responsive enough, which makes the entropy curves still an observation-based metric, not customizable. In contrast, the proposed Entrocraft makes the entropy control accurate enough for entropy curve crafting, turning entropy dynamics from an observation-only metric to a hyperparameter like learning-rate schedules.

- A.2 Limitation and Future Directions

This paper focuses on the stability and continual improvement of RL algorithms, and has validated the method’s effectiveness on standard math reasoning tasks with dense models. However, in more challenging settings like multi-turn RL and mixture-of-expert (MoE) models, the entropy instability is more catastrophic [42, 50] due to the more sparse solution space and the variable policies. The current method design is optimized for single-turn math reasoning task. We plan to extend the entropy analysis and entropy-control algorithm to such settings.

### B Proofs

#### B.1 Proof of Theorem 1

Token-Level Entropy Change of LLMs. Consider a single RL update step. Let pk be the probability that token k is sampled during rollout generation. The sign of the entropy change triggered by token k is bounded by the sign of the advantage score:

###### Aˆk · ∆H ≤ 0,

− δpδpi

with probability 1 − i∈V\{k} p

i , where δpi is the probability change at this RL step. Proof:

k

Let p be the probability vector before the RL update and δp be the update. The entropy change can be approximated by Taylor expansion [1]:

7Shen [33] indicates that entropy loss only works well under traditional RL tasks where the discrete action space is small, and this effect is marginal for LLM RL.

∆H = H(p + δp) − H(p)

∂H ∂pi

δpi + O(∥δp∥22)

=

i∈V

(1 + log pi) · δpi + O(∥δp∥22)

= −

i∈V

=(a) −

δpi log pi + O(∥δp∥22).

i∈V

(6)

- Here, (a) is due to the constraint on output probabilities: i∈V pi ≡ 1. Then, we identify the condition for entropy to decrease (i.e., ∆H < 0). This occurs when:

δpi log pi > 0. (7)

i∈V

For token k with positive advantage Aˆk > 0, the RL update increases its probability: δpk > 0. By the probability conservation constraint i∈V δpi ≡ 0, we have i∈V\{k} δpi = −δpk < 0. Then, the entropy change can be rewritten as:

∆H = −δpk log pk −

δpi log pi + O(∥δp∥22). (8)

i∈V\{k}

Since δpi = δpk δpδpi

for i ̸= k, we obtain:

k

  + O(∥δp∥22)

 log pk +

δpi δpk

∆H = −δpk

log pi

i∈V\{k}

(9)

 log pk − log

  + O(∥δp∥22).

− δpδpi

= −δpk

k

p

i

i∈V\{k}

− δpδpi

Entropy decreases (∆H < 0) when the term in parentheses is positive: log pk > log i∈V\{k} p

i . This condition holds with probability 1 − i∈V\{k} p

k

− δpδpi

i , which approaches 1 as probability mass concentrates on

k

token k. Symmetrically, for tokens with negative advantage, entropy increases. Therefore, Aˆk · ∆H ≤ 0 holds with high probability.

□

#### B.2 Proof of Theorem 2

Sequence-Level Entropy Change of LLMs. Consider a single RL update step and assume all tokens share the same outcome reward. Let pt,i = πθ(yt = i|x, y<t) be the probability that i is sampled as the t-th token in the sequence. The sign of the entropy change triggered by response y is bounded by the sign of the advantage score:

|y|

Aˆ(x, y) · ∆H ≤ 0, if πθ(y|x) ≥

t=1 i∈V\{yt}

δpt,i δpt,yt

−

p

t,i ,

where δpt,i is the probability change at this RL step. Proof:

Similar to the token-level entropy, by Taylor expansion [1], the sequence-level entropy change can be expressed as:

 

 

|y|

|y|

∂H ∂pt,i · δpt,i + O

∥δpt∥22

∆H =

t=1 i∈V

t=1

 

 

|y|

|y|

1 |y|

∥δpt∥22

= −

(1 + log pt,i) · δpt,i + O

t=1 i∈V

t=1

(10)

 

 

|y|

|y|

1 |y|

=(b) −

∥δpt∥22

δpt,i · log pt,i + O

t=1 i∈V

t=1

  + O

 

  .

 log pt,yt +

|y|

|y|

1 |y|

δpt,i δpt,yt · log pt,i

∥δpt∥22

= −

δpt,yt

i∈V\{yt}

t=1

t=1

- Here, (b) is due to the constraint on output probabilities: i∈V pt,i ≡ 1. Now assume the estimated advantage is positive: Aˆ(x, y) > 0. The corresponding probability changes for the entire response are then all positive:

mint δpt,yt > 0. To simplify the expression, we define the effective token probability change as the weighted average of all independent token probabilities:

δpt,i

|y| t=1 δpt,yt · log pt,yt + i∈V\{y

δpt,yt · log pt,i

t}

> 0. (11)

δpy =

δpt,i

|y| t=1 log pt,yt + i∈V\{y

δpt,yt · log pt,i

t}

The entropy change can then be simplified as:

 log

 

|y|

|y|

δpt,i δpt,yt

−

1 |y|

∆H ≈ −

· δpy ·

pt,yt − log

p

t,i

t=1 i∈V\{yt}

t=1

(12)

 log πθ(y|x) − log

  ,

|y|

δpt,i δpt,yt

−

1 |y|

· δpy ·

= −

p

t,i

t=1 i∈V\{yt}

showing that entropy change is negatively related to advantage when the likelihood of the generated response

δpt,i δpt,yt

−

exceeds a threshold: πθ(y|x) > |ty=1| i∈V\{y

t,i . The same derivation applies to the negative advantage case (A(x, yˆ) < 0). Therefore, Aˆ(x, y) · ∆H ≤ 0 holds when the rollout sample likelihood is sufficiently high.

t} p

□

### C Additional Experiment Details

#### C.1 Implementation Details

The implementation is based on the verl framework [34], which uses vLLM [21] for inference and FSDP2 [49] for training. All experiments are conducted on a node of 8 NVIDIA H100 GPUs. We summarize the core hyperparameters in Table 4. The average training time for Qwen3-4B-Base is around 10K training samples per hour.

#### C.2 Entropy Curves of Baselines

We show the full entropy curves of baselines in Fig. 6. Standard GRPO [32] exhibits entropy collapse. On top of GRPO, many entropy-preserving methods successfully alleviate the entropy-decreasing trend. However, they are not necessarily responsive enough to enable accurate entropy control and may cause the entropy curves to be unstable in the long term. In contrast, the proposed Entrocraft accurately stabilizes entropy at the target (0.8), making exploration controllable as a hyperparameter.

- Table 4: Core hyperparameters for training implementation. The origin of each hyperparameter is listed, and all empty-source hyperparameters are determined by our experiments.

#### Hyperparameter Value Source

train_batch_size 1024 verl examples ppo_mini_batch_size 8×32 verl examples max_context_length 1024 + 3072 Xiong et al. [41] rollout.n 8

optim.lr 1e-6 verl examples kl_loss_coef 1e-3 verl examples

val_kwargs.n 32 val_kwargs.temperature 0.6

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- Figure 6: Entropy curve comparison across baselines. Other entropy-preserving methods may induce instability during long-term training or may not be sufficiently responsive. In contrast, the proposed Entrocraft effectively controls the entropy curve to be the customized shape.

#### C.3 Effective Batch Size

To explicitly show the effect of Entrocraft on the number of rollout samples used for gradient update, we pick the steps that trigger rejection sampling, and show their effective batch sizes throughout the training in Fig 7. The RL training is GRPO + Entrocraft with initial rollout.n=8, and the entropy range setting is the linear decaying as used in Section 5.3. The dropping effective batch sizes verify that Entrocraft requires less gradient computation than naive GRPO.

#### C.4 Full Evaluation on AIME

In addition to the benchmark evaluations of Table 2, we list the full AIME results in Table 5. The conclusions on all AIME benchmarks are consistent with Table 2. Entrocraft outperforms all other entropy-preserving methods across diverse RL algorithms.

#### C.5 Results on Other Models

Aside from Qwen3-4B-Base, we also try Entrocraft on larger models and models from different model families, as shown in Table 6, 7, and 8. The results report consistent improvement over GRPO across different base models.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- Figure 7: Effective batch sizes for the RL steps affected by Entrocraft. The negative samples become less as the model’s performance improves.

- Table 5: Full Evaluations on AIME. The proposed Entrocraft consistently improves the final performance of RL algorithms, better than other entropy-preserving methods. The best scores are in bold, and the second-best scores are marked with underlines.

Method

AIME-24 AIME-25 AIME-26 mean@32 pass@32 mean@32 pass@32 mean@32 pass@32

GRPO 7.7 30.0 8.9 40.0 8.1 33.3 + Entropy Loss 9.0 26.7 9.9 33.3 9.1 26.7 + Clip-Higher 10.2 23.3 10.8 40.0 9.1 36.7 + Clip-Cov 9.7 26.7 10.8 33.3 10.9 36.7 + Entrocraft 16.7 36.7 16.5 46.7 15.2 40.0

GSPO 8.8 26.7 8.9 33.3 8.8 33.3 + Entropy Loss 7.9 23.3 11.9 33.3 9.6 36.7 + Clip-Higher 10.0 20.0 10.0 40.0 9.5 36.7 + Clip-Cov 9.7 26.7 10.0 40.0 9.1 36.7 + Entrocraft 10.2 26.7 14.7 40.0 13.3 36.7

W-Reinforce 8.8 26.7 11.3 40.0 9.3 26.7 EntroPIC 9.9 20.0 10.9 43.3 9.7 30.0

- Table 6: Benchmark results on Llama-3.1-8B-Instruct. Entrocraft demonstrates consistent improvements over GRPO. The best scores are in bold.

Method

MATH-500 AMC-23 AIME-25 Avg. Score mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 GRPO 56.2 82.2 30.8 75.0 10.4 16.7 32.5 58.0

+ Entrocraft 57.1 85.2 36.8 85.0 18.8 26.7 37.6 65.6

- Table 7: Benchmark results on Qwen3-8B-Base. Entrocraft demonstrates consistent improvements over GRPO. The best scores are in bold.

MATH-500 AMC-23 AIME-25 Avg. Score mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 GRPO 77.2 91.8 59.7 92.5 13.1 43.3 50.0 75.9

Method

+ Entrocraft 78.4 94.0 63.5 97.5 13.8 46.7 51.9 79.4

- C.6 Failure Case: Maintaining High Entropy May Induce Instability in The Long Term

To discuss the boundary of entropy control methods, we use Entrocraft with different entropy targets and record their training dynamics in long-term RL. We show two distinct examples in Fig. 8. We find that an overly high entropy level introduces instability into RL training, and such instability will accumulate and become more catastrophic in the long term. This also validates our design of entropy curve annealing schemes.

- Table 8: Benchmark results on Qwen3-14B-Base. Entrocraft demonstrates consistent improvements over GRPO. The best scores are in bold.

MATH-500 AMC-23 AIME-25 Avg. Score mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 mean@32 pass@32 GRPO 80.8 92.2 66.7 97.5 14.7 40.0 54.1 76.6

Method

#### + Entrocraft 81.7 93.8 67.1 97.5 17.0 53.3 55.3 81.5

- Figure 8: Failure case study. Overly high entropy level introduces instability into RL training, making the empirical performance fragile to subtle perturbations.

