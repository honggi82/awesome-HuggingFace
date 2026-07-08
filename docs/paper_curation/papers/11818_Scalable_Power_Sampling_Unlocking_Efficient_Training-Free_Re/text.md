## Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening

Xiaotong Ji*1 Rasul Tutunov*1 Matthieu Zimmer1 Haitham Bou Ammar12

# arXiv:2601.21590v1[cs.LG]29Jan2026

### Abstract

Reinforcement learning (RL) post-training is a dominant approach for improving the reasoning performance of large language models (LLMs), yet growing evidence suggests that its gains arise primarily from distribution sharpening rather than the acquisition of new capabilities. Recent work has shown that sampling from the power distribution of LLMs using Markov chain Monte Carlo (MCMC) can recover performance comparable to RL post-training without relying on external rewards; however, the high computational cost of MCMC makes such approaches impractical for widespread adoption.

In this work, we propose a theoretically grounded alternative that eliminates the need for iterative MCMC. We derive a novel formulation showing that the global power distribution can be approximated by a token-level scaled low-temperature one, where the scaling factor captures future trajectory quality. Leveraging this insight, we introduce a training-free and verifier-free algorithm that sharpens the base model’s generative distribution autoregressively.

Empirically, we evaluate our method on math, QA, and code tasks across four LLMs, and show that our method matches or surpasses one-shot GRPO without relying on any external rewards, while reducing inference latency by over 10× compared to MCMC-based sampling.

### 1. Introduction

The current frontier of large language models (LLMs) reasoning is largely defined by reinforcement learning (RL) post-training. By optimising models against automated verifiers, such as unit tests in coding tasks (Chen et al., 2021;

*Equal contribution 1Huawei Noah’s Ark Lab 2UCL Centre for Artificial Intelligence. Correspondence to: Haitham Bou Ammar <haitham.ammar@huawei.com>.

Preprint. January 30, 2026.

Roziere et al., 2023; Hou et al., 2024), final-answer checks or Lean certificates in mathematics (Cobbe et al., 2021; Yang et al., 2023; Ahn et al., 2024; Xin et al., 2024; Zimmer et al., 2025), methods like Group Relative Policy Optimisation (GRPO) and its variants have established state-of-theart performance across a range of benchmarks (Zhao et al., 2023; Shao et al., 2024; Yu et al., 2025). However, the mechanisms underlying these gains remain the subject of active debate. A growing body of evidence suggests that RL may not introduce fundamentally new reasoning capabilities, but instead acts as a form of distribution sharpening (He et al., 2025; Song et al., 2025; Yue et al., 2025; Gai et al., 2025; Ni et al., 2025). Under this view, the reasoning trajectories required to solve complex problems already exist within the base model; RL primarily reshapes the probability mass.

If post-trained models are indeed “sharpened” versions of the base model, it follows that comparable reasoning capabilities should be attainable by directly targeting a sharpened distribution of the base LLM without relying on external validation rewards. Recent work by Karan & Du (2025) provides compelling evidence for this view. They make use of the power distribution, pα(x), as a principled mechanism for inducing such sharpening without any additional training. Strikingly, their results show that sampling from the power distribution enables base models to match and, in some cases, surpass the performance of RL-trained models on challenging benchmarks, such as MATH500 and GPQA. These findings strongly support the hypothesis that the reasoning capabilities attributed to RL are largely latent within the base model and can be revealed through an appropriate sampling strategy.

Although the power distribution offers a compelling mechanism for sharpening base model capabilities, current methods for sampling from it are computationally prohibitive. Karan & Du (2025) argue that standard autoregressive sampling methods like those in low-temperature sampling (Wang et al., 2020) fail to capture the global properties of pα. Consequently, they resort to MCMC techniques, specifically a Metropolis-Hastings algorithm that iteratively resamples a token subsequence. While effective, this approach introduces significant latency, requiring multiple forward passes and iterative refinement steps to generate a single response.

Such computational overhead restricts the method’s scalability for real-world deployment.

To enable broad adoption, we seek a faster and more scalable sampling procedure while retaining the performance advantages of MCMC. To this end, we revisit the fundamental relationship between the power distribution and the low-temperature distribution. While prior work has established that these distributions are theoretically distinct, we derive a novel formulation showing that the power distribution can be effectively approximated by a suitably scaled low-temperature policy. We explicitly identify the required scaling factor and theoretically quantify the accuracy of this approximation. This insight allows us to bypass the iterative cost of MCMC entirely: by proposing a (training-free and verifier-free) algorithm that applies this scaling during standard autoregressive generation, we efficiently target the sharpened distribution, retaining the reasoning benefits of the power distribution while reducing inference latency by over 10×. In short, our contributions can be summarised as:

- (1) We bridge the theoretical gap between global power distributions and local autoregressive sampling. We prove that the power distribution can be decomposed into a scaled low-temperature one, deriving a closedform scaling factor and providing rigorous bounds on the approximation error.
- (2) We introduce a training- and verifier-free algorithm that enables direct autoregressive generation from the sharpened distribution. This eliminates the iterative overhead of MCMC, reducing inference latency by over 10× while preserving the reasoning improvements of the global power distribution.
- (3) Across four LLMs and three benchmarks (MATH500, HumanEval, GPQA), we show that our method matches or surpasses one-shot GRPO performance on in-domain and out-of-domain tasks, without any parameter updates or external rewards.

### 2. Power & Low-Temperature Distributions

In prior work, Karan & Du (2025) show that the power distribution is fundamentally distinct from standard lowtemperature sampling. We formalise both distributions here and discuss those challenges. Let p(·|q) denote the conditional probability distribution induced by a pretrained language model on sequences from a finite vocabulary V conditioned on an input prompt q. Given a parameter α > 1 and a partially generated token sequence x0:t−1 = {x1,...,xt−1}, we define the low-temperature (Wang et al., 2020) and the associated power distributions

for the next token xt as follows:

pα(xt|q,x0:t−1) x′t∈V pα(x′

p(αlow. temp)(xt|q,x0:t−1) =

, (1)

t|q,x0:t−1)

pα(q,x0:t−1,xt,xt+1:T) xt:T pα(q,x0:t−1,xt:T)

p(αpow)(xt|q,x0:t−1) = xt+1:T

,

Equation 1 shows that sampling from the low-temperature distribution, p(αlow. temp), is relatively straightforward, as it depends only on the current history and token-level probabilities. In contrast, sampling from the power distribution is substantially more challenging, since it requires marginalising over all possible future completions in both the numerator and denominator of p(αpow):

pα(q,x0:t−1,xt,xt+1:T) xt:T pα(q,x0:t−1,xt:T)

p(αpow)(xt | q,x0:t−1) = xt+1:T

,

which we colour-coded in blue and green in the above equation for clarity. Of course, those marginalisations require summing over all possible future token sequences, whose cardinality grows exponentially with the remaining horizon T − t, i.e., |V|T−t for a vocabulary V. This intractability makes direct sampling impractical in general and motivates the need for approximate inference procedures.

It also highlights that the low-temperature and power distributions are fundamentally different: whereas lowtemperature sampling applies a local transformation at each token, the power distribution induces a global, trajectorylevel reweighting. In fact, Karan & Du (2025) show that these two distributions generally do not coincide and can differ at every token position.

Approximate Sampling from p(αpow). To approximately sample from this sharper power distribution, Karan & Du (2025) resort to approximate inference via Markov chain Monte Carlo (MCMC). They employ a Metropolis–Hastings scheme that constructs a Markov chain over full token sequences, using the base model likelihoods to define acceptance ratios. Concretely, their method iteratively proposes partial resampling of token subsequences and accepts or rejects these proposals to asymptotically target the power distribution. To improve practicality in the autoregressive setting, they further introduce a variant that samples from a sequence of intermediate, prefix-level power distributions, gradually extending the sequence while running a bounded number of MCMC steps at each stage.

This approach inherits MCMC’s classical high-dimensional limitations, where mixing times can be exponential (Levin & Peres, 2017; Gheissari et al., 2018). Practically, this causes significant overhead: Karan & Du (2025) reports an approximately 8.8× cost increase over standard sampling for MATH500 sequences. Motivated by this, we derive an

efficient approximation via a new principled connection between low-temperature sampling and the power distribution. This eliminates iterative MCMC loops whilst preserving future-aware reweighting behaviours.

### 3. Approximating Power Distributions

We now turn to the central technical contribution of this work. Although prior results (Karan & Du, 2025) establish that low-temperature sampling generally does not recover the power distribution, we show that the power distribution nevertheless admits a structured token-level decomposition that directly relates it to low-temperature sampling. This observation allows us to move beyond a binary distinction between the two and instead characterise their relationship explicitly. In particular, we prove that the power distribution can be expressed as a scaled version of the low-temperature distribution p(αlow. temp). This result provides a principled foundation for approximating power-distribution sampling using standard autoregressive procedures, and forms the basis of the efficient inference methods developed in the remainder of this section.

The following theorem makes this decomposition explicit by providing a closed-form expression for the scaling term that relates the power distribution to the low-temperature counterpart at each token position:

Theorem 3.1. Let p(x) be a pretrained large language model, q an input prompt and α > 1 be an exponent parameter. Then, for any partially generated sequence x0:t−1 = {x0,...,xt−1} we have:

pα(xt|q,x0:t−1)ζt(xt) x′t∈V pα(x′

p(αpow)(xt|q,x0:t−1) =

,

t|q,x0:t−1)ζt(x′

t)

(2) where for each token x′t ∈ V, we have that: ζt(x′t) =

xt+1:T pα(xt+1:T|q,x0:t−1,x′t).

Theorem 3.1 reveals that the difference between powerdistribution sampling and low-temperature sampling is entirely captured by a token-specific scaling factor that reflects the expected likelihood of future completions. While low-temperature sampling favours tokens that are locally probable under the model, the power distribution further reweights these tokens according to the average quality of the sequences they induce downstream. In this sense, power sampling can be viewed as low-temperature sampling augmented with a future-aware correction. Importantly, this decomposition localises the global, trajectory-level dependence of the power distribution into a per-token scalar term, explaining both why MCMC-based methods were previously required and how they can be avoided. As the generation progresses and the remaining horizon shortens, the power distribution naturally approaches the low-temperature

one, where the two exactly coincide on the final token xT: p(αpow)(xT|q,x0:T−1) = p(αlow. temp)(xT|q,x0:T−1).

#### 3.1. Autoregressive Monte Carlo Estimation

The key insight of Theorem 3.1 is that power-distribution sampling can be performed autoregressively, provided that each token probability is modulated by an appropriate tokenwise scaling factor ζt(x′t). Crucially, this scaling factor admits an expectation form under the base language model, which allows it to be estimated via standard autoregressive Monte Carlo sampling. As a result, the power distribution can be efficiently approximated without resorting to trajectory-level inference such as MCMC.

Formally, for any candidate token x′t ∈ V, the scaling factor can be rewritten as an expectation with respect to the base language model:

ζt(x′t) = Ex

t+1:T∼pf(·) pα−1(xt+1:T|q,x0:t−1,x′t) ,

with pf(·) ≡ p(xt+1:T|q,x0:t−1,x′t) denoting the base language model’s distribution over future completions. Thus, a

Monte Carlo approximation of ζt(x′t) can be easily attained as follows: i) generate Mt i.i.d. completions {x[tr+1:] T}M

t

r=1 from the LLM when conditioned on {q,x0:t−1,x′t}, and ii) compute an empirical estimate of ζt(x′t) as:

1 Mt

ζˆt(x′t) =

Mt

pα−1(x[tr+1:] T|q,x0:t−1,x′t). (3)

r=1

Substituting those Monte Carlo estimates into the decomposition of Theorem 3.1 yields the following approximation to the power distribution:

pα(xt | q,x0:t−1)ζˆt(xt) x′t∈V pα(x′

pˆ(pow)α (xt | q,x0:t−1) =

,

t | q,x0:t−1)ζˆt(x′

t)

(4) with ζˆt(xt) denoting the scaling factor at the token xt.

#### 3.2. Bias Analysis and Correction

Although each scaling factor is estimated without bias using ζˆ(xt) and ζˆ(x′t), the substitution step back in Theorem 3.1 to form pˆ(αpow)(·) induces bias in the resulting estimator, since the expectation of a ratio does not equal the ratio of expectations. Although one could directly use this estimator in practice, we instead explicitly analyse the source of this bias and introduce a jackknife-based correction that significantly reduces it while preserving the efficiency of the method.

The following Lemma shows that the bias of the estimator pˆ(αpow)(xt|q,x0:t−1) decays asymptotically at a rate inversely proportional to the sampling budget Mt:

Lemma 3.2. Let pˆ(αpow)(·|q,x0:t−1) be the Monte Carlo estimator of the power distribution defined in Equation 4. Then, we have:

E p ˆ(αpow)(·|q,x0:t−1) = p(αpow)(·|q,x0:t−1) + O

1 Mt

,

where the expectation E[·] is taken with respect to the random completions used to estimate the scaling factors ζt(x′t) for all tokens x′t ∈ V.

From the above, we see that the bias can be reduced by increasing the number of sampled trajectories Mt from the base language model. However, since each new sample requires additional autoregressive generation, this approach quickly becomes computationally expensive. We therefore seek an alternative that reduces the bias more rapidly with respect to Mt, enabling accurate estimation with substantially fewer samples from the LLM.

Jackknife Estimator for Bias Reduction. To do so, we adopt a jackknife correction. The jackknife is a classical bias-reduction technique that constructs corrected estimators by systematically recombining leave-one-out versions of an original estimator (Miller, 1974; McIntosh, 2016). When applied to estimators that depend smoothly on empirical averages, the jackknife removes the leading-order bias term while preserving the overall computational structure. This makes it particularly well-suited to our setting, where the bias arises from taking ratios of Monte Carlo estimates.

The key idea is to construct a bias-reduced estimator by combining the original Monte Carlo approximation with a collection of leave-one-out variants, each obtained by removing a single sampled trajectory. This construction yields the following jackknife-corrected estimator of the power distribution:

pˆ(α,powJK)(xt|q,x0:t−1) = Mtpˆ(αpow)(xt|q,x0:t−1) (5) −

Mt

Mt − 1 Mt

pˆ(α,pow−s)(xt),

s=1

where pˆ(α,pow−s)(xt) denotes the leave-one-out Monte Carlo estimator obtained by excluding the sth trajectory. Here, we

define the leave-one-out-estimation of ζt(x′t) as:

Mt

1 Mt − 1

ζˆt,(LOO−s )(x′t) =

pα−1(x[tr+1:] T|q,x0:t−1,x′t),

r=1,r̸=s

(6) This, in turn, leads us to the following definition of the leave-one-out approximation of the power distribution:

pα(xt|q,x0:t−1)ζˆt,(LOO−s )(xt) x′t∈V pα(x′

pˆ(α,pow−s)(xt) =

,

t|q,x0:t−1)ζˆt,(LOO−s )(x′

t)

which when substituted back in Equation 5 gives us the final Jackknife estimator pˆ(α,powJK)(xt|q,x0:t−1).

The following lemma formalises the benefit of this construction by showing that the jackknife estimator cancels the leading-order bias term, yielding a bias that decays at an inverse-quadratic rate in Mt:

Lemma 3.3. Let pˆ(α,powJK)(·|q,x0:t−1) be the jackknife estimator of the power distribution defined in Equation 5. Then

E p ˆ(α,powJK)(·|q,x0:t−1) = p(αpow)(·|q,x0:t−1) + O

1 Mt2

,

This result has an important computational implication. For the original Monte Carlo estimator without jackknife correction, achieving a bias of at most ϵ requires Mt = O(1/ϵ) sampled trajectories. In contrast, the inverse-quadratic bias decay established above implies that the jackknife estimator attains the same bias level with only O(1/√ϵ) samples. Thus, the jackknife correction substantially reduces the number of required language-model generations for a given accuracy.

It is easy to see that an immediate corollary of Lemma 3.3 is convergence in expectation of pˆ(α,powJK)(·) to the true power distribution p(αpow)(·) as the number of Monte Carlo trajectories Mt → ∞. In Appendix A.4, we even derive high-probability guarantees on this convergence.

3.3. Scalable Power Sampling Based on the above results, this section develops a practical algorithm for scalable, efficient trajectory sampling from LLM power distributions. Algorithm 1 presents our pseudocode for the single-token setting, while Algorithm in Appendix B extends this implementation to batched token settings. Algorithm 1 approximates the global power distribution via sequential, local estimations. At each autoregressive step t, the algorithm identifies a set of promising candidates, Gt, using a Top-Kt filter on the base model’s logits. To evaluate the long-term viability of each candidate under the power objective, we perform Mt independent Monte Carlo lookahead rollouts. We can further restrict the rollout length to the next Ht tokens (see Appendix B). Such truncation further reduces inference times. These trajectories are aggregated to compute likelihood scaling factors, which are subsequently refined using a Jackknife estimator (pˆ(α,powJK)) to mitigate finite-sample bias. The next token xt is then sampled according to these bias-corrected probabilities, effectively steering the generation toward high-likelihood region token-by-token. Finally, we note that since we evaluate Kt candidates using Mt independent rollouts at each step t, the total inference burden is bounded by O Tt=0−1 KtMt generated sequences. This structured cost offers a distinct efficiency advantage over rejection sampling (Best-of-N),

Algorithm 1 Scalable Power Sampling for a Query q

this prefix, the model produces the correct answer with high probability, such that: p(ANSWER4 | PLAN,CALC) = 0.95, and the incorrect answer (ANSWER5) with complementary probability. In contrast, if the model directly selects GUESS, it outputs: p(ANSWER4 | GUESS) = 0.55, again followed by the end-of-sequence token EOS. All sequences terminate after producing an answer token.

- 1: Inputs: Power distribution exponent α ∈ [1,∞), upper

bounds on trajectory length T, size of groups {Kt}Tt=0−1, and total budgets {Mt}Tt=0−1

- 2: for t = 0 → T − 1 do
- 3: Compute Gt(x0:t−1) = Top@Kt [p(·|q,x0:t−1)]
- 4: for x′t ∈ Gt(x0:t−1) do
- 5: Sample Mt i.i.d. completions
- 6: Compute ζˆt(x′t) and ζˆt,(LOO−s )(x′t) from Eq. 3 and 6
- 7: end for
- 8: Set V = Gt(x0:t−1) in Eq. 5
- 9: Compute the Jackknife estimator
- 10: Sample xt ∈ Gt(x0:t−1) w.p. pˆ(α,powJK)(xt|q,x0:t−1)
- 11: Concatenate the sampled tokens x0:t = x0:t−1 ⊕ xt
- 12: end for
- 13: Sample the Tth token using low temperature sampling
- 14: Concatenate the sampled token x0:T = x0:T−1 ⊕ xT
- 15: Return Sampled trajectory x0:T = {x0,...,xT}

In this toy problem, the model faces an early choice between a locally attractive shortcut and a more structured reasoning strategy. Although GUESS has a higher probability than PLAN at the first step, committing to the PLAN → CALC trajectory leads to a much higher chance of correct answers.

Concretely, the four feasible sequences are

- x(1) = (PLAN,CALC,ANSWER4,EOS),
- x(2) = (PLAN,CALC,ANSWER5,EOS),
- x(3) = (GUESS,ANSWER4,EOS,EOS),
- x(4) = (GUESS,ANSWER5,EOS,EOS),

| | | | | | |
|---|---|---|---|---|---|
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

0.6

Probability

with base-model probabilities

0.4

0.2

p(x(1)) = 0.38 p(x(2)) = 0.02 p(x(3)) = 0.33 p(x(4)) = 0.27.

0.0

x(1) x(2) x(3) x(4)

x(1) x(2) x(3) x(4)

(a) Small budget: NMCMC = 2, Mt = 2

(b) Large budget: NMCMC = 50, Mt = 16

target pα low-temp τ MCMC Ours

If we apply low-temperature decoding with α = 4 (i.e., τ = 0.25), the model becomes confident in its local preference and overwhelmingly selects GUESS at the first step:

- Figure 1. Toy example comparing the target power distribution pα, the low-temperature distribution, and the empirical histograms of MCMC and our method (α = 4, τ = 1/α = 0.25).

which scales linearly as O(N) full trajectories. Unlike Bestof-N, which must “pay” for N complete generations even when errors occur early, our formulation allows for targeted compute allocation, investing budget (Kt,Mt) specifically at critical decision points to steer generation, thereby achieving high-fidelity sampling without brute-force costs.

### 4. Experiments & Results

#### 4.1. Building Intuition: Planning vs. Guessing

To build intuition, we consider a simple reasoning problem: “what is the result of 2+2”. In this example, the model generates sequences of fixed length L = 4 from:

V = {PLAN,GUESS,CALC,ANSWER4,ANSWER5,EOS}. We imagine the case when the base model assigns a higher probability to a guessing strategy than to an explicit planning strategy, reflecting a locally plausible but globally suboptimal shortcut: p(PLAN) = 0.4, and p(GUESS) = 0.6.

If the model selects PLAN, it deterministically transitions to CALC, i.e., p(CALC | PLAN) = 1. Conditioned on

0.44 0.44 + 0.64 ≈ 0.165, pτ(GUESS) ≈ 0.835.

pτ(PLAN) =

As a result, low-temperature sampling amplifies the shortcut, despite its lower probability of leading to a correct answer.

In contrast, the power distribution π(x) ∝ p(x)4 evaluates entire trajectories rather than individual tokens. Under this distribution, the structured PLAN branch dominates:

π(x(1)) ≈ 0.55, π(x(3)) ≈ 0.31, π(x(4)) ≈ 0.14, π(x(2)) ≈ 0.

This illustrates why low-temperature decoding can fail to recover the desired power distribution: sharpening local probabilities does not account for downstream consequences. While both MCMC-based power sampling and our estimator-based method converge to π with sufficient computation, their finite-budget behaviour differs substantially. MCMC tends to inherit the bias of its low-temperature proposal when few steps are used, whereas our rollout-based future correction explicitly reweights early decisions based on expected future outcomes, yielding samples that more closely match π even at small budgets (see Figure 1).

[Figure 1]

- Figure 2. Pass@1 performance and per-prompt inference time for Qwen-2.5-7B (top left), Qwen2.5-Math-7B (top right), DeepSeek-Math7B (bottom left), and DeepSeek-Math-7B-RL (bottom right).

#### 4.2. Large-Scale Evaluation

We evaluate our scalable power-distribution sampling approximation on large-scale benchmarks. We assess whether training- and verifier-free autoregressive sampling can recover RL-based post-training gains without MCMC’s inference overhead. Comparing against standard decoding, low-temperature sampling, Best-of-N, and MCMC, we characterise accuracy–latency trade-offs across mathematics, code generation, and knowledge-intensive QA.

Models. We test four base models to assess robustness across families and training regimes. We use Qwen2.57B (Team et al., 2024) to assess general reasoning, and two math-focused models, Qwen2.5-Math-7B (Team et al., 2024) and DeepSeek-Math-7B (Shao et al., 2024), to examine transferability across fine-tuning pipelines. Finally, we evaluate the RL-tuned DeepSeek-Math-7B-RL (via GRPO) (Shao et al., 2024) to determine if our method remains effective on models already sharpened by RL.

MATH500 HumanEval GPQA Qwen2.5-7B

Base 0.498 0.329 0.278 Low-temperature 0.628 0.524 0.303 Best-of-N 0.650 0.609 0.282 MCMC Power Sampling 0.706 0.622 0.318 Power Sampling (ours) 0.708 0.756 0.349

GRPO (MATH) 0.740 0.561 0.354 Qwen2.5-Math-7B

Base 0.496 0.329 0.278 Low-temperature 0.690 0.512 0.353 Best-of-N 0.684 0.512 0.343 MCMC Power Sampling 0.748 0.573 0.389 Power Sampling (ours) 0.758 0.604 0.409

GRPO (MATH) 0.785 0.537 0.399 DeepSeek-Math-7B

Base 0.362 0.415 0.333 Low-temperature 0.366 0.427 0.430 Best-of-N 0.420 0.433 0.338 MCMC Power Sampling 0.424 0.470 0.345 Power Sampling (ours) 0.464 0.487 0.364

GRPO (MATH) 0.492 0.524 0.333

Table 1. Performance comparison of Power Sampling against base models and sampling baselines.

Benchmarks. We use three complementary benchmarks. For mathematics, we use MATH500 (Lightman et al., 2023) to assess multi-step problem solving via a final-answer exact match. For code, we employ HumanEval (Chen et al., 2021), comprising 164 Python tasks assessed via functional correctness. For QA, we use GPQA-diamond (Rein et al., 2024), a set of 198 expert-authored questions in biology, physics, and chemistry designed to challenge non-experts.

Baselines. For all methods, we cap the maximum generation length at Tmax = 3072 tokens, as suggested in Karan & Du (2025); generation may terminate earlier upon emission of the EOS token. Full hyperparameter settings are provided in Appendix C. For low-temperature decoding (Wang et al., 2020), we set the sampling temperature to 1/α = 0.25 and sample autoregressively from p(lowα . temp). For the Best-ofN baseline (Stiennon et al., 2020), we generate NBoN = 32 independent completions and select the candidate with the

highest model score under the base model’s log-likelihood. For MCMC power sampling, we follow the implementation and hyperparameters of Karan & Du (2025): α = 4, block size B = 192 (i.e., Tmax/16), NMCMC = 10 MCMC steps per stage.

Our method uses the same settings as MCMC power sampling for a fair comparison (α = 4, B = 192) and performs autoregressive power sampling with block-level lookahead (see Appendix B). At each step, we restrict attention to the Top-Kt = 8 candidate tokens, estimate future-scaling factors using Mt = 8 Monte Carlo rollouts per candidate.

For RL baselines, we use the GRPO-trained checkpoint from Shao et al. (2025) for the Qwen and the publicly released DeepSeek-Math-7B-RL model for DeepSeek, both of which are post-trained on the MATH training split. We report these post-trained models as GRPO(MATH) in Table 1.

Main Results. Table 1 summarises pass@1 accuracies on MATH500, HumanEval, and GPQA across the four models. Relative to standard decoding, low-temperature sampling generally improves performance, but the gains are modest in some settings, suggesting that purely local token-level sharpening has limited ability to extract higher-quality trajectories across tasks. The Best-of-N baseline improves over lowtemperature decoding in several regimes, but remains consistently weaker than power-sampling-based methods. This behaviour is expected, as selecting the highest-likelihood completion under the base model does not explicitly target the globally sharpened power distribution. Power sampling methods show the best performance across all the sampling methods. Although MCMC and our method target the same power distribution in the limit of infinite compute, their finite-budget behaviour differs, consistent with the intuition in Section 4.1. With the same sampling budget, our estimator more effectively concentrates probability mass on high-quality trajectories, matching or exceeding MCMC performance with up to +13.4%s.

Following the evaluation protocol of Karan & Du (2025), we further examine the in-domain and out-of-domain behaviour of reinforcement-learning-based post-training; rows titled GRPO(Math) in Table 1. Specifically, we consider a GRPO model trained exclusively on mathematical data and evaluate its generalisation performance not only on MATH500, but also on the out-of-domain tasks of code generation and knowledge-intensive question answering. This setting allows us to assess whether improvements induced by reward-based post-training transfer extend beyond the domain on which the reward signal was defined. Overall, our results reproduce the conclusions of Karan & Du (2025) regarding the effectiveness of MCMC-based power sampling, suggesting that a substantial fraction of the gains attributed to reward-based post-training can be recovered through principled sampling from the base model alone.

Power Sampling Post-Trained LLMs. Table 2 examines whether power sampling can further improve performance when applied to an already post-trained model. Specifically, we consider DeepSeek-Math-7B-RL (Shao et al., 2024), which has been post-trained via GRPO on the MATH training split, and evaluate whether power-based sampling remains effective beyond standard decoding.

MATH500 HumanEval GPQA DeepSeek-Math-7B-RL

Base 0.492 0.524 0.333 Low-temperature 0.412 0.524 0.303 Best-of-N 0.492 0.507 0.297 MCMC Power Sampling 0.494 0.530 0.349 Power Sampling (ours) 0.502 0.549 0.364

Table 2. Pass@1 accuracy on MATH500, HumanEval, and GPQA for DeepSeek-Math-7B-RL (GRPO-MATH).

95

85

Accuracy(%)

75

65

55

45

1 2 4 8 16

k

Base

GRPO

MCMC

Ours

| |
|---|

| |
|---|

| |
|---|

Figure 3. Pass@K performance on MATH500 with Qwen2.5Math-7B of the base model, GRPO, MCMC, and our method. GRPO improves pass@1 but quickly shows diversity collapse.

Across all three benchmarks, power sampling on top of the GRPO-trained model yields gains over standard decoding. Notably, low-temperature sampling degrades performance on MATH500 and GPQA (e.g., MATH500: 0.492→0.412), suggesting that after GRPO post-training, the token distribution is already sharpened in a different way and additional local temperature scaling can be ineffective. In contrast, both MCMC power sampling and our autoregressive approximation provide small improvements, with our method achieving the best performance across all benchmarks. Overall, these results indicate that power-based sampling can still enhance an RL-post-trained model, though the magnitude of improvement is smaller than for non-post-trained LLMs, consistent with the view that GRPO already performs a form of distribution sharpening.

Inference Efficiency. Having established that we achieve improved pass@1 performance, we now turn to understanding the inference-time efficiency gains of our approximation. Figure 2 reports inference time for MCMC and our autoregressive approximation across all four models and three benchmarks. Across all settings, our method is consistently faster, achieving speedups of up to approximately 10×. In the slowest MCMC regime (Qwen2.5-Math-7B on MATH500), MCMC requires an average of 2.5 minutes per prompt, compared to 0.22 minutes for our method. Even in the closest regime (DeepSeek-Math-7B on GPQA), MCMC still incurs a substantially higher cost, taking 0.63 minutes per prompt versus 0.25 minutes for our approach. We also notice that completion lengths are similar between MCMC and our method (∼700 tokens on average), indicating that the observed speedups result from eliminating iterative MCMC refinement rather than differences in output length, while preserving the benefits of power-based sampling.

#### 4.3. Diversity Analysis & Ablation

Pass@K Results. We report pass@K performance for K∈ {1,2,4,8,16} by sampling K independent completions

|[Figure 2]|
|---|

- 4 6 8

10 16

=tt

MATH

|[Figure 3]|
|---|

1 2 4 5 8

4 6 8

10 16

HumanEval

|[Figure 4]|
|---|

1 2 4 5 8

4 6 8

10 16

GPQA

[Figure 5]

0.55

0.60

0.65

0.70

0.75

[Figure 6]

0.52

0.54

0.56

0.58

0.60

[Figure 7]

0.300

0.325

0.350

0.375

0.400

Figure 4. Pass@1 on MATH500, HumanEval, and GPQA for Qwen2.5-Math-7B under different hyperparameters.

per prompt and marking a problem as solved if any of the completions is correct. We summarise the results on the MATH500 obtained with Qwen2.5-Math-7B in Figure 3. Further Pass@K results can be found in Appendix C.

We note that although GRPO (Shao et al., 2025) yields a strong boost at pass@1, its gains diminish as K increases, consistent with recent observations of diversity collapse (mode-seeking behaviour that reduces coverage of alternative solution paths and suppresses pass@K performance (Yue et al., 2025)). In contrast, power-distribution sampling sharpens the sampling distribution at inference time without updating parameters, improving pass@1 over the base model while better preserving diversity; our method closely tracks base model performance for larger K (e.g., K≥ 8) and is stronger at small K (notably K∈ {1,2,4}), matching patterns reported in prior powersampling work (Karan & Du, 2025).

Ablation Study. We analyse sensitivity to the power exponent α, rollout budget Mt, and candidate set size Kt. Figure 4 shows the method is robust to Kt and Mt, where moderate values offer a favourable accuracy–compute tradeoff. In contrast, performance is sensitive to α. Intermediate values (e.g., α ∈ {4,5}) yield the strongest results, consistent with MCMC findings (Karan & Du, 2025), surpassing both the base distribution (α = 1) and overly aggressive sharpening (α = 8). For those reasons, we fix α = 4 and Kt = Mt = 8 in our experiments. Further analysis and results on additional models are provided in Appendix C.

- 5. Related Works

1 2 4 5 8

Reinforcement Learning and Distribution Sharpening. RL post-training has emerged as a dominant paradigm for enhancing the reasoning capabilities of LLMs. Approaches such as Rejection Sampling Fine-tuning (Liu et al., 2023; Xiong et al., 2025) and GRPO and its variants (Shao et al., 2024; Yu et al., 2025) optimise models against automated verifiers; see the surveys in Xu et al. (2025); Zhang et al. (2025) for potential applications and an in-depth discussion.

Inference-Time Reasoning and Sampling. Various inference-time strategies exist to bypass training complexity. While low-temperature sampling (Wang et al., 2020)

sharpens local probabilities, it ignores global quality, and Best-of-N (Brown et al., 2024; Huang et al., 2025) only implicitly targets rewards; we outperform both (see Section 4.2). Although our use of Monte Carlo rollouts shares conceptual roots with Tree of Thoughts (Yao et al., 2023) and MCTS (Zhang et al., 2023; 2024; Cao et al., 2025), our approach avoids their complex state tracking and backtracking, remaining a pure sampling method without persistent memory. Similarly, while speculative (Leviathan et al., 2023; Chen et al., 2023; Zimmer et al., 2024) and lookahead decoding (Fu et al., 2024) use “draft-then-verify” parallelism like our batched algorithm (Appendix B), they aim to preserve p(x) for speed, whereas we explicitly alter it to pα(x) for reasoning. Finally, inference-aware training (Vijayakumar et al., 2016; Wang et al., 2022; Chow et al., 2024) is orthogonal to our method and can be combined in the future.

Most relevantly, Karan & Du (2025) utilised the concept of Power Distribution Sampling, proving that sampling from a globally sharpened distribution pα allows base models to match or exceed the performance of RL-trained models. However, their proposed implementation relies on MCMC, which suffers from high latency due to iterative resampling and slow mixing times in high-dimensional token spaces (Levin & Peres, 2017; Gheissari et al., 2018). Our approach targets the same theoretical distribution but employs a novel autoregressive approximation, eliminating the need for iterative MCMC and significantly reducing inference latency. Theoretically, targeting pα aligns with sampling from an annealed or tempered distribution (Neal, 2001), a concept widely used in statistical physics and diffusion models to avoid mode collapse. While recent works have applied annealing to diffusion guidance, we demonstrate its efficacy for discrete autoregressive reasoning. We also note that sampling from the power distribution can also be viewed as sampling from an energy-based model with inverse temperatures; see Ackley et al. (1985); LeCun et al. (2006).

### 6. Conclusions and Future Work

We demonstrate that latent reasoning capabilities in base models are accessible via distribution sharpening. By bridging global distributions and local policies (Theorem 3.1), our training-free sampler matches GRPO on MATH500, HumanEval, and GPQA. It achieves this with over 10× lower latency than MCMC, providing a scalable alternative to post-training pipelines.

Future work includes reducing variance via control variates (Lavenberg & Welch, 1981), optimising compute via adaptive budgets (Alomrani et al., 2025), and amortising rollout costs with speculative decoding (Leviathan et al., 2023). We also plan to extend our method to agentic settings (Lu et al., 2024; Grosnit et al., 2025; Yuksekgonul et al., 2026).

### Impact Statement

Democratisation of Advanced Reasoning. The primary impact of this work is the democratisation of highperformance reasoning capabilities. By demonstrating that base models can match the performance of RL-finetuned models (e.g., GRPO) through inference-time sampling alone, we reduce the necessity for massive computational resources typically required for post-training alignment. This lowers the barrier to entry for academic researchers and smaller organisations, allowing them to deploy state-of-theart reasoning systems without access to large-scale GPU clusters for training.

Environmental Considerations. From an environmental perspective, our approach offers a favourable trade-off. While our autoregressive lookahead mechanism increases inference cost relative to standard decoding, it eliminates the substantial carbon footprint associated with iterative RL fine-tuning and reduces inference energy consumption by an order of magnitude compared to MCMC-based baselines. We believe that this represents a significant step towards “Green AI” for high-reliability tasks.

Safety and Dual-Use Risks. However, our method also introduces specific safety considerations. Because our algorithm amplifies the latent probability mass of the base model (“distribution sharpening”) without an external reward model or human feedback loop, it relies entirely on the safety alignment of the underlying base model. If a base model contains latent hazardous knowledge or biases, our method could theoretically “sharpen” these behaviours just as effectively as it sharpens mathematical reasoning. Consequently, we strongly recommend that this sampling strategy be applied to base models that have undergone rigorous safety filtering or instruction tuning before deployment.

Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Cao, P., Men, T., Liu, W., Zhang, J., Li, X., Lin, X., Sui, D., Cao, Y., Liu, K., and Zhao, J. Large language models for planning: A comprehensive and systematic survey. arXiv preprint arXiv:2505.19683, 2025.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., and Jumper, J. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde de Oliveira Pinto, H., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chow, Y., Tennenholtz, G., Gur, I., Zhuang, V., Dai, B., Thiagarajan, S., Boutilier, C., Agarwal, R., Kumar, A., and Faust, A. Inference-aware fine-tuning for best-ofn sampling in large language models. arXiv preprint arXiv:2412.15287, 2024.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Fu, Y., Bailis, P., Stoica, I., and Zhang, H. Break the sequential dependency of llm inference using lookahead decoding. arXiv preprint arXiv:2402.02057, 2024.

Gai, J., Zeng, G., Zhang, H., and Raghunathan, A. Differential smoothing mitigates sharpening and improves llm reasoning. arXiv preprint arXiv:2511.19942, 2025.

### References

Ackley, D. H., Hinton, G. E., and Sejnowski, T. J. A learning algorithm for boltzmann machines. Cognitive science, 9

(1):147–169, 1985.

Ahn, J., Verma, R., Lou, R., Liu, D., Zhang, R., and Yin, W. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157, 2024.

Alomrani, M. A., Zhang, Y., Li, D., Sun, Q., Pal, S., Zhang, Z., Hu, Y., Ajwani, R. D., Valkanas, A., Karimi, R., et al. Reasoning on a budget: A survey of adaptive and controllable test-time compute in llms. arXiv preprint arXiv:2507.02076, 2025.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., R´e, C., and Mirhoseini, A. Large language monkeys:

Gheissari, R., Lubetzky, E., and Peres, Y. Exponentially slow mixing in the mean-field swendsen-wang dynamics. In Proceedings of the Twenty-Ninth Annual ACMSIAM Symposium on Discrete Algorithms, pp. 1981–1988. SIAM, 2018.

Grosnit, A., Maraval, A., N, R. S., Zhao, Z., Doran, J., Paolo, G., Thomas, A., Gonzalez, J., Kumar, A., Khandelwal, K., Benechehab, A., Cherkaoui, H., El-Hili, Y. A., Shao, K., Hao, J., Yao, J., K´egl, B., Bou-Ammar, H., and Wang, J. Kolb-based experiential learning for generalist agents with human-level kaggle data science performance, 2025.

He, A. W., Fried, D., and Welleck, S. Rewarding the unlikely: Lifting grpo beyond distribution sharpening. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 25559–25571, 2025.

Hou, X., Zhao, Y., Liu, Y., Yang, Z., Wang, K., Li, L., Luo, X., Lo, D., Grundy, J., and Wang, H. Large language models for software engineering: A systematic literature review. ACM Transactions on Software Engineering and Methodology, 33(8):1–79, 2024.

Huang, A., Block, A., Liu, Q., Jiang, N., Krishnamurthy, A., and Foster, D. J. Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment. arXiv preprint arXiv:2503.21878, 2025.

Karan, A. and Du, Y. Reasoning with sampling: Your base model is smarter than you think, 2025.

Lavenberg, S. S. and Welch, P. D. A perspective on the use of control variables to increase the efficiency of monte carlo simulations. Management Science, 27(3):322–335, 1981.

LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F., et al. A tutorial on energy-based learning. Predicting structured data, 1(0), 2006.

Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pp. 19274– 19286. PMLR, 2023.

Levin, D. A. and Peres, Y. Markov chains and mixing times, volume 107. American Mathematical Soc., 2017.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M., Liu, P. J., and Liu, J. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha,

- D. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

McIntosh, A. The jackknife estimation method. arXiv preprint arXiv:1606.00497, 2016.

Miller, R. G. The jackknife-a review. Biometrika, 61(1): 1–15, 1974.

Neal, R. M. Annealed importance sampling. Statistics and computing, 11(2):125–139, 2001.

Ni, K., Tan, Z., Liu, Z., Li, P., and Chen, T. Can grpo help llms transcend their pretraining origin? arXiv preprint arXiv:2510.15990, 2025.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Sauvestre, R., Remez, T., et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S. S., Lambert, N., Min, S., Krishna, R., et al. Spurious rewards: Rethinking training signals in rlvr. arXiv preprint arXiv:2506.10947, 2025.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Song, Y., Kempe, J., and Munos, R. Outcomebased exploration for llm reasoning. arXiv preprint arXiv:2509.06941, 2025.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33: 3008–3021, 2020.

Team, Q. et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3), 2024.

Vijayakumar, A. K., Cogswell, M., Selvaraju, R. R., Sun, Q., Lee, S., Crandall, D., and Batra, D. Diverse beam search: Decoding diverse solutions from neural sequence models. arXiv preprint arXiv:1610.02424, 2016.

Wang, P.-H., Hsieh, S.-I., Chang, S.-C., Chen, Y.-T., Pan, J.-Y., Wei, W., and Juan, D.-C. Contextual temperature for language modeling. arXiv preprint arXiv:2012.13575, 2020.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Xin, H., Ren, Z., Song, J., Shao, Z., Zhao, W., Wang, H., Liu, B., Zhang, L., Lu, X., Du, Q., et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152, 2024.

Xiong, W., Yao, J., Xu, Y., Pang, B., Wang, L., Sahoo, D., Li, J., Jiang, N., Zhang, T., Xiong, C., et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

Xu, F., Hao, Q., Zong, Z., Wang, J., Zhang, Y., Wang, J., Lan, X., Gong, J., Ouyang, T., Meng, F., et al. Towards large reasoning models: A survey of reinforced reasoning with large language models. arXiv preprint arXiv:2501.09686, 2025.

Yang, K., Swope, A., Gu, A., Chalamala, R., Song, P., Yu, S., Godil, S., Prenger, R. J., and Anandkumar, A. Leandojo: Theorem proving with retrieval-augmented language models. Advances in Neural Information Processing Systems, 36:21573–21612, 2023.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Yuksekgonul, M., Koceja, D., Li, X., Bianchi, F., McCaleb, J., Wang, X., Kautz, J., Choi, Y., Zou, J., Guestrin, C., et al. Learning to discover at test time. arXiv preprint arXiv:2601.16175, 2026.

Zhang, D., Huang, X., Zhou, D., Li, Y., and Ouyang, W. Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b. arXiv preprint arXiv:2406.07394, 2024.

Zhang, K., Zuo, Y., He, B., Sun, Y., Liu, R., Jiang, C., Fan, Y., Tian, K., Jia, G., Li, P., et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025.

Zhang, S., Chen, Z., Shen, Y., Ding, M., Tenenbaum, J. B., and Gan, C. Planning with large language models for code generation. arXiv preprint arXiv:2303.05510, 2023.

Zhao, S., Dang, J., and Grover, A. Group preference optimization: Few-shot alignment of large language models. arXiv preprint arXiv:2310.11523, 2023.

Zimmer, M., Gritta, M., Lampouras, G., Ammar, H. B., and Wang, J. Mixture of attentions for speculative decoding. arXiv preprint arXiv:2410.03804, 2024.

Zimmer, M., Ji, X., Tutunov, R., Bordg, A., Wang, J., and Ammar, H. B. Bourbaki: Self-generated and goalconditioned mdps for theorem proving. arXiv preprint arXiv:2507.02726, 2025.

### A. Theoretical Results

#### A.1. Proof of Theorem 3.1

Theorem A.1. Let p(x) be a pretrained large language model, q an input prompt and α > 1 be an exponent parameter. Then, for any partially generated sequence x0:t−1 = {x0,...,xt−1} we have:

pα(xt|q,x0:t−1)ζt(xt) x′t∈V pα(x′

p(αpow)(xt|q,x0:t−1) =

. (7)

t|q,x0:t−1)ζt(x′

t)

where for each token x′t ∈ V, we have that: ζt(x′t) = x

pα(xt+1:T|q,x0:t−1,x′t).

t+1:T

Proof. Let us fix some t ∈ [0,T) and consider the partially generated sequence x0:t−1 such that:

x0,...,xt−1, if t ≥ 1, {}, if t = 0.

x0:t−1 =

For a fixed token xt, let us use the following notation:

x0:t = {x0,...,xt−1,xt}, x0:t−1,x′t = {x0,...,xt−1,x′t}, xt:T = {xt,...,xT}. Consider the conditional probability:

pα(q,x0:t−1,xt,xt+1:T) xt:T pα(q,x0:t−1,xt:T)

p(αpow)(xt|q,x0:t−1) = xt+1:T

pα(q,x0:t,xt+1:T) xt+1:T pα(q,x0:t,xt+1:T) + x′

= xt+1:T

t̸=xt xt+1:T pα(q,x0:t−1x′

t,xt+1:T)

1 1 + x′

=

pα(q,x0:t−1x′t,xt+1:T )

xt+1:T

t̸=xt

pα(q,x0:t,xt+1:T )

xt+1:T

1 1 + x′

=

pα(xt+1:T |q,x0:t−1x′t)pα(x′t|q,x0:t)pα(q,x0:t−1)

xt+1:T

t̸=xt

pα(xt+1:T |q,x0:t)pα(xt|q,x0:t−1)pα(q,x0:t−1)

xt+1:T

1 1 + x′

1 1 + x′

=

=

pα(xt+1:T |q,x0:t−1x′t)

ζt(x′t) ζt(xt)

pα(x′t|q,x0:t−1) pα(xt|q,x0:t−1)

pα(x′t|q,x0:t−1) pα(xt|q,x0:t−1)

xt+1:T

t̸=xt

t̸=xt

pα(xt+1:T |q,x0:t)

xt+1:T

pα(xt|q,x0:t−1)ζt(xt) x′t∈V pα(x′

=

.

t|q,x0:t−1)ζt(x′

t)

To complete the proof, let us consider the last token case. For the sequence of tokens x = x1,...,xT we have:

pα(q,x0:T−1,xT) x′T∈V pα(q,x0:T−1,x′

p(αpow)(xT|q,x0,...xT−1) =

T)

pα(xT|q,x0:T−1)pα(q,x0:T−1) x′T∈V pα(x′

=

=

T|q,x0:T−1)pα(q,x0:T−1)

= p(αlow. temp)(xT|q,x0:T−1).

pα(xT|q,x0,...,xT−1) x′T∈V pα(x′

T|q,x0,...,xT−1)

| |
|---|

#### A.2. Proof of Lemma 3.2

- Lemma A.2. Let pˆ(αpow)(xt|q,x0:t−1) be the Monte Carlo estimator of the power distribution defined in Equation 4. Then:

1 Mt

E p ˆ(αpow)(xt|q,x0:t−1) = p(αpow)(xt|q,x0:t−1) + O

.

Proof. First, we note that:

E pα(xt|q,x0:t−1) Mr=1t pα−1(x[tr+1:] T|q,x0:t−1,xt) E x′

ζt(xt)pα(xt|q,x0:t−1) x′t∈V ζt(x′

p(αpow)(xt|q,x0:t−1) =

=

.

t)pα(x′

t|q,x0:t−1)

t|q,x0:t−1) Mr=1t pα−1(x[tr+1:] T|q,x0:t−1,x′

t∈V pα(x′

t)

Mt r=1 pα(xt|q,x0:t−1)pα−1(x[tr+1:] T|q,x0:t−1,xt) − ζt(xt)pα(xt|q,x0:t−1). Then, we have:

Let us denote δ(xt) = M1

t

ζt(xt)pα(xt|q,x0:t−1) + δ(xt) x′t∈V [ζt(x′

pˆ(αpow)(xt|q,x0:t−1) =

t)pα(x′

t|q,x0:t−1) + δ(x′

t)]

1 x′t∈V [ζt(x′

= (ζt(xt)pα(xt|q,x0:t−1) + δ(xt))

t)pα(x′

t|q,x0:t−1) + δ(x′

t)]

(ζt(xt)pα(xt|q,x0:t−1) + δ(xt)) x′t∈V ζt(x′

=

′t∈V δ(x′t) x′t∈V ζt(x′t)pα(x′t|q,x0:t−1)

t|q,x0:t−1) 1 + x

t)pα(x′

−1

t∈V δ(x′t) x′t∈V ζt(x′

ζt(xt)pα(xt|q,x0:t−1) + δ(xt) x′t∈V ζt(x′

′

1 + x

=

t)pα(x′

t)pα(x′

t|q,x0:t−1)

t|q,x0:t−1)

ζt(xt)pα(xt|q,x0:t−1) + δ(xt) x′t∈V ζt(x′

t|q,x0:t−1) ×

=

t)pα(x′

 1 − x

2 

t∈V δ(x′t) x′t∈V ζt(x′

t∈V δ(x′t) x′t∈V ζt(x′

′

′

+ x

×

t)pα(x′

t)pα(x′

t|q,x0:t−1)

t|q,x0:t−1)

3

  

 

 

 ,

δ(x′t)

+ O

x′t∈V

where in the last step we used the Taylor expansion for (1 + z)−1. The above expression can be rewritten as:

ζt(xt)pα(xt|q,x0:t−1) x′t∈V ζt(x′

δ(xt) x′t∈V ζt(x′

pˆ(αpow)(xt|q,x0:t−1) =

+

t)pα(x′

t)pα(x′

t|q,x0:t−1)

t|q,x0:t−1)

t∈V δ(x′t) x′t∈V ζt(x′

t∈V δ(x′t) x′t∈V ζt(x′

ζt(xt)pα(xt|q,x0:t−1) x′

δ(xt) x′

−

2 −

2

t)pα(x′

t)pα(x′

t|q,x0:t−1)

t|q,x0:t−1)

  

3

 

 

2

t∈V δ(x′t)

ζt(xt)pα(xt|q,x0:t−1) x′

 .

δ(x′t)

3 + O

+

x′t∈V ζt(x′

t)pα(x′

t|q,x0:t−1)

x′t∈V

Now, notice:

(8)

E[δ(x′t)] = 0 for all x′t ∈ V,

 δ(xt)

  = E δ2(xt) = O

1 Mt

δ(x′t)

for all xt ∈ V,

E

x′t∈V

  

2

 

 

 

  =

  = E

δ(x′t)

δ2(x′t) +

δ(x′t)δ(x′′t )

E

x′t∈V

x′t∈V

x′t,x′′t : x′t̸=x”t

  

3

 

 

1 Mt

  = O

δ(x′t)

E

.

x′t∈V

E δ2(x′t) = O

x′t∈V

1 Mt

,

Taking expectation E[·] in Equation 8 and using the above facts, we get:

ζt(xt)pα(xt|q,x0:t−1) x′t∈V ζt(x′

E p ˆ(αpow)(xt|q,x0:t−1) =

+ O

t)pα(x′

t|q,x0:t−1)

1 Mt

.

t(xt)pα(xt|q,x0:t−1)

Upon using p(αpow)(xt|q,x0:t−1) = ζ

x′t∈V ζt(x′t)pα(x′t|q,x0:t−1) in the above result, we eventually get:

E p ˆ(αpow)(xt|q,x0:t−1) = p(αpow)(xt|q,x0:t−1) + O

which establishes the statement of the lemma.

1 Mt

,

| |
|---|

#### A.3. Proof of Lemma 3.3

- Lemma A.3. Let pˆ(α,powJK)(·|q,x0:t−1) be the jackknife estimator of the power distribution defined in Equation 5. Then:

E p ˆ(α,powJK)(·|q,x0:t−1) = p(αpow)(·|q,x0:t−1) + O

1 Mt2

.

Proof. Let us denote for brevity α[r](x′t) = pα(x′t|q,x0:t−1)pα−1(x[tr+1:] T|q,x0:t−1,x′t) for any x′t ∈ V. From the previous lemma we can write the following result for the Monte-Carlo estimator:

Mt r=1 α[r](xt)

1 Mt

E p ˆ(αpow)(xt|q,x0:t−1) = E

Mt r=1 x′t∈V α[r](x′

1 Mt

t)

∞

bi Mti

= p(αpow)(xt|q,x0:t−1) +

,

i=1

for some collection of real numbers {bi}∞i=1. Then, for the jackknife estimator, we can write:

Mt r=1 α[r](xt)

Mt r=1,r̸=s α[r](xt)

Mt

1 Mt−1

1 Mt

Mt − 1 Mt

- E p ˆ(α,powJK)(xt|q,x0:t−1) = E Mt

−

Mt r=1 x′t∈V α[r](x′

Mt r=1,r̸=s x′t∈V α[r](x′

1 Mt

1 Mt−1

t)

t)

s=1

Mt r=1 α[r](xt)

Mt r=1,r̸=s α[r](xt)

Mt

1 Mt−1

1 Mt

Mt − 1 Mt

E

= MtE

−

Mt r=1 x′t∈V α[r](x′

Mt r=1,r̸=s x′t∈V α[r](x′

1 Mt

1 Mt−1

t)

t)

s=1

∞

∞

Mt

Mt − 1 Mt

bi (Mt − 1)i

bi Mti −

p(αpow)(xt|q,x0:t−1) +

= Mt p(αpow)(xt|q,x0:t−1) +

s=1

i=1

i=1

∞

bi Mti−1 − (Mt − 1)p(αpow)(xt|q,x0:t−1)

= Mtp(αpow)(xt|q,x0:t−1) +

i=1

∞

∞

bi Mti−1 −

bi (Mt − 1)i−1

bi (Mt − 1)i

= p(αpow)(xt|q,x0:t−1) +

− (Mt − 1)

i=1

i=1

∞

∞

(Mt − 1)i−1 − Mti−1 Mti−1(Mt − 1)i−1 = p(αpow)(xt|q,x0:t−1) +

= p(αpow)(xt|q,x0:t−1) +

bi

bici.

i=2

i=2

i−1−Mti−1 Mti−1(Mt−1)i−1

where ci = (Mt−1)

= O M 1i

t

. Hence, we have

E p ˆ(α,powJK)(xt|q,x0:t−1) = p(αpow)(xt|q,x0:t−1) + O

1 Mt2

.

| |
|---|

#### A.4. High-Probability Convergence Guarantees

Corollary A.4. Let ϵ ∈ (0,1) be any arbitrary precision parameter, δ ∈ (0,1) be arbitrary confidence parameter, and let sample complexity Mt for computing the scaling factors ζt(x′t) for any xt ∈ V be equal to Mt = O ϵ 12 log 1δ . Then, for any fixed token xt ∈ V we have:

P |pˆ(α,powJK)(xt|q,x0:t−1) − p(αpow)(xt|q,x0:t−1)| ≤ ϵ ≥ 1 − δ. Proof. Let us introduce the following notation for brevity:

Mt

1 Mt

pα−1(x[tr+1:] T|q,x0:t−1,xt), E[X¯t] = µt with {x[tr+1:] T} ∼ p(·|q,x0:t−1,xt),

X¯t =

r=1

Mt

1 Mt

′[r] t+1:T} ∼ p(·|q,x0:t−1,x′t),

′[r] t+1:T|q,x0:t−1,x′t), E[X¯] = µ with {x

X¯ =

pα−1(x

r=1 x′t∈V

Mt

1 Mt − 1

pα−1(x[tr+1:] T|q,x0:t−1,xt), E[X¯t\s] = µt with {x[tr+1:] T} ∼ p(·|q,x0:t−1,xt),

X¯t\s =

r=1,r̸=s

Mt

1 Mt − 1

′[r] t+1:T|q,x0:t−1,x′t), E[X¯\s] = µ with {x

′[r] t+1:T} ∼ p(·|q,x0:t−1,x′t).

X¯\s =

pα−1(x

r=1,r̸=s x′t∈V

In these notations, we have

Mt

X¯t\s X¯\s

X¯t X¯ −

Mt − 1 Mt

pˆ(α,powJK)(xt|q,x0:t−1) = Mt

,

s=1

µt µ

p(αpow)(xt|q,x0:t−1) =

.

Following a Chernoff bound, for any η ∈ (0,1) we have:

2Mt, P |X¯ − µ| ≥ η ≤ 2e−

P |X¯t − µt| ≥ η ≤ 2e−2η

2η2Mt |V|2

, P |X¯t\s − µt| ≥ η ≤ 2e−2η

2(Mt−1), P |X¯\s − µ| ≥ η ≤ 2e−

2η2(Mt−1) |V|2

.

From this, we have:

2Mt, 0 < µ − η ≤ X¯ ≤ µ + η w.p. at least 1 − 2e

0 < µt − η ≤ X¯t ≤ µt + η w.p. at least 1 − 2e−2η

−2η2Mt |V|2

2(Mt−1), 0 < µ − η ≤ X¯\s ≤ µ + η w.p. at least 1 − 2e

0 < µt − η ≤ X¯t\s ≤ µt + η w.p. at least 1 − 2e−2η

−2η2(Mt−1) |V|2

. Then, we have:

X¯t X¯ ≤

µt − η µ + η ≤

0 <

X¯t\s X¯\s ≤

µt − η µ + η ≤

0 <

−2η2Mt |V|2

µt + η µ − η

2Mt − 2e

w.p. at least 1 − 2e−2η

,

−2η2(Mt−1) |V|2

µt + η µ − η

2(Mt−1) − 2e

w.p. at least 1 − 2e−2η

.

From the second result, we see that:

X¯t X¯ ≤ Mt

µt − η µ + η ≤ Mt

Mt

Mt

X¯t\s X¯\s ∈ Mt

µt − η µ + η

,Mt

s=1

Combining these two results, gives us:

−2η2Mt |V|2

µt + η µ − η

2Mt + e

w.p at least 1 − 2 e−2η

##### ,

−2η2(Mt−1) |V|2

µt + η µ − η

2(Mt−1) + e

w.p at least 1 − 2Mt e−2η

##### .

Mt

X¯t\s X¯\s ∈ Mt

X¯t X¯ −

µt − η µ + η − (Mt − 1)

Mt − 1 Mt

µt + η µ − η

µt + η µ − η − (Mt − 1)

Mt

,Mt

s=1

−2η2Mt |V|2

−2η2(Mt−1) |V|2

2Mt + e

2(Mt−1) + e

− 2 e−2η

w.p. at least 1 − 2Mt e−2η

. Let us study the boundaries in the above result. To do this, notice:

µt − η µ + η

µt + η µ − η

µt µ

µ + µt µ(µ − η)

µt µ − η

µ + µt µ(µ + η)

. We start with the right hand side in Equation 9:

=

+ η

,

=

µt − η µ + η

µt − η µ + η

µ + µt µ(µ − η) − (Mt − 1)

µt µ − η

µ + µt µ(µ + η)

µt µ

µt + η µ − η − (Mt − 1)

= Mt

+ η

Mt

µt µ

µ + µt µ(µ + η) ≤

µ + µt µ(µ − η)

+ (Mt − 1)η

=

+ Mtη

µt µ

µ + µt µ(µ − η)

µ + µt µ(µ + η)

+ η

2Mt + η

2Mt

4Mt(µ + µt) µ2 − η2

µt µ

. Similarly, for the left hand side in Equation 9:

+ η

=

, (9)

(10)

µt − η µ + η − (Mt − 1)

µt + η µ − η

µ + µt µ(µ + η) − (Mt − 1)

µt µ

µ + µt µ(µ − η)

µt µ − η

Mt

= Mt

+ η

µ + µt µ(µ − η) ≥

µt µ − Mtη

µ + µt µ(µ + η) − (Mt − 1)η

=

µt µ − η

µ + µt µ(µ + η)

µ + µt µ(µ − η)

2Mt − η

2Mt

4Mt(µ + µt) µ2 − η2

µt µ − η

.

=

µ2−η2 ≤ η|V|TO(1) Hence, denoting c = η|V|TO(1), using pˆ(α,powJK)(xt|q,x0:t−1) = MtX¯

Notice, that η4M

t(µ+µt)

X¯t − Mt−1 Mt

X¯t\s X¯\s and p(αpow)(xt|q,x0:t−1) = µ

Mt s=1

µ we have:

t

pˆ(α,powJK)(xt|q,x0:t−1) ∈ p(αpow)(xt|q,x0:t−1) − c,p(αpow)(xt|q,x0:t−1) + c , w.p. at least

−2η2Mt |V|2

−2η2(Mt−1) |V|2

2Mt + e

2(Mt−1) + e

− 2 e−2η

1 − 2Mt e−2η

. Or, equivalently:

|pˆ(α,powJK)(xt|q,x0:t−1) − p(αpow)(xt|q,x0:t−1)| ≤ c, w.p. at least

−2η2(Mt−1) |V|2

−2η2Mt |V|2

2(Mt−1) + e

2Mt + e

1 − 2Mt e−2η

− 2 e−2η

##### .

Now, let us simplify the confidence probabilities:

−2η2(Mt−1) |V|2

2(Mt−1) + e

1 − 2Mt e−2η

−2η2(Mt−1) |V|2

2(Mt−1) + e

1 − 2Mt e−2η

−2η2(Mt−1) |V|2

2(Mt−1) + e

1 − 4Mt e−2η

So, in other words, we have:

−2η2Mt |V|2

2Mt + e

− 2 e−2η

≥

−2η2Mt |V|2

2Mt + e

− 2Mt e−2η

−2η2(Mt−1) |V|2

≥ 1 − 8Mte

.

≥

−2η2(Mt−1) |V|2

P |pˆ(α,powJK)(xt|q,x0:t−1) − p(αpow)(xt|q,x0:t−1)| ≤ c ≥ 1 − 8Mte

.

Let ϵ = η|V|TO(1), then η = |V|ϵ

T O(1), then:

−O(1)ϵ2(Mt−1) |V|2T +2

P |pˆ(α,powJK)(xt|q,x0:t−1) − p(αpow)(xt|q,x0:t−1)| ≤ ϵ ≥ 1 − 8Mte

. We need the confidence bound to be at least 1 − δ for a given δ, hence:

−O(1)ϵ2(Mt−1) |V|2T +2

−O(1)ϵ2(Mt−1) |V|2T +2

−O(1)ϵ2(Mt−1) |V|2T +2

δ 8|V|T

1 − 8Mte

≥ 1 − δ =⇒ 8Mte

≤ δ =⇒ e

≤

≥ log |V|T δ

O(1)ϵ2(Mt − 1) |V|2T+2

1 ϵ2

=⇒ Mt ≥ O

=⇒

log

This finishes the proof of the claim.

1 δ

.

| |
|---|

### B. Batch Tokens Extension

In Algorithm 1, we provide a batched version of the algorithm presented in the main paper. Next, we provide a detailed description of the main steps in this algorithm and further discuss its advantages. To proceed in batches of size B over trajectories with length L, we first identify the total number of iterations in the main for loop as ⌊BT ⌋, and the remaining part of the trajectory is of size q = T mod B. Next, in each iteration t, based on the trajectory generated so far x0:(t−1)B−11, we aim to sample the next chunk x(t−1)B:tB−1 of B tokens from our approximation of the power distribution. To achieve this, we first generate a collection of Lt candidates for the next chunk x(t−1)B:tB−1. We call this collection Lt (see line 4) and its elements are sampled from the base model, i.e., x[(rt−] 1)B:tB−1 ∼ p(·|q,x0:(t−1)B−1) for r = 1,...Lt. Next, from the collection Lt we pick Kt, the most promising candidates, based on their likelihood provided by the base model. In line 5, these most promising candidates are aggregated in the set Gt(x0:(t−1)B−1). In lines 6-9 we start the first inner for loop that aims to compute the following scaling factors for each candidate chunk x′(t−1)B:tB−1 ∈ Gt(x0:(t−1)B−1):

Mt

1 Mt

pα−1(x[tBr]:tB+H

ζˆt(x′(t−1)B:tB−1) =

t−1|q,x0:(t−1)B−1,x′(t−1)B:tB−1) (11)

r=1

Mt

1 Mt − 1

ζˆt,(LOO−s )(x′(t−1)B:tB−1) =

pα−1(x[tBr]:tB+H

t−1|q,x0:(t−1)B−1,x′(t−1)B:tB−1), (12)

r=1,r̸=s

where each roll-out trajectory x[tBr]:tB+H

t−1 has length Ht tokens and is sampled from the base model (line 7). In lines 10-14 we start the second inner for loop that compute standard and leave-one-out probabilities:

pα(x′(t−1)B:tB−1|q,x0:(t−1)B−1)ζˆt(x′(t−1)B:tB−1)

pˆ(αpow)(x′(t−1)B:tB−1) =

. (13)

x(t−1)B:tB−1∈Gt(x0:(t−1)B−1) pα(x(t−1)B:tB−1|q,x0:(t−1)B−1)ζˆt(x(t−1)B:tB−1)

1When t = 1 we have (t − 1)B − 1 = −1 and we use the notation x0:−1 = {}

Algorithm 2 Batch Version of Scalable Power Sampling for a Query q

- 1: Inputs: Power distribution exponent α ∈ [1,∞), upper bounds on trajectory length T, size of a batch B, size of groups

{Kt}⌊

T B ⌋+1

t=1 , size of top evaluation budgets {Lt}⌊

T B ⌋+1

t=1 , horizon lengths {Ht}⌊

T B ⌋

t=1 , and the total Monte Carlo budgets {Mt}⌊

T B ⌋

t=1

- 2: Let q = T mod B and w = ⌊BT ⌋ + 1

- 3: for t = 1 → ⌊BT ⌋ do

- 4: Sample Lt blocks of size B: Lt = {{x[(rt−] 1)B:tB−1}L

t

r=1 such that x[(rt−] 1)B:tB−1 ∼ p(·|q,x0:(t−1)B−1)}

- 5: Compute Gt(x0:(t−1)B−1) = Top@Kt in Lt with respect to p(·|q,x0:(t−1)B−1)
- 6: for x′(t−1)B:tB−1 ∈ Gt(x0:(t−1)B−1) do
- 7: Sample Mt i.i.d. completions of length Ht:

{{x[tBr]:tB+H

t−1}M

t

r=1 such that x[tBr]:tB+H

t−1 ∼ p(·|q,x0:(t−1)B−1,x′(t−1)B:tB−1)}

- 8: Compute ζˆt(x′(t−1)B:tB−1) and ζˆt,(LOO−s )(x′(t−1)B:tB−1) from Eq. 11 and 12
- 9: end for
- 10: for x′(t−1)B:tB−1 ∈ Gt(x0:(t−1)B−1) do
- 11: for s ∈ {1,...,Mt} do
- 12: Compute pˆ(αpow)(x′(t−1)B:tB−1) and pˆ(α,pow−s)(x′(t−1)B:tB−1) using Equations 13 and 14
- 13: end for
- 14: end for
- 15: Compute the Jackknife estimator using Equation 15
- 16: Sample x(t−1)B:tB−1 ∈ Gt(x0:(t−1)B−1) w.p. pˆ(α,powJK)(x(t−1)B:tB−1|q,x0:(t−1)B−1)
- 17: Concatenate the sampled tokens x0:tB−1 = x0:(t−1)B−1 ⊕ x(t−1)B:tB−1
- 18: end for
- 19: Sample Lw blocks of size q + 1: Lw = {{x[⌊r]

T

B⌋B:T}L

w

r=1 such that x[⌊r]

T

B⌋B:T ∼ p(·|q,x0:B⌊T

B⌋−1)}

- 20: Compute Gw(x0:B⌊T

B⌋−1) = Top@Kw in Lw with respect to p(·|q,x0:B⌊T

B⌋−1)

- 21: Sample x⌊T

B⌋B:T ∈ Gw(x0:B⌊T

B⌋−1) w. p. pˆ(αlow.temp)(x⌊T

B⌋B:T|q,x0:B⌊T

B⌋−1) given in Equation 16

- 22: Concatenate the sampled token x0:T = x0:B⌊T

B⌋−1 ⊕ x⌊T

B ⌋B:T

- 23: Return Sampled trajectory {x0,...,xT} = x0:T

pα(x′(t−1)B:tB−1|q,x0:(t−1)B−1)ζˆt,(LOO−s )(x′(t−1)B:tB−1)

pˆ(α,pow−s)(x′(t−1)B:tB−1) =

. (14)

x(t−1)B:tB−1∈Gt(x0:(t−1)B−1) pα(x(t−1)B:tB−1|q,x0:(t−1)B−1)ζˆt,(LOO−s )(x(t−1)B:tB−1)

for all s ∈ {1,...,Mt}. Being equipped with these probabilities, for each candidate block x′(t−1)B:tB−1 ∈ Gt(x0:(t−1)B−1), we compute jackknife likelihoods in line 15 using the following equation:

Mt − 1 Mt

pˆ(α,powJK)(x′(t−1)B:tB−1|q,x0:(t−1)B−1) = Mtpˆ(αpow)(x′(t−1)B:tB−1) −

Mt

pˆ(α,pow−s)(x′(t−1)B:tB−1). (15)

s=1

In lines 16-17, we sample the next block in the trajectory using the computed jackknife probability distribution and concatenate this block to the already constructed prefix. Finally, in lines 19-20 we construct the last part of the trajectory of

length q + 1 = T − B⌊BT ⌋ + 1 by generating a collection of Lw candidate completions x[⌊r]

B⌋−1)} (see line 19). In line 20, we select Kw, the most promising completions from group Lw, with respect to their likelihood values. Then, we sample one of these promising candidates, using a batch version of low temperature distribution (line 21):

B⌋B:T ∼ p(·|q,x0:B⌊T

T

pˆ(αlow.temp)(x⌊T

B⌋B:T|q,x0:B⌊T

B⌋−1) =

pα(x⌊T

B⌋B:T|q,x0:B⌊T

B⌋−1) x ˆ⌊ T

. (16)

B⌋−1) pα(xˆ⌊T

B⌋B:T|q,x0:B⌊T

B⌋−1)

B ⌋B:T ∈Gw(x0:B⌊ T

The selected completion is concatenated to the constructed trajectory (line 22) and forms the output of the algorithm.

Task Qwen Prompt DeepSeek Prompt MATH500

Please reason step by step, and put your final answer within \\boxed{}

Can you solve the following math problem? Please reason step by step, and put your final answer within \\boxed{}

###### HumanEval

Write a Python function to solve the following problem:

Complete the following Python function:

###### GPQA

Answer the following multiple-choice question. The last line of your response should be of the following format: ’\\boxed{LETTER}’ (without quotes) where LETTER is one of ABCD (ex. ’\\boxed{A}’). Think step by step before answering. {Question} A) {} B) {} C) {} D) {}

Answer the following multiple-choice question. The last line of your response should be of the following format: ’\\boxed{LETTER}’ (without quotes) where LETTER is one of ABCD (ex. ’\\boxed{A}’). Think step by step before answering. {Question} A) {} B) {} C) {} D) {}

Table 3. Prompt templates used for baselines and our method. Adapted from the MCMC power-sampling repository (Karan & Du, 2025) and DeepSeek official instruction format.

Please notice, in Algorithm 2 each generation of future rollouts (lines 4,7 and 19) can be performed in a parallel fashion, thus reducing further the overall processing time. The overall inference complexity of the above algorithm is given by

⌊ BT ⌋ t=1 (Lt + KtMt) + Lw. We can also derive such bounds for the original algorithm in the main paper in terms of token

T B ⌋

generation complexity. This amounts to: O ⌊

t=1 [LtB + KtMtHt] + LwB .

### C. Further Experimental Results

#### C.1. Experimental Setups and Implementation Details

We implement our method following Algorithm B and will release our code upon acceptance. All experiments are run on the same single GPU using vLLM v0.6.3. Since our approach is purely a sampling procedure (no backpropagation). Therefore, compared to post-training approaches, the method is comparatively lightweight in memory and can be run on small-memory devices (depending primarily on the chosen model size).

Prompts. Table 3 lists the prompt templates used for each benchmark. For the Qwen models, we adopt the prompt templates used in the MCMC power-sampling repository to ensure fairness. For the DeepSeek models, we follow the official DeepSeek instruction format, keeping the task content identical while changing only the wrapper (instruction formatting). All the methods reported in the main evaluation section use these prompts.

Hyperparameter settings. We summarise the key hyperparameters used for each sampling method in Table 4. Across all methods, we use a maximum generation length of Tmax = 3072 tokens and terminate early upon emitting an EOS token. Unless otherwise stated, we set the random seed to 0 for all main results. For evaluation, we use the same scoring functions as the open-source MCMC power-sampling implementation (Karan & Du, 2025) to ensure a fair comparison across decoding strategies.

For Low-temperature sampling, we apply token-level temperature scaling with temperature 1/α and set α = 4. For the verifier-free version of Best-of-N, we generate NBoN = 32 independent completions and select the candidate with the highest base-model sequence score, ranked by log p(x | q) (using the same normalisation choice across benchmarks).

For the MCMC sampling method, we follow the open-source implementation of (Karan & Du, 2025) with α = 4, Tmax = 3072, block size B = 192, NMCMC = 10 MCMC steps per stage, and proposal model pprop given by the base model sampled at temperature 1/α = 0.25.

For the GRPO baseline, we use the GRPO checkpoints from (Shao et al., 2025; 2024). Training hyperparameters are

###### Method Hyperparameters

Low-temperature temperature = 0.25 = 1/α (with α = 4) Best-of-N NBoN = 32; selection score log p(x | q) MCMC power sampling α = 4, B = 192, NMCMC = 10, pprop: base temperature 1/α Power sampling (Ours) α = 4, B = 192, Kt = 8, Mt = 8

Table 4. Main hyperparameter settings for each method.

described in the corresponding papers, and the released model weights are available at 2 and 3. These checkpoints correspond to post-training on the MATH training split for Qwen2.5-Math-7B and DeepSeek-Math-7B, respectively.

Our autoregressive approximation uses the same global settings as MCMC power sampling (α = 4, Tmax = 3072, B = 192) and applies token-level lookahead with Top-Kt candidates and Mt rollouts per candidate; we set Kt = Mt = 8 and use the jackknife-corrected estimator pˆ(pow)α,JK throughout. All likelihood computations are performed in log-space for numerical stability.

Inference-time measurement and practical scaling. All reported inference times are measured on a single GPU under the same runtime stack. We report wall-clock generation time per prompt (excluding dataset loading and metric computation), averaged over the full evaluation set with the same batching configuration across methods, and fix the random seed to ensure comparable decoding behaviour. In our current configuration, our method is typically about 2.5–3.5× slower than standard decoding due to additional rollouts, but remains up to 10× faster than MCMC power sampling, which requires iterative refinement steps. The overhead of our method scales primarily with the rollout budget and candidate set size (Kt,Mt) and the maximum generation length Tmax, and can be reduced by tuning these budgets with limited accuracy loss (see ablation study). Compared with RL post-training methods, our method requires no training compute or external rewards, while post-training can improve pass@1, it typically requires many generations per training example over large training sets, whereas sampling-based sharpening can be deployed immediately and is particularly attractive in small-to-moderate deployment regimes.

#### C.2. More Pass@k Results and Analysis

Pass@K on out-of-domain tasks. Figure 5 reports pass@k on HumanEval and GPQA for Qwen2.5-Math-7B and its GRPO (MATH) post-trained variant. Although both tasks are out-of-domain with respect to GRPO training (which uses the MATH training split), we observe qualitatively different behaviours across domains. On HumanEval, GRPO retains comparable pass@k scaling to the base model, suggesting no strong diversity collapse in this setting. In contrast, on GPQA the GRPO curve shows similar behaviour as the in-domain MATH500 task (in Figure 3): GRPO boosts pass@1 but provides limited additional gains for larger k, consistent with diversity reduction, suggesting that this kind of collapse also persists even in the out-of-domain task of GRPO. This behaviour is broadly consistent with recent findings that reward-based post-training can improve pass@1 while reducing diversity at larger k (Yue et al., 2025), and we leave a systematic multi-model study of this effect on different post-training models to future work.

In comparison, power sampling methods improve pass@1 without inducing the same degradation in pass@k scaling. Both MCMC power sampling and our method achieve strong pass@k growth. This supports the main message of the paper: inference-time distribution sharpening can increase the probability of sampling high-quality trajectories while preserving diversity, without requiring training or external verifiers.

Qualitative examples. We include two representative failure cases for GRPO under the pass@k setting, one in-domain example from MATH500 and one out-of-domain example from GPQA, that remain incorrect at pass@16 (as shown in Tables 5 and 6, we abbreviate long generations with “...” for readability). We observe that these failures are often driven less by a lack of capability than by reduced diversity: across 16 samples in practice, the GRPO model frequently repeats near-identical reasoning and the same final answer on many problems, so increasing k provides limited additional coverage. This behaviour is not observed in non-RL models evaluated under identical prompts and sampling parameters, including

- 2https://huggingface.co/stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_

7b-lr5e-7-kl0.00-step150

- 3https://huggingface.co/deepseek-ai/deepseek-math-7b-rl

a) HumanEval

b) GPQA

95

95

85

85

75

75

Accuracy(%)

Accuracy(%)

65

65

55

55

45

45

35

35

25

25

1 2 4 8 16

1 2 4 8 16

k

k

Base

GRPO

MCMC

Ours

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 5. Pass@k performance k ∈ {1, 2, 4, 8, 16} on out-of-domain tasks ((a) HumanEval and (b) GPQA). On HumanEval, GRPO exhibits similar scaling with k to the base model (no obvious diversity collapse). On GPQA, GRPO improves pass@1 but scales more weakly with k, indicating reduced diversity even on an out-of-domain benchmark. In contrast, power sampling (MCMC and ours) improves pass@1 while preserving strong pass@k scaling.

Qwen2.5-Math-7B and DeepSeek-Math-7B, which are also trained on math-related data and typically produce a broader set of distinct candidates, thereby benefiting more from pass@k.

The GPQA example further illustrates a common out-of-domain failure mode: in the QA task, the GRPO-MATH model can produce largely correct reasoning yet still answer incorrectly by selecting the wrong option letter, or by terminating after intermediate calculations without outputting a final multiple-choice selection. This failure mode is less frequent in the corresponding base model, suggesting it may be a side effect of GRPO post-training on the MATH training split for this checkpoint. In contrast, our power sampling recovers correct trajectories by exploring alternative solution paths while preserving stochastic diversity.

#### C.3. Ablation Study on Different Models

We conduct additional ablations to assess sensitivity to hyperparameters on Qwen2.5-7B and DeepSeek-Math-7B in math, code, QA tasks separately on a validation subset. Figure 6 shows consistent trends across models and tasks, which align with our findings in Qwen2.5-Math-7B (in Figure 4). First, performance is robust to the rollout and candidate budgets: under the same sharpening exponent α, varying Mt = Kt over the tested range only mildly affects pass@1, indicating that most gains can be achieved with relatively small sampling budgets. Second, for all the three base models, the strongest region is typically in a moderate range (α ≈ 4–5), while overly aggressive sharpening (e.g., α = 8) can degrade performance, and α = 1 performs poorly on most tasks. A notable exception appears on GPQA for DeepSeek-Math-7B, α = 1 can be competitive in parts of the budget range, which also aligns with the empirical observation that low-temperature decoding can be particularly strong on GPQA for this model. Overall, these ablations suggest that power sampling can “unlock” improved pass@1 across diverse base models without post-training using a small and stable hyperparameter range, and that the optimal α reflects a task-dependent balance between sharpening and preserving exploratory diversity.

MATH

HumanEval

GPQA

0.75

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

4 6 8

4 6 8

4 6 8

0.70

0.34

0.70

0.68

=tt

0.32

0.65

0.66

0.60

10 16

10 16

10 16

0.30

0.64

0.55

0.28

0.62

1 2 4 5 8

1 2 4 5 8

1 2 4 5 8

(a) Qwen2.5-7B

MATH

HumanEval

GPQA

0.46

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

0.36

4 6 8

4 6 8

4 6 8

0.48

0.44

0.35

0.47

=tt

0.42

0.46

0.34

0.45

0.40

0.33

10 16

10 16

10 16

0.44

0.32

0.38

0.43

1 2 4 5 8

1 2 4 5 8

1 2 4 5 8

(b) DeepSeek-Math-7B

- Figure 6. Pass@1 performance for MATH, HumanEval, and GPQA with α ∈ {1, 2, 4, 5, 8} (x-axis) and Mt = Kt ∈ {4, 6, 8, 10, 16} (y-axis) in Qwen2.5-7B (top) and DeepSeek-Math-7B (bottom).

###### Example 2 (Task 82, MATH500, Qwen2.5-Math-7B).

Question. Point A lies somewhere within or on the square with opposite corners (0, 0) and (2, 2). Point B lies somewhere within or on the square with opposite corners (4, 2) and (5, 3). What is the greatest possible value of the slope of the line containing points A and B? Express your answer as a common fraction.

###### GRPO (MATH).

To determine the greatest possible value of the slope of the line containing points A and B, we need to consider the coordinates of the points A and B within their respective squares. The slope of a line passing through points (x1, y1) and (x2, y2) is given by y2−y1 x2−x1 . To maximize this slope, we need to maximize the difference in the y-coordinates and minimize the difference in the x-coordinates. ... The coordinates of point A can range from (0, 0) to (2, 2), and the coordinates of point B can range from (4, 2) to (5, 3). ...This means we should consider the points (0, 0) for A and (4, 3) for B, as these points will give the maximum possible slope. ... Let’s calculate the slope for these points:

- 3 − 0

- 4 − 0

- 3

- 4

slope =

=

.

Let’s verify this using Python code to ensure accuracy. ...output 3.0 Therefore, the greatest possible value of the slope of the line containing points A and B is:

|3<br><br>4<br>|
|---|

.

###### Ours.

To determine the greatest possible value of the slope of the line containing points A and B, we need to consider the coordinates of the points A and B within their respective squares. The slope m of the line passing through points A(x1, y1) and B(x2, y2) is given by:

y2 − y1 x2 − x1

m =

.

The coordinates of point A can range from (0, 0) to (2, 2), and the coordinates of point B can range from (4, 2) to (5, 3). To maximize the slope, we need to maximize the numerator y2 − y1 and minimize the denominator x2 − x1. ... The maximum value of y2 − y1 occurs when y2 = 3 and y1 = 0, giving y2 − y1 = 3. The minimum value of x2 − x1 occurs when x2 = 4 and x1 = 2, giving x2 − x1 = 2. Therefore, the greatest possible value of the slope is:

3 2

m =

.

Let’s confirm this with Python code ...output 1.5 Thus, the final answer is:

|3 2<br><br>|
|---|

.

|3 2<br><br>|
|---|

Ground truth.

- Table 5. Qualitative example illustrating pass@k behaviour. We show one sampled completion from GRPO (MATH) and our method on the same prompt.

###### Example 2 (Task 19, GPQA-Diamond, Qwen2.5-Math-7B).

Question. A common approximation made in many-body nuclear calculations of some interaction (for example an electroweak interaction) is to suppose that nucleons in a nucleus behave in the same way as free nucleons. Which of the following four assumptions jointly imply this “impulse” approximation?

- 1. The interaction current only interacts with individual nucleons.
- 2. The nucleus is transparent apart from the selected nucleon.
- 3. The quarks internal to the selected nucleon are non-relativistic.
- 4. The interaction proceeds as if the selected nucleon experiences no binding forces. A) 2,3,4 B) 1,2,4 C) 1,2,3 D) 1,3,4 GRPO (MATH).

To answer this question, we need to understand the conditions under which the impulse approximation holds. The impulse approximation is a simplification used in many-body nuclear calculations where the interaction current only interacts with individual nucleons, the nucleus is transparent apart from the selected nucleon, ...and the interaction proceeds as if the selected nucleon experiences no binding forces. Let’s break down the conditions: 1. The interaction current only interacts with individual nucleons.

- 2. The nucleus is transparent apart from the selected nucleon.
- 3. The quarks internal to the selected nucleon are non-relativistic.
- 4. The interaction proceeds as if the selected nucleon experiences no binding forces. The impulse approximation is valid when these conditions are met. Therefore, the correct answer is the set of conditions that includes all of these. ... Let’s check the options: A) 2,3,4 B) 1,3,4 C) 1,2,4 D) 1,2,3 The correct answer is C) 1,2,4 because it includes all the conditions required for the impulse approximation. ...

|C|
|---|

.

Ours. To determine which assumptions jointly imply the “impulse” approximation, let’s analyze each assumption step by step:

- 1. The interaction current only interacts with individual nucleons. ...This is a key aspect of the impulse approximation. ...
- 2. The nucleus is transparent apart from the selected nucleon. ...This is another important aspect of the impulse approximation. ...
- 3. The quarks internal to the selected nucleon are non-relativistic. ...not directly related to the impulse approximation. ...
- 4. The interaction proceeds as if the selected nucleon experiences no binding forces. ...another key aspect of the impulse approximation. ... Based on this analysis, the assumptions that jointly imply the impulse approximation are 1, 2, and 4. Therefore, the correct answer is:

|B|
|---|

.

|B|
|---|

Ground truth.

- Table 6. Qualitative example on GPQA illustrating pass@k behaviour. We show one sampled completion from GRPO (MATH) and our method on the same prompt.

