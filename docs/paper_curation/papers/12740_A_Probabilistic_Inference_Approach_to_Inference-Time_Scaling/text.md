# arXiv:2502.01618v5[cs.LG]14Aug2025

## Rollout Roulette: A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods

Isha Puri1 Shiv Sudalairaj2 GX Xu2 Kai Xu2 Akash Srivastava2

1MIT CSAIL 2RedHat AI Innovation

### Abstract

Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating a pivot to scaling test-time compute. Existing deterministic inference-time scaling methods, usually with reward models, cast the task as a search problem, but suffer from a key limitation: early pruning. Due to inherently imperfect reward models, promising trajectories may be discarded prematurely, leading to suboptimal performance. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods. Our method maintains a diverse set of candidates and robustly balances exploration and exploitation. Our empirical evaluation demonstrates that our particle filtering methods have a 4–16x better scaling rate over deterministic search counterparts on both various challenging mathematical and more general reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct surpasses GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code and further information is available at https://probabilistic-inference-scaling.github.io/

### 1 Introduction

Large language models (LLMs) continue to improve through larger architectures and massive training corpora [13, 22]. In parallel, inference-time scaling (ITS)—allocating more computation at inference time—has emerged as a complementary approach to improve performance without increasing model size. Recent work has shown that ITS enables smaller models to match or exceed the accuracy of larger ones on complex reasoning tasks [3], with proprietary systems like OpenAI’s o1 and o3 explicitly incorporating multi-trial inference for better answers [21].

Beyond answer-level scaling methods like self-consistency and best-of-n sampling [4], a popular class of ITS methods formulates inference as search guided by a process reward model (PRM), which scores partial sequences step-by-step. This perspective has led to the adoption of algorithms like beam search [34] and Monte Carlo tree search [11]. These methods refine LLM outputs by prioritizing trajectories that score highly according to the PRM. However, they also inherit a major limitation from classical search: early pruning. Because the PRM is an imperfect approximation of true correctness, these methods often eliminate promising candidates too early-based on noisy or miscalibrated partial scores. This failure mode is illustrated in Figure 1. The PRM assigns a higher initial score to Answer 2, which ultimately turns out to be incorrect, and a lower score to the correct

Preprint. Under review.

|[Figure 1]<br><br>[Figure 2]<br><br>Jane has twice as many pencils as Mark.<br><br>Step 1:<br><br>Total parts = 2 (Jane) + 1 (Mark) = 3<br><br>|Score: 0.5156|
|---|
<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Step 2: Each part = 9 / 3 = 3 pencils<br><br>Score: 0.9375<br><br>Step 3: Jane has 2 parts -> 2×3= 6 Final Answer: 6 pencils<br><br>Score: 0.9883<br><br>Step 1:<br><br>Total parts = 3 -> each part = 9/3 = 3<br><br><br>|Score: 0.9453|
|---|
<br><br>Step 2: Jane gets 1 part -> 3 Final Answer: 3 pencils<br><br>Score: 0.0133<br><br>Together they have 9 pencils. How many pencils does Jane have?|
|---|

- Figure 1: A true example of PRM assigning a lower score to the first step of a solution that turns out to be correct. In deterministic scaling methods, this solution would have been discarded in favor for one that had a higher initial PRM score but turned out to be incorrect.

Answer 1. A deterministic method like beam search would discard Answer 1 after the first step, never recovering it—even though it is the correct solution. Such brittleness is inherent to greedy search: once a path is pruned, it cannot be revisited.

To address this limitation, we instead maintain a distribution over the possible paths during generation to represent the uncertainty we would like to account for due to the imperfection in reward modeling and propagate it through sampling. We realize this approach in a probabilistic inference framework for inference-time scaling, in which we use particle filtering to sample from the posterior distribution over accepted trajectories. Our method maintains a weighted population of candidate sequences that evolves over time, balancing exploitation of high-PRM paths with stochastic exploration of alternatives. Unlike beam search, which targets the mode of an approximate distribution, particle filtering samples from the typical set, making it inherently more robust to noise and multi-modality in the reward landscape. High-scoring candidates are favored but not allowed to dominate, ensuring that low-probability (but potentially correct) paths remain in play.

We demonstrate that this simple shift-from deterministic search to sampling with uncertaintyproduces strong empirical gains. On mathematical and reasoning tasks, our method significantly outperforms existing ITS approaches across multiple model families. For instance, using Qwen2.5Math-1.5B-Instruct and a compute budget of 4, our method surpasses GPT-4o; with a budget of 32, the 7B model surpasses o1 accuracy. Our key contributions are as follows.

- 1. We propose a particle filtering algorithm for inference-time scaling that mitigates early pruning by maintaining exploration across trajectories with uncertainty. We formulate ITS as posterior inference over a state space model defined by an LLM (transition model) and PRM (emission model), enabling principled application of probabilistic inference tools.
- 2. We present strong results on mathematical and out-of-domain reasoning tasks and study Particle Filtering’s scaling performance.
- 3. We ablate compute allocation strategies (parallel vs. iterative), PRM aggregation methods, and generation temperature, proposing a new model-based reward aggregation method that improves stability and performance.
- 4. We demonstrate that our proposed methods have 4–16x faster scaling speed than previous methods based on a search formulation on the MATH500 and AIME 2024 datasets, with small language models in the Llama and Qwen families. We show that PF can scale Qwen2.5-Math-1.5B-Instruct to surpasses GPT-4o accuracy with only a budget of 4 and scale Qwen2.5-Math-7B-Instruct to o1 accuracy with a budget of 32.

### 2 Background

State space models describe sequential systems that evolve stepwise, typically over time [25]. They consist of a sequence of latent states {xt}Tt=1 and corresponding observations {ot}Tt=1 The evolution of states is governed by a transition model p(xt | x<t−1), and the observations are

governed by the emission model p(ot|xt). The joint distribution of states and observations is given by: p(x1:T,o1:T) = p(x1) Tt=2p(xt | x<t−1) Tt=1p(ot | xt), where p(x1) is the prior.

Probabilistic inference in SSMs involves estimating the posterior distribution of the hidden states given the observations, p(x1:T|o1:T) [25], which is generally intractable due to the high dimensionality of the state space and the dependencies in the model. Common approaches approximate the posterior through sampling-based methods or variational approaches [19]. Particle filtering (PF) is a sequential Monte Carlo method to approximate the posterior distribution in SSMs [24, 7]. PF

represents the posterior using a set of N weighted particles {x(ti),wt(i)}Ni=1, where xt(i) denotes the ith particle at time t, and wt(i) is its associated weight. The algorithm iteratively propagates particles using the transition model and updates weights based on the emission model: wt(i) ∝ wt(−i)1p(ot | x(ti)).

### 3 Method

generate a 1st step / score

generate a next step / score

generate a next step / score

Select particle with highest reward as ﬁnal answer

Resampling

Resampling

[Figure 5]

Step 3: Mark has 1 part = 3 pencils. Jane has other 6. Final Answer: 6 Pencils

Step 2: Each part = 9 / 3 = 3 pencils

Step 1:

- Total parts = 2

(Jane)+1(Mark)=3

Score: 0.3198

Step 1:

- Total parts = 3 -> each part = 9/3

Particle 1

- Step 1: Total parts = 2 (Jane) + 1 (Mark) = 3
- Step 2: Each part = 9 / 3 = 3 pencils
- Step 3: Jane has 2 parts -> 2×3= 6 Final Answer: 6 pencils

Score: 0.9375

Score: 0.8133

[Figure 6]

Jane has twice as many pencils as Mark. Together

this particle has completed its answer

Step 2: Jane gets 1 part ->3 Final Answer: 3 pencils Score: 0.0133

Particle 2

they have 9 pencils. How many pencils does Jane have?

= 3

Score: 0.9453

this particle has completed its answer

Step 3: Jane has 2 parts -> 2×3= 6 Final Answer: 6 pencils

Step 1: Let J = x, M = 2x

Particle 3

Step 2: Mark = 1 part

Score: 0.5898

Score: 0.0392

Score: 0.9883

this particle has completed its answer

- Figure 2: Inference-time scaling with particle filtering: initialize n particles, generate a step for each, score with the PRM, resample via softmax-weighted scores, and repeat until full solutions are formed.

We formulate inference-time scaling for a language model pM as approximate posterior inference in a state space model (SSM) defined over token sequences. Let x1:T denote the sequence of generated tokens (or chunks, such as steps in a math problem), and let o1:T denote binary observations indicating whether the steps so far are accepted, given a prompt c.

The forward model defines the joint distribution over latent states and observations as:

pM(x1:T,o1:T | c) ∝

T

pM(xt | c,x<t−1)

t=1

T

p(ot | c,x<t), (1)

t=1

where the transition model pM(xt | c,x<t−1) is given by the language model M, and the emission model p(ot | c,x<t) is a Bernoulli distribution: p(ot | c,x<t) = B(ot;r(c,xt)), with reward function r(c,xt) encoding the acceptability of the step xt. Figure 3 illustrates this SSM.

Given this model, our goal is to infer the posterior distribution over latent trajectories that yield fully accepted sequences, i.e., pM(x1:T | c,o1:T = 1). This formulation makes particle filtering a natural and theoretically grounded choice for inference.

In practice, the true reward function r is often unavailable. Following prior work, we approximate it using a pre-trained preference or reward model (PRM) rˆ suited to the target domain (e.g., math reasoning). This results in an approximate emission model: pˆ(ot | c,x<t) = B(ot;rˆ(c,x<t)). Substituting this into the forward model, we obtain the approximate joint distribution:

pˆM(x1 : T,o1:T | c) ∝

T

T

pM(xt | c,x<t−1)

t=1

t=1

pˆ(ot | c,x<t), (2)

c

language model

x1 x2 ··· xT

pM (x1 | c) pM (x2 | c, x1) pM (xT | c, x<T )

reward model

o1 o2 ··· oT

pˆ(o1 | c, x1) pˆ(o2 | c, x2) pˆ(ot | c, x<t)

- Figure 3: State-space model for inference-time scaling. c is a prompt, x1,...,xT are LLM outputs, and o1,...,oT are “observed” acceptances from a reward model. We estimate the latent states conditioned on ot = 1 for all t. and correspondingly aim to estimate the posterior pˆM(x1 : T | c,o1:T = 1).

#### 3.1 Particle Filtering for Estimating the Posterior

We now apply particle filtering (PF) to approximate the posterior over accepted sequences x1:T under the model in (2). Each particle represents a partial trajectory, and inference proceeds by iteratively generating, scoring, and resampling these particles based on their reward-induced likelihood. At each time step t, PF maintains a population of N weighted particles. The algorithm proceeds as follows:

- • Initialization (t = 1): Each particle is initialized by sampling the first token from the LLM: x(1i) ∼ pM(x1 | c).
- • Propagation (t > 1): Each particle is extended by sampling the next token from the LLM conditioned on its history: xt(i) ∼ pM(xt | c,x(<ti)−1).
- • Weight Update: Using a reward model rˆ, each particle is assigned an unnormalized weight that reflects the likelihood of acceptance: wt(i) ∝ wt(−i)1 · rˆ(c,x(<ti)).
- • Resampling: To focus computation on promising trajectories, we sample a new population of particles from the current set using a softmax distribution over weights: Pt(j = i) =

′) t ).

exp(wt(i))/ i′ = 1N exp(w(i

The next generation of particles is formed by sampling indices jt(1),...,jt(N) ∼ Pt and retaining the corresponding sequences.

This procedure balances exploitation of high-reward partial generations with stochastic exploration, increasing the chances of recovering correct completions even when early steps are uncertain.

The algorithm can be implemented efficiently: both token sampling and reward computation can be parallelized across particles. With prefix caching, the total runtime is comparable to generating N full completions independently.

Final Answer Selection Particle filtering yields a weighted set of samples approximating the posterior, enabling principled answer selection strategies. While selecting the highest-weighted particle or using (weighted) majority voting better reflects the typical set, we adopt the standard practice of scoring all final outputs with an outcome reward model (ORM) and selecting the top-scoring one for fair comparison with prior work. Additionally, the posterior samples allow richer evaluation—for instance, estimating the expected correctness of the model under its own distribution, rather than relying solely on point estimates. Notably, samples from Algorithm 1 can be used to construct unbiased estimates of any expectation over (2). In the context of math and reasoning, it guarantees the expected accuracy is unbiased, which we formulate in Theorem 2 (proof in Appendix C).

Theorem 1 (Unbiasedness of Expected Accuracy). Let {(w(i),x(i))} be weighted particles from Algorithm 1 and is_correct(x) is a function to check the correctness of response x. We have

##### E

i

w(i) is_correct(x(i)) =

x

p ˆM(x1:T | c,o1:T = 1) is_correct(x(i)) , (3)

where the expectation is over the randomness of the algorithm itself.

Reward Aggregation with PRMs To compute particle weights during generation, we aggregate per-step scores from the process reward model (PRM) rˆ. Our default uses a product of step-level rewards to align with the factorized likelihood structure, but alternative aggregation strategies (e.g., min, last-step, or model-based) may offer different trade-offs. We describe and compare these strategies in detail in Appendix A and report ablation results in 4.5.

Sampling v.s. deterministic search An alternative to our sampling-based approach is to treat inference-time scaling as an optimization problem under the approximate posterior (2), reducing to deterministic search methods like beam search or MCTS. However, these methods assume the reward model rˆ is accurate at every step and prune aggressively based on early scores. In practice, PRMs are noisy and their preferences often shift during unrolling. As a result, deterministic search can irreversibly discard trajectories that may have low early scores but high posterior mass overall. Once pruned, such paths cannot be recovered.

In contrast, particle filtering maintains a distribution over partial sequences and uses stochastic resampling to balance exploration and exploitation. This allows recovery from early errors and robustness to reward noise. While beam search targets the mode of the approximate posterior-making it sensitive to local errors-PF samples from the typical set, smoothing over inconsistencies in rˆ. Unlike search heuristics, PF provides consistent, unbiased estimators under mild assumptions, and naturally captures multi-modal solutions—critical when multiple valid completions exist. We validate these advantages empirically in Section 4.5.

Multiple iterations and parallel chains The PF approach to inference-time scaling can be used to define a MCMC kernel that enables two new types of scaling: multiple iterations of complete answers inspired by PG and parallel simulations inspired by parallel tempering. We detail the methodology and results for both extensions in sections B.1 and B.2 in the appendix.

### 4 Evaluation

We evaluate our proposed methods in this section. We detail our experimental setup in Section 4.1 and start with highlighted results comparing against closed-source models and competitive inference-time scaling methods with open-source models (Section 4.2). We then study how the main algorithm, particle filtering, scales with more computation and compare it with competitors (Section 4.4). We further perform an extensive ablation study on key algorithmic choices like reward models, reward aggregation, final answer aggregation, and LLM temperatures (Section 4.5). Finally, we study different allocations of the compute budget through iterative and parallel extensions (Section 4.5).

#### 4.1 Setup

Models We consider two types of open-source small language models (SLMs) as our policy models for generating solutions. The first is general models, for which we evaluate Qwen2.5-1.5B-Instruct [29], Qwen2.5-7B-Instruct, Llama-3.2-1B-Instruct, and Llama-3.1-8B-Instruct [10]. The second is math models, using Qwen2.5-Math-1.5B-Instruct and Qwen2.5-Math-7B-Instruct. These small models are well-suited for inference-time scaling, enabling efficient search of multiple trajectories.

Process Reward Models To guide our policy models, we utilized Qwen2.5-Math-PRM-7B [31], a 7B process reward model. We selected this model for its superior performance over other PRMs we tested, including Math-Shepherd-mistral-7b-prm [26], Llama3.1-8B-PRM-Deepseek-Data [28], and EurusPRM-Stage2 [30]. This result as an ablation study is provided in Section 4.5, where we also study the different ways to aggregate step-level rewards from PRMs discussed in Section 3.1.

#### Baselines

- • Greedy: single greedy generation from the model, serving as the “bottom-line” performance.
- • Self Consistency [27]: simplest scaling method, majority voting across candidates
- • BoN/WBoN [4]: simple RM-based scaling method, scores outputs with outcome reward models
- • Beam Search [23]: structured search procedure that incrementally builds sequences by keeping the top-k highest-scoring partial completions at each generation step.

- • DVTS [3]: a parallel extension of beam search that improves the exploration and performance.

Datasets We evaluate our methods and baselines across a wide variety of tasks that span difficulty level and domain to test basic and advanced problem-solving and reasoning.

- • MATH500 [16]: 500 competition-level problems from various mathematical domains.
- • AIME 2024 [1]: 30 high difficulty problems from the American Invitational Mathematics Examination (AIME I and II) 2024.
- • NumGLUE Task 2 (Chemistry) [20]: 325 questions that test advanced reasoning across real-world tasks, mostly centered on the chemistry domain.
- • FinanceBench [12]: 150 open-book financial question answering tasks grounded in real-world financial documents and analysis.

Parsing and scoring Details on parsing and scoring functions across datasets in the Appendix H

#### 4.2 Results on Mathematical Reasoning Datasets

GPT-4o

GPT-4o

0.7

0.7

Llama-3.1-70B-Instruct

Llama-3.1-70B-Instruct

0.6

0.6

Accuracy

Accuracy

0.5

0.5

Llama-3.1-8B-Instruct

Llama-3.1-8B-Instruct

0.4

0.4

Weighted BoN (Llama-3.2-1B-Instruct)

Weighted BoN (Llama-3.1-8B-Instruct)

0.3

Ours-Particle Filtering (Llama-3.2-1B-Instruct)

Ours-Particle Filtering (Llama-3.1-8B-Instruct)

0.3

DVTS (Llama-3.2-1B-Instruct)

DVTS (Llama-3.1-8B-Instruct)

Llama-3.2-1B-Instruct

Llama-3.2-1B-Instruct

0-shot CoT (Greedy)

0-shot CoT (Greedy)

0.2

0.2

20 21 22 23 24 25 26 27

20 21 22 23 24 25 26 27

Budget (# of model generations)

Budget (# of model generations)

(a) Llama-3.2-1B-Instruct

(b) Llama-3.1-8B-Instruct

0.875

0.875

o1-preview

o1-preview

0.850

0.850

0.825

0.825

Qwen2.5-Math-7B-Instruct

Qwen2.5-Math-7B-Instruct

Accuracy

Accuracy

0.800

0.800

0.775

0.775

GPT-4o

GPT-4o

0.750

0.750

Ours-Particle Filtering (Qwen2.5-Math-1.5B-Instruct)

Ours-Particle Filtering (Qwen2.5-Math-7B-Instruct)

Weighted BoN (Qwen2.5-Math-1.5B-Instruct)

Weighted BoN (Qwen2.5-Math-7B-Instruct)

0.725

0.725

DVTS (Qwen2.5-Math-1.5B-Instruct)

DVTS (Qwen2.5-Math-7B-Instruct)

Qwen2.5-Math-1.5B-Instruct

Qwen2.5-Math-1.5B-Instruct

0-shot CoT (Greedy)

0-shot CoT (Greedy)

0.700

0.700

20 21 22 23 24 25 26 27

20 21 22 23 24 25 26 27

Budget (# of model generations)

Budget (# of model generations)

(c) Qwen2.5-Math-1.5B-Instruct

(d) Qwen2.5-Math-7B-Instruct

Figure 4: Accuracy vs. Generation Budget across models using different inference-time strategies.

We present our main results in Table 1, comparing Particle Filtering (PF) with a suite of strong inference-time scaling baselines on two challenging mathematical reasoning tasks: MATH500 and AIME 2024. All inference-time scaling methods are evaluated under a fixed compute budget of 32 generations per instance, using Qwen2.5-Math-PRM-7B as the reward model. Specifically, it serves as the PRM for PF, Beam Search, and DVTS, and as the ORM in WBoN.

- • PF consistently achieves the best performance across all model sizes. Among all inferencetime scaling methods, PF delivers the highest accuracy on both benchmarks, often outperforming alternatives by a significant margin.
- • PF unlocks competitive performance even for small models. For instance, Qwen2.5-1.5BInstruct, when scaled using PF, surpasses the much larger GPT-4o on both MATH500 and AIME 2024. This showcases the ability of inference-time compute to significantly improve performance without increasing model size.
- • PF on Qwen2.5-Math-7B-Instruct outperforms o1-preview on MATH500. Scaling Qwen2.5Math-7B with PF results in a new state-of-the-art among open models: 87.7% on MATH500

Model Method MATH500 AIME 2024 Closed Source LLMs GPT-4o – 76.2 4/30 o1-preview – 87.0 12/30 Claude 3.5 Sonnet – 78.2 5/30 Open Source LLMs

Llama-3.1 70B Instruct – 65.6 5/30 Qwen-2.5 Math 72B Instruct – 82.0 9/30 Open Source General SLMs

Qwen-2.5 1.5B Instruct Greedy 54.4 1/30 Self Consistency 61.0 2/30

BoN 67.8 1/30 WBoN 69.2 2/30

Beam Search 76.2 5/30

DVTS 76.6 4/30 Particle Filtering (Ours) 79.3 6/30

Open Source Math SLMs

|Qwen-2.5 Math 1.5B Instruct Greedy Self Consistency BoN WBoN Beam Search DVTS Particle Filtering (Ours)<br><br>|70.0 3/30 79.6 6/30<br><br>81.8 4/30<br>82.6 4/30<br>83.0 4/30 82.8 5/30<br><br>84.6 7/30<br><br><br>|
|---|---|
|Qwen-2.5 Math 7B Instruct Greedy Self Consistency BoN WBoN Beam Search DVTS Particle Filtering (Ours)<br><br>|79.6 5/30 84.0 4/30<br><br>82.6 5/30<br>83.0 5/30<br><br>86.9 7/30<br><br>84.6 6/30<br><br>87.7 10/30<br><br><br><br><br>|

Table 1: Results of various LLMs on MATH500 and AIME 2024, highlighting particle filtering performance. All methods used a compute budget of 32 generations with Qwen2.5-Math-PRM-7B as the reward model. Notably, Qwen2.5-Math-7B, with just 32 particles, matches o1-preview on MATH500, demonstrating PF’s effectiveness.

and 10/30 on AIME. This surpasses the o1-preview model and highlights the potential of inference-time scaling to close the gap with—or even exceed—the performance of proprietary frontier LLMs using smaller, open models.

For results on additional model families and broader ablations, see Appendix F.

#### 4.3 Results on Generalized Reasoning Datasets

To evaluate whether our inference-time scaling method generalizes beyond mathematical reasoning, we test Particle Filtering on two diverse instruction-following benchmarks: FinanceBench [12], which evaluates financial QA over real-world documents, and NumGLUE Task 2 (Chemistry) [20], which targets numerical reasoning in scientific contexts.

As shown in Table 2, Particle Filtering consistently outperforms all other inference-time scaling baselines, achieving the highest accuracy on both datasets. This shows that our method is effective not only for mathematical reasoning but also for broader instruction-following and domain-specific tasks. We use Llama 3.1-8B-Instruct as the policy model and Qwen2.5-Math-PRM-7B as the reward model, with 8 particles for FinanceBench and 32 for NumGLUE.

We note that although we use Qwen2.5-Math-PRM-7B - a reward model trained primarily for mathematical process evaluation - it performs surprisingly well as a reward model on these non-math domains. We hypothesize that such RMs implicitly learn broader reasoning evaluation capabilities during their training, not limited strictly to mathematical content. We leave deeper exploration of this hypothesis and the development of domain-specific or generalized PRMs for future work.

#### 4.4 Scaling with inference-time compute

We now zoom in on how PF scales with inference-time compute. Fig-

|Method|FinanceBench<br><br>|NumGLUE Task 2 (Chemistry)|
|---|---|---|
|Greedy BoN Self Consistency Beam Search Particle Filtering (Ours)<br><br>|62.67 68.00 68.67 67.33<br><br>70.33|71.69 80.92 79.32 80.47<br><br>84.22|

- ure 4 shows the change of performance (in terms of accuracy) with an increasing computation budget (N = 1,2,4,8,16,32,64,128) across Math SLMs and Non-Math SLMs. As we can see, PF scales 4–16x faster than the next best competitor DVTS, e.g. DVTS requires a budget of 32 to reach the same performance of PF with a budget of 8 with Llama-3.2-1B-Instruct and requires a budget of 128 to reach the performance of PF with a budget of 8 with LLama-3.1-8B-Instruct.

4.5 Ablation study

20 21 22 23 24 25 26 27

Budget (# of model generations)

0.2

0.3

0.4

0.5

0.6

Accuracy

Qwen2.5-Math-PRM-7B

Llama3.1-8B-PRM-Deepseek-Data

EurusPRM-Stage2

math-shepherd-mistral-7b-prm

- (a) Results of ablation comparing the performance of PF across PRMs.

20 21 22 23 24 25 26 27

Budget (# of model generations)

0.2

0.3

0.4

0.5

0.6

Accuracy

Model Aggregation

Last

Min

Product

- (b) Effect of different aggregation strategies for Qwen2.5-Math-PRM-7B.

20 21 22 23 24 25 26 27

Budget (# of model generations)

0.2

0.3

0.4

0.5

0.6

Accuracy

Temp=0.4

- Temp=0.6

- Temp=0.8

- Temp=1.0

- Temp=1.2

- Temp=1.4

- (c) Results of ablation comparing the effect of temperature across different particle budget.

Performance of different PRMs Figure 5a shows an ablation on 100 MATH500 questions, comparing our method’s accuracy across reward functions as the number of particles increases.

Qwen2.5-Math-PRM-7B consistently outperforms other models, making it the natural choice for our main results. Interestingly, while EurusPRM-Stage2 performs poorly with smaller budgets, it improves and eventually matches Qwen2.5-Math-PRM-7B at higher budgets.

Reward aggregation within PRMs As mentioned in Section 3.1 / Appendix A and reported by prior works [32], there are multiple ways to use PRMs to calculate reward scores, which can significantly impact final performance. Fig-

- ure 5b studies three existing methods for combining PRM scores—using the last reward, the minimum reward, and the product of all rewards. We also study “Model Aggregation," where the PRM is used as an ORM with partial answers.

Table 2: Results on Non-Math Datasets

As we can see, using Model Aggregation—in essence, feeding into a PRM the entire partial answer alongside the question - scales the best with an increasing budget.

Controlling the state transitiontemperatures in LLM generation We investigate the effect of different LM sampling temperatures on the scaling of our method across varying numbers of particles. The results of our ablation study on a 100-question subset of MATH questions are shown in Figure 5c.

Our findings show that the common LLM temperature range of 0.4–1.0 performs well, with minimal accuracy variation across budgets. Following [3], we use temperature 0.8 for all experiments.

Budget allocation over iterations and parallelism The multi-iteration and parallel-chain extensions introduced in Section 3.1 / Appendix B provide two more axes to spend computation beyond the number of particles. We explore how different budget allocations affect performance in the appendix.

### 5 Related Work

Process reward models (PRMs) aim to provide more granular feedback by evaluating intermediate steps rather than only final outputs. They are trained via process supervision, a training approach where models receive feedback on each intermediate step of their reasoning process rather than only on the final outcome. [17] propose a step-by-step verification approach to PRMs, improving the reliability of reinforcement learning. DeepSeek PRM [26] uses Mistral to annotate training data for PRMs . [32] introduces Qwen-PRM, which combines both Monte Carlo estimation and model/human annotation approach to prepare training data for a PRM. PRIME [6] proposes to train an outcome reward model (ORM) using an implicit reward objective. The paper shows that implicit reward objective directly learns a Q-function that provides rewards for each token, which can be leveraged to create process-level reward signal. This process eliminates the need for any process labels, and reaches competitive performance on PRM benchmarks.

Inference-time scaling is a key training-free strategy for enhancing LLM performance. [4] investigates best-of-N (BoN) decoding, showing quality gains via selective refinement. [22] analyzes how scaling compute improves inference efficiency from a compute-optimality view. While not implementing full Monte Carlo tree search (MCTS), [34] explores a tree-search-inspired approach within language models. [11] introduces rSTAR, which combines MCTS for data generation and training to improve mathematical reasoning. [3] discusses beam search and dynamic variable-time search (DVTS) as inference-time scaling methods for open-source LLMs. DVTS runs multiple subtrees in parallel to avoid all leaves getting stuck in local minima.

Particle-based Monte Carlo methods are powerful tools for probabilistic inference. Sequential Monte Carlo [7] or particle filtering [24] has been a classical way to approximate complex posterior distributions over state-space models. Particle Gibbs (PG) sampling [2] extends these approaches by integrating MCMC techniques for improved inference. [15] and [18] use token-based SMC within probabilistic programs to steer LLMs, while [9] apply token-based SMC for self-constrained generation. [33] and [8] introduce Twisted SMC for inference in language models. We note that our method differs from [8] in several key ways. First, TSMC relies on a ground-truth verifier and requires joint training of a generator and value function, whereas our approach uses an off-the-shelf generator and a noisy pretrained reward model (PRM), requiring no additional training. Second, our method generalizes to domains lacking ground-truth verifiers, such as instruction and broader language tasks. Finally, we demonstrate significantly stronger empirical results. The authors of TSMC did not release code, and their method requires additional training. We therefore compare our method—using the same generator model (DeepSeekMath7B) and dataset (MATH500) as in their results table—and find it achieves 75.4% accuracy with 128 samples, outperforming TSMC by 14.6 points using fewer than half the samples and no fine-tuning.

Decision making with uncertainty The way of representing uncertainty using softmax is often referred as Boltzmann exploration in the multi-armed bandit (MAB) literature [14]. While formulating it as a MAB problem allows it to use a scheduling on the softmax temperature and to derive regret bounds [5], we no longer have the same unbiasedness from the particle filtering / SMC formulation.

### 6 Conclusion

In this paper, we introduce a particle filtering algorithm for inference-time scaling. To address the limitations of deterministic inference-time scaling—namely, early pruning from imperfect reward models—we adapt particle-based Monte Carlo methods to maintain a diverse population of candidate sequences and balance exploration and exploitation. This probabilistic framing enables more resilient generation and opens a principled path for integrating uncertainty into LLM inference. Our evaluation shows these algorithms consistently outperform search-based approaches by a significant margin.

However, inference-time scaling comes with computational challenges. Hosting and running a reward model often introduces high latency, making the process more resource-intensive. For smaller models, prompt engineering is often required to ensure outputs adhere to the desired format. Finally, hyperparameters like budget are problem-dependent and may require tuning across domains.

We hope that the formal connection of inference scaling to probabilistic modeling established in this work will lead to systematic solutions for current limitations of these methods and pave the way for bringing advanced probabilistic inference algorithms into LLM inference-time scaling in future work.

### References

- [1] AI-MO. Aimo validation aime dataset. https://huggingface.co/datasets/AI-MO/ aimo-validation-aime, 2023. Accessed: 2025-01-24.
- [2] Christophe Andrieu, Arnaud Doucet, and Roman Holenstein. Particle Markov Chain Monte Carlo Methods. Journal of the Royal Statistical Society Series B: Statistical Methodology, 72(3):269–342, June 2010.
- [3] Edward Beeching, Lewis Tunstall, and Sasha Rush. Scaling test-time compute with open models, 2024.
- [4] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, July 2024.
- [5] Nicolò Cesa-Bianchi, Claudio Gentile, Gábor Lugosi, and Gergely Neu. Boltzmann exploration done right. Advances in neural information processing systems, 30, 2017.
- [6] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards, 2025.
- [7] Pierre Del Moral. Sequential Monte Carlo Methods for Dynamic Systems: Journal of the American Statistical Association: Vol 93, No 443. https://www.tandfonline.com/doi/abs/10.1080/01621459.1998.10473765, 1997.
- [8] Shengyu Feng, Xiang Kong, Shuang Ma, Aonan Zhang, Dong Yin, Chong Wang, Ruoming Pang, and Yiming Yang. Step-by-step reasoning for math problems via twisted sequential monte carlo, 2024.
- [9] Gabriel Grand, Joshua B. Tenenbaum, Vikash K. Mansinghka, Alexander K. Lew, and Jacob Andreas. Self-steering language models, 2025.
- [10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao

Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez,

Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024.

- [11] Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking, January 2025.
- [12] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. Financebench: A new benchmark for financial question answering, 2023.
- [13] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020.
- [14] Volodymyr Kuleshov and Doina Precup. Algorithms for multi-armed bandit problems. arXiv preprint arXiv:1402.6028, 2014.
- [15] Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, and Vikash K. Mansinghka. Sequential monte carlo steering of large language models using probabilistic programs, 2023.
- [16] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023.
- [17] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step, May 2023.
- [18] João Loula, Benjamin LeBrun, Li Du, Ben Lipkin, Clemente Pasti, Gabriel Grand, Tianyu Liu, Yahya Emara, Marjorie Freedman, Jason Eisner, Ryan Cotterell, Vikash Mansinghka, Alexander K. Lew, Tim Vieira, and Timothy J. O’Donnell. Syntactic and semantic control of large language models via sequential monte carlo. In The Thirteenth International Conference on Learning Representations, 2025.
- [19] David JC MacKay. Information theory, inference and learning algorithms. Cambridge university press, 2003.
- [20] Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. ACL, 2022.
- [21] OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren,

- Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024.
- [22] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters, August 2024.
- [23] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024.
- [24] Robert H. Swendsen and Jian-Sheng Wang. Nonlinear filtering: Interacting particle resolution

- ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0764444297847787, 1986.

- [25] Simo Särkkä. Bayesian Filtering and Smoothing. Institute of Mathematical Statistics Textbooks. Cambridge University Press, 2013.
- [26] Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024.
- [27] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023.
- [28] Wei Xiong, Hanning Zhang, Nan Jiang, and Tong Zhang. An implementation of generative prm. https://github.com/RLHFlow/RLHF-Reward-Modeling, 2024.
- [29] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.
- [30] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun. Advancing llm reasoning generalists with preference trees, 2024.

- [31] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301, 2025.
- [32] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The Lessons of Developing Process Reward Models in Mathematical Reasoning, January 2025.
- [33] Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Grosse. Probabilistic inference in language models via twisted sequential monte carlo, 2024.
- [34] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models, 2024.

### A Aggregation Strategies for PRM Scores

To compute particle weights during generation, we aggregate per-step scores from the process reward model (PRM) rˆ. Our default uses a product of step-level rewards to align with the factorized likelihood structure, but alternative aggregation strategies (e.g., min, last-step, or model-based) may offer different trade-offs.

The weight update step in particle filtering depends on how rewards are assigned to partial trajectories using a preference or reward model (PRM) rˆ. Since PRMs often provide per-step scores, aggregating them into a single weight requires a strategy that balances theoretical correctness and practical utility.

We consider the following four aggregation approaches:

- • Product (prod): Computes the product of step-level rewards across all tokens. This aligns directly with the factorized likelihood structure used in the PF objective ((2)), enabling online weight updates as generation proceeds.
- • Minimum (min): Takes the minimum reward seen so far. This penalizes trajectories for weak intermediate steps, which may help in discouraging risky completions. However, it prevents online updates because the entire prefix must be scored to determine the weight.
- • Last-step (last): Uses only the most recent step’s reward. Although not aligned with a likelihood-based interpretation, this method is computationally efficient and reflects the scoring mode used in [3].
- • Model-based aggregation: Instead of relying on step-wise rewards, this method repurposes the PRM in a black-box fashion to assign a single scalar score to the full partial trajectory. This helps smooth over noisy token-level scores and can be more stable, especially when PRMs are inconsistent across steps. The model receives the prompt and prefix and returns a scalar reward.

Appendix G shows how the input format for this black-box mode differs from standard per-step PRM usage. We compare all strategies empirically in 4.5 and find that the optimal choice varies with the PRM’s training and evaluation objective.

### B Multiple Iterations and Parallel Chains

Here, we explore how different ways to allocate budgets change the performance in the appendix. Specifically, we study for a fixed budget N × T × M, how the combination of N,T,M can yield the best performance, where N is the number of particles, T is the number of iterations, and M is the number of parallelism.

#### B.1 Particle Gibbs

Particle Gibbs is a type of MCMC algorithm that uses PF as a transition kernel [2]. Specifically, at each iteration, PG samples a new set of particles using PF with a reference particle from the previous iteration. This integration combines the efficiency of PF with the theoretical guarantees of MCMC, making PG suitable for high-dimensional or challenging posterior distributions. The adaption of PG to inference-time scaling is essentially a multi-iteration extension of the PF algorithm presented, which works as follows: For each iteration, we run a modified PF step with an additional sampling step to sample 1 reference particle according to the equation in the resampling step of Section 3.1. For any PF step that is not the initial step, the PF is executed with a reference particle: This reference particle is never replaced during the resampling step, but its partial trajectory can still be forked during resampling. We detail the PG version of inference-time scaling in Algorithm 2 of Appendix D. Note that typically, a reasonably large number of particles is needed to show the benefits of multiple iterations, which we also confirm in our results in Section 4.5.

Allocating budget between N and T Figure 6 shows results of Llama-3.2 1B model when configured with various test-time compute budget allocations. Although the plot shows that various Particle Gibbs configurations do not have a marked benefit over an equivalently budgeted particle filtering run, a PG experiment with 16 particles and 4 iterations powered by a Qwen 2.5 7B Math Instruct

0.60

0.55

0.50

Accuracy

0.45

Particle Filtering

0.40

Particle Gibbs (4 particles per group) Particle Gibbs (8 particles per group) Particle Gibbs (16 particles per group)

0.35

22 23 24 25 26 27

Budget (# of model generations)

- Figure 6: Comparison of PF and Particle Gibbs with different numbers of iterations, evaluated on a 100-question subset of the MATH-500 dataset using Llama-3.2-1B-Instruct as the policy model.

policy model achieved a 87.2% accuracy on MATH500, beating o1 performance. Configurations with larger N values typically do better than equivalently budgeted runs with less particles.

B.2 Parallel Tempering

Parallel tempering In parallel tempering (aka replica exchange MCMC sampling), multiple MCMC chains run in parallel at different temperatures and swap the states to allow better exploration. The key idea is that the chain running in high temperature can explore better, e.g. traversing between different modes of the target, and the swap makes it possible to let the low temperature chain exploit the new region found by the other chain. We detail the complete parallel tempering version of inference-time scaling in the Algorithms section of the Appendix below Algorithm 3 of D, while we only explore a special case of it (multiple chains with single iteration) in our experiments.

Allocating budget between N and M Figure 7 shows PF and 3 PT configurations over a set of increasing numbers of budgets. First, as we can see, for any fixed N, increasing M also improves

22 23 24 25 26 27

Budget (# of model generations)

0.36

0.38

0.40

0.42

0.44

0.46

0.48

Accuracy

Particle Filtering

Parallel Tempering (4 particles per group) Parallel Tempering (8 particles per group) Parallel Tempering (16 particles per group)

- Figure 7: Comparison of PF and PT with different particle group sizes, evaluated on a 100-question subset of the MATH500 dataset using Llama-3.2-1B-Instruct as the policy model.

the performance. This may be helpful when combining batch generation with distributed computing. Second, PT with N = 16 has a better overall scaling than PF. This indicates that there is some optimal budget allocation over parallel chains that can further improve the overall performance of our main results.

We leave the exploration over the optimal configuration of N,T,M jointly as a future work.

### C Proof of Theorem 2

Theorem 2 (Unbiasedness of Expected Accuracy). Let {(w(i),x(i))} be weighted particles from Algorithm 1 and is_correct(x) is a function to check the correctness of response x. We have

w(i) is_correct(x(i)) =

p ˆM(x1:T | c,o1:T = 1) is_correct(x(i)) ,

E

x

i

where the expectation is over the randomness of the algorithm itself.

Proof. This is a direct result of applying the unbiasedness property of particle filtering on a welldefined expectation p Ex∼p{f(x)} of any function f over a distribution p: the Monte Carlo estimate using weighted samples from particle filtering is an unbiased estimate of this expectation. As is_correct(·) is a binary function, the expectation of the estimate is finite thus well-defined and therefore the unbiasedness of accuracy holds.

| |
|---|

### D Side by Side Comparison of Particle Filtering vs Beam Search

math problem

softmax

+2.1 -1.2 +1.3 +0.1

distribution:

sample N: 1,1,3,4

(a) Particle filtering uses the rewards to produce a softmax distribution and does stochastic expansion of N based sampling.

math problem

top-MN

+2.1 -1.2 +1.3 +0.1

selected: 1,3

expand M

(b) Beam search treats the rewards as exact and performs deterministic expansion based on beam size N and beam width M.

- Figure 8: A side-by-side comparison between particle filtering and its closest search-based counterpart, beam search. Compared with beam search in Figure 8b where the selection and expansion is deterministic (implicitly assumes the rewards are correct), particle filtering in Figure 8a trusts the rewards with uncertainty and propagates the expansion via sampling.

### E Algorithms

- Algorithm 1 Particle Filtering for Inference-Time Scaling Input: the number of particles N, a reward model rˆ, a LLM pM and the prompt c

Initialize N particles {x(1i) ∼ pM(· | c)}Ni=1 t ← 1

while not all particles stop do

Update rewards w = [ˆr(x(1)1:t),...,rˆ(x(1:Nt))] Compute softmax distribution θ = softmax(w)

Sample indices {jt(i)}Ni=1 ∼ Pt(j = i) = θi Update the set of particles as {x(j

(i) t )

1:t }Ni=1

Transition {x(t+1i) ∼ pM(· | c,x(1:i)t)}Ni=1 t ← t + 1

end while Return: the set of weighted particles in the end

- Algorithm 2 Particle Gibbs for Inference-Time Scaling Input: same as Algorithm 1 with the number of Gibbs iterations T

Run Algorithm 1 to get a set of particles {x(1:i)t}Ni=1 for j = 1,...,T do

Compute rewards w = [ˆr(x(1)1:t),...,rˆ(x(1:Nt))] Compute softmax distribution θ = softmax(w)

Sample reference particle xref1:t := x(1:jt) where j ∼ P(j = i) = θi Initialize N − 1 particles {x(1i) ∼ pM(· | c)}Ni=1−1 t ← 1 while not all particles stop do

##### Update w = [ˆr(x(1)1:t),...,rˆ(x1:(Nt−1)),rˆ(xref1:t)] Compute softmax distribution θ = softmax(w)

Sample indices {jt(i)}Ni=1 ∼ Pt(j = i) = θi Update the set of particles as {x(j

(i) t )

1:t }Ni=1

Transition {x(t+1i) ∼ pM(· | c,x(t+1i) )}Ni=1 t ← t + 1

end while end for Return: the set of particles in the end

For a set of parallel chains with temperatures T1 > T2 > ..., at each iteration, we swap the states of every pair of neighboring chains k,k + 1 with the following probability

πk(x(k+1))πk+1(x(k)) πk(x(k))πk+1(x(k+1))

A = min 1,

, (4)

where πk,πk+1 are the two targets (with different temperatures) and xk,xk+1 are their states before swapping.

- Algorithm 3 Particle Gibbs with Parallel Tempering for Inference-Time Scaling

Input: same as Algorithm 2 with the number of parallel chains M and a list of temperature T1,...,TM for j = 1,...,T do

for k = 1,...,M do if j = 1 then

Run Algorithm 1 to get a set of particles {x(1:i)t}Ni=1 for chain k else

Initialize N − 1 particles {x(1i) ∼ pM(· | c)}Ni=1−1 t ← 1

while not all particles stop do

Update w = [ˆr(x(1)1:t),...,rˆ(x1:(Nt−1)),rˆ(xref1:t)] Compute softmax distribution θ = softmax(w/Tk)

Sample indices {jt(i)}Ni=1 ∼ Pt(j = i) = θi Update the set of particles as {x(j

(i) t )

1:t }Ni=1

Transition {x(t+1i) ∼ pM(· | c,x(t+1i) )}Ni=1 t ← t + 1

end while end if

Compute rewards w = [ˆr(x(1)1:t),...,rˆ(x(1:Nt))] Compute softmax distribution θ = softmax(w/Tk)

Sample reference particle xref1:t := x(1:jt) where j ∼ P(j = i) = θi end for for k = 1,...,M − 1 do

Exchange the reference particle between chain k and k + 1 with probability according to

(4)

end for end for Return: M set of particles in the end

### F Particle Filtering Results with More Generator Models

Below, we show further results using particle filtering to inference scale a wider variety of generator models.

|Model Method<br><br>|MATH500 AIME2024|
|---|---|
|Llama-3.2-1B-Instruct Greedy Particle Filtering (Ours)<br><br>|26.8 0.0 59.6 10.0<br><br>|
|Llama-3.2-8B-Instruct Greedy Particle Filtering (Ours)<br><br>|49.9 6.6 74.4 16.6<br><br>|
|phi-4 Greedy Particle Filtering (Ours)<br><br>|79.8 16.6 83.6 26.6<br><br>|
|Mistral-Small-24B-Instruct-2501 Greedy Particle Filtering (Ours)<br><br>|69.2 10 83.4 23.3<br><br>|

Table 3: Performance of LLMs on MATH500 and AIME 2024 using greedy decoding and Particle Filtering (ours). Particle Filtering is run with 64 generations per problem.

### G Inference Prompt Template

Evaluation System Prompt Solve the following math problem efficiently and clearly:

- - For simple problems (2 steps or fewer): Provide a concise solution with minimal explanation.
- - For complex problems (3 steps or more): Use this step-by-step format:

- ## Step 1: [Concise description] [Brief explanation and calculations]
- ## Step 2: [Concise description] [Brief explanation and calculations]

Regardless of the approach, always conclude with: Therefore, the final answer is: $\boxed{answer}$. I hope it is correct. Where [answer] is just the final number or expression that solves the problem. PRM Input Format

- ## Step 1: [Concise description] [Brief explanation and calculations] <reward_token>
- ## Step 2: [Concise description] [Brief explanation and calculations] <reward_token>
- ## Step 3: [Concise description] [Brief explanation and calculations] <reward_token>

ORM Input Format

- ## Step 1: [Concise description] [Brief explanation and calculations]
- ## Step 2: [Concise description] [Brief explanation and calculations]
- ## Step 3: [Concise description] [Brief explanation and calculations] <reward_token>

### H Evaluation details

Parsing and scoring Following prior work on mathematical reasoning benchmarks [29], we apply their heuristic-based parsing and cleaning techniques to robustly extract the boxed expression. These heuristics handle spacing variations, formatting inconsistencies, and other artifacts in model outputs. For answer verification, we follow [3], converting responses to canonical form. Ground truth and generated answers are transformed from LaTeX into SymPy expressions, simplified for normalization, and converted back to LaTeX. Exact match is determined using two criteria: numerical equality, where expressions evaluate to the same float, and symbolic equality, where they are algebraically equivalent in SymPy [3]. Accuracy is computed as the fraction of problems where the generated answer exactly matches the ground truth.

### NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We show experiments that directly address and back up every claim we make in the abstract and introduction. Please see the Evaluation section for our empirical results.

Guidelines:

- • The answer NA means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

###### 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We discuss several limitations of our work in the conclusion section. For clarity, we copy them here: "However, inference-time scaling comes with computational challenges. Hosting and running a reward model often introduces high latency, making the process more resource-intensive. For smaller models, prompt engineering is often required to ensure outputs adhere to the desired format. Finally, hyperparameters like budget are problem-dependent and may require tuning across domains."

Guidelines:

- • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate "Limitations" section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes] Justification: We have 1 theorem in our paper, and its proof is provided in Appendix C. Guidelines:

- • The answer NA means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We include full details of how exactly our overall methodology work. We also provide key details into the hyperparameter selection and ablation process, which significantly helps reproducability. We also provide details into several reward model aggregation techniques, as well as why we chose the one we chose, in our paper. We also include details on several different Process Reward Models and include ablations showing why we chose the one we chose. Our experiments are reproducible.

Guidelines:

- • The answer NA means that the paper does not include experiments.
- • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: We do not currently include code as to not break author confidentiality. However, we will completely open source our code upon acceptance of the paper to encourage as many people as possible to use our work.

Guidelines:

- • The answer NA means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide full details on all datasets we used, as well as their citations. We list exactly which models and versions we used as generator models. We also provide key details into the hyperparameter selection and ablation process, which significantly helps reproducability. We also provide details into several reward model aggregation techniques, as well as why we chose the one we chose, in our paper. We also include details on several different Process Reward Models and include ablations showing why we chose the one we chose. Our experiments are reproducible.

Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Due to computational limitations on an academic budget, we were not able to run every single experiment multiple times to produce accurate and fair error bars across every experiment in the paper. However, we have run many experiments several times during the research and development process (both several times and by several different people) and are very confident in our results.

Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: Although we include information about the computational burden of running inference time scaling experiments, we do not provide formal information about which exact computational resources we used, as we used different numbers of GPUs for different experiments. That being said, we have developed a (to be released upon acceptance) open source library for our work that is able to be used completely off the shelf and contains very simple information about how to run it.

Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We can confirm that, in every respect, we do not violate the NeurIPS Code of Ethics. Our research does not have negative societal consequences, nor does it involve human subjects. We do not use any private or sensitive data.

Guidelines:

- • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Our work does not have any negative societal consequences. We discuss the positive impacts of inference scaling, as it opens up higher level language model performance to those who are only able to access smaller models.

Guidelines:

- • The answer NA means that there is no societal impact of the work performed.
- • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [No] Justification: We do not release any data or models of our own. Instead, we only use off-the-shelf open source models, and therefore there are no possibilities of misuse. Guidelines:

- • The answer NA means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: Our work only uses open source models, and we cite every model that we use. Therefore, all creators of the original models are credited in this work. Guidelines:

- • The answer NA means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA] . Justification: We do not release any new assets in this paper - instead, we discuss how to enhance the performance of already existing open-sourced models. Guidelines:

- • The answer NA means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] Justification: Our research does not include any human subject experiments or crowdsourcing experiments. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] . Justification: Our work does not include any human subjects, and we did not need IRB approvals. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] Justification: LLM usage did not impact the core methodology, scientific rigorousness, or originality of the research. Guidelines:

- • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

