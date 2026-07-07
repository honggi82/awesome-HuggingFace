arXiv:2507.04103v4[cs.AI]13Feb2026

# How to Train Your LLM Web Agent: A Statistical Diagnosis

Dheeraj Vattikonda∗,1,2,5, Santhoshi Ravichandran∗,1,2, Emiliano Penaloza∗,1,2,6, Hadi Nekoei1,2,6, Megh Thakkar1, Thibault Le Sellier de Chezelles1,2,3, Nicolas Gontier1, Miguel Muñoz-Mármol1, Sahar Omidi Shayegan1,2,5, Stefania Raimondo1, Xue Liu2,5, Alexandre Drouin1, Laurent Charlin2,4, Alexandre Piché1, Alexandre Lacoste1, Massimo Caccia∗,1

1ServiceNow AI Research, 2Mila-Quebec AI Institute, 3Polytechnique Montréal, 4HEC Montréal, 5McGill University, 6Univeristé de Montréal

∗Equal contribution

LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges, first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions, and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via SFT, followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices in setting where exhaustive sweeps are impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy only requires 55% of the compute to match the peak of pure SFT on MiniWob++, pushing the compute–performance Pareto frontier and is the only strategy that can close the gap with closed-source models.

: www.youtube.com/watch?v=zTI4kFqy9dw Correspondence: {dheeraj.vattikonda, massimo.caccia1}@servicenow.com

[Figure 1]

Held-out Goals

Held-out Tasks

0.65

| |
|---|

| |
|---|

0.65

0.60

0.60

0.55

0.55

SuccessRate

SuccessRate

0.50

0.50

0.45

0.45

0.40

0.40

0.35

0.35

0.30

0e18 5e18 10e18 15e18 Compute (FLOPs)

0e18 5e18 10e18 15e18 Compute (FLOPs)

20e18 25e18

20e18 25e18

[Figure 2]

[Figure 3]

[Figure 4]

Llama-3.1 8b (SFT) GPT-4o 10

Llama-3.3 70b (Teacher)

[Figure 5]

0 2

4 6 8

RL branching index

Figure 1 Compute–performance frontier on MiniWoB++ (results averaged over two seeds). The blue curve shows pure SFT on teacher demonstrations. Warm-colored curves represent hybrid runs that branch off from SFT checkpoints and continue training with RL. Early transitions to RL push the Pareto frontier achieving higher success rates for the same compute and is the only approach able to achieve over 30% improvement on both held-out goals (left) and held-out tasks (right) closing the gap between open and closed-source models. See figure 4 for the Qwen2.5 7B plot.

## 1 Introduction

Large language model (LLM) agents for web interfaces have advanced rapidly, but open-source systems still trail proprietary ones. Bridging this gap would allow organizations to train smaller, cost-efficient agents tailored to their needs while maintaining data privacy. Yet, despite impressive progress of open-source models in domains like math and code generation, advances in training web-capable LLM agents remain limited by mainly by the lack of attention to multi-turn, long-horizon tasks and the high cost and low reproducibility of current training pipelines.

Most research centers on single-step tasks like code or math domains with rapid feedback and simplified credit assignment which fall short of real-world web environments requiring sequential decisions and longhorizon planning. Recent benchmarks like WebArena (Zhou et al., 2023), WorkArena (Drouin et al., 2024), OSWorld (Xie et al., 2024), and The Agent Company (Xu et al., 2024) have exposed how brittle current methods become under delayed rewards, sparse feedback, and compounding errors. Addressing these settings demands not just better agents, but reproducible, compute-efficient training pipelines an area we directly tackle.

However, building such pipelines is nontrivial. Modern LLM post-training often involves a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), with performance sensitive to a large number of interacting hyperparameters (Hochlehnert et al., 2025). Single-seed results are noisy and misleading, but exhaustive tuning and multi-seed sweeps (e.g. (Abdin et al., 2025)) remain infeasible for most labs due to the cost of LLM training. This makes it all the more critical to design pipelines that are not only effective but statistically robust and accessible within realistic compute budgets.

In this paper, we tackle these gaps by providing a statistically driven diagnosis of training LLM agents for web-based, multi-step tasks. Specifically, we study how to allocate compute between expensive, high-quality off-policy demonstrations from a large teacher model and cheaper on-policy rollouts from a smaller student model. We analyze this tradeoff across two levels of generalization, held-out goals being tasks encountered during training but with novel goals and held-out tasks, which are entirely unseen during training.

To study this compute-performance trade-off we use a two-stage training pipeline. First, a LLaMA 3.3 70B teacher model generates K successful trajectories to warm-start a smaller LLaMA 3.1 8B student via SFT. We then branch of various SFT checkpoints where training continues with an on-policy RL phase using Group Relative Policy Optimization (GRPO) (DeepSeek-AI et al., 2025). Our central objective is to determine the optimal compute allocation and hyperparameter mix for training web-based LLM agents. To this end, we run 1,370 training configurations, varying key hyperparameters and branching points between SFT and RL. We then apply bootstrap-based analysis to estimate the impact of different hyper-parameters on downstream performance and how they vary across branching SFT checkpoints. This data-driven approach enables us to identify important considerations to get the most out of each run. We use this method to show the optimal mix between SFT and RL in the MiniWob++ environment, achieving better task accuracy at a significantly lower cost. Additionally, we provide concrete recommendations on compute allocations between SFT and RL on the more demanding WorkArena environment.

Putting things together, we show how our study yields several actionable insights. First, branching into RL early, but not immediately after SFT leads to better outcomes. This hybrid strategy consistently outperforms pure SFT and pure RL, and reaches the maximal performance of pure SFT while requiring only 55% of the compute, effectively pushing the compute–performance Pareto frontier. It is also the only strategy that can close the gap with closed-source models. Second, curriculum learning and error log feedback help the less SFT warmup has been applied but can become counterproductive thereafter. Third, in GRPO, applying zero-advantage filtering consistently improve performance, while dividing the advantage with the standard deviation and wether to use an importance ratio are dependent on the amount of SFT warmup done. Fourth, decoding temperature is consistently critical, while learning rate and discount rate must also be carefully tuned.

These findings are significant for two reasons. First, they give smaller research groups a reproducible, budgetaware playbook for pushing open LLM agents closer to state-of-the-art without scaling model size. Second, they address a slice of the broader reproducibility concerns recently highlighted in the RL community (Agarwal

et al., 2021; Hochlehnert et al., 2025), offering a template for rigorous, statistically-grounded hyperparameter tuning and compute allocation in LLM agent training.

## 2 Background

This section consolidates the algorithmic ingredients used throughout the paper: (i) the MDP formulation of web-based language agents, (ii) SFT on expert traces, (iii) GRPO for RL, and (iv) curriculum and normalization techniques that stabilize training.

### 2.1 Language Agents as MDPs

We model each task as a Markov Decision Process (MDP) M = ⟨S,A,P,r,ρ0,γ⟩. A state st ∈ S is a textual context–in our case a prompt and an action at ∈ A is a textual response generated by the agent. Each action at consists of a sequence of tokens o1

t:Kt sampled autoregressively from the policy πθ(at|st) parameterized by an LLM sampled with temperature ρLLM. The environment then returns a scalar reward rt ∈ {−1,1} indicating task failure/success. In our setting we assume the environment dynamics are p(st+1 | at) ∈ P . We optimize the policy πθ to maximize the expected discounted return:

J(θ) = Eτ∼π

θ

T

γtrt (1)

t=0

Here, γ ∈ [0,1] is the discount rate, which controls the agent’s preference for immediate versus future rewards and τ = (s0,a0,r0,s1,a1,r1,...,sT) refers to a trajectory sampled from the policy.

#### 2.2 Off-policy boot-strapping via SFT We first imitate a stronger expert policy πE by minimizing the cross-entropy loss

LSFT(θ) = −Eτ∼π

E

T

log πθ(at | st) . (2)

t=0

SFT offers a high-quality, low-variance gradient but is inherently off-policy which can lead to poor generalization (Chu et al., 2025).

### 2.3 Rejection Fine-Tuning

Similar to SFT, Rejection Fine-Tuning (RFT) optimizes the policy using successful trajectories, but in this case bootstrapped of the training policy rather than a larger expert

LRFT(θ) = −Eτ∼π

θ

T

log πθ(at | st) . (3)

1{r

t=1}

t=0

### 2.4 On-Policy Improvement with Multi-Turn GRPO

After the initial SFT warmup phase, we continue training using on-policy RL using GRPO as the optimization algorithm. Like REINFORCE (Sutton et al., 1999), GRPO maximizes the expected return using a policy gradient objective. However, GRPO introduces additional structure by leveraging per-goal advantage normalization and importance weighting, wherein each trajectory is associated with a specific goal in our context, referring to the seed of the target task. For a given goal g, the group-normalized advantage function is:

rt,g − mean(Rt,g) std(Rt,g)

At,g =

,

where Rt,g = (rt,1,...,rt,G) is the set of rewards across the G goals at timestamp t. In addition to this similar to proximal policy optimization (PPO) (Abdin et al., 2025) an importance-ratio is often applied to the GRPO objective:

πθ(at | st) πθ

ηt,g =

,

(at | st)

old

where πθ is the current policy and πθ

is the behavior policy used to collect trajectories. Finally, a clippedminimum is also applied to stabilize the training process. Putting things together the GRPO objective for our multi-turn setting is:

old

JMT-GRPO(θ) = Eτ∼πθold

G

T

1 G

1 T

min ηt,gAt,g, clip(ηt,g, 1−ϵ, 1+ϵ)At,g . (4)

g=1

t=0

Traditionally the GRPO objective includes a KL penalty between the optimizing and reference policies which we do not use as early experiments showed they did not improve performance, slowed down training and required additional compute budget.

Zero-advantage filtering. Tokens with At,g = 0 contribute no learning signal yet still consume memory. Dropping them yields a constant effective batch size and modestly accelerates training (Yu et al., 2025).

### 2.5 Curriculum through Variance-Aware Boltzmann Sampling

To promote steady learning progress, we design a curriculum that prioritizes challenging tasks, neither trivial nor too difficult (Thakkar et al., 2023). Specifically, we select tasks according to a Boltzmann distribution centered around a target return µtarget which specifies the desired performance threshold, encouraging focus on partially mastered tasks, with a temperature parameter ρCurr controlling the sharpness of the distribution, with lower values concentrating probability mass tightly around µtarget.

This sampling mechanism dynamically adapts the training distribution, concentrating learning on tasks where the agent is neither already proficient nor entirely unskilled. As a result, the agent avoids premature convergence on easy tasks and prevents wasted effort on tasks far beyond its current capabilities.

## 3 Methodology

Our training pipeline consists of two sequential stages SFT followed by RL framed as a resource allocation problem. We also describe our hyperparameter sweep strategy and statistical analysis protocol that consolidates hundreds of runs into reliable conclusions. We evaluate our recipe along two axes: compute cost, measured in FLOPs (using the formula from (ben)), and model performance, assessed on both unseen training goals and held-out testing tasks.

- Stage 1 – Supervised Fine-Tuning (SFT). We begin by generating NE expert trajectories using a large teacher model. Only successful trajectories are retained after filtering, and the corresponding (s,a) pairs, along with chain-of-thought annotations, form the SFT dataset. Note that computing the cost of the SFT dataset includes both successful and discarded unsuccessful trajectories.

We then train a smaller student model for TSFT gradient steps. To explore the trade-off between SFT and RL, we branch off B times at fixed intervals along the SFT trajectory, yielding checkpoints at timesteps

tb ∈ [0,TSFT]. Each checkpoint corresponds to a student policy πθ(t

b), which initializes a distinct RL phase. The total compute used up to each branching point, FSFT(tb), includes both teacher inference and student training FLOPs accumulated over tb steps.

- Stage 2 – RL Fine-Tuning. Each policy πθ(t

b) is further trained using GRPO for TRL steps. The compute cost of this phase, FRL(TRL), includes both the FLOPs required for data collection (online rollouts) and for student updates across all TRL steps. The total FLOPs for a full training run starting from θ(tb) is computed as:

FLOPs(tb) = FSFT(tb) + FRL(TRL). (5)

By varying tb throughout SFT, we assess how shifting compute between expert-driven supervision and on-policy learning impacts final performance. This setup highlights the trade-off between the high compute cost of expert supervision and the lower-cost, but noisier, nature of on-policy learning.

### 3.1 Estimating the Uncertainty of the Hyperparameter Selection Process

Across the different SFT checkpoints, we sample 1,370 distinct configurations with ten varying hyperparameters (see Section L for details). Our objective is to study the effect of various hyperparameters (HP) on the downstream success rate of our trained agents. This comes with two important considerations. First, if we change the value of, e.g., the batch size, and we want to know if a bigger batch size is better, the learning rate and other parameters need to be readjusted close to their optimal configuration (under a fixed budget). Secondly, to account for noise, we would need to restart the same experiment several times to avoid spurious conclusions. In practice, this is out of reach. For a more computationally friendly approach, we resort to bootstrapping the collection of trials over different HP configurations. The boostrap algorithm has many desriable properties such as being a consistent and unbiased estimator for the sampling distribution of any data statistic without requiring parametric assumptions (see Section K for details), thus allowing us to reliably obtain robust estimates of both the expected value and variance of the success rate given by any HP.

Bootstrapping the hyperparameter selection process. From the full set of 1,370 training runs, we perform bootstrap resampling by drawing individual runs (with replacement). For each resample, we identify the best-performing configuration and repeat this process 1,000 times. We also compute the fraction of times each hyperparameter value "wins", which serves as an estimated probability that it belongs to the global optimum. This procedure serves two purposes: to estimate the maximum relative improvement a specific hyper-parameter provides while accounting for variation in other parameters and to offer uncertainty estimates in the form of win-rate distributions between different configurations. In addition, to better understand how optimal hyper-parameters may change depending on the amount of SFT warmup we apply this analysis across various SFT checkpoints.

Balancing unequal coverage. Due to random search, some HP values were explored more than others, biasing the winner toward the larger groups. To correct for this, each run is sampled with probability ∝ 1/group size, approximating an equal compute budget for every HP value.

We provide a detailed explanation of this procedure in Algorithm 1. Additionally in Section J we describe how use this procedure to obtain the results reported in Figure 1.

## 4 Experimental Setup

In this section, we describe the experimental setup used to validate our findings. We detail the models, benchmarks, action spaces, training framework, compute infrastructure, and evaluation protocols, ensuring that all components are aligned with the goals of studying compute-efficient, reproducible training for LLM web agents.

Models. We evaluate our approach using two teacher models, Llama 3.3 70B and Qwen 2.5 72B to generate SFT traces, with Llama 3.1 8B and Qwen 2.5 7B acting as the student model for fine-tuning. All models operate with a 16k token context window to handle the complexity of web-based tasks. For the LLama models we report results using GRPO, while for Qwen we report results using RFT as we found it more stable during training.

Benchmarks. Our experiments focus on two benchmarks. The first is MiniWoB++, a suite of 30 mediumhorizon web interaction tasks, where we observe that optimal policies typically complete tasks in 2 to 5 steps. The second is WorkArena (Drouin et al., 2024), a more challenging benchmark of 33 enterprise knowledge-work tasks, where we observe optimal policies generally complete tasks in 3 to 10 steps. These benchmarks provide a representative spectrum of sequential decision-making challenges faced by interactive LLM agents. Both benchmarks are depicted in Figure 7.

Algorithm 1 Bootstrap Estimation of Hyperparameter Importance Require: Set of training runs Results, evaluation metric column M, hyperparameter of interest H, number

of bootstrap iterations B

- 1: W = dict( ) ▷ Win counts per HP value
- 2: S = defaultdict(List) ▷ Score samples per HP value
- 3: Hvals = unique(Results[H]) ▷ Distinct values of the hyperparameter
- 4: weights = { hi : 1/count(Results[hi]) for each hi ∈ Hvals }
- 5: for b = 1 to B do
- 6: sampleb = resample(Results,weights)
- 7: h∗ = argmaxh sampleb[M] ▷ Get the winning HP value for sampleb
- 8: W[h∗] += 1
- 9: for each h in Hvals do
- 10: S[h]+ = maxsampleb[sampleb[H] == h][M]
- 11: end for
- 12: end for
- 13: for each h in Hvals do
- 14: Compute 95% confidence interval from S[h]
- 15: end for
- 16: return W, S with confidence intervals

Observation & Action Spaces. MiniWoB++ provides raw HTML trees, whereas WorkArena leverages accessibility trees (AxTrees), which we truncate to 16k tokens to meet hardware constraints. The agent operates in a discrete action space composed of high-level UI primitives: noop, fill(node, text), click(node), type(node, text), select_option(node, option), scroll(node), and hover(node). This abstraction allows the agent to interact effectively with diverse web interfaces. All agents employ chain-of-thought prompting (Wei et al., 2023). We also experiment with applying error log feedback, allowing the agent to receive explicit error messages when it takes invalid actions (during both training and testing).

Training Framework. To manage the training pipeline, we use BrowserGym (de Chezelles et al., 2025) for orchestrating Chromium-based web environments and structuring the agent’s action space, while AgentLab (de Chezelles et al., 2025) handles agent design. Model fine-tuning is conducted with Torchtune (torchtune maintainers & contributors, 2024), utilizing Fully Sharded Data Parallelism (FSDP) to enable scalable training across multiple GPUs. Given the high memory demands of long-sequence processing, we apply activation offloading and gradient checkpointing techniques, achieving approximately 40% reduction in memory usage.

SFT

Cont. SFT

RL

LLaMA 70B

0.72 0.24 0.82 0.44 0.57 0.71 0.05 0.00 0.19 0.18 0.19 0.25 0.25 0.03 0.47 0.17 0.02 0.44 0.48 0.58 0.00 0.17 0.31

0.85 0.48 0.79 0.45 0.73 0.81 0.00 0.00 0.00 0.12 0.20 0.25 0.26 0.03 0.51 0.39 0.08 0.31 0.38 0.50 0.00 0.18 0.31

|0.25| |0.47| |
|---|---|---|---|
|0.00| |0.35| |
|0.50| |0.48| |
|0.00| |0.22| |
|0.00| |0.26| |
|0.00| |0.48| |
|0.00| |0.05| |
|0.00| |0.00| |
|0.00| |0.06| |
|0.00| |0.08| |
|0.00| |0.00| |
|0.25 0.23| | | |
|0.25| |0.25| |
|0.00| |0.01| |
|0.00| |0.31| |
|0.07| |0.21| |
|0.00| |0.00| |
|0.29| |0.29| |
|0.00| |0.05| |
|0.00| |0.11| |
|0.00| |0.02| |
|0.04| |0.10| |
|0.05| |0.14| |
| | | | |

| |1.00|
|---|---|
| | |
| |1.00|
| | |
| |1.00|
| | |
| |1.00|
| | |
| |1.00|
| | |
| |0.75|
| | |
| |0.33|
| | |
| |0.33|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.00|
| | |
| |0.80|
| | |
| |0.71|
| | |
| |0.60|
| | |
| |0.57|
| | |
| |0.43|
| | |
| |0.36|
| | |
| |0.00|
| | |
| |0.27|
| | |
| |0.50|
| | |

order-apple-mac-book-pro15 knowledge-base-search order-apple-watch order-ipad-mini order-ipad-pro order-development-laptop-p-c multi-chart-value-retrieval single-chart-value-retrieval all-menu

1.0

[Figure 6]

0.8

create-incident order-sales-laptop sort-hardware-list

0.6

Task

0.4

sort-incident-list 13/26 Failed tasks

0.2

order-loaner-laptop

0.0

create-user create-change-request

multi-chart-min-max-retrieval order-standard-laptop create-problem

impersonation Average Train

Average Test

0 4 Epoch

20 Epoch

20 Epoch

Figure 2 Per-task performance of SFT and SFT+RL agents on WorkArena. The Llama 3.1 8B model is initially finetuned for 4 epochs on trajectories from a teacher Llama 3.3 70B model. Training then continues either with additional SFT or with GRPO fine-tuning up to epoch 20. The teacher model’s success rate is also shown.

Evaluation Protocol. Both environments are stochastic, and we control for randomness during training by fixing the task seed. Each task–seed pair is referred to as a ”goal”. The evaluation protocol assesses

generalization at two levels: (i) performance on held-out goals within the training tasks, and (ii) performance on entirely new, unseen held-out tasks. We consider both settings important as they capture different aspects of generalization held-out goals test reliability on trained tasks, while held-out tasks test the model’s ability to transfer skills learnt in training to entirely new situations.

In both settings we use the the average task success rate as the evaluation metric, which reflects the agent’s ability to generalize beyond its training distribution. To reduce the impact of evaluation noise during model selection, we apply a rolling average with a window size of 3 on the held-out goals, and select the checkpoint with the highest smoothed score.

For all our reported runs (SFT, RL, SFT + RL) we report the average of the non-smoothed scores at the selected checkpoint, aggregated over four runs chosen through the described model selection procedure based on our random search. For all other models, we report average scores over 100 independent runs.

Compute Allocation Protocol. We track the total floating-point operations (FLOPs) consumed during both SFT and GRPO phases, following the procedure in Section 3 and Appendix D. For RL branching, we start from the SFT run that (i) attains the highest SFT performance and (ii) exhibits stable learning, allowing us to sample checkpoints at regular intervals and cleanly study the effect of SFT warm-up. Because this run sits in the top 10% of all SFT trials by area-under-the-training-curve (AUC) the mean success rate across all checkpoints of the run, serving as a proxy for overall training efficiency, we match that selection pressure for RL by averaging the top two seeds out of ten RL trials, thus approximating the 90th-percentile of the RL distribution and ensuring a fair compute-aware comparison between strategies.

## 5 Main Results and Compute Trade-Offs

In this section, we present our primary findings and analyze the trade-offs between SFT and RL in terms of both performance and compute efficiency.

Performance Overview. Table 1 summarizes the results on MiniWoB++ and WorkArena. Combining SFT with RL consistently yields the best performance among student models. Pure SFT and pure RL each fall short, with RL from scratch particularly struggling due to sparse rewards and unstable learning dynamics. These results highlight the complementary strengths of expert demonstrations and on-policy fine-tuning.

On MiniWoB++, this approach not only maximizes student performance but also matches the teacher and significantly closes the gap with proprietary models. In contrast, WorkArena remains more challenging: while SFT+RL improves over SFT alone, student performance still lags behind the teacher and proprietary models, suggesting that stronger supervision or more effective RL strategies may be needed for complex enterprise tasks.

We observe that agent performance eventually saturates for both SFT and SFT+RL, especially on more difficult tasks. Further analysis of this saturation behavior is provided at the end of this section. Notably, WorkArena’s test set can be easier than its training set for certain agents, which adds nuance to the observed performance gaps and complicates straightforward interpretation of generalization performance.

Compute–Performance Trade-Off. Our analysis focuses on the trade-off between costly but high-quality teacher demonstrations and cheaper, noisier on-policy rollouts. To examine this, we branch RL fine-tuning from SFT checkpoints sampled at fixed intervals along the training trajectory. Figure 1 illustrates the compute–performance frontier on MiniWoB++, where runs that combine SFT with RL (warm colors) consistently outperform pure SFT (blue), pushing the Pareto frontier forward. Early branching into RL unlocks substantial gains in compute efficiency, delivering stronger performance at lower compute budgets. For example, SFT+RL reaches the maximal performance of pure SFT on the test set (achieved at approximately 11 exaFLOPs with pure SFT) using only around 6 exaFLOPs translating to roughly 45% of compute saved (11 vs 6 exaFLOPs).

Notably, this is the only strategy that closes the performance gap with closed-source models like GPT-4o. The trend is consistent across both held-out goals and held-out tasks: warm-started RL yields higher performance

Table 1 Comparison of our method with baselines on the held-out train and test splits of MiniWoB++ and WorkArena. Here ± are the averaged standard errors computed separately for each of the two top seeds.

MiniWoB++ WorkArena Held-out Goals Held-out Tasks Held-out Goals Held-out Tasks

Model

Claude-3.5-Sonnet 70.5±2.0 70.4±4.3 52.5±3.2 70.0±5.5 GPT-4o 65.7±2.1 64.3±4.5 42.1±3.2 55.7±5.9 GPT-4o-Mini 56.2±2.2 66.1±4.4 27.1±2.9 28.6±5.4 Llama-3.1-70B-Instruct 57.0±2.2 65.2±4.4 25.0±2.8 32.9±5.6 o1-Mini 69.7±2.1 66.1±4.4 53.8±3.2 68.6±5.5 Llama-3.1-405B-Instruct 65.9±2.4 65.2±2.4 39.2±2.4 58.6±2.5

Llama-3.3-70B Instruct (Teacher) 63.2±2.4 61.9±2.4 36.0±2.4 44.0±2.5 Llama-3.1-8B Instruct (Student) 29.5±2.3 36.4±2.4 8.3±1.5 4.2±1.2 Llama-3.1-8B RL (Ours) 43.5±3.5 43.5±3.5 8.0±1.9 11.5±2.3 Llama-3.1-8B SFT (Ours) 55.2±2.5 56.7±2.5 28.4±2.0 26.4±2.0 Llama-3.1-8B SFT+RL (Ours) 67.2±2.1 63.3±2.2 35.4±2.1 28.8±2.0

Qwen-2.5-72B Instruct (Teacher) 61.0±3.4 59.0±3.4 33.3±2.4 27.0±2.5 Qwen-2.5-7B Instruct (Student) 32.8±3.3 37.0±3.4 5.2±2.2 10.0±3.8 Qwen-2.5-7B RL (Ours) 52.5±3.5 53.0±3.5 4.5±1.5 7.0±1.8 Qwen-2.5-7B SFT (Ours) 57.0±3.5 56.5±3.5 25.0±4.3 21.0±4.1 Qwen-2.5-7B SFT+RL (Ours) 65.0±3.4 61.5±3.4 32.5±3.3 25.0±3.1

than either SFT or RL alone, reinforcing the importance of blending expert supervision with on-policy learning. These findings underscore the need to balance sample efficiency from demonstrations with the compute efficiency of on-policy learning, and the results generalize to Qwen2.5 7B (see figure 4).

Task Performance Saturation and Analysis Despite extensive post-training, agent performance on the WorkArena benchmark plateaus at around 40% after just 9–10 epochs. This stagnation appears to stem from the intrinsic difficulty of certain tasks (e.g as sorting and filtering tasks) that even frontier models struggles to solve (see Figure 2). A per-task breakdown shows that while both SFT and RL agents gradually close the performance gap with the teacher model, with RL achieving a slightly higher final success rate, a significant portion of tasks (14 out of 33) remain completely unsolved. These failures are attributed to either the limitations of the teacher model or the sparsity of reward signals, both of which hamper the learning process. On-policy RL exploration proves ineffective in overcoming these challenges due to the lack of foundational skills and informative feedback. These findings underscore the need for more effective methods to address complex tasks under sparse reward settings. Additional per-task performance results for WorkArena and Miniwob are provided in Section C.

Shortcut-Seeking Agents and Task Integrity During extended training we found that giving agents full administrative privileges invites creative but problematic shortcuts. For instance, in WorkArena’s “Order hardware devices” task, rather than navigating through the catalogue as intended, the agent simply edits its homepage to add a direct link to the ordering form. While this hack fulfills the success condition, it violates the task’s spirit and, worse, the modifications persist across sessions. Other agents entering the environment later inherit the altered UI, leading to failures unrelated to their own policies.

In general, agents exploited admin rights to create or delete elements, reshaping pages in ways that saved clicks but destabilised the environment. These findings argue for a tighter sandbox enough access for exploration and accomplishment, but safeguards against permanent, instance-wide side-effects.

## 6 Ablation and Sensitivity Analysis

We simulate re-running hyperparameter configurations and selecting the best-performing ones using the method described in Section 3.1. This is done across three checkpoints: the base LLaMA 3.1 8B Instruct model

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

0.20

0.24

0.18

0.22

0.16

Zero-AdvantageFiltering

RelativeReward

RelativeReward

0.20

CurriculumLearning

0.16

0.18

0.18

0.17

0.14

0.14

0.13

0.15

0.16

0.12

0.12

0.14

0.14

0.11

0.12

0.11 0.11

0.12

0.10

0.10

False True

False True

WinRate(%)

WinRate(%)

0.08

0.08

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

90%

100

100

70%

70%

70%

60%

###### 50% 50%

40%

30%

30%

30%

50

50

10%

0

0

False True

False True

False True

False True

False True

False True

###### Instruct + 2.5e+18 SFT FLOPs + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | |0.1|8|0.12<br><br>0.1| |7<br><br>0.14| | | |0.16<br><br>0|.14|0.14<br><br>0.| |15| | |0.12|0.12| | | | | | |
| | |0.25 0.5<br><br>| |0.75 1.0| | | | | | | | | | | | | | | |0.10| |0.08| | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | |0.4 0.2<br><br>| | |0.0|0.2|0.4<br><br>| |0.<br><br>|4 0.2<br><br>|0.0| |0.2|0.4| |0.<br><br>|4 0.2<br><br>| |0.0| |0.2| |0.4| |
| | |70%| | |30|%| | |50|%<br><br>30|%| | |20%| |60|%<br><br>40%| | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

0.24

0.23

0.22

0.20

RelativeReward

RelativeReward

0.19

DecodingTemperature

0.20

0.18

0.18

DiscountRate

0.15

0.16

0.16

0.15

0.15

0.16

0.12

0.14

0.14

0.13

0.14

0.13

0.10

0.12 0.11

0.12

0.11

0.08

0.5 0.9

0.95 0.98

0.10

0.05

WinRate(%)

WinRate(%)

0.08

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

80%

100

100

50% 40%

###### 40% 30% 30%

50

50

20%

10%

0

0

0.25 0.5 0.75 1.0

0.25 0.5 0.75 1.0 0.25 0.5 0.75 1.0

0.5 0.9 0.95 0.98

0.5 0.9 0.95 0.98

0.5 0.9 0.95 0.98

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

0.24

0.22

0.22

0.20

0.18

0.20

RelativeReward

RelativeReward

0.18

AdvantageApplication

0.17

0.18

ErrorLogUsage

0.16

0.15

0.15

0.16

0.16

0.16

0.14

0.13

0.14

0.12

0.14

0.12 0.12

0.12

0.12

0.10

0.10

0.10

False True

False True

WinRate(%)

WinRate(%)

0.08

0.08

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

80%

80%

100

100

70%

70%

70%

60%

40%

30%

30%

30%

50

50

20%

20%

0

0

False True

False True

False True

False True

False True

False True

###### Instruct + 2.5e+18 SFT FLOPs + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

0.25

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | |0.| |16|0.18|0.1|7| | |0.14|0.16<br><br>0.1| |5| | | | | | | | | | | |
| | |0.12<br><br>1024| |512| | | | |0.13| | | | | |0.1|0|0.10| |0.12| |0.1|0| | |
| | |256| |64| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.4<br><br>| |0.2<br><br>|0.0| |0.2|0.4|0.<br><br>|4 0.2<br><br>|0.0| |0.2|0.4|0.<br><br>|4 0.2<br><br>| | |0.0100| |%0.2| | |0.4| |
| | | |20|%<br><br>50%| |3|0%| | |10%<br><br>60%| |3|0%| | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

0.24

0.23

ImportanceRatio/TrustRegion

0.22

RelativeReward

RelativeReward

0.20

0.20

0.19

EffectiveBatchSize

0.17

0.18

0.16

0.15

0.16

0.14

0.14

0.12

0.13

0.12

0.12

0.10

0.10

0.10

False True

0.08

WinRate(%)

WinRate(%)

0.08

100%

###### 100%

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

90%

100

100

50

50

10%

0

0

1024 256 512 64

1024 256 512 64 1024 256 512 64

False True

False True

False True

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

Standard-DeviationNormalizedAdvantage

0.20

0.20

0.18

0.18

0.17

0.18

RelativeReward

RelativeReward

0.16

0.15 0.15

0.16

0.15 0.15

0.15

LearningRate

0.12

0.14

0.12

0.13

0.12

0.10

0.12

0.10

0.08

0.06

0.10

0.05

False True

1e-06 5e-07

WinRate(%)

WinRate(%)

0.08

100% 0.4 0.2 0.0 0.2 0.4

100%

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

90%

100

100

70%

60%

60%

40%

40%

30%

50

50

10%

0

0

False True

False True

False True

1e-06 5e-07

1e-06 5e-07

1e-06 5e-07

Figure 3 Bootstrap analysis (n = 1000 samples) of hyperparameter optimization across different SFT compute budgets on training held out tasks. Each subplot examines a different hyperparameter, including increasing SFT compute: the base instruct model (left), +2.5×1018 SFT FLOPs (middle), and +7.6e×1018 SFT FLOPs (right). For each hyperparameter-compute combination, the top panel shows relative reward performance with error bars indicating 95% confidence intervals, while the bottom panel displays win rates representing the percentage of bootstrap iterations where each parameter value achieved maximum performance. Results demonstrate that optimal hyperparameter values shift as model pre-training compute increases, suggesting that hyperparameter selection should be adapted to the computational budget allocated to SFT.

and two warm-started variants with an additional 2.5 × 1018 and 7.6 × 1018 FLOPs of supervised fine-tuning, respectively, to assess variations across compute budgets. We evaluate the held-out goals and held-out tasks performance of 10 hyper-parameters across 1,370 runs. We report final held-out task performance to verify generalization in Section G finding no significant deviations between held-out tasks and held-out goals parameters.

- Figure 3 displays our findings, which we summarize as follows. Curriculum learning is beneficial when starting RL from scratch but becomes detrimental after warm-starting, likely because warm-started models already perform well on easy tasks, and curriculum forces them to overfocus on harder ones. Error log feedback helps when there’s no SFT but otherwise does not, likely because SFT warmup removes many common errors made by weaker models. A decoding temperature of 0.25 consistently yields the best results, striking a balance between exploration and exploitation; lower values led to under-exploration and were discarded. Grouped-relative advantage helps only after SFT, while using raw rewards works better when starting directly from the Instruct model, possibly due to how advantage scaling interacts with the initial weights. Zeroadvantage filtering improves training across most settings by ensuring batches focus on informative updates. Standard-deviation normalized advantage, as noted by (Liu et al., 2025), seems to aid performance under less finetuned models and decrease in value the more finetuning is done. Importance ratio correction and trust region, though standard, also hurt models with little or no SFT, likely because conservative updates slow down learning. In contrast, for models that start from a stronger SFT checkpoint, these mechanisms can help stabilize training and avoid catastrophic updates. For the learning rate, the larger value (1e-6) generally worked better. Effective batch size of 512 appears to be generally be a safe and robust choice in all our experiments. Finally, regarding the discount rate, values above 0.9 work well for models with little or no SFT warmup, while heavily warm-started models benefit from a lower rate around 0.5 likely because it encourages the agent to optimize more aggressively on tasks it already handles well. A further hypothesis is that the large deviation in optimal hyperparameters between the base and warm-start models could be attributed to the significant entropy reduction introduced by SFT (see Section H).

## 7 Related Work

Best prectices in deep RL. Building on the recognition of reproducibility challenges and unstable RL training of LLM agents, recent studies have proposed best practices for training LLM agents using RL methods. Dang & Ngo (2025) recommend leveraging high quality data, balancing easy and hard problems, and controlling length generation with cosine reward. Yu et al. (2025) promote higher clipping in the GRPO loss to promote diversity and avoid entropy collapse, dynamic sampling to improve training efficiency and stability, token level gradients for long CoT sequences, and overlong reward shaping to reduce reward noise. Roux et al. (2025) introduce tapered variant of importance sampling to speed up learning while maintaining stable learning dynamics. The proposed method (TOPR) allows the handling of both positive and negative examples in a fully offline setting. More generally, Hochlehnert et al. (2025) emphasizes the need for greater methodological precision, particularly concerning decoding parameters, random seeds, prompt formatting, as well as the hardware and software frameworks, to guarantee transparent and thorough assessments of model performance. These practices are essential for developing robust and reproducible agents.

LLM Agents trained with RL on multi-step environments. Recent advancements have sought to bridge the gap in training LLM agents for multi-step environments, with approaches like WebRL (Qi et al., 2025) and SWEET-RL (Zhou et al., 2025) demonstrating significant progress. WebRL employs a self-evolving curriculum to address the challenges of sparse feedback and task scarcity, successfully enhancing the performance of open LLMs in web-based tasks (Qi et al., 2025). Similarly, SWEET-RL introduces a hierarchical structure that enables effective credit assignment over multiple turns, improving policy learning and generalization in collaborative reasoning tasks (Zhou et al., 2025). These studies collectively illustrate the necessity of adapting RL techniques to accommodate the complexities of multi-step interactions, paving the way for more capable and versatile LLM agents. Similar in spirit to our work, (Andreux et al., 2025) propose an empiral study on the inference cost of trained LLM web agents. An extended version of the related work is provided in Section F.

LLM agents trained with RL in multi-step environments. Recent work has begun closing the gap for LLM agents that must reason over multiple steps. WebRL (Qi et al., 2025) introduces a self-evolving curriculum that tackles sparse feedback and task scarcity, substantially boosting open-source agents on web tasks. SWEET-RL (Zhou et al., 2025) adds a hierarchical credit-assignment scheme spanning many turns, improving both learning stability and generalization in collaborative reasoning. Together, these studies underscore the need to adapt RL techniques to the intricacies of long-horizon interaction, paving the way for more capable, versatile agents. Similar in spirit, (Andreux et al., 2025) provide an empirical analysis of inference costs for trained LLM web agents. An extended discussion of related work appears in Section F.

## 8 Discussion

Limitations. Our focus is on providing a comprehensive perspective on training an LLM-based web agent, studying compute trade-offs, hyperparameter selection, and analyzing failure cases. With this in mind, our results are limited to English-language web interfaces and Llama 3 models in the 8B–70B parameter range, where larger models may alter trade-offs.

Regarding our statistical method, it does not account for the lack of coverage from the random search. A more exhaustive search could discover configurations that would change the conclusions drawn in this study. We note also that a significant portion of the reported uncertainty is due to epistemic uncertainty that could be reduced by evaluating more configurations.

Conclusion. We present a statistically grounded study on training LLM web agents, analyzing the tradeoff between SFT and RL. We perform a random sweep across 1,370 configurations to identify optimal hyperparameter choices, finding interesting variation across compute budgets. Using the bootstrap prescribed hyper-parameters we show that branching into RL early, but not immediately, after SFT achieves better performance–compute trade-offs, matching peak SFT with 45% less compute and closing the gap with closed-source agents. Our findings offer a reproducible, budget-aware blueprint for advancing open-source LLM web agents in complex multi-step environments.

## References

Dgxc benchmarking. URL https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/ llama31-8b-dgxc-benchmarking-a.

Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, Piero Kauffmann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, and Guoqing Zheng. Phi-4-reasoning technical report, 2025. URL https://arxiv.org/abs/2504.21318.

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C Courville, and Marc Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Advances in neural information processing systems, 34:29304–29320, 2021.

Mathieu Andreux, Breno Baldas Skuk, Hamza Benchekroun, Emilien Biré, Antoine Bonnet, Riaz Bordie, Matthias Brunel, Pierre-Louis Cedoz, Antoine Chassang, Mickaël Chen, et al. Surfer-h meets holo1: Cost-efficient web agent powered by open weights. arXiv preprint arXiv:2506.02865, 2025.

Léo Boisvert, Megh Thakkar, Maxime Gasse, Massimo Caccia, Thibault Le Sellier De Chezelles, Quentin Cappart, Nicolas Chapados, Alexandre Lacoste, and Alexandre Drouin. Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks, 2024. URL https://arxiv.org/abs/2407.05291.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training, 2025. URL https://arxiv.org/abs/2501.17161.

Quy-Anh Dang and Chris Ngo. Reinforcement learning for reasoning in small llms: What works and what doesn’t. arXiv preprint arXiv:2503.16219, 2025. URL https://arxiv.org/abs/2503.16219.

Thibault Le Sellier de Chezelles, Maxime Gasse, Alexandre Lacoste, Massimo Caccia, Alexandre Drouin, Léo Boisvert, Megh Thakkar, Tom Marty, Rim Assouel, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F. Xu, Siva Reddy, Graham Neubig, Quentin Cappart, Russ Salakhutdinov, and Nicolas Chapados. The browsergym ecosystem for web agent research. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=5298fKGmv3. Expert Certification.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. URL https://arxiv.org/abs/2501.12948.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks?, 2024.

B. Efron. Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1):1–26, 1979. Bradley Efron and R.J. Tibshirani. An Introduction to the Bootstrap. Chapman and Hall/CRC, May 1994. ISBN

9780429246593. doi: 10.1201/9780429246593. URL http://dx.doi.org/10.1201/9780429246593.

Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, and Matthias Bethge. A sober look at progress in language model reasoning: Pitfalls and paths to reproducibility, 2025. URL https: //arxiv.org/abs/2504.07086.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, March 1963. ISSN 1537-274X. doi: 10.1080/01621459.1963.10500830. URL http: //dx.doi.org/10.1080/01621459.1963.10500830.

Jing Yu Koh and et al. Visualwebarena: Evaluating multimodal agents on realistic visually grounded web tasks. In ACL 2024, 2024. URL https://aclanthology.org/2024.acl-long.50.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. In International Conference on Learning Representations (ICLR),

2018. URL https://arxiv.org/abs/1802.08802. Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025. URL https://arxiv.org/abs/2503.20783. Shikhar Murty and et al. Nnetnav: Unsupervised learning of browser agents through environment interaction in the wild. arXiv preprint arXiv:2410.02907, 2025. URL https://arxiv.org/abs/2410.02907.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, Jie Tang, and Yuxiao Dong. WebRL: Training LLM web agents via self-evolving online curriculum reinforcement learning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=oVKEAFjEqv.

Nicolas Le Roux, Marc G. Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alex Fréchette, Carolyne Pelletier, Eric Thibodeau-Laufer, Sándor Toth, and Sam Work. Tapered off-policy reinforce: Stable and efficient reinforcement learning for llms. arXiv preprint arXiv:2503.14286, 2025. URL https://arxiv.org/abs/2503.14286.

Richard S. Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Proceedings of the 13th International Conference on Neural Information Processing Systems, NIPS’99, pp. 1057–1063, Cambridge, MA, USA, 1999. MIT Press.

Megh Thakkar, Tolga Bolukbasi, Sriram Ganapathy, Shikhar Vashishth, Sarath Chandar, and Partha Talukdar. Self-influence guided data reweighting for language model pre-training. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview.net/forum?id=rXn9WO4M2p.

torchtune maintainers and contributors. torchtune: Pytorch’s finetuning library, April 2024. URL https//github.com/ pytorch/torchtune.

Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge University Press, February 2019. ISBN 9781108498029. doi: 10.1017/9781108627771. URL http://dx.doi.org/10.1017/9781108627771.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/ 2201.11903.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024.

Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, and Graham Neubig. Theagentcompany: Benchmarking llm agents on consequential real world tasks, 2024. URL https://arxiv.org/abs/2412.14161.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. URL https://arxiv.org/abs/2503.14476.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023. URL https://arxiv.org/abs/2307.13854.

Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, and Xian Li. Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks, 2025. URL https://arxiv.org/abs/2503.15478.

- A Compute-peformance frontier for Qwen2.5 7B
- B Compute Infrastructure

Our computational infrastructure comprises 8 × H100-80GB GPUs for expert data generation with the 70B model. For student model training, we allocate 2 × H100 GPUs for MiniWoB++ experiments and 4 × H100 GPUs for WorkArena experiments, reflecting the increased complexity of the latter.

- C Extended Learning and Saturation Analysis

Challenges in Agent-Environment Interaction In this section we talk about the general challenges faced by the agent to interact effectively with the environment.

• Observation/action space mismatch: One of the important thing to note specifically in our web environment is the observation space which the agent uses is a bit different from the action space. Multiple times, the agent can see the correct action in the AxTree but the action space, the icon is not

[Figure 7]

###### Figure 4 Compute–performance frontier on MiniWoB++ (results averaged over two seeds) for Qwen2.5 7B. The blue curve shows pure SFT on teacher demonstrations. Warm-colored curves represent hybrid runs that branch off from SFT checkpoints and continue training with RL. Early transitions to RL push the Pareto frontier achieving higher success rates for the same compute and is the only approach able to achieve over 30% improvement on both held-out goals (left) and held-out tasks (right).

SFT

Approach 0: Continued SFT

Approach 1: On-Policy RL

LLaMA 70B

choose-list

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

click-checkboxes-soft click-button-sequence

copy-paste-2 count-shape click-widget

click-color click-collapsible-nodelay

click-collapsible-2-nodelay click-collapsible-2 click-checkboxes-transfer click-dialog-2 click-dialog

click-link click-tab-2-medium

click-tab click-tab-2 click-tab-2-hard click-menu-2

click-test-2 click-test-transfer

click-test email-inbox-star-reply enter-date enter-text enter-text-dynamic enter-time drag-sort-numbers draw-circle draw-line

email-inbox email-inbox-delete

email-inbox-forward email-inbox-forward-nl email-inbox-forward-nl-turk use-spinner

social-media-all visual-addition text-transform

social-media-some simple-algebra sign-agreement

scroll-text-2 social-media

order-food

navigate-tree multi-orderings

login-user-popup email-inbox-reply

email-inbox-noscroll email-inbox-important

1.0

find-greatest

[Figure 8]

find-word focus-text-2

generate-number

read-table phone-book

0.8

scroll-text read-table-2 odd-or-even

0.6

multi-layouts identify-shape login-user

Task

ascending-numbers circle-center bisect-angle

0.4

book-flight

count-sides copy-paste click-menu

0.2

click-scroll-list click-shape

choose-date-easy choose-date-medium choose-date-nodelay

0.0

choose-date drag-shapes

drag-items-grid drag-items drag-shapes-2 drag-circle

drag-box daily-calendar guess-number

grid-coordinate form-sequence-2

form-sequence drag-cube resize-textarea highlight-text

hot-cold right-angle

search-engine text-editor tic-tac-toe

use-autocomplete-nodelay use-autocomplete use-colorwheel-2 use-colorwheel

buy-ticket click-button

click-checkboxes click-collapsible click-option

click-tab-2-easy click-shades unicode-test

simple-arithmetic enter-password

focus-text form-sequence-3

email-inbox-nl-turk enter-text-2 click-checkboxes-large click-pie

find-midpoint click-pie-nodelay

drag-single-shape highlight-text-2 stock-market

use-slider use-slider-2

Average Train Average Test

0 2 Epoch

5 7 10 14 17 20 Epoch

10 15 20

Epoch

###### Figure 5 Per task performance of SFT and SFT+RL agents on MiniWob++.

SFT

Approach 0: Continued SFT

Approach 1: On-Policy RL

LLaMA 70B

knowledge-base-search order-apple-mac-book-pro15

0.38 0.37 0.24 0.81 0.78 0.72

|0.00| |0.35| |
|---|---|---|---|
|0.25| |0.47| |
|0.00| |0.22| |
|0.00| |0.26| |
|0.50| |0.48| |
|0.00| |0.48| |
|0.00| |0.05| |
|0.00| |0.00| |
|0.00| |0.08| |
|0.00| |0.06| |
|0.25| |0.25| |
|0.25| |0.23| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.17| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.00| |
|0.00| |0.31| |
|0.07| |0.21| |
|0.00| |0.00| |
|0.29| |0.29| |
|0.00| |0.05| |
|0.00| |0.11| |
|0.00| |0.02| |
|0.03| |0.08| |
|0.05| |0.14| |
| | | | |

|0.77| |0.55| |0.48| |
|---|---|---|---|---|---|
|1.00| |0.86| |0.85| |
|0.75| |0.71| |0.45| |
|0.55| |0.65| |0.73| |
|0.98| |0.92| |0.79| |
|0.65| |0.62| |0.81| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.03| |0.02| |0.12| |
|0.10| |0.01| |0.00| |
|0.25| |0.25| |0.26| |
|0.25| |0.25| |0.25| |
|0.26| |0.14| |0.20| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.12| |0.23| |0.25| |
|0.12| |0.16| |0.14| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.00| |0.00| |
|0.00| |0.07| |0.06| |
|0.69| |0.57| |0.51| |
|0.30| |0.38| |0.39| |
|0.04| |0.04| |0.08| |
|0.32| |0.39| |0.31| |
|0.37| |0.41| |0.38| |
|0.33| |0.42| |0.50| |
|0.00| |0.00| |0.00| |
|0.15| |0.14| |0.13| |
|0.29| |0.31| |0.31| |
| | | | | | |

| |1.00|
|---|---|
| | |
| |1.00|
| | |
| |1.00|
| | |
| |1.00|
| | |
| |1.00|
| | |
| |0.75|
| | |
| |0.33|
| | |
| |0.33|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.25|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.00|
| | |
| |0.80|
| | |
| |0.71|
| | |
| |0.60|
| | |
| |0.57|
| | |
| |0.43|
| | |
| |0.36|
| | |
| |0.00|
| | |
| |0.19|
| | |
| |0.50|
| | |

order-ipad-mini order-ipad-pro order-apple-watch order-development-laptop-p-c multi-chart-value-retrieval single-chart-value-retrieval create-incident all-menu

- 0.51 0.67 0.44 0.61 0.51 0.57 0.73 0.80 0.82 0.67 0.67 0.71 0.00 0.00 0.05 0.00 0.00 0.00 0.16 0.05 0.18

- 0.13 0.18 0.19 0.25 0.25 0.25

- 0.25 0.25 0.25 0.16 0.44 0.19 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.27 0.22 0.28 0.07 0.06 0.06 0.00 0.00 0.00 0.00 0.00 0.00

- 0.00 0.00 0.00 0.02 0.00 0.00

0.52 0.57 0.47 0.14 0.17 0.17

- 0.01 0.04 0.02 0.41 0.45 0.44 0.35 0.55 0.48 0.40 0.62 0.58 0.00 0.00 0.00 0.13 0.13 0.12

- 0.26 0.34 0.31

1.0

[Figure 9]

sort-incident-list sort-hardware-list order-sales-laptop

0.8

filter-incident-list

filter-asset-list filter-change-request-list

0.6

create-hardware-asset filter-hardware-list filter-user-list filter-service-catalog-item-list order-developer-laptop single-chart-min-max-retrieval sort-change-request-list sort-asset-list sort-service-catalog-item-list

Task

0.4

0.2

sort-user-list order-loaner-laptop

create-user create-change-request

0.0

multi-chart-min-max-retrieval order-standard-laptop create-problem

impersonation Average Train

Average Test

0 4 Epoch

10 15 20 Epoch

10 15 20 Epoch

- Figure 6 Per task performance of SFT and SFT+RL agents on WorkArena.

visible and to make it visible, the agent needs to scroll down and do the action. This mismatch causes huge problems Koh & et al. (2024)

- • UI Misuse: The agent tries to interact with items in the environment in ways that it is not designed. For example, the agent trying to fill in a checkbox with value True while it should just click on it. Murty & et al. (2025)
- • Repeating actions: A common issue we observed is the repetition of actions across multiple consecutive steps, often accompanied by verbose and redundant chains of thought. The agent frequently restates similar thoughts or re-executes the same actions unnecessarily, leading to inefficiencies and sometimes getting stuck in loops. (Murty & et al., 2025).

## D Deriving Compute Cost

FLOPs Estimation Methodology Flop calculations are based on model architecture, token counts, and average sequence lengths observed during training and evaluation.

FLOPs per Token We estimate FLOPs per token using the following formula, adapted from nvidia benchmarkingben:

FLOPsper token = (FLOPsattn + FLOPsMLP + FLOPsembed) × (1 + backward multiplier) (6) Where:

FLOPsattn = 12 × (number of layers) × (hidden size)2 × 1 +

number of query groups number of attention heads

+

sequence length hidden size

(7)

FLOPsMLP = 18 × (number of layers) × (hidden size) × FFN (8) FLOPsembed = 6 × vocabulary size × (hidden size) (9)

On-Policy FLOPs (LLaMA-8B) We compute the total FLOPs for each on-policy epoch by summing the training and testing FLOPs:

FLOPstrain = Ntrain × FLOPs(perbackwardtoken =3) (10)

FLOPstest = Ntest × FLOPs(perbackwardtoken =0) (11) FLOPsepoch = FLOPstrain + FLOPstest (12)

Where Ntrain and Ntest are the number of tokens used for training and evaluation respectively. Sequence length S is measured per epoch from logged metrics.

Offline FLOPs (Generation: LLaMA-70B, Training: LLaMA-8B) Offline training includes two compute components:

- • Data Generation (LLaMA-70B, forward-only):

FLOPsgen = Ngen × FLOPs(70perB,tokenbackward=0) (13) where Ngen = avg seq len × samples per epoch (from dataset metadata).

- • Training (LLaMA-8B, with backward pass):

FLOPstrain = Ngen × FLOPs(8perB,tokenbackward=3) (14)

The total FLOPs per offline epoch is: FLOPsepoch = FLOPsgen + FLOPstrain (15) All FLOPs values are reported in exaFLOPs by dividing the total FLOPs by 1018.

## E Benchmark Descriptions

[Figure 10]

Figure 7 Example tasks in MiniWoB++ Liu et al. (2018) (top) and WorkArena Drouin et al. (2024) (bottom). MiniWoB consists of single-page simple tasks such as selecting a particular date and using a basic text editor, while WorkArena comprises multi-page complex tasks like filling forms and placing orders in an enterprise environment.

## F Extended Related Work

The Reproducibility Crisis in RL. The reproducibility crisis in large language models (LLMs) and reinforcement learning (RL) has garnered increasing attention, particularly due to the reliance on single seed results that distort the perceived performance of models. The reproducibility challenge1 organized every year is a positive step towards addressing this. More concretely, Hochlehnert et al. (2025) provide a critical examination of how such practices undermine the reliability of published findings, revealing that many reported gains are sensitive to implementation choices, such as random seeds and prompt formatting Hochlehnert et al. (2025).

1https://reproml.org/

Bandit-domain RLHF with LLMs. Previous work in RL for LLMs has predominantly focused on single-step tasks, which have shown effectiveness in mathematical reasoning and code generation Yu et al. (2025); DeepSeek-AI et al. (2025); Roux et al. (2025). While these approaches exhibit promising results, they are limited in their applicability to real-world scenarios, which often require multistep decision-making capabilities. The narrow focus on bandit-style problems fails to address the complexities inherent in tasks that demand sequential interaction, highlighting a significant gap in the current research landscape.

Interactive Agent Benchmarks. To assess the capabilities of LLM agents in more realistic environments, benchmarks such as WebArena Zhou et al. (2023), WorkArena Drouin et al. (2024); Boisvert et al. (2024), the Agent Company Xu et al. (2024), and OSWorld Xie et al. (2024) have been designed to evaluate agents on multi-step tasks across various domains. These benchmarks expose the limitations of current LLM agents, revealing that while they may perform well in controlled settings, their performance in practical applications remains subpar, underscoring the need for further advancements in agent robustness and generalization to multi-step planning.

Best practices in deep RL. Building on the recognition of reproducibility challenges and unstable RL training of LLM agents, recent studies have proposed best practices for training LLM agents using RL methods. Dang & Ngo (2025) recommend leveraging high quality data, balancing easy and hard problems, and controlling length generation with cosine reward. Yu et al. (2025) promote higher clipping in the GRPO loss to promote diversity and avoid entropy collapse, dynamic sampling to improve training efficiency and stability, token level gradients for long CoT sequences, and overlong reward shaping to reduce reward noise. Roux et al. (2025) introduce tapered variant of importance sampling to speed up learning while maintaining stable learning dynamics. The proposed method (TOPR) allows the handling of both positive and negative examples in a fully offline setting. More generally, Hochlehnert et al. (2025) emphasizes the need for greater methodological precision, particularly concerning decoding parameters, random seeds, prompt formatting, as well as the hardware and software frameworks, to guarantee transparent and thorough assessments of model performance.

LLM Agents trained with RL on multi-step environments. Recent advancements have sought to bridge the gap in training LLM agents for multi-step environments, with approaches like WebRL Qi et al. (2025) and SWEET-RL Zhou et al. (2025) demonstrating significant progress. WebRL employs a self-evolving curriculum to address the challenges of sparse feedback and task scarcity, successfully enhancing the performance of open LLMs in web-based tasks Qi et al. (2025). Similarly, SWEET-RL introduces a hierarchical structure that enables effective credit assignment over multiple turns, improving policy learning and generalization in collaborative reasoning tasks Zhou et al. (2025). These studies collectively illustrate the necessity of adapting RL techniques to accommodate the complexities of multi-step interactions, paving the way for more capable and versatile LLM agents.

## G Test Set Hyper-Parameter Bootstrap Analysis

We overall find similar results between the held-out train and test tasks with respect to optimal hyperparameters. While we see no large deviations, we find that some parameters such as curriculum learning from the instruct model and using error logs can have a larger beneficial effect on the held-out testing tasks.

## H Entropy

Figure 9 reports log-entropy as a function of training compute (FLOPs) across the training run.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

1.0

0.5

0.0

LogEntropy

0.5

1.0

1.5

2.0

2.5

3.0

0e18 5e18 10e18

Compute (FLOPs)

Figure 9 relationship between log-entropy and compute allocated to SFT warm-starting. We observe an initial sharp drop in log-entropy, followed by a period of stabilization and gradual decline. We speculate that this reduction in entropy, which makes rollouts more deterministic, may explain the difference in optimal hyperparameters between the base and warm-started checkpoints.

## I Random Search Space

We conduct a random hyperparameter sweep over 1,370 training runs over the following parameter configurations:

- • Decoding temperature (ρLLM): Sampled from {0.1, 0.25, 0.5, 0.75, 1}
- • Curriculum learning: Enabled or disabled (True, False)
- • Curriculum mean (µtarget): {0.25, 0.5, 0.75}
- • Curriculum Temperature (ρCurr): {0.1, 0.3}
- • Discount rate: {0.5, 0.8, 0.9, 0.95, 0.98, 1.0}
- • Grouped-relative advantage: Enabled or disabled
- • Zero-advantage filtering: Enabled or disabled
- • Standard-deviation normalized advantage: Enabled or disabled
- • Effective batch size: {64, 256, 512, 1024}
- • Learning rate: {1e-6, 5e-6, 5e-7}
- • Error log feedback: Enabled or disabled

- • Importance ratio: Enabled or disabled

## J Compute Allocation Hyperparameter Selection

For consistency, we select a single set of hyperparameters for all runs in Figure 1. To do so, we use the bootstrap algorithm described in Algorithm 1 where we use all the runs over all SFT checkpoints as input into Results. We then analyze the results of this bootstrap and use them to obtain the reported results.

The optimal parameters given by this aggregate bootstrap are:

- • Decoding temperature ρLLM: 0.25
- • Curriculum learning: False
- • Discount rate: 0.90
- • Grouped-relative advantage: True
- • Zero-advantage filtering: True
- • Standard-deviation normalized advantage: True
- • Effective batch size: 512
- • Learning rate: 1e-6
- • Error log feedback: True
- • Importance Ratio: False

## K Correctness of the Bootsrap

One of the main motivations for using the boostrap algorithm for our hyper-parameter evaluation is the vast amount of literature showing its desrable properties for estimating the sampling distribution of data statistics.

The most important attribute for the robustness of the boostrap is its ability to exploit the Central Limit Theorem (CLT) to approximate the sampling distribution of a statistic. Using the CLT it can allow us to estimate its expectation and variability without parametric assumptions (Efron & Tibshirani, 1994).

Specifically, in our use case, we want to evaluate, for each hyperparameter value h, how well it performs when paired with the best possible values of the other hyperparameters. The bootstrap algorithm yields an unbiased estimate for a target statistic given an initial n samples (full runs in our case). Thus, we choose the statistic of interest to be the evaluation score M conditioned on the hyper-parameter of interst h assuming all other hyper-parameters g are optimal:

Th(D) = max{M(h′,g) : (h′,g) ∈ D, h′ = h} (16)

Bootstrapping this yields an empirical distribution over Th values, providing an unbiased estimate and confidence intervals for the maximum score achievable for h when other hyperparameters are optimal.

Additionally, there exists results showing the rate of convergence and consistency of the bootstrap algorithm which we directly exploit. It is a well known result that as the number of training runs n → ∞, the bootstrap estimate Tˆh of the maximum achievable score for hyperparameter value h becomes consistent. That is:

Tˆh −→p Th = sup{M(h,g) : g ∈ G} (17)

This follows from the law of large numbers (Wainwright, 2019), assuming sufficient coverage of other hyperparameters G. The rate of convergence depends on the tail behavior of the conditional distribution M | H = h. Since our evaluation metric is bounded in [0,1], the sub-Gaussian assumption holds automatically

by Hoeffding’s Lemma (Hoeffding, 1963), ensuring concentration of the estimate. Under these conditions, the expected bias decays as O(1/nh), where nh is the number of runs with H = h. These are well-known results, which is why the bootstrap is an extremely useful and simple algorithm for estimating any statistic of interest (Efron, 1979; Efron & Tibshirani, 1994).

## L Train and Test Splits Used in the Experiments

We evaluate generalization by training only on the train split and reporting performance on held-out tasks from the test split. For the held-out goals metric, we instantiate goal variations using seed ranges [0,2] de Chezelles et al. (2025) for training tasks (3 goals per task) and [0,9] de Chezelles et al. (2025) for test tasks (10 goals per task). Below we list the exact task identifiers used in our experiments for both benchmarks. Task names are the registry keys from the respective environments.

WorkArena (Train: 24 tasks, Test: 7 tasks)

Train:

- • workarena.servicenow.all-menu
- • workarena.servicenow.create-hardware-asset
- • workarena.servicenow.create-incident
- • workarena.servicenow.filter-asset-list
- • workarena.servicenow.filter-change-request-list
- • workarena.servicenow.filter-hardware-list
- • workarena.servicenow.filter-service-catalog-item-list
- • workarena.servicenow.filter-user-list
- • workarena.servicenow.knowledge-base-search
- • workarena.servicenow.order-apple-mac-book-pro15
- • workarena.servicenow.order-development-laptop-p-c
- • workarena.servicenow.order-ipad-mini
- • workarena.servicenow.order-ipad-pro
- • workarena.servicenow.order-sales-laptop
- • workarena.servicenow.sort-change-request-list
- • workarena.servicenow.sort-hardware-list
- • workarena.servicenow.sort-incident-list
- • workarena.servicenow.sort-service-catalog-item-list
- • workarena.servicenow.sort-user-list
- • workarena.servicenow.single-chart-value-retrieval
- • workarena.servicenow.create-change-request
- • workarena.servicenow.order-loaner-laptop
- • workarena.servicenow.order-standard-laptop
- • workarena.servicenow.create-problem

##### Test:

- • workarena.servicenow.create-user
- • workarena.servicenow.filter-incident-list
- • workarena.servicenow.sort-asset-list
- • workarena.servicenow.impersonation
- • workarena.servicenow.order-apple-watch
- • workarena.servicenow.order-developer-laptop
- • workarena.servicenow.single-chart-min-max-retrieval

MiniWoB++ (Train: 99 tasks, Test: 23 tasks)

Train:

- • miniwob.ascending-numbers
- • miniwob.bisect-angle
- • miniwob.book-flight
- • miniwob.choose-date
- • miniwob.choose-date-easy
- • miniwob.choose-date-medium
- • miniwob.choose-date-nodelay
- • miniwob.choose-list
- • miniwob.circle-center
- • miniwob.click-button-sequence
- • miniwob.click-checkboxes-soft
- • miniwob.click-checkboxes-transfer
- • miniwob.click-collapsible-2
- • miniwob.click-collapsible-2-nodelay
- • miniwob.click-collapsible-nodelay
- • miniwob.click-color
- • miniwob.click-dialog
- • miniwob.click-dialog-2
- • miniwob.click-link
- • miniwob.click-menu
- • miniwob.click-menu-2
- • miniwob.click-scroll-list
- • miniwob.click-shape
- • miniwob.click-tab
- • miniwob.click-tab-2
- • miniwob.click-tab-2-hard
- • miniwob.click-tab-2-medium

- • miniwob.click-test
- • miniwob.click-test-2
- • miniwob.click-test-transfer
- • miniwob.click-widget
- • miniwob.copy-paste
- • miniwob.copy-paste-2
- • miniwob.count-shape
- • miniwob.count-sides
- • miniwob.daily-calendar
- • miniwob.drag-box
- • miniwob.drag-circle
- • miniwob.drag-cube
- • miniwob.drag-items
- • miniwob.drag-items-grid
- • miniwob.drag-shapes
- • miniwob.drag-shapes-2
- • miniwob.drag-sort-numbers
- • miniwob.draw-circle
- • miniwob.draw-line
- • miniwob.email-inbox
- • miniwob.email-inbox-delete
- • miniwob.email-inbox-forward
- • miniwob.email-inbox-forward-nl
- • miniwob.email-inbox-forward-nl-turk
- • miniwob.email-inbox-important
- • miniwob.email-inbox-noscroll
- • miniwob.email-inbox-reply
- • miniwob.email-inbox-star-reply
- • miniwob.enter-date
- • miniwob.enter-text
- • miniwob.enter-text-dynamic
- • miniwob.enter-time
- • miniwob.find-greatest
- • miniwob.find-word
- • miniwob.focus-text-2
- • miniwob.form-sequence
- • miniwob.form-sequence-2

- • miniwob.generate-number
- • miniwob.grid-coordinate
- • miniwob.guess-number
- • miniwob.highlight-text
- • miniwob.hot-cold
- • miniwob.identify-shape
- • miniwob.login-user
- • miniwob.login-user-popup
- • miniwob.multi-layouts
- • miniwob.multi-orderings
- • miniwob.navigate-tree
- • miniwob.odd-or-even
- • miniwob.order-food
- • miniwob.phone-book
- • miniwob.read-table
- • miniwob.read-table-2
- • miniwob.resize-textarea
- • miniwob.right-angle
- • miniwob.scroll-text
- • miniwob.scroll-text-2
- • miniwob.search-engine
- • miniwob.sign-agreement
- • miniwob.simple-algebra
- • miniwob.social-media
- • miniwob.social-media-all
- • miniwob.social-media-some
- • miniwob.text-editor
- • miniwob.text-transform
- • miniwob.tic-tac-toe
- • miniwob.use-autocomplete
- • miniwob.use-autocomplete-nodelay
- • miniwob.use-colorwheel
- • miniwob.use-colorwheel-2
- • miniwob.use-spinner
- • miniwob.visual-addition

##### Test:

- • miniwob.buy-ticket

- • miniwob.click-button
- • miniwob.click-option
- • miniwob.click-pie-nodelay
- • miniwob.drag-single-shape
- • miniwob.email-inbox-nl-turk
- • miniwob.enter-text-2
- • miniwob.find-midpoint
- • miniwob.focus-text
- • miniwob.simple-arithmetic
- • miniwob.stock-market
- • miniwob.use-slider-2
- • miniwob.click-checkboxes
- • miniwob.click-checkboxes-large
- • miniwob.click-collapsible
- • miniwob.click-pie
- • miniwob.click-shades
- • miniwob.click-tab-2-easy
- • miniwob.enter-password
- • miniwob.form-sequence-3
- • miniwob.highlight-text-2
- • miniwob.unicode-test
- • miniwob.use-slider

###### Instruct + 2.5e+18 SFT FLOPs + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

0.20

0.20

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |0.1|6| | |0.14| | | |0.13| | | |0.13| | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |0.04| | | |0.0|6| |
| | |False| |True| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.4100<br><br>| |%0.2 0.0 0.2<br><br>| | | |0.4| |0.4<br><br>| |0.2 0.0 0.2<br><br>| | | |0.4<br><br>| |0.4| |0.2 0.0 0.290<br><br>| | | |%0.4| |
| | | | | | | | | |60| |%<br><br>40| | | |%| | | | | | | | | |
| | | | | | | | | | | | | | | | | |10| |%| | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

0.18

0.18

0.16

0.15

Zero-AdvantageFiltering

RelativeReward

RelativeReward

0.15

0.14

0.15

CurriculumLearning

0.12

0.12

0.12

0.10

0.10

0.08

0.08

0.05 0.05

0.05

0.05

False True

0.03

0.03

WinRate(%)

WinRate(%)

100%

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

100

100

70%

60%

40%

30%

50

50

0

0

False True

False True False True

False True

False True

False True

###### Instruct + 2.5e+18 SFT FLOPs + 7.6e+18 SFT FLOPs

###### Instruct + 2.5e+18 SFT FLOPs + 7.6e+18 SFT FLOPs

0.20

| | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | |0.16| | | | | | |0.14|0.16| |0.12| |0.14| | | | | | | | |
| | |0.1|1|0.08| | | |0.0|8<br><br>| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |0.06<br><br>|0.04|0.05|0.0|5| | |
| | |0.25<br><br>0.5<br><br>| |0.75 1.0| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | |0.4 0.2<br><br>| | |0.0100| |%0.2| | |0.4| |0.<br><br>|4 0.290%<br><br>| |0.0| |0.2| |0.4 0.4<br><br>| |0.2 0.0 0| |.2| |0.4| |
| | | | | | | | | | | | | | | | | |10| |%<br><br>30%| |20% 10%| |40| |%| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | |0.12| | | | | | | | | |0.11|0.13| |0.14| |0.13| | | | | | | | | |
| | | |0.09| |0.08| |0.0|8| | | | | | | | | | |0|.06|0.0|6|0.| |05| |
| | |0.5 0.9<br><br>|0.95 0.98<br><br>| | | | | | | | | | | | | | | | | | | |0.03| | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | |90%0.4 0.2<br><br>|0.0| | |0.2| | |0.4| |0.<br><br>|4 0.2<br><br>| |0.0| |0.2| |0.4 0.4<br><br>| |0.2| |0.0| |0.2|0.4| |
| | | |10%| | | | | | | | |40| |%<br><br>60%| | | |30| |%<br><br>60%| | | | |10%| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.18

0.18

0.16

0.15

RelativeReward

RelativeReward

0.14

DecodingTemperature

0.12

0.12

DiscountRate

0.10

0.10

0.08

0.08

0.06

0.05

0.04

0.03

0.02

WinRate(%)

WinRate(%)

0.00

100

100

50

50

0

0

0.25 0.5 0.75 1.0

0.25 0.5 0.75 1.0 0.25 0.5 0.75 1.0

0.5 0.9 0.95 0.98

0.5 0.9 0.95 0.98 0.5 0.9 0.95 0.98

###### Instruct + 2.5e+18 SFT FLOPs + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

0.20

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | |0.16| | | |0.15| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |0.13| | | | | | | | | | |
| | |0.1|2| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |0.05| | | |0.05| | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | |False| |True| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.4<br><br>| |0.2 0.0 0.2100<br><br>| | | |%0.4| |0.490<br><br>| |%0.2 0.0 0.2<br><br>| | | |0.4| |0.4<br><br>| |0.2 0.0 0.2<br><br>| | | |0.4| |
| | | | | | | | | | | | | | | | | |40| |%<br><br>60| | | |%| |
| | | | | | | | | | | |10| | | |%| | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

0.18

0.16

0.18

0.16

0.15

RelativeReward

RelativeReward

0.15

0.14

AdvantageApplication

0.14

0.12

ErrorLogUsage

0.12

0.09

0.10

0.10

0.08

0.08

0.05 0.05

0.06

0.05

0.04

False True

0.03

WinRate(%)

WinRate(%)

0.02

100%

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

90%

100

100

70%

30%

50

50

10%

0

0

False True

False True False True

False True

False True

False True

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

0.20

0.18

0.16

0.16

0.16

0.18

ImportanceRatio/TrustRegion

0.16

0.15

0.13

RelativeReward

RelativeReward

0.13

0.14

0.15

0.14

0.12

EffectiveBatchSize

0.12

0.12

0.12

0.11

0.09

0.10

0.10

0.08

0.07

0.06

0.08

0.05 0.05

0.06

0.06

0.05

0.05

0.04

0.05

1024

512

256

64

False True

0.02

0.03

WinRate(%)

WinRate(%)

100%

100%

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

80%

100

100

70%

70%

50% 40%

30%

30%

50

50

20%

10%

0

0

1024 256 512 64

1024 256 512 64

1024 256 512 64

False True

False True

False True

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

###### Instruct

###### + 2.5e+18 SFT FLOPs

###### + 7.6e+18 SFT FLOPs

Standard-DeviationNormalizedAdvantage

0.20

0.18

0.17

0.17

0.18

0.16

0.15

0.14

0.14

RelativeReward

RelativeReward

0.15

0.13 0.13

0.14

0.12

0.12

0.11

LearningRate

0.10

0.10

0.08

0.08

0.05

0.06

0.04 0.04

0.04

0.05

0.04

False True

1e-06 5e-07

0.03

0.02

WinRate(%)

WinRate(%)

100%

100%

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

0.4 0.2 0.0 0.2 0.4

100

100

70%

70%

70%

50% 50%

30%

30%

30%

50

50

0

0

False True

False True

False True

1e-06 5e-07

1e-06 5e-07

1e-06 5e-07

Figure 8 Bootstrap analysis (n = 1000 samples) of hyperparameter optimization across different SFT compute budgets on test held out tasks. Each subplot examines a different hyperparameter, including increasing SFT compute: the base instruct model (left), +2.5e+18 SFT FLOPs (middle), and +7.6e+18 SFT FLOPs (right). For each hyperparametercompute combination, the top panel shows relative reward performance with error bars indicating 95% confidence intervals, while the bottom panel displays win rates representing the percentage of bootstrap iterations where each parameter value achieved maximum performance. Results demonstrate that optimal hyperparameter values often shift as model pre-training compute increases, suggesting that hyperparameter selection should be adapted based on the computational budget allocated to SFT.

