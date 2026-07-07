## Think Longer to Explore Deeper: Learn to Explore In-Context via Length-Incentivized Reinforcement Learning

Futing Wang*123 Jianhao Yan*123 Yun Luo3† Ganqu Cui3 Zhi Wang43 Xiaoye Qu3 Yue Zhang25 Yu Cheng6† Tao Lin2†

# arXiv:2602.11748v1[cs.CL]12Feb2026

### Abstract

Achieving effective test-time scaling requires models to engage in In-Context Exploration the intrinsic ability to generate, verify, and refine multiple reasoning hypotheses within a single continuous context. Grounded in State Coverage theory, our analysis identifies a critical bottleneck to enabling this capability: while broader state coverage requires longer reasoning trajectories, the probability of sampling such sequences decays exponentially during autoregressive generation, a phenomenon we term the “Shallow Exploration Trap”. To bridge this gap, we propose LengthIncentivized Exploration(LIE). This simple yet effective recipe explicitly encourages models to explore more via a length-based reward coupled with a redundancy penalty, thereby maximizing state coverage in two-step manner. Comprehensive experiments across different models (Qwen3, Llama) demonstrate that LIE effectively incentivize in-context exploration. As a result, our method achieves an average improvement of 4.4% on in-domain tasks and a 2.7% gain on out-ofdomain benchmarks. Our code is publicly available in https://github.com/LINs-lab /LIE.

### 1. Introduction

Scaling test-time computation, often conceptualized as enabling models to “think harder” before answering, has emerged as a powerful paradigm for breaking the performance ceiling of Large Language Models (LLMs) (Snell et al., 2024; Wu et al., 2024; Liu et al., 2025a). Broadly,

1Zhejiang University 2Westlake University 3Shanghai AI Laboratory 4Nanjing University 5Institute of Advanced Technology, Westlake Institute for Advanced Study 6The Chinese University of Hong Kong. Correspondence to: Yun Luo <luoyun1@pjlab.org.cn>, Yu Cheng <chengyu@cse.cuhk.edu.hk>, Tao Lin <lintao@westlake.edu.cn>.

Preprint. February 13, 2026.

###### LLM In-Context Exploration Flow

Query 𝑆 𝑆 𝑆

𝑆 𝑆

… 𝑆 𝑆 Answer

𝑆

𝑆

…

###### LLM Training Exploration Flow

Query 𝑆 𝑆

… 𝑆 …

Answer

Query 𝑆 𝑆

… 𝑆 …

Answer

Query 𝑆 … 𝑆 … 𝑆 Answer

* Different colors refer to different states

Figure 1. The difference between In-Context Exploration and Training Exploration. Our framework distinguishes between the exploration of the training process and in-context inference. In the training phase, reinforcement learning incentivizes the model to explore and learn from diverse state distributions. In contrast, during test-time inference, in-context exploration empowers the model to actively traverse and navigate states.

Test-Time Scaling (TTS) strategies fall into two primary regimes: Parallel Scaling (Lightman et al., 2023; Snell et al., 2024; Liu et al., 2025a; Brown et al., 2024; Wang et al., 2022), which aggregates outputs from multiple independent samples, and Sequential Scaling (Guo et al., 2025; Jaech et al., 2024; Kumar et al., 2024), which prioritizes extended reasoning chains or iterative refinement. While Parallel Scaling structures the search space externally, Sequential Scaling via Long CoT represents an intrinsic dimension of TTS, as it directly harnesses the model’s intrinsic reasoning capabilities (Snell et al., 2024).

One iconic capability among this intrinsic reasoning is incontext exploration (Setlur et al., 2025), which is achieved through the generation, verification, and refinement of multiple hypotheses within a continuous context (Gandhi et al., 2024; Setlur et al., 2025). Consequently, fostering such in-context exploration serves as the critical catalyst for unlocking the full potential of sequential scaling. As illustrated in Figure 1, in-context exploration enables the model to sequentially traverse diverse reasoning states to find the correct answer. However, how to effectively train LLMs to

acquire in-context exploration remains underexplored.

To study this problem, we first analyze the bottleneck of in-context exploration through the lens of Count-Based Exploration theory in RL training (Auer, 2002), which establishes state coverage as a fundamental proxy for exploration quality. We extend the theory into test-time in-context exploration and adopt state coverage, the in-context state diversity, as a theoretical proxy for the exploration quality. Figure 1 demonstrates the difference between in-context exploration and the classic rl training exploration. Our theoretical analysis of state coverage exposes a critical bottleneck (Remark 4.3): achieving broader state coverage requires longer reasoning sequences, yet such sequences face exponentially diminishing sampling probabilities (Shallow Exploration Trap) (Lemma 4.2). Empirically, we analyze the training dynamics of GSPO and GRPO baselines. Our observations validate our theoretical results and also reveal that these methods implicitly incentivize response length to some degree, while decreasing states density (Section 4.3). This observation prompts our primary research question:

How to effectively incentivize in-context exploration in RL for test-time scaling?

To address this, we propose Length-Incentivized Exploration (LIE), an RL recipe that maximizes in-context state coverage, in a two-step manner. The length reward elevates the upper bound of states, and the redundancy penalty matches the number of states given length (as introduced in Section 5). Experiments demonstrate that LIE significantly improves performance compared to GRPO and GSPO baselines across diverse models (Qwen3 (Yang et al., 2025a) and LLaMA (Meta, 2024)). Specifically, our method achieves consistent performance boosts, yielding a 4.4% average gain on in-domain reasoning tasks and a 2.7% improvement on out-of-domain benchmarks, underscoring its superior generalization capabilities.

Our key contributions are summarized as follows:

- • Identifying the In-Context Exploration Bottleneck.. We identify “Shallow Exploration Trap” as the bottleneck of in-context exploration, substantiating its existence through both theoretical derivations (Proposition 4.1 & Lemma 4.2) and empirical validations (Figure 3).
- • Length-Incentivized Exploration (LIE) Recipe. We propose a simple yet effective training recipe that explicitly incentivizes the model to overcome the bottleneck, enabling the model to explore during both training and inference.
- • Empirical Validation and Test-Time Scaling. Comprehensive experiments demonstrate that LIE significantly outperforms standard baselines, achieving effective test-time scaling and eliciting more cognitive behaviors.

### 2. Related Work

Scaling Test-Time Compute via Long CoT. Recent works have shown remarkable potential in scaling test-time computation (Snell et al., 2024) by training models to generate the long chains of thoughts (CoT) that enable strategic behaviors, including verification, self-correction (Kumar et al., 2024), etc. Scaling test-time computation significantly enhances the performance of Large Language Models (Guo et al., 2025; Team et al., 2025; Luo et al., 2025; Jaech et al., 2024) on various domains. Reinforcement Learning (RL), in particular, facilitates “natural” scaling of test-time computation through intrinsic adaptive exploration and strategy application (Guo et al., 2025; Team et al., 2025; Gandhi et al., 2025; Setlur et al., 2025; Yeo et al., 2025). These processes are mechanistically driven by negative gradients (Setlur et al., 2025; Zhu et al., 2025) or high entropy (Cui et al., 2025; Wang et al., 2025a; Cheng et al., 2025). S1 (Muennighoff et al., 2025) forces the model to think longer through explicitly forcing the generation of additional tokens to extrapolate thinking. However, we focus directly on increasing the computation usage during reinforcement training to activate in-context exploration.

Length-Aware Reasoning. Recent studies have investigated length-aware reasoning, specifically aiming to mitigate overthinking and improve both efficiency and controllability. Several works (Jiang et al., 2025; Huang et al., 2025; Zhang et al., 2025b; Kang et al., 2025; Zhang et al., 2025c; Yu et al., 2025c; Yang et al., 2025b; Liu et al., 2025b) propose adaptively switching between long and short Chainof-Thought (CoT) based on problem difficulty through training. Others (Zhang et al., 2025a; Ma et al., 2025) focus on controlling token usage during inference, while some studies address reasoning within a specific budget (Wen et al., 2025; Aggarwal & Welleck, 2025; Xu et al., 2025). In contrast to these efficiency-centric approaches that primarily aim to optimize token usage or curb overthinking, we posit that extended reasoning is a prerequisite for reaching deep reasoning states. We investigate explicitly scaling up reasoning trajectories not as a burden, but as a mechanism to break the “Shallow Exploration Trap” and activate intrinsic exploratory behaviors.

### 3. Background

In this section, we formulate LLM reasoning as a Markov Decision Process (MDP) and review the theoretical foundations of Count-Based Exploration in traditional reinforcement learning.

#### 3.1. MDP Formulation of LLM Reasoning

LLM Reasoning as an MDP. We model the autoregressive generation process of a Large Language Model (LLM)

as a deterministic MDP tuple (S,A,π,T), where

- • State Space (S): The state space S consists of all possible sequences of tokens from the vocabulary V. Crucially,

a specific state st ∈ S at time step t is the concatenation of the input query (prompt) x and the thought chain generated so far y<t. Formally, st = [x,y1,...,yt−1].

- • Action Space (A): The action space is discrete and equivalent to the model’s vocabulary V, where an action

at ∈ V corresponds to selecting the next token to append to the current sequence.

- • Transition Dynamics (T): The transition is determinis-

tic. Given the current state st and action at, the next state is uniquely determined by appending the token to the history: st+1 = st ⊕ at. The process terminates when a special end-of-sequence (EOS) token is generated.

- • Policy (πθ): The LLM functions as a parameterized policy πθ(at|st), which maps the current context st to a probability distribution over the vocabulary V.

Due to the combinatorial nature of language, the size of S grows exponentially with sequence length (|S| ≈ |V|L), making the state space extremely vast and sparse.

Reinforcement Learning for LLM Reasoning. To optimize the policy πθ to maximize the expected return J(πθ) = E[ rt], the optimization is typically performed via Policy Gradient (Sutton et al., 1999):

∇θJ(πθ) = Eτ

T

∇θ log πθ(at | st)Aπ(st,at) ,

t=0

where Aπ represents the advantage of taking action at. GRPO and GSPO are instantiations of this framework.

Group Relative Policy Optimization (GRPO). GRPO (Shao et al., 2024) performs optimization at the token level without a value function. It utilizes the standard per-token probability ratio:

πθ(yi,t|x,yi,<t) πθ

, (1)

ρi,t(θ) =

(yi,t|x,yi,<t)

old

where yi,t is the t-th token of the i-th sequence. The advantage is computed via group normalization:

Ri − mean({Rj}Gj=1) std({Rj}Gj=1)

Aˆi =

. (2)

The objective averages the PPO-clip loss (Schulman et al.,

2017) over all tokens in the generated sequences:

|yi|

G

1 G

1 |yi|

JGRPO(θ) = Ex,{yi}∼πθ

min ρi,t(θ)Aˆi,

old

t=1

i=1

clip(ρi,t(θ), 1 − ϵ, 1 + ϵ)Aˆi .

(3)

Group Sequence Policy Optimization (GSPO). GSPO (Zheng et al., 2025) elevates optimization to the sequence level using length-normalized importance ratios:

 

 

1/|yi|

|yi|

πθ(yi,t|x,yi,<t) πθ

, (4)

ρi(θ) =

(yi,t|x,yi,<t)

old

t=1

where the 1/|yi| exponent reduces variance from varying sequence lengths. Using the same group-based advantage Aˆi as GRPO, the objective is computed once per sequence.

3.2. Theoretical Foundation: Incentivizing State Coverage via Count-Based Exploration

Count-based exploration is a fundamental strategy to address the exploration-exploitation tradeoff. Central to this approach is the state visitation count N(s), which records the cumulative frequency of visiting a state s. These counts serve as a proxy for state coverage — the diversity of states visited during exploration. Guided by the principle of Optimism in the Face of Uncertainty (Auer et al., 2002), count-based methods augment the extrinsic reward with an exploration bonus b(s), typically inversely proportional to the counts (e.g., ∝ 1/ N(s)).

Theoretical Guarantees for Exploration. The optimality of count-based exploration is formally established in the Multi-Armed Bandit (MAB) setting. More detailed descriptions are shown in Appendix A.1. The Upper-Confidence Bound (UCB) algorithm selects action at by balancing reward estimates with visitation counts:

Rˆt(a) +

at = arg max

a∈A

2log t nt(a)

, (5)

where Rˆt(a) and nt(a) is the reward estimate and visitation count for action a at time t. The bonus term 2log t/nt(a) decreases as the action is sampled more frequently. Crucially, this bonus minimizes cumulative regret, yielding a theoretical guarantee for efficiently identifying the optimal action, formalized as:

Theorem 3.1 (Optimality of Count-based Exploration (Auer, 2002)). In an MAB setting, let L(T) =

E[ Tt=1 (R∗ − R(at))] denote the total regret over T steps, where R(a) = ERa[R] is the expected reward for any action a. Ra is the reward distribution for action a and R∗ is the reward for the optimal action. The UCB algorithm (Eq. 15) achieves this optimal bound:

∆a , (6)

L(T) ≤ 8log T ·

lim

T→∞

a|∆a>0

where ∆a is the reward gap between action a and the optimal action.

- Remark 3.2. The count-based exploration principle provides a fundamental insight for LLM reasoning: effective exploration requires maximizing state coverage. It is crucial to note that in standard RL training, the visitation count N(s) is aggregated across the whole state space over many training episodes. For in-context exploration, however, we must adapt this principle to quantify state diversity within a single reasoning trajectory.

### 4. In-Context Exploration

While the standard MDP formulation addresses trainingtime optimization, In-Context Exploration addresses how the model scales at test time. As highlighted in Remark 3.2 and Figure 1, the focus shifts from global state coverage to maximizing state coverage within a single trajectory.

#### 4.1. In-Context State Space

Defining in-context states. Given a single reasoning trajectory τ = (y1,y2,...,yL), where L is the trajectory length, we define the in-context state space as the sequence of autoregressive states visited during generation:

SIC(τ) = {s1,s2,...,sL}, (7)

A direct application of count-based states to in-context states faces a fundamental challenge: since every state st contains a unique history prefix, no raw state is ever visited twice within a trajectory, rendering state coverage measurement trivial and meaningless.

State abstraction. To derive meaningful visitation counts, we introduce a state abstraction function ϕ that maps raw states to abstract states capturing the logical state of reasoning rather than the literal history. Following evidence that semantics is predominantly encoded in immediate local patterns (Levy & Goldberg, 2014), we employ last-n-grams as our abstraction:

ϕ(st) = (yt−n+1,...,yt). (8)

By aggregating states based on these local patterns, we obtain pseudo-visitation counts that meaningfully estimate state coverage within a single context.

Quantifying in-context exploration. Furthermore, we can now quantify in-context exploration quality. Following the count-based exploration principle from Section 3.2, we define the In-Context Distinct State Count Ccontext(τ) as the cardinality of unique context states visited within a trajectory:

Ccontext(τ) = {ϕ(st) | t = 1,...,L} . (9)

This metric captures the volume of the state space covered during reasoning.

8k

Qwen3-4B-Base

y = x

GRPO GSPO

P(endbystep=t)

1.0

6k

C(τ)context

0.8

4k

0.6

0.4

2k

0.2

0

0.0

0 2k 4k 6k 8k

0 10k 20k 30k

Sequence Length (tokens)

Token index (step)

(a)

(b)

Figure 2. The Length Bottleneck of In-Context Exploration. (a) Capacity: Trajectory length dictates the maximum possible state coverage (Proposition 4.1). (b) “Shallow Exploration Trap”: The probability of reaching deep states decays exponentially (Lemma 4.2), preventing the model from utilizing this capacity.

Section 3.2 posits that the optimal exploration strategy involves assigning a bonus reward proportional to the inverse of state visitation counts, i.e, ∝ 1/ N(s), which corresponds to maximizing Ccontext. However, as described in Goodhart’s law (Goodhart, 1984) and discussed in Appendix C.1, directly maximizing Ccontext leads to losing efficacy and reward hacking in RL training.

#### 4.2. The Length Bottleneck of In-Context Exploration

Given the unreliability of direct optimization, we instead analyze a more fundamental question: what structural factor bottlenecks the growth of Ccontext?

Length as the capacity for exploration. From the definition in Eq. 8 and Eq. 9, we first derive a relationship between trajectory length and exploration potential:

Proposition 4.1 (Length as the Capacity for Exploration). The cumulative exploration utility of a trajectory is strictly upper-bounded by its length L:

Ccontext(τ) ≲ L. (10)

Consequently, extending the sequence length L is a necessary condition to raise the ceiling of exploration potential.

Proposition 4.1 establishes that sequence length L defines the “theoretical capacity” for state discovery (as evidenced in Figure 2a). To maximize state coverage, the model must generate sufficiently long trajectories.

The Shallow Exploration Trap. However, language models face intrinsic difficulties in sampling long sequences, as stated below.

Lemma 4.2 (Exponential Decay of Long Sequences). Let SL denote the set of states (i.e., responses in LLM reasoning) with sentence length L, and p(SL) denote the sampling probability for that state set. Then, there exists a constant ϵ ∈ (0,1), such that the sampling probability of SL is upper-bounded by an exponentially decaying function of its length L, the proof can be found in Appendix A.2:

p(SL) < (1 − ϵ)L−1 . (11)

- 0.3k

- 0.6k

- 0.9k

- 1.2k

- 1.5k

Ccontext(τ)

ρL,C = 0.977 ρC,Acc = 0.587

0 100 200 300 400 500

Training Step

80

84

88

92

96

Rcontext(τ)

0 150 300 450 600

Training Step

0.3k

- 0.6k

- 0.9k

- 1.2k

- 1.5k

Ccontext(τ)

ρL,C = 0.964 ρC,Acc = 0.777

0 150 300 450 600

Training Step

80

84

88

92

96

Rcontext(τ)

- 0.8k

- 1.6k

- 2.4k

- 3.2k

- 4.0k

32

40

48

56

- 0.8k

- 1.6k

- 2.4k

- 3.2k

- 4.0k

32

40

48

56

GRPO GSPO

Metric Response Length Accuracy

- Figure 3. The training dynamics of Ccontext and RContext in GRPO and GSPO on Qwen3-4B-Base. Two limitations are observed: (1) Shallow Exploration Trap: GRPO faces bottlenecks in extending trajectory length and performance, while GSPO shows slow length expansion. (2) Degrading Information Density: Both methods display degradation in ratio over time.

Lemma 4.2 characterizes the Shallow Exploration Trap: the model is exponentially biased against the long trajectories required to reach reasoning states (Figure 2b).

Remark 4.3 (The Exploration-Length Conflict). Together, Proposition 4.1 and Lemma 4.2 reveal a fundamental conflict for in-context exploration: long trajectories are necessary for broad state coverage, yet they are exponentially unlikely to be sampled.

4.3. Empirical Validation in RL Training

To further validate the relationship between response length, state coverage, and model performance (Proposition 4.1), we conduct a pilot study using GRPO and GSPO.

Length correlates with coverage and performance. As illustrated in Figure 3, we observe a remarkably high Pearson correlation (ρ(L,Ccontext) > 0.96) between Response Length and Ccontext. This confirms our Proposition 4.1: extending the physical length of the trajectory is the primary mechanism by which the model expands its exploration volume. Crucially, as the model accesses these longer sequences, reasoning accuracy improves.

The limitations of standard RL Training. However, our analysis reveals bottlenecks to incentivized in-context exploration in current RL training:

- 1. Shallow exploration trap (Lemma 4.2): In GRPO, performance plateaus along with the response length stabilization. While GSPO shows continuous length growth, it is prohibitively slow. Both algorithms eventually succumb to the “Shallow Exploration Trap”, failing to push L far enough to reach complex states.
- 2. Degrading state density: As shown in Figure 3, length

0 100 200 300 400 500

Training Step

These limitations indicate that enhancing in-context exploration requires not only longer reasoning trajectories but also efficient and non-redundant exploration.

### 5. Length-Incentivized Exploration

In this section, we introduce a targeted reward shaping recipe — Length-Incentivized Exploration(LIE) to enforce in-context exploration in reinforcement learning.

#### 5.1. The Length-Incentivized Reward

We first introduce a Length-Incentivized Reward (Rlen) designed to explicitly encourage the model to extend its reasoning trajectories beyond its initial policy’s tendencies when it fails to reach the correct answer Racc=0.

We define a sample-wise target length Ltarget,i = Lref,i+∆L, where Lref,i denotes the response length of the initial policy of i-th sample and ∆L represents the desired least increment for exploration. The length reward is defined as:

 

0 L ≥ Ltarget, Racc = 0 −η(Ltarget − L) L < Ltarget, Racc = 0 0 Racc = 1

. (12)

Rlen =



This formulation creates a curriculum: the model is rewarded for extending its reasoning process beyond its usual depth whenever it cannot solve a problem immediately.

#### 5.2. Redundancy Reward: Enforcing Effective State Coverage

While Rlen increases the exploration capacity L, it does not guarantee that this capacity is converted into effective state coverage Ccontext (as revealed in Section 4.3). Recall Proposition 4.1: Ccontext(τ) ≤ L. The equality holds only when redundancy is minimized.

increases, the Distinct Ccontext Ratio (Rcontext) tends to decrease. This indicates that standard RL incentives increase trajectory length but fail to improve state density. The model fills the extra capacity with repetitive tokens rather than novel reasoning steps, degrading the efficiency of exploration.

Redundancy Reward. We adopt a hard-threshold approximation of redundant in-context states that is in line with the widely used repetition penalty (Yu et al., 2025b). Let Nτ(ϕ(st)) be the visitation count of state abstraction ϕ(st)

in the current trajectory. The reward is defined as:

Rred = −β · I[Nτ(ϕ(st)) > Θ], (13) where Θ is the threshold and β is the penalty magnitude.

Theoretical Consistency. This formulation aligns with the count-based exploration bonus bt ∝ 1/√Nt (Section

- 3.2). Maximizing Rlen + Rred is equivalent to maximizing state coverage Ccontext, in a two-step manner. The length reward elevates the upper bound of states, and the redundancy penalty matches the number of states given length.

Final recipe. To prioritize solution accuracy while encouraging test-time scaling, we shape the reward to favor correct responses and longer reasoning trajectories, while penalizing redundant exploration:

R = Racc + Rlen + β · Rred , (14)

### 6. Experiments

In this section, we empirically validate the effectiveness of LIE. We conduct comprehensive experiments to address the following core research questions:

- • RQ1: performance & test-time scaling. Does our recipe mitigate the "Shallow Exploration Trap" to achieve more effective test-time scaling?
- • RQ2: impact of model capability. How does the effectiveness of our recipe on the different policy models’ initial state?
- • RQ3: exploration dynamics. How does the lengthincentivized recipe impact global state diversity and exploration during training?

#### 6.1. Experimental Setup

Evaluation. We assess performance across eight reasoning benchmarks, categorized into in-domain mathematical tasks and out-of-distribution (OOD) general reasoning. The in-domain suite comprises AIME 2024/2025, AMC (Li et al., 2024), MATH-500 (Hendrycks et al., 2021), and OlympiadBench (He et al., 2024). The OOD evaluation includes ARC-c (Clark et al., 2018), GPQA-Diamond (denoted as GPQA* (Rein et al., 2024)), and MMLU-Pro (Wang et al., 2024). Regarding metrics, for benchmarks with limited sample sizes (AIME and AMC), we report the average accuracy over 32 independent runs (Avg@32). For all other benchmarks, we report Pass@1. All evaluations are conducted with a sampling temperature of 0.6, top-p=1.0, and a maximum response length of 32k tokens, which surpasses the training budget.

RLVR setups. Training is performed with a prompt batch size of 128, generating 8 rollouts per prompt. We update the policy using a mini-batch size of 32 and a learning rate of

1e-6. During the training phase, the sampling temperature is set to 1.0, with a maximum response length of 8,192 tokens. Experiments are implemented using the verl framework1 on nodes equipped with 4×H100 GPUs, employing MathVerify2 for outcome-based reward calculation. We default set n = 10, η = 0.3/9000, β = 0.6 and Θ = 10 in reward. We discussed these hyperparameters in Appendix C.5.

Models and baselines. We designate Qwen3-4B-Base as our primary testbed. To verify the universality of our recipe across different training stages and model families, we also extend our evaluation to Qwen3-4B (non-thinking) and Llama-OctoThinker-3B-Base (Wang et al., 2025b). The training datasets are selected to align with the respective models: DAPO-Math-17k (Yu et al., 2025a) for Qwen34B-Base, Polaris (An et al., 2025) for Qwen3-4B-Instruct, and DeepMath-5k (He et al., 2025; Tan et al., 2025) for Llama-OctoThinker. We primarily investigate GSPO as the core algorithm. Additionally, for Qwen3-4B-Base, we benchmark against standard GRPO and a GRPO variant with a high clip ratio (0.28) to validate our recipe’s effectiveness across different algorithms.

#### 6.2. Results

LIE achieves superior performance across algorithms on Qwen3-4B-Base. As shown in Table 1, our recipe consistently outperforms baselines across all benchmarks. On Qwen3-4B-Base, applying our method to GSPO boosts average in-domain accuracy from 49.4% to 53.8% and OOD accuracy from 62.7% to 67.1%. Notably, we achieve a 6.2% gain on the challenging AIME25, confirming that breaking the “Shallow Exploration Trap” effectively unlocks complex problem-solving capabilities. Moreover, our recipe achieves consistent gains across different model sizes (as shown in Appendix C.3).

Effects of recipe on in-context state coverage. To understand the source of these gains, we analyze the training dynamics in Figure 4. Compared to the GSPO baseline, our method explicitly drives longer trajectories, resulting in a rapid expansion of Ccontext. This confirms that our recipe incentive effectively increases the “Capacity for Exploration” (Proposition 4.1). While the state density Rcontext initially drops, the model eventually learns to utilize this extended budget, converting the increased state coverage into reasoning paths that correlate with the accuracy spike.

Test-time scaling via Long CoT. Figure 5 illustrates the performance trajectory as we increase the inference compute budget. While standard RL models (Blue line) saturate or degrade when forced beyond their learned policy length, our recipe (Red line) exhibits a superior scaling curve, main-

- 1https://github.com/volcengine/verl
- 2https://github.com/huggingface/Math-Verify

Table 1. In-Domain and Out-of-Domain evaluation performance based on Qwen3-4B-Base. Bold and underline indicate the best and second-best results. Gains of our methods compared to corresponding baselines are marked in blue.

In-Domain Performance Out-of-Domain Performance

Model

MATH Olympiad AMC AIME AIME25 Avg. ARC-c GPQA MMLU-Pro Avg. Qwen3-4B-Base 66.0 33.2 36.6 8.5 6.9 30.2 66.9 26.3 30.9 41.4 GRPO 80.4 47.1 55.2 16.8 18.7 43.6 84.6 44.4 60.1 63.0 GRPO w/Clip-higher 86.4 54.1 61.8 25.2 22.2 49.9 89.6 46.0 62.8 66.1 GSPO 85.2 51.7 62.7 26.7 20.5 49.4 88.4 48.5 61.5 66.1

GRPO + LIE 85.0 49.9 60.5 22.9 16.4 46.9+3.3 90.3 46.5 60.4 65.7+2.7 GRPO w/Clip-higher + LIE 88.8 54.1 65.5 30.4 24.5 52.6+2.7 91.5 43.9 63.0 66.1+0.0 GSPO + LIE 88.4 57.2 66.2 30.5 26.7 53.8+4.4 91.4 47.5 63.8 67.6+1.5

- 0.5k
- 1.0k

- 1.5k
- 2.0k

- 2.5k

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

96

C(τ)context

R(τ)context

88

80

72

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

- 1.0k

- 2.0k

- 3.0k

- 4.0k

- 5.0k

56

###### Accuracy

48

###### L

40

32

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

###### GSPO GSPO + LIE

- Figure 4. Ccontext, Rcontext, response length, and performance on the valid dataset comparing GSPO baseline and our recipe.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

4k 8k 12k 16k 24k 32k

Budget

0.42

0.44

0.46

0.48

0.50

0.52

0.54

Accuracy

GRPO GSPO GRPO w/Clip-higher Baseline + LIE

- Figure 5. Test-time extrapolation performance. While standard baselines saturate or degrade when forced beyond their learned policy length, the Length-Incentivized Exploration recipe exhibits a superior scaling curve.

taining an upward trend. This confirms that LIE effectively utilizes the extended token budget to explore broader hypothesis spaces by improved in-context exploration.

LIE is effective across different models. We extended our recipe to Qwen3-4B (post-trained) and LlamaOctoThinker (Wang et al., 2025b). Table 2 reports consistent

- Table 2. In-Domain Evaluation performance based on Qwen34B and Llama-OctoThinker-3B. Bold indicates the best result. Gains of our methods compared to corresponding baselines are marked in blue.

MATH Olympiad AMC AIME AIME25 Avg.

Qwen3-4B 82.8 51.9 60.4 24.2 19.4 47.7 GSPO 94 68.1 82 54.2 42.5 68.2 GSPO + LIE 94.4 67 85.3 57.7 46.4 70.2+2.0

OctoThinker 23.4 9.0 10.4 1.1 0.6 8.9 GSPO 55.8 23.1 28.2 3.8 2.3 22.6 GSPO + LIE 60.8 28.1 30.3 4.5 4.4 25.6+3.0

- Table 3. Continual Scaling via Curriculum Training based on Qwen3-4B-Base. “Stage 1” represents the initial training, while “Stage 2” employs further length extension. The results demonstrate consistent improvements across benchmarks.

MATH Olympiad AMC AIME AIME25 Avg.

Qwen3-4B-Base 66.0 33.2 36.6 8.5 6.9 30.2 GSPO 85.2 51.7 62.7 26.7 20.5 49.4

- Stage1 88.4 57.2 66.2 30.5 26.7 53.8+4.4
- Stage2 89.4 59.0 71.7 33.4 27.9 56.3+2.5

gains (2%–3%) across models. The training dynamics in Figures 11 and 6 reveal that our recipe universally drives the expansion of state coverage, converting the increased trajectory length into improved in-context exploration.

#### 6.3. Analysis and Discussion

Global diversity. Beyond in-context state diversity, we track Global Distinct Count (Cglobal) and policy entropy to assess exploration across the state space during training time. As shown in Figure 7, standard GSPO exhibits saturation and a rapid entropy drop, indicating premature convergence (mode collapse). In contrast, our recipe maintains continuous growth in and significantly higher entropy. This sustained global exploration prevents mode collapse, enabling the policy to discover rare, high-reward states during training that standard methods miss.

Continual scaling via curriculum training. To validate scaling laws beyond the initial setup, we conducted a second training stage (“Stage 2”) with a relaxed 12k token limit. As shown in Table 3, the model exhibits continuous improvement. This confirms that our recipe effectively converts

###### L

Ccontext(τ)

Rcontext(τ)

###### Accuracy

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 1.0k

- 1.5k
- 2.0k

- 2.5k

72

- 1.6k

- 2.4k

- 3.2k

- 4.0k

98

68

- 96

- 97

- 98

64

60

56

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

Training Step

Training Step

###### GSPO GSPO + LIE

- Figure 6. Tracking Ccontext, Rcontext, response length L, and valid performance during the training of Qwen3-4B. Compared to the GSPO baseline (Blue), the LIE (Purple) successfully breaks the “Shallow Exploration Trap” and achieves performance gains.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 150 300 450 600

Training Step

1.50M

- 3.00M
- 4.50M

Cglobal()

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 150 300 450 600

Training Step

0.0

0.3

0.6

0.9

1.2

Entropy

GSPO GSPO + LIE

- Figure 7. Global Exploration Dynamics: Diversity and Entropy. Our recipe maintains significantly higher entropy and continuous growth in global state coverage compared with the baseline.

Backtracking Verification Subgoal Setting Enumeration

40

60

80

100

120

140

Count

103

87

101

50

121 123

114

63

GSPO

+ LIE

- Figure 8. Frequency analysis of cognitive behaviors. The proposed recipe (+ LIE) significantly boosts all behaviors, with a notable enhancement in Backtracking.

100

- 1.8k

- 2.4k

80

C(τ)context

R(τ)context

60

- 0.6k

- 1.2k

40

20

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

- 6.0k

8.0k

L

0 150 300 450 600

Training Step

20

30

40

50

60

Accuracy

GSPO GSPO + Rlen

Figure 9. Impact of the Length-Incentivized Reward (Llen). Forcing the model to "think longer" yields performance gains. However, the accompanying drop in the distinct ratio Rcontext reveals a tendency towards repetition.

Only Rlen incentives exploration but introduce repetition. As shown in Figure 9, explicitly incentivizing longer trajectories rapidly expands Ccontext and improves accuracy, confirming that length is capacity for additional in-context exploration (Proposition 4.1). However, this expansion is accompanied by a decline in Rcontext revealing that the model often satisfies the length reward through redundant rather than informative generation. This observation necessitates a redundancy penalty to maintain exploration quality, a synergy further detailed in the ablation study of our reward components (Equation 14) in Appendix C.6.

- 7. Conclusion

4.0k

2.0k

0 150 300 450 600

Training Step

additional compute into reasoning accuracy, establishing a scaling trend with the allocated reasoning horizon.

Changes in Reasoning Behaviors. Adopting the framework from (Gandhi et al., 2025), we employ Claude-3.7Thinking to quantify four representative cognitive behaviors: Backtracking, Verification, Subgoal Setting, and Enumeration. As illustrated in Figure 8, the integration of LIE consistently stimulates more frequent reasoning activities across all categories. Crucially, we observe a substantial increase in Backtracking (from 103 to 121), which is a sophisticated cognitive capability. This improvement underscores that LIE effectively enhances the model’s capacity for in-context exploration. We also discuss the changes in reasoning structure in Appendix D.

In this work, we identified the conflict between the length required for broader state coverage and the “Shallow Exploration Trap” as a critical bottleneck in in-context exploration through theoretical and empirical analysis. To overcome this limitation, we proposed Length-Incentivized Exploration, a simple yet effective recipe that maximizing the in-context state coverage. Our experiments across diverse models and benchmarks confirm that LIE incentivize the in-context exploration during test time. Overall, our work advances

test-time scaling by explicitly activating in-context exploration, demonstrating how additional computation can be systematically converted into more effective reasoning.

### Impact Statements

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Aggarwal, P. and Welleck, S. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

An, C., Xie, Z., Li, X., Li, L., Zhang, J., Gong, S., Zhong, M., Xu, J., Qiu, X., Wang, M., and Kong, L. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https:

//hkunlp.github.io/blog/2025/Polaris.

Auer, P. Using confidence bounds for exploitationexploration trade-offs. Journal of machine learning research, 3(Nov):397–422, 2002.

Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., and Munos, R. Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information Processing Systems, volume 29, 2016.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Cheng, D., Huang, S., Zhu, X., Dai, B., Zhao, W. X., Zhang, Z., and Wei, F. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mech-

anism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Feng, Y., Kempe, J., Zhang, C., Jain, P., and Hartshorn, A. What characterizes effective reasoning? revisiting length, review, and structure of cot. arXiv preprint arXiv:2509.19284, 2025.

Gandhi, K., Lee, D., Grand, G., Liu, M., Cheng, W., Sharma, A., and Goodman, N. D. Stream of search (sos): Learning to search in language. arXiv preprint arXiv:2404.03683, 2024.

Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., and Goodman, N. D. Cognitive behaviors that enable selfimproving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Goodhart, C. A. Problems of monetary management: the uk experience. In Monetary theory and practice: The UK experience, pp. 91–121. Springer, 1984.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

He, Z., Liang, T., Xu, J., Liu, Q., Chen, X., Wang, Y., Song, L., Yu, D., Liang, Z., Wang, W., et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Huang, S., Wang, H., Zhong, W., Su, Z., Feng, J., Cao, B., and Fung, Y. R. Adactrl: Towards adaptive and controllable reasoning via difficulty-aware budgeting. arXiv preprint arXiv:2505.18822, 2025.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jiang, L., Wu, X., Huang, S., Dong, Q., Chi, Z., Dong, L., Zhang, X., Lv, T., Cui, L., and Wei, F. Think only

when you need with large hybrid-reasoning models. arXiv preprint arXiv:2505.14631, 2025.

Kang, Y., Sun, X., Chen, L., and Zou, W. C3ot: Generating shorter chain-of-thought without compromising effectiveness. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 24312–24320, 2025.

Kolter, J. Z. and Ng, A. Y. Near-Bayesian exploration in polynomial time. In Proceedings of International Conference on Machine Learning, pp. 513–520, 2009.

Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Levy, O. and Goldberg, Y. Neural word embedding as implicit matrix factorization. Advances in neural information processing systems, 27, 2014.

Li, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S., Rasul, K., Yu, L., Jiang, A. Q., Shen, Z., et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,

- B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Liu, R., Gao, J., Zhao, J., Zhang, K., Li, X., Qi, B., Ouyang, W., and Zhou, B. Can 1b llm surpass 405b llm? rethinking compute-optimal test-time scaling. arXiv preprint arXiv:2502.06703, 2025a.

Liu, W., Zhou, R., Deng, Y., Huang, Y., Liu, J., Deng, Y., Zhang, Y., and He, J. Learn to reason efficiently with adaptive length-based reward shaping, 2025b. URL https://arxiv.org/abs/2505.15612.

Luo, M., Tan, S., Wong, J., Shi, X., Tang, W. Y., Roongta, M., Cai, C., Luo, J., Li, L. E., Popa, R. A., and Stoica, I. Deepscaler: Surpassing 01-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notio n.site/DeepScaleR-Surpassing-01-Previ ew-with-a-1-5B-Model-by-Scaling-RL-1 9681902c1468005bed8, 2025.

Ma, X., Wan, G., Yu, R., Fang, G., and Wang, X. Cot-valve: Length-compressible chain-of-thought tuning. arXiv preprint arXiv:2502.09601, 2025.

Meta, A. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. Meta AI Blog. Retrieved December, 20:2024, 2024.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candès, E., and Hashimoto, T. B. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20286– 20332, 2025.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Setlur, A., Yang, M. Y., Snell, C., Greer, J., Wu, I., Smith, V., Simchowitz, M., and Kumar, A. e3: Learning to explore enables extrapolation of test-time compute for llms. arXiv preprint arXiv:2506.09026, 2025.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Strehl, A. L. and Littman, M. L. An analysis of modelbased interval estimation for markov decision processes. Journal of Computer and System Sciences, 74(8):1309– 1331, 2008.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

Tan, Y., Wang, M., He, S., Liao, H., Zhao, C., Lu, Q., Liang, T., Zhao, J., and Liu, K. Bottom-up policy optimization: Your language model policy secretly contains internal policies. arXiv preprint arXiv:2512.19673, 2025.

Tang, H., Houthooft, R., Foote, D., Stooke, A., Xi Chen,

- O., Duan, Y., Schulman, J., DeTurck, F., and Abbeel,
- P. # exploration: A study of count-based exploration for deep reinforcement learning. In Advances in Neural Information Processing Systems, volume 30, 2017.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025a.

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.
- Wang, Z., Zhou, F., Li, X., and Liu, P. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025b.

Wen, H., Wu, X., Sun, Y., Zhang, F., Chen, L., Wang, J., Liu, Y., Liu, Y., Zhang, Y.-Q., and Li, Y. Budgetthinker: Empowering budget-aware llm reasoning with control tokens. arXiv preprint arXiv:2508.17196, 2025.

Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models. arXiv preprint arXiv:2408.00724, 2024.

Yu, Z., Xu, T., Jin, D., Sankararaman, K. A., He, Y., Zhou, W., Zeng, Z., Helenowski, E., Zhu, C., Wang, S., et al. Think smarter not harder: Adaptive reasoning with inference aware optimization. arXiv preprint arXiv:2501.17974, 2025c.

Zhang, J., Dong, R., Wang, H., Ning, X., Geng, H., Li, P., He, X., Bai, Y., Malik, J., Gupta, S., et al. Alphaone: Reasoning models thinking slow and fast at test time. arXiv preprint arXiv:2505.24863, 2025a.

Zhang, J., Lin, N., Hou, L., Feng, L., and Li, J. Adaptthink: Reasoning models can learn when to think. arXiv preprint arXiv:2505.13417, 2025b.

Zhang, S., Wu, J., Chen, J., Zhang, C., Lou, X., Zhou, W., Zhou, S., Wang, C., and Wang, J. Othink-r1: Intrinsic fast/slow thinking mode switching for over-reasoning mitigation. arXiv preprint arXiv:2506.02397, 2025c.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

Xu, Y., Dong, H., Wang, L., Sahoo, D., Li, J., and Xiong, C. Scalable chain of thoughts via elastic reasoning. arXiv preprint arXiv:2505.05315, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, W., Ma, S., Lin, Y., and Wei, F. Towards thinkingoptimal scaling of test-time compute for llm reasoning. arXiv preprint arXiv:2502.18080, 2025b.

Yeo, E., Tong, Y., Niu, M., Neubig, G., and Yue, X. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025a.

Yu, T., Ji, B., Wang, S., Yao, S., Wang, Z., Cui, G., Yuan, L., Ding, N., Yao, Y., Liu, Z., et al. Rlpr: Extrapolating rlvr to general domains without verifiers. arXiv preprint arXiv:2506.18254, 2025b.

Contents

- A. Theoretical Foundation

- A.1. Count-Based Exploration Theoretical Foundation

The exploration-exploitation tradeoff is a long-standing challenge in RL literature. This becomes more pronounced in LLM reasoning tasks due to the vast state-action space with reward sparsity, where existing methods often suffer from deficient exploration and poor sample efficiency. A classic, theoretically justified exploration principle is optimism in the face of uncertainty, which augments the reward estimates of less-visited states/actions with an exploration bonus proportional to their uncertainty. In the minimal multi-armed bandit (MAB) setting, the well-known upper-confidence bound (UCB) algorithm (Auer, 2002) chooses the action at according to:

at = arg max

a∈A

Rˆt(a) +

2log t nt(a)

, (15)

where Rˆt(a) and nt(a) is the reward estimate and visitation count for action a at time t. Theorem A.1 establishes the theoretical optimality of this count-based exploration strategy, demonstrating that it yields a cumulative regret that grows only logarithmically over time.

Theorem A.1. (Auer, 2002) In an MAB setting, let L(T) = E[ Tt=1 (R∗ − R(at))] denote the total regret over T steps, where R(a) = ERa[R] is the expected reward for any action a and R∗ is the reward for the optimal action. Let ∆a = R∗ − R(a) denote the reward gap between action a and the optimum. For any bandit algorithm, the asymptotic total regret is at least logarithmic in the number of steps:

lim

T→∞

L(T) ≥ log T ·

a|∆a>0

∆a KL(Ra||Ra∗)

. (16)

The UCB algorithm in Eq. 15 achieves logarithmic asymptotic total regret:

lim

T→∞

L(T) ≤ 8log T ·

a|∆a>0

∆a. (17)

Subsequent studies (Bellemare et al., 2016; Tang et al., 2017) extend this principle to MDPs by counting state-action pairs n(s,a) and adding a bonus reward to encourage exploring less-visited pairs as

R′(s,a) = R(s,a) +

β n(s,a)

, (18)

which is shown to be formally near-optimal within the probably approximately correct (PAC-MDP) framework (Strehl & Littman, 2008; Kolter & Ng, 2009).

- A.2. Exponential Decay of Long Sequences

Broadening the coverage of the state space naturally drives the model to discover rare yet correct reasoning chains, preventing premature convergence to local optima in sparse-reward landscapes and incentivizing reasoning capacities. However, naively counting n(s,a) in a vast language space becomes computationally prohibitive and statistically inefficient. Instead, we employ the response length as a coarse-grained abstraction of the state space. Lemma A.2 shows that the expected count of a response decays exponentially w.r.t. its length, resulting in a long-tailed distribution.

Lemma A.2 (Exponential Decay of Long Sequences). Let SL denote the set of states (i.e., responses in LLM reasoning) with sentence length L, and p(SL) denote the sampling probability for that state set. Then, there exists a constant ϵ ∈ (0,1), such that the sampling probability of SL is upper-bounded by an exponentially decaying function of its length L:

p(SL) < (1 − ϵ)L−1. (19)

Proof. Given a generative LLM model, let pL([EOS]) denote the sampling probability of the end token at step L. With the softmax function as the output probability in language models, we have pL([EOS]) ∈ (0,1), ∀L. There exists a lower bound

of sampling probability ϵ ∈ (0,1), such that pL([EOS]) ≥ ϵ, ∀L. In an autoregressive manner, the sampling probability of a response with length L is:

L−1

(1 − pi([EOS])) · pL([EOS])

p(SL) =

i=1

(20)

≤ (1 − ϵ)L−1 · pL([EOS]) < (1 − ϵ)L−1.

This completes the proof.

| |
|---|

### B. Experimental Details

- B.1. Template Prompt

We adopt the following template for all experiments involving Qwen models, building upon the Qwen3 template for Qwen3-4B-Base and the Qwen-Nothinking template for Qwen3.

Qwen3 Template

<|im_start|>user {problem} Let’s think step by step and output the final answer within \boxed{}. <|im_end|> <|im_start|>assistant

Qwen3-NoThinking Template

<|im_start|>user {problem} Let’s think step by step and output the final answer within \boxed{}. <|im_end|> <|im_start|>assistant <think>

</think> For training the Llama-OctoThinker models, we adopt the original prompt in (Wang et al., 2025b) to ensure performance. OctoThinker Template

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: You must put your answer inside \boxed{} and Your final answer will be extracted automatically by the \boxed{} tag. {problem} Assistant:

- B.2. Implementation of RL

In this section, we describe the RL training setup in detail. We implement GRPO and other baseline algorithms using the Verl framework. Across all algorithms and model variants, we adopt a unified set of hyperparameters, as reported in Table 4, and do not employ entropy regularization or KL-based losses.

### C. Additional Results

#### C.1. Pitfalls of Direct State Coverage Maximization

As discussed in Section 4, a natural inclination for incentivizing in-context exploration is to directly use the In-Context Distinct State Count Ccontext(τ) as a reward signal. However, our empirical analysis in Figure 10 reveals critical failure

Table 4. RL Hyperparameters.

#### Hyperparameter Value

Optimizer AdamW Policy learning rate 1e−6 Training batch size 128 prompts Samples per prompt 8 Mini-batch size 32 prompts Policy updates per rollout 16 Max prompt length 1024 tokens Max response length 8192 Rollout temperature 1.0

modes: reward hacking.

When Ccontext is directly maximized (as shown in the red curves), the model initially exhibits a sharp increase in state coverage. However, this is quickly followed by a "collapse" where the model learns to generate semantically hollow but structurally diverse tokens (e.g., random character permutations or irrelevant LaTeX symbols) to artificially inflate the state count. This results in a significant drop in reasoning accuracy. In contrast, our LIE recipe (purple curves) uses length as a proxy for capacity and redundancy as a constraint, which leads to a more stable and meaningful expansion of the state space, ultimately translating into higher and more stable task performance.

L

Ccontext(τ)

Rcontext(τ)

###### Accuracy

8.0k

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 0.0k
- 1.0k
- 2.0k
- 3.0k
- 4.0k

60

90

6.0k

45

75

4.0k

30

60

2.0k

15

45

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

Training Step

Training Step

GSPO GSPO + LIE GSPO + Maximize Ccontext

Figure 10. Training dynamics of maximizing Cdistinct as a reward.

#### C.2. Training dynamics of OctoThinker

As discussed in Section 6.2, our Length-Incentivized (LIE) recipe demonstrates strong generalization across various model architectures. Figure 11 illustrates the training dynamics of OctoThinker-3B-Base, comparing standard GSPO with our LIE-enhanced approach. These observations reinforce that "thinking longer" via length incentivization is a reliable mechanism for performance extrapolation, regardless of the underlying policy model.

###### L

Ccontext(τ)

Rcontext(τ)

###### Accuracy

- 6.0k

- 7.5k

90

- 0.8k

- 1.6k

- 2.4k

- 3.2k

30

80

25

- 3.0k

- 4.5k

70

20

60

15

1.5k

50

0 40 80 120 160 200

0 40 80 120 160 200

0 40 80 120 160 200

0 40 80 120 160 200

Training Step

Training Step

Training Step

Training Step

###### GSPO GSPO +LIE

Figure 11. The training dynamics of OctoThinker-3B-Base.

#### C.3. Scalability Across Model Scales

To verify the scalability of LIE, we evaluate the recipe across different model sizes: Qwen3-1.7B-Base, 4B-Base, and 8B-Base. As summarized in Table 5, LIE consistently delivers performance gains regardless of the base model’s capacity.

Specifically, on the in-domain reasoning average, LIE improves the 1.7B model by 7.6%, the 4B model by 4.4%, and the 8B model by 2.5% compared to the GSPO baseline. Notably, on out-of-distribution (OOD) tasks, the 8B model with LIE achieves a 73.4% average accuracy, a 2.9% improvement over the baseline. These results demonstrate that the "Shallow Exploration Trap" is a universal bottleneck for LLM reasoning, and LIE provides a robust mechanism to unlock deeper reasoning capabilities across the scaling spectrum.

Table 5. LIE on different model sizes.

In-Domain Performance Out-of-Domain Performance

Model

MATH Olympiad AMC AIME AIME25 Avg. ARC-c GPQA MMLU-Pro Avg. Qwen3-1.7B-Base 51.2 20.7 25.8 3.4 1.7 20.6 54.1 20.2 27.5 33.9

GSPO 70.6 32.3 38.4 8.1 3.8 30.6 79.4 28.2 41.0 49.5

+ LIE 77.0 39.6 45.2 17.5 11.5 38.2+7.6 79.1 33.8 45.6 52.8+3.3 Qwen3-4B-Base 66.0 33.2 36.6 8.5 6.9 30.2 66.9 26.3 30.9 41.4

GSPO 85.2 51.7 62.7 26.7 20.5 49.4 88.4 48.5 61.5 66.1

+ LIE 88.4 57.2 66.2 30.5 26.7 53.8+4.4 91.4 47.5 63.8 67.6+1.5 Qwen3-8B-Base 67.8 35.3 38.9 10.3 8.5 32.2 58.5 32.3 51.2 47.3

GSPO 89.8 58.8 72.9 34.4 25.8 56.3 93.2 50.0 68.3 70.5

+ LIE 91.4 60.4 73.4 37.2 31.6 58.8+2.5 94.5 55.1 70.7 73.4+2.9

#### C.4. Synergizing with SFT: Injection vs. Activation

We further investigate the interplay between Supervised Fine-Tuning (SFT) and our LIE. We posit that these two stages are orthogonal and complementary: SFT serves as the injection stage, embedding specific reasoning patterns and primitives into the model, while LIE serves as the activation mechanism, compelling the model to compose these patterns into longer, more complex reasoning chains.

Setup. We constructed a curated SFT dataset by randomly sampling 4k problems from the training set and generating oracle Chain-of-Thought (CoT) data using GPT-OSS-120B (Agarwal et al., 2025). We fine-tuned the Qwen3-4B-Base model on this data for 3 epochs with a 4,096 token context limit.

Results. The training dynamics in Figure 12 strongly support our hypothesis:

- 1. SFT provides the capability foundation (Injection). Both SFT-initialized runs (Red and Purple) start with significantly

higher state coverage (Ccontext) and accuracy compared to the Base model (Blue), confirming that SFT successfully injects necessary reasoning primitives.

- 2. LIE drives further extrapolation (Activation). While the standard SFT+GSPO baseline (Purple) shows improvement, it tends to plateau in trajectory length (L ≈ 5k) and coverage, remaining tethered to the SFT data distribution. In contrast, applying LIE on top of SFT (Red) effectively "activates" the model to explore beyond the supervised horizon. The model utilizes the injected patterns to sustain a continued expansion in reasoning length (L → 7k+) and distinct state coverage.
- 3. Orthogonality. The highest performance is achieved by SFT GSPO + ILE (Red). This demonstrates that our recipe functions as an independent scaling lever that synergizes with the strong priors from SFT to unlock the maximum reasoning potential.

64

- 6.0k

- 7.5k

96

- 0.8k

- 1.6k

- 2.4k

- 3.2k

56

88

- 3.0k

- 4.5k

48

80

40

72

1.5k

32

0 60 120 180 240 300

0 60 120 180 240 300

0 60 120 180 240 300

0 60 120 180 240 300

Training Step

Training Step

Training Step

Training Step

###### GSPO + ILE SFT GSPO SFT GSPO + ILE

Figure 12. Training Dynamics on Qwen3-4B-Base after SFT.

#### C.5. Hyperparameter Sensitivity Analysis

We conduct an ablation study to verify that our method’s improvements are robust to hyperparameter choices and to understand the impact of exploration constraints.

###### 6-gram

###### 10-gram

###### 15-gram

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 1.8k
- 2.4k

- 1.8k
- 2.4k

- 1.8k
- 2.4k

C(τ)context

C(τ)context

C(τ)context

- 0.6k
- 1.2k

- 0.6k
- 1.2k

- 0.6k
- 1.2k

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

| | | | | | |
|---|---|---|---|---|---|
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

96

96

96

88

88

88

R(τ)context

R(τ)context

R(τ)context

80

80

80

72

72

72

64

64

64

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

Training Step

###### GSPO GSPO + ILE

Figure 13. Ccontext and Rcontext dynamics across 6, 10, 15-grams.

Sensitivity to State Abstraction (n-gram). We first analyze the impact of the context window size n used for defining state abstraction (Equation 8). As shown in Figure 13, the relative performance trends between standard GSPO and our LIE-enhanced method remain consistent across n ∈ {6,10,15}, confirming that the improvements are not an artifact of a specific metric configuration. However, qualitative inspection reveals the trade-offs at extreme values:

- • Small n (n = 6): The metric becomes overly sensitive to trivial syntactic patterns. We observed that n = 6 frequently flags structural tokens (e.g., repeated LaTeX formatting like “\ \”) as redundancy, introducing noise into the reward signal.
- • Large n (n = 15): The state space becomes overly sparse. With longer windows, the Unique State Count remains naturally high even for semantically repetitive chains, making it harder to detect the "Shallow Exploration Trap."

We adopt n = 10 as the optimal balance, capturing semantic logical steps while filtering out low-level syntactic noise.

- 0.8k
- 1.6k
- 2.4k
- 3.2k

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

6.0k

96

56

- 3.0k

- 4.5k

88

48

80

40

1.5k

72

32

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

Training Step

Training Step

GSPO ΔL = 100 ΔL = 500 ΔL = 1k ΔL = 8k

Figure 14. Training dynamics of different ∆L

Impact of Lengthening (∆L). In the training stage, we further ablate the effect of the exploration increment limit ∆L, which governs the pace at which we force the model to expand its reasoning horizon. As illustrated in Figure 14, we observe that:

- • Moderate Extension (∆L = 100,500): These settings successfully activate the "capacity for exploration," yielding stable growth in both trajectory length and accuracy.
- • Aggressive Extension (∆L = 8k): Removing the progressive constraint (i.e., immediately demanding max length) destabilizes training. The model, unable to legitimately reason at such lengths instantly, collapses into degenerate repetition loops to satisfy the length reward, leading to a degradation in reasoning quality.

Impact of Repetition Threshold (Θ). We examine the sensitivity of the redundancy penalty threshold Θ. As shown in

- Table 6, our method achieves optimal performance at Θ = 10 (Avg 52.7%), surpassing both stricter (Θ = 6) and looser (Θ = 15) settings. Notably, the strict threshold (Θ = 6) leads to a performance drop on the challenging AIME25 benchmark (21.5%) compared to the baseline (22.2%). We speculate the reasons are twofold:

- • Over-penalization (Θ = 6): Mathematical reasoning requires a degree of natural repetition (e.g., restating variables or structural connectors like "therefore"). An overly strict threshold likely triggers false positives, penalizing valid reasoning steps and disrupting the logical flow required for complex problems.
- • Ineffective Constraint (Θ = 15): A loose threshold fails to sufficiently curb the "Shallow Exploration Trap." It allows

the model to "game" the length-incentivized reward (Rlen) by generating repetitive low-entropy sequences that satisfy the length requirement without achieving genuine state expansion, thereby reducing exploration efficiency.

Table 6. The Ablation study of Θ.

MATH Olympiad AIME AMC AIME25 Avg. GRPO w/Clip-Higher 86.4 54.1 25.2 61.8 22.2 49.9

+LIE (Θ = 6) 86.0 55.4 29.2 65.4 21.5 51.5 +LIE (Θ = 10) 88.8 54.1 30.4 65.5 24.5 52.7 +LIE (Θ = 15) 86.8 56.0 28.5 63.9 22.4 51.5

Impact of Repetition Magnitude β The redundancy penalty weight β (Equation 13) serves as a hyperparameter for balancing exploration capacity and efficiency. As illustrated in the training dynamics in Figure 15, we observe that the final reasoning accuracy is robust to the choice of β. Both β = 0.3 and β = 0.6 achieve nearly identical convergence in accuracy, indicating that the penalty magnitude does not affect the model’s fundamental problem-solving capability.

The primary effect of a higher β is the promotion of conciseness. As shown in the length (L) and entropy plots, β = 0.6 leads to shorter reasoning trajectories and a more rapid decrease in policy entropy compared to β = 0.3. This suggests that while a smaller β allows for more “loose” exploration, a larger β encourages the model to find more efficient, high-density reasoning paths to the correct answer.

###### L

###### Critic Acc

###### Entropy

###### ACC

| | | | | | |
|---|---|---|---|---|---|
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

0.60

6.0k

1.00

0.56

0.45

- 3.0k
- 4.5k

0.75

0.48

0.30

0.50

0.40

0.25

0.15

0.32

1.5k

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

Training Step

Training Step

beta=0.3 beta=0.6

Figure 15. Training dynamics of different β.

#### C.6. Ablation Study of the Reward Shaping Recipe

To understand the contribution of each component in our final recipe (Equation 14), we perform an exhaustive ablation study. As illustrated in Figure 16, the interaction between the length reward (Rlen) and the redundancy penalty (Rred) is key to unlocking effective in-context exploration.

- • Length reward (Rlen) as the capacity driver: (Proposition 4.1) The brown curves (GSPO + Rlen) demonstrate that explicitly incentivizing length successfully breaks the “Shallow Exploration Trap,” leading to a surge in the total

distinct state count Ccontext. However, this comes at a significant cost: the distinct ratio Rcontext plummets, and length becomes extremely large. This indicates that without constraints, the model tends to satisfy the length requirement through “thought padding” — generating repetitive or low-value tokens.

- • Redundancy reward (Rred) as the quality filter: Conversely, when only the redundancy penalty is added (red curves), the model maintains a high Rcontext, but the average trajectory length L remains low. This suggests that while the penalty prevents repetition, it does not provide the necessary drive to navigate deeper into the state space for complex reasoning.
- • Synergistic effect of the LIE recipe: Our complete recipe (GSPO + LIE, purple curves) achieves a superior balance. By combining Rlen and Rred, we create a synergistic effect: Rlen provides the computational budget to explore, while Rred ensures that this budget is spent on diverse and meaningful reasoning states. As a result, the model exhibits a steady growth in both Ccontext and length, while maintaining a healthy Rcontext. This structured exploration ultimately leads to the highest and most stable accuracy gains among all variants.

In summary, the LIE recipe is not merely a sum of its parts; it is a mechanism that converts raw sequence length into effective reasoning depth.

###### L

Ccontext(τ)

Rcontext(τ)

###### Accuracy

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

60

8.0k

- 1.8k
- 2.4k

80

50

6.0k

60

40

4.0k

- 0.6k
- 1.2k

40

30

2.0k

20

20

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

Training Step

Training Step

Training Step

Training Step

GSPO GSPO + LIE GSPO + Redundancy Penalty GSPO + Rlen

Figure 16. Training dynamics of different reward components.

### D. Case Study

To analyze the internal reasoning dynamics, we follow the principle of (Feng et al., 2025) and employ Claude-3.7-Thinking to extract the underlying reasoning graph structure from the model’s generated responses. This allows us to visualize and

quantify the "thought process" beyond simple token sequences or behaviors.

Qualitative Analysis. We select a representative problem from the AIME benchmark where the baseline fails while our method succeeds. As visualized in Figure 17a, the GSPO baseline (Left) follows a linear and shallow reasoning path. It attempts a direct derivation but fails to cross-check its intermediate steps, quickly converging to an incorrect answer (324).

In contrast, the model trained with our LIE recipe (Figure 17b) exhibits a significantly richer reasoning topology. Crucially, the length incentive activates in-context exploration behaviors: the model spawns alternative hypotheses (“Alternative Approach: Using Polar Form”), performs explicit self-verification (“Step 5: Verification”), and successfully identifies and corrects a calculation error (“Calculation Error”). This capacity to branch out and backtrack allows the model to recover from initial pitfalls and ultimately derive the correct solution (540).

Quantitative Structural Metrics To verify if this observation holds statistically, we compute the average Depth (maximum path length in the reasoning graph) and Width (average branching factor) across the 40 samples from AIME. As shown in

- Table 7, our method consistently expands the reasoning structure:

- • Increased Depth (13.8 → 14.75): The model constructs longer logical chains, enabling deeper decomposition of complex problems.
- • Increased Width (2.15 → 2.30): More importantly, the increased width indicates that the model is not merely “padding” the response with empty tokens. Instead, it engages in broader exploration by considering multiple parallel hypotheses or verification paths.

These structural metrics confirm that explicitly incentivizing response length effectively translates into a broader search horizon, enabling the model to navigate the state space more thoroughly.

Table 7. Exploration width and depth metrics.

Model Depth Width GSPO 13.8 2.15

GSPO + LIE 14.75 2.3

[Figure 1]

[Figure 2]

###### (a) GSPO (b) GSPO + LIE Figure 17. A case from AIME on GSPO and our recipe models.

