# arXiv:2601.08605v2[cs.CL]15Apr2026

## ExpSeek: Self-Triggered Experience Seeking for Web Agents

Wenyuan Zhang1,2, Xinghua Zhang3, Haiyang Yu3, Shuaiyi Nie1,2, Bingli Wu3, Juwei Yue1,2, Tingwen Liu1,2∗, Yongbin Li3* 1Institute of Information Engineering, Chinese Academy of Sciences 2School of Cyber Security, University of Chinese Academy of Sciences 3Tongyi Lab , Alibaba Group

[Figure 1]

{zhangwenyuan, liutingwen}@iie.ac.cn, {zhangxinghua.zxh, shuide.lyb}@alibaba-inc.com

### Abstract

Experience intervention in web agents emerges as a promising technical paradigm, enhancing agent interaction capabilities by providing valuable insights from accumulated experiences. However, existing methods predominantly inject experience passively as global context before task execution, struggling to adapt to dynamically changing contextual observations during agent-environment interaction. We propose ExpSeek, which shifts experience toward step-level proactive seeking: (1) estimating step-level entropy thresholds to determine intervention timing using the model’s intrinsic signals; (2) designing step-level tailored experience content. Experiments on Qwen3-8B and 32B models across four challenging web agent benchmarks demonstrate that ExpSeek achieves absolute improvements of 9.3% and 7.5%, respectively. Our experiments validate the feasibility and advantages of entropy as a self-triggering signal, reveal that even a smallscale 4B experience model can significantly boost the performance of larger agent models. The code is released at https://github.com/ WYRipple/ExpSeek.

### 1 Introduction

Advances in large language models (LLMs) are gradually unlocking greater potential for agents (Team et al., 2025; Qu et al., 2025; Zhang et al., 2025h). Recently, web agents powered by search engines have gained considerable attention for their capability to retrieve relevant information from the web and address complex user queries (Ning et al., 2025; Song et al., 2025). Accordingly, the agent needs to possess the ability to conduct multi-turn interactions with the web to obtain evidence (Wei et al., 2025), and leverage it to provide answers to the user’s queries (Lu et al., 2024). However, the open web is noisy and partially observable with sparse useful evidence,

*corresponding authors.

- (A) Passive Injection
- (B) Active Seeking (Ours) Experience

[Figure 2]

[Figure 3]

[Figure 4]

Web Agent

Experience Base

Thinking Tool Call

Thinking Tool Call

Thinking Answer

...

Tool Response

Query

Web Env

Step 2 Step N

Step 1

Guidance

model

###### No Trigger

###### No Trigger

Trigger !

Seek

[Figure 5]

[Figure 6]

[Figure 7]

| |
|---|
| |
| |

0.45

0.27

0.11

Experience Base

Thinking Tool Call

Thinking Tool Call

Thinking Answer

###### ...

Tool Response

Query

Web Env

Step 1

Step 2 Step N

Figure 1: Comparison of experience intervention frameworks. Panel A shows the traditional global passive injection of experience, while we extend the framework to Panel B, where the agent proactively seeks guidance at each step based on its own signals.

posing challenges to the agent’s reliability (Lee et al., 2025). Agents powered by LLMs, particularly small-scale, cost-effective models, often exhibit inefficient exploration in multi-turn interactions with the environment or respond prematurely, resulting in unreliable answers (Gao et al., 2025a).

As demonstrated by previous studies (Zhao

- et al., 2024; Zheng et al., 2024), agents, like humans (Silver and Sutton, 2025), can learn efficiently from experience, drawing on information accumulated from historical interaction trajectories. Existing methods mainly follow two lines: offline refinement, which post-processes trajectories into reusable patterns and retrieves them at inference time (Kim et al., 2024; Gao et al., 2025b); and online self-evolution, which accumulates experience through iterative interaction and feedback (Wang
- et al., 2025d; Liu et al., 2025d; Zhang et al., 2025a). Despite their effectiveness, the experience is often passively injected as a global context before the task execution, as shown in Figure 1 (A). However, during the agent’s interaction with the environment,

the contextual observations continuously change. If the agent relies solely on the initial context without proactively acquiring and integrating fresh experience, its decision-making may become suboptimal or even misaligned with the current situation. Compared with passively injecting experience, why not empower the agent to proactively seek experience during its interaction with the environment for more precise guidance?

This paper proposes ExpSeek, a self-triggered experience-seeking framework that clarifies when to seek experiences and which ones to seek, as briefly depicted in Figure 1 (B):

(1) The timing for seeking experience: Frequent or sparse experience seeking makes it difficult to achieve ideal results in terms of effectiveness and efficiency. An optimal timing for seeking experience is when the agent becomes confused and genuinely in need of guidance during interaction. To this end, we estimate a promising threshold interval for each step based on entropy value through logistic regression and bootstrap resampling, and use the estimated interval to control the timing of experience seeking. (2) The concrete content of experience: After determining the timing for seeking experience, the next step is to decide its concrete content. Firstly, we formalize an experience base from pairs of successful and failed trajectories. Each triplet contains the erroneous behavior, mistake analysis, and corrective cues, and is grouped by topic. Subsequently, the experience model reads the historical context of the current step, retrieves the related experience triplets from the experience base, and generates guidance tailored to the ongoing interaction.

We evaluate ExpSeek on four challenging web agent benchmarks using agents powered by Qwen3-8B and Qwen3-32B. ExpSeek respectively achieves 9.3% and 7.5% absolute improvements over the 8B and 32B base models, and outperforms passive experience injection baselines by 6.7% and 6.0%, validating the effectiveness of proactive experience seeking during interaction with environment. Extensive analyses reveal some key insights: (1) The entropy of a model itself is an effective trigger signal and can indicate whether it should receive experience guidance, controlling the timing for seeking experience. (2) Even with a small-scale 4B model, the proactive experience seeking paradigm of ExpSeek can still yield significant performance gains for a slightly larger 32B model. (3) The selftriggered experience seeking increases the entropy

of the model during the intermediate steps of interaction with the environment, while decreasing entropy at the final answer step, confirming the enhanced exploration and ultimately enabling more effective convergence toward the correct answer.

In summary, our contributions are as follows:

- • We propose ExpSeek, a self-triggered experience seeking framework, inspiring a proactive paradigm for seeking experience distinct from passive experience injection.
- • We explore and confirm that the entropy of a model itself can serve as an intrinsic signal of the timing for proactively seeking experience. Additionally, we build the experience base with experience triplets, and an experience model is designed to dynamically generate experience guidance during agent–environment interactions, based on experience triplets and historical context.
- • Extensive experiments and analyses show the significant advantages of ExpSeek with average improvements of 9.3% and 7.5% on 8B/32B models, an increase of up to 14.6%.

### 2 Related Work

In this section, we discuss the most relevant lines of work. A broader discussion of related work is provided in Appendix D.

#### 2.1 Experience Intervenes in Agents

Experience (Silver and Sutton, 2025) serves as long-term memory to prevent repeated mistakes and accumulate insights, distinct from short-term contextual memory (Hu et al., 2025).

Recent work can be categorized into two lines: (1) Offline Experience refines experience from offline training sets. Early methods directly use raw trajectories (Zheng et al., 2024), while later work structures experience (Zhao et al., 2024; Kim et al., 2024; Fang et al., 2025a; Kirtania et al., 2025) and induces patterns from successful and failed trajectories (Cai et al., 2025a) to extract transferable (Gao et al., 2025b; Tang et al., 2025) and reusable reasoning units. (2) Self-Evolution accumulates domain-specific experience online through gradient-free training (Cai et al., 2025b) or by shifting the model’s output distribution (Luo et al., 2025), with real-time updates (Wang et al., 2025d; Liu et al., 2025d; Yang et al., 2025; Cao et al., 2025; Zhang et al., 2025a; Ouyang et al., 2025; Cui et al., 2025). However, such passive experience injection

is difficult to align with step decisions. 2.2 Entropy in LLM Reasoning

Entropy serves as a fundamental uncertainty metric widely used for static reasoning evaluation, including answer selection (Ren et al., 2023; Raj et al.,

- 2025), confidence calibration (Chen and Mueller,

- 2024), and error detection (Farquhar et al., 2024; Liu et al., 2025c). As reasoning scales to multi-step paradigms where responses are decomposed into atomic steps (Zhang et al., 2025e; Nie et al., 2026; Ai et al., 2026), the role of entropy extends to training. Particularly in reinforcement learning, entropy not only reflects sampling diversity to facilitate exploration (Wang et al., 2025c; Zheng et al., 2025), but also acts as a fine-grained signal for step-level credit assignment in long-horizon reasoning (Wang et al., 2025b,a), and further extends to incentivize exploration across multi-turn interactions (Dong et al., 2025). Recent work also investigates entropy collapse during RLVR training and proposes mechanisms to regulate entropy dynamics (Jin et al.,

- 2026). Confidence-based triggering has been explored in retrieval-augmented settings (Ding et al., 2026). This demonstrates the potential of entropy as a self-trigger signal for web agents. 3 Preliminaries

Agent Framework. Following Wu et al. (2025b), we construct web agents based on the ReAct framework (Yao et al., 2023), modeling problem-solving as an interleaved sequence of reasoning and acting. A trajectory τ is defined as:

##### τ = (q,R1,O1,...,Rt,Ot,...,RT), (1)

where q is the query, Rt is the agent’s response at step t, and Ot is the environment observation. We distinguish two types of steps:

- • Process Step (Stp = (Rt,Ot), t < T): The response Rt = ⟨zt,at⟩ contains reasoning thoughts zt (enclosed in <think> tags) and an action at (in <tool_call> tags), followed by the tool’s response Ot.
- • Answer Step (STa = RT): The terminal step produces RT = ⟨zT,y⟩, where the final answer y is wrapped in <answer> tags.

Step Entropy. To quantify the agent’s confidence in each step of interaction, we compute the step entropy as the average token entropy across each response Rt. Specifically, the entropy of the i-th token xi with preceding context hi is defined as

H(xi) = − v∈V P(v | hi)log P(v | hi), where V is the vocabulary and P(· | hi) is the model’s predicted distribution. Step entropy is computed:

1 |Rt| x∈R

H¯t =

t

##### H(x), (2)

where |Rt| denotes the number of tokens in Rt.

Experience Intervention. We formally define experience intervention as a class of methods comprising two phases:

- • Construction Phase: Given a training corpus

Dtrain = {(qi,τi,yi)}Ni=1 with queries qi, trajectories τi, and ground truth yi, experience acqui-

sition is formalized as E = F(Dtrain), where F is a function that extracts an experience base E from Dtrain1.

- • Utilization Phase: At step t with context ht = (q,...,Rt,Ot), the agent obtains applicable experience et=G(E,ht) through mapping function G, which serves as additional input for reasoning.

Traditional methods represent a special case where experience is concatenated at the beginning of reasoning, with e = G(E,q). The function G typically returns the entire base or retrieves relevant cases, providing static experience. In contrast, our method invokes G(E,ht) at any step t to allow the agent to seek appropriate guidance.

### 4 Methodology

This section formally introduces ExpSeek, elaborating on three key components: experience base construction (§ 4.1), self-triggering mechanism (§ 4.2), and step-level guidance (§ 4.3).

#### 4.1 Experience Base Construction

The core of guiding experience lies in recreating the problem behavior and simulating guidance.

Guiding Experience Schema. We design experience triplets containing: (1) Behavior: objectively describes the state and action at the current step; (2) Mistake: identifies errors by contrasting with correct trajectories; (3) Guidance: provides directional guidance based on error analysis, without directly offering answers or specific clues.

Construction Process. As shown in Figure 2 (A), construction involves three steps. First, for each query in Dtrain, we sample k trajectories with the agent model and pair successful and failed

1Self-evolution methods have the same Dtrain and Dtest, but typically do not provide ground truth y.

(a) Success-Failure Trajectory Collection

|<think>Based on the curent information, I cannot find the specific number, so the answer is uncertain.</think> <answer>uncertain</answer>|
|---|

|How many of the top 50 most-subscribed YouTube channels are based in India?|
|---|

√ √

- Trajectory 1

- Trajectory 2

Query

...

Training Dataset

Step Entropy

×

Trajectory n

Thinking Tool Call

... Guidance Thinking Tool Call

Thinking Answer

...

- (b) Experience Triplet Construction

...

- (c) Triplet Topic Generation

Tool Response

Query

[Figure 8]

[Figure 9]

[Figure 10]

Triplet Set

Continue Interaction

###### Seek Experience

[Figure 11]

Select Topics

|Topic 1: High-relevance evidence ignored<br>Topic 2: Information misjudgment exists<br><br><br>...|
|---|

Experience model

[Figure 12]

[Figure 13]

|Based on prior guidance experience, the agent encountered some issues . Therefore, it should be encouraged to focus on . page and further invoke tools<br><br>Guidance: You should pay more attention to the previously searched ..., and further collect clues, rather than directly answer ...|
|---|

Experience model

Triplet subset 1

Triplet subset 2

Utilize Experience

Experience Base

A. Experience Base Construction B. Seek Guiding Experience during Inference

- Figure 2: The overall architecture of ExpSeek, including experience base construction and actively seeking experience guidance during inference. The step entropy threshold calculation process is not depicted here.

ones. Second, for each trajectory pair (τ+,τ−), we use a tool model to analyze the failed trajectory τ− = {S1,S2,...,ST} by contrasting with the successful one τ+. The tool model assigns each step St a binary correctness label yt ∈ {0,1} and outputs triplets for incorrect steps (where yt = 0). Finally, we prompt the tool model to induce topics for triplets using an iterative batch processing approach: when processing each new batch, the model takes all previously generated triplets with their assigned topics as input, then either assigns existing topics, modifies them, or creates new topics for the current batch. This yields a guiding experience base E organized into topics, with separate collections Ep and Ea for process and answer steps respectively.

- 4.2 Entropy as Self-Trigger

- 4.2.1 Entropy Analysis for Step Correctness

Determining when to trigger intervention is challenging. Using reward models to analyze each step incurs prohibitive costs. Inspired by prior research on entropy-based error detection and confidence calibration (Chen and Mueller, 2024; Liu et al.,

- 2025c), we hypothesize that the step entropy of web agents has the potential to reflect the agent’s inherent state. We focus on (1) whether entropy can distinguish correct from incorrect steps in web agent reasoning, and (2) whether this distinguishability differs between process and answer steps.

We analyze training trajectories from § 4.1. Based on correctness labels yt assigned during trajectory pairing, we construct two step collections:

S+ = {St ∈ τ+} ∪ {St ∈ τ− | yt = 1}, S− = {St ∈ τ− | yt = 0},

(3)

where S+ aggregates all steps from successful trajectories and correctly-labeled steps from incorrect trajectories, while S− comprises only the incorrect steps from failed trajectories. We further partition these collections by step type into process steps (Sp+,Sp−) and answer steps (Sa+,Sa−), then analyze their entropy distributions H¯t.

Figure 3 shows that S+ consistently exhibits lower entropy than S− for both step types. The Kolmogorov-Smirnov test (Berger and Zhou, 2014) confirms that the distributions of correct and incorrect steps are statistically separable for both types (process steps: KS=0.1998, p<0.001; answer steps: KS=0.3809, p<0.001). However, separability differs substantially in practice: process steps show considerable overlap as agents naturally explore multiple paths, producing high entropy even when correct (AUC=0.6223, indicating weak discrimination (Bradley, 1997)), while answer steps demonstrate much clearer separation (AUC=0.7187, indicating acceptable discrimination). This suggests that entropy serves as a noisy but valid signal for process steps, and becomes more reliable for answer steps, motivating us to estimate thresholds for triggering intervention.

#### 4.2.2 Threshold-Based Triggering

Threshold Estimation via Bootstrap. To quantify when intervention should occur, we estimate threshold intervals that capture uncertainty in distinguishing correct from incorrect steps. We formulate this as binary classification: given step entropy H¯t, predict correctness yt ∈ {0,1}.

We fit separate logistic regression models for process steps (Sp+,Sp−) and answer steps (Sa+,Sa−),

(a) Process Steps

| | |
|---|---|
| | |
| |Safe Zone Buffer Zone<br><br>|
|| |
|---|
| |
| |Intervention Zone Median Probability Curve<br><br>|
|| |
|---|
<br><br>| |
| || |
|---|
<br><br>True Dist. False Dist.<br><br>| |
|---|
<br><br>|
| | |
| | |
| | |
| | |

1.0

ProbabilityofIntervention

- 0
- 1
- 2
- 3
- 4

0.8

SampleDensity

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Simulated Thresholds

Frequency

Lower 2.5%

100

Median

Upper 97.5%

0

0.30 0.35 0.40 0.45 0.50

Entropy Value

(b) Final Steps

| | |
|---|---|
| | |
| |Safe Zone<br><br>|
|| |
|---|
| |
| |Buffer Zone Intervention Zone Median Probability Curve<br><br>|
|| |
|---|
| |
| |True Dist. False Dist.<br><br>|
|| |
|---|
<br><br>| |
|---|
| |
| | |
| | |
| | |

1.0

- 0
- 1
- 2
- 3
- 4

ProbabilityofIntervention

0.8

SampleDensity

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8

Simulated Thresholds

100

Frequency

Lower 2.5%

Median

Upper 97.5%

0

0.20 0.21 0.22 0.23 0.24 0.25 0.26

Entropy Value

- Figure 3: Entropy distributions of process and answer

steps on Dtrain for Qwen3-8B, with fitted logistic regression curves. Green zone indicates no intervention during inference, red indicates intervention, and yellow indicates probabilistic intervention.

Model θlowp θhighp θlowa θhigha

Qwen3-8B 0.314 0.413 0.225 0.257 Qwen3-32B 0.877 1.384 0.714 0.820

Table 1: Threshold intervals obtained through bootstrap resampling for process and answer steps.

Probabilistic Intervention at Inference. During inference, for each step St, we compute its entropy H¯t and determine the intervention probability based on the corresponding threshold interval (using [θlowerp ,θupperp ] for process steps or [θlowera ,θuppera ] for answer steps):

 

- 0 H¯t < θlower H¯t−θlower

θupper−θlower θlower ≤ H¯t ≤ θupper

- 1 H¯t > θupper

pintervene =



(6) We trigger experience guidance (§ 4.3) with probability pintervene, where low-entropy steps receive no intervention, high-entropy steps (indicating low confidence) always receive guidance, and intermediate cases are handled probabilistically to balance intervention frequency with agent autonomy.

where each model learns:

1 1 + e−(w·H¯t+b), (4)

P(yt = 0 | H¯t) =

modeling the probability of incorrectness, where higher entropy corresponds to higher error probability. The decision boundary at P = 0.5 yields threshold θ = −b/w.

Since a single fitted model on limited data may be unstable, we employ bootstrap resampling: we sample with replacement from the respective step collections (preserving original sizes) N times, fit logistic regression on each sample to obtain {θ(i)}Ni=1, and compute the 95% confidence interval as our threshold range:

θlower = Q0.025({θ(i)}), θupper = Q0.975({θ(i)}),

(5)

where Qp denotes the p-th quantile. This yields separate threshold intervals [θlowerp ,θupperp ] for process steps and [θlowera ,θuppera ] for answer steps. More implementation details are provided in Appendix A.1.

#### 4.3 Guided Intervention at Inference

At inference, we implement G(E,ht) through a experience model Me that dynamically generates contextualized interventions. The process is illustrated in Figure 2 (B).

For each step St, we first compute H¯t and sample from pintervene (§ 4.2). If triggered and no intervention occurred at the previous step, Me selects the three most relevant topics from Ep (for process steps) or Ea (for answer steps) based on context ht, then adapts experiences under selected topics to generate guidance et for the current situation.

The generated guidance is injected differently by step type. For process steps Stp, et is appended to Ot. For answer steps STa, we extend it to {RT,OT} with et as OT, enabling the agent to continue at step T + 1 (either refining the answer or invoking tools for further reasoning).

To prevent over-intervention, we disable intervention at step t + 1 after any intervention at step t, allowing the agent to incorporate guidance before receiving further intervention.

WebWalkerQA

GAIA SEAL xbench Avg. Easy Medium Hard Avg.

Method

Qwen3-8B No Experience 42.67-0.0 40.19-0.0 32.56-0.0 38.47-0.0 29.13-0.0 23.23-0.0 25.60-0.0 32.23-0.0 Training-Free GRPO 40.66↓2.0 42.86↑2.7 38.33↑5.8 40.62↑2.2 29.32↑0.2 25.59↑2.4 26.00↑0.4 33.79↑1.6 REASONINGBANK+ 44.50↑1.8 41.52↑1.3 36.33↑3.8 40.78↑2.3 32.04↑2.9 26.38↑3.2 28.00↑2.4 34.80↑2.6 ExpSeek (Ours) 51.00↑8.3 49.52↑9.3 44.22↑11.7 48.25↑9.8 36.89↑7.8 30.16↑6.9 37.20↑11.6 41.50↑9.3

- - only process step 44.17↑1.5 47.91↑7.7 38.89↑6.3 43.66↑5.2 33.01↑3.9 27.17↑3.9 28.40↑2.8 36.59↑4.4
- - only answer step 49.50↑6.8 48.48↑8.3 41.22↑8.7 46.40↑7.9 30.29↑1.2 30.08↑6.9 34.80↑9.2 39.06↑6.8 Qwen3-32B No Experience 46.67-0.0 47.24-0.0 41.11-0.0 45.01-0.0 36.50-0.0 27.80-0.0 27.40-0.0 37.79-0.0 Training-Free GRPO 46.92↑0.3 46.85↓0.4 41.67↑0.6 45.15↑0.1 36.89↑0.4 28.19↑0.4 28.20↑0.8 38.12↑0.3 REASONINGBANK+ 47.50↑0.8 47.62↑0.4 41.67↑0.6 45.60↑0.6 33.01↓3.5 29.84↑2.0 36.33↑8.9 39.33↑1.5 ExpSeek (Ours) 54.33↑7.7 53.05↑5.8 45.89↑4.8 51.09↑6.1 43.88↑7.4 32.76↑5.0 42.00↑14.6 45.32↑7.5
- - only process step 49.17↑2.5 47.24±0.0 45.56↑4.5 47.32↑2.3 40.58↑4.1 31.50↑3.7 30.80↑3.4 40.81↑3.0
- - only answer step 50.83↑4.2 48.57↑1.3 42.78↑1.7 47.39↑2.4 37.11↑0.6 32.52↑4.7 35.40↑8.0 41.20↑3.4

- Table 2: Main results of ExpSeek on four benchmarks using two backbone agents. We report mean accuracy (%) over five independent runs and absolute improvements over vanilla ReAct without experience. We also provide full ablation results for guiding only process or answer steps.

### 5 Experiments

#### 5.1 Setup

Benchmarks and Metrics. We evaluate on four challenging real-world web agent reasoning benchmarks: GAIA (Mialon et al., 2023), WebWalkerQA (Wu et al., 2025b), xbenchDeepSearch (Chen et al., 2025), and SEALHARD (Pham et al., 2025). We construct our training set by sampling 25% from each difficulty level (easy, medium, hard) in WebWalkerQA with random seed 0, yielding 170 examples, with the remainder used as the test set. We employ the LLMas-a-Judge approach for evaluation and report average accuracy across five independent runs.

Tool Environment. Agents are equipped with two fundamental tools: (1) Search, which queries a search engine to return relevant URLs with snippets; and (2) Visit, which accesses a specific URL to retrieve its content.

Configuration. We set the maximum number of ReAct steps to 30, treating episodes exceeding this limit as failures. We use Qwen3-8B and Qwen3-32B as agents with sampling temperature of 1.0 and top-p of 0.95. For the tool model and Me, we employ Qwen3-235B-A22B-Instruct-2507 in the main experiments. During experience construction, trajectories are sampled five times, and bootstrap sampling uses N = 1000. The constructed experience repositories contain: for 8B, |Ep| = 196 (17 topics) and |Ea| = 190 (11 topics); for 32B, |Ep| = 276 (18 topics) and |Ea| = 143 (23 topics). The threshold intervals derived from bootstrap resampling are shown in Table 1.

Baseline. We select two typical experience-based methods as baselines. Training-Free GRPO (Cai et al., 2025a) uses semantic advantages to iteratively acquire and update high-quality experiences from offline trajectories, globally leveraging the experience repository at test time. REASONINGBANK (Ouyang et al., 2025) is a self-evolving experience acquisition scheme that accumulates experiences from online tasks and retrieves them from a continuously updated experience repository in the system prompt during subsequent reasoning. We implement an enhanced version REASONINGBANK+ using 235B instead of a weaker reasoning agent to generate experiences. All experimental and setup details are provided in Appendix C, E.

#### 5.2 Main Results

Table 2 reports the main experimental results. The key findings are as follows:

- (1) ExpSeek demonstrates significantly superior performance. ExpSeek achieves average absolute improvements of 9.3% and 7.5% over vanilla ReAct on Qwen3-8B and 32B respectively, substantially outperforming baselines across all benchmarks and highlighting the advantages of step-level experience guidance.
- (2) Global intervention methods show limited effectiveness. Both baseline types struggle with challenging open web reasoning tasks, showing improvements under 3% or even performance degradation. This indicates that carefully designed global experience injection struggles to adapt to noisy web environments while adding reasoning burden to smaller agents.

WebWalker-Easy

###### WebWalker-Medium

WebWalker-Hard

###### GAIA

###### SEAL

###### xbench

| | |Vanilla| |
|---|---|---|---|
| | |ExpSeek| |
| | | | |
| | | | |
| | | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla ExpSeek|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla<br><br>ExpSeek|
| | | |
| | | |
| | | |
| | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

5.0

5.0

3.0

6.0

(a)ProcessStep(b)FinalStep

4.0

4.0

4.0

4.0

2.5

5.0

Density

3.0

3.0

2.0

3.0

3.0

4.0

1.5

3.0

2.0

2.0

2.0

2.0

1.0

2.0

1.0

1.0

1.0

1.0

0.5

1.0

0.0

0.0

0.0

0.0

0.0

0.0

0.2 0.4 0.6 0.8

0.2 0.4 0.6

0.2 0.4 0.6

0.2 0.4 0.6

0.1 0.2 0.3 0.4 0.5

0.2 0.4 0.6 0.8

4.0

3.5

| | | |
|---|---|---|
| | |Vanilla|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | |Vanilla|
| | | |ExpSeek|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla<br><br>ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

4.0

2.5

4.0

3.5

5.0

3.0

3.5

3.5

3.0

2.0

3.0

4.0

2.5

3.0

###### Density

2.5

2.5

2.5

2.0

1.5

3.0

2.0

2.0

2.0

1.5

1.5

1.0

1.5

2.0

1.5

1.0

1.0

1.0

1.0

0.5

1.0

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8 1.0

Entropy

Entropy

Entropy

Entropy

Entropy

Entropy

- Figure 4: Entropy distributions of process and answer steps for Qwen3-8B before and after applying ExpSeek across all benchmarks. Results for Qwen3-32B are provided in Figure 9.

Method Acc. ↑ Rej. ↑ Step ↓ Time ↓ Qwen3-8B GAIA

No Experience 29.13 - 4.47 66.94s

Rule + Me 38.81 0.0 9.52 329.71s Claude-4 + Me 39.47 7.4 8.55 370.82s

- Entropy + Me 36.89 67.8 5.75 127.57s Entropy + emb 30.92 74.2 5.54 110.61s

Qwen3-8B xbench No Experience 25.60 - 3.31 51.06s

Rule + Me 36.40 0.0 8.52 300.09s Claude-4 + Me 36.60 4.8 7.40 315.39s

- Entropy + Me 37.20 42.2 5.73 143.81s Entropy + emb 27.80 44.4 4.69 88.17s

- Table 3: Performance and efficiency with different triggers and retrieval-based guidance in GAIA and xbench. Rej: the proportion of trigger checks that result in nonintervention; Step & Time: average per question.

(3) Strong cross-task generalization capability. Despite being derived entirely from the WebWalkerQA training set, ExpSeek maintains robust performance across three out-of-distribution benchmarks.

We also report pass@3 performance, demonstrating ExpSeek’s competitive sampling diversity. Additionally, we provide extensive case studies to intuitively illustrate the effectiveness of our method. Supplementary materials and further discussions can be found in Appendix B.

#### 5.3 Ablation Study

To validate the individual effectiveness of guiding process and answer steps, we report complete ablation results in Table 2. Guiding only process or answer steps fails to surpass full intervention, showing significant performance drops on both 8B (-4.91%, -2.44%) and 32B (-4.51%, -4.12%). Notably, guiding only answer steps achieves performance closer to the full method, which validates our observation in § 4.2 that answer steps exhibit

###### GAIA

xbench

50

No Guidance 4B Guidance 30B Guidance

No Guidance 4B Guidance 30B Guidance 235B Guidance

45

| |
|---|

| |
|---|

42.0

| |
|---|

| |
|---|

43.9

45

235B Guidance

| |
|---|

41.8 42.3

39.3

40

37.2

37.0

7.3%

40

Accuracy(%)

Accuracy(%)

5.2% 5.8%

36.9

34.3

35.9

35

14.7%

32.3

12.0%

35

33.3

9.7%

11.6%

6.8% 7.8%

30

8.7%

4.2%

6.7%

30

25

25

20

Qwen3-8B Qwen3-32B

Qwen3-8B Qwen3-32B

Figure 5: Scaling Law of experience model Me.

stronger distributional distinguishability.

### 6 Experimental Analyses

In this section, we conduct an in-depth analysis of ExpSeek across four dimensions: internal mechanisms, design rationale, scalability, and efficiency.

#### 6.1 How Does ExpSeek Work Internally?

To reveal how ExpSeek works internally, we visualize the entropy distribution shifts of agent outputs before and after experience guidance (Figure 4). The results show that guidance increases entropy in process steps, enabling the agent to escape local decisions and explore broader reasoning spaces; conversely, the entropy distribution of answer steps shifts toward lower values with a sharper peak, indicating the agent converges to correct answers with higher confidence. This diverge-then-converge behavior balances exploration and exploitation in complex reasoning.

#### 6.2 Why Self-Trigger and Generation?

To validate entropy-based self-triggering, we compare two variants: Reward Model-based2 (RM) and

2We employ claude-sonnet-4-20250514 to judge intervention necessity at each step based on the full history.

Accuracy (%)

45.0

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

[Figure 14]

[Figure 15]

70

GuidanceInterval

No Experience Acc ExpSeek Accuracy Guidance Times per Sample

- 0

- 1

- 2

- 42.07 41.75 40.78 37.55 36.25 35.28 33.98
- 43.01 42.07 38.89 36.89 35.60 36.25 31.72

42.5

60

GuidanceTimesperSample

50

40.0

40

40.78 39.75 37.66 35.28 34.30 33.98 31.07

Accuracy(%)

30

37.5

###### Average Steps

35.0

[Figure 16]

[Figure 17]

GuidanceInterval

12

- 0

- 1

- 2

10.94 8.61 7.46 6.57 5.50 5.66 5.01

32.5

10

30.0

7.30 7.37 6.53 5.75 5.66 5.14 4.66

8

6

27.5

6.02 5.95 5.72 5.43 5.26 4.77 4.76

4

-0.15 -0.1 -0.05 0 0.05 0.1 0.15 Threshold Shift

4.74.8 5.05.15.35.45.5 5.8 6.0 6.56.6 7.37.47.5 8.6 10.9

Average Steps

Figure 6: Cross-comparison results of performance and efficiency after adjusting intervention intensity.

Method GAIA xbench

Qwen3-8B ← E-8B / 32B 36.89 / 35.60 37.20 / 36.00 Qwen3-32B ← E-32B / 8B 43.88 / 40.33 42.00 / 37.20

- Table 4: Experience repository swapping: E-8B/32B denote dedicated repositories built for each agent.

###### GAIA

xbench

400

400

Vanilla Baseline

Vanilla Baseline

42

40

Accuracy Final Step Exp Process Step Exp

Accuracy

350

350

ExperienceRepositorySize

Final Step Exp Process Step Exp

40

| |
|---|

38

| |
|---|

| |
|---|

300

300

38

Accuracy(%)

250

250

36

36

34

200

200

34

32

150

150

32

30

100

100

28

30

50

50

26

| | |
|---|---|
| | |
| | |

28

0

0

1/1 exp 1/2 exp 1/4 exp 1 exp 0 exp

1/1 exp 1/2 exp 1/4 exp 1 exp 0 exp

Experience Ratio

Experience Ratio

Figure 7: Correlation between repository size and performance on Qwen3-8B.

rule-based (continuous intervention from step one), both with one-step post-trigger silence.

Table 3 shows similar accuracy but divergent efficiency. Rule-based triggers result in 1.7× step and 2.6× time overhead on GAIA (1.5× and 2.1× on xbench) compared to entropy-based triggering. RM reduces step overhead (1.3-1.5×) but increases time (2.2-2.9×), suggesting over-intervention. In contrast, self-trigger balances efficiency and performance by adapting intervention intensity to problem difficulty. On the more challenging xbench, its trigger rate increases 25.6% compared to GAIA while maintaining similar accuracy, confirming its ability to precisely identify intervention timing based on internal state.

We also explore retrieval-based guidance, selecting the most similar experience (via text-embedding-v43) instead of generation. Table 3 shows that retrieval-based guidance yields

3https://www.alibabacloud.com/help/en/modelstudio/embedding

substantially lower accuracy than generative guidance, confirming the necessity of generative guidance.

#### 6.3 Does ExpSeek Scale and Transfer?

We experiment with smaller 4B and 30B models4

as Me to explore the scaling of intervention models. As shown in Figure 5, the three model sizes exhibit a scaling law on both GAIA and xbench, all achieving substantial performance gains. Notably, the 4B guidance model improves the 32B agent by 5.2% and 9.7% points respectively, validating the feasibility of weak-to-strong guidance given reasonable experience. Additionally, we swap the experience pools between 8B and 32B Agents. Results in Table 4 show that while experience has model dependency, the abstract guidance knowledge it contains still holds transfer value.

Furthermore, Figure 7 shows we reduce experiences per topic. Even with just one experience per topic, Me still identifies key intervention points and maintains high accuracy, demonstrating that the experience model can understand and generalize to current reasoning even with only a few high-quality seed experiences, highlighting the necessity of experience topics. While removing the repository entirely degrades performance, the remaining substantial accuracy indicates that Me’s inherent world experience alone is beneficial.

#### 6.4 What Is the Efficiency Trade-off?

Figure 6 illustrates the relationship between accuracy and reasoning steps under varying intervention frequencies and thresholds. We expand the intervention interval from 1 to {0,1,2} and shift the threshold by ± 0.05 three times, yielding 21 configurations. We conduct experiments with Qwen3-8B on the GAIA. Results show that

4Qwen3-4B-Instruct and Qwen3-30B-A3B-Instruct

as the trigger threshold decreases, reasoning steps increase rapidly while accuracy rises then plateaus with diminishing returns. With ~2 interventions, accuracy reaches 43.01%; beyond 6 interventions, performance barely improves, indicating that while increasing intervention intensity does not degrade performance, it also fails to yield higher gains. The results also demonstrate that in practice, the web agent produces stable performance even when thresholds fluctuate at a small scale.

### 7 Conclusion

We propose ExpSeek, a framework enabling web agents to actively seek step-level guidance based on step entropy. Even with small open-source agents, ExpSeek demonstrates significant performance improvements and excellent generalization on complex real-world web reasoning tasks, showing great potential for future development.

### Limitations

Although ExpSeek achieves significant performance advantages, it has the following limitations for future research: (1) The current threshold estimation relies on the training set and the tool model’s assessment of step quality; more accurate strategies need to be investigated. (2) It remains unexplored whether ExpSeek can extend to nonweb domains and integrate more tools. (3) Given ExpSeek’s improvements in pass@k performance, it has not yet been studied whether it can serve as a rollout enhancement technique for Agentic Reinforcement Learning to improve convergence speed and sampling quality.

### Ethical Considerations

Our method is intended for academic research only and does not support applications involving risks, religion, racial discrimination, or ethical violations.

### Acknowledgments

We would like to thank the anonymous reviewers, the meta-reviewer, as well as the area chairs and program chairs for their valuable comments and efforts. We would like to thank the members of IIE KDSec and Alibaba Tongyi Lab for their valuable discussions and exchanges. This work is supported by the National Natural Science Foundation of China (Grant No. 62572465).

### References

Zhengyang Ai, Zikang Shan, Xiaodong Ai, Jingxian Tang, Hangkai Hu, and Pinyan Lu. 2026. Shape: Stage-aware hierarchical advantage via potential estimation for llm reasoning. Preprint, arXiv:2604.06636.

Sikai Bai, Haoxi Li, Jie Zhang, Yongjiang Liu, and Song Guo. 2026. Ttvs: Boosting self-exploring reinforcement learning via test-time variational synthesis. arXiv preprint arXiv:2604.08468.

Vance W Berger and YanYan Zhou. 2014. Kolmogorov– smirnov test: Overview. Wiley statsref: Statistics reference online.

Andrew P Bradley. 1997. The use of the area under the roc curve in the evaluation of machine learning algorithms. Pattern recognition, 30(7):1145–1159.

Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Yong Mao, Ke Li, and Xing Sun. 2025a. Training-free group relative policy optimization. Preprint, arXiv:2510.08191.

Zhicheng Cai, Xinyuan Guo, Yu Pei, Jiangtao Feng, Jinsong Su, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, and Hao Zhou. 2025b. Flex: Continuous agent evolution via forward learning from experience. Preprint, arXiv:2511.06449.

Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. 2025. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. Preprint, arXiv:2512.10696.

Jiuhai Chen and Jonas Mueller. 2024. Quantifying uncertainty in answers from any language model and enhancing their trustworthiness. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5186–5200, Bangkok, Thailand. Association for Computational Linguistics.

Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Haotong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang, Hongzhang Liu, Yuan Gong, Chen Sun, Han Hou, Hui Yang, James Pan, Jianan Lou, Jiayi Mao, Jizheng Liu, Jinpeng Li, Kangyi Liu, and 14 others. 2025. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations. Preprint, arXiv:2506.13651.

Xi Chen, Wei Xue, and Yike Guo. 2026. Actormind: Emulating human actor reasoning for speech roleplaying. Preprint, arXiv:2604.11103.

Xi Chen and Min Zeng. 2025. Prototype conditioned generative replay for continual learning in NLP. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Albuquerque, New Mexico. Association for Computational Linguistics.

Zihao Cheng, Zeming Liu, Yingyu Shan, Xinyi Wang, Xiangrong Zhu, Yunpu Ma, Hongru Wang, Yuhang Guo, Wei Lin, and Yunhong Wang. 2026. Mem2evolve: Towards self-evolving agents via coevolutionary capability expansion and experience distillation. Preprint, arXiv:2604.10923.

Sijia Cui, Aiyao He, Shuai Xu, Hongming Zhang, Yanna Wang, Qingyang Zhang, Yajing Wang, and Bo Xu. 2025. Self-guided function calling in large language models via stepwise experience recall. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 10842–10854, Suzhou, China. Association for Computational Linguistics.

Guoxuan Ding, Yuqing Li, Ziyan Zhou, Zheng Lin, Daren Zha, and Jiangnan Li. 2026. Exdr: Explanation-driven dynamic retrieval enhancement for multimodal fake news detection. arXiv preprint arXiv:2601.15820.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025. Agentic reinforced policy optimization. Preprint, arXiv:2507.19849.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2025a. Memp: Exploring agent procedural memory. Preprint, arXiv:2508.06433.

Zhen Fang, Zhuoyang Liu, Jiaming Liu, Hao Chen, Yu Zeng, Shiting Huang, Zehui Chen, Lin Chen, Shanghang Zhang, and Feng Zhao. 2025b. Dualvla: Building a generalizable embodied agent via partial decoupling of reasoning and action. arXiv preprint arXiv:2511.22134.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630.

Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. 2025a. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous RL. In First Workshop on Multi-Turn Interactions in Large Language Models.

Jinglong Gao, Xiao Ding, Lingxiao Zou, Bibo Cai, Bing Qin, and Ting Liu. 2025b. ExpeTrans: LLMs are experiential transfer learners. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10577–10616, Vienna, Austria. Association for Computational Linguistics.

Yu Guo, Shenghao Ye, Shuangwu Chen, Zijian Wen, Tao Zhang, Qirui Bai, Dong Jin, Yunpeng Hou, Huasen He, Jian Yang, and 1 others. 2026. Rethinking table pruning in tableqa: From sequential revisions to gold trajectory-supervised parallel search. arXiv preprint arXiv:2601.03851.

Ruiyan Han, Zhen Fang, XinYu Sun, Yuchen Ma, Ziheng Wang, Yu Zeng, Zehui Chen, Lin Chen, Wenxuan Huang, Wei-Jie Xu, and 1 others. 2026. Unicorn: Towards self-improving unified multimodal models through self-generated supervision. arXiv preprint arXiv:2601.03193.

Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, and 1 others. 2025a. Mimo-embodied: X-embodied foundation model technical report. arXiv preprint arXiv:2511.16518.

Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, and Jiawei Chen. 2025b. Rethinking entropy interventions in rlvr: An entropy change perspective. arXiv preprint arXiv:2510.10150.

Zhezheng Hao, Hong Wang, Jian Luo, Jianqing Zhang, Yuyan Zhou, Qiang Lin, Can Wang, Hande Dong, and Jiawei Chen. 2026. Recreate: Reasoning and creating domain agents driven by experience. arXiv preprint arXiv:2601.11100.

Yongquan He, Wenyuan Zhang, Xuancheng Huang, Peng Zhang, Lingxun Meng, Xiang Zhou, Ke Zeng, and Xunliang Cai. 2025. Don’t half-listen: Capturing key-part information in continual instruction tuning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23649–23668, Vienna, Austria. Association for Computational Linguistics.

Junan Hu, Shudan Guo, Wenqi Liu, Jianhua Yin, and Yinwei Wei. 2026a. Context-agent: Dynamic discourse trees for non-linear dialogue. arXiv preprint arXiv:2604.05552.

Yupeng Hu, Zixu Li, Zhiwei Chen, Qinlei Huang, Zhiheng Fu, Mingzhu Xu, and Liqiang Nie. 2026b. Refine: Composed video retrieval via shared and differential semantics enhancement. ACM Transactions on Multimedia Computing, Communications and Applications.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, and 28 others. 2025. Memory in the age of ai agents. Preprint, arXiv:2512.13564.

Shiting Huang, Zhen Fang, Zehui Chen, Siyu Yuan, Junjie Ye, Yu Zeng, Lin Chen, Qi Mao, and Feng Zhao. 2025. Critictool: Evaluating self-critique capabilities of large language models in tool-calling error scenarios. arXiv preprint arXiv:2506.13977.

Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, and Deyi Xiong. 2026. Revisiting entropy in reinforcement learning for large reasoning models. Preprint, arXiv:2511.05993.

Minsoo Kim, Victor Bursztyn, Eunyee Koh, Shunan Guo, and Seung-won Hwang. 2024. RaDA: Retrieval-augmented web agent planning with LLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13511–13525, Bangkok, Thailand. Association for Computational Linguistics.

Shashank Kirtania, Param Biyani, Priyanshu Gupta, Yasharth Bajpai, Roshni Iyer, Sumit Gulwani, and Gustavo Soares. 2025. Improving language agents through brew. Preprint, arXiv:2511.20297.

Dongjun Lee, Juyong Lee, Kyuyoung Kim, Jihoon Tack, Jinwoo Shin, Yee Whye Teh, and Kimin Lee. 2025. Learning to contextualize web pages for enhanced decision making by llm agents. In The Thirteenth International Conference on Learning Representations.

Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng, and Huan Liu. 2025a. From generation to judgment: Opportunities and challenges of LLM-as-a-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2757–2791, Suzhou, China. Association for Computational Linguistics.

Mengdi Li, Jiaye Lin, Xufeng Zhao, Wenhao Lu, Peilin Zhao, Stefan Wermter, and Di Wang. 2025b. Curriculum-rlaif: Curriculum alignment with reinforcement learning from ai feedback. arXiv preprint arXiv:2505.20075.

Ruoran Li, Xinghua Zhang, Haiyang Yu, Shitong Duan, Xiang Li, Wenxin Xiang, Chonghua Liao, Xudong Guo, Yongbin Li, and Jinli Suo. 2026a. Mempo: Self-memory policy optimization for long-horizon agents. Preprint, arXiv:2603.00680.

Songze Li, Xiaoke Guo, Tianqi Liu, Biao Yi, Zhaoyan Gong, Zhiqiang Liu, Huajun Chen, and Wen Zhang. 2026b. What’s missing in screen-to-action? towards a ui-in-the-loop paradigm for multimodal gui reasoning. Preprint, arXiv:2604.06995.

Zixu Li, Yupeng Hu, Zhiwei Chen, Qinlei Huang, Guozhi Qiu, Zhiheng Fu, and Meng Liu. 2026c. Retrack: Evidence-driven dual-stream directional anchor calibration network for composed video retrieval. Proceedings of the AAAI Conference on Artificial Intelligence, 40(28):23373–23381.

Zixu Li, Yupeng Hu, Zhiwei Chen, Shiqi Zhang, Qinlei Huang, Zhiheng Fu, and Yinwei Wei. 2026d. Habit: Chrono-synergia robust progressive learning framework for composed image retrieval. Proceedings of the AAAI Conference on Artificial Intelligence, 40(8):6762–6770.

Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Hongzhang Liu, Ronghao Chen, Yangfan He, and 1 others. 2025. Seagent: Self-evolution trajectory optimization in multistep reasoning with llm-based agents. arXiv preprint arXiv:2508.02085.

Peiyang Liu, Ziqiang Cui, Di Liang, and Wei Ye. 2025a. Who stole your data? a method for detecting unauthorized rag theft. arXiv preprint arXiv:2510.07728.

Peiyang Liu, Xi Wang, Ziqiang Cui, and Wei Ye. 2025b. Queries are not alone: Clustering text embeddings for video search. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 874– 883.

Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, and Hua Wei. 2025c. Uncertainty quantification and confidence calibration in large language models: A survey. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 6107–6117.

Yitao Liu, Chenglei Si, Karthik R Narasimhan, and Shunyu Yao. 2025d. Contextual experience replay for self-improvement of language agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14179–14198, Vienna, Austria. Association for Computational Linguistics.

Yunbo Long, Yuhan Liu, and Liming Xu. 2026. Emomas: Emotion-aware multi-agent system for highstakes edge-deployable negotiation with bayesian orchestration. Preprint, arXiv:2604.07003.

Xing Han Lu, Zdenˇek Kasner, and Siva Reddy. 2024. Weblinx: Real-world website navigation with multiturn dialogue. In International Conference on Machine Learning, pages 33007–33056. PMLR.

Haohao Luo, Jiayi Kuang, Wei Liu, Ying Shen, Jian Luan, and Yang Deng. 2025. Browsing like human: A multimodal web agent with experiential fastand-slow thinking. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14232– 14251, Vienna, Austria. Association for Computational Linguistics.

Kexin Ma, Bojun Li, Yuhua Tang, Liting Sun, and Ruochun Jin. 2026a. Cast: Character-andscene episodic memory for agents. Preprint, arXiv:2602.06051.

Shichao Ma, Zhiyuan Ma, Ming Yang, Xiaofan Li, Xing Wu, Jintao Du, Yu Cheng, Weiqiang Wang, Qiliang Liu, Zhengyang Zhou, and 1 others. 2026b. Tspo: Breaking the double homogenization dilemma in multi-turn search policy optimization. arXiv preprint arXiv:2601.22776.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations.

Shuaiyi Nie, Siyu Ding, Wenyuan Zhang, Linhao Yu, Tianmeng Yang, Yao Chen, Tingwen Liu, Weichong

Yin, Yu Sun, and Hua Wu. 2026. Attnpo: Attentionguided process supervision for efficient reasoning. Preprint, arXiv:2602.09953.

Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S. Yu, and Qing Li. 2025. A survey of webagents: Towards next-generation ai agents for web automation with large foundation models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 6140–6150, New York, NY, USA. Association for Computing Machinery.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2025. Reasoningbank: Scaling agent self-evolving with reasoning memory. Preprint, arXiv:2509.25140.

Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, and Tu Vu. 2025. Sealqa: Raising the bar for reasoning in search-augmented language models. Preprint, arXiv:2506.01062.

Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-rong Wen. 2025. Tool learning with large language models: a survey. Frontiers of Computer Science, 19(8).

Harsh Raj, Vipul Gupta, Domenic Rosati, and Subhabrata Majumdar. 2025. Improving consistency in large language models through chain of guidance. Transactions on Machine Learning Research.

Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, and Balaji Lakshminarayanan. 2023. Self-evaluation improves selective generation in large language models. In Proceedings on "I Can’t Believe It’s Not Better: Failure Modes in the Age of Foundation Models" at NeurIPS 2023 Workshops, volume 239 of Proceedings of Machine Learning Research, pages 49–64. PMLR.

David Silver and Richard S Sutton. 2025. Welcome to the era of experience. Google AI, 1.

Yueqi Song, Frank F. Xu, Shuyan Zhou, and Graham Neubig. 2025. Beyond browsing: API-based web agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11066–11085, Vienna, Austria. Association for Computational Linguistics.

Miao Su, Yucan Guo, Zhongni Hou, Long Bai, Zixuan Li, Yufei Zhang, Guojun Yin, Wei Lin, Xiaolong Jin, Jiafeng Guo, and Xueqi Cheng. 2026. Beyond dialogue time: Temporal semantic memory for personalized llm agents. Preprint, arXiv:2601.07468.

Rui Sun, Jie Ding, Chenghua Gong, Tianjun Gu, Yihang Jiang, Juyuan Zhang, Liming Pan, and Linyuan Lü. 2026. Topodim: One-shot topology generation of diverse interaction modes for multi-agent systems. Preprint, arXiv:2601.10120.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, and Wangchunshu Zhou. 2025. AGENT KB: A hierarchical memory framework for cross-domain agentic problem solving. In ICML 2025 Workshop on Collaborative and Federated Agentic Workflows.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, and Yi Wu. 2025a. Offline reinforcement learning for LLM multi-step reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8881–8893, Vienna, Austria. Association for Computational Linguistics.

Jiawei Wang, Jiacai Liu, Yuqian Fu, Yingru Li, Xintao Wang, Yuan Lin, Yu Yue, Lin Zhang, Yang Wang, and Ke Wang. 2025b. Harnessing uncertainty: Entropy-modulated policy gradients for long-horizon llm agents. Preprint, arXiv:2509.09265.

Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, and Wenji Mao. 2026. Adaptive social learning via mode policy optimization for language agents. In The Fourteenth International Conference on Learning Representations.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. 2025c. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2025d. Agent workflow memory. In Forty-second International Conference on Machine Learning.

Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, and Lihong Li. 2025. WebAgent-r1: Training web agents via end-to-end multi-turn reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7909–7928, Suzhou, China. Association for Computational Linguistics.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhenglin Wang, Zhengwei Tao, DingChu Zhang, Zekun Xi, Xiangru Tang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025a. Webdancer: Towards autonomous information seeking agency. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025b. WebWalker: Benchmarking LLMs in web traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10290–10305, Vienna, Austria. Association for Computational Linguistics.

Lvhua Wu, Xuefeng Jiang, Sheng Sun, Tian Wen, Yuwei Wang, and Min Liu. 2025c. Zofia: Zero-shot fake news detection with entity-guided retrieval and multillm interaction. Preprint, arXiv:2511.01188.

Zhaofen Wu, Hanrong Zhang, Fulin Lin, Wujiang Xu, Xinran Xu, Yankai Chen, Henry Peng Zou, Shaowen Chen, Weizhi Zhang, Xue Liu, Philip S. Yu, and Hongwei Wang. 2026. Gam: Hierarchical graphbased agentic memory for llm agents. Preprint, arXiv:2604.12285.

Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, Yu Qiao, and Haifeng Li. 2025. Learning on the job: An experience-driven self-evolving agent for long-horizon tasks. Preprint, arXiv:2510.08002.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Shenghao Ye, Yu Guo, Dong Jin, Yikai Shen, Yunpeng Hou, Shuangwu Chen, Jian Yang, and Xiaofeng Jiang. 2025. When tableqa meets noise: A dual denoising framework for complex questions and large-scale tables. arXiv preprint arXiv:2509.17680.

Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, and Yafeng Deng. 2026. Hypermem: Hypergraph memory for long-term conversations. Preprint, arXiv:2604.08256.

Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, and 1 others. 2026. Vision-deepresearch benchmark: Rethinking visual and textual search for multimodal large language models. arXiv preprint arXiv:2602.02185.

Ding-Chu Zhang, Yida Zhao, Jialong Wu, Liwen Zhang, Baixuan Li, Wenbiao Yin, Yong Jiang, Yu-Feng Li, Kewei Tu, Pengjun Xie, and Fei Huang. 2025a. EvolveSearch: An iterative self-evolving search agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13134–13147, Suzhou, China. Association for Computational Linguistics.

Kangning Zhang, Wenxiang Jiao, Kounianhua Du, Yuan Lu, Weiwen Liu, Weinan Zhang, and Yong Yu. 2025b. Looptool: Closing the data-training loop for robust llm tool calls. Preprint, arXiv:2511.09148.

Qianchi Zhang, Hainan Zhang, Liang Pang, Yongxin Tong, Hongwei Zheng, and Zhiming Zheng. 2026a. Less is more: Compact clue selection for efficient retrieval-augmented generation reasoning. In Proceedings of the ACM Web Conference 2026, pages 1971–1982.

Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng. 2024. Adacomp: Extractive context compression with adaptive predictor for retrieval-augmented large language models. arXiv preprint arXiv:2409.01579.

Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng. 2026b. Stable-rag: Mitigating retrieval-permutation-induced hallucinations in retrieval-augmented generation. arXiv preprint arXiv:2601.02993.

Wenyuan Zhang, Tianyun Liu, Mengxiao Song, Xiaodong Li, and Tingwen Liu. 2025c. SOTOPIA-Ω: Dynamic strategy injection learning and social instruction following evaluation for social agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24669–24697, Vienna, Austria. Association for Computational Linguistics.

Wenyuan Zhang, Shuaiyi Nie, Jiawei Sheng, Zefeng Zhang, Xinghua Zhang, Yongquan He, and Tingwen Liu. 2025d. Revealing and mitigating the challenge of detecting character knowledge errors in LLM roleplaying. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 33279–33302, Suzhou, China. Association for Computational Linguistics.

Wenyuan Zhang, Shuaiyi Nie, Xinghua Zhang, Zefeng Zhang, and Tingwen Liu. 2025e. S1-bench: A simple benchmark for evaluating system 1 thinking capability of large reasoning models. Preprint, arXiv:2504.10368.

Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. 2023. Wider and deeper llm networks are fairer llm evaluators. Preprint, arXiv:2308.01862.

Yuanjun Zhang, Fuzel Ahamed Shaik, Suvojit Acharjee, Fahad Khalid, and Mourad Oussalah. 2026c. Towards reliable multimodal disaster severity assessment through preference optimization and explainable vision-language reasoning. Reliability Engineering & System Safety, page 112674.

Yunyao Zhang, Yihao Ai, Zuocheng Ying, Qirui Mi, Junqing Yu, Wei Yang, and Zikai Song. 2026d. Coupling macro dynamics and micro states for long-horizon social simulation. Preprint, arXiv:2604.05516.

Yunyao Zhang, Zikai Song, Hang Zhou, Wenfeng Ren, Yi-Ping Phoebe Chen, Junqing Yu, and Wei Yang. 2025f. ga − s3: Comprehensive social network simulation with group agents. In Findings of the Association for Computational Linguistics: ACL 2025,

pages 8950–8970, Vienna, Austria. Association for Computational Linguistics.

Zefeng Zhang, Xiangzhao Hao, Hengzhu Tang, Zhenyu Zhang, Jiawei Sheng, Xiaodong Li, Zhenyang Li, Li Gao, Daiting Shi, Dawei Yin, and Tingwen Liu. 2025g. Cooper: A unified model for cooperative perception and reasoning in spatial intelligence. Preprint, arXiv:2512.04563.

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025h. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17):19632–19642.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. 2024. Synapse: Trajectory-as-exemplar prompting with memory for computer control. In The Twelfth International Conference on Learning Representations.

Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, Qian Liu, Ge Zhang, and Zejun Ma. 2025. First return, entropyeliciting explore. Preprint, arXiv:2507.07017.

Keyang Zhong, Junlin Xie, Hefeng Wu, Haofeng Li, and Guanbin Li. 2026. Collaborative multiagent scripts generation for enhancing imperfectinformation reasoning in murder mystery games. Preprint, arXiv:2604.11741.

### A Details of Method

#### A.1 Details of Threshold Estimation

We implement the threshold estimation algorithm using the LogisticRegression class from scikit-learn with default hyperparameters, which is well-suited for our single-feature binary classification task (correct vs. incorrect steps based on entropy H¯t).

Bootstrap Resampling. Sampling with replacement (Lines 4-5 of Algorithm 1) generates bootstrap datasets where some original samples appear multiple times while others are omitted. Across N = 1000 iterations, this produces a distribution

Algorithm 1 Threshold Estimation Input: Entropy sets H+ (correct steps), H− (in-

correct steps), Bootstrap iterations N

Output: Threshold interval [θlower,θupper]

- 1: Initialize Θ ← ∅
- 2: for i = 1 to N do
- 3: // Bootstrap resampling
- 4: Sample Hi+ by drawing |S+| elements from H+ with replacement
- 5: Sample Hi− by drawing |S−| elements from H− with replacement
- 6: // Construct training data
- 7: Di ← {(H,¯ 0) | H¯ ∈ Hi+} ∪ {(H,¯ 1) | H¯ ∈ Hi−}
- 8: // Train logistic regression
- 9: Train logistic regression: (wi,bi) ←

arg minw,b (H,y¯ )∈D

i

L(y,σ(wH¯ + b))

- 10: where σ(z) = 1/(1 + e−z)
- 11: // Extract threshold at decision boundary
- 12: θ(i) ← −bi/wi
- 13: Θ ← Θ ∪ {θ(i)}
- 14: end for
- 15: Sort Θ in ascending order
- 16: θlower ← Q0.025(Θ)
- 17: θupper ← Q0.975(Θ)
- 18: return [θlower,θupper]

of decision boundaries Θ = {θ(1),...,θ(N)}, enabling uncertainty quantification via confidence intervals.

Decision Boundary Interpretation. Each logistic regression learns parameters (w,b) such that the decision boundary θ = −b/w represents the entropy value where P(correct | H¯) = 0.5. Geometrically, this corresponds to the inflection point of the sigmoid function σ(wH¯ + b) projected onto the entropy axis—the point of maximum model uncertainty, making it a natural intervention threshold.

Confidence Interval Construction. The 95% confidence interval [θlower,θupper] is derived by extracting the 2.5th and 97.5th percentiles of the bootstrap distribution:

θlower = Q0.025(Θ), θupper = Q0.975(Θ) (7) where Qp(·) denotes the p-th quantile function.

Computational Efficiency. On an Intel Xeon CPU, estimating thresholds for one step type (process or answer) completes within seconds. Since

we estimate thresholds independently for both types, the total offline computation time is negligible compared to online inference costs.

#### A.2 Additional Advantages of the Algorithm

As illustrated in Figure 8, the threshold estimation algorithm adapts effectively to cases with reduced distributional differences. In the 32B process steps, the predominance of the yellow region indicates that under higher uncertainty, the algorithm adaptively randomizes trigger decisions to avoid excessive intervention.

### B Details of Experiment Results

#### B.1 Pass@k Results

As shown in Figure 5, ExpSeek demonstrates stronger performance in sampling diversity. Compared to vanilla, the absolute improvements in pass@3 are 12.9% and 8.8%, exceeding the improvement margins in average accuracy. In contrast, the two baselines do not exhibit better diversity, and even fall below vanilla on multiple datasets. This highlights the significant advantage of ExpSeek, with potential even as a rollout augmentation strategy in agentic RL training.

#### B.2 Why prior methods underperform?

Based on the experimental results, we reflect on why prior experience-based intervention methods are ineffective. We attribute the reasons to two aspects.

- (1) Experience acquisition is disconnected from actual reasoning. Most experience repository construction methods meticulously process training trajectories, perform multiple rounds of careful denoising, and cautiously distill core experience items with deep observation of successes and failures. However, the construction process itself is challenging, and when such heavily processed experiences are given to reasoning agents, the agents may not even understand some abstract expressions. This differs from how humans use experience—when humans recall experience, the amount of information retrieved instantly is enormous, including scenes, behavioral details, and even emotions, far exceeding simplified experience items.
- (2) Experience is difficult to utilize. During multi-turn agent interactions, even when accurate experiences are provided in system prompts, it is difficult to require models to precisely locate a few short effective experience items across ultra-long

(a) Process Steps

1.0

ProbabilityofIntervention

0.8

- 0

- 1

- 2

SampleDensity

Safe Zone

Buffer Zone

0.6

Intervention Zone

Median Probability Curve

0.4

True Dist.

False Dist.

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0 1.2

Simulated Thresholds

Frequency

Lower 2.5%

Median

200

Upper 97.5%

0

1.0 1.5 2.0 2.5 3.0

Entropy Value

(b) Final Steps

| | |
|---|---|
| | |
| |Safe Zone Buffer Zone Intervention Zone Median Probability Curve<br><br>| |
|---|
<br><br>|
| | |
|| |
|---|
<br><br>| |
| || |
|---|
<br><br>True Dist. False Dist.<br><br>| |
|---|
|
| | |
| | |
| | |

1.0

- 0
- 1
- 2

ProbabilityofIntervention

0.8

SampleDensity

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4

Simulated Thresholds

Frequency

Lower 2.5%

100

Median

Upper 97.5%

0

0.70 0.75 0.80 0.85 0.90

Entropy Value

Figure 8: Entropy distributions of process and answer steps on Dtrain for Qwen3-32B, with fitted logistic regression curves.

contexts. Moreover, models must also leverage experience to correct their original reasoning tendencies.

ExpSeek essentially simulates a user through an experience model, providing effective guidance that does not require deep model understanding to be leveraged. Since each guidance is generated in real-time based on context, the experience is not disconnected from actual reasoning.

#### B.3 Case Studies

As shown in Table 7, 8, 9and 10. The contrasting outcomes reveal a critical limitation in agent’s reasoning when relying on incomplete information, and demonstrate how strategic guidance can redirect agents toward reliable evidence.

Why the unguided trajectory failed: The agent’s error stemmed from reliance on search snippets that listed "Spider-Man: No Way Home" alongside "The Super Mario Bros. Movie" and "Jurassic World: Dominion" in billion-dollar film discussions, but omitted distributor information. Without verification mechanisms, the agent incorrectly inferred all three were Universal releases, when "Spider-Man: No Way Home" was actually

WebWalkerQA

GAIA SEAL xbench Avg. Easy Medium Hard Avg.

Method

Qwen3-8B No Experience 57.50-0.0 60.00-0.0 46.11-0.0 54.54-0.0 47.57-0.0 39.76-0.0 43.00-0.0 48.99-0.0 Training-Free GRPO 56.67↓0.8 62.54↑2.5 52.67↑6.6 57.29↑2.8 48.54↑1.0 42.13↑2.4 45.00↑2.0 51.26↑2.3 REASONINGBANK+ 61.17↑3.7 58.19↓1.8 50.00↑3.9 56.45↑1.9 48.70↑1.1 44.09↑4.3 46.80↑3.8 51.49↑2.5 ExpSeek (Ours) 69.17↑11.7 72.45↑12.5 63.33↑17.2 68.32↑13.8 54.37↑6.8 50.00↑10.2 62.00↑19.0 61.89↑12.9

Qwen3-32B No Experience 62.50-0.0 61.43-0.0 58.33-0.0 60.75-0.0 54.37-0.0 47.64-0.0 53.00-0.0 56.21-0.0 Training-Free GRPO 63.95↑1.5 59.57↓1.9 57.45↓0.9 60.32↓0.4 56.62↑2.3 48.35↑0.7 53.60↑0.6 56.59↑0.4 REASONINGBANK+ 68.33↑5.8 64.33↑2.9 56.44↓1.9 63.03↑2.3 46.60↓7.8 48.82↑1.2 59.00↑6.0 57.25↑1.0 ExpSeek (Ours) 72.50↑10.0 72.86↑11.4 63.33↑5.0 69.56↑8.8 63.11↑8.7 51.97↑4.3 66.00↑13.0 64.96↑8.8

Table 5: Extended main results reporting Pass@3 accuracy for baselines and ExpSeek. We also provide absolute performance differences from experience-free ReAct.

WebWalker-Easy

WebWalker-Medium

WebWalker-Hard

###### GAIA

###### SEAL

xbench

3.0

3.5

3.0

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla<br><br>ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |

2.5

2.5

2.5

(a)ProcessStep(b)FinalStep

3.0

2.5

2.5

2.0

2.0

2.0

2.5

2.0

2.0

Density

1.5

2.0

1.5

1.5

1.5

1.5

1.5

1.0

1.0

1.0

1.0

1.0

1.0

0.5

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0

0.4 0.6 0.8 1.0 1.2

0.2 0.4 0.6 0.8 1.0 1.2

0.4 0.6 0.8 1.0

0.25 0.50 0.75 1.00 1.25

0.2 0.4 0.6 0.8 1.0

0.4 0.6 0.8 1.0 1.2

2.0

2.0

| | | |
|---|---|---|
| | |Vanilla|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | |Vanilla|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |Vanilla|
|---|---|---|---|
| | | |ExpSeek|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | |Vanilla|
|---|---|---|
| | |ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | |Vanilla|
| | | |ExpSeek|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | |Vanilla ExpSeek|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

2.0

1.8

1.8

1.8

1.8

1.8

1.8

1.5

1.5

1.5

1.5

1.5

1.5

1.2

1.2

1.2

###### Density

1.2

1.2

1.2

1.0

1.0

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

0.8

0.8

0.5

0.5

0.5

0.5

0.5

0.5

0.2

0.2

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0 1.5

0.0 0.5 1.0 1.5

0.0 0.5 1.0 1.5

0.0 0.5 1.0 1.5

0.00 0.25 0.50 0.75 1.00 1.25

0.0 0.5 1.0 1.5

Entropy

Entropy

Entropy

Entropy

Entropy

Entropy

Figure 9: Entropy distributions of process and answer steps for Qwen3-32B before and after applying ExpSeek across all benchmarks.

distributed by Sony Pictures. This represents a classic entity attribute confusion—the agent failed to verify the critical "distributor" attribute before counting.

Why guidance succeeded: The guidance implemented a two-pronged strategy. First, it redirected the agent toward authoritative box office sites (Screen Rant, Rotten Tomatoes, Box Office Mojo) that systematically label films with distributors. Second, it reinforced the task’s dual requirement—counting films satisfying both revenue threshold and distributor constraint—prompting a verification-based approach. The Screen Rant article provided explicit evidence that only two Universal-distributed films (via Illumination and Amblin partnerships) appeared on the billion-dollar list, enabling correct filtering and counting.

The guidance’s success lies in methodological redirection rather than direct answer provision. By steering the agent toward sources with richer structural metadata and emphasizing attribute verification, it enabled the agent to overcome reasoning

failure through improved evidence quality rather than external correction.

#### B.4 Guiding Experience

In this section, we demonstrate the details of the guiding experience. Table 6 presents experience demos from both stages. The process step experience focuses on strategic decisions during reasoning, such as how the student selects information sources for verification after searching. Its mistake identifies methodological issues affecting subsequent reasoning paths (e.g., relying on snippets rather than accessing authoritative sources), while guidance provides directional suggestions for process optimization (e.g., prioritizing official sources). In contrast, the final step experience concentrates on answer formulation, such as extracting the final answer from search results. Its mistake targets detail-oriented errors directly impacting output quality (e.g., failing to faithfully reproduce complete proper nouns), while guidance emphasizes result accuracy, prompting the model

Experience base demos. Process Step demo {

"Behavior": "The student recognized the need to confirm which African countries participated in the 7th China International Import Expo (CIIE) under the theme of ’60th Anniversary of Diplomatic Relations’ through external information sources, and therefore initiated multi-angle keyword searches.",

"Mistake": "Although the search keywords were reasonably designed, the student did not prioritize visiting sources most likely to contain authoritative information (such as the official CIIE website) for in-depth access. Instead, they relied only on search result snippets for preliminary judgment, laying hidden risks for subsequent misjudgment.",

"Guidance": "When reviewing search results, you should prioritize identifying and planning to visit authoritative or official sources (such as ciie.org). Especially when multiple results cite the same webpage, it should be set as the primary verification target to avoid prematurely synthesizing incomplete information to draw conclusions." }

Final Step demo {

"Behavior": "The student reviewed the search results, found that multiple results pointed to a website design company called ’Niegao’ (niegoweb.com), and inferred from this that the website was designed by this company.",

"Mistake": "The student overlooked the complete official company name that clearly appeared in the search results—’Niegao Brand Network’ or ’Niegao Brand Design’—and instead simplified it on their own to ’Niegao Website Design Company,’ resulting in an answer inconsistent with the standard answer. Although semantically similar, the student failed to faithfully reproduce the proper noun from the original text.",

"Guidance": "Before drawing conclusions, please carefully check the complete institutional names that appear in the search results, especially the official names of collaborating parties in case study pages. Prioritize using the full brand name directly mentioned in the original text rather than summarizing it based on your own understanding." }

Table 6: Experience base demos for Qwen3-8B.

to focus on answer details and encouraging trajectory extension for thorough verification when necessary. This triplet design reproduces step-level error patterns from training trajectories and provides targeted guidance, enabling the experience model to more accurately guide both process and final steps during test inference.

### C Details of Setting

Following the LLM-as-a-judge approach (Zheng et al., 2023; Zhang et al., 2023; Li et al., 2025a), we use Qwen3-235B-A22B-Instruct-2507 as the evaluator with temperature set to 0 and top-p set to 1 (see Table 19 for the evaluation prompt). As Qwen3-8B and Qwen3-32B are hybrid reasoning models, we deploy them in non-thinking mode and adopt the <thought></thought> tag rather than <think></think> to prevent conflicts with their original output structure. For tool implementation, we use Bright Data5 to provide stable web API services for the search tool, Jina6 as the web access service for the visit tool, and Qwen3-235B-A22B-Instruct-2507 as the sum-

- 5https://www.bright.cn
- 6https://jina.ai

marization model within the visit tool. All other experimental settings and prompts remain consistent with prior work (Wu et al., 2025a). The four benchmarks employed in our evaluation are widely recognized in web agent research and are permitted for use in academic studies under their distribution terms.

### D Extended Related Work

The related work in the main text is organized along two technical dimensions: experience intervention and entropy. This section introduces the broader background and recent progress in agent research.

Agents in long-horizon interactions rely on effective memory mechanisms to maintain coherence. Recent studies have advanced long-term memory through hierarchical and graph-based structuring (Yue et al., 2026; Wu et al., 2026), temporalsemantic modeling (Su et al., 2026), episodic representations (Ma et al., 2026a), dynamic context management in non-linear dialogues (Hu et al., 2026a), and autonomous memory compression (Li et al., 2026a). Complementarily, challenges of redundancy, noise, and ordering sensitivity in retrieved information have been addressed via adaptive com-

pression, minimal sufficient clue selection, and permutation robustness (Zhang et al., 2024, 2026a,b), while IP protection for retrieval-augmented systems (Liu et al., 2025a) and multi-agent collaborative fact verification (Wu et al., 2025c) have also attracted attention.

The ability of agents to continuously selfimprove is another active direction. A line of work advances agent self-evolution through trajectorylevel optimization (Lin et al., 2025), experiencedriven agent creation (Hao et al., 2026), coevolutionary capability expansion and experience distillation (Cheng et al., 2026), curriculum-based feedback alignment (Li et al., 2025b), and test-time variational synthesis over unlabeled queries (Bai et al., 2026). For tool use, closed-loop data evolution that couples capability diagnosis with targeted data refinement has proven highly effective (Zhang

- et al., 2025b). At the policy level, adaptive reasoning depth switching (Wang et al., 2026), turnlevel reward assignment in multi-turn search (Ma
- et al., 2026b), and efficient multi-agent communication topology generation (Sun et al., 2026) have improved decision-making efficiency, and a theoretical analysis of token-level entropy dynamics during reinforcement learning (Hao et al., 2025b) provides an important reference for understanding entropy’s role in agent systems.

Agent research also continues to expand toward broader modalities and scenarios. In multimodal and embodied domains, visual-textual retrieval evaluation (Zeng et al., 2026), self-supervised multimodal model refinement (Han et al., 2026), selfcritique under tool-calling errors (Huang et al., 2025), partial decoupling of reasoning and action (Fang et al., 2025b), cross-embodied foundation models (Hao et al., 2025a), UI-in-the-Loop GUI reasoning (Li et al., 2026b), unified spatial intelligence modeling with auxiliary modality generation (Zhang et al., 2025g), and explainable multimodal disaster assessment (Zhang et al., 2026c) collectively advance perception and execution capabilities. Research on composed video and image retrieval (Li et al., 2026c; Hu et al., 2026b; Li et al., 2026d; Liu et al., 2025b) further improves the precision of multimodal information acquisition. In social intelligence, strategy-injected agent training (Zhang et al., 2025c), role-playing agents (Zhang et al., 2025d; Chen et al., 2026), collaborative reasoning under imperfect-information games (Zhong et al., 2026), large-scale social simulation integrating macro-micro modeling (Zhang

et al., 2026d, 2025f), and multi-agent negotiation (Long et al., 2026) provide new paradigms for studying agent behavior in open environments. Reasoning over noisy structured data (Ye et al., 2025; Guo et al., 2026) and knowledge retention in continual learning (Chen and Zeng, 2025; He et al., 2025) also remain active directions. Collectively, these diverse lines reflect a broader trend of agents shifting from passive execution toward proactive decision-making in complex environments, and the intrinsic-signal-driven experience seeking explored by ExpSeek represents a concrete manifestation of this trend in the web agent setting.

### E Details of Prompts

In this section, we present all key prompts. Table 11 shows the system prompt for ReAct, which we have slightly adjusted based on guidance experience. Tables 12 and 13 are prompts for generating the experience base, where Table 12 implements identifying step mistakes and generating triplets, and Table 13 iteratively generates topics based on all triplets. Tables 14 and 15 are prompts for utilizing the experience model during inference. Table 14 is for the experience model to determine which topics to utilize, and Table 15 generates appropriate proactive guidance based on all experiences under the topics and the current history context. Table 17 is the prompt for using a LLM as an intervention trigger, where outputting “yes” triggers guidance for that step. Table 18 is for utilizing retrieved experience base experiences during inference. Table 16 is for generating guidance directly using the world experience of the experience model without relying on the experience base.

Case study (Incorrect): Qwen3-8B interaction without guiding experience Question: How many Universal Studios films have grossed over $1 billion worldwide since 2020? Agent Action: <think> To determine how many Universal Studios films have grossed over $1 billion worldwide since 2020, I need to search for a list of Universal Studios films and their box office earnings. This will involve first identifying which films are associated with Universal Studios and then checking their gross earnings. I’ll begin by searching for a list of Universal Studios films and their box office performance since 2020. </think> <tool_call> Search: Universal Studios films gross over $1 billion since 2020 Search: List of Universal Studios movies with box office earnings over $1 billion since 2020 </tool_call> Env Tool Response: A Google search for ’Universal Studios films gross over $1 billion since 2020’ found 5 results:

......

- 2. [Universal - The First Half of the 2020s In Review : r/boxoffice] Universal - The First Half of the 2020s In Review · The Super Mario Bros. Movie - $1.36B · Jurassic World Dominion -

- $1.001B · Oppenheimer- $975.6M.

......

- 5. [6 Movies That Have Made $1 Billion At The Box Office Since Covid]
- 6 Jurassic World: Dominion (2022) · 5 The Super Mario Bros. Movie (2023) · 4 Spider-Man: No Way Home (2021) · 3 Avatar: The Way of Water (2022) · 2 ...

...... Agent Action: <think> Based on the search results, I need to determine which Universal Studios films have grossed over $1 billion worldwide since 2020. The key information comes from the Reddit post and the The Numbers website. According to the Reddit post, "The Super Mario Bros. Movie" has grossed $1.36B, and "Jurassic World Dominion" has grossed $1.001B. From the The Numbers website, there is a list of the top Universal movies by worldwide box office, and the most recent ones include "The Super Mario Bros. Movie" and "Jurassic World Dominion." Additionally, it appears that other films such as "Avatar: The Way of Water" might also qualify based on the context provided in the search results. To confirm this, I will visit the The Numbers website directly to get the exact data for Universal Studios films and their box office earnings since 2020. </think> <tool_call> Visit: https://www.the-numbers.com/box-office-records/worldwide/all-movies/theatrical-distributors/universal Visit Goal: Find the list of Universal Studios films with box office earnings over $1 billion since 2020. </tool_call> Env Tool Response: The requested webpage did not load fully and contains no usable data. As a result, there is insufficient information to identify Universal Studios films with box office earnings exceeding $1 billion since 2020.

...... Agent Action: <think> The attempt to access the The Numbers website directly did not yield usable data. This may be due to technical issues or incomplete page loading. I will need to find an alternative source of information. Based on the previous search results, the Reddit post mentioned "The Super Mario Bros. Movie" and "Jurassic World Dominion" as having grossed over $1 billion. However, I should verify this with another reliable source to confirm the accuracy of these figures. </think> <tool_call> Search: Universal Studios films over $1 billion since 2020 Search: List of Universal Studios movies with box office earnings over $1 billion since 2020 </tool_call>

Table 7: Case study: Qwen3-8B interaction without guiding experience

Case study (Incorrect): Qwen3-8B interaction without guiding experience Env Tool Response: A Google search for ’Universal Studios films over $1 billion since 2020’ found 5 results:

......

- 3. [8 Movies That Broke $1 Billion at the Box Office Since Covid]Avatar: The Way of Water’s box office comes in at over

- $2.3 billion...

......

5. [6 Movies That Have Made $1 Billion At The Box Office Since Covid]6 Jurassic World: Dominion (2022) · 5 The Super Mario Bros. Movie (2023) · 4 Spider-Man: No Way Home (2021) · 3 Avatar: The Way of Water (2022) · 2 ...

...... Agent Action: <think> From the search results, I can see that "The Super Mario Bros. Movie" (2023) and "Jurassic World: Dominion" (2022) are mentioned as having grossed over $1 billion worldwide. Additionally, "Spider-Man: No Way Home" (2021) is noted as one of the first films to reach the $1 billion mark since the start of the pandemic. These three films are distributed by Universal Studios, and they all have box office earnings exceeding $1 billion globally since 2020. Therefore, the answer to the question is that three Universal Studios films have grossed over $1 billion worldwide since 2020. </think> <answer> Three Universal Studios films have grossed over $1 billion worldwide since 2020: "The Super Mario Bros. Movie," "Jurassic World: Dominion," and "Spider-Man: No Way Home." </answer>

Table 8: Case study: Qwen3-8B interaction without guiding experience (Table continued)

Question: How many Universal Studios films have grossed over $1 billion worldwide since 2020?

Agent Action: <think> To determine how many Universal Studios films have grossed over $1 billion worldwide since 2020, I need to search for a list of Universal Studios films and their box office earnings. This will involve first identifying which films are associated with Universal Studios and then checking their gross earnings. I’ll begin by searching for a list of Universal Studios films and their box office performance since 2020. </think> <tool_call> Search: Universal Studios films gross over $1 billion since 2020 Search: List of Universal Studios movies with box office earnings over $1 billion since 2020 </tool_call>

Env Tool Response: A Google search for ’Universal Studios films gross over $1 billion since 2020’ found 5 results:

......

2. [Universal - The First Half of the 2020s In Review : r/boxoffice] Universal - The First Half of the 2020s In Review · The Super Mario Bros. Movie - $1.36B · Jurassic World Dominion $1.001B · Oppenheimer- $975.6M.

......

5. [6 Movies That Have Made $1 Billion At The Box Office Since Covid] 6 Jurassic World: Dominion (2022) · 5 The Super Mario Bros. Movie (2023) · 4 Spider-Man: No Way Home (2021) · 3 Avatar: The Way of Water (2022) · 2 ...

...... Agent Action: <think> Based on the search results, I need to determine which Universal Studios films have grossed over $1 billion worldwide since 2020. The key information comes from the Reddit post and the The Numbers website. According to the Reddit post, "The Super Mario Bros. Movie" has grossed $1.36B, and "Jurassic World Dominion" has grossed $1.001B. From the The Numbers website, there is a list of the top Universal movies by worldwide box office, and the most recent ones include "The Super Mario Bros. Movie" and "Jurassic World Dominion." Additionally, it appears that other films such as "Avatar: The Way of Water" might also qualify based on the context provided in the search results. To confirm this, I will visit the The Numbers website directly to get the exact data for Universal Studios films and their box office earnings since 2020. </think> <tool_call> Visit: https://www.the-numbers.com/box-office-records/worldwide/all-movies/theatrical-distributors/universal Visit goal: Find the list of Universal Studios films with box office earnings over $1 billion since 2020 </tool_call> Env Tool Response: The requested webpage did not load fully and contains no usable data. As a result, there is insufficient information to identify Universal Studios films with box office earnings exceeding $1 billion since 2020.

...... Agent Action: <think> The attempt to access the The Numbers website directly did not yield usable data. This may be due to technical issues or incomplete page loading. I will need to find an alternative source of information. Based on the previous search results, the Reddit post mentioned "The Super Mario Bros. Movie" and "Jurassic World Dominion" as having grossed over $1 billion. However, I should verify this with another reliable source to confirm the accuracy of these figures. </think> <tool_call> Search: Universal Studios films over $1 billion since 2020 Search: List of Universal Studios movies with box office earnings over $1 billion since 2020 </tool_call>

Table 9: Case study: Qwen3-8B interaction with guiding experience

Env Tool Response (with Guidance): A Google search for ’Universal Studios films over $1 billion since 2020’ found 5 results:

......

3. [8 Movies That Broke $1 Billion at the Box Office Since Covid] Avatar: The Way of Water’s box office comes in at over $2.3 billion...

......

- 5. [6 Movies That Have Made $1 Billion At The Box Office Since Covid]
- 6 Jurassic World: Dominion (2022) · 5 The Super Mario Bros. Movie (2023) · 4 Spider-Man: No Way Home (2021) · 3 Avatar: The Way of Water (2022) · 2 ...

...... <guidance> Review the core of the question: you need to count films released by Universal Studios with worldwide box office over $1 billion since 2020. This means the answer is not listing a specific film, but a complete count based on reliable sources. You have found some films that may meet the criteria, but different sources show inconsistencies. Instead of repeating previous searches, carefully examine those webpages in the existing search results that specifically compile "billion-dollar movie" lists, such as links from Screen Rant, Rotten Tomatoes, or Box Office Mojo. These pages often label each film with its distributor and specific box office figures, helping you verify one by one whether it belongs to Universal Studios and meets the time and amount conditions. Special reminder: Even if a page doesn’t directly state "there are X films in total," you can still obtain the answer through manual counting—the key is finding a reliable list that completely catalogs such films and marks their distributors. Prioritize visiting such summary articles or professional box office ranking pages. </guidance>

Agent Action: <think> Based on the guidance, I should focus on reliable sources that list billion-dollar movies and identify which ones are distributed by Universal Studios. The Screen Rant article mentioned "Jurassic World: Dominion" and "The Super Mario Bros. Movie"

- as having grossed over $1 billion, but I need to confirm this with direct access to the full article or a more comprehensive list. To do this, I will visit the Screen Rant page to verify the information. </think> <tool_call> Visit: https://screenrant.com/billion-dollar-movies-box-office-2020 Visit goal: Identify the number of Universal Studios films with box office earnings over $1 billion since 2020 </tool_call>

Env Tool Response: Evidence in page: ["5 The Super Mario Bros. Movie (2023) - The movie’s success has led to Nintendo teasing a range of films based on more of its iconic video game characters.", "Illumination earned a much higher net profit than other billion-dollar movies", "6 Jurassic World: Dominion (2022) - However, after scathing reviews... the movie was always going to flounder. The movie did face unlikely box office competition, too. Top Gun: Maverick was still playing when Jurassic World: Dominion was released...", "Jurassic World: Dominion made almost exactly $1 billion worldwide."]

Summary: Since 2020, two Universal Studios films have earned over $1 billion at the global box office: *Jurassic World: Dominion* (2022) and *The Super Mario Bros. Movie* (2023). Both were produced by Universal-owned or partnered studios—Illumination for *The Super Mario Bros. Movie* and Amblin Entertainment in partnership with Universal for *Jurassic World: Dominion*. These are the only two Universal-released films on the list of six post-2020 billion-dollar earners.

Agent Action: <think> Based on the information from the Screen Rant article, there are exactly two Universal Studios films that have grossed over $1 billion worldwide since 2020: "Jurassic World: Dominion" (2022) and "The Super Mario Bros. Movie" (2023). These are the only two films from Universal Studios listed among the billion-dollar earners since 2020. </think> <answer> Since 2020, two Universal Studios films have grossed over $1 billion worldwide: "Jurassic World: Dominion" (2022) and "The Super Mario Bros. Movie" (2023). Therefore, the answer is 2. </answer>

Table 10: Case study: Qwen3-8B interaction with guiding experience (Table continued)

###### ReAct system prompt.

You are a problem-solving expert, skilled at analyzing complex problems, forming solutions, and gradually resolving them step-by-step. Your approach is flexible and can be adjusted based on the progress of problem-solving. Any problem you encounter will certainly be solvable; therefore, no matter how complex, you must not give up until you find the answer you consider correct.

You must output two parts based on the current input, both enclosed by specific angle bracket position markers:

**The first part** is enclosed by <thought></thought> and contains your self-thinking process, including analysis of the problem, determining whether tool usage is required, which tool to call, and what values to pass to the tool.

**The second part** is divided into two situations:

- (1) If you wish to call a tool, then after </thought> output <tool_call></tool_call> containing standard tool call content,

with the format introduced below.

- (2) If you believe the current problem can already be solved, then after </thought> output <answer></answer> containing

the final answer.

**Available tools and their function overview**

- 1. search: Input a keyword you think needs to be searched, returns multiple website links and corresponding summaries.
- 2. visit: Input a website link and the problem you hope to solve by visiting that site, returns the problem’s answer and a snippet of the original content from the site corresponding to the answer.

**Detailed tool call format** {search_format} {visit_format}

**Model workflow description**

- 1. Model input consists of the historical loop of interactions, including the model’s generated thoughts and tool calls, and the user’s returned tool results. The history may have multiple rounds, all designed to solve a given problem.
- 2. The model must output two parts: thinking process, and tool call or answer, each enclosed by specific angle bracket position markers:

- - If thinking, follow the format: <thought> here is the thinking process </thought>
- - If calling a tool, follow the format: <tool_call> tool call here </tool_call>
- - If determining the final answer, follow the format: <answer> final answer </answer>

- 3. The user will respond with tool call results or occasionally provide guidance. If there is guidance, you should carefully consider whether the user’s ideas are reasonable and try to follow them:

- - If responding to a tool call, you will see the format: <tool_response> here is the tool’s return value </tool_response>
- - If the user provides careful guidance, you will see the format: <user_guidance> here is the user’s guidance content

</user_guidance>

- 4. Every problem must have an answer; during multi-round resolution, do not forget your past planning and process results, and do not forget the details embedded in the problem.
- 5. The search tool’s return value is only website links and snippet summaries; they are hardly reliable references and can only serve as search direction. To obtain accurate information, you must call the visit tool to visit a site.
- 6. If the interaction includes a resolution plan, follow the plan. Do not blindly ignore key constraints in the plan to avoid potential cascading errors.
- 7. In tool_call generation, the format must follow the above definitions and be valid JSON. An incorrect format will cause tool calls to fail.

- *8*. **IMPORTANT**: If the user provides guidance after the answer, prioritize regenerating <tool_call></tool_call> to continue searching for missing clues, or provide only when you are absolutely certain of the answer.
- **Multi-round interaction full example** ˋˋˋ {example} ˋˋˋ

**Notes**

1. **You must generate the position markers** in accordance with the requirements stated above (<thought></thought>; <tool_call></tool_call> or <answer></answer>). 2. In particular, do not forget to generate the closing tags: </thought>, </tool_call>, OR </answer>

- 3. You must not generate extra angle bracket position markers. Table 11: ReAct system prompt.

###### Prompt for generating experience triplets.

# Questions for Students to Solve {question}

# Standard Answer for the Question {answer}

# This is a complete trajectory that ultimately got the correct answer as your reference: ˋˋˋ{true_traj}ˋˋˋ

# This is a complete trajectory that ultimately got the wrong answer: ˋˋˋ{false_traj}ˋˋˋ

# Pattern of All Trajectories: Question, R1, O1, R2, O2, R3, O3, R4...

# Instructions and Your Task

- 1. Define a STEP as R_i+O_i, but the last STEP only has R_N
- 2. Each R is a student’s response, attempting to call tools to further solve the problem, but the second trajectory with wrong answer always has some issues
- 3. Your core task is to answer this question for each STEP: ˋˋˋIn order to avoid the final error, if guidance is provided after this STEP ends, what should be done to make the agent perform better?ˋˋˋ
- 4. Of course, a complete guidance is a triplet <student’s current state, reason why this STEP leads to the final error, what to say before the next STEP to improve the current state> ˋˋˋExplanation of the triplet:

- - Student’s current state: A relatively general description, introducing what the student saw and what they did. The description does not involve error attribution, focuses on objectively stating the student’s behavior, and does not evaluate whether the behavior is good/bad or right/wrong
- - Reason why this STEP leads to the final error: Unlike the current state, this part explicitly points out what mistake the student made in this STEP
- - What to say before the next STEP to improve the current state: Based on the errors mentioned above, provide specific guidance that will help the student perform better in the next STEP if they follow it. Of course, do not directly tell the student the answer! ˋˋˋ

- 5. The guidance opinion in the triplet generated for STEP_i will be concatenated after O_i, which means the student can see it before generating R_i+1
- 6. Not every STEP necessarily needs guidance, you can skip after analysis, but since the trajectory is wrong, **there must be

at least one STEP that has issues and can be summarized into a triplet**

- 7. Finally, briefly summarize what three good pieces of advice could be given before working on this problem
- 8. **!!Must Note!!** The total number of rounds you analyze in the trajectory is **{step_num}**, you must generate the corresponding number of STEPs before you can continue to generate TOTAL!

# Output Format (strictly follow the markdown format I give you) ˋˋˋ

- # STEP 1: ## Analysis

- - Write analysis content here ## Triplet (If there is no error, directly write "- None", do not generate a triplet when there is no error)
- - Student’s current state: Write current state here
- - Reason why this STEP leads to the final error: Write reason here
- - What to say before the next STEP to improve the current state: Write guidance here

- # STEP 2:

... ˋˋˋ

- Table 12: Prompt for generating experience triplets.

###### Prompt for iteratively generating topics.

A teacher is analyzing each step taken by students when solving complex problems. I will give you several “behavior + mistake” items that the teacher has summarized for students at certain steps, defined as: ˋˋˋ behavior: A relatively general description introducing what the student saw and then did. The description does not involve error attribution, focuses on objectively stating the student’s behavior, and does not evaluate whether the behavior is good/bad or right/wrong mistake: The student is satisfied with preliminary information (such as “Deputy Party Secretary”) and fails to realize the need to cross-reference multiple search results to extract a complete and accurate answer, particularly overlooking that the specific name “xx xx xx” has already appeared in the first entry of the third set of search results. ˋˋˋ

# Overall Overview Your goal is to give **each behavior + mistake** a **scenario-narrative label**. A label’s description should be concise enough to clearly express the characteristics of the behavior and be reusable. For each given new behavior + mistake, you can choose one of the following three actions:

- 1. Reuse: Do not change any current labels, and select an existing label for the new behavior (recognizing the existing classification)
- 2. Create: Do not change existing labels, create a new label for the new behavior (existing classification is incomplete)
- 3. Modify: Modify certain current labels, and assign that label to the new behavior (existing classification is inaccurate) # Detailed Requirements

- 1. Each label must be concise and clear, but needs to have certain semantic information that allows people to understand the characteristics of the current behavior + mistake without explanation. It should be at least a dozen or dozens of words (e.g., in the pattern of xxx: xxx xxx xx).
- 2. There cannot be too many labels; each label should have distinguishability in scenario content.
- 3. One label can correspond to multiple behaviors, so you must ensure their textual content is consistent.
- 4. Use the given id as the unique identifier for behaviors. When outputting, you need to output the ids and labels of all existing behaviors and new behaviors.
- 5. Try to keep the number of different labels balanced.

# List of Behaviors Already Given Labels {exp_list}

# List of New Behaviors {new_exp_list}

# Output Format: ˋˋˋ {output_format} ˋˋˋ

- Table 13: Prompt for iteratively generating topics.

###### Prompt for experience model topic selection stage.

# Overall Instructions You are a teacher who is very good at guiding students to answer difficult questions Now, the student has made some attempts on the given problem, but there may be some non-absolute but potential issues, or even errors, in the **current step**. I will give you the student’s multiple rounds of attempts from the beginning to the current state, where assistant represents the student’s response, user represents the environment’s return value, and may also include your previous guidance. In addition, a topic list full of your guidance experience is provided for you to choose from.

# Student’s Multiple Rounds of Interaction ˋˋˋ{history}ˋˋˋ

# Candidate Topic List ˋˋˋ{topic_list}ˋˋˋ

# Your Task

- 1. You need to combine the student’s current state and select **3** topics from the several potential error topics I give you. After you carefully state the reasons for selection, just output the idx of the selected topics.
- 2. The student may not have actually made a mistake, but your subsequent guidance can prevent problems before they occur.

# Output Format Strictly follow the markdown format below for output ˋˋˋ # Analysis of the Current Step Write your analysis here # Selected Topic idx (separated by spaces) idx1 idx2 idx3 ˋˋˋ

Output:

Table 14: Prompt for experience model topic selection stage.

###### Prompt for experience model guidance generation stage.

# Overall Instructions You are a teacher who is very good at guiding students to answer difficult questions. Now, the student has made some attempts on the given problem, but there may be some non-absolute but potential issues, or even errors, in the **current step**. I will give you the student’s multiple rounds of attempts from the beginning to the current state, where assistant represents the student’s response, user represents the environment’s return value, and may also include your previous guidance. In addition, you are given your previous rich teaching experience, which consists of guidance you provided for different problem steps. Your goal is to provide the most appropriate guidance for the current step.

# Student’s Multiple Rounds of Interaction ˋˋˋ {history} ˋˋˋ

# Previous Rich Guidance Experience These guidances are based on the step problems you discovered in students, so each content is a triplet, including the student’s behavior, the error, and your previous guidance ˋˋˋ {topic_kb} ˋˋˋ

# Your Task and Requirements

- 1. The question must have an answer. If the student thinks there is insufficient evidence, it must be because they haven’t found the evidence. After careful analysis, provide your guidance for the student’s current step, with the goal of helping the student actually answer the question correctly.
- 2. Since this step may not necessarily be wrong, please carefully choose your wording to prevent your guidance from introducing bias.
- 3. Your analysis must include a brief review of the **problem** that the student needs to solve, emphasizing the content of the problem to the student to prevent answering off-topic.
- 4. The guidance you provide will be given to the student together with the tool call results after this step ends.

- - If the student generates a tool call in this step, the guidance will be given to the student together with the tool return value
- - If the student generates an answer in this step, the guidance will be given directly to the student, and the student will

choose whether to answer again or continue to generate tool calls based on your guidance

- 5. It is forbidden to find answers on behalf of the student, and it is forbidden to hint at what the answer is under any circumstances. **You are a teacher, not someone helping students cheat**.
- 6. It is forbidden to provide **direct clues** to students. Your purpose is only to **guide**.
- 7. Guidance should be clear and easy to understand. If necessary, you can encourage students to continue calling tools or switch tools.
- 8. The guidance you provide should try to imitate the previous guidance patterns, don’t improvise freely.

# Output Format Strictly follow the markdown format below for output ˋˋˋ # Analysis combining student’s current behavior and previous experience to provide appropriate guidance for the present moment Write your detailed analysis here # Guidance Content Write your guidance content here ˋˋˋ

Output:

Table 15: Prompt for experience model guidance generation stage.

###### Prompt for generating guidance experience directly without referencing the experience base.

# Overall Instructions You are a teacher who is very good at guiding students to answer difficult questions. Now, the student has made some attempts on the given problem, but there may be some non-absolute but potential issues, or even errors, in the **current step**. I will give you the student’s multiple rounds of attempts from the beginning to the current state, where assistant represents the student’s response, user represents the environment’s return value, and may also include your previous guidance. Your goal is to provide the most appropriate guidance for the current step.

# Student’s Multiple Rounds of Interaction ˋˋˋ {history} ˋˋˋ

# Your Task and Requirements

- 1. The question must have an answer. If the student thinks there is insufficient evidence, it must be because they haven’t found the evidence. After careful analysis, provide your guidance for the student’s current step, with the goal of helping the student actually answer the question correctly.
- 2. Since this step may not necessarily be wrong, please carefully choose your wording to prevent your guidance from introducing bias.
- 3. Your analysis must include a brief review of the **problem** that the student needs to solve, emphasizing the content of the problem to the student to prevent answering off-topic.
- 4. The guidance you provide will be given to the student together with the tool call results after this step ends

- If the student generates a tool call in this step, the guidance will be given to the student together with the tool return value - If the student generates an answer in this step, the guidance will be given directly to the student, and the student will

choose whether to answer again or continue to generate tool calls based on your guidance

- 5. It is forbidden to find answers on behalf of the student, and it is forbidden to hint at what the answer is under any circumstances. **You are a teacher, not someone helping students cheat**.
- 6. It is forbidden to provide **direct clues** to students. Your purpose is only to **guide**.
- 7. Guidance should be clear and easy to understand. If necessary, you can encourage students to continue calling tools or switch tools.

# Output Format Strictly follow the markdown format below for output ˋˋˋ # Analysis combining student’s current behavior to provide appropriate guidance for the present moment Write your detailed analysis here # Guidance Content Write your guidance content here ˋˋˋ

Output:

Table 16: Prompt for generating guidance experience directly without referencing the experience base.

###### Prompt for model-based guidance triggering decision.

# Overall Instructions You are a teacher who is very good at guiding students to answer difficult questions. Now, the student has made some attempts on the given question, but there may be some non-absolute yet potential issues or even errors in the current step. I will provide you with the student’s multi-turn attempts from the beginning to the current point, where “assistant” represents the student’s responses and "user" represents environmental feedback, which may also include the teacher’s prior guidance. Your purpose is to determine whether guidance is needed at the current step.

# Student’s Multi-turn Interactions ˋˋˋ {history} ˋˋˋ

# Your Task and Requirements Analyze whether the student’s behavior in the current step has potential issues, and provide a signal indicating whether guidance is needed.

# Output Format Strictly follow the markdown format below for output ˋˋˋ # Your Analysis Your analysis content here

# Whether to Provide Guidance yes/no ˋˋˋ

Output:

Table 17: Prompt for model-based guidance triggering decision.

Prompt for retrieving and directly using the experience base.

Your current situation resembles a mistake made by a previous student. That student’s behavior was: {behavior}

Their mistake was: {mistake}

My guidance to them was: {guidance}

Learn from this error, examine your own behavior, and proceed with better decision-making.

Table 18: Prompt for retrieving and directly using the experience base.

Prompt for evaluation.

You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer. As long as the Labeled Answer is contained within the Predicted Answer, it is considered correct, even if additional, more detailed explanations are included. You should focus on whether the Predicted Answer truly answers the question correctly.

# Question: {question}

# Labeled Answer: {correct_answer}

# Predicted Answer: {response}

Respond **only with** "Correct" or "Incorrect", no other tokens.

Table 19: Prompt for evaluation.

