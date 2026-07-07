## Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning

# arXiv:2511.19900v2[cs.CV]26Nov2025

Jiaqi Liu1 Kaiwen Xiong1 Peng Xia1 Yiyang Zhou1 Haonian Ji1 Lu Feng1 Siwei Han1 Mingyu Ding1 Huaxiu Yao1 1UNC-Chapel Hill

##### Abstract

Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of humanannotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely textbased self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a selfevolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into selfevaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-externalreward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at https://github.com/aiming-lab/Agent0.

##### 1. Introduction

Vision-language agents have shown remarkable capabilities in tackling complex multimodal tasks [54], including robotic manipulation [21], visual question reasoning [80], and scientific discovery [28]. Currently, most vision-language agents

and Vision-Language Models (VLMs) are trained using human-annotated preference data [20, 48, 57] or externalreward signals. However, such training paradigms are inherently constrained by the limitations of human annotators’ preferences and the sparsity or incompleteness of environment-generated feedback, ultimately capping the upper bound of the agent’s capabilities [54, 55]. To enable agents to move beyond static human supervision and achieve continuous self-evolution, recent research has explored selfrewarding learning [63, 81], in which the agent or model itself acts as a Critic or Reward Model, providing feedback and reward signals for its own learning [9, 55].

Nevertheless, for many complex visual reasoning tasks, purely text-based self-evaluation faces two key limitations. First, limited evaluation capability: when relying solely on textual reflection, models struggle to verify complex multistep computations, spatial reasoning, or precise physical and geometric calculations, abilities that are critical for tasks such as visual geometry and chart analysis [64]. Second, unreliable evaluation process: excessive textual reasoning causes models to rely on language shortcuts, bypassing finegrained visual understanding and depending instead on linguistic priors or contextual bias [27, 81]. This often leads to evaluation hallucination, where the model incorrectly rewards a linguistically plausible yet visually incorrect answer, or penalizes a visually correct answer that fails to align with its language-based expectations.

To address these challenges, inspired by recent advances in tool-integrated reasoning [13, 48, 65, 76], we propose a simple yet effective idea: enable the model to employ external tools not only during reasoning, but also during its own self-evaluation and self-repair. By doing so, the model can perform closed-loop self-improvement under zero external reward supervision, learning to analyze, critique, and refine its reasoning in a verifiable manner. Building on this idea, we present Agent0-VL, a self-evolving visionlanguage agent framework that integrates tool-use paradigms into both the reasoning and self-evaluation processes. As illustrated in Figure 1, Agent0-VL unifies reasoning, verification, and self-repair within a single LVLM, which alternates

[Figure 1]

[Figure 2]

Qwen2.5-VL-7B

[Figure 3]

[Figure 4]

Qwen2.5-VL-7B-TIR Agent0-VL-7B (Ours)

[Figure 5]

[Figure 6]

80

Base Policy Agent0-VL

Self-Evolution

+11.5%

+12.2%

[Figure 7]

[Figure 8]

Performance(Pass@1)

75.6

72.9

70

Solver

Solver

Self-Repair

68.1

67.8

#### …

67.2

65.0

###### +4.3%

60

61.1 59.6

[Figure 9]

[Figure 10]

58.6

+14.7%

Evaluation

Verifier

Verifier

53.1

50

#### …

[Figure 11]

[Figure 12]

47.2

Iteration 1

Iteration N

46.3

40

MathVista

MathVerse

MMMU

HallBench

（a) （b)

- Figure 1. The evolve loop of Agent0-VL and its performance comparison. The left part illustrates the iterative evolution between the Solver and Verifier, where the Solver progressively refines reasoning strategies under Verifier feedback. The right part presents results showing that Agent0-VL outperforms tool-integrated reasoning methods across multiple representative benchmarks. TIR: Tool-Integrated Reasoning.

between two synergistic roles: (1) the Solver, which performs multi-turn reasoning and selectively invokes external tools for grounded computation and visual perception; and

a partially observable Markov decision process (POMDP): M = (S,A,O,T,R,γ), where γ is the discount factor, R is the reward function, which will be introduced in section 3. The whole multimodal reasoning dynamics and tool feedback are explicitly modeled through the following components:

- (2) the Verifier, which validates intermediate reasoning steps through generative critique and tool-based feedback, generating fine-grained reward signals and repair instructions.

By alternating between these roles, the model forms a closed feedback loop that supports continual selfimprovement. We formalize this iterative learning paradigm as a Self-Evolving Reasoning Cycle (SERC). In the inner loop, the model performs multi-turn reasoning, tool-based verification, and selective repair to progressively refine its reasoning trajectory. In the outer loop, we employ reinforcement learning, specifically, Group Relative Policy Optimization (GRPO) [44], to update the shared policy, jointly enhancing reasoning and verification capabilities over time. This process transforms the learning objective from static reward maximization into a process of distributional selfconsistency, where reasoning and evaluation behaviors are jointly aligned and continually optimized.

State Space. st ∈ S denotes the latent multimodal reasoning state, encoding the textual reasoning context, visual features, and past tool input and output traces.

Action Space. Each action at ∈ A can be either: (1) a textual reasoning step a(ttext), or (2) a structured tool invocation a(ttool), which executes an external program, such as Python code. Hence, A = Atext ∪ Atool.

Observation Space. An observation ot ∈ O contains feedback from external tools or environment, such as returned numerical results or textual retrievals. The transition function T(st+1|st,at,ot) captures how state evolves given generated response and its corresponding tool feedback.

Trajectory. Given an input x = (I,q), the model generates a multimodal trajectory: τ = {(s1,a1,o1),(s2,a2,o2),...,(sT,aT,oT)}, where each transition encodes one reasoning-tool-feedback interaction.

Our main contribution is Agent0-VL, a unified selfevolving LVLM that integrates reasoning, verification, and self-repair into a single model trained entirely from zero external rewards. Experiments on geometric reasoning and visual scientific analysis demonstrate that Agent0-VL-7B achieves stable multi-iteration performance improvement, showing an average 12.5% improvement over the Qwen-VL base model. Furthermore, when used independently as a process reward model, Agent0-VL also improves the model’s test-time scaling performance by an average of 7.3%.

##### 3. Methodology

In this section, we introduce Agent0-VL, a self-evolving vision-language agent that integrates tool-use paradigms into both the reasoning and self-evaluation processes. The core idea is to enable a VLM to not only reason and solve problems, but also to verify, critique, and repair its own reasoning trajectories through a unified, self-evolving loop. We first describe the dual-role architecture composed of a Solver and a Verifier, then present how the model performs tool-grounded reasoning, verification, and confidence-gated self-repair. Finally, we detail the Self-Evolving Reasoning Cycle (SERC), where these roles interact through inner and outer optimization loops to achieve continual self-improvement.

##### 2. Preliminaries

Multi-turn Tool-Integrated Reasoning (TIR) agents trained via Reinforcement Learning (RL) present substantial challenges, primarily due to the inherently compositional nature of reasoning and the partial observability of multimodal environments. To address these complexities, we formulate the reasoning process of our agent within the framework of

[Figure 13]

[Figure 14]

[Figure 15]

###### Verifier

[Figure 16]

###### Agent0-VL

[Figure 17]

[Figure 18]

[Figure 19]

###### Input

[Figure 20]

Reasoning Output Step 1 Step 2 … Step n-1 Step n Answer

Query + Image

[Figure 21]

[Figure 22]

[Figure 23]

###### Solver

[Figure 24]

Evaluate

[Figure 25]

Sandbox

###### Tool calling

[Figure 26]

Think

Tool List

Step 1 Step 2 … Step n

###### Step n-1

Answer

… Score n-1 Score n Answer Score

Score 1 Score2

[Figure 27]

[Figure 28]

Tool calling

Tool calling

Critique Score Confidence

[Figure 29]

[Figure 30]

𝑜 = 𝑇𝑜𝑜𝑙(𝑎 )

###### Self-Repair

[Figure 31]

[Figure 32]

[Figure 33]

Code Sandbox

Fail

Verification Decision

𝜏 = { 𝑠 ,𝑎 ,𝑜 ,…,(𝑠 ,𝑎 ,𝑜 )}

Pass

[Figure 34]

Policy Update

[Figure 35]

g(𝜏) = 𝛼 · 𝑟 + · 𝛴 (𝛼 𝑟 + 𝛼 · 𝑟 − 𝛼 · 𝐶)

[Figure 36]

Reinforcement Optimization

Reward

- • Verify reward𝑟
- • Cost penalty 𝐶

- • Tool reward 𝑟
- • Outcome reward 𝑟 𝜏

- Figure 2. The Framework of Agent0-VL. The unified policy πθ alternates between two internal roles: the Solver that generates reasoning trajectories with tool calls, and the Verifier that performs generative verification using tool feedback to produce critiques and step-wise rewards. These roles are jointly optimized through the Self-Evolving Reasoning Cycle, where self-generated rewards guide policy updates via RL.
- 3.1. Unified Solver-Verifier Architecture

triplet (st,at,ot), the Verifier outputs a feedback tuple:

To achieve autonomous evolution, as shown in Figure 2, Agent0-VL adopts a unified dual-role design, where a single LVLM alternates between two internal modes: a Solver that performs tool-integrated reasoning, and a Verifier that introspectively evaluates, critiques, and repairs the Solver’s outputs. We detail the architecture as follows:

Vt = (scoret,conft,critiquet),

where scoret ∈ [−1,1] measures factual correctness, conft ∈ [0,1] estimates epistemic certainty, and critiquet provides natural-language reflection describing potential reasoning flaws. The Verifier can also re-invoke tools to crosscheck correctness, providing a grounded and interpretable signal for self-evaluation.

Unified Policy Formulation. We define a shared policy πθ that governs both roles through a role indicator m ∈ {S,V}:

###### 3.2. Tool-Grounded Verification and Self-Repair

πθS(at|st), m = Solver, πθV(at|st,at,ot), m = Verifier,

(1)

πθ(at|st,m) =

Within the dual-role framework, the Solver performs multiturn reasoning by invoking external tools, while the Verifier provides process-level feedback and evaluation based on the Solver’s reasoning trajectory and outputs. If the Verifier identifies deficiencies or inconsistencies in the Solver’s reasoning, it initiates a self-repair process that leverages self-evaluation signals to revise and refine the reasoning trajectory, thereby improving reasoning accuracy.

where st denotes the multimodal state, at the generated reasoning action, and ot the tool or environment feedback.

Solver. When m = S, the model performs multi-turn reasoning and selectively invokes external tools. The resulting observations ot are incorporated into the context, allowing the agent to iteratively refine its reasoning with grounded evidence. Formally, the Solver follows

Specifically, Agent0-VL introduces a structured toolgrounded generative verification mechanism designed to produce dense and interpretable feedback signals for reinforcement learning. At each step t, the Verifier produces Vt. During verification, the Verifier can query external tools to obtain factual evidence. By combining language-based reflection with executable tool-derived evidence, the model transforms verification from a static correctness check into a

at ∼ πθS(at|st), bt+1 = f(bt,ot),

where bt represents the latent belief state encoding accumulated reasoning context and multimodal information.

Verifier. When m = V, the model switches into a generative verification mode to evaluate each reasoning step. Given the

###### Algorithm 1 Agent0-VL Training Process Require: Unified policy πθ, reference policy πθ

, confidence threshold τc.

old

- 1: Initialize πθ with supervised data to learn tool usage and verification formats.
- 2: for each iteration k = 1,...,Niter do
- 3: // Inner Loop: Generation and Self-Evaluation
- 4: for each task xi = (Ii,qi) do
- 5: Initialize trajectory τi ← ∅, state s1.
- 6: for t = 1,...,T do
- 7: Sample action at ∼ πθ(·|st,m = S) and execute to get ot.
- 8: Generate verification Vt = (scoret,conft,critiquet) ∼ πθ(·|st,at,ot,EE).
- 9: Compute process reward rproc(t) (Eq. 2).
- 10: if conft < τc then
- 11: Generate repair ∆t = fθ(st,at,Vt); resample a′t ∼ πθ(·|st,∆t,RA).
- 12: end if
- 13: Compute effective step reward(Eq. 4).
- 14: end for
- 15: Compute total trajectory return g(τi) (Eq. 5).
- 16: end for
- 17: // Outer Loop: Policy Update
- 18: Optimize πθ via GRPO using all collected trajectories (Eq. 6).
- 19: end for

dynamic evaluation procedure, enabling iterative refinement of its reasoning trajectories.

Based on the Verifier’s step-wise assessments, we further design a corresponding process-level reward that integrates semantic reliability, tool-based validation, and cross-role regularization.

rproc(t) =λtool · r(toolt) + scoret · conft

(2)

semantic reliability − βdiv · DKL πθV∥πθE ,

where λtool scales tool-based correctness, and βdiv stabilizes the dual-role distributional alignment.

After completing self-verification, the Verifier determines whether the model should perform self-repair to correct its reasoning process based on the confidence score conft obtained during verification. Let τc denote a confidence threshold. The repair gate at step t is defined as:

gt = σ κ(τc − conft) , (3)

where σ(·) is the sigmoid function, and κ controls the gating temperature. When gt is activated, the Verifier issues a local repair instruction ∆t = fθ(st,at,Vt), and the Solver regenerates a corrected segment a′t ∼ πθ(· | st,∆t,m = S).

Based on the above verification and self-repair processes, the resulting step-wise reward is then defined as:

rt = rproc(t) − gt · Crepair(t) , (4) where Crepair(t) penalizes unnecessary repairs.

###### 3.3. Self-Evolving Reasoning Cycle

Having defined the agent’s architecture and verification mechanisms, we now introduce the model training process (see Algorithm 1 for the whole process). Specifically, the training process is divided into two stages. First, a supervised fine-tuning (SFT) stage for cold-start initialization, ensuring the model learns proper tool usage and self-evaluation formats. Second, a RL-based self-evolving reasoning cycle (SERC) that operationalizes this architecture, enabling the agent to learn from its own tool-grounded feedback. SERC consists of two nested loops: an inner loop for data generation and an outer loop for policy evolution.

The inner loop is responsible for generating experience by coupling reasoning and verification. For a given multimodal task x, the Solver first generates a complete reasoning trajectory τ = {(st,at,ot)}Tt=1, invoking external tools as needed. Immediately following this, the Verifier re-evaluates each step t of the trajectory to produce the verification tuple Vt and the corresponding process reward rproc(t) (using Eq. 2). If the Verifier’s confidence for a step is below the threshold τc, the selective repair mechanism (Eq. 4) is triggered, and a cost Crepair is applied. Finally, the total return g(τ) for the trajectory is aggregated, integrating both the final outcome reward rout and the accumulated step-wise process rewards:

T

γt−1rt, (5)

g(τ) = αout rout +

t=1

where αout balances the two reward types and γ is the discount factor.

The outer loop then uses these generated trajectories to evolve the unified policy πθ. We employ Group Relative Policy Optimization (GRPO) [44], a variant of PPO designed

for generative tasks. For a group of G trajectories {τi}Gi=1 sampled under the current policy, we compute a normalized advantage for each trajectory:

g(τi) − mean(g1,...,gG) std(g1,...,gG) + ε

Aˆi =

,

where gi = g(τi) and ε is a normalization constant. This relative advantage Aˆi reframes the learning objective as ”being better than the group average.” The policy is then optimized to increase the likelihood of trajectories with positive advantages while maintaining stability through a KL regularization term:

LEDLP = −Ei min ρiAˆi, clip(ρi,1 − ϵ,1 + ϵ)Aˆi

+ βKL DKL πθ∥πθ

,

old

(6)

where ρi = π

θ(τi)

πθold(τi) is the importance sampling ratio.

[Figure 37]

Pass @1 BoN with Qwen2.5-VL-7B BoN with Agent-Zero-VL-7B

OverallPerformance

69.1

##### 4. Experiments

70

66.9

64.4

62.8

In this section, we evaluate Agent0-VL across multiple visual reasoning benchmarks to answer the following questions: (1) Can it improve reasoning ability over existing opensource baselines? (2) Does repeated self-evolution through multiple iterations yield consistent performance gains? (3) How effective are the proposed components? and (4) Can Agent0-VL serve as a process reward model to enhance other LVLMs?

58.3 59.5

60

57.2

55.3

53.6

53.0

51.2

50.0

50

QwenVL2.5-7B

QwenVL2.5-3B

Qwen2.5-VL-32B

InternVL-2.5-8B

Figure 3. The overall Best-of-8 evaluation results across seven multimodal reasoning benchmarks with different critic models. Our model greatly enhances the overall performance compared with Qwen2.5-VL-7B model.

###### 4.1. Experimental Setup

Benchmarks. To evaluate the effectiveness of our method, we conducted experiments on a set of seven benchmarks. The science and mathematical benchmarks include MathVerse [74], MathVision [53], MathVista [31], WeMath [40], and MMMU [70], while other benchmarks consist of HallusionBench [16], and ChartQA [32].

tool-integrated variant Qwen2.5-VL-7B-TIR. Even with a stronger base model (Qwen3-VL-8B), Agent0-VL maintains strong compatibility and further improves performance, surpassing the base by 6.1% and its tool-augmented variant (Qwen3-VL-8B-TIR) by 4.6%. Notably, Agent0-VL-8B also outperforms closed-source systems such as GPT-4o on key benchmarks including MathVista, HallBench, and ChartQA, underscoring the generalization and reasoning strength of our approach.

Baselines. Additionally, we compared Agent0-VL against three types of baselines: (1) Closed-Source LVLMs, including GPT-4o [1], o1 [35], Gemini-2.0-pro [7], and Claude-

- 3.7-Sonnet[2]; (2) Open-Source General MLLMs, including Qwen2.5-VL-3B [3], Qwen2.5-VL-7B [3], Qwen2.5VL-32B [3], InternVL-2.5-8B [6] and InternVL3-8B [84];

(3) Open-Source Reasoning MLLMs, including R1VL-7B [71],Vision-R1-7B [20], R1-Onevision-7B [66], OpenVLThinker-7B [8], VLAA-Thinker-7B [4], MMEureka-7B [33], Thyme [76], Qwen3-VL-8B [49], and ThinkLite-VL-7B [58]; and (4) self-evolving methods, including EvoLMM [51], VisPlay [18], and Vision-Zero [55]. Training Datasets. We construct a large-scale multimodal reasoning corpus tailored for both SFT and RL stages. The SFT dataset (200k samples) is built from open-source benchmarks including Geometry3K [30], GeoQA [5], Mulberry dataset [68], MM-Eureka [33], and Retool [11], where we automatically generate tool-augmented reasoning trajectories using GPT-5 and Qwen2.5-VL-72B as teacher models. For the RL stage, we construct an additional 40k dataset. Detailed data construction procedures are provided in Appendix D.

- 4.2. Main Results

Performance Across Task Domains. In domain-specific evaluations, Agent0-VL demonstrates the most significant improvements on mathematical reasoning benchmarks such as MathVista and WeMath, where tool-grounded execution and verification play a crucial role in accurate symbolic reasoning. Our 7B and 8B models achieve 18.1% and 7.4% improvements, respectively, over their base models on mathrelated benchmarks. For perception-heavy benchmarks such as HallusionBench and ChartQA, integrating the Verifier’s factual grounding substantially reduces visual hallucinations, leading to 12.2% and 3.1% improvements compared to the base model. These results indicate that Agent0-VL enhances both symbolic reasoning and visual understanding, confirming the robustness and generality of our framework across diverse task domains.

Iterative Self-Evolution. We further investigate how the model’s reasoning capabilities evolve across multiple iterations of SERC. As shown in Table 2, Agent0-VL demonstrates a steady and monotonic improvement over time: in the first iteration, its overall performance increases by 5.2% compared to the base model, followed by additional gains of 4.0% and 2.8% in the second and third iterations, respectively. These results validate the effectiveness of our selfevolving framework, showing that the model can achieve stable and continuous performance gains through iterative self-improvement.

Overall Performance. Table 1 summarizes the performance of Agent0-VL across all visual reasoning benchmarks. Our model consistently achieves substantial improvements over all open-source baselines. Specifically, Agent0-VL-7B outperforms existing open-source 7B models, achieving a 4.29% average gain over ThinkLite-VL-7B. Agent0-VL also demonstrates clear advantages over existing self-evolving vision–language models, including Vision-Zero [55] and EvoLMM [51]. Compared to the base Qwen2.5-VL-7B, it shows a 12.5% improvement, and a 10.3% gain over its

- Table 1. Comparison of model performance across representative visual reasoning benchmarks. Closed-source LVLMs are listed for reference, while open-source models are grouped by general-purpose and reasoning-oriented categories. The best results among all open-sourced models are bolded and the second best results are underlined in the table. Note: Qwen2.5-VL-7B-TIR and Qwen3-VL-8B-TIR denote the results of integrating tool-enhanced reasoning into the corresponding base models, obtained from our local fine-tuning and evaluation.

Model MathVerse MathVision MathVista WeMath HallBench ChartQA MMMUval Avg.

Close-source MLLMs

GPT-4o [1] 50.8 30.4 63.8 68.8 55.0 85.7 69.1 60.5 OpenAI-o1 [35] 57.0 60.3 73.9 - - 83.1 77.6 Claude-3.7-Sonnet [2] 52.0 41.3 66.8 72.6 55.4 56.5 75.0 59.9

Open-Source General MLLMs

InternVL-2.5-8B [6] 39.5 19.7 64.4 53.5 61.7 79.1 62.7 54.4 InternVL-3-8B [84] 39.8 29.3 71.6 58.1 64.3 85.9 60.7 58.5 Qwen2.5-VL-7B [3] 46.3 25.1 67.8 62.1 65.0 83.5 58.6 58.3 Qwen2.5-VL-7B-TIR∗ 47.2 26.3 68.1 63.7 67.2 84.1 59.6 59.5 Qwen3-VL-8B [49] 62.1 53.9 77.2 72.5 72.1 84.6 69.6 70.3 Qwen3-VL-8B-TIR∗ 63.1 54.7 79.4 73.1 72.8 85.4 70.9 71.3

Open-Source Reasoning MLLMs

R1-VL-7B [71] 40.4 24.7 63.5 60.1 54.7 76.1 - Vision-R1-7B [20] 51.9 30.7 73.5 73.9 68.8 79.8 50.5 61.3 R1-OneVision-7B [66] 46.4 29.9 64.1 61.8 67.5 77.8 - OpenVLThinker-7B [8] 45.7 26.3 71.2 66.7 70.2 78.4 - VLAA-Thinker-7B [4] 52.7 29.2 69.7 70.2 68.2 80.1 - MM-Eureka-Qwen-7B [33] 50.5 27.9 73.6 67.4 66.9 82.1 52.7 60.2 ThinkLite-VL-7B [58] 52.1 32.9 75.1 69.3 70.9 84.8 55.5 62.9 Thyme-VL-7B [76] 51.3 27.6 70.0 - 71.0 86.1 - EvoLMM [51] 44.9 27.8 70.5 86.7 52 VisPlay [18] - 31.2 - - 92.3 - - Vision-Zero [55] 52.2 28.1 72.6 - - - - -

- Agent0-VL-7B (Ours) 53.1 37.3 75.6 71.7 72.9 87.3 61.1 65.6

- Agent0-VL-8B (Ours) 65.5 56.2 83.7 79.6 74.3 89.7 73.4 74.6

- Table 2. Performance comparison of Agent0-VL-7B across iterative training stages. Each iteration (Iter 1-3) progressively refines reasoning and tool-grounded capabilities, consistently outperforming the base model across all benchmarks.

Model Name MathVerse MathVision MathVista WeMath HallBench ChartQA MME-Real MMMU Avg. Base Model 46.3 25.1 67.8 62.1 65.0 83.5 58.3 50.6 57.3

- Iter 1 48.4 29.6 69.2 66.8 67.9 84.7 63.9 53.7 60.5
- Iter 2 51.1 35.3 72.8 70.1 70.3 86.1 64.7 58.3 63.6
- Iter 3 53.1 37.3 75.6 71.7 72.9 87.3 65.3 61.1 65.5

- Table 3. Ablation study of Agent0-VL on different benchmarks. Removing Self Repair, Tool-Grounded Reward, or SERC notably degrades performance, highlighting the complementary roles of these modules.

Setting Math Avg. HallBench ChartQA MME-Real MMMU

Agent0-VL-7B 59.4 72.9 87.3 65.3 61.1 w/o Self Repair 57.5 71.6 86.1 64.1 57.9 w/o Tool Use 53.1 67.5 86.2 61.9 54.7 w/o SERC (SFT only) 51.8 65.8 85.4 60.5 52.5 Qwen2.5-VL-7B(Base Model) 50.3 65.0 83.5 58.3 50.6

- 4.3. Ablation Studies

learning (ii) Tool Use, and (iii) Self Repair. For each ablation, we remove the corresponding module while keeping all other components intact, and retrain the model under the same settings. The results are summarized in Table 3.

To understand the contribution of each major component in Agent0-VL, we conduct controlled ablation experiments focusing on three key modules: (i) the SERC for reinforcement

We report the results in Table 3. According to the results, first, we observe that eliminating the outer reinforcement learning loop (SERC) and relying solely on SFT training leads to the most significant performance drop, with an average decrease of 8.7% across all benchmarks. This result confirms that the bidirectional interaction between reasoning and evaluation is essential for enabling long-term self-evolution, which constitutes one of the core contributions of this work. Second, when tool usage is removed and the model performs only text-based reasoning and self-evaluation, the average performance declines by 6.5%. This highlights the crucial role of tool integration in enhancing reasoning accuracy, especially for mathematical and fact-verification tasks. Finally, removing the self-repair mechanism, allowing the model to self-evaluate without performing explicit correction, results in a moderate average performance drop of 2.5%. In particular, for mathematical reasoning benchmarks, re-executing reasoning through selective repair provides additional opportunities for resampling and correction, thereby improving overall reliability.

###### 4.4. Performance as a Process Reward Model

To further evaluate the generalization and standalone utility of the Verifier module, we deploy it as a Process Reward Model (PRM) to assess reasoning trajectories generated by other VLMs. This experiment serves two key purposes: (i) to examine whether the Verifier can generalize beyond Agent0-VL and reliably evaluate the step-level correctness of external models’ outputs, and (ii) to validate the effectiveness of our tool-grounded reward modeling approach, even when decoupled from policy learning.

As shown in Figure 3, our verifier significantly improves Best-of-8 (BoN) performance when used as a reward scorer across various LVLMs. Compared to Qwen2.5-VL-7B as the critic model, Agent0-VL consistently yields stronger trajectory selection across models of different scales, from Qwen2.5-VL-3B up to Qwen2.5-VL-32B, demonstrating more effective reward assignment and sharper step-level discrimination. Table 4 further highlights the effectiveness of our model, showing substantial improvements in both steplevel reward correlation and overall accuracy across seven multimodal reasoning benchmarks. When integrated as a PRM into different open-source LVLMs, Agent0-VL consistently enhances reasoning stability and factual grounding, yielding an average gain of 7.3%. Notably, even smaller models such as Qwen2.5-VL-3B benefit significantly from the structured feedback, indicating strong generalization across architectures and model scales.

###### 4.5. Case Study

To illustrate how Agent0-VL performs self-evolving reasoning, we present a representative visual reasoning case in Figure 4 (full version in Appendix 4.5). Unlike conven-

tional LVLMs that stop after producing an incorrect answer, Agent0-VL continues to introspect and repair its reasoning. In this example, the Solver initially misinterprets the “blind spot” segment, leading to an incorrect result. The Verifier detects this logical error using tool-grounded verification and provides structured feedback pinpointing the faulty step. The Self-Repair module then synthesizes a corrective patch, which the Solver reuses to regenerate a valid reasoning chain and reach the correct answer. This case demonstrates how Agent0-VL converts errors into successful reasoning through its integrated reasoning–evaluation–repair loop.

##### 5. Related Work

Self-Evolving Methods. To reduce the reliance on costly human supervision, recent research in both LLMs and VLMs has explored self-evolving techniques that enable models to improve autonomously using unlabeled data and selfgenerated feedback [62]. In LLMs, a common strategy is to derive pseudo-rewards from the model’s own outputs. For example, self-consistency leverages agreement among multiple generated solutions as a learning signal [56], which has enhanced reasoning in tasks like math and code generation, as seen in TTRL [85], MM-UPT [59], and SRT [43]. MM-UPT further improves spatial reasoning by generating synthetic geometry problems. Other approaches use a strong model as an automated evaluator [38], or rely on internal cues such as confidence scores [25, 78] and output entropy [73]. DeepConf [12], for instance, enhances reasoning efficiency by selecting responses based on token entropy.

In the VLM domain, self-evolving agents are also progressing rapidly. ViPER [72] proposes a two-stage coarseto-fine training framework that integrates instance- and image-level reconstruction with self-critiquing and prediction. In contrast, Vision-Zero [55] employs a domainagnostic, game-based self-play strategy. By alternating between self-play and reinforcement learning with verifiable rewards, it achieves scalable and sustained improvement. These methods demonstrate that VLMs, like LLMs, can evolve through structured feedback loops involving uncertainty modeling, task generation, and curriculum-based learning. In comparison, our work builds on these advances by introducing a tool-integrated visual reasoning and selfevaluation framework, enabling continual and stable performance improvements.

Tool-Integrated Reasoning. Tool-Integrated Reasoning (TIR) aims to enable models to invoke external tools (e.g., search engines [14, 37, 50], calculators [15, 52]) to overcome knowledge limitations and execute complex tasks [13, 17, 42, 69, 82, 83]. Subsequent work significantly enhanced LLM capabilities in multi-tool coordination, planning, and execution through instruction-tuning and agentic frameworks [22, 26, 39, 41]. Recent research began to ex-

Table 4. Performance of the Agent0-VL as a PRM. “+Ours” denotes models integrated with the Agent0-VL for reward scoring. Across diverse multimodal reasoning benchmarks, our method consistently enhances accuracy, validating its effectiveness as a generalizable, tool-grounded reward model.

###### Model MathVerse MathVision MathVista WeMath HallBench ChartQA Overall

Qwen2.5-VL-3B [3] 34.8 21.9 58.4 51.7 59.8 73.1 50.0 +Ours 38.9 26.1 65.8 54.2 61.2 75.2 53.6

Qwen2.5-VL-7B [3] 46.3 25.1 67.8 62.1 65.0 83.5 58.3 +Ours 51.2 33.6 72.3 66.9 68.1 84.6 62.8

InternVL2.5-8B [6] 39.5 19.7 64.4 53.5 61.7 79.1 53.0 +Ours 43.2 26.2 67.3 59.8 63.6 83.0 57.2

InternVL2.5-8B [6] 39.8 29.3 71.6 58.1 64.3 85.9 58.2 +Ours 45.4 33.0 74.6 62.5 67.2 88.3 61.8

Qwen2.5-VL-32B [3] 48.5 38.4 74.7 69.1 71.8 84.0 64.4 +Ours 53.0 44.3 78.6 75.2 74.7 88.5 69.1

[Figure 38]

[Figure 39]

###### Phase 1: Initial Reasoning

###### Phase 2: Evaluation

- {"step_index":1, "score":-1.0, "confidence":1.0, "critique":"Reasoning contains a critical logic error. The definition of 'blind spot' is wrong… Therefore, the true blind spot is Q4 ($x > 0, y < 0$).", "tool_check":false}
- {"step_index":2, **}

<think> The goal is to calculate…The description "from due East (positive x-axis) counter-clockwise by $270^\circ$, ending at due North (positive y-axis)" is ..One possible interpretation is that …

[Figure 40]

[Figure 41]

###### Error Found

[Figure 42]

[Figure 43]

3. I need to (a) Check if segment A-B enters Q2. * (b) Check if segment B-C enters Q2. * (c) Sum the results. (a) Segment A-B: …. (b) Segment B-C: From $B(7, -1)$ [Q4] to $C(-4, -3)$ [Q3]. Both points have $y < 0$. Therefore, the entire segment is in the $y < 0$ region and **cannot enter Q2**. Length in blind spot is $0$. (c) Total length = $0 + 0 = 0$. </think> <answer>0.00 </answer>

[Figure 44]

###### Phase 3: Self-Repair

EE Key Feedback:…<think> 1. Diagnosis:*EE has assigned a score ... 2. Root Cause: RA's definition of the "blind spot" was incorrect. 3. EE Correction: …4. Repair Strategy: I must issue a `PATCH` to correct the core logical premise…</think> { "action": "PATCH", "target_step": 1, "patch_type": "text",justification": "Original logic incorrectly…}

Failed Attempt

[Figure 45]

Question: As shown in the figure, a coast guard watchtower is located at the coordinate origin O(0, 0). Three key reef locations, A, B, and C, are marked on the nautical grid... Calculate how much of the ship's total journey (segments A-B and B-C) is sailed within the watchtower's blind spot.

[Figure 46]

###### Phase 4: Corrected Reasoning

<think> Goal: Restart reasoning based on the `PATCH`. 2. Corrected Premise: Blind Spot = Quadrant 4 (Q4, $x > 0, y < 0$). 3. New Plan…</think> {"tool_name":"calc_intersection_and_distance", "tool_input":..} à [Tool Output]

- <think> **Integrate (a):** Tool returns length of A-B in Q4 as **$1.01015$**. 6. **Execute (b) -…* I will call the tool. </think>{"tool_name":"calc_intersection_and_distance", "tool_input":..} à [Tool Output]
- <think> **Integrate (b):** Tool returns length of B-C in Q4 as **$7.11601…</think>

Correct

<answer> 8.13 </answer>

Figure 4. Simplified illustration of Agent0-VL’s self-evolving reasoning process on a geometric reasoning task during training phase. The model first produces an incorrect answer (Phase 1), after which the Verifier identifies the logical error (Phase 2), triggers Self-Repair to generate a correction (Phase 3), and finally re-executes reasoning via the Solver to reach the correct solution (Phase 4). The complete multi-phase case is provided in Appendix 4.5 (Figure 8).

tend TIR to VLMs, enabling models to not only process text but also dynamically interact with visual information. One line of work positions the VLM as an orchestrator that calls upon external vision expert tools [19, 67] or skill repositories [29]. Another line of work explores how VLMs can perform dynamic reasoning at the “vision level,” treating image exploration itself as a tool, like zooming [45] and visual querying [61]. Recently, many efforts have adopted Reinforcement Learning (RL) for training [46–48]. GRIT [10] and DeepEyes [79] leverage RL to incentivize the model to actively cite image regions (e.g., bounding boxes) within its reasoning chains. Also some works (e.g., WebWatcher [13]) utilize RL to enhance the generalization of tool use for multimodal agents in deep research tasks [34, 60]. Building on this line of work, our method further integrates toolaugmented reasoning into the model’s self-evaluation process, enabling the generation of process-level reward signals

that guide more effective self-improvement.

##### 6. Conclusion

We presented Agent0-VL, a self-evolving vision–language agent that unifies reasoning, verification, and self-repair within a single model. Through its dual roles, the Solver and the Verifier, Agent0-VL operates under the Self-Evolving Reasoning Cycle, where the model continually refines its reasoning through tool-grounded verification, confidence-gated repair, and reinforcement learning. Empirically, Agent0VL outperforms existing open-source models across a wide range of visual reasoning benchmarks, achieving stronger factual consistency and multi-turn stability.

##### Acknowledgement

This work is partially supported by the AI for Math Fund from Renaissance Philanthropy, Amazon Research Award,

and Cisco Faculty Research Award. The Authors also acknowledge the National Artificial Intelligence Research Resource (NAIRR) Pilot, Purdue Anvil AI for contributing to this research result.

##### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report, 2024.
- [2] Anthropic. Claude 3.7 Sonnet, anthropic.com. https:// www.anthropic.com/claude/sonnet, 2025. 202502-24.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report, 2025.
- [4] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large visionlanguage models, 2025.
- [5] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning, 2022.
- [6] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025.
- [7] Google Deepmind. Gemini 2.5: Our most intelligent AI model — blog.google. https://blog.google/ technology/google-deepmind/gemini-modelthinking-updates-march-2025/#gemini-2-5thinking, 2025. Accessed: 2025-03-25.
- [8] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement, 2025.
- [9] Yi Ding and Ruqi Zhang. Sherlock: Self-correcting reasoning in vision-language models. arXiv preprint arXiv:2505.22651, 2025.
- [10] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879, 2025.
- [11] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025.
- [12] Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.
- [13] Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, et al. Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025.

- [14] Google. Try deep research and our new experimental model in gemini, your ai assistant, 2024.
- [15] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.
- [16] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. pages 14375–14385, 2024.
- [17] Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, and Yang Liu. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. arXiv preprint arXiv:2403.07714, 2024.
- [18] Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, and Yonghui Yang. Visplay: Self-evolving vision-language models from images. arXiv preprint arXiv:2511.15661, 2025.
- [19] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024.
- [20] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025.
- [21] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0. 5: a vision-language-action model with open-world generalization,

2025. URL https://arxiv. org/abs/2504.16054, 1(2):3.

- [22] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Searchr1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [23] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [24] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231, 2024.
- [25] Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, and Ivan Oseledets. Confidence is all you need: Few-shot rl fine-tuning of language models. arXiv preprint arXiv:2506.06395, 2025.
- [26] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl. arXiv preprint arXiv:2503.23383, 2025.
- [27] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding visionlanguage model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025.

- [28] Jiaqi Liu, Songning Lai, Pengze Li, Di Yu, Wenjie Zhou, Yiyang Zhou, Peng Xia, Zijun Wang, Xi Chen, Shixiang Tang, et al. Mimicking the physicist’s eye: A vlm-centric approach for physics formula discovery. arXiv preprint arXiv:2508.17380, 2025.
- [29] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. In European conference on computer vision, pages 126–142. Springer, 2024.
- [30] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning, 2021.
- [31] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The 3rd Workshop on Mathematical Reasoning and AI at NeurIPS’23, 2023.
- [32] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. pages 2263–2279, 2022.
- [33] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025.
- [34] Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, and Zhe Gan. Deepmmsearch-r1: Empowering multimodal llms in multimodal web search. arXiv preprint arXiv:2510.12801, 2025.
- [35] OpenAI. OpenAI o1 System Card — openai.com. https: //cdn.openai.com/o1-system-card.pdf, 2024. Accessed: 2024-09-12.
- [36] OpenAI. Introducing gpt-5. https://openai.com/ index/introducing-gpt-5/, 2025.
- [37] OpenAI. Openai deep research system card, 2025.
- [38] Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. Language model self-improvement by reinforcement learning contemplation. In International Conference on Learning Representations, pages 1–11, 2024.
- [39] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.
- [40] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning?, 2024.
- [41] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.

- [42] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.
- [43] Sheikh Shafayat, Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, and Andrea Zanette. Can large reasoning models self-train? arXiv preprint arXiv:2505.21444, 2025.
- [44] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.
- [45] Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6613–6629, 2025.
- [46] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025.
- [47] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025.
- [48] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.
- [49] Qwen-VL Team. Qwen3-vl: Sharper vision, deeper thought, broader action. https://qwen.ai/blog?id= 99f0335c4ad9ff6153e517418d48535ab6d8afef& from=research.latest- advancements- list,

2025. 2025-09-22.

- [50] Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.
- [51] Omkar Thawakar, Shravan Venkatraman, Ritesh Thawkar, Abdelrahman M Shaker, Hisham Cholakkal1 Rao Muhammad Anwer, Salman Khan, and Fahad Khan. Evolmm: Selfevolving large multimodal models with continuous rewards. arXiv preprint arXiv:2511.16672, 2025.
- [52] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. arXiv preprint arXiv:2310.03731, 2023.
- [53] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. pages 95095–95169, 2024.
- [54] Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yip-

- ing Lu, Zhengyuan Yang, et al. Vagen: Reinforcing world model reasoning for multi-turn vlm agents. arXiv preprint arXiv:2510.16907, 2025.
- [55] Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian Lin, Yiran Chen, Hai Helen Li, Kun Wan, and Wentian Zhao. Vision-zero: Scalable vlm self-improvement via strategic gamified self-play. arXiv preprint arXiv:2509.25541, 2025.
- [56] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [57] Xiyao Wang, Chunyuan Li, Jianwei Yang, Kai Zhang, Bo Liu, Tianyi Xiong, and Furong Huang. Llava-critic-r1: Your critic model is secretly a strong policy model. arXiv preprint arXiv:2509.00676, 2025.
- [58] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement, 2025.
- [59] Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, and Lichao Sun. Unsupervised post-training for multi-modal LLM reasoning via GRPO. arXiv preprint arXiv:2505.22453, 2025.
- [60] Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei Liu. Mmsearch-r1: Incentivizing lmms to search. arXiv preprint arXiv:2506.20670, 2025.
- [61] Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, Xiaoshuai Sun, and Rongrong Ji. Grounded chain-of-thought for multimodal large language models, 2025.
- [62] Peng Xia, Kaide Zeng, Jiaqi Liu, Can Qin, Fang Wu, Yiyang Zhou, Caiming Xiong, and Huaxiu Yao. Agent0: Unleashing self-evolving agents from zero data via tool-integrated reasoning. arXiv preprint arXiv:2511.16043, 2025.
- [63] Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, and Tong Zhang. Self-rewarding correction for mathematical reasoning. arXiv preprint arXiv:2502.19613, 2025.
- [64] Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, and Hongkun Yu. Incentivizing agentic reasoning in llm judges via tool-integrated reinforcement learning. arXiv preprint arXiv:2510.23038, 2025.
- [65] Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning. arXiv preprint arXiv:2509.02479, 2025.
- [66] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization, 2025.
- [67] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.
- [68] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng

- Feng, Li Shen, et al. Mulberry: Empowering mllm with o1like reasoning and reflection via collective monte carlo tree search, 2024.
- [69] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.
- [70] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [71] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization, 2025.
- [72] Juntian Zhang, Song Jin, Chuanqi Cheng, Yuhan Liu, Yankai Lin, Xun Zhang, Yufei Zhang, Fei Jiang, Guojun Yin, Wei Lin, et al. Viper: Empowering the self-evolution of visual perception abilities in vision-language model. arXiv preprint arXiv:2510.24285, 2025.
- [73] Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025.
- [74] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? pages 169–186. Springer, 2024.
- [75] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llavahd: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487, 2024.
- [76] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025.
- [77] Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang Wu, et al. Mm-rlhf: The next step forward in multimodal llm alignment. arXiv preprint arXiv:2502.10391, 2025.
- [78] Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.
- [79] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.
- [80] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024.
- [81] Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, and Huaxiu Yao. Calibrated self-rewarding vision lan-

- guage models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [82] Yiyang Zhou, Yangfan He, Yaofeng Su, Siwei Han, Joel Jang, Gedas Bertasius, Mohit Bansal, and Huaxiu Yao. Reagent-v: A reward-driven multi-agent framework for video understanding. arXiv preprint arXiv:2506.01300, 2025.
- [83] Yiyang Zhou, Zhaoyang Wang, Tianle Wang, Shangyu Xing, Peng Xia, Bo Li, Kaiyuan Zheng, Zijian Zhang, Zhaorun Chen, Wenhao Zheng, et al. Anyprefer: An automatic framework for preference data synthesis. In International Conference on Learning Representations (ICLR), 2025.
- [84] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025.
- [85] Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. TTRL: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

## Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning

### Supplementary Material

- A. Notation
- B. Implementation Details

Base Model. Agent0-VL is implemented upon Qwen2.5VL-7B-Instruct and Qwen3-VL-8B. Both the Solver and Verifier share parameters θ.

Training Details. Our training pipeline consists of two sequential stages: supervised fine-tuning (SFT) and RL. In the SFT stage, Agent0-VL is first trained on tool-usage and image manipulation data, followed by gradual annealing on mathematical code reasoning data. The learning rate is set to 1 × 10−5, while other hyperparameters remain consistent. We adopt a batch size of 128, train for 3 epochs, and apply a linear warmup ratio of 0.05. To ensure training stability and prevent early-stage collapse or entropy degeneration in the RL process, we perform a short external-reward pretraining phase before self-evolution begins. Specifically, the model is trained for 2 epochs using conventional RL with externally defined correctness signals to initialize stable exploration and tool-usage behaviors. This pretraining serves as a warm-up that activates structured exploration, after which the model transitions to the self-evolving RL phase driven purely by internal process rewards. In the self-evolving RL stage, the learning rate is set to 5 × 10−7, and training is performed for 1 epoch with a batch size of 256. The reinforcement learning follows the GRPO [44] paradigm with a group size N = 8 for relative normalization. We set the KL divergence coefficient βKL = 0.001 to regularize policy drift, collect 4 rollouts per task, and apply a repetition penalty of 1.05 to discourage redundant reasoning. The entropy coefficient is βent = 0.01, the selective repair threshold τc = 0.7, and the repair penalty η = 0.05. All experiments are conducted with mixed-precision training (bfloat16) on 8 NVIDIA H200 GPUs.

- C. Prompting and Templates

We provide system prompts for our framework, as shown in Figure 5-7, respectively.

- D. Training Data Construction Pipeline

This section describes the multi-stage data construction pipeline used to train Agent0-VL. The objective is to build a large-scale corpus of high-quality, tool-integrated reasoning trajectories that equip the model with both reasoning and verification capabilities before entering the self-evolving reinforcement stage.

System Prompt(Solver)

You are the Reasoner in a unified tool-integrated VLM agent. You must solve tasks through multi-turn reasoning and selective tool use.

Rules:

- - Wrap internal reasoning in <think>...</think>.
- - Call tools only when necessary using: {"tool_name":"<Tool>","tool_input":{...}}
- - Wait for tool_output before continuing.
- - Be concise and deterministic; no hallucinated values.

Process:

<think>

- 1. Restate goal & plan next step.
- 2. Decide if a tool is needed; specify input.
- 3. Integrate tool_output; update reasoning.

</think> Emit one tool call or continue reasoning. When finished:

- (1) Output CONFIDENCE: <0-1>
- (2)Output FINAL_ANSWER: <answer>.

Figure 5. System prompt for the Solver.

###### D.1. Overview

To ensure diversity and robustness, the training data are constructed through a progressive curriculum, covering a wide range of multimodal reasoning tasks. The resulting corpus is divided into three functional tiers:

- • Direct reasoning data: single-turn textual or visual questions solvable without tool calls, grounding linguistic reasoning and visual perception.
- • Tool-augmented data: problems that require code execution, OCR, or visual analysis, providing factual grounding and verifiable evidence.
- • Multi-turn reasoning data: complex visual–mathematical tasks involving iterative tool

Symbol Meaning x Task instance (image + text) st,at,ot State/context, action (reasoning/tool call), observation (tool output) at step t τ Trajectory {(st,at,ot)}Tt=1 πθ Unified policy with role tokens for RA/EE rproc(t) Process-level reward at step t g(τ) Composite trajectory reward (Eq. 5) τc Confidence threshold for triggering local repair

Table 5. Key notations used throughout the paper.

System Prompt

You are the Verifier in a unified tool-integrated VLM agent. Your role is to verify a given reasoning trajectory step-by-step, optionally calling tools to check facts.

Inputs: trajectory prefix $\tau_{1:t} =

{(s_k, a_k, o_k)}_{k=1..t}$. Rules:

- (1)Wrap internal thought in <think>...</think>.
- (2)Use the same tool schema for factual checks.
- (3) Output exactly one JSON line per step:

{"step_index":t, "score":<-1-1>, "confidence":<0-1>, "critique":"<<=2 sentences>", "tool_check":true|false }

Principles:

- (1) Ground verification on objective tool evidence.
- (2) Penalize unsupported or inconsistent reasoning.
- (3) High confidence requires agreement between tool and text.

Figure 6. System prompt for the Verifier.

use, correction, and reflection.

All prompts are derived or adapted from existing multimodal benchmarks, including Geometry3K [30], GeoQA [5], Mulberry [68], LLaVA-OV-Image [23], MM-RLHF [77], SMR [75], MM-Eureka [33], Retool [11], and arXivQA [24].

For mathematical and chart-based reasoning, we further include MathVerse [74], MathVista [31], WeMath [40], and ChartQA [32].

###### D.2. Multi-Stage SFT Data Construction

The SFT stage aims to teach Agent0-VL to perform endto-end, tool-integrated reasoning and self-verification under minimal human supervision. We employ a three-step automatic pipeline inspired by recent multimodal reasoning works [33, 77].

- (1) Task Sampling and Prompt Construction. We randomly sample multimodal problems (Q,I) from the datasets above. For each problem, a structured prompt is applied that requests: (i) a detailed reasoning trace (wrapped in <think>...</think>) and (ii) explicit tool-use plans in JSON format. Teacher models (GPT-5 [36] and Qwen2.5VL-72B [3]) are used to bootstrap the generation of complete trajectories.
- (2) Tool Execution and Verification. All tool calls are executed in a sandbox environment to obtain real results. For each reasoning step at, the corresponding observation ot is logged, forming complete multimodal trajectories τ =

{(at,ot)}Tt=1. An auxiliary verifier model then checks the consistency between textual reasoning, tool outputs, and the final answer. Inconsistent or unexecutable trajectories are automatically filtered.

- (3) Tool-Grounded Self-Verification and Repair Examples. To initialize the model’s self-evaluation ability, we further include examples where the verifier generates struc-

tured feedback Vt = (scoret,conft,critiquet) and correction signals for low-confidence steps. These “reasoning–verification–repair” triples serve as bridge samples for transitioning from supervised learning to self-evolving reinforcement learning.

After filtering and deduplication, we obtain approximately 200k high-quality multimodal trajectories. This unified SFT dataset initializes the dual-role reasoning and verification behaviors of Agent0-VL, laying the foundation for self-evolving optimization.

System Prompt

You are the Self-Repair module of a unified tool-integrated VLM.

You receive: (i) the original trajectory prefix \tau_{1:t}, (ii) the EE verification triple for step t (score_t, confidence_t, critique_t), and (iii) the minimal repair target

(segment uˆ(t)). GOAL

- - If confidence_t < \tau_c, propose a minimal, local patch to uˆ(t) that fixes the specific error WITHOUT rewriting validated context.
- - Use tools to recompute only what is necessary to validate the patch.

REASONING FORMAT

- - Put your full planning and diagnostics inside <think>...</think>.
- - After <think>, either (A) emit a PATCH JSON (and optionally a tool call), or (B) emit a NO_CHANGE JSON if repair is not warranted.

PATCH JSON (single line): {

"action": "PATCH", "target_step": t, "patch_type":

"<text|code|tool_call|parameter>", "new_content": "<the minimal

replacement content>",

"justification": "<<= 2 sentences referencing critique/evidence>"

} NO_CHANGE JSON (single line): {

"action": "NO_CHANGE", "target_step": t,

"reason": "<why repair is not warranted

or evidence is insufficient>" }

Figure 7. System prompt for Self-Repair Mode.

###### D.3. Dataset For Reinforcement Learning

We start from math and perception problems from MathVerse [74], MathVista [31], WeMath [40], arXivQA [24], and ChartQA [32], and ThinkLiteVL [58]. The model rolls out reasoning trajectories under its current policy, while the internal Verifier generates step-level rewards and critiques to guide optimization.

###### D.4. Quality Control

To ensure factual reliability and semantic alignment, we apply:

- • Execution validation: all tool invocations are re-run in a sandbox to remove invalid traces.
- • Semantic consistency check: textual reasoning must align with tool results and visual observations.
- • Redundancy filtering: near-duplicate reasoning traces are pruned via embedding similarity.
- • Manual spot-check: about 10k samples are manually reviewed for multimodal and reasoning correctness.

This systematic construction ensures that Agent0-VL learns not only to reason and act across modalities, but also to verify, repair, and self-evolve through grounded, tool-based evidence.

In total, Agent0-VL is trained on approximately 240k curated multimodal reasoning trajectories: ∼200k SFT trajectories for cold-start reasoning, and ∼40k data for RL.

##### E. Case Studies

To provide an intuitive understanding of how Agent0-VL performs reasoning, verification, and tool interaction in practice, we visualize several representative cases covering both single-step and multi-step reasoning scenarios. These examples illustrate how the model dynamically invokes external tools, integrates tool outputs into its internal reasoning process, and leverages the Verifier for process-level verification and correction. Together, these cases highlight the model’s ability to ground reasoning in visual evidence, maintain factual consistency, and perform self-reflective improvement across diverse visual reasoning tasks.

[Figure 47]

- Phase 1: Initial Reasoning

Failed Attempt

System Prompt for Solver

[Figure 48]

- {"step_index":1, "score":-1.0, "confidence":1.0, "critique":"Reasoning contains a critical logic error. The definition of 'blind spot' is wrong. The instruction 'from due East (0°) counterclockwise by 270°' is explicit and covers the angular range [0°, 270°], which is Q1, Q2, and Q3. The 'ending at due North' text contradicts the 270° rotation and should be treated as interference or a typo. Therefore, the true blind spot is Q4 ($x > 0, y < 0$).", "tool_check":false}
- {"step_index":2, "score":-1.0, "confidence":1.0, "critique":"The subsequent calculations (steps 4a, 4b), while correct based on the faulty premise (Q2), are invalid because the premise is wrong. The actual route segments A-B and B-C do transit the true blind spot, Q4.", "tool_check":false}

- Phase 2: Evaluation

As shown in the figure, a coast guard watchtower is located at the coordinate origin O(0, 0). Three key reef locations, A, B, and C, are marked on the nautical grid. A ship departs from point A, planning to sail in sequence to point B, and then to point C. Its route consists of two straight-line segments: A-B and B-C**. The watchtower‘s searchlight beam is limited. It can only rotate counter-clockwise from due East (positive x-axis) by **$270^\circ$**, ending at due North (positive y-axis). Calculate how much of the ship's total journey (segments A-B and B-C) is sailed within the watchtower's blind spot.

[Figure 49]

<think>

- 1. The goal is to calculate **
- 2. *The description "from due East (positive x-axis) counter-clockwise by $270^\circ$, ending at due North (positive y-axis)" is **self-contradictory**. * (a) A $270^\circ$ counterclockwise rotation from East ($0^\circ$) should end at South ($270^\circ$). * (b) Reaching North ($90^\circ$) from East ($0^\circ$) is only a $90^\circ$ rotation.

- • One possible interpretation is that the beam covers the $270^\circ$ **major arc** from East to North, i.e., $0^\circ \rightarrow -90^\circ(\text{or } 270^\circ) \rightarrow 180^\circ \rightarrow 90^\circ$.
- • * Under this interpretation, the covered areas are Q1, Q4, and Q3.
- • * Therefore, the **Blind Spot = Quadrant 2 ($x < 0, y > 0$)**.

- 3. I need to (a) Check if segment A-B enters Q2. * (b) Check if segment B-C enters Q2. * (c) Sum the results.
- 4. **Execute Plan:**

[Figure 50]

- • * (a) Segment A-B: From $A(8, 6)$ [Q1] to $B(7, -1)$ [Q4]. Both points have $x > 0$. Therefore, the entire segment is in the $x > 0$ region and **cannot enter Q2**. Length in blind spot is $0$.
- • * (b) Segment B-C: From $B(7, -1)$ [Q4] to $C(-4, -3)$ [Q3]. Both points have $y < 0$. Therefore, the entire segment is in the $y < 0$ region and **cannot enter Q2**. Length in blind spot is $0$.
- • * (c) Total length = $0 + 0 = 0$. </think> <answer>0.00 </answer>

[Figure 51]

###### Phase 3: Self-Repair Self-Repair System Prompt

**EE Key Feedback:** $\text{score}_1 = -1.0$, $\text{conf}_1 = 1.0$, $\text{critique}_1$ = "The true blind spot is Q4 ($x > 0, y < 0$)".

**Trigger Condition:** $\text{conf}_t \ge \tau_c$ and $\text{score}_t < 0$ (highconfidence error), activating repair gate $g_t$. <think> 1. **Diagnosis:** EE has assigned a score of $-1.0$ with $1.0$ confidence at `step_index:1`. 2. **Root Cause:** RA's definition of the "blind spot" was incorrect.

[Figure 52]

System Prompt for Verifier

3. **EE Correction:** The `critique` explicitly states the correct blind spot is

**Quadrant 4 (Q4)**. 4. **Repair Strategy:** I must issue a `PATCH` to correct the core logical premise in `step_index:1` of the reasoning. This patch will change the computational basis for all subsequent steps. </think> { "action": "PATCH", "target_step": 1, "patch_type": "text", "new_content": "Corrected Premise: Based on EE verification, the blind spot is identified as Quadrant 4 (Q4, $x > 0, y < 0$).", "justification": "Original logic incorrectly identified Q2 as the blind spot due to contradictory input text. EE has clarified the correct definition is Q4." }

[Figure 53]

Error Found

[Figure 54]

[Figure 55]

###### Phase 4: Corrected Reasoning

Solver System Prompt, with `PATCH` injected from SR

<think>

1. **Goal:** Restart reasoning based on the `PATCH`. 2. **Corrected Premise:** Blind Spot = Quadrant 4 (Q4, $x > 0, y < 0$). 3. **New Plan:** * (a) Calculate the length of segment A-B within Q4. * (b) Calculate the length of segment B-C within Q4. * (c) Sum the lengths and round. 4. **Execute (a) - A-B:** * Segment starts at $A(8, 6)$ [Q1] and ends at $B(7, -1)$ [Q4]. * It crosses the x-axis ($y=0$) into the blind spot Q4. * I need to find the intersection $P$ of the line A-B with $y=0$, and then calculate the distance between $P$ and $B$. * I will call a tool for this geometric calculation. </think>

[Figure 56]

{"tool_name":"calc_intersection_and_distance", "tool_input":{"p1":[8, 6], "p2":[7, -1], "axis_to_cross":"y=0", "target_quadrant":4}}

**[Tool Output]** `{"intersection_point":[7.142857, 0.0], "distance_in_quadrant":1.01015}`

<think>

5. **Integrate (a):** Tool returns length of A-B in Q4 as **$1.01015$**. 6. **Execute (b) - B-C:** * Segment starts at $B(7, -1)$ [Q4] and ends at $C(-4, -3)$ [Q3]. * It starts *in* the blind spot Q4 and leaves by crossing the y-axis ($x=0$). * I need to find the intersection $Q$ of the line B-C with $x=0$, and then calculate the distance between $B$ and $Q$. * I will call the tool. </think>

[Figure 57]

{"tool_name":"calc_intersection_and_distance", "tool_input":{"p1":[7, -1], "p2":[-4, -3], "axis_to_cross":"x=0", "target_quadrant":4}}

**[Tool Output]** `{"intersection_point":[0.0, -2.272727], "distance_in_quadrant":7.11601}` <think>

7. **Integrate (b):** Tool returns length of B-C in Q4 as **$7.11601$**. 8. **Execute (c) - Summation:** * Total Distance = (Length of A-B in Q4) + (Length of B-C in Q4) * Total = $1.01015 + 7.11601 = 8.12616$. * Rounding to two decimal places, the result is $8.13$. </think> <answer> 8.13 </answer>

Correct

- Figure 8. A full reasoning–evaluation–repair cycle of Agent0-VL on a geometric reasoning task. The Solver first generates an incorrect solution (Phase 1), which the Verifier identifies and critiques through tool-grounded verification (Phase 2). Based on this feedback, the model performs Self-Repair to patch the faulty premise (Phase 3) and re-executes reasoning with the corrected logic (Phase 4), producing the verified final answer.

[Figure 58]

[Figure 59]

###### Tool Calling

Question: What street is indicated by the sign on the pole near the center bottom of the image? Options: A. MACHIGAN B. MACHLGUN C. MICHIGUN D. MICHIGAN Please select the correct answer from the options above.

import os import random from PIL import Image # Define paths image_path = "hr_bench_4k/179.jpg" output_dir = "/mnt/data/temp_processed_images" # Ensure the output directory exists os.makedirs(output_dir, exist_ok=True) # Load the image image = Image.open(image_path) # Define cropping region (x1, y1, x2, y2) crop_coords = (1550, 2650, 1850, 2750) # Crop and zoom cropped = image.crop(crop_coords) zoom_factor = 2 zoomed = cropped.resize( (cropped.width * zoom_factor, cropped.height * zoom_factor)) # Generate output filename suffix = random.randint(1000, 9999) output_path = os.path.join(output_dir, f"179_{suffix}.jpg") # Save and print path zoomed.save(output_path) print(output_path)

Model Response

<Think>

The task is to identify the street name displayed on the sign located near the bottom center of the image. The sign is relatively small and distant, making the text difficult to read without magnification. To improve readability, the image should be cropped and zoomed in around this specific region, enhancing the visibility of the sign’s text for accurate recognition. The sign lies slightly to the right of the image’s central lower area, so cropping this portion will allow for focused inspection. The cropping coordinates can be estimated based on the sign’s relative position within the image.

Tool Calling

Tool execution

After examining the cropped and enlarged section, the sign clearly reads “N MICHIGAN AV 100 E”, confirming that the indicated street is Michigan Avenue. </think> <answer> D. MICHIGAN </answer>

[Figure 60]

Tool execution

<sandbox\_output> <subimage\_sign> </sandbox\_output>

- Figure 9. Single-step tool-integrated reasoning. The example demonstrates how Agent0-VL identifies a street name from an image by reasoning about the task, deciding to crop and zoom in on a specific region of the image, and invoking the image-cropping tool to enhance visibility. The resulting output confirms the street name, showing how the model effectively grounds its reasoning process in visual manipulation.

[Figure 61]

Question: Find the volume of the cone shown.

Round your answer to

**two decimal places**.

Model Response

<Think>

We get the result from python code. The result is 32.45. Therefore, when rounded to two decimal places:\[\boxed{V = 32.54 \, \text{cm}^3}\] </think>

<answer> 32.54 cm³ </answer>

Tool execution

Tool Calling

<tool> import math r = 2 l = 8 h = math.sqrt(l**2 - r**2) V = (1/3) * math.pi * (r**2) * h round(V, 2) </tool>

Tool Calling

[Figure 62]

We need the cone’s **vertical height (h)**, but the question provides the **slant height (l)**. From the Pythagorean theorem for the right triangle formed by height, radius, and slant height:\[h = \sqrt{l^2 - r^2}\] Once we find \(h\), we can compute the cone’s volume using:\[ V

= \frac{1}{3}\pi r^2 h \]

<sandbox\_output> 32.45 </sandbox\_output>

[Figure 63]

Tool execution

- Figure 10. Mathematical reasoning with code execution. This case illustrates how the model decomposes a geometry problem into structured reasoning steps, formulates the necessary equations using the Pythagorean theorem, and calls the Python computation tool to verify and compute the cone’s volume. The tool-grounded reasoning ensures both the numerical correctness and interpretability of the final solution.

Reward Model Evaluation

- Step 1: Identify the Goal The problem asks us to determine the number of common roots between two functions, $f(x) = x^2 + x - 6$ and $g(x)$, which is represented by a graph. To do this, we must find the roots of each function and then compare them.

Reasoning Model Solution

- Step 2: Find the Roots of $f(x)$ The roots of a function are the values of $x$ for which the function's value is zero. For $f(x)$, we set the equation to zero:$$x^2 + x - 6 = 0$$ We can solve this quadratic equation by factoring. We look for two numbers that multiply to -6 and add to +1. These numbers are +3 and -2.$$(x + 3)(x - 2) = 0$$ This gives us two distinct roots for $f(x)$:$$x_1 = -3 \quad \text{and} \quad x_2 = 2$$

- The set of roots for $f(x)$ is $\{-3, 2\}$.

Step 3: Find the Roots of $g(x)$ from the Graph The roots of a function shown on a graph are its x-intercepts, which are the points where the curve crosses the horizontal x-axis. By visually inspecting the provided graph of $y = g(x)$, we can identify these points. The parabola intersects the x-axis at:$$x_1 = -1 \quad \text{and} \quad x_2 = 2$$

- The set of roots for $g(x)$ is $\{-1, 2\}$.

- Step 4: Compare the Roots of $f(x)$ and $g(x)$ Now, we compare the sets of roots we found for both functions: Roots of $f(x)$: $\{-3, 2\}$; Roots of $g(x)$: $\{-1, 2\}$. The common root is the value that appears in both sets.$$\text{Common Root(s): } \{2\}$$ Each function also has a root that is not shared: $x = -3$ for $f(x)$ and $x = -1$ for $g(x)$.
- Step 5: Final Conclusion Based on the comparison, the functions $f(x)$ and $g(x)$ share exactly one root ($x=2$), and each function has another root that is unique to it. This corresponds to choice B. Final Answer: B

{✓}Clear problem statement: Correctly identified the goal {✓} Appropriate strategy: Finding roots separately then comparing {✓}Complete scope: Recognized need to handle both algebraic and graphical analysis {✓}Explicit methodology: Stated the approach clearly

[Figure 64]

Tool Calling

- Step 1
- Step 2
- Step 3
- Step 4
- Step 5

Correct

###### Question

<tool> import sympy as sp # Define the function x = sp.symbols('x') f = x**2 + x - 6 # Solve for the roots roots = sp.solve(f, x) print("Roots of f(x):", roots) <tool>

[Figure 65]

Correct

{✓}Correct equation setup: $x^2 + x - 6 = 0$ {✓}Valid factorization method: Identified factors that multiply to $6$ and sum to $+1$ {✓}Accurate factorization: $(x + 3)(x - 2) = 0$ {✓}Correct root extraction: $x = -3, 2$ {✓}Clear presentation: Well-organized algebraic steps

Incorrect {✓} Correct method: Identifying x-intercepts for roots {✓} Appropriate visual approach: Using graph inspection {❌} CRITICAL ERROR: Incorrect root identification {❌} No verification: Failed to check reading against graph properties {❌} Symmetry violation: Claimed roots violate parabola symmetry

[Figure 66]

[Figure 67]

[Figure 68]

{✓} Clear presentation: Well-organized comparison {✓} Correct logical structure: Proper set intersection approach {✓} Accurate computation: Given the inputs, correctly identified common root {❌} Propagated error: Used incorrect $g(x)$ roots from Step 3 {!} Fortuitous result: Error did not affect final answer

Correct

[Figure 69]

Correct {✓} Correct conclusion: One common root identified{✓} Accurate mapping: Correctly linked to choice B {✓} Complete reasoning: Acknowledged unique roots for each function {!} Lucky correctness: Final answer correct despite intermediate error

Answer

Correct

- Figure 11. Evaluator-based process verification. This example showcases the Verifier role of Agent0-VL, where the model critically inspects each reasoning step produced by the Solver in a function-root comparison task. The model performs step-level judgments (Correct/Incorrect), identifies propagation errors, and recognizes when the final conclusion remains valid despite intermediate mistakes. This demonstrates the model’s capacity for fine-grained self-evaluation and process-level reasoning analysis.

Question: Below is the graph of a trigonometric function. It has a minimum point at $\left(\frac{1}{2} \pi, 2.4\right)$ and a maximum point at $\left(\frac{3}{2} \pi, 13\right)$. What is the midline equation of the function? Give an exact expression.

Model Response

<Think>

Through systematic reasoning and computational verification, I've determined that the midline equation is: y = 7.7

Tool execution

Tool Calling

<tool> min_y = 2.4 max_y = 13 # Calculate midline midline = (min_y + max_y) / 2 print(f"Midline = ({min_y} + {max_y}) / 2 = {midline}")</tool>

Tool Calling

[Figure 70]

Looking at this problem, I need to find the midline equation of a trigonometric function. I'm given two critical points: a minimum at (-π/2, 2.4) and a maximum at (3π/2, 13). Let me think through this systematically and use code to verify my reasoning.

<sandbox\_output> 32.45 </sandbox\_output>

[Figure 71]

Tool execution

[Figure 72]

</think>

The midline of any trigonometric function is the horizontal line exactly halfway between its maximum and minimum y-values. This is because trigonometric functions oscillate symmetrically around their midline

- Step 1
- Step 2

I have: Minimum point: (-π/2, 2.4); Maximum point: (3π/2, 13) Goal: Find exact midline equation The x-coordinates don't matter for the midline calculation - only the y-values do.

For the midline: Midline = (Maximum y-value + Minimum y-value) / 2 Let me calculate this:

This gives me: Midline = (2.4 + 13) / 2 = 7.7

<answer> 7.7 </answer>

- Figure 12. Analytical reasoning with visual grounding. In this trigonometric midline problem, Agent0-VL interprets graphical input, reasons step-by-step through symbolic computation, and validates its reasoning using code execution. By combining perceptual understanding and analytical computation, the model achieves consistent reasoning grounded in both mathematical and visual evidence.

