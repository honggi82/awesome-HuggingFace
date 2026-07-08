# arXiv:2605.07237v2[cs.CL]11May2026

## Teaching Language Models to Think in Code

Hyeon Hwang Korea University Seoul, South Korea hyeon-hwang@korea.ac.kr

Jiwoo Lee Korea University Seoul, South Korea hijiwoo7@korea.ac.kr

#### Jaewoo Kang∗

Korea University, AIGEN Sciences Seoul, South Korea kangj@korea.ac.kr

### Abstract

Tool-integrated reasoning (TIR) has emerged as a dominant paradigm for mathematical problem solving in language models, combining natural language (NL) reasoning with code execution. However, this interleaved setup has three key limitations: code often acts as a post-hoc verifier, intermediate NL computations are error-prone, and NL and code play overlapping rather than clearly distinct roles. We propose THINC (Thinking in Code), a framework in which code itself serves as the reasoner rather than as a tool invoked by NL. A THINC trajectory begins with a brief NL planning step, after which all reasoning unfolds through code blocks connected only by their execution outputs. We distill 12.2k code-centric trajectories from a teacher model and train THINC-1.7B and THINC-4B with supervised fine-tuning followed by reinforcement learning. THINC-4B consistently outperforms every TIR baseline on five competition-level math benchmarks and even surpasses the much larger Qwen3-235B-A22B-Thinking. Further analysis shows that THINC reasons through code: 99.2% of its final answers are grounded in interpreter output, and the model recovers reliably from code execution failures without intermediate NL reasoning. Our code and models will be released soon.

### 1 Introduction

Recent advances in reinforcement learning (RL) over long chains of thought [19] have substantially enhanced the mathematical reasoning capabilities of Large Language Models (LLMs), leading to powerful natural-language (NL) reasoners such as OpenAI o1 [9] and DeepSeek-R1 [7]. Despite this progress, mathematical reasoning remains challenging for NL reasoners, particularly on problems requiring precise multi-step computation, where even a single arithmetic error can invalidate the entire reasoning process.

To make computation reliable, prior work has increasingly incorporated executable code into the reasoning process. Prompting-based approaches such as PAL [5] and PoT [2] generate Python programs that solve mathematical problems end-to-end, delegating precise computation to a code interpreter. These methods demonstrated the reliability of code for mathematical computation and symbolic expression, but remain limited to single-pass program generation without iterative interaction with execution results. To combine the complementary strengths of NL reasoning and code execution, subsequent work introduced tool-integrated reasoning (TIR) [6, 18], where NL handles high-level planning while code performs precise computation. TIR interleave NL reasoning with code

∗Corresponding author.

Preprint.

(A) Post-hoc Tool Verification

###### (B) Unreliable NL-based Computation

###### (C) Misallocated Reasoning Roles

How many two-digit numbers have

A factory produces 500 widgets/day. After a 12% increase, how many does it produce in 5 days?

The roots of x² − 7x + 10 = 0 are α and β. Find the value of α² + β².

digits that sum to 15?

User

User

User

[Figure 1]

[Figure 2]

[Figure 3]

I will loop through all two-digit numbers from 10 to 99. For each number, I extract the tens and ones digits. Then I check if their sum equals 15, …

By Vieta's formulas, α + β = 7 and αβ

Let’s calculate 12% of 500. 500 × 0.12 = 50.0, (Should be 60.0) New daily = 500 + 50.0 = 550.0

= 10. Since α² + β² = (α + β)² − 2αβ = 49 − 20 = 29. The answer is 29. Let me confirm with code.

LLM

LLM

LLM

Incorrect NL intermediate value copied into the tool

Tool used after computation is complete

Logically Identical

count = 0

from sympy import symbols, solve

new_daily = 550.0 total_days = 5 result = new_daily * total_days print(result) >>> 2,750.0

x = symbols('x')

for n in range(10, 100): if n // 10 + n % 10 == 15:

roots = solve(x**2 - 7*x + 10, x) assert sum(r**2 for r in roots)

Tool

Tool

Tool

count += 1 print(count)

[Figure 4]

== 29

|Tool invoked after the reasoning is done|
|---|

|Error propagates unchecked|
|---|

|Functional Role Separation Failure|
|---|

- Figure 1: Three structural limitations of interleaved tool-integrated reasoning. (A) Post-hoc tool verification: the model completes a derivation in NL and runs code only to confirm the answer, so the interpreter performs no new computation. (B) Unreliable NL-based computation: an NL arithmetic error propagates silently into the next code block as a hard-coded constant. (C) Misallocated Reasoning Roles: the NL reasoning describes the very algorithm that the subsequent code re-implements.

execution over multiple turns, enabling iterative refinement and intermediate verification through interpreter feedback. Recent work has further expanded this paradigm along several directions. ReTool [4] uses RL to optimize tool-use strategies, ASTER [23] emphasizes dense tool interaction throughout reasoning, and Tool-Star [3] extends TIR to collaborative reasoning across multiple tools.

53

However, as shown in Figure 1, TIR’s interleaved reasoning paradigm suffers from three recurring structural limitations. First, the model often completes a derivation in NL first and then runs code only to confirm it; code becomes a post-hoc verifier rather than a reasoner, contributing no new computation. Second, when the model carries out arithmetic or algebraic steps in NL, a wrong value can be copied into the next code block as a hard-coded constant. The interpreter cannot detect the mistake, and the error silently affects the final answer. Third, although NL reasoning excels at high-level planning and code can serve as a reasoner for precise mathematical expression and computation, interleaved TIR fails to separate these roles, leaving the two to do the same job. The NL reasoning lays out the algorithm step by step, taking on work that code is better suited for, while the code that follows merely transcribes the NL reasoning.

To address these limitations, we propose THINC (Thinking in Code), a training framework in which code itself serves as the reasoner rather than as a tool driven by NL reasoning. A THINC reasoning begins with a single brief planning step in NL that frames the strategy, after which all reasoning unfolds through code blocks connected only by their execution outputs. This structure resolves the three limitations by design: code performs derivations rather than verifying NL conclusions, every intermediate value is produced by the interpreter and therefore verified, and NL is restricted to high-level planning while code carries out all reasoning. We realize this paradigm in three stages: trajectory distillation from a teacher model via few-shot prompting to construct the 12.2k THINCSFT dataset, supervised fine-tuning (SFT) to establish the code-centric behavior prior, and RL with verifiable rewards [16] to strengthen problem-solving.

We evaluate THINC at two scales, THINC-1.7B and THINC-4B, built on Qwen3-1.7B and Qwen3-4BThinking-2507 [20] respectively, across five competition-level math benchmarks (AIME 2024–2026, HMMT 2025, and BeyondAIME [14]). THINC-1.7B reaches 42.8% average accuracy, exceeding Qwen3-1.7B by 10.6 percentage points. THINC-4B reaches 78.1%, surpassing every TIR baseline in our evaluation and exceeding Qwen3-235B-A22B-Thinking, a much larger NL reasoner, on four of the five benchmarks. Further analysis shows that THINC-4B reasons in a genuinely code-centric manner, with 99.2% of its final answers grounded in interpreter output rather than generated through NL reasoning. This behavior also makes THINC robust when initial code executions fail, while interleaved TIR baselines degrade sharply.

#### Our contributions are as follows.

- • We propose THINC, a training framework that teaches language models to treat code as the primary reasoner for mathematical problem solving rather than as a tool called by NL reasoning. THINC consists of trajectory distillation, SFT, and RL with verifiable rewards.
- • We present the THINC-SFT dataset of 12.2k code-centric trajectories, together with two trained models, THINC-1.7B and THINC-4B. THINC-4B reaches 78.1% average accuracy across five competition-level math benchmarks, outperforming both all TIR baselines and the much larger Qwen3-235B-A22B-Thinking.
- • We provide comprehensive analyses showing that THINC-4B reasons in a genuinely code-centric manner at inference time, and identify robustness to early code execution failures as a concrete consequence of this structure.

### 2 Preliminaries

#### 2.1 Tool-Integrated Reasoning

TIR augments a language model with one or more external tools that can be invoked during generation, such as code interpreters, search engines, or symbolic solvers. Solving a problem in TIR is a multiturn process: the model alternates between generating text and invoking tools, conditioning each subsequent action on the tool’s output. In this work, we focus on the mathematical reasoning setting, where the tool is a Python interpreter E and each turn consists of a natural-language thought block t ∈ T , a code block c ∈ C generated by the model, and an execution output o = E(c) produced deterministically by the interpreter and appended to the context as a non-trainable observation. Given a problem q, the standard interleaved TIR paradigm produces trajectories of the form

τTIR = (q, t1,c1,o1, t2,c2,o2, ..., tN,cN,oN, a), (1) where N is the number of turns and a is the final answer. All recent TIR systems [6, 4, 23] follow this structure.

#### 2.2 Supervised Fine-Tuning

Supervised fine-tuning (SFT) adapts a pre-trained LLM to a target behavior by training on demonstration trajectories with a next-token prediction objective. In TIR, demonstrations are typically distilled from a stronger teacher model, and the choice of trajectories directly shapes the tool-use patterns that the model learns to produce [22, 4, 23]. Given a dataset DSFT of trajectories, the SFT objective is

 

 , (2)

|τ|

LSFT(θ) = −Eτ∼D

mk log πθ(xk | x<k)

SFT

k=1

where xk is the k-th token of τ and mk ∈ {0,1} is a per-token loss mask. Prior TIR work commonly sets mk = 0 for tool execution output tokens, restricting supervision to model-generated tokens only. We find no significant performance difference between the two choices and therefore use mk = 1 for all tokens in this work.

#### 2.3 Reinforcement Learning with Verifiable Rewards

For RL training in TIR, verifiable rewards are commonly used: each problem has a known groundtruth answer a⋆(q), and a trajectory receives reward

r(τ) = 1[a(τ) = a⋆(q)], (3) where a(τ) is the answer extracted from τ. Exact-match verification removes the need for a learned reward model.

Group Relative Policy Optimization (GRPO) [16] is a critic-free policy gradient algorithm widely used in this setting. For each problem q, GRPO samples a group of G trajectories {τ(g)}Gg=1 from the current policy πθ, and computes a group-relative advantage from their rewards:

G

G

r(τ(g)) − µq σq

2

1 G

1 G

A(g) =

r(τ(g)), σq2 =

r(τ(g)) − µq

. (4)

, µq =

g=1

g=1

Following DAPO [21], we adopt two modifications to the standard GRPO objective: token-level normalization across the entire group rather than per-trajectory averaging, and asymmetric clipping with ϵlow < ϵhigh that allows larger positive policy updates. The resulting clipped surrogate objective is

  1

 , (5)

|τ(g)|

G

min ρ(kg)A(g), clip ρ(kg),1−ϵlow,1+ϵhigh A(g)

J (θ) = Eq,{τ(g)}

g |τ(g)|

g=1

k=1

where ρ(kg) = πθ(τk(g) | q,τ<k(g))/πθ

(τk(g) | q,τ<k(g)) is the per-token importance ratio.

old

### 3 THINC: Teaching Models to Think in Code

We present THINC, a training framework that teaches language models to treat code itself as the reasoner for mathematical problem solving rather than as a tool invoked by natural language. THINC consists of three components: (1) a code-centric trajectory format in which code itself serves as the reasoner (Section 3.1); (2) a distillation and supervised fine-tuning procedure that induces this format in a student model (Section 3.2); and (3) a multi-stage reinforcement learning procedure that further refines the resulting policy (Section 3.3).

#### 3.1 THINC Reasoning

In interleaved TIR (Eq. 1), an NL reasoner carries out the derivation and calls code as a tool. THINC treats code itself as the reasoner. Code is a natural fit for this role because programming languages, like mathematics, are symbolic systems. Variables, operations, and functions in a program correspond directly to mathematical objects, allowing each reasoning step to be expressed and executed precisely.

A THINC trajectory takes the form

τThinC = (q, t1, c1,o1, c2,o2, ..., cN,oN, a), (6)

where t1 is constrained to express strategy, a high-level plan for solving the problem, rather than any step-by-step derivation of the answer. Unlike prior multi-turn TIR frameworks, which interleave

thought and code at each step, our code-centric formulation uses a single initial thought t1 to specify the overall solution strategy, and all subsequent reasoning is carried out through code. Each code

block ci builds on the execution outputs of the preceding blocks, o1,...,oi−1, and the final answer a is obtained from the final execution output oN.

This simple structural change, illustrated in Figure 2, resolves the three limitations of interleaved TIR identified in Section 1 by construction:

- • Tool as a reasoner. No thought block precedes ci for i ≥ 2, so each code block directly performs a derivation step rather than acting as a post-hoc verifier, making the interpreter an integral part of the reasoning process.
- • Verified intermediates. All intermediate values are produced through the interpreter E, ensuring they are verified by construction and eliminating unverified numerical computation in NL.
- • Specialized roles. NL is restricted to high-level planning in t1, while code carries out all subsequent reasoning, restoring the role separation that interleaved TIR fails to maintain.

#### 3.2 Supervised Fine-tuning: Establishing Code-Centric Behavior

To train models to reason through code, we distill THINC trajectories from a strong teacher model and use them as supervised fine-tuning data.

Following prior work [23], we draw problems from Skywork-OR1 [8] and OpenMathReasoning [11], restricted to English-language problems with positive integer answers. We sample one trajectory per problem from Qwen3.5-27B using a 3-shot prompt that demonstrates the structure of Eq. 6 (full prompt in Appendix B). We retain a distilled trajectory only if it (i) is correct, (ii) executes every code block without interpreter error, (iii) contains at least three code blocks, and (iv) spends less than 50% of its tokens in the planning thought (|t1|/|τ| < 0.5). Criteria (iii) and (iv) together enforce

Question: Find the total area covered by the union of two regions: R1 = {(x,y) : |x| + 2|y| ≤ 6} and R2 = {(x,y) : 2|x| + |y| ≤ 6}.

(a) Interleaved TIR (b) ThinC (Ours)

- Turn 1
- Turn 2
- Turn 3

- Turn 1
- Turn 2
- Turn 3

|Think: To find … I will write a code …., d_x = 12 and d_y = 6. Then, I will multiply these … and divide the result by 2 to compute … . Finally, … called area_base and print the result.|
|---|

Think: Both … rhombuses symmetric ... compute the area of $R_1$ and $R_2$ using …union area: $Area(R_1) + Area(R_2) Area(R_1 \cap R_2)$. Code: area = lambda p: abs(sum(…))/2

Think Used to

No Clear Division of Roles

=

VS

###### Code:

Plan Only

|d_x, d_y = 12, 6 area_base = (d_x * d_y) / 2 print(area_base) >>> 36.0|
|---|

- r1 = [(6,0), …, …, (0,-3)] print(area(r1)) >>> 36.0

|from sympy import symbols, solve x, y = symbols('x y') sol = solve([…], [x, y]) q1_pts = […, (sol[x], sol[y]), (0,3)] print(float(4 * area(q1_pts))) >>> 24.0|
|---|

Code:

- r2 = [(3, 0),… , (0, -6)] ar2 = area(r2) final_area = ar1 + ar2 – 24.0

Tool is the Primary Solver

Think: Next, solving $x + … Q1 is two triangles with total area 6. … 4 quadrants = $4 \times 6 = 24$. I will now write a quick code to verify if my calculated value of 24 is correct. Code: manual = 24 code_inter = 4 * ((2 * 6) / 2 assert manual == code_inter

Code:

VS

Post-Compute Verifier

Think: Finally, union area = $A(R_1) + A(R_2) - … is 36. By geometric symmetry, $A(R_2)$ is also 36. First, I will calculate the sum of the two base areas: 36 + 36 = 62. Code:

Deterministic Execution Prevents Errors

VS

Incorrect NL Input Passed to Tool

sum_base = 62.0

print(f"Final Answer:{final_area}") >>> Final Answer:48.0

inter = 24.0 final_area = sum_base – inter print(f“Answer:{final_area}") >>> Answer:38.0

[Figure 5]

[Figure 6]

- Figure 2: Comparison of interleaved TIR (left) and THINC (right). The three rows pair the structural limitations from Section 1 with how THINC avoids each. (Top) Interleaved TIR has NL describe the algorithm step by step before code re-implements that description, with NL and code doing the same job; THINC restricts NL to high-level planning and lets code perform the derivation. (Middle) Interleaved TIR runs code as a post-hoc verifier of an NL-derived answer; in THINC, code is the primary solver. (Bottom) An NL arithmetic error propagates into the next code block as a hard-coded constant, leading interleaved TIR to the wrong final answer; THINC computes every intermediate value through the interpreter and reaches the correct answer.

54

the code-centric structure of THINC reasoning. Filtering yields the THINC-SFT dataset of 12,200 trajectories.

We fine-tune two base models, Qwen3-1.7B and Qwen3-4B-Thinking-2507, on THINC-SFT using the SFT objective in Eq. 2, with a context length of 32K, learning rate 7 × 10−6 with cosine schedule, global batch size 16, and 3 epochs. We refer to the resulting checkpoints as THINC-1.7B-SFT and THINC-4B-SFT.

#### 3.3 Reinforcement Learning

Starting from the SFT checkpoints, we further refine the policy using GRPO [16] on DAPO-Math17k [21]. Following DAPO [21], we optimize the token-level policy gradient objective with ClipHigher (ϵlow = 0.20, ϵhigh = 0.28) and no KL divergence penalty, with a rollout prompt batch size of 128 and G = 8 trajectories per prompt.

We train in three stages with increasing context budget; one epoch over the prompt set corresponds to roughly 140 optimization steps. Stage 1 runs for 280 steps (two epochs) on the full prompt set with a context length of 16K tokens and up to 20 tool calls per trajectory. Stage 2 continues with the same context and tool budget but filters out problems whose Stage 1 policy already solves with 100% pass rate, since these contribute zero group-relative advantage (Eq. 4); it runs for 120 steps, ending at step 400. Stage 3 begins at step 400 with the same difficulty filtering, expanding the context to

Table 1: Comparison on AIME 2024, AIME 2025, AIME 2026, HMMT 2025 February, BeyondAIME, and their arithmetic mean. Scores are reported as avg@16 (mean per-problem accuracy averaged across problems) under a 32K-token inference budget. Smaller numbers indicate 95% confidence intervals. Asterisk (∗) denotes models prompted to use the Python interpreter without further training. Qwen3.5-27B is the teacher model used for distillation, evaluated with the same 3-shot prompt used during THINC-SFT dataset generation. Bold: best; underline: second-best.

Model AIME24 AIME25 AIME26 HMMT25 BeyondAIME Average NL-Only Reasoning Models

Qwen3-1.7B 47.3±13.3 35.0±14.2 36.2±14.0 22.5±12.6 19.8±6.0 32.2±5.5 Qwen3-4B-Thinking-2507 79.2±11.9 73.1±13.5 76.7±12.9 50.2±15.6 45.8±8.6 65.0±5.7 OpenReasoning-Nemotron-7B 76.9±11.8 73.1±13.0 72.1±13.3 52.7±14.4 38.6±7.7 62.7±5.5 gpt-oss-20B (medium) 79.4±11.5 69.8±12.0 76.9±10.7 51.0±13.9 46.2±7.6 64.7±5.1 Qwen3-235B-A22B-Thinking 90.6±9.8 80.6±12.3 82.1±11.5 68.8±14.2 54.1±8.6 75.2±5.1

Tool-Integrated Reasoning Models

Qwen3-1.7B* 42.1±9.1 36.2±15.1 32.1±13.3 19.6±12.6 18.9±6.0 29.8±5.2 Qwen3-4B-Thinking-2507* 77.3±8.8 69.6±14.5 72.3±13.6 46.0±15.5 47.8±8.8 62.6±5.6 CoRT-1.5B 35.8±12.0 30.0±13.8 25.8±11.2 23.1±11.6 13.8±4.6 25.7±5.0 DemyAgent-4B 71.5±13.8 71.5±13.9 65.0±13.2 41.5±14.9 37.1±7.7 57.3±5.8 ASTER-4B 78.8±11.5 84.6±10.2 78.8±11.0 73.1±11.8 54.0±8.1 73.8±4.7 rStar2-Agent-14B 79.0±12.5 68.5±14.0 72.3±13.7 50.4±14.7 37.4±7.9 61.5±5.7 ReTool-32B 65.8±14.1 56.0±14.6 52.7±14.6 47.1±14.1 31.8±7.6 50.7±5.9

THINC Reasoning Models Teacher Model

Qwen3.5-27B* 74.8±10.5 64.0±11.0 75.6±10.2 63.3±12.9 45.7±7.5 64.7±4.7 Student Models

THINC-1.7B 59.0±9.2 50.2±14.9 42.9±13.7 39.0±14.5 22.7±6.6 42.8±5.5 THINC-4B 88.3±8.8 85.8±8.8 86.0±9.8 74.0±12.7 56.1±8.1 78.1±4.4

32K and the tool budget to 40 to allow longer trajectories on harder problems. We refer to the final checkpoints as THINC-1.7B and THINC-4B.

### 4 Experiments

#### 4.1 Experimental Setup

Benchmarks. We evaluate on five competition-level mathematical reasoning benchmarks: AIME 2024, AIME 2025, AIME 2026, HMMT 2025 February, and BeyondAIME [14].

Baselines. We compare THINC to two groups of baselines. The first is NL-only reasoners: Qwen31.7B and Qwen3-4B-Thinking-2507 [20] (our base models), OpenReasoning-Nemotron-7B [1], gpt-oss-20B [12], and Qwen3-235B-A22B-Thinking [20]. The second is tool-integrated reasoners: CoRT-1.5B [10], DemyAgent-4B [22], ASTER-4B [23], rStar2-Agent-14B [15], and ReTool-32B [4]. We additionally evaluate Qwen3-1.7B and Qwen3-4B-Thinking-2507, our base models prompted to use the Python interpreter without additional training, to separate the effect of THINC training from those of the underlying base model and the tool-use prompt. We also report results for Qwen3.5-27B with our 3-shot demonstration, as this model is used as the teacher for trajectory distillation.

Evaluation Protocol. For each benchmark, we sample 16 trajectories per problem under a 32Ktoken inference budget and report the average accuracy (avg@16). We sample THINC with temperature 0.6 and top-p 1.0, and follow the sampling parameters recommended by each baseline’s original publication or official release. All models are evaluated with a Python interpreter providing access to the standard library (including itertools and collections) and the scientific computing libraries numpy, scipy, and sympy. All baselines are run in the same environment.

#### 4.2 Main Results

THINC delivers consistent gains at both scales. As shown in Table 1, THINC-4B achieves the strongest overall result, with an average score of 78.1% and the best performance on four of the five benchmarks. It outperforms all tool-integrated reasoning baselines, including substantially larger systems such as rStar2-Agent-14B and ReTool-32B. In addition, it surpasses Qwen3-235BA22B-Thinking, the strongest open-source NL-only reasoner in our comparison, by 2.9 points on average. The advantage is particularly large on the more challenging benchmarks, HMMT25 and BeyondAIME. Remarkably, THINC-4B also exceeds its distillation teacher, Qwen3.5-27B with our 3-shot demonstration, on all five benchmarks by 13.4 points on average, despite being much smaller. The same pattern holds at the smaller scale: THINC-1.7B reaches 42.8%, outperforming both Qwen3-1.7B (32.2%), Qwen3-1.7B∗ (29.8%), and CoRT-1.5B (25.7%). Together, these results indicate that THINC training yields consistent gains across scales beyond those obtained from tool-use prompting alone.

THINC reasoning outperforms interleaved TIR. To isolate the effect of the reasoning format, we treat ASTER-4B as a natural ablation baseline for the interleaved approach. The two systems share a base model(Qwen3-4B-thinkning-2507), teacher capacity, and RL pipeline, differing primarily in trajectory structure. Under these matched conditions, THINC-4B exceeds ASTER-4B on every benchmark by an average of 4.1 points. The gain comes with lower inference cost, as THINC-4B requires fewer tool calls per trajectory (6.1 vs. 11.1) and produces shorter responses (13.5k vs. 15.4k tokens; see Appendix C). Code-centric reasoning therefore delivers higher accuracy over interleaved TIR, while naturally reducing inference overhead.

#### 4.3 SFT Cold-Start and RL Training Dynamics

SFT establishes the format; RL drives the gains. After SFT, THINC-4B-SFT reaches 48.1% on average, below both the teacher Qwen3.5-27B (64.7%) and the tool-prompted base model (62.9%); THINC-1.7B reaches 18.1%, also below its base (30.2%). This drop is by design: SFT teaches the model to reason in the THINC format, not to maximize accuracy. RL produces the benchmark gains, adding 29.9 points at 4B and 24.6 points at 1.7B (Figure 3a) and lifting both policies well above their bases and teachers.

RL improves the policy steadily throughout training. Figures 3b,c plot validation accuracy and response length on AIME 2024 over RL steps. Both scales show smooth, near-monotonic accuracy climbs with no plateau or collapse, and the three-stage curriculum (Section 3.3) is visible as a mild inflection at each stage boundary. Notably, THINC-4B’s response length stays in the 7K–11K range throughout RL, even when Stage 3 expands the context budget to 32K. AIME 2024 accuracy meanwhile climbs from 63.5% at the SFT checkpoint to 88.3% at the end of RL. The 1.7B model relies more on the extra context, with response length roughly doubling in Stage 3.

#### 4.4 Does THINC Actually Think in Code?

We next verify that the trained model exhibits THINC reasoning at inference time, as defined in Section 3.1, rather than simply imitating the format of the training trajectories. Figure 4 compares THINC-4B with five TIR baselines on AIME 2024–2026, HMMT 2025, and BeyondAIME using two complementary metrics. We consider both metrics together because they capture different aspects of code-centric reasoning. One measures how extensively code is used throughout the reasoning trajectory, while the other measures whether the final answer is grounded in interpreter outputs. Taken together, they provide a more complete view of whether a model not only writes code during reasoning, but also relies on execution outputs to produce its final answer.

THINC shifts the reasoning process to code. THINC writes an average of 349 lines of code per sample (Figure 4a), substantially more than ASTER (102), CoRT (40), and ReTool (261). Notably, ASTER and CoRT are the two TIR systems most explicitly designed to strengthen tool use, while ReTool is the strongest baseline on this metric. These results show that THINC makes substantially heavier use of code throughout the reasoning trajectory. We next examine whether this code also serves as the primary driver of reasoning, rather than merely accompanying an NL derivation.

###### (a) SFT vs RL Benchmark avg@16

120

ThinC-1.7B SFT ThinC-1.7B RL

ThinC-4B SFT ThinC-4B RL

| |
|---|

| |
|---|

| |
|---|

100

88.3

85.8 86.0

80

74.0

avg@16(%)

63.5

58.8

60

56.1

51.9

50.2

49.4

45.8

42.9

39.0

40

30.1

24.2 23.3

22.7

20

17.5 17.3

8.2

0

AIME '24

AIME '25

AIME '26

HMMT '25

Beyond AIME

###### (b) AIME 2024 avg@16

###### (c)

###### Response Length (tokens)

90

Model

Model

18000

ThinC-4B

ThinC-4B

80

ThinC-1.7B

ThinC-1.7B

16000

70

avg@16(%)

14000

60

Tokens

12000

50

40

10000

Stage

Stage

30

- Stage 1

- Stage 2

- Stage 3

- Stage 1

- Stage 2

- Stage 3

8000

20

0 100 200 300 400 500

0 100 200 300 400 500

Training Step

Training Step

- Figure 3: Training dynamics. (a) Benchmark avg@16 after SFT (light) and after RL (dark) for THINC-1.7B and THINC-4B across five math benchmarks. (b) AIME 2024 avg@16 over RL training steps. (c) Average response length over RL training steps. Shaded regions in (b) and (c) denote the three RL stages.

THINC answers are grounded in code execution. We next examine whether the final answer of each trajectory appears in the execution output of at least one code block (Figure 4b). THINC-4B satisfies this condition in 99.2% of trajectories, compared with 88.4% for ReTool and 74.3% for rStar2. Several other baselines are lower still, indicating that a large fraction of their final answers are generated through NL reasoning rather than code execution. As a result, they bypass the interpreter and remain vulnerable to the arithmetic and algebraic errors discussed in Section 1, where even a single NL mistake can corrupt the result. THINC largely removes this failure mode by design: because its trajectory format contains no NL channel between code blocks, the final answer must be derived from interpreter output. Appendix A traces a representative THINC-4B rollout that makes this pattern concrete. On AIME 2026 Problem 3, the model’s t1 contains only an algebraic restructuring of the problem (a + b + ab = (a+1)(b+1) − 1); the answer is then computed and cross-validated through multiple code-driven turns, with the model auditing and refining its own loop logic entirely within the next code block rather than via NL reasoning between blocks.

#### 4.5 Can THINC Recover from Code Failures Without NL Reasoning?

THINC’s code-centric design raises a natural question about robustness: "With no NL reasoning between code blocks, what happens when a code execution fails?" Interleaved TIR can absorb the failure in NL reasoning and reframe the next attempt; THINC cannot. Whether this hurts robustness or helps it is an empirical question we test here.

###### (a) Lines of Code per Trajectory

(b) Code-Grounded Answer Rate

%sampleswherepredtoolresult

Avg.linesofcodepertrajectory

400

99.2%

100

349

88.4%

300

80

74.3%

261

60

49.9% 49.5%

200

40

34.9%

112

102

100

58

20

40

0

0

ThinCReToolrStar2 CoRTASTERDemyAgent

ThinCReToolrStar2 CoRTASTERDemyAgent

- Figure 4: Code-centric reasoning behavior measured on overall benchmarks. (a) Average lines of code per trajectory. (b) Code-grounded answer rate, the fraction of trajectories whose final answer appears in the output of some code block. THINC-4B leads on both metrics.

We measure this with Recovery@k: among trajectories whose first k code blocks all raise an interpreter error, the fraction that still arrives at the correct final answer. We compute the metric on AIME 2024–2026, HMMT 2025, and BeyondAIME, sweeping k from 1 to 5.

1 2 3 4 5

Number of consecutive initial failures (k)

0

10

20

30

40

50

60

70

RecoveryRate(%)

Recovery Rate after Consecutive Code Failures

| |
|---|

| |
|---|

| |
|---|

| |
|---|

ThinC

ThinC-SFT ASTER rStar2 ReTool DemyAgent

| |
|---|
| |

CoRT

Figure 5: Recovery@k under initial code failures. THINC remains substantially more robust as early execution failures accumulate.

Interleaved baselines degrade with k; THINC-4B stays robust. Every interleaved TIR system loses ground as initial failures accumulate (Figure 5). ASTER drops from 52.1% at k = 1 to 18.5% at k = 5; rStar2-Agent collapses from 39.1% to 0%; ReTool, DemyAgent, and CoRT decline along similar trajectories. THINC, in contrast, stays in a narrow 64–69% band across k = 1,2,3, before declining to 54.5% at k = 4 and 33.3% at k = 5. Even at k = 5, it recovers nearly 2× as often as any interleaved baseline.

Partial robustness from the format, the rest from RL. Our SFT data is filtered to retain only trajectories that execute every code block without error, so recovery from failures is not directly demonstrated during SFT. Even so, THINC-4B-SFT already exceeds most interleaved baselines, recovering in 42.9% of trajectories at k = 1 and 19.1% at k = 5. Only ASTER surpasses it, and only at k ≤ 2. The code-centric format, therefore, confers some robustness even before RL. RL on top of this prior lifts recovery by another 20+ points across k = 1,2,3, accounting for most of THINC’s gap over the interleaved baselines.

- 5 Discussion and Conclusion

We introduce THINC, a code-centric approach to TIR in which the model reasons directly in code after a brief NL planning step. THINC-4B reaches 78.1% on five competition-level math benchmarks, outperforming all baselines. Two findings explain the gain: 99.2% of final answers come from interpreter output rather than NL, and THINC maintains 64–69% recovery through three consecutive code failures where interleaved baselines degrade sharply.

Several limitations remain. Due to computational constraints, our experiments are restricted to 1.7B and 4B parameters, and our evaluation is confined to competition-level mathematics. Whether THINC scales to larger models, and whether the code-centric format extends to other tool-integrated reasoning domains, are natural directions for future work.

### References

- [1] Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding, 2025. URL https://arxiv.org/abs/2504.01943.
- [2] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/ forum?id=YfZ4ZPt8zd.
- [3] Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. Tool-star: Empowering llm-brained multi-tool reasoner via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.16410.
- [4] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms, 2025. URL https://arxiv.org/abs/2504.11536.
- [5] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models, 2023. URL https://arxiv.org/ abs/2211.10435.
- [6] Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. ToRA: A tool-integrated reasoning agent for mathematical problem solving. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=Ep0TtjVoap.
- [7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [8] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.
- [9] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [10] Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, and Dayiheng Liu. Teaching language models to reason with tools, 2025. URL https://arxiv.org/abs/2510.20342.
- [11] Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.
- [12] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/ 2508.10925.
- [13] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=Ti67584b98.

- [14] ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.
- [15] Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, Ying Xin, Ziming Miao, Scarlett Li, Fan Yang, and Mao Yang. rstar2-agent: Agentic reasoning technical report, 2025. URL https: //arxiv.org/abs/2508.20722.
- [16] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [17] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [18] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in LLMs for enhanced mathematical reasoning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=z8TW0ttBPp.
- [19] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J.
- [20] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [21] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https: //arxiv.org/abs/2503.14476.
- [22] Zhaochen Yu, Ling Yang, Jiaru Zou, Shuicheng Yan, and Mengdi Wang. Demystifying reinforcement learning in agentic reasoning, 2025. URL https://arxiv.org/abs/2510. 11701.
- [23] Xuqin Zhang, Quan He, Zhenrui Zheng, Zongzhang Zhang, Xu He, and Dong Li. Aster: Agentic scaling with tool-integrated extended reasoning, 2026. URL https://arxiv.org/ abs/2602.01204.
- [24] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

### A Case Study: A THINC-4B Trajectory on AIME 2026 Problem 3

To illustrate the structural pattern of THINC, we walk through one full rollout on AIME 2026 Problem 3. The trajectory shows how a single brief strategic plan in NL transitions into a sequence of code-centric reasoning steps, each conditioned on the previous interpreter output, with the final answer emerging from code execution rather than from NL-level computation.

#### A.1 Problem

Find the number of integers less than or equal to 100 that are equal to a + b + ab for some choice of distinct positive integers a and b.

|70|
|---|

The reference answer is

.

#### A.2 Stage 1 — Strategic Planning (t1)

The single <think> block contains no numerical computation. It performs one algebraic restructuring of the problem—factoring a+b+ab = (a+1)(b+1)−1—and specifies the resulting search problem to be solved in code.

<think>

I need to find the number of integers ≤ 100 that can be written as a + b + ab where a and b are distinct positive integers.

Let me think about this expression: a + b + ab = (a+1)(b+1) − 1. So I need to find how many integers n ≤ 100 satisfy n = (a+1)(b+1) − 1 for some distinct positive integers a, b. This means n + 1 = (a+1)(b+1) where a, b ≥ 1 and a ̸= b.

Let m = a+1, n′ = b+1. Then m, n′ ≥ 2, m ̸= n′, and m · n′ = n + 1. So I need to count how many values of n ≤ 100 can be written as m · n′ − 1 where m, n′ ≥ 2, m ̸= n′, and m · n′ ≤ 101.

Actually, let me just brute force this. Since a and b are distinct positive integers, let me iterate over possible values and check which sums are ≤ 100. Let me write code to compute this directly.

This realises the role THINC reserves for t1 (Section 3.1): a high-level strategic plan, not a step-bystep derivation. Three properties of this think block illustrate the design intent:

- 1. No arithmetic in NL. The model never evaluates an expression or enumerates a case manually. The think channel is used purely for symbolic restructuring of the problem.
- 2. Search-space specification. It fixes the exact constraints (m,n′ ≥ 2, m ̸= n′, m·n′ ≤ 101) before any code is written, providing a clean starting point for the code blocks that follow.
- 3. Single transition to code. The block ends with one sentence (“Let me write code to compute this directly”) and from this point onward, NL does not return: every subsequent reasoning step is carried out inside a <python> block.

#### A.3 Stage 2 — Code-Centric Reasoning

The remainder of the rollout consists of five <python>/<result> exchanges. Each code block builds on the execution outputs of the preceding blocks, as prescribed by Eq. 6, and every reasoning step—including self-correction, structural verification, and independent re-derivation—is performed in code.

- Turn 1: Initial brute force. The model implements a direct enumeration with an early break that it pre-emptively flags as potentially incorrect.

##### <python> Turn 1

# (a+1)(b+1) - 1 -- brute force over distinct (a,b) results = set() for a in range(1, 100):

for b in range(a+1, 100): # b > a to keep pairs distinct n = a + b + a*b if n <= 100:

results.add(n) elif n > 100:

break # potentially premature -- revisited next turn print(f"Number of distinct values: {len(results)}")

##### <result>

Number of distinct values: 70

- Turn 2: Self-correction in code. Conditioning on Turn 1’s output, the model revisits its loop bound and formalises the monotonicity argument (n = a + b(1+a) is monotone in b for fixed a) directly as a code comment, then reruns with the corrected condition. The audit and fix occur entirely within the code lines; no NL reasoning between blocks is required.

<python> Turn 2

# For fixed a, n = a + b(1+a) is monotone in b. # So ‘break‘ is justified -- but only after we hit n > 100, not earlier. results = set() for a in range(1, 100):

for b in range(1, 100): if a == b: continue n = a + b + a*b if n > 100: break results.add(n)

print(f"Number of distinct values: {len(results)}") print("All values sorted:", sorted(results))

<result>

Number of distinct values: 70 All values sorted: [5, 7, 9, 11, 13, 14, 15, 17, 19, 20, 21, 23, 25, 26, 27, 29, 31, 32, 33, 34, 35, 37, 38, 39, 41, 43, 44, 45, 47, 49, 50, 51, 53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99]

- Turn 3: Structural sanity check. The model decomposes the result by parity to confirm there is no off-by-one error in the value set.

<python> Turn 3

values = sorted(results) odds = [v for v in values if v % 2 == 1] evens = [v for v in values if v % 2 == 0] print(f"Count: {len(values)}") print(f"Odd: {len(odds)}, Even: {len(evens)}")

<result>

Count: 70 Odd: 48, Even: 22

- Turn 4: Independent re-derivation. The model re-implements the count in the dual formulation (u,v) = (a+1,b+1) and verifies that it matches Turn 2.

##### <python> Turn 4

# Method 2: enumerate (u, v) with u, v >= 2, u != v, u*v <= 101. results2 = set() for u in range(2, 102):

for v in range(2, 102): if u == v: continue if u * v <= 101:

n = u * v - 1 if n <= 100:

results2.add(n) print(f"Method 2 count: {len(results2)}") print(f"Same as method 1? {len(results2) == 70}")

<result>

Method 2 count: 70 Same as method 1? True

- Turn 5: Complement audit. As a final empirical check, the model lists the 30 values in {1,...,100} that are not attained.

<python> Turn 5

missing = [i for i in range(1, 101) if i not in values] print(f"Missing values (first 10): {missing[:10]}") print(f"Total missing: {len(missing)}") print(f"Total present: {len(values)}")

<result>

Missing values (first 10): [1, 2, 3, 4, 6, 8, 10, 12, 16, 18] Total missing: 30 Total present: 70

##### <answer>

|70|
|---|

<answer> The final answer is

</answer>

#### A.4 What This Trajectory Illustrates

The rollout exhibits, in a single sample, the four behaviours that the THINC format establishes by construction:

Strategy-only t1. The think channel carries a single algebraic insight (a+b+ab = (a+1)(b+1)−1) and the search constraints, with no arithmetic performed in NL. This realises the THINC constraint that t1 expresses strategy rather than derivation (Section 3.1).

Code as the reasoner. Every reasoning step from Turn 1 onward is performed in code. Selfcorrection (Turn 2), structural verification (Turn 3), independent re-derivation (Turn 4), and complement audit (Turn 5) all occur within <python> blocks; NL appears only as inline code comments and never between code blocks.

Conditioning on execution outputs. Turn 2’s repair is grounded in the model’s audit of Turn 1’s

output. Each subsequent turn similarly conditions on o<i, consistent with the formal trajectory structure in Eq. 6.

Code-grounded final answer. The committed answer (70) appears in the interpreter output of

Turns 2, 4, and 5, rather than being generated in NL.

This rollout is the qualitative correlate of the quantitative gap between THINC-4B and interleaved TIR baselines in Table 1 and the 99.2% code-grounded answer rate reported in Section 4.4: the policy has learned to confine NL to strategic planning and to carry out all subsequent reasoning in code.

### B Few-Shot Prompt for Trajectory Distillation

#### B.1 Few-Shot Prompt for Trajectory Distillation

This appendix presents the full 3-shot prompt used to elicit THINC trajectories from the teacher model (Qwen3.5-27B).

Few-shot prompt

You are a math problem solver that works exclusively through Python code execution.

Rules:

- 1. Write Python code inside <python> </python> tags. You will receive execution results inside <result> </result> tags.

- 2. Each code block runs in a separate process. Variables do NOT persist between turns. Re-define or hardcode values from previous results.

- 3. You may think freely before the first code block, but your final output to the user must be ONLY <python> blocks and one <answer> block.

- 4. Embed your reasoning as concise comments inside the code. Each code block should contain short comments explaining: what you are computing and why, based on previous results.

- 5. Solve problems step by step: compute intermediate results in each code block, observe the output, then decide what to compute next based on those results. Each block should advance the solution by one logical step.

- 6. When you have the final answer, provide it inside <answer> The final answer is \boxed{answer} </answer>.

CRITICAL: Each code block is executed in a completely fresh Python process. You MUST re-import libraries and re-define all variables in every code block. Use hardcoded values from previous <result> outputs, not variable names.

Here are some examples: Question: Find the smallest positive integer $k$ with the following property: if each

cell of a $100\\times 100$ grid is dyed with one color and the number of cells of each color is not more than $104$, then there is a $k\\times1$ or $1\\times k$ rectangle that contains cells of at least three different colors.

Output: <python> # 100x100 grid, each color used at most 104 times # Minimum number of distinct colors required by pigeonhole import math total_cells = 100 * 100 max_per_color = 104 min_colors = math.ceil(total_cells / max_per_color) print(f"total cells: {total_cells}, min colors needed: {min_colors}") </python> <result> total cells: 10000, min colors needed: 97 </result> <python> # If every k consecutive cells in a row have <= 2 colors, # colors must form contiguous blocks. # Intermediate blocks need width >= k-1 to prevent a k-window spanning 3 blocks. # So max colors per row = floor((100 - 2) / (k-1)) + 2 # (first and last block can be width 1, each intermediate block >= k-1) for k in range(10, 15):

max_colors_per_row = (100 - 2) // (k - 1) + 2 print(f"k={k}: max colors per row = {max_colors_per_row}")

</python> <result>

- k=10: max colors per row = 12

- k=11: max colors per row = 11

- k=12: max colors per row = 10

- k=13: max colors per row = 10

- k=14: max colors per row = 9 </result> <python> # Construction attempt: divide grid into s x s blocks, one color per block. # Block size s >= k-1 satisfies both row and column constraints. # Each color occupies s^2 cells, need s^2 <= 104. import math for s in range(9, 13):

num_colors = math.ceil(100 / s) ** 2 cells_per_color = s * s print(f"block {s}x{s}: colors={num_colors}, cells_per_color={cells_per_color},

feasible={cells_per_color <= 104}") </python> <result>

- block 9x9: colors=144, cells_per_color=81, feasible=True

- block 10x10: colors=100, cells_per_color=100, feasible=True

- block 11x11: colors=100, cells_per_color=121, feasible=False

- block 12x12: colors=81, cells_per_color=144, feasible=False </result> <python> # s=10 works (100 <= 104). Verify: build 10x10 block grid, # check every 11-window in all rows and columns has <= 2 colors from collections import Counter grid = [[0]*100 for _ in range(100)] color = 0 for bi in range(10):

for bj in range(10): for r in range(bi*10, (bi+1)*10): for c in range(bj*10, (bj+1)*10): grid[r][c] = color

color += 1 counts = Counter(grid[r][c] for r in range(100) for c in range(100)) # check all 11-windows in rows row_ok = all(len(set(grid[r][c:c+11])) <= 2 for r in range(100) for c in range(90)) # check all 11-windows in columns col_ok = all(len(set(grid[r2][c] for r2 in range(r, r+11))) <= 2 for c in range(100)

for r in range(90))

- print(f"k=11 row: {’PASS’ if row_ok else ’FAIL’}, col: {’PASS’ if col_ok else ’FAIL’}")

print(f"colors: {len(counts)}, max per color: {max(counts.values())}") </python> <result>

- k=11 row: PASS, col: PASS colors: 100, max per color: 100 </result> <python> # Same 10x10 blocks: check k=12. Block size 10 < 12-1=11, # so a 12-window crossing two block boundaries will see 3 colors. grid = [[0]*100 for _ in range(100)] color = 0 for bi in range(10):

for bj in range(10): for r in range(bi*10, (bi+1)*10): for c in range(bj*10, (bj+1)*10): grid[r][c] = color

color += 1 found = None for r in range(100):

for c in range(89):

if len(set(grid[r][c:c+12])) >= 3: found = (r, c, grid[r][c:c+12]) break

if found:

break

- print(f"k=12 window: row={found[0]}, start={found[1]}, cells={found[2]}") </python> <result>

- k=12 window: row=0, start=9, cells=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2] </result> <python> # For k=12, any block coloring needs both dimensions >= 11. # Exhaustively check: no rectangle with both dims >= 11 has area <= 104. import math feasible = [] for p in range(11, 105):

for q in range(11, 105): if p * q <= 104:

feasible.append((p, q, math.ceil(100/p) * math.ceil(100/q))) if feasible:

for p, q, nc in feasible:

print(f"{p}x{q}: colors={nc}") else:

print(f"no block with both dims >= 11 fits in 104 (min 11x11={11*11})") </python> <result> no block with both dims >= 11 fits in 104 (min 11x11=121) </result> <answer> The final answer is \\boxed{12} </answer>

Question: In the Cartesian coordinate system \\( xOy \\), the area of the region corresponding to the set of points \\( K = \\{(x, y) \\mid (|x| + |3y| - 6)(|3x|

+ |y| - 6) \\leq 0 \\} \\) is ________. Output: <python> # (f1)(f2) <= 0 means exactly one of f1<=0, f2<=0

- # Region A: |x|+3|y|<=6 -> |x|/6+|y|/2<=1, area = 2*a*b = 2*6*2

- # Region B: |3x|+|y|<=6 -> |x|/2+|y|/6<=1, area = 2*a*b = 2*2*6 from sympy import Rational, solve, symbols x, y = symbols(’x y’)

- area_A = 2 * 6 * 2

- area_B = 2 * 2 * 6 print(f"area A: {area_A}, area B: {area_B}") </python> <result> area A: 24, area B: 24 </result> <python> # Find where the two boundaries intersect in the first quadrant # x+3y=6 and 3x+y=6 from sympy import symbols, solve x, y = symbols(’x y’) sol = solve([x + 3*y - 6, 3*x + y - 6], [x, y]) print(f"intersection of boundaries: {sol}") </python> <result> intersection of boundaries: {x: 3/2, y: 3/2} </result> <python> # First quadrant intersection region AâĹľB has vertices: # (0,0), (2,0) from |3x|+|y|=6, (3/2,3/2) from boundary crossing, (0,2) from

|x|+3|y|=6 # Compute area via shoelace, then multiply by 4 for all quadrants from sympy import Rational pts = [(0,0), (2,0), (Rational(3,2), Rational(3,2)), (0,2)] n = len(pts) s = 0 for i in range(n):

j = (i+1) % n s += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]

area_q1 = abs(s) / 2 area_intersection = 4 * area_q1 print(f"first quadrant: {area_q1}, total intersection: {area_intersection}") </python> <result> first quadrant: 3, total intersection: 12 </result> <python> # K = symmetric difference of A and B = A + B - 2*(AâĹľB)

- area_A = 24

- area_B = 24 area_intersection = 12 area_K = area_A + area_B - 2 * area_intersection print(f"area K = {area_K}") </python> <result> area K = 24 </result> <answer> The final answer is \\boxed{24} </answer> Question: The area of an equilateral triangle ABC is 36. Points P, Q, R are located

on BC, AB, and CA respectively, such that BP = 1/3 BC, AQ = QB, and PR is perpendicular to AC. Find the area of triangle PQR.

Output: <python> # Equilateral triangle area = (sqrt(3)/4)*s^2 = 36, solve for side length import sympy as sp

- s = sp.Symbol(’s’, positive=True) side = sp.solve(sp.sqrt(3)/4 * s**2 - 36, s) print(f"side: {side[0]}, side^2: {sp.simplify(side[0]**2)}") </python> <result> side: 4*3**(3/4), side^2: 48*sqrt(3) </result> <python> # Place triangle: B at origin, C at (a,0), A at (a/2, a*sqrt(3)/2) # P on BC with BP = 1/3 BC, Q is midpoint of AB (AQ = QB) import sympy as sp from sympy import sqrt, Rational, Matrix a = sp.Symbol(’a’, positive=True)

- B = Matrix([0, 0])

- C = Matrix([a, 0]) A = Matrix([a/2, a*sqrt(3)/2])

- P = B + Rational(1,3) * (C - B)

- Q = (A + B) / 2

- print(f"P = {P.T}")

- print(f"Q = {Q.T}") </python> <result>

- P = Matrix([[a/3, 0]])

- Q = Matrix([[a/4, sqrt(3)*a/4]]) </result> <python> # R on CA: R = A + t*(C - A). Find t such that PR is perpendicular to AC. import sympy as sp from sympy import sqrt, Rational, Matrix a, t = sp.symbols(’a t’, positive=True) A = Matrix([a/2, a*sqrt(3)/2])

C = Matrix([a, 0]) P = Matrix([a/3, 0])

- R = A + t * (C - A) AC = C - A PR = R - P # PR Âů AC = 0 for perpendicularity t_val = sp.solve(PR.dot(AC), t)[0] R_val = A + t_val * (C - A)

print(f"t = {t_val}")

- print(f"R = {sp.simplify(R_val).T}") </python> <result>

- t = 2/3

- R = Matrix([[5*a/6, sqrt(3)*a/6]]) </result> <python> # Area of PQR using shoelace formula with P=(a/3,0), Q=(a/4,sqrt(3)a/4),

R=(5a/6,sqrt(3)a/6) import sympy as sp from sympy import sqrt, Abs a = sp.Symbol(’a’, positive=True)

- P = (sp.Rational(1,3)*a, 0)

- Q = (sp.Rational(1,4)*a, sqrt(3)/4*a)

- R = (sp.Rational(5,6)*a, sqrt(3)/6*a) area = Abs(P[0]*(Q[1]-R[1]) + Q[0]*(R[1]-P[1]) + R[0]*(P[1]-Q[1])) / 2 area = sp.simplify(area) print(f"area PQR = {area}") </python> <result> area PQR = 5*sqrt(3)*a**2/72 </result> <python> # Substitute a^2 = 48*sqrt(3) from the first step import sympy as sp area_PQR = 5 * sp.sqrt(3) * 48 * sp.sqrt(3) / 72 print(f"area PQR = {sp.simplify(area_PQR)}") </python> <result> area PQR = 10 </result> <answer> The final answer is \\boxed{10} </answer>

### C Tool Call and Response Length per Benchmark

This section reports the average number of tool calls per trajectory and the average response length on each evaluation benchmark.

- C.1 Tool Calls by Benchmark C.2 Response Length by Benchmark

### D OOD Generalization

We evaluate THINC-4B on GPQA-Diamond [13], a graduate-level science benchmark covering physics, chemistry, and biology, as an out-of-distribution (OOD) test. Table 2 reports avg@16 and best@16 accuracy. THINC-4B leads on both metrics, exceeding ASTER-4B by 3.1 points on avg@16 and the base model by 7.6 points on best@16, suggesting that the code-centric reasoning format generalizes more effectively than the interleaved alternative beyond the mathematical training domain.

Table 2: GPQA-Diamond accuracy (%). Best per column in bold. Model avg@16 best@16

Qwen3-4B-Thinking 66.32 83.84 ASTER-4B 63.42 90.40 THINC-4B 66.48 91.41

Mean Tool Calls per Benchmark

###### Integrated

###### AIME '24

###### AIME '25

12

14

11.1

11.8

13.2

12

11.0

12

10

11.2

ToolCallsperSample

ToolCallsperSample

ToolCallsperSample

10

9.1

9.2

9.7

10

8

7.9

8

8.3

7.0

6.7

7.0

8

6.1

7.2

6

6.7

5.8

6

5.4

6

4.7

4.7

4

3.8

4

4

2.7

2.5

2.7

2

1.7

2

1.6

1.7

2

0

0

0

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

###### AIME '26

###### HMMT '25

Beyond AIME

16

14

12

11.4

13.8

12.3

14

12

10

12

ToolCallsperSample

ToolCallsperSample

ToolCallsperSample

9.4

10.9

10.9

10

8.7

9.2

9.1

10

8

7.4

8.8

8

7.0

8

6.3

7.2

6

5.6

6.0

6

6.3

5.6

5.7

6

4.4

4.8

4

4

4

3.1

2.3

2.2

1.8

2.2

2

2

2

1.4

0

0

0

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

- Figure 6: Average tool calls per benchmark. We compare how often models invoke the Python interpreter across AIME 2024–2026, HMMT 2025 February, and BeyondAIME.

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

0

5000

10000

15000

20000

ResponseLength(tokens)

13462 13068

20366

18641

12620

15396

10955

7301

11037

Integrated

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

0

2500

5000

7500

10000

12500

15000

17500

ResponseLength(tokens)

9219 9366

16848 16614

12460 12646

9682

7252

8659

AIME '24

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

0

5000

10000

15000

20000

ResponseLength(tokens)

11205 11116

19267

17440

12069

13686

10009

8717

10062

AIME '25

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

0

2500

5000

7500

10000

12500

15000

17500

20000

ResponseLength(tokens)

11030 11111

18608

17657

12854

13857

10393

7203

10028

AIME '26

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

0

5000

10000

15000

20000

ResponseLength(tokens)

13342

12659

19819

18485

12587

14011

11200

8128

14012

HMMT '25

ThinC-4BThinC-4B-SFTThinC-1.7BThinC-1.7B-SFTCoRT-1.5BASTER-4BDemyAgent-4BReTool-32BrStar2-14B

0

5000

10000

15000

20000

25000

ResponseLength(tokens)

16178

15474

22442

19951

12772

17610

11716

6671

11453

Beyond AIME

Mean Response Length per Benchmark

- Figure 7: Average response length per benchmark. We report the mean trajectory length across AIME 2024–2026, HMMT 2025 February, and BeyondAIME.

### E Training Details

All training runs use a single node with 8× NVIDIA H200 GPUs. We perform SFT using the LLaMA-Factory framework [24] and RL using the verl framework [17]. Both THINC-1.7B and THINC-4B share identical hyperparameter configurations at each training stage.

- E.1 Supervised Fine-Tuning
- E.2 Reinforcement Learning

Following DAPO [21], we apply token-level loss normalization across the rollout group, asymmetric clipping with ϵlow = 0.20 and ϵhigh = 0.28, and disable the KL penalty. Table 4 lists the full hyperparameter configuration along with the three-stage curriculum.

Table 3: SFT hyperparameters for THINC-1.7B-SFT and THINC-4B-SFT.

#### Hyperparameter Value

Per-device batch size 1 Gradient accumulation steps 2 Global batch size 16 Learning rate 7 × 10−6 LR scheduler Cosine Warmup ratio 0.1 Number of epochs 3 Context length 32,768 Precision bfloat16

Table 4: RL hyperparameters for THINC-1.7B and THINC-4B. Hyperparameter Value

Algorithm GRPO Learning rate 1 × 10−6 Loss normalization Token-level

ϵlow 0.20 ϵhigh 0.28 KL coefficient 0 Rollout prompt batch size 128 Trajectories per prompt (G) 8

- Stage 1 Context length 16,384 Max tool calls 20 Problem set Full DAPO-Math-17k

- Stage 2 Context length 16,384 Max tool calls 20 Problem set Stage 1 set with 100%-pass-rate problems removed

- Stage 3 Context length 32,768 Max tool calls 40 Problem set Same as Stage 2

