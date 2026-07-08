# arXiv:2603.03202v3[cs.CL]1Jun2026

## Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?

Dadi Guo1*, Yuejin Xie2*, Qingyu Liu3*, Weixian Huang4*, Jiayu Liu1, Zhiyuan Fan1, Qihan Ren5, Shuai Shao5, Tianyi Zhou6, Jianjie Feng7, Wenze Su2, Yujiu Yang2, Dongrui Liu5†, Yi R. (May) Fung1†

1Hong Kong University of Science and Technology, 2Tsinghua University, 3Zhejiang University, 4Nanjing Tech University, 5Shanghai Jiao Tong University, 6University of Michigan, 7Independent Researcher

### Abstract

As large language models (LLMs) advance their mathematical capabilities toward the IMO and research level, the scarcity of challenging, high-quality problems has become a significant bottleneck for training, evaluation and self-evolution of LLMs. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the originals. This work provides empirical evidence that code-driven agents can serve as a viable mechanism for synthesizing high-difficulty mathematical reasoning problems within scalable computational environments. Code and data is available at https: //github.com/TarferSoul/Code2Math.

### 1 Introduction

Recent large language models (LLMs) have achieved substantial advances in mathematical reasoning, reaching performance comparable to International Mathematical Olympiad (IMO)–level and research problem solving (Huang and Yang, 2025; Shao et al., 2025b; DeepSeek-AI et al., 2025; Gao et al., 2025b). While these results demonstrate the effectiveness of current training paradigms, they also expose an emerging bottleneck: further progress increasingly depends on the availability of

*Equal contribution. †Corresponding authors.

[Figure 1]

SEED PROBLEM: A list of positive integers has the following properties: (1) The sum of the items in the list is 30. (2)The unique mode of the list is 9. (3) The median of the list is a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list.

[Figure 2]

candidates_U = []

def solve_max_n(TARGET_SUM): best_n = 0 best_configs = [] for f in range(2, 50):

for val in range(M + 1, 10): candidates_U.append(val) current_needed = needed

for M in range(1, 10):

for val in candidates_U: if current_needed <= 0: break take = min(current_needed, f - 1) cost_U += take * val current_needed -= take

for h in range(f, 100): cost_L = 0 possible_L = True if M < 2:

possible_L = False

next_val = 11 while current_needed > 0:

else: capacity_L = (M - 1) * (f - 1) if h > capacity_L:

take = min(current_needed, f - 1) cost_U += take * next_val current_needed -= take next_val += 1

possible_L = False

else: full_vals = h // (f - 1) remainder = h % (f - 1) cost_L += (f - 1) * (full_vals * (full_vals + 1) // 2) cost_L += remainder * (full_vals + 1) max_val_used = full_vals + (1 if remainder > 0 else 0) if max_val_used >= M:

total_sum = cost_L + cost_U if total_sum <= TARGET_SUM:

n = 2 * h if n > best_n:

best_n = n best_configs = [(f, M, h, total_sum)]

possible_L = False if not possible_L:

elif n == best_n:

continue

best_configs.append((f, M, h, total_sum)) return best_n, best_configs

cost_U = 10 * f needed = h - f

print(solve_max_n(320))

###### Explore by Code

[Figure 3]

Python Output: (72, [(19, 3, 36, 312), (20, 3, 36, 317)])

[Figure 4]

EVOLVED PROBLEM: A list of positive integers has a sum of 323. The list satisfies three conditions:(1) The unique mode of the list is 10. (2) The median of the list is a positive integer that does not appear in the list. (3) The list contains the maximum possible number of items consistent with the first two conditions. Find this maximum number of items.

Figure 1: Example of code-driven problem evolution. The agent analyzes the seed problem and performs computational exploration to enumerate valid configurations under structural constraints. The empirical findings are then abstracted into an evolved problem with increased combinatorial and structural complexity.

novel and high-difficulty mathematical problems. Such problems are difficult to scale through manual curation, as their construction typically requires deep domain expertise and significant human effort. Consequently, the scarcity of sufficiently challenging and diverse mathematical problems has become a limiting factor for training, evaluation, as well as the self-evolution, which motivating the search for automated approaches to synthesizing high-difficulty mathematical reasoning data.

Many challenging mathematical problems arise from exploratory processes such as conjecture formation, counterexample search, and systematic experimentation over structured spaces. These processes are inherently computational, involving iterative hypothesis testing and verification rather

than purely deductive reasoning. Recent advances in code agents (Yang et al., 2025b; DeepSeek-AI et al., 2025) combine strong reasoning capabilities with access to scalable and executable computational environments (Huang et al., 2025c), enabling large-scale simulation, symbolic manipulation, and automated verification (Cheng et al., 2026). For example, a code agent can empirically explore numerical sequences to uncover latent patterns, or exhaustively search for counterexamples to validate or refute candidate propositions. Such capabilities closely mirror the workflows through which human mathematicians discover and refine new problems. This alignment suggests that code agents provide a promising mechanism for autonomously exploring mathematical spaces and synthesizing novel, challenging problems, offering a scalable source of high-quality mathematical reasoning data.

In this paper, we investigate whether code agents can autonomously evolve existing mathematical problems into new, more challenging ones. We aim to answer three research questions: 1) Are the evolved problems mathematically sound and solvable? 2) Do they present a genuine increase in difficulty for current reasoning models? 3) How efficient is the problem evolution process? To study these questions, we collect 100 seed problems from diverse sources, including textbooks, regional competitions, and mainstream benchmarks like the IMO and AIME. These problems serve as a baseline for the agents to explore systematic modifications and provide a controlled setting for evaluating solvability and difficulty escalation.

Given that adapting mathematical problems is a long-horizon task with long contexts (Luo et al., 2025), we decompose it into three stages handled by distinct agents: the Evolution Agent, the Solvability Verification Agent, and the Difficulty Verification Agent (You et al., 2026), forming a multiagent system (Tran et al., 2025; Han et al., 2024). Inspired by Theory of Mind (Chen et al., 2025b; Qian et al., 2024, 2025; Liu et al., 2025b), the Evolution Agent anticipates the solver’s likely reasoning paths and injects new Aha moments (Guo et al., 2025c) to make the entry point more elusive, even for experienced competitors. The Solvability Verification Agent then checks the evolved problem and proposed solution for consistency and feasibility, rejecting flawed outputs. A logically valid solution provides evidence that at least one solution path exists. Finally, we define difficulty as the Burden of Discovery, namely the challenge of uncovering the

key insight, which guides the Difficulty Verification Agent’s assessment of difficulty increase.

Our system follows the test-time scaling paradigm (Muennighoff et al., 2025; Zhang et al., 2025) by using multiple rollouts (Wang et al., 2025c) from the Evolution Agent until both verification agents’ criteria are satisfied. This gives the agent sufficient room for exploration while also yielding efficiency metrics for problem creation, such as rollout count. Unlike scaling methods based only on text-based long-chain reasoning (Wei et al., 2022), our agent can write code and use mathematical Python libraries such as SymPy, NetworkX, and itertools (Li et al., 2023), enabling symbolic computation and deterministic intermediate feedback to guide evolution.

We evaluate problem evolution with DeepSeekChat, DeepSeek-Reasoner, Gemini-3-Pro-PreviewThinking, Kimi-K2-Thinking, and Seed-2.0-Pro, and test the evolved problems across six solver models. The generated problems remain highly solvable; for example, DeepSeek-Reasoner reaches a 94/98 agreement rate with the external judge, approximately 96%. More importantly, we observe a capability asymmetry: models can construct challenges that exceed their own solving baselines, suggesting that agents can synthesize Burden of Discovery beyond their immediate reasoning capacity. This process, however, is computationally costly, requiring 1.56 to 6.55 failures per success on average, with difficult cases often exceeding 10 iterations. We further present a case study showing that code execution serves as a key exploration engine, enabling a shift from simple verification to deeper structural exploration. Our contributions are as follows:

- • We propose a multi-agent framework that decomposes mathematical problem adaptation into evolution, solvability verification, and difficulty verification, with code execution supporting symbolic reasoning and structured exploration.
- • We conduct experiments with multiple evolution models and six solver models, showing that our framework maintains high solvability while substantially increasing problem difficulty.
- • We identify three key findings: code-driven exploration helps discover hidden insights; models can generate challenges beyond their own

solving baselines; and stronger difficulty enhancement requires nontrivial computational overhead.

### 2 Related Works

Data Synthesis through Exploration. Recent studies have leveraged models’ environment exploration ability to synthesize new data. AgentEvolver (Zhai et al., 2025), WebExplorer (Liu et al., 2025c), TaskCraft (Shi et al., 2025), Go-Browse (Gandhi and Neubig, 2025), and Cognitive KernelPro (Fang et al., 2025) enable models to explore environments and progressively generate agent data. TRACE (Guo et al., 2025b) and AutoCode (Zhou

- et al., 2025) further show that multi-agent systems can evolve general-purpose and coding tasks through exploration with verification. However, these works mainly focus on agent task generation and rarely address mathematical reasoning tasks. AlphaGeometry (Trinh et al., 2024) explores new geometric problems from known structures, but relies on a specialized symbolic deduction engine for geometry. Math Problem Adaptation and Generation. Prior work has adapted mathematical problems into new training or benchmark data. MATH-Perturb (Huang et al., 2025b), EvolMathEval (Wang et al.,

- 2025a), and Benchmark Self-Evolving (Wang et al.,
- 2025b) modify existing benchmark problems to evaluate reasoning robustness. However, they often depend on manual effort or simple rule-based LLM edits, leaving the agentic potential of models underexplored. Another line of work, including R-zero (Huang et al., 2025a), Self-Question Language Model (Chen et al., 2025a), UltraLogic (Liu

- et al., 2026b), SANDMath (Manem et al., 2025), and RLVE (Zeng et al., 2025), directly generates math problems for training. While effective, these methods make limited use of agentic exploration and may lack systematic evaluation of generated problem quality.

### 3 Method

In this section, we introduce the seed problems used for evolution, the multi-agent framework including three different agents, and the evaluation method.

#### 3.1 Data for Evolution

We collect 100 mathematical problems from diverse sources, covering various fields such as algebra, combinatorics, calculus, sequences, and graph

theory. The data is sourced from standard mathematics problem books, recent regional exams or competitions, the IMO, and common benchmarks such as AIME-2024 and AIME-2025. The diverse sources ensure diversity in both problem content and difficulty levels. We select an additional 6 pairs of problems to serve as examples for adaptation and evaluation. These include expert demonstrations as well as pairs generated through reverse creation (Sun et al., 2025) by LLMs (i.e., derived by first constructing a simple problem from a complex one, then providing the logic to adapt the simple version back to the complex one).

#### 3.2 Multi-Agent System

Our multi-agent system consists of three agents: the Evolution Agent, the Solvability Verification Agent, and the Difficulty Verification Agent. In this section, we introduce the design of each agent individually.

Evolution Agent. The Evolution Agent takes a seed problem and its solution steps as input. We design this agent to operate in two phases. In the first phase, it analyzes the solution of the input problem to identify the cognitive bottleneck for a solver. In the second phase, it performs free exploration based on the original problem to design a more challenging new problem. Drawing inspiration from Theory of Mind (Chen et al., 2025b), we define difficulty here as the Burden of Discovery. Specifically, we require the agent to anticipate how an experienced competition solver would approach the problem, and then deliberately conceal potential insights to make them difficult to uncover, thereby creating an Aha moment (Guo et al., 2025c) within the solution process of the new problem. Additionally, we offer guidance on potential directions by encouraging the agent to explore areas such as tighter mathematical bounds, more sophisticated combinatorial constructions, and underlying patterns in numerical sequences. The output of the Evolution Agent comprises the new problem statement and the proposed solution steps.

Solvability Verification Agent. Determining problem solvability (Peng et al., 2025; Liu et al., 2026a) is non-trivial, as flaws may include not only surface-level errors in the problem statement but also subtle logical inconsistencies. We therefore design the Solvability Verification Agent to operate in two stages: first detecting obvious surface-level errors, then scrutinizing the solution steps proposed by the Evolution Agent. The rationale is that a flaw-

[Figure 5]

[Figure 6]

###### Evolution Agent

###### Solvability Verification Agent

[Figure 7]

###### Phase 1

[Figure 8]

[Figure 9]

[Figure 10]

Check surface-level errors

Analyze cognitive

[Figure 11]

[Figure 12]

[Figure 13]

bottleneck

[Figure 14]

…

System prompt

Check logical errors

###### Phase 2

Generate

[Figure 15]

Evolved problem

[Figure 16]

[Figure 17]

[Figure 18]

Input

[Figure 19]

Thought

Filter

[Figure 20]

[Figure 21]

[Figure 22]

Seed problem

###### Difficulty Verification Agent

[Figure 23]

Empirical Inquiry

Candidate problems

[Figure 24]

Solution

[Figure 25]

[Figure 26]

[Figure 27]

Observation

Evaluate the relative difficulty

Solution

[Figure 28]

Support Support

[Figure 29]

[Figure 30]

Symbolic computation Constraint satisfaction Graph Tools Combinatorial enumeration

Code Tools

[Figure 31]

[Figure 32]

Figure 2: Overview of our multi-agent system. Our pipeline consists of three components: the Evolution Agent, the Solvability Verification Agent, and the Difficulty Verification Agent. It is equipped with code tools related to mathematics. The framework takes an original problem and its solution as input and outputs a validated new problem along with a solution for reference.

less logical chain implies the existence of at least one solution path, serving as a proxy for solvability. Conversely, if the proposed solution contains logical flaws, we discard the problem even if it might be intrinsically solvable. Since many selected problems lack deterministic final answers (e.g., proof problems), we use a set of failure modes (Guo et al., 2025a; Yuan et al., 2025; Liu et al., 2026a) to help the agent diagnose and categorize solution errors.

Difficulty Verification Agent. The Difficulty Verification Agent receives the original and adapted problems, along with their respective solution steps. Following the ToM-based approach above, it evaluates whether the adapted problem introduces an Aha moment that is harder for an experienced competition solver to uncover.

To rigorously assess adaptation quality, the agent uses a 5-point scoring mechanism that distinguishes Artificial Complexity from Cognitive Depth. Scores 1–2 fall below the acceptance threshold and indicate failures to induce genuine difficulty: Score 1 corresponds to an unchanged or regressed solution path, while Score 2 captures difficulty increases caused merely by computational tedium or repetitive procedures. Penalizing such cases prevents the agent from mistaking labor-intensive algebra for intellectual challenge.

Successful adaptations start at Score 3, where the problem breaks standard solution templates and forces a deviation from rote application. Scores 4–

5 are reserved for stronger anti-templating adaptations that turn standard heuristics into traps and require deeper Aha moments. The highest tier, Score 5, further rewards mathematical beauty, such as deep symmetries or unexpected conceptual connections.

#### 3.3 Test-time Exploration through Code

Leveraging the test-time scaling paradigm, our approach involves generating multiple rollouts per input to satisfy the criteria of our dual-verification system. Agents are explicitly governed to utilize code as a tool for empirical inquiry. For instance, they may run numerical simulations to probe for tighter inequality bounds, print out sequences to intuitively spot regularities, or validate their hypotheses by actively searching for counter-examples. We equip the agent with a comprehensive Python sandbox containing a curated suite of libraries spanning symbolic computation (SymPy), constraint satisfaction (Z3), graph theory (NetworkX), and combinatorial enumeration (itertools). This rich toolset empowers the agent to perform rigorous empirical verification across diverse mathematical domains, ranging from high-precision arithmetic to complex topological analysis.

#### 3.4 Evaluation Method

We conduct evaluations on the solvability of the generated questions, the increase in difficulty, the efficiency of the model in evolving questions, and

the role of code during exploration, respectively.

Solvability. There is no deterministic way to assess the solvability of a natural language math problem, so we adopt an LLM-as-a-judge approach (Gu et al., 2024). Our earlier multi-agent framework already includes a solvability check, but frameworks based on different models can vary a lot. Therefore, we introduce a unified third-party model to perform the check in a consistent manner. To maximize the reliability of the evaluation, this third party must be a model with strong and dependable reasoning ability (in our experiments, we chose GPT-5.2-high) (OpenAI, 2025). A problem is deemed solvable only when the third-party judge and the agent reach agreement.

Difficulty. We evalaute different models on the original questions and on the new questions that pass the solvability check, observing whether their accuracy and reasoning length change. Lower accuracy and more reasoning tokens indicate that the evolved questions are harder. Accuracy is also evaluated using GPT-5.2-high as the judge to determine whether there are logical errors in the solution process.

Efficiency. We use the average number of Evolution Agent rollouts to get a qualified new question as a metric to evaluate the agent’s efficiency. In addition, we also compile the distribution of rollout counts across different models as an auxiliary metric.

### 4 Experiments

In this section, we present our experimental setup and the results of our experiments.

#### 4.1 Setup

Models. We use five backbones for the evolution pipeline: DeepSeek-Chat, DeepSeekReasoner (DeepSeek-AI et al., 2025), Gemini3-Pro-Preview-Thinking (Google, 2025), KimiK2-Thinking (Team et al., 2025), and Seed-2.0Pro (Seed, 2026). We evaluate the resulting problems with six solver models: DeepSeekChat, DeepSeek-Reasoner, Qwen3-235B-A22BThinking-2507 (Yang et al., 2025a), Gemini-3Flash-Thinking, GPT-5.2-Medium, and GPT-5.2High (OpenAI, 2025). We use GPT-5.2-High as the external judge for solvability and answer correctness; when it also appears as a solver, judge calls are run as separate invocations.

Agentic Environment. Our multi-agent system is implemented with Smolagents (Roucher

et al., 2025), where agents can execute Python code in a controlled environment. The sandbox provides standard Python utilities, scientific and symbolic computation packages, constraint solving, and graph-analysis tools; the full package list is provided in Appendix A.1.

Implementation Details. We use 100 seed problems, with each seed assigned one evolution run. Each run allows at most 20 rollouts and a maximum trajectory length of 30 steps; a run succeeds only if a generated problem passes both verification agents. During solver evaluation, each model receives up to three attempts per problem and is stopped after either reaching its maximum token limit or a 30minute wall-clock timeout. All models are queried with temperature 0.

Evaluation Metrics. We report Evolution Success Count (ESC), Certified Solvability Count (CSC), Agreement Rate (AR) between internal and external solvability checks, Solve Rate (SR) on both seed and evolved problems, and Average Token Consumption (ATC). Lower Evolution-SR than Origin-SR and higher token consumption indicate increased difficulty. Detailed metric definitions are provided in Appendix A.1.

#### 4.2 Result Analysis

Solvability Verification. Table 1 shows high consistency between the internal verifier and the external judge. Across evolution backbones, the external judge certifies between 74 of 90 and 98 of 98 internally accepted problems; DeepSeekReasoner reaches 94 of 98, and Gemini-3-ProPreview-Thinking reaches 98 of 98. These results provide a direct check on the reliability of the Solvability Verification Agent: although the verifier operates inside the evolving pipeline, most internally accepted generations are also judged valid by the held-out external judge. This suggests that scrutinizing the proposed solution steps for logical flaws is an effective way to filter invalid generations before they enter downstream difficulty evaluation.

Difficulty Escalation. Most solver–evolver pairs show lower Evolution-SR than Origin-SR, indicating that the evolved problems are systematically harder than the seeds. The effect appears not only on weaker solvers but also on strong closedsource models: GPT-5.2-High drops from 70% to 64% on DeepSeek-Reasoner evolutions and to 61% on Gemini-3-Pro-Preview-Thinking and Seed-2.0Pro evolutions, while Gemini-3-Flash-Thinking drops by up to 32 points. These decreases sug-

Table 1: Cross-model evaluation of problem evolution effectiveness. Column groups indicate the LLM used in the evolutionary phase, while rows indicate the solver model used in the evaluation phase. Each AR cell lists the number of evolved problems certified by the external judge, followed by the number that passed internal verification. Origin-SR(%) and Evolution-SR(%) represent the solve rates on the original seed problems and the evolved problems, respectively. The difference (Evolution-SR − Origin-SR) measures the extent of problem evolution, where a smaller (more negative) value indicates a greater increase in difficulty introduced by the evolutionary process.

DeepSeek-Chat DeepSeek-Reasoner Gemini-3-Pro-Thinking Kimi-K2-Thinking Seed-2.0-Pro AR Origin-SR Evolution-SR AR Origin-SR Evolution-SR AR Origin-SR Evolution-SR AR Origin-SR Evolution-SR AR Origin-SR Evolution-SR

Solver model in evaluation phase

DeepSeek-Chat 83/94 16 19+3 94/98 16 9−7 98/98 16 7−9 74/90 16 11−5 83/97 16 9−7 DeepSeek-Reasoner 83/94 48 34−14 94/98 48 36−12 98/98 48 8−40 74/90 48 39−9 83/97 48 26−22 Qwen3-235B-A22B-Thinking 83/94 20 15−5 94/98 20 19−1 98/98 20 18−2 74/90 20 14−6 83/97 20 11−9

Opensource

Gemini-3-Flash-Thinking 83/94 56 33−23 94/98 56 35−21 98/98 56 24−32 74/90 56 28−28 83/97 56 24−32 GPT-5.2-Medium 83/94 70 59−11 94/98 70 55−15 98/98 70 52−18 74/90 70 58−12 83/97 70 51−19 GPT-5.2-High 83/94 70 70±0 94/98 70 64−6 98/98 70 61−9 74/90 70 67−3 83/97 70 61−9

Closesource

gest that the evolved problems introduce additional Burden of Discovery, forcing solvers to depart from standard solution templates rather than merely handling superficial paraphrases.

Reasoning Strength of the Evolver. The crossmodel pattern also suggests that the reasoning ability of the evolution backbone matters. DeepSeekChat has limited impact on the strongest solver, leaving GPT-5.2-High at 70%, whereas DeepSeekReasoner reduces it to 64% and produces larger drops for other solvers such as Gemini-3-FlashThinking. This contrast indicates that reasoningenhanced evolution agents are more capable of introducing structural modifications that transfer across solver models. It also reveals a capability asymmetry: in some settings, models can synthesize problems that substantially reduce the solve rates of other, sometimes stronger, solvers, leaving room for iterative self-evolution through codedriven exploration (Gao et al., 2025a; Shao et al., 2025a).

Model Robustness. The evolved problems further differentiate solver robustness. Some opensource solvers show smaller absolute drops partly because their Origin-SR is already low, while stronger solvers retain higher absolute accuracy but still degrade on structurally altered problems. For example, Qwen3-235B-A22B-Thinking has a low 20% seed solve rate and often changes only slightly, whereas Gemini-3-Flash-Thinking starts from a stronger 56% baseline and can lose more than 20 points on multiple evolved sets. This pattern indicates that the evolved benchmark is not only harder on average, but also more discriminative: it exposes robustness gaps that are less visible on the original seed problems.

Computational Cost. Figure 3 shows that evolved problems shift the ATC distribution to the right, with more high-token and timeout cases. This supports the SR results from a different angle:

###### 1e 5 Token Distribution Comparison

Original (median=9,606)

- 0

- 1

- 2

- 3

- 4

- 5

- 6

DS-Chat Evolved (median=11,434)

DS-Reasoner Evolved (median=17,277) Gemini-3-Pro Evolved (median=12,726) Kimi-K2 Evolved (median=14,231)

Seed-2-Pro Evolved (median=16,626)

Density

0 10000 20000 30000 40000 50000

Average Tokens per Problem

Figure 3: Distribution of Average Token Consumption (ATC) across original and agent-evolved problems. For each problem, we compute the average output tokens across all solver models. Timeout samples (where solvers failed to produce output) are assigned the maximum token limit to reflect their high difficulty.

Table 2: Average number of failures during problem evolution, and the average numbers due to solvability and difficulty verification failures, respectively.

Model Total Solvability Difficulty

DeepSeek-Chat 4.11 2.65 1.46 DeepSeek-Reasoner 3.10 2.08 1.02 Gemini-3-Pro-Preview-Thinking 1.56 0.46 1.10 Seed-2.0-Pro 2.57 0.78 1.79 Kimi-K2-Thinking 6.55 5.71 0.84

evolved problems require longer reasoning chains and more test-time exploration even when solvers eventually produce an answer. The high-token tail is especially important because it suggests that the added difficulty comes from deeper search and selfcorrection rather than only from longer wording or heavier arithmetic.

Efficiency and Failure Analysis. Table 2 shows that qualified evolution usually requires multiple rollouts. Gemini-3-Pro-Preview-Thinking is the most efficient evolver, with 1.56 failed rollouts on average, while Kimi-K2-Thinking has the highest average failure count at 6.55. The failure decomposition is also informative: most failures come

[Figure 33]

Figure 4: Efficiency Analysis of Agentic Problem Evolution. We visualize the distribution of failure counts encountered during the evolutionary process across three base models: DeepSeek-Chat, DeepSeek-Reasoner, and Gemini-3-Pro-Preview-Thinking. The histograms depict the Total Failures (left), decomposed into rejections by the Solvability Verification Agent (middle) and the Difficulty Verification Agent (right).

from solvability verification rather than difficulty verification, suggesting that maintaining a mathematically valid problem and solution chain is the dominant bottleneck in autonomous problem evolution. Figure 4 further shows that although many problems converge quickly, a non-negligible long tail requires many attempts. This highlights the central trade-off of our pipeline: strict dual verification improves the reliability of accepted problems, but it introduces computational overhead (Liu et al., 2025a).

#### 4.3 Human Evaluation

Audit Setup. Since our automatic evaluation relies on GPT-5.2-High as a judge, we conduct a human audit to validate the reliability of these judgments. The audit covers three parts of the evaluation pipeline: evolved-problem validity, solveranswer correctness, and relative difficulty. The main audit is conducted by a postgraduate student in mathematics, using the same evaluation criteria as the automatic prompts. For solver-answer correctness, we binarize scores by mapping 0.5 to 0 and use a balanced audit set with equal numbers of GPT-positive and GPT-negative cases.

Judge Agreement. Table 3 shows that GPT judgments align closely with human judgments. Among 127 deduplicated evolved problems judged valid by GPT-5.2-High, humans judge 126 as valid, yielding 99.2% validity precision. For solveranswer correctness, human and GPT judgments agree on 1134 of 1226 audited responses. For difficulty scoring, exact agreement on the 1–5 scale is 79.3%, while 95.9% of examples differ by at most one point, with a small mean bias of −0.041. These results suggest that GPT-5.2-High provides a reliable automatic judge for the main evaluation

Table 3: Human audit summary for GPT-5.2-High judgments.

Audit item Metric Result Validity Human-valid rate 126/127 = 99.2% Solver correctness Human–GPT agreement 1134/1226 = 92.5% Difficulty score Exact agreement 115/145 = 79.3% Difficulty score Within-one agreement 139/145 = 95.9% IMO/CMO subset Mean score, Human/GPT 4.03/3.80 IMO/CMO subset Score ≥ 3, Human/GPT 34/35; 34/35

signals used in our experiments; full audit statistics are reported in Appendix A.3.

Olympiad-Level Audit. We further audit 35 evolved problems whose seeds come from highlevel olympiad sources, including IMO shortlist, IMO official, and CMO official problems. Because these examples require stronger mathematical expertise, this subset is additionally reviewed by a mathematics postdoctoral researcher. Under the score ≥ 3 threshold for successful difficulty improvement, both humans and GPT-5.2-High judge 34 of 35 problems as improved. The average human difficulty score is 4.03, compared with 3.80 from GPT. This indicates that the evolved problems are not merely passing a weak validity filter; even on competition-level seeds, human judgments confirm substantial difficulty gains.

Failure After Evolution. Beyond judge agreement, we also test whether evolved problems disrupt solver behavior under human-validated validity. We focus on cases where a solver correctly solves the seed problem but fails on the corresponding evolved problem, restricting the analysis to evolved problems judged valid by both GPT-5.2High and human annotators. This setting directly measures whether evolution breaks the solver’s original solution pattern rather than only changing aggregate solve rates.

- Table 4: Failure after evolution on human-valid problems.

Solver Rate Evolver Rate Diff. DS-Chat 74.2% Gemini-Pro 61.8% 4.55 Qwen-235B 64.3% Seed 55.7% 3.41 DS-Reasoner 55.2% DS-Reasoner 50.7% 3.62 Gemini-Flash 55.2% DS-Chat 41.4% 3.46 GPT-5.2-M 40.0% Kimi-K2 33.9% 3.77 GPT-5.2-H 37.3% Overall 49.1% 3.93 Overall 49.1%

- Table 5: Code ablation solve rates (%). Lower is harder; negative ∆ means code reduces solver accuracy.

Evolver Solver Code No Code ∆ DS-NT DS-NT 13 11 +2 DS-NT DS-T 33 36 -3 DS-NT Gm-F 35 39 -4 DS-T DS-NT 6 9 -3 DS-T DS-T 31 38 -7 DS-T Gm-F 30 34 -4 Seed DS-NT 2 11 -9 Seed DS-T 29 22 +7 Seed Gm-F 29 31 -2

Breakdown. As shown in Table 4, this occurs in 166 of 338 applicable solver–problem pairs, a 49.1% failure-after-evolution rate. Stronger solvers such as GPT-5.2-High and GPT-5.2-Medium are more robust, while DeepSeek-Chat and Qwen3235B-A22B-Thinking are more frequently disrupted. By evolver, Gemini-3-Pro produces the highest disruption rate at 61.8% and also receives the highest average human difficulty score, 4.55. Full count breakdowns are reported in Appendix A.3.

#### 4.4 Ablation on Code

Ablation Design. To isolate the contribution of executable exploration, we compare a code-enabled setting with a no-code setting under the same evolution pipeline. We use three evolution backbones, DeepSeek-V3.2-Non-Thinking (DS-NT), DeepSeek-V3.2-Thinking (DS-T), and Seed-2.0-Pro (Seed), and evaluate the resulting problems with three solvers: DS-NT, DS-T, and Gemini-3-FlashThinking (Gm-F). The code-enabled agents can perform symbolic computation, numerical search, and combinatorial exploration, whereas no-code agents must rely only on internal reasoning.

Quantitative Effect. Table 5 shows that codeenabled evolution lowers solve rates in seven of nine evolver–solver pairs. Averaged across all pairs, the solve rate decreases from 25.7% without code

to 23.1% with code. The effect is therefore positive on average, but not deterministic: DS-NT evaluated by DS-NT and Seed evaluated by DS-T show higher solve rates with code, suggesting that the benefit depends on both the evolver and the solver.

Why Code Helps. The main advantage of code is not merely lower aggregate solve rate, but the type of difficulty it encourages. Code-enabled evolution more often produces concrete and computationally grounded problems, with fixed constants, extremal targets, enumerable structures, or verifiable closed forms. No-code evolution can also reduce solver accuracy, but it more often does so by broadening the task scope, adding parameters, or introducing multi-part proof requirements. Additional qualitative details are provided in Appendix A.4.

Takeaway. The ablation supports executable exploration as a useful but not standalone mechanism for difficulty scaling. Its contribution is strongest when it helps the evolver search structured spaces and verify candidate constructions before finalizing the problem. This makes the accepted problems easier to audit and better aligned with our goal of generating valid, well-specified mathematical challenges.

### 5 Conclusion and Discussion

We presented a code-driven framework for autonomously evolving mathematical problems through test-time exploration and dual verification. By combining executable environments with structured reasoning, the system generates mathematically valid problems that are empirically harder for contemporary solvers, as reflected by consistent declines in solve rates and increased reasoning effort across models.

The evolution process often requires multiple rollouts to satisfy solvability and difficulty criteria, with logical consistency emerging as a primary bottleneck, revealing a trade-off between reliability and computational efficiency. While code execution enables local validation and structural probing, more systematic mechanisms for structural synthesis remain to be explored.

Future work may improve rollout efficiency, strengthen solvability guarantees, and evaluate whether similar exploratory strategies generalize beyond mathematical reasoning. Overall, executable exploration appears to be a viable direction for autonomous difficulty scaling in structured

reasoning domains.

### Limitations

Our work has several limitations. First, the scale of our seed problems is relatively small. We use 100 seed problems in our experiments, mainly because the full pipeline is computationally expensive: each evolved problem requires repeated generation, solving, solvability verification, and difficulty verification. Scaling to a larger and more diverse seed set would provide stronger evidence for the generality of our findings.

Second, although our experiments show that the generated problems are solvable and substantially more difficult than their seed problems, we do not further verify whether these problems can improve model performance when used as training data. Since our focus is on problem adaptation and difficulty enhancement, evaluating the downstream training value of the generated problems remains an important direction for future work.

Third, assessing the quality of mathematical problems is inherently labor-intensive and often requires expert human judgment. Therefore, we only conduct human evaluation on sampled cases rather than the entire generated set. While these samples provide useful evidence for the quality of our framework, a more comprehensive human evaluation would further strengthen the reliability of the conclusions.

### Ethics Statements

This work studies the automatic evolution of mathematical reasoning problems through code-driven agents. The main potential risk is that automatically generated problems may contain subtle errors, misleading solution steps, or inflated difficulty caused by artificial complexity rather than genuine mathematical insight. To mitigate this risk, our framework includes both solvability and difficulty verification agents, and we further conduct sampled human inspection of generated problems. Our work uses and creates scientific artifacts, including seed mathematical problems, generated problem adaptations, agent prompts, code-execution traces, and evaluation results. We release the created data for research purposes only and require users to respect the licenses and terms of use of the original source artifacts from which seed problems are derived. Our use of existing mathematical problems is intended for research on problem adap-

tation, reasoning evaluation, and data synthesis, and the generated derivatives should not be used outside research contexts when the original access conditions impose such restrictions. Since the data consists of mathematical problem statements and solutions rather than personal or user-generated records, it is not expected to contain personally identifying information. Nevertheless, we manually inspected sampled data and filtered problematic outputs, including malformed, offensive, or otherwise inappropriate content.

All computational experiments are conducted using publicly accessible or provider-supported model APIs. Since we do not fine-tune or train models, our experimental setup does not involve hyperparameter search in the conventional training sense; instead, we specify API model versions, prompts, decoding parameters when controllable, tool settings, and evaluation procedures.

We use human annotation only for expert inspection of mathematical problem quality, and do not conduct research with external human subjects or crowdworkers. The annotators are authors of this paper, including researchers with relevant mathematical expertise: one postgraduate student in mathematics and one postdoctoral researcher. They evaluate sampled generated problems for solvability, correctness, and difficulty following the same instruction criteria used for the LLM-based verification agents, so that human inspection is aligned with the automated evaluation protocol. The full annotation instructions are provided in the appendix together with the prompts used for the LLM verification agents. Since the annotation is conducted internally by the paper authors, there is no external participant recruitment, crowdsourcing platform, or separate payment scheme involved. Our data consists of mathematical problem statements, solutions, generated adaptations, and model outputs, rather than personal data collected from individuals; therefore, consent for the use of personal data is not applicable. The annotators were fully aware that their annotations would be used for research evaluation in this paper. As the study does not involve external human subjects or personal data collection, it was not submitted for formal ethics review; it may be considered exempt under typical human-subjects review criteria.

We used web-based GPT models as AI writing assistants to help polish the presentation of the paper, including improving clarity, grammar, and conciseness. The assistants were not used as a

substitute for scientific judgment, experimental design, data annotation, or result interpretation. All substantive claims, experimental results, analyses, and final manuscript content were reviewed and verified by the authors.

### References

Lili Chen, Mihir Prabhudesai, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. 2025a. Selfquestioning language models. arXiv preprint arXiv:2508.03682.

Ruirui Chen, Weifeng Jiang, Chengwei Qin, and Cheston Tan. 2025b. Theory of mind in large language models: Assessment and enhancement. arXiv preprint arXiv:2505.00026.

Daixuan Cheng, Shaohan Huang, Yuxian Gu, Huatong Song, Guoxin Chen, Li Dong, Wayne Xin Zhao, Ji-Rong Wen, and Furu Wei. 2026. Llm-insandbox elicits general agentic intelligence. Preprint, arXiv:2601.16206.

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bing-Li Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, and 244 others. 2025. Deepseek-v3.2: Pushing the frontier of open large language models.

Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, and 1 others. 2025. Cognitive kernel-pro: A framework for deep research agents and agent foundation models training. arXiv preprint arXiv:2508.00414.

Apurva Gandhi and Graham Neubig. 2025. Go-browse: Training web agents with structured exploration. arXiv preprint arXiv:2506.03533.

Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, and 1 others. 2025a. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. arXiv preprint arXiv:2507.21046.

Songyang Gao, Yuzhe Gu, Zijian Wu, Lingkai Kong, Wenwei Zhang, Zhongrui Cai, Fan Zheng, Tianyou Ma, Junhao Shen, Haiteng Zhao, Duanyang Zhang, Huilun Zhang, Kuikun Liu, Chengqi Lyu, Yanhui Duan, Chiyu Chen, Ningsheng Ma, Jianfei Gao, Han Lyu, and 2 others. 2025b. Long-horizon reasoning agent for olympiad-level mathematical problem solving.

Google. 2025. Gemini 3 pro - model card. Technical report, Google.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, and 1 others. 2024. A survey on llm-as-a-judge. The Innovation.

Dadi Guo, Jiayu Liu, Zhiyuan Fan, Zhitao He, Haoran Li, Yuxin Li, Yumeng Wang, and Yi R Fung. 2025a. Mathematical proof as a litmus test: Revealing failure modes of advanced large reasoning models. arXiv preprint arXiv:2506.17114.

Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, and Jing Shao. 2025b. Towards self-evolving benchmarks: Synthesizing agent trajectories via test-time exploration under validateby-reproduce paradigm. ArXiv, abs/2510.00415.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025c. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Shanshan Han, Qifan Zhang, Yuhang Yao, Weizhao Jin, and Zhaozhuo Xu. 2024. Llm multi-agent systems: Challenges and open problems. arXiv preprint arXiv:2402.03578.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. 2025a. R-zero: Selfevolving reasoning llm from zero data. arXiv preprint arXiv:2508.05004.

Kaixuan Huang, Jiacheng Guo, Zihao Li, Xiang Ji, Jiawei Ge, Wenzhe Li, Yingqing Guo, Tianle Cai, Hui Yuan, Runzhe Wang, and 1 others. 2025b. Mathperturb: Benchmarking llms’ math reasoning abilities against hard perturbations. arXiv preprint arXiv:2502.06453.

Yichen Huang and Lin F Yang. 2025. Winning gold at imo 2025 with a model-agnostic verification-and-refinement pipeline. arXiv preprint arXiv:2507.15855.

Yuchen Huang, Sijia Li, Minghao Liu, Wei Liu, Shijue Huang, Zhiyuan Fan, Hou Pong Chan, and Yi R Fung. 2025c. Environment scaling for interactive agentic experience collection: A survey. arXiv preprint arXiv:2511.09586.

Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li FeiFei, Fei Xia, and Brian Ichter. 2023. Chain of code: Reasoning with a language model-augmented code emulator. arXiv preprint arXiv:2312.04474.

Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, and Yi R Fung. 2025a. Costbench: Evaluating multi-turn cost-optimal planning and adaptation in dynamic environments for llm tool-use agents. arXiv preprint arXiv:2511.02734.

Jiayu Liu, Rui Wang, Qing Zong, Qingcheng Zeng, Tianshi Zheng, Haochen Shi, Dadi Guo, Baixuan Xu, Chunyang Li, and Yangqiu Song. 2026a. Naacl: Noise-aware verbal confidence calibration for llms in rag systems. arXiv preprint arXiv:2601.11004.

Jiayu Liu, Qing Zong, Weiqi Wang, and Yangqiu Song. 2025b. Revisiting epistemic markers in confidence estimation: Can markers accurately reflect large language models’ uncertainty? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 206–221, Vienna, Austria. Association for Computational Linguistics.

Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, Jiayuan Song, Zhengmao Zhu, Wenhu Chen, Pengyu Zhao, and Junxian He. 2025c. Webexplorer: Explore and evolve for training long-horizon web agents. ArXiv, abs/2509.06501.

Yile Liu, Yixian Liu, Zongwei Li, Yufei Huang, Xinhua Feng, Zhichao Hu, Jinglu Hu, Jianfeng Yan, Fengzong Lian, and Yuhong Liu. 2026b. Ultralogic: Enhancing llm reasoning through large-scale data synthesis and bipolar float reward. arXiv preprint arXiv:2601.03205.

Haotian Luo, Huaisong Zhang, Xuelin Zhang, Haoyu Wang, Zeyu Qin, Wenjie Lu, Guozheng Ma, Haiying He, Yingsha Xie, Qiyang Zhou, Zixuan Hu, Hongze Mi, Yibo Wang, Naiqiang Tan, Hong Chen, Yi R. Fung, Chun Yuan, and Li Shen. 2025. Ultrahorizon: Benchmarking agent capabilities in ultra longhorizon scenarios. Preprint, arXiv:2509.21766.

Chaitanya Manem, Pratik Prabhanjan Brahma, Prakamya Mishra, Zicheng Liu, and Emad Barsoum. 2025. Sand-math: Using llms to generate novel, difficult and useful mathematics questions and answers. arXiv preprint arXiv:2507.20527.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20286–20332.

OpenAI. 2025. Update to gpt-5 system card: Gpt-5.2. Technical report, OpenAI.

Dengyun Peng, Qiguang Chen, Bofei Liu, Jiannan Guan, Libo Qin, Zheng Yan, Jinhao Liu, Jianshu Zhang, and Wanxiang Che. 2025. Learning the boundary of solvability: Aligning llms to detect unsolvable problems. arXiv preprint arXiv:2512.01661.

Cheng Qian, Bingxiang He, Zhong Zhuang, Jia Deng, Yujia Qin, Xin Cong, Zhong Zhang, Jie Zhou, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. Tell me more! towards implicit user intention understanding of language model driven agents. Preprint, arXiv:2402.09205.

Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, and Huan Wang. 2025. Userbench: An interactive gym environment for user-centric agents. Preprint, arXiv:2507.22034.

Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. 2025. ‘smolagents‘: a smol library to build great agentic systems. https://github.com/ huggingface/smolagents.

Bytedance Seed. 2026. Seed2.0 model card. Technical report, Bytedance.

Shuai Shao, Qihan Ren, Chen Qian, Boyi Wei, Dadi Guo, Jingyi Yang, Xinhao Song, Linfeng Zhang, Weinan Zhang, Dongrui Liu, and 1 others. 2025a. Your agent may misevolve: Emergent risks in self-evolving llm agents. arXiv preprint arXiv:2509.26354.

Zhihong Shao, Yuxiang Luo, Chengda Lu, Z.Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. 2025b. Deepseekmath-v2: Towards self-verifiable mathematical reasoning.

Dingfeng Shi, Jingyi Cao, Qianben Chen, Weichen Sun, Weizhen Li, Hongxuan Lu, Fangchen Dong, Tianrui Qin, King Zhu, Minghao Liu, Jian Yang, Ge Zhang, Jiaheng Liu, Changwang Zhang, Jun Wang, Yuchen Eleanor Jiang, and Wangchunshu Zhou. 2025. Taskcraft: Automated generation of agentic tasks. ArXiv, abs/2506.10055.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, and 1 others. 2025. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5555–5579.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D Nguyen. 2025. Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322.

Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. 2024. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482.

Shengbo Wang, Mingwei Liu, Zike Li, Anji Li, Yanlin Wang, Xin Peng, and Zibin Zheng. 2025a. Evolmatheval: Towards evolvable benchmarks for mathematical reasoning via evolutionary testing. arXiv preprint arXiv:2508.13003.

Siyuan Wang, Zhuohan Long, Zhihao Fan, Xuan-Jing Huang, and Zhongyu Wei. 2025b. Benchmark selfevolving: A multi-agent framework for dynamic llm evaluation. In Proceedings of the 31st international conference on computational linguistics, pages 3310– 3328.

Yumeng Wang, Zhiyuan Fan, Jiayu Liu, and Yi R. (May) Fung. 2025c. Diversity-enhanced reasoning for subjective questions. CoRR, abs/2507.20187.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. ArXiv, abs/2505.09388.

Jian Yang, Xianglong Liu, Weifeng Lv, Ken Deng, Shawn Guo, Lin Jing, Yizhi Li, Shark Liu, Xianzhen Luo, Yuyu Luo, Changzai Pan, Ensheng Shi, Yingshui Tan, Renshuai Tao, Jiajun Wu, Xianjie Wu, Zhenhe Wu, Daoguang Zan, Chenchen Zhang, and 52 others. 2025b. From code foundation models to agents and applications: A comprehensive survey and practical guide to code intelligence. Preprint, arXiv:2511.18538.

Runyang You, Hongru Cai, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li, and Wenjie Li. 2026. Agent-as-a-judge. arXiv preprint arXiv:2601.05111.

Youliang Yuan, Qiuyang Mang, Jingbang Chen, Hong Wan, Xiaoyuan Liu, Junjielong Xu, Jen-Tse Huang, Wenxuan Wang, Wenxiang Jiao, and Pinjia He. 2025. Curing miracle steps in llm mathematical reasoning with rubric rewards. ArXiv, abs/2510.07774.

Zhiyuan Zeng, Hamish Ivison, Yiping Wang, Lifan Yuan, Shuyue Stella Li, Zhuorui Ye, Siting Li, Jacqueline He, Runlong Zhou, Tong Chen, and 1 others. 2025. Rlve: Scaling up reinforcement learning for language models with adaptive verifiable environments. arXiv preprint arXiv:2511.07317.

Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, and Jingren Zhou. 2025. Agentevolver: Towards efficient self-evolving agent system. Preprint, arXiv:2511.10395.

Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, and 1 others. 2025. A survey on test-time scaling in large language models: What, how, where, and how well? arXiv preprint arXiv:2503.24235.

Shang Zhou, Zihan Zheng, Kaiyuan Liu, Zeyu Shen, Zerui Cheng, Zexing Chen, Hansen He, Jianzhu Yao, Huanzhi Mao, Qiuyang Mang, and 1 others. 2025. Autocode: Llms as problem setters for competitive programming. arXiv preprint arXiv:2510.12803.

### A Appendix

In this appendix, we present additional experimental setup details, the prompts used for the three types of agents, and cases of evolution.

#### A.1 Experimental Setup Details

Model Roles. Our model selection strategy distinguishes between the evolutionary and evaluation phases. For the multi-agent evolving system, we employ DeepSeek-Chat, DeepSeekReasoner (DeepSeek-AI et al., 2025), Gemini3-Pro-Preview-Thinking (Google, 2025), KimiK2-Thinking (Team et al., 2025), and Seed-2.0Pro (Seed, 2026) as the base models responsible for problem evolution and initial verification. In the subsequent evaluation phase, we assess the evolved problems using six solver models: DeepSeek-Chat, DeepSeek-Reasoner, Qwen3235B-A22B-Thinking-2507 (Yang et al., 2025a), Gemini-3-Flash-Thinking, GPT-5.2-Medium, and GPT-5.2-High (OpenAI, 2025). We use GPT-5.2High as the external judge for the intrinsic solvability of generated problems and the correctness of solver-generated step-by-step solutions.

Agentic Environment. The multi-agent evolving system is built upon the Smolagents framework (Roucher et al., 2025), which enables agents to execute user-defined Python code within a controlled environment. To support complex problem generation and verification, we equip the agents with a comprehensive toolset. This includes standard utility libraries for general-purpose functionality (json, math, random, statistics, as well as itertools and collections for efficient data manipulation). For precision and textual processing, we include fractions, decimal, re, and functools. Furthermore, the environment supports advanced scientific and symbolic computing through numpy, scipy, pandas, openpyxl, sympy, mpmath, and z3, alongside networkx for graph operations.

Implementation Details. We collect 100 mathematical problems from diverse sources to serve as seed inputs (detailed in Section 3.1); consequently, our experimental pipeline is designed such that each original problem corresponds to exactly one evolved problem instance. For the evolutionary process, we impose a maximum agent trajectory length of 30 steps and a rollout budget of 20 attempts. An evolution instance is deemed successful only if a generated problem passes both verification agents within these 20 rollouts; otherwise, the

evolution is recorded as a failure and terminated. In the evaluation phase, a timeout is triggered if a model exceeds either the maximum token limit or a 30-minute wall-clock duration. We allow each solver model up to three attempts per problem; a consistent timeout across all attempts results in a failure. To ensure reproducibility and deterministic outputs during evaluation, all models are queried with a temperature of 0, utilizing their respective default maximum token limits.

Evaluation Metrics. We use the following metrics:

- • Evolution Success Count (ESC): The total number of problems that successfully pass both verification agents within the 20-rollout limit.
- • Certified Solvability Count (CSC): The number of evolved problems (and their canonical solutions) that are independently verified as solvable by the external judge model.
- • Agreement Rate (AR): The proportion of internally accepted evolved problems that are also certified as solvable by the external judge. In tables, each AR cell lists the external-judgecertified count first and the internally accepted count second. This measures the consistency between the internal Solvability Verification Agent and the external judge.
- • Solve Rate (SR): The proportion of evolved problems correctly solved by a specific solver model. This is determined by the judge model, which evaluates the solver’s step-by-step reasoning and final answer for errors. To quantify the progression of problem complexity, we report both Origin-SR and Evolution-SR, corresponding to the accuracy on the original seed dataset and the evolved dataset, respectively. We posit that a decrease in performance (i.e., Evolution-SR < Origin-SR) reflects an increase in difficulty, with a larger divergence between the two metrics indicating a greater degree of evolution.
- • Average Token Consumption (ATC): The mean number of output tokens generated by solver models when solving each problem. For problems where solvers time out, we impute the maximum token limit to reflect the excessive computational effort. We use this metric as a proxy for problem-solving difficulty,

as harder problems typically require more extensive reasoning chains.

#### A.2 Additional Result Analysis Details

Solvability Verification. The high Agreement Rates (AR) in Table 1 validate the reliability of the Solvability Verification Agent. Across evolution models, the external judge certifies most internally accepted problems as solvable. This suggests that checking the proposed solution steps for logical flaws is an effective filter for invalid generations, and that the framework largely maintains mathematical soundness before the final external judging stage.

Difficulty Escalation. The reduction from OriginSR to Evolution-SR across most solver–evolver combinations indicates that the evolved problems are not merely paraphrases of the seed problems. On problems evolved by DeepSeekReasoner, Gemini-3-Flash-Thinking drops from 56% to 35%, and GPT-5.2-High drops from 70% to 64%. Evolvers such as Gemini-3-Pro-PreviewThinking and Seed-2.0-Pro also reduce GPT-5.2High to 61%, whereas DeepSeek-Chat has little effect on this solver. These results suggest a capability asymmetry: models can sometimes synthesize evolved instances that substantially reduce the solve rates of other, sometimes stronger, solvers, leaving room for self-evolution through iterative code-driven exploration (Gao et al., 2025a; Shao et al., 2025a).

Reasoning Strength and Robustness. Comparing DeepSeek-Chat and DeepSeek-Reasoner shows that stronger reasoning in the evolution phase is associated with more transferable difficulty increases. DeepSeek-Reasoner reduces the accuracy of GPT-5.2-High and produces larger drops for other solvers such as Gemini-3-Flash-Thinking, whereas DeepSeek-Chat has limited impact on the strongest solver. The model-wise variance also shows that evolved problems differentiate solver robustness: lower-baseline models may show small absolute drops, while stronger solvers retain higher absolute accuracy but still lose performance on structurally altered problems.

Token Cost and Evolution Efficiency. The ATC distribution shift in Figure 3 supports the accuracybased difficulty result by showing that evolved problems require longer reasoning chains and more test-time computation. For evolution efficiency, Table 2 and Figure 4 show that strict dual verification introduces nontrivial overhead. Most failures

arise from solvability verification rather than difficulty verification, indicating that maintaining a valid solution chain is the dominant bottleneck in autonomous problem evolution.

- A.3 Additional Human Evaluation Details

Full Audit Statistics. Table 6 reports the full human-audit metrics summarized in the main text, including rank agreement, score calibration, and the high-level olympiad subset.

Full Failure Breakdown. Table 7 reports the counts behind the compact failure-after-evolution table in the main text.

- A.4 Additional Code Ablation Details

Code-Enabled Evolution. Manual inspection shows that code-enabled evolution often yields problems with concrete computational anchors, such as fixed constants, extremal targets, enumerable configurations, or verifiable closed forms. These anchors make the evolved problems easier to audit because the agent can test candidate patterns, rule out invalid constructions, and verify intermediate claims before finalizing the problem.

No-Code Evolution. No-code evolution sometimes produces lower solver accuracy, but the source of difficulty is less consistently tied to a precise mathematical bottleneck. In many cases, the evolved problem becomes broader or more verbose, frequently introducing additional parameters, multi-part proof requirements, or classificationstyle objectives. Such changes may be challenging for solvers, but they can also expand the task scope rather than create a sharper structural obstacle.

Qualitative Distinction. This distinction explains why the code ablation is not deterministic at the level of every solver–evolver pair. Code tools do not guarantee that every generated problem is harder, but they provide a mechanism for grounded exploration and local verification. In representative code-enabled case studies, this process often introduces a genuine increase in mathematical difficulty while preserving coherent and well-specified problem design.

- A.5 Prompt Templates
- A.6 Case Study

- Table 6: Full human audit of GPT-5.2-High judgments. The upper block measures agreement between human annotators and GPT-5.2-High; the lower block focuses on difficulty improvement for IMO/CMO-source problems.

Audit type Scope Metric Result Agreement with human judgments Validity GPT-valid evolved problems Human-valid rate 126 / 127 = 99.2% Solver correctness Balanced solver responses Human–GPT agreement 1134 / 1226 = 92.5% Difficulty score Clean evolved problems Exact agreement 115 / 145 = 79.3% Difficulty score Clean evolved problems Within-one agreement 139 / 145 = 95.9% Difficulty score Clean evolved problems MAE / RMSE 0.262 / 0.632 Difficulty score Clean evolved problems Human − GPT bias −0.041 Difficulty score Clean evolved problems Spearman / QWK 0.714 / 0.632 Difficulty improvement on IMO/CMO-source problems Difficulty score IMO/CMO subset Number of audited problems 35 Difficulty score IMO/CMO subset Mean score, Human / GPT 4.03 / 3.80 Difficulty pass IMO/CMO subset Score ≥ 3, Human / GPT 34 / 35 = 97.1% / 34 / 35 = 97.1% Strong improvement IMO/CMO subset Score ≥ 4, Human / GPT 27 / 35 = 77.1% / 26 / 35 = 74.3% Top score IMO/CMO subset Score = 5, Human / GPT 10 / 35 = 28.6% / 3 / 35 = 8.6% Difficulty pass IMO/CMO subset Pass/fail agreement 33 / 35 = 94.3%

- Table 7: Full failure-after-evolution breakdown. We condition on cases where a solver correctly solves the seed problem, and report how often it fails on the corresponding evolved problem.

Breakdown by solver Breakdown by evolver Solver Seed solved Failed evolved Rate Evolver Seed solved Failed evolved Rate Human diff.

DeepSeek-Chat 31 23 74.2% Gemini-3-Pro 76 47 61.8% 4.55 Qwen-3-235B 14 9 64.3% Seed-2.0-Pro 61 34 55.7% 3.41 DeepSeek-Reasoner 58 32 55.2% DeepSeek-Reasoner 69 35 50.7% 3.62 Gemini-3-Flash 67 37 55.2% DeepSeek-Chat 70 29 41.4% 3.46 GPT-5.2-Medium 85 34 40.0% Kimi-K2 62 21 33.9% 3.77 GPT-5.2-High 83 31 37.3%

Overall 338 166 49.1% Overall 338 166 49.1% 3.93

#### Evolution Agent Prompt Template

- Part 1: Your Mission and Core Principles You are an expert agent specializing in Mathematical Problem Adaptation. Your task is to analyze a given problem and solution, identify its primary bottleneck (what makes it difficult), and the core mathematical insight required to solve it. Using this analysis, you will then formulate a novel, higher-order problem of substantially greater difficulty, and provide a comprehensive, step-by-step solution for the new problem.
- Part 2: Your Working Process Your entire process is a continuous, step-by-step cycle. In each round, analyze the original problem and think carefully about potential directions for adapting the problem, then write code (for example, using sympy for symbolic math) to explore and validate those directions. - Code is your whiteboard for performing symbolic algebra, testing numeric cases, or verifying properties needed to construct the new problem. - After your code is executed, you will receive deterministic output (for example, a simplified expression or a numerical result).
- Part 3: Critical Mandates (Non-Negotiable Process Rules) Workflow Mandate: Exploration First, Final Answer Last (VERY IMPORTANT) Your process has two phases:

- 1) Exploration Phase Move beyond simple derivation. First, adopt the mindset of an elite mathematical competitor and analyze the provided solution to identify the original problem’s true bottleneck. Locate the precise conceptual hurdle or non-obvious starting point that causes difficulty. After isolating this core challenge, engineer a more formidable obstacle. Either escalate the existing bottleneck or design a new, related one, ensuring that the path to the solution is significantly more obscured. This design process must be iterative, using deep mathematical knowledge and computational validation with Python libraries to confirm the integrity and heightened difficulty of your construct. Examples of valued insights: Combinatorics: A delicate construction or a clever bijective argument based on keen observation, not just a standard formula. - Sequences or Number Theory: A subtle underlying pattern, hidden periodicity, or a law governing the distribution of terms. - Analysis: A key qualitative property of a function (such as symmetry, bounds, or geometric meaning of its derivative) that standard procedures would overlook.
- 2) Finalization Phase This is your very last action. After completing derivations and being confident in your new problem, call the final_answer tool with a correctly formatted Python dictionary.

- Part 4: Guiding Principles of Mathematical Construction (Content Rules)

- 1) The Golden Rule of Problem Design: The Burden of Discovery and Insight (CRITICAL PRINCIPLE) Maximize conceptual difficulty to force a hard-won “Eureka” moment. The adapted problem must be constructed so that even a competition-level solver struggles to find the entry point. The solution should only be reachable after extensive observation, experimentation, and trial to unearth the deep insight. Avoid superficial difficulty such as heavier calculations or more procedural steps; the challenge must originate from the intellectual leap required to begin.
- 2) Principle of Logical Integrity and Solvability (CRITICAL PRINCIPLE) Ensure the constructed problem is well-defined, solvable, and unambiguous. Clearly state conditions and constraints. Aim for challenging but fair puzzles.
- 3) New Problem Categories and Answer Formatting (CRITICAL PRINCIPLE) Create challenging math problems in one of the following categories. Match the output format to the problem type.

- Category 1: Definitive Answer Problems (Calculation or Derivation) - Accepted answer types: number, simplified algebraic expression, function, interval or set of numbers. - Output format: - “new_problem”: the problem statement. - “new_solution_steps”: the full derivation and step-by-step logic. - “new_answer”: the final computed result. - If the answer format could be ambiguous, specify it in “new_problem” (for example, “give your answer as a fraction in simplest terms” or “express the function in polynomial form”).
- Category 2: Proof-Based Problems - Nature: ask the solver to prove a mathematical statement. - Output format: - “new_problem”: the statement to prove. - “new_solution_steps”: a complete, rigorous, and logically sound proof. - “new_answer”: None.
- Category 3: Algorithm Design Problems - Nature: ask the solver to design an algorithm meeting specific constraints. - Output format: - “new_problem”: the algorithm design task and constraints. - “new_solution_steps”: a clear description of the algorithm, proof of correctness, and time/space complexity analysis. - “new_answer”: a concise description of the algorithm’s output if applicable. When adapting, preserve the fundamental nature of the original problem. If the original required a numerical result, the adapted problem should also have an answer that is simple to verify. For proof or algorithm design problems, maintain the original format.

- Part 5: Final Output Specification Final Output Format (VERY IMPORTANT) After exploration, your final action is to call the final_answer tool. Its argument must be a single, valid Python dictionary with exactly three keys: “new_problem”, “new_solution_steps”, and “new_answer”. Key Descriptions - “new_problem”: a clear, complete, self-contained description of the evolved problem. - “new_solution_steps”: a numbered, human-readable summary of the key logical steps for the solution (derivation, proof, or algorithm). - “new_answer”: for definitive-result problems, a string representing the value; for proof or algorithm problems, use the Python literal None.
- Part 6: Available Resources Examples: demonstrations. These show adaptations of math problems and illustrate the workflow and baseline for increasing difficulty. In practice, you must rigorously explore the subject and elevate complexity to the maximum possible extent. Do not oversimplify.
- Part 7: Final Checklist

1) Strictly follow all formatting requirements. 2) You may draw inspiration from high-level competitions (e.g., IMO), but do not copy or superficially adapt known problems. Now begin!

###### Figure 5: The prompt template of our Evolution Agent.

#### Solvability Verification Agent Prompt Template

- Part 1: Your Role & Mission You are the Lead Mathematical Solvability Auditor. Your sole purpose is to ruthlessly stress-test a given mathematical problem to ensure it is logically sound, non-contradictory, and correctly solvable. You act as a firewall against "bad math". You do not care if the problem is interesting or hard; you only care if it is broken. You will receive: 1. problem_text`` : The statement of the new problem. 2. proposed_solution`` : The step-by-step derivation provided by the creator. 3. ànswer`: The expected final result. - Note: This is NOT limited to numerical values. It can be an algebraic expression, a tuple, a set of values, or a function. - For Proof Problems: If the problem asks to "Prove that..." or "Show

that...", this field may be None` `, null` `, or a placeholder string (e.g., "N/A"). In these cases, the "answer" is considered the successful completion of the logical argument in the solution steps. Your work is separated into two distinct phases. You must pass Phase 1 (The Static Check) before moving to Phase 2 (The Logic Audit).

- Part 2: Phase 1 - Static Problem Analysis (The "Sanity Check") Before looking at the solution steps, you must validate the problem_textìtself.` You are looking for internal inconsistencies or definitions that are mathematically illegal. Verification Checklist for Phase 1: 1. Domain & Value Rationality: - Are all constants and variables within valid domains? (e.g., denominators ̸= 0, even roots of negatives, arcsin(x) where |x| > 1, logarithms of non-positive numbers). - Are the physical/geometric values inherently possible? (e.g., Probability P ∈ [0, 1], Triangle sides satisfy a + b > c, Friction coefficient µ > 0, Mass m ≥ 0). 2. Constraint Consistency (Over-definition Check): - Does the problem provide too many conditions that contradict each other? - Action: You MUST attempt to model the problem’s geometric or algebraic constraints in Python (e.g., using sympyòr` geometric coordinate geometry). If the constraints lead to an empty set (e.g., "Find a real number x such that x > 5 and x < 3"), the problem is INVALID. If the problem fails Phase 1 (contains illegal definitions or contradictions), you stop immediately and report the error.
- Part 3: Phase 2 - Step-by-Step Logic Audit (The "Deep Dive") If the problem text is valid, you proceed to audit the proposed_solution` `. You must verify each step S_1, S_2, ...S_n extensively. For Every Step, Perform These 3 Checks: 1. Conflict Check: Does the intermediate conclusion obtained in this step contradict the original problem_text` ?` (e.g., The problem says x is an integer, but this step derives x = 0.5). 2. Derivation Verification (The Code Audit): - Do not trust the text. Use Python to independently calculate the

transformation from Step N to Step N + 1. - Validating an integral? Use sympy.integrate` `. Solving an equation? Use sympy.solve` `. 3. Logical Fallacy Detection: check if the step commits any of the following specific errors. Formal Failure Modes (The "Red Flags"): - [Transformation Error]: Recasting a target statement into a non-equivalent or strictly weaker

one (e.g., proving A ⇒ B when A ⇐⇒ B was required). - [Over Generalization]: Drawing a universal conclusion from special cases (e.g., verifying for n = 1, 2, 3 and assuming for all n). - [Invalid Construction]: Introducing an object that cannot exist (e.g., "Let f(x) be a polynomial with infinite roots"). - [Wrong Division]: Case analysis that misses possibilities (e.g., checking x > 0 and x < 0 but forgetting x = 0). - [Circular Reasoning]: Using the conclusion as a hidden premise. - [Logic Violation]: Algebraic illegal moves (e.g., dividing by a variable that could be zero). - [Hidden Assumption]: Using a theorem without verifying its preconditions (e.g., applying L’Hopital’s rule without checking 0/0 form). - [Boundary Neglect]: Ignoring edge cases in optimization or integration limits. - [Vague Argument]: Using "obviously" or hand-waving instead of rigorous derivation. Final Holistic Review (The "Verdict"): - Determine whether there are failure modes in the overall logical chain. - Determine whether the final logical conclusion actually answers the specific question asked in problem_text` `.

- Part 4: Your Interactive Workflow: A Multi-Turn Process Your entire process is a continuous, step-by-step cycle. In each round, you should think carefully, then write code (e.g., using sympy` `for symbolic math) to validate your thoughts. - Remember code is your whiteboard for performing symbolic algebra, testing numeric cases, or verifying properties. - After your code is executed, you will receive the deterministic output of your code (e.g., a simplified expression, a numerical result).
- Part 5: Final Output Specification Your final output must be a single call to the `final_answer`tool. The only argument must be a Python dictionary with exactly two keys: "status"ànd`

"reason"` `. Example of a final answer (Passing Scenario): "status": "PASS", "reason": "The problem text is rigorous and self-consistent. The proposed solution’s derivation was verified step-by-step using Python/SymPy. The logic chain is complete and accurate." Example of a final answer (Failing Scenario): "status": "FAIL", "reason": "The solution fails at Step 4

due to a [Wrong Division] error. The logical argument assumes x > 0, but the problem domain allows for x = 0, which leads to a singularity. Furthermore, the Global check reveals a sufficiency failure; the final answer includes a value that satisfies the derived equation but violates the initial geometric constraints."

- Part 6: Notes 1. Your final output must be a single call to the `final_answer`tool. 2. Be meticulous and objective. Your role is a strict verifier. Any discrepancy must be reported and result in a failure. Now begin your audit.

###### Figure 6: The prompt template of our Solvability Verification Agent.

#### Difficulty Verification Agent Prompt Template

- Part 1: Your Mission & Role You are a Specialist in Mathematical Problem Difficulty Assessment. Your mission is to determine if a new_problem represents a significant and elegant leap in difficulty compared to an original_question, warranting a PASS status. You must embody the mindset of an experienced mathematician and educator. Your evaluation should confirm that the adapted problem’s difficulty is elevated in terms of conceptual depth and mathematical insight, rather than simply being made more tedious through increased computational complexity or longer but straightforward procedures. Your judgment is about whether the path to its solution requires a fundamentally deeper level of thinking. A minor increase in difficulty is insufficient and must result in a FAIL. You will be provided with the original problem and its solution, followed by the new problem and its solution. You are to assume the provided solutions are mathematically correct. Your exclusive focus is on comparing the required problem-solving methodologies.
- Part 2: The Core Difficulty Assessment Criteria To receive a PASS, the new task must satisfy the following high-level conditions: The Nature of the Increased Difficulty: Insight over Execution The difficulty increase must stem from a higher-order cognitive demand, not merely from increased computational labor. You must strictly filter out "trivial" difficulty increases, which automatically lead to a FAIL. - INVALID (FAIL) Difficulty Increases: - Computational Inflation: Using larger numbers, more complex functions that don’t change the underlying logic, or requiring more steps of a known algorithm. - Variable Substitution: Simply changing variables or the presentation format without altering the core solution strategy.

- Minor Twists: Adding a simple, straightforward condition that is easily handled by a small modification to the original method. - VALID (Required for PASS) Difficulty Increases: The adapted problem actively resists solution by common templates or straightforward, algorithmic approaches. Its solution path should not be immediately apparent, forcing the solver beyond mere pattern-matching or procedural execution. For clarification, consider a problem that a student in a standard curriculum would find very challenging, but which a student trained for math Olympiads would instantly recognize as a standard problem type solvable by a learned trick. This is a form of pseudo-difficulty based on specialized training, not the universal conceptual depth we seek, and should be avoided. - The solution of the new problem requires a Eureka moment or a non-obvious insight. This insight should either be entirely new to the problem or represent a significantly deeper, elegant and more sophisticated application of the insight required for the original problem. To be specific, we highly value insights that stem from a genuine mathematical discovery. Favored examples include: - In Combinatorics: Devising a delicate construction or a clever bijective argument based on keen observation, rather than just applying a standard formula. - In Sequences or Number Theory: Uncovering a subtle underlying pattern, a hidden periodicity, or a law governing the distribution of terms. - In Analysis: Grasping a key qualitative property of a function (e.g., its symmetry, bounds, or the geometric implication of its derivative) that standard procedures would overlook. - The adapted problem, in its statement or conclusion, represents a clear mathematical escalation from the original. This means the new problem is not just different, but fundamentally deeper. We specifically value adaptations that achieve one of the following: - Generalize the Original Result: The new problem asks to prove a broader theorem, for which the original problem’s result is merely a specific instance or a stepping stone. - Optimize a Condition: The new problem seeks a provably tighter bound or an exact, optimal constant, where the original might have only asked for a simpler inequality or estimate. - Refine the Constraints: The new problem introduces a more subtle or elegant set of constraints that fundamentally alters the problem’s landscape, demanding a more sophisticated understanding to even begin the analysis. In essence, these three criteria work in concert to substantially elevate the burden of discovery. The adapted problem should be constructed such that its entry point or the key idea is deliberately obscured. A successful adaptation is one where the solver’s primary struggle is not with the complexity of computation or the length of the deduction, but with the profound, creative challenge of finding that first crucial insight.

- Part 3: Your Analytical Workflow Your task is a purely intellectual process of comparison and judgment, culminating in a single, final tool call. 1. Analyze the Original: First, deeply understand the original_question and its original_solution_steps. Classify its difficulty and the core insight required to solve it. 2. Analyze the New: Next, analyze the new_problem and its new_solution_steps. Deconstruct the argument to pinpoint the crucial logical steps and insights required. 3. Compare and Contrast: Directly compare the methodologies. Is the new method just a more laborious version of the old one, or is it fundamentally different? Does it fit the Difficulty Assessment Criteria? 4. Formulate Judgment: Based on the comparison, make a final decision. Does the new problem represent a significant and elegant leap in difficulty, or is the increase minor/computational? This decision directly determines the PASS/FAIL status. 5. Final Output: Once your judgment is formed, proceed directly to the final parts to call the final_answer tool. To calibrate your assessment, you must adopt a specific persona: imagine you are evaluating the problem from the perspective of a skilled and experienced competitor in mathematical olympiads. This is not a novice. This individual has a robust toolkit of standard theorems, inequalities, and problem-solving heuristics. They can quickly identify common patterns and apply standard techniques. Therefore, when you analyze the adapted problem, ask yourself these critical questions: - Would this problem force such a competitor to pause and think? Or would the solution path be immediately obvious to them? - Are their go-to, standard techniques (e.g., a straightforward application of AM-GM, pigeonhole principle, or modular arithmetic) insufficient or intentionally misleading here? - Does the problem present a genuine, non-trivial challenge that would give even this skilled solver a sense of accomplishment upon finding the solution? A problem that is easily dispatched by this benchmark competitor, even if difficult for a layperson, has failed to create a sufficient burden of discovery and should be judged accordingly.
- Part 4: Your Interactive Workflow: A Multi-Turn Process Your entire process is a continuous, step-by-step cycle. In each round, you should think carefully, then write code (e.g., using sympy for symbolic math) to validate your thoughts. - Remember code is your whiteboard for performing symbolic algebra, testing numeric cases, or verifying properties. - After your code is executed, you will receive the deterministic output of your code (e.g., a simplified expression, a numerical result).
- Part 5: Final Output Specification Your final output must be a single call to the final_answer tool. The only argument must be a Python dictionary with exactly three keys: "status", "score", and "reason". - "score": Must be an integer from 1 to 5. You will determine this score based on the scoring rubric below. - "status": Must be one of two exact string values: "PASS" or "FAIL". This value is strictly determined by the score: - If score is 3, 4, or 5, status must be "PASS". - If score is 1 or 2, status must be "FAIL". - "reason": A detailed string of text explaining why you assigned that specific score, referencing the rubric. You will use the following criteria to score the quality of the adaptation. Your reason text must justify your choice of score. - Score 1 (FAIL - Unacceptable): - Fails to change the core solution path of the original problem. - May even lower the difficulty by removing key constraints or adding unhelpful but trivial information. - The burden of discovery is unchanged or reduced. - Score 2 (FAIL - Poor Adaptation): - The difficulty is increased, but only superficially. This increase comes from increased computational complexity (e.g., solving a messier polynomial) or more procedural steps (e.g., applying the same simple idea three times instead of once). - It does not require any new, profound insight. A skilled solver would find it "tedious," not "difficult." The burden of discovery is not meaningfully increased. - Score 3 (PASS - Acceptable / Borderline): - A competent adaptation that successfully increases the burden of discovery. - It manages to break the standard templates required for the original problem, forcing the solver to pause and think of a new angle. - The adaptation might lack elegance, or the new Eureka insight might be relatively straightforward for a top competitor, but it meets the minimum requirement of creating a non-trivial obstacle. - Score 4 (PASS - Excellent): - A high-quality, impressive adaptation that fully aligns with our goals. Anti-Templating: Renders standard solution methods ineffective or turns them into traps. - Requires Eureka: The solution depends on one or more non-trivial, insightful "Aha!" moments. - Mathematical Advancement: The adapted problem is more mathematically interesting—perhaps by connecting different fields, refining constraints, or representing a more general, profound statement. - It provides a genuine challenge and a sense of accomplishment for the skilled competitor. - Score 5 (PASS - Exemplary / Perfect): - Meets all the criteria for a score of 4, but additionally possesses a striking quality of mathematical beauty. - This could be found in the problem’s surprising simplicity, its deep symmetry, an unexpected connection between disparate fields, or its power as a toy model that illustrates a grander concept. - The Eureka moment is not just a key to the solution but also a source of aesthetic satisfaction and a moment of genuine mathematical enlightenment for the solver. This is an adaptation worthy of a textbook.

###### Figure 7: The prompt template of our Difficulty Verification Agent.

#### Solvability Evaluator Prompt Template Your Task:

- 1. Check if the problem statement has any logical errors, contradictions, or is ill-defined.
- 2. Check if the problem is mathematically solvable. Examine the provided reference solution. If the solution is correct as is, or if it can be modified and supplemented to solve the problem, then the problem is considered solvable. Be lenient with the solution; rough drafts with errors are acceptable if they can be corrected to successfully solve the problem. Response Format (JSON): { "has_logic_error": true/false, "logic_error_description": "description of logic error if any, or null", "is_solvable": true/false, "solution_correct": true/false, "solution_issues": ["list of issues in solution, empty if correct"], "overall_valid": true/false, "reason": "brief explanation of your verdict" } Important:

- - A problem is valid if it has no logic errors and is solvable.
- - Small bugs in the reference solution (calculation errors, typos) should not invalidate the problem.
- - Only mark solution_correct as false if there are fundamental errors in the approach.
- - Respond with only the JSON, no additional text.

Figure 8: The prompt template of the solvability evaluator.

#### Solver Prompt Template

Role: You are a distinguished mathematics expert with a strong academic background and a Gold Medal Math Olympiad coach. You excel at solving complex problems in algebra, geometry, calculus, and statistics, placing the highest importance on logical rigor and step-by-step clarity. Task: Solve the mathematical problem provided by the user. Guidelines:

- 1. Deep Reasoning: Before providing the final answer, you must break down the problem and derive each step in detail.
- 2. LaTeX Formatting: All mathematical formulas and variables must be written in LaTeX format (e.g., x2 + y2 = z2).
- 3. Output Format: Output the result strictly as a valid JSON object based on the schema below. Do not output any conversational text, explanations, or Markdown outside of the JSON object. Output JSON Schema: { "question_summary": "A brief summary of the problem", "solution_steps": [ "step_number": 1, "description": "Detailed text explanation of this step", "calculation": "Key calculations or formulas involved in this step (in LaTeX)" , "step_number": 2, "description": "...", "calculation": "..." ], "final_answer": "The final concise answer (numerical value or expression in LaTeX)" }

Figure 9: The prompt template of the problem solver.

#### Solution Evaluator Prompt Template

Role: You are a strict and precise Mathematics Examiner. Your task is to evaluate a student’s solution to a given math problem based strictly on the validity of their logic and result. Input Data:

- **Problem**: problem - **Student’s Solution Steps**: solution_steps - **Student’s Final Answer**: final_answer Evaluation Rules

- 1. **Logic Check**: Review the student’s solution steps. Determine if the mathematical derivation is logically sound and mathematically valid.
- 2. **Consistency Check**: Verify that the final answer naturally follows from the steps provided.
- 3. **Scoring**: Assign a score based strictly on the rubric below. Scoring Rubric

- - **1.0 (Perfect)**:
- - The reasoning is logically sound and complete.
- - The calculation steps are error-free.
- - The final answer is correct.
- - **0.5 (Minor Flaws)**:
- - The final answer is **correct**.
- - The core reasoning logic is correct.
- - BUT there are minor presentation errors, non-standard notation, or slightly skipped trivial steps that do not affect the validity of the result.
- - **0.0 (Incorrect)**:
- - The final answer is **incorrect**.
- - OR the final answer is correct but derived from fundamentally wrong logic (lucky guess).
- - OR the solution is completely missing or irrelevant. Output Format Output **only** a valid JSON object. Do not output any markdown code blocks or additional text. { "score": number, either 0, 0.5, or 1, "reason": "string, a concise explanation of why this score was given, citing specific errors or confirming correctness" }

Figure 10: The prompt template of the solution evaluator.

#### Original Problem:

Given two distinct points P(x1,y1) and Q(x2,y2) on the quadratic function y = x2 + bx − 14, if both points P and Q are on the left side of the axis of symmetry, and x1 and x2 are integers with x1 < x2, prove that x1 − x2 + y1 − y2 > 0.

#### Evolved Problem:

Let fn(x) = x3 + nx2 for a positive integer n. Let Sn be the set of all triangles riangleABC such that the vertices A,B,C are distinct points on the graph of y = fn(x) and their x-coordinates are integers strictly less than the inflection point of the graph. Let ∆n be the minimum possible area of

such a triangle in Sn. Calculate the value of 2025n=1 ∆n. Comment: The adaptation creates an exemplary problem that transforms a specific inequality about integers on a quadratic curve into a rich geometric-algebraic optimization problem on a cubic curve. Justification:

- 1. **Mathematical Depth (Score 5 Criteria):** The problem creates a deep connection between the geometry of the cubic curve (inflection point, collinearity condition) and number theory. The ’Eureka’

moment is realizing that the area of a triangle on the curve y = x3 + nx2 factorizes into a product of coordinate differences and a linear term (x1 + x2 + x3 + n). This mirrors the collinearity condition (x1 + x2 + x3 = −n) and generalizes the original problem’s reliance on x1 + x2 vs −b.

- 2. **Anti-Templating:** The solution path is unique. It requires deriving the specific area determinant,

recognizing the product-minimization implies consecutive integers, and then solving the modular arithmetic constraints imposed by the inflection point. It is far from a standard template.

- 3. **Elegant Complexity:** Unlike a mere computational increase, the move to a cubic curve introduces the inflection point as the natural boundary (analogous to the axis of symmetry), making

the constraint intrinsic to the function’s geometry. The resulting periodic sequence (5, 4, 6) is aesthetically satisfying."

Case 2 Evolver: Gemini-3-Pro-Preview-Thinking

#### Original Problem (From IMO 2025):

Let n ≥ 3 be a given integer. A line in the plane is called **sunny** if it is not parallel to the x-axis, the y-axis, or the line x + y = 0. Consider the set of points Sn = {(a,b) ∈ Z+ | a + b ≤ n + 1}. We want to cover all points in Sn using exactly n distinct lines.

Determine all possible values of k (where k is a non-negative integer) such that there exists a configuration of n lines covering Sn with exactly k sunny lines.

#### Evolved Problem:

Let n ≥ 1 be an integer. Consider the set of points Sn = {(x,y) ∈ Z2 | x ≥ 1,y ≥ 1,x+y ≤ n+1}. We wish to cover all points in Sn using exactly n lines in the Euclidean plane, such that no two lines are parallel (i.e., all n lines have pairwise distinct slopes). Find the largest possible value of n for which such a configuration exists.

#### Comment:

The adaptation is exemplary. It transforms a problem relying on an artificial definition (’sunny lines’) into a natural optimization problem (’distinct slopes’) that forces the solver to rediscover the geometric

constraints. By removing the explicit distinction between boundary and non-boundary lines, the problem elevates the burden of discovery, requiring the solver to recognize that the three boundary directions are limited resources that bound the maximum problem size. The derivation of n=6 is elegant, rigorous, and logically deeper than the original classification task.

#### Original Problem (From IMO 2024):

Let a1,a2,a3,... be an infinite sequence of positive integers, and let N be a positive integer. Suppose that, for each n > N, the number an is equal to the number of times an−1 appears in the list (a1,a2,...,an−1). Prove that at least one of the sequences a1,a3,a5,... and a2,a4,a6,... is eventually periodic.

#### Evolved Problem:

- Let a1 = 1,a2 = 1. For n > 2, let an be the sum of the number of times an−1 has appeared in the sequence (a1,...,an−1) and the number of times an−2 has appeared in the same sequence. Prove that every integer k ≥ 2 appears in the sequence a1,a2,....

Comment: This is an excellent adaptation that merits a score of 4.

- 1. **Shift in Paradigm:** The problem transforms the goal from proving **periodicity** (which relies on identifying a finite set of states and applying the Pigeonhole Principle) to proving **surjectivity** (which requires establishing unboundedness and identifying a specific constructive mechanism). This effectively invalidates the ’tower/block’ visualization and the finite-state automaton approach used in the original problem.

- 2. **Burden of Discovery:** The adaptation elevates the difficulty by hiding the generative mechanism. While the original problem is a technical exercise in bounding relative heights, the new problem is a structural puzzle. The core insight—that the number 2 appears infinitely often and acts as a ’cursor’

which, when paired with a ’fresh’ number, generates the sequence k,k+1,k+2,...—is a non-trivial ’Eureka’ moment. The solver must realize that the most frequent element (2) is paradoxically the engine that creates all the unique large numbers.

- 3. **Mathematical Depth:** The adaptation adds elegance by connecting the *frequency* of terms

to the *value* of future terms in a surprising way. It satisfies the ’Anti-Templating’ and ’Requires Eureka’ criteria perfectly, offering a genuine challenge for a skilled competitor that cannot be solved by rote application of standard sequence techniques.

Determine all composite integers n > 1 that satisfy the following property: if d1 < d2 < ··· < dk are all the positive divisors of n, then di divides di+1 + di+2 for every 1 ≤ i ≤ k − 2.

#### Evolved Problem:

Determine all composite integers n > 1 satisfying the following property: if d1 < d2 < ... < dk are all the positive divisors of n, then the difference di+1 − di divides the difference di+2 − di+1 for every 1 ≤ i ≤ k − 2.

Comment: This is an excellent adaptation that transforms a routine divisibility exercise into a rigorous structural analysis challenge.

**Justification for Score 4:**

- 1. **Escalation of Insight:** The original problem is easily solved using a standard ’Tail Strategy’ (inspecting only the largest divisors n,n/p,n/q), which quickly leads to a contradiction for non-primepowers. The new problem renders this simple heuristic insufficient. It requires a **bi-directional analysis**: the solver must derive constraints from the *start* of the divisor chain (proving q = 2p−1) and synthesize them with constraints from the *end* (limiting the exponents).

- 2. **Anti-Templating:** The condition di+1 − di | di+2 − di+1 governs the *growth rate* of the divisors. This is not a standard template. It forces the solver to discover that the divisors must locally resemble an arithmetic progression, a specific structural insight not present in the original.

- 3. **Richer Solution Set:** The emergence of specific families of composite solutions (n = p(2p − 1)k and n = 2k · 3) adds a non-trivial layer of complexity. Distinguishing the case p = 2 from p > 2 requires careful handling of the gap ratios, demonstrating a clear leap in difficulty.

Let x1,x2,...,x2023 be pairwise different positive real numbers. Define the sequence an by an = (x1 + ··· + xn)(x1

##### + ··· + x1

). Given that an is an integer for every n = 1,2,...,2023, prove that a2023 ≥ 3034.

n

1

Evolved Problem: Let x1,x2,...,x2025 be a strictly increasing sequence of positive real numbers. Define the sequence

an by:

n

n

1 xi

an =

xi

i=1

i=1

Given that an is an integer for every n = 1,2,...,2025, find the minimum possible value of a2025. Comment:

The adaptation transforms a standard inequality problem into a sophisticated sequence analysis task. By changing the constraint from ’distinct’ to ’strictly increasing’, it forces the solver to abandon local algebraic tricks (like the original’s 2-step grouping) in favor of a global structural analysis involving recurrence relations and convexity. The discovery that the monotonicity of inputs enforces a minimum integer step size of 2 (via a derived second-difference inequality) is a profound insight that qualitatively changes the problem. The result is mathematically elegant, optimizes the bound significantly, and requires a high-level ’Eureka’ moment regarding the behavior of the recurrence function.

#### Original Problem (From IMO 2022):

Let k be a positive integer and let S be a finite set of odd prime numbers. Prove that there is at most one way (up to rotation and reflection) to place the elements of S around a circle such that the product of any two neighbors is of the form x2 + x + k for some positive integer x.

#### Evolved Problem:

Let k be a fixed positive integer. Let P be the set of all odd prime numbers. We define a simple undirected graph G = (P,E) where an edge {p,q} exists between distinct primes p,q ∈ P if and only if their product is of the form pq = n2 + n + k for some non-negative integer n. Prove that

the graph G contains no subgraph homeomorphic to the complete graph K4 (i.e., G contains no subdivision of K4). Comment:

This is an excellent adaptation that elevates a specific combinatorial puzzle into a structural graph theory theorem. It retains the deep number-theoretic bottlenecks of the original (quadratic reciprocity/descent) but requires the solver to apply them in a more abstract topological context (proving the absence of a K4 subdivision via graph reduction). This shifts the challenge from ’finding a unique path’ to ’characterizing a graph class’, which represents a significant and elegant increase in conceptual depth.

Case 7 Evolver: Gemini-3-Pro-Preview-Thinking

#### Original Problem (From IMO 2021):

Let n ≥ 100 be an integer. Ivan writes the numbers n,n + 1,...,2n each on different cards. He then shuffles these n + 1 cards, and divides them into two piles. Prove that at least one of the piles contains two cards such that the sum of their numbers is a perfect square.

#### Evolved Problem:

Let n ≥ 100 be an integer. Ivan writes the numbers n,n + 1,...,2n each on different cards. He then shuffles these n + 1 cards and divides them into three piles. Prove that at least one of the piles contains two cards x and y such that either their sum x + y or their absolute difference |x − y| is a

perfect square. Comment: This is an excellent adaptation that significantly raises the burden of discovery.

- 1. **Shift in Combinatorial Hardness:** Moving from 2 piles (requiring a K3) to 3 piles (requiring a

K4) changes the fundamental goal. While K3s are common and easy to parameterize, K4s are rare and structurally demanding.

- 2. **The ’Mixed-Constraint’ Insight:** The introduction of the ’absolute difference’ condition is not merely an additive complication; it is the key that makes the problem solvable. A K4 based

solely on sums or solely on differences is nearly impossible to find in the given range. The solver must have the ’Eureka’ moment that the K4 must be constructed by **interweaving** sum-edges and difference-edges (specifically, the elegant {x,x + 1,y,y + 1} ’prism’ structure). This forces the solver out of the standard ’system of equations’ template used in the original problem and into a task

requiring genuine structural insight.

- 3. **Anti-Templating:** The adaptation invalidates the straightforward algebraic approach of the original (solving a + b = k2) and demands a creative construction. It successfully transforms a standard Diophantine exercise into a complex graph-theoretic challenge.

#### Original Problem (From IMO 2021):

Show that the inequality ni=1 nj=1 |xi − xj| ≤ ni=1

n j=1 |xi + xj| holds for all real

numbers x1,x2,...,xn. Evolved Problem: Determine the set of all positive real numbers p such that for every positive integer n and every sequence of vectors v1,v2,...,vn in a real Hilbert space H, the following inequality holds:

n

n

n

n

∥vi − vj∥p ≤

∥vi + vj∥p

i=1

j=1

i=1

j=1

#### Comment:

This is an exemplary adaptation. It transforms a problem solvable by a specific elementary trick (shifting and concavity for p=1/2) into a profound question regarding the geometry of Hilbert spaces.

The adaptation effectively destroys the original solution method (which fails for p > 1 due to convexity) and forces the solver to discover a much more powerful, dimension-independent tool: the integral representation of negative definite kernels (Schoenberg/Levy-Khintchine). The resulting proof is mathematically beautiful, reducing a complex norm inequality to a trivial identity for complex

numbers.

Case 9 Evolver: Gemini-3-Pro-Preview-Thinking

#### Original Problem (From Chinese Gaokao):

Let n ∈ N∗. Prove that sin2 x · sin2 2x · sin2 4x · ··· · sin2 2nx ≤ 34nn. Evolved Problem: Let fn(x) = nk=0−1 sin2(2kx). We define two asymptotic limits describing the behavior of this sequence:

- 1. The **Analytic Mean Limit**, A, defined by the limit of the n-th root of the average value over the interval [0,π]:

A = lim

n→∞

1 π

π

0

fn(x)dx

1/n

- 2. The **Probabilistic Limit**, B, defined as the almost-everywhere limit of the geometric mean:

fn(x) 1/n (You may assume that this limit exists and is constant for almost all x ∈ [0,π] in the sense of Lebesgue measure.) Calculate the value of the ratio BA. Comment: This is an exemplary adaptation (Score 5) that transforms a standard inequality problem into a profound exploration of the asymptotic properties of chaotic dynamical systems.

B = lim

n→∞

- 1. **Conceptual Depth:** The adaptation elevates the mathematical landscape from elementary calculus (analyzing a recurrence relation’s fixed point) to advanced analysis. It requires two distinct, sophisticated insights: - **Harmonic Analysis (for Limit A):** Recognizing that the integral of the

product simplifies due to the orthogonality of the cosine terms (a property of Riesz products/lacunary series). The ’Eureka’ moment is realizing the binary uniqueness of the frequencies prevents cancellation, allowing for an exact integration. - **Ergodic Theory (for Limit B):** Recognizing the geometric mean limit as a Birkhoff average over the ergodic doubling map. This replaces an

intractable trajectory calculation with a standard log-sine integral.

- 2. **Mathematical Significance:** The problem highlights the subtle difference between the ’Analytic Mean’ (dominated by large outliers) and the ’Probabilistic Limit’ (the typical behavior almost everywhere). It demonstrates that while the original problem’s bound (0.75n) holds, the typical decay (0.25n) is much faster, providing a deeper understanding of the sequence’s behavior.

- 3. **Elegance:** Despite the high-level machinery required, the final result is a clean integer ratio, and the solution path is aesthetically satisfying. It perfectly embodies the criteria of ’Anti-Templating’ and ’Mathematical Beauty’.

A list of positive integers has the following properties: (1) The sum of the items in the list is 30. (2) The unique mode of the list is 9. (3) The median of the list is a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list.

#### Evolved Problem:

A list of positive integers has a sum of 323. The list satisfies three conditions: (1) The unique mode of the list is 10. (2) The median of the list is a positive integer that does not appear in the list. (3) The list contains the maximum possible number of items consistent with the first two conditions. Find this

maximum number of items. Comment: This is an excellent adaptation that transforms a simple arithmetic logic puzzle into a sophisticated constructive optimization challenge.

**Scoring Justification:** 1. **Shift in Cognitive Demand:** The original problem involves trivial case-checking of small partitions summing to 30. The adapted problem, with a sum of 323 and a goal of maximization, makes brute force impossible. It forces the solver to model the problem algebraically, identifying variables for the list length (n = 2h), the mode frequency (f), and the median value (M).

2. **Required Insight (The ’Eureka’ Element):** The problem creates a non-trivial optimization landscape. The solver must discover that the median acts as a ’gate’ for density (choosing M = 3 optimally allows packing 1s and 2s) and that there is a critical trade-off between the mode frequency and the sum: increasing the frequency of the mode (10) allows for more ’cheap’ numbers (1s and 2s)

to be added to the list, but effectively taxes the sum budget. Balancing this ’capacity vs. cost’ is a genuine mathematical hurdle. 3. **Elegant Tuning:** The constraints are fine-tuned such that the maximum length (74) hits the sum target (323) exactly, indicating a high level of craftsmanship in the problem design.

It merits a score of 4 for successfully elevating the burden of discovery from local arithmetic to global optimization.

Case 11 Evolver: Gemini-3-Pro-Preview-Thinking

#### Original Problem (From AIME-2024):

Alice chooses a set A of positive integers. Bob lists all finite nonempty sets B of positive integers such that the maximum element of B belongs to A. Bob’s list has 2024 sets. Find the sum of the elements of A.

#### Evolved Problem:

Alice chooses a finite set A of positive integers such that the distance between any two distinct elements of A is at least 3 (i.e.,|x − y| ≥ 3 for all distinct x, y in A). Bob lists all finite nonempty sets B of positive integers that satisfy two conditions: 1. The maximum element of B belongs to A. 2.

The distance between any two distinct elements of B is at least 3. Bob finds that there are exactly 2024 sets in his list. Determine the sum of the elements of A. Comment:

This is an exemplary adaptation. It transforms a routine problem about binary representation into a sophisticated challenge involving combinatorial recurrence relations and Ostrowski’s numeration systems. The adaptation is mathematically beautiful because it perfectly generalizes the Gap ≥ 1 hidden structure of binary numbers to a Gap ≥ 3 structure. The solver must derive the recurrence relation (an = an−1 + an−3) from scratch and realize that the constraints on set A imply a unique greedy decomposition in this custom number system. This creates a high burden of discovery and offers deep mathematical insight.

Find the number of triples of nonnegative integers (a, b, c) satisfying a + b + c = 300 and a2b + a2c + b2a + b2c + c2a + c2b = 6,000,000.

Evolved Problem: Find the number of triples of positive integers (a, b, c) satisfying a + b + c = 1729 and (ab + bc + ca)3 = abc(a + b + c)3. Comment:

This is an exemplary adaptation that transforms a contrived algebraic puzzle into a deep structural problem connecting algebra and number theory.

- **1. Elevation of Insight (Algebraic):** The original problem relies on a numerical coincidence (manufactured coefficients) that allows the solver to spot that 100 is a root of the cubic polynomial

describing the variables. This is a specific ’trick’ that dissolves the problem instantly. The new problem replaces this with a homogeneous algebraic identity: (ab + bc + ca)3 = abc(a + b + c)3 implies that a,b,c form a Geometric Progression. This is a robust mathematical property, not a numerical artifact, and realizing it requires a significantly deeper insight into symmetric polynomials.

- **2. Elevation of Technique (Number Theoretic):** Once the structure is identified, the original problem trivializes to a basic partition count (b + c = 200). The new problem, however, evolves into a rich number-theoretic task: parametrizing the geometric progression on integers (x = ku2,y = kuv,z = kv2) and solving the resulting Diophantine equation k(u2+uv+v2) = 1729. This requires

knowledge of binary quadratic forms (specifically related to Eisenstein integers) and the behavior of primes modulo 3, moving the problem from simple arithmetic to multiplicative number theory.

- **3. Aesthetic and Depth:** The adaptation replaces arbitrary constants with the Hardy-Ramanujan number 1729, which is thematically appropriate for a problem involving cubic structures. The

solution path moves elegantly from a symmetric identity to a specific study of the quadratic form u2 + uv + v2, demonstrating the kind of ’mathematical beauty’ and cross-disciplinary connection (Algebra to Number Theory) that characterizes a perfect score.

#### Case 13 Evolver: Gemini-3-Pro-Preview-Thinking Original Problem (From AIME-2024):

Let b ≥ 2 be an integer. Call a positive integer n b-eautiful if it has exactly two digits in base b and these two digits sum to √n. Example: 81 is 13-eautiful since 81 = (6 3)_13 and 6 + 3 = √81. Find the least integer b ≥ 2 for which there are more than ten b-eautiful integers.

Evolved Problem:

Let b ≥ 2 be an integer. A positive integer n is called "circular" in base b if it has exactly two digits in base b, say n = (xy)b (where x ∈ {1,...,b − 1} and y ∈ {0,...,b − 1}), and the sum of the squares of these two digits equals n (i.e., x2 + y2 = n). Find the least integer b ≥ 2 for which there

are more than 20 circular integers. Comment:

The adaptation is exemplary. It transforms a problem of elementary divisibility and partitions into a sophisticated challenge involving Diophantine geometry and the Sum of Two Squares Theorem. The

’Eureka’ moment—completing the square to reveal a circle equation (2x−b)2+(2y−1)2 = b2+1—is elegant and conceptually deep. This step bridges the domain of digit problems with the theory of Gaussian integers, satisfying the criteria for mathematical advancement and aesthetic beauty.

#### Case 14 Evolver: Gemini-3-Pro-Preview-Thinking Original Problem (From IMO-2025):

An infinite sequence a1,a2,... consists of positive integers has each of which has at least three proper divisors. Suppose that for each n ≥ 1, an+1 is the sum of the three largest proper divisors of an. Determine all possible values of a1.

Evolved Problem: For a positive integer n, let Σk(n) denote the sum of the k largest proper divisors of n. We say that k is admissible if there exists an integer n having at least k proper divisors such that Σk(n) = n. Determine the set of all admissible integers k in the range 1 ≤ k ≤ 6.

#### Comment:

The adaptation extracts the core fixed-point condition of the original problem and generalizes it into a profound structural question. It introduces a significant conceptual barrier: realizing that the ’sum

of k largest divisors’ condition transforms into a ’divisor-closed Egyptian Fraction’ problem. This insight is non-trivial and turns a calculation-heavy problem into a elegant logical challenge involving deep properties of integer divisors. The result (3, 5) is clean, surprising, and aesthetically pleasing.

- Case 15 Evolver: Gemini-3-Pro-Preview-Thinking Original Problem (From IMO-2025):

Alice and Bazzaz are playing the inekolaty game, a two-player game whose rules depend on a positive real number λ which is known to both players. On the nth turn of the game (starting with n = 1) the following happens:

- - If n is odd, Alice chooses a nonnegative real number xn such that x1 + x2 + ··· + xn ≤ λn.

- - If n is even, Bazzaz chooses a nonnegative real number xn such that x21 + x22 + ··· + x2n ≤ n.

If a player cannot choose a suitable xn, the game ends and the other player wins. If the game goes on forever, neither player wins. All chosen numbers are known to both players. Determine all values of λ for which Alice has a winning strategy and all those for which Bazzaz has a winning strategy.

#### Evolved Problem:

Alice and Bazzaz are playing the Inekolaty game, a two-player game whose rules depend on a positive real number λ which is known to both players. On the n-th turn of the game (starting with n = 1) the following happens:

- - If n is not divisible by 3 (i.e., n ≡ 1,2 (mod 3)), Alice chooses a nonnegative real number xn such

that n

i=1

x2i ≤ λn.

- - If n is divisible by 3 (i.e., n ≡ 0 (mod 3)), Bazzaz chooses a nonnegative real number xn such that n

xi ≤ n.

i=1

If a player cannot choose a suitable xn, the game ends and the other player wins. If the game goes on forever, neither player wins. All chosen numbers are known to both players. Determine all values of λ for which Alice has a winning strategy and all those for which Bazzaz has a winning strategy.

#### Comment:

The adaptation successfully elevates the problem by introducing a strategic degree of freedom (2 moves vs 1 move) that creates a conceptual trap. In the original problem, the player with the quadratic constraint had only one move, making their strategy trivial. In the new problem, that player has two

moves, forcing the solver to choose between ’concentration’ (playing one big move) and ’distribution’ (playing two equal moves). Crucially, while the original problem rewards concentration for the linear-constrained player, the new problem demands distribution for the quadratic-constrained player (x1 = x2 via AM-QM). This forces the solver to understand the fundamental difference between

maximizing L1 norm under L2 constraint versus L2 under L1, rather than simply applying a ’play large moves’ heuristic. The adaptation creates a genuine burden of discovery where the standard intuition leads to the wrong answer. 35

- Case 16 Evolver: Gemini-3-Pro-Preview-Thinking Original Problem (From CMO-2025):

Consider n red cards and n blue cards. Initially, every red card shows 0 and every blue card shows 1. One operation selects a red card with value x and a blue card with value y with x < y, and sets both values to x+2y. The goal is to perform finitely many operations so that the sum of the red card values exceeds 100. Find the minimal positive integer n.

#### Evolved Problem:

Consider n red cards, n white cards, and n green cards. Initially, every red card has value 0, every white card has value 1, and every green card has value 2. An operation consists of selecting two

cards with distinct values x and y (where x < y) and replacing both values with x+2y. The goal is to perform finitely many operations such that the sum of the values of the red cards exceeds 100. Find

the minimal positive integer n for which this is possible.

#### Comment:

The adaptation successfully elevates the problem from finding a static combinatorial bound to analyzing a dynamic process with decaying contributions. It introduces a non-trivial trap: the value gained from the first phase is largely ’quenched’ by the second phase. This requires the solver to

understand linearity and error propagation, rather than simply applying the original formula twice. The shift from a pure combinatorial identity to an asymptotic stability analysis represents a significant and elegant leap in difficulty.

- Case 17 Evolver: Gemini-3-Pro-Preview-Thinking Original Problem (From CMO-2025):

There are 30 colors of cards, with 70 cards for each color. Initially, choose any 70 cards and arrange them in a single stack from top to bottom. One operation is: choose a card X from the bottom 20 cards whose color does not appear among the top 50 cards; then choose a card Y from the top 50 cards whose color appears at least twice among the top 50; remove X and insert it immediately above Y.

Repeat until no such X can be chosen. (I) Prove that only finitely many operations can be performed. (II) Over all initial configurations and sequences of valid operations, determine the maximal possible number of operations.

#### Evolved Problem:

Let there be a deck of N distinct cards with values 1,2,...,N. We maintain a ’Buffer’ consisting of exactly K cards, arranged in a specific vertical order from top to bottom (positions 1 to K). The remaining N − K cards form a ’Reserve’. Initially, you may distribute the N cards between the Buffer and the Reserve and arrange the Buffer’s order in any way you choose.

An operation consists of the following steps: 1. Choose a card X from the Reserve. 2. Choose a card Y from the Buffer such that X < Y . 3. Insert X into the Buffer immediately above Y (at the position Y currently occupies). 4. The cards originally at Y ’s position and below are shifted down by one position. 5. The card that ends up at position K + 1 (the overflow) is removed from the Buffer

and placed into the Reserve. Let f(N,K) be the maximum number of operations that can be performed starting from an optimal initial configuration. (I) Prove that for any finite N, the sequence of operations must eventually terminate. (II) Calculate

the exact value of f(10,3). Comment: The adaptation transforms a problem with arbitrary constants and a somewhat ad-hoc ’color count’ metric into a pristine, generalized combinatorial problem involving permutations and order theory.

- 1. **Mathematical Elevation**: The original solution relies on an arithmetic progression derived from

a specific ’worst-case’ construction. The new problem’s solution (f(N,K) = Kj=1 inomNj − (2K − 1)) links the dynamical process to the volume of a Hamming ball in the Boolean lattice (the

number of subsets of size ≤ K). This is a profound structural insight that elevates the problem from a puzzle to a theorem.

- 2. **Burden of Discovery**: The optimal initial configuration in the new problem is far less obvious than in the original. The solver cannot simply ’guess’ the worst case; they must understand the

underlying state graph’s properties. My computational verification confirmed that the pattern holds, but establishing it requires a non-trivial ’Aha!’ moment connecting the operation to subset generation.

- 3. **Elegance**: By stripping away the ’30 colors, 70 cards’ clutter and replacing it with parameters N,K, the adapter has created a problem of textbook quality that tests deep combinatorial intuition rather than just careful accounting.

- Case 18 Evolver: Gemini-3-Pro-Preview-Thinking Original Problem (From CMO-2025):

Given an odd integer n > 1 such that every prime divisor p of n satisfies gcd(p − 1,n) = 1. Determine the number of arrays (a,b,c) that satisfy: (1) a,b,c ∈ {1,2,...,n} and gcd(a,b,c,n) = 1; (2) there exists a permutation x1,x2,...,xn of {1,2,...,n} such that for all 1 ≤ k ≤ n, n | axk+2 + bxk+1 + cxk, where indices are taken modulo n.

#### Evolved Problem:

Given an odd integer n > 1 such that every prime divisor p of n satisfies gcd(p − 1,n) = 1. Determine the number of arrays (a,b,c,d) that satisfy: (1) a,b,c,d ∈ {1,2,...,n}, d = 1, and

gcd(a,b,c,n) = 1; (2) there exists a permutation x1,x2,...,xn of {1,2,...,n} such that for all 1 ≤ k ≤ n, xk+3 + axk+2 + bxk+1 + cxk ≡ 0 (mod n), where indices are taken modulo n. (Note: the recurrence is xk+3 + axk+2 + bxk+1 + cxk ≡ 0)

Comment: The adaptation is excellent. It elevates the difficulty from handling a specific edge case to mastering a general structural property.

- 1. **Generalization of Difficulty:** In the original problem, the rigorous analysis of ’lifting’ (deter-

mining which coefficient perturbations preserve the permutation property) was only strictly necessary for the prime p = 3. For all other primes, a standard heuristic (’all lifts work’) sufficed. The new problem forces this rigorous analysis for *every* prime. The solver cannot bypass the ’stability condition’ of the recurrence; they must derive it.

- 2. **Structural Depth:** By increasing the order to 3, the problem introduces a free parameter (the third root y) for the characteristic polynomial. This requires the solver to not only identify the ’double root 1’ condition (insight from the original) but also to classify and sum over the possible behaviors of the third root. This integrates the ’root finding’ and ’lifting’ phases in a way that is more complex and mathematically satisfying than the original.

- 3. **Anti-Templating:** The new problem invalidates the ’count roots and multiply by powers of p’ template. The density of valid solutions is non-trivial (< 1) for all cases, requiring a genuine understanding of how the recurrence coefficients influence the distribution of values modulo pk.

