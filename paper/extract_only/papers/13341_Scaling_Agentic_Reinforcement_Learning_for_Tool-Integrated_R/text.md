# arXiv:2511.19773v1[cs.AI]24Nov2025

## Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs

Meng Lu1*, Ran Xu2*, Yi Fang1, Wenxuan Zhang3, Yue Yu4, Gaurav Srivastava1, Yuchen Zhuang4, Mohamed Elhoseiny3, Charles Fleming5, Carl Yang2, Zhengzhong Tu6, Guanghua Xiao7, Yang Xie7, Hanrui Wang8, Di Jin8†, Wenqi Shi7†, Xuan Wang1† 1Virginia Tech 2Emory University 3KAUST 4Georgia Tech 5Cisco 6TAMU 7UT Southwestern Medical Center 8Eigen AI

Abstract

While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to “think with images,” i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs. Code and data of VISTA-Gym and VISTA-R1 can be found at https://github.com/Lucanyc/VISTA-Gym.

### 1. Introduction

Recent progress in VLMs has demonstrated strong performance across tasks such as visual question answering, multimodal reasoning, and grounded mathematical problem solving [3, 14, 20, 68]. Many of these advancements stem from incorporating text-based reasoning paradigms, particularly the Chain-of-Thought (CoT) [73] and reinforcement learning (RL) [18], which decomposes complex tasks into

*Equal contribution first authors: Meng Lu, Ran Xu. †Equal correspondence to: Di Jin, Wenqi Shi, Xuan Wang.

intermediate text reasoning steps for problem solving, and leverages outcome-based rewards to refine reasoning quality [22, 32, 36, 38, 54, 62, 63, 89].

Most of the existing VLM reasoning processes still rely on static visual embeddings and shallow cross-modal alignment. As a result, their text-only reasoning struggles to capture the fine-grained visual structures, spatial relationships, and quantitative dependencies present in real-world scenes [16, 90]. These limitations underscore the need for thinking-with-image paradigms [67], where reasoning is tightly coupled with visual perception, enabling richer cross-modal interaction and step-by-step visual reasoning.

To enhance visual-centric reasoning and encourage thinking-with-image behaviors in VLMs, tool-integrated reasoning (TIR) [19, 79, 90] has been introduced recently. TIR equips models with external tools, such as grounding, zoom-in, and search, to facilitate fine-grained perception and reasoning over object interactions. Despite its promise, current open-sourced VLMs still remain inadequate at leveraging these tools for effective reasoning, as shown in Figure 1. This limitation highlights the urgent need for an effective training environment and strategies to select, invoke, and coordinate visual tools dynamically. While there are several studies dedicated to improving tool usage capabilities for VLMs [16, 66, 74, 90], those works are often narrow in scope, often confined to specific tasks (e.g., using image search for knowledge or zoom-in for object grounding). In parallel, recent efforts in gaming and robotics [9, 70] have introduced unified environments for training VLM agents. These platforms provide controllable dynamics, grounded feedback, and structured task spaces, offering clear advantages for robot learning and conceptually aligning with TIR. However, the simulation of toolintegrated thinking in VLMs—especially for open-domain visual reasoning—remains largely underexplored.

Motivated by these challenges, we introduce VISTAGym (Visual-centric Tool-integrated Agentic training environment), a scalable, agentic training environment designed to systematically enhance the reasoning and decision-

ChartQA

###### UniGeo

ChartQA

###### UniGeo

MapQA

MapQA

36

80

50

60

36

90

Acc(%)

Acc(%)

65

28

80

40

50

32

50

70

30

40

28

20

35

60

20

30

24

12

gpt-5 gpt-5 w/ T

gpt-5 w/ R

gpt-5 w/ T&R

Internvl3 Internvl3 w/ T

Internvl3 w/ R Internvl3 w/ T&R

| |
|---|

| |
|---|

| |
|---|

(a) Close-source VLM (gpt-5).

(b) Open-source VLM (Internvl3-8B).

- Figure 1. Directly augmenting VLMs with tools significantly degrades accuracy (w/ T), yet intrinsic reasoning offers limited gains on complex VQA (w/ R). Supplying tool-selection prior knowledge and interleaving reasoning with tool execution improve performance (w/ T&R); gains are task-dependent for commercial VLMs, while small open-source VLMs remain particularly struggling.

making capabilities of VLM agents across complex, realworld scenarios. Specifically, VISTA-Gym wraps up visual tool operations and provides textual feedback on receiving tool call commands by VLMs, and features:

- • Diverse multi-modal reasoning tasks. VISTA-Gym provides a comprehensive VQA suite spanning 7 reasoning tasks across 13 public datasets, supporting training and evaluation of agents with strong generalization and tool-integration skills.
- • Unified, extensible tool interface. VISTA-Gym exposes a standardized API over 26 pre-defined tools that support perception, symbolic manipulation, chart and document interpretation, grounding high-level reasoning in structured intermediate results.
- • Scalable interactive infrastructure. VISTA-Gym accelerates agent training via multithreading, parallel execution, and sequential sampling, enabling efficient trajectory collection and large-scale automated evaluation compatible with diverse agent scaffolds.

Building on VISTA-Gym, we further develop VISTA-R1, a VLM-based agent trained for robust, tool-augmented reasoning. Across five in-domain and six out-of-domain reasoning-intensive VQA benchmarks, VISTA-R1-8B surpasses state-of-the-art open-source baselines of comparable size by 9.51%–18.72%. VISTA-Gym proves to be an effective and scalable solution for developing VLM agents that can robustly interleave reasoning and tool use to solve complex, multi-step visual problems.

### 2. Related Works

RL for VLM Reasoning. Improving the reasoning capabilities for VLMs has been an active research front. [78] synthesize reasoning chains via distilling reasoning knowledge from teacher models. Inspired by the success of R1style training [18], several works attempt to leverage RL [22, 31, 35, 38, 61] to improve VLM reasoning capabilities on visual tasks, including general visual understanding and mathematical reasoning.

RL for Tool-Integrated Reasoning in VLMs. To better characterize the visual information, recent works explore

the “thinking with images” paradigm, which goes beyond standard reasoning steps by incorporating additional tools in the reasoning process [21, 29]. DeepMMSearch-R1 [48] and MMSearch-R1 [74] leverage image search tools to augment the context with external knowledge. Other approaches [16, 66, 89, 90] utilize zoom-in operations to focus on fine-grained visual regions, thereby improving grounding and multi-step perception reasoning. ReLook [30] utilizes a VLM as an auxiliary tool to enable agentic training via cross-model interaction. Despite these advances, most existing methods rely on a single specialized tool and are restricted to narrow task domains, thereby limiting their generalization to broader multimodal reasoning scenarios.

RL Training Environment for Agentic Reasoning. To facilitate agentic model training, several frameworks have been proposed, including GEM [37], AgentGym-RL [76], SkyRL-Gym [5], Collabrative GYM [59] and RAGEN [72] for training general agents. Beyond general frameworks, many efforts target domain-specific applications, such as text-based reasoning [65], software engineering [15, 23, 52, 83], machine learning engineering [49, 55, 56], search and web browsing [10, 77, 80], and scientific reasoning [71, 81]. Moreover, the vast majority of these environments operate purely in text-only settings, providing limited support for multimodal grounding or visual reasoning. For VLMs, to the best of our knowledge, only two environments, namely VLM-Gym [9] and VAGEN [70], are designed to support VLM training, focusing respectively on compositional visual games and embodiment tasks that require intermediate state information. VISTA-Gym fills this gap by providing a scalable environment for tool-integrated RL, enabling systematic development of agentic visual reasoning for VLMs.

### 3. Preliminaries

Problem Setup. We study visually grounded question answering requiring complex reasoning over both image and text to derive a final prediction y. The model is equipped with a predefined, finite tool set (i.e., action space) A. Formally, the objective is to synthesize a trajectory U with

- Table 1. Error pattern identification and distribution from 500 error samples. Note that one case may contain multiple error types.

ID Error Type GPT-5 InternVL3-8B

- E1 Invocation schema violation (wrong function call structure) 12.8% 3.8%

- E2 Invalid argument name (wrong augment name) 0.0% 44.8%

- E3 Invalid argument value (wrong augment value format) 18.2% 18.0%

- E4 Incorrect argument value (wrong argument value content) 39.6% 57.4%

- E5 Invalid output from tool execution (wrong answer format) 25.6% 0.0%

- E6 Incorrect reasoning from tool execution 28.1% 64.8%

interleaved reasoning steps gt and external tool invocations at ∈ A. The resulting solution path is defined as U = (g0,a0,...,gT, y), where T denotes the number of tool-interaction turns required to reach the final answer y.

Tool-Integrated Reasoning in VLMs. Formally, we frame TIR in VLMs as a Partially Observable Markov Decision Process (POMDP) defined by the tuple ⟨S,O,A,I,P,R⟩, where I, S, O, A, and R represent the spaces for instructions, environment states, agent observations, actions, and rewards, respectively. Let P : S × A → S represent the deterministic state transition function. Following ReAct [85], we structure the agent outputs to prioritize reasoning thought before action. Given a problem description x ∈ I, the history, and the current feedback, at each turn t, the agent generates the thought gt+1 ∼ πθ(·|x,g1,a1,o1,··· ,gt,at,ot) first and the subsequent action at+1 ∼ πθ(·|x,g1,a1,o1,··· ,gt,at,ot,gt+1). Following at+1, the environment transitions to a new state st+1 via P(st+1|st,at), and provides a new partial observation ot+1 ∼ O(·|st+1). Thus, the entire trajectory is:

τ = (g0,a0,o0,··· ,oT−1,gT, y) ∼ πθ(τ|x),

where y is the final answer obtained from the model generations, and πθ can be decomposed:

T−1

πθ(gt,at|x,ct−1)

πθ(τ|x) = πθ(gT|x,cT−1) ·

t=0

T−1

= πθ(gT|x,cT−1) ·

πθ(at|x,ct−1,gt) · πθ(gt|x,ct−1),

t=0

where ct−1 = (g0,a0,o0,··· ,gt−1,at−1,ot−1) represents the interactive history up to t − 1.

Exploratory Experiments. We conduct exploratory experiments on three VQA tasks [6, 8, 44] with both proprietary and open-source VLMs. As shown in Fig. 1, naively enabling tools for base models induces a sharp accuracy drop: without instruction or reasoning priors, external tools act as distractors rather than aids. To diagnose failure modes, we annotate 500 tool-enabled errors across GPT-5 and InternVL3-8B (Table 1). The majority of failures arise from the if/when/which/how of tool calls, schema/argument selection, and correctness (E1–E5); fol-

lowed by incorrect post-tool reasoning (E6). These observations motivate us to train VLMs for TIR; To facilitate easy scale-up, we introduce VISTA-Gym (Figure 2): we curate verifiable, visual-centric tasks with tools (Sec. 4.1), instantiate an interactive, executable environment (Sec. 4.2) with scalable training facilities (Sec. 4.3), and systematically improve open-source VLM agent within VISTA-Gym that interleaves reasoning with tool execution (Sec. 5).

### 4. VISTA-Gym: A Scalable Tool-Integrated Agentic Training Environment for VLMs

#### 4.1. Diverse Task and Tool Collection

Reasoning-Intensive VQA Tasks. VISTA-Gym comprises a unified training and evaluation environment that couples diverse multimodal tasks with executable tools and verifiers, emphasizing not only final answers but also auditable sequences of tool calls that ground those answers. We curate verifiable instances from 13 established benchmarks to ensure broad coverage of difficulty and reasoning types, spanning perception (vision), symbolic manipulation (math/geometry), and language understanding (document/chart interpretation). Specifically, training tasks span seven complementary axes (Appendix A): (1) Chart Understanding (FigureQA [26], ChartQA [44]), (2) Geometric Reasoning (Geometry3K [39], GeoQA [7], UniGeo [8]), (3) Geospatial Reasoning (MapQA [6], InfographicVQA [46]), (4) Scientific Reasoning (ScienceQA [41], VizWiz [4]), (5) Document Understanding (DocVQA [45]), (6) Spatial/Compositional Reasoning (CLEVR [25]), and (7) Others (ThinkVL [11], A-OKVQA [58]).

Visual-Centric Tool Sets. Small open-source VLMs often lack the fine-grained perception and domain routines (e.g., precise localization, symbolic formalization, dense chart parsing) required by the above tasks. VISTA-Gym therefore exposes a standardized, extensible tool interface that lets agents offload these subproblems to reliable modules, grounding high-level reasoning in structured intermediate results (e.g., tables, text). Twenty-six tools are organized into four families (Appendix B): (1) Perception, such as GroundingDINO [34], SAM [28], EasyOCR; (2) Chart Understanding, such as ChartMoE [82]; (3) Diagram Formalization, such as CDL, Inter-GPS; and (4) Math Solvers, such as G-LLaVA [17], MultiMath [53].

#### 4.2. Executable Interactive Environment

Interface. VISTA-Gym exposes a Gymnasium-style API with reset() and step(). Given an instruction x ∈ I, reset returns the initial partial observation o0 (question text and supporting image[s]) and initializes the interaction history c0. Each episode is a POMDP mentioned in Sec. 3.

Action Space. The action space A is strictly constrained by the available toolset. We formalize an action as a typed

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### Example Visual-Grounded Question Answering Tool Collection

Action 𝑎

{ 𝑜,𝑎,𝑟 } 𝜋

[Figure 6]

result

tool use answer

Perception:

|[Figure 7]|
|---|

[Figure 8]

|[Figure 9]|
|---|

[Figure 10]

[Figure 11]

- • Detection: […]
- • OCR : […] …

Q: Name the regions that have a value in the range 188-912? A: [California, Texas, Florida, Ohio]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Chart Understanding:

[Figure 18]

[Figure 19]

thinking tool use result answer

- • ChartToTable : […]
- • ChartToSVG : […] … Diagram Formalization

- • Symbolic Parsing : […]
- • CDL:[…] … Math Solvers:

- • Math: […]

[Figure 20]

Geometric Reasoning: In the circle with center (O), points A, B, and C lie on the circumference. If (∠AOB = 120°) and (∠BOC = 80°), what is the measure of (∠ ABC)? A.60 B.80 C.100 D.120 Answer: B.80

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Geospatial Reasoning: Name the regions that have a value in the range 188-912? Answer: [“California”, “Texas”, “Florida”, “Ohio”]

Q: Name the regions that have a value in the range 188-912? A: [California, Texas, Florida, Ohio]

[Figure 26]

###### VISTA-Gym VLM Agent

Chart Reasoning: Is the median of first 3 points of orange smaller than largest value of green graph? Answer: [“No”]

[Figure 27]

……

[Figure 28]

thinking tool use result thinking

[Figure 29]

𝜋

Obs 𝑜, Reward 𝑟

update policy

… • Multi-modal: […] …

- Figure 2. Overview of VISTA-Gym. VISTA-Gym contains a comprehensive suite of reasoning-intensive VQA tasks and tools in an interactive execution environment, scaling visual-centric tool-integrated agentic training for VLM agents.

### 5. VISTA-R1: Bootstrapping Tool-Integrated Reasoning via RL for VLMs

tuple at ∈ A consisting of the unique identifier and the corresponding arguments passed to the interface for execution. Observation Space. The observation space O comprises environmental feedback after tool execution. Each ot ∈ O encapsulates either successful execution results or runtime error messages (if any), serving as external signals for verifying intermediate hypotheses and adjusting the subsequent reasoning trajectory.

#### 5.1. Two-Stage Training Framework

With the developed VISTA-Gym, we enable a two-stage training framework. First, we bootstrap an interactive agent with behavioral cloning (BC), instilling instructionfollowing and basic tool-use. We then refine it with multiturn, online RL, which encourages deeper reasoning and disciplined, reasoning-guided tool orchestration.

#### 4.3. Scalable Training Facility

VLMs as Tools. To enhance visual perception, we include compute-intensive VLMs like G-LLaVA [17] and ChartMoE [82] as interactive services. To integrate them efficiently into distributed RL, we deploy a highly concurrent microservice architecture that encapsulates each VLM as an independent HTTP service with three layers: (i) a FastAPI front end that exposes RESTful endpoints with asynchronous batched requests; (ii) an intermediate Tool layer that parses instruction actions, retrieves image data from trajectory metadata, and formats observations; and (iii) a Ray Actor layer that keeps model weights resident in GPU memory after initialization, eliminating the prohibitive latency of repeated model (re)loading under high-frequency tool calls.

- Stage I: Warmup w. Imitation Learning. We initialize the policy by BC on synthesized expert trajectories that explicitly interleave thoughts and actions. (i) Candidate generation and filtering. We first generate tool-executing trajectories with a proprietary model (GPT-5) and retain only those whose final answers exactly match ground truth (outcome-based filtering). (ii) Rationale densification. To strengthen supervision on reasoning, we replace concise rationales with extended traces produced by an open-weights expert (Qwen3-VL-235B-A22B-Thinking). Let D denote the resulting set of tuples (x,τ) where

τ=(g1,a1,...,gT, y) and ct−1 is the interaction history up to step t−1. The BC objective maximizes the likelihood of interleaved thought–action tokens:

LBC(θ) = E(x,τ)∼D[log πθ(τ|x)]

= E

T−1

t=0

log πθ(at|x,ct−1,gt) +

T

t=0

log πθ(gt|x,ct−1) .

The synthesized corpus covers a diverse tool mix (Figure 3), providing a robust prior for tool syntax and selection.

- Stage II: Online RL. After SFT, we train the agent in the executable environment with multi-turn rollouts, where each step emits a reasoning segment followed by a function call; VISTA-Gym executes the call and appends the (structured) feedback to the context for continued reasoning. For policy improvement, we adopt Group Relative Policy Optimization (GRPO) [60] with group-normalized advantages

Asynchronous Training. To sustain high throughput during RL rollouts, we use Ray to orchestrate concurrency. At each step, the policy emits a <think> segment followed by a <tool_call>; on generation of the sentinel token </tool_call>, decoding halts and the framework assembles batched HTTP requests that include trajectory identifiers and image paths. Ray manages request queues and load balancing across tool servers. For resource efficiency, compute-heavy VLM tools are pinned to dedicated GPUs, while lightweight utilities are multiplexed on shared CPUs. Extensible Tool Set. To support rapid expansion beyond tools used in our experiments, we provide a generic BaseTool interface that enables plug-and-play integration of new tools with minimal boilerplate. Operational robustness is ensured via health and metric endpoints (/health, /metrics) and Ray’s automated failure recovery, which restarts crashed actors without disrupting ongoing training.

over G rollouts:

|τi|

G

1 G

1 |τi|

LGRPO(θ) =

min[ri,k(θ) · Ai,k,

i=1

k=1

clip(ri,k(θ),1 − ϵ,1 + ϵ) · Ai,k],

where ri,k(θ) denotes the token-level importance ratio at the k-th token.

πθ(τi,k|τi,<k) πold(τi,k|τi,<k)

ri,k(θ) =

,

and Ai,k is the normalized advantage across all tokens:

R(τi) − mean({R(τ1),··· ,R(τG)} std({R(τ1),··· ,R(τG)})

Ai,k =

.

#### 5.2. Reward Design

We employ a multi-round interaction protocol, U = (u0,u1,··· ,uT), for agent training, and design the following rewards R for policy update. This protocol mandates the structural format for each turn ui, enforcing an explicit think→tool_call loop prior to a think→answer termination:

- • Turn (< T) (Tool Call): the first turns ui<T should

generate reasoning with function calls:

ui<T = <think>···</think><tool_call>···</tool_call>.

- • Turn T (Final Answer): the last turn uT should generate reasoning with final answer:

uT = <think>···</think><answer>···</answer>. Repetition Penalty. We first apply a high-priority repetition detector Rrep(U) ∈ {−3.0,−2.0,−1.5,0} that scans for contiguous token/phrase/character repeats and assigns a severity-dependent negative reward (extreme, severe, moderate, none). This term dominates all subsequent logic.

Format Reward. Conditional on no repetition, Rrep(U)=0, we validate structural well-formedness (e.g., correct tags, order, and non-nested closure) at every turn. We define:

Rformat(U) = (I{all ui conforms the format} − 0.5) × 2.

Correctness Reward. We extract the final prediction y from <answer>...</answer> for rule-based checking. To keep the signal low-noise and format-aware, correctness is credited only for repetition-free, well-formed outputs:

Rcorrect(U) = I{ y = y}.

Final Reward. The rollout-level reward is the sum of the three components:

R(U) = Rrep(U) + Rformat(U) + Rcorrect(U).

This sparse, format-aware design allocates positive reward only to generations that are repetition-free, structurally valid, and correct, thereby encouraging the policy to internalize the intended think→tool_call→answer protocol rather than exploiting intermediate heuristics.

- 0

100

200

300

400

500

600

#oftoolcalls

Chart Geometry Geospatial Scientific Document 3DSpatial Others

Figure 3. Top tool call distribution of different tasks in SFT data.

6. Experiments

6.1. Experiment Setups

Evaluation Datasets. Following several prior works [21, 22, 31, 35], we focus on reasoning-intensive VQA tasks, evaluating the effectiveness of VISTA-Gym in improving the TIR of open-source VLMs. For in-distribution evaluation, we select five out of the thirteen datasets in VISTA-Gym training pool with official held-out test splits: (1) ChartQA [44], (2) Geometry3K [39], (3) GeoQA [7], (4) UniGeo [8], and (5) MapQA [6]. To assess out-of-distribution generalization, we evaluate on six additional benchmarks not used for training: (6) TABMWP [42], (7) AI2D [27], (8) PlotQA [47], (9) CLEVR-Math [33], (10) IconQA [40], and (11) MathVista [43]. Detailed dataset information is in Appendix A.

Baselines. We consider the following baselines on VISTA-Gym: (i) API-based proprietary VLMs, including GPT-5 [50], GPT-5-mini [50], GPT-o4-mini [51], GPT-o3 [51], Gemini-2.5-Pro [12], Gemini-2.5-Flash [12], Claude-4.5-Sonnet [2]; (ii) Open-source VLMs, including InternVL3 [91], Qwen2.5-VL [3], LLaVA-OneVision-

- 1.5 [1]; (iii) Tool/Reasoning-integrated VLMs, including VTool-R1 [75], R1-VL [88], R1-Onevision [84], Perception-R1 [86]. All baselines are evaluated without tool access, as naive tool exposure significantly degrades performance without extensive agentic training (see Table 3); except for tool-integrated baselines following their original tool configurations. Additional baseline details are available in Appendix C. Implementation Details. We consider four different backbones for VISTA-R1 with varying sizes, including InternVL3-2B/8B/14B and Qwen2.5-VL-7B, and implement training with Verl-Tool [24]. For SFT, we train for one epoch with a batch size of 128, a learning rate of 2 × 10−6, and a context length of 16,384 tokens. For RL, we use a micro-batch size of 8 per GPU, a mini-batch size of 128, and G = 8 rollouts per update. We set the regularization coefficient β = 10−3, cap the maximum response length at 26,780 tokens, and optimize with a learning rate of 5×10−7 for E = 300 steps. We adopt accuracy (ACC) as the primary evaluation metric. All experiments are conducted with 8 NVIDIA H200 GPUs, each equipped with 141GB of

ChartToTableQAAnalysisOCR MultimodalMath ChartToTableQAAnalysisOCRCDLQAAnalysisChartToTableOCRCDLQAAnalysisChartToTableQAAnalysisChartToTableOCRCDLMultimodal

- Table 2. Main results (acc%) on five in-distribution and six out-of-distribution VQA benchmarks. † indicates results reported from the original papers. w/o Tools excludes tool access from both training and inference stage; w/o Reasoning removes RL training stage.

Baselines (↓) In-Distribution Out-of-Distribution All Datasets (→) ChartQA Geometry3K GeoQA UniGeo MapQA avg. TABMWP AI2D PlotQA CLEVR-Math IconQA MathVista avg. avg. Commercial VLMs (For reference)

GPT-5 85.92 94.84 90.91 49.34 60.95 76.39 99.90 86.10 72.90 26.00 87.20 80.20 75.38 75.84 GPT-5-mini 82.80 91.01 88.01 46.02 61.35 73.84 98.57 85.85 70.95 27.15 87.73 79.30 74.93 74.43 GPT-o4-mini 85.24 90.02 89.17 40.72 62.25 73.48 96.73 84.38 73.35 27.35 86.53 78.00 74.39 73.98 GPT-o3 85.63 92.35 90.14 50.27 54.25 74.53 99.18 79.58 71.85 25.55 81.87 75.40 72.24 73.28 Gemini-2.5-Pro 83.64 94.34 86.85 60.34 64.20 77.87 99.90 90.28 77.40 25.70 96.93 82.00 78.70 78.33 Gemini-2.5-Flash 85.85 92.85 88.78 46.15 53.35 73.40 97.96 84.50 68.75 23.45 84.73 77.90 72.88 73.12 Claude-4.5-Sonnet 85.03 85.19 92.26 55.57 91.85 81.98 99.49 80.44 82.95 23.95 94.50 75.10 76.07 78.76

Base Size: < 7B parameters

InternVL3-2B 24.08 40.26 40.21 21.88 16.35 28.56 82.41 63.59 38.94 17.90 60.13 32.70 49.28 39.86 VISTA-R1 (InternVL3-2B) 88.55 56.24 68.54 42.31 33.35 57.80 85.35 73.80 63.85 39.81 84.13 60.00 67.82 63.27

w/o Tools 72.88 43.93 66.30 39.52 39.45 52.42 74.23 64.70 35.90 15.93 63.53 49.60 50.65 51.45 w/o Reasoning 43.76 43.43 54.93 23.47 25.05 38.13 85.27 51.29 30.85 21.24 68.53 28.30 47.58 43.28

Qwen2.5-VL-3B 40.08 38.09 34.40 19.00 18.00 29.91 82.05 64.69 40.02 23.07 41.00 34.60 47.57 39.51 LLaVA-OneVision-1.5-4B 61.60 50.92 29.79 14.06 27.25 36.72 93.15 73.68 43.90 24.00 84.80 44.50 60.67 49.79

Large Size: 7 - 13B parameters

Qwen2.5-VL-7B 76.40 39.43 41.39 30.64 26.30 42.83 80.64 70.44 65.53 25.18 61.45 59.20 60.41 52.42 VISTA-R1 (Qwen2.5-VL-7B) 90.08 53.27 71.32 44.43 66.85 65.19 86.40 72.68 74.82 33.37 72.51 63.00 67.13 66.25

w/o Tools 83.32 48.92 68.11 33.42 51.35 57.02 64.13 56.09 76.45 26.92 68.58 56.10 58.05 57.58 w/o Reasoning 79.68 47.92 58.61 34.88 26.70 49.56 82.11 72.09 66.90 17.93 82.87 42.50 60.73 55.65

VTool-R1-7B 80.70† 62.06 65.18 33.29 19.65 52.18 64.21 79.34 48.85 19.70 75.67 36.60 54.06 53.20 R1-VL-7B 83.90† 58.90 62.09 37.93 23.87 57.34 92.64 76.51 48.50 18.00 87.27 63.50† 65.20 61.63 R1-Onevision-7B 59.92 55.79 57.47 28.30 33.20 46.94 84.19 61.75 35.40 19.30 79.30 64.10 57.34 52.61 Perception-R1-7B 76.41 48.59 51.23 21.22 40.52 47.59 84.05 67.77 44.30 18.00 81.87 74.20† 61.70 55.29 InternVL3-8B 77.32 47.09 44.87 33.71 34.70 47.54 95.32 74.54 63.05 28.95 73.27 59.60 65.79 57.49 VISTA-R1 (InternVL3-8B) 91.92 61.27 76.41 49.67 68.45 69.54 96.52 79.86 66.38 38.20 91.09 62.80 72.48 71.14

w/o Tool 84.24 51.47 73.10 34.70 53.50 59.40 94.68 75.37 60.25 29.24 80.93 62.80 67.21 63.66 w/o Reasoning 68.56 35.64 55.90 27.06 26.45 42.72 92.34 50.45 38.70 28.18 80.07 29.10 53.14 48.40

LLaVA-OneVision-1.5-8B 64.96 54.08 32.69 16.05 32.40 40.04 94.27 77.37 46.60 25.00 89.67 47.70 63.44 52.80 XL Size: > 13B parameters

InternVL3-14B 87.30 68.89 75.05 41.78 50.77 64.76 97.71 75.40 74.12 32.83 81.47 71.67 72.20 68.82 VISTA-R1 (InternVL3-14B) 93.60 83.87 78.85 52.94 70.50 75.95 98.02 87.60 76.00 40.86 92.13 68.00 77.10 76.58 Qwen2.5-VL-32B 87.30 65.22 50.29 42.94 89.20 66.99 94.79 82.16 71.17 35.14 92.60 68.20 74.01 70.82 InternVL3-38B 89.20 70.72 79.11 45.14 55.13 67.86 98.35 83.03 78.48 33.88 83.73 74.03 75.25 71.89 InternVL3-78B 89.70 78.76 85.43 53.40 58.15 73.09 99.39 85.85 82.35 25.15 86.33 78.80 76.31 74.85

memory. See Appendix D and G for prompt templates and other implementation details, respectively.

#### 6.2. Main Experiment Results

- Table 2 shows the main results of VISTA-R1 trained in VISTA-Gym against baselines on eleven visual-reasoning datasets. From the results, we have the following key observations: (i) VISTA-R1 achieves strong performance over baselines. Notably, VISTA-R1-8B outperforms baselines with similar sizes by 9.51%-18.72% with tools and 2.03%-11.24% even without tools, respectively. (ii) RL is critical for boosting TIR in VLMs. Simply augmenting VLMs with tools without explicit reasoning supervision degrades accuracy. By contrast, RL delivers substantial gains, indicating that base checkpoints do not exhibit robust TIR ability and that RL is crucial for unlocking tool-use capabilities in visual reasoning. Moreover, RL confers strong generalization: for example, VISTA-R1-8B attains accuracy on out-of-distribution VQA benchmarks comparable to substantially larger proprietary models (e.g., GPT-o3 and Claude-4.5-Sonnet). (iii) VISTA-R1 has a strong parameter efficiency. VISTA-R1-2B achieves competitive or even better performance with 8B baselines, while VISTA-R1-8B performs comparably to baselines with 38B

parameters. The superiority of VISTA-R1 demonstrates that VISTA-Gym provides a scalable training ground for tool-integrated agentic RL, enabling robust visual reasoning in open-source VLMs.

#### 6.3. Ablation Studies of VISTA-R1

Effect of Tool-Integrated Reasoning. Table 2 also disentangles the roles of reasoning and tool use by reporting three variants: (1) w/o Tools (63.66%) that enables reasoning but disables tool access; (2) w/o Reasoning (48.40%) that exposes tools (with prior tool selection knowledge) without reinforcing reasoning; (3) and the full VISTA-R1 (71.14%) that interleaves both. While strengthening reasoning or tool-use alone sometimes yields measurable gains, it remains suboptimal compared to VISTA-R1, which synergistically couples reasoning with tool execution. Two key findings follow: (i) simply exposing tools can be detrimental without sufficient reasoning to navigate the action space, and (ii) optimal problem solving arises when reasoning guides (if/when/which) tool invocation in an interleaved loop, rather than relying on either capability in isolation.

Effect of Training Stage. Figure 4a shows that SFT serves as a critical warm-up, aligning the model with tool-use formats and syntax (+3.46%); RL then unlocks substantially

100

Reward1

Reward2

Reward3

Ours

SFT

PPO

DAPO

GRPO

Base

Base w/ tool

Two-Stage FT

| |
|---|

| |
|---|

| |
|---|

75

| |
|---|

| |
|---|

| |
|---|

70

| |
|---|

| |
|---|

80

SFT Only

GRPO Only

70

65

Acc(%)

Acc(%)

60

Acc(%)

65

60

40

60

55

20

55

50

0

50

In-Distribution Out-of-Distribution Overall

In-Distribution Out-of-Distribution Overall

In-Distribution Out-of-Distribution Overall

(a) Effect of training stages

(b) Effect of RL algorithms (E=100)

###### (c) Effect of reward design

100

100

GPT-5 Qwen2.5VL-72B Qwen3-235B

75

SFT

GRPO w/ all tools

80

80

GRPO

GRPO w/ ChartMoE only

70

GRPO w/ ChartQA-only

Acc(%)

60

60

Acc(%)

Acc(%)

GRPO w/ UniGeo-only GRPO w/ MapQA-only

65

40

40

60

20

20

55

0

0

50

ChartQA UniGeo MapQA

ChartQA UniGeo MapQA

In-Distribution Out-of-Distribution Overall

(d) Effect of data diversity

(e) Effect of tool diversity (E=50)

(f) Effect of thinking trajectory quality

Figure 4. Ablation studies and diversity analysis with InternVL3-8B as backbone VLM.

- Table 3. Effect of reasoning and tool-use results without RL training. w/ Tools refers to directly augmenting VLMs with tools significantly degrades accuracy; w/ Reasoning refers to CoT reasoning without tools. w/ T&R refers to a different setting compared to TIR, only supplying tool-selection prior knowledge and interleaving reasoning with tool execution improve performance.

Baselines (↓) In-Distribution Out-of-Distribution All Datasets (→) ChartQA Geometry3K GeoQA UniGeo MapQA avg. TABMWP AI2D PlotQA CLEVR-Math IconQA MathVista avg. avg. GPT-5 85.92 94.84 90.91 49.34 60.95 76.39 99.90 86.10 72.90 26.00 87.20 80.20 75.38 75.84

w/ Tools 58.60 74.04 65.00 28.78 31.20 51.52 69.83 65.07 46.75 18.96 60.53 42.60 50.62 51.03 w/ Reasoning 85.68 91.01 86.27 51.25 59.48 74.74 99.81 83.03 84.73 25.50 86.27 80.90 76.71 75.81 w/ T&R 94.00 93.84 91.56 51.69 57.35 77.69 98.92 84.62 79.45 27.79 91.07 57.80 73.28 75.28

InternVL3-8B 77.32 47.09 44.87 33.71 34.70 47.54 95.32 74.54 63.05 28.95 73.27 59.60 65.79 57.49 w/ Tools 25.44 17.06 27.27 26.90 16.22 22.58 58.15 51.74 58.31 18.01 25.86 22.70 39.13 31.61 w/ Reasoning 81.28 47.09 45.07 35.81 31.10 48.07 94.38 76.51 65.85 24.15 75.53 60.70 66.19 57.95 w/ T&R 68.56 35.64 55.90 27.06 26.45 42.72 92.34 50.45 38.70 28.18 80.07 29.10 53.14 48.40

larger gains (+10.19% over SFT). Both stages contribute: SFT establishes the necessary behavioral priors for reliable tool interaction, while RL triggers deeper, interleaved reasoning required to navigate complex tasks effectively.

Effect of RL Algorithm. In addition to GRPO, we evaluate PPO [57] and DAPO [87] as preference-optimization algorithms; yet we observe that GRPO delivers the most robust performance, as shown in Figure 4b. The DAPO deficit may be due to the elimination of uniform-reward groups in its preference objective: in early stage, removing all-incorrect collapses the effective batch and starves the policy of learning signal; in later stage, removing all-correct erases useful supervision for reasoning–tool coordination. In contrast, GRPO retains all rollouts and uses group-normalized advantages to deliver low-variance, difficulty-adaptive credit assignment, yielding more robust gains

Effect of Reward Design. Figure 4c compares different types of reward design, including (1) dense reward Rdense = −1 + 0.5Rformat + 0.5Rcorrect + Rformat · Rcorrect, (2) sparse reward Rsparse = Rformat · Rcorrect, (3) difficulty-adaptive reward Rdiff = R · w, where w = clamp(2 − D,0,1) × 0.5 + 0.5 and D represents the difficulty measured by the mean base reward across all rollouts, D = E[R], and (4) ours. In comparison, our reward design couples a verifiable,

low-noise binary end signal with difficulty-adaptive scaling via group-normalized advantages, yielding superior accuracy while avoiding overfitting to easy cases or exploitation of intermediate signals.

Effect of Tool and Reasoning in Vanilla VLMs. Table 3 summarizes the effect of tools and reasoning on vanilla VLMs for open- and closed-source models. Directly augmenting VLMs with tools significantly degrades accuracy, and intrinsic chain-of-thought alone yields only modest gains on complex VQA. Supplying tool-selection priors improves performance; however, gains are strongly taskdependent for commercial VLMs, and smaller open-source VLMs remain particularly challenged. These observations underscore that it is both important and non-trivial to endow VLMs with robust, generalizable tool-use capabilities that adapt across diverse visual reasoning scenarios.

#### 6.4. Additional Studies of VISTA-Gym

Data Diversity: Effect of Data Mixture. Figure 4d examines how task composition during RL shapes TIR. Single-task training exhibits weak cross-task transfer, especially across distinct domains such as ChartQA (chart understanding) and UniGeo (geometric reasoning), likely due to shifts in tool-use schemas and argument distributions that

GPT-5

Qwen2.5VL-72B

Qwen3VL-235B

Average Length across Expert Models

| |
|---|

| |
|---|

| |
|---|

###### 5k Geometry

Chart

E6: Incorrect Reasoning from Tool

###### Geospatial

###### Scientific

5k

5k

5k

#oftakens

3k

3k

3k

3k

Solved

1k

1k

1k

1k

- E1: Invocation schema violation

- E2: Invalid argument name

- E3: Invalid argument value

- E4: Incorrect argument value

Document

###### 3DSpatial

###### Others

Avg. across Tasks

5k

5k

5k

5k

#oftakens

3k

3k

3k

3k

E6: Incorrect Reasoning from Tool

1k

1k

1k

1k

Unsolved

Figure 5. Quality of expert thinking trajectories by length.

###### Figure 6. Error analysis.

100

In-Distribution

ChartQA

UniGeo MapQA Avg.

Out-of-Distribution

Geometry3K

90

70

Overall

GeoQA

80

Acc(%)

Acc(%)

| |
|---|

65

70

60

60

50

55

40

10 30 50 70

100 200 300

Training Steps

Training Steps

(a) Training scaling

(b) Tailpatch

Figure 7. Training scaling from easy to hard.

narrow the learned policy’s coverage of the action space and impede general TIR. By contrast, multi-task training consistently improves generalization, indicating that exposure to a diverse task mixture during training is key to learning transferable TIR policies.

Tool Diversity: Effect of Tool Mixture. Figure 4e analyzes how tool composition during RL shapes TIR using ChartMoE as an example. Training exclusively with chartunderstanding tools leaves chart-centric tasks (ChartQA; to a lesser extent, MapQA) largely unchanged, but transfers poorly to tasks with different tool affordances (UniGeo). Similar to data mixture, exposure to heterogeneous tool schemas and argument formats also mitigates over-specialization and broadens the learned action policy.

Reasoning Trajectory Quality. We assess the quality of expert trajectories (GPT-5, Qwen2.5VL-72B, Qwen3VL235B) using token length as a simple proxy in Figure 5. As expected, longer trajectories during the SFT warm-up establish better behavior prior, thus contribute to higher performance after RL (Figure 4f). Note that commercial APIs such as GPT-5 often do not expose full intermediate reasoning, yielding significantly shorter trajectories and weaker gains from agentic RL under the same training recipe.

Tail-Patch Curriculum. To overcome the performance plateau, we adopt an easy-to-hard curriculum that tail-patch training with increasingly challenging samples. We use rollout pass rates from the previous checkpoint as a dynamic difficulty signal and isolate hard-but-learnable instances

- 5.0

Score

InternVL3-2B

| |
|---|

InternVL3-8B

| |
|---|

InternVL3-14B

| |
|---|

Qwen2.5VL-7B

Figure 8. Human study on trajectory quality.

with pass rates in [0.125,0.375]. Continuing RL exclusively on this high-difficulty slice pushes performance beyond the initial convergence point (69.54% → 71.27%) (Figure 7). This suggests a general principle: focusing the optimization budget on the frontier of competence—cases where the model is uncertain yet occasionally succeeds—sharpens long-horizon reasoning and delivers fine-grained gains.

- 6.5. Error Analysis and Case Study

4.5

4.0

3.5

3.0

Base SFT SFT+GRPO

Error Analysis. We re-evaluate error patterns on the same 500 samples from Table 1 after InternVL3-8B (left in Figure 6) training with VISTA-Gym (right). Agentic training within VISTA-Gym markedly resolves the majority of toolcall and reasoning failures observed in base models.

Human Study for Trajectory Quality. As TIR trajectories are a key component of VISTA-Gym, we evaluate the quality of the generated interleaved tool-use and reasoning steps. Figure 8 reports average human ratings (1–5 scale) over 40 randomly sampled solution trajectory per dataset. We observe that both training stages significantly improve TIR across tasks and model sizes.

Case Studies. Table 4 presents four case studies illustrating the capabilities acquired through agentic training in VISTAGym: (i) schema- and argument-correct tool invocation; (ii) post-tool reasoning that grounds the final answer in tool outputs; (iii) cross-tool generalization that repurposes atypical yet useful tools (e.g., applying ChartMoE on geometry problems); and (iv) robustness to imperfect tool outputs, where the agent identifies and exploits partial signals over multiple turns to complete multi-step reasoning.

Table 4. Case studies on geometric reasoning tasks with VISTA-R1-8B trained in VISTA-Gym.

- Case Study 1: Find m ∠ R. Answer: A: 58

[Figure 30]

InternVL3-8B-Tool: <tool_call>{"tool": "multimath", "task": "solve_the_problem", "arguments": {"image_url": <image_url>, "data": {"image/png": base64, iVBORw0KGgoAAAANSUhEUgAAAOAAAAHCAIAAADYXJpAAAAA3NCSV ··· (including long base64 image data)}}</tool_call> Tool Call Error: Invalid Arguments. (✘)

VISTA-R1: <tool_call>{"tool": "gllava", "task": "solve", "arguments": "Find m ∠R from the diagram: a diamond-shaped quadrilateral with vertices R (left), Q (top), T (right), S (bottom). Angles labeled: at R = x◦, at T = x◦, at Q = (2x+5)◦, at S = (2x+7)◦."}.</tool_call> Tool Call: Correct (✓)

- Case Study 2: As shown in the figure, in ⊙O, AB is a diameter and CD is a chord. AB ⊥ CD, with the foot of the perpendicular at point E. Connect CO and AD. If ∠BOC = 30◦, then the degree measure of ∠BAD is ( ). Answer: 15◦

[Figure 31]

InternVL3-8B-Tool: ···```output ···Step 12 (Property of isosceles triangle): ∠BAD = ∠OAD. Answer: ∠BAD = 15◦.```<think> Alright, let’s tackle this geometry problem step by step. ··· But according to the tool output, the answer is 15 degrees, which is option D. But my computation also says the answer is 52.5 degrees. So the answer is closer to C. </think>··· Correct Tool, Wrong Reasoning: Incorrect reasoning to answer.. (✘)

VISTA-R1: ··· ```output ···Step 12 (Property of isosceles triangle): ∠BAD = ∠OAD. Answer: ∠BAD = 15◦.```<think> ···Based on the previous verification and tool output, I can analyze and provide the answer of 15◦.</think> <answer>15◦</answer>. Reasoning: Correct (✓)

- Case Study 3: AB||CD. Line EF intersects AB at point E and intersects CD at point F. EG bisects ∠BEF and intersects CD at point G. If ∠1 = 50◦, then ∠2 equals ( ). Choices:A: 50◦B: 60◦C: 65◦D: 90◦. Answer: C: 90◦.

[Figure 32]

VISTA-R1: <think> ··· ChartMoE is appropriate because it is designed to analyze structured geometric diagrams, accurately reading angle labels, identifying parallel-line configurations, and aligning textual annotations with diagram elements. I will have it parse the diagram to confirm AB||CD, locate the transversal EF and the points E, F, G, and verify that EG bisects ∠BEF while intersecting CD at G. By interpreting the angles formed at E relative to BE and FE and using the parallel-line angle relations (corresponding, alternate interior, and consecutive interior angles), ChartMoE will relate ∠1 to the angle between BE and EG, propagate the bisector relation to ∠BEF, and then deduce the measure of ∠2 at the intersection of EF and CD from the resulting angle measures. Finally, it will align that computed value with the provided choices. Therefore, I will call chartmoe now.</think><tool_call>{"tool": "chartmoe", "task": "analyze"}</tool_call> ··· New Tool Selection: Expected G-LLaVA but got ChartMoE. (✓)

- Case Study 4: Find X Z. Answer: D: 34.

InternVL3-8B-Tool: <tool_call>{"tool": "multimath", "task": "Solve the problem", "arguments": {"image": "<image>", "question": "Find X Z. ···"}}</tool_call>```output ··· Step 5 (Simplify the equation): XZ2 = 29y2 + 12y + 360.\nAnswer: \boxed{29y2 + 12y + 360}```<think> ··· But as we saw earlier, this leads to a quadratic with no real solutions.\n Wait, maybe I made a mistake in the tool output. Let me double-check the steps: \n Step 3: XZ2 = (5y − 6)2 + (18 + 2y)2 = 25y2 − 60y + 36 + 324 + 72y + 4y2 = 29y2 + 12y + 360. That seems correct.\n ··· Step 5: Simplify the equation: XZ2 = 29y2 + 12</think><answer> \boxed{C}.</answer> Tool Output Error: Incorrect reasoning to answer.. (✘)

InternVL3-8B-Reasoning: <think> Alright, let’s tackle this problem step by step. First, I notice that triangle WYZ is a right triangle because it has a right angle at W. ··· Using the Pythagorean theorem again:\nXZ2 = WZ2 + XY 2\nXZ2 = 342 + 342, XZ2 = 2312, XZ = √2312 = 34 ∗

√2\n Therefore, the length of XZ is 34√2.\n</think>\n<answer>34√2</answer>" No Tool-use: Insufficient capability to answer.. (✘)

[Figure 33]

VISTA-R1: <think> ··· Multimath is appropriate because ··· I will read from the image that WX ⊥ ZX ···. Multimath will translate these visual relations into algebraic constraints, ···. </think><tool_call>{"tool": "multimath", "task": "solve"}</tool_call>```output ··· Step 10 (Calculate the value of XZ): XZ = 18 + 2y = 18 + 2(65) = 18 + 125 = 905 + 125 = 1025 = 20.4. ```<think> Looking at the tool’s solution, it set up the equation 25y2 − 60y + 36 = 0 and solved to get y = 6/5. Let me verify this step by step. ··· Wait, let me check if this makes sense geometrically. ··· Actually, if the tick marks indicate equal segments, shouldn’t we have 5y − 6 = 18 + 2y? Let me solve this: 5y − 6 = 18 + 2y3y = 24y = 8 With y = 8, let’s calculate XZ. If XZ = 18 + 2y, then: XZ = 18 + 2(8) = 18 + 16 = 34. Hmm, that’s different from the tool’s answer of 20.4. Let me reconsider. ··· The tool calculated 20.4, but based on the equal segment marks, the answer is 34.</think><answer>34</answer>" TIR: correct (✓)

### 7. Conclusion

In this work, we introduce VISTA-Gym, a scalable training ground for tool-integrated, visual-centric agentic RL in VLMs, and VISTA-R1, an agent trained to interleave multi-turn reasoning with structured tool use. Practitioners leveraging VISTA-Gym for boosting TIR in VLMs may consider: (i) train policies that interleave reasoning with tools; (ii) warm up with SFT for tool call syntax/format, then apply online RL for in-depth tool-integrated visual reasoning; (iii) diversify tasks and tools to expand action-space coverage; and (iv) allocate additional RL budget to hard-but-learnable slices via tail-patching. Our verifiers emphasize terminal correctness and structural validity; richer stepwise semantics and broader tool ecosystems may further benefit long-horizon TIR. VISTA-Gym provides a scalable agentic training environment—unified interfaces, executable feedback, and efficient logging—for systematic progress in thinking with images.

### References

- [1] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. 5, 3
- [2] Team Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/claude-sonnet-4-5, 2025. 5
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 5, 4
- [4] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010. 3, 1

- [5] Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao Wang, Akshay Malik, Graham Neubig, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Skyrl-v0: Train real-world long-horizon agents via reinforcement learning, 2025. 2
- [6] Shuaichen Chang, David Palzer, Jialin Li, Eric FoslerLussier, and Ningchuan Xiao. MapQA: A dataset for question answering on choropleth maps. In NeurIPS 2022 First Table Representation Workshop, 2022. 3, 5, 1
- [7] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 513–523,

2021. 3, 5, 1

- [8] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3313– 3323, 2022. 3, 5, 1
- [9] Liang Chen, Hongcheng Gao, Tianyu Liu, Zhiqi Huang, Flood Sung, Xinyu Zhou, Yuxin Wu, and Baobao Chang. G1: Bootstrapping perception and reasoning abilities of vision-language model via reinforcement learning. arXiv preprint arXiv:2505.13426, 2025. 1, 2
- [10] De Chezelles, Thibault Le Sellier, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F Xu, Siva Reddy, Quentin Cappart, et al. The browsergym ecosystem for web agent research. arXiv preprint arXiv:2412.05467, 2024. 2
- [11] Krishna Teja Chitty-Venkata and Murali Emani. Imagenetthink-250k: A large-scale synthetic dataset for multimodal reasoning for vision language models. arXiv preprint arXiv:2510.01582, 2025. 3
- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 5
- [13] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: Complex visionlanguage reasoning via iterative sft-rl cycles, 2025. 1
- [14] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9062–9072, 2025. 1
- [15] Weihua Du, Hailei Gong, Zhan Ling, Kang Liu, Lingfeng Shen, Xuesong Yao, Yufei Xu, Dingyuan Shi, Yiming Yang, and Jiecao Chen. Generalizable end-to-end tool-use rl with synthetic codegym. arXiv preprint arXiv:2509.17325, 2025. 2
- [16] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze

- Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879, 2025. 1, 2
- [17] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing HONG, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-LLaVA: Solving geometric problem with multi-modal large language model. In The Thirteenth International Conference on Learning Representations, 2025. 3, 4
- [18] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633– 638, 2025. 1, 2
- [19] Xingang Guo, Utkarsh Tyagi, Advait Gosai, Paula Vergara, Ernesto Gabriel Hernández Montoya, Chen Bo Calvin Zhang, Bin Hu, Yunzhong He, Bing Liu, and Rakshith Sharma Srinivasa. Beyond seeing: Evaluating multimodal llms on tool-enabled image perception, transformation, and reasoning. arXiv preprint arXiv:2510.12712, 2025. 1
- [20] Zixian Guo, Ming Liu, Qilong Wang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Integrating visual interpretation and linguistic reasoning for geometric problem solving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3988–3998, 2025. 1
- [21] Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, and Shanghang Zhang. Tiger: Tool-integrated geometric reasoning in vision-language models for robotics. arXiv preprint arXiv:2510.07181, 2025. 2, 5
- [22] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 1, 2, 5

- [23] Naman Jain, Jaskirat Singh, Manish Shetty, Tianjun Zhang, Liang Zheng, Koushik Sen, and Ion Stoica. R2e-gym: Procedural environment generation and hybrid verifiers for scaling open-weights SWE agents. In Second Conference on Language Modeling, 2025. 2
- [24] Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, et al. Verltool: Towards holistic agentic reinforcement learning with tool use. arXiv preprint arXiv:2509.01055, 2025. 5
- [25] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 3, 1
- [26] Samira Ebrahimi Kahou, Adam Atkinson, Vincent Michalski, Ákos Kádár, Adam Trischler, and Yoshua Bengio. FigureQA: An annotated figure dataset for visual reasoning,

2018. 3, 1

- [27] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016. 5, 1

- [28] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3
- [29] Ming Li, Jike Zhong, Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Yuxiang Lai, Wei Chen, Konstantinos Psounis, and Kaipeng Zhang. Tir-bench: A comprehensive benchmark for agentic thinking-with-images reasoning, 2025. 2
- [30] Yuhang Li, Chenchen Zhang, Ruilin Lv, Ao Liu, Ken Deng, Yuanxing Zhang, Jiaheng Liu, Wiggin Zhou, and Bo Zhou. Relook: Vision-grounded rl with a multimodal llm critic for agentic web coding. arXiv preprint arXiv:2510.11498, 2025. 2
- [31] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding visionlanguage model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025. 2, 5
- [32] Yiqing Liang, Jielin Qiu, Wenhao Ding, Zuxin Liu, James Tompkin, Mengdi Xu, Mengzhou Xia, Zhengzhong Tu, Laixi Shi, and Jiacheng Zhu. Modomodo: Multi-domain data mixtures for multimodal llm reinforcement learning. arXiv preprint arXiv:2505.24871, 2025. 1
- [33] Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358, 2022. 5, 1
- [34] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 3
- [35] Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, Bohao Peng, Shu Liu, Bei Yu, and Jiaya Jia. Visionreasoner: Unified visual perception and reasoning via reinforcement learning. arXiv preprint arXiv:2505.12081, 2025. 2, 5
- [36] Zuyan Liu, Yuhao Dong, Yongming Rao, Jie Zhou, and Jiwen Lu. Chain-of-spot: Interactive reasoning improves large vision-language models. arXiv preprint arXiv:2403.12966,

2024. 1

- [37] Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, et al. Gem: A gym for agentic llms. arXiv preprint arXiv:2510.01051, 2025. 2
- [38] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 1, 2
- [39] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774–6786,

2021. 3, 5, 1

- [40] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 5, 1
- [41] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521,

2022. 3, 1

- [42] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, SongChun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022. 5, 1
- [43] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 5, 1
- [44] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, 2022. 3, 5, 1
- [45] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 3, 1
- [46] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 3, 1
- [47] Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In Proceedings of the ieee/cvf winter conference on applications of computer vision, pages 1527–1536, 2020. 5, 1
- [48] Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, and Zhe Gan. Deepmmsearch-r1: Empowering multimodal llms in multimodal web search. arXiv preprint arXiv:2510.12801, 2025. 2
- [49] Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, et al. Mlgym: A new framework and benchmark for advancing ai research agents. arXiv preprint arXiv:2502.14499,

2025. 2

- [50] Team OpenAI. Introducing gpt-5. https://openai.com/index/introducing-gpt-5/, 2025. 5
- [51] Team OpenAI. Introducing openai o3 and o4-mini. https://openai. com/index/introducing-o3-and-o4-mini/,

2025. 5

- [52] Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with SWE-gym. In

- Forty-second International Conference on Machine Learning, 2025. 2
- [53] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024. 3
- [54] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. Cogcom: A visual language model with chain-ofmanipulations reasoning. arXiv preprint arXiv:2402.04236,

2024. 1

- [55] Rushi Qiang, Yuchen Zhuang, Yinghao Li, Rongzhi Zhang, Changhao Li, Ian Shu-Hei Wong, Sherry Yang, Percy Liang, Chao Zhang, Bo Dai, et al. Mle-dojo: Interactive environments for empowering llm agents in machine learning engineering. arXiv preprint arXiv:2505.07782, 2025. 2
- [56] Rushi Qiang, Yuchen Zhuang, Anikait Singh, Percy Liang, Chao Zhang, Sherry Yang, and Bo Dai. Mle-smith: Scaling mle tasks with automated multi-agent pipeline. arXiv preprint arXiv:2510.07307, 2025. 2
- [57] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 7
- [58] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022. 3, 1
- [59] Yijia Shao, Vinay Samuel, Yucheng Jiang, John Yang, and Diyi Yang. Collaborative gym: A framework for enabling and evaluating human-agent collaboration. arXiv preprint arXiv:2412.15701, 2024. 2
- [60] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 4
- [61] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 2
- [62] Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6613–6629, 2025. 1
- [63] Xiaoqian Shen, Wenxuan Zhang, Jun Chen, and Mohamed Elhoseiny. Vgent: Graph-based retrieval-reasoningaugmented generation for long video understanding. arXiv preprint arXiv:2510.14032, 2025. 1
- [64] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Mathllava: Bootstrapping mathematical reasoning for multimodal large language models, 2024. 1
- [65] Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, and Andreas

- Köpf. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2505.24760, 2025. 2
- [66] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025. 1, 2
- [67] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 1
- [68] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamavo1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025. 1
- [69] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024. 1
- [70] Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, et al. Vagen: Reinforcing world model reasoning for multi-turn vlm agents. arXiv preprint arXiv:2510.16907, 2025. 1, 2
- [71] Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, 2022. 2
- [72] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. Ragen: Understanding selfevolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025. 2
- [73] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022. 1
- [74] Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei Liu. Mmsearch-r1: Incentivizing lmms to search. arXiv preprint arXiv:2506.20670, 2025. 1, 2
- [75] Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255, 2025. 5, 3
- [76] Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, Rui Zheng, Junjie Ye, Jiazheng Zhang, Wenxiang Chen, et al. Agentgym-rl: Training llm agents for long-horizon decision making through multi-turn reinforcement learning. arXiv preprint arXiv:2509.08755,

2025. 2

- [77] Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang, Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing Song, Dengyu

- Wang, Minjia Zhang, et al. Rag-gym: Optimizing reasoning and search agents with process supervision. arXiv preprint arXiv:2502.13957, 2025. 2
- [78] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2087– 2098, 2025. 2
- [79] Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, and Hongkun Yu. Incentivizing agentic reasoning in llm judges via tool-integrated reinforcement learning. arXiv preprint arXiv:2510.23038, 2025. 1
- [80] Ran Xu, Yuchen Zhuang, Zihan Dong, Jonathan Wang, Yue Yu, Joyce C Ho, Linjun Zhang, Haoyu Wang, Wenqi Shi, and Carl Yang. Acesearcher: Bootstrapping reasoning and search for llms via reinforced self-play. arXiv preprint arXiv:2509.24193, 2025. 2
- [81] Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu, May D Wang, Peifeng Ruan, Donghan Yang, Tao Wang, et al. Medagentgym: Training llm agents for code-based medical reasoning at scale. arXiv preprint arXiv:2506.04405, 2025. 2
- [82] Zhengzhuo Xu, Bowen Qu, Yiyan Qi, SiNan Du, Chengjin Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of diversely aligned expert connector for chart understanding. In The Thirteenth International Conference on Learning Representations, 2025. 3, 4
- [83] John Yang, Kilian Lieret, Carlos E Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. Swe-smith: Scaling data for software engineering agents. arXiv preprint arXiv:2504.21798, 2025. 2
- [84] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 5, 3
- [85] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. 3
- [86] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954,

2025. 5, 3

- [87] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 7
- [88] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025. 5, 3

- [89] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025. 1, 2
- [90] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 1, 2
- [91] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 5, 3

## Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs Supplementary Material

### A. Task and Data Information

Table 5 presents the statistics of the in-distribution (ID) and out-of-distribution (OOD) datasets; the additional details are as follows:

#### A.1. Information for In-Distribution Datasets

We curate a diverse suite of training tasks integrated into VISTA-Gym, spanning fifteen datasets across eight distinct reasoning domains:

- • Chart Understanding: FigureQA [26] and ChartQA [44] demand precise quantitative extraction and logical inference over diverse data visualizations, requiring the agent to align visual features (bars, lines) with numerical values.
- • Geometric Reasoning: Geometry3K [39], GeoQA [7], and UniGeo [8] necessitate grounding symbolic theorem application in diagrammatic parsing to solve multi-step spatial and quantitative problems.
- • Mathematical Reasoning: MathVision [69] and MathV360 [64] evaluate the interpretation of complex mathematical diagrams and symbolic expressions, requiring the synthesis of visual perception with algebraic deduction.
- • Geospatial Reasoning: MapQA [6] and InfographicVQA [46] challenge agents to perform rigorous spatial lookups, legend-symbol alignment, and information retrieval across dense graphical layouts.
- • Visual Scientific Reasoning: ThinkVL [13], VizWiz [4], and ScienceQA [41] encompass a broad spectrum of tasks ranging from accessibility-focused visual description to domain-specific multimodal scientific inquiry.
- • Document Understanding: DocVQA [45] tests layoutaware information extraction from unstructured documents, including financial reports, forms, and scanned articles.
- • Spatial Reasoning: CLEVR [25] provides a diagnostic environment for compositional logic, assessing the agent’s ability to execute complex reasoning chains regarding attribute recognition, spatial relations, and counting in 3D scenes.
- • General Visual Reasoning (Others): A-OKVQA [58] requires the retrieval of external world knowledge and commonsense reasoning to address open-ended visual questions.

#### A.2. Information for Out-of-Distribution Datasets

To assess generalization, we evaluate on six additional benchmarks featuring distinct reasoning:

Table 5. Dataset statistics in VISTA-Gym.

Dataset Type # of training instances # of testing instances

FigureQA ID 2000 – ChartQA ID 1500 1250 Geometry3k ID 2000 600 GeoQA ID 2000 500 UniGeo ID 2000 750 MathVision ID 2000 – MathV360 ID 1500 – MapQA ID 1500 2000 InfographicVQA ID 1000 – ThinkVL ID 2000 – VizWiz ID 200 – ScienceQA ID 1000 – DocVQA ID 970 – CLEVR ID 1000 – A-OKVQA ID 1500 –

TABMWP OOD – 970 AI2D OOD – 810 PlotQA OOD – 2000 Clever-math OOD – 930 IconQA OOD – 1500 Mathvista OOD – 1000

Total – 22,170 12,310

- • TABMWP: TABMWP [42] focuses on table-based math word problems requiring cell selection, row/column aggregation, and arithmetic reasoning grounded in semistructured text, testing symbolic computation beyond visual chart parsing.
- • AI2D: AI2D [27] is a diagram-centric science QA benchmark involving annotated figures (labels, arrows, parts); it stresses the joint interpretation of schematic layouts and textual cues via multi-hop reasoning.
- • PlotQA: PlotQA [47] centers on scientific plots (axes, legends, bars/lines) demanding high-precision value reading and aggregation, serving as a complementary but distributionally distinct test bed to ChartQA.
- • CLEVR-Math: CLEVR-Math [33] extends compositional reasoning in rendered 3D scenes with arithmetic operations, assessing program-like visual logic coupled with numeric computation.
- • IconQA: IconQA [40] comprises textbook-style diagram and icon questions (often multiple-choice) that emphasize abstract visual relations and symbolic understanding, diverging from natural image statistics.
- • MathVista: MathVista [43] is a holistic mathematical reasoning suite spanning real images, charts, and diagrams; it integrates OCR, quantitative reading, geometry, and multi-step deduction for comprehensive evaluation.

### B. Toolset Information

In VISTA-Gym, we enable access to 26 tools from four categories detailed as follows:

Perception. The perception layer provides low-level visual signals that supply structural cues for higher-level reasoning, including:

- • Detection: GroundingDINO is an open-set object detection model that unifies visual and textual modalities, allowing for the detection of arbitrary objects described by natural language. It performs open-set, text-conditioned object detection, localizing arbitrary entities referenced by natural-language queries.
- • Segmentation: SAM (Segment Anything Model) is a foundational model for promptable image segmentation that can zero-shot segment any object based on flexible inputs like points, boxes, or text. It offers promptable, zero-shot segmentation, isolating regions given points, boxes, or textual prompts.
- • OCR: EasyOCR is a versatile and ready-to-use optical character recognition tool supporting over 80 languages and various writing scripts for robust text extraction from images. It conducts multilingual optical character recognition, extracting text from images across diverse scripts and layouts.

Chart Understanding. The chart layer specializes in plotand table-centric inference. Chart understanding tool collections like ChartMoE use a Mixture-of-Experts connector to enhance visual–text alignment, supporting both structured data extraction and high-level analytical reasoning for chart question answering, including:

- • Chart To Table. The ChartToTable tool subset converts chart images into structured tabular data suitable for downstream numerical reasoning. UniChart and DePlot recover underlying data points by detecting axes, legends, and visual marks, while ChartMoE.to_table and ChartMoE.extract_data leverage the ChartMoE backbone to extract series-wise values and metadata (e.g., categories, units) with improved robustness across chart styles.
- • Chart To SVG. The subset of ChartToSVG tools reconstructs vectorized representations of charts. OpenCV provides low-level image processing to segment graphical primitives, and ChartDet localizes key chart components (axes, legends, bars, lines, markers); ChartOCR recognizes textual elements (titles, axis labels, tick labels, legends) and anchors them to detected regions, enabling SVG-style reconstruction.
- • Chart To SCG. The subset of ChartToSCG tools maps charts into structured scene graphs (SCG). ChartDet and ChartOCR are combined to instantiate nodes for visual and textual elements and edges for relations such as series membership, axis association, and legend–color bindings, yielding a graph representation amenable to symbolic rea-

- soning.
- • Captioning Color. The subset of CaptioningColor generates natural-language descriptions of charts with explicit grounding in visual encodings. ChartAssistant produces concise summaries of chart type, trends, and salient extrema, whereas ChartVLM provides more detailed captions that explicitly reference colors, legends, and value ranges.
- • QA Analysis. The QAAnalysis supports chart question answering and intermediate analysis. ChartMoE.answer directly predicts answers to chart-related questions, while ChartMoE.analyze performs intermediate computations (e.g., differences, ratios, aggregations) over extracted data, and ChartMoE.describe generates explanatory textual analyses that connect the visual structure, quantitative reasoning steps, and final answer.

Diagram Formalization. This layer converts visual diagrams into symbolic structures amenable to automated reasoning. DiagramFormalizer parses geometric and schematic diagrams into ConsCDL and ImgCDL-style formal languages, enabling symbolic representations for downstream deductive geometric reasoning.

- • CDL. The CDL tools convert geometric diagrams and problem statements into formal constraint languages suitable for symbolic solvers. image_cdl maps raw diagram images to ImgCDL-style primitives (points, lines, circles, incidences), whereas text_cdl parses textual problem descriptions into ConsCDL-style constraints and goals. construction_cdl recovers the underlying construction sequence of a diagram, and goal_cdl extracts explicit target predicates (e.g., lengths, angles, congruence relations), enabling theorem-driven reasoning in downstream geometry solvers.
- • Symbolic Parsing. The SymbolicParsing tools build fully symbolic problem representations by combining diagram parsing with formal language construction. Inter-GPS parses both the problem text and the associated diagram into a unified formal language and then applies theorembased symbolic reasoning step by step to solve geometry problems. DiagramFormalizer provides complementary parsing endpoints that expose ImgCDL/ConsCDL representations directly, supplying structured inputs for interpretable geometry problem solvers such as Inter-GPS.

Math Solvers. The solver layer performs domain-specific high-level reasoning on top of structured perceptual inputs:

- • Multimodal: G-LLaVA is a domain-specific model that integrates a CLIP-based vision encoder with DeepSeekMath-RL to bridge the gap between visual perception and complex mathematical reasoning. It executes geometry-aware multimodal reasoning by aligning figure interpretation with textual reasoning through specialized instruction tuning.
- • Math: MultiMath is a specialized multimodal model for

geometric problem solving, optimized through geometric cross-modal alignment and instruction tuning to accurately interpret figures and relationships. It bridges visual perception and symbolic mathematics by combining a CLIP-based vision encoder with DeepSeekMath-RL, tackling complex multimodal math problems end-to-end.

- C. Baseline Details

We include additional details of baseline tool-integrated or reasoning-augmented VLMs as follows:

- • VTool-R1 [75] integrates multimodal tool invocation directly into the VLM reasoning loop via reinforcement learning. By optimizing for feedback signals during problem-solving, it surpasses standard prompting and supervised fine-tuning baselines in complex visual reasoning tasks.
- • Perception-R1 [86] utilizes RL to learn active perception policies that optimize how VLMs attend to and interpret visual inputs. This approach moves beyond passive image encoding, fostering decision-oriented visual understanding through reward-driven refinement.
- • R1-VL [88] employs GRPO to enhance step-by-step visual reasoning. By incentivizing effective intermediate reasoning steps rather than solely focusing on outcomes, it aligns visual perception with logical inference throughout the generated trajectory.
- • R1-OneVision [84] establishes a framework for converting visual inputs into structured, cross-modal formal representations. By leveraging these symbolic intermediates during reinforcement learning, the model achieves generalized reasoning capabilities across heterogeneous task types.
- • LLaVA-OneVision [1] proposes a unified training paradigm that consolidates diverse visual tasks under a consistent interface. This design enables a single generalpurpose VLM to generalize across wide-ranging visual domains without requiring task-specific adaptation.

- D. Prompt Templates We include the prompt template in Figure 9.
- E. Additional Method Details of VISTA-Gym

Interface Details. VISTA-Gym follows the Gymnasium API with core methods reset and step. On reset, the environment samples a task x ∈ I (question + image) and returns the initial observation. At time step t, given the interaction history ct−1, the environment provides the current observation ot, and the agent produces a thought–action pair (gt,at). The environment executes at, transitions according to the POMDP formalization in Section 3, and yields the next observation ot+1, enabling multi-turn tool-integrated interaction.

##### System Prompt:

You are an advanced AI agent capable of complex reasoning and tool usage. You must strictly adhere to the following protocol for every interaction:

- 1. ALWAYS call the appropriate tool first;
- 2. NEVER provide answers without tool results;
- 3. Call appropriate tools based on the task; {tool_descriptions}
- 4. Reasoning Before Action: before selecting a tool, you must analyze the user’s request and determine the necessary steps. Output your internal monologue and logic inside <think> and </think> tags;
- 5. Tool Execution: If a tool is required, generate the tool call immediately after your reasoning. Use the following standard JSON format wrapped in XML tags: <tool_call>{"tool": "<tool_name>", "task": "<task_name>"}</tool_call>
- 6. Reasoning After Action: Once you receive the output from a tool, you must analyze the results to determine if further actions are needed or if the task is complete. Output this analysis inside <think> and </think> tags;
- 7. Final Output: When you have formulated your conclusion, you must wrap your final answer in <answer> and </answer> tags.

User Prompt: Please answer the following {task_type} question: Question: {Question} Please provide a complete step-by-step solution to this problem. Your reasoning should:

- 1. Analyze the problem systematically
- 2. Check if the tool execution and answer are correct
- 3. If there are errors, explain what went wrong and provide the correct reasoning
- 4. Provide the final answer Use natural expressions like ’let me think’ or ’hmm’ when helpful, but keep it concise. It’s encouraged to use self-reflection or verification especially in the verifying tool output in the reasoning process. Provide your detailed reasoning between <think> and </think> tags, then give your final answer between <answer> and </answer> tags. Output format: {output_format}

Figure 9. Prompt template.

### F. Additional Method Details of VISTA-R1

Image Additional Design. Integrating the InternVL3 [91] family into our reinforcement learning framework requires a bespoke adaptation of the visual processing pipeline, as its

composite architecture differs substantially from monolithic models such as Qwen2.5-VL [3]. Instead of performing early tensor conversion, we preserve raw <image> placeholders in the prompt and rely on InternVL’s native processor to expand them into 256 <IMG_CONTEXT> tokens and inject the corresponding visual embeddings during the forward pass. To ensure stability in distributed training, we customize Ray and Fully Sharded Data Parallel (FSDP) to wrap only the language decoder layers, leaving the vision encoder unsharded for memory efficiency, and we adjust attention masks and position IDs in VLLM to accommodate the extended visual token sequence. This minimally invasive interface adaptation preserves InternVL’s native visual processing while enabling robust, efficient multimodal GRPO training.

### G. Additional Implementation Details

#### G.1. Detailed Error Types for Error Analysis

We define the detailed error types as follows:

- • E1: Invocation schema violation (wrong function-call structure). The model produces an invalid tool call that violates the prescribed schema, such as missing required fields, extraneous keys, incorrect nesting, or non-JSONconformant structures that prevent execution.
- • E2: Invalid argument name (wrong argument key). The tool call structure is syntactically valid, but one or more argument names do not match the tool specification (e.g., using "x_axis" instead of "x_label"), causing the call to be rejected by schema validation.
- • E3: Invalid argument value (wrong argument format). Argument names are correct, but the value types or formats are invalid, such as providing strings where numbers are required, out-of-range values, or malformed lists that violate the tool’s input constraints.
- • E4: Incorrect argument value (wrong argument content). The tool call is syntactically valid and passes type checks, but the semantic content of one or more arguments is wrong (e.g., selecting the wrong region, axis, or series for analysis), leading the tool to operate on an incorrect target.
- • E5: Invalid output from tool execution (wrong answer format). The tool executes but returns an output that does not conform to the expected format for downstream use, such as missing required fields, malformed JSON, or values that cannot be parsed into the canonical answer representation.
- • E6: Incorrect reasoning from tool execution. The tool output is valid and informative, but the model fails to map it to the correct final answer, due to faulty logical deductions, misinterpretation of intermediate results, or inconsistent multi-step reasoning.

#### G.2. Additional Training Setup Details

We configure the training environment with a maximum of three turns per interaction, balancing exploration depth and computational cost. The agent retains full access to the conversation history and tool interaction logs across turns, and up to 24 trajectories are processed concurrently to maximize throughput. Agent-side code execution is implemented in Python 3.10. Training uses 8 NVIDIA H200 GPUs with the FSDP2 (Fully Sharded Data Parallel) strategy for efficient distributed optimization. We set parameters to temperature 0.7. The pipeline integrates a toolrouter service for real-time tool execution and feedback, and employs asynchronous rollouts with a batch size of 128 to maintain high GPU utilization. We consider four different backbones for VISTA-R1 with varying sizes, including InternVL3-2B/8B/14B and Qwen2.5-VL-7B, in the main experiment. We perform ablation experiments and additional analysis with InternVL3-8B as the VLM.

