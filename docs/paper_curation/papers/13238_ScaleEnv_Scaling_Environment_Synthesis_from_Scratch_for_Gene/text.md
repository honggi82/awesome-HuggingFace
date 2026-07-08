## ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training

Dunwei Tu*12† Hongyan Hao*3 Hansi Yang*3 Yihao Chen4† Yi-Kai Zhang12† Zhikang Xia3 Yu Yang3 Yueqing Sun3 Xingchen Liu5† Furao Shen12 Qi Gu3 Hui Su3 Xunliang Cai3

# arXiv:2602.06820v1[cs.AI]6Feb2026

### Abstract

Training generalist agents capable of adapting to diverse scenarios requires interactive environments for self-exploration. However, interactive environments remain critically scarce, and existing synthesis methods suffer from significant limitations regarding environmental diversity and scalability. To address these challenges, we introduce ScaleEnv, a framework that constructs fully interactive environments and verifiable tasks entirely from scratch. Specifically, ScaleEnv ensures environment reliability through procedural testing, and guarantees task completeness and solvability via tool dependency graph expansion and executable action verification. By enabling agents to learn through exploration within ScaleEnv, we demonstrate significant performance improvements on unseen, multi-turn tool-use benchmarks such as τ2-Bench and VitaBench, highlighting strong generalization capabilities. Furthermore, we investigate the relationship between increasing number of domains and model generalization performance, providing empirical evidence that scaling environmental diversity is critical for robust agent learning.

### 1. Introduction

The rapid evolution of Large Language Models (LLMs) has established a foundation for Artificial General Intelligence (AGI), driven largely by the success of Data Scaling

*Equal contribution †Work done during an internship at Meituan 1National Key Laboratory for Novel Software Technology, Nanjing University 2School of Artificial Intelligence, Nanjing University, Nanjing, China 3Meituan, Beijing, China 4Institute of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China 5School of Statistics, East China Normal University, Shanghai, China. Correspondence to: Hongyan Hao <haohongyan02@meituan.com>, Qi Gu <guqi03@meituan.com>.

Preprint. February 9, 2026.

and Model Parameter Scaling laws (Kaplan et al., 2020; Grattafiori et al., 2024; OpenAI, 2025). However, transforming these models from text generators into agents requires a fundamental shift: agents must effectively interact with dynamic environments and iteratively refine their actions based on environment feedback (Yao et al., 2022). Achieving such capabilities calls for training LLMs within dynamic environments, equipped with executable tools to allow agents to interact and receive immediate feedback. However, constructing environments that can be used for training LLM agents presents two core challenges. The first is Realism: tools synthesized directly by LLMs are often functionally unreliable, while LLM-based simulators are prone to severe hallucinations (Liu et al., 2024; Li et al., 2025). We argue that environments must be grounded in verified, executable code rather than probabilistic text generation to ensure robust feedback. The second challenge is Scalability: synthesis cannot rely on finite external documentation or manual human intervention (Cai et al., 2025).

To overcome these challenges, we introduce SCALEENV, a framework enabling the fully automated construction of high-fidelity, interactive environments and verifiable tasks. SCALEENV operates through two synergistic phases that ensure both code-level rigor and real-world complexity. In the first phase, the system builds the Domain Foundation by leveraging LLMs to define tool and database schemas from simple domain keywords. It employs a multi-agent architecture to generate functional code for tools and databases, which is rigorously validated via a Procedural Testing mechanism to guarantee error-free execution. These components are then consolidated into a global Tool Dependency Graph to map logical relationships. In the second phase, we focus on Task Construction using a dependency-aware expansion strategy. By sampling seed tool chains from the graph and dynamically introducing associated database states and distractor data, the system “snowballs” the environment state from linear paths into complex non-linear subgraphs. This process ensures the synthesized environment supports openended exploration and diverse solution paths, providing a verifiable foundation for grounding natural language user intents.

SCALEENV produces a complete training ecosystem comprising executable toolkits, high-fidelity environment states, and verifiable user intents. By training Qwen-3 models (Yang et al., 2025) via Zero RL on this synthesized universe, we observe substantial performance boosts on unseen Out-of-Distribution (OOD) benchmarks, including τ2Bench (Barres et al., 2025) and VitaBench (He et al., 2025). Crucially, this evaluation is strictly OOD: the synthesized training domains are entirely disjoint from the evaluation domains, and the benchmarks present distinct data formats (e.g., policy-constrained dialogues) not encountered during training. Furthermore, we empirically characterize the impact of Environment Scaling on model generalization performance, revealing a distinct curve that validates environmental diversity as a critical determinant for robust generalization.

Our main contributions are as follows:

- • We propose SCALEENV, a fully automated framework that synthesizes high-fidelity interactive environments from scratch. It establishes an large-grade pipeline for agentic data, circumventing the limitations of fixed environment and manual API integration.
- • We design a robust synthesis mechanism combining Procedural Testing and Graph Expansion. This approach ensures that generated environments possess rigorous codelevel verifiability while maintaining the logical complexity required for deep reasoning.
- • We demonstrate that agents trained on SCALEENV achieve significant zero-shot generalization on unseen benchmarks. Additionally, we provide empirical evidence for the Environment Scaling Curve, establishing a new paradigm for data-centric agent training.

### 2. Related Works

#### 2.1. Tool Learning

The field of tool learning has evolved from SFT to autonomous exploration. Early works (Schick et al., 2023; Qin et al., 2023; Liu et al., 2024; Prabhakar et al., 2025) demonstrated that LLMs could master function calls through static demonstrations. To reduce dependency on expensive expert trajectories and enhance agents’ self-exploration capabilities, frontier research has pivoted towards RL (Luo et al., 2025; Jin et al., 2025; Lu et al., 2025). However, large-scale RL exploration necessitates scalable interaction environments. This paper aims to bridge this gap by synthesizing diverse and reliable environments, enabling the training of highly generalizable tool-use agents.

#### 2.2. Environment Scaling

Constructing effective environments for agents requires considering three critical dimensions: diversity, realism, and

scalability. Existing approaches fall into three categories. (1) Real-world environments. (Fang et al., 2025; Xu et al., 2025; Yao et al., 2026) collect actual tools or remote APIs. Although offering high realism, they are constrained by limited domain availability and safety policies that restrict action spaces. Consequently, the lack of diverse state-altering tasks, combined with prohibitive latency and costs, creates a bottleneck for scalable agent training. (2) LLM-simulated environments. (Liu et al., 2024; Chen et al., 2025; Li et al., 2025; Team et al., 2025; Ye et al., 2025) leverage large language models to generate tool responses and execution results, offering significant advantages in scalability, lowcost execution, and flexible domain definition. However, these approaches often suffer from fundamental limitations in realism and fidelity; specifically, they are prone to hallucinations (Kadavath et al., 2022; Zhang et al., 2025) and frequently fail to maintain authentic environment states. (3) Synthetic environments. While recent frameworks like AutoForge (Cai et al., 2025) and EnvScaler (Song et al., 2026) offer executable pipelines for environment synthesis, they face distinct limitations: the former is constrained by the limited scalability of document-based generation, while the latter struggles to construct complex, user-interactive tasks. Furthermore, both approaches exhibit inadequate consistency between the generated tasks and their corresponding environmental states, undermining the reliability of the resulting sandboxes. To address these defects, we introduce SCALEENV, which ensures coherence and execution reliability through execution-based verification. By providing a high-fidelity sandbox, SCALEENV enables the robust and scalable reinforcement learning necessary for complex reasoning tasks.

### 3. Preliminaries

To construct a task for agentic RL training, we first need a domain foundation B = ⟨Σ,T⟩. It consists of two parts: a set of databases Σ and a set of executable tools T. Σ defines the valid space of environment state Svalidenv = {senv | senv |= Σ}, while T contains all available functions or APIs in this domain.

Following the domain foundation B, we can construct an environment E = ⟨senv0 ,T⟩ consisting of a set of databases senv0 with values filled following the database schema in B and executable tools T inherited from B. E represents the external world at the start of an episode, as senv0 directly gives the initial hidden external environment state.

Under a specific environment E, a task ψ = ⟨E,u,Puser⟩ then binds it to a specific user with some hidden goals, denoted as u, as well as the user profile Puser that contains all necessary information about the user (e.g., permissions, location, history). The interaction between an LLM agent and the simulated user to accomplish this task can then be

[Figure 1]

[Figure 2]

###### Tool Code

###### Tool Test

[Figure 3]

###### Tool Schema

[Figure 4]

| |[Figure 5]|
|---|---|
|- name: search_job<br>- description: filter jobs……<br>- pre-condition: login<br>- post-condition: get_details<br>- parameter:<br>| |
| | |

Tool Dependency Analysis

class JobSeekingTools:

class ToolTestRunner :

[Figure 6]

[Figure 7]

def __init__(self):

def run_test_case(

self.db = JobSeekingDB()

tool_code, database_instance,

def search_job(self,keyword,…):

Domain

###### Domain Definition

keyword = keyword.strip() ……

test_case) -> TestResult:

[Figure 8]

[Figure 9]

db_state_before = ……

- keywords

Graph 𝑮

Debuger

Tool related Database

Database

[Figure 10]

- - name: job seeking
- - description: Centralizes job seeking with resume submission and maintenance, personalized job matching……

Database Mapping

[Figure 11]

D

- A

- B

Analysis

C

L

Database Code

[Figure 12]

E

[Figure 13]

[Figure 14]

###### Database Test

###### Database Schema

|[Figure 15]| |
|---|---|
| |class JobSeekingDB(DB):<br><br>job:Optional[Dict[str, Job]] = …<br><br>interview:Optional[Dict[str, …<br><br>app_stage:Optional[Dict[str, …<br><br>app_note:Optional[Dict[str, …|
| | |

G

- - table_name: job
- - description: stores job information…
- - columns:
- - application_id
- - application_name
- - primary_key: appilication_id

class DatabaseTestRunner : def run_test_case(

F

[Figure 16]

database_instance,

H K

test_case) -> TestResult:

db_state_before = ……

[Figure 17]

- Figure 1. Overview of Executable Graph Construction. The pipeline proceeds from left to right: (1) Schema Definition for tools and databases; (2) Implementation validated via procedural testing; and (3) Tool Dependency Graph Construction to model execution logic.

formulated as as a Partially Observable Markov Decision Process (POMDP) M = ⟨S,A,O,T ,R⟩. Each state st = (senvt ,ht,u) ∈ S consists of three components: current environment state senvt , the interaction history ht and the user intent u. The action space A = Aresp∪Atool is defined as the union of natural language responses space Aresp, and the space of tool execution commands Atool for tools in T. Correspondingly, the observation space O = Oresp ∪ Otool comprises user feedback space Oresp and tool execution results space Otool. The state transition function T : S × A → S × O evolves the state based on the action type: an action at ∈ Atool triggers a deterministic update to both the environment senvt and the history ht with the return value of tools, whereas an action at ∈ Aresp updates only the history ht with the simulated user’s reply, leaving senvt unchanged. Finally, suppose the interaction is terminated at timestep T, the outcome reward signal is defined as r = R(senvT ,u), which evaluates whether the final environment state senvT successfully satisfy the user intent u.

domain. Ultimately, these synthesized tasks serve as the foundation for scalable agentic RL training.

4.1. Executable Graph Construction 4.1.1. TOOL & DATABASE SCHEMA DEFINITION

To establish a robust interactive domain foundation, we introduce a two-step synthesis pipeline. As shown in Figure 1, we first rigorously define Tool & Database Schemas to formalize the domain’s operational logic, and subsequently implement the corresponding Executable Code to transform these abstract definitions into a functional, verifiable sandbox.

Top-Down Tool Schema Synthesis. Starting with a specific domain name (e.g., “Job Seeking”), we employ a topdown synthesis approach by using an LLM to first conceptualize the domain logic and generate the Tool Schema. This schema rigorously defines the interface of the atomic tool set T, including precise functional descriptions, parameters, and logical pre/post-conditions (e.g., submit application logically necessitates a preceding upload resume).

### 4. Method

Motivated by the limitations of existing works that fail to synthesize diverse and reliable environments for agentic RL training, we present SCALEENV, a unified framework designed to synthesize multi-turn, interactive, and strictly verifiable environments, which can effectively scale up agentic RL training along with their corresponding tasks. To ensure modularity and extensibility, we decouple the environment and task construction into two distinct phases: Executable Graph Construction (Section 4.1) and Task Instantiation (Section 4.2). The executable graph establishes the logical skeleton of a given domain. Defined as a graph comprising a set of executable tools, it determines the available action space and the dependency relationships within the domain. Following the construction of executable graph, we can further generate a diverse set of high-quality tasks for each

Database Schema Derivation & Mapping. With the tool schema synthesized, a Database Agent analyzes the tool definitions to reverse-engineer the database structure necessary to support the environment. For instance, the presence of a submit application tool implies the existence of an Application table (and a reference Job table) in the environment database. Through de-duplication and filtering, the agent derives a consolidated Database Schema for Svalidenv , defining table structures and integrity constraints. We simultaneously establish a tool-database mapping that explicitly identifies the specific tables associated with each tool, laying the necessary groundwork for the subsequent code implementation.

Reward Specification While the LLM-as-a-judge paradigm is widely adopted to implement the reward function R, it often suffers from high computational overhead and vulnerability to reward hacking (Gabor et al., 2025; Pan et al., 2024). As such, we introduce a rule-based evaluator that directly checks the agent’s final database state senvT against the ground-truth state senvgt . Note that different types of data naturally require different criteria ( e.g., critical data like prices or quantities require exact match, while text comments may only require fuzzy match), we categorize database columns into three matching policies: (1) Exempt Fields: dynamically generated IDs and optional columns that do not affect task success; (2) Hard Constraints: critical data such as timestamps or quantities that require strict character-level or numerical equality; and (3) Semantic Alignment: descriptive text that only require fuzzy semantic matching.

- 4.1.2. TOOL & DATABASE SCHEMA IMPLEMENTATION

Given the synthesized tool and database schema, we then proceed to implementing them through LLM-powered code generation. Since the database defines the underlying state structure required for tool execution, we prioritize generating and verifying the database code, which then serves as the necessary prerequisite for the subsequent tool implementation and verification.

Database Implementation & Verification. We first utilize an LLM to translate the database schema into executable code. To ensure reliability, we concurrently generate test scripts to validate the database implementation against integrity constraints. Any execution failures trigger another LLM agent that works as a debuger and iteratively refines the code based on error tracebacks until all tests pass, establishing a stable storage layer for the environment.

Tool Implementation via Procedural Testing. Note that generating valid tool code is a non-trivial process involving intricate logic and interactions across multiple databases, and direct generation is prone to hallucination. As such, we propose a Procedural Testing mechanism. Specifically, the Code Agent implements the tool logic, while a Test Agent simultaneously synthesizes unit test cases and corresponding matched database instances. We then execute the tool code on the matched database instances and validate its correctness based on three distinct outcomes:

• Unexpected Failure: Any other runtime errors or state inconsistencies indicate defects. In this case, the Debug Agent analyzes the error logs to iteratively rectify either the tool implementation or the database instance until the procedural test is satisfied.

- 4.1.3. TOOL DEPENDENCY GRAPH CONSTRUCTION

To facilitate the synthesis of semantically coherent multistep tasks, we further employ a Tool Dependency Agent to systematically evaluate pairwise relationships between the verified tools. This analysis is grounded in three key dimensions: data flow (parameter passing), pre/postconditions (logical prerequisites), and state dependencies (shared database tables). Based on these criteria, the agent establishes directed edges representing causal links, consolidating the atomic tools into a unified Tool Dependency Graph G. This graph serves as the basis for subsequent task instantiation.

- 4.2. Task Instantiation via Graph Expansion

Building upon the tools and databases constructed in Section 4.1, we instantiate diverse tasks for agentic RL training. The primary challenge for task instantiation is to construct a high-fidelity environment E capable of supporting the extensive trial-and-error in RL. Unlike SFT where no trialand-error mechanism exists, an RL environment must satisfy two critical requirements:

- • Entity Consistency. The synthesized environment should be consistent across all database tables. An entity appearing in one table (e.g., a user id in the Order table) must map correctly to corresponding entities in related tables (e.g., the User table).

- • Interaction Completeness. The environment must support execution fidelity across the entire feasible action space, not merely along the optimal trajectory. Formally,

for any valid tool calling action a ∈ Atool taken by the agent, the environment E must return a valid, semantically meaningful observation otool, ensuring that exploration is not artificially terminated due to missing database entries or implementation gaps.

- • Success: The execution completes without error, and the resulting state transitions strictly match the expected database states.
- • Anticipated Rejection: The tool correctly identifies and handles invalid inputs by raising the pre-defined exceptions as specified in the schema.

To generate tasks with environments satisfy the two constraints above, we propose a Graph Expansion strategy. As illustrated in Figure 2, this process constructs complex environmental states via a two-stage iterative procedure: seed tool chain sampling, which initializes a seed subgraph K(C1), followed by controlled environment expansion, which expands the environment based on complexity metrics and scales via a fallback mechanism.

g(𝒟𝓃) : new tool chain selection difficulty

###### Subgraph 𝑲(𝑪𝟏)

###### Task

𝓓𝓷 = 𝑮 \ 𝓗𝓷

Tool Chain 𝑪𝟏

- - user profile
- - task description
- - evaluation rubrics

###### Expanded Tool Chain 𝑲(𝑪𝟏)

Task

- A C D

- A C

D

[Figure 18]

- B

- B

|Construction|BFS Expand|
|---|---|
|- user profile<br>- task description<br>- evaluation rubrics<br>| |
| | |

Sample Tool Chain

###### Domain Graph 𝑮

E

G H K

[Figure 19]

F

// //

Task-specific new Tool Chain 𝑪𝒏 𝟏

//

[Figure 20]

- A C

D

- B E

| | | |
|---|---|---|
| |DB Instantiated| |

Strong Solver

L //

D

- A

- B

C

L

E

G

// //

···

//

N

N

p=𝜋 c(ℋ𝓃), g(𝒟𝓃), |𝒟𝓃| Subgraph 𝑲(𝑪𝒏) p > threshold 𝜏 ?

F

|ℋ𝓃| < 20?

H K

Y

###### Scaled Environment 𝓔

Random new tool chain 𝒕

ℋ𝓃 = 𝐾(𝐶 )

c(ℋ𝓃): environment structural complexity

Y

Random

- A C

D

- B

|Subgraph 𝑲(𝒕)|𝑲(𝒕) ∪ 𝓗|
|---|---|
| | |

𝓷

L

// //

Subgraph 𝑲(𝑪𝒏  )

- Figure 2. Overall pipeline of Task Instantiation via Graph Expansion. The process involves: (1) Seed Chain Sampling from the dependency graph; (2) Task Initialization with verifiable execution; and (3) Controlled Environment Expansion to scale complexity while maintaining solvability.

capabilities.

- 4.2.1. TASK INITIALIZATION WITH SEED TOOL CHAIN

Given the domain-specific tool dependency graph G, we initialize the complete task construction process through three stages: (1) executable seed tool chain sampling, (2) constraint-satisfying environment instantiation and (3) grounded instruction synthesis.

Instruction Synthesis. Given the verified seed chain C1 and environment state senv0 , we then employ an LLM to synthesize the user profile Puser and user instruction u. We ensure the generation of u to be strictly grounded in C1 as the reference solution, which prevents the introduction of external priors or hallucinations unsupported by the underlying environment. The evaluation criteria R is also directly derived from the final state senvgt after executing C1, which ensures alignment between the seed tool chain C1 and reward R for robust RL training.

Executable Seed Tool Chain Sampling. We initiate the process by sampling a seed tool chain C1 = (a1,a2,...,ak), which serves as a valid reference path to solve the task to be constructed. The tool chain is formulated as an executable code, and generated by prompting an LLM with the tool dependency graph G and the relevant database schema. This setup enables the joint modeling of tool sequences and their arguments, preventing possible disconnection if parameter instantiation is treated as an isolated step. Representing the tool chain as code also inherently satisfies data flow constraints; the output of a preceding action ai is programmatically propagated as the input to subsequent actions ai+1.

4.2.2. CONTROLLED ENVIRONMENT EXPANSION

While the three-stage procedure in Section 4.2.1 ensures Entity Consistency, restricting the agent to the minimal environment state encourages overfitting, where the agent collapses into memorizing sparse trajectories rather than learning generalized reasoning. To foster robust RL training and ensure Interaction Completeness, we propose an iterative environment refinement strategy. We first expand the initial seed chain C1 into a semantically dense subgraph H1 = K(C1) ⊂ G and incrementally refine the environment. Then we further try to construct more tool chains C2,...,Cn until we reach the capability ceiling of current LLM, and repeat the expansion-refinement process of C1 to these newly constructed tool chains C2,...,Cn for further environment refinement.

Initial State Construction with Distractor Injection. Based on the generated tool chain C1, we construct an initial environment state senv0 that strictly supports its execution while preserving Entity Consistency. We employ an LLMbased generation pipeline to synthesize senv0 and validate it by executing C1 on it to ensure task feasibility. Furthermore, to encourage robust reasoning, we populate the database tables within senv0 with additional records that act as distractors. The density of these distractors is dynamically scaled according to the predefined task complexity. While these distractors adhere to all schema constraints, they remain functionally orthogonal to the ground-truth trajectory, forcing the agent to acquire precise information filtering

Dependency-Aware Topological Expansion. We first expand the initial tool chain C1 into a local subgraph H1 = K(C1) ⊂ G for environment refinement. A naive stochastic injection of tools risks introducing dependency dead-ends—nodes whose prerequisite inputs cannot be sat-

isfied by the current available tool output. We mitigate this via Dependency-Aware BFS: starting with H1 = C1, we iteratively traverse G and add a new tool node v ∈ G to H1 if and only if the input and output dependencies of v can be fully satisfiable by a tool subset of H1. Then for any newly added tool node v, we further execute it with its argument derived from H1 and refine the environment if any errors arise from the execution.

Finally, to ensure sufficient exploration space, we enforce a minimum constraint |Hn| ≥ 20 for Hn. If it falls short, we randomly sample valid auxiliary chains and merge it into Hn to satisfy this constraint. This strategy leverages the semantic reasoning of LLMs to dynamically balance the environment construction, producing high-fidelity tasks that support complex reasoning while maintaining verifiable supervision signals.

LLM-Gated Chain Expansion. Since the subgraph H1 is derived from the only seed tool chain C1, a natural idea is to extend similar procedure to multiple tool chains C1,C2,...,Cn. Nevertheless, such extension cannot be done forever as the new tool chain should only contain tools not used in previous subgraphs Hn = ni=1 K(Ci), As such, letting Dn = G \ Hn denote the set of candidate tools available after iteration n, we determine whether to inject a new seed chain Cn+1 ⊆ Dn via a parametric gating policy π. Instead of relying on brittle heuristics, we employ a strong LLM to approximate a value function for expansion, conditioning on the following metrics:

- • Structural Complexity (c(Hn)). To quantify the representational sufficiency of tools currently in Hn, we intro-

duce a complexity metric c(Hn) = |VH

n|+λ|EHn|

Ssat , where VH

n

and EH

n

represent the number of nodes and edges in Hn, λ = 0.5 weighs the complexity of dependencies, and Ssat = 50 is the saturation constant. Intuitively, this score assesses whether Hn captures the necessary structural and semantic depth required to support complex environmental interactions.

- • Feasibility Score (g(Dn)). Complementary to the tool complexity, we also need to consider the feasibility of the

task within the remaining tools in Dn. Since ground truth labels for complex tool chains are often unavailable in open-ended environments, we rely on a powerful “Oracle” agent and define g(Dn) ∈ [0,1] as the success rate of this oracle agent in identifying executable tool chains within Dn. In our implementation, we instantiate use Qwen3-235B-A22B augmented with a best-of-k search strategy (k = 16 rollouts) to maximize the likelihood of discovering valid paths.

Along with the number of available tools |Dn|, we construct a prompt π(|Dn|,c(Hn),g(Dn)) instructing the LLM to balance the trade-off between diversity and solvability. The model outputs a compatibility score p ∈ [0,1]. If p ≥ τ, we sample a new chain Cn+1, expand it into its dependency subgraph K(Cn+1) following similar procedure that obtains K(C1) from C1. We then also execute tools in K(Cn+1) and refine the environment if any errors arise from the execution, before merging it into Hn as Hn+1 = Hn ∪ K(Cn+1).

### 5. Experiments

#### 5.1. Experimental Setup

For the domain foundation and task synthesis phases, we utilized a diverse suite of high-performance LLMs, including Deepseek-V3.2 (Liu et al., 2025), GLM-4.7 (Zeng et al., 2025), GPT-5.1, and Qwen3-32B (Yang et al., 2025), to instantiate various agent roles. Across the 16 synthesized domains, each environment comprises approximately 50 tools and 5–20 database tables. Our model series, denoted as Qwen3-SE (ScaleEnv), is trained from Qwen3 (Yang et al., 2025) using group relative policy optimization (GRPO) (Shao et al., 2024) on our synthesized domains and tasks. We use Qwen2.5-72B-Instruct (Qwen et al., 2025) as the user simulator to provide natural language feedback. Regarding hyperparameters, the Qwen38B model was trained with a rollout batch size of 1024, while the Qwen3-32B model utilized a rollout batch size of 2048. Both models were trained for 48 steps with a learning rate of 10−6. Detailed domain and task compositions are provided in Appendix B.

#### 5.2. Main Results: Generalization to Unseen Domains

A critical question in agentic training is whether performance gains stem from genuine reasoning capabilities or merely from overfitting to the training distribution. To answer this question, we assess the generalization capabilities of Qwen3-SE model series across three dimensions using established benchmarks. (1) Reasoning Generalization: We utilize the cross-domain subset of VitaBench (He et al., 2025), which presents ambiguous user needs requiring proactive information retrieval and complex multi-step planning, testing the transfer of reasoning skills to challenging logical structures not seen during training. (2) Domain Generalization: We evaluate the model performance across a wide spectrum of functional areas, including the Airline, Retail, and Telecom domains from τ2-Bench (Barres et al., 2025), and the Delivery, In-store, and OTA domains from VitaBench. As visually evidenced by the tool embedding distribution in Figure 4, these evaluation domains are semantically distinct and spatially separated from our 16 synthesized training domains, ensuring a rigorous Out-Of-Distribution (OOD) evaluation. (3) Format Generalization: While SCALEENV focuses on direct user-tool

- Table 1. Zero-shot generalization performance. The Qwen3-SE model series, trained with environments and tasks constructed from our SCALEENV consistently outperforms baselines across diverse domains.

Model

τ2-Bench VitaBench

Retail Airline Telecom Cross Delivery Instore OTA

Open-weights Models GPT-OSS-120B-A5B 57.0 38.0 45.6 15.0 37.0 42.0 12.0 Qwen3-235B-A22B-2507 71.9 58.6 47.3 14.5 45.0 32.0 15.8 Kimi-K2-0905 70.6 56.5 65.8 11.5 32.5 30.0 18.8 Seed-OSS-36B 68.4 52.0 41.2 6.1 26.0 39.0 7.0 xLAM-2-32B-fc-r 55.3 52.0 16.7 4.0 26.0 17.0 10.0

Main Results Qwen3-8B 38.4 30.5 21.5 1.5 18.3 14.8 4.5 Qwen3-SE-8B 50.9(+12.5) 37.5(+7.0) 27.2(+5.7) 3.0(+1.5) 26.3(+8.0) 23.8(+9.0) 7.0(+2.5) Qwen3-32B 59.5 48.0 27.2 5.3 27.0 22.5 4.5 Qwen3-SE-32B 63.6(+4.1) 48.0(+0.0) 30.9(+3.7) 10.8(+5.5) 31.3(+4.3) 34.5(+12.0) 12.5 (+8.0)

- Table 2. Pass@4 on VitaBench which shows the model upperbound performance.

[Figure 21]

[Figure 22]

Model Cross Delivery Instore OTA Avg. Qwen3-8B 6 43 44 15 27.0 Qwen3-SE-8B 12 56 57 18 35.8 Qwen3-32B 15 57 58 14 36.0 Qwen3-SE-32B 29 65 66 27 46.8

(a) VitaBench (b) τ2-Bench

Figure 3. Domain Scaling Analysis (Pass@4). Comparison of zero-shot generalization as training domains scale from N = 2 to 16. N = 0 denotes the base model. Performance improves monotonically across both benchmarks.

interaction without explicit policy documents, τ2-Bench requires agents to strictly adhere to lengthy textual policies during dialogue. Success here demonstrates that our agents can generalize their learned behavior to novel interaction formats and constraints not explicitly present during training.

policies. While fundamentally differnet from SCALEENV, the consistent gains on τ2-Bench suggest that our agents have generalized the ability to follow constraints and handle complex state transitions, even when presented in a novel textual policy format.

Reasoning Generalization. A critical challenge in agentic tasks is interpreting ambiguous user needs that require multi-step reasoning and proactive planning. We observe substantial gains on VitaBench, a benchmark specifically designed to test these capabilities. For instance, when a user states ”I am sick,” the agent must infer a latent intent to ”recommend light food”. As shown in Table 1, Qwen3-SE-32B achieves a remarkable improvement in the most challenging cross-domain subset, doubling the performance of the base model. This indicates that training on SCALEENV’s verifiable environments enables the agent to transfer high-level reasoning skills to diverse, logic-heavy scenarios.

Performance Upper Bound Analysis. Beyond improving average stability, SCALEENV significantly elevates the model’s capability ceiling on complex tasks, as is illustrated by the Pass@4 score on VitaBench in Table 2 that measures the probability of generating at least one correct solution within four attempts. Notably, in the complex cross-domain subset, our method nearly doubles the success potential. This confirms that SCALEENV does not merely teach the model to memorize simple patterns, but fundamentally enhances its capacity to search for and execute correct solutions in more difficult tasks.

Domain and Format Generalization. SCALEENV demonstrates robust transferability across entirely novel domains and interaction formats. (1) Domain Adaptation: Despite the strict exclusion of test domains from our training set, our method consistently boosts performance across all 7 evaluation domains. (2) Format Adaptation: Notably, τ2Bench requires agents to strictly adhere to lengthy textual

#### 5.3. Domain Scaling Analysis

To investigate how environment diversity drives generalization, we evaluate models trained on N ∈ {2,4,8,16} unique domains while keeping the same number of 1024

- Table 3. Ablation of Executability Verification (Avg@4) on τ2-Bench

Setting Retail Airline Telecom

Qwen3-8B 38.4 30.5 21.5 w/o EV 42.3 30.0 25.2 Qwen3-SE-8B 50.9 37.5 27.2

- Table 4. Ablation of Reward Mechanisms. Results are averaged across three τ2-Bench domains (Retail, Airline, Telecom).

Reward Setting Avg@4 Pass@4 Passˆ4

LLM-as-a-Judge 36.5 58.8 14.6 Rule-Based (Ours) 38.5 62.9 15.0

tasks. As illustrated in Figure 3, we observe a steady upward trend in zero-shot generalization as the number of training domains increases. This consistent growth across disparate benchmarks confirms that environmental richness is a decisive factor in unlocking the model’s transfer capabilities. Furthermore, exposing agents to a wider variety of tools and databases allow them to internalize abstract, domain-agnostic reasoning strategies rather than overfitting to specific templates. Notably, performance has not yet fully plateaued at N = 16, suggesting that further scaling of environmental diversity remains a promising direction for data-centric agent training.

- 5.4. Analysis Experiment

Ablation Study: Executability Verification. A core tenet of SCALEENV is the rigorous Execution-Based Verification (EV) of synthesized tasks. To quantify its impact, we conducted an ablation study by training a model on a dataset generated without this verification step (denoted as “w/o EV”). In this setting, while the tool dependency graph was constructed, the tools are never subjected to actual parameter-driven execution, and the environment states are not iteratively patched to ensure solvability.

As illustrated in Table 3, the removal of execution verification leads to a consistent degradation in performance across all domains of τ2-Bench. Without EV, the training data contains tool calls that appear semantically plausible but fail during runtime due to unsatisfied preconditions or mismatched database states (e.g., attempting to refund a nonexistent order). Such noisy rollouts introduce conflicting reward signals, preventing the policy from learning precise, logic-grounded decision-making.

Ablation Study: Reward Mechanism. To evaluate the impact of different reward signals, we compare our deterministic rule-based evaluator against the standard LLM-asa-Judge paradigm. As shown in Table 4, our rule-based approach yields superior performance across all metrics.

Table 5. Domain Stability Analysis (Avg@4). Set A contains wedding planning, knowledge management, job seeking and healthcare telemedicine. Set B contains express logistics, job seeking, email management and pet care.

Setting Cross Delivery Instore OTA Avg. Qwen3-8B 1.5 18.3 14.8 4.5 9.8

- Set A (4 Domains) 2.0 24.0 22.3 7.0 13.8

- Set B (4 Domains) 2.0 25.5 18.3 7.5 13.3

These results demonstrate that while LLM-based judges provide semantic flexibility, they are often susceptible to reward hacking, where agents optimize for linguistic alignment at the expense of logical correctness. In contrast, our rule-based reward enforces rigorous, database-level fidelity, providing a more objective and robust learning signal. Furthermore, the rule-based approach minimizes computational overhead by replacing expensive LLM inference with efficient rule-based verification, facilitating stable large-scale RL exploration.

Domain Stability Analysis. A potential concern in procedural generation is whether performance gains stem from specific ”lucky” domains or general synthesis robustness. To investigate this, we conducted a stability analysis by training the Qwen3-8B model on two distinct, non-overlapping subsets of synthesized dommains, fixing the domain count at 4 and task count at 1024. As reported in Table 5, both subsets consistently outperform the baseline across all metrics on VitaBench. This consistency demonstrates that EnvZero produces high-fidelity environments reliably across diverse scenarios, ensuring that generalization improvements are driven by the structural advantages of our framework rather than artifacts of specific cherry-picked domains.

### 6. Conclusion

In this paper, we presented SCALEENV, a framework to synthesize high-fidelity interactive environments and verifiable tasks. By shifting from static dataset interpolation to a complete generation pipeline, SCALEENV circumvents the inherent limitations of data scarcity. Our extensive evaluations across multiple model scales demonstrate that training on SCALEENV-synthesized environments and tasks significantly boosts the performance of baseline models on unseen benchmarks, evidencing robust zero-shot generalization. Furthermore, we empirically formulated an domain scaling curve, establishing that scaling environmental diversity is more critical than task quantity for cultivating generalist agent capabilities. These results confirm that SCALEENV provides a stable and scalable paradigm for data-centric reinforcement learning, paving the way for the development of robust, general-purpose autonomous agents.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically by addressing the data scarcity bottleneck in training generalist tool-use agents. Our proposed framework, SCALEENV, provides a safe, virtual sandbox for agents to learn complex tool-use capabilities, thereby mitigating the risks of accidental harm associated with training directly on real-world APIs.

However, we acknowledge the potential risks associated with the generative nature of our method. Theoretically, SCALEENV is domain-agnostic and capable of synthesizing arbitrary interactive environments. Without proper oversight, this capability could be misused to construct environments that model harmful or unethical behaviors. We strongly emphasize that the application of procedural environment synthesis must adhere to strict ethical guidelines. Future research should explore safety alignment mechanisms that prevent the synthesis of malicious domains while maintaining the diversity required for robust generalist training.

### References

Barres, V., Dong, H., Ray, S., Si, X., and Narasimhan, K. tau2-bench: Evaluating conversational agents in a dualcontrol environment. arXiv preprint arXiv:2506.07982, 2025.

Cai, S., Fang, R., Wu, J., Li, B., Wang, X., Jiang, Y., Su, L., Zhang, L., Yin, W., Zhang, Z., Feng, F., Xie, P., and Wang, X. Autoforge: Automated environment synthesis for agentic reinforcement learning, 2025. URL https:

//arxiv.org/abs/2512.22857.

Chen, Z., Zhao, Z., Zhang, K., Liu, B., Qi, Q., Wu, Y., Kalluri, T., Cao, S., Xiong, Y., Tong, H., et al. Scaling agent learning via experience synthesis. arXiv preprint arXiv:2511.03773, 2025.

Fang, R., Cai, S., Li, B., Wu, J., Li, G., Yin, W., Wang, X., Wang, X., Su, L., Zhang, Z., et al. Towards general agentic intelligence via environment scaling. arXiv preprint arXiv:2509.13311, 2025.

Gabor, J., Lynch, J., and Rosenfeld, J. Evilgenie: A reward hacking benchmark. arXiv preprint arXiv:2511.21654, 2025.

marking llm agents with versatile interactive tasks in realworld applications. arXiv preprint arXiv:2509.26490, 2025.

Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H., and Han, J. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Li, Y., Inan, H. A., Yue, X., Chen, W.-N., Wutschitz, L., Kulkarni, J., Poovendran, R., Sim, R., and Rajmohan, S. Simulating environments with reasoning models for agent training, 2025. URL https://arxiv.org/ abs/2511.01824.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Liu, W., Huang, X., Zeng, X., Hao, X., Yu, S., Li, D., Wang, S., Gan, W., Liu, Z., Yu, Y., et al. Toolace: Winning the points of llm function calling. arXiv preprint arXiv:2409.00920, 2024.

Lu, F., Zhong, Z., Liu, S., Fu, C.-W., and Jia, J. Arpo: Endto-end policy optimization for gui agents with experience replay. arXiv preprint arXiv:2505.16282, 2025.

Luo, M., Jain, N., Singh, J., Tan, S., Patel, A., Wu, Q., Ariyak, A., Cai, C., Tarun Venkat, S. Z., Athiwaratkun, B., Roongta, M., Zhang, C., Li, L. E., Popa, R. A., Sen, K., and Stoica, I. Deepswe: Training a stateof-the-art coding agent from scratch by scaling rl. https://pretty-radio-b75.notion.site/DeepSWE-Traininga-Fully-Open-sourced-State-of-the-Art-Coding-Agentby-Scaling-RL-22281902c1468193aabbe9a8c59bbe33, 2025. Notion Blog.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,

- A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

He, W., Sun, Y., Hao, H., Hao, X., Xia, Z., Gu, Q., Han, C., Zhao, D., Su, H., Zhang, K., et al. Vitabench: Bench-

Maaten, L. v. d. and Hinton, G. Visualizing data using t-sne. Journal of machine learning research, 9(Nov): 2579–2605, 2008.

OpenAI. Introducing GPT-5-2. https://openai. com/index/introducing-gpt-5-2/, 2025. Accessed: February 28, 2025.

Pan, A., Jones, E., Jagadeesan, M., and Steinhardt, J. Feedback loops with language models drive in-context reward hacking. arXiv preprint arXiv:2402.06627, 2024.

Prabhakar, A., Liu, Z., Zhu, M., Zhang, J., Awalgaonkar, T., Wang, S., Liu, Z., Chen, H., Hoang, T., Niebles, J. C., et al. Apigen-mt: Agentic pipeline for multi-turn data generation via simulated agent-human interplay. arXiv preprint arXiv:2504.03601, 2025.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., et al. Qwen2.5 technical report, 2025. URL https: //arxiv.org/abs/2412.15115.

Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Song, X., Chang, H., Dong, G., Zhu, Y., Dou, Z., and Wen, J.-R. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis, 2026. URL https://arxiv.org/abs/2601.05808.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Xu, Z., Soria, A. M., Tan, S., Roy, A., Agrawal, A. S., Poovendran, R., and Panda, R. Toucan: Synthesizing 1.5 m tool-agentic data from real-world mcp environments. arXiv preprint arXiv:2510.01179, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Yao, Z., Xu, Z., Guo, Y., Han, Z., Yang, C., Zhang, S., Zhang, W., Zeng, X., and Liu, W. Toolace-mcp: Generalizing history-aware routing from mcp tools to the agent web. arXiv preprint arXiv:2601.08276, 2026.

Ye, J., Jiang, C., Du, Z., Xu, Y., Yao, X., Xi, Z., Fan, X., Zhang, Q., Gui, T., Huang, X., et al. Feedbackdriven tool-use improvements in large language models via automated build environments. arXiv preprint arXiv:2508.08791, 2025.

Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C., Wang, C., Yin, D., Zeng, H., Zhang, J., et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., et al. Siren’s song in the ai ocean: A survey on hallucination in large language models. Computational Linguistics, pp. 1–46, 2025.

###### Visualization of Tool Embeddings from Different Domains

tau2_airline

tau2_retail

tau2_telecom

vita_delivery

40

vita_instore

vita_ota

entertainment_media_query

wedding_planning

email_management

agriculture_environmental

stock_trading

20

pet_care

knowledge_management

social_media_messaging

online_learning

smart_home

express_logistics

0

football

job_seeking

second_hand_marketplace

project_management

healthcare_telemedicine

20

40

40 20 0 20 40

- Figure 4. Visualization of Tool Embeddings across Domains. We use t-SNE (Maaten & Hinton, 2008) to project the semantic embeddings of tools from our 16 synthesized training domains (circles) and the evaluation benchmarks (crosses and pluses). The clear spatial separation between the training clusters and the τ2 / Vita domains empirically demonstrates the OOD nature of our evaluation.

- Table 6. Tool Schema Example (Domain: Job Seeking). This table illustrates a subset of the tool action schema automatically generated by the meta-agent, detailing input requirements and output structures.

###### Tool Name Description Input Parameters Output Structure

Records notes and performance ratings after an interview is completed.

interview id* (str) feedback content* (str) performance rating (int) created at* (str: YYYY-MM-DD)

Object containing the new feedback id and confirmed interview id.

add interview feedback

Retrieves the complete schedule and details of interviews for a specific application.

application id* (str) List of interview objects (date,

get application interviews

type, interviewer, location).

Filters job applications based on keywords in the job title or company name.

keyword* (str) search fields (List[str])

List of matching application summaries and total match count.

search apps by keyword

Updates a job application record with the user’s expected salary range.

application id* (str) expected salary min* (num) expected salary max (num) salary currency (str)

Confirmation object with application id and success boolean.

add salary expectation

### A. Tool Semantic Diversity and OOD Verification

To further investigate the structural diversity of the synthesized “domain universe”, we visualize the semantic embeddings of all tools across both training and evaluation domains. We utilize a pre-trained encoder to extract tool descriptions’ embeddings and project them into a 2D space using t-SNE.

As illustrated in Figure 4, our 16 synthesized domains (represented by colored circles) form widely distributed clusters, covering a broad semantic spectrum from Smart Home to Healthcare Telemedicine. Crucially, the evaluation domains from τ2-Bench and VitaBench (represented by cross and plus markers) are situated in distinct regions, exhibiting significant semantic separation from the training clusters.

This visualization provides empirical evidence for the Out-Of-Distribution (OOD) nature of our evaluation setup. It

- Table 7. Database Schema Example (Domain: Job Seeking). This table illustrates a subset of the database schema automatically generated by the database agent. It defines entity structures, data types, and integrity constraints required to support the recruitment process tracking.

Field Name Data Type Constraint / Description Table: Job Application (Core application records)

application id VARCHAR(10) Primary Key. Unique identifier. applicant name VARCHAR(100) Full name of the applicant. job title VARCHAR(200) Title of the position applied for. company name VARCHAR(200) Name of the hiring company. status VARCHAR(50) Default: submitted. e.g., under review, rejected. priority level INTEGER Priority (1-5). Higher value indicates higher priority. salary currency VARCHAR(10) Default: USD. Currency for expectations. created at DATETIME Record creation timestamp.

Table: Application Stage (Tracks hiring progression)

stage id VARCHAR(10) Primary Key. Stage record identifier. application id VARCHAR(10) Foreign Key → job application.application id. stage name VARCHAR(100) e.g., phone screening, technical interview. stage date DATETIME Timestamp when the stage was reached. stage notes TEXT Optional notes regarding this stage.

Table: Interview Schedule (Manages interview logistics)

interview id VARCHAR(10) Primary Key. Schedule identifier. application id VARCHAR(10) Foreign Key → job application.application id. interview type VARCHAR(50) e.g., behavioral, system design. interview date DATETIME Scheduled date and time. interviewer name VARCHAR(100) Name of the assigned interviewer.

Table: Interview Feedback (Post-interview evaluation)

feedback id VARCHAR(10) Primary Key. Feedback identifier. interview id VARCHAR(10) Foreign Key → interview schedule.interview id. feedback content TEXT Detailed notes or feedback content. performance rating INTEGER Self-assessment score (e.g., 1-5 scale).

confirms that the performance gains reported in Section 5 are not achieved through simple memorization of tool templates or domain-specific logic interpolation. Instead, the model must rely on the generalized reasoning and tool-invocation strategies internalized from the diverse ScaleEnv training environments to succeed in these semantically novel tasks.

### B. Detailed Statistics of Synthesized Domains and Tasks

#### B.1. Visualization

We provide a comprehensive visualization of the 16 synthesized domains used in our training set. As illustrated in Figure 5, the domains exhibit significant structural diversity across three dimensions:

- • Action Space Scale (X-axis): The number of executable tools per domain ranges from approximately 25 (e.g., Online Learning) to over 70 (e.g., Entertainment Media Query), ensuring the agent learns to navigate varying sizes of action spaces.
- • State Space Complexity (Y-axis): The number of database tables ranges from 5 (e.g., Job Seeking) to 22 (e.g., Agriculture Environment), representing different levels of state-tracking difficulty.

[Figure 23]

- Figure 5. Structural statistics of the 16 domains synthesized. The x-axis and y-axis represent the number of tools and database tables, respectively. The color intensity and bubble size indicate the Graph Density of the Tool Dependency Graph, reflecting the complexity of inter-tool causal relationships within each domain.

• Dependency Density (Color/Size): The color intensity represents the Graph Density of the Tool Dependency Graph. Domains like Job Seeking and Knowledge Management exhibit higher density (darker colors), indicating more complex inter-tool causal relationships, whereas others like Agriculture Environment are relatively sparser.

#### B.2. Domain and Task Synthesis Cost

As detailed in Table 9, the synthesis of a complete domain environment requires approximately 546k tokens, while generating a single verifiable task consumes roughly 93.2k tokens.

### C. Scalable RL in Hybrid Environments

To cultivate generalist agents, we construct a unified training universe U = {(Bk,ψj) | ψj ∈ Tasks(Bk)}, where each instance pairs a specific synthesized domain environment Bk with a verifiable task ψj. We deploy an LLM-based user simulator, initialized with intent U from ψj, to generate natural language feedback Oresp, closing the multi-turn interaction loop.

Within this high-fidelity POMDP, we optimize the agent policy πθ using Group Relative Policy Optimization (GRPO) (Shao et al., 2024). For each task query q ∼ U, we sample a group of G trajectories {o1,...,oG} from the old policy πθ

. Recognizing the multi-step nature of tool use, we formulate the objective over time steps t as:

old

J (θ) = Eq∼U

Li

G

1 G

1 Li

t=1

i=1

min ρi,tAˆi,clip(ρi,t,1 − ϵ,1 + ϵ)Aˆi − βDKL(πθ(·|hi,t)||πref(·|hi,t)) , (1)

where Li is the trajectory length, and ρi,t = ππθ(ai,t|hi,t)

θold(ai,t|hi,t) denotes the step-wise importance sampling weight given history hi,t. Crucially, GRPO utilizes the group context to estimate the advantage Aˆi. By normalizing the trajectory reward ri against the group statistics, we obtain a stable baseline without a value network as:

ri − µ σ

Aˆi =

, (2)

where µ and σ represent the mean and standard deviation of the intra-group rewards {r1,...,rG}.

Table 8. Number of Tasks by Domain

#### Domain Name Number of Tasks

wedding planning 512 knowledge management 512

job seeking 256 healthcare telemedicine 256

pet care 128 smart home 128 email management 128 football 128

second hand marketplace 64 express logistics 64 social media messaging 64 entertainment media query 64 online learning 64 stock trading 64 project management 64 agriculture environmental 64

#### Total 2560

Table 9. Token Consumption Analysis. We report the average token usage for synthesizing a single domain foundation (including schemas and codes) and for generating a single verifiable task.

###### Synthesis Stage Prompt Tokens Completion Tokens Total Tokens

Domain Foundation Construction (per domain) ∼379k ∼167k ∼546k Task Generation (per task) ∼74.6k ∼18.6k ∼93.2k

### D. Examples of Tool and Database

This section provides illustrative examples of the synthesized domain foundations for the “Job Seeking” domain. Table 7 presents the structured database schema, while Table 6 defines the functional tool interfaces. The corresponding executable implementations for both the database and the tools are provided in the subsequent listings.

@with_instance_key("application_id") class JobApplication(BaseModel, ThreadSafeBase["JobApplication"]):

application_id: str = Field(..., description="Unique identifier for the job application") # primary key applicant_name: str = Field(..., description="Full name of the job applicant") email: str = Field(..., description="Email address of the applicant") phone: Optional[str] = Field(default=None, description="Phone number of the applicant" ) job_title: str = Field(..., description="Title of the job position being applied for") company_name: str = Field(..., description="Name of the company") application_date: datetime = Field(..., description="Date and time when the application was submitted") status: str = Field(default="submitted", description="Current status of the application (e.g., submitted, under_review, rejected, accepted, withdrawn, archived)") resume_content: Optional[str] = Field(default=None, description="Text content of the resume") resume_format: Optional[str] = Field(default=None, description="Format of the resume document (e.g., pdf, docx)") resume_uploaded_at: Optional[datetime] = Field(default=None, description="Timestamp when the resume was uploaded") cover_letter_content: Optional[str] = Field(default=None, description="Text content of

the cover letter") cover_letter_uploaded_at: Optional[datetime] = Field(default=None, description=" Timestamp when the cover letter was uploaded") deadline_date: Optional[datetime] = Field(default=None, description="Follow-up or response deadline for the application") deadline_type: Optional[str] = Field(default=None, description="Type of deadline (e.g

., follow_up, response)") referral_source: Optional[str] = Field(default=None, description="Source of the job referral (e.g., LinkedIn, Indeed, employee referral)") referral_person: Optional[str] = Field(default=None, description="Name of person who referred the applicant") priority_level: Optional[int] = Field(default=None, description="Priority level of the

application (1-5, where 5 is highest)") priority_reason: Optional[str] = Field(default=None, description="Reason for the assigned priority level") expected_salary_min: Optional[float] = Field(default=None, description="Minimum expected salary") expected_salary_max: Optional[float] = Field(default=None, description="Maximum expected salary") salary_currency: Optional[str] = Field(default="USD", description="Currency for salary

expectations (e.g., USD, EUR)") created_at: datetime = Field(..., description="Timestamp when the record was created") updated_at: Optional[datetime] = Field(default=None, description="Timestamp when the record was last updated")

@with_instance_key("note_id") class ApplicationNote(BaseModel, ThreadSafeBase["ApplicationNote"]):

note_id: str = Field(default=..., description="Unique identifier for the note") application_id: str = Field(default=..., description="Reference to the associated job application") note_content: str = Field(default=..., description="Content of the note or comment") note_type: Optional[str] = Field(default=None, description="Type or category of the note (e.g., follow_up, reminder, general)") created_at: datetime = Field(default=..., description="Timestamp when the note was created")

@with_instance_key("stage_id") class ApplicationStage(BaseModel, ThreadSafeBase["ApplicationStage"]):

stage_id: str = Field(default=..., description="Unique identifier for the stage record ") application_id: str = Field(default=..., description="Reference to the associated job application") stage_name: str = Field(default=..., description="Name of the application stage (e.g.,

phone_screening, technical_interview, final_interview)") stage_date: datetime = Field(default=..., description="Date and time when the stage was reached") stage_notes: Optional[str] = Field(default=None, description="Additional notes about the stage") # stage_notes is optional

@with_instance_key("interview_id") class InterviewSchedule(BaseModel, ThreadSafeBase["InterviewSchedule"]):

interview_id: str = Field(default=..., description="Unique identifier for the interview schedule") application_id: str = Field(default=..., description="Reference to the associated job application") interview_type: str = Field(default=..., description="Type of interview (e.g., phone_screening, technical_interview, behavioral_interview)") interview_date: datetime = Field(default=..., description="Scheduled date and time of the interview") interviewer_name: Optional[str] = Field(default=None, description="Name of the interviewer") interview_location: Optional[str] = Field(default=None, description="Location or platform for the interview (e.g., Zoom, office address)") interview_duration_minutes: Optional[int] = Field(default=None, description="Expected duration of the interview in minutes")

@with_instance_key("feedback_id") class InterviewFeedback(BaseModel, ThreadSafeBase["InterviewFeedback"]):

feedback_id: str = Field(default=..., description="Unique identifier for the interview

feedback") interview_id: str = Field(default=..., description="Reference to the associated interview") feedback_content: str = Field(default=..., description="Content of the feedback or notes") performance_rating: Optional[int] = Field(default=None, description="Self-assessment rating of interview performance (e.g., 1-5 scale)") created_at: datetime = Field(default=..., description="Timestamp when the feedback was

created")

class JobSeekingDB(DB): """Database containing all job_seeking_Job_Application-related data""" job_application: Optional[Dict[str, JobApplication]] = Field(

default=None, description="Schema JobApplication"

) application_note: Optional[Dict[str, ApplicationNote]] = Field(

default=None, description="Schema ApplicationNote"

) application_stage: Optional[Dict[str, ApplicationStage]] = Field(

default=None, description="Schema ApplicationStage"

) interview_schedule: Optional[Dict[str, InterviewSchedule]] = Field(

default=None, description="Schema InterviewSchedule"

) interview_feedback: Optional[Dict[str, InterviewFeedback]] = Field(

default=None, description="Schema InterviewFeedback"

)

Listing 1. Python implementation for database code in job seeking domain.

- # ==================== 1. delete_job_application ==================== def delete_job_application(self, application_id: str) -> dict:

""" Delete a job application from the system

Args:

application_id: Unique identifier of the application to delete Returns:

Dictionary containing:

- - application_id: Unique identifier of the deleted application

- - deletion_status: Status of the deletion operation

- - deleted_at: Timestamp when the application was deleted in yyyy-mm-dd HH:MM:SS

format Raises:

KeyError: If the application_id does not exist in the system

""" # Validate input parameter if not application_id or not isinstance(application_id, str):

raise ValueError("application_id must be a non-empty string")

# Access the database db = self.db

# Get the job_application table

job_application_table = getattr(db, ’job_application’, None) # Check if the table exists if job_application_table is None:

raise KeyError(f"Application with ID ’{application_id}’ does not exist in the system") # Check if the application exists in the table if application_id not in job_application_table:

raise KeyError(f"Application with ID ’{application_id}’ does not exist in the system") # Remove the application from the table updated_table = {k: v for k, v in job_application_table.items() if k != application_id } # Update the database table with the new dictionary setattr(db, ’job_application’, updated_table) # Generate deletion timestamp deleted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Return the deletion result return {

’application_id’: application_id, ’deletion_status’: ’deleted’, ’deleted_at’: deleted_at

}

- # ==================== 2. batch_update_application_status ==================== def batch_update_application_status(self, application_ids: list, new_status: str,

updated_at: str) -> dict: """ Update status for multiple applications at once

Args: application_ids: List of application identifiers to update new_status: New status to apply to all applications updated_at: Timestamp of the batch update in yyyy-mm-dd HH:MM:SS format

Returns: dict: Contains updated_count (number of successfully updated applications)

and failed_updates (list of application IDs that failed to update) Raises:

ValueError: If parameters are invalid or applications don’t exist

""" # Validate input parameters if not application_ids:

raise ValueError("application_ids cannot be empty") if not isinstance(application_ids, list):

raise ValueError("application_ids must be a list") if not new_status or not isinstance(new_status, str):

raise ValueError("new_status must be a non-empty string") if not updated_at or not isinstance(updated_at, str):

raise ValueError("updated_at must be a non-empty string")

# Parse and validate the updated_at timestamp try:

update_datetime = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S") formatted_updated_at = update_datetime.strftime("%Y-%m-%d %H:%M:%S")

except ValueError: raise ValueError("updated_at must be in ’yyyy-mm-dd HH:MM:SS’ format")

# Access the database db = self.db job_application_table = getattr(db, "job_application", None)

if job_application_table is None: raise ValueError("job_application table not found in database")

# Initialize tracking variables updated_count = 0 failed_updates = []

# Process each application ID for app_id in application_ids:

try:

# Check if the application exists if app_id not in job_application_table:

# Application doesn’t exist, add to failed list failed_updates.append(app_id) continue

# Retrieve the application record application = job_application_table[app_id]

# Update the status and updated_at timestamp (as string) application.status = new_status application.updated_at = formatted_updated_at

# Save the updated application back to the database job_application_table[app_id] = application

# Increment successful update count updated_count += 1

except Exception: # If any error occurs during update, add to failed list failed_updates.append(app_id) continue

# Update the database table setattr(db, "job_application", job_application_table)

# Return the results return {

"updated_count": updated_count, "failed_updates": failed_updates

}

- # ==================== 3. archive_old_applications ==================== def archive_old_applications(self, cutoff_date: str, archive_status: str = "archived") ->

dict: """ Archive applications older than a specified date by updating their status.

Args:

cutoff_date: Date before which applications should be archived in yyyy-mm-dd format

archive_status: Status to set for archived applications (default: ’archived’) Returns:

dict: Contains ’archived_count’ (number of applications archived) and ’archived_application_ids’ (list of archived application IDs)

Raises: ValueError: If cutoff_date format is invalid or if no job_application table exists

""" # Validate cutoff_date format try:

cutoff_datetime = datetime.strptime(cutoff_date, "%Y-%m-%d") except ValueError as e:

raise ValueError(f"Invalid cutoff_date format. Expected yyyy-mm-dd, got: {

cutoff_date}") from e # Access the database db = self.db

# Get the job_application table job_application_table = getattr(db, "job_application", None)

# Validate that the table exists if job_application_table is None:

raise ValueError("job_application table does not exist in the database")

# Initialize tracking variables archived_count = 0 archived_application_ids = []

# Iterate through all applications in the table for application_id, application in job_application_table.items():

# Check if application_date is before the cutoff_date if application.application_date.date() < cutoff_datetime.date():

# Update the application status to archive_status application.status = archive_status

# Update the updated_at timestamp to current time application.updated_at = datetime.now()

# Track the archived application archived_count += 1 archived_application_ids.append(application_id)

# Write the updated table back to the database setattr(db, "job_application", job_application_table)

# Return the results return {

"archived_count": archived_count, "archived_application_ids": archived_application_ids

}

Listing 2. Part of Python implementation for tool code in job seeking domain.

### E. Agent-User Interaction Trajectory in ScaleEnv Assistant [1] Hello, how can I help you?

#### User [2]

Hi, I’ve been applying to several energy analyst positions recently, and it’s getting a bit chaotic. Could you help me check the status of all the energy analyst-related positions I’ve applied for? I want to make sure I have a clear overview.

Assistant [3] Tool Calls: Tool 1: search applications by keyword Arguments: {”keyword”: ”energy analyst”, ”search fields”: [”job title”]}

#### Tool/Response [4]

{”matching applications”: [{”application id”: ”APP001”, ”job title”: ”Senior Energy Analyst”, ”company name”: ”PetroChina Group”}, {”application id”: ”APP002”, ”job title”: ”Energy Market Analyst”, ”company name”: ”CNOOC”}, {”application id”: ”APP003”, ”job title”: ”New Energy Strategic Analyst”, ”company name”: ”Envision Energy”}, {”application id”: ”APP004”, ”job title”: ”Data Analyst”, ”company name”: ”Tencent Technology”}, {”application id”: ”APP005”, ”job title”: ”Energy Investment Consulta...

Assistant [5] Tool Calls:

- Tool 1: get application stage history

- Arguments: {”application id”: ”APP001”}

Tool 2: get application stage history

- Arguments: {”application id”: ”APP002”}

Tool 3: get application stage history

- Arguments: {”application id”: ”APP003”}

Tool 4: get application stage history

- Arguments: {”application id”: ”APP004”}

Tool 5: get application stage history

- Arguments: {”application id”: ”APP005”}

- Tool 6: get application stage history

- Arguments: {”application id”: ”APP007”}

Tool 7: get application stage history

- Arguments: {”application id”: ”APP008”}

Tool 8: get application stage history

- Arguments: {”application id”: ”APP009”}

- Tool/Response [6] {”stages”: [{”stage id”: ”STAGE001”, ”stage name”: ”Resume screening passed”, ”stage date”: ”2024-02-25 15:00:00”, ”stage notes”: ”HR notified that the resume passed the initial screening, waiting for interview arrangement”}, {”stage id”: ”STAGE002”, ”stage name”: ”Waiting for technical interview”, ”stage date”: ”2024-03-10 10:00:00”, ”stage notes”: ”Received interview invitation email, confirmed interview time”}]}

- Tool/Response [7] {”stages”: [{”stage id”: ”STAGE003”, ”stage name”: ”Resume screening passed”, ”stage date”: ”2024-03-01 09:00:00”, ”stage notes”: ”Initial HR phone communication”}, {”stage id”: ”STAGE004”, ”stage name”: ”Phone Interview Completed”, ”stage date”: ”2024-03-08 10:30:00”, ”stage notes”: ”Performed well in the phone interview, awaiting next round notification”}]}

- Tool/Response [8]

- {”stages”: [{”stage id”: ”STAGE005”, ”stage name”: ”Resume Submitted”, ”stage date”: ”2024-03-01 09:20:00”, ”stage notes”: ”Resume submitted via official website”}]}

- Tool/Response [9]

- {”stages”: [{”stage id”: ”STAGE006”, ”stage name”: ”Resume Screening Passed”, ”stage date”: ”2024-02-20 14:00:00”, ”stage notes”: ”Received interview invitation”}, {”stage id”: ”STAGE007”, ”stage name”: ”Technical Interview Completed”, ”stage date”: ”2024-02-25 11:00:00”, ”stage notes”: ”Average interview performance”}, {”stage id”: ”STAGE008”, ”stage name”: ”Application Rejected”, ”stage date”: ”2024-02-28 10:30:00”, ”stage notes”: ”Received rejection letter, feedback was insufficient industry ...

- Tool/Response [10] {”stages”: [{”stage id”: ”STAGE009”, ”stage name”: ”Resume Screening Passed”, ”stage date”: ”2024-03-05 11:00:00”, ”stage notes”: ”Referral assisted with internal recommendation, HR responded quickly”}, {”stage id”: ”STAGE010”, ”stage name”: ”Phone Interview Completed”, ”stage date”: ”2024-03-16 14:45:00”, ”stage notes”: ”Initial communication was effective”}]}

- Tool/Response [11]

- {”stages”: [{”stage id”: ”STAGE013”, ”stage name”: ”Resume Submitted”, ”stage date”: ”2024-03-08 14:00:00”, ”stage notes”: ”Submitted via job portal”}]}

Tool/Response [12]

- {”stages”: [{”stage id”: ”STAGE014”, ”stage name”: ”Resume Submitted”, ”stage date”: ”2024-03-10 16:30:00”, ”stage notes”: ”Submitted through alumni referral”}]}

Tool/Response [13]

- {”stages”: [{”stage id”: ”STAGE015”, ”stage name”: ”Resume Submitted”, ”stage date”: ”2024-03-12 09:00:00”, ”stage notes”: ”Submitted via government recruitment website”}, {”stage id”: ”STAGE016”, ”stage name”: ”Resume Screening Passed”, ”stage date”: ”2024-03-18 15:00:00”, ”stage notes”: ”Received interview notification”}]}

Assistant [14] Here’s a clear overview of all your energy analyst-related applications:

— # — Position — Company — Current Status — Last Update — ——————-——————————-———

——-— — 1 — **Senior Energy Analyst** — PetroChina Group — Waiting for Technical Interview — 2024-03-10

— — 2 — **Energy Market Analyst** — CNOOC — Phone Interview Completed (awaiting next round) 2024-03-08 — — 3 — **New Energy Strategic Analyst** — Envision Energy — Resume Submitted (pending review) — 2024-03-01 — — 4 — **Data Analyst** — Tencent Technology — Rejected — 2024-02-28 — — 5 **Energy Investment Consultant** — CITIC Securities — Phone Interview Completed — 2024-03-16 — — 6 **Strategic Analysis Manager** — State Grid Corporation — Resume Submitted (pending review) — 2024-03-08

— —...

#### User [15]

Thanks for the summary! I’m particularly interested in the Senior Energy Analyst position at PetroChina. Their HR mentioned they can arrange a technical interview for me next Monday at 2 PM, at Zhongguancun Software Park in Haidian, with Manager Zhang as the interviewer, and the interview will last about an hour. Could you help me record this information and set a reminder for the interview?

#### Assistant [16] Tool Calls:

- Tool 1: add interview schedule Arguments: {”application id”: ”APP001”, ”interview date”: ”2024-03-18 14:00:00”, ”interview duration minutes”: 60, ”interview location”: ”Zhongguancun Software Park, Haidian”, ”interview type”: ”technical interv

- Tool 2: add application note

- Arguments: {”application id”: ”APP001”, ”created at”: ”2024-03-15 09:30:00”, ”note content”: ”INTERVIEW REMINDER: Technical interview scheduled for Monday, March 18, 2024 at 2:00 PM. Location: Zhongguancun Softw

Tool 3: calculate interview preparation time Arguments: {”current date”: ”2024-03-15 09:30:00”, ”interview date”: ”2024-03-18 14:00:00”}

- Tool/Response [17] {”interview id”: ”INT-0550001”, ”application id”: ”APP001”}

- Tool/Response [18] {”note id”: ”NOTE-2024-235720”, ”application id”: ”APP001”}

- Tool/Response [19] {”days until interview”: ”3”, ”hours until interview”: ”76”, ”is urgent”: ”False”}

Assistant [20]

I’ve recorded all the interview information and set a reminder for you. Here’s a summary: ### Interview Scheduled Successfully

— Detail — Information — ———–—————-— — **Position** — Senior Energy Analyst — — **Company** — PetroChina Group — — **Date & Time** — Monday, March 18, 2024 at 2:00 PM — — **Location** Zhongguancun Software Park, Haidian — — **Interviewer** — Manager Zhang — — **Duration** — ˜60 minutes — — **Interview Type** — Technical Interview — — **Interview ID** — INT-0550001 ### Preparation Time You have **3 days (about 76 hours)** to prepare for this interview - plenty of time to get ready! ### Reminder Note Added A reminder note has been added to your application with all the key details.

**Suggested Preparation Tips:** - Review PetroChina’s rece...

User [21]

Thanks for setting up the reminder! I also want to record my impressions of the interview I finished last week. The overall conversation was decent, and we had an in-depth discussion on industry trends, but I didn’t answer the specific model details very well and need to improve. I’d give myself a score of 4 for that interview. Could you help me document this feedback?

Assistant [22] Tool Calls: Tool 1: get application interviews

- Arguments: {”application id”: ”APP002”}

#### Tool/Response [23]

{”interviews”: [{”interview id”: ”INT002”, ”interview type”: ”phone screening”, ”interview date”: ”2024-03-08 10:00:00”, ”interviewer name”: ”Liu HR”, ”interview location”: ”Phone interview”}]}

Assistant [24] Tool Calls: Tool 1: add interview feedback Arguments: {”created at”: ”2024-03-15 09:30:00”, ”feedback content”: ”Overall the conversation went well. Had an in-depth discussion on industry trends which was productive. Area for improvement: Did not answer

#### Tool/Response [25]

{”feedback id”: ”FEEDBACK-2024-0EDA01”, ”interview id”: ”INT002”}

#### Assistant [26]

I’ve documented your interview feedback. Here’s a summary: ### Interview Feedback Recorded

— Detail — Information — ———–—————-— — **Position** — Energy Market Analyst — — **Company**

— CNOOC — — **Interview Date** — March 8, 2024 (Phone Screening) — — **Self-Rating** — 4/5 —

**Feedback Summary:** — Strengths — Areas for Improvement — ————–————————-— — Overall conversation went well — Specific model details need work — — In-depth discussion on industry trends — Need to strengthen technical modeling knowledge This feedback will be helpful for your upcoming PetroChina technical interview on Monday! You might want to spend some of your preparation time reviewing specific analytical models and technical details. Is there anything else you’d ...

#### User [27]

Thanks for recording the feedback. I’d like to analyze my job search efficiency during this period. Could you help me calculate my interview conversion rate and offer conversion rate? I want to see if I need to adjust my strategy based on these metrics.

#### Assistant [28] Tool Calls:

- Tool 1: calculate application roi Arguments: {”total applications”: 8, ”total interviews”: 5, ”total offers”: 0}

- Tool 2: calculate application success rate Arguments: {”rejected applications”: 1, ”successful applications”: 0, ”total applications”: 8}

- Tool/Response [29] {”interview conversion rate”: ”62.5”, ”offer conversion rate”: ”0.0”, ”interview to offer rate”: ”0.0”}

- Tool/Response [30] {”success rate”: ”0.0”, ”rejection rate”: ”12.5”, ”pending rate”: ”87.5”}

#### Assistant [31]

Here’s a comprehensive analysis of your job search efficiency: ### Job Search Performance Metrics #### Conversion Rates — Metric — Rate — Interpretation — ———–—————————-— — **Interview Conversion Rate** — **62.5%** — 5 out of 8 applications reached interview stage — — **Offer Conversion Rate** — 0% — No offers received yet — — **Interview → Offer Rate** — 0% — Still in progress #### Application Status Breakdown — Status — Count — Percentage — ———–———-—————— Pending/In Progress — 7 — 87.5% — — Rejected — 1 — 12.5% — — Offers — 0 — 0% —

### Analysis & Insights

**Strengths:** - **Strong resume performance** - Your 62.5% interview conversion rate is excellent! This means your resume and qualifications are well-aligned with energy an...

#### User [32]

Thanks for the detailed analysis! Given my current interview conversion rate and the feedback I received, I think I should focus on improving my technical modeling skills. Could you recommend any specific resources or practice methods that would be most effective for preparing for technical interviews in the energy industry?

#### Assistant [33]

I appreciate you asking, but recommending specific learning resources or study methods is outside the scope of what I can help with in this job application tracking system. My tools are focused on managing and tracking your job applications, interviews, and related activities. However, there are a few things I **can** help you with to support your preparation:

- 1. **Add a preparation note** to your PetroChina application with specific technical topics you want to review before Monday’s interview
- 2. **Set a follow-up reminder** to track your study progress
- 3. **Add action items** to your applications noting areas to improve based on interview feedback
- 4. **Review your application materials** - I could help you check if your resume highlights relevant technical/modeling skills Would any...

#### User [34]

Thanks for the suggestions! I’d like to add a preparation note to my PetroChina application, focusing on specific technical topics like energy market models, data analysis tools, and recent industry trends. Could you help me draft a concise note for this?

Assistant [35] Tool Calls: Tool 1: add application note

Arguments: {”application id”: ”APP001”, ”created at”: ”2024-03-15 09:30:00”, ”note content”: ”TECHNICAL INTERVIEW PREPARATION CHECKLIST:\{}n\{}n1. ENERGY MARKET MODELS:\{}n - Review supply/demand forecasting models\{}

#### Tool/Response [36]

{”note id”: ”NOTE-2024-AF1DEF”, ”application id”: ”APP001”}

#### Assistant [37]

I’ve added a detailed preparation note to your PetroChina application. Here’s what I documented: ### Preparation Note Added

**Application:** Senior Energy Analyst @ PetroChina Group **Note Type:** Action Item **Interview Date:** Monday, March 18, 2024 at 2:00 PM

### Technical Interview Preparation Checklist

— Topic — Key Focus Areas — ———-——————–— — **1. Energy Market Models** — • Supply/demand forecasting models¡br¿• Price elasticity & market equilibrium¡br¿• Examples of models you’ve worked with —

— **2. Data Analysis Tools** — • Excel (advanced functions, pivot tables)¡br¿• Python/R for data analysis¡br¿• Energy-specific software/databases — — **3. Recent Industry Trends** — • PetroChina’s latest projects & strategy¡br¿• Current oil & gas market conditions¡br...

#### User [38]

Thanks for the preparation note! I’d also like to set a follow-up reminder for my other pending applications, especially for Envision Energy, State Grid Corporation, and Procter & Gamble. Could you help me schedule these reminders?

#### Assistant [39] Tool Calls:

- Tool 1: generate follow up reminder Arguments: {”application date”: ”2024-03-01 09:20:00”, ”company name”: ”Envision Energy”, ”follow up date”: ”2024-03-18 10:00:00”, ”job title”: ”New Energy Strategic Analyst”}

- Tool 2: generate follow up reminder Arguments: {”application date”: ”2024-03-08 14:00:00”, ”company name”: ”State Grid Corporation of China”, ”follow up date”: ”2024-03-20 10:00:00”, ”job title”: ”Strategic Analysis Manager”}

- Tool 3: generate follow up reminder Arguments: {”application date”: ”2024-03-10 16:30:00”, ”company name”: ”Procter & Gamble”, ”follow up date”: ”2024-03-22 10:00:00”, ”job title”: ”Market Research Analyst”}

- Tool/Response [40] {”reminder message”: ”Follow up on your New Energy Strategic Analyst application at Envision Energy (applied on 2024-0301)”}

- Tool/Response [41] {”reminder message”: ”Follow up on your Strategic Analysis Manager application at State Grid Corporation of China (applied on 2024-03-08)”}

- Tool/Response [42] {”reminder message”: ”Follow up on your Market Research Analyst application at Procter & Gamble (applied on 2024-03-10)”}

Assistant [43] Tool Calls:

- Tool 1: add application note

Arguments: {”application id”: ”APP003”, ”created at”: ”2024-03-15 09:30:00”, ”note content”: ”FOLLOW-UP REMINDER: Contact Envision Energy on Monday, March 18, 2024 to check application status. It will be 17 days

- Tool 2: add application note

- Arguments: {”application id”: ”APP007”, ”created at”: ”2024-03-15 09:30:00”, ”note content”: ”FOLLOW-UP REMINDER: Contact State Grid Corporation on Wednesday, March 20, 2024 to check application status. It will

Tool 3: add application note

- Arguments: {”application id”: ”APP008”, ”created at”: ”2024-03-15 09:30:00”, ”note content”: ”FOLLOW-UP REMINDER: Contact Procter & Gamble on Friday, March 22, 2024 to check application status. It will be 12 day

- Tool 4: set application deadline Arguments: {”application id”: ”APP003”, ”deadline date”: ”2024-03-18 10:00:00”, ”deadline type”: ”follow up”}

- Tool 5: set application deadline

- Arguments: {”application id”: ”APP007”, ”deadline date”: ”2024-03-20 10:00:00”, ”deadline type”: ”follow up”}

Tool 6: set application deadline

- Arguments: {”application id”: ”APP008”, ”deadline date”: ”2024-03-22 10:00:00”, ”deadline type”: ”follow up”}

- Tool/Response [44] {”note id”: ”NOTE-2024-369CF6”, ”application id”: ”APP003”}

- Tool/Response [45] {”note id”: ”NOTE-2024-45E718”, ”application id”: ”APP007”}

- Tool/Response [46] {”note id”: ”NOTE-2024-B579B8”, ”application id”: ”APP008”}

- Tool/Response [47] {”application id”: ”APP003”, ”deadline set”: ”True”}

- Tool/Response [48]

- {”application id”: ”APP007”, ”deadline set”: ”True”}

Tool/Response [49]

- {”application id”: ”APP008”, ”deadline set”: ”True”}

#### Assistant [50]

I’ve set up follow-up reminders and deadlines for all three pending applications. Here’s your follow-up schedule: ### Follow-Up Reminders Scheduled

— # — Company — Position — Applied On — Follow-Up Date — Days Since Applied — —————————-———————————-————————— — 1 — **Envision Energy** — New Energy Strategic Analyst — Mar 1 — **Mon, Mar 18** — 17 days — — 2 — **State Grid Corporation** — Strategic Analysis Manager Mar 8 — **Wed, Mar 20** — 12 days — — 3 — **Procter & Gamble** — Market Research Analyst — Mar 10

— **Fri, Mar 22** — 12 days —

### Your Week Ahead (Mar 15-22)

— Date — Action Items — ————————–— — **Mon, Mar 18** — Technical Interview @ PetroChina (2:00 PM) Follow up with Envision Energy — — **We...

#### User [51]

Thanks for setting up the follow-up reminders! I think that covers everything I needed for now. If I have any more updates or need further assistance, I’ll reach out.

#### Assistant [52]

You’re welcome! Here’s a quick summary of everything we accomplished today: ### Session Summary

— Task — Status — ——————–— — Reviewed all 8 energy analyst applications — Complete — — Scheduled PetroChina technical interview (Mar 18, 2:00 PM) — Complete — — Added interview reminder for PetroChina — Complete — — Recorded CNOOC interview feedback (Rating: 4/5) — Complete — — Analyzed job search metrics (62.5% interview conversion rate) — Complete — — Added technical preparation checklist for PetroChina

— Complete — — Set follow-up reminders for 3 pending applications — Complete ### Upcoming Reminders - **Mon, Mar 18** - PetroChina Interview + Envision Energy follow-up - **Wed, Mar 20** - State Grid Corporation follow-up - **Fri, Mar 22** - Procter & Gamble foll...

