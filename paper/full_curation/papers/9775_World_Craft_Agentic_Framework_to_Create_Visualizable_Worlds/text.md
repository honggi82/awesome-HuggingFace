# arXiv:2601.09150v4[cs.HC]29Jan2026

## World Craft: Agentic Framework to Create Visualizable Worlds via Text

Jianwen Sun 1,2,3∗ Yukang Feng 1,2∗ Kaining Ying 4∗ Chuanhao Li 7 Zizhen Li 1,2,3 Fanrui Zhang 2 Jiaxin Ai 4,2 Yifan Chang 2 Yu Dai 3 Yifei Huang 1 Kaipeng Zhang 1,2†

1 Shanda AI Research 2 Shanghai Innovation Institute 3 Nankai University 4 Fudan University 5 Wuhan University 6 Shanghai AI Laboratory jianwen.sun@shanda.com, kaipeng.zhang@shanda.com† Project Page: https://github.com/HerzogFL/World-Craft

### Abstract

Large Language Models (LLMs) motivate generative agent simulation (e.g., AI Town) to create a “dynamic world”, holding immense value across entertainment and research. However, for non-experts, especially those without programming skills, it isn’t easy to customize a visualizable environment by themselves. In this paper, we introduce World Craft, an agentic world creation framework to create an executable and visualizable AI Town via user textual descriptions. It consists of two main modules, World Scaffold and World Guild. World Scaffold is a structured and concise standardization to develop interactive game scenes, serving as an efficient scaffolding for LLMs to customize an executable AI Town-like environment. World Guild is a multi-agent framework to progressively analyze users’ intents from rough descriptions, and synthesizes required structured contents (e.g.environment layout and assets) for World Scaffold . Moreover, we construct a high-quality error-correction dataset via reverse engineering to enhance spatial knowledge and improve the stability and controllability of layout generation, while reporting multidimensional evaluation metrics for further analysis. Extensive experiments demonstrate that our framework significantly outperforms existing commercial code agents (Cursor and Antigravity) and LLMs (Qwen3 and Gemini-3-Pro). in scene construction and narrative intent conveyance, providing a scalable solution for the democratization of environment creation.

### 1 Introduction

AI Town represents a novel form of entertainment (Gong et al., 2024; Sudhakaran et al., 2023) and social simulation (Yao et al., 2023; Wang et al., 2023c; Xi et al., 2023), providing an ideal environment for observing complex emergent behaviors of agents. However, the construction of such environments still faces obstacles (Li et al., 2024). Existing development workflows often rely on preset maps,

[Figure 1]

Figure 1: An illustration of our motivation and goal.

suffer from fragmented toolchains and lack unified standards. They typically require users to possess professional programming skills (e.g., Unity or Godot) (Yang et al., 2024; Lin et al., 2023), posing a high barrier for those without programming backgrounds, limiting widespread adoption (Xie et al., 2023). To address this, we face two main challenges: (1) The toolchains of traditional game engines are highly fragmented and complex to operate, lacking unified interfaces. This makes it difficult for AI agents to directly invoke low-level APIs for environment creation. (2) Human language is highly ambiguous. It is extremely challenging to directly model the precise layout content required for game environment construction from vague text descriptions. To this end, we introduce World Craft as in Fig.1, comprising two subsystems: World Scaffold and World Guild . First, World Scaffold serves as the infrastructure that automatically constructs executable game scenes from structured content, thereby accelerating the creation process and significantly lowering the entry barrier.

Although World Scaffold bridges the underlying protocol, achieving fully automated construction requires relying on general LLMs to drive this process. However, addressing the second challenge encounters a fundamental obstacle: there is a significant semantic gap between abstract human narrative intents and the precise spatial instructions required for environment creation (Tan et al., 2018; Paschalidou et al., 2021; Feng et al., 2023). Lacking embodied perception and precise spatial layout capabilities (Valmeekam et al., 2023; Bisk et al., 2020), general LLMs often yield designs plagued by “physical hallucinations” such as floating objects or blocked paths (Tang et al., 2023; Wu et al., 2023). Inspired by research on Chain-of-Thought and modular reasoning (Li et al., 2025; Shen et al.,

- 2023), we propose the World Guild multi-agent framework. To mitigate the impact of the semantic gap, it decouples intent analysis from spatial planning, transforms complex cross-modal generation into a controllable step-by-step reasoning process, effectively improves the performance of LLMs, and leverages our built asset library to ensure the visual and physical consistency of the final output.

While World Guild mitigates the impact of the semantic gap, general LLMs still face bottlenecks under complex geometric constraints due to a lack of spatial commonsense (Stogiannidis et al., 2025; Jia et al., 2024). Addressing the scarcity of highquality layout data (Brazil et al., 2022), we utilize a “Reverse Synthesis” data construction paradigm. Instead of relying on expensive fully manual annotation (Zhou et al., 2023), this method leverages “Golden Layouts” constructed via procedural algorithms, model verification, and minimal human correction. By applying reverse semantic restoration and controlled “intentional corruption,” it synthesizes full-chain supervision signals covering “semantic mapping,” “generation from scratch,” and “error correction,” thereby injecting key spatial reasoning and correction capabilities into the model (Shinn et al., 2023; Gou et al., 2024). Combined with our proposed multi-dimensional evaluation benchmark (Liu et al., 2025), our method demonstrates superior performance in logical correctness and intent conveyance.

Our contributions are summarized as follows:

• We propose World Craft, a framework integrating World Scaffold and World Guild which enables creating an interactive AI Town-like environments from natural language.

- • World Scaffold is a flexible and standardized scaffold for LLMs to customize game environments. World Guild mitigates the semantic gap from users’ rough description to structured world and synthesize assets through multi-agent collaboration with step-by-step reasoning and our introduced high-quality asset library.
- • We establish multi-dimensional evaluation metrics and construct a high-quality dataset which can effectively fill the knowledge gap of LLMs in complex spatial reasoning. Extensive experiments demonstrate that our framework significantly outperforms existing commercial code agents and LLMs.

### 2 Related Works

Generative Agents. Represented by Generative Agents (Park et al., 2023), the “AI Town” research initiated a wave of behavioral simulation. Subsequent works like Concordia (Mao et al., 2025), AgentVerse (Chen et al., 2024b), and CAMEL (Li et al., 2023) expanded the scope to group evolution. However, compared to advancements in agent memory and planning mechanisms, environment construction remains lagging: existing methods mostly rely on unmodifiable pre-built maps (e.g., Minecraft (Wang et al., 2023a; Zhu

- et al., 2023), 2D grids (Park et al., 2022)) or textbased sandboxes lacking physical properties (Zhou
- et al., 2024). Furthermore, existing open-source projects (Microverse, or TinyTroupe (Salem et al.,
- 2025))are built on different engines (e.g., Unity or Godot). This fragmentation of technical stacks significantly raises the barrier. Therefore, developing a standardized scenario construction tool is crucial for promoting the popularization of this field.

Layout Generation. Works such as HouseGAN++, HouseDiffusion, FloorPlan-LLaMa and others (Nauata et al., 2021; Shabani et al., 2022; Yin et al., 2025; Leng et al., 2023) have demonstrated capabilities in topological layout generation, while SceneCraft (Hu et al., 2024) and 3D-GPT (Sun et al., 2025) explored text-to-3D visual synthesis. Unlike these studies that focus on visual or geometric aspects, this paper is dedicated to generating structured layouts with complete functional logic. However, as noted by Mind’s Eye (Liu et al., 2022), Spatial-VLM (Chen et al., 2024a), and PlanQA (Rodionov et al., 2025), general LLMs lack embodied perception and suffer from a “semantic gap” when mapping abstract language to physical con-

[Figure 2]

- Figure 2: Architecture of WorldCraft. It comprises the World Guild for intent analysis and layout generation, and the World Scaffold for automated scene construction.

straints. Consequently, end-to-end generation models struggle to ensure the correctness of spatial logic. To address this, we introduce a multi-agent collaboration mechanism to decouple intent parsing from spatial planning, significantly reducing generation difficulty through stepwise reasoning.

where M (Metadata) defines the overall scene style and grid size; A (Assets) describes the visual style and layer attributes; L (Layout) records the precise spatial coordinates of components; and P (Properties) specifies interaction properties. Given that I typically implies ambiguous narrative intents while G demands determinate geometric parameters and physical properties, directly modeling P(G|I) faces a significant semantic gap. For this, we introduce an intermediate variable Z as a semantic bridge-representing scene topology and functional distribution without specific coordinates. By decomposing the generation objective into:

Knowledge Enhancement. LLMs still suffer from a knowledge deficit when handling complex spatial layouts and geometric constraint (Sun et al.,

- 2024). To address this lack of domain knowledge, mainstream methods employ RAG (Lewis et al., 2020; Guu et al., 2020) or utilize instruction finetuning (Ouyang et al., 2022; Wang et al., 2023b) to align task distributions. However, in the layout domain, existing open-source datasets primarily focus on static visual representations (Deitke et al., 2023; Fu et al., 2020), lacking high-quality “instructionlayout” pairs (Hong et al., 2023). Furthermore, studies (Madaan et al., 2023; Chen et al., 2024c) indicate that iterative correction capability is equally critical for resolving complex constraints.To address this data scarcity, we utilize a reverse data construction method to generate a large-scale corpus containing correction trajectories, and adopt a two-stage training strategy to equip the model with professional layout planning and self-correction capabilities (Wei et al., 2022). 3 Method

P(Z|I) · P(G|Z), (2)

P(G|I) =

Z

we achieve a logical decoupling of parsing intents before grounding parameters.

#### 3.2 Collaborative Multi-Agent Framework

To effectively solve the aforementioned decomposition process, we propose the World Guild framework. This framework introduces a multiagent collaboration mechanism designed to alleviate the semantic gap inherent in direct modeling by decoupling intent parsing from spatial planning. Through step-by-step reasoning, it transforms the direct cross-modal mapping into a series of executable logical operations, thereby significantly reducing generation difficulty. World Guild consists of four core agents: Semantic Enrichment (Enricher), Layout Generation (Manager), Quality Assurance (Critic), and Asset Synthesis (Artist). As shown in Fig. 2, the detailed functional descriptions of each agent are as follows.

- 3.1 Problem Formulation We define text-based game scene design as a mapping from natural language instruction I to structured layout G. It is formalized as a quadruple (see Appendix H for examples):

Semantic Enrichment The Enricher is responsible for transforming the user instruction I into

G = (M,A,L,P) (1)

a layout description Z endowed with spatial logic. Since user inputs often exhibit significant disparities in information density-ranging from sparse keywords to abstract descriptions that are difficult to ground, the Enricher needs to concretize narrative intents into a coherent scene topology, explicitly defining connectivity and the rough distribution of core components. This process does not involve specific coordinate calculations but focuses on constructing a spatial sketch that is consistent with common sense and logically self-consistent, thereby eliminating ambiguity at the semantic level and providing guidance for the subsequent precise design by the Manager.

Constrained Layout Generation The Manager is responsible for executing the grounding process of P(G|Z). It receives the layout description Z from the Enricher and converts it into an initial layout file G0 that conforms to physical definitions. As the core planning agent, the Manager’s function is to parse the topological logic and relative positional constraints contained in the natural language and map them into quantitative, precise geometric parameters. Specifically, guided by Z, it determines the scene metadata M, instantiates the asset library A and property set P, and designs the grid coordinates and orientation for each component in the layout layer L. This outputs an initial layout file with a complete hierarchy and asset attributes, achieving the cross-modal transformation from abstract text descriptions to executable data.

Iterative Critique and Refinement To ensure the generated results meet physical and logical constraints, we introduce the Critic to establish an iterative feedback loop. In the t-th iteration, the Critic performs rule-based physical checks (such as collision and connectivity detection) and modelbased semantic evaluations on the current layout Gt, generating specific correction instructions Ct. If defects are detected (i.e., Ct ̸= ∅), the Manager executes targeted spatial editing operations (such as moving object coordinates or replacing assets) based on these instructions to generate a corrected layout Gt+1. This process continues until all checks are passed or the maximum number of rounds Tmax is reached, ensuring the rationality and logical selfconsistency of the final output layout.

Reference-Guided Asset Synthesis The Artist is responsible for transforming the asset definition set A within the layout design G into visual assets. To address the common issue of style fragmenta-

tion in pure text-to-image generation, we employ a retrieval-augmented texture synthesis strategy: for each component, the Artist first retrieves a reference image vref from the pre-built library Dlib (see Appendix C for library examples and algorithm details). Using this as a style anchor to guide the generative model, it produces Tile resources that possess a unified visual style while maintaining semantic accuracy. Finally, the World Scaffold automatically assembles the generated visual resources with the layout layer L and property set P, constructing a playable game scene complete with navigation meshes and interaction logic.

#### 3.3 Data Construction

Although World Guild mitigates the impact of the semantic gap, LLMs still face performance bottlenecks under complex geometric constraints due to a lack of spatial commonsense (Wang et al., 2024; Xu et al., 2025; Fu et al., 2024). To equip LLMs with professional layout planning and logical correction capabilities, we designed a data construction pipeline comprising three stages: first, Scenario Initialization establishes the diversity of scene configurations; then, Scene Design combines procedural rules and verification to construct the golden layout Ggold; finally, Data Annotation applies controlled degradation to them to generate fine-tuning data with complete correction trajectories.

Scenario Initialization To ensure data coverage and generalization, we constructed a base scenario library spanning four dimensions: real-world, literature, film, and games. We selected 125 seed scenarios per category, partitioned into training and held-out test sets with a 4:1 ratio to prevent data leakage. For the training set, we established a prompt pool containing 560 style descriptions and randomly injected 5 variants (e.g., “Cyberpunk”, “Primitive”) into each scenario, expanding the dataset to 2,000 samples. This strategy aims to enhance the model’s spatial logic robustness across cross-domain scenarios by leveraging highly diverse semantic atmospheres. Details of the scenario data are provided in Fig. 3 and the Appendix G.

Scene Design To construct the golden layout Ggold satisfying strict physical constraints, we designed an offline generation pipeline with multistage verification. First, procedural algorithms(see Appendix D) generate empty room structures; then, LLM assigns specific functional attributes based on scene descriptions. During the filling stage, to

[Figure 3]

- Figure 3: Two-stage fine-tuning data construction process. Utilizing Gemini-3-Pro as all the agents, we perform 10 runs for each of scenario descriptions. During the filtering process, approximately 5k invalid samples are discarded, and 1.2k long-tail samples undergo human rectification, resulting in a final dataset of approximately 14k samples.

overcome the spatial perception deficits of LLMs, we introduced the “12-zone grid” strategy to assist relative orientation generation. This strategy partitions each room into left-center-right visible walls and an internal 9-grid, guiding the model to output component coordinates based on relative orientations, while coordinating with a “Physical Placer” to eliminate collision conflicts in real-time. Finally, a Teacher Model equipped with editing tools is employed to automatically review and refine the layout room-by-room, supplemented by human experts for long-tail samples, thereby ensuring the logical and physical rigor of all Ggold data.

where ρ ∈ {short,medium,long} denotes the simulated instruction density. Fig. 3 illustrate the complete construction flow and data distribution details.

#### 3.4 Training Strategy

Based on the aforementioned high-quality datasets, we adopt a decoupled two-stage fine-tuning strategy to specifically optimize semantic understanding and spatial execution capabilities.

Semantic Alignment. The first stage aims to endow the Enricher (parameterized by θE) with intent normalization capabilities. Utilizing dataset DB, we establish a deterministic mapping from arbitrary natural language I to standard layout descriptions Z by maximizing the conditional likelihood of semantic tokens:

Data Annotation Based on the constructed golden layouts Ggold, we first utilize a LLM to reverse-engineer them into coordinate-free layout descriptions Z, serving as a unified semantic foundation. Subsequently, we introduce a “Chaos Monkey” (perturbation agent) to execute four levels of controlled destruction with weights of 1:2:3:4 (e.g., components exchange or creating collisions), generating error samples containing 2 to 15 issues along with correction instructions: (Φ(Ggold) → Gerror,C). On this basis, we define two core datasets. First, we construct Dataset A:

L(SFTE)(θE) = − E(I,Z)∼DB

(5)

|Z|

log PθE(zt | I,z<t).

t=1

Under this objective, by mixing instruction data of varying densities, the model acquires robust normalization capabilities: it can perform commonsense logical completion for sparse instructions while extracting key topological information from verbose descriptions.

DA = {Z →Ggold} ∪ {(Gerr,C)→Ggold}, (3)

which records the trajectory from generating initial layouts via Z to iteratively repairing Gerror into Ggold using C. Secondly, we constructed Dataset B by simulating users rewriting Z into natural language instructions I of three densities (short, medium, and long), forming paired data mapping user inputs to layout descriptions dedicated:

Spatial Refinement. The second stage is focusing on enhancing the Manager’s (parameterized by θM) spatial planning and dynamic correction capabilities. Based on dataset DA, we unify the tasks of “initial generation from Z” and “correction based on C” into a sequence prediction format. Defining the input context as X ∈ {Z,(Gerror,C)}, the

DB = {(I,Z) | I ∼ Simuser(Z,ρ)}, (4)

optimization objective is:

L(SFTM)(θM) = − E(X,Ggold)∼DA

(6)

|G|

log PθM(gt | X,g<t).

t=1

This strategy not only enables the model to master the logic of converting layout descriptions into quadruplets G but also endows it to respond to correction instructions C. Consequently, the model can execute precise editing operations upon receiving negative feedback from the Critic to optimize error states. Specific training settings and hyperparameter configurations are detailed in the Experiments section and Appendix B.

### 4 Experiments

#### 4.1 Experimental Setup

Implementation Details and Baselines. Following the proposed two-stage training strategy, we fine-tuned the Qwen3 series (8/32B) (Yang et al., 2025) open-source models, utilizing different model sizes to explore the optimal balance between performance and efficiency. Given the lack of specialized models for such structured spatial reasoning tasks, we focused the comparison on generalpurpose LLMs: we selected Qwen3-235B as the performance upper bound of open-source models (Open SOTA) and Gemini-3-Pro to represent the peak level of closed-source commercial models (Closed SOTA), thereby establishing a widely representative and fair comparison baseline.

Evaluation Datasets and Metrics. To ensure objective evaluation, we constructed a manually annotated test set derived from the scene library in Section 3.3. Specifically, we selected 100 “heldout” seeds (25 each from Reality, Literature, Film, and Games) strictly excluded from training. We employed an LLM to generate instructions of three complexity levels (Short, Medium, Long) for each seed, followed by expert refinement, yielding 300 test samples. We evaluate eight core metrics across three dimensions: Layout Design via CollisionFree Rate (CFR), Room Connectivity Score (RCS), and Object Placement Score (OPS); Element Design via Component Existence Rate (CER), Object Volume Density (OVD), and Property Consistency (PAC); and Intent Alignment using VSA-C (CLIP) and VSA-V (VLM) to verify visual-semantic consistency. The detailed prompt for each metric can be found in the Appendix F.

#### 4.2 Main Results

Framework Design. The upper part of Table 1 validates the effectiveness of the stepwise reasoning design. First, the necessity of the Critic module is confirmed: introducing the Critic to Direct Gen. baseline yields significant improvements in layout design. Second, the effectiveness of stepwise reasoning (Enricher+Manager) is further demonstrated: with the introduction of stepwise reasoning, the model achieves notable gains in metrics such as RCS, OPS, and OVD. This indicates that decoupling the complex generation task into two sub-steps-“semantic completion” and “spatial management”-allows each module to focus on specific tasks, thereby verifying the rationality of the framework design.

Data and Training Strategy. The lower part of Table 1 validates the rationality of our training strategy. First, decoupled training outperforms endto-end fine-tuning, demonstrating the necessity of separating semantic and spatial tasks. Second, the (8+32)B combination outperforms the (8+8)B version in multiple aspects, indicating that the spatial planning task demands higher model capacity. Finally, we observe that when trained solely on standard data, the model possesses generation capabilities but struggles to make correct revisions based on feedback. In contrast, models trained on correction data benefit significantly during the correction phase. This proves the effectiveness of “error-correction”.

#### 4.3 Fine-grained Analysis

Instruction Robustness. To analyze model performance under different input conditions, we conducted a fine-grained breakdown of the data in Table 1. Fig. 4 records the results for the four models marked with “*” across three instruction lengths. The data shows that general models exhibit significant performance fluctuations across different instruction lengths, indicating a struggle to cope with varying information densities and generate stable results. In contrast, our method maintains stability and infers reasonable layouts. This proves that our training strategy successfully establishes a mapping from abstract instructions to layout descriptions, granting the model robustness in handling instruction ambiguity.

Correction Trajectory. To explore the dynamic process of iterative critique and refinement, we

###### Methods CFR ↑ RCS ↑ OPS ↓ CER ↑ OVD ↑ PAC ↓ VSA-C ↑ VSA-V ↑

Part 1: Effectiveness of the Inference Framework Design

Base: Qwen3-32B 0.59 0.44 9.27 0.76 3.27 10.55 20.22 3.73 Open: Qwen3-235B 0.73 0.60 7.18 0.80 4.09 7.22 23.34 4.74 Closed: Gemini-3-Pro 0.80 0.59 5.56 0.86 5.02 6.87 23.29 4.67

Direct Gen. (Few-shot)

Base: Qwen3-32B 0.66 0.49 8.68 0.78 3.43 9.31 20.24 3.88 Open: Qwen3-235B 0.78 0.67 6.88 0.83 4.37 6.17 23.73 5.03 Closed: Gemini-3-Pro 0.84 0.67 4.99 0.90 5.30 5.92 23.51 4.98

Direct Gen.

+ Critic

Base: Qwen3-(8+32)B∗ 0.66 0.52 7.47 0.77 4.46 9.33 21.45 4.03 Open: Qwen3-235B∗ 0.81 0.71 4.72 0.84 5.28 6.13 24.72 5.66 Closed: Gemini-3-Pro ∗ 0.83 0.68 3.77 0.92 5.92 5.90 25.08 5.71

Enricher

+Manager

+Critic

Part 2: Effectiveness of Data and Training Strategies

End-to-End Fine-tuning 0.86 0.78 5.36 0.87 6.12 4.81 24.83 5.33 Qwen3-(8+8)B 0.83 0.73 4.64 0.90 6.35 5.42 25.31 5.64 Qwen3-(8+32)B 0.88 0.79 3.93 0.92 7.10 4.76 26.36 6.17

Training Variants (w/o Critic)

- Stage 1 Only (Enricher) 0.63 0.46 8.16 0.76 3.97 9.88 25.22 5.22
- Stage 2 Only (Manager) 0.89 0.78 4.11 0.91 6.84 4.76 22.16 4.34

Two-Stage

w/ Standard Data† 0.89 0.81 3.67 0.94 7.07 4.39 26.83 6.04 w/ Correction Data(Ours)∗† 0.94 0.88 3.03 0.99 7.13 3.64 28.07 6.80

+ Critic

- Table 1: Experimental results of scene generation. The top and bottom sections validate the framework design and the data/training strategies, respectively. The critic is based on GPT-5.1 with max rounds T = 4. Note that for VSA-C, evaluation is strictly limited to samples with fewer than 77 tokens due to CLIP’s length constraint.

Dim 1: Layout Rationality Dim 2: Element Richness Dim 3: Visual Consistency Method CFR ↑ RCS ↑ OPS ↓ HWR ↑ CER ↑ OVD ↑ PAC ↓ HWR ↑ VSA-C ↑ VSA-V ↑ HWR ↑

Base: Qwen3-(8+32)B 0.66 0.52 7.47 0.28 0.77 4.46 9.33 0.19 21.45 4.03 0.38 Open: Qwen3-235B 0.81 0.71 4.72 0.47 0.84 5.28 6.13 0.45 24.72 5.66 0.46 Closed: Gemini-3-Pro 0.83 0.68 3.77 0.46 0.92 5.92 5.90 0.53 25.08 5.81 0.47 Ours 0.94 0.88 3.03 0.78 0.99 7.13 3.64 0.82 28.07 6.80 0.69

Reliability r = 0.95 r = 0.96 r = −0.86 κ = 0.61 r = 0.97 r = 0.99 r = −0.98 κ = 0.54 r = 0.98 r = 0.91 κ = 0.65

- Table 2: Human Evaluation and Metric Correlation. Comparison between automated metrics and Human Win Rate (HWR) across three dimensions. The strong Pearson correlation (|r|) and substantial Fleiss’ Kappa (κ) validate that our automated metrics are reliable proxies for human preference.

conducted a fine-grained breakdown of the data in Table 1. Fig. 5 records the result changes across correction rounds T = 0,1,...,4 for the two models marked with † in the table. Results: The model trained using only standard data, despite decent initial performance, shows a flat metric improvement during the multi-round correction process. In contrast, the model trained on correction data exhibits a robust growth trend, particularly in spatial layout metrics. This indicates that correction data is crucial for the model to correctly understand and execute modification instructions, ensuring the effectiveness of iterative optimization.

#### 4.4 Human Evaluation and Metric Validation

To verify whether the metrics accurately reflect human perception of generation quality, we organized a subjective evaluation with 5 game players. The experiment adopted a pairwise forced-choice for-

mat, covering the four models marked with “*” in Table 1, with 150 instructions randomly sampled from the test set (50 each for short, medium, and long). We summarized the eight metrics into three questions for the evaluators.(The content of the questions and details of the reviewers are provided in the Appendix E).We used the Pearson coefficient |r| to report the consistency between metrics and human evaluation, and Fleiss’ κ to report InterAnnotator Agreement. As shown in Table 2, the results demonstrate a strong correlation between our metrics and human preference (mean Pearson’s |r|>0.90), and evaluators reached substantial agreement (mean κ = 0.60). Specifically, the highest consensus was reached on Visual Consistency (κ = 0.65), while slight divergence was observed in Element Richness (κ = 0.54). Overall, the results robustly validate the scientific validity and reliability of the proposed automated evaluation system.

Ours Closed Open Base

###### CFR

###### RCS

###### OPS

###### CER

###### OVD

PAC

8

1.0

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

10.0

1.0

8

0.8

6

7.5

0.9

0.8

6

0.6

5.0

0.8

4

4

0.6

0.4

2

0.7

2.5

Short Medium Long

Short Medium Long

Short Medium Long

Short Medium Long

Short Medium Long

Short Medium Long

Figure 4: Results of fine-grained comparison on performance stability under different input lengths in the test set.

w/ Standard Data w/ Correction Data

###### CFR

###### RCS

###### OPS

###### CER

###### OVD

PAC

- 0.95

1.00

4.0

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

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
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

7.125

4.5

0.85

0.95

3.5

0.90

7.100

4.0

0.80

7.075

0.90

3.0

0.85

3.5

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

Turns

Turns

Turns

Turns

Turns

Turns

Figure 5: Dynamic changes in model output quality during multi-round correction processes.

Methods TR (min) ↓ TS (min) ↓ HWR ↑ VWR ↑

Cursor 15.42 48.15 0.23 0.35 Antigravity 9.83 42.40 0.35 0.39 Ours 4.25 4.25 0.92 0.76

- Table 3: Comparison with Code Agents. HWR/VWR: Human/Visual evaluation win rates. Tile generation: ∼20s/image (Nano Banana Pro, 8 threads).

#### 4.5 Comparison with Code Agents

To verify the superiority of our framework (World Guild ) in creating generative agent simulation environments, we compared it with general code agents (Cursor and Antigravity) using the same Gemini3-Pro. Setup: Three operators tested 15 prompts of different lengths (5 short, 5 medium, 5 long). General agents allowed multi-turn human debugging (max 60 mins), recording Time-to-Runnable (TR) and Time-to-Satisfaction (TS) in minutes; our method used fully automated one-shot generation. Evaluation: Five evaluators and a VLM(in Appendix F) performed double-blind pairwise comparisons. The results are shown in Table 3: In terms of efficiency, our method has a significant advantage in construction speed. In terms of quality, despite human corrections, our method still achieved the highest win rates in the evaluation. This proves that our method not only lowers technical barriers but also constructs high-fidelity generative agent simulation environments with exceptional speed.

#### 4.6 Ablation Study on Visual Generation

To verify the role of the asset library in unifying visual styles, we performed an ablation comparison on the four models marked with “*” in Table 1 (see Table 4). We use VGG Gram matrix distance to measure style differences and Visual Harmony

Methods VGG Loss ↓ VH ↑ VSA-C ↑ VSA-V ↑

- Part 1: Generation w/o Asset Library

Base: Qwen3-(8+32)B 259.03 4.62 21.67 4.21 Open: Qwen3-235B 273.19 5.09 25.02 5.17 Closed: Gemini-3-Pro 229.33 4.62 25.07 5.20 Ours 248.62 4.47 27.93 6.35

- Part 2: Generation w/ Asset Library

Base: Qwen3-(8+32)B 56.77 7.56 21.45 4.03 Open: Qwen3-235B 58.62 7.17 24.72 5.66 Closed: Gemini-3-Pro 56.63 7.62 25.08 5.71 Ours 55.80 7.36 28.07 6.80

Table 4: Visual Generation Ablation. Top/Bottom: w/o vs.w/ asset library. VH uses flattened tiles to exclude layout influence. Prompts in the Appendix F.

(VH) to assess visual perception. Results show that removing the asset library leads to a sharp increase in VGG Loss and a significant drop in VH for all models, demonstrating that unprocessed tiles suffer from severe style discrepancies. Furthermore, comparing evaluation metrics reveals that VSA-C remains stable, while VSA-V shows a slight decline without the asset library. This indicates that inconsistent art styles interfere with the VLM’s judgment. In summary, the asset library effectively resolves style discrepancy issues and ensures the visual consistency of generated scenes.

### 5 Conclusion

We present World Craft, integrating World Scaffold , a standardized infrastructure addressing the issues of fragmented toolchains and high technical barriers, and World Guild , a multi-agent framework mitigating the semantic gap between narrative and spatial instructions. Additionally, we introduce a “reverse synthesis” method to generate high-quality supervision signals, enhancing LLM spatial reasoning. Experiments demonstrate that

our method outperforms other methods in physical consistency and intent alignment, achieving automated construction from natural language to executable game scenes and providing a standardized solution for the democratization of AI Towns.

### Limitations

Although World Craft successfully achieves automated creation from natural language to executable game scenes, the current system still has limitations, which outline directions for future research.

- 1. Limitations on Scene Scale and Complexity. Current generation primarily focuses on indoor environments within single scenes (e.g., residences, offices, or interiors of single buildings). While the system can handle room layouts and component placement, it does not yet fully support complete “town-level” macroscopic planning that covers outdoor terrain, road networks, and multi-building coordination. Large-scale outdoor scenes involve more complex hierarchical structures, which is a key challenge we need to address in the next stage.
- 2. Depth of Interaction Logic. While World Scaffold ensures basic physical interactability, the generated environments primarily support navigation, life simulation, and social activities. Current capabilities remain limited regarding advanced interaction logic involving complex physical simulations (e.g., fluids, destruction effects) or dynamic environmental evolution (e.g., constructing new environments in real-time during simulation).

Summary. In the future, we aim to extend World Craft to coordinated multi-scene open-world construction and further enrich asset diversity and interaction depth, thereby achieving truly fully automated AI Town generation.

### Ethics Statement

Data Privacy and Usage. All training data utilized in this paper were constructed based on our proposed method. Detailed descriptions of the construction algorithms and referenced content are provided in the Appendix. Additionally, the APIs for both open-source and closed-source models employed in data generation are listed in the Appendix. Finally, the tile sets used in our asset library were sourced from open-source works by authors on professional asset platforms; detailed credits and URLs will be provided in our open-source release.

All data have been anonymized to eliminate any personal or confidential information.

Human Evaluation Statement. This study involves human subjects, and we strictly adhere to ethical guidelines to safeguard participant rights. Key measures include: (1) Informed Consent: Prior to the experiment, we fully disclosed the research objectives, procedures, and participant rights. Participants were informed of their freedom to withdraw from the study unconditionally at any stage without facing any adverse consequences. (2) Data De-identification: All evaluation data (including interaction logs and questionnaires) have undergone strict anonymization. By removing personal identifiers, we ensure that data cannot be traced back to specific individuals.

### References

Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, Nicolas Pinto, and Joseph Turian. 2020. Experience grounds language. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8718–8735, Online. Association for Computational Linguistics.

Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. 2022. Omni3d: A large benchmark and model for 3d object detection in the wild. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13154–13164.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. 2024a. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14455–14465.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2024b. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In International Conference on Representation Learning, volume 2024, pages 20094–20136.

Xinyun Chen, Maxwell Lin, Nathanael Schaerli, and Denny Zhou. 2024c. Teaching large language models to self-debug. In International Conference on Representation Learning, volume 2024, pages 8746–8825.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsanit, Aniruddha Kembhavi, and

Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13142–13153.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Layoutgpt: compositional visual planning and generation with large language models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Cao Li, Zengqi Xun, Chengyue Sun, Yiyun Fei, Yu qiong Zheng, Ying Li, Yi Liu, Peng Liu, Lin Ma, Le Weng, Xiaohang Hu, Xin Ma, Qian Qian, Rongfei Jia, Binqiang Zhao, and Hao Helen Zhang. 2020. 3d-front: 3d furnished rooms with layouts and semantics. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10913–10922.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, WeiChiu Ma, and Ranjay Krishna. 2024. Blink: Multimodal large language models can see but not perceive. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXIII, page 148–166, Berlin, Heidelberg. Springer-Verlag.

Ran Gong, Qiuyuan Huang, Xiaojian Ma, Yusuke Noda, Zane Durante, Zilong Zheng, Demetri Terzopoulos, Li Fei-Fei, Jianfeng Gao, and Hoi Vo. 2024. MindAgent: Emergent gaming interaction. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3154–3183, Mexico City, Mexico. Association for Computational Linguistics.

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. Critic: Large language models can self-correct with tool-interactive critiquing. In International Conference on Representation Learning, volume 2024, pages 57734–57811.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Realm: retrievalaugmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 2023. 3d-llm: Injecting the 3d world into large language models. In Advances in Neural Information Processing Systems, volume 36, pages 20482–20494. Curran Associates, Inc.

Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A Ross, Cordelia Schmid, and Alireza Fathi. 2024. Scenecraft: an llm agent for synthesizing 3d scenes as blender code. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. 2024. Sceneverse: Scaling 3d visionlanguage learning for grounded scene understanding. In European Conference on Computer Vision.

Sicong Leng, Yang Zhou, Mohammed Haroon Dupty, Wee Sun Lee, Sam Joyce, and Wei Lu. 2023. Tell2Design: A dataset for language-guided floor plan generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14680– 14697, Toronto, Canada. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Wensi Ai, Benjamin Martinez, Hang Yin, Michael Lingelbach, Minjune Hwang, Ayano Hiranaka, Sujay Garlanka, Arman Aydin, Sharon Lee, Jiankai Sun, Mona Anvari, and 16 others. 2024. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation. Preprint, arXiv:2403.09227.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: communicative agents for "mind" exploration of large language model society. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Xi Li, Xiping Liu, Qing Shu, Zhao Tan, Changxuan Wan, Dexi Liu, and Qizhi Wan. 2025. Automatic contrastive chain-of-thought prompting: Learning from reasoning errors of large language models. Expert Systems with Applications, page 130919.

Jiaju Lin, Haoran Zhao, Aochi Zhang, Yiting Wu, Huqiuyue Ping, and Qin Chen. 2023. Agentsims: An open-source sandbox for large language model evaluation. Preprint, arXiv:2308.04026.

Ruibo Liu, Jason Wei, Shixiang Shane Gu, Te-Yen Wu, Soroush Vosoughi, Claire Cui, Denny Zhou, and Andrew M. Dai. 2022. Mind’s eye: Grounded language model reasoning through simulation. Preprint, arXiv:2210.05359.

Xinhang Liu, Chi-Keung Tang, and Yu-Wing Tai. 2025. Worldcraft: Photo-realistic 3d world creation and customization via llm agents. Preprint, arXiv:2502.15601.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,

Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, pages 46534–46594. Curran Associates, Inc.

Yuren Mao, Peigen Liu, Xinjian Wang, Rui Ding, Jing Miao, Hui Zou, Mingjie Qi, Wanxiang Luo, Longbin Lai, Kai Wang, Zhengping Qian, Peilun Yang, Yunjun Gao, and Ying Zhang. 2025. Agent-kernel: A microkernel multi-agent system framework for adaptive social simulation powered by llms. Preprint, arXiv:2512.01610.

Nelson Nauata, Sepidehsadat Hosseini, Kai-Hung Chang, Hang Chu, Chin-Yi Cheng, and Yasutaka Furukawa. 2021. House-gan++: Generative adversarial layout refinement network towards intelligent computational agent for professional architects. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13627–13636.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, UIST ’23, New York, NY, USA. Association for Computing Machinery.

Joon Sung Park, Lindsay Popowski, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2022. Social simulacra: Creating populated prototypes for social computing systems. Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology.

Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis, Andreas Geiger, and Sanja Fidler. 2021. Atiss: Autoregressive transformers for indoor scene synthesis. In Advances in Neural Information Processing Systems, volume 34, pages 12013–12026. Curran Associates, Inc.

Fedor Rodionov, Abdelrahman Eldesokey, Michael Birsak, John Femiani, Bernard Ghanem, and Peter Wonka. 2025. Floorplanqa: A benchmark for spatial reasoning in llms using structured representations. Preprint, arXiv:2507.07644.

Paulo Salem, Robert Sim, Christopher Olsen, Prerit Saxena, Rafael Barcelos, and Yi Ding. 2025.

Tinytroupe: An llm-powered multiagent persona simulation toolkit. Preprint, arXiv:2507.09788.

Mohammad Amin Shabani, Sepidehsadat Hosseini, and Yasutaka Furukawa. 2022. Housediffusion: Vector floorplan generation via a diffusion model with discrete and continuous denoising. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5466–5475.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: solving ai tasks with chatgpt and its friends in hugging face. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Ilias Stogiannidis, Steven McDonagh, and Sotirios A. Tsaftaris. 2025. Mind the gap: Benchmarking spatial reasoning in vision-language models. Preprint, arXiv:2503.19707.

Shyam Sudhakaran, Miguel González-Duque, Matthias Freiberger, Claire Glanois, Elias Najarro, and Sebastian Risi. 2023. Mariogpt: open-ended text2level generation through large language models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. 2025. 3d-gpt: Procedural 3d modeling with large language models. In 2025 International Conference on 3D Vision (3DV), pages 1253–1263.

Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. 2024. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29469–29478.

Fuwen Tan, Song Feng, and Vicente Ordonez. 2018. Text2scene: Generating compositional scenes from textual descriptions. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6703–6712.

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. 2023. Makeit-3d: High-fidelity 3d creation from a single image with diffusion prior. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22819–22829.

Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. 2023. On the planning abilities of large language models: a critical investigation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An openended embodied agent with large language models. Preprint, arXiv:2305.16291.

Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Yixuan Li, and Neel Joshi. 2024. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. In Advances in Neural Information Processing Systems, volume 37, pages 75392–75421. Curran Associates, Inc.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023b. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada. Association for Computational Linguistics.

Zhilin Wang, Yu Ying Chiu, and Yu Cheung Chiu. 2023c. Humanoid agents: Platform for simulating human-like generative agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 167–176, Singapore. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023. Autogen: Enabling next-gen llm applications via multi-agent conversation. Preprint, arXiv:2308.08155.

Wenming Wu, Xiao-Ming Fu, Rui Tang, Yuhan Wang, Yu-Hao Qi, and Ligang Liu. 2019. Data-driven interior plan generation for residential buildings. ACM Trans. Graph., 38(6).

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, and 10 others. 2023. The rise and potential of large

language model based agents: A survey. Preprint, arXiv:2309.07864.

Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, Leo Z. Liu, Yiheng Xu, Hongjin Su, Dongchan Shin, Caiming Xiong, and Tao Yu. 2023. Openagents: An open platform for language agents in the wild. Preprint, arXiv:2310.10634.

Rui Xu, Dakuan Lu, Zicheng Zhao, Xiaoyu Tan, Xintao Wang, Siyu Yuan, Jiangjie Chen, and Yinghui Xu. 2025. Origamispace: Benchmarking multimodal llms in multi-step spatial reasoning with mathematical constraints. Preprint, arXiv:2511.18450.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli Vanderbilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. 2024. Holodeck: Language guided generation of 3d embodied ai environments. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16277–16287.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Jun Yin, Pengyu Zeng, Haoyuan Sun, Yuqin Dai, Han Zheng, Miao Zhang, Yachao Zhang, and Shuai Lu. 2025. FloorPlan-LLaMa: Aligning architects’ feedback and domain knowledge in architectural floor plan generation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6640–6662, Vienna, Austria. Association for Computational Linguistics.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: less is more for alignment. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. 2024. Sotopia: Interactive evaluation for social intelligence in language agents. In International Conference on Representation Learning, volume 2024, pages 40975–41019.

Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, Yu Qiao, Zhaoxiang Zhang, and Jifeng Dai. 2023. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. Preprint, arXiv:2305.17144.

Appendix

- A Comparison of Model Output Results

To intuitively demonstrate the performance differences between methods, we visualized the generated layouts for three diverse test scenarios:

- Scene 1 (A cool, glowing mycelium chamber

holds ancient scrolls with floating spores and faint rustles.);

- Scene 2 (A luxurious underground bathhouse,

where steam rises from hexagonal copper pools filled with mineral-rich water that shimmers with a faint blue glow. Exquisite mosaic tiles depict scenes of ancient inventors. Hidden panels on the walls slide open silently, revealing niches containing ticking lockboxes and half-burned blueprints.);

- Scene 3 (The once magnificent study is now sub-

merged on the seabed. Its arched windows have shattered, obscured by swaying kelp. Coral has spread over the bookshelves and lectern, encasing the leather-bound classics in calcified, lace-like formations. Luminescent fish dart through floating clouds of ink, remnants of broken inkwells. On a large stone table lies a glowing slab inscribed with indecipherable symbols. Sunlight filters down from far above in fractured beams, illuminating drifting particles like underwater snowflakes.).

Fig 6 first illustrates the generation results of three baseline models on Scene 2 and Scene 3 across the three stages defined in the upper part of Table 1. Observations indicate that in the absence of domain knowledge, relying solely on the multiagent framework remains insufficient to effectively handle complex geometric constraints. In contrast, Fig. 7 presents the performance of our method (Ours in Table 1) across all scenarios, demonstrating the effectiveness of our data and domain knowledge injection. Finally, Fig. 8 compares the results of code agents (Cursor and Antigravity). These methods tend to construct environments with simplistic layouts and sparse elements, lacking visual expressiveness, thus failing to meet the requirements of ideal simulation environments.

- B Experimental Parameters and Settings

All training experiments were conducted on NVIDIA 8*H200 GPUs(141G). The detailed hyperparameter settings are listed in Table 5.

[Figure 4]

Figure 6: Examples of output results of the three models in three reasoning stages, Refer to the upper part of Table 1.

Hyperparameter Stage 1 (Enricher) Stage 2 (Manager) Model Configuration Base Model Qwen3-8B Qwen3-32B Fine-tuning Method Full-parameter Full-parameter Precision bfloat16 bfloat16 Max Sequence Length 2048 12000 DeepSpeed Strategy ZeRO-3 ZeRO-3 Training Optimization Optimizer AdamW AdamW Learning Rate (lr) 2.0e-5 1.0e-5 LR Scheduler Type Cosine Cosine Warmup Ratio 0.1 0.1 Num. Epochs 5.0 5.0 Batch Size (per device) 4 1 Gradient Accum. Steps 4 1 Total Batch Size1 128 8

Table 5: Detailed Hyperparameters for Two-Stage Instruction Tuning.

[Figure 5]

Figure 7: Examples of layout designs and final output results of our method in three scenarios.

[Figure 6]

Figure 8: Examples of the performance of Cursor and Antigravity in three scenarios.

### C Details of Asset Library

To address style fragmentation in text-to-image generation, we constructed a asset library(5500+), as shown in Fig. 9. The retrieval process aims to identify the most relevant reference image that matches both the semantic content and spatial dimensions of the target component. This algorithm employs a two-stage strategy: Token Matching and Dimension Ranking. As outlined in Algorithm 1, the system first filters candidate assets based on keyword intersection. Subsequently, it calculates the Manhattan distance between the target dimensions and candidate tile dimensions to minimize spatial distortion.

Algorithm 1: Asset Retrieval Strategy

Input :Target Asset ID Sid, Description Sdesc, Target Dims Dtarget(w,h), Index Dlib

Output:Best Matching Reference Image

vref

// Step 1: Query Normalization

- 1 Qtext ← Sid ⊕ “ ” ⊕ Sdesc;
- 2 Qtokens ← Normalize(Qtext); // Remove stop words
- 3 vref ← None;
- 4 Pmin ← ∞; // Initialize penalty // Step 2: Traverse Asset Library
- 5 for each candidate asset Ai ∈ Dlib do

- 6 Ti ← Ai.tokens; // Stage 1: Semantic Filtering
- 7 if Intersection(Qtokens,Ti) ̸= ∅ then

- 8 Di ← Ai.dimensions; // Stage 2: Dimension Ranking
- 9 Pcurr ← |Dtarget.w − Di.w| + |Dtarget.h − Di.h|;
- 10 if Pcurr < Pmin then

- 11 Pmin ← Pcurr;
- 12 vref ← Ai.path;
- 13 if Pmin == 0 then

- 14 break; // Perfect match

- 15 return vref

[Figure 7]

###### Figure 9: Examples of entries in the Asset Library (Dlib). We utilize open-source tile sets as the foundation for our localized library. Due to copyright restrictions, we present only a subset of representative examples here. Full credits and links to the original artists will be provided after open-sourcing.

### D Procedural Generation Logic and Data-Driven Priors

To ensure that the synthetic layouts generated in Stage 1 structurally align with real-world patterns, we extracted a set of architectural priors from the RPLAN dataset (Wu et al., 2019). Given that RPLAN is strictly limited to residential floor plans, whereas our framework aims to construct generalpurpose agent environments covering diverse functions (e.g., offices, retail), we focused on extracting generalizeable geometric and topological rules rather than relying directly on domain-specific samples. This strategy enables us to generate infinite and diverse scenes using limited data. Below are examples of key constraints applied in the pipeline:

Geometric Orthogonality. Observations indicate that the vast majority of wall segments are axis-aligned. Therefore, our generation algorithm enforces a strict orthogonality constraint and operates on a discrete grid. This ensures all walls remain horizontal or vertical, preventing irregular angles and ensuring the generated structures comply with general architectural norms.

Topological Centrality. Real-world layouts typically evolve around public spaces. To simulate this topological feature, the algorithm adopts a “periphery-to-center partitioning” logic: it first initializes the overall building envelope and then iteratively carves private rooms from the boundary inwards. The remaining unpartitioned area naturally forms the central public core, mathematically guaranteeing a connected backbone structure and avoiding isolated regions.

Boundary Morphology. To simulate complex contours formed in real buildings due to lighting or zoning requirements, our algorithm implements a shape grammar. This iteratively augments the initial building envelope with randomized substructures, producing complex non-convex boundaries that more closely resemble real-world floor plan footprints.

Dimensional Calibration. We constrain generation using aspect ratio and area thresholds derived from real-world distributions, rather than using random parameters. This ensures that every generated space is physically usable, preventing the creation of geometrically valid but functionally uninhabitable “splinter” corners.

Path Optimization. Door placement determines circulation efficiency. When connecting rooms, our algorithm employs a distance-minimization cost function to automatically locate wall segments that minimize the distance to the central area. This placement strategy effectively replicates the efficient circulation patterns found in human-designed floor plans.

Wall Continuity and Thickness. To align with RPLAN annotation standards (where walls are represented with consistent pixel thickness), our generator includes a geometric regularization step. This step merges fragmented boundary segments into continuous geometric primitives, ensuring the result is topologically equivalent to the semantic segmentation maps provided in the original dataset.

The pseudocode for the synthetic layout generation step is shown in Algorithm 2.

Algorithm 2: RPLAN-Aligned Layout Synthesis

Input :Target Room Count Nrooms, Map

Dims W,H, RPLAN Priors Θ Output:Semantic Grid Layout G // Step 1: Non-Convex Boundary

Generation

- 1 Score ← InitSeed(W,H);
- 2 Bpoly ← AugmentShape(Score,Θshape); // Add sub-rects per Shape Grammar
- 3 G ← Rasterize(Bpoly); // Step 2: Topology-Preserving

Partitioning

- 4 Qcorners ← GetVertices(Bpoly);
- 5 Rlist ← ∅;
- 6 while |Rlist| < Nrooms and Qcorners ̸= ∅ do

- 7 c ← PopRandom(Qcorners); // Constraint: Dimensional

Calibration

- 8 Rcand ← ScanRegion(c,Θdim);
- 9 if IsValid(Rcand) and IsConnected(G \ Rcand) then

- 10 G ← PartitionRoom(G,Rcand);
- 11 Rlist.append(Rcand); // Step 3: Circulation Optimization

- 12 Ccore ← Centroid(Score);
- 13 for each room r ∈ Rlist do

- 14 Wcand ← FindValidWallSegments(r); // Minimize distance to

functional core

- 15 pbest ← arg minp∈Wcand Dist(p,Ccore);
- 16 G[pbest] ← DOOR; // Step 4: Geometric Regularization

- 17 G ← RegularizeWalls(G); // Merge segments & uniform thickness
- 18 return G

### E Details of manual evaluation

Evaluator Profile To ensure professional judgment regarding game scene layouts, we recruited five independent evaluators for the reviews in Sections 4.5 and 4.4. All participants hold at least a Bachelor’s degree, possess an average of over five years of experience in RPG or simulation strategy games, and are familiar with common game map mechanics and navigation logic. Additionally, for the Code Agent (Cursor, Antigravity) operation tasks in Section 4.5, the three operators were doctoral students specializing in Artificial Intelligence, ensuring the standardized use of the tools.

Questionnaire Design To minimize cognitive load during the evaluation, we synthesized the eight automated metrics from the main text into three dimensions of pairwise forced-choice questions. Evaluators were presented with the input instruction and two generated scene images, and asked to make decisions based on the following criteria:

Layout Plausibility (Correlating with CFR, RCS, OPS): “Which scene’s layout is physically more reasonable and visually more harmonious?” This dimension assesses spatial connectivity and the logical placement of objects.

Content Richness (Correlating with CER, OVD, PAC): “Which scene contains more valid details and fewer incongruous objects?” This dimension focuses on asset diversity and visual artifacts such as floating or overlapping objects.

Intent Consistency (Correlating with VSA): “Which scene more accurately reflects the input text description?” This dimension evaluates the semantic fidelity of the generated result to the natural language instruction.

### F Details of Evaluation Metrics

This section compiles the specific prompts utilized for the VLM-based scoring metrics discussed in the main text. To ensure reproducibility and transparency, we present the full content of the instructions input to the Gemini-3-Pro model. The detailed prompts for Object Placement Reasonableness (OPS), Object Volume Density (OVD), Physical Attribute Consistency (PAC), and Visual Harmony (VH) are illustrated in Fig. 10, Fig. 11, Fig. 12, and Fig. 14, respectively.

#### Prompt for Object Placement Reasonableness (OPS)

This is a design document for a game scene. Please read it carefully, and your task is

to identify semantically unreasonable elements in the scene. Scene Description: "{scene_desc}" Complete Original Design Data of the Scene:{design_data_str} Data Reading Guidelines

- 1.Position Information: Check the object_layer (including position coordinates) under the layout field, as well as the wall_layer/floor_layer (including area regions).
- 2.Object Attributes: Use the asset_id to find the corresponding object name (name) and size (base_size) in the assets field.
- 3.Task: Judge the design logic by combining coordinate positions and object attributes.

Please identify objects that are logically incompatible or extremely unusual in the context of this specific scene.(Examples: A toilet should never appear in a kitchen or main seating area such as a living room corridor; A main entrance must not be blocked by large obstacles like bookshelves or pianos, unless it is an intentional decorative screen.)

Coordinate-Based Judgment Required: Check if objects are located in incorrect room areas (e.g., A toilet falls within the kitchen floor area based on coordinate calculations).

You must return a JSON object: {{

"unreasonable_objects": ["Object name1", "Object name2"], "count": 2, "reason": "Brief explanation (if position errors are involved, please reference

coordinates for explanation)"

}} If all elements are reasonable, set "count": 0. Do not output any extra content, especially greetings or irrelevant remarks.

Figure 10: Evaluation Prompt for Object Placement Reasonableness (OPS).

#### Prompt for Object Volume Density (OVD)

Role: You are an interior design critic. Task: Evaluate the "object density" and "spatial suitability" of the provided floor

plan image.Scene Background: "{scene_desc}" The image shows a layout: Green = floor, Red = objects/obstacles, Black = empty space. Detailed Scoring Rubric (0-10):

[0-2] Severe Failure (Unusable) Severe clutter: Objects completely block the main entrance or passageways. No

navigable paths exist. OR Extreme emptiness: Despite being described as a functional space, the room is

essentially vacant. [3-4] Poor (Imbalanced/Awkward) Poor circulation: Paths exist but are zigzag, overly narrow, or cramped. Uneven distribution: All objects cluster in one corner, leaving the rest as unused

space. Proportion issues: Objects appear excessively large or small relative to the room. [5-6] Fair (Functional but Mediocre) Navigable: Passage is possible, and furniture is placed. Lack of design sense: Looks randomly arranged or a basic grid layout, with no

consideration for aesthetics or functional zones. [7-8] Good (Professional Layout) Clear zoning: Distinct areas designated for specific activities. Excellent circulation: Retains clear, spacious traffic paths (main thoroughfares). Balanced density: Objects are reasonably distributed in the space, without causing

a sense of overcrowding. [9-10] Excellent (Masterful/Perfect) Perfect context alignment: Density perfectly matches the scene description (e.g., a

"cozily cluttered study" feels cozy rather than messy; a "minimalist gallery" feels open rather than empty).

Negative space: Exceptional use of empty space ("breathing room") to complement objects.

Natural composition: The layout feels natural and lived-in, not computer-generated. Output JSON ONLY: {{

"score": 8, "reason": "The cafe has distinct zones and good flow."

}}

Figure 11: Evaluation Prompt for Object Volume Density (OVD).

#### Prompt for Physical Attribute Consistency (PAC)

This is the size of all elements in a game scene. You need to check the physical attributes and size consistency of these elements to see if their sizes are reasonable.

Below is a list of assets used in the scene, including: Base size: The grid footprint on the floor (e.g., [1, 1] represents a 1x1 grid). For

example, a refrigerator has a footprint of 2x1, and a table has a footprint of 2x2.

Visual size: The actual visual size of the object. For objects with low or no height ( such as carpets), their visual size must be equal to the base size. For objects with a certain height or considerable height, the height of the visual size is often one, two, or three units higher than the base size.

For example, a table with a base size of 2x2 may have a visual size of 2x3; a

refrigerator with a base size of 2x1 may have a visual size of 2x4. Here is the data content:{data_str} Evaluation Criteria:

- 1.Footprint logic: Is the base size reasonable? (e.g., a "grand piano" cannot be 1 x1).
- 2.Vertical proportion and size (critical): Check the visual_size (especially the height).

- -Relative proportion: Compare objects with each other.
- -Violation example: If the visual height of a "chair" is 5.0 while the height of a "table" is 1.0, this is a serious violation.
- -Violation example: A "refrigerator" should not be smaller than a "stool".

You need to return a JSON object: {{

"violation_objects": ["name_of_bad_object_1", "name_of_bad_object_2"], "count": <int>, "reason": "Specific explanation referencing the dimensions (e.g., ’Chair height 5.0

is disproportionate to Table height 1.2’)."

}} If everything is reasonable, return "count": 0. Do not output any other extra content, especially various greeting contents.

Figure 12: Evaluation Prompt for Physical Attribute Consistency (PAC).

#### Prompt for Pairwise Preference Evaluation

Here are two AI-generated town environments based on the textual description: "{text}". Please evaluate them considering the following criteria:

- 1. Instruction Alignment: Does the scene accurately include the elements and atmosphere described in the text?
- 2. Visual Aesthetics: Is the artistic style consistent, detailed, and visually appealing?
- 3. Layout & Playability: Is the spatial structure logical and inviting for a player to explore?

Based on a comprehensive assessment of these factors, you only need to answer the final question: In which game environment would you prefer to play? The output result is 1 or 2.

Figure 13: Evaluation Prompt for Pairwise Preference (Human & VLM).

#### Prompt for Visual Harmony (VH)

Role: You are an Art Director for a top-tier game studio. Task: Evaluate the "Stylistic Consistency" and "Visual Harmony" of the generated game

scene image.

Context: This scene is constructed from various tiles and resources. Your goal is to strictly determine whether these elements blend into a unified artistic whole or appear disjointed.

Evaluation Dimensions:

- 1. Artistic Consistency: Do all assets share the same rendering style (e.g., pixel art resolution, stroke style, level of detail)?
- 2. Color Harmony: Is there a unified color palette and lighting temperature across the scene?
- 3. Integration: Do objects look naturally grounded, or do they look like stickers pasted onto a background (artifacts, jagged edges, mismatched perspective)?

Detailed Scoring Rubric (0-10): [0-2] Visual Chaos (Frankenstein-like) Severe mismatch: The scene looks like a random collage. For example, a photorealistic

chair placed next to a 8-bit pixel art table. Clashing palettes: Colors are jarring and uncoordinated. [3-4] Disjointed (Lack of Polish) Noticeable inconsistency: Most items match, but 3-4 objects clearly belong to a different

art style or asset pack. Poor blending: Visible seams between tiles or distinct lighting differences between objects and the floor.

[5-6] Acceptable (Generic Consistency) Unified but dull: The style is generally consistent, but looks flat or mechanical. Minor flaws: Slight variations in pixel density or color saturation, but nothing scene-

breaking.

[7-8] Good (Cohesive Style) Strong theme: All elements clearly belong to the same specific world (e.g., "Cyberpunk" or

"Victorian"). Harmonious colors: The color palette is pleasing and consistent. Objects interact well with the background.

[9-10] Masterpiece (Perfect Unity) Seamless integration: It is impossible to tell individual assets apart; the scene looks

like a single, hand-painted illustration. Atmospheric unity: Lighting, shadows, and textures work together to create a compelling mood.

You must return a JSON object: {{

"score": <int>, "reason": "Specific critique (e.g., ’The vending machine has a different pixel resolution

than the wall, breaking immersion.’)"

}} Do not output any extra content.

Figure 14: Evaluation Prompt for Visual Harmony (VH).

### G Dataset Construction and Statistics

To complement the general description in Section 3.3, this section details the construction sources and distribution characteristics of the dataset. Fig.s 15(a)-(d) first visualize the semantic word clouds from the four collection sources of the raw data-Real-world Scenarios, Literature, Film & TV, and TRPG Games-highlighting the diversity of the initial corpus. Subsequently, Fig. 15(e) presents the vocabulary distribution of the final dataset (2,000 samples) after stylistic augmentation based on the training set (400 samples). Finally, Fig. 15(f) illustrates the distribution of the Top-25 style categories in the augmented dataset.

### H Data Structure Definitions and Examples

To complement the content in Section 3.1, this section provides a complete data example of the structured layout G, as shown in Fig. 16. This example specifically illustrates the internal details of the quadruple (M,A,L,P), including Metadata (M) defining basic scene configurations, Asset Definitions (A) describing visual styles and layer attributes, Layout (L) establishing precise spatial topology, and Properties (P) specifying collision and interaction logic, intuitively demonstrating how the model instantiates the generation target into executable game environment data.

[Figure 8]

###### Figure 15: Dataset statistics and distribution. (a)-(d) Semantic word clouds for the four initial data collection domains. (e) Vocabulary distribution of the final augmented dataset. (f) The frequency distribution of the Top-25 style categories.

#### Structured Layout Output (G)

"metadata": { "scene_name": "The Verdant Scholar’s Hub", "grid_size": [35, 28], "description": "A bustling establishment. The interior is subdivided into a quiet

reading zone and a social cafe area. The kitchen is fully equipped. The garden is romantic and well-lit.",

"style_prompt": "16-bit pixel art, top-down RPG style, cozy lighting, clutter and details"

}, "assets": {

"floor_dark_wood": {"type": "tile", "description": "Polished dark mahogany wood floor

.", "base_size": [1, 1], "visual_size": [2, 2]}, "floor_kitchen_tile": {"type": "tile", "description": "White and blue checkerboard tile

.(floor)", "base_size": [1, 1], "visual_size": [2, 2]}, "floor_garden_grass": {"type": "tile", "description": "Lush green grass.(floor)", " base_size": [1, 1], "visual_size": [2, 2]}, "wall_brick_library": {"type": "tile", "description": "Vintage red brick wall with wood trim.(wall)", "base_size": [1, 1], "visual_size": [1, 6]}, "wall_kitchen_white": {"type": "tile", "description": "Clean white tiled wall.", " base_size": [1, 1], "visual_size": [1, 5]}, "wall_garden_hedge": {"type": "tile", "description": "Manicured green bush hedge.(wall) ", "base_size": [1, 1], "visual_size": [1, 4]}, "window_large_arched": {"type": "object", "description": "A large arched window letting in sunlight.", "base_size": [2, 1], "visual_size": [2, 3]}, "bookshelf_tall": {"type": "object", "description": "Towering wooden bookshelf.", " base_size": [3, 2], "visual_size": [2, 4]}, "bookshelf_wide": {"type": "object", "description": "A wider, lower bookshelf filled with magazines.", "base_size": [3, 2], "visual_size": [3, 2]}, "table_long_study": {"type": "object", "description": "Long sturdy wooden table with lamps.", "base_size": [4, 2], "visual_size": [4, 3]}, "table_cafe_round": {"type": "object", "description": "Small round wooden table for coffee.", "base_size": [2, 2], "visual_size": [2, 3]}, "chair_library": {"type": "object", "description": "Cushioned wooden chair.", " base_size": [2, 2], "visual_size": [2, 3]}, "chair_stool": {"type": "object", "description": "A simple wooden stool.", "base_size": [2, 2], "visual_size": [2, 3]}, "bar_counter_L": {"type": "object", "description": "L-shaped wooden service counter.", "base_size": [4, 3], "visual_size": [4, 4]}, "kitchen_stove": {"type": "object", "description": "Stainless steel industrial stove.", "base_size": [3, 2], "visual_size": [3, 4]}, "kitchen_fridge": {"type": "object", "description": "A large silver double-door fridge

.", "base_size": [3, 2], "visual_size": [3, 4]}, "kitchen_sink_counter": {"type": "object", "description": "Counter with a sink and dirty dishes.", "base_size": [3, 2], "visual_size": [3, 3]}, "plant_indoor_fern": {"type": "object", "description": "A potted fern plant.", " base_size": [2, 2], "visual_size": [2, 3]}, "lamp_standing": {"type": "object", "description": "A tall vintage floor lamp casting warm light.", "base_size": [2, 2], "visual_size": [2, 3]}, "fountain_stone": {"type": "object", "description": "Three-tiered stone fountain.", " base_size": [3, 3], "visual_size": [3, 4]}, "table_garden_iron": {"type": "object", "description": "White iron garden table.", " base_size": [3, 3], "visual_size": [3, 4]}, "chair_garden_iron": {"type": "object", "description": "White metal garden chair.", "

- base_size": [2, 2], "visual_size": [2, 3]},

"lamp_street_garden": {"type": "object", "description": "Black iron street lamp.", "

- base_size": [2, 2], "visual_size": [2, 4]},

"flower_bush_red": {"type": "object", "description": "Bush with red roses.", "base_size ": [2, 2], "visual_size": [2, 2]}, "door_glass": {"type": "object", "description": "Double glass door.", "base_size": [3, 2], "visual_size": [2, 4]}, "rug_persian": {"type": "object", "description": "Intricate red Persian rug.", " base_size": [4, 3], "visual_size": [4, 3]}, "agent_librarian": {"type": "agent", "description": "The librarian wearing white clothes and black glasses", "base_size": [1, 1], "visual_size": [2, 3]}, "npc_student_f": {"type": "npc", "description": "A female student in white clothes", " base_size": [1, 1], "visual_size": [2, 3]},

"npc_chef": {"type": "npc", "description": "A male chef wearing white clothes", " base_size": [1, 1], "visual_size": [2, 3]}, "npc_hipster": {"type": "npc", "description": "A male customer wearing blue fashion clothes", "base_size": [1, 1], "visual_size": [2, 3]}

}, "layout": {

"floor_layer": [ {"asset_id": "floor_dark_wood", "command": "fill_rect", "area": [0, 0, 25, 20]}, {"asset_id": "floor_kitchen_tile", "command": "fill_rect", "area": [25, 0, 10, 20]},

{"asset_id": "floor_garden_grass", "command": "fill_rect", "area": [0, 20, 35, 8]}

], "wall_layer": [

- {"asset_id": "wall_brick_library", "command": "fill_rect", "area": [0, 0, 35, 1]},
- {"asset_id": "wall_brick_library", "command": "fill_rect", "area": [0, 1, 1, 19]}, {"asset_id": "wall_brick_library", "command": "fill_rect", "area": [34, 1, 1, 19]}, {"asset_id": "wall_kitchen_white", "command": "fill_rect", "area": [24, 1, 1, 10]},

- {"asset_id": "wall_garden_hedge", "command": "fill_rect", "area": [0, 19, 11, 1]}, {"asset_id": "wall_garden_hedge", "command": "fill_rect", "area": [14, 19, 21, 1]},
- {"asset_id": "wall_garden_hedge", "command": "fill_rect", "area": [0, 20, 1, 7]}, {"asset_id": "wall_garden_hedge", "command": "fill_rect", "area": [34, 20, 1, 7]}, {"asset_id": "wall_garden_hedge", "command": "fill_rect", "area": [0, 27, 35, 1]}

], "object_layer": [

{ "asset_id": "window_large_arched", "position": [3, 0] }, { "asset_id": "window_large_arched", "position": [10, 0] }, { "asset_id": "window_large_arched", "position": [17, 0] },

{ "asset_id": "bookshelf_tall", "position": [1, 1] }, { "asset_id": "bookshelf_tall", "position": [5, 1] }, { "asset_id": "bookshelf_tall", "position": [1, 6] }, { "asset_id": "bookshelf_wide", "position": [5, 6] },

{ "asset_id": "rug_persian", "position": [6, 12] }, { "asset_id": "table_long_study", "position": [6, 12] },

- { "asset_id": "chair_library", "position": [4, 12] },
- { "asset_id": "chair_library", "position": [5, 12] },

- { "asset_id": "chair_library", "position": [8, 12] },
- { "asset_id": "chair_library", "position": [9, 12] }, { "asset_id": "lamp_standing", "position": [3, 11] },

{ "asset_id": "table_cafe_round", "position": [15, 10] }, { "asset_id": "chair_library", "position": [14, 10] }, { "asset_id": "chair_library", "position": [17, 10] }, { "asset_id": "table_cafe_round", "position": [15, 15] }, { "asset_id": "chair_library", "position": [14, 15] }, { "asset_id": "chair_library", "position": [17, 15] }, { "asset_id": "plant_indoor_fern", "position": [18, 8] },

{ "asset_id": "bar_counter_L", "position": [20, 4] },

- { "asset_id": "chair_stool", "position": [20, 7] },
- { "asset_id": "chair_stool", "position": [21, 7] },

{ "asset_id": "kitchen_stove", "position": [26, 1] }, { "asset_id": "kitchen_fridge", "position": [30, 1] }, { "asset_id": "kitchen_sink_counter", "position": [32, 4] },

{ "asset_id": "door_glass", "position": [12, 20] }, { "asset_id": "fountain_stone", "position": [28, 23] }, { "asset_id": "flower_bush_red", "position": [26, 22] }, { "asset_id": "flower_bush_red", "position": [32, 22] },

{ "asset_id": "table_garden_iron", "position": [6, 23] }, { "asset_id": "chair_garden_iron", "position": [5, 23] }, { "asset_id": "chair_garden_iron", "position": [8, 23] }, { "asset_id": "table_garden_iron", "position": [15, 23] },

{ "asset_id": "chair_garden_iron", "position": [14, 23] }, { "asset_id": "chair_garden_iron", "position": [17, 23] }, { "asset_id": "lamp_street_garden", "position": [10, 22] }, { "asset_id": "lamp_street_garden", "position": [20, 22] }

], "npc_layer": [

{ "asset_id": "agent_librarian", "position": [21, 5] }, { "asset_id": "npc_student_f", "position": [5, 13] }, { "asset_id": "npc_chef", "position": [28, 3] }, { "asset_id": "npc_hipster", "position": [14, 11] }

]

}, "properties": {

"floor_dark_wood": { "physics": "passable", "navigation": "walkable", "semantic_tag": " floor_main" }, "floor_kitchen_tile": { "physics": "passable", "navigation": "walkable", "semantic_tag ": "floor_kitchen" }, "floor_garden_grass": { "physics": "passable", "navigation": "walkable", "semantic_tag ": "floor_garden" }, "wall_brick_library": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " wall" }, "wall_kitchen_white": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " wall" }, "wall_garden_hedge": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " fence" }, "window_large_arched": { "physics": "solid", "navigation": "obstacle", "semantic_tag": "window" }, "bookshelf_tall": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " bookshelf" }, "bookshelf_wide": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " bookshelf" }, "table_long_study": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " table_study" }, "table_cafe_round": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " table_cafe" }, "kitchen_stove": { "physics": "solid", "navigation": "obstacle", "semantic_tag": "stove " }, "kitchen_fridge": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " fridge" }, "kitchen_sink_counter": { "physics": "solid", "navigation": "obstacle", "semantic_tag": "sink" }, "bar_counter_L": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " counter" }, "chair_library": { "physics": "passable", "navigation": "obstacle", "semantic_tag": " chair" }, "chair_stool": { "physics": "passable", "navigation": "obstacle", "semantic_tag": " chair" }, "chair_garden_iron": { "physics": "passable", "navigation": "obstacle", "semantic_tag": "chair" }, "table_garden_iron": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " table_garden" }, "fountain_stone": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " decoration" }, "lamp_standing": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " light_source" }, "lamp_street_garden": { "physics": "solid", "navigation": "obstacle", "semantic_tag": " light_source" }, "door_glass": { "physics": "passable", "navigation": "walkable_door", "semantic_tag": " door_main" }, "agent_librarian": {"character_name": "Mr. Bookman", "is_agent": True, "soul_file": " librarian_soul.json"}, "npc_student_f": {"character_name": "Sarah", "is_agent": False, "soul_file": " student_soul.json"}, "npc_chef": {"character_name": "Gordon", "is_agent": False, "soul_file": "chef_soul. json"}, "npc_hipster": {"character_name": "Liam", "is_agent": False, "soul_file": "hipster_soul

.json"}, }

}

Figure 16: Instantiation of the Structured Layout Quadruple .

