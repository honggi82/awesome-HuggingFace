### Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution

Chenghao Li1, Chaoning Zhang1*, Yi Lu2, Jiaquan Zhang1, Qigan Sun3, Xudong Wang3, Jiwei Wei1, Guoqing Wang1, Yang Yang1, Heng Tao Shen4,1

1UESTC; 2CNU; 3KHU; 4Tongji Univ.

##### Abstract

GSM8K

[Figure 1]

95.2

Chain-of-Thought prompting enhances the reasoning of large language models by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, for complex tasks with vast solution spaces and ambiguous constraints, a single reasoning chain often proves insufficient. Inspired by Minimal Free Resolutions (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)—a new framework that extends CoT by introducing auxiliary, interrelated reasoning paths. By explicitly modeling dependencies among these paths, SoT captures deeper logical structure and enables more robust, structured problem solving. MFR decomposes a module into a sequence of free modules of minimal rank, providing a principled way to analyze complex systems. Drawing on MFR’s principle of “minimal representation, precise constraints, and layer-by-layer relation tracing,” SoT systematically decomposes an original complex problem into logically complete minimal subproblems while preserving key problem characteristics and reducing reasoning length. We evaluate SoT across diverse datasets (e.g., GSM8K, MATH) and model families (e.g., GPT-4o-mini, Qwen2.5). Results show that SoT achieves inference accuracy comparable to or exceeding mainstream CoT baselines. Moreover, by aligning the sampling process with algebraic constraints, SoT improves the scalability of inference-time computation, offering both transparent reasoning and strong performance. The code is available at https://github.com/dlMARiA/Syzygyof-thoughts

SVAMP

# arXiv:2504.09566v3[cs.CL]22Dec2025

CLUTRR

92.2

75.7

[Figure 2]

86.1

90.6

26.3

99.7

Data

MultiArith

80.8

97.0

36.2

99.8

57.3

97.3

72.8

57.1

63.3

ASDiv

BBH

75.6 75.2

AQUA

MMLU

4o-mini SOT

4o-mini COT

4o-mini COT-SC

Qwen2.5 COT-SC

Qwen2.5 COT

Qwen2.5 SOT

Figure 1: SoT (Ours) achieved performance improvements compared to CoT and CoT-SC on two models across nine datasets. The inner circle shows three methods of Qwen2.5, while the outer circle shows three methods of 4o-mini.

demonstrated outstanding reasoning capabilities across various tasks (Zhao et al., 2023b; Min et al., 2023). In recent years, the Chain of Thought (CoT) prompting method (Wei et al., 2022) has emerged as an effective strategy for simulating human reasoning by generating intermediate reasoning steps, significantly enhancing model performance in solving complex problems. The core advantage of the CoT framework lies in its ability to decompose problems into a series of explicit and easily interpretable intermediate steps, thereby improving the transparency and explainability of the reasoning process. This explicit stepwise reasoning not only facilitates in-depth analysis and debugging of model outputs but also provides a clear logical pathway for solving complex problems.

##### 1 Introduction

The rapid advancements in natural language processing (Vaswani et al., 2017; Devlin et al., 2019; Radford et al., 2018; Brown et al., 2020) have

However, despite its success in many tasks, CoT faces inherent limitations with high-dimensional, nonlinear, and abstract logical problems (Wang et al., 2022a; Shi et al., 2023; Jin et al., 2024). Tra-

*Corresponding author

ditional CoT methods rely on fixed or heuristic decompositions, which fail to capture essential details in multidimensional problems (Wang et al., 2022a). Additionally, the lack of structured planning in intermediate steps leads to redundant computations and increased inference latency (Jin et al., 2024). While improvements like multi-path exploration, voting, and hierarchical strategies have helped, they still fail to fully address logical incompleteness and poor high-dimensional decomposition. Thus, challenges in robustness and accuracy remain for complex reasoning tasks.

To address these challenges, we propose a new reasoning framework—Syzygy of Thoughts (SoT). “Syzygy” is derived from the Greek word syzygia (συζυγια), meaning “union” or “pairing”. This method draws inspiration from the concept of MFR in algebraic geometry and computational algebra (Eisenbud, 2013; Rossi and Sharifan, 2009; Botbol et al., 2021), systematically decomposing and reconstructing problems by constructing minimal free module sequences. The incorporation of MFR theory ensures a more rigorous and efficient problem decomposition process while preserving the core characteristics of the problem, effectively reducing redundant computations and logical inconsistencies. Specifically, SoT retains the stepwise reasoning advantages of CoT while introducing algebraic tools such as "module", "Betti numbers", "freeness", "mapping", "exactness" and "minimality" to deeply explore and systematically express the intrinsic structure of complex problems. This approach provides a new theoretical perspective for analyzing high-dimensional and multivariable problems.

To validate the effectiveness of SoT, we conducted extensive experiments across diverse datasets, including GSM8K, MATH, and others, spanning mathematical reasoning and highdimensional problem domains. These datasets, with their complex structures and multifaceted features, provide a robust testbed for evaluating reasoning frameworks in real-world scenarios. Experimental results show that SoT achieves inference accuracy that matches or surpasses mainstream CoTs standards, demonstrating its superior performance and practical potential.

The main contributions of this work:

• We introduce SoT, an MFR-inspired inferencetime reasoning framework that generalizes CoT from a single linear chain to interdepen-

dent, syzygy-structured reasoning paths.

- • We formalize the reasoning process through an MFR lens by mapping algebraic notions (Module, Betti numbers, Freeness, Mapping, Exactness, Minimality) to concrete prompting and search operations, enabling structured decomposition, logical-closure verification, and redundancy pruning.
- • We propose an algebraically constrained sampling/selection procedure that improves the accuracy–cost trade-off and provides stable inference under sampling randomness.
- • We conduct extensive experiments across nine benchmarks and multiple LLM backbones, showing that SoT consistently outperforms CoT, CoT-SC, GoT, and AoT, with lower or comparable variance, supported by cost and ablation analyses.

##### 2 Related Work

2.1 Advanced Reasoning Techniques in LLMs Recent advances in prompting have substantially improved the reasoning of LLMs, with Chain-ofThought (CoT) prompting (Wei et al., 2022) being a seminal example. CoT elicits step-by-step solutions but can still struggle on multidimensional or nonlinear tasks (e.g., abstract logic and advanced mathematics). To mitigate these issues, extensions such as Zero-shot-CoT (Kojima et al., 2022), Self-Consistency CoT (Wang et al., 2022b), and Auto-CoT (Zhang et al., 2022) have been proposed, together with verification-based methods including VerifyCoT (Zhao et al., 2023a) and CoFCoT (Nguyen et al., 2023). Beyond linear CoT, structured prompting further strengthens reasoning. Least-to-Most Prompting (LtM) (Zhou et al., 2022) decomposes tasks into simpler subproblems, while programmatic paradigms—Program of Thought (PoT) (Chen et al., 2022), Chain of Code (CoC) (Li et al., 2023), and Buffer of Thought (BoT) (Yang et al., 2025)—integrate discrete variables and procedural execution. Algorithm of Thought (AoT) (Sel et al., 2023) then iteratively synthesizes and refines intermediate results to improve token efficiency. Search-based paradigms explore multiple solution paths. Tree-ofThought (ToT) (Yao et al., 2023) performs hierarchical branching but may suffer from exponential growth, whereas Graph-of-Thought (GoT) (Besta

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

et al., 2024) generalizes this to flexible graph structures for aggregation and recalibration. Complementary approaches such as Skeleton-of-Thought (SoT) (Ning et al., 2023), Self-Refine (Madaan et al., 2023), and Step-Back Prompting (Zheng et al., 2023) emphasize outlining and iterative refinement to improve accuracy and stability.

Module Initial Complex Problem

Complete Lego Structure

Betti Numbers

Number of Hints per Leve

Number of Blocks per Level

Decomposed Subproblems with Hints.

Freeness

Basic Units

Solution Strategy

###### 2.2 Minimal Free Resolution

Mapping

Assembly Instructions

MFR is a fundamental tool in homological algebra (Fieldsteel and Nagel, 2021) and algebraic geometry (Botbol et al., 2021) for characterizing module structures (Eisenbud, 2013; Rossi and Sharifan, 2009) and deriving invariants such as rank, symmetry, and relations (Chen, 2011). In computational algebraic geometry, syzygy-based MFR supports the study of singularities and invariants of varieties (Evans and Griffith, 1981) and improves the complexity of Gröbner-basis computations (Eisenbud et al., 2001; Stewart, 1993; Weispfenning, 1992; Capani et al., 1997). Beyond pure algebra, MFR has been used to speed up persistent-homology computation in TDA (Wasserman, 2018) and to analyze structures arising in physics and bioinformatics, including Calabi–Yau singularities (Polchinski, 1994), gauge fields (Hitchin, 2003), and gene regulatory networks (Li et al., 2011). These strengths—exposing structured dependencies while reducing computational redundancy—motivate our integration of MFR into CoT for LLM reasoning. By decomposing symbolic dependencies among intermediate steps, we prune redundant branches, streamline intermediate computation, and improve the transparency and efficiency of reasoning, providing a more structured pathway for solving complex problems.

Logical Completeness

Exactness

Seamless Integration

Minimality Optimal Solution

Optimal Solution

Figure 2: The mathematical, abstract analogy, and CoT analogy of Module, Freeness, Mapping, Exactness, Minimality, and Betti Number.

such as:

a1f1 + a2f2 + ··· + akfk = 0, ai ∈ R, (1)

and the collection of all such relations forms the first syzygy module.

An MFR of M is an exact sequence:

###### ··· → F2 → F1 → F0 → M → 0, (2)

where each Fi is free and the maps contain no unit elements. The resolution is minimal if the number of generators in each Fi is smallest possible, making the structure unique up to isomorphism.

The i-th Betti number, βi(M) = rank(Fi), quantifies the number of generators at each level and reflects the complexity of M.

Motivation for SoT. As illustrated in Figure 3, we reinterpret CoT reasoning within LLMs through the lens of MFR. Each reasoning stage—e.g., problem understanding, decomposition, subgoal chaining, and final decision—is treated as a generator or relation within a module. This enables SoT to capture latent logical dependencies and hierarchical structure, moving beyond CoT’s linear stepwise form toward a more principled and algebraically grounded representation.

3 Method

###### 3.1 Preliminaries

In algebraic geometry and homological algebra, Minimal Free Resolution is a foundational tool for analyzing finitely generated modules over a ring, offering a structured way to reveal their internal dependencies and complexity.

The SoT reasoning paradigm employs the MFR framework to model the structure of CoT reasoning paths in LLMs. Each reasoning step—such as problem interpretation, task decomposition, subgoal chaining, and final inference—is abstracted as generators or relations within modules, reinterpreting CoT as algebraic objects with generative structures and latent dependencies, rather than merely a flat sequence of symbols.

Key Concepts. A module M over a ring R general-

izes vector spaces by allowing scalars from R. M is free if it admits a basis: M ∼= R⊕n; it is finitely generated if a finite set {m1,...,mn} spans M via Rmi.

A syzygy refers to a relation among generators,

|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>Freeness<br><br>Generate necessary auxiliary conditions from the initial problem.<br><br>[Figure 11]<br><br>C1. Spring (k = 6.5N/m, Δx = 0.3m) C2. Horizontal launch<br><br>[Figure 12]<br><br>[Figure 13]<br><br>C3. Heights (table = 4m, ramp = 0.5m)<br><br>[Figure 14]<br><br>Cn. Mass (m = 0.06 kg）<br><br>[Figure 15]<br><br>Module<br><br>[Figure 16]<br><br>Base Problem<br><br>[Figure 17]<br><br>[Figure 18]<br><br>Betti Number<br><br>LLM<br><br>[Figure 19]<br><br>Base Problem Freeness<br><br>[Figure 20]<br><br>required<br><br>[Figure 21]<br><br>Mapping Number<br><br>[Figure 22]<br><br>Task<br><br>[Figure 23]<br><br>Profile<br><br>[Figure 24]<br><br>Q: A60g box islaunched by a spring (k = 6.5 N/m, compressed 0.3 m) on a frictionless 30° ramp (0.5 m high) atop a 4 m table. Find how far (Δd) it lands. Ignore friction.<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>Profile<br><br>Num. Date. Option.<br><br>[Figure 29]<br><br>[Figure 30]<br><br>Task<br><br>Math. Planning. Fact Reasoning.<br><br>(1)<br>(2)<br><br><br>...<br><br>... ...<br><br>Ci not in {C1, ..., Ci-1} for i > 1 Ci = fi(module, C1, ..., Ci-1)|
|---|

|[Figure 31]<br><br>[Figure 32]<br><br>Exactness<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>Accuracy Complexity Simplicity<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>Spring releases → kinetic energy → Moves up ramp → Launches off ramp → Falls and lands.<br><br>No gaps：Clear, connected logic<br><br>[Figure 40]<br><br>[Figure 41]<br><br>The Final answer is 2.94<br><br>[Figure 42]<br><br>redundancy check<br><br>[Figure 43]<br><br>Minimality<br><br>Mapping(M1,Mapping(M1, M2,M2,M3M3......Mn)Mn) Si<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>Conciseness Evaluation<br><br>Logical Soundness Evaluation<br><br>[Figure 47]<br><br>[Figure 48]<br><br>M1<br><br>Slides down ramp → Immediately flies off →<br><br>Lands.<br><br>Pops out → Lands.<br><br>[Figure 49]<br><br>Step-linked<br><br>[Figure 50]<br><br>Defined conditions<br><br>[Figure 51]<br><br>No gaps<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>M2<br><br>[Figure 55]<br><br>M3<br><br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>Aggregate<br><br>[Figure 59]<br><br>Traverse<br><br>...<br><br>...<br><br>...<br><br>|
|---|

[Figure 60]

|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>Mapping<br><br>|[Figure 65]<br><br>[Figure 66]<br><br>Freeness Purpose Result<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>C1.<br><br>[Figure 72]<br><br>Cn.<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>| | |
|---|---|
| | |
| | |
| | |
| | |
<br><br>| | |
|---|---|
| | |
<br><br>Calculate spring potential energy<br><br>[Figure 76]<br><br>Convert PE to velocity<br><br>[Figure 77]<br><br>C3. Get(3.5m)fall→heighttime<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>C2. Simplifiesmotion<br><br>[Figure 81]<br><br>[Figure 82]<br><br>... ...<br><br>PE k x 0.2925J 2<br><br>= 1  2 =<br><br>m s m<br><br>PE v 3.5 /<br><br>= 2 =<br><br>s<br><br>g<br>h<br><br><br>t 0.84<br><br>= 2 =<br><br>d = vt = 2.94m<br><br>...|
|---|
<br><br>[Figure 83]<br><br>Strategy<br><br>[Figure 84]<br><br>C1.<br><br>[Figure 85]<br><br>C2.<br><br><br>[Figure 86]<br><br>Strategy-M1<br><br>[Figure 87]<br><br>[Figure 88]<br><br>C3.<br><br>[Figure 89]<br><br>[Figure 90]<br><br>Cn. Strategy-MN<br><br>[Figure 91]<br><br>Strategy-M2<br>Strategy-M3<br><br><br>[Figure 92]<br><br>Freeness<br><br>... ...|
|---|

Figure 3: SoT Overview. Through the six modules: Module Freeness, Mapping, Exactness, Minimality, and Betti, MFR can decompose and deconstruct a complex reasoning problem, aiding LLMs in generating more accurate answers.

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

Module Initial Complex Problem

Complete Lego Structure

Betti Numbers

Number of Hints per Leve

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

Number of Blocks per Level

Module: Initial Complex Problem

Decomposed Subproblems with Hints.

Freeness : Auxiliary Conditions

Module Initial Complex Problem

Freeness

Complete Lego Structure

Basic Units

Complex problems often exhibit high-dimensional structures, intricate logical dependencies, and ambiguous constraints, rendering direct solutions intractable. In this initial module, we conceptualize the problem as a unified entity of interdependent substructures, necessitating systematic decomposition. The goal is not immediate resolution but to unravel latent complexity, identify core logical components, and establish a foundation for subsequent modular reasoning. This stage ensures that downstream processes operate within well-defined boundaries, setting the stage for structured analysis.

Freeness involves generating auxiliary conditions to simplify problems and clarify logical relationships. By introducing intermediate variables, hypotheses, or formalized constraints, complex problems are partitioned into smaller, independent subproblems. This reduces dimensionality and provides clear entry points for resolution, enhancing the tractability of subsequent steps while maintaining logical coherence.

Betti Numbers

Number of Hints per Leve

Solution Strategy

Mapping

Number of Blocks per Level

Assembly Instructions

Decomposed Subproblems with Hints.

Freeness

Logical Completeness

Exactness

Seamless Integration

Basic Units

Solution Strategy

Mapping

Minimality Optimal Solution

Optimal Solution

Assembly Instructions

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

Logical Completeness

Exactness

Seamless Integration

Module Initial Complex Problem

Complete Lego Structure

Minimality Optimal Solution

Betti Numbers

Optimal Solution

Number of Hints per Leve

Number of Blocks per Level

Decomposed Subproblems with Hints.

Freeness

Basic Units

Mapping: Solving Strategies

Solution Strategy

Mapping

Assembly Instructions

Effective problem-solving requires systematic strategies that map auxiliary conditions to actionable reasoning paths. These paths must exhibit (i) directness, minimizing redundant operations, and (ii) logical soundness, grounding each step in explicit premises. LLMs iteratively solve subproblems along these paths, linking current conditions to new conclusions, ensuring continuity and interpretability throughout the reasoning chain.

Logical Completeness

Exactness

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

Seamless Integration

Minimality Optimal Solution

Module Initial Complex Problem

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

Optimal Solution

Complete Lego Structure

Betti Numbers

Module Initial Complex Problem

Betti Numbers: Quantifying Complexity

Number of Hints per Leve

Complete Lego Structure

Number of Blocks per Level

To measure decomposition complexity, we introduce Betti numbers as a quantitative metric, where each number reflects the count of auxiliary conditions at a reasoning level. Higher Betti numbers indicate greater intricacy, signaling potential optimization opportunities. We leverage LLMs to minimize these numbers by regenerating or filtering conditions, streamlining the decomposition process. Betti numbers thus guide reasoning efficiency, balancing structural complexity with computational simplicity.

Betti Numbers

Decomposed Subproblems with Hints.

Number of Hints per Leve

Freeness

Basic Units

Number of Blocks per Level

Solution Strategy

Mapping

Decomposed Subproblems with Hints.

Freeness

Assembly Instructions

Basic Units

Logical Completeness

Exactness

Solution Strategy

Seamless Integration

Mapping

Assembly Instructions

Minimality Optimal Solution

Exactness: Logical Completeness

Optimal Solution

Logical Completeness

Exactness

Seamless Integration

Logical completeness is critical for rigor, preventing gaps or leaps in the reasoning chain. Each step must derive from explicit premises, with implicit assumptions formalized to avoid invalid conclusions. LLMs verify consistency across auxiliary condi-

Minimality Optimal Solution

Optimal Solution

Mathematical Concept LEGO/Puzzle Analogy Chain of Thought Analogy

Module Initial Complex Problem

Complete Lego Structure

Betti Numbers

Number of Hints per Leve

3.2.1 SoT Reasoning.

tions, subproblems, and the final solution, systematically reviewing the chain against original constraints. This ensures a coherent and reliable framework, minimizing errors from incomplete logic.

Number of Blocks per Level

Decomposed Subproblems with Hints.

Freeness

Figure 3 explains the entire process of SoT. The SoT framework begins with an initial complex problem M, characterized by high-dimensional structure, intricate logical dependencies, and ambiguous constraint relationships. We treat M as a unified representation of interdependent substructures, setting the stage for systematic decomposition. To analyze the inherent complexity of M, we introduce Betti numbers as quantitative indicators of structural complexity. Specifically, the Betti number corresponds to the number of auxiliary conditions required for decomposition, with higher values suggesting deeper structural intricacy. These values guide the subsequent generation of auxiliary components and indicate potential optimization points.

Basic Units

Solution Strategy

Mapping

Assembly Instructions

Logical Completeness

Exactness

Seamless Integration

Minimality: Optimal Solutions

Minimality Optimal Solution

Optimal Solution

Minimality optimizes efficiency by deriving solutions with the fewest auxiliary conditions and simplest strategies, without sacrificing correctness. Redundant conditions are filtered, and reasoning paths are pruned of unnecessary steps. LLMs prioritize straightforward approaches over complex transformations, yielding concise, clear solutions. This reduces overall complexity, enhancing both computational efficiency and solution transparency.

After introducing the above six concepts, LLMs can be directly stimulated conversationally. Figure 6 illustrates the conversational framework of these six concepts. First, complex reasoning problems are defined as a free module. Then, through the Betti number and auxiliary conditions of degrees of freedom, the complex problem is reduced and decomposed. By analyzing both the module and degrees of freedom simultaneously, the model can construct multiple mappings to solve the problem. Finally, minimality and accuracy are used to select the optimal solution.

We then introduce the notion of Freeness, which refers to the generation of auxiliary conditions that serve to simplify and structurally reorganize the problem. These conditions—denoted as Ψ = {ψ1,...,ψβ}—include intermediate hypotheses, latent constraints, or explicit sub-conclusions, and are generated via autoregressive reasoning using LLMs conditioned on previously established states. Each ψj decomposes the original problem into finer-grained logical units, reducing its dimensional complexity and establishing solvable subspaces.

- 3.2 Syzygy of Thoughts

The process of LLM reasoning for complex problems can be viewed as solving a complex reasoning problem M through MFR. The entire process is formalized via an MFR sequence, where the structured representation of the reasoning process is a set of mappings φi induced by the LLM, forming the following sequence:

To resolve the decomposed problem, the framework constructs a set of mappings M = {S1,...,Sµ}, each corresponding to a distinct reasoning path. These mappings are designed to satisfy two essential properties: directness, ensuring that each inference step avoids redundant operations, and logical soundness, ensuring that every conclusion is firmly grounded in its premises. Each mapping Sk is generated using LLMs to traverse from auxiliary conditions to intermediate or final conclusions, thereby forming a continuous and interpretable reasoning chain.

··· −−−−−−→LLM(φn) Fn ··· −−−−−→LLM(φ0) F0 −→ M → 0

(3)

Moreover, we define the intermediate space Fi as a free module constructed from basic reasoning units:

To ensure logical completeness (Exactness) of the reasoning process, all steps within each reasoning path are verified for inferential closure. Implicit assumptions are formalized, and the consistency of auxiliary conditions, mappings, and final conclusions is evaluated. This validation step leverages the LLM’s reasoning and scoring capabilities via a scoring function ϕ(·). Finally, the optimal reasoning sequence Sopt is parsed using structured pattern-matching strategies (e.g., regular expressions, JSON parsing) to extract the final solution.

R · mij (4)

Fi =

j

where R is the reasoning space (LLM Reasoning Space), mij are the basic units (Tokens) used in the reasoning construction. Each φi is induced by the LLM, representing a single-step logical deduction in the CoT.

##### 4 Experiments

###### 4.1 Applicability Evaluation

Configuration. As shown in Table 1, we compare the proposed SoT method with four representative reasoning strategies: Chain-of-Thought (CoT) (Wei et al., 2022), Self-Consistency CoT (CoT-SC) (Wang et al., 2022b), Graph-of-Thought (GoT) (Besta et al., 2024), and Atom-of-Thought (AoT) (Teng et al., 2025). The evaluation covers nine benchmarks spanning four categories: mathematical word problems (GSM8K, SVAMP, MultiArith, ASDiv, AQUA), general-knowledge and multitask QA (MMLU, BBH), temporal reasoning (Date), and relational logical reasoning (CLUTRR). For CoT-SC, we follow the standard setting with n = 5 sampled reasoning paths. GoT is instantiated with beam size b = 4 and search depth L = 3, while AoT uses a maximum search depth of 5, consistent with the original works. We instantiate all methods on five backbone LLMs: GPT4o-mini, Qwen2.5-Coder-7B-Instruct, Qwen2.5VL-72B-Instruct, Gemma-3-27b-it, and Gemma-

- 3-12b-it. For each backbone, we keep decoding hyper-parameters (e.g., temperature, top-p, maximum generation length) fixed across methods and only change the reasoning protocol. We report mean accuracy and standard deviation over multiple random seeds to account for sampling variance. Results. From Table 1, SoT consistently achieves the best performance across all benchmarks and all backbone models. On the mathematical reasoning benchmarks, SoT improves over the strongest baseline AoT by around 1 percentage point on GSM8K and SVAMP for every model family (e.g., on GPT-
- 4o-mini, GSM8K accuracy increases from 95.3% to 96.4%), while also yielding smaller but steady gains on MultiArith and ASDiv. On the more challenging AQUA dataset, SoT brings additional improvements of roughly 0.7–1.0 points across different backbones, indicating that searching over structured thoughts helps the model better handle multi-step algebraic reasoning and eliminate distractor options. For general-knowledge and multitask reasoning, SoT further improves MMLU and BBH by approximately 0.7–1.0 points over AoT on all models, with Qwen2.5-VL-72B and Gemma-

- 3-27b-it achieving the highest absolute accuracies. Finally, on temporal and relational logical reasoning (Date and CLUTRR), SoT again attains the best

Accuracy vs #Paths vs Token Cost

Macro-averageAccuracyacross9tasks(%)(better)

88

|Avg Tokens pe<br><br>|r Problem ( be<br><br>~1000 tokens|tter)| | | | |
|---|---|---|---|---|---|---|
| | |SoT (<br><br>| |=7,µ=5) SoT (<br><br>|=9,µ=7)| |
|SoT ( =5<br><br>|AoT (d=3)<br><br>,µ=3)| | |AoT (d=5)| | |
|G<br><br>SoT ( =3,µ=2)<br><br>|oT (b=2,L=2)<br><br>| | |GoT (b=|CoT-SC (n=8)<br><br>4,L=3)| |
|CoT-SC|(n=3)|CoT-SC| |(n=5)| | |
|CoT| | | | | | |
| | |CoT CoT<br><br>| |Method Type<br><br>-SC GoT<br><br>|AoT SoT<br><br>| |
| | | | | | | |

86

84

82

80

78

76

74

72

0 2 4 6 8 10

Avg #Candidate Paths / Syzygies per Problem

Figure 4: Cost–accuracy trade-off across different reasoning frameworks. Each bubble corresponds to one configuration of CoT, CoT-SC, GoT, AoT, or the proposed SoT, and the bubble area is proportional to the average number of tokens consumed per problem.

results under every backbone, with gains of up to about 1 point, suggesting that the structured search procedure is beneficial beyond pure arithmetic reasoning. Overall, SoT dominates all baselines in every setting in Table 1 while maintaining comparable or even lower variance, demonstrating that it provides a stable and broadly applicable improvement to LLM reasoning.

###### 4.2 Cost Test

To systematically compare the trade-offs between accuracy and search cost across different reasoning frameworks, we construct the three-dimensional bubble plot shown in Figure 4, which places five methods under a unified lens: CoT, CoT-SC, GoT, AoT, and our proposed SoT. All methods are evaluated using the same model and the same set of nine benchmark tasks. In the plot, the vertical axis represents the macro-average accuracy over the nine tasks, the horizontal axis denotes the average number of candidate reasoning paths generated per problem, and the bubble area is proportional to the average number of tokens consumed per problem. We provide the detailed definition of our path-counting metric in Appendix 4.2.

Based on the above definitions, we visualize a representative set of method configurations. Along the horizontal axis, the number of candidate paths increases from left to right; along the vertical axis, the macro-average accuracy improves from bottom to top; and the bubble size grows with the token cost. When the number of candidate paths and the token budget are comparable, SoT attains

Math Reasoning Gene. Mul. Tem. Log. GSM8K SVAMP MultiArith ASDiv AQUA MMLU BBH Date CLUTRR GPT-4o-mini

Method

- CoT (Wei et al., 2022) 85.4±0.5% 84.7±1.6% 89.5±0.5% 92.3±0.8% 65.3±1.5% 63.4±0.5% 66.6±0.7% 52.1±1.0% 66.2±1.3% CoT-SC (Wang et al., 2022b) 89.9±0.6% 85.8±0.8% 89.7±0.5% 93.3±1.0% 70.7±1.0% 67.1±0.5% 69.0±0.8% 54.7±0.7% 72.2±1.1% GoT (Besta et al., 2024) 93.2±0.7% 89.2±0.6% 90.9±0.5% 94.2±0.6% 73.3±1.1% 71.3±0.8% 71.1±0.9% 65.2±0.7% 74.2±1.0% AoT (Teng et al., 2025) 95.3±0.5% 91.5±0.6% 91.0±0.5% 94.1±0.5% 75.1±1.0% 74.7±0.9% 72.4±0.7% 74.7±0.8% 75.2±1.2% SoT (Ours) 96.4±0.6% 92.6±0.7% 92.1±0.7% 95.2±0.7% 76.0±1.1% 75.6±0.5% 73.2±0.6% 75.6±0.7% 76.1±0.7%

Qwen2.5-Coder-7B-Instruct

CoT (Wei et al., 2022) 77.5±1.4% 82.7±1.0% 84.6±0.9% 87.3±1.4% 60.9±1.2% 55.4±0.6% 47.4±0.6% 31.3±1.1% 20.4±1.1% CoT-SC (Wang et al., 2022b) 80.0±0.9% 83.9±1.1% 86.8±0.6% 89.8±0.6% 62.0±1.4% 56.1±0.8% 49.1±0.9% 32.7±1.0% 20.8±1.0% GoT (Besta et al., 2024) 84.8±0.7% 87.5±0.8% 88.1±0.5% 92.5±0.6% 62.9±1.2% 56.8±0.7% 53.4±0.7% 34.7±0.9% 23.8±1.0% AoT (Teng et al., 2025) 88.5±0.6% 90.0±0.5% 88.4±0.7% 94.1±0.5% 63.0±1.0% 56.8±1.1% 57.0±1.0% 36.2±0.8% 26.4±1.3% SoT (Ours) 89.5±0.6% 91.0±0.8% 89.4±0.6% 95.2±0.6% 63.7±1.0% 57.5±0.8% 57.7±0.7% 36.6±1.0% 26.7±1.1%

Qwen2.5-VL-72B-Instruct

- CoT (Wei et al., 2022) 86.4±0.9% 87.2±0.9% 90.1±0.6% 92.3±0.6% 81.4±0.6% 80.4±0.5% 77.6±0.3% 75.5±0.5% 70.4±0.6% CoT-SC (Wang et al., 2022b) 88.9±0.6% 88.0±0.6% 91.1±0.8% 93.6±0.6% 83.7±1.0% 82.7±0.5% 78.8±0.8% 77.8±0.8% 74.8±0.7% GoT (Besta et al., 2024) 92.7±0.5% 92.1±0.4% 91.6±0.6% 93.1±0.5% 86.8±0.8% 83.7±0.6% 82.2±0.6% 79.2±0.7% 77.0±1.2% AoT (Teng et al., 2025) 95.3±0.4% 95.1±0.5% 91.0±0.5% 93.5±0.5% 88.8±0.7% 83.7±0.7% 84.7±0.6% 79.7±0.8% 78.4±1.1% SoT (Ours) 96.4±0.3% 96.2±0.2% 92.1±0.3% 94.6±0.5% 89.8±0.7% 84.7±0.8% 85.7±0.8% 80.6±1.5% 79.3±2.6%

###### Gemma-3-27b-it

- CoT (Wei et al., 2022) 83.4±0.8% 86.2±0.9% 84.2±0.9% 90.8±0.4% 80.6±1.0% 71.1±1.6% 71.0±1.7% 77.2±1.4% 65.6±3.9% CoT-SC (Wang et al., 2022b) 86.9±1.7% 86.8±1.8% 84.5±0.9% 92.0±0.5% 85.2±1.9% 73.0±3.5% 73.0±3.6% 80.0±2.7% 66.2±4.8% GoT (Besta et al., 2024) 91.7±0.7% 91.5±0.6% 88.3±0.5% 91.3±0.5% 87.5±1.1% 78.8±1.0% 79.3±0.9% 80.3±0.8% 72.8±1.4% AoT (Teng et al., 2025) 93.2±0.4% 93.1±0.4% 88.4±0.5% 91.8±0.6% 85.6±0.8% 81.7±0.8% 82.9±0.7% 77.4±0.7% 76.1±1.2% SoT (Ours) 94.2±0.4% 95.3±0.2% 91.9±0.3% 93.3±0.4% 88.1±0.8% 83.4±1.4% 84.7±1.2% 79.6±1.6% 78.4±2.7%

Gemma-3-12b-it

- CoT (Wei et al., 2022) 83.5±1.9% 79.3±2.0% 82.7±0.8% 91.0±0.4% 69.2±3.2% 68.4±2.7% 64.9±1.8% 78.0±1.9% 49.3±4.1% CoT-SC (Wang et al., 2022b) 85.9±1.8% 80.8±0.9% 85.1±0.7% 93.2±0.4% 71.5±2.1% 70.4±1.6% 66.5±3.7% 80.0±1.8% 52.0±2.0% GoT (Besta et al., 2024) 89.2±0.7% 86.8±0.6% 86.8±0.5% 93.9±0.6% 74.5±1.0% 71.6±1.1% 68.0±1.0% 81.5±0.8% 53.7±1.3% AoT (Teng et al., 2025) 91.4±0.5% 91.8±0.4% 87.5±0.6% 93.5±0.5% 76.7±1.2% 71.9±1.3% 68.7±1.1% 82.0±0.9% 54.8±1.5% SoT (Ours) 92.5±0.4% 92.9±0.5% 88.5±0.7% 94.6±0.3% 77.6±1.6% 72.7±1.5% 69.5±2.6% 82.9±0.7% 55.4±2.9%

Table 1: Performance comparison of CoT, CoT-SC (n = 5), GoT (b = 4, L = 3), AoT (depth < 6), and the proposed SoT method across various tasks, including mathematical reasoning, general knowledge (Gene.), multitask QA (Mul.), temporal reasoning (Tem.) and logical reasoning (Log.). We report mean accuracy ± standard deviation over 5 seeds. Results are grouped by model architectures, with SoT consistently achieving the best performance across benchmarks while maintaining comparable or lower variance.

substantially higher macro-average accuracy than CoT-SC, GoT, and AoT. Moreover, as the number of syzygies increases from µ = 3 to µ = 7, the marginal gains diminish noticeably—indicating that SoT achieves near-saturated performance even with only a small number of syzygies. Overall, SoT occupies a particularly favorable region in the three-way trade-off space: with a moderate number of candidate paths and an acceptable token cost, it achieves the highest macro-average accuracy and significantly enhances the effectiveness of each individual candidate path through its syzygy-based structure.

###### 4.3 Abalation Study

Betti Number Sensitivity. The SoT reasoning chain regulates topological constraints through the Betti number, and its value directly affects performance. To investigate the optimal configuration of the Betti number and its dynamic impact on structural expressiveness and regularization effects, we designed ablation experiments to reveal its key

role in optimizing the reasoning process. For this, we systematically adjusted the Betti number and recorded accuracy changes under different mapping configurations. The goal of the experiment was to analyze the impact of the Betti number on the reasoning chain’s expressiveness and identify the performance saturation point. The results are presented in Figure 5, showing the trend of accuracy changes with varying Betti numbers, with particular attention to its non-monotonic properties and optimal configuration.

Configuration. The experiment was based on GPT4o-mini and conducted on the GSM8K dataset. The Betti number was incremented from 1 to 10, with mapping set to 1, 2, and 3 to evaluate performance under different topological decomposition granularities.

Results. As shown in Figure 5 (left), accuracy changed non-monotonically as the Betti number increased under mapping configurations of 1, 2, and 3. Specifically, when the Betti number started from 1, accuracy improved significantly, indicating that a

Mapping Accuracy Curve

###### Stability Boxplot: SoT vs CoT

Stability Line Chart: SoT vs CoT

96

CLUTRR - SoT

SVAMP - SoT

| | |
|---|---|
| | |

MultiArith - SoT

90

90

ASDiv - SoT AQUA - SoT MMLU - SoT BBH - SoT

94

80

80

Accuracy(%)

92

Date - SoT

GSM8K - SoT

70

70

CLUTRR - CoT

90

SVAMP - CoT

MultiArith - CoT

60

60

ASDiv - CoT AQUA - CoT MMLU - CoT BBH - CoT

88

Mapping=3 Mapping=2 Mapping=1

50

50

SoT (Ours)

86

Date - CoT

CoT

GSM8K - CoT

0 1 2 3 4 5 6 7 8

0.0 0.2 0.4 0.6 0.8

CLUTRRSVAMPMultiArithASDivAQUAMMLUBBH DateGSM8K

Betti Number

Temperature

- Figure 5: Overall analysis: (left) Betti number sensitivity, (middle) stability under different temperatures, and (right) accuracy distribution across tasks.

small number of topological constraints effectively enhanced structural expressiveness. However, as the Betti number increased further, the performance gain gradually diminished and stabilized. The optimal saturation point was reached at 7, beyond which there was no noticeable improvement. The results reveal the dynamic regulatory effect of the Betti number in reasoning chain modeling, and that moderate topological constraints are crucial for optimizing SoT’s structured design, providing strong support for its theoretical foundation.

Results. As shown in Figure 5 (right), SoT’s accuracy remains stable across temperature variations (0.0 to 1.0) with minimal fluctuation. In contrast, CoT’s accuracy fluctuates significantly, especially at higher temperatures, where the amplitude increases. The boxplot in Figure 5 (middle) reveals that SoT’s accuracy is concentrated with little variation and few outliers across datasets, even at high temperatures. On the other hand, CoT shows a wider spread with more outliers, especially at higher temperatures, where reasoning consistency decreases. This analysis indicates that CoT’s high temperature-induced diversity weakens logical coherence, while SoT maintains stability.

Stability Analysis. The temperature parameter influences the diversity of content generated by LLMs, which may challenge reasoning stability. We aimed to explore whether the SoT method could maintain its performance advantage under varying temperature conditions, thereby validating the adaptability of its structured design. To this end, we systematically evaluated SoT and CoT under different temperature conditions, focusing on the stability of accuracy and the relationship between reasoning diversity and consistency. The experiment involved adjusting the temperature step by step and observing the performance trends of the two methods under changing diversity.

##### 5 Conclusions

We introduced SoT, an MFR-inspired test-time reasoning framework that reframes CoT from a single linear trace into a syzygy-structured family of interdependent reasoning paths. By mapping MFR notions to concrete prompting and selection operations, SoT explicitly organizes auxiliary conditions, constructs candidate “syzygies,” verifies logical closure, and prunes redundancy to produce compact yet faithful solutions. Across nine benchmarks spanning math, general knowledge, multitask QA, temporal reasoning, and relational logic, and across five LLM backbones, SoT consistently outperforms strong baselines while maintaining comparable or lower variance, indicating both improved accuracy and more stable inference. Moreover, SoT occupies a favorable region in the accuracy–cost–paths trade-off, achieving near-saturated gains with only a small number of syzygies, and exhibiting strong robustness under sampling diversity.

Configuration. The experiment was based on GPT-

- 4o-mini, with the Betti number fixed at 7 and mapping set to 3. The nine test datasets included GSM8K, SVAMP, MultiArith, ASDiv, AQUA (mathematical reasoning), MMLU (generalization), BBH, Date (multi-task QA), and CLUTRR (logical reasoning). The temperature parameter was adjusted from 0.0 to 1.0 in steps of 0.1, and the performance of SoT and CoT was compared. Five reasoning paths were generated in each experiment, and the average accuracy was recorded to reduce the impact of randomness.

##### 6 Limitations

While SoT demonstrates strong reasoning capabilities, further research is needed. We plan to extend the topological decomposition to multimodal reasoning, including images and tables, to test its adaptability in cross-modal tasks. Additionally, incorporating iterative MFR concepts will refine problem-solving steps, optimize reasoning paths, and improve SoT’s efficiency in multimodal and high-dimensional tasks.

##### References

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and 1 others. 2024. Graph of thoughts: Solving elaborate problems with large language models. In AAAI.

Nicolás Botbol, Alicia Dickenstein, and Hal Schenck.

2021. The simplest minimal free resolutions in P1 × P1. In Commutative Algebra: Expository Papers.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. In NeurIPS.

Antonio Capani, Gabriel De Dominicis, Gianfranco Niesi, and Lorenzo Robbiano. 1997. Computing minimal finite free resolutions. In J. Pure Appl. Algebra.

Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wangxiang Che. 2025. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. In arXiv.

Ri-Xiang Chen. 2011. Hilbert functions and free resolutions.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. In arXiv.

Yezeng Chen, Zui Chen, and Yi Zhou. 2024a. Braininspired two-stage approach: Enhancing mathematical reasoning by imitating human thought processes. In arXiv.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024b. Self-play fine-tuning converts weak language models to strong language models. In arXiv.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro

Nakano, and 1 others. 2021. Training verifiers to solve math word problems. In arXiv.

Yuntian Deng, Yejin Choi, and Stuart Shieber. 2024. From explicit CoT to implicit CoT: Learning to internalize CoT step by step. In arXiv.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT.

David Eisenbud. 2013. Commutative algebra: With a view toward algebraic geometry. In Book.

David Eisenbud, Daniel R Grayson, Mike Stillman, and Bernd Sturmfels. 2001. Computations in algebraic geometry with Macaulay 2. In Book.

E Graham Evans and Phillip Griffith. 1981. The syzygy problem. In Ann. Math.

Nathan Fieldsteel and Uwe Nagel. 2021. Minimal and cellular free resolutions over polynomial OI-algebras. In arXiv.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. ToRA: A tool-integrated reasoning agent for mathematical problem solving. In arXiv.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In NeurIPS.

Nigel Hitchin. 2003. Generalized calabi–yau manifolds. In Q. J. Math.

Shima Imani, Liang Du, and Harsh Shrivastava. 2023. MathPrompter: Mathematical reasoning using large language models. In arXiv.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024. The impact of reasoning step length on large language models. In arXiv.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In NeurIPS.

Joshua Ong Jun Leang, Aryo Pradipta Gema, and Shay B Cohen. 2024. CoMAT: Chain of mathematically annotated thought improves mathematical reasoning. In arXiv.

Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li FeiFei, Fei Xia, and Brian Ichter. 2023. Chain of code: Reasoning with a language model-augmented code emulator. In arXiv.

Junlong Li, Daya Guo, Dejian Yang, Runxin Xu, Yu Wu, and Junxian He. 2025. CodeI/O: Condensing reasoning patterns via code input-output prediction. In arXiv.

Xiaoliang Li, Chenqi Mou, Wei Niu, and Dongming Wang. 2011. Stability analysis for discrete biological models using algebraic methods. In Math. Comput. Sci.

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-ofthought reasoning. In IJCNLP-AACL.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. In NeurIPS.

Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heintz, and Dan Roth. 2023. Recent advances in natural language processing via large pre-trained language models: A survey. In ACM Comput. Surv.

Hoang H Nguyen, Ye Liu, Chenwei Zhang, Tao Zhang, and Philip S Yu. 2023. Cof-cot: Enhancing large language models with coarse-to-fine chain-of-thought prompting for multi-domain nlu tasks. In arXiv.

Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. 2023. Skeleton-ofthought: Large language models can do parallel decoding. In ENLSP-III.

Joseph Polchinski. 1994. What is string theory? In arXiv.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, and 1 others. 2018. Improving language understanding by generative pre-training. In arXiv.

Leonardo Ranaldi, Marco Valentino, Alexander Polonsky, and Andrè Freitas. 2025. Improving chain-ofthought reasoning via quasi-symbolic abstractions. In arXiv.

Maria Evelina Rossi and Leila Sharifan. 2009. Minimal free resolution of a finitely generated module over a regular local ring. In J. Algebra.

Bilgehan Sel, Ahmad Al-Tawaha, Vanshaj Khattar, Ruoxi Jia, and Ming Jin. 2023. Algorithm of thoughts: Enhancing exploration of ideas in large language models. In arXiv.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In ICML.

Saksham Sahai Srivastava and Ashutosh Gandhi. 2024. MathDivide: Improved mathematical reasoning by large language models. In arXiv.

Gilbert W Stewart. 1993. On the early history of the singular value decomposition. In SIAM Rev.

Fengwei Teng, Zhaoyang Yu, Quan Shi, Jiayi Zhang, Chenglin Wu, and Yuyu Luo. 2025. Atom of thoughts for Markov LLM test-time scaling. In arXiv.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NeurIPS.

Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. 2022a. Towards understanding chain-of-thought prompting: An empirical study of what matters. In arXiv.

Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023a. MathCoder: Seamless code integration in LLMs for enhanced mathematical reasoning. In arXiv.

Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, William Yang Wang, and Alessandro Sordoni. 2023b. Guiding language model reasoning with planning tokens. In arXiv.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022b. Self-consistency improves chain of thought reasoning in language models. In arXiv.

Larry Wasserman. 2018. Topological data analysis. In Annu. Rev. Stat. Appl.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS.

Volker Weispfenning. 1992. Comprehensive gröbner bases. In J. Symb. Comput.

Fangzhi Xu, Qiushi Sun, Kanzhi Cheng, Jun Liu, Yu Qiao, and Zhiyong Wu. 2024. Interactive evolution: A neural-symbolic self-training framework for large language models. In arXiv.

Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. 2025. Buffer of thoughts: Thoughtaugmented reasoning with large language models. In NeurIPS.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS.

Dian Yu, Baolin Peng, Ye Tian, Linfeng Song, Haitao Mi, and Dong Yu. 2024. SIAM: Self-improving codeassisted mathematical reasoning of large language models. In arXiv.

Fei Yu, Anningzhe Gao, and Benyou Wang. 2023a. OVM: Outcome-supervised value models for planning in mathematical reasoning. In arXiv.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023b. MetaMath: Bootstrap your own mathematical questions for large language models. In arXiv.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. MAmmoTH: Building math generalist models through hybrid instruction tuning. In arXiv.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. STAR: Bootstrapping reasoning with reasoning. In NeurIPS.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain of thought prompting in large language models. In arXiv.

Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing. 2023a. Verify-and-edit: A knowledge-enhanced chain-of-thought framework. In arXiv.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, and 1 others. 2023b. A survey of large language models. In arXiv.

Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H Chi, Quoc V Le, and Denny Zhou. 2023. Take a step back: Evoking reasoning via abstraction in large language models. In arXiv.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and 1 others. 2022. Least-to-most prompting enables complex reasoning in large language models. In arXiv.

Algorithm 1 LLM Minimal Free Resolution (LLM-MFR) for Reasoning Module M Input: Reasoning question M; metadata D Output: Final answer Answer

- 1: (β,µ) ← ANALYSE(M,D) ▷ Estimate regularity β and projective dimension µ (Sec. 3)
- 2: ▷ Generate auxiliary conditions (Freeness)
- 3: Ψ ← [] ▷ Initialize empty list of auxiliary conditions
- 4: for j ← 1 to β do
- 5: ψj ← LLMGENERATEAUXCONDITION(F0,...,Fj−1,φ0,...,φj−1) ▷ Use LLM to generate the j-th freeness condition
- 6: Ψ ← Ψ ∪ {ψj}
- 7: ▷ Construct candidate resolutions (syzygies)
- 8: M ← [] ▷ Initialize empty list of candidate solutions
- 9: for k ← 1 to µ do
- 10: Sk ← LLMRESOLVE(M,Ψ) ▷ LLM resolves M under auxiliary conditions Ψ
- 11: M ← M ∪ {Sk}
- 12: ▷ Select optimal resolution from syzygies (Minimality)
- 13: k∗ ← ARGMINSCORE(M,ϕ) ▷ ϕ: scoring function implemented via LLM scoring prompt
- 14: Sopt ← Sk∗
- 15: Answer ← EXTRACTFINALANSWER(Sopt,D) ▷ Ensure logical Exactness via pattern matching (e.g., regex / JSON parsing)
- 16: return Answer

##### A Appendix

###### A.1 Full Pseudocode of the SoT Pipeline

In this appendix, we present the full pseudocode of the SoT pipeline. Algorithm 1 provides an algorithmic implementation of the LLM-based minimal free resolution discussed in Section 3, turning the conceptual modules in Figure 3 and the abstract diagram in Figure 6 into an executable procedure.

Explanation of Algorithm 1. We briefly explain the main components of Algorithm 1, which instantiates the SoT framework described in Section 3. Analyse (M,D) and homological complexity. Given a reasoning problem M and its metadata D, the subroutine ANALYSE(M,D) estimates the regularity β and the projective dimension µ. These quantities summarize the Betti-number profile introduced in Section 3: β determines the number of auxiliary (freeness) conditions, and µ specifies how many candidate resolutions (syzygies) are explored. Larger values of (β,µ) indicate higher problem complexity. Generating auxiliary conditions (Freeness). The loop over j = 1,...,β constructs a set Ψ of auxiliary conditions, corresponding to the “Freeness” module in Figure 3. Each call to LLMGENERATEAUXCONDITION conditions on previously created free modules F0,...,Fj−1 and maps φ0,...,φj−1 to propose a new constraint ψj. These constraints guide the LLM toward structurally coherent reasoning decompositions. Constructing candidate syzygies (Syzygy of Thoughts). The loop over k = 1,...,µ implements the “Syzygy of Thoughts” mechanism from Section 3. Each call to LLMRESOLVE(M,Ψ) yields a candidate reasoning chain Sk that satisfies the auxiliary conditions. From an algebraic viewpoint, each Sk functions as a syzygy relating different generators of M, and the list M collects all such candidates. Selecting a minimal resolution (Minimality). Once the candidate set M is formed, the subroutine ARGMINSCORE(M,ϕ) selects the best resolution according to a scoring function ϕ. This function, implemented via an LLM scoring prompt, evaluates both structural regularity (e.g., consistency with freeness and mapping constraints) and task-specific correctness. This step aligns with the “Minimality” module in Figure 3. Extracting the final answer (Exactness). Finally, EXTRACTFINALANSWER(Sopt,D) converts the selected reasoning chain into the final task-specific answer. Pattern matching techniques (e.g., regular expressions or JSON parsing) ensure the output is consistent with the resolved structure and adheres to benchmark formatting requirements, realizing the “Exactness” principle.

- A.2 Conceptual Illustration of the SoT Pipeline

Explanation. Figure 6 provides a concrete walkthrough of the SoT framework using a simple arithmetic problem. The illustration makes explicit how SoT transforms a natural-language question into a structured multi-stage reasoning process.

(1) Module and Betti Numbers. The LLM first rewrites the question into a normalized “module” form and performs a lightweight complexity analysis. Based on the estimated problem structure, it predicts the required number of auxiliary conditions, corresponding to the Betti-number estimate β discussed in Section 3. (2) Freeness. The LLL generates non-redundant auxiliary conditions (e.g., price per pair, number of children, number of pairs per child). These auxiliary facts form the initial free modules F0,F1,..., which constrain the search space of valid reasoning chains. (3) Mappings. The LLM organizes auxiliary conditions into candidate reasoning strategies by combining freeness conditions in different orders. Each such combination corresponds to a candidate syzygy, i.e., a structured reasoning chain that reflects different possible mappings between the modules in the minimal free resolution. (4) Exactness. The LLM inspects each candidate strategy and verifies whether every reasoning step is justified, gap-free, and based on explicitly stated assumptions. This matches the exactness requirement in the algebraic SoT formulation. (5) Minimality. Among all valid strategies, the LLM selects the one with the smallest reasoning footprint, balancing accuracy, complexity, and simplicity. This corresponds to the minimality principle described in Section 3. (6) Final Answer. Once the optimal syzygy is chosen, the LLM extracts the final numerical answer while ensuring consistency with all earlier reasoning stages. This mirrors the final extraction step in Algorithm 1.

Relation to Algorithm 1. This example illustrates the same pipeline implemented formally in Algorithm 1: ANALYSE(M,D) corresponds to Step 1; LLMGENERATEAUXCONDITION and LLMRESOLVE correspond to Steps 2–3; ARGMINSCORE corresponds to Step 4; and EXTRACTFINALANSWER corresponds to Step 5. The illustration provides an intuitive view complementary to the algorithmic description.

###### A.3 Complex Reasoning Evaluation

To compare the complex problem reasoning capabilities of SoT with mainstream methods, we evaluated SoT and compared it with extensive opensource data from various papers (Chen et al., 2025). Specifically, we applied the SoT reasoning chain to GPT-4o-mini and tested its reasoning accuracy on the GSM8K and MATH datasets, while comparing it with other methods from the literature.

Experimental Configuration: The test model was GPT-4o-mini, with the Betti number set to 7 and mapping set to 3. The baselines included CoT and various popular chain-of-thought variants (such as MathPrompter, QuaSAR, MathDivide, etc.). The datasets used were GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021).

Method Model GSM8k MATH

No-CoT (Deng et al., 2024) Mistral-7B 38.0% ICoT-SI (Deng et al., 2024) Mistral-7B 51.0% -

- - RecurrentBlock-3.5B 42.1% MathCoder-CL (Wang et al., 2023a) Code-Llama-7B 67.8% 30.2% MAmmoTH (Yue et al., 2023) Code-Llama-7B 59.4% Brain (Chen et al., 2024a) Code-Llama-7B 74.0% SQ-VAE (Wang et al., 2023b) Llama-2-7B 40.0% 7.0% Self-Rewarding (Chen et al., 2024b) Llama-2-7B 40.0% 10.7% STaR (Zelikman et al., 2022) Llama-2-7B 58.2% 16.0% ENVISIONS (Xu et al., 2024) Llama-2-7B 59.0% 19.0% MetaMath (Yu et al., 2023b) Llama-2-7B 66.5% ToRA-Code (Gou et al., 2023) Llama-2-7B 72.6% OVM (Yu et al., 2023a) Llama-2-7B 73.7% -

- - Llama-3.1-8B 56.7% 20.3%

- - Llama-3.1-70B 85.5% 41.4%

- - Llama-3.1-405B 89.0% 53.8%

- - NuminaMath-7B-CoT 75.4% 55.2%

- - DeepSeek-Coder-7B 77.4% 44.4%

- - Qwen2-7B 79.9% 44.2%

- - Qwen2-Math-7B 80.4% 50.4% SIaM (Yu et al., 2024) Qwen-2-Math-Base 81.5% 50.0%

- - Internlm2-math-plus-7B 84.0% 54.4% OMI2 (Li et al., 2025) Qwen2.5-Coder-7B 84.1% 72.3% CODEI/O++ (Li et al., 2025) Qwen2.5-Coder-7B 85.7% 72.1% PyEdu (Li et al., 2025) Qwen2.5-Coder-7B 85.8% 71.4% CODEI/O (Li et al., 2025) Qwen2.5-Coder-7B 86.4% 71.9% OC-SFT-1 (Li et al., 2025) Qwen2.5-Coder-7B 86.7% 70.9% WI (Li et al., 2025) Qwen2.5-Coder-7B 87.0% 71.4% WI (Full) (Li et al., 2025) Qwen2.5-Coder-7B 87.0% 71.1% OMI2 (Full) (Li et al., 2025) Qwen2.5-Coder-7B 88.5% 73.2%

- - DeepSeekMath-7B-RL 88.2% 51.7%

CoMAT (Leang et al., 2024) GPT-4 93.7% CoT (Ranaldi et al., 2025) GPT-4 94.5% FCoT (Lyu et al., 2023) GPT-4 95.0% MathPrompter (Imani et al., 2023) GPT-4 95.6% QuaSAR (Radford et al., 2018) GPT-4 96.5% MathDivide (Srivastava and Gandhi, 2024) GPT-4 96.8% SoT (Ours) GPT-4o-mini 96.0% 79.1%

Table 2: Performance on Different Benchmarks

Experimental Results: As shown in Table 2, SoT achieved 96.0% on GSM8K and 79.1% on MATH on GPT-4o-mini. On GSM8K, SoT’s performance is close to the best result achieved by GPT-4 (e.g., MathDivide’s 96.8%) (Chen et al., 2025) and outperforms the best 7B model (e.g., OMI2 Full’s 88.5%) (Chen et al., 2025). On MATH, SoT’s 79.1% significantly outperforms the best mainstream 7B models (e.g., OMI2 Full’s 73.2%). This result indicates that SoT can achieve reasoning capabilities close to those of closed-source large models on lightweight models, significantly narrowing

|[Figure 93]<br><br>Chat with LLMs<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>So, the question is:<br><br>[Figure 100]<br><br>John buys 2 pairs of shoes for each of his 3<br><br>children. They cost $60 each. How much did he pay?<br><br>[Figure 101]<br><br>complexity_analysis:<br><br>This is a arithmetic problem that involves<br><br>understanding the wording to correctly calculate<br><br>the total shoes and cost.<br><br>[Figure 102]<br><br>[Figure 103]<br><br>How many helper conditions do you think we'll need?<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>(Betti_numbers) Based on the complexity of theproblem,We'll need 3 auxiliaryconditions<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>Chat with LLMs<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>.<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>Chat with LLMs<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>Yes, Strategy 2 is :<br><br>1) Step-linked 2) No gaps 3) Defined conditions<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>The final answer is $360<br><br>[Figure 133]<br><br>(Freeness)<br><br>•Freeness1: Each pair costs $60.<br>•Freeness2: Each child receives 2 pairs<br>•Freeness3: Total children = 3<br><br><br>(Final_answer) What is the final answer,<br><br>and is it based on complete and correct<br><br>reasoning?<br><br>Can these conditions be organized into a<br><br>clear, logical sequence of steps to solve the<br><br>problem efficiently?<br><br>What known conditions are required to solve the<br><br>problem? List all non-redundant auxiliary facts.<br><br>(Minimality) Is the solution strategy minimal, using the fewest necessary steps and conditions?<br><br>(Mappings)<br><br>Strategy 1 ...<br>Strategy 2:<br><br><br>(1) Freeness3 Freeness2 Multiply the number of children by 2 to get the total shoe pairs:<br><br>[Figure 134]<br><br>3×2 = 6 (pairs of shoes)<br><br>(2) Freeness1 Multiply shoe pairs by price to find total<br><br><br>cost: 6×60 = 360 (dollars)<br><br>Strategy N ...<br><br>(Exactness) Does each step in the solution have a clear basis, with no gaps or undefined<br><br>assumptions?<br><br>(Module) John buys 2 pairs of shoes for<br><br>each of his 3 children. They cost $60 each.<br><br>How much did he pay?<br><br>Yes, Strategy 2 is<br><br>1) Accuracy 2) Complexity 3) Simplicity|
|---|

- Figure 6: Conceptual framework illustrating the process of navigating LLMs’ latent space through modular reasoning. The diagram highlights key components, including Module Freeness, Betti Numbers, Mappings, Exactness, and Minimality, to decompose and solve a complex problem. While the figure aids in understanding the theoretical underpinnings and logical flow of the methodology, it does not represent the exact procedural steps of the method.

the performance gap between open-source models and GPT-4, validating its superiority in complex mathematical reasoning.

- graph-expansion calls that generate intermediate nodes without a final answer do not count as paths, though their token usage is included in the bubble size.
- • AoT(d): Along the Markov chain, solv-

ing each of {Qi,Gi,Qi+1} produces a directly scorable candidate answer. Hence each solve(·) call is counted as one path, and AvgPaths approximates the average number of solver calls per Markov chain.

- • SoT(β,µ): In SoT, LLMRESOLVE generates µ

###### A.4 Path Counting and Token-Cost Metrics

To align the notion of “paths” across different methods, we adopt a unified definition. For a method m

and a sample x, let Ncand(m) (x) denote the number of candidate reasoning trajectories generated on x that

contain a complete chain of reasoning and yield a directly scorable answer. The horizontal axis in our plots reports

syzygies {Sk}µk=1, each constituting a full structured reasoning chain capable of produc-

1 |D| x∈D

Ncand(m) (x),

AvgPaths(m) =

ing an answer. Therefore the number of candidate paths is exactly the number of syzygies:

which reflects how many full candidate solutions, on average, the method explores per problem.

Ncand(SoT)(x) = µ.

Under this formulation:

The β auxiliary calls used during the freeness stage do not count as paths, but all of their token usage is included in the overall cost.

- • CoT: Each problem produces exactly one chain-of-thought, hence AvgPaths ≈ 1.
- • CoT-SC(n): Each problem samples n independent CoT trajectories followed by majority voting, thus AvgPaths ≈ n.
- • GoT(b,L): During multi-round graph search, each invocation that produces a complete candidate solution is counted as a path. Internal

The bubble area in the figure corresponds to the average tokens per problem, computed as the total number of tokens consumed by all LLM calls (generation, intermediate analysis, candidate solutions, and scoring), averaged over the evaluation set. This metric reflects the overall reasoning cost of each method.

