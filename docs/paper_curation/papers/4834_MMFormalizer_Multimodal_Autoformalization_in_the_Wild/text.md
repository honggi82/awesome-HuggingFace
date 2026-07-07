[Figure 1]

## MMFormalizer: Multimodal Autoformalization in the Wild

Jing Xiong1, Qi Han1, Yunta Hsieh2, Hui Shen2, Huajian Xin3, Chaofan Tao1, Chenyang Zhao6, Hengyuan Zhang1, Taiqiang Wu1, Zhen Zhang4, Haochen Wang1, Zhongwei Wan5, Lingpeng Kong1, Ngai Wong1

1The University of Hong Kong 2University of Michigan, Ann Arbor 3University of Edinburgh 4University of California, Santa Barbara 5Ohio State University 6University of California, Los Angeles

junexiong@connect.hku.hk

[Figure 2]

https://MMFormalizer.github.io/ https://huggingface.co/datasets/menik1126/PhyX-AF

[Figure 3]

# arXiv:2601.03017v1[cs.CL]6Jan2026

### Abstract

Autoformalization, which translates natural language mathematics into formal statements to enable machine reasoning, faces fundamental challenges in the wild due to the multimodal nature of the physical world, where physics requires inferring hidden constraints (e.g., mass or energy) from visual elements. To address this, we propose MMFORMALIZER, which extends autoformalization beyond text by integrating adaptive grounding with entities from real-world mathematical and physical domains. MMFORMALIZER recursively constructs formal propositions from perceptually grounded primitives through recursive grounding and axiom composition, with adaptive recursive termination ensuring that every abstraction is supported by visual evidence and anchored in dimensional or axiomatic grounding. We evaluate MMFORMALIZER on a new benchmark, PHYX-AF, comprising 115 curated samples from MathVerse, PhyX, Synthetic Geometry, and Analytic Geometry, covering diverse multimodal autoformalization tasks. Results show that frontier models such as GPT-5 and Gemini3-Pro achieve the highest compile and semantic accuracy, with GPT-5 excelling in physical reasoning, while geometry remains the most challenging domain. Overall, MMFORMALIZER provides a scalable framework for unified multimodal autoformalization, bridging perception and formal reasoning. To the best of our knowledge, this is the first multimodal autoformalization method capable of handling classical mechanics (derived from the Hamiltonian), as well as relativity, quantum mechanics, and thermodynamics.

### 1 Introduction

Recent advances in large language models (LLMs) show strong capabilities in formal reasoning (Wang et al., 2023b,a; Xiong et al., 2023; Xuejun et al., 2025). In geometry domain, symbolic reasoning

has become a key research frontier (Lu et al., 2021; Murphy et al., 2024; Zhang et al., 2024b; Ping et al., 2025; He et al., 2025), supported by rich geometric data and domain-specific formal languages such as context-free grammar-based predicate forms (Lu et al., 2021; Ping et al., 2025) and the Condition Declaration Language (CDL) (Zhang et al., 2025, 2024b). However, the above systems rely mainly on symbolic inputs, leaving a gap between visual geometric understanding and formal reasoning, motivating the need for multimodal autoformalization. Meanwhile, LEAN (mathlib Community, 2025; de Moura and Ullrich, 2021) enables rigorous encoding and verification of geometric reasoning, providing a potential pathway.

Multimodal Autoformalization in geometry (Murphy et al., 2024; He et al., 2025) emerges as a promising research direction, going beyond the mere recognition of geometric entities in text. This progress is driven not only by the growing interest in connecting perceptual and formal reasoning but also by the extensive collection of geometryrelated lemmas and tactics in mathlib (mathlib Community, 2025), which provides a robust foundation for integrating perceptual representations with formal reasoning. Nevertheless, multimodal autoformalization beyond the geometric domain remains largely underexplored, particularly in modeling complex phenomena that arise in the real physical world. This is partly due to the lack of supporting infrastructure; although dependencies like PhysLean (Tooby-Smith, 2024) exist, there is still a shortage of tools to integrate them into autoformalization frameworks. To address this issue, we introduce a toolkit for deploying physical theorem search engines and their dependencies.

Another core challenge is: How can a multimodal autoformalization system be grounded in the wild? A representative example of this difficulty is

Newton’s laws of motion. Although derived from empirical observation and logical reasoning, their formulation reveals a fundamental limitation: relations among quantities such as mass, length, and time cannot be obtained from logic alone but require empirical grounding through measurement. This dependence shows that even a mathematically coherent system must refer back to physical experience through explicit dimensional definitions. Consequently, an autoformalization system cannot arise purely from observation data; it must integrate dimensional analysis as a constraint that links formal statements with empirical interpretability. While nondimensionalization (Buckingham, 1914) abstracts away physical units and offers a potential means to avoid dealing with dimensions, it requires human-provided physical expertise. In contrast, we adopt a dimensional formalism here to maintain a direct bridge between formal reasoning and measurable reality, thereby eliminating the need for intervention by domain experts.

A more advanced example is provided by the theory of relativity. Rather than being constructed inductively from empirical data, relativity arises from a compact set of foundational axioms—chiefly, the invariance of the speed of light and the equivalence of physical laws across all inertial reference frames. From these axioms, the entire theoretical edifice can be deduced through thought experiments, yielding results such as time dilation, length contraction, and the mass–energy equivalence (E = mc2).

Crucially, the above example demonstrates that identifying the fundamental axioms reveals the most elementary dimensional groundings of the theory: the unification of space and time through the constant c, and the equivalence of mass and energy through the same invariant. In this sense, the formal system itself determines what counts as the basic dimensional primitives. We extend a similar idea to the framework of classical mechanics, as illustrated in Fig. 2, where the three Newtonian laws are derived from the Hamiltonian formalism.

In this work, we focus on addressing the core challenge of identifying fundamental axioms that ground multimodal autoformalization in real-world settings. Our key insight is to recursively ground visual elements through the process of dimensional grounding to determine an appropriate point of recursive termination, using the notion of dimension as a bridge between physical semantics and formal logic. We introduce MMFORMALIZER, which recursively translates perceptual inputs such as di-

agrams, text, or physical scenes into structured formal statements by identifying intermediate lemmas, aligning them with symbolic predicates and image elements, and refining them into verifiable axioms within the LEAN (de Moura and Ullrich, 2021) system.

To ensure validity and generality, we build PHYX-AF, a benchmark for evaluating multimodal autoformalization across diverse topics. The dataset is designed to minimize redundancy between visual and textual modalities, so that the model cannot rely solely on textual information while ignoring visual input thereby enforcing genuine multimodal reasoning. Our main contributions are threefold:

- • We propose MMFORMALIZER, a multimodal autoformalization framework that recursively decomposes physical objects into axiomatic steps, translating perceptual elements into consistent formal statements.
- • We introduce tools for searching and interacting with PhysLean, allowing the integration of visual elements into reusable lemmas in LEAN, thereby linking perception with verifiable proofs.
- • We present PHYX-AF, a benchmark for evaluating multimodal autoformalization in the wild, which introduces new challenges for assessing formal reasoning.

### 2 Related Work

#### 2.1 Autoformalization

Autoformalization (Wu et al., 2022) refers to the process of translating informal mathematical text into formal languages such as Isabelle/HOL (Nipkow et al., 2002), LEAN (de Moura and Ullrich, 2021), MetaMath (Megill and Wheeler, 2019), and Coq (Barras et al., 1999). Previous advancements in this field have expanded the scope to multilingual autoformalization (Jiang et al., 2023). Later works propose frameworks for semantic alignment, e.g., FormalAlign (Lu et al., 2024), and retrieval-based methods like RAutoformalizer (Liu et al., 2025). Recent efforts, such as Kimina-prover (Wang et al., 2025) and Mathesis (Xuejun et al., 2025), emphasize long-horizon formal reasoning to enhance the robustness of the formalization process. Despite notable progress, multimodal autoformalization remains an unexplored area.

- Figure 1: Multimodal autoformalization performance of five representative models across mathematical and physical domains, reported in terms of compilation accuracy (left) and human verification accuracy (right).

- Figure 2: Conceptual dependency graph illustrating how Newton’s three laws give rise to key structures in classical mechanics, including momentum, Hamiltonian formulation, phase space, and spacetime representations.

lem solving to build interpretable reasoning systems. Lu et al. (2021) address geometry problems by transforming them into formal symbolic representations grounded in first-order predicate logic. Similarly, Ping et al. (2025) propose an automated geometry problem-solving framework that deductive reasoning within a first-order logical language. Zhang et al. (2024b) introduce a neurosymbolic system that combines a many-sorted firstorder logic–based formal language with a hypergraph neural network to achieve readable and traceable human-like geometric reasoning. Zhang et al. (2025) present DFE-GPS, a multimodal geometry solver that leverages a many-sorted first-order logic formal language to improve diagram comprehension and reasoning.

While first-order and many-sorted logical formal languages provide a structured and interpretable foundation for symbolic geometric reasoning, they fundamentally differ from type-theoretic frameworks such as the dependent type theory underlying LEAN. In first-order logic, entities are abstract symbols belonging to predefined sorts (e.g., Point, Line, Circle), and reasoning proceeds through predicate inference over fixed domains. By contrast, dependent type theory treats mathematical objects as constructive types, where the existence and properties of an object depend on previously defined data. This paradigm enables not only logical deduction but also object construction—turning “proofs” into verifiable computational objects.

In this context, Murphy et al. (2024) pioneer autoformalization in geometry using LEAN, formalizing multimodal content in Euclidean geometry. However, models still face difficulties with complex geometric figures, such as regular hexagons, which require recursively composing intricate angle relations. To formalize such shapes, the formal system needs to define dependent types that encode geometric constraints, such as equal-length sides and angle constructions, within the LEAN. These constraints are expressed as types parameterized by the underlying geometric properties.

The connection between these paradigms lies in their shared goal of achieving rigor and interpretability: first-order logical formalization captures relational structure and deductive closure, while dependent type theory extends this framework into a constructive, computational foundation. Consequently, while systems such as Formal-

#### 2.2 Multimodal Formalization

Multimodal formalization integrates symbolic reasoning, formal languages, and geometric prob-

Geo (Zhang et al., 2023) excel at symbolic geometric reasoning within a fixed formal universe, LEAN-style frameworks can represent and manipulate mathematical objects that depend on parameters, constraints, and even continuous quantities.

This distinction becomes crucial for multimodal autoformalization in the wild, where models must describe and reason about complex physical entities with dimensions rather than purely symbolic ones. Let us continue with the example in Figure 2. Although we can mathematically and rigorously derive the classical Newtonian mechanical system from the Hamiltonian, it still depends on the definition of its dimensional quantity—namely, energy. Such dependency cannot be adequately captured within a first-order or many-sorted logical formal language, where symbols and relations are dimensionless abstractions. By contrast, in dependent type theory, the dimensional structure of quantities (e.g., length, time, mass, energy) can be encoded directly into the type system, ensuring that all physical expressions are well-typed and dimensionally consistent by construction. Thus, entities like the Hamiltonian, which intrinsically carry physical dimensions, can only be faithfully and constructively defined within a type-theoretic framework.

### 3 Multimodal Autoformalization

In this section, we demonstrate how complex physical entities can be synthesized from a compact set of primitive constructors. Higher-order formal structures (e.g., PropChain, SceneGraph) are recursively composed from perceptually grounded primitives and visual evidence, forming a hierarchy of formally consistent representations.

#### 3.1 Recursive Grounding

Given an input image I : Image, we define a recursive grounding process:

RG : I → PropChain, (1)

which incrementally constructs a dependent hierarchy of formal propositions whose semantics are grounded in perceptual data. This process is operationalized by the MMFORMALIZER model in three inductive stages.

Definition of Lemma. We define a lemma as a dependent pair consisting of a formal proposition and its constructive inhabitant:

Lemma := Σ(P : PropChain), (p : P), (2)

where P denotes a formal statement, and p : P represents its proof term within the underlying typetheoretic system.

Propositional Grounding. A propositional grounding is formalized as a dependent chain of lemmas:

PropChain := Σ(Lt : List Lemma), (3)

where each element Lt corresponds to a perceptually grounded proposition indexed by t, denoting its position or level within the recursive grounding hierarchy. This chain ensures formal consistency across layers while preserving compositional dependencies between propositions.

Formally, the recursive grounding establishes a compositional mapping between perceptual scene structures and propositional representations:

lift : SceneGraph → PropChain, (4)

such that for each subgraph Gt ⊆ SceneGraph, there exists a corresponding lemma Lt within the propositional chain.

Visual Decomposition. Let SceneGraph denote a dependent type encoding primitive visual entities and their spatial relations:

SceneGraph := Σ(Vt : List Primitive), Rel(Vt),

(5) where Vt = [v1,v2,...,vn] is a list of visual primitives, each vi : Primitive representing a perceptual entity such as a point, line, or region. The type Primitive thus enumerates the basic perceptual elements, formally defined as

Primitive ::= point | line | region. (6)

The dependent component Rel(Vt) specifies the spatial or topological relations among the primitives in Vt,

Rel(Vt) : Π(vi,vj ∈ Vt), SpatialRel(vi,vj), (7) where SpatialRel captures geometric or structural predicates such as adjacent, parallel, or contained_in. The parsing function

parse : I → SceneGraph (8) decomposes an input image I into a base-level scene graph G0 : SceneGraph.

Each primitive vi ∈ V0 within G0 is assigned an initial semantic hypothesis lt, which is typically a short phrase as illustrated by the brown regions in Figure 2. These hypotheses constitute the perceptual priors of the grounding chain RG(I), forming the base layer L0 within the propositional structure.

[Figure 4]

[Figure 5]

[Figure 6]

Figure 3: The pipeline overview consists of three stages: Recursive Grounding, identifying physical primitives (the red parts in the figure, e.g., the Hamiltonian or dimensional quantities) for Termination, and Axiom Composition. The blue parts in the figure indicate the compiler checking process. The green part indicates the formal statements we retrieved from the dependency library.

3.2 Recursive Termination We define a mapping

map : (Vt,lt) → Lt, (9)

which maps each visual element Vt into the space of axioms, thereby constructing a hierarchy of verifiable axiomatic statements within the LEAN. At each stage, we ensure that every formal abstraction is both supported by perceptual grounding and capable of constraining subsequent semantic parsing. We implement the above process through two major phases: the Grounding and the Termination, culminating in termination through axiomatic or dimensional grounding.

Grounding. We use the decomposed intermediate lemma statements as queries to search for theorems in LeanSearch (Gao et al., 2024), retrieving them from mathlib (mathlib Community, 2025) and PhysLean (Tooby-Smith, 2024).

In local deployment, LeanSearch runs on a LEAN + mathlib + PhysLean environment, uses indexing scripts to extract all declarations (theorems, lemmas, definitions, structures) together with their type signatures and comments, encodes these statements into embeddings, and enables offline semantic search to retrieve the most top-k relevant formal items for any Lt. For each node t ∈ N, we assume a substructure Gt ⊆ Gt−1 associated with a predicate set Pt. Formally, we define

Pt := p(vi,lt | Gt−1) | vi ∈ Vt,, (10)

where Vt denotes the set of visual elements extracted from picture, and lt represents the predicted informal term that encodes their semantic relationships. The set Pt is thus regarded as a collection of informal statements, each expressing a perceptually grounded yet not fully formalized correspondence

between the visual configuration and its prospective symbolic abstraction. In this way, Pt serves as a conceptual bridge linking perceptual structures to their formal statements. The alignment operator

Grounding : Gt−1 → Pt → Lemma (11)

then maps p(vi,lt) into the corresponding formal statements. We then prompt the LLMs to select from these statements the ones that best align with the current Pt, and designate it as the Lt This grounding is inductively extended by defining

##### Lt+1 := Grounding(Gt,Pt+1). (12)

Termination. The recursion terminates when Pt is grounded either in a primitive dimensional or an axiom:

 

dim(p), if p ∈ Dt, axiom(p), if p ∈ At.

Termination(Pt) =



(13)

where Dt denotes the physical dimensional quantity, which in geometric problems usually corresponds to length; At denotes the fundamental axiom; dim maps a physical quantity to its fundamental dimensional basis.

#### 3.3 Axiom Composition

At this stage, we define the AxiomChain as the formal closure of the propositional hierarchy:

AxiomChain := Σ(At : List Axiom,Dt : List Dim)).

(14) Formally, there exists a grounding mapping

##### ground : PropChain → AxiomChain. (15)

AxiomChain represents the final, ontologically closed layer of the multimodal formalization hierarchy, where every perceptually grounded proposition is instantiated within a physically or axiomatically formal statements.

Through successive applications of Compose, performed by the LLM, all child nodes are recursively combined to construct a non-leaf node:

##### Compose(Lkt+1,Gt,Pt) → Lt, (16)

where Lkt+1 denotes the set of lemmas from the child nodes of node t and k is the number of childs. This process captures the transition from perceptual substructures to hierarchically dependent propositions within the recursive grounding framework. After each composition at node t, the resulting formal statements are passed through a syntax checker for compilation verification, ensuring their structural and logical validity within the LEAN.

Each composed type thus preserves both dimensional structure and symbolic manipulability, enabling reasoning over geometric, physical, and logical domains within a unified formal statements. In this sense, TypeComposition acts as the metalevel operator that aligns LEAN’s dependent type hierarchy with the multimodal autoformalization pipeline:

TypeComposition : SceneGraph → PropChain

→ AxiomChain, (17)

ensuring that every physical entity corresponds to a grounded formal statement. Through such recursive compositionality, LEAN supports the construction of formal models that faithfully reflect the generative structure of physical and conceptual reality—where primitives (points, vectors, forces) combine to form law-constrained, formal systems.

#### 3.4 Semantic Checking

The semantic checking module verifies whether each formal statement is jointly supported by visual and textual evidence. For each image–text–formula triplet, the checker outputs an indicator value, where 1 denotes semantic acceptance and 0 denotes rejection. If all propositions Lt ∈ PropChain are assigned 1, the chain is considered valid and grounded across modalities.

Figure 4: Distribution of Problem Types.

### 4 Experiment

#### 4.1 Experimental Setup

PhyX-AF. To rigorously evaluate the multimodal autoformalization capability, we construct the PHYX-AF benchmark comprising 115 meticulously curated samples drawn from representative sources, including MATHVERSE (Zhang et al., 2024a), PHYX (Shen et al., 2025), and SYNTHETIC GEOMETRY (Trinh et al., 2024), which is further augmented with a rule-based computation engine (Hubert et al., 2025) for automated synthesis and verification, as well as an extended ANALYTIC GEOMETRY dataset designed to assess autoformalization within coordinate-based geometric contexts. We present the dataset statistics in Figure 4, which illustrates that the benchmark covers both mathematics and physics domains.

Data Filtering. To ensure genuine multimodal reasoning, we apply a strict visual dependency criterion: only problems where the diagram is indispensable for solving are retained; samples solvable by text alone are removed.

Multimodal Mathematical Setup. This setup primarily samples data from MATHVERSE, covering three categories: Plane Geometry, Solid Geometry, and Function, mainly drawn from a real-exam setting with authentic geometry problems designed for in-distribution autoformalization. Within this setup, the Function category mainly involves solving multivariate and higher-order equations based on given function graphs, while the Plane Geometry and Solid Geometry categories primarily focus on proof-based problems.

Analytic Geometry. We include analytic geometry problems extracted from GEOMETRY3K (Lu et al., 2021) and GEOINT (Wei et al., 2025), as well as procedurally generated 2D and 3D figures composed of geometric primitives (points, lines, planes,

- Table 1: Comparison across MATHVERSE, PHYX, SYNTHETIC GEOMETRY, and ANALYTIC GEOMETRY datasets. We report Compile accuracy, semantic correctness, and human verification results. The Modern category under PHYX includes problems from both quantum mechanics and relativity.

Model Metric MATHVERSE PHYX SYNTHETIC GEOMETRY ANALYTIC GEOMETRY

Plane Geometry Solid Geometry Function Modern Mechanics

Electromagnetism

Therodynamics

Plane Geometry Solid Geometry Plane Geometry Solid Geometry

Img Text Img Text Img Text Img Text Img Text Img Text Img Text Img Text Img Text Img Text Img Text Frontier Model

GPT-5

Compile 24.0 8.0 28.0 8.0 80.0 0.0 71.4 37.5 71.4 16.7 66.7 50.0 0.0 20.0 40.0 30.0 10.0 20.0 40.0 0.0 20.0 100.0 Semantics 20.0 0.0 28.0 8.0 30.0 0.0 71.4 12.5 71.4 0.0 50.0 0.0 0.0 0.0 40.0 30.0 0.0 10.0 20.0 0.0 20.0 100.0

Human Check 12.0 – 28.0 – 30.0 – 42.9 – 71.4 – 50.0 – 0.0 – 20.0 – 10.0 – 20.0 – 0.0 –

Gemini-3-Pro

Compile 76.0 8.0 52.0 4.0 100.0 40.0 14.3 57.1 28.6 42.9 16.7 16.7 0.0 0.0 70.0 40.0 80.0 70.0 60.0 0.0 60.0 40.0 Semantics 76.0 0.0 52.0 0.0 40.0 0.0 14.3 42.9 28.6 28.6 33.3 0.0 0.0 0.0 70.0 30.0 80.0 70.0 60.0 0.0 40.0 20.0

Human Check 72.0 – 52.0 – 40.0 – 14.3 – 28.6 – 33.3 – 0.0 – 40.0 – 50.0 – 60.0 – 40.0 –

Gemini-2.5-Pro

Compile 24.0 8.0 32.0 8.0 60.0 60.0 14.3 0.0 0.0 0.0 0.0 0.0 0.0 20.0 30.0 30.0 30.0 30.0 20.0 0.0 60.0 0.0 Semantics 20.0 0.0 24.0 0.0 60.0 0.0 14.3 0.0 0.0 0.0 0.0 0.0 0.0 20.0 30.0 30.0 10.0 20.0 20.0 0.0 40.0 0.0

Human Check 20.0 – 24.0 – 60.0 – 0.0 – 0.0 – 0.0 – 0.0 – 20.0 – 10.0 – 20.0 – 60.0 –

Open-source Model

Qwen3-VL-235B

Compile 28.0 4.0 20.0 0.0 30.0 30.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 20.0 0.0 20.0 0.0 Semantics 16.0 0.0 16.0 0.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 20.0 0.0 20.0 0.0

Human Check 12.0 – 12.0 – 10.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 20.0 –

Qwen2.5-VL-72B

Compile 0.0 0.0 12.0 0.0 20.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 Semantics 0.0 0.0 8.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

Human Check 0.0 – 4.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 – 0.0 –

- Table 2: Ablation Study on the Model Components: Evaluating the Effect of Different Modifications on Performance. The experiments include: (1) Ablation on synthesizer without code: We did not provide the model with reference code while synthesizing new types. (2) Ablation on termination condition: We explicitly specified the decomposition recursion termination condition. (3) Ablation on grounding with image: We investigated the impact of inputting images during the grounding process. (4) Ablation on pass@k: We set sampling attempts (k=3) for each node.

Metric MATHVERSE PHYX SYNTHETIC GEOMETRY ANALYTIC GEOMETRY

Electromagnetism

Therodynamics

Plane Geometry

Solid Geometry

Solid Geometry original

Plane Geometry

Solid Geometry

Plane Geometry

Function Modern Mechanics

Compile 50.0 33.3 66.7 50.0 60.0 100.0 100.0 33.3 0.0 33.3 0.0 Semantics 25.0 33.3 66.7 50.0 60.0 100.0 0.0 33.3 0.0 0.0 0.0

Average nodes 21.8 10.3 14.3 27.3 15.0 13.5 35.0 15.7 21.3 12.7 30.0

- Average depth 3.4 5.0 3.0 3.0 4.4 4.5 6.0 2.3 4.3 3.3 5.0 Ablation on synthesizer without code

Compile 25.0 33.0 100.0 100.0 40.0 0.0 100.0 33.3 66.7 0.0 0.0 Semantics 25.0 16.7 33.3 100.0 40.0 0.0 0.0 33.3 66.7 0.0 0.0

Ablation on termination condition

Average nodes 34.3 24.3 14.3 5.5 16.4 31.5 28.0 16.0 22.0 17.0 21.0

- Average depth 4.3 4.0 3.0 3.0 6.6 7.0 8.0 3.0 5.3 3.7 3.0 Ablation on grounding with image

Compile 25.0 33.0 100.0 100.0 80.0 50.0 0.0 33.3 66.7 0.0 0.0 Semantics 25.0 16.7 66.7 100.0 60.0 50.0 0.0 33.3 66.7 0.0 0.0

###### Ablation on pass@k

Compile 50.0 66.7 100.0 100.0 80.0 0.0 0.0 66.7 100.0 10.0 0.0 Semantics 25.0 16.7 33.3 100.0 60.0 0.0 0.0 66.7 66.7 0.0 0.0

arcs, polyhedra). This task requires LLMs to understand both geometric and numerical relations, often involving uncommon or composite numerical and visual configurations that are absent from mathlib or other pretraining corpora.

Multimodal Physical Setup. This setup involves visual scenes and physical interactions from natural or simulated environments, assessing visual–physical generalization. The PHYX subset (21.7%) includes problems from mechanics, electromagnetism, thermodynamics, and relativity and quantum mechanics. Each image is paired with a textual statement specifying geometric or physical constraints, evaluating end-to-end visual-to-formal grounding under realistic perceptual noise.

Synthetic Geometry Settings. In this setup, we aim to test the model’s ability to synthesize new dependent types and constructors at test time, thereby evaluating out-of-distribution autoformalization and formal generalization beyond its pretrained knowledge. Many constructed geometric objects and relational schemas do not exist in mathlib or other pretrained corpora. Thus, the MMFORMALIZER must create novel dependent types at test time, testing its ability to generalize beyond its training distribution. This setup evaluates out-ofdistribution autoformalization and synthesis of new type constructors for unseen perceptual structures. Following AlphaGeometry (Trinh et al., 2024), we adopt a symbolic deduction engine for Olympiadlevel geometric synthesis. The specific synthesis rules and verification setup can be found in Ap-

- Table 3: Accuracy of Semantic Checking by Different LLMs, evaluated through human validation. The table reports the agreement rate between each model’s semantic checking and human verification on LEAN code generated by G (GPT-5) and Q (Qwen3-VL-235B). ⋄ indicates that the model failed to generate compilable code, thereby precluding human verification.

Accuracy (%) G Q

Data Model

GPT-5 69.2 76.5 Gemini-2.5-Pro 84.6 88.9 Gemini-3-Pro 84.6 77.9 Qwen3-VL-235B 30.8 66.7 Qwen2.5-VL-72B 15.4 77.9

MATHVERSE

GPT-5 71.4 ⋄ Gemini-2.5-Pro 78.6 ⋄ Gemini-3-Pro 78.6 ⋄ Qwen3-VL-235B 53.9 ⋄ Qwen2.5-VL-72B 76.9 ⋄

PHYX

GPT-5 40.0 ⋄ Gemini-2.5-Pro 40.0 ⋄ Gemini-3-Pro 60.0 ⋄ Qwen3-VL-235B 60.0 ⋄ Qwen2.5-VL-72B 60.0 ⋄

SYNTHETIC GEOMETRY

GPT-5 0.0 100.0 Gemini-2.5-Pro 66.7 50.0 Gemini-3-Pro 33.3 100.0 Qwen3-VL-235B 33.3 100.0 Qwen2.5-VL-72B 33.3 100.0

ANALYTIC GEOMETRY

pendix A.2.

#### 4.2 Main Results

Our main experimental results are presented in Table 1. Our main findings are as follows: (i) Frontier models demonstrate stronger multimodal reasoning overall. Among all evaluated systems, Gemini-3Pro achieves the highest overall compile and semantic accuracy on MATHVERSE and ANALYTIC GEOMETRY, while GPT-5 shows a clear advantage on the PHYX dataset. In particular, GPT-5 performs notably better in the MODERN category of PHYX, which includes quantum mechanics and relativity problems, reflecting stronger physical reasoning and grounding. (ii) Geometry reasoning remains challenging. All models exhibit significantly lower accuracy on the SYNTHETIC GEOMETRY and ANALYTIC GEOMETRY subsets, indicating persistent difficulties in bridging visual reasoning with formal reasoning. It suggests that models still face challenges in accurately understanding concrete length and angle relationships, as well as generalizing to distributions beyond those seen during training. Even advanced LLMs such as Gemini-3Pro and Gemini-2.5-Pro show large performance

gaps between image and text modalities. (iii) Advanced open-source model lag behind frontier models. The most powerful open-source model, Qwen3VL-235B, is almost unable to solve problems in the physical domain or in out-of-distribution synthetic geometry tasks.

#### 4.3 What matters in MMFORMALIZER?

In this section, we perform ablation studies to analyze how several design factors affect model performance, including (1) using only retrieved theorem names instead of full code snippets (synthesizer without code), (2) applying an explicit recursion termination condition (termination condition), (3) incorporating images during the grounding stage (grounding with image), and (4) enabling parallel sampling for each node (pass@k). As presented in Table 5, we make the following observations: (i) For the SYNTHETIC GEOMETRY setting where the data are likely never seen in the pre-training corpus we find that removing the retrieved reference code significantly improves model performance. This suggests that allowing the model to synthesize freely, rather than constraining its output space with retrieved reference code, can substantially enhance its out-of-distribution generalization ability. (ii) Not specifying a detailed termination condition leads to an excessively deep recursive tree and an overly large dependency graph, which eventually causes the synthesis process to fail. (iii) Grounding with image can significantly enhance performance in more challenging settings, such as the Modern Physics category, which includes relativity and quantum mechanics, as well as in Synthetic Geometry. (iv) Increasing the sampling number (pass@k) can improve performance on more difficult problems, indicating that test-time scaling holds great potential for MMFORMALIZATION.

#### 4.4 Semantic Checking Analysis

In this section, we employ the most powerful closed-source model, GPT-5, and the most powerful open-source model, Qwen3-VL-235B, to perform multimodal autoformalization. As shown in Table 3, we use the following five models for semantic checking: GPT-5, Gemini-2.5-pro, Gemini3-pro, Qwen3-VL-235B, and Qwen2.5-VL-72B. Finally, human checking is conducted to verify the accuracy of the semantic checking results produced by these five models. We have the following main findings: (i) In the MATHVERSE, PHYX, and ANALYTIC GEOMETRY subsets, we find that

Gemini-2.5-Pro achieves the highest accuracy. (ii) The weakest Qwen2.5-VL-72B model performs on par with the state-of-the-art Gemini-3-Pro in semantic checking tasks using code generated by the open-source Qwen3-VL-235B model, revealing the potential of weak models supervising strong models. (iii) Qwen2.5-VL-72B demonstrates strong semantic verification capabilities, slightly weaker than those of Qwen3-VL-235B.

### 5 Conclusion

In this work, we presented MMFORMALIZER, a unified framework for multimodal autoformalization that bridges perceptual understanding and formal reasoning. By recursively grounding visual and textual inputs into verifiable logical structures, our method enables interpretable formalization across mathematics and physics, including classical mechanics, relativity, quantum mechanics, and thermodynamics. Evaluations on the newly proposed PHYX-AF benchmark demonstrate the effectiveness of our approach and reveal both the promise and limitations of current large multimodal models in formal reasoning.

### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Bruno Barras, Samuel Boutin, Cristina Cornes, Judicaël Courant, Yann Coscoy, David Delahaye, Daniel de Rauglaudre, Jean-Christophe Filliâtre, Eduardo Giménez, Hugo Herbelin, and 1 others. 1999. The coq proof assistant reference manual. INRIA, version, 6(11):17–21.

Edgar Buckingham. 1914. On physically similar systems; illustrations of the use of dimensional equations. Physical review, 4(4):345.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Leonardo de Moura and Sebastian Ullrich. 2021. The lean 4 theorem prover and programming language (system description). In Automated Deduction – CADE 28, volume 12699 of Lecture Notes in Computer Science, pages 625–635. Springer.

Guoxiong Gao, Haocheng Ju, Jiedong Jiang, Zihan Qin, and Bin Dong. 2024. A semantic search engine for mathlib4. arXiv preprint arXiv:2403.13310.

Google DeepMind. 2025. Introducing gemini 3: A new era of intelligence with gemini 3. https://blog. google/products/gemini/gemini-3/. Accessed: 2025-12-29.

Zhitao He, Zongwei Lyu, Dazhong Chen, Dadi Guo, and Yi R Fung. 2025. Matp-bench: Can mllm be a good automated theorem prover for multimodal problems? arXiv preprint arXiv:2506.06034.

Thomas Hubert, Rishi Mehta, Laurent Sartran, Miklós Z Horváth, Goran Žuži´c, Eric Wieser, Aja Huang, Julian Schrittwieser, Yannick Schroecker, Hussain Masoom, and 1 others. 2025. Olympiad-level formal mathematical reasoning with reinforcement learning. Nature, pages 1–3.

Albert Q Jiang, Wenda Li, and Mateja Jamnik. 2023. Multilingual mathematical autoformalization. arXiv preprint arXiv:2311.03755.

Qi Liu, Xinhao Zheng, Xudong Lu, Qinxiang Cao, and Junchi Yan. 2025. Rethinking and improving autoformalization: towards a faithful metric and a dependency retrieval-based approach. In The Thirteenth International Conference on Learning Representations.

Jianqiao Lu, Yingjia Wan, Yinya Huang, Jing Xiong, Zhengying Liu, and Zhijiang Guo. 2024. Formalalign: Automated alignment evaluation for autoformalization. arXiv preprint arXiv:2410.10135.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. 2021. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165.

The mathlib Community. 2025. mathlib4: The mathematical library of lean 4. https://github.com/ leanprover-community/mathlib4. Version 4.x, accessed December 4, 2025.

Norman Megill and David A Wheeler. 2019. Metamath: a computer language for mathematical proofs. Lulu. com.

Logan Murphy, Kaiyu Yang, Jialiang Sun, Zhaoyu Li, Anima Anandkumar, and Xujie Si. 2024. Autoformalizing euclidean geometry. arXiv preprint arXiv:2405.17216.

Tobias Nipkow, Lawrence C. Paulson, and Markus Wenzel. 2002. Isabelle/HOL: A Proof Assistant for Higher-Order Logic, volume 2283 of Lecture Notes in Computer Science. Springer.

OpenAI. 2025. Introducing gpt-5: The most advanced

generative pre-trained transformer model. https: //openai.com/research/gpt-5. Accessed: 202512-29.

Bowen Ping, Minnan Luo, Zhuohang Dang, Chenxi Wang, and Chengyou Jia. 2025. Autogps: Automated geometry problem solving via multimodal formalization and deductive reasoning. arXiv preprint arXiv:2505.23381.

Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, and 1 others. 2025. Phyx: Does your model have the" wits" for physical reasoning? arXiv preprint arXiv:2505.15929.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Joseph Tooby-Smith. 2024. Formalization of physics index notation in lean 4. arXiv preprint arXiv:2411.07667.

Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. 2024. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482.

Haiming Wang, Mert Unsal, Xiaohan Lin, Mantas Baksys, Junqi Liu, Marco Dos Santos, Flood Sung, Marina Vinyes, Zhenzhe Ying, Zekai Zhu, and 1 others. 2025. Kimina-prover preview: Towards large formal reasoning models with reinforcement learning. arXiv preprint arXiv:2504.11354.

Haiming Wang, Huajian Xin, Chuanyang Zheng, Lin Li, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, and 1 others. 2023a. Lego-prover: Neural theorem proving with growing libraries. arXiv preprint arXiv:2310.00656.

Haiming Wang, Ye Yuan, Zhengying Liu, Jianhao Shen, Yichun Yin, Jing Xiong, Enze Xie, Han Shi, Yujun Li, Lin Li, and 1 others. 2023b. Dt-solver: Automated theorem proving with dynamic-tree sampling guided by proof-level value function. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12632–12646.

Jingxuan Wei, Caijun Jia, Qi Chen, Honghao He, Linzhuang Sun, Conghui He, Lijun Wu, Bihui Yu, and Cheng Tan. 2025. Geoint-r1: Formalizing multimodal geometric reasoning with dynamic auxiliary constructions. arXiv preprint arXiv:2508.03173.

Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. 2022. Autoformalization with large language models. Advances in neural information processing systems, 35:32353–32368.

Jing Xiong, Jianhao Shen, Ye Yuan, Haiming Wang, Yichun Yin, Zhengying Liu, Lin Li, Zhijiang Guo, Qingxing Cao, Yinya Huang, and 1 others. 2023. Trigo: Benchmarking formal mathematical proof reduction for generative language models. arXiv preprint arXiv:2310.10180.

Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, and 1 others. 2025. Mathesis: Towards formal theorem proving from natural languages. arXiv preprint arXiv:2506.07047.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. 2024a. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In arXiv.

Xiaokai Zhang, Na Zhu, Yiming He, Jia Zou, Qike Huang, Xiaoxiao Jin, Yanjun Guo, Chenyang Mao, Yang Li, Zhe Zhu, and 1 others. 2023. Formalgeo: An extensible formalized framework for olympiad geometric problem solving. arXiv preprint arXiv:2310.18021.

Xiaokai Zhang, Na Zhu, Cheng Qin, Yang Li, Zhenbing Zeng, and Tuo Leng. 2024b. Fgeo-hypergnet: Geometric problem solving integrating formal symbolic system and hypergraph neural network. arXiv preprint arXiv:2402.11461.

Zeren Zhang, Jo-Ku Cheng, Jingyang Deng, Lu Tian, Jinwen Ma, Ziran Qin, Xiaokai Zhang, Na Zhu, and Tuo Leng. 2025. Diagram formalization enhanced multi-modal geometry problem solver. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Minfeng Zhu, Zi Wang, Sizhe Ji, Zhengtong Du, Junming Ke, Xiao Deng, Zanlang Yin, Xiuqi Huang, Heyu Wang, and Wei Chen. 2025. Genesisgeo: Technical report. arXiv preprint arXiv:2509.21896.

Figure A.1: A Regular Hexagonal Prism.

### A Appendix

#### A.1 Case Study

In this section, we present several case studies that directly demonstrate how we construct dependency graphs and perform grounding.

A Regular Hexagonal Prism. We present our hexagonal prism 1example in Figure A.1. In this case study, we take a regular hexagonal prism as an illustrative example to show how our system constructs the underlying dependency graph and performs logical grounding in LEAN.

A regular hexagonal prism is a solid bounded by two congruent regular hexagons and six rectangular faces. From a geometric reasoning perspective, its structure naturally exhibits hierarchical dependencies: the prism depends on its base polygons, each base polygon depends on its edges and vertices, and the regularity constraint introduces equalities among edge lengths and face angles.

In MMFORMALIZER:, we begin by defining the primitive entities—points, line segments, and

- Figure A.2: Newton’s First Law of Motion.

- Figure A.3: Newton’s Second Law of Motion.

planes—as the lowest-level dependent types. A vertex is represented as a point inhabiting a partic-

Figure A.4: Newton’s third Law of Motion.

ular coordinate space, while an edge is defined as a dependent pair of vertices satisfying adjacency constraints. The face of the prism is then expressed as a dependent type over a collection of edges satisfying coplanarity and closure. Finally, the solid itself is defined as a higher-order structure dependent on the two hexagonal bases and the lateral faces that connect corresponding edges.

This hierarchical organization gives rise to a dependency graph in which each node represents a typed geometric entity, and edges represent type dependencies (e.g., an edge depends on its endpoints, a face depends on its edges). During grounding, abstract predicates such as “is regular” or “is parallel” are instantiated with geometric relations derived from the spatial configuration, ensuring that each dependent type is associated with concrete witnesses.

Within Lean, this structure can be encoded using Σ-types and Π-types, providing both compositionality and proof-relevance. The construction thus enforces internal consistency: for instance, the regularity condition is represented as a dependent proposition ensuring that all edges on the hexago-

[Figure 7]

- Figure A.5: Quantum Tunneling.

- Figure A.6: The Theory of Relativity.

nal base have equal length and that corresponding lateral faces are mutually congruent and perpendic-

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . .. . . . . . . . . . .

. . . . . . . . . . .. . . . . . . . . . .

Figure A.7: A failed case. In the absence of explicit termination conditions, recursive decomposition of fundamental laws leads to infinite loops and spurious primitives, producing a dependency graph of 8 layers and 684 decomposition steps, and resulting in exponential computational growth.

ular to the bases.

This example demonstrates how our framework unifies geometric representation and logical inference: the dependency graph captures the compositional structure of the prism, while grounding links symbolic entities to measurable geometric constraints, forming a coherent and verifiable reasoning chain within the dependent type theoretic setting.

Newton’s First Law of Motion. Although humans originally induced Newton’s First Law from empirical observations, we can rigorously derive it from the Hamiltonian using formal logic. We present our case study on Newton’s First Law in Figure A.2. The recursive grounding process builds dependencies among physical entities (Hamiltonian, Momentum, Time and Space), while the composition layer formalizes the relationships via Hamiltonian. This example illustrates how our system automatically constructs symbolic grounding and verifies F = dpdt under the canonical representation of Hamiltonian.

The root problem (“Force and motion relationship”) recursively expands into dependent nodes (Momentum, Hamiltonian Equations, Acceleration), yielding a structured conceptual graph. The theorem second_law_force_defined shows that in a conservative physics system, the time derivative of momentum equals the applied force, recovering Newton’s Second Law within the formal

grounding framework.

Newton’s Second Law of Motion. This figure A.3 study shows how our system automatically formalizes Newton’s Second Law from natural language input (“Newton’s Second Law”) into a machine-verifiable theorem. The process combines recursive grounding—linking core physical concepts such as time, phase space, and the Hamiltonian—to compositional construction, which builds reusable formal definitions and derives the target law.

We assume a single-particle Hamiltonian in three-dimensional space, with kinetic and potential energy but no interaction terms or couplings among particles. Under this assumption, the system instantiates Hamilton’s equations, identifies canonical relationships between momentum, velocity, and force, and reconstructs the reasoning chain that leads to the formal theorem stating that force equals mass times acceleration.

The resulting proof is fully checkable by a theorem prover, with intermediate constructs like phasespace definitions and momentum–velocity relations explicitly generated. This demonstrates that, within a clear single-particle Hamiltonian framework, the classical derivation from Hamiltonian to Newton’s Second Law can be automated and verifiably reproduced, while keeping the non-interacting physical assumption explicit and traceable.

Newton’s Third Law of Motion. Figure A.4 presents the case study for Newton’s Third Law of Motion, illustrating how our framework formalizes the principle of action and reaction from a many-body Hamiltonian. The recursive grounding process begins from the natural language statement of the law (“For every action, there is an equal and opposite reaction”) and expands into dependencies among physical concepts such as Force, Interaction, and Translational Symmetry. The compositional layer then encodes these dependencies into formal definitions and verifiable theorems in LEAN.

We start from a general N-particle Hamiltonian:

This theorem is mechanically verified by the theorem prover, connecting the abstract notion of symmetry to concrete force relations within the Hamiltonian structure.

This case study thus demonstrates how the system grounds Newton’s Third Law in the formal logic of many-body mechanics: translational symmetry in the Hamiltonian serves as the logical root of reciprocal interactions, and the dependency graph traces this relationship through explicit force definitions and interaction terms. The result is a verifiable reasoning chain that encodes the physical intuition of equal and opposite forces into a rigorous, type-theoretic formalization.

##### 2mi + i<jV(∥ri −rj∥), (18)

2 i

H(p1,...,pN;r1,...,rN) = Ni=1 p

where each term V (∥ri − rj∥) represents the potential energy of pairwise interactions that depend only on relative positions. Translational invariance of this Hamiltonian implies conservation of total momentum:

d dt i

pi =

i

Fi = 0, (19)

which leads directly to the statement that the internal forces between particles satisfy Fij = −Fji. Hence, Newton’s Third Law emerges as a necessary consequence of translational symmetry in the many-body Hamiltonian.

In our framework, this symmetry and its induced constraints are explicitly represented in the dependency graph. Nodes correspond to typed entities such as Force, Interaction, and Symmetry, and edges encode dependency relations—for instance, forces depend on the gradient of interaction potentials, while symmetry constrains their algebraic structure. During grounding, the predicate isTranslationallySymmetric ensures that any pairwise potential is invariant under global translation, automatically enforcing antisymmetry of force pairs.

At the composition level, the Lean formalization defines the potential, kinetic energy, and interaction components as dependent types parameterized by particle indices. The theorem newton_third_law_of_translational_symmetry

establishes that under translationally symmetric interaction definitions,

∀(i,j), Fij = −Fji. (20)

Quantum Tunneling. Figure A.5 illustrates our case study on the quantum tunneling phenomenon, showing how the system formalizes the reasoning process that leads from the natural language problem statement to a machine-verifiable theorem in Lean. The task begins with a particle of mass m and kinetic energy E < U0 encountering a finite potential barrier of height U0. Physically, in the region x > L, the particle’s total energy is insufficient to overcome the potential barrier, and hence the region is classically forbidden. The question asks: What is the form of the wavefunction ψ(x) in this region, given that it must remain finite as x → ∞?

MMFORMALIZER recursively grounds this question into a dependency graph of interrelated physical and mathematical entities, as shown in the upper panel. The Root Problem (time-independent Schrödinger equation in a forbidden region) expands into dependent nodes including ConstantPotentialRegion, Energy, Mass, ReducedPlanckConstant, BoundaryConditionFiniteAtInfinity, and ExponentialDecaySolution. Each node captures a typed entity in the reasoning chain—e.g., the wavefunction depends on the potential region and energy parameters, while the boundary condition constrains the admissible solutions.

At the compositional level, shown in the lower panel, the system defines a structured parameter type:

##### BarrierParams = {m,U0,E,¯h,L | E < U0},

(21) encoding the physical assumptions of the problem. The time-independent Schrödinger equation in the

forbidden region is formalized as:

##### ψ′′(x) = κ2ψ(x), (22)

where κ = 2m(U0 − E)/¯h. The boundary condition limx→∞ ψ(x) < ∞ eliminates exponentially divergent solutions, ensuring physical admissibility.

The Lean definition decayingSolution constructs the corresponding decaying exponential form:

##### ψ(x) = Ae−κ(x−L), (23)

which satisfies both the differential equation and the boundary constraint. The final theorem, forbiddenRegion_form, formally proves that under these assumptions, the only valid solution for ψ(x) in the forbidden region must be of exponential decay form.

This case study demonstrates how our system integrates symbolic reasoning with physical intuition: the recursive grounding captures conceptual relations among energy, potential, and boundary conditions, while the compositional layer generates the formal derivation of the decaying wavefunction within a theorem prover. The result is a rigorous, verifiable reconstruction of quantum tunneling behavior in the classically forbidden region, linking the physical narrative of wave attenuation to its formal logical counterpart.

Theory of Relativity. Figure A.6 presents our case study on the relativistic velocity addition problem, illustrating how our system grounds and composes physical reasoning within the framework of special relativity. The root problem originates from a natural language statement describing two cosmic-ray particles moving along the Earth’s north–south axis, one toward and one away from the geographic pole. The question asks: What is the relative speed of approach between the two particles as measured in the Earth frame?

In the Recursive Grounding process (upper panel), the system decomposes this problem into three interdependent conceptual nodes: RelativitySpeedOfLight, RelativityVelocityAddition, and MechanicsVelocity. The node SpeedOfLight encodes the universal constant c, while VelocityAddition captures the relativistic one-dimensional velocity composition law. Dependencies between nodes explicitly record that the relativistic addition formula arises from

preserving the invariance of the speed of light, linking classical and relativistic velocity constructs within the dependency graph.

At the COMPOSITION level (lower panel), the system defines these relationships formally in Lean. First, the speed of light is introduced as an axiom of type Velocity. Then, the relativistic velocity addition is defined in terms of the dimensionless ratios β = u/c and β′ = v/c, yielding the formal definition:

u + v 1 + uvc2

. (24)

u ⊕ v =

Finally, the system applies this composition law to the specific cosmic-ray problem. Given that one particle travels northward at 0.8c and the other southward at 0.6c relative to Earth, the framework computes the relative velocity of approach as:

0.8c + 0.6c 1 + 0.8 × 0.6

= 0.946c. (25)

vrel =

This computation is mechanically verified in the theorem prover via the formal statement:

theorem relativeApproach_beta :

##### ∀(uv : Velocity), u ⊕ v < c. (26)

The proof establishes that under the axiomatic assumption of the invariance of c, the relativistic velocity composition never exceeds the speed of light.

This case study demonstrates how MMFORMALIZER bridges conceptual grounding and formal verification in relativistic physics. The recursive grounding layer captures the dependencies between the invariance of light speed and the velocity addition law, while the compositional layer ensures that the resulting expressions and proofs adhere to the physical constraints of special relativity. Together, these layers yield a coherent, verifiable reasoning chain connecting natural-language physics problems to machine-checkable formal logic.

A Failed Case Study. This case illustrates a failure scenario in the multimodal autoformalization pipeline when explicit termination conditions for physical axioms are absent. As shown in Figure A.7, the system begins with the root problem—deriving the charge of a point mass suspended by an electric field at an angle. During recursive grounding, the model attempts to decompose the reasoning chain into progressively more

fundamental physical primitives, such as Mechanics Force, Electromagnetism LorentzForceLaw, and Math Vector.

However, without a termination criterion to recognize sufficient grounding or to prevent reapplication of the same axioms, the system enters a loop of redundant decomposition. Physical laws such as Newton’s second law and Lorentz force are recursively expanded into overlapping representations (Force → Mass × Acceleration → Force), producing spurious primitives such as Geometry Point or Math Integration that do not contribute to the target synthesis.

This uncontrolled recursion results in exponential graph expansion, as depicted by the red “Overdecomposition” nodes. The explosion of symbolic branches not only increases computational overhead but also dilutes semantic coherence—making the final synthesis step infeasible.

This case highlights the necessity of well-defined termination conditions and grounding heuristics to constrain recursive reasoning in multimodal physical autoformalization systems.

#### A.2 Synthetic Geometry Setup

This section describes the synthetic geometry generation and verification pipeline used in Experiment 4.1.All geometric instances in our dataset are produced entirely by this pipeline. We follow the synthetic geometry framework introduced in AlphaGeometry and its open reimplementation GenesisGeo (Zhu et al., 2025), and we do not introduce additional construction operators, deduction rules, or verification procedures beyond those prior works. The purpose of this section is to describe, in a precise and reproducible manner, how geometric configurations are constructed, how symbolic conclusions are derived, how training targets are selected, and how correctness is ensured through numerical validation. The logical decomposition of this process and the data flow between its stages are shown in Figure A.8, which serves as a structural reference for the description below. Table 4 lists the geometric construction operators used during synthetic data generation.

Each synthetic instance is generated by sampling a bounded sequence of geometric constructions, computing the symbolic deduction closure of the resulting configuration, selecting a derived statement

- as the goal, extracting a minimal set of premises sufficient to derive that goal, and finally validating the instance under a concrete numerical realization.

Figure A.8: Synthetic Geometry Generation and Verification Pipeline. Symbolic pipeline for generating synthetic geometry problem instances, from construction and deduction to dependency extraction and verification.

Table 4: Geometric construction operators used during synthetic data generation.

Category Operator Inputs Outputs

Basic point – a Basic line a, b ℓ Basic circle a, b c Basic plane a, b, c π

Intersection line_line_intersection ℓ1, ℓ2 x Intersection line_circle_intersection ℓ, c x, y Intersection circle_circle_intersection c1, c2 x, y Intersection line_plane_intersection ℓ, π x Intersection plane_plane_intersection π1, π2 ℓ

Center midpoint a, b m Center circumcenter a, b, c o Center incenter a, b, c i Center centroid a, b, c g Center orthocenter a, b, c h

Projection perpendicular_line a, ℓ ℓ′ Projection parallel_line a, ℓ ℓ′ Projection foot_of_perpendicular a, ℓ f Projection angle_bisector a, b, c ℓ

These stages correspond directly to the successive modules illustrated in Figure A.8, and each stage produces an explicit intermediate representation that is consumed by the next.

Synthetic Construction Space. Geometric configurations are generated using the declarative construction language adopted in AlphaGeometry and GenesisGeo. A configuration is defined by a finite sequence of construction operators, where each operator introduces new geometric objects such as points, lines, or circles and establishes corresponding relations including incidence, perpendicularity, and midpoint constraints. The output of this stage is a symbolic description of the configuration, which constitutes the input to the symbolic deduction stage shown in Figure A.8.

The construction process is explicitly bounded to control complexity and avoid ill-defined configura-

tions. We limit both the length of the construction sequence and the total number of geometric objects introduced, including both primitive objects and those derived through construction operators, to fixed global constants. Any partial construction that violates the preconditions of an operator, such as intersecting parallel lines or introducing prohibited collinearity, is immediately discarded and resampled. These constraints ensure that all retained constructions are finite and geometrically well defined.

In addition to validity checks at the level of construction operators, we apply lightweight filtering and canonicalization at the level of representation. Newly introduced objects follow a deterministic naming and ordering convention, and each construction operator and geometric predicate is serialized using a fixed and consistent surface form. This ensures that the symbolic output of the construction stage has a unique and stable representation before entering the deduction stage.

Symbolic Deduction Closure. Given a valid geometric construction, we compute its symbolic deduction closure using forward chaining. The input to this stage consists of the geometric facts implied directly by the applied construction operators. The output consists of the full set of geometric statements that can be derived from these facts, together with a derivation graph that records how each statement is obtained from its immediate prerequisites.

The deduction rules follow the same inference semantics as those used in AlphaGeometry and GenesisGeo. No additional rule schemas are introduced. Deduction proceeds by repeatedly applying all applicable rules until a fixed point is reached,

- at which no new statements can be derived. The process is not guided by a target goal and does not rely on heuristic pruning.

Because the construction contains a finite number of geometric objects and deduction is restricted to a fixed vocabulary of geometric predicates over those objects, the deduction process terminates in practice. The resulting derivation graph explicitly captures dependency relations between statements and provides the structural basis for selecting goals and extracting supporting premises. The role of this deduction stage within the overall pipeline is illustrated in Figure A.8.

Minimal Dependency Extraction. After computing the full deduction closure, we select a derived statement as the goal of a synthetic instance.

Candidate goals are drawn from the set of derived statements and exclude facts that arise directly from the construction operators. We further restrict selection to statements that admit an explicit derivation trace in the recorded derivation graph, ensuring that each selected goal is connected to the construction facts through a well defined dependency structure, as required by the dependency extraction stage shown in Figure A.8.

As a concrete illustration, a circumcenter construction applied to three noncollinear points introduces a point together with defining relations such as equal distances to the vertices. From these relations, the deduction closure may derive additional statements, for example segment congruences that in turn support conclusions about angle equality. Such derived relations are typical candidates for goal selection, as they are not asserted directly by the construction but arise through symbolic inference.

Unless otherwise specified, goals are sampled uniformly from this filtered set. For a selected goal, we extract a minimal set of premises by traversing the derivation graph backward from the goal to the construction facts, as illustrated in Figure A.9. This backward traversal induces a subgraph of the full derivation graph that contains exactly the statements and construction facts necessary to support the recorded derivation of the goal. Minimality is defined operationally with respect to this recorded derivation graph rather than as a globally minimal proof across all possible derivations.

When a statement admits multiple recorded derivations, the union of their prerequisite facts is included to preserve derivational sufficiency. Auxiliary constructions introduced solely to express intermediate relations are retained if and only if they appear on the dependency subgraph reachable from the selected goal. Figure A.9 shows a concrete instance of this process, illustrating how a selected goal induces a dependency subgraph and a corresponding minimal premise set extracted from a larger deduction closure.

Numerical Verification. To ensure semantic correctness, each extracted instance is subjected to numerical verification. We instantiate a concrete coordinate realization by assigning values in the real numbers to primitive geometric objects and then evaluating each construction operator sequentially to obtain coordinates for all derived objects. This realization corresponds to the final stage of

Table 5: Comparison of Termination Logic for Autoformalization in Mathematics and Physics Domains

Category Mathematics Domain Physics Domain

Phase 0: “Parameter vs. System” Test. Heuristic: Determine whether a concept represents a given number (parameter) or a system.

Check if concept_name fits any category below. If YES, output [‘concept_name’] and STOP.

General Principle

Primitives Atomic Parameters

GeometryPoint, Line, Plane, Vector, Segment, Set, List, Real.

Mass, Charge, Time, Length, Radius, Area, Angle, Position, Temperature, Moles.

GeometryRegular, GeometryRight, GeometryConvexHull, GeometryPlanar, Parallel, Coplanar.

Standard Predicates Quantum & Relativistic Parameters

PlanckConstant, SpeedOfLight, Wavelength, Frequency, QuantumNumber.

Numeric or Specific Constraints Mathematical Primitives

SideCount=6, Angle=90, VertexCountEq6.

Vector, CoordinateSystem, Axis, Direction, ReferenceFrame.

Resistance, Inductance, and Capacitance are treated as Atomic unless construction is described.

Exception Recursion depth exceeded

Figure A.9: Example Instantiation Trace for Synthetic Geometry Generation. An example showing how sampled points are expanded into a symbolic geometric configuration and yield a derived relation via Horn-style deduction.

Figure A.8, in which symbolic instances are either retained or discarded based on numerical consistency.

Predicate satisfaction is checked under a fixed numerical tolerance, which is treated as an implementation constant. Equality relations are accepted if they hold within this tolerance, while incidence, collinearity, perpendicularity, and parallelism are evaluated using standard algebraic tests under the same criterion. If numerical instantiation fails due to invalid operator evaluation, predicate violations, or configurations that are close to degeneracy and lead to numerical instability, the instance is dis-

Table 6: Termination Logic for Autoformalization in the Physics–only Domain

Category Physics Domain

Dim.Primitive = {[M], [L], [T], [Q], [Θ]} Dim.Derived = {Force, Energy, Power} Dimensionless = {Re, α, β, ε} Termination: If a concept reduces to any of the above, STOP.

Dimensional Closure

NewtonLaws, ConservationLaws, MaxwellEquations, SchrodingerEquation.

Fundamental Laws

Mechanics.Force, Mechanics.Energy, Mechanics.Power.

Base Abstract Types

Mathematical Operations Math.VectorSum, Math.ScalarSum.

carded and the construction or goal is resampled. Only instances that pass numerical verification are retained in the final dataset. Together, these constraints ensure that the generation process is finite, deterministic given a fixed random seed, and reproducible across implementations.

#### A.3 Experimental Setup

Hyperparameter. We implement MMFORMALIZER based on state-of-the-art large language models. For deterministic tasks in the Recursive Grounding and Semantic Checking phases, we set the sampling temperature to 0.1. For the Axiom Composition phase, we adopt a dynamic temperature strategy to balance precision and diversity: the temperature is set to 0.1 for greedy decoding (pass@1) and adjusted to 0.6 when generating multiple candidates (pass@3). During the retrieval process within Recursive Grounding, we fetch the top-10 results from the LeanSearch engine

for the grounding reasoner to identify the single best-matching definition.. To ensure the Recursive Termination condition remains tractable, we enforce a hard constraint on the dependency graph with a maximum size of 100 nodes and a maximum depth of 6.

Baselines. We include several representative baselines for comparison, covering both open and closed-source large models. Specifically, we evaluate Qwen2.5-VL-72B-Instruct (Bai et al., 2025), Qwen3-VL-235B-A22B-Instruct (Team, 2025), Gemini-3-Pro (Google DeepMind, 2025), and Gemini-2.5-Pro (Comanici et al., 2025), alongside our GPT-5 (OpenAI, 2025) frontier model. These baselines represent advanced multimodal reasoning systems with varying architectures and training paradigms.

Metric. We evaluate models using three complementary metrics: compile accuracy, semantic correctness, and human verification. Compile accuracy measures whether the generated code or symbolic expression can execute or parse successfully without syntax errors. Semantic correctness assesses whether the produced solution yields correct or equivalent results to the reference for both image and text modalities, independent of surface form. Human verification (human check) involves expert annotators manually reviewing the outputs for logical validity, mathematical soundness, and multimodal consistency, particularly in image–text reasoning tasks. All metrics are reported separately for image and text modalities, following the evaluation protocol used in the MATHVERSE, PHYX, SYNTHETIC GEOMETRY, and ANALYTIC GEOMETRY benchmarks.

Annotator Qualifications. The human verification stage was conducted by two Ph.D. students with advanced domain expertise. One annotator is a doctoral candidate in computer science, specializing in automated theorem proving, with multiple publications in Neural Theorem Proving. The other annotator is a theoretical physicist pursuing a Ph.D. in physics, with several first-author papers published in leading journals such as Physical Review Letters (PRL). Both annotators possess extensive experience in evaluating mathematical reasoning systems and multimodal problem-solving tasks, ensuring the reliability and rigor of the human verification results.

Ablation Study. As shown in Table 2, we extracted subsets from the full dataset as test sets for our ablation study. Specifically, 4 samples were selected from MathVerse Plane Geometry, 6 from MATHVERSE Solid Geometry, 2 from PHYX Modern Physics, 5 from PHYX Mechanics, 2 from PHYX Electromagnetism, 1 from PHYX Thermodynamics, 3 from SYNTHETIC GEOMETRY Plane Geometry, 3 from SYNTHETIC GEOMETRY Solid Geometry, 3 from ANALYTIC GEOMETRY Plane Geometry, and 1 from ANALYTIC GEOMETRY Solid Geometry. The numbers above indicate the number of problems used from each subset.

#### A.4 Termination Condition

We provide a detailed explanation of the recursive termination condition of MMFORMALIZER in this section, which directly determines the decomposition depth of our dependency graph. We present the respective recursive termination conditions for the physics and mathematics domains in Table 5. We present the recursive termination conditions adopted exclusively in the physics domain in Table 6.

