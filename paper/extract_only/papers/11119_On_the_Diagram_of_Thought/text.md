## arXiv:2409.10038v6[cs.CL]14May2026

# On the Diagram of Thought

Yifan Zhang1 Yang Yuan1,2 Andrew Chi-Chih Yao1,2† 1IIIS, Tsinghua University 2Shanghai Qi Zhi Institute

Abstract

Large Language Models (LLMs) excel at many tasks but often falter on complex problems that require structured, multi-step reasoning. We introduce the Diagram of Thought (DoT), a framework that enables a single LLM to build and navigate a mental map of its reasoning. Instead of thinking in a straight line, the model constructs a dynamic diagram of ideas, where it can propose different lines of thought, critique its own steps, and synthesize validated insights into a final conclusion. This process is controller-light: it does not require an external search algorithm or planner, but it does use a deterministic online validator for grammar-constrained typed traces, register constraints, and optional solver checks. To clarify the reliability target of this process, we ground DoT in a mathematical framework from category theory. We interpret accepted typed reasoning records as diagrams in a slice topos and model synthesis of the selected proposer subdiagram as a finite limit. In the predicate fragment, this same object is equivalently a variance-reversed colimit in the opposite information order. The resulting formalism gives an auditable, step-by-step trace of the LLM’s typed reasoning and separates semantic guarantees for the typed subtrace from unconstrained natural-language text and uncertified operational edges.

Project Page: https://github.com/diagram-of-thought/diagram-of-thought

### 1 Introduction

Large Language Models (LLMs) (Brown et al., 2020; Touvron et al., 2023) have exhibited remarkable proficiency across a spectrum of natural language tasks. However, achieving robust performance on complex reasoning problems that necessitate structured exploration, iterative refinement, backtracking, and self-correction remains a formidable challenge (Huang and Chang, 2022). Initial prompting strategies, such as Chain-of-Thought (CoT) (Wei et al., 2022), encourage step-by-step reasoning by eliciting intermediate steps. While beneficial, the inherent linearity of CoT struggles to capture the dynamic, non-sequential nature of sophisticated problem-solving, which often involves generating parallel hypotheses, critical evaluation, and synthesis, processes ill-suited to a strictly linear progression.

Recognizing these limitations, subsequent research has explored more complex reasoning structures. Frameworks like Tree-of-Thought (ToT) (Yao et al., 2023) utilize tree structures to manage multiple reasoning paths, while Graph-of-Thought (GoT) (Besta et al., 2024) generalizes this to arbitrary graphs. Other approaches, such as Cumulative Reasoning (CR) (Zhang et al., 2023), leverage multi-agent paradigms with specialized roles. Despite their advancements, these methods frequently require external controllers or multi-component pipelines. In contrast, DoT retains a single-model setting while making the operational constraints explicit and checkable via grammar-constrained masking with register constraints and typed validation.

In this paper, we introduce the Diagram of Thought (DoT) framework, which internalizes complex, iterative reasoning within a single auto-regressive language model, guided by learned role tokens and a typed serialization enforced by an online validator with finite control over record kinds and register/solver checks. DoT conceptualizes the reasoning process as the construction of a Directed Acyclic Graph (DAG), as in Figure 1. Nodes represent propositions, critiques, refinements, or verified statements, while edges capture logical or procedural dependencies. The model explores alternatives, responds to critiques, and consolidates validated content toward a conclusion.

A cornerstone of the DoT framework is its operationalization through role-specific tokens (e.g., <proposer>, <critic>, <summarizer>). By learning to predict and condition on these tokens, the model transitions between cognitive roles, generating hypotheses, evaluating them, refining based on feedback, and synthesizing results, within standard auto-regressive decoding. Tokens are training/decoding aids; the recorded @node role is the validator’s source of truth. This unifies the entire reasoning process inside one model while keeping the interface auditable. All formal guarantees below are scoped to the typed subtrace accepted by V ; untyped/free text is semantically inert for Φ and may diverge in natural language from the typed content.

Crucially, to ensure logical consistency and provide a principled aggregation mechanism, we establish a mathematical foundation for DoT using Topos Theory (MacLane and Moerdijk, 2012; Johnstone, 2002; Lambek and Scott, 1988). This framework allows us to model reasoning steps as formal objects and their synthesis as a universal construction. Specifically, we fix a presheaf topos E = SetCop and a semantic object S ∈ E. A typed proposer node is interpreted as an object of the slice (E/S) (a map X → S); in the predicate instantiation used for most intuitions, this map is a monomorphism P → S and thus corresponds to a subobject. Synthesis by the <summarizer> role is modeled as a finite limit of the selected proposer diagram in the slice. In the predicate case this is a meet in Sub(S), equivalently the corresponding colimit after reversing variance into the opposite information order Pred(S) = Sub(S)op. Operational dependency edges contribute to provenance; they become semantic arrows only when accompanied by accepted typed certificates. Reflection along a Lawvere–Tierney topology captures validation; for a finite family of already validated (c-closed) predicates, the outer closure is redundant because the associated closure is a nucleus. For general slice objects that are not monomorphisms, validation is expressed by applying the induced slice sheafification functor; the phrase “c-closed” is reserved for subobjects.

Our main contributions are therefore:

- 1. We propose the Diagram of Thought (DoT), a single-model framework for DAG-structured proposals, critiques, and summaries, with a deterministic record serialization whose regular core is supplemented by register and solver checks, together with a finite-control online validator that enables auditable extraction.
- 2. We develop a categorical semantics for DoT in the slice topos (E/S) and show that synthesis of a selected finite validated proposer diagram is a finite limit in the slice. In the predicate/mono fragment this is the meet of the selected validated subobjects, equivalently a variance-reversed information-order colimit; in the general-arrow fragment, left-exact slice sheafification reflects this finite limit into the validated/sheaf slice.
- 3. We define a total and deterministic extraction map from LLM-generated traces to formal diagrams, enabled by a well-formedness discipline (BNF grammar, operational rules) and constrained decoding, ensuring that typed reasoning traces are auditable and structurally sound.

Problem Statement

Valid Proposition

Invalidated Proposition

Proposition P1

Critique C1

Critique

Summarization

Refine

Proposition P1’

Critique C2

Verified

Proposition P1’ (Verified)

Proposition P3

Critique C3

Verified

Proposition P3 (Verified)

Summarization

- Figure 1 High-level illustration of the Diagram of Thought (DoT) process. Edges encode dependencies from earlier to later nodes: a critic depends on the proposition it evaluates (proposer → critic), and the summarizer depends on validated propositions. A single LLM generates the DAG representing proposing (circles), critiquing (rectangles), repairing/verification, and synthesis (ellipse).

### 2 Related Work

The pursuit of robust reasoning within Large Language Models (LLMs) has driven considerable research beyond basic input-output functionality. Initial breakthroughs like Chain-of-Thought (CoT) prompting (Wei et al., 2022; Kojima et al., 2022) demonstrated that eliciting intermediate reasoning steps significantly improves performance on complex tasks. CoT effectively linearizes reasoning, enhancing transparency but suffering from rigidity; its sequential nature hinders exploration of alternatives or recovery from early errors without restarting. Methods like Self-consistency (Wang et al., 2022) mitigate this by sampling multiple reasoning paths and selecting the majority answer, implicitly acknowledging path diversity but lacking explicit refinement mechanisms.

Recognizing the constraints of linearity, subsequent work explored more complex structures. Tree-of-Thought (ToT) (Yao et al., 2023) introduced tree structures where nodes represent partial

solutions and edges denote reasoning operators. ToT utilizes search algorithms (e.g., BFS, DFS) guided by heuristic evaluations (often LLM-based) to explore possibilities, enabling systematic search and backtracking. However, ToT generally necessitates an external controller for search management and pruning. Graph-of-Thought (GoT) (Besta et al., 2024) extends this to arbitrary graphs, allowing for more intricate dependency modeling, such as merging reasoning paths, but often requiring more sophisticated external graph management systems.

Collaborative and iterative refinement approaches offer another perspective. Cumulative Reasoning (CR) (Zhang et al., 2023) employs multiple LLM instances (or prompts) assigned specific roles (e.g., proposer, verifier), interacting iteratively. While modular, this introduces coordination overhead. Self-Refine (Madaan et al., 2023) focuses on iterative improvement where an LLM critiques and refines its own output, though typically applied to the entire output rather than intermediate reasoning steps within a structured process.

From a foundational perspective, Yuan (2023) uses category theory to analyze the inherent capabilities and limitations of LLMs. This work proves that prompt-based tuning is restricted to “representable” tasks within the pretext task category, potentially explaining the limitations of simpler methods like CoT. Conversely, the theory suggests fine-tuning offers broader potential, theoretically enabling a sufficiently powerful model to solve any task within that category given adequate resources.

Diagram of Thought (DoT) builds upon these diverse approaches while offering key distinctions. Like ToT and GoT, DoT utilizes non-linear structures (DAGs) for reasoning. However, it distinctively internalizes the graph construction and navigation within a single auto-regressive model via role tokens, minimizing external control dependencies. This contrasts with the external orchestration often required by ToT and GoT. DoT employs explicit cognitive roles (propose, critique, summarize), similar to CR, but integrates them seamlessly within one model through conditional generation, avoiding multi-agent coordination complexities. The use of rich natural language critiques potentially offers more nuanced feedback than the simple heuristic scores sometimes used in ToT. Importantly, by grounding the typed reasoning trace in Topos Theory, DoT aims for a level of formal rigor and consistency criteria that distinguishes it from purely heuristic methods, resonating with the structural analysis provided by works like Yuan (2023). DoT thus presents a unified, self-contained, interpretable, and formally grounded approach to advance complex reasoning in LLMs.

### 3 The Diagram of Thought Framework

In this section, we formally define the Diagram of Thought (DoT) framework. DoT operationalizes complex, iterative reasoning as the dynamic construction and traversal of a Directed Acyclic Graph (DAG) G = (V,E) entirely within a single auto-regressive language model, LM. This internal graph structure allows the model to manage parallel lines of thought, critique intermediate steps, repair ideas based on feedback, and synthesize validated conclusions.

- Definition 3.1 (DoT Graph Components). The DoT graph G = (V,E) is composed of:

• Nodes v ∈ V : Each node represents a semantic unit or reasoning step. Every node v is associated with:

- – A specific role role(v) ∈ R = {Problem, Proposer, Critic, Summarizer}.
- – Textual content content(v), generated by the LLM LM while assuming the role role(v).

[Figure 1]

- Figure 2 Illustrative example: Applying DoT reasoning steps to compare numerical values. Critiques might identify incorrect digit comparisons.

– Optionally, an internal state state(v) ∈ {active, validated, invalidated, initial}. For example, a ‘Proposer’ node might start as ‘active’, become ‘validated’ after a positive critique, or ‘invalidated’ after a negative critique.

• Edges (u,v) ∈ E: A directed edge from node u to node v encodes that v depends on u. We standardize that, when emitting node v (with fresh ID v), all edges in its block have dst= v and src< v. This covers:

- – Logical dependency (a new proposition depends on earlier premises). A node may depend on multiple sources, indicated by multiple edge records in its emission block.
- – Procedural dependency (a critic depends on the proposition it evaluates: proposer → critic).
- – Contextual dependency (a summarizer depends on the validated nodes it uses). The operational provenance graph is acyclic by construction since every operational edge targets the current node and all sources are previously emitted nodes. This acyclicity statement concerns the provenance DAG only: the extracted semantic index category may contain arrows opposite to refinement provenance, isomorphisms, or cycles induced by certified equivalence records.

The construction of this graph is implicitly managed by the LLM’s standard auto-regressive generation process, strategically guided by special role tokens and constrained by the validator whenever typed serialization is enabled.

[Figure 2]

- Figure 3 Illustrative example: A character-counting task where intermediate steps (identifying ’r’s) and potential critiques (missed counts, double counts) could form a DoT graph.

#### 3.1 Roles, Generation, and Iterative Reasoning

A core mechanism of DoT involves augmenting the LLM’s vocabulary V with a distinct set of role-specific tokens:

Troles = {<problem>,<proposer>,<critic>,<summarizer>}.

Let V′ = V ∪ Troles be the augmented vocabulary. The LLM LM operates by predicting the next token wt ∈ V′ based on the preceding sequence (history) Ht−1 = w1,...,wt−1:

Pr

(wt | Ht−1),

LMθ

where θ denotes the model parameters. The generated sequence HT = w1,...,wT represents a serialized traversal and construction of the DoT graph G.

The role tokens function as control signals, prompting the LLM to adopt a specific cognitive function for the subsequent text generation, thereby determining the role and content of the next node(s) in the graph:

- • <problem>: Typically precedes the initial problem statement P. This establishes the root node vstart ∈ V with role(vstart) = Problem and content(vstart) = P. Its state is state(vstart) = initial.
- • <proposer>: Signals the LLM to generate a hypothesis, intermediate reasoning step, or potential

solution fragment Pi. This creates a new node vPi with role(vPi) = Proposer and content(vPi) = Pi. All dependencies are declared explicitly via serialized @edge records defined in Section 3.2. The new node typically starts with state(vPi) = active.

- • <critic>: Instructs the LLM to evaluate one or more preceding ‘Proposer’ nodes. The LLM generates a critique Cj, assessing validity, identifying flaws, or suggesting improvements. This

creates a node vCj with role(vCj) = Critic and explicit dependency edges from the propositions to the critic. We adopt a monotone validation discipline:

- – If the critique validates a proposition, its state transitions to ‘validated’. This state is absorbing under the single-assignment discipline.
- – If the critique identifies flaws, its state transitions to ‘invalidated’. This renders the proposition ineligible for summarization. The reasoning path is expected to continue from the critique node to generate a repair or alternative proposition.

- • <summarizer>: Prompts the LLM to synthesize a final answer or consolidated conclusion. The model is trained to condition its summary generation on validated propositions explicitly selected by @edge kind=use records into the summarizer node, which are accessible through the serialized history Ht−1. It performs a conceptual aggregation respecting the declared dependencies and generates the summary text S. This creates a final node vS with role(vS) = Summarizer and explicit @edge records from the validated propositions that contributed to the summary. A fullrun summary may select all validated propositions by convention. Generation often terminates after this step.

The LLM learns to predict appropriate role token transitions based on the entire preceding history Ht−1, effectively learning to navigate and structure the reasoning process. For instance, after generating a proposition via <proposer>, the model learns it is often appropriate to predict <critic>. Following a critical critique (<critic> leading to state ‘invalidated’), the model might predict <proposer> again to generate a repair, or explore an alternative branch.

The DoT reasoning process unfolds as the LLM generates a serialized representation of the DAG G:

- 1. Initialization: The process begins with the problem statement, formatted using the typed serialization from Section 3.2 (e.g., @node id=1 role=problem ...). This defines the root node v1.
- 2. Proposal: The LLM predicts <proposer> and generates the text for a proposition P2, along with its node definition (@node id=2 role=proposer) and an edge from a prior node (@edge src=1 dst=2 kind=use). This adds node v2 (state: active) to G.
- 3. Critique: The LLM predicts <critic>, generates a critique C3 for P2, and emits the corresponding node and edge records (@node id=3 role=critic, @edge src=2 dst=3 kind=critique). It also emits a status record (@status target=2 mark=validated|invalidated) that updates the state of v2.
- 4. Continuation (Branching/Repair/Exploration): Based on the history (including the state of v2), the LLM predicts the next role token:

- • If v2 was validated: The LLM might predict <proposer> to generate a new proposition P4 building upon v2 (adding node v4 and an @edge from v2).
- • If v2 was invalidated by C3: The LLM might predict <proposer> to generate a repaired proposition P4 that addresses the critique, with a procedural edge from v3 (@edge src=3 dst=4 kind=refine). Such a repair edge records provenance; it induces a semantic entailment arrow only if accompanied by a typed certificate such as @entails. Alternatively, it might backtrack to generate an alternative proposition P5 stemming from the root node v1.

- 5. Iteration: Steps 2–4 repeat, progressively extending the operational DAG. The requirement that operational edge destinations must have higher IDs than their sources syntactically enforces acyclicity of provenance.
- 6. Summarization: Eventually, the LLM predicts <summarizer>. It is trained to generate a summary by synthesizing information from selected validated nodes, adding a summary node,

and explicit @edge records from the nodes it used. This process yields a structured, interpretable trace of reasoning, captured within a single, selfcontained generated sequence.

#### 3.2 Typed Serialization, Validation, and Extraction

To ground the reasoning process and ensure auditability, we introduce a disciplined, typed serialization format. Natural language content is interleaved with structured records (prefixed with ‘@’) that define the DAG’s nodes, edges, and states. For example, @node id=3 role=critic creates a critic node, and @edge src=2 dst=3 kind=critique establishes its dependency on a previous proposition. Node IDs are strictly increasing, guaranteeing acyclicity of the operational provenance DAG by construction.

During inference, a lightweight, controller-light validator enforces this structure. It uses grammarand register-constrained masking to ensure the LLM only generates syntactically valid records (e.g., an edge’s source must be a previously defined node) and, where typed blocks are present, calls the selected solver checks for semantic certificates. This process is deterministic and avoids external search algorithms or planners. A deterministic extraction map, Φ, converts any well-formed typed trace into a formal syntactic diagram, and then into a semantic diagram after a typed interpretation is fixed, enabling the categorical semantics described in Section 4. The full grammar, operational semantics, and validation rules are detailed in Appendix B.

#### 3.3 Training and Controller-Light Inference

Training: The DoT capability is instilled in the LLM LM through fine-tuning on datasets formatted according to the DoT structure. Such data consists of sequences H = w1,...,wT containing appropriately interleaved role tokens (Troles) and natural language text segments, representing valid, coherent reasoning DAGs. Potential data sources include:

- • Curated examples derived from human step-by-step reasoning traces, augmented with role tokens and serialized records.
- • Synthetically generated examples from structured problem-solving processes (e.g., program execution traces, mathematical proofs), formatted using the typed serialization from Section 3.2.
- • Bootstrapped data generated by an initial version of a DoT model, potentially filtered or refined based on correctness or coherence metrics.

The training objective is the standard auto-regressive language modeling loss (e.g., cross-entropy) applied over the entire sequence, including role tokens and content tokens:

|H|

1 |H|

L(θ) = −

log Pr

(wt | w1,...,wt−1).

LMθ

t=1

This objective trains the model LM to simultaneously learn the reasoning patterns associated with each role (proposing, critiquing, summarizing) and the appropriate transitions between these roles based on the context, thereby internalizing the ability to construct and navigate the DoT graph structure.

Inference: To solve a new problem P using DoT, inference proceeds as follows:

- 1. Initialize the generation history H with the serialized problem statement, e.g., @node id=1 role=problem ....

- 2. Perform auto-regressive generation using the trained LLM LM. At each step t, sample or select

the next token wt ∼ LM(Ht−1) using a chosen decoding strategy (e.g., greedy decoding, nucleus sampling) and the well-formedness constraints from Section 3.2.

- 3. Append the generated token wt to the history: Ht = Ht−1 ⊕ wt.
- 4. Repeat steps 2 and 3 until a termination condition is met. Common conditions include:

- • Generation of the <summarizer> token and its subsequent content, followed by a special end token.
- • Reaching a predefined maximum sequence length or generation budget.
- • Generation of a specific end-of-sequence token.

The final output is typically the textual content associated with the <summarizer> node, although the complete generated sequence H provides the full reasoning trace (the serialized DoT graph) for interpretability. Notably, this entire process is self-contained within the single LLM LM; no external graph management system or search/planning controller is required during inference, beyond a deterministic syntax validator and, when typed blocks are present, a decidable entailment check for the chosen typed fragment.

### 4 Topos-Theoretic Formalization of DoT

Order convention. We write Sub(S) for the Heyting algebra of subobjects of S, ordered by inclusion (P ≤ Q iff P → Q). Under this order, more informative / more constrained propositions correspond to smaller subobjects. This order-theoretic presentation applies to the predicate/mono instantiation in which proposer nodes are interpreted as monomorphisms into S. To align “more information” with “larger” elements, we work in the opposite poset

Pred(S) := Sub(S)op, so that P ⪯ Q in Pred(S) iff Q ≤ P in Sub(S).

For a finite discrete family, the colimit in Pred(S) is its join; translated back to Sub(S) this is a meet (intersection):

colimPred(S){Pv}v∈V = Pred(S){Pv}v∈V = v∈V Pv (computed in Sub(S)). The empty meet is S, the terminal object of the slice (E/S). For an actual semantic diagram

- D : J → Sub(S), the variance is explicit: the corresponding information-order diagram is Dop : J op → Pred(S) = Sub(S)op, and

colimPred(S) Dop ∼= limSub(S) D.

Equivalently (and this is the formulation we use in the general-arrow setting), conjunction-like synthesis is a finite limit in the slice (E/S): for subobjects, finite limits are pullbacks and compute intersections. We keep the posetal fragment explicit (Assumption (A4)); the general-arrow case is handled via slice sheafification together with finite limits (Cor. 4.6).

General slice instantiation. When proposer nodes are interpreted as general objects X → S

in (E/S) (not necessarily monos), the poset Sub(S) no longer captures the whole semantics. Synthesis is then the finite limit of the extracted selected diagram in (E/S), not merely a meet

of its objects; validation reflects this finite limit into the sheaf slice via the induced left-exact functor a/Sj : (E/S) → (Shj(E)/ajS). The predicate/mono equations are recovered by restricting to monomorphisms and comparing the sheafified result back over S by pullback along the unit ηS : S → ajS.

While the operational description in Section 3 details the DoT mechanism, establishing its logical soundness and robustness requires a deeper, formal framework. We leverage Topos Theory (MacLane and Moerdijk, 2012; Johnstone, 2002; Lambek and Scott, 1988), which provides a setting with finite limits, exponentials, and a subobject classifier to interpret intuitionistic logic. This makes it suitable for formalizing the dynamic, evidence-aggregating, and context-dependent reasoning inherent in the DoT process.

An elementary topos E is a category that encapsulates key properties needed for modeling logical systems and computation:

- 1. Finite Limits: E has a terminal object 1, binary products A × B, and pullbacks. This allows for combining and constraining information.
- 2. Cartesian Closure: For any objects A,B ∈ E, there exists an exponential object BA, representing the internal collection of morphisms A → B. This enables modeling functions, predicates, and higher-order logic.
- 3. Subobject Classifier Ω: There exists an object Ω and a truth arrow ⊤ : 1 → Ω such that every

monomorphism m : P → A corresponds uniquely to a characteristic morphism χm : A → Ω. Ω internalizes the logic of the topos, which is generally a Heyting algebra, supporting intuitionistic reasoning.

Crucially for DoT, presheaf topoi E = SetCop are complete and cocomplete. For the synthesis step we focus on, the relevant universal construction is a finite limit (conjunction of constraints) in the slice (E/S); dually, this is a variance-reversed colimit in the information order Pred(S) = Sub(S)op. Since slices of a presheaf topos are again topoi, the slice (E/S) has the finite limits we require (and also all small colimits, should one wish to model disjunctive/quotient-style aggregation).

In what follows, we fix a presheaf topos E = SetCop. This is the category of functors Cop → Set for a small category C. We also fix a designated semantic object S ∈ E representing the universe of discourse. Our formalization takes place within the slice category (E/S), where predicate-like propositions are subobjects of S and general typed propositions are objects over S. For concreteness, one may take C to index problem contexts and S(c) to be the set of admissible semantic states at context c. For the QF-LIA instantiation used in examples, we take C to be a finite discrete category (a single object in simple cases) and interpret terms componentwise so that interpretation remains decidable.

- Definition 4.1 (Categorical Semantics of DoT Components). Within the fixed presheaf topos

- E = SetCop and slice (E/S), fix a Lawvere–Tierney topology j : Ω → Ω with associated universal closure operator c = (cA)A∈E on subobjects (extensive, idempotent, monotone, pullback-stable). When discussing only subobjects of S, we write c for the component cS : Sub(S) → Sub(S).

- • Semantic Space (S): A base object S ∈ E representing the universe of discourse or the space of all possible solutions and intermediate states.
- • Propositions (P): A proposer node is interpreted as an object p : P → S of the slice category (E/S). In the predicate/mono instantiation, p is a monomorphism P → S, and we freely identify it with the corresponding subobject (a “subset” of S).
- • Entailment vs. dependency variance: In the predicate instantiation, entailment P ⇒ Q cor-

- responds to inclusion P ≤ Q in Sub(S) (equivalently, the unique morphism P → Q over S). Operational dependency edges are not, by themselves, semantic entailments. The extracted index category JG (Def. 4.3) generates semantic arrows only from certified strengthening/refinement records and certified @entails records. Thus a certified strengthening of proposer u by proposer v induces an arrow v → u in JG, interpreted as a map DG(v) → DG(u) over S (predicate case: DG(v) ≤ DG(u)), while @entails src=i dst=k induces an arrow i → k interpreted as DG(i) → DG(k). In particular, an operational record @edge src=u dst=v kind=refine points from the old node to the new node as provenance, but if it carries a typed strengthening certificate, the induced semantic arrow is v → u. Ordinary use, critique, and uncertified repair-style refine edges record provenance/control flow only.
- • Critiques as judgements (typed): A <critic> node is typed evidence about one or more propositions. Semantically, an accepted critique record contributes (i) a validation mark for its target proposer, and optionally (ii) typed arrows, isomorphisms, or path-equality constraints between proposer objects over S (e.g., certified @entails/@eq records), which become generators or relations in the extracted diagram. In the predicate/mono fragment, certified entailment arrows are inclusions between subobjects; in the general slice fragment, they may be arbitrary morphisms over S. Critic nodes themselves are witnesses for these records rather than objects of the semantic synthesis diagram.
- • Validation as a nucleus/reflection: On subobjects, validation is modeled by the universal closure c : Sub(S) → Sub(S) induced by j; in particular c is extensive, monotone, idempotent, pullback-stable, and satisfies c(P ∧ Q) = c(P) ∧ c(Q) for finite meets (i.e., c is a nucleus). In the predicate/mono fragment, a proposition is semantically validated iff c(P) = P (i.e., it is c-closed). For a general slice object X → S, validation is represented by applying the induced left-exact sheafification functor a/Sj ; we do not call arbitrary objects c-closed unless they are subobjects.
- • Strengthening vs. repair: A strengthening step produces a new proposer object p′ : P′ → S equipped with a certified morphism P′ → P over S (interpreted as “P′ entails/strengthens P”). In the predicate/mono fragment, this is precisely an inclusion P′ → P in Sub(S). A repair step after an invalidation need not strengthen the rejected proposition; it contributes to the semantic diagram only through separately certified entailment or equivalence records.
- • Selected Validated Diagram: Only validated proposer nodes selected by the summarizer enter the synthesis computation; critique nodes provide certified morphisms and equivalence constraints between proposer nodes.

Assumption 4.2 (Fixed, pullback-stable validation modality). There exists a Lawvere–Tierney topology j : Ω → Ω on E whose induced universal closure operator c = (cA)A∈E models validation. Each component cA : Sub(A) → Sub(A) is extensive, monotone, idempotent, and pullback-stable. All results in this section are relative to this fixed j. Moreover, since c is induced by a Lawvere– Tierney topology, each cA is a nucleus on Sub(A): it preserves finite meets. When we apply c to propositions over S, we mean the component cS. We work throughout in the presheaf topos setting, so the finite limits we use in synthesis exist in (E/S); the induced functor a/Sj is left exact and therefore preserves these finite limits.

#### 4.1 Semantic Target and Normative Conditions

The following assumptions connect the practical LLM behavior with our formal model. They are idealized, normative conditions that define the target semantics for a well-behaved DoT agent. The

operational mechanisms are designed to make LLM traces more likely to satisfy these conditions upon interpretation.

- (A1) Typed proposer nodes admit interpretations as objects over S (as subobjects in the predicate/mono fragment); certified critique content interprets as arrows/predicates that constrain these objects. Certified strengthening/refinement and @entails records correspond to morphisms over S.1
- (A2) In the predicate/mono fragment, critique-driven validation corresponds to the Lawvere–Tierney closure c on Sub(S); validated predicate nodes are exactly the c-closed subobjects. In the general-arrow fragment, validated synthesis is computed after applying the induced slice sheafification functor.
- (A3) Equivalences and path equalities identified by validated critiques are respected. Formally, we quotient the generated index category by the smallest congruence induced by certified equality records (see Def. 4.3); in the predicate/posetal syntax, @eq src=i dst=k abbreviates mutual entailment/equality of proposer subobjects, while in the general slice syntax it requires a certified isomorphism over S.
- (A4) For our main result, we consider the fragment in which every selected validated proposer is interpreted as a monomorphism P → S and every selected validated arrow between proposers is (hence) an inclusion in Sub(S), so the selected diagram lands in the posetal category Sub(S) ⊆ (E/S).

#### 4.2 The Reasoning Diagram and its Synthesis via Finite Limit

The DoT-DAG G = (V,E) specifies the operational trace. The categorical diagram used for synthesis is extracted from the certified semantic records inside that trace.

Definition 4.3 (DoT Index Category JG). Given a well-formed typed trace with DoT DAG G = (V,E), let Vprop ⊆ V be the subset of proposer nodes. We distinguish the operational DAG from the semantic diagram: ordinary dependency edges do not automatically generate semantic morphisms.

Let Asem(G) be the directed multigraph on Vprop with the following certified semantic generators:

- • an arrow v → u whenever the trace contains an accepted typed certificate that proposer v strengthens proposer u; for a certified operational record @edge src=u dst=v kind=refine, the semantic arrow is therefore opposite to the operational dependency edge;
- • an arrow i → k for each accepted record @entails src=i dst=k between proposer nodes;
- • in the general slice fragment, inverse arrows for each accepted @eq record certified as an isomorphism over S.

Pure use, critique, and uncertified repair edges are excluded from Asem(G). The semantic index category is

JG := Free(Asem(G))/≡,

where ≡ is the smallest arrow congruence generated by certified path-equality records and, for certified isomorphism records, by the inverse equations. In the predicate/posetal fragment, @eq src=i dst=k abbreviates mutual entailment and hence equality of subobjects; equivalently, it

1This is a strong assumption; our framework is normative: it specifies the target semantics an ideal DoT agent should realize, enabled by the typed serialization in Section 3.2.

identifies the corresponding objects in the thin category. Richer path-equality syntax can be added by extending the same congruence.

For all validated content, Jvalid denotes the finite presentation generated by validated proposer objects and certified semantic arrows whose source and target are validated. Paths through invalidated proposer nodes are not used unless the trace separately certifies an arrow between validated endpoints.

For a particular summarizer node s, let VΣ(s) = {v ∈ Vprop : v is validated and @edge src=v dst=s kind=use is accepted}.

The synthesis presentation JΣ(s) is the subpresentation generated by VΣ(s) and the certified semantic arrows and relations whose endpoints lie in VΣ(s). If no summarizer node is specified and one wants a global full-run summary, we set VΣ = Vvalid and JΣ = Jvalid by convention.

- Theorem 4.4 (DoT Process as Diagram Construction). A DoT reasoning process generating a well-formed typed DAG G = (V,E), together with a fixed semantic interpretation of typed propositions and certified arrows, defines a functor (a diagram) DG : JG → (E/S) with:

- • Each typed proposer node v mapped to an object DG(v) → S in the slice (a subobject DG(v) → S in the predicate/mono fragment);
- • Accepted critic records supply certified arrows, isomorphisms, or path equalities between proposer objects; critic nodes are witnesses for these records rather than objects of DG;
- • Each arrow v → u in JG is mapped to a certified morphism over S, witnessing entailment or strengthening coherence (posetal case: an inclusion DG(v) → DG(u)).

Functoriality ensures that composite certified dependencies (via paths modulo ≡) are respected in the slice.

Proof Sketch. By Assumption (A1), each typed proposer node v admits an interpretation as an object DG(v) → S of the slice (E/S), and as a subobject in the predicate/mono fragment. The generators of JG are certified strengthening/refinement arrows, certified @entails arrows, and certified isomorphism/path-equality records. The validator requires each such generator to have a typed semantic witness, so Assumption (A1) assigns it a morphism over S. We define DG on composite arrows by composition in (E/S). Certified equality records generate the congruence used in JG, and Assumption (A3) ensures that equivalent syntactic paths induce equal semantic composites. Hence DG is well-defined and preserves identities and composition.

| |
|---|

Synthesis by information-colimit (slice-limit) and reflection. The <summarizer> aggregates selected validated content. We distinguish an inclusion/posetal setting (our main focus) and a general-arrow

setting. In the latter, sheafification induces a base-change: a/Sj : (E/S) → (Shj(E)/ajS), and summaries are read over ajS via the unit S → ajS.

- Theorem 4.5 (Summarization as finite meet-plus-closure in the reflective subposet). Assume every proposer selected for synthesis is interpreted as a c-closed subobject of S and every certified semantic

arrow between selected proposers is an inclusion, so the selected diagram DΣ = DG|JΣ lands in the thin category Sub(S). Let VΣ be the finite set of validated proposer nodes selected by the summarizer, with the convention that the empty meet is S. The finite-limit summary is

SummaryΣ = limSub(S) DΣ =

DG(v).

v∈VΣ

Equivalently, this is the colimit of the variance-reversed diagram DΣop : JΣop → Pred(S) = Sub(S)op. If one writes the reflected form

DG(v) ,

c

v∈VΣ

then the outer c is redundant because c is a nucleus and all inputs are already c-closed.

Proof Sketch. In the predicate fragment, selected validated proposer nodes are subobjects of S. Since Sub(S) is thin, the limit of a finite diagram is the greatest lower bound of its objects, namely their meet. Equivalently, this meet is the finite join of the corresponding opposite diagram in the information order Pred(S) = Sub(S)op. Since each selected proposition is c-closed and c preserves finite meets, the meet is again c-closed; writing the result as c( v DG(v)) emphasizes compatibility with the reflected construction.

| |
|---|

Corollary 4.6 (General case via slice sheafification). For a general selected diagram DΣ : JΣ → (E/S) (not necessarily posetal), presented by a finite generating graph and a finite set of certified coherence/path-equality relations already satisfied by DΣ, the synthesized summary in the validated/sheaf slice is

SummaryΣ ∼= limShj(E)/ajS a/Sj ◦ DΣ ∼= a/Sj limE/S DΣ ,

where a/Sj : (E/S) → (Shj(E)/ajS) is the left-exact functor induced by sheafification, sending X → S to ajX → ajS. If one wishes to compare the resulting object back to (E/S), one pulls back along the unit ηS : S → ajS. For a subobject P → S, this inverse image represents the j-closure cS(P) → S; hence, in the posetal fragment, the pulled-back sheafified limit reduces to the reflected form in Theorem 4.5.

Proof Sketch. Since runs are finite, the extracted semantic shape has a finite graph presentation and finitely many certified path-equality relations. The accepted relations are part of the well-definedness of DΣ; once they are satisfied, the limit can be computed from finite products and equalizers imposing cone compatibility for the finitely many generating arrows. The slice sheafification functor a/Sj is left exact (as aj is), hence preserves finite limits. In the posetal fragment, limits in Sub(S) are meets (intersections). The sheafified mono lives over ajS; pulling it back along ηS : S → ajS gives the j-closure of the original meet. Thus the subobject seen back over S is cS( v DG(v)), recovering Theorem 4.5.

| |
|---|

#### 4.3 Formal Guarantees: Consistency and Robustness

Theorem 4.7 (Conditional consistency via closure validation). Fix E = SetCop and the Lawvere– Tierney closure c on Sub(S). For a finite DoT run and a chosen summarizer, let VΣ be the finite set of validated, c-closed subobjects selected for synthesis. The following two consistency readings are valid:

- 1. Fixed-stage finite family: If the interpreted family is jointly satisfiable relative to a fixed stage a ∈ C, i.e. there exists x ∈ S(a) lying in every DG(v)(a), then

 

DG(v)

(a) ̸= ∅

c

v∈VΣ

and the summary is non-initial. Componentwise non-emptiness of the individual DG(v)(a) is not sufficient; the premise requires a common element.

- 2. Model-compact setting: Let Γ be the set of first-order formulas represented by an idealized validated typed family, and suppose T is a first-order background theory. If every finite subset of Γ is satisfiable together with T, then compactness gives a model M |= T and an assignment satisfying all formulas in Γ. Thus the conjunction represented by the family is satisfiable in that model. If the chosen semantic interpretation of S and c is evaluated in the same model, extensivity of c preserves satisfiability of the corresponding closed summary. This is a modeltheoretic satisfiability statement, not a claim of satisfiability in the intended/standard model, and not a claim of componentwise non-emptiness in a fixed presheaf stage.

Proof Sketch. Case (1): at the fixed stage a, joint satisfiability gives a common element of all interpreted subobjects, so the componentwise finite intersection is non-empty; extensivity of c preserves non-emptiness. Case (2) is the standard compactness theorem for first-order logic, followed by the observation that an extensive closure cannot destroy a realizing assignment when the closure is interpreted in the same semantic structure.

| |
|---|

Corollary 4.8 (Soundness w.r.t. satisfiability). If [[·]] maps into Sub(S) and the selected validated family {Pv}v∈VΣ is jointly satisfiable in the same fixed stage or semantic structure in which the summary is interpreted, then the meet v Pv is satisfiable. Consequently, by extensivity, the reflected summary c( v Pv) is satisfiable in that same interpretation.

Remark 4.9 (Internal Logic, Modality, and Variance). The internal logic of E equips Sub(S) with a Heyting algebra. Validation via a Lawvere–Tierney topology promotes propositions to c-closed ones. In inclusion order, synthesis is a finite meet of constraints; after reversing variance, the corresponding diagram in the opposite information order Pred(S) = Sub(S)op has this same object as a finite colimit. Thus “gluing validated parts” should be read as a finite slice-limit, or equivalently as an information-order colimit in the predicate fragment, not as an ordinary colimit in Sub(S).

- Proposition 4.10 (Robustness under Diagram Isomorphisms). Let DΣ,1 : JΣ,1 → (E/S) and DΣ,2 : JΣ,2 → (E/S) represent selected validated reasoning steps from two different runs. If there is an isomorphism of diagrams, then their slice-limits (equivalently, information-colimits in the predicate fragment) are isomorphic:

lim DΣ,1 ∼= lim DΣ,2.

This implies that the synthesized semantic content depends on the abstract structure of the selected validated reasoning diagram, not on incidental variations that preserve this structure.

Proof Sketch. This is a direct consequence of the universal property of a limit. If two diagrams are isomorphic, there is a canonical isomorphism between their respective limits.

| |
|---|

#### 4.4 Immediate Consequences

The formalization of the DoT synthesis step as a colimit in the variance-reversed information order (equivalently, as meet-plus-closure under inclusion) entails several immediate and desirable properties, grounded in the lattice-theoretic structure of subobjects and the properties of the closure operator c.

- Proposition 4.11 (Properties of Synthesis). Let P = {Pv}v∈VΣ be a finite selected family of predicate-fragment subobjects. Define the reflected synthesis operation by Summary(P) =

c( P∈P P); when all selected inputs are validated, this equals the ordinary meet because they are c-closed and c is a nucleus. The operation exhibits the following properties:

- 1. Finiteness in practice: as runs are finite, all meets are finite; infinitary variants require compactness assumptions and are outside our core claims.
- 2. Monotonicity (information order): Adding a new selected proposition Pnew can only refine (never weaken) the reflected summary. In inclusion order,

Summary(P ∪ {Pnew}) ⊆ Summary(P).

- 3. Idempotence (finite meets): The system is robust to redundant validation. Re-processing already validated information over finite families does not alter the conclusion:

Summary(P) = c

P∈P

c(P) = c

P∈P

P .

The first equality uses finite-meet preservation of c; when validated propositions are already c-closed (P = c(P)), the statement is immediate.

- 4. Conservativity (Redundancy Elimination): If a validated proposition Pw is already no stronger

than the others’ conjunction (i.e., Pw ⊇ P∈P\{Pw} P), its explicit inclusion does not change the summary.

Summary(P) = Summary(P \ {Pw}).

Proof. These properties are direct consequences of the underlying mathematics. Monotonicity follows from the monotonicity of and c. Idempotence follows from finite-meet preservation of c, and is immediate for already c-closed inputs. Conservativity follows because if Pw contains the meet of the others, it does not change that meet.

| |
|---|

- Proposition 4.12 (Greatest c-closed Lower Bound (Canonicity)). Let P = {Pi}ni=1 ⊆ Sub(S) be a finite family of selected validated subobjects (so c(Pi) = Pi for all i), and assume c is a nucleus

(finite-meet preserving; Assumption 4.2). Then the meet ni=1 Pi is itself c-closed and is the greatest c-closed lower bound of P (in inclusion order). In particular,

c

n

i=1

Pi =

n

i=1

Pi.

Proof. Since c preserves finite meets and fixes each Pi, we have c(

i

Pi) =

i

c(Pi) =

i

Pi,

so the meet is c-closed. If X is c-closed and X ≤ Pi for all i, then X ≤ i Pi by the universal property of the meet.

| |
|---|

- Proposition 4.13 (Generalization of Linear Reasoning). A linear Chain-of-Thought (CoT) process corresponds to a special case of a DoT diagram. Suppose the selected, operationally ordered validated propositions are P1,...,Pn, and each later step strengthens the previous one, so that

P1 ⊇ P2 ⊇ ··· ⊇ Pn in Sub(S).

Equivalently, the semantic inclusion arrows point from the later stronger proposition to the earlier weaker one:

Pn −→ Pn−1 −→ ··· −→ P1. Then the synthesis simplifies to the final step:

Summary({P1,...,Pn}) = Pn.

Proof. For a chain P1 ≥ ··· ≥ Pn under inclusion, their meet is ni=1 Pi = Pn. Applying c gives Summary = c(Pn). Since Pn is validated, it is c-closed, so c(Pn) = Pn.

| |
|---|

- Proposition 4.14 (Composition of Branches). For any two finite selected families A and B, the overall summary of their union is the validated conjunction of their individual summaries (equivalently, the join in the information order). This identity is lattice-theoretic and does not require probabilistic or causal independence.

Summary(A ∪ B) = c(Summary(A) ∧ Summary(B)) = Summary(A) ∧ Summary(B).

Proof. By definition, Summary(A∪B) = c(( v∈A Pv) ∧ ( w∈B Pw)). Since c is a nucleus (finite-meet preserving), c(X ∧ Y ) = c(X) ∧ c(Y ), and thus

Summary(A ∪ B) = c

Pw = Summary(A) ∧ Summary(B),

Pv ∧ c

v∈A

w∈B

which is equivalent to the displayed formula because both individual summaries are c-closed.

| |
|---|

#### 4.5 Bridging Formalism and LLM Generation

It is crucial to understand the relationship between this formal topos-theoretic model and the actual behavior of an LLM. The LLM does not explicitly perform computations within a topos. Instead:

- • The topos framework provides the normative semantic model. It defines what constitutes sound, consistent, and robust synthesis. Theorems 4.5, 4.7, and Proposition 4.10 describe desirable properties of an ideal reasoning process.
- • The LLM, trained on DoT-structured data (using the serialization from Section 3.2), learns to generate text sequences that functionally approximate the operations described by the formalism. The generated sequence induces an abstract diagram that is then interpreted under Assumptions (A1)–(A4).
- • Specifically, the <summarizer> role learns to generate text that effectively acts like the finitelimit/information-colimit described above: it synthesizes information from selected validated precursor nodes, respects their certified dependencies, and aims for a coherent, non-redundant aggregation.
- • The fidelity of this approximation depends heavily on the training data and model capacity. The topos model provides a precise target against which the LLM’s reasoning behavior can be evaluated. One could design specific training objectives, such as a discriminative loss that penalizes generated summaries whose typed content violates the entailments dictated by the finite-limit construction.

This formalism offers a rigorous language for defining correctness criteria and provides a theoretical target for DoT behavior. Operational invalidation can either be modeled (i) as the absence of a validated inclusion (posetal fragment), or (ii) via a separate counterevidence object I ∈ Sub(S) with summaries computed as c ( valid) ∧ ¬I in the Heyting algebra Sub(S). We adopt the posetal refinement fragment for the main development and leave full revision semantics to future work.

#### 4.6 Separation from Linear Chain-of-Thought

Theorem 4.15 (Structural separation from linear CoT). Let E = SetCop, fix S ∈ E, and assume validated arrows are inclusions in Sub(S) (posetal case). Suppose a DoT summarizer selects two validated, incomparable propositions P,Q ∈ Sub(S) (i.e., P ̸≤ Q and Q ̸≤ P). Consider any attempt to faithfully embed this selected validated diagram into a linear Chain-of-Thought (a single chain of inclusions) via an order-preserving and order-reflecting functor that maps P and Q to distinct selected chain objects. No such embedding exists. Consequently, the two-branch DoT diagram cannot be represented as a faithful chain without changing the diagram; a linear trace may compute an equivalent conjunction only by adding or replacing nodes, not by embedding the original branching structure.

Proof sketch. Interpret the selected validated proposer subdiagram as a poset-category. A linear CoT is a chain (total order) in Sub(S). To rule out degenerate collapse maps, we require an order embedding, i.e., a functor that is order-preserving and order-reflecting (equivalently, full and faithful for posets, and injective on the selected objects). In a chain, any two distinct images are comparable; by order-reflection this would force P and Q to be comparable in Sub(S), contradicting incomparability. Hence no order embedding of the selected validated DoT poset into a chain exists. The DoT summary is c(P ∧Q) by Thm. 4.5; a chain can contain a later node denoting this meet, but doing so adds/replaces semantic content rather than faithfully embedding the original two-branch diagram.

| |
|---|

### 5 Conclusion

This paper introduced the Diagram of Thought (DoT), a framework that internalizes complex reasoning as DAG construction within a single auto-regressive LLM, guided by role-specific tokens and enforced by a lightweight, controller-light validator. We showed how DoT unifies proposition generation, critique, repair, and summarization without a heavyweight search/planning controller. We established a normative formalization using Topos Theory, where the synthesis of selected validated evidence corresponds to computing a finite limit in the slice (equivalently, a variancereversed information-order colimit in the predicate fragment) and reflecting it along a Lawvere– Tierney topology when necessary.

Crucially, we moved beyond informal assumptions by specifying a typed serialization with online validation, decidable checks for selected typed fragments, a monotone state-update discipline, and support for multi-premise critiques. We separated operational critique records from the Lawvere– Tierney modality they are interpreted against. Our correctness claims are conditional on these checkable and auditable mechanisms, providing a solid bridge between the operational system and its semantic model.

This topos-theoretic perspective provides several key benefits:

- • It assigns clear mathematical meaning (subobjects and slice diagrams in (E/S)) to DoT components.
- • It formalizes synthesis via closure-based validation together with a variance-reversed informationcolimit / slice-limit construction, with a precise split between the posetal and general-arrow settings (Theorem 4.5, Cor. 4.6).
- • It demonstrates semantic invariance under isomorphic rearrangements (Proposition 4.10) and compositional gluing via pullbacks/fiber products of limits (Proposition C.3).
- • It structurally extends linear CoT in the posetal setting (Proposition 4.13 and Theorem 4.15), matching intuitive gains from branching with a crisp categorical witness.

In summary, the primary advantages of DoT include an auditable reasoning trace, explicit compositional structure, and a clear theoretical target.

### References

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 17682–17690, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. arXiv preprint arXiv:2212.10403, 2022.

Peter T Johnstone. Sketches of an Elephant: A Topos Theory Compendium: Volume 2, volume 2. Oxford University Press, 2002.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35: 22199–22213, 2022.

Joachim Lambek and Philip J Scott. Introduction to higher-order categorical logic, volume 7. Cambridge University Press, 1988.

Saunders MacLane and Ieke Moerdijk. Sheaves in geometry and logic: A first introduction to topos theory. Springer Science & Business Media, 2012.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

Yang Yuan. On the power of foundation models. In International Conference on Machine Learning, pages 40519–40530. PMLR, 2023.

Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. arXiv preprint arXiv:2308.04371, 2023.

# Appendix

- A Worked Example Trace 22
- B DoT Serialization and Validation Details 23

- B.1 Grammar and Record Specification . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.2 Validation and Extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.3 Operational Semantics and Meta-Theory . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.4 Algorithm and Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- C Additional Formal Details 28

- C.1 From Critique Schemas to a Nucleus . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- C.2 Further Categorical Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- C.3 Validation and Termination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- D Detailed Proofs 29

- D.1 Proofs for Section 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- D.2 Proofs for Section 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

### A Worked Example Trace

This example shows DoT applied to a problem with multiple independent conditions, demonstrating branching and invalidation.

- @node id=1 role=problem Is the point (4, 6) inside the region defined by x > 2, y > 5, and x < 10?
- @node id=2 role=proposer

- @edge src=1 dst=2 kind=use @prop id=2 { "gt": [4, 2] } First, I will check the condition x > 2. For the point (4, 6), x=4. The inequality is 4 > 2, which is true.

@node id=3 role=critic

- @edge src=2 dst=3 kind=critique The arithmetic is correct. This condition holds. @status target=2 mark=validated

- @node id=4 role=proposer @edge src=1 dst=4 kind=use @prop id=4 { "gt": [6, 5] } Next, I will check the condition y > 5. For the point (4, 6), y=6. The inequality is 6 > 5, which is also true.
- @node id=5 role=critic @edge src=4 dst=5 kind=critique Correct. The second condition holds. @status target=4 mark=validated
- @node id=6 role=proposer @edge src=1 dst=6 kind=use @prop id=6 { "gt": [6, 0] } Let me try an alternative, irrelevant check. Maybe the condition was y > 0. For y=6,

6 > 0 is true.

- @node id=7 role=critic @edge src=6 dst=7 kind=critique This proposition is arithmetically correct, but it addresses a condition not present in the problem statement. This path is irrelevant. @status target=6 mark=invalidated
- @node id=8 role=proposer

- @edge src=1 dst=8 kind=use @prop id=8 { "lt": [4, 10] } Finally, I check the third condition, x < 10. For x=4, 4 < 10 is true.

- @node id=9 role=critic

- @edge src=8 dst=9 kind=critique This is correct. The final condition is met. @status target=8 mark=validated

- @node id=10 role=summarizer

- @edge src=2 dst=10 kind=use @edge src=4 dst=10 kind=use

- @edge src=8 dst=10 kind=use Summary: All three conditions (x > 2, y > 5, and x < 10) are met for the point (4, 6). Therefore, the point is inside the specified region. The validated propositions are ID=2, ID=4, and ID=8.

### B DoT Serialization and Validation Details

This section provides the complete details for the typed serialization, validation, and extraction mechanisms summarized in Section 3.2.

Serializer. We present a node-first, into-the-current-node serializer that guarantees acyclicity of the operational provenance DAG and supports online checks. The serializer is optional for usability, but recommended whenever formal guarantees or auditability are invoked. When it is disabled, DoT traces remain interpretable as natural language; the formal guarantees in Section 4 apply to the typed subtrace (if any), and free text is semantically inert.

- B.1 Grammar and Record Specification Node identifiers and roles. Each node carries a fresh natural-number ID and an explicit role:

@node id=⟨n⟩ role={problem,proposer,critic,summarizer}. IDs are strictly increasing in emission order.

Typed edges and emission order. Edges are explicit, typed dependencies into the current node j: @edge src=⟨i⟩ dst=j kind={refine,critique,use}, i < j.

Well-formedness requires that sources src refer only to previously emitted node IDs. For critiques, the dependency direction is proposer → critic (the critic depends on the proposition it evaluates).

Admissible role/kind pairs (type table). Let r(·) be node roles. Allowed triples for an @edge src=i dst=j kind=k are:

- • kind=critique: r(i) = proposer, r(j) = critic. A critic node’s block may only contain edges of this kind.
- • kind=refine: r(i) ∈ {proposer,critic}, r(j) = proposer. This edge is procedural unless the validator also accepts a typed strengthening witness; repair after invalidation is not assumed to entail the rejected proposition. If both endpoints are proposers and the witness certifies that the new proposer j strengthens the old proposer i, extraction adds the semantic arrow j → i, opposite to the operational edge direction.
- • kind=use: r(i) ∈ {problem,proposer}, r(j) ∈ {proposer,summarizer}. A problem source is contextual and is not itself included in the validated-proposition meet. Critics are relational and cannot be a source for use edges; instead, their semantic effect is reified via certified @entails or @eq records.

Arrow declarations (entailments) between proposers. Critiques can declare typed entailments between proposer nodes that become arrows in the extracted diagram:

@entails src=⟨i⟩ dst=⟨k⟩ [witness=⟨j⟩],

with typing requirements: r(i) = r(k) = proposer, and the witness, if written, must be a critic node j whose typed certificate is accepted by the validator. If the witness field is omitted, the current node must be a critic and is used as the witness. The intended meaning is Pi ⇒ Pk, hence a morphism DG(i) → DG(k) over S (an inclusion Pi ≤ Pk in the predicate fragment). Entailments are accepted only when certified by the typed witness and, in solver-backed fragments, by the corresponding entailment check.

##### Equivalence declarations. The basic syntax

@eq src=⟨i⟩ dst=⟨k⟩ [witness=⟨j⟩]

is an abbreviation for mutual entailment in the predicate/posetal fragment. In the general slice fragment, it is accepted only when the witness certifies an isomorphism over S or explicit inverse/pathequality data. It is not, by itself, a general path-equality language. If one implements richer path equalities, those records are added to the same congruence used in Def. 4.3.

Validation marks. Critiques emit a status for their target proposition:

@status target=⟨k⟩ mark={validated,invalidated} [just=⟨i⟩].

Only propositions with a validated status can contribute objects to the categorical synthesis diagram, and only when selected by the summarizer. A status record is well formed only when emitted by a critic that has a kind=critique edge from the target, and the optional just field must name that critic. We adopt a monotone state discipline: a proposition’s status may be set only once (first-writer wins).

Lexical discipline (fencing). Typed records must begin with the reserved sigil @. Free-form natural-language text lines must not begin with @. This ensures unambiguous lexing. To embed arbitrary text (including leading @) we use a length-prefixed fence that is streaming-safe:

@@len=⟨N⟩@@ ⟨exactly N bytes of raw text⟩

.

may contain @ and newlines

Concrete grammar (BNF). Let N be decimal naturals and Role be the set of roles.

Trace ::= NodeBlock+ NodeBlock ::= Node ; BlockItem∗ BlockItem ::= Edge | Status | Entails | Eq | PropBlock | Free

Free ::= Text | Esc Node ::= @node id=N role=(problem|proposer|critic|summarizer) Edge ::= @edge src=N dst=N kind=(refine|critique|use)

Status ::= @status target=N mark=(validated|invalidated) [ just=N] Entails ::= @entails src=N dst=N [ witness=N]

Eq ::= @eq src=N dst=N [ witness=N]

PropBlock ::= @prop id=N Prop Prop ::= solver-checkable typed formula in the selected background theory Text ::= arbitrary natural-language line not starting with @

Esc ::= @@len=N@@ Bytes{N}

The grammar permits free text before or after typed records inside a node block. The validator, not the BNF alone, enforces role-specific side conditions such as “status records may only be emitted by critic blocks” and “summarizer use-edges may point only to validated proposer nodes, except for optional contextual edges from the problem node.”

#### B.2 Validation and Extraction

Well-formedness judgment. We write Γ ⊢ Trace ok, where the context Γ = (seen,role,state,current) tracks: seen node IDs, each node’s role, node states, and the current node being emitted. Selected rules:

n > max(seen(Γ)) r ∈ {problem,proposer,critic,summarizer} Γ ⊢ @node id=n role=r ok

(Node-Intro) i ∈ seen(Γ) i < j dst = j = current(Γ) (r(i),r(j),k) admissible Γ ⊢ @edge src=i dst=j kind=k ok

(Edge-Intro)

k ∈ seen(Γ) role(k) = proposer state(k) = active role(current(Γ)) = critic k ∈ critiqueTargets(current(Γ)) Γ ⊢ @status target=k mark=m ok

Online validation. We employ a lightweight online validator V with finite control over record kinds and register-based side conditions: (i) a monotone counter for the next ID, (ii) hash maps for roles and states, (iii) a table of critique targets and accepted typed witnesses, and (iv) a congruence-closure structure for equalities. Given a candidate token, V performs local checks (e.g., src < dst = current_id) and, in typed fragments, calls the selected decision procedure for entailment/equality witnesses. At inference, this validator provides masks to the LLM decoder, ensuring only well-formed sequences are generated.

(Status-Intro)

Extraction map. Given a well-formed trace H, the syntactic extraction map Φsyn(H) returns: (i) a finite typed DAG G = (V,E); (ii) the validated proposer subgraph; (iii) for each summarizer node s, the selected synthesis presentation JΣ(s) generated by validated proposer nodes with accepted use edges into s; and (iv) an index category JG generated by certified strengthening/refinement arrows, certified @entails arrows, and certified equivalence/path-equality relations as in Def. 4.3.

Operational use, critique, and uncertified repair edges remain in the provenance DAG but do not become semantic arrows. After fixing a semantic interpretation [[·]] for typed proposition blocks and certified arrows, Φ[[·]](H) returns the corresponding diagram DG : JG → (E/S) and its selected subdiagrams DΣ(s).

Theorem B.1 (Totality and Determinism of Extraction). Any well-formed trace H satisfying the ID and edge-typing rules yields a unique typed DAG G, unique selected synthesis presentations JΣ(s) for summarizer nodes, and a unique syntactic index category JG under Φsyn. After fixing a semantic interpretation [[·]], it yields a unique diagram DG : JG → (E/S) and selected subdiagrams DΣ(s). Excluding solver-call time, extraction is linear up to dictionary operations in the absence of nontrivial equality/path-equality declarations; with union-find equivalence classes it runs in O(|H|α(|H|)) time and O(|H|) space. Validation adds the sum of the selected theory solver costs. When general path equalities are present, we maintain congruence-closure on arrows (e.g., via e-graphs); this is near-linear in typical traces but can be super-linear in the worst case, depending on the signature and saturation strategy.

#### B.3 Operational Semantics and Meta-Theory

Standing scope. We treat DoT’s semantics as a normative target: the LLM approximates the finite-limit/information-colimit behavior, while V ensures the typed fraction is well-formed and auditable.

- Proposition B.2 (Relative soundness to typed subtrace). Any conclusion in Section 4 is a function of the extracted typed diagram Φ(H) alone; untyped/free text does not affect the semantics.

We give selected small-step rules over states (G,σ,H) where G is the partial DAG, σ the node-state map, and H the emitted prefix.

Rules (sketch). Proposer-Intro: on a well-typed @node with role=proposer, extend G with a vertex v, set σ(v) = active. Critic-Intro: on role=critic, add the critic node. Validate: on @status with mark=validated for target u where σ(u) = active, set σ(u) = validated. Invalidate: on @status with mark=invalidated for target u where σ(u) = active, set σ(u) = invalidated.

Theorem B.3 (Order-Invariance (semantic; typed-content invariant)). Let a well-posed trace be one that (i) obeys the NodeBlock grammar, and (ii) contains at most one @status record for each proposer node ID (single-assignment). Normalize typed records by resolving omitted witness and just fields to their enclosing critic block and by discarding free text. Consider two well-posed traces, H1 and H2, that contain the same multiset of normalized typed records (@node, @edge, @status, @entails, @eq) and induce the same dependency partial order on those records after quotienting by validated equalities. Then their extracted index categories are isomorphic, JG1

∼= JG2. For corresponding summarizer nodes, the selected synthesis presentations are also isomorphic. In the posetal case, the resulting subobjects are equal; in the general case, their reflected slice-limits are isomorphic.

#### B.4 Algorithm and Implementation Details

The state of the reasoning process (the DAG) is implicitly encoded within the auto-regressive history Ht. The LLM conditions its prediction on this history, using its attention mechanism to represent the current state of the DoT graph. Algorithm 1 provides a high-level sketch.

Algorithm 1 Diagram of Thought (DoT) Generation Process

- 1: Input: Problem statement P
- 2: Initialize generation history H with serialized problem statement for node v1; set current node j ← 1.
- 3: Initialize node states (e.g., in a dictionary) σ[v1] ← initial.
- 4: while termination condition not met (e.g., max length, <summarizer> generated) do
- 5: Predict next role token r ∈ Troles based on history H: r ∼ LM(H).
- 6: Append r to H.
- 7: if r = <proposer> then
- 8: Emit @node id= j+1 role=proposer; set j ← j+1.
- 9: Emit zero or more well-typed @edge src= i dst= j with i < j.
- 10: Generate proposition text Pj. Append records and text to H.
- 11: Update state: σ[j] ← active.
- 12: else if r = <critic> then
- 13: Emit @node id= j+1 role=critic; set j ← j+1.
- 14: Emit one or more @edge src= km dst= j kind=critique.
- 15: Generate critique text Cj; then emit one or more @status target= k mark=validated|invalidated.
- 16: Optionally emit certified @entails or @eq records.
- 17: Append records and text to H.
- 18: Update target states monotonically: if σ[k] = active, set σ[k] ← m.
- 19: else if r = <summarizer> then
- 20: Emit @node id= j+1 role=summarizer; set j ← j+1.
- 21: Emit zero or more @edge src= i dst= j kind=use from selected validated proposer nodes (and optionally the problem node as context).
- 22: Generate summary text S. Append records and text to H.
- 23: Set termination condition to true.
- 24: end if
- 25: end while
- 26: Output: Final text S from the summarizer node vS.

Controller-light decoding. Decoding uses a deterministic, state-dependent mask derived from the validator’s finite control, registers, and local typing maps. Illegal token classes are never sampled. There is no branching search or external resampling loop. The only state external to the LM is the validator’s finite map store and any solver state needed for the chosen typed fragment.

Auxiliary supervision for summaries-as-limits. To align the <summarizer> with the normative target, one can add an auxiliary loss that penalizes violations of literals that are entailed by the finite meet of selected validated propositions. This complements the standard maximum-likelihood training on full traces.

### C Additional Formal Details

#### C.1 From Critique Schemas to a Nucleus

Critique schemas do not automatically determine a Lawvere–Tierney topology. To avoid this gap, the main text assumes a fixed topology j and its universal closure operator. When one wants to derive such a topology from typed critique schemas, only the modality-producing part of the schema should be compiled into pullback-stable closure sequents; certified entailments, equivalences, and exclusions remain separate typed records unless explicitly encoded as such sequents. Thus the items below are constraints on a topology, not arbitrary closure rules on the single lattice Sub(S):

- • Certified entailment (LE): a validated critique may introduce a morphism P → Q over S (an

inclusion P ≤ Q in the predicate fragment). This is a semantic arrow in JG; it is not, by itself, a closure axiom.

- • Modal validation sequent: if a schema is intended to enlarge the closure of a predicate, it is compiled as a pullback-stable lower-bound sequent Q ≤ cA(P) for suitable subobjects P,Q → A.
- • Equivalence identification (EQ): a validated critique emitting mutual entailments in the predicate fragment or an isomorphism/equivalence record between proposer objects in the general slice fragment. This is handled by certified arrows and congruence equations, not by closure alone.
- • Type refinement (TR): narrows a subobject P by pullback along a mono R → S, producing P ∧ R and a certified strengthening arrow P ∧ R → P.

The local operators on a topos form a complete lattice. For the closure constraints used here, the relevant modality sequents are lower-bound closure requirements of the form Q ≤ cA(P), closed under pullback. Satisfaction of such sequents is stable under arbitrary meets of local operators under the usual lattice order; hence the satisfying topologies, when nonempty, have a least element. For purely lower-bound closure constraints the chaotic topology satisfies the constraints, so nonemptiness is automatic. Equality, isomorphism, and exclusion constraints are not generated by this lattice argument alone; they must be certified separately or compiled into stable closure sequents whose consistency has been checked. The associated universal closure operator is then extensive, monotone, idempotent, pullback-stable, and finite-meet preserving by construction. This is the limited sense in which critique schemas can induce a validation modality.

- Lemma C.1 (Rule closure & nucleus completion). Let R be a set of pullback-stable lower-bound

closure sequents Qr ≤ cAr(Pr) generated by the modality-producing part of the typed critique schemas. If at least one Lawvere–Tierney topology satisfies R, and if satisfaction of R is closed under

meets of local operators (as it is for these lower-bound sequents), let jR be the meet of all satisfying topologies. Then jR is the least topology satisfying R. Its associated universal closure operator cR = (cRA)A∈E is extensive, monotone, idempotent, pullback-stable, and finite-meet preserving. In particular, cRS is a nucleus on Sub(S).

C.2 Further Categorical Results

- Lemma C.2 (Slice reflection and finite-limit stability). Let aj : E → Shj(E) be sheafification, which is left exact. The induced functor on slices

a/Sj : (E/S) → (Shj(E)/ajS), (X → S)  → (ajX → ajS), is also left exact, and hence preserves finite limits. Moreover, for a mono m : P → S, the inverse image in E of the mono ajm : ajP → ajS along the unit ηS : S → ajS represents the j-closure

cS(P) → S. This is the stability property required for the synthesis construction in Cor. 4.6 (finite limits in (E/S) followed by a/Sj , and comparison back to (E/S) by pullback along ηS).

- Proposition C.3 (Gluing validated diagrams via pullbacks of limits). Let D1 : J1 → (E/S) and D2 : J2 → (E/S) be selected validated diagrams whose overlap is a common subdiagram

- K : JK → (E/S). Then, in a presheaf topos, there is a canonical isomorphism lim(D1 ∪K D2) ∼= lim(D1) ×lim(K) lim(D2),

whenever the displayed union is the pushout of finite index shapes along the common subshape and the two diagrams agree on K. Hence, after reflection, the overall summary satisfies

Summary(D1 ∪K D2) ∼= a/Sj lim(D1) ×lim(K) lim(D2) .

Proof sketch. Giving a cone over the union diagram is equivalent to giving a cone over D1 and a cone over D2 whose restrictions to K agree. The representing object for such compatible pairs of cones is the pullback of the two individual limits over the overlap limit. Finally, a/Sj preserves finite limits by Lem. C.2.

| |
|---|

- C.3 Validation and Termination

We enforce well-formedness with the validator V (Section B). In inference, decoding is constrained by V ; violations are rejected before token commitment when they are locally detectable, and completed typed records are checked before acceptance. Termination is enforced operationally by a budget on node blocks/typed records together with the explicit <summarizer:end> token.

Defeasible validation. For more complex, non-monotone reasoning, one may choose in advance a finite priority set ρ ∈ {0,...,R} and an increasing chain of Lawvere–Tierney topologies, with associated nuclei c≤ρ. A retraction step from rank ρ to ρ′< ρ replaces the active modality by c≤ρ′ on affected objects. This optional extension lies outside the monotone core; the order-invariance result applies for a fixed active modality.

D Detailed Proofs

This appendix provides detailed proofs for the theorems, propositions, and lemmas presented in the main text. We assume familiarity with basic concepts from category theory and topos theory, as found in references like (MacLane and Moerdijk, 2012; Johnstone, 2002).

- D.1 Proofs for Section 3

Proof of Theorem B.1. The proof proceeds by demonstrating totality, determinism, and analyzing the computational complexity.

The syntactic extraction map Φsyn is defined for any trace H that is deemed well-formed by the validator V . Well-formedness ensures that every record in the trace can be unambiguously parsed and assigned a syntactic action.

Every @node record has a unique, strictly increasing ID. Every @edge record refers to a src ID that has already been emitted and a dst ID corresponding to the current node, preventing forward

references and cycles in the operational dependency graph of records. Typed records for status, entailments, and equalities have their targets and roles checked for validity.

Since every syntactically valid record has a defined extraction action (e.g., add a node, add an edge, update a state, record a summarizer selection, record an entailment/equivalence), the extraction process is defined for the entire trace. Hence, Φsyn is total on the set of well-formed traces.

We must show that a given well-formed trace H maps to a single, unique syntactic diagram. The set of nodes V in the extracted DAG G is uniquely determined by the set of @node records. The set of operational provenance edges E is uniquely determined by the set of @edge records. The selected synthesis presentations are uniquely determined by the accepted @edge kind=use records into each summarizer and by the validated states of their source proposer nodes. The index category JG is formed by taking the generated category on proposer nodes, certified strengthening/refinement arrows, and certified entailment arrows, then quotienting by the smallest congruence generated by certified equivalence/path-equality records. Operational provenance edges that lack a typed semantic certificate do not enter JG. The smallest congruence is unique.

After a semantic interpretation [[·]] is fixed, the diagram functor DG : JG → (E/S) maps each object (proposer node) to its interpreted object over S and each certified arrow to its interpreted morphism over S.

Because each syntactic step is a deterministic function of the input trace, the final syntactic output (G,JG,{JΣ(s)}s) is unique; with fixed [[·]], the semantic diagram DG and selected subdiagrams DΣ(s) are unique as well.

Now, let us calculate the complexity:

- • Parsing the trace H is a single linear pass, O(|H|).
- • Building the graph structure, selected synthesis presentations, and semantic generators involves processing each record once. Using hash maps to store node information (roles, states), this takes O(|H|) time and space, excluding solver calls.
- • When only node equivalences are present, managing these with a union-find data structure takes O(|H|α(|H|)) time, where α is the inverse Ackermann function.
- • Solver-backed validation adds the sum of the costs of the selected decision procedures for the typed witnesses.
- • When path equalities are introduced, a more complex congruence closure algorithm is needed. While algorithms like e-graphs perform with near-linear amortized time in practice, their worstcase complexity can be higher depending on the specific theory. We state the complexity parametrically in this case.

The stated complexity bounds follow.

| |
|---|

Proof of Theorem B.3. The core insight is that the extraction map Φ is sensitive only to the set of normalized typed records and their dependency partial order, not the specific linear sequence in which they appear, provided that sequence is a valid topological sort of the dependency graph.

Since H1 and H2 contain the same multiset of normalized typed records, they will result in the same set of proposer nodes, the same dependency edges between nodes, the same status assignments, the same selected summarizer sources, the same set of entailment records, and the same set of declared equivalences. The normalization step resolves implicit fields such as omitted witnesses, so equality is being tested on explicit records rather than on surface syntax.

The well-formedness rules (src < dst, etc.) ensure that any valid trace is a topological sort of the underlying operational dependency graph of records. If H1 and H2 induce the same dependency partial order, they are simply two different valid topological sorts of the same abstract structure.

The extraction map Φ constructs the diagram by first identifying all nodes, provenance edges, summarizer selections, and semantic generators from the records and then applying the validated equalities. This process does not depend on the linear order of emission, only on the final set of normalized records and their dependency constraints. Therefore, Φ(H1) and Φ(H2) produce isomorphic abstract graphs, the same validated equalities, and thus isomorphic index categories JG and selected presentations JΣ. By Proposition 4.10, isomorphic diagrams have isomorphic limits. After reflection, the resulting summaries are isomorphic. In the posetal sub-case where the summary is a specific subobject (the meet-plus-closure), the summaries are equal.

| |
|---|

#### D.2 Proofs for Section 4

- Theorem D.1 (DoT Process as Diagram Construction). A DoT reasoning process generating a well-formed typed DAG G = (V,E), together with a fixed semantic interpretation of typed propositions and certified arrows, defines a functor (a diagram) DG : JG → (E/S) with:

- • Each typed proposer node v mapped to an object DG(v) → S in the slice (a subobject DG(v) → S in the predicate/mono fragment);
- • Accepted critic records supply certified arrows, isomorphisms, or path equalities between proposer objects; critic nodes are witnesses for these records rather than objects of DG;
- • Each arrow v → u in JG is mapped to a certified morphism over S, witnessing entailment or strengthening coherence (posetal case: an inclusion DG(v) → DG(u)).

Functoriality ensures that composite certified dependencies (via paths modulo ≡) are respected in the slice.

Proof. We construct the functor DG : JG → (E/S) by defining its action on objects and morphisms and verifying that it satisfies the functoriality axioms.

As per Definition 4.3, the extraction map Φ applied to the trace yields a DAG G and an index category JG with proposer nodes as objects and syntactically certified arrows as morphism generators, modulo certified equality constraints.

For each object v ∈ Ob(JG), Assumption (A1) states that its content can be interpreted as an object over S (a subobject of S in the predicate/mono fragment). We define the action of DG on objects as this interpretation:

DG(v) := [[content(v)]] → S. This defines an object in the slice category (E/S).

Now let α : v → u be a generator in JG. If α is induced by a certified strengthening/refinement record, Assumption (A1) assigns it a morphism DG(v) → DG(u) over S. If α is induced by @entails src=v dst=u, the certified entailment directly supplies the same kind of morphism. If α is part of a certified @eq record, then in the predicate fragment it is mutual entailment, while in the general slice fragment the witness supplies an isomorphism and inverse equations. We then extend DG from generators to arbitrary morphisms by composition.

Well-definedness with respect to ≡ follows from Assumption (A3): if two generated composites are identified by validated equality records, then their interpreted composites in the slice are equal. Identities map to identities, and composition in JG maps to composition in (E/S). Hence DG is a functor. Thus, DG is a valid functor from the index category JG to the slice category (E/S).

| |
|---|

- Theorem D.2 (Summarization as finite meet-plus-closure in the reflective subposet). Assume every proposer selected for synthesis is interpreted as a c-closed subobject of S and every certified semantic arrow between selected proposers is an inclusion, so the selected diagram DΣ lands in the thin category Sub(S). Let VΣ be the finite set of selected validated proposer nodes. Then the finite-limit summary is

SummaryΣ = limSub(S) DΣ =

DG(v),

v∈VΣ

equivalently the finite colimit of DΣop in Pred(S) = Sub(S)op. The reflected form c

DG(v)

v∈VΣ

is equal to this meet.

Proof. Work in Sub(S) ordered by inclusion, and in the opposite poset Pred(S) = Sub(S)op (information order). For a finite family {Pv}, the join (colimit) in Pred(S) corresponds to the meet (intersection) in Sub(S):

Pred(S) Pv = Sub(S) Pv.

More generally, since Sub(S) is thin, the finite limit of the selected posetal diagram is the greatest lower bound of its objects. Equivalently, after passing to the opposite diagram DΣop : JΣop → Pred(S), this same object is the information-order colimit. Now assume each validated proposition is c-closed, i.e. c(Pv) = Pv, and that c is a nucleus (finite-meet preserving; Assumption 4.2). Then the meet of validated propositions is again c-closed:

Pv =

c(Pv) =

c

Pv.

v

v

v

Thus, the synthesis object lies in the c-closed fragment and agrees with the information-colimit. We may write the summary as c( v Pv) to match the general-arrow presentation, noting that under the standing assumptions the outer c is redundant.

| |
|---|

Corollary D.3 (General case via slice sheafification). For a general selected diagram DΣ : JΣ → (E/S) with non-posetal arrows and certified coherence/path-equality relations already satisfied by the extracted functor, the synthesized summary is

SummaryΣ ∼= limShj(E)/ajS a/Sj ◦ DΣ ∼= a/Sj limE/S DΣ ,

where a/Sj : (E/S) → (Shj(E)/ajS) is the functor induced by sheafification. Proof. The logic is analogous to the posetal case but lifted from posets to general categories, using finite limits rather than meets.

In the presheaf topos setting, the slice (E/S) is finitely complete. Since selected diagrams from finite runs have finite graph presentations, the required limit L = lim DΣ is a finite-limit construction in (E/S): finite products collect cone components, and equalizers impose compatibility with the finitely many generating arrows. Certified path-equality records are checked when constructing the functor DΣ; they are not used to repair an otherwise incoherent diagram at synthesis time. Intuitively, this limit enforces simultaneous satisfaction/compatibility of the selected validated constraints represented by the diagram.

The Lawvere–Tierney topology j defines a reflective subcategory of j-sheaves, Shj(E) → E. The reflector is the sheafification functor aj. This induces a left exact functor on slice categories, a/Sj : (E/S) → (Shj(E)/ajS), as stated in Lemma C.2. This functor maps objects in the slice to their validated/sheaf counterparts over ajS.

Since a/Sj is left exact, it preserves finite limits. Therefore, the summary (synthesis computed within the validated/sheaf slice) can be obtained by computing the finite limit in the ambient slice and then applying the reflector:

SummaryΣ ∼= lim

(a/Sj ◦ DΣ) ∼= a/Sj ( lim

DΣ).

Shj(E)/ajS

(E/S)

When restricted to subobjects, the sheafified mono lives over ajS. Pulling it back along ηS : S → ajS gives the j-closure cS of the original subobject. The colimit in the information order corresponds to the meet in inclusion order after reversing variance. Thus, for a posetal diagram, comparison back over S gives cS( v DG(v)), recovering the result of Theorem 4.5.

| |
|---|

Theorem D.4 (Conditional consistency via closure validation). Fix E = SetCop and a closure c on Sub(S). In the fixed-stage finite-family setting, if the selected validated family is jointly satisfiable relative to a fixed stage a0 ∈ C, then the summary is non-initial. In the model-compact setting, finite satisfiability of the corresponding first-order family implies satisfiability in some model, but not necessarily in the intended/standard model and not necessarily as componentwise non-emptiness in a preassigned presheaf stage.

Proof. Let L = v∈V

DG(v) be the meet of the interpretations of the selected validated propositions.

Σ

In the fixed-stage case, the summary is SummaryΣ = c(L). We show that the summary is a non-initial subobject of S.

The premise states that the finite family {DG(v)}v∈VΣ is jointly satisfiable relative to a fixed stage a0 ∈ C. In the presheaf topos E = SetCop, this means there exists an element x ∈ S(a0) such that for every v ∈ VΣ, x lies in the subset DG(v)(a0) ⊆ S(a0).

The existence of such an element x implies that the intersection of these subsets is non-empty:

DG(v)(a0) ̸= ∅.

v∈VΣ

In a presheaf topos, finite limits are computed componentwise. Therefore, the component of the meet subobject L at stage a0 is precisely this intersection:

DG(v)

L(a0) = 

(a0) =

DG(v)(a0).



v∈VΣ

v∈VΣ

Since this set is non-empty, the subobject L is non-initial.

The closure operator c is extensive, meaning that for any subobject P, we have an inclusion P → c(P). Since presheaf monomorphisms are componentwise injections, applying this to our meet

- L at stage a0 gives: L(a0) ⊆ (c(L))(a0).

Since we established that L(a0) is non-empty, its superset (c(L))(a0) must also be non-empty. The summary, SummaryΣ = c(L), has a non-empty component at stage a0. Therefore, the summary is a non-initial subobject.

For the model-compact setting, compactness is applied to the first-order family represented by the typed propositions. If every finite subset is satisfiable together with the background theory, then there is a model and assignment satisfying the entire family. When S and c are interpreted in that same model, extensivity again preserves satisfiability after closure. This proves consistency in the model-theoretic sense while deliberately avoiding any intended-model or fixed-stage presheaf claim.

| |
|---|

- Proposition D.5 (Robustness under Diagram Isomorphisms). Let DΣ,1 : JΣ,1 → (E/S) and DΣ,2 : JΣ,2 → (E/S) represent selected validated reasoning steps from two different runs. If there is an isomorphism of diagrams, then their slice-limits are isomorphic.

Proof. Let (σ,η) be an isomorphism of diagrams, i.e. σ : J1 → J2 is an isomorphism of index categories and η : D1 ⇒ D2 ◦ σ is a natural isomorphism. Precomposition with σ induces an equivalence between cones over D2 and cones over D2 ◦ σ; moreover, the natural isomorphism η induces an equivalence between cones over D1 and cones over D2 ◦σ. Equivalences preserve terminal objects, so the terminal cone over D1 (its limit) corresponds to the terminal cone over D2 (its limit). Hence lim D1 ∼= lim D2.

| |
|---|

- Proposition D.6 (Properties of Synthesis). The synthesis operation, Summary(P) = c( P∈P P), exhibits Monotonicity, Idempotence, and Conservativity.

Proof. Let P = {Pv}v∈VΣ. The properties follow directly from the definition of the summary and the standard properties of meet (∧) in a poset and a closure operator (c).

For monotonicity, we want to show that Summary(P ∪ {Pnew}) ⊆ Summary(P). The meet of a larger set of subobjects is always a subobject of the meet of a smaller set:

P =

P ∧ Pnew ⊆

P.

P∈P

P∈P

P∈P∪{Pnew}

The closure operator c is monotone: if X ⊆ Y , then c(X) ⊆ c(Y ). Applying c to both sides of the inclusion above preserves the relation:

 

  ⊆ c

c

P

P .

P∈P

P∈P∪{Pnew}

This is the desired result.

For idempotence, we want to show Summary(P) = c( P∈P c(P)). Using finite-meet preservation of c,

c(P) =

c(c(P)) =

c(P).

c

P∈P

P∈P

P∈P

If all propositions in P are validated, then each is already c-closed, so P = c(P), and the displayed expression equals c( P∈P P).

For conservativity, assume Pw ⊇ P∈P\{Pw} P. We want to show Summary(P) = Summary(P \ {Pw}). The meet of all propositions in P is:

P = 

  ∧ Pw.



P

P∈P

P∈P\{Pw}

Since Pw contains the meet of the others, intersecting with Pw does not change the result. That is, if X ⊆ Y , then X ∧ Y = X. Therefore:

  ∧ Pw =

 

P.

P

P∈P\{Pw}

P∈P\{Pw}

Applying the closure operator c to both sides gives:

 

 ,

P = c

c

P

P∈P

P∈P\{Pw}

which is precisely Summary(P) = Summary(P \ {Pw}).

| |
|---|

