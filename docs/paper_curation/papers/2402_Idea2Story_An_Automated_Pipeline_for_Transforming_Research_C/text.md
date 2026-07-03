# arXiv:2601.20833v1[cs.CE]28Jan2026

## Idea2Story: An Automated Pipeline for Transforming Research Concepts into Complete Scientific Narratives

Tengyue Xu*, Zhuoyang Qian*, Gaoge Liu*, Li Ling*, Zhentao Zhang*, Biao Wu*, Shuo Zhang, Ke Lu, Wei Shi, Ziqi Wang, Zheng Feng, Yan Luo, Shu Xu, Yongjin Chen, Zhibo Feng, Zhuo Chen, Bruce Yuan, Harry Wang†, Kris Chen†

AgentAlpha Team

†Corresponding author

Autonomous scientific discovery with large language model (LLM)-based agents has recently made substantial progress, demonstrating the ability to automate end-to-end research workflows. However, existing systems largely rely on runtime-centric execution paradigms, repeatedly reading, summarizing, and reasoning over large volumes of scientific literature online. This on-the-spot computation strategy incurs high computational cost, suffers from context window limitations, and often leads to brittle reasoning and hallucination. We propose Idea2Story, a pre-computation–driven framework for autonomous scientific discovery that shifts literature understanding from online reasoning to offline knowledge construction. Idea2Story continuously collects peer-reviewed papers together with their review feedback, extracts core methodological units, composes reusable research patterns, and organizes them into a structured methodological knowledge graph. At runtime, underspecified user research intents are aligned to established research paradigms, enabling efficient retrieval and reuse of high-quality research patterns instead of open-ended generation and trial-and-error. By grounding research planning and execution in a pre-built knowledge graph, Idea2Story alleviates the context window bottleneck of LLMs and substantially reduces repeated runtime reasoning over literature. We conduct qualitative analyses and preliminary empirical studies demonstrating that Idea2Story can generate coherent, methodologically grounded, and novel research patterns, and can produce several high-quality research demonstrations in an end-to-end setting. These results suggest that offline knowledge construction provides a practical and scalable foundation for reliable autonomous scientific discovery. Our codebase is publicly available at https://github.com/AgentAlphaAGI/Idea2Paper.git.

[Figure 1]

Date: January 29, 2026

#### 1 Introduction

As research increasingly moves toward fully autonomous scientific discovery, large language model (LLM)-based agents have attracted growing attention for their ability to automate complex research workflows (Chai et al., 2025; Cornelio et al., 2023; Wang et al., 2023; Xu et al., 2021). Recent systems (Lu et al., 2024; Yamada et al., 2025; Gottweis et al., 2025) demonstrate that LLM-based agents can autonomously execute an end-to-end research loop, including literature review, code generation, experiment execution, and manuscript drafting. These results suggest that automated scientific discovery is becoming practically feasible and that LLM-based agents are approaching a level of functional completeness required for autonomous research (Jin et al., 2024; Sahu et al., 2025; Ajith et al., 2024; Zhang et al., 2025b, 2026).

Despite this progress, existing systems remain constrained by a fundamental inefficiency in their execution paradigm, which limits their scalability and robustness in practice. In particular, most current research agents (Wang et al., 2025b; Yang et al., 2024; Mitchener et al., 2025; Luo et al., 2025) rely on an on-the-spot computation strategy, where nearly all information acquisition, reasoning, and synthesis are performed online at runtime. Under this paradigm, each new research attempt requires the agent to dynamically retrieve large volumes of scientific literature, read and summarize long and heterogeneous documents in real time, and explore a broad space of candidate methods and experimental designs through open-ended generation and trial-and-error. As a result, the cost of producing a single effective scientific discovery remains substantial. For example, a complete execution of the overall pipeline often requires several hours and, in some cases, up to 15 hours to progress from ideation to experimentation (Lu et al., 2024). Similarly, in (Schmidgall

[Figure 2]

- Figure 1 Overview of the two-stage framework in Idea2Story. The offline stage constructs a structured knowledge graph by extracting and organizing reusable method units from a curated paper corpus. The online stage retrieves and composes research patterns from the knowledge graph to ground underspecified user intent into concrete and coherent research directions.

et al., 2025b), literature review and experimental planning alone account for a significant portion of total inference time and place heavy demands on the language model’s ability to maintain coherent reasoning over long contexts. More importantly, this runtime-centric design repeatedly forces the model to re-process large volumes of unstructured and partially redundant information, even when much of the underlying scientific knowledge is already well established, thereby increasing computational overhead and exacerbating the risk of hallucination and reasoning errors (Wang et al., 2025a; Shin et al., 2025).

To address the efficiency and reliability limitations of existing autonomous research agents, we propose Idea2Story, a scientific discovery framework that explicitly separates offline knowledge construction from online research generation, with the goal of reducing repeated reasoning over scientific literature and alleviating the context window bottleneck of large language models. Most current systems rely on runtime-centric execution, where agents repeatedly retrieve, read, summarize, and reason over large collections of highly overlapping papers for each new research attempt, resulting in substantial computational cost and prolonged execution time. Idea2Story mitigates this inefficiency by shifting literature understanding from online reasoning to an offline stage. In the offline phase, the system periodically collects recently accepted, peer-reviewed papers together with their full review feedback, extracts core methodological units and research patterns, and organizes these units and their observed composition relations into a continuously updated structured knowledge graph. This knowledge graph serves as a compact and reusable representation of established scientific methods and their empirical compatibility, replacing repeated processing of raw documents at runtime. Building on this offline knowledge infrastructure, Idea2Story performs online research generation by aligning underspecified user research intents with existing research paradigms encoded in the knowledge graph. Rather than relying on open-ended generation and trial-and-error, the system retrieves high-quality research patterns as structured compositions of method units, which act as stable methodological blueprints for downstream experimental design and execution. Guided by these validated research patterns, Idea2Story conducts feasibility-driven experimentation and ultimately generates a complete, submission-ready paper in an end-to-end manner.

Our work makes the following contributions to autonomous scientific discovery : (1) We introduce Idea2Story, a framework that formalizes autonomous research as a pre-computation–driven process, where scientific knowledge is extracted, structured, and maintained in a continuously updated methodological knowledge graph, addressing the

inefficiency and unreliability of runtime-centric research agents. (2) We propose a knowledge-grounded planning and execution pipeline that alleviates the context window bottleneck and reduces repeated runtime reasoning over literature by converting paper reading into retrieval over a pre-built knowledge graph. (3) We conduct preliminary empirical studies and comparative evaluations, demonstrating that Idea2Story can produce several high-quality research demos and establishing the practical feasibility of the proposed paradigm in an end-to-end setting.

#### 2 Related Work

##### 2.1 Autonomous Scientific Discovery

Recent advances in large language models (LLMs) have driven growing interest in autonomous scientific discovery agents that aim to automate the full research lifecycle, from code generation to experimental execution (Hu et al., 2026; Zhang et al., 2025a; Lin et al., 2025). Early systems such as The AI Scientist (v1) (Lu et al., 2024) demonstrate the viability of end-to-end automation but rely heavily on manually crafted code templates and largely linear exploration workflows, which restrict discovery depth and adaptability. Later approaches, including The AI Scientist-v2 (Yamada et al., 2025) and Kosmos (Mitchener et al., 2025), reduce reliance on explicit template through the incorporation of agentic tree search and experiment management agents, enabling iterative and multi-round exploration.

In research ideation, LLM-generated ideas are often perceived as highly novel during initial screening; however, prior studies (Si et al., 2024) uncover a critical paradox whereby such ideas tend to underperform after implementation relative to human-generated ideas, indicating limited feasibility and practical utility. As more ideas are generated, LLM outputs exhibit growing similarity, leading to diminished meaningful diversity. Similar limitations have also been observed in research evaluation and peer review (Liang et al., 2024; Xu et al., 2025; Thakkar et al., 2025; Zhang et al., 2026). Existing AI-based reviewers display systematic blind spots: Shin et al. (2025) shows that LLM reviewers place disproportionate emphasis on technical correctness while undervaluing novelty, deviating from human expert judgment, while Sahu et al. (2025) demonstrates that AI reviewers struggle to distinguish fine-grained acceptance categories and are susceptible to sycophancy, with review scores increasing unreasonably after exposure to author rebuttals. Although recent approaches such as AgentReview (Jin et al., 2024) seek to mitigate these deficiencies by simulating diverse reviewer roles, automated evaluation systems remain less reliable than human experts in identifying robust accept/reject decision boundaries.

##### 2.2 LLM-Driven Agents

LLM-driven agents still struggle to interact effectively with complex real-world environments. Despite their strong generative capabilities, many existing systems—such as OpenHands (Wang et al., 2025b) and SWE-Agent (Yang et al., 2024)—exhibit limited performance when applied to realistic codebases. These limitations largely stem from insufficient reasoning over hierarchical dependencies and structural constraints, as well as the inherent restrictions imposed by finite context windows. As a result, LLM-driven agents achieve relatively low task completion rates on challenging benchmarks such as MLE-bench (Chan et al., 2024) and SciCode (Tian et al., 2024). RepoMaster (Wang et al., 2025a) further identifies inadequate modeling of codebase structure, including function call graphs and module dependency graphs, as a key bottleneck for LLM-driven agents operating in large and complex environments.

Beyond execution limitations, LLM-driven agents also exhibit notable deficiencies in scientific rigor and evaluative judgment. When tasked with autonomous assessment, these agents are prone to hallucination and overconfidence. For instance, Agent Laboratory (Schmidgall et al., 2025b) reports that automated evaluations produced by LLM-driven agents substantially overestimate paper quality compared to human reviewers. Evaluations of Kosmos (Mitchener et al., 2025) further reveal a tendency to invent opaque quantitative metrics and to conflate statistical significance with scientific value, leading to weak interpretability of experimental conclusions. Moreover, long-horizon autonomous execution exacerbates these issues by introducing behavioral drift (Arike et al., 2025), where LLM-driven agents gradually deviate from intended research trajectories or generate overly strong and insufficiently justified claims (Lu et al., 2024; Schmidgall et al., 2025a; Baek et al., 2025; Hong et al., 2023; Wu et al., 2023; Lin et al., 2025; Hu et al., 2026). This drift further undermines reliability and highlights the need for stronger structural grounding and validation mechanisms in LLM-based autonomous research systems.

#### 3 General Idea Generation

Idea2Story is designed to interact with users through high-level and often informal research ideas that reflect human intuition rather than fully specified technical plans. The system transforms such underspecified inputs into structured and academically grounded research directions through a two-stage paradigm that separates offline knowledge construction from online research generation:

- • Offline Knowledge Construction. In the offline stage, Idea2Story builds a reusable methodological foundation from existing scientific literature. This includes curating a large-scale paper pool from peer-reviewed venues, extracting reusable method units that capture core methodological contributions, and organizing these units into a structured knowledge graph that encodes their semantic and compositional relations. The resulting knowledge graph serves as a persistent repository of methodological abstractions, decoupling literature understanding from runtime reasoning.
- • Online Research Generation. In the online stage, Idea2Story grounds user-provided research ideas through retrieval and composition over the pre-built knowledge graph. Given an informal user idea, the system aligns the input with existing research paradigms, retrieves relevant research patterns, and composes compatible method units into concrete research directions. These instantiated patterns are further refined through a review-guided process that iteratively evaluates and revises them with respect to novelty, methodological soundness, and conceptual coherence. The refined research patterns then serve as structured blueprints for subsequent planning, feasibility-driven experimentation, and end-to-end paper generation.

##### 3.1 Offline Knowledge Construction

The offline knowledge construction stage aims to distill reusable methodological structure from existing scientific literature and to organize it in a form that can be efficiently accessed during online research generation. Instead of performing document-level reasoning at runtime, Idea2Story pre-computes a structured representation of prior work that captures both methodological abstractions and their observed compatibility in accepted research. This stage consists of three main components: (i) constructing a curated paper pool from peer-reviewed venues, (ii) extracting core method units that represent reusable methodological contributions, and (iii) organizing these units and their composition relations into a structured knowledge graph. Together, these components form a persistent methodological memory that decouples literature understanding from downstream idea grounding and research generation.

- 3.1.1 Paper Pool Construction

We construct a paper pool from accepted machine learning papers and their associated peer reviews collected from top-tier conferences. Let C = {NeurIPS, ICLR} denote the set of venues considered, and let T denote the most recent three-year time window. The resulting paper pool is defined as

P = { 𝑝 | 𝑝 is an accepted paper from 𝑐 ∈ C during T },

which consists of approximately 5,000 papers from NeurIPS and 8,000 papers from ICLR. For each paper 𝑝 ∈ P, we retain the full textual content

x𝑝 = (title𝑝, abstract𝑝, body𝑝), together with its associated review artifacts

r𝑝 = {comments, ratings, confidence scores, meta-reviews}.

This yields a temporally aligned corpus that jointly captures research contributions and evaluation signals.

To protect privacy, we apply an anonymization function A(·) that removes all author- and reviewer-identifying information, including names, affiliations, email addresses, and explicit identity references. In addition, we apply a safety filtering function F (·) to review content to remove toxic or abusive language and personal attacks. The final stored representation of each paper is given by

𝑝˜ = F (A(𝑝)), resulting in a de-identified paper pool

###### P˜ = { 𝑝˜ | 𝑝 ∈ P },

which preserves technical content and review feedback while minimizing exposure to private or harmful information.

- 3.1.2 Method Unit Extraction

Based on the de-identified paper pool P˜, we define an automated extraction procedure that identifies the core methodological contributions of each paper in a structured and reusable form. Formally, we model method unit extraction as a mapping

E : 𝑝˜ → U𝑝 = {𝑢(𝑝1), . . . , 𝑢(𝑝𝐾𝑝)},

where 𝑝˜ ∈ P˜ denotes a single paper and U𝑝 is a small set of method units that capture its essential technical ideas.

As illustrated in Figure 2, the extraction procedure leverages the standardized structure of academic papers and analyzes different sections to collect complementary methodological signals. Let x𝑝 = (intro𝑝, method𝑝, exp𝑝) denote the partition of a paper into its introduction, method, and experiments sections. The introduction is used to identify the high-level research motivation and the precise problem formulation, the method section provides signals about core technical mechanisms such as modeling assumptions, learning objectives, model architectures, and optimization strategies, and the experiments section reflects how these mechanisms are instantiated and evaluated in practice. By jointly aggregating information from these sections, the extractor isolates method units that correspond to the primary algorithmic or modeling contributions of the paper, rather than surface-level experimental details.

We define a method unit 𝑢 ∈ U𝑝 as a self-contained description of how a research problem is formulated or solved, abstracted away from specific implementation choices and experimental configurations. Elements that primarily involve dataset selection, hyperparameter tuning, or engineering-level optimizations are excluded unless they induce substantive changes to the problem formulation, model structure, or learning objective. In practice, most papers yield one or a small number of method units. Each extracted unit is further normalized into structured methodological attributes, including atomic meta-methods, which correspond to indivisible methodological elements, and composition-level patterns, which describe how multiple method units are combined within a single paper.

After extracting method units for all papers, we represent each paper 𝑝 ∈ P˜ by a vector embedding derived from its associated method units. Formally, let

z𝑝 = 𝑔(U𝑝),

where U𝑝 denotes the set of extracted method units for paper 𝑝 and 𝑔(·) is an embedding function that maps a set of method units to a fixed-dimensional representation.

To induce higher-level research patterns, we first apply a nonlinear dimensionality reduction operator

y𝑝 = UMAP(z𝑝),

which projects the high-dimensional embeddings into a lower-dimensional space while preserving local semantic neighborhoods. We then perform density-based clustering on the reduced representations using DBSCAN, yielding a partition

C = {𝐶1, . . . ,𝐶𝑀},

where each cluster 𝐶𝑚 ⊂ P˜ corresponds to a coherent research pattern.

These induced clusters serve as higher-level abstractions over individual papers, capturing recurring methodological structures that are reused across the literature. The resulting research patterns form the basis for subsequent retrieval and composition.

- 3.1.3 Knowledge Graph Construction

Building on the extracted method units, we organize reusable methodological components into a structured knowledge graph that supports systematic method discovery and composition. While individual method units capture isolated algorithmic or modeling ideas, effective research methods in practice typically arise from structured combinations of multiple method units. The knowledge graph provides a unified representation that explicitly encodes canonicalized method units, meta-methods, and their empirically observed composition relations in prior work.

Formally, we define the knowledge graph as a directed graph

###### G = (V, E),

[Figure 3]

- Figure 2 Offline knowledge graph construction in Idea2Story. Academic papers and their associated review artifacts are first anonymized and safety-filtered, then deconstructed into layered methodological representations. These layers capture complementary aspects of a paper, including its core research idea, domain context, high-level story skeleton, and packaging actions. The extracted elements are normalized into atomic method units and meta-methods, which are connected through composition and similarity relations. Reviewer feedback is incorporated as additional signals to refine relations and validate abstractions.

where each node 𝑣 ∈ V corresponds to a canonicalized method unit or a meta-method. Canonicalization groups semantically similar method units across the corpus into shared meta-method abstractions, reducing surface-level variation while preserving core methodological intent. As a result, nodes in the graph represent atomic or minimally indivisible methodological elements that are reused across papers.

Edges in the graph encode composition relations between method units. For a given paper 𝑝 ∈ P˜ with extracted method unit set U𝑝, we add directed edges between pairs of method units (𝑢𝑖, 𝑢𝑗) ∈ U𝑝 × U𝑝 to indicate that they are jointly instantiated as part of the same methodological pipeline. These edges capture empirical evidence of method compatibility observed in prior work, reflecting how different method units are combined in practice rather than hypothetical or manually specified relations.

Aggregating composition relations across the full corpus yields a graph structure that encodes both methodological abstraction and empirical compatibility. In particular, the graph captures two complementary levels of structure: (i) reusable methodological elements represented as canonicalized method units and meta-methods, and (ii) composition constraints induced from co-occurrence statistics in accepted papers. This separation allows Idea2Story to reason about methods at a higher level of abstraction than individual papers, while remaining grounded in observed research practice.

##### 3.2 Online Research Generation.

Given a target research objective, Idea2Story treats method discovery as a graph-based retrieval and composition problem over G. The system retrieves relevant subgraphs and composes compatible method units by following connectivity constraints in the graph, producing candidate research patterns that correspond to structured combinations of method units. These research patterns serve as high-level methodological blueprints that bridge abstract research intent and concrete experimental design, enabling downstream planning, feasibility analysis, and end-to-end paper generation.

- 3.2.1 Research Pattern Retrieval

Given a user-provided research idea expressed in natural language, we formulate research pattern identification as a structured retrieval problem over the knowledge graph G. Let 𝑞 denote the input research idea, and let C = {𝐶1, . . . ,𝐶𝑀} denote the set of research patterns induced from the paper corpus. The goal is to rank patterns in C according to their relevance to 𝑞.

Rather than relying on a single similarity metric, Idea2Story adopts a multi-view retrieval formulation that aggregates complementary signals from different semantic abstractions. Formally, for each research pattern 𝐶𝑚, we compute a relevance score

###### 𝑠(𝐶𝑚 | 𝑞) = ∑︁ 𝑣∈V

𝜆𝑣 𝑠𝑣(𝐶𝑚 | 𝑞),

where V = {idea, domain, paper} indexes the retrieval views, 𝑠𝑣(·) denotes a view-specific scoring function, and 𝜆𝑣 are fixed weighting coefficients that balance the contribution of different views.

Idea-level retrieval. At the idea level, the system retrieves previously observed research ideas that are semantically similar to the input query 𝑞. Let I denote the set of stored research ideas extracted from the corpus, and let simidea(𝑞,𝑖) denote a semantic similarity function between 𝑞 and an idea 𝑖 ∈ I. The idea-level score of a research pattern 𝐶𝑚 is computed by aggregating the similarity scores of ideas associated with the pattern:

𝑠idea(𝐶𝑚 | 𝑞) = max

𝑖∈I(𝐶𝑚)

where I(𝐶𝑚) denotes the set of ideas linked to pattern 𝐶𝑚.

simidea(𝑞,𝑖),

Domain-level retrieval. At the domain level, the system interprets the input idea 𝑞 in terms of its underlying research domains and methodological themes. Let D denote the set of research domains, and let simdomain(𝑞, 𝑑) measure the relevance between 𝑞 and domain 𝑑 ∈ D. The domain-level score of pattern 𝐶𝑚 is computed as

𝑠domain(𝐶𝑚 | 𝑞) = ∑︁

simdomain(𝑞, 𝑑) 𝑤(𝑑,𝐶𝑚),

𝑑∈D(𝐶𝑚)

where D(𝐶𝑚) denotes the domains associated with pattern 𝐶𝑚, and 𝑤(𝑑,𝐶𝑚) captures empirical effectiveness signals derived from the knowledge graph.

Paper-level retrieval. At the paper level, the system retrieves papers whose technical content is semantically aligned with the input idea. Let P(𝐶𝑚) denote the set of papers instantiating pattern 𝐶𝑚. The paper-level score is computed as

𝑠paper(𝐶𝑚 | 𝑞) = max

simpaper(𝑞, 𝑝) · 𝛼(𝑝),

𝑝∈P(𝐶𝑚)

where simpaper(𝑞, 𝑝) measures semantic similarity between 𝑞 and paper 𝑝, and 𝛼(𝑝) denotes a quality-related weight derived from peer review metadata.

The final ranked list of research patterns is obtained by ordering patterns according to their aggregated multi-view relevance scores. Formally, we define

C∗(𝑞) = Rank𝐶𝑚∈C ∑︁

𝜆𝑣 𝑠𝑣(𝐶𝑚 | 𝑞) ,

𝑣∈{idea,domain,paper}

where patterns are sorted in descending order of the aggregated score.

- 3.2.2 Review-Guided Refinement

After candidate research patterns are retrieved, Idea2Story refines them using an explicit LLM-based review loop. In each iteration, a large language model is prompted to act as a reviewer and evaluate the current research pattern along several predefined criteria, including technical soundness, novelty with respect to existing literature, and overall clarity of the problem–method alignment. The reviewer produces both scalar judgments and concrete revision suggestions.

The system then uses this feedback to update the research pattern in a targeted manner. When the review indicates insufficient novelty, the system modifies the pattern by recombining compatible method units or introducing alternative realizations within the same pattern family. When the review identifies issues in feasibility or ambiguity in formulation, the system revises the problem definition or method structure to improve consistency and executability. Each revised pattern is re-submitted to the same review process, forming an explicit generate–review–revise loop.

Case 1: Method Unit Extraction Demo Paper Title: Learning Dynamics of LLM Finetuning Base Problem: Understanding how specific training examples influence model predictions during finetuning is challenging, particularly in large language models.

Solution Pattern: Develop a framework to analyze step-wise influence accumulation among potential responses during finetuning, providing insights into phenomena like hallucination and the squeezing effect in off-policy direct preference optimization.

Story: Reframe the understanding of LLM finetuning through the lens of learning dynamics, offering a unified interpretation of training behaviors and inspiring methods to enhance model alignment and performance.

Application: Improving alignment in large language models, enhancing finetuning strategies for better model performance, diagnosing and mitigating hallucination in AI systems.

- Figure 3 An example of a method unit extracted from an accepted paper, illustrating the separation of the base problem, solution pattern, and higher-level research story.

To prevent uncontrolled drift, only revisions that improve the reviewer scores are retained; otherwise, the system rolls back to the previous version. This process repeats until the reviewer judges the pattern to be sufficiently novel, coherent, and technically plausible, or until further iterations no longer yield improvement. The output of this stage is a refined research pattern that has been iteratively vetted by an LLM-based reviewer and is suitable for downstream validation and paper generation.

- 4 Experiments and Analysis

We evaluate Idea2Story through a set of experiments focusing on its ability to extract reusable methodological structure and to generate high-quality research patterns from ambiguous user input. Our experiments are conducted on a corpus of accepted papers from ICLR and NeurIPS over the past three years, including approximately 13K papers and their associated peer reviews, which serves as the foundation for all subsequent analyses. Based on this corpus, we first analyze the properties of the extracted method units to assess whether Idea2Story captures meaningful and reusable methodological abstractions. We then present qualitative demonstrations of research patterns instantiated as structured research stories, illustrating how the system transforms vague research intent into coherent and methodologically grounded research directions.

##### 4.1 Implementation Details

To further assess the effectiveness of Idea2Story in practical research ideation settings, we conduct additional qualitative experiments on a small set of representative cases. Specifically, we evaluate three user-provided research ideas curated by an external collaborator. For each case, Idea2Story generates research patterns using the GLM-4.7 (Zeng et al., 2025) model as the underlying language backbone. As a baseline, we compare against direct LLM generation, where the same model is prompted to produce a complete research story without explicit pattern modeling or retrieval.

##### 4.2 Case Study: Method Unit Extraction

We present a representative case study to illustrate the behavior of the proposed method unit extraction agent. Case 1 shows an example extracted from an accepted paper, where the system decomposes the full paper into a structured set of methodological elements.

As shown in the example, the extracted method unit explicitly separates the underlying research problem, the core solution pattern, and the resulting research story. The Base Problem describes the core challenge addressed by the paper, namely understanding how individual training examples influence model behavior during finetuning, without depending on specific datasets or implementation details. The Solution Pattern summarizes the central methodological idea as an analysis framework for step-wise influence accumulation, highlighting the key mechanism without binding it to a particular optimization setup or experimental configuration. Importantly, the extracted Story reframes the technical

contribution at a higher level of abstraction, connecting learning dynamics to broader phenomena such as hallucination and alignment in large language models. This abstraction reflects how the method unit goes beyond algorithmic details to capture the conceptual contribution of the paper. Finally, the Application field grounds the method unit by indicating downstream research and system-level implications, without enumerating task-specific benchmarks.

This example demonstrates that the extraction agent isolates reusable methodological structure while filtering out implementation-level details. By representing the paper as a coherent method unit rather than a collection of experimental components, Idea2Story enables subsequent reuse, comparison, and composition of methodological ideas across papers.

##### 4.3 Knowledge Graph Analysis

We analyze the structure of the constructed knowledge graph to understand how extracted method units are distributed across papers and research domains. As illustrated in Figure 2, the graph exhibits a clear hub-and-spoke structure, where a small number of high-frequency domains connect to a large number of papers and research patterns. This reflects the uneven distribution of research activity across domains, while also highlighting domains that function as central hubs for methodological reuse. Importantly, many research patterns are observed to connect multiple domains simultaneously, indicating that the extracted method units often capture methodological abstractions that generalize beyond a single application area. In contrast, paper-level nodes are typically associated with a single domain, whereas pattern-level nodes frequently act as bridges between otherwise weakly connected domains. This structural separation suggests that the knowledge graph encodes two distinct levels of organization—instance-level Figure 4 Visualization of the knowledge graph substructure

[Figure 4]

induced by high-frequency research domains.

research artifacts and reusable methodological abstractions—enabling Idea2Story to retrieve and compose research patterns at a higher level of abstraction rather than relying on domain-specific or paper-specific similarity alone.

##### 4.4 Qualitative Comparison of Generated Research Patterns

We further compare the quality of research patterns generated by Idea2Story and a direct LLM baseline. Both systems start from the same underspecified user input and produce structured research proposals, enabling a controlled comparison of how different generation mechanisms transform vague research intent into concrete research patterns.

Table 1 presents a side-by-side comparison of representative outputs along multiple dimensions, including problem formulation, methodological structure, and innovation claims. Rather than evaluating surface-level writing quality, the comparison focuses on the resulting research patterns as methodological blueprints—i.e., how the generated ideas frame the research problem, identify gaps in prior work, and organize methodological components into a coherent approach. As shown in the table, Idea2Story tends to induce higher-level problem reformulation, transforming intent understanding from a fixed classification task into a dynamic structural reasoning process. The resulting research pattern emphasizes generative refinement, structural priors, and evolving representations. In contrast, the direct LLM baseline largely operates within a conventional task formulation, proposing a stronger system through the integration of additional components such as context modeling and hierarchical objectives.

To reduce evaluation bias, the generated research stories from both approaches are subsequently assessed by an independent large language model (Gemini 3 Pro) (Team et al., 2025), which is not involved in either generation process. The evaluator is instructed to compare the outputs in terms of novelty, methodological substance, and overall research quality, without access to the generation method used. Across all evaluated cases, the externally evaluated results consistently favor the outputs generated by Idea2Story. In particular, the research stories produced by direct LLM generation tend to remain at a high level of abstraction, with less concrete methodological grounding and reliance on

Aspect Idea2Story Generated (IntentDiff) LLM Direct Generated (EcoIntent) Title IntentDiff: Reframing E-commerce Intent Classifi-

EcoIntent: A Context-Aware Multi-Granularity Agent for E-commerce Intent Understanding via Hierarchical Contrastive Learning

cation via Structural Evolution and Context-Aware Diffusion

Abstract Focus Reinterprets intent classification as a structural evolution process rather than static text classification. The approach leverages a diffusion-based framework to iteratively refine noisy query representations into precise intent labels, integrates product graph embeddings to ground predictions in e-commerce context, and introduces a discrete, context-aware tokenizer to handle long-tail domain vocabulary.

Targets improved intent classification performance by integrating heterogeneous behavioral context and hierarchical product knowledge. A dual-stream architecture aligns semantic representations with user interaction history, and hierarchical contrastive learning enforces consistency across fine- and coarse-grained intent categories.

Problem Definition Reframes e-commerce intent classification from static text prediction to dynamic structural reasoning. User queries are short, ambiguous, and heavily dependent on implicit catalog structure, which fixed-label classification fails to capture. Intent understanding is modeled as an evolving process under structural constraints.

Formulates intent understanding as a conventional multi-class classification problem, where the input is a query augmented with session context and the output is an intent label from a predefined set. The main challenge is semantic sparsity caused by short and ambiguous queries.

Core Research Gap Existing intent classification methods treat queries in isolation and ignore domain-specific structural priors in e-commerce. They fail to exploit rich relationships between products and attributes, and standard vocabularies struggle with long-tail, domain-specific terminology. No prior work unifies diffusion-based refinement with structural graph embeddings for intent disambiguation.

Prior work suffers from (1) context isolation, where behavioral signals such as clicks are underutilized, and (2) a flat-label assumption that ignores the hierarchical nature of e-commerce taxonomies, leading to inconsistent predictions for fine-grained, long-tail intents.

Method Skeleton A diffusion-based classifier that iteratively denoises intent representations; a context-aware discrete tokenizer based on a VQ-VAE variant to encode diverse e-commerce queries; and integration of pretrained product graph embeddings as structural priors during the denoising process.

A dual-stream discriminative architecture consisting of a BERT-based text encoder, a lightweight GNN for aggregating behavioral interaction graphs, and a prediction head trained with hierarchical contrastive learning; parameter-efficient adaptation via LoRA.

Innovation Claims (1) Reformulates intent classification as a diffusionbased dynamic refinement process; (2) Introduces discrete, context-aware intent tokenization to better handle long-tail domain vocabulary; (3) Enhances intent reasoning by incorporating product graph structural embeddings.

(1) Contextualized intent modeling via joint reasoning over text and behavioral graphs; (2) Hierarchical contrastive learning leveraging product taxonomies; (3) Parameter-efficient system design achieving strong performance at reduced computational cost.

Table 1 Comparison of research patterns generated by Idea2Story and a direct LLM baseline, both starting from the same underspecified user input: “I want to build an e-commerce agent that can better understand user intent.” The table contrasts how different generation mechanisms transform the same vague research intent into concrete research patterns.

relatively standard techniques. In contrast, Idea2Story-generated research patterns exhibit clearer problem framing, more specific methodological structures, and stronger signals of novelty.

#### 5 Future Work

While Idea2Story focuses on grounding vague research intent into structured and high-quality research patterns, an important direction for future work is to extend this framework toward a fully closed-loop research generation pipeline. A promising extension is the integration of experiment-driven agents that can instantiate, validate, and iteratively refine generated research patterns through empirical feedback, including automated experimental design, dataset selection, and

preliminary execution. Experimental outcomes can then serve as additional signals to refine the instantiated research stories, forming a feedback loop between method design and empirical validation. Beyond experimentation, future work may further explore how refined research patterns can be systematically translated into complete paper drafts, covering method descriptions, experimental results, and discussion sections. By grounding paper generation in empirically validated research patterns, such a system could move beyond surface-level text generation and provide more faithful, end-to-end support for executable and publishable scientific discovery.

#### 6 Conclusion

We presented Idea2Story, a pre-computation–driven framework for autonomous scientific discovery that shifts literature understanding from runtime reasoning to offline knowledge structuring. By explicitly extracting reusable method units and organizing them into a continuously updated knowledge graph, Idea2Story enables research agents to reason over stable research patterns rather than repeatedly processing raw papers. Our qualitative analyses and comparative studies show that this design leads to research patterns with clearer problem reformulation, stronger methodological structure, and higher conceptual novelty than direct LLM generation. These results highlight the importance of explicit pattern modeling as a foundation for scalable and reliable autonomous research. Looking ahead, integrating Idea2Story with experimental agents to close the loop from abstract research patterns to validated empirical results represents a promising direction toward fully autonomous and trustworthy scientific discovery.

#### References

Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya Goyal, Danqi Chen, and Tianyu Gao. Litsearch: A retrieval benchmark for scientific literature search. arXiv preprint arXiv:2407.18940, 2024.

Rauno Arike, Elizabeth Donoway, Henning Bartsch, and Marius Hobbhahn. Technical report: Evaluating goal drift in language model agents, 2025. https://arxiv.org/abs/2505.02709.

Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. ResearchAgent: Iterative research idea generation over scientific literature with large language models. Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6709–6738, 2025.

Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Yuzhi Zhang, Linfeng Zhang, Siheng Chen, et al. Scimaster: Towards general-purpose scientific ai agents, part i. x-master as foundation: Can we lead on humanity’s last exam? arXiv preprint arXiv:2507.05241, 2025.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander Mądry. MLE-bench: Evaluating machine learning agents on machine learning engineering, 2024.

Cristina Cornelio, Sanjeeb Dash, Vernon Austel, Tyler R Josephson, Joao Goncalves, Kenneth L Clarkson, Nimrod Megiddo, Bachir El Khadir, and Lior Horesh. Combining data and theory for derivable scientific discovery with AI-descartes. Nature Communications, 14(1):1777, 2023.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an AI co-scientist, February 2025. arXiv:2502.18864 [cs].

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. MetaGPT: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.

Tu Hu, Ronghao Chen, Shuo Zhang, Jianghao Yin, Mou Xiao Feng, Jingping Liu, Shaolei Zhang, Wenqi Jiang, Yuqi Fang, Sen Hu, Huacan Wang, and Yi Xu. Controlled self-evolution for algorithmic code optimization. arXiv preprint arXiv:2601.07348, 2026.

Yiqiao Jin, Qinlin Zhao, Yiyang Wang, Hao Chen, Kaijie Zhu, Yijia Xiao, and Jindong Wang. AgentReview: Exploring peer review dynamics with LLM agents. arXiv preprint arXiv:2406.12708, 2024.

Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Yi Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Scott Smith, Yian Yin, et al. Can large language models provide useful feedback on research papers? a large-scale empirical analysis. NEJM AI, 1(8):AIoa2400196, 2024.

Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Hongzhang Liu, Ronghao Chen, Yangfan He, Daxin Jiang, Binxing Jiao, Chen Hu, and Huacan Wang. SE-Agent: Self-evolution trajectory optimization in multi-step reasoning with LLM-based agents. arXiv preprint arXiv:2508.02085, 2025.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

Ziming Luo, Zonglin Yang, Zexin Xu, Wei Yang, and Xinya Du. Llm4sr: A survey on large language models for scientific research. arXiv preprint arXiv:2501.04306, 2025.

Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C. Landsness, Daniel L. Barabasi, Siddharth Narayanan, Nicky Evans, Shriya Reddy, et al. Kosmos: An AI scientist for autonomous discovery. arXiv preprint arXiv:2511.02824, 2025.

Gaurav Sahu, Hugo Larochelle, Laurent Charlin, and Christopher Pal. ReviewerToo: Should AI join the program committee? arXiv preprint arXiv:2510.08867, 2025.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants. arXiv preprint arXiv:2501.04227, 2025a.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. Agent Laboratory: Using LLM agents as research assistants. arXiv preprint arXiv:2501.04227, 2025b.

Hyungyu Shin, Jihoon Kim, Hwaran Lee, Kyohoon Jin, and Seung won Hwang. Mind the blind spots: A focus-level evaluation framework for LLM reviews. arXiv preprint arXiv:2502.17086, 2025.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. arXiv preprint arXiv:2409.04109, 2024.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Naitian Thakkar, Yilun Xu, Shikhar Varma, Ke Wu, Zhaofeng Wang, Dawn Song, Huazhe Xu, Trevor Darrell, Shanghang Wang, and Joseph E Gonzalez. Can llm feedback enhance review quality? a randomized study of 20k reviews at iclr 2025. arXiv preprint arXiv:2504.09737, 2025.

Yian Tian, Lijun Wu, Kevin Liu, Zecheng Zhang, Xun Liang, et al. SciCode: A research coding benchmark for scientific discovery. arXiv preprint arXiv:2407.13168, 2024.

Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, 2023.

Huacan Wang, Ziyi Ni, Shuo Zhang, Shuo Lu, Sen Hu, Ziyang He, Chen Hu, Jiaye Lin, Yifu Guo, Ronghao Chen, et al. Repomaster: Autonomous exploration and understanding of github repositories for complex task solving. arXiv preprint arXiv:2505.21577, 2025a.

Xingyao Wang, Bowei Yang, Yiqiao Jin, Jiaqi Li, Yijia Xiao, Wenghua Lin, Xiaotian Cheng, Ruicheng Zheng, Huieu Le, Maosong Cao, et al. OpenHands: An open platform for AI software developers as generalist agents. arXiv preprint arXiv:2407.16741, 2025b.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.

Yanjie Xu, Xin Liu, X Cao, C Huang, E Liu, S Qian, X Liu, Y Wu, F Dong, CW Qiu, et al. Artificial intelligence: A powerful paradigm for scientific research. The Innovation, 2(4):100179, 2021.

Zhijian Xu, Yilun Zhao, Manasi Patwardhan, Lovekesh Vig, and Arman Cohan. Can llms identify critical limitations within scientific research? a systematic evaluation on ai research papers. arXiv preprint arXiv:2507.02694, 2025.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search, 2025. https://arxiv.org/abs/2504.08066.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793, 2024.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Haoxuan Zhang, Ruochi Li, Yang Zhang, Ting Xiao, Jiangping Chen, Junhua Ding, and Haihua Chen. The evolving role of large language models in scientific innovation: Evaluator, collaborator, and scientist. arXiv preprint arXiv:2507.11810, 2025a.

Ming Zhang, Kexin Tan, Yueyuan Huang, Yujiong Shen, Chunchun Ma, Li Ju, Xinran Zhang, Yuhui Wang, Wenqing Jing, Jingyi Deng, et al. Opennovelty: An llm-powered agentic system for verifiable scholarly novelty assessment. arXiv preprint arXiv:2601.01576, 2026.

Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, and Daphne Ippolito. NoveltyBench: Evaluating creativity and diversity in language models. arXiv preprint arXiv:2504.05228, 2025b.

### Appendix

