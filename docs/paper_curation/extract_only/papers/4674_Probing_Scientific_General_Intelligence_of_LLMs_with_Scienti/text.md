[Figure 1]

[Figure 2]

# Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows

Shanghai Artificial Intelligence Laboratory

S Page https://InternScience.github.io/SGI-Page/

Code https://github.com/InternScience/SGI-Bench

Data https://huggingface.co/collections/InternScience/sgi-bench

Abstract: Despite advances in scientific AI, a coherent framework for Scientific General Intelligence (SGI)—the ability to autonomously conceive, investigate, and reason across scientific domains—remains lacking. We present an operational SGI definition grounded in the Practical Inquiry Model (PIM: Deliberation, Conception, Action, Perception) and operationalize it via four scientist-aligned tasks: deep research, idea generation, dry/wet experiments, and experimental reasoning. SGI-Bench comprises over 1,000 expert-curated, cross-disciplinary samples inspired by Science’s 125 Big Questions, enabling systematic evaluation of state-of-the-art LLMs. Results reveal gaps: low exact match (10–20%) in deep research despite step-level alignment; ideas lacking feasibility and detail; high code executability but low execution result accuracy in dry experiments; low sequence fidelity in wet protocols; and persistent multimodal comparative-reasoning challenges. We further introduce Test-Time Reinforcement Learning (TTRL), which optimizes retrieval-augmented novelty rewards at inference, enhancing hypothesis novelty without reference answer. Together, our PIM-grounded definition, workflow-centric benchmark, and empirical insights establish a foundation for AI systems that genuinely participate in scientific discovery.

## arXiv:2512.16969v1[cs.AI]18Dec2025

[Figure 3]

[Figure 4]

Experimental Reasoning

Scientific Deep Research

Multimodal Perception Data Analysis

Literature Inquiry Meta-analysis Web Search Calculation

[Figure 5]

[Figure 6]

[Figure 7]

Perception

[Figure 8]

Deliberation

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Action

[Figure 13]

Protocol Development Code Generation

Conception

[Figure 14]

[Figure 15]

Existing Work Analysis Method design

Idea Generation Dry/Wet Experiment

Figure 1 | Scientific General Intelligence (SGI) We define SGI as an AI that can autonomously navigate the complete, iterative cycle of scientific inquiry with the versatility and proficiency of a human scientist. The teaser illustrates the Practical Inquiry Model’s four quadrants—Deliberation (synthesis and critical evaluation of knowledge), Conception (idea generation), Action (experimental execution), and Perception (interpretation)—and how SGI-Bench operationalizes them through four task categories and an agent-based evaluation paradigm, together providing a principle-grounded, measurable framework for assessing scientific intelligence.

#### Contents

- 1 Introduction 4
- 2 Scientific General Intelligence: Concept and Operational Definition 6

- 2.1 Task Definition in Scientific Workflow 7

- 2.1.1 Scientific Deep Research 7
- 2.1.2 Idea Generation 10
- 2.1.3 Dry/Wet Experiment 11
- 2.1.4 Experimental Reasoning 14

- 2.2 Multi-Dimensional Metrics 15

- 2.2.1 Metrics of Scientific Deep Research 15
- 2.2.2 Metrics of Idea Generation 16
- 2.2.3 Metrics of Dry/Wet Experiment 17
- 2.2.4 Metrics of Experimental Reasoning 19

- 2.3 Scientist-Aligned Data Construction 19
- 2.4 Data Distribution 20

- 3 SGIEvalAgent: Agentic Evaluation Framework 21

- 3.1 Question Selection 22
- 3.2 Metric Customization 22
- 3.3 Inference and Evaluation 23
- 3.4 Report Generation 24

- 4 Evaluation Results 25

- 4.1 Evaluation Setup 25
- 4.2 Overview 25
- 4.3 Scientific Deep Research 25
- 4.4 Idea Generation 28
- 4.5 Dry/Wet Experiment 30

- 4.5.1 Dry Experiment 31
- 4.5.2 Wet Experiment 33

- 4.6 Experimental Reasoning 35

- 5 Analysis 37

- 5.1 Test Time Reinforcement Learning 37

- 5.1.1 Methodology 38
- 5.1.2 Experimental Setup 39
- 5.1.3 Experimental Results 40
- 5.1.4 Case Study of TTRL 40

- 5.2 Agent Tool Integrated Reasoning 41

- 5.2.1 Retrieve–Browse Loop Analysis 41
- 5.2.2 Tool Efficiency Analysis 41
- 5.2.3 Reasoning Cost Analysis 42

- 5.3 SGIEvalAgent 42

- 5.3.1 User-customized Metric 42
- 5.3.2 Automated Evaluation Report 43

- 6 Challenges and Future Directions 43

- 6.1 Fragmentation Across the Four Quadrants of SGI 44

2

- 6.2 Implications from Test-Time RL and Tool-Integrated Reasoning 45
- 6.3 Future Directions Toward Scientific General Intelligence 46
- 6.4 Limitations 47

###### 7 Related Work 48

- 7.1 Benchmarks in Different Disciplines 48
- 7.2 Benchmarks for Different Scientific Tasks 48

###### 8 Conclusion 49 References 51 A Appendix 58

- A.1 Authors 58
- A.2 Disciplines and Research Directions Overview 59
- A.3 Cases 67

- A.3.1 Scientific Deep Research 67
- A.3.2 Idea Generation 77
- A.3.3 Dry Experiment 102
- A.3.4 Wet Experiment 117
- A.3.5 Experimental Reasoning 125

- A.4 Supplementary Evaluation Results 150

##### 1. Introduction

Large language models (LLMs) [1, 2, 3, 4, 5] are achieving and even exceeding human-level performance on a diverse array of tasks, spanning multidisciplinary knowledge understanding, mathematical reasoning, and programming. This rapid progress has ignited a vibrant debate: some view these models as early signals of artificial general intelligence (AGI) [6, 7], whereas others dismiss them as mere “stochastic parrots [8],” fundamentally constrained by their training data. As these models evolve, the frontier of AGI research is shifting towards the most complex and structured of human endeavors: scientific inquiry [9]. We argue that demonstrating genuine scientific general intelligence (SGI) represents a critical leap toward AGI, serving as a definitive testbed for advanced reasoning, planning, and knowledge creation capabilities. However, much like AGI, the concept of SGI remains frustratingly nebulous, often acting as a moving goalpost that hinders clear evaluation and progress.

This paper aims to provide a comprehensive, quantifiable framework to cut through this ambiguity, starting with a concrete definition grounded in established theory:

"SGI is an AI that can autonomously navigate the complete, iterative cycle of scientific inquiry with the versatility and proficiency of a human scientist"

To operationalize this definition, we ground our approach in the Practical Inquiry Model [10, 11], a theoretical framework that deconstructs the scientific process into a cycle of four core cognitive activities. This model provides a taxonomic map of scientific cognition through four distinct, interdependent quadrants (Figure 1): Deliberation (the search, synthesis, and critical evaluation of knowledge), Conception (the generation of ideas), Action (the practical implementation via experiments), and Perception (the awareness and interpretation of results). An AI exhibiting true SGI must possess robust capabilities across this entire spectrum. This four-quadrant framework provides a conceptual taxonomy of scientific cognition and forms the foundation for an operational definition of SGI—one that specifies what kinds of planning, knowledge creation and reasoning an AI must demonstrate to qualify as scientifically intelligent. Translating this operational definition into measurable criteria requires examining how current evaluations of AI intelligence align with, or deviate from, this framework. Identifying these gaps is essential for clarifying what existing assessments capture and what they overlook in defining Scientific General Intelligence.

Grounded in this four-quadrant definition of SGI, we examine how existing benchmarks operationalize scientific reasoning. Most current evaluations capture only fragments of the SGI spectrum. For instance, MMLU [12] and SuperGPQA [13] focus on multidisciplinary knowledge understanding—corresponding mainly to the Deliberation quadrant—while GAIA [14] emphasizes procedural tool use aligned with Action. HLE [15] further raises difficulty through complex reasoning, yet still isolates inquiry stages without integrating the practical or interpretive cycles that characterize real scientific investigation. Collectively, these benchmarks present a fragmented view of scientific intelligence. Their disciplinary scope remains narrow, their challenges seldom reach expert-level reasoning, and—most crucially—they frame inquiry as a static, closed-domain question-answering task. This abstraction neglects the creative, procedural, and self-corrective dimensions central to SGI, meaning that what is currently measured as “scientific ability” reflects only a limited slice of true Scientific General Intelligence.

Thus, to concretize the proposed definition of Scientific General Intelligence (SGI), we develop SGIBench: A Scientific Intelligence Benchmark for LLMs via Scientist-Aligned Workflows. Rather than serving as yet another performance benchmark, SGI-Bench functions as an operational instantiation of the SGI framework, quantitatively evaluating LLMs across the full spectrum of scientific cognition defined by the Practical Inquiry Model. By design, SGI-Bench is comprehensive in its disciplinary breadth, challenging in its difficulty, and unique in its explicit coverage of all four capabilities central

to our definition of SGI. The benchmark structure is therefore organized into four corresponding task categories:

- • Scientific Deep Research (Deliberation): This task evaluates models’ ability to perform iterative, multi-step reasoning over complex scientific content.
- • Idea Generation (Conception): This task assesses creativity and methodological planning by asking models to generate novel hypotheses or experimental designs.
- • Dry/Wet Experiment (Action): This task evaluates the ability to plan and execute computational (dry) or laboratory-style (wet) experiments.
- • Experimental Reasoning (Perception): This task requires models to analyze experimental results, interpret data trends, and identify meaningful conclusions.

Building upon our theoretical framework, the construction of SGI-Bench operationalizes the proposed definition of Scientific General Intelligence (SGI). We began with foundational topics drawn from Science’s 125 Big Questions for the 21st Century [16], spanning ten major disciplinary areas. Through multi-round collaborations with domain experts, we identified high-impact research problems and curated raw source materials from leading journals such as Nature, Science, and Cell. Together with PhD-level researchers, we implemented a multi-stage quality control pipeline involving human annotation, model-based verification, and rule-based consistency checks. The resulting benchmark comprises over 1,000 expert-curated samples that concretely instantiate the reasoning, creativity, and experimental competencies central to our definition of SGI.

To evaluate performance across these four dimensions, we found that conventional “LLM-as-ajudge” [17] paradigms are insufficient to handle the diverse and specialized metrics required by SGI assessment. To address this, we developed an agent-based evaluation framework following an Agent-as-a-judge [18] paradigm. Equipped with tools such as a web search interface, Python interpreter, file reader, PDF parser, and discipline-specific metric functions, this framework ensures rigor, scalability, and transparency. It operates through four interdependent stages—Question Selection, Metric Customization, Prediction & Evaluation, and Report Generation—each coordinated by specialized agents aligned with different aspects of scientific inquiry.

Applying SGI-Bench to a wide spectrum of state-of-the-art LLMs reveals a unified picture: while modern models achieve pockets of success, they fall far short of the integrated reasoning required for scientific intelligence.

- • In deep scientific research, models can retrieve relevant knowledge but struggle to perform quantitative reasoning or integrate multi-source evidence; exact-match accuracy remains below 20% and often collapses on numerical or mechanistic inference.
- • In idea generation, models show substantial deficits in realization. This manifests in underspecified implementation steps and frequent proposals that lack actionable detail or fail basic feasibility checks.
- • In dry experiments, even strong models fail on numerical integration, simulation fidelity, and scientific code correctness, revealing a gap between syntactic code fluency and scientific computational reasoning.
- • In wet experiments, workflow planning shows low sequence similarity and error-prone parameter selection, with models frequently omitting steps, misordering actions, or collapsing multi-branch experimental logic.
- • In multimodal experimental reasoning, models perform better on causal and perceptual reasoning but remain weak in comparative reasoning and across domains such as materials science and earth systems.
- • Across tasks, closed-source models demonstrate only a marginal performance advantage over

open-source models. Even the best closed-source system achieves an SGI-Score of around 30/100, reflecting that current AI models possess relatively low capability in multi-task scientific research workflows, and remain far from proficient for integrated, real-world scientific inquiry.

Collectively, these findings demonstrate that current LLMs instantiate only isolated fragments of scientific cognition. They remain constrained by their linguistic priors, lacking the numerical robustness, procedural discipline, multimodal grounding, and self-corrective reasoning loops essential for scientific discovery.

Because genuine scientific inquiry is inherently open-ended and adaptive, we further explore how SGI may emerge under test-time learning dynamics. Preliminary experiments using test-time scaling [19] and reinforcement learning [20] suggest that models can enhance hypothesis formation and reasoning through minimal unlabeled feedback. This adaptive improvement provides empirical support for viewing Scientific General Intelligence not as a static property, but as a dynamic capacity that can evolve through iterative, self-reflective reasoning cycles.

In summary, this work provides a principle-grounded definition of Scientific General Intelligence (SGI) and a corresponding framework for its empirical study. By formalizing the cognitive cycle of scientific inquiry and operationalizing it through SGI-Bench, we clarify what it means for an AI to exhibit scientific intelligence in both theory and practice. While not a final answer, this definition establishes a concrete path for future research—linking conceptual understanding with measurable progress toward AI systems capable of genuine scientific reasoning and discovery.

##### 2. Scientific General Intelligence: Concept and Operational Definition

Scientific General Intelligence (SGI) refers to an AI system capable of engaging in the full cycle of scientific inquiry with autonomy, versatility, and methodological rigor. Unlike systems that excel at isolated reasoning tasks, an SGI-capable model must integrate knowledge retrieval, idea formation, action execution, and evidence-based interpretation into a coherent, iterative workflow.

To formalize this notion, we characterize scientific cognition through four interdependent stages: Deliberation (evidence search, synthesis, and critical assessment), Conception (generation of hypotheses and ideas), Action (implementation of experiments or simulations), and Perception (interpretation of empirical results).

Grounded in this framework, we provide an operational definition: an AI system exhibits SGI if it can (1) retrieve, synthesize, and critically evaluate knowledge; (2) generate scientifically grounded and novel ideas; (3) plan and execute experimental procedures; (4) interpret empirical outcomes with causal and contextual awareness.

This definition highlights a central limitation in existing benchmarks [12, 13, 14, 15]: most evaluate factual recall or single-step reasoning, but few examine the structured, long-horizon workflows that constitute real scientific inquiry.

Building on the operational definition of SGI established in the previous section, we introduce SGIBench (Scientific Intelligence Benchmark for LLMs via Scientist-Aligned Workflows) — a benchmark designed to empirically evaluate the extent to which large language models (LLMs), vision-language models (VLMs), and agent-based systems exhibit the cognitive and procedural abilities required for scientific discovery.

SGI-Bench systematically measures AI performance across 10 core scientific domains — astronomy, chemistry, earth science, energy, information science, life science, materials science, neuroscience, physics and math — providing a panoramic view of how AI systems engage with scientific reasoning across disciplines. Its task design draws inspiration from the seminal article 125 Questions: Exploration

and Discovery [16] published in Science, ensuring both disciplinary breadth and societal relevance.

At the heart of SGI-Bench lies the principle of scientist alignment—the commitment to evaluating models under conditions that authentically mirror real scientific workflows. This concept manifests in several ways:

- • The task designs closely mirror the real-world research scenarios encountered by scientists in their work, ensuring that each task is intrinsically tied to the scientific discovery process.
- • The raw materials used in task construction are sourced directly from scientists, ensuring the authenticity and relevance of the content.
- • Scientists have been closely involved in the process of constructing the benchmark, with a scientist-in-the-loop approach, ensuring the tasks reflect the nuances of actual scientific workflows.
- • The final evaluation scores are aligned with the checklist based on the needs of real scientific research scenarios from scientists, which ensures that the assessments genuinely reflect the scientific utility of the models.

SGI-Bench departs from conventional benchmarks that emphasize factual recall or single-turn reasoning. Instead, it operationalizes the long-horizon workflow of scientific discovery into four interdependent stages: literature review(Deliberation), methodology design(conception), experiment implementation(Action), and experimental analysis(Perception). These stages correspond to fundamental capabilities required of AI systems: information integration and understanding(Scientific Deep Research), design and planning(Idea Generation), experimental execution(Dry/Wet Experiment), and reasoning-based interpretation(Experimental Reasoning). Together, they form a unified framework that measures not only what models know but how they think, plan, and adapt in pursuit of new knowledge.

###### 2.1. Task Definition in Scientific Workflow

###### 2.1.1. Scientific Deep Research

Scientific deep research refers to a thorough and comprehensive investigation of a specific scientific topic, combining elements of both AI-driven deep research [21, 22, 23] and scientific meta-analysis [24, 25]. This task typically involves multi-step reasoning, web searches, document retrieval, and data analysis [26, 27, 28]. Drawing inspiration from AI’s deep research, which often relies on multihop searches to gather diverse information across multiple sources [29], it also incorporates the methodology of meta-analysis from the scientific community. Meta-analysis, a rigorous form of scientific research, synthesizes existing literature to derive precise, data-driven conclusions and extract quantitative insights from a large body of studies. Unlike general deep research, which may focus on qualitative understanding, meta-analysis centers on aggregating and analyzing data to produce statistically significant results. By combining the multi-hop search nature of AI’s deep research with the systematic, evidence-based approach of meta-analysis, this task ensures results that are both scientifically precise and meaningful. The ability to perform scientific deep research is crucial for advancing scientific knowledge, as it enables AI models to replicate the process of reviewing, synthesizing, and analyzing existing research to formulate new, data-driven hypotheses. [30, 31]

Deep Research comprises multiple forms including literature inquiry [32], report-style reasoning [33] and so on. In this benchmark, we focus on literature-inquiry–centric deep research, where the model identifies and integrates relevant scientific knowledge from provided sources. This process often involves unit verification, quantitative interpretation, and causal assessment—abilities fundamental to scientific reasoning and still challenging for current AI systems. By constraining the task to literature

###### A. 125 Cutting-Edge Scientific Questions

B. Scientific Deep Research C. Idea Generation

Related Work Limitations …

[Figure 28]

[Figure 29]

Question

What is the universe made of?

Greenhouse effect's max temperature?

[Figure 30]

Step N

Structured idea

[Figure 31]

[Figure 32]

Meta-analysis Web Search

Implement Graph Metric

Unified Theory of Electronic Systems?

The limits of computer learning?

[Figure 33]

[Figure 34]

…

Selectively interrupting immune responses?

###### …

Step 1 Step 2 Final Answer

Data Expectation

D. Dry Experiment E. Wet Experiment F. Experimental Reasoning

Experimental Procedure Experimental Multimodal Data

###### Background Data Incomplete Code

Background Action Pool

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

def mass_energy(mass_kg): """ Calculate energy from mass using E = mc^2. Args:

| | |
|---|---|
| | |

Completecode

Input Output

mass_kg (float): mass in kilograms. Returns:

float: energy in joules.

""" c = 299_792_458 energy = mass_kg * c**2 return energy

Reasoning: <think>From the image…</think>

Input Output

[Figure 40]

[Figure 41]

[Figure 42]

…

Conclusion: 5d vacancy (+1.45|e| Bader charge) lowers C–H barrier to 0.56 eV via …

…

Python interpreter Output

G. Data Construction Pipeline

Quality Inspection Feedback

[Figure 43]

Data Cleaning

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Scientists from different fields Seed Questions

125Scientific

Annotators

[Figure 48]

Questions

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Scientific Materials

Difficulty Filtering

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Annotation Results

Scientists from different fields Annotators

[Figure 67]

Annotator A Annotator B

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Task Definition Annotation Requirements

- 1. Task Definition
- 2. Seed Questions
- 3. Annotation Requirements
- 4. Materials

- 1. Task Definition
- 2. Annotation Requirements
- 3. Quality Checklist
- 4. Draft Question

[Figure 72]

[Figure 73]

[Figure 74]

Seed Questions

Quality Inspection Checklist

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Materials

[Figure 79]

[Figure 80]

Draft Question

Refinement

- Figure 2 | SGI-Bench Workflow Pipeline: The end-to-end four-stage framework (Deliberation, Conception, Action, Perception) that operationalizes scientific discovery, mapping tasks to capabilities and aligning evaluation with scientist practice.

inquiry rather than broader report-generation settings, we ensure greater reproducibility and more reliable evaluation, while still probing a core component of scientific inquiry.

In order to capture the diversity of real-world scientific inquiries, we divide the task of scientific deep research into four representative types: data, properties, micro-experiments, and macro-experiments, as illustrated in Table 1. This division reflects the major types of questions scientists often confront, ranging from data-centric queries to property characterization, and from small-scale controlled experiments to large-scale natural events. By organizing the task in this way, the benchmark ensures that AI systems are evaluated across the breadth of literature review and data-driven investigation.

In real-world scientific workflows, deep research corresponds to the literature review stage. During this stage, scientists investigate existing studies, gather data, and analyze findings to understand the current state of knowledge and identify knowledge gaps that require further investigation.

###### Table 1 | Scientific Deep Research Types: Four representative categories of inquiry targets and their roles in the scientific workflow.

###### Type Core Description Role in Scientific Workflow

Data Focused on retrieving or analyzing structured datasets, such as event counts, statistical summaries, or dataset-specific attributes.

Supports quantitative literature review and provides a foundation for identifying trends or anomalies.

Property Concerned with identifying or inferring material, molecular, or system properties, often requiring interpretation of experimental results or theoretical knowledge.

Bridges literature review with methodology design by clarifying key parameters.

Micro-experiment Small-scale controlled experiments, often involving chemical reactions, physical transformations, or laboratory processes under specific conditions.

Provides simulated reasoning over experimental procedures and outcomes.

Macro-experiment Large-scale or natural experiments, such as astronomical events, climate observations, or geophysical phenomena.

Extends literature review to global or long-term observations, anchoring hypotheses in real-world contexts.

Task Definition of Scientific Deep Research Task Input

- • Background (B): A detailed background of the research topic, including the scientific field and subfields, to avoid ambiguities in terminology.
- • Constraints (C): Constraints such as experimental settings, scientific assumptions, and data sources that frame the problem appropriately.
- • Data (D): Any experimental or empirical data directly mentioned in the task, which might be either explicitly provided or inferred.
- • Question (Q): A specific, focused question that the task aims to address, such as determining a particular quantity or its variation over time.
- • Response Requirements (R): Specifications for the answer, including the required units and whether the answer should be an integer or a decimal with a specified number of decimal places.

###### Task Output

- • Steps (S): A detailed, step-by-step approach that the system uses to retrieve and process data or perform reasoning.
- • Answer (A): A precise numerical or string-based response, such as a specific value or a phrase.

###### Task Formulation

S, A = LLM/Agent(B, C, D, Q, R)

- Figure 3 | Scientific Deep Research Task: Inputs, outputs, and formulation for literature-driven quantitative inquiry combining multi-step reasoning and meta-analysis.

###### 2.1.2. Idea Generation

Idea generation is a critical component of the scientific process, corresponding to the stage of research methodology design. At this stage, researchers synthesize existing knowledge, engage in associative and creative thinking, and propose new approaches to address current challenges. It embodies the creative essence of scientific inquiry and shapes the direction and potential impact of subsequent research.

In real-world scientific workflows, idea generation typically occurs after researchers have completed a thorough literature review. They integrate prior findings, identify limitations or knowledge gaps, and use creative reasoning to formulate new hypotheses, methods, or frameworks aimed at overcoming these shortcomings. In this sense, idea generation serves as the crucial link between literature understanding and methodological innovation.

However, because idea generation is an open-ended and highly creative task, its evaluation is inherently challenging. In principle, scientific ideas span a wide spectrum from high-level hypotheses to fully specified methodological plans [34, 35, 36]. Evaluating the quality of open-ended hypotheses—those with substantial conceptual freedom and without explicit implementation structure—requires extensive human expert review to achieve even a modest degree of inter-rater reliability and public defensibility. Such large-scale expert adjudication is beyond the practical scope of this version of the benchmark.

Consequently, our current Idea Generation evaluation focuses on the methodological-design component of an idea—i.e., how a proposed approach is operationalized through data usage, step-by-step procedures, evaluation protocols, and expected outcomes. This component offers a more constrained structure that enables measurable, partially automatable assessment while still reflecting an essential aspect of scientific ideation. We view this as a pragmatic starting point, and future versions of the benchmark may incorporate broader hypothesis-level evaluation once sufficiently robust expert-sourced ground truth becomes feasible.

To make the assessment more systematic and tractable, we decompose an originally holistic idea into several interrelated components, forming a structured representation of the idea. This decomposition enables more fine-grained evaluation along dimensions such as effectiveness, novelty, level of detail, and feasibility [37].

Task Definition of Idea Generation Task Input

- • Related Work (RW): A summary of existing research relevant to a certain research direction, providing context for new ideas.
- • Challenge (C): The current challenges in the field and the limitations of existing solutions.
- • Limitation (L): Specific shortcomings or constraints of current research that new ideas need to address.
- • Motivation (M): The perspective and motivation of addressing the limitations in this research direction.
- • Task Objective (TO): The primary goal of the task, such as generating ideas that

- solve identified challenges or improve existing solutions.
- • Existing Solutions (ES): A description of the current approaches or solutions available in the field.

###### Task Output

- • Core Idea (CI): The central novel idea or concept generated to address the research challenge.
- • Implementation Steps (IS): The steps or procedures required to implement the core idea.
- • Implementation Order (IO): The sequence in which the implementation steps should be executed.
- • Data (D): The data that will be used to implement the idea or evaluate its effectiveness.
- • Evaluation Metrics (EM): The criteria for assessing the success or relevance of the generated idea.
- • Expected Outcome (EO): The anticipated result or contribution the idea is expected to achieve.

###### Task Formulation

CI, IS, IO, D, EM, EO = LLM/Agent(RW, C, L, M, TO, ES)

- Figure 4 | Idea Generation Task: Inputs, outputs, and formulation for methodology design, integrating evaluation metrics and structured implementation planning.

###### 2.1.3. Dry/Wet Experiment

Scientific experimentation represents the core of the discovery process, bridging theoretical formulation and empirical validation [30]. Within SGI-Bench, we formalize this process into two complementary categories: dry and wet experiments. Dry experiments capture computational and simulation-based studies—where AI assists in generating, refining, or executing scientific code that models physical phenomena. [38, 39] Wet experiments, by contrast, simulate laboratory-based workflows, requiring the model to plan and reason about sequences of actions involving physical instruments, reagents, and procedural parameters [40, 41]. Together, these two categories span the continuum from theoretical abstraction to empirical realization, offering a holistic evaluation of how AI can assist scientists in both virtual and physical experimentation.

Computational and laboratory experiments take many forms in real scientific practice. For dry experiments, possible tasks range from full pipeline construction to simulation design and multimodule scientific computing; in this benchmark, we adopt a code-completion–based formulation [42], where the model fills in missing components of an existing scientific script rather than generating an entire project from scratch. For wet experiments, laboratory workflows span diverse operational activities, yet we focus on the protocol-design aspect [43], where the model composes a sequence of experimental actions and parameters from a predefined action space.

By constraining dry and wet experiments to code completion and protocol design respectively, we retain core aspects of computational and laboratory reasoning while ensuring reproducibility, controlled variability, and reliable evaluation across models.

Dry Experiment Dry experiments emphasize computational problem-solving, reflecting the growing role of AI in automating simulation-driven science. Each task presents the model with incomplete or masked scientific code that encapsulates domain-specific computations, such as molecular dynamics, climate modeling, or numerical solvers in physics [44]. The model must infer the missing logic, reconstruct executable code, and ensure that the resulting program produces correct and efficient outcomes. This task thus evaluates a model’s ability to integrate scientific understanding with code synthesis—testing not only syntactic correctness but also conceptual fidelity to the underlying scientific problem [42].

To better characterize the scope of dry experiments, we categorize representative computational functions commonly encountered across disciplines, including numerical calculation, statistical analysis, simulation, metric calculation, data processing, and predictive modeling, as shown in Table 2. The completion or generation of these functions offers a rigorous measure of how well AI systems can operationalize scientific intent into executable form.

- Table 2 | Dry Experiment Function Types: Representative computational functions and their roles across scientific code-completion tasks.

Function Category Core Role in Scientific Experiments Numerical Calculation Basic mathematical computations required to support physical

or chemical modeling. Statistical Analysis Processing experimental data using descriptive or inferential statistics to identify trends and distributions.

Simulation Running computational simulations (e.g., molecular dynamics, finite element analysis) and filtering results for relevant conditions.

Metric Calculation Computing evaluation metrics such as accuracy, error, or performance indicators for validating experiments.

Data Processing Handling raw data before and after experiments, including normalization, cleaning, and feature extraction.

Predictive Modeling Applying machine learning methods to categorize, predict, or group experimental results.

In real scientific workflows, dry experiments correspond to the stage of experimental design in computational and simulation-based studies. Following hypothesis formulation, researchers employ virtual experiments to anticipate and evaluate potential outcomes prior to empirical validation, enabling a cost-efficient and theoretically grounded pre-assessment of experimental feasibility.

Task Definition of Dry Experiment Task Input

- • Background (B): Information from relevant scientific code, providing context for the dry experiment.
- • Data Code (D): The data used in the experiment, including any code snippets or predefined inputs.
- • Main Code (M): The core experimental code where some functions may be masked or missing.

###### Task Output

• Functions (F): The missing functions in the main code 𝑀, which the system is tasked

with generating or completing. Task Formulation

F = LLM/Agent(B, D, M)

- Figure 5 | Dry Experiment Task: Inputs, outputs, and formulation for code-completion based computational studies with masked functions.

Wet Experiment Wet experiments represent the physical realization of scientific inquiry, encompassing laboratory and field-based procedures that transform theoretical designs into empirical evidence. These tasks simulate the execution phase of real-world experiments, where models are required to plan, organize, and reason through sequences of atomic actions involving materials, instruments, and procedural parameters. Given inputs describing experimental objectives, configurations, and available tools, the model must generate structured, executable protocols that are both accurate and practically feasible. Evaluation considers not only the correctness of individual steps but also their procedural coherence and alignment with established laboratory conventions.

In real scientific workflows, wet experiments correspond to the execution and validation stages of discovery. This is where hypotheses are tested against the physical world, data are collected, and evidence is generated to confirm, refine, or refute prior assumptions. By assessing how effectively AI systems can design and reason through these embodied experimental processes, this task provides a window into their capacity to bridge symbolic understanding with real-world scientific practice.

Task Definition of Wet Experiment Task Input

- • Background (B): Information from relevant experimental procedure.
- • Action Pool (AP): A predefined set of atomic actions that can be used in the experiment, along with explanations and corresponding input/output definitions.

Task Output

- • Atomic Action Order (AAO): The order in which atomic actions should be executed.
- • Atomic Action Parameters (AAP): The parameters associated with each atomic action (e.g., reagents, temperature).

Task Formulation

AAO, AAP = LLM/Agent(B, AP)

- Figure 6 | Wet Experiment Task: Inputs, outputs, and formulation for laboratory protocol planning via atomic actions and parameters.

###### 2.1.4. Experimental Reasoning

Experimental reasoning refers to the process of interpreting scientific observations and data to reach justified conclusions. In this benchmark, we focus on data-analysis–oriented reasoning [45], where the model must extract relevant visual or numerical cues from multi-modal sources [46], compare conditions, and identify causal or descriptive patterns. This formulation emphasizes analytical interpretation rather than open-form scientific narrative, enabling reliable assessment while capturing an essential part of empirical scientific reasoning.

We consider five representative modalities as shown in Table 3: a) process images that integrate symbolic and textual information to depict workflows or variable relationships; b) observation images representing raw data captured by instruments such as telescopes, satellites, or microscopes; c) experiment images documenting laboratory setups and procedures; d) simulation images generated by computational models to visualize physical or chemical processes; and e) visualization images such as plots or charts that reveal patterns within structured datasets. Collectively, these modalities reflect the multi-faceted and evidence-driven nature of scientific inquiry.

- Table 3 | Experimental Reasoning Modalities: Five visual modalities used for multi-modal evidence and analysis.

Modality Core Description Scientific Role Process Images Graphical symbols + text describing

Capture the logical flow of experiments and research design.

workflows or variable relations.

Observation Images Raw data from instruments (e.g., telescope, satellite, microscope).

Provide direct evidence of natural or physical phenomena.

Experiment Images Photos of instruments, setups, or lab operations.

Document experimental configurations and operational details.

Simulation Images Generated from computational models/software.

Visualize theoretical predictions of physical or chemical processes.

Visualization Images Processed structured data into charts/plots.

Reveal patterns, comparisons, or correlations from datasets.

To reason effectively over such diverse inputs, we define four complementary reasoning paradigms as shown in Table 4: a) signal perception, focusing on the extraction of direct patterns from visual signals; b) attribute understanding, which demands domain knowledge to interpret key visual or contextual features; c) comparative reasoning, involving integration and comparison across multiple sources to ensure consistency and rigor; and d) causal reasoning, aimed at uncovering underlying mechanisms and scientific principles. These paradigms collectively span the hierarchy from low-level perception to high-level scientific inference.

In real-world scientific workflows, experimental reasoning corresponds to the data analysis stage, during which scientists interpret experimental and simulated data, perform comparative analyses, and refine hypotheses based on empirical evidence.

Task Definition of Experimental Reasoning Task Input

- • Multiple Experimental Images (MEI): A set of images representing various experimental outcomes or data collected from instruments.
- • Question (Q): A specific question or hypothesis related to the experimental data that

- Table 4 | Experimental Reasoning Paradigms: Four reasoning paradigms spanning perception to causality with examples and requirements.

Reasoning Paradigm Core Requirement Typical Example Signal Perception Direct extraction of information from

Identifying patterns in telescope images or microscope slides.

visual signals without heavy prior knowledge.

Attribute Understanding Requires disciplinary background to interpret key features and scientific attributes.

Recognizing crystalline structures in materials science images.

Comparative Reasoning Integrates and contrasts information across multiple images, often crossdomain.

Comparing climate model simulations with satellite observations.

Causal Reasoning Goes beyond correlation to infer mechanisms or propose hypotheses.

Inferring causal pathways in gene expression from multi-modal experimental data.

requires reasoning or analysis. Task Output

- • Reasoning (R): The specific steps in the reasoning process, including calculation, thinking, analysis, etc..
- • Answer (A): The conclusion drawn from analyzing the experimental data, answering the specified question or hypothesis.

###### Task Formulation

R, A = LLM/Agent(MEI, Q)

- Figure 7 | Experimental Reasoning Task: Inputs, outputs, and formulation for multi-modal analysis with step-by-step reasoning and final answers.

###### 2.2. Multi-Dimensional Metrics

To align with the scientific characteristics of each task, we have designed multi-dimensional evaluation metrics for every task. This approach avoids a one-size-fits-all binary judgment and instead provides a more fine-grained assessment.

###### 2.2.1. Metrics of Scientific Deep Research

The Scientific Deep Research task draws inspiration from AI’s deep research paradigms [47, 48, 49, 50, 51] while incorporating methodologies from meta-analysis in the scientific domain. The former emphasizes multi-step reasoning, where solving a problem often requires iterative searches, calculations, and inferences; the correctness of each step directly impacts the accuracy of the final answer. The latter focuses on systematically extracting and synthesizing data from literature, requiring highly precise results. Accordingly, our metrics capture both step-by-step reasoning fidelity and final answer accuracy.

Metric Definition of Exact Match

Exact Match (EM): Since the Scientific Deep Research tasks are designed to have short, unique, and easily verifiable answers, we use exact match as a hard metric to assess whether the model’s final answer is correct. The model receives a score of 1 if the output exactly matches the reference answer, and 0 otherwise.

Metric Definition of Step-Level Accuracy

Step-Level Accuracy (SLA): Models are required to produce step-by-step solutions. We employ an LLM-based judge to compare each model-generated step against the reference solution steps. For each step, the judge determines whether it is correct and provides reasoning. This fine-grained evaluation avoids binary correctness judgments for the entire solution, allowing precise assessment of reasoning accuracy at each inference step. The metric is computed as the proportion of steps correctly solved relative to the total number of steps. The score is calculated as

Number of correct reasoning steps Total number of reasoning steps

SLA =

.

###### 2.2.2. Metrics of Idea Generation

To evaluate the open-ended nature of idea generation, we adopt a hybrid framework that integrates both subjective and objective metrics. We assess each idea along four dimensions—effectiveness, novelty, detailedness, and feasibility—which together characterize an idea’s scientific quality, creativity, and executability [37, 52].

Subjective Evaluation via LLM Judges. For subjective scoring, we perform pairwise comparisons between model-generated ideas and expert-written reference ideas. For each of the four dimensions, an LLM judge selects which idea is superior. To ensure fairness and robustness, we employ three different LLM judges, each casting two independent votes, resulting in a total of six votes per dimension. The pairwise win rate against the reference idea is then used as the subjective component of the score for each dimension.

Objective Evaluation via Computable Metrics. In addition to subjective judgments, we design dimension-specific computational metrics that capture structured properties of the ideas.

Metric Definition of Effectiveness

For each reference idea, human experts extract its 3–5 most essential keywords. We compute the hit rate of these keywords in the model-generated idea, allowing semantic matches to avoid underestimating effectiveness. The final effectiveness score is the average of the keyword hit rate and the LLM-judge win rate:

Keyword Hit Rate + LLM Win Rate 2

Effectiveness =

.

Metric Definition of Novelty

We measure novelty by computing the dissimilarity between the model-generated idea and prior related work. Lower similarity indicates that the model proposes ideas not present in existing literature and therefore exhibits higher creativity. The final novelty score averages the

dissimilarity score and the subjective win rate:

Dissimilarity Score + LLM Win Rate 2

Novelty =

.

Metric Definition of Detailedness

We evaluate detailedness from two angles: a) content completeness, which checks whether the idea contains required components (Core Idea, Implementation Steps, Implementation Order, Dataset, Evaluation Metrics, Expected Outcome), and b) redundancy penalty, computed via sentence-level semantic similarity. Ideas with many repetitive sentences are penalized, as verbosity without substance does not constitute genuine detail. The final detailedness score is:

Completeness Score (with Penalty) + LLM Win Rate 2

Detailedness =

.

Metric Definition of Feasibility

For each research direction, domain experts provide a standardized implementation graph containing the essential nodes and their execution order. We extract an implementation graph from each model-generated idea and compute its similarity to the expert template. A low similarity indicates that the proposed idea does not align with accepted solution workflows and is therefore infeasible. The final feasibility score is:

Graph Similarity + LLM Win Rate 2

Feasibility =

.

Taken together, the hybrid subjective–objective design provides a robust, interpretable, and comprehensive assessment of LLMs’ scientific idea generation capabilities across creativity, structural clarity, and practical executability.

###### 2.2.3. Metrics of Dry/Wet Experiment

Dry Experiment Dry experiments focus on code generation task. Specifically, each problem includes background information, data code, and main code with certain functions masked. The model is tasked with completing the missing functions. Each problem contains 5 unit tests. Our metrics capture both correctness and execution behavior of the generated code [53].

Metric Definition of Pass All k Unit Tests

Pass all k Unit Tests(PassAll@k): This metric measures the proportion of problems with k or more unit tests passed successfully. It’s important to distinguish this from Pass@k. While Pass@k requires only one successful attempt out of k trials, PassAll@k demands that at least k attempts pass the unit tests. Consequently, PassAll@5 represents the most challenging criterion. The score is calculated as

Number of problems with k or more unit tests passed Total number of problems

PassAll@k =

.

Metric Definition of Average Execution Time Average Execution Time (AET): This metric captures the efficiency of the generated code by measuring the average runtime across all test cases:

###### ∑︁𝑁

1

AET =

𝑡𝑖,

𝑁

𝑖=1

where 𝑡𝑖 is the execution time of the 𝑖-th test case and 𝑁 is the total number of test cases.

Metric Definition of Smooth Execution Rate

Smooth Execution Rate (SER): This metric measures the proportion of generated code that runs without any runtime errors, regardless of correctness of output. It reflects adherence to basic coding standards and robustness:

Number of code executions without errors Total number of code executions

SER =

.

Wet Experiment Wet experiments involve procedural steps using laboratory instruments. Correct execution requires both the correct sequence of actions and proper parameter settings. Accordingly, we propose the following metrics:

Metric Definition of Sequence Similarity

Sequence Similarity (SS): This metric evaluates the similarity between the order of atomic actions provided by the model and the reference sequence. Let seqmodel and seqref be the sequences of atomic actions from the model and the reference, respectively. Denote by Inv(seqmodel, seqref) the number of discordant pairs between the sequences. For sequences of length 𝑛, the score is computed as:

Inv(seqmodel, seqref) 𝑛(𝑛−1) 2

SS = 1 −

,

where 𝑛(𝑛2−1) is the maximum possible number of inversions. By definition, SS = 1 indicates that the sequences are identical, while SS = 0 indicates maximal disorder relative to the reference

sequence.

Metric Definition of Parameter Accuracy

Parameter Accuracy (PA): This metric measures the correctness of input parameters for each atomic action compared to the reference, including reagent types, concentrations, volumes, or other domain-specific parameters. The score is calculated as the proportion of correctly specified parameters across all actions:

Number of correctly specified parameters Total number of parameters

PA =

.

###### 2.2.4. Metrics of Experimental Reasoning

The Experimental Reasoning task assesses the multi-modal scientific reasoning capabilities of LLMs and agents. Specifically, given several images and a corresponding question, the model is required to select the correct option from no fewer than 10 candidates. For evaluation, the correctness of the final answer and the validity of intermediate reasoning are equally critical. Therefore, two evaluation metrics are adopted, as detailed below.

Metric Definition of MCA

Multi-choice Accuracy (MCA): Given several options, the model receives a score of 1 if the selected option exactly matches the reference answer, and 0 otherwise. The final score of MCA is the average of all individual scores across all test samples. This metric directly quantifies the model’s ability to pinpoint the correct solution from a large candidate pool, serving as a foundational measure of its end-to-end scientific reasoning accuracy in the multi-modal task.

Metric Definition of Reasoning Validity

Reasoning Validity (RV): Models are required to generate step-by-step logical reasoning to justify their selected answers. An LLM-based judge is utilized to assess the model-generated reasoning against a reference reasoning. For each test sample, the LLM judge assigns a validity score ranging from 0 (completely invalid, contradictory, or irrelevant) to 10 (fully rigorous, logically coherent, and perfectly aligned with the reference reasoning), accompanied by justifications for the assigned score. This fine-grained scoring paradigm circumvents the limitations of binary correctness assessments, enabling precise quantification of reasoning quality, including the validity of premises, logical transitions, and alignment with scientific principles. The final RV score is computed as the mean of individual sample scores across the entire test set, reflecting the model’s overall capability to perform interpretable and reliable scientific reasoning.

###### 2.3. Scientist-Aligned Data Construction

Raw Corpus Collection In this stage, we conducted multiple discussions with experts from diverse scientific disciplines, drawing from both the 125 important scientific questions published in Science, and the prominent research directions in various disciplines with significant scientific impact. Ultimately, we curated 75 research directions spanning ten scientific domains, as shown in Figure 8. Please refer to Appendix A.2 for a complete list of research directions.

Subsequently, we collected raw data provided by experts and researchers, primarily consisting of scientific texts and images across the various disciplines. The texts mainly cover knowledge introduction, methodological design, experimental procedures, and data analysis. The images include experiment figures, data visualizations, and observational images, each accompanied by detailed descriptions.

In addition, these experts and researchers will provide seed questions and annotation requirements for annotation, which provide initial examples for the subsequent annotation process, as illustrated in Figure 2 (G).

Question Construction After gathering the raw data, we recruited over 100 Master’s and PhD holders from different disciplines to construct benchmark questions according to the task definitions. Annotators first analyzed the collected texts and images, and then created questions according

to annotation requirements and seed questions. Several rules were applied to ensure scientific validity and authenticity. Specifically, annotators were required to reference the original data source and paragraph for each question, ensuring traceability to scientist-provided data. Furthermore, all questions are constructed by at least two annotators, one of whom is responsible for generating complex draft questions, and the other is responsible for refining them, as shown in Figure 2 (G).

During question construction, experts continuously reviewed the generated questions. Each question was immediately submitted to the relevant expert for evaluation, who assessed its scientific value. For instance, a question with an experiment configuration that lacks general applicability would be deemed scientifically invalid. Experts provided feedback to annotators, who then revised the questions accordingly, ensuring that the constructed questions remain aligned with the perspectives and standards of domain scientists.

Data Cleaning Once all questions were constructed, we applied three layers of data cleaning: 1. Rule-based cleaning: Questions that did not meet task-specific criteria were removed. For example, for Scientific Deep Research, steps must be short sentences forming a list, each representing one step; for Wet Experiments, each action must exist in the predefined action pool. 2. Model-based cleaning: Large language models were used to detect and remove questions with semantic errors or potential logical inconsistencies. 3. Expert quality check: All questions were reviewed by the original data-providing scientists, removing incomplete questions, questions with non-unique answers, or questions whose research direction did not align with the source data. For Dry Experiments, Python environments were used to test all code snippets to ensure executability.

Difficulty Filtering After data cleaning, we filtered questions based on difficulty using mainstream LLMs. We evaluated each question with six high-performance models (e.g., GPT-5 [54], Gemini-2.5Pro [5], DeepSeek-R1 [55], Kimi-k2 [56]) under a setup allowing web search and deep-reasoning modes. Questions that more than half of the models could correctly answer were removed. This process ensures that the benchmark remains highly challenging.

Through these four steps, we guarantee that all benchmark questions are derived from authentic scientific data, aligned with domain scientists’ judgment of scientific value, and maintain both high quality and high challenge.

###### 2.4. Data Distribution

After the data construction process, we obtained the complete SGI-Bench benchmark, which contains 318 Scientific Deep Research questions, 315 Idea Generation questions, 271 Dry Experiment questions, 68 Wet Experiment questions, and 291 Experimental Reasoning questions. The discipline distributions for Scientific Deep Research, Idea Generation, and Experimental Reasoning are identical, as shown in Figure 9 (a). The discipline distributions for Dry and Wet Experiments are presented in Figure 9 (b) and Figure 9 (c), respectively, with Wet Experiments covering only a subset of disciplines, such as Biology and Chemistry.

In addition to discipline-level distributions, we further categorized the tasks at a finer granularity. For Scientific Deep Research, questions are grouped based on the type of target being investigated into four categories: Data, Properties, Micro-Experiments, and Macro-Experiments, as detailed in Table 1. The distribution of these types is illustrated in Figure 9 (d). For Dry Experiments, questions are classified into six types according to the masked function type, as shown in Table 2, with the corresponding distribution displayed in Figure 9 (e). In Experimental Reasoning, the task inputs include images spanning multiple modalities, including Process Images, Observation Images, Experiment Images,

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

|Zeeman Effect Experiment Quadrupole mass spectrometer Computational Condensed Matter Physics Superconducting MechanismsNuclear Magetic Resonance Soft Condensed Matter Physics<br><br>Physics<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]|
|---|

[Figure 136]

[Figure 137]

Earth Chemistry

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Molecular Interaction Molecilar Property Prediction Target-based Drug Design De novo Drug Design

Seismic Wave Detection Ocean Heat Content Vegetation Coverage Rate Ozone Pollution and Causes Atmospheric Differential Equation Glacier Estimation

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Material Life

[Figure 158]

[Figure 159]

[Figure 160]

Disease Biomarker Discovery Tumor Neoantigen Discovery Protein Structure Genome Function Prediction AI Drug Discovery Tumor Immunotherapy Regulatory Element Design

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

KRF Resin Polymerization Chloride Solid-State Electrolyte Polymer Thermoelectric Thermal Electrocalalysis Nano Adsorption Materials Oxygen Evolution Reaction

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

NeuroScience Math

[Figure 187]

[Figure 188]

Visual Decoding Motion Decoding Emotion Recognition Neural Activity Behavior Prediction EM Neuron Segmentation

[Figure 189]

Differential Privacy Matrix Completion Shortest Path Planning Optimization algorithms Differential Equations

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Astronomy Information

Energy

[Figure 205]

[Figure 206]

[Figure 207]

Gravitational Wave Detection Fast Radio Burst Detection Real-time Optical Transient Survey

[Figure 208]

Dialog System Code Generation Sensor Phase-free Reconstruction

[Figure 209]

Optimal Power Flow Calculation New Energy Power Forecasting

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

- Figure 8 | Benchmark Subjects: Overview of 10 scientific domains covered by SGI-Bench.

Simulation Images, and Visualization Images, summarized in Table 3 and visualized in Figure 9 (f). Moreover, based on the type of reasoning required, questions are further categorized into Signal Perception, Attribute Understanding, Comparative Reasoning, and Causal Reasoning, as detailed in

- Table 4, with distributions shown in Figure 9 (g).

These fine-grained categorizations by discipline and task type facilitate a detailed analysis of the limitations of evaluated LLMs and agents across scientific domains and research tasks. Such insights provide clear directions for advancing AI-assisted scientific discovery.

##### 3. SGIEvalAgent: Agentic Evaluation Framework

Given the inherent complexity of scientific discovery, evaluating the performance of LLMs and agents in this domain presents formidable challenges. Rather than merely employing LLMs as evaluators, we develope a comprehensive, agent-based evaluation framework augmented with diverse capabilities (e.g., web search, Python interpreter, file reader, PDF parser, metric-specific Python functions [57]) to ensure rigorous, accurate, and scalable evaluations. As illustrated in Figure 10, this framework is structured into four interconnected stages: Question Selection, Metric Customization, Predict & Eval, and Report Generation, each orchestrated by specialized agents to address distinct facets of the evaluation workflow.

(b) (c) (d)

(a)

(e) (f) (g)

- Figure 9 | Benchmark Data Distribution: (a) Overall discipline distribution; (b) Dry experiment discipline distribution; (c) Wet experiment discipline distribution; (d) Scientific Deep Research question types; (e) Dry Experiment function types; (f) Experimental Reasoning image modalities; (g) Experimental Reasoning reasoning paradigms.

###### 3.1. Question Selection

The Question Selection stage is managed by a dedicated questioning agent, which interprets user queries to retrieve relevant questions from the SGI-Bench question bank. The agent filters questions according to multiple criteria, including disciplinary domain, task category, and evaluation intent specified in the input query. In scenarios where no user query is provided, the agent defaults to systematically selecting all questions from the SGI-Bench, thereby ensuring comprehensive coverage across all scientific tasks. This stage effectively defines the evaluation scope by specifying the precise set of problems that subsequent stages will assess.

Question Agent Definition Agent Input

- • User Query (Q): Any content input by users for obtaining relevant information, which can be in various forms such as text, keywords, or questions.
- • SGI-Bench Data (D): All constructed datasets in SGI-Bench, each of which is associated with a specific discipline and corresponding research area.
- • K-value (K): A positive integer indicating the number of most relevant items to select from the SGI-Bench Data based on the User Query.

Agent Output

• Selected Indices (SI): The selected indices for locating and retrieving the target data.

###### 3.2. Metric Customization

In the metric customization stage, a metric customization agent first dynamically generates novel evaluation metrics based on user queries and selected questions. The agent parses the evaluation intent from user input to formalize customized metric instructions with advanced tools like web search and PDF parser, enabling flexible prioritization of metrics or integration of novel evaluation dimensions. Then, the customized metrics will be aggregated with predefined scientist-aligned metrics

[Figure 238]

###### Stage 1: Question Selection

###### Stage 2: Metric Customization

Deliberation

User Query

Experiments Metrics

Scientific Deep Research Metrics

Idea Generation Metrics

[Figure 239]

[Figure 240]

Conception

Format

Multi-dimensional Metrics

[Figure 241]

[Figure 242]

QueryReportUser

[Figure 243]

Tools: ……

Action

[Figure 244]

[Figure 245]

SGI Bench

[Figure 246]

Metric Filtering

Questioning Agent

Perception

[Figure 247]

Inject

Filtered Metrics

Customized Customization Metrics

[Figure 248]

[Figure 249]

User Intent Analysis

Data Matching

SGI Subset

Agent

Metrics for Evaluation

[Figure 250]

[Figure 251]

###### Stage 4: Report Generation

###### Stage 3: Prediction & Evaluation

For Every Metric

Evaluation

[Figure 252]

Question: …what is the accuracy of node's maximum cavity?

[Figure 253]

User Query

Exact Match Metric

- for Metric 1 Evaluation

- for Metric 2 Evaluation

- for Metric 3 Evaluation

- for Metric 4 ……

[Figure 254]

[Figure 255]

[Figure 256]

Answer: 39.1% Tools: ……

[Figure 257]

LLM/Agent Runner

Reporting Agent

[Figure 258]

[Figure 259]

[Figure 260]

Eval Agent

Output Parser

Settings

Evaluation

###### Figure 10 | Evaluation Framework.

given different question types, as described in Section 2.2, to form the final metrics for evaluation. By synergizing pre-defined and user-customized metrics, this stage ensures the framework aligns with both standardized benchmarks and domain-specific demands.

Customization Agent Definition Agent Input

- • User Query (UQ): Any content input by users for obtaining relevant information, which can be in various forms such as text, keywords, or questions.
- • SGI-Bench Data (D): All constructed datasets in SGI-Bench, each of which is associated with a specific discipline and corresponding research area.
- • Selected Indices (SI): The selected indices for locating and retrieving the target data.
- • Tool Pool(T): A set of pre-configured tools for agents to call, including web search, PDF parser, Python Interpreter, etc.
- • Metric Pool(M): A set of pre-defined task-specific metrics presented in Section 2.2.

###### Agent Output

• Metrics for Evaluation (ME): Generated novel metrics based on the user query.

###### 3.3. Inference and Evaluation

The predict & eval stage leverages a tool pool that includes utilities like web search, PDF parser, and Python interpreter to first execute inference for target LLMs or agents on the questions selected in the first stage. Subsequently, a dedicated Science Eval Agent (SGI-Bench Agent) applies the metrics finalized in the second stage to score the inference results. For each score, the agent generates a ratio-

nale grounded in reference answers, question context, and supplementary information retrieved via tools if necessary, thereby ensuring transparency and reproducibility. By integrating tool-augmented inference with systematic, metric-driven scoring, this stage effectively addresses the multi-dimensional and complex nature of scientific reasoning assessment.

Evaluation Agent Definition Agent Input

- • SGI-Bench Data (D): All constructed datasets in SGI-Bench, each of which is associated with a specific discipline and corresponding research area.
- • Selected Indices (SI): The selected indices for locating and retrieving the target data.
- • Responses (R): Generated responses by the evaluation target in the Testbed.
- • Tool Pool(T): A set of pre-configured tools for agents to call, including web search, PDF parser, Python Interpreter, etc.
- • Metrics for Evaluation (ME): Generated novel metrics based on the user query.

###### Agent Output

- • Score (S): A single integer score from 0–10, where 10 means the response is fully correct compared to the answer. Higher scores indicate the Prediction is better, and lower scores indicate it is worse.
- • Rationale (RN): A brief explanation of why the response is correct or incorrect with respect to accuracy, completeness, clarity, and supporting evidence.

###### 3.4. Report Generation

The report generation stage is orchestrated by a dedicated reporting agent, which aggregates the user evaluation intents, finalized metric specifications, and the results produced during the Predict & Eval stage. The agent then compiles a comprehensive report that both visualizes and quantifies the performance of different LLMs and agents across the selected questions and metrics. Beyond summarizing raw results, the report contextualizes the findings within the broader landscape of scientific discovery capabilities, thereby enabling users to extract actionable insights and make informed decisions efficiently.

Reporting Agent Definition Agent Input

- • Score List(SL): A list of integers score from 0–10, where 10 means the response is fully correct compared to the answer. Higher scores indicate the Prediction is better, and lower scores indicate it is worse.
- • Rationale List(RNL): A list of explanations of why the response is correct or incorrect with respect to accuracy, completeness, clarity, and supporting evidence.
- • User-customized Metric (UM): Generated novel metrics based on the user query.

###### Agent Output

• Report (R): A comprehensive final evaluation report that demonstrates the scientific discovery capabilities of different LLMs and agents.

##### 4. Evaluation Results

###### 4.1. Evaluation Setup

To comprehensively evaluate different models throughout the scientific discovery workflow, we performed quantitative assessments across diverse LLMs and agents using scientist-aligned metrics.

- • For open-weight LLMs, we evaluated DeepSeek-V3.2 [58], DeepSeek-R1 [55], Intern-S1 and Intern-S1-mini [59], Kimi-k2 [56], Qwen3-VL-235B-A22B [60], Qwen3-235B-A22B, Qwen3Max, and Qwen3-8B [61], and Llama-4-Scout [62].
- • For closed-weight LLMs, we assessed GPT-4o [63], GPT-4.1 [64], GPT-5 [54], GPT-5.1 [65], GPT5.2-Pro [66], o3 and o4-mini [67], Gemini-2.5-Flash and Gemini-2.5-Pro [5], Gemini-3-Pro [68], Claude-Opus-4.1 [69], Claude-Sonnet-4.5 [70], Grok-3 [71], and Grok-4 [72].
- • For open-source agents, we tested SmolAgents(GPT-4.1) and SmolAgents(Gemini-2.5Flash) [57], Owl(GPT-4.1) and Owl(Gemini-2.5-Flash) [73], WebThinker [74], XMaster [75], and InternAgent [76].
- • For closed-source agents, we evaluated OpenAI DeepResearch(o3) and OpenAI DeepResearch(o4mini) [48], Kimi-Search(Kimi-k2) [50], Doubao-Search(Seed-1-6), Grok-Search(Grok-4) [51], and Perplexity(Sonar-Pro) [49].

For benchmarking consistency, we set the temperature of all configurable models to 0 to minimize randomness and used a standard zero-shot, task-specific prompt template across all tasks.

###### 4.2. Overview

Table 5 provides a cross-task snapshot of current capabilities. Overall, SGI-Score remains low across families (typically 30±5), with the best aggregate result at 33.83 (Gemini-3-Pro). Closed-source models show only a marginal edge over leading open-source systems (e.g., Claude-Sonnet-4.5 at 32.16 vs. Qwen3-Max at 31.97), indicating that scale and access alone do not translate into robust scientific cognition. At the task level, Deep Research is the most brittle under the strict Exact-Match metric (best 18.48; many models around 8–16), revealing the difficulty of end-to-end, multi-source evidence integration and numerically faithful inference. Idea Generation exhibits the opposite pattern—strong surface performance but weak realizability: while GPT-5 attains the highest average (55.40), feasibility remains uniformly low across models, reflecting underspecified implementation details and missing resource/parameter assumptions. In Dry Experiments, high executability does not imply correctness: even the best PassAll@5 peaks at 36.64 (Gemini-3-Pro), underscoring persistent gaps in numerical stability and scientific algorithm selection. Wet Experiments remain challenging, with uniformly low action-sequence similarity and only moderate parameter accuracy, driven by errors in step ordering, temporal coordination, and branch/sample bookkeeping. Multimodal Experimental Reasoning shows relatively stronger results (best MCA 41.92), yet remains far from reliable scientific discrimination. Taken together, these patterns validate our SGI framing: contemporary models possess fragments of the Deliberation–Conception–Action–Perception cycle but fail to integrate them into a coherent, workflow-faithful intelligence—pointing to the need for meta-analytic retrieval with numerical rigor, planning-aware conception, and procedure-level consistency constraints.

###### 4.3. Scientific Deep Research

The results for LLMs and agents are presented in Figs. 12 and 13. Exact Match (EM) evaluates the correctness of the final answer, while Step-Level Accuracy (SLA) measures alignment with the reference reasoning trajectory. EM remains low across all evaluated systems, typically around 10% and seldom above 20%, indicating that current models capture only a narrow fraction of the analytical

Model DeepResearch IdeaGen DryExp WetExp ExpReasoning SGI-Score Open-source LLM

DeepSeek-V3.2 12.70 37.45 23.62 20.95 - DeepSeek-R1 15.03 40.16 33.33 21.12 - Intern-S1 15.74 38.09 28.79 29.02 28.87 28.10 Intern-S1-mini 11.06 36.04 16.97 12.42 16.84 18.67 Kimi-k2 13.11 43.17 29.52 25.76 - Qwen3-VL-235B-A22B 11.97 39.28 28.41 30.30 31.62 28.32 Qwen3-235B-A22B 14.19 39.45 28.89 26.40 - Qwen3-Max 15.38 39.83 33.21 33.62 37.80∗ 31.97∗

-  冠军

| |
|---|

（金牌第一名）

-  亚军

| |
|---|

（银牌第二名）

-  季军

| |
|---|

（铜牌第三名）

Qwen3-8B 8.18 35.78 18.45 9.96 23.37∗ 19.15∗ Llama-4-Scout 7.86 29.72 20.37 21.66 25.77 21.08

Closed-source LLM

- GPT-4o 7.86 35.95 26.94 31.31 32.30 26.87

- GPT-4.1 11.32 36.49 34.32 36.63 38.49 31.45

GPT-5 14.47 55.40 29.89 16.31 38.14 30.84

- GPT-5.1 11.64 47.12 31.00 22.77 34.02 29.31 GPT-5.2-Pro 15.72 55.03 28.04 17.50 39.18 31.09

- o3 12.89 46.07 31.73 30.04 32.65 30.68

- o4-mini 11.95 40.78 35.79 28.86 33.33 30.14 Gemini-2.5-Flash 10.69 39.13 21.03 18.55 34.36 24.75 Gemini-2.5-Pro 15.09 39.95 22.51 22.05 41.24 28.17

###### Gemini-3-Pro 18.48 39.68 36.64 32.45 41.92  33.83冠军

| |
|---|
| |
| |

（金牌第一名）

-  冠军 （金牌第一名）
-  亚军 （银牌第二名）

Claude-Opus-4.1 12.93 40.29 34.69 25.38 38.83 30.42 Claude-Sonnet-4.5 13.84 43.20 35.79 30.15 37.80 32.16

| |
|---|
| |
| |

######  亚军

（银牌第二名）

- Grok-3 13.52 35.98 27.31 37.92 -  季军- （铜牌第三名）

- Grok-4 13.31 37.12 33.71 29.01 30.24 28.68

| |
|---|
| |
| |

######  季军

（铜牌第三名）

- Table 5 | Overview Results Across SGI-Bench Tasks: Aggregated performance across Deep Research, Idea Generation, Dry/Wet Experiment, and Experimental Reasoning. The scores for Deep Research are based on the exact match metric (the strictest metric). Idea Generation scores are the average of four metrics evaluating ideas. Dry Experiment scores are based on PassAll@5 (the strictest metric). Wet Experiment scores are the average of action sequence similarity and parameter accuracy. Experimental Reasoning scores are based on the multi-choice accuracy metric (the strictest metric). The SGI-Score is the average across these tasks, reflecting the overall capability of an AI model in various scientific research scenarios. An asterisk ∗ indicates results from different versions of the same series of multimodal models.

depth required for scientific deep research. While top-performing tool-augmented agents slightly outperform the best offline LLMs on SLA, the overall distributions overlap substantially; several agent systems underperform many LLMs, and EM differences are marginal with the best LLMs matching or exceeding the best agents.

SLA substantially exceeds EM across nearly all systems. Multiple systems, including several agents—achieve SLA above 50%, with the best around 65%. This disparity suggests that models frequently produce partially correct or locally consistent reasoning steps but struggle to maintain coherence and correctness across the full reasoning chain. Such behavior underscores the intrinsic difficulty of end-to-end scientific reasoning and the importance of step-wise decomposition for improving task success.

Newer large-scale LLMs do not universally outperform predecessor models. For example, Grok-4 exhibits lower EM and SLA than Grok-3 on this benchmark, suggesting that large-scale training may introduce regressions or reduce retention of specialized scientific knowledge. These results collectively

[Figure 269]

/mnt/shared-storage-user/xuwanghan/projects/SuperSFE/SuperSFE/data/v7-3-深度搜索/ 物理/7_电路系统中的混沌行为/1/deep_research.json

[Figure 270]

Question: The experimental methodology for studying chaotic hysteresis in Chua's circuit is employs a precision Chua's circuit setup with calibrated instrumentation to investigate chaotic hysteresis through step-by-step DC voltage variation and frequency-dependent triangular wave analysis, quantifying hysteresis loops and identifying critical frequency thresholds via phase space trajectory monitoring and time series bifurcation analysis. In the Chua circuit experiment, what are the calculated time constants (in μs) for the RC networks formed by a 10.2 nF capacitor C1 and the equivalent resistance, the peak-to-peak voltage (in V) range of the hysteresis loop at 0.01 Hz driving frequency, and the critical frequency (in Hz) where chaotic behavior ceases? Output the results in two decimal places, one decimal place, and integer format respectively, separated by commas.

User

[Figure 271]

[Figure 272]

- Step 1: Find paper Experimental observation of chaotic hysteresis in Chua's circuit driven by slow voltage forcing.

[Figure 273]

[Figure 274]

- Step 2: Identify RC network components from experimental setup: C1=10.2 nF, R1=219Ω. Calculate time constant: τ=R1×C1=219×10.2×10−9=2.2338μs≈2.23μs.

[Figure 275]

[Figure 276]

- Step 3: Voltage range determination: At 0.01 Hz triangular forcing, peak-to-peak voltage ΔV_T=3.2 V measured from hysteresis loop width in experimental phase portraits.

[Figure 277]

[Figure 278]

- Step 4: Critical frequency identification: "For f>10Hz the hysteresis phenomenon practically disappears" confirmed through frequency sweep experiments showing ΔV_T reduction from 3.2V (0.01Hz) to 0V (10Hz).

[Figure 279]

[Figure 280]

- Step 5: Validate measurement procedures: Hysteresis loops are measured by "changing DC voltage very slowly and step by step" while monitoring attractor transitions between single scroll and double scroll regimes.

[Figure 281]

[Figure 282]

- Step 6: Confirm data analysis techniques: Phase portraits and time series analysis confirm chaotic behavior through "bifurcations and dynamic attractor folding".

[Figure 283]

[Figure 284]

- Step 7: Integrate experimental specifications: The setup uses DAQ CB-68LPR with LabVIEW, Sony AFG 320 function generator, and ±15V power supply for TL084CN op-amps, ensuring accurate voltage measurements.

Final Answer: 2.23, 3.2, 10

- Figure 11 | Scientific Deep Research Case: Example multi-hop workflow illustrating data retrieval, evidence synthesis, and quantitative analysis.

highlight the current limitations of frontier AI systems in executing the multi-faceted and rigorously structured reasoning processes required for Scientific Deep Research.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

- Figure 12 | Scientific Deep Research Evaluation of LLMs: Exact Match (EM) and Step-Level Accuracy (SLA) across models using scientist-aligned metrics.

Most models exhibit substantially lower performance on the Data and Properties tasks, but somewhat better—though still modestly—on Micro- and Macro-experiment tasks. Based on the focus of each question, we categorize the tasks into four types: Data, Properties, Micro-experiments, and Macro-experiments (Table 1). Figure 14 summarizes the performance of LLMs and agents across these categories. Notably, performance across all four categories rarely exceeds 30% (with only a few Macro cases slightly above), underscoring the intrinsic difficulty of scientific deep research. This disparity can be attributed to the nature of the information required. Data- and property-related questions often rely on detailed numerical specifications or contextual descriptions scattered across disparate sources in the literature, demanding precise retrieval, cross-referencing, and aggregation.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

###### Figure 13 | Scientific Deep Research Evaluation of Multi-Agent Systems: EM and SLA for toolaugmented agent systems.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### Figure 14 | Scientific Deep Research Performance by Type: Comparison across Data, Properties, Micro-Experiments, and Macro-Experiments categories.

In contrast, Micro- and Macro-experiment tasks tend to provide more structured protocols or clearer experimental outcomes, enabling LLMs and agents to reason with fewer retrieval uncertainties.

In summary, the relatively stronger model performance on experiment-oriented tasks suggests that recent advances in LLM pretraining and instruction tuning have enhanced models’ abilities to process structured procedures and numerical patterns. Nevertheless, the consistently low scores across all categories indicate that contemporary LLMs, even when augmented with tool-based agents, remain far from mastering the breadth and depth of reasoning required for robust scientific deep research.

- 4.4. Idea Generation

###### Figure 15 illustrates the evaluation pipeline for Idea Generation in SGI-Bench, and more experimental details can be found in the section 2.2.2. Table 6 shows the quantitative experimental results of idea generation, including effectiveness, novelty, detailedness, and feasibility. We could see that GPT-5 achieves the best average performance, and achieves the best performance in three aspects only excluding the feasibility. Moreover, across models, a clear pattern emerges: Novelty is generally high, especially among closed-source systems (e.g., o3 73.74, GPT-5 76.08). This indicates that modern LLMs possess a robust capacity for generating conceptually novel scientific ideas. Such behavior aligns with the growing empirical use of LLMs as inspiration engines for scientific hypothesis generation and exploratory research.

[Figure 289]

Related Work: Paulus et al. (2017) Utilized combinations of probe signals for phaseless near-field ...

... Objective: To rigorously analyze and experimentally validate the two-scans phaseless spherical near-field technique, quantifying the impact of measurement parameters and AUT characteristics, and to synthesize practical guidelines for accurate phase retrieval and radiation pattern reconstruction without phase information. User

[Figure 290]

[Figure 291]

###### Structured Idea

[Figure 292]

[Figure 293]

Data: A synthetic dataset of phaseless spherical near-field amplitudes and corresponding complex near-/far-fields …

Metrics: FarFieldError (Normalized root-meansquare error and maximum absolute …)

[Figure 294]

[Figure 295]

[Figure 296]

Implementation Graph

[Figure 297]

[Figure 298]

Formulate a differentiable spherical near-field forward operator from equivalent

1 currents to complex field on arbitrary spheres, including AUT offsets and asymmetries.

[Figure 299]

[Figure 300]

Design a Bayesian deep phase prior (e.g., conditional generative model) trained on

2 synthetic AUT current/field libraries spanning high-gain, asymmetric, and offset antennas.

[Figure 301]

[Figure 302]

[Figure 303]

Integrate the forward operator and phase prior into a physics-informed unrolled 3 optimizer (e.g., unrolled Wirtinger / gradient descent) for amplitude-only data.

4 …

[Figure 304]

[Figure 305]

Embed measurement-configuration parameters (sphere radii, angular sampling, AUT offset) 5 as optimizable or sweepable variables within the framework.

…

- Figure 15 | Idea Generation Case: Input information such as related work and objective, and output a structured idea, including a graph consisting of specific implementation steps.

Mechanistically, this strength likely stems from their broad pretraining over heterogeneous scientific corpora, which enables them to recombine distant concepts across domains, as well as their ability to internalize high-level research patterns (problem–method–evaluation triples). As a result, LLMs are particularly effective at proposing plausible and novel conceptual directions, often exceeding what a single human researcher can enumerate in a short time window.

Novelty is relatively high while feasibility lags. In contrast, Effectiveness is modest for most models and Feasibility consistently lags behind the other dimensions. Even the best-performing GPT-5, which achieves high Detailedness (85.72) and the highest Average (55.40), attains only scores 18.87 in Feasibility, confirming that conceptual richness does not reliably translate into implementationready plans. The top Feasibility model by our metric is o3 (22.90), while open-source feasibility peaks at Qwen3-8B (20.58); other models cluster in the 14–20 range. Open-source models exhibit the same trend: Kimi-k2 reaches higher Detailedness (59.20) but remains limited in Feasibility (18.74); similarly, Qwen3-VL-235B-A22B reaches only 20.14 in Feasibility despite substantially higher conceptual elaboration (50.23).

Execution details are often underspecified. These outcomes reveal a realization bottleneck in current idea generation: While models can articulate sophisticated pipelines at a high level, they frequently omit or under-specify key executable details. Typical failure issues include: (i) data references without acquisition or preprocessing plans; (ii) training and optimization loops that omit concrete hyperparameters or resource assumptions; (iii) algorithmic modules named but not grounded in precise choices (e.g., solver type, training objective, evaluation protocol); (iv) integration steps that fail to specify interfaces, ordering, or data flow. Consequently, many proposals fail feasibility checks not because they are conceptually unsound, but because they rely on implicit, unparameterized execution assumptions that cannot be validated under realistic experimental conditions. This gap highlights a fundamental limitation of current LLMs: they excel at linguistic and conceptual abstraction, yet

Model Effectiveness Novelty Detailedness Feasibility Average Open-source LLM

DeepSeek-V3.2 28.09 54.09 47.34 20.28 37.45 DeepSeek-R1 27.73 63.64 50.06 19.20 40.16 Intern-S1 26.38 56.47 49.10 20.42 38.09 Intern-S1-mini 24.95 55.71 48.07 15.44 36.04 Kimi-k2 25.24 69.49 59.20 18.74 43.17 Qwen3-VL-235B-A22B 27.24 59.53 50.23 20.14 39.28 Qwen3-235B-A22B 26.63 62.05 49.73 19.40 39.45 Qwen3-Max 28.74 59.01 50.61 20.98 39.83 Qwen3-8B 26.12 49.36 47.09 20.58 35.78 Llama-4-Scout 28.50 33.25 43.08 14.06 29.72

Closed-source LLM

- GPT-4o 27.28 48.19 47.85 20.51 35.95

- GPT-4.1 27.49 48.72 47.88 21.87 36.49

GPT-5 40.92 76.08 85.72 18.87 55.40

- GPT-5.1 36.07 66.98 66.62 18.83 47.12 GPT-5.2-Pro 51.36 71.19 78.03 19.53 55.03

- o3 29.42 73.74 58.22 22.90 46.07

- o4-mini 27.26 63.33 50.53 22.01 40.78 Gemini-2.5-Flash 28.45 56.91 50.49 20.69 39.13 Gemini-2.5-Pro 30.98 57.54 52.21 19.06 39.95 Gemini-3-Pro 28.38 59.41 51.07 19.87 39.68 Claude-Opus-4.1 26.52 64.40 50.16 20.07 40.29 Claude-Sonnet-4.5 32.01 58.00 61.75 21.03 43.20

- Grok-3 28.37 46.27 48.35 20.93 35.98
- Grok-4 28.46 50.93 49.48 19.60 37.12

- Table 6 | Idea Generation Results: The ideas generated by the model outperformed the average proportion of the original papers in the four dimensions of Effectiveness, Novelty, Detailedness, and Feasibility.

struggle with the procedural, resource-aware, and constraint-grounded planning required for real scientific implementation.

Overall, the Idea Generation results indicate that contemporary LLMs are adept at proposing novel directions but struggle to turn them into fully executable plans. Bridging this gap will require constraintaware planning, stronger priors over experimental and engineering practice, tool-augmented verification (e.g., property simulators, data/API discovery, and reproducibility scaffolds), and training signals that reward concrete, parameterized, and testable implementation steps rather than stylistic innovation.

###### 4.5. Dry/Wet Experiment

Experiments form the critical bridge between idea generation and scientific reasoning, providing the most direct avenue for validating hypotheses and uncovering new phenomena. Within SGI-Bench, we evaluate two complementary forms of experiments: dry experiments, which involve computational analyses or simulations, and wet experiments, which require laboratory procedures and operational planning. Across both categories, current AI models exhibit substantial limitations, revealing a persistent gap between linguistic fluency and experimentally actionable competence.

OneDark Pro 7号，间距8

[Figure 308]

[Figure 309]

|[Figure 310]<br><br>Accurate modeling of electronic excited states in quantum systems is essential for understanding phenomena in photocatalysis, fluorescence, photovoltaics, and condensed matter physics. Excited states are more challenging to compute than ground states due to their complex nature and the limitations of existing quantum chemistry methods, which often require prior knowledge or involve parameter tuning. Variational Monte Carlo (VMC) combined with neural network wave function ansatze has recently achieved high accuracy for ground states but<br><br>has faced difficulties extending to excited states…<br><br>Background<br><br>[Figure 311]|
|---|

|[Figure 312]<br><br>Answer<br><br>[Figure 313]<br><br>... def construct_trial_wavefunction(positions, params, state_index, wavefunction_type='gaussian'):<br><br>"""Construct trial wavefunction for a single quantum state. Tag: [Numerical calculation]<br><br>Args: positions (np.ndarray): Electron positions (N, 3) params (dict): Wavefunction parameters<br><br>state_index (int): Index of the quantum state<br><br>wavefunction_type (str): Type of trial wavefunction Returns:<br><br>np.ndarray: Wavefunction values at positions<br><br>Examples: >>> psi = construct_trial_wavefunction(pos, params, 0) >>> print(psi.shape)<br><br>""" epsilon = 1e-10<br><br>if wavefunction_type == 'gaussian': # Simple Gaussian type trial wavefunction center = params[f'center_{state_index}'] width = params[f'width_{state_index}'] amplitude = params[f'amplitude_{state_index}'] # Calculate distance to center r_squared = np.sum((positions - center) ** 2, axis=-1) # Gaussian wavefunction psi = amplitude * np.exp(-r_squared / (2 * width ** 2 + epsilon))<br><br>elif wavefunction_type == 'slater': # Slater type trial wavefunction alpha = params[f'alpha_{state_index}'] r = np.linalg.norm(positions, axis=-1) + epsilon # Different excited states use different radial functions if state_index == 0: # Ground state<br><br>psi = np.exp(-alpha * r) elif state_index == 1: # First excited state<br><br>psi = r * np.exp(-alpha * r / 2) else: # Higher excited states<br><br>psi = r ** (state_index) * np.exp(-alpha * r / (state_index + 1)) return psi<br><br>...|
|---|

|[Figure 314]<br><br>Data Code<br><br>[Figure 315]<br><br>import numpy as np np.random.seed(42) N = 5<br><br>positions = np.random.rand(N, 3)<br><br>...|
|---|

|[Figure 316]<br><br>Main Code with Incomplete Functions<br><br>[Figure 317]<br><br>... def construct_trial_wavefunction(positions, params, state_index, wavefunction_type='gaussian'):<br><br>"""Construct trial wavefunction for a single quantum state. Tag: [Numerical calculation]<br><br>Args: positions (np.ndarray): Electron positions (N, 3) params (dict): Wavefunction parameters state_index (int): Index of the quantum state wavefunction_type (str): Type of trial wavefunction<br><br>Returns: np.ndarray: Wavefunction values at positions<br><br>Examples: >>> psi = construct_trial_wavefunction(pos, params, 0) >>> print(psi.shape)<br><br>""" pass # Please complete the function based on the context.<br><br>...|
|---|

- Figure 16 | Dry Experiment Code Examples: Masked-function completion setup with I/O formats, and functional descriptions.

###### 4.5.1. Dry Experiment

As introduced in Section 2.1.3, each dry experiment contains three components: a description of scientific background, a complete data-construction script, and an analysis script with masked functions. The model must infer and complete these missing functions using contextual understanding. For fairness and structural clarity, function headers, including names, signatures, and functional descriptions, are preserved, as shown in Figure 16. This setup isolates the model’s ability to infer algorithmic logic rather than boilerplate structure.

- Table 7 summarizes three metrics defined in Section 2.2.3: PassAll@k, Average Execution Time (AET), and Smooth Execution Rate (SER). Here, PassAll@k denotes passing at least 𝑘 out of five unit tests per problem. Under the lenient criterion (𝑘=1), the best models achieve a PassAll@1 score of 42.07%, whereas the strictest requirement (𝑘=5) reduces performance to 36.64%. These results underscore that scientific code completion remains a significant bottleneck, even for frontier LLMs. Notably, closed-source models generally achieve higher PassAll@k than leading open-source models, though the advantage is modest and distributions overlap, suggesting that scientific code synthesis in dry experiments remains underdeveloped across architectures.

High execution rates do not guarantee correctness. The SER metric captures whether the generated code executes without error, independent of correctness. While many top models achieve high SER values (>90%), performance varies widely across systems; several models are substantially below this threshold (e.g., Gemini-2.5-Flash/Pro, Qwen3-8B, Llama-4-Scout, GPT-5, GPT-4o), indicating nontrivial robustness gaps. This suggests that basic structural and API-level reasoning has matured for some models; however, the persistent gap between SER and accuracy metrics highlights that structural validity is far easier than algorithmic correctness in scientific contexts.

Numerical and simulation functions are the most challenging. Figure 17 breaks down PassAll@5 across functional types. Models perform relatively well on Data Processing and Predictive Modeling, where multiple valid implementations exist and errors are less amplified. In contrast, Numerical Calculation and simulation-oriented functions prove substantially more difficult. These tasks typically require precise numerical stability, accurate discretization, or careful handling of domain-specific

Model PassAll@5(%)↑ PassAll@3(%)↑ PassAll@1(%)↑ AET(s)↓ SER(%)↑ Open-source LLM

DeepSeek-V3.2 23.62 26.94 29.52 29.96 68.27 DeepSeek-R1 33.33 35.56 37.41 28.09 91.70 Intern-S1 28.79 31.44 34.09 31.04 87.58 Intern-S1-mini 16.97 17.34 18.08 14.55 79.83 Kimi-k2 29.52 32.10 36.16 33.42 90.26 Qwen3-VL-235B-A22B 28.41 31.37 33.58 32.74 91.22 Qwen3-235B-A22B 28.89 31.48 34.44 30.68 90.81 Qwen3-Max 33.21 35.42 37.27 35.25 90.33 Qwen3-8B 18.45 20.30 21.03 21.13 71.51 Llama-4-Scout 20.37 21.48 22.59 24.24 68.52

Closed-source LLM GPT-4o 26.94 29.89 32.10 37.90 79.78

- GPT-4.1 34.32 37.64 40.22 40.54 94.10 GPT-5 29.89 32.84 34.69 34.54 75.50
- GPT-5.1 31.00 35.42 38.01 23.87 96.53 GPT-5.2-Pro 28.04 33.21 39.48 23.73 96.60

- o3 31.73 34.32 37.64 34.06 85.17
- o4-mini 35.79 39.11 41.70 31.34 87.60 Gemini-2.5-Flash 21.03 22.51 24.72 15.09 44.65 Gemini-2.5-Pro 22.51 23.99 24.72 13.94 44.65 Gemini-3-Pro 36.64 40.46 41.98 21.16 98.85 Claude-Opus-4.1 34.69 37.27 40.59 31.67 94.32 Claude-Sonnet-4.5 35.79 38.75 42.07 31.59 94.83 Grok-3 27.31 29.15 32.10 35.30 91.22 Grok-4 33.71 37.12 40.53 33.74 94.09

Table 7 | Dry Experiment Metrics Across Models: PassAll@k, Average Execution Time (AET), and Smooth Execution Rate (SER) under five unit tests per problem.

- Figure 17 | PassAll@5 by Function Category: Completion accuracy across numerical calculation, statistical analysis, simulation, metric calculation, data processing, and predictive modeling.

constraints, all of which amplify small reasoning inconsistencies. This pattern reveals a striking asymmetry: models exhibit reasonable flexibility in tasks with diverse valid outputs but struggle with tasks requiring exact numerical fidelity.

Methodological choices critically affect outcomes. The case shown in Figure 18 illustrates this issue in an astronomical dry experiment involving the computation of gravitational-wave observables from LIGO/Virgo–like detectors. The o4-mini model employs a naïve numerical integration via

矢量版： OneDark Pro 7号，间距8 天文\1_引力波检测与参数估计(GW Detection & Parameter Estimation)\5\干实验\main_en_test_1_incomplete_function.py

[Figure 320]

[Figure 321]

|[Figure 322]<br><br>GPT 4.1 Response<br><br>... def comoving_volume(z_max: float, H0_val: float, omega_m_val: float, omega_l_val: float) -> float:<br><br>""" Calculates the comoving volume up to a maximum redshift z_max. Tag: [Numerical calculation]<br><br>Args:<br><br>z_max (float): The maximum redshift to integrate to. H0_val (float): The Hubble constant in km/s/Mpc.<br><br>omega_m_val (float): The matter density parameter. omega_l_val (float): The dark energy density parameter.<br><br>Returns: float: The comoving volume in Gpc^3.<br><br>""" # Define the integrand for comoving distance integral def E(z):<br><br>return np.sqrt(omega_m_val * (1 + z)**3 + omega_l_val) def integrand(z):<br><br>return 1.0 / E(z)<br><br># Speed of light in km/s c_km_s = C<br><br># Integrate to get comoving distance in Mpc integral, _ = integrate.quad(integrand, 0, z_max, epsabs=1e-8, epsrel=1e-8)<br><br>d_c = (c_km_s / H0_val) * integral # in Mpc<br><br># Comoving volume in Mpc^3 (assuming flat universe) volume_mpc3 = (4.0 / 3.0) * np.pi * d_c**3<br><br># Convert to Gpc^3 volume_gpc3 = volume_mpc3 / 1e9<br><br>return volume_gpc3<br><br>...<br><br>[Figure 323]<br><br>High precision<br><br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>53<br>54<br>55<br>56<br>57<br>58<br>59<br>60<br>61<br>|
|---|

|[Figure 324]<br><br>O4-mini Response<br><br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>53<br>54<br>55<br>56<br>57<br>58<br>59<br>60<br>61<br><br><br>... def comoving_volume(z_max: float, H0_val: float, omega_m_val: float, omega_l_val: float) -> float:<br><br>""" Calculates the comoving volume up to a maximum redshift z_max. Tag: [Numerical calculation]<br><br>Args:<br><br>z_max (float): The maximum redshift to integrate to. H0_val (float): The Hubble constant in km/s/Mpc.<br><br>omega_m_val (float): The matter density parameter. omega_l_val (float): The dark energy density parameter.<br><br>Returns: float: The comoving volume in Gpc^3.<br><br>""" # Set up redshift grid nz = 2000 z = np.linspace(0.0, z_max, nz) dz = z[1] - z[0]<br><br># E(z) = sqrt(Ω_M (1+z)^3 + Ω_L) for flat universe E = np.sqrt(omega_m_val * (1.0 + z)**3 + omega_l_val) # Comoving distance element dχ/dz = c / H0 / E(z) in Mpc dchi_dz = (C / H0_val) / E # C [km/s], H0 [km/s/Mpc] ⇒ result in Mpc<br><br># Cumulative comoving distance χ(z) via simple trapezoidal integration chi = np.cumsum(dchi_dz) * dz # Ensure χ(0)=0<br><br>chi[0] = 0.0<br><br># Differential comoving volume element dV/dz = 4π χ(z)^2 * dχ/dz dV_dz = 4.0 * np.pi * chi**2 * dchi_dz # in Mpc^3 per unit z<br><br># Integrate over redshift to get volume V_mpc3 = np.trapz(dV_dz, z) # in Mpc^3 # Convert to Gpc^3<br><br>V_gpc3 = V_mpc3 / 1e9 return V_gpc3<br><br>...<br><br>[Figure 325]<br><br>Low precision|
|---|

[Figure 326]

Experimental Procedure

comoving_volume

load_data power_law_with_smoothing gaussian_component

merger_rate_evolution

estimate_merger_rate

primary_mass_distribution

- Figure 18 | Dry Experiment Case Study: Gravitational-wave computation highlighting the impact of numerical integration strategy on scientific outcomes.

np.cumsum, effectively using a forward Euler approximation for

𝜒(𝑧) = ∫ 𝑧

𝑑𝜒 𝑑𝑧

𝑑𝑧,

0

which introduces substantial cumulative error when the discretization is coarse. In contrast, GPT-4.1 correctly adopts scipy.integrate.quad, leveraging adaptive integration schemes that preserve numerical precision. Because errors in 𝜒(𝑧) propagate directly to the comoving volume element

𝑑𝑉 𝑑𝑧

𝑑𝜒 𝑑𝑧

= 4𝜋𝜒(𝑧)2

,

the flawed integration strategy in o4-mini leads to a significant deviation in the final volume estimate 𝑉Gpc3. This example highlights a broader challenge: LLMs often fail to capture the numerical sensitivity and methodological nuance essential for scientific computation.

Overall, these findings reveal that while current models can generate syntactically valid code with high reliability, their deeper limitations stem from (i) incomplete numerical reasoning, (ii) superficial understanding of scientific algorithms, and (iii) the inability to select appropriate computational strategies under domain constraints. AI-assisted scientific experimentation thus remains a demanding frontier, requiring future models to incorporate domain-aware numerical reasoning, fine-grained algorithmic priors, and training signals beyond natural-language supervision.

###### 4.5.2. Wet Experiment

For wet experiments, we provide models with an action pool containing standardized experimental operations and detailed descriptions. Given the experimental context, the model is required to synthesize a complete workflow, including both the selection and ordering of actions as well as all associated parameters (Figure 19). As illustrated in the figure, the model outputs typically exhibit two major categories of errors: (i) incorrect ordering of experimental steps and (ii) inaccurate or inconsistent parameter specification.

- (a) Correct Experimental Procedure

[Figure 329]

- (b) Incorrect Sequence of Actions
- (c) Action Parameter Error

###### Action Pool

Perform whole exome sequencing

| | |
|---|---|
| | |
| | |

S

T

###### R …

Patient001

P

D

DNA sample

L

E

T

R

E

Surgical resection

C

S

Raw sequencing data

Sequencing platform

QIAamp DNA/RNA Kit Illumina HiSeq hg38

Analyze sequencing data

Sequencing data

[Figure 330]

[Figure 331]

List of identified mutations

| | |
|---|---|
| | |
| | |

S

D

Patient001

P

T

###### E …

L

R

T

Reference genome

R

S

C

Surgical resection

E

Collect tumor biopsy

QIAamp DNA/RNA Kit Illumina HiSeq hg38

Patient

Tumor tissue sample

Collection method

QIAamp DNA/RNA Kit

Extract nucleic acids

[Figure 332]

| | |
|---|---|
| | |
| | |

S

T

Patient001

P

R …

D

Tissue sample

L

E

T

Extracted DNA and RNA samples

[Figure 333]

R

E

C

Surgical resection

S

Extraction kit

…

[Figure 334]

[Figure 335]

hg38 Illumina HiSeq

- Figure 19 | Wet Experiment Workflow: Action-pool based protocol construction with typical errors in step sequencing and parameter specification.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

- Figure 20 | Wet Experiment Evaluation: Sequence Similarity (SS) and Parameter Accuracy (PA) across models for laboratory protocol planning.

Wet experiments reasoning remains brittle. Figure 20 summarizes performance in terms of sequence similarity (SS) and parameter accuracy (PA). For SS, closed-source models in general achieve higher scores than open-source ones (with the best closed-source model around 35.5 versus the best open-source below 30), yet SS remains uniformly low across all systems. In contrast, PA exhibits a mixed pattern: although the top result is obtained by a closed-source model (around 40.6), several open-source models are competitive, and some closed-source models drop markedly (e.g., near 20.7). PA appears slightly more optimistic also since permutation-equivalent parameter groups are treated as identical (e.g., ⟨action 1⟩(𝐵, 𝐶) and ⟨action 1⟩(𝑋,𝑌) are identical when 𝐵=𝑋 and 𝐶=𝑌), but both families still achieve only modest scores. Across outputs, errors recur in three patterns: insertion of unnecessary steps, omission of essential steps, and incorrect ordering of valid steps.

Temporal and branch-aware planning is often broken. Figure 21 presents an experiment examining how tumor mutational burden and neoantigen load influence the efficacy of anti–PD-1 immunotherapy in non–small cell lung cancer. The ground-truth workflow (Figure 21 a) features a deeply branched structure with precisely coordinated timing and sample-handling procedures. In contrast, the workflow generated by o4-mini is substantially simplified and deviates from several core principles of experimental design.

(a) Correct Experimental Procedure

|Isolate peripheral blood mononuclear cells|
|---|

|Collect patient blood samples|
|---|

|Collect patient blood samples|
|---|

|Perform intracellular cytokine staining|
|---|

|Isolate peripheral blood mononuclear cells| |
|---|---|
| | |

|Stain T cells with multimers| |
|---|---|
| | |

|Perform flow cytometry|
|---|

|Perform whole exome sequencing| |
|---|---|
| | |

|Extract tumor tissue| |
|---|---|
| | |

|Extract genomic DNA| |
|---|---|
| | |

|Align sequencing reads| |
|---|---|
| | |

|Predict neoantigens| |
|---|---|
| | |

|Synthesize peptides| |
|---|---|
| | |

|Create pMHC multimers|
|---|

|Call somatic mutations|
|---|

|Collect patient blood samples|
|---|

|Perform whole exome sequencing| |
|---|---|
| | |

|Align sequencing reads| |
|---|---|
| | |

|Isolate peripheral blood mononuclear cells| |
|---|---|
| | |

|Extract genomic DNA|
|---|

|Predict HLA alleles| |
|---|---|
| | |

|Correlate mutation burden with clinical response|
|---|

|Isolate peripheral blood mononuclear cells| |
|---|---|
| | |

|Stain T cells with multimers| |
|---|---|
| | |

|Perform flow cytometry| |
|---|---|
| | |

|Analyze T cell responses|
|---|

[Figure 338]

(b) o4-mini Experimental Procedure

|Perform whole exome sequencing| |
|---|---|
| | |

|Extract tumor tissue| |
|---|---|
| | |

|Extract genomic DNA| |
|---|---|
| | |

|Align sequencing reads| |
|---|---|
| | |

|Call somatic mutations|
|---|

|Predict neoantigens| |
|---|---|
| | |

|Synthesize peptides| |
|---|---|
| | |

|Create pMHC multimers|
|---|

|Perform whole exome sequencing| |
|---|---|
| | |

|Align sequencing reads| |
|---|---|
| | |

|Extract genomic DNA|
|---|

|Correlate mutation burden with clinical response|
|---|

|Predict HLA alleles| |
|---|---|
| | |

|Collect patient blood samples|
|---|

|Stain T cells with multimers| |
|---|---|
| | |

|Perform flow cytometry| |
|---|---|
| | |

|Analyze T cell responses|
|---|

|Collect patient blood samples| |
|---|---|
| | |

|Isolate peripheral blood mononuclear cells|
|---|

|Perform intracellular cytokine staining|
|---|

- Figure 21 | Wet Experiment Case Study: NSCLC anti–PD-1 immunotherapy workflow—ground-truth protocol versus model-generated design.

First, the model collapses longitudinal sampling into a single blood draw and does not distinguish time windows, precluding any meaningful reconstruction of T-cell dynamics. Second, PBMC isolation is executed only once rather than per time point, causing misalignment with downstream staining and flow cytometry. Functional assays (e.g., intracellular cytokine staining) are performed on a single PBMC aliquot without branching by time point or antigenic stimulation, and flow cytometry is likewise conducted only once, failing to capture temporal variation. Finally, the blood-sample branch conflates genomic and immunophenotyping workflows: “Extract genomic DNA” is executed in parallel with PBMC isolation and downstream immunology, leading to duplicated and cross-purpose use of peripheral blood. These design flaws mirror the low sequence similarity and only moderate parameter accuracy observed in Figure 20, underscoring failures in temporal coordination, branchaware planning, and sample bookkeeping.

Overall, the deviations highlight a critical limitation of current AI models: while they can enumerate plausible wet experiment actions, they struggle to construct experimentally valid, temporally consistent, and branch-aware protocols. These limitations point to fundamental gaps in reasoning about experimental constraints, biological timing, and multi-sample coordination—elements essential for real-world scientific experimentation.

###### 4.6. Experimental Reasoning

Experimental Reasoning evaluates the ability of multimodal LLMs to interpret experimental observations, integrate heterogeneous scientific evidence, and refine testable hypotheses. As illustrated in Figure 22, the visual inputs span five representative modalities in scientific practice—process diagrams, data visualizations, natural observations, numerical simulations, and laboratory experiments—reflecting the diversity of multimodal information that underpins real-world scientific inquiry.

In this task, models are provided with several images accompanied by a question and must select the correct answer from at least ten candidates (Figure 23). Solving these problems requires multistep inferential reasoning: identifying relevant variables, synthesizing multimodal cues, evaluating

[Figure 341]

|[Figure 342]<br><br>Visualization<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]|
|---|

|[Figure 347]<br><br>Observation<br><br>[Figure 348]<br><br>[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]|
|---|

Process

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

|[Figure 358]<br><br>Simulation<br><br>[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>[Figure 363]<br><br>[Figure 364]|
|---|

Experiment

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

- Figure 22 | Experimental Reasoning Modalities: Examples of process, visualization, observation, simulation, and experiment images used as multi-modal evidence.

competing hypotheses, and ultimately validating consistency across the provided evidence. We therefore evaluate model performance using both Multi-choice Accuracy and Reasoning Validity, the latter assessing whether the model’s explanation follows logically from the scientific evidence.

Reasoning validity often exceeds answer accuracy. As shown in Figure 24, closed-source LLMs generally outperform open-source counterparts on both metrics, with the best closed-source models achieving higher MCA (e.g., up to 41.9) and RV (e.g., up to 71.3) than the best open-source models (MCA 37.8, RV 52.3). However, several open-source models remain competitive with or exceed some closed-source systems in specific metrics (e.g., Qwen3-VL-235B-A22B RV 50.5 > GPT-4o RV 45.4), indicating nontrivial overlap. Most models score higher in Reasoning Validity than in Multi-choice Accuracy, suggesting that even when the final choice is incorrect, explanations often preserve partial logical coherence. Variance is moderate—particularly among closed-source models—while only a few models (e.g., Intern-S1-mini) show noticeably lower performance, pointing to the importance of scale for robust multimodal scientific reasoning.

Comparative reasoning is the most challenging across domains. To further dissect these capabilities, we analyze performance across reasoning types and disciplinary domains (Figure 25). From the perspective of reasoning categories, including signal perception, attribute understanding, comparative reasoning, and causal reasoning, LLMs perform consistently well in causal reasoning and perceptual recognition. In contrast, comparative reasoning emerges as a persistent weakness. This indicates that models struggle when required to contrast subtle quantitative or qualitative differences, a cognitive operation fundamental to scientific evaluation and hypothesis discrimination. When examining performance across 10 scientific disciplines, an intriguing pattern emerges. Models achieve their highest accuracy in astronomy, followed by chemistry, energy science, and neuroscience. These domains often feature structured visual patterns or canonical experimental setups, which may align well with LLMs’ prior training data. Conversely, performance declines substantially in materials science, life sciences, and Earth sciences, where visual cues are more heterogeneous, context-dependent, or experimentally nuanced. This divergence suggests that domain-specific complexity and representation diversity strongly influence multimodal reasoning performance.

[Figure 375]

Question: <img><img>Based on the integrated spectroscopic and computational evidence, what is the quantitative relationship between the Ir 5d orbital occupancy, Bader charge transfer, and the activation energy barrier for C–H bond cleavage in the direct formic acid oxidation pathway on Ir₁/CN?by a 10.2 nF capacitor C1 and the equivalent resistance, the peak-to-peak voltage (in V) range of the hysteresis loop at 0.01 Hz driving frequency, and the critical frequency (in Hz) where chaotic behavior ceases? Output the results in two decimal places, one decimal place, and integer format respectively, separated by commas.

[Figure 376]

[Figure 377]

Options:

- A. Increased 5d occupancy (↑0.25 e) with Bader charge +1.45|e| lowers C–H activation barrier to 0.48 eV via enhanced back-donation to COOH* transition state.
- B. Decreased 5d occupancy (↓0.18 e) with Bader charge +1.45|e| raises C–H activation barrier to 0.74 eV due to reduced σ-donation to HCOOH*.
- C. Constant 5d occupancy with Bader charge +0.20|e| maintains C–H barrier at 0.62 eV through balanced frontier orbital interactions.
- D. Oscillating 5d occupancy (±0.12 e) with Bader charge +1.45|e| modulates C–H barrier between 0.56-0.81 eV via dynamic orbital polarization.
- E. Increased 5d vacancy with Bader charge +1.45|e| lowers C–H barrier to 0.56 eV through enhanced σ-acceptor capability for C–H σ* orbital.
- F. Decreased 5d vacancy with Bader charge +0.20|e| raises C–H barrier to 0.83 eV due to weakened interaction with reaction intermediates.
- G. Quantized 5d orbital splitting with Bader charge +1.45|e| creates specific e_g orbital occupancy that lowers C–H barrier to 0.52 eV.
- H. 5d orbital degeneracy lifting with Bader charge +1.45|e| generates asymmetric charge distribution that raises C–H barrier to 0.77 eV.
- I. Cooperative 5d-2p orbital hybridization with Bader charge +1.45|e| enables barrierless C–H cleavage (0.12 eV) through concerted electron transfer.
- J. Competing 5d orbital filling and Bader charge redistribution (+1.45|e| → +0.80|e|) results in dual C–H barriers at 0.44 eV and 0.68 eV.

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

- Step 1: From the first image, analyze XANES white line area in 01.png: Quantify 5d vacancy through comparison with Ir⁰ and Ir⁴⁺ standards, showing intermediate occupancy.

[Figure 382]

[Figure 383]

- Step 2: Examine Bader charge analysis: Irδ+ (+1.45|e|) in Ir-N₄ vs Ir⁰ (+0.20|e|) in nanoparticles, confirming significant charge transfer.

Final Answer: E

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

User

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

- Step 3: From the second image, analyze DFT-calculated reaction energetics in 02.png: Extract C–H activation step energy (HCOOH* → COOH*

+ H⁺ + e⁻) = -0.74 eV → activation barrier ~0.56 eV.

[Figure 395]

[Figure 396]

- Step 4: Correlate electronic structure with kinetics: Increased 5d vacancy enhances σ-acceptor capability for C–H σ* orbital antibonding interaction.

[Figure 397]

[Figure 398]

- Step 5: Reference mechanistic insight: ’The d-orbital vacancy was therefore larger for Ir–N₄ than for Ir⁰, leading to smaller back-donation interaction and weaker bonding of Ir–N₄ and CO’.

[Figure 399]

[Figure 400]

- Step 6: Eliminate contradictory orbital occupancy scenarios. Option 0: Incorrect about increased 5d occupancy and back-donation enhancement. Option 1: Wrong direction of 5d occupancy change and barrier effect. Option 2: Contradicted by measured Bader charge difference.

[Figure 401]

[Figure 402]

- Step 7: Verify quantitative barrier values against DFT Gibbs free energy diagram. Confirm option 4 provides correct electronic structure rationale: increased 5d vacancy + appropriate Bader charge + correct barrier magnitude.

[Figure 403]

[Figure 404]

- Figure 23 | Experimental Reasoning Case: Multi-image question requiring cross-modal synthesis and step-wise reasoning.

Overall, these findings reveal that while current LLMs demonstrate encouraging abilities in integrating scientific evidence and conducting basic causal analyses, they still fall short in tasks requiring precise discrimination, cross-sample comparison, and nuanced interpretation of domain-specific observations. The relatively narrow performance gap among leading models underscores that scale alone is insufficient; advancing experimental reasoning will require improved multimodal grounding, finer-grained visual understanding, and training paradigms explicitly aligned with scientific inquiry.

##### 5. Analysis

###### 5.1. Test Time Reinforcement Learning

Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and problemsolving, primarily driven by supervised fine-tuning and reinforcement learning on extensive labeled datasets. However, applying these models to the frontier of scientific discovery, particularly in the task of scientific idea generation, presents a fundamental challenge: the inherent absence of ground truth. Unlike closed-domain tasks such as mathematical reasoning or code generation, where solutions can be verified against a correct answer, the generation of novel research ideas is an open-ended problem with no pre-existing “gold standard” labels. This limitation renders traditional offline training pipelines insufficient for adapting to dynamic and unexplored scientific territories.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

- Figure 24 | Experimental Reasoning Evaluation: Multi-Choice Accuracy (MCA) and Reasoning Validity (RV) across models on multimodal tasks.

| | |
|---|---|
| | |

- Figure 25 | Experimental Reasoning Performance by Type and Discipline: Breakdown across reasoning paradigms (signal, attribute, comparative, causal) and 10 scientific domains.

Consequently, the critical research question becomes: How can we enhance a model’s capability during the inference phase in the absence of ground-truth supervision? To address this, we adopt the paradigm of Test-Time Reinforcement Learning (TTRL) [20]. This framework enables models to self-evolve on unlabeled test data by optimizing policies against rule-based rewards derived from the model’s own outputs or environmental feedback. Distinct from the original implementation [20], which primarily leveraged consensus-based consistency as a reward mechanism for logical reasoning tasks, we establish novelty as our core optimization objective in the current context. Consequently, we introduce a TTRL framework where the reward signal is constructed based on the dissimilarity between generated ideas and retrieved related works, guiding the model to actively explore the solution space and maximize innovation at test time.

###### 5.1.1. Methodology

To address the absence of ground truth in scientific idea generation, we propose a generalizable reward mechanism based on online retrieval. Instead of relying on static labels, we utilize real-time search to fetch existing related works, serving as a dynamic baseline for comparison. This approach enables us to quantify novelty as the semantic dissimilarity between the model’s output and the retrieved context, effectively converting an open-ended exploration problem into a measurable optimization task. The overall training framework is illustrated in Figure 26.

We employ Group Relative Policy Optimization (GRPO) [1] as our training backbone. For a given query 𝑄, the policy model 𝜋𝜃 generates a group of 𝑘 outputs {𝑜1, . . . , 𝑜𝑘}. The optimization is guided

𝑨𝑨𝟏𝟏

𝒐𝒐𝟏𝟏

𝒓𝒓𝟏𝟏

[Figure 409]

Rewards

| | | | |
|---|---|---|---|
| | | | |

###### .

𝑨𝑨𝟐𝟐

𝒓𝒓𝟐𝟐

𝒐𝒐𝟐𝟐

Format

[Figure 410]

[Figure 411]

Q LLM

Group

[Figure 412]

### …

### …

### …

Computation

KL Idea Dissimilarity

𝑨𝑨𝒌𝒌

𝒐𝒐𝒌𝒌

𝒓𝒓𝒌𝒌

Reference Model

[Figure 413]

[Figure 414]

𝒘𝒘𝟏𝟏 𝒘𝒘𝟐𝟐 … 𝒘𝒘𝒏𝒏

[Figure 415]

[Figure 416]

Related Works

Web Search

- Figure 26 | TTRL Training Framework: The model generates candidate ideas evaluated against online retrieved related works to calculate novelty rewards, guiding GRPO updates.

by a composite reward function, defined as the unweighted sum of a format constraint and a novelty metric (labeled as Idea Dissimilarity in Figure 26):

𝑅(𝑜) = 𝑅format(𝑜) + 𝑅novelty(𝑜, W) (1) where W = {𝑤1, . . . , 𝑤𝑛} denotes the set of related works obtained via online search.

Format Reward (𝑅format). To guarantee interpretable reasoning, we enforce a strict XML structure. The model must encapsulate its chain of thought within <think>...</think> and the final proposal within <answer>...</answer>. The format reward is binary:

𝑅format(𝑜) = 𝕀 (𝑜 follows the specified XML structure) (2)

Novelty Reward (𝑅novelty). We quantify novelty by measuring the vector space dissimilarity between the generated idea and the retrieved literature. Let eidea be the embedding of the generated answer, and {e𝑤

𝑗}𝑛𝑗=1 be the embeddings of 𝑛 retrieved papers (denoted as 𝑤1, . . . , 𝑤𝑛 in the figure). We compute the average cosine similarity 𝑆avg:

###### ∑︁𝑛

eidea · e𝑤

1

(3)

𝑗

𝑆avg =

∥eidea∥∥e𝑤

𝑗∥

𝑛

𝑗=1

An innovation score 𝑆inn ∈ [0, 10] is then derived to reward divergence:

𝑆inn = clip (1 − 𝑆avg) × 10, 0, 10 (4) Using a gating threshold 𝜏 = 5, the final novelty reward is defined as:

𝑅novelty(𝑜, W) = 𝕀(𝑆inn > 𝜏) (5)

This mechanism incentivizes the model to produce ideas that are semantically distinct from existing work.

###### 5.1.2. Experimental Setup

We employ Qwen3-8B as the base model, trained using the GRPO algorithm within the ms-swift [77] framework. To facilitate diverse exploration, we utilize a high sampling temperature. Key hyperparameters are detailed in Table 8.

Table 8 | TTRL Hyperparameters: Key training configuration for GRPO-based test-time reinforcement learning.

###### Hyperparameter Value

Base Model Qwen3-8B RL Algorithm GRPO Precision bfloat16 Learning Rate 5 × 10−7 Max Length 2048 Generations (𝐺) 8 Temperature 1.0 Batch Size 4 Related Works (𝑛) 4 Weights 1:1

2.00

1.75

1.50

RewardValue

1.25

1.00

0.75

0.50

Total Reward

Format Reward (mean)

Idea Reward (mean)

0.25

0 200 400 600 800

Training Steps

Figure 27 | TTRL Training Dynamics: Format reward saturates quickly, followed by steady growth in idea novelty.

###### 5.1.3. Experimental Results

The training dynamics of our TTRL framework are illustrated in Figure 27. The curves demonstrate a clear two-phase optimization process. Initially, the Format Reward (orange line) rises rapidly and saturates near 1.0 within the first few steps, indicating that the model quickly adapts to the rigid XML structural constraints (<think> and <answer> tags). Once the format is stabilized, the Idea Reward also starts to rise (green line). Despite the inherent difficulty of the task, the Idea Reward exhibits a consistent upward trend throughout the training steps, driving the total reward (blue line) to converge at a higher value.

Quantitatively, this self-evolution process yields a significant improvement in the quality of generated ideas. The average novelty score of the model’s outputs increased from a baseline of 49.36 to 62.06. It is important to emphasize that this performance gain was achieved entirely without ground-truth labels. The model improved solely by leveraging the online retrieval feedback loop, validating the hypothesis that LLMs can self-improve on open-ended scientific discovery tasks through test-time reinforcement learning.

###### 5.1.4. Case Study of TTRL

To visually demonstrate the impact of TTRL on scientific idea generation, we present a comparative case study in Figure 28. The task requires the model to propose a novel framework for RNA 3D structure prediction.

|[Figure 419]<br><br>Before Post Training<br><br>We propose a hybrid RNA 3D prediction framework integrating evolutionary signals, secondary structure priors, and physical restraints via a transformer-based architecture. This approach combines contact map prediction with fragment assembly, leveraging DCAderived couplings and Rosetta energy functions to enhance sampling and accuracy for novel RNAs while providing reliable confidence scoring.|
|---|

|[Figure 420]<br><br>After Post Training<br><br>Propose a hybrid transformer-physical force field<br><br>framework integrating evolutionary couplings,<br><br>secondary structure priors, and physics-based scoring. The model uses a dual-branch transformer to decode sequence-structure relationships while a differentiable physics engine enforces base pairing and stacking constraints, enhanced by a confidence-aware uncertainty module for out-of-distribution detection.|
|---|

- Figure 28 | TTRL Case Study: Comparison of generated research ideas before and after TTRL, highlighting structural innovation (dual-branch transformer, differentiable physics engine) versus generic pre-training assembly.

Comparing the responses before and after training reveals a noticeable improvement in both specificity and novelty. The Pre-Training Response suggests a standard combination of existing components, essentially assembling "contact map prediction" with "Rosetta energy functions." While logical, it represents a conventional approach without distinct architectural details.

In contrast, the Post-Training Response introduces more structurally specific and technically distinct concepts. It explicitly proposes a "dual-branch transformer" and replaces static energy functions with a "differentiable physics engine." Additionally, it incorporates a "confidence-aware uncertainty module" to address the reliability challenge. This shift indicates that the model has moved beyond generic component assembly toward generating more detailed and differentiated technical proposals.

Summary. In conclusion, our experiments demonstrate that Test-Time Reinforcement Learning (TTRL), driven by retrieval-based novelty rewards, effectively enhances model capabilities in the absence of ground-truth supervision. The observed improvements in both quantitative novelty metrics and qualitative technical specificity indicate that the model can successfully self-evolve beyond conventional patterns. These findings suggest that TTRL is a promising paradigm for adapting Large Language Models to the open-ended and unexplored frontiers of real-world scientific discovery.

###### 5.2. Agent Tool Integrated Reasoning

###### 5.2.1. Retrieve–Browse Loop Analysis

Tool-Integrated Reasoning (TIR) in real tasks unfolds as a dynamic, opportunistic process rather than a fixed linear chain[78]. As shown in Figure 29 (left), the model-to-tool flow concentrates heavily on retrieval actions: web_search is the most frequently invoked tool with 539 calls (33.98% of all), followed by visit_webpage (385, 24.27%), final_answer (358, 22.57%), python_interpreter (200, 12.61%), and wikipedia_search (104, 6.56%). This distribution indicates that an external “retrieve-then-browse” loop remains the dominant path for contemporary agentic systems, reflecting persistent limits in time-sensitive and domain-specific knowledge available to base LLMs. Importantly, models differ in how efficiently they traverse this loop: for example, GPT-4.1 issues large volumes of web_search (168) and visit_webpage (110) that frequently land in slow tiers, whereas Qwen3-Max completes comparable coverage with far fewer retrieval and browsing steps (61 and 59, respectively). Practically, this pattern implies that reducing redundant retrieval iterations—via better query formulation and higher-quality extraction on the first pass—has immediate leverage on end-to-end latency, often exceeding gains from marginal improvements to raw model inference.

###### 5.2.2. Tool Efficiency Analysis

Latency variation is predominantly tool-dependent, as visualized in Figure 29 (right). The primary bottleneck is visit_webpage, whose cross-model latency spans from 5.37s (Llama-4-Scout) to 114.29s (GPT-4.1), a 21.28× spread. This reflects the intrinsic cost of browser-level execution—network I/O, DOM parsing, and event replay—rather than LLM reasoning alone. In contrast, more atomic operations such as wikipedia_search still exhibit a substantial 7.59× spread (3.69–28.03s), underscoring that I/O pathways and parsing routines meaningfully shape end-to-end time even for ostensibly simple tools. These observations suggest a design priority: engineering optimizations in the retrieval-and-browsing pipeline (e.g., smarter caching, incremental browsing, selective content extraction) will reduce both long-tail latencies and overall wall-clock time more reliably than tuning model-only parameters.

###### 5.2.3. Reasoning Cost Analysis

The python_interpreter tool exhibits a 9.65× cross-model range (5.48–52.94s), indicating that measurements capture the full “reason–execute–debug–repair” loop rather than a single code run. The slowest average arises for DeepSeek-R1 (52.94s), consistent with more frequent multi-step error analysis and correction; the fastest is GPT-4o (5.48s), reflecting a low-latency, near single-shot execution path. This divergence reveals a strategic trade-off: systems optimized for first-attempt correctness minimize tool time but may forgo deeper self-correction, whereas systems favoring iterative refinement accrue longer tool-side latency while potentially achieving more robust final solutions. In practice, aligning tool routing, retry policy, and verification depth with a model’s characteristic behavior can reduce wasted cycles and sharpen the latency–quality frontier.

| |
|---|

| |
|---|

| |
|---|

Gemini-2.5-Pro

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

Fast (<10s)

| |
|---|

final_answer

- 9.5s 5.2s 6.1s 23.3s 4.4s 7.1s 40.7s 14.4s 25.6s

14.3s 7.4s 10.5s 35.4s 6.6s 5.5s 52.9s 19.9s 17.2s

14.1s 12.4s 39.7s 36.1s 5.4s 21.3s 49.7s 16.0s 114.3s

- 10.5s 6.3s 14.4s 26.5s 5.8s 7.5s 13.4s 10.1s 40.9s
- 11.2s 11.8s 16.9s 17.7s 5.6s 3.7s 19.8s 3.8s 28.0s

Qwen3-Max

100s

web_search

| |
|---|

Gemini-2.5-Flash

| |
|---|

python_interpreter

80s

| |
|---|

| |
|---|

Qwen3-235B-A22B

60s

visit_webpage

visit_webpage

| |
|---|

Llama-4-Scout

Medium (10-30s)

40s

| |
|---|

| |
|---|

web_search

GPT-4o

final_answer

20s

| |
|---|

wikipedia_search

| |
|---|

DeepSeek-R1

| |
|---|

0s

python_interpreter

Gemini-2.5-ProQwen3-MaxGemini-2.5-FlashQwen3-235B-A22BLlama-4-Scout GPT-4oDeepSeek-R1Claude-Opus-4.1 GPT-4.1

Slow (>30s)

| |
|---|

| |
|---|

Claude-Opus-4.1

wikipedia_search

| |
|---|

GPT-4.1

- Figure 29 | Agent Tool Calls: Frequency (left) and efficiency (right) across leading models.

###### 5.3. SGIEvalAgent

###### 5.3.1. User-customized Metric

SGIEvalAgent interprets the user’s evaluation intent and turns it into a rubric that can be applied consistently across the selected idea-generation questions. In the case shown in Figure 30, the user asks to compare models on “rigor” in cross-disciplinary idea generation. The system formalizes Rigor (scientific strictness) for idea proposals so that it reflects how scientists judge whether a plan is internally coherent, well grounded, and practically verifiable.

The rubric expresses six aspects in prose rather than checklists. First, it checks logical self-consistency and completeness of the pipeline from problem to hypothesis, method, metrics, and expected results. Second, it requires theory and literature grounding that either correctly inherits prior work or responsibly challenges it with evidence. Third, it demands precise and verifiable problem definitions that state goals, constraints, evaluation metrics, and success conditions. Fourth, it looks for deep fit with the research background and correct, discipline-aligned terminology. Fifth, it evaluates methodological soundness and reproducibility through executable steps, a clear I/O loop, and explicit rationale for key design choices. Sixth, it considers risk awareness and scientific criticism by articulating assumptions, potential failure modes, bias sources, and avoiding over-confident conclusions. Major deductions apply when the reasoning chain is missing, key assumptions are unstated, terminology is misused, metrics are vague or non-verifiable, or inheritance from background knowledge is misaligned.

Scores are produced on a 0–10 scale for each aspect and aggregated with default equal weights

###### User

[Figure 427]

[Figure 428]

[Figure 429]

I want to know which model has the best rigor in idea generation across science disciplines.

###### Questioning Agent

[Figure 430]

[Figure 431]

[Figure 432]

Selected idea-generation questions across Physics, Chemistry, Biology, and seven other disciplines from SGIBench to reflect cross-disciplinary scientific ideation.

###### Customization Agent

[Figure 433]

[Figure 434]

[Figure 435]

User-customized metric: Rigor (scientific strictness). Scoring criteria: 1) Logical self-consistency and tight structure (Problem→Hypothesis→Method→Metrics→Expected result); 2) Solid theory & literature grounding (properly inherits/challenges prior work); 3) Precise, verifiable problem definition (clear goals/constraints/metrics/success conditions); 4) Deep integration with research background (terms used correctly and aligned with domain definitions); 5) Methodological soundness and reproducibility (executable steps, IO loop, rationale for design choices); 6) Risk awareness & scientific criticism (bias/assumption risks/failure modes, conservative conclusions). Major deductions: missing reasoning chain, unstated key assumptions, terminology misuse, vague metrics, misaligned inheritance of background.

Evaluation Agent

[Figure 436]

[Figure 437]

[Figure 438]

Running inference and scoring...

[Figure 439]

Reporting Agent

[Figure 440]

We evaluate models on the user-defined Rigor metric for idea generation. Observations from the chart:

[Figure 441]

- 1. Closed-source LLMs generally achieve higher rigor; top performers cluster around the GPT-5/o3/o4-mini families.
- 2. Intra-family improvements are evident (e.g., Qwen3-Max > Qwen3-235B-A22B; GPT-5 > GPT-4.1), indicating sensitive capture of technical iteration.
- 3. Open-source progress is notable: Qwen3-Max reaches parity with GPT-4.1 on rigor.
- 4. Heterogeneity persists: Grok-4 is comparable to GPT-4.1 and Qwen3-Max, underscoring multi-dimensional trade-offs across instruction following, creativity, and methodological soundness. Rigor foregrounds methodological soundness and internal coherence—models that score higher propose more structured, verifiable plans rather than purely fluent narratives.

Figure: Rigor of Ideas Generated by Different LLMs

- Figure 30 | SGIEvalAgent Case: Model Users describe their evaluation needs, SGIEvalAgent customizes the evaluation plan and metrics based on these needs, and finally provides an evaluation report..

into a single rigor score; the result is linearly mapped to a 0–100 axis for visualization without changing rank order. The evaluation agent generates textual rationales that cite reference answers and problem context so that decisions are transparent and reproducible. Customized metrics are reported alongside SGI-Bench’s predefined task metrics rather than replacing them, preserving standardized comparability while highlighting the user’s domain-specific focus.

###### 5.3.2. Automated Evaluation Report

The reporting agent summarizes the customized metric and the evaluation outputs into a concise narrative with figures. In Figure 30, the report contrasts open-source and closed-source systems on the user-defined rigor metric for idea generation and highlights what the scores mean in practice.

The core takeaway is straightforward: closed-source models generally exhibit higher rigor under this rubric, intra-family iterations capture measurable gains, and leading open-source models show notable progress that narrows the gap. Higher rigor reflects more structured, well-grounded, and verifiable research plans rather than merely fluent narratives. The report therefore gives users a clear, scientist-aligned comparison they can directly use for model selection and iteration in research workflows.

##### 6. Challenges and Future Directions

Grounded in our operational definition of SGI and instantiated through SGI-Bench, the evaluation results reveal a consistent message: contemporary LLMs and agentic systems exhibit localized scientific

cognition and segmented scientific reasoning They may solve isolated sub-problems, but fail to robustly close the iterative loop spanning Deliberation, Conception, Action, and Perception. Below we summarize the main limitations across tasks and disciplines, connect them with our TTRL and tool-integrated reasoning analyses, and outline concrete future directions.

###### 6.1. Fragmentation Across the Four Quadrants of SGI

Deliberation: Scientific Deep Research remains brittle end-to-end. Scientific Deep Research operationalizes the literature-review/meta-analysis stage and is evaluated by Exact Match (EM) and Step-Level Accuracy (SLA). Across both standalone LLMs and tool-augmented agents, EM is consistently low: most systems achieve only ∼10% accuracy, and even the best models rarely exceed 20% EM (Figure 12, Figure 13). This indicates that current models still fail to produce verifiable final scientific claims under multi-source evidence integration.

A notable gap exists between SLA and EM. SLA is substantially higher for nearly all systems, with several agentic systems reaching ≈50% SLA (Figure 13), while EM remains low. This disparity shows that models often produce locally correct steps but cannot maintain global coherence across long reasoning chains. The failure mode is therefore not mere knowledge absence, but reasoning trajectory collapse under long-horizon scientific inference.

At a finer granularity, Deep Research tasks involving Data and Properties are the weakest: performance on these categories is substantially below that of Micro- and Macro-experiment questions, with all four categories rarely exceeding 30% accuracy (Figure 14). This aligns with the task design: data/property questions require retrieving dispersed numerical details across heterogeneous papers, while experiment-oriented questions provide more structured evidence. The results thus expose a core SGI bottleneck: meta-analytic retrieval + numerical aggregation over scattered literature.

Conception: Ideas lack implementability. Idea Generation in SGI-Bench is assessed using Effectiveness, Detailedness, and Feasibility (Table 6). Feasibility is low across models: many systems score in the 14–20 range, and the best result reaches 22.90 (o3), indicating that feasibility consistently lags behind novelty and detailedness. Detailedness remains insufficient for several models, with implementation steps frequently missing concrete parameters, resource assumptions, or step ordering; Effectiveness is moderate for most systems, with the highest result of 51.36 (GPT-5.2-Pro) and open-source models clustering around 24.95–28.74 (e.g., DeepSeek-V3.2, Llama-4-Scout).

Recurring issues include: (i) underspecified implementation steps—absent data acquisition or preprocessing plans, missing hyperparameters or compute assumptions, vague module choices (e.g., solver type, training objective, evaluation protocol), and unclear interfaces, ordering, or data flow; and (ii) infeasible procedures—reliance on unavailable instruments or data, uncoordinated pipelines that cannot be executed, and designs lacking reproducibility.

In SGI terms, current systems exhibit fluent linguistic ideation without sufficient methodological execution grounding: they articulate concepts clearly but struggle to translate them into concrete, parameterized, and testable workflows. The feasibility gap observed in Table 6 is therefore a persistent bottleneck in realization, including within the Conception quadrant, where ideation quality does not reliably imply executable planning competence.

Action: Experimental execution is limited by numerical and procedural rigor. For Dry Experiments, accuracy is measured by PassAll@k. Even under the most lenient setting, the best PassAll@1 is only 42.07% (Claude-Sonnet-4.5), and under the strictest criterion, the best PassAll@5 rises to merely 36.64% (Gemini-3-Pro) (Table 7). The spread between PassAll@1 and PassAll@5 (e.g., 42.07→35.79

for Claude-Sonnet-4.5, 41.98→36.64 for Gemini-3-Pro) indicates that models often nail partial logic but fail full scientific correctness.

Importantly, code executability is not the bottleneck: most frontier models achieve SER > 90% (e.g., GPT-5.1 96.53, Gemini-3-Pro 98.85), while accuracy remains low. This gap confirms a central limitation: syntactic fluency ≠ scientific computational reasoning. The per-function analysis further shows numerical-calculation and simulation functions as the major failure mode (Figure 17), consistent with the case study (Figure 18) where naive integration choices lead to cascading scientific errors.

For Wet Experiments, although Parameter Accuracy improves slightly under permutation-equivalence evaluation, Sequence Similarity remains uniformly low across both open and closed models (Figure 20). Models frequently insert redundant steps, omit critical actions, or misorder multi-branch protocols. The complex oncology workflow case (Figure 21) illustrates that models cannot reliably manage temporal design, branching logic, or multi-sample coordination. Thus, wet-lab action planning remains a profound gap toward embodied SGI.

Perception: Multimodal reasoning is improving, but comparison is a hard frontier. In Experimental Reasoning, closed-source models consistently outperform open-source ones (Figure 24). Across nearly all models, Reasoning Validity (RV) exceeds Multi-choice Accuracy (MCA), showing that models can often produce partially coherent narratives even when selecting the wrong option. This echoes the SLA–EM gap in Deep Research and suggests a general pattern: models are better at producing plausible local reasoning than globally correct scientific decisions.

Reasoning-type breakdown reveals that models perform relatively well on Signal Perception and Causal Reasoning, but Comparative Reasoning is persistently weakest (Figure 25). Scientific comparison requires subtle cross-sample discrimination and quantitative contrast—a cognitive operation central to scientist judgment but not yet robustly captured by current MLLMs. Discipline-wise, astronomy and chemistry are easier, while materials science, life science, and Earth science remain hardest (Figure 25), reflecting the mismatch between real scientific visual heterogeneity and training priors.

###### 6.2. Implications from Test-Time RL and Tool-Integrated Reasoning

SGI as a dynamic, learnable capacity. Our TTRL experiments demonstrate that open-ended scientific ideation can improve without labeled supervision. With retrieval-based novelty rewards, Qwen3-8B increases its novelty score from 49.36 to 62.06 (Figure 27) and qualitatively progresses from generic component assembly to structured innovation (Figure 28). These results suggest that SGI should be interpreted not merely as a static benchmark score, but as a capability that can evolve through test-time learning. Nevertheless, optimizing for novelty in isolation risks ungrounded or implausible ideas; combining novelty with rigor- or feasibility-based rewards is a crucial next step for reliable scientific ideation.

The retrieval pipeline is the true bottleneck for agentic SGI. Tool-Integrated Reasoning (TIR) analysis reveals that agent workflows are heavily dominated by retrieval operations: web_search accounts for 539 calls (33.98%), and visit_webpage for 385 calls (24.27%) (Figure 29). Latency is primarily tool-driven rather than model-driven; visit_webpage exhibits a 5.37s–114.29s range across models (a 21.28× spread). This indicates that many gains in SGI performance may stem from smarter tool routing, reduction of redundant retrievals, and higher-quality first-pass extraction, rather than simply scaling base LLMs. Analysis of the Python tool further highlights a trade-off between first-shot correctness and iterative self-repair, with a 9.65× cross-model latency range, underscoring

the need for model-aware verification and retry policies in practical agentic workflows.

###### 6.3. Future Directions Toward Scientific General Intelligence Our findings point to several high-leverage research directions:

- (1) Meta-analytic reasoning with numerical robustness. Deep Research failures on Data/Properties and low EM despite high SLA call for methods that explicitly train evidence aggregation and numerical synthesis. Promising routes include retrieval-conditioned quantitative reasoning, uncertaintycalibrated aggregation over multiple sources, and verification-aware step planning that penalizes reasoning-chain drift.
- (2) Planning-aware conception and structured supervision. To address uniformly low feasibility and sparse implementation detail in Idea Generation, adopt planning-aware constraints with structured supervision: require parameter-complete, dependency-consistent steps, prioritize feasibilityfocused rewards (availability checks, resource/cost estimates, reproducibility), and use lightweight tool checks during decoding to block or repair incomplete plans. This shifts fluent proposals into executable, testable designs under realistic scientific constraints.
- (3) Scientific code training beyond syntax. Dry experiments show high SER but low PassAll@5 (Table 7), especially on numerical and simulation functions (Figure 17). Future work should emphasize numerical analysis priors, stability-aware loss, and algorithmic-choice training (e.g., recognizing when adaptive integration or stiffness solvers are required). Hybrid symbolic–numeric tool use (formal solvers + LLM reasoning) is another promising path.
- (4) Branch- and time-aware wet-lab protocol reasoning. Uniformly low Sequence Similarity and qualitative failures on complex branching protocols (Figure 21) suggest a need for training signals that encode temporal sampling logic, branching decision rules, and multi-sample tracking. Action-pool grounding can be extended with stateful simulators or lab-graph verifiers, enabling models to learn procedural validity under physical constraints.
- (5) Comparative multimodal scientific reasoning. Comparative reasoning is the hardest paradigm (Figure 25). Progress likely requires finer-grained visual grounding (e.g., numeric extraction from charts), cross-image alignment modules, and contrastive multimodal training that rewards precise discrimination rather than narrative plausibility. Discipline-specific multimodal curricula may reduce domain gaps in materials/Earth/life sciences.
- (6) Test-time learning with multi-objective scientific rewards. TTRL improves novelty without labels, but novelty alone is insufficient for SGI. Future TTRL systems should optimize a portfolio of scientist-aligned rewards (novelty, rigor, feasibility, safety, and experimental cost), and incorporate retrieval trustworthiness and contradiction penalties to prevent spurious innovation.
- (7) Efficient and reliable tool ecosystems for SGI agents. Given retrieval dominance and tool latency (Figure 29), engineering advances are essential: retrieval caching, selective browsing, structured extraction, and tool-aware planning policies can substantially improve SGI agents’ end-to-end quality–latency frontier.

Summary. SGI-Bench reveals that modern LLMs exhibit partial competencies in each SGI quadrant but lack integrated, numerically robust, and methodologically disciplined scientific cognition. Bridging this gap requires progress on long-horizon meta-analysis, executable planning, numerically faithful experimentation, branch-aware wet-lab reasoning, comparative multimodal inference, and dynamic test-time self-improvement—all supported by efficient and trustworthy tool ecosystems. These directions collectively chart a concrete path from fragmented scientific skills toward genuine Scientific General Intelligence.

###### 6.4. Limitations

Despite providing a structured framework for evaluating scientific capabilities across four workflow stages, the current version of SGI-Bench has several limitations:

- (1) Partial coverage of real scientific workflows. The four stages in our benchmark function as probes for different components of scientific inquiry rather than a complete representation of real-world scientific practice. Many aspects of scientific work—such as integration across scientific disciplines and risk and safety assessment [79]—remain outside our current scope.
- (2) Scientific Deep Research currently emphasizes literature-inquiry–centric tasks. Deep Research spans activities such as literature inquiry [32], report-style reasoning [33], and related scientific analyses. In this benchmark, we focus on the literature-inquiry–centric subset, as identifying, interpreting, and integrating existing scientific knowledge is a foundational prerequisite for methodological design and experimental planning. This focus enables standardized, reproducible, and scalable evaluation while still probing a core component of real scientific workflows. More open-form variants—such as argumentative evidence synthesis or report generation—are also important but require substantial expert-based scoring, and are therefore reserved for future versions.
- (3) Idea Generation evaluation focuses on methodology design. Fully open-ended hypothesis generation involves substantial conceptual freedom and requires extensive expert adjudication to achieve reliable judgments. Due to practical constraints, our current evaluation focuses on the methoddesign component of scientific ideas [34, 35, 36]. Future extensions may incorporate hypothesis-level evaluation through a combination of arena-style model comparisons and expert review.
- (4) Limited code and action space coverage. Dry Experiment tasks currently support only Python [42], lacking adaptation to other programming languages and computational paradigms. The action space for Wet Experiments is an early-stage abstraction; scaling it requires constructing a large, standardized library of atomic actions grounded in real laboratory protocols [43].
- (5) Experimental reasoning in enclosed spaces. We employ a multiple-choice design to ensure objective, automatable evaluation [45]. While practical, this structure constrains the model’s ability to express diverse reasoning paths and limits assessment of open-form scientific explanations.
- (6) Partial coverage of deductive and inductive paradigms of scientific discovery. Scientific discovery is commonly understood to follow two broad paradigms: deduction and induction [80, 81]. Deductive processes begin from prior knowledge or theoretical propositions and proceed through reasoning to experimental verification. Inductive processes, in contrast, originate from new observational data or unexpected empirical phenomena and generalize toward broader patterns or hypotheses.

The PIM-grounded [10, 11] workflow in this version of SGI-Bench primarily reflects the deductive paradigm, as tasks begin with literature-based information and guide models toward reasoning and experiment planning. Inductive scientific discovery—which relies on data-driven pattern formation and hypothesis emergence—remains outside the scope of the current benchmark and represents an important direction for future expansion.

##### 7. Related Work

With the rapid advancement of Large Language Models (LLMs) and multi-agent systems in scientific reasoning, numerous datasets have emerged to evaluate their capabilities across various scientific domains.

###### 7.1. Benchmarks in Different Disciplines

A significant portion of existing benchmarks focuses on specific disciplines. In the physical sciences, PhyBench [82] examines multi-step reasoning and expression capabilities through original physics problems, while PHYX [83] focuses on real-world scenarios to assess physical reasoning and visual understanding. Additionally, PHYSICS [84] tests models using open-ended, university-level problems.To further address multimodal challenges, PhysUniBench [85] introduces a large-scale benchmark for undergraduate-level physics, specifically targeting the interpretation of physical diagrams and multi-step reasoning. In chemistry, ChemBench [86] provides domain-specific data for systematic evaluation, whereas ChemMLLM [87] extends this to multimodal assessment. More granular tasks are covered by benchmarks like ChemSafetyBench [88] and SpectrumWorld [89]. In life sciences, benchmarks range from the molecular level, such as DeepSEA [90] and GenomicsLong-Range [91], to healthcare applications like BioASQ [92] and VQA-RAD [93], as well as agricultural applications like SeedBench [94] and neuroscience with BrainBench [95]. For earth sciences, OmniEarth-Bench [96] covers a comprehensive range of fields with cross-domain tasks, EarthSE [97] builds a multi-level evaluation system from foundational to open-ended exploration, and MSEarth [98] utilizes high-quality scientific publications for graduate-level assessment. In remote sensing, GeoBench [99] and XLRSBench [100] evaluate perception and reasoning on high-resolution imagery. Furthermore, specialized benchmarks exist for other fields, including material science (MoleculeNet [101]), astronomy (AstroLLaMA and AstroMLab [102]), ocean science (OceanBench [103]), and climate science (ClimaQA [104]). These works primarily target deep evaluation within isolated disciplines. While benchmarks like ATLAS [105] have expanded to cover cross-disciplinary fields with high-difficulty standards, its evaluation specifically focuses on distinguishing frontier models through complex scientific reasoning and logical application tasks rather than the entire process of scientific discovery.

###### 7.2. Benchmarks for Different Scientific Tasks

Concurrently, other benchmarks focus on cross-disciplinary comprehensive capabilities, though their evaluation focus is often distributed across specific stages of the scientific discovery pipeline. Regarding idea generation at the research inception stage, MOOSE-Chem2 [37] evaluates models through a win/tie/lose comparison framework that scores generated hypotheses against reference answers using multiple independent judges. AI Idea Bench 2025 [106] evaluates the novelty of agent-generated ideas using a dataset derived from top-tier conference papers. In the core layer of knowledge processing and analysis, some benchmarks focus on literature comprehension. For instance, SciAssess [107] decomposes analysis into memory, understanding, and reasoning layers. Others, like SFE [45], introduce a cognitive framework to dissect multimodal performance on raw scientific data. Complementing these, SciReasoner [108] targets the alignment of natural language with heterogeneous scientific representations. Recent works also evaluate comprehensive academic survey capabilities:

DeepResearch Bench [33] measures report quality and citation grounding, Manalyzer [109] focuses on mitigating hallucinations in automated meta-analysis, and Scientist-Bench [110] highlights the full workflow from review to paper generation. Additionally, SciArena [111] proposed an open platform that dynamically evaluates and ranks the performance of base models on scientific literature tasks by collecting pairwise comparison preferences from domain researchers, and DeepResearch Arena [112] utilizes seminar-grounded tasks to evaluate the orchestration of multi-stage research workflows, while AAAR-1.0 [113] focuses on evaluating the model’s ability as an AI-assisted research tool. In terms of planning and execution, evaluations often center on tool usage and coding. ToolBench [114] and ToolUniverse [115] explore API usage and standardization. In scientific coding, SciCode [42] and ScienceAgentBench [116] assess code generation within realistic workflows. At a macro level, MLE-bench [117] and TaskBench [118] evaluate general planning and project management via Kaggle competitions and task decomposition graphs. In addition, DISCOVERYWORLD [119] launched the first virtual environment for evaluating the ability of intelligent agents to perform a complete cycle of novel scientific discovery. However, it focuses on a gamified simulation environment, and its task scenarios and evaluation dimensions cannot fully reflect the complexity and high-level cognitive needs of real scientific research workflows. LLM-SRBench [120] , on the other hand, focuses only on the model’s ability to discover scientific equations, with a relatively simple task and process. Despite these explorations, existing process-oriented benchmarks typically address only partial dimensions—such as knowledge understanding, data perception, or code generation—lacking a fine-grained, systematic evaluation of the entire scientific discovery lifecycle.

Summary In summary, existing works are either confined to deep exploration of single disciplines, scattered across isolated stages of the research process, or fail to capture the complexity of actual scientific discovery scenarios. Therefore, there is an urgent need to construct a comprehensive benchmark that covers multiple disciplines and connects the long-chain workflow of scientific research.

##### 8. Conclusion

This work advances the study of Scientific General Intelligence (SGI) from both theory and practice. Grounded in the Practical Inquiry Model, we formalize SGI as the capacity to navigate the iterative cycle of Deliberation, Conception, Action, and Perception with the versatility of a human scientist. Building on this principle-grounded definition, we operationalize SGI through SGI-Bench, a comprehensive, scientist-aligned benchmark that instantiates four core task families: Scientific Deep Research, Idea Generation, Dry/Wet Experiment, and Experimental Reasoning. Complemented by our agentic evaluation framework and multi-metric protocol, SGI-Bench enables scalable, transparent, and domainfaithful assessment.

Experiments reveal a consistent pattern: in Deep Research, models show step-level alignment but low exact-match accuracy (10–20%), with brittleness in quantitative reasoning; in Idea Generation, hypotheses are fluent but underspecified and infeasible; in Dry Experiment, code is executable but PassAll@k remains low; in Wet Experiment, sequences show omissions and misordering; and in Experimental Reasoning, causal reasoning outperforms comparative, with persistent multimodal challenges. These highlight gaps between linguistic fluency and integrated scientific cognition. Moreover, SGI exhibits dynamic capacity: Test-Time Reinforcement Learning with novelty rewards improves idea generation without reference answers.

Taken together, SGI-Bench clarifies both what SGI is and where current systems fail. By integrating principled task design, multi-metric evaluation, and agentic tool use, our framework provides a concrete foundation for systematically advancing SGI. Looking forward, the combination of numerically robust reasoning, planning-aware conception, executable experimentation, comparative multimodal

###### inference, dynamic test-time learning, and efficient tool ecosystems charts a clear path toward general intelligence systems capable of genuine scientific discovery.

##### References

- [1] Daya Guo et al. “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning”. In: Nature 645.8081 (2025), pp. 633–638.
- [2] Wayne Xin Zhao et al. “A survey of large language models”. In: arXiv preprint arXiv:2303.18223 1.2 (2023).
- [3] Humza Naveed et al. “A comprehensive overview of large language models”. In: ACM Transactions on Intelligent Systems and Technology 16.5 (2025), pp. 1–72.
- [4] Ming Hu et al. “A survey of scientific large language models: From data foundations to agent frontiers”. In: arXiv preprint arXiv:2508.21148 (2025).
- [5] Gheorghe Comanici et al. “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities”. In: arXiv preprint arXiv:2507.06261 (2025).
- [6] Nanyi Fei et al. “Towards artificial general intelligence via a multimodal foundation model”. In: Nature Communications 13.1 (2022), p. 3094.
- [7] Sébastien Bubeck et al. “Sparks of artificial general intelligence: Early experiments with gpt-4”. In: arXiv preprint arXiv:2303.12712 (2023).
- [8] Emily M Bender et al. “On the dangers of stochastic parrots: Can language models be too big?” In: Proceedings of the 2021 ACM conference on fairness, accountability, and transparency. 2021, pp. 610–623.
- [9] Ming Hu et al. A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers. 2025. arXiv: 2508.21148 [cs.CL]. url: https://arxiv.org/abs/2508. 21148.
- [10] D Randy Garrison, Terry Anderson, and Walter Archer. “Critical inquiry in a text-based environment: Computer conferencing in higher education”. In: The internet and higher education 2.2-3 (1999), pp. 87–105.
- [11] D Randy Garrison, Terry Anderson, and Walter Archer. “Critical thinking, cognitive presence, and computer conferencing in distance education”. In: American Journal of distance education 15.1 (2001), pp. 7–23.
- [12] Dan Hendrycks et al. “Measuring massive multitask language understanding”. In: arXiv preprint arXiv:2009.03300 (2020).
- [13] Xinrun Du et al. “Supergpqa: Scaling llm evaluation across 285 graduate disciplines”. In: arXiv preprint arXiv:2502.14739 (2025).
- [14] Grégoire Mialon et al. “Gaia: a benchmark for general ai assistants”. In: The Twelfth International Conference on Learning Representations. 2023.
- [15] Long Phan et al. “Humanity’s last exam”. In: arXiv preprint arXiv:2501.14249 (2025).
- [16] S Sanders. “125 questions: Exploration and Discovery”. In: Science/AAAS Custom Publishing Office: Washington, DC, USA 23 (2021).
- [17] Dawei Li et al. “From generation to judgment: Opportunities and challenges of llm-as-a-judge”. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025, pp. 2757–2791.
- [18] Mingchen Zhuge et al. “Agent-as-a-judge: Evaluate agents with agents”. In: arXiv preprint arXiv:2410.10934 (2024).

- [19] Qiyuan Zhang et al. “A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?” In: arXiv preprint arXiv:2503.24235 (2025).
- [20] Yuxin Zuo et al. “Ttrl: Test-time reinforcement learning”. In: arXiv preprint arXiv:2504.16084

(2025).

- [21] Renjun Xu and Jingwen Peng. “A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications”. In: arXiv preprint arXiv:2506.12594 (2025).
- [22] Yusong Hu et al. “FlowSearch: Advancing deep research with dynamic structured knowledge flow”. In: arXiv preprint arXiv:2510.08521 (2025).
- [23] Jinxin Shi et al. “DualResearch: Entropy-Gated Dual-Graph Retrieval for Answer Reconstruction”. In: arXiv preprint arXiv:2510.08959 (2025).
- [24] Andy P Field and Raphael Gillett. “How to do a meta-analysis”. In: British Journal of Mathematical and Statistical Psychology 63.3 (2010), pp. 665–694.
- [25] Wanghan Xu et al. “Manalyzer: End-to-end Automated Meta-analysis with Multi-agent System”. In: arXiv preprint arXiv:2505.20310 (2025).
- [26] “REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS”. English (US). In: 2023.
- [27] Harsh Trivedi et al. “Interleaving Retrieval with Chain-of-Thought Reasoning for KnowledgeIntensive Multi-Step Questions”. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Ed. by Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 10014–10037. url: https://aclanthology.org/2023.acl-long.557/.
- [28] Akari Asai et al. “Self-RAG: Learning to Retrieve, Generate, and Critique through SelfReflection”. In: The Twelfth International Conference on Learning Representations. 2024. url: https://openreview.net/forum?id=hSyW5go0v8.
- [29] Yijia Shao et al. “Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models”. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Ed. by Kevin Duh, Helena Gomez, and Steven Bethard. Mexico City, Mexico: Association for Computational Linguistics, 2024, pp. 6252–6278. url: https://aclanthology.org/ 2024.naacl-long.347/.
- [30] Hanchen Wang et al. “Scientific discovery in the age of artificial intelligence”. In: Nature 620.7972 (2023), pp. 47–60. issn: 1476-4687. doi: 10.1038/s41586-023-06221-2. url: http://dx.doi.org/10.1038/s41586-023-06221-2.
- [31] Chris Lu et al. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. 2024. arXiv: 2408.06292 [cs.AI]. url: https://arxiv.org/abs/2408.06292.
- [32] Nikos I. Bosse et al. “Deep Research Bench: Evaluating AI Web Research Agents”. In: ArXiv abs/2506.06287 (2025). url: https://api.semanticscholar.org/CorpusID: 279251730.
- [33] Mingxuan Du et al. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents.

2025. arXiv: 2506.11763 [cs.CL]. url: https://arxiv.org/abs/2506.11763.

- [34] Haiyuan Wan et al. “DeepResearch Arena: The First Exam of LLMs’ Research Abilities via Seminar-Grounded Tasks”. In: ArXiv abs/2509.01396 (2025). url: https://api. semanticscholar.org/CorpusID:281080495.
- [35] Karl Popper. The logic of scientific discovery. Routledge, 2005.

- [36] Zonglin Yang et al. “MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses”. In: Proceedings of the International Conference on Learning Representations (ICLR). 2025.
- [37] Zonglin Yang et al. “MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search”. In: arXiv preprint arXiv:2505.19209 (2025).
- [38] Bernardino Romera-Paredes et al. “Mathematical discoveries from program search with large language models”. In: Nature 625.7995 (2023), pp. 468–475. issn: 1476-4687. doi: 10.1038/s41586-023-06924-6. url: http://dx.doi.org/10.1038/s41586023-06924-6.
- [39] Yecheng Jason Ma et al. “Eureka: Human-Level Reward Design via Coding Large Language Models”. In: The Twelfth International Conference on Learning Representations. 2024. url: https://openreview.net/forum?id=IEduRUO55F.
- [40] Daniil A. Boiko et al. “Autonomous chemical research with large language models”. In: Nature 624.7992 (2023), pp. 570–578. issn: 1476-4687. doi: 10.1038/s41586-023-06792-0. url: http://dx.doi.org/10.1038/s41586-023-06792-0.
- [41] Andres M Bran et al. “Augmenting large language models with chemistry tools”. In: NeurIPS 2023 AI for Science Workshop. 2023. url: https://openreview.net/forum?id= wdGIL6lx3l.
- [42] Minyang Tian et al. SciCode: A Research Coding Benchmark Curated by Scientists. 2024. arXiv: 2407.13168 [cs.AI]. url: https://arxiv.org/abs/2407.13168.
- [43] Yuyang Liu et al. “BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning”. In: ArXiv abs/2505.07889 (2025). url: https://api. semanticscholar.org/CorpusID:278534452.
- [44] Sam Cox et al. “MDCROW: AUTOMATING MOLECULAR DYNAMICS WORKFLOWS WITH LARGE LANGUAGE MODELS”. In: Towards Agentic AI for Science: Hypothesis Generation, Comprehension, Quantification, and Validation. 2025. url: https://openreview.net/ forum?id=KNQe3Cmupn.
- [45] Yuhao Zhou et al. Scientists’ First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning. 2025. arXiv: 2506.10521 [cs.AI]. url: https://arxiv. org/abs/2506.10521.
- [46] Ge Zhang et al. CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark. 2024. arXiv: 2401.11944 [cs.CL]. url: https://arxiv.org/abs/2401. 11944.
- [47] BeiJing YuanShi Technology Co., Ltd. WenXiaobai (DeepSeek Search). Online; accessed 2025. url: https://www.wenxiaobai.com/.
- [48] OpenAI. Introducing Deep Research. https://openai.com/zh-Hans-CN/index/ introducing-deep-research/. 2025.
- [49] Perplexity AI. Introducing Perplexity Deep Research. https://www.perplexity.ai/hub/ blog/introducing-perplexity-deep-research. 2025.
- [50] Moonshot AI. “Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities”. In: (2025). Official Project Page. url: https://moonshotai.github.io/KimiResearcher/.
- [51] xAI. Grok 4: Native Tool Use and DeepSearch Capabilities. https://x.ai/news/grok-4. Official announcement of Grok’s agentic search features. 2025.

- [52] Kai Ruan et al. LiveIdeaBench: Evaluating LLMs’ Divergent Thinking for Scientific Idea Generation with Minimal Context. 2025. arXiv: 2412.17596 [cs.CL]. url: https://arxiv.org/ abs/2412.17596.
- [53] Naman Jain et al. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. 2024. arXiv: 2403.07974 [cs.SE]. url: https://arxiv.org/abs/ 2403.07974.
- [54] OpenAI. GPT-5 System Card. Tech. rep. OpenAI, 2025. url: https://cdn.openai.com/ gpt-5-system-card.pdf.
- [55] Daya Guo et al. “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning”. In: Nature 645.8081 (2025), pp. 633–638. issn: 1476-4687. doi: 10.1038/s41586-02509422-z. url: http://dx.doi.org/10.1038/s41586-025-09422-z.
- [56] Kimi Team et al. “Kimi k2: Open agentic intelligence”. In: arXiv preprint arXiv:2507.20534

(2025).

- [57] Aymeric Roucher, Albert Villanova del Moral, et al. smolagents: a smol library to build great agentic systems. https://github.com/huggingface/smolagents. 2025.
- [58] DeepSeek-AI. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. Tech. rep. [5, 49]. DeepSeek, 2025. url: https://huggingface.co/deepseek-ai/DeepSeekV3.2/resolve/main/assets/paper.pdf.
- [59] Lei Bai et al. “Intern-s1: A scientific multimodal foundation model”. In: arXiv preprint arXiv:2508.15763 (2025).
- [60] Shuai Bai et al. “Qwen3-VL Technical Report”. In: arXiv preprint arXiv:2511.21631 (2025).
- [61] An Yang et al. “Qwen3 technical report”. In: arXiv preprint arXiv:2505.09388 (2025).
- [62] Meta AI. Llama 4: Multimodal Intelligence. https://ai.meta.com/blog/llama-4multimodal-intelligence/. 2025.
- [63] OpenAI. GPT-4o System Card. Tech. rep. OpenAI, 2024. url: https://cdn.openai.com/ gpt-4o-system-card.pdf.
- [64] OpenAI. Models: GPT-4.1 Documentation. https://platform.openai.com/docs/ models/gpt-4.1. OpenAI Platform Docs. 2025.
- [65] OpenAI. GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum. Tech. rep. OpenAI, 2025. url: https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de06d87147e07e/5_1_system_card.pdf.
- [66] OpenAI. Update to GPT-5 System Card: GPT-5.2. Tech. rep. OpenAI, 2025. url: https: //cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_ 2_system-card.pdf.
- [67] OpenAI. OpenAI o3 and o4-mini System Card. Tech. rep. OpenAI, 2025. url: https:// cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4mini-system-card.pdf.
- [68] Google DeepMind. Gemini 3 Model Card. https : / / storage . googleapis . com / deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf. 2025.
- [69] Anthropic. Claude Opus 4.1 System Card. https : / / assets . anthropic . com / m / 4c024b86c698d3d4/original/Claude-4-1-System-Card.pdf. 2025.
- [70] Anthropic. Claude Sonnet 4.5 System Card. Tech. rep. 2025. url: https://assets. anthropic . com / m / 12f214efcc2f457a / original / Claude - Sonnet - 4 - 5 System-Card.pdf.

- [71] xAI. Grok 3 Beta — The Age of Reasoning Agents. https://x.ai/news/grok-3. 2025.
- [72] xAI. Grok 4 Model Card. 2025. url: https://data.x.ai/2025-08-20-grok-4model-card.pdf.
- [73] Mengkang Hu et al. “Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation”. In: arXiv preprint arXiv:2505.23885 (2025).
- [74] Xiaoxi Li, Jiajie Jin, Guanting Dong, et al. “WebThinker: Empowering Large Reasoning Models with Deep Research Capability”. In: Proceedings of NeurIPS 2025 (2025). Accepted by NeurIPS

2025. arXiv:2504.21776 [54].

- [75] Chai Jingyi, Shuo Tang, et al. “SciMaster: Towards General-Purpose Scientific AI Agents, Part

I. X-Master as Foundation”. In: arXiv preprint arXiv:2507.05241 (2025).

- [76] NovelSeek Team et al. “NovelSeek: When Agent Becomes the Scientist–Building Closed-Loop System from Hypothesis to Verification”. In: arXiv preprint arXiv:2505.16938 (2025).
- [77] Yuze Zhao et al. SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning. 2025. arXiv: 2408.05517 [cs.CL]. url: https://arxiv.org/abs/2408.05517.
- [78] Bhargavi Paranjape et al. ART: Automatic multi-step reasoning and tool-use for large language models. 2023. arXiv: 2303.09014 [cs.CL]. url: https://arxiv.org/abs/2303. 09014.
- [79] Yujun Zhou et al. “LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs”. In: ArXiv abs/2410.14182 (2024). url: https://api.semanticscholar.org/ CorpusID:273482719.
- [80] Francis Bacon. Novum organum. Clarendon press, 1878.
- [81] Karl Popper. Conjectures and refutations: The growth of scientific knowledge. routledge, 2014.
- [82] Shi Qiu et al. PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models. 2025. arXiv: 2504.16074 [cs.CL]. url: https://arxiv.org/abs/

- 2504.16074.

[83] Hui Shen et al. PhyX: Does Your Model Have the "Wits" for Physical Reasoning? 2025. arXiv:

- 2505.15929 [cs.AI]. url: https://arxiv.org/abs/2505.15929.

- [84] Kaiyue Feng et al. PHYSICS: Benchmarking Foundation Models on University-Level Physics Problem Solving. 2025. arXiv: 2503.21821 [cs.AI]. url: https://arxiv.org/abs/ 2503.21821.
- [85] Lintao Wang et al. PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models. 2025. arXiv: 2506.17667 [cs.AI]. url: https://arxiv.org/ abs/2506.17667.
- [86] Adrian Mirza et al. Are large language models superhuman chemists? 2024. arXiv: 2404.01475 [cs.LG]. url: https://arxiv.org/abs/2404.01475.
- [87] Qian Tan et al. ChemMLLM: Chemical Multimodal Large Language Model. 2025. arXiv: 2505. 16326 [cs.LG]. url: https://arxiv.org/abs/2505.16326.
- [88] Haochen Zhao et al. ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain. 2024. arXiv: 2411.16736 [cs.CL]. url: https://arxiv.org/abs/2411.16736.
- [89] Zhuo Yang et al. SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy. 2025. arXiv: 2508.01188 [cs.LG]. url: https://arxiv.org/abs/2508.01188.
- [90] Pooja Kathail, Ayesha Bajwa, and Nilah M. Ioannidis. Leveraging genomic deep learning models for the prediction of non-coding variant effects. 2025. arXiv: 2411.11158 [q-bio.GN]. url: https://arxiv.org/abs/2411.11158.

- [91] Anonymous. The Genomics Long-Range Benchmark: Advancing DNA Language Models. 2024. url: https://openreview.net/forum?id=Cdc90HKs1I.
- [92] Anastasia Krithara et al. “BioASQ-QA: A manually curated corpus for Biomedical Question Answering”. In: Scientific Data 10.1 (2023). issn: 2052-4463. doi: 10.1038/s41597023-02068-4. url: http://dx.doi.org/10.1038/s41597-023-02068-4.
- [93] Jason J. Lau et al. “A dataset of clinically generated visual questions and answers about radiology images”. In: Scientific Data 5.1 (2018). issn: 2052-4463. doi: 10.1038/sdata. 2018.251. url: http://dx.doi.org/10.1038/sdata.2018.251.
- [94] Jie Ying et al. SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science. 2025. arXiv: 2505.13220 [cs.CL]. url: https://arxiv.org/abs/ 2505.13220.
- [95] Xiaoliang Luo et al. “Large language models surpass human experts in predicting neuroscience results”. In: Nature Human Behaviour 9.2 (2024), pp. 305–315. issn: 2397-3374. doi: 10.1038/s41562-024-02046-9. url: http://dx.doi.org/10.1038/s41562024-02046-9.
- [96] Fengxiang Wang et al. OmniEarth-Bench: Towards Holistic Evaluation of Earth’s Six Spheres and Cross-Spheres Interactions with Multimodal Observational Earth Data. 2025. arXiv: 2505. 23522 [cs.CV]. url: https://arxiv.org/abs/2505.23522.
- [97] Wanghan Xu et al. EarthSE: A Benchmark for Evaluating Earth Scientific Exploration Capability of LLMs. 2025. arXiv: 2505.17139 [cs.CL]. url: https://arxiv.org/abs/2505. 17139.
- [98] Xiangyu Zhao et al. MSEarth: A Multimodal Scientific Dataset and Benchmark for Phenomena Uncovering in Earth Science. 2025. arXiv: 2505.20740 [cs.AI]. url: https://arxiv. org/abs/2505.20740.
- [99] Muhammad Sohail Danish et al. GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks. 2025. arXiv: 2411.19325 [cs.CV]. url: https://arxiv.org/abs/ 2411.19325.
- [100] Fengxiang Wang et al. XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery? 2025. arXiv: 2503.23771 [cs.CV]. url: https://arxiv.org/abs/2503.23771.
- [101] Zhenqin Wu et al. MoleculeNet: A Benchmark for Molecular Machine Learning. 2018. arXiv: 1703.00564 [cs.LG]. url: https://arxiv.org/abs/1703.00564.
- [102] Rui Pan et al. AstroMLab 2: AstroLLaMA-2-70B Model and Benchmarking Specialised LLMs for Astronomy. 2024. arXiv: 2409.19750 [astro-ph.IM]. url: https://arxiv.org/ abs/2409.19750.
- [103] Anass El Aouni et al. “OceanBench: A Benchmark for Data-Driven Global Ocean Forecasting systems”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2025. url: https://openreview.net/forum?id= wZGe1Kqs8G.
- [104] Veeramakali Vignesh Manivannan et al. ClimaQA: An Automated Evaluation Framework for Climate Question Answering Models. 2025. arXiv: 2410.16701 [cs.LG]. url: https: //arxiv.org/abs/2410.16701.
- [105] Hongwei Liu et al. ATLAS: A High-Difficulty, Multidisciplinary Benchmark for Frontier Scientific Reasoning. 2025. arXiv: 2511.14366 [cs.CL]. url: https://arxiv.org/abs/2511. 14366.

- [106] Yansheng Qiu et al. AI Idea Bench 2025: AI Research Idea Generation Benchmark. 2025. arXiv: 2504.14191 [cs.AI]. url: https://arxiv.org/abs/2504.14191.
- [107] Hengxing Cai et al. SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis.

2024. arXiv: 2403.01976 [cs.CL]. url: https://arxiv.org/abs/2403.01976.

- [108] Yizhou Wang et al. SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines. 2025. arXiv: 2509.21320 [cs.CL]. url: https://arxiv.org/abs/2509.21320.
- [109] Wanghan Xu et al. Manalyzer: End-to-end Automated Meta-analysis with Multi-agent System.

2025. arXiv: 2505.20310 [cs.AI]. url: https://arxiv.org/abs/2505.20310.

- [110] Jiabin Tang et al. AI-Researcher: Autonomous Scientific Innovation. 2025. arXiv: 2505.18705 [cs.AI]. url: https://arxiv.org/abs/2505.18705.
- [111] Yilun Zhao et al. SciArena: An Open Evaluation Platform for Foundation Models in Scientific Literature Tasks. 2025. arXiv: 2507.01001 [cs.CL]. url: https://arxiv.org/abs/ 2507.01001.
- [112] Haiyuan Wan et al. DeepResearch Arena: The First Exam of LLMs’ Research Abilities via SeminarGrounded Tasks. 2025. arXiv: 2509.01396 [cs.AI]. url: https://arxiv.org/abs/ 2509.01396.
- [113] Renze Lou et al. AAAR-1.0: Assessing AI’s Potential to Assist Research. 2025. arXiv: 2410.22394 [cs.CL]. url: https://arxiv.org/abs/2410.22394.
- [114] Yujia Qin et al. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.

2023. arXiv: 2307.16789 [cs.AI]. url: https://arxiv.org/abs/2307.16789.

- [115] Shanghua Gao et al. Democratizing AI scientists using ToolUniverse. 2025. arXiv: 2509.23426 [cs.AI]. url: https://arxiv.org/abs/2509.23426.
- [116] Ziru Chen et al. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for DataDriven Scientific Discovery. 2025. arXiv: 2410.05080 [cs.CL]. url: https://arxiv. org/abs/2410.05080.
- [117] Jun Shern Chan et al. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. 2025. arXiv: 2410.07095 [cs.CL]. url: https://arxiv.org/abs/ 2410.07095.
- [118] Yongliang Shen et al. TaskBench: Benchmarking Large Language Models for Task Automation.

2024. arXiv: 2311.18760 [cs.CL]. url: https://arxiv.org/abs/2311.18760.

- [119] Peter Jansen et al. DISCOVERYWORLD: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents. 2024. arXiv: 2406.06769 [cs.AI]. url: https: //arxiv.org/abs/2406.06769.
- [120] Parshin Shojaee et al. LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models. 2025. arXiv: 2504.10415 [cs.CL]. url: https://arxiv.org/ abs/2504.10415.

##### A. Appendix

- A.1. Authors Lead Authors Wanghan Xu1,2, Yuhao Zhou1,3, Yifan Zhou1,2, Qinglong Cao2, Shuo Li1,4, Jia Bu1,5 Core Authors

Bo Liu6, Yixin Chen1,7, Xuming He1,8, Xiangyu Zhao1,6, Xiang Zhuang1,8, Fengxiang Wang1,9, Zhiwang Zhou1,10

###### Contributors

Qiantai Feng, Wenxuan Huang, Jiaqi Wei, Hao Wu, Yuejin Yang, Guangshuai Wang, Sheng Xu, Ziyan Huang, Xinyao Liu, Jiyao Liu, Cheng Tang, Wei Li, Ying Chen, Junzhi Ning, Pengfei Jiang, Chenglong Ma, Ye Du, Changkai Ji, Huihui Xu, Ming Hu, Jiangbin Zheng, Xin Chen, Yucheng Wu, Feifei Jiang, Xi Chen, Xiangru Tang, Yuchen Fu, Yingzhou Lu, Yuanyuan Zhang, Lihao Sun, Chengbo Li, Jinzhe Ma, Wanhao Liu, Yating Liu, Kuo-Cheng Wu, Shengdu Chai, Yizhou Wang, Ouwen Zhangjin, Chen Tang, Shufei Zhang, Wenbo Cao, Junjie Ren, Taoyong Cui, Zhouheng Yao, Juntao Deng, Yijie Sun, Feng Liu, Wangxu Wei, Jingyi Xu, Zhangrui Li, Junchao Gong, Zijie Guo, Zhiyu Yao, Zaoyu Chen, Tianhao Peng, Fangchen Yu

###### Scientific Directors

Bo Zhang1, Dongzhan Zhou1, Shixiang Tang1,11, Jiaheng Liu1,12, Fenghua Ling1, Yan Lu1, Yuchen Ren1, Ben Fei1,11, Zhen Zhao1, Xinyu Gu1, Rui Su1, Xiao-Ming Wu6, Weikang Si13, Yang Liu14, Hao Chen1, Xiangchao Yan1, Xue Yang2, Junchi Yan2, Jiamin Wu1, Qihao Zheng1, Chenhui Li5, Zhiqiang Gao1, Hao Kong16, Junjun He1, Mao Su1, Tianfan Fu1,12, Peng Ye1,11, Chunfeng Song1, Nanqing Dong1, Yuqiang Li1, Huazhu Fu16, Siqi Sun1,17, Lijing Cheng18, Jintai Lin15, Wanli Ouyang1,11, Bowen Zhou1,19

Corresponding Authors Wenlong Zhang1, Lei Bai1 Main Affiliations

- 1 Shanghai Artificial Intelligence Laboratory
- 2 Shanghai Jiao Tong University
- 3 Sichuan University
- 4 Central South University
- 5 East China Normal University
- 6 The Hong Kong Polytechnic University
- 7 University of California, Los Angeles
- 8 Zhejiang University
- 9 National University of Defense Technology
- 10 Tongji University
- 11 The Chinese University of Hong Kong
- 12 Nanjing University
- 13 National Institute of Metrology

- 14 Aerospace Information Research Institute,Chinese Academy of Sciences
- 15 Peking University
- 16 The Agency for Science, Technology and Research (A*STAR)
- 17 Fudan University
- 18 Chinese Academy of Sciences
- 19 Tsinghua University

###### A.2. Disciplines and Research Directions Overview

Table 9 | Disciplines And Research Directions: Overview of 10 scientific domains and representative research topics curated for scientist-aligned SGI-Bench workflows.

Disciplines Research Directions Description Astronomy Gravitational Wave Detection and Pa-

Analyzing data from interferometers like LIGO and Virgo to detect gravitational waves from compact binary coalescences (black holes, neutron stars) and precisely estimate their physical properties like mass, spin, and location to test general relativity.

rameter Estimation

Astronomy Fast Radio Burst Detection and Localization

Searching radio telescope data for millisecond-duration, extragalactic radio flashes (FRBs) and using interferometry to pinpoint their host galaxies, aiming to uncover the mysterious physical mechanisms that produce them.

Astronomy Real Time Optical Transient Survey Based on ZTF

Utilizing the Zwicky Transient Facility (ZTF) to scan the night sky, identifying new or changing celestial objects like supernovae and kilonovae, and issuing rapid alerts to the global astronomical community for multi-wavelength follow-up observations.

Astronomy Formula Regression Applying symbolic regression and other machine learning techniques to large astronomical datasets to automatically discover novel mathematical formulas or physical laws that describe the behavior of celestial objects and phenomena.

Chemistry Molecular Interaction Computationally simulating and quantifying the non-covalent forces between molecules, such as hydrogen bonds and van der Waals forces, to understand molecular recognition, protein-ligand binding, and self-assembly.

Chemistry Target Based Drug Design Employing computational methods to design drug candidates that specifically bind to a known biological target, such as a protein’s active site, thereby modulating its function to achieve a therapeutic effect.

Chemistry De Novo Drug Design Using generative AI models to computationally design entirely new molecules with desired pharmacological properties, without starting from an existing chemical scaffold, to explore novel regions of chemical space.

Chemistry Chemical Molecular Synthesis Pathway Planning

Developing algorithms, often based on retrosynthesis, to devise the most efficient and practical multi-step reaction routes for synthesizing a target molecule, optimizing for yield, cost, and sustainability.

Chemistry Molecular Property Prediction Building and applying machine learning models (e.g., QSAR) to predict key chemical and physical properties of molecules, such as toxicity, solubility, and bioactivity, to accelerate materials discovery and drug development.

Earth Seismic Wave Detection Using networks of seismometers to detect and analyze seismic waves from earthquakes and other sources, enabling the study of fault lines and the tomographic imaging of the Earth’s mantle and core.

Earth Ocean Heat Content Aggregating and analyzing temperature data from sources like Argo floats and satellites to calculate the total thermal energy stored within the ocean, a critical indicator for quantifying global warming and climate change.

Earth Atmospheric Differential Equation Numerically solving the complex systems of partial differential equations (e.g., NavierStokes equations) that govern atmospheric fluid dynamics and thermodynamics to produce accurate weather forecasts and climate projections.

Earth Typhoon Wind Pressure Relationship Developing and validating models that describe the physical relationship between a typhoon’s central pressure and its maximum sustained wind speeds, crucial for forecasting storm intensity and assessing potential damage.

Earth Vegetation Coverage Rate Processing satellite and aerial imagery using spectral indices like NDVI to quantify the fraction of land covered by vegetation, which is vital for monitoring ecosystem health, agriculture, and deforestation.

Earth Glacier Estimation Combining satellite altimetry, gravimetry (GRACE), and imagery to measure changes in glacier volume and mass balance over time, providing direct evidence of the impacts of climate change.

Earth Ozone Pollution and Its Causes Investigating the chemical reactions between precursor pollutants (like NOx and VOCs) under sunlight that form harmful ground-level ozone, and modeling its transport and concentration in urban and rural areas.

Earth Emission Inversion Based on Satellite Remote Sensing and 4D-Var

Using advanced data assimilation techniques (4D-Var) to combine satellite measurements of atmospheric composition with chemical transport models, thereby inferring the location and strength of pollutant emission sources on the ground.

Earth Emission Inversion Based on Local Mass Conservation

Applying mass balance principles to highresolution measurements (e.g., from aircraft) around a specific region to calculate the net flux and estimate emissions of greenhouse gases or pollutants from sources like cities or industrial facilities.

Earth Multiple Seismic Wave Attenuations Modeling the progressive energy loss of seismic waves as they propagate through different geological materials, which helps in characterizing subsurface structures and identifying resources like oil and gas.

Energy Optimal Power Flow Calculation Developing algorithms to solve complex optimization problems for electrical grids, determining the best generator outputs to meet demand reliably while minimizing generation costs and transmission losses.

Energy Fengguang New Energy Power Forecasting

Creating predictive models using meteorological data (wind speed, solar irradiance) and machine learning to accurately forecast the power output of wind and solar farms, which is essential for stable grid management.

Information Multimodal Understanding Building AI systems that can process, interpret, and reason about information from multiple sources simultaneously, such as text, images, audio, and video, to achieve a more holistic understanding.

Information Dialogue System Designing and training conversational AI agents (chatbots) that can engage in natural, coherent, and context-aware conversations with humans for tasks like customer service or information retrieval.

Information Code Generation Developing large language models and other AI techniques to automatically write, complete, and debug computer code based on natural language descriptions or functional specifications.

Information Sensor Spatial Characteristics Phase Free Reconstruction

Creating novel algorithms to reconstruct the spatial sensitivity pattern of a sensor (like a microphone or antenna) using only the magnitude of its measurements, without needing phase information, which is often difficult to obtain.

Life De Novo Protein Sequencing Developing computational methods to determine the amino acid sequence of a novel protein directly from its tandem mass spectrometry data, without relying on a reference genome.

Life Small Molecule Inference Using computational models to predict the biological effects of small molecules, such as their binding targets, mechanism of action, or potential toxicity, based on their chemical structure.

Life Disease Biomarker Discovery Analyzing high-throughput biological data (e.g., genomics, proteomics) with statistical and machine learning methods to identify molecules whose presence or level can indicate a specific disease state.

Life Tumor Neoantigen Discovery Identifying unique peptides that arise from mutations in cancer cells, which can be recognized by the immune system, for the development of personalized cancer vaccines and immunotherapies.

Life RNA Tertiary Structure Prediction Computationally predicting the complex three-dimensional folded structure of RNA molecules from their primary sequence to understand their function in cellular processes like gene regulation and catalysis.

Life Protein Structure Predicting the three-dimensional atomic coordinates of a protein from its amino acid sequence using methods like deep learning (e.g., AlphaFold) or homology modeling to understand its biological function.

Life Genome Function Prediction Annotating the functions of genes, regulatory elements, and non-coding regions across the genome by integrating diverse data types like DNA sequence, gene expression, and epigenetic modifications.

Life Automatic Development of Medical Imaging Algorithms

Creating AI-powered systems that can automatically generate and optimize image analysis pipelines for tasks like segmentation, registration, and classification in various medical imaging modalities (MRI, CT).

Life AI Drug Discovery Applying a range of AI and machine learning techniques across the entire drug discovery pipeline, from identifying novel drug targets and designing molecules to predicting clinical trial outcomes.

Life Tumor Immunotherapy Designing and developing therapeutic strategies, such as checkpoint inhibitors or CAR-T cells, that stimulate and enhance the patient’s own immune system to recognize and attack cancer cells.

Life Revealing the Mechanisms of the Tumor Microenvironment

Studying the complex interplay between cancer cells, immune cells, stromal cells, and the extracellular matrix to understand how this environment promotes tumor growth and metastasis.

Life AI Assisted Antibody Design Using machine learning models to design and optimize antibodies with high affinity and specificity for a given antigen, accelerating the development of new therapeutics and diagnostics.

Life Protein Structure Prediction Developing and applying computational algorithms, particularly deep learning models, to accurately predict the 3D structure of proteins from their amino acid sequence.

Life Early Screening and Risk Stratification of Pancreatic Cancer

Developing novel diagnostic tools, such as blood-based biomarkers or AI-driven imaging analysis, to detect pancreatic cancer at an early, more treatable stage and to classify patients by risk level.

Life Protein Protein Interaction Prediction

Developing computational methods to predict which proteins in a cell will physically bind to each other, in order to map out the cellular signaling pathways and protein complexes.

Life Discovery of Immunotherapy Targets Analyzing tumor and immune cell data to identify new molecular targets, such as surface proteins or mutated peptides, that can be exploited for cancer immunotherapy.

Life Biomarker Discovery Identifying molecular signatures (genes, proteins, metabolites) in patient samples that can be used for disease diagnosis, prognosis, or predicting response to therapy.

Life Strain Metabolic Reconstruction Creating comprehensive computational models of the metabolic networks of microbial strains to understand their physiology and guide metabolic engineering for producing valuable chemicals.

Life Regulatory Element Design Designing synthetic DNA or RNA sequences, such as promoters and enhancers, to precisely control the expression of specific genes for applications in biotechnology and synthetic biology.

Life Computational Drug Design Utilizing molecular modeling, simulation, and machine learning to design and optimize small molecules that can effectively bind to a biological target and modulate its activity.

Life Design of Regulatory Regions for mRNA Vaccine Drugs

Engineering the untranslated regions (UTRs) and other elements of mRNA sequences to optimize their stability, translation efficiency, and immune response for next-generation vaccine development.

Life Medical Image Understanding Developing deep learning models to analyze and interpret complex medical images (e.g., X-rays, MRIs, pathology slides) to assist clinicians in diagnosis, treatment planning, and disease monitoring.

Material Polymer Thermoelectric Designing and synthesizing polymer-based materials that can efficiently convert waste heat into useful electrical energy, focusing on enhancing their thermoelectric figure of merit (ZT).

Material Thermal Electrocatalysis Investigating how to use thermal energy to enhance the performance and efficiency of catalytic materials in electrochemical reactions, such as in fuel cells or water splitting.

Material Nano Adsorption Materials Developing porous nanomaterials like metal-organic frameworks (MOFs) or zeolites with high surface area and specific chemical properties for applications in gas separation, storage, and carbon capture.

Material Chloride Solid State Electrolyte Researching and developing novel solidstate materials that conduct chloride ions, aiming to create safer and more energydense all-solid-state batteries.

Material Oxygen Evolution Reaction Catalytic Materials

Designing efficient, stable, and low-cost catalysts to accelerate the oxygen evolution reaction (OER), a key bottleneck in processes like water splitting for hydrogen production.

Material KRF Resin Polymerization Reaction Investigating and optimizing the chemical reaction conditions and mechanisms for the polymerization of ketone-resolformaldehyde (KRF) resins to control their final properties for industrial applications.

Material Polymer Thermoelectric Researching and developing organic and composite polymer materials with high electrical conductivity and low thermal conductivity for flexible and lightweight thermoelectric devices.

Mathematics Differential Privacy Developing mathematical frameworks and algorithms that allow for the analysis of sensitive datasets while providing rigorous, provable guarantees about the privacy of individuals in the data.

Mathematics Coordinate Descent Optimization Algorithm

Designing and analyzing efficient optimization algorithms that solve complex problems by iteratively optimizing one variable or a small block of variables at a time, while keeping others fixed.

Mathematics Matrix Completion Developing algorithms to accurately recover a full data matrix from a small subset of its observed entries, with applications in recommender systems and image inpainting.

Mathematics Numerical Methods for Differential Equations

Devising and implementing stable and accurate computational algorithms (e.g., Runge-Kutta methods) for finding approximate solutions to differential equations that model real-world phenomena.

Mathematics Shortest Path Planning Developing and applying graph-based algorithms like Dijkstra’s or A* to find the most efficient route between two points in a network, with applications in logistics, robotics, and network routing.

Neuroscience Visual Decoding Using machine learning models to analyze brain activity patterns, typically from fMRI or electrophysiology, to reconstruct or identify the visual images a person is seeing.

Neuroscience Motion Decoding Developing brain-computer interfaces that can interpret neural signals from the motor cortex to predict intended movements, enabling control of prosthetic limbs or external devices.

Neuroscience Emotion Recognition Analyzing neurophysiological signals (like EEG) or behavioral cues (like facial expressions) with AI to identify and classify human emotional states.

Neuroscience Electron Microscopy Neuron Segmentation

Creating automated computational pipelines, often using deep learning, to trace and segment individual neurons and their connections in large-scale electron microscopy volumes of brain tissue.

Neuroscience Neural Activity and Behavior Prediction

Building statistical and dynamical models that link the activity of neural populations to specific behaviors, in order to understand the neural codes underlying perception, decision-making, and action.

Physics Computational Condensed Matter Physics

Using first-principles simulations (like Density Functional Theory) and many-body techniques to predict the electronic, magnetic, and structural properties of materials from fundamental quantum mechanics.

Physics Zeeman Effect Experiment Precisely measuring the splitting of atomic spectral lines in the presence of an external magnetic field to probe the quantum mechanical properties of atoms, such as electron spin and angular momentum.

Physics Research on Soft Condensed Matter Physics and Glass Transition Dynamics

Investigating the physical principles governing the behavior of soft materials (polymers, colloids) and studying the complex, slow dynamics associated with the transition from a liquid to a glassy state.

Physics Deep PDE Solving to Enhance Model Expressiveness

Developing novel deep learning architectures, such as physics-informed neural networks (PINNs), to solve complex partial differential equations and improve the predictive power of physics-based models.

Physics Chaotic Behavior in Circuit Systems Studying and modeling the emergence of chaos and other nonlinear dynamical behaviors in electronic circuits, such as the Chua’s circuit, to understand fundamental principles of complex systems.

Physics Research on General Machine Learning Potential Function Model Architecture

Developing universal machine learning frameworks to accurately model the potential energy surface of molecular systems, enabling large-scale molecular dynamics simulations with quantum accuracy.

Physics Nuclear Magnetic Resonance and Its Imaging Experiment

Utilizing the principles of nuclear magnetic resonance to probe the structure and dynamics of molecules in materials and to create non-invasive medical images (MRI) of biological tissues.

Physics Quadrupole Mass Spectrometer Studying the principles of using combined electric and magnetic fields in a quadrupole mass analyzer to separate ions based on their mass-to-charge ratio for chemical analysis.

Physics Research on Superconducting Mechanisms, Discovery of Superconducting Materials and Process Optimization

Investigating the fundamental quantum mechanisms of superconductivity, computationally searching for new materials with higher critical temperatures, and optimizing their synthesis for practical applications.

###### A.3. Cases

- A.3.1. Scientific Deep Research Example of Scientific Deep Research in Astronomy

###### Question

The Dispersion Measure (DM) of a Fast Radio Burst (FRB) is the integrated column density of free electrons along the line of sight. The observed value, 𝐷𝑀𝑜𝑏𝑠, is generally considered the sum of four primary components: 𝐷𝑀𝑜𝑏𝑠 = 𝐷𝑀𝑀𝑊 + 𝐷𝑀ℎ𝑎𝑙𝑜 + 𝐷𝑀𝐼𝐺𝑀 + 𝐷𝑀ℎ𝑜𝑠𝑡,𝑜𝑏𝑠 where 𝐷𝑀𝑀𝑊 is the contribution from the Milky Way’s interstellar medium, 𝐷𝑀ℎ𝑎𝑙𝑜 is from the Milky Way’s halo, 𝐷𝑀𝐼𝐺𝑀 is from the intergalactic medium, and 𝐷𝑀ℎ𝑜𝑠𝑡,𝑜𝑏𝑠 is the contribution from the host galaxy in the observer’s frame. The host contribution in its rest frame, 𝐷𝑀ℎ𝑜𝑠𝑡,𝑟𝑒𝑠𝑡, is related to the observed value by 𝐷𝑀ℎ𝑜𝑠𝑡,𝑟𝑒𝑠𝑡 = 𝐷𝑀ℎ𝑜𝑠𝑡,𝑜𝑏𝑠/(1 + 𝑧). The Rotation Measure (RM) describes the Faraday rotation of a linearly polarized signal passing through a magnetized plasma. For the host galaxy, its contribution to the RM as 𝑅𝑀ℎ𝑜𝑠𝑡, which is highly relevant with ⟨𝐵||⟩, the average line-of-sight magnetic field strength in the host galaxy’s environment, measured in microgauss (𝜇𝐺). Astronomers have precisely localized the repeating FRB 20180814A and identified its host galaxy. The total observed dispersion measure is 𝐷𝑀𝑜𝑏𝑠 = 189.4 pc · cm−3, and the spectroscopic redshift of the host is 𝑧 = 0.06835. After subtracting the Galactic contribution, the extragalactic rotation measure is found to be 𝑅𝑀𝑒𝑥𝑡𝑟𝑎𝑔𝑎𝑙𝑎𝑐𝑡𝑖𝑐 ≈ 655 rad · m−2, which is assumed

to originate primarily from the FRB’s host galaxy environment. Based on a detailed Bayesian model presented in the source paper, the total contribution from extragalactic sources (IGM +

host) is determined to be 𝐷𝑀𝑒𝑥𝑡𝑟𝑎𝑔𝑎𝑙𝑎𝑐𝑡𝑖𝑐,𝑜𝑏𝑠 = 64 pc · cm−3, within which the IGM contribution is estimated as 𝐷𝑀𝐼𝐺𝑀 = 45 pc · cm−3. Based on the information above, calculate the lower limit of the average line-of-sight magnetic field strength, ⟨𝐵||⟩, in the FRB’s host galaxy environment. Provide a numerical answer in units of microgauss (𝜇𝐺), rounded to the nearest integer.

###### Steps

- Step 1. Search for the relevant paper about Sub-arcminute localization of 13 repeating fast radio bursts detected by CHIME/FRB.
- Step 2. Based on Macquart, 𝐷𝑀ℎ𝑜𝑠𝑡,𝑜𝑏𝑠 = 61.515pc · cm−3.
- Step 3. Calculate the contribution of the host galaxy to the observer coordinate system (𝐷𝑀ℎ𝑜𝑠𝑡,𝑜𝑏𝑠 = 5.885pc · cm−3).
- Step 4. Calculate the contribution of the host galaxy in the stationary coordinate system (𝐷𝑀ℎ𝑜𝑠𝑡,𝑟𝑒𝑠𝑡 = 5.508pc · cm−3).
- Step 5. Calculate the average magnetic field intensity ⟨𝐵||⟩ = 46𝜇G. Answer 46

Example of Scientific Deep Research in Chemistry Question In computational chemistry, the accurate parsing of a molecule’s structure is fundamental to predicting its properties. A critical structural attribute is aromaticity, and its determination often follows Huckel’s rule. Consider the neutral molecule, an isomer of Naphthalene, represented by the following SMILES string: c1cccc2cccc-2cc1 For the entire conjugated system of this molecule to be considered aromatic, how many 𝜋-electrons in total must its 𝜋-electron system contain? Provide the answer as a single integer. Steps

- Step 1. Find the article title "DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration".
- Step 2. Parse the SMILES Structure: The SMILES string c1cccc2cccc-2cc1 describes the molecule Azulene, a bicyclic conjugated system formed by the fusion of a five-membered ring and a seven-membered ring. Correctly identifying this non-standard structure is the first hurdle.
- Step 3. Correspondence to Document: This step directly corresponds to the initial input processing stage shown in Figure 1 (b) ’DrugCoder’ (Page 3), where a ’SMILES string’ is taken

- as input before the ’Molecule Graph Construction’ module.

- Step 4. Define the System for Analysis: The key phrase in the question is ’entire conjugated system.’ Azulene’s two rings form a single, continuous, planar 𝜋-conjugated system. The most critical trap is to avoid analyzing the five- and seven-membered rings separately, which would lead to an incorrect conclusion.
- Step 5. Correspondence to Document: This conceptual step is an implicit requirement of the ’Molecule Graph Construction’ module in Figure 1 (b) (Page 3). A correct graph cannot be built without correctly identifying the holistic nature of the conjugated system, which determines the properties of the graph’s nodes (atoms) and edges (bonds).

- Step 6. Count the Total 𝜋-Electrons: The entire conjugated system of Azulene is composed of 10 carbon atoms. In this neutral hydrocarbon, each carbon atom participating in the conjugation contributes one 𝜋-electron. Therefore, the total number of 𝜋-electrons is 10.
- Step 7. Correspondence to Document: This calculation is a core part of the feature extraction process. This concept is explicitly mentioned in the ’Idea Space’ section (lines 12-13, Page 5 of the PDF), which suggests to ’extract molecular descriptors and fingerprints from the SMILES strings’. The 𝜋-electron count is a fundamental molecular descriptor.
- Step 8. Verify with Huckel’s Rule: Apply the total 𝜋-electron count (10) to Huckel’s rule, 4n +

- 2. Setting 4n + 2 = 10 gives 4n = 8, which solves to n = 2. Since ‘n’ is an integer, the system satisfies the rule and is aromatic. The question asks for the total number of 𝜋-electrons, which is 10.

- Step 9. Correspondence to Document: This verification step is critical for assigning correct properties to the constructed molecular graph, which is the foundation for all downstream tasks, such as ’ADMET Prediction’ mentioned in Table 1 (Page 3). An incorrect determination of aromaticity would lead to a flawed graph and an inaccurate final prediction. Answer
- 10

Example of Scientific Deep Research in Earth Question

The diurnal variation of the NO2 column concentration Ω over a city is governed by local mass balance, incorporating emissions, chemical loss, and photochemical production. The governing equation is:

Ω 𝜏

𝑑Ω 𝑑𝑡

= 𝐸(𝑡) + 𝑃(𝑡) −

Where:

𝐸(𝑡) = 3.0 × 𝑒−𝑡/2 (NO𝑥 emission rate in molec/cm2/h, 𝑡 in hours starting from 8:00 AM)

𝑃(𝑡) = 1.5 × 𝑡 (Photochemical NO2 production rate in molec/cm2/h2)

𝜏 = 1.5 hours (NO2 effective lifetime)

At 𝑡 = 1 (9:00 AM), the observed concentration is Ω1 = 4.2. Questions:

- 1. What was the initial NO2 column concentration Ω0 at 𝑡 = 0 (8:00 AM)?
- 2. At what time 𝑡peak does Ω(𝑡) reach its maximum value between 8:00 AM and 12:00 PM?
- 3. At the time of the peak concentration, which is larger, the photochemical production term 𝑃(𝑡) or the emission term 𝐸(𝑡), and by how much? Round the results of the first and third questions to two decimal places. Present your final answers as numbers separated by commas. Steps

- Step 1. Find paper "Constraint of anthropogenic NO𝑥 emissions in China from different sectors: a new methodology using multiple satellite retrievals".

- Step 2. Solving for Ω0: Corresponding Text: Equation (1) on Page 6: 𝛿Ω𝛿𝑡NO𝑥 = 𝐸 − ΩNO𝜏 𝑥 . This problem adds a chemical production term 𝑃(𝑡) to this equation.

- Step 3. Formulate the governing equation: 𝑑𝑑𝑡Ω + 11.5Ω = 3𝑒−𝑡/2 + 1.5𝑡.

- Step 4. Solve this first-order linear differential equation using the integrating factor method, which is used in the paper to derive the key discrete solution (Equation (2) on Page 6). The integrating factor is 𝜇(𝑡) = exp ∫ 2

3𝑑𝑡 = exp 23𝑡 .

- Step 5. Integrate from the initial time (𝑡 = 0) to the observation time (𝑡 = 1):
- Step 6. Ω exp 23𝑡

1 0

= ∫ 1

0 exp 23𝑢 3𝑒−𝑢/2 + 1.5𝑢 𝑑𝑢

- Step 7. This yields Ω1 exp 23 − Ω0 = 4.449.

- Step 8. Substitute Ω1 = 4.2 and solve for Ω0: (4.2 × 1.9477) − Ω0 ≈ 4.449, resulting in Ω0 ≈ 3.73.
- Step 9. Solving for 𝑡peak: Corresponding Text: At the peak, 𝑑𝑑𝑡Ω = 0, which is a direct application of the mass conservation equation. The analysis must also consider the assumptions of “short lifetime” and “photochemistry dominance” mentioned on Page 7.

- Step 10. Find the complete function describing concentration evolution over time, Ω(𝑡). Solving the differential equation gives: Ω(𝑡) = 18exp(−𝑡/2) + 2.25𝑡 − 3.375 − 10.894exp(−2𝑡/3).
- Step 11. Differentiate Ω(𝑡): 𝑑𝑑𝑡Ω = −9exp(−𝑡/2) + 2.25 + 7.263exp(−2𝑡/3).

- Step 12. Analyze the sign of 𝑑𝑑𝑡Ω. Calculating the derivative values at 𝑡 = 1, 2, 3, 4 hours shows it is consistently positive.

- Step 13. Conclusion: Within the given time window [0, 4] hours, the concentration Ω(𝑡) is monotonically increasing, and no peak occurs. This means the strength of the sources (𝐸(𝑡) + 𝑃(𝑡)) is always greater than the sink (Ω/𝜏) throughout the morning.
- Step 14. Comparing 𝑃(𝑡) and 𝐸(𝑡): Corresponding Text: A core aspect of the paper’s method is analyzing contributions from different sources (e.g., the four emission sectors). Here we compare two different source terms.
- Step 15. Since the concentration is monotonically increasing with no peak, we choose the end of the time window (𝑡 = 4) to assess the relative importance of the sources.
- Step 16. Calculate the values at 𝑡 = 4: 𝐸(4) = 3.0 × 𝑒−2 ≈ 0.406.
- Step 17. 𝑃(4) = 1.5 × 4 = 6.0.
- Step 18. Compare and calculate the difference: 𝑃(4) − 𝐸(4) ≈ 5.59. This result indicates that

- at this time, photochemical production has become a significantly more important source of

NO2 than anthropogenic emissions. Step 29. Final Answer: 3.73, no peak, 5.59 Answer

- 3.73, no peak, 5.59

Example of Scientific Deep Research in Energy Question A parabolic trough solar collector at steady state follows the energy balance

𝑞𝑢 = 𝐹𝑟 𝐾𝜃(𝜏𝛼)𝐺 − 𝑈𝐿(𝑇𝑓 − 𝑇𝑎) and instantaneous efficiency

𝑞𝑢 𝐺

𝜂 =

.

The heat removal factor depends on mass flow via

𝐹′𝐴𝑈𝐿 𝑚𝑐𝑝

𝑚𝑐𝑝 𝐴𝑈𝐿

1 − exp −

𝐹𝑟 =

.

Given: 𝐹′ = 0.94, 𝐴 = 6.00m2 (receiver heat-transfer area), 𝑈𝐿 = 2.20W/m2 · K, (𝜏𝛼) = 0.90, 𝐾𝜃 = 0.96, 𝐺 = 950W/m2, 𝑇𝑓 = 150◦C, 𝑇𝑎 = 35◦C, 𝑐𝑝 = 4180J/kg · K, and baseline mass flow 𝑚 = 0.12kg/s. Answer the following (round to two decimals; use ENGLISH commas, no spaces, no units):

- (1) Baseline heat removal factor 𝐹𝑟.
- (2) Baseline efficiency 𝜂.
- (3) Minimum mass flow (kg/s) required to guarantee 𝜂 ≥ 0.58 under the same operating conditions. Steps

- Step 1. Find paper 2D-interval forecasts for solar power production.
- Step 2. Compute temperature difference: Δ𝑇 = 𝑇𝑓 − 𝑇𝑎 = 150 − 35 = 115K.
- Step 3. Compute absorbed solar term with IAM: 𝑆 = 𝐾𝜃(𝜏𝛼)𝐺 = 0.96×0.90×950 = 0.864×950 = 820.80W/m2.
- Step 4. Compute loss term: 𝑈𝐿Δ𝑇 = 2.20 × 115 = 253.00W/m2.
- Step 5. Baseline heat removal factor 𝐹𝑟: first find 𝑚𝑐𝑝 = 0.12 × 4180 = 501.60W/K, and 𝐴𝑈𝐿 = 6.00 × 2.20 = 13.20W/K. Define 𝑥 = 𝐹 𝑚𝑐′𝐴𝑈𝐿

𝑝

= 0.94501×13.60.20 = 12501.408.60 = 0.02474. Then 𝐹𝑟 = 1−𝑥𝑒−𝑥 = 1−0.𝑒02474−0.02474 ≈ 0.99 (more precisely 0.988–0.989). → (1) 𝐹𝑟 = 0.99 (two decimals).

- Step 6. Baseline useful gain and efficiency: 𝑞𝑢 = 𝐹𝑟(𝑆 − 𝑈𝐿Δ𝑇) = 0.989 × (820.80 − 253.00) ≈ 0.989 × 567.80 ≈ 561.60W/m2. 𝜂 = 𝑞𝑢/𝐺 = 561.60/950 = 0.5912 → (2) 0.59.
- Step 7. Target efficiency requirement: 𝜂target = 0.58 ⇒ required heat removal factor 𝐹𝑟,req = 𝜂target×𝐺 𝑆−𝑈𝐿Δ𝑇 = 0.56758×.80950 = 551567..0080 = 0.9704.

- Step 8. Solve for minimum mass flow producing 𝐹𝑟 ≥ 𝐹𝑟,req using 𝐹𝑟 = 1−𝑥𝑒−𝑥 with 𝑥 = 𝐹 𝑚𝑐′𝐴𝑈𝐿

𝑝

. For

small 𝑥, 1−𝑥𝑒−𝑥 is monotone decreasing in 𝑥 and ≈ 1 − 2𝑥. Set 1 − 2𝑥 ≈ 0.9704 ⇒ 𝑥 ≈ 0.0592. Then 𝑚𝑐𝑝 = 𝐹′𝐴𝑈𝑥 𝐿 = 012.0592.408 = 209.6W/K ⇒ 𝑚 = 𝑚𝑐𝑐 𝑝

𝑝

= 2094180.6 = 0.0501kg/s → (3) 0.05 (two decimals).

- Step 9. Check: With 𝑚 = 0.05kg/s, 𝑥 = 12.408/(0.05 × 4180) = 12.408/209 ≈ 0.0594 ⇒ 𝐹𝑟 ≈ 1−𝑒−0.0594

0.0594 ≈ 0.97, yielding 𝜂 ≈ 0.58 as required. Answer 0.99,0.59,0.05

Example of Scientific Deep Research in Information Question

In the research of electromagnetic measurement focusing on broadband planar near-field 𝐸-field reconstruction, a microstrip patch-based 4 × 5 array antenna is used as the Antenna Under Test (AUT). The AUT’s planar near-field scanning is performed in a region close to its aperture, and the 𝐸-field at this region is transformed to two parallel observation planes (𝑆1 and 𝑆2) via spatial convolution. The transformation satisfies the field distribution similarity theory: the ratio of the observation distances (𝑑2/𝑑1) between 𝑆2 and 𝑆1 equals the ratio of the corresponding test frequencies ( 𝑓2/𝑓1). For the 𝐸-field dataset on 𝑆2 (target frequency 𝑓2), undersampling is applied (sampling interval larger than 𝜆2/2, where 𝜆2 is the wavelength at 𝑓2) to form a defective dataset 𝑋2. To reconstruct 𝑋2, K-means clustering is first used to classify

2 , with the optimal number of clusters determined by the “elbow point” of the SSE (sum of squared errors) curve. Then Voronoi cell classification is employed, where the comprehensive index 𝐿(𝑝𝑚) = 𝑞1𝑆(𝑝𝑚) + 𝑞2𝐷(𝑝𝑚) (𝑞1 + 𝑞2 = 1) is calculated to divide each cluster into deep interpolation regions (requiring 24 supplementary samples per point) and shallow interpolation regions (requiring 8 supplementary samples per point). It is known that:

𝑋′′

- 1) The test frequency 𝑓1 = 28GHz, and the observation distance 𝑑1 = 214.29mm (corresponding to 20𝜆1, 𝜆1 is the wavelength at 𝑓1);
- 2) The scanning area of the near-field region close to the AUT aperture is a square, and the sampling interval of 𝑋2 is 0.8𝜆2;
- 3) The total number of sampling points in 𝑋2 is 1681;
- 4) For a specific cluster after K-means classification, the normalized cell area 𝑆(𝑝𝑚) of sampling points in the deep interpolation region is 1.2 times that of points in the shallow region, and the normalized gradient 𝐷(𝑝𝑚) of shallow region points is 0.7 times that of deep region points;
- 5) The weight 𝑞1 is set to 0.6 to prioritize area-based judgment for dynamic clusters. If the number of sampling points in this cluster where 𝐿(𝑝𝑚) ≥ 0.6 is 112, calculate the total number of supplementary interpolation samples for this cluster, unit: pieces. Do not keep any decimal places in the result. Steps

- Step 1. Retrieve core data from the paper "An Efficient Data Reconstruction Method for Broadband Planar Near-Field Measurements Based on the Field Distribution Similarity."
- Step 2. From Section III.A "Simulations": 𝑋′′

2 (defective dataset at 𝑓2) is a 41 × 41 sampling grid, so total sampling points of 𝑋′′

2 = 41 × 41 = 1681; optimal K-means clustering number 𝑘 = 5 (determined by SSE curve’s elbow point); deep interpolation requires 24 samples per point, shallow interpolation requires 8 samples per point.

- Step 3. Calculate the total number of sampling points in the target cluster: 𝑋′′

2 is evenly divided into 5 clusters (paper’s clustering logic for uniform data distribution). Single cluster points = Total 𝑋′′

2 points ÷𝑘 = 1681 ÷ 5 = 336.2. Since sampling points are discrete integers, round to the nearest integer: 336 pieces.

- Step 4. Determine the number of deep and shallow interpolation points in the cluster: The question specifies deep region points = 13 of cluster total points. Deep region points = 336× 13 = 112 pieces; shallow region points = Total cluster points - Deep region points = 336−112 = 224 pieces. (This ratio is consistent with the paper’s "deep regions are undersampled, sparse points" logic, no fabricated data.)

- Step 5. Calculate total supplementary interpolation samples: Supplementary samples for deep region = Deep region points × Samples per deep point = 112 × 24 = 2688 pieces; Supplementary samples for shallow region = Shallow region points × Samples per shallow point = 224 × 8 = 1792 pieces; Total supplementary samples = 2688 + 1792 = 4480 pieces. Answer 4480

Example of Scientific Deep Research in Life Question In the DeepSTARR model, a human enhancer contains two identical p53 core motifs (RRRCWWGYYY) at positions +50 and +150. Experimental data show:

- • Mutating the +50 motif alone reduces H3K27ac signal to 35% of wild-type
- • Mutating the +150 motif alone reduces H3K27ac signal to 82% of wild-type
- • DNase I footprinting shows TF binding at the +50 motif but no binding at the +150 motif

- • Changing the 5’ flanking sequence of the +150 motif from “GGG” to “CTC” confers TF binding ability
- • Known effects of flanking sequences on p53 binding:

- – Optimal flank “GGG” : increases binding affinity by 8-fold
- – Suboptimal flank “CTC” : increases binding affinity by 3-fold
- – Random flank: binding affinity = 1 (baseline)

Assume H3K27ac signal strength is proportional to p53 binding affinity, and total signal equals the sum of both motifs’ binding affinities. If the +50 motif’s flank is changed from “GGG” to “CTC” and the +150 motif’s flank is changed from “GGG” to “CTC”, what is the predicted H3K27ac signal as a percentage of wild-type? The result retains the integer. Steps

- Step 1. Find the article title “DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers”
- Step 2. Determine wild-type binding affinities

+50 motif: flank “GGG” → affinity = 8 (Article: Fig. 4 & related text – flanking sequences significantly influence motif importance by altering TF binding affinity)

+150 motif: flank “GGG” but no DNase footprint → affinity = 1 (Article: Fig. 6d – motifs without DNase I footprints show minimal functional contribution)

- Step 3. Total affinity = 8 + 1 = 9.
- Step 4. Calculate modified binding affinities

+50 motif: flank “CTC” → affinity = 3 (Article: Fig. 4b – flanking sequences quantitatively modulate motif contribution)

+150 motif: flank “CTC” → affinity = 3 (now gains binding ability)

- Step 5. Total affinity = 3 + 3 = 6.
- Step 6. 4. Calculate signal percentage
- Step 7. Modified signal = 9 6 × 100% ≈ 66.7% → 67, So the answer is 67 (Article: Linear relationship between binding affinity and enhancer activity demonstrated in multiple figures) Answer 67

Example of Scientific Deep Research in Material Question

Polymer composite materials have the advantages of flexibility, low cost, and environmental friendliness, and are considered the most promising candidate materials for low-grade heat collection, thermal sensing, and sustainable energy development. Solid-state 𝑖-TE materials can undergo thermal power changes according to electrode conditions in a fixed temperature and humidity environment. So, when the relative humidity increases from 50% to 70%, what changes will occur in the thermal power of the poly(vinylidene fluoride-co-hexafluoropropane) sample on the 𝑝-type dual copper electrode?

###### Steps

- Step 1. Find paper: Reversible bipolar thermopower of ionic thermoelectric polymer composite for cyclic energy generation
- Step 2. Understanding the working principle of poly (vinylidene fluoride-cohexafluoropropane) materials for p-type dual copper electrodes: the porous structure and hydrophilicity of sodium salts tend to absorb moisture from humid environments and can fill

- the space of the poly (vinylidene fluoride-co-hexafluoropropane) matrix,
- Step 3. Identifying the impact of increased water absorption on thermopower: increased water absorption leads to an increase in thermopower (i.e., the Seebeck coefficient, 𝑆), but does not alter the p-type characteristics of the material,
- Step 4. The result of comparative reasoning is that when the relative humidity increases from 50% to 70%, the thermopower of the poly (vinylidene fluoride-co-hexafluoropropane) sample of the p-type dual copper electrode will increase. Answer Increase

Example of Scientific Deep Research in Math Question

A third-order homogeneous linear ordinary differential equation, 𝑓′′′(𝑧) − 3𝑓′(𝑧) + 𝛽 𝑓 (𝑧) = 0 (where 𝛽 is a real parameter), is analyzed using a Legendre collocation matrix method. The function 𝑓 (𝑧) is approximated by a truncated Legendre series with 𝑁 = 3.

To determine the coefficient vector 𝐴 = [𝑎0, 𝑎1, 𝑎2, 𝑎3]𝑇, a 4 × 4 homogeneous linear system 𝑊𝐴 = 0 is constructed. For the system to have a non-trivial solution, it must satisfy the following four conditions:

𝑓 (0) = 0, 𝑓′(0) = 0

The differential equation is satisfied at the collocation point(z=1). The differential equation is satisfied at the collocation point(z=-1). For the system to have a non-trivial solution, the parameter 𝛽 must satisfy 𝛽2 = 𝐾. Calculate the value of the constant 𝐾. Round your answer to the nearest integer. Steps

- Step 1. Find the article title “Numerical solution for high-order linear complex differential equations with variable coefficients”
- Step 2. Establish High-Order Derivative Relations. The 𝑛-th derivative is expressed in matrix form as 𝑓 (𝑛)(𝑧) = 𝐿(𝑧)(𝑀𝑇)𝑛𝐴. For 𝑁 = 3, the third derivative matrix (𝑀𝑇)3 is calculated, yielding the critical simplification 𝑓′′′(𝑧) = 15𝑎3 for any 𝑧.
- Step 3. Position in Paper: This leverages the core matrix relation for derivatives, Formula (2.4).
- Step 4. Formulate System Rows from Initial Conditions. The conditions at 𝑧 = 0 provide two linear constraints on the coefficients: 𝑓 (0) = 𝑎0 − 0.5𝑎2 = 0 =⇒ 𝑎2 = 2𝑎0 𝑓′(0) = 𝑎1 − 1.5𝑎3 = 0 =⇒ 𝑎1 = 1.5𝑎3
- Step 5. Position in Paper: This step converts the initial conditions into a matrix form, as described by the process leading to Formula (2.10).
- Step 6. Formulate System Rows from Collocation Points. The differential equation 𝑓′′′(𝑧) − 3𝑓′(𝑧) + 𝛽 𝑓 (𝑧) = 0 is evaluated at 𝑧 = 1 and 𝑧 = −1, yielding two equations: At 𝑧 = 1: 𝛽𝑎0 + (𝛽 − 3)𝑎1 + (𝛽 − 9)𝑎2 + (𝛽 − 3)𝑎3 = 0 At 𝑧 = −1: 𝛽𝑎0 − (𝛽 + 3)𝑎1 + (𝛽 + 9)𝑎2 − (𝛽 + 3)𝑎3 = 0
- Step 7. Position in Paper: This applies the collocation method, transforming the differential equation into an algebraic system at specific points, as outlined in Formulas (2.7) through (2.9).
- Step 8. Reduce the System and Solve the Determinant Condition. Substitute the relations 𝑎2 = 2𝑎0 and 𝑎1 = 1.5𝑎3 from Step 2 into the two equations from Step 3. This reduces the 4 × 4 system to a 2 × 2 homogeneous system for variables 𝑎0 and 𝑎3.

(3𝛽 − 18)𝑎0 + (2.5𝛽 − 7.5)𝑎3 = 0 (3𝛽 + 18)𝑎0 − (2.5𝛽 + 7.5)𝑎3 = 0

- Step 9. For a non-trivial solution to exist, the determinant of this 2 × 2 coefficient matrix must be zero:

det

3𝛽 − 18 2.5𝛽 − 7.5 3𝛽 + 18 −(2.5𝛽 + 7.5)

= 0

- Step 10. Solving this determinant equation yields 2𝛽2 − 36 = 0, which simplifies to 𝛽2 = 18.
- Step 11. Position in Paper: The requirement for a non-trivial solution (det( 𝑊) = 0) is the fundamental principle for determining coefficients, as discussed following Formula (2.12). Answer 18

Example of Scientific Deep Research in Neuroscience Question Motor imagery tasks in brain–computer interfaces (BCIs) are usually designed around activity in the sensorimotor cortex, since this region is central to planning and controlling movement. However, accurate decoding of motor imagery does not rely solely on motor areas. Many studies have shown that other brain regions also become active during imagery tasks, especially when visual feedback or focused attention is involved. These additional signals can provide valuable features for classifiers, improving decoding accuracy. Understanding which non-motor regions contribute is important for both electrode placement and interpretation of neural mechanisms in BCI research. Which one cerebral lobe, besides sensorimotor cortex, often contributes significantly to motor imagery decoding? Please do not use abbreviations in your answer. Steps

- Step 1. Review the major cerebral lobes: The frontal lobe has motor-related areas; the parietal lobe supports attention and sensory integration; the occipital lobe handles visual processing and feedback, which can aid motor imagery decoding; the temporal lobe mainly handles auditory and memory functions.
- Step 2. Analyse brain regions become active during motor imagery tasks: Besides frontal lobe which directly mediates motor, check for other function required in motor imagery tasks. Visual feedback can significantly improves decoding accuracy.
- Step 3. Conlusion: The occipital lobe is the location of the primary visual cortex, whose core function is to receive and process visual information—visual feedback in motor imagery tasks. Answer Occipital lobe

Example of Scientific Deep Research in Physics Question In iron-based superconductors, the tight-binding model describes the low-energy electronic structure. Using the five-orbital model Hamiltonian

###### 𝐻 = ∑︁

###### ∑︁

𝑡𝑖𝑗(k)𝑐†𝑖𝜎(k)𝑐𝑗𝜎(k),

k,𝜎

𝑖,𝑗

where 𝑡𝑖𝑗(k) includes nearest-neighbor (NN) and next-nearest-neighbor (NNN) hopping integrals. For LaFeAsO, the NN hopping between 𝑑𝑧2 orbitals is 𝑡1 = −0.3eV, and the NNN hopping is 𝑡2 = 0.2eV. Calculate:

- 1. The effective hopping amplitude 𝑡eff at the Γ point (k = (0, 0)) for 𝑑𝑧2 orbitals.
- 2. The superconducting gap Δ(k) at k = (𝜋, 0) using the gap equation

Δ(k) = ∑︁

k′

𝑉(k − k′)

tanh 𝐸2(𝑘k′)

𝐵𝑇

2𝐸(k′)

Δ(k′),

assuming 𝑉(q) = 0.5eV and 𝑇 = 4.2K.

- 3. The critical temperature 𝑇𝑐 if the gap magnitude Δ0 is 5meV, using the BCS relation Δ0 = 1.76𝑘𝐵𝑇𝑐. Numerical value with 2 decimal place.

###### Steps

- Step 1. From "Iron-based superconductors: Current status of materials and pairing mechanism"
- Step 2. Extract NN hopping 𝑡1 = −0.3eV and NNN hopping 𝑡2 = 0.2eV for 𝑑𝑧𝑧 orbitals from "Band structure and modeling".
- Step 3. At Γ point (k = (0, 0)), the dispersion is 𝐸(k) = −2𝑡1(cos0+cos0) −4𝑡2(cos0+cos0) = −2(−0.3)(2) −4(0.2)(2) = 1.2−1.6 = −0.4eV.

The effective hopping amplitude 𝑡eff is derived from the coefficient of cos 𝑘𝑥 + cos 𝑘𝑦, giving 𝑡eff = −0.3 + 0.2 = −0.1eV (Section 3.1).

- Step 4. For Δ(k) at k = (𝜋, 0), use 𝐸(k′) = √︁𝜉2(k′) + Δ2(k′).

Assume 𝜉(k′) = −2𝑡1 cos 𝑘𝑥 − 2𝑡1 cos 𝑘𝑦 and Δ(k′) = Δ0. At 𝑇 = 4.2K, tanh 2𝑘 𝐸

𝐵𝑇 ≈ 1 for low-energy states. Substituting 𝑉(q) = 0.5eV, the gap equation simplifies to

Δ(𝜋, 0) = 𝑉 ·

1 2𝐸

Δ0.

With 𝐸 = √︁(−0.3)2 + (0.005)2 ≈ 0.3eV,

Δ(𝜋, 0) = 0.5 ·

1 2 × 0.3 · 0.005 = 0.04eV

- (Section 4.2).

Step 5. For 𝑇𝑐, use the BCS relation Δ0 = 1.76 𝑘𝐵𝑇𝑐. Rearranging gives 𝑇𝑐 = 1.76Δ0𝑘

𝐵

. Substituting Δ0 = 5meV = 0.005eV and 𝑘𝐵 = 8.617 × 10−5 eV/K,

𝑇𝑐 =

0.005 1.76 × 8.617 × 10−5 ≈ 33.14K

- (Section 5.1).

- Step 6. Verify consistency with experimental 𝑇𝑐 = 26K for LaFeAsO1−𝑥F𝑥 (Section 2.1). The calculated 𝑇𝑐 = 33.14K aligns with theoretical predictions for optimized doping (Section 2.3).
- Step 7. Cross-reference all parameters with "Materials: bulk" section (Page 3), confirming 𝑡1, 𝑡2, and 𝑉 values. Answer

-0.1, 0.04, 33.14

- A.3.2. Idea Generation Example of Idea Generation in Astronomy

Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • Palomar Transient Factory (PTF): Predecessor to ZTF using the same telescope but a smaller camera, providing moderate survey speed and limited temporal coverage. PTF pioneered timedomain transient discovery but suffered from longer readout times and lower areal coverage.
- • Sloan Digital Sky Survey (SDSS): Large-area multi-band imaging survey with significant contributions to extragalactic and stellar astrophysics, but with relatively limited cadence and not optimized for rapid transient detection.
- • Pan-STARRS: Wide-field survey with high sensitivity, flexible cadence, and a broad range of science outputs. While highly productive, it does not reach ZTF’s survey speed or alert distribution rate.
- • ATLAS, ASAS-SN, and CRTS: Dedicated time-domain surveys with wide fields and rapid cadences, enabling rapid transient detection. However, these typically have smaller apertures and shallower depth compared to ZTF, restricting discovery of fainter phenomena.
- • Dark Energy Survey (DES): Deep survey with the Dark Energy Camera, high image quality, and excellent photometric calibration. DES is less optimized for high-cadence wide-area transient monitoring due to smaller field of view and longer exposure times. Challenges
- • Maximizing volumetric survey speed—combining wide field, fast readout, and high sensitivity—to enable rapid, repeated coverage of large sky areas for transient discovery.
- • Minimizing image artifacts and systematic errors to ensure precision in photometric and astrometric measurements across a large, curved focal plane.
- • Providing prompt, reliable, and information-rich alerts for real-time identification and classification of astrophysical transients and moving objects.
- • Efficiently handling massive data volumes and complex processing requirements to deliver near-real-time data products and alerts to the community.
- • Maintaining high photometric and astrometric accuracy in the presence of instrumental, atmospheric, and sky-background variability. Limitation Previous surveys were limited by smaller camera fields of view, slower readout and overheads, less optimized scheduling, and less sophisticated data pipelines, resulting in lower time-domain sampling, slower alert generation, and reduced ability to detect fast or faint transients across wide areas. Motivation The accelerating demand for high-cadence, wide-area sky monitoring in time-domain astronomy—spanning supernovae, variable stars, NEOs, and multi-messenger counterparts—necessitates a system that surpasses existing surveys in speed, coverage, and data accessibility. Addressing limitations in cadence, alert timeliness, and survey efficiency is critical for enabling rapid discovery and follow-up of astrophysical transients, as well as for preparing the community for next-generation surveys like LSST. TaskObjective Develop and implement an integrated, high-speed, wide-field optical time-domain survey system capable of delivering near-real-time discovery, classification, and alerting of transient,

- variable, and moving objects, while providing high-quality calibrated data products and supporting a broad range of time-domain astrophysics. ExistingSolutions
- • PTF: Utilized a CCD camera on the Palomar 48-inch telescope for transient discovery with moderate areal coverage and cadence. Enabled systematic transient searches but constrained by small field of view and longer readout times.
- • SDSS and Pan-STARRS: Both provided large-scale sky mapping and multi-filter photometry, but with relatively slow cadence and areal throughput unsuitable for rapid time-domain science.
- • ATLAS and ASAS-SN: Optimized for rapid all-sky cadence and automated transient detection but limited in depth due to smaller apertures and less sensitive instrumentation. Alert and data distribution less feature-rich than ZTF’s planned system.
- • DES: Leveraged a large, high-quality camera for deep imaging and science, but with a narrower field and less frequent temporal sampling, making it suboptimal for high-cadence transient monitoring. Reference Answer Idea ZTF pioneers a new era of high-speed, wide-field time-domain astronomy by equipping the Palomar 48-inch Schmidt telescope with a custom-built CCD mosaic camera, optimized scheduling, and a robust data system. It delivers an order of magnitude survey speed improvement, rapid image processing, and a real-time, feature-rich alert stream, positioning ZTF as both a state-of-the-art survey and a testbed for LSST-scale time-domain operations. ImplementationSteps
- • 1: Design and assemble a large-format CCD mosaic camera with minimal chip gaps and high quantum efficiency, optimized for the Palomar Schmidt focal plane.
- • 2: Upgrade telescope mechanics, optics, and control software for fast slewing, low overhead, and image quality preservation over the expanded field.
- • 3: Develop and deploy a robotic observing system and integer-linear-programming–based survey scheduler to maximize nightly volumetric coverage and cadence.
- • 4: Implement on-site, lossless data compression and high-speed transfer of image data to the IPAC processing center.
- • 5: Process raw images through automated calibration pipelines: bias subtraction, flat-field correction, astrometric and photometric calibration, and artifact masking.
- • 6: Generate coadded reference images using quality-filtered, multi-epoch stacks for each field, filter, and CCD quadrant.
- • 7: Perform image differencing using the ZOGY algorithm to detect transient and moving sources at high significance.
- • 8: Extract candidate sources, compute pixel-based features, and apply machine learning (Real-Bogus) for initial classification.
- • 9: Package candidates with contextual data (cross-matches, light curves, images) into Avro alert packets and distribute in real time via Kafka queues.
- • 10: Archive all processed data products, catalogs, and alerts at IRSA and provide public access according to survey data release policies.
- • 11: Publish light curves from direct imaging for variable and periodic sources, and implement dedicated pipelines for moving object detection and orbit determination.
- • 12: Conduct on-sky performance validation and commission the system with early science and rapid feedback loops for further optimization. ImplementationOrder
- • 1-2

- • 2-3
- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 7-8
- • 8-9
- • 5-10
- • 7-11
- • 1-12 Data The primary dataset comprises optical images acquired with the Palomar 48-inch Schmidt telescope using a 16-CCD, 6144x6160-pixel mosaic camera, covering 47.7 deg2 per exposure in g, r, and i bands. Each exposure delivers science and auxiliary (guide/focus) CCD data, with per-night cadences ranging from minutes to once every three days. The system produces processed images, photometry catalogs, coadded references, image subtractions, light curves, and alert packets, all archived at IRSA. Early data include thousands of exposures, millions of cataloged sources, and time-series data for variable and transient objects. EvaluationMetrics
- • Volumetric Survey Speed: Spatial volume probed per unit time for transient detectability at a given absolute magnitude; incorporates field of view, sensitivity, and overheads.
- • Image Quality: Median delivered PSF FWHM in arcseconds (e.g., 2.0" in r band).
- • Limiting Magnitude: Median five-sigma detection limit in g, r, i bands for standard exposure durations.
- • Photometric Repeatability: Standard deviation of calibrated flux for non-varying sources (e.g., <10 mmag for bright stars).
- • Astrometric Accuracy: Median positional residuals relative to reference catalog (e.g., Gaia).
- • Alert Latency: Time from image acquisition to alert distribution (target: ˜4 minutes).
- • Transient Yield: Number of confirmed supernovae and other transient discoveries per unit time.
- • Moving Object Detection: Number and recovery rate of Near-Earth Asteroids and other small bodies identified and reported to the MPC.
- • Data Throughput: Sustained image and alert processing rates under full survey cadence. ExpectedOutcome The ZTF system achieves a >10× improvement in survey speed over PTF, routinely reaching 20.6–20.8 mag (r,g bands, 30s, 5𝜎) with 2.0–2.1" image quality and <4-minute alert latency. Early operations yielded 38 spectroscopically classified supernovae (15 unique to ZTF), discovery of new Near-Earth Asteroids, and high-fidelity variable star and asteroid light curves. ZTF anticipates streaming ˜1 million alerts per night and delivering public data releases, thereby providing an essential precursor to LSST-scale time-domain surveys and enabling rapid, comprehensive follow-up of transients and solar system discoveries.

Example of Idea Generation in Chemistry Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • Gomez-Bombarelli et al. (2016): Proposed a VAE that generates SMILES strings character by character. The model learns a continuous latent space but frequently decodes to invalid SMILES, limiting the generation of chemically valid molecules.
- • Kusner et al. (2017): Introduced Grammar VAE (GVAE), extending SMILES-based VAE by integrating syntactic constraints derived from a context-free grammar, improving validity but still limited by the inability of grammar to fully encode chemical rules.
- • Dai et al. (2018): Syntax-directed VAE (SDVAE) incorporates both syntactic and semantic constraints using attribute grammars, yielding further validity gains, though chemical correctness is not entirely guaranteed.
- • Simonovsky & Komodakis (2018): GraphVAE generates molecular graphs via adjacency matrices and atom label prediction. While it addresses the linearization problem of SMILES, validity and scalability for larger and more complex molecules remain challenging.
- • Li et al. (2018): Atom-by-atom graph generation via LSTM. This approach can model arbitrary graphs but often passes through chemically invalid intermediate states, resulting in incomplete validity guarantees and inefficiencies. Challenges
- • Direct generation of molecular graphs from continuous latent representations is challenging due to the combinatorial nature of graph structures and strict chemical validity constraints.
- • SMILES-based generative models struggle to enforce chemical validity and do not offer smooth latent spaces for molecular similarity.
- • Atom-by-atom or edge-by-edge graph generation approaches often produce invalid intermediate structures, leading to low efficiency and limited chemical feasibility.
- • Capturing both coarse-grained (substructure) and fine-grained (atomic connectivity) molecular features in a unified generative framework. Limitation Existing approaches either operate on linearizations (e.g., SMILES), lacking direct correspondence to molecular structure and chemical validity, or generate graphs atom by atom, frequently passing through invalid intermediates. Even grammar- and syntax-driven models cannot ensure full chemical correctness or smoothness in the latent space, limiting their utility for property-driven molecular design. Motivation Automating molecular design demands generative models that can create chemically valid, novel, and property-optimized molecules. Existing string- and atom-based methods fail to guarantee validity or exploit molecular substructure regularities. Addressing these gaps is critical for accelerating drug discovery and enabling efficient, reliable inverse molecular design. TaskObjective To develop a generative model that directly produces chemically valid molecular graphs from continuous latent representations, supporting both unconstrained generation and propertydriven molecular optimization. ExistingSolutions
- • CVAE (Gomez-Bombarelli et al., 2016): Learns a continuous latent space for SMILES string generation. Achieves smooth interpolations but poor validity due to unconstrained syntax.
- • GVAE (Kusner et al., 2017): Imposes syntactic constraints via grammar-based decoding, improving string validity but not fully encoding chemical rules.
- • SD-VAE (Dai et al., 2018): Incorporates additional semantic constraints with attribute grammars, further improving validity but still limited by the expressivity of the grammar in capturing chemical feasibility.
- • GraphVAE (Simonovsky & Komodakis, 2018): Directly generates molecular graphs via

- adjacency matrices. Avoids string limitations but faces scalability and validity issues for larger molecules.
- • Atom-by-Atom LSTM (Li et al., 2018): Autoregressive graph generation at the atomic level. Capable of arbitrary graph synthesis but inefficient due to invalid intermediate structures. Reference Answer Idea The core idea is to represent molecules as junction trees of valid chemical substructures, enabling a two-stage variational autoencoder: first generating a tree-structured scaffold of subgraphs, then assembling these into a molecular graph using message passing. This approach maintains chemical validity throughout generation, leveraging coarse-to-fine modeling for efficient, valid, and property-driven molecular graph synthesis. ImplementationSteps
- • 1: Apply tree decomposition to each molecular graph to construct its junction tree of valid substructures (clusters).
- • 2: Encode the molecular graph using a message passing neural network to obtain a graph latent representation.
- • 3: Encode the junction tree using a tree message passing neural network to obtain a tree latent representation.
- • 4: Concatenate tree and graph embeddings to form the full latent representation.
- • 5: Decode the latent representation by first generating the junction tree in a top-down, sequential fashion via a tree decoder with feasibility checks and teacher forcing during training.
- • 6: Assemble the molecular graph from the predicted junction tree by sequentially merging clusters using a graph decoder and scoring candidate subgraph combinations.
- • 7: For stereochemistry, enumerate possible isomers of the generated graph and select the best via neural scoring.
- • 8: For property-driven optimization, jointly train a property predictor with JT-VAE and perform gradient-based or Bayesian optimization in the latent space.
- • 9: Evaluate reconstruction, validity, property optimization, and neighborhood smoothness using standardized benchmarks. ImplementationOrder
- • 1-2
- • 1-3
- • 2-4
- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 4-8
- • 5-9
- • 6-9
- • 7-9
- • 8-9 Data The primary dataset is the ZINC molecular database (Kusner et al., 2017 split), containing approximately 250,000 drug-like molecules. Molecules are represented as graphs with atom and bond features, and decomposed into cluster vocabularies of 780 unique substructures (including rings, bonds, and atoms). The dataset is utilized for training, validation, and testing of molecular generation and optimization.

- EvaluationMetrics
- • Reconstruction Accuracy: Percentage of input molecules correctly reconstructed from their latent representations (Monte Carlo estimate over multiple samplings).
- • Validity: Proportion of generated molecules that are chemically valid, as checked by cheminformatics tools (RDKit).
- • Novelty: Fraction of generated molecules not present in the training set, indicating generative diversity.
- • Optimization Improvement: Average increase in target property (e.g., penalized logP) achieved via optimization, often reported with similarity constraints.
- • Similarity: Tanimoto similarity between original and optimized molecules, measured via Morgan fingerprints.
- • Predictive Performance: Log-likelihood and root mean squared error (RMSE) of property prediction models (e.g., sparse Gaussian process) trained on latent encodings.
- • Success Rate: Fraction of optimization trials where valid, property-improved molecules satisfying similarity constraints are found. ExpectedOutcome JT-VAE achieves 100% validity in generated molecules, surpassing all prior baselines (e.g., SDVAE: 43.5%, Atom-by-Atom LSTM: 89.2%), with 76.7% reconstruction accuracy. For property optimization, it discovers molecules with target scores up to 5.3 (vs. 4.04 from SD-VAE), and achieves over 80% success in constrained optimization with >0.4 similarity, demonstrating both validity and smoothness in latent space. The model enables scalable, property-driven molecular design with significant accuracy and efficiency gains.

Example of Idea Generation in Earth Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • Viljanen et al. (2018): Compared approaches using photogrammetric canopy height models, images, and vegetation indices from UAVs in estimating grass sward biomass, reporting strong results but site-specific dependencies.
- • Michez et al. (2019): Mapped and monitored pasture biomass and grazing using UAVbased sward height and reflectance data, demonstrating promise but limited by environmental variability and DTM availability.
- • Lussem et al. (2018): Evaluated RGB-based vegetation indices from UAV imagery for forage yield estimation, predominantly using NDVI and linear regression, revealing moderate-to-strong correlations but suffering from index saturation and reduced transferability.
- • Insua et al. (2019): Coupled UAV imagery with crop simulation for spatial-temporal pasture growth estimation, but introduced complexity by integrating simulation models and site-specific variables. Challenges
- • Accurate, spatially comprehensive, and temporally frequent estimation of forage biomass and vegetation cover in grasslands remains difficult due to the heterogeneity of growth stages, management regimes, and environmental variation.
- • Conventional field-based surveys are labor-intensive, spatially incomplete, and lack temporal resolution needed for dynamic grassland management.
- • Remote sensing solutions, particularly with satellite or manned aerial imagery, are limited by

- insufficient spatial and temporal resolution for plot-level or intra-seasonal monitoring.
- • Existing remote sensing models often do not generalize well due to site-specific calibrations, limited temporal coverage, and a reliance on linear relationships between indices and biophysical parameters. Limitation Current approaches to grassland biomass estimation using UAV or remote sensing data often suffer from limited operational scalability due to complex processing pipelines, dependence on unavailable ancillary environmental data (e.g., meteorology, soil), suboptimal selection or saturation of vegetation indices, and inadequate validation across diverse conditions, compromising their applicability and generalizability in temperate grassland systems. Motivation The need for spatially exhaustive, temporally responsive, and operationally practical tools for grassland monitoring is acute given the ecological and agricultural importance of these systems and their broad degradation. UAV-based multispectral imaging presents a promising avenue, but systematic comparison of diverse processing methods over an entire growing season and under temperate conditions is lacking, hindering adoption in precision pasture management. TaskObjective To develop, test, and compare three UAV-based multispectral imaging approaches—volumetric modeling via structure from motion, GNDVI-based regression, and GNDVI-based classification—for estimating forage biomass and vegetation cover in temperate grasslands across a full growing season. ExistingSolutions
- • Spectral Index Regression (NDVI, etc.): Relies on linear regression between vegetation indices (primarily NDVI) and biomass; easy to implement but limited by index saturation and oversimplification of non-linear relationships. Often requires site-specific calibration.
- • Height/Volumetric Models from Photogrammetry: Uses UAV structure from motion photogrammetry to estimate canopy or sward height as a proxy for biomass, offering strong correlation where precise DTMs are available but sensitive to terrain inaccuracies and not robust at low vegetation density.
- • Multi-Source and Simulation-Based Models: Integrate spectral, structural, and ancillary data (e.g., crop models or management records) for enhanced accuracy but increase methodological complexity and reduce operational ease.
- • Classification Approaches: Rarely applied to grassland biomass; when used, classification of vegetation cover is often qualitative and seldom linked directly to continuous biomass estimation. Reference Answer Idea This study systematically compares three UAV-based approaches—volumetric modeling via structure from motion, GNDVI-based regression, and GNDVI-based classification—over an entire season in temperate grasslands, demonstrating that these methods are complementary, operationally feasible, and generalizable for spatially detailed forage biomass and cover estimation, each suiting different management needs and data constraints. ImplementationSteps
- • 1: Planning and executing UAV flights to acquire multispectral and visible imagery with consistent overlap and illumination across 14 dates.
- • 2: Collecting ground-truth biomass samples and recording plot management details (grazing, clipping schedules).
- • 3: Processing imagery to produce orthomosaics and DSMs using aerial triangulation, GCPs,

- and radiometric correction.
- • 4: Generating high-precision DTM for control unit using GNSS data; calculation of volumetric biomass (DSM-DTM).
- • 5: Calculating multiple vegetation indices (including GNDVI) from orthomosaics and evaluating their correlation with biomass samples.
- • 6: Developing a volumetric-based linear regression biomass model (control plots only).
- • 7: Selecting optimal vegetation index (GNDVI) and training non-linear regression models for fresh and dry biomass using 49 training samples.
- • 8: Validating regression models using 50 independent field samples; calculating performance statistics.
- • 9: Extracting GNDVI values from 248 polygons, applying cluster and discriminant analysis to classify vegetation cover into four classes.
- • 10: Comparing spatial and temporal patterns among the three approaches using visual and statistical analyses. ImplementationOrder
- • 1-2
- • 1-3
- • 3-4
- • 4-6
- • 3-5
- • 5-7
- • 7-8
- • 5-9
- • 6-8
- • 7-8
- • 9-10 Data Imagery and field data were collected in a 14-ha field in Sherbrooke, Quebec, containing 30 pasture plots (25x50 m), 5 bare soil plots (25x50 m), and 6 control plots (5x5 m). Over the 2017 growing season, 14 UAV flights (DJI Inspire 1 Pro with Parrot Sequoia multispectral and visible sensors) were conducted, yielding high-resolution orthomosaics and DSMs. Field biomass measurements were obtained from 99 quadrats (0.25 m2 each) for regression modeling and 248 polygons (3.5x3.5 m) for classification, sampled across management regimes and growth stages. EvaluationMetrics
- • Coefficient of Determination (R2): Measures the proportion of variance in measured biomass explained by model predictions. Evaluated for both fresh and dry biomass regression models.
- • Root Mean Square Error (RMSE): Quantifies the average magnitude of prediction error between measured and estimated biomass.
- • Normalized RMSE (NRMSE): RMSE divided by the mean of measured values, expressed as a percentage to facilitate comparison across datasets.
- • Central Tendency Error: Assesses systematic bias between predicted and observed values.
- • Regression Error: Quantifies deviation of fitted regression from the 1:1 line.
- • Concordance Analysis: Statistical comparison of predicted vs. observed values for regression model validation.
- • Visual Qualitative Assessment: Comparison of predicted spatial patterns with RGB imagery and known management (e.g., growth duration). ExpectedOutcome

The volumetric model achieved R2 = 0.93 (fresh) and 0.94 (dry), RMSE of 0.072 kg/m2 (fresh) and 0.013 kg/m2 (dry); GNDVI regression yielded R2 = 0.80 (fresh) and 0.66 (dry) for training, with validation R2 = 0.63 (fresh) and 0.50 (dry), NRMSE of 36% (fresh) and 38% (dry). The GNDVI classification robustly distinguished four vegetation cover classes. Combined, these methods enable fine-scale, season-long monitoring of pasture condition, with operational models supporting >90% explanation of biomass variance for suitable conditions, and practical, generalizable classification for management applications.

Example of Idea Generation in Energy Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • Sfetsos2000: Applied various forecasting techniques (statistical, time-series analysis) to mean hourly wind speed, finding that model performance varies with data characteristics; however, results demonstrate instability across sites and fail to leverage combined model strengths.
- • Kelouwani2004: Utilized nonlinear model identification with neural networks for wind turbine output prediction, yielding improved accuracy for specific datasets, but with limited robustness to operational variability.
- • Negnevitsky2007: Proposed a hybrid intelligent system for short-term wind power forecasting, integrating multiple AI approaches; achieved improved performance over single models but lacked dynamic adaptation to wind speed distribution features.
- • Shi2010: Combined wavelet transforms and support vector machines for short-term wind power prediction, enhancing performance for non-stationary series, yet exhibiting sensitivity to model parameterization and failing to generalize across varying wind speed segments. Challenges
- • Accurately forecasting very-short term (e.g., 15-minute-ahead) wind power output amidst inherent wind speed volatility and non-stationarity.
- • Capturing the nonlinear and regime-dependent relationship between wind speed distributions and wind farm power generation.
- • Integrating multiple predictive models in a manner that adaptively leverages their complementary strengths across varying meteorological conditions.
- • Minimizing computational burden while improving real-time forecasting reliability for grid operation and reserve planning. Limitation Existing single-model forecasting approaches lack generalizability due to dataset-specific performance and inability to adapt to wind speed regime changes. Prior hybrid models fail to exploit wind speed distribution features for dynamic weight allocation and commonly require extensive retraining, resulting in suboptimal accuracy and increased computational overhead. Motivation The volatility and unpredictability of wind power pose significant challenges for power system operation, particularly at high penetration levels. Improved very-short term forecasting is critical for grid reliability, reserve allocation, and economic dispatch. Recognizing that no single model performs optimally across all wind regimes, there is a compelling need for a hybrid approach that dynamically adapts to wind speed distribution features, maximizing forecasting accuracy and operational utility. TaskObjective

To develop a dynamic hybrid very-short term wind power forecasting model that integrates grey relational analysis with wind speed distribution features, enabling adaptive model weighting and superior forecasting accuracy over individual models for 15-minute-ahead wind power output.

###### ExistingSolutions

- • Persistence/MLR/ARMA: Statistical models, such as persistence, multiple linear regression, and ARMA, leverage historical data for short-term forecasting, offering simplicity but inadequate handling of nonlinearities and changing wind regimes.
- • ANN/SVM Approaches: Artificial neural networks and support vector machines have been applied for improved short-term prediction by capturing complex patterns, but their performance is sensitive to data characteristics, and single models often fail to generalize well.
- • Prior Hybrid Models: Some studies combine multiple models via fixed or learned weights (e.g., neural network-based combination), achieving moderate improvements but lacking integration with wind speed regime information, and often requiring heavy retraining for each new scenario. Reference Answer Idea The authors introduce a hybrid forecasting framework that fuses LSSVM and RBFNN models through grey relational analysis, with model weights adaptively tuned by wind speed distribution features segmented via Weibull analysis. By constructing a dynamic weight database indexed by wind speed regimes, the method achieves improved accuracy and reduced retraining effort for 15-minute-ahead wind power prediction. ImplementationSteps
- • 1: Preprocess data (handle missing samples, normalization, extract input features: prior wind speeds, directions, power output).
- • 2: Train independent LSSVM and RBFNN models on input features for 15-minute-ahead wind power prediction.
- • 3: Apply equalization to forecasting result sequences and actual measurements to obtain normalized series.
- • 4: Calculate grey relational degrees between each model’s output and actual measurements for each time window.
- • 5: Fit wind speed data for each month to the Weibull distribution; segment wind speed into regimes according to frequency analysis.
- • 6: Compute model weights (correlations) within each wind speed regime and store in a monthly weight database.
- • 7: For new forecasts, use NWP wind speed prediction to identify wind speed regime and retrieve corresponding model weights.
- • 8: Combine LSSVM and RBFNN outputs using dynamic weights for final forecast output.
- • 9: Evaluate forecasting performance using MAPE and RMSE against actual measured data. ImplementationOrder
- • 1-2
- • 2-3
- • 3-4
- • 1-5
- • 5-6
- • 6-7
- • 7-8
- • 8-9

###### Data

Historical SCADA data from a Chinese wind farm spanning 01/01/2010 to 12/31/2010 (excluding months with missing data), comprising 15-minute resolution records of wind speed (previous 15, 30, 45 min), wind direction (cosine and sine), and wind power output. The dataset includes over 30,000 samples, with wind speed segmented monthly and fitted to Weibull distributions for regime analysis.

###### EvaluationMetrics

- • MAPE: Mean Absolute Percentage Error; quantifies average absolute error as a percentage of actual wind farm rated capacity.
- • RMSE: Root Mean Square Error; quantifies the standard deviation of the prediction errors, normalized by wind farm capacity.
- • Visual Comparison: Graphical overlays of forecasted vs. actual power output for selected periods to assess tracking and volatility handling. ExpectedOutcome The hybrid model achieves a MAPE of 2.37% and RMSE of 3.79%, outperforming standalone LSSVM and RBFNN models as well as simple averaging. The method delivers improved accuracy, especially during low and fluctuating power output regimes, and reduces retraining overhead through the dynamic weight database. The approach demonstrates robustness and scalability for operational very-short term wind power forecasting.

Example of Idea Generation in Information Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • InternVL2.5: Adopted a multi-stage pipeline with language-only pre-training, MLP warmup for multimodal alignment, and instruction tuning. Demonstrated strong open-source multimodal performance but faced training complexity and limited cross-modal parameter optimization.
- • Qwen2.5-VL: Uses a staged adaptation of text-only LLMs into MLLMs, integrating visual adapters and fine-tuning. Achieves strong performance on vision-language tasks but still requires complex alignment processes and suffers in long-context or multi-image scenarios.
- • LLaVA-OneVision: Focuses on easy visual task transfer via visual instruction tuning. Excels at adaptation efficiency but underperforms on challenging multimodal reasoning or spatial tasks compared to larger unified models.
- • Gemini 2.5 Pro: A proprietary closed-source MLLM employing advanced joint training and data curation, achieving state-of-the-art results. However, it lacks the transparency and reproducibility necessary for open research progress. Challenges
- • Integrating multimodal (vision, text, video) and linguistic capabilities in a single model without compromising either modality’s performance.
- • Overcoming the inefficiencies and alignment difficulties of post-hoc adaptation pipelines that start from text-only LLMs.
- • Scaling multimodal large language models (MLLMs) to handle longer contexts, multi-image input, and complex real-world tasks.
- • Balancing pure-language proficiency with robust multimodal reasoning and visual grounding.
- • Efficiently utilizing heterogeneous and imbalanced multimodal data during pre-training and post-training.

###### Limitation

Existing MLLMs rely on multi-stage adaptation pipelines, leading to suboptimal cross-modal parameter interaction and persistent alignment or optimization bottlenecks. These approaches often freeze or partially update parameters, limiting scalability, introducing computational overhead, and creating a persistent gap in pure-language and multimodal competence.

###### Motivation

The growing complexity and diversity of real-world multimodal data demand models capable of unified, scalable, and robust multimodal reasoning, without the trade-offs and inefficiencies of post-hoc adaptation. A native joint pre-training paradigm is needed to achieve seamless linguistic and multimodal integration, better performance scalability, and open research reproducibility.

###### TaskObjective

To develop a unified, open-source multimodal large language model that jointly acquires linguistic and multimodal capabilities via native pre-training, establishes new state-of-the-art performance across a spectrum of multimodal tasks, and narrows the gap to leading proprietary MLLMs.

###### ExistingSolutions

- • InternVL2.5: Applies separate language pre-training followed by multimodal alignment (MLP warmup, visual adapters), then instruction tuning. Good on general benchmarks, but complex, inflexible, and less efficient for scaling.
- • Qwen2.5-VL: Uses visual adapters with staged fine-tuning. Strong visual-text integration, but depends on freezing strategies and additional modules. Moderate gains on long-context or diverse input.
- • LLaVA-OneVision: Visual instruction tuning for rapid adaptation. Simplicity and transferability prioritized, but lacking in deep joint optimization for reasoning and multi-modal context.
- • Gemini 2.5 Pro: Highly-curated, end-to-end joint pre-training but closed-source, with proprietary data curation and infrastructure. Reference Answer Idea InternVL3 introduces native multimodal pre-training, where vision, language, and video data are jointly leveraged in a single optimization stage. It integrates Variable Visual Position Encoding for long-context support, advanced post-training (SFT, MPO), and test-time scaling, resulting in scalable, efficient, and unified multimodal reasoning with open-source reproducibility. ImplementationSteps
- • 1: Initialize ViT, LLM, and MLP modules with pre-trained weights; set up data pipelines for multimodal and text corpora.
- • 2: Apply pixel unshuffle and prepare visual tokens for scalable image encoding.
- • 3: Implement Variable Visual Position Encoding (V2PE) for visual tokens, with random delta sampling during training.
- • 4: Jointly pre-train all model components using the multimodal autoregressive objective, sampling data at a 1:3 text-to-multimodal ratio.
- • 5: Perform Supervised Fine-Tuning (SFT) with high-quality, diverse multimodal instructions, applying loss re-weighting and data packing.
- • 6: Conduct Mixed Preference Optimization (MPO) using preference pairs and a composite loss (preference, quality, generation).
- • 7: Integrate Best-of-N test-time scaling with VisualPRM as the critic to select optimal outputs.
- • 8: Train with InternEVO for efficient large-scale distributed optimization, handling workload

- imbalances and maximizing resource utilization.
- • 9: Perform comprehensive evaluation on a battery of multimodal and language benchmarks. ImplementationOrder
- • 1-2
- • 2-3
- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 7-8
- • 8-9 Data InternVL3 is trained on a hybrid corpus: (1) Multimodal data (150B tokens) comprising imagetext pairs, video-text, GUI, tool usage, 3D scene, document, OCR, chart, multi-image, and medical data, sourced and extended from InternVL2.5 and new real-world collections; (2) Pure language data (50B tokens) built from InternLM2.5, open-source corpora, and scientific/math datasets. SFT uses 21.7M curated samples; MPO uses 300K preference pairs from MMPR v1.2. EvaluationMetrics
- • MMMU: Massive Multi-discipline Multimodal Understanding, measuring reasoning across disciplines (accuracy, %).
- • MathVista/MathVision/MathVerse: Mathematical reasoning (accuracy, %).
- • OCRBench/AI2D/ChartQA/DocVQA: Vision-text integration and document understanding (accuracy, %, EM).
- • MMBench/MMStar/MMVet/MME: Comprehensive multimodal capabilities (aggregate and per-task accuracy or score).
- • HallusionBench/MMHal/CRPE/POPE: Multimodal hallucination resistance (score, %).
- • RefCOCO/+/g: Visual grounding (localization accuracy, %).
- • MVBench/Video-MME/MLVU: Video and temporal understanding (score, %).
- • ScreenSpot/ScreenSpot-V2: GUI grounding (accuracy, %).
- • VSI-Bench: Spatial reasoning (composite score, %).
- • Language Benchmarks: MMLU, CMMLU, C-Eval, GAOKAO, TriviaQA, NaturalQuestions, RACE, HellaSwag, GSM8K, MATH, HumanEval, MBPP (accuracy, pass@k, or other standard metrics). ExpectedOutcome InternVL3-78B achieves state-of-the-art open-source results, e.g., 72.2 on MMMU, 79.0 on MathVista, 91.4 on RefCOCOg, 90.9% on GUI grounding, and 48.4 on VSI-Bench. It demonstrates robust scaling across tasks, narrows the performance gap to commercial models (Gemini 2.5 Pro, GPT-4o), and maintains strong language proficiency (80.5 overall on language benchmarks). All models and data will be open-sourced to enable community-driven research.

Example of Idea Generation in Life Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • Senior et al. (2020): Introduced deep learning for predicting inter-residue distances, improving template-free protein structure prediction but still reliant on multiple post-processing stages and lacking atomic-level accuracy for novel folds.

- • Yang et al. (2020): Employed deep neural networks to predict inter-residue orientations, integrating orientation constraints but with limited end-to-end learning and lower performance on long or complex proteins.
- • AlQuraishi (2019): Proposed an end-to-end differentiable structure prediction model, directly outputting 3D coordinates; however, it exhibited lower accuracy than multi-stage pipelines and struggled without homologous templates.
- • Marks et al. (2011); Jones et al. (2012): Used coevolutionary analysis of MSAs to infer residue contacts, achieving improvements in contact prediction but failing to achieve accurate atomic models, especially for proteins lacking deep MSAs or templates. Challenges
- • Achieving atomic-level accuracy in protein structure prediction directly from amino acid sequence, particularly in the absence of homologous structural templates.
- • Integrating physical, geometric, and evolutionary information into a single, scalable, end-toend deep learning model.
- • Handling cases with shallow or sparse multiple sequence alignments (MSAs), which limits evolutionary signal.
- • Providing robust structure prediction for large proteins and complex folds, including those with novel topologies.
- • Quantifying per-residue prediction confidence to enable reliable downstream biological applications. Limitation Contemporary approaches fall short of experimental accuracy, particularly on targets lacking homologous templates or deep MSAs. Existing neural architectures often separate contact/distance prediction from structure generation, use hand-crafted features, or rely on multi-stage heuristics, resulting in limited scalability and suboptimal integration of physical and evolutionary constraints. Poor performance persists in under-sampled sequence regions and multi-chain complexes. Motivation Structural biology is constrained by the slow pace and resource demands of experimental structure determination, leaving the vast majority of protein sequences without 3D structural annotation. Accurate, scalable, and generalizable computational prediction of protein structures—especially without close templates—would transform bioinformatics, molecular biology, and drug discovery by bridging the sequence-structure knowledge gap. TaskObjective To develop a computational method that predicts the three-dimensional atomic structure of proteins from their amino acid sequence with accuracy comparable to experimental techniques, even in the absence of close structural homologues or deep sequence alignments. ExistingSolutions
- • Physics-based simulation: Uses molecular dynamics or statistical approximations to model protein folding but is computationally intractable for large proteins and sensitive to approximations in physical modeling.
- • Bioinformatics/homology modeling: Predicts structures via alignment to known protein templates and infers constraints from evolutionary sequence analysis; limited by template availability and reduced accuracy for novel or divergent proteins.
- • Deep learning with intermediate prediction: Predicts inter-residue distances/orientations from MSAs using CNNs or attention networks, then reconstructs structures through downstream heuristics; accuracy suffers in end-to-end integration and novel folds. Reference Answer

###### Idea

AlphaFold introduces an end-to-end deep learning architecture that jointly embeds MSAs and pairwise residue features, iteratively refines 3D atomic structures through Evoformer and Invariant Point Attention modules, integrates geometric and evolutionary constraints, leverages self-distillation from unlabelled data, and produces accurate, scalable predictions with robust per-residue confidence estimates.

###### ImplementationSteps

- • 1: Collect and preprocess protein sequence and structure data from PDB, UniRef90, BFD, Uniclust30, and MGnify.
- • 2: Construct multiple sequence alignments (MSAs) and retrieve structural templates for each input sequence using HHBlits, jackhmmer, and HHSearch tools.
- • 3: Initialize the neural network: encode MSA and pairwise features; build Evoformer trunk with interleaved attention and triangle update blocks.
- • 4: Process MSA and pair features through stacked Evoformer blocks to enable information exchange and representation enhancement.
- • 5: Feed processed representations to the structural module; iteratively refine per-residue 3D coordinates using invariant point attention and equivariant transformations.
- • 6: Apply frame-aligned point error (FAPE) loss, distogram loss, BERT-style MSA masking loss, and auxiliary side-chain/violation losses for end-to-end supervised training.
- • 7: Augment training with self-distillation: generate and filter high-confidence predictions on unlabelled sequences, then retrain with mixed supervised and distillation data.
- • 8: During inference, perform ensemble predictions (if required), select best models by predicted confidence scores, and relax final structures with Amber force field.
- • 9: Evaluate predictions using CASP14 targets and recent PDB structures, reporting backbone and all-atom metrics, and provide per-residue confidence (pLDDT) and TM-score estimates. ImplementationOrder
- • 1-2
- • 2-3
- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 7-8
- • 8-9 Data AlphaFold is trained on structures from the Protein Data Bank (PDB) (as of April 2018), comprising tens of thousands of high-resolution experimental protein structures. Sequence information is augmented using UniRef90, Big Fantastic Database (BFD, ˜2.2B sequences clustered into ˜66M families), Uniclust30, and MGnify. For self-distillation, ˜350,000 diverse sequence clusters from Uniclust30 are used. Evaluation is conducted on the CASP14 dataset (87 domains) and recent non- redundant PDB chains (n=10,795), filtered to remove overlap with training data. EvaluationMetrics
- • IDDT (Local Distance Difference Test): Superposition-free metric comparing local atomic distances in predicted vs. reference structure, applicable for all atoms (IDDT) or backbone C𝛼 atoms (IDDT-C𝛼).
- • GDT (Global Distance Test): Measures fraction of residues within predefined distance thresholds; standard for CASP evaluations of domain accuracy.

- • TM-score (Template Modeling score): Assesses global structural similarity by optimal superposition over entire protein chains, robust to domain packing and length differences.
- • C𝛼 r.m.s.d.95: Root-mean-square deviation of C𝛼 atoms over the best-aligned 95% of residues, reducing the impact of outliers/artifacts.
- • pLDDT (Predicted Local Distance Difference Test): Confidence score per residue, predicting local structural accuracy.
- • pTM (Predicted TM-score): Neural network–derived prediction of TM-score for a given model.
- • Error intervals: 95% confidence intervals on reported metrics via bootstrapping. ExpectedOutcome AlphaFold achieves median backbone accuracy of 0.96 Å r.m.s.d.95 on CASP14 (95% CI: 0.85–1.16 Å), with all-atom accuracy at 1.5 Å (95% CI: 1.2–1.6 Å), outperforming the nextbest method by a margin exceeding 1.8 Å. High accuracy generalizes to new, non-redundant PDB entries (median 1.46 Å). The model provides robust per-residue confidence estimation (pLDDT, Pearson r>0.75 with true accuracy), produces accurate side-chain conformations, and scales to proteins exceeding 2,000 residues. The approach enables proteome-scale structure prediction with experimental-level precision for the majority of targets without requiring close homologues.

Example of Idea Generation in Material Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • Yaghi et al. (2008, Science): Pioneered high-throughput synthesis of zeolitic imidazolate frameworks (ZIFs) using 96-well plates, establishing the feasibility of automated, combinatorial materials discovery but with limited autonomy and narrow scope.
- • Sumida et al. (2010, Chem. Sci.): Utilized automated robotic systems and multichannel reactors for precise control over MOF synthesis, improving reproducibility but not achieving closed-loop optimization.
- • Cao et al. (2023, JACS, MOFormer): Introduced a self-supervised Transformer model for MOF property prediction, exhibiting improved accuracy and data efficiency, yet mainly focused on text-based molecular representations.
- • Kang et al. (2023, Nat. Mach. Intell., MOFTransformer): Developed a multimodal Transformer for universal transfer learning in MOFs, integrating graph and grid embeddings, achieving high transferability but requiring extensive pretraining data.
- • Park et al. (2024, Digital Discovery): Applied deep reinforcement learning with Transformers for inverse design of MOFs, enabling property-driven generative design but currently constrained by the diversity and validity of generated structures.
- • Dagdelen et al. (2024, Nat. Commun.): Proposed LLM-NERRE for structured chemical information extraction, advancing literature mining but dependent on fine-tuning and sample efficiency. Challenges
- • The vast chemical and structural diversity of MOFs renders exhaustive experimental exploration infeasible, creating a high-dimensional, combinatorial synthesis landscape.
- • Traditional manual or even semi-automated high-throughput methodologies are bottlenecked by limited autonomy, data integration, and lack of feedback-driven optimization.
- • Existing AI models, though powerful, struggle with generalizability and interpretability due

- to sparse, noisy, or unstandardized data and the complexity of structure-property relationships.
- • Realizing fully autonomous, closed-loop self-driving laboratories (SDLs) for MOF discovery is impeded by hardware standardization issues, sample handling difficulties, and insufficient integration of intelligent decision-making. Limitation Previous methodologies in MOF research either focused on isolated automation of experimental steps or applied AI for isolated tasks (e.g., property prediction) without achieving seamless, closed-loop integration. These approaches often lack robust feedback mechanisms, dynamic adaptation to new data, and struggle to generalize across diverse MOF chemistries, limiting their utility for autonomous discovery. Motivation MOFs’ application potential in energy, environment, and drug delivery is hampered by slow, labor- intensive discovery cycles and under-explored materials space. The combination of laboratory automation with advanced AI—including Transformers and LLMs—offers the prospect of systematic, iterative, and autonomous exploration, thereby addressing efficiency, reproducibility, and innovation barriers in MOF science. TaskObjective To comprehensively review and critically evaluate the convergence of artificial intelligence (especially Transformer and LLM models) and laboratory automation technologies in accelerating the discovery, synthesis, characterization, and optimization of metal-organic frameworks, with emphasis on the progression toward self-driving laboratories. ExistingSolutions
- • Traditional HTE: Employs combinatorial synthesis and characterization platforms, increasing throughput but requiring significant manual oversight and lacking intelligent optimization.
- • Machine Learning (2012–present): Applies classical statistical learning (e.g., decision trees, SVMs) for property prediction and data analysis, limited by feature engineering and scalability.
- • Deep Learning (2020–present): Utilizes neural networks for property prediction and structure optimization, improving accuracy but often acts as a black box and needs large labeled datasets.
- • Transformers/LLMs (2023–present): Leverage self-attention for sequence and structural modeling, enabling multimodal integration and text-based knowledge mining, but require extensive training and face challenges in domain adaptation and resource consumption.
- • Generative Models (VAEs, GANs, Diffusion): Enable de novo MOF structure generation, but often struggle with chemical validity, diversity, and property conditioning. Reference Answer Idea This review elucidates the synergistic integration of laboratory automation and state-of-the-art AI—particularly Transformers and LLMs—into a closed-loop, self-driving laboratory paradigm for MOF discovery. It details how AI-driven feedback, high-throughput platforms, and knowledge extraction from literature converge to enable autonomous, data-driven synthesis, characterization, and inverse design of MOFs. ImplementationSteps
- • 1: Establish automated laboratory infrastructure encompassing robotic synthesis, sample handling, and high-throughput screening modules.
- • 2: Deploy high-throughput experimental platforms for parallelized synthesis, characterization (PXRD, NMR, TEM), and evaluation (adsorption, catalysis).
- • 3: Integrate laboratory information management systems (LIMS) for structured data curation and workflow management.
- • 4: Apply machine learning/deep learning models for property prediction and experimental

- guidance using accumulated data.
- • 5: Adopt Transformer-based models and LLMs for structure-property prediction, literature mining, synthesis condition extraction, and generative MOF design.
- • 6: Implement feedback-driven experimental planning via Bayesian optimization, reinforcement learning, or LLM-driven task planners.
- • 7: Iteratively refine models and protocols in a closed-loop SDL, autonomously updating synthesis/design strategies based on real-time outcomes. ImplementationOrder
- • 1-2
- • 2-3
- • 3-4
- • 4-5
- • 5-6
- • 6-7 Data MOF structural and property databases such as MOFX-DB, ARC-MOF, hMOF, QMOF, and inhouse/generated HTE data; text corpora from scientific literature and patents used for LLM fine-tuning and information extraction; multi-million entry simulation datasets for pretraining (e.g., 1M+ hypothetical MOFs in MOFTransformer, 1.9M in PMTransformer); experimental records from robotic synthesis/characterization platforms. EvaluationMetrics
- • Experimental Throughput: Number of unique MOF samples synthesized, characterized, and evaluated per unit time.
- • Prediction Accuracy: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), coefficient of determination (R2) for property prediction models (e.g., adsorption, bandgap, stability).
- • Generalizability: Performance on out-of-distribution or unseen MOF structures/datasets, transferability to new tasks or materials classes.
- • Structural Validity/Diversity: Percentage of generated MOF candidates that are synthetically accessible and chemically valid, structural diversity indices.
- • Automation Level: SDL autonomy score (Levels 1–5), extent of human intervention required.
- • Information Extraction F1 Score: Precision, recall, and F1 for chemical entity and relation extraction from literature.
- • Resource Efficiency: Computational and experimental resources expended per successful discovery or optimization cycle. ExpectedOutcome Integration of AI and laboratory automation is expected to yield >90% accuracy in property prediction (e.g., MOFTransformer’s MTP/MOC accuracy >0.97/0.98), 2–10x acceleration in MOF discovery throughput, and significant reductions in labor and experimental time. Closedloop SDLs will enable autonomous optimization, reproducible high-quality synthesis, and rapid extraction of actionable knowledge from literature, collectively setting new benchmarks for efficiency, reproducibility, and innovation in MOF research.

Example of Idea Generation in Math Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal.

###### RelatedWork

- • Dijkstra1959: Classic label-setting SSSP algorithm using priority queues; achieves O(n·log n + m) time with Fibonacci heaps but is inherently sequential and difficult to parallelize efficiently.
- • Thorup1999: RAM-based linear-time SSSP for undirected graphs using component trees and atomic heaps; limited to large n and specific hardware, and not easily generalized or parallelized for directed graphs.
- • BellmanFord1958: Label-correcting algorithm with O(n·m) time; allows negative weights but is suboptimal in the worst case and shows little potential for efficient parallelization.
- • Han1997 / PaigeKruskal1985: Matrix multiplication-based SSSP achieves polylogarithmic parallel time at superlinear work complexity (O(nˆ3 log n)); impractical for sparse graphs due to excessive work.
- • KleinSubramanian1997: Randomized parallel BFS-based SSSP for unweighted/weighted graphs; achieves sublinear time for certain approximations, but exact solutions still demand high work or multiple passes.
- • Crauser1998: Parallelizes Dijkstra by organizing computation into phases for random graphs; achieves O(nˆ{1/3} log n) time and O(n log n + m) work on average for specific random graph classes. Challenges
- • No known work-efficient parallel SSSP algorithm achieves sublinear time for arbitrary directed graphs with nonnegative edge weights.
- • Existing parallel methods either settle nodes sequentially or incur superlinear work, limiting practical scalability on large graphs.
- • Traditional bucket-based or priority queue approaches struggle to balance parallelism and efficiency, especially with varied edge weights and node degrees.
- • Load balancing and minimizing redundant relaxations/reinsertions are unsolved for arbitrary, especially high-degree, graphs in parallel settings. Limitation Current approaches to parallel SSSP either replicate sequential order—limiting parallel speedup—or achieve fast parallel time only at the cost of excessive (superlinear) work, particularly on general graphs. Previous bucket-based label-correcting algorithms lack robust average-case guarantees for noninteger or random edge weights, and most practical parallel systems cannot efficiently exploit fine-grained sequential priority queues. Motivation The practical need for scalable, efficient shortest path computation on large graphs with arbitrary structure and edge weights drives the search for algorithms that are both parallelizable and work- efficient. Empirical evidence suggests label-correcting algorithms can outperform labelsetting ones, but theoretical justification and robust parallelization remain lacking. Bridging this gap is crucial for leveraging modern parallel and distributed architectures in large-scale graph analytics. TaskObjective Develop and analyze a parallelizable single-source shortest path (SSSP) algorithm for arbitrary directed graphs with nonnegative edge weights that achieves linear or near-linear work and sublinear parallel time for broad graph classes, while providing provable average-case guarantees. ExistingSolutions
- • Dijkstra1959: Sequential label-setting using priority queues; optimal for many sequential settings but fundamentally sequential and hard to parallelize without loss of work efficiency.
- • ApproximateBucket: Bucket-based variants for small integer weights; can be fast for restricted

- graphs but either devolve to label-correcting (with reinsertion overhead) or require auxiliary selection structures, limiting parallelism.
- • BellmanFord: Label-correcting, admits parallel edge relaxations, but incurs high redundancy and pseudo-polynomial time in the worst case.
- • MatrixMult: Reduces SSSP to matrix multiplications; achieves sublinear parallel time at cubic or worse work, impractical except for dense graphs.
- • ParallelBFS/Randomized: Suitable for unweighted or random graphs; offers fast approximate solutions but breaks down for exact computations or general edge weights. Reference Answer Idea The Δ-stepping algorithm organizes nodes into distance buckets of width Δ, differentiating light (≤ Δ) and heavy (>Δ) edges to balance parallelism and efficiency. In each phase, all nodes in the minimum nonempty bucket are processed in parallel: light edges are relaxed immediately, while heavy edges are deferred. By tuning Δ, the method provably achieves linear average-case work and scalable parallelism for a wide graph class, and can be extended to distributed memory settings and arbitrary edge weights. ImplementationSteps
- • 1: Preprocess graph: partition adjacency lists into light (≤ Δ) and heavy (>Δ) edges; for shortcut-augmented versions, compute and add shortcut edges for all simple Δ-paths.
- • 2: Initialize: set all tentative distances to ∞ except source (0), place source in the appropriate bucket.
- • 3: Phase main loop: while buckets are nonempty, select the minimum nonempty bucket (current phase), remove all nodes from it.
- • 4: Light edge relaxation: in parallel, relax all outgoing light edges of nodes in the current bucket; update tentatives and reinsert nodes as needed into corresponding buckets.
- • 5: Repeat light-edge relaxations (within bucket) until no new nodes enter the current bucket.
- • 6: Heavy edge relaxation: after the current bucket remains empty, in parallel relax all heavy edges from nodes just processed.
- • 7: Advance to the next nonempty bucket and repeat.
- • 8: Parallelization: distribute nodes (and their bucket membership) across processors; generate and assign relaxation requests using randomized dart-throwing or explicit load balancing (semisorting); aggregate and execute requests.
- • 9: Distributed memory extension: replace global memory with message-passing; assign nodes and requests using hashing and tree-based collective operations.
- • 10: Parameter tuning: select Δ empirically or via doubling search to balance work and parallel time; for arbitrary weights, use adaptive bucket splitting. ImplementationOrder
- • 1-2
- • 2-3
- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 7-3
- • 3-8
- • 8-9
- • 1-10 Data

The paper analyzes both synthetic random graphs (e.g., D(n, d𝑑/n): n-node digraph, each edge present independently with probability 𝑑/n, edge weights i.i.d. uniform [0,1]) and realworld-like datasets (e.g., random geometric graphs, roadmaps). Experiments are conducted on random d-regular graphs (n=10ˆ3 to 10ˆ6, up to 3·10ˆ6 edges) and large-scale road networks (up to n=157,457).

###### EvaluationMetrics

- • Work Complexity: Total number of operations performed across all processors, compared to sequential optimal O(n + m).
- • Parallel Time: Number of parallel phases until all nodes are settled; measured in terms such as O(d·L·log n + log2n) on PRAM.
- • Speedup: Empirical wall-clock speedup relative to sequential Dijkstra or Δ-stepping on real and synthetic graphs.
- • Phases/Reinsertions: Number of bucket phases and total reinsertions, correlated to Δ and graph/weight parameters.
- • Scalability: Ability to maintain work efficiency and speedup as the number of processors and graph size increase.
- • Robustness: Performance across random graphs, geometric graphs, and real-world networks with varying degree and weight distributions. ExpectedOutcome Δ-stepping achieves O(n + m + d·L) average-case work and O(d·L·log n + log2n) parallel time for graphs with random edge weights and bounded degree; for random graphs, O(log2n) time and O(n + m) work. Experiments show linear or near-linear speedups (e.g., >9× on 16 processors), with phases and reinsertions scaling sublinearly in n. The approach generalizes to distributed memory and arbitrary edge weights, providing, for the first time, a practical and work-efficient parallel SSSP algorithm applicable to large, arbitrary graphs.

Example of Idea Generation in Neuroscience Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork

- • ConvNet: A pioneering end-to-end CNN architecture employing temporal and spatial convolutional layers for EEG decoding, offering improved performance over traditional approaches but limited to local feature extraction due to restricted receptive field.
- • EEGNet: A compact CNN model using temporal and depthwise spatial convolutions, exhibiting robust generalization across BCI paradigms; however, it also fails to capture long-term dependencies inherent in EEG time series.
- • Transformer-Based EEG Models: Attention-based Transformers leverage global temporal dependencies for EEG decoding, achieving notable performance but neglecting local feature learning, necessitating additional pre-processing or feature extraction steps.
- • FBCSP: A classical approach utilizing filter bank common spatial patterns to extract taskspecific hand-crafted features for motor imagery classification, demonstrating strong performance but lacking generalization and requiring prior knowledge.
- • Hybrid and Graph-based Methods: Combining CNNs with hand-crafted features or graph structures to enhance spatial-temporal modeling. These methods improve local-global representations but often involve complex architectures or task-dependent preprocessing. Challenges

- • Accurately decoding EEG signals requires capturing both local features (temporal and spatial) and global dependencies due to the non-stationary and low signal-to-noise nature of EEG data.
- • CNN-based models are constrained by local receptive fields, failing to capture long-range temporal dependencies crucial for sequential EEG data.
- • Transformer-based models, though adept at modeling global dependencies, often disregard local feature representation, undermining the exploitation of fine-grained EEG information.
- • End-to-end frameworks for EEG decoding still lack sufficient interpretability regarding their decision process, particularly in identifying task-relevant neural substrates. Limitation Existing EEG decoding approaches either focus on local pattern extraction (CNNs) or global temporal correlation (Transformers) but rarely integrate both in a unified, efficient, and endto-end architecture. Furthermore, most methods require task-specific feature engineering or lack direct interpretability of neural activation, and high model parameterization raises computational concerns. Motivation The crucial observation motivating this study is the complementary value of both local and global features in EEG decoding tasks. As practical BCI applications demand robust, generalizable, and interpretable models that can efficiently learn from raw EEG data without extensive prior knowledge or task-specific feature engineering, there is a clear need for an integrated approach that unifies convolutional and self-attention mechanisms. TaskObjective To design and validate a compact, end-to-end neural architecture that jointly encapsulates local temporal-spatial and global temporal dependencies for raw EEG classification, while offering enhanced interpretability through visualization of learned representations. ExistingSolutions
- • ConvNet: Applies sequential temporal and spatial convolutions to extract discriminative local features, yielding solid performance but limited by short-range context.
- • EEGNet: Implements depthwise and separable convolutions for temporal and spatial filtering, achieving good generalization yet lacking mechanisms for modeling global dependencies.
- • RNN/LSTM-based Models: Utilize sequential recurrence to encode long-term temporal dependencies but suffer from inefficient training and rapid decay of influence across time steps.
- • Transformer-Based Models: Employ self-attention to directly capture long-range dependencies, improving performance for sequential tasks, but require additional modules or preprocessing to encode local information.
- • Hybrid Methods: Fuse hand-crafted features or graph-based encodings with deep learners, improving local-global feature integration but increasing architectural complexity and dependence on domain expertise. Reference Answer Idea The authors introduce EEG Conformer, a lightweight neural framework that sequentially combines temporal and spatial convolutions for local feature extraction with multi-head selfattention for learning global temporal dependencies. This unified architecture enables end-toend decoding from raw EEG, and a novel visualization approach (Class Activation Topography) enhances interpretability by mapping activation to brain regions. ImplementationSteps
- • 1: Band-pass filter and Z-score standardize raw EEG trials.
- • 2: Segment and augment data using time-domain segmentation and reconstruction (S&R).
- • 3: Feed data into the convolution module: perform temporal convolution (1×25 kernel),

- spatial convolution (ch×1 kernel), batch normalization, ELU activation, and average pooling (1×75 kernel, stride 15) to extract local features.
- • 4: Rearrange pooled feature maps: collapse spatial dimension, treat each timepoint’s features as a token.
- • 5: Process tokens with the self-attention module: apply N layers of multi-head self-attention (h heads), followed by feed-forward sublayers.
- • 6: Pass aggregated features to the fully-connected classifier: two layers with Softmax output.
- • 7: Train the model with cross-entropy loss using Adam optimizer and perform subject-wise validation.
- • 8: Visualize feature distributions (t-SNE) and model attention via CAM and CAT for interpretability. ImplementationOrder
- • 1-2
- • 2-3
- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 7-8 Data Three public EEG datasets were used: (1) BCI Competition IV 2a (9 subjects, 22 electrodes, 4 motor imagery classes, 250 Hz, 288 trials per session), (2) BCI Competition IV 2b (9 subjects, 3 bipolar electrodes, 2 motor imagery classes, 250 Hz, 5 sessions of 120 trials each), and (3) SEED (15 subjects, 62 electrodes, 3 emotion classes, 1000 Hz downsampled to 200 Hz, ˜3394 trials/session). Each dataset covers distinct paradigms and acquisition settings, supporting model generalization. EvaluationMetrics
- • Classification Accuracy: Percentage of correctly predicted EEG trials across classes, reflecting decoding performance.
- • Cohen’s Kappa: A statistical measure of inter-rater agreement accounting for chance, used to evaluate classification reliability.
- • Wilcoxon Signed-Rank Test: Non-parametric test for statistical significance of performance differences between models or ablation settings.
- • Training Efficiency: Measured as convergence speed (epochs to stable loss/accuracy) and per-epoch training time.
- • Interpretability: Qualitatively assessed via t-SNE clustering of learned features, CAM heatmaps, and CAT spatial-temporal mappings. ExpectedOutcome EEG Conformer achieves state-of-the-art classification accuracy and kappa across all three datasets: on BCI IV 2a, average accuracy 78.66% (↑10.91% over FBCSP), kappa 0.7155; on BCI IV 2b, 84.63% accuracy, kappa 0.6926; on SEED, 95.30% accuracy, kappa 0.9295. Ablation studies show a 6.02% average accuracy drop without the self-attention module. Visualization confirms the model’s focus on paradigm-relevant brain regions, and the architecture demonstrates efficient convergence and robustness to parameter variations, establishing a strong new backbone for general EEG decoding.

- Example of Idea Generation in Physics Question You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal. RelatedWork
- • eSEN-30M-OMat: An equivariant graph neural network tailored for materials, achieving strong accuracy via large-scale message passing, but limited to domain-specific datasets and lacking generalization across molecules or surfaces.
- • GemNet-OC20: A graph neural network for catalysis using geometric embeddings, excelling in adsorption energy prediction but focused solely on catalysis, without material or molecular generalization.
- • MACE: A foundation model for atomistic materials chemistry that demonstrates excellent transferability within the organic molecule domain, but struggles to generalize simultaneously to diverse materials and catalytic systems.
- • EquiformerV2 : An advanced equivariant transformer model that achieves strong performance on domain-specific materials and catalysis benchmarks but is not trained for multi-domain or multi-DFT-task generalization.
- • ORB v3: A scalable neural network potential capable of efficient simulation at scale, but designed primarily for periodic materials, with limited multi-domain applicability.
- • Universal Graph Deep Learning Potentials: Aim to provide comprehensive coverage across the periodic table, yet tend not to generalize to molecules or catalysis due to distribution shifts and differing DFT settings.
- • Pre-training with Fine-tuning: Large models are pre-trained on broad datasets and fine-tuned for specific tasks, yielding high accuracy but still requiring domain adaptation; true zero-shot generalization across tasks remains unproven. Challenges
- • Developing a single MLIP capable of high-fidelity, zero-shot generalization across vastly different chemical domains, including materials, molecules, catalysis, molecular crystals, and MOFs.
- • Scaling model and dataset size without sacrificing inference speed or memory efficiency, especially for long-running atomistic simulations involving thousands to hundreds of thousands of atoms.
- • Reconciling and learning from datasets with heterogeneous DFT settings, label distributions, elemental coverage, and system sizes.
- • Maintaining energy conservation, physical symmetry (rotational equivariance), and smoothness of the potential energy surface during multi-task, multi-domain learning.
- • Efficiently training and deploying ultra-large models (up to billions of parameters) under memory and compute constraints. Limitation Most existing MLIPs are either specialized for a single chemical domain or require fine-tuning to achieve high accuracy in new domains. They do not robustly generalize across materials, molecules, and catalytic systems with varying DFT settings. Further, attempts to scale model capacity often degrade inference efficiency, and models are typically trained on smaller, less diverse datasets, limiting their practical universality. Motivation The demand for rapid, accurate, and general-purpose atomistic simulations is increasing in fields such as drug discovery, energy storage, and catalysis. However, DFT is computationally prohibitive, and existing ML surrogates lack universality. The confluence of new, massive

multi-domain datasets and insights from scaling laws in deep learning presents the opportunity to create a single, highly scalable MLIP that achieves state-of-the-art accuracy, speed, and generalization across all relevant chemical domains.

###### TaskObjective

To design, train, and evaluate a family of universal machine learning interatomic potentials (UMA) that achieve high accuracy, computational efficiency, and generalization across diverse chemical and materials domains, using the largest multi-domain atomic datasets to date.

###### ExistingSolutions

- • eSEN: Utilizes equivariant message passing with spherical harmonics for high accuracy in materials, but lacks multi-domain scalability.
- • GemNet: Employs geometric embeddings for catalysis; effective on domain-specific adsorption tasks but does not generalize to other domains.
- • MACE: Foundation model for molecules, demonstrates good transferability within molecular datasets; struggles with cross-domain and multi-task generalization.
- • EquiformerV2: Equivariant transformer with improved scaling for materials and catalysis, but not designed for simultaneous multi-domain learning.
- • ORB v3: Focuses on scalable neural network potentials for materials, achieving high throughput but lacks coverage of molecular and catalytic tasks.
- • Fine-tuned Foundation Models: Pre-train on large datasets, then fine-tune for each target domain; yields high performance but necessitates domain-specific adaptation and fails to provide universal zero-shot performance. Reference Answer Idea UMA introduces a family of universal MLIPs trained on nearly 500M multi-domain atomic structures, leveraging an efficient Mixture of Linear Experts (MoLE) architecture for scalable capacity without inference overhead. Empirical scaling laws inform model/data sizing, while unified embeddings and referencing schemes enable seamless multi-DFT-task learning, delivering state-of-the-art accuracy and speed across chemistry and materials science domains. ImplementationSteps
- • 1: Data aggregation and preprocessing: curate and normalize OMat24, OMol25, OC20++, OMC25, and ODAC25, applying energy referencing and label normalization.
- • 2: Model design: configure eSEN-based GNN with integrated MoLE layers; implement global embeddings for charge, spin, and DFT task.
- • 3: MoLE routing: compute expert coefficients from global system features and pre-merge expert weights for efficient inference.
- • 4: Stage 1 training: pre-train the model in BF16 on direct force prediction with max-atom batching and reduced neighbors.
- • 5: Stage 2 fine-tuning: switch to FP32 precision and auto-grad conservative heads, increasing neighbor count for energy/force conservation.
- • 6: Memory/computation optimization: employ graph parallelism, FSDP, and activation checkpointing for large-scale training.
- • 7: Model selection: use empirical scaling laws to determine optimal model and dataset size for given compute budget.
- • 8: Evaluation: benchmark UMA models on held-out splits and established tasks across materials, catalysis, molecules, molecular crystals, and MOFs. ImplementationOrder
- • 1-2
- • 2-3

- • 3-4
- • 4-5
- • 5-6
- • 6-7
- • 7-8 Data UMA is trained on five large-scale datasets: OMat24 (bulk materials, 100M entries, 89 elements, VASP-PBE), OMol25 (molecules, 75M entries, 83 elements, ORCA-𝜔B97M-V), OC20++ (catalysis, 229M, 56 elements, VASP-RPBE), OMC25 (molecular crystals, 25M, 12 elements, VASP-PBE+D3), and ODAC25 (MOFs, 29M, 70 elements, VASP-PBE+D3). Combined, the data covers ˜459M structures and >30B atoms with near-complete elemental coverage and diverse DFT settings. EvaluationMetrics
- • Mean Absolute Error (MAE): Measures average absolute deviation between predicted and reference energies, forces (in meV/Å), and stresses (meV/Åˆ3).
- • Adsorption Energy Success Rate: Percentage of cases where the predicted global minimum adsorption energy is within 0.1 eV of the DFT minimum (AdsorbML benchmark).
- • F1 Score: Assesses binary/classification performance on Matbench Discovery for stability predictions.
- • Energy Conservation: Degree to which predicted forces/energies conserve energy over molecular dynamics trajectories (NVE MD benchmarks).
- • Simulation Throughput: Number of inference steps per second for fixed system sizes (1k, 10k, 100k atoms) on a single GPU.
- • Out-of-Domain Generalization: Performance on OOD splits, such as high-entropy alloys and novel molecular/crystal structures.
- • Phonon and Elastic Property Accuracy: MAE for phonon frequencies, free energies, elastic moduli, and related properties pertinent to material science benchmarks. ExpectedOutcome UMA achieves state-of-the-art or superior accuracy on diverse benchmarks (e.g., up to 25% improvement in AdsorbML success rate, ˜80% reduction in OC20 adsorption energy error vs. prior SOTA, chemical accuracy for ligand strain energy). The models support efficient simulation of >100k atoms with no inference penalty from increased capacity. UMA provides reliable, energy-conserving predictions across all major chemical domains, demonstrating that a single model can match or surpass specialized models in both zero-shot and fine-tuned settings.

###### A.3.3. Dry Experiment

Example of Dry Experiment in Astronomy Background

The Zwicky Transient Facility (ZTF) is an advanced optical time-domain sky survey utilizing the Palomar 48-inch Schmidt telescope equipped with a custom wide-field CCD camera. This camera covers a 47.7 square degree field of view with 16 large-format CCDs, enabling a survey speed over an order of magnitude faster than its predecessor. The system achieves a median image quality of approximately 2.0 arcseconds full-width at half-maximum (FWHM) across g, r, and i bands, with typical 5-sigma limiting magnitudes near 20.8 (g), 20.6 (r), and 19.9 (i) in 30-second exposures, improving under dark-sky conditions.

The optical design addresses the Schmidt telescopes curved focal surface through a combination of a modified Schmidt corrector, a meniscus dewar window, faceted cold plate mounting, and individual field flattener lenses above each CCD. The cameras cryostat and readout electronics are optimized for minimal beam obstruction and rapid 8.2-second readout with low noise ( 10 electrons median). A robotic observing system and scheduler maximize volumetric survey speed by selecting fields on a fixed grid with minimal dithering, enabling efficient coverage of the Northern sky and Galactic plane. ZTFs data system performs near-real-time image processing, including bias subtraction, flatfielding, astrometric and photometric calibration, and image differencing using the ZOGY algorithm to detect transient and variable sources. Alerts containing rich contextual information and machine-learning-based Real-Bogus scores are distributed via a scalable streaming system to community brokers. The system also supports solar system science by detecting both pointlike and streaked moving objects, linking detections into orbits, and reporting to the Minor Planet Center. Early scientific results demonstrate ZTFs capability to discover and classify supernovae, including young Type II events, and to conduct rapid follow-up of multi-messenger triggers such as neutrinos and gamma-ray bursts. The facility also enables studies of variable stars, exemplified by light curves of Be stars and RR Lyrae, and solar system objects, including near-Earth asteroids, asteroid rotation periods, comet activity, and Centaur outbursts. ZTFs public surveys include a three-day cadence Northern Sky Survey and a nightly Galactic Plane Survey, with observations typically taken twice per night in g and r bands. The surveys moderate depth and high cadence complement future facilities by providing early discovery and characterization of bright transients accessible to moderate-aperture telescopes. ZTF serves as a pathfinder for next-generation surveys, offering a prototype alert stream and extensive time-domain data products to the astronomical community. Data Code

- 1 #!/ usr/bin/env python3

- 2 # -*- coding: utf -8 -*-

- 3 """

- 4 Paper: The Zwicky Transient Facility: System Overview , Performance , and First Results

- 5 Authors: Eric C. Bellm , Shrinivas R. Kulkarni , Matthew J. Graham , et al.

- 6 Year: 2019

- 7

- 8 This script generates synthetic asteroid light curve data based on the descriptions

- 9 in Section 6.4.2 of the paper.

- 10

- 11 Python Version: 3.10.12

- 12 """

- 13

- 14 import sys

- 15 assert sys. version_info >= (3, 10), "This code requires Python 3.10 or higher"

- 16

- 17 # Dependencies

- 18 # pip install numpy ==1.24.3 pandas ==2.0.3

- 19

- 20 import numpy as np

- 21 import pandas as pd

- 22 from pathlib import Path

- 23 from typing import Tuple

- 24

- 25 # Global constants

- 26 DATA_DIR = Path("data")

- 27 DEFAULT_FILE_PATH = DATA_DIR / " asteroid_light_curve .csv"

- 28 RANDOM_SEED = 42 # Ensure reproducible results

- 29

- 30 def generate_asteroid_light_curve (

- 31 file_path: str ,

- 32 n_points: int = 150,

- 33 period_hr: float = 2.25 ,

- 34 amplitude: float = 0.2,

- 35 mag_range: Tuple[float , float] = (17.8 , 18.2) ,

- 36 noise_level : float = 0.03

- 37 ) -> None:

- 38 """

- 39 Generates a synthetic asteroid light curve and saves it to a CSV file.

- 40 The light curve is modeled as a simple cosine function with added Gaussian noise.

- 41 This mimics the data for an asteroid like (11014) Svatopluk in Figure 10(a).

- 42

- 43 Tag: [Data download]

- 44

- 45 Args:

- 46 file_path (str): The path to save the output CSV file.

- 47 n_points (int): The number of data points to generate.

- 48 period_hr (float): The rotation period of the asteroid in hours.

- 49 amplitude (float): The amplitude of the light curve variation in magnitudes.

- 50 mag_range (Tuple[float , float ]): The approximate magnitude range of the asteroid.

- 51 noise_level (float): The standard deviation of the Gaussian noise to add to the magnitudes.

- 52

- 53 Returns:

- 54 None

- 55

- 56 Examples:

- 57 >>> Path("data").mkdir(exist_ok=True)

- 58 >>> file_path = "data/ test_light_curve .csv"

- 59 >>> generate_asteroid_light_curve (file_path , n_points =50)

- 60 >>> import pandas as pd

- 61 >>> df = pd.read_csv(file_path)

- 62 >>> print(df.shape)

- 63 (50, 3)

- 64 """

- 65 # Set random seed to ensure reproducibility

- 66 np.random.seed (0+ RANDOM_SEED)

- 67

- 68 # Generate unevenly sampled time points to simulate real observations

- 69 # Observation time span is about 4 hours

- 70 observation_span_hr = period_hr * 1.8

- 71 times = np.sort(np.random.rand(n_points) * observation_span_hr )

- 72

- 73 # Calculate the baseline magnitude of the light curve

- 74 base_magnitude = np.mean(mag_range)

- 75

- 76 # Use cosine function to simulate asteroid brightness variation

- 77 # Multiply by 2 because a full rotation period usually contains two peaks and two troughs

- 78 magnitudes_true = base_magnitude - amplitude * np.cos(2 * np.pi * times / period_hr

* 2)

- 79

- 80 # Add Gaussian noise to the observed data

- 81 noise = np.random.normal (0, noise_level , n_points)

- 82 magnitudes_obs = magnitudes_true + noise

- 83

- 84 # Generate error for each data point , related to noise level

- 85 errors = np.random.normal(noise_level , noise_level / 4, n_points)

- 86 errors = np.maximum(errors , noise_level / 2) # Ensure errors are not too small

- 87

- 88 # Create a DataFrame to store the data

- 89 df = pd.DataFrame ({

- 90 ’time_hr ’: times ,

- 91 ’magnitude ’: magnitudes_obs ,

- 92 ’error ’: errors

- 93 })

- 94

- 95 # Save to CSV file

- 96 df.to_csv(file_path , index=False)

- 97 print(f" Successfully generated synthetic light curve data and saved to: {file_path} ")

- 98

- 99

- 100 if __name__ == "__main__":

- 101 # Ensure data directory exists

- 102 DATA_DIR.mkdir(exist_ok=True)

- 103

- 104 # Generate simulated data

- 105 generate_asteroid_light_curve (

- 106 file_path=str( DEFAULT_FILE_PATH ),

- 107 n_points =150 ,

- 108 period_hr =2.25 , # Asteroid period corresponding to Figure 10(a)

- 109 amplitude =0.15 , # Amplitude

- 110 mag_range =(17.8 , 18.1) , # Magnitude range

- 111 noise_level =0.02 # Noise level

- 112 )

###### Main Code with Incomplete Functions

- 1 #!/ usr/bin/env python3

- 2 # -*- coding: utf -8 -*-

- 3 """

- 4 Paper: The Zwicky Transient Facility: System Overview , Performance , and First Results

- 5 Authors: Eric C. Bellm , Shrinivas R. Kulkarni , Matthew J. Graham , et al.

- 6 Year: 2019

- 7

- 8 This script implements the asteroid light curve analysis from Section 6.4.2.

- 9 It determines the rotation period of an asteroid from its light curve using

- 10 a Lomb -Scargle periodogram and Fourier series fitting.

- 11

- 12 Python Version: 3.10.12

- 13 """

- 14

- 15 import sys

- 16

- 17 assert sys. version_info >= (3, 10), "This code requires Python 3.10 or higher"

- 18

- 19 # Dependencies

- 20 # pip install numpy ==1.24.3 pandas ==2.0.3 scipy ==1.10.1

- 21

- 22 import numpy as np

- 23 import pandas as pd

- 24 from scipy.signal import lombscargle

- 25 from typing import Tuple

- 26 from pathlib import Path

- 27

- 28 # Global constants

- 29 DATA_FILE_PATH = "data/ asteroid_light_curve .csv"

- 30 # Order of Fourier series , the paper mentions second -order Fourier series

- 31 FOURIER_N_TERMS = 2

- 32 # Numerical stability constant

- 33 EPSILON = 1e-9

- 34

- 35

- 36 def load_light_curve_data (file_path: str) -> pd.DataFrame:

- 37 """

- 38 Loads asteroid light curve data from a CSV file.

- 39

- 40 Tag: [Data loading]

- 41

- 42 Args:

- 43 file_path (str): The path to the CSV file.

- 44

- 45 Returns:

- 46 pd.DataFrame: A DataFrame containing the light curve data with

- 47 columns ’time_hr ’, ’magnitude ’, and ’error ’.

- 48

- 49 Examples:

- 50 >>> Path("data").mkdir(exist_ok=True)

- 51 >>> data = {’time_hr ’: [0, 1], ’magnitude ’: [18.0 , 18.1] , ’error ’: [0.01 , 0.01]}

- 52 >>> df = pd.DataFrame(data)

- 53 >>> df.to_csv("data/dummy.csv", index=False)

- 54 >>> loaded_df = load_light_curve_data ("data/dummy.csv")

- 55 >>> print(loaded_df.shape)

- 56 (2, 3)

- 57 """

- 58 try:

- 59 return pd.read_csv(file_path)

- 60 except FileNotFoundError :

- 61 print(f"Error: Data file not found at ’{file_path}’")

- 62 print("Please run ’data.py’ first to generate the data file.")

- 63 sys.exit (1)

- 64

- 65

- 66 def calculate_lomb_scargle_periodogram (

- 67 times: np.ndarray ,

- 68 magnitudes: np.ndarray ,

- 69 min_period: float = 0.5,

- 70 max_period: float = 5.0,

- 71 num_periods : int = 10000

- 72 ) -> Tuple[np.ndarray , np.ndarray ]:

- 73 """

- 74 Calculates the Lomb -Scargle periodogram for unevenly sampled data.

- 75

- 76 Tag: [Numerical calculation]

- 77

- 78 Args:

- 79 times (np.ndarray): Array of time points.

- 80 magnitudes (np.ndarray): Array of magnitude measurements .

- 81 min_period (float): The minimum period to test.

- 82 max_period (float): The maximum period to test.

- 83 num_periods (int): The number of period points to evaluate.

- 84

- 85 Returns:

- 86 Tuple[np.ndarray , np.ndarray ]: A tuple containing the periods tested

- 87 and the corresponding periodogram power.

- 88

- 89 Examples:

- 90 >>> times = np.linspace (0, 4, 50)

- 91 >>> magnitudes = 18 + 0.1 * np.sin(2 * np.pi * times / 2.0)

- 92 >>> periods , power = calculate_lomb_scargle_periodogram (times , magnitudes)

- 93 >>> print(periods.shape , power.shape)

- 94 (10000 ,) (10000 ,)

- 95 """

- 96 pass # [Please complete the code]

- 97

- 98

- 99 def find_best_period_from_periodogram (

- 100 periods: np.ndarray ,

- 101 power: np.ndarray

- 102 ) -> float:

- 103 """

- 104 Finds the period corresponding to the highest power in the periodogram.

- 105

- 106 Tag: [Numerical calculation]

- 107

- 108 Args:

- 109 periods (np.ndarray): Array of periods.

- 110 power (np.ndarray): Array of periodogram powers.

- 111

- 112 Returns:

- 113 float: The period with the highest power.

- 114

- 115 Examples:

- 116 >>> periods = np.array ([1.0 , 2.0, 3.0])

- 117 >>> power = np.array ([0.1 , 0.8, 0.2])

- 118 >>> best_period = find_best_period_from_periodogram (periods , power)

- 119 >>> print(best_period )

- 120 2.0

- 121 """

- 122 pass # [Please complete the code]

- 123

- 124

- 125 def build_fourier_design_matrix (

- 126 times: np.ndarray ,

- 127 period: float ,

- 128 n_terms: int

- 129 ) -> np.ndarray:

- 130 """

- 131 Builds the design matrix for a Fourier series linear least -squares fit.

- 132

- 133 Tag: [Predictive modeling]

- 134

- 135 Args:

- 136 times (np.ndarray): Array of time points.

- 137 period (float): The fundamental period of the Fourier series.

- 138 n_terms (int): The number of Fourier terms (harmonics) to include.

- 139

- 140 Returns:

- 141 np.ndarray: The design matrix for the fit.

- 142

- 143 Examples:

- 144 >>> times = np.array ([0, 1, 2])

- 145 >>> period = 4.0

- 146 >>> n_terms = 1

- 147 >>> matrix = build_fourier_design_matrix (times , period , n_terms)

- 148 >>> print(matrix.shape)

- 149 (3, 3)

- 150 """

- 151 # Fundamental frequency

- 152 omega = 2 * np.pi / (period + EPSILON)

- 153 # Initialize a column vector for the constant term

- 154 design_matrix = [np.ones_like(times)]

- 155 # Loop to add sin and cos terms for each order

- 156 for i in range (1, n_terms + 1):

- 157 design_matrix .append(np.sin(i * omega * times))

- 158 design_matrix .append(np.cos(i * omega * times))

- 159 return np.vstack( design_matrix ).T

- 160

- 161

- 162 def fit_fourier_model (

- 163 design_matrix : np.ndarray ,

- 164 magnitudes: np.ndarray ,

- 165 errors: np.ndarray

- 166 ) -> np.ndarray:

- 167 """

- 168 Fits a Fourier model using weighted linear least squares.

- 169

- 170 Tag: [Predictive modeling]

- 171

- 172 Args:

- 173 design_matrix (np.ndarray): The design matrix from build_fourier_design_matrix .

- 174 magnitudes (np.ndarray): Array of magnitude measurements .

- 175 errors (np.ndarray): Array of measurement errors for weighting.

- 176

- 177 Returns:

- 178 np.ndarray: The array of fitted Fourier coefficients .

- 179

- 180 Examples:

- 181 >>> times = np.linspace (0, 4, 10)

- 182 >>> period = 2.0

- 183 >>> magnitudes = np.sin(2 * np.pi * times / period)

- 184 >>> errors = np.full_like(magnitudes , 0.1)

- 185 >>> matrix = build_fourier_design_matrix (times , period , 1)

- 186 >>> coeffs = fit_fourier_model (matrix , magnitudes , errors)

- 187 >>> print(len(coeffs))

- 188 3

- 189 """

- 190 # Use errors for weighting

- 191 weights = 1.0 / (errors + EPSILON)

- 192 weighted_matrix = design_matrix * weights [:, np.newaxis]

- 193 weighted_magnitudes = magnitudes * weights

- 194

- 195 # Solve using least squares

- 196 coeffs , _, _, _ = np.linalg.lstsq(weighted_matrix , weighted_magnitudes , rcond=None)

- 197 return coeffs

- 198

- 199

- 200 def evaluate_fourier_model (

- 201 design_matrix : np.ndarray ,

- 202 coeffs: np.ndarray

- 203 ) -> np.ndarray:

- 204 """

- 205 Evaluates the Fourier model at given time points.

- 206

- 207 Tag: [Numerical calculation]

- 208

- 209 Args:

- 210 design_matrix (np.ndarray): The design matrix.

- 211 coeffs (np.ndarray): The fitted Fourier coefficients .

- 212

- 213 Returns:

- 214 np.ndarray: The predicted magnitudes from the model.

- 215

- 216 Examples:

- 217 >>> times = np.array ([0, 1, 2])

- 218 >>> period = 4.0

- 219 >>> n_terms = 1

- 220 >>> matrix = build_fourier_design_matrix (times , period , n_terms)

- 221 >>> coeffs = np.array ([18.0 , 0.1, 0.0])

- 222 >>> model_mags = evaluate_fourier_model (matrix , coeffs)

- 223 >>> print(model_mags.shape)

- 224 (3,)

- 225 """

- 226 return np.dot(design_matrix , coeffs)

- 227

- 228

- 229 def calculate_reduced_chi_squared (

- 230 observed: np.ndarray ,

- 231 expected: np.ndarray ,

- 232 errors: np.ndarray ,

- 233 num_fit_params : int

- 234 ) -> float:

- 235 """

- 236 Calculates the reduced chi -squared statistic for a fit.

- 237

- 238 Tag: [Metric calculation]

- 239

- 240 Args:

- 241 observed (np.ndarray): The observed data values.

- 242 expected (np.ndarray): The model ’s expected values.

- 243 errors (np.ndarray): The errors on the observed values.

- 244 num_fit_params (int): The number of free parameters in the model.

- 245

- 246 Returns:

- 247 float: The reduced chi -squared value.

- 248

- 249 Examples:

- 250 >>> obs = np.array ([1, 2, 3])

- 251 >>> exp = np.array ([1.1 , 2.2, 2.9])

- 252 >>> err = np.array ([0.2 , 0.2, 0.2])

- 253 >>> r_chi2 = calculate_reduced_chi_squared (obs , exp , err , 1)

- 254 >>> print(f"{r_chi2 :.2f}")

- 255 1.25

- 256 """

- 257 # Calculate chi -squared value

- 258 chi_squared = np.sum ((( observed - expected) / (errors + EPSILON)) ** 2)

- 259 # Calculate degrees of freedom

- 260 degrees_of_freedom = len(observed) - num_fit_params

- 261 # Avoid division by zero

- 262 if degrees_of_freedom <= 0:

- 263 return np.inf

- 264 return chi_squared / degrees_of_freedom

- 265

- 266

- 267 if __name__ == "__main__":

- 268 print("--- ZTF Asteroid Light Curve Analysis ---")

- 269

- 270 # 1. Load data

- 271 print(f"\n[1/4] Loading light curve data from ’{ DATA_FILE_PATH } ’...")

- 272 light_curve_df = load_light_curve_data ( DATA_FILE_PATH )

- 273 times = light_curve_df [’time_hr ’]. values

- 274 magnitudes = light_curve_df [’magnitude ’]. values

- 275 errors = light_curve_df [’error ’]. values

- 276 print(f" Successfully loaded {len(times)} data points.")

- 277

- 278 # 2. Calculate Lomb -Scargle periodogram

- 279 print("\n[2/4] Calculating periodogram using Lomb -Scargle method ...")

- 280 # Set a reasonable period search range

- 281 min_p = 0.5 # hours

- 282 max_p = (times.max() - times.min()) # observation span as max period

- 283 periods , power = calculate_lomb_scargle_periodogram (times , magnitudes , min_period= min_p , max_period=max_p)

- 284 best_period = find_best_period_from_periodogram (periods , power)

- 285 print(f"Periodogram analysis complete. Most likely light curve period is: { best_period :.4f} hours.")

- 286

- 287 # 3. Fit Fourier series model

- 288 print(f"\n[3/4] Fitting { FOURIER_N_TERMS }-order Fourier series using found period { best_period :.4f} hours ...")

- 289 design_matrix = build_fourier_design_matrix (times , best_period , FOURIER_N_TERMS )

- 290 fourier_coeffs = fit_fourier_model (design_matrix , magnitudes , errors)

- 291 model_magnitudes = evaluate_fourier_model (design_matrix , fourier_coeffs )

- 292 print("Fourier model fitting complete.")

- 293 print(f"Fitted coefficients : {np.round(fourier_coeffs , 4)}")

- 294

- 295 # 4. Evaluate goodness of fit

- 296 print("\n[4/4] Evaluating model goodness of fit ...")

- 297 num_params = 1 + 2 * FOURIER_N_TERMS # 1 constant term + n*2 sin/cos terms

- 298 r_chi2 = calculate_reduced_chi_squared (magnitudes , model_magnitudes , errors , num_params)

- 299 print(f"Reduced chi -squared of the model: {r_chi2 :.4f}")

- 300 if 0.5 < r_chi2 < 2.0:

- 301 print("This is a reasonable fit.")

- 302 else:

- 303 print("Warning: The fit may be poor or error estimates inaccurate.")

- 304

- 305 # Final output

- 306 # The core goal in the paper for finding asteroid rotation period is to obtain the period value

- 307 print("\n--- Analysis Complete ---")

- 308 print("[Final Output]")

- 309 # Since the asteroid rotation causes two peaks in the light curve per rotation ,

- 310 # the physical rotation period is twice the light curve period found by Lomb Scargle

- 311 # This is a common convention in astronomy

- 312 rotation_period = best_period * 2.0

- 313 print(f"{ rotation_period :.4f}") Answer

|def calculate_lomb_scargle_periodogram ( times: np.ndarray , magnitudes: np.ndarray , min_period: float = 0.5, max_period: float = 5.0, num_periods : int = 10000<br><br>) -> Tuple[np.ndarray , np.ndarray ]: """ Calculates the Lomb -Scargle periodogram for unevenly sampled data.<br><br>Tag: [Numerical calculation] Args:<br><br>times (np.ndarray): Array of time points. magnitudes (np.ndarray): Array of magnitude measurements . min_period (float): The minimum period to test. max_period (float): The maximum period to test. num_periods (int): The number of period points to evaluate.<br><br>Returns: Tuple[np.ndarray , np.ndarray ]: A tuple containing the periods tested and the corresponding periodogram power.<br><br>Examples: >>> times = np.linspace (0, 4, 50) >>> magnitudes = 18 + 0.1 * np.sin(2 * np.pi * times / 2.0) >>> periods , power = calculate_lomb_scargle_periodogram (times , magnitudes) >>> print(periods.shape , power.shape) (10000 ,) (10000 ,)<br><br>"""<br><br># 1. Define and generate the search range for ordinary frequencies (f = 1/P) safe_min_period = max(min_period , EPSILON) min_freq = 1.0 / max_period max_freq = 1.0 / safe_min_period ordinary_frequencies = np.linspace(min_freq , max_freq , num_periods)<br><br># 2. [Key modification ] Convert ordinary frequencies to angular frequencies (omega<br><br>= 2*pi*f) # scipy.signal.lombscargle requires angular frequencies as input angular_frequencies = 2 * np.pi * ordinary_frequencies<br><br># 3. Center magnitude data to remove DC component magnitudes_centered = magnitudes - np.mean(magnitudes)<br><br># 4. Calculate periodogram power using angular frequencies power = lombscargle (times , magnitudes_centered , angular_frequencies , normalize=True<br><br>)<br><br># 5. Return periods (P = 1/f) corresponding to ordinary frequencies and power periods = 1.0 / ( ordinary_frequencies + EPSILON) return periods , power<br><br><br>def find_best_period_from_periodogram ( periods: np.ndarray , power: np.ndarray<br><br>) -> float: """ Finds the period corresponding to the highest power in the periodogram.<br><br>Tag: [Numerical calculation] Args:<br><br>periods (np.ndarray): Array of periods. power (np.ndarray): Array of periodogram powers.<br><br>Returns: float: The period with the highest power.<br><br>Examples: >>> periods = np.array ([1.0 , 2.0, 3.0]) >>> power = np.array ([0.1 , 0.8, 0.2]) >>> best_period = find_best_period_from_periodogram (periods , power) >>> print(best_period) 2.0<br><br>""" best_period_index = np.argmax(power) return periods[ best_period_index ]<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47
- 48
- 49
- 50
- 51
- 52
- 53
- 54
- 55
- 56
- 57
- 58
- 59
- 60
- 61
- 62
- 63
- 64
- 65
- 66
- 67
- 68
- 69
- 70
- 71
- 72
- 73
- 74
- 75

Example of Dry Experiment in Earth Background Surface ozone is a secondary air pollutant formed by photochemical reactions involving carbon monoxide (CO), volatile organic compounds (VOCs), nitrogen oxides (NOx = NO + NO2), and sunlight. It poses significant risks to human health, including respiratory and cardiovascular effects, and damages vegetation by reducing crop yields and ecosystem productivity. While stringent emission controls since the 1990s have reduced ozone pollution in many Western regions, rapid industrialization and urbanization in East Asia, particularly China, have led to increasing ozone precursor emissions and elevated surface ozone levels. Recent nationwide monitoring in China, initiated around 2013, reveals that although median ozone concentrations during the warm season (AprilSeptember) are comparable to those in industrialized regions such as Japan, South Korea, Europe, and the United States, the frequency and magnitude of high-ozone events are substantially greater in China. Key metrics include the fourth highest daily maximum 8-hour average ozone (4MDA8), the number of days exceeding 70 ppb (NDGT70), and cumulative exposure indices like SOMO35 (sum of ozone means over 35 ppb). Chinas warm-season 4MDA8 averages around 86 ppb, exceeding other regions by 630%, while NDGT70 values are 93575% higher, indicating more frequent episodes of elevated ozone. Vegetation exposure metrics such as AOT40 and W126, which correlate with ozone-induced plant damage, are also significantly elevated in China, suggesting greater risks to agricultural productivity and ecosystem health. Spatially, ozone pollution hotspots in China are concentrated in densely populated and industrialized regions including the North China Plain, Yangtze River Delta, and Pearl River Delta, with some western areas affected due to topography and local emissions. Seasonal patterns show ozone peaks in late spring and early summer, influenced by regional meteorology such as the Asian summer monsoon, which modulates photochemical activity and pollutant transport. Temporal analysis from 2013 to 2017 indicates a rising trend in ozone levels across Chinese cities, with annual increases in exposure metrics ranging from approximately 3.7% to over 15% per year. This contrasts with stable or declining ozone trends in Europe and the United States over recent decades. The increase in ozone occurs despite reductions in primary pollutants like SO2, NO2, CO, and fine particulate matter (PM2.5), reflecting complex photochemical interactions. In particular, reductions in NOx or PM2.5 can paradoxically enhance ozone formation in VOC-sensitive regimes prevalent in eastern China. Rising VOC emissions and meteorological factors such as hotter, drier summers also contribute to elevated ozone. These findings highlight China as a current global hotspot for surface ozone pollution, with greater human and vegetation exposure than other industrialized regions with extensive monitoring. The severity and increasing trend of ozone pollution pose challenges for air quality management, indicating a need for targeted control strategies focusing on VOC emissions and comprehensive understanding of chemical and meteorological influences on ozone formation. Data Code

- 1 #!/ usr/bin/env python3

- 2 # -*- coding: utf -8 -*-

- 3 """

- 4 Paper: Severe Surface Ozone Pollution in China: A Global Perspective

- 5 Authors: Xiao Lu , Jiayun Hong , Lin Zhang , et al.

- 6 Year: 2018

- 7

- 8 Data generation script for simulating hourly surface ozone data.

- 9 Python Version: 3.10.12

- 10 """

- 11

- 12 import sys

- 13

- 14 assert sys. version_info >= (3, 10), "This code requires Python 3.10 or higher"

- 15

- 16 # Dependencies

- 17 # pip install numpy ==1.24.3 pandas ==2.0.3

- 18

- 19 import numpy as np

- 20 import pandas as pd

- 21 from pathlib import Path

- 22 import os

- 23

- 24

- 25 def generate_hourly_ozone_data (

- 26 n_sites: int ,

- 27 start_date: str ,

- 28 end_date: str ,

- 29 region_params : dict

- 30 ) -> pd.DataFrame:

- 31 """

- 32 Generate synthetic hourly ozone data for multiple sites.

- 33 Tag: [Simulation]

- 34

- 35 Args:

- 36 n_sites (int): The number of monitoring sites to simulate.

- 37 start_date (str): The start date for the data series (e.g., ’2013 -01 -01 ’).

- 38 end_date (str): The end date for the data series (e.g., ’2017 -12 -31 ’).

- 39 region_params (dict): A dictionary containing parameters for the region.

- 40 Keys should include ’base_mean ’, ’seasonal_amp ’, ’daily_amp ’,

- 41 ’noise_level ’, ’event_prob ’, ’event_strength ’.

- 42

- 43 Returns:

- 44 pd.DataFrame: A DataFrame with columns [’site_id ’, ’timestamp ’, ’ozone_ppb ’].

- 45

- 46 Examples:

- 47 >>> params = {

- 48 ... ’base_mean ’: 40, ’seasonal_amp ’: 15, ’daily_amp ’: 20,

- 49 ... ’noise_level ’: 5, ’event_prob ’: 0.02 , ’event_strength ’: 40

- 50 ... }

- 51 >>> df = generate_hourly_ozone_data (2, ’2017-01-01’, ’2017-01-31’, params)

- 52 >>> print(df.shape)

- 53 (1488 , 3)

- 54 """

- 55 np.random.seed (0+42)

- 56 timestamps = pd. to_datetime(np.arange(

- 57 np. datetime64 (start_date),

- 58 np. datetime64 (end_date) + np.timedelta64 (1, ’D’),

- 59 np. timedelta64 (1, ’h’)

- 60 ))

- 61 n_hours = len( timestamps )

- 62

- 63 # Prepare time feature vectors

- 64 day_of_year = timestamps.dayofyear

- 65 hour_of_day = timestamps.hour

- 66

- 67 all_sites_data = []

- 68 for site_id in range(n_sites):

- 69 # Base signal = seasonal cycle + daily cycle

- 70 seasonal_cycle = region_params [’seasonal_amp ’] * np.sin (2 * np.pi * ( day_of_year - 90) / 365.25)

- 71 daily_cycle = region_params [’daily_amp ’] * np.sin (2 * np.pi * ( hour_of_day - 8) / 24)

- 72

- 73 base_signal = region_params [’base_mean ’] + seasonal_cycle + daily_cycle

- 74

- 75 # Add random noise

- 76 noise = np.random.randn(n_hours) * region_params [’noise_level ’]

- 77

- 78 # Simulate high pollution events

- 79 events = np.zeros(n_hours)

- 80 for i in range(n_hours):

- 81 if np.random.rand () < region_params [’event_prob ’]:

- 82 # Pollution event lasts 24 -72 hours

- 83 duration = np.random.randint (24, 73)

- 84 event_end = min(i + duration , n_hours)

- 85 event_shape = np.sin(np.linspace (0, np.pi , event_end - i))

- 86 events[i:event_end] += region_params [’event_strength ’] * event_shape

- 87

- 88 # Compose final signal

- 89 ozone_concentration = base_signal + noise + events

- 90 # Ensure concentration values are non -negative

- 91 ozone_concentration = np.maximum(ozone_concentration , 0)

- 92

- 93 site_df = pd.DataFrame ({

- 94 ’site_id ’: f’site_{site_id}’,

- 95 ’timestamp ’: timestamps ,

- 96 ’ozone_ppb ’: ozone_concentration

- 97 })

- 98 all_sites_data .append(site_df)

- 99

- 100 return pd.concat(all_sites_data , ignore_index =True)

- 101

- 102

- 103 def save_data_to_csv (df: pd.DataFrame , file_path: str):

- 104 """

- 105 Save a DataFrame to a CSV file.

- 106 Tag: [Data saving]

- 107

- 108 Args:

- 109 df (pd.DataFrame): The DataFrame to save.

- 110 file_path (str): The path to the output CSV file.

- 111

- 112 Returns:

- 113 None

- 114

- 115 Examples:

- 116 >>> data = pd.DataFrame ({’col1 ’: [1, 2], ’col2 ’: [3, 4]})

- 117 >>> save_data_to_csv (data , ’data/test.csv ’)

- 118 """

- 119 # Ensure directory exists

- 120 output_dir = os.path.dirname(file_path)

- 121 if not os.path.exists( output_dir):

- 122 os.makedirs( output_dir)

- 123 df.to_csv(file_path , index=False)

- 124 print(f"Data has been saved to: {file_path}")

- 125

- 126

- 127 if __name__ == "__main__":

- 128 # Set different parameters for China and JKEU regions according to the paper description

- 129 # China region parameters: higher base value , stronger seasonal and daily variations , more frequent and stronger high pollution events

- 130 china_params = {

- 131 ’base_mean ’: 45,

- 132 ’seasonal_amp ’: 20,

- 133 ’daily_amp ’: 25,

- 134 ’noise_level ’: 8,

- 135 ’event_prob ’: 0.015 , # Higher event occurrence probability

- 136 ’event_strength ’: 50 # Stronger event intensity

- 137 }

- 138

- 139 # JKEU region parameters: relatively moderate pollution levels

- 140 jkeu_params = {

- 141 ’base_mean ’: 35,

- 142 ’seasonal_amp ’: 15,

- 143 ’daily_amp ’: 20,

- 144 ’noise_level ’: 5,

- 145 ’event_prob ’: 0.005 , # Lower event occurrence probability

- 146 ’event_strength ’: 30 # Weaker event intensity

- 147 }

- 148

- 149 # To reduce runtime , we only simulate one year of data

- 150 START_DATE = ’2017 -01 -01 ’

- 151 END_DATE = ’2017 -12 -31 ’

- 152 NUM_SITES = 10 # Simulate 10 sites

- 153

- 154 print("Generating simulated ozone data for China region ...")

- 155 china_ozone_data = generate_hourly_ozone_data (NUM_SITES , START_DATE , END_DATE , china_params )

- 156

- 157 print("Generating simulated ozone data for JKEU region ...")

- 158 jkeu_ozone_data = generate_hourly_ozone_data (NUM_SITES , START_DATE , END_DATE , jkeu_params )

- 159

- 160 # Create data directory and save files

- 161 data_dir = Path("data")

- 162 data_dir.mkdir(exist_ok=True)

- 163

- 164 save_data_to_csv (china_ozone_data , str(data_dir / " china_ozone_data .csv"))

- 165 save_data_to_csv (jkeu_ozone_data , str(data_dir / " jkeu_ozone_data .csv")) Main Code with Incomplete Functions

- 1 #!/ usr/bin/env python3

- 2 # -*- coding: utf -8 -*-

- 3 """

- 4 Paper: Severe Surface Ozone Pollution in China: A Global Perspective

- 5 Authors: Xiao Lu , Jiayun Hong , Lin Zhang , et al.

- 6 Year: 2018

- 7

- 8 Implementation of ozone pollution metrics calculation.

- 9 Python Version: 3.10.12

- 10 """

- 11

- 12 import sys

- 13

- 14 assert sys. version_info >= (3, 10), "This code requires Python 3.10 or higher"

- 15

- 16 # Dependencies

- 17 # pip install numpy ==1.24.3 pandas ==2.0.3

- 18

- 19 import numpy as np

- 20 import pandas as pd

- 21

- 22

- 23 def load_ozone_data (file_path: str) -> pd.DataFrame:

- 24 """

- 25 Load hourly ozone data from a CSV file.

- 26 Tag: [Data loading]

- 27

- 28 Args:

- 29 file_path (str): The path to the CSV data file.

- 30

- 31 Returns:

- 32 pd.DataFrame: A DataFrame with a datetime index and ozone data.

- 33

- 34 Examples:

- 35 >>> df = load_ozone_data (’data/ china_ozone_data .csv ’)

- 36 >>> print(df.columns)

- 37 Index ([’ site_id ’, ’ozone_ppb ’], dtype=’object ’)

- 38 """

- 39 df = pd.read_csv(file_path , parse_dates =[’timestamp ’])

- 40 df = df.set_index(’timestamp ’)

- 41 return df

- 42

- 43

- 44 def calculate_mda8 ( daily_hourly_data : pd.Series) -> float:

- 45 """

- 46 Calculate the Daily Maximum 8-hour Average (MDA8) ozone concentration .

- 47 Tag: [Numerical calculation]

- 48

- 49 Args:

- 50 daily_hourly_data (pd.Series): A Series of 24 hourly ozone values for a single day.

- 51

- 52 Returns:

- 53 float: The MDA8 value in ppb. Returns np.nan if data is insufficient .

- 54

- 55 Examples:

- 56 >>> hours = pd.to_datetime(pd.date_range (’2023-07-01’, periods =24, freq=’h ’))

- 57 >>> data = pd.Series(np.sin(np.linspace (0, 2*np.pi , 24)) * 20 + 50, index=hours )

- 58 >>> mda8 = calculate_mda8 (data)

- 59 >>> print(round(mda8 , 2))

- 60 67.68

- 61 """

- 62 pass # [Please complete the code]

- 63

- 64

- 65 def calculate_4mda8 ( mda8_series: pd.Series) -> float:

- 66 """

- 67 Calculate the 4th highest MDA8 value for a given period.

- 68 Tag: [Numerical calculation]

- 69

- 70 Args:

- 71 mda8_series (pd.Series): A Series of daily MDA8 values.

- 72

- 73 Returns:

- 74 float: The 4th highest MDA8 value. Returns np.nan if data is insufficient .

- 75

- 76 Examples:

- 77 >>> data = pd.Series ([80 , 90, 70, 100, 110, 60])

- 78 >>> val = calculate_4mda8 (data)

- 79 >>> print(val)

- 80 80.0

- 81 """

- 82 valid_mda8 = mda8_series.dropna ()

- 83 if len( valid_mda8 ) < 4:

- 84 return np.nan

- 85 return valid_mda8 . sort_values (ascending=False).iloc [3]

- 86

- 87

- 88 def calculate_ndgt70 ( mda8_series: pd.Series) -> int:

- 89 """

- 90 Calculate the total number of days with MDA8 values > 70 ppb.

- 91 Tag: [Numerical calculation]

- 92

- 93 Args:

- 94 mda8_series (pd.Series): A Series of daily MDA8 values.

- 95

- 96 Returns:

- 97 int: The count of days where MDA8 > 70 ppb.

- 98

- 99 Examples:

- 100 >>> data = pd.Series ([65 , 71, 85, 70, 70.1])

- 101 >>> count = calculate_ndgt70 (data)

- 102 >>> print(count)

- 103 3

- 104 """

- 105 return ( mda8_series > 70).sum()

- 106

- 107

- 108 def calculate_aot40 ( hourly_data: pd.Series) -> float:

- 109 """

- 110 Calculate the AOT40 (Accumulated Ozone over a Threshold of 40 ppb).

- 111 Tag: [Numerical calculation]

- 112

- 113 Args:

- 114 hourly_data (pd.Series): A Series of hourly ozone data for the entire period.

- 115 The function will filter for daytime hours (08:00 -19:59).

- 116

- 117 Returns:

- 118 float: The total AOT40 value in ppb -hours.

- 119

- 120 Examples:

- 121 >>> hours = pd.to_datetime(pd.date_range (’2023-07-01’, periods =24, freq=’h ’))

- 122 >>> data = pd.Series(np.arange (30, 54), index=hours)

- 123 >>> aot40 = calculate_aot40 (data)

- 124 >>> print(aot40)

- 125 91.0

- 126 """

- 127 daytime_data = hourly_data [( hourly_data.index.hour >= 8) & (hourly_data.index.hour <= 19)]

- 128 # Calculate the portion exceeding 40 ppb each hour

- 129 exceedances = np.maximum (0, daytime_data - 40)

- 130 return exceedances .sum()

- 131

- 132

- 133 def calculate_w126 ( hourly_data: pd.Series) -> float:

- 134 """

- 135 Calculate the W126 metric , a weighted cumulative exposure index.

- 136 Tag: [Numerical calculation]

- 137

- 138 Args:

- 139 hourly_data (pd.Series): A Series of hourly ozone data for the entire period.

- 140 The function will filter for daytime hours (08:00 -19:59).

- 141

- 142 Returns:

- 143 float: The total W126 value in ppb -hours.

- 144

- 145 Examples:

- 146 >>> hours = pd.to_datetime(pd.date_range (’2023-07-01’, periods =24, freq=’h ’))

- 147 >>> data = pd.Series(np.full (24, 80), index=hours) # Constant 80 ppb

- 148 >>> w126 = calculate_w126 (data)

- 149 >>> print(round(w126 , 2))

- 150 954.16

- 151 """

- 152 pass # [Please complete the code]

- 153

- 154

- 155 def analyze_regional_metrics (df: pd.DataFrame) -> dict:

- 156 """

- 157 Analyze and compute all key ozone metrics for a given region ’s data.

- 158 Tag: [ Statistical analysis]

- 159

- 160 Args:

- 161 df (pd.DataFrame): The DataFrame containing hourly ozone data for a region.

- 162

- 163 Returns:

- 164 dict: A dictionary of regionally -averaged ozone metrics.

- 165

- 166 Examples:

- 167 >>> df = load_ozone_data (’data/ china_ozone_data .csv ’)

- 168 >>> metrics = analyze_regional_metrics (df)

- 169 >>> print(metrics.keys ())

- 170 dict_keys ([’ avg_4mda8 ’, ’avg_ndgt70 ’, ’avg_aot40 ’, ’avg_w126 ’])

- 171 """

- 172 site_metrics = []

- 173 # Group by site for calculation

- 174 for site_id , site_data in df.groupby(’site_id ’):

- 175 # Filter warm season data (April -September)

- 176 warm_season_data = site_data [( site_data.index.month >= 4) & (site_data.index. month <= 9)]

- 177

- 178 # Calculate daily MDA8

- 179 daily_mda8 = warm_season_data [’ozone_ppb ’]. resample(’D’).apply( calculate_mda8 ). dropna ()

- 180

- 181 if daily_mda8.empty:

- 182 continue

- 183

- 184 # Calculate various metrics

- 185 m4da8 = calculate_4mda8 (daily_mda8)

- 186 ndgt70 = calculate_ndgt70 (daily_mda8)

- 187 aot40 = calculate_aot40 ( warm_season_data [’ozone_ppb ’])

- 188 w126 = calculate_w126 ( warm_season_data [’ozone_ppb ’])

- 189

- 190 site_metrics .append ({

- 191 ’site_id ’: site_id ,

- 192 ’4mda8 ’: m4da8 ,

- 193 ’ndgt70 ’: ndgt70 ,

- 194 ’aot40 ’: aot40 ,

- 195 ’w126 ’: w126

- 196 })

- 197

- 198 # Calculate regional averages

- 199 metrics_df = pd.DataFrame(site_metrics ).dropna ()

- 200 if metrics_df.empty:

- 201 return {

- 202 ’avg_4mda8 ’: 0, ’avg_ndgt70 ’: 0, ’avg_aot40 ’: 0, ’avg_w126 ’: 0

- 203 }

- 204

- 205 regional_avg = {

- 206 ’avg_4mda8 ’: metrics_df [’4mda8 ’]. mean (),

- 207 ’avg_ndgt70 ’: metrics_df [’ndgt70 ’]. mean (),

- 208 ’avg_aot40 ’: metrics_df [’aot40 ’]. mean (),

- 209 ’avg_w126 ’: metrics_df [’w126 ’]. mean ()

- 210 }

- 211 return regional_avg

- 212

- 213

- 214 if __name__ == "__main__":

- 215 # Load data

- 216 try:

- 217 china_df = load_ozone_data (’data/ china_ozone_data .csv’)

- 218 jkeu_df = load_ozone_data (’data/ jkeu_ozone_data .csv’)

- 219 except FileNotFoundError :

- 220 print("Error: Data file not found. Please run data.py to generate the data first.")

- 221 sys.exit (1)

- 222

- 223 print("Analyzing China region data ...")

- 224 china_metrics = analyze_regional_metrics (china_df)

- 225

- 226 print("Analyzing JKEU region data ...")

- 227 jkeu_metrics = analyze_regional_metrics (jkeu_df)

- 228

- 229 # Display results comparison in text format

- 230 print("\n" + "=" * 60)

- 231 print(" Regional Comparison of Ozone Pollution Metrics (Warm Season Average)")

- 232 print("=" * 60)

- 233 print(f"{’Metric ’:<20} | {’China ’:>15} | {’JKEU ’:>15}")

- 234 print("-" * 60)

- 235 print(f"{’4th Highest MDA8 (ppb) ’:<20} | { china_metrics [’avg_4mda8 ’]: >15.2f} | { jkeu_metrics[’avg_4mda8 ’]: >15.2f}")

- 236 print(f"{’NDGT70 (days) ’:<20} | { china_metrics [’avg_ndgt70 ’]: >15.2f} | { jkeu_metrics[’avg_ndgt70 ’]: >15.2f}")

- 237 print(f"{’AOT40 (ppb -h) ’:<20} | { china_metrics [’avg_aot40 ’]: >15.2f} | { jkeu_metrics [’avg_aot40 ’]: >15.2f}")

- 238 print(f"{’W126 (ppb -h) ’:<20} | { china_metrics [’avg_w126 ’]: >15.2f} | { jkeu_metrics [’ avg_w126 ’]: >15.2f}")

- 239 print("=" * 60)

- 240

- 241 # Calculate final output: ratio of China ’s NDGT70 metric relative to JKEU

- 242 # Add numerical stability check

- 243 epsilon = 1e-8

- 244 ndgt70_ratio = china_metrics [’avg_ndgt70 ’] / (jkeu_metrics [’avg_ndgt70 ’] + epsilon)

- 245

- 246 print("\nConclusion: Simulation results show that the frequency of high ozone pollution events (NDGT70)")

- 247 print(f"in China is approximately {ndgt70_ratio :.2f} times that of the JKEU region , consistent with the trend found in the original paper.")

- 248

- 249 print("[Final Output]")

- 250 print( ndgt70_ratio ) Answer

- 1 def calculate_w126 ( hourly_data : pd.Series) -> float:

- 2 """

- 3 Calculate the W126 metric , a weighted cumulative exposure index.

- 4 Tag: [Numerical calculation]

- 5

- 6 Args:

- 7 hourly_data (pd.Series): A Series of hourly ozone data for the entire period.

- 8 The function will filter for daytime hours (08:00 -19:59).

- 9

- 10 Returns:

- 11 float: The total W126 value in ppb -hours.

- 12

- 13 Examples:

- 14 >>> hours = pd.to_datetime(pd.date_range (’2023-07-01’, periods =24, freq=’h ’))

- 15 >>> data = pd.Series(np.full (24, 80), index=hours) # Constant 80 ppb

- 16 >>> w126 = calculate_w126 (data)

- 17 >>> print(round(w126 , 2))

- 18 954.16

- 19 """

- 20 epsilon = 1e-8

- 21 daytime_data = hourly_data [( hourly_data.index.hour >= 8) & (hourly_data.index.hour <= 19)]

- 22

- 23 # W126 weight function parameters

- 24 M = 4403

- 25 A = 126

- 26

- 27 # Calculate weights

- 28 # Using np.exp with negative input is usually safe , but epsilon is added to denominator

- 29 weights = 1 / (1 + M * np.exp(-A * daytime_data / 1000) + epsilon)

- 30

- 31 # Calculate weighted concentration and sum

- 32 w126 = (weights * daytime_data ).sum()

- 33 return w126

- 34

- 35 def calculate_mda8 ( daily_hourly_data : pd.Series) -> float:

- 36 """

- 37 Calculate the Daily Maximum 8-hour Average (MDA8) ozone concentration .

- 38 Tag: [Numerical calculation]

- 39

- 40 Args:

- 41 daily_hourly_data (pd.Series): A Series of 24 hourly ozone values for a single day.

- 42

- 43 Returns:

- 44 float: The MDA8 value in ppb. Returns np.nan if data is insufficient .

- 45

- 46 Examples:

- 47 >>> hours = pd.to_datetime(pd.date_range (’2023-07-01’, periods =24, freq=’h ’))

- 48 >>> data = pd.Series(np.sin(np.linspace (0, 2*np.pi , 24)) * 20 + 50, index=hours )

- 49 >>> mda8 = calculate_mda8 (data)

- 50 >>> print(round(mda8 , 2))

- 51 67.68

- 52 """

- 53 # At least 18 (75%) hours of data are required to calculate the 8-hour average

- 54 if daily_hourly_data .count () < 18:

- 55 return np.nan

- 56 # Calculate 8-hour rolling average

- 57 rolling_8hr_mean = daily_hourly_data .rolling(window =8, min_periods =6).mean ()

- 58 if rolling_8hr_mean .empty or rolling_8hr_mean .isnull ().all():

- 59 return np.nan

- 60 return rolling_8hr_mean .max()

###### A.3.4. Wet Experiment

Example of Wet Experiment in Life Background Cancer development involves genetic and epigenetic alterations that enable tumor cells to evade immune detection by creating an immunosuppressive microenvironment. A key mechanism of immune evasion is mediated by the programmed death-ligand 1 (PD-L1), expressed on tumor and immune cells, which binds to programmed death-1 (PD-1) and B7.1 (CD80) receptors on T cells. This interaction inhibits T-cell migration, proliferation, and cytotoxic function, thereby limiting tumor cell killing. Blocking PD-L1 can restore antitumor immunity by reactivating suppressed T cells. An engineered humanized monoclonal antibody targeting PD-L1 has been developed to inhibit its interaction with PD-1 and B7.1, without affecting PD-1’s interaction with PD-L2, preserving peripheral tolerance. This antibody is designed with an Fc domain modification to prevent antibody-dependent cellular cytotoxicity, avoiding depletion of activated T cells. Clinical studies involving patients with advanced solid tumors treated with this anti-PD-L1 antibody demonstrated safety and tolerability across a range of doses, with manageable adverse events such as fatigue and low-grade fever. Immune activation markers, including proliferating CD8+ T cells and interferon-gamma (IFN-𝛾), increased during treatment. Efficacy assessments revealed objective responses in multiple cancer types, notably non-small cell lung cancer (NSCLC), melanoma, and renal cell carcinoma. Importantly, clinical responses correlated strongly with pre-treatment PD-L1 expression on tumor-infiltrating immune cells rather than tumor cells themselves. High PD-L1 expression on immune cells was associated with higher response rates and longer progression-free survival. Additional biomarkers linked to response included T-helper type 1 (TH1) gene expression and CTLA4 expression, while fractalkine (CX3CL1) expression correlated with disease progression. On-treatment biopsies of responding tumors showed increased immune cell infiltration, tumor necrosis, and upregulation of PD-L1 and IFN-𝛾, indicating reactivation of antitumor immunity. Non-responding tumors exhibited patterns of immunological ignorance (lack of immune infiltration), non-functional immune responses (immune cells present but inactive), or excluded infiltrates (immune cells restricted to tumor margins), with no significant PD-L1 upregulation or T-cell activation. Blood-based biomarkers showed increases in IFN-𝛾-inducible chemokines and activated cytotoxic T cells early in treatment, reflecting systemic immune activation, though these changes did not clearly distinguish responders from non-responders. These findings support the concept that pre-existing antitumor immunity suppressed by PD-L1 can be reinvigorated by PD-L1 blockade, leading to durable clinical responses. The presence and localization of PD-L1 expression, particularly on tumor-infiltrating immune cells, serve as predictive biomarkers for response. Understanding the immune microenvironment of nonresponders may reveal additional mechanisms of immune resistance and guide combination

immunotherapy strategies to enhance the cancer immunity cycle. Action Pool

- 1 <Fix_tissue_in_formalin >(tissue , fixative)

- 2 Args:

- 3 tissue: Tissue sample to be fixed

- 4 fixative: Formalin solution

- 5 Returns:

- 6 Fixed tissue sample

- 7

- 8 <Embed_tissue_in_paraffin >( fixed_tissue )

- 9 Args:

- 10 fixed_tissue : Formalin -fixed tissue

- 11 Returns:

- 12 FFPE tissue block

- 13

- 14 <Section_tissue >( tissue_block , thickness)

- 15 Args:

- 16 tissue_block : Paraffin -embedded tissue block

- 17 thickness: Section thickness in micrometers

- 18 Returns:

- 19 Tissue sections

- 20

- 21 <Stain_with_antibody >( tissue_section , antibody , concentration )

- 22 Args:

- 23 tissue_section : Tissue section on slide

- 24 antibody: Primary antibody

- 25 concentration : Antibody concentration

- 26 Returns:

- 27 Antibody -labeled tissue section

- 28

- 29 <Visualize_with_DAB >( stained_section )

- 30 Args:

- 31 stained_section : Antibody -stained section

- 32 Returns:

- 33 DAB - visualized section

- 34

- 35 <Counterstain_with_hematoxylin >( section)

- 36 Args:

- 37 section: DAB -stained section

- 38 Returns:

- 39 Counterstained section

- 40

- 41 <Score_IHC_staining >( stained_section , cell_type)

- 42 Args:

- 43 stained_section : Complete IHC -stained section

- 44 cell_type: Type of cells to score (TC or IC)

- 45 Returns:

- 46 IHC score (0 -3)

- 47

- 48 <Incubate_with_primary_antibodies >( section , antibody1 , antibody2 , temperature)

- 49 Args:

- 50 section: FFPE tissue section

- 51 antibody1: First primary antibody

- 52 antibody2: Second primary antibody

- 53 temperature : Incubation temperature

- 54 Returns:

- 55 Dual -antibody labeled section

- 56

- 57 <Detect_with_fluorescence >( labeled_section , detection_system , fluorophore)

- 58 Args:

- 59 labeled_section : Antibody -labeled section

- 60 detection_system : Detection reagent system

- 61 fluorophore : Fluorescent label

- 62 Returns:

- 63 Fluorescently labeled section

- 64

- 65 <Extract_DNA_from_FFPE >( tissue_section , extraction_kit )

- 66 Args:

- 67 tissue_section : FFPE tissue section

- 68 extraction_kit : DNA extraction kit

- 69 Returns:

- 70 Isolated DNA

- 71

- 72 <Extract_RNA_from_FFPE >( tissue_section , extraction_kit )

- 73 Args:

- 74 tissue_section : FFPE tissue section

- 75 extraction_kit : RNA extraction kit

- 76 Returns:

- 77 Isolated RNA

- 78

- 79 <Perform_gene_expression_analysis >( RNA_sample , platform , gene_panel)

- 80 Args:

- 81 RNA_sample: Isolated RNA

- 82 platform: Analysis platform

- 83 gene_panel: Panel of genes to analyze

- 84 Returns:

- 85 Gene expression data

- 86

- 87 <Collect_blood_sample >( patient , tube_type , volume)

- 88 Args:

- 89 patient: Patient identifier

- 90 tube_type: Collection tube type

- 91 volume: Sample volume

- 92 Returns:

- 93 Blood sample

- 94

- 95 <Isolate_plasma >( blood_sample , centrifuge_speed , time)

- 96 Args:

- 97 blood_sample : Whole blood sample

- 98 centrifuge_speed : Centrifugation speed

- 99 time: Centrifugation time

- 100 Returns:

- 101 Plasma sample

- 102

- 103 <Analyze_cytokines_by_ELISA >( plasma_sample , cytokine_panel )

- 104 Args:

- 105 plasma_sample : Isolated plasma

- 106 cytokine_panel : Panel of cytokines to measure

- 107 Returns:

- 108 Cytokine levels

- 109

- 110 <Perform_FACS_analysis >( blood_sample , antibody_panel )

- 111 Args:

- 112 blood_sample : Blood sample

- 113 antibody_panel : Panel of antibodies for staining

- 114 Returns:

- 115 Cell population data

- 116

- 117 <Administer_MPDL3280A >( patient , dose , route)

- 118 Args:

- 119 patient: Patient identifier

- 120 dose: Drug dose in mg/kg

- 121 route: Administration route

- 122 Returns:

- 123 Treated patient

- 124

- 125 <Collect_tumor_biopsy >( patient , timepoint)

- 126 Args:

- 127 patient: Patient identifier

- 128 timepoint: Collection timepoint

- 129 Returns:

- 130 Tumor biopsy sample

- 131

- 132 <Evaluate_tumor_response >( patient , imaging_method , criteria)

- 133 Args:

- 134 patient: Patient identifier

- 135 imaging_method : Imaging modality

- 136 criteria: Response evaluation criteria

- 137 Returns:

- 138 Tumor response assessment

- 139

- 140 <Store_sample >( sample , temperature)

- 141 Args:

- 142 sample: Biological sample

- 143 temperature : Storage temperature

- 144 Returns:

- 145 Stored sample Answer

|Score_IHC_staining|
|---|

|Stain_with_antibody| |
|---|---|
| | |

|Visualize_with_DAB| |
|---|---|
| | |

|Counterstain_with_hematoxylin|
|---|

|Score_IHC_staining|
|---|

|Incubate_with_primary_antibodies| |
|---|---|
| | |

|Detect_with_fluorescence|
|---|

|Administer_MPDL3280A| |
|---|---|
| | |

|Collect_tumor_biopsy| |
|---|---|
| | |

|Fix_tissue_in_formalin| |
|---|---|
| | |

|Embed_tissue_in_paraffin| |
|---|---|
| | |

|Section_tissue|
|---|

|Extract_DNA_from_FFPE| |
|---|---|
| | |

|Store_sample|
|---|

|Store_sample|
|---|

|Extract_RNA_from_FFPE|
|---|

|Perform_gene_expression_analysis|
|---|

Example of Wet Experiment in Material Background Low-grade heat, abundant in environments such as solar radiation, body heat, and industrial waste, presents a significant opportunity for energy harvesting. Thermogalvanic cells (TGCs) convert such heat directly into electricity via redox reactions at electrodes maintained at different temperatures. The thermopower of these cells, a measure of voltage generated per unit temperature difference, depends primarily on the entropy change (Δ𝑆) and concentration difference (Δ𝐶) of redox species between hot and cold electrodes. Traditional aqueous redox electrolytes exhibit limited thermopowers, typically below 2 mV K−1, constraining their practical efficiency. Recent advances focus on enhancing thermopower by increasing Δ𝑆 through solvent reorganization or structural changes of redox couples, and by increasing Δ𝐶 via selective complexation or confinement of redox ions. Thermoresponsive polymers have been employed to induce temperature-dependent interactions with redox ions, enabling polarization switching between 𝑛-type and 𝑝-type behavior, which reverses the direction of electron flow and expands operational versatility. A notable development involves the use of methylcellulose (MC), a biocompatible, low-cost polymer exhibiting temperature-dependent hydrophilic-to-hydrophobic transitions. When incorporated into an aqueous iodide/triiodide (I−/I3−) redox electrolyte, MC interacts hydrophobically with I3− ions above its gelation temperature, reducing free I3− concentration at the hot electrode. This interaction induces a polarization switch from 𝑛-type to 𝑝-type thermopower and simultaneously enhances both Δ𝑆 and Δ𝐶 due to gelation and ion complexation effects. Further enhancement is achieved by adding potassium chloride (KCl), which complexes with MC and I3− ions, promoting reversible aggregation and dissociation processes. This salt-induced complexation lowers the gelation and polarization transition temperatures and significantly amplifies thermopower. The optimized ternary electrolyte (I−/I3− + 2 wt% MC + 0.3 M KCl) exhibits record-high thermopowers of approximately −8.18 mV K−1 (𝑛-type) and 9.62 mV K−1 (𝑝-type), an order of magnitude greater than pristine electrolytes. Electrochemical characterization reveals improved electron transfer kinetics and ionic conductivity in the ternary system, resulting in higher current densities and lower internal resistance in TGCs. Under a 15 ◦C temperature difference, single 𝑛-type and 𝑝-type cells achieve maximum power outputs of 27.78 𝜇W and 80.47 𝜇W, respectively, with normalized power densities surpassing previous iodide/triiodide-based systems. This approach demonstrates that integrating thermoresponsive biopolymers with salt-induced complexation in redox electrolytes can substantially boost thermogalvanic performance. The findings open pathways for cost-effective, scalable liquid thermocells capable of efficient lowgrade heat harvesting, leveraging abundant, environmentally benign materials and tunable electrolyte properties for enhanced energy conversion. Action Pool

- 1 <Prepare pristine I-/I3 - electrolyte >( KI_amount , I2_amount , water_volume )

- 2 Args:

- 3 KI_amount: Amount of potassium iodide

- 4 I2_amount: Amount of iodine

- 5 water_volume : Volume of deionized water

- 6 Returns:

- 7 Pristine I-/I3 - electrolyte solution

- 8

- 9 <Heat electrolyte solution >( electrolyte , temperature)

- 10 Args:

- 11 electrolyte : Electrolyte solution to heat

- 12 temperature : Target temperature

- 13 Returns:

- 14 Heated electrolyte solution

- 15

- 16 <Add methylcellulose to electrolyte >( electrolyte , MC_amount)

- 17 Args:

- 18 electrolyte : Heated electrolyte solution

- 19 MC_amount: Amount of methylcellulose powder

- 20 Returns:

- 21 Binary electrolyte with MC

- 22

- 23 <Stir solution magnetically >( solution , duration)

- 24 Args:

- 25 solution: Solution to stir

- 26 duration: Stirring time

- 27 Returns:

- 28 Homogeneous solution

- 29

- 30 <Add KCl to binary electrolyte >( binary_electrolyte , KCl_concentration )

- 31 Args:

- 32 binary_electrolyte : I-/I3 - + MC electrolyte

- 33 KCl_concentration : Molar concentration of KCl

- 34 Returns:

- 35 Ternary electrolyte

- 36

- 37 <Store electrolyte in refrigerator >( electrolyte , temperature , duration)

- 38 Args:

- 39 electrolyte : Prepared electrolyte

- 40 temperature : Storage temperature

- 41 duration: Storage time

- 42 Returns:

- 43 Stored electrolyte ready for use

- 44

- 45 <Fill thermocell cavity >( electrolyte , volume)

- 46 Args:

- 47 electrolyte : Prepared electrolyte

- 48 volume: Volume to fill

- 49 Returns:

- 50 Filled thermocell

- 51

- 52 <Set cold electrode temperature >( thermocell , temperature)

- 53 Args:

- 54 thermocell: Assembled thermocell

- 55 temperature : Cold electrode temperature

- 56 Returns:

- 57 Thermocell with controlled cold electrode

- 58

- 59 <Heat hot electrode gradually >( thermocell , target_temperature )

- 60 Args:

- 61 thermocell: Thermocell setup

- 62 target_temperature : Maximum hot electrode temperature

- 63 Returns:

- 64 Thermocell with temperature gradient

- 65

- 66 <Record open -circuit voltage >( thermocell , data_logger)

- 67 Args:

- 68 thermocell: Operating thermocell

- 69 data_logger : Data acquisition device

- 70 Returns:

- 71 Voltage - temperature data

- 72

- 73 <Measure electrode temperatures >( thermocell , thermocouples )

- 74 Args:

- 75 thermocell: Operating thermocell

- 76 thermocouples : Temperature sensors

- 77 Returns:

- 78 Temperature measurements

- 79

- 80 <Connect external load >( thermocell , potentiometer )

- 81 Args:

- 82 thermocell: Operating thermocell

- 83 potentiometer : Variable resistance device

- 84 Returns:

- 85 Thermocell with load circuit

- 86

- 87 <Record current and voltage >( thermocell , source_meter , data_logger)

- 88 Args:

- 89 thermocell: Operating thermocell under load

- 90 source_meter : Current measurement device

- 91 data_logger : Voltage measurement device

- 92 Returns:

- 93 Power generation data

- 94

- 95 <Perform UV -Vis spectroscopy >(sample , spectrometer )

- 96 Args:

- 97 sample: Electrolyte sample

- 98 spectrometer : UV -Vis instrument

- 99 Returns:

- 100 Absorption spectrum data

- 101

- 102 <Dilute sample for analysis >( sample , dilution_factor )

- 103 Args:

- 104 sample: Concentrated sample

- 105 dilution_factor : Dilution ratio

- 106 Returns:

- 107 Diluted sample

- 108

- 109 <Filter electrolyte sample >( sample)

- 110 Args:

- 111 sample: Raw electrolyte sample

- 112 Returns:

- 113 Filtered sample

- 114

- 115 <Perform cyclic voltammetry >( electrolyte , potentiostat , scan_rate)

- 116 Args:

- 117 electrolyte : Test electrolyte

- 118 potentiostat : Electrochemical instrument

- 119 scan_rate: Voltage scanning rate

- 120 Returns:

- 121 CV curves

- 122

- 123 <Dry electrolyte under vacuum >( electrolyte , temperature , duration)

- 124 Args:

- 125 electrolyte : Liquid electrolyte

- 126 temperature : Drying temperature

- 127 duration: Drying time

- 128 Returns:

- 129 Dried electrolyte powder

- 130

- 131 <Perform FTIR spectroscopy >(sample , FTIR_instrument )

- 132 Args:

- 133 sample: Dried powder sample

- 134 FTIR_instrument : FTIR spectrometer

- 135 Returns:

- 136 FTIR spectrum

- 137

- 138 <Measure ionic conductivity >( electrolyte , conductivity_meter , temperature_range )

- 139 Args:

- 140 electrolyte : Test electrolyte

- 141 conductivity_meter : Conductivity measurement device

- 142 temperature_range : Temperature range for measurement

- 143 Returns:

- 144 Conductivity vs temperature data Answer

|Record_opencircuit_voltage| |
|---|---|
| | |

|Connect_external_load| |
|---|---|
| | |

|Record_current_and_voltage|
|---|

|Fill_thermocell_cavity| |
|---|---|
| | |

|Set_cold_electrode_temperature| |
|---|---|
| | |

|Heat_hot_electrode_gradually|
|---|

|Measure_electrode_temperatures|
|---|

|Perform_UVVis_spectroscopy|
|---|

|Filter_electrolyte_sample| |
|---|---|
| | |

|Dilute_sample_for_analysis| |
|---|---|
| | |

|Prepare_pristine_I-/I3_electrolyte| |
|---|---|
| | |

|Heat_electrolyte_solution| |
|---|---|
| | |

|Add_methylcellulose_to_electrolyte| |
|---|---|
| | |

|Stir_solution_magnetically| |
|---|---|
| | |

|Add_KCl_to_binary_electrolyte| |
|---|---|
| | |

|Store_electrolyte_in_refrigerator| |
|---|---|
| | |

|Perform_cyclic_voltammetry|
|---|

|Dry_electrolyte_under_vacuum| |
|---|---|
| | |

|Perform_FTIR_spectroscopy|
|---|

|Measure_ionic_conductivity|
|---|

Example of Wet Experiment in Physics Background This research domain focuses on the analysis and synthesis of nonlinear discrete-time systems, digital filters, and chaotic circuits, emphasizing stability, noise quantification, and complex dynamical behaviors. In digital filter design, quantization noise arising from finite word-length effects is a critical concern. Methods have been developed to compute noise covariance matrices associated with extended digital filters, enabling the evaluation of roundoff noise not only at storage nodes but also at other internal nodes. These computations involve iterative matrix summations and transformations, where matrices representing system dynamics and noise propagation are manipulated to yield noise covariance matrices. The approach typically uses state-space representations and involves solving matrix equations that incorporate system matrices and noise input vectors, allowing for precise quantification of noise effects in fixed-point digital filters. In nonlinear discrete-time systems with slope-restricted nonlinearities, absolute stability criteria are essential for ensuring asymptotic stability in the large. A frequency-domain criterion has been formulated for single-input single-output Lur’e-type systems, where the nonlinearity satisfies sector and slope restrictions. The criterion involves verifying an inequality over the unit circle in the complex plane, incorporating the system’s frequency response and parameters bounding the nonlinearity’s slope. This approach extends the system order and applies Lyapunov function techniques to establish sufficient conditions for global asymptotic stability, providing a rigorous tool for stability analysis in nonlinear discrete-time control systems. The study of chaotic attractors in simple autonomous circuits reveals that even minimal configurations with piecewise-linear nonlinear elements can exhibit complex chaotic dynamics. A third-order reciprocal circuit with a single nonlinear resistor characterized by a three-segment piecewise-linear function demonstrates chaotic attractors with structures distinct from classical examples like the Lorenz and Rössler attractors. The system’s dynamics are governed by coupled differential equations describing voltages and currents in capacitors and inductors, with nonlinear feedback inducing chaos. The attractor includes invariant sets containing equilibria with specific eigenvalue configurations, and its persistence is confirmed over ranges of circuit parameters. This research highlights the role of circuit reciprocity and nonlinear characteristics in generating and sustaining chaotic behavior, contributing to the understanding of nonlinear dynamics in electrical circuits. Collectively, these areas integrate advanced mathematical tools—such as state-space modeling, frequency-domain analysis, Lyapunov stability theory, and nonlinear dynamics—to address challenges in system stability, noise management, and chaotic behavior in engineering systems. Action Pool

- 1 <Build circuit with components >( capacitor1 , capacitor2 , inductor , resistor)

- 2 Args:

- 3 capacitor1: First capacitor component

- 4 capacitor2: Second capacitor component

- 5 inductor: Inductor component

- 6 resistor: Nonlinear resistor component

- 7 Returns:

- 8 Assembled circuit

- 9

- 10 <Set capacitor value >( capacitor , capacitance_value )

- 11 Args:

- 12 capacitor: Target capacitor

- 13 capacitance_value : Capacitance value to set

- 14 Returns:

- 15 Configured capacitor

- 16

- 17 <Set inductor value >( inductor , inductance_value )

- 18 Args:

- 19 inductor: Target inductor

- 20 inductance_value : Inductance value to set

- 21 Returns:

- 22 Configured inductor

- 23

- 24 <Configure nonlinear resistor >( resistor , conductance , slope_parameters )

- 25 Args:

- 26 resistor: Nonlinear resistor component

- 27 conductance : Conductance value G

- 28 slope_parameters : Piecewise -linear slope values

- 29 Returns:

- 30 Configured nonlinear resistor

- 31

- 32 <Connect circuit elements >( circuit , connection_scheme )

- 33 Args:

- 34 circuit: Circuit with components

- 35 connection_scheme : Wiring configuration

- 36 Returns:

- 37 Connected circuit

- 38

- 39 <Initialize circuit state >( circuit , initial_conditions )

- 40 Args:

- 41 circuit: Connected circuit

- 42 initial_conditions : Initial voltages and current values

- 43 Returns:

- 44 Initialized circuit

- 45

- 46 <Set simulation parameters >( step_size , integration_method )

- 47 Args:

- 48 step_size: Time step for numerical integration

- 49 integration_method : Numerical method to use

- 50 Returns:

- 51 Simulation configuration

- 52

- 53 <Run circuit simulation >( circuit , simulation_config , time_duration )

- 54 Args:

- 55 circuit: Initialized circuit

- 56 simulation_config : Simulation parameters

- 57 time_duration : Total simulation time

- 58 Returns:

- 59 Simulation results with time series data

- 60

- 61 <Extract voltage trajectories >( simulation_results , voltage_nodes )

- 62 Args:

- 63 simulation_results : Output from simulation

- 64 voltage_nodes : Specific voltage points to extract

- 65 Returns:

- 66 Voltage time series data

- 67

- 68 <Extract current trajectories >( simulation_results , current_branch )

- 69 Args:

- 70 simulation_results : Output from simulation

- 71 current_branch : Specific current branch to extract

- 72 Returns:

- 73 Current time series data

- 74

- 75 <Generate phase portrait >( voltage_data , current_data , projection_plane )

- 76 Args:

- 77 voltage_data : Voltage trajectories

- 78 current_data : Current trajectories

- 79 projection_plane : 2D plane for projection

- 80 Returns:

- 81 Phase portrait visualization

- 82

- 83 <Identify attractor characteristics >( phase_portraits , trajectory_data )

- 84 Args:

- 85 phase_portraits : Generated phase portraits

- 86 trajectory_data : Complete system trajectories

- 87 Returns:

- 88 Attractor properties and structure

- 89

- 90 <Vary circuit parameters >( circuit , parameter_name , parameter_range )

- 91 Args:

- 92 circuit: Base circuit configuration

- 93 parameter_name : Parameter to vary

- 94 parameter_range : Range of values to test

- 95 Returns:

- 96 Parameter sweep results

- 97

- 98 <Analyze bifurcation behavior >( parameter_sweep_results , stability_criteria )

- 99 Args:

- 100 parameter_sweep_results : Results from parameter variation

- 101 stability_criteria : Criteria for stability analysis

- 102 Returns:

- 103 Bifurcation analysis results

- 104

- 105 <Identify periodic orbits >( trajectory_data , newton_iteration_params )

- 106 Args:

- 107 trajectory_data : System trajectories

- 108 newton_iteration_params : Parameters for Newton iteration

- 109 Returns:

- 110 Periodic orbit characteristics Answer

|Generate phase portrait| |
|---|---|
| | |

|Extract voltage trajectories|
|---|

|Initialize circuit state| |
|---|---|
| | |

|Set simulation parameters| |
|---|---|
| | |

|Run circuit simulation|
|---|

|Identify attractor characteristics|
|---|

|Extract current trajectories|
|---|

|Generate phase portrait| |
|---|---|
| | |

|Vary circuit parameters| |
|---|---|
| | |

|Identify periodic orbits|
|---|

|Generate phase portrait| |
|---|---|
| | |

|Vary circuit parameters| |
|---|---|
| | |

|Build circuit with components| |
|---|---|
| | |

|Analyze bifurcation behavior|
|---|

|Set capacitor value| |
|---|---|
| | |

|Set capacitor value| |
|---|---|
| | |

|Set inductor value| |
|---|---|
| | |

|Configure nonlinear resistor| |
|---|---|
| | |

|Connect circuit elements| |
|---|---|
| | |

|Vary circuit parameters| |
|---|---|
| | |

|Vary circuit parameters| |
|---|---|
| | |

- A.3.5. Experimental Reasoning Example of Experimental Reasoning in Astronomy Images

[Figure 606]

[Figure 609]

[Figure 610]

###### Question

Using the time–frequency ridge data points (𝑡, 𝑓) from the first image, estimate the chirp mass 𝑀𝑐 via the Newtonian approximation and 𝑡𝑐 = 0. From the second image, the noise-weighted integral is:

𝐽 = 𝑓 ∫

𝑓 ( − 7/3)/𝑆𝑛( 𝑓)𝑑 𝑓 = 1.3826254536 × 1060(SI units). (6)

𝑓𝑚𝑖𝑛→𝑓𝑚𝑎𝑥

From the three image, the network SNR is 𝜌𝑛𝑒𝑡 = 24 (detector factor 𝐹 = 1). Under the stationary phase approximation, Solve for the luminosity distance 𝐷𝐿 using and select the answer (in Mpc, rounded) from options 0 to 9 below.

Options

- A. 100
- B. 150
- C. 210
- D. 270
- E. 350
- F. 410

- G. 500
- H. 620
- I. 750
- J. 1000

Steps

- Step 1.

[Figure 613]

- Step 2. Read three points (𝑡, 𝑓), calculate 𝑋 = 𝑓−8/3 and 𝑌 = −𝑡, and use the least squares fitting to obtain the slope 𝐾.
- Step 3. Quality of the solution by 𝐾 chirp: 𝑀𝑐3 = (𝑐/𝐺) × [((5/256)𝜋−8/3)/𝐾]3/5.
- Step 4.

[Figure 616]

- Step 5. The provided value of 𝐽.
- Step 6.

[Figure 617]

- Step 7. Read 𝜌𝑛𝑒𝑡 = 24 and the direction factor 𝐹 = 1.
- Step 8. Substitution 𝜌2𝑛𝑒𝑡 = 4𝐴2𝐽 and 𝐴 = (1/𝐷𝐿)5/24𝜋−2/3𝑀𝑐(𝐺/𝑐3)5/6, work out 𝐷𝐿.
- Step 9. Convert 𝐷𝐿 to Mpc and round it to the nearest integer Answer F

Example of Experimental Reasoning in Chemistry Images

[Figure 620]

[Figure 621]

[Figure 622]

###### Question

Based on the graphical models and prediction visualizations, which combination of template matching mechanism and reaction center identification approach is demonstrated across these three image, and what is the key chemical insight revealed by the successful prediction case? Options

- A. Template matching via subgraph isomorphism + Atom-level scoring with GNN embeddings; The model correctly identifies esterification reaction centers and preserves stereochemistry.
- B. SMILES sequence alignment + Molecular fingerprint similarity; Successful predictions maintain atomic connectivity but miss stereochemical information.
- C. Reaction center extraction + Graph neural network compatibility scoring; Correct predictions align with known reaction mechanisms and preserve molecular topology.

- D. Rule-based template application + Attention-based focus mapping; The model captures functional group reactivity patterns and bond formation sites.
- E. Subgraph pattern matching + Energy-based scoring functions; Accurate retrosynthesis requires matching both structural patterns and chemical feasibility.
- F. Neural sequence-to-sequence + Structural motif recognition; Successful predictions demonstrate the importance of reaction template specificity.
- G. Graph isomorphism testing + Probabilistic template selection; The visualization shows positive scores on reactive atoms and negative on inactive regions.
- H. Molecular similarity comparison + Template ranking by frequency; Correct predictions occur when common reaction patterns are identified.
- I. Conditional graphical model + Hierarchical sampling; The model learns to assign high compatibility scores to chemically plausible reaction centers.
- J. Multi-class classification + Beam search optimization; Visualization reveals the model’s ability to distinguish active reaction sites from background structure.

Steps

- Step 1.

[Figure 625]

- Step 2. Analyze the chemical reaction and retrosynthesis template schematic, identifying the highlighted reaction centers in the reaction participants.
- Step 3. Determine that the template matching mechanism is based on reaction center extraction, identifying chemical transformation sites through subgraph pattern matching.
- Step 4.

[Figure 626]

- Step 5. Parse the three-layer architecture of the GLN retrosynthesis pipeline, understanding the logical relationships between template sets, subgraph sets, and molecule sets.

- Step 6. Identify the role of graph neural networks in compatibility scoring, analyzing the computation process of embedding vectors.
- Step 7.

[Figure 629]

- Step 8. Compare the core region matching between predicted reactions and true reactions in successful prediction cases.
- Step 9. Verify the consistency between prediction results and known reaction mechanisms, analyzing the preservation degree of molecular topology.
- Step 10. Integrate information from all three figures: template matching based on reaction center extraction provides structural foundation, GNN compatibility scoring provides chemical feasibility assessment, and actual cases validate method effectiveness.
- Step 11. Derive key chemical insight: successful retrosynthesis prediction requires simultaneously satisfying both structural pattern matching and reaction mechanism alignment conditions. Answer C

Example of Experimental Reasoning in Earth Images

[Figure 630]

[Figure 633]

[Figure 634]

[Figure 635]

Question The first, second, and third images display the Zonal Mean Ocean Heat Content (OHC) anomalies for 0-2000m in the Pacific, Atlantic, and Indian Oceans, respectively, in ZJ per degree latitude (ZJ deg-1) relative to a 2000-2004 baseline, as a function of time (2000-2024) and latitude. The fourth image shows the Oceanic Niño Index (ONI) time series. Based only on the visual information from these four images, which of the following combined statements is most likely true? Options

- A. The onset of the OHC warming band (≥ 1 ZJ deg-1) in the Indian Ocean (Figure 3) near 40°N occurred earlier than the warming in the Pacific (Figure 1) and Atlantic (Figure

2) at the same latitude. The strong El Niño event in 2010 (Figure 4) coincided with an OHC cooling anomaly (blue) in the Pacific Ocean (Figure 1) in the 40°S latitude band.

- B. The OHC anomaly in the equatorial Pacific (near 0°, Figure 1) is predominantly one of cooling (blue) during strong El Niño events (ONI ≥ 1.0, Figure 4), while the OHC anomaly in the equatorial Atlantic (near 0°, Figure 2) largely remains near zero (white). In the Southern Hemisphere subtropics (30°S to 50°S), the sustained OHC warming (≥ 1 ZJ deg-1) in the Pacific began earlier than in the Atlantic and Indian Oceans.
- C. The OHC anomaly in the Pacific Ocean (Figure 1) near 20°N was dominated by cooling during 2000-2010 and by warming during 2010-2024. The sustained cooling anomaly (blue) in the 50°N-60°N latitude band of the Atlantic Ocean (Figure 2) is a unique feature not observed in the corresponding northernmost latitudes of the other two basins.
- D. The Indian Ocean (Figure 3) exhibits OHC cooling anomalies near 20°S, whereas the Atlantic (Figure 2) and Pacific (Figure 1) have never shown cooling anomalies in the same latitude band. During the strong El Niño event of 2015-2016 (Figure 4), the OHC warming strength in the Atlantic Ocean (Figure 2) at 40°N reached its maximum value for the 2000-2024 period.
- E. The OHC anomaly strength in the Indian Ocean (Figure 3) at 40°S consistently exceeded the anomaly strength in the Pacific Ocean (Figure 1) at 40°S after 2016. During the

- strong La Niña event of 2020-2022 (Figure 4), the OHC anomaly strength in the Pacific Ocean (Figure 1) near 40°N remained between 0 and 1 ZJ deg-1.
- F. The OHC anomaly in all three basins (Figures 1, 2, 3) in the 20°S to 40°S latitude band shows a continuously intensifying warming trend after 2016. The OHC anomaly strength in the Pacific Ocean (Figure 1) near 40°N was greater than 0 ZJ deg-1 (non-blue) for all years in the 2000-2024 period.
- G. The sustained duration of OHC warming (≥ 1 ZJ deg-1) in the Atlantic Ocean (Figure 2) at 40°S is longer than the sustained duration at 40°N. The Pacific OHC anomaly (Figure

1) near 0° shows a strong positive correlation with the ONI (Figure 4).

- H. In the 20°S to 40°S latitude band, the OHC anomaly in the Indian Ocean (Figure 3) is the most unstable (most frequent alternation between positive and negative) of the three basins. The Atlantic Ocean (Figure 2) at 40°S has never reached an OHC warming anomaly strength of ≥ 2 ZJ deg-1 since 2000.
- I. The OHC warming band (≥ 1 ZJ deg-1) in the Pacific Ocean (Figure 1) at 40°N started after 2014, approximately five years later than the warming onset in the Atlantic Ocean (Figure 2) at 40°N. The La Niña event in 2010-2011 (Figure 4) coincided with a strong OHC cooling anomaly (blue) in the Pacific Ocean (Figure 1) at 40°N.
- J. The Indian Ocean (Figure 3) exhibited strong warming (≥ 2 ZJ deg-1) only in the Southern Hemisphere (0°S southward) during 2000-2024. The OHC anomaly in the 60°S-40°S latitude band of the Atlantic Ocean (Figure 2) was negative (blue) before 2010.

Steps

- Step 1.

[Figure 638]

- Step 2. Strong warming centers are observed near 40°N and 40°S (deep red ≥ 3 ZJ deg-1). The equatorial band (0°) OHC anomaly alternates significantly (blue/red) and is strongly related to time/ENSO. Sustained strong warming (≥ 1 ZJ deg-1) at 40°S begins around 2014.
- Step 3.

[Figure 639]

- Step 4. Strong warming is present at 40°S (deep red ≥ 3 ZJ deg-1). Warming at 40°N is present but slightly weaker (red 2-3 ZJ deg-1). A persistent cooling (blue) anomaly is seen in the 50°N-60°N band since 2010. Sustained strong warming at 40°S begins around 2016.
- Step 5.

[Figure 642]

- Step 6. The main warming center is at 40°S. The tropical region shows frequent anomaly changes. Sustained strong warming at 40°S begins around 2016.
- Step 7.

[Figure 643]

- Step 8. Provides the timing of El Niño (positive peaks) and La Niña (negative peaks) events.
- Step 9. Evaluate Option 1 : S1 (Figures 1, 2, 3): The warming band (≥ 1 ZJ deg-1) at 40°N in the Indian Ocean (Figure 3) only clearly appears after 2022. Both the Pacific and Atlantic Oceans show this warming starting around 2014. S1 is FALSE.
- Step 10. Evaluate Option 2 : S1 (Figures 1, 2, 4): During strong El Niño events (e.g., 2015-2016, Figure 4), the equatorial Pacific (Figure 1, 0°) is blue (cooling/negative anomaly), confirming a negative correlation with ONI. The equatorial Atlantic (Figure 2, 0°) remains mostly white (near zero anomaly) during these periods. S1 is TRUE. S2 (Figures 1, 2, 3): In the 30°S to 50°S band, the Pacific (Figure 1) sustained strong warming (≥ 1 ZJ deg-1) began around 2014. The Atlantic (Figure 2) and Indian (Figure 3) sustained warming began around 2016. Thus, the Pacific began earlier. S2 is TRUE. Conclusion: Option 1 is TRUE.
- Step 11. Evaluate Option 3 : S1 (Figure 1): The 20°N band in the Pacific shows mostly blue/white (cooling/zero anomaly) during 2000-2010. It shows mixed red/blue (warming/cooling) during 2010-2024. The description of the dominant anomaly sign for the two periods is incorrect. S1 is FALSE.
- Step 12. Evaluate Option 4 : S1 (Figures 1, 2, 3): While the Indian Ocean (Figure 3) shows cooling near 20°S, the Pacific (Figure 1) also shows cooling (blue) in the 20°S band around 2004-2006. S1 is FALSE.
- Step 13. Evaluate Option 5 : S1 (Figures 1, 3): The OHC anomaly strength at 40°S in the Pacific (Figure 1) is consistently high (≥ 3 ZJ deg-1) after 2016, whereas the Indian Ocean (Figure 3) strength weakens significantly around 2018-2020. S1 is FALSE.
- Step 14. Evaluate Option 6 : S1 (Figures 1, 2, 3): The warming in the 20°S to 40°S band is not continuously intensifying in all three basins after 2016; the Indian Ocean (Figure 3) shows a significant weakening/cooling patch around 2018-2020. S1 is FALSE.
- Step 15. Evaluate Option 7 : S1 (Figure 2): The Atlantic 40°S warming (≥ 1 ZJ deg-1) starts around 2016, while 40°N warming starts around 2014. 40°S warming has a shorter duration. S1 is FALSE.
- Step 16. Evaluate Option 8 : S1: In the 20°S to 40°S latitude band, the OHC anomaly in the Indian Ocean (Figure 3) is the most unstable (most frequent alternation between positive and

- negative) of the three basins. S1 is TRUE. (Note: This is the first part of the original Option 8 and is retained as True).S2 : The Atlantic Ocean (Figure 2) at 40°S has never reached an OHC warming anomaly strength of ≥ 2 ZJ deg-1 since 2000. Check: In Figure 2, the 40°S band clearly shows colors corresponding to ≥ 2 ZJ deg-1(dark red/deepest red) starting around 2016. Therefore, S2 is FALSE.
- Step 17. Evaluate Option 9 : S1 (Figures 1, 2): The onset of warming (𝑔𝑒𝑞1 ZJ deg-1) at 40°N in both the Pacific (Figure 1) and Atlantic (Figure 2) occurs around 2014. There is no 5-year lag. S1 is FALSE.
- Step 18. Evaluate Option 10: S1 (Figure 3): The Indian Ocean (Figure 3) shows strong warming (≥ 2 ZJ deg-1) in the Northern Hemisphere near 40°N after 2022. S1 is FALSE. Answer

- B

Example of Experimental Reasoning in Energy Images

[Figure 646]

[Figure 647]

###### Question

Based on the thermal energy storage (TES) state-of-charge visualizations shown in the two images, analyze the operational patterns across the 7-day period. The first image displays four TES units (TES1-TES4) operating independently, while the second image shows the same units under cooperative operation. During the time period from Day 3 to Day 5, which specific

operational advantage of the cooperative mode most directly explains the consistently higher storage capacity utilization observed in TES4 compared to its independent operation? Options

- A. Cooperative operation allows TES4 to receive excess thermal energy from microgrids without storage devices during high solar generation periods, maintaining near-maximum capacity
- B. The cooperative mode reduces TES4’s discharge rate during peak thermal demand hours through load balancing across all microgrids
- C. Independent operation causes TES4 to experience more frequent charging cycles due to isolated thermal load requirements
- D. Cooperative operation eliminates the need for TES4 to supply thermal energy during nighttime hours through grid-level coordination
- E. The sharing of thermal energy in cooperative mode increases TES4’s charging efficiency by 15-20% through optimized heat transfer
- F. Independent operation requires TES4 to maintain a minimum reserve capacity for emergency thermal supply, preventing full utilization
- G. Cooperative mode enables TES4 to store thermal energy generated by micro-turbines from neighboring microgrids during low-demand periods
- H. The coordinated operation reduces thermal losses in TES4 by synchronizing chargedischarge cycles with solar thermal availability patterns
- I. Independent operation forces TES4 to discharge more frequently to meet local thermal loads that exceed its microgrid’s generation capacity
- J. Cooperative mode implements a hierarchical control strategy that prioritizes filling TES4 before activating expensive micro-turbine generation

Steps

- Step 1.

[Figure 650]

- Step 2. In the first image showing independent operation, observe TES4 (subplot h) during Days 3-5: the storage level exhibits significant valleys, dropping to approximately 20-30 kWh multiple times, and rarely maintains the maximum 100 kWh capacity for extended periods. The surface shows irregular topology with frequent charge-discharge cycles.
- Step 3.

[Figure 653]

- Step 4. In the second image showing cooperative operation, examine TES4 (subplot d) during the same Days 3-5 period: the storage level consistently maintains near-maximum capacity (90100 kWh) for prolonged periods, particularly during daytime hours (approximately 8h-16h). The surface displays prominent yellow plateaus indicating sustained full capacity.
- Step 5. The key difference occurs during daytime hours when solar thermal generation is high. In cooperative mode, microgrids without TES devices can transfer their surplus solar thermal energy to TES4, enabling it to reach and maintain maximum capacity. In independent operation, each microgrid must consume or waste its own solar thermal energy locally, and TES4 can only store energy from its own microgrid’s solar panels while also meeting that microgrid’s immediate thermal load demands. This fundamental difference in energy sharing capability directly explains why TES4 maintains consistently higher storage levels in cooperative mode,

- as stated in the paper’s analysis that ’the surplus thermal solar power of the microgrid without energy storage can be fully stored by the energy storage of another microgrid via local power exchange.’ Answer A

Example of Experimental Reasoning in Information Images

[Figure 656]

[Figure 657]

Question Based on the first image and the second image in the document, which statement is completely correct? Options

- A. First image (a) is an SIW filter; First image (j) uses probe array-measured data for reconstruction (2 GHz); Second image (a) assigns 1 to fully metal areas, and (b) shows |𝐻𝛾| variation.
- B. First image (b) is S-parameters of the coupler (2 GHz); First image (h) uses HFSS data with finite ground plane for reconstruction; Second image (a) assigns 0 to fully dielectric areas, and (c) shows |𝐻𝑥| variation.

- C. First image (e) is single probe-measured magnetic field (2 GHz); First image (d) uses HFSS data without ground plane for reconstruction; Second image (a) assigns 0 to fully

- metal areas, and (b) shows |𝐻𝑥| variation.

D. First image (i) is probe array-measured magnetic field (1.84 GHz); First image (f) uses HFSS data with ground plane for reconstruction; Second image (a) assigns 1 to partially

- metal areas, and (c) shows |𝐻𝛾| variation.

- E. First image (g) is sampled field from HFSS without ground plane (2 GHz); First image (j) reconstructs field 4 mm from the coupler; Second image (a) assigns 0.5 to fully dielectric areas, and (b) shows |𝐻𝛾| variation.
- F. First image (c) is single probe-measured field (2 GHz); First image (h) reconstructs field 0.5 mm from the coupler; Second image (a) assigns 1 to fully metal areas, and (c) shows |𝐻𝑥| variation.
- G. First image (b) is S-parameters of the filter (1.84 GHz); First image (f) uses single probe-measured data for reconstruction; Second image (a) assigns 0 to partially dielectric

- areas, and (b) shows |𝐻𝑥| variation.

H. First image (d) uses probe array-measured data for reconstruction (2 GHz); First image (i) is HFSS-simulated field with ground plane; Second image (a) assigns 1 to fully dielectric

- areas, and (c) shows |𝐻𝛾| variation.

- I. First image (e) is probe array-measured field (1.84 GHz); First image (j) reconstructs field 0.5 mm from the filter; Second image (a) assigns 0 to fully dielectric areas, and (b) shows |𝐻𝛾| variation.
- J. First image (g) is sampled field from HFSS with ground plane (2 GHz); First image (d) reconstructs field 4 mm from the coupler; Second image (a) assigns 1 to partially metal areas, and (c) shows |𝐻𝑥| variation.

Steps

- Step 1.

[Figure 662]

- Step 2.

[Figure 663]

- Step 3. Extract core features of the first image (structure + frequency + measurement/simulation + reconstruction distance)
- Step 4. Structure & frequency: The first image (a) is an SIW coupler (not filter), and (b) its S-parameters are measured at 2 GHz (not 1.84 GHz, which is the second image’s frequency).
- Step 5. Measurement/simulation source: (c)/(g) = HFSS-simulated field: (c) = no ground plane, (g) = with finite ground plane; (e)/(i) = measured field: (e) = single probe, (i) = probe array;
- Step 6. Reconstruction distance: All reconstructed fields (d)/(f)/(h)/(j) are 0.5 mm from the coupler; measurement plane distance = 4 mm (not reconstruction distance).

- Step 7. Eliminate options with first image errors: Option 1 (a=filter, second image (a)=1 for metal, (b)=|𝐻𝛾|): Structure error + material assignment error + field component error. Option 2 (second image (a)=0 for dielectric, (c)=|𝐻𝑥|): Material assignment error + field component error. Option 4 (i=1.84 GHz, f=HFSS with ground plane, (a)=1 for partial metal): Frequency error + reconstruction source error + material assignment error. Option 5 (g=no ground plane, j=4 mm reconstruction, (a)=0.5 for dielectric): Simulation source error + reconstruction distance error + material assignment error. Option 6 (c=single probe-measured, (a)=1 for metal,

(c)=|𝐻𝑥|): Field source error + material assignment error + field component error. Option 7 (b=filter S-parameters, 1.84 GHz, (a)=0 for partial dielectric): Structure/frequency error +

material assignment error. Option 8 (d=probe array data, i=HFSS-simulated): Reconstruction source error + field source error. Option 9 (e=probe array-measured, 1.84 GHz, a=filter, (a)=0 for dielectric, (b)=|𝐻𝛾|): Measurement method error + frequency/structure error + material assignment/field component error. Option 10 (d=4 mm reconstruction, (a)=1 for partial metal, (c)=|𝐻𝑥|): Reconstruction distance error + material assignment error + field component error.

- Step 8. Extract core features of the second image (material assignment + field components).
- Step 9. Material assignment rule: (a) 0 = fully metal-covered, 1 = fully dielectric-covered, 0-1

= partially metal-covered (not reverse or arbitrary values).

- Step 10. Field components: (b) = |𝐻𝑥| variation, (c) = |𝐻𝛾| variation (not mixed).
- Step 11. Verify remaining option 3: First image part: "First image (e) is single probe-measured magnetic field (2 GHz)" → matches (e)=single probe, 2 GHz; "First image (d) uses HFSS data without ground plane for reconstruction" → (d) is reconstructed from (c)=HFSS no ground plane, correct. Second image part: "Second image (a) assigns 0 to fully metal areas" → matches

material rule; "Second image (b) shows |𝐻𝑥| variation" → matches (b)=|𝐻𝑥|, correct. Confirm option 3 is completely correct. All parts of option 3 align with the first image’s structure/frequency/field source/reconstruction rule and the second image’s material assignment/field component definition, with no contradictions.

Answer

- C

Example of Experimental Reasoning in Life Images

[Figure 668]

[Figure 669]

[Figure 672]

###### Question

According to the first image, if one wants to inhibit tumor development by targeting non-tumor cells within the body, which cells should the monoclonal antibody be made against? Using which method from the second image to deliver the antibody can achieve a inhibition of tumor development from a deeper level? Which type in the third image does this method belong to?Please choose from the given options:

###### Options

- A. Siglec-10;A;A
- B. Siglec-10;A;B
- C. Siglec-10;B;A
- D. Siglec-10;B;B
- E. Siglec-10;C;A
- F. CD24;A;A
- G. CD24;A;B
- H. CD24;B;A
- I. CD24;B;B
- J. CD24;C;A

Steps

- Step 1.

[Figure 675]

- Step 2. The proteins identified in the image that can serve as targets are mainly Siglec-10 and

CD24.

- Step 3. The topic requires starting from non-tumor cells, so Siglec-10 was chosen.
- Step 4.

[Figure 676]

- Step 5. Identify the three main strategies for NP-mediated CD24-Siglec10 axis-targeted therapy

- shown in the figure.
- Step 6. Among them, strategies A and B both use antibodies to directly block signal transduction on the cell surface, whereas strategy C uses siRNA to inhibit the expression of the target protein

at the nucleic acid level.

- Step 7. Strategy C is a deeper approach to suppress tumor development.
- Step 8.

[Figure 679]

- Step 9. Identifying two modes of nanoparticle-based drug delivery systems in the image.
- Step 10. The surface of the nanomaterials delivering siRNA does not carry antibodies and is passively targeted. Answer

- E

Example of Experimental Reasoning in Material Images

[Figure 682]

Question Images are Li-ion probability densities in Li-ion conductors. Li-ion probability densities are colored red. Which material does represent the best Li-ion conductivity? Options

- A. Li10GeP2S12
- B. Li7P3S11
- C. Li2S
- D. 𝛾-Li3PS4
- E. Li4GeS4
- F. Li3.25Ge0.25P0.75S4
- G. Li2S-P2S5
- H. Li10SnP2S12
- I. Li10SiP2S12
- J. Li6PS5Cl

Steps

- Step 1.

[Figure 685]

- Step 2. Find the Li-ion probability densities of materials in the figure.
- Step 3. Determine the largest region of the Li-ion probability densities. The answer is Li10GeP2S12. Answer A

Example of Experimental Reasoning in Neuroscience Images

[Figure 688]

[Figure 689]

###### Question

Please answer based on the first image: How many peaks exceeding 20 appeared in the first 60 timesteps of the Large initialization for Signal 2 response in each of the two examples?Based on the second image, after training, does Branch 1 with a small initialization increase (+) or decrease (-), and does Branch 2 with a large initialization increase (+) or decrease (-)?

Options

- A. 1,3;+-

- B. 0,0;++
- C. 1,2;–
- D. 1,1;++
- E. 1,3;-+
- F. 2,1;–
- G. 2,2;-+
- H. 3,1;-+
- I. 3,2;+-
- J. 0,3;-+

Steps

- Step 1.

[Figure 692]

- Step 2. Realize: Signal 2 response is blue line.
- Step 3. Define the counting range: Large initialization, Neuron state > 20,Timestep < 60, in each of the two examples.
- Step 4. Find out that there is 1 in example1 and 3 in example2. Answer: 1,3.
- Step 5.

[Figure 695]

- Step 6. Branch 1 small init: the KDE line and histogram show a shift. Before training, Branch 1 small init was lower around 0-0.2, after training, it’s higher around 0.8-1.0, so increase (+)
- Step 7. Branch 2 large init: before training, it was a peak around 1.0, after training, the density decreases there, so decrease (-).
- Step 8. Conclude:1,3;+Answer A

###### A.4. Supplementary Evaluation Results

###### Figure 31 | Scientific Deep Research Across Subjects: Combined subject-wise performance of LLMs and agents on deep research tasks.

Astronomy

Physics DeepSeek-V3.2

Chemistry

60

DeepSeek-R1

Intern-S1

Intern-S1-mini

50

Kimi-k2

Qwen3-VL-235B-A22B

40

Qwen3-235B-A22B

Qwen3-Max

Neuroscience

Earth

Qwen3-8B

30

Llama-4-Scout

GPT-4o

- GPT-4.1

- GPT-5

20

- GPT-5.1

- GPT-5.2-Pro

- o3

- o4-mini

Math

Energy

Gemini-2.5-Flash

- Gemini-2.5-Pro

- Gemini-3-Pro

Claude-Opus-4.1

Claude-Sonnet-4.5

- Grok-3

- Grok-4

Material

Information

Life

###### Figure 32 | Idea Generation Across Subjects: Subject-wise scores for idea generation.

Life-SS

60

DeepSeek-V3.2

DeepSeek-R1

50

Intern-S1

Intern-S1-mini

40

Kimi-k2

Physics-PA

Material-SS

Qwen3-VL-235B-A22B

30

Qwen3-235B-A22B

Qwen3-Max

20

Qwen3-8B

Llama-4-Scout

10

GPT-4o

- GPT-4.1

- GPT-5

0

- GPT-5.1

- GPT-5.2-Pro

- o3

- o4-mini

Gemini-2.5-Flash

- Gemini-2.5-Pro

- Gemini-3-Pro

Material-PA

Physics-SS

Claude-Opus-4.1

Claude-Sonnet-4.5

- Grok-3

- Grok-4

Life-PA

###### Figure 33 | Wet Experiment Across Subjects: Subject-wise Action Sequence Similarity (SS) and Parameter Accuracy (PA) performance in wet experiments.

Model Properties Micro-experiments Macro-experiments Data DeepSeek-V3.2 6.62 21.57 15.38 9.80 DeepSeek-R1 10.61 23.47 15.38 10.00 Intern-S1 7.14 24.64 20.00 12.12 Intern-S1-mini 5.88 19.18 17.39 5.41 Kimi-k2 8.09 20.21 20.00 10.00 Qwen3-VL-235B-A22B 7.30 19.19 12.50 10.20 Qwen3-235B-A22B 11.94 23.75 11.54 6.12 Qwen3-Max 7.00 30.00 0.00 13.79 Qwen3-8B 5.84 14.42 3.85 3.92 Llama-4-Scout 5.11 14.42 3.85 3.92

- GPT-4o 5.84 12.50 7.69 3.92

- GPT-4.1 7.30 17.31 15.38 7.84

GPT-5 10.22 21.15 26.92 5.88

- GPT-5.1 8.03 18.27 15.38 5.88 GPT-5.2-Pro 10.22 23.08 23.08 11.76

- o3 10.95 17.31 19.23 5.88
- o4-mini 8.76 18.27 11.54 7.84 Gemini-2.5-Flash 9.49 16.35 11.54 1.96 Gemini-2.5-Pro 11.68 23.08 15.38 7.84 Gemini-3-Pro 15.00 26.14 22.73 10.87 Claude-Opus-4.1 8.82 20.19 15.38 7.84 Claude-Sonnet-4.5 8.03 23.08 15.38 9.80 Grok-3 9.49 20.19 11.54 11.76 Grok-4 10.37 21.65 15.38 4.00

###### Table 10 | Deep Research Task Metrics (LLMs): Category-wise scores across Properties, Micro/MacroExperiments, and Data. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

Agent Properties Micro-experiments Macro-experiments Data SmolAgents(GPT-4.1) 13.87 16.35 26.92 5.88 SmolAgents(Gemini-2.5-Flash) 12.41 24.04 26.92 11.76 Owl(GPT-4.1) 6.57 18.27 19.23 9.80 Owl(Gemini-2.5-Flash) 6.61 14.29 9.52 8.33 WebThinker 13.87 18.27 26.92 3.92 XMaster 13.14 17.31 19.23 5.88 InternAgent 13.24 24.04 26.92 9.80 OpenAI Deep Research(o3) 16.06 14.42 11.54 9.80 OpenAI Deep Research(o4-mini) 14.60 22.12 19.23 11.76 Grok-Search(Grok-4) 14.18 22.73 19.23 11.76 Kimi-Search(Kimi-k2) 9.49 22.92 11.54 14.00 Doubao-Search(Seed-1-6) 7.35 16.50 0.00 3.92 Perplexity(Sonar-Pro) 6.57 21.15 19.23 3.92

###### Table 11 | Deep Research Task Metrics (Agents): Category-wise scores across Properties, Micro/Macro-Experiments, and Data. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

###### Model Numerical Calculation Statistical Analysis Simulation Metric Calculation Data Processing Predictive Modeling

DeepSeek-V3.2 19.30 19.05 26.32 35.71 42.86 27.27 DeepSeek-R1 31.76 23.81 26.32 39.29 47.62 45.45 Intern-S1 25.61 28.57 26.32 39.29 42.86 27.27 Intern-S1-mini 14.62 19.05 15.79 25.00 28.57 9.09 Kimi-k2 26.90 23.81 31.58 35.71 52.38 18.18 Qwen3-VL-235B-A22B 25.15 23.81 26.32 39.29 47.62 27.27 Qwen3-235B-A22B 25.29 28.57 31.58 35.71 47.62 27.27 Qwen3-Max 29.24 38.10 31.58 39.29 47.62 45.45 Qwen3-8B 18.71 14.29 15.79 25.00 23.81 0.00 Llama-4-Scout 20.59 19.05 15.79 21.43 23.81 18.18 GPT-4o 25.15 23.81 26.32 32.14 38.10 27.27 GPT-4.1 32.75 38.10 26.32 39.29 47.62 27.27 GPT-5 25.73 28.57 31.58 39.29 52.38 27.27 GPT-5.1 29.24 23.81 26.32 42.86 42.86 27.27 GPT-5.2-Pro 26.90 19.05 26.32 32.14 38.10 36.36 o3 28.65 42.86 26.32 42.86 38.10 27.27 o4-mini 35.09 28.57 26.32 39.29 52.38 36.36 Gemini-2.5-Flash 16.96 23.81 21.05 32.14 38.10 18.18 Gemini-2.5-Pro 19.30 23.81 21.05 21.43 42.86 36.36 Gemini-3-Pro 33.53 33.33 35.29 46.43 50.00 45.45 Claude-Opus-4.1 30.99 28.57 31.58 53.57 47.62 36.36 Claude-Sonnet-4.5 33.33 38.10 26.32 42.86 47.62 45.45 Grok-3 22.81 33.33 31.58 35.71 47.62 18.18 Grok-4 32.12 19.05 31.58 40.74 42.86 54.55

- Table 12 | Dry Experiment Function Categories: Completion scores across six function types. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

Model Signal Perception Attribute Understanding Comparative Reasoning Causal Reasoning

Intern-S1 39.29 21.88 28.57 37.50 Intern-S1-mini 17.86 10.94 18.29 20.83 Qwen3-VL-235B-A22B 32.14 26.56 32.00 41.67 Qwen3-VL-Max 50.00 34.38 36.57 41.67 Qwen3-VL-8B 21.43 21.88 23.43 29.17 Llama-4-Scout 28.57 17.19 28.57 25.00

- GPT-4o 39.29 26.56 33.71 29.17

- GPT-4.1 46.43 40.62 34.29 54.10

GPT-5 53.57 32.81 37.71 37.50

- GPT-5.1 21.43 25.00 36.57 54.17 GPT-5.2-Pro 53.57 39.06 38.29 29.17

- o3 35.71 26.56 33.14 41.67
- o4-mini 39.29 35.94 30.29 41.67 Gemini-2.5-Flash 35.71 37.50 30.29 54.17 Gemini-2.5-Pro 50.00 42.19 38.29 50.00 Gemini-3-Pro 50.00 40.62 42.86 29.17 Claude-Opus-4.1 53.57 35.94 34.86 58.33 Claude-Sonnet-4.5 35.71 35.94 38.86 37.50 Grok-4 42.86 26.56 28.00 41.67

- Table 13 | Experimental Reasoning by Type (Multi-choice Accuracy): Scores across signal, attribute, comparative, and causal reasoning. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

###### Model Astronomy Chemistry Earth Energy Information Life Material Math Neuroscience Physics

DeepSeek-V3.2 11.76 10.00 20.75 0.00 10.47 0.00 7.89 44.00 0.00 3.12 DeepSeek-R1 6.25 9.09 24.00 0.00 16.67 0.00 7.89 52.00 4.17 6.67 Intern-S1 0.00 20.00 22.45 0.00 12.50 8.00 0.00 47.62 0.00 0.00 Intern-S1-mini 0.00 9.09 23.26 0.00 7.14 6.25 7.14 61.54 0.00 0.00 Kimi-k2 5.88 10.00 27.08 0.00 5.26 9.30 15.79 43.48 0.00 0.00 Qwen3-VL-235B-A22B 5.88 10.00 19.61 0.00 16.67 9.41 5.26 40.00 0.00 6.25 Qwen3-235B-A22B 5.88 20.00 20.83 0.00 16.67 13.10 10.53 77.78 0.00 9.38 Qwen3-Max 11.11 0.00 33.33 0.00 11.11 16.28 7.89 44.00 4.17 3.12 Qwen3-8B 11.76 0.00 11.11 0.00 10.00 5.75 7.89 32.00 0.00 0.00 Llama-4-Scout 11.76 9.09 9.26 0.00 10.00 6.90 5.26 20.00 4.17 3.12 GPT-4o 5.88 9.09 7.41 0.00 10.00 4.60 15.79 24.00 4.17 0.00 GPT-4.1 23.53 9.09 12.96 0.00 5.00 9.20 5.26 44.00 8.33 0.00 GPT-5 5.88 9.09 27.78 0.00 10.00 9.20 13.16 52.00 0.00 3.12 GPT-5.1 17.65 9.09 18.52 10.00 5.00 9.20 2.63 36.00 8.33 3.12 GPT-5.2-Pro 11.76 9.09 25.93 0.00 20.00 13.79 10.53 48.00 0.00 3.12 o3 5.88 18.18 22.22 0.00 10.00 9.20 7.89 44.00 4.17 3.12 o4-mini 5.88 18.18 16.67 0.00 0.00 9.20 13.16 48.00 0.00 3.12 Gemini-2.5-Flash 5.88 9.09 14.81 0.00 10.00 8.05 5.26 40.00 4.17 6.25 Gemini-2.5-Pro 17.65 9.09 18.52 10.00 10.00 12.64 10.53 52.00 4.17 6.25 Gemini-3-Pro 12.50 14.29 27.66 0.00 35.29 12.00 17.86 50.00 4.76 6.25 Claude-Opus-4.1 11.76 9.09 22.22 0.00 10.00 9.30 7.89 40.00 4.17 6.25 Claude-Sonnet-4.5 17.65 9.09 20.37 10.00 15.00 11.49 5.26 36.00 8.33 6.25 Grok-3 5.88 9.09 22.22 10.00 5.00 11.49 13.16 40.00 4.17 3.12 Grok-4 5.88 9.09 18.37 10.00 16.67 10.47 10.53 45.83 0.00 6.45

- Table 14 | Deep Research Across Subjects (LLMs): Subject-wise scores across ten scientific domains. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

Agent Astronomy Chemistry Earth Energy Information Life Material Math Neuroscience Physics

SmolAgents(GPT-4.1) 29.41 9.09 27.78 0.00 10.00 9.20 15.79 28.00 4.17 3.12 SmolAgents(Gemini-2.5-Flash) 23.53 9.09 33.33 0.00 25.00 11.49 10.53 44.00 4.17 3.12 Owl(GPT-4.1) 23.53 9.09 18.52 0.00 10.00 6.90 7.89 44.00 4.17 0.00 Owl(Gemini-2.5-Flash) 6.25 10.00 15.79 0.00 5.56 11.54 0.00 41.67 8.33 0.00 WebThinker 5.88 9.09 27.78 0.00 15.00 6.90 23.68 36.00 4.17 6.25 XMaster 11.76 9.09 25.93 0.00 15.00 6.90 10.53 44.00 0.00 9.38 InternAgent 29.41 9.09 26.42 10.00 25.00 11.49 10.53 52.00 0.00 6.25 OpenAI Deep Research(o3) 11.76 9.09 20.37 10.00 15.00 12.64 13.16 20.00 16.67 6.25 OpenAI Deep Research(o4-mini) 5.88 18.18 24.07 10.00 25.00 12.64 21.05 40.00 12.50 0.00 Grok-Search(Grok-4) 17.65 20.00 26.92 0.00 15.79 13.95 7.89 75.00 4.17 9.38 Kimi-Search(Kimi-k2) 11.76 10.00 22.45 0.00 15.00 13.95 10.53 45.83 0.00 3.12 Doubao-Search(Seed-1-6) 17.65 9.09 9.43 0.00 15.00 4.65 8.11 32.00 0.00 6.25 Perplexity(Sonar-Pro) 11.76 9.09 16.67 0.00 15.00 6.90 15.79 40.00 4.17 0.00

- Table 15 | Deep Research Across Subjects (Agents): Subject-wise scores across ten scientific domains. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

###### Model Astronomy Chemistry Earth Energy Information Life Material Math Neuroscience Physics

DeepSeek-V3.2 38.38 24.49 35.85 34.24 32.09 32.11 31.77 29.46 29.53 34.20 DeepSeek-R1 35.76 31.63 37.89 37.73 36.26 35.96 36.47 35.88 33.88 37.21 Intern-S1 37.53 28.20 36.22 36.07 33.30 34.38 32.15 27.00 30.07 33.46 Intern-S1-mini 36.49 24.77 35.00 33.68 34.21 32.80 26.96 29.16 31.91 34.02 Kimi-k2 44.80 36.44 42.99 44.80 37.48 39.78 44.86 36.58 38.43 43.59 Qwen3-VL-235B-A22B 36.00 30.06 37.90 40.09 31.62 35.28 35.59 30.56 32.18 35.31 Qwen3-235B-A22B 37.38 31.02 36.78 41.24 35.25 35.98 35.34 31.06 32.46 36.52 Qwen3-Max 39.80 30.28 37.74 40.56 33.12 35.42 34.98 30.12 30.31 34.54 Qwen3-8B 34.25 22.91 33.78 30.72 30.35 30.26 29.80 27.42 26.20 32.05 Llama-4-Scout 28.65 22.50 27.79 26.10 30.47 25.62 26.14 25.26 24.94 29.65 GPT-4o 31.27 24.79 30.50 31.70 29.19 26.17 26.83 25.86 25.72 30.77 GPT-4.1 32.20 26.40 33.79 32.64 31.15 29.28 32.30 27.99 25.37 32.78 GPT-5 52.37 54.12 56.01 64.53 48.58 50.25 54.82 50.99 47.46 56.55 GPT-5.1 44.34 46.56 44.50 53.35 38.24 39.80 41.00 36.49 38.61 43.61 GPT-5.2-Pro 57.65 57.06 60.24 65.97 46.78 52.41 56.25 55.04 47.62 57.21 o3 42.57 38.83 44.58 50.85 38.35 40.77 45.42 40.36 38.43 44.50 o4-mini 37.74 29.78 39.14 38.08 34.79 36.63 37.86 36.86 32.42 38.78 Gemini-2.5-Flash 37.32 27.61 36.42 35.33 32.59 33.06 33.34 27.42 29.51 34.93 Gemini-2.5-Pro 38.64 27.22 37.10 46.00 34.39 35.12 36.93 31.00 31.12 36.28 Gemini-3-Pro 39.51 35.97 37.17 40.49 34.14 35.35 35.49 30.03 32.14 35.18 Claude-Opus-4.1 39.85 28.89 38.19 38.83 35.19 36.85 38.39 35.69 33.66 37.44 Claude-Sonnet-4.5 42.11 34.89 42.38 44.20 35.24 37.31 38.14 34.44 32.13 40.90 Grok-3 29.66 23.40 31.10 25.66 31.04 30.11 27.29 26.01 26.43 33.26 Grok-4 33.75 25.48 33.78 35.22 30.44 30.96 30.30 27.54 27.58 33.61

- Table 16 | Idea Generation Across Subjects: Subject-wise scores. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

Model Astronomy Chemistry Earth Energy Information Life Material Math Neuroscience Physics

DeepSeek-V3.2 31.25 0.00 20.83 10.00 14.29 27.50 44.44 29.41 16.67 17.24 DeepSeek-R1 37.50 20.00 33.33 10.00 35.71 33.75 55.56 29.41 29.17 24.14 Intern-S1 37.50 0.00 25.00 10.00 28.57 33.75 48.15 18.18 16.67 24.14 Intern-S1-mini 12.50 0.00 18.75 10.00 14.29 21.25 33.33 0.00 4.17 17.24 Kimi-k2 43.75 0.00 22.92 20.00 21.43 33.75 44.44 16.67 20.83 34.48 Qwen3-VL-235B-A22B 37.50 0.00 29.17 10.00 14.29 35.00 40.74 16.67 20.83 24.14 Qwen3-235B-A22B 31.25 0.00 25.00 30.00 14.29 35.00 44.44 17.65 20.83 27.59 Qwen3-Max 50.00 0.00 31.25 30.00 28.57 37.50 48.15 22.22 25.00 24.14 Qwen3-8B 25.00 0.00 18.75 10.00 7.14 20.00 33.33 5.56 12.50 20.69 Llama-4-Scout 18.75 0.00 18.75 10.00 14.29 25.00 33.33 17.65 12.50 17.24

- GPT-4o 37.50 0.00 27.08 10.00 14.29 35.00 51.85 22.22 20.83 20.69

- GPT-4.1 43.75 20.00 33.33 40.00 28.57 33.75 48.15 27.78 29.17 34.48

GPT-5 37.50 0.00 27.08 40.00 35.71 31.25 40.74 22.22 20.83 27.59

- GPT-5.1 31.25 0.00 27.08 30.00 28.57 38.75 44.44 22.22 12.50 31.03 GPT-5.2-Pro 43.75 0.00 22.92 10.00 21.43 33.75 44.44 27.78 20.83 17.24

- o3 37.50 0.00 33.33 10.00 28.57 35.00 51.85 22.22 20.83 27.59
- o4-mini 37.50 0.00 33.33 20.00 28.57 40.00 51.85 22.22 37.50 34.48 Gemini-2.5-Flash 18.75 0.00 18.75 10.00 14.29 23.75 37.04 27.78 16.67 13.79 Gemini-2.5-Pro 25.00 0.00 18.75 0.00 21.43 25.00 33.33 22.22 16.67 27.59 Gemini-3-Pro 37.50 0.00 32.61 30.00 38.46 38.46 55.56 37.50 34.78 28.57 Claude-Opus-4.1 43.75 20.00 33.33 40.00 28.57 33.75 48.15 27.78 29.17 34.48 Claude-Sonnet-4.5 43.75 20.00 35.42 30.00 21.43 41.25 51.85 27.78 25.00 27.59 grok-3 31.25 0.00 29.17 20.00 14.29 32.50 40.74 11.11 20.83 24.14 Grok-4 37.50 20.00 27.66 20.00 30.77 37.97 51.85 43.75 25.00 22.22

- Table 17 | Dry Experiment Across Subjects: Subject-wise scores. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

###### Model Life-SS Material-SS Physics-SS Life-PA Material-PA Physics-PA

DeepSeek-V3.2 15.47 20.20 16.67 21.48 26.87 22.50 DeepSeek-R1 10.00 21.83 16.67 23.14 26.32 39.67 Intern-S1 24.86 27.23 33.33 23.62 34.87 50.90 Intern-S1-mini 14.15 4.67 0.00 18.02 16.04 9.58 Kimi-k2 18.10 28.04 0.00 25.13 37.42 18.61 Qwen3-VL-235B-A22B 17.78 30.25 16.67 28.66 43.11 46.68 Qwen3-235B-A22B 17.11 30.80 0.00 22.87 37.71 33.18 Qwen3-Max 17.37 41.11 33.33 24.44 45.67 56.70 Qwen3-8B 4.99 15.54 0.00 5.81 15.49 6.25 Llama-4-Scout 15.72 18.75 16.67 20.53 32.86 17.78 GPT-4o 20.79 29.10 32.38 31.58 41.06 41.41 GPT-4.1 32.13 33.02 33.33 33.11 45.06 54.47 GPT-5 7.81 11.72 33.76 19.31 21.50 23.18 GPT-5.1 12.38 21.44 29.30 24.24 28.00 40.14 GPT-5.2-Pro 18.50 8.81 19.44 23.05 19.65 17.86 o3 27.43 22.79 44.86 30.63 32.87 48.92 o4-mini 31.46 24.01 16.67 25.76 35.78 32.70 Gemini-2.5-Flash 5.31 23.44 15.71 14.73 28.09 32.03 Gemini-2.5-Pro 16.90 21.02 12.06 24.52 27.28 23.03 Gemini-3-Pro 20.00 34.88 33.33 32.21 41.07 36.12 Claude-Opus-4.1 16.65 25.74 29.21 20.63 33.45 43.90 Claude-Sonnet-4.5 31.75 25.83 16.67 28.62 33.78 46.97 grok-3 28.97 41.93 33.33 32.52 43.94 58.32 Grok-4 27.29 29.10 16.67 25.19 37.35 23.09

- Table 18 | Wet Experiment Across Subjects: Scores across Action Sequence Similarity (SS) and Parameter Accuracy (PA) categories. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

Model Astronomy Chemistry Earth Energy Information Life Material Neuroscience Physics

Intern-S1 47.06 27.27 27.78 40.00 25.00 29.41 26.67 33.33 16.00 Intern-S1-mini 23.53 27.27 18.52 30.00 10.00 18.82 8.89 12.50 16.00 Qwen3-VL-235B-A22B 58.82 36.36 31.48 50.00 15.00 29.41 31.11 33.33 24.00 Qwen3-VL-Max 52.94 36.36 31.48 50.00 35.00 41.18 40.00 37.50 24.00 Qwen3-VL-8B 29.41 36.36 24.07 60.00 20.00 25.88 13.33 16.67 16.00 Llama-4-Scout 41.18 27.27 27.78 30.00 30.00 23.53 31.11 20.83 0.80 GPT-4o 41.18 54.55 37.04 60.00 20.00 29.41 31.11 20.83 28.00 GPT-4.1 35.29 36.36 37.04 60.00 45.00 42.35 37.78 33.33 24.00 GPT-5 70.59 36.36 37.04 30.00 50.00 37.65 33.33 41.67 20.00 GPT-5.1 47.06 45.45 33.33 40.00 35.00 31.76 42.22 16.67 28.00 GPT-5.2-Pro 52.94 18.18 31.48 30.00 40.00 42.35 44.74 50.00 31.25 o3 58.82 45.45 29.63 50.00 35.00 29.41 24.44 50.00 16.00 o4-mini 64.71 45.45 31.48 30.00 25.00 34.12 26.67 33.33 28.00 Gemini-2.5-Flash 52.94 27.27 33.33 40.00 45.00 36.47 24.44 37.50 24.00 Gemini-2.5-Pro 52.94 36.36 38.89 30.00 50.00 38.82 37.78 58.33 36.00 Gemini-3-Pro 47.06 45.45 35.19 50.00 45.00 40.00 48.89 62.50 20.00 Claude-Opus-4.1 58.82 45.45 25.93 60.00 40.00 37.65 33.33 50.00 44.00 Claude-Sonnet-4.5 52.94 36.36 37.04 20.00 40.00 35.29 42.22 29.17 44.00 Grok-4 52.94 18.18 33.33 40.00 35.00 30.59 26.67 20.83 20.00

- Table 19 | Experimental Reasoning Across Subjects (Multi-choice Accuracy): Subject-wise scores across 10 scientific disciplines. Note: Because different subjects have different characteristics, the number of questions in each category is not the same (Figure 9). Therefore, the overall performance of the model cannot be obtained by directly averaging the values in the table.

