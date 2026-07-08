# arXiv:2506.03828v3[cs.AI]2Apr2026

## AssetOpsBench: A Real-World Evaluation Benchmark for AI-Driven Task Automation in Industrial Asset Management

Dhaval Patel∗

pateldha@us.ibm.com IBM Yorktown Heights, NY, USA

Shuxin Lin∗

shuxin.lin@ibm.com IBM Yorktown Heights, NY, USA

James Rayfield∗

jtray@us.ibm.com IBM Yorktown Heights, NY, USA

Nianjun Zhou∗

jzhou@us.ibm.com IBM Yorktown Heights, NY, USA

Chathurangi Shyalika

jayakodc@email.sc.edu University of South Carolina USA

Suryanarayana R Yarrabothula

surya.427@gmail.com Steel Authority of India Bokaro, India

Roman Vaculin

vaculin@us.ibm.com IBM Yorktown Heights, NY, USA

Natalia Martinez

Natalia.Martinez.Gil@ibm.com IBM Yorktown Heights, NY, USA

Fearghal O’Donncha

feardonn@ie.ibm.com IBM Dublin, Ireland

Jayant Kalagnanam

jayant@us.ibm.com IBM Research Yorktown Heights, NY, USA

### Abstract

AI for Industrial Asset Lifecycle Management aims to automate complex operational workflows, such as condition monitoring and maintenance scheduling, to minimize system downtime. While traditional AI/ML approaches solve narrow tasks in isolation, Large Language Model (LLM) agents offer a next-generation opportunity for end-to-end automation. In this paper, we introduce AssetOpsBench, a unified framework for orchestrating and evaluating domain-specific agents for Industry 4.0. AssetOpsBench provides a multimodal ecosystem comprising a catalog of four domain-specific agents, a curated dataset of 140+ human-authored natural-language queries grounded in real industrial scenarios, and a simulated, CouchDB-backed IoT environment. We introduce an automated evaluation framework that uses three key metrics to analyze architectural trade-offs between the Tool-As-Agent and Plan-Executor paradigms, along with a systematic procedure for the automated discovery of emerging failure modes. The practical relevance of AssetOpsBench is demonstrated by its broad community adoption, with 250+ users and over 500 agents submitted to our public benchmarking platform, supporting reproducible and scalable research for real-world industrial operations.

Dataset GitHub Competition Demo

### CCS Concepts

• Computing methodologies → Multi-agent systems.

### Keywords

AI Agent, Multi-Agent System, LLM-As-Judge, AI Competition

∗These authors contributed equally and are considered co-primary authors.

### 1 Introduction

Industrialassets,suchasdata center chillers [34]andwindfarms[32], are complex, multi-component systems that generate vast amounts of multimodal data, including time-series sensor readings, textual inspection and workorder records, operational logs, and images, throughout their lifecycle. The ability to monitor and interpret heterogeneous data from diverse sources, such as IoT SCADA [52] (Supervisory Control and Data Acquisition) sensors, operational KPIs, failure mode libraries, maintenance work orders, and technical manuals, is key to effective Asset Lifecycle Management [49]. However, subject matter experts such as maintenance engineers, site operators, and plant managers face considerable challenges in synthesizing insights from these disparate data streams to support timely and condition-aware decisions. As highlighted in Figure 1a, the scale, semantic diversity of assets, and application-specific contexts often make traditional monitoring and management systems inadequate.

To address these challenges, the research and industrial communities are increasingly turning to AI agents: autonomous and goal-driven systems capable of integrating data, reasoning over complex conditions, and triggering appropriate actions. AI agents are particularly promising in the context of Industry 4.0, where the confluence of real-time IoT telemetry (e.g., Oracle IoT [36], enterprise asset management (EAM) systems [51], and IBM Maximo [19]) and reliability engineering frameworks necessitates scalable and intelligent automation. These agents promise to support a wide range of industrial workflows, from anomaly detection to maintenance scheduling, by bridging the gap between raw sensor data, maintaiance report, work-order and business-level insights.

Despite significant advances in LLM-based agents and generalist models like ReAct [57], HuggingGPT [39], Chameleon [29],

[Figure 1]

[Figure 2]

(b) Distribution of open-sourced scenarios for benchmarking agents in a simulated environment.

(a) Complex Industrial Asset – Data Centers managing Chiller and Air Handling Units (AHUs)

#### Figure 1: Industrial Asset Management Context and Utterance Scenario Distribution Used in AssetOpsBench.

Magentic-One [10], and enterprise-ready computer use agents [31], a substantial gap remains in adapting these innovations for physical, sensor-driven industrial operations. While recent applicationspecific benchmarks such as ITBench [23], SWE-bench [4],𝜏−bench [56] and its multi-agent extensions [11], LangChain’s customer support benchmarks [44], TheAgentCompany [55], and CRMArenaPro [15] offer rigorous evaluation in digital knowledge-work or IT domains, they do not address the unique challenges of industrial settings. These environments demand handling data modality diversity across time-series and text, managing complex business objects like asset hierarchies and failure modes, and facilitating collaboration across specialized operational personas.

Motivated by real-world deployment experience and the inherent application diversity of industrial environments (Figure 1a), we propose a multi-agent architecture that maps heterogeneous workflows to modular agents. In this architecture, specialized entities, such as an IoT agent for sensor telemetry and an FMSR agent for fault history, must intelligently collaborate to resolve complex queries like “Why is the chiller efficiency dropping?”, which require a deep integration of physical reasoning and operational semantics. To formalize the evaluation of such systems and bridge the gap between synthetic benchmarks and industrial reality, we introduce a unified framework AssetOpsBench.

The key innovation and contributions in AssetOpsBench are structured around three synergistic pillars:

- ⋄ Pillar I: The AssetOps Ecosystem. We provide a high fidelity industrial sandbox featuring (1) a catalog of specialized agents (IoT, FMSR, TSFM, WO) with domain-specific tools; (2) 141 intent-aware scenarios grounded in real data center assets (4 Chillers and 2 AHUs); and (3) a dockerized environment with CouchDB access for reproducible, end-to-end multi-agent testing.
- ⋄ Pillar II: Empirical & Diagnostic Framework. We establish a rigorous methodology for agent evaluation, including a comparative architectural study (Agent-As-Tool vs. Plan-Execute) and a hybrid scoring system. We introduce an automated failure discovery procedure that identifies non-obvious reasoning and recovery bottlenecks.
- ⋄ Pillar III: Community & Open Source. We operationalized AssetOpsBench as a live competition on Codabench,

yielding 250+ users, 500+ submissions and 881+ trajectories. We release all containerized interfaces and starter templates to ensure the benchmark serves as a diagnostic utility for the broader research community.

Finally, we experimented with an additional closed-source 162 scenarios to demonstrate generality, spanning 10 asset classes, 53 failure modes, and 20 sensors. These include 42 live-productiondeployment scenarios (>90% correctness verified by a domain expert), 17 hydraulic system, 15 metro train, and 88 failure-mode scenarios encompassing diverse asset–failure–sensor relationships.

### 2 Related Work

Generalist Agents. In this section, we review prior work on agent architectures and benchmarks relevant to AssetOpsBench, focusing on generalist agents that orchestrate sub-agents to solve complex tasks, as seen in web-based systems like Magentic [10] and CUGA [31], multimodal frameworks like GEA [43], and software engineering platforms such as HyperAgent [18], ChatDev [37], and MetaGPT [13]. These systems typically rely on fixed sub-agents (e.g., browsers, editors) and hard-coded reasoning paradigms like ReAct or plan-execute. Consequently, they often lack the flexibility to adapt to novel tasks or support alternative coordination strategies like AOP [28] and Prospector [25].

Domain and Application-Specific Agents. Specialized benchmarks like MLEBench [4], MLAgentBench [16], and MLGym [33] evaluate agents on scientific discovery and machine learning workflows. Similarly, application-focused frameworks like ITBench [23] and AIOpsLab [6] replicate site reliability engineering and system auditing. However, these benchmarks often prioritize tabular or image data, lacking the integrated support for the hybrid temporal and textual modalities critical for physical asset monitoring. We have given a detailed comparison in Appendix Table 9. Furthermore, they frequently lack the composability needed to evaluate agents across diverse, cross-modal industrial environments.

Action-OrientedModels.Performanceisincreasinglyenhanced

via fine-tuned models or Large Action Models (LAMs) designed for grounded execution, including TaskBench [40], xLAM [59], AgentGen [14], AgentBank [42], AgentRM [53], FireAct [5], and ActionStudio [58]. While effective in Windows-based [45], programming,

or web environments, these models have yet to demonstrate applicability to the complex reasoning required for industrial automation with hybrid agent compositions.

Open Challenges. Despite recent advances, several critical gaps remain in the industrial agentic AI landscape. We categorize these into three primary deficiencies:

- (1) Domain-Specific Data Scarcity: There is a profound lack of comprehensive benchmarks targeting industrial asset domains, particularly for tasks involving condition-based monitoring and work order planning. To quantify this gap, we conducted a systematic analysis of a catalog of 135 public industrial datasets [24]. Our findings reveal that:

- • Only 1/135 datasets include any work-order or operational context.

- • Only 39% (53/135) mention failure modes, typically limited to 1 or 2 modes.

- • 0% of these datasets support agentic applications or integrated sensor-history-to-action workflows.

- (2) Underrepresentation of Time-Series Modalities: While time-series data is the backbone of infrastructure monitoring, it remains underrepresented in existing agentic benchmarks, which favor text-heavy or web-navigation tasks.
- (3) Orchestration Complexity: Few systems support orchestration across heterogeneous agents, such as blending text, code, and simulation, nor do they offer modular reasoning strategies adaptable to high-stakes, multi-agent workflows.

Addressing these gaps is essential for advancing general-purpose agent intelligence in real-world industrial domains.

### 3 Intelligent Agent-Based Asset Operations

Industrial asset operations involve complex, heterogeneous workflows where experts must interpret multi-modal sensor data and make timely decisions. Interdependent tasks, such as root cause analysis, predictive maintenance, and work order bundling, require reasoning across historical telemetry, asset metadata, and operational constraints. This necessitates intelligent agents capable of decomposing high-level intent into structured, executable recommendations. A representative complex workflow is illustrated by the following user request:

|“Help configure an anomaly detection model to monitor power consumption of CUXP and trigger alerts when usage is projected to exceed 8 Watts above the maximum deviation observed over the past 30 days.”|
|---|

Such scenarios facilitate critical corrective actions, such as automated service request creation. The diversity of these tasks, spanningdatainterpretation,anomalyreasoning, and operational decisionmaking, underscores the need for a coordinated multi-agent framework designed for the rigors of industrial workflows.

To addressthesedemands,weoperationalize a multi-agent framework within a simulated Docker environment (Figure 2), centered on a global coordinator: the AssetOps Agent. While generalpurpose systems [10] often rely on generic sub-agents like websurfers or terminal handlers, our architecture utilizes specialized, domain-inspired modules: IoT for telemetry, FMSR for failure mode mapping, TSFM for foundation model-driven time-series analysis,

[Figure 3]

#### Figure 2: AssetOpsBench Simulated Environment.

and WO for work-order modeling. This containerized sandbox provides agents with secure, standardized API access to Industry 4.0 data, including FMEA logs, alerts, and telemetry, facilitating reproducible end-to-end execution. Furthermore, the environment natively supports diverse orchestration paradigms, such as AgentAs-Tool and Plan-Execute, enabling comparative research into how distinct reasoning strategies handle complex, real-world queries.

To ensure AssetOpsBench reflects operational reality, we move beyond the API-driven scenario generation typical of prior work [40, 56] in favor of a domain-authentic, intent-driven approach. Grounded in ISO 55000 [22] and ISO 14224 [21] standards, we established a four-stage taxonomy to guide the scenario generation process, mirroring the lifecycle of physical asset management:

- Phase I: Asset Configuration. Focuses on foundational activities such as retrieving Failure Mode and Effects Analysis (FMEA) documentation and defining performance KPIs.
- Phase II: Model Selection and Analysis. Agentsapplyanomaly detection models and leverage LLM-powered retrieval to surface critical historical failures from multi-modal records.
- Phase III: Monitoring and Execution. Mimics live operations, requiring agents to manage real-time telemetry, refine detection parameters, and enforce safety guardrails.
- Phase IV: Maintenance and Response. Focuses on actionable outputs, including automated work-order generation, system health summarization, and intervention.

By grounding scenario definitions in international ISO standards, we achieve a level of platform-agnosticism that allows findings to generalize across diverse industrial ecosystems, from Oracle to IBM Maximo [19, 36]. This shift from “synthesizing tasks around tools” to “aligning tools with industrial intent” represents a significant step toward evaluating agents for true operational reliability.

### 4 AssetOpsBench: Dataset, Scenarios, and Agents

AssetOpsBench is a comprehensive benchmarking ecosystem featuring 141 expert-curated scenarios and specialized AI agents integrated with a multi-source industrial dataset.

#### Table 1: Key Data Modalities: Overview of the multi-source foundation used for scenario construction

Data Source Example Fields Sensor Telemetry Assets: 6 Quantity: 2.3M pts

Chiller Return Temp. Chiller % Loaded Condenser Water Flow

Failure Location / Component Degradation Mechanism Degradation Influences

FMEA Records Assets: 3 Quantity: 53 entries

ISO Failure Code Event Log Timestamp Linked Anomaly / Alert

Work Orders Assets: 10+ Quantity: 4.2K records

### 4.1 Multi-Source Dataset

As detailed in Table 1, the benchmark incorporates over 2.3 million sensor data points across six primary assets, comprising four chillers and two Air Handling Units (AHUs). These streams are ingested from Building Management Systems (BMS) [50] and SkySpark [41], capturing critical time-series signals such as return temperatures and condenser flow rates. To provide a diagnostic "ground truth," we include structured failure models derived from Failure Mode and Effects Analysis (FMEA) records sourced from the Reliability Strategy Library [20]. These 53 entries provide deep insights into the physical mechanisms of failure, such as wear or erosion, and the influencing stressors, such as shock loading, that an agent must interpret to provide accurate recommendations.

Furthermore, AssetOpsBench integrates a longitudinal history of 4.2K work orders spanning 11 years, extracted from IBM Maximo systems. These records are enriched with ISO-standard failure codes and event timestamps, allowing agents to correlate historical maintenance actions with detected anomalies. Collectively, this foundation comprises nine distinct modalities, including sensors, work orders, alerts, FMEA, and expert-derived rules, to facilitate a comprehensive evaluation of multi-hop reasoning in high-stakes industrial environments. Refer to Appendix B for further details.

### 4.2 Scenario Design and Coverage

Each scenario in AssetOpsBench represents a structured operational query grounded in our ISO-aligned task taxonomy and multi-source datasets. We formalize each scenario as a 5-tuple:

𝑃 = ⟨id, type, text, category, characteristic form⟩

where id is a unique identifier and category specifies the functional requirement, such as knowledge retrieval or analytical reasoning. The text field contains the natural language query, which is specifically designed to reflect the intent patterns of industrial operators. Rather than referencing database schemas, these queries utilize physical operational terms, such as energy consumption or chiller performance, requiring the agent to map human intent to the underlying data modalities. The type denotes the operational domain, including specialized modules (IoT, FMSR, TSFM, WO) or end-toend tasks, while characteristic form defines the expected output, such as an executable API call or a multi-step action plan.

- Figure 3 illustrates Utterance 507, a representative case requir-

ing the agent to forecast future energy consumption. To resolve

[Figure 4]

#### Figure 3: An example of an expert-authored query.

this query, the agent must map high-level user intent to specific operational variables, such as power input, while navigating industrial data artifacts like high zero-value density. Crucially, the agent is expected to recognize that the chiller was non-operational during the specified window; it must then correctly conclude that data is insufficient for forecasting rather than generating a spurious result. This scenario demonstrates how our expert-designed tasks evaluate an agent’s ability to navigate the ambiguity of operator-centric language and physical state logic rather than merely assessing tool-calling syntax. Refer to Appendix C for further details.

Beyond individual tasks, the 11-year longitudinal history of work orders provides the temporal depth required for highlevel strategic synthesis. This scale enables the evaluation of longterm lifecycle strategies, such as determining if year-over-year increases in corrective maintenance justify capital replacement over continued repair. Scenarios IDs 442–446 specifically leverage this breadth to assess strategic oversight through trend analysis and probability forecasting across the asset’s operational lifespan.

Development Effort. To ensure operational authenticity, 141 scenarios were synthesized through an 18-month collaboration with reliability engineers to capture realistic fault signatures and cross-sensor interactions across assets like AHUs and chillers. As shown in Figure 1b, the benchmark comprises 99 single-agent utterances targeting specific domain modules and 42 multi-agent tasks necessitating coordinated reasoning and data exchange.

### 4.3 Agents and Implementation

AssetOpsBenchfeaturesfourspecializeddomainagents:IoT,TSFM,

WO, and FMSR. To navigate the complexity of industrial telemetry and metadata, these agents utilize a suite of over 15 distinct tools. As shown in Figure 4, the TSFM agent integrates a pretrained timeseries foundation model from Hugging Face for forecasting, while the FMSR agent utilizes an LLM-driven get_mapping function to bridge the semantic gap between physical failure modes and sensor tags. Methodologically, the TSFM, IoT, and FMSR agents employ

[Figure 5]

#### Figure 4: Representative tools for time-series and failuremode reasoning agent

the ReAct [57] paradigm, whereas the WO agent utilizes CodeReAct [46] to handle the structured logic required for maintenance record analysis.

To coordinate these heterogeneous modules, we implement a global AssetOps Agent that supports two primary orchestration paradigms:

- • Agent-As-Tool: Domain agents are registered as high-level tools within a supervisor agent. This supervisor uses ReAct to iteratively select and query specific agents, emulating the layered decision-making of hierarchical industrial organizations.
- • Plan-Execute: A specialized Planner and Reviewer decompose the user query into a Directed Acyclic Graph (DAG). An Orchestrator then executes the DAG, utilizing a shared memory module to transfer context between agents. This approach adapts the ReWoo [54] framework with an additional verification layer inspired by [28].

These components, comprising specialized tools, agents, and their orchestration, are implemented using open-source toolkits [27, 31, 35]. Our agent selection and coordination strategies follow modular architectural trends established in recent agentic benchmarking literature [30, 47]. An in-depth discussion of these domain-specific agents, including their tool schemas, workflow illustrations, and coordination mechanics, is provided in Appendix A.

### 5 Experiments and Leaderboard

To evaluate LLM capabilities for industrial automation within AssetOpsBench, we benchmark seven state-of-the-art models across the two orchestration paradigms from Section 4.3. Following recent frameworks [1, 27, 45, 48], we adopt a dual-evaluation strategy: a rubric-based assessment of reasoning quality and a referencebased mechanism to verify ground-truth alignment [3, 48, 56].

### 5.1 Evaluation Methodology

Rubric-based LLM-As-Judge Scoring. Each scenario is paired with a characteristic form (𝐶), which is a structured specification of expected outputs and procedural steps serving as the soft ground truth. We formalize the evaluation via a scoring function Φ(𝑄,𝑇,𝐶)

→ {𝑦1,𝑦2,𝑦3}, where 𝑄 is the query and𝑇 is the agent trajectory. The function yields three scalar scores𝑦𝑖 ∈ [0, 1] for (1) Task Completeness (𝑦1), (2) Data Retrieval Accuracy (𝑦2), and (3) Result Verification (𝑦3). We employ llama-4-maverick as the primary judge; human validation of this judge is provided in Section 5.4, with prompts detailed in Appendix D.1.

Reference-Based Scoring. To evaluate planning and decomposition rigor, we establish a structured ground-truth reference for each scenario comprising: (1) planning_steps for strategic intent, (2) execution_steps for tool-calling actions, and (3) causal dependencies via execution_links. Appendix Figure 15 illustrates a complete reference trace. We quantify the alignment between these benchmarks and the agent-generated reasoning traces or Directed Acyclic Graphs (DAGs) using the ROUGE metric to assess lexical and structural overlap [40, 56].

### 5.2 Experimental Setting

To quantify agent effectiveness, we adopt the Pass𝑘 metric [56]. Unlike Pass@k, which requires only one success in k attempts, Pass𝑘 estimates the probability of succeeding across all k trials. This stricter criterion aligns with industrial reliability standards where consistent behavior is mandatory and stochastic retries are operationally impractical [26, 56]. We report Pass1 by default, representing the success rate over single task instances.

Parameters. To ensure stable performance estimates for our rubric-based evaluation, the judge agent is executed five times per trajectory, and we report the average. All specialized LLM agents operate with a sampling temperature of 0 to prioritize deterministic reasoning, while the evaluation judge utilizes a temperature of 0.3 to allow for nuanced qualitative assessment.

Models. We include closed-source models (e.g., gpt-4.1), frontier open-source models (e.g., llama-4-maverick, llama-4-scout, mistral-large,llama-3-405b), andmedium-to-small open-source

models (e.g., llama-3-70b, granite-3-8b). We release a public leaderboard for AssetOpsBench and will continuously update it as new models and budget become available to our ecosystem.

### 5.3 AssetOpsBench Leaderboard

Rubric-based Performance Analysis. Figure 5 summarizes performance across three dimensions: task completion, data retrieval accuracy, and result verification. AssetOpsBench remains significantly challenging for current LLMs, with no model exceeding a 70% completion rate. Under the Agent-As-Tool setup (Figure 5a), gpt4.1 leads with the highest task completion (65%) and retrieval accuracy (77%). Among open-source candidates, llama-4-maverick is the most competitive, matching the leading models in reasoning stability.

In contrast, Plan-Execute paradigm (Figure 5b) exhibits a significant performance regression. The peak completion rate drops to 46%, suggesting that the “plan-ahead” pattern introduces cascading failure modes in industrial contexts. Most strikingly,gpt-4.1shows the sharpest decline (65%→38% completion), indicating that its proficiency in iterative tool use does not generalize to rigid planning interfaces. Conversely, mistral-large and llama-4-maverick emerge as the most robust for Plan-Execute workflows: mistrallarge achieves the highest retrieval (57%) and verification (48%)

[Figure 6]

[Figure 7]

(a) Agent-As-Tool Paradigm (b) Plan-Execute Paradigm

#### Figure 5: LLM-As-Judge scoring results across both orchestration systems, ranked by task completion rate (𝑦1).

scores, while llama-4-maverick ties for the top completion rate. Due to its balanced performance across both paradigms, we adopt llama-4-maverick as the default LLM for our ablation studies.

Reference-Based Alignment. As shown in Table 2, referencebased scoring reveals a discrepancy between reasoning quality and lexical fidelity. For Plan-Execute, open-weight models like mistral-large and llama-3-405b achieve strong sequence-level alignment (ROUGE-L ≈ 0.34). However, in the Agent-As-Tool setting, ROUGE scores are heavily influenced by trajectory length. While gpt-4.1 excels in qualitative reasoning, its verbose traces result in lower lexical overlap compared to the concise references. This highlight’s a critical limitation: ROUGE captures textual fidelity rather than semantic validity. Consequently, reference-based metrics should be interpreted as measures of workflow adherence rather than absolute diagnostic accuracy.

### 5.4 Operational Efficiency and Evaluation Reliability

Execution Efficiency. Table 3 reports the mean number of steps (tool invocations) and wall-clock runtime per task for seven LLMs under two agentic implementations. Overall, Plan-Execute is consistently more step-efficient: most models complete tasks in ≈2.6–

- 4.4 steps, whereas Agent-As-Tool typically requires ≈4–6+ steps due to its iterative think–act–observe loop. However, Plan-Execute exhibits limited sensitivity to task complexity: the step counts for single-agent and multi-agent tasks are often nearly identical (e.g., gpt-4.1: 2.6 vs. 2.9; granite-3-3-8b: 5.2 vs. 5.1), suggesting a tendency to over-plan even when the objective is relatively simple.

Runtime trends are more nuanced than step counts (See Runtime columns in Tables 3). While fewer steps can reduce execution time for some models (e.g., gpt-4.1 remains the fastest and most stepefficient Plan-Execute backbone, especially on multi-agent tasks), several models incur substantial overhead in Plan-Execute (e.g., llama-3-70b-instruct and granite-3-3-8b exceed ∼370–430 seconds on multi-agent tasks). In contrast, Agent-As-Tool can be competitive—and for some backbones, dramatically faster—despite taking more steps (e.g., llama-4-maverick drops from ∼377s in Plan-Execute to ∼137s in Agent-as-Tool on multi-agent tasks), indicating lower per-step overhead and more direct tool control. At the same time, tool/data effects can dominate end-to-end latency: mistral-large shows high variance in Agent-As-Tool due to a

pathological outlier with prolonged JSONReader calls over large histories.

Overall, Plan-Execute offers structural and step efficiency, but its wall-clock cost depends strongly on model–planner–executor alignment, whereas Agent-As-Tool often spends more steps yet tends to deliver stronger end-task quality when correctness is prioritized.

Small Language Models (SLM) Analysis. Overall, SLMs tend to underperform. However, we conduct a more detailed analysis to identify where SLMs are effective and where they break down. In the Agent-As-Tool evaluation, models such as granite-3-8b and llama-3-3-70b exhibit weaker aggregate performance, yet they demonstrate clear areas of specialization, as shown in Figure 6a. Both models perform strongly on structured sensing and diagnostic tasks: for example, granite-3-3-8b achieves 15/20 on IoT, 18/22 on FMSR, and 19/23 on TSFM, while llama-3-3-70b reaches 12/20, 18/22, and 20/23 on the same categories. However, they struggle substantially on Work Order tasks, with scores of only 2/36 and 7/36, indicating that procedural, multi-step coordination remains difficult even under the Agent-As-Tool mechanism. This underscores a key insight: industrial deployments may benefit most from hybrid LLM–SLM agent architectures, where strong specialists handle sensing and diagnostics while more capable generalist models manage planning and end-to-end coordination [2].

Human Validation of LLM Judge. To assess the reliability of using LLMs as automatic evaluators for benchmarking tasks, we compare model-generated judgments against human annotations on a sample of 40 tasks. Each task is evaluated along three dimensions by four domain experts, all operating under the same information constraints as the LLMs. Before selecting a default evaluator, we compared several candidate judge models, including gpt-4.1. In this comparison, gpt-4.1 showed only moderate alignment with expert assessments, achieving 69% accuracy and Cohen’s 𝜅 of 0.44. In contrast, llama-4-maverick provided substantially stronger agreement with human judgments and was therefore selected as the default judge model for the main analysis. Across experts, inter-rater reliability scores indicate substantial agreement on key evaluation dimensions, with Data Retrieval Accuracy exhibiting the strongest consistency (Cohen’s 𝜅 = 0.79, 90.48% accuracy). Task Completion (𝜅 = 0.62) and Generalized Result Verification (𝜅 = 0.71) also show high alignment among evaluators.

#### Table 2: ROUGE-based reference scoring across both orchestration systems. Plan-Execute scores are computed on planner outputs; Agent-as-Tool scores are computed on extracted thinking traces, where verbosity can reduce ROUGE.

Plan-Execute Agent-As-Tool Model R-1 R-2 R-L R-Lsum R-1 R-2 R-L R-Lsum mistral-large 0.420 0.251 0.343 0.404 0.3691 0.1933 0.2971 0.3124

- llama-3-405b-instruct 0.406 0.243 0.337 0.381 0.3394 0.1673 0.2740 0.2787

- llama-4-maverick 0.403 0.240 0.325 0.383 0.2560 0.1252 0.2067 0.2273 llama-4-scout-17b-16e-instruct 0.395 0.227 0.265 0.368 0.3126 0.1522 0.2398 0.2621 gpt-4.1-2025-04-14 0.354 0.182 0.289 0.335 0.1628 0.0816 0.1332 0.1389 granite-3-3-8b-instruct 0.373 0.214 0.291 0.353 0.2473 0.1001 0.1867 0.2079 llama-3-3-70b-instruct 0.297 0.172 0.242 0.280 0.3661 0.1963 0.2971 0.3177

#### Table 3: Execution efficiency summary: mean steps per tasks and mean runtime per tasks. Single-agent tasks contain 99 utterances, while multi-agent tasks contain 42 utterances.

Agent-As-Tool Plan-Execute Single-agent tasks Multi-agent tasks Single-agent tasks Multi-agent tasks

Model Steps Runtime (Sec) Steps Runtime (Sec) Steps Runtime (Sec) Steps Runtime (Sec)

gpt-4.1 6.0 104 6.4 218 2.6 93 2.9 180 mistral-large 4.9 347 5.2 289 2.7 187 3.0 210

- llama-3-405b-instruct 4.8 250 5.6 255 3.1 208 4.0 224

- llama-3-70b-instruct 3.9 101 4.3 151 6.7 382 6.5 370
- llama-4-maverick-17b-128e 4.3 120 4.5 137 4.0 385 3.9 377

- llama-4-scout-17b-16e 4.4 101 5.8 178 3.9 172 4.4 218 granite-3-3-8b 5.3 197 6.6 228 5.2 413 5.1 433

See Appendix G for ablation studies on distractor agent impact and single- versus multi-agent performance comparisons.

### 5.5 Scenarios Realism Analysis

To validate the operational authenticity of AssetOpsBench, we conducted a formal expert validation study. A panel of domain experts—including reliability engineers, maintenance specialists, and data scientists—evaluated a random sample of 25 representative scenarios. As detailed in Appendix Section C.4, the results demonstrate high statistical robustness. The study yielded a Cronbach’s Alpha of 0.89, confirming strong internal consistency across the scenario ratings. Although we observed natural variability of individual raters (ICC(1) = 0.23), aggregate reliability was high (ICC(2) = 0.88), indicating that averaged expert judgments provide a highly stable estimate of perceived realism. Furthermore, a Fleiss’ Kappa of 0.21 was obtained; while seemingly modest, this level of agreement is consistent with subjective realism assessments in complex, high-stakes industrial domains. These findings verify that our 141 scenarios faithfully encapsulate the diagnostic reasoning and technical rigor required in professional industrial practice.

### 5.6 Failure Analysis via Agent Trajectories

We analyze agent reasoning traces (trajectories) to identify systemic failure modes. Across approximately 881 collected trajectories for the Agent-As-Tool strategy, we conduct a two-pronged error analysis focusing on tool-related errors and agent failure modes.

Failure Analysis on Tool Use. Each agent step in a trajectory is logged as a structured JSON record capturing the action type and execution state. At the sub-agent level, we distinguish between Tool-oriented actions, which invoke predefined functions with well-defined inputs and outputs, and CodeReAct-oriented actions, where agents dynamically generate and execute Python code. Our analysis shows that Tool-oriented actions achieve higher validexecution rates, whereas CodeReAct-oriented actions incur more runtime failures due to the variability of the generated code. Tooloriented failures are concentrated in a small number of tools, including jsonreader, tsfm_integrated_tsad, and Read Sensors From File, highlighting challenges related to input validation and hallucinated parameter passing.

Emerging Failure Modes Discovery. Now we investigate trajectories from a semantic perspective. Recent work [3] defines 14 failure modes for agent trajectories. Figure 6b shows the distribution of the failure mode in our 881 trajectories in this taxonomy. We found that system design is the most common source of failures. This taxonomy provides guidance to improve agent development. For instance, since the “Fail to Ask for Clarification” mode occurs around 10% of the time, we introduced a feature in the AgentAs-Tool strategy that allows sub-agents to ask the parent agent a clarification questions at any point during execution. We reran the entire benchmark on the default LLM, and this change led to significant performance improvements for llama-4-maverick, increasing task completion from 59% to 66%, surpassing gpt-4.1.

[Figure 8]

[Figure 9]

(a) Agent-level task accomplishment w.r.t. AgentAs-Tool approach.

(b) Distribution of failure modes over 881 selected agent trajectories.

#### Figure 6: Agent performance and failure-mode coverage in AssetOpsBench.

To capture failure mode behaviors beyond this taxonomy, we allowed self-discovery of up to two novel failure modes per trace, revealing emergent and compound failures not covered by existing classifications. This automated diagnostic find several emergent failure modes: Overstatement of Task Completion (122 cases, 23.8%), Extraneous or Ambiguous Output Formatting (110 cases, 21.4%) and Ineffective Error Recovery (160 cases).

### 6 Community Adoption and Generalization

Community Engagement. To move beyond static evaluation [17], we operationalized AssetOpsBench as a live, code-executed competition on Codabench [7]. This platform enables end users to integrate custom agent architectures and evaluate them against hidden scenarios via containerized submissions. Over a three-month period, the competition engaged 250+ participants (44.2% Undergraduate, 28.4% Industry, 23.6% Postgraduate, and 3.8% Other), resulting in 500+ agent submissions across specialized planning and execution tracks. This deployment demonstrates the system’s capability to host custom agents and maintain a dynamic leaderboard; furthermore, to maximize academic impact and inclusivity, we provided full LLM API support funded through our internal budget to ensure participants could iterate without financial barriers.

Generalization Across Industrial Assets. To evaluate generalization capabilities, we developed 162 additional scenarios across four distinct datasets and an extended environment (Figure 2) with new datasets and scenarios. These datasets include MetroPT-3 [9] (15 scenarios) focused on compressor faults, the UCI Hydraulic System [12](17 scenarios) for hydraulic component failures, an internal Asset Health dataset (42 scenarios) derived from industrial work orders, and FailureSensorQA (88 scenarios) based on ISOstandardized sensor-to-failure mapping. Figure 7 shows summary of results for Agent-As-Tool approach. Among all the datasets, scenarios of MetroPT-3 are difficult as we observed poor performance

[Figure 10]

Figure 7: Extended Validation for Different Asset Class (task completion rate = 26.7%), whereas Asset Health (a production usecase) had a near perfect solution.

### 7 Conclusion and Future Outlook

This paper introduces AssetOpsBench, a formalized framework for industrial AI agents using a standardized evaluation methodology and the Agent-As-Tool paradigm to orchestrate complex multi-agent interactions. Through a competition that engaged 250+ participants across academia and industry, we demonstrated the system’s capability to host custom agents and foster iterative refinement via reproducible, code-executed evaluations. Future extensions will incorporate realistic constraints, such as compute limits and API costs, to drive the development of cost-aware autonomous industrial algorithms.

### References

[1] Pierre Andrews, Amine Benhalloum, Gerard Moreno-Torres Bertran, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Romain Froger, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurençon, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Ménard, Grégoire Mialon, Ulyana Piterbarg, Mikhail Plekhanov, Mathieu Rita, Andrey Rusakov, Thomas Scialom, Vladislav Vorotilov, Mengjue Wang, and Ian Yu. 2025. ARE: Scaling Up Agent Environments and Evaluations. arXiv:2509.17158 [cs.AI] https://arxiv.org/abs/2509.17158

- [2] Peter Belcak, Greg Heinrich, Shizhe Diao, Yonggan Fu, Xin Dong, Saurav Muralidharan, Yingyan Celine Lin, and Pavlo Molchanov. 2025. Small Language Models are the Future of Agentic AI. arXiv:2506.02153 [cs.AI] https: //arxiv.org/abs/2506.02153
- [3] Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. 2025. Why Do Multi-Agent LLM Systems Fail?. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https: //openreview.net/forum?id=fAjbYBmonr
- [4] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Aleksander Madry, and Lilian Weng. 2025. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=6s5uXNWGIh
- [5] Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik R Narasimhan, and Shunyu Yao. 2024. FireAct: Toward Language Agent Finetuning. https: //openreview.net/forum?id=RqUMWdDg52
- [6] Yinfang Chen, Manish Shetty, Gagan Somashekar,Minghua Ma, Yogesh Simmhan, Jonathan Mace, Chetan Bansal, Rujia Wang, and Saravan Rajmohan. 2025. AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds. CoRR abs/2501.06706 (2025). arXiv:2501.06706 doi:10.48550/ARXIV.2501. 06706
- [7] CodabenchTeam. [n. d.]. Codabench: A Framework for Reproducible, Code-Based Benchmarking. https://www.codabench.org/competitions/10206/. Accessed: 2026-02-06.
- [8] Christodoulos Constantinides, Dhaval C Patel, Shuxin Lin, Claudio Guerrero, SUNIL DAGAJIRAO PATIL, and Jayant Kalagnanam. 2025. FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https://openreview.net/forum?id= 9KfkMAy2ut
- [9] Narjes Davari, Bruno Veloso, Rita Ribeiro, and Joao Gama. 2021. MetroPT-3 Dataset. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5VW3R.
- [10] Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, and Saleema Amershi. 2024. Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks. CoRR abs/2411.04468

(2024). arXiv:2411.04468 doi:10.48550/ARXIV.2411.04468

- [11] Will Fu-Hinthorn. 2025. Benchmarking Multi-Agent Architectures. LangChain Blog. https://blog.langchain.com/benchmarking-multi-agent-architectures/ Accessed: 2026-01-01.
- [12] Nikolai Helwig, Eliseo Pignanelli, and Andreas Schtze. 2015. Condition monitoring of hydraulic systems. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CW21.
- [13] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. In The Twelfth International Conference on Learning Representations. https: //openreview.net/forum?id=VtmBAGCN7o
- [14] Mengkang Hu, Pu Zhao, Can Xu, Qingfeng Sun, Jian-Guang Lou, Qingwei Lin, Ping Luo, and Saravan Rajmohan. 2025. AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (Toronto ON, Canada) (KDD ’25). Association for Computing Machinery, New York, NY, USA, 496–507. doi:10.1145/3690624.3709321
- [15] Kung-Hsiang Huang, Akshara Prabhakar, Onkar Thorat, Divyansh Agarwal, Prafulla Kumar Choubey, Yixin Mao, Silvio Savarese, Caiming Xiong, and ChienSheng Wu. 2026. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions. Transactions on Machine Learning Research (2026). https://openreview.net/forum?id=EPlpe3Fx1x
- [16] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. 2024. Benchmarking Large Language Models as AI Research Agents. https://openreview.net/forum? id=N9wD4RFWY0
- [17] Hugging Face. 2026. Community Evals: Because We’re Done Trusting Black-Box Leaderboards Over the Community. https://huggingface.co/blog/communityevals. Blog post.
- [18] Phan Nhat Huy, Tien N Nguyen, and Nghi D. Q. Bui. 2025. HyperAgent: Generalist Software Engineering Agents to Solve Coding Tasks at Scale. https://openreview. net/forum?id=PZf4RsPMBG
- [19] IBM. [n.d.]. IBM Maximo Application Suite. https://www.ibm.com/products/ maximo Accessed: May 13, 2025.
- [20] IBM. 2024. Reliability centered maintenance software with IBM Maximo Application Suite. https://www.ibm.com/products/maximo/reliability-centeredmaintenance Accessed: 2024-02-08.

- [21] ISO. 2016. ISO 14224:2016 Petroleum, petrochemical and natural gas industries

— Collection and exchange of reliability and maintenance data for equipment. ISO (2016). https://www.iso.org/standard/14224.html Provides asset taxonomy and data structuring guidance.

- [22] ISO-2024. 2024. ISO 55000:2024 Asset management — Vocabulary, overview and principles. ISO (2024). https://www.iso.org/standard/55000.html Defines asset management lifecycle and taxonomy.
- [23] Saurabh Jha, Rohan R. Arora, Yuji Watanabe, Takumi Yanagawa, Yinfang Chen, Jackson Clark, Bhavya Bhavya, Mudit Verma, Harshit Kumar, Hirokuni Kitahara, Noah Zheutlin, Saki Takano, Divya Pathak, Felix George, Xinbo Wu, Bekir O Turkkan, Gerard Vanloo, Michael Nidd, Ting Dai, Oishik Chatterjee, Pranjal Gupta, Suranjana Samanta, Pooja Aggarwal, Rong Lee, Jae wook Ahn, Debanjana Kar, Amit Paradkar, Yu Deng, Pratibha Moogi, Prateeti Mohapatra, Naoki Abe, Chandrasekhar Narayanaswami, Tianyin Xu, Lav R. Varshney, Ruchi Mahindru, Anca Sailer, Laura Shwartz, Daby Sow, Nicholas C. M. Fuller, and Ruchir Puri.

2025. ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks. In Forty-second International Conference on Machine Learning. https: //openreview.net/forum?id=jP59rz1bZk

- [24] jonathanwvd. 2025. awesome-industrial-datasets: A curated collection of public industrial datasets. GitHub repository. https://github.com/jonathanwvd/ awesome-industrial-datasets Accessed: 2026-01-01.
- [25] Byoungjip Kim, Youngsoo Jang, Lajanugen Logeswaran, Geon-Hyeong Kim, Yu Jin Kim, Honglak Lee, and Moontae Lee. 2024. Prospector: Improving LLM Agents with Self-Asking and Trajectory Ranking. https://openreview.net/forum? id=YKK1jXEWja
- [26] LangChain. 2025. Agent Evaluation Metric. https://www.philschmid.de/agentspass-at-k-pass-power-k Accessed: 2026-01-01.
- [27] LangChain. 2025. Benchmarking Single Agent Performance. https://blog. langchain.dev/react-agent-benchmarking/ Accessed: 2025-05-12.
- [28] Ao Li, Yuexiang Xie, Songze Li, Fugee Tsung, Bolin Ding, and Yaliang Li. 2025. Agent-Oriented Planning in Multi-Agent Systems. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=EqcLAU6gyU
- [29] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023. Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id= HtqnVSCj3q
- [30] Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, and Junnan Li. 2025. MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers. arXiv:2508.14704 [cs.AI] https://arxiv.org/abs/2508. 14704
- [31] Sami Marreed, Alon Oved, Avi Yaeli, Segev Shlomov, Ido Levy, Aviad Sela, Asaf Adi, and Nir Mashkif. 2025. Towards Enterprise-Ready Computer Using Generalist Agent. arXiv:2503.01861 [cs.DC] https://arxiv.org/abs/2503.01861
- [32] Claire Bizon Monroc, Ana Busic, Donatien Dubuc, and Jiamin Zhu. 2024. WFCRL: A Multi-Agent Reinforcement Learning Benchmark for Wind Farm Control. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https://openreview.net/forum?id=ZRMAhpZ3ED
- [33] Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, Dieuwke Hupkes, Ricardo Silveira Cabral, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach, William Yang Wang, and Roberta Raileanu.

2025. MLGym: A New Framework and Benchmark for Advancing AI Research Agents. arXiv:2502.14499 [cs.CL] https://arxiv.org/abs/2502.14499

- [34] Avisek Naug, Antonio Guillen, Ricardo Luna, Vineet Gundecha, Cullen Bash, Sahand Ghorbanpour, Sajad Mousavi, Ashwin Ramesh Babu, Dejan Markovikj, Lekhapriya D Kashyap, Desik Rengarajan, and Soumyendu Sarkar. 2024. SustainDC: Benchmarking for Sustainable Data Center Control. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates, Inc., 100630–100669. https://proceedings.neurips. cc/paper_files/paper/2024/file/b6676756f8a935e208f394a1ba47f0bc-PaperDatasets_and_Benchmarks_Track.pdf
- [35] NVIDIA. 2025. About NVIDIA Agent Intelligence Toolkit Workflows, Version 1.1.0. https://docs.nvidia.com/aiqtoolkit/1.1.0/workflows/about/index.html. Accessed: 2025-11-23.
- [36] Oracle. 2025. Add Failure Diagnostics Information to Asset Incidents and Anomalies. https://docs.oracle.com/en/cloud/saas/iot-asset-cloud/iotaa/add-failurediagnostics-information-asset-incidents-and-anomalies.html. Oracle IoT Asset Monitoring Cloud Service.
- [37] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. ChatDev: Communicative Agents for Software Development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,

- 15174–15186. doi:10.18653/v1/2024.acl-long.810
- [38] James T Rayfield, Shuxin Lin, Nianjun Zhou, and Dhaval C Patel. 2025. ReAct Meets Industrial IoT: Language Agents for Data Access. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, Saloni Potdar, Lina Rojas-Barahona, and Sebastien Montella (Eds.). Association for Computational Linguistics, Suzhou (China), 364–382. doi:10.18653/v1/2025. emnlp-industry.24
- [39] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. HuggingGPT: solving AI tasks with chatgpt and its friends in hugging face. In Proceedings of the 37th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’23). Curran Associates Inc., Red Hook, NY, USA, Article 1657, 27 pages.
- [40] Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. 2024. TaskBench: Benchmarking Large Language Models for Task Automation. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates, Inc., 4540–4574. https://proceedings.neurips.cc/ paper_files/paper/2024/file/085185ea97db31ae6dcac7497616fd3e-PaperDatasets_and_Benchmarks_Track.pdf
- [41] SkyFoundry contributors. [n.d.]. SkySpark (software). https://skyfoundry.com/. Proprietary software; not a Wikipedia page.
- [42] Yifan Song, Weimin Xiong, Xiutian Zhao, Dawei Zhu, Wenhao Wu, Ke Wang, Cheng Li, Wei Peng, and Sujian Li. 2024. AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories. arXiv:2410.07706 [cs.CL] https://arxiv.org/abs/2410.07706
- [43] Andrew Szot, Bogdan Mazoure, Omar Attia, Aleksei Timofeev, Harsh Agrawal, Devon Hjelm, Zhe Gan, Zsolt Kira, and Alexander Toshev. 2024. From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons. arXiv:2412.08442 [cs.LG] https://arxiv.org/abs/2412.08442
- [44] LangChain Team. 2025. Benchmarking Single Agent Performance. LangChain Blog. https://blog.langchain.com/react-agent-benchmarking/ Accessed: 202601-01.
- [45] Lu Wang, Fangkai Yang, Chaoyun Zhang, Junting Lu, Jiaxu Qian, Shilin He, Pu Zhao, Bo Qiao, Ray Huang, Si Qin, Qisheng Su, Jiayi Ye, Yudi Zhang, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. 2025. Large Action Models: From Inception to Implementation. arXiv:2412.10047 [cs.AI] https://arxiv.org/abs/2412.10047
- [46] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. 2024. Executable code actions elicit better LLM agents. In Proceedings of the 41st International Conference on Machine Learning (Vienna, Austria) (ICML’24). JMLR.org, Article 2054, 25 pages.
- [47] Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, and Eugene Siow.

2025. MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex RealWorld Tasks via MCP Servers. arXiv:2508.20453 [cs.CL] https://arxiv.org/abs/ 2508.20453

- [48] Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxin Xu, Yiming Liu, Jie Tang, Hongning Wang, and Minlie Huang. 2024. Benchmarking Complex Instruction-Following with Multiple Constraints Composition. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates, Inc., 137610–137645. https://proceedings.neurips. cc/paper_files/paper/2024/file/f8c24b08b96a08ec7a7a975feea7777e-PaperDatasets_and_Benchmarks_Track.pdf
- [49] WikiALM. [n.d.]. Asset lifecycle management. https://en.wikipedia.org/wiki/ Enterprise_asset_management. Accessed: 2025-11-25.
- [50] WikiBMS. [n.d.]. Building management system. https://en.wikipedia.org/wiki/ Energy_management_system_(building_management). Accessed: 2025-11-25.
- [51] WikiEAM. [n.d.]. Enterprise asset management. https://en.wikipedia.org/wiki/ Enterprise_asset_management. Accessed: 2025-11-25.
- [52] WikiSCADA. [n.d.]. SCADA. https://en.wikipedia.org/wiki/SCADA. Accessed: 2025-11-25.
- [53] Yu Xia, Jingru Fan, Weize Chen, Siyu Yan, Xin Cong, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2025. AgentRM: Enhancing Agent Generalization with Reward Modeling. arXiv:2502.18407 [cs.CL] https://arxiv. org/abs/2502.18407
- [54] Binfeng Xu, Zhiyuan PENG, Bowen Lei, Subhabrata Mukherjee, and Dongkuan Xu. 2024. DECOUPLING REASONING FROM OBSERVATIONS FOR EFFICIENT AUGMENTED LANGUAGE MODELS. https://openreview.net/forum? id=CpgoO6j6W1
- [55] Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Zhiruo Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Melroy Maben, Raj Mehta, Wayne Chi, Lawrence Keunho Jang, Yiqing Xie, Shuyan Zhou, and Graham Neubig. 2025. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks. In The Thirty-ninth Annual Conference on Neural Information Processing

- Systems Datasets and Benchmarks Track. https://openreview.net/forum?id= LZnKNApvhG
- [56] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. 𝜏bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045 [cs.AI] https://arxiv.org/abs/2406.12045
- [57] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. Publisher Copyright: © 2023 11th International Conference on Learning Representations, ICLR 2023. All rights reserved.; 11th International Conference on Learning Representations, ICLR 2023 ; Conference date: 01-05-2023 Through 05-05-2023.
- [58] Jianguo Zhang, Thai Hoang, Ming Zhu, Zuxin Liu, Shiyu Wang, Tulika Awalgaonkar, Akshara Prabhakar, Haolin Chen, Weiran Yao, Zhiwei Liu, Juntao Tan, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, and Caiming Xiong. 2025. ActionStudio: A Lightweight Framework for Data and Training of Large Action Models. arXiv:2503.22673 [cs.AI] https://arxiv.org/abs/2503.22673
- [59] Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Quoc Hoang, Shirley Kokane, Weiran Yao, Juntao Tan, Akshara Prabhakar, Haolin Chen, Zhiwei Liu, Yihao Feng, Tulika Manoj Awalgaonkar, Rithesh R N, Zeyuan Chen, Ran Xu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, and Caiming Xiong. 2025. xLAM: A Family of Large Action Models to Empower AI Agent Systems. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association for Computational Linguistics, Albuquerque, New Mexico, 11583–11597. https: //aclanthology.org/2025.naacl-long.578/
- [60] Nianjun Zhou, Dhaval C. Patel, and Anamitra Bhattacharyya. 2026. Deployed AI Agents for Industrial Asset Management: CodeReAct Framework for Event Analysis and Work Order Automation. In Proceedings of the AAAI Conference on Artificial Intelligence (Innovative Applications of Artificial Intelligence (IAAI)). AAAI 2026 Innovative Applications of AI Track.

### A Details on Multi-Agent Systems

AssetOpsBench uses an extended ReAct framework [38] as the base agent to both multi-agent systems. Subsequent sections present the architecture in details, highlighting the distinction between two architectural paradigms: Agent-As-Tool (See A.1) and PlanExecute (See A.2). Note that, we can replace ReAct by any other agent development methdology such as Reflect, RAFA, etc.

### A.1 Agent As Tools

An Agent-as-Tool multi-agent system is a multi-agent setup where each agent is treated like a callable tool, meaning agents are not fully autonomous “actors,” but modular capabilities that can be invoked on demand by an orchestrator agent or a meta agent.

The Agent-As-Tool paradigm, coupled with the Agent-Ask mechanism, offers a robust framework for complex task execution by decomposing a problem and routing sub-tasks to specialized AI agents. As illustrated in the workflow Figure 8, the primary AgentAs-Tool receive a user query. The Agent-As-Tool has to fulfill three core responsibilities: (1) determining the appropriate specialized agent (e.g., the TSFM Agent or IoT Agent) for a given sub-task, (2) correctly assigning the task, and (3) facilitating clarification from the specialized agent if needed. Once the Agent-As-Tool receives an input query, it enters a standard Think-Act-Observe loop, and it decides on the appropriate specialized agent in think step. The specialized agent executes its designated task and provides an Answer. Answer is embedded inside to the orchestration agent’s observation for next set of action. In this case, an artifact like a data-file.json, back to the main orchestrator, allowing the overall system to complete complex, multi-faceted operations that exceed the capability of any single agent.

Algorithm 1 details an iterative, multi-turn execution framework akin to the ReAct (Reasoning and Acting) paradigm. This

[Figure 11]

#### Figure 8: Typical Workflow: Agent-As-Tool using Agent-Ask.

framework enables a central agent to solve a complex User Task (Q) by strategically engaging specialized agents (A𝑠𝑝𝑒𝑐) within a bounded number of steps 𝑇𝑚𝑎𝑥. The process begins with the initialization of the agent’s internal state 𝑀 and an empty execution history H. The agent then enters an iterative loop where a thinking policy Π𝑡ℎ𝑖𝑛𝑘(𝑀) determines the next action (either THINK-ACT or FINISH) and the continuation signal. When the action is THINK-ACT, the agent selects the best agent_id via Π𝑠𝑒𝑙𝑒𝑐𝑡 and formulates a precise sub_task via Π𝑓 𝑜𝑟𝑚𝑢𝑙𝑎𝑡𝑒 for execution. Conversely, if the action is FINISH, the agent summarizes the full Π, updates the memory, and terminates the loop. Following any action, the resulting output is compressed into a concise observation (Π𝑐𝑜𝑚𝑝𝑟𝑒𝑠𝑠) to manage context length. This observation is logged to H, and critically, the agent’s internal memory 𝑀 is updated with this new context, driving the decision-making in the subsequent round. Finally, once the loop terminates, a policy Π𝑓 𝑖𝑛𝑎𝑙 generates the complete Final Output (𝑂) by synthesizing the entire execution history H.

To implement Agent-As-Tools, we implemented the following components:

- • A standard ReAct (Think–Act–Observe) agent loop using open source framework. In the initial setup, the number of reflections was set to one—effectively disabling reflection.
- • A curated list of tools, the majority of which are stub interfaces that delegate functionality to specialized sub-agents. The onlystandaloneutilitytool inthis set was theJSONReader, which reads a JSON object from a file and returns its contents as the tool’s direct response.

The sub-agent stubs were intentionally designed to be minimal. Each stub accepted a single input parameter: a string called request and returned a structured JSON output. The output JSON object included the following fields:

• answer – the primary answer returned by the sub-agent,

represented as a plain string.

- • review – a nested JSON object capturing a review of the response,typicallyincludingfieldssuch asstatus,reasoning, and suggestions.
- • summary – a brief description of the JSON object’s structure and semantics, useful for interpretability or chaining with downstream tools.

|Question: download asset history for CU02004 at SiteX from 2016-07-14T20:30:00 -04:00 to 2016-07-14T23<br><br>↩→ :30:00 -04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"<br><br>Action 1: IoTAgent Action Input 1: request=download asset history for<br><br>↩→ CU02004 at SiteX from 2016-07-14T20:30:00 -04:00 to 2016-07-14T23:30:00 -04:00 for "CHILLED WATER LEAVING TEMP<br><br>↩→ " and "CHILLED WATER RETURN TEMP" Observation 1: {<br><br>"site_name": "SiteX", "assetnum": "CU02004", "total_observations": 25, "start": "2025-03-26T00:00:00.000000+00:00" , "final": "2025-04-02T00:00:00.000000+00:00" , "file_path": "/var/folders/fz/.../cbmdir/c328516a -643f<br><br>↩→ -40e6 -8701-e875b1985c38.json",<br><br>"message": "found 25 observations. file_path contains a<br><br>↩→ JSON array of Observation data" }<br><br>|
|---|

#### Listing 1: Example of Trajectory calling IoTAgent in AgentAs-Tool workflow

We have provided a sample (partial) trajectory trace in Listing 1, which show how patent agent call one of the tool (in this case IoTAgent) and receive a response. The recorded metrics include:

- • Question: the input query being processed
- • Total execution time: duration of the entire ReAct loop
- • Number of ReAct steps: count of action-observation cycles

Algorithm 1: Agent-As-Tool (Simplified as ReAct)

Input :User Query Q ∈ Q, Maximum Steps 𝑇𝑚𝑎𝑥, Set of Agents

A𝑠𝑝𝑒𝑐 ⊆ A Output:Final Output 𝑂 ∈ O, Execution Plan Π 𝑀 ← InitializeMemory(Q) //Initialize the global Memory

System 𝑀 Π ← ∅; H ← ∅; for 𝑡 = 1 to 𝑇𝑚𝑎𝑥 do

(𝑐𝑜𝑛𝑡𝑖𝑛𝑢𝑒, action) ← Π𝑡ℎ𝑖𝑛𝑘 (𝑀) //Agent decides next

step using current Memory 𝑀 if action = THINK-ACT then

agent_id ← Π𝑠𝑒𝑙𝑒𝑐𝑡 (action,𝑀, A𝑠𝑝𝑒𝑐 ) //Select best Agent 𝐴𝑖 ∈ A𝑠𝑝𝑒𝑐 𝜏 ← Π𝑓𝑜𝑟𝑚𝑢𝑙𝑎𝑡𝑒 (action,𝑀) //Formulate sub-task 𝜏 ∈ T based on current Memory 𝑀 output ← ExecuteAgent(agent_id,𝜏) //Agent executes

task and returns structured output 𝑜 ∈ O Π ← Π ∪ {⟨𝜏, agent_id⟩} //Update the execution

plan Π with task-agent assignment else if action = FINISH then

final_output ← Summarize(Π); 𝑀 ← UpdateMemory(𝑀, final_output); break;

observation ← Π𝑐𝑜𝑚𝑝𝑟𝑒𝑠𝑠 (output) //Generate concise observation summary (optional

H ← H ∪ {(𝑡, action, observation)} //Log step to

history 𝑀 ← UpdateMemory(𝑀, observation) //Update Memory 𝑀

with new context from observation if 𝑐𝑜𝑛𝑡𝑖𝑛𝑢𝑒 = False then

break //Stop if agent decides termination 𝑂 ← Π𝑓𝑖𝑛𝑎𝑙 (H) //Produce final answer from the complete

history

return (𝑂, Π, H)

• Review status: success or failure determined by the LLM-

based reviewer

### A.2 Plan Execute

Plan-Execute is a widely used architectural paradigm for multiagent systems. Figure 9 depicts the implementation adopted in our work. It is derived from specialized multi-agent system [31]. The process initiates when a user submits a query, which is first processed by the Planner. The Planner decomposes the query into discrete, executable tasks. These tasks are then vetted by a Reviewer component to ensure quality, completeness, and relevance. Upon approval, the Orchestrator assigns the tasks to the most appropriate agents. Each agent independently executes its assigned task and returns a structured response. These responses are then aggregated by the Summarization module, which synthesizes

- them into a coherent final output that is returned to the user. This architecture supports modularity, robustness, and inter-

pretability across the task lifecycle. We have provided two system prompts where first prompt guides an AI to generate a structured

[Figure 12]

#### Figure 9: Plan-Execute Multi-Agent System

step-by-step plan using external agents, while the second prompt instructs a reviewer agent to evaluate the plan’s correctness and completeness in JSON format.

System Prompt (Planning Agent)

You are an AI a s s i s t a n t who makes step −by− step plan to solve a complicated problem under the help of external agents .

For each step , make one task followed by one agent − c a l l . Each step denoted by #S1 , #S2 , # S3 . . .

can be r e f e r r e d to in l a t e r s t e p s as a dependency .

Each step must contain Task , Agent , Dependency and ExpectedOutput .

- 1 . ∗ ∗ Task ∗ ∗ : A d e t a i l e d description of what needs to be done in t h i s step

. I t should include all necessary d e t a i l s and requirements .

- 2 . ∗ ∗ Agent ∗ ∗ : The external agent to be used for solving t h i s task . Agent

needs to be s e l e c t e d from the a v a i l a b l e agents .

- 3 . ∗ ∗ Dependency ∗ ∗ : A l i s t of previous steps ( denoted as #S1 , #S2 , e t c . ) that t h i s step depends on . I f no previous s t e p s are required , use None .
- 4 . ∗ ∗ ExpectedOutput ∗ ∗ : The a n t i c i p a t e d r e s u l t from the agent ' s ␣ execution .

## ␣ Output ␣ Format ␣ ( Replace ␣ ' <... > ' ) ␣ ##

## ␣ Step ␣ 1

- #Task1 : ␣ < describe ␣ your ␣ task ␣ here >

- #Agent1 : ␣ <agent_name >

- #Dependency1 : ␣ None

- # ExpectedOutput1 : ␣ < describe ␣ the ␣ expected ␣ output ␣ of ␣ the ␣ call >

## ␣ Step ␣ 2 #Task2 : ␣ < describe ␣ next ␣ task > #Agent2 : ␣ <agent_name > #Dependency2 : ␣ [# S1 ]

- # ExpectedOutput2 : ␣ < describe ␣ the ␣ expected ␣ output ␣ of ␣ the ␣ call >

And ␣ so ␣ on . . .

Here ␣ are ␣ the ␣ a v a i l a b l e ␣ agents : { agent_descriptions }

You ␣ are ␣ going ␣ to ␣ solve ␣ the ␣ following ␣

complicated ␣ problem : { task . description }

Guidelines : − ␣ Task ␣ should ␣ be ␣ something ␣ that ␣ can ␣ be

␣ solved ␣ by ␣ the ␣ agent . − ␣A␣ plan ␣ usually ␣ contains ␣ l e s s ␣ than ␣ 5 ␣ steps . − ␣ Only ␣ output ␣ the ␣ generated ␣ plan .

Output ␣ ( your ␣ generated ␣ plan ) :

System Prompt (Review Agent)

review_plan_system_prompt_template = """You are a critical reviewer tasked with evaluating the effectiveness and accuracy of a plan. Your

goal is to determine whether the plan is valid or not given the context of the input question and agent expertise. A valid plan should:

- 1. **Ensure all necessary actions are addressed :** The plan must cover all required steps to

successfully complete the task as specified in the question. Ensure that each action directly contributes to the task goal.

- 2. **Include appropriate dependencies between steps:** Actions should be logically ordered with clear dependencies. Each step must rely on the completion of the previous step to ensure a coherent and efficient workflow.

- 3. **Ensure no crucial steps are missed:** The plan must not overlook any essential actions required to solve the task. If any crucial steps are absent , the plan must be flagged as

incomplete.

- 4. **Confirm all actions align with agent capabilities:** Each step in the plan must fall within the designated expertise of the agents involved. No action should require expertise or knowledge outside of the agent 's

specified capabilities. Any plan that violate this condition is an invalid plan.

- 5. **Strictly follow the task 's question:** Carefully compare the provided question with the task. The plan should only include actions that directly relate to the question ' s explicit requirements , without introducing any unnecessary tasks or assumptions.
- 6. **Avoid Abstract task/step:** Ensure steps/tasks are grounded with respect

to the data generated by previous steps or the question.

### Evaluation Criteria:

- 1. **Completeness:**

- - Verify that the system prompt leads to a plan that includes all necessary steps to accomplish the task.
- - Ensure the description of each step contains all the relevant information needed to

execute the step , including any required parameters or inputs that are mentioned in

the task 's question.

- 2. **Relevance:**

- - Confirm that each step in the plan directly contributes to solving the task.
- - Eliminate any steps that do not serve a clear purpose in achieving the goal.

- 3. **Correctness:**

- - Ensure that all steps are logically consistent and ordered correctly.
- - Ensure that the dependencies between the steps are valid and follow a correct sequence.

- 4. **Expertise Alignment:**

- - Confirm that the steps in the plan are within the capabilities of the agent.
- - Validate that the agents used in each steps are among the available agents mentioned in the agents ' expertise.

- 5. **Efficiency:**

- - Make sure the plan doesn 't introduce redundant actions.
- - Avoid unnecessary complexity in the plan.

- 6. **Clarity:**

- Ensure that the plan is easy to understand and logically structured.

---

**Question:** {question}

**Agents ' Expertise:** {agent_expertise}

**Plan:** {plan}

---

### Output Format: Your review must always be in JSON format. Do not

include any additional formatting or Markdown in your response.

```json {{

"status": "Valid | Invalid | Other", "reasoning": "A concise explanation for your

evaluation. If a specific step is wrong , point it out directly.",

"suggestions": "Actions or improvements for rectifying the plan if applicable."

}} ```

Output: """

Figure 10 and Figure 11 illustrate the Plan-Execute approach to tackle a complex industrial query, such as “discover the most relevant sensor for Chiller 6 at POKMAIN site for detecting Compressor Overheating failure?”. The process begins with the main agent receiving the Query and formulating a detailed Plan. This plan is meticulously broken down into sequential steps. For instance, Step 1 involves a Task to “Identify the sensors available for Chiller 6 at POKMAIN site” and specifies Agent 1 (e.g., an IoT Data Download Agent) to execute this task, with an Expected Output of “A list of sensors available for Chiller 6 at POKMAIN site.” Following this, Step 2 takes this output as a Dependency (#S1) to execute the Task: “Determine which of these sensors can detect Compressor Overheating failure,” assigning it to a specialized Agent 2 (e.g., a FMSR Agent).

Afterreviewingtheplan,itistranslated into a dynamic, dependency-

aware Workflow represented as a directed graph. This graph outlines the logical flow and potential parallel execution paths (e.g., tasks 3, 4, and 5 running concurrently) based on the sequential nature of the task dependencies. The Context-aware Execution phase

- then involves a specialized execution engine that manages these tasks, tracking their state, inputs (like the JSON objects and strings containing the intermediate results), and dependencies between agents. For example, the output of the first stage (ID 1) becomes a structured input for subsequent tasks (ID 2), ensuring that information is seamlessly and accurately passed between the specialized agents. The entire process culminates in a Result Summary that provides the final, actionable answer to the initial complex query. This methodology ensures traceability, modularity, and the effective integration of multiple specialized AI agents for industrial problem-solving.

### A.3 Tools used by Agents

In this section, we describe the development of over 15 LangChainbased tools that form the backbone of our agent framework. We

follow a standardized methodology for tool construction, and, with the exception of the WO agent, all agents operate through toolcalling APIs. Table 4 lists thirteen of these tools along with their names, descriptions, and parameters. For brevity, we omit some of the lower-level parameters associated with the time-series tool suite. In case of WO agent, which is a coding agent, we needed to build a generic business driven object, as given in Table 5.

[Figure 13]

Figure 10: Plan-Execute Approach using Review

[Figure 14]

Figure 11: Plan-Execution Workflow Stepwise Concrete Example

Table 4: List of Available Tools and Their Parameters.

Tool Name Description Parameters (Required Fields) Get Failure Modes Retrieves failure modes linked to a spe-

asset_name: name of the asset.

cific asset.

Returns relevancy mapping between failure modes and sensors for downstream tasks.

input_str: string with asset name, failure modes, and sensors.

Get Failure Mode and Sensor Relevancy Mapping

Reads available sensors of an asset from a file and outputs sensor variable names.

input_str: sensor file path.

Read Sensors From File

sites Retrieves a list of available sites. v__args: optional array (default: null). history Returns sensor values for an asset within

site_name, assetnum, start, final.

a given time range.

assets Lists all assets available at a given site. site_name. sensors Lists all sensors for an asset at a given

site_name, assetnum.

site.

jsonreader Parses a JSON file and returns its content. file_name. currentdatetime Returns current date and time as JSON. v__args: optional array (default: null). aitasks Lists available AI tasks and their methods

v__args: optional array (default: null).

(task_id, description).

v__args: optional array (default: null).

tsfmmodels Lists supported forecasting models (ID,

checkpoint, description).

tsfm_forecasting Forecasts sensor or KPI variables using

dataset_path, model_checkpoint, timestamp_column, target_columns.

pretrained time-series models.

Finetunes a pretrained forecasting model on new data.

dataset_path, model_checkpoint, timestamp_column, target_columns.

tsfm forecasting finetune

tsfm integrated tsad Performs time-series anomaly detection

dataset_path, timestamp_column, target_columns.

using model predictions.

Table 5: WO Agent Summary of Business Objects, Source, Role, and Number of Records

Business Object Source Role Count Content Objects WorkOrder Work Order Man-

Tracks scheduled and unscheduled maintenance tasks, categorized as preventive or corrective.

4392

ager

Event Aggregated by Authors

Consolidates event logs for tracking and decisionmaking.

6929

Alert Events IoT Repository Logs real-time alerts triggered by IoT sensors based on predefined conditions.

1995

Anomaly Events ML Generated Detects KPI deviations using machine learning for predictive maintenance.

542

Meta/Profile Objects ISO Failure Code Developed by Au-

Standardizes failure classification for structured maintenance analysis.

137

thors

ISO Primary Failure_Code

Developed by Authors

Defines primary failure categories and links related secondary codes.

68

AlertRule SME Provided Specifies conditions for triggering alerts based on system behaviors.

77

Equipment SME Provided Represents industrial assets, including status and specifications.

22

Relationship Causality Objects Alert-Rule Mapping Relationship

Links alert rules to failure codes for automated diagnostics.

46

Causality

12

Anomaly Mapping Relationship Causality

Associates anomalies with failure codes for predictive insights.

Recommendation Objects WorkOrder Recommendation

Recommendation Suggests maintenance actions based on historical patterns.

N/A

Note: The design and structure of the business objects and corresponding analysis in this section are valid for other industrial asset types, such as standby generators.

- B Benchmark Dataset Details

- B.1 Datasets

This section zooms in on the datasets utilized by the various AssetOpsBench agents.

Sensor Telemetry Dataset for IoT Agent and TSFM Agent. Both IoT Agent and TSFM Agent leverage the Sensor Telemetry Dataset, which comprises sensor telemetry collected from Building Management Systems (BMS) and the SkySpark analytics platform. This dataset captures fifteen-minute interval operational data from industrial HVAC systems, specifically a fleet of chillers. Each chiller unit (e.g., Chiller 4, Chiller 14) is instrumented with a standardized

#### Table 6: Representative sensors in AssetOpsBench.

Sensor Name Description Chiller Return Temperature Temperature of water returning to the chiller. Supply Temperature Temperature of water exiting the chiller. Power Input Electrical power consumption. Tonnage Heat extraction rate (cooling capacity). Condenser Water Supply to Chiller Temperature

Temperature of water supplied to the condenser.

Chiller Efficiency Instantaneous performance metric. Chiller % Loaded Current load as a percentage of the maximum. Condenser Water Flow Flow rate through the condenser. Liquid Refrigerant Evaporator Temperature

Temperature of refrigerant in the evaporator.

Run Status Binary indicator of whether the chiller is operating. Setpoint Temperature Current setpoint for chiller operation.

suite of physical sensors that monitor key operational parameters in real-time.

A representative subset of these sensors is summarized in Table 6. These sensors record various kinematic, dynamic, thermodynamic, electrical, and operational metrics essential to assessing the performance and health of chiller systems. Each sensor stream is accompanied by rich metadata, including sensor type, measurement units, physical location, and structured device tags that define device associations. The dataset captures realistic operational variability, encompassing noise, missing data, and seasonal patterns. As such, it provides a robust foundation for developing and benchmarking models that require temporal reasoning, fault detection, and decision-making under uncertainty.

As illustration, Figure 12 presents layered time series subplots for key chiller sensors over a selected snapshot period in June 2020 for Chiller 6. Each subplot corresponds to one sensor variable, enabling a clear view of temporal dynamics and inter-variable behavior. This figure provides insight into the operational profile of a single chiller unit during real-world usage.

Failure Mode Datasets for FMSR Agent. The failure mode datasets in AssetOpsBench are modeled using the principles of Failure Modes and Effects Analysis (FMEA), a structured framework used in reliability engineering to identify failure risks, assess root causes and effects, and inform condition-based maintenance strategies. Each failure is defined by its mode, degradation mechanism, detection opportunity, and operational impact, enabling structured reasoning for both rule-based diagnostics and machine learning.

Failures in the dataset are annotated at the asset and subsystem levels, with a primary focus on centrifugal chillers. These failures reflect realistic degradation pathways and operational stressors derived from field experience. Each record in the failure model includes:

• Failure Location and Component: The subsystem or part where failure occurs, such as bearings, gearboxes, impellers, or lubrication systems.

[Figure 15]

#### Figure 12: Snapshot of time series data from Chiller 6 for June 2020. Each subplot shows an individual sensor’s trend over time.

- • Degradation Mechanism: The underlying physical process driving the failure, including wear, erosion, oil degradation, vibration-induced fatigue, and misalignment.
- • Degradation Influences: External or internal stressors such as run time, two-phase process fluid, personnel error, or shock loading.
- • Functional Failure Mode: The resulting operational defect, such as decreased oil pressure, audible noise, low head pressure, or capacity loss.
- • Detection Opportunities: Observable precursors or symptoms, including sensor readings (e.g., oil sampling, vibration signals), condition-based alarms, or inspection results.
- • Repair Time and Criticality: Estimated downtime and classification of failure risk, supporting cost-based prioritization and scheduling.
- • Preventive Task Type: Associated maintenance activity, such as oil analysis, vibration analysis, or visual inspection, tagged with effectiveness ratings and intervention intervals.

For example, bearing wear a recurring failure across chiller subsystems may arise from lubrication failure, misalignment, or fluid shock loading. This degradation is detectable via a combination of oil analysis and vibration monitoring, with failure symptoms including increased vibration, reduced oil pressure, and audible anomalies. Similarly, impeller erosion is linked to aging and twophase fluid exposure, typically presenting as reduced capacity and lower head pressure. More of these relations could be found in [8].

Work Order Datasets for WO Agent. Table 5 provide the summary of work order datasets (as business objects) and the size for each dataset. Those work order datasets in AssetOpsBench

provide a structured view of maintenance activity across industrial assets, encompassing both preventive and corrective interventions using work orders. Each work order is associated with rich contextual data including equipment metadata, failure classification codes (e.g., ISO Failure Code, ISO Primary Failure Code), event logs, sensor-triggered alerts, and machine-generated anomalies. These records are linked temporally and causally, allowing agents to reason about asset history, detect recurring failure patterns, and recommend actions based on past interventions.

Data Quality and Operational Realism. The datasets in AssetOpsBench are derived from long-running production building management and enterprise maintenance platforms and therefore reflect the characteristics of real-world industrial data. Sensor telemetry contains missing values, flat segments, and intermittent gaps, arising primarily from genuine asset behavior such as intermittent operation and seasonal usage patterns rather than sensor faults. Maintenance work orders are event-driven and often logged retrospectively by technicians, leading to substantial variability in textual structure and imperfect temporal alignment with sensor measurements and site-specific alerts triggered from those sensors.

Rather than aggressively imputing, filtering, or normalizing these artifacts, we intentionally preserve them and treat asset inactivity and data availability as meaningful operational signals, so that model performance reflects robustness under realistic operating conditions. Downstream agents are designed to explicitly handle these characteristics. In particular, the TSFM agent infers operational on/off periods from sensor telemetry, aligns measurements to a common fifteen-minute grid via resampling, applies limited imputation only when appropriate, and detects anomalies using

a conformal prediction framework during active operation. Similarly, the imperfect alignment between work orders, sensor data, and alerts is preserved to reflect real maintenance practice, consistent with observations from deployed work-order systems [60]. Together, these design choices ensure that AssetOpsBench prioritizes operational realism over artificially curated or cleaned data.

- C Scenario Creation

- C.1 Scenarios Creation Principles

The scenarios in AssetOpsBench are designed to evaluate the capabilities required for autonomous agents operating in real industrial environments. Although grounded in real operational data and engineering practices, each scenario is intentionally framed to test a specific dimension of agent reasoning, tool interaction, and decision-making relevant to asset management. The scenarios are built around four core principles:

- • Reasoning and Tool Use: Scenarios require agents to perform domain-specific reasoning such as time-based logic, schema interpretation, and multi-step tool invocation. Common failure cases include premature termination, incorrect parameter selection, or misuse of diagnostic tools.
- • Data Handling and Forecasting: Agents must interpret telemetry, detect anomalies, and configure appropriate models for forecasting or anomaly detection. Tasks emphasize the translation of real-world engineering intuition into ML configuration steps (e.g., model selection, training windows, thresholds).
- • Agent Communication and Coordination: Many scenarios simulate multi-agent workflows where the agent must ask clarifying questions, summarize findings, or coordinate subtasks. This reflects how real engineering teams collaborate during diagnostics or planning.
- • Workflow Orchestration and Decision-Making: Scenarios measure the agent’s ability to plan complex workflows, handle dependencies, reason under uncertainty, and determine when to stop or escalate due to missing or conflicting information.

These principles ensure that scenarios remain faithful to real asset-management workflows while systematically probing the capabilities of autonomous agents.

### C.2 Scenario Examples

We include two examples (Tables 7 and 8) that showcase distinct behaviors of agent outputs. Although the questions are superficially similar, the required characteristic form differs substantially.

### C.3 Scenario Comparison with Other Bench

We prepare a table to compare with the literature in Table 9. AssetOpsBench extends prior benchmarks by incorporating temporal/dynamic queries, name disambiguation, and tool-output–driven operations. These capabilities not present in TaskBench or ITBench. Additionally, while earlier benchmarks rely on either complex tool graphs or simpler single-step tools, AssetOpsBench emphasizes multi-step tool reuse, aligning better with real industrial agent workflows.

### C.4 User Study Realism Analysis

To quantitatively assess the realism of AssetOpsBench scenarios, we conducted a human evaluation study. We randomly selected 25 representative scenarios covering four categories: IoT queries, time-series forecasting (TSFM), work orders/events, and failure

#### Table 7: Example knowledge query: energy prediction for Chiller 9.

Field Description ID 507 Type Knowledge Query Text What is the predicted energy consumption for Chiller 9 in the

week of 2020-04-27 based on data from the MAIN site? Characteristic form • Verify correct asset/location/time: Chiller 9,

MAIN, week of 2020-04-27.

- • Confirm sensor identification (power input) and historical data retrieval for the specified window.
- • Explain why forecasting cannot be performed: power input is 0.0 from 2020-04-20 to 202004-25 (chiller not operating), implying insufficient signal for prediction.

#### Table 8: Example knowledge query: predicting energy usage for Chiller 9.

Field Description ID 511 Type Knowledge Query Text Can you predict Chiller 9’s energy usage for next week based

on data from the week of 2020-04-27 at MAIN? Characteristic form • Verify correct asset/location and the reference

week: Chiller 9, MAIN, week of 2020-0427.

- • Confirm sensor inventory and selection of Chiller 9 Power Input, followed by data retrieval.
- • Report the file path where the retrieved data is stored.
- • Confirm recovery from initial analysis errors and successful fine-tuning of a time-series forecasting model on the retrieved data.
- • Confirm next-week predictions are generated and written to the specified output file.

mode reasoning (FMSR). Participants were domain experts, including reliability engineers, maintenance engineers, and data scientists familiar with condition-based monitoring and predictive maintenance. Each participant evaluated scenarios using a 3-point scale (1 = Not Realistic, 2 = Realistic, 3 = High Realistic) and could optionally provide qualitative comments. Background questions captured participants’ role, years of experience, and familiarity with predictive maintenance. Responses were collected via a Google Form as shown in Figure 13.

The following metrics were computed to assess internal consistency, inter-rater agreement, and the reliability of aggregated scores.

C.4.1 Reliability Metrics. The internal consistency of the 25 scenario ratings is excellent, as indicated by a Cronbach’s alpha of

Table 9: Comparative overview of general-purpose and domain-specific benchmarks.

Benchmark TaskBench

ITBench (ICML 2025)

AssetOpsBench (Ours)

(NeurIPS 2024)

|Data Generation|Tool Graph + BackInstruct|Manual<br><br>|Manual|
|---|---|---|---|
|Tool Dependency<br><br>|✓|✓<br><br>|✓|
|Quality Control<br><br>|LLM Self-critique + Rulebased|Human Verification|Human Verification|
|Evaluation<br><br>|Task Decomposition + Tool Selection + Parameter Prediction<br><br>|ReActive Planning + Tool Selection|ReActive Planning + Tool Selection + Parameter Prediction|
|Tool Complexity<br><br>|Single tool to complex tool graph|–<br><br>|Multiple tools; same tools can be called multiple times|
|Dataset Scale<br><br>|17,331 samples|141 scenarios|141 scenarios|
|Temporal / Dynamic Query<br><br>|×|×|✓|
|Name Disambiguation<br><br>|×|×|✓|
|Tools Output Operation|×|×|✓|

[Figure 16]

- Figure 13: Representative example of Scenario for Collecting user feedback

### D LLM-As-Judge Evaluation

- D.1 System Prompt

Table 10 is the prompt instruction to the evaluation agent, which outlines the specific evaluation dimensions, constraints, and response formatting guidelines that the model follows when scoring task outputs. The evaluation criteria is also provided to human judges which ensures consistency across evaluations.

- D.2 Human Validation

We conducted human validation of LLM-as-Judge using Google Forms. As shown in Figure 14, domain experts were provided with the original task description, the agent’s reasoning and final answer, and a checklist spanning six evaluation dimensions. Each dimension was assessed using a binary (True/False) judgment, enabling structured comparison between human and model evaluations. We distributed four forms, each containing 10 samples, resulting in 240 total data points. The aggregated results are reported in Section 5.3.

0.887. The ICC(1) value of 0.233 reflects moderate agreement at the individual-participant level, whereas ICC(2) of 0.882 demonstrates that the aggregated ratings across participants are highly reliable. Fleiss’ kappa of 0.209 indicates slight-to-fair categorical agreement among participants, which is consistent with the subjective nature of realism judgments. Overall, while individual participants may vary in their ratings, the averaged scores per scenario provide a stable and trustworthy measure of perceived realism.

### E Reference-Based Evaluation E.1 Ground Truth Preparation

To ensure that each scenario can be objectively evaluated, we first construct a ground-truth specification that precisely defines the expected reasoning steps and final answer. The example in Figure 15 illustrates the task of the FMSR agent in which the system must retrieve sensor names associated with a wind turbine. The ground truth includes the task description, the required planning step, and the exact sequence of execution actions that a correct agent should follow. By explicitly defining the operations — such as calling get

|You are a critical reviewer tasked with evaluating the effectiveness and accuracy of an AI agent’s response to a given task. Your goal is to determine whether the agent has successfully accomplished the task correctly based on the expected or characteristic behavior.<br><br>Evaluation Criteria:<br><br>1. Task Completion:<br><br>- Verify whether the agent executed all required actions (e.g., using the correct tools, retrieving data, performing the necessary analysis).<br>- Ensure the response aligns with the predefined expected behavior for task completion.<br><br><br>2. Data Retrieval & Accuracy:<br><br>- Confirm that the correct asset, location, time period, and sensor (if applicable) were used.<br>- Check that the retrieved data and results (forecasting, anomaly detection, etc.) are correct and consistent with the task requirements.<br><br><br>3. Generalized Result Verification:<br><br><br>- Task Type Verification: Assess if the agent returned the expected results for the task type (forecasting, anomaly detection, classification, etc.).<br>- Forecasting: Ensure forecasts cover the specified future period.<br>- Anomaly Detection: Verify that anomalies were correctly detected when expected.<br>- Other Tasks (e.g., classification): Check that results match expected format and values.<br>- Comparison with Expected Output: Validate that results match the characteristic answer.<br>- Data Integrity: Ensure correct data (sensor, time period) was used and output format is consistent. Inputs: Question: {question} Characteristic Answer (Expected Behavior): {characteristic_answer} Agent’s Thinking: {agent_think} Agent’s Final Response: {agent_response} Output Format: Provide your review strictly in JSON format without any additional text or Markdown. { "task_completion": true/false, "data_retrieval_accuracy": true/false, "generalized_result_verification": true/false, "suggestions": "Optional. Recommended actions to improve the agent’s response if needed." } (END OF RESPONSE) Evaluate the agent’s performance according to the above criteria.<br>|
|---|

#### Table 10: Prompt instruction for LLM-as-a-judge evaluation agent

available sensor information with the asset name “Wind Turbine”, the ground truth provides a verifiable reference trace.

### E.2 ROUGE Scoring Specification ROUGE metrics used include:

- • rouge1: unigram (1-gram) overlap between generated and reference outputs.
- • rouge2: bigram (2-gram) overlap.
- • rougeL: longest common subsequence between generated and reference sequences.
- • rougeLsum: line-wise longest common subsequence for multi-line outputs.

- F Additional Experimental Results

### F.1 Planning Capability Analysis

We extended our evaluation to include models that were not part of the original benchmark, specifically Anthropic Claude variants (claude-3-7-sonnet, claude-4-sonnet) and GCP Gemini (gemi ni-2.5-pro). Model performance was evaluated using planning

accuracy metrics (BERTScore, ROUGE, and alignment with groundtruth plans), consistent with our original execution-accuracy leaderboard. We also analyzed planned step statistics to understand model behavior in generating task plans.

For each model, we compute a combined planning score 𝑆combined that integrates BERTScore (𝐵) and ROUGE-L (𝑅):

𝐵 + 𝑅 2

𝑆combined =

,

where 𝐵 ∈ [0, 1] is the average BERTScore between the modelgenerated plan and the ground-truth plan, and 𝑅 ∈ [0, 1] is the average ROUGE-L F1 score. This provides a balanced measure of both semantic similarity (via BERTScore) and sequence-level overlap (via ROUGE-L).

We summarize planning quality and plan length statistics across models in Tables 11 and 12, respectively. The following are a few key insights from the results.

• Top-performingmodels:mistral-medium-2505achieves

the highest planning accuracy (0.620 ± 0.063) and produces concise, low-variance plans (2.38 ± 1.04 steps), combining high performance with efficiency. gemini-2.5-pro is also highly competitive (0.615 ± 0.068).

[Figure 17]

- Figure 14: Google Forms: questionnaire to domain experts for human validation

Table 11: Planning accuracy (Avg ± Std) for all evaluated models.

Model Avg ± Std #Q mistral-medium-2505 0.620 ± 0.063 141 gemini-2.5-pro 0.615 ± 0.068 141 gpt-oss-120b 0.606 ± 0.077 141 mistral-small-3-1-24b 0.604 ± 0.062 141 claude-4-sonnet 0.595 ± 0.068 141

- llama-3-405b-instruct 0.588 ± 0.074 141 claude-3-7-sonnet 0.571 ± 0.071 141
- llama-4-maverick-17b 0.558 ± 0.071 141 gpt-5-2025-08-07 0.544 ± 0.092 141 granite-3-3-8b-instruct 0.529 ± 0.067 141 llama-3-3-70b-instruct 0.522 ± 0.068 141

- • Anthropic Claude models: Both claude-3-7-sonnet and claude-4-sonnet show solid planning accuracy with moderate plan lengths ( 2.5–3 steps) and low variance, indicating reliable reasoning and execution alignment.
- • Instruction tuning matters: Medium-sized instructiontuned models (mistral-medium, mistral-small) consistently produce efficient plans with low variance, outperforming larger models with longer, more variable plans (llama-3-3-70b, granite-3-3-8b).
- • Step length vs. performance: Shorter plans with low variance generally correlate with higher planning accuracy, while overly long plans may introduce redundancy without improving alignment with ground-truth executions.

#### Table 12: Planned step statistics for all evaluated models.

Model Avg steps ± Std Min Max Zero llama-3-405b-instruct 3.14 ± 1.84 1 9 0

- llama-3-3-70b-instruct 6.55 ± 1.55 3 12 0
- llama-4-maverick-17b 4.34 ± 1.80 1 9 0 granite-3-3-8b-instruct 5.56 ± 2.44 2 30 0 gpt-oss-120b 1.91 ± 1.21 1 10 0 mistral-medium-2505 2.38 ± 1.04 1 5 0 mistral-small-3-1-24b 2.77 ± 1.33 1 6 0

- claude-3-7-sonnet 3.10 ± 1.15 1 5 0 gpt-5-2025-08-07 2.33 ± 1.16 0 5 1 gemini-2.5-pro 1.87 ± 1.01 1 5 0
- claude-4-sonnet 2.45 ± 1.34 1 5 0

• Consistency vs. variability: High-variance models (gp t-5-2025-08-07, granite-3-3-8b) occasionally generate very long or empty plans, which may reduce reliability despite moderate average scores.

Conclusion: The unified analysis demonstrates that mediumsized, instruction-tuned models offer the best balance of planning accuracy and step efficiency, while the inclusion of Claude and Gemini models extends benchmark coverage and validates performance trends. These results are consistent with our original executionaccuracy leaderboard, confirming the robustness of the benchmark for evaluating reasoning-capable language models.

- F.2 Cost Analysis

Table 13 compares the estimated cost of running the full 140+ utterance task suite under the Agent-as-Tool paradigm. We report the average tokens consumed per task and the resulting total cost for each provider/model configuration.

Table 13: Runtime and estimated cost for executing 140+ utterance tasks using the Agent-as-Tool paradigm.

LLM Provider Avg. tokens Total cost (USD) gpt-4.1 OpenAI 3,664 $300 llama-4-maverick watsonx 3,730 $130

As shown in Table 13, token usage is comparable across the two models, but the total estimated cost differs substantially, reflecting provider pricing rather than task-specific token demand. In our setup, llama-4-maverick yields a lower total cost while maintaining similar average tokens per task, which can be advantageous when scaling evaluation runs or performing repeated ablations.

- F.3 Uncertainty Analysis

As discussed in Section 5.2, we run the evaluation agent five times to estimate the stability of rubric-based metrics. Table 14 reports inter-rater agreement across the five runs, together with the derived uncertainty, computed as 1 − agreement.

|{<br><br>"id": 105, "type": "FMSR", "deterministic": false, "characteristic_form": "the answer should contain a list of sensor names for asset wind turbine.", "text": "Provide some sensors of asset Wind Turbine.", "planning_steps": ["Provide some sensors of asset Wind Turbine."], "execution_steps": [<br><br>{"name": "get_available_sensor_information", "action": "Get Available Sensor Information",<br><br>"arguments": "Wind Turbine", "outputs": "[a list of sensor names]"}, {"name": "finish", "action": "Finish", "arguments": "", "outputs": ""}<br><br>], "execution_links": [{"source": "get_available_sensor_information", "target": "finish"}]<br><br>}|
|---|

Figure 15: An Example FMSR task specification and ground truth.

Table 14: Inter-rater agreement and derived uncertainty across five evaluation runs.

Metric Agreement Uncertainty

Task Completion 0.9731 2.69% Data Retrieval Accuracy 0.9697 3.03% Generalized Result Verification 0.9681 3.19%

Overall, agreement is consistently high (≥0.968), corresponding to uncertainties below 3.2%, which indicates that the LLM-as-judge evaluation is stable under repeated runs.

- G Ablation Studies

- G.1 Distractor Agents

We introduce 10 distractor agents to intentionally increase the complexity and ambiguity faced by a global agent. Table 15 summarizes these distractors by domain and role, spanning generalpurpose behaviors (e.g., echoing inputs, off-topic responses) and domain-specific capabilities (e.g., predictive maintenance, sensor summarization, edge ML deployment). This taxonomy increases the realism of multi-agent environments by supporting modular integration while injecting controlled confusion.

Across 99 scenarios, we compare model performance with and without distractor agents to evaluate robustness under noisy agentselection spaces. Table 16 reports rubric-based scores. gpt-4.1-2 025-04-14 remains the strongest overall, with only a small drop under distractors (e.g., task completion 52 → 48). Among openweight models, llama-4-maverick is the most robust and even improves with distractors (task completion 46 → 48, verification 46 → 49), whereas mistral-large and llama-4-scout degrade more noticeably. Interestingly, llama-3-405b-instruct improves across all three metrics, suggesting strong corrective behavior under ambiguity, while granite-3-3-8b-instruct remains stable but at a lower accuracy level.

### G.2 Impact of in-context examples

Table 17 compares gpt-4.1 and granite-3-3-8b with and without in-context examples on a subset of single-agent benchmark

tasks (𝑛 = 65). Consistent with our main findings, in-context examples are critical for enabling effective ReAct-style reasoning and coordination.

Removing in-context examples causes a large performance drop for both models: gpt-4.1 decreases from an average of 80% (with context) to 33% (without), while granite-3-3-8b falls from 60% to 3%. These results reinforce that in-context examples are essential for ReAct-style reasoning in LLM-based agents. We do not include work-order and end-to-end tasks here because their baseline performance is already low.

### G.3 Baseline using Single-Agent

Instead of using four domain-specific sub-agents and an orchestration agent, we build a tool-calling ReAct agent with a single prompt as a baseline, giving it access to tools and in-context examples from all agents. In doing so, we increase the complexity of the problem, as it must handle many tools as well as an expanded context. We run a default LLM, llama-4-maverick, on all 141 scenarios. As a single-agent baseline, it achieves task completion of 26.95%, data retrieval accuracy of 34.04%, and generalized result verification of 28.37%. Under the Agent-As-Tool setup, the same model achieves roughly two-fold improvements (See Figure 5a).

#### Table 15: Distractor agent types and their roles.

##### Agent Domain Role

EchoAgent General Repeats the input verbatim (debugging / coherence checks). OffTopicAgent General Returns unrelated facts when a query is off-topic or unrecognized. CustomerSupportAgent Support Ops Handles password resets, login issues, and service availability. SREAgent Site Reliability Diagnoses performance degradation, downtime, and infrastructure issues. FrontendDevAgent Software Eng. Assists with UI/UX, React, and rendering bugs. HRPolicyAgent Human Resources Answers policy questions (leave, benefits, compliance). SensorDataSummarizer Industrial IoT Summarizes sensor time series (trends, anomalies). HistoricalTrendsAgent Analytics Extracts historical patterns for failure analysis/optimization. EdgeMLAgent Edge Computing Recommends strategies for ML deployment on constrained hardware. RULPredictorAgent Predictive Maint. Estimates remaining useful life from sensor and degradation signals.

#### Table 16: Model performance with vs. without distractor agents (99 scenarios).

Model Setting Task Completion Data Accuracy Result Verification gpt-4.1-2025-04-14 Without distractors 52 57 55

With distractors 48 56 54 granite-3-3-8b-instruct Without distractors 40 44 41

With distractors 40 44 41 mistral-large Without distractors 42 46 43

With distractors 40 44 41

- llama-3-405b-instruct Without distractors 41 41 38 With distractors 44 44 44

- llama-3-3-70b-instruct Without distractors 38 43 34 With distractors 41 43 36

- llama-4-maverick Without distractors 46 49 46 With distractors 48 49 49

- llama-4-scout Without distractors 45 44 46 With distractors 40 40 40

Note: Higher is better. Scores are reported as percentages.

#### Table 17: Impact of in-context examples on single-agent tasks (𝑛 = 65). Scores are percentages.

Model Examples Completion Retrieval Verification gpt-4.1 Yes 52 57 55

No 22 21 24 granite-3-3-8b Yes 40 44 41

No 2 3 3

