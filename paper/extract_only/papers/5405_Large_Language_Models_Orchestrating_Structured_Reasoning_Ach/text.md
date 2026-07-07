# arXiv:2411.03562v3[cs.LG]15Sep2025

## Kolb-Based Experiential Learning for Generalist Agents with Human-Level Kaggle Data Science Performance

###### Antoine Grosnit1,3,†, Alexandre Maraval1,†, Refinath S N1, Zichao Zhao1,2, James Doran1, Giuseppe Paolo1, Albert Thomas1, Jonas Gonzalez1, Abhineet Kumar1, Khyati Khandelwal1, Abdelhakim Benechehab1, Hamza Cherkaoui1, Youssef Attia El-Hili1, Kun Shao1, Jianye Hao1, Jun Yao1, Balázs Kégl1,⋆, Haitham Bou-Ammar1,2,⋆, Jun Wang2,⋆

1 Huawei Noah’s Ark Lab 2 AI Centre, UCL 3 TU Darmstadt ⋆ Corresponding Authors, †Equal contributions

Abstract: Human expertise emerges through iterative cycles of interaction, reflection, and internal model updating, which are central to cognitive theories such as Kolb’s experiential learning and Vygotsky’s zone of proximal development. In contrast, current AI systems, particularly large language models (LLMs) agents, rely on static pretraining or rigid workflows and lack mechanisms for continual adaptation. Recent studies have identified early cognitive traits in LLM agents, including reflection, revision, and self-correction, which suggest foundational elements of human-like experiential learning. This leads to a key question: Can we design LLM agents capable of structured, cognitively grounded learning similar to human processes? Toaddressthis,weproposeacomputationalframeworkofKolb’slearningcyclewith Vygotsky’s ZPD for autonomous agents. Our architecture separates extrinsic functions(environmentinteraction)fromintrinsicfunctions(internalreflectionandabstraction), enabling cognitively grounded scaffolded learning, where the agent initially learns within structured, supportive environments, followed by open-ended generalisation. This approach empowers agents to master complex, many-step tasks ; domains that traditional fine-tuning or simple reflective methods could not tackle effectively. Its potential is powerfully demonstrated through direct competition with humans in real-worldKaggle data science challenges. Learningfully automated, end-to-end data science code generation across 81 tasks, our system, Agent K, demonstrated the ability to perform the entire workflow without human intervention, achieving an Elo-MMR score of 1694, placing it beyond median performance of the Kaggle Masters (the top 2% among over 200,000 users) included in our study. With 9 gold, 8 silver, and 12 bronze medals level performance – including 4 gold and 4 silver on prize-awarding competitions – Agent K is the first AI system to successfully integrate Kolb- and Vygotsky-inspired human cognitive learning, marking a major step toward generalist AI.

Asearlyas350BCE,Aristotleobservedthatwelearnnotbybeingtold, butbydoing: “For the things we have to learn before we can do them, we learn by doing them” (Nicomachean Ethics). From philosophy to cognitive science, this principle has remained central to our

understanding of human learning.

From the earliest days of AI, the field has aspired to create systems that learn from experience rather than rely on hand-crafted rules [19, 28]. This ambition has driven decades of progress, from developing neural networks to the rise of deep learning [16], where powerful modelsaretrainedonvastamountsofstaticdata. Theseadvanceshaveculminatedinlarge language models (LLMs) that exhibit remarkable generalisation and emergent reasoning abilities [2, 31], achieving behaviours once thought to be uniquely human. Yet these systems learn from experience that is mostly fixed in advance: despite involving pretraining, fine-tuning, and RLHF, these models depend on static, pre-collected data, mostly scraped from the internet. As global data sources saturate [1], the limits of this approach become clear.

The next frontier is to build experiential agents that can learn through experience they actively generate, notably by interacting with environments, reflecting on outcomes, and adapting their internal strategies over time. Reinforcement learning represents a step in this direction, enabling agents to master complex tasks through trial-and-error, with highprofile successes such as AlphaGo, AlphaZero, and MuZero [22, 25, 26]. These systems achieved superhuman performance, but primarilyin environmentsthatare well-specified, simulation-friendly, and governed by clearly defined objectives [24].

Aimingtoextendthesecapabilitiestomoregeneralandopen-endedsettings,recentadvances in LLMs have enabled a new class of agents that demonstrate the capacity for basic reflection and internal reasoning across diverse tasks. Often combining LLMs with reinforcement learning or programmatic feedback loops, methods such as ReAct, Reflexion, and Voyager [23, 30, 34] allow agents to reason about past actions, revise plans, and interact more flexibly within their environments. While these approaches represent important progress, they typically rely on prompt-level heuristics and lack a principled architecture for structured, long-term learning or internal strategy adaptation. Nonetheless, they reveal a critical shift: reflection, long regarded as a core mechanism in human cognition [10], is now emerging as a viable computational capability. Thus, a natural question arises: “Can we design agents that learn the way humans do—through structured cycles of experience, reflection, abstraction, and adaptation?”

To explore this, we draw on Kolb’s experiential learning theory [15], a foundational model in the cognitive sciences that describes learning as an iterative cycle comprising four stages to support the development of internal models: concrete experience, reflective observation, abstract conceptualization, and active experimentation. This framework has shaped educational theory and practice, emphasising that effective learning requires not only doing, but also structured internal reorganisation. This was demonstrated through empirical studies [21, 3, 17] following cohorts of students who experienced Kolb-cyclebasedinstructionalsequencesandshowedsignificantimprovementsonobjectivelearning measures. Complementing this, Vygotsky’s zone of proximal development (ZPD) [29] suggests that learners benefit most when guided through tasks just beyond their current ability, an idea that underpins modern approaches to scaffolding. Empirical studies across domains, from clinical training [35] to psychology education [14], show that embedding experiential learning within scaffolded environments enhances outcomes. With LLMs now exhibitingreasoningandself-reflectivecapabilities,thesefoundationaltheoriesofferatimely blueprint for computational models of agent experiential learning.

In this work, we propose a computational framework that implements Kolb’s experien-

tial learning cycle, enabling autonomous agents to effectively learn through experience. To structure progression, we incorporate Vygotsky’s ZPD, guiding agents from scaffolded stages toward open-ended tasks. Echoing Kolb’s alternation between action and reflection, wemodelagentlearningasacyclebetweenextrinsicandintrinsicfunctions. Extrinsicfunctions govern outward interaction, such as executing code, selecting actions, and gathering feedback. Intrinsic functions operate over the agent’s internal state, enabling it to reflect, abstract, hypothesise, and adapt its strategy. These components are modular and composable, allowing nested, multi-step reasoning and ongoing self-improvement.

Unlike traditional gradient-based approaches that rely on model parameters updates, ourframeworkenablesautonomousadaptationthroughinternalstatetransformations. By separating and dynamically interleaving internal cognition with external interaction, our systemoffersacomputationalanaloguetohumanexperientiallearning, supportingagents that do not merely react or act, but learn and evolve from their own experience effectively.

To test the hypothesis that modelling Kolb’s experiential learning cycle enables generalist intelligence, we evaluate our framework on Kaggle [13], the world’s leading platform forcompetitivedatascience. Kagglechallengescomprisehigh-stakes, real-worldproblems in domains like finance, healthcare, and climate science, where success demands not only technicalexpertise–suchasdatapreprocessing,featureengineering,andmodelselection– but also iterative refinement, strategic experimentation, and adaptability, all of which align with Kolb’s learning phases.

Unlike synthetic benchmarks, Kaggle competitions are designed for human experts and evaluated via public and private leaderboards, providing a rigorous test of generalisation. While previous automation efforts such as AutoML have focused on specific subtasks like hyperparameter tuning [6], they rely on fixed heuristics and struggle to generalise across different modalities. In contrast, our experiential agent autonomously manages the entire data science pipeline, from fetching Kaggle problems, building and refining solutions, to submitting its results to the platform. This clearly differs from earlier attempts on Kaggle that depend on offline datasets or partial automation [12, 4] and do not include direct comparisons against human participants on the official final leaderboard.

We argue that fully automatic Kaggle serves as a milestone environment for agents, akin to Atari [18] in deep reinforcement learning and Go [27] in multi-agent self-play. Just as those benchmarks demonstrated emergent planning and learning capabilities, Kaggle offers a rigorous testbed for measuring generality, adaptability, and human-level performance in autonomous data science.

We instantiate our framework in Agent K, a fully autonomous system that learns to construct and refine high-performance data science pipelines without human intervention. Across a broad range of Kaggle competitions, including tabular, computer vision, and natural language processing challenges, Agent K achieved performance at the level of experienced human data scientists. Its Elo-MMR places it on par with the median of Kaggle Masters, an elite group representing less than 2% of the platform’s 200,000+ users. In featuredandresearchcompetitionsgrantingKagglemedals,AgentKwouldhaveearned4gold and silver medals, and it demonstrated medal-equivalent performance (5 gold, 4 silver and 12 bronze) in many others. To our knowledge, this is the first demonstration of a fully autonomous agent achieving consistent, human-competitive results across the full data science pipeline in real-world environments, offering empirical evidence that a computationally grounded cycle of Kolb’s experiential learning can serve as a viable foundation for gen-

eralist AI.

## Computational Models of Kolb’s Experiential Learning

We now formalise our experiential learning framework by distinguishing between two core computationalroles: extrinsicfunctions,whichgoverntheagent’soutwardinteractionwith theenvironment(e.g.,selectingactions,receivingfeedback),andintrinsicfunctions,which operate over the agent’s internal state to support reflection, abstraction, and adaptation. Thesefunctionsarecomposableandcanbeappliediteratively,enablingstructuredinternal reasoning processes prior to action [5]. A key enabler of this framework is the use of LLMs as they naturally support open-ended inputs and outputs, which is a fundamental prerequisite for experiential learning in unstructured and dynamic environments. We present actual implementations of the extrinsic and intrinsic functions using LLM calls in Figures 8, and 9 of the Method section.

Interestingly, ReAct-like behaviour [23, 30, 34] – a widely used prompting strategy in which LLMs interleave reasoning (“thought”) with actions by reflecting on intermediate outcomes – can be seen as the fundamental cognitive primitive here: a single reflect–act loop that instantiates the minimal intrinsic–extrinsic cycle. By chaining multiple ReAct steps, our framework naturally generalises to the full Kolb cycle of repeated reflection, abstraction and experimentation (Figure 1). We notably show on data science problem solvingthatproperabstractionenablesReAct-basedagentstoachievebetterperformancewith half time budget (Figure 6). This two-phase structure mirrors the alternation in Kolb’s cycle between outward experimentation and inward conceptualisation, and provides a computational foundation for agents that learn through structured cycles of internal reorganisation and external engagement.

Specifically, the agent applies a composition of k intrinsic functions, denoted by It(k), to its internal state Σt, which corresponds to an internal memory or a summary of past experiences. This produces a new refined internal state Σ′t = It(k)(Σt). Once the intrinsic phase is complete, the agent interacts with the environment via an extrinsic function Et, which takes Σ′t as input and returns an action to apply to the environment to gather a new interaction outcome (e.g., observation, feedback, test feedback)Ft. The agent thenupdatesits internal state to Σt+1 using an update func-

Itk ◦ ··· ◦ It1

Reflective Observation Intrinsic Functions Abstract Conceptualisation Intrinsic Functions

[Figure 1]

Et (Σ′t)

Σt

Extrinsic Function Active Expert

Internal Agent State

Ut (Σ′t,Ft)

Ft

Environment Feedback

Internal Agent State Update

Figure 1. Our computational formalisation of Kolb’s experiential learning theory.

tion Ut, such that: Σt+1 = Ut(Σ′t,Ft). This formalisation preserves the alternation at the heart of Kolb’s theory, enabling the agent to reflect, abstract, and adapt before engaging with the world, thus forming a computationally grounded cycle of experiential learning as depicted in Figure 1.

Now that we have introduced a computational framework for Kolb’s experiential learning theory, we turn our attention to formalising Vygotsky-inspired scaffolded learning,

Beyond ZPD

###### Agent’s ZPD - Scaffolding Learning

###### Feedback

###### Solution Scaffold

Kolb-Like Experiential Learning

Structuring ML model production

[Figure 2]

Internal State

Extrinsic Functions

Intrinsic Functions

[Figure 3]

Tools... ...

Models Compute... ...

[Figure 4]

... ...

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Intrinsic Functions

Extrinsic Functions

Internal State

[Figure 9]

[Figure 10]

Pre-processing

Optimisation

Feedback

Modelling

###### Scaffold-Guided Scaffold-Free Learning

...

[Figure 11]

###### Workspace Scaffold

...

Structuring Raw Data ZPD

.

###### Guide

Unit Test

Pass Pass

2nd Scaffold Task

###### : Abstract

...

maps metric code preprocessing

Solution 1.

###### LLM

- Solution 2.
- Solution 3.
- Solution 4.

###### Scaffold-CoTs

[Figure 12]

[Figure 13]

(Prompt to summarise)

Task’s URL

1st Scaffold Task

...

##### ZPD Experience

It & Et Abstraction & External action

It1(Σt) Analyse

It2(It1(Σt)) Plan

Prompt (Abstract Scaffold CoT, Plan & Write code)

Prompt (e.g. Error Analysis)

Prompt (e.g. New Planning)

Task info: 〈task_summary〉, 〈data_descr〉, ...

Σt Internal State

Task info: 〈task_info〉 This was your code: ⟨codek⟩ Explain how to fix ⟨errork⟩, given former guidance: {⟨error_analysisi⟩}ki=−11

Task info: 〈task_info〉 You wrote ⟨codek⟩ and got ⟨errork⟩. Given this error and former guidance {⟨error_analysisi⟩}ki=1, write a plan...

Past submissions attempts: {⟨scaffold_coti⟩}ki=1. Generate a plan followed by a solution in python.





〈task_info〉, ⟨plan1⟩,⟨code1⟩,⟨error1⟩, ⟨error_analysis1⟩

Σt

LLM Answer

Internal State

LLM Answer





“A first attempt to tackle this ... ⟨plan⟩ ```python ⟨code⟩ ```”

LLM Answer





To avoid this error, ⟨error_analysisk⟩

〈task_summary〉, 〈data_descr〉, ...

To solve this, I should ⟨plank+1⟩

..., ⟨plank⟩,⟨codek⟩,⟨errork⟩

It1(Σt) = Σt ∪ {⟨error_analysisk⟩}









Σ′t = It1(Σt) ∪ {⟨plank+1⟩}

zt := Et(It(Σt)) = ⟨code⟩

{⟨scaffold_coti⟩}ki=1 , 〈submit_pool : ∅〉

Experiential Learning Loop

Et(Σ′





###### t)

t,Ft)

Ut(Σ′

External action

Exp. Learning Loop

Update the state

t,Ft)

Prompt (e.g. Implement)

Ft(Tt,zt)

Ut(Σ′

Ft(Tt,zt) Get env. feedback

Update (e.g. Store Elements)

Task info: 〈task_info〉 You wrote ⟨codek⟩ and got ⟨errork⟩. Considering {⟨error_analysisi⟩}ki=1, implement a solution given ⟨plank+1⟩

Internal State Update

Get env. feedback

Σt+1 := Σ′t ∪ {⟨codek+1⟩,⟨errork+1⟩}

Update (e.g. Add solution to the pool)

Env. (e.g. Run zt in a Terminal)

###### Env. (e.g. Execute ⟨code⟩)

- • Σt+1 ← Σ′t
- • Σt+1[〈submit_pool〉] ← {⟨code⟩,⟨run_log⟩}

user@machine:~$ python script.py Traceback [...] ValueError: ’NaN’ encountered

user@agk:~$ python script.py Training model...

LLM Answer

. Validation MSE: 0.324

```python ⟨codek+1⟩ ```

Ft = {⟨codek+1⟩,⟨errork+1⟩}

Ft := Ft(Tt,zt) = {⟨run_log⟩}

zt := Et(Σ′t) = ⟨codek+1⟩

Figure 2. From Scaffolded Experiential Learning to Autonomous Generalisation. The top part of the figure shows how an autonomous agent progresses from scaffolded learning tasks within its Zone of Proximal Development (ZPD) to open-ended problem solving. In the scaffolded environment (on top left), the agent generates solutions though structured tasks gated by success and supported by feedback. As the agent masters scaffolded tasks, it internalises strategies into Scaffold-CoTs – realised through LLM summarisation in our setup. In the open-ended environment (on the top right), the scaffold is removed, and the abstracted knowledge supports self-directed adaptation to increase the likelihood of success. Learning in both regimes follows our computational model of Kolb’s experiential learning cycle: concrete interaction with the environment (extrinsic functions), reflective observation and internal strategy formation (intrinsic functions), and active experimentation based on revised hypotheses. The two bottom graphs illustrate this experiential learning process via prompt-based intrinsic and extrinsic functions. The left graph displays an experiential learning loop for error solving during scaffold, while the right loop shows how the agent abstracts scaffold-CoTs to generate open-ended solutions.

whichunderpinstheagent’sdevelopmentaltrajectory. Together, theaforementionedKolbbased computational framework and the guided progression through increasingly challenging environments define the core of our approach.

## Scaffolded Learning and the Agent’s ZPD

Inspired by Vygotsky’s Zone of Proximal Development (ZPD) [29], we introduce the concept of the Agent’s ZPD as the range of task complexity where an agent cannot yet succeed autonomously, but can succeed with appropriate scaffolded support. This region defines the agent’s learning frontier: the space where internal adaptation is still possible, provided that the environment offers the right structure, feedback, or constraints. Just as human learners grow most effectively when challenged, agents benefit from carefully structured experiences that push their boundaries without overwhelming them.

To make the above concept concrete, we define scaffolded learning as a structured progressionoverasetoftasksorenvironmentsT = {T1,T2,...,Tn},whicharestagedtoloosely reflect the natural workflow a practitioner might follow when approaching the problem domain. These correspond to the scaffolded subtasks shown on the left-hand side of Figure 2 labelled “Agent’s ZPD - Scaffolding Learning” and “Beyond ZPD”. Each task Ti ∈ T is designed to target a specific capability or reasoning skill, and to build on knowledge acquired in earlier stages incrementally. This staged progression resembles curriculum learning, in that transitions are competency-gated and heuristically adjusted [11, 32]. The agent advances only by satisfying explicit success criteria (e.g., passing unit tests), making progress contingent on demonstrated ability. This structure aligns more closely with the ZPD, where support is withdrawn as internal competence emerges.

Additionally, we introduce a conceptual feedback function F (Ti, Σt) that evaluates the agent’sperformanceontaskTi givenitsinternalstateΣt. Thisfeedbackmaytaketheformof explicit performance signals, such as scalar rewards (as in reinforcement learning), binary success indicators, orricher environmental responses, andservestoguideinternaladaptation throughout thelearning cycle. As the agent progressesthrough T , it iterativelyupdates its internal state through experiential interaction, enabling performance on increasingly complex environments without direct supervision or fine-tuning. This formal structure allows us to represent scaffolding as a trajectory through a task space structured by cognitive dependencies between tasks, such as needing to align data modalities before attempting predictive modelling in data science.

After completing the scaffolded stages, the agent transitions into a fully autonomous, open-ended learning phase; see the right-hand side of Figure 2. To initiate this process, the agent consolidates its prior scaffolded experience by abstracting patterns from previously constructed pipelines within the scaffold. These past solutions, generated through structured experiential learning, are internally abstracted into chain-of-thought traces that capture summaries of successful reasoning steps, component structure, and validation logic. Interestingly, these traces do not merely support reasoning in similar future tasks. They function as autonomous, agent-generated cognitive scaffolds, effectively bootstrapping a chain-of-thought process in the absence of external guidance.

In the open-ended setting, the agent reuses these internalised traces as cognitive scaffolds: they guide hypothesis formation, code synthesis, and self-evaluation in tasks where

noexternalstructureisprovided. Fromthispointonward, theagentcontinuesapplyingexperiential learning principles independently, completing the full cycle of action, reflection, abstraction, and adaptation. This developmental path parallels that of human learners, whofirstacquireskillswithinstructuredenvironmentsandlaterapplythemautonomously once scaffolds are removed [20, 33].

## Agent K: Integrating Kolb’s and Vygotsky’s Principles to Master Kaggle Competitions

We instantiate our framework in Agent K, the first effective AI agent to tackle autonomous data science, which is the end-to-end process of generating high-performing solutions from raw datasets and natural language problem descriptions, without human intervention. This setting combines strategic reasoning, iterative experimentation, and abstraction across heterogeneous data types, including tabular data, computer vision, natural language, and evenmulti-modal domains. WegroundourworkinKagglecompetitions, which require not only technical skills but also strategic generalisation across a wide range of domains. Kaggle presents a particularly challenging benchmark due to a combination of concretetechnicaldifficulties: thediverseandoftenlooselydocumentedfilestructuresofcompetitiondatasets,thehighriskofoverfittingtosmallpublicleaderboardsplits,andtheneed to match or outperform expert human data scientists who leverage advanced ensembling and domain-specific modelling techniques. These factors make Kaggle a uniquely rigorous testbed for real-world, autonomous data science.

#### Scaffolded Data Science Environments

We begin by designing the ZPD of Agent K, defining a structured learning environment where the agent can succeed with scaffolded support, but not yet independently. This scaffolded phase forms the first stage of Agent K’s developmental arc. It prepares the agent to later operate autonomously in open-ended settings (generating end-to-end data science workflows), where external guidance is removed. Just as a teacher structures learning to progressively build students’ capabilities, our scaffolded environment guides Agent K throughstagedcomponentsofthedatascienceworkflow. Thissettingallowsexploration, hypothesis formation, and skill acquisition, enabling the agent to learn through structured experience.

Ourscaffoldedenvironmentmirrorsthehumandatascienceprocess: first, understanding the task and structuring a workspace, then solving the problem. Reflecting this progression, the environment is organised into two phases, as shown in Figure 2: an initial setup phase focused on data abstraction and exploration, and a solution-building phase focused on modelling and strategic optimisation.

While our scaffolded environment offers structure, navigating it remains a non-trivial challenge. Agent K receives only high-level task descriptions or templates (not detailed implementations), and must independently determine how to construct each component. For example, it may recognise the need for a performance metric or submission interface, but must devise the logic and implementation itself. This mirrors how human learners are often given structured guidance while still needing to solve problems through reasoning,

experimentation, and adaptation.

###### Stage I: Workspace Scaffold

Thefirstphaseofourenvironment(denotedby“WorkspaceScaffoldinFigure2)mirrorsthe early, often ambiguous steps human data scientists take: transforming potentially messy, real-world inputs into structured workspaces. For Agent K, this requires inferring meaningful abstractions from diverse inputs (i.e. text, images, and tables) and adapting to varied outputs, from classification to regression and ranking. Even within a single task type, the agent must reason about output semantics (e.g., class probabilities vs. hard labels) and align input-output mappings accordingly. This stage challenges the agent to develop a functional understanding of the task from minimal supervision, enabling it to transform raw, heterogeneous inputs into a coherent, structured workspace suitable for downstream modelling.

To generalise across diverse data science tasks, Agent K must learn to construct unified representations from inconsistent inputs, typically comprising a labelled training set and an unlabelled test set, distributed across multiple modalities. Here, the agent autonomously generates code to align inputs with expected outputs and define task-specific evaluation criteria. These components include input–output mappings, transformation routines, and formatting logic for predictions; see Figure 7 in the Methods section for implementation details.

Agent K advances through scaffolded learning stages only when its generated solutions satisfy general-purpose tests (e.g., pass/fail validations or execution traces), provided by the environment as feedback signals. These constraints validate properties such as data alignment, execution correctness, and inter-component consistency. When a stage fails, the agent revises its internal strategy and attempts the step again. The environment evaluates not just isolated components but their combined behaviour across multiple stages, allowing for recursive correction and consolidation. This mechanism supports learning through trial, failure, reflection, and abstraction, mirroring experiential learning cycles in structured problem-solving environments.

###### Stage II: Solution Generation Scaffold

With the workspace constructed, the environment transitions to the second phase: solution generation (see the “Solution Scaffold” part in Figure 2). In this stage, Agent K takes autonomous steps toward building a complete task-specific solution: designing modelling strategies, engineering features, training models, and refining performance through iteration. This stage supports experimentation by allowing the agent to explore multiple approaches, revise underperforming solutions, and learn from feedback signals that emerge through training and evaluation.

The environment exposes Agent K to reference patterns drawn from common practices in modern data science, such as feature encoders, hyperparameter tuning strategies, and domain-specific model families, as further explained in the Methods section. However, the agent does not reuse these as static templates. Instead, it must interpret their structure, adapt them to the task context, and implement viable solutions in code.

#### Experiential Learning in Agent K

Agent K’s learning process is grounded in the same experiential learning loop, inspired by Kolb’stheoryasdetailedbefore. Thisloopalternatesbetweenextrinsicfunctions,wherethe agent interacts with its environment, and intrinsic functions, where it reflects, abstracts, and adapts its internal strategies, as summarised in Table 1 in the Methods section.

ScaffoldedIntrinsicFunctions: Inthescaffoldedsetting, intrinsicfunctionsaretriggered by feedback such as unit test failures or low validation scores. The agent uses LLM-based reasoning loops to reflect on these signals, identify the source of failure, and revise its internal plans. These cycles may involve summarising and abstracting console logs, identifying likely bugs, or proposing new solution strategies, all performed autonomously through iterative prompt completions, as we show in Figure 8.

Open-Ended Intrinsic Functions In the open-ended phase, the agent operates without structural constraints, aiming to autonomously generate complete data science solutions from raw inputs. While it continues to follow Kolb’s experiential learning loop, still alternating between action, intrinsic processes, and adaptation, it now builds on the internal knowledge acquired during the scaffolded phase. Specifically, intrinsic functions are enhanced with LLM-based summarisation/abstraction mechanisms that distil prior experiences into high-level conceptual traces. These distilled summaries are repurposed as chain-of-thought prompts, guiding hypothesis generation, strategy formation, and iterative debugging.

The agent begins by using these chain-of-thought prompts to propose an initial set of candidate solutions. Each candidate forms the root of a dynamically constructed tree of code, where each node represents a fully executable data science pipeline. After execution, the agent evaluates feedback, such as validation scores or runtime errors, and decides how to evolve the tree. It may refine an existing node to fix bugs or generate a new variant to improve performance.

This approach goes beyond simple ReAct-style agents such as AIDE [12], which rely solely on LLM pretraining and generic ReAct loops. By contrast, Agent K leverages scaffoldderived knowledge to seed its reasoning with abstractions from domain-specific insights, resulting in more focused exploration and substantially improved performance. Rather than improvising from scratch, the agent bootstraps its learning from prior conceptualisations, demonstrating how structured experiential learning can scale to autonomous generalisation in complex, unconstrained settings.

## Quantitative Results

We evaluated Agent K on 81 real-world Kaggle competitions spanning tabular (55%), computer vision (24%), natural language (10%), and multimodal (11%) tasks. Unlike benchmarks that target isolated aspects of data science (e.g., tabular-only tasks [9] or hyperparametertuning[8]),ourbenchmarktestsend-to-endgeneralisationacrossthefullpipeline. It enables cross-domain evaluation using standardised Kaggle leaderboard submissions,

assessing both autonomy and predictive performance in practical, real-world settings, surpassing the scope and fidelity of prior benchmarks.

To ensure a meaningful evaluation, we selected Kaggle competitions with high human participation, averaging over 4000 participants in tabular tasks, 1200 in NLP, and 1000 in multimodal domains. The benchmark includes a balanced mix of accessible and challenging tasks, ranging from Kaggle Playground competitions to high-stakes featured and research challenges, which are widely regarded as the platform’s most competitive and demanding. We evaluated Agent K under the same conditions as human Kaggle participants. It interacts with the Kaggle API to submit predictions and is ranked on the private leaderboard, enabling direct, transparent comparison with both human data scientists and existing automated systems.

In addition to reporting performance quantiles, we evaluated whether Agent K would earn gold, silver, or bronze medals using Kaggle’s official criteria. Following standard benchmarking practice [4, 12], we computed medals even for competitions that did not officially award them. However, we clearly distinguish between official and inferred medals in our reporting and apply the same rules to human participants to ensure a fair and transparent comparison.

#### Agent K’s Medal Performance

Figure3summarisesAgentK’sperformanceacrossKaggle’sprivateleaderboards. Theagent earned the equivalent of four gold and four silver in real medal-awarding competitions spanning tabular, computer vision, natural language, and multimodal tasks. These medals were awarded in challenges with up to 5000+ participants and prize pools as high as $65000 underscoring both the competitiveness and practical difficulty of the tasks. For example, Agent K achieved gold in “Galaxy Zoo” (computer vision), “Give Me Some Credit” (tabular), and in the multimodal challenge “Stumble Upon”.

Beyond the featured competitions, Agent K achieved medal-equivalent rankings in a broad set of non-medal-awarding tasks, earning five gold, four silver, and twelve bronze equivalents across tabular, computer vision, and natural language domains. These results further demonstrate its versatility and generalisation across diverse modalities. Notably, these competitions included large-scale challenges such as the “Sentiment Analysis on Movie Reviews” (NLP; 1,011 participants) and “House Prices for ML Course” (tabular; 6,999 participants). A full breakdown is provided in Extended Data in the Methods section.

Taken together, these results show that Agent K can compete at a high level, earning medals in featured Kaggle competitions with substantial prize pools and large participant pools. Beyond these, it achieves strong, medal-equivalent rankings across a wide range of additional challenges. Notably, Agent K demonstrates versatility not only within individual domains but also across the full spectrum of data science tasks, including tabular data, computer vision, natural language processing, and multimodal problems. This breadth of performance provides compelling evidence of its general capabilities.

But a central question remains: how does Agent K compare to human data scientists? We turn to this next.

Post-scaffold improvement

playground-series-s3e5

avito-prohibited-content

dont-overfit-ii

scrabble-player-rating

Post-scaffold overfitting

ventilator-pressure-prediction

nitro-language-processing-3

playground-series-s3e22

sp-society-camera-model-identification

statoil-iceberg-classifier-challenge

human-protein-atlas-image-classification

cat-in-the-dat-ii

humpback-whale-identification

playground-series-s3e19

planttraits2024

tabular-playground-series-jan-2022

tabular-playground-series-may-2022

dog-breed-identification

playground-series-s4e5

cat-in-the-dat

plant-pathology-2020-fgvc7

histopathologic-cancer-detection

kaggle-pog-series-s01e03

tabular-playground-series-aug-2022

leaf-classification

tmdb-box-office-prediction

sberbank-russian-housing-market

noaa-right-whale-recognition

tabular-playground-series-apr-2021

whereami

playground-series-s3e3

tabular-playground-series-sep-2022

jigsaw-toxic-comment-classification-challenge

porto-seguro-safe-driver-prediction

playground-series-s3e25

playground-series-s3e1

CompetitionID

playground-series-s3e16

playground-series-s3e11

bike-sharing-demand

mercedes-benz-greener-manufacturing

tabular-playground-series-aug-2021

tabular-playground-series-feb-2021

forest-cover-type-prediction

plant-seedlings-classification

tabular-playground-series-jan-2021

playground-series-s3e7

- playground-series-s3e24
- playground-series-s4e2

playground-series-s4e1

otto-group-product-classification-challenge

playground-series-s5e5

playground-series-s3e14

h2oai-predict-the-llm

nitro-language-processing-2

nlp1000-ml-challenge

ml-olympiad-landscape-image-classification

goodreads-books-reviews-290312

vehicle

invasive-species-monitoring

nlp-getting-started

- playground-series-s3e26

dogs-vs-cats-redux-kernels-edition

- playground-series-s4e4

sf-crime

playground-series-s3e9

5-flowers-image-classification

predict-who-is-more-influential-in-a-social-network

nlpsci

diabetic-retinopathy-detection

santander-customer-satisfaction

state-farm-distracted-driver-detection

restaurant-revenue-prediction

home-data-for-ml-course

world-championship-2023-embryo-classification

galaxy-zoo-the-galaxy-challenge

playground-series-s3e15

bioresponse

GiveMeSomeCredit

stumbleupon

sentiment-analysis-on-movie-reviews

sbu-ai-finalproject

sign-language-image-classification

0 20 40 60 80 100

Quantile (higher the better)

Figure3. AgentK’sperformanceacrossKagglecompetitionsspanningtabular, computervision, NLP, and multimodal tasks. The y-axis lists competition IDs; the x-axis shows quantile performance on the private leaderboard (higher is better). Bars with darker shading correspond to Kaggle competitions that granted actual medals as featured or research competitions. On these, Agent K would earn 4 gold and 4 silver medals, and achieve the equivalent of 5 gold, 4 silver, and 12 bronze medals in others.

#### Agent K versus Human Data Scientists

TocompareAgentK’sperformancetohumandatascientists,weapplyamultiplayerElorating system following the approach in [7]. Elo provides a principled way to compare agents

###### Kaggle Level

Agent K (1694) 83rd percentile

Base

Expert Master Grandmaster

###### Q1 Q2 Q3

200

150

100

Count

50

0

Base

Expert

Master

Grandmaster

500 1000 1500 2000 2500

Elo-MMR Score

Figure 4. Comparison of Agent K’s Elo-MMR score with that of human participants. The top plot shows the Elo-MMR distribution of Kaggle users who participated in at least three of the same competitions (7,311 in total). Agent K ranks within the top 18% of this group. Bar colours reflect users’ Kaggle levels at the time of writing. The lower panel breaks down Elo-MMR scores by Kaggle level.

across tasks of varying difficulty and participant pools by modelling performance as a series of head-to-head matchups. It is widely usedfor rankingin competitive settingsand has been adopted by large-scale platforms such as CodeChef, reinforcing its relevance as a robust and interpretable benchmark for AI systems. We identify participants who competed inatleastthreeofthesamecompetitionsasAgentK,yieldingapoolof7,311“active”human competitors. This group spans a diverse range of Kaggle skill levels, from base Kagglers to competition Grandmasters (161).

The histograminFigure4 summarises those results. Thex-axisrepresentstheElo-MMR scores, while the y-axis shows the number of participants at each score level. The Elo-MMR scoresfollowanapproximatelynormaldistribution,peakingbetween1400and1500,where most participants are concentrated. A red dashed line marks our agent’s Elo-MMR score at 1694, placing it in the top 18th percentile. This means our agent outperforms about 82% of the 7,311 participants in the dataset. The lower section of Figure 4 presents a more detailed viewofElo-MMRdistributionsforparticipantsgroupedbyKagglelevels, fromBaseKaggler to Grandmasters. At the intersection of the red line and these distributions, we see that Agent K’s Elo-MMR score falls slightly beyond the median score achieved by Master-level participants.

#### Agent K’s Experiential Learning in Competitive Contexts

Having established that Agent K achieves human-competitive performance across a diverse set of real-world challenges, we now evaluate how it compares to AI agents that do not rely on structured learning cycles like Kolb’s, but instead leverage the emergent reactive

Over 70 tasks

100

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

80

Quantiles

60

40

20

ReAct (Qwen2.5-72b) 3 G | 2 S | 4 B

ReAct + RAG (Qwen2.5-72b) 5 G | 7 S | 4 B

ReAct (Deepseek-R1) 5 G | 6 S | 9 B

Agent K Scaffold (Qwen2.5-72b) 7 G | 8 S | 11 B

Agent K (Qwen2.5-72b) 8 G | 7 S | 11 B

0.3 0.4 0.5 0.6 0.7 0.8 0.9

ReAct (Qwen2.5-72b) [0.386] ReAct + RAG (Qwen2.5-72b) [0.500]

[0.763] Agent K (Qwen2.5-72b)

[0.714] Agent K Scaffold (Qwen2.5-72b)

[0.637] ReAct (Deepseek-R1)

All Tabular Tasks Classification Only

100

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

80

Quantiles

60

40

20

TabPFN-v2 (0-shot) 2 G | 0 S | 1 B

TabPFN-v2 (Fine-tuned) 2 G | 0 S | 1 B

TabPFN-v2-Ext. (0-shot) 2 G | 0 S | 1 B

Agent K 3 G | 4 S | 3 B

TabICL 0 G | 0 S | 0 B

Agent K 0 G | 1 S | 2 B

0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0.5 0.6 0.7 0.8 0.9 1.0

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

TabPFN-v2 (0-shot) [0.438] TabPFN-v2 (Fine-tuned) [0.476]

[0.893] Agent K

[0.693] TabPFN-v2-Ext. (0-shot)

TabICL [0.583] [0.917] Agent K

- Figure 5. Performance Comparison of Agent K versus Competing Agents and Foundational Models. (Top Row). We compare Agent K to three ReACT-style agents: ReACT (Qwen), ReACT (Qwen) with RAG, and ReACT (DeepSeek-R1). We also include Agent K (Scaffold Only), which is limited to scaffolded learning environments and does not support open-ended generalisation. We show for each method the distribution of the performance quantiles and the number of medals it achieves, as well as a critical difference diagram among each group of methods. The full Agent K achieves the highest median performance (near the 83th percentile) and the strongest medal-equivalent record across over 69 tasks: 3 gold (G), 4 silver (S), and 3 bronze (B). (Bottom Row). We benchmark Agent K against a familyofTabPFN-v2models, includingbothzero-shotandfine-tunedvariants. AgentK consistentlyoutperformsallTabPFN-v2baselinesonreal-worldtabulartasks,wherethestrongest baseline achieves a 30% median and only 2 gold, 1 silver, and 2 bronze medals. For classification tasks, we additionally compare against TabICL, a long-context in-context learning variant of TabPFN-v2, which performs notably worse than Agent K. These results demonstrate that Agent K’s structured experiential learning architecture enables broad generalisation and competitive performance across diverse data science domains.

properties of LLM.

InthetoprowofFigure5, wecompareAgentK againstseveralvariants: ReAct(equipped with Qwen), RAG-augmented ReAct (dubbed ReAct (Qwen) + RAG), and ReAct (equipped with DeepSeek-R1), as well as a variant of Agent K limited to scaffolded learning only. ReAct-based agents serve as meaningful baselines since they share our goal of leveraging LLMs for autonomous reasoning, but without a principled learning architecture grounded in cognitive theory. We use AIDE [4, 12] implementation for the ReAct-based agents, employing the same tree-based exploration strategy used by Agent K in the post-scaffold phase. The ReAct (Qwen) + RAG variant retrieves relevant Kaggle notebook or discussion elements to guide its initial solutions generation. Comparing to this baseline allows us to address the natural question of whether the solutions obtained through our scaffolded setting could simply be retrieved rather than discovered experientially. On the other hand, ReAct (DeepSeek-R1) provides a strong open-source foundation model baseline, allowing ustoaskwhetherAgentK’sperformancestemsfromitsstructuredlearningprocessorfrom the choice of the backbone language model.

Agent K outperforms all competing agents, achieving the highest median performance across 69 tasks. Its structured alternation between extrinsic functions (task execution and environment interaction) and intrinsic functions (reflection, abstraction, and strategy formation) enables better generalisation and learning efficiency. Importantly, Agent K surpasses reactive agents, suggesting that principled, cognitively inspired agent design, grounded in Kolb’s learning cycle and scaffolded within a zone of proximal development, can offer substantial advantages over purely reactive, emergent strategies.

Agent K vs. TabPFN-v2: While Agent K performs strongly across a wide range of tasks, it is important to assess how it compares to domain-specialised models designed and optimised for specific data modalities. The bottom row of Figure 5 presents this comparison. We evaluate Agent K against multiple variants of TabPFN-v2, including zero-shot, fine-tuned, and long-context versions. Despite being a generalist system, Agent K outperforms all TabPFN-v2 baselines, achieving a median performance near the 70th percentile and earning 3 gold (G), 4 silver (S), and 3 bronze (B) medal-equivalent scores across the evaluation set. In contrast, the strongest variant among TabPFN-v2 baselines achieves a median of 30% and collects only 2 gold, 1 silver, and 2 bronze medals. These results highlight that Agent K’s structured experiential learning not only generalises across domains but also competes with or surpasses task-specific state-of-the-art models in their domain, without any direct supervision or handcrafted features.

Abstract to Act, Not Just ReAct: Agent K achieves strong results in medal count and human-level performance; we now examine how scaffolded abstraction and ZPD-driven summarisationsupportitssuccessinopen-endedtasksthroughadedicatedablationstudy on the use of abstract conceptualisation. In these ablation experiments, agents tackle data science problems from scratch by growing a tree of solutions, where each node represents a complete code attempt, and subsequent steps involve debugging or improving past nodes. We compare Agent K’s strategy, which invests time in the scaffold to develop initial viable solutions and then abstracts and summarises them as CoTs for open-ended tasks, to the ReAct-based agent equipped with the same LLM as Agent K (Qwen), and that runs without

100

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

60

+ 4.8% in perf. @ 2x less time

80

AveragePrivateQuantile

###### Quantiles

50

60

40

40

30

20

20

ReAct Agent (w/o Conceptualisation) 3 Gold | 2 Silver | 4 Bronze

ReAct Agent (w/ Abstract Conceptualisation) 4 Gold | 2 Silver | 6 Bronze

10

ReAct Agent w/o Conceptualisation ReAct Agent w/ Abstract Conceptualisation

0

0.7 0.8

1 time unit 2 time units 3 time units 4 time units

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

Time Since Start

ReAct Agent (w/o Conceptualisation) [0.717] [0.783] ReAct Agent (w/ Abstract Conceptualisation)

(a) Impact of abstract conceptualisation on performance.

(b)FinalPerformanceofReActAgentswith and without abstract conceptualisation.

- Figure 6. Ablation Study on the Impact of Abstract Conceptualisation. A plot comparing average quantile performance in open-ended data science tasks of Agent K (with abstraction and summarisation from scaffold) versus shallow ReAct strategies. Since Agent K already spent half of the total runtime to build its CoT in the scaffold, the ReAct-based agent without abstraction was provided with this total runtime to enable a fair comparison. We can see that ZPD abstractions allow Agent K to achieve 5% quantile improvements, while requiring 2x less exploration time. On the x-axis, one “time unit” corresponds to 12 hours for tabular competitions, and 24 hours for CV, NLP or multimodal competitions.

the abstraction phase. To ensure a fair comparison, the ReAct Agent (Qwen) without abstraction was given additional exploration time to match Agent K’s total runtime, including its scaffold phase.

InFigure6a,weplottheaveragequantileperformanceinopen-endeddatasciencetasks achieved by the ReAct-based agent with abstraction and summarisation from scaffold, versus the shallow ReAct strategies. We see that ZPD abstractions lead to 5% quantile improvements, while requiring twice as little exploration time. Moreover, after 2 time units, we see that ReAct-based agent with abstract conceptualisation achieves an average of 19% higher quantiles than its shallow counterpart. Conducting a Welch t-test on the two groups of leaderboard quantiles obtained with and without abstraction, we get a p-value of 1.67e10−4 after2timeunits,meaningthataveragequantilesarestatisticallydifferent. WhendoinganotherWelcht-testontheresultsachievedafter2timeunits(resp. 4timeunits)fortheReActbased agent with (resp. without) abstract conceptualisation, the p-value is 0.26, which does not allow to reject the hypothesis of equal means. Moreover, we display in Figure 6b the distribution of final performance quantiles and the number of medals achieved with and without starting from abstract conceptualisation. This shows that abstract conceptualisation notably enables the ReAct-based agent to obtain one extra gold and two extra bronze medals over the 69 competitions for which both agents made at least one valid submission.

## Discussion

This paper introduced Agent K, the first LLM-based agent to implement Kolb’s theory and Vygotsky’s principle to achieve high-performance results on a wide range of data science tasks, including tabular data, computer vision, natural language processing, and multimodal Kaggle challenges. Our agent operates fully autonomously, seamlessly handling everything from navigating a URL to building models, making submissions, and generating high-score submissions. While our results are successful and Agent K achieves new stateof-the-art performance of data science agents, we still want to highlight four potential limitations of our work.

###### i) Use of recent technology compared to the competition release date. Since for each

Kaggle competition in our benchmark we compare against human participants who developed their solutions during the competition’s active period, we acknowledge that Agent K may benefit from using more recent technology that was not available to human competitors at the time. For example, nothing prevents Agent K from generating a code using a Vision Transformer (ViT) architecture such as MaxViT pretrained on ImageNet, which was released in September 2022, to solve a competition that ended months or years before. To partially control for this factor, we report in Figure 11 (Methods) the number of medals that Agent K would have achieved across our competition pool after excluding submissions that relied on models whose public release dates postdate the competition deadline. While this restriction reduces its performance, Agent K still secures a substantial number of medals, getting 19 out of the 29 medals obtained without this constraint. We emphasizethatthis“technologyreleasedate”discrepancyonlyaffectscomparisonswithhistorical human leaderboards and does not impact comparisons with baseline agents or foundation models, which are evaluated under the same technological conditions.

###### ii) Risk of solution memorization. While using newer technology can inflate perfor-

mance relative to human competitors, a separate risk is that Agent K might simply reproduce solutions from top public kernels or repositories available online, which the underlying LLM would have seen during its training. In such a scenario, high performance could result from memorization rather than genuine problem-solving ability. We believe this risk isminimalattheAgentK scaffoldinglevel: theagent’ssolution-generationprocessinvolves significantadaptation,multi-stepreasoning,andintegrationofdiversetools,makingdirect copying unlikely. Moreover, we observe that Agent K still performs strongly in several competitions that ended after the cut-off date of the most recent LLM in our setup (e.g., end of 2023 for Qwen-2.5), where public solutions could not have influenced training. This is the case for competitions like “playground-series-s4e4” (ended in May 2024) or “playgroundseries-s5e5” (ended in June 2025) on which Agent K outperformed more than 88% of the participants. This suggests that its performance is not merely attributable to regurgitating pre-existing solutions.

###### iii) Competition difficulty heterogeneity. Kaggle competitions vary widely in difficulty.

Leaderboard-based metrics such as percentile rank or medal count are inherently influencedbytheskilllevelandsizeoftheparticipantpool. Forexample,achievingatop5%rank in a high-profile competition that attracts many grandmasters is significantly harder than achieving the same rank in a less popular challenge. Furthermore, not all competitions offermedals,andthesetendtoattractdifferentlevelsofparticipantcommitment. Toquantify

difficulty, we compute an Elo-based competition level: through all medal-awarding competitions in Kaggle’s history, we calculate the Elo-MMR ratings of all Kaggle users and we consider for each competition the average Elo of the participants who earned at least a Bronze medal. This provides a relative measure of how competitive a given competition was. In Figure 12 in the Appendix, we analyse Agent K’s medal distribution as a function of the competition level, enabling a fairer interpretation of its achievements across diverse competition types.

###### iv) Absence of recent featured Kaggle competitions. While Agent K demonstrates

strong performance on traditional Kaggle competitions, the platform’s scope has recently expandedtoincludemorediverseandunconventionalchallenges. Theserangefrommathematical puzzle solving 1 and agent design competitions 2 to ambitious tasks aimed at advancing Artificial General Intelligence (AGI), such as the ARC-AGI challenge 3.

Enabling Agent K to effectively participate in such cutting-edge competitions remains a significant challenge. It would require substantial enhancements to both the scaffolding framework and the integration of advanced reasoning tools, domain-specific knowledge, and potentially multi-agent collaboration mechanisms. Nonetheless, the milestones achieved by Agent K on more standard data science and machine learning tasks represent a major breakthrough. They demonstrate the power of embedding human-like learning paradigmsintoautonomousagenticsystems,pavingthewayforfutureexpansionintoeven more complex problem domains.

## References

- [1] Rishi Bommasani et al. “On the Opportunities and Risks of Foundation Models”. In: ArXiv abs/2108.07258 (2021).
- [2] Tom B. Brown et al. “Language Models are Few-Shot Learners”. In: ArXiv abs/2005.14165 (2020).
- [3] Gerald F. Burch et al. “A Meta-Analysis of the Relationship Between Experiential Learning and Learning Outcomes”. In: Decision Sciences Journal of Innovative Education (2019).
- [4] Jun Shern Chan et al. “MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering”. In: The Thirteenth International Conference on Learning Representations. 2025.
- [5] FilipposChristianosetal.“Pangu-Agent:AFine-TunableGeneralistAgentwithStructured Reasoning”. In: ArXiv abs/2312.14878 (2023).
- [6] AlexanderCowen-Riversetal.“HEBO:PushingTheLimitsofSample-EfficientHyperparameter Optimisation”. In: Journal of Artificial Intelligence Research 74 (July 2022).
- [7] Aram Ebtekar and Paul Liu. “An elo-like system for massive multiplayer competitions”. In: arXiv preprint arXiv:2101.00400 (2021).

- 1https://www.kaggle.com/competitions/santa-2024
- 2https://www.kaggle.com/competitions/konwinski-prize
- 3https://www.kaggle.com/competitions/arc-prize-2025

- [8] Katharina Eggensperger et al. “HPOBench: A Collection of Reproducible MultiFidelity Benchmark Problems for HPO”. In: ArXiv abs/2109.06716 (2021).
- [9] Nick Erickson et al. “TabArena: A Living Benchmark for Machine Learning on Tabular Data”. In: arXiv preprint arXiv:2506.16791 (2025).
- [10] Stephen M. Fleming and Raymond J. Dolan. “The neural basis of metacognitive ability”. In: Philosophical Transactions of the Royal Society B: Biological Sciences 367

(2012), pp. 1338–1349.

- [11] Yujing Hu et al. “Learning to Utilize Shaping Rewards: A New Approach of Reward Shaping”.In:AdvancesinNeuralInformationProcessingSystems.Ed.byH.Larochelle et al. Vol. 33. Curran Associates, Inc., 2020, pp. 15931–15941.
- [12] Zhengyao Jiang et al. AIDE: the Machine Learning CodeGen Agent. https://github. com/WecoAI/aideml. Accessed: 2024-08-29. 2024.
- [13] Kaggle. Kaggle: Your Machine Learning and Data Science Community. https://www. kaggle.com. Accessed: 2025-07-12.
- [14] Vikki Knott, Anita S Mak, and James T. Neill. “Teaching intercultural competencies in introductory psychology via application of the Excellence in Cultural Experiential Learning and Leadership model”. In: Australian Journal of Psychology 65 (2013), pp. 46–53.
- [15] David A. Kolb. “Experiential Learning: Experience as the Source of Learning and Development”. In: 1983.
- [16] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. “Deep learning”. In: nature 521.7553 (2015), pp. 436–444.
- [17] Eric G. Meyer et al. “Experiential Learning Cycles as an Effective Means for Teaching Psychiatric Clinical Skills via Repeated Simulation in the Psychiatry Clerkship”. In: Academic Psychiatry 45 (2020), pp. 150–158.
- [18] Volodymyr Mnih et al. “Playing atari with deep reinforcement learning”. In: arXiv preprint arXiv:1312.5602 (2013).
- [19] AllenNewellandHerbertA.Simon.“Computerscienceasempiricalinquiry:symbols and search”. In: Commun. ACM 19 (1976), pp. 113–126.
- [20] Janneke van de Pol, Monique L.L. Volman, and Jos Beishuizen. “UvA-DARE (Digital Academic Repository) Scaffolding in teacher-student interaction: a decade of research”. In: 2010.
- [21] Michael Raschick, Donald E. Maypole, and Priscilla A. Day. “Improving Field Education Through Kolb Learning Theory”. In: Journal of Social Work Education 34 (1998), pp. 31–42.
- [22] Julian Schrittwieser et al. “Mastering Atari, Go, chess and shogi by planning with a learned model”. In: Nature 588 (2019), pp. 604–609.
- [23] Noah Shinn et al. “Reflexion: language agents with verbal reinforcement learning”. In: Neural Information Processing Systems. 2023.
- [24] David Silver and Richard S Sutton. “Welcome to the era of experience”. In: Google AI 1 (2025).

- [25] David Silver et al. “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm”. In: ArXiv abs/1712.01815 (2017).
- [26] David Silver et al. “Mastering the game of Go with deep neural networks and tree search”. In: Nature 529 (2016), pp. 484–489.
- [27] David Silver et al. “Mastering the game of go without human knowledge”. In: nature 550.7676 (2017), pp. 354–359.
- [28] Alan M. Turing. “Computing Machinery and Intelligence”. In: Mind LIX (1950), pp. 433–460.
- [29] L.S.Vygotski˘ıandMichaelCole.“Mindinsociety:thedevelopmentofhigherpsychological processes”. In: 1978.
- [30] GuanzhiWangetal.“Voyager:AnOpen-EndedEmbodiedAgentwithLargeLanguage Models”. In: Trans. Mach. Learn. Res. 2024 (2023).
- [31] Jason Wei et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”. In: Advances in Neural Information Processing Systems. Ed. by S. Koyejo et al. Vol. 35. Curran Associates, Inc., 2022, pp. 24824–24837.
- [32] Lucas Willems, Salem Lahlou, and Yoshua Bengio. Mastering Rate based Curriculum Learning. 2020. arXiv: 2008.06456 [cs.LG].
- [33] DavidJ.Wood,JérômeSeymourBruner,andGailP.Ross.“Theroleoftutoringinproblem solving.” In: Journal of child psychology and psychiatry, and allied disciplines 17 2 (1976), pp. 89–100.
- [34] Shunyu Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models”. In: International Conference on Learning Representations (ICLR). 2023.
- [35] SarahYardley,PimW.Teunissen,andTimDornan.“Experientiallearning:Transforming theory into practice”. In: Medical Teacher 34 (2012), pp. 161–164.

## Methods

#### Stage-Wise Scaffolded Learning with Feedback Control

We define scaffolded learning as an experiential interaction process in which an agent progresses through a sequence of staged environments: T = {T1,T2,...,Tn}, where each task Ti is designed to train a specific capability, and tasks reflect increasing complexity or dependency. This structure is inspired by Vygotsky’s theory of the ZPD, where learners develop most effectively when placed in tasks slightly beyond their current ability, provided the environment offers necessary support. In our context, for example, LLM-based agents may struggle to generate correct, multi-step code solutions from scratch in open-ended settings, but can succeed when the task is scaffolded into modular stages with explicit validation. These stages are not hand-coded solutions, but reflect natural decompositions that a practitioner might follow, such as parsing data formats, defining metrics, or preparing submissions, each aligned with interpretable feedback and domain structure.

Each environment Ti is paired with a feedback function F(i)(Ti,zt) → Ft(i) which evaluatestheagent’sactionzt = E(Σ′t),returningafeedbacksignalFt(i),thatdetermineswhether

theagenthasprogressedandcompletedthestage. TheformofFt(i) canvaryacrossthescaffold (e.g., passing a unit test or matching output structure), as we detail in the next section.

Transitions between stages are gated by competency: the agent may only progress to Ti+1 onceFt(i) indicatesthattheagenthassuccessfullydemonstratedthenecessarycompetency for the task at hand. This gating mechanism ensures that scaffolded learning is not purely sequential but also adaptive, contingent on the agent’s behaviour.

At each scaffolded stage, the agent receives a task description (typically in natural language), access to relevant tools (such as a Python interpreter), and feedback from the environment after attempting a solution. This feedback is then used to update the agent’s internal state via the update function, enabling it to revise its strategy.

#### Implementing Agent K: From Scaffold to Autonomy

Agent K is implemented as a modular agentic system that instantiates our experiential learningframeworkwithinastagedenvironmentdesignedtoreflectthestructureandcomplexity of real-world data science tasks. This section details how the scaffolded environment supports structured capability acquisition, how intrinsic and extrinsic functions are operationalised through LLMs and code execution, and how the agent transitions to autonomous open-ended solution generation.

###### Scaffolded Learning Environment

This section details the design and implementation of the workspace and solutiongeneration scaffolds that Agent K navigates.

Workspace Scaffold: Agent K’s workspace scaffold is designed to support structured explorationofhowtoorganise, interpret, andpreparerawtaskenvironmentsfordownstream learning. Given only a competition URL, the agent autonomously builds an interpretable workspace by progressing through modular stages, each requiring reasoning, code generation, and validation via structured feedback.

ThescaffoldbeginsbydownloadingcompetitiondatausingtheKaggleAPIandscraping theassociatedwebpageforkeyelements: taskdescriptions,evaluationmetrics,submission formats, and data specifications. These raw texts are summarised into focused prompts that compress relevant information and remove distractors (e.g., emojis, formatting artefacts). Examples of these prompts are provided in Appendix H.1.

Theagentthendetectstheinputandtargetmodalities(e.g., tabular, image, text)andbegins constructing the workspace through a series of code generation tasks. It creates mapping files that split the data by modality (e.g., input and output maps in Figure 7), as well as transformation scripts that convert targets into model-consumable formats and back. For instance, it may generate Python code to convert textual class labels into one-hot vectors and a corresponding inverse function to decode predictions into submission-ready labels.

Each scaffolded stage is paired with a task-agnostic unit test that verifies the correctness of the output, checking file existence, column structure, path validity, and basic data integrity. Forinstance, atestforanimage inputmapmightcheckthatthefilecontainsan“id” column, references to valid image paths, and has no empty or duplicated fields. These tests are not written or modified for individual tasks; they are general-purpose and apply across

all competitions, ensuring consistency without manual intervention (see Section B.1.3 in the Appendix for examples). If a test fails, the agent revises its code and retries the stage. Meta-unit tests validate consistency across stages, for example, ensuring that all generated maps can be jointly loaded into a DataLoader object for model training.

These scaffolded stages are defined by the environment, but the agent’s behaviour within them – its code, strategies, and retries – is not hand-coded. Instead, it dynamically constructs its solutions through reflection, planning, and feedback-driven code generation. The process instantiates early experiential learning: the agent experiments, receives structured feedback, and adapts.

Solution Scaffold: After completing the workspace scaffold, Agent K enters the solution scaffold, where it autonomously constructs and refines predictive models using feedback from public leaderboard scores. The agent’s behaviour in this phase is conditioned on the modalities it identified. For tabular tasks, it leverages AutoML tools; for image, text, or multimodal tasks, it generates deep learning models using PyTorch. All model training, evaluation, and submission routines are, of course, automated and implemented via LLMs.

In tabular competitions, Agent K solves the task by invoking an AutoML tool. In our implementation, this AutoML tool is RAMP [7], an in-house AutoML library, and writes the necessary components, such as the metric to use or the name of the target, to interface with it. However, the system is not restricted to RAMP; the agent can be extended to use any compatible AutoML system, provided it can reason about the required interfaces and generate appropriate invocation code. We also introduce the following novel tools:

LLM-Based Tool for Automated Feature Engineering: Feature engineering is critical in enhancingmachinelearningperformance,particularlyintabulardataproblems,byrevealing informative patterns beyond the raw features. However, manual feature engineering is time-consuming, domain-specific, and challenging to scale. To address this, we developed an automated feature engineering tool within the environment, enabling Agent K to dynamically propose and implement feature transformations. The tool leverages an LLM conditioned on the problem description, feature distribution statistics, and small random data samples. Given these inputs, the LLM generates Python code to create new features, which Agent K applies to augment the dataset.

Automated Blending Tools: Model blending is a widely used ensembling technique in machine learning competitions, improving performance by aggregating the complementary strengths of diverse models [10, 12, 11]. To leverage this, we developed a dedicated blending tool within the environment. After training multiple models, Agent K can select a subset and invoke the tool to construct a final submission based on a weighted combination of their outputs. To select models for blending, we leverage an LLM that reviews each candidate’s architecture, loss functions, and validation performance. Based on its training over large code and language corpora, the LLM proposes a combination of subsets of models. Their predictions, such as logits in classification tasks, are aggregated and fed to a small multi-layer perceptron (MLP) trained to produce final outputs, yielding a new submission written to the workspace.

meta-unit test failure

Unit Test Unit Test Unit Test

Unit Test Unit Test

Formatted Data Organised Workspace

Fail Pass Fail Pass Fail Pass Fail Pass

Fail Pass

meta-unit test passes

### Workspace Scaffold

###### input map

output meta-unit test submission metric func map

transforms

Models

Tools

Compute

Choice

[Figure 14]

... ...

... ...

[Figure 15]

[Figure 16]

... ...

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Create

[Figure 21]

[Figure 22]

Solution Scaffold

feature engineering

model training & hyperparam. opt.

multimodal ML models

adding new submission to workspace

Blend

[Figure 23]

blend old submission to create new

blended submission to workspace

[Figure 24]

Submit

###### submission to Kaggle and getting public leaderboard scores

- Figure 7. Two-stage scaffolded learning environment in Agent K. In the Workspace Scaffold (top), the agent learns to build a structured workspace, mapping inputs and outputs, transforming data, and defining evaluation metrics, by passing a sequence of gated unit tests. At each stage, it must autonomously generate executable code that satisfies structuralandfunctionalconstraintsprovidedbytheenvironment. IntheSolutionScaffold(bottom), the agent explores multiple solution strategies, including those that may access established tools for model training, feature engineering, and blending. These tools act as a learned foundation, akin to guidance from a teacher, allowing the agent to focus on reasoning. Feedbackfromvalidationscoresdrivesiterativeimprovement,supportingexperiential learning through reflection, abstraction, and adaptation.

Class Balancing and Target Scaling Tools: Addressing class imbalance and target scaling is critical for building effective classification and regression pipelines. Agent K can adjust for imbalanced class distributions in classification tasks by dynamically designing resampling strategies that rebalance the training set based on observed label frequencies and evaluation metrics. In regression tasks, it normalises target distributions before training to improve convergence and accuracy, and reverses the transformations at inference time. These strategies are implemented through LLM-generated code and iteratively refined based on prior solutions and task-specific feedback.

###### Experiential Learning in Agent K

ToimplementexperientiallearninginAgentK,weusedifferentpromptstrategiestoenable reflection, abstraction, andintegrationofenvironmentfeedback, duringthescaffoldphase and during the open-ended generation phase.

During Scaffolded Learning The implementation of intrinsic functions relies on the LLM-prompting, where the prompts integrate elements from the agent’s internal state Σt that contain characteristics of the competition to solve (e.g. the original task and data description from Kaggle) and elements obtained during previous steps (e.g. previously generated codes, summaries, or queried table views). The details of the prompts for the intrinsic thinking and external action generation, as well as the internal state update rule, depend on each scaffold stage. We provide in Table 1 details on the various prompting schemes that make Agent K progress through the different stages, and we report the exact prompts in Appendix H.1.

Stage Intrinsic Functions I Extrinsic Functions E Environment Feedback F Internal State Update U Competition understanding Summarise Output summaries Keep the summaries Add Summary

Modality identification (Tab., Image, Text)

Think & Summarise JSON with modalities Store modalities Add modality Create Input / Target maps Plan Code Run Unit Test CSV map files

Select Metric Plan Code Run Unit Test Python code Create submission formats Plan Code Run Unit Test Python code

Feature Engineering Plan Code Run Unit Test Python code Create Embedders Plan Code Run Unit Test Python code Class Imbalance Identify & Plan Code Run Unit Test Python code Create Target Transform Identify & Plan Code Run Unit Test Python code

Run Unit Test & Trigger training

Add validation scores Create Solution summary Summarise Output the summary Keep summary Summary

Create Model Head Plan Code

Ensemble Think Select solutions to ensemble Run Blending Add validation scores Error Code Think Summarise the error Keep summary Add error summary

Table1. OverviewofIntrinsicandExtrinsicFunctions, EnvironmentRole,andInternalState updates. Thebottomrowscorrespondtothesolutiongenerationscaffold,whichappliesfor all types of competitions. The group of seven rows below correspond to the solution design scaffold, which applies to CV, NLP, and Multimodal problems, as for tabular-only competitions, the agent uses RAMP to create the solutions. The “Error Code” stage corresponds to the retrial of a stage requiring a code output and that failed during a previous attempt. For thesestages, weindicateinthecolumn“InternalStateUpdate”whattheupdateiswhenthe unit test passes, as when an error happens, the code and error logs are added to the internal state.

We illustrate in Figure 8 a typical loop of experiential learning loop when Agent K tries to generate a code to pass some setup stage Tt. In the example, we assume that the agent already generated some code at the previous steps, which led to some error, whose traces were stored inthe internal state. We prompt the agentto analysethe encounterederror and to design a new plan that serves to generate an external action consisting of a piece of code which should solve the current stage. The code is parsed and executed within a unit test in the environment, and the console output is stored to be appended to the internal state, which acts as the update function.

DuringOpen-EndedSolutionGeneration Intheopen-endedsolutiongenerationphase, the agent generates a diverse set of solution nodes, sequentially building a tree of submissions. The experiential learning loop is depicted in Figure 9, showing the content of the

It1(Σt) Analyse

It2(It1(Σt)) Plan

Prompt (e.g. Error Analysis)

Prompt (e.g. New Planning)

You are a Kaggle GM... Task info: 〈task_summary〉, 〈data_descr〉, 〈data_view〉,...

Σt Internal State

You are a Kaggle GM... Task info: 〈task_summary〉, 〈data_descr〉, 〈data_view〉,...

This was your code: ⟨codek⟩ Explain how to fix: ⟨error_tracek⟩ Also consider previous guidance: {⟨error_analysisi⟩}ki=−11

〈task_summary〉, 〈data_descr〉, 〈data_view〉, ...,

This was your code: ⟨codek⟩ and you got ⟨error_tracek⟩ Design a plan taking into account: {⟨error_analysisi⟩}ki=1

⟨plan1⟩, ⟨code1⟩, ⟨error_trace1⟩, ⟨error_analysis1⟩ ..., ⟨plank⟩, ⟨codek⟩, ⟨error_tracek⟩

LLM Answer

LLM Answer

“Given error message and ... ⟨error_analysisk⟩ ”

“To solve this, I should... ⟨plank+1⟩ ”

It1(Σt) = Σt ∪ {⟨error_analysisk⟩}

Σ′t = It1(Σt) ∪ {⟨plank+1⟩}

Experiential Learning Loop

Et(Σ′

###### t)

t,Ft)

Ut(Σ′

External action

Update the state

###### Prompt (e.g. Implement)

Ft(Tt,zt) Get env. feedback

Update (e.g. Store Elements)

You are a Kaggle GM... Task info: 〈task_summary〉, 〈data_descr〉, 〈data_view〉,...

Σ′t ∪ {⟨codek+1⟩,⟨error_tracek+1⟩}

Env. (e.g. Run zt in a Terminal)

This was your code: ⟨codek⟩ and you got ⟨error_tracek⟩ Taking into account {⟨error_analysisi⟩}ki=1, implement a solution following ⟨plank+1⟩...

user@machine:~$ python script.py Traceback (most recent call last): [...] ValueError: invalid literal for int() with base 10: ’NaN’

Σt+1 := Ut(Σ′t,Ft)

Ft = {⟨codek+1⟩,⟨error_tracek+1⟩}

LLM Answer

```python ⟨codek+1⟩ ```

zt := Et(Σ′t) = ⟨codek+1⟩

- Figure 8. Experiential Learning in Agent K Scaffold - Error Handling. Given internal state Σt containingcompetitionsummarisedinformationandpastimplementationattemptsfor a given stage Tt along with error logs, the experiential learning loop for error handling starts with (It1) the generation of analysis of the error trace obtained at the previous step. It is followed by another intrinsic step (It2) to generate a new plan to pass this stage, taking into account the past error analyses. The external action function (Et) prompt the Agent’s LLM to generate a piece of code that follows the plan stored in the provisory internal state Σ′. The code is then tested in the environment, and the console output is added to the internal state to enable further debugging (if there is again an error). If the unit test passes, the environment switch to the next pipeline stage and feedback new stage-based information to the agent.

internal state, the intrinsic and extrinsic functions, as well as the environment feedback and state update.

Internal Agent State Σt: The internal state consists of the task and data descriptions, the available device, the remaining runtime, and the list of the previously generated openended solutions along with their respective outputs. Moreover, Σt contains the specialized CoT obtained during the scaffolded phase to guide the initial node generation to enable abstraction and guide the exploration of the solution space.

###### Solution generation through reflection and abstraction: From internal state Σt, a new

Σt

It

&

Et

zt

 

 

〈task_summary〉,〈data_descr〉,〈compute_resources〉,〈remaining_time〉,..., {⟨scaffold_coti⟩}ki=1 ,〈submit_pool〉





###### Solution Generation Prompt Draft New Node Improve Best Node Debug Node Shared Instructions

You are a Kaggle grandmaster attending a competition... Given a task you need to create a plan and implement a solution in Python...

- - 〈task_summary〉, 〈data_descr〉, ...
- - 〈compute_resources〉& 〈remaining_time〉
- - ...

Reflection & Abstraction elements

Scaffold CoTs

- - Tried 〈solution A〉, got validationscore〈val_A 〉
- - Tried 〈solution B〉, got valid. score 〈val_B 〉
- - ...

###### Prev. Error

###### Prev. Best Node

Here is the code of the best attempt so far: 〈code〉

Previous (buggy) implementation: 〈code〉 Execution output: 〈error log〉

###### LLM Response

###### Debug Node

###### Improve Node

###### Draft Node

A first attempt to tackle this would be ... ⟨plan⟩

To further improve the model, I will ... ⟨plan⟩

To fix the issue, we need to ... ⟨plan⟩

```python ⟨code⟩ ```

```python ⟨code⟩ ```

```python ⟨code⟩ ```

|Environment (Run ⟨code⟩ and collect ⟨run_log⟩)|
|---|
|user@agk:~$ python script.py Training model...<br><br>. Validation MSE: 0.324|

Ft

###### Internal State Update

###### Prompt (Logs entity recognition) From ⟨run_log⟩ identify:

LLM Answer <log_elements>:

Ut

{

{

"is_successful": "?", "score": "?", ...

"is_successful": "true", "score": "0.324" ,...

}

}

Σt+1[〈submit_pool〉] ← {⟨code⟩,<log_elements>}

###### Figure 9. Experiential Learning in Open-Ended Generation of Agent K.

solution is obtained by prompting the LLM with the relevant elements of the state. The prompting scheme defining the intrinsic and extrinsic functions I&E depends on whether the agent should i) generate an initial solution draft, ii) try to improve the current best solution stored in Σt, or iii) debug a previously deficient code. The three cases are illustrated in Figure 9, where i) in the first column, the scaffold CoTs are given in the prompt to guide the first solution generations, ii) in the middle column, the code of the best solution so far

is provided, and iii) in the right-most column, the buggy implementation along with the error message are added to the prompt.

Environment feedback Ft: the code generated by the agent is executed in the environment, which produces logs that are recorded to provide a feedback Ft to the agent.

State update Σt+1 = Ut(Σ′t,Ft): to update its internal state, the update function involves a prompting mechanism to let the Agent analyse whether the feedback corresponds to a successful run or not, identify the validation metric, and whether this metric should be maximised or minimised, in order to be able to identify the best solution.

This experiential learning loop is repeated until we reach a specified total runtime or when a maximum number of solutions is produced.

#### Experimental Setup, Baselines, Resources

We consider two families of baselines: 1) ReAct-based agents [5], and 2) foundational tabular prediction models [4]. Before detailing these two categories, we first provide insights into the composition of the Kaggle competitions we used as an evaluation set and on how we compute the performance quantiles.

###### Kaggle Competitions Set

All competitions included in our benchmark are listed on kaggle.com and are accessible through their API. Table 7 in the Appendix lists those competitions with their respective URL and Figure 10 shows different statistics of the selected tasks. Figure 10a shows the varied sets of metrics we considered, where some are standard, such as RMSE, others are less common, e.g., median absolute errors, or quadratic weighted kappas. Metrics relate to the nature of the competitions, whose distribution is presented in Figure 10d. Most competitions are regressions or classifications, with a few being more complex multi-target tasks. Figure 10c presents the starting year of the competitions included in our benchmark spanning 2011 to 2025. Finally, Figure 10b illustrates the scale of our competitions by showing the distribution of available labelled inputs. The dataset sizes range from a few hundred samples for competitions designed to emphasize overfitting risks, to several million examples, demonstrating that our benchmark aims to address data science challenges at realworld scale.

###### Performance Quantile Computation

Given a competition C, let kC be the number of submissions that any participant can decide to select as their final submissions, and assume that this competition uses a metric that should be minimised. If a method A generated n distinct valid submissions with

public scores ppub1 ≤ ··· ≤ ppubn (potentially across different attempts), we need to assess its final performance based on at most kC of them. To do so, we use a greedy selection process, i.e., we select the top-min(kC,n) submissions and observe their private scores

ppriv1 ,...,pprivmin(k

C,n) – which are not necessarily in increasing order. Finally, we consider pAfinal = maxi∈{1,...,min(kC,n} pi as the final score of A, and compute the associated quantile by determining the fraction of participants who obtained a score better than pAfinal. If the

10M

12 NLP CV TAB

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

NumberofTrainingSamples

10

1M

8

Count

6

100k

| | |
|---|---|
| | |
| | |
| | |

4

10k

2

0

AUROC

ACCURACY

LOGLOSS

RMSE

RMSLE

F1SCORE

ROCAUC

MAE

R²

BalancedAcc.

SMAPE

QWK

MAP@5

MeanFScore

MacroF1Score

AP@K

MedianAbs.Error

Norm.GiniCoef.

1000

100

NLP CV Tab

Competition Metric

(a) Different Metric functions for all input modalities.

###### (b) Training samples per modality.

16 NLP CV TAB

25 NLP CV TAB

14

12

20

10

Count

Count

15

8

6

10

4

5

2

0

0

2015 2020 2025

MC RG BC MTC MTR RG BC

Competition Start Year

Competition Type

(c) Competition Start Dates.

###### (d) Distribution of competition types.

- Figure 10. Overview of competition metrics, start dates, and types across different input modalities. Those are computed computed over the 81 benchmark tasks we consider.

final competition leaderboard contains N entries with scores s1,...,sN:

|{i ∈ 1,...,N | si < pAfinal}| N

qA = 100 − 100 ·

(1)

such that if pAfinal matches or outperforms the best score in this competition, qA = 100 and if it is worse than any score, qA = 0.

The medals we report are based on the Kaggle medals attribution system, which takes into account the final leaderboard quantile achieved and the number of participants. We provide the precise medal attribution rules in Table 2. Note that we apply this system even for competitions that did not award actual Kaggle medals.

- Table 2. Kaggle Medals Attribution. The thresholds follow Kaggle’s guidelines, and the “⋆ (Top 10 + 0.2 %) means that an extra gold medal will be awarded for every 500 additionalteamsinthecompetition. Forexample, acompetitionwith500teamswillawardgold medals to the top 11 teams, and a competition with 5000 teams will award gold medals to the top 20 teams.”

Medal 0-99 Teams 100-249 Teams 250-999 Teams 1000+ Teams Bronze Top 40% Top 40% Top 100 Top 10% Silver Top 20% Top 20% Top 50 Top 5% Gold Top 10% Top 10 Top 10 + 0.2%* Top 10 + 0.2%*

ReAct-based agents

ReAct-based agent corresponds to the second stage of Agent K, initialised without the scaffold chain-of-thought. To ensure a fair comparison with our method, we assigned it a runtime budget equal to the combined budgets of both phases of Agent K, as shown in Table 4. Each version was executed for two attempts, as was Agent K. We take the implementation from [1] and adapt the main hyperparameters to the different runtimes, as summarised in

- Table 3. Hyperparameter Value Role

Nmax 5000 Max number of iterations τnode 3/16 × Total_Runtime Max runtime per node Ndraft 5 Max initial nodes allowed

Max debug depth 3 Max node debug iterations Probability of debug 50 % Choose to debug node

- Table 3. ReAct-based agent hyperparameters. We report τnode as a function of the total runtime, which depends on the nature of the competition, as specified in Table 4. These hyperparameters are also used for the post-scaffold stage of Agent K.

Shallow ReAct-based agents We evaluate ReAct augmented with a tree-exploration strategy as defined in [5]. We run this method with Qwen2.5-72B, the same model used in Agent K experiments. To further assess the merit of our scaffolded approach compared to a stronger reasoning model, we ran ReAct-based agent equipped with Deepseek-R1 using the same hyperparameters.

ReAct (Qwen) + RAG To address the concern that any Data-science-specific chain-ofthought could independently lead to performance as strong as the ones we achieve with Agent K, we evaluated another variant of ReAct-based agent, where the initial solution generation prompts are guided using examples drawn from a Kaggle-based RAG database. We construct this database of cases similar to [3], selecting high-quality discussions and notebooks from past Kaggle competitions. These documents are pre-processed into structured

entries containing notebook summaries and technical discussions , and are indexed for semantic retrieval. The agent is allowed to generate up to Ndraft new solution nodes per problem and is allowed to retrieve the top N most relevant cases from the database based on semantic similarity to the current problem description. Each new node incorporates one of these retrieved examples into the prompt, applied sequentially, allowing the agent to leverage concrete prior cases while formulating a new solution. When the agent revisits or improves existing nodes, it does so without additional retrieval, ensuring that external knowledge is only introduced during the generation of new branches. This protocol ensures a disciplined and competition-based prompting strategy that facilitates the reuse of relevant knowledge while preserving internal consistency in solution development.

###### Tabular Foundation Models

Ouragentwasevaluatedonseveraltabularbenchmarks, withitsperformancecomparedto TabPFN v2 [4] and other variants of this model. TabPFN is a state-of-the-art tabular foundation model trained on a large set of synthetic tasks. It predicts by processing a sequence of labelled examples without requiring additional parameter updates during inference. As this model by itself does not handle the full data science pipeline, we run it from the automated setup conducted by Agent K, to focus the comparison on the predictions generation quality.

Since TabPFN v2 supports a maximum of 10,000 input samples, we used a K-Meansbased sampling strategy to adapt larger datasets to this constraint. For each batch of 10,000 test samples, we applied K-Means clustering on the test set and used the resulting cluster centres to select a representative subset of 10,000 samples from the training data. These selectedtrainingpoints,alongwiththecorrespondingtestbatch,werethenprovidedasinput to TabPFN v2. The model’s predictions were aggregated across all batches to generate the final submission files. While our own K-means strategy proved effective, we also compared Agent K’s to other TabPFN extensions (TabPFN-Ext.) that were released in the literature to address its limitations.

TabICL: TabICL [9] is another foundation model for tabular data, designed to handle datasets with up to 100,000 samples on affordable hardware. Currently, it supports only classification tasks. We included TabICL in our benchmarking experiments to provide an additional point of comparison for Agent K.

TabPFN Fine-Tuning: To further enhance TabPFN’s performance, we also conducted a fullfine-tuningofthevanillaTabPFNV2modelsforbothclassificationandregressiontasks. Each vanilla model was fine-tuned on the training set of the specific competition, and the resulting model was used to make predictions on the test set. Due to the 10,000-sample input constraint, we again used K-Means-based sampling to divide the training set into N subsets, each containing S samples. Even though we used a different sample size S for training due to GPU memory limitations while maintaining model fidelity, we kept 10,000 samples while performing the inference.

Runtime Limit (Tabular)

Runtime Limit (CV/NLP)

Method Model

Agent K (Scaffold & Beyond ZPD) Qwen2.5-72B 1 day & 1 day 2 days & 2 days

ReAct (Qwen) Qwen2.5-72B 2 days 4 days ReAct (Deepseek R1) Deepseek-R1 2 days 4 days ReAct (Qwen) + RAG Qwen2.5-72B 2 days 4 days

TabPFN-v2 Fine-Tuned TabPFN-v2 2 days -

- Table 4. Runtimes per method for Tabular and CV/NLP competitions, grouped by method type. For the Tab-PFN baselines we only report the time limit for TabPFN-v2 Fine-Tuned as for the other versions we let it run without specifying a time limit (in practice they run within at most a few hours).

###### Computational Resources

Agent K was evaluated on modest hardware to emphasise its accessibility and potential for broad democratisation. Each experiment was executed in an isolated container running Ubuntu 22.04.3 LTS. For each job, compute resources were limited to a single NVIDIA V100 GPU(32GBmemory)and9IntelXeonCPUcores. Furthermore,Table4reportstheruntime limits assigned to Agent K and each baseline across different competition types, ensuring that all methods operated under equivalent time constraints for a fair comparison.

###### Statistical Tests

In order to compare the results obtained by different agents, we conducted several statistical tests. We display in Figure 5 the commonly used critical difference (CD) diagrams to visualise the statistical difference of the results from different methods. The CD diagram summarises the result of multiple pairwise comparisons. Its construction starts with a Friedman test which is used to detect if there are statistically significant differences in the performance ranks of the compared methods. If this test fails, it means that there is a possibility that the hypothesis that all methods have the same average performance cannot be rejected. On the other hand, if the test succeeds, we move towards a post-hoc analysis via pairwise comparisons with a Wilcoxon signed-ranked test, which allows us to determine which pairs of methods are significantly different.

Finally, the data used to perform the tests are the percentiles achieved on the competitionsleaderboards(asdetailedinthePerformanceQuantileComputationsection), consideringallthecompetitionswhereeachmethodmadeatleastonevalidsubmission. Theyare transcribed into relative ranks and averaged over each method. The CD diagram therefore shows at the same time the relative rank of each method (x-axis) and the pairwise statistical difference of the results. The latter is denoted by a horizontal bar that denotes that two methods joined by that bar are not statistically different.

#### Discussion on technology timeline

During the scaffold, the LLM generates fusion models using a modality-dependent embedder. One limitation when comparing to human participants is that, depending on the competition date, not all models and pre-trained weights were available at the time. Figure 11 shows that Agent K manages to get medals even when it uses technology only available at the time of the competition. Even when technology is constrained the agent is still capable of achieving 19 medals.

30

29

Gold Silver Bronze

8

25

NumberofMedals

9

20

19

4

15

4

12

12

3

11

10

9

8

5

7

3

6 6

3

5

4 4 4

4 4

3

0

All Tab NLP CV All Tab NLP CV

Agent K Scaffold (Time-Free Tech.)

Agent K Scaffold (Time-Constrained Tech.)

- Figure 11. Medal Distribution by Modality. On the left we show the number of medals achieved on all competitions without time-restricting the models used by Agent K in its solutions. On the right we show the same count when restricting the models and the version of their weights if needed, to be consistent with the competition timeline.

Time-controlled performance on CV and NLP competitions At scaffold phase, Agent K often relies on existing architectures and pre-training weights which were not released at the time of some Kaggle competitions. To assess the impact of using more advanced technology, we identify for each scaffold submission which architectures, as well as the version ofthepre-trainedweightsthatareused,andchecktheirreleasedatestocomparethemwith the competition end date. If we remove all submissions made using a model not available during the competition period, we see that Agent K still secures 7 medals on CV competitions, and 6 medals on NLP competitions, showing that the good performance of our agent is not only attributable to the use of recent technology.

Time-controlled performance on Tabular tasks. On tabular tasks, the agent generates submissionsat scaffoldphase using RAMP tool, which isbasedon optimisingand blending several base models from lightgbm [6], xgboost [2] and catboost [8]. Considering submissions made on competitions posterior to these base models still yields a total of 6 medals.

#### Data Availability

In our experiments, we rely on Kaggle competitions (listed in Table 7 in the Appendix) whose data and leaderboards are publicly available on the dedicated Kaggle page of each competition. Moreover, we make the RAG database available, uploading a zip file in the repository where we host our agent implementation.

#### Code Availability

WeopensourcedourcodeonGitHubunderMITlicense4. Ourrepositoryincludesthecomplete implementation of Agent K (for both scaffold and post-scaffold stages), the scripts to run the ReAct-based agent baselines, and the pre-processing and post-processing code we designed to run the various TabPFN experiments. Additionally, we curated a documentation with the setup instructions for each type of experiments, and also detailing the configurations and hyperparameters we used, in order to ease replicating our results.

#### Author contributions

Jun Wang and Haitham Bou-Ammar conceived the research direction and made key scientific decisions on human-inspired learning theory for agentic systems with applications to data science.

Antoine Grosnit and Alexandre Maraval designed the agent architecture and scaffold mechanisms and implemented its core structure.

Refinath S N, James Doran, Jonas Gonzalez, Abhineet Kumar, and Khyati Khandelwal implemented the different elements involved in the data-science agent, and extensively tested by Zichao Zhao.

Agent K experiments on CV and NLP competitions were handled by Alexandre Maraval, and the ones on Tabular competitions were run by Abhineet Kumar. ReAct-based agent baselines were evaluated by Alexandre Maraval (with Qwen) and Refinath S N (with Deepseek-R1 and with RAG). TabPFN baselines were integrated and run by Refinath S N.

Balázs Kégl, Albert Thomas, Giuseppe Paolo, Abdelhakim Benechehab, Hamza Cherkaoui, and Youssef Attia El-Hili developed the RAMP-based AutoML tool used in the scaffold for tabular tasks.

Jianye Hao, Jun Yao, and Shao Kun provided strategic oversight and high-level guidance throughout the project.

## References

- [1] Jun Shern Chan et al. “MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering”. In: The Thirteenth International Conference on Learning Representations. 2025.
- [2] Tianqi Chen and Carlos Guestrin. “XGBoost: A Scalable Tree Boosting System”. In: Proceedingsofthe22ndACMSIGKDDInternationalConferenceonKnowledgeDiscovery and Data Mining. KDD ’16. San Francisco, California, USA: Association for Computing Machinery, 2016, pp. 785–794. DOI: 10.1145/2939672.2939785.

4https://github.com/huawei-noah/HEBO/tree/dev-agent/Agent_K

- [3] Siyuan Guo et al. “DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning”. In: Proceedings of the 41st International Conference on Machine Learning. Vol. 235. Proceedings of Machine Learning Research. PMLR, 2024, pp. 16813–16848.
- [4] Noah Hollmann et al. “Accurate predictions on small data with a tabular foundation model”. In: Nature 637.8045 (2025), pp. 319–326.
- [5] Zhengyao Jiang et al. AIDE: the Machine Learning CodeGen Agent. https://github. com/WecoAI/aideml. Accessed: 2024-08-29. 2024.
- [6] Guolin Ke et al. “LightGBM: A Highly Efficient Gradient Boosting Decision Tree”. In: Neural Information Processing Systems. 2017.
- [7] Balázs Kégl et al. The RAMP framework: from reproducibility to transparency in the design and optimization of scientific workflows. 2018.
- [8] Liudmila Prokhorenkova et al. “CatBoost: unbiased boosting with categorical features”. In: Proceedings of the 32nd International Conference on Neural Information Processing Systems. NIPS’18. Montréal, Canada: Curran Associates Inc., 2018, pp. 6639–6649.
- [9] Jingang Qu et al. “TabICL: A Tabular Foundation Model for In-Context Learning on Large Data”. In: arXiv preprint arXiv:2502.05564 (2025).
- [10] Zhiqiang Tang et al. “AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models”. In: arXiv preprint arXiv:2404.16233 (2024).
- [11] Andreas Töscher and Michael Jahrer. “The BigChaos Solution to the Netflix Grand Prize”. In: (Jan. 2009).
- [12] David H. Wolpert. “Stacked generalization”. In: Neural Networks 5.2 (1992), pp. 241–

259. DOI: https://doi.org/10.1016/S0893-6080(05)80023-1.

## A Related Work

#### A.1 Data-science benchmarking

Assessing agents’ performance on data science tasks has been approached from various perspectives. While many benchmarks, such as those for coding proficiency [26, 71], focus on general code generation capabilities, they do not necessarily capture the full range of skills required for end-to-end data science. Assessing the performance of agents on datascience tasks has been considered from different scopes.

Some benchmarks are tailored to specific aspects of data science. For example, [32] focuses on code infilling based on data from Stack Overflow and common libraries, targeting localised code generation rather than comprehensive pipeline development.

Unlike traditional AutoML benchmarks [14], which typically provide a standardised input format to facilitate solver design, data science agent benchmarks evaluate broader capabilities. These include understanding the task requirements and adapting solutions accordingly, offering flexibility in problem formulation and solution strategies.

Authors in [24] introduce a benchmark using six Kaggle tasks, where success is defined as achieving a 10% improvement over a baseline provided by starter code. However, this baseline can sometimes represent minimal predictive performance, such as random predictions, rather than a robust comparison against established leaderboards. This benchmark also offers predefined scripts and environments for each task, guiding the problemsolving approach. Other benchmarks, such as [5], emphasise the orchestration of workflows and interactions with professional software tools, while [22] provides a carefully curated set of questions requiring language models to generate code that answers specific dataset-related queries. This differs from end-to-end problem-solving in competitions like Kaggle,whosequestionsareprimarilyfocusedontabulardata. Additionally,[54]centreson understandingmachinelearningrepositoriesratherthanthebroaderdatascienceprocess, which limits its applicability to end-to-end DS agent evaluation. [69] uses Kaggle datasets to derive specific questions about dataset characteristics, such as imbalance, missing data, and correlations. Although this approach provides a focused evaluation, it does not capture the full scope of capabilities required for complete data science workflows, which is a broader objective we aim to achieve.

Closer to our experimental study, [28] curates a list of 74 Kaggle competitions to assess the data modelling capacity of LLM agents. Nevertheless, the authors pick the benchmark competitions based on the simplicity of their raw structures, featuring only a train, test and sample submission file. Moreover, they do not support comparison with the Kaggle leaderboard but split the training set to create a custom test set instead. Likewise, MLE-bench [6] includes 75 machine learning competitions from Kaggle to assess agent performance by comparing it against user submissions. However, because this assessment relies on a custom split of the training data rather than the competition’s actual private test set, it remains uncertain whether an agent’s high ranking in this context would align with results on the original Kaggle leaderboard.

Our experiments aim to truly assess the end-to-end capacity of agents to achieve high performance in data-science tasks autonomously.

#### A.2 Traditional automated data science (Auto-DS)

Achieving performance and automation in data science tasks has been a long-standing challenge. Numerous techniques have been developed to automate the human-intensive steps of the DS pipeline using machine learning. Given that each step in the DS pipeline poses challenges, many approaches have focused on specific aspects of the problem, often resulting in narrow application ranges or use-case limitations. For example, probabilistic inference rules have been used to automate data cleaning [46], while various deep learning methods have been developed to filter, select, or transform features to enhance the accuracyofdownstreammodels[29,30,31]. Inthemodellingdomain,asignificantresearcharea known as neural architecture search (NAS) optimises the structure of neural networks for predictions [36, 62], shifting the manual effort from model design to the design of a model search space. Beyond neural architecture, many hyperparameters can be automatically tuned to achieve better performance using black-box optimisation methods. Techniques likeBayesianoptimisationaddressblack-boxoptimisationinasampleefficientwayandare supported by various libraries [2, 4, 9, 11], which users can utilise by correctly identifying andprovidingthehyperparametersoftheirpipelines. Additionally,meta-learninghasbeen introducedtofurtherimprovedataefficiencybyleveragingpastexperimentstopredictperformance on new tasks based on related tasks solved with similar configurations [21, 39, 41].

Contrastingwiththesetargetedmethods, severallibrarieshavebeendevelopedtocover broader parts of the data science pipeline, such as H2O-AutoML [33], AutoKeras [27], Autosklearn [13], and AutoGluon [55]. These tools aim to automate and optimise by making rule-based decisions based on input data modality and combining a fixed collection of models that can be refined using preset optimiser tools. However, these approaches have limitations, includingrestrictedscope, flexibility, andusabilitybarriers. Theirperformance is limited by the set of predefined models and hyperparameters they support, and they require expert knowledge to properly format the problem [7] (e.g., manual specification of column types for AutoKeras). Tools like H2O-AutoML are also limited to tabular data tasks, requiring data to be structured in specific ways.

#### A.3 Data-science Agents

The emergence of data-science agents powered by large language models and multi-agent systems represents a shift towards more autonomous and adaptable solutions. Unlike traditional Auto-DS approaches, these agents are designed to perform end-to-end tasks by leveraging advanced reasoning and decision-making capabilities.

###### A.3.1 Coding Agents

Recent advancements in LLMs [1, 25, 40, 47], combined with techniques like chain of thought prompting [61, 70] and ReAct [66] have significantly expanded the scope of AI applications beyond natural language processing. These models are now capable of generating code and interacting with APIs [10, 49, 44], laying the groundwork for their integration into multi-agent systems [16, 20, 56, 57, 63] that can execute complex tasks by thinking, planning, calling tools [45], and making decisions [23, 34, 48, 67] across various domains, including robotics [38, 42, 51], medicine [53, 58], and also data science [15, 19, 68]. DS

agents exploit the intrinsic capacities of LLM-Agents. They are augmented with specific tools for writing, executing and debugging code, data visualisation tools [65], or access to standard ML libraries APIs [50], offering a more integrated approach to data science automation.

###### A.3.2 Intelligent DS assistants

Several agents function as intelligent assistants, enhancing human decision-making by providing automated solutions while keeping users in the loop. For instance, CleanAgent [43] handles data cleaning based on user specifications, and LAMBDA [52] generates data analysis reports based on technical queries. MatPlotAgent [65] pairs a coding agent with a visual agent to meet the expert’s needs in terms of data visualisation, while WaitGPT [64] offers a smoother monitoring of the code generation by creating graph representations of the generated code and supporting on the fly manual edition or questions. Although these agents enhance user efficiency, they require substantial user input. They are not designed to autonomously achieve high performance in complex, real-world scenarios, such as winning a Kaggle competition.

#### A.4 End-to-end DS agents

End-to-end data-science agents aim to manage the entire data science pipeline autonomously, from data preprocessing to model evaluation and hyperparameter optimisation, without requiring significant human oversight. While some existing approaches address specific aspects of this workflow, such as CAAFE [18] which allows agents to iteratively generate codes to modify tabular features and reflect based on accuracy feedback, or EVAPORATE-CODE+ [3], which focuses on data mining by generating code to extract structured data from semi-structured documents, other methods have tackled the entire data science pipeline more holistically.

Recent multi-agent systems like SEED [7], AutoM3L [37], Data Interpreter [19], AutoML-Agent [60], HuggingGPT [50], MLCopilot [68], DS-Agent [15], SELA [8], MLZero [12], AutoKaggle[35]andAIRA-dojo[59]achievecapacitiestodealwithdata-sciencebyempowering their system with numerous tools, using retrieval from expert demonstrations, and supporting complex multiple steps reasoning.

These systems employ various strategies for generating automated pipelines for different stages of the data science process such as preprocessing, modelling, and feature selection.

While DS-Agent, MLCopilot , AIDE and AIRA-dojo generate and refine the entire solution script as a single block, SEED performs a skyline-based optimisation to select the modules to apply to solve the task. Alternatively, HuggingGPT delegates the planning to a high-level agent, and Data-Interpreter lets this agent decide on a hierarchical and dynamic plan. Meanwhile, AutoM3L follows a pre-defined modality-specific pipeline to produce final predictions. Our data-science agent integrates elements from these various approaches by imposing a modality-dependent structure during the setup scaffold phase while allowing agents greater flexibility in selecting the optimal sequence of steps to achieve the best possible outcome.

Moreover, several existing systems also incorporate mechanisms for continual learning, enabling them to leverage past experiences to improve performance on new tasks.

For instance, DS-Agent achieves that by adopting a case-based reasoning approach combined with an offline-online setup, retrieving insights during the online phase from a pool of scripts that performed well on offline tasks. Similarly, MLCopilot divides its framework into offline and online phases, using an LLM to infer verifiable machine learning strategies from diverse offline experiments, such as determining valid learning rate ranges for specific models. Beyond building a knowledge database to retrieve from, Data-Interpreter uses prior experiences to create new tools that the agents can later call.

However, these systems theoretically or practically suffer from partial coverage or limited capacities. Indeed, AutoM3L cannot generate solutions involving model ensembling and requires specifically structured input data, while HuggingGPT lacks support for model training and tabular models. Moreover, SEED and DS-Agent do not demonstrate capabilities for handling computer vision tasks, and AIDE, AutoKaggle, and Data Interpreter have onlyshownstrongperformanceonmorestraightforwardtabulardatasetsorKagglecompetitions. AutoML-Agent requires comprehensive task-specific prompts which can be advantageousfor expertscustomising itsperformancebutposesachallengeforless-experienced users. Additionally, while some efforts have been made to utilise past experiments for new task resolution, the strategic selection of task order to enable progressive learning remains an underexplored area. We aim to address these gaps by developing a system capable of handling various input modalities and achieving consistently high performance in realworld data science competitions.

Beyond these practical continual learning strategies, there is a rich literature on online learning and online convex optimization [17] that provides strong theoretical guarantees for regret minimization in sequential decision-making. These methods assume a welldefined and typically convex loss structure, with each step involving an update based on newly observed feedback. While valuable for formal analysis, such frameworks are generally restricted to lower-dimensional, well-specified optimisation problems and do not directly address the high-dimensional, multi-modal, and tool-rich environment faced by end-to-end DS agents in real-world competitions. Our approach instead integrates flexible decision-making across heterogeneous subtasks, where the feedback signal may be sparse, delayed, or only indirectly comparable (e.g., via leaderboard or validation scores).

## B Agent K - Pipeline Details

According to our formulation, the design of the agent framework is broken down into two main parts: the setup pipeline and the solution generation pipeline. The setup pipeline’s goal is to preprocess the raw information and raw data to prepare a workspace and a DataLoader object directly usable by the solution generation pipeline to solve the task. All the steps related to downloading, summarising, organising and cleaning the data are done in the setup pipeline, whereas steps related to feature engineering, model training, hyperparameter optimisation and submission belong to the prediction pipeline. In this section we give more detail about the pipeline steps, their objective and important aspects of how they work.

#### B.1 Setting up Competitions

The goal is to automate the process of setting up a task. The pipeline, presented in Figure 7 depicts the sequence of stages that Agent K tackles in order to setup a new Kaggle competition, given its URL.

###### B.1.1 Expected Result - Usable & Interpretable Workspace

At the end of a successful setup, we expect to have a workspace containing useful files, including raw information and raw data from the competition website, as well as new files created by Agent K, such as metadata, different CSV files as well as runnable code files. Our agent creates additional CSV files only if it deems it necessary from the context of the competition:

- • train_tab_input_map.csv - tabular inputs from the train set
- • train_txt_input_map.csv - text inputs from the train set
- • train_img_input_map.csv - paths to the image inputs from the train set
- • train_tab_target_map.csv - tabular targets from the train set
- • train_txt_target_map.csv - text targets from the train set
- • train_img_target_map.csv - paths to the image targets from the train set
- • test_tab_input_map.csv - tabular inputs from the test set
- • test_txt_input_map.csv - text inputs from the test set
- • test_img_input_map.csv - paths to the image inputs from the test set

When Agent K produces code to generate the CSV files, this code is kept in the workspace, which contributes to the interpretability of the result. Furthermore, our agent also need to implement the useful functions:

- • code_transform_tab_target_train.py - code to transform the targets into a format usable for training models and back to the original format
- • code_metric.py - the metric function implementation
- • code_submission_format.py - code to transform from the predicted targets format to the exact format accepted by the competition (e.g. from a probability of a class to the label of that class)
- • code_submission_format_alt.py - an alternative to the previous function, in case it fails. It can also be used as a backup for the solution generation pipeline when we format the predictions of the model for submission.

###### B.1.2 Setup Pipeline Steps

Fetch & Scrape Raw Data Given the URL of the competition, Agent K can use the Kaggle API to download its associated data. The agent is also given a scraping tool which automatically retrieves information from the competition’s web page, such as the raw description of the competition, its data description, metric description, etc.

Summarise From the scraped raw descriptions, Agent K creates digestible summaries that it can later reuse for LLM prompting. The raw text from the competition pages can contain elements that are irrelevant to the task (e.g. emojis, images, urls, ...) or worse, information that is misleading. Summarising the raw descriptions can deal with those issues and enables the agent to compress the important information from the task into a more useful and shorter text that takes less tokens in the future prompts.

DetectModalities Weconsiderthreemainmodalities: visiondata,language dataandtabular data, which can be purely tabular or contain time-series. The agent categorises all competitions’ inputs and targets with respect to these three modalities, based on the summarised descriptions. For instance, a competition where participants are asked to create an image classifier would have image inputs and tabular targets (classification). In another case, if a competition consists in attributing a toxicity score to movie reviews and includes as input the reviews as well as the categories of the movies, then the input modalities would be a mixture of text and tabular, while the modality of the target would be tabular (regression).

Create Plan & Metadata After the raw data is fetched, the summaries are written and the modalities are detected, Agent K creates its plan, i.e. its workflow. This plan determines the sequence of steps that should be done in order to complete the setup. This plan is also dynamic in the sense that our agent can decide to prioritise stages over others provided it is permitted by the general sequence order of the environment. Agent K can come back to previous stages when executing its plan, and it can attempt some stages multiple times if they haven’t been done properly the first time, as we detail in the paragraph dedicated to unit tests.

Create Maps & Transforms The main task in the setup pipeline is for the agent to create maps and transforms, as outlined in Section B.1.1. The idea is to split all raw training data into input and target maps, as well as to split all raw data (train and test) into different modalities. A task could have multiple types of inputs and targets so we require a map for each of them. These maps are then collated together in a custom DataLoader object that makes it easy to consume by a model in the prediction pipeline. Importantly, the agent is also tasked with creating transform functions in order to treat targets as consumable data for a model. For instance if the targets in the training set are text-encoded labels (in a classificationtask)thetransform functioncouldmapthesetoaone-hotencodingandtheinverse transform function would map them back to the text labels. These functions are crucial to enable the prediction pipeline.

Create Submission Format & Metric Function Finally, Agent K generates code to create the submission format and the metric function. Both are generally described in the raw competition description. The submission format is the exact format expected by the competition to read the predictions. It can sometimes be different from the format of the raw train targets so in addition to the transforms, we also need a function that maps the predicted targets on the test inputs, in the exact format expected.

###### B.1.3 Unit Tests

An important part of the setup pipeline is the set of unit tests used to assert if a stage or a group of stages has been successfully done by the agent. It is worth noting that these unit tests are manually designed and not generated by the agent itself. However, this does not mean that they are tailored to each task. In fact they are designed to be agnostic to the competition to setup and are rather specific to the framework itself. Knowing the properties of the expected result of a particular stage, we can test if the code or the data our agent generate follows an expect format. This way our agent receives insightful feedback through error messages when its code is faulty, which it integrates into its reasoning process to correct itself.

For example, suppose our agent generates a runnable code to create the training input map of the image modality. During the state transition of the environment, the unit test associated to this stage is ran. The unit test checks that a CSV file with a specific name exists and is loads it. It then assesses that the table has a column named “id” and that it has at least one row. It also checks the existence of a column apart from the “id” column and that these other columns contain paths to images with known extensions. Moreover, the unit testthrowsanerrorifoneoftheimagepathslistedinthatmapisinvalid. Finallyitcancheck various trivial things such as non emptiness of columns or absence of duplicated columns. This shows that the unit tests are specific to the setup environment and not specific to the task.

If an error is raised in the unit test, the error is saved in memory and the current stage is attempted again. In this new trial, our agent will be able modify and fix its previous code based on the unit test error message that is accessible in its memory. This is repeated until the unit test passes or a budget limit is reached (in which case the setup would fail there).

Finallywehavemeta-unittestswhichareunitteststhatdependonmultiplestages. They verify that the result of multiple single stages coordinate well together. If a meta-unit test fails, the agent has the opportunity to go back multiple stages and re-attempt the relevant stages. For instance, after all maps and transforms are done for the training data, we test if the DataLoader can be constructed properly and if we can successfully load a batch. This step is crucial as it validates that all the maps generated by the agent up to this point share a common “id” column so the DataLoader can collate them properly. If this test fails, the agent has to go back to the start and figure out which codes to correct given the error raised in the meta-unit test to eventually correct its mistakes.

This paragraph highlights the usefulness of the unit tests, showing that they do not only serve as checks for the agent, but that they also provide useful error messages to fuel effective reflection. The meta-unit tests also provide additional verifications that can only be done once a group of stages have passed their own individual unit tests. However, there is always a limit to what we can test. It is possible that the agent finds a way to pass all unit

- Table 5. Hyperparmeters used for the training of the solutions generated during the scaffold stage.

|Name|Value|Role|
|---|---|---|
|max_epochs max_time batch_size lr optimizer n_trials k_folds tta_rounds blend_after<br><br>|30 10h 32 [1e − 6,1e − 2] {Adam, SGD, AdamW} 20 5 4 3|Max training epochs allowed<br><br>Max training time allowed<br><br>Training & testing batch size<br><br>BO search space for learning rate<br><br>BO search space for optimiser<br><br>Max BO rounds allowed<br><br>Cross-Validation folds<br><br>Number of TTA rounds averaged<br><br>max nb. of solutions before blending|

tests but ends up setting up the competition incorrectly. Because we want our unit tests to remain task-agnostic we cannot extend our coverage indefinitely. In the future we would like to extend this work to also use the agent to generate additional unit tests that would be specific to the current competition being setup. In that way we can extend the coverage at inference-time.

#### B.2 Generating Solutions

After the agent has generated code that constitutes a fully-functioning solution, and while training that solution, we allow for some optimisations and support such as creating additional submission files to give Agent K more choice. These are part of the scaffold and are not directly coded by the agent itself but are there rather as a design choice.

First,wedoBayesianOptimisation(BO)oversomeparameters,i.e. thelearningrateand the optimiser choice. The training is run for a maximum of n_trials trials, each trial being limited to max_epochs epochs or max_time, whichever is attained first.

We then generate Cross-Validated submissions by training the model with the best learning rate and optimiser found during BO, on k_folds folds of the training set. We then average the predictions to create a new submission file. For the competitions involving images, we also add a submission file created via simple Test-Time-Adaptation (TTA). This method augments the data at test time with the same data augmentation pipeline used for the training dataloader. Using the best checkpoint saved from the training, we reproduce this for tta_rounds rounds and average the predictions to create a new submission file.

Finally, blending is used after at least blend_after full working and different solutions have been generated. Values are specified in Table 5.

#### B.3 Pipeline Pseudocodes

We give in Algorithms 1 and 2 the pseudocode for the setup pipeline and the pseudocode for the solution generation pipeline, respectively. Note that the input of the setup pipeline isthecompetitionURLandtheoutputsarethesetupfilesandthedataloaders. Thesolution

- Algorithm 1 Pseudocode for the Setup Pipeline Inputs: URL

D ← FetchAndScrapeData(URL) S ← SummariseDataAndTaskDetails(D) {X,Y} ← DetectModalities(S) P ← CreatePlan({X,Y},S)

▷ Every while loop in the pipeline is subject to a retry limit while meta-unit test fails do while unit test fails do

Mtraininput ← CreateTrainInputMaps(D,S,{X,Y},P) end while while unit test fails do

Mtraintarget ← CreateTrainTargetMaps(D,S,{X,Y},P) end while while unit test fails do

Ttargettrain ← CreateTrainTargetTransforms(D,S,{X,Y},P)

end while end while while meta-unit test fails do while unit test fails do

Mtestinput ← CreateTestInputMaps(D,S,{X,Y},P)

end while end while while unit test fails do

Fsubmission ← CreateSubmissionFormat(D,S,{X,Y},P) end while while unit test fails do

Fmetric ← CreateMetric(D,S,{X,Y},P) end while

Outputs: Train and test dataloaders DLtrain and DLtest

generation pipeline takes as inputs the outputs of the setup pipeline and carries on generating code to create one or more submission files. The output of the solution generation pipeline is a set of solutions along with their performance on the leaderboard.

Retry Budget Note that in the pseudocode, there are steps such as “while ... end while”. Inpractice, wedonotretryeachstepindefinitely, thereisanupperboundonthenumberof retries we allow Agent K to do for each step. As soon as this number is reached we consider the agent to be stuck and we end the episode by raising an error.

- Algorithm 2 Pseudocode for the Solution Generation Pipeline

Inputs: Ssetup ← {D,S,{X,Y},P,Mtraininput,Mtraintarget,Mtestinput,Ttargettrain,Fsubmission,Fmetric}, DLtrain and DLtest, Embedders E ← {}, Heads H ← {}, Solutions Sol ← {}

if tabular-only task then AdoptClassicalMLFramework() Sol ← UseRampTool(Ssetup,DLtrain,DLtest)

###### else

AdoptDNNFramework() while Agent decides to improve performance do

▷ Create Input Encoder(s)

if Ssetup has tabular inputs then Ff.e. ← TabularFeatureEngineering(Ssetup,DLtrain) Etab ← TabularEmbedding(Ssetup,DLtrain) E ← E ∪ {Ff.e.,Etab}

end if if Ssetup has image inputs then

Fimg ← ImageTransforms(Ssetup,DLtrain) Eimg ← ImageEmbedding(Ssetup,DLtrain) E ← E ∪ {Fimg,Eimg}

end if if Ssetup has text inputs then

Ftxt ← TextPreprocessing(Ssetup,DLtrain) Etxt ← TextEmbedding(Ssetup,DLtrain) E ← E ∪ {Ftxt,Etxt}

###### end if

▷ Create Target Head(s)

Htab ← TabularHead(Ssetup) H ← H ∪ {Htab}

▷ Generate submission.csv and get validation score sub,val ← TrainSolutionAndPredict(Ssetup,DLtrain,DLtest,E,H) Sol ← Sol ∪ {sub} if enough solutions created then ▷ Blend existing solutions

sub,val ← BlendSolutionsAndPredict(Ssetup,DLtrain,DLtest,E,H,Sol) Sol ← Sol ∪ {sub}

end if end while

▷ Send submissions and get private score and rank r ← SendSubmissionandGetLeaderboardRank(sub)

###### end if

## C Competition difficulty analysis

As detailed in the “Quantitative Results” section, Agent K’s achievements include several gold medals across different types of competitions — community, playground, research, and featured — however, some of these competitions are less complex than others, and not all competitions award medals according to the Kaggle system. In this section, we provide further insights on competitions’ difficulties by measuring the levels of their participants. To do so, we analyse the Elo scores of all Kaggle participants, going through all the medalawarding competitions on Kaggle, and updating the Elo-MMR of the competitors based on their private ranks (the Elo score of a new participant being initialised at 1,500). Doing so, we can access for each competition (medal-awarding or not), the Elo scores of the participants at the time this competition ended.

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

2100

AverageEloofMedal-LevelParticipants

2000

|Competition type<br><br>Medal-awarding<br><br>Not medal-awarding|
|---|

1900

1800

|Agent K medal-level<br><br>Gold Silver Bronze<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

1700

1600

1500

2012 2014 2016 2018 2020 2022 2024 2026

Competition Deadline

- Figure 12. Competition difficulty analysis. Each marker in the plot represents a Kaggle competition included in our benchmark. The x-axis indicates the competition’s closing date, while the y-axis shows the average Elo score of participants who achieved at least a bronze medal (or a bronze medal-level score for non-medal awarding competitions). For competitions in which Agent K attained a medal-level score, the marker is outlined in the corresponding medal color. Notably, since Agent K achieved 4 gold medals across medalawarding competitions, the figure includes 4 blue circular markers with gold outlines.

We present the results of our analysis on Figure 12 where we display for each competition included in our benchmark, the average Elo scores of the participants that have reachedatleastabronzemedal-levelscore. WeobserveagapbetweentheaverageEloscore on the non-medal-awarding competitions (purple diamond markers) and on the medalawarding ones (blue circle markers), which confirms the idea that stronger data scientists tend to participate more in medal-awarding competitions. Moreover, we highlight the medal-level scores achieved by Agent K by outlining the competition markers in the colors of the medals. For instance, the 4 gold (resp. 4 silver) medals obtained by Agent K on

medal-awarding competitions appear as golden (resp. silver) disks on the figure, with their y-position matching the average Elo of the participants who won at least a bronze medals on these competitions.

Thisanalysispushesustofurtherimproveourperformanceonthecompetitionsinvolving stronger data-scientists.

## D Benchmark details

We provide the list of competitions included in our benchmark in Table 7.

- Table 6. Symbols associated to competition types and competition categories on Kaggle. Competition Type Meaning Kaggle Category Meaning

RG Regression GS Getting Started BC Binary Classification P Playground MC Multiclass Classification F Featured MTR Multitarget Regression C Community MTC Multitarget Classification RS Research

- Table 7. List of the Kaggle competitions included in our benchmark. Each row represents a Kaggle competition, showing its Kaggle ID (first column), its type, its category on Kaggle, the input modalities it involve, the number of submissions that can be retained for the final scoring(#SelectSubs), andthenumberofteamsthatparticipatedinthiscompetition. Note

that the column # Select Subs corresponds to kc in equation (1). See Table 6 for the symbols appearing in the Competition Type and Kaggle Type columns.

Competition Type

Kaggle Category

#Select Subs

#Teams

Competition

Modality

nlp-getting-started BC GS Tab,Txt ≥ 100 856 sentiment-analysis-onmovie-reviews

MC P Txt 2 860

BC P Img 2 1314

dogs-vs-cats-reduxkernels-edition

dog-breed-identification MC P Img 1 1280 state-farm-distracteddriver-detection

MC F Img 2 1438

histopathologic-cancerdetection

BC P Img 2 1149

whereami MC C Img 20 211 5-flowers-imageclassification

MC C Img 20 69

Kaggle Type

#Select Subs

#Teams

Competition Task type

Modality

sign-language-imageclassification

MC C Img 2 41

ml-olympiad-landscapeimage-classification

MC C Img 2 13

noaa-right-whalerecognition

MC RS Img 2 364

humpback-whaleidentification

MC F Img 2 2120

planttraits2024 MTR RS Tab,Img 1 398 invasive-speciesmonitoring

BC P Img 2 511

tabular-playgroundseries-feb-2021

RG P Tab 2 1433

playground-series-s3e11 RG P Tab 2 952

- playground-series-s3e14 RG P Tab 2 1875

- playground-series-s3e15 RG P Tab 2 693 home-data-for-ml-course RG GS Tab 2 106930 bike-sharing-demand RG P Tab 2 3242

tabular-playground-

RG P Tab 2 1753

- series-aug-2021

forest-cover-typeprediction

MC P Tab 2 1692

mercedes-benz-greenermanufacturing

RG F Tab 2 3823

goodreads-booksreviews-290312

MC C Tab,Txt 5 261

sberbank-russianhousing-market

RG F Tab 2 3264

restaurant-revenueprediction

RG F Tab 2 2257

scrabble-player-rating RG P Tab 2 301 nlp1000-ml-challenge BC C Txt 2 16 nlpsci MC C Txt 20 44 sf-crime MC P Tab 2 2331 otto-group-productclassification-challenge

MC F Tab 2 3507

nitro-languageprocessing-3

BC C Txt 3 45

sbu-ai-finalproject MC C Txt 5 37

Kaggle Type

#Select Subs

#Teams

Competition Task type

Modality

avito-prohibited-content BC F Tab,Txt 2 284 playground-series-s3e1 RG P Tab 2 689

- playground-series-s3e9 RG P Tab 2 765

playground-series-s3e16 RG P Tab 2 1429

- playground-series-s3e25 RG P Tab 2 1632 tabular-playground-

- series-jan-2021

RG P Tab 2 1728

tabular-playgroundseries-aug-2022

BC P Tab 2 1888

tabular-playgroundseries-apr-2021

BC P Tab 2 1244

- playground-series-s4e4 RG P Tab 2 2606 ventilator-pressureprediction

RG RS Tab 2 2605

- playground-series-s4e5 RG P Tab 2 2788 porto-seguro-safe-driverprediction

BC F Tab 2 5156

playground-series-s3e3 BC P Tab 2 665 santander-customersatisfaction

BC F Tab 2 5115

- playground-series-s3e5 RG P Tab 2 901 cat-in-the-dat BC P Tab 2 1338

- playground-series-s4e1 BC P Tab 2 3632 tabular-playgroundseries-sep-2022

RG P Tab 2 1381

cat-in-the-dat-ii BC P Tab 2 1161 playground-series-s3e22 MC P Tab 2 1541 playground-series-s3e26 MC P Tab 2 1661 playground-series-s3e7 BC P Tab 2 678

MTC F Txt 2 4539

jigsaw-toxic-commentclassification-challenge

kaggle-pog-series-s01e03 MC C Tab,Img 2 91 vehicle MC C Img 2 65 plant-seedlingsclassification

MC P Img 1 833

sp-society-cameramodel-identification

MC F Img 2 580

Kaggle Type

#Select Subs

#Teams

Competition Task type

Modality

plant-pathology-2020fgvc7

MC RS Img 2 1317

world-championship-

BC C Img 2 111

- 2023-embryoclassification

leaf-classification MC P Tab,Img 2 1595 nitro-languageprocessing-2

MC C Txt 3 46

- playground-series-s3e24 BC P Tab 2 1908

- playground-series-s4e2 MC P Tab 2 3587 tmdb-box-officeprediction

RG P Tab,Txt 2 1395

playground-series-s3e19 RG P Tab 2 1172 tabular-playgroundseries-may-2022

BC P Tab 2 1151

tabular-playgroundseries-jan-2022

RG P Tab 2 1591

dont-overfit-ii BC P Tab 2 2315 statoil-iceberg-classifierchallenge

BC F Tab,Img 2 3330

diabetic-retinopathydetection

MC F Img 2 660

stumbleupon BC F Tab,Txt 2 624 h2oai-predict-the-llm MC C Txt 2 92 human-protein-atlasimage-classification

MTC F Img 2 2160

predict-who-is-moreinfluential-in-a-socialnetwork

RG BC F Tab 5 132

bioresponse RG F Tab 5 698 GiveMeSomeCredit RG F Tab 5 924 galaxy-zoo-the-galaxychallenge

MTC RS Img 5 326

- playground-series-s5e5 RG P Tab 2 4316

## E Additional results

We show in Figure 13 the success rate of each stage of the setup pipeline for the different types of tasks. For each stage, we track if it was a success (i.e. associated unit test passed) or a failure. We also track if that stage was not reached, caused by the failure of a preceding stage. Finally we also track if that stage is skipped as not all stages are required for each competition. As described in Section B.1, the stages to complete depends on the nature of the task and notably of the modalities of the input data. Figure 13 shows the reason why a stage is skipped on the right-hand side of each bar. Finally, the success rate (percentage in the green portion of the bar) shown for each stage is reflective of the number of times this particular stage is successfully done by the agent, if this stage is reached, averaged over all competitions. This explains why, for example, there is a 77.8% success rate for the stage "Create Map for Inputs Images Train" with a bar that is larger than the 87.2% of the stage just below "Create Map for Inputs Text Train".

The takeaway from Figure 13 is that most stages have a high success rate. There is a weaker point for creating the training image input map. This can be explained by the often complicated of files structures of Kaggle competitions requiring image inputs. The relatively smaller percentages for the stages such as "Create Map for Inputs Text Train" or "Unit Test for the Text train maps" can be explained by the fact that there are less competitions in our benchmark that require text inputs. Nevertheless, across input modalities the overall stage success rate is high for Agent K. In particular the overall setup success rate is 92.5%.

## F Details on React-Agent baselines

This sectionoutlinesthe modificationsmade to theAIDEversiondesignedforMLE-Bench, toalignitwiththespecificrequirementsandconditionsofourexperiments. Thesechanges were necessary to ensure that the ReAct Agent from AIDE could effectively operate within the experimental setup, particularly in relation to open-source models and task-specific execution constraints.

Open-Source Models and Structured Output Generation A key modification involved adapting AIDE to support open-source models, such as Qwen2.5, that do not natively generate structured output. Originally, AIDE was designed to work with models that inherently produced structured results, simplifying downstream processing. However, open-source models like Qwen2.5 often produce unstructured or loosely structured text. In response to this, the input prompts given to the models were explicitly updated to request the generation of structured outputs, such as JSON format. Additionally, a post-processing step was introduced to parse and transform the raw LLM output into the required structured format. When issues arose, such as incomplete or misstructured output (e.g., missing expected keys), we sent a new request to the LLM, adding the formatting error to the prompt, in order to get a new valid output. If after five retrials the output is still not in the expected format, we stop the runs and restart from scratch, which happened only a few times.

Prompt Update The prompt to the LLM was updated to include detailed information about the Kaggle competition, such as an overview of the competition and a description

Passed Failed Not Reached Skipped

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|77.8 %<br><br>87.2 %| | | | | | | | | |No| | | |Image input<br><br>No Text input| | |
|98.4 %| | | | | | | | | | | | | |No Tab. input| | |
| | | | | | | | | | | | | | | | | |
|92.9 %| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
|96.7 %| | | | | | | | | | | | | | | | |
|100 %| | | | | | | | | |Not| | | |Bin. Classif.| | |
|98.7 %| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
|100 %<br><br>100 %| | | | | | | | | |No| | | |Image input<br><br>No Text input| | |
|97.7 %<br><br>96.9 %| | | | | | | | | | | | | | | | |
|98.5 %| | | | | | | | | | | | | |No Tab. input| | |
| | | | | | | | | | | | | | | | | |
|99.5 %| | | | | | | | | |No| | | |Image input| | |
|94.8 %| | | | | | | | | | | | | |No Text input| | |
|99.2 %| | | | | | | | | | | | | | | | |
|98.8 %| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
|96.7 %| | | | | | | | | | | | | | | | |
|98.9 %| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
|91.2 %| | | | | | | | | | | | | | | | |

Create Map for Inputs Tab. Train

94.1 % No Tab. input

Create Map for Inputs Images Train

Create Map for Inputs Text Train

Get the Tab. features dtype

Create Map for Targets Tab. Train

Transform Targets Tab.

Identify Positive Class (Bin. Classif.)

Unit test for the Tab. train maps

SetupScaffoldStages

Unit test for the Images train maps

Unit test for the Text train maps

Unit test for all the train maps

Unit test for Train DL

Create Map for Inputs Tab. Test

Create Map for Inputs Images Test

Create Map for Inputs Text Test

Unit test for Test DL

Conversion to Submission Format

Conversion to Alt. Submission Format

Implement metric

Final unit test

0 20 40 60 80 100

Rate (%)

Figure 13. Success Rate per Setup Stage. On this percentage stacked bar graph, each row represents a stage in the process of setting up a competition. Its corresponding bar represents the percentage of times this stage is a success, or is failed, or is not reached, or is skipped, averaged over all competitions.

of the data. Additionally, the prompt was enhanced to provide system resource details, including the remaining total time, the total allowed number of iterations, and the number of iterations that had been completed. This update ensured that the LLM had access to comprehensive context for better performance and decision-making during the execution process.

Execution time and Best solutions All computer vision and natural language processing competitions were allocated a runtime budget of four days, while tabular competitions were assigned a two-day budget. This allocation was designed to provide the system with ample time to generate and refine high-quality solutions. For CV and NLP tasks, AIDE was configured with a maximum execution time of 18 hours per node, whereas tabular tasks were limited to 9 hours per node to reflect their generally lower computational requirements. During execution, AIDE was permitted to perform up to 5,000 improvement itera-

tions to iteratively enhance its solutions. Additionally, the system was allowed a maximum debug depth of 3 from an initial node, with a 50% probability of selecting a given node for debugging. For all tasks, AIDE retained the top four solutions based on the competitionspecific evaluation metric.

Kaggle for Real-Time Evaluation AIDE was integrated with Kaggle to facilitate real-time evaluation and validation of its generated submissions. Once the best four submissions were identified, they were submitted to Kaggle for public and private scoring. The resulting Kaggle scores were then used to assess the performance of each submission, providing an external benchmark for comparison and ensuring that the system’s outputs were evaluated against established competition standards.

RAGDatabasesetup ThedatabasefortheReAct(Qwen)+RAGbaselinewascarefullyconstructed by collecting 24 Kaggle competitions that began in or after 2021 and were not included in our benchmark set. The selected competitions span a balanced mix of domains, including7computervision(CV),7naturallanguageprocessing(NLP),7tabulardatachallenges, and 3 multimodal tasks, ensuring diverse coverage of data science problem types.

- Table 8 lists all the selected competitions, along with the number of notebook summaries and technical reports used to build the RAG database, as well as the corresponding year of each competition.

## G Algorithmic Overview of TabPFN Fine-Tuning

Algorithms 3, 4, and 5 present the pseudocode for dataset sampling using KMeans clustering, prediction using TabPFN, and TabPFN fine-tuning, respectively. It is important to note that the input datasets for all TabPFN variants, including the baseline and fine-tuned models, are generated by the setup pipeline of Agent K.

- Algorithm 3 KMeans Clustering-Based Sampling Require: ctx_length, train_data, test_data

- 1: (train_data,test_data) ← Preprocess(train_data, test_data)
- 2: kmeans ← ApplyKMeansClustering(test_data)
- 3: (test_data,train_data) ← AssignClusterLabels(kmeans)
- 4: train_data_subset ← SampleTrainData(train_data,ctx_length)
- 5: return train_data_subset

## H Examples of Prompts and Responses

Below are some examples of prompts and answers from the LLM that our agent can query. Even though they are presented hereafter in a chat format, note that the LLM does not keep a memory of the chat as generally thought of in recent chat-based systems. The prompts are long because at each interaction we add all the necessary information about the specific step the agent is performing, knowing that the queried LLM has no context or records

ID Competition Name Year # Tech. Reports # Notebooks

- 1 UBC-OCEAN 2024 10 10
- 2 womens-march-mania-2022 2022 7 10
- 3 predict-student-performance-from-game-play 2023 10 10
- 4 novozymes-enzyme-stability-prediction 2023 7 10
- 5 czii-cryo-et-object-identification 2025 10 10
- 6 jigsaw-toxic-severity-rating 2022 10 10
- 7 feedback-prize-english-language-learning 2022 10 10
- 8 icr-identify-age-related-conditions 2023 10 10
- 9 happy-whale-and-dolphin 2022 10 10
- 10 commonlitreadabilityprize 2021 10 10
- 11 amex-default-prediction 2022 10 10
- 12 rsna-miccai-brain-tumor-radiogenomic-classification 2021 10 10
- 13 rsna-2023-abdominal-trauma-detection 2023 10 10
- 14 commonlit-evaluate-student-summaries 2023 10 10
- 15 feedback-prize-effectiveness 2022 10 10
- 16 predict-energy-behavior-of-prosumers 2024 10 10
- 17 mens-march-mania-2022 2022 6 10
- 18 petfinder-pawpularity-score 2022 10 10
- 19 cassava-leaf-disease-classification 2021 10 10
- 20 llm-detect-ai-generated-text 2024 10 10
- 21 home-credit-credit-risk-model-stability 2024 2 10
- 22 rsna-breast-cancer-detection 2023 10 10
- 23 learning-agency-lab-automated-essay-scoring-2 2024 8 10
- 24 hpa-single-cell-image-classification 2021 10 10

###### Table 8. Overview of Inputs for RAG Database: Selected competitions, Year of the competitions, Number of technical reports and Number of Notebook summaries .

- Algorithm 4 TabPFN-Based Prediction Require: ctx_len, train, test, is_regression

- 1: test_batches ← CreateBatches(test, ctx_len)
- 2: all_predictions ← [ ]
- 3: for all test_batch in test_batches do
- 4: if len(train) > ctx_len then
- 5: sampled_ctx ← KMeansSampling(ctx_len, train, test_batch)
- 6: else
- 7: sampled_ctx ← train
- 8: end if
- 9: x_ctx ← getFeatures(sampled_ctx)
- 10: y_ctx ← getLabels(sampled_ctx)
- 11: x_target ← getFeatures(test_batch)
- 12: predictions ← TabPFN(x_ctx, y_ctx, x_target, is_regression)
- 13: all_predictions.append(predictions)
- 14: end for
- 15: return all_predictions

- Algorithm 5 TabPFN Fine-Tuning Require: ctx_len, org_train, org_test, is_regression

- 1: train_data,val_data, test_data ← SplitDataset(org_train)
- 2: val_data_subsets ← CreateBatches(val_data, ctx_len)
- 3: train_data_subsets ← [ ]
- 4: for all v in val_data_subsets do
- 5: sample ← KMeansSampling(ctx_len, train_data, v)
- 6: train_data_subsets.append(sample)
- 7: end for
- 8: for i = 1 to len(train_data_subsets) do
- 9: train_subset ← train_data_subsets[i]
- 10: val_subset ← val_data_subsets[i]
- 11: model ← LoadModel(model_path, is_regression)
- 12: initial_loss ← EvaluateModel(model, train_subset, test_data)
- 13: for all data in TrainDataLoader(train_subset) do
- 14: model ← FineTune(model, data, val_subset)
- 15: validation_loss ← EvaluateModel(model, data, test_data)
- 16: end for
- 17: end for
- 18: all_preds ← Pred(model_path, org_train, org_test, is_regression)
- 19: return all_preds

of previous interactions. The prompts are therefore shortened for the purpose of display. The paths in the code snippets are taken from our runs so we also anonymised them e.g. /PATH/TO/RAW/DATA/... or /PATH/TO/WORKSPACE/...

A Note on Prompts The agent is performing a complex task broken up into a sequence of smaller subtasks. These include summarising information and also past responses. These summaries are the main bulk of the text present in the prompts. They include summarisation of the competition description, the raw data and all past steps the agent has done if relevant. The other big part of the prompts are views. These are usually a raw text output from a terminal command such as “$ls”or python “print()” statement, e.g. the view of the raw data directory or a head view of some dataframes along with their column types, etc. Finally, some prompts can contain examples of similar situations seen in similar competitions that the agent can pick itself and reuse as examples through RAG. The rest is a set of general instructions manually written.

#### H.1 Setup

The example below is taken from the setup of the competition leaf-classification which has tabular and image inputs. We show examples of prompts and responses from the LLM in the particular case where the agent creates a plan for coding and then fill a code template based on the plan. They contain all the necessary information about the task and instructions about the specific step the agent is currently doing for a successful answer. In the particular case below, the LLM is prompted to generate a plan and then the code to create the training map for the image modality, having already done the tabular modality.

###### Prompt

# Data Description Dataset Description Thedatasetconsistsapproximately1,584imagesofleafspecimens(16samples each of 99 species) which have been converted to binary black leaves against white backgrounds. Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a fine-scale margin histogram. For each feature, a 64-attribute vector is given per leaf sample. Note that of the original 100 species, we have eliminated one on account of incomplete associated data in the original dataset.

File descriptions train.csv - the training set test.csv - the test set sample_submission.csv - a sample submission file in the correct format images - the image files (each image is named with its corresponding id)

Data fields id - an anonymous id unique to an image margin_1, margin_2, margin_3, ..., margin_64 - each of the 64 attribute vectors

for the margin feature shape_1, shape_2, shape_3, ..., shape_64 - each of the 64 attribute vectors for the shape feature texture_1, texture_2, texture_3, ..., texture_64 - each of the 64 attribute vectors for the texture feature

### View of the top-level directory, these are the data you can use to solve this task:

|/PATH/TO/RAW/DATA<br><br>+-leaf -classification/ |<br><br>+-images/ | +-226.jpg | +-342.jpg | +-1534.jpg | +-... <additional files > |<br><br>+-train.csv |<br><br>+-test.csv|
|---|

# Modalities Description For the leaf classification task, the input modalities needed to solve the task are as follows:

### Input Modalities

- - **Tabular Input Map:**
- - **YES** because the dataset includes tabular data with pre-extracted features (margin, shape, and texture).
- - **Related Files:**
- - ‘train.csv‘
- - ‘test.csv‘
- - **Image Input Map:**
- - **YES** as there are images of leaf specimens that can be used for additional feature extraction or as an alternative input.
- - **Related Files:**
- - ‘images/‘ (directory containing the leaf images)
- - **Text Input Map:**
- - **NO** because there is no presence of text modality in this task. ### Summary
- - **Tabular Input Map:** YES -> Related Files: ‘train.csv‘, ‘test.csv‘

- - **Image Input Map:** YES -> Related Files: ‘images/
- - **Text Input Map:** NO

This structured approach ensures that all relevant input modalities are considered and utilized for the leaf classification task.

### These are some of the tables you have asked information about: #### View of table ‘/PATH/TO/RAW/DATA/leaf-classification/train.csv‘:

|/PATH/TO/RAW/DATA/leaf -classification/train.csv<br><br>... <Skipped for prompt display purposes in paper appendix ><br><br>... [2 rows x 194 columns]<br><br>|
|---|

<class ’pandas.core.frame.DataFrame’> RangeIndex: 990 entries , 0 to 989 Columns: 194 entries , id to texture64 dtypes: float64 (192), int64(1), object(1) memory usage: 1.5+ MB

- - column species contains strings with values in [’ Acer_Opalus’, ’Pterocarya_Stenoptera’, ’ Quercus_Hartwissiana’, ’Tilia_Tomentosa’, ’ Quercus_Variabilis’, ’Magnolia_Salicifolia ’, ’ Quercus_Canariensis’, ’Quercus_Rubra’, ’Quercus_Brantii’,

’Salix_Fragilis’, ’Zelkova_Serrata’, ’ Betula_Austrosinensis’, ’Quercus_Pontica’, ’ Quercus_Afares’, ’Quercus_Coccifera’, ’Fagus_Sylvatica’, ’Phildelphus’, ’Acer_Palmatum’, ’Quercus_Pubescens’, ’ Populus_Adenopoda’, ’Quercus_Trojana’, ’Alnus_Sieboldiana ’, ’Quercus_Ilex’, ’Arundinaria_Simonii’, ’ Acer_Platanoids’, ’Quercus_Phillyraeoides ’, ’ Cornus_Chinensis’, ’Liriodendron_Tulipifera ’, ’ Cytisus_Battandieri’, ’Rhododendron_x_Russellianum ’, ’ Alnus_Rubra’, ’Eucalyptus_Glaucescens’, ’ Cercis_Siliquastrum’, ’Cotinus_Coggygria ’, ’ Celtis_Koraiensis’, ’Quercus_Crassifolia ’, ’ Quercus_Kewensis’, ’Cornus_Controversa’, ’ Quercus_Pyrenaica’, ’Callicarpa_Bodinieri ’, ’ Quercus_Alnifolia’, ’Acer_Saccharinum’, ’ Prunus_X_Shmittii’, ’Prunus_Avium’, ’Quercus_Greggii’, ’ Quercus_Suber’, ’Quercus_Dolicholepis’, ’Ilex_Cornuta’, ’ Tilia_Oliveri’, ’Quercus_Semecarpifolia ’, ’Quercus_Texana ’, ’Ginkgo_Biloba’, ’Liquidambar_Styraciflua ’, ’ Quercus_Phellos’, ’Quercus_Palustris’, ’ Alnus_Maximowiczii’, ’Quercus_Agrifolia’, ’Acer_Pictum’,

’Acer_Rufinerve’, ’Lithocarpus_Cleistocarpus ’, ’ Viburnum_x_Rhytidophylloides ’, ’Ilex_Aquifolium’, ’ Acer_Circinatum’, ’Quercus_Coccinea’, ’Quercus_Cerris’, ’ Quercus_Chrysolepis’, ’Eucalyptus_Neglecta ’, ’ Tilia_Platyphyllos’, ’Alnus_Cordata’, ’Populus_Nigra’, ’ Acer_Capillipes’, ’Magnolia_Heptapeta’, ’Acer_Mono’, ’ Cornus_Macrophylla’, ’Crataegus_Monogyna ’, ’ Quercus_x_Turneri’, ’Quercus_Castaneifolia ’, ’ Lithocarpus_Edulis’, ’Populus_Grandidentata ’, ’ Acer_Rubrum’, ’Quercus_Imbricaria’, ’Eucalyptus_Urnigera ’ , ’Quercus_Crassipes’, ’Viburnum_Tinus’, ’Morus_Nigra’, ’ Quercus_Vulcanica’, ’Alnus_Viridis’, ’Betula_Pendula’, ’ Olea_Europaea’, ’Quercus_Ellipsoidalis ’, ’ Quercus_x_Hispanica’, ’Quercus_Shumardii ’, ’ Quercus_Rhysophylla’, ’Castanea_Sativa’, ’ Ulmus_Bergmanniana’, ’Quercus_Nigra’, ’Salix_Intergra’, ’ Quercus_Infectoria_sub ’, ’Sorbus_Aria’]

Your goal is to create a map for the training image inputs. The first column is the "id" containing the unique ID of each data point, you need to create id so it can be used later to match each data point to their target. For example, the filename name can be use as unique identifier. The subsequent columns are the paths to the training input images for ‘train_img_input_map.csv‘. Do not include target columns in this table.

Now, write a **short** (no more than 3 steps) plan of how you would do that. If there is an error in a previous attempt, explain in **one** sentence why this error happened and how to fix it. Do not write code yet, simply the plan.

Assistant ### Plan to Create the Training Image Input Map

- 1. **Extract Image IDs and Paths:**

- - Read the ‘train.csv‘ file to extract the ‘id‘ column, which contains the unique IDs for each image.
- - Construct the image file paths using the ‘id‘ values and the directory path to the images (e.g., ‘images/226.jpg‘).

- 2. **Create the Image Input Map CSV:**

- - Create a new DataFrame with two columns: ‘id‘ and ‘image_path‘.
- - Populate the ‘image_path‘ column with the constructed file paths.

- 3. **Save the Image Input Map CSV:**

- - Save the DataFrame to a CSV file named ‘train_img_input_map.csv‘.

- ### Error Handling
- - If the ‘id‘ values in ‘train.csv‘ do not match the filenames in the ‘images/‘ directory, ensure that the filenames are correctly formatted and match the ‘id‘ values exactly.

###### Prompt

|<Skipped first part of prompt repeating the previous prompt<br><br>for display purposes in paper appendix >|
|---|

### Create ‘train_img_input_map.csv‘ Your goal is to create a map for the training image inputs. The first column of that map must be named "id" and must contain the unique ID of each data point. If this column exists already in the data, make sure to rename it to "id", otherwise create it. The subsequent columns must be the paths to the training input images. Do not include target columns in the table you are creating, this will be done later in another table.

### This is the code you helped write earlier to create the train tabular inputs:

- 1 """

- 2 This script creates the data map for the tabular training inputs , i.e. ‘train_tab_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_tab_input_map.csv ‘ are the tabular training input features , if any.

- 6

- 7 >>> # Example

- 8 >>> train_tab_input_map = pd.DataFrame(columns =["id", " tab_input_feature1", "tab_input_feature2", ...])

- 9

- 10 Create train_tab_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15

- 16 import os

- 17 import pandas as pd

- 18

- 19 # Define the absolute path to the train.csv file

- 20 train_csv_path = "/PATH/TO/RAW/DATA/leaf -classification/

- train.csv"
- 21

- 22 # Read the train.csv file

- 23 train_df = pd.read_csv(train_csv_path)

- 24

- 25 # Select the relevant columns: id and all feature columns , excluding the species column

- 26 train_tab_input_map = train_df.drop(columns=[’species’])

- 27

- 28 # Save the new DataFrame to a CSV file

- 29 train_tab_input_map.to_csv("./ train_tab_input_map.csv", index=False)

- 30 print("‘train_tab_input_map.csv ‘ created and saved.")

- 31

- 32

- 33 # <|fim_end|>

- 34 # save

- 35 train_tab_input_map.to_csv("./ train_tab_input_map.csv", index=False)

- 36 print("‘train_tab_input_map.csv ‘ created and saved.")

The script reads the train.csv file, selects the id and feature columns (excluding species), and saves the result to ‘train_tab_input_map.csv‘.

### Complete the code Complete the following code template in order to create the data map ‘train_img_input_map.csv‘:

- 1 """

- 2 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 6

- 7 >>> # Example

- 8 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 9

- 10 Create train_img_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15 # <|fim_hole|>

- 16

- 17 # <|fim_end|>

- 18 # save

- 19 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 20 print("‘train_img_input_map.csv ‘ created and saved.")

Pay EXTREME attention to the path where you load from. You should always provide absolute path to avoid ‘FileNotFound‘ error. Do not re-set index of the dataframe to create a new id column, instead use the columns present in the dataframe already. Create a new index column only when absolutely necessary. Finally, the step to save the final table is already written in the template - do not rewrite this part. The map is saved in the current directory on purpose.

Tip: IF you are using the image file name as id, make sure they are converted to strings. ### Plan: ### Plan to Create the Training Image Input Map

- 1. **Extract Image IDs and Paths:**

- - Read the ‘train.csv‘ file to extract the ‘id‘ column, which contains the unique IDs for each image.
- - Construct the image file paths using the ‘id‘ values and the directory path to the images (e.g., ‘images/226.jpg‘).

- 2. **Create the Image Input Map CSV:**

- - Create a new DataFrame with two columns: ‘id‘ and ‘image_path‘.
- - Populate the ‘image_path‘ column with the constructed file paths.

- 3. **Save the Image Input Map CSV:**

- - Save the DataFrame to a CSV file named ‘train_img_input_map.csv‘. ### Error Handling
- - If the ‘id‘ values in ‘train.csv‘ do not match the filenames in the ‘images/‘ directory, ensure that the filenames are correctly formatted and match the ‘id‘ values exactly. Your response should be the code to complete the above template, in the format: “‘python code “‘

###### Assistant

|""" This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.<br><br>This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.<br><br>The first column of the created map must be the "id" of each test data point. The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.<br><br>>>> # Example >>> train_img_input_map = pd.DataFrame(columns =["id", "<br><br>img_input1", "img_input2", ...])<br><br>Create train_img_input_map """ # <|fim_begin|> import os import pandas as pd<br><br># Define the absolute path to the train.csv file train_csv_path = "/PATH/TO/RAW/DATA/leaf -classification/<br><br>train.csv" images_dir = "/PATH/TO/RAW/DATA/leaf -classification/images/ "<br><br># Read the train.csv file train_df = pd.read_csv(train_csv_path)<br><br># Ensure the id column is a string train_df[’id’] = train_df[’id’].astype(str)<br><br># Construct the image file paths train_df[’image_path’] = train_df[’id’].apply(lambda x: os.<br><br>path.join(images_dir , f"{x}.jpg"))<br><br># Create the image input map DataFrame train_img_input_map = train_df[[’id’, ’image_path’]]<br><br># <|fim_end|> # save train_img_input_map.to_csv("./ train_img_input_map.csv",<br><br>index=False)<br><br>print("‘train_img_input_map.csv ‘ created and saved.")|
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

#### H.2 Setup with RAG Example

In the example below we show how an example of a prompt containing an example retrieve looks like. This is during the setup of the competition ml-olympiad-landscape-image-classification and in particular when generating the code for the training map for the image modality. The previous attempt had en error in the unit test which is shown in the prompt as the agent is asked to retry. The RAG engine retrieves some similar example from a competition already setup. In that other competition, namely kaggle-pog-series-s01e03, the agent had a similar issue and managed to fix it. These steps are retrieved and displayed in the current prompt so that the agent can take inspiration from that similar error being fixed and try to adapt a fix to the current issue it is facing.

###### Prompt

### Data Description Dataset Description The data contains masked images with random patches. These random patches can cover anything between 20% to 60% of the image. Your goal is to correctly predict the classes of the images from the "test" directory. All images are 256x256 having 3 channels.

Files train.csv - the training set test.csv - the test set sample.csv - a sample submission file in the correct format Columns image - Path of the image relative to directory. Eg. In train.csv, mountain/12345.jpg is relative to the "train" directory, whereas in test.csv, 12345.jpg is relative to the "test" directory. label - True label of the image (only in train.csv) . Submission files should have these column headers, but not the column numbers. Usedf.to_csv(filepath,index=False)tosavewithoutthecolumnnumbers.

### View of the top-level directory, these are the data you can use to solve this task:

/PATH/TO/RAW/DATA

+-ml-olympiad -landscape -image -classification/ |

+-train/ | | | +-forest/ | | + -64935341.jpg | | + -89262487.jpg | | + -16354101.jpg

| | +-... <additional files > | | | +-buildings/ | | + -15220800.jpg | | + -30615322.jpg | | + -48061322.jpg | | +-... <additional files > | | | +-glacier/ | | + -10983747.jpg | | + -72741892.jpg | | + -11747834.jpg | | +-... <additional files > | | | +-street/ | | + -16518706.jpg | | + -88133361.jpg | | + -17503215.jpg | | +-... <additional files > | | | +-mountain/ | | + -11476151.jpg | | + -14340600.jpg | | + -72470093.jpg | | +-... <additional files > | +-... <additional folders > |

+-test/ | + -62449337.jpg | + -18432742.jpg | + -84761017.jpg | +-... <additional files > |

+-test.csv |

+-train.csv

### Modalities Description ### Input Modalities for the Task

- - **tabular input map**: NO because there is no presence of tabular data (other than the image paths and labels, which are not used as features in the model).
- - **image input map**: YES as there are images in this task -> related files:
- - ‘/PATH/TO/RAW/DATA/ml-olympiad-landscape-imageclassification/train/‘ (training images)
- - ‘/PATH/TO/RAW/DATA/ml-olympiad-landscape-image-classification/test/‘ (test images)

- - **text input map**: NO because there is no presence of text modality in this task. ### Summary
- - **Tabular Input**: NO
- - **Image Input**: YES (related files: ‘train/‘, ‘test/‘)
- - **Text Input**: NO ### These are some tips to avoid common mistakes in the code:
- - To avoid the error related to non-unique filenames, append the class name to the filenames to ensure they are unique. For example, change ’12345678.jpg’ in the ’forest’ directory to ’forest_12345678.jpg’. This can be done by concatenating the class name with the filename, separated by an underscore or another delimiter.
- - To avoid the ’File not found’ error, ensure that all file paths listed in the ’train.csv’ and ’test.csv’ files are correct and that the corresponding files exist in the dataset. You can write a script to verify the existence of all files listed in these CSV files.

### Create ‘train_img_input_map.csv‘ Your goal is to create a map for the training image inputs. The first column of that map must be named "id" and must contain the unique ID of each data point. If this column exists already in the data, make sure to rename it to "id", otherwise create it. The subsequent columns must be the paths to the training input images. Do not include target columns in the table you are creating, this will be done later in another table.

### This was the resulting code after your last attempt:

- 1 """

- 2 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 6

- 7 >>> # Example

- 8 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 9

- 10 Create train_img_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15

- 16 """

- 17 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 18 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 19 The first column of the created map must be the "id" of each test data point.

- 20 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 21

- 22 >>> # Example

- 23 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 24

- 25 Create train_img_input_map

- 26 """

- 27 # <|fim_begin|>

- 28 import os

- 29 import pandas as pd

- 30

- 31 # Load the training data

- 32 train_df = pd.read_csv("/PATH/TO/RAW/DATA/ml-olympiad landscape -image -classification/train.csv")

- 33

- 34 # Extract the image filenames and use them as unique IDs

- 35 train_df[’id’] = train_df.apply(lambda row: f"{row[’label ’]}_{row[’image ’].split(’/’)[-1]}", axis=1)

- 36

- 37 # Ensure that the filenames are unique

- 38 if train_df[’id’].nunique() != len(train_df):

- 39 raise ValueError("Filenames are not unique. Please append a unique identifier to duplicate filenames.")

- 40

- 41 # Create the input map DataFrame

- 42 train_img_input_map = train_df[[’id’, ’image’]].rename( columns={’image’: ’img_input1’})

- 43

- 44 # <|fim_end|>

- 45 # save

- 46 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 47 print("‘train_img_input_map.csv ‘ created and saved.")

- 48

- 49

- 50 # <|fim_end|>

- 51 # save

- 52 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 53 print("‘train_img_input_map.csv ‘ created and saved.") ### This was the output:

|‘train_img_input_map.csv ‘ created and saved. ‘train_img_input_map.csv ‘ created and saved.<br><br>|
|---|

###### ### The error in the unit test was:

multiprocessing.pool.RemoteTraceback: """ Traceback (most recent call last):

File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 125, in worker

result = (True , func(*args , **kwds))

File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 48, in mapstar

return list(map(*args))

File "/PATH/TO/WORKSPACE/ml-olympiad -landscape -image classification/seed_0/unit_test_map_img_input_train.py" , line 45, in check_file_exists

raise FileNotFoundError(f"File not found: {file_path}") FileNotFoundError: File not found: glacier /59804579. jpg """

The above exception was the direct cause of the following exception:

Traceback (most recent call last):

File "/PATH/TO/WORKSPACE/ml-olympiad -landscape -image classification/seed_0/unit_test_map_img_input_train.py" , line 65, in <module >

check_files(files_to_check)

File "/PATH/TO/WORKSPACE/ml-olympiad -landscape -image classification/seed_0/unit_test_map_img_input_train.py" , line 58, in check_files

results = pool.map(check_file_exists , file_list)

File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 367, in map

return self._map_async(func , iterable , mapstar ,

chunksize).get()

File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 774, in get

raise self._value FileNotFoundError: File not found: glacier /59804579. jpg

###### You should fix this error next. ### An example of a similar error being fixed: Failing code:

- 1 """

- 2 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 6

- 7 >>> # Example

- 8 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 9

- 10 Create train_img_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15

- 16 """

- 17 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 18 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 19 The first column of the created map must be the "id" of each test data point.

- 20 The subsequent columns of ‘train_img_input_map.csv ‘ are the

paths to the training input images , if any.

- 21

- 22 >>> # Example

- 23 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 24

- 25 Create train_img_input_map

- 26 """

- 27 # <|fim_begin|>

- 28 import os

- 29 import pandas as pd

- 30

- 31 # Define the absolute path to the train.csv file

- 32 train_csv_path = "/PATH/TO/RAW/DATA/kaggle -pog -series s01e03/train.csv"

- 33

- 34 # Load the train.csv file into a DataFrame

- 35 train_df = pd.read_csv(train_csv_path)

- 36

- 37 # Generate a unique ID by concatenating seed_id and view

- 38 train_df[’id’] = train_df[’seed_id’].astype(str) + ’_’ + train_df[’view’]

- 39

- 40 # Check for duplicates in the generated ID

- 41 if train_df[’id’].duplicated().any():

- 42 raise ValueError("There are duplicate IDs in the generated unique ID column.")

- 43

- 44 # Construct the full path to each image

- 45 base_image_dir = "/PATH/TO/RAW/DATA/kaggle -pog -series s01e03/train/"

- 46 train_df[’img_input1’] = base_image_dir + train_df[’image’]

- 47

- 48 # Select the relevant columns: ’id’ and ’img_input1’

- 49 train_img_input_map = train_df[[’id’, ’img_input1’]]

- 50

- 51 # <|fim_begin|>

- 52 # save

- 53 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 54 print("‘train_img_input_map.csv ‘ created and saved.")

- 55

- 56

- 57 # <|fim_end|>

- 58 # save

- 59 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 60 print("‘train_img_input_map.csv ‘ created and saved.") Failing code error:

|multiprocessing.pool.RemoteTraceback: """ Traceback (most recent call last):<br><br>File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 125, in worker<br><br>result = (True , func(*args , **kwds))<br><br>File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 48, in mapstar<br><br>return list(map(*args))<br><br>File "/PATH/TO/WORKSPACE/kaggle -pog -series -s01e03/seed_0/ unit_test_map_img_input_train.py", line 45, in check_file_exists<br><br>raise FileNotFoundError(f"File not found: {file_path}") FileNotFoundError: File not found: /PATH/TO/RAW/DATA/kaggle -<br><br>pog -series -s01e03/train/train /03511. png """<br><br>The above exception was the direct cause of the following exception:<br><br>Traceback (most recent call last): File "/PATH/TO/WORKSPACE/kaggle -pog -series -s01e03/seed_0/<br><br>unit_test_map_img_input_train.py", line 65, in <module > check_files(files_to_check)<br><br>File "/PATH/TO/WORKSPACE/kaggle -pog -series -s01e03/seed_0/ unit_test_map_img_input_train.py", line 58, in check_files<br><br>results = pool.map(check_file_exists , file_list)<br><br>File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 367, in map<br><br>return self._map_async(func , iterable , mapstar ,<br><br>chunksize).get()<br><br>File "/home/jdoran/uv_python/cpython -3.10.15 -linux -x86_64 gnu/lib/python3 .10/ multiprocessing/pool.py", line 774, in get<br><br>raise self._value FileNotFoundError: File not found: /PATH/TO/RAW/DATA/kaggle pog -series -s01e03/train/train /03511. png<br><br>|
|---|

###### Fixed code:

- 1 """

- 2 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 6

- 7 >>> # Example

- 8 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 9

- 10 Create train_img_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15

- 16 """

- 17 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 18 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 19 The first column of the created map must be the "id" of each test data point.

- 20 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 21

- 22 >>> # Example

- 23 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 24

- 25 Create train_img_input_map

- 26 """

- 27 # <|fim_begin|>

- 28 import os

- 29 import pandas as pd

- 30

- 31 # Define the absolute path to the train.csv file

- 32 train_csv_path = "/PATH/TO/RAW/DATA/kaggle -pog -series s01e03/train.csv"

- 33

- 34 # Load the train.csv file into a DataFrame

- 35 train_df = pd.read_csv(train_csv_path)

- 36

- 37 # Generate a unique ID by concatenating seed_id and view

- 38 train_df[’id’] = train_df[’seed_id’].astype(str) + ’_’ + train_df[’view’]

- 39

- 40 # Check for duplicates in the generated ID

- 41 if train_df[’id’].duplicated().any():

- 42 raise ValueError("There are duplicate IDs in the generated unique ID column.")

- 43

- 44 # Construct the full path to each image

- 45 base_image_dir = "/PATH/TO/RAW/DATA/kaggle -pog -series s01e03/train/"

- 46 train_df[’img_input1’] = base_image_dir + train_df[’image’ ].str.split(’/’).str[-1]

- 47

- 48 # Select the relevant columns: ’id’ and ’img_input1’

- 49 train_img_input_map = train_df[[’id’, ’img_input1’]]

- 50

- 51 # <|fim_end|>

- 52 # save

- 53 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 54 print("‘train_img_input_map.csv ‘ created and saved.")

- 55

- 56

- 57 # <|fim_end|>

- 58 # save

- 59 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 60 print("‘train_img_input_map.csv ‘ created and saved.")

###### ### Complete the code Complete the following code template in order to create the data map ‘train_img_input_map.csv‘:

- 1 """

- 2 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_img_input_map.csv ‘ are the

paths to the training input images , if any.

- 6

- 7 >>> # Example

- 8 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 9

- 10 Create train_img_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15 # <|fim_hole|>

- 16

- 17 # <|fim_end|>

- 18 # save

- 19 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 20 print("‘train_img_input_map.csv ‘ created and saved.")

Pay EXTREME attention to the path where you load from. You should always provide absolute path to avoid ‘FileNotFound‘ error. Do not re-set index of the dataframe to create a new id column, instead use the columns present in the dataframe already. Create a new index column only when absolutely necessary. Finally, the step to save the final table is already written in the template - do not rewrite this part. The map is saved in the current directory on purpose.

Tip: IF you are using the image file name as id, make sure they are converted to strings.

### Plan:

- 1. **Load the Training Data**: Read the ‘train.csv‘ file to get the image paths and labels.
- 2. **Create Unique IDs**: Generate a unique ID for each image by concatenating the label and the image filename.
- 3. **Construct the Input Map**: Create a DataFrame with the unique IDs and the corresponding image paths, and save it as ‘train_img_input_map.csv‘.

### Error Explanation: The error occurred because the file path in the ‘train.csv‘ file was incorrect or the file did not exist in the specified directory. To fix it, ensure that the file paths in ‘train.csv‘ are correct and that the files exist in the ‘train/‘ directory. Your response should be the code to complete the above template, in the format: “‘python code

“‘

###### Assistant

- 1 """

- 2 This script creates the data map for the image training inputs , i.e. ‘train_img_input_map.csv ‘.

- 3 This map is a ‘pd.DataFrame ‘ object populated with respect to the data description and the data present in the raw data directory.

- 4 The first column of the created map must be the "id" of each test data point.

- 5 The subsequent columns of ‘train_img_input_map.csv ‘ are the paths to the training input images , if any.

- 6

- 7 >>> # Example

- 8 >>> train_img_input_map = pd.DataFrame(columns =["id", " img_input1", "img_input2", ...])

- 9

- 10 Create train_img_input_map

- 11 """

- 12 # <|fim_begin|>

- 13 import os

- 14 import pandas as pd

- 15

- 16 # Define the absolute path to the train.csv file

- 17 train_csv_path = "/PATH/TO/RAW/DATA/ml-olympiad -landscape image -classification/train.csv"

- 18

- 19 # Load the train.csv file into a DataFrame

- 20 train_df = pd.read_csv(train_csv_path)

- 21

- 22 # Generate a unique ID by concatenating the label and the image filename

- 23 train_df[’id’] = train_df.apply(lambda row: f"{row[’label ’]}_{row[’image ’].split(’/’)[-1]}", axis=1)

- 24

- 25 # Check for duplicates in the generated ID

- 26 if train_df[’id’].duplicated().any():

- 27 raise ValueError("There are duplicate IDs in the generated unique ID column.")

- 28

- 29 # Construct the full path to each image

- 30 base_image_dir = "/PATH/TO/RAW/DATA/ml-olympiad -landscape image -classification/train/"

- 31 train_df[’img_input1’] = base_image_dir + train_df[’image’]

- 32

- 33 # Select the relevant columns: ’id’ and ’img_input1’

- 34 train_img_input_map = train_df[[’id’, ’img_input1’]]

- 35

- 36 # <|fim_end|>

- 37 # save

- 38 train_img_input_map.to_csv("./ train_img_input_map.csv", index=False)

- 39 print("‘train_img_input_map.csv ‘ created and saved.")

#### H.3 Solution Generation

The example below is taken from the solution generation pipeline on the comeptition dogs-vs-cats-redux-kernels-edition. In this particular example, the agent has already done one submission which appears in some prompts. The agent is at the start of a new submission, generating the plans and codes for preprocessing the images and the image embedding model.

###### Prompt

## Task description ### Summary of the Task Description

**Main Goal:** The goal of this competition is to classify images as either dogs or cats. The competition revisits the original "Dogs vs. Cats" challenge from 2013, but with modern machine learning techniques and tools, such as Kernels (Kaggle’s online coding environment) and TensorFlow.

**Task Type:** This is a binary classification task where the model must predict the probability that a given image is a dog. The evaluation metric is log loss, which measures the performance of the model by penalizing in correct predictions more severely as the predicted probability diverges from the actual label.

**Key Points:**

- - **Dataset:** The dataset consists of images of dogs and cats.
- - **Evaluation Metric:** Log loss.
- - **Submission Format:** A CSV file with the format ‘id,label‘, where ‘id‘ is the image ID and ‘label‘ is the predicted probability that the image is a dog.

## Data description ### View of the top-level directory /PATH/TO/RAW/DATA/dogs-vs-catsredux-kernels-edition: after having downloaded and extracted the raw data from the source, these are

the data you can use to solve this task:

|/PATH/TO/RAW/DATA<br><br>+-dogs -vs-cats -redux -kernels -edition/ |<br><br>+-test/ | +-8715.jpg | +-12271.jpg | +-156.jpg | +-... <additional files > |<br><br>+-train/<br><br>+-cat.8852.jpg +-dog.9444.jpg +-cat .12231. jpg<br><br>+-... <additional files >|
|---|

### Summary of Data Organization The dataset is organized as follows: #### Directory Structure

|/PATH/TO/RAW/DATA/dogs -vs-cats -redux -kernels -edition/ |<br><br>+-test/ | +-8715.jpg | +-12271.jpg | +-156.jpg | +-... (additional files) |<br><br>+-train/ | +-cat.8852.jpg | +-dog.9444.jpg | +-cat .12231. jpg | +-... (additional files) |<br><br>+-sample_submission.csv|
|---|

#### File Descriptions

- 1.**/PATH/TO/RAW/DATA/dogs-vs-cats-redux-kernels-edition/train/**

- - Contains 25,000 images of dogs and cats.
- - Each image filename includes the label:
- - ‘cat.<number>.jpg‘ for cat images.
- - ‘dog.<number>.jpg‘ for dog images.

- 2.**/PATH/TO/RAW/DATA/dogs-vs-cats-redux-kernels-edition/test/**

- - Contains 12,500 images.
- - Each image is named according to a numeric ID (e.g., ‘8715.jpg‘).

- 3.**/PATH/TO/RAW/DATA/dogs-vs-cats-redux-kernelsedition/sample_submission.csv**

- - A sample submission file with the following columns:
- - ‘id‘: The numeric ID of the test image.
- - ‘label‘: The predicted probability that the image is a dog (1 = dog, 0 = cat). #### Feature and Target Names
- - **Features:**
- - ‘id‘: The numeric ID of the test image (used for the test set).
- - **Target:**
- - ‘label‘: The predicted probability that the image is a dog (1 = dog, 0 = cat). ### Note
- -Thedataissavedunder‘/PATH/TO/RAW/DATA/dogs-vs-cats-redux-kernelsedition‘.
- - The directory and file names must remain unchanged.

## Available solutions This is the list of the solutions that have already been generated:

* 2024-10-16-07-22-08: The provided code implements a comprehensive pipeline for the Dogs vs. Cats classification task. It includes a set of torchvision transforms to preprocess images, a PyTorch class ‘ImageEmbedder‘ that uses a pre-trained ResNet50 model to generate latent representations of images, and a ‘TabularHead‘ model to decode these embeddings into classification logits. The pipeline also defines appropriate loss functions for both regression and classification tasks, ensuring a robust and flexible approach to the problem.

## Current solution progress You have started creating a new solution. Its status is as follows:

- - [PENDING] Embedder for images
- - [PENDING] Transform functions for images
- - [TODO] Embedder for transformed images
- - [TODO] Logits and regression targets decoder

## What to do next You have to generate a code to transform the images using a composition of torchvision transforms. You need to write a pytorch torchvision.transforms.Compose sequence that transforms the images appropriately

withrespecttothetask. Youneedtomakesuretheimagesarealsotransformed to torch.Tensor at some point as the images will be fed to a vision neural network. You can use any image transforms (Resize, RandomCrop, ToTensor, ...) or even write custom transforms with Lambda() ...

Output a plan presented with maximum 4 bullet points describing the different steps to construct a pytorch torchvision transforms composition. The plan should be tailored to the current task as it will be used to guide the implementation of the image transforms.

Assistant ### Plan for Constructing PyTorch torchvision Transforms Composition

- 1. **Resize Images**: Resize all images to a standard size (e.g., 224x224 pixels) toensuretheycanbefedintoapre-trainedmodellikeResNet50, whichexpects input of this size.
- 2. **Apply Random Augmentations**: Use random augmentations such as ‘RandomHorizontalFlip‘, ‘RandomRotation‘, and ‘ColorJitter‘ to increase the diversity of the training data and improve the model’s generalization capabilities.
- 3. **Convert to Tensor**: Use ‘ToTensor‘ to convert the images to PyTorch tensors, which is a required format for feeding the images into a neural network.
- 4. **Normalize Images**: Apply normalization using the mean and standard deviationoftheImageNetdataset, aspre-trainedmodelslikeResNet50aretypically trained on ImageNet. This ensures the input data is in the expected range and format for the pre-trained model.

###### Prompt

|<Skipped first part of prompt repeating the previous prompt<br><br>for display purposes in paper appendix >|
|---|

## Current solution progress You have started creating a new solution. Its status is as follows:

- - [PENDING] Embedder for images
- - [PENDING] Transform functions for images
- - [TODO] Embedder for transformed images
- - [TODO] Logits and regression targets decoder

## What to do next You have to generate a code to transform the images using a composi-

tion of torchvision transforms. You need to write a pytorch torchvision.transforms.Compose sequence that transforms the images appropriately withrespecttothetask. Youneedtomakesuretheimagesarealsotransformed to torch.Tensor at some point as the images will be fed to a vision neural network. You can use any image transforms (Resize, RandomCrop, ToTensor, ...) or even write custom transforms with Lambda() ...

For example a good default transform is:

|import numpy as np import pandas as pd import torch from PIL import Image from torch import Tensor from torch.utils.data import Dataset from torchvision import transforms as T<br><br>DefaultImageInputTransform: T.Compose = T.Compose( [<br><br>T.Resize((512, 512)), # Resize to a fixed size ( should make sense with respect to the task , not always needed)<br><br>T.Lambda(lambda x: x if isinstance(x, torch.Tensor)<br><br>else T.ToTensor()(x)), # Convert image to tensor if needed<br><br>T.Lambda(<br><br>lambda x: x.repeat([3 if (i - x.ndim == -3 and d == 1) else 1 for i, d in enumerate(x.shape )])<br><br>), # grayscale to rgb T.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),<br><br># take only first 3 channels ]<br><br>)|
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

Note that the last two transforms are using a Lambda function. The penultimate one is an important one to include as its goal is to duplicate the number of channels to 3 if there happen to be grayscale images in the data mixed with colored images. The ultimate one is taking on ly the first 3 channels in case the images in the dataset have an inconsistent number of channels. These two last transforms hence force all images to have the same number of channels. If you are not sure all images in the data will have the same number of channels, it is advised to add a similar transform. If you find it necessary to normalize the images, do it after taking care fo the number of channels and make sure the vectors you use match the image dimensions.

Use your plans to help you implement the image transform: ### Plan for Constructing PyTorch torchvision Transforms Composition

- 1. **Resize Images**: Resize all images to a standard size (e.g., 224x224 pixels) toensuretheycanbefedintoapre-trainedmodellikeResNet50, whichexpects input of this size.
- 2. **Apply Random Augmentations**: Use random augmentations such as ‘RandomHorizontalFlip‘, ‘RandomRotation‘, and ‘ColorJitter‘ to increase the diversity of the training data and improve the model’s generalization capabilities.
- 3. **Convert to Tensor**: Use ‘ToTensor‘ to convert the images to PyTorch tensors, which is a required format for feeding the images into a neural network.
- 4. **Normalize Images**: Apply normalization using the mean and standard deviation of the ImageNet dataset, as pre-trained models like ResNet50 are typically trained on ImageNet. This ensures the input data is in the expected range and format for the pre-trained model. Write the code to create an image transforms composition.

- 1. You must fill all the required parameters of the transforms you are using in a sensible way with respect to the image shapes of the current task.
- 2. You must not forget to import the packages when needed.
- 3. The transforms should be transforming the images to tensor at the appropriate step.
- 4. Try to use different transforms than this last attempt: The provided code implements a comprehensive pipeline for the Dogs vs. Cats classification task. It includes a set of torchvision transforms to preprocess images, a PyTorch class ‘ImageEmbedder‘ that uses a pre-trained ResNet50 model to generate latent repr esentations of images, and a ‘TabularHead‘ model to decode these embeddings into classification logits. The pipeline also defines appropriate loss functions f or both regression and classification tasks, ensuring a robust and flexible approach to the problem.. Do not repeat all the transforms from the attempt. Your task is to fill the blank and implement the image transform in this code:

- 1 # <|fim_begin|>

- 2 import os

- 3 import pandas as pd

- 4 import torch

- 5 from torch import nn

- 6 from PIL import Image

- 7 import torchvision.transforms as T

- 8 from torch.utils.data import DataLoader

- 9 from tqdm import tqdm

- 10

- 11 from agent.tools.data_map.map_dataset import MapDataset , map_dataset_collate_function

- 12

- 13 # --- Create CustomImageInputTransform , a composition of torchvision.transforms using torchvision.transforms. Compose

- 14 # <|fim_hole|>

- 15 # --- [End]

- 16

- 17 # <|fim_end|>

- 18 # Test that the transform works

- 19 root_path = "/PATH/TO/RAW/DATA/dogs -vs-cats -redux -kernels edition/qwen2.5-72b-vp.3.5 _seed_2"

- 20 tab_target_map_path=os.path.join(root_path , " train_tab_target_map.csv")

- 21 img_target_map_path=os.path.join(root_path , " train_img_target_map.csv")

- 22 train_dataset = MapDataset(

- 23 img_input_map_path=os.path.join(root_path , " train_img_input_map.csv"),

- 24 tab_target_map_path=tab_target_map_path if os.path. exists(tab_target_map_path) else None ,

- 25 img_target_map_path=img_target_map_path if os.path. exists(img_target_map_path) else None ,

- 26 img_input_transform=CustomImageInputTransform ,

- 27 )

- 28 train_dataloader = DataLoader(

- 29 train_dataset , batch_size=256, shuffle=True , collate_fn

=map_dataset_collate_function , num_workers =20

- 30 )

- 31 shapes = set()

- 32 for batch in tqdm(train_dataloader):

- 33 (indices , (_, img_inputs_batch , _), (tab_targets_batch , _, _)) = batch

- 34 if img_inputs_batch.shape not in shapes:

- 35 shapes.add(img_inputs_batch.shape [1:]) # ignore batch size as last batch can have fewer elements

- 36 if len(shapes) > 1:

- 37 raise RuntimeError(

- 38 f"Multiple shapes detected! After applying CusotmImageTransform , some images have shapes "

- 39 f"{’ and ’.join([str(s) for s in shapes])} "

- 40 f"but should have the same shape for all images after the transform. "

- 41 f"(Note that the first dimension is the batch size {train_dataloader.batch_size}).\n"

- 42 f"Hint: if needed add a lambda transform that ensures the number of channels is, e.g. 3 in

this example:\n"

- 43 f"‘T.Lambda(lambda x: x.repeat([3 if (i - x. ndim == -3 and d == 1) else 1 for i, d in enumerate(x.shape)])),‘"

- 44 f"\nand\n‘T.Lambda(lambda x: x[:3] if x.shape [0] > 3 else x)‘"

- 45 )

- 46

- 47 print(f"Transformed image tensor shape: {img_inputs_batch. shape}")

###### Assistant

|CustomImageInputTransform = T.Compose( [<br><br>T.Resize((256, 256)), # Resize to a standard size T.RandomCrop (224), # Randomly crop to the input<br><br>size expected by the model T.RandomHorizontalFlip(p=0.5), # Random horizontal flip with 50% probability T.ColorJitter(brightness =0.2, contrast=0.2,<br><br>saturation =0.2, hue=0.1), # Random color jitter T.ToTensor(), # Convert image to tensor T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using ImageNet mean and std<br><br>T.Lambda(<br><br>lambda x: x.repeat([3 if (i - x.ndim == -3 and d == 1) else 1 for i, d in enumerate(x.shape )])<br><br>), # grayscale to rgb T.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),<br><br># take only first 3 channels ]<br><br>)|
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

###### Prompt

|<Skipped first part of prompt repeating the previous prompt<br><br>for display purposes in paper appendix >|
|---|

## Current solution progress You have started creating a new solution. Its status is as follows:

- - [TODO] Embedder for text fields
- - [TODO] Logits and regression targets decoder

## What to do next You have to generate a code to embed the images using a pre-trained torchvision model. You need to write a pytorch torchvision model, where forward() takes the tensor transformed images as input, and that outputs a latent representation as a ten sor. You can use any pre-trained model from this list: resnet18, resnet50, resnet101, AlexNet, EfficientNetV2, GoogLeNet, MaxVit, MNASNet, RegNet

Output a plan presented with maximum 4 bullet points describing the different steps to construct a pytorch torchvision model to embed images. The plan should be tailored to the current task as it will be used to guide the implementation of the embedder.

Assistant ### Plan to Construct a PyTorch torchvision Model to Embed Images

- 1. **Select and Load Pre-trained Model:**

- - Choose a pre-trained model from the provided list (e.g., ‘resnet50‘).
- - Load the pre-trained model and remove the final fully connected layer to use it for feature extraction.

- 2. **Define Image Preprocessing:**

- - Define a set of torchvision transforms to preprocess the images (e.g., resizing, normalization).
- - Ensure the transforms are consistent with the pre-processing used during the training of the pre-trained model.

- 3. **Create Image Embedder Class:**

- - Define a PyTorch class ‘ImageEmbedder‘ that takes the pre-processed image tensor as input.
- - Use the pre-trained model to extract the latent representation (features) from the images.

- 4. **Implement Forward Method:**

- - Implement the ‘forward‘ method in the ‘ImageEmbedder‘ class to pass the

- input image tensor through the pre-trained model.
- - Return the latent representation tensor as the output of the ‘forward‘ method.

Thisplanwillguidetheimplementationoftheembedder,ensuringitistailored to the current task of classifying dogs and cats.

###### Prompt

|<Skipped first part of prompt repeating the previous prompt<br><br>for display purposes in paper appendix >|
|---|

## Current solution progress You have started creating a new solution. Its status is as follows:

- - [TODO] Embedder for text fields
- - [TODO] Logits and regression targets decoder

## What to do next You have to generate a code to embed the images using a pre-trained torchvision model. You need to write a pytorch torchvision model, where forward() takes the tensor transformed images as input, and that outputs a latent representation as a ten sor. You can use any pre-trained models like resnet18, resnet50, resnet101, AlexNet, EfficientNetV2, GoogLeNet, MaxVit, MNASNet, RegNet, vit etc...

Ensure to unfreeze last few layers of the model based on the model you choose. An example of this model code is:

- 1 import torch

- 2 import torchvision.models as models

- 3 from torch import nn

- 4

- 5 class ImageEmbedder(nn.Module):

- 6 def __init__(self , pretrained=True):

- 7 super().__init__()

- 8

- 9 resnet50 = models.resnet50(pretrained=pretrained)

- 10 self.features = nn.ModuleList(resnet50.children()) [:-1]

- 11 self.features = nn.Sequential(*self.features)

- 12 # optionally freeze pre -trained model weights

- 13 for param in self.features.parameters():

- 14 param.requires_grad = False

- 15

- 16 def forward(self , x):

- 17 output = self.features(x)

- 18 output = output.view(x.size(0), -1)

- 19 return output

- 20

- 21 def unfreeze(self , n_last_layers =2):

- 22 """ Unfreeze some layers for fine -tuning."""

- 23 layers = list(self.children())[:-1]

- 24

- 25 for layer in layers[-n_last_layers :]:

- 26 for param in layer.parameters():

- 27 param.requires_grad = True

Use your plans to help you implement the pytorch embedder model: ### Plan to Construct a PyTorch torchvision Model to Embed Images

- 1. **Select and Load Pre-trained Model:**

- - Choose a pre-trained model from the provided list (e.g., ‘resnet50‘).
- - Load the pre-trained model and remove the final fully connected layer to use it for feature extraction.

- 2. **Define Image Preprocessing:**

- - Define a set of torchvision transforms to preprocess the images (e.g., resizing, normalization).
- - Ensure the transforms are consistent with the pre-processing used during the training of the pre-trained model.

- 3. **Create Image Embedder Class:**

- - Define a PyTorch class ‘ImageEmbedder‘ that takes the pre-processed image tensor as input.
- - Use the pre-trained model to extract the latent representation (features) from the images.

- 4. **Implement Forward Method:**

- - Implement the ‘forward‘ method in the ‘ImageEmbedder‘ class to pass the input image tensor through the pre-trained model.
- - Return the latent representation tensor as the output of the ‘forward‘ method.

This plan will guide the implementation of the embedder, ensuring it is tailored to the current task of classifying dogs and cats.

Write the code to create a model taking tensors of preprocessed features as input, and outputting an embedding tensor.

- 1. You can assume that the input dimension is already computed in the variable ‘input_dim‘.
- 2. You must not forget to import the packages when needed.
- 3. The model should be able to take as input a torch tensor of size ‘(batch,

- input_dim)‘ and output a tensor of shape ‘(batch, latent_dim)‘.
- 4. Try to use newer models than this last attempt: The provided code implements a comprehensive pipeline for the Dogs vs. Cats classification task. It include s a set of torchvision transforms to preprocess images, a PyTorch class ‘ImageEmbedder‘ that uses a pre-trained ResNet50 model to generate latent representations of images, and a ‘TabularHead‘ model to decode these embeddings into classification logits. The pipeline also defines appropriate loss functions for both regression and classification tasks, ensuring a robust and flexible approach to the problem.. Do not repeat these pre-trained models from the attempt. Your task is to fill the blank and implement the image embedder in this code:

- 1 # <|fim_begin|>

- 2 import os.path

- 3

- 4 import torch

- 5 import pandas as pd

- 6 import torchvision.models as models

- 7 import torchvision.transforms as T

- 8 from torch import nn

- 9 from PIL import Image

- 10

- 11 from submissions.new_submission.img_transform import CustomImageInputTransform

- 12

- 13 # Load image features

- 14 root_path = "/PATH/TO/SETUP"

- 15 train_data_path = os.path.join(root_path , " train_img_input_map.csv")

- 16 x = pd.read_csv(train_data_path , index_col="id")

- 17

- 18 # --- Design torch model using a pre -trained torchvision model

- 19 # Create class ImageEmbedder(nn.Module) and implementing the methods:

- 20 # - __init__() defines the model architecture , using a pretrained model such as resnet50

- 21 # - forward:(x: torch.Tensor) -> torch.Tensor , embeds a batch of inputs and loads the image embeddings.

- 22 # <|fim_hole|>

- 23 # --- [End]

- 24

- 25 # Test that the model works

- 26 path = x.iloc[0, 0]

- 27 image = Image.open(path)

- 28 tensor_image = CustomImageInputTransform(image)

- 29 model = ImageEmbedder()

- 30 X_embed = model(tensor_image.unsqueeze (0))

- 31

- 32

- 33 # <|fim_end|>

- 34 # @NO_MEMORY_START@

- 35

- 36 def test_unfreeze_functionality () -> None:

- 37 model = ImageEmbedder()

- 38 assert hasattr(model , ’unfreeze’), "The model does not have an ’unfreeze’ method."

- 39

- 40 try:

- 41 model.unfreeze()

- 42 except Exception as e:

- 43 assert False , f"Unfreeze method raised an exception : {e}"

- 44

- 45 print("Unfreeze method exists and runs successfully.")

- 46

- 47

- 48 test_unfreeze_functionality ()

- 49

- 50 print(f"Embedding size: {X_embed.shape}")

- 51 # @NO_MEMORY_END@

Only output the code that should be filled in the blank with no extra text. Don’t forget to import any additional packages that you will use in the code. Your response should be in the format: “‘python <code> “‘

###### Assistant

- 1 class ImageEmbedder(nn.Module):

- 2 def __init__(self , pretrained=True , latent_dim =2048):

- 3 super().__init__()

- 4

- 5 # Choose a newer model like EfficientNetV2

- 6 efficientnet = models.efficientnet_v2_m(pretrained= pretrained)

- 7 self.features = nn.ModuleList(efficientnet.features

.children())

- 8 self.features = nn.Sequential(*self.features)

- 9 # Optionally freeze pre -trained model weights

- 10 for param in self.features.parameters():

- 11 param.requires_grad = False

- 12

- 13 def forward(self , x):

- 14 output = self.features(x)

- 15 output = output.mean(dim=[2, 3])

- 16

- 17 def unfreeze(self , n_last_layers =2):

- 18 """ Unfreeze some layers for fine -tuning."""

- 19 layers = list(self.features.children())

- 20

- 21 for layer in layers[-n_last_layers :]:

- 22 for param in layer.parameters():

- 23 param.requires_grad = True

#### H.4 Scaffold with CoT

The example below is taken from a natural language processing competition (sentiment-analysis-on-movie-reviews). The solution below was generated by Agent K as one of the draft nodes. Agent K is provided with a CoT from the scaffolding stage to guide its initial solution generations.

###### Prompt

## Introduction You are a Kaggle grandmaster attending a competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python. We will now provide a description of the task.

## BENCHMARK INSTRUCTIONS First, please read the instructions below

- - **Competition-specific instructions**: Competition specific descriptions will be provided.
- - **Dataset**: The dataset files for the competition are available in ‘*/PATH/TO/RAW/DATA/sentiment-analysis-on-movie-reviews‘.Test dataset does not contain any targets.
- - **Submission**: You can find sample submission format inside ‘*/PATH/TO/RAW/DATA/sentiment-analysis-on-movie-reviews‘ as specified ‘Data Overview.Always use same ’id’ available inside sample submission file. You are not allowed to change ’id’.Based on sample submission file you have to generate submission.csv for the test data. - **Cheating**: You are not allowed to view others’ solutions to this competition. If we detect that yoursubmissionisplagiarizedfromothers,thesubmissionwillbedisqualified.

Wherever there is a conflict between the points above and the competitionspecific instructions, competition points always takes priority.

Yoursubmissionwillbegradedbasedonthemetricinthecompetition-specific instructions. Your goal is to achieve the best score possible on the competition task.

## ADDITIONAL NOTES

**Compute**: You have access to 1 Tesla V100-PCIE-32GB with the appropriate drivers installed.

**Total Runtime**: You have a maximum of 172800 seconds to attempt this task. You may iteratively develop your solution within this time frame, but bear in mind that program runtime also counts towards this limit.

**Total Steps**: You have a maximum of 5000 steps to submit your solution. You will be informed of the remaining time and steps at each step .**Use as much of the available time as possible to refine and optimize your submis-

sion**: Your goal is to achieve the best possible score, not just to submit your first working solution.

## Response format Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (wrapped in “‘) which implements this solution and prints out the evaluation metric. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block.

## Solution sketch guideline

- - Carefully examine the past solutions. These solutions may not be identical to the task at hand, but they provide valuable insights on strategies and techniques.
- - Look for common strategies that led to success, such as specific modeling techniques, evaluation metrics, or feature engineering methods.
- - Incorporate the lessons learned from these solutions into your proposed solution, making sure to adapt them to the current task.
- - The solutions should inform how to approach the current task, focusing on critical decisions made in those examples and any relevant best practices.
- - Do not copy the solutions verbatim, but leverage them as a foundation for crafting a novel, efficient, and creative approach.
- - You are encouraged to explore new angles while respecting the context and recommendations from the previous successful strategies.
- - Take the Memory section into consideration when proposing the design
- - don’t propose the same modelling solution but keep the evaluation the same.
- - The solution sketch should be 3-5 sentences.
- - Propose an evaluation metric that is reasonable for this task.
- - Don’t suggest to do EDA.

## Installed Packages Your solution can use any relevant machine learning packages such as: ‘torchvision‘, ‘xgboost‘, ‘timm‘, ‘pandas‘, ‘torch-geometric‘, ‘scikit-learn‘, ‘statsmodels‘, ‘numpy‘, ‘torch‘, ‘bayesian-optimization‘, ‘lightGBM‘. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow.

## Implementation guideline

- - <TOTAL_TIME_REMAINING: 41.0hrs 23.0mins 32secs>
- - <TOTAL_STEPS_REMAINING: 4987>
- - The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,
- - **AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED

- UNLABELED TEST DATA IN A ‘submission.csv‘ FILE IN THE ./submission/ DIRECTORY.**
- - The code should be a single-file python program that is self-contained and can be executed as-is.
- - No parts of the code should be skipped, don’t terminate the before finishing the script.
- - Your response should only contain a single code block.
- - Be aware of the running time of the code, it should complete within 18 hours.
- - Do not copy data to workspace, read data from the source specified.
- - You can use the "./working" directory to store any temporary files that your code needs to create.
- - The evaluation should be based on 5-fold cross-validation but only if that’s an appropriate evaluation for the task at hand. ## Data Overview

|/PATH/TO/RAW/DATA |<br><br>+-sampleSubmission.csv (66293 lines) |<br><br>+-.. and 2 other files|
|---|

## Task description Dataset Description The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data. train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence. test.tsv contains just phrases. You must assign a sentiment label to each phrase. The sentiment labels are:

- 0 - negative
- 1 - somewhat negative
- 2 - neutral
- 3 - somewhat positive
- 4 - positive

Evaluation Submissions are evaluated on classification accuracy (the percent of labels

that are predicted correctly) for every parsed phrase. The sentiment labels are:

- 0 - negative
- 1 - somewhat negative
- 2 - neutral
- 3 - somewhat positive
- 4 - positive Submission Format For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following: PhraseId,Sentiment

- 156061,2
- 156062,2
- 156063,2

... ## Summary of Past Submissions

- ### Submission 0 summary and metric value on a validation set: The project involves a comprehensive pipeline for sentiment analysis. A TextEmbedder class utilizes a pre-trained DistilBERT model to convert text inputs into embeddings. To address class imbalances, appropriate techniques are applied. A PyTorch model with a TabularHead class is implemented to transform the text embeddings into sentiment logits, and both regression and classification loss functions are defined to train the model effectively.

- "submission.csv": 65.57734204793029, "submission_alt.csv":

65.57734204793029

—– ### Submission 1 summary and metric value on a validation set: The code implements a PyTorch model that uses a pre-trained RoBERTa model to embed text fields, tokenizes the input text, and returns the pooler output as the text embedding. It handles class imbalances and includes a TabularHead class to transform the RoBERTa embeddings into sentiment logits using fully connected layers with dropout and batch normalization. The code also implements regression and classification loss functions using Mean Squared Error (MSE) and Cross-Entropy loss, respectively.

- "submission.csv": 66.00794566192491, "submission_alt.csv":

- ### Submission 2 summary and metric value on a validation set: The code implements a PyTorch model that uses a pre-trained BERT model to embed text fields. It tokenizes the input text and returns the pooler output as the text embedding. To handle class imbalances, appropriate techniques are applied. The model also includes a TabularHead, which is a series of fully connected layers that transform the BERT embeddings into sentiment logits.

66.00794566192491

—–

For the classification task, Cross-Entropy loss is used, and for the regression task, Mean Squared Error (MSE) loss is employed. "submission.csv": 66.42573369216967, "submission_alt.csv": 66.42573369216967

###### Assistant

### To tackle this sentiment analysis task, I will use a pre-trained BERT model to extract features from the text and then train a simple classifier on top of these features. The model will be trained using a cross-entropy loss function, and class imbalance will be handled using weighted loss. The performance will be evaluated using classification accuracy on a hold-out validation set..

- 1 import pandas as pd

- 2 import numpy as np

- 3 import torch

- 4 from torch.utils.data import Dataset , DataLoader

- 5 from transformers import BertTokenizer , BertModel

- 6 from sklearn.model_selection import train_test_split

- 7 from sklearn.metrics import accuracy_score

- 8 import torch.nn as nn

- 9 import torch.optim as optim

- 10

- 11

- 12 # Define the dataset

- 13 class SentimentDataset(Dataset):

- 14 def __init__(self , df, tokenizer , max_len =128):

- 15 self.tokenizer = tokenizer

- 16 self.max_len = max_len

- 17 self.df = df

- 18

- 19 def __len__(self):

- 20 return len(self.df)

- 21

- 22 def __getitem__(self , idx):

- 23 text = self.df.iloc[idx]["Phrase"]

- 24 if pd.isna(text):

- 25 text = "" # Handle NaN values

- 26 label = (

- 27 self.df.iloc[idx]["Sentiment"] if "Sentiment" in self.df.columns else None

- 28 )

- 29

- 30 encoding = self.tokenizer.encode_plus(

- 31 text ,

- 32 add_special_tokens=True ,

- 33 max_length=self.max_len ,

- 34 padding="max_length",

- 35 truncation=True ,

- 36 return_attention_mask=True ,

- 37 return_tensors="pt",

- 38 )

- 39

- 40 input_ids = encoding["input_ids"].flatten()

- 41 attention_mask = encoding["attention_mask"].flatten ()

- 42

- 43 if label is not None:

- 44 return {

- 45 "input_ids": input_ids ,

- 46 "attention_mask": attention_mask ,

- 47 "label": torch.tensor(label , dtype=torch. long),

- 48 }

- 49 else:

- 50 return {"input_ids": input_ids , "attention_mask ": attention_mask}

- 51

- 52

- 53 # Define the model

- 54 class SentimentClassifier(nn.Module):

- 55 def __init__(self , bert_model , num_classes =5):

- 56 super(SentimentClassifier , self).__init__()

- 57 self.bert = bert_model

- 58 self.classifier = nn.Linear(bert_model.config. hidden_size , num_classes)

- 59

- 60 def forward(self , input_ids , attention_mask):

- 61 outputs = self.bert(input_ids=input_ids , attention_mask=attention_mask)

- 62 pooled_output = outputs.pooler_output

- 63 logits = self.classifier(pooled_output)

- 64 return logits

- 65

- 66

- 67 # Load the data

- 68 train_df = pd.read_csv(

- 69 "/PATH/TO/RAW/DATA/sentiment -analysis -on-movie -reviews/ train.tsv",

- 70 sep="\t",

- 71 )

- 72 test_df = pd.read_csv(

- 73 "/PATH/TO/RAW/DATA/sentiment -analysis -on-movie -reviews/ test.tsv",

- 74 sep="\t",

- 75 )

- 76

- 77 # Split the data into training and validation sets

- 78 train_df , val_df = train_test_split(train_df , test_size

=0.2, random_state =42)

- 79

- 80 # Tokenizer and model

- 81 tokenizer = BertTokenizer.from_pretrained("bert -base uncased")

- 82 bert_model = BertModel.from_pretrained("bert -base -uncased")

- 83

- 84 # Create datasets and dataloaders

- 85 train_dataset = SentimentDataset(train_df , tokenizer)

- 86 val_dataset = SentimentDataset(val_df , tokenizer)

- 87 test_dataset = SentimentDataset(test_df , tokenizer)

- 88

- 89 train_loader = DataLoader(train_dataset , batch_size=16, shuffle=True)

- 90 val_loader = DataLoader(val_dataset , batch_size=16, shuffle

=False)

- 91 test_loader = DataLoader(test_dataset , batch_size=16, shuffle=False)

- 92

- 93 # Model , loss , and optimizer

- 94 model = SentimentClassifier(bert_model)

- 95 model = model.to("cuda")

- 96

- 97 criterion = nn.CrossEntropyLoss(

- 98 weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to("cuda ")

- 99 )

- 100 optimizer = optim.Adam(model.parameters(), lr=2e-5)

- 101

- 102 # Training loop

- 103 num_epochs = 3

- 104 for epoch in range(num_epochs):

- 105 model.train()

- 106 for batch in train_loader:

- 107 input_ids = batch["input_ids"].to("cuda")

- 108 attention_mask = batch["attention_mask"].to("cuda")

- 109 labels = batch["label"].to("cuda")

- 110

- 111 optimizer.zero_grad()

- 112 outputs = model(input_ids , attention_mask)

- 113 loss = criterion(outputs , labels)

- 114 loss.backward()

- 115 optimizer.step()

- 116

- 117 # Validation

- 118 model.eval()

- 119 val_predictions = []

- 120 val_labels = []

- 121 with torch.no_grad():

- 122 for batch in val_loader:

- 123 input_ids = batch["input_ids"].to("cuda")

- 124 attention_mask = batch["attention_mask"].to(" cuda")

- 125 labels = batch["label"].to("cuda")

- 126

- 127 outputs = model(input_ids , attention_mask)

- 128 _, preds = torch.max(outputs , dim=1)

- 129

- 130 val_predictions.extend(preds.cpu().numpy())

- 131 val_labels.extend(labels.cpu().numpy())

- 132

- 133 val_accuracy = accuracy_score(val_labels , val_predictions)

- 134 print(f"Epoch {epoch + 1}/{ num_epochs}, Validation Accuracy: {val_accuracy :.4f}")

- 135

- 136 # Generate predictions for the test set

- 137 model.eval()

- 138 test_predictions = []

- 139 with torch.no_grad():

- 140 for batch in test_loader:

- 141 input_ids = batch["input_ids"].to("cuda")

- 142 attention_mask = batch["attention_mask"].to("cuda")

- 143

- 144 outputs = model(input_ids , attention_mask)

- 145 _, preds = torch.max(outputs , dim=1)

- 146

- 147 test_predictions.extend(preds.cpu().numpy())

- 148

- 149 # Save predictions to submission.csv

- 150 submission = pd.DataFrame(

- 151 {"PhraseId": test_df["PhraseId"], "Sentiment": test_predictions}

- 152 )

- 153 submission.to_csv("./submission/submission.csv", index= False)

The example below is taken from a computer vision competition 5-flowers-image-classification.The solution was generated by Agent K as one of the draft nodes. Agent K is provided with CoT from the scaffolding stage to guide its initial solution generation.

###### Prompt

|... <Skipped the common instructions for prompt display purposes<br><br>in paper appendix , only changes are added below ><br><br>- <TOTAL_TIME_REMAINING: 44.0hrs 28.0mins 57secs ><br><br>- <TOTAL_STEPS_REMAINING: 4996><br><br><br>...|
|---|

###### ## Data Overview

/PATH/TO/RAW/DATA |

+-Sample_submission.csv (401 lines) |

+-test/ | +-1.jpg (31.8 kB) | +-10.jpg (4.5 kB) | +-100.jpg (16.9 kB) | +-... and 397 other files |

+-train/ | +daisy/ | | + -100080576 _f52e8ee070_n.jpg (26.8 kB) | | + -10140303196 _b88d3d6cec.jpg (117.2 kB) | | + -10172379554 _b296050f82_n.jpg (36.4 kB) | | +-... and 688 other files | +dandelion/ | | + -10043234166 _e6dd915111_n.jpg (26.7 kB) | | + -10200780773 _c6051a7d71_n.jpg (23.3 kB) | | + -10294487385 _92a0676c7d_m.jpg (21.7 kB) | | +-... and 948 other files | +rose/ | | + -10090824183 _d02c613f10_m.jpg (20.2 kB) | | + -102501987 _3cdb8e5394_n.jpg (29.8 kB) | | + -10503217854 _e66a804309.jpg (208.6 kB) | | +-... and 691 other files | +sunflower/ | | + -1008566138 _6927679c8a.jpg (143.2 kB) | | + -1022552002 _2b93faf9e7_n.jpg (42.7 kB) | | + -1022552036 _67d33d5bd8_n.jpg (60.8 kB) | | +-... and 656 other files | +tulip/

| | + -100930342 _92e8746431_n.jpg (26.2 kB) | | + -10094729603 _eeca3f2cb6.jpg (214.8 kB) | | + -10094731133 _94a942463c.jpg (225.0 kB) | | +-... and 867 other files

## Task description Sample_submission.csv has 400 rows and 2 columns. The columns are: id, label Dataset Description Image classification: 5 kinds of flowers There are pictures with 5 kinds of flowers. labels: daisy/dandelion/rose/sunflower/tulip -> 0/1/2/3/4 The labels in submission file should be integer (0/1/2/3/4). train data: 5 folders of train data test data: 1 folder of test data Evaluation The evaluation metric for this competition is Categorization Accuracy. Labels in your submission file should be integer (or string) type, i.e., 0/1/2/3/4. rgin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns ## Summary of Past Submissions

- ### Submission 0 summary and metric value on a validation set: The project includes a comprehensive pipeline for processing and extracting features from images using a pre-trained EfficientNetV2-S model. Custom transformation pipelines are defined for both training and testing images, incorporating various augmentations and normalization techniques. The ImageEmbedder model is designed to generate latent representations of images, with the option to fine-tune the last few layers. Additionally, the project addresses class imbalances and implements a TabularHead model for handling both classification and regression tasks, complete with appropriate loss functions. "submission.csv": 0.9627714581178903

—–

- ### Submission 1 summary and metric value on a validation set: The project involves creating a comprehensive pipeline for image processing and feature extraction using EfficientNetB0. For training images, a custom transformation pipeline is defined, including resizing, center cropping, random horizontal flips, random rotations, color jitter, conversion to tensor, and normalization. For test images, a simpler transformation pipeline is used, involving resizing, center cropping, conversion to tensor, and normalization.

An ImageEmbedder class is implemented to extract latent representations from the transformed images, with the option to unfreeze the last few layers for fine-tuning. Additionally, class imbalances are handled, and a TabularHead model is implemented to decode the image embeddings and predict both regression and classification targets, along with corresponding loss functions. "submission.csv": 0.9524301964839711

—–

- ### Submission 2 summary and metric value on a validation set: The project involves creating an end-to-end image processing pipeline for a multi-task learning problem. It includes defining custom image transformation pipelines for both training and testing, using a pre-trained ResNet50 model to extract image embeddings, handling class imbalances, and implementing a custom PyTorch model (TabularHead) to process these embeddings and produce both classification and regression outputs. Loss functions for both tasks are defined to optimize the model’s performance. "submission.csv": 0.9503619441571872

###### Assistant

### To tackle this image classification task, I will use a pre-trained ResNet50 model from ‘torchvision.models‘ to extract features from the images. The model will be fine-tuned on the training data, and predictions will be made on the test set. The evaluation metric will be the categorization accuracy on a hold-out validation set.

- 1 import os

- 2 import pandas as pd

- 3 import numpy as np

- 4 import torch

- 5 import torch.nn as nn

- 6 import torch.optim as optim

- 7 from torchvision import models , transforms

- 8 from torch.utils.data import DataLoader , Dataset

- 9 from PIL import Image

- 10 from sklearn.model_selection import train_test_split

- 11 from sklearn.metrics import accuracy_score

- 12

- 13 # Define the dataset

- 14 class FlowerDataset(Dataset):

- 15 def __init__(self , root_dir , transform=None , mode=’ train’):

- 16 self.root_dir = root_dir

- 17 self.transform = transform

- 18 self.mode = mode

- 19 self.image_paths = []

- 20 self.labels = []

- 21

- 22 if mode == ’train’:

- 23 for label , flower in enumerate([’daisy’, ’ dandelion’, ’rose’, ’sunflower’, ’tulip’]):

- 24 folder_path = os.path.join(root_dir , flower )

- 25 for image_name in os.listdir(folder_path):

- 26 self.image_paths.append(os.path.join( folder_path , image_name))

- 27 self.labels.append(label)

- 28 else:

- 29 for image_name in os.listdir(root_dir):

- 30 self.image_paths.append(os.path.join( root_dir , image_name))

- 31

- 32 def __len__(self):

- 33 return len(self.image_paths)

- 34

- 35 def __getitem__(self , idx):

- 36 image_path = self.image_paths[idx]

- 37 image = Image.open(image_path).convert(’RGB’)

- 38 if self.transform:

- 39 image = self.transform(image)

- 40

- 41 if self.mode == ’train’:

- 42 label = self.labels[idx]

- 43 return image , label

- 44 else:

- 45 return image , image_path

- 46

- 47 # Define transformations

- 48 train_transform = transforms.Compose([

- 49 transforms.Resize((224, 224)),

- 50 transforms.RandomHorizontalFlip(),

- 51 transforms.ToTensor(),

- 52 transforms.Normalize(mean=[0.485, 0.456, 0.406], std

=[0.229, 0.224, 0.225])

- 53 ])

- 54

- 55 test_transform = transforms.Compose([

- 56 transforms.Resize((224, 224)),

- 57 transforms.ToTensor(),

- 58 transforms.Normalize(mean=[0.485, 0.456, 0.406], std

=[0.229, 0.224, 0.225])

- 59 ])

- 60

- 61 # Load datasets

- 62 train_dataset = FlowerDataset(’/PATH/TO/RAW/DATA/5-flowers image -classification/train’, transform=train_transform , mode=’train’)

- 63 train_data , val_data , train_labels , val_labels = train_test_split(train_dataset , train_dataset.labels , test_size=0.2, stratify=train_dataset.labels , random_state =42)

- 64

- 65 train_loader = DataLoader(train_data , batch_size=32, shuffle=True)

- 66 val_loader = DataLoader(val_data , batch_size=32, shuffle= False)

- 67

- 68 # Define the model

- 69 model = models.resnet50(pretrained=True)

- 70 num_features = model.fc.in_features

- 71 model.fc = nn.Linear(num_features , 5)

- 72 model = model.to(’cuda’)

- 73

- 74 # Define loss function and optimizer

- 75 criterion = nn.CrossEntropyLoss()

- 76 optimizer = optim.Adam(model.parameters(), lr=0.001)

- 77

- 78 # Training loop

- 79 num_epochs = 10

- 80 for epoch in range(num_epochs):

- 81 model.train()

- 82 running_loss = 0.0

- 83 for inputs , labels in train_loader:

- 84 inputs , labels = inputs.to(’cuda’), labels.to(’cuda ’)

- 85 optimizer.zero_grad()

- 86 outputs = model(inputs)

- 87 loss = criterion(outputs , labels)

- 88 loss.backward()

- 89 optimizer.step()

- 90 running_loss += loss.item() * inputs.size(0)

- 91

- 92 # Validation

- 93 model.eval()

- 94 val_loss = 0.0

- 95 correct = 0

- 96 total = 0

- 97 with torch.no_grad():

- 98 for inputs , labels in val_loader:

- 99 inputs , labels = inputs.to(’cuda’), labels.to(’

cuda’)

- 100 outputs = model(inputs)

- 101 loss = criterion(outputs , labels)

- 102 val_loss += loss.item() * inputs.size(0)

- 103 _, predicted = torch.max(outputs , 1)

- 104 total += labels.size(0)

- 105 correct += (predicted == labels).sum().item()

- 106

- 107 val_accuracy = correct / total

- 108 print(f’Epoch {epoch +1}/{ num_epochs}, Train Loss: { running_loss/len(train_data):.4f}, Val Loss: { val_loss/len(val_data):.4f}, Val Accuracy: { val_accuracy :.4f}’)

- 109

- 110 # Test set predictions

- 111 test_dataset = FlowerDataset(’/PATH/TO/RAW/DATA/5-flowers image -classification/test’, transform=test_transform , mode=’test’)

- 112 test_loader = DataLoader(test_dataset , batch_size=32, shuffle=False)

- 113

- 114 model.eval()

- 115 predictions = []

- 116 image_ids = []

- 117 with torch.no_grad():

- 118 for inputs , image_paths in test_loader:

- 119 inputs = inputs.to(’cuda’)

- 120 outputs = model(inputs)

- 121 _, predicted = torch.max(outputs , 1)

- 122 predictions.extend(predicted.cpu().numpy())

- 123 image_ids.extend([os.path.basename(path) for path in image_paths])

- 124

- 125 # Save predictions to submission.csv

- 126 submission = pd.DataFrame({’id’: image_ids , ’label’: predictions})

- 127 submission[’id’] = submission[’id’].str.replace(’.jpg’, ’’)

- 128 submission.to_csv(’./submission/submission.csv’, index= False)

###### Theexamplebelowistakenfrom atabularcompetitionbike-sharing-demand.Thesolution was generated by Agent K as one of the draft nodes. Agent K is provided with CoT from the scaffolding stage to guide its initial solution generation.

###### Prompt

|... <Skipped the common instructions for prompt display purposes<br><br>in paper appendix , only changes are added below ><br><br><**Total Runtime **: You have a maximum of 86400 seconds to attempt this task. You may iteratively develop your solution within this time frame , but bear in mind that program runtime also counts towards this limit.<br><br>**Total Steps**: You have a maximum of 5000 steps to submit your solution.<br><br>- <TOTAL_TIME_REMAINING: 22.0hrs 39.0mins 51secs ><br><br>- <TOTAL_STEPS_REMAINING: 4982><br><br><br>...|
|---|

## Data Overview

|/PATH/TO/RAW/DATA |<br><br>+-sampleSubmission.csv (6494 lines) |<br><br>+-.. and 2 other files|
|---|

## Task description Dataset Description See, fork, and run a random forest benchmark model through Kaggle Scripts You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period. Data Fields datetime - hourly date + timestamp season - 1 = spring, 2 = summer, 3 = fall, 4 = winter holiday - whether the day is considered a holiday workingday - whether the day is neither a weekend nor holiday weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy

- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog temp - temperature in Celsius atemp - "feels like" temperature in Celsius humidity - relative humidity windspeed - wind speed casual - number of non-registered user rentals initiated

registered - number of registered user rentals initiated count - number of total rentals

sampleSubmission.csv has 6493 rows and 2 columns. The columns are: datetime, count

test.csv has 6493 rows and 9 columns. The columns are: datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed

train.csv has 10886 rows and 12 columns. The columns are: datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count

## Summary of Past Submissions We test multiple ensemble models and combined the best-performing ones to make final predictions. Each model has a different contribution based on its performance. Below are some models and their contributions in the final prediction, along with their hyperparameters. Use this information to come up with a solution for the competition. There may be better models and sets of hyperparameter that can be used so you are free to explore and come up with a better solution ### catboost

|Learning Rate|Max Depth<br><br>|Estimators|Bootstrap Type<br><br>|Borders|Grow Policy<br><br>|L2 Leaf Reg<br><br>|Min Data Leaf|Random Strength|Contrib. (%)|
|---|---|---|---|---|---|---|---|---|---|
|0.05 0.05 0.15 0.05 0.05 0.05 0.25 0.1 0.05 0.25<br><br>|12 16<br>13 16 16 16 16<br><br><br>15<br>16 15<br>|100 300 70 300 300 400 40 200 500 40<br><br>|MVS MVS MVS MVS MVS MVS<br><br>Bayesian_0<br>Bayesian_1 Bayesian_0 Bayesian_5<br>|64 510 32 510 1000 510 32 250 1000 32<br><br>|Depthwise SymmetricTree Depthwise SymmetricTree SymmetricTree SymmetricTree Depthwise Depthwise SymmetricTree Depthwise<br><br>|0.3 0.3 0.1 0.2 0.3 0.3 5 0.4 0.5 2<br><br>|5 10 5 1 20 10 5 10 100 5|1 10 1 100 100 100 0 50 100 0<br><br>|22 12 11 8.9 8.2 7.8 6.6 6.4 3.2 3.1<br><br>|

###### Assistant

### I propose using a CatBoost model, which is known for its effectiveness in handling categorical features and reducing overfitting. CatBoost can capture complex patterns in the data, potentially leading to better performance. The evaluation metric will still be the Root Mean Squared Logarithmic Error (RMSLE).

- 1 import pandas as pd

- 2 import numpy as np

- 3 from catboost import CatBoostRegressor

- 4 from sklearn.model_selection import train_test_split

- 5 from sklearn.metrics import mean_squared_log_error

- 6

- 7 # Load the data

- 8 train_df = pd.read_csv(’/PATH/TO/RAW/DATA/bike -sharing demand/train.csv’)

- 9 test_df = pd.read_csv(’/PATH/TO/RAW/DATA/bike -sharing demand/test.csv’)

- 10

- 11 # Preprocess the data

- 12 train_df[’datetime’] = pd.to_datetime(train_df[’datetime’])

- 13 test_df[’datetime’] = pd.to_datetime(test_df[’datetime’])

- 14

- 15 # Extract features and target

- 16 X_train = train_df.drop([’datetime’, ’casual’, ’registered’ , ’count’], axis=1)

- 17 y_train = np.log1p(train_df[’count’])

- 18

- 19 X_test = test_df.drop([’datetime’], axis=1)

- 20

- 21 # Split the data into training and validation sets

- 22 X_train , X_val , y_train , y_val = train_test_split(X_train , y_train , test_size=0.2, random_state =42)

- 23

- 24 # Initialize the CatBoost model

- 25 model = CatBoostRegressor(

- 26 learning_rate =0.05,

- 27 max_depth=16,

- 28 n_estimators =300,

- 29 bootstrap_type=’MVS’,

- 30 border_count =510,

- 31 grow_policy=’SymmetricTree’,

- 32 l2_leaf_reg =0.3,

- 33 min_data_in_leaf=10,

- 34 random_strength =100,

- 35 silent=True

- 36 )

- 37

- 38 # Train the model

- 39 model.fit(X_train , y_train)

- 40

- 41 # Make predictions on the validation set

- 42 y_pred_val = model.predict(X_val)

- 43

- 44 # Calculate the RMSLE on the validation set

- 45 rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np. expm1(y_pred_val)))

- 46 print(f’Validation RMSLE: {rmsle}’)

- 47

- 48 # Make predictions on the test set

- 49 y_pred_test = model.predict(X_test)

- 50

- 51 # Prepare the submission file

- 52 submission = pd.DataFrame({

- 53 ’datetime’: test_df[’datetime’],

- 54 ’count’: np.expm1(y_pred_test)

- 55 })

- 56

- 57 # Save the submission file

- 58 submission.to_csv(’./submission/submission.csv’, index= False)

#### H.5 AIDE with RAG

The example below is taken from the solution generation process of one of the nodes in the invasive-species-monitoring competition. In this case, AIDE is provided with one of the most similar examples from the RAG database to serve as a guide to its solution generation.

###### Prompt

|... <Skipped the common instructions(same as \OurAgent{} CoT<br><br>cases) for prompt display purposes in paper appendix , only changes are added below ><br><br>- <TOTAL_TIME_REMAINING: 94.0hrs 33.0mins 7secs ><br><br>- <TOTAL_STEPS_REMAINING: 4998><br><br><br>...|
|---|

##Hereisafewpastexperiencecaseswrittenbyanhumanexpertforarelevant (but not the same) task

The overall design of the code is centered around a machine learning pipeline for the Kaggle Cassava Leaf Disease Classification competition. The pipeline includes data loading, transformation, model creation, and inference. It utilizes several pre-trained models for image classification, which are fine-tuned using cross-validation. The main script is designed to handle multiple models, different transformations, and test-time augmentation (TTA) to improve the robustness and accuracy of the predictions.

The model architecture is composed of multiple pre-trained models from the PyTorch Image Models (timm) library and Facebook Research’s DeiT. The models used include ‘tf_efficientnet_b4_ns‘, ‘vit_base_patch16_384‘, and ‘seresnext50_32x4d‘. Each model is modified to fit the specific classification task by replacing thefinal fully connected layer to outputthe numberof classes (5 for the Cassava leaf disease dataset). Here is a detailed breakdown of each model architecture:

- 1. **tf_efficientnet_b4_ns**: EfficientNet-B4 variant that is optimized for performance and efficiency. The model’s classifier layer is replaced with a new linear layer of size 5.
- 2. **vit_base_patch16_384**: A Vision Transformer (ViT) base model with patch size 16 and input image size 384x384. The head layer is replaced with a linear layer of size 5.
- 3. **seresnext50_32x4d**: A variant of the ResNeXt50 model with SE (Squeezeand-Excitation) blocks. The final fully connected layer is replaced with a linear layer of size 5.

All models are loaded with or without pretraining and are adapted to the task by adjusting the output layer to match the number of classes in the dataset.

The important hyper-parameters in this code are set within the ‘CFG‘ class. Key parameters include:

- - ‘models‘: A list of model names to be used in the ensemble.
- - ‘size‘: Dictionary mapping each model to its input image size.
- - ‘batch_size‘: Number of samples per batch during training and inference.
- - ‘target_size‘: Number of classes in the dataset (5).
- - ‘n_fold‘: Number of folds for cross-validation.
- - ‘trn_fold‘: A dictionary specifying the best and final folds for each model that should be used during training and inference.
- - ‘data_parallel‘: Boolean dictionary indicating whether to use data parallelism for each model.
- - ‘transform‘: Transformation types applied to the data during inference.
- - ‘weight‘: Weights assigned to each model during the final prediction aggregation.
- - ‘tta‘: Number of test-time augmentations to perform.
- - ‘no_tta_weight‘: Weight assigned to the non-augmented inference results.

The optimization objective is not explicitly defined in the provided code snippet, as the code focuses on the inference process. However, based on typical practices in such competitions, the model is likely trained to minimize a loss function, such as cross-entropy loss, using the Adam optimizer or another suitable optimizer. The model is evaluated using metrics like accuracy, and the goal is to maximize the accuracy score on the validation set during training.

The advanced machine learning techniques used in this code include:

- 1. **Ensemble Learning**: Multiple pre-trained models are combined to form an ensemble, which helps in improving the generalization and robustness of the predictions.
- 2. **Cross-Validation**: The dataset is split into multiple folds, and the models are trained and validated using these folds to ensure better performance and avoid overfitting.

- 3. **Data Augmentation**: Various data augmentation techniques are applied during inference to improve the model’s ability to handle variations in the input data.
- 4. **Test-Time Augmentation (TTA)**: Multiple predictions are generated for each input image using different transformations, and the results are averaged to produce the final prediction.

Other important tricks that contribute to the high performance of this code include:

- -**ModelWeighting**: Eachmodelintheensembleisassignedaweight, which is used to aggregate the predictions. This helps in leveraging the strengths of different models.
- - **Careful Selection of Pre-trained Models**: The choice of pre-trained models (e.g., EfficientNet, ViT, and ResNeXt) ensures that the ensemble covers a diverse range of architectures, which can capture different aspects of the data.
- - **Custom Transformations**: The use of custom transformations tailored to each model helps in optimizing the performance of each individual model.
- - **Logging and Monitoring**: The use of logging allows for detailed tracking of the training and inference processes, which is crucial for debugging and performance optimization.

## Data Overview Your solution can use any relevant machine learning packages such as: ‘torchvision‘, ‘xgboost‘, ‘timm‘, ‘pandas‘, ‘torch-geometric‘, ‘scikit-learn‘, ‘statsmodels‘, ‘numpy‘, ‘torch‘, ‘bayesian-optimization‘, ‘lightGBM‘. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow.

/PATH/TO/RAW/DATA |

+-sample_submission.csv (1532 lines) |

+-test/ | +-1.jpg (31.8 kB) | +-10.jpg (4.5 kB) | +-100.jpg (16.9 kB) | +-... and 397 other files |

+-test/ | +-1.jpg (648.4 kB) | +-10.jpg (595.3 kB) | +-100.jpg (637.2 kB) | +-... and 1528 other files

+-train/ | +-1.jpg (974.4 kB)

| +-10.jpg (1.1 MB) | +-100.jpg (1.1 MB) | +-... and 2291 other files

## Task description Dataset Description The data set contains pictures taken in a Brazilian national forest. In some of the pictures there is Hydrangea, a beautiful invasive species original of Asia. Based on the training pictures and the labels provided, the participant should predict the presence of the invasive species in the testing set of pictures. File descriptions train.7z - the training set (contains 2295 images). train_labels.csv - the correct labels for the training set. test.7z - the testing set (contains 1531 images), ready to be labeled by your algorithm. sample_submission.csv - a sample submission file in the correct format. Data fields name - name of the sample picture file (numbers) invasive - probability of the picture containing an invasive species. A probability of 1 means the species is present. Evaluation SubmissionsareevaluatedonareaundertheROCcurvebetweenthepredicted probability and the observed target. Submission File For each image in the test set, you must predict a probability for the target variable on whether the image contains invasive species or not. The file should contain a header and have the following format: name,invasive 2,0.5

- 5,0
- 6,0.2 etc.

###### Assistant

### The solution will involve building a machine learning pipeline for image classification using a combination of pre-trained models. We will use a blend of models from the PyTorch Image Models (timm) library, including ‘tf_efficientnet_b4_ns‘ and ‘vit_base_patch16_384‘. The models will be finetuned on the training data using cross-validation, and test-time augmentation (TTA) will be applied to improve the robustness of the predictions. The final predictions will be aggregated using weighted averaging, and the results will be saved in the required submission format.

- 1 import os

- 2 import pandas as pd

- 3 import numpy as np

- 4 import torch

- 5 from torch.utils.data import DataLoader , Dataset

- 6 import torchvision.transforms as transforms

- 7 from timm import create_model

- 8 from sklearn.model_selection import StratifiedKFold

- 9 from sklearn.metrics import roc_auc_score

- 10 from PIL import Image

- 11

- 12 # Configuration

- 13 class CFG:

- 14 models = [’tf_efficientnet_b4_ns ’, ’ vit_base_patch16_384 ’]

- 15 size = {’tf_efficientnet_b4_ns ’: 384, ’ vit_base_patch16_384 ’: 384}

- 16 batch_size = 16

- 17 target_size = 1

- 18 n_fold = 5

- 19 trn_fold = {model: [0, 1, 2, 3, 4] for model in models}

- 20 transform = {

- 21 ’train’: transforms.Compose([

- 22 transforms.Resize((384, 384)),

- 23 transforms.RandomHorizontalFlip(),

- 24 transforms.ToTensor(),

- 25 transforms.Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

- 26 ]),

- 27 ’test’: transforms.Compose([

- 28 transforms.Resize((384, 384)),

- 29 transforms.ToTensor(),

- 30 transforms.Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

- 31 ])

- 32 }

- 33 tta = 5

- 34 no_tta_weight = 0.5

- 35

- 36 # Dataset

- 37 class InvasiveDataset(Dataset):

- 38 def __init__(self , df, root_dir , transform=None , mode=’ train’):

- 39 self.df = df

- 40 self.root_dir = root_dir

- 41 self.transform = transform

- 42 self.mode = mode

- 43

- 44 def __len__(self):

- 45 return len(self.df)

- 46

- 47 def __getitem__(self , idx):

- 48 img_name = os.path.join(self.root_dir , f"{self.df[’ name ’].iloc[idx]}.jpg")

- 49 image = Image.open(img_name).convert(’RGB’)

- 50 if self.transform:

- 51 image = self.transform(image)

- 52 if self.mode == ’train’:

- 53 label = self.df[’invasive’].iloc[idx]

- 54 return image , label

- 55 else:

- 56 return image

- 57

- 58 # Model

- 59 def get_model(model_name , pretrained=True):

- 60 model = create_model(model_name , pretrained=pretrained , num_classes =1)

- 61 return model

- 62

- 63 # Training and Evaluation

- 64 def train_model(model , dataloader , criterion , optimizer , device):

- 65 model.train()

- 66 running_loss = 0.0

- 67 for inputs , labels in dataloader:

- 68 inputs , labels = inputs.to(device), labels.to( device).float().unsqueeze (1)

- 69 optimizer.zero_grad()

- 70 outputs = model(inputs)

- 71 loss = criterion(outputs , labels)

- 72 loss.backward()

- 73 optimizer.step()

- 74 running_loss += loss.item() * inputs.size(0)

- 75 return running_loss / len(dataloader.dataset)

- 76

- 77 def evaluate_model(model , dataloader , device):

- 78 model.eval()

- 79 all_preds = []

- 80 all_labels = []

- 81 with torch.no_grad():

- 82 for inputs , labels in dataloader:

- 83 inputs , labels = inputs.to(device), labels.to( device).float().unsqueeze (1)

- 84 outputs = model(inputs)

- 85 preds = torch.sigmoid(outputs).cpu().numpy()

- 86 all_preds.extend(preds)

- 87 all_labels.extend(labels.cpu().numpy())

- 88 return np.array(all_preds), np.array(all_labels)

- 89

- 90 # Main

- 91 def main():

- 92 train_df = pd.read_csv(’/PATH/TO/RAW/DATA/invasive species -monitoring/train_labels.csv’)

- 93 test_df = pd.read_csv(’/PATH/TO/RAW/DATA/invasive species -monitoring/sample_submission.csv’)

- 94 skf = StratifiedKFold(n_splits=CFG.n_fold , shuffle=True , random_state =42)

- 95

- 96 for model_name in CFG.models:

- 97 model = get_model(model_name)

- 98 model.to(’cuda’)

- 99 criterion = torch.nn.BCEWithLogitsLoss()

- 100 optimizer = torch.optim.Adam(model.parameters(), lr

=1e-4)

- 101

- 102 for fold , (train_idx , val_idx) in enumerate(skf. split(train_df , train_df[’invasive’])):

- 103 if fold not in CFG.trn_fold[model_name]:

- 104 continue

- 105

- 106 train_data = train_df.iloc[train_idx]

- 107 val_data = train_df.iloc[val_idx]

- 108

- 109 train_dataset = InvasiveDataset(train_data , ’/ PATH/TO/RAW/DATA/invasive -species -monitoring /train’, CFG.transform[’train’], ’train’)

- 110 val_dataset = InvasiveDataset(val_data , ’/PATH/ TO/RAW/DATA/invasive -species -monitoring/ train’, CFG.transform[’test’], ’train’)

- 111

- 112 train_loader = DataLoader(train_dataset , batch_size=CFG.batch_size , shuffle=True , num_workers =4)

- 113 val_loader = DataLoader(val_dataset , batch_size

=CFG.batch_size , shuffle=False , num_workers

=4)

- 114

- 115 for epoch in range(5): # 5 epochs for simplicity

- 116 train_loss = train_model(model , train_loader , criterion , optimizer , ’ cuda’)

- 117 val_preds , val_labels = evaluate_model( model , val_loader , ’cuda’)

- 118 val_auc = roc_auc_score(val_labels , val_preds)

- 119 print(f"Fold {fold}, Epoch {epoch}, Train Loss: {train_loss :.4f}, Val AUC: { val_auc :.4f}")

- 120

- 121 test_dataset = InvasiveDataset(test_df , ’/PATH/TO/ RAW/DATA/invasive -species -monitoring/test’, CFG. transform[’test’], ’test’)

- 122 test_loader = DataLoader(test_dataset , batch_size= CFG.batch_size , shuffle=False , num_workers =4)

- 123

- 124 test_preds = []

- 125 for _ in range(CFG.tta):

- 126 preds = evaluate_model(model , test_loader , ’ cuda’)[0]

- 127 test_preds.append(preds)

- 128 test_preds = np.mean(test_preds , axis=0)

- 129

- 130 test_df[’invasive’] = test_preds

- 131 test_df[[’name’, ’invasive’]].to_csv(f’./submission /{model_name}_submission.csv’, index=False)

- 132

- 133 if __name__ == "__main__":

- 134 main()

## References

- [1] OpenAI Josh Achiam et al. “GPT-4 Technical Report”. In: 2023.
- [2] Takuya Akiba et al. “Optuna: A Next-Generation Hyperparameter Optimization Framework”. In: The 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019, pp. 2623–2631.
- [3] Simran Arora et al. “Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes”. In: Proc. VLDB Endow. 17.2 (Oct. 2023), pp. 92–105. DOI: 10.14778/3626292.3626294.
- [4] Maximilian Balandat et al. “BOTORCH: a framework for efficient monte-carlo Bayesian optimization”. In: Proceedings of the 34th International Conference on Neural Information Processing Systems. NIPS ’20. Vancouver, BC, Canada: Curran Associates Inc., 2020.

- [5] Ruisheng Cao et al. “Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows?” In: CoRR abs/2407.10956 (2024). arXiv: 2407.10956.
- [6] Jun Shern Chan et al. “MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering”. In: The Thirteenth International Conference on Learning Representations. 2025.
- [7] ZuiChenetal.“SEED:Domain-SpecificDataCurationWithLargeLanguageModels”. In: Arxiv abs/2310.00749 (2023).
- [8] Yizhou Chi et al. SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning. 2024.
- [9] AlexanderCowen-Riversetal.“HEBO:PushingTheLimitsofSample-EfficientHyperparameter Optimisation”. In: Journal of Artificial Intelligence Research 74 (July 2022).
- [10] Yinlin Deng et al. “Large Language Models Are Zero-Shot Fuzzers: Fuzzing DeepLearning Libraries via Large Language Models”. In: Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis (2022).
- [11] Kamil Dreczkowski, Antoine Grosnit, and Haitham Bou Ammar. “Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization”. In: Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2023.
- [12] Haoyang Fang et al. “MLZero: A Multi-Agent System for End-to-end Machine Learning Automation”. In: ArXiv abs/2505.13941 (2025).
- [13] Matthias Feurer et al. “Auto-sklearn 2.0: hands-free AutoML via meta-learning”. In: J. Mach. Learn. Res. 23.1 (Jan. 2022).
- [14] P.J.A.Gijsbersetal.“AnOpenSourceAutoMLBenchmark”.In:ArXiv abs/1907.00909

(2019).

- [15] Siyuan Guo et al. “DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning”. In: Proceedings of the 41st International Conference on Machine Learning. Vol. 235. Proceedings of Machine Learning Research. PMLR, 2024, pp. 16813–16848.
- [16] Taicheng Guo et al. “Large Language Model based Multi-Agents: A Survey of Progress and Challenges”. In: ArXiv abs/2402.01680 (2024).
- [17] EladHazan.“IntroductiontoOnlineConvexOptimization”.In:Found.TrendsOptim. 2 (2016), pp. 157–325.
- [18] NoahHollmann,SamuelMüller,andFrankHutter.“LargeLanguageModelsforAutomated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering”. In: Thirty-seventh Conference on Neural Information Processing Systems. 2023.
- [19] Sirui Hong et al. “Data Interpreter: An LLM Agent For Data Science”. In: ArXiv abs/2402.18679 (2024).

- [20] Sirui Hong et al. “MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework”. In: The Twelfth International Conference on Learning Representations. 2024.
- [21] T. Hospedales et al. “Meta-Learning in Neural Networks: A Survey”. In: IEEE Transactions on Pattern Analysis & Machine Intelligence 44.09 (Sept. 2022), pp. 5149–5169. DOI: 10.1109/TPAMI.2021.3079209.
- [22] Xueyu Hu et al. “InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks”. In: ArXiv abs/2401.05507 (2024).
- [23] Jie Huang and Kevin Chen-Chuan Chang. “Towards Reasoning in Large Language Models: A Survey”. In: ArXiv abs/2212.10403 (2022).
- [24] Qian Huang et al. “MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation”. In: ArXiv abs/2310.03302 (2023).
- [25] Albert Qiaochu Jiang et al. “Mistral 7B”. In: ArXiv abs/2310.06825 (2023).
- [26] Carlos E Jimenez et al. “SWE-bench: Can Language Models Resolve Real-world Github Issues?” In: The Twelfth International Conference on Learning Representations. 2024.
- [27] Haifeng Jin et al. “AutoKeras: An AutoML Library for Deep Learning”. In: Journal of Machine Learning Research 24.6 (2023), pp. 1–6.
- [28] LiqiangJingetal.DSBench:HowFarAreDataScienceAgentstoBecomingDataScience Experts? 2024.
- [29] JamesMaxKanterandKalyanVeeramachaneni.“Deepfeaturesynthesis:Towardsautomating data science endeavors”. In: 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (2015), pp. 1–10.
- [30] AmbikaKaul,SaketMaheshwary,andVikramPudi.“AutoLearn—AutomatedFeature Generation and Selection”. In: 2017 IEEE International Conference on Data Mining (ICDM) (2017), pp. 217–226.
- [31] Udayan Khurana et al. “Cognito: Automated Feature Engineering for Supervised Learning”. In: 2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW). 2016, pp. 1304–1307. DOI: 10.1109/ICDMW.2016.0190.
- [32] Yuhang Lai et al. “DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation”. In: ArXiv abs/2211.11501 (2022).
- [33] Erin LeDell and S. Poirier. “H2O AutoML: Scalable Automatic Machine Learning”. In: 2020.
- [34] Shuang Li et al. “Pre-Trained Language Models for Interactive Decision-Making”. In: AdvancesinNeuralInformationProcessingSystems.Ed.byS.Koyejoetal.Vol.35.Curran Associates, Inc., 2022, pp. 31199–31212.
- [35] ZimingLietal.“AutoKaggle:AMulti-AgentFrameworkforAutonomousDataScience Competitions”. In: 2024.
- [36] HanxiaoLiu,KarenSimonyan,andYimingYang.“DARTS:DifferentiableArchitecture Search”. In: International Conference on Learning Representations. 2019.

- [37] DaqinLuoetal.“AutoM3L:AnAutomatedMultimodalMachineLearningFramework with Large Language Models”. In: ACM Multimedia 2024. 2024.
- [38] Zhao Mandi, Shreeya Jain, and Shuran Song. RoCo: Dialectic Multi-Robot Collaboration with Large Language Models. 2023. arXiv: 2307.04738 [cs.RO].
- [39] Alexandre Max Maraval et al. “End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes”. In: Thirty-seventh Conference on Neural Information Processing Systems. 2023.
- [40] Meta Llama team. “Introducing Meta Llama 3: The most capable openly available LLM to date”. In: Meta AI Blog (2024).
- [41] Ambarish Moharil et al. “Towards efficient AutoML: a pipeline synthesis approach leveraging pre-trained transformers for multimodal data”. In: Mach. Learn. 113

(2024), pp. 7011–7053.

- [42] Christopher E. Mower et al. “ROS-LLM: A ROS framework for embodied AI with task feedback and structured reasoning”. In: ArXiv abs/2406.19741 (2024).
- [43] Danrui Qi and Jiannan Wang. “CleanAgent: Automating Data Standardization with LLM-based Agents”. In: ArXiv abs/2403.08291 (2024).
- [44] Yujia Qin et al. “ToolLLM: Facilitating Large Language Models to Master 16000+ Realworld APIs”. In: ArXiv abs/2307.16789 (2023).
- [45] Changle Qu et al. “Tool Learning with Large Language Models: A Survey”. In: ArXiv abs/2405.17935 (2024).
- [46] Theodoros Rekatsinas et al. “HoloClean: holistic data repairs with probabilistic inference”. In: Proc. VLDB Endow. 10.11 (2017), pp. 1190–1201. DOI: 10.14778/3137628. 3137631.
- [47] Xiaozhe Ren et al. “PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing”. In: ArXiv abs/2303.10845 (2023).
- [48] Hao Sha et al. “LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving”. In: ArXiv abs/2310.03026 (2023).
- [49] Bo Shen et al. “PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback”. In: ArXiv abs/2307.14936 (2023).
- [50] Yongliang Shen et al. “HuggingGPT: solving AI tasks with chatgpt and its friends in hugging face”. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. NIPS ’23. New Orleans, LA, USA: Curran Associates Inc., 2024.
- [51] Ishika Singh et al. “ProgPrompt: Generating Situated Robot Task Plans using Large Language Models”. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). 2023, pp. 11523–11530. DOI: 10.1109/ICRA48891.2023.10161317.
- [52] Maojun Sun et al. “LAMBDA: A Large Model Based Data Agent”. In: arXiv preprint arXiv:2407.17535 (2024).

- [53] Xiangru Tang et al. “MedAgents: Large Language Models as Collaborators for Zeroshot Medical Reasoning”. In: Findings of the Association for Computational Linguistics ACL 2024. Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand and virtual meeting: Association for Computational Linguistics, Aug. 2024, pp. 599–621.
- [54] Xiangru Tang et al. “ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code”. In: (2024). arXiv: 2311 . 09835 [’cs.CL’].
- [55] Zhiqiang Tang et al. “AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models”. In: arXiv preprint arXiv:2404.16233 (2024).
- [56] Open-interpreter Team. Open-interpreter: A natural language interface for computers. https://github.com/OpenInterpreter/open-interpreter. 2013.
- [57] XAgent Team. XAgent: An Autonomous Agent for Complex Task Solving. 2023.
- [58] Arun James Thirunavukarasu et al. “Large language models in medicine”. In: Nature Medicine 29 (2023), pp. 1930–1940.
- [59] Edan Toledo et al. “AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench”. In: arXiv preprint arXiv:2507.02554 (2025).
- [60] PataraTrirat,WonyongJeong,andSungJuHwang.AutoML-Agent:AMulti-AgentLLM Framework for Full-Pipeline AutoML. 2024.
- [61] Rasul Tutunov et al. “Why Can Large Language Models Generate Correct Chain-ofThoughts?” In: arXiv preprint arXiv:2310.13571 (2023).
- [62] Colin White et al. “Neural Architecture Search: Insights from 1000 Papers”. In: ArXiv abs/2301.08727 (2023).
- [63] Qingyun Wu et al. “AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework”. In: COLM. 2024.
- [64] Liwenhan Xie et al. “WaitGPT: Monitoring and Steering Conversational LLM Agent in DataAnalysiswithOn-the-FlyCodeVisualization”.In:Proceedingsofthe37thAnnual ACMSymposiumonUserInterfaceSoftware andTechnology.UIST’24. Pittsburgh,PA, USA: Association for Computing Machinery, 2024. DOI: 10.1145/3654777.3676374.
- [65] Zhiyu Yang et al. MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization. 2024. arXiv: 2402.11453 [cs.CL].
- [66] Shunyu Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models”. In: International Conference on Learning Representations (ICLR). 2023.
- [67] ShunyuYaoetal.“TreeofThoughts:DeliberateProblemSolvingwithLargeLanguage Models”. In: Advances in Neural Information Processing Systems. Ed. by A. Oh et al. Vol. 36. Curran Associates, Inc., 2023, pp. 11809–11822.
- [68] Lei Zhang et al. “MLCopilot: Unleashing the Power of Large Language Models in Solving Machine Learning Tasks”. In: Proceedings of the 18th Conference of the European ChapteroftheAssociationforComputationalLinguistics(Volume1:LongPapers).Ed. by Yvette Graham and Matthew Purver. St. Julian’s, Malta: Association for Computational Linguistics, Mar. 2024, pp. 2931–2959.

- [69] Yuge Zhang et al. “Benchmarking Data Science Agents”. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 5677–5700.
- [70] Qinkai Zheng et al. “CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X”. In: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2023).
- [71] Terry Yue Zhuo et al. “BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions”. In: arXiv preprint arXiv:2406.15877

(2024).

