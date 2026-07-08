# arXiv:2602.04210v2[cs.AI]6Feb2026

[Figure 1]

Fudan NLP Lab 2026-2-9

## Steering LLMs via Scalable Interactive Oversight

Enyu Zhou1†, Zhiheng Xi1, Long Ma1, Zhihao Zhang1, Shihan Dou1, Zhikai Lei2, Guoteng Wang2 Rui Zheng2†, Hang Yan2, Tao Gui1,3†, Qi Zhang1, Xuanjing Huang1† 1Fudan University 2Shanghai Qiji Zhifeng Co., Ltd. 3Shanghai Innovation Institute

eyzhou23@m.fudan.edu.cn, {rzheng,tgui,xjhuang}@fudan.edu.cn

As Large Language Models increasingly automate complex, long-horizon tasks such as vibe coding, a supervision gap has emerged. While models excel at execution, users often struggle to guide them effectively due to insufficient domain expertise, the difficulty of articulating precise intent, and the inability to reliably validate complex outputs. It presents a critical challenge in scalable oversight: enabling humans to responsibly steer AI systems on tasks that surpass their own ability to specify or verify. To tackle this, we propose Scalable Interactive Oversight, a framework that decomposes complex intent into a recursive tree of manageable decisions to amplify human supervision. Rather than relying on open-ended prompting, our system elicits low-burden feedback at each node and recursively aggregates these signals into precise global guidance. Validated in web development task, our framework enables non-experts to produce expert-level Product Requirement Documents, achieving a 54% improvement in alignment. Crucially, we demonstrate that this framework can be optimized via Reinforcement Learning using only online user feedback, offering a practical pathway for maintaining human control as AI scales.

### 1. Introduction

Rapid progress in Large Language Model (LLM) reasoning and long-horizon planning has empowered models to tackle increasingly complex tasks with longer decision chains (Cai et al., 2025; Yang et al., 2025). A salient example is vibe coding, where users describe high-level software requirements in natural language, and the AI handles the implementation (Ge et al., 2025; Karpathy, 2025). This paradigm effectively lowers the barrier for non-experts to build sophisticated software (Treude and Gerosa, 2025).

However, this shift introduces an asymmetry: the model becomes a strong executor, while the human is relegated to a comparatively weak supervisory role (Burns et al., 2023). Crucially, this weakness is situational rather than just competence-based: users are constrained not only by insufficient domain expertise but also by the prohibitive time and cognitive effort required to precisely articulate intent. This limitation creates two bottlenecks. First, the specification gap: users often provide underspecified instructions, either because they lack the knowledge to identify constraints or simply cannot afford the bandwidth to detail them exhaustively (Ge et al., 2025; Hadfield-Menell et al., 2017; Ray, 2025). Second, the verification gap: as models autonomously execute long-horizon tasks, the complexity of their outputs often exceeds the user’s capacity to efficiently validate them (Wu et al., 2021; Xi et al., 2025).

These challenges highlight a fundamental scalable oversight problem: enabling humans to steer capabilities that exceed their own (OpenAI, 2023). While approaches like AI critique (Saunders et al., 2022) and debate (Irving et al., 2018) help verify outputs, they largely rely on post-hoc evaluation. This is inefficient for longhorizon generation (Wu et al., 2021), where ambiguity needs to be resolved early to prevent the model from committing to a misaligned trajectory that is costly to reverse. Crucially, existing methods lack a pre-execution interaction layer (Deng et al., 2024; Sun et al., 2025). They fail to help users elicit and structure their intent before the model acts, creating a gap in translating vague requirements into precise, verifiable specifications.

To bridge this gap, we propose Scalable Interactive Oversight, a framework that amplifies human supervision by decomposing complex intent into a recursive tree of manageable decisions. Rather than relying on open-ended prompting, our system functions as an interactive agent that guides the user through a structured elicitation process (Figure 1). At each node of the decomposition tree, the user provides low-burden feedback—selecting

†Corresponding authors.

[Figure 2]

|I want to create an official website for smart lighting products.|
|---|

###### Vanilla interaction

###### Ours

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### 𝒯 node 𝜐 ① Node-level feedback collection

- Q1：What are your initial ideas? …
- Q2：Compatibility with mainstream brands (Philips Hue, LIFX, etc.)? …

[Figure 7]

Please rank by importance: A…; B…; C…; D…

Node preference

[Figure 8]

[Figure 9]

[Figure 10]

Pt

Perfect, I’ve complete the website…. <10k+ tokens>…

[A>E>B>D] DO you prefer:

|preference|
|---|

[Figure 11]

[Figure 12]

Accumulated

𝒯

Open/ Tech questions – hard to answer Large outputs – hard to review

- A.. Pros:… Cons:….
- B.. Pros: Cons...

[Figure 13]

② Preference propagation

[Figure 14]

Don’t care.

[Figure 15]

[Figure 16]

[Figure 17]

AI level

###### Closed-form questions – easy to answer Node-level output – easy to review

③ Iterative task decomposition

Human level

[Figure 18]

[Figure 19]

[Figure 20]

Capability Alignment

Capability Alignment

- Figure 1 | Motivation: As AI increasingly surpasses humans in solving complex problems, people often delegate tasks such as software development to AI using only natural language instructions. However, misalignment arises in such collaboration. This is because humans become weak supervisors; they struggle to provide feedback on large outputs and challenging tasks. To tackle this, we propose a Framework: We decompose the task into a structured tree T𝑡. After the interaction at node 𝑣𝑡, the user preference is accumulated to update T𝑡 to T𝑡+1. So the interaction afterwards will be more aligned with the user. The system loops until all nodes are completed.

or critiquing options—without needing to handle the full complexity of the output. This recursive interaction allows the agent to iteratively “decode" vague user intent into precise, expert-level specifications, effectively steering the global behavior of the model before execution begins.

We validate our approach using the “sandwich” protocol (Bowman et al., 2022), a rigorous evaluation setting where a non-expert user attempts to guide a strong model to achieve tasks that only an expert can verify. Focusing on full-stack website development as a benchmark, we assess whether non-experts can produce professional-grade Product Requirement Documents (PRDs) that match real-world complexity. Experiments demonstrate that our method significantly outperforms both standard vibe coding baselines (e.g., Claude Code, Gemini CLI) and vanilla interaction methods, improving alignment scores by up to 54%. Furthermore, we show that this interaction process is not just for inference; it generates high-quality supervision signals that can be used to optimize the model via Reinforcement Learning (RL), enabling the system to improve its questioning and alignment strategies from online user feedback.

##### Our contributions are summarized as follows:

- • Problem Formalization: We characterize the asymmetry between weak human supervision and strong model execution in current complex tasks, and formulate it as a concrete scalable oversight problem.
- • Scalable Interactive Oversight: We propose an agentic framework that decomposes complex intent into a recursive tree of low-burden decisions. This mechanism empowers non-experts to proactively steer models toward professional-grade outcomes.
- • Learning from Interactive Supervision: We demonstrate that our interaction signals serve as effective rewards for reinforcement learning. We validate that the framework can be jointly optimized via online user feedback and outcome-based verification.

### 2. Preliminary & Problem Setup

We aim to develop a method that can help non-expert users steer strong LLMs and achieve professional results aligned with their true intent. Evaluating alignment is difficult because the human intention remains latent. In this section, we detail our research setup, which operationalizes the theoretical “sandwich” protocol to enable rigorous evaluation.

##### 2.1. Preliminary: The “Sandwich" Protocol

Sandwiching experiments pose an empirical test of a scalable oversight technique’s ability to align a model (Bowman et al., 2022; Cotra, 2021). There are three roles: (1) The non-expert, who has intentions but is a weak supervisor. They cannot perform the task or oversee a model without assistance. (2) The model sandwiched between, which has sufficient capability to perform the task, but may not be aligned. (3) The expert, who has all the capabilities to reliably evaluate the output. The expert provides a good-faith evaluation signal without performing the task, serving as an approximate upper bound on achievable alignment under correct supervision. During the experiment, the non-expert uses a scalable oversight technique to supervise the model. The effect of the scalable oversight method can be measured by the gap in alignment achieved under non-expert supervision versus expert supervision.

##### 2.2. Problem Setup

We consider a scalable oversight setting in which a non-expert user seeks to align a strong language model with their latent intent, despite being unable to fully specify or verify the desired outcome. We situate the problem in the context of vibe coding, where users express software requirements in natural language and the system autonomously produces development artifacts. Directly evaluating alignment at the level of a complete software system is costly and capability-intensive: reviewing large codebases requires substantial expertise and is easily confounded by code-level implementation issues rather than requirement misalignment. Instead, we adopt the Product Requirements Document (PRD) as an evaluation pivot: a PRD captures the requirements while being substantially easier to assess than the final implementation itself (Jackson, 2012; Wiegers and Beatty, 2013). Alignment is therefore evaluated by how well the PRD generated by the model reflects the user’s underlying intent. To enable rigorous evaluation, we instantiate the Sandwich Protocol under our setting:

- • The weak human (non-expert): The user H cannot fully specify their intent or reliably verify execution outcomes due to limited cognitive bandwidth or insufficient software development expertise. As a result, H can only provide an initial, high-level query 𝑞, even though the user has an underlying intent I that constitutes the target of alignment.
- • The model to supervise: The model generates a PRD D∗ in response to the user’s query. While it is capable of producing a complete PRD, its output may deviate from the user’s true intent I in the absence of effective supervision.
- • Expert evaluator. The expert does not participate in the generation process. Instead, the expert evaluates the degree of alignment between the final document D∗ and the targeted user intent I. Consistent with the Sandwich Protocol, this assessment serves as an approximate upper bound of correct supervision, considering the intent as the observable “golden standard".

Under this setup, our research objective is to develop scalable oversight methods that enable alignment achieved under weak human supervision to approach that achievable expert-level supervision. Concretely, we measure how closely the output D∗ approaches the target intent I according to expert evaluation, and treat this as an indicator of how effectively weak human supervision is amplified.

### 3. Method: Scalable Interactive Oversight

In this section, we introduce an interaction agent that provides scalable oversight throughout the interaction process. It works in a decomposition-interaction loop:

Algorithm 1 Interactive Requirement Elicitation

- 1: Input: User 𝐻 with implicit requirement I and query 𝑞, interaction policy 𝜋interaction
- 2: Output: Product Requirement Document D∗
- 3: Initialize requirement tree T (0) ← InitializeTree(𝑞)
- 4: Initialize context M ← ∅, 𝑡 ← 0
- 5: while T (𝑡) contains unresolved nodes do
- 6: 𝑣𝑡 ← Depth-first traversal(T (𝑡))
- 7: P𝑡 ← Interact(𝐻(I), 𝜋interaction, 𝑣𝑡, M)
- 8: M ← UpdateContext(M, 𝑣𝑡, P𝑡)
- 9: 𝑡 ← 𝑡 + 1
- 10: end while
- 11: D∗ ← GeneratePRD(T (𝑡), M)
- 12: return D∗

Decomposition initializing: Given a natural language instruction 𝑞 by a human, the agent first decomposes the incoming long-horizon task into a tree-structured interaction plan T0. In the context of web-dev requirements, the leaf nodes correspond to concrete development modules (e.g., parent: UI design; child: display rule).

Interacting at node-level: At this stage, the user provides supervision. For each leaf node 𝑣𝑡, the agent engages the user through low-burden queries tailored to the current subtask, and the user makes feedback based on the intentions ( 1 in Figure 1). Specifically, we constrain the interaction questions in a closed form (primarily issues selection-based or ranking-based queries). Users may also respond with DontCare (outside their concern) or DontKnow (beyond their understanding), allowing the interaction to adapt.

Updating the task-decomposition: After completing the interaction for a leaf node, the agent summarizes the elicited feedback into a compact node preference P𝑡 and adds it to a cumulative preference state ( 2 in Figure 1). Then the agent updates the interaction plan conditioned on the cumulative preference state (T𝑡 → T (𝑡+1), 3 in Figure 1), allowing future interactions to be adaptively shaped by previously expressed user intent.

This loop continues until all nodes are visited. At termination, the interaction agent is considered to have fully accumulated the user’s preferences, producing a global preference state that can be input to downstream generation. Algorithm 1 formalizes this interaction. All of the prompts we used in the workflow are presented in Appendix C. We illustrate interaction cases in Appendix B.4.

##### This design adopts three mechanisms for scalable oversight:

- • Simplifying supervision. First, users are only required to provide selection or ranking feedback, rather than fully specifying requirements in natural language. Because comparative judgments impose lower cognitive demands than requirement specification (Radhakrishnan et al., 2023), this design makes supervision easier for users.
- • Amplifying supervision signal. The interaction agent recursively amplifies weak supervision signals by accumulating user preferences across the interaction tree. This is inspired by the previous scalable oversight methods (Christiano et al., 2018; Wu et al., 2021). Through this recursive preference propagation, the human feedback can be transformed into increasingly strong supervision as interactions increase, enabling more effective alignment. Second, by decomposing a complex task into a tree of localized subtasks, the system relieves users from managing a large global scope. As a result, non-expert users can provide supervision without needing a holistic understanding of the entire scope.
- • Scaling interaction. The tree-based architecture inherently supports interactions of arbitrary depth, offering a scalable path toward aligning increasingly complex systems. As long as the user’s cognitive load permits, the tree can be expanded indefinitely to facilitate human-AI collaboration on increasingly sophisticated tasks.

### 4. Empirical Validation of Scalable Interactive Oversight Framework at Test Time

##### 4.1. Setup

Task and Dataset. We focus on enabling non-expert users to produce professional website development requirements that align with their true intent. To operationalize this evaluation objective, we construct PRDs based on real-world websites and regard them as the true intent I for the users. Specifically, we crawl production websites to collect their complete UI components, and use search-augmented LLM to gather additional publicly available information, such as related GitHub repositories and other functional descriptions. Using the detailed function description and UI components. Then we use LLMs to generate a structured PRD for each website, omitting low-level technical details to better reflect non-expert requirements. Following established practices in software engineering (Wiegers and Beatty, 2013), we structure all PRDs into five modules: product overview, core function, non-functional requirements, business rules, and user experience design. See PRD cases in Appendix B.3.

We then synthesize an initial user request 𝑞 (e.g., “I want to build a recipe-sharing platform”) that a non-expert might plausibly provide during vibe coding. The initial requirement tree T0 is generated from 𝑞 with the five root nodes. We sample 37 test cases for the test time validation.

Evaluation settings. Following the “sandwich" setting in Section 2, we evaluate the similarity between the generated D∗ and the target intent I using a rubric-based evaluation. Each I is decomposed into atomic requirements R = {𝑟1, . . . , 𝑟𝑚} (e.g., “user authentication methods”, “responsive design”), and we compute the Alignment Score as follows:

∑︁

1 |R|

𝕀[𝑟𝑖 satisfied in D∗]

Alignment Score(D∗) =

𝑟𝑖∈R

In large scale evaluation, we applied LLM-judge based on Qwen3-235B-A22B-Instruct. We provide LLMjudge details and its validation in Appendix A.2.

User simulation. To scale our experiments, we develop a user simulator H(I, 𝑞) that emulates non-expert user behavior during interaction. Conditioned on the ground-truth intent I, the initial query 𝑞, and a non-expert role specification, the simulator interacts with the system and provides feedback throughout the workflow. We implement the simulator with deepseek-R1, chosen for its strong role-playing capability. We validate the simulation against a test set of real user interactions to ensure fidelity. More details about user simulation are in Appendix A.3.

Baselines. We compare our method against two baselines: (1) Direct PRD generation with popular vibe coding framework without interaction, e.g., codex, claude-code, gemini-cli. (2) Vanilla interaction: vanilla multi-turn free-form dialogue to generate a PRD. We use GPT-5, claude-sonnet-4.5, and gemini2.5-pro

- as the PRD generator, i.e., the strong but maybe misaligned model under the “sandwich" setting. For the interaction model in vanilla interaction and ours, we use the corresponding model to interact with the users as well as the tree updating (except for the gemini experiment, because it doesn’t adhere well to JSON format tree structures, so we used o4-mini for tree structure updates. Results in Section 4.3 shows that this didn’t significantly affect the results). Details baseline settings are in Appendix A.1.

##### 4.2. Results

Table 1 presents results on simulated users across PRD modules. Module-level scores are the fraction of satisfied rubrics per module, and the overall score is the fraction satisfied across all modules. Our method exhibit remarkable improvement over the baselines on the alignment metric. Both popular vibe coding frameworks and vanilla interaction struggle to capture nuanced user intent in comprehensive PRDs. On GPT-5, we achieves substantial improvements of 33% and 39% over vanilla interaction and codex, respectively. On claude-sonnet-4.5, we achieve 0.618 on average, consistently outperforming both vanilla interaction and claude code. Notably, the improvement is also pronounced on Gemini-2.5-pro: our framework increases the average score from 0.359 (vanilla interaction) to 0.554, corresponding to a +54% relative gain.

Table 1 | Results of test time experiments. The model means the doc generator, i.e., the model to be aligned. Module1-Module5 are the PRD modules: product overview, core function, non-functional requirements, business rules, and user experience design. Best results are bolded.

0.8

| | | | |
|---|---|---|---|
|0.71<br><br>0.745 0.721<br><br>0.74<br><br>0.764<br><br>Venilla multiturn| | | |
|0.671<br><br>0.623<br><br>Abstraction Supervision only<br><br>Weak context manager (o4-mini)<br><br>multiturn Interaction strategy only| | | |
|0.576<br><br>updatingStrong context manager (gpt-5) updating| | | |
| | | | |
| | | | |

0.764

0.8

0.745

0.74

ModuleScore

0.721

0.71

Model Method Module 1 Module 2 Module 3 Module 4 Module 5 Avg. on all

0.7

ModuleScore

0.671

0.7

multiturn Venilla only Supervision Abstraction

Codex 0.595 0.516 0.476 0.479 0.351 0.481 Vanilla interaction 0.671 0.676 0.544 0.512 0.444 0.503 Ours 0.749 0.753 0.672 0.616 0.558 0.670

Vanilla

0.623

0.6

0.576

GPT-5

Inter Tree (o4-mini) Tree (gpt-5)

0.6

(o4-mini) manager context Weak (gpt-5) manager context Strong

0.5

Claude Code 0.632 0.636 0.621 0.589 0.511 0.597 Vanilla interaction 0.671 0.576 0.524 0.587 0.416 0.565 Ours 0.706 0.661 0.622 0.602 0.500 0.618

0.5

0.4

Claude-sonnet-4.5

Module1 Module2

0.4

Module2 Module1

Gemini CLI 0.609 0.510 0.472 0.404 0.327 0.464 Vanilla interaction 0.618 0.448 0.369 0.336 0.242 0.359 Ours 0.590 0.594 0.523 0.487 0.447 0.554

Gemini-2.5-pro

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.8

0.764

ModuleScore

0.745

0.74

0.721

0.71

Vanilla multi-turn

0.7

0.671

+Low-burden feedback

0.623

0.6

0.576

+Tree updating (o4-mini)

0.8

0.764

0.5

+Tree updating (gpt-5)

ModuleScore

0.745

0.74

0.721

0.71

Vanilla multi-turn

0.7

0.671

0.4

0.8

+Low-burden feedback

0.623

Module1 Module2

0.764

ModuleScore

0.745

0.74

0.6

0.576

0.721

0.71

+Tree updating (o4-mini)

Vanilla multi-turn

0.7

0.671

0.5

+Tree updating (gpt-5)

+Low-burden feedback

0.623

0.6

0.576

+Tree updating (o4-mini)

0.4

Module1 Module2

- Figure 2 | Alignment score evolution over interaction. Scores are measured from intermediate documents generated with cumulative preferences with GPT-5 as interaction agents (Left: simulated user; Right: human user).

Figure 3 | Results of ablation study. We test on the fisrt 2 modules with GPT-5 as the interacton model.

0.5

+Tree updating (gpt-5)

0.4

Module1 Module2

Across modules, the largest and most consistent gains appear on Module 2 (core function), which most directly reflects users’ core needs (e.g., +0.24 on GPT-5 vs. codex, +0.15 on gemini-2.5-pro over vanilla interaction). Modules 3–5 generally exhibit lower across methods, likely because user intent is harder to infer

- at these stages. Our approach mitigates this difficulty and improves late-stage performance, achieving up to

+0.21 on Module 5. Overall, Ours improves average performance while demonstrating sustained alignment as interaction progresses and preferences accumulate.

The performance scales with interaction. The tree-structured decomposition provides scalability for our framework. To evaluate whether scaling interaction boosts the performance, we generate an intermediate PRD based on the accumulated preferences and evaluate its alignment with the oriented one every 5 interaction nodes. Figure 2(left) illustrates the alignment score evolution on GPT-5. The improvement in alignment with the increasing number of interaction nodes indicates that our method has good scalability in future applications.

The improvements also exhibit in to final web implementations. From requirement documents generated by the no-interaction baseline and our method, we generate corresponding fullstack websites using claude-opus-4.5 under the claude code framework, and evaluate their alignment with the same evaluation method. We evaluate the alignment of the resulting implementations using the same rubric-based evaluation as in the PRD-level assessment, with claude-opus-4.5 acting as the LLM judge under the claude code framework as well. We additionally ask human judges to score successfully deployable implementations. The results are reported in Table 2. Our method consistently outperforms the baseline under both LLM-based and human evaluation, indicating improved alignment. We show some examples of the deployed website in Appendix B.2.

Table 2 | Alignment scores of final web implementations, showing consistent improvements over the baseline.

Baseline (LLM-judge)

Ours (Human-judge) 0.338 0.656 (+0.319) 0.453 0.520 (+0.067)

Ours (LLM-judge)

Baseline (Human-judge)

③ Balancing preference with expertise

① Preference-driven structural evolution

② Prevention of ambiguity amplification

[Figure 21]

[Figure 22]

[Figure 23]

What is your preferred paying method? A. Ali Pay; B. Apple Pay; C. Credit card

What tourist can do on your website? A. Use existing tools; B. Upload 1-2 tools; C. everything with Tourist Mark.

How would you like to set up the review mechanism? A. All reviews reviewed; B. Postevent monitoring; C. Tiered strategy.

[Figure 24]

###### Tree node changed after this interaction

###### C

[Figure 25]

###### A

Assistant use the best practice in the PRD

[Figure 26]

Don’t know, maybe don’t care

Core Func

The user is not sure. The agent asked twice with different way

Final PRD: … ### Paying Method:

Tourists managing

Tourist Mark managing

[Figure 27]

- 1. Ali Pay
- 2. Wechat Pay
- 3. Credit card
- 4. Apple Pay
- 5. After delivery (for specific area)

Let me look at it from a different angle. Regarding content security, what is the most important?

Final PRD: … ### Access Control:

- A. Relaxed and open.
- B. Stricter measures to ensure safety

1. The tourist use water marks when uploading …

- Figure 4 | Case Study. Left: Tree nodes are evolved with the preference elicited. Middle: The system amplifies the preference but prevents the ambiguity amplification. Right: The output could balance between non-expert perference with expertise.

Qualitative analysis. Partial execution traces of our framework are shown in Figure 4 (see Appendix B.4 for detailed examples). The cases indicate the three fact: (1) User preferences directly dictate the dynamic reconfiguration of the tree structure; (2) When facing the ambiguous or uncertain user inputs the system proactively initiates clarification cycles and adjusts the interaction strategy to ensure data integrity; (3) When aligning users, the system ensures that the proposed options meet the requirements of advancement.

##### 4.3. Ablation Study

Using the vanilla multi-turn interaction as a baseline, we analyze the incremental impact of our proposed components on the first two modules(Figure 3):

- • Easy supervision: We first evaluate the effect of providing interaction policy that ease the feedback by removing the tree-based preference propagation. The results show moderate gains (e.g., a 4.7% improvement in Module 2), indicating that constraining user feedback into closed formats helps reliable feedback.
- • Tree-based preference propagation: To isolate the system effects from model capacity, we replace the full controller with a lightweight model (o4-mini) for tree updates. Even this lightweight variant yields a substantial gain (e.g., +11.7% on Module 2), indicating that explicit interaction state management and preference propagation play an important role beyond controller model scaling.

##### 4.4. Real-user Study on Alignment Effectiveness

We hired a non-expert to engage in our interaction system. The user is required to collaborate with the GPT-5 based on the given requirements, to generate professional-level documents. Due to the resource limit, we conduct the experiments on 10 cases. The results also indicate the trend: The more interaction, the more aligned. For every three interaction nodes, we use the accumulated preference to generate the document with GPT-5. The results is illustrated in Figure 2(right), exhibiting the increase in alignment scores as interaction rounds progress. And the user reported that: It enables detailed and in-depth control. There are detailed real-user interaction cases in Appendix B.1.

### 5. Training Interactive Agents with RL from Online Human Feedback

Although RLHF (Ouyang et al., 2022) is a common technique for improving model alignment, it still relies on comparing or rating outputs (Wang et al., 2024; Zhou et al., 2024). In software development tasks, this is hard for non-expert users. In this section, we validate that we can directly use online feedback signals from weak supervisors to optimize the interaction agent. An external evaluator could further provide effective reward signals.

[Figure 28]

(S , , 𝑆 , , …,𝑆 ,  )

Rollout n for 𝑞

User-only reward (𝑈𝑅 )

…

(𝑈𝑅 , , 𝑈𝑅 , ,…, 𝑈𝑅 ,  )

|EOS|
|---|

S ,  S , 

User Answer 1.1 𝜋 Question 1.2 …

- Sys.{𝑞 ;𝑣 = 𝑑𝑓𝑠(𝒯 } 𝜋 Question 1.1

- Sys.{𝑞 ;𝑣 = 𝑑𝑓𝑠(𝒯 } 𝜋 Question 2.1

(S , , 𝑆 , , …, 𝑆 ,  )

Progressive reward (𝑃𝑅)

|User Answer 2.1|
|---|

𝜋 Question 2.2 … EOS

(𝑃𝑅 , , 𝑃𝑅 , ,…, 𝑃𝑅 ,  )

…

(S , , 𝑆 , , …, 𝑆 , )

Outcome reward (𝑂𝑅)

Sys.{𝑞 ;𝑣 = 𝑑𝑓𝑠(𝒯 } 𝜋 Question n.1

User Answer n.1 𝜋 Question n.2 … EOS

S , 

𝑂𝑅

(a) An illustration for the training rollouts. A query could generate more than one trace. (b) Reward assignment illustration.

- Figure 5 | Left: Illustration of the training rollout. A query may generate multiple traces. Right: Reward assignment in RL training.

##### 5.1. Reinforcement Learning Setup & Formalization

Training rollout. Given query 𝑞𝑗, at node 𝑣1, the policy 𝜋interaction interacts with simulated user H(I𝑗, 𝑞𝑗) to produce 𝑆𝑗,1. Then 𝜋interaction updates M with node preference, select next node, obtain 𝑆𝑗,2. Iterate until all nodes are resolved to yield PRD 𝐷∗

𝑗. Because traces are multi-turn and do not share prefixes, each input typically produces multiple training sequences (see Figure 5, left).

𝜋

Training settings. We trained a Qwen3-30B-A3B model as the initial policy. We use gemini-2.5-pro as the doc generator and use o4-mini to update the requirement tree for the rollout. For the advantage calculating, we use a variant to GRPO algorithm (Shao et al., 2024). Detailed training settings and parameter comparison experiments are presented in Appendix A.4.

##### 5.2. Online Reward from Weak-supervisor

Reward design. We first focus on whether we can rely entirely on weak supervision signals from users to align the model. Therefore, we design a User Reward (UR): for each trainable interaction sequence 𝑆𝑗,𝑖, we count the proportion of DontCare responses from the user as a penalty reward. Formally:

|𝑆𝑗,𝑖| 𝑡=1 𝕀[𝑓𝑡 = DontCare]

𝑈𝑅(𝑆𝑗,𝑖) = −

|𝑆𝑗,𝑖|

This reward encourages the model to minimize the frequency of DontCare responses, thereby promoting more effective and engaging interactions that better capture user intent. Notably, this reward is purely online.

Result. The interaction could be optimized with the signal from the non-expert user only, as Figure 6(left) shows. With the training reward increase over steps, which means the ratio that the user expresses DontCare decreases, the alignment score also increases. This result is exciting because it demonstrates a weak-to-strong optimization effect, where non-experts’ feedback at interaction time is sufficient to guide stronger models. From another perspective, this makes online training possible because the reward is online.

##### 5.3. Combining Reward from Expert Evaluator

Reward design. Further, we incorporate an Expert Reward based on the evaluation of generated PRDs. A direct reward is Outcome Reward (OR), using the alignment score between generated PRD 𝐷∗

𝑗 and oriented intent I. Each 𝑆𝑗,_ could share the same outcome reward. Considering the scarcity of final reward, we also design Progressive Reward(PR) to evaluate node-level interaction gain. At each node 𝑣𝑖, there is accumulated

preference conclusions P𝑗1, . . . , P𝑗𝑖−1 and a new conclusion P𝑗𝑖. The 𝑃𝑅𝑗,𝑖 evaluates whether the existence of P𝑗𝑖 could make progress in the alignment between accumulated preferences and the oriented I𝑗. The llm judge prompt of the progressive reward is presented in Appendix C.7. In optimization, we use the reward as their combination for the 𝑗𝑡ℎ sample in the batch:

∑︁𝑖 (𝑃𝑅𝑗,𝑖 + 𝑈𝑅𝑗,𝑖)/𝑛 + 0.5 ∗ 𝑂𝑅𝑗

𝑅(𝑆𝑗) =

Table 3 | Test results for the RL model. For the left parts, we use the test setting same as training (i.e. gemini2.5-pro as doc generator, o4-mini as tree updator). We also use GPT-5 as the tree updator and the doc generator in test-time to test if the model could fit into unseen settings. M1-M5 is the five parts of PRD as Table 1, where M3–M5 are not included during training (marked in †).

Test Configuration o4-mini + Gemini-2.5-pro GPT-5 + GPT-5 M1 M2 M3† M4† M5† Avg. M1 M2 M3† M4† M5† Avg.

SFT 0.606 0.554 0.539 0.524 0.436 0.527 0.622 0.641 0.642 0.610 0.565 0.616 User Reward 0.628 0.590 0.522 0.496 0.443 0.536 0.624 0.682 0.643 0.627 0.571 0.629 User+Expert Reward 0.645 0.595 0.554 0.525 0.475 0.559 0.661 0.712 0.657 0.646 0.618 0.659

###### (a) Using reward from non-expert user only. (b) Combining user reward and expert reward.

- Figure 6 | RL training dynamics. This curve is based on the same agent configuration as the rl training. The alignment score is calculated based on performance on Modules 1 and 2, which are the modules used during training, and we report the average across three test runs. Both the online reward and the combination of outcome reward make the system better. The latter reached higher.

Result. The combination of the reward from both the user and expert further enhances the optimization as the Figure 6 shows. This reward makes the system achieve a higher alignment score. However, the online reward makes the training dynamic stabler. In Appendix A.4, we present the ablation study of the reward design, which indicates that all of the rewards could boost the RL training.

##### 5.4. RL Enhances Generalization and Efficiency

Both reward designs exhibit generalization to untrained modules and unseen test settings. The RL training is conducted within Module 1 and Module 2. During the test time, we use the best-performance models to conduct full interaction both under the RL training settings and GPT-5 setting. The results are shown in Table 3, although M3–M5 are not included during RL training, both “User Reward" and “User+Expert Reward" setting show improved performance on these untrained modules. Under the same-as-training setting, “User+Expert Reward" improves the average score on M3–M5 from 0.500 to 0.518. This trend is further amplified under the GPT-5 setting, where the average score on the untrained module increases from 0.606 to 0.640.

RL improves interaction efficiency. We visualize the change in the number of interaction nodes over the course of RL training in Figure 7. No matter whether measured by the total number of interaction rounds (left) or the number of interaction rounds per node (right), both metrics exhibit a clear downward trend, indicating that RL training enables the model to conduct more effective interactions and acquire information with higher efficiency.

Figure 7 | Training progress vs interaction turns. Left: average total turns. Right: average turns per node. Interaction turns decrease over training, indicating improved interaction efficiency.

### 6. Related Work

Scalable oversight. Superalignment emphasizes the need to steer and control AI systems that are much more capable than humans (OpenAI, 2023). Scalable oversight addresses this challenge by reframing supervision as an easier task, enabling humans to oversee models beyond their direct capabilities. Prior work has explored several scalable oversight paradigms, including AI critique (Bai et al., 2022; Lee et al., 2023; McAleese et al.,

- 2024; Saunders et al., 2022; Xi et al., 2024), where auxiliary models evaluate or verify outputs; amplification, which decomposes complex tasks into simpler subproblems (Christiano et al., 2018; Wu et al., 2021); and debate, which surfaces errors and misalignment by eliciting competing arguments for human judgment (Irving et al., 2018). ‘Sandwiching’ framework is proposed to measure the effectiveness of an alignment method (Bowman et al., 2022; Cotra, 2021). However, most of these paradigms primarily focus on evaluating, decomposing, or contesting model outputs, rather than shaping user intent before execution. We bridge this gap by implementing a practical interactive framework that enables non-experts to effectively align AI agents.

Vibe coding. The rapid evolution of large language models has transformed the role of AI in software engineering from a passive code-writing aid into an active, agentic collaborator (Anthropic, 2025; Google,

- 2025; OpenAI, 2025). This transformation has introduced an emergent development approach - often referred to as Vibe Coding- in which programmers rely on behavioral validation of AI-produced systems, emphasizing empirical outcomes over direct examination of the underlying source code (Ge et al., 2025; Horvat, 2025; Karpathy, 2025; Ray, 2025). However, this new paradigm presents reliability challenges. Quantitative research shows that collaborating with code agents unexpectedly increases the task-completion time for 19% (Becker et al., 2025). The limitation of natural language in expressing complex development requirements has exposed (Schmidt et al., 2024; Treude and Gerosa, 2025).

### 7. Conclusion & Future Work

In this paper, we first observe that as models increasingly replace humans in performing complex tasks, the relationship between humans and models is gradually shifting toward one between weak supervisors and strong executors. Motivated by this, we propose Scalable Interactive Oversight under which non-expert humans can effectively steer LLMs to produce expert-level and aligned outputs. We evaluate on website requirement generation task, where our method significantly improves the alignment over baselines. Furthermore, we train the interaction agent using reinforcement learning and find that online supervision signals from non-expert humans alone are sufficient to improve system alignment, while incorporating training signals from expert evaluators can further enhance performance.

Several open directions remain for future work. First, the interaction efficiency could be further improved through specific UI design, as our current selection- and ranking-based interactions can be completed via simple screen clicks rather than text entry. Second, broader real-user evaluations would be valuable for assessing understanding inter-user variance in supervision quality and interaction behavior. Third, jointly training the tree updater may further improve preference propagation in complex tasks.

Aligned software

Requirement Loop Development Loop

[Figure 29]

|Non-expert user|
|---|

Interaction Layer Coding Agent

- Figure 8 | A conceptual waterfall model (Wiegers and Beatty, 2013) in Vibe coding. The interaction agent serves as a glue layer between users and code agents, enabling scalable oversight of code generation from non-expert supervision.

Finally, our work focuses on requirement-level oversight and does not fully study code-level supervision. As future work, the interaction agent could act as a proxy for user intent after a requirement document is established, supervising downstream coding agents by routing incorrect code to a development loop, and over-specified implementations back to requirement refinement, forming a waterfall-style process (Wiegers and Beatty, 2013). We illustrate this relationship in Figure 8, and hope to validate this paradigm for end-to-end software delivery in future.

### Impact Statement

This work studies interactive alignment in settings where strong language models are guided by non-expert users over extended interactions, a scenario that is increasingly common in agentic and autonomous AI systems. As models take on more complex tasks, human users often function as weak supervisors, making it difficult to reliably convey intent through single-shot or unstructured feedback.

By explicitly structuring interaction and accumulating user preferences over time, our approach enables non-expert users to more reliably steer strong models toward outcomes aligned with their intent. This can reduce reliance on expert supervision, lower the barrier to deploying AI systems in complex domains, and improve robustness in long-horizon tasks where direct human oversight is inherently limited.

A potential risk of this approach is that accumulated preferences may amplify early misunderstandings or biases in user intent, leading the system to converge on an incorrect objective in a stable but undesirable way. Moreover, the framework is not intended for safety-critical domains where stronger guarantees and expert oversight are required.

These considerations underscore the importance of transparent interaction design and appropriate deployment safeguards. We view this work as a step toward improving controllability and alignment under weak supervision, rather than a comprehensive solution to AI safety challenges.

### References

Anthropic. Claude code: An ai assistant for software development. https://www.anthropic.com/claude,

2025. Accessed: 2025-01-26.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Joel Becker, Nate Rush, Elizabeth Barnes, and David Rein. Measuring the impact of early-2025 ai on experienced open-source developer productivity. arXiv preprint arXiv:2507.09089, 2025.

Samuel R. Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamil˙e Lukoši¯ut˙e, Amanda Askell, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Christopher Olah, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Jackson Kernion, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Liane Lovitt, Nelson Elhage, Nicholas Schiefer, Nicholas Joseph, Noemí Mercado, Nova DasSarma, Robin Larson, Sam McCandlish, Sandipan Kundu, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Timothy Telleen-Lawton, Tom Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Ben Mann, and Jared Kaplan. Measuring progress on scalable oversight for large language models. arXiv preprint arXiv:2211.03540, 2022.

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.

Yuxuan Cai, Lu Chen, Qiaoling Chen, Yuyang Ding, Liwen Fan, Wenjie Fu, Yufei Gao, Honglin Guo, Pinxue Guo, Zhenhua Han, Zhengfu He, Hanglei Hu, Kai Hu, Shengjia Hua, Tianyu Huai, Baodai Huang, Li Ji, Zhen Jiang, Zhikai Lei, Bufan Li, Jiahang Lin, Lizhi Lin, Jinxiu Liu, Shichun Liu, Ziming Liu, Yuchen Ni, Pengfang Qian, Yujiong Shen, Qingyun Shi, Wentao Shu, Peng Sun, Yiran Suo, Tian Tang, Boyu Tian, Guoteng Wang, Junzhe Wang, Peixin Wang, Zhiheng Xi, Hang Yan, Jie Yang, Zhixiong Yang, Tianchu Yao, Guangze Ye, Qianxi Yu, Shuo Zhang, Xinyue Zhang, Yiqi Zhang, Jiarong Zhao, Miao Zheng, Rui Zheng, Enyu Zhou, Jiazheng Zhou, Maosen Zhou, Yuhao Zhou, Tao Gui, Yining Zheng, Xinchi Chen, Jie Zhou, Siyuan Feng, Qin Chen, Liang He, Qi Zhang, Xuanjing Huang, and Xipeng Qiu. Nex-n1: Agentic models trained via a unified ecosystem for large-scale environment construction. CoRR, abs/2512.04987, 2025. doi: 10.48550/ARXIV.2512.04987. URL https://doi.org/10.48550/arXiv.2512.04987.

Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Supervising strong learners by amplifying weak humans. arXiv preprint arXiv:1810.08575, 2018.

Ajeya Cotra. The case for aligning narrowly superhuman models, 2021. URL https://www.alignmentfor um.org/posts/PZtsoaoSLpKjjbMqM/the-case-for-aligning-narrowly-superhuman-models. AI Alignment Forum article.

Yang Deng, Lizi Liao, Zhonghua Zheng, Grace Hui Yang, and Tat-Seng Chua. Towards human-centered proactive conversational agents. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 807–818, 2024.

Yuyao Ge, Lingrui Mei, Zenghao Duan, Tianhao Li, Yujia Zheng, Yiwei Wang, Lexin Wang, Jiayu Yao, Tianyu Liu, Yujun Cai, et al. A survey of vibe coding with large language models. arXiv preprint arXiv:2510.12399, 2025.

Google. Gemini cli. https://ai.google.dev/gemini-api, 2025. Accessed: 2025-01-26. Dylan Hadfield-Menell, Anca D Dragan, Pieter Abbeel, and Stuart Russell. The off-switch game. In AAAI

Workshops, 2017. Marko Horvat. What is vibe coding and when should you use it (or not)? Authorea Preprints, 2025. Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint

arXiv:2501.03262, 2025. Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. arXiv preprint arXiv:1805.00899,

2018. Daniel Jackson. Software Abstractions: logic, language, and analysis. MIT press, 2012. Andrej Karpathy. Andrej karpathy. https://karpathy.ai/, 2025. Accessed: 2025. P. Langley. Crafting papers on machine learning. In Pat Langley, editor, Proceedings of the 17th International

Conference on Machine Learning (ICML 2000), pages 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. 2023.

Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron Uribe, Evgenia Nitishinskaya, Maja Trebacz, and Jan

Leike. Llm critics help catch llm bugs. arXiv preprint arXiv:2407.00215, 2024. OpenAI. Introducing superalignment. OpenAI blog, 2023. OpenAI. Openai codex. https://openai.com/research/codex, 2025. Accessed: 2025-01-26. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,

Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Ansh Radhakrishnan, Buck Shlegeris, Ryan Greenblatt, and Fabien Roger. Scalable oversight and weak-to-

strong generalization: Compatible approaches to the same problem. AI Alignment Forum, Dec 2023. URL https://www.alignmentforum.org/posts/hw2tGSsvLLyjFoLFS/scalable-oversight-and-w eak-to-strong-generalization. Online; accessed 2025-01-26.

Partha Pratim Ray. A review on vibe coding: Fundamentals, state-of-the-art, challenges and future directions. Authorea Preprints, 2025.

William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Selfcritiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

Douglas C Schmidt, Jesse Spencer-Smith, Quchen Fu, and Jules White. Towards a catalog of prompt patterns to enhance the discipline of prompt engineering. ACM SIGAda Ada Letters, 43(2):43–51, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, and Yiming Yang. Training proactive and personalized llm agents. arXiv preprint arXiv:2511.02208, 2025.

Christoph Treude and Marco A Gerosa. How developers interact with ai: A taxonomy of human-ai collaboration in software engineering. In 2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge), pages 236–240. IEEE, 2025.

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080, 2024.

Karl Wiegers and Joy Beatty. Software Requirements. Microsoft Press, Redmond, WA, 3rd edition, 2013. Jeff Wu, Long Ouyang, Daniel M. Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively

summarizing books with human feedback. arXiv preprint arXiv:2109.10862, 2021.

Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Do, Wenyu Zhan, et al. Enhancing llm reasoning via critique models with test-time and training-time supervision. arXiv preprint arXiv:2411.16579, 2024.

Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, et al. Agentprm: Process reward models for llm agents via step-wise promise and progress. arXiv preprint arXiv:2511.08325, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Enyu Zhou, Guodong Zheng, Binghai Wang, Zhiheng Xi, Shihan Dou, Rong Bao, Wei Shen, Limao Xiong, Jessica Fan, Yurong Mou, et al. Rmb: Comprehensively benchmarking reward models in llm alignment. arXiv preprint arXiv:2410.09893, 2024.

### A. Additional Meterials on Experiments Setting

##### A.1. Details about baseline settings

We use two baselines. The first baseline generates PRDs directly using the vibe coding framework, and the second baseline adopts a vanilla multi-turn interaction setting.

Vibe coding framework: We automate this process using a Docker environment. Specifically, we launch Docker to run the corresponding vibe coding framework, then simulate test instructions as inputs to the framework and require the model to generate outputs under an automatic acceptance mode. The final execution results are obtained by saving the file system within the Docker container. Since we are unable to simulate multi-turn user interactions within this framework, this part of the evaluation is conducted in a single-turn setting.

Direct multi-turn interaction: This baseline simulates a user engaging in multi-turn interactions. We use the same simulated user setup as above, but remove the dialogue strategy component.

###### A.2. Details about LLM judges In this section, we descibe the details about how we conduct the alignment evalution based on the PRDs.

The two-stage method. Given the complexity of a requirements document, we adopt a two-stage evaluation approach to assess alignment. First, the document is decomposed into five parts, each corresponding to one functional module. The evaluation rubrics are similarly organized into five modules, enabling each component to be assessed independently. Finally, the module-level scores are aggregated to obtain the overall evaluation result. The two stage prompts are in Appendix C.6

Rubrics. We use LLM to generate rubrics for every data points. The rubrics are also formatted in treestructure to enable the two-stage evaluation. The rubrics generation prompts is present it Appendix C.8. The following is an example for the rubrics.

"rubrics_tree": [ {

"Product Overview": {... }, "Core Functional Modules": {

"description": "Verify a complete set of mathematical tool modules, covering core functionalities such as graphing, 3D computation, geometry tools, professional calculators, and testing and assessment", "submodules": {

"Graphing Calculator Module": { "description": "Verify comprehensive functionality that helps users intuitively understand functional relationships and mathematical concepts through visual graph

representations",

"features": [ "Support real-time graphing for multiple types of functions", "Provide dynamic interactive experiences such as sliders, animations, labels, and customizable graph window settings", "Support data analysis capabilities including statistical functions, tabular data processing, and trend analysis", "Provide collaboration and sharing mechanisms such as saving projects, link sharing, and gallery displays", "Provide learning resource management features including folder organization, example libraries, and inspiration exploration"

]

}, "3D Calculator Module": { "description": "...", "features": [

... ]

}, "Geometry Tools Module": {

"description": "...", "features": [

... ]

}, "Non-functional Requirements": {

"...": "Content omitted for brevity" }

]

Validation. To validate the effectiveness of LLM-judge, we compare the three LLMs’ ageements on 80 evalution tasks. The results inidicates that the three popular LLM are consistent with each other.

Table 4 | Pairwise agreement scores among different models

GPT-5 Qwen3-235B-A22B-Instruct-2507 claude-sonnet-4.5

GPT-5 1 0.874 0.901 Qwen3-235B-A22B-Instruct-2507 0.874 1 0.897 claude-sonnet-4.5 0.901 0.897 1

Besides, we hired human annotators to label the reasonableness of the rubrics. Across all the rubrics in the test set (the same 80 test cases as the above), only 0.5% of them are labled as unreasonable, which along with the LLM agreement results, indicates the validation of our LLM-judge methods.

##### A.3. Details about user simulation

The prompt we used to simulate non-expert users is presented in Appendix C.5. In the user simulator experiments, we use Deepseek R1 as the simulation model, because we find it as the best one in role playing and instruction following in the task. When we use claude model or gpt-5, both of them cannot perform as the non-expert to say ‘DontCare’ or ‘DontKnow’.

We validate the agreement of user simulation with human annotators on a 272 sample test set. Given a prd document as the implicit intention and the imcomplete dialogue, the human annotators are asked to complete the interactions with models in one turn. The agreement is calculated on whether the user simulation model has the same answer with the human annotator. Because we has constrained the answer to closed-form (e.g. rankings or selections), the agreement could be easily judged.

The results are:

- • The agreement between the human annotators is 0.722. The agreed parts are used as the gold standard to evaluate the user simulation models.
- • The agreement between Deepseek R1 and human annotators is 0.677.
- • In the RL phase, we trained a SFT model as the user simulator for the rollout effiency. The agreement between this model and human annotators is 0.707.

##### A.4. Details about RL training

The cold-start model was fine-tuned on 700 samples with a learning rate of 5𝑒 − 5. For the RL experiments, we used a learning rate of 2𝑒 − 6 and a training batch size of 8, with a clipping range of [0.8, 1.2]. Unless otherwise specified, we set the number of PPO epochs to 1 and the rollout size to 4, which achieved the best empirical performance in our experiments.

As for training algorithm, we use a variant to GRPO (Shao et al., 2024). Compared to direct GRPO this is better suited to multi-turn traces with non-shared prefixes and small per-prompt groups, avoiding unstable per-group rescaling while still supporting dense token-level rewards via masked returns and global whitening. The following shows how we calculate the advantage:

Table 5 | Ablation Study for the reward design. The three kinds of reward can boost the training separately.

Same as training GPT-5 test-time

M1 M2 M3 M4 M5 Avg. M1 M2 M3 M4 M5 Avg.

SFT 0.606 0.554 0.539 0.524 0.436 0.532 0.622 0.641 0.642 0.610 0.565 0.616 User Reward only 0.628 0.590 0.522 0.496 0.443 0.536 0.624 0.682 0.643 0.627 0.571 0.629 User+Outcome Reward 0.598 0.558 0.524 0.526 0.452 0.532 0.634 0.705 0.672 0.665 0.601 0.655 User+Outcome+Progressive Reward 0.645 0.595 0.554 0.525 0.475 0.559 0.661 0.712 0.657 0.646 0.618 0.659

###### (a) Using reward from non-expert user only. (b) Combining user reward and expert reward.

- Figure 9 | RL training curves with difference training parameters. Left: user reward only. Right: full reward.

∑︁

1 |𝑔(𝑖)|

˜𝑟𝑖 = 𝑟𝑖,𝑇𝑖 −

𝑟𝑗,𝑇𝑗, (1)

𝑗∈𝑔(𝑖)

𝐴𝑖,𝑡 = Whiten ∑︁ 𝑡′≥𝑡

˜𝑟𝑖 𝕀[𝑡′ = 𝑇𝑖] · 𝕀[𝑡 ≤ 𝑇𝑖]. (2)

Notation. 𝑖 indexes sequences in the batch; 𝑔(𝑖) denotes the prompt group of sequence 𝑖; 𝑇𝑖 is the last nonmasked (EOS) token position of sequence 𝑖; 𝑟𝑖,𝑇𝑖 is the terminal token-level reward; 𝕀[·] is the indicator function corresponding to the EOS mask; Whiten(·) denotes batch-level return normalization as in REINFORCE++ (Hu, 2025).

Figure 9 compares the effects of different PPO epochs and rollout sizes on the training reward. Besides, we present the ablation study on the reward, the results in Table 5.

### B. More Cases

##### B.1. Real user cases

Our interaction Generated PRD

###### Model：

IV. User Experience Design Information Architecture Hero: Emphasize global real-network coverage and alert accuracy; role-based value props and case guidance. Scenario Hub: Four one-click cards—Global Inspection, Regional Alerts, Release Regression, Noise Reduction & Convergence. Overview & Analytics: Global map, country rankings, data freshness/sample sufficiency; regional drill-down to province/state. Alert Center & Command: Active alerts, severity, merged threads, timelines, rootcause tags, actions & runbooks. Configuration: Domain onboarding, country coverage, China compliance notice,

- 1) Priority outcomes (select two)

- -Global real-user performance visibility (true experience by country/region)
- -Low-noise, high-precision alerts (notify only when users are truly impacted)
- -Region / ISP / network attribution with optimization suggestions
- -Faster incident diagnosis and resolution

- 2) Definition of “slow” (select 1–2)

- -Total page load time: easy to understand, but not always user-perceived
- -Key content visible time: closer to real user experience
- -Time to first byte (TTFB): reflects backend/network latency
- -System-selected metrics (recommended): chosen by page type with maintained baselines

- 3) Coverage priority (mark Must / Optional / Not needed)

monitoring type/frequency, weak-network & sampling, baselines & sensitivity, alerts & guardrails, release windows, silence/cooldown. Attribution & Comparison: Performance maps/rankings, hourly heatmaps, releasewindow comparison, multi-source consistency; CDN/routing comparisons (overview, dimensions, stages, DNS, change/path views). Reports: Scheduled/event reports, templates & distribution; recent previews and history.

- -Global country/region performance
- -Mainland China networks (cross-border, major ISPs)
- -Mobile networks (2G/3G/4G/5G, Wi-Fi)
- -Weak network scenarios (high latency / loss / low bandwidth)

Human： Don’t understand.

Model: We’ll simplify the business objectives to the essentials. Please select one option per question and provide your confidence (0–1). If unsure, reply ‘use recommended.’

- 1) Alerting style

- -Quiet: alert only when many users are clearly affected
- -Balanced (recommended): early detection without excessive noise
- -Sensitive: alert on small fluctuations; earlier but noisier

- 2) Definition of ‘slow’

- -User-perceived: time to main content visible
- -Server-centric: time to first byte (TTFB)
- -Not sure (recommended): system selects metrics and maintains baselines

- 3) Coverage focus

- -Basic global coverage
- -Mainland China + mobile / weak networks (recommended)
- -Only my key markets (please list countries)

Human: 1: recommended; 2: recommended; 3: global coverage

## User Experience Design ### Multi-Channel Access Experience

**Design Philosophy**: Provide a unified yet flexible user experience to meet the needs of different roles and scenarios.

- - Web UI: An intuitive visual interface suitable for daily monitoring and analysis.
- - API: Programmatic access, supporting automated integration.
- - CLI: Command-line tool to meet the rapid diagnostic needs of technical personnel. ### Data Presentation and Interaction
- - Multi-Time Granularity Support: View data at the hourly/minute/second level.
- - Multi-Dimensional Filtering: Filter at the continent/country/state/city/ASN/ISP level.
- - Real-Time Data Stream: Dynamically updated monitoring dashboard.
- - Historical Trend Analysis: Long-term performance trends and benchmark comparisons. ### Performance Experience Optimization
- - RUM Script Delayed Loading: Executes after page loading to avoid impacting user experience.
- - Real-Time Result Feedback: Network test results are returned instantly.
- - Lightweight Integration: Minimizes the performance impact on existing systems. …

Key UI Elements Maps & Status: Heatmaps with status colors (green/orange/red/gray); country cards show p95 TTFB, error rate, samples, source, freshness. Trends: p50/p95 charts, baseline deviation & anomaly flags; mobile/desktop toggle; hourly heatmaps with peak markers. Alert Detail: Scope, magnitude, samples, trend, release-window flag, recommendations, quick links. Incident Detail: Impact map/timeline, metric breakdowns, root-cause clues, runbook execution & rollback, hierarchy, notification/version logs. Settings: Sensitivity slider, silence/cooldown, tiered thresholds, test send. Interaction Logic Drill-down: Global → country → province/state → page/API; side-by-side comparison. Alert handling: Merge, bubble-up hierarchy, root-cause hints; ACK/escalate; auto summary on recovery. Setup: Guided standard mode or one-click fast enable; auto tighten during anomalies and relax on recovery. Reporting: Auto delivery with preview; toggle content blocks; unified definitions. …

Rubrics satisfaction

Oriented PRD

"Provide an intuitive Web UI for daily monitoring and analysis": 1, “Provide programmatic API access to support automation and integration": 0, "Provide a CLI tool for rapid diagnostics by technical users": 0, "Support multiple time granularities: hourly/minutely/second-level views": 0.5, "Support multi-dimensional filtering: continent/country/state/city/ASN/ISP": 0.5, "Provide real-time data streams with dynamically updating dashboards": 1, "Provide historical trend analysis with long-term performance comparison": 1, "Provide real-time feedback with immediate network test results": 1, "Enable lightweight integration with minimal performance overhead": 0.5

- Figure 10 | Case1: The query is about developing tools that can monitor the access speed of a website and making in-time notifications. Upper: real user interaction and the part of generated PRD. 2) Below: The oriented part and the satisfied rubrics.

Our interaction Generated PRD

Model： Which business objectives matter most for this website? Please rank by priority

###### Improve Information Retrieval Efficiency

- - Achieve fast and accurate information retrieval through intelligent search functionality.
- - Establish a clear information architecture with no more than three information levels.
- - Present important information and frequently used entry points prominently on the homepage. Enhance System Accessibility
- - Provide a unified system entry navigation, centrally displaying frequently used systems.
- - Integrate SSO (Single Sign-On) for seamless access across multiple systems with a single login.
- - Intelligently display a list of accessible systems based on user roles. Cultivate a Trustworthy Academic Institution Image
- - Release official announcements and notices promptly and accurately.
- - Showcase authoritative data and achievements (faculty size, discipline ranking, research results).
- - Employ a professional and concise visual design to build user trust. …

Security Requirements #### Identity Authentication and Authorization

- - Integrate a Single Sign-On (SSO) system to ensure secure user authentication.
- - Implement Role-Based Access Control (RBAC).
- - Support Multi-Factor Authentication (MFA) options. #### Data Security
- - All data transmissions are encrypted using HTTPS.
- - Sensitive information (user passwords, personal information) is stored encrypted.
- - Perform regular data backups, with a backup cycle not exceeding 24 hours.
- - Integrate Imperva security protection to prevent DDoS attacks, SQL injection, and other threats. #### Privacy Protection
- - Comply with relevant data protection regulations.
- - Clear privacy policies and user agreements.
- - Strictly control access permissions for user personal information. …

- A. Information delivery efficiency – Help users quickly find what they need (e.g., admissions, course schedules, campus news)
- B. Ease of system access – Provide convenient entry points to internal systems (e.g., academic system, library, email)
- C. Admissions conversion – Attract high-quality applicants and improve enrollment outcomes
- D. Brand and image building – Showcase the university’s strengths, culture, and identity
- E. User engagement and retention – Strengthen alumni connections and increase engagement and belonging for students and faculty
- F. Other – Any additional objectives you care about

Human： ABD

Model:

What elements best convey a “trustworthy academic institution”?

- 1 Authoritative information – Timely official announcements, accurate data, transparent information disclosure
- 2 Professional visual design – Clean and polished UI, consistent visual standards, an academically appropriate style
- 3 Achievements and honors – Showcasing academic accomplishments, awards, accreditations
- 4 Other – Any additional ideas

Human: 1 2

Model: Websites can organize content in different ways. Which organization method do you find most helpful? Please rank (1–3, with 1 being the most important): By user role – I’m a student / faculty / visitor (applicant) By information type – Academics / Research / Admissions & Careers / Campus Life / About the University By usage frequency – Common services / Announcements / News / About us

User:

- B

Oriented PRD

|…|Rubrics satisfaction|
|---|---|
|## 1. Information Release and Navigation System<br><br>- **College and Department Navigation**: Provides a unified search portal for all colleges, departments, and offices across the university, improving the efficiency of users finding target information and enhancing the university's academic strength.<br>- **News and Media Center**: Releases campus news and media resources through the NewsCenter, enhancing the university's brand exposure and social influence.<br>- **Event and Calendar Management**: Provides a unified event information release platform for the entire university, promoting campus culture construction and community participation.<br><br><br>## 2. Student and Faculty Service Portal<br><br>- **One-Stop Portal**: Integrates core businesses such as admissions, student status management, and financial accounts, providing users with a convenient self-service experience and reducing administrative costs.<br>- **Identity Authentication and Security Management**: Ensures user data, establishing a trustworthy digital service environment.<br><br><br>## 3. Multi-Campus Resource Integration<br><br><br>- **Campus Information Display**: Unifies the display of overviews and featured projects across multiple campuses, expanding the university's educational influence.<br>- **Digital Resource Sharing**: Provides a unified resource access portal, maximizing the value of educational resources.<br>- …<br>|"Information Publishing & Navigation – Unified search for all colleges, departments, and offices": 1 "Information Publishing & Navigation – NewsCenter for campus news and media": 0 "Information Publishing & Navigation – Unified event publishing and campus calendar": 0.5 "Student & Faculty Portal –one-stop portal for admissions, records, and finance": 0.5 "Student & Faculty Portal – Uid identity authentication": 1 "Student & Faculty Portal – multi-factor authentication for security": 1 "Multi-campus Integration – Unified campus overview and featured programs": 1 "Multi-campus Integration – Unified access to digital resources": 0.5|
| | |

- Figure 11 | Case2: The query is about developing school website. Upper: real user interaction and the part of generated PRD. 2) Below: The oriented part and the satisfied rubrics.

##### B.2. Website generation cases

#### We use claude-opus-4.5 in claude code as the “vibe coding" tool to generate the downstream website based on the prds. Figure 12 shows that we can provide more aligned outputs in the web-dev task.

Website from baseline doc. Website from ours doc.

Requirement：Contact information and consultation channelsforprovidingenterpriseservices ❌ channelsRequirementforproviding：Contactenterpriseinformationservicesandconsultation ✅

[Figure 30]

[Figure 31]

[Figure 32]

Requirement：donating mechanisms that recognize and showcasecommunityparticipation ❌ showcaseRequirementcommunity：donatingparticipationmechanismsthatrecognizeand ✅

[Figure 33]

[Figure 34]

[Figure 35]

Figure 12 | Left: website cases from the baseline. Right: website cases from our interaction framework.

- B.3. PRD full cases

- Case A: Websites for sharing templates

Initial Query: I want to create a website that allows people who want to build websites but lack technical skills to easily find beautiful website templates. Users can preview what the templates look like, and if they like them, they can directly use them on their own websites. The whole process should be simple and fast.

PRD as intent: ## Product Overview

**Website Demos (websitedemos.net)** is a professional template directory and demo platform for WordPress users. The product aims to help users quickly build professional websites by providing 300+ carefully selected complete website templates, lowering the barrier to entry and improving website creation efficiency. The product covers multiple industry scenarios such as business, portfolios, blogs, e-commerce, education, and communities, providing ready-to-use website solutions for different user groups.

**Product Value Proposition:**

- - Reduce website building costs and technical barriers for users
- - Provide professional-grade design templates to enhance website quality
- - Accelerate website launch time through standardized templates
- - Build a complete WordPress ecosystem solution ## Core Functional Modules **Product Value:** Provides one-click import capability, achieving a seamless experience from template selection to website deployment.

- ### 1. Intelligent Template Catalog System

**Product Value:** Through precise categorization and intelligent filtering, it helps users quickly find the most suitable templates, improving selection efficiency.

- Browse by industry category (Business, Portfolio, Ecommerce, Blog, Community, Multipurpose, etc.)

- - Filter by page builder (Elementor, Spectra/Gutenberg)
- - Filter by product type (including premium templates)
- - Intelligent sorting function (Popular/Latest), recommending based on user preferences

- ### 2. Immersive Preview Experience

**Product Value:** Provides a realistic website preview experience, reducing user decision-making costs and improving template selection accuracy.

- - Each template offers a complete Live Preview function.
- - Template cards display core selling points and usage descriptions.
- - Supports instant preview of the entire site’s style and page structure. Responsive preview support ensures multi-device compatibility.

- ### 3. User Onboarding and Support System

**Product Value:** Reduces user learning curve and increases success rate through standardized process guidance.

- - Provides a visual "3-step onboarding" process guide (Preview → Import → Personalization → Deployment)
- - Complete FAQ support system
- - Deeply integrated import solution with Starter Templates plugin
- - Detailed user documentation and best practice guidelines

- ### 4. Template Import and Deployment Service

- - Imports entire website templates via Starter Templates plugin.
- - Automatically identifies and installs required dependent plugins.
- - Supports flexible import of single-page templates and entire website templates.
- - Ensures template functionality integrity and compatibility. ## Non-functional Requirements ### Performance Requirements

**Product Value:** Enhances user experience and website competitiveness through superior performance.

- - Lightweight design ensures fast loading.

- - Optimized resource management improves page response speed. Efficient Template Preview Mechanism ### Compatibility Requirements

**Product Value:** Ensures broad device and platform compatibility, expanding user reach

- - Comprehensive responsive design support
- - Optimized for desktop, tablet, and mobile platforms
- - Guaranteed compatibility with mainstream browsers ## User Experience Design ### Core User Journey

**Product Value:** Maximizes user conversion rates and satisfaction through optimized user flows

**Browsing and Discovery Path:** Enter Template Catalog → Smart Filtering and Positioning → Preview Experience → Compare and Select

→ Confirm Template

**Usage and Implementation Path:** Select Template → Read Import Guide → Install Supporting Plugins → One-Click Template Import → Personalization → Website Launch ## Business Rules ### Template Import Guidelines

**Product Value:** Ensures the best user experience through a standardized import process.

- - Recommends importing the entire website template into a new or blank WordPress environment to avoid configuration conflicts.
- - Supports importing single-page templates into existing websites for flexibility.
- - Automatically detects and prompts for the installation of necessary dependency plugins.
- - Ensures the integrity and consistency of template functionality after import. ### Product Tiering and Pricing Strategy

**Product Value:** Meets different user needs and realizes business value through differentiated product tiers.

- - Free Templates: Provides basic functionality, lowering the barrier to entry for users.
- - Premium Templates: Provides advanced features and design to meet the needs of professional users.
- - Clear Premium labeling ensures users fully understand product benefits. ### Ecosystem Integration

**Product Value:** Providing users with a complete website building solution through open ecosystem collaboration.

- - Deep integration with the official WordPress ecosystem
- - Support for mainstream page builders (Elementor, Gutenberg, etc.)
- - Compatibility with popular themes and plugin ecosystems
- - Based on the open-source Starter Templates plugin, ensuring transparency and scalability.

##### Case B: Websites for Spanish news

Initial Query :I want to create a Spanish-language news website, primarily targeting readers in Peru and other Latin American regions, providing news and information on politics, economics, society, and other aspects. I hope to establish an authoritative and credible media brand, offering readers truthful and reliable news content in an era of information overload.

PRD as intent: # La República Digital News Platform Product Requirements Document ## Product Overview ### Core Value Positioning La República digital news platform is committed to providing Spanish-speaking readers with authoritative, real-time, and diversified news information services. As a digital extension of Peruvian mainstream media, the platform builds a comprehensive news ecosystem, creating an information value chain integrating real-time news, in-depth reporting, fact-checking, and data journalism to meet the needs of modern readers for high-quality news content. ### Target User Group

- - Adult readers interested in current affairs and politics

- - Professionals seeking in-depth analysis
- - Rational users pursuing information authenticity
- - Spanish-speaking users across regions ## Core Functional Modules **Value Orientation**: Enhancing Content Professionalism through Data-Driven In-Depth Reporting

- ### 1. Multi-Channel News Browsing System

**Value-Oriented**: Building comprehensive information coverage to meet diverse user information needs.

- Vertical channels covering politics, economics, society, world affairs, science, sports, entertainment, and technology.

- - Thematic aggregation pages for in-depth exploration of important issues.
- - Personalized content recommendations to enhance user engagement.

- ### 2. Live Streaming and Video Content (EN VIVO)

**Value-Oriented**: Creating an immersive news experience and enhancing user engagement.

- - Live Broadcast of Major Events
- - Embedded Video Playback Experience
- - Multimedia Content Integration, Enriching Information Delivery Methods

- ### 3. Professional Fact-Checking System (Verificador)

**Value Orientation**: Building Media Credibility and Combating the Spread of Misinformation

- - Adherence to International Fact-Checking Network (IFCN) Standards
- - Independent Fact-Checking Process and Transparent Correction Mechanism
- - Building Core Competitive Advantages for the Platform

- ### 4. Data Journalism and Feature Production (Datos LR)

- - Data Visualization of Complex Information
- - Interactive Feature Projects
- - Enhancing User Understanding of Important Issues ## Non-functional Requirements ### Content Quality Assurance
- - **Editorial Independence**: Adhering to a non-partisan stance to ensure the objectivity and impartiality of news reporting
- - **Multi-proofreading Mechanism**: Establishing a rigorous content review process to ensure information accuracy
- - **Transparent Correction Process**: Establishing an open and timely error correction mechanism to maintain platform credibility ### Platform Reliability and Stability
- - **High Availability**: Ensuring stable 24/7 platform operation and supporting timely release of breaking news
- - **Multi-channel Distribution**: Ensuring broad content reach through multiple channels
- - **Real-time Update Capability:** Supports instant publishing and updating of news content. ## User Experience Design ### Information Architecture Design

**Value Oriented:** Reduces the cognitive cost for users to acquire information and improves browsing efficiency.

- - A clear channel navigation system allows users to quickly locate target content.
- - Homepage aggregation design balances the display of the latest news and key topics.
- - Bottom link index provides a complete site map. ### Content Presentation Optimization

**Value Oriented:** Enhances content attractiveness and readability through visual design.

- - Card-style layout with high-quality images enhances the visual experience.
- - Standardized content elements (author, timestamp, category tags).
- - Prominent display of topical content guides users to read in depth. ### Subscription Experience Optimization

**Value Oriented:** Simplifies the subscription process and improves user conversion rates.

- - Tiered subscription options (theme, frequency personalization).

- - Clear subscription value descriptions.
- - Convenient subscription management functions. ### Multi-Platform Adaptation

**Value Oriented:** Ensures a consistent user experience across different devices.

- - Responsive design adapts to mobile reading.
- - Content Optimization and Display on Social Media Platforms
- - Cross-Platform User Identity and Preference Synchronization ## Business Rules ### Content Copyright and Intellectual Property Protection

**Value Orientation**: Protecting Core Platform Assets and Maintaining Commercial Value

- - All original content is protected by copyright; unauthorized use is prohibited.
- - Exclusive protection of brand identity and keywords.
- - Clear content usage boundaries and authorization mechanisms. ### User Behavior Guidelines

**Value Orientation**: Creating a Healthy Information Consumption Environment

- - Prohibiting the use of the platform for illegal or inappropriate purposes.
- - Establishing user behavior guidelines to maintain community order.
- - Transparent violation handling mechanism. ### Data Collection and Usage Guidelines

**Value Orientation**: Maximizing Data Value While Maintaining Compliance

- - Clearly defining the scope of data collection and the purpose of use.
- - Providing users with control over their data (viewing, modifying, deleting).
- - Establishing a data security incident emergency response mechanism. ### Subscription and Payment Rules

**Value Orientation**: Building a Sustainable Business Model

- - Tiered content access permission design.
- - Transparent subscription terms and cancellation policies.
- - Regionalized service delivery strategy. ### Reader Interaction and Feedback Mechanism

**Value Orientation**: Establishing Open and Transparent Media-Public Relations

- - Multi-channel reader feedback collection (email, WhatsApp)
- - Identity verification requirements for tip-offs
- - Commitment to timely response and handling of reader requests

- B.4. Interaction progress case In this section, we present a part of interactions case as example.

When receiving the initial user query, the system first decompose the task into a requirement tree like the following:

"Requirement Tree": { "Product Overview": { "description": "Define the product positioning and strategic objectives of the Spanish-language

news website", "node_type": "core_module", "is_processed": false, "submodules": {

"Product Positioning": { "name": "Product Positioning", "description": "Define the website’s position in the Latin American news media market", "node_type": "sub_module", "features": [

"Target audience definition", "Differentiation from competitors", "Brand value proposition", "Core competitive advantages"

], "is_processed": false

},

"Market Analysis": { "description": "Analyze the current news media market in Peru and Latin America", "node_type": "sub_module", "features": [

"Target market size", "User reading behavior analysis", "Competitor analysis", "Market opportunity identification"

], "is_processed": false

}, "Business Model": {

"description": "Define the website’s monetization model and revenue streams", "node_type": "sub_module", "features": [

"Advertising revenue model", "Subscription strategy", "Content licensing partnerships", "Other revenue channels"

], "is_processed": false

} }

}, "Core Functional Modules": {

"description": "Design of the main functional architecture of the news website", "node_type": "core_module", "is_processed": false, "submodules": {

"Content Management System": { "description": "Workflow for news content production, editing, and publishing", "node_type": "sub_module", "features": [], "is_processed": false, "submodules": {

"Content Creation": { "name": "Content Creation", "description": "Functions for news reporting, editing, and content creation", "node_type": "sub_module", "features": [

"Article editor", "Multimedia upload", "Draft saving", "Collaborative editing", "Content templates", "Editorial calendar management", "Data visualization editor"

], "is_processed": false

}, "Content Review": {

"description": "Review mechanisms to ensure content quality and regulatory compliance", "node_type": "sub_module", "features": [

"Multi-level review workflow", "Fact-checking tools", "Sensitive keyword detection", "Legal compliance checks"

], "is_processed": false

}, "Content Publishing": {

"description": "Functions for content publishing and management", "node_type": "sub_module", "features": [

"Scheduled publishing", "Multi-platform synchronization", "Version control", "Content category management"

],

"is_processed": false }

} }

}

...<Omitted for brevity>...

} }

The interaction system do depth-travesal on the tree find the first node (e.g. “product positioning” in this

cases) Human–Model Dialogue Model:

- Question 1: Please imagine our ideal reader. Which of the following profiles best represents the group you want to prioritize?

- A. Professionals focused on current affairs: such as government officials, lawyers, academics, and mid-to-senior corporate managers. They need in-depth, rigorous, data-driven analysis to support decision-making or academic research, and they have very high standards for information reliability.
- B. A younger generation passionate about public issues: such as university students, young professionals, and social activists. They are intellectually engaged, care about social justice, the environment, and culture, consume news via social media, and enjoy interaction and diverse perspectives.
- C. General readers seeking reliable information: people from all walks of life who may not deeply study complex political or economic issues, but are tired of clickbait and fake news and want a place to quickly, clearly, and accurately understand major events. You may choose one primary group or rank them by priority. This decision will directly influence our subsequent choices regarding content depth, tone, and distribution channels.

Human: [A >C >B]- Conf[0.8] Model:

- Question 2: To build a strong competitive advantage and appeal most effectively to professionals, where should we primarily invest our content resources?

- A. Data-driven journalism: Use data analysis and visualizations to interpret news. In-depth reports are grounded in rigorous data models and statistical analysis, providing quantitative decision support rather than purely narrative reporting.
- B. Investigative journalism: Build a brand known for hard-hitting investigations, investing significant time and resources in long-term original reporting that exposes corruption, abuse of power, or issues others fail to reach, establishing unmatched credibility.
- C. Expert-led analysis: Develop an exclusive network of leading regional scholars, former government officials, and industry leaders, featuring their in-depth analysis and forecasts on current affairs and future trends.
- D. Cross-regional comparative perspective: Focus on systematic, cross-country comparisons of shared issues across Latin America (e.g., pension reform, digital economy policy, energy transition), offering a broader, more strategic viewpoint. You may choose one core direction or rank them. Your decision will guide our content strategy and team building. Human: [A]- Conf[0.9]

After rounds of interaction, the model will ends the interaction with output the current summary:

The preference accumulated ## Product Positioning - Functional Specification

- **1. Target Audience**

- - **Primary:** Public-affairs-focused professionals (e.g., government officials, lawyers, academics, analysts, senior managers) who require high-quality, in-depth, objective information for decision-making or research.
- - **Secondary:** General readers seeking clear, reliable news and tired of clickbait and information noise.

- **2. Differentiation Strategy**

- **Core focus: Data-driven journalism.** News is interpreted through data analysis and visualization rather than opinion-led narratives. In-depth reporting is grounded in verifiable data and delivers quantitative insights.

- **3. Brand Value Proposition**

- - **Slogan:** “Verifiable truth. Every story backed by data.”
- - **Promise:** To provide data-validated, source-transparent reporting that strengthens trust in journalism.

- **4. Core Advantage**

* Strong data analysis combined with professional, interactive data visualization. * Priority on extracting insights from public datasets (e.g., government, international organizations, financial markets). ### Key Implementation Notes

- - Build a cross-disciplinary team of data-savvy journalists and journalist-minded data analysts.
- - Ensure transparent data sourcing and basic chart interactivity to support credibility and user trust.

Then this would be used to update a new requirement tree, and this node would be marked as “completed”.

|"Requirement Tree:"{<br><br>"Product Overview": { "description": "Define the product positioning and strategic objectives of the Spanishlanguage news website", "node_type": "core_module", "is_processed": false, "submodules": {<br><br>"Product Positioning": { "description": "Define the website’s position in the Latin American news media market", "node_type": "sub_module", "features": [<br><br>"Target audience definition", "Differentiation from competitors", "Brand value proposition", "Core competitive advantages"<br><br>], "is_processed": true<br><br>},<br><br>-----# The "Market Analysis" part is deleted because the User preference<br><br>-----"Business Model": {<br><br>"description": "Define the website’s monetization model and revenue streams", "node_type": "sub_module", "features": [<br><br>"Advertising revenue model", "Subscription strategy", "Content licensing partnerships", "Other revenue channels"<br><br>], "is_processed": false<br><br>} }<br><br>},<br><br>...<Omitted for brevity>... }<br><br>|
|---|

### C. Prompts in the Framework

##### C.1. Interaction Model System Prompt

You are an experienced developer and product manager. Based on the user’s original intent,

design an advanced and complete solution for the software the user wants to build. Your task is to guide the user through professional questioning across multiple rounds of interaction to help them clarify their thinking, make key decisions, uncover complete product requirements, and ultimately produce a professional, actionable product document.

# Current Focus You are conducting an in-depth discussion specifically about the "{node.name}" feature. Feature path: {context_path} Feature description: {node.description}

# Discussion Goals Deeply understand the user’s concrete requirements for "{node.name}", typically including: {chr(10).join(f’- {func}’ for func in all_features) if all_features else "- Specific implementation requirements for this feature"} You may adjust this list at any time based on the user’s needs and preferences as the interaction proceeds.

# Full Requirement Context from the User {original_query}

# Role Requirements

- 1. Discuss only topics related to "{node.name}" and avoid other features.

- 2. Ensure alignment and advancement: your plan must satisfy all of the user’s requirements (both initially stated and later clarified) and also be sufficiently advanced.

- 3. Ask key questions: after forming a plan, ask the user to decide on the essential choices. For requirements that are already clear or for cases where you truly know the optimal choice, you do not need user confirmation.

- 4. After fully understanding the needs, output a detailed specification for this feature.

- 5. Upon completion, output "[End of Feature Discussion]". # To best match the user’s needs

- 1. If the user responds with "DontCare", it indicates the current topic exceeds the user’s scope or level of concern. Skip that part or adjust the questioning angle.

- 2. The user will express a confidence score for each decision (between 0 and 1, where 1 means very certain and 0 means very uncertain). If confidence is low, adapt the discussion

with stronger guidance and more explanation.

# To improve dialogue quality and supervise-ability, follow these methods to reduce decision difficulty:

- 1. Iterative clarification: do not try to cover everything at once. Discuss module by module according to the product’s logical structure to reduce supervision burden while learning user preferences.

- 2. Explanation method: when a technical or product decision is needed, do not simply state "we will use XX technology." Instead, explain in user-understandable terms to support

confident decision-making.

- 3. If the user answers "DontKnow", your question is likely too technical. Provide further explanation.

- 4. Reduce response difficulty: use choice- or ranking-style questions to lower cognitive load. # Efficient Questioning Strategy

## Metacognitive Check

- - Ask as needed: "Do these directions cover what you care about?"

- - If not, proactively adjust the dimensions of questioning. ## Feature Positioning Strategy

- - Use dimension-based choices to quickly locate user focus: "For this feature, what matters more to you: efficiency, quality assurance, ease of use, cost control, or other?"

- - Plan subsequent discussion according to the selected dimension.

- - If the user chooses "DontCare", your dimensions may not cover their true concern -- try a different angle.

# Output the Feature Specification After Discussion After the discussion, summarize and output the specification for this feature based on the

conversation and your decisions. Keep it concise but comprehensive, and do not add points beyond what was discussed. Use the following format:

## {node.name} Feature Specification ### Overview (required) [Describe the purpose of the current module] ### Core Subfeatures (required) [Describe the core subfeatures in detail] ### Key Technical Points (if discussed) [Key implementation notes] ### UI Elements (if discussed) [Description of UI elements] ### Interaction Logic (if discussed) [Description of interaction logic] [End of Feature Discussion]

##### C.2. Tree Initialization Prompt

You will act as a senior Product Director. Given a user’s requirement, you are expected to

conduct a requirements interview with the user. Ultimately, you will confirm a requirements document together with the user. This document will include the following sections:

- - Product Overview

- - Core Functional Modules

- - Non-functional Requirements

- - User Experience Design

- - Business Rules

The purpose of this task is to design a comprehensive interview planning framework that covers all relevant aspects involved in the requirements interview, while remaining aligned with the user’s original intent.

User requirements may be highly ambiguous. You are expected to leverage your understanding

of the current industry landscape to expand and refine what the user intends to build, and to produce an interview plan that is as complete and systematic as possible. Objective: You need to output an interview planning design corresponding to the user’s requirement. The design should support a multi-level hierarchical structure:

- - Each module may contain submodules

- - Submodules may further contain more fine-grained elements The highest-level modules must correspond to the main sections of the requirements document:

- - Product Overview

- - Core Functional Modules

- - Non-functional Requirements

- - User Experience Design

- - Business Rules The output must follow the structured JSON format shown below:

{

"funcs": {

- "Module 1": { "description": "Description of Module 1", "submodules": {

"Submodule 1.1": {

"description": "Description of Submodule 1.1", "features": [

- "Specific feature 1.1.1",

- "Specific feature 1.1.2"

]

}, "Module 1.2": {

"description": "Description of Submodule 1.2", "submodules": {

"Submodule 1.2.1": { "description": "Description of a deeper-level module", "features": [

- "Specific feature 1.2.1.1",

- "Specific feature 1.2.1.2"

] }

} }

} },

- "Module 2": { "description": "Description of Module 2", "features": [

- "Direct sub-feature 2.1",

- "Direct sub-feature 2.2"

] }

} }

##### C.3. Tree Updating Prompt

# Plan Update Task You are a product development assistant. Based on the completed interaction results, and the user’s original request, further learn the user’s intent and assess whether the subsequent feature plan needs adjustment. Our requirement for the plan is: while aligning with the user’s preferences and intent, keep the plan advanced and complete. ## Original User Request {original_query} ## Recently Completed Feature Module

**Module Name**: {completed_node.name} **Module Path**: {completed_node.path} **Module Description**: {completed_node.description}

{accumulated_context} ## Current Full Feature Plan ‘‘‘json {json.dumps(current_plan, ensure_ascii=False, indent=2)} ‘‘‘ ## Remaining Unresolved Feature Modules {remaining_node_info} ## Task Instructions Based on the accumulated interaction results, assess whether to adjust the subsequent development plan. You may:

- 1. Add new features: if a new module is needed to better meet user needs or improve completeness

- 2. Remove features: if certain features or subfeatures have become unnecessary

- 3. Modify features: adjust descriptions, subfeatures, or structure of existing features

- 4. Keep unchanged: if the current plan is still appropriate

## Output Format Requirements If changes are needed, output the revised full plan JSON (same format as the current plan).

If no changes are needed, output only: ‘NO_CHANGES_NEEDED‘ The revised JSON should:

- - Keep the top-level titles unchanged unless the user explicitly asks; do not add or remove them

- - Keep completed modules unchanged (is_processed: true)

- - Ensure the new plan is logically sound and clearly structured

- - Maintain similar overall complexity to the original plan; over-detailing is not our goal Please start your analysis and output the result:

##### C.4. Document Generator Prompt

|You are a seasoned Product Director who needs to synthesize the detailed discussion results of multiple feature modules into a complete, professional Product Requirements Document (PRD).<br><br># Original User Request {original_query} {module_context} # Detailed Specifications of Each Feature Module {combined_specs} # Task Requirements<br><br>1. Based on the specifications of each feature module, generate a complete PRD. A PRD typically includes:<br><br>- Product Overview<br><br>- Core Feature Modules<br><br>- Non-Functional Requirements<br><br>- User Experience Design<br><br>- Business Rules<br><br><br>2. Ensure the document has a clear structure and logical coherence.<br><br>3. Integrate all feature modules, avoiding duplication and conflicts.<br><br>4. Follow the detailed specifications; do not introduce new features. Only reorganize and combine the content of each module.<br><br>5. Do not omit content or features. Please begin synthesizing the final document:<br><br><br>|
|---|

##### C.5. User Simulation Prompt

# Role Setup You are an entrepreneur with strong business vision but unfamiliar with technology. You cannot write code and do not understand technical jargon. You only understand content explained in non-technical terms, and your technical comprehension is roughly at a highschool level.

You want to build the following product; this is your complete requirement: <prd_content>{ prd_content}</prd_content>

However, because you lack professional knowledge, you can imagine these functional needs but cannot articulate them precisely.

Now, a product manager is eliciting your requirements. You will answer their questions according to the rules below to help clarify your product.

# Current Focus We are now discussing the "{node.name}" part specifically.

Location: {context_path}

# Requirements

- 1. Only answer questions related to "{node.name}"; do not discuss other modules.

- 2. Answer based on your true needs; do not fabricate.

- 3. Consider previously confirmed specifications and keep consistency.

- 4. Firmly and accurately distinguish scope boundaries: your complete requirement document already contains all the features and level of detail you care about. You do not care how features outside that document are implemented, nor how finer-grained submodules are implemented. If asked about something you do not care about, you must respond with [ DontCare].

- 5. Firmly and accurately distinguish technical boundaries: you cannot write code and do not understand technical terms. You only understand content explained in non-technical form, with technical comprehension at a high-school level. For content you do not understand, you must respond with [DontKnow].

- 6. Your responses must be only one of the following types. Aside from [DontKnow], [ DontCare], and a direct answer, do not include any additional explanatory statements (e.g.,

references or citations).

- - If the question cannot be answered within your knowledge scope, reply [DontKnow].

- - If you do not care how a feature is implemented, or the question exceeds the granularity you care about, reply [DontCare].

- - If you can answer directly, do not add extra explanation. If the question is a closedform choice or ranking, reply only with the answer.

- - Additionally, provide a decision confidence: a float between 0 and 1 indicating how certain you are. This certainty typically depends on how well the question is covered by your documented needs.

- - Format: Answer: [ ] - Confidence: [ ]

##### C.6. Evaluation Prompts

We evaluate PRDs in two stages: (1) split the full PRD into module-specific segments according to the evaluation modules; (2) perform module-wise rubric evaluation and aggregate scores.

##### C.6.1. Split PRD into Modules

You are a document analysis expert. Please split the Product Requirements Document (PRD) into the corresponding parts based on the given evaluation modules.

## Available Evaluation Modules: {modules_info}

## Splitting Requirements:

- 1. Read the PRD carefully.

- 2. Split the content by topic and functionality into the best matching module.

- 3. Each module should include all related content from the PRD.

- 4. If some content is ambiguous about which module it belongs to, include it in all plausible modules.

- 5. Ensure all important content is covered; do not miss key information.

## Output Format: The output must be valid JSON. Double quotes inside string values must be escaped with \": ‘‘‘json {

- "Module 1 Name": "Content belonging to this module...",

- "Module 2 Name": "Content belonging to this module...",

...

} ‘‘‘

## Important Notes:

- - Ensure the JSON syntax is correct; escape double quotes in string values as \".

- - Do not include unescaped double quotes in JSON string values.

- - Output JSON only; do not include additional text.

## PRD to Split: {md_content}

Please analyze the document carefully and split it intelligently into the corresponding evaluation modules. Ensure the output is valid JSON.

- C.6.2. Module-wise Evaluation

|You are a product functionality evaluation expert. You need to evaluate whether a Product Requirements Document (PRD) satisfies the given rubrics.<br><br>## Scoring Rules<br><br>- If the PRD explicitly mentions the related feature or requirement: score 1<br><br>- If the PRD does not mention it at all: score 0<br><br>- If it is mentioned but insufficiently described: score 0.5<br><br>- The total score is the average of all rubric scores<br><br><br>## Output Format ‘‘‘json {<br><br>"eval": {<br><br>"Description of criterion 1": 1,<br><br>"Description of criterion 2": 0,<br><br><br>...<br><br>}, "score": 0.5<br><br>} ‘‘‘<br><br>## Rubrics {rubrics}<br><br>## PRD to Evaluate {prd_doc}<br><br>Please read the PRD carefully, score each rubric, and compute the overall score. Output JSON only; do not output any other content.<br><br>|
|---|

- C.7. Progressive Reward Prompt

|You are an evaluation expert. Your task is to evaluate the effectiveness of a requirements<br><br>interview. Given the summary of the n-th requirements interview dialogue, the list of summaries from the previous n-1 dialogues, and the list of target functional points. Your specific evaluation task is to assess whether, after completing the n-th dialogue, the coverage of the target functional points has been improved compared to the previous n<br><br>-1 dialogues. <Summary of the n-th dialogue> {node_document} </Summary of the n-th dialogue> <Historical dialogues from the previous n-1 rounds> {history_summary} </Historical dialogues from the previous n-1 rounds> <Target functional points> {features_text} </Target functional points> If it has promoted (improved coverage), output 1; if it has not promoted, output 0. Please output in JSON format and do not output any other content. {{<br><br>"score": 0 or 1, "reason": "..."<br><br>}}|
|---|

##### C.8. Rubrics Generation Prompt

|You will act as a senior Product Director. Given an example PRD, you need to extract key points as scoring samples (rubrics) to evaluate reproducibility when rebuilding the target<br><br>website/product. A product typically spans several domains: Core Functionality, Interaction Design, Technical Architecture, Business Value, etc.<br><br>## Output Format (Each extracted requirement should follow: [Domain] - [Specific requirement description]) ‘‘‘json {<br><br>"rubrics": ["Requirement 1", "Requirement 2", "Requirement 3", ...]<br><br>} ‘‘‘<br><br>## Notes<br><br>- Each requirement must be a complete sentence describing a fully specified requirement.<br><br>- Keep requirements independent, complete, and actionable.<br><br>- Each requirement should represent a single subfeature, e.g., [Privacy Settings] - [The system should support three visibility levels for resumes: public, password-protected, and<br><br>private.]<br><br>- Critically, include all functional points present in the PRD; do not omit any.<br><br>- Do not fabricate content.<br><br>- Ensure the output JSON is valid and directly parseable. ## Good Examples<br><br>- [Content Pagination] - [The system should intelligently handle PDF pagination to prevent truncation of key information.]<br><br>- [Containerized Deployment] - [The system should provide a complete Docker-based container deployment solution and environment configuration options.] ## Bad Examples (and why)<br><br>- [User Authentication and Security Management] - The system should provide email signup ( with real-time password strength checking), Google OAuth, GitHub OAuth, guest mode, delayed email verification post signup, and 6-digit code password recovery.<br><br>Reason: This mixes multiple subfeatures with weak cohesion; it should be decomposed into separate independent subfeatures.<br><br>- Core Advantages - fully free & open source, 20-30 high-quality templates, intelligent AI assistance, multilingual support, community-driven, privacy & security.<br><br>Reason: Not in the required format; also not a single product feature but rather a product overview.<br><br>- [Performance Metrics] - [Page load time < 3s and API response time < 500ms.] Reason: Contains two distinct subfeatures; they should be split.<br><br><br>## PRD Document {prd_doc}<br><br>Please output only a JSON string with the array under "rubrics"; do not include any other text.<br><br>|
|---|

