# arXiv:2511.06449v2[cs.LG]8Dec2025

[Figure 1]

[Figure 2]

[Figure 3]

## FLEX: Continuous Agent Evolution via Forward Learning from Experience

Zhicheng Cai1,6,∗, Xinyuan Guo1,4,∗, Yu Pei1∗, Jiangtao Feng1,3,†, Jinsong Su3,5, Jiangjie Chen2,6, Ya-Qin Zhang1,6, Wei-Ying Ma1,6, Mingxuan Wang2,6, Hao Zhou1,6,‡

1Institute for AI Industry Research (AIR), Tsinghua University 2ByteDance Seed 3Shanghai Artificial Intelligence Laboratory 4University of Chinese Academy of Sciences 5School of Informatics, Xiamen University 6SIA-Lab of Tsinghua AIR and ByteDance Seed

∗Equal Contribution, †Project Lead, ‡Corresponding Author Correspondence: zhouhao@air.tsinghua.edu.cn

### Abstract

Autonomous agents driven by Large Language Models (LLMs) have revolutionized reasoning and problem-solving but remain static after training, unable to grow with experience as intelligent beings do during deployment. We introduce Forward Learning with EXperience (FLEX), a gradient-free learning paradigm that enables LLM agents to continuously evolve through accumulated experience. Specifically, FLEX cultivates scalable and inheritable evolution by constructing a structured experience library through continual reflection on successes and failures during interaction with the environment. FLEX delivers substantial improvements on mathematical reasoning, chemical retrosynthesis, and protein fitness prediction (up to 23% on AIME25, 10% on USPTO50k, and 14% on ProteinGym). We further identify a clear scaling law of experiential growth and the phenomenon of experience inheritance across agents, marking a step toward scalable and inheritable continuous agent evolution. Project Page: https://flex-gensi-thuair.github.io.

[Figure 4]

Figure 1 An overview of our FLEX paradigm and main results. Top: A comparison between traditional gradient-based learning, which uses Back-Propagation as the optimizing method and our proposed Forward Learning from Experience paradigm. Bottom: Main experimental results demonstrating FLEX’s effectiveness. We evaluate FLEX against strong baselines across three challenging scientific domains (Mathematics, Chemistry, and Biology) on a diverse suite of over 10 models, where FLEX consistently and substantially outperforms the baselines with the cost less than 100$ for both training and evaluation of a single agent.

### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2 Forward Learning with Experience . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 2.1 Overview of FLEX . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Information-Theoretic Insights into FLEX . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.3 Optimize FLEX as Meta-MDP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Concrete Instantiation of FLEX . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3.1 Extensive Experience Exploration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Experience Library Evolution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 4 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.3 The Scaling Law of Experience . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.4 Inheritance of the Experience Library . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.5 Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 5 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 6 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 6.1 Learning Paradigms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 6.2 Self-Evolving Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

A Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26 A.1 Biology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

### 1 Introduction

Autonomous agents have achieved remarkable advances across diverse domains, including code generation (e.g., Claude Code [1]), scientific discovery (e.g., Biomni [2], STELLA [3], and SciMaster [4]), and in-depth research (e.g., Tongyi, OpenAI, and Gemini DeepResearch [5–7]), showcasing their immense potential to solve complex, open-ended real-world tasks end to end.

However, pretrained agents remain static with frozen parameters, fundamentally precludes learning from on-the-fly experience concluded during trial-and-error, causing substantial performance degradation on challenging or untrained tasks [8, 9]. While gradient-based learning has underpinned model optimization [10– 12], it is inherently ill-suited for continuous agent evolution due to three key obstacles: (i) back-propagation incurs prohibitive computational costs [13, 14]; (ii) parameter-centric paradigm suffers from catastrophic forgetting [15, 16], failing to effectively learn form accumulated experiences over times; (iii) the closed-source nature of most state-of-the-art LLMs [17–21] renders direct parameter optimization infeasible.

Recent self-evolving paradigms attempt to overcome this limitation by evolving non-parametric components (i.e., prompts [22, 23], workflows [24–26], and tools [27, 28]) through environmental feedback and textual gradient [22, 29, 30]. Nevertheless, these approaches encounter three vital bottlenecks towards continuous agent evolution: (i) their evolving components are task-specific, thus they cannot generalize across tasks, limiting cross-task evolution. (ii) Their evolving components (i.e. prompts, workflows, and tools) are not continuously scalable, thus they can only leverage a limited set of experiences, preventing performance from scaling with accumulated knowledge. (iii) Their evolving components are model-specific, thus new agents must interact from scratch during deployment, unable to continuously evolve from agents’ previous off-policy experiences, leading to substantial computational overhead.

To address these limitations, we propose Forward Learning from Experience (FLEX), a novel learning paradigm that shifts learning from modifying model parameters to constructing and leveraging an evolvable experience library (Fig.1). In contrast to traditional gradient-based learning paradigms, which optimize through both forward- and backward-propagation, FLEX involves mere forward passes in three stages: (i) exploring forwardly to acquire extensive problem-solving experiences; (ii) evolving the experience library via the updater; (iii) guiding forward reasoning by utilizing pertinent experiences from the library.

Hence, FLEX overcomes the prevailing bottlenecks with three profound properties for continuous evolution: (i) FLEX evolves a persistent experience library that continuously aggregates cross-task experiences, ensuring that knowledge is retained and available for future use. (ii) The library continuously expands and refines, enabling agents to progressively acquire deeper insights and knowledge. This accumulation allows agent performance to scale with experience. (iii) FLEX stores strategies distilled from experiences in a semantic form, making the library both transferable and interpretable. This design enables seamless inheritance across agents, avoiding redundant and costly re-training.

To validate the general effectiveness of FLEX paradigm, we conduct extensive experiments across diverse challenging scientific domains, including Olympiad-level mathematics (AIME25 [31]), chemical retrosynthesis (USPTO50k [32]), and protein fitness prediction (ProteinGym [33]). FLEX demonstrates substantial and consistent improvements on these tasks, from 40% to 63% on AIME25, 20% to 30% on USPTO50k, and 46% to 60% on ProteinGym, exhibiting great enhancement in the capacity of reasoning and knowledge leverage.

In summary, our contributions are as follows:

- • We propose FLEX, a new learning paradigm for the agentic era. It redefines learning as a forward exploration and experience distillation process, enabling LLM agents to continuously evolve through accumulated experience without gradient-based tuning.
- • We provide a comprehensive framework for FLEX, including a unified mathematical formulation with theoretical justifications, a practical instantiation with concrete mechanisms, and an empirical demonstration of its effectiveness on diverse scientific benchmarks.
- • We identify and empirically validate a scaling law of the experience library, showing that agent performance scales predictably with accumulated experience. We also introduce and demonstrate the

principle of experience inheritance, where distilled experience can be transferred between agents in a plug-and-play manner, enabling instant knowledge assimilation and bypassing redundant learning.

### 2 Forward Learning with Experience

In this section, we establish a unified mathematical formulation of FLEX paradigm. A learning paradigm typically consists of two key components: an optimization objective (defining what to learn) and an optimization process (defining how to learn it). We next illustrate how FLEX implements these components.

#### 2.1 Overview of FLEX

FLEX enables LLM agents to continually evolve through learning from experience. Instead of fine-tuning model parameters, FLEX focuses on constructing an experience library that captures and reuses knowledge distilled from past problem-solving trajectories, guiding future reasoning.

FLEX paradigm introduces a forward learning loop with two interleaved phases: (i) extensive forward exploration by the actor to collect and distill new experiences, which are integrated into the library by the updater; (ii) utilizing the most pertinent experiences to guide forward reasoning on new tasks. As the experience library expands, it accumulates diverse, high-quality problem-solving strategies, allowing agents to leverage prior knowledge for increasingly superior reasoning, all without gradient-based parameter updates. Through this iterative process, agents continuously learn from accumulated experience, progressively enhancing their performance on challenging problems. Fig. 1 illustrates the overall workflow of FLEX, showing the interaction between the actor π, updater µ, and the experience library E.

Specifically, the LLM agent π remains frozen while being driven to explore the environment extensively, generating interaction trajectories {τ | X,π} for a given query X. These trajectories are dynamically consolidated into the ever-evolving experience library E by an auxiliary updater agent µ. Thus the library update can be denoted as Enew ∼ µ(· | Eold,{τ | X,π}). The learned library accumulates distilled knowledge over time and subsequently serves as a guidance source for reasoning on future tasks.

At inference time, the most pertinent and helpful experiences are identified and fetched through ε ∼ ρ(· | q,E) from the learned experience library E given query q. Conditioned on these retrieved experiences ε, π is enabled to invoke learned strategies and latent cognitive patterns, producing superior response π(· | q,ε) that exhibit significantly enhanced reasoning capability and expert-level knowledge utilization compared to the base model.

Formally, we define the optimization objective of FLEX as constructing an optimal experience library E∗ that maximizes expected correctness on training tasks, and operationalize it through a forward update process.

Optimization Objective. The optimization goal of FLEX is to construct an experience library that maximizes the expected correctness of the model’s outputs, thus we have:

- Definition 1 (Optimization Objective of FLEX). Given a training set D = {(Xi,Yi)}Ni=1, where Xi is a query and Yi is the corresponding ground truth, the objective function J (E) and the optimal experience library E∗ is defined as:

J (E) = E(X

i,Yi)∼D, εi∼ρ(·|Xi,E) Φ π(· | Xi,εi), Yi , E∗ = arg max

(1)

J (E)

E

where ρ(· | Xi,E) retrieves experiences εi from library E given query Xi, π(· | Xi,εi) is the LLM agent conditioned on both the query and retrieved experiences, and Φ(Yˆ,Yi) measures the correctness of prediction Yˆ against ground truth Yi.

This objective serves as the guiding principle for the experience library evolution in FLEX, explicitly linking the quality of the experience library to the model’s ultimate performance on target tasks.

Optimization Process. FLEX operationalizes the objective in Definition 1 through a forward learning process. Instead of relying on gradient-based parameter updates, learning occurs via forward probabilistic updates to the experience library, carried out by the updater agent µ. This process distills knowledge from interaction trajectories generated by π, steering the experience library toward optimization of the objective function. Therefore, the learning update rule for FLEX can be analogously defined as follows:

- Definition 2 (Update Rule of FLEX). The learning process in FLEX is governed by a forward update rule that evolves the experience library through forward probabilistic updates with experiential exploration, defined as:

Ei+1 ∼ µ · | Ei,{τi | Xi,π} , ∇EJ (Ei) ≜ µ · | Ei,{τi | Xi,π} − Ei

(2)

where {τi | Xi,π} represents the set of interaction trajectories generated by actor agent π given input Xi, µ(· | Ei,{τi | Xi,π}) is the update distribution that distills new experiences from {τi | Xi,π} into Ei, and ∇EJ (E) denotes the optimization direction for improving the experience library.

In contrast to traditional gradient-based methods, which involves computing gradients of the objective function with respect to model parameters and updating them via backpropagation, i.e., θi+1 = θi − ∇θJ (θi), FLEX introduces a fundamentally distinct paradigm by updating the experience library E through forward probabilistic exploration rather than optimizing parameters via backward gradient propagation, allowing the experience library to dynamically adapt and enhance model performance through accumulated experiential knowledge. Fig. 1 exhibits the differences between gradient-based learning and FLEX.

#### 2.2 Information-Theoretic Insights into FLEX

To gain deeper insight into how the experience library drives learning, we reinterpret the optimization objective from an information-theoretic perspective. This reformulation reveals that FLEX effectively learns to reduce predictive uncertainty and maximize the information gained from retrieved experiences.

Typically, Φ(Yˆi,Yi) = I(Yˆi=Yi), thus the objective J (E) in Eq. 1 can be equivalently written as maximizing the log-likelihood:

J (E) = E(X

i,Yi)∼D, εi∼ρ(·|Xi,E) [log π(Yi | Xi,εi)], E∗ = arg max

(3)

J (E)

E

Further, the log-likelihood can be equivalently expressed in terms of conditional entropy H and KL divergence:

H(Yi | Xi,εi) + KL p∗(· | Xi,εi)∥π(· | Xi,εi) (4)

J (E) = −E (X

i,Yi)∼D εi∼ρ(·|Xi,E)

After the extensive pre- and post-training, the LLM π is assumed to possess sufficient comprehension and reasoning capabilities to effectively utilize input information. Hence, the model distribution π(· | Xi,εi) can be regarded as sufficiently close to the true conditional distribution p∗(· | Xi,εi). Consequently, the second term becomes negligible, and the optimization objective can be approximated as minimizing the expected conditional entropy:

Corollary 1 (Information-Theoretic Reformulation of the Objective). The optimal experience library E∗ that maximizes J (E) can be approximated by minimizing the expected conditional entropy:

E∗ = arg max

H(Yi | Xi,εi) . (5)

E (X

L(E) ≈ arg min

i,Yi)∼D εi∼ρ(·|Xi,E)

E

E

This perspective provides a more intuitive interpretation, FLEX learns to construct an experience library that reduces the model’s predictive uncertainty while aligning the model distribution with the underlying true distribution.

Information-Theoretic Role of Experience. To further elucidate the role of retrieved experiences, we analyze their contribution from an information-theoretic perspective. Specifically, the expected mutual information I

[Figure 5]

- Figure 2 Illustration of the Meta-MDP formulation of FLEX. The Base-level MDP performs intra-sample exploration and experience distillation, while the Meta-level MDP integrates these experiences to evolve the global experience library through forward updates.

between the target output Yi and the reasoning experience εi ∼ ρ(· | Xi,E) is defined as: E (X

H(Yi | Xi,εi) . (6)

I(Yi;εi | Xi) = E (X

H(Yi | Xi) − E (X

i,Yi)∼D εi∼ρ(·|Xi,E)

i,Yi)∼D εi∼ρ(·|Xi,E)

i,Yi)∼D εi∼ρ(·|Xi,E)

Since H(Y | X) remains constant for a given dataset, minimizing H(Y | X,ε) is equivalent to maximizing I(Y ;ε | X). This implies that retrieved experiences ε provide additional information about the target Y beyond what is contained in the query X, thereby enhancing prediction confidence and shaping the conditional output distribution π(Y | X,ε).

We can further quantify the contribution of informative experiences by comparing their information gain to irrelevant ones. Let ε+i denote effective experiences and ε−i denote irrelevant ones. The difference in conditional mutual information is:

∆I = I(Yi;ε+i | Xi) − I(Yi;ε−i | Xi) = H(Yi | Xi,ε−i ) − H(Yi | Xi,ε+i ) (7)

The optimization objective in Corollary 1 guarantees H(Yi | Xi,ε+i ) < H(Yi | Xi,ε−i ), thus ∆I > 0, indicating that effective experiences yield greater information gain. To conclude, incorporating informative experiences

substantially reduces predictive uncertainty, guiding the model toward more accurate and confident reasoning.

- 2.3 Optimize FLEX as Meta-MDP

Unlike conventional gradient-based optimization, FLEX adopts a forward, experience-driven semantic evolution paradigm. Since the update rule of FLEX in Definition 2 produces an updated experience library that depends solely on the current state and is independent of historical states, we can formalize the optimization procedure as a hierarchical Meta Markov Decision Process (Meta-MDP) as shown in Fig 2, comprising Base-level MDPs that explores extensive experience, and a Meta-level MDP that governs the experience library evolution, enabling forwardly learning from accumulated experience instead of gradient back-propagation.

##### Meta-level MDP for Experience Library Evolution

- Definition 3 (Meta-level MDP for Experience Library Evolution). The Meta-level MDP is formally defined as the tuple (Sm,Am,T m,Rm):

- • Meta-state smi : The current experience library Ei
- • Meta-action ami = Ψ(EiT | Xi,Yi,Ei): Invokes the Base-level MDP on sample (Xi,Yi) under Ei, yielding explored experiences EiT
- • Transition function T m: Conducted by the updater µ, evolves the experience library as Ei+1 ∼ T m(Ei+1 | Ei,EiT) under the meta-policy

- • Reward rim = Φ π(· | Xi,ρ(· | Xi,Ei+1)),Yi : Measures correctness of model output given Ei+1

Consequently, the cumulative return Gm = Ni=0−1 rim over one meta-episode reflects the aggregated model accuracy across all N training samples. Therefore, maximizing Gm is essentially equivalent to optimizing the

objective in Eq. 1.

In summary, this Meta-level formulation establishes a higher-order optimization loop that continuously refines the experience library through forward experiential updates, orchestrating Base-level exploration with long-term performance optimization.

Base-level MDP for Extensive Experiences Exploration The Base-level MDP operationalizes the exploratory component of FLEX with an actor–critic framework. The actor iteratively generates diverse reasoning trajectories, while the critic provides semantic feedback that reflects on these trajectories, distilling

- them into structured experiences for Meta-level optimization, thus can be formulated as,

- Definition 4 (Base-level MDP for Experience Exploration). The Base-level MDP is formally defined as the tuple (Sb,Ab,T b,Rb):

- • Base-state sbt: The accumulated experience Eit derived from the first t reasoning trajectories explored on Xi given Ei
- • Base-action abt: One complete reasoning trajectory proposed by the actor π
- • Base-reward rtb = Γ(abt,Yi): Semantic feedback distilled by the critic, evaluating both the reasoning process and outcome relative to Yi, capturing success patterns and interpretative insights
- • Transition function T b: Updates the intra-sample experience accumulation as sbt+1 ∼ T b(sbt+1 | sbt,rtb), progressively integrating reflective knowledge until exploration terminates.

Upon completion, the accumulated experience EiT = sbT is returned to the Meta-level MDP for high-level experience library evolution.

Together, the Base-level and Meta-level MDPs form a hierarchical, closed-loop optimization process, the Base-level conducts fine-grained exploration and reflection within individual samples, while the Meta-level integrates these experiences across samples to continuously evolve the experience library in a gradient-free, forward-learning manner.

### 3 Concrete Instantiation of FLEX

Building on the mathematical formulation, we instantiate FLEX as a concrete mechanism consisting of two core components, namely, extensive experience exploration via the actor–critic forward loop and experience library evolution that consolidates and reuses experiential semantics. Together, these two components realize continual semantic evolution through structured experience interaction as shown in Fig 3.

#### 3.1 Extensive Experience Exploration

The generation and collection of experiences are realized through per-case actor–critic interactions defined by the Base-level MDP. The key objective is to perform extensive exploration while ensuring that the collected experiences are both diverse and semantically informative.

Extensive Exploration Mechanism. To enable extensive high-quality exploration, we design a dualscaling mechanism that combines parallel and sequential scaling. For parallel scaling, multiple reasoning trajectories are sampled in parallel for the same input through rejection sampling [34–36], allowing diverse reasoning hypotheses to be explored and high-quality outputs to be retained. For sequential scaling, the actor iteratively improves upon its previous output under the critic’s semantic feedback [22, 29], where the critic provides constructive guidance and refinement cues derived from the prior round. This cooperative

[Figure 6]

- Figure 3 Concrete Instantiation of FLEX. The refinement loop of actor-critic iteratively explores and refines experiences, then the meta-level updater dynamically organizes the distilled experiences into the evolving experience library.

exploration strategy balances breadth and depth, ensuring both coverage of alternative reasoning paths and progressive improvement over iterations.

Iterative Refinement with Semantic Signal. To ensure that collected experiences contribute meaningfully to learning, we design an iterative refinement mechanism driven by semantic evaluation. For reasoning trajectories that lead to incorrect conclusions, the critic first identifies the underlying causes of failure and summarizes them into abstract, task-agnostic improvement suggestions. These generalized feedback signals, denoted as new experiences Et, are fed back into the actor together with its previous reasoning trace to initiate the next actor–critic iteration. This process continues until the actor produces a correct or sufficiently improved reasoning trajectory, or a predefined iteration limit is reached. In this loop, the critic’s feedback functions as a semantic update signal [22, 29, 37] that guides the actor’s reasoning refinement, analogous to the gradient signal in traditional optimization, yet expressed in natural language semantics rather than numerical differentials. Unlike backpropagation-based learning, which relies on scalar gradients with limited interpretive bandwidth and heavy computational budget, this forward semantic feedback mechanism provides rich interpretable guidance without costly parameter updating, aligning with the cognitive characteristics of human-like learning from experience.

Once the iterative refinement completes, validated experiences are distilled and passed to the Meta-level MDP, evolving experience library for long-term accumulation.

#### 3.2 Experience Library Evolution

The experience library is hierarchically organized and continually evolves within the Meta-level MDP formulated above, driven by two complementary operations, update and retrieval, which together constitute the core meta-level transition dynamics.

Hierarchical Experience Library. The experience library adopts an explicit textual storage mechanism, which enables explicit recording of experiential rules, externalization of internal reasoning behaviors, and visualization of the learning dynamics, thereby enhancing interpretability and transparency. To effectively organize learned experiences varying in granularity, abstraction, and contextual dependency, we design a hierarchical organization mechanism with three interconnected levels, namely, high-level strategic principles and guidelines, mid-level reasoning patterns and methodological templates, and low-level factual knowledge and concrete instances. In addition, the library is partitioned into two complementary zones: the golden zone, which stores experiences distilled from correct reasoning trajectories, and the warning zone, which records failure cases and diagnostic insights extracted from erroneous trajectories. Such dual-zone organization facilitates balanced learning from both successes and failures, allowing the agent to not only consolidate validated reasoning paths but also internalize the lessons encoded in previous mistakes. This structure allows

experiences to be modularly indexed and retrieved across abstraction levels, enabling both top-down guidance and bottom-up consolidation of knowledge. Hierarchical organization also supports dynamic restructuring as new experiences are incorporated, maintaining the coherence of the library during continual evolution.

Experience Update Mechanism. Given a newly generated experience ε, the updater agent autonomously decides whether and how to incorporate it into the experience library. If an identical entry already exists, ε is discarded to maintain compactness. If semantically similar entries are found, the agent performs a selective merge, preserving the more informative or higher-quality trajectory while minimizing redundancy. Otherwise, ε is inserted into the appropriate partition and hierarchical level according to its semantic granularity. This adaptive update process ensures that the experience library incrementally evolves toward a more structured and informative representation of accumulated knowledge, rather than expanding indiscriminately or degenerating into redundant memory accumulation.

Experience Retrieval Mechanism. During inference, the experience library is encapsulated as an interactive tool callable by the agent [28, 38]. Instead of static vector-based semantic-similarity search [39–41], the agent performs contextualized retrieval conditioned on the specific query and its current reasoning state. This allows the agent to interpret relevance not only by semantic similarity but also by contextual consistency and goal alignment, thereby avoiding the pitfalls of retrieving semantically similar yet pragmatically contradictory entries. Retrieval proceeds hierarchically, the agent first identifies relevant high-level strategies,

- then locates corresponding procedural patterns, and finally accesses concrete case examples within each pattern. At each stage, the top-k (typically k = 5) most relevant entries are selected to balance precision and efficiency. The retrieved experiences support the agent’s ongoing reasoning, reflective self-evaluation, and corrective refinement. Moreover, retrieval is not restricted to pre-inference access, instead, the agent can dynamically query the library during reasoning, enabling adaptive integration of prior knowledge into evolving cognitive trajectories.

Overall, the update–retrieval dynamics establish a self-organizing experiential system that progressively refines its hierarchical organization and semantic granularity, facilitating the agent’s continual evolution through accumulated experience.

### 4 Experiments

We evaluate the FLEX paradigm across a diverse suite of challenging environments, testing its effectiveness, and demonstrating the scaling law, transferability of the learned experience library, and a thorough case study in all settings.

#### 4.1 Experimental Setup

Our experimental setup are structured to comprehensively evaluate the performance and generalizability of FLEX, involving diverse benchmarks, comparisons against established baselines, and a detailed illustration of the training procedures.

Evaluation Benchmarks. We evaluate the proposed FLEX paradigm across four challenging benchmarks spanning three scientific domains (i.e., mathematics, chemistry, and biology) to assess both reasoning capability and domain adaptability. Specifically: (1) AIME25 competition dataset [31] for complicated mathematical reasoning; (2) GSM8k dataset [42] multi-step arithmetic reasoning; (3) USPTO50k [32] benchmark for canonical single-step retrosynthesis; (4) ProteinGym [33] benchmark for protein fitness prediction.

Compared Methods. Our evaluation spans a range of base models, tested on four distinct experimental configurations for comparison: (1) Vanilla LLM, (2) LLM with In-Context Learning (ICL) [43], (3) LLM agent with reasoning-and-acting workflow (Agent) [44], and (4) our proposed FLEX. For FLEX, each model constructs an experience library during training, which is later utilized at test-time. The library structure is

- Table 1 Results on four benchmarks. All values are accuracies in percentage, which is accuracy for AIME25, GSM8k, USPTO50k and spearman’s ρ for ProteinGym. ICL stands for the performance of LLM with in context learning, and Agent indicates the performance of LLM agent with ReAct workflow.

Benchmark Models LLM ICL Agent FLEX Math

AIME25 Claude-Sonnet-4 [20] 40.0 30.0 (-10.0) 50.0 (+10.0) 63.3 (+23.3) DeepSeek-V3.1-Terminus [45] 56.7 53.3 (-3.3) 60.0 (+3.3) 66.6 (+10.0)

GPT-3.5 [46] 80.8 81.4 (+0.6) 78.5 (-2.3) 83.3 (+3.3) GPT-4 [47] 93.8 94.2 (+0.4) 94.0 (+0.2) 95.9 (+2.1) Llama-3.2-1B [48] 74.3 75.8 (+1.5) 71.0 (-3.3) 80.9 (+6.6) Llama-3.2-3B [48] 78.4 78.8 (+0.4) 74.5 (-3.9) 81.1 (+2.7)

GSM8k

Chemistry

GPT-5 [18] 9.0 14.0 (+5.0) 12.0 (+3.0) 16.0 (+7.0) Gemini-2.5-Pro [17] 9.0 15.0 (+6.0) 12.0 (+3.0) 18.0 (+9.0) Claude-Sonnet-4.5 [19] 20.0 24.0 (+4.0) 23.0 (+3.0) 30.0 (+10.0)

USPTO50k

Biology

DeepSeek-V3.1-Terminus [45] 47.9 48.9 (+1.0) 48.6 (+0.7) 56.8 (+8.9) GPT-OSS-120B [49] 47.7 49.8 (+2.1) 51.5 (+3.8) 57.3 (+9.6) Claude-Sonnet-4 [20] 46.0 49.8 (+3.8) 50.2 (+4.2) 59.7 (+13.7)

ProteinGym

adapted to the benchmark: hierarchical for AIME25 and USPTO50k, and non-hierarchical for GSM8k and ProteinGym.

Training Configurations. For AIME25, the training was conducted on 49 historical problems from AIME83

- to AIME24. On GSM8k, we adhere to the official training and testing splits [42]. For USPTO50k, 100 instances are randomly sampled from the original 5k test split for evaluation, and 50 instances from the training split are randomly sampled for training. For ProteinGym, 100 mutational sequences are sampled for each wild-type protein target as training set, which is on average 1.47% of the available data. This design reflects practical constraints in protein engineering, where only a limited number of labeled mutations are known, while the vast majority of sequence space remains unexplored.

#### 4.2 Main Results

Our main results demonstrate that FLEX achieves significant and consistent improvements across all scientific domains and benchmarks, as detailed below.

Math. As shown in Table 1 and Figure 1, our FLEX paradigm consistently and significantly outperforms all baselines across both mathematical reasoning benchmarks. On the challenging AIME25 dataset, FLEX boosts the accuracy of Claude-Sonnet-4 from a 40.0% baseline to an impressive 63.3% after learning from only 49 examples. This substantial 58.7% relative improvement demonstrates its powerful learning capability. Similarly, on GSM8k, FLEX consistently enhances the performance of already strong models, such as elevating GPT-4’s accuracy to 95.9%. These results demonstrate that our parameter-free paradigm can yield substantial gains in reasoning, an advancement previously achieved primarily through parameter-based learning methods [50, 51].

Chemistry. In the specialized domain of chemistry, FLEX proves to be highly effective at bridging the performance gap of generalist models. On the USPTO50k benchmark, baseline models exhibit limited

[Figure 7]

- Figure 4 Training dynamics and scaling laws of FLEX on the GSM8K dataset across 5 epochs. Training accuracy and test accuracy both show strong scalability with the size of the experience library. Experience library also exhibits scaling law with the epochs.

proficiency, with GPT-5 and Gemini-2.5-Pro achieving only 9.0% accuracy. By learning from a small set of 50 examples, FLEX nearly doubles these scores, increasing them to 16.0%(+7.0%) and 18.0%(+9.0%), respectively. The most notable gain is observed with Claude-Sonnet-4.5, which improves from 20.0% to 30.0% accuracy (+10.0%). These results underscore the efficacy of FLEX as a sample-efficient, parameter-free approach for adapting models to complex scientific domains.

Biology. For the ProteinGym zero-shot benchmark, the prevailing state-of-the-art Spearman’s correlation remains near 0.52. As shown in Table 1 and Figure 7, off-the-shelf large language models (LLMs) tend to underperform relative to specialized protein language models even conducting subtask of identifying and appropriately combining functionally important positions. This shortfall persists even when LLMs are supplemented with simple in-context examples, which suggests that mere exposure to a few labeled instances is insufficient to endow these models with the inductive biases required for robust mutational effect prediction. Importantly, we find that a forward-learning paradigm substantially mitigates this limitation. By extracting compact, generalizable “experience” from a small training set and encoding it as reusable rules or heuristics, the forward-learned model is able to transfer these insights to a much larger and more diverse test set yielding an average improvement of roughly 0.10 in Spearman’s ρ.

#### 4.3 The Scaling Law of Experience

To understand how experience accumulation influences model performance, we systematically analyze the scaling properties of FLEX. Our investigation reveals a predictable and principled relationship between the size of the experience library and model capability. As illustrated in Fig. 4, we identify three distinct scaling regimes through rigorous analysis across 5 training epochs on GSM8k.

Training Accuracy Scaling with Experience. Figure 4(a) shows a strong power-law relationship, with training accuracy increasing from 81.2% to 94.2% as the experience library grows from 1,001 to 1,904 items. This demonstrates substantial returns from experience accumulation, with performance gains following a predictable scaling trajectory.

Generalization Scaling with Experience. The test accuracy in Figure 4(b) exhibits consistent improvement from 81.3% to 83.3%, significantly surpassing the 80.8% baseline. The reduced variance with increasing experience indicates that accumulated knowledge stabilizes predictions and enhances reasoning robustness.

Experience Scaling with Epochs. Figure 4(c) uncovers a distinct scaling regime for experience accumulation itself. The process follows a logistic-like curve: rapid initial expansion (+576 experiences from epoch 1 to 2) transitions to selective refinement (+64 from epoch 4 to 5). This indicates an intelligent, phase-changing scaling law that efficiently covers the problem space while strategically avoiding redundancy.

These scaling laws establish FLEX as a principled framework for continuous model improvement, offering predictable performance gains through experience accumulation. The consistent scaling relationships across training and generalization metrics provide strong empirical grounding for experience-driven learning paradigms,

- Table 2 Performance comparison on cross-model memory transfer across two benchmarks (AIME25 and USPTO50k). Values in parentheses denote the absolute accuracy improvement over the ReAct baseline for each model. The models in columns represent the source of inherited experience. For AIME25, Claude and DeepSeek in columns stand for Claude-Sonnet-4 and DeepSeek-V3.1-Teminus. For USPTO50k, GPT, Gemini, and Claude in columns stand for GPT-5, Gemini-2.5-Pro, and Claude-Sonnet-4.5. For ProteinGym, Qwen and DeepSeek in columns stand for Qwen3-8B and DeepSeek-V3.1-Terminus.

AIME25 Model Agent + Claude + DeepSeek

Claude-Sonnet-4 50.0 63.3 (+13.3) 66.7 (+16.7) DeepSeek-V3.1-Terminus 60.0 66.7 (+6.7) 66.7 (+6.7)

USPTO50k Model Agent + GPT + Gemini + Claude

GPT-5 12.0 16.0 (+4.0) 18.0 (+6.0) 15.0 (+3.0) Gemini-2.5-Pro 12.0 14.0 (+2.0) 18.0 (+6.0) 23.0 (+11.0) Claude-Sonnet-4.5 23.0 28.0 (+5.0) 30.0 (+7.0) 30.0 (+7.0)

ProteinGym Model Agent + Qwen + DeepSeek

GPT-OSS 51.5 55.0 (+3.5) 56.6 (+5.1)

suggesting new pathways for model enhancement that transcend traditional scaling approaches based solely on parameter count or compute budget to the scaling with experience.

#### 4.4 Inheritance of the Experience Library

A key advantage of FLEX over parameter-based approaches is its exceptional inheritance property (Fig 6). By decoupling learned knowledge from model weights, the experience library acts as a lightweight, plug-and-play module that can be seamlessly transferred across different agents. Our experiments in Table 2 reveal two significant and economically valuable properties: the distillation of expertise from strong to weaker models, and the generality of strategies from weaker to stronger models.

First, we observe a powerful distillation effect, where expertise from a top-performing model can significantly uplift weaker ones. For instance, on USPTO50k, the experience library generated by the strongest model, Claude-Sonnet-4.5, boosts the performance of Gemini-2.5-Pro by an impressive 11 absolute points. This highlights a cost-effective pathway: the expensive exploration of a single, powerful agent can be captured and reused to enhance a fleet of less capable models without the need for individual fine-tuning. A similar trend is evident on AIME25, where the experience from the more capable DeepSeek-V3.1-Terminus model provides the largest performance gain (+16.7%) for Claude-Sonnet-4.

Even more strikingly, our results demonstrate a strong generality effect, where experience from weaker models provides substantial benefits to stronger ones. Notably, on AIME25, the experience library from the weaker Claude-Sonnet-4 elevates the stronger DeepSeek-V3.1-Terminus by 6.7 points, achieving the exact same performance level as when DeepSeek uses its own meticulously generated experience. This finding is profound: it proves that the learned experience captures fundamental, high-level strategies that are model-agnostic, rather than model-specific artifacts. The same trend is corroborated on USPTO50k and ProteinGym, where experiences from weaker models(Qwen3-8B, DeepSeek-V3.1-Terminus, GPT-5, Gemini-2.5-Pro) substantially improve the already capable Claude-Sonnet-4.5 or GPT-OSS-120B.

Taken together, these phenomena establish the experience library as a truly portable knowledge module. This opens a path towards a new paradigm: creating a single, universal experience library through a one-time

training process, or even mixing experiences from diverse sources, to provide general performance improvements across a wide ecosystem of agents.

#### 4.5 Case Study

This section presents case studies across three domains to illustrate how the learned experience library by FLEX corrects initial failures by providing critical procedural knowledge and transforms flawed reasoning into correct solutions.

Math. The distinct problem-solving trajectories in Figure 5 highlight the transformative impact of our FLEX paradigm. The naive LLM fails by decoupling geometric constraints, leading to infeasible side lengths. The standard ReAct agent, while more structured, drifts into special-case assumptions without verifying global consistency. In stark contrast, FLEX leverages its experience library to inject structured, procedural knowledge into the reasoning cycle. By retrieving a reusable algebraic template and critical feasibility checks (e.g., a2 + b2 = 1444, a,b ≤ 28), it directly addresses the core LLM limitations of logical drift and constraint violation. This distilled experience transforms the agent’s process from a heuristic-driven exploration into a deterministic, solve-verify loop, pruning invalid branches and enforcing global reconciliation. Essentially, the experience library provides the procedural scaffolds and logical guardrails that the other approaches lack, enabling the agent to systematically find the correct parameters and compute the final area of 104√3. This demonstrates how our paradigm bridges the gap between an LLM’s declarative knowledge and the effective procedural execution required for complex problem-solving.

Chemistry. The trajectories in Figure 5 reveal a critical gap in the domain-specific reasoning of standard models. Both the naive LLM and the vanilla ReAct agent make the same fundamental error: despite correctly identifying the mesylate group, they misidentify the disconnection point by prioritizing a plausible but incorrect N-alkylation pathway. This demonstrates a failure to translate declarative knowledge (what a mesylate is) into correct procedural execution (how to disconnect it). In stark contrast, FLEX succeeds by leveraging its experience library to bridge this gap. It retrieves an explicit template (R−O−SO2−CH3 → R−OH + CS(= O)(= O)Cl) that overrides the flawed heuristic and enforces the canonical O−S bond disconnection. The experience library thus functions as an external knowledge substrate, injecting proven procedural scaffolds and domain-specific constraints into the agent’s workflow. This transforms the reasoning process from speculative hypothesis testing into a deterministic, domain-grounded execution, ensuring the correct retrosynthesis is identified.

Biology. Figure 5 provides an example of how agents learn from a deliberately designed experience library—golden rules, warnings, and core know-how. Unlike the math and retro-synthesis problems above, protein-fitness prediction rarely admits absolute right/wrong answers; instead, useful guidance must be distilled from empirical trial trajectories and cross validation signals. The library captures those trajectories as reusable micro-strategies and failure patterns, enabling the agent to adapt a generic regression objective to the idiosyncrasies of a specific protein. In practice this means shifting attention away from globally renowned, aggregate metrics toward complementary, target-specific features and modeling choices that actually improve performance on the individual proteins. Moreover, the distilled tips actively drive algorithmic evolution: they suggest pre-processing pipelines, more detailed feature-combination heuristics, and validation gates that make final formula more robust. By overlaying proven procedural scaffolds onto the frozen LLM’s answer, the experience library encourages more valuable and diverse explorations and simultaneously enlarges meaningful solutions that better tuned to the peculiarities scenarios.

### 5 Discussion

The aspiration to create intelligent agents that evolve through lived experience is a long-standing and central theme in artificial intelligence [52, 53]. This discussion aims to connect our research to this enduring quest, reflecting on the potential futures it might suggest and the historical context from which it emerges. We hope

[Figure 8]

- Figure 5 Qualitative case studies in Mathematics, Chemistry, and Biology demonstrating the effectiveness of FLEX. In each domain, baseline agents (LLM Response, ReAct Response) fail due to critical reasoning errors (marked with

✗). In contrast, by retrieving and applying distilled knowledge (e.g., Golden rules and Warnings) from its experience library, FLEX successfully refines its strategy, overcomes the initial failures, and arrives at the correct solution (marked with ✓).

[Figure 9]

- Figure 6 The process of inheriting the experience library across diverse agents. The experience library acts as a plug-and-play module that can boost agents’ performance with a single training procedure.

to offer a thoughtful perspective on the path ahead, framing our contributions not as conclusions, but as a humble step in an ongoing, collective journey.

Toward Continuous Learning. A central goal in AI is to enable agents to learn continuously from their interactions, much like living organisms do [54, 55]. This pursuit hints at a future where AI systems are not static artifacts, but dynamic partners capable of adapting in real time. For years, the research community has explored various avenues toward this goal, often seeking learning paradigms that are more "forward-looking" than traditional back-propagation [56]. The semantic capabilities of modern LLMs [18–21], however, offer a new lens for this quest. It becomes possible to explore learning not only as a numerical optimization task but as a process of reasoned reflection. Our explorations in this work suggest this is a worthwhile direction, raising the possibility for a new generation of AI systems whose learning processes are inherently more auditable and aligned with a lifelong learning model.

Heading to a Collective Wisdom. Beyond individual capability, another grand challenge is fostering a form of collective intelligence, mirroring how human progress is built upon shared knowledge [57]. One can envision a future where a global network of specialized agents collaborate, forming a collective scientific mind to accelerate progress on humanity’s greatest challenges. A key historical obstacle to this vision has been the absence of a robust medium to inherit wisdom across distinct AI agents [58, 59]. The concept of decoupling learned experience from an agent’s internal parameters represents a hopeful path forward. Our explorations in this direction (Fig 6) suggest that distilled experience can indeed serve as a viable medium for inheritance, adding a small contribution to the larger effort of building a foundation upon which a more synergistic AI ecosystem might one day be built.

Learning in a transparent way. Integral to the future of AI is the development of systems whose reasoning is transparent. This points toward a future where human-AI collaboration is built on a foundation of shared understanding, which is crucial for their integration into critical societal functions. The "black box" nature of many deep learning models has presented a persistent barrier to this goal [60–62]. An experience-driven learning paradigm inherently addresses this challenge. When an agent’s growth is chronicled as a series of explicit, human-readable experiences, its evolution becomes an open book. This transparency provides a direct mechanism for meaningful human-in-the-loop interaction [63], allowing experts to understand, guide, and even enrich an agent’s learning journey. It represents a step toward a more white-box model of AI development, where the process of learning is as important as the final performance.

In conclusion, the ideas explored here are offered as a contribution to an ongoing conversation. It is our

sincere hope that this work serves as one humble effort for the field, contributing to a future where AI evolves as a more dynamic, collaborative, and transparent partner in the human endeavor.

### 6 Related Work

#### 6.1 Learning Paradigms

Supervised fine-tuning (SFT) adapts LLMs to downstream distributions via gradient-based optimization on labeled data or human-written instuctions [64–67]. It originates from the pretrain-finetune paradigm established by ULMFiT [64], BERT [65] and GPT-2 [66], and was later formalized as the first stage of alignment training in InstructGPT [68] and FLAN [69]. SFT further demonstrates remarkable progress across multiple fronts, including instruction following [68–70], problem solving [71–73], domain-specific adaptation [74–77], and so on [78–81].

Reinforcement learning (RL) further optimizes LLMs through policy gradient [82–84], aligning model behaviors with human preferences or task-specific rewards, as exemplified by RLHF [68, 85–88] and RLAIF [89, 90] Beyond alignment, RL has recently proven effective in enhancing model reasoning capabilities on challenging tasks, such as mathematical reasoning [91–94], code generation [95–97], as well as broader structured decisionmaking tasks [98–101].

Despite the effectiveness, gradient-based learning paradigms suffer from heavy computational cost and catastrophic forgetting [13, 16], and the resulting models remain static after training, unable to continually evolve through interaction with the environment [15].

Non-parametric adaptation, including prompt engineering [102] and in-context learning [43], is another paradigm without gradient-based optimization. Prompt engineering focuses on explicitly designing task instructions or linguistic cues that steer model outputs through semantic conditioning, eliciting step-by-step reasoning without parameter updates [102–107]. In-context learning adapts frozen LLMs by conditioning them on exemplars of input–output pairs or intermediate reasoning trajectories [43, 108, 109]. However, these works focus on optimizing prompt or context, failing to make model learn from the experiences through interaction with environment.

#### 6.2 Self-Evolving Agent

Recent research has begun to explore the vision of self-evolving agents [110, 111], that can autonomously improve their reasoning and decision-making through continual interaction with the environment. A first line of work focuses on tool evolution, where agents autonomously master, create, and utilize tools to extend their capabilities, such as Toolformer [112], ReAct [38], Voyager [113], CREATOR [27], and ALITA [28]. Another line emphasizes agents architecture evolution, such as CAMEL [114], MetaGPT [115], AgentSquare [26], MaAS [25], AutoFlow [116], and GPTSwarm [117], which orchestra multiple agents for cooperative problem solving. A third line focuses on context evolution, where agents continually refine their internal reasoning via reflection and semantic feedback Representative studies include Reflexion [29], Self-Refine [37], GEPA [118], SE-Agent [24], and ACE [23]. Similarly, TextGrad [22] and REVOLVE [30] interpret semantic feedback as textual gradients to guide model refinement.

Recently, experience-driven evolution [14, 119–121] has emerged, allowing LLM agents to accumulate and reuse past interaction trajectories. AgentKB [119] emphasizes cross-domain knowledge transfer through a knowledge base. Memento [120] stores raw reasoning trajectories for later retrieval, but lacks generalization and interpretive guidance for novel contexts. ReasoningBank [121] scales memory retrieval at inference time without introducing an explicit learning process. TF-GRPO [14] simply mimics the GRPO [91] procedure to compute group-relative semantic advantages, but its extensibility and generalization are constrained by the GRPO-style design.

Overall, these approaches do not establish a principled learning paradigm that formalizes how LLM agents can learn from experience in a continual, gradient-free manner. Moreover, most are validated only on

simple reasoning benchmarks, without extending to more challenging scientific domains such as chemical retrosynthesis or protein fitness prediction.

### 7 Conclusion

In this paper, we propose FLEX, a new paradigm that empowers LLM agents to continuously evolve through learning from experience. FLEX constructs an ever-evolving experience library by extensive explorations and semantic distillation of environmental interactions during the learning process, which subsequently augments the model’s capabilities of reasoning and leveraging expert knowledge without gradient back-propagation or parameter updates. Extensive experiments demonstrate that FLEX significantly enhances the performance of baselines across diverse challenging scientific tasks, including mathematical reasoning, chemical retrosynthesis, and biological protein fitness prediction. Conclusively, our work reveals the necessity of a fundamental shift in learning paradigms to address real-world challenges, thereby illuminating a promising pathway toward artificial intelligence endowed with open-ended evolution.

### Acknowledgments and Disclosure of Funding

This work is supported by the National Key R&D Program of China (2022ZD0160501).

### References

- [1] Anthropic. Claude code. https://www.claude.com/product/claude-code, 2025. Accessed: 2025-10-28.
- [2] Kexin Huang, Serena Zhang, Hanchen Wang, Yuanhao Qu, Yingzhou Lu, Yusuf Roohani, Ryan Li, Lin Qiu, Junze Zhang, Yin Di, et al. Biomni: A general-purpose biomedical ai agent. bioRxiv, pages 2025–05, 2025.
- [3] Ruofan Jin, Zaixi Zhang, Mengdi Wang, and Le Cong. Stella: Self-evolving llm agent for biomedical research,

2025. URL https://arxiv.org/abs/2507.02004.

- [4] Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Weinan E, Yuzhi Zhang, Linfeng Zhang, and Siheng Chen. Scimaster: Towards general-purpose scientific ai agents, part i. x-master as foundation: Can we lead on humanity’s last exam?, 2025. URL https://arxiv.org/abs/2507.05241.
- [5] Tongyi DeepResearch Team. Tongyi deepresearch: A new era of open-source ai researchers. https://github. com/Alibaba-NLP/DeepResearch, 2025.
- [6] OpenAI. Introducing deep research. https://openai.com/index/introducing-deep-research/, Feb 2025. Accessed: 2025-10-28.
- [7] Google. Gemini deep research — your personal research assistant. https://gemini.google/overview/ deep-research/, 2025. Accessed: 2025-10-28.
- [8] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 3(4), 2023.
- [9] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023.
- [10] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986.
- [11] Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016.
- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [13] Ian J Goodfellow, Mehdi Mirza, Da Xiao, and Courville. An empirical investigation of catastrophic forgetting in gradient-based neural networks. arXiv preprint arXiv:1312.6211, 2013.
- [14] Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, et al. Training-free group relative policy optimization. arXiv preprint arXiv:2510.08191, 2025.
- [15] Zaheer Abbas, Rosie Zhao, Joseph Modayil, Adam White, and Marlos C Machado. Loss of plasticity in continual deep reinforcement learning. In Conference on lifelong learning agents, pages 620–636. PMLR, 2023.
- [16] Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah, Muhammad Irfan, Anas Zafar, Muhammad Bilal Shaikh, Naveed Akhtar, Jia Wu, Seyedali Mirjalili, et al. Large language models: a comprehensive survey of its applications, challenges, limitations, and future prospects. Authorea preprints, 1(3):1–26, 2023.
- [17] Google DeepMind. Gemini 2.5 pro technical resport. Technical report, Google DeepMind, 2025. URL https: //storage.googleapis.com/deepmindâĂŚmedia/gemini/gemini_v2_5_report.pdf. Accessed: 2025-11-05.
- [18] OpenAI. Introducing GPT-5. https://openai.com/index/introducing-gpt-5/, Oct 2025. Accessed: 2025-1028.
- [19] Anthropic. Introducing Claude Sonnet 4.5. https://www.anthropic.com/news/claude-sonnet-4-5, Sep 2025. Accessed: 2025-10-28.
- [20] Anthropic. Introducing Claude 4. https://www.anthropic.com/news/claude-4, Jul 2025. Accessed: 2025-10-28.
- [21] xAI. Grok 4. https://x.ai/news/grok-4, Aug 2025. Accessed: 2025-10-28.
- [22] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic" differentiation" via text. arXiv preprint arXiv:2406.07496, 2024.

- [23] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618, 2025.
- [24] Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Hongzhang Liu, Ronghao Chen, Yangfan He, et al. Se-agent: Self-evolution trajectory optimization in multi-step reasoning with llm-based agents. arXiv preprint arXiv:2508.02085, 2025.
- [25] Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet. arXiv preprint arXiv:2502.04180, 2025.
- [26] Yu Shang, Yu Li, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, and Yong Li. Agentsquare: Automatic llm agent search in modular design space. arXiv preprint arXiv:2410.06153, 2024.
- [27] Cheng Qian, Chi Han, Yi R Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. Creator: Tool creation for disentangling abstract and concrete reasoning of large language models. arXiv preprint arXiv:2305.14318, 2023.
- [28] Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, et al. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. arXiv preprint arXiv:2505.20286, 2025.
- [29] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36: 8634–8652, 2023.
- [30] Peiyan Zhang, Haibo Jin, Leyang Hu, Xinnuo Li, Liying Kang, Man Luo, Yangqiu Song, and Haohan Wang. Revolve: Optimizing ai systems by tracking response evolution in textual optimization. arXiv preprint arXiv:2412.03092, 2024.
- [31] AIME. AIME problems and solutions. https://artofproblemsolving.com/wiki/index.php/AIME_Problems_ and_Solutions, 2025. Accessed: 2025-10-28.
- [32] Nadine Schneider, Nikolaus Stiefl, and Gregory A Landrum. What’s what: The (nearly) definitive guide to reaction role assignment. Journal of chemical information and modeling, 56(12):2336–2346, 2016.
- [33] Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood Van Niekerk, Steffanie Paul, Han Spinner, Nathan Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, et al. Proteingym: Large-scale benchmarks for protein fitness prediction and design. Advances in Neural Information Processing Systems, 36:64331–64379, 2023.
- [34] Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in neural information processing systems, 34:20132–20145, 2021.
- [35] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023.
- [36] Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.
- [37] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.
- [38] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.
- [39] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.
- [40] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331, 2023.

- [41] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP (1), pages 6769–6781, 2020.
- [42] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [43] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.
- [44] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv.org/abs/2210.03629.
- [45] DeepSeek. Deepseek-v3.1-terminus. https://api-docs.deepseek.com/news/news250922, 9 2022. Accessed: 2025-11-05.
- [46] OpenAI. Gpt-3.5 turbo. https://platform.openai.com/docs/models/gpt-3.5-turbo. Accessed: 2025-11-05.
- [47] OpenAI. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.
- [48] Meta AI. Llama 3.2: A new family of models for vision and on-device applications. https://ai.meta.com/blog/ llama-3-2-connect-2024-vision-edge-mobile-devices/, 9 2024. Accessed: 2025-11-05.
- [49] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925.
- [50] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.
- [51] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.
- [52] Alan M. Turing. Computing Machinery and Intelligence, pages 23–65. Springer Netherlands, Dordrecht, 2009. ISBN 978-1-4020-6710-5. doi: 10.1007/978-1-4020-6710-5_3. URL https://doi.org/10.1007/ 978-1-4020-6710-5_3.
- [53] Stuart J. Russell and Peter Norvig. Artificial Intelligence: A Modern Approach. Pearson, 4th edition, 2021. URL https://www.amazon.com/Artificial-Intelligence-A-Modern-Approach/dp/0134610997.
- [54] Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski. The alberta plan for ai research, 2023. URL https://arxiv.org/abs/2208.11173.
- [55] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application, 2024. URL https://arxiv.org/abs/2302.00487.
- [56] Geoffrey Hinton. The forward-forward algorithm: Some preliminary investigations, 2022. URL https://arxiv. org/abs/2212.13345.
- [57] Shweta Suran, Vishwajeet Pattanaik, and Dirk Draheim. Frameworks for collective intelligence: A systematic literature review. 53(1), February 2020. ISSN 0360-0300. doi: 10.1145/3368986. URL https://doi.org/10. 1145/3368986.
- [58] Murray Campbell, A.Joseph Hoane, and Feng hsiung Hsu. Deep blue. Artificial Intelligence, 134(1):57–83,

2002. ISSN 0004-3702. doi: https://doi.org/10.1016/S0004-3702(01)00129-1. URL https://www.sciencedirect. com/science/article/pii/S0004370201001291.

- [59] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. Mastering chess and shogi by self-play with a general reinforcement learning algorithm, 2017. URL https://arxiv.org/abs/1712.01815.

- [60] David Gunning, Eric Vorm, Jennifer Yunyan Wang, and Matt Turek. Darpa’s explainable ai (xai) program: A retrospective. Applied AI Letters, 2(4):e61, 2021. doi: https://doi.org/10.1002/ail2.61. URL https: //onlinelibrary.wiley.com/doi/abs/10.1002/ail2.61.
- [61] Prashant Gohel, Priyanka Singh, and Manoranjan Mohanty. Explainable ai: current status and future directions,

2021. URL https://arxiv.org/abs/2107.07045.

- [62] Weiche Hsieh, Ziqian Bi, Chuanqi Jiang, Junyu Liu, Benji Peng, Sen Zhang, Xuanhe Pan, Jiawei Xu, Jinlang Wang, Keyu Chen, Pohsun Feng, Yizhu Wen, Xinyuan Song, Tianyang Wang, Ming Liu, Junjie Yang, Ming Li, Bowen Jing, Jintao Ren, Junhao Song, Hong-Ming Tseng, Yichao Zhang, Lawrence K. Q. Yan, Qian Niu, Silin Chen, Yunze Wang, and Chia Xin Liang. A comprehensive guide to explainable ai: From classical models to llms, 2024. URL https://arxiv.org/abs/2412.00800.
- [63] Xingjiao Wu, Luwei Xiao, Yixuan Sun, Junhang Zhang, Tianlong Ma, and Liang He. A survey of human-inthe-loop for machine learning. Future Generation Computer Systems, 135:364–381, October 2022. ISSN 0167-739X. doi: 10.1016/j.future.2022.05.014. URL http://dx.doi.org/10.1016/j.future.2022.05.014.
- [64] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.
- [65] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
- [66] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [67] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [68] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [69] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.
- [70] Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021.
- [71] Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. Advances in Neural Information Processing Systems, 37:34737–34774, 2024.
- [72] Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157, 2024.
- [73] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.
- [74] Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics, 23(6):bbac409, 2022.
- [75] Di Zhang, Wei Liu, Qian Tan, Jingdan Chen, Hang Yan, Yuliang Yan, Jiatong Li, Weiran Huang, Xiangyu Yue, Wanli Ouyang, et al. Chemllm: A chemical large language model. arXiv preprint arXiv:2402.06852, 2024.
- [76] Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, and Daochen Zha. Fingpt: Democratizing internet-scale data for financial large language models. arXiv preprint arXiv:2307.10485, 2023.

- [77] Zhi Zhou, Jiang-Xin Shi, Peng-Xiao Song, Xiao-Wen Yang, Yi-Xuan Jin, Lan-Zhe Guo, and Yu-Feng Li. Lawgpt: A chinese legal knowledge-enhanced large language model. arXiv preprint arXiv:2406.04614, 2024.
- [78] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [79] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.
- [80] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023.
- [81] Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. Incoder: A generative model for code infilling and synthesis. arXiv preprint arXiv:2204.05999, 2022.
- [82] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [83] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [84] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.
- [85] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.
- [86] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
- [87] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.
- [88] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [89] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.
- [90] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint, 2023.
- [91] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [92] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [93] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

- [94] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.
- [95] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.
- [96] Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, and Chandan K Reddy. Execution-based code generation using deep reinforcement learning. arXiv preprint arXiv:2301.13816, 2023.
- [97] Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and Zibin Zheng. Rlcoder: Reinforcement learning for repository-level code completion. arXiv preprint arXiv:2407.19487, 2024.
- [98] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.
- [99] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasonerzero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.
- [100] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [101] Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, et al. A survey of reinforcement learning for large reasoning models. arXivpreprintarXiv:2509.08827, 2025.
- [102] Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and Aman Chadha. A systematic survey of prompt engineering in large language models: Techniques and applications. arXiv preprint arXiv:2402.07927, 2024.
- [103] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.
- [104] Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the few-shot paradigm. In Extended abstracts of the 2021 CHI conference on human factors in computing systems, pages 1–7, 2021.
- [105] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [106] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances inneural information processing systems, 36:11809–11822, 2023.
- [107] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 17682–17690, 2024.
- [108] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [109] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [110] Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, et al. A survey of self-evolving agents: On path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 2025.

- [111] Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, et al. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. arXiv preprint arXiv:2508.07407, 2025.
- [112] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.
- [113] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.
- [114] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008, 2023.
- [115] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2023.
- [116] Zelong Li, Shuyuan Xu, Kai Mei, Wenyue Hua, Balaji Rama, Om Raheja, Hao Wang, He Zhu, and Yongfeng Zhang. Autoflow: Automated workflow generation for large language model agents. arXiv preprint arXiv:2407.12821, 2024.
- [117] Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024.
- [118] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025.
- [119] Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.
- [120] Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, et al. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint

- arXiv:2508.16153, 2025.

[121] Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint

- arXiv:2509.25140, 2025.

- [122] Jin Su, Chenchen Han, Yuyang Zhou, Junjie Shan, Xibin Zhou, and Fajie Yuan. Saprot: Protein language modeling with structure-aware vocabulary. BioRxiv, pages 2023–10, 2023.
- [123] Céline Marquet, Julius Schlensok, Marina Abakarova, Burkhard Rost, and Elodie Laine. Expert-guided protein language models enable accurate and blazingly fast fitness prediction. Bioinformatics, 40(11):btae621, 2024.
- [124] Céline Marquet, Michael Heinzinger, Tobias Olenyi, Christian Dallago, Kyra Erckert, Michael Bernhofer, Dmitrii Nechaev, and Burkhard Rost. Embeddings from protein language models predict conservation and variant effects. Human genetics, 141(10):1629–1647, 2022.
- [125] Elodie Laine, Yasaman Karami, and Alessandra Carbone. Gemme: a simple and fast global epistatic model predicting mutational effects. Molecular biology and evolution, 36(11):2604–2619, 2019.
- [126] Timothy Truong Jr and Tristan Bepler. Poet: A generative model of protein families as sequences-of-sequences.

- Advances in Neural Information Processing Systems, 36:77379–77415, 2023.

[127] Mingchen Li, Yang Tan, Xinzhu Ma, Bozitao Zhong, Huiqun Yu, Ziyi Zhou, Wanli Ouyang, Bingxin Zhou, Pan Tan, and Liang Hong. Prosst: Protein language modeling with quantized structure and disentangled attention.

- Advances in Neural Information Processing Systems, 37:35700–35726, 2024.

- [128] Yang Tan, Ruilin Wang, Banghao Wu, Liang Hong, and Bingxin Zhou. From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model. Bioinformatics, 41 (Supplement_1):i401–i409, 2025.
- [129] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b and gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

## Appendix

### A Experimental Details

#### A.1 Biology

Protein Fitness Prediction We apply FLEX on the task of Protein Fitness Prediction. Proteins play a fundamental role in sustaining cellular functions that underpin organismal survival, growth, and reproduction. The capacity of a protein to perform its biological function—typically referred to as protein fitness—is determined by its three-dimensional structure, which is ultimately encoded by its amino acid sequence. Recent advances in machine learning have enabled zero-shot protein fitness prediction by leveraging representations derived from protein sequences, structural information, and auxiliary biological signals such as multiple sequence alignments (MSAs). Despite the progress, the task remains far from being fully solved. First, the absolute correlation between predicted and experimentally validated fitness scores remains limited (e.g., approximately 0.52 for the current state-of-the-art on ProteinGym), not confidential enough when encountering newly-found data with mysterious fitness. Second, no existing model consistently outperforms all others across diverse protein families and functional categories, including activity, expression, binding affinity, stability, and organismal fitness.

Agentic Task Transformation. Large language model (LLM) agents, without specialized pre-training on protein sequences or domain-specific fine-tuning, inherently lack detailed biochemical knowledge and cannot directly infer subtle mutational effects on protein function. However, these agents possess strong capabilities in reasoning, ranking, knowledge synthesis, and few-shot learning. This opens a new opportunity: rather than directly predicting fitness from sequence, agents can be tasked with meta-reasoning over predictions made by high-performing protein language models. Specifically, we reformulate protein fitness prediction as an agentic feature selection and fusion task, where the agent learns to identify, prioritize, and integrate salient features derived from existing regression-based models. Through this process, the agent proposes explicit, interpretable algorithms that capture the underlying relationships among predictive signals, transforming opaque model outputs into a more accurate and explainable fitness prediction framework tailored to each target protein.

Benchmark. We conduct experiments using the ProteinGym benchmark, which serves both as the source for experience extraction and as the evaluation dataset. Protein targets included in this benchmark align with those defined in SaProt [122]. To emulate realistic biological discovery scenarios where experimental data is scarce, we adopt an imbalanced data split for each protein: a small experience set consisting of only 100 sequences (on average 1.47% of the available data), and a large out-of-distribution test set containing all remaining sequences. This design reflects practical constraints in protein engineering, where only a limited number of labeled mutations are known, while the vast majority of sequence space remains unexplored.

Setup. Prediction performance is evaluated using the Spearman rank correlation between the ground truth DMS (deep mutational scanning) scores and the proxy metrics generated by our FLEX method. Baseline models are grouped into two main categories:

Zero-shot Protein Fitness Prediction Models. These models represent the current state-of-the-art on the ProteinGym leader-board and provide diverse biological priors through different input modalities:

- • VespaG [123, 124]: A top-performing single-sequence model leveraging PLM embeddings and evolutionary guidance from GEMME [125].
- • PoET [126]: A homolog-based model that encodes multiple sequence alignments using a sequence-ofsequences transformer architecture.
- • ProSST [127]: A hybrid model incorporating structural quantization and disentangled attention mechanisms to jointly model sequence and structural representations.

[Figure 10]

Figure 7 ProteinGym Results from both protein language model and large language model.

Table 3 Ablation study results in ProteinGym benchmark of forward learning paradigm component techniques: Experience Exploration, Experience Evolution, Assistance of Regression tools. Where Fixed mean best one for all regression tools is used instead of each different regression tools for protein target selected by agent.

Model Experience Exploration Experience Evolution Regression Tools Spearman’s ρ

- ✗ ✗ ✗ 0.472
- ✗ ✗ Fixed 0.531

✓ ✗ ✗ 0.537

- ✗ ✓ ✗ 0.547

GPT-OSS

✓ ✓ ✗ 0.573 ✓ ✓ Fixed 0.568 ✓ ✓ ✓ 0.581

• VenusREM [128]: The current state-of-the-art on the ProteinGym scoreboard, integrating logits from single-sequence, homology, and structural tokenization signals.

These models provide the input features that our agent uses to reason about and construct enhanced fitness metrics.

Native Large Language Models (LLMs). To isolate the contribution of agentic reasoning, we evaluate several high-performance LLMs—such as GPT-OSS [129] and Claude-Sonnet-4.5 as direct baselines. These models are prompted to perform the same prediction task as our forward learning system but without employing experience validation, rejection sampling and refinement. This comparison highlights the added value of structured forward learning beyond raw language model capabilities.

Ablation Study. From Table 3 we can learn that, three core parts of our FLEX paradigm contributed to

performance enhancement: Experience Exploration, Experience Evolution, Regression Tools. Specificlly, Experience Exploration provide success and failure cases agent used to engage to, in the ProteinGym Scenario, common features included in most successful cases suggest its effect-ness in prediction final proxy metric. Experience Evolution enables agent to summarize past experience, update lessons learned recently with new cross-validation facts. Moreover, based on accurate selection of top importance features, regression tools could help agents to further adjust limited hyper-parameters, where deciding what kinds of regression works best still important for agents compared to the fix, one for all regression cases.

