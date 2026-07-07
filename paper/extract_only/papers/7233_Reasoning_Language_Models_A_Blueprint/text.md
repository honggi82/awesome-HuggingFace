# Reasoning Language Models: A Blueprint

Maciej Besta1†, Julia Barth1, Eric Schreiber1, Ales Kubicek1, Afonso Catarino1, Robert Gerstenberger1, Piotr Nyczyk2, Patrick Iff1, Yueling Li3, Sam Houliston1, Tomasz Sternal1, Marcin Copik1, Grzegorz Kwasniewski´ 1, J¨urgen M¨uller3, Łukasz Flis4, Hannes Eberhard1, Zixuan Chen1, Hubert Niewiadomski2, Torsten Hoefler1

†Corresponding author 1ETH Zurich 2Cledar 3BASF SE 4Cyfronet AGH

## arXiv:2501.11223v4[cs.AI]11Jun2025

Abstract—Reasoning language models (RLMs), also known as Large Reasoning Models (LRMs), such as OpenAI’s o1 and o3, DeepSeek-R1, and Alibaba’s QwQ, have redefined AI’s problem-solving capabilities by extending large language models (LLMs) with advanced reasoning mechanisms. Yet, their high costs, proprietary nature, and complex architectures—uniquely combining reinforcement learning (RL), search heuristics, and LLMs—present accessibility and scalability challenges. To address these, we propose a comprehensive blueprint that organizes RLM components into a modular framework, based on a survey and analysis of all RLM works. This blueprint incorporates diverse reasoning structures (chains, trees, graphs, and nested forms), reasoning strategies (e.g., Monte Carlo Tree Search, Beam Search), RL concepts (policy, value models and others), supervision schemes (Outcome-Based and Process-Based Supervision), and other related concepts (e.g., Test-Time Compute, Retrieval-Augmented Generation, agent tools). We also provide detailed mathematical formulations and algorithmic specifications to simplify RLM implementation. By showing how schemes like LLaMA-Berry, QwQ, Journey Learning, and Graph of Thoughts fit as special cases, we demonstrate the blueprint’s versatility and unifying potential. To illustrate its utility, we introduce x1, a modular implementation for rapid RLM prototyping and experimentation. Using x1 and a literature review, we provide key insights, such as multi-phase training for policy and value models, and the importance of familiar training distributions. Finally, we discuss scalable RLM cloud deployments and we outline how RLMs can integrate with a broader LLM ecosystem. Our work demystifies RLM construction, democratizes advanced reasoning capabilities, and fosters innovation, aiming to mitigate the gap between “rich AI” and “poor AI” by lowering barriers to RLM design and experimentation.

Index Terms—Reasoning Language Model, Large Reasoning Model, Survey of Reasoning Language Models, Survey of RLMs, RLM, LRM, Reasoning LLMs, Reinforcement Learning for LLMs, MCTS for LLMs, Large Language Model, LLM, Generative AI.

✦

###### 1 INTRODUCTION

Reasoning Language Models (RLMs), such as OpenAI’s o1 [120], o3 [79], DeepSeek-R1 [92] and Alibaba’s QwQ [152], also referred to as Large Reasoning Models (LRMs)1, represent a transformative breakthrough in AI, on par with the advent of ChatGPT [118]. These advanced systems have fundamentally redefined AI’s problem-solving capabilities, enabling nuanced reasoning, improved contextual understanding, and robust decision-making across a wide array of domains, reshaping science [47], industries [23], governance [53], and numerous other aspects of

1We use the term “Reasoning Language Model” instead of “Large Reasoning Model” because the latter implies that such models are always large. This does not necessarily have to be the case – as a matter of fact, smaller RLM can outperform larger LLMs [57].

human life [48], [78], [83], [147], [148]. By extending the capabilities of standard large language models (LLMs) with sophisticated reasoning mechanisms, RLMs have emerged as the new cornerstone of cutting-edge AI, bringing us closer to AGI.

However, the high cost and proprietary nature of stateof-the-art RLMs, such as those developed by OpenAI, risk exacerbating the divide between “rich AI” and “poor AI”, raising significant concerns about accessibility and equity. Even the publicly available QwQ only comes with its model weights, and Alibaba does not disclose details about their training or data generation methodologies. Businesses and individuals unable to afford these advanced systems face a growing disadvantage, threatening to stifle innovation and reinforce systemic inequities. As RLMs become integral to critical applications, from healthcare to science, manage-

###### Reasoning Language Model (RLM): What is it and how to build one? Basics of RLMs

###### Essence of RLMs Blueprint of RLMs

###### x1 Framework & Insights

- §4 Blueprint: a toolbox with ingredients to build various RLMs

Appendices C-D Algorithmic formula ons of RLMs: how diﬀerent parts of RLMs work in detail, facilita ng implementa on

Appendix B Details on value and reward models

Appendix A Mathema cal speciﬁca ons of RLMs

- §5 How exis ng schemes compare to the blueprint

- §7.5 Enabling eﬃcient scaling, modern cloud deployments

- §7.6 Example analyses

§8 Example insights for building eﬀec ve RLMs

- §6 Hints on how to use the blueprint for user's applica on

§3 Essence of RLMs: an overview and the most important details of the RLM architecture

§7 Design of the x1 framework: how to easily implement and experiment with RLM designs

§2.1-§2.2 History & main pillars of RLMs

§2.3-§2.4 Diﬀerent categories of RLMs

Fig. 4 An overview and details of the inference, training, and data genera on pipelines of RLMs

- Fig. 2 History of RLMs

- Fig. 3 Pillars and categories of RLMs

Fig. 5 Toolbox overview TABLE 1 RLM comparison

§9 Benchmarks for RLMs

Fig. 1: Summary of the contributions made by this paper. The x1 framework can be found at https://github.com/spcl/x1.

ment, and beyond, it is imperative to address these disparities and ensure that the benefits of advanced reasoning capabilities are broadly accessible.

The technical foundations of RLMs remain opaque and complex, compounding the accessibility challenge. Emerging analyses suggest that their design likely integrates elements such as Monte Carlo Tree Search (MCTS) or Beam Search, reinforcement learning (RL), Process-Based Supervision (PBS) [91], [91], [155], [155], and advanced in-context learning (ICL) techniques like Chain-of-Thought (CoT) [165] or Tree of Thoughts (ToT) [174], and possibly even retrievalaugmented generation (RAG) [15], [59], [86], [87].

Additionally, these architectures employ multiple specialized subcomponents—such as synthetic data generation engines and policy, value, and reward models—trained through some form of novel loss functions and possibly several fine-tuning schemes. However, the intricate interplay of these components and their integration into a cohesive and effective architecture remains poorly understood. Here, the “holy-grail question” is: what is the detailed design of an RLM and how to make it simultaneously achieve effectiveness (i.e., high accuracy in delivered answers), low cost, and scalability?

To help answer this question and to address the above challenges, we propose a comprehensive blueprint for constructing, analyzing, and experimenting with RLMs (contribution #1; a roadmap of all the contributions and the paper is in Figure 1). Our approach identifies and crystallizes the fundamental building blocks of RLMs, organizing them into a cohesive framework. This blueprint is presented with increasing levels of granularity, starting from highlevel overview, finishing at low-level details that can be directly harnessed when implementing. Further, to maximize the clarity and comprehensiveness, we present the blueprint using three perspectives: (1) architecture diagrams and descriptions, (2) detailed mathematical formulations, and (3) in-depth algorithmic specifications. By employing these complementary perspectives, we aim to provide a clear and actionable guide for developing RLMs tailored to specific applications, settings, and constraints.

Our blueprint comprehensively encompasses the potential building blocks of RLMs, offering a flexible and modular framework. It incorporates a variety of reasoning structures, such as chains, trees, graphs, and even higher-order structures such as hierarchical (or nested) trees, along with numerous operations that transform and advance the reasoning process. The blueprint supports different granularities of reasoning steps, ranging from individual tokens to full sentences or structured segments. Additionally, it enables diverse training schemes, including Outcome-Based Supervision (OBS) and PBS, and the related Outcome & Process Reward Models (ORMs & PRMs). Next, in order to illustrate the capability of the blueprint to accommodate novel design ideas, we describe several novel schemes and how they fit within the blueprint. One such example is Trace-Based Supervision (TBS), which extends PBS by incorporating labeled traces of traversal paths through entire reasoning structures, rather than just linear chains of reasoning steps. By unifying all these components, our blueprint serves as a versatile toolbox for constructing RLMs—ranging from

simple models to sophisticated designs—tailored to specific reasoning tasks and performance objectives.

We conduct a broad analysis of existing reasoning schemes (contribution #2), demonstrating how they fit into our blueprint as special cases. This analysis encompasses not only standard MCTS and reinforcement learning-based designs, such as LLaMA-Berry [182], but also models like QwQ [152]. Additionally, we include paradigms diverging from standard MCTS, such as Journey Learning [123] or Beam Search, which redefines reasoning through implicit long-chain structures, and advanced structured prompting techniques like CoT [165], ToT [174], and Graph of Thoughts [11]. We also consider reasoning utilities such as Retrieval-Augmented Generation (RAG) and data stores, tools, and others. By mapping these diverse approaches to one blueprint, we showcase its versatility and expressive power, highlighting its ability to unify a wide range of reasoning methodologies within a coherent framework.

To demonstrate the utility of our framework, we introduce x1, a modular and user-friendly implementation2 designed to simplify the process of developing and experimenting with new RLM architectures, covering not only training and inference, but also synthetic data generation (contribution #3). We design x1 to facilitate supporting various optimizations, design decisions, and overall scalability, such as batch processing, making it a well-suited foundation of experimentation infrastructure. We also discuss key aspects of deployment in cloud environments, ensuring that x1 can be seamlessly integrated into modern infrastructure for both research and production use cases.

By providing both theoretical insights and practical tools, this work aims to democratize access to advanced RLMs, enabling researchers and practitioners to design, train, and deploy sophisticated reasoning models with reduced complexity and cost. Our blueprint offers a clear and adaptable framework that lowers the barriers to entry, fostering broader experimentation and innovation. Additionally, the modular implementation of x1 serves as a foundation for rapid prototyping and large-scale experimentation, empowering users to explore new reasoning paradigms and optimize performance across diverse applications. By bridging the gap between conceptual advancements and practical implementations, this work seeks to accelerate progress in the field, unlock new possibilities for intelligent systems across research, industry, and education, and to mitigate the risk of the growing gap between “rich AI” and “poor AI”.

###### 2 EVOLUTION & FOUNDATIONS OF RLMS

We first summarize the evolution and foundations of reasoning language models. Figure 2 shows an overview of the history of the development of these models.

###### 2.1 Basic Pillars of Reasoning LMs

The development of reasoning-capable LLMs represents a convergence of three critical threads: (1) advances in LLMs such as GPT-4, (2) RL designs such as AlphaZero, and (3)

2https://github.com/spcl/x1

AI Supercomputers

Legend: start of an era Supercomputer CPU/GPU Model

Start of the Petascale era

H100 Helios A100

[Figure 1]

[Figure 2]

#### HPC

P100

Titan

GH200

GB200

V100

Piz Daint

Start of the Exascale era

Tianhe-2

Alps

Breakthroughs in compute resources enabled the introduc on of LLMs

Breakthroughs in compute resources enabled breakthroughs in RL models

Breakthroughs in compute resources enabled the introduc on of RLMs

The ongoing growth of compute power and data processing capabili es of supercomputers and high performance systems, previously driven by Moore's law and now by the massively parallel processing capabili es of GPUs, TPUs, and AI accelerators.

AlphaZero

Deep Q-Network

OpenAI Five AlphaFold

#### RL

AlphaGo

Value model (neural network)

Policy model (neural network)

the branching factor is identical for all nodes

AlphaZero

DreamerV3

MuZero

π π π π

v=0.09

v=0.11 v=0.04

v=0.001

RL models for board games become a pillar of RLMs

π π π π

π

LLM

v=0.01 v=0.05 v=0.12 v=0.08

v=0.01

LLaMA-3

Chat-GPT

Claude GPT-2

MCTS samples mul ple tree searches to some depth and propagates ﬁnal values up the path, which keeps sta s cs for each (state, ac on)-pair (edge). At the end, it chooses the most promising ac on from the root and prepares the next move.

#### LLM

Transformer GPT-3

LaMDA PaLM

LLM

LLaMA

GPT-4o

GPT-4

RLM

Transformer

LLMs become a pillar of RLMs

The policy model is an LLM that is ﬁnetuned with a special loss func on to generate the best subsequent reasoning steps.

The value model is an LLM that replaces the ﬁnal token output layer with a regression to a value.

Policy model

Value model

autoregressive token generation

Transformer

Transformer

o1

QwQ

Sort the numbers "3,2,4,5,6,12,5,6"

#### RLM

autoregressive token generation

autoregressive token generation

x1

π π π π

TS-LLM: the ﬁrst proposal to use AlphaZero-like tree search to enhance LLM's training & decoding

DeepSeek

o3

###### Look up Quicksort Sor ng is simple Split into two sets

Numbers are blue

π π π π

v=0.09

v=0.11 v=0.04

v=0.001

π

Quicksort sorts Split into two sets "3,4" & "5,2" numbers Pick Pivot "3,2,4,5" & "7,12,5,6"

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |

2010 2015 2020 2025

v=0.08 v=0.01 v=0.05 v=0.12

v=0.01

Fig. 2: The history of RLMs. This class of models has been the result of the development of three lines of works: (1) reinforcement learning based models such as AlphaZero [138], (2) LLM and Transformer based models such as GPT-4o [119], and (3) the continuous growth of compute power and data processing capabilities of supercomputers and high performance systems.

High-Performance Computing (HPC) resources. Together, these threads have shaped models capable of efficient System 2 Thinking – a level of reasoning that combines explicit deliberation with novel problem-solving abilities, distinct from the intuitive, fast, and automatic heuristics of System 1 Thinking. Figure 2 compares example designs in these pillars while Figure 3 (left side) further discusses the details of these pillars.

2.1.1 Large Language Models: A Reservoir of Knowledge

LLMs such as GPT-4o [119] or LLaMA [55] represent an extraordinary leap in the field of AI, constituting a vast repository of world knowledge encoded directly in their weights. Trained on huge corpora of text from diverse sources, LLMs are capable of understanding and generating human language with remarkable fluency. However, their reasoning abilities largely align with the fast, automatic, and intuitive System 1 Thinking. While they can generate coherent responses and even perform simple reasoning tasks, LLMs have limitations. The reasoning they exhibit is often shallow, rooted in the simple mechanism of predicting the next most probable token in a sequence rather than engaging in explicit problem-solving or structured analysis. While LLMs may generate plausible-sounding solutions to a problem, these outputs are the result of statistical language modeling rather than a deliberate, iterative reasoning process. This distinction highlights the need for integrating more advanced mechanisms capable of explicit reasoning into AI systems—paving the way for hybrid designs that combine

the knowledge-rich foundation of LLMs with structured reasoning methodologies.

2.1.2 Reinforcement Learning: Exploring and Innovating

RL has historically provided a framework for decisionmaking and exploration in environments where an agent must learn optimal strategies through trial and error. Landmark systems like AlphaZero [138] and a long line of others such as AlphaGo [137] or MuZero [134] demonstrated the profound potential of RL by achieving superhuman performance in games such as chess, shogi, and Go. Unlike traditional AI systems, AlphaZero began with no embedded domain knowledge. Instead, it mastered these games purely through self-learning, discovering novel strategies that even human experts had not considered.

One of the most striking examples of RL’s innovative capacity came during an AlphaZero match, where the system made a move initially deemed a mistake by human observers. This move [109] later proved to be both surprising and strategically brilliant, illustrating the capacity of RL agents to explore unconventional solutions that lie outside the bounds of human intuition. Such capabilities are fundamentally rooted in RL’s ability to navigate vast search spaces effectively.

However, traditional RL systems lacked the ability to encode real-world knowledge or handle complex, multifaceted reasoning tasks. This limitation spurred the integration of RL principles with LLMs, combining the structured exploration and optimization capabilities of RL with the knowledge-rich reasoning foundation of language models.

###### Hierarchy of Language Models

###### Three Pillars of Reasoning Language Models (RLMs)

See §2.2

###### Language Models (LMs)

###### Reasoning Language Models (RLMs)

RLMs surpass the capabili es of LLMs and RL agents

###### Large Language Models (LLMs)

Reasoning Language Models (RLMs)

Examples: OpenAI o1, OpenAI o3, QwQ, DeepSeek-R1

###### See §2.1.1 See §2.2

See §2.1.1

See §2.1.2

See §2.1.3

Capable of System 1 Thinking; can do Interpola on (see §2.3)

Capable of System 2 Thinking; be er at Extrapola on (see §2.3)

Pillar 1: Large Language Models (LLMs)

Pillar 2: Reinforcement Learning (RL)

Pillar 3: High-Performance Compu ng (HPC)

Examples: o1, o3, DeepSeek-R1, QwQ

Examples: GPT-4o, LLaMA, Qwen

Strengths: Finds op mal strategies for decision-making problems by trial and error; no embedded domain knowledge needed.

Strengths: Ongoing growth of compute and data processing capabili es enables con nuous increase in models' capabili es

Strengths: Understanding and genera ng human language

Explicit RLMs (see §2.4.2)

Implicit RLMs

(see §2.4.1)

Limita ons: End of Moore's Law and Dennard's Scaling requires more elaborate designs to drive the growth of compute capabili es

Limita ons: Lacks the ability to encode real-world knowledge or handle mul -faceted reasoning tasks

Limita ons: Lack of a deliberate, itera ve reasoning process

Explicit structure of reasoning (e.g., MCTS) and the strategy for its evolving, e.g., a strategy based on reinforcement learning analogous to

LLM

Value model

Policy model

Structure encoded in weights (matrices)

Transformer

Transformer

Transformer

Examples: AlphaZero, AlphaGo, MuZero

Examples: GPT-4o, LLaMA, Qwen, Grok

Example AI-focused architectures: CSCS Alps, Cyfronet Helios

AlphaZero Training

autoregressive token generation

autoregressive token generation

autoregressive token generation

Previously driven by Moore's law and now by the massively parallel processing capabili es of GPUs, TPUs, and AI accelerators, HPC is the founda on of LLMs, RL, and RLMs.

Examples: LLaMA-Berry, Marco-o1

Example: QwQ

Fig. 3: Hierarchy of language models (right) and the three pillars of RLMs (left).

- 2.1.3 HPC: Scalability & Efficiency The growth of LLM and RL systems has been propelled by advancements in High-Performance Computing (HPC). Initially driven by Moore’s Law, which enabled a doubling of transistor density approximately every two years, HPC benefited from both technological advancements and the economic feasibility of manufacturing smaller transistors. However, as the costs of further miniaturization have risen sharply, Moore’s Law has reached practical limits, necessitating alternative strategies like parallelism and heterogeneous computing.

Modern HPC systems rely heavily on GPUs, TPUs, and AI accelerators for their parallel processing capabilities, alongside CPUs for sequential and general-purpose tasks. Heterogeneous computing leverages these components to optimize task-specific performance. Distributed frameworks, employing techniques such as data, model, and pipeline parallelism [10], [14], [18], further enable the training of enormous models across thousands of compute nodes.

Energy efficiency innovations, including sparsity, quantization, and pruning, mitigate the growing energy demands of scaling AI systems. These advancements ensure that HPC remains a cornerstone for developing and deploying AI models, supporting the combination of vast knowledge, reasoning capabilities, and computational scalability – allowing AI evolution to continue beyond the limits of traditional Moore’s Law scaling.

###### 2.2 The Convergence: System 2 Thinking in AI

The intersection of these three threads – LLMs, RL, and HPC – has culminated in the emergence of models capable of System 2 Thinking. These advanced systems combine the knowledge-rich foundation of LLMs with the exploratory and optimization capabilities of RL, all supported by the scalability and performance of modern HPC. The result is a new class of AI models that can engage in explicit, deliberate reasoning processes.

These models possess a world model encoded in the weights of their LLM components, allowing them to reason

about complex scenarios and contexts. Their RL capabilities combined with the HPC capabilities enable them to navigate truly immense decision spaces, evaluate multiple strategies, and iteratively refine solutions.

###### 2.3 Interpolation (LLMs) vs. Extrapolation (RLMs)

Standard LLMs, driven by their autoregressive token prediction mechanism, primarily perform interpolation within the vast search space of solutions. They excel at generating responses that align with patterns seen in their training data, effectively synthesizing knowledge from known contexts. However, this process limits them to producing outputs that remain within the boundaries of their training distribution. In contrast, reasoning LMs enable extrapolation beyond these boundaries. By combining structured exploration, reasoning LMs navigate uncharted areas of the solution space, generating novel insights and solutions that extend past the limits of their training data. This enables a shift from basic pattern completion to active problem-solving.

###### 2.4 Hierarchy of Reasoning-Related Models

The evolution of RLMs can be understood as a hierarchical progression, with earlier models such as GPT-4o being less capable in terms of reasoning, and the o1-like architectures demonstrating increasing sophistication and explicit reasoning abilities. This hierarchy reflects the integration of System 1 (LLMs) and System 2 (RLMs) Thinking. RLMs can be further divided based on how reasoning is implemented into Implicit RLMs and Explicit RLMs; the details of this categorization can be found in Figure 3 (the right side).

2.4.1 Implicit Reasoning Models

In this subclass, the reasoning structure is embedded entirely within the model’s weights. Models such as QwQ [152] operate as “black boxes”, where reasoning is implicit and cannot be explicitly disentangled or manipulated. While these models exhibit improved reasoning capabilities compared to standard LLMs, their reasoning processes are opaque and rely on the internalized patterns learned during training.

- 2.4.2 Explicit Reasoning Models

These models introduce explicit reasoning mechanisms external to the model’s core weights. Examples include designs such as LLaMA-Berry [182], Marco-o1 [187], and potentially OpenAI’s o3, which incorporate mechanisms like explicit MCTS combined with RL for decision-making. This explicit structure enables the model to simulate, evaluate, and refine solutions iteratively, facilitating novel problemsolving and extrapolation. By separating reasoning from the static knowledge encoded in the weights, these models achieve greater flexibility and interpretability in their reasoning processes. Note that the explicit reasoning can be internalized via training making it implicit – we discuss it later in the blueprint.

- 3 ESSENCE OF REASONING LMS

We now describe the general architecture of RLMs, which we summarize in Figure 4. In the following sections, we generalize this description to the full RLM blueprint.

- 3.1 Basic Architecture, Pipelines, & Concepts

We now outline the foundational architecture, operational pipelines, and core concepts. Figure 4 offers three levels of detail. In general (the top-left part), the whole RLM architecture consists of three main pipelines: inference, training, and data generation. The inference serves user requests, using models (e.g., the value or policy model) provided by the training pipeline. Data generation mirrors the inference pipeline in its internal design; the main difference is that it runs independently of the user requests, generating data that is then used to re-train the models. As such, training combined with data generation from various domains [131], [181] offers self-learning capabilities and is analogous to the self-play setting of AlphaZero [138].

- 3.1.1 Inference The inference process begins when the user provides an

input prompt 1 , which typically describes the problem or question to be addressed by the RLM. This input serves as the root of the reasoning process and initiates the construction of a reasoning structure 2 that organizes RLM’s progress. The structure is usually represented as a tree. The root of this tree corresponds to the user’s input, and subsequent nodes are generated to explore the search space – the domain of possible reasoning paths or solutions. The purpose of this reasoning structure is to systematically investigate potential solutions, progressively refining and extending reasoning paths to converge on an optimal or satisfactory answer.

An individual point in the search space, represented as a node in the reasoning structure, corresponds to a reasoning step 3 . A reasoning step is defined as a coherent and self-contained unit of thought – a sequence of tokens that advances the solution by either exploring a new branch of the problem or building upon existing progress. These steps form the building blocks of the reasoning process.

The details of how the structure evolves are usually governed by the MCTS scheme, enhanced with policy and value models (we also distinguish other reasoning

strategies, described below). This approach, inspired by methods used in AlphaZero, ensures that the search process is both efficient and directed toward promising solutions. The policy model 4 is responsible for generating new reasoning steps at each node, predicting the next most likely and logical steps to expand the reasoning process. Meanwhile, the value model 5 evaluates the quality of a reasoning path starting at a given node, helping the system prioritize the most promising steps to follow. Sometimes, a reward model3 6 is used instead, to assess the quality of an individual specific node and its corresponding reasoning step. In our blueprint, as detailed in the next section, we abstract the models into a more general notion of operators 7 to enable more flexibility in how they are implemented.

The search and reasoning processes continue iteratively until a terminal step is reached 8 . This terminal step represents a completion of the reasoning chain that forms the final answer to the posed problem. It serves as the leaf node in the tree, concluding that particular reasoning path.

This architecture provides a unified framework that accommodates a wide range of reasoning tasks. Whether reasoning steps are fine-grained (e.g., individual token sequences) or coarse-grained (e.g., entire reasoning chains treated as single nodes), the architecture adapts seamlessly. By structuring the search space explicitly and guiding exploration with policy and value models, the RLM achieves a level of reasoning capability bridging intuitive pattern recognition and deliberate problem-solving.

A detailed specification of the inference pipeline can be found in Appendix C.1 and in Algorithm 1.

3.1.2 Training

Training details depend on what model is trained (value, policy, reward, ...). In general, we assume fine-tuning a model such as LLaMA. Here, we follow an approach where one first harnesses supervised data, usually coming from existing datasets such as PRM800K [91] 1 , which becomes a part of the supervised training data 2 used in the supervised training pipeline 3 of the framework to train some, or all, of the models 4 considered in the blueprint. The second part of the overall training framework in RLMs is the unsupervised (self-learning) training pipeline, in which training data is being continually generated 5 and used to improve the models. The data can be obtained from inference, assuming quality control [57], but also from a dedicated synthetic data generation pipeline that mirrors that of the inference. To collect the data, one executes the respective RLM pipeline for a given input task and gathers the results 6 ; depending on how detailed the gathering process is, the data collected can contain only outcomebased labels 7 , process-based labels 8 , or some other variant such as trace-based labels 9 suggested in our blueprint, that generalize process-based samples to samples that contain also information about operators applied during the task solution process. All this data becomes a part of the replay buffer 10 and is used in the unsupervised training

3We use a naming scheme in which a model used to estimate the quality of a whole reasoning path starting at a given node, is called the value model, while a model used to estimate the quality of a given reasoning step, is called the reward model.

###### Legend

Medium-level overview (§3.1)

References to descrip ons in text (inference pipeline)

1

One can use external data such as human-prepared chains of thoughts

Part of the pipeline

Models & Operators

Training Data

Reasoning scheme

External sources

References to descrip ons in text (training pipelines)

user

provide data

###### Explicit RLM Inference

###### Implicit RLM

Reasoning Scheme (§4.2.2)

###### High-level overview (§3.1)

executes uses executes uses

...

Training Data

###### user

Models are used to run inference New self-learning data is generated and used by training

###### Data ... Genera on

...

Inference

Training

Reasoning Scheme

...

uses uses

· Policy model · Value model

generates

Inference uses reasoning scheme

...

Reasoning u li es

###### More details

uses uses

Self-Learning

###### Training

...

...

· Supervised

ﬁne-tuning data · Replay buﬀer

...

Data Genera on

trains

Models (§4.3 - §4.4)

value model, policy model, ...

Data genera on uses reasoning scheme

becomes

New self-learning data is generated and used by training Use updated models and buﬀer for data genera on

One can train an Implicit RLM during the execu on of the Explicit RLM pipelines, for example by training the model on the execu on traces from the Explicit RLM

More details

Detailed view (§3.1.1, §3.1.2)

External sources

user

provide data

- 1

1

- 2

1

Implicit Explicit RLM RLM

###### 13

Reasoning Strategy

Reasoning Scheme

Reasoning Structure

Reasoning Scheme

= +

###### Inference

executes

Input

Sort the numbers "3,2,4,5,6,12,5,6" Split into two sets

Numbers are blue

(can also be used for data genera on )

5

3

v=6

v=3

v=9 v=2

uses

uses

Supervised ﬁne-tuning

Look up Quicksort

Sor ng is simple

"3,4" & "5,2"

Instance of Reasoning Structure

Pick Pivot

data

v=3 v=5

v=1 v=2

v=8

2

Split into Quicksort two sets

5

"3,2,4,5" & "7,12,5,6"

sorts nubers

Data Genera on

v=3 v=2

executes

"3,2,4" & "5,2,6"

"3,2,4,5" & "6,12,5,6"

8

Output

"2,3,4,5,5,6,6,12"

Training Data

uses

uses

Data collec on

6

- 7

- 8

- 9

Samples for Outcome-Based Supervision

Reasoning u li es

is included

"2,3,4,5,5,6,6,12"

Sort the numbers "3,2,4,5,6,12,5,6"

into

Samples for Process-Based Supervision

Tools

Look up Quicksort Split into two sets "3,2,4" & "5,2,6" "2,3,4,5,5,6,6,12"

Sort the numbers "3,2,4,5,6,12,5,6"

Unsupervised ﬁne-tuning data

is included

into

Databases, RAG

Web access

Samples for Trace-Based Supervision

is included

(replay buﬀer)

10

into

Sort the numbers "3,2,4,5,6,12,5,6" Generate

Agents

Generate

Evaluate Backtrack

Numbers are blue

Coding on-the-ﬂy

"2,3,4,5,5,6,6,12"

Look up Quicksort

Select

...

uses

uses uses

Training

Unsupervised (self-learning) fine-tuning

uses

Supervised fine-tuning Implicit RLM training

3

11 12

trains

trains trains

trains trains

trains

4

Models (§4.4)

Policy model Value model Reward model

6 Implicit RLM

4 5

13

implements implements implements

Operators (§4.3) Generate Refine Evaluate Backtrack Select Prune

7

- Fig. 4: Overview of a general RLM design and core concepts. We provide a high-level overview (the top-left part), a more detailed medium-level overview (the top-right part), and a very detailed diagram showing the inference and training pipelines (the bottom part). A detailed specification of the inference pipeline can be found in Appendix C.1 and in Algorithm 1. Details on the pipelines for different training phases and paradigms can be found in Appendices C.2 and C.3 as well as in Algorithms 2–7. The data generation pipeline is detailed in Appendix D.

scheme 11 or it can also be used to train 12 a model that would become an Implicit RLM 13.

A detailed specification of the pipelines for different training phases and paradigms can be found in Appendices C.2 and C.3 as well as in Algorithms 2–7. The data generation pipeline is detailed in Appendix D.

###### 3.2 Encompassing Diverse RLM Architectures

The above-described design is applicable to many RLM designs. However, there are numerous other variants of architectures, some of which do not fully conform to this framework. In this section, we discuss these variants, highlighting how our blueprint accommodates such variations.

In some RLM designs [182], a single node in the MCTS tree could represent an entire reasoning structure, such as a complete chain of reasoning steps. In this case, the action space involves transitioning between different reasoning structures rather than individual steps. This approach changes the nature of the search, as the focus shifts from iteratively constructing a single reasoning path to evaluating and refining entire structures within the search space. Our blueprint accommodates this with the concept of nesting, where a node in the reasoning structure can contain another reasoning structure.

Other architectures introduce even more novel paradigms. For instance, Journey Learning [123] adds an additional layer of complexity by incorporating a transformation step that “rewires” the search or reasoning structure. This transformation consolidates multiple paths in the tree, synthesizing them into a new form that is used as input for subsequent reasoning iterations.

Despite these variations, our blueprint is sufficiently general to encompass all these cases and beyond, as we illustrate more formally in the following. This generality ensures that the blueprint is not only applicable to existing designs but also provides a foundation for future innovations in RLM development.

###### 3.3 Integration with Broader LLM Agent Ecosystems

The integration of RLMs into broader LLM agent ecosystems would enable these models to interact dynamically with external tools, databases, and resources during execution. This interaction can occur within the inference or data generation pipeline, leveraging value or policy models to extend the reasoning process through access to retrievalaugmented generation (RAG), web queries, and specialized tools. For example, during a reasoning task, the value or the reward model could query a database to verify intermediate steps, ensuring factual correctness or retrieving additional context to refine its reasoning. Similarly, these models could utilize computational tools for mathematical or symbolic computations, thereby expanding the scope and accuracy of their reasoning.

###### 4 BLUEPRINT FOR REASONING LMS

We now introduce our RLM blueprint that can be used to develop novel reasoning models and to provide ground for analysis, evaluation, and comparison of such designs. We overview the blueprint in Figure 5.

###### 4.1 Overview & Main Components

The blueprint specifies a toolbox of components that can be used to build an arbitrary RLM. We identify several classes of such components. First, an RLM includes a reasoning scheme, which specifies a reasoning structure (e.g., a tree) together with a reasoning strategy (e.g., MCTS) of how this structure evolves in order to solve a given input task. Second, there is a set of operators (e.g., refine) that can be applied to the reasoning structure (as specified by the reasoning strategy) in order to evolve it and make progress towards solving the input task. Operators are specified based on what they do (i.e., what effect they have on the reasoning structure). How this effect is achieved, depends on how a given operator is implemented. Here, many operators rely on neural models (e.g., policy model), which – together with their training paradigms – form the third class of the blueprint components. Finally, we also distinguish a set of pipelines, i.e., detailed specifications of operations that orchestrate the interaction between the reasoning scheme and the operators in order to achieve a specific objective, such as training, inference, or data generation. Hence, an RLM can be defined as a composition of a reasoning scheme, a set of operators and associated models, and a set of pipelines.

###### 4.2 Reasoning Scheme

A reasoning scheme is the part of the blueprint that specifies the details of the reasoning steps progressing toward the solution, how they are interconnected to form coherent chains, trees, or more complex reasoning structures, and how these structures evolve in the course of solving the input task.

4.2.1 Reasoning Step

A reasoning step is a fundamental unit of the reasoning structure – a sequence of tokens that advances the RLM towards the solution. Reasoning steps can vary in length, ranging from a single token to entire segments of text. The variability in their granularity depends on the user design choice. In existing schemes, a reasoning step is typically conceptualized as a “coherent and self-contained unit of thought”. For instance, in mathematical proofs, this may correspond to an individual logical argument or deduction.

The flexibility in defining reasoning steps allows models to adapt to different problem domains, balancing finegrained and coarse-grained reasoning. Coarse steps, such as logical arguments (or even complete reasoning pathways [182]), simplify preparation and adoption of training data, enhance interpretability, and – as we discuss in Section 8 – reduce computational overhead. On the other hand, single-token steps enable the utilization of concepts like token entropy [105] to incorporate the model’s uncertainty, as well as the integration of advanced decoding schemes (e.g., speculative decoding [85] or contrastive decoding [88]) explicitly into the RLM design. Yet, while making the reasoning steps more fine-grained allows for a more detailed exploration of solution paths, this increased flexibility results in greater computational demands, particularly when combined with search algorithms such as MCTS.

###### 1

Reasoning Scheme (§4.2)

A toolbox of paradigms for modeling and evolving the reasoning structure

- 1.1 1.2

1.3

- 2.1

###### Reasoning Structure (§4.2.2)

Reasoning Step (§4.2.1)

What is the content of an individual reasoning step?

What is the connec on structure of reasoning steps?

Coarse-grained (e.g., unit of thought) Graph

Chain Tree

Nes ng

Example: TS-LLM, Tree of Thoughts

Example: Graph of Thoughts

A node can contain another structure

(not a par cularly good reasoning step)

Sort the numbers "3,2,4,5,6,12,5,6"

###### Input task

Input task Input task Input task Input task

(input task statement)

Numbers are blue

(a reasonably good reasoning step)

Split into two sets

Look up

Quicksort Sor ng is simple

"3,4" & "5,2"

Split into Quicksort two sets

"3,2,4,5" & "7,12,5,6"

Pick Pivot

sorts numbers

Nodes form a DAG

Example: TS-LLM

Example: LLaMA-Berry

"3,2,4,5" & "6,12,5,6" "3,2,4" & "5,2,6"

Fine-grained (e.g., single token)

Root contains the input task

Input task

Sort the numbers "3,2,4,5,6,12,5,6"

Reasoning Strategy (§4.2.3)

How does the reasoning structure evolve in order to progress solving the input task?

Split

Look Numbers

Sor ng

Ensemble methods

###### Beam Search Ensemble Methods R1 style

MCTS

The value func on assigns a score to each node

Select a ﬁxed number of con nua ons per level

Diﬀerent structures can evolve independently

Input task Input task Input task

Input task Input task

up over

across

down

Example: Token entropy

v=9

v=9

v=6

- v=5
- v=6

MergeSort QuickSort

##### ...

v=3

v=2

v=3

v=2

Decoding Strategy (§4.2.4)

v=8

v=8

v=3 v=5

v=1 v=2

v=3

v=1 v=2

...

Select conﬁgura ons with highest scores

Nucleus sampling

Greedy search

Example: Tree of Thoughts

Example: TS-LLM, Marco-o1

v=3 v=2 v=4 v=1

v=3 v=2

Example: Best-of-N

Select best solu on Select best

###### 4

Pipelines Inference: §3.1.1, Appendix C.1 Training: §3.1.2, Appendix C.2/C.3 Data Genera on: Appendix D

Toolbox of reasoning schemes

Toolbox of pipelines

### RLM

Toolbox of operators

Toolbox of models

A toolbox of neural models for implemen ng operators and of paradigms for training these models

A toolbox of opera ons for changing & interac ng with the reasoning structure

###### 3

###### 2

Operators (§4.3) Models (§4.4)

Structure Operators (§4.3.1)

###### 3.1 Models Harnessed (§4.4)

Modify the reasoning structure

What operators are being implemented as models?

Aggregate Example: Graph of Thoughts

Generate

###### Valuemodel Policymodel Rewardmodel ... onMoremodelsdetailsin

Prune

Example: "Expand" rou ne in MCTS

Appendix B

3.2

Training Paradigm (§4.4.1)

Gather data from more than one node

remove nodes

add nodes

How is a given model being trained?

Direct Preference Proximal Policy Op miza on Op miza on

Restructure

(apply arbitrary structural transforma ons)

Rejec on Sampling

Example 1: Summary of ensemble structures

Example 2: Lineariza on

More details on training in Appendix C

Supervised ﬁne-tuning Reasoning Policy Op miza on

+ =

Example: Journey Learning

3.3

###### Training Data Scope (§4.4.2)

What informa on does a single training sample contain?

Evaluate (§4.3.4)

Traversal Operators (§4.3.2)

Update (§4.3.3)

###### Outcome-Based Supervision Process-Based Supervision

2.2 2.3 2.4

Training samples contain all intermediate steps between input and output, annotated with a quality score (q) or a label that is either correct ( ) or incorrect ( ).

Training samples only contain inputs and outputs as well as a label that is either correct ( ) or incorrect ( ).

Map the structure to values

Specify which nodes to select for next opera on Modify the nodes' contents but not the structure

###### Select Backtrack

Example: "Evaluate" rou ne in MCTS

Example: "Select" rou ne in MCTS

Example 1

Example: "Backprop" rou ne in MCTS

Example 1

Input: Order the following numbers in ascending order: "3,2,4,5,6,12,5,6"

Input Output

Output:

q=0.7 q=0.5 q=0.8

- "2,3,4,5,5,6,6,12"

Input: Order the following numbers in ascending order: "3,2,4,5,6,12,5,6"

Output:

- "2,3,4,5,6,12"

Example 21

Example 21

back-track

Input Output

Example: Backtrack scheme in Tree of Thoughts

Step selected for further expansion (e.g., due to a high score from a value model)

Value

- Fig. 5: A blueprint for reasoning LMs. It consists of four main toolboxes: the reasoning scheme (the top part), operators (the bottom-left part), and models (the bottom-right part); pipelines are mentioned in the center and detailed in Appendix C.1 and in Algorithm 1 (the inference pipeline), Appendix C.2, Appendix C.3, and in Algorithms 2–7 (the training pipelines), and in Appendix D (the data generation pipeline).

- 4.2.2 Reasoning Structure The reasoning structure specifies how individual reasoning steps are connected and organized. Common structures include chains (linear sequences), trees (hierarchical branching), and graphs (arbitrary connections).

Chains are sequential reasoning flows, where each step builds directly on the preceding one. Chain structures are prevalent in CoT-based models, where each reasoning step follows logically from the previous step in a linear progression. In tree structures, each reasoning step can branch into multiple continuations, forming a decision tree. This structure is commonly used in MCTS-based frameworks, where multiple potential paths are explored before selecting a branch that will be further investigated. It enables more effective exploration of the space of reasoning steps, but simultaneously makes the RLM design more complex and costly. Finally, graph structures allow for arbitrary dependencies between reasoning steps, enabling graph-based reasoning, such as that found in the Graph of Thoughts (GoT) framework [11].

Further generalization involves nested structures, where reasoning nodes themselves may contain substructures. For example, a node in a tree structure may represent a CoT chain, as proposed in LLaMA-Berry [182]. This hierarchical organization could be particularly useful for multi-step tasks where high-level decisions guide low-level computations, such as meta-reasoning frameworks [182]. One could harness any other higher-order structures, such as hypergraphs, motifs, and others [12], [13], [16], [19].

- 4.2.3 Reasoning Strategy The reasoning strategy governs how the reasoning structure evolves, specifying the process by which new reasoning steps are added and integrated. Example strategies include:

- • MCTS [80] A popular approach that balances exploration and exploitation by simulating multiple reasoning paths and selecting the most promising one based on a scoring function.
- • Beam Search [141] A breadth-limited search that keeps a fixed number of top-ranked continuations at each step. While commonly used for decoding token sequences, beam search can also be applied to reasoning steps.
- • Ensemble Methods These methods involve aggregating multiple independent reasoning strategies, such as combining chains and trees to enhance robustness and accuracy. One example is Best-of-N [157], [163] – a strategy where multiple independent reasoning paths are generated, and the most effective solution is selected based on predefined criteria, e.g., accuracy or completeness. Another example is tree ensemble (Forest) [20] where, instead of a single reasoning tree, a reasoning “forest” consists of multiple disconnected trees, which may eventually converge at a shared solution node. This approach supports diverse reasoning pathways that parallelize exploration.

Reasoning Strategy vs. Decoding Strategy. It is crucial to distinguish reasoning strategies from token-level decoding strategies. While decoding strategies, such as greedy search and nucleus sampling [66], generate the internal token sequences within a reasoning step, reasoning strategies focus on the higher-level process of integrating and expanding reasoning steps within the reasoning structure.

###### 4.3 Operators

Operators specify operations that can be applied to various parts of the reasoning structure to progress the reasoning process. We now provide an extensive toolbox of operators. Many of them have been widely used in RLM-related designs, but some – to our best knowledge – are still unexplored, we include them to foster innovation and propel the design of more effective and more efficient RLMs.

- 4.3.1 Structure Operators Structure operators transform the reasoning structure by taking it as input and producing a modified version, typically through addition of reasoning steps. For instance, they may add new children to a specific node, facilitating the exploration of alternative reasoning paths.

- • Generate The generate operator adds one or more new reasoning steps to a reasoning structure. Within the MCTS reasoning strategy, this operator is typically implemented as a policy model to generate new steps. In other strategies, the generation operator may involve sequentially appending steps (CoT) or exploring multiple candidate steps in parallel (Beam Search).
- • Aggregate This operator combines multiple reasoning steps, paths, or structures into the next individual step. This enables consolidating information or improving coherence. It is used in Ensemble Methods [20] or in Graph of Thoughts [11].
- • Prune This operator removes nodes or reasoning steps from the structure that are deemed suboptimal or irrelevant based on evaluation metrics. It enables optimizing the reasoning structure in order to, e.g., reduce token costs.
- • Restructure The restructure operator applies arbitrary transformations to the reasoning structure, enabling flexible reorganization of its components. A notable example is the conversion of a reasoning tree into a linear chain by rearranging its branches into a sequential series of steps, as done in Journey Learning [123]. This restructuring facilitates the integration of insights from diverse branches into a cohesive flow, “flattening” it and making it easier for the model to process and utilize information within a single, unified context.

Discussion on Diversity In structure operators, there is a notion of how diverse the outcomes of the operator are. For example, when generating k new reasoning steps, one may want to make the contents of these steps as different to one another as possible. While different mechanisms to steer diversity exist, a typical approach is the use of the policy model temperature. We additionally propose to consider the Diverse Beam Search [156] which promotes diversity by maintaining multiple diverse candidate sequences during decoding. In MCTS, there is also a distinction between exploitation (expanding the structure by applying generation operators within an already established tree branch) and exploration (generating new branches). Here, one impacts diversity by manipulating the exploitation-exploration tradeoff, as determined by the Upper Confidence Bound for Trees (UCT) formula [80] or its variants.

- 4.3.2 Traversal Operators Traversal operators define how the reasoning process navigates through the existing reasoning structure. These opera-

tors play a crucial role in shaping the flow of reasoning by determining which paths to pursue.

- • Select The select operator determines which reasoning steps to pick for further exploration, evaluation, or refinement within the reasoning process. It evaluates existing elements based on predefined criteria, such as heuristic scores, likelihood estimates, performance metrics or search strategies like PUCT [127] or UCT [80], selecting the most promising candidates to guide the next stages of reasoning. By balancing exploration (considering diverse alternatives) and exploitation (focusing on high-potential paths), the selection operator optimizes resource allocation and ensures efficient reasoning progression.
- • Backtrack The backtrack operator enables the model to explicitly return to a previous reasoning step and continue along a different reasoning path. This operator supports error correction, divergence handling, and hypothesis revision by abandoning unproductive directions in favor of alternative trajectories. The QwQ model output indicates that the reasoning structures used as training data in this model harnessed backtrack.

- 4.3.3 Update Operators The update operator enhances specific parts of the reasoning structure without altering the structure itself. A common example is the backpropagation phase in MCTS, where evaluation scores are propagated and updated along existing reasoning steps to inform future decisions. Another form of update involves refining the content of individual nodes or subsets of nodes, replacing their original versions with improved iterations. The refine operator could address ambiguities, correct errors, and optimize inefficiencies, resulting in a more robust version of the step [103]. It could also integrate suggestions from self-critique [132] (evaluates steps to identify weaknesses and suggest targeted improvements), summarization [191] (condenses key elements into concise representations to streamline the reasoning structure), or rephrasing [45] (reformulates steps to improve clarity and coherence while preserving their logical integrity). An example would be the “improve” thought transformation in Graph of Thoughts [11].
- 4.3.4 Evaluate Operators Evaluate operators take as input a segment of the reasoning structure and output a value without any modifications to the structure. They are widely used with reasoning strategies, such as MCTS.

One important type of evaluation occurs when the reasoning structure reaches a terminal state, allowing the full reasoning sequence to be assessed against a known solution—applicable to tasks with definitive answers, such as mathematical problems. This terminality evaluation verifies whether the final step provides a correct and complete solution.

One can also evaluate intermediate steps (i.e., nonterminal ones). This can involve estimating the reward associated with specific reasoning steps, using heuristics, aggregated simulation outcomes, or a trained reward model for more efficient assessments. Other methods such as embedding-based verification could also potentially be harnessed [17].

Another form of evaluation employs a value estimator, which judges a given reasoning step based on its expected contribution to a correct final outcome. This method evaluates both the correctness of the step and its alignment with the overall solution goal. Such evaluations can be performed through simulations, as in the original MCTS algorithm, or more efficiently using a learned value model [139].

A critical aspect of evaluation is the selection of appropriate metrics. For instance, in value estimation, an ideal metric considers both the correctness of a reasoning step and the extent of progress it represents toward the final solution, ensuring a balanced assessment of its contribution.

4.3.5 Discussion: Test-Time Compute

One of the recent trends in next-generation LLMs [104], [158] is to shift from merely increasing model sizes to enhancing computational strategies during inference, a concept known as the test-time compute (TTC). This approach allocates additional computational resources during a model’s execution to improve performance, particularly in complex reasoning tasks. This methodology mirrors human cognitive processes, where increased deliberation is applied to more challenging problems.

Recent studies [141] indicate that optimizing test-time compute can be more effective than merely increasing model size. For instance, employing a compute-optimal strategy—where computational resources are adaptively allocated based on the problem’s complexity—can enhance efficiency by over four times compared to traditional methods. Moreover, in scenarios where smaller base models achieve moderate success rates, augmenting test-time compute enables them to outperform models up to 14 times larger.

While test-time compute offers significant benefits, it also presents challenges, related to – among others – resource allocation (determining the optimal amount of computational resources for each inference task requires sophisticated strategies to balance performance gains against computational costs), dynamic scaling (implementing adaptive compute strategies necessitates models capable of assessing problem difficulty in real-time and adjusting their computational efforts accordingly) [106], and hardware implications (the shift towards increased test-time computation may influence hardware requirements, putting more pressure on delivering specialized inference-focused hardware solutions).

Test-Time Compute in the Context of the Blueprint. Our blueprint offers mechanisms to dynamically allocate computational resources during inference to improve performance, particularly for more complex problems. By leveraging the modular structure of the blueprint, TTC can be effectively implemented through specific operators designed for reasoning tasks. We now provide several examples.

• The generate operator can be used to implement TTC by dynamically increasing the number of next reasoning steps generated for harder problems. For simpler tasks, the operator may only generate a minimal set of continuations. However, for more complex problems, the operator can be used to create a larger set of potential reasoning steps, thereby expanding the search space.

- • The refine operator provides another avenue for implementing TTC by enhancing a given reasoning step multiple times for harder problems. In this approach, the operator iteratively improves the quality of a reasoning step, addressing ambiguities, rectifying errors, or improving clarity. For simpler tasks, the operator might only refine a step once, while for more complex reasoning, it can perform multiple enhancement iterations to ensure the output meets a higher standard of precision and robustness.
- • The traversal operators, such as select, enable the exploration of multiple reasoning paths at test time, offering another key mechanism for implementing TTC [184]. By using select on several next reasoning steps, the model can dynamically expand its search tree for more challenging problems, thereby increasing the diversity and depth of reasoning paths under consideration. For example, in a complex task, the model might select multiple highprobability steps and explore their corresponding continuations in parallel. This approach facilitates broader exploration of the reasoning space, ensuring that promising paths are not prematurely discarded.
- • To efficiently manage the expanded set of possibilities, the blueprint allows integration with the aggregate operator. This operator evaluates the generated reasoning paths and selects the most promising ones based on predefined criteria, such as the likelihood of correctness or the quality of intermediate steps. This combination ensures that while more computational resources are allocated for challenging tasks, only the most relevant paths are explored further, optimizing both accuracy and efficiency.

###### 4.4 Models

Models are used to implement various types of operators. Most common are the value model (implementing the value evaluation operator) and the policy model (implementing the generate operator).

Models are further categorized and discussed in detail in Appendix B; we discuss the variants of the value model (QValue Model, V-Value Model), we compare Process-Based and Outcome-Based Reward Models, and we formally identify a new variant of models, the Outcome-Driven ProcessBased Reward Model.

- 4.4.1 Training Paradigm

Each model must be trained according to a specified paradigm, which outlines the methodology for optimizing its performance. This paradigm defines key training components such as the loss function, data generation and labeling procedures, and other critical training details.

A wide range of training schemes has been developed for models used in RLMs, with early foundational work stemming from advancements related to AlphaZero. These schemes have since evolved to support the complex requirements of reasoning tasks within LLMs. Common training paradigms include supervised fine-tuning (SFT), where models are trained on reasoning sequences labeled with q-values; rejection sampling [25], [144], which involves filtering generated outputs based on quality criteria; and RL-based methods such as Proximal Policy Optimization (PPO) [135], Direct Preference Optimization (DPO) [125],

and reasoning-specific variants like Reasoning Policy Optimization (RPO) [121]. Several training paradigms also incorporate self-learning, where the model iteratively improves by generating and evaluating its own reasoning sequences, thereby simulating competitive or cooperative reasoning scenarios.

4.4.2 Training Data Scope

The training data for RLMs can vary significantly in terms of how much of the reasoning structure it captures. We now outline two established approaches, Outcome-Based Supervision (OBS) and Process-Based Supervision (PBS). More details regarding both OBS and PBS can be found in Appendix B.1.

In Outcome-Based Supervision (also known as a sparse training signal) [38], [155] each training sample consists solely of the input and the corresponding output. For example, in mathematical problem-solving, a sample may include the task statement and the final solution, labeled as correct or incorrect. This approach is straightforward to implement, and the required data is relatively easy to collect. However, it can limit the model’s reasoning accuracy, as it provides minimal insight into the intermediate steps that led to the solution [91].

An alternative approach is Process-Based Supervision (also known as a dense training signal) [91], [160], where a training sample reflects the entire reasoning structure. In this case, the sample contains not only the input and final output but also all intermediate reasoning steps, annotated with labels indicating the quality of each step. This richer training data allows the model to learn more granular reasoning patterns, improving its ability to generate accurate and interpretable solutions by understanding the reasoning process in detail. However, such data is much more challenging to generate or gather [91].

OBS vs. PBS By varying the training data scope, developers can strike a balance between ease of data collection and the depth of reasoning insights provided to the model, with dense supervision generally offering improved performance at the cost of increased data complexity. We detail these, and additional aspects of ORMs and PRMs in regards to pipelines for different training phases and paradigms in Appendix B, Appendix C.2, Appendix C.3, and in Algorithms 2–7.

Trace-Based Supervision (TBS) is a potential way to extend PBS by incorporating detailed information about the sequence of applied operators, including traversal operators, within the reasoning structure. By capturing the full trace of how reasoning steps are generated, refined, or revisited, TBS would provide richer supervision that teaches the model to internalize not just the reasoning steps but also the process of navigating and manipulating the reasoning structure itself. This approach could enable the training of more powerful Implicit RLMs by guiding them to replicate the reasoning dynamics of explicit structures, improving their ability to reason flexibly and efficiently.

###### 4.5 Pipelines

A pipeline is a detailed specification of operations that orchestrates the details of the interaction between the reasoning scheme and the operators and models to achieve a

Reasoning Reasoning Operator Models Pipeline Scheme

Structure Traversal Update Evaluation

Remarks Structure Step Strategy Gen. Agg. Pr. Res. Sel. BT Ref. Bp. Inter. Term. PM VM Inf. Tr. DG

Explicit RLMs (Section 5.1)

rStar-Math [57] E Tree C Thought + Code Block E MCTS PRIME [41], [176] E Multiple Chains F Token

E Best-of-N Marco-o1 [187] E Tree F Token Sequence

C Thought

E MCTS

C Thought

Journey Learning (Tr.) [123] E Tree E Thought E Tree Search * *Separate entry OpenR [158] E Tree C Thought E Best-of-N

E Beam Search E MCTS

LLaMA-Berry [182] E Tree of Chains C Solution E MCTS ReST-MCTS* [183] E Tree C Thought E MCTS * *Advice by critic AlphaMath Almost Zero [26] E Tree F Thought E MCTS * * *Single model MCTS-DPO [168] E Tree F Token Sequence E MCTS * * *Single model AlphaLLM [153] E Tree C Option E MCTS TS-LLM [157] E Tree F Token

E MCTS E Tree Search

F Sentence

###### Implicit RLMs (Section 5.2)

DeepSeek-R1 [58] I Chain F Token QwQ [152] I Chain* F Token *Linearized Tree Journey Learning (Inf.) [123] I Chain* C Thought I DFS *Linearized Tree

###### Structured Prompting Schemes (Section 5.3)

Graph of Thoughts (GoT) [11] E Graph* C Thought E Controller *DAG Tree of Thoughts (ToT) [174] E Tree C Thought E DFS

E Beam Search Self-Consistency (SC) [163] E Multiple Chains C Thought E Majority Voting Chain of Thought (CoT) [165] I Chain C Thought

TABLE 1: Comparison of RLMs with respect to the provided taxonomy (Section 4 and Figure 5). “Reasoning”: Details of the reasoning approach, specifically what is its Structure and its Strategy? “Reasoning Operator”: Does a given scheme support operators on the reasoning structure? If yes, which classes (and specific functionalities) are supported Structure (“Gen.”: generate, “Agg.”: aggregate, “Pr.”: prune, “Res.”: restructure), Traversal (“Sel”: select, “BT”: backtrack), Update (“Ref.”: refine, “Bp.”: backpropagate), and Evaluation of “Inter.”: intermediate steps and “Term.”: terminal steps? “Model“: Does a given scheme use models to implement its operators and if so, which ones (“PM”: policy model, “VM”: value model)? “Pipeline”: Which pipelines are harnessed by a given scheme (“Inf.”: inference, Tr.”: training, “DG”: data generation)? When describing representations, we use the following abbreviations: “E”: explicit, “I”: implicit. “F”: fine-grained. “C”: coarse-grained. “ ”: full support (i.e., YES), “ ”: partially [supported], “ ”: no support (i.e., NO).

specific objective. Typically, an RLM would incorporate a single pipeline for inference and a separate pipeline for training each model used in an RLM. Moreover, there could also be pipelines for synthetic data generation used for training models. One can also distinguish a pipeline that trains an Implicit RLM using the provided reasoning traces from the Explicit RLM.

The details of pipelines depend on arbitrary design choices. In Section 3, we provided a general description of how these pipelines work. In Appendix C, we present detailed algorithmic specifications of our pipelines, along with insights into the reasoning behind these design choices. Specifically, the inference pipeline can be found in Appendix C.1 and in Algorithm 1. Pipelines for different training phases and paradigms can be found in Appendix C.2, Appendix C.3, and in Algorithms 2–7. The data generation pipeline is detailed in Appendix D.

###### 5 EXPRESSING EXISTING SCHEMES

We now showcase the expressivity of our blueprint, by illustrating how it can be used to model a broad scope of existing RLMs and other related works. We summarize the outcomes of the analysis in Table 1. We start with typical and most prevalent Explicit RLM architectures based on MCTS and policy and/or value models, where a single reasoning step is an individual logical argument (Section 5.1). We also discuss there schemes that generalize this typical design, by harnessing nesting or restructure operators. Finally, we study Implicit RLMs (Section 5.2) and various structured prompting schemes such as CoT or ToT (Section 5.3), showing that they also fit our blueprint.

###### 5.1 Explicit RLMs

We start with the most widespread variant of RLMs that follows the architecture outlined in Section 3.1. These reasoning models such as TS-LLM [157], AlphaLLM [153], MCTS-DPO [168], and others [26], [57], [158], [182], [183], [187] generally employ an explicit tree structure in which a node represents a distinct reasoning step. The reasoning strategy is based on the MCTS and focuses on iterative exploration, expansion and evaluation of nodes within the tree. By incorporating value mechanisms—such as promptbased evaluation or dedicated value models, the system identifies and prioritizes promising branches, facilitating more informed decision-making and refinement of the reasoning process. All MCTS based reasoning models implement at least a next-step generation operator, an evaluation operator, and the update operator for back-propagating the values. In addition, ReST-MCTS*, LLaMA-Berry, and Marco-o1 support a refinement operator to further improve produced reasoning steps.

Journey Learning [123] exhibits two main differences to typical MCTS-based RLMs. First, it harnesses a linearization restructure operator, in which the tree reasoning structure is transformed into a chain, by extracting several selected reasoning chains from it and combining them together into an individual long chain. This way, the scheme attempts to harness insights from different tree branches. By maintaining a chain-based structure, Journey Learning preserves the simplicity of linear reasoning while embedding the capacity for self-correction and exploration of multiple hypotheses. Additionally, Journey Learning introduces a pipeline for the internalization of such long reasoning chains into its

weights. This enables the final model to generate such long reasoning chains, possibly containing different reasoning branches, directly from its weights, illustrating the path towards the construction of an implicit RLM.

###### 5.2 Implicit RLMs

Qwens’s QwQ [152] embodies a fully implicit reasoning model, characterized by an implicit reasoning structure that is generated autoregressively directly by the model weights. The reasoning strategy in QwQ – as indicated by the model output – harnesses next-step generation, backtracking, summarization, and critique generation to derive the final solution. At each step, the model implicitly generates a new node within the chain by employing one of these four implicit operators, presumably implemented using special tokens.

###### 5.3 Structured Prompting Schemes

Finally, we also illustrate that advanced structured prompting schemes, such as CoT, ToT, and GoT, constitute a fully explicit RLM structure without any implicit reasoning than what is originally presented in the used LLM, i.e., no models nor training or data generation pipelines.

CoT [165] utilizes an implicit reasoning structure consisting of a chain of reasoning steps. The reasoning strategy employed in CoT is oriented towards constructing a single coherent chain of reasoning, culminating in a solitary solution, thus only needing the generate operator. CoT serves as the foundational framework for a range of advanced reasoning strategies, including prompting methodologies such as Self-Consistency and Self-Refinement, among others.

Self-Consistency (SC) [163] extends the CoT framework by introducing redundancy into the reasoning process. It generates multiple reasoning chains and employs a majority-voting mechanism to determine the most consistent solution, which implements a select operator from our blueprint.

ToT [174] adopts an explicit reasoning structure organized in a hierarchical, tree-based format. Within this framework, each node corresponds to a distinct reasoning step, and branching facilitates exploration across multiple inferential pathways (the generate operator). Additionally, an evaluation operator, implemented via a specialized prompt and the LLM itself, assesses branches of the tree.

GoT [11] introduces a more intricate reasoning structure by employing an explicit graph-based representation. In this framework, nodes represent individual reasoning steps, and the graph architecture supports non-linear, interdependent relationships between these steps. The reasoning strategy in GoT is orchestrated by an external controller, realized as a separate LLM, which guides the exploration, refinement and aggregation of the graph’s nodes.

###### 6 HOW TO USE THE BLUEPRINT

We now outline how to use our blueprint for the user’s application; we keep this section in a tutorial style.

###### 6.1 Part 1: Define the Reasoning Scheme

The first step in using the blueprint is to define the reasoning scheme, which specifies the foundational structure and strategy of your RLM. Start by selecting the reasoning structure. Chains are the most affordable in terms of token costs, at least when it comes to ICL [16]. Trees, while the most expensive, offer rich branching that enhances exploratory reasoning. Graphs, though slightly cheaper than trees, introduce additional challenges in implementation but can yield significant accuracy gains due to their flexibility.

Next, decide on the granularity of reasoning steps. Coarse-grained steps, such as thoughts or sentences, are widely used due to their simplicity and ease of scaling. However, token-based granularity, which operates at the level of individual tokens, offers the potential for greater precision and unexplored accuracy improvements. This approach, while promising, demands significantly more computational resources and careful design. This decision defines your action space (possible operations) and state space (configuration of the reasoning structure).

Another decision is choosing a reasoning strategy to govern how the reasoning structure evolves. MCTS combined with some variants of policy and value models remains the most widely adopted approach due to its balance of exploration and exploitation. However, alternative strategies that have not been deeply studied, such as ensembles of reasoning structures, may offer untapped potential.

Finally, determine the specific details of your chosen strategy, including parameters like exploration coefficients, decoding strategy, scoring functions, and step evaluation methods. These choices will significantly impact the model’s reasoning dynamics, scalability, and overall effectiveness. Each decision at this stage lays the foundation for tailoring the RLM to your specific application requirements.

###### 6.2 Part 2: Define the Operators

The next step is to specify the set of operators that will govern the reasoning process. For an MCTS-based design, the simplest approach is to implement the core operators: generate (often called expand for MCTS), select, and backpropagate. These fundamental operations suffice for many scenarios, providing a straightforward framework for reasoning.

Beyond the basics, consider whether you want to incorporate less mainstream operators, such as backtrack. By explicitly including backtrack, you enable a clearer tracking of progress within the search tree, making it potentially easier to revisit and refine earlier reasoning steps. This approach also facilitates advanced training schemes, like Trace-Based Supervision, by generating richer and more structured data. Consider using this and other operators within our toolbox.

You will also need to determine the implementation details for each operator. Decide which operators will be implemented as neural models—such as using a policy model to guide selection or a value model for backpropagation—and which will rely on non-neural methods. This choice affects both the computational complexity and the flexibility of the system, so it’s important to align these decisions with your reasoning scheme and performance goals.

###### 6.3 Part 3: Determine the Training Details

In this phase, you need to outline the specifics of training for the models that will implement operators. For an MCTSbased design, consider the typical approach of using the policy model to implement generate (expand) and the value model for evaluation. If necessary, you might also train a separate model to calculate the reward at individual nodes, enhancing the precision of the reward signals.

Identify the application or training domain in order to address generalization requirements. This step ensures that your models are trained on data representative of the tasks you want them to handle.

Define the models, including their architectures and the selection of suitable base models. Consider how the design of these models—such as transformer-based architectures or more specialized designs—aligns with your reasoning structure and overall objectives.

Collect training data for both the policy and value models. For the policy model, consider generating data automatically with our pipeline or using a scheme such as CoT prompting, and include a special end-of-step token to ensure clean segmentation. For the value model, generate data through full MCTS simulations, which provide rich, structured information about reasoning paths and outcomes.

Fine-tune the models as needed. If using coarse reasoning steps, perform supervised fine-tuning (SFT) on the policy model to teach it how to reason step-by-step. Similarly, apply SFT to the value model to initialize it as a reliable evaluator.

Run MCTS with initialized models to collect additional data. You might filter this data to keep only high-quality reasoning paths (terminal states) or strong signals (high absolute advantages) for further training.

Finally, train both models either by additional SFT rounds or with reinforcement learning methods such as Proximal Policy Optimization (PPO). This ensures that the models are optimized not only for accuracy but also for the efficiency and robustness needed in complex reasoning tasks.

7 FRAMEWORK X1: DESIGN & IMPLEMENTATION

We now introduce x14, an extensible and minimalist framework that can serve as ground to design and experiment with RLMs, and currently provides one example of the blueprint.5 An overview of the framework is in Figure 6.

###### 7.1 Reasoning Scheme

The x1 framework employs a tree reasoning structure in conjunction with MCTS as the reasoning strategy. This combination allows for a systematic exploration of reasoning paths while balancing exploration of new possibilities and exploitation of promising solutions judged by a value model. The framework achieves this alignment through the implementation of a series of operators that guide the construction, traversal, evaluation, and updating of the reasoning tree.

4https://github.com/spcl/x1 5We are working continuously on expanding the framework as well as

adding more RLMs.

###### 7.2 Operators

The generate operator plays a crucial role in expanding the tree by adding new children to a selected node. To improve the diversity of these newly generated nodes, we employ Diverse Beam Search [156], which ensures variability among the children. Alternatively, high-temperature sampling can be used to introduce stochasticity into the generation process, fostering the exploration of different reasoning paths.

Traversal of the reasoning tree is managed by the select operator, which uses the PUCT function to identify the next node to expand. This operator balances a trade-off between exploration, favoring less-visited nodes, and exploitation, reinforcing nodes with higher potential based on previous evaluations. Always starting from the root node, the traversal mechanism ensures that the system can dynamically explore alternative paths and recover from suboptimal decisions by backtracking and selecting new branches.

The backpropagation update operator refines the qvalues which can be used as guidance for the select operator along the path from an expanded node back to the root. This process incorporates new information from downstream nodes, leading to progressively more accurate q-values for the intermediate nodes. These refined q-values subsequently inform future decisions, making the reasoning process increasingly robust over time.

The framework implements two different evaluate operators. First, the value estimate operator predicts the discounted expected future reward for a chain extending from the root to a specific node. This prediction is derived from the q-value model, offering a quantitative measure of the path’s quality. Second, when the ground truth is available, the terminality evaluation operator directly evaluates leaf nodes for correctness, assigning fixed rewards to verified solutions. These rewards are incorporated into the q-values of upstream nodes, ensuring that the reasoning process is informed by both model predictions and objective validation.

###### 7.3 Models & Training Paradigms

Both the value and the policy model in x1 are fine-tuned versions of an LLM6, without reliance on prompting, which is used in several other RLM architectures [57], [183]. This design decision aims to maximize the quality of results. We now outline briefly selected key aspects of how we train these models, full details can be found in Appendices B, C, and D.

7.3.1 Training the Policy Model

The policy model also leverages an LLM to generate new nodes during the MCTS. It is fine-tuned to output an individual next reasoning step instead of a whole chain of thoughts towards a completion (which LLMs commonly do). We achieve this by introducing a novel token, the end of intermediate step (eois) token, which denotes the completion of each reasoning step. The eois token complements the standard end of sequence (eos) token, which indicates the conclusion of an entire reasoning chain. By incorporating

6We currently use Llama-3.1-8B-Instruct as base model.

###### Phase 2 Training: Reinforcement Learning

###### Phase 1 Training: Ini alize models

Alternate between and to improve models ( ) and data ( )

2 2

- 1

- 2

1

Policy Model

LLaMA 3.1

1

Training

Policy Server

RL Training Data (data generator)

generates

generates

context + reasoning step, advantage

[Figure 3]

PPO Trainer

Mul ple CoT examples

Mul ple MCTS trees

Input

Input

Input

Input

Input Output Input Output Input Output

Policy &batchedsynced Model

Data Genera on

Output

Output

Output

Output

context

Buﬀer

algorithmically insert \\eois tokens between reasoning steps

extract one training sample per node

Input

new reasoning step

Output

Training Data Training Data

....\\eois..........\\eois........\\eois....... context + reasoning step + MCTS q-value

Value Server

1

Training

Input

context + reasoning step + MCTS q-value

..\\eois.......\\eois..........\\eois........... .......\\eois........\\eois.......\\eois......

context + reasoning step, q-value

context + reasoning step + MCTS q-value

[Figure 4]

MSE Trainer

Output

LLaMA 3.1

Value Model

2

Data Genera on

LLaMA 3.1 Training

SFT Training

Stack of linear layers

Input

context + reasoning step value

Policy Model Value Model

Output

- Fig. 6: An overview of the x1 framework is presented, highlighting its two-phase training process. In phase 1, the models are initialized, while in phase 2, the models are iteratively refined by alternating between constructing a sufficient number of MCTS trees and training the models on data derived from these trees.

the eois token, the framework enables the explicit identification of intermediate reasoning steps, allowing for greater interpretability and precise determination of whether the reasoning process is complete or ongoing. This dual-token strategy enhances the LLM’s capability to decompose complex problems into manageable substeps while ensuring the model recognizes when a solution has been reached.

7.3.2 Training the Value Model

The value model is designed to estimate the sum of the expected discounted future rewards for a sequence of reasoning steps and a newly proposed reasoning step, quantifying the value of the node modeling this step. For a given node in the MCTS tree, its value (referred to in the MCTS literature as state action value or q-value) is defined as the expected cumulative reward discounted by the number of steps required to achieve it. Formally, the q-value Qπ(st,at) for traversing the edge to node st+1 when taking action at from st at depth t in the MCTS tree is expressed as

Qπ(st,at) = E γT−tr(sT,aT) | st,at

N

1 N

γT−tr(s(Ti),a(Ti)), (1)

≈

i=1

where γ is the discount factor, T marks the last reasoning step aT that is added, resulting in the terminal state sT+1 containing the complete reasoning structure and rewards are modeled sparse. The terminal state sT+1 is defined as the state in which no additional reasoning steps can be added. It typically represents the state containing the final solution to the problem at hand. Accordingly, r(sT,aT) is the terminal reward. We chose to model rewards as sparse, where only the final reasoning step receives a non-zero reward, since for most reasoning tasks, only the final answer can be evaluated against the true solution. As a result, one can only obtain a reward signal when the last step is reached.

We can approximate the q-value by sampling N reasoning chains until the terminal state, as in Eq. 1, and averaging the terminal rewards discounted by the depth required.

The q-value model is trained using data from completed MCTS searches. Initially, when the q-value model is unavailable, N simulations (complete rollouts) are performed, and the average discounted reward is used to initialize the qvalues for each node. More information can be found in the Appendix D.2.

###### 7.4 Enabling Scalability and Efficiency

The current implementation is built to scale to multiple GPUs on multiple nodes. To further enhance the scalability and computational efficiency, several architectural and operational improvements have been implemented.

One design decision involves the decoupling of the value and policy models. The deployment of dedicated value and policy servers confers several advantages:

- • Scalability The decoupling of value and policy servers from the MCTS instance facilitates scalability and the execution of multiple parallel MCTS instances.
- • Batch Processing The policy server incorporates batching capabilities, allowing the concurrent processing of multiple queries, thereby enhancing throughput.
- • Resource Optimization The independent allocation of computational resources to the value and policy models is inherently supported by the framework’s architecture, enhancing efficient resource utilization.
- • Replication and Distribution The separation of value and policy models facilitates the application of distinct replication and distribution strategies.

Figure 6 illustrates the implementation of the framework as a server architecture, demonstrating how these structural enhancements contribute to improved scalability and efficiency. Building on these architectural enhancements, we employ the following strategies to further optimize the

framework’s efficiency and scalability, focusing on inference and parallelization.

In the framework, we incorporate the standard optimizations of batching, quantization, and KV caching. Inference calls are batched in the policy model, enabling simultaneous processing of multiple queries. To expedite the reasoning process, the framework creates multiple child nodes in parallel during the node expansion phase. Specifically, N new nodes are generated concurrently in each expansion step, reducing computational overhead and enhancing overall system performance. Further optimization of inference speed is achieved through KV caching and quantization. KV caching mechanisms mitigate redundant computations, while quantization techniques reduce the memory consumption of both policy and value models.

###### 7.5 Blueprint for Efficient Scaling

Our blueprint can be deployed to AI HPC systems and clouds, as both systems provide the performance and resources necessary to scale RLMs. Deployment on HPC systems is straightforward: compute tasks are distributed across statically allocated nodes, connected with a lowlatency and high-bandwidth interconnect, and with training data being available on a high-performance parallel filesystem. On the other hand, the cloud provides many configurable services that offer different trade-offs between performance, cost, and reliability. There, it becomes the user’s responsibility to choose the storage options and compute granularity that provides the best match for expected performance and cost. The architecture of our blueprint fits into the microservice architecture, with a clear separation of compute tasks, data storage, and coordination. This architecture helps to ease the configuration process, as different components of the system can be deployed, scaled, and optimized independently. In particular, the separation of value and policy servers allows them to be scaled separately according to the complexity of reasoning steps that might require different resource allocations to handle task generation and evaluation.

First, we outline the major decisions users must make before deploying the x1 scaling blueprint:

- • Deployment Training and inference tasks are typically allocated to virtual machines and containers, with the latter typically deployed as managed services with an orchestrator such as Kubernetes. There, x1 can benefit from modern frameworks like Ray [115] that hide the complexity of managing a service in a Kubernetes cluster.
- • Data Storage In the cloud, object storage provides automatic bandwidth scalability that allows to scale computations operating on the same data. To overcome latency and power constraints, data can also be placed in in-memory caches like Redis and hybrid solutions that combine disks with flash memory [185].
- • Communication Requirements of the x1 blueprint differ from classical microservices, that rely on high-level abstractions like RPC and REST interfaces. RLM must utilize high-performance network fabrics offered by modern clouds, such as InfiniBand on Azure and Elastic Fabric Adapter (FBA) on AWS, both capable of achieving throughput of 400 Gb/s [42]. These are also available

to training processes distributed across many GPUs, e.g., through specializations of the NVIDIA collectives library NCCL.

• Parallelism We apply parallelism at multiple blueprint levels, including the classic data, model, and pipeline parallelism. These can scale horizontally across a larger number of virtual machines and containers. On the other hand, reasoning steps can benefit from elastic scaling, like in distributed MCTS and Beam Search, where each path can be explored in parallel. There, containers can be allocated on the fly to support new paths and deallocated as soon as the parallelism scale of the computation decreases.

New developments in the machine learning infrastructure can significantly impact RLM deployment strategies:

- • Elastic Compute Computing tasks can be executed on ephemeral resources that trade the guaranteed lifetime and reliability for lower costs, such as spot virtual machines [111]. Serverless functions provide elasticity scalability with fine-grained pricing models [40], which can be a good fit for dynamically generated reasoning steps. However, serverless functions are stateless and suffer from cold starts, which requires optimization techniques dedicated to LLMs [51]. Furthermore, restricted network communication in functions forces the adoption of new communication protocols [39], [76].
- • GPU Management Cloud rental of GPU devices is particularly expensive, and procuring a sufficient number of devices can be challenging, specifically when constrained to a single cloud region. Given the large compute and memory requirements of base models, space-sharing might not be feasible. On the other hand, time-sharing of GPU devices between different x1 services could be a viable alternative, but it is currently constrained by large memory allocations and the cost of swapping model checkpoints between CPU and GPU memory. To increase resource utilization, new techniques for efficient GPU checkpoint and restore are needed [51].
- • Parameter-Efficient Resource Sharing Resource-sharing can be further enhanced by utilizing a shared base model architecture for the policy and value models, while dynamically swapping task-specific parameter layers - such as Low-Rank Adaptation [68], prefix tuning [89], or other adapter layers - on the GPU during inference. These modular strategies keep the base model loaded in device memory and replace only the lightweight task-specific layers, eliminating redundant loading and reducing both latency and memory usage. An example of an RLM, which uses a shared base model with separate additional linear layers for policy and value model, is AlphaMath [26].
- • Cross-Region Deployment Cloud applications are often deployed in a single region to avoid the performance and cost of cross-region data access. However, workloads can be scheduled globally, suspended, and migrated across regions to avoid hardware resource exhaustion and achieve lower carbon emissions [36], [166].

###### 7.6 Example Analysis: Token Probability Distributions

As an illustrative example, we use the framework to directly leverage the token probability distribution, thereby facilitating

the use of associated properties—such as entropy and variance—for guiding subsequent reasoning decisions. By focusing on these probabilistic characteristics, the framework can help identify when to expand a given reasoning step. Using token probability distributions can be used for navigating the reasoning based on both coarse and fine steps. To support this analysis, the x1 implementation includes scripts that provide insights into token-level metrics, such as entropy fluctuations and distribution patterns, to inform reasoning strategies.

7.6.1 Relevance of Token Probability Distribution

The token probability distribution provides critical information about the likelihood of different next-step candidates in a reasoning process. By examining this distribution, we can gain insight into how certain tokens dominate or diversify the reasoning space, and in turn, guide more informed decisions about which step to take next.

We now list a few scenarios where different token distributions offer insights into which reasoning decision is best to take at a given step.

- • Flat Token Distribution. A flat probability distribution occurs when all tokens have roughly equal probabilities. In this scenario, there is significant uncertainty about which step is the best to choose because no single token stands out as a clear candidate. This can make the reasoning process more exploratory, as the model may need to consider multiple tokens equally and rely on additional strategies—such as external heuristics or learned policies—to identify the most promising step. While this can foster exploration, it may also lead to inefficiencies since the model might need to evaluate many equally plausible paths before finding an optimal solution. Another decision that could be taken in such a scenario, is to delay initiating a reasoning step till the token distribution changes to be more skewed.
- • Skewed Distribution with One Dominant Token. When one token has a much higher probability than others, the distribution is highly skewed. This often signals that the model is confident about the next step in the reasoning process. If the dominant token corresponds to a logical or well-supported continuation, this confidence can streamline decision-making and reduce computational overhead. However, if the model’s confidence is misplaced—perhaps due to biases in the training data or a lack of context—relying on a single dominant token may cause the reasoning process to follow a suboptimal path. In such cases, it’s crucial to assess whether the high-probability token genuinely represents the most logical next step or if additional validation is needed.
- • Skewed Distribution with Multiple High-Probability Tokens. In some cases, the distribution may be skewed with a small set of tokens receiving much higher probabilities than others. This indicates that the model sees several plausible continuations, each with a reasonable chance of being correct. While this is generally a positive sign—offering a diversity of credible options—it also complicates the decision-making process. The reasoning strategy must weigh the trade-offs between these top candidates, considering not only their individual probabilities

but also how each choice impacts the subsequent reasoning trajectory. This scenario highlights the need for effective evaluation metrics (like entropy or Gini coefficient) to help select the step that contributes most to reaching the correct or desired outcome.

By analyzing token probability distribution and identifying the cases above and others, reasoning strategies can, for example, improve efficiency (identifying when a distribution is flat allows the reasoning algorithm to focus on diversification or introduce additional constraints to narrow down choices), enhance decision confidence (recognizing when one token is dominant can help expedite decisions, provided the model’s confidence is well-founded), or foster balanced exploration (detecting multiple high-probability tokens facilitates exploring various credible paths without being overly committed to a single option).

7.6.2 Analyzing Token Probability Distribution

To understand the form of a token probability distribution, we examine variance, entropy, VarEntropy, and the Gini coefficient as key metrics that offer distinct perspectives on the distribution’s shape and characteristics.

Variance provides a broad measure of uncertainty by reflecting how spread out the probabilities are across the vocabulary. When variance is low, the probabilities are nearly uniform, indicating a flat distribution. However, variance alone does not capture the specific structure or shape of the distribution. For example, two distributions can have the same variance but differ in their overall form, such as one having multiple minor peaks versus another being nearly uniform with a single dominant token. To address this, we consider further measures below.

Entropy has long been a standard measure of uncertainty and information content in a probability distribution. Higher entropy corresponds to greater unpredictability—requiring more information to describe the system’s state. For instance, if all tokens have nearly equal probabilities, the entropy is high, reflecting a flat distribution. In contrast, low entropy occurs when a small number of tokens dominate, resulting in a skewed distribution. The entropy of a distribution is given by H = − i pi log2(pi), where pi is the probability of the i-th token. This metric provides valuable insight into whether the distribution is diffuse and exploratory or concentrated and decisive.

VarEntropy extends this analysis by measuring the variability of entropy itself, thus offering a dynamic view of how uncertainty changes. A high VarEntropy combined with low entropy often indicates a sharp, focused distribution with a few dominant outcomes. Conversely, low VarEntropy and high entropy typically reflect a flat, uniform distribution where no single token stands out. The VarEntropy is defined as i pi(|log(pi)| − |H|)2. This metric captures the nuanced shifts in distribution shape, helping to pinpoint how tightly probabilities cluster around certain tokens versus how broadly they spread.

The Gini Coefficient, traditionally used to measure inequality, provides another lens on the form of the distribution. A perfectly equal distribution has a Gini coefficient of 0, signifying that all tokens have identical probabilities. A Gini coefficient closer to 1 indicates high inequality, where a few tokens hold most of the probability mass. By visualizing

[Figure 5]

[Figure 6]

(a) 1st example (b) 2nd example

[Figure 7]

[Figure 8]

(c) 3rd example (d) 4th example

- Fig. 7: Four examples of model output with highlighted tokens indicating uncertainty levels. The outputs have been color-coded to reflect the confidence levels of the model’s token predictions. Tokens are highlighted in purple when the highest probability is below 0.8 (indicating lower certainty without significant contention), in blue when the second-highest probability exceeds 0.1 (indicating contention, where another token is a close alternative), and in red when both conditions are met (indicating high uncertainty). These examples illustrate varying levels of prediction confidence and contention in reasoning steps, emphasizing regions of high ambiguity or competition between plausible continuations. This type of visual analysis is useful for identifying points in the reasoning process where the model lacks confidence or is torn between alternatives, guiding refinements in reasoning strategies and model design. It also helps pinpoint critical areas where additional supervision or context may improve model performance.

the cumulative distribution of sorted probabilities, the Gini coefficient highlights how the probability is concentrated or dispersed.

Together, these metrics—variance, entropy, VarEntropy, and Gini—enable a detailed examination of token probability distributions. By leveraging each metric’s unique strengths, we can effectively characterize whether a distribution is flat, skewed with a dominant token, or skewed across several highly probable tokens, ultimately guiding more informed decisions in reasoning and model development.

7.6.3 Example Results

Figure 7 and 8 illustrate example model outputs and their respective token probability distributions. By analyzing the highest probabilities, the second-highest probabilities, and the sum of the remaining probabilities, we gain valuable insights into the underlying token distribution, which can subsequently be quantified through the uncertainty metrics discussed earlier.

In Figures 8a and 8d, specific regions emerge where the top two probabilities are very close, while the remaining probabilities are significantly smaller. Such regions likely indicate scenarios where forking the reasoning process (e.g., exploring multiple paths) could disproportionately benefit future outcomes, as the competing high-probability tokens suggest alternative plausible continuations. Conversely, in instances where the first probability is notably high, with much lower second and remaining probabilities, the model exhibits strong confidence in a single continuation. These cases are conducive to more deterministic reasoning, as forking may be unnecessary.

Additionally, regions with a relatively high sum of the remaining probabilities (close to the top two) highlight flatter distributions with high uncertainty. These scenarios signal a need for cautious reasoning, where clarification or additional contextual refinement may help reduce ambiguity. For instance, such uncertainty may suggest that the model has not yet committed to a specific path and could benefit from

revisiting earlier reasoning steps to address potential errors or misalignments.

Figure 9 further analyzes these results using metrics such as variance, entropy, VarEntropy, and the Gini coefficient. In Figure 9a, a zero-shot prompt demonstrates lower uncertainty overall, suggesting that it yields more confident predictions and potentially higher-quality outputs. However, the presence of specific high-probability tokens (e.g., “472”) raises concerns about potential data leakage into the training set or the tokenizer, which could bias the results. Another notable observation is the high uncertainty associated with <thought>tokens, which appear challenging for the model to predict accurately. This highlights the complexity introduced by token granularity, where most words correspond to single tokens, resulting in a roughly even distribution for the next token across the vocabulary in some contexts.

The uncertainty metrics provide actionable insights for reasoning strategy design. For example, cases with high VarEntropy and low entropy indicate a distribution where a few outcomes dominate, making tree-based search strategies effective. These strategies prioritize exploring highprobability outcomes while avoiding unnecessary evaluations of less probable branches. In contrast, low VarEntropy and high entropy reflect a flat distribution where no clear outcome dominates. Such cases could benefit from clarification mechanisms or intermediate step refinements to reduce ambiguity before proceeding further.

Interestingly, the Gini coefficient often highlights critical regions more effectively than other metrics. In vital reasoning areas, it captures the inequality in token probabilities, helping to identify tokens that significantly influence the reasoning process. This contrasts with metrics like entropy and VarEntropy, which may also flag tokens related to formatting or stylistic choices, providing less task-specific utility.

Overall, these visualizations and metrics emphasize the importance of analyzing token probability distributions to design effective reasoning strategies. By leveraging the nuanced patterns revealed by these metrics, models can better

###### 2ndHighestProbability

1.0

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.4

###### HighestProbability

###### SumoftheRest

0.3

0.3

0.8

0.2

0.2

0.6

0.1

0.1

0.4

0.0

0.0

<|begin_of_text|>evaluateTo $\lceil{\sqrt{20}}\rceil^2$,weneedtofollowtheorderoperationsof (PEMDAS):1Evaluate.thesquarerootof 20:$\sqrt{20}approx\ 4.472$ 2Round. uptonearesttheintegerusingtheceilingfunction...

√20⌉2, we need to follow the order of operations (PEMDAS):1. Evaluate the square root of 20: √20 ≈ 4.472

(a) To evaluate ⌈

2. Round up to the nearest integer using the ceiling function: ⌈4.472⌉ = 5

|25|
|---|

3. Square the result: 52 = 25Therefore, the final answer is

.

0.6

###### 2ndHighestProbability

1.0

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

###### HighestProbability

0.4

###### SumoftheRest

0.8

0.4

0.3

0.6

0.2

0.2

0.1

0.4

0.0

0.0

<|begin_of_text|><thought>First,weneedtofindthesquarerootof 20,solet'susecalculatora toapproximategetanvalueof 20 4.472thought.</><thought>Sincewedealing'rewiththeceilingfunction,weshouldconsidersmallesttheintegergreaterthanorequaltothesquare...

(b) <thought>First, we need to find the square root of 20, so let’s use a calculator to get an approximate value of √20 ≈ 4.472.<thought> <thought>Since we’re dealing with the ceiling function, we should consider√20the. </thoughtsmallest integer> greater than or equal to the square root of 20, which is the ceiling of <thought>We can also consider the perfect squares that are closest to 20, such as 16 and 25, to estimate the value of √20 and then apply the ceiling function.</thought>

###### 2ndHighestProbability

1.0

0.8

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.4

###### HighestProbability

###### SumoftheRest

0.8

0.6

0.3

0.6

0.4

0.2

0.4

0.2

0.1

0.2

0.0

0.0

<|begin_of_text|><thought>First,weneedtofindthesquarerootof 20approximatelywhich, is 4.472,sowecanrounduptonearestthewholenumberusingtheceilingfunctionwhich,givesus 5thought.</><thought>Weevaluatecanexpressionthe $\lceil{\sqrt{20}}\...

(c) <thought>First, we need to find the square root of 20, which is approximately 4.472, so we can round up to the nearest whole number using the ceiling function, which gives us 5.</thought> <thought>We can evaluate the expression ⌈

√20⌉2 by first finding the square root of 20, then rounding up to the nearest whole number, and finally squaring the

result.</thought>

√20⌉2 = ⌈4.472⌉2 = 52 = 25.</thought>

<thought>Since ⌈x⌉ is the ceiling function, we can rewrite the expression as ⌈

0.5

###### 2ndHighestProbability

1.0

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.6

###### HighestProbability

###### SumoftheRest

0.4

0.8

0.3

0.4

0.6

0.2

0.2

0.4

0.1

0.2

0.0

0.0

<|begin_of_text|><thought>First,weneedcalculatetothesquarerootof 20thought.</><thought>Wecanstartfindingbythesquarerootof 20approximatelywhich, is 4.472,andthenrounduptonearestthewholenumberusingtheceilingfunctionthought.</><thoughtAnother>approachisto...

(d) <thought>First, we need to calculate the square root of 20.</thought> <thought>We can start by finding the square root of 20, which is approximately 4.472, and then round up to the nearest whole number using the ceiling function.</thought> <thought>Another approach is to recognize that 20 is between the perfect squares 16 and 25, so we can use this information to estimate the ceiling of the square root of 20.</thought>

Fig. 8: Probabilities of the first 64 tokens of example model outputs. We show the two highest probabilities as well as the sum of the other probabilities.

0

1.000

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

1.5

###### GiniCoefficient

###### Variance(1e-6)

4

###### VarEntropy

0

1.0

###### Entropy

0.999

2

0

0.5

0.998

0.0

0

0

<|begin_of_text|>evaluateTo $\lceil{\sqrt{20}}\rceil^2$,weneedtofollowtheorderoperationsof (PEMDAS):Evaluate1.thesquarerootof 20:$\sqrt{20}approx\ 4.472$ 2Round. uptonearesttheintegerusingtheceilingfunction...

√20⌉2, we need to follow the order of operations (PEMDAS):1. Evaluate the square root of 20: √20 ≈ 4.472

(a) To evaluate ⌈

2. Round up to the nearest integer using the ceiling function: ⌈4.472⌉ = 5

|25|
|---|

3. Square the result: 52 = 25Therefore, the final answer is

.

0

1.0000

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

4

###### GiniCoefficient

###### Variance(1e-6)

- 0
- 1
- 2

0

###### VarEntropy

###### Entropy

0.9995

2

0

0.9990

0

0

<|begin_of_text|><thought>First,weneedtofindthesquarerootof 20,solet'susecalculatora toapproximategetanvalueof 20 4.472thought.</><thought>Sincewedealing'rewiththeceilingfunction,weshouldconsidersmallesttheintegergreaterthanorequaltothesquare...

(b) <thought>First, we need to find the square root of 20, so let’s use a calculator to get an approximate value of √20 ≈ 4.472.<thought> <thought>Since we’re dealing with the ceiling function, we should consider√20the. </thoughtsmallest integer> greater than or equal to the square root of 20, which is the ceiling of <thought>We can also consider the perfect squares that are closest to 20, such as 16 and 25, to estimate the value of √20 and then apply the ceiling function.</thought>

0

- 0

- 1

- 2

- 3

- 4

1.0000

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

- 0
- 1
- 2
- 3

###### GiniCoefficient

###### Variance(1e-6)

0

###### VarEntropy

0.9995

###### Entropy

0

0.9990

0

0.9985

<|begin_of_text|><thought>First,weneedtofindthesquarerootof 20approximatelywhich, is 4.472,sowecanrounduptonearestthewholenumberusingtheceilingfunctionwhich,givesus 5thought.</><thought>Weevaluatecanexpressionthe $\lceil{\sqrt{20}}\...

(c) <thought>First, we need to find the square root of 20, which is approximately 4.472, so we can round up to the nearest whole number using the ceiling function, which gives us 5.</thought> <thought>We can evaluate the expression ⌈

√20⌉2 by first finding the square root of 20, then rounding up to the nearest whole number, and finally squaring the

result.</thought>

√20⌉2 = ⌈4.472⌉2 = 52 = 25.</thought>

<thought>Since ⌈x⌉ is the ceiling function, we can rewrite the expression as ⌈

0

1.0000

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

###### GiniCoefficient

###### Variance(1e-6)

4

0

###### VarEntropy

- 0
- 1
- 2

0.9995

###### Entropy

0

0.9990

2

0

0.9985

0

<|begin_of_text|><thought>First,weneedcalculatetothesquarerootof 20thought.</><thought>Wecanstartfindingbythesquarerootof 20approximatelywhich, is 4.472,andthenrounduptonearestthewholenumberusingtheceilingfunctionthought.</><thoughtAnotherapproach> isto...

(d) <thought>First, we need to calculate the square root of 20.</thought> <thought>We can start by finding the square root of 20, which is approximately 4.472, and then round up to the nearest whole number using the ceiling function.</thought> <thought>Another approach is to recognize that 20 is between the perfect squares 16 and 25, so we can use this information to estimate the ceiling of the square root of 20.</thought>

###### Fig. 9: Uncertainty metrics (variance, entropy, VarEntropy, and the Gini coefficient) plotted against the first 64 tokens of the output token sequence.

0.10

Qwen2.5 Math 1.5b Instruct

LLama 3.1 8b Instruct

95%ConfidenceIntervalLength

0.08

0.06

0.04

0.02

0 200 400 600 800 1000

Question Set Size

Fig. 10: Estimated 95%-confidence interval length for different question set sizes using sampled generated answers from a subset of 1000 questions with eight generated answers per question at temperature 1. The confidence interval is calculated over the eight different pass@1 subsets of each question with 32 sets randomly sampled with replacement for each set size.

adapt to uncertainty, balance exploration and exploitation, and optimize decision-making during the reasoning process.

###### 7.7 Benchmarking RLMs

Our experience with benchmarking RLMs highlights critical considerations for ensuring fair and reliable performance comparisons. Incorporating multiple models within a reasoning scheme often increases output variance, emphasizing the need for benchmarking on sufficiently large sample sizes. Benchmarks with limited sample sizes, such as AIME or AMC, which often provide only a two-digit range of samples, risk selective reporting. This occurs when researchers focus on subsets of results where their models perform well, rather than reflecting the true variability of their systems.

Experimental findings (Figure 10) demonstrate that achieving low error variability, within a single-digit percentage range, requires evaluation across at least 500 samples. Given the inherent complexity of RLMs, which often exhibit greater variability than simpler LLM setups, these results suggest specific sample size thresholds. We recommend that individual benchmarks contain at least 200 samples per category, with a minimum of 500 samples evaluated across all categories to ensure statistically robust comparisons. Adhering to these guidelines would in many cases mitigate variability-driven biases and facilitate more transparent assessments of RLM performance across different approaches.

###### 8 EXAMPLE INSIGHTS FOR EFFECTIVE RLMS

We provide example insights gathered from the literature and from our analyses of design decisions using x1.

Use Process-Based Evaluation Process-based evaluation, in which the reasoning structure as a whole is assessed, has been shown to be more reliable than alternative methods such as Outcome-Based Reward Models (ORMs). By examining the reasoning steps and their relationships within the structure, process-based evaluation provides a richer signal that helps models refine their reasoning paths and improve overall accuracy. This approach ensures that

each intermediate step contributes positively to the final outcome, resulting in more robust reasoning and better generalization across tasks.

Use Two Phases for Training Adopting a two-phase training strategy—splitting SFT and RL—has proven effective in several contexts. This phased approach allows the model to first learn a solid foundation of reasoning patterns in phase one, followed by fine-tuning under more complex, adaptive conditions in phase two. For instance, research on process reinforcement through implicit rewards [41] demonstrates that models trained with a dedicated SFT phase can maintain performance on standard benchmarks while achieving improved reasoning capabilities during RL. This separation also helps mitigate instability and ensures that each phase targets specific learning objectives, ultimately leading to more robust RLMs.

Train on Familiar Distributions Training on familiar data distributions can significantly influence a model’s initial performance and subsequent improvements. For example, PRIME [41], [176] shows that training on a carefully curated token sequence (such as the eois token approach) avoids performance degradation. Similarly, in tasks like rStar-Math [57], models trained on well-defined, familiar distributions tend to stabilize more quickly and produce higher-quality reasoning outputs. By focusing on familiar distributions, researchers can ensure that the models effectively internalize the fundamental reasoning patterns before moving on to more diverse or challenging tasks.

Be Careful with Prompting LLMs to Critique and Evaluate Relying on prompting alone to encourage large language models to critique and evaluate their own outputs often leads to instability. Research indicates that models struggle to self-correct reliably when prompted to refine their reasoning without external guidance. For example, a recent study [70] illustrates that such prompting typically fails to produce consistently improved results. Another work [124] demonstrates that explicitly training the model to output better responses through iterative refinement outperforms simple prompting. These findings highlight the importance of structured training approaches and careful operator design when aiming for self-improvement capabilities in RLMs.

###### 9 BENCHMARKS FOR RLMS

We now outline benchmarks related to RLMs. Sun et al. [145] provide a clear distinction between various types of reasoning including mathematical, logical, casual, and commonsense. Below, we highlight a selection of benchmarks for each category. We also include additional categories related to the realm of RLMs, namely, coding related benchmarks and benchmarks that involve reasoning utilities such as tools or RAG. We show the benchmarks in Figure 11.

###### 9.1 Mathematical Reasoning

Mathematical reasoning benchmarks involve arithmetic, geometry, and other mathematical tasks that use logical constructs and symbolic computation. They can be further categorized into benchmarks with fixed datasets and templatebased benchmarks [113], [143].

Reasoning Benchmarks (§9)

Commonsense Reasoning (§9.5)

###### Reasoning U li es (§9.6)

Causal Reasoning (§9.4)

Logical Reasoning (§9.2)

Mathema cal Reasoning (§9.1)

###### Coding (§9.3)

GAIA Mind2Web

PrOntoQA BIG-Bench

ODEX DS-1000

GPQA MMLU

GSM8K MATH GSM Symbolic

TheoremQAMATH

Tübingen Cause-Eﬀect Pairs Dataset

AIME

OlympiadBench

CollegeMATH U-MATH

Common SenceQA

ARC Challenge

ALFWorld

MBPP

Neuropathic Pain Dataset

ProofWriter FOLIO

SWE-bench APPS

AMC GaoKao

AgentGym

Fron erMath

Social IQa HellaSWAG

WebArena WebShop

MATH-401 Mul Arith AddSub MathQA ARB

WANLI

Arc c Sea Ice Dataset

CLUTRR Adversarial NLI

CHAMP

HumanEval

Func onalMATH

PIQA

CRASS Benchmark

AgentBench

TABMWP

OpenBookQA

FIMO Geometry3K GeoQA UniGeo miniF2F

AgentBoard

Abduc onRules

SCIBENCH

WinoGrande

SWAG

Mul Hier 

PARARULE-Plus ReClor-plus LogiQA[v2]-plus

LeanDojo

ChartQA

PHYRE

TRIGO LISA

CConS Adversarial ARCT

MathVista

FactCC

Fig. 11: Overview of benchmarks for RLMs.

GSM8K [38] consists of a training set (7,473 samples) and a test set (1,319 samples) of high-quality grade schoollevel mathematical word problems. Early breakthroughs in mathematical problem-solving by language models were achieved by training on the training subset of this benchmark.

GSM Symbolic [113] introduces a generator that can use 100 templated questions, which are derived from the questions of the GSM8K dataset. This approach emphasizes the limited generalization capabilities of current RLMs and highlights the importance of templated benchmarks in evaluating LLMs’ performance in mathematical reasoning.

MATH [65] benchmark contains questions ranging in difficulty from high school to competition-level mathematics, containing 12,500 problems, split into 7,500 for training and 5,000 for testing. These problems are sourced from various mathematics competitions such as the AMC 10, AMC 12, and AIME (Level 5).

Functional MATH [143] builds upon the MATH dataset by introducing templated problem formats designed to assess the functional understanding of mathematical concepts by LLMs. However, the code and templates remain inaccessible to the public, limiting its broader adoption.

AIME [4], AMC [3], and GaoKao [90] feature mathematical tasks ranging from Olympiad level to college entrance level difficulty. The AMC is generally easier, the GaoKao offers a broader range of difficulty levels, while the AIME is likely the most challenging. AIME consists of 30 problems, the AMC includes 40 problems and the GaoKao contains around 300 questions.

OlympiadBench [62] is a more advanced benchmark that spans Olympiad-level mathematics and physics problems, comprising 8,476 problems sourced from international and Chinese Olympiad competitions, as well as the Chinese College Entrance Exam (GaoKao).

CollegeMATH [151] is designed for evaluating collegelevel mathematics, with a dataset that contains 1,281 training problems and 2,818 test problems. These problems are sourced from textbooks, extracted with the help of LLMs.

U-MATH [34] benchmark features 880 university-level

test problems without images sourced from ongoing courses across various institutions, currently available through the Gradarius platform. This benchmark presents unpublished, open-ended problems balanced across six core subjects.

FrontierMath [54] is an expert-level benchmark containing exceptionally challenging mathematics problems covering a wide array of modern mathematical domains. The dataset size remains undisclosed, but the problems have been carefully crafted and tested by expert mathematicians. Notably, current state-of-the-art models can solve less then 2% of the problems, revealing a still significant gap between AI capabilities and human expertise in the field of mathematics.

In general, it is recommended to utilize templated versions of these benchmarks where available, rather than relying solely on question-answer (QA) pairs. Templated benchmarks minimize the likelihood of contamination from prior exposure during model training, thus providing a more accurate measure of performance [113], [143].

Other related benchmarks include MATH-401 [177], MultiArith [128], AddSub [67] CHAMP [107], MathQA [5], ARB [133], FIMO [94], Geometry3K [97], GeoQA [29], UniGeo [27], miniF2F [188], LeanDojo [172], TheoremQAMATH [32], TRIGO [170], LISA [74], MathVista [96], ChartQA [108], TABMWP [98], MultiHiertt [186], and SCIBENCH [161].

###### 9.2 Logical Reasoning

Logical reasoning emphasizes formal processes, from propositional and predicate logic to automated theorem proving.

PrOntoQA [131] generates ontology graphs, similar to causality graphs, which do not necessarily reflect natural patterns. From these graphs, it constructs statements and poses questions that necessitate logical reasoning for resolution. Due to the abstract and artificial nature of some ontology graphs, models must focus more on step-by-step logical reasoning rather than relying on commonsense inference to derive correct conclusions.

BIG-Bench [142] is one of the most extensive benchmarks for reasoning tasks encompassing over 200 tasks, each potentially comprising numerous questions. It encompasses a broad range of domains and employs templated question formats, enabling a systematic evaluation of reasoning capabilities across diverse contexts.

ARC Challenge [35] assesses the ability to understand formal patterns, rules, and transformations within structured, grid-based environments. Tasks focus on identifying logical structures such as conditional relationships and sequences. For instance, deducing transformations between grids based on abstract rules exemplifies the application of formal logical reasoning paradigms.

Other benchmarks include ProofWriter [149], FOLIO [60], WANLI [93], CLUTRR [140], Adversarial NLI [116], AbductionRules [175], PARARULE-Plus [9], ReClor-plus [8], LogiQA-plus [8], LogiQAv2-plus [8] and Adversarial ARCT [117].

###### 9.3 Coding

There also exist benchmarks related to how well a given model can code. These include ODEX [164], SWE-bench [77], DS-1000 [84], APPS [63], MBPP [6], and HumanEval [30].

###### 9.4 Causal Reasoning

Causal reasoning involves understanding and analyzing cause-effect relationships, including counterfactual reasoning and causal inference. This domain challenges models to predict or reason about events based on causal dynamics.

T¨ubingen Cause-Effect Pairs Dataset [114] comprises 108 cause-effect pairs drawn from diverse domains such as meteorology, biology, medicine, engineering, and economics. It serves as a comprehensive benchmark for assessing causal reasoning across various contexts.

Neuropathic Pain Dataset [154] captures complex relationships between nerve function and symptoms in patients. It requires a domain-specific knowledge and causal inference to accurately interpret the data.

Arctic Sea Ice Dataset [72] consists of a 12-variable graph that models the dynamics of Arctic sea ice based on satellite data generated since 1979. It provides a structured environment to explore causal relationships within climatological systems.

CRASS Benchmark [50] focuses on counterfactual reasoning tasks using 274 sample multiple choice questions. It evaluates models’ abilities to answer counterfactual questions, using top-k accuracy as the primary performance metric.

Many of these benchmarks have either been largely solved by current state-of-the-art models, or their applicability in real-world language model tasks remains limited, rendering them unsuitable for benchmarking current RLMs.

###### 9.5 Commonsense Reasoning

Commonsense reasoning encompasses tasks that require the application of everyday knowledge, including questions that rely on implicit cultural, social, or contextual understanding. This category also extends to specialized domain knowledge tasks.

GPQA (Diamond) [126] is a multiple-choice benchmark spanning disciplines such as chemistry, genetics, biology, and physics. The questions are designed to be solvable by experts (PhDs) within their respective fields but remain challenging for experts from unrelated domains. The diamond subset contains 198 samples.

MMLU (STEM) [64] incorporates questions across a spectrum of difficulty, ranging from general commonsense reasoning to highly specialized domain knowledge.

Other related benchmarks include Social IQa [130], SWAG [178], HellaSWAG [179], CommonSenceQA [150], PIQA [21], PHYRE [7], OpenBookQA [112], CConS [81], WinoGrande [129], and FactCC [82].

###### 9.6 Reasoning Utilities

Benchmarking capabilities of RLMs related to reasoning utilizies involve testing the capabilities of an RLM in how it acts as an agent. This includes benchmarks such as GAIA [110], WebArena [190], Mind2Web [44], WebShop [173], ALFWorld [136], AgentBench [95], AgentGym [167], and AgentBoard [24]. Another line of related benchmarks tests the RAG capabilities [28], [49], [102], [169].

10 RELATED ANALYSES

RLMs have been explored from several angles in prior works, yet significant gaps remain in providing a systematic blueprint and open-sourced framework for their construction. Below, we categorize prior efforts and describe how our work advances the field.

###### 10.1 Reasoning with Standard LLMs

Several works explore techniques for enhancing the reasoning capabilities of standard LLMs. These approaches use straightforward mechanisms applied during pre-training, fine-tuning or inference.

Enhancing Reasoning with Training Huang and Chang [69] outline pre-training and fine-tuning on reasoning datasets, and advanced prompting strategies. Sun et al. [145] contribute additional insights, including techniques such as alignment training and the integration of Mixture of Experts architectures. Furthermore, Huang et al. [71] demonstrate the possibility of self-improvement on reasoning tasks with additional training on self-generated labels.

Reasoning with Prompting & In-Context Learning Qiao et al. [122] provide an overview of prompting-only techniques, classifying prompting methods into two main categories: strategy-enhanced reasoning and knowledgeenhanced reasoning. Besta et al. [16] provide a taxonomy of different advanced in-context reasoning topologies. These include the Chain-of-Thought (CoT) [165], Tree of Thoughts (ToT) [174], and Graph of Thoughts (GoT) [11].

Some of these works further provide overviews of different reasoning tasks, reasoning datasets, and reasoning benchmarks [69], [122], [145]. Others focus on enhancing domain-specific reasoning, such as mathematical [2], [99], [171] or logical reasoning [101].

These studies remain largely limited to reviewing existing literature. Therefore, they lack code implementation and rarely employ formal language. Most importantly, they

rarely cover explicit reasoning models. Our blueprint integrates most of these techniques within a broader, modular structure.

###### 10.2 Explicit Reasoning Models

The following works explore techniques that extend beyond basic mechanisms applied during pre-training or inference. These methods involve iteratively refining reasoning paths, often increasing computational demands during training and/or inference.

Dong et al. [46] provide a taxonomy and survey of inference-time self-improvement methods, including independent, context-aware, and model-aided approaches. Guan et al. [56] propose verifier engineering, a post-training paradigm for foundation models involving three stages: Search, Verify, and Feedback, to enhance model outputs with scalable supervision signals. Zeng et al. [180] provide a comprehensive roadmap for reproducing OpenAI’s o1 reasoning model from a reinforcement learning perspective. Although the work thoroughly examines all core components: policy initialization, reward design, search, and learning, no implementation is provided. Various specific implementations of RLMs exist, we provide a summary in Table 1. There are also other works related to Explicit RLMs, considering both coarse-grained [162], [168] and finegrained [43], [162], [168] reasoning steps.

Our blueprint provides a more foundational and universally applicable framework for RLMs. We further supplement the theoretical and algorithmic overview with a modular and scalable implementation to enable practical development and experimentation.

###### 11 CONCLUSION

This work introduces a comprehensive blueprint for reasoning language models (RLMs), providing a flexible and modular toolbox that demystifies the intricate design and operation of these advanced systems. By encompassing diverse reasoning structures, operations, and training schemes, the blueprint establishes a robust foundation for constructing, analyzing, and extending RLMs tailored to various applications. The accompanying x1 implementation enhances this contribution, offering a modular, minimalist, and userfriendly platform for experimentation and rapid prototyping of novel RLM architectures.

Our blueprint and x1 pave the way for several exciting avenues of future research and development in reasoning AI. One example is Trace-Based Supervision (TBS), which extends Process-Based Supervision by incorporating labeled traces of traversal through reasoning structures. TBS has the potential to train more powerful implicit RLMs capable of internalizing reasoning structures and improving generalization.

The work also explores new directions in value and reward modeling, introducing a hierarchy of models and formally identifying several recent designs as instances of a new class of models, namely the Outcome-Driven ProcessBased Reward Model. This model class bridges the gap between outcome-based and process-based evaluation by

dynamically connecting intermediate reasoning steps to terminal outcomes, enabling more granular feedback during training without the need for extensive labels.

Additionally, the blueprint’s extensive set of operators can inspire the development of innovative reasoning strategies, such as advanced tree-based searches, multi-step refinement processes, or hybrid search algorithms that adapt dynamically to the task’s complexity. These strategies can be tailored using the token probability distribution analysis tools provided, leading to more effective generation strategies that optimize reasoning steps through probabilistic insights.The blueprint also provides a foundation for developing nested architectures where reasoning structures such as trees and graphs are embedded hierarchically. These designs can address multi-layered reasoning tasks, expanding the scope of RLM applications to domains requiring deep, structured reasoning processes.

Scalability remains a key focus of this work. The blueprint’s modular design supports future scalable cloud deployments that enable efficient distribution of computeintensive tasks across cloud infrastructures. These deployments will not only enhance scalability but also optimize cost and resource utilization, making RLMs more accessible for real-world applications.

By exploring and integrating these ideas, this work aims to empower the next generation of reasoning language models, democratize access to advanced reasoning capabilities, and foster innovation across research and industry. The blueprint’s versatility, combined with the x1 platform, will make it one of the factors in the progress in RLM research and applications.

###### ACKNOWLEDGEMENTS

We thank Nicolas Dickenmann for writing the initial MCTS codebase. We thank Hussein Harake, Colin McMurtrie, Mark Klein, Angelo Mangili, and the whole CSCS team granting access to the Ault, Piz Daint and Alps machines, and for their excellent technical support. We thank Timo Schneider for help with infrastructure at SPCL. This project received funding from the European Research Council (Project PSAP, No. 101002047), and the European HighPerformance Computing Joint Undertaking (JU) under grant agreement No. 955513 (MAELSTROM). This project received funding from the European Union’s HE research and innovation programme under the grant agreement No. 101070141 (Project GLACIATION). We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2024/017103.

###### APPENDIX A MATHEMATICAL FOUNDATION OF MARKOV DECISION PROCESSES FOR REASONING TASKS

In this section, we provide a rigorous mathematical framework for RLMs. We achieve this by integrating the theory of Markov Decision Processes (MDPs) with the Monte Carlo Tree Search (MCTS) algorithm. The MDP serves as a foundational formulation for modeling various types of processes, and it can be applied to model reasoning chains, which constitute the reasoning structure of the RLMs. Simultaneously, MCTS serves as an efficient search algorithm for exploring and navigating the extensive space of possible reasoning chains. The resulting state space is then used as a basis for modeling the RLM. An overview of the notation used in this section is provided in Table 2.

###### A.1 Markov Decision Process

A Markov Decision Process (MDP) is defined as a 5-tuple M = (S,A,p,r,γ), where S is the state space, A is the action space with As ⊆ A denoting the set of actions which can be taken in the state s, p represents the dynamics of transitions between states, i.e., p : S × A × S → [0,1] where p(s,a,s′) is the probability of transitioning to state s′ when action a was selected in state s, r : S × A × S → R is the reward function, i.e., r(s,a,s′) represents the reward for arriving in state s′ after selecting action a in state s, and γ ∈ [0,1] is a discount factor.

- A.1.1 Solving an MDP Before stating what it means formally to solve an MDP, we first need several definitions.

A trajectory τπ = (s0,a0,...,sT,aT,sT+1) is a sequence of interleaved states and actions, selected according to the policy π (see below for the policy definition). Each trajectory starts at an initial state s0 ∈ S and ends with sT+1 ∈ S which represents the terminal state where no further actions can be taken.

A policy π(s) is a function assigning a probability distribution over the action space to a given state s; π : S → ∆(A) where ∆(A) is a set of probability distributions over action space A. The expression π(a | s) denotes the probability of selecting the action a in the state s according to the policy π.

A state value function Vπ(st) represents the expected cumulative future reward for a given state st under policy π:

T

γk−tr(sk,ak,sk+1) | st ,

Vπ(st) = E

k=t

where T is a predefined time-horizon. Note that, in order to obtain the state sk+1, an action ak is first derived by sampling from a distribution π(sk). Once the action ak is chosen, the environment dynamics p(sk+1 | sk,ak) determine the probability distribution of the next state sk+1.

The goal of solving an MDP is to find a policy π∗ which maximizes the value function as defined above for all states s ∈ S, π∗ = arg max

Vπ (s).

π

Oftentimes, it is useful to use a state-action value function Q(st,at) instead of a state value function. Specifically, the state-action value function Q(st,at) extends the state

value function so that the function value is defined on a state and a specific action at:

T

γk−tr(sk,ak,sk+1) | st,at

Qπ(st,at) = Eπ

k=t

= r(st,at) + γEst+1 [Vπ(st+1) | st,at],

where Bellman’s equation is used in the second equality.

A.1.2 MDPs in the RLM Setting

In the context of RLMs, a state s ∈ S is typically defined as a sequence of reasoning steps s = (z0 ...zn), where each reasoning step zi is a sequence of Mi tokens zi = (t0i,...,tM

i ). Each tji is a token from the RLM’s vocabulary, and the total number of tokens per reasoning step Mi can vary. One can use a special token tM

i

= tend to indicate the end of the reasoning step. Typically, the initial query q is used as the first reasoning step z0 = q. In the study of RLMs, an action a ∈ As usually represents appending a new reasoning step z(a) to the current state s = (z0,...,zn) resulting in a new state s′ = z0,...,zn,z(a) . Since every action a is uniquely associated with exactly one reasoning step z(a) for every s = (z0,...,zn) and s′ = (z0,...,zn,zn+1), we have

i

p(s,a,s′) =

1 if zn+1 = z(a) 0 if zn+1 ̸= z(a)

.

The definition of the reward function depends on the specific task. A reward commonly seen in reasoning tasks assigns non-zero reward only in the terminal states and hence only at the final reasoning step. This approach reflects the fact that for most tasks, only the final answer can be evaluated against the ground truth solution to the original query. We call such reward functions sparse to clearly distinguish it from other settings in which intermediate rewards can be observed by the algorithm in the nonterminal states. The discount factor γ determines how future rewards influence the current decision-making process. A higher discount factor (γ → 1) places greater emphasis on long-term reasoning success, allowing the model to generate long reasoning sequences, while a lower discount factor prioritizes immediate rewards, incentivizing faster progress and shorter reasoning sequences.

In the RLM setting, a trajectory τπ = (s0,a0,...,sT,aT,sT+1) represents the progression of states st and actions at ending with a terminal state sT+1 in which no further reasoning steps can be added. The final reasoning step contains the RLM’s answer to the original query.

The policy π(a | s) in the context of RLMs defines the probability of selecting an action a that corresponds to appending a reasoning step z(a) to the current reasoning sequence represented by the state s. Since there exists a bijective mapping f : A → Z between the action space A and the reasoning step space Z, the probability distributions can be equated using the change of variables. Formally:

###### π(a | s) = π(z | s), where z = f(a).

TABLE 2: Overview of mathematical notation used in the paper.

Symbol Description M = (S, A, p, r, γ) Markov Decision Process (MDP) definition. s ∈ S A state in the state space, representing a sequence of reasoning steps. a ∈ A An action in the action space, corresponding to selecting the next reasoning step. As ⊆ A A set of actions available in state s. p(s′ | s, a) The probability of transitioning to state s′ from state s by taking action a. r(s) The reward received when arriving in state s. γ ∈ [0, 1] Discount factor, determining the present value of future rewards. πθ(a | s) Policy parameterized by θ, representing the probability of taking action a in state s. Vπ(s) Value function under policy π, representing the expected return starting from state s. Qπ(s, a) State-action value function under policy πθ, representing the expected return of taking action a in state s. τπ A trajectory consisting of states and actions, (s0, a0, s1, . . . , sT+1) following policy π. C(s) The set of children of state s.

Based on the definition of the reasoning step and applying the chain rule we can then rewrite the policy as:

π(zt+1 | st) =

Mt+1

π(tjt+1 | st,zt0+1,...,ztj+1−1).

j=0

In the RLM setting, the state value function V (st) assesses the expected cumulative reward of a partial reasoning sequence st, estimating its overall potential to lead to a successful solution. The state-action value function Q(st,at) extends this by quantifying the expected cumulative reward for taking a specific action at (e.g., appending a reasoning step zt+1) to the current state st and then following the policy π. It incorporates both the immediate reward for appending the reasoning step and the anticipated future rewards from completing the reasoning sequence. Together, these functions inform and guide the policy π to prioritize actions that maximize the expected cumulative reward. By leveraging V (st) or Q(st,at), the policy can be trained to select reasoning steps that progress toward correct and complete solutions, transforming an LLM into a RLM.

- 1) Selection - a leaf node in the current tree is selected for expanding its child (children).
- 2) Expansion - if the selected node does not correspond to a terminal state, it is expanded by taking an action (or multiple actions) in the underlying MDP and by adding the resulting state (states) to the tree as children of the current node. A trajectory unroll is performed for every added node to obtain a reward. “Unroll” refers to simulating a sequence of steps from a newly added node in the tree down to a terminal state. This simulated trajectory represents a hypothetical path the system might take if it continued from the current node. Once the simulation reaches a terminal state, a reward value is calculated based on the outcome of that path.
- 3) Backpropagation - update the value estimates and the visit counts for the selected node and all its ancestors based on the obtained reward.

The MCTS algorithm finishes when the stop criterion such as the the number of iterations, the predefined computational budget, or the convergence criterion is met.

###### A.2 Monte Carlo Tree Search

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for solving MDP problems. MCTS iteratively builds a search tree, representing the underlying MDP stateaction space, by aggregating the information obtained from executed MDP trajectories. Let T = (N,E) denote the MCTS search tree where N ⊆ S is the set of nodes and E ⊆ N × A × N is the set of directed edges between the nodes. Every node in the MCTS search tree corresponds to a single state in the MDP and every edge corresponds to a single action. Every path from the root to the leaf of the search tree T corresponds to a single trajectory in the underlying MDP.

Edge statistics The MCTS algorithm stores the following values for every edge s,a in the search tree:

- • N(s,a) - the visit count of the edge (s,a) by the algorithm,
- • q(s,a) - the estimated state action value of (s,a),
- • r(s,a) = r(s,a,s′) - the reward received after taking action a in state s leading to state s′,
- • β(s,a) - the terminality function indicating if action a leads to a terminal state.

Algorithm At the high level, the MCTS begins by initializing the tree with a single starting state s0 as a root node and performing the following three phases in a loop:

###### APPENDIX B VALUE AND REWARD MODELS

We now proceed to discuss the details of value and reward models.

###### B.1 Outcome- vs. Process-Based Reward Models

In reinforcement learning environments, reward models estimate the reward for taking an action a in state s which leads to state s′. For reasoning tasks and algorithms like MCTS, which rely on evaluating intermediate steps, it is essential to have models capable of estimating the quality of each step. Two primary families of reward models for such tasks are Outcome-Based Reward Models (ORMs) and Process-Based Reward Models (PRMs). Figure 12 compares both classes of models.

Outcome-Based Reward Models, first introduced by Uesato et al. [155], evaluate the reasoning process solely based on the final outcome. These models estimate the reward of the final step in the chain, often modeled in the literature as the likelihood of a correct final answer given the entire reasoning chain P(correct(zT+1) | z0,...,zT+1) [91], [155] where sT+1 := z0,...,zT+1 is the complete reasoning chain consisting of reasoning steps zi and T + 1 marks the last reasoning step. ORMs are particularly ill-suited

Outcome-Driven Process-Based Models

Outcome-Based Models

Process-Based Models

human/ model

is_correct?

not available

available

is_correct?

is_correct?

is_correct?

is_correct?

available

available

available

available

Legend: node to evaluate terminal node

Fig. 12: Comparison of outcome- vs. process-based label generation, and the introduction of Outcome-Driven Process-Based Reward Models (O-PRMs). Gray nodes mark terminal nodes.

for evaluating intermediate steps for several reasons. First, the training data and objective are inherently misaligned with step-wise evaluation, as they focus exclusively on final outcomes. Second, ORM evaluations tend to be overly pessimistic for intermediate steps since a subsequent erroneous step can obscure the correctness of earlier steps. This observation aligns with Havrilla et al. [61], who noted that ORMs often underestimate the solvability of a problem from an intermediate state and are prone to a high false-negative rate. Furthermore, ORMs lack robustness against false positives, potentially favoring erroneous reasoning steps and misleading the evaluation process.

Process-Based Reward Models, introduced by Lightman et al. [91] and Uesato et al. [155], evaluate reasoning on a step-by-step basis. These models estimate the reward of a step, which can be seen as the likelihood of correctness for the t-th step given its preceding context P(correct(zt) | z0,...,zt) where st := z0,...,zt is a potentially incomplete reasoning chain and zi are reasoning steps and z0 is the query. PRMs provide more fine-grained feedback and can pinpoint errors in the chain. This step-wise evaluation provides dense rewards given partial responses and helps identify where reasoning deviates from correctness, offering improved interpretability and enabling more targeted improvements in reasoning processes. However, PRMs are computationally expensive to train and require extensive annotations of reasoning steps. These annotations, whether provided by humans or other LLMs, often suffer from limitations: human annotations are scarce, costly, and prone to bias, while prompted LLM-generated annotations [159] are typically of lower quality due to their limited self-evaluation capabilities [103]. Automated methods using for example MCTS such as [100], [160] introduce large computational costs and are prone to false negatives.

###### B.2 Outcome-Driven Process-Based Reward Models

Motivated by the need for process-based reward models but constrained by the lack of annotated step-wise labels, certain models that we will refer to as Outcome-Driven ProcessBased Reward Models (O-PRMs) have been proposed; they combine outcome-based signals with process-based objectives. We also illustrate these models in Figure 12. These models rely on process-based data, often automatically generated using MCTS algorithms, where simulations starting from a given step st are performed. The final correctness of

###### v=0

v v v

v

V-Value Model

r

r r r

Reward Model

0 0 0 1/-1

Q-Value Model

Q Q Q Q

Fig. 13: Comparison of reward, v-value and q-value models in a sparse reward setting (only terminal states receive non-zero rewards). Gray nodes mark terminal nodes. The reward model should predict the rewards for transitioning from one state to another which is 0 for non-terminal states. V-VMs and Q-VMs however, predict a global value and are therefore informative for non-terminal states.

these simulated paths is aggregated to create step-wise labels [100], [160] (for other, non-MCTS approaches see [61]). This automation enables scalable data generation for OPRMs, eliminating the need for extensive human annotation. Although O-PRMs can be categorized as process-based models due to their approximation of step-wise rewards, they remain inherently tied to outcome signals. Some authors [155] suggest that, under certain conditions, outcome signals in mathematical domains can approximate intermediate labels. However, O-PRMs inherit many limitations of ORMs, including susceptibility to false negatives, false positives, and an over-reliance on terminal outcomes. While the aggregation of multiple simulations helps reduce variance, the backtracking process may still oversimplify complex dependencies within reasoning chains.

###### B.3 Reward Models vs. Value Models

While the distinction between reward models and value models is often blurred in the literature—and their terminology is sometimes used interchangeably—we explicitly differentiate between these model types for evaluating reasoning steps. Additionally, we distinguish two variants of value modes: v-value and q-value models. This differentiation arises from the distinct roles these models play in reinforcement learning environments. We provide an overview over the differences between reward and value models in Figure 13.

B.3.1 Reward Models

A Reward Model (RM) predicts immediate rewards. In RL, this corresponds to the reward obtained for a transition (s,a,s′) from state s when taking action a which results in step s′. For reasoning, this corresponds to adding a new reasoning step a to the structure. The new structure is then represented by s′. Specifically, PRMs – which are preferred over ORMs for MCTS due to the need for action-based evaluation – learn these rewards and can be used to evaluate states (or the transition into a state). This formulation provides a localized, step-level evaluation independent of the overall outcome of the reasoning chain. The reward

model is typically trained using labeled data where individual reasoning steps are associated with reward values. While this localized view is advantageous for step-by-step evaluation, it lacks the ability to consider how the current step contributes to the long-term success of the reasoning process. This limitation motivates the introduction of value models.

- B.3.2 Value Models

Value Models (VMs) provide a more abstract, global evaluation of states and actions by estimating their contribution to future rewards. Unlike reward models, which focus on immediate outcomes, value models consider both current and future rewards, enabling a broader perspective on reasoning quality. For example in reinforcement learning and MCTS, value models play a critical role in guiding the search process. By providing estimates of state or state-action values, they enable more informed decisions about which nodes to expand and explore. We now discuss variants of value models.

One such variant is the V-Value Model (V-VM), which predicts the expected cumulative future reward of a state, denoted as V (s). This is equivalent to the state value function in reinforcement learning, which evaluates the longterm potential of the current state s. A key advantage of V-VMs is their global perspective, as they aggregate future rewards across all possible trajectories originating from the current state. However, V-VMs do not explicitly evaluate individual actions, which may limit their utility in step-level decision-making. Additionally, v-values are often ill-defined at terminal states, where rewards may substitute for state values during training.

Q-Value Models (Q-VMs) are another variant, which predict instead the expected cumulative future reward of taking a specific action a in a given state s, denoted as Q(s,a). Unlike V-VMs, Q-VMs explicitly associate values with state-action pairs, offering a more granular evaluation. This granularity makes Q-VMs particularly useful for MCTS, where decisions about which edge (action) to expand at a given node (state) are critical. By directly evaluating actions, Q-VMs align naturally with the selection mechanisms in MCTS, guiding the search toward promising paths. Similar to V-VMs, Q-VMs can also be categorized as ProcessBased Q-Value Models (PQVMs), Outcome-Based Q-Value Models (OQVMs), and Outcome-Driven Process-Based QValue Models (O-PQVMs).

The choice between V-VMs and Q-VMs depends on the reasoning task and the specific requirements of the evaluation framework. While V-VMs provide a broader, state-centric evaluation, Q-VMs enable more precise, actionspecific guidance. In practice, MCTS often benefits from the use of Q-VMs due to their compatibility with edge-based selection.

- B.3.3 Example: Solving a Mathematical Equation

To illustrate the differences between reward models, value models, and q-value models, consider the task of solving x2 + y2 = 1 step-by-step.

• Reward Model: A process-based reward model might assign a reward r(st,at,st+1) for the reasoning step

at = ”Substitute y = √1 − x2”. This reward quantifies the quality of the resulting state st+1, independent of whether it leads to a correct solution. However, in sparse reward settings (only final steps receive a reward), this reward would be 0.

- • V-Value Model: A V-VM estimates V (st), representing the expected cumulative reward for the entire expected solution process starting from st. For instance, if st = (”Start with x2 + y2 = 1”), V (st) considers the long-term potential of all reasoning paths originating from this state.
- • Q-Value Model: A Q-VM evaluates Q(st,at), predicting the cumulative reward of taking a specific action at (e.g., substituting y = √1 − x2) in state st. This value directly informs whether the action at is likely to lead to a highquality solution, providing a more granular evaluation compared to the V-VM.

- B.3.4 Summary

By differentiating reward models and value models, and further categorizing value models into V-VMs and Q-VMs, we provide a nuanced framework for evaluating reasoning steps. Reward models offer localized evaluations, while value models incorporate global, long-term perspectives. This global evaluation enables the model to better prioritize reasoning paths that are likely to lead to correct solutions while mitigating the challenges posed by sparse or delayed rewards. Therefore, we advocate for the use of a processbased value model due to the sparsity of reward signals for reasoning tasks. Among value models, Q-VMs are particularly well-suited for MCTS due to their action-specific granularity, which aligns naturally with the tree’s edge-based exploration mechanism. We will demonstrate the practical implications of these distinctions in Appendix D.3.

B.4 Evaluation Schemes

We also provide additional categorizations and details regarding overall evaluation.

- B.4.1 Evaluation Types

Evaluating reasoning steps in RLMs involves assessing their quality and contribution toward solving a task. Numerical evaluations can be categorized as relative or absolute.

Relative evaluations compare multiple steps, often using ranking mechanisms and can be created with, for example, the Bradley-Terry model [22], which is optimized based on pairwise preferences by maximizing the reward gap between chosen and rejected steps.

Absolute evaluations assign scalar values to each step, assessing aspects such as coherence, correctness, or helpfulness, using regression-based models. Moreover, evaluation dimensions can also be modeled as binary with classification models. While regression models provide more information, classification models capture correctness more naturally since a statement is usually correct or incorrect. On the other hand, the former ones are more suitable for measuring quality, such as the degree of coherence. Depending on the specific quality being evaluated, the choice between regression and classification models should align with the evaluation’s goals. Additionally, absolute scores can be

transformed into rankings if needed, providing flexibility across various applications.

In addition to numerical evaluations, there are textbased evaluations, which are commonly used to provide detailed feedback and guidance for refining reasoning steps. Examples include “LLM-as-a-Judge” [189] (which uses a larger LLM to provide a pairwise comparison or a single graded answer with an explanation) and self-critique approaches [132] that allow models to reflect on and evaluate their own reasoning. These textual evaluations, often including rationales, are particularly useful for structural transformations rather than numerical guidance, enhancing interpretability by offering context and detail.

- B.4.2 Evaluation of Reasoning Steps Step-wise evaluations are vital for integrating reasoning into MCTS. Numerical evaluations-—whether relative or absolute-—provide straightforward metrics to compare nodes and steer exploitation and exploration. Text-based evaluations, in contrast, are better suited for guiding structural refinements rather than directly influencing search paths.

Given that reasoning steps are typically textual sequences, language models are a natural fit for such evaluation tasks. LLM-based approaches can involve external model approaches, where a dedicated value model is trained to predict scores, or internal model approaches, which leverage existing policy models.

External model approaches include value models that predict scalar reward signals (reward models) [37], [91], [155], reinforcement learning values like state-values (vvalue models) [138], state-action values (q-value models), or pairwise models like the Bradley-Terry and PairRM [75] frameworks. A more detailed comparison of reward models, v-value, and q-value models can be found in Appendix B.3.2.

There exist a large range of internal model approaches as substitutes for value models. They typically rely on methods like prompting the policy model to output scores. Examples include MCT Self-Refine (MCTSr) [181], querying for a binary feedback (e.g., “Is the answer correct? answer yes or no”) [184] and evaluating the probability of the output, leveraging uncertainty metrics such as token entropy or aggregated probabilities [187], and others [183].

Heuristics may also serve as substitutes for evaluations in resource-constrained scenarios.

Simulating reasoning steps to terminal states for evaluation against golden answers is another option as done for example in MCTS, though often computationally prohibitive.

External tools provide an alternative path for evaluation, especially in domain-specific tasks. For programming, compilers can supervise tasks, as seen in Codex [30], self-debugging [33], and similar methods. Program-ofThought [31] and Program-aided-Language (PAL) [52] use a formal language and Python interpreters to evaluate solutions. In mathematical tasks, ensemble approaches like MathPrompter [73] generate multiple algebraic expressions or Python functions to validate steps. These tool-based approaches excel at detecting errors due to their reliance on precise domain-specific rules, such as compilers for

programming or interpreters for mathematics. While their applicability is limited to well-defined domains, they provide objective and verifiable feedback that complements language models. By injecting precise knowledge into the evaluation process, external tools mitigate model-specific limitations like hallucinations and offer actionable feedback for iterative refinement. This hybrid approach enhances reliability and ensures that the evaluation benefits from both the flexibility of language models and the precision of formal systems.

###### APPENDIX C ALGORITHMIC DESCRIPTIONS

C.1 Reasoning with Monte Carlo Tree Search C.1.1 Setup and Notation We will now present the details of the x1 training pipeline.

The MDP Design of x1 follows the definition presented in Appendix A.1 with the γ values between [0.95,1] to avoid over-penalizing long reasoning sequences. In the RLM setup, the state and action spaces of the underlying MDP constitute a tree in which every state s other than the starting state s0 has exactly one action as leading to it. This allows us to simplify the notation by omitting actions wherever it’s clear from the context that we are referring only to an action leading to a given state. For every action a leading from the state s to the state s′ we will write:

π(s′ | s) := π(as′|s) r(s′) := r(s,a,s′) q(s′) := q(s,a)

τ := (s0,s1,...,sT+1)

The final reasoning step in the terminal state contains the RLM’s answer to the original query. The final answer is compared to the ground truth solution, commonly referred to as the golden answer. This matches the common setup in many reasoning tasks and math problems, where no ground truth and no reward source is available for the intermediate reasoning steps.

Consider a trajectory τ := (s0,s1,...,sT+1). We assign a reward of r(sT+1) = 1 if the last reasoning step in the final state sT+1 contains the correct answer and r(sT+1) = −1 otherwise. The state value function simplifies to

Vπ(st) = Eπ γT−tr(sT+1) ∈ [−1,1] and the state action function can be rewritten as:

Qπ(st) =

r(sT+1), if t = T + 1 γVπ(st+1), otherwise ∈ [−1,1] (2)

hence both the value and the state-action value functions are bounded between -1 and 1 for all states and state-action pairs.

MCTS Design We define the MCTS tree as in Appendix A.2 as T = (N,E), where N is a set of nodes, and E is the set of edges. We use the notation of a node-edgenode relationship denoted by (s,a′,s′) where s represents the origin node, a′ describes the action corresponding to an edge, and s′ denotes the target node. This notation symbolically ties the action and the target state together, as

the action uniquely identifies the target state and is therefore indicative of it.

We use a pre-trained LM with parameters θ as a policy model, which we denote as πθ. The model autoregressively generates a sequence of tokens. We use a special token ’End of Intermediate Step’ (eois) to indicate the end of the reasoning step. We use a standard end-of-sequence (eos) token to indicate the end of the final reasoning step concluding the reasoning trajectory.

A parametric value model is used to evaluate the quality of states. While MCTS traditionally approximates these values through extensive simulations, such an approach is computationally expensive and impractical in the RLM context. Inspired by AlphaZero [138], which replaces simulations with a parameterized value model, we estimate the state-action values, i.e. the q-values, for reasoning sequences using a value model — effectively employing a processbased q-value model Qφ (see Appendix B.3). The value model is instantiated as a pre-trained transformer-based LM, modified by adding three linear layers and a shifted, rescaled sigmoid activation to align the output domain to the state action function domain [−1,1] (see Eq. 2). This setup proved more stable than alternatives, such as a tanh activation or a cropped linear layer. We will show in the following how such a model can be trained and provide a description for the data generation process in Appendix D. During training, we assume access to a final answer verifier, which evaluates the correctness of the model’s final answer and provides the true reward.

- C.1.2 MCTS Algorithm We now present the algorithmic steps of a Monte Carlo Tree Search variant similar to AlphaZero as implemented in the x1 reasoning framework, which we detail in Algorithm 1. The MCTS search operates in two distinct modes: training and inference. The core difference is that, during training, a final answer verifier evaluates and scores the final reasoning steps, providing a true reward signal that is backpropagated through the MCTS tree. This reward serves as a reliable

learning signal for the value model Qφ. During inference, however, the verifier is unavailable, and decisions rely solely on the value model.

Notation We chose to store all values in nodes instead of edges, which defines the following set of statistics saved for each node s:

- • N(s) - the visit count of node s,
- • q(s) - the running estimate of the q-value of the transition leading to state s,
- • β(s) - the binary terminality function, returns 1 if the node s is terminal and 0 otherwise.

The Selection phase iteratively identifies the most promising child node with a selection policy, which in x1 is a node-based variant of the PUCT algorithm in AlphaZero [139] (which is defined on edge-based values) without a prior for selecting a child of s:

N(s) − 1 1 + N(sc) · c1 + log

N(s) + c2 c2

arg max

q(sc) +

,

sc∈C(s)

where c1 and c2 are hyperparameters controlling the exploration bias, and the other values can be taken from the node statistics.

Expansion We append M nodes to the selected leaf, M being a hyperparameter. One of the major challenges in applying RLMs is maintaining the diversity of reasoning paths. By adding M nodes, we increase the exploration of alternative reasoning paths.

The Backpropagation step serves to propagate information from the terminal nodes back to their ancestors. In our implementation, we update the running estimates of the qvalues using the following formula:

 

 ,

q(s) ←(1 − α)q(s) + αγ

ws(sc) · q(sc)

scC(s)

where we look at the node-edge-node tuples (s,ac,sc) and sc ∈ C(s). The weights ws(sc) for combining the children q-values are defined over the visit scores of the nodes as follows:

N(sc) sc˜∈C(s) N(sc˜)

ws(sc) =

.

True Reward Propagation We improve the quality of the q-values by propagating the real final rewards back through the tree when a terminal state sT+1 is reached. During training, terminal nodes can be evaluated against a reference golden answer g∗ using an external verifier. For actions leading to terminal states, the associated reward is equal to the q-value (see Eq. 2). Therefore, instead of using the prediction of the q-value model, we initialize q(sT+1) with the true reward r(sT+1) based on the evaluation of the external verifier. The reward is then backpropagated via the q-values through the tree with our backpropagation operator. This adjustment anchors the q-value model predictions with real reward signals and prevents the q-value model predictions to diverge.

Best Path Selection After N iterations, MCTS will have formed a tree in which every path corresponds to one of the explored reasoning trajectories. The final reasoning step in a path with the highest terminal value estimate is returned as the final solution.

###### C.2 Training Phase 1

To adequately employ the MCTS-based reasoning scheme introduced in Appendix C.1, the policy model must be finetuned to generate responses in the format of semanticallyrelevant reasoning steps. The value model – a q-value model in our case – must be trained to accurately estimate the values of the sequences of reasoning steps.

We propose a two-phase training approach designed to let the policy effectively leverage the structured exploration and iterative refinement capabilities of the search process to generate optimal sequences of reasoning steps. A detailed algorithmic description of the pipeline is in Figure 14.

The first phase focuses on preparing the policy and value models to generate and evaluate reasoning trajectories effectively. This is achieved by supervised fine-tuning (SFT) training on a dataset of example sequences of reasoning steps (where intermediate reasoning steps are terminated by an ”End of Intermediate Step” eois token). The objective is twofold: (1) to fine-tune the policy model πθ to produce semantically coherent reasoning steps, and (2) to train the

- Algorithm 1 MCTS for Reasoning (Training mode in red)

Input: Policy model πθ, value model Qφ, question z0, golden answer g∗, binary correctness verifier Γ, number of MCTS iterations N, number of children expanded in every selection phase M, exploration constants c1,c2, Backpropagation weight α.

Output: Search tree T = (N,E) containing the best path τ∗.

- 1: s0 ← (z0) {Initialize root node}
- 2: N(s0) = 0
- 3: N ← {s0} {Initialize node set}
- 4: E ← ∅ {Initialize edge set}
- 5: i ← 1
- 6: while i ≤ N or β(s) ̸= 1 do
- 7: s ← s0 {Start from root node}
- 8: ————– Selection ——————————————
- 9: while s is not a leaf node do
- 10: {Select child sc ∈ C(s) with highest selection score}
- 11: sc ← arg max sc∈C(s)

q(sc) +

√

N(s)−1

1+N(sc) c1 + log N(s)+c

2 c2

- 12: s ← sc {Move to the selected child}
- 13: end while
- 14: ————– Expansion —————————————–
- 15: for j = 1 to M do
- 16: zc ← (t1,...tM

zc) ∼ πθ{Sample a new reasoning step}

- 17: sc ← s ⌢ zc {Append zc to the current state s}
- 18: q(sc) ← Qφ(s) {Predict with the Q-VM}
- 19: N(sc) ← 1 {Initialize visit count}
- 20: β(sc) ← 0 {Initialize terminality function}
- 21: if sc terminal then
- 22: β(sc) ← 1 {Mark as terminal}
- 23: r(sc) ←

1, if Γ(sc,g∗) = 1, −1, if Γ(sc,g∗) = 0. {Check for cor-

rectness to determine the reward}

- 24: q(sc) ← r(sc) {Overwrite by true reward}
- 25: end if
- 26: N ← N ∪ {sc} {Add the node to the tree}
- 27: E ← E ∪ {(s,sc)} {Add the edge to the tree}
- 28: end for
- 29: ————– Backpropagation ——————————–
- 30: while s ̸= s0 do
- 31: N(s) ← N(s) + 1 {Update the visit count}
- 32: q(s) ← (1 − α)q(s) + αγ s

c∈C(s) ws(sc)q(sc)

- 33: {Update the value}
- 34: s ← sp {Move to the parent}
- 35: end while
- 36: i ← i + 1
- 37: end while
- 38: Best Path Selection:
- 39: Select the best reasoning sequence s∗T.
- 40:
- 41: return s∗T, all reasoning sequences {s(ji)}j

q-value model Qφ to accurately assign scalar scores to reasoning trajectories, distinguishing between high-quality and suboptimal reasoning paths.

This supervised fine-tuning phase ensures that the policy model can generate reasoning steps consistent with the structured format required for downstream MCTSbased exploration, while the q-value model provides reliable evaluations of intermediate and terminal states. Together, these components form the foundation for the subsequent online reinforcement learning in the second phase, where the policy and q-value models are further refined through interaction with the reasoning framework.

- C.2.1 Datasets Generation and Preparation Performing SFT of the policy model requires a dataset of high-quality reasoning sequences denoted as DSFT = { x(SFTi) , ySFT(i) }. Each pair in the dataset consists of a prompt

x(SFTi) composed of a sequence of reasoning steps (for example x(SFTi) = (z0(i),...,zj(i))), and a target completion ySFT(i) = zj(i+1) which is the subsequent reasoning step or final answer. Appendix D contains a detailed account of the dataset creation and processing. It covers how the special eois token is appended to reasoning steps to mark the end of a step during inference.

Similarly to the policy model, training the q-value model requires a supervised dataset of reasoning sequences and corresponding scores. We denote this dataset

DQVM-train = {(x(QVM-traini) ,yQVM-train(i) )}, with reasoning sequences x(QVM-traini) = (z0(i),...,zt(i)) and target q-value yQVM-train(i) . Appendix D explains how this dataset can be generated using an initial list of questions, a base LLM for querying, and a verifier program to label reasoning sequences as conducive to a correct final answer or not.

- C.2.2 SFT of the Policy Model The supervised fine-tuning of the policy model, which we

illustrate in Algorithm 2, is performed on the dataset DSFT of prompts and target completions of the next reasoning step. The policy πθ is instantiated as a general pre-trained LLM. Specifically, we perform ’completion-only’ SFT such that for every (prompt, target completion) pair, the base model is trained to minimize the cross-entropy loss between its predicted token probabilities and the ground truth target completion.

- C.2.3 Q-Value Model Training

The q-value model Qφ is trained on DQVM-train to assign appropriate scalar scores to the candidate reasoning trajectories with the details being shown in Algorithm 3. It is instantiated as a pre-trained LLM with additional linear layers and to which a shifted and rescaled classification head is added; we denote all of its trainable weights as φ. Depending on the reward design, the q-value model can be trained via scalar (least squares) regression if continuous rewards are chosen, or with a classification objective such as the Binary Cross-Entropy (BCE) loss, if trajectories are labeled with binary rewards or as chosen-rejected preference pairs.

###### Phase 1

###### Phase 2

MCTS with value model

process-based RL training

process-based SFT

Policy Model

Policy Model

Policy Model

LLM

Thought LLM

RLM

CoT/MCTS Training data

MCTS with value model

Replay Buﬀer

solu on ﬁltering

MCTS with value model

process-based SFT

process-based RL training/SFT MCTS with value model

Value Model

Value Model

Value Model

PBV LLM

LLM

PBV LLM

MCTS with simula ons

Lin. layer + ac va on Lin. layer + ac va on

Lin. layer + ac va on

Training data

Replay Buﬀer

solu on ﬁltering

Fig. 14: The two phases of the training pipeline.

- Algorithm 2 SFT of Policy Model πθ (completion-only)

Input: Policy Model πθ, tokenized dataset DSFT = {(x(i),y(i))}, training hyperparameters (optimizer, learning rate η, batch size B, and maximum number of epochs E).

Output: Fine-tuned policy model πθ.

Algorithm 3 Fine-tuning of the Q-Value Model Qφ

Input: Q-value model Qφ (Q-VM), dataset DQVM-train = {(x(i),y(i))}, training hyperparameters (optimizer, learning rate η, batch size B, and maximum number of epochs E).

Output: Fine-tuned q-value model Qφ.

- 1: for epoch e = 1 to E do
- 2: Shuffle dataset DSFT.
- 3: Divide DSFT into batches {Bk} of size B.
- 4: for each batch Bk do
- 5: Initialize batch loss: Lbatch = 0.
- 6: for each sample (x(i),y(i)) ∈ Bk do
- 7: Iteratively predict completion tokens: yˆt(i) ∼ πθ(x(1:i)t−1),

where x(1:i)t−1 represents the context (prompt + previously predicted tokens).

- 8: Compute CE loss for each completion token:

L(i) = − |y

(i)|

t=1 log P(ˆyt(i) = yt(i)|x(i),πθ).

- 9: Accumulate the loss: Lbatch += L(i).
- 10: end for
- 11: Normalize batch loss: Lbatch = Lbatch/|Bk|.
- 12: Backpropagate gradients, update θ via optimizer.
- 13: end for
- 14: end for

By the end of training, Qφ should output accurate qvalue scores, which will later guide policy refinement in the second phase and improve the search accuracy when used in the MCTS.

###### C.3 Training Phase 2

Phase 2 involves generating reasoning sequences from the policy model with MCTS and the q-value model, and finetuning the policy model with an RL-based alignment algorithm to generate better completions. The q-value model must also be continually updated in this training loop to keep in-distribution with the policy model’s outputs. Sufficient pre-training of the policy and q-value models in the first phase is crucial to ensure stable training of these models in the second phase. The MCTS structure which provides a balanced exploration-exploitation search combined with

- 1: for epoch e = 1 to E do
- 2: Shuffle the dataset DQVM-train.
- 3: Divide DQVM-train into batches {Bk} of size B.
- 4: for each batch Bk do
- 5: for each sample (x(i),y(i)) ∈ Bk do
- 6: Predict the q-value with QVM yˆ(i) = Qφ(x(i)).
- 7: {Compute the loss:}
- 8: if Regression Loss then
- 9: L = B1 (x(i),y(i))(ˆy(i) − y(i))2.

- 10: end if
- 11: if Classification Loss then
- 12: L = B1 (x(i),y(i)) BCE(ˆy(i),y(i)).

- 13: end if
- 14: Backpropagate gradients, update φ via optimizer.
- 15: end for
- 16: end for
- 17: end for

repeated sampling of the policy model ensures sufficient exploration during this online-RL phase. This final training phase returns the fine-tuned policy and q-value models.

C.3.1 Training Algorithm

Phase 2 uses a set Dp = {p(i)} of prompt questions, which may be isolated from the phase 1 dataset DSFT. The training process, displayed in Algorithm 4, involves a repetition of a MCTS rollout phase followed by a training (reinforcement) phase.

To obtain data for the training, a MCTS tree T (i) is build w.r.t. each question p(i) using Algorithm 1 in training mode. The set of hyperparameters for MCTS ΞMCTS, denotes the number of MCTS iterations N (per question), the number of children expanded in every selection phase M, exploration constants c1,c2, and backpropagation weight α. To enhance the quality of the data, we prune the generated MCTS tree T˜(i) = (N˜(i),E˜(i)) to only include paths that reached a terminal state since only these paths received the reward. Then,

Algorithm 4 Phase 2: RL of the Policy and Q-Value Model Input: Policy πθ, q-value model Qφ, dataset Dp = {p(i)}, MCTS hyperparameters ΞMCTS. Output: Trained πθ and updated Qφ.

- 1: for each training iteration do
- 2: ————– Rollout ———————————
- 3: for each question p(i) ∈ Dp do
- 4: {Generate MCTS tree with πθ and Qφ (Algorithm 1)}

- 5: T (i) ← MCTS(p(i),Qφ,πθ,ΞMCTS)
- 6: {Remove incomplete paths from the tree}
- 7: T˜(i) ← Prune(T (i))
- 8: {Extract nodes and values, store them in replay buffer}
- 9: R ← R ∪ {(s(ji),zj(i),q(s(ji))}sj∈N˜(i)
- 10: end for
- 11: ————– Training ———————————
- 12: for each epoch do
- 13: Sample a batch B from replay buffer R.
- 14: Update policy πθ (Algorithm 5).

- 15: Update q-value model Qφ (Algorithm 7).

- 16: end for
- 17: end for

we extract all nodes and a set of node characteristics from the pruned tree. The dataset comprises of state, action and q-

value triplets of the pruned tree: {(s(ji),zj(i),q(s(ji))}sj∈N˜(i). The data is stored in a replay buffer R.

The reinforcement phase samples a batch of reasoning sequences from the replay buffer. From each trajectory, constituent states, actions, value estimates and the corresponding values attributed during MCTS are used to perform RL training (for example with PPO or reinforcement). Alternative schemes may involve selecting preference pairs among trajectories and then aligning the policy using DPO, or simply selecting the most desirable trajectory per question and performing further SFT training.

During this reinforcement phase, the value model is updated to mimic the (backpropagated) values from the MCTS process, which we illustrate in Algorithm 7.

- C.3.2 Policy Update The policy update, which we present in Algorithm 5, is performed on a batch B of reasoning sequences. As mentioned above, the reasoning sequences can be decomposed into state-action-value triplets to then perform RL training. We will discuss now three different methods (standard RL, preference-based RL, and SFT training) to improve the policy model during the second phase training.

Standard Policy Gradient Methods such as Proximal Policy Optimization (PPO) [135] or REINFORCE [1], [146] are particularly suited for tasks where trajectories are collected (online) and reliably evaluated by the q-value model Qφ.

PPO relies on the computation of trajectory (reasoning sequence) advantages Aˆ(st), which quantify how much better or worse an action taken in a given state is compared to the expected baseline value of that state. The advantage function is estimated by:

Aˆ(st) = Rt + γV (st+1) − V (st),

- Algorithm 5 Policy Update (PPO, DPO, or SFT)

Input: Batch B, policy πθ, reference policy πref, learning rate η, clipping parameter ε, preference data Bpref for DPO. Output: Updated policy πθ.

- 1: ————– Train via PPO ———————————
- 2: Select state-action-value triplets from sequences in B
- 3: for each (st,at,qt) ∈ B do
- 4: Compute the policy ratio: rθ = π

θ(at|st) πθref(at|st).

- 5: Compute the advantages Aˆ(st) (Algorithm 6).

- 6: Compute the PPO loss: LPPO = min(rθAˆ(st),clip(rθ,1 − ε,1 + ε)Aˆ(st)).
- 7: end for
- 8: (Optional) Add KL divergence or entropy regularization: LPPO ← LPPO + λKLKL(πθ||πref) + λHLH.
- 9: Perform gradient update to refine πθ.
- 10:
- 11: ————– Train via DPO (pairwise preferences) ——
- 12: Select preference pairs of reasoning sequences in B
- 13: for each pair (s+,s−) ∈ Bpref do
- 14: Compute DPO objective:

LDPO =

1 |Bpref| (s+,s−)

log σ β log

πθ(s+) πθ(s−)

.

- 15: end for
- 16: Perform gradient update to refine πθ.
- 17:
- 18: ————– Train via SFT (single target sequence) ——
- 19: Select high-value reasoning sequences s+ from B
- 20: for each reasoning sequence s+ do
- 21: Perform SFT on s+
- 22: end for

where Rt is the immediate environment reward at step t, V (st) is the state value of state st, and γ is the discount factor. We can derive the state value easily from the q-values obtained via the q-value model or the running estimates in the MCTS as follows:

V (st+1) =

1 γ

Qφ(st,at),

since rewards are sparse. The standard PPO approach trains the critic model from scratch on bootstrapped rewards for this purpose. We introduce an alternative advantage computation scheme that leverages the backpropagated values from the MCTS in conjunction with Qφ, as detailed in

- Algorithm 6. This integration combines MCTS’s exploration and evaluation capabilities with the RL update, enhancing robustness and efficiency in reasoning tasks.

Further regularization can be imposed on the PPO training procedure. To align the policy πθ with a reference policy πref (usually instantiated as πθ before the second phase) during training, the KL divergence KL(πθ||πref) between the two distributions can be added to the training loss. Additionally, to maintain the diversity of policy generations (and exploration during training), the entropy of the policy distribution can be enhanced by subtracting it from the loss. The entropy penalty is estimated over a batch B of stateaction pairs (s,a), where s denotes a reasoning sequence

Algorithm 6 Advantage Calculation in MCTS Framework Input: MCTS Tree T = (N,E), node statistics: rewards and q-values, q-value model Qφ, discount factor γ, and λ. Output: Advantages {Aˆ(st)}.

- 1: for each node si ∈ N do
- 2: Compute state values: vsMCTS

i+1

= γ1qMCTS(si)

- 3: Compute state values: vsMCTS

i

= γ1qMCTS(si−1)

- 4: Compute the advantage on the TD error: Aˆ(si) =

r(si,ai) + γvsMCTS

i+1

− vsMCTS

i

.

- 5: end for

and a the next reasoning step. The entropy of a single completion a is computed by summing the entropy of its individual tokens a1:|a| of a:

1 |B| (s,a)∈D a

LH = −

πθ(ai|[s,a1:i−1])log πθ(a|[s,a1:i−1]).

i∈a

Direct Preference Optimization (DPO) [125] aligns the policy to user preferences expressed as pairwise comparisons between reasoning sequences. Given pairs (s+,s−), where s+ is preferred over s−. This method may not require a process-based reward/value model. The loss involves the sigmoid function which we denote as σ.

Supervised Fine-Tuning (SFT) As a straightforward alternative to RL, high-value reasoning sequences can be selected to perform SFT, i.e. train the policy to maximize the likelihood of these reasoning steps. The high-value reasoning sequences may be selected as terminal nodes having the highest q-value, or highest aggregated intermediatestep values. This approach is inspired by AlphaZero-like frameworks, focusing on iteratively refining the policy to generate high-quality reasoning trajectories without requiring explicit rewards.

- C.3.3 Advantage Calculation

While standard advantage computation in PPO (e.g., via Generalized Advantage Estimation (GAE) [135]) is widely applicable, we propose an alternative approach tailored to our reasoning framework in Algorithm 6. Specifically, for each state/node s, we leverage the q-value estimates q(s) obtained during the MCTS process. They were updated in the backpropagation phase to provide a more informed estimate of the q-values incorporating the estimates of the children and potentially true reward signals from terminal paths in the tree. We expect these MCTS-derived values to be more reliable as they incorporate the ground truth terminal reward, propagated back through the tree, ensuring that a node’s value reflects both its immediate reward and the aggregated values of subsequent child states.

- C.3.4 Q-Value Model Update

During the second training phase, the q-value model Qφ is also updated (see Algorithm 7) to track the MCTSbackpropagated value estimates qMCTS(st), which should be of higher quality (thanks to the final answer verifier and score aggregation from child nodes). For each state-action pair (s,a), we train the q-value model Qφ via squared error

Algorithm 7 Q-Value Model Update Input: Batch B, q-value model Qφ, learning rate η. Output: Updated Qφ.

- 1: Compute loss: Lq = |B|1 (s,a,s′)(Qφ(s,a) − qMCTS(s′))2.

- 2: Perform gradient update on Lq.

minimization, to match its q-value Qφ(s,a) as closely as possible to the corresponding MCTS-value qMCTS(s′), i.e. updated q-value of action a taken in state s leading to state s′.

This has the benefit of both improving the accuracy of the value model, and keeping it ”in-distribution” with the new policy outputs during this online-RL training.

APPENDIX D DATA GENERATION

###### D.1 Phase 1 Policy Model Training

The objective of this training process is to introduce a new ’End of Intermediate Step’ (eois) token that serves to delimit individual reasoning steps while preserving the original distribution of the model as much as possible. To achieve this, the model is trained on data generated by itself using greedy decoding.

The training data is derived from eight chain-of-thought (CoT) completions generated for 1,000 questions sampled from the training split of the MATH dataset [65]. These completions are produced using the same model intended for subsequent training with greedy decoding. During this generation process, the reasoning steps in the data are observed to be separated by two consecutive ’\n\n’. This observation informs the method of delimitation used to construct pairs of questions and their corresponding sequences of reasoning steps.

For each data point, consisting of a question prompt and its associated target response comprising multiple reasoning steps (q(i),[z1(i),...,zT(i+1) ]), additional tokens are introduced to explicitly mark the boundaries of the reasoning steps. Specifically, the ’End of Intermediate Step’ (eois) token is defined and inserted after each reasoning step zj(i), resulting in a modified step zj(i)∗. Additionally, the ’End of Sequence’ (eos) token is appended to the final reasoning step zT(i+1) , yielding zT(i+1)∗ = [zT(i+1) ;eos]. This augmentation ensures that the model can consistently identify when a final solution has been reached during inference.

For LLaMA models, it has been empirically observed that introducing an ’assistant’ token after each reasoning step enhances the model’s effective utilization of the eois token. However, this behavior may not generalize to other base models, necessitating careful consideration when applying this approach.

Accordingly, the target sequence for supervised finetuning (SFT) is constructed as:

###### ySFT(i) = [z1(i),eois,assistant,z2(i),...,zT(i+1) ,eos].

This approach yields a training dataset comprising pairs of prompts and their corresponding target completions, formally represented as:

DSFT = {(q(i),ySFT(i) )}.

- D.2 Phase 1 Value Model Training

The original MCTS framework relies on simulations to evaluate a state. Given the state, n rollouts are performed till a terminal state is reached. The terminal states usually can be evaluated (e.g., in math by comparing it with the golden answer). This enables the distribution of terminal rewards based on their success which are then aggregated to provide a value estimate of the state. These Monte Carlo simulations serve as an estimate of a state’s ability to lead to a correct answer. The value estimated in this manner corresponds to the expected cumulative future reward for a given state:

T

γt−ir(st,at) | si = s ,

(s) = Eτ∼πθ

Vπ

θ

t=i

where T is the terminal step of the partial reasoning chain τ = (si,ai,ri,si+1,...,sT,aT,rT,sT+1).

Since rewards are sparse (i.e., r(st,at) = 0 for all t < T), the value function simplifies to:

(st) = Eπθ γT−tr(sT,aT) | st .

Vπ

θ

This represents the expected terminal reward, which can be empirically estimated using Monte Carlo (MC) estimates:

N

1 N

γT−tr(s(Ti),a(Ti)) := Vˆ(st),

(st) ≈

Vπ

θ

i=1

where N is the number of sampled reasoning chains, and s(Ti),a(Ti),s(Ti+1) denote the last transition of the simulation trajectory τ(i) = (st,a(ti),s(t+1i) ,...,s(Ti),a(Ti),s(Ti+1) ) for i ∈ {1,...,N}.

To avoid sample inefficiencies and high computational burdens, AlphaGo Zero [139] and AlphaZero [138] introduce a value model to replace simulations by using its predictions for a state. We follow this approach by defining a process-based value model Vφ. Notably, we train this model with simulation data (instead of true value functions), thereby building a model that predicts state value function estimates Vˆ. We denote this model as Vˆφ, parameterized by φ.

Given that the input of a value model is a sequence of reasoning steps - therefore a sequence of tokens, the natural value model architecture is to use a language model on which one adds linear layer(s) and a suitable output activation function. Typically, it is designed to output a scalar value Vˆφ(st) ∈ C ⊆ R.

The core distinction between different modeling approaches to state value functions lies in how rewards are modeled. Depending on whether a binary reward setting or a continuous (bounded) one is used, the aggregation mechanism, model architecture, training loss, and interpretation of the predictions vary. We provide an overview of both scenarios and, although often omitted for simplicity, we consider both γ = 1 and γ ∈ (0,1] for continuous rewards in our analysis.

D.2.1 Binary Rewards: Modeling the Likelihood of a Correct Terminal State

For this approach the rewards are modeled binary, therefore r(sT,aT) = +1 for correct solutions and r(sT,aT) = 0 for incorrect solutions. We will adopt a discount factor of γ = 1 which we will see aligns more with the interpretation this reward model provides and is widely adopted in literature. This approach corresponds to the value model proposed in AlphaGo Zero [139].

D.2.1.1 State Value Estimation: The value function then further simplifies to:

(st) = Eπθ [r(sT,aT) | st] = Pπθ (r(sT,aT) = 1 | st)

Vπ

θ

This formulation represents the probability of reaching a correct terminal state from a given state st. Empirically, this probability is estimated using simulations as follows:

#correct simulations #simulations

###### := Vˆ(st).

(st) ≈

Vπ

θ

D.2.1.2 Data Generation: To generate labels for estimating the state-value function during the training of a value model, we use MCTS with simulations till a terminal node is reached and calculate the ratio between the number of correct simulations to the number of simulations. There is one very important detail, for a trajectory τ = (si,ai,ri,si+1,...,sT+1) where sT+1 is a terminal state. By definition, the true state value function at sT+1 is zero. However, in training the value model, we avoid instructing it to output zero for terminal states. Instead, in a supervised learning setting, we can identify terminal states and directly compare the model’s predictions against the known correct outcomes (golden answers). This comparison negates the need to rely solely on the value model to estimate the value of terminal states or to determine the reward associated with transitioning into these states. During inference, while we can still recognize terminal states, we cannot evaluate them by comparing the model’s output to a golden answer. Therefore, an alternative metric is necessary. We train the value model to predict whether transitioning to sT+1 leads to a correct terminal outcome. By learning the relationship between a node’s content and the correctness of the resulting terminal state, the model can estimate the likelihood that a terminal state leads to a correct answer. To approximate the terminal reward during inference, we define: r(sT,aT,sT+1) ≈ [0.5,1](Vˆφ(sT+1)). Here Vˆφ(sT+1) represents the value predicted by the value model for the terminal state sT+1. If this predicted likelihood exceeds a threshold (e.g., 0.5), we assign a terminal reward of 1; otherwise, we assign a reward of 0. This approach allows the value model to indirectly influence the terminal reward by predicting the likelihood of a correct outcome. Consequently, during training, terminal rewards serve as labels for terminal states in the value model. It is important to note that Vˆφ(sT+1) is not used in any other context but solely to estimate the terminal reward.

Vˆφ(sT+1) ̸= Vˆ(sT+1) This distinction clarifies that the predicted value for the terminal state Vˆφ(sT+1) differs from the standard value function’s definition Vˆ(sT+1) = 0.

D.2.1.3 Model Training Vˆφ : S → [0,1]: When trained with these labels we obtain a value model Vˆφ, parameterized by φ, that represents the likelihood of a correct terminal state emanating from state st. Therefore, the model will output values between 0 and 1. To accommodate the binary classification nature of this task, the model should employ a sigmoid activation function in the output layer. The training objective is then to minimize the binary crossentropy (CE) loss between the predicted probabilities and the empirical estimates derived from the simulations:

L(φ) = −

N

1 N

i=1

yi log V ˆφ(s(ti)) + (1 − yi)log 1 − Vˆφ(s(ti)) ,

where yi ∈ {0,1} denotes the binary label indicating whether the i-th simulation resulted in a correct terminal state.

D.2.1.4 Summary: Employing a binary reward structure offers several benefits. First of all, simplicity since binary rewards simplify the learning process, reducing the complexity associated with continuous reward signals. Moreover, the clear distinction between correct and incorrect states facilitates faster convergence during training making this approach effective. In addition, binary classification is less susceptible to noise in reward signals, ensuring more stable value estimates. Furthermore, this approach aligns with the objectives of reinforcement learning in achieving clear and unambiguous rewards, thereby streamlining the optimization of the policy πθ.

D.2.2 Continuous and Bounded Rewards: Modeling the Expected Future Reward

We model the rewards to be continuous and bounded by allowing values in [a,b]:

###### (st) ∈ [a,b].

###### Vπ

θ

A common design, is to set the borders to −1 and 1 such that a terminal reward is r(sT,aT) = +1 for correct terminal states and r(sT,aT) = −1 for incorrect states. This approach models the expected future reward as a continuous and bounded value, capturing the degree of correctness or quality of the terminal state. In contrast to the binary reward structure, continuous and bounded rewards provide a more nuanced representation of the outcomes in reasoning tasks. Note, that without discounting this approach resembles the proposed value model of AlphaZero [138].

D.2.2.1 Bounded rewards: By constraining rewards within a predefined interval [a,b], we effectively create a correctness scale where the extremities represent the definitive outcomes of the reasoning process. Specifically, the lower bound a corresponds to reaching an incorrect terminal state, while the upper bound b signifies a correct terminal state. This bounded framework mirrors the spectrum of possible correctness, allowing the model to capture varying degrees of solution quality between these extremes. Such a scale facilitates a more nuanced evaluation of intermediate states, reflecting partial correctness or varying levels of reasoning quality. Moreover, this approach ensures that the

reward signals remain interpretable and consistent, fostering a clear distinction between successful and unsuccessful outcomes.

D.2.2.2 State Value Estimation: With a discount factor γ ∈ (0,1], the value function is defined as:

###### (st) = E γT−tr(sT,aT) | st ,

###### Vπ

θ

where r(sT,aT) = b for correct terminal states and r(sT,aT) = a for incorrect ones. Empirically, this expectation is approximated by averaging the rewards of the simulations:

N

1 N

γT−tr(s(Ti),a(Ti)) := Vˆ(st),

(st) ≈

###### Vπ

θ

i=1

where N denotes the number of sampled reasoning chains, and (s(Ti),a(Ti),s(Ti+1) ) represent the final transition of the i-th simulation trajectory τ(i) = (st,a(ti),st(+1i) ,...,sT(i),a(Ti),s(Ti+1) ) for i ∈ {1,...,N}. If a discount factor is applied γ ∈ (0,1) then each terminal reward is discounted proportional to the number of steps needed to reach the terminal state. This corresponds to the soft estimation proposed by Wang et al. [160]. We want to note that this estimator typically underestimates V due to its proneness to false negatives [61], [176].

- D.2.2.3 Data Generation: To generate labels for

state-value function estimate pairs to train a value model, we use MCTS with simulations and average the outcomes of the simulations. Therefore, at each newly generated node s we simulate till a terminal node is reached and we record the depth - the number of steps needed starting from s (since T is not identical per trajectory). We then record the terminal reward which in our case is r(sT,aT) = 1 for correct and r(sT,aT) = −1 for incorrect answers. Discounted by the depth we can average these rewards and obtain an estimation of the node value which serves as a label for the initial value model training.

- D.2.2.4 Model Training Vˆφ : S → [a,b]: The value

model Vˆφ, parameterized by φ, is designed to predict the expected terminal reward from any given state st. To accommodate the continuous and bounded nature of this task, the model employs a scaled and shifted sigmoid activation function in the output layer, ensuring that the predictions remain within the range [a,b]. The training objective is to minimize the mean squared error (MSE) loss between the predicted values and the empirical estimates derived from the simulations:

N

1 N

2

V ˆφ(s(ti)) − γT−tr(s(Ti),a(Ti))

L(φ) =

###### .

i=1

We also experimented with a tanh activation output and a linear layer with clipping of the values. However, both methods proved to be unstable in training in contrast to the scaled and shifted sigmoid layer. A tanh and sigmoid layer naturally bounds the output but also pushes values towards the extremes, enhancing the separation between high and low value estimates. This characteristic can improve the model’s ability to distinguish between highly correct and highly incorrect states which is why we are particularly interested in these activation functions.

- D.2.2.5 Discounting: Introducing a discount factor

γ aligns the value function with the incremental nature of reasoning tasks. Unlike traditional games, where all moves contribute indirectly and trajectories are not penalized for length, reasoning benefits from discouraging unnecessary or redundant steps. The inclusion of the discount factor γ ensures that rewards achieved sooner have a greater impact on the value function, the model incentivizes reaching correct solutions with fewer steps which ultimately enhances efficiency and suppresses redundancies. Moreover, this models the uncertainty decay in the trajectories; the further into the future a reward lies, the more uncertain its prediction becomes. Discounting naturally reduces the reliance on these uncertain long-term rewards, thereby stabilizing the learning process by focusing on more predictable and immediate outcomes. However, the model’s performance becomes sensitive to the choice of γ, requiring careful tuning to balance the influence of immediate versus long-term rewards. Balancing the discount factor is essential to ensure that the model effectively captures the importance of both progress and the final correctness of the reasoning chain.

- D.2.2.6 Summary: Employing a continuous and

bounded reward structure offers several benefits. Unlike binary rewards, continuous rewards provide a finer distinction between varying degrees of correctness, allowing the model to capture subtle differences in terminal states. Continuous rewards can encode more information about the quality of solutions, facilitating more informed decisionmaking during the search process. Bounded rewards prevent extreme values, promoting numerical stability and consistent training dynamics. However, it also shows that the choice of reward values and their scaling can significantly impact the learning process, necessitating careful calibration to ensure effective training.

- D.3 State Action Value Function Modeling The state-action value function, commonly denoted as

(st,at), represents the expected cumulative reward of taking action at in state st under policy πθ. Formally, it is defined in our framework as:

###### Qπ

θ

###### Qπ

###### (st,at)

θ

T

γi−tr(si,ai) | st,at

= Eτ∼πθ

i=t

T

γi−(t+1)r(si,ai) | st,at

= r(st,at) + γEτ∼πθ

i=t+1

= r(st,at) + γEst+1 [Vπ

###### (st+1) | st,at] det.=P r(st,at) + γVπ

θ

###### (st+1),

θ

where T denotes the terminal step of the trajectory τ = (st,at,rt,st+1,...,sT,aT,rT,sT+1). In environments characterized by sparse rewards, where r(st,at) = 0 for all t < T, the q-value simplifies to:

(st+1). At terminal states, where the state value Vπ

###### Qπ

###### (st,at) = γVπ

θ

θ

(sT+1) = 0, the q-value further reduces to:

θ

###### Qπ

(sT,aT) = r(sT,aT).

θ

- D.3.1 Process-Based Q-Value Modeling

A process-based q-value model utilizes the same architecture as a process-based value model, typically leveraging a LLM enhanced with additional linear layers and an appropriate output activation function. The output is a scalar value Qˆφ(st,at) ∈ C ⊆ R. Specifically, the q-value model takes a state-action pair—comprising a sequence of past steps and the current action—and predicts the corresponding q-value based on the aforementioned formulation.

D.3.1.1 Training Data Generation: To train the qvalue model, it is essential to compute the q-values for various state-action pairs. For t < T, q-values can be estimated using N Monte Carlo simulations as follows:

Qπ

θ

(st,at) = r(st,at) + γVπ

θ

(st+1)

= γVπ

θ

(st+1) (since r(st,at) = 0) ≈ γ ·

1 N

N

i=1

γT−(t+1)r(s(Ti),a(Ti))

=

1 N

N

i=1

γT−tr(s(Ti),a(Ti)) := Qˆ(st,at),

where N is the number of sampled reasoning chains, and τ(i) = (st,a(ti),st(+1i) ,...,sT(i),a(Ti),s(Ti+1) ) represents the i-th simulation trajectory for i ∈ {1,...,N}. This estimation aligns with the state value estimation under the sparse reward formulation:

Qˆ(st,at) = Vˆ(st).

For t = T, the q-value is directly given by the immediate reward:

Qπ

θ

(sT,aT) = r(sT,aT) = Vπ

θ

(sT+1) ̸= Vˆ(sT) = 0.

D.3.1.2 Reward Modeling: The considerations about reward modeling apply to q-value models as well since these models are trained very similar, so we omit their discussion here.

- D.3.2 The Difference Between Value and Q-Value Models

The difference of VMs and Q-VMs can be easily shown in how they are used in the evaluation processes of an MCTS algorithm. Q-VMs predict Qˆφ(st,at), which evaluates the action at taken in state st that deterministically transitions to st+1. Thus, the value Qˆ(st,at) is used to evaluate adding the node st+1 to the tree. On the other hand, for VMs, adding a node st+1 to the tree is determined by Vˆ(st+1) = γ1Qˆφ(st,at), where γ is the discount factor.

This distinction is making the training processes different. Note that st ⌢ at = st+1. For Q-VMs, the training tuples are ((st,at),Qˆ(st,at)) = (st+1,Qˆ(st,at)) due to the deterministic transition. For VMs, the corresponding training tuples are (st+1,Vˆ(st+1)). Since we propose training VMs on terminal rewards for terminal states instead of assigning a label of 0, VMs and Q-VMs become equivalent under the following transformation for any t ∈ {0,...,T} for evaluating adding node st+1:

Vˆ(st+1) =

1 γ

Qˆφ(st,at).

We introduced q-value models since they address a critical inconsistency of value models in terminal states. Specifically, while value models assign a flat value of zero to terminal states, q-value models provide a meaningful evaluation of the final action’s correctness through

(sT,aT) = r(sT,aT). This distinction is essential for accurately assessing whether a terminal step leads to a correct or incorrect response during inference.

Qπ

θ

###### REFERENCES

- [1] A. Ahmadian, C. Cremer, M. Gall´e, M. Fadaee, J. Kreutzer, O. Pietquin, A. Ust¨ un,¨ and S. Hooker. Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’24, pages 12248–12267, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
- [2] J. Ahn, R. Verma, R. Lou, D. Liu, R. Zhang, and W. Yin. Large Language Models for Mathematical Reasoning: Progresses and Challenges. In N. Falk, S. Papi, and M. Zhang, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop, EACL ’24, pages 225–237, St. Julian’s, Malta, Mar. 2024. Association for Computational Linguistics.
- [3] AI-MO. AIME 2024. https://huggingface.co/datasets/AI-MO/ aimo-validation-aime, July 2024. (accessed Jan. 19, 2025).
- [4] AI-MO. AMC 2024. https://huggingface.co/datasets/AI-MO/ aimo-validation-amc, July 2024. (accessed Jan. 19, 2025).
- [5] A. Amini, S. Gabriel, S. Lin, R. Koncel-Kedziorski, Y. Choi, and H. Hajishirzi. MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), NAACL ’19, pages 2357–2367, Minneapolis, MN, USA, June 2019. Association for Computational Linguistics.
- [6] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, and C. Sutton. Program Synthesis with Large Language Models, Aug. 2021. arXiv:2108.07732.
- [7] A. Bakhtin, L. van der Maaten, J. Johnson, L. Gustafson, and R. Girshick. PHYRE: A New Benchmark for Physical Reasoning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´eBuc, E. Fox, and R. Garnett, editors, Proceedings of the ThirtyThird Annual Conference on Neural Information Processing Systems (NeurIPS ’19), volume 32 of Advances in Neural Information Processing Systems, pages 5082–5093, Vancouver, Canada, Dec. 2019. Curran Associates.
- [8] Q. Bao, G. Gendron, A. Peng, W. Zhong, N. Tan, Y. Chen, M. Witbrock, and J. Liu. Assessing and Enhancing the Robustness of Large Language Models with Task Structure Variations for Logical Reasoning. In Proceedings of the 31st International Conference on Neural Information Processing, ICONIP ’24, Auckland, New Zealand, Dec. 2024.
- [9] Q. Bao, A. Peng, T. Hartill, N. Tan, Z. Deng, M. Witbrock, and J. Liu. Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation. In A. d’Avila Garcez and E. Jim´enez-Ruiz, editors, Proceedings of the 16th International Workshop on Neural-Symbolic Learning and Reasoning (NeSy ’22), volume 3212 of Workshop Proceedings, pages 202–217, Windsor Great Park, United Kingdom, Sept. 2022. CEUR.
- [10] T. Ben-Nun and T. Hoefler. Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis. ACM Comput. Surv., 52(4):65:1–65:43, Aug. 2019.
- [11] M. Besta, N. Blach, A. Kubicek, R. Gerstenberger, L. Gianinazzi, J. Gajda, T. Lehmann, M. Podstawski, H. Niewiadomski, P. Nyczyk, and T. Hoefler. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16):17682–17690, Mar. 2024.
- [12] M. Besta, A. C. Catarino, L. Gianinazzi, N. Blach, P. Nyczyk, H. Niewiadomski, and T. Hoefler. HOT: Higher-Order Dynamic Graph Representation Learning with Efficient Transformers. In S. Villar and B. Chamberlain, editors, Proceedings of the Second

- Learning on Graphs Conference (LOG ’23), volume 231 of Proceedings of Machine Learning Research, pages 15:1–15:20, Virtual Event, Nov. 2023. PMLR.
- [13] M. Besta, R. Grob, C. Miglioli, N. Bernold, G. Kwa´sniewski, G. Gjini, R. Kanakagiri, S. Ashkboos, L. Gianinazzi, N. Dryden, and T. Hoefler. Motif Prediction with Graph Neural Networks. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, pages 35–45, Washington DC, USA, Aug. 2022. Association for Computing Machinery.
- [14] M. Besta and T. Hoefler. Parallel and Distributed Graph Neural Networks: An In-Depth Concurrency Analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):2584–2606, May 2024.
- [15] M. Besta, A. Kubicek, R. Gerstenberger, M. Chrapek, R. Niggli, P. Okanovic, Y. Zhu, P. Iff, M. Podstawski, L. Weitzendorf, M. Chi, J. Gajda, P. Nyczyk, J. Muller,¨ H. Niewiadomski, and T. Hoefler. Multi-Head RAG: Solving Multi-Aspect Problems with LLMs, June 2025. arXiv:2406.05085.
- [16] M. Besta, F. Memedi, Z. Zhang, R. Gerstenberger, N. Blach, P. Nyczyk, M. Copik, G. Kwa´sniewski, J. Muller,¨ L. Gianinazzi, et al. Demystifying Chains, Trees, and Graphs of Thoughts, Feb.

2025. arXiv:2401.14295.

- [17] M. Besta, L. Paleari, M. Copik, R. Gerstenberger, A. Kubicek, P. Nyczyk, P. Iff, E. Schreiber, T. Srindran, T. Lehmann, H. Niewiadomski, and T. Hoefler. CheckEmbed: Effective Verification of LLM Solutions to Open-Ended Tasks, June 2025. arXiv:2406.02524.
- [18] M. Besta, P. Renc, R. Gerstenberger, P. Sylos Labini, A. Ziogas, T. Chen, L. Gianinazzi, F. Scheidl, K. Szenes, A. Carigiet, P. Iff, G. Kwa´sniewski, R. Kanakagiri, C. Ge, S. Jaeger, J. Was, F. Vella, and T. Hoefler. High-Performance and Programmable Attentional Graph Neural Networks with Global Tensor Formulations. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’23, pages 66:1– 66:16, Denver, CO, USA, Nov. 2023. Association for Computing Machinery.
- [19] M. Besta, Z. Vonarburg-Shmaria, Y. Schaffner, L. Schwarz, G. Kwa´sniewski, L. Gianinazzi, J. Beranek, K. Janda, T. Holenstein, S. Leisinger, P. Tatkowski, E. Ozdemir, A. Balla, M. Copik, P. Lindenberger, M. Konieczny, O. Mutlu, and T. Hoefler. GraphMineSuite: Enabling High-Performance and Programmable Graph Mining Algorithms with Set Algebra. Proc. VLDB Endow., 14(11):1922–1935, July 2021.
- [20] Z. Bi, K. Han, C. Liu, Y. Tang, and Y. Wang. Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning, Dec.

2024. arXiv:2412.09078.

- [21] Y. Bisk, R. Zellers, R. Le Bras, J. Gao, and Y. Choi. PIQA: Reasoning about Physical Commonsense in Natural Language. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):7432– 7439, Apr. 2020.
- [22] R. A. Bradley and M. E. Terry. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4):324–345, Dec. 1952.
- [23] T. Burstr¨om, V. Parida, T. Lahti, and J. Wincent. AI-Enabled Business-Model Innovation and Transformation in Industrial Ecosystems: A Framework, Model and Outline for Further Research. Journal of Business Research, 127:85–95, 2021.
- [24] M. Chang, J. Zhang, Z. Zhu, C. Yang, Y. Yang, Y. Jin, Z. Lan, L. Kong, and J. He. AgentBoard: An Analytical Evaluation Board of Multi-Turn LLM Agents. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS ’24), volume 37 of Advances in Neural Information Processing Systems, pages 74325– 74362, Vancouver, Canada, Dec. 2024. Curran Associates.
- [25] E. Charniak and M. Johnson. Coarse-to-Fine n-Best Parsing and MaxEnt Discriminative Reranking. In K. Knight, H. T. Ng, and K. Oflazer, editors, Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics, ACL ’05, pages 173–180, Ann Arbor, MI, USA, June 2005. Association for Computational Linguistics.
- [26] G. Chen, M. Liao, C. Li, and K. Fan. AlphaMath Almost Zero: Process Supervision without Process. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS ’24), volume 37 of

- Advances in Neural Information Processing Systems, pages 27689– 27724, Vancouver, Canada, Dec. 2024. Curran Associates.
- [27] J. Chen, T. Li, J. Qin, P. Lu, L. Lin, C. Chen, and X. Liang. UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP ’22, pages 3313– 3323, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics.
- [28] J. Chen, H. Lin, X. Han, and L. Sun. Benchmarking Large Language Models in Retrieval-Augmented Generation. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16):17754–17762, Mar. 2024.
- [29] J. Chen, J. Tang, J. Qin, X. Liang, L. Liu, E. Xing, and L. Lin. GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 513–523, Virtual Event, Aug.

2021. Association for Computational Linguistics.

- [30] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating Large Language Models Trained on Code, July 2021. arXiv:2107.03374.
- [31] W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. Transactions on Machine Learning Research, Nov. 2023.
- [32] W. Chen, M. Yin, M. Ku, P. Lu, Y. Wan, X. Ma, J. Xu, X. Wang, and T. Xia. TheoremQA: A Theorem-Driven Question Answering Dataset. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP ’23, pages 7889–7901, Singapore, Dec. 2023. Association for Computational Linguistics.
- [33] X. Chen, M. Lin, N. Sch¨arli, and D. Zhou. Teaching Large Language Models to Self-Debug. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [34] K. Chernyshev, V. Polshkov, E. Artemova, A. Myasnikov, V. Stepanov, A. Miasnikov, and S. Tilga. U-MATH: A UniversityLevel Benchmark for Evaluating Mathematical Skills in LLMs, Jan. 2025. arXiv:2412.03205.
- [35] F. Chollet. On the Measure of Intelligence, Nov. 2019. arXiv:1911.01547.
- [36] A. Choudhury, Y. Wang, T. Pelkonen, K. Srinivasan, A. Jain, S. Lin, D. David, S. Soleimanifard, M. Chen, A. Yadav, R. Tijoriwala, D. Samoylov, and C. Tang. MAST: Global Scheduling of ML Training Across Geo-Distributed Datacenters at Hyperscale. In Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation, OSDI ’24, pages 563–580, Santa Clara, CA, USA, July 2024. USENIX Association.
- [37] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep Reinforcement Learning from Human Preferences. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Proceedings of the Thirty-First Annual Conference on Neural Information Processing Systems (NIPS ’17), volume 30 of Advances in Neural Information Processing Systems, pages 4299–4307, Long Beach, CA, USA, Dec.

2017. Curran Associates.

- [38] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training Verifiers to Solve Math Word Problems, Nov. 2021. arXiv:2110.14168.
- [39] M. Copik, R. B¨ohringer, A. Calotoiu, and T. Hoefler. FMI: Fast and Cheap Message Passing for Serverless Functions. In Proceedings of the 37th International Conference on Supercomputing, ICS ’23, pages 373–385, Orlando, FL, USA, June 2023. Association for Computing Machinery.
- [40] M. Copik, G. Kwa´sniewski, M. Besta, M. Podstawski, and T. Hoefler. SeBS: A Serverless Benchmark Suite for Function-as-aService Computing. In Proceedings of the 22nd International Middleware Conference, Middleware ’21, pages 64–78, Virtual Event, Dec. 2021. Association for Computing Machinery.
- [41] G. Cui, L. Yuan, Z. Wang, H. Wang, W. Li, B. He, Y. Fan, T. Yu, Q. Xu, W. Chen, J. Yuan, H. Chen, K. Zhang, X. Lv, S. Wang, Y. Yao, X. Han, H. Peng, Y. Cheng, Z. Liu, M. Sun, B. Zhou, and N. Ding. Process Reinforcement Through Implicit Rewards, Feb.

2025. arXiv:2502.01456.

- [42] D. De Sensi, T. De Matteis, K. Taranov, S. Di Girolamo, T. Rahn, and T. Hoefler. Noise in the Clouds: Influence of Network Performance Variability on Application Scalability. Proc. ACM Meas. Anal. Comput. Syst., 6(3):49:1–49:27, Dec. 2022.
- [43] M. DeLorenzo, A. B. Chowdhury, V. Gohil, S. Thakur, R. Karri, S. Garg, and J. Rajendran. Make Every Move Count: LLMBased High-Quality RTL Code Generation Using MCTS, Feb.

2024. arXiv:2402.03289.

- [44] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su. Mind2Web: Towards a Generalist Agent for the Web. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS ’23), volume 36 of Advances in Neural Information Processing Systems, pages 28091–28114, New Orleans, LA, USA, Dec. 2023. Curran Associates.
- [45] Y. Deng, W. Zhang, Z. Chen, and Q. Gu. Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves, Apr. 2024. arXiv:2311.04205.
- [46] X. Dong, M. Teleki, and J. Caverlee. A Survey on LLM InferenceTime Self-Improvement, Dec. 2024. arXiv:2412.14352.
- [47] I. El Naqa, M. A. Haider, M. L. Giger, and R. K. Ten Haken. Artificial Intelligence: Reshaping the Practice of Radiological Sciences in the 21st Century. British Journal of Radiology, 93(1106):20190855, Jan. 2020.
- [48] A. Elliott. The Culture of AI: Everyday Life and the Digital Revolution. Routledge, 2018.
- [49] S. Es, J. James, L. Espinosa Anke, and S. Schockaert. RAGAs: Automated Evaluation of Retrieval Augmented Generation. In N. Aletras and O. De Clercq, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, EACL ’24, pages 150– 158, St. Julian’s, Malta, Mar. 2024. Association for Computational Linguistics.
- [50] J. Frohberg and F. Binder. CRASS: A Novel Data Set and Benchmark to Test Counterfactual Reasoning of Large Language Models. In N. Calzolari, F. B´echet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, J. Odijk, and S. Piperidis, editors, Proceedings of the Thirteenth Language Resources and Evaluation Conference, LREC ’22, pages 2126–2140, Marseille, France, June 2022. European Language Resources Association.
- [51] Y. Fu, L. Xue, Y. Huang, A.-O. Brabete, D. Ustiugov, Y. Patel, and L. Mai. ServerlessLLM: Low-Latency Serverless Inference for Large Language Models. In Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation, OSDI ’24, pages 135–153, Santa Clara, CA, USA, July 2024. USENIX Association.
- [52] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. PAL: Program-Aided Language Models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning (ICML ’23), volume 202 of Proceedings of Machine Learning Research, pages 10764–10799, Honolulu, HI, USA, July

2023. PMLR.

- [53] S. N. Giest and B. Klievink. More Than a Digital System: How AI Is Changing the Role of Bureaucrats in Different Organizational Contexts. Public Management Review, 26(2):379–398, 2024.
- [54] E. Glazer, E. Erdil, T. Besiroglu, D. Chicharro, E. Chen, A. Gunning, C. F. Olsson, J.-S. Denain, A. Ho, E. de Oliveira Santos, O. J¨arviniemi, M. Barnett, R. Sandler, M. Vrzala, J. Sevilla, Q. Ren, E. Pratt, L. Levine, G. Barkley, N. Stewart, B. Grechuk, T. Grechuk, S. V. Enugandla, and M. Wildon. FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI, Dec. 2024. arXiv:2411.04872.
- [55] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. AlDahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. The Llama 3 Herd of Models, Nov. 2024. arXiv:2407.21783.
- [56] X. Guan, Y. Liu, X. Lu, B. Cao, B. He, X. Han, L. Sun, J. Lou, B. Yu, Y. Lu, and H. Lin. Search, Verify and Feedback: Towards Next Generation Post-Training Paradigm of Foundation Models via Verifier Engineering, Nov. 2024. arXiv:2411.11504.
- [57] X. Guan, L. L. Zhang, Y. Liu, N. Shang, Y. Sun, Y. Zhu, F. Yang, and M. Yang. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking, Jan. 2025. arXiv:2501.04519.

- [58] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, Jan. 2025. arXiv:2501.12948.
- [59] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang. RetrievalAugmented Language Model Pre-Training. In H. D. III and A. Singh, editors, Proceedings of the 37th International Conference on Machine Learning (ICML ’20), volume 119 of Proceedings of Machine Learning Research, pages 3929–3938, Virtual Event, July

2020. PMLR.

- [60] S. Han, H. Schoelkopf, Y. Zhao, Z. Qi, M. Riddell, W. Zhou, J. Coady, D. Peng, Y. Qiao, L. Benson, L. Sun, A. Wardle-Solano, H. Szab´o, E. Zubova, M. Burtell, J. Fan, Y. Liu, B. Wong, M. Sailor, A. Ni, L. Nan, J. Kasai, T. Yu, R. Zhang, A. Fabbri, W. M. Kryscinski, S. Yavuz, Y. Liu, X. V. Lin, S. Joty, Y. Zhou, C. Xiong, R. Ying, A. Cohan, and D. Radev. FOLIO: Natural Language Reasoning with First-Order Logic. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP ’24, pages 22017–22031, Miami, FL, USA, Nov. 2024. Association for Computational Linguistics.
- [61] A. Havrilla, S. C. Raparthy, C. Nalmpantis, J. Dwivedi-Yu, M. Zhuravinskyi, E. Hambro, and R. Raileanu. GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning (ICML ’24), volume 235 of Proceedings of Machine Learning Research, pages 17719–17733, Vienna, Austria, July 2024. PMLR.
- [62] C. He, R. Luo, Y. Bai, S. Hu, Z. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun. OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’24, pages 3828–3850, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
- [63] D. Hendrycks, S. Basart, S. Kadavath, M. Mazeika, A. Arora, E. Guo, C. Burns, S. Puranik, H. He, D. Song, and J. Steinhardt. Measuring Coding Challenge Competence with APPS. In J. Vanschoren and S. Yeung, editors, Proceedings of the Thirty-Fifth Neural Information Processing Systems: Track on Datasets and Benchmarks, NeurIPS ’21, Virtual Event, Dec. 2021.
- [64] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring Massive Multitask Language Understanding. In Proceedings of the Ninth International Conference on Learning Representations, ICLR ’21, Virtual Event, May 2021. OpenReview.
- [65] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring Mathematical Problem Solving with the MATH Dataset. In J. Vanschoren and S. Yeung, editors, Proceedings of the Thirty-Fifth Conference on Neural Information Processing Systems: Track on Datasets and Benchmarks, NeurIPS ’21, Virtual Event, Dec. 2021.
- [66] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi. The Curious Case of Neural Text Degeneration. In Proceedings of the Eighth International Conference on Learning Representations, ICLR ’20, Virtual Event, Apr. 2020. OpenReview.
- [67] M. J. Hosseini, H. Hajishirzi, O. Etzioni, and N. Kushman. Learning to Solve Arithmetic Word Problems with Verb Categorization. In A. Moschitti, B. Pang, and W. Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP ’14, pages 523–533, Doha, Qatar, Oct. 2014. Association for Computational Linguistics.
- [68] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. In Proceedings of the Tenth International Conference on Learning Representations, ICLR ’22, Virtual Event, Apr. 2022. OpenReview.
- [69] J. Huang and K. C.-C. Chang. Towards Reasoning in Large Language Models: A Survey. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 1049–1065, Toronto, Canada, July

2023. Association for Computational Linguistics.

- [70] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou. Large Language Models Cannot Self-Correct Reasoning Yet. In Proceedings of the Twelfth International Conference

- on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [71] J. Huang, S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han. Large Language Models Can Self-Improve. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP ’23, pages 1051–1068, Singapore, Dec. 2023. Association for Computational Linguistics.
- [72] Y. Huang, M. Kleindessner, A. Munishkin, D. Varshney, P. Guo, and J. Wang. Benchmarking of Data-Driven Causality Discovery Approaches in the Interactions of Arctic Sea Ice and Atmosphere. Frontiers in Big Data, 4(32):642182:1–642182:19, Aug. 2021.
- [73] S. Imani, L. Du, and H. Shrivastava. MathPrompter: Mathematical Reasoning using Large Language Models. In S. Sitaram, B. Beigman Klebanov, and J. D. Williams, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), ACL ’23, pages 37–42, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [74] A. Q. Jiang, W. Li, J. M. Han, and Y. Wu. LISA: Language Models of ISAbelle Proofs. In Proceedings of the 6th Conference on Artificial Intelligence and Theorem Proving, AITP ’21, Aussois, France, Sept. 2021.
- [75] D. Jiang, X. Ren, and B. Y. Lin. LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’23, pages 14165–14178, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [76] J. Jiang, S. Gan, Y. Liu, F. Wang, G. Alonso, A. Klimovic, A. Singla, W. Wu, and C. Zhang. Towards Demystifying Serverless Machine Learning Training. In Proceedings of the 2021 International Conference on Management of Data, SIGMOD ’21, pages 857–871, Virtual Event, June 2021. Association for Computing Machinery.
- [77] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan. SWE-bench: Can Language Models Resolve RealWorld Github Issues? In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [78] E. Kislev. Relationships 5.0: How AI, VR, and Robots Will Reshape Our Emotional Lives. Oxford University Press, 2022.
- [79] W. Knight. OpenAI Unveils New A.I. That Can ‘Reason’ Through Math and Science Problems. https://www.nytimes.com/2024/1 2/20/technology/openai-new-ai-math-science.html, Dec. 2024. (accessed Dec. 27, 2024).
- [80] L. Kocsis and C. Szepesv´ari. Bandit Based Monte-Carlo Planning. In J. Furnkranz,¨ T. Scheffer, and M. Spiliopoulou, editors, Proceedings of the European Conference on Machine Learning ECML ’06, volume 4212 of Lecture Notes in Computer Science (LNAI), pages 282–293, Berlin, Germany, Sept. 2006. Springer.
- [81] K. Kondo, S. Sugawara, and A. Aizawa. Probing Physical Reasoning with Counter-Commonsense Context. In A. Rogers, J. BoydGraber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL ’23, pages 603–612, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [82] W. Kryscinski, B. McCann, C. Xiong, and R. Socher. Evaluating the Factual Consistency of Abstractive Text Summarization. In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP ’20, pages 9332–9346, Virtual Event, Nov.

2020. Association for Computational Linguistics.

- [83] P. Kumar. Artificial Intelligence: Reshaping Life and Business. BPB Publications, 2019.
- [84] Y. Lai, C. Li, Y. Wang, T. Zhang, R. Zhong, L. Zettlemoyer, W.-T. Yih, D. Fried, S. Wang, and T. Yu. DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 18319–18345, Honolulu, HI, USA, July 2023. PMLR.
- [85] Y. Leviathan, M. Kalman, and Y. Matias. Fast Inference from Transformers via Speculative Decoding. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning (ICML ’23), volume 202 of Proceedings of Machine Learning Research, pages 19274–19286, Honolulu, HI, USA, July 2023. PMLR.

- [86] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler,¨ M. Lewis, W.-t. Yih, T. Rockt¨aschel, S. Riedel, and D. Kiela. Retrieval-Augmented Generation for KnowledgeIntensive NLP Tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Proceedings of the ThirtyFourth Annual Conference on Neural Information Processing Systems (NeurIPS ’20), volume 33 of Advances in Neural Information Processing Systems, pages 9459–9474, Virtual Event, Dec. 2020. Curran Associates.
- [87] X. Li, G. Dong, J. Jin, Y. Zhang, Y. Zhou, Y. Zhu, P. Zhang, and Z. Dou. Search-o1: Agentic Search-Enhanced Large Reasoning Models, Jan. 2025. arXiv:2501.05366.
- [88] X. L. Li, A. Holtzman, D. Fried, P. Liang, J. Eisner, T. Hashimoto, L. Zettlemoyer, and M. Lewis. Contrastive Decoding: OpenEnded Text Generation as Optimization. In A. Rogers, J. BoydGraber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’23, pages 12286–12312, Toronto, Canada, July

2023. Association for Computational Linguistics.

- [89] X. L. Li and P. Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), ACL-IJCNLP ’21, pages 4582–4597, Virtual Event, Aug. 2021. Association for Computational Linguistics.
- [90] M. Liao, C. Li, W. Luo, W. Jing, and K. Fan. MARIO: MAth Reasoning with code Interpreter Output - A Reproducible Pipeline. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 905–924, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
- [91] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let’s Verify Step by Step. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [92] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. DeepSeek-V3 Technical Report, Feb.

2025. arXiv:2412.19437.

- [93] A. Liu, S. Swayamdipta, N. A. Smith, and Y. Choi. WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Findings of the Association for Computational Linguistics: EMNLP 2022, pages 6826–6847, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics.
- [94] C. Liu, J. Shen, H. Xin, Z. Liu, Y. Yuan, H. Wang, W. Ju, C. Zheng, Y. Yin, L. Li, M. Zhang, and Q. Liu. FIMO: A Challenge Formal Dataset for Automated Theorem Proving, Dec. 2023. arXiv:2309.04295.
- [95] X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang, S. Shen, T. Zhang, Y. Su, H. Sun, M. Huang, Y. Dong, and J. Tang. AgentBench: Evaluating LLMs as Agents. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [96] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [97] P. Lu, R. Gong, S. Jiang, L. Qiu, S. Huang, X. Liang, and S.-C. Zhu. Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), ACL-IJCNLP ’21, pages 6774–6786, Virtual Event, Aug.

2021. Association for Computational Linguistics.

- [98] P. Lu, L. Qiu, K.-W. Chang, Y. N. Wu, S.-C. Zhu, T. Rajpurohit, P. Clark, and A. Kalyan. Dynamic Prompt Learning via Policy Gradient for Semi-Structured Mathematical Reasoning. In Proceedings of the Eleventh International Conference on Learning Representations, ICLR ’23, Kigali, Rwanda, May 2023. OpenReview.
- [99] P. Lu, L. Qiu, W. Yu, S. Welleck, and K.-W. Chang. A Survey of Deep Learning for Mathematical Reasoning. In A. Rogers,

- J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’23, pages 14605–14631, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [100] L. Luo, Y. Liu, R. Liu, S. Phatale, M. Guo, H. Lara, Y. Li, L. Shu, Y. Zhu, L. Meng, J. Sun, and A. Rastogi. Improve Mathematical Reasoning in Language Models by Automated Process Supervision, Dec. 2024. arXiv:2406.06592.
- [101] M. Luo, S. Kumbhar, M. Shen, M. Parmar, N. Varshney, P. Banerjee, S. Aditya, and C. Baral. Towards LogiGLUE: A Brief Survey and a Benchmark for Analyzing Logical Reasoning Capabilities of Language Models, Mar. 2024. arXiv:2310.00836.
- [102] Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. ACM Trans. Inf. Syst., 43(2):41:1–41:32, Jan. 2025.
- [103] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark. Self-Refine: Iterative Refinement with Self-Feedback. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS ’23), volume 36 of Advances in Neural Information Processing Systems, pages 46534–46594, New Orleans, LA, USA, Dec. 2023. Curran Associates.
- [104] F. Mai, N. Cornille, and M.-F. Moens. Improving Language Modeling by Increasing Test-Time Planning Compute. In Proceedings of the Eighth Widening NLP Workshop, WiNLP ’24, Miami, FL, USA, Nov. 2024. OpenReview.
- [105] A. Malinin and M. Gales. Uncertainty Estimation in Autoregressive Structured Prediction. In Proceedings of the Ninth International Conference on Learning Representations, ICLR ’21, Virtual Event, May 2021. OpenReview.
- [106] R. Manvi, A. Singh, and S. Ermon. Adaptive Inference-Time Compute: LLMs Can Predict If They Can Do Better, Even MidGeneration, Oct. 2024. arXiv:2410.02725.
- [107] Y. Mao, Y. Kim, and Y. Zhou. CHAMP: A Competition-Level Dataset for Fine-Grained Analyses of LLMs’ Mathematical Reasoning Capabilities. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 13256–13274, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
- [108] A. Masry, X. L. Do, J. Q. Tan, S. Joty, and E. Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics.
- [109] C. Metz. In Two Moves, AlphaGo and Lee Sedol Redefined the Future. https://www.wired.com/2016/03/two-moves-alphago

-lee-sedol-redefined-future/, Mar. 2016. Wired (accessed Feb 7, 2025).

- [110] G. Mialon, C. Fourrier, T. Wolf, Y. LeCun, and T. Scialom. GAIA: A Benchmark for General AI Assistants. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [111] X. Miao, C. Shi, J. Duan, X. Xi, D. Lin, B. Cui, and Z. Jia. SpotServe: Serving Generative Large Language Models on Preemptible Instances. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS ’24, pages 1112–1127, La Jolla, CA, USA, Apr. 2024. Association for Computing Machinery.
- [112] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP ’18, pages 2381– 2391, Brussels, Belgium, Nov. 2018. Association for Computational Linguistics.
- [113] S. I. Mirzadeh, K. Alizadeh, H. Shahrokhi, O. Tuzel, S. Bengio, and M. Farajtabar. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. In Proceedings of the Thirteenth International Conference on Learning Representations, ICLR ’25, Singapore, Apr. 2025. OpenReview.
- [114] J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, and B. Sch¨olkopf. Distinguishing Cause from Effect Using Observational Data:

- Methods and Benchmarks. Journal of Machine Learning Research, 17(32):1–102, 2016.
- [115] P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan, and I. Stoica. Ray: A Distributed Framework for Emerging AI Applications. In Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation, OSDI ’18, pages 561–577, Carlsbad, CA, Oct. 2018. USENIX Association.
- [116] Y. Nie, A. Williams, E. Dinan, M. Bansal, J. Weston, and D. Kiela. Adversarial NLI: A New Benchmark for Natural Language Understanding. In D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL ’20, pages 4885–4901, Virtual Event, July 2020. Association for Computational Linguistics.
- [117] T. Niven and H.-Y. Kao. Probing Neural Network Comprehension of Natural Language Arguments. In A. Korhonen, D. Traum, and L. M`arquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL ’19, pages 4658– 4664, Florence, Italy, July 2019. Association for Computational Linguistics.
- [118] OpenAI. Introducing ChatGPT. https://openai.com/index/cha tgpt/, Nov. 2022. (accessed Dec. 27, 2024).
- [119] OpenAI. Hello GPT-4o. https://openai.com/index/hello-gpt-4 o/, May 2024. (accessed Jan. 1, 2025).
- [120] OpenAI. Introducing OpenAI o1. https://openai.com/o1/,

2024. (accessed Dec. 27, 2024).

- [121] R. Y. Pang, W. Yuan, H. He, K. Cho, S. Sukhbaatar, and J. E. Weston. Iterative Reasoning Preference Optimization. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS ’24), volume 37 of Advances in Neural Information Processing Systems, pages 116617–116637, Vancouver, Canada, Dec. 2024. Curran Associates.
- [122] S. Qiao, Y. Ou, N. Zhang, X. Chen, Y. Yao, S. Deng, C. Tan, F. Huang, and H. Chen. Reasoning with Language Model Prompting: A Survey. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’23, pages 5368–5393, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [123] Y. Qin, X. Li, H. Zou, Y. Liu, S. Xia, Z. Huang, Y. Ye, W. Yuan, H. Liu, Y. Li, and P. Liu. O1 Replication Journey: A Strategic Progress Report – Part 1, Oct. 2024. arXiv:2410.18982.
- [124] Y. Qu, T. Zhang, N. Garg, and A. Kumar. Recursive Introspection: Teaching Language Model Agents How to Self-Improve. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Proceedings of the ThirtyEighth Annual Conference on Neural Information Processing Systems (NeurIPS ’24), volume 37 of Advances in Neural Information Processing Systems, pages 55249–55285, Vancouver, Canada, Dec. 2024. Curran Associates.
- [125] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS ’23), volume 36 of Advances in Neural Information Processing Systems, pages 53728–53741, New Orleans, LA, USA, Dec. 2023. Curran Associates.
- [126] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. GPQA: A Graduate-Level GoogleProof Q&A Benchmark. In Proceedings of the First Conference on Language Modeling, COLM ’24, Philadelphia, PA, USA, Oct. 2024. OpenReview.
- [127] C. D. Rosin. Multi-Armed Bandits with Episode Context. Annals of Mathematics and Artificial Intelligence, 61(3):203–230, Mar. 2011.
- [128] S. Roy and D. Roth. Solving General Arithmetic Word Problems. In L. M`arquez, C. Callison-Burch, and J. Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP ’15, pages 1743–1752, Lisbon, Portugal, Sept.

2015. Association for Computational Linguistics.

- [129] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8732–8740, Apr. 2020.

- [130] M. Sap, H. Rashkin, D. Chen, R. Le Bras, and Y. Choi. Social IQa: Commonsense Reasoning about Social Interactions. In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP ’19, pages 4463–4473, Hong Kong, China, Nov.

2019. Association for Computational Linguistics.

- [131] A. Saparov and H. He. Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought. In Proceedings of the Eleventh International Conference on Learning Representations, ICLR ’23, Kigali, Rwanda, May 2023. OpenReview.
- [132] W. Saunders, C. Yeh, J. Wu, S. Bills, L. Ouyang, J. Ward, and J. Leike. Self-Critiquing Models for Assisting Human Evaluators, June 2022. arXiv:2206.05802.
- [133] T. Sawada, D. Paleka, A. Havrilla, P. Tadepalli, P. Vidas, A. Kranias, J. J. Nay, K. Gupta, and A. Komatsuzaki. ARB: Advanced Reasoning Benchmark for Large Language Models, July 2023. arXiv:2307.13692.
- [134] J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, and D. Silver. Mastering Atari, Go, Chess and Shogi by Planning With a Learned Model. Nature, 588:604–609, Dec. 2020.
- [135] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy Optimization Algorithms, Aug. 2017. arXiv:1707.06347.
- [136] M. Shridhar, X. Yuan, M.-A. Cˆot´e, Y. Bisk, A. Trischler, and M. Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations, ICLR ’21, Virtual Event, May 2021. OpenReview.
- [137] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner,

I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 529:484–489, Jan. 2016.

- [138] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, and D. Hassabis. A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play. Science, 362(6419):1140–1144, Dec. 2018.
- [139] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis. Mastering the Game of Go without Human Knowledge. Nature, 550:354–359, Oct. 2017.
- [140] K. Sinha, S. Sodhani, J. Dong, J. Pineau, and W. L. Hamilton. CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text. In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP ’19, pages 4506–4515, Hong Kong, China, Nov. 2019. Association for Computational Linguistics.
- [141] C. V. Snell, J. Lee, K. Xu, and A. Kumar. Scaling Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters. In Proceedings of the Thirteenth International Conference on Learning Representations, ICLR ’25, Singapore, Apr. 2025. OpenReview.
- [142] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. Transactions on Machine Learning Research, May 2023. Featured Certification.
- [143] S. Srivastava, A. M. B, A. P. V, S. Menon, A. Sukumar, A. S. T, A. Philipose, S. Prince, and S. Thomas. Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap, Feb. 2024. arXiv:2402.19450.
- [144] N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to Summarize with Human Feedback. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Proceedings of the Thirty-Fourth Annual Conference on Neural Information Processing Systems (NeurIPS ’20), volume 33 of Advances in Neural Information Processing Systems, pages 3008–3021, Virtual Event, Dec. 2020. Curran Associates.

- [145] J. Sun, C. Zheng, E. Xie, Z. Liu, R. Chu, J. Qiu, J. Xu, M. Ding, H. Li, M. Geng, Y. Wu, W. Wang, J. Chen, Z. Yin, X. Ren, J. Fu, J. He, Y. Wu, Q. Liu, X. Liu, Y. Li, H. Dong, Y. Cheng, M. Zhang, P. A. Heng, J. Dai, P. Luo, J. Wang, J.-R. Wen, X. Qiu, Y. Guo, H. Xiong, Q. Liu, and Z. Li. A Survey of Reasoning with Foundation Models: Concepts, Methodologies, and Outlook. ACM Comput. Surv., Apr. 2025.
- [146] R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2015.
- [147] R. Tadeusiewicz and L. Ogiela. Modern Methods for the Cognitive Analysis of Economic Data and Text Documents and Their Application in Enterprise Management. In V. Sn´aˇsel, A. Abraham, K. Saeed, and J. Pokorn´y, editors, Proceedings of the 7th Computer Information Systems and Industrial Management Applications, CISIM ’08, pages 11–23, Ostrava, Czech Republic, June 2008. IEEE Press.
- [148] R. Tadeusiewicz, L. Ogiela, and M. R. Ogiela. Cognitive Analysis Techniques in Business Planning and Decision Support Systems. In L. Rutkowski, R. Tadeusiewicz, L. A. Zadeh, and J. M. Zurada,˙ editors, Proceedings of the 8th International Conference on Artificial Intelligence and Soft Computing (ICAISC ’06), volume 4029 of Lecture Notes in Computer Science (LNCS), pages 1027–1039, Zakopane, Poland, June 2006. Springer.
- [149] O. Tafjord, B. Dalvi, and P. Clark. ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634, Virtual Event, Aug. 2021. Association for Computational Linguistics.
- [150] A. Talmor, J. Herzig, N. Lourie, and J. Berant. CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), NAACL ’19, pages 4149–4158, Minneapolis, MN, USA, June 2019. Association for Computational Linguistics.
- [151] Z. Tang, X. Zhang, B. Wang, and F. Wei. MathScale: Scaling Instruction Tuning for Mathematical Reasoning. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning (ICML ’24), volume 235 of Proceedings of Machine Learning Research, pages 47885–47900, Vienna, Austria, July 2024. PMLR.
- [152] Q. Team. QwQ: Reflect Deeply on the Boundaries of the Unknown. https://qwenlm.github.io/blog/qwq-32b-preview/, Nov. 2024. (accessed Jan. 1, 2025).
- [153] Y. Tian, B. Peng, L. Song, L. Jin, D. Yu, L. Han, H. Mi, and D. Yu. Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS ’24), volume 37 of Advances in Neural Information Processing Systems, pages 52723–52748, Vancouver, Canada, Dec.

2024. Curran Associates.

- [154] R. Tu, K. Zhang, B. Bertilson, H. Kjellstrom, and C. Zhang. Neuropathic Pain Diagnosis Simulator for Causal Discovery Algorithm Evaluation. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´e-Buc, E. Fox, and R. Garnett, editors, Proceedings of the Thirty-Third Annual Conference on Neural Information Processing Systems (NeurIPS ’19), volume 32 of Advances in Neural Information Processing Systems, pages 12793–12804, Vancouver, Canada, Dec.

2019. Curran Associates.

- [155] J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins. Solving Math Word Problems with Process-and Outcome-Based Feedback, Nov. 2022. arXiv:2211.14275.
- [156] A. Vijayakumar, M. Cogswell, R. Selvaraju, Q. Sun, S. Lee, D. Crandall, and D. Batra. Diverse Beam Search for Improved Description of Complex Scenes. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1):7371–7379, Apr. 2018.
- [157] Z. Wan, X. Feng, M. Wen, S. M. McAleer, Y. Wen, W. Zhang, and

- J. Wang. AlphaZero-Like Tree-Search Can Guide Large Language Model Decoding and Training. In R. Salakhutdinov, Z. Kolter,
- K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine

Learning (ICML ’24), volume 235 of Proceedings of Machine Learning Research, pages 49890–49920, Vienna, Austria, July 2024. PMLR.

- [158] J. Wang, M. Fang, Z. Wan, M. Wen, J. Zhu, A. Liu, Z. Gong, Y. Song, L. Chen, L. M. Ni, L. Yang, Y. Wen, and W. Zhang. OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models, Oct. 2024. arXiv:2410.09671.
- [159] K. Wang, H. Ren, A. Zhou, Z. Lu, S. Luo, W. Shi, R. Zhang, L. Song, M. Zhan, and H. Li. MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [160] P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-Shepherd: Verify and Reinforce LLMs Stepby-Step without Human Annotations. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’24, pages 9426–9439, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
- [161] X. Wang, Z. Hu, P. Lu, Y. Zhu, J. Zhang, S. Subramaniam, A. Loomba, S. Zhang, Y. Sun, and W. Wang. SCIBENCH: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models. In Proceedings of the 3rd Workshop on Mathematical Reasoning and AI, MATH-AI ’23, New Orleans, LA, USA, Dec. 2023. OpenReview.
- [162] X. Wang, L. Song, Y. Tian, D. Yu, B. Peng, H. Mi, F. Huang, and D. Yu. Towards Self-Improvement of LLMs via MCTS: Leveraging Stepwise Knowledge with Curriculum Preference Learning, Oct. 2024. arXiv:2410.06508.
- [163] X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings of the Eleventh International Conference on Learning Representations, ICLR ’23, Kigali, Rwanda, May 2023. OpenReview.
- [164] Z. Wang, S. Zhou, D. Fried, and G. Neubig. Execution-Based Evaluation for Open-Domain Code Generation. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1271–1290, Singapore, Dec. 2023. Association for Computational Linguistics.
- [165] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Proceedings of the Thirty-Sixth Annual Conference on Neural Information Processing Systems (NeurIPS ’22), volume 35 of Advances in Neural Information Processing Systems, pages 24824– 24837, New Orleans, LA, USA, Dec. 2022. Curran Associates.
- [166] P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, and L. Thamsen. Let’s Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. In Proceedings of the 22nd International Middleware Conference, Middleware ’21, pages 260–272, Virtual Event, Dec. 2021. Association for Computing Machinery.
- [167] Z. Xi, Y. Ding, W. Chen, B. Hong, H. Guo, J. Wang, D. Yang, C. Liao, X. Guo, W. He, S. Gao, L. Chen, R. Zheng, Y. Zou, T. Gui, Q. Zhang, X. Qiu, X. Huang, Z. Wu, and Y.-G. Jiang. AgentGym: Evolving Large Language Model-Based Agents Across Diverse Environments, June 2024. arXiv:2406.04151.
- [168] Y. Xie, A. Goyal, W. Zheng, M.-Y. Kan, T. P. Lillicrap, K. Kawaguchi, and M. Shieh. Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning. In Proceedings of the Workshop on System-2 Reasoning at Scale, Sys2-Reasoning ’24, Vancouver, Canada, Dec. 2024. OpenReview.
- [169] G. Xiong, Q. Jin, Z. Lu, and A. Zhang. Benchmarking RetrievalAugmented Generation for Medicine. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 6233–6251, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
- [170] J. Xiong, J. Shen, Y. Yuan, H. Wang, Y. Yin, Z. Liu, L. Li, Z. Guo, Q. Cao, Y. Huang, C. Zheng, X. Liang, M. Zhang, and Q. Liu. TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP ’23, pages 11594–11632, Singapore, Dec. 2023. Association for Computational Linguistics.
- [171] Y. Yan, J. Su, J. He, F. Fu, X. Zheng, Y. Lyu, K. Wang, S. Wang, Q. Wen, and X. Hu. A Survey of Mathematical Reasoning in the

- Era of Multimodal Large Language Model: Benchmark, Method & Challenges, Dec. 2024. arXiv:2412.11936.
- [172] K. Yang, A. Swope, A. Gu, R. Chalamala, P. Song, S. Yu, S. Godil, R. J. Prenger, and A. Anandkumar. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS ’23), volume 36 of Advances in Neural Information Processing Systems, pages 21573– 21612, New Orleans, LA, USA, Dec. 2023. Curran Associates.
- [173] S. Yao, H. Chen, J. Yang, and K. Narasimhan. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Proceedings of the Thirty-Sixth Annual Conference on Neural Information Processing Systems (NeurIPS ’22), volume 35 of Advances in Neural Information Processing Systems, pages 20744–20757, New Orleans, LA, USA, Dec. 2022. Curran Associates.
- [174] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS ’23), volume 36 of Advances in Neural Information Processing Systems, pages 11809–11822, New Orleans, LA, USA, Dec. 2023. Curran Associates.
- [175] N. Young, Q. Bao, J. Bensemann, and M. Witbrock. AbductionRules: Training Transformers to Explain Unexpected Inputs. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 218– 227, Dublin, Ireland, May 2022. Association for Computational Linguistics.
- [176] L. Yuan, W. Li, H. Chen, G. Cui, N. Ding, K. Zhang, B. Zhou, Z. Liu, and H. Peng. Free Process Rewards without Process Labels, Dec. 2024. arXiv:2412.01981.
- [177] Z. Yuan, H. Yuan, C. Tan, W. Wang, and S. Huang. How Well Do Large Language Models Perform in Arithmetic Tasks?, Mar.

2023. arXiv:2304.02015.

- [178] R. Zellers, Y. Bisk, R. Schwartz, and Y. Choi. SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP ’18, pages 93–104, Brussels, Belgium, Nov.

2018. Association for Computational Linguistics.

- [179] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a Machine Really Finish Your Sentence? In A. Korhonen, D. Traum, and L. M`arquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL ’19, pages 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics.
- [180] Z. Zeng, Q. Cheng, Z. Yin, B. Wang, S. Li, Y. Zhou, Q. Guo, X. Huang, and X. Qiu. Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective, Dec. 2024. arXiv:2412.14135.
- [181] D. Zhang, X. Huang, D. Zhou, Y. Li, and W. Ouyang. Accessing GPT-4 Level Mathematical Olympiad Solutions via Monte Carlo Tree Self-Refine with LLaMa-3 8B, June 2024. arXiv:2406.07394.
- [182] D. Zhang, J. Wu, J. Lei, T. Che, J. Li, T. Xie, X. Huang, S. Zhang, M. Pavone, Y. Li, W. Ouyang, and D. Zhou. LLaMA-Berry: Pairwise Optimization for Olympiad-Level Mathematical Reasoning via O1-like Monte Carlo Tree Search. In L. Chiruzzo, A. Ritter, and L. Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL ’25, pages 7315–7337, Albuquerque, NM, USA, Apr.

2025. Association for Computational Linguistics.

- [183] D. Zhang, S. Zhoubian, Z. Hu, Y. Yue, Y. Dong, and J. Tang. ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS ’24), volume 37 of Advances in Neural Information Processing Systems, pages 64735–64772, Vancouver, Canada, Dec.

2024. Curran Associates.

- [184] L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, and R. Agarwal. Generative Verifiers: Reward Modeling as NextToken Prediction. In Proceedings of the 4th Workshop on Mathe-

matical Reasoning and AI, MATH-AI ’24, Vancouver, Canada, Dec.

2024. OpenReview.

- [185] M. Zhao, S. Pan, N. Agarwal, Z. Wen, D. Xu, A. Natarajan, P. Kumar, S. S. P, R. Tijoriwala, K. Asher, H. Wu, A. Basant, D. Ford, D. David, N. Yigitbasi, P. Singh, C.-J. Wu, and C. Kozyrakis. Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training. In Proceedings of the USENIX Annual Technical Conference, ATC ’23, pages 433–449, Boston, MA, USA, July 2023. USENIX Association.
- [186] Y. Zhao, Y. Li, C. Li, and R. Zhang. MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’22, pages 6588–6600, Dublin, Ireland, May 2022. Association for Computational Linguistics.
- [187] Y. Zhao, H. Yin, B. Zeng, H. Wang, T. Shi, C. Lyu, L. Wang, W. Luo, and K. Zhang. Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions, Nov. 2024. arXiv:2411.14405.
- [188] K. Zheng, J. M. Han, and S. Polu. miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics. In Proceedings of the Tenth International Conference on Learning Representations, ICLR ’22, Virtual Event, Apr. 2022. OpenReview.
- [189] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS ’23), volume 36 of Advances in Neural Information Processing Systems, pages 46595–46623, New Orleans, LA, USA, Dec. 2023. Curran Associates.
- [190] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, U. Alon, and G. Neubig. WebArena: A Realistic Web Environment for Building Autonomous Agents. In Proceedings of the Twelfth International Conference on Learning Representations, ICLR ’24, Vienna, Austria, May 2024. OpenReview.
- [191] D.-H. Zhu, Y.-J. Xiong, J.-C. Zhang, X.-J. Xie, and C.-M. Xia. Understanding Before Reasoning: Enhancing Chain-ofThought with Iterative Summarization Pre-Prompting, Jan. 2025. arXiv:2501.04341.

