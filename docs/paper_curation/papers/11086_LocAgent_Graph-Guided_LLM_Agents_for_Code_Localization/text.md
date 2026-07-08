## LocAgent: Graph-Guided LLM Agents for Code Localization

Zhaoling Chen♠∗, Xiangru Tang♠*, Gangda Deng♦∗, Fang Wu♣, Jialong Wu♠, Zhiwei Jiang, Viktor Prasanna♦, Arman Cohan♠, Xingyao Wang♥ ♠Yale University ♦University of Southern California ♣Stanford University ♥All Hands AI xiangru.tang@yale.edu, gangdade@usc.edu, xingyao@all-hands.dev

Issue Description

###### Localized Code

### Abstract

[Figure 1]

###### Issue

Bug report

File: /forms/widgets.py class: Media function: Media._css

[Figure 2]

file: ./forms/widgets.py class: Media function: merge

###### Merging 3 or more media objects can throw unnecessary MediaOrderConflictWarnings.

Code localization–identifying precisely where in a codebase changes need to be made–is a fundamental yet challenging task in software maintenance. Existing approaches struggle to efficiently navigate complex codebases when identifying relevant code sections. The challenge lies in bridging natural language problem descriptions with the appropriate code elements, often requiring reasoning across hierarchical structures and multiple dependencies. We introduce LOCAGENT, a framework that addresses code localization through graph-based representation. By parsing codebases into directed heterogeneous graphs, LOCAGENT creates a lightweight representation that captures code structures (files, classes, functions) and their dependencies (imports, invocations, inheritance), enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning. Experimental results on real-world benchmarks demonstrate that our approach significantly enhances accuracy in code localization. Notably, our method with the fine-tuned Qwen-2.5-Coder-Instruct-32B model achieves comparable results to SOTA proprietary models at greatly reduced cost (approximately 86% reduction), reaching up to 92.7% accuracy on file-level localization while improving downstream GitHub issue resolution success rates by 12% for multiple attempts (Pass@10). Our code is available at https:

c

f

...

Codebase

# arXiv:2503.09089v2[cs.SE]29Apr2025

[Figure 3]

###### Issue

Feature Request

[Figure 4]

###### django

file: django/shortcuts.py function: redirect

[Figure 5]

file: ./http/response.py class: HttpResponseRedirectBase

Add 307 and 308 redirect response codes to django.shortcuts.redirect.

[Figure 6]

c

[Figure 7]

...

[Figure 8]

[Figure 9]

[Figure 10]

Security

###### Issue

[Figure 11]

file:./middleware/csrf.py Class: CsrfViewMiddleware function: process_response

file:django/middleware/c srf.py Class:CsrfViewMiddleware

… …

[Figure 12]

[Figure 13]

c

Prevent repetitive output to counter BREACH-type attacks. ...

f

6f046d7

[Figure 14]

###### Issue

file: django/contrib/ auth/management/__init__. py function:create_permissio ns

Performance

[Figure 15]

file: ./contrib/auth/man agement/__init__.py

Optimize post-migrate permission creation.

function: create_permissions

f

...

Figure 1: Code localization across four common programming scenarios. Given a codebase and an issue description, the goal of code localization is to identify the relevant code snippets that require modification to resolve the issue.

make changes, and automated tools often struggle with the same challenge. Poor code localization leads to incomplete fixes, introduces new bugs, and significantly extends development cycles. Unlike traditional retrieval tasks that primarily focus on lexical or semantic matching between queries and documents (Guo et al., 2016, 2020), code localization requires bridging the gap between natural language and programming languages. It also necessitates reasoning capabilities to analyze the issue, while considering the structural and semantic properties of code (Lewis et al., 2020; Guu et al., 2020; Qu et al., 2020). This capability has become fundamental to powerful AI assistants (OpenAI, 2023; Anthropic, 2023), code-aware search engines (PerplexityAI, 2023), and automated programming agents (Cognition.ai, 2024; Wang et al., 2025; Gauthier, 2024). In particular, accurate code localization is crucial for software maintenance and evolution, as it enables precise code modifications for bug fixes, refactoring, and feature additions (Wang et al., 2024), thereby streamlining the development workflow.

//github.com/gersteinlab/LocAgent.

### 1 Introduction

Code localization can be viewed as an information retrieval (IR) task that aims to identify relevant code snippets given natural language descriptions (Yu et al., 2025; Yang et al., 2024; Xia et al., 2024). Developers spend up to 66% of their debugging time (Böhme et al., 2017) understanding code to

* Equal contribution. This work was done during Zhaoling’s time at Yale.

Existing approaches to code localization face

significant limitations. Dense retrieval methods require maintaining and continuously updating vector representations of the entire codebase (Wang et al., 2023b; Günther et al., 2023), creating engineering challenges for large, evolving repositories where code changes frequently. While LLMs demonstrate strong code understanding capabilities (Kang et al., 2023; Wu et al., 2023), models with large context windows cannot process entire codebases at once, necessitating strategic navigation through relevant parts. Moreover, issue descriptions often mention only symptoms rather than underlying causes. For instance, a report of ‘XSS vulnerability in user profile’ might require changes to a shared validation utility used throughout the codebase but not explicitly referenced in the issue. This disconnect between issue descriptions and affected code components presents a substantial challenge for traditional retrieval approaches, which struggle to trace implicit dependencies across the codebase structure. Recent agent-based methods attempt to address these limitations through iterative exploration (Yang et al., 2024; Qin et al., 2024) but still struggle to efficiently navigate and comprehend complex code structures and dependencies, particularly when multi-hop reasoning is required to trace from issue descriptions to affected code regions that aren’t directly mentioned.

This raises a key question: How can we design efficient indexing as intermediate representations that are structure-aware and both easy and performant for LLM agents to consume? It is intuitive to design an agentic retrieval system that carefully combines traditional IR methods and LLM agent’s reasoning ability to achieve accurate, efficient, and cost-effective code localization in codebases.

To address this challenge, we propose LOCAGENT, a framework that builds directed heterogeneous graph indexing to unify code structures, dependencies, and contents. Our approach leverages a structured graph representation that enables powerful multi-hop reasoning capabilities, allowing agents to navigate complex dependency relationships between code elements even when target code isn’t explicitly mentioned in issue descriptions. This graph-based approach significantly outperforms previous methods on challenging localization tasks that require traversing multiple code relationships. Our lightweight representation, coupled with sparse indexing techniques, enables efficient entity search while maintaining rich structural information. The indexing process typically

takes only a few seconds per codebase, making it highly practical for real-time use. The framework integrates a set of unified tools that guide the agent through a systematic exploration of the codebase, allowing autonomous navigation based on contextual needs. Furthermore, by fine-tuning Qwen-2.5-Coder-Instruct (Hui et al., 2024) 7B and 32B models(abbr. as Qwen-2.5-7B and Qwen-2.5-32B respectively), our system achieves performance comparable to state-of-the-art models like Claude-3-5-sonnet-20241022 (Anthropic, 2023) (abbr. as Claude-3.5) while significantly reducing API costs by over 80% (from $0.66 to $0.09 per example), making it practical for realworld deployment.

Additionally, to facilitate a comprehensive evaluation of code localization methods, we introduce LOC-BENCH, a new benchmark specifically designed for this task. Existing benchmarks like SWE-Bench present significant limitations: (1) they risk contamination through data overlap with LLM training sets (Mündler et al., 2024), and (2) they primarily focus on bug fixing, lacking diversity in maintenance scenarios such as feature requests, performance optimizations, and security fixes. In contrast, LOC-BENCH covers diverse scenarios and mitigates potential contamination concerns by incorporating more recent examples from popular Python repositories collected after known LLM training cutoff dates. Additionally, we provide tooling to continuously update the benchmark with new examples, allowing researchers to maintain a fresh evaluation dataset as models evolve and training data cutoffs advance.

Our contributions address critical gaps in existing approaches:

- • We introduce a heterogeneous graph representation that captures both explicit and implicit code relationships, enabling efficient multihop reasoning. Our lightweight graph-based indexing process takes only seconds per repository and requires minimal storage.
- • We design unified tools for agent-based code exploration that leverage our graph representation, allowing LLM agents to perform complex multi-hop navigation and reasoning across code dependencies even when target code isn’t explicitly mentioned in issue descriptions.
- • We introduce Loc-Bench, a new benchmark

specifically designed for code localization that addresses limitations in existing datasets. Unlike previous benchmarks dominated by bug reports, Loc-Bench offers a balanced distribution across bug fixes, feature requests, security patches, and performance optimizations.

• By fine-tuning open-source models on this task, we reduce the cost of code localization by 86% while maintaining competitive performance.

### 2 Related Work

#### 2.1 Traditional Retrieval-based Methods

Traditional IR methods rely on lexical or semantic matching to return ranked lists of code snippets. Sparse retrievers, such as BM25 (Robertson et al., 1994, 2009), have demonstrated robustness to domain adaptation. Dense retrievers utilize embeddings for improved semantic searching, including models with open checkpoints such as general text embedding models E5-base-v2 (Wang et al., 2022) and proprietary APIs (VoyageAI, 2024). Code embedding models such as Jina-Code-v2 (Günther et al., 2023), Codesage-large-v2 (Zhang et al., 2024), and CodeRankEmbed (Suresh et al., 2024), trained specifically for code related tasks, showing significant performance in Code2Code and NL2Code semantic search tasks. However, while the embedding models themselves are small, the engineering challenges of maintaining these indexing systems (e.g., storage requirements, update mechanisms, and infrastructure maintenance) make them difficult to adapt to fast-evolving codebases.

#### 2.2 LLM-based Generative Retrieval Methods

Recently, LLMs with advanced code reasoning capabilities have demonstrated superior performance by directly processing queries and raw code for code localization (Kang et al., 2023; Wu et al., 2023; Xia et al., 2024; Kang et al., 2024). For example, Agentless (Xia et al., 2024), initially designed for automated program repair, uses a simplistic hierarchical localization process powered by LLM. It employs a straightforward three-phase approach that first localizes relevant code sections before attempting to fix the identified issues, challenging the assumption that complex agent architectures are necessary for effective code understanding and modification tasks.

Expanding on these techniques, agent-based methods utilize multi-step reasoning to enable automated codebase traversal. Specifically, OpenHands (Wang et al., 2025) implements a generalist coding agent that supports bash commands like grep and tools for viewing files. SWEAgent (Yang et al., 2024) integrates a custom Agent-Computer Interface to support agents to navigate entire repositories. MoatlessTools (Örwall, 2024) combines an agentic searching loop and semantic search to obtain code locations. However, existing agent-based methods face two critical limitations: (a) they primarily navigate codebases through directory traversal rather than understanding semantic relationships, (b) and they struggle to extract and reason about complex cross-file dependencies when these relationships aren’t explicitly represented in the repository structure. This significantly impairs their ability to locate code that requires modification when the issue involves interactions between structurally distant components in the codebase.

2.3 Graph-based Code Representation Methods

Due to the inherent structure of code, several works have employed graph-based representations to improve code understanding by capturing key relationships between components. Aider (2023) constructs a RepoMap and uses a graph ranking algorithm to identify the most significant contextual elements. Similarly, as a plugin, RepoGraph (Ouyang et al., 2025) performs subgraph retrieval – extracting an ego-network of relevant lines and their neighbors – to provide structured context. CodexGraph (Liu et al., 2024) indexes the repository into a Neo4j graph database, where LLM agents query the database precisely using Cypher. The efficiency of its retrieval process depends heavily on the querying capabilities of the LLM. These methods focus primarily on providing relevant context but do not enhance the traversal process itself, as they do not explicitly model directory structure or file hierarchies.

In contrast, RepoUnderstander (Ma et al., 2024) builds hierarchical and function-call graphs, using Monte Carlo Tree Search (MCTS) guided by an LLM for exploration. While thorough, MCTS introduces extra computational overhead, making it less efficient than simpler traversal methods like BFS, particularly in large repositories. OrcaLoca (Yu et al., 2025) uses a simplified graph

[Figure 16]

###### Codebase Indexing

###### Agent Runtime

[Figure 17]

[Figure 18]

Issue

[Figure 19]

[Figure 20]

contain

More Accurate

[Figure 21]

[Figure 22]

Autoreloader with StatReloader doesn't track changes in manage.py. …

[Figure 23]

[Figure 24]

c

[Figure 25]

Entity ID Index

c

f

…

c

…

f

f

Task Description: Please localize related modules…

[Figure 26]

[Figure 27]

f

Entity Name Index

[Figure 28]

[Figure 29]

invoke

[Figure 30]

BM25 Index on Entity IDs

[Figure 31]

[Figure 32]

Observation

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Tools

c

###### Codebase

[Figure 37]

c

f

…

BM25 Index on Entity Contents

[Figure 38]

c

Event Log

…

f

[Figure 39]

f

[Figure 40]

SearchEntity

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

f

###### 2 Entity Indexing

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

import

TraverseGraph

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Action

[Figure 60]

[Figure 61]

[Figure 62]

c

[Figure 63]

LLM Agent

[Figure 64]

c

…

f

RetrieveEntity

c

…

f

f

[Figure 65]

File

[Figure 66]

f

[Figure 67]

[Figure 68]

[Figure 69]

Localized Code Sections

inherit

[Figure 70]

Directory

[Figure 71]

[Figure 72]

[Figure 73]

Result(2) file: path2/b.py line: 30 function: func2

Result(1) file: path1/a.py line: 10 class: Clazz1 function: func1

[Figure 74]

c

. . .

c

f

…

f Function c Class

c

…

f

f

f

1 Code Graph Indexing

Figure 2: Overview of LOCAGENT framework. LOCAGENT first parses the given codebase to build a graph-based code representation with various types of entities and relations. It then constructs sparse indexes for exploring structures and searching content. Using these indexes, it performs agent-guided searches that combine the graph and tools.

Relation Types Node Types

Method

Search/Traversal Strategy Contain Import Inherit Invoke Directory File Class Function

✗ ✔ ✔ ✗ ✗ ✔ ✔ Cypher queries RepoGraph(Ouyang et al., 2025) ✔

CodexGraph(Liu et al., 2024) ✔

✗

✗ ✔ ✔ ✗ ✗ ✔ ✔ Ego-graph retrieval RepoUnderstander(Ma et al., 2024) ✔ ✗ ✔ ✔ ✔ ✔ ✔ ✔ MCTS OrcaLoca(Yu et al., 2025) ✔ ✗ ✗ ✔

✗

✔ ✔ ✔ ✔ Simple search tools LOCAGENT(Ours) ✔ ✔ ✔ ✔ ✔ ✔ ✔ ✔ Unified retrieval tools

✗

Table 1: Comparison of Graph-Based Code Representation Methods.

enhanced by priority scheduling and context pruning. It maintains efficient search but may miss complex invocation dependencies. Table 1 summarizes the differences between these methods and LOCAGENT. Compared to these approaches, LOCAGENT offers a more comprehensive and unified representation of the repository, along with efficient, unified retrieval tools specifically designed for LLM consumption.

representation for codebases. Section 3.2 presents our agent-based code search on the indexes and Section 3.3 describes our model fine-tuning and distillation process.

#### 3.1 Graph-based Code Representation

Codebases contain rich structural information, both explicit and implicit, that is essential for agent reasoning. Building on this insight, we develop a graph-based indexing that comprehensively captures codebase relationships while maintaining a granularity suitable for LLM-agents to retrieve.

### 3 The LOCAGENT Framework

We introduce LOCAGENT, a graph-oriented LLMagent framework for code localization. Figure 2 illustrates the overall framework. When given a repository, LOCAGENT can locate all the relevant code sections at various granularities (file, class, function, or line level) for different types of GitHub issues (such as bug reports, feature requests, performance bottlenecks, and security vulnerabilities) through automated in-depth exploration and analysis of the codebase. Section 3.1 proposes a novel graph-based indexing approach as an intermediate

Code Graph Construction. We construct a heterogeneous directed graph G(V,E,A,R) to index the codebase, where V = {vi}ni=1 is the node set and E ⊆ V × V is the edge set. Each node v ∈ V and edge e ∈ E has an associated type mapping function. For nodes, τ(v) ∶ V → A maps to types A = {directory,file,class,function}. For edges, ϕ(e) ∶ E → R maps to relationships R = {contain,import,invoke,inherit}. In this paper, we focus our study on Python reposito-

ries and leave codebases with other programming languages as future work.

First, we include all directories and Python files as nodes. Then, we parse each Python file using the abstract syntax tree (AST) to identify inner functions and classes recursively as nodes. We set the function level as the smallest node granularity and use each function’s code content as the document for agent retrieval. This approach creates a good balance of information density between the index and documents, allowing LLMs to reason effectively within their context window limitations.

As shown in Figure 2, all nodes with different types can be connected as a single tree using the contain relationship. This structure supports standard codebase-navigation operations from existing works. Our code graph further incorporates more advanced codebase relationships as edges: (1) the invoke relationship from function/class to function/class, where an invoke to a class represents class instantiation; (2) the import relationship from file to function/class; and (3) the inherit relationship between classes.

Sparse Hierarchical Entity Indexing. We treat nodes in our code graph as entities and build hierarchical indexing based on their contents. For each keyword, we lookup the indexes from top to bottom: (1) We build an entity ID index as a unique identifier for each node using its fully qualified name. For example, a function calculate_sum in the MathUtils class located in src/utils.py would be represented as: src/utils.py:MathUtils.calculate_sum. (2) We construct a global dictionary to map the entity name (e.g., calculate_sum) to all nodes that share the same name. (3) We index entity IDs through an inverted index (i.e., BM25) to handle keyword searches that don’t exactly match the IDs or names of entities. (4) For cases where input keywords aren’t part of the entities’ IDs (e.g., when a keyword refers to a global variable), we build an inverted index that maps code chunk(s) to each entity to cover all possible matches.

Remark. Rather than relying solely on directory structures or hierarchical module indexing, our approach captures module dependencies that transcend directory boundaries. Two modules in distant directories (A and B) may appear unrelated in traditional navigation, but if they invoke each other or share inheritance, they’re syntactically close in our graph representation. This syntactic

Tool Name Input Params Output

Related Entities with Code Snippets

SearchEntity Keywords

Start Entity IDs

Direction Traverse Hops Entity Types Relation Types

Traversed Subgraph, including Entites and Relations

TraverseGraph

Complete Code of Specified Entities

RetrieveEntity Entity IDs

Table 2: List of unified APIs provided by LocAgent for code search and exploration.

proximity is essential for code localization because issues typically manifest through call relationships rather than directory structure. By capturing these functional dependencies, our approach efficiently identifies related components even when physically distant in the codebase.

#### 3.2 Agent-guided Code Search

We develop tools based on the indexes built offline. During runtime, LOCAGENT takes issue statements as input and launches agents that autonomously use tools to localize target code sections. While the agent may iteratively invoke multiple tools internally to explore the codebase, LOCAGENT presents a simplified interface to users, requiring only a single-turn interaction—users submit an issue statement and receive localization results without additional input. This autonomous, self-contained workflow makes LOCAGENT both easy to deploy and highly practical for real-world use.

Tool Design for Codebase Exploration. Recent works (Örwall, 2024; Wang et al., 2025), inspired by GUI-based IDEs, have developed numerous specialized tools for agents to explore codebases. However, these tools are initially designed for human readability, which sacrifices the compactness and efficiency that LLM agents prefer (Yang et al., 2024). Building upon our graph-based code representation, we can develop tools that support efficient higher-order codebase exploration to address these challenges. We unify all codebase navigation, search, and view operations into three tools (Table 2), introduced as follows.

SearchEntity: This tool searches codebases using keywords to locate relevant entities through our Hierarchical Entity Index. When an exact match isn’t found in the upper index, the system performs a fuzzy search using the lower index. For each entity found, we return its code snippet in three detail

levels: fold, preview, and full code (Figure 6). This effectively prevents lengthy code context and reduces noise fed into agents.

TraverseGraph: This tool performs a typeaware breadth-first search (BFS) on the code graph, starting from input entities and allowing control over both traversal direction and number of hops. This supports agents to perform arbitrary multihop codebase navigation through only one action, significantly improving the efficiency compared with existing agent systems. Note that by allowing agents to select entity types and relation types for each traversal, this tool effectively leverages the LLM agents’ coding expertise to generate proper meta paths—a crucial element for heterogeneous graph analysis (Lv et al., 2021). For example, by specifying entity types to {class, function} and relation types to {contain, inherit}, this tool returns the UML diagram. Additionally, we design an expanded tree-based format for the output subgraph that encodes both relation types and directions (Figure 7). (Fatemi et al., 2023) demonstrates that LLM performance on graph reasoning depends on the input graph format. Converting a graph into a tree structure encodes topology through the spatial distance between entity names, thereby deriving better performance. For detailed comparisons with alternative graph formats, please see Appendix A.1.2.

RetreiveEntity: This tool retrieves complete entity attributes for each input entity ID, including essential information such as file path, line number, and code content.

Chain-of-Thought Agent Planning. We use chain-of-thought (CoT) prompting (shown in Appendix D) to guide the agent in solving code localization problems step by step. The agent systematically follows these steps: (1) Keyword extraction. The agent begins by breaking down the issue statement into different categories and then extracts relevant keywords that are closely related to the problem. (2) Linking keywords to code entities. The agent invokes SearchEntity to complete and clarify each extracted keyword.

- (3) Generate the logical flow from fault to failure. The agent first identifies the entry points that trigger the problem. Then, it iteratively traverse the codebase with TraverseGraph, retrieves code contents with RetrieveEntity, and searches new keywords with SearchEntity. Finally, it generates the logic flow based on the issue and additional context.
- (4) Locate the target entities. The agent pinpoints all suspicious code entities that need modification

based on the logic flow. Then, it ranks these entities based on their relevance.

Confidence Estimation Based on Consistency. After generating a complete ranked list of candidate entities, to obtain a more consistent ranking, we measure the consistency (Wang et al., 2023a) of the LLM’s predictions across multiple iterations. Specifically, we use the Reciprocal Rank as the initial confidence score for each predicted location. We then aggregate the scores for each entity across iterations to compute its final confidence score. The intuition behind this approach is that if the LLM consistently ranks a location higher in multiple iterations, it is more likely to be relevant.

#### 3.3 Open-source Model Fine-tuning

Given the high costs of proprietary LLM APIs and data security concerns, we fine-tuned open-source models to improve their code localization capabilities and enable local deployment. We collect 433 successful trajectories generated with Claude-3.5, where the agent completed tasks from the SWEbench training set. Due to budget constraints, we sample an additional 335 trajectories generated by the initially fine-tuned Qwen2.5-32B model. Importantly, we only select successful trajectories where the model correctly localized the issues, creating a high-quality dataset of correct reasoning paths. These successful examples are then used to refine the same 32B model further, reinforcing effective reasoning patterns through this self-improvement loop. The entire dataset, combining both Claude-

- 3.5 trajectories and successful Qwen2.5-32B samples, was then used to distill knowledge to a smaller 7B model.

To fine-tune the smaller model, we employ Supervised Fine-Tuning (SFT) with LoRA (Hu et al., 2021). Our experiments show that this straightforward distillation method significantly enhances the performance of smaller models. See Appendix C.1.3 for more training details.

4 LOC-BENCH: A New Benchmark for Code Localization

- 4.1 Revisiting Existing Benchmark

SWE-Bench(Jimenez et al., 2023) is a widely used benchmark that collects GitHub issues and corresponding code patches that resolve them. Xia et al. (2024); Suresh et al. (2024) adapt its subset, SWE-Bench-Lite, for code localization, treating the patched files and functions as the targets.

However, existing datasets, including SWE-Bench, present challenges for effectively evaluating code localization methods. First, they are at risk of contamination, as they may include data overlapping with the repositories or issues used by modern models during pre-training. Second, existing datasets are not specifically designed for code localization (Tomassi et al., 2019). SWE-Bench, for instance, was created primarily to evaluate end-toend bug-fixing capabilities, with localization being only an implicit intermediate step. This focus results in datasets dominated by bug reports (85% of SWE-Bench-Lite examples) while severely underrepresenting other common software maintenance tasks such as feature requests (14%), security vulnerabilities (1%), and performance optimizations (0%). This imbalance fails to capture the diverse localization challenges faced in real-world software development.

#### 4.2 Dataset Construction

To address the limitations of existing benchmarks, we introduce LOC-BENCH, a new dataset specifically designed for code localization. This dataset collects up-to-date issues from Python repositories to mitigate the influence of pre-training bias in the latest LLMs. Additionally, LOC-BENCH covers wider categories, including bug reports, feature requests, security, and performance issues, enabling a more comprehensive evaluation of code localization methods. The statistics of LOC-BENCH are shown in Table 3.

For the Bug Report category, we collect GitHub issues created after October 2024, which is later than the release dates of most modern LLMs. To enrich the dataset with more instances of security and performance issues, we use the GitHub Search API to search for relevant keywords, such as "latency improvement" for performance-related issues. We exclude instances that involve modifying more than five Python files or more than ten functions in the corresponding patch. For further details, see Appendix B.1.

### 5 Experiments

Our experiments aim to evaluate four key aspects of LOCAGENT: (1) the effectiveness of our graphbased representation and tooling for code localization compared to existing methods, (2) the performance of fine-tuned open-source models as costeffective alternatives to proprietary LLMs, (3) a detailed analysis of how performance varies across

Dataset Category #Sample

Bug Report 254 Feature Request 43

SWE-Bench-Lite

(Total = 300) Security Issue 3 Performance Issue 0

Bug Report 242 Feature Request 150

Loc-Bench

(Totoal = 560) Security Issue 29 Performance Issue 139

Table 3: Distribution of samples across different categories in the SWE-Bench-Lite and Loc-Bench datasets.

task categories, and (4) the contribution of each component in our framework through comprehensive ablation studies. We evaluate on both SWEBench-Lite and our introduced Loc-Bench dataset. Additionally, we examine the impact of improved localization on downstream software maintenance tasks.

#### 5.1 Experimental Settings

Datasets. We first conduct experiments on SWEBench-Lite, treating the patched files and functions as the targets for localization. Following Suresh et al. (2024), we excluded examples where no existing functions were modified by the patch, ultimately retaining 274 out of the original 300 examples.

Metrics. To assess performance, we use a modified accuracy metric inspired by R-Precision from information retrieval, following Agentless(Xia et al., 2024). To assess performance, we use Acc@k (Accuracy at k) as our evaluation metric, following Agentless(Xia et al., 2024). For each example, we select the top-k predicted locations and consider a localization attempt successful only if all relevant locations are correctly identified within these top-k predictions. This approach measures the ability to fully identify all necessary code sections that require modification. We report results across multiple k values: file localization at Acc@1, Acc@3, and Acc@5, and function localization at Acc@5 and Acc@10. Additionally, to provide a more relaxed evaluation criteria, we assess module localization, which only requires finding any function within the patched class.

#### 5.2 Baselines

We evaluate LOCAGENT against three categories of competitive baselines: (a) Retrieval-based meth-

File (%) Module (%) Function (%) Acc@1 Acc@3 Acc@5 Acc@5 Acc@10 Acc@5 Acc@10

Type Method Loc-Model

BM25 (Robertson et al., 1994) 38.69 51.82 61.68 45.26 52.92 31.75 36.86 E5-base-v2 (Wang et al., 2022) 49.64 74.45 80.29 67.88 72.26 39.42 51.09 Jina-Code-v2 (Günther et al., 2023) 43.43 71.17 80.29 63.50 72.63 42.34 52.19 Codesage-large-v2 (Zhang et al., 2024) 47.81 69.34 78.10 60.58 69.71 33.94 44.53 CodeRankEmbed (Suresh et al., 2024) 52.55 77.74 84.67 71.90 78.83 51.82 58.76

Embedding-Based

Agentless (Xia et al., 2024)

GPT-4o 67.15 74.45 74.45 67.15 67.15 55.47 55.47 Claude-3.5 72.63 79.20 79.56 68.98 68.98 58.76 58.76

Procedure-Based

MoatlessTools (Örwall, 2024)

GPT-4o 73.36 84.31 85.04 74.82 76.28 57.30 59.49 Claude-3.5 72.63 85.77 86.13 76.28 76.28 64.60 64.96

SWE-agent (Yang et al., 2024)

GPT-4o 57.30 64.96 68.98 58.03 58.03 45.99 46.35 Claude-3.5 77.37 87.23 90.15 77.74 78.10 64.23 64.60

Agent-Based

Openhands (Wang et al., 2025)

GPT-4o 60.95 71.90 73.72 62.41 63.87 49.64 50.36

- Claude-3.5 76.28 89.78 90.15 83.21 83.58 68.25 70.07

LOCAGENT (Ours)

Qwen2.5-7B(ft) 70.80 84.67 88.32 81.02 82.85 64.23 71.53 Qwen2.5-32B(ft) 75.91 90.51 92.70 85.77 87.23 71.90 77.01

- Claude-3.5 77.74 91.97 94.16 86.50 87.59 73.36 77.37

- Table 4: Performance comparison with baseline methods on code localization on SWE-bench lite. Results show the accuracy at file, module, and function levels. For Agent-Based methods, we use GPT-4o-2024-0513 (abbr.

- as GPT-4o) and Claude-3-5-sonnet-20241022 (abbr. as Claude-3.5) as the localization model. Additionally, the performance of our fine-tuned open-source models, Qwen2.5-7B(ft) and Qwen2.5-32B(ft), are included for comparison.

File level - Acc@5

ods: We include the sparse retrieval approach BM25 (Robertson et al., 1994) and several state-ofthe-art embedding models, including the generalpurpose E5-base-v2 (Wang et al., 2022) and specialized code embedding models such as JinaCode-v2 (Günther et al., 2023), Codesage-largev2 (Zhang et al., 2024), and the current SOTA code embedding model CodeRankEmbed (Suresh et al., 2024). Proprietary embedding solutions were excluded due to API costs. (b) Procedurebased methods: We compare against Agentless (Xia et al., 2024), which employs a structured hierarchical approach to code localization without complex agent architectures. (c) Agent-based methods: We include several advanced agent frameworks designed for code exploration and modification, specifically OpenHands (Wang et al., 2025) (using its default CodeActAgent implementation), SWEAgent (Yang et al., 2024), and MoatlessTools (Örwall, 2024). For implementation details, please refer to Appendix C.1.1.

1.0

80

0.9

60

SampleCount

Accuracy

0.8

40

0.7

Methods

E5-Base-v2

Openhands

0.6

20

CodeRankEmbed

MoatlessTools

Agentless

Ours

0.5

SWE-agent

0

Hop 0 Hop 1 Hop 2 Hop 3+

Function level - Acc@10

80

0.9

0.8

60

SampleCount

Accuracy

0.7

40

0.6

0.5

20

0.4

0

0.3

Hop 0 Hop 1 Hop 2 Hop 3+

Figure 3: Performance analysis at different difficulty levels for file- and function-level localization. All agentbased methods and Agentless use Claude-3.5 as the localization model. Hop N refers to the distances between functions mentioned in the issue description and the ground truth patch on our code graph.

#### 5.3 Evaluation Results on SWE-Bench-Lite

As shown in Table 4, Agent-Based methods consistently outperform other approaches, and our method demonstrates competitive performance by achieving the best results across all levels of code localization. Unlike traditional retrieval-based methods, Agentless identifies only a limited number of locations due to its narrow repository scope, which hinders performance gains when considering a broader set of candidates. The results of the NDCG are presented in Table 11 in the Appendix.

To further analyze the results, we examine performance across different task difficulty levels. We measure the task difficulty by calculating the shortest hops between the functions mentioned in the issue descriptions and the patched functions on our code graph (See Appendix C.1.2 for more details). As shown in Figure 3, performance decreases for all methods as the task becomes more challenging. However, Agent-based methods demonstrate better robustness as the difficulty increases, with

Fine-tune Qwen 2.5

94.2 87.6

80

77.4

60

Acc(%)

40

file-level module-level func-level Original FT Improv

| |
|---|

| |
|---|

20

| |
|---|

| |
|---|

Claude-3.5

0

Qwen2.5-Coder-7B Qwen2.5-Coder-32B

Figure 4: Comparison of performance between the original and fine-tuned Qwen models. The metrics used are file-level Acc@5 and module/function-level Acc@10. Dashed lines represent the performance of the Claude-3.5 model for reference.

our method maintaining competitive performance across various difficulty levels. Retrieval-based methods, such as E5-Base-v2 and CodeRankEmbed, perform poorly at the function level, even when the patched functions are explicitly mentioned in the query. This is because they treat the query as a whole, failing to capture fine-grained details. Agentless performs even worse than retrievalbased methods when exploration beyond the query is needed (hop ≥ 0) due to its simplistic localization process and limited view focused only on the repository structure.

#### 5.4 Fine-tuned Open-source Models

Figure 4 demonstrates that after fine-tuning, both the 7B and 32B models show significant improvements on this task. LOCAGENT with finetuned Qwen-2.5-Coder-Instruct-32B (abbreviated as Qwen2.5-32B(ft)) achieves performance comparable to Claude-3.5, and LOCAGENT with Qwen2.5-7B(ft) also delivers results on par with that obtained using GPT-4o. As shown in Table 4, our method with Qwen2.5-32B(ft) outperforms nearly all baselines, including those that use larger and more powerful LLMs. The original 7B model performs poorly due to its limited tool-use capability (Chen et al., 2024). These results validate the feasibility of deploying our fine-tuned opensource models as promising alternatives to proprietary APIs, especially in resource-constrained applications.

#### 5.5 Efficiency Analysis

- Table 5 presents an efficiency analysis comparing agent-based methods in terms of cost and the number of agent interactions required. MoatlessTools demonstrates good cost-efficiency and requires relatively fewer rounds of interaction. However, the

Acc@10 Cost MoatlessTools

Method LM #Round Cost($)

GPT-4o 5 0.46 1.3 Claude-3.5 5 0.46 1.4

GPT-4o 8 0.56 0.8 Claude-3.5 9 0.67 1.0

SWE-agent

GPT-4o 15 0.83 0.6 Claude-3.5 13 0.79 0.9

Openhands

Claude-3.5 7 0.66 1.2 Qwen2.5-7B(ft) 6 0.05 13.2 Qwen2.5-32B(ft) 9 0.09 8.6

Ours

- Table 5: Efficiency analysis comparing the average cost and number of agent interaction rounds required by different methods. The cost-efficiency of each method is evaluated using the ratio of function-level Acc@10 to average cost.

Model Setting

File Acc@5

Module Acc@10

Function Acc@10

Ours 88.32 82.85 71.53 w/o TraverseGraph 86.13 78.47 66.06 Relation Types: contain 86.50 79.56 66.42 Traverse Hops: 1 86.86 80.29 66.79 w/o RetrieveEntity 87.59 81.39 69.34 w/o SearchEntity 68.98 61.31 53.28 w/o BM25 index 75.18 68.98 60.22

- Table 6: The ablation study of our model. The metrics used here are file-level Acc@5, module-level Acc@10, and function-level Acc@10. The impact of removing or fixing components is analyzed to observe how each component contributes to the overall accuracy.

dense embeddings it uses make it difficult and slow to adapt to fast-evolving codebases. SWEagent and Openhands also show moderate costs but still do not match the efficiency of LOCAGENT. For LOCAGENT with Claude-3.5, although more rounds of interaction are required, the cost remains lower than that of Openhands, illustrating the token efficiency of our tool’s outputs. LOCAGENT with fine-tuned Qwen models stands out for its superior efficiency1. Qwen2.5-7B(ft) is the most cost-efficient option, requiring only $0.05 per example, while Qwen2.5-32B(ft) offers a more costeffective alternative to Claude-3.5. These results highlight the potential of fine-tuned open-source models as efficient alternatives, providing an optimal balance of cost-effectiveness and performance that surpasses other methods.

1We calculate the cost based on the prices from AI inference providers (Hyperbolic, 2025; artificialanalysis.ai, 2025). Specifically, for the Qwen2.5-32B(ft) model, the cost is $0.20/1M tokens for both input and output. For the Qwen2.5-7B(ft) model, the cost is $0.14/1M tokens for input and $0.28/1M tokens for output.

#### 5.6 Ablation Study

We conduct an ablation study to evaluate the effectiveness of each component of our toolsets. Due to budget constraints, we use the fine-tuned Qwen-2.5-7B as the localization model for these experiments.

- (1) Each tool in our toolset plays a critical

role in code localization performance. As shown in Table 6, removing any tool, especially the SearchEntity tool, leads to varying degrees of accuracy degradation, particularly in module and function level localization. This highlights the critical role each tool plays in identifying relevant modules and functions.

- (2) The graph structure provides essential infor-

mation for accurate code localization. Removing TraverseGraph tool decreases module and function level performance since the agent cannot obtain any structure information about the codebase and relies on reasoning capability to identify call relationship or directory structure. Adding contain relationship provides only marginal improvements compared to fully removing TraverseGraph, emphasizing the importance of the other three relationship types and explaining why our method surpasses others relying only on the repository structure.

- (3) Multi-hop exploration is crucial for deep

code understanding. When compared to the full setting, fixing Hops=1 leads to a moderate decline in file and module-level accuracy, but it causes a more significant decrease in function-level accuracy, underscoring the importance of multi-hop exploration for identifying relevant entities.

- (4) Sparse indexing significantly enhances lo-

calization performance. Removing SearchEntity tool, or even partial removal of its index, causes a substantial drop in performance across all metrics. This demonstrates the effectiveness of building a sparse index on our code graph for improving localization performance.

#### 5.7 Evaluation Results on Loc-Bench

To ensure the robustness and generalization of our methods and fine-tuned Qwen models, and to eliminate potential data leakage, we evaluate our new dataset. Since Loc-Bench includes examples that edit 1 to 5 files, we assess file localization at top-5 and top-10 ranks, and function/module localization

- at top-10 and top-15 ranks. Table 7 shows that our fine-tuned Qwen2.5-7B model exhibits strong gen-

File level - Acc@5

1.0

0.8

Accuracy

0.6

0.4

Bug Report Feature Request Performance Security

Function level - Acc@15

Methods

0.8

CodeRankEmbed

Openhands

Agentless

Ours(Qwen-7B)

SWE-agent

Ours(Claude-3.5)

0.6

Accuracy

0.4

0.2

Bug Report Feature Request Performance Security

Figure 5: Performance analysis at different difficulty category for file- and function-level localization. All agent-based baselines and Agentless use Claude-3.5 as the localization model.

eralization capabilities, maintaining competitive performance compared to SWE-agent using more expensive and strong model. These results highlight the practicality of the fine-tuned Qwen2.5-7B model for real-world applications. Despite being an open-source alternative, it achieves a performance comparable to Claude-3.5, supporting its feasibility as a cost-effective substitute for commercial models in practical scenarios.

Additionally, we evaluate the performance across four different difficulty categories. Figure 5 clearly shows that our method outperforms other methods in almost all categories of code localization. However, it also highlights a noticeable decrease in performance across the other three categories compared to the Bug Report category. This performance gap likely reflects our training data distribution, which contained more bug report examples, potentially leading to scaffolds better optimized for bug localization tasks. This trend suggests that while our method is highly effective for bug report localization, there is still room for improvement in handling the other categories through more balanced training data and category-specific optimization strategies.

5.8 Application: Better Localization Leads to More Solved GitHub Issues

To assess the impact of localization methods on downstream tasks, we evaluated their effectiveness in solving GitHub issues. We choose Agentless as the baseline, ranking among the top-performing

###### File (%) Module (%) Function (%)

Method Loc Model

Acc@5 Acc@10 Acc@10 Acc@15 Acc@10 Acc@15

IR-Based CodeRankEmbed 74.29 80.89 63.21 67.50 43.39 46.61 Agentless Claude-3.5 67.50 67.50 53.39 53.39 42.68 42.68 OpenHands Claude-3.5 79.82 80.00 68.93 69.11 59.11 59.29 SWE-agent Claude-3.5 77.68 77.68 63.57 63.75 51.96 51.96

Qwen2.5-7B(ft) 78.57 79.64 63.04 63.04 51.43 51.79 Claude-3.5 83.39 86.07 70.89 71.07 59.29 60.71

LocAgent (Ours)

Table 7: Performance evaluation on the real-world LocBench dataset.

Method Localization LM Acc@5 Pass@1 Pass@10 Agentless Claude-3.5 58.39 26.31 33.58 Ours

Qwen2.5-32B(ft) 69.34 26.79 36.13 Claude-3.5 73.36 27.92 37.59

Table 8: Impact of localization accuracy on downstream bug repair tasks.

open-source submissions on SWE-Bench-Lite. For consistency, we utilized Claude-3.5 as the editing model in conjunction with the Agentless editing method. Table 8 shows that the success rate for solving GitHub issues improves significantly with better code localization accuracy.

### 6 Conclusion

In conclusion, LOCAGENT enhances code localization by structuring codebases as graphs, enabling efficient repository-level exploration for LLM agents. With fine-tuned open-source models, our method achieves high localization accuracy while significantly reducing costs compared to larger proprietary models. Experimental results demonstrate the effectiveness of LOCAGENT in identifying relevant code components and improving downstream tasks.

### Limitations

First, our study primarily focused on fine-tuning Qwen-2.5-Coder models. Exploring a broader range of base models, including other open-source LLMs like CodeLlama, Mistral, or Yi, could provide valuable insights into model selection tradeoffs. Additionally, investigating different finetuning approaches beyond LoRA, such as full finetuning or other parameter-efficient methods, could potentially yield better performance.

Second, though we demonstrated improved bug repair performance with better localization, we only scratched the surface of potential downstream

applications. Future work should evaluate LocAgent’s impact on other software engineering tasks like refactoring, feature addition, security vulnerability patching, and performance optimization. This would provide a more comprehensive understanding of the framework’s practical utility.

Moreover, our fine-tuning process relied heavily on trajectories generated by Claude-3.5 and the fine-tuned Qwen2.5-32B model. A more diverse training dataset incorporating examples from different models, tasks, and repositories could improve the robustness and generalization of finetuned models. Additionally, analyzing the impact of different dataset compositions and filtering strategies on model performance could yield valuable insights.

Finally, the current evaluation focuses primarily on Python codebases. Extending LOCAGENT to support other programming languages and evaluating its performance across different language paradigms would better demonstrate its generalizability. Further, our evaluation metrics could be expanded to include more nuanced measures of localization quality beyond accuracy and NDCG.

### References

Aider. 2023. Building a better repository map with tree sitter. Accessed: April 15, 2025.

Anthropic. 2023. Claude: Conversational ai by anthropic. Accessed: January 21, 2025.

artificialanalysis.ai. 2025. Artificial analysis. https: //artificialanalysis.ai/models/. Accessed: 2025-04-28.

Marcel Böhme, Ezekiel O Soremekun, Sudipta Chattopadhyay, Emamurho Ugherughe, and Andreas Zeller. 2017. Where is the bug and how is it fixed? an experiment with practitioners. In Proceedings of the 2017 11th joint meeting on foundations of software engineering, pages 117–128.

Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, et al. 2024. T-eval: Evaluating the tool utilization capability of large language models step by step. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9510–9529.

Cognition.ai. 2024. Introducing devin, the first ai software engineer.

John Ellson, Emden Gansner, Lefteris Koutsofios, Stephen C North, and Gordon Woodhull. 2002. Graphviz—open source graph drawing tools. In Graph Drawing: 9th International Symposium, GD 2001 Vienna, Austria, September 23–26, 2001 Revised Papers 9, pages 483–484. Springer.

Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. 2023. Talk like a graph: Encoding graphs for large language models. arXiv preprint arXiv:2310.04560.

Paul Gauthier. 2024. How aider scored sota 26.3% on swe bench lite | aider. Accessed: January 21, 2025.

Jiafeng Guo, Yixing Fan, Qingyao Ai, and W Bruce Croft. 2016. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM international on conference on information and knowledge management, pages 55–64.

Jiafeng Guo, Yixing Fan, Liang Pang, Liu Yang, Qingyao Ai, Hamed Zamani, Chen Wu, W Bruce Croft, and Xueqi Cheng. 2020. A deep look into neural ranking models for information retrieval. Information Processing & Management, 57(6):102067.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR.

Michael Günther, Louis Milliken, Jonathan Geuter, Georgios Mastrapas, Bo Wang, and Han Xiao. 2023. Jina embeddings: A novel set of highperformance sentence embedding models. Preprint, arXiv:2307.11224.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. 2024. Qwen2.5-coder technical report. Preprint, arXiv:2409.12186.

Hyperbolic. 2025. Hyperbolic website. https:// hyperbolic.xyz/. Accessed: 2025-04-15.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770.

- Sungmin Kang, Gabin An, and Shin Yoo. 2023. A preliminary evaluation of llm-based fault localization. arXiv preprint arXiv:2308.05487.
- Sungmin Kang, Gabin An, and Shin Yoo. 2024. A quantitative and qualitative evaluation of llm-based explainable fault localization. Proceedings of the ACM on Software Engineering, 1(FSE):1424–1446.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474.

Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Shieh, and Wenmeng Zhou. 2024. Codexgraph: Bridging large language models and code repositories via code graph databases. Preprint, arXiv:2408.03910.

Qingsong Lv, Ming Ding, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, and Jie Tang. 2021. Are we really making much progress? revisiting, benchmarking and refining heterogeneous graph neural networks. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pages 1150–1160.

Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. 2024. How to understand whole software repository? arXiv e-prints, pages arXiv–2406.

Niels Mündler, Mark Müller, Jingxuan He, and Martin Vechev. 2024. Swt-bench: Testing and validating real-world bug-fixes with code agents. Advances in Neural Information Processing Systems, 37:81857– 81887.

OpenAI. 2023. Chatgpt: Language model by openai. Accessed: January 21, 2025.

Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, Jiawei Han, Hongming Zhang, and Dong Yu. 2025. Repograph: Enhancing AI software engineering with repository-level code graph. In The Thirteenth International Conference on Learning Representations.

PerplexityAI. 2023. Perplexity ai: An ai-powered search engine. Accessed: January 21, 2025.

Yihao Qin, Shangwen Wang, Yiling Lou, Jinhao Dong, Kaixin Wang, Xiaoling Li, and Xiaoguang Mao. 2024. Agentfl: Scaling llm-based fault localization to project-level context. arXiv preprint arXiv:2403.16362.

Chen Qu, Liu Yang, Cen Chen, Minghui Qiu, W Bruce Croft, and Mohit Iyyer. 2020. Open-retrieval conversational question answering. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 539–548.

Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at trec-3. In Text Retrieval Conference.

Tarun Suresh, Revanth Gangi Reddy, Yifei Xu, Zach Nussbaum, Andriy Mulyar, Brandon Duderstadt, and Heng Ji. 2024. Cornstack: High-quality contrastive data for better code ranking. arXiv preprint arXiv:2412.01007.

David A. Tomassi, Naji Dmeiri, Yichen Wang, Antara Bhowmick, Yen-Chuan Liu, Premkumar Devanbu, Bogdan Vasilescu, and Cindy Rubio-González. 2019. Bugswarm: Mining and continuously growing a dataset of reproducible failures and fixes. Preprint, arXiv:1903.06725.

VoyageAI. 2024. Voyage-code-2: Elevate your code retrieval. Accessed: 2024-02-02.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. 2025. Openhands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023a. Self-consistency improves chain of thought reasoning in language models. Preprint, arXiv:2203.11171.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D. Q. Bui, Junnan Li, and Steven C. H. Hoi. 2023b. Codet5+: Open code large language models for code understanding and generation. Preprint, arXiv:2305.07922.

Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F. Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. 2024. Coderag-bench: Can retrieval augment code generation? Preprint, arXiv:2406.14497.

Yonghao Wu, Zheng Li, Jie M Zhang, Mike Papadakis, Mark Harman, and Yong Liu. 2023. Large language models in fault localisation. arXiv preprint arXiv:2308.15276.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2024. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793.

Zhongming Yu, Hejia Zhang, Yujie Zhao, Hanxian Huang, Matrix Yao, Ke Ding, and Jishen Zhao. 2025. Orcaloca: An llm agent framework for software issue localization. arXiv preprint arXiv:2502.00350.

Dejiao Zhang, Wasi Uddin Ahmad, Ming Tan, Hantian Ding, Ramesh Nallapati, Dan Roth, Xiaofei Ma, and Bing Xiang. 2024. CODE REPRESENTATION LEARNING AT SCALE. In The Twelfth International Conference on Learning Representations.

Albert Örwall. 2024. Moatless tools.

### A LOCAGENT Design Details

- A.1 Tool Output Design

- A.1.1 Three-level format for SearchEntity output

Once invoked by the LLM agent, the retrieval APIs search for files, classes, methods, and code snippets in the codebase, and return the results back to the agent. To avoid forming very lengthy code context that may containing noisy information to LLM, we return only necessary information as API outputs. To achieve this, we desgined four granular standard output formats (Figure 6): fold, preview, full code.

- A.1.2 Tree-based Subgraph Formatting for TraverseGraph Output

The TraverseGraph tool traverses the code graph and returns a local subgraph for each input entity. The agent reasons about these subgraphs to understand each entity’s complex dependencies. However, reasoning about graphs remains challenging for LLMs. Research by (Fatemi et al., 2023) demonstrates that LLM performance varies significantly based on graph formatting (how graphs are encoded as text). This makes the format design for output subgraphs crucial.

We have developed a new tree-based format, shown in Figure 7, with several features that enhance LLM reasoning: (1) We represent subgraphs as trees, allowing LLMs to use indentation to determine a node’s distance from the root, (2) We display complete entity IDs for each node (e.g., django/core/validators.py:RegexValidator)

to help LLMs locate nodes easily, and (3) We explicitly specify relation types for each edge, including reversed relations

To evaluate how different graph formats impact code localization performance, we conducted an experiment using 37 challenging samples from SWEBench-Lite. These samples were considered "challenging" because they could not be solved by any baseline agent methods. Using Claude-3.5 as the Localization Model across all settings, we compared various output formats. Table 9 presents our findings. The baseline output formats we tested are described below:

• row: For each line, list one row of the adjacency matrix. For example,

function "fileA.py:funcA" invokes function "fileA.py:funcB", "fileA.py:funcC"

- • row (w/ entity attributes): Additionally include entity attributes for format row.
- • incident: The incident format mentioned in (Fatemi et al., 2023). An integer instead of entity ID is used to represent each node. For example,

Map function "fileA.py:funcA" to index 0. Map function "fileA.py:funcB" to index 1. Map function "fileA.py:funcC" to index 2.

function 0 invokes function 1,2.

- • Graphviz DOT: Represent graph in Graphviz DOT language (Ellson et al., 2002).
- • JSON: Expand the subgraph as a tree, and convert it to JSON format.

As shown in Table 9, expanding subgraphs as trees (i.e., JSON, tree-based) can significantly improve the performance. Our tree-based format achieves the best overall performance across different levels of localization tasks. We also test returning entity attributes along with subgraphs. We notice that row (w/ entity attributes) consistently underperforms row, indicating the attributes for all the nodes may be very noisy. Besides, although using incident format can simplify the output and show improvements in file-level localization, it degradation the module- and file-level localization.

#### A.2 Implementation

To enable the LLM agent to invoke the Code Localization APIs, we handle the interaction differently based on the LLM’s capabilities. For LLMs that support tool-calling features, we define the tools as a list of JSON objects, which are then used as parameters for the API calls. For LLMs that do not support tool-calling (such as Qwen), we provide the description of the API and the expected output as part of the LLM’s prompt. When the agent decides to invoke a set of retrieval APIs, it responds with a list of API call names and their corresponding arguments. These retrieval API requests are processed locally by searching over the built code graph. The results from executing these APIs locally are returned to the agent.

By default, we query the LLM with a temperature setting of 1.0. We conduct two interactions, after which we rerank the results based on mean reciprocal rank (MRR) scores. We also leverage multiprocess execution to speed up the process. Since all our tools are read-only, LOCAGENT does

[Figure 75]

[Figure 76]

[Figure 77]

###### Full Code Output format

###### Preview Output format

###### Fold Output format

Show Full Code Content if the size of matched entities not exceed 3.

Show the skeleton of modules if the content is too long (exceed 300 lines)

Show only node ids if the size of matched entities exceed 3.

[Figure 78]

[Figure 79]

[Figure 80]

Tool Call

Tool Call

Tool Call

search_code_snippets(

search_code_snippets(

search_code_snippets(

search_terms=[" Piecewise"] )

search_terms=["__repr__"] )

search_terms=["ResolverMatch"] )

OBSERVATION: ##Searching for term "ResolverMatch"... ### Search Result: Found class `django/urls/resolvers.py:ResolverMatch`. Source: Match found for entity name `ResolverMatch`.

OBSERVATION: ##Searching for term "Piecewise"... ### Search Result: Found class `sympy/functions/elementary/piecewise.py: Piecewise`. Source: Match found for entity name `Piecewise`. Just show the structure of this class due to response length limitations:

OBSERVATION: ##Searching for term "__repr__"... ### Search Result: Found function `django/urls/resolvers.py:ResolverMatch.__ repr__`. Found function `django/urls/resolvers.py:URLPattern.__rep r__`. Found function `django/urls/resolvers.py:URLResolver.__re pr__`. Found function `django/db/models/expressions.py:F.__repr_ _`. Source: Match found for entity name `__repr__`.

- 32 | class ResolverMatch:
- 33 | def __init__(self, func, args, kwargs, url_name=None, app_names=None, namespaces=None, route=None, tried=None):
- 34 | self.func = func
- 35 | self.args = args
- 36 | self.kwargs = kwargs
- 37 | self.url_name = url_name
- 38 | self.route = route
- 39 | self.tried = tried

class Piecewise(Function): nargs = None is_Piecewise = True def __new__(cls, *args, **options):

...

@classmethod def eval(cls, *_args):

... Hint: Use more detailed query to get the full content of some if needed.

...

... def _eval_simplify(self, **kwargs):

...

- Figure 6: Different output formats designed for efficient agent-code interaction. Left: Full code output when matched entities ≤ 3. Middle: Preview output showing module skeleton for large files. Right: Fold output showing only entity IDs when matches > 3.

OBSERVATION: django/contrib/auth/validators.py ├── contains ── django/contrib/auth/validators.py:ASCIIUsernameValidator │ └── inherits ── django/core/validators.py:RegexValidator ├── imports ── django/core/validators.py │ ├── contains ── django/core/validators.py:_lazy_re_compile │ ├── contains ── django/core/validators.py:RegexValidator │ ├── imports ── django/core/exceptions.py:ValidationError │ └── imports ── django/utils/deconstruct.py:deconstructible ├── imports ── django/utils/deconstruct.py:deconstructible │ ├── contains ── django/utils/deconstruct.py:deconstructible.decorator │ └── invokes ── django/utils/deconstruct.py:deconstructible.decorator ├── contains-by ── django/contrib/auth │ └── contains-by ── django/contrib └──imports-by ── django/contrib/auth/migrations/0008_alter_user_username_max_length.py

└── contains-by ── django/contrib/auth/migrations

traverse_graph( start_entities=[“django/contrib/auth/validators.py”], direction=“both”, traverse_hops=2,

)

Tool Call

[Figure 81]

Expanded Tree-Based Format

[Figure 82]

Relations Reversed Relations

- Figure 7: A truncated example of the expanded tree-based format for the output subgraph of tool TraverseGraph.

not require a specialized Docker environment to operate.

the patches, are considered the target locations for the given issue. While it is possible to fix a bug in a location different from the ground truth, the extracted ground-truth locations still serve as approximate targets for localization. Additionally, edited code such as documents, import statements, and comments are excluded from the localization target. These elements are not considered relevant for bug localization, as they do not directly impact the functionality of the code or its execution. By filtering out these elements, the focus is maintained on the core code changes that are relevant for localization.

### B Dataset construction and statistics

#### B.1 Dataset construction details

Example collection. We collected examples on popular Python repositories on Github follow (Jimenez et al., 2023). To gather issues related to performance and security, we searched for the keywords listed in Table 10 using the GitHub Search APIs. We then used GPT-4o-2024-0513 as the classifier based on the issue descriptions.

Ground Truth Locations. The affected files or functions in the original codebase, as identified in

File(%) Module(%) Function(%)

Output Format

Acc@1 Acc@3 Acc@5 Acc@5 Acc@10 Acc@5 Acc@10

row 41.18 67.65 70.59 61.76 61.76 35.29 38.24 row (w/ entity attributes) 41.18 64.71 64.71 50.00 50.00 32.35 32.35 incident 41.18 70.59 73.53 55.88 55.88 29.41 32.35 Graphviz DOT 41.18 73.53 82.35 64.71 64.71 35.29 35.29 JSON 41.18 67.65 76.47 67.65 70.59 38.24 41.18 tree-based (Ours) 47.06 79.41 79.41 64.71 64.71 38.24 41.18

Table 9: Localization performance under different TraverseGraph output formats.

###### Category Keywords

Performance bottleneck, performance improvement, memory usage optimization, time complexity reduction, latency improvement, scalability improvement, CPU usage reduction, caching improvement, concurrency optimization

Security Out-of-bounds Write, Out-of-bounds Read, NULL Pointer Dereference, Missing Authorization, memory leak fix, security vulnerability, security issue, authentication bypass, authentication issue, better maintained, buffer overflow, denial of service, security hardening, security patch, unsafe deserialization, Use After Free, Integer Overflow or Wraparound, Uncontrolled Resource Consumption, Missing Authentication for Critical Function

Table 10: We use these Keywords to search for Performance and Security related issues with Github Search APIs.

### C Additional Experiments

#### C.1 Implementation Details

- C.1.1 Baselines Implementation Regarding the embedding-based methods in our evaluation, these approaches operate primarily at the function level, where each function is embedded as a separate unit. The function’s context (its containing file and class) is appended to the function representation before embedding, rather than being embedded separately. While theoretically these methods could employ hierarchical indexing, the standard implementations we evaluated use flat indexing structures where each function is embedded as a single unit.

We use OpenHands’s remote runtime feature to parallelize evaluation on OpenHands and SWEagent. We use Openhands version 0.12.0 released on Oct 31, 2024.

- C.1.2 Quantifying Task Difficulty Based on Code Graph Distance

We measure task difficulty by computing the average shortest hop distance between the functions mentioned in the issue descriptions and the patched functions within our code graph. Specifically, we first extract potential function names from each

issue description using GPT-4o-2024-0513, and identify their corresponding nodes in the code graph using the global dictionary. These identified nodes form the set of predicted nodes, denoted as C. Similarly, we link the ground truth functions from the patch to their corresponding nodes in the code graph, forming the set of target nodes, denoted as T . To quantify the difficulty δ, we calculate the average shortest hop distance between the predicted nodes C and the target nodes T , defined as:

##### 1 ∣C∣

1 mint∈T d(c,t) + 1

∑

δ =

c∈C

where d(c,t) represents the shortest hop distance between nodes c and t in the graph. For performance analysis stratified by difficulty, we round δ down to ⌊δ⌋ to group samples by difficulty levels, and we exclude samples where the LLM fails to extract any valid function names.

#### C.1.3 Training details.

Fine-tuning Settings. We use Qwen-2.5-Coder-Instruct (Hui et al., 2024) 7B and 32B variants as our base models. We fine-tuned Qwen-2.5-Coder-Instruct 7B and 32B models on 768 training samples from the SWE-Bench training dataset, leveraging LoRA

File (%) Module (%) Function (%) NDCG@1 NDCG@3 NDCG@5 NDCG@5 NDCG@10 NDCG@5 NDCG@10

Type Method Loc-Model

BM25 (Robertson et al., 2009) 38.69 46.5 50.61 37.31 39.86 26.15 27.92

E5-base-v2 (Wang et al., 2022) 49.64 64.19 66.6 53.15 54.45 31.39 35.3

Embedding-Based

Jina-Code-v2 (Günther et al., 2023) 43.43 59.93 63.7 51.02 54.13 33.28 36.44 Codesage-large-v2 (Zhang et al., 2024) 47.81 60.82 64.39 49.38 52.22 27.03 30.74 CodeRankEmbed (Suresh et al., 2024) 52.55 67.54 70.39 57.51 59.76 40.28 42.55

Agentless (Xia et al., 2024)

GPT-4o 67.15 71.76 71.76 64.31 64.31 53.81 53.81 Claude-3.5 72.63 76.72 76.87 67.36 67.36 57.55 57.55

Procedure-Based

MoatlessTools (Örwall, 2024)

GPT-4o 73.36 80.03 80.33 68.57 69.09 49.77 50.62 Claude-3.5 72.63 80.73 80.88 69.11 69.11 53.03 53.16

SWE-agent (Yang et al., 2024)

GPT-4o 57.3 63.96 64.12 53.95 53.95 42.32 42.44 Claude-3.5 77.37 84.32 84.93 72.77 72.9 59.67 59.79

Agent-Based

Openhands (Wang et al., 2025)

GPT-4o 60.95 67.62 68.39 58.18 58.6 44.34 44.66

- Claude-3.5 76.28 84.27 84.43 75.79 75.92 63.13 63.8

LocAgent (Ours)

Qwen2.5-7B(ft) 70.80 79.36 80.9 70.99 71.68 55.62 58.09 Qwen2.5-32B(ft) 75.91 84.74 85.64 76.28 76.77 64.27 65.93

- Claude-3.5 77.74 86.19 87.14 77.73 78.1 64.34 65.57

Table 11: NDCG scores comparison showing ranking quality of different methods.

for efficient adaptation. The training set included 447 samples generated by Claude-3.5, while the remaining samples were iteratively generated using the fine-tuned Qwen2.5-32B model. The fine-tuning process was conducted over 5 epochs with max_token set to 128k and a learning rate of 2 × 10−4.

### D Prompt

In this section, we go through the prompt template that make up the agent’s history.

###### Prompt

Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.

Follow these steps to localize the issue:

- ## Step 1: Categorize and Extract Key Problem Information

- - Classify the problem statement into the following categories: Problem description, error trace, code to reproduce the bug, and additional context.
- - Identify modules in the '{{package_name}}' package mentioned in each category.
- - Use extracted keywords and line numbers to search for relevant code references for additional context.

- ## Step 2: Locate Referenced Modules

- - Accurately determine specific modules
- - Explore the repo to familiarize yourself with its structure.
- - Analyze the described execution flow to identify specific modules or components being referenced.
- - Pay special attention to distinguishing between modules with similar names using context and described execution flow.
- - Output Format for collected relevant modules:
- - Use the format: 'file_path:QualifiedName’
- - E.g., for a function `calculate_sum` in the `MathUtils` class located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.

- ## Step 3: Analyze and Reproducing the Problem

- - Clarify the Purpose of the Issue
- - If expanding capabilities: Identify where and how to incorporate new behavior, fields, or modules.
- - If addressing unexpected behavior: Focus on localizing modules containing potential bugs.
- - Reconstruct the execution flow
- - Identify main entry points triggering the issue.
- - Trace function calls, class interactions, and sequences of events.
- - Identify potential breakpoints causing the issue. Important: Keep the reconstructed flow focused on the problem, avoiding irrelevant details.

- ## Step 4: Locate Areas for Modification

- - Locate specific files, functions, or lines of code requiring changes or containing critical information for resolving the issue.
- - Consider upstream and downstream dependencies that may affect or be affected by the issue.
- - If applicable, identify where to introduce new fields, functions, or variables.
- - Think Thoroughly: List multiple potential solutions and consider edge cases that could impact the resolution.

## Output Format for Final Results: Your final output should list the locations requiring modification, wrapped with triple backticks ``` Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance. Your answer would better include about 5 files.

### Examples: ```

- full_path1/file1.py line: 10 class: MyClass1 function: my_function1
- full_path2/file2.py line: 76 function: MyClass2.my_function2
- full_path3/file3.py line: 24 line: 156 function: my_function3 ```

Return just the location(s) Note: Your thinking should be thorough and so it's fine if it's very long.

Figure 8: The task instruction prompt for LOCAGENT.

