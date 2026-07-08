# arXiv:2605.00180v2[cs.NI]26May2026

## RouteProfile: Graph-Based Profiling for Cold-Start LLM Routing

Jingjun Xu1∗, Hongji Pu1∗, Tao Feng1∗, Haozhen Zhang2∗, Jiaxuan You1†, Ge Liu1† 1University of Illinois Urbana-Champaign 2Nanyang Technological University {jingjunx,hongjip2,taofeng2}@illinois.edu wazhz14@gmail.com

∗Equal contribution. †Corresponding author.

#### ulab-uiuc/RouteProfile Hugging Face Collection Abstract

[Figure 1]

constraints (Chen et al., 2023a). However, the practical effectiveness of routing systems depends on their ability to continually adapt to emerging queries and newly released models. New-LLM integration is particularly challenging: newly released models lack the query–response–reward interactions typically required for router training, and unlike new queries, they cannot be directly profiled through semantic embeddings. Existing routing systems, therefore, require large-scale inference, query-level data collection, and router retraining to incorporate new models, making adaptation costly and slow as model releases accelerate, as illustrated in Fig. 1. This motivates cold-start LLM routing, where a target LLM must be routed without queryresponse-reward data, analogous to cold-start recommendation for new items without interaction histories (Schein et al., 2002). Our key observation is that newly released LLMs often come with public signals from technical reports or model cards, which may support cold-start profiling despite the absence of query-response-reward interactions. We therefore ask: Can LLMs be effectively profiled using such coarse public signals to support cold-start routing?

LLM routing is increasingly important for selecting suitable models under diverse user needs and deployment constraints, but its practical effectiveness depends on continual adaptation to emerging queries and newly released models. New-LLM integration is particularly challenging, as newly released models lack the query-response-reward interactions required for router training and cannot be profiled as directly as new queries via semantic embeddings. Existing profiles are limited: LLMgenerated descriptions are often coarse, while interaction-based embeddings are costly to construct. To address this problem, we propose RouteProfile, a graph-based profiling framework that constructs LLM profiles from public signals in technical reports or model cards, including model family, model description, reported benchmark scores, and benchmark domains. RouteProfile organizes these heterogeneous signals into a graph and studies profile construction along four dimensions: organizational form, representation type, aggregation depth, and learning configuration. We evaluate RouteProfile in training-free cold-start routing and new-LLM integration settings. Experiments show that: (1) structured profiles outperform flat baselines in training-free cold-start routing; (2) model family metadata is more reliable than benchmark domain information; and (3) effective new-LLM integration requires profile–router co-design. Overall, our findings highlight the importance of profile design for enabling routing systems to adapt to the evolving model ecosystem.

Constructing model profiles under the cold-start constraint is challenging because newly released LLMs lack the query–response–reward interactions typically used for router training. Profiles must therefore be inferred from coarse public signals in technical reports or model cards, including model family, model descriptions, reported benchmark scores, and benchmark domains. Although accessible, these signals are sparse, heterogeneous, and only partially comparable: models are evaluated on different benchmark subsets, while textual descriptions, numerical scores, and relational metadata encode different types of information.

### 1 Introduction

As the large language model (LLM) ecosystem expands, individual models exhibit heterogeneous capabilities across queries, benchmarks, and domains. This heterogeneity motivates LLM routing, which selects the most suitable model for each query under diverse user needs and deployment

Existing LLM profile designs remain inadequate for cold-start routing. Index-based profiles represent each model as a one-hot vector (Zheng

- (a) Conventional Integration

- (b) Cold-Start Scenario

Large-Scale Inference Data Collection

Router Retraining

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Query Response Reward

Retrain on Interaction Histories

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

New LLM

[Figure 15]

Costly & Slow

T = 0

[Figure 16]

Model Family

###### Information from Technical Reports

###### No Query-Response -Reward Data

[Figure 17]

[Figure 18]

[Figure 19]

Add New LLM

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

LLM1

[Figure 25]

[Figure 26]

Model

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

LLM2

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Benchmark

[Figure 44]

New LLM

Domain

[Figure 45]

[Figure 46]

[Figure 47]

User Queries

Candidate LLMs

LLM Profiling Routing Decision

- Figure 1: Cold-start LLM Routing. (A) Conventional routing incurs high latency and cost for new-model integration due to large-scale inference, data collection, and router retraining; (B) Cold-start LLM routing profiles LLMs from public signals, including model family metadata, model description, reported benchmark scores, and benchmark domain information.

- et al., 2023), providing little semantic information for generalizing to unseen queries or new models. LLM-generated profiles describe candidate models using a strong LLM (Feng et al., 2025a; Zhang et al., 2025), but are often coarse, knowledgelimited, and incomplete. Benchmark-level summaries (Shnitzer et al., 2023) are cheaper to obtain, but discard structured relationships among models, benchmarks, and domains.

We systematically evaluate RouteProfile to understand how profile design affects cold-start LLM routing. Experiments cover two complementary scenarios: training-free cold-start routing with SimRouter, where no candidate LLM has queryresponse-reward interactions, and new-LLM integration with MLPRouter (Hu et al., 2024) and GraphRouter (Feng et al., 2025a), where a newly released LLM is added to an existing router using only its public profile, without retraining. In both settings, all profiles are constructed exclusively from public signals. Our evaluation yields three key findings: (1) profile structure (organizational form and aggregation depth) substantially improves training-free cold-start routing; (2) model family metadata is a more reliable public signal than benchmark domain information; (3) effective new-LLM integration depends critically on profile–router co-design (representation type and learning configuration). Overall, our results highlight profile design as a critical component for adapting routing systems to the expanding LLM ecosystem.

These limitations suggest that effective cold-start routing requires profiles that are both informative and structurally grounded. We therefore propose RouteProfile, a graph-based profiling framework that organizes heterogeneous public signals into a structured graph and analyzes profile construction along four dimensions: organizational form, representation type, aggregation depth, and learning configuration. Organizational form specifies whether profiles leverage graph structure. Representation type determines whether profiles are constructed through textual summaries or dense embedding computations. Aggregation depth controls how far information propagates over the graph. Learning configuration indicates whether aggregation is training-free or optimized through learning. Rather than enumerating all possible design variants, we aim to identify which profile design choices most critically affect routing performance under the cold-start constraints.

### 2 Related Work

LLM Routing. Recent work formulates multiLLM routing as an inference-time decision problem, assigning each query to a model under quality, cost, or latency constraints (Ding et al., 2024; Ong et al., 2025; Chen et al., 2023a). Existing

methods mainly focus on router design, including preference-trained, reward-guided, contrastive, and graph-based routing (Zhang et al., 2025; Ong et al., 2025; Chen et al., 2024; Feng et al., 2025a; Šakota et al., 2024). Some methods also use modelside signals such as benchmark statistics, metadata, or structured benchmark–query–model relations (Ong et al., 2025; Chen et al., 2024; Feng et al., 2025a), but typically treat them as auxiliary inputs rather than a standalone design problem. In contrast, we study LLM profile design and its effect across routers.

LLM Profiling. Prior work studies explicit profiling of model capabilities. QualEval (Murahari

- et al., 2024) derives natural-language capability groups for diagnosis, Skill-Slices (Moayeri et al.,

2024) recovers latent skills to reveal trade-offs hidden by aggregate benchmark scores, and EvalTree (Zeng et al., 2025) organizes model weaknesses through capability trees. More recently, BELLA explores skill-based profiling for costaware LLM routing (Okamoto et al., 2026). However, these works mainly target evaluation, diagnosis, or a specific routing framework, rather than profile design as a general routing problem.

### 3 Public Signals as a Heterogeneous Graph for Cold-start Profiling

This section first introduces the public data sources used for profile construction, and then formalizes them as a heterogeneous graph for principled LLM profile definition and systematic analysis.

Under the cold-start constraint, LLM profiles must be constructed without query-responsereward data, relying only on coarse public signals such as reported benchmark scores and model metadata. As illustrated in Figure 2, we construct LLM profiles from four primary sources: model family metadata, model description, reported benchmark scores, and benchmark domain information. Model family encodes the structural prior of each model, including its architectural lineage, series, and developer, and thus provides insight into inherent capabilities. Model description summarizes publicly available textual description of the model from technical reports or model cards, such as training objectives and model scale. Benchmark evaluation captures the model’s standardized performance in technical reports or model cards and, therefore, offers a comparable assessment of model capabilities. Domain coverage characterizes the

benchmark areas in which a model exhibits competence, highlighting its specialization and heterogeneity across domains.

Public data sources are sparse, heterogeneous, and only partially comparable, making direct profile construction challenging. To systematically integrate them, we formulate the multi-source information as a heterogeneous graph G = (V,E). Each node v ∈ V and edge e ∈ E are assigned types through mapping functions, with node type defined by ϕ : V → C and edge type defined by ψ : E → D. An edge connecting a pair of nodes is denoted as euv = (u,v). Specifically, we define 4 node types: model node vm, model family node vf, domain node vd, benchmark node vb; and 3 edge types: model-model family edge emf, modelbenchmark edge emb, and benchmark-domain edge ebd.

We then describe the features associated with nodes and edges. For node features x, we adopt different initialization strategies given the inherent differences among node types. In particular, we utilize an additional LLM, such as GPT-4o-mini (OpenAI, 2024), to generate textual descriptions for model nodes, domain nodes, and benchmark nodes using tailored prompts based on publicly available information. All generated descriptions can be found in Appendix A.2. These descriptions serve as node features in the text space and are further encoded by a pre-trained language model (PLM), such as Longformer (Beltagy et al., 2020), to obtain dense embeddings. For edge features r, only the model–benchmark edges are associated with features, which encode performance scores demonstrated on technical reports or authoritative LLM leaderboards, such as the Open LLM Leaderboard (Fourrier et al., 2024).

Finally, we define the LLM profile pm of a model node vm as:

pm = xˆvm = f(G)vm, (1)

where xˆvm denotes the aggregated representation of vm, and f is the information aggregation function over the heterogeneous graph G.

### 4 RouteProfile: Profile Design for Cold-Start LLM Routing

Next, we propose RouteProfile for LLM profile design in cold-start routing. Specifically, we focus on the design of the information aggregation function f.

###### Data Sources

###### Cold-Start Routing

###### LLM Profile Design Choices

[Figure 48]

[Figure 49]

###### Organizational Form

Aggre. Depth

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Model Family Task

Model Family

[Figure 55]

[Figure 56]

Model

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

3

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[

[Figure 74]

[Figure 75]

[Figure 76]

2

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

1

Model Family

[Figure 84]

[Figure 85]

Domain Query

[Figure 86]

[Figure 87]

[Figure 88]

Task

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

LLM Profile

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Model Profile

[Figure 113]

[Figure 114]

[Figure 115]

No QueryResponse-Reward Interactions!

Domain

K-Hop Aggregation

Flat Concatenation Structured Integration

Model

[Figure 116]

[Figure 117]

Learning Config.

Representation Type

[Figure 118]

Vectors

[Figure 119]

[Figure 120]

SimRouter MLPRouter

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

New Query

[Figure 146]

[Figure 147]

Task

GraphRouter

Text

Training-Free

Vector Computation

[Figure 148]

[Figure 149]

[Figure 150]

LLM Summarization

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Choose Best Model

Text-based Message Propagation

Emb-based Message Propagation

[Figure 162]

Domain

Trainable

- Figure 2: Overview of the RouteProfile. LLM profiles are constructed from coarse public signals comprising model family metadata, model description, reported benchmark scores, and benchmark domain information. The design choices are characterized along four dimensions: organizational form (flat/structured), representation type (text/embedding), aggregation depth (hop ∈ {0,1,2,...}), and learning configuration (training-free/trainable). Experiments are conducted across three representative routers to evaluate how profile design choices affect cold-start routing performance across different routing settings. "Aggre." and "Config." denote Aggregation and Configuration.

The RouteProfile includes four key dimensions as illustrated in Figure 2: organizational form, representation type, aggregation depth, and learning configuration. In defining the LLM profile design choices, we follow two guiding principles: (1) inclusiveness of dimensions that materially affect downstream routing performance; (2) conciseness by excluding overly task-specific choices, such as particular LLMs or graph neural networks (GNNs) used for information aggregation. Our goal is not to enumerate all possible design variants, but to show how a systematic view for understanding how different profile design choices affect cold-start routing performance.

tion within the graph, determining whether only direct neighbors or also higher-order neighborhoods contribute to the LLM profiles. Learning configuration indicates whether the aggregation function f is trainable. In a trainable setting, the aggregation function f can be optimized, for example, via self-supervised learning on the graph.

Formally, we define the function f as:

pm = xˆvm = f(ω,γ,K,ℓ)(G)vm, (2)

where ω ∈ {Flat,Structured} denotes the organizational form, γ ∈ {Text,Embedding} denotes the representation type, K ∈ {0,1,2,3,4} denotes the aggregation depth, and ℓ ∈ {Training-free,Trainable} denotes the learning configuration.

In particular, organizational form specifies whether the structural information in the heterogeneous graph is leveraged during aggregation. In a structured form, relational information is typically modeled through a GNN, whereas in a flat form, the available information is directly concatenated into plain text or a single vector. Representation type determines the information fusion mechanism, which can either be textual descriptions or dense embeddings. Textual representations are usually summarized by LLMs, whereas dense embeddings are often computed through neural networks, such as those in GNNs. Aggregation depth controls the extent of information propaga-

### 5 Experimental Setup

In this section, we describe the experimental setup for evaluating how design choices in LLM profiles affect cold-start routing performance. The setup comprises two parts: upstream profile construction, covering heterogeneous graph construction (Section 5.1) and instantiated design choices for constructing LLM profiles (Section 5.2); and downstream routing evaluation, including datasets and candidate LLMs (Section 5.3), evaluation tasks (Section 5.4), and routing methods (Section 5.5).

###### 5.1 Heterogeneous Graph Construction

We construct the heterogeneous graph using 15 datasets spanning 4 capability domains: knowledge, reasoning, math, and coding. Dataset statistics are summarized in the upper portion of Table 8, with detailed descriptions provided in Table 5. The graph further incorporates 25 LLMs from 5 model families to enrich relational signals across model families and task performance, of which 8 serve as candidate LLMs for downstream routing evaluation and the remainder as auxiliary nodes to improve graph connectivity and evidence diversity. Full statistics are reported in Table 9, with descriptions in Table 4.

###### 5.2 Instantiated Design Choices for LLM Profile Construction

We present concrete instantiations of the aggregation function f(γ,ω,K,ℓ), covering four representative configurations that vary along the dimensions defined in Section 4.

Flat Aggregation (γ = Text,ω = Flat,K = 0,ℓ = Training-free). Flat aggregation constructs the LLM profile directly in the text space without exploiting graph structure. Data associated with vm is sampled from G and concatenated into a textual description:

pm = f(Text,Flat,0,Training-free)(G)vm = C(S(vm)),

(3) where S(vm) denotes the sampled data for vm, and C(·) is a concatenation operator.

Text-based GNN (γ = Text,ω = Structured,K ∈ {1,2,3,4},ℓ = Training-free). Inspired by (Yu et al., 2025), the text-based GNN performs message passing entirely in the text space. The aggregation function updates each node v by prompting an LLM to summarize the textual attributes propagated from neighborhood N(v). At each propagation hop k, a node-type-specific prompt template T (·) organizes the current node text with neighboring textual states and available edge features into a unified prompt πv(k):

πv(k) = T x(vk−1),{(x(uk−1),ruv) | u ∈ N(v)} ,

(4) and the updated representation is obtained by querying an LLM:

x(vk) = LLM πv(k) . (5)

The LLM profile is then defined as pm = f(Text,Structured,K,Training-free)(G)vm = x(vKm).

Embedding-based GNN (γ = Emb,ω = Structured,K ∈ {1,2,3,4},ℓ = Training-free). The embedding-based GNN performs feature aggregation on the heterogeneous graph at the embedding level through message passing. Following a simplified GCN-style propagation inspired by (Feng et al., 2025b), node representations are updated at the embedding level:

x(vk) =

wuv |N(v) ∪ {v}||N(u) ∪ {u}|

x(uk−1).

u∈N(v)∪{v}

(6) where wuv = ruv if an edge feature is available, and wuv = 1 otherwise. Then the profile is pm = f(Emb,Structured,K,Training-free)(G)vm = x(vKm).

Trainable GNN(γ = Emb,ω = Structured,K ∈ {1,2,3,4},ℓ = Trainable). The trainable GNN extends the embedding-based GNN with a learnable aggregation optimized via a self-supervised masked reconstruction objective. A proportion of node and edge features is randomly masked, and the model is trained to reconstruct the masked attributes from the remaining graph context:

L = Lnode + Ledge, (7)

where Lnode and Ledge are both implemented as mean squared error (MSE) losses. The resulting

profile is pm = f(Emb,Structured,K,Trainable)(G)vm = x(vKm).

###### 5.3 Downstream Datasets and Candidate LLMs

We select 12 datasets spanning math, reasoning, knowledge, and coding, sampling 50 instances per dataset for downstream evaluation. Statistics are summarized in the lower portion of Table 8, with detailed descriptions in Table 5. Furthermore, routing is evaluated over a fixed candidate pool of 8 LLMs drawn from the Qwen2, Llama, Gemma2, Mistral, and Mixtral families, covering model scales from 3B to 176B parameters. Full statistics are reported in Table 9, with detailed descriptions in Table 4.

###### 5.4 Cold-Start Routing Tasks and Metrics

We evaluate RouteProfile under two complementary settings reflecting cold-start scenarios.

Training-free cold-start. In this setting, no candidate LLM has query-response-reward interactions, and routing decisions rely entirely on pro-

files derived from public signals. We adopt SimRouter because it requires no training, enabling routing without query-response-reward interactions and isolating the effect of profile design from router optimization. The evaluation metric is average response performance across queries, as introduced in Table 8. We additionally report three reference points: Oracle (per-query best model), SingleBest (the globally best single model applied to all queries), and Random (random selection averaged over seeds 1–5). New-LLM integration. This setting evaluates whether public-signal profiles allow a fixed trained router to generalize to a newly released LLM at inference time, without retraining. Specifically, we first train the router on queryresponse-reward interaction data from seven existing candidate LLMs and then freeze its parameters. We then introduce the new LLM into the candidate pool by adding it as a new node in the heterogeneous graph, initialized it using different profiling methods using only public information. The fixed router is then applied directly to user queries over the expanded candidate pool, with no parameter updates.

Mistral-Small-24B-Instruct-2501 is designated as the new LLM in our experiments. Beyond average performance, we define a metric called New-LLM Correct Integration Rate (NCIR) that jointly captures selection frequency and correctness for the new LLM:

Nnew∧correct N

, (8)

NCIR =

where N is the total number of queries, and Nnew∧correct is the number of queries both routed to and correctly answered by the new LLM.

###### 5.5 Routing Methods

We consider three representative routers across the two settings: SimRouter is used in the training-free cold-start setting; MLPRouter and GraphRouter are used in the new-LLM integration setting.

SimRouter is a similarity-based, non-parametric router that selects models by measuring similarity between the query representation and each candidate’s profile. As a training-free baseline, it directly reflects profile quality without any learned transformation, making it well-suited for evaluating profiles under the cold-start constraint.

MLPRouter (Hu et al., 2024) is a trainable router that projects query representations and model profiles into a shared latent space via sep-

arate MLPs, ranking candidates by similarity between projected representations. In the new-LLM integration setting, it is trained on query-responsereward interactions from old LLMs only; its ability to route queries to the new LLM therefore depends on the quality of the new LLM’s public-signal profile.

GraphRouter (Feng et al., 2025a) organizes tasks, queries, and candidate LLMs into a heterogeneous graph and applies a GNN with selfsupervised learning to model their relational interactions. Like MLPRouter, it is trained exclusively on old LLM query-response-reward data in the new-LLM integration setting, and evaluates whether graph-structured relational reasoning can further leverage public-signal profiles to generalize to newly introduced models.

### 6 Experimental Results

We present results across training-free cold-start (Tables 1–2) and new-LLM integration (Table 3) settings to examine how LLM profile design choices affect cold-start routing.

- 6.1 Profile Structure Improves Training-Free Cold-start Routing

Table 1 shows that profile structure substantially affects training-free cold-start routing. PlainText, which simply concatenates public signals, underperforms Single-Best (0.532 vs. 0.547), suggesting that flat profiles are insufficient. In contrast, structured profiles with sufficient aggregation depth close this gap: TextGNN-4hop (0.580) and EmbGNN-4hop (0.577) exceed Single-Best, showing that structural integration can improve routing without router training. Shallow aggregation is less effective, as TextGNN-1hop (0.526) and EmbGNN1hop (0.531) provide little or no gain over PlainText, indicating that one-hop propagation offers insufficient context for distinguishing model capabilities. As depth increases, TextGNN improves steadily, whereas EmbGNN varies across depths, suggesting that aggregation depth interacts with representation type.

- 6.2 Trainable Configurations Yield Larger but Depth-Sensitive Gains

Trainable profiles further improve routing performance. TrainGNN-1hop and TrainGNN-2hop both reach 0.613, outperforming the best trainingfree profile (TextGNN-4hop, 0.580), and moving

- Table 1: Routing performance under training-free cold-start across profile designs. Abbreviations: “Org.” = organizational form, “Rep.” = representation type, “Aggre.” = aggregation depth, “Learn. Config.” = learning configuration, “TF” = training-free, and “Tr” = trainable.

Method Org. Form Rep. Type Aggre. Hop Learn. Config. Average Performance

Oracle – – – – 0.679 Single-Best – – – – 0.547 Random Selection – – – – 0.508

PlainText Flat Text 0 TF 0.532

- EmbGNN-1hop Structured Emb 1 TF 0.531
- EmbGNN-2hop Structured Emb 2 TF 0.560
- EmbGNN-3hop Structured Emb 3 TF 0.534
- EmbGNN-4hop Structured Emb 4 TF 0.577

- TextGNN-1hop Structured Text 1 TF 0.526
- TextGNN-2hop Structured Text 2 TF 0.550
- TextGNN-3hop Structured Text 3 TF 0.566
- TextGNN-4hop Structured Text 4 TF 0.580

- TrainGNN-1hop Structured Emb 1 T 0.613
- TrainGNN-2hop Structured Emb 2 T 0.613
- TrainGNN-3hop Structured Emb 3 T 0.600
- TrainGNN-4hop Structured Emb 4 T 0.555

closer to Oracle performance (0.679). However, TrainGNN is sensitive to aggregation depth: performance drops from 0.613 at 2 hops to 0.600 at 3 hops and 0.555 at 4 hops. This degradation is consistent with over-smoothing, where repeated aggregation homogenizes node representations and weakens their discriminability. By contrast, training-free profiles generally benefit from deeper aggregation.

6.3 Model Metadata is More Reliable than Benchmark Domain

- Table 2 shows that model family metadata consistently improves routing across all profile designs. Adding family information to benchmark evaluation yields gains for EmbGNN-3hop (0.551 vs. 0.500), TextGNN-3hop (0.552 vs. 0.509), and TrainGNN-3hop (0.519 vs. 0.502). This consistency suggests that architectural lineage provides a stable structural prior for estimating model capabilities, even when no query-response-reward data is available. Such metadata is likely less sensitive to benchmark coverage variation and therefore serves as a robust public signal for cold-start profiling. In contrast, domain information is more dependent on the profile design and learning configuration. When added to Benchmark+Model Family, domain information substantially improves TrainGNN-3hop (0.600 vs. 0.519), suggesting that trainable profiles can learn to exploit coarse domain structure. However, it degrades EmbGNN-3hop (0.534 vs. 0.551), indicating that training-free embedding profiles may be more sensitive to noisy aligned domain

signals.

- 6.4 Embedding-Based Profiles Enable New-LLM Integration

Table 3 reveals a clear gap among profile types in routing queries to newly introduced LLMs. PlainText yields NCIR = 0.000 for both MLPRouter and GraphRouter, showing that flat profiles provide no effective signal for new-LLM integration. Textbased structured profiles also perform poorly on NCIR: TextGNN remains near zero across hops, reaching at most 0.007 with MLPRouter, despite competitive average performance. This gap indicates that textual profiles can support overall routing quality but fail to position the new LLM as a viable candidate within the router’s decision space.

In contrast, embedding-based structured profiles consistently improve new-LLM integration. EmbGNN-3hop achieves NCIR = 0.411 with GraphRouter and 0.272 with MLPRouter, while also obtaining the best average performance for both routers (MLPRouter: 0.624; GraphRouter: 0.613). These results show that dense graph-based representations better transfer to unseen LLMs and can improve both new-model integration and overall routing quality.

- 6.5 Profile–Router Co-Design Determines New-LLM Generalization

New-LLM integration further depends on the alignment between profile design and router architecture. TrainGNN illustrates this dependence: TrainGNN-

- Table 2: Cold-start routing performance across public data sources. We compare benchmark-, domain-, and model-family-level information across three profile configurations. Method abbreviations follow Table 1. All configurations include model nodes; profile design details are provided in Section 5.2.

Method Benchmark Domain Model Family Average Performance

EmbGNN-3hop

✓ ✓ ✓ 0.534 ✓ ✓ 0.503 ✓ ✓ 0.551 ✓ 0.500

TextGNN-3hop

✓ ✓ ✓ 0.566 ✓ ✓ 0.520 ✓ ✓ 0.552 ✓ 0.509

TrainGNN-3hop

✓ ✓ ✓ 0.600 ✓ ✓ 0.548 ✓ ✓ 0.519 ✓ 0.502

- Table 3: New-LLM integration performance across profile designs. Method abbreviations follow Table 1. NCIR denotes the New-LLM Correct Integration Rate defined in Eq. 8.

MLPRouter GraphRouter Average Performance NCIR Average Performance NCIR PlainText 0.532 0.000 0.532 0.000

Method

- EmbGNN-1hop 0.602 0.283 0.527 0.198
- EmbGNN-2hop 0.612 0.020 0.547 0.079
- EmbGNN-3hop 0.624 0.272 0.613 0.411

- TextGNN-1hop 0.610 0.007 0.541 0.000
- TextGNN-2hop 0.592 0.000 0.610 0.000
- TextGNN-3hop 0.594 0.007 0.503 0.000

- TrainGNN-1hop 0.530 0.000 0.582 0.400
- TrainGNN-2hop 0.529 0.000 0.524 0.000
- TrainGNN-3hop 0.530 0.000 0.515 0.013

1hop achieves NCIR = 0.400 with GraphRouter but NCIR = 0.000 across all hops with MLPRouter under the same profile. This suggests that graphstructured routers are better able to exploit relational information encoded by trainable GNN profiles, whereas MLP-based routers struggle to use such structure for unseen models. A similar pattern appears for EmbGNN. EmbGNN-3hop performs best with GraphRouter (NCIR = 0.411), whereas MLPRouter reaches its best NCIR at 1hop (0.283), indicating that the optimal aggregation depth is router-dependent. Overall, GraphRouter achieves higher or equal NCIR across profile designs, suggesting that graph-structured routers are better aligned with relational public-signal profiles. Thus, effective new-LLM integration requires not only informative profiles, but also profile–router co-design.

### 7 Conclusion

In this work, we propose RouteProfile, a graphbased profiling framework for cold-start LLM rout-

ing. RouteProfile constructs model profiles from public signals in technical reports or model cards, including model family metadata, model descriptions, reported benchmark scores, and benchmark domain information, and organizes them into a heterogeneous graph. This enables systematic analysis of profile design along four dimensions: organizational form, representation type, aggregation depth, and learning configuration. We evaluate RouteProfile in two settings: training-free coldstart routing and new-LLM integration. Our results reveal three key findings. First, structured profiles substantially improve training-free coldstart routing. Second, model family metadata is more reliable than benchmark domain information across profile designs. Third, effective new-LLM integration depends critically on profile–router codesign. Overall, RouteProfile demonstrates that public-signal profiling provides a viable basis for cold-start routing and highlights profile design as an important direction for adapting routing systems to the expanding LLM ecosystem.

### Limitations

This work has several limitations. First, public signals are incomplete and inconsistently reported across LLMs. Benchmark scores and model metadata are often drawn from technical reports or model cards with different evaluation protocols, benchmark coverage, and reporting granularity, which may introduce reporting bias into the constructed profiles. Second, our experiments cover representative profile designs and router architectures, but do not exhaustively evaluate all possible routers, model families, or deployment scenarios. The effectiveness of profile–router co-design may therefore vary with the candidate model pool and routing architecture. Finally, the new-LLM integration setting assumes that the router incorporates a newly released model using only its public profile and without retraining. In practice, routing systems may benefit from continually updating profiles as query-response-reward interactions accumulate. Exploring hybrid cold-start and continual adaptation strategies remains an important direction for future work.

### References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and 1 others. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. CoRR, abs/2004.05150.

Lingjiao Chen, Matei Zaharia, and James Zou. 2023a. Frugalgpt: How to use large language models while reducing cost and improving performance. arXiv preprint arXiv:2305.05176.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang. 2024. Routerdc: Query-based router by dual contrastive learning for assembling large language models. In Advances in Neural Information Processing Systems, volume 37. Curran Associates, Inc.

Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. 2023b. Theoremqa: A theorem-driven question

answering dataset. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7889–7901.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers), pages 2924–2936.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ruhle, Laks V. S. Lakshmanan, and Ahmed Hassan Awadallah. 2024. Hybrid llm: Cost-efficient and quality-aware query routing. In The Twelfth International Conference on Learning Representations.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378.

Tao Feng, Yanzhen Shen, and Jiaxuan You. 2025a. Graphrouter: A graph-based router for llm selections. In The Thirteenth International Conference on Learning Representations.

Tao Feng, Yexin Wu, Guanyu Lin, and Jiaxuan You. 2025b. Graph world model. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. 2024. Open llm leaderboard v2. https://huggingface. co/spaces/open-llm-leaderboard/open_llm_ leaderboard.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. 2024. Routerbench: A benchmark for multi-llm routing system. arXiv preprint arXiv:2403.12031.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, and 1 others. 2023. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in neural information processing systems, 36:62991– 63010.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), pages 3214–3252.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2381–2391.

Mazda Moayeri, Vidhisha Balachandran, Varun Chandrasekaran, Safoora Yousefi, Thomas Fel, Soheil Feizi, Besmira Nushi, Neel Joshi, and Vibhav Vineet. 2024. Unearthing skill-level insights for understanding trade-offs of foundation models. arXiv preprint arXiv:2410.13826.

Vishvak Murahari, Ameet Deshpande, Peter Clark, Tanmay Rajpurohit, Ashish Sabharwal, Karthik Narasimhan, and Ashwin Kalyan. 2024. Qualeval: Qualitative evaluation for model improvement. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2093–2111.

Mika Okamoto, Ansel Kaplan Erol, and Glenn Matlin. 2026. Trust by design: Skill profiles for transparent, cost-aware llm routing. arXiv preprint arXiv:2602.02386.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. 2025. Routellm: Learning to route llms from preference data. In The Thirteenth International Conference on Learning Representations.

OpenAI. 2024. Gpt-4o system card. CoRR, abs/2410.21276.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 conference on empirical methods in natural language processing, pages 2383–2392.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106.

Marija Šakota, Maxime Peyrard, and Robert West. 2024. Fly-swat or cannon? cost-effective language model choice via meta-modeling. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining. Association for Computing Machinery.

Andrew I. Schein, Alexandrin Popescul, Lyle H. Ungar, and David M. Pennock. 2002. Methods and metrics for cold-start recommendations. In Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 253–260. ACM.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, and 1 others. 2022. Language models are multilingual chain-ofthought reasoners. arXiv preprint arXiv:2210.03057.

Tal Shnitzer, Anthony Ou, Mírian Silva, Kate Soule, Yuekai Sun, Justin Solomon, Neil Thompson, and Mikhail Yurochkin. 2023. Large language model routing with benchmark datasets. CoRR, abs/2309.15789.

Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. 2024. Musr: Testing the limits of chain-of-thought with multistep soft reasoning. In International Conference on Learning Representations, volume 2024, pages 14670–14728.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and 1 others. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Haofei Yu, Zhaochen Hong, Zirui Cheng, Kunlun Zhu, Keyang Xuan, Jinwei Yao, Tao Feng, and Jiaxuan You. 2025. Researchtown: Simulator of human research community. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, and Pang Wei Koh. 2025. Evaltree: Profiling language model weaknesses via hierarchical capability trees. arXiv preprint arXiv:2503.08893.

Haozhen Zhang, Tao Feng, and Jiaxuan You. 2025. Router-r1: Teaching llms multi-round routing and aggregation via reinforcement learning. In The Thirtyninth Annual Conference on Neural Information Processing Systems.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,

Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36. Curran Associates, Inc.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2024. Agieval: A human-centric benchmark for evaluating foundation models. In Findings of the association for computational linguistics: NAACL 2024, pages 2299–2314.

### A Appendix

###### A.1 Implementation Details

All experiments were conducted on 1 NVIDIA A6000 GPU. Each profile construction with the text-based GNN required approximately 0.2 GPU hours due to iterative LLM calls using vLLM. Each profile construction with the emb-based GNN required approximately 0.1 GPU hours in total. Each profile construction with the trainable GNN experiments required approximately 0.3 GPU hours in total. Inference outputs from the candidate LLMs were obtained via the NVIDIA API, requiring approximately 15 hours in total. We use the allenai/longformer-base-4096 checkpoint from HuggingFace for encoding node text features, with a maximum sequence length of 4096 tokens. For the text-based GNN, Qwen3-8B (Yang et al., 2025) is queried with temperature 0 to ensure reproducibility. Graph neural network experiments are implemented using PyTorch Geometric (version 2.7.0). TrainGNN is trained for 100 epochs with a learning rate of 1e-3, batch size 64, and a masking ratio of 0.3 for both node and edge features, using the Adam optimizer. MLPRouter and GraphRouter are trained for 100 epochs with a learning rate of 1e-3 and 1e-4. Results in Tables 1–3 are reported from single runs. Random Selection is averaged over 6 random seeds (0–5).

###### A.2 Data Sources for LLM Profile Construction

We describe the initial node features used to construct the interaction graph for LLM profiling, covering four types of nodes: model family, model, benchmark, domain, and query. All datasets and models used in this work are publicly available and used in accordance with their intended academic research purposes.

###### A.2.1 Model Family Nodes

Each model family node is initialized with a natural language description capturing its architectural design, training methodology, and general capabilities:

- • Qwen2: A family of decoder-only Transformer models developed by Alibaba Cloud, trained on large-scale multilingual corpora with improvements in data quality, alignment, and long-context handling.
- • Gemma2: An open model family released by Google, featuring grouped-query attention and interleaved local-global attention for efficient inference and high-quality language modeling.
- • LLaMA: A family of autoregressive Transformer models developed by Meta AI, widely adopted as foundation models for research and downstream applications, including instruction following and conversational agents.
- • Mistral: A family of high-efficiency decoderonly models developed by Mistral AI, incorporating grouped-query and sliding-window attention for scalable and memory-efficient inference.
- • Mixtral: A mixture-of-experts extension of the Mistral architecture that selectively activates sparse expert networks per token, achieving high model capacity with efficient computation.

###### A.2.2 Model Nodes

Each model node is initialized using its model family description as the base text feature, supplemented with model-specific attributes including parameter count, instruction-tuning status, and available model card information from representative model families such as Qwen (Yang et al., 2025), Gemma (Team et al., 2024), Llama (Grattafiori et al., 2024), and Mixtral (Jiang et al., 2024). This allows model nodes to inherit shared architectural priors from their family while retaining individual characteristics.

###### A.2.4 Domain Nodes

Each domain node represents a high-level capability category. We define six domains, each characterized by a natural language description that serves as the initial text feature:

- • Knowledge: Tasks requiring broad factual knowledge, academic understanding, and evidence-grounded question answering across domains.
- • Reasoning: Tasks requiring commonsense reasoning, multi-step inference, logical deduction, and robust decision making.
- • QA: Tasks centered on question answering with retrieval, reading comprehension, and answer faithfulness to provided evidence.
- • Math: Tasks requiring arithmetic, symbolic manipulation, theorem application, and multistep mathematical problem solving.
- • Coding: Tasks requiring program synthesis, debugging, and functional correctness under executable unit tests.
- • Alignment: Tasks evaluating instruction following, helpfulness, harmlessness, truthfulness, and preference alignment in assistant behavior.

###### A.2.5 Query Nodes

Query nodes represent individual queries sampled from each benchmark dataset. For each dataset, up to 1,000 queries are randomly sampled to serve as query nodes in the interaction graph. Each query node is initialized by encoding the raw query text using a pre-trained language model. Table 6 lists the datasets and their corresponding Hugging Face identifiers used for query sampling.

###### A.3 Prompts for Text-based GNN

At each propagation hop, every node in the interaction graph is updated by an LLM that synthesises information from its local neighbourhood. The prompts are designed to reflect the heterogeneous structure of the graph, with distinct templates for each node type.

###### A.2.3 Benchmark Nodes

Each benchmark node is initialized with a natural language description of the benchmark. Table 5 summarizes all datasets used in this work along with their descriptions.

###### A.4 Dataset Statistics

Table 8 summarizes the datasets used in this work, divided into two groups: those used for evidence graph construction during LLM profiling, and those used for routing evaluation.

Table 4: Model nodes and their descriptions used in the interaction graph.

Model Description Candidate LLMs Qwen2.5-7B-Instruct An upgraded 7B Qwen model with enhanced multilingual capabilities across

diverse language tasks. Gemma-2-9B-IT A 9B instruction-tuned model from Google designed for general text processing and conversational applications.

- Llama-3.1-8B-Instruct Meta’s 8B model from the Llama-3 series, designed for conversational AI and complex reasoning tasks.

Mixtral-8x7B-Instruct A 56B mixture-of-experts model composed of eight 7B expert models, optimized for creative text generation. Mixtral-8x22B-Instruct An advanced 176B MoE model comprising eight 22B expert components, delivering strong performance across diverse tasks.

- Llama-3.2-3B-Instruct Meta’s ultra-lightweight 3B model optimized for speed and efficiency, ideal for simple tasks requiring fast responses.

Mistral-Small-24B-Instruct Mistral AI’s latest compact model delivering strong performance from 24B

parameters, excelling at instruction-following tasks. Auxiliary Models

- Llama-3.3-70B-Instruct Meta’s 70B multilingual instruction model focused on high-quality dialogue, reasoning, coding, and tool use.

Qwen2.5-3B-Instruct A 3B instruction model in the Qwen2.5 family, suited for low-cost applications and efficient local inference. Qwen2.5-14B-Instruct A 14B instruction model offering strong reasoning, knowledge use, and instruction-following for production workflows. Qwen2.5-32B-Instruct A 32B model built for stronger reasoning, richer world knowledge, and reliable long-form generation. Qwen2.5-72B-Instruct The 72B model in the Qwen2.5 series, built for top-tier reasoning and knowledge-intensive generation. Gemma-2-2B-IT Google’s 2B instruction-tuned Gemma 2 model, offering a balanced blend of reasoning and response generation. Gemma-2-27B-IT Google’s 27B instruction-tuned Gemma 2 model, delivering strong reasoning and response quality for high-quality workloads. Llama-3.2-1B-Instruct Meta’s 1B instruction model optimized for fast, efficient text generation in constrained environments. Mistral-Nemo-Instruct A compact yet capable 12B instruction model jointly developed by Mistral AI and NVIDIA, strong in chat, coding, and multilingual tasks. Qwen2.5-7B-Instruct-1M Extended-context version of Qwen2.5-7B, supporting up to 1M tokens for long-document analysis and complex workflows. Qwen2.5-14B-Instruct-1M Combines stronger 14B reasoning with 1M token context support for advanced long-context enterprise workflows. Qwen2-7B-Instruct A 7B instruction model from Qwen2, offering a strong balance of chat quality, reasoning, and multilingual usability. Qwen2-72B-Instruct The 72B instruction model in the Qwen2 family, designed for premium assistants and demanding production workloads. Llama-3.1-70B-Instruct Meta’s high-capability multilingual 70B instruction model for strong dialogue, reasoning, coding, and knowledge-intensive generation. Ministral-8B-Instruct Mistral AI’s edge-focused 8B model, built for latency-sensitive assistants and compact production systems.

Mistral-Small-Instruct-2409 A capable mid-sized 22B instruction model for general text generation, multilingual tasks, and function-calling workflows.

Mistral-Large-Instruct-2411 Mistral AI’s advanced 123B model built for state-of-the-art reasoning, coding, and long-context understanding.

###### A.5 LLM Statistics

Table 9 summarizes the LLMs used in this work, divided into candidate models that participate in routing and auxiliary models that serve as additional graph context nodes during profile construction.

###### A.6 AI Usage

We used AI-assisted writing tools (ChatGPT, Claude) to polish the language and improve the

clarity of this paper. All technical content, experimental design, analysis, and conclusions are entirely the work of the authors. The use of these tools is limited to linguistic refinement and does not affect the scientific contributions of this work.

Table 5: Benchmark nodes and their descriptions used in the interaction graph.

Benchmark Description BBH (Suzgun et al., 2023) A challenging subset of BIG-Bench focusing on tasks where earlier models

performed significantly below human level, spanning multi-step arithmetic, logical reasoning, and complex language understanding.

MATH500 (Lightman et al., 2024) A curated 500-problem subset of the MATH benchmark covering competitionlevel mathematics including algebra, geometry, number theory, and combinatorics.

GPQA (Rein et al., 2023) A graduate-level benchmark with expert-authored multiple-choice questions in physics, chemistry, and biology, designed to resist simple retrieval-based answering.

MuSR (Sprague et al., 2024) A benchmark for multi-step and structured reasoning that requires integrating multiple pieces of information through sequential inference. MMLU-Pro (Wang et al., 2024) An enhanced version of MMLU with more difficult questions and expanded answer choices, designed to better evaluate reasoning ability. MMLU (Hendrycks et al., 2020) A broad multiple-choice benchmark covering 57 academic and professional subjects across humanities, social science, STEM, and other domains.

C-Eval (Huang et al., 2023) A Chinese standardized-exam benchmark spanning dozens of disciplines for evaluating Chinese language understanding and reasoning in exam-style settings. AGIEval (Zhong et al., 2024) A human-centric benchmark derived from official admission and qualification

exams (e.g., SAT, Gaokao) to evaluate general reasoning and problem-solving.

TriviaQA (Joshi et al., 2017) A large-scale QA benchmark with trivia questions and evidence documents, testing knowledge retrieval and answer generation under noisy real-world evidence. Natural Questions (Kwiatkowski et al., 2019) A real-user QA benchmark based on anonymized Google queries with Wikipedia

evidence, evaluating short-answer and long-answer question answering. SQuAD (Rajpurkar et al., 2016) A reading comprehension benchmark of crowd-authored questions on Wikipedia passages, where answers are extracted text spans.

TheoremQA (Chen et al., 2023b) A STEM theorem-driven QA benchmark with university-level problems across math, CS/EE, physics, and finance, testing formal reasoning and theorem application.

CommonsenseQA (Talmor et al., 2019) A multiple-choice commonsense benchmark built from ConceptNet relations, requiring implicit everyday knowledge. WinoGrande (Sakaguchi et al., 2021) A large-scale pronoun coreference benchmark testing robust commonsense reasoning in Winograd-style disambiguation. ARC-Challenge (Clark et al., 2018) The difficult split of the AI2 Reasoning Challenge, containing grade-school science questions that require deeper reasoning beyond retrieval. OpenBookQA (Mihaylov et al., 2018) A science QA benchmark requiring multi-hop reasoning by combining core facts with external commonsense knowledge. BoolQ (Clark et al., 2019) A yes/no QA benchmark built from real user queries paired with evidence passages, testing binary reading comprehension and inference. DROP (Dua et al., 2019) A reading comprehension benchmark requiring discrete reasoning such as counting, comparison, and arithmetic over paragraphs. GSM8K (Cobbe et al., 2021) A grade-school math word-problem benchmark with multi-step natural-language solutions for evaluating arithmetic reasoning. MGSM (Shi et al., 2022) A multilingual extension of GSM8K-style math problems enabling cross-lingual evaluation of multi-step mathematical reasoning. HumanEval (Chen et al., 2021) A code-generation benchmark of hand-written programming problems with hidden unit tests evaluating functional correctness. MBPP (Austin et al., 2021) A benchmark of around one thousand crowd-sourced entry-level Python tasks with reference tests for practical code generation. TruthfulQA (Lin et al., 2022) A benchmark measuring whether models produce truthful answers rather than imitating common human misconceptions.

Table 6: Hugging Face dataset identifiers used for query node construction.

Dataset Hugging Face Identifier

IFEval google/IFEval BBH lukaemon/bbh MATH500 HuggingFaceH4/MATH-500 GPQA Idavidrein/gpqa MuSR TAUR-Lab/MuSR MMLU-Pro TIGER-Lab/MMLU-Pro EvalPlus evalplus/humanevalplus MultiPL-E nuprl/MultiPL-E C-Eval ceval/ceval-exam AGIEval English lighteval/agi_eval_en SQuAD rajpurkar/squad TheoremQA TIGER-Lab/TheoremQA WinoGrande allenai/winogrande BoolQ google/boolq DROP ucinlp/drop TruthfulQA domenicrosati/TruthfulQA WildBench allenai/WildBench

Table 7: Prompt templates for Text-GNN aggregation across different node types.

Node Type Input Context Instruction Output

Model Model family; benchmark scores grouped by domain; representative queries ranked by similarity to connected datasets

Synthesise all context into a unified capability profile covering architecture, domain-level performance, and query suitability

3–5 sentence capability profile

Dataset Parent domain; models evaluated with scores; representative queries from the benchmark

Describe what capability the benchmark evaluates, which models perform well or poorly, and what query types it covers

2–4 sentence benchmark profile

2–4 sentence domain profile

Domain All benchmark datasets belonging to this domain

Characterise the capability area and summarise the benchmark landscape within it

Model Family All model nodes that instantiate this architecture

Describe key design characteristics and the typical capability profile of models built on this architecture

2–4 sentence architecture profile

Query Not updated — raw query text is preserved throughout all hops as a stable semantic anchor.

Table 8: Overview of Datasets for Profile Construction and Routing Evaluation.

##### Usage Dataset Benchmark Type Metric Cases

BBH Reasoning Accuracy 1000 MATH500 Math Accuracy 500 GPQA-Diamond Knowledge / Reasoning Accuracy 198 MUSR Reasoning Accuracy 756 MMLU-Pro Knowledge Accuracy 1000 AGIEval Knowledge Accuracy 29 TheoremQA Math / Reasoning Accuracy 800 DROP Reasoning Accuracy 1000 TruthfulQA Reasoning Accuracy 817 WinoGrande Reasoning Accuracy 1000 BoolQ Reasoning Accuracy 1000 C-Eval Knowledge Accuracy 1000 SQuAD Knowledge Accuracy 1000 MultiPL-E Coding Accuracy 1000 EvalPlus Coding Accuracy 164

Profile Construction

MGSM Math Accuracy 50 GSM8K Math Accuracy 50 AgentVerse Reasoning Accuracy 50 CommonsenseQA Reasoning Accuracy 50 OpenBookQA Reasoning Accuracy 50 ARC-Challenge Reasoning Accuracy 50 MMLU Knowledge Accuracy 50 NaturalQA Knowledge Accuracy 50 TriviaQA Knowledge Accuracy 50 CommonGen Knowledge Accuracy 50 MBPP Coding Accuracy 50 HumanEval Coding Accuracy 50

Routing Evaluation

Table 9: Statistics of Candidate and Auxiliary LLMs.

###### Role LLM Size Model Family

- Llama-3.2-3B-Instruct 3B Llama Qwen2.5-7B-Instruct 7B Qwen2.5 Llama-3.1-8B-Instruct 8B Llama Gemma-2-9B-IT 9B Gemma2 Mistral-Small-24B-Instruct-2501 24B Mistral Mixtral-8x7B-Instruct-v0.1 56B Mixtral
- Llama-3.3-70B-Instruct 70B Llama Mixtral-8x22B-Instruct-v0.1 176B Mixtral

Candidate

Llama-3.2-1B-Instruct 1B Llama Gemma-2-2B-IT 2B Gemma2 Qwen2.5-3B-Instruct 3B Qwen2.5 Qwen2-7B-Instruct 7B Qwen2 Qwen2.5-7B-Instruct-1M 7B Qwen2.5 Ministral-8B-Instruct-2410 8B Mistral Mistral-Nemo-Instruct-2407 12B Mistral Qwen2.5-14B-Instruct 14B Qwen2.5 Qwen2.5-14B-Instruct-1M 14B Qwen2.5 Mistral-Small-Instruct-2409 22B Mistral Gemma-2-27B-IT 27B Gemma2 Qwen2.5-32B-Instruct 32B Qwen2.5 Qwen2-72B-Instruct 72B Qwen2 Qwen2.5-72B-Instruct 72B Qwen2.5 Llama-3.1-70B-Instruct 70B Llama Mistral-Large-Instruct-2411 123B Mistral

Auxiliary

