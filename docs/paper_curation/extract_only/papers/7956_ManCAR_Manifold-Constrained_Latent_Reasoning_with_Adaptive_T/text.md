# arXiv:2602.20093v1[cs.IR]23Feb2026

## ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation

[Figure 1]

Kun Yang1, Yuxuan Zhu2, Yazhe Chen1, Siyao Zheng3, Bangyang Hong2, Kangle Wu2, Yabo Ni2, Anxiang Zeng2, Cong Fu2, Hui Li1†

Abstract

Sequential recommendation increasingly employs latent multi-step reasoning to enhance test-time computation. Despite empirical gains, existing approaches largely drive intermediate reasoning states via target-dominant objectives without imposing explicit feasibility constraints. This results in latent drift, where reasoning trajectories deviate into implausible regions. We argue that effective recommendation reasoning should instead be viewed

ManCAR Reasoning

Unconstrained Reasoning

- as navigation on a collaborative manifold rather than free-form latent refinement. To this end, we propose ManCAR (ManifoldConstrained Adaptive Reasoning), a principled framework that grounds reasoning within the topology of a global interaction graph. ManCAR constructs a local intent prior from the collaborative neighborhood of a user’s recent actions, represented as a distribution over the item simplex. During training, the model progressively aligns its latent predictive distribution with this prior, forcing the reasoning trajectory to remain within the valid manifold. At test time, reasoning proceeds adaptively until the predictive distribution stabilizes, avoiding over-refinement. We provide a variational interpretation of ManCAR to theoretically validate its drift-prevention and adaptive test-time stopping mechanisms. Experiments on seven benchmarks demonstrate that ManCAR consistently outperforms state-of-the-art baselines, achieving up to a 46.88% relative improvement w.r.t. NDCG@10. Our code is available
- at https://github.com/FuCongResearchSquad/ManCAR.

Target Item

Start

Graph Induced

Graph Neighbors

Manifold

Figure 1: Illustration of constrained versus unconstrained latent reasoning. Graph-conditioned reasoning trajectories remain within a collaborative manifold defined by neighbor items, enabling stable and directed refinement toward the target. In contrast, unconstrained reasoning may drift outside feasible regions, leading to inefficient or unstable paths.

Computation for Sequential Recommendation . In . ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

#### 1 Introduction

Sequential recommendation has been significantly reshaped by the growing adoption of generative modeling paradigms [4]. Inspired by Large Language Models (LLMs), recent work has begun to explore latent multi-step reasoning in sequential recommendation to extend test-time computation [3, 20, 30, 31]. In LLMs, such reasoning is commonly realized through the recursive depth paradigm, where chain-of-thought (CoT) tokens are replaced by un-decoded latent states produced by the model [5, 8, 17, 25]. These latent states are iteratively refined for multiple steps using shared model parameters, and only decoded back to the output space at the final step, effectively increasing the model’s computational depth without expanding its architecture. This paradigm offers a natural and efficient template for incorporating reasoning into sequential recommendation, without requiring explicit textual representations.

#### CCS Concepts

• Information systems → Recommender systems.

#### Keywords

Sequential Recommendation, Latent Reasoning

ACM Reference Format:

Kun Yang1, Yuxuan Zhu2, Yazhe Chen1, Siyao Zheng3, Bangyang Hong2, Kangle Wu2, Yabo Ni2, Anxiang Zeng2, Cong Fu2, Hui Li1† . 2026. ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time

1Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, China.

- 2Shopee Pte. Ltd.
- 3School of Informatics, Xiamen University, China. † Hui Li is the corresponding author. hui@xmu.edu.cn .

Despite empirical gains, existing methods remain poorly understood. They typically guide latent reasoning using target-dominant objectives [30, 31], such as supervising only the final reasoning state with the target item, or mapping each intermediate state to an item probability distribution and progressively concentrating it toward a target one-hot distribution. However, they impose no explicit constraints on the evolution of intermediate reasoning states. As a result, the latent reasoning trajectory is largely unconstrained and retains excessive degrees of freedom while “walking” through the item space. This often leads to latent drift (Fig. 1), where intermediate states migrate into regions that are poorly aligned with

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference’17, Washington, DC, USA © 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

user preferences. Such drift is particularly detrimental at test time, degrading model robustness and generalization.

From a recommendation perspective, an overlooked but fundamental property of user behavior is that interactions are inherently collaborative rather than independent. This naturally motivates guiding latent reasoning using an item interaction graph, which encodes collective patterns across users. Users with similar preferences tend to interact with similar items, and item transitions exhibit regularities shaped by population-level behaviors. These collaborative signals naturally define a notion of plausibility: given a user’s recent interactions, only a subset of items is realistically relevant in the near future. Such assumptions are widely adopted in graph-based recommendation [2, 14, 35, 36, 39, 40, 43, 45], where a user’s next interaction is expected to lie within the local neighborhood of their recent interests.

In this paper, we propose ManCAR (Manifold-Constrained Adaptive Reasoning), a principled framework that grounds latent reasoning within the topology of a global interaction graph. Rather than naively enumerating graph traversal paths as reasoning trajectories, which is computationally expensive and unnecessary for latent reasoning, we leverage the interaction graph as a feasibility constraint on the reasoning process. Specifically, we treat the neighborhood induced by the item graph as a manifold constraint, restricting latent reasoning trajectories to evolve within collaboratively reachable regions while refining toward the target item. In probabilistic terms, this constraint corresponds to a region on the item probability simplex where items connected to the user’s recent actions are assigned substantially higher probability mass than unrelated items. This feasibility view naturally admits a variational interpretation of latent reasoning. Introducing latent reasoning states can be viewed as performing inference over an intermediate intent variable, with the graph-induced neighborhood serving as a structure-aware prior. Latent reasoning can then be formulated using an objective similar to the Evidence Lower Bound (ELBO), which balances target prediction with reasoning feasibility.

Besides, while the manifold constraint defines where latent reasoning can evolve, it leaves open the question of when reasoning should terminate. Since we train reasoning states to traverse collaboratively feasible regions on the item probability simplex toward the target, further refinement becomes uninformative once the item probability distribution produced by the latent state stabilizes. This motivates us to design a convergence-based stopping criterion for MacCAR, allowing test-time computation to terminate adaptively when the model has sufficiently localized the target region.

Our contributions can be summarized as follows:

- • We propose ManCAR, a framework guiding latent reasoning by interpreting collaborative neighborhoods in the interaction graph as feasibility constraints on the item probability simplex, mitigating latent drift. ManCAR further enables adaptive testtime computation via a convergence-based stopping criterion.
- • We theoretically establish a variational interpretation of ManCAR, demonstrating how it prevents latent drift and confirm the validity of our adaptive test-time stopping mechanism.
- • Experiments on benchmarks demonstrate that ManCAR consistently improves effectiveness over state-of-the-art baselines, achieving up to a 46.88% relative improvement w.r.t. NDCG@10.

N Neighbor Item T Target Item A Recent Action

Teacher prior (q)

|Top-k List|
|---|

Only Inference H Reasoning

[Figure 2]

Representation E Items

𝑝𝜃𝑡'

KL KL KL KL

Representation

N N

|A|
|---|

|A|
|---|

|A|
|---|

A

|𝑫𝑲𝑳 𝒑𝜽𝒕′−1| 𝒑𝜽𝒕′ < 𝜺<br><br>|
|---|

|Recent History|
|---|

T

N

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Adaptive Halting

### SeqRec Model

| |𝑯<br><br>||𝑯||avg(E)|
|---|---|

| |
|---|

| |
|---|

|A|
|---|

|A|
|---|

|A|
|---|

| |
|---|

|N|
|---|

|N|
|---|

|N|
|---|

| |
|---|

|Interaction History (H)|
|---|

|Context (C)|
|---|

|Reasoning|
|---|

Figure 2: Overview of ManCAR. ManCAR performs multistep latent reasoning constrained by a graph-induced candidate set. At each step, the reasoning state is regularized toward a scheduled teacher prior defined on collaboratively reachable items, ensuring manifold-consistent refinement. Adaptive test-time termination stops reasoning when the induced item distributions stabilize.

#### 2 Our Proposed ManCAR

Fig. 2 depicts ManCAR illustrated in this section. We begin by formalizing the problem setting and the used notation in Sec. 2.1. In Sec. 2.2, we introduce the core concept of manifold-constrained reasoning, postulating that valid latent reasoning trajectories should be confined to the local neighborhood of the user’s recent interactions. To operationalize this, Sec. 2.3 derives a variational training objective that treats latent reasoning as approximate inference over an intent variable, regularized by a graph-conditioned teacher prior. Sec. 2.4 provides a theoretical analysis of this objective, interpreting the KL-divergence regularization as a gradient flow that enforces local graph smoothness. Based on this theoretical foundation, Sec. 2.5 details the practical implementation of ManCAR, including teacher prior construction strategies and the overall loss function. Finally, Sec. 2.6 introduces a training mechanism for scheduling the teacher distribution during training, which naturally enables an adaptive termination criterion at test time.

#### 2.1 Problem Setting and Notation

Let I denote a finite set of items and U the set of users. We consider the sequential recommendation setting, where each user 𝑢 ∈ U interacts with items over time. For a given user, the interaction history is denoted by 𝐻 = (𝑖1,𝑖2, . . .,𝑖𝑇−1),𝑖𝑡 ∈ I, where items are ordered chronologically. The objective is to predict the next item 𝑖∗ = 𝑖𝑇 conditioned on the observed history 𝐻.

We model collaborative signals among items using a global item interaction graph G = (I, E), where nodes correspond to items and edges encode co-interaction relationships aggregated across users. An edge (𝑖, 𝑗) ∈ E indicates that items 𝑖 and 𝑗 are frequently cointeracted or consecutively consumed by users. For an item 𝑖 ∈ I, we denote its 𝑘-hop graph neighborhood by N(𝑖; G;𝑘).

Following standard latent reasoning settings [31], we denote by h𝑡 ∈ R𝑑 the hidden representation produced by the backbone encoder at step 𝑡, i.e., h𝑡 = 𝑓 (𝐻 [: 𝑡]), where 𝑓 (·) is typically a Transformer-based encoder. We further introduce r𝑡′ ∈ R𝑑, 𝑡′ ∈ {1, . . .,𝑇′}, to denote the latent reasoning states generated through iterative refinement. Unless otherwise specified, the initial reasoning state is set to the final encoder state, i.e., r1 = h𝑇−1.

#### 2.2 Manifold-Constrained Latent Reasoning

We now articulate the core conceptual motivation behind ManCAR. As discussed in Sec. 1, standard latent reasoning based methods suffer from unconstrained degrees of freedom, leading to latent drift. To address this, we propose a geometric perspective: reasoning should not traverse the latent space freely but must be viewed as navigation constrained to a “collaborative manifold”.

We adopt a latent multi-step reasoning setting [31]. Starting from the final encoder state h𝑇−1, the model generates a sequence of reasoning states r𝑡′ (𝑡′ ∈ {1, . . .,𝑇′}), each representing an intermediate hypothesis of user intent, and produces the final recommendation by decoding the last state.

A central challenge in this setting is that, without additional knowledge, the evolution of latent reasoning states is weakly constrained and exhibits excessive degrees of freedom, particularly in high-dimensional spaces. To address this issue, we introduce a graph-conditioned feasibility constraint that explicitly restricts where latent reasoning can evolve. This design is motivated by a fundamental property of recommender systems: user behavior is inherently collaborative, and given a user’s most recent interactions 𝐼𝑛 = (𝑖𝑇−𝑛, . . .,𝑖𝑇−1) where 𝑛 denotes the window size of the recent interaction, only a limited subset of items is plausibly relevant in the near future [36, 43]—namely, those that are 𝑘-hop reachable on the collaborative item interaction graph G.

From a geometric perspective, this graph-conditioned neighborhood defines a low-dimensional feasible region within the highdimensional item space. To operationalize this constraint during latent reasoning, we explicitly regulate how reasoning states are translated into item-level beliefs. Specifically, each reasoning state is mapped to an item probability distribution on the item probability simplex, and the graph-conditioned neighborhood restricts valid distributions to those that concentrate probability mass on collaboratively reachable items. This restriction defines a structured subregion of the simplex, which we refer to as the collaborative manifold. Latent reasoning is therefore constrained to evolve along this manifold, rather than freely over the entire simplex as in unconstrained latent refinement (Fig. 1).

To make the collaborative manifold explicit and tractable, we instantiate it using a finite set of collaboratively reachable items conditioned on the user’s most recent interactions. This set defines the feasible support of item probability distributions during latent reasoning. Concretely, we define a finite candidate set:

###### C(𝐼𝑛, G,𝑘) ⊆ {𝐼𝑛} ∪ N(𝐼𝑛; G;𝑘).

Unless otherwise specified, we use C(𝑘) as shorthand for C(𝐼𝑛, G,𝑘), since the candidate set is always constructed from the most recent items 𝐼𝑛 on the interaction graph G in this paper. In the next subsection, we derive a training objective from a variational

interpretation that leverages this graph-conditioned feasible set to regularize latent reasoning.

#### 2.3 Variational Training Objective

While the manifold constraint provides a strong geometric intuition, optimizing it effectively requires a rigorous mathematical formulation. In this subsection, we translate the conceptual constraint from Sec. 2.2 into probabilistic variational inference. We derive an Evidence Lower Bound (ELBO)-like objective, treating latent reasoning as approximate inference over an intermediate intent variable. This derivation introduces the “Teacher Prior” distribution induced from the interaction graph, guiding the implementation of the loss function (Sec. 2.5) necessary to train the model.

Latent Variable Formulation. Given a user history 𝐻, we introduce a discrete latent variable 𝑐 ∈ C(𝑘) representing an intermediate intent prototype that mediates the prediction of the next item. The conditional likelihood of the target item can be written as

𝑝𝜃 (𝑖∗ | 𝐻) = ∑︁

𝑝𝜃 (𝑐 | 𝐻) 𝑝𝜃 (𝑖∗ | 𝑐,𝐻),

𝑐∈C(𝑘)

where 𝑝𝜃 (𝑐 | 𝐻) is a history-conditioned intent distribution parameterized by 𝜃, and 𝑝𝜃 (𝑖∗ | 𝑐,𝐻) models the likelihood of the target given the inferred intent. We restrict 𝑐 to lie in the graph-induced candidate set, reflecting the assumption that user intent at each reasoning step is best characterized by collaboratively reachable items. This design anchors latent intent to observable interaction patterns and aligns with the collaborative manifold constraint.

Graph-Conditioned Variational Prior. Based on the above design, we introduce a graph-conditioned teacher distribution 𝑞(𝑐 |

𝐼𝑛, G), defined over the candidate set C(𝑘). It encodes prior knowledge about plausible intents reachable from the user’s most recent interaction, and is constructed independently of the model parameters 𝜃. Intuitively, 𝑞 assigns higher probability mass to items that are strongly connected to 𝐼𝑛 in the interaction graph.

ELBO-like Objective. For any choice of 𝑞(𝑐 | 𝐼𝑛, G), the loglikelihood admits the following lower bound:

log𝑝𝜃 (𝑖∗ | 𝐻) ≥ E𝑞(𝑐|𝐼𝑛,G) log𝑝𝜃 (𝑖∗ | 𝑐,𝐻)

(1)

− 𝐷KL(𝑞(𝑐 | 𝐼𝑛, G) ∥ 𝑝𝜃 (𝑐 | 𝐻)) ,

where 𝐷𝐾𝐿(· ∥ ·) is the KL-divergence. The derivation of Eq. 1 is provided in Appendix A. This formulation can be interpreted as an Evidence Lower Bound (ELBO). The first term encourages accurate target prediction under graph-feasible intents, while the KL term regularizes the model’s inferred intent distribution to align with the graph-conditioned prior.

Connection to Context Engineering and Latent Reasoning. In practice, the first term in Eq. 1 encourages the model to predict the target item 𝑖∗ conditioned on both the user history 𝐻 and the intent prototype𝑐. This requires injecting future candidate knowledge into the model’s input or conditioning pathway, which closely parallels Context Engineering for LLMs [22], where external or structured context is provided to guide prediction and reduce uncertainty. In our setting, the candidate set derived from the interaction graph serves as structured contextual knowledge that narrows the model’s predictive focus to collaboratively plausible regions.

Meanwhile, the intent distribution 𝑝𝜃 (𝑐 | 𝐻) is implicitly induced by latent reasoning states through model’s output layer, by projecting each latent state onto item probability simplex. Minimizing the KL divergence term aligns this induced distribution with the graph-conditioned prior, constraining each reasoning step to remain within the collaborative manifold supported by C(𝑘). The variational regularization explicitly limits the freedom of latent refinement and mitigates latent drift during iterative reasoning.

#### 2.4 Local Graph Smoothness by KL Distillation

The variational objective in Eq. 1 introduces a KL-divergence regularization term. To justify why this specific term is effective against latent drift, we provide a theoretical analysis in this subsection. We prove that minimizing this KL term induces a gradient flow that promotes “local graph smoothness”. This analysis bridges the gap between our probabilistic objective and the geometric manifold constraint, showing how the variational objective explicitly confines the reasoning trajectory by collaboratively feasible items.

- Proposition 2.1 (Local Graph Smoothness Induced by KL

Distillation). Let C be a finite candidate set and e𝑐 ∈ R𝑑 denote the embedding of item 𝑐 ∈ C. Given a reasoning state r ∈ R𝑑, define the induced predictive distribution as:

exp(r⊤e𝑐) 𝑐′∈C exp(r⊤e𝑐′)

𝑃(𝑐 | 𝐻) =

, 𝑐 ∈ C.

Let 𝑄 be any fixed teacher distribution supported on C. Then the KL distillation loss L(r) = 𝐷KL(𝑄 ∥ 𝑃(· | 𝐻)) is differentiable with respect to r, with gradient

∇rL(r) = E𝑃(·|𝐻) [e𝑐] − E𝑄 [e𝑐].

The proof of Proposition 2.1 is provided in Appendix B. Interpretation. The distribution 𝑃(· | 𝐻) defines a point on the probability simplex over C, and its embedding expectation E𝑃 [e𝑐] lies in the convex hull of candidate embeddings conv{e𝑐 : 𝑐 ∈ C}. The teacher expectation E𝑄 [e𝑐] lies in the same region. Proposition 2.1 shows that KL distillation induces a gradient flow that directly moves the prediction barycenter toward the teacher barycenter within this graph-restricted convex hull. As a result, latent reasoning states are encouraged to evolve such that their induced predictions remain confined to a graph-local embedding region defined by C. In ManCAR, C is a graph-conditioned candidate set derived from the most recent interactions, and 𝑄 is a scheduled teacher distribution supported on this set. Together, they impose a local graph smoothness prior on latent reasoning: each refinement step reduces uncertainty while remaining restricted to collaboratively reachable items. Progressive sharpening of the teacher distribution yields a stable coarse-to-fine trajectory on the simplex, mitigating latent drift during multi-step reasoning.

#### 2.5 Implementation of ManCAR Objective

Having established the theoretical validity of the variational training objective, we now turn to the implementation of ManCAR. In this subsection, we describe how the variational objective in Eq. 1 is instantiated in practice. Our implementation is designed to faithfully realize the graph-conditioned manifold constraint while remaining compatible with standard latent reasoning paradigms.

Item Interaction Graph. We construct the item interaction graph using a standard Swing-style item-to-item co-interaction algorithm that is widely adopted in industrial systems [38]. Each node corresponds to an item, and weighted edges encode collaborative strength measured by co-interaction frequency. As this graph construction follows established practice and is not a contribution of this work, we defer algorithmic details to Appendix D. Concretely, the resulting graph G associates each directed edge (𝑖 → 𝑗) with a weight 𝑤𝑖𝑗, indicating the strength of collaborative relevance of item 𝑗 with respect to item 𝑖.

Teacher Prior Construction. The variational objective in Equation (1) requires a teacher distribution 𝑞(𝑐 | 𝐼𝑛, G) defined over the candidate set C(𝑘). This distribution encodes prior knowledge about plausible intent prototypes that are collaboratively reachable from the user’s most recent interactions. In practice, we consider the following strategy to construct the teacher prior efficiently:

Rank-Based Distribution Mass Assignment (RDMA). We construct the teacher prior based on the relative ranking of candidates. The target item 𝑖∗ is always assigned a rank 0, while the remaining candidates in C(𝑘) are ranked in descending order of their graph edge weights 𝑤𝐼𝑛,𝑐 (from a recent interacted item in 𝐼𝑛 to its neighbor𝑐), receiving ranks [1, 2, . . . ]. The probability mass is then assigned using a softmax over negative ranks,

exp(−rank(𝑐)/𝛾) 𝑛∈C(𝑘) exp(−rank(𝑛)/𝛾)

, (2)

𝑞(𝑐 | 𝐼𝑛, G) =

where 𝛾 > 0 controls the sharpness of the teacher distribution. Using negative ranks ensures that higher-ranked (i.e., more strongly connected) items and the target receive the larger probability mass.

This strategy ensures that the teacher prior is strictly supported on the collaborative neighborhood while emphasizing the target item. By default, we assign zero probability mass to non-candidate items, resulting in a teacher distribution that lies on a sparse region of the item probability simplex with only a small number of active entries. Alternatively, a small amount of probability mass can be distributed over non-candidate items as a form of label smoothing, which we find does not materially affect the main conclusions. A dynamic scheduling is applied to the teacher prior distributions to guide latent reasoning progressively toward the target by adjusting the concentration of the teacher distribution across reasoning steps. As this scheduling mechanism is closely tied to adaptive test-time termination, we defer its detailed formulation to Sec. 2.6.

Training Loss. The complete training objective instantiates the ELBO derived in Eq. 1 and consists of a target prediction loss and a graph-conditioned manifold regularization loss, both applied at each reasoning step.

Latent Reasoning and Decoding. At reasoning step 𝑡′, the model produces a latent reasoning state r𝑡′ ∈ R𝑑, obtained by iteratively refining the initial state r1 = h𝑇−1 = 𝑓𝜃 (𝐻) through a shared reasoning module 𝑓𝜃 (·). Concretely, the refinement follows 𝑟𝑡′ = 𝑓𝜃 (𝐻;𝑟1:𝑡′−1). Then each latent reasoning state r𝑡′ is projected onto the item space to produce logits:

###### z𝑡′ = r𝑡⊤′E,

where E ∈ R𝑑×|I| denotes the item embedding matrix. The logits define an item probability distribution over I via a temperaturescaled softmax,

exp z𝑡′,𝑖/𝜏𝑡′ 𝑗∈I exp z𝑡′,𝑗/𝜏𝑡′

𝑝𝜃(𝑡′)(𝑖 | 𝐻) =

.

We use this same distribution to represent the intent distribution 𝑝𝜃(𝑡′)(𝑐 | 𝐻) by restricting 𝑐 to the candidate set C(𝑘). Target Prediction Loss. Following the ELBO in Eq. 1, the target prediction term at reasoning step 𝑡′ is given by

Lmain(𝑡′) = −E𝑐∼𝑞(·|𝐼𝑛,G) log𝑝𝜃(𝑡′)(𝑖∗ | 𝐻,𝑐) . In practice, explicitly marginalizing over𝑐 ∈ C(𝑘) at each reasoning step is costly and unnecessary. Instead, we adopt a standard conditioning strategy by exposing the entire candidate set C(𝑘) to the model as additional contexts (like context engineering). Concretely, we approximate the above expectation by:

Lmain(𝑡′) = −log𝑝𝜃(𝑡′)(𝑖∗ | 𝐻, C(𝑘)), (3) where C(𝑘) is injected as auxiliary input alongside the history 𝐻.

Graph-Conditioned Manifold Regularization. To enforce graphconditioned feasibility, the induced distribution at each step is regularized toward the teacher prior via

Lreg(𝑡′) = 𝐷KL 𝑞(𝑐 | 𝐼𝑛, G) ∥ 𝑝𝜃(𝑡′)(𝑐 | 𝐻) . (4) This term restricts latent reasoning trajectories to remain within the collaborative manifold defined by C(𝑘) and mitigates latent drift.

To reduce computation, the student distribution 𝑝𝜃(𝑡′)(𝑐 | 𝐻) can be obtained from the same forward pass used for target prediction Overall Objective. The complete training objective is as follows:

∑︁𝑇′

Lmain(𝑡′) + 𝜆Lreg(𝑡′) , (5)

L =

𝑡′=1

where 𝜆 controls the strength of graph-conditioned regularization. Appendix E provides the detailed training algorithm of ManCAR.

#### 2.6 Training Scheduling and Adaptive Test-Time Reasoning

We now introduce a dynamic scheduling mechanism that sharpens the teacher distribution over time and present a theoretical proposition regarding bounded error. This analysis not only guides the training schedule but also naturally motivates our convergencebased stopping criterion, enabling adaptive test-time reasoning. Appendix E provides the detailed adaptive reasoning algorithm.

Proposition 2.2 (Continuation tracking under contraction and bounded teacher drift). Fix a query 𝐻 and a finite candidate set C = C(𝑘). Let Δ(C) denote the item probability simplex over C. Consider distribution sequences {𝑝𝑡′}𝑡′≥1 ⊂ Δ(C) (student) and {𝑞𝑡′}𝑡′≥1 ⊂ Δ(C) (scheduled teacher), and let

∑︁

- 1

- 2

𝑑TV(𝑝,𝑞) :=

|𝑝(𝑐) − 𝑞(𝑐)|

𝑐∈C

be the total variation distance. We assume:

###### (1) Stepwise contraction toward the current teacher: There exists 𝜆 ∈ (0, 1) such that for all 𝑡′ ≥ 1,

𝑑TV(𝑝𝑡′+1,𝑞𝑡′) ≤ (1 − 𝜆)𝑑TV(𝑝𝑡′,𝑞𝑡′). (6)

###### (2) Bounded teacher drift (controlled schedule): There exists 𝛿 ≥ 0 such that for all 𝑡′ ≥ 1,

𝑑TV(𝑞𝑡′+1,𝑞𝑡′) ≤ 𝛿. (7) Then for all 𝑡′ ≥ 1,

𝛿 𝜆

𝑑TV(𝑝𝑡′,𝑞𝑡′) ≤ (1 − 𝜆)𝑡′−1 𝑑TV(𝑝1,𝑞1) +

. (8)

We provide the proof of Proposition 2.2 in Appendix C. Proposition 2.2 shows that, if each refinement step contracts the student toward the current teacher and the teacher distribution evolves smoothly, then the student distribution can track a progressively changing teacher with bounded error. This provides a formal motivation for using a coarse-to-fine teacher schedule during training.

The proposition is stated in total variation (TV) distance to leverage its metric properties. In ManCAR, refinement is trained via KL distillation rather than TV. To bridge this gap, we invoke Pinsker’s inequality, which guarantees that for any distributions 𝑝,𝑞 on C,

√︃

- 1

- 2𝐷KL(𝑞 ∥ 𝑝).

𝑑TV(𝑝,𝑞) ≤

Hence, minimizing the KL loss ensures a small student-teacher mismatch in TV, providing a conservative stability guarantee for the continuation tracking behavior described in Proposition 2.2.

Teacher Scheduling Strategies. We extend the teacher construction (Sec. 2.5) into a scheduled mechanism that generates a smoothly evolving sequence of teacher distributions across reasoning steps.

Adjustment for Strategy RDMA. We define the teacher as: 𝑞𝑡′(𝑐) ∝ exp − rank(𝑐)/𝛾𝑡′ , 𝛾𝑡′ =𝛾base · (𝑇′ − 𝑡′ + 1),

with 𝛾base ≥ 1. As 𝛾𝑡′ decreases linearly, the teacher distribution transitions smoothly from a diffuse graph-aware prior to a sharply peaked distribution centered on the target. Properly tuning 𝛾base and total steps 𝑇 yields a smoothly evolving teacher distribution with bounded drift across refinement steps (satisfying Assumption (2) in Proposition 2.2).

Connection to Adaptive Test-Time Reasoning. The continuation view provided by Proposition 2.2 directly motivates adaptive termination at test time. Since the student distribution 𝑝𝑡′ tracks the scheduled teacher with bounded error, convergence of successive student distributions indicates that further refinement yields diminishing returns. We therefore terminate reasoning early when the change between consecutive steps falls below a threshold, e.g.,

𝐷KL(𝑝𝑡′−1 ∥ 𝑝𝑡′) < 𝜀.

Scheduling the Main Prediction Loss. In addition to scheduling the teacher prior, we apply a step-dependent temperature schedule to the main target prediction loss to control the magnitude of distributional updates induced by target supervision at each reasoning step. Concretely, we use an exponential temperature schedule:

𝜏𝑡′ = 𝜏base · 𝑡′𝛼,

where 𝛼 > 1 controls 𝜏𝑡′’s increasing magnitude, which yields an increasing temperature sequence across reasoning steps with a flexible initial temperature base 𝜏base.

From the continuation perspective formalized in Proposition 2.2, supervising all refinement steps with an identical, sharply peaked target loss may induce overly large early updates that violate the bounded-drift and contraction conditions, potentially destabilizing manifold-constrained reasoning. By starting with a low effective temperature, early refinement steps are encouraged to make conservative progress (near the local neighborhood of recent user interactions), while remaining within the graph-consistent manifold.

This design is related in spirit to progressive refinement losses such as the PRL mechanism in ReaRec [31], but differs in directionality. Whereas ReaRec adopts a decreasing temperature schedule to accelerate early-stage convergence, our increasing-temperature design aligns with our theoretical analysis and supports stable multi-step reasoning and adaptive test-time termination.

Stabilizing Optimization via Latent State Norm Rescaling. In addition to scheduling-based control, we apply normalization to stabilize multi-step latent reasoning. After each refinement step, we rescale the latent reasoning state as:

h ∥h∥

· avg(E),

h ← 𝜙 ·

where avg(E) denotes the average norm of item embeddings, and 𝜙 is a learnable affine scaling parameter.

This operation aligns the scale of latent states with that of the item embedding space. This rescaling alleviates the burden on the Transformer to simultaneously accommodate heterogeneous modalities with mismatched norms between original input items and latent reasoning states. By keeping latent states on a scale comparable to item embeddings, this normalization mitigates empirical norm growth with (recursive) depth [27], improves stability in longhorizon reasoning, and complements the manifold-constrained and continuation-based design of ManCAR. In particular, it helps maintain a well-conditioned softmax geometry during refinement, which empirically supports the stepwise contraction behavior assumed in

- Proposition 2.2 (facilitating Assumption (1)).
- 3 Experiments

Our empirical study is guided by the following research questions: (1) Overall performance. How does ManCAR perform compared with strong sequential recommendation baselines across standard benchmarks? (2) Effect of teacher scheduling and adaptive reasoning. How do teacher scheduling and adaptive termination shape step-wise refinement behavior and enable near-ceiling performance during inference? (3) Ablation analysis. What is the impact of individual components in ManCAR? (4) Parameter sensitivity. How sensitive is ManCAR to key hyperparameters? (5) KL-Based Halting Analysis. How does the KL divergence between steps reflect the stability of the reasoning trajectory? (6) Attention Visualization Analysis. What do attention patterns reveal about the information flow within the manifold-constrained design?

- 3.1 Experimental Setup

Table 1: Dataset statistics.

#Avg. Inter./User

#Avg. Inter./Item

Dataset #Users #Items #Interactions

CDs 35,238 87,969 943,399 26.77 10.72 Video 54,001 22,735 562,193 10.41 24.73 Office 118,617 64,679 1,116,568 9.41 17.3 Arts 112,401 77,596 1,180,363 10.50 15.21 Music 15,685 21,171 221,980 14.15 10.49 Toys 244,231 134,676 2,515,203 10.30 18.68 Grocery 246,304 119,860 2,788,430 11.32 23.26

- 3.1.1 Datasets and Preprocess. We evaluate ManCAR on seven subcategory datasets from the Amazon 2023 Reviews corpus [12]: CDs & Vinyl (CDs), Video & Games (Video), Office Products (Office), Arts, Crafts & Sewing (Arts), Grocery & Gourmet Food (Grocery), Musical Instruments (Music), and Toys & Games (Toys). Tab. 1 provides data statistics. Following prior work [20, 30, 31], user–item interactions with ratings above 3 are treated as positive feedback.

To improve data quality, we remove users with fewer than 10 interactions in CDs and fewer than 5 interactions in the remaining datasets. We adopt the official absolute-timestamp split provided by the corpus.1 Consistent with previous studies [20, 30, 31], we truncate each user’s interaction history to a maximum length of 50.

- 3.1.2 Evaluation Metrics. To evaluate the performance of our proposed model and the baselines, we employ two widely-used metrics: Recall@𝐾 and Normalized Discounted Cumulative Gain (NDCG@𝐾), with 𝐾 ∈ {5, 10}. Specifically, Recall@𝐾 measures the model’s ability to include the ground-truth item within the top-𝐾 recommendation list, reflecting its retrieval coverage. NDCG@𝐾 further assesses the ranking quality by assigning higher weights to items at higher positions, thereby rewarding models that prioritize the correct item in more prominent ranks.
- 3.1.3 Baselines. We compare ManCAR with representative stateof-the-art baselines spanning different modeling paradigms. Specifically, we include:

- (1) SASRec [15]: Utilizing a unidirectional Transformer encoder,

SASRec represents users by the final item in their interaction sequence.

- (2) BERT4Rec [26]: adopts a bidirectional approach inspired by

BERT, training the model to reconstruct masked items within the sequence.

- (3) ContextBERT4Rec: extends BERT4Rec by using the same con-

text engineering as ManCAR.

- (4) ReaRec-ERL [31]: As a pioneer in latent space reasoning, this

model treats the reasoning process as a collective trajectory. Instead of relying on a single state, it synthesizes the implicit information from all autoregressive steps using a mean pooling mechanism to form a comprehensive user representation.

- (5) ReaRec-PRL [31]: In contrast to ERL, this variant emphasizes

iterative optimization. It leverages contrastive learning with noise injection to progressively distill the latent representation, discarding intermediate states to rely solely on the converged output of the final reasoning step.

1https://amazon-reviews-2023.github.io/data_processing/5core.html

- (6) LARES [20]: This framework introduces pre-blocks and core-

blocks. To maximize reasoning fidelity, it adopts a hybrid training pipeline that sequentially applies self-supervised pre-training followed by reinforcement learning-based fine-tuning.

- (7) PLR [30]: A width-scaled (parallel) latent reasoning frame-

work for sequential recommendation that launches multiple parallel reasoning streams via learnable trigger tokens, enforces interstream diversity with global reasoning regularization, and adaptively fuses the stream outputs (mixture-of-streams) to improve next-item prediction.

- 3.1.4 Implementation. We conduct all experiments on eight NVIDIA 3090 GPUs. To ensure a fair comparison, we set the embedding size and batch size for all methods to 256 and 512, respectively. We optimize all models using the Adam optimizer with a learning rate of 0.001. For baselines without open-source code, we conducted our own implementation. For those with available source code, we utilized the official implementations. All baselines were tuned via grid search based on the hyperparameters specified in their original papers, and the optimal results are reported. To mitigate overfitting, we employ early stopping, terminating training if NDCG@10 on the validation set shows no improvement for 5 consecutive epochs. Following prior work [31], we adopt a pre-norm Transformer backbone. It consists of two Transformer layers, each with two-head multi-head self-attention and GeLU activation.
- 3.2 Overall Performance (Tab. 2).

ManCAR outperforms all baselines. ManCAR achieves the best performance across all datasets and evaluation metrics, demonstrating consistent improvements in both ranking and retrieval quality. Compared with the second-best method on each dataset, ManCAR delivers up to a 46.88% relative improvement on certain metrics. Notably, the gains are more pronounced on NDCG, indicating that ManCAR is particularly effective at ranking relevant items higher, which reflects a stronger ability to capture and refine user intent.

ContextBERT4Rec outperforms BERT4Rec, highlighting the benefit of graph-induced context. ContextBERT4Rec augments the input sequence with the same graph-conditioned candidate set used by ManCAR, enabling the model to leverage collaborative signals beyond the independent raw user interaction sequence. Its consistent improvement over BERT4Rec suggests that incorporating graph-induced context serves as an effective form of context engineering for sequential recommendation.

Explicit latent reasoning consistently improves sequential recommendation. ContextBERT4Rec represents the strongest non-reasoning baseline by incorporating graph-conditioned context into the input. Across all datasets, ManCAR achieves notable gains over ContextBERT4Rec, demonstrating that explicit multistep reasoning provides additional modeling capacity beyond contextual encoding alone. More broadly, all reasoning-based methods (ManCAR, ERL, PRL, PLR, and LARES) outperform non-reasoning baselines like SASRec and BERT4Rec, suggesting that iterative refinement of intermediate hypotheses enables more effective uncertainty resolution and user intent modeling, particularly in sparse or challenging settings.

###### Office

Toys

0.013

0.0095

0.012

0.0090

+5.8% +7.0%

+12.0%

+7.4%

0.0085

0.011

###### NDCG@5

0.0080

0.010

0.0075

+36.5%

+42.1%

0.009

0.0070

0.0065

0.008

0.0060

ContextBert4RecManCAR-last-step ManCARManCAR-ceiling

ContextBert4RecManCAR-last-step ManCARManCAR-ceiling

###### Figure 3: Performance ceiling analysis on Office and Toys.

ManCAR consistently outperforms existing latent reasoning approaches. Across all datasets, ManCAR achieves consistent gains over prior reasoning-based methods such as ERL, PRL, PLR, and LARES. While these methods introduce latent refinement or progressive reasoning, they typically lack explicit constraints on how reasoning trajectories evolve. In contrast, ManCAR integrates graph-conditioned manifolds, scheduled teacher supervision, and adaptive test-time control, which together provide a more structured and stable reasoning process. By explicitly controlling the feasible manifold region and stepwise dynamics of latent refinement, ManCAR is better able to exploit collaborative signals and avoid unstable or suboptimal reasoning paths, leading to better performance with varying data sparsity and sequence lengths.

Performance gains increase with higher interaction density. ManCAR exhibits larger performance margins over the second-best baseline on datasets with higher interaction density (average interactions per item). For instance, improvements are more pronounced on Video and Toys than on Music and Arts. This trend suggests that ManCAR benefits from reduced sparsity, where multi-step reasoning can more effectively refine user intent by leveraging a more reliable item interaction graph and richer collaborative signals. When interactions are sparse, graph edge connection become noisier, which limits the advantage of graph-conditioned reasoning over strong baselines. Improving robustness under limited preference evidences (cold start) is left for future work.

#### 3.3 In-Depth Analysis in Adaptive Reasoning

We analyze ManCAR’s adaptive reasoning ability from two angles. Data-Aware Train-Test Compute Allocation. Tab. 3 summarizes the reasoning-step configurations at which different reasoning methods achieve their best performance. Baselines adopt identical and shallow reasoning depths at both stages, typically limited to 2-3 steps, regardless of data characteristics. This indicates that reasoning depth is treated as a static architectural hyperparameter.

In contrast, ManCAR exhibits data-aware and asymmetric traintest computation. The optimal number of training and inference steps varies substantially across datasets, reflecting differences in data sparsity and sequence complexity. On datasets with complex interaction patterns, such as CDs and Toys, ManCAR employs deeper reasoning and achieves significantly larger performance gains, while prior methods are unable to adapt beyond 3 steps.

###### Table 2: Performance comparison on seven datasets. The best results are in bold and the second best results are underlined.

Dataset Metric SASRec BERT4Rec ContextBERT4Rec ERL PRL PLR LARES ManCAR Uplift

NDCG@5 0.0098 0.0110 0.0148 0.0099 0.0122 0.0131 0.0159 0.0198 24.53 % NDCG@10 0.0132 0.0130 0.0182 0.0129 0.0149 0.0168 0.0192 0.0282 46.88 % Recall@5 0.0181 0.0207 0.0243 0.0190 0.0227 0.0249 0.0235 0.0346 38.96 % Recall@10 0.0286 0.0269 0.0391 0.0283 0.0315 0.0363 0.0351 0.0516 31.97 %

CDs

NDCG@5 0.0083 0.0070 0.0119 0.0149 0.0165 0.0159 0.0162 0.0217 31.52 % NDCG@10 0.0104 0.0091 0.0163 0.0187 0.0215 0.0210 0.0219 0.0275 25.57 % Recall@5 0.0164 0.0131 0.0190 0.0217 0.0253 0.0312 0.0277 0.0339 8.65 % Recall@10 0.0226 0.0196 0.0330 0.0336 0.0410 0.0467 0.0455 0.0521 11.56 %

Video

NDCG@5 0.0063 0.0061 0.0076 0.0079 0.0071 0.0082 0.0090 0.0108 20.00 % NDCG@10 0.0077 0.0078 0.0093 0.0107 0.0096 0.0109 0.0115 0.0133 15.65 % Recall@5 0.0113 0.0109 0.0129 0.0144 0.0127 0.0139 0.0161 0.0174 8.07 % Recall@10 0.0157 0.0160 0.0184 0.0229 0.0205 0.0224 0.0240 0.0250 4.17 %

Office

NDCG@5 0.0019 0.0014 0.0044 0.0052 0.0072 0.0068 0.0077 0.0087 12.99 % NDCG@10 0.0026 0.0023 0.0059 0.0082 0.0097 0.0092 0.0101 0.0114 12.87 % Recall@5 0.0037 0.0028 0.0082 0.0086 0.0118 0.0106 0.0119 0.0141 18.49 % Recall@10 0.0057 0.0055 0.0129 0.0177 0.0196 0.0181 0.0196 0.0225 14.80 %

Arts

NDCG@5 0.0017 0.0014 0.0049 0.0066 0.009 0.0058 0.0063 0.0097 7.78 % NDCG@10 0.0027 0.002 0.0059 0.0095 0.0117 0.0084 0.0092 0.0120 2.56 % Recall@5 0.0033 0.0026 0.0086 0.0124 0.0138 0.0111 0.0117 0.0147 6.52 % Recall@10 0.0061 0.0046 0.0118 0.0215 0.0225 0.0191 0.0205 0.0217 -3.56 %

Music

NDCG@5 0.0037 0.0045 0.0051 0.0054 0.0046 0.0052 0.0061 0.0086 40.98 % NDCG@10 0.0048 0.0059 0.0063 0.0073 0.0064 0.0068 0.0082 0.0108 31.71 % Recall@5 0.0072 0.0086 0.0089 0.0102 0.0082 0.0096 0.0115 0.0136 18.26 % Recall@10 0.0104 0.0127 0.0126 0.0160 0.0136 0.0148 0.0180 0.0203 12.78 %

Toys

NDCG@5 0.0028 0.0026 0.0044 0.0062 0.0072 0.0078 0.0063 0.0095 21.79 % NDCG@10 0.0038 0.0035 0.0059 0.0087 0.0098 0.0105 0.0088 0.0118 12.38 % Recall@5 0.0055 0.0085 0.0079 0.0102 0.0118 0.0124 0.0119 0.0149 20.16 % Recall@10 0.0085 0.0078 0.0126 0.0180 0.0197 0.0218 0.0197 0.0220 0.92 %

Grocery

###### Table 3: Best performing step setting of reasoning-based methods on four datasets. See Appendix F.1 for full results.

|Dataset Reason step<br><br>|ERL PRL PLR LARES|ManCAR|
|---|---|---|
|CDs<br><br>Train step Infer step|2 2 3 4 2 2 3 4<br><br>|5 1.84|
|Arts<br><br>Train step Infer step<br><br>|2 2 1 4 2 2 1 4<br><br>|1 1|
|Toys<br><br>Train step Infer step|1 2 1 4 1 2 1 4<br><br>|4 3.58|
|Grocery<br><br>Train step Infer step<br><br>|2 2 3 4 2 2 3 4<br><br>|2 1.74|

Conversely, on simpler datasets such as Arts and Grocery, ManCAR stops early at inference, avoiding unnecessary computation while still outperforming baselines which over-allocate reasoning steps.

Overall, these results indicate that ManCAR performs genuine iterative refinement with adaptive inference depth, enabling an effective balance between reasoning expressiveness and computational efficiency across diverse data properties.

Near-Optimal Reasoning through the Lens of Ceiling Performance Analysis. Fig. 3 showcases a step-wise performance analysis of ManCAR. We report three variants: (i) the prediction from the final reasoning step (ManCAR-last-step), (ii) adaptive halting based on convergence (ManCAR), and (iii) an oracle ceiling

that selects the best-performing step per sample using groundtruth labels (ManCAR-ceiling). These results are compared with ContextBERT4Rec, a non-reasoning variant of ManCAR.

When ManCAR is forced to use a fixed (symmetric) number of reasoning steps, performance degrades relative to adaptive halting, though it remains closer to the ceiling than the non-reasoning variant. In contrast, adaptive reasoning consistently outperforms the symmetric setting and achieves performance that is very close to the oracle ceiling, indicating effective reasoning and termination.

In contrast, prior reasoning-based methods such as PLR [30] and ReaRec [31] (including PRL and ERL) also report ceiling performance, but exhibit a substantially larger gap between their actual inference performance and the ceiling. This highlights ManCAR’s ability to translate iterative refinement into near-optimal test-time behavior, rather than relying on a fixed reasoning budget.

#### 3.4 Ablation Study (Tab. 4).

Graph-driven manifold constraint (w/o teacher prior) results in the largest performance drop among ManCAR variants, though it still outperforms ContextBERT4Rec. This indicates that graph context alone provides limited gains, while the absence of teacher guidance makes target-driven reasoning susceptible to latent drift.

###### Table 4: Ablation results on CDs and Video.

CDs Video N@10 R@10 N@10 R@10

method

ManCAR 0.0282 0.0516 0.0275 0.0521 w/o Teacher Prior 0.0212 0.0436 0.0231 0.0445 w/o Context 0.0234 0.0469 0.0265 0.0497 w/o Norm Rescale 0.0277 0.0513 0.0271 0.0515 w/o Schedule 0.0261 0.0496 0.0251 0.0491 w/ Decrease Schedule 0.0273 0.0490 0.0250 0.0518

Context engineering (w/o context). Removing candidate-set context injection causes a clear performance drop, though this variant still outperforms ReaRec-style baselines. This suggests that teacher guidance alone can partially steer reasoning, while injecting graph-conditioned candidates as auxiliary context further narrows the predictive search space and improves target localization.

Latent state norm rescaling (w/o rescaling). Removing this module causes a consistent performance drop, highlighting its role in aligning latent states with item embeddings. This normalization mitigates empirical norm growth and improves numerical stability, supporting stable stepwise refinement in multi-step reasoning.

Loss scheduling (w/o schedule or decreasing schedule). Removing the schedule or adopting a decreasing one in target prediction loss leads to clear performance degradation. This agrees with our analysis (Proposition 2.2) that conservative early updates helps to preserve manifold-walking stability and avoids premature convergence associated with decreasing schedules.

#### 3.5 Parameter Sensitivity

Fig. 4 and 5 present the parameter sensitivity of ManCAR with respect to #context items (the number of graph neighbors used to construct the candidate set), #training-steps (the step configuration used at training time), 𝜆 (balancing target prediction and KL regularization), 𝛾base (controlling the teacher sharpness schedule), and 𝜏𝑏𝑎𝑠𝑒 (controlling the temperature schedule for target prediction).

Among the major hyperparameters, ManCAR is most sensitive to the number of graph neighbors (construct the candidate set) and training-time steps, since either noise-injection or insufficient support and shaping of the manifold may degrade the performance. In contrast, the model is relatively insensitive to the choice of 𝜆 (balancing target prediction and KL regularization), 𝛾base (controlling the teacher sharpness schedule), and𝜏1 (controlling the temperature schedule for target prediction). Across these parameters, performance exhibits smooth and well-behaved trends, allowing optimal values to be reliably identified via simple grid search.

#### 3.6 KL-Based Halting Analysis

We report the KL divergence between consecutive reasoning steps on two datasets, CDs and Video. For each test batch, we compute the average KL divergence across samples and then report the mean and variance across batches. As shown in Fig. 6, after sufficient training, the KL divergence between adjacent reasoning steps decreases sharply, indicating stable convergence of the reasoning trajectory as expected.

NDCG@10 Recall@10

Video

###### CDs

| | | | | | | |
|---|---|---|---|---|---|---|
| |(|b)| | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.052

0.028

(a)

0.051

0.027

0.027

0.050

0.050

0.026

0.026

0.048

0.049

0.025

0.025

0.024

14 21 28 35

14 21 28 35

#context items

#context items

Video

###### CDs

0.0275

0.0523

0.0515

(c)

0.0280 (d)

0.0270

0.0522

0.0510

0.0265

0.0275

0.0505

0.0260

0.0521

0.0270

0.0255

0.0500

0.0520

0.2 1.0 2.0

0.2 1.0 2.0

λ

λ

Video

CDs

0.0275

(e)

(f)

0.052

0.0270

0.0285

0.0515

0.0265

0.051

0.0280

0.0510

0.0260

0.050

0.0275

0.0505

0.0255

0.1 0.5 1.0 5.0 10.0

0.1 0.5 1.0 5.0 10.0

τbase

τbase

Video

CDs

0.052

(g)

- 0.026

- 0.027

- 0.028 (h)

0.026

0.050

0.050

0.024

0.048

0.048

0.5 1.0 2.0 5.0

0.5 1.0 2.0 5.0

γbase

γbase

- Figure 4: Sensitivity analysis on Video and CDs. (a) and (b): NDCG@10 and Recall@10 w.r.t. #context items; (c) and (d): NDCG@10 and Recall@10 w.r.t. regularization loss weight 𝜆; (e) and (f): NDCG@10 and Recall@10 w.r.t. temperature 𝜏base; (g) and (h): NDCG@10 and Recall@10 w.r.t. 𝛾base.

1 2 3 4 5

Training Reason Step T0

0.000

0.005

0.010

0.015

0.020

0.025

NDCG@10

CDs

1 2 3 4 5

Training Reason Step T0

0.0000

0.0025

0.0050

0.0075

0.0100

0.0125

Office

ManCAR-last-step

| |
|---|

ManCAR-adaptive

- Figure 5: NDCG@10 w.r.t. reason step 𝑇′ on CDs and Office.

#### 3.7 Attention Visualization Analysis

The attention heatmaps in Fig. 7 (two layers with two heads, with tokens partitioned into Context C, Interaction History H, and Reasoning Steps R) reveal a consistent routing pattern that aligns with ManCAR’s manifold-constrained latent reasoning design. Across all heads, we observe a prominent concentration of attention mass from reasoning tokens toward a small subset of context positions (i.e., strong vertical bands within the 𝐶 region), while the H region is comparatively diffuse and weaker. This indicates that the

###### CDs

Video

0.35

0.35

0.30

0.30

0.25

0.25

D(t-1,t)KL

D(t-1,t)KL

0.20

0.20

0.15

0.15

0.10

0.10

0.05

0.05

0.00

0.00

1 2 3 4 5

1 2 3 4 5

Reason step t0

Reason step t0

###### Figure 6: KL divergence between two adjacent step 𝑡′ − 1 and 𝑡 w.r.t. inference steps 𝑡′.

Layer-1,Head-1

Layer-1,Head-2

| | | |
|---|---|---|
|[Figure 3]| | |
| | | |

| | | |
|---|---|---|
| |[Figure 4]| |
| | | |

C

C

H

H

[Figure 5]

- 0

- 1

- 2

- 3

- 4

AttentionScore

R

R

C H R

C H R

Layer-2,Head-1

Layer-2,Head-2

| | | |
|---|---|---|
| |[Figure 6]| |
| | | |

| | | |
|---|---|---|
|[Figure 7]| | |
| | | |

C

C

H

H

R

R

C H R

C H R

###### Figure 7: Attention Analysis of ManCAR on CDs. Attention scores are averaged over 1024 randomly sampled user histories from the test set.

intermediate reasoning states do not evolve in a free-form manner; instead, they repeatedly query the injected candidate context C(𝑘) during refinement.

Moreover, the deeper layer exhibits sharper and more structured attention: the 𝑅 → 𝐶 concentration becomes stronger, and we also see increased self-referential aggregation near the 𝑅 boundary (visible as emphasis close to the rightmost columns / bottom-right region). This suggests that later layers increasingly perform interstep consolidation, integrating previous reasoning states while still grounding each update in the graph-conditioned candidate set.

Additionally, recent user interactions—particularly the latest action—receive consistently larger attention scores, reflecting the recency bias commonly observed in practical recommender systems. Together, this indicates that ManCAR is building the data channel: Recent Action → Graph-Anchors (neighbors) → Reasoning States to achieve adaptive, stable, and constrained refinement within the local intent manifold.

#### 3.8 Additional Analyses

For a comprehensive breakdown of the reasoning steps used during training and inference across all seven datasets, please refer to Appendix F.1. Additionally, we provide a rigorous theoretical derivation of the computational complexity (FLOPs) for ManCAR and other baseline models in Appendix F.2.

4 Related Work

#### 4.1 Sequential Recommendation

As a coreparadigm inrecommendation, sequential recommendation captures user preferences to forecast the next item of interest.

Non-LLM-Based sequential recommendation evolves from sequential pattern mining [41] and Markov chains [10, 11] to recent deep learning approaches [15, 26]. Detailed surveys on non-LLMBased sequential recommendation are available in [6, 32].

Recently, the emergence of LLMs has greatly affected the field of sequential recommendation, diverging into two paradigms [13]: (1) LLM-Augmented sequential recommendation uses LLMs as feature extractors. LLMSeq [9] and SAID [13] utilize LLM-derived embeddings for initialization and semantic alignment. Meanwhile, LRD [37] and SERALM [24] leverage language knowledge to discover latent relations and refine generation via feedback from IDbased recommenders. (2) LLM-Centric sequential recommendation employs the LLM as the predictor. Methods range from processing item “sentences” (RecFormer [18]) and ID sequences (E4SRec [19]) to managing long sequences via summarization (LLM-TRSR [47]). Other works enhance reasoning through intent-driven prompting (LLM4ISR [28]) and self-reflection agents (Re2LLM [34]).

Besides, there is a burgeoning sequential recommendation paradigm called generative sequential recommendation [23, 29, 33, 46] that replaces pre-fixed item IDs with identifiers constructed from generated tokens. By synthesizing tokens, these methods better leverage content to encode item semantics directly into the ID structure. However, this direction remains under-explored due to optimization challenges, such as the difficulty of distinguishing similar items with identical token sequences [48].

#### 4.2 Reasoning-Enhanced Recommendation

Reasoning-enhanced recommendation augments sequential recommendation with deliberative capabilities. It can be categorized into Explicit Reasoning (using visible, text-based chains) and Latent Reasoning (employing implicit, internal computation) to enhance recommendation accuracy.

Explicit Reasoning-Enhanced Recommendation. Explicit reasoning approaches leverage the generative capabilities of LLMs to articulate the decision-making process through interpretable text or symbolic chains. R2ec [44] introduces a unified dual-head architecture that simultaneously generates reasoning chains and predicts items. This design significantly reduces inference latency. ReasoningRec [1] bridges recommendations and explanations, and it uses CoT prompting to distill a LLM’s synthetic reasoning into a smaller model. Reason4Rec [7] formulates the deliberative recommendation task that incorporates explicit reasoning about user preferences as an alignment goal and enhances model’s reasoning capabilities utilizing verbalized user feedback in a step-wise manner.

Exp3rt [16] distills reasoning capabilities into a student LLM via a three-step process: preference extraction, profile construction, and prediction. It effectively utilizes rich review data for personalized recommendation. OneRec-Think [21] introduces a “Think-Ahead” architecture that seamlessly integrates dialogue, reasoning, and personalized recommendation. RecGPT [42] employs a Hierarchical Multi-Agent System for agentic intent reasoning and hybrid representation for efficiency, thereby solving the scalability issues of its predecessor, yet the complex multi-agent coordination introduces new challenges in system stability and debugging.

Latent Reasoning-Enhanced Recommendation. Inspired by latent reasoning for LLMs [8], recent sequential recommendation models have adopted latent reasoning to perform multi-step deliberation before prediction, without requiring explicit CoT data. ReaRec [31] pioneers inference-time computing by autoregressively feeding the last hidden state back into the encoder to enhance performance. OnePiece [3] applies latent reasoning to industrial retrieval and ranking by integrating context engineering with blockwise latent reasoning to progressively refine user intent. LARES [20] employs depth-recurrent latent reasoning that leverages all the input tokens to perform multi-step reasoning. PLR [30] introduces a width-level scaling paradigm that explores diverse reasoning paths simultaneously via parallel streams to alleviate diminishing returns as reasoning depth increases.

#### 5 Conclusion

We proposed ManCAR, a manifold-constrained latent reasoning framework for sequential recommendation. By restricting latent refinement to a graph-locality-induced manifold and guiding it with progressive teacher supervision towards the target item, ManCAR enables stable and structured multi-step reasoning. A continuationbased analysis motivates both the teacher scheduling strategy and adaptive test-time termination. Extensive experiments on seven public datasets demonstrate that ManCAR consistently outperforms strong sequential and reasoning-based baselines, yielding substantial improvements in retrieval and ranking quality. These results highlight the importance of explicit constraints over latent reasoning with concrete collaborative signals, and position ManCAR as a principled approach for controllable reasoning in sequential recommendation.

#### References

- [1] Millennium Bismay, Xiangjue Dong, and James Caverlee. 2025. ReasoningRec: Bridging Personalized Recommendations and Human-Interpretable Explanations through LLM Reasoning. In NAACL (Findings). 8132–8148.
- [2] Jianxin Chang, Chen Gao, Yu Zheng, Yiqun Hui, Yanan Niu, Yang Song, Depeng Jin, and Yong Li. 2021. Sequential Recommendation with Graph Neural Networks. In SIGIR. 378–387.
- [3] Sunhao Dai, Jiakai Tang, Jiahua Wu, Kun Wang, Yuxuan Zhu, Bingjun Chen, Bangyang Hong, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Wenjie Wang, Xu Chen, Jun Xu, and See-Kiong Ng. 2025. OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System. arXiv Preprint

(2025). https://arxiv.org/abs/2509.18091

- [4] Yashar Deldjoo, Zhankui He, Julian J. McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys). In KDD. 6448–6458.
- [5] Jingcheng Deng, Liang Pang, Zihao Wei, Shicheng Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng. 2025. Latent Reasoning in LLMs as a Vocabulary-Space Superposition. arXiv Preprint (2025). https://arxiv.org/abs/ 2510.15522

- [6] Hui Fang, Danning Zhang, Yiheng Shu, and Guibing Guo. 2020. Deep Learning for Sequential Recommendation: Algorithms, Influential Factors, and Evaluations. ACM Trans. Inf. Syst. 39, 1 (2020), 10:1–10:42.
- [7] Yi Fang, Wenjie Wang, Yang Zhang, Fengbin Zhu, Qifan Wang, Fuli Feng, and Xiangnan He. 2025. Reason4Rec: Large Language Models for Recommendation with Deliberative User Preference Alignment. arXiv Preprint (2025). https: //arxiv.org/abs/2502.02061
- [8] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training Large Language Models to Reason in a Continuous Latent Space. arXiv Preprint (2024). https://arxiv.org/abs/2412.06769
- [9] Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, and Marios Fragkoulis. 2023. Leveraging Large Language Models for Sequential Recommendation. In RecSys. 1096–1102.
- [10] Ruining He, Chen Fang, Zhaowen Wang, and Julian J. McAuley. 2016. Vista: A Visually, Socially, and Temporally-aware Model for Artistic Recommendation. In RecSys. 309–316.
- [11] Ruining He and Julian J. McAuley. 2016. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. In ICDM. 191–200.
- [12] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian J. McAuley.

2024. Bridging Language and Items for Retrieval and Recommendation. arXiv Preprint (2024). https://arxiv.org/abs/2403.03952

- [13] Jun Hu, Wenwen Xia, Xiaolu Zhang, Chilin Fu, Weichang Wu, Zhaoxin Huan, Ang Li, Zuoli Tang, and Jun Zhou. 2024. Enhancing Sequential Recommendation via LLM-based Semantic Embedding Learning. In WWW. 103–111.
- [14] Wei Ju, Zheng Fang, Yiyang Gu, Zequn Liu, Qingqing Long, Ziyue Qiao, Yifang Qin, Jianhao Shen, Fang Sun, Zhiping Xiao, Junwei Yang, Jingyang Yuan, Yusheng Zhao, Yifan Wang, Xiao Luo, and Ming Zhang. 2024. A Comprehensive Survey on Deep Graph Representation Learning. Neural Networks 173 (2024), 106207.
- [15] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In ICDM. 197–206.
- [16] Jieyong Kim, Hyunseo Kim, Hyunjin Cho, SeongKu Kang, Buru Chang, Jinyoung Yeo, and Dongha Lee. 2025. Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation. In SIGIR. 1697–1706.
- [17] Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, and Rex Ying. 2025. Implicit Reasoning in Large Language Models: A Comprehensive Survey. arXiv Preprint (2025). https://arxiv.org/abs/2509.02350
- [18] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian J. McAuley. 2023. Text Is All You Need: Learning Language Representations for Sequential Recommendation. In KDD. 1258–1267.
- [19] Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, and Chunxiao Xing. 2023. E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation. arXiv Preprint (2023). https://arxiv.org/ abs/2312.02443
- [20] Enze Liu, Bowen Zheng, Xiaolei Wang, Wayne Xin Zhao, Jinpeng Wang, Sheng Chen, and Ji-Rong Wen. 2025. LARES: Latent Reasoning for Sequential Recommendation. arXiv Preprint (2025). https://arxiv.org/abs/2505.16865
- [21] Zhanyu Liu, Shiyao Wang, Xingmei Wang, Rongzhou Zhang, Jiaxin Deng, Honghui Bao, Jinghao Zhang, Wuchao Li, Pengfei Zheng, Xiangyu Wu, Yifei Hu, Qigen Hu, Xinchen Luo, Lejian Ren, Zixing Zhang, Qianqian Wang, Kuo Cai, Yunfan Wu, Hongtao Cheng, Zexuan Cheng, Lu Ren, Huanjie Wang, Yi Su, Ruiming Tang, Kun Gai, and Guorui Zhou. 2025. OneRec-Think: In-Text Reasoning for Generative Recommendation. arXiv Preprint (2025). https://arxiv.org/abs/2510.11639
- [22] Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, and Shenghua Liu. 2025. arXiv Preprint (2025). https://arxiv.org/ abs/2507.13334
- [23] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Mahesh Sathiamoorthy. 2023. Recommender Systems with Generative Retrieval. In NeurIPS. 10299–10315.
- [24] Yankun Ren, Zhongde Chen, Xinxing Yang, Longfei Li, Cong Jiang, Lei Cheng, Bo Zhang, Linjian Mo, and Jun Zhou. 2024. Enhancing Sequential Recommenders with Augmented Knowledge from Aligned Large Language Models. In SIGIR. 345–354.
- [25] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He.

2025. CODI: Compressing Chain-of-Thought into Continuous Space via SelfDistillation. In EMNLP. 677–693.

- [26] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang.

2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In CIKM. 1441–1450.

- [27] Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, and Shiwei Liu. 2025. The Curse of Depth in Large Language Models. https://arxiv.org/abs/ 2502.05795
- [28] Zhu Sun, Hongyang Liu, Xinghua Qu, Kaidong Feng, Yan Wang, and Yew Soon Ong. 2024. Large Language Models for Intent-Driven Session Recommendations. In SIGIR. 324–334.
- [29] Juntao Tan, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Zelong Li, and Yongfeng Zhang. 2024. IDGenRec: LLM-RecSys Alignment with Textual ID Learning. In

- SIGIR. 355–364.
- [30] Jiakai Tang, Xu Chen, Wen Chen, Jian Wu, Yuning Jiang, and Bo Zheng. 2026. Parallel Latent Reasoning for Sequential Recommendation. arXiv Preprint (2026). https://arxiv.org/abs/2601.03153
- [31] Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Wu Jian, and Yuning Jiang. 2025. Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation. arXiv Preprint (2025). https://arxiv.org/ abs/2503.22675
- [32] Shoujin Wang, Liang Hu, Yan Wang, Longbing Cao, Quan Z. Sheng, and Mehmet A. Orgun. 2019. Sequential Recommender Systems: Challenges, Progress and Prospects. In IJCAI. 6332–6338.
- [33] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, SeeKiong Ng, and Tat-Seng Chua. 2024. Learnable Item Tokenization for Generative Recommendation. In CIKM. 2400–2409.
- [34] Ziyan Wang, Yingpeng Du, Zhu Sun, Haoyan Chua, Kaidong Feng, Wenya Wang, and Jie Zhang. 2025. Re2LLM: Reflective Reinforcement Large Language Model for Session-based Recommendation. (2025), 12827–12835.
- [35] Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. LLMRec: Large Language Models with Graph Augmentation for Recommendation. In WSDM. 806–815.
- [36] Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan.

2019. Session-Based Recommendation with Graph Neural Networks. In AAAI. 346–353.

- [37] Shenghao Yang, Weizhi Ma, Peijie Sun, Qingyao Ai, Yiqun Liu, Mingchen Cai, and Min Zhang. 2024. Sequential Recommendation with Latent Relations based on Large Language Model. In SIGIR. 335–344.
- [38] Xiaoyong Yang, Yadong Zhu, Yi Zhang, Xiaobo Wang, and Quan Yuan. 2020. Large Scale Product Graph Construction for Recommendation in E-commerce. arXiv Preprint (2020). https://arxiv.org/abs/2010.05525
- [39] Yuhao Yang, Chao Huang, Lianghao Xia, and Chenliang Li. 2022. Knowledge Graph Contrastive Learning for Recommendation. In SIGIR. 1434–1443.
- [40] Yonghui Yang, Le Wu, Zihan Wang, Zhuangzhuang He, Richang Hong, and Meng Wang. 2024. Graph Bottlenecked Social Recommendation. In KDD. 3853–3862.
- [41] Ghim-Eng Yap, Xiaoli Li, and Philip S. Yu. 2012. Effective Next-Items Recommendation via Personalized Sequential Pattern Mining. In DASFAA, Vol. 7239. 48–64.
- [42] Chao Yi, Dian Chen, Gaoyang Guo, Jiakai Tang, Jian Wu, Jing Yu, Mao Zhang, Wen Chen, Wenjun Yang, Yujie Luo, Yuning Jiang, Zhujin Gao, Bo Zheng, Binbin Cao, Changfa Wu, Dixuan Wang, Han Wu, Haoyi Hu, Kewei Zhu, Lang Tian, Lin Yang, Qiqi Huang, Siqi Yang, Wenbo Su, Xiaoxiao He, Xin Tong, Xu Chen, Xunke Xi, Xiaowei Huang, Yaxuan Wu, Yeqiu Yang, Yi Hu, Yujin Yuan, Yuliang Yan, and Zile Zhou. 2025. RecGPT-V2 Technical Report. arXiv Preprint (2025). https://arxiv.org/abs/2512.14503
- [43] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec. 2018. Graph Convolutional Neural Networks for Web-Scale Recommender Systems. In KDD. 974–983.
- [44] Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, and

Liqiang Nie. 2025. R2ec: Towards Large Recommender Models with Reasoning. arXiv Preprint (2025). https://arxiv.org/abs/2505.16994

- [45] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022. Are Graph Augmentations Necessary?: Simple Graph Contrastive Learning for Recommendation. In SIGIR. 1294–1303.
- [46] Jianyang Zhai, Zi-Feng Mai, Chang-Dong Wang, Feidiao Yang, Xiawu Zheng, Hui Li, and Yonghong Tian. 2025. Multimodal Quantitative Language for Generative Recommendation. In ICLR. https://openreview.net/forum?id=v7YrIjpkTF
- [47] Zhi Zheng, Wenshuo Chao, Zhaopeng Qiu, Hengshu Zhu, and Hui Xiong. 2024. Harnessing Large Language Models for Text-Rich Sequential Recommendation. In WWW. 3207–3216.
- [48] Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, and Xiu Li.

2024. CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation. In RecSys. 969–974.

#### A Derivation of Eq. 1

Proposition A.1 (Graph-Conditioned Variational Regularization Training Objective). Let 𝐻 denote a user interaction history and 𝑖∗ the ground-truth next item observed at training time. Let C(𝑘) be the candidate set induced by the 𝑘-hop neighborhood of the most recent items 𝐼𝑛 on the interaction graph G.

Consider the latent-variable formulation 𝑝𝜃 (𝑖∗ | 𝐻) = ∑︁

𝑝𝜃 (𝑐 | 𝐻) 𝑝𝜃 (𝑖∗ | 𝑐,𝐻),

𝑐∈C(𝑘)

where 𝑐 is a discrete latent intent prototype. Let 𝑞(𝑐 | 𝐼𝑛, G) be any categorical distribution supported on C(𝑘) that does not depend on 𝜃. Then, for all 𝜃, the following inequality holds:

log𝑝𝜃 (𝑖∗ | 𝐻) ≥ E𝑞(𝑐|𝐼𝑛,G) log𝑝𝜃 (𝑖∗ | 𝑐,𝐻)

− 𝐷KL(𝑞(𝑐 | 𝐼𝑛, G) ∥ 𝑝𝜃 (𝑐 | 𝐻)) .

The right-hand side defines an ELBO-like objective that regularizes the inferred intent distribution toward the graph-conditioned prior.

Proof. Starting from the marginal likelihood, 𝑝𝜃 (𝑖∗ | 𝐻) = ∑︁

𝑝𝜃 (𝑐 | 𝐻) 𝑝𝜃 (𝑖∗ | 𝑐,𝐻).

𝑐∈C(𝑘)

For any categorical distribution 𝑞(𝑐 | 𝐼𝑛, G) supported on C(𝑘), we rewrite the sum as an expectation:

𝑝𝜃 (𝑖∗ | 𝐻) = ∑︁

𝑝𝜃 (𝑐 | 𝐻) 𝑝𝜃 (𝑖∗ | 𝑐,𝐻) 𝑞(𝑐 | 𝐼𝑛, G)

𝑞(𝑐 | 𝐼𝑛, G)

𝑐∈C(𝑘)

𝑝𝜃 (𝑐 | 𝐻) 𝑝𝜃 (𝑖∗ | 𝑐,𝐻) 𝑞(𝑐 | 𝐼𝑛, G)

=E𝑞

.

Taking logarithm and applying Jensen’s inequality yields

log𝑝𝜃 (𝑖∗ | 𝐻) ≥ E𝑞 log𝑝𝜃 (𝑐 | 𝐻) + log𝑝𝜃 (𝑖∗ | 𝑐,𝐻)

− log𝑞(𝑐 | 𝐼𝑛, G) . Rearranging terms gives

log𝑝𝜃 (𝑖∗ | 𝐻) ≥ E𝑞[log𝑝𝜃 (𝑖∗ | 𝑐,𝐻)]

− 𝐷KL(𝑞(𝑐 | 𝐼𝑛, G) ∥ 𝑝𝜃 (𝑐 | 𝐻)) ,

which completes the proof. □

##### B Proof of Proposition 2.1 Proof. For a fixed candidate set C, define

exp(r⊤e𝑐) 𝑐′∈C exp(r⊤e𝑐′)

𝑃(𝑐 | 𝐻) =

exp(r⊤e𝑐) 𝑍(r)

=

, 𝑍(r) := ∑︁

exp r⊤e𝑐′ ,

𝑐′∈C

where we suppress the dependence on 𝐻 in r for notational simplicity.

The KL distillation loss is

L(r) = D𝐾𝐿(𝑄 ∥ 𝑃(· | 𝐻))

= ∑︁

𝑄(𝑐) 𝑃(𝑐 | 𝐻)

𝑄(𝑐) log

= ∑︁

𝑄(𝑐) log𝑄(𝑐) − ∑︁

𝑐∈C

𝑄(𝑐) log𝑃(𝑐 | 𝐻).

𝑐∈C

𝑐∈C

The first term is constant with respect to 𝑟, hence

∇rL(r) = − ∑︁ 𝑐∈C

𝑄(𝑐) ∇r log𝑃(𝑐 | 𝐻).

Next, using log𝑃(𝑐 | 𝐻) = r⊤e𝑐 − log𝑍(r), we have

1 𝑍(r)

∇r log𝑃(𝑐 | 𝐻) = ∇r(r⊤e𝑐) − ∇r log𝑍(r) = e𝑐 −

∇r𝑍(r). Moreover,

∇r𝑍(r) = ∇r ∑︁ 𝑐′∈C

exp r⊤e𝑐′ = ∑︁ 𝑐′∈C

exp r⊤e𝑐′ e𝑐′.

Substituting back yields

∑︁

1 𝑍(r)

exp r⊤e𝑐′ e𝑐′

∇r log𝑃(𝑐 | 𝐻) = e𝑐 −

𝑐′∈C

= e𝑐 − ∑︁ 𝑐′∈C

exp(r⊤e𝑐′) 𝑍(r)

e𝑐′

= e𝑐 − ∑︁ 𝑐′∈C

𝑃(𝑐′ | 𝐻) e𝑐′

= e𝑐 − E𝑃(·|𝐻) [e𝑐]. Therefore,

∇rL(r) = − ∑︁ 𝑐∈C

𝑄(𝑐) e𝑐 − E𝑃(·|𝐻) [e𝑐]

= − ∑︁

𝑄(𝑐)e𝑐 + ∑︁ 𝑐∈C

𝑄(𝑐) E𝑃(·|𝐻) [e𝑐].

𝑐∈C

Since 𝑄 is a probability distribution on C, 𝑐∈C 𝑄(𝑐) = 1, and thus ∇rL(r) = E𝑃(·|𝐻) [e𝑐] − E𝑄 [e𝑐], which completes the proof. □

#### C Proof of Proposition 2.2

Proof. By the triangle inequality of the total variation distance, we have

𝑑TV(𝑝𝑡′+1,𝑞𝑡′+1) ≤ 𝑑TV(𝑝𝑡′+1,𝑞𝑡′) + 𝑑TV(𝑞𝑡′,𝑞𝑡′+1). Applying the bounded teacher drift assumption in Eq. 7 yields 𝑑TV(𝑝𝑡′+1,𝑞𝑡′+1) ≤ 𝑑TV(𝑝𝑡′+1,𝑞𝑡′) + 𝛿.

Using the stepwise contraction assumption in Eq. 6, we obtain the recursive bound:

𝑑TV(𝑝𝑡′+1,𝑞𝑡′+1) ≤ (1 − 𝜆)𝑑TV(𝑝𝑡′,𝑞𝑡′) + 𝛿.

Unrolling the recursion for 𝑡′ − 1 steps gives

𝑡∑︁′−2

𝑑TV(𝑝𝑡′,𝑞𝑡′) ≤ (1 − 𝜆)𝑡′−1 𝑑TV(𝑝1,𝑞1) + 𝛿

(1 − 𝜆)𝑗

𝑗=0

1 − (1 − 𝜆)𝑡′−1 𝜆 ≤ (1 − 𝜆)𝑡′−1 𝑑TV(𝑝1,𝑞1) +

= (1 − 𝜆)𝑡′−1 𝑑TV(𝑝1,𝑞1) + 𝛿 ·

𝛿 𝜆

, which establishes the desired bound in Eq. 8. □

#### D Global Relation Modeling via Swing Graph

To capture stable collaborative signals, we construct a global item graph G = (V, E) using an enhanced variant of the Swing algorithm used in industrial practice (e.g., Alibaba [38]). This variant incorporates user activity normalization and popularity smoothing to mitigate the impact of noise from hyper-active users and hot items.

Formally, let U𝑖 denote the set of users who interacted with item 𝑖, and 𝐼𝑢 denote the interaction history of user 𝑢. For a pair of items (𝑖, 𝑗), we first identify the set of common users K𝑖𝑗 = U𝑖 ∩ U𝑗. To improve efficiency, if |K𝑖𝑗| exceeds a threshold 𝑀, we perform random sampling to obtain a subset Kˆ𝑖𝑗 ⊂ K𝑖𝑗.

The similarity score𝑆𝑖𝑚(𝑖, 𝑗) is defined as a weighted summation over user pairs (𝑢,𝑣) from Kˆ𝑖𝑗:

∑︁

∑︁

1 √︁|U𝑗|

𝑆𝑖𝑚(𝑖, 𝑗) =

𝑤𝑢𝑣,

𝑢∈Kˆ𝑖𝑗

𝑣∈Kˆ𝑖𝑗,𝑣≠𝑢

where the pair weight 𝑤𝑢𝑣 combines user activity decay and substructure strength:

1 (|𝐼𝑢| + 𝛼1)𝛽 · (|𝐼𝑣| + 𝛼1)𝛽 User Activity Weight

1 |𝐼𝑢 ∩ 𝐼𝑣| + 𝛼2

𝑤𝑢𝑣 =

·

Overlap Penalty

where 𝛼1,𝛼2 are smoothing parameters, and 𝛽 controls the strength of user activity penalization. The term 1/√︁|U𝑗| acts as a normalization factor to prevent popular items from dominating the retrieval results. This formulation ensures that the “Intent Anchors” are derived from high-quality, non-trivial collaborative structures.

#### E ManCAR Algorithms

We summarize the implementations of ManCAR’s training and adaptive reasoning in Algorithms 1 and 2, respectively.

Notably, we find that 𝑘 = 1 is sufficient for all Amazon Reviews datasets and adopt this setting throughout our experiments. Increasing 𝑘 consistently degrades performance on these datasets, likely due to their high sparsity induced by the dataset construction process. In such settings, expanding to higher-order neighborhoods introduces noisy candidates that dilute useful collaborative signals. We note that larger 𝑘 may be beneficial in denser, real-world industrial scenarios, where higher-order relations are more reliable, and we recommend applying exponential decay to 𝑘-hop neighbor weights to mitigate noise when increasing 𝑘.

- Algorithm 1 ManCAR Training Algorithm

Required: train set {𝐻(𝑗),𝑖∗(𝑗)}𝑁𝑗=1, batch size 𝐵, #hops 𝑘, recent

interaction window size 𝑛, reasoning step 𝑇

- 1: Construct global interaction graph G ← SWING({𝐻(𝑗)}𝑁𝑗=1)
- 2: for randomly sampled mini-batch {𝐻(𝑗),𝑖∗(𝑗)}𝐵𝑗=1 do
- 3: C(𝑗)(𝑘) ← C(𝐼𝑛(𝑗); G;𝑘)
- 4: for 𝑡 = 1, . . .𝑇 do
- 5: Get teacher prior 𝑞(𝑡)(𝑐(𝑗)|𝐼𝑛(𝑗), G) via Eq. 2
- 6: Compute the main loss Lmain(𝑡) via Eq. 3
- 7: Compute the regularization loss Lreg(𝑡) via Eq. 4
- 8: end for
- 9: Compute the overall loss L via Eq. 5
- 10: Minimize L w.r.t. 𝜃
- 11: end for
- 12: return 𝜃

- Algorithm 2 ManCAR Adaptive Reasoning Inference Algorithm

Required: inference input 𝐻, global interaction graph G constructed on train set, #hops 𝑘, recent interaction window size 𝑛, max reasoning step 𝑇max, early stop threshold 𝜖

- 1: Define 𝑝𝜃(0) ← NULL
- 2: for 𝑡 = 1 . . .𝑇max do
- 3: C(𝑘) ← C(𝐼𝑛; G;𝑘)
- 4: 𝑝𝜃(𝑡) ← 𝑓𝜃(𝑡)(𝐻, C(𝑘))
- 5: if 𝑝𝜃(𝑡−1) is not NULL then
- 6: if 𝐷KL(𝑝𝜃(𝑡−1)||𝑝𝜃(𝑡)) < 𝜖 then
- 7: End the reasoning at step 𝑡
- 8: end if
- 9: end if
- 10: end for
- 11: return 𝑝𝜃(𝑡)

F Additional Analyses

- F.1 More Results for Data-Aware Train-Test Compute Allocation

Tab. 5 reports the reason step in train and inference phase of ERL, PRL, PLR, and ManCAR on seven datasets, providing supplementary results to Tab. 3 of Sec. 3.3.

- F.2 Computation Complexity Analysis

Let |𝐶| and |𝐻| denote the length of context prompt and user history, respectively, 𝑑 denote the hidden dimension, 𝐿 denote the number of Transformer layers, 𝑇′ denote the number of reasoning steps.

Transformer Encoder. In multi-head self attention, the Q/K/V/out projection and weighted sum costs O((|𝐶| + |𝐻|)𝑑2) and O((|𝐶| + |𝐻|)2𝑑). In FFN, the two linear-layer costs O((|𝐶| + |𝐻|)𝑑2). Thus, the total FLOPs for 𝐿-layer Transformer encoder is O(𝐿((|𝐶| + |𝐻|)2𝑑 + (|𝐶| + |𝐻|)𝑑2)).

Autoregressive Reasoning. With KV cache enabled, for each step 𝑡′ ∈ [1, ...,𝑇′], the cost of Q/K/V/out projection is O(𝑑2), the cost of attention weighted sum is O((|𝐶| + |𝐻| +𝑡′ − 1)𝑑), the cost of FFN

- Table 5: Number of reasoning steps used during training and inference for ERL, PRL, PLR, and ManCAR across seven datasets. Note that LARES adopts a loop-architecture-based method rather than forward step-wise reasoning; the reported step count for LARES corresponds to the number of loop iterations.

|Dataset Reason step<br><br>|ERL PRL PLR LARES|ManCAR|
|---|---|---|
|CDs<br><br>Train step Infer step<br><br>|2 2 3 4 2 2 3 4<br><br>|5 1.84|
|Video<br><br>Train step Infer step<br><br>|2 2 2 4 2 2 2 4|4 3.68<br><br>|
|Office<br><br>Train step Infer step|2 2 2 4 2 2 2 4<br><br>|2 1.93|
|Arts<br><br>Train step Infer step<br><br>|2 2 1 4 2 2 1 4|1 1<br><br>|
|Music<br><br>Train step Infer step|2 2 2 4 2 2 2 4<br><br>|2 1.84|
|Toys<br><br>Train step Infer step|1 2 1 4 1 2 1 4<br><br>|4 3.58|
|Grocery<br><br>Train step Infer step|2 2 3 4 2 2 3 4<br><br>|2 1.74|

- Table 6: FLOPs for ERL, PRL, PLR, LARES, and ManCAR.

Method FLOPs

O 𝐿(|𝐻 |2𝑑 + |𝐻 |𝑑2)

∑︁𝑇′

ERL/PRL

(|𝐻 | + 𝑡′ − 1)𝑑 + 𝑑2

+ 𝐿

𝑡′=1

O 𝐿(|𝐻 |2𝑑 + |𝐻 |𝑑2)

∑︁𝑇′

PLR

(|𝐻 | + 𝑛𝑡′ − 𝑛)𝑛𝑑 + 𝑛𝑑2

+ 𝐿

𝑡′=1

O (𝐿pre + 𝐿core)(|𝐻 |2𝑑 + |𝐻 |𝑑2)

LARES

+ 𝐿core𝑇′ |𝐻 |𝑑 + 𝑑2

O 𝐿 (|𝐶| + |𝐻 |)2𝑑 + (|𝐶| + |𝐻 |)𝑑2

∑︁𝑇′

ManCAR

(|𝐶| + |𝐻 | + 𝑡′ − 1)𝑑 + 𝑑2

+ 𝐿

𝑡′=1

reasoning iterations, the computational cost of the reasoning component scales as 𝐿core𝑇′ |𝐻|𝑑 + 𝑑2 . For PLR, which adopts 𝑛 parallel reasoning streams, the cost of the reasoning component is 𝐿 𝑇𝑡′′=1 (|𝐻| + 𝑛𝑡′ − 𝑛)𝑛𝑑 + 𝑛𝑑2 .

Tab. 6 summarizes the corresponding FLOPs for these methods. The additional computation in ManCAR mainly stems from processing this extra context, which has been shown in earlier experiments to yield substantial performance gains. We argue that this overhead is justified, especially in light of the effectiveness of test-time scaling strategies widely adopted in modern LLM systems.

is O(𝑑2). Thus for 𝐿-layer Transformer, the total FLOPs for𝑇′-step autoregressive reasoning part is O(𝐿 𝑇𝑡′′=1((|𝐶|+|𝐻|+𝑡′−1)𝑑+𝑑2)). ManCAR Overall. Combining these two parts, the total FLOPs for ManCAR is O(𝐿((|𝐶| + |𝐻|)2𝑑 + (|𝐶| + |𝐻|)𝑑2) +𝐿 𝑇𝑡′′=1((|𝐶| + |𝐻| + 𝑡′ − 1)𝑑 + 𝑑2)).

Among latent reasoning baselines such as ERL, PRL, PLR, and LARES, the primary architectural difference in ManCAR lies in the introduction of the graph-conditioned context prompt C, which extends the computation of sequential encoding. Specifically, for the recurrent reasoning method LARES, which consists of a 𝐿prelayer pre-encoder and a 𝐿core-layer core encoder reused across

