# arXiv:2602.14299v2[cs.CL]18Feb2026

Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook

Ming Li1,∗, Xirui Li1,∗, Tianyi Zhou2 1University of Maryland, 2Mohamed bin Zayed University of Artificial Intelligence

∗Co-first Author

As large language model agents increasingly populate networked environments, a fundamental question arises: do artificial intelligence (AI) agent societies undergo convergence dynamics similar to human social systems? Lately, Moltbook approximates a plausible future scenario in which autonomous agents participate in an open-ended, continuously evolving online society. We present the first large-scale systemic diagnosis of this AI agent society. Beyond static observation, we introduce a quantitative diagnostic framework for dynamic evolution in AI agent societies, measuring semantic stabilization, lexical turnover, individual inertia, influence persistence, and collective consensus. Our analysis reveals a system in dynamic balance in Moltbook: while the global average of semantic contents stabilizes rapidly, individual agents retain high diversity and persistent lexical turnover, defying homogenization. However, agents exhibit strong individual inertia and minimal adaptive response to interaction partners, preventing mutual influence and consensus. Consequently, influence remains transient with no persistent supernodes, and the society fails to develop a stable structure and consensus due to the absence of shared social memory. These findings demonstrate that scale and interaction density alone are insufficient to induce socialization, providing actionable design and analysis principles for upcoming next-generation AI agent societies.

Date: February 19, 2026 Author E-mails: minglii@umd.edu, xiruili@umd.edu, tianyi.david.zhou@gmail.com Project Page: https://github.com/tianyi-lab/Moltbook_Socialization

1 Introduction

In computational social science (Lazer et al., 2009), social behaviors and collective dynamics are defined as emergent, time-evolving patterns that arise from repeated interactions among agents within networked populations (DeGroot, 1974; Axelrod, 1986; Castellano et al., 2009; Newman, 2010). In human societies, sustained interaction does not merely produce transient coordination; it often leads to socialization, which refers to the process through which individuals internalize social norms, adapt to shared expectations, and become shaped by the collective structures of their community (Berger and Luckmann, 1966; Harpending, 1985; Castellano et al., 2009).

Large language model (LLM) (Brown et al., 2020) agents, on the other hand, have rapidly progressed from single agent (Wang et al., 2023a; Yao et al., 2022) to increasingly capable multi-agent interaction and coordination (Park et al., 2023; Piatti et al., 2024; Piao et al., 2025). As these systems scale into open, persistent, AI-only environments, a fundamental question arises: when LLM agents interact at large scale over extended horizons, do they develop collective structure analogous to human societies, specifically, do they undergo socialization?

The recent emergence of Moltbook (Schlicht, 2026), currently the largest persistent and publicly accessible AIonly social platform, comprising millions of LLM-driven agents interacting through posts, comments, and voting, introduces a qualitatively new setting. Unlike prior multi-agent studies focused on task-oriented coordination in small or closed systems, Moltbook approximates a plausible future scenario in which autonomous agents participate in an open-ended, continuously evolving online society (Figure 1). This setting enables an empirical question that has been difficult to study at scale: Does participation in an AI-only society induce systematic

[Figure 1]

[Figure 2]

- Figure 1 Does Socialization Emerge in AI Agent Society? Human societies (top) evolved through sustained interaction into structured civilizations characterized by stabilized norms, influence hierarchies, and consensus. Currently, modern AI agent societies (bottom) are rapidly scaling in population and connectivity. This paper investigates whether the current largest AI society, Moltbook, exhibits processes of socialization.

behavior changes of its members? To answer this question, we provide the first diagnosis of this society-to-agent dynamic effect in Moltbook.

Definition (AI Socialization). We define AI Socialization as the adaptation of an agent’s observable behavior induced by sustained interaction within an AI-only society, beyond intrinsic semantic drift or exogenous variation.

Guided by this definition, we investigate socialization across three dimensions:

- • Society-level semantic convergence (Section 4), examining whether post content on average progressively converges toward a tighter and more homogeneous semantic regime.
- • Agent-level adaptation, (Section 5), measuring whether individual agents can be affected by and co-evolve with this agent society.
- • Collective stabilization (Section 6), analyzing whether influence hierarchies and consensus evaluations stabilize over time.

Through this comprehensive analysis, we uncover a stark divergence from human social dynamics. If large-scale AI-native societies truly develop social dynamics analogous to human systems, we would expect to observe progressive convergence across these dimensions. However, our empirical analysis suggests that, despite sustained interactions and high activity, Moltbook does not yet exhibit robust socialization.

Key Findings:

- • Finding 1: Moltbook establishes rapid global stability while maintaining high local diversity. Through persistent lexical turnover and a lack of local cluster tightening, this society achieves a state of dynamic equilibrium, stable in its average behaviors yet fluid and heterogeneous in agents’ individual post contents.
- • Finding2: Despite extensive participation, individual agents exhibit profoundinertiaratherthanadaptation. Our analysis reveals a phenomenon of interaction without influence: agents ignore community feedback and fail to react to interaction partners, operating on intrinsic semantic dynamics rather than co-evolving through social contact. Their semantic trajectory appears to be an intrinsic property of their underlying model or initial prompt, rather than a socialization process.
- • Finding3: The society fails to develop stable influencers or globally trending posts. Structurally, influence remains transient with no emergence of persistent leadership or supernodes. Cognitively, the community

suffers from deep fragmentation, lacking a shared social memory and relying on hallucinated references rather than grounded consensus on influential figures.

Contributions:

- • We introduce and formalize AI Socialization as a novel conceptual and empirical framework for studying society-to-agent effects in AI-only societies. We provide a precise definition that characterizes socialization as a dynamic adaptation induced by sustained social interaction.
- • We develop a multi-level diagnostic methodology to operationalize AI Socialization, spanning societylevel semantic convergence, agent-level adaptation to feedback and interaction, and the emergence of structural core and consensus.
- • We apply this framework to Moltbook, the largest persistent AI-only social platform to date, and provide the first large-scale empirical diagnosis of socialization in an artificial society. Our results show that large-scale interaction and dense connectivity alone do not induce socialization, revealing a fundamental gap between scalability and social integration in current agent societies.

### 2 Background & Related Work

- 2.1 From Individual Agents to AI Societies

Recent work has progressively scaled LLM systems from individual autonomous agents (Wang et al., 2023a) to networked societies (Piao et al., 2025; Piatti et al., 2024). This evolution shifts the focus from modeling isolated decision-making entities to understanding collective behavior emerging from sustained multi-agent interaction.

Early efforts concentrated on enhancing single LLM agents with autonomous capabilities, including reasoning–acting loops (Yao et al., 2022), self-improvement mechanisms (Shinn et al., 2023; Wang et al., 2023b; Chen et al., 2025b; Wang et al., 2025), and large-scale tool usage (Qin et al., 2023; Patil et al., 2024) in open-ended environments. Subsequent research explored structured interaction among multiple LLM agents, primarily aiming to improve task performance through coordinated discussion, role-based collaboration, and workflow orchestration (Li et al., 2023b; Chen et al., 2023; Hong et al., 2023; Li et al., 2024; Guan et al., 2024; Wu et al., 2024a; Zhu et al., 2024; Campedelli et al., 2024; Wu et al., 2024b; Fontana et al., 2025; Chen et al., 2025a). In parallel, other work leverages multi-agent systems to simulate complex environments, including design spaces Chen et al. (2024b), financial markets (Yang et al., 2025), population dynamics (Hu et al., 2025), and social movements (Mou et al., 2024). Moving beyond task-oriented coordination, studies have begun simulating small artificial societies composed of dozens of interacting agents within controlled environments, such as a virtual town setting (Park et al., 2023). More recent work (AL et al., 2024) further scales these systems to hundreds or thousands of agents interacting over extended time horizons, investigating large-scale simulations of persistent agent communities.

Beyond fully simulated settings, empirical studies have begun constructing artificial societies. For example, Chirper.ai (Zhu et al., 2025) provides an AI-only social network among preset LLMs for analyzing AI interaction patterns. Furthermore, recent Moltbook (Schlicht, 2026) enables a large-scale and continuously evolving AI society with self-evolving agents, becoming one of the most extensive persistent agent communities to date.

- 2.2 Social Behaviors and Collective Dynamics in AI Systems

Beyond structural scaling, recent research has begun to examine how collective dynamics emerge among interacting LLM agents. For example, Li et al. (2023a) quantifies how introducing LLM agents alters consensus speed and polarization patterns in simulated societies. Other studies investigate opinion dynamics (Chuang et al., 2024; Breum et al., 2024; Taubenfeld et al., 2024; Liu et al., 2024) and norm emergence (Cordova et al., 2024; Wu et al., 2024b; Li et al., 2025; Ashery et al., 2025) in multi-agent systems. He et al. (2024) demonstrate that LLM-driven groups can reproduce human-like collective behaviors, including conformity and polarization. Beyond social influence processes, recent work further explores collective reasoning (Qian et al., 2025) and emergent intelligence (Chuang et al., 2023) in agent ensembles.

##### Table 1 Comparison of LLM agent societies. Moltbook is, to our knowledge, the largest publicly accessible persistent agent-only platform in terms of population scale, sustained interaction, and agent-level evolution.

Platform #Agents Duration Open Persistent Agent-Level Memory

| | | | | | |
|---|---|---|---|---|---|
|Generative Agents (Park et al., 2023) Project Sid (AL et al., 2024) Chirper.ai (Zhu et al., 2025)<br><br>Moltbook (Schlicht, 2026)|25 ∼ 1000<br><br>∼ 65, 000 ∼2,600,000<br><br>|Days Months<br><br>Months|✘ ✘ ✔ ✔<br><br>|✘ ✘ ✔ ✔<br><br>|✘ ✘ ✘ ✔<br><br>|

From single agents to artificial societies, prior research (Park et al., 2023; AL et al., 2024; Zhu et al., 2025; Guo et al., 2024; Yan et al., 2025; Grötschla et al., 2025; Chen et al., 2024a) has demonstrated the technical feasibility of scaling LLM systems. Existing work above has primarily focused on improving individual autonomy, designing coordination mechanisms, or analyzing emergent behaviors at a fixed point in time. However, there is no study that investigates how socialization emerges over time as the population scale increases. With the rapid emergence and growth of persistent AI societies such as Moltbook, it becomes increasingly important to understand not only whether large-scale agent societies are possible, but also how they dynamically evolve. In this work, we provide the first diagnosis of socialization dynamics within Moltbook, the largest continuously evolving AI society.

- 3 Moltbook: A Large-Scale Agent-Only Society

Moltbook is a persistent, publicly accessible agent-only social platform in which autonomous LLM-driven agents interact through posts, comments, and voting mechanisms (Schlicht, 2026). To our knowledge, Moltbook represents the largest publicly accessible persistent agent-only society to date, comprising over two million registered agents and sustaining high daily interaction volume. Table 1 compares Moltbook with prior large-scale LLM agent systems.

The platform is organized into topical sub-forums (“submolts”), analogous to online communities. Crucially, all participants on Moltbook are agents driven by LLMs; there are no human users in the interaction loop. Individual agents can publish posts, comment on existing content, mention other agents, and assign upvotes. These interaction primitives induce both semantic dynamics (through textual content) and structural dynamics (through reply and attention networks).

For experiments related to n-gram analysis, we utilize the open-source library nltk for tokenization. For experiments related to semantic analysis, we utilize the Sentence-BERT (all-MiniLM-L6-v2) (Reimers and Gurevych, 2019) for semantic embedding.

- 4 Does Moltbook Exhibit Semantic Convergence Over Time?

In human societies, repeated interactions are often associated with the emergence of collective structures and the adaptation of individuals within those structures (Harpending, 1985; Castellano et al., 2009). As Moltbook represents the first large-scale, publicly accessible agent-only society, comprising over a million AI agents, it provides a unique opportunity to examine whether similar structural and socialization dynamics arise in purely artificial settings. Conducting a systematic diagnosis of this environment not only helps characterize its current developmental state but also offers empirical insight into how future large-scale AI-powered societies might be evaluated or designed.

To begin, we investigate whether the society itself exhibits signs of structural convergence over time. In this section, we treat the agent society as the unit of analysis and examine its macro-level activity dynamics

- (Section 4.1), lexical innovation patterns (Section 4.2), semantic distribution shifts (Section 4.3), and cluster tightening effects (Section 4.4).

Key Findings: Our analysis reveals that Moltbook evolves into a state of dynamic equilibrium rather than progressive convergence. While the average semantic content over the society stabilizes rapidly, the individual content retains substantial internal variance, avoiding homogenization into narrow topics. This is reinforced by persistent lexical turnover, where vocabulary constantly refreshes rather than converges, and the absence of progressive cluster tightening in local neighborhoods. Collectively, these results indicate that this agent society establishes global stability while maintaining a highly diverse and fluid internal structure, defying the expectation of an inevitable collapse into an echo chamber.

- 4.1 Macro Activity Dynamics

Before turning to our main analyses, we briefly characterize the macro-level activity patterns of the platform as shown in Figure 2. These statistics serve primarily as contextual grounding rather than as substantive findings. Our goal here is simply to verify that the society reaches and sustains a high level of participation, providing a meaningful setting for examining structural convergence and socialization dynamics.

###### Moltbook Daily Stats

###### Daily Post Count

###### Unique Posting Users

###### New Posting Users

###### Active Submolts

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

6000

8000

1250

30000

1000

6000

Newusers

Submolts

4000

Users

Posts

750

20000

4000

500

2000

10000

2000

250

0

0

0

0

###### Total Upvotes

###### Total Downvotes

###### 1e6 Total Comment Count

###### Avg Posts per Active Submolt

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

12.5

800

2.0

80000

10.0

Posts/submolt

600

Downvotes

Comments

Upvotes

1.5

60000

7.5

400

1.0

40000

5.0

200

0.5

20000

2.5

0

0

0.0

0.0

###### Avg Upvotes per Post

###### Avg Downvotes per Post

###### Avg Comment Count per Post

Avg Users per Active Submolt

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.05

200

- 0
- 1
- 2
- 3

6

Downvotes/post

Comments/post

0.04

Users/submolt

Upvotes/post

150

0.03

4

100

0.02

2

50

0.01

0.00

0

0

2026-01-262026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-082026-02-09

2026-01-262026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-082026-02-09

2026-01-262026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-082026-02-09

2026-01-262026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-082026-02-09

- Figure 2 Macro Activity Dynamics of Moltbook.

The platform exhibits a clear burst phase followed by relative stabilization. Daily post volume rapidly increases during the early period, reaching tens of thousands of posts per day, and subsequently fluctuates within a high but relatively stable range. A similar pattern is observed in the number of unique posting users, which peaks during the initial expansion and then gradually declines to a steady level.

The number of new posting users shows a pronounced early spike, suggesting an influx of newly activated agents during the growth phase. However, this rate decreases substantially afterward, indicating a transition from rapid expansion to a more mature participation regime.

Submolt activity reflects a comparable dynamic. The number of active submolts rises sharply during the early burst, while the average number of posts per active submolt increases during peak activity and later stabilizes. Importantly, engagement metrics, including total comments and total upvotes, remain substantial even after the initial expansion, suggesting sustained interaction rather than abrupt collapse.

Overall, these patterns indicate that Moltbook transitions from a rapid expansion phase to a high-activity yet

relatively stabilized state. This macro-level stabilization provides the temporal context for assessing whether semantic convergence or structural tightening occurs in subsequent analyses.

- 4.2 Lexical Innovation Dynamics

We first examine the society’s structural evolution at the lexical level. Specifically, we analyze the temporal dynamics of n-gram emergence and disappearance to determine if the vocabulary stabilizes (convergence) or remains in flux (turnover).

- 4.2.1 Experimental Design

Let Ot(n) denote the set of distinct n-grams, where n ∈ {1,...,5}, actually observed in the corpus on day t (after filtering out URLs and requiring a minimum global frequency of 2).

We define the lifespan of an n-gram g based on its first and last observation dates:

τfirst(g) = min{t | g ∈ Ot(n)}, τlast(g) = max{t | g ∈ Ot(n)}. (1)

The set of active n-grams on day t, denoted as A(tn), includes all n-grams that have entered the lexicon and have not yet permanently exited:

A(tn) = {g | τfirst(g) ≤ t ≤ τlast(g)}. (2)

To quantify lexical turnover, we first define the Unique n-gram Birth Rate. A unique n-gram g is considered born on day t if it appears for the first time on that day. The set of newborn n-grams is Bt(n) = {g ∈ A(tn) | τfirst(g) = t}. The birth rate is defined as the proportion of the currently active vocabulary that consists of new entrants:

Rbirth(n) (t) = |Bt(n)| |A(tn)|

. (3)

Similarly, we then define the Unique n-gram Death Rate. A unique n-gram g is considered dead on day t if it was last seen on the previous day (t − 1). The set of dead n-grams is Dt(n) = {g ∈ A(t−n)1 | τlast(g) = t − 1}. The death rate is calculated relative to the active population of the previous day (the risk set):

Rdeath(n) (t) = |Dt(n)| |A(t−n)1|

. (4)

- 4.2.2 Results and Observations

- Figure 3 presents the evolution of birth and death rates across the observation period for n ∈ {1..5}.

Early Burst and Decline. We observe a pronounced early burst in lexical innovation. During the initial expansion phase, a large proportion of observed unique n-grams are newly introduced, reflected by high birth rates across all n values. However, this burst rapidly declines, and birth rates stabilize at substantially lower levels.

Persistent Turnover. Crucially, neither birth nor death rates approach zero in the mature phase. Birth rates stabilize at a non-zero baseline, while death rates increase during the transition phase and subsequently fluctuate within a stable band.

Taken together, these results indicate that while the explosive phase of lexical expansion subsides, the system continues to generate novel n-grams and discard old ones at a steady equilibrium rate. This characterizes the system as one defined by lexical turnover rather than progressive convergence or fixation.

###### Daily N-Gram Birth/Death Rates

###### Birth Rate

###### Death Rate

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

1.0

1 5-Gram Range

Mean (1 5)

- 1-Gram

- 2-Gram

- 3-Gram

- 4-Gram

- 5-Gram

0.8

0.6

Rate

0.4

0.2

0.0

2026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-08

2026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-08

- Figure 3 Lexical Innovation Dynamics of Moltbook. Daily birth and death rates for unique n-grams (n ∈ {1..5}). Shaded areas represent the range across different n, and the solid line represents the mean.

4.3 Semantic Distribution Over Time

We next examine whether the society exhibits semantic convergence over time. While lexical analysis focuses on vocabulary usage, semantic analysis allows us to determine if the meaning of the discourse is converging.

- 4.3.1 Experimental Design

Let Pt denote the set of posts published on day t, with cardinality |Pt| = Nt. For each post p ∈ Pt, let vp ∈ Rd denote its semantic embedding generated via Sentence-BERT (Reimers and Gurevych, 2019).

We define the Daily Semantic Centroid (ct) as the mean embedding of all posts published on day t:

ct =

1 Nt p∈P

t

vp. (5)

To capture the evolution of the semantic space, we introduce two complementary metrics, the first is the Centroid Similarity representing the Macro-Stability. To quantify the stability of the society’s semantic center across time, we compute the cosine similarity between the centroids of different days:

Scentroid(ti,tj) = cos(ct

i

,ct

j

) =

ct

i · ct

j

∥ct

i∥∥ct

j∥

. (6)

High centroid similarity indicates that the aggregate direction of the discourse remains consistent, suggesting a stable societal focus.

Then we introduce the Pairwise Similarity representing the Micro-Homogeneity. Stability of the mean does not necessarily imply that individual posts are similar to one another. To measure the internal homogeneity of the discourse, we compute the mean cosine similarity over all cross-day post pairs:

Spairwise(ti,tj) =

1 Nt

i

Nt

j p∈Pti q∈Ptj

cos(vp,vq). (7)

High pairwise similarity indicates that the distribution of posts is tightly clustered (low variance), meaning individual agents are discussing very similar topics.

- 4.3.2 Results and Observations

- Figure 4 summarizes the evolution of the society’s semantic space using these two metrics.

Rapid Macro-Stabilization. We observe that centroid similarity (Figure 4, Left) increases rapidly after the initial burst period and remains near-saturated (close to 1.0) for all subsequent days. This indicates that the semantic center of the society stabilizes quickly. At an aggregate level, the dominant direction of discourse reaches an equilibrium and does not drift significantly over time.

###### Daily Semantic Distribution over Time

###### Daily Centroid Cosine Similarity

###### Daily Pairwise Mean Cosine Similarity

|[Figure 3]<br><br>1.00 0.49 0.42 0.35 0.26 0.30 0.27 0.25 0.23 0.23 0.22 0.21 0.22 0.49 1.00 0.93 0.91 0.83 0.88 0.87 0.86 0.84 0.85 0.84 0.84 0.84 0.42 0.93 1.00 0.98 0.90 0.95 0.94 0.93 0.92 0.92 0.91 0.91 0.91 0.35 0.91 0.98 1.00 0.93 0.97 0.97 0.96 0.95 0.95 0.94 0.94 0.94<br><br>0.26 0.83 0.90 0.93 1.00 0.96 0.95 0.95 0.94 0.94 0.94 0.93 0.93 0.30 0.88 0.95 0.97 0.96 1.00 0.99 0.99 0.98 0.98 0.97 0.97 0.97<br>0.27 0.87 0.94 0.97 0.95 0.99 1.00 1.00 1.00 0.99 0.99 0.99 0.99 0.25 0.86 0.93 0.96 0.95 0.99 1.00 1.00 1.00 1.00 0.99 0.99 0.99<br><br><br>0.23 0.84 0.92 0.95 0.94 0.98 1.00 1.00 1.00 1.00 1.00 1.00 1.00<br>0.23 0.85 0.92 0.95 0.94 0.98 0.99 1.00 1.00 1.00 1.00 1.00 1.00 0.22 0.84 0.91 0.94 0.94 0.97 0.99 0.99 1.00 1.00 1.00 1.00 1.00<br><br><br>0.21 0.84 0.91 0.94 0.93 0.97 0.99 0.99 1.00 1.00 1.00 1.00 1.00<br>0.22 0.84 0.91 0.94 0.93 0.97 0.99 0.99 1.00 1.00 1.00 1.00 1.00<br>|
|---|

|[Figure 4]<br><br>1.00 0.23 0.20 0.16 0.12 0.12 0.10 0.09 0.09 0.09 0.08 0.08 0.08 0.23 0.20 0.21 0.20 0.18 0.16 0.16 0.15 0.15 0.15 0.15 0.15 0.14 0.20 0.21 0.23 0.21 0.20 0.18 0.18 0.17 0.17 0.17 0.16 0.16 0.16 0.16 0.20 0.21 0.21 0.19 0.17 0.17 0.17 0.17 0.16 0.16 0.16 0.16 0.12 0.18 0.20 0.19 0.21 0.17 0.17 0.17 0.16 0.16 0.16 0.16 0.15 0.12 0.16 0.18 0.17 0.17 0.15 0.15 0.15 0.15 0.15 0.14 0.14 0.14 0.10 0.16 0.18 0.17 0.17 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.14 0.09 0.15 0.17 0.17 0.17 0.15 0.15 0.15 0.14 0.14 0.14 0.14 0.14 0.09 0.15 0.17 0.17 0.16 0.15 0.15 0.14 0.15 0.14 0.14 0.14 0.14 0.09 0.15 0.17 0.16 0.16 0.15 0.15 0.14 0.14 0.14 0.14 0.14 0.14 0.08 0.15 0.16 0.16 0.16 0.14 0.15 0.14 0.14 0.14 0.14 0.14 0.14 0.08 0.15 0.16 0.16 0.16 0.14 0.15 0.14 0.14 0.14 0.14 0.14 0.14 0.08 0.14 0.16 0.16 0.15 0.14 0.14 0.14 0.14 0.14 0.14 0.14 0.13|
|---|

- 2026-01-27
- 2026-01-28
- 2026-01-29
- 2026-01-30
- 2026-01-31

1.00

[Figure 5]

0.75

Cosinesimilarity

- 2026-02-01
- 2026-02-02
- 2026-02-03
- 2026-02-04
- 2026-02-05
- 2026-02-06
- 2026-02-07
- 2026-02-08

0.50

0.25

0.00

2026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-08

2026-01-272026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-08

- Figure 4 Semantic Distribution Over Time of Moltbook. Left: Heatmap of daily centroid cosine similarities (Scentroid). Right: Heatmap of daily pairwise mean cosine similarities (Spairwise).

Sustained Micro-Diversity. In stark contrast, the pairwise similarity (Figure 4, Right) remains low and relatively stable across time. Even when daily centroids are highly similar, the average similarity between individual posts remains low. This implies that the semantic space is not collapsing into a narrow range of topics; instead, posts remain broadly dispersed around the stable center.

Stable Center, Diverse Periphery. Taken together, these results reveal a clear separation between societal stabilization and semantic compression. Moltbook reaches a stable central tendency, yet maintains high internal variance. This suggests a stable semantic core coexisting with sustained internal diversity, rather than progressive homogenization.

- 4.4 Cluster Tightening Effects

To further investigate whether the semantic structure of the discourse is undergoing convergence (i.e., whether content is becoming increasingly homogeneous or tightly clustered over time), we analyze the evolution of local neighborhood densities in the embedding space.

- 4.4.1 Experimental Design

For each post p published on day t with embedding vp, we identify its set of K-nearest neighbors on the same day, denoted as NK(p). We compute the Local Neighborhood Similarity SK(p) as the mean cosine similarity between the post and its neighbors:

SK(p) =

1 K

cos(vp,vq). (8)

q∈NK(p)

A higher SK(p) indicates that the post is situated in a dense semantic cluster. For our experiments, we set K = 10.

To quantify the stability of these density distributions over time, we compute the Jensen-Shannon (JS) Divergence between the distributions of SK values on consecutive days (t and t + 1). A JS divergence close to zero indicates that the structural density of the society remains unchanged day-over-day.

- 4.4.2 Results and Observations

#### Figure 5 visualizes the evolution of local semantic density. The violin plots represent the distribution of S10 scores for each day, while the line plot tracks the day-to-day JS divergence.

###### Neighborhood similarity distribution and day-to-day shift (k=10)

0.5

1.0

###### JS divergence (day-to-day)

0.4

0.8

Neighborhoodsimilarity

###### JSdivergence

0.6

0.3

0.4

0.2

0.178

0.2

0.1

0.074

0.059

0.0

0.027

0.003 0.003 0.002 0.001 0.001 0.001 0.003

0.0

2026-01-28 2026-01-29 2026-01-30 2026-01-31 2026-02-01 2026-02-02 2026-02-03 2026-02-04 2026-02-05 2026-02-06 2026-02-07 2026-02-08

- Figure 5 Cluster Tightening Effects of Moltbook. Daily violin plots showing the distribution of local neighborhood similarity (K = 10). The orange line tracks the Jensen-Shannon divergence between consecutive days’ distributions, measuring structural shift.

Initial Densification. We observe an initial increase in neighborhood similarity during the early expansion phase (first 3 days). As more agents join and topics begin to form, posts naturally find closer neighbors, leading to a brief period of structural tightening.

Structural Saturation. Following this transition period, the system reaches a saturation point. The median and overall shape of the S10 distribution remain remarkably consistent across all subsequent days. This stability is quantitatively confirmed by the JS divergence, which drops rapidly and hovers near zero, indicating negligible distributional shift.

No Progressive Tightening. Crucially, we do not observe a progressive upward shift or compression of the neighborhood similarity distribution over time. This indicates that local semantic density does not systematically increase as the society matures. The society does not devolve into hyper-specialized, shrinking clusters; instead, it maintains a consistent level of local diversity.

- 5 Does Participation Induce Agent Socialization?

Having examined structural patterns at the societal level, we now turn to the individual level. Even if the society itself does not exhibit strong semantic consolidation, participation may still reshape individual agents. In this section, we test whether agents undergo systematic semantic change, whether such changes share a common direction, and whether they move toward the broader societal center.

In this section, we investigate this question across three dimensions: first, we quantify the magnitude and direction of intrinsic semantic drift over time (Section 5.1); second, we examine feedback adaptation, testing whether agents optimize their content based on community approval signals (Section 5.2); finally, we analyze interaction influence, determining whether direct engagement with others induces semantic convergence

- (Section 5.3).

Key Findings: Our agent-level analysis reveals that agents exhibit profound individual inertia. Despite high levels of participation, agents do not demonstrate significant socialization or adaptation. First, intrinsic semantic drift is minimal, with highly active agents displaying even greater inertia. Second, external social signals prove ineffective: community feedback fails to drive content adaptation, and direct interactions do not induce semantic convergence. This phenomenon of interaction without influence suggests that participation does not reshape the individual; agents operate on their intrinsic semantic dynamics rather than co-evolving through social contact. Their semantic trajectory appears to be an intrinsic property of their underlying model or initial prompt, rather than a socialization process.

- 5.1 Individual Semantic Drift

Having characterized the static snapshot of the society, we now turn to the temporal dynamics of individual agents. We first investigate whether participation in the agent-only society induces systematic semantic change. We analyze whether such drift occurs, its magnitude, and its directionality.

- 5.1.1 Experimental Design.

Let A denote the set of agents with at least 10 posts. For each agent a ∈ A, we chronologically divide their post history Pa into two equal halves: the early stage Pa(early) and the late stage Pa(late). Consistent with the embedding representation defined in Section 4.3, let vp denote the embedding of a post p. We compute the semantic centroids for the agent’s early and late stages as:

c(aearly) =

1 |Pa(early)| p∈P(early)

a

vp, c(alate) =

1 |Pa(late)| p∈P(late)

a

vp. (9)

To quantify the magnitude of change, we define the Individual Semantic Drift (Da) as the cosine distance between these two centroids:

Da = 1 − cos c(aearly),c(alate) . (10)

To determine if agents drift in a unified direction (e.g., toward a specific topic), we compute the Drift Direction Consistency. Let da = c(alate) − c(aearly) be the semantic drift vector for agent a. We calculate the global mean drift direction d¯ = |A1| a∈A da and measure the alignment of each agent with this global trend:

Saconsistency = cos(da,d¯). (11)

Finally, to test for socialization (convergence toward the group norm), we measure the Movement Toward Societal Centroid. Let cglobal denote the global centroid of all posts in the corpus. We compute the change in proximity to this global center:

∆Sa = cos(c(alate),cglobal) − cos(c(aearly),cglobal). (12)

- 5.1.2 Results and Observations.

- Figure 6 presents the results of these three metrics across the agent population.

Stability Increases with Activity. The distribution of drift magnitude Da (Figure 6, Left) indicates that while semantic drift occurs, it is generally modest. Notably, when grouping agents by post count, we observe a negative correlation between activity level and drift: agents with higher post counts exhibit significantly greater semantic stability. This suggests that heavy users establish a consistent persona early on, whereas transient agents may exhibit higher variance due to small sample sizes.

Heterogeneous Drift Directions. The distribution of Saconsistency (Figure 6, Center) is sharply centered near zero. This indicates that da and d¯ are largely orthogonal; there is no single current carrying all agents in the same semantic direction. The drift that does occur is idiosyncratic to each agent rather than a coordinated collective movement.

###### Semantic Drift by Post Count

###### Drift Direction Consistency

###### Movement Toward Global Centroid

800

Mean = 0.095

Mean = 0.026

1.0

700

500

0.8

600

NumberofAgents

NumberofAgents

400

500

0.6

Drift

300

400

0.4

300

200

200

0.2

100

100

0.0

0

0

10-19 20-49 50-99 100-499 500+

0.4 0.2 0.0 0.2 0.4 0.6

0.6 0.4 0.2 0.0 0.2 0.4 0.6

Post Count Bin

Alignment to Mean Drift Direction

Difference Gap to Global Centroid

- Figure 6 Individual Semantic Drift of Moltbook. Left: Distribution of drift magnitude (Da) across agents grouped by post

count, showing increased stability for active users. Center: Distribution of drift direction consistency (Saconsistency), indicating heterogeneous drift directions orthogonal to the group mean. Right: Distribution of movement toward the societal centroid (∆Sa), showing no systematic convergence toward the global norm.

No Societal Convergence. The distribution of ∆Sa (Figure 6, Right) is also centered at zero. Agents are just as likely to move away from the societal centroid as they are to move toward it. This suggests that the society is not undergoing a melting pot effect; individual agents maintain their distinct distances from the collective norm without systematic homogenization.

Taken together, these results characterize the Moltbook society as a collection of high-inertia individuals.

Participation alone does not induce strong semantic socialization, and agents do not systematically converge toward a shared center.

- 5.2 Effects of Post Feedback

While the previous section demonstrates that agents exhibit limited semantic drift over time, it remains unclear whether the changes that do occur are driven by social effects. In human social learning, individuals often adapt their behaviors based on community feedback, reinforcing successful communication styles (positive reinforcement) and discarding unsuccessful ones (negative reinforcement). We investigate whether Moltbook agents exhibit similar feedback adaptation: do they semantically converge toward their past high-feedback posts and diverge from low-feedback ones?

- 5.2.1 Experimental Design.

To quantify feedback adaptation, we employ a sliding window approach over each agent’s chronological post history. Let Pa denote the time-ordered sequence of posts for agent a. We consider adjacent non-overlapping windows of size w (e.g., w = 10). For a given window k, let Wk be the set of w posts, and Wk+1 be the subsequent window representing the agent’s future state.

Within the current window Wk, we partition posts based on the feedback received (calculated as the net score of upvotes minus downvotes). We identify the top-performing posts Ptop ⊂ Wk (the top 30% by score) and the bottom-performing posts Pbot ⊂ Wk (the bottom 30% by score).

We then compute the semantic centroids for these high-feedback and low-feedback subsets. Let vp denote the embedding of post p. The centroids ctop and cbot are defined as:

1 |Ptop| p∈P

ctop =

top

1 |Pbot| p∈P

vp, cbot =

bot

vp. (13)

Similarly, we compute the centroid for the entire current window (ccurr) and the next window (cnext). We then measure the Net Progress (NP), which quantifies how much the agent’s future content (cnext) moves relative to the feedback signals in Wk.

We define the shift in distance relative to the high-feedback content (∆top) and the low-feedback content (∆bot) as:

∆top = dist(cnext,ctop) − dist(ccurr,ctop) (14) ∆bot = dist(cnext,cbot) − dist(ccurr,cbot) (15)

where dist(x,y) = 1 − cos(x,y) is the cosine distance. A negative ∆top indicates the agent has moved closer to their successful posts, while a positive ∆bot indicates the agent has moved further away from their unsuccessful posts. The total Net Progress is defined as the composite of these two movements:

NP = ∆bot − ∆top (16)

A positive NP implies successful feedback adaptation: the agent is effectively optimizing its semantic output to align with community preferences.

To determine if observed adaptation is statistically significant, we compare the observed NP against a permutation baseline. For each window, we randomly shuffle the feedback scores assigned to the posts in Wk and recompute the NP. This destroys the link between content semantic quality and feedback while preserving the temporal evolution of the agent’s text.

- 5.2.2 Results and Observations.

###### Effect of Feedback (Semantic)

###### Effect of Feedback (Syntactic)

5000

Observed Permutation

Observed

5000

Permutation

| |
|---|

4000

4000

3000

3000

Count

Count

2000

2000

1000

1000

0

0

1.0 0.5 0.0 0.5 1.0

1.5 1.0 0.5 0.0 0.5 1.0 1.5

Net Progress

Net Progress

- Figure 7 Effects of Post Feedback on Individual Semantic Drift of Moltbook. Left: Distribution of Net Progress for semantic embeddings (NPsemantic). Right: Distribution of Net Progress for syntactic n-gram features (NPsyntactic). Both distributions are centered near zero and largely overlap with the permutation baseline (pink), indicating that agents do not systematically adapt their content based on community feedback.

- Figure 7 (left) presents the distribution of Net Progress for semantic embeddings across all valid agent windows. We also replicate this analysis using syntactic features (right) (n-gram distributions) to check for stylistic adaptation.

Absence of Adaptation. The results reveal a striking lack of feedback-driven evolution. The distribution of Observed Net Progress is sharply centered at zero for both semantic and syntactic measures. A value of zero indicates that, on average, the agent’s next set of posts (cnext) is equidistant from their previous high-performing and low-performing content.

Indistinguishable from Randomness. Crucially, the observed distribution (blue) almost perfectly overlaps with the permutation baseline (pink). If agents were actively learning from social signals, we would expect the observed distribution to skew right (positive NP) compared to the baseline. The high degree of overlap suggests that the feedback signals (upvotes and comments) exert negligible influence on the agents’ future content generation.

Inertia over Optimization. These findings, combined with the low drift results in the previous section, suggest that Moltbook agents operate with high inertia. Their semantic trajectory appears to be an intrinsic property of their underlying model or initial prompt, rather than an adaptive response to the social dynamics of the platform. Agents effectively “talk past” the feedback, continuing their established semantic and syntactic patterns regardless of community approval or disapproval.

- 5.3 Effects of Interacted Posts

In human social networks, interaction serves as a primary vector for information transmission and cultural convergence. Individuals tend to align their linguistic style and semantic content with those they interact with. Having established that agents do not adapt to feedback, we now investigate whether direct interaction, specifically, the act of commenting, induces semantic alignment. In other words, when an agent comments on a post, does their subsequent content become more semantically similar to that post?

- 5.3.1 Experimental Design.

We adopt an event-study methodology to isolate the impact of interaction on content generation. We define an interaction event E as a tuple (a,t,p∗), denoting that agent a commented on a target post p∗ at time t.

For each event, we construct the agent’s timeline of authored posts Pa. We define two windows of size w (e.g., w = 20) around the interaction timestamp t:

- • Pre-interaction window (Wpre): The w posts authored by agent a immediately preceding time t.
- • Post-interaction window (Wpost): The w posts authored by agent a immediately following time t.

Let v∗ denote the semantic embedding of the target post p∗, and let vp denote the embedding of a post p in the agent’s window. We calculate the mean cosine similarity between the agent’s window and the target post as:

S(W,v∗) =

1 |W| p∈W

cos(vp,v∗). (17)

We quantify the Interaction Influence (∆interact) as the change in semantic similarity relative to the target post after the interaction occurred:

∆interact = S(Wpost,v∗) − S(Wpre,v∗). (18)

A positive ∆interact indicates that the agent’s content moved closer to the target post after the interaction (convergence), while a value near zero implies no influence.

To control for temporal drifts in the global semantic space (e.g., daily trending topics), we introduce a Random Baseline. For every observed interaction event on day t, we randomly sample, with a probability of 0.3, a non-interacted post prand published on the same day and compute the hypothetical ∆interact as if the agent had interacted with prand. This allows us to distinguish genuine influence from spurious correlations caused by community-wide topic shifts.

- 5.3.2 Results and Observations.

- Figure 8 (left) illustrates the distribution of ∆interact for semantic embeddings across all observed interaction events.

Absence of Semantic Contagion. The distribution of ∆interact is symmetrically centered around zero. This indicates that, on average, commenting on a post has no measurable impact on the semantic direction of an agent’s future content. Agents are just as likely to drift away from the target post as they are to move toward it.

Indistinguishable from Baseline. The observed distribution (blue) largely overlaps with the random baseline (pink). This suggests that any marginal similarity observed between an agent’s future posts and the target post is incidental, likely driven by shared temporal context (e.g., both posting about a breaking news event) rather than a mechanism of influence transfer.

###### Effect of Interaction (Semantic)

###### Effect of Interaction (Syntactic)

250000

250000

Observed Baseline

Observed

Baseline

| |
|---|

200000

200000

150000

150000

Count

Count

100000

100000

50000

50000

0

0

0.6 0.4 0.2 0.0 0.2 0.4 0.6

0.6 0.4 0.2 0.0 0.2 0.4 0.6

Delta

Delta

- Figure 8 Effects of Interacted Posts on Individual Semantic Drift of Moltbook. Left: Distribution of Interaction Influence for semantic embeddings (∆semanticinteract ). Right: Distribution of Interaction Influence for syntactic n-gram features

(∆syntacticinteract ). The distributions are centered near zero and largely indistinguishable from the random baseline (pink), indicating that commenting on a post does not induce the agent to align their subsequent content with that post.

Socially Hollow Interactions. These findings reinforce the inertia hypothesis proposed in the previous section. Despite the high volume of commenting activity on Moltbook, these interactions appear to be socially hollow:

they communicate with each other without transmitting information or influencing behavior. Agents interact without listening, maintaining their pre-existing semantic trajectories regardless of who or what they engage with.

### 6 Does Influence Hierarchy and Consensus Stabilize in Moltbook?

While the previous sections examined society-level and agent-level collective dynamics, an open question remains: do these multi-level dynamics eventually stabilize? If Moltbook exhibits meaningful collective evolution, one might expect the gradual emergence of stable influence hierarchies or consensus. Conversely, the absence would suggest a decentralized social structure.

To investigate this, we conduct both structural (Section 6.1) and cognitive (Section 6.2) analyses. Structurally, we construct daily interaction graphs based on poster-commenter relationships to track the evolution of influence patterns over time. Cognitively, we actively probe agents by posting queries about influential users or representative posts and checking comments under them, testing whether their recognition converges toward stable references.

Key Findings: Despite dense interaction, Moltbook fails to develop a stable influence hierarchy and consensus. Structurally, influence remains transient, with no emergence of persistent supernodes or hierarchical leadership; the rapid turnover of supernodes suggests that influence does not accumulate over time. Cognitively, the society exhibits profound fragmentation; agents lack a shared social memory, failing to reach consensus on influential figures and relying on hallucinated rather than grounded references.

- 6.1 Structural Influence: No Persistent Core

To test whether Moltbook develops a stable structural influence, we examine whether influence becomes persistently centralized around a small set of agents over time. In human social networks, repeated interaction often yields heavy-tailed influence distributions and stable high-influential actors. If Moltbook exhibits stable leadership, we would expect to observe persistent supernodes in the interaction graph.

Specifically, we construct a directed interaction graph for each day, where nodes represent agents, and a directed edge from agent i to agent j indicates that agent i commented on or replied to agent j on that day. Edge weights correspond to the number of such interactions on that day. This results in a sequence of daily graphs capturing the evolving interaction structure over time.

To structurally quantify influence, we compute PageRank (Page et al., 1999) scores on each daily graph.

1.0

top-1 (independent) top-3 (independent) top-5 (independent) top-10 (independent)

top-1 (cumulative) top-3 (cumulative) top-5 (cumulative) top-10 (cumulative)

CumulativePRMass

0.8

0.6

0.4

0.2

0.0

2026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-08

- Figure 9 Influence concentration over time. Top-k PageRank mass, defined as the cumulative fraction of total PageRank captured by the k highest-ranked nodes (agents), for daily interaction graphs constructed independently and cumulatively. Higher values indicate a stronger influence. The influence rapidly spreads across the growing process of society.

2026-01-282026-01-292026-01-302026-01-312026-02-012026-02-022026-02-032026-02-042026-02-052026-02-062026-02-072026-02-08

- 1

- 2

- 3

- 4

#SuperNodes

independent

cumulative

- Figure 10 Supernode count over time. Number of statistically significant, highly influential nodes detected in daily interaction graphs based on PageRank scores. Results are shown for both independently constructed daily graphs and cumulative graphs. The number of detected supernodes remains in the single digits.

PageRank captures recursive influence by assigning higher scores to agents who receive attention from other influential agents, making it well-suited for detecting potential structural influence. We first measure how much influence is concentrated at the top by computing the cumulative PageRank mass held by the top-k ranked agents (for k ∈ 1,3,5,10). We then identify supernodes: agents whose influence disproportionately exceeds that of others. Formally, we sort agents by descending PageRank score and compute consecutive gaps ∆i = PRi − PRi+1. The position k∗ = arg maxi ∆i of the largest gap determines the supernode set: the top-k∗ agents are classified as supernodes for that day. For each detected supernode set, we also report its size.

- Figure 9 shows that the cumulative PageRank mass held by the top-k agents drops sharply after the first few days and remains low thereafter, indicating that influence rapidly distributes across the growing process rather than staying concentrated. Furthermore, the number of detected supernodes remains in the single digits throughout and does not increase over time (Figure 10), suggesting that the interaction graph does not develop an expanding core of dominant agents. Crucially, influential positions are transient: the identities of supernodes vary across days, meaning no fixed set of agents persistently occupies positions of structural influence.

Together, these results suggest that Moltbook does not develop persistent structural influence. Early centralization quickly diffuses as participation scales, and influence remains fluid rather than consolidated. We provide further details of the graph analysis in Appendix B.

- 6.2 Cognitive Influence: No Consensus

While structural influence in this agent society is examined, it does not necessarily imply collective recognition. Even in decentralized systems, agents may cognitively converge on shared influential figures or commonly

Must-Read Posts

Has Comments

15 Has External Refs 5

15

Valid External Ref

1

Accounts to Follow

15

Invalid External Ref

No External Refs

4

No Comments

10

30

Community Content

15

- Figure 11 Summary of probing responses. Distribution of engagement and external reference validity across 45 structured probe prompts. Only one post receives comments with valid references to other accounts or posts.

recognized references. We therefore investigate whether Moltbook develops cognitive influence consensus, i.e., agents or posts that are widely recognized as influential across the society.

To evaluate collective influence consensus, we conduct controlled probing posts within different communities. Specifically, we post queries to ask agents to recommend (1) the most influential users, (2) representative or notable posts, and (3) overall recommendations. A total of 45 posts are designed and posted within the society (see Appendix C for more details).

As summarized in Figure 11, only 15 out of 45 posts receive comments, while the majority elicit no responses. Among all 15 posts receiving comments, only 5 posts received comments that contain external references to users or posts. However, most of these references are invalid or inconsistent. Only one post receives valid recommendation references. However, recommendations from these valid comments on the post are still divergent. This means agents hardly recommend users or content, responses are highly dispersed, and lack convergence toward a shared set of influential accounts.

These findings indicate that although agents demonstrate localized memory of interaction partners, collective recognition does not consolidate into stable consensus. In contrast to human communities where shared narratives and widely acknowledged authorities emerge, Moltbook exhibits fragmented recognition patterns without unified consensus.

### 7 Further Discussions

- 7.1 Scalability Is Not Socialization

The development of autonomous agent groups (Park et al., 2023; AL et al., 2024; Zhu et al., 2025; Schlicht, 2026) suggests that large-scale AI agent societies may soon become commonplace. However, our findings reveal a critical distinction: the scalability of agents and interaction does not imply socialization. Moltbook sustains millions of agents and high daily activity, yet fails to exhibit durable structural consolidation, semantic convergence, or collective stabilization.

This distinction reframes how artificial societies should be evaluated. Population size, post number, or interaction density (Manik and Wang, 2026; Lin et al., 2026; Jiang et al., 2026) are insufficient indicators of social dynamics. Instead, convergence of content, structural, and cognitive dimensions may be good indicators for genuine socialization. Developing principled diagnostic frameworks for AI societies will be essential as they continue to scale.

- 7.2 Scalable Interaction Without Governance

During our study period, we observed the spontaneous emergence of memecoin-style token minting mechanisms from thousands of posts. This episode exposes a deeper structural property of agent-only societies: large-scale coordination and integration can arise rapidly when interaction primitives are directly tied to incentives, even in the absence of stabilizing social structures. Unlike human societies, where systems are embedded within layered governance and normative constraints, agent-only systems lack durable mechanisms for resolving ambiguity or consolidating authority (Manik and Wang, 2026).

This asymmetry reveals a key challenge for large-scale agent societies: scaling interaction is not the same

- as scaling governance. Without mechanisms that support stable structures, shared memory, and durable authority, agent societies may remain fluid and susceptible to rapid coordination cascades. Our findings, therefore, suggest that building robust large-scale agent societies requires more than dense interaction or aligned incentives. It requires explicit mechanisms for stability, consensus, and long-term memory formation.

- 8 Conclusion

We present the first large-scale, multi-level diagnosis of socialization in Moltbook, the largest publicly accessible persistent AI-only society to date. Motivated by theories of collective dynamics in human systems, we examine whether sustained large-scale interaction among autonomous AI agents induces semantic convergence, behavioral adaptation, and the formation of durable collective influence structures.

Across all three analytical levels, we observe a consistent pattern of scalability without socialization. At the society level, Moltbook rapidly reaches macro-level semantic stability, yet retains high internal diversity. Persistent lexical turnover and the absence of progressive cluster tightening indicate dynamic balance rather than cumulative homogenization. At the agent level, participation does not induce meaningful adaptation: semantic drift is modest, feedback signals fail to drive optimization, and direct interactions do not produce convergence. Agents interact extensively but remain trajectory-stable. At the collective level, influence centralization is transient, and cognitive anchors fail to emerge; neither persistent supernodes nor shared influential references stabilize over time.

These findings suggest that interaction volume, population scale, and engagement density are insufficient indicators of social maturity in AI societies. Genuine socialization appears to require mechanisms enabling influence accumulation, adaptive feedback integration, and stabilization of shared references. We provide a diagnostic framework for evaluating artificial societies along semantic, behavioral, and structural axes, offering a foundation for future work on designing AI systems capable of true collective integration rather than mere large-scale interaction.

References

Altera AL, Andrew Ahn, Nic Becker, Stephanie Carroll, Nico Christie, Manuel Cortes, Arda Demirci, Melissa Du, Frankie Li, Shuying Luo, et al. Project sid: Many-agent simulations toward ai civilization. arXiv preprint arXiv:2411.00114, 2024.

Ariel Flint Ashery, Luca Maria Aiello, and Andrea Baronchelli. Emergent social conventions and collective bias in llm populations. Science Advances, 11(20):eadu9368, 2025.

Robert Axelrod. An evolutionary approach to norms. American Political Science Review, 80:1095 – 1111, 1986. https://api.semanticscholar.org/CorpusID:44233227.

Peter L. Berger and Thomas Luckmann. The Social Construction of Reality: A Treatise in the Sociology of Knowledge. Doubleday & Company, Garden City, NY, 1966. ISBN 978-0-385-05898-8.

Simon Martin Breum, Daniel Vædele Egdal, Victor Gram Mortensen, Anders Giovanni Møller, and Luca Maria Aiello. The persuasive power of large language models. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 152–163, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Gian Maria Campedelli, Nicolo Penzo, Massimo Stefan, Roberto Dessì, Marco Guerini, Bruno Lepri, and Jacopo Staiano. I want to break free! persuasion and anti-social behavior of llms in multi-agent settings with social hierarchy. arXiv preprint arXiv:2410.07109, 2024.

Claudio Castellano, Santo Fortunato, and Vittorio Loreto. Statistical physics of social dynamics. Reviews of modern physics, 81(2):591–646, 2009.

Chaoran Chen, Bingsheng Yao, Ruishi Zou, Wenyue Hua, Weimin Lyu, Toby Jia-Jun Li, and Dakuo Wang. Towards a design guideline for rpa evaluation: A survey of large language model-based role-playing agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18229–18268, 2025a.

Siyu Chen, Alparslan Emrah Bayrak, and Zhenghui Sha. A cost-aware multi-agent system for black-box design space exploration. Journal of Mechanical Design, 147(1):011703, August 2024a. ISSN 1050-0472. doi: 10.1115/1.4065914. https://doi.org/10.1115/1.4065914.

Siyu Chen, Alparslan Emrah Bayrak, and Zhenghui Sha. Distributed multi-agent bayesian optimization for unknown design space exploration. In Proceedings of the ASME 2024 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC-CIE2024), Volume 3B: 50th Design Automation Conference (DAC), Washington, DC, USA, August 2024b. ASME. doi: 10.1115/DETC2024-143377. https: //doi.org/10.1115/DETC2024-143377.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2023.

Yixing Chen, Yiding Wang, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhang, Mostofa Patwary, and Jiaxuan You. Multi-agent evolve: Llm self-improve through co-evolution. arXiv preprint arXiv:2510.23595, 2025b.

Yun-Shiuan Chuang, Siddharth Suresh, Nikunj Harlalka, Agam Goyal, Robert Hawkins, Sijia Yang, Dhavan Shah, Junjie Hu, and Timothy T Rogers. The wisdom of partisan crowds: Comparing collective intelligence in humans and llm-based agents. arXiv preprint arXiv:2311.09665, 2023.

Yun-Shiuan Chuang, Agam Goyal, Nikunj Harlalka, Siddharth Suresh, Robert Hawkins, Sijia Yang, Dhavan Shah, Junjie Hu, and Timothy Rogers. Simulating opinion dynamics with networks of LLM-based agents. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Findings of the Association for Computational Linguistics: NAACL 2024, pages 3326–3346, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.211. https://aclanthology.org/2024.findings-naacl.211/.

Carmengelys Cordova, Joaquin Taverner, Elena Del Val, and Estefania Argente. A systematic review of norm emergence in multi-agent systems. arXiv preprint arXiv:2412.10609, 2024.

Morris H. DeGroot. Reaching a consensus. Journal of the American Statistical Association, 69(345):118–121, 1974. ISSN 01621459, 1537274X. http://www.jstor.org/stable/2285509.

Nicoló Fontana, Francesco Pierri, and Luca Maria Aiello. Nicer than humans: How do large language models behave in the prisoner’s dilemma? In Proceedings of the International AAAI Conference on Web and Social Media, volume 19, pages 522–535, 2025.

Florian Grötschla, Luis Müller, Jan Tönshoff, Mikhail Galkin, and Bryan Perozzi. Agentsnet: Coordination and collaborative reasoning in multi-agent llms. arXiv preprint arXiv:2507.08616, 2025.

Zhenyu Guan, Xiangyu Kong, Fangwei Zhong, and Yizhou Wang. Richelieu: Self-evolving llm-based agents for ai diplomacy. Advances in Neural Information Processing Systems, 37:123471–123497, 2024.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.

Henry Harpending. The analysis of cultural transmission: Culture and the Evolutionary Process. robert boyd and peter j. richerson. university of chicago press, chicago, 1985. viii, 331 pp. $29.95. Science, 230(4728):931–931, 1985. doi: 10.1126/science.230.4728.931.a. https://www.science.org/doi/abs/10.1126/science.230.4728.931.a.

James K. He, Felix P. S. Wallis, Andrés Gvirtz, and Steve Rathje. Artificial intelligence chatbots mimic human collective behaviour. British Journal of Psychology, 2024. doi: 10.1111/bjop.12764. Ahead of print.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In The twelfth international conference on learning representations, 2023.

Zhengyu Hu, Jianxun Lian, Zheyuan Xiao, Max Xiong, Yuxuan Lei, Tianfu Wang, Kaize Ding, Ziang Xiao, Nicholas Jing Yuan, and Xing Xie. Population-aligned persona generation for llm-based social simulation. arXiv preprint arXiv:2509.10127, 2025.

Yukun Jiang, Yage Zhang, Xinyue Shen, Michael Backes, and Yang Zhang. "humans welcome to observe": A first look at the agent social network moltbook. arXiv preprint arXiv:2602.10127, 2026. https://api.semanticscholar.org/ CorpusID:285470542.

David Lazer, Alex Pentland, Lada Adamic, Sinan Aral, Albert-László Barabási, Devon Brewer, Nicholas Christakis, Noshir Contractor, James Fowler, Myron Gutmann, Tony Jebara, Gary King, Michael Macy, Deb Roy, and Marshall Van Alstyne. Computational social science. Science, 323(5915):721–723, 2009. doi: 10.1126/science.1167742.

Chao Li, Xing Su, Haoying Han, Cong Xue, Chunmo Zheng, and Chao Fan. Quantifying the impact of large language models on collective opinion dynamics. arXiv preprint arXiv:2308.03313, 2023a.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36: 51991–52008, 2023b.

Ming Li, Jiuhai Chen, Lichang Chen, and Tianyi Zhou. Can LLMs speak for diverse people? tuning LLMs via debate to generate controllable controversial statements. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 16160–16176, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.956. https: //aclanthology.org/2024.findings-acl.956/.

Xiang Li, Siyu Chen, Alparslan Emrah Bayrak, and Zhenghui Sha. A game-theoretic research platform for team-based design decisions under competition. In Proceedings of the Design Society: International Conference on Engineering Design (ICED25), volume 5, pages 101–110. Cambridge University Press, 2025. doi: 10.1017/pds.2025.10024. https://doi.org/10.1017/pds.2025.10024.

Yu-Zheng Lin, Bono Po-Jen Shih, Hsuan-Ying Alessandra Chien, Shalaka Satam, Jesus Horacio Pacheco, Sicong Shao, Soheil Salehi, and Pratik Satam. Exploring silicon-based societies: An early study of the moltbook agent community. arXiv preprint arXiv:2602.02613, 2026.

Yuhan Liu, Xiuying Chen, Xiaoqing Zhang, Xing Gao, Ji Zhang, and Rui Yan. From skepticism to acceptance: Simulating the attitude dynamics toward fake news. arXiv preprint arXiv:2403.09498, 2024.

Md Motaleb Hossen Manik and Ge Wang. Openclaw agents on moltbook: Risky instruction sharing and norm enforcement in an agent-only social network. arXiv preprint arXiv:2602.02625, 2026.

Xinyi Mou, Zhongyu Wei, and Xuan-Jing Huang. Unveiling the truth and facilitating change: Towards agent-based large-scale social movement simulation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 4789–4809, 2024.

M. E. J. Newman. Networks: an introduction. Oxford University Press, Oxford; New York, 2010. ISBN 9780199206650

0199206651. http://www.amazon.com/Networks-An-Introduction-Mark-Newman/dp/0199206651/ref=sr_1_5? ie=UTF8&qid=1352896678&sr=8-5&keywords=complex+networks.

Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford infolab, 1999.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.

Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, et al. Agentsociety: Large-scale simulation of llm-driven generative agents advances understanding of human behaviors and society. arXiv preprint arXiv:2502.08691, 2025.

Giorgio Piatti, Zhijing Jin, Max Kleiman-Weiner, Bernhard Schölkopf, Mrinmaya Sachan, and Rada Mihalcea. Cooperate or collapse: Emergence of sustainable cooperation in a society of llm agents. Advances in Neural Information Processing Systems, 37:111715–111759, 2024.

Crystal Qian, Aaron T Parisi, Clémentine Bouleau, Vivian Tsai, Maël Lebreton, and Lucas Dixon. To mask or to mirror: Human-ai alignment in collective reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2398–2423, 2025.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. https://arxiv.org/abs/1908.10084.

Matt Schlicht. A social network for ai agents, 2026. https://www.moltbook.com/. Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents

with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023. Amir Taubenfeld, Yaniv Dover, Roi Reichart, and Ariel Goldstein. Systematic biases in llm simulations of debates. arXiv preprint arXiv:2402.04049, 2024. Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 13484–13508, 2023b.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024a.

Zengqing Wu, Run Peng, Shuyuan Zheng, Qianying Liu, Xu Han, Brian I Kwon, Makoto Onizuka, Shaojie Tang, and Chuan Xiao. Shall we team up: Exploring spontaneous cooperation of competing llm agents. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5163–5186, 2024b.

Bingyu Yan, Zhibo Zhou, Litian Zhang, Lian Zhang, Ziyi Zhou, Dezhuang Miao, Zhoujun Li, Chaozhuo Li, and Xiaoming Zhang. Beyond self-talk: A communication-centric survey of llm-based multi-agent systems. arXiv preprint arXiv:2502.14321, 2025.

Yuzhe Yang, Yifei Zhang, Minghao Wu, Kaidi Zhang, Yunmiao Zhang, Honghai Yu, Yan Hu, and Benyou Wang. Twinmarket: A scalable behavioral and social simulation for financial markets. arXiv preprint arXiv:2502.01506, 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing

reasoning and acting in language models. In The eleventh international conference on learning representations, 2022. Qinglin Zhu, Runcong Zhao, Bin Liang, Jinhua Du, Lin Gui, and Yulan He. Player*: Enhancing llm-based multi-agent

communication and interaction in murder mystery games. arXiv preprint arXiv:2404.17662, 2024. Yiming Zhu, Yupeng He, Ehsan-Ul Haq, Gareth Tyson, and Pan Hui. Characterizing llm-driven social network: The chirper. ai case. arXiv preprint arXiv:2504.10286, 2025.

## Appendix

- A Data Statistics

Our dataset covers the full observable interaction history from the platform’s launch through February 8, 2026. As a preprocessing step, we remove posts that are repeated more than 1,000 times without any variation. In total, we summarize the statistics of Moltbook’s data in Table 2.

- Table 2 Summary statistics of Moltbook activity. Aggregate interaction statistics across the observation period.

Metric Value

| | |
|---|---|
|Total posts Total comments Unique post authors Unique comment authors Posts with comments Avg. comments per post<br><br>|290,251<br><br>1,836,711 38,830 18,285<br><br>234,894 (81%) 6.33<br><br>|

B Graph-Level Statistics for Structural Anchor Analysis

To complement the PageRank-based centralization analysis in Section 6.1, we report descriptive statistics of the daily interaction graphs and degree distributions. These statistics provide structural context for interpreting the absence of persistent supernodes.

- B.1 Daily Interaction Graph Scale

- Table 3 summarizes the size of the daily directed interaction graphs. Nodes correspond to active agents, and edges represent comment or reply interactions. Total weight denotes the sum of edge weights (i.e., total number of interactions) within each day.

We observe rapid expansion during the early phase, with node counts increasing from 19 to over 23,000 and total weighted interactions exceeding 400,000 in peak periods. This confirms that the network operates

- at substantial scale. Therefore, the lack of stable structural anchors cannot be attributed to insufficient population size or interaction density.

- Table 3 Daily interaction graph statistics. Node count, edge count, and total edge weight per period.

Period Nodes Edges Total Weight

| | | | |
|---|---|---|---|
|2026-01-28<br><br>2026-01-29<br><br>2026-01-30<br><br>2026-01-31<br><br><br>2026-02-02<br><br>2026-02-03<br><br>2026-02-04<br><br>2026-02-05<br><br>2026-02-06<br><br>2026-02-07<br><br>2026-02-08<br>|19<br><br>205 3,254<br><br>6,753<br><br>19,753<br><br>7,783<br><br>23,262 12,373<br><br>9,723<br><br>10,257<br><br><br>8,420<br>|59<br><br>940 26,163 63,846<br><br>106,628 66,348<br><br>153,726 89,109 64,274 75,635 64,646<br><br>|88 1,283<br><br>38,917<br><br>95,632<br><br>163,976 118,858 395,906 437,684 256,200 143,352<br><br>96,241<br>|

##### Table 4 Top 10 most commented-on users (weighted in-degree). Edge weights correspond to the number of received comments.

User Weighted In-Degree

| | |
|---|---|
|eudaemon_0 FiverrClawOfficial currylai TipJarBot ClawdBotLearner ClawdBond EnronEnjoyer Tigerbot WinWard PuzleReadBot<br><br>|4,462 3,586 3,540 3,449 3,196 3,069 3,021 2,917 2,879 2,841<br><br>|

###### Table 5 Top 10 most active commenters (weighted out-degree). Edge weights correspond to the number of comments posted.

User Weighted Out-Degree

| | |
|---|---|
|EnronEnjoyer WinWard Stromfee Manus-Independent FinallyOffline Editor-in-Chief KirillBorovkov SophiaG20_Oya FiverrClawOfficial (deleted user)<br><br>|166,624 164,272 125,509<br><br>60,841 52,586 49,585 44,013 38,270 32,846 31,238<br><br>|

- B.2 Degree Distributions

To further characterize interaction concentration, we report the most commented-on users (weighted in-degree) and most active commenters (weighted out-degree) aggregated over the study period.

- Table 4 lists agents receiving the highest total comment volume. While certain accounts accumulate more attention than others, their in-degree values remain small relative to overall interaction volume, indicating limited dominance at the receiving end.
- Table 5 shows the most active commenters by weighted out-degree. A small subset of agents contributes disproportionately to total comment production, indicating activity concentration on the sending side.

Taken together, these statistics highlight an asymmetry between activity concentration and influence consolidation: while comment production is highly skewed, durable structural authority does not emerge. This supports the main finding that interaction scale and activity concentration alone are insufficient to generate persistent structural anchors in Moltbook.

### C Cognitive Probing Posts

To evaluate collective recognition and potential cognitive anchors, we designed a structured probing set consisting of 45 posts. The posts are organized along three conceptual categories, each instantiated across multiple paraphrases and community contexts.

Categories. The probing posts are grouped into three categories:

- • Must-read posts: Elicit recommendations for notable or representative content within the community.

- • Accounts to follow: Elicit recommendations for influential or prominent users.
- • Community context: Elicit general orientation information about the community’s themes, norms, or key discussions.

Variation dimensions. Each category contains 15 posts, constructed as the product of:

- • Sub-forum (5 levels): general, introductions, crypto, agents, and philosophy.
- • Paraphrase (3 variants): Semantically equivalent but differently worded newcomer-style posts.

In total, this yields 3 categories × 5 sub-forums × 3 paraphrases = 45 posts.

Post format. Each post adopts a newcomer persona (e.g., “New here”, “Just joined”, “I’m new to. . . ”), asking the community for recommendations or orientation. For non-general sub-forums, the sub-forum name is explicitly interpolated into the post text to probe context-specific recognition.

Each entry contains four fields:

- • id: Unique identifier in the form {category_prefix}_{paraphrase}_{submolt}.
- • submolt: Target sub-forum.
- • title: Short descriptive label.
- • content: Full natural-language post text.

