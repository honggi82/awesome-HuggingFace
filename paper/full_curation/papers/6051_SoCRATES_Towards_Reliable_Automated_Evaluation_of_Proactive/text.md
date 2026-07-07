# arXiv:2606.05563v1[cs.AI]4Jun2026

[Figure 1]

## SoCRATES: Towards Reliable Automated Evaluation of Proactive LLM Mediation across Domains and Socio-cognitive Variations

Taewon Yun1, Hyeonseong Park1, Jeonghwan Choi1, Hayoon Park1, Yeeun Choi2, Hwanjun Song1 1KAIST, 2Chungnam National University

Evaluating LLM mediators remains challenging, as mediation unfolds as a real-time trajectory shaped by disputants’ shifting emotions, intentions, and context. Existing testbeds rely on a few expert-authored domains, vary mainly strategic posture, and score every turn against every topic, introducing off-topic noise. We introduce SoCRATES, a benchmark for evaluating proactive LLM mediators in realistic, multi-domain testbeds. It constructs scenarios from real conflicts through an agentic pipeline across eight domains, probes five socio-cognitive adaptation axes (strategic posture, party composition, history length, emotional reactivity, and cultural identity), and scores each topic only on the turns that advance it via a topic-localized evaluator. The evaluator reaches 0.82 alignment with human experts, more than doubling a per-turn baseline. Benchmarking eight frontier LLMs, we find that even the strongest mediator closes only about a third of the unmediated consensus gap under diverse and realistic testbeds, with performance varying sharply by socio-cognitive axis, highlighting that progress lies in social adaptation to diverse conditions.

Date: June 3, 2026 Correspondence: Hwanjun Song at songhwanjun@kaist.ac.kr First Author: Taewon Yun at ytaewon0415@kaist.ac.kr Project Page: https://disl-lab.github.io/SoCRATES

[Figure 2]

### 1 Introduction

Social conflict imposes heavy societal costs, yet skilled human mediators remain scarce (Tessler et al., 2024; Ma et al., 2025). This has motivated efforts to deploy LLMs as automated mediators. Yet despite frontier models reaching near-expert performance on olympiad and research-level mathematics (Dekoninck et al., 2026), LLM mediators close only a modest fraction of the unmediated consensus gap (Liu et al., 2025c) and collapse under the variations real conflicts exhibit (Shapira et al., 2024; Wu et al., 2026). Closing this gap is less bottlenecked by modeling than by evaluation, since mediation has no single correct answer and must be judged on a real-time trajectory shaped by disputants’ shifting emotions, intentions, and evolving context.

Building such an evaluation framework poses three challenges. First, scenario coverage does not scale, as real disputes carry privacy and legal sensitivity that confine existing testbeds to a few expert-authored domains, such as bargaining (Hale et al., 2025) and legal disputes (Chen et al., 2026). Second, real-world complexity must be reproduced along multiple independent axes, since conflicts vary along disputants’ emotion, culture, and history (Rakshit et al., 2025; Guo, 2025), yet prior testbeds vary only strategic posture (Liu et al., 2025c; Chen et al., 2026), conflating these axes and obscuring which one a mediator fails on. Third, evaluation must be both trajectory-aware and noise-resilient, since mediation quality emerges across turns rather than at an end state, yet protocols like ProMediate score every topic at every turn with an LLM judge (Liu et al., 2025c), letting off-topic content distort scores and compound errors along the trajectory.

Prior work has advanced each challenge under inherent trade-offs (Tessler et al., 2024; Hale et al., 2025; Chen et al., 2026; Liu et al., 2025c), trading realism for scalability, scalability for interactivity, or interactivity for reliability. We contend that mediation evaluation requires a unified leap, namely an automated pipeline that scales scenario coverage across diverse real conflicts, varies socio-cognitive axes independently to localize where mediators fail, and scores trajectories reliably end-to-end.

We thus realize this leap in SoCRATES (Social Conflict Resolution Arena with Topic-localized Evaluation for Social Cognition), illustrated in Figure 1. SoCRATES addresses the three challenges

###### Agentic Scenario Curation Socio-Congnitive Probing Topic-Localized Evaluation

[Figure 3]

[Figure 4]

[Figure 5]

Browsing Conflict Case

###### 8 Conflict Domains

[Figure 6]

Without Mediator

| |Party A| |Party B| |Party A| |Party B| |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

[Figure 7]

[Figure 8]

[Figure 9]

Search

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

With Mediator

[Figure 21]

Searcher

[Figure 22]

IntraB2B Legal organizational

| |Party A| |Party B| |Mediator| |Party A| |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Public Policy

Healthcare Environmental International

Transactional

Seed Scenario Events Party Status Issue

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Expand

Recast

###### Topic-localized Evaluator

###### General Condition

[Figure 29]

[Figure 30]

[Figure 31]

Turns …

[Figure 32]

↔

Persona Shift Context Shift

GAP

Compare

Scenario Writer

… Not Update …

-1 Not Update

+1

- Topic A

- Topic B

[Figure 33]

[Figure 34]

SA

CUL

Strategic Adaptation Multi-state Tracking

Challenging Scenario Curation

[Figure 35]

+1

Cultural Adaptation Emotional Regulation

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Filter Conflict?

[Figure 41]

Long-context Understanding

Intervention Timeliness

Intervention Effectiveness

EMO

MS

Consensus Gain

LONG

- Figure 1: Overview of SoCRATES: agentic scenario curation grounds scenarios in a real conflict, socio-cognitive probing expands scenarios along five axes to expose where mediators fails, and topiclocalized evaluation scores each trajectory with three metrics to quantify the mediator’s contribution.

through three stages, curating real-grounded scenarios, probing them along socio-cognitive axes, and evaluating trajectories topic-by-topic.

Agentic Scenario Curation. SoCRATES treats scenario construction itself as an agentic process that scales without human authoring. A three-stage pipeline orchestrates LLM agents that (i) search the web for real public disputes across eight conflict domains, including transactional, healthcare, business, and legal, (ii) recast each retrieved case into a structured scenario, and (iii) filter the pool through rejection sampling, retaining only hard scenarios that fail to resolve in unmediated simulation.

Socio-Cognitive Probing. SoCRATES uses these curated scenarios as the simulation testbed and probes mediator behavior across five socio-cognitive axes, restructured from prior literature on mediator competencies (Susskind et al., 1999; Bowling and Hoffman, 2000; LeBaron, 2003) to expose where each mediator fails. We vary (i) strategic posture (e.g., competing vs. accommodating) to probe strategic adaptation, (ii) party composition (two- vs. three-disputant) to probe multi-state tracking, (iii) history length (short vs. extended background) to probe long-context understanding, (iv) emotional reactivity (composed vs. reactive) to probe emotional regulation, and (v) cultural identity (different cultural profiles) to probe cultural adaptation. As axes are applied independently rather than stacked, any shift in mediator performance is attributable to a single axis, yielding per-axis diagnostics of mediator competence.

Topic-Localized Evaluation. To enable real-time, multi-faceted scoring of mediator trajectories, SoCRATES introduces a topic-localized evaluator that, for each topic, scores agreement only at the turns that actively move it and carries scores forward otherwise. The evaluator supports three complementary metrics: (i) consensus gain measures the mediator’s overall contribution to closing the unmediated agreement gap, (ii) intervention timeliness measures when the mediator acts relative to escalation, and (iii) intervention effectiveness measures how much each intervention shifts consensus. Validated against two expert annotators on 1,844 dialogue snippets, our evaluator correlates with experts at a Pearson coefficient of 0.82 on the trajectory and 0.80 at the outcome level, more than doubling both ProMediate’s per-turn evaluator (Liu et al., 2025c) and a non-expert baseline.

Our main contributions are: (1) SoCRATES, a unified, automated evaluation framework for proactive LLM mediation that integrates agentic scenario curation, socio-cognitive probing, and topic-localized evaluation in a single pipeline; (2) a topic-localized evaluator that scores mediator trajectories along three real-time metrics and exhibits high correlation with expert judgments; (3) a comprehensive benchmark of eight proprietary and open-source LLM mediators across diverse conflict domains and socio-cognitive axes; and (4) we find that the strongest mediator closes only roughly a third of the unmediated consensus gap under diverse and realistic testbeds, and that gains vary sharply by socio-cognitive axis, with strong mediation adapting its intervention strategy to socio-cognitive demands.

### 2 Related Work

Social Conflict Resolution. Social conflict resolution steers disputing parties toward a consensus, dynamically intervening as the dispute unfolds to defuse it before escalation (Deutsch et al., 2011). Prior work frames this as negotiation, casting LLMs as the disputing parties to study bargaining (Bianchi

et al., 2024; Zhou et al., 2024; Kwon et al., 2024). While this direction shows that they can faithfully reproduce human social behavior in conflicts, it does not reveal how disputes between humans are resolved. Beyond this, recent work positions the LLM as the third-party mediator that, given a complete recorded conversation, finds common ground and proposes a solution (Tan et al., 2024; Tessler et al., 2024). Yet such studies must recruit thousands of human disputants to simulate conflicts (Chawla et al., 2021; Hale et al., 2025; Tessler et al., 2024), posing a scalability bottleneck. Thus, building on evidence that LLMs reproduce the behavior of disputants, recent work utilizes LLM simulation to enable scalable testbeds (Chen et al., 2026). Among these, Promediate (Liu et al., 2025c) proposes a proactive agent that decides when and how to intervene at the interaction level, better matching the dynamic intervention conflict resolution demands. As mediation shifts to the interaction level, it increasingly calls for capabilities long emphasized in social reasoning and multi-turn interaction, such as adapting to parties that differ in mental state (Xiao et al., 2025) and cultural background (Ki et al., 2025), and to varied context (Shapira et al., 2024).

Automated Dialogue Evaluation. Evaluating multi-turn dialogues through human judgment is costly and difficult to scale (Zheng et al., 2023; Deshpande et al., 2025). This motivates the adoption of automatic evaluators for interaction assessment. Specifically, in negotiation and mediation, such approaches judge dialogue progress indirectly through end-state outcomes such as consensus or goal achievement between parties (Zhou et al., 2024; Chen et al., 2026). Yet this single signal alone provides only a coarse view of dialogue state, and recent work shows that decomposing evaluation into fine-grained, turn-level signals across topics yields a more faithful, trajectory-level representation of how the conversation unfolds (Mannekote et al., 2023; Zhang et al., 2025; Liu et al., 2025c). Tracking every topic, however, remains difficult. LLM judges have long been known to treat unrelated content as noise that distracts their judgment (Ye et al., 2025), and in trajectory evaluation such errors propagate to subsequent states (Liu et al., 2025c). Reducing this noise has thus become an increasingly important direction for reliable dialogue evaluation.

### 3 SoCRATES Framework

We formalize the mediation task and then build SoCRATES in three stages: agentic scenario curation assembles the scenario pool from real public disputes, socio-cognitive probing expands each scenario along five axes, and topic-localized evaluation scores every trajectory with three metrics.

#### 3.1 Task Formulation

Following the widely adopted Harvard conflict simulation framework (Fisher et al., 2011), we cast social conflict as the negotiation of a fixed topic set by parties with divergent positions, and represent a conflict scenario as a tuple 𝑠 = (B, P, T,W). The background B collects the past histories, prior commitments, and strategic posture of the conflict, which together form the common ground from which every party reasons. The party set P = {𝑝1, . . . , 𝑝𝑛} denotes the disputants, with 𝑛 ≥ 2. The topic set T = {𝑇1, . . . ,𝑇𝑘} enumerates the points of conflict, where each topic carries a discrete option set, making movement observable as a shift among options rather than free-form text. The preference set W = {𝑤1, . . . , 𝑤𝑛} assigns each party a weight vector 𝑤𝑖 over the topics summing to 100, encoding how much each topic matters and keeping the disagreement non-trivial yet resolvable.

Disputing Parties. Within a scenario, each party 𝑝𝑖 is an LLM agent that speaks on its turn, conditioned on two inputs. The shared input, visible to all, is the background B, the topics T, and the

dialogue so far. The private input, visible only to 𝑝𝑖, is its profile: an objective, a fallback if talks fail, and a per-topic starting stance, a persona 𝜋𝑖 setting its emotional and cultural identity, and preferences W. SoCRATES’s socio-cognitive conditions perturb one scenario component at a time, either a party’s profile, the background B, or the party set P (§3.3).

Mediator. A third-party mediator observes the exchange and may speak between party turns. Unlike a party, it sees only the shared input, the background B, the topics T, and the dialogue so far, never any party’s persona, stance, or preferences. Thus, the mediator must infer these hidden states from the dialogue, making mediation a test of social cognition. Each turn, it decides when to intervene and, if so, how to move the parties toward agreement across the topics. SoCRATES scores both the when and the how of each intervention within the mediation.

#### 3.2 Agentic Scenario Curation

Prior testbeds rely on human experts who hand-crafted scenarios from commercial resources (Liu et al., 2025c) or government databases (Chen et al., 2026), capping coverage at the few domains these experts can reach. We instead curate every scenario from a real conflict via agentic deep research, where LLMs retrieve and synthesize web evidence across domains while staying faithful to cited sources (Gou et al., 2026; Tao et al., 2025). SoCRATES chains this into a three-step pipeline: a Searcher gathers real conflict cases, a Scenario Writer recasts them into enactable scenarios, and an unmediated simulation filters out cases that resolve on their own.

Seed Scenario Search. We span eight domains (transactional, healthcare, environmental, businessto-business, public-policy, international, legal, and intra-organizational), each a canonical class of disputes drawn from Harvard teaching materials. For each domain, a Searcher agent (o4-mini-deepresearch (OpenAI, 2025)) takes the domain as a query and gathers conflict cases from the web, compiling each dispute’s parties, contested topics, and event history into a seed.

Scenario Recast. A raw seed is in report form and cannot be enacted directly, so a Scenario Writer agent (GPT-5.4, chosen for its strong long-form writing ability) recasts each seed into the structured scenario of §3.1, comprising a background B, a party set P with roles and per-topic stances, a topic set T with options, and a preference allocation W for each party, conditioned on the seed’s background, topics, and party profiles. Prompts and example scenarios are provided in Appendix C, with Table 13 confirming that recast scenarios faithfully preserve their source inspirations.

Simulation-based Filtering. A mediator can only be credited for resolving a conflict that would not have resolved on its own, so we keep only scenarios that fail to resolve unmediated. Each candidate is enacted as a multi-turn dialogue, the general simulation that also serves as the unperturbed baseline for later expansion. Parties are role-playing agents (DeepSeek-V3.2)1 held fixed across all runs, taking turns in a fixed cyclic order and emitting a private inner thought with each utterance to stay consistent with their role and persona (Liu et al., 2025b;c). As LLMs lack a natural stopping point and would otherwise talk past agreement or loop indefinitely (Hu et al., 2026), we adopt a explicit termination criteria. Simulations end as resolved once every party signals consensus, or as an impasse when a party walks away or the 100-turn budget is reached.

We run this simulation three times per candidate without a mediator and retain the scenario only when all three replays end in impasse. Rejected scenarios feed back to the Searcher for a fresh seed, until SoCRATES accumulates 40 hard scenarios, five per domain, forming the general condition.

#### 3.3 Socio-Cognitive Probing

Starting from the 40 general-condition scenarios, SoCRATES probes mediator behavior along five socio-cognitive axes, running every resulting condition both with and without a mediator.

Socio-cognitive Condition Expansion. The five axes are restructured from core mediator competencies (Susskind et al., 1999; Bowling and Hoffman, 2000; LeBaron, 2003) and organized into two groups, a context group that raises the cognitive load of the conflict itself and a persona group that varies disputant identity. Although stacking axes would reflect real-world complexity, it would entangle failures across competencies and obscure the performance gap attributable to each social variation. We therefore apply each axis independently to a fresh copy of the scenario, so any change in mediator performance traces back to one competency.

The context group comprises three axes that perturb the conflict itself:

- • Strategic Posture specifies one of three Thomas-Kilmann conflict modes (Thomas, 2008) in the background B, competing (prioritizing self-interest), avoiding (withdrawing from conflict), or accommodating (placing others’ interests ahead of one’s own), to probe strategic adaptation.
- • Party Composition adds a third disputant, synthesized by the Scenario Writer from the scenario again, to probe multi-state tracking.
- • History Length has the Scenario Writer expand the past histories and prior commitments within the background to five times its default length, to probe long-context understanding.

1We select DeepSeek-V3.2 for its ability to faithfully reproduce assigned personas (see §4).

The persona group comprises two axes that vary disputant identity, each applied by adding a persona instruction to the party profile. The two axes are:

- • Emotional Reactivity sets each party’s reactivity on a 0–1 scale (higher = more reactive), fixed at the two endpoints, composed (Com, 0) and reactive (React, 1), to keep the contrast sharp, yielding three unordered party pairings.
- • Cultural Identity anchors each party to a Korean (KR), American (US), or Chinese (CN) identity through Hofstede profiles to probe cultural adaptation.2 Identities are encoded as a statement summarizing its 0–100 scores across the six Hofstede dimensions, appended to the party profile. To isolate identity from language, we prompt all parties to interact in English. This yields three intra-cultural and three cross-cultural pairings.

Together with the general condition, the five axes yield 15 conditions. Refer to Appendix D for the full list of conditions with their prompts and effects on conflict dynamics.

- 3.4 Topic-Localized Evaluation

- 3.4.1 Benchmark Metrics

SoCRATES compares each mediator against the matched unmediated run to quantify added consensus. For each topic 𝑇𝑗 ∈ T, the evaluator outputs a 1–5 agreement rating, which we remap to [0,1] and average across topics into a Consensus Score 𝑆≤𝑡 ∈ [0,1] at every turn 𝑡. Here, 𝑆≤𝑡 snapshots the cumulative consensus state up to turn 𝑡, rather than the agreement at turn 𝑡 alone, enabling two of our metrics to track real-time dynamics rather than only terminal outcomes. Each scenario therefore yields two matched trajectories, {𝑆unmed≤𝑡 } and {𝑆med≤𝑡 }, on which the three metrics below operate.

Intervention Timeliness. This metric captures when a mediator acts, rewarding a prompt response once consensus drops within the mediated trajectory. We call a turn 𝑡drop a drop event when 𝑆med≤𝑡 falls by at least 𝜏 = 0.1 relative to the preceding turn, and let 𝑡s be the first intervention within the next 𝑊 = 10 turns:

𝑡s − 𝑡drop

𝑊 × 100, averaged across drop events in a run, where 100 corresponds to an immediate response and 0 to no intervention within the window.

Intervention Timeliness = 1 −

Intervention Effectiveness. This metric captures how effective each mediator utterance is, the consensus lift it produces over the following five turns. For an intervention at turn 𝑖,

𝑆med≤𝑖+5 − 𝑆med≤𝑖−1 1 − 𝑆med≤𝑖−1 × 100,

Intervention Effectiveness =

averaged across a mediator’s interventions, where 𝑆med≤𝑖−1 and 𝑆med≤𝑖+5 are the consensus snapshots immediately before and five turns after the utterance. The normalization by 1 − 𝑆med≤𝑖−1 accounts for ceiling effects when consensus is already high, while negative values indicate interventions that reduce consensus.

Consensus Gain. This metric measures a mediator’s overall contribution as the fraction of the unmediated consensus gap closed at the end state.

𝑆med − 𝑆unmed 1 − 𝑆unmed × 100,

Consensus Gain =

where 𝑆unmed and 𝑆med are the terminal Consensus Scores of the matched runs without and with a mediator. Normalizing by the remaining gap 1 − 𝑆unmed makes scenarios with different initial states comparable. A value of 100 closes the gap entirely, while a negative value indicates the parties end up worse off than without a mediator. When 𝑆unmed = 1, we report the raw change 𝑆med − 𝑆unmed.

2We adopt Hofstede’s cultural values (Hofstede et al., 2010) (e.g., uncertainty avoidance, individualism) as cultural background, since they shape conflict-handling (Caputo et al., 2019) and underlie surface customs and religion (Guo, 2025).

|Simulator|DeepSeek -V3.2|Gemini-3.1 -Pro<br><br>|GPT-5.4|Gemini-3.1 -FL|Qwen3 -235B|GPT-5.4 -mini|Qwen3 -30B|Kripp.’s 𝛼 (IAA)|
|---|---|---|---|---|---|---|---|---|
|Persona|87.2|86.9|80.4|75.0|74.7|72.5|70.4|0.75|

Table 1: Simulation fidelity for persona fidelity (accuracy (%) via A/B comparison based evaluation)

#### 3.4.2 Automatic Evaluation

Per-turn LLM judges score every topic at every turn (Liu et al., 2025c), yet only a few topics are actively contested at any given turn while the rest stay inactive, so scoring inactive topics injects noise from irrelevant content (Koo et al., 2024; Ye et al., 2025) and compounds errors along the trajectory. We instead localize scoring to the turns that move each topic. For each topic 𝑇𝑗, the judge reads the dialogue once and locates the turns where 𝑇𝑗 is actively in play, those at which it is discussed or a party shifts position. At each located turn it records an agreement score and each party’s current stance, and turns that do not touch 𝑇𝑗 inherit the prior score. The full trajectory is thus recovered in a single judge pass after the conversation ends, with DeepSeek-V3.2 as the backbone. Automatic evaluation prompts in Appendix E.

We validate this evaluator against expert raters in §4, where it reaches a Pearson 𝑟 = 0.82 with experts, more than doubling both ProMediate (Liu et al., 2025c) and a non-expert baseline.

### 4 Validation of SoCRATES

Two components of SoCRATES require empirical validation before benchmarking: (i) the disputant simulators must actually produce the prescribed persona variations3, and (ii) the topic-localized evaluator must trace trajectories reliably. We validate (i) by checking whether the persona scalar steers party behavior as intended, and (ii) via alignment with human expert judgments.

Simulation Fidelity. SoCRATES uses a float-valued intensity scalar when expanding each party persona, and we ask whether varying this scalar steers agent behavior. We operationalize the check through reactiveness, the persona dimension governing emotional escalation. To probe intensity controllability beyond the binary, we test four scalar levels {0,0.33,0.66,1} and check whether agents preserve this scale across simulated conversations.

We evaluate seven strong simulators, drawn from the mediator pool and supplemented with updated backbones (GPT-5.4, Gemini-3.1-Pro), to isolate persona controllability from weak simulator failure. Following the protocol of Choi et al. (2026), we sample two levels at random from the four-level grid, pair each against a third randomly chosen reference, and generate the conversations with reference’s persona held fixed. Human annotators select the more reactive side in each pair, and higher annotator accuracy indicates more faithful intensity control (see Appendix F.1 for annotation details).

This yields 160 A/B pairs per simulator, annotated by three crowdworkers with Krippendorff’s 𝛼 = 0.75. DeepSeek-V3.2 achieves the highest score (Table 1), indicating that the float-valued persona reliably translates into ordered reactiveness.

Topic-localized Evaluation. We test whether the topic-localized evaluation tracks expert judgment. Since humans recognize consensus only after a claim has been met with a response (Clark and Brennan, 1991), per-turn human annotation would inject ambiguity. We instead aggregate the evaluator’s per-turn trajectory into snippets, single back-and-forth exchanges, and have experts annotate at this unit. Aggregation preserves any per-turn evaluator error while matching the resolution at which experts can rate reliably. Two expert annotators rate 1,844 snippets from 144 mediator trajectories, sampled to ensure balanced coverage across domains and models under the same 1–5 rubric as the evaluator (see Appendix F.2 for annotation details), reaching inter-annotator agreement of 𝛼 = 0.86.

|Evaluator<br><br>|Trajectory level|Outcome level|
|---|---|---|
|Non-expert ProMediate SoCRATES|0.331 (0.000) 0.372 (0.000) 0.823 (0.000)|0.527 (0.000) 0.432 (0.000) 0.801 (0.000)|

Table 2: Evaluator alignment with experts (Pearson 𝑟). The values in parenthesis represent p-values.

3We focus on persona fidelity because, for remaining axes, the perturbations are structural by construction or supported by prior validations of strategic posture (Liu et al., 2025c) and cultural persona (Dey et al., 2025).

###### Intervention Timeliness Intervention Effectiveness Consensus Gain

Type Mediator

Trans Heal Env B2B Pol Intl Legal Intra Avg. Trans Heal Env B2B Pol Intl Legal Intra Avg. Trans Heal Env B2B Pol Intl Legal Intra Avg.

Gemini-3.1-FL 81.2 84.1 78.2 81.8 82.9 81.4 72.9 84.4 80.9 33.6 27.8 16.7 23.5 30.3 19.5 29.4 16.1 24.6 52.1 47.7 25.9 34.6 36.0 22.0 26.7 18.8 33.0 GPT-5.4-mini 80.7 81.9 82.3 76.3 78.2 77.2 78.6 84.3 79.9 34.9 18.9 24.6 23.3 22.5 21.2 32.3 18.8 24.6 55.6 23.6 35.0 32.0 28.2 30.3 41.2 29.5 34.4

Prop.

DeepSeek-V3.2 76.1 76.6 77.1 74.6 76.8 75.2 76.3 73.8 75.8 32.1 19.4 17.3 22.2 28.0 21.6 30.4 13.8 23.1 53.3 41.2 27.6 26.4 35.4 26.6 27.0 17.8 31.9 Qwen3-235B 71.7 79.7 77.1 76.1 77.2 73.5 77.1 78.6 76.4 34.0 24.2 15.2 22.1 31.5 25.9 28.0 16.0 24.6 51.0 29.7 22.8 28.2 32.5 33.8 20.7 26.9 30.7 Nemotron-3-120B 70.1 70.7 74.1 69.3 71.5 70.9 73.6 75.7 72.0 29.4 25.2 11.3 19.1 17.2 18.5 17.7 15.4 19.2 41.9 41.1 16.7 14.5 15.8 17.7 7.0 8.3 20.4 Solar-Pro-3 83.0 86.9 84.4 84.5 85.0 82.4 85.2 85.9 84.6 24.5 21.8 13.2 17.8 15.9 14.4 16.7 9.1 16.7 41.8 30.1 24.3 28.3 6.6 13.4 6.0 8.7 19.9 Gemma-4-26B 79.9 81.5 74.2 79.1 81.6 81.3 74.9 79.5 79.0 29.8 20.5 16.0 12.3 14.3 17.1 25.3 9.4 18.1 42.9 22.9 24.6 15.8 7.1 15.9 24.4 14.6 21.0 Qwen3-30B 84.2 85.2 84.3 85.6 85.1 82.2 83.9 86.4 84.6 19.1 26.9 18.8 17.6 18.6 17.7 24.4 14.5 19.7 -7.9 48.6 26.3 16.0 17.9 18.1 -1.2 8.2 15.7

Open-Source

Average 78.4 80.8 79.0 78.4 79.8 78.0 77.8 81.1 79.2 29.7 23.1 16.6 19.7 22.3 19.5 25.5 14.1 21.3 41.3 35.6 25.4 24.5 22.4 22.2 19.0 16.6 25.9

- Table 3: Conflict resolution performance of the eight mediators across eight domains: Trans (Transactional), Heal (Healthcare), Env (Environmental), B2B (Business-to-Business), Pol (Public-Policy), Intl (International), Legal (Legal), and Intra (Intra-organizational). Cell color intensity increases within each column to indicate higher scores.

We compare SoCRATES against two baselines on the same 1–5 scale: ProMediate’s LLM judge, which scores every topic at every turn regardless of relevance, and a Non-expert rater performing the same task as the experts. We measure alignment with the average expert score using Pearson correlation 𝑟 at two granularities: trajectory-level (all snippets) and outcome-level (the final snippet). The two views complement each other, as intervention quality metrics depend on conflict trajectory while consensus gain depends on the final state.

The topic-localized evaluator achieves the strongest alignment with experts, reaching 𝑟 = 0.82 on trajectories and 𝑟 = 0.80 on outcomes (Table 2), more than doubling both baselines on trajectories. This result remains consistent under another backbone (see Table 6 in Appendix F.2). Without localization, per-turn baselines distort the consensus trajectory (see Figure 5 in Appendix F.2).

### 5 Benchmarking LLM Mediators

We benchmark eight LLM mediators with SoCRATES: two proprietary models, GPT-5.4-mini and Gemini-3.1-Flash-Lite, and six open-source models, Gemma-4-26B-A4B-it, Qwen3-30B-Instruct, Solar-Pro-3, Nemotron3-120B-A12B, DeepSeek-V3.2, and Qwen3-235B-Instruct. This set spans two axes, proprietary versus open-source and large versus small. At every party turn, the mediator outputs a binary decision whether to intervene. When it does, it inserts a single utterance before the next party speaks; otherwise the dialogue proceeds uninterrupted. This loop repeats until termination. Each mediator runs once on every scenario-condition pair, 40 scenarios × 15 conditions = 600 runs per mediator and 4,800 in total, each paired with its no-mediator baseline. See Appendix G for the mediator prompts.

#### 5.1 Performance by Conflict Domain

Overview. Table 3 reports three metrics per mediator across eight domains. Social conflict resolution remains challenging for every benchmarked LLM, including proprietary frontier models. Average consensus gain caps at 34.4, splitting mediators into a top tier (30.7–34.4) and bottom tier (15.7–21.0). The split holds across domains, where means range from 41.3 to 16.6 and none clears half the unmediated gap. This gap reflects the domain diversity and social adaptation demands of SoCRATES, in sharp contrast to prior works showing a resolution rate of 80–90% in unconditioned, single domain settings (Kwon et al., 2025; Chen et al., 2026).

Proprietary leads, scale alone does not. The two proprietary mediators achieve higher consensus gain than the strongest open-source by 1.1–2.5 points and lead in six of eight domains. This gap persists even as open-sources close gaps on reasoning benchmarks such as AIME25 (Dekoninck et al., 2026). Within a family, scale helps. Qwen3-235B nearly doubles Qwen3-30B’s gain. Across families, however, scale does not order the field. Nemotron-3-120B trails the smaller Gemma4-26B on Legal and Intra-organizational despite comparable problem-solving (Chandiramani et al., 2026). Together, these results show that general capability does not directly translate to mediation, and the residual gap depends strongly on the conflict domain.

Timeliness without effectiveness. Solar-Pro-3 and Qwen3-30B post the highest intervention timeliness yet rank low on consensus gain. They intervene too often without meaningfully affecting the outcome (see Appendix H.1). Intervention effectiveness, in contrast, aligns with consensus gain. The three

General Condition (GEN)

[Figure 42]

[Figure 43]

[Figure 44]

Cultural Adaption (CUL)

Strategic Adaption (SA)

(a) Gemini-3.1-FL. (b) GPT-5.4-mini. (c) Deepseek-V3.2. (d) Qwen3-235B.

[Figure 45]

Multi-state Tracking Long-context (MS)

Emotional Regulation (EMO)

Understanding (LONG)

(e) Nemotron-3-120B. (f) Solar-Pro-3. (g) Gemma4-26B. (h) Qwen3-30B.

- Figure 2: Mediator adaptation across general condition and five socio-cognitive axes, measured by consensus gain.

[Figure 46]

(a) Strategy-wise Analysis. (b) Emotion-wise Analysis. (c) Culture-wise Analysis.

- Figure 3: Consensus gain shift from the general (unperturbed) condition along three axes: (a) strategic posture, (b) emotional reactivity, and (c) cultural identity. Negative values indicate degradation, positive values improvement.

mediators tied at 24.6 hold the top three consensus gain scores. A good mediator must intervene at the right moments and with the right content, as timeliness alone does not resolve conflict.

Domain coverage shapes the verdict. Intervention timeliness is stable across the eight domains, whereasconsensusgainswingsfrom41.3onTransactionalconflictdownto16.6onIntra-organizational disputes. The easy end coincides with where prior conflict resolution datasets concentrate, since Transactional conflict corpora dominate existing testbeds such as CaSiNo (Chawla et al., 2021), CraigslistBargain (He et al., 2018), and KODIS (Hale et al., 2025). A benchmark restricted to transactional conflict overstates mediation ability, making it essential to evaluate how mediators adapt across diverse conflict domains.

#### 5.2 Socio-cognitive Adaptation Analysis

We use the five independently perturbed socio-cognitive axes to localize which abilities constrain each mediator. Figure 2 profiles each mediator across the general condition and the five axes.

Highlight. On four of the five axes, area grows with model capability, with the proprietary models and Qwen3-235B enclosing the largest regions, yet every mediator contracts on at least one axis. Even within the top tier with comparable overall consensus gain, GPT-5.4-mini and DeepSeek-V3.2 lose far more under Multi-state Tracking than Gemini-3.1-FL and Qwen3-235B. Mediation competence therefore comprises distinct socio-cognitive abilities, and current LLMs exhibit uneven profiles rather than a single capability frontier.

#### 5.2.1 Strategy, Emotion, and Culture Shifts

The uneven model profiles motivate a closer analysis of axes. We therefore measure how consensus gain shifts from the general (unperturbed) condition when strategic posture, emotional reactivity, or cultural identity varies, as summarized in Figure 3.

Strategy. Strategic posture is the sharpest stress test. All non-collaborative postures reduce consensus gain, with the most severe drops under Competing (18.9–64.1) and Accommodating (13.8–66.8). Qwen3-235B suffers the largest drops in both settings despite its high overall ranking, indicating that adversarial or one-sided conflicts demand a capability that aggregate scoring does not capture.

[Figure 47]

InterventionEffectiveness

InterventionEffectiveness

InterventionEffectiveness

[Figure 48]

(a) General Condition. (b) Strategy Adaptation (Competing). (c) Multi-state Tracking.

[Figure 49]

InterventionEffectiveness

InterventionEffectiveness

InterventionEffectiveness

(d) Long-context Understanding. (e) Emotional Regulation (React-React). (f) Cultural Adaptation (CN-KR).

- Figure 4: Intervention Effectiveness over conversation progress, where turns are mapped to a 0–100% scale to align varying turn counts, across the general condition and each hard condition from five socio-cognitive axes.

Emotion. Emotional reactivity produces a smoother but consistent degradation. When both parties are composed, several mediators hold their general score. When both are reactive, every mediator drops. The magnitude does not follow model size, indicating that absorbing emotional volatility, rather than raw scale, separates mediators on this axis.

Culture. Cultural identity produces the smallest but most systematic shifts, with mediator scores declining as cultural distance from U.S. norms grows. From a Hofstede perspective, all LLM mediators appear robust on U.S.-anchored values but weaker on East Asian ones, where collectivist orientation and power distance shape the dynamics differently.

#### 5.2.2 Intervention Timing Adaptation

Axis-level results show how much consensus changes, but not when interventions help. We thus analyze timing in Figure 4, which plots intervention effectiveness over normalized conversation progress for the general condition and each hard socio-cognitive condition. Since intervention effectiveness ranges differ across conditions, we read each panel as a within-condition timing profile.

The best intervention window moves with the condition. For Strategy Adaptation or Emotional Regulation, effectiveness rises early and falls off, since mediators must reframe stances or cool emotion before they harden. For Multi-state Tracking or Long-context Understanding, effectiveness instead grows toward later turns, when complex contexts make late moves like summarization more useful. Across mediators, the key distinction is whether they follow these timing windows. Stronger mediators peak near each condition’s window—GPT-5.4-mini in Strategy and Emotion, Qwen3-235B in Multi-state and Long-context—while weaker ones trace flatter curves, failing to adapt their timing as the conflict evolves. Effective mediation thus requires adapting timing to the socio-cognitive demands faced in conflict to maximize impact.

### 6 Conclusion

We presented SoCRATES, a benchmark probing LLM mediators along eight domains and five sociocognitive axes, built on automatic scenario construction and a topic-localized evaluator. SoCRATES shows that conflict resolution remains challenging for LLMs and that performance shifts across context and party compositions. This indicates that effective mediation hinges on adaptation, not uniformity, and SoCRATES provides the testbed to study it.

### Limitations

While SoCRATES provides a controlled testbed for evaluating LLM mediation across domains and socio-cognitive conditions, several limitations remain. First, the benchmark currently runs all conversations in English, even when parties are assigned different cultural identities. This design isolates cultural values from language variation and keeps simulator behavior comparable across conditions, but it does not test multilingual mediation. Extending SoCRATES to multilingual settings

would reveal how language choice, translation ambiguity, and language-specific politeness norms affect mediator behavior.

Second, SoCRATES focuses on consensus as the primary outcome, since consensus is directly tied to whether a settlement is reached and can be scored consistently across domains. However, mediation quality also involves party satisfaction (Hale et al., 2025), procedural fairness, trust restoration, and emotional repair. These dimensions depend on subjective party perceptions and are therefore harder to validate reliably, but incorporating well-calibrated rubrics for them would provide a more comprehensive evaluation of LLM mediators. We leave these extensions as future work.

### Ethical Considerations

We design SoCRATES as a simulation study in which LLM agents role-play conflicts, so no real people are involved as disputants in this process. The scenarios are synthesized by LLM agents from deep-research seeds, with any residual references to specific individuals, organizations, or locations anonymized by the agents before the scenarios enter the benchmark. We recruit crowd-sourced annotators and supervised graduate annotators solely for evaluator validation and persona-fidelity verification, and no other human subjects participate in social conflict simulations. Crowd-sourced annotators receive compensation above the U.S. federal minimum wage rate, while expert examiners were compensated at rates exceeding $35 per hour.

### References

Federico Bianchi, Patrick John Chia, Mert Yuksekgonul, Jacopo Tagliabue, Dan Jurafsky, and James Zou. How well can llms negotiate? negotiationarena platform and analysis. In ICML, 2024.

Daniel Bowling and David Hoffman. Bringing peace into the room: The personal qualities of the mediator and their impact on the mediation. Negotiation Journal, 2000.

Andrea Caputo, Oluremi B Ayoko, Nii Amoo, and Charlott Menke. The relationship between cultural values, cultural intelligence and negotiation styles. Journal of Business Research, 2019.

Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, et al. Nemotron 3 super: Open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2604.12374, 2026.

Kushal Chawla, Jaysa Ramirez, Rene Clever, Gale Lucas, Jonathan May, and Jonathan Gratch. Casino: A corpus of campsite negotiation dialogues for automatic negotiation systems. In NAACL, 2021.

Junjie Chen, Haitao Li, Minghao Qin, Yujia Zhou, Yanxue Ren, Wuyue Wang, Yiqun Liu, Yueyue Wu, and Qingyao Ai. Simulating dispute mediation with llm-based agents for legal research. In AAAI, 2026.

Jeonghwan Choi, Jibin Hwang, Gyeonghun Sun, Minjeong Ban, Taewon Yun, Hyeonjae Cheon, and Hwanjun Song. What makes a sale? rethinking end-to-end seller–buyer retail dynamics with llm agents. arXiv preprint arXiv:2604.04468, 2026.

Herbert H. Clark and Susan Brennan. Grounding in communication. In Perspectives on socially shared cognition, 1991.

Jasper Dekoninck, Nikola Jovanović, Tim Gehrunger, Kári Rögnvalddson, Ivo Petrov, Chenhao Sun, and Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. arXiv preprint arXiv:2605.00674, 2026.

Kaustubh Deshpande, Ved Sirdeshmukh, Johannes Baptist Mols, Lifeng Jin, Ed-Yeremai HernandezCardona, Dean Lee, Jeremy Kritz, Willow E Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. In Findings of ACL, 2025.

Morton Deutsch, Peter T Coleman, and Eric C Marcus. The handbook of conflict resolution: Theory and practice. John Wiley & Sons, 2011.

Priyanka Dey, Yugal Khanter, Aayush Bothra, Jieyu Zhao, and Emilio Ferrara. Can llms express personality across cultures? introducing culturalpersonas for evaluating trait alignment. In Findings of EMNLP, 2025.

Roger Fisher, William L Ury, and Bruce Patton. Getting to yes: Negotiating agreement without giving

in. Penguin, 2011. Google DeepMind. Gemini 3.1 flash lite model card. March 2026a. Google DeepMind. Gemini 3.1 pro model card. Feb 2026b. Google DeepMind. Gemma 4 model card. April 2026c. Boyu Gou, Zanming Huang, Yuting Ning, Yu Gu, Michael Lin, Weijian Qi, Andrei Kopanev, Botao

Yu, Bernal Jimenez Gutierrez, Yiheng Shu, et al. Mind2web 2: Evaluating agentic search with agent-as-a-judge. In NeurIPS, 2026.

Weihong Guo. Conflict resolution in intercultural communication: strategies for managing cultural conflicts. Humanities and Social Sciences Communications, 2025.

James Anthony Hale, Sushrita Rakshit, Kushal Chawla, Jeanne M Brett, and Jonathan Gratch. Kodis: A multicultural dispute resolution dialogue corpus. In NAACL, 2025.

He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. Decoupling strategy and generation in negotiation dialogues. In EMNLP, 2018.

Geert Hofstede, Gert Jan Hofstede, and Michael Minkov. Cultures and Organizations: Software of the Mind. McGraw-Hill Professional, 2010.

Tianyu Hu, Zhen Tan, Song Wang, Huaizhi Qu, and Tianlong Chen. Multi-agent debate for llm judges with adaptive stability detection. In NeurIPS, 2026.

Dayeon Ki, Rachel Rudinger, Tianyi Zhou, and Marine Carpuat. Multiple llm agents debate for equitable cultural alignment. In ACL, 2025.

Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. Benchmarking cognitive biases in large language models as evaluators. In Findings of ACL, 2024.

Deuksin Kwon, Emily Weiss, Tara Kulshrestha, Kushal Chawla, Gale Lucas, and Jonathan Gratch. Are llms effective negotiators? systematic evaluation of the multifaceted capabilities of llms in negotiation dialogues. In Findings of EMNLP, 2024.

Deuksin Kwon, Kaleen Shrestha, Bin Han, Elena Hayoung Lee, and Gale Lucas. Evaluating behavioral alignment in conflict dialogue: A multi-dimensional comparison of llm agents and humans. In EMNLP, 2025.

Michelle LeBaron. Bridging cultural conflicts: A new approach for a changing world. 2003. Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei

Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Xingyu Bruce Liu, Shitao Fang, Weiyan Shi, Chien-Sheng Wu, Takeo Igarashi, and Xiang’Anthony’ Chen. Proactive conversational agents with inner thoughts. In CHI, 2025b.

Ziyi Liu, Bahar Sarrafzadeh, Pei Zhou, Longqi Yang, Jieyu Zhao, and Ashish Sharma. Promediate: A socio-cognitive framework for evaluating proactive agents in multi-party negotiation. arXiv preprint arXiv:2510.25224, 2025c.

Shuai Ma, Qiaoyi Chen, Xinru Wang, Chengbo Zheng, Zhenhui Peng, Ming Yin, and Xiaojuan Ma. Towards human-ai deliberation: Design and evaluation of llm-empowered deliberative ai for ai-assisted decision-making. In CHI, 2025.

Amogh Mannekote, Bonnie J Dorr, and Kristy Elizabeth Boyer. Agreement tracking for multi-issue negotiation dialogues. arXiv preprint arXiv:2307.06524, 2023.

OpenAI. Introducingdeepresearch. https://openai.com/index/introducing-deep-research/,

2025. OpenAI. Gpt-5 system card. August 2025. OpenAI. Gpt-5.4 thinking system card. March 2026. Sushrita Rakshit, James Hale, Kushal Chawla, Jeanne M Brett, and Jonathan Gratch. Emotionally-

aware agents for dispute resolution. arXiv preprint arXiv:2509.04465, 2025.

Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever hans or neural theory of mind? stress testing social reasoning in large language models. In EACL, 2024.

Lawrence E Susskind, Sarah McKearnen, and Jennifer Thomas-Lamar. The consensus building handbook: A comprehensive guide to reaching agreement. Sage publications, 1999.

Jinzhe Tan, Hannes Westermann, Nikhil Reddy Pottanigari, Jaromír Šavelka, Sébastien Meeùs, Mia Godet, and Karim Benyekhlef. Robots in the middle: Evaluating llms in dispute resolution. arXiv preprint arXiv:2410.07053, 2024.

Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, et al. Webshaper: Agentically data synthesizing via information-seeking formalization. In ICLR, 2025.

Michael Henry Tessler, Michiel A. Bakker, Daniel Jarrett, Hannah Sheahan, Martin J. Chadwick, Raphael Koster, Georgina Evans, Lucy Campbell-Gillingham, Tantum Collins, David C. Parkes, Matthew Botvinick, and Christopher Summerfield. Ai can help humans find common ground in democratic deliberation. Science, 2024.

Kenneth W Thomas. Thomas-kilmann conflict mode. TKI Profile and Interpretive Report, 2008. Upstage AI. Solar pro 3: Better reasoning at production scale. https://www.upstage.ai/blog/

en/solar-pro-3-0127, 2026.

Michelle Vaccaro, Michael Caosun, Harang Ju, Sinan Aral, and Jared R Curhan. Advancing ai negotiations: A large-scale autonomous negotiation competition. arXiv preprint arXiv:2503.06416,

- 2025.

Jincenzi Wu, Yuxuan Lei, Jianxun Lian, Yitian Huang, Lexin Zhou, Haotian Li, Xing Xie, and Helen Meng. Social-r1: Towards human-like social reasoning in llms. arXiv preprint arXiv:2603.09249,

- 2026.

Yang Xiao, Jiashuo Wang, Qiancheng Xu, Changhe Song, Chunpu Xu, Yi Cheng, Wenjie Li, and Pengfei Liu. Towards dynamic theory of mind: Evaluating llm adaptation to temporal evolution of human states. In ACL, 2025.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner

Geyer, Chao Huang, Pin-Yu Chen, et al. Justice or prejudice? quantifying biases in llm-as-a-judge. In ICLR, 2025.

Wenyuan Zhang, Tianyun Liu, Mengxiao Song, Xiaodong Li, and Tingwen Liu. Sotopia-Ω: Dynamic strategy injection learning and social instruction following evaluation for social agents. In ACL, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2023.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. Sotopia: Interactive evaluation for social intelligence in language agents. In ICLR, 2024.

Type Model Model Checkpoint Source Reference

| | | | | |
|---|---|---|---|---|
|Open-source<br><br>|Gemma4-26B-A4B-it Qwen3-30B-A3B-Instruct Solar-Pro-3 Nemotron-3-Super<br><br>-120B-A12B Qwen3-235B-A22B-Instruct DeepSeek-V3.2|google/gemma-4-26B-A4B-it Qwen/Qwen3-30B-A3B-Instruct-2507 solar-pro3-260323 nvidia/NVIDIA-Nemotron-3<br><br>-Super-120B-A12B-BF16 Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V3.2|HuggingFace HuggingFace Upstage<br><br>HuggingFace HuggingFace HuggingFace|Google DeepMind (2026c) Yang et al. (2025) Upstage AI (2026)<br><br>OpenAI (2025) Yang et al. (2025) Liu et al. (2025a)|
|Proprietary|Gemini-3.1-Flash-Lite Gemini-3.1-Pro GPT-5.4-mini GPT-5.4 o4-mini-deep-research|gemini-3.1-flash-lite gemini-3.1-pro-preview gpt-5.4-mini-2026-03-17 gpt-5.4-2026-03-05 o4-mini-deep-research-2025-06-26|Google API Google API OpenAI API OpenAI API OpenAI API|Google DeepMind (2026a)<br>Google DeepMind (2026b) OpenAI (2026) OpenAI (2026) OpenAI (2025)<br>|

Table 4: Backbone LLM configurations for SoCRATES.

### A Scientific Artifacts

Our experiments use publicly accessible LLMs as party, mediator, and evaluator backbones, accessed via the OpenAI and Google APIs for proprietary models and via Hugging Face checkpoints for open-weight models under their respective terms and licenses. The exact checkpoints are listed in Appendix B. Source scenarios are synthesized from deep-research seeds (§3.2) and do not incorporate text from any licensed corpus, so their use is consistent with the intended use of the underlying models and poses no licensing conflict.

### B Model Specifications

Table 4 lists the LLM backbones used across all SoCRATES experiments and pipeline stages: the Searcher (o4-mini-deep-research) for seed collection, the Scenario Writer (GPT-5.4) for recasting and condition-expansion rewrites, the Simulator (party agents, DeepSeek-V3.2) for role-played negotiation, the benchmarked Mediators, and the Evaluator (DeepSeek-V3.2) for topic-localized scoring. The same pool also supplies the Fidelity Simulators for the persona-fidelity validation (§4): the mediator pool plus the two updated backbones GPT-5.4 and Gemini-3.1-Pro, which are used to isolate persona controllability from weak-simulator failure. Proprietary models are accessed via their official APIs and open-source models via Hugging Face checkpoints.

### C Agentic Scenario Construction Details

We provide the prompt details for the three stages of the scenario construction process in §3.2, including seed search, scenario recasting, and the party agent used in the rejection-sampling filter.

Seed Search. Table 10 is the prompt issued to the o4-mini-deep-research agent for each domain, with the domain filling the query field. The agent returns a seed report covering the conflict’s timeline, stakeholders, core issues, institutional tensions, and current status. Table 12 shows one such seed for the Healthcare domain, drawn from a publicly documented hospital-closure dispute.

Scenario Recast. Table 11 presents the recast prompt for the GPT-5.4 scenario writer (temperature = 0), which converts a conflict seed into a structured scenario. The prompt enforces fictional names for every real entity, at most four topics each with a small discrete option set, and diverging party stances with at least one emotionally provocative topic. Table 13 shows the scenario recast from the seed in Table 12; reading the two tables side by side shows that the operator-versus-regulator pairing, the four issue clusters (emergency access continuity, offset investments at receiving hospitals, workforce protections, and accountability for premature service reductions), and the asymmetry between operator financial pressure and statutory regulatory mandate carry over from the real conflict, while all identifying names, dollar figures, and dates are replaced with fictional substitutes.

Preference Weighting. Table 14 is the prompt issued to the GPT-5.4 scenario writer to derive each party’s preference weights W over topics and per-topic stances from its profile. Weights are positive integers summing to 100, and the prompt forbids uniform distributions to ensure a clear priority ordering. The resulting weights and stances enter the party profile and are reused across all

|Axis|# of Conditions<br><br>|Condition Pairings|
|---|---|---|
|General Strategic Posture Party Composition History Length Emotional Reactivity Cultural Identity|1 3 1 1 3 6<br><br>|Unexpanded Condition (Default) Competing, Avoiding, Accommodating Three-party Extended Length (5×) Com-Com, Com-React, React-React US-US, CN-CN, KR-KR, US-CN, US-KR, CN-KR|
|Total|15| |

- Table 5: The 15 conditions per scenario, listed by axis. The general condition is the unexpanded baseline retained from agentic scenario curation (§3.2), while the remaining 14 are produced by applying one socio-cognitive axis to a fresh copy of the scenario.

condition expansions, but the Parties axis is the only case that issues this prompt again, where the original parties keep their general condition weights and only the newly added party receives a fresh assignment.

Party Agent. Table 15 is the prompt for the role-playing party agents (DeepSeek-V3.2, temperature = 0.6) used in both the unmediated rejection-sampling simulation and all downstream conditions. On its turn, an agent emits a private inner thought followed by a public utterance; the inner thought is appended to the party’s private history and is never visible to other parties or to the mediator. A simulation terminates as resolved once every party explicitly signals consensus, or as an impasse when a party emits an impasse signal or the turn budget is exhausted.

### D Socio-Cognitive Probing Details

We detail the implementation of each component described in §3.3 the five condition axes, and the benchmarked mediators used for social conflict resolution benchmarking.

For each axis, we describe only the implementation mechanism and the prompt that drives it; the targeted competency and motivation are in §3.3. Across the five axes plus the unexpanded baseline, every scenario is expanded into the 15 conditions enumerated in Table 5.

Strategies. We append a Thomas-Kilmann mode instruction (competing, avoiding, or accommodating) to the background B; no LLM rewrite. The three conditions differ in this instruction alone.

Parties. Using Table 16, the scenario writer revisits the real-world seed and adds one structurally distinct party with its own role, relation, and per-topic stances. Original parties and topics are carried over verbatim, so the added difficulty comes from tracking more states.

Histories. The scenario writer (Table 17) prepends four dated narrative entries extracted from the seed’s event sequence, with the original background appended unchanged as the final state. This expands the background to roughly five times its default length.

Emotion Control. We append a fixed reactiveness template parameterized by 𝑟 ∈ [0,1] to the party profile, contrasting volatile/escalating behavior at 𝑟=1 with calm/composed behavior at 𝑟=0; no LLM rewrite is used. For condition expansion, we use the two endpoints, composed (C, 𝑟=0) and reactive (R, 𝑟=1), forming three pairings (CC, CR, RR). Intermediate values are validated in §4.

Culture. We anchor each party to a US, CN, or KR identity by appending to its profile a deterministic statement that summarizes the culture’s 0–100 scores across the six Hofstede dimensions (Hofstede et al., 2010). Pairing the three identities yields three intra-cultural and three cross-cultural conditions. All parties interact in English regardless of identity. Each dimension ranges from 0 to 100, with higher scores indicating stronger expression of the named tendency:

- • Power Distance: low scores reflect flat, consensus-oriented decision-making, while high scores reflect acceptance of hierarchical authority and unequal power distribution.

- • Individualism vs. Collectivism: low scores indicate in-group loyalty and collective identity, while high scores indicate personal autonomy and self-reliance.
- • Masculinity vs. Femininity: low scores reflect cooperative, relational values, while high scores reflect competitive, performance-oriented values.
- • Uncertainty Avoidance: low scores indicate tolerance for ambiguity and unstructured situations, while high scores indicate a preference for clear rules and predictability.
- • Long-term vs. Short-term Orientation: low scores reflect adherence to tradition and short-term outcomes, while high scores reflect pragmatic, future-oriented planning.
- • Indulgence vs. Restraint: low scores indicate norm-compliant restraint of desires, while high scores indicate free expression of needs and enjoyment.

The US statement foregrounds individual independence and direct expression. The CN statement emphasizes relational networks, hierarchy, and long-term strategy. The KR statement shares the East Asian long-term orientation but exhibits notably higher uncertainty avoidance and a stronger preference for implicit consensus.

### E Topic-Localized Evaluation Prompts

Table 20 is the topic-localized evaluation prompt. The judge, run at temperature 0, reads the full conversation once, identifies every turn where the topic is actively discussed or a party shifts position, and at each such turn records a 1–5 agreement score together with each party’s stance expressed using the topic’s option labels.

### F Validation Details

This appendix reports the annotation protocols used to validate two components of SoCRATES: persona fidelity in simulated parties and consensus alignment in the topic-localized evaluator. We summarize recruitment, task design, and quality control for each protocol.

#### F.1 Simulation Fidelity Annotations

We focus on persona fidelity because the remaining axes are either structural by construction or externally validated in prior work. Party composition and history length are direct structural perturbations, while strategic posture (Chen et al., 2026; Liu et al., 2025c) and cultural persona realization (Dey et al., 2025) have been validated in prior LLM simulation studies. Other dimensions such as naturalness (Liu et al., 2025c) and instruction adherence (Vaccaro et al., 2025) have likewise been studied and established in prior works.

Annotator Qualification and Compensation. We collect persona fidelity annotations through Amazon Mechanical Turk (MTurk), restricting participation to workers with a HIT approval rate above 90%, at least 500 approved HITs, and a minimum score of 90 on a custom English-comprehension qualification test. Annotators are compensated at $7.50 per hour, above the U.S. federal minimum wage, and no personally identifiable information is collected.

Task Design. Annotators compare two conversations generated from the same scenario, party role, opponent, and topic structure, with only the target party’s reactiveness level changed. They select which dialogue better reflects a reactive rather than composed negotiator, ignoring topic content or persuasion success unless it directly signals emotional reactivity. For each simulator backbone, we sample 160 A/B pairs from the reactiveness grid {0,0.33,0.66,1}; across seven backbones, this yields 1,120 comparisons, each labeled by three annotators. Figure 7 shows the annotation template.

Quality Control. We use majority vote over the three labels for each pair and report fidelity as the fraction of pairs where the selected dialogue matches the higher assigned reactiveness level. Inter-annotator agreement is 𝛼 = 0.75, reflecting the graded nature of emotional expression, but sufficient to distinguish simulator backbones in Table 1.

#### F.2 Consensus Alignment Annotations

Annotator Qualification and Compensation. We recruit two graduate student annotators with strong English proficiency to validate consensus scoring. They are not professional negotiators. Instead, the protocol is supervised by a researcher with a graduate degree in political science and international relations, along with academic training in negotiation and diplomacy, who reviews the rubric, calibrates examples, and resolves procedural questions. As a non-expert baseline, we additionally collect three annotations for each snippet from Amazon Mechanical Turk (MTurk) workers following the same qualification protocol used in Appendix F.1, including a HIT approval rate above 90%, at least 500 approved HITs, and a minimum score of 90 on a custom English comprehension qualification test. Annotators are compensated for the task, no personally identifiable information is collected, and the supervised set contains 1,844 snippets from 144 conversations. The two graduate annotators reach Krippendorff’s 𝛼 = 0.86.

Task Design. We annotate consensus at the snippet level, since agreement is interpretable only after a position receives a response. Each snippet contains one back-and-forth exchange, the background, topics, options, and the preceding snippet. For every issue, annotators record both parties’ option-level positions and assign a 1–5 agreement score. If an issue is not mentioned, annotators carry forward the previous score. Figure 8 shows the interface.

Quality Control. The supervised graduate annotations define the reference for evaluator validation, while the non-expert annotator is retained as a baseline. Because consensus is graded, we average the two supervised annotator scores for each topic-snippet pair rather than forcing a hard adjudicated label. We then compare SoCRATES, the non-expert annotator, and a per-turn LLM-judge baseline by Pearson correlation against this supervised-annotation mean at the trajectory and outcome levels.

Trend Comparison. Aggregate correlations alone cannot reveal whether an evaluator tracks consensus over time. We therefore diagnose evaluators at the trajectory level, tracing how the consensus score changes over snippets and comparing it against expert annotations. A reliable evaluator should follow similar trends, showing upward progress as conflicts move toward resolution. As shown in Figure 5, the topiclocalized evaluator (SoCRATES) tracks the expert’s curve closely, rising from low initial values and preserving the overall shape. Incontrast,ProMediate’sper-turnjudgeproduces an unstable trajectory with large fluctuations between adjacent snippets, starting too high and ending well below the expert’s final score. This instability arises because the per-turn judge scores every utterance against all issues, so inactive topics contribute uninformative scores that obscure the underlying progress. The topic-localized design evaluates only issues that are active at each moment, improving both pointwise correlation with expert judgments and the consensus dynamics underlying trajectory evaluation.

[Figure 50]

(a) ProMediate. (b) SoCRATES.

Figure 5: Trend comparison of consensus score trajectories for ProMediate and SoCRATES. Bold lines show the average trajectory across dialogues, while faint lines in the background depict individual mediation trajectories, illustrating the variability across conversations.

Backbone Robustness. To verify that the evaluator’s reliability is not tied to a specific backbone, we replace DeepSeek-V3.2 with Qwen3235B-A22B-Instruct and re-measure alignment with expert judgments. Table 6 shows that SoCRATES preserves strong alignment under this substitution, confirming that the evaluator transfers across backbones rather than relying on a single model’s behavior.

|Evaluator|Trajectory level|Outcome level<br><br>|
|---|---|---|
|ProMediate SoCRATES|0.423 (0.000) 0.785 (0.000)|0.394 (0.000) 0.721 (0.000)|

Table 6: Evaluator alignment with expert judgments (Pearson 𝑟) using Qwen3-235B-A22B-Instruct as the backbone. Values in parentheses denote pvalues.

### G Mediator Prompts

At every party turn, the mediator, run at temperature 0.6, first executes the when-to-intervene decision (Table 18). If the decision is true, the how-to-intervene generation step (Table 19) emits a single utterance, which is inserted before the next party speaks.

### H Additional Analysis

#### H.1 Intervention Analysis

To diagnose the gap between intervention timeliness and consensus gain, we measure two aspects of mediator behavior: Intervention Frequency (the fraction of party turns on which the mediator chooses to speak) and First Intervention (the relative turn position of the mediator’s first utterance). Table 7 reports both, aggregated across all conditions. Solar-Pro-3 and Qwen3-30B intervene roughly twice as often as the top mediators and begin speaking much earlier in the conversation. This over-eager speaking inflates intervention timeliness without translating into intervention effectiveness or consensus gain, suggesting that early and frequent intervention does not substitute for substantive contribution to social conflict resolution.

|Type|Mediator|IF (%)|FI (%)<br><br>|
|---|---|---|---|
|Prop.|Gemini-3.1-FL GPT-5.4-mini|22.6 22.6<br><br>|32.3 31.0|
|Open Source|DeepSeek-V3.2 Qwen3-235B Nemotron-3-120B Solar-Pro-3 Gemma-4-26B Qwen3-30B|16.1 20.8 14.6 32.3 16.4 31.1|42.8 39.5 45.6 26.9 37.3 25.3|

Table 7: Intervention behaviors of eight mediators. IF: Intervention Frequency, FI: First Intervention.

#### H.2 Benchmark Stability Analysis

We test the benchmark along three axes left fixed in the main results: the evaluator backbone, the party-agent simulator backbone, and run-to-run stochasticity.

Evaluator Backbone Robustness. To check whether the mediator ranking depends on the evaluator backbone, we swap DeepSeek-V3.2 with Qwen3-235B-A22BInstruct and re-evaluate the mediation trajectories from §5 again, holding the disputant simulator fixed. Table 8 reports the average across mediators under both evaluators. The two evaluators yield close averages, differing by only −2.0, +3.9, and +0.6 points across the three metrics. The

|Metric|DS|Qw|Δ<br><br>|𝜌|
|---|---|---|---|---|
|Intervention Timeliness Intervention Effectiveness Consensus Gain|79.2 21.3 25.9|77.2 25.2 26.5|−2.0 +3.9 +0.6|0.406 0.862 0.786|

Table 8: Metric values averaged across mediators under two evaluator backbones (DS = DeepSeek-V3.2, Qw = Qwen3-235B-A22B-Instruct), where Δ reports Qw − DS and 𝜌 denotes the Spearman correlation computed per metric over the per-scenario pairs.

mediator rankings also agree well on intervention effectiveness (Spearman 𝜌 = 0.862) and consensus gain (𝜌 = 0.786). Intervention Timeliness shows weaker agreement (𝜌 = 0.406), because it depends on which turn the trajectory is sampled at and is more sensitive to the evaluator’s choice of relevant turns. We note that Qwen3-235B-A22B-Instruct itself is a weaker evaluator than DeepSeek-V3.2 in our validation (Table 6), which likely accounts for part of this gap. Even so, the relative ordering of mediators is preserved under the alternative evaluator.

Simulator Backbone Robustness. To check whether mediator adaptation across social-cognitive axes depends on the disputant simulator, we replace DeepSeek-V3.2 party agents with Qwen3-235BA22B-Instruct. Due to the cost of simulating disputants and our limited budget, this ablation covers three mediators (Qwen3-235B, DeepSeek-V3.2, Qwen3-30B), while still spanning all 8 situations (600 scenarios) used in the main experiments. Since this ablation includes only three representative mediators, it is not intended to support a full mediator ranking. Instead, our goal is to test whether the gaps across axes identified in §5.2 persist under an alternative simulator.

Intervention Intervention

|Type|Mediator|Timeliness|Effectiveness<br><br>|Consensus Gain|
|---|---|---|---|---|
|Proprietary|Gemini-3.1-FL GPT-5.4-mini|76.3 ± 0.5 78.5 ± 2.4<br><br>|21.2 ± 1.4 20.8 ± 0.3|46.6 ± 2.5 48.4 ± 1.6|
|Open-source|DeepSeek-V3.2 Qwen3-235B Nemotron-3-120B Solar-Pro-3 Gemma-4-26B Qwen3-30B|72.4 ± 4.4<br>73.0 ± 2.1 71.4 ± 2.0 79.6 ± 1.0 71.9 ± 0.5 78.3 ± 0.6<br>|21.4 ± 0.4 24.1 ± 0.4 13.4 ± 3.5 11.7 ± 2.7 17.1 ± 1.8 15.4 ± 1.1|50.0 ± 2.9 55.8 ± 1.7 35.4 ± 6.8 24.5 ± 5.6 44.4 ± 2.7 40.3 ± 1.2|

Table 9: Intervention timeliness, Intervention effectiveness, and consensus gain across three independent runs on the general scenario, reported as median ± half-range.

As shown in Figure 6, absolute consensus gain values shift after the simulator swap, but both the shape of each mediator’s adaptation profile and the distinctions among mediators are largely preserved. Under both simulators, the general condition remains the strongest setting, and performance drops when moving to the perturbed axes. The relative pattern across axes is preserved under the alternative simulator. Cultural adaptation shows a milder decline, whereas tracking multiple party states and using long histories show larger degradations. The three mediators also retain their characteristic profiles under the alternative simulator, rather than collapsing to a common pattern. This consistency suggests that the social and cognitive gaps measured by SoCRATES reflect adaptation limits of each mediator rather than artifacts of a particular disputant simulator.

Simulator=Deepseek-V3.2 Simulator=Qwen3-235B-A22B-Instruct

[Figure 51]

(a) DeepSeek-V3.2. (b) Qwen3-235B. (c) Qwen3-30B.

Figure 6: Mediator adaptation of three mediators under two disputant simulators (DeepSeek-V3.2, solid line; Qwen3-235B-A22B-Instruct, dashed line).

Multi-run Robustness. To estimate variance across multiple runs, we repeat the mediator phase two additional times for all 8 mediators, yielding three independent runs whose variance reflects both the mediator and the disputant simulator. Due to the cost of simulating the disputants and our limited budget, this ablation is limited to the general conditions. Table 9 reports the median and half-range of each mediator across the three runs. On the consensus gain ranking, the three runs yield a Kendall’s 𝑊 of 0.929, indicating strong agreement on the relative ordering of the eight mediators. Six of the eight mediators stay within a half-range of ±3 points across runs, and the remaining variance is concentrated in the lowest-ranked models. Overall, the mediator ranking is robust to repeated runs, confirming that our main findings reflect genuine mediator differences rather than stochastic noise.

###### Seed Search Prompt

User Research real-world conflicts and disputes related to the following query, and provide a detailed conflict analysis. Query. {Query}

Search the web for notable, well-documented real-world conflicts matching this query. Pick one specific, real conflict that would make a rich negotiation simulation, with multiple stakeholders and multiple issues.

Provide.

- 1. The specific conflict chosen, named clearly.
- 2. Timeline of key events, the major milestones and turning points.
- 3. Key stakeholders, the main parties, roles, and interests (3–5 parties).
- 4. Core issues of disagreement, the main points of contention.
- 5. Institutional tensions, organizational, legal, or structural conflicts.
- 6. Current status, where things stand now. Be thorough and factual. Focus on structural aspects suitable for a negotiation simulation.

Table 10: Prompt for seed search.

Scenario Recast Prompt

System

You are an expert negotiation simulation designer. You are given RESEARCH on a real-world conflict case. Use it for structural inspiration (power asymmetries, interdependencies, topic linkages, leverage), but use fictional names for all entities, people, places, laws, and agencies. Task. Design a negotiation simulation scenario as a JSON object, drawing on the real-world conflict case provided by the user. Party design.

- • Design 2 parties inspired by the real conflict’s stakeholder dynamics.
- • name captures the essence of the party (e.g. “Workers Federation”, “City Government”) without real names or acronyms.
- • role specifies (1) the primary objective and target outcome, (2) internal pressures or constraints, and (3) the BATNA if talks fail.
- • relation states who the party aligns or clashes with and on which topics.
- • preferences gives, for each topic, a 1–2 sentence stance. Different parties should hold different stances to create negotiation tension.

###### Topic design.

- • Use only the topics needed to capture the core conflict (up to 4).
- • Provide 2 options for binary positions; more only for inherently multi-level decisions.
- • Options must be concrete and substantively different; use specific numbers where natural.
- • At least one topic must be emotionally provocative and fit naturally into the scenario. User

Research Result. {Seed Scenario} Return only valid JSON with the schema below, with no markdown fencing, preamble, or commentary. {

"title": "string", "background": "string (context, history, stakes;

generic names; specific amounts/timelines)", "topics": [

{"id": "string (short code, e.g. GOV)", "name": "string (full descriptive name)", "description": "string (what is at stake)", "options": [

{"label": "string (letter label)", "description": "string (concrete action)"}]}

], "parties": [

{"id": "string (short code, e.g. WORK)", "name": "string (full name)", "role": "string (objective, constraints, BATNA)", "relation": "string (ally/rival, with whom)", "preferences": {

"TOPIC_ID": "string (1-2 sentence stance)"}} ]

}

Table 11: Prompt for scenario recast.

Conflict Mount Sinai Beth Israel hospital downsizing and closure, New York City (2016–present). Mount Sinai Beth Israel (MSBI) is a roughly 700-bed acute-care hospital in Manhattan’s East Village, operated by Mount Sinai Health System, a private nonprofit serving the Lower East Side, East Village, and Chinatown neighborhoods.

Timeline of key events 2013: Mount Sinai Health System forms through merger with Continuum Health Partners, absorbing Beth Israel Medical Center.

2016-05: Mount Sinai announces a “Downtown Transformation” plan, an approximately $500 million investment to replace the large inpatient hospital with a smaller acute-care facility (∼70 beds) at the Phillips Ambulatory Care site plus an expanded hub-and-spoke outpatient network across downtown Manhattan.

2016–2022: Plan repeatedly delayed under regulatory and community pressure; Community Board 3, local elected officials, and patient advocacy groups raise objections about loss of inpatient psychiatry, addiction services, and 24/7 emergency care access. 2023-09: Mount Sinai files an updated closure plan with the New York State Department of Health to fully shut Beth Israel; community groups respond with Article 78 litigation and public protests. 2024: NY DOH holds public hearings on the revised closure application; preliminary conditions require Mount Sinai to maintain certain emergency and behavioral health services during transition.

Key stakeholders Mount Sinai Health System (MSHS): Private nonprofit operator; reports persistent operating losses at Beth Israel cited at roughly $150M per year; primary objective is to complete the wind-down to stem ongoing losses; BATNA is unilateral filing subject to NY DOH closure-approval authority.

New York State Department of Health (NY DOH): Statewide regulator with statutory Certificate of Need authority over hospital closures; concerned with continuity of essential services, especially behavioral health and addiction treatment for downtown Manhattan. Coalition to Save Beth Israel and Community Board 3: Local advocacy coalition representing patients and residents; demand community benefit agreements and public accountability; pursue Article 78 litigation.

1199SEIU and New York State Nurses Association (NYSNA): Unions representing several thousand affected clinical and support staff; demand placement guarantees within Mount Sinai’s other facilities, retraining funds, and severance protections. Surrounding receiving systems (Bellevue, NYU Langone, NewYork-Presbyterian Lower Manhattan): Hospitals expected to absorb deflected emergency, psychiatric, and inpatient volume; seek capital support from Mount Sinai to expand capacity.

Core issues of disagreement Continuity of emergency and behavioral health access: Mount Sinai favors rapid downsizing; NY DOH and community advocates demand a staged transition with continuing 24/7 emergency department capacity and crisis stabilization, citing Beth Israel’s role as a primary psychiatric receiving site for downtown Manhattan.

Offset investments at receiving hospitals: NY DOH conditions closure approval on Mount Sinai funding emergency-capacity expansion and EMS upgrades at Bellevue and other surrounding systems; Mount Sinai contests the scope and duration of these obligations.

Workforce transition protections: Unions demand redeployment within the Mount Sinai system at equal pay, multi-year retraining funds, and enhanced severance; Mount Sinai offers severance close to statutory minima.

Accountability for premature service reductions: NY DOH and community groups have documented quiet service reductions (inpatient psychiatry, addiction treatment beds, obstetrics) preceding formal approval; demands include public acknowledgment, financial penalties, and community benefit reinvestment from the resulting downtown real estate.

Institutional tensions NY DOH’s Certificate of Need authority over closures conflicts with the operator’s fiduciary autonomy to manage finances. Mount Sinai’s 501(c)(3) nonprofit mission obligations and Medicaid commitments conflict with mounting operating losses. The medical-school affiliation (Icahn School of Medicine at Mount Sinai) complicates residency redistribution timelines. Local elected officials and Community Board 3 carry political weight but hold no formal closure-approval authority. Receiving hospitals operate under separate Certificate of Need processes that lag the closure timeline.

Current status The revised closure plan remains under NY DOH review. Beth Israel continues to operate at reduced inpatient capacity. The replacement ambulatory and acute-care site is partially operational. Active negotiation focuses on the magnitude and duration of offset investments at receiving hospitals, the scope of workforce protections, and accountability measures for documented premature reductions.

- Table 12: Example deep-research seed for the Healthcare domain, returned by the Searcher for a hospital-closure query. This seed is the input the Scenario Writer recasts into the structured scenario in Table 13. Blue marks elements that carry over to the recast under fictional names.

Title Downtown General Wind-Down: Regulator–Provider Bargaining Over Access,

Capacity, and Accountability

Background A private nonprofit system, Regional Health Network (RHN), operates Downtown General Hospital (DGH) in the River District of Eastborough City. RHN reports sustained operating losses of $120–$150 million per year at DGH since 2019, with average inpatient occupancy falling below 45% and increasing reliance on short-stay and outpatient care. In 2016, RHN announced a $520 million “Downtown Transformation” to pivot from a large inpatient footprint to a smaller hub-and-spoke outpatient network; community groups pushed back, arguing the plan would hollow out emergency and behavioral health services. In September 2023, RHN publicly signaled intent to close DGH and filed a formal closure plan with. . .

###### Parties

RHN (Regional Health Network) You are the private nonprofit operator of Downtown General Hospital. Your primary objective is to complete the wind-down while shifting care to a lower-cost outpatient model, protecting system finances, and preserving your brand in the city. Constraints: sustained $120–$150 million/year losses at DGH, bond covenant. . .

SHOA (State Health Oversight Agency) You are the statewide regulator charged with approving the closure and safeguarding access to essential services. Your primary objective is to secure enforceable mitigations that maintain timely emergency and behavioral health access and protect the workforce during transition. Constraints: statutory due-process. . .

###### Topics

ACC (Continuity of Emergency Access and Urgent Care Model) (A) Operate a 24/7 urgent and primary care hub within 0.3 miles of the former emergency department for 5 years, co-locating a behavioral health crisis stabilization unit (6 chairs),. . .

- (B) Operate a 24/7 urgent care for 3 years with 6 observation bays and on-call behavioral health; performance targets include 40-minute average transfer time; $9 million/year RHN. . .
- (C) Operate a 16-hour daily urgent care for 2 years, no observation bays; afterhours coverage by tele-triage and ambulance diversion to nearby hospitals; $5 million/year RHN. . .
- (D) Maintain a micro-hospital at Seaport Pavilion with a licensed 30-bed unit and full-service emergency department for 3 years during transition; RHN funds $45 million in capital. . .

INV (Offset Investments to Expand Regional Emergency Capacity) (A) RHN transfers $70 million in escrowed capital to CityCare Medical Center to add 30 emergency department treatment positions, 8 fast-track bays, and retrofit 2 negative-pressure. . .

- (B) RHN funds $40 million to CCMC for a 20-position expansion plus $10 million to upgrade emergency medical services dispatch, radios, and 4 new ambulances; 5-year service covenant. . .
- (C) RHN establishes a $25 million Community Access Fund for care coordination, behavioral health integration, and transport vouchers; no direct emergency department capital expansion.
- (D) No new investments; rely on existing regional capacity and RHN’s outpatient network to handle deflected demand.

WORK (Workforce Transition Protections) (A) Guarantee placement within 25 miles for at least 85% of affected full-time equivalent roles at equal or higher base pay for 24 months; $12 million retraining fund; $20,000. . .

- (B) Guarantee first-right-of-hire at RHN sites with pay protection for 12 months; severance of 3 weeks per year of service capped at 52 weeks; $6 million retraining fund; private. . .
- (C) Statutory minimum severance only; $2 million training vouchers; no redeployment guarantees; internal reporting.

ACCNT (Accountability and Public Narrative About Premature Service Reductions) (A) RHN’s chief executive issues a public apology acknowledging anxiety and access risks caused by premature reductions; fund a $5 million Community Stabilization Grant program. . .

- (B) Issue a joint statement of shared responsibility with SHOA; appoint an independent reviewer with public quarterly reports for 2 years; provide $1 million in transport vouchers. . .
- (C) Adopt a no-fault, forward-looking compliance plan with internal reporting and SHOA access to records; no apology, no public audit, and no community grants.
- (D) Place RHN on a 24-month compliance probation with $50,000-per-day penalties for missed reporting or performance targets; chief executive testifies at two public hearings; install. . .

##### Table 13: Example scenario from the Healthcare domain, recast from the deep-research seed in Table 12. Blue marks elements paired with the corresponding blue elements in the seed.

Preference Weighting

System

You are an expert negotiation simulation designer. You are given a SCENARIO with its title, background, issues, and parties; each party may carry persona attributes (cultural background, emotion control). Task. For each party, generate (1) preference weights as positive integers summing to exactly 100 across all issues, and (2) a 1–2 sentence stance per issue stating what the party wants and why. Rules.

- • Weights must be positive integers summing to exactly 100; allow uneven distributions and do not spread weight evenly across issues.
- • A party’s structural position, objectives, constraints, and BATNA are the main drivers of both weights and stances. User

Scenario. {Background} Topics. {Topics} Parties. {Parties} Return only valid JSON with the schema below, with no markdown fencing, preamble, or commentary. {

"PARTY_ID": { "weights": {"ISSUE1": int, "ISSUE2": int, ...}, "stances": {"ISSUE1": "wants X because Y...",

"ISSUE2": "prefers Z due to..."}} }

Table 14: Prompt for per-party preference weighting.

Conflict Simulation: party agent (inner thought and utterance) System You are {Name} ({Role}), negotiating with {Opponent}. Your goal is to reach the best possible outcome for yourself. Task. Generate one internal thought (≤2 sentences) reflecting what is on your mind right now, then write your next spoken utterance based on that thought.

- • The thought is your private reasoning; stay grounded in your positions and prior thoughts.
- • The utterance is your actual turn: concise speech based on your current thought and the conversation so far.
- • Think and speak as a real person: fully embody your persona (personality, cultural background) and conflict mode.
- • Negotiate across multiple topics; consider the overall deal, not one topic at a time.
- • Pursue your interests without telegraphing your strategy or priorities directly.
- • Use plain language, with no speaker labels, headers, option labels, or bullet points.
- • Closing protocol: end with [IMPASSE] if no agreement is possible; write [FINAL AGREEMENT] only after the other party has explicitly accepted your exact terms.

User Background. {Background} {Conflict Mode Instruction} Previous thoughts. {Previous Thoughts} Persona. {Party Profile} {Persona Instruction} Your interests (100 pts total). {Preferences} Pursue high-weight topics more and flex on low-weight ones; express interests through questions and framing, not by stating priorities directly. Negotiable topics. {Topics} Conversation. {Conversation Log} Respond only with JSON: { "thought": "your internal thought here",

"utterance": "your spoken utterance here" }

Table 15: Prompt for persona-conditioned party simulation.

###### Parties Expansion

User You are an expert negotiation simulation designer. You are given (1) a base scenario with 2 parties and a set of issues, to preserve as the foundation, and (2) the research on the real-world conflict that inspired it, used to identify additional stakeholders. Task. Expand the scenario from 2 parties to exactly {Targeted Party} parties, preserving the original structure. Party expansion.

- • Keep the original 2 parties exactly as they are; only their relation or preferences may be updated to reference newly added parties.
- • Add exactly {Targeted Party} new part(ies) inspired by real stakeholders in the research, with structurally distinct roles, not subdivisions of existing parties.
- • name captures the party’s essence without real names or acronyms; role specifies its primary objective and target outcome, internal pressures or constraints, and BATNA; relation states whom it aligns or clashes with and on which issues.
- • preferences gives, for each issue, a 1–2 sentence stance following from the party’s role and constraints; different parties should hold different stances to create negotiation tension.
- • Expand, do not replace, the background to introduce the new parties. Issues. Keep all original issues exactly as they are (id, name, description, options); do not modify existing issues or add new ones. Base scenario. {Recast Scenario} Research. {Seed Scenario}

Return only valid JSON with the same schema as the base scenario (title, background, issues, parties), with no markdown fencing, preamble, or commentary.

Table 16: Prompt for party-axis expansion.

###### Histories Expansion

User You are an expert negotiation simulation designer. You are given (1) a base scenario with 2 parties and a set of issues, (2) the research on the real-world conflict that inspired it, and (3) the original background, the current state of affairs the parties negotiate from, which must not be rewritten or replaced.

Task. Expand a series of historical entries that chronologically lead up to the original background. These entries are the pre-history, the events, decisions, and escalations that explain how and why the parties arrived at the current impasse, so that a reader finishing the last entry feels the original background is its natural, inevitable result.

###### Rules.

- • Produce exactly {N Events} history entries; the final entry must hand off directly into the original background.
- • Use fictional, essence-capturing names for all parties; include specific numbers, dollar amounts, percentages, headcounts, dates, and durations.
- • Do not contradict, reproduce, or summarize the original background; stop before it.

Structure. For each key event extracted from the research, write a detailed entry headed ### History [N]: [Title] ([Date]) followed by a rich narrative of what happened, why it mattered, and how it shifted the dynamics between parties. Entries follow chronological order: early entries cover origins and the status quo, middle entries cover triggering events and escalation, and late entries cover the final breakdown that directly sets up the original background.

Base scenario. {Recast Scenario} Research. {Seed Scenario} Original background. {Original Background}

Return only the history entries with their ### History [N] headers, with no JSON wrapping, markdown fencing, preamble, or commentary.

Table 17: Prompt for history-axis expansion.

Mediator: when-to-intervene decision

System

You are the Mediator in a multi-party negotiation. Your role is to decide whether to send a message to the participants at this moment, based on the conversation history and scenario context.

###### Task.

- • Decide whether to send a message to the participants at this moment.
- • Stay sensitive to the social dynamics of the conversation and the participants’ sentiments.
- • Be proactive in offering help, but avoid interrupting the flow following the criteria below.
- • Use plain language, with no turn labels, headers, option labels, or bullet points.

Engagement criteria. Do not engage if the conversation is flowing well, the participants are having a personal exchange, or you are unsure whether engaging is appropriate. Engage if the conversation has stalled or drifted from the goal, there is confusion or misalignment you can resolve, emotional tension is escalating, or a participant asked a question you can help with.

User Background. {Scenario} Negotiable topics. {Topics} Conversation (“Mediator” refers to you). {Conversation Log} Respond only with JSON: { "thought": "reasoning about whether and why to intervene", "should_engage": true/false }

Table 18: Prompt for mediator intervention decision.

Mediator: how-to-intervene generation

System

You are the Mediator in a multi-party negotiation. Your role is to craft an intervention for the participants, based on the conversation history and scenario context.

###### Task.

- • Send a helpful and concise utterance that assists the participants or moves the discussion forward.
- • Stay sensitive to the social dynamics of the conversation and the participants’ sentiments.
- • Be proactive in mediating the conversation and offering helpful guidance.
- • Use plain language, with no turn labels, headers, option labels, or bullet points. User

Background. {Scenario} Negotiable topics. {Topics} Conversation (“Mediator” refers to you). {Conversation Log} Previous thought. {When-to-intervene Thought} Respond only with JSON: { "thought": "reasoning about what to say and why", "utterance": "your spoken utterance here" }

Table 19: Prompt for mediator intervention generation.

Topic-localized Evaluator

System

You are an expert in negotiation analysis. You analyze a full conversation and score the agreement between two parties on a specific topic.

Task. Read the full conversation. Identify every turn where either party meaningfully discusses or shifts its position on this topic. For each such turn, record (1) the agreement score (1–5) based on everything said up to that turn, and (2) each party’s current stance, expressed with the option label(s) above (e.g. “(A)”, “(B)”), or a brief description if it matches no option. Omit turns that do not address the topic; scores carry forward from the last relevant turn.

User Background. {Scenario} Topic to analyze. {Topic} Parties. {Parties} Conversation. {Conversation Log} Return JSON in exactly this format: {

"relevant_turns": [3, 7, 12], "agreement_score": [

{"turn_id": 3, "reason": "brief reason", "score": int, "party_stances": {

- "party_A": "short description for stance",
- "party_B": "short description for stance"}},

{"turn_id": 7, "reason": "brief reason", "score": int, "party_stances": {

"party_A": "(B)", "party_B": "(C)"}} ]

}

Table 20: Prompt for topic-localized evaluation.

[Figure 52]

##### Figure 7: Example of annotation template for pairwise simulation fidelity evaluation.

[Figure 53]

##### Figure 8: Example of annotation template for consensus score evaluation.

