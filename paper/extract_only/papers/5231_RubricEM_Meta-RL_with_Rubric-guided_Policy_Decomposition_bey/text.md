## arXiv:2605.10899v1[cs.CL]11May2026

[Figure 1]

2026-5-12

# RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards

###### Gaotang Li1, Bhavana Dalvi Mishra2, Zifeng Wang2, Jun Yan2, Yanfei Chen2, Chun-Liang Li2, Long T. Le2, Rujun Han2, George Lee2, Hanghang Tong1, Chen-Yu Lee2 and Tomas Pfister2

1University of Illinois Urbana-Champaign, 2Google Cloud AI Research

Training deep research agents—systems that plan, search, evaluate evidence, and synthesize long-form reports—pushes reinforcement learning beyond the regime of verifiable rewards. Their outputs lack ground-truth answers, their trajectories span many tool-augmented decisions, and standard post-training offers little mechanism for turning past attempts into reusable experience. In this work, we argue that rubrics should serve not merely as final-answer evaluators, but as the shared interface that structures policy execution, judge feedback, and agent memory. Based on this view, we introduce RubricEM, a rubric-guided reinforcement learning framework that combines stagewise policy decomposition with reflection-based meta-policy training. RubricEM first makes research trajectories stage-aware by conditioning planning, evidence gathering, review, and synthesis on self-generated rubrics. It then assigns credit with Stage-Structured GRPO, which uses stagewise rubric judgments to provide denser semantic feedback for long-horizon optimization. In parallel, RubricEM trains a shared-backbone reflection meta-policy that distills judged trajectories into reusable rubric-grounded guidance for future attempts. The resulting RubricEM-8B achieves strong performance across four representative longform research benchmarks, outperforming comparable open models and approaching proprietary deep-research systems. Beyond final performance, we perform thorough analyses to understand the key ingredients of RubricEM.

### 1. Introduction

Deep research agents answer complex information-seeking questions by autonomously planning, searching, evaluating evidence, and synthesizing long-form reports. Yet how to train this capability remains unclear: proprietary systems such as Gemini and OpenAI’s deep research (Google DeepMind, 2025; OpenAI, 2025) reveal little about their methodology, while most existing efforts rely on verifiable search proxies (Jin et al., 2025; Nguyen et al., 2025; Song et al., 2025) or high-quality imitation data (Moonshot AI, 2025; Perplexity AI, 2025; Tongyi DeepResearch Team et al., 2025). End-to-end RL for long-form research is difficult because outputs lack ground-truth verification, judge feedback is coarse and delayed over long tool-augmented rollouts, and conventional post-training mostly converts judged attempts into parametric updates without producing explicit reusable guidance. This raises the central question of this work:

How can reinforcement learning train deep research agents beyond verifiable rewards, while enabling long-horizon credit assignment and learning from experience?

[Figure 2]

Rubrics offer a natural handle for open-ended tasks whose quality cannot be verified by exact answers (Chen et al., 2025b; Gunjal et al., 2025; Shao et al., 2025). Prior work mainly uses them as judge-side criteria for assigning rewards to final responses. Our key perspective is that rubrics should instead serve as a shared interface throughout reinforcement learning. The same criteria that define success can guide the agent’s planning and search, support process-level judgment over intermediate decisions, and be distilled into reusable reflections for learning from experience. Based on this view, we propose RubricEM, a rubric-guided reinforcement learning framework that combines stagewise

Corresponding author(s): Gaotang Li <gaotang3@illinois.edu> and Bhavana Dalvi Mishra <bhavana@google.com>. This work was done while Gaotang Li interned at Google Cloud AI Research.

[Figure 3]

- Figure 1 | RubricEM first instills a rubric-guided structured scaffold into the task policy, so that each trajectory follows stage-structured reasoning under self-generated rubrics. During RL, we propose Stage-Structured GRPO to provide finer-grained, denser credit assignment. In parallel, a sharedbackbone reflection meta-policy is jointly trained to generate rubric-grounded reflections, which are stored in a rubric bank to support both cross-episode transfer and within-episode refinement. Together, rubrics serve as the shared interface across the agent, the judge, and their evolutions.

policy decomposition with reflection-based meta-policy evolution. The name RubricEM reflects an Expectation–Maximization (EM)-inspired estimate–maximize view (Dempster et al., 1977) (beyond supervised settings): the latent structure of an open-ended research task—what matters, where credit belongs, and what should be remembered—is estimated through rubrics, which condition policy reasoning, judge scoring, and memory evolution. Training then maximizes the task policy and reflection meta-policy under these rubric-conditioned estimates.

RubricEM first realizes rubric-guided policy decomposition through a rubric-guided reasoning scaffold. During planning, the agent generates task-specific rubrics and carries them through four stages: planning, research, review, and answer synthesis. This converts a flat long-horizon rollout into rubric-conditioned decision stages, where each stage defines both a distinct decision mode and a natural unit for optimization. The scaffold also makes rubrics operational across the training loop: they guide search and synthesis, serve as on-policy references for the judge, and produce structured traces that can be distilled into reusable reflections.

Building on this decomposition, RubricEM assigns credit with Stage-Structured GRPO (SSGRPO). Rather than broadcast a single terminal score to all tokens, SS-GRPO scores Plan, Research, Review, and Answer with stage-specific rubrics. The judge maintains an evolving rubric buffer for each stage, extending prior evolving-rubric evaluation (Shao et al., 2025) from final-answer judging to process-level feedback. These stagewise scores define denser returns that combine local stage quality with downstream impact, giving GRPO finer-grained credit signals while remaining critic-free.

Finally, RubricEM makes experience reuse an explicit RL objective through Reflection MetaPolicy training. The task policy and reflection meta-policy share one backbone: after a task rollout is judged, the backbone samples rubric-grounded reflection candidates conditioned only on the query and raw trajectory, while a separate judge scores these candidates using the task-rollout judgments. The reflection scores provide auxiliary RL rewards on reflection tokens, updating the shared parameters; the highest-scored accepted reflection is also written into an agent rubric bank as natural-language memory. The bank conditions future rollouts in two modes: within-episode refinement retrieves the previous reflection for the same query, while cross-episode transfer retrieves reflections from related

[Figure 4]

[Figure 5]

- S1 Plan
- S2 Research
- S3 Review
- S4 Answer

S1 Plan: deep_analysis → rubrics → research_plan

How do sleep patterns affect

cognitive decline in aging? Plenty of evidence will link poor sleep to cognitive decline – but does disrupted sleep cause decline or is it a symptom of neurodegeneration? I also need to identify which aspect of sleep matters: duration, timing etc.

Task

S2 Research: tool_call → tool_output → state_evaluation

1. The response needs to establish casual directionality – not just that poor sleep and cognitive decline cooccur 2. The response needs to identify which specific aspect of sleep drives the effect, rather than treating `poor sleep` as a single variable 3. The response should not conflate different types of cognitive decline, such as age-related memory loss, Alzheimer’s disease

[Figure 6]

`snippet_search`: sleep cognitive decline aging causal mechanism longitudinal

[Figure 7]

Search result: The glymphatic system clears amyloid-beta from the brain during slow-wave sleep – reduced deep sleep allows plague accumulation …. Uhm, the output looks good, but I need another search with the research plan

- 1. Search for evidence linking sleep to cognitive decline in aging
- 2. Follow up on which sleep components are most predictive

[Figure 8]

###### S4 Answer

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Cognitive decline in aging is driven not by poor sleep generally, but by the loss of deep sleep specifically. During slow-wave sleep, the brain’s glymphatic system flushes amyloid-beta – the protein that forms Alzheimer’s plaques. As deep sleep naturally decreases with age, amyloid accumulates, and the buildup itself further disrupts deep sleep, creating a self-reinforcing cycle. Longitudinal data shows each 1% annual decline in slow-wave sleep is associated with 1.2x increased dementia risk over 17 years. Critically, interventions that enhance slow-wave sleep (acoustic stimulation, timed exercise) can improve amyloid clearance in early stages – but only before the feedback loop becomes self-sustaining…

… SufficientInformation. S3 Review: rubric_review → writing_plan

[Figure 13]

Causal directionality: glymphatic mechanism provides biological pathway; longitudinal data shows deep sleep decline precedes cognitive decline by years Sleep component specificity: …..

[Figure 14]

[Figure 15]

Retrospective rubrics: for health-behavior questions, check whether the relationship forms a self-reinforcing cycle – if so, the critical follow-up is at what stage intervention still break the loop

Structure around the causal chain – lead with mechanism, then address directionality directly, then discuss intervention implications. Avoid a study-bystudy literature review; organize by insights, not by source.

[Figure 16]

[Figure 17]

Takeaway: Trace findings to their most specific actionable form – which subtypes does the mechanism cover, and which variable can you actually intervene on. “Sleep affects cognition” is a correlation. “slow-wave drives amyloid clearance in Alzheimer’s type decline” is an answer.

Specific section outline: Opening (…..) Main Body Paragraph 1 (….) ….

Reflection

[Figure 18]

[Figure 19]

Plan Research Answer

“Causal Framing”: The plan identifies that correlation between sleep and decline is insufficient; demands evidence for directionality “Search Precision”: The research query targets mechanistic and longitudinal evidence specifically, not just correlational studies

[Figure 20]

Stagewise Rubrics

“Mechanistic Completeness”: The answer needs to address whether amyloid clearance is the only pathway, or whether sleep affects cognitive decline through additional mechanisms (memory consolidation, neuroinflammation)

[Figure 21]

- Figure 2 | Example of the rubric-guided stage-structured search agent trajectory, meta-policy reflection, and stagewise judge rubrics during a single RL step.

questions. Thus, each reflection updates the agent both parametrically and textually. We designed an efficient asynchronous reflection branch to train this meta-policy alongside task-policy RL without adding a sequential bottleneck, a notable problem in prior meta-RL literature (Jiang et al., 2026).

Together, these components yield RubricEM-8B, an 8B deep research agent trained with 1400 RL steps. Fig. 1 gives an overview of the framework, and Fig. 2 illustrates a concrete example. Across four representative long-form research benchmarks, RubricEM-8B achieves state-of-the-art performance among comparable open models, improves over strong prior RL systems with fewer training steps, and approaches proprietary deep-research systems such as Gemini and OpenAI Deep Research. Beyond final scores, we conduct extensive ablations and analyses, including multiple 600-step RL ablations, scaffold comparisons, inference scaling, and out-of-domain short-form transfer. These results support a broader recipe for long-horizon RL beyond verifiable rewards: expose task structure, assign credit to that structure, and convert judged attempts into reusable experience.

### 2. Related Work

The Post-training recipes of deep research agents. Most existing open-source training efforts for deep research focus on short-answer tasks with verifiable rewards (Chen et al., 2025a; Han

- et al., 2025; Jiang et al., 2025; Jin et al., 2025; Song et al., 2025; Zhao et al., 2025). Meanwhile, proprietary systems mainly report scaling high-quality imitation data and training on verifiable short-form settings (Google, 2025; Moonshot AI, 2025; OpenAI, 2025; Perplexity AI, 2025). Our work takes an orthogonal direction: making reinforcement learning effective for open-ended, long-form deep research. The closest work to ours is DR Tulu (Shao et al., 2025), which studies end-to-end RL for deep research beyond verifiable rewards. We build on this foundation by introducing fine-grained credit assignment and jointly trained meta-policy evolution, yielding denser learning signals and reusable guidance during the challenging long-horizon RL process.

Credit assignment and meta-RL with language models. Recent work on agentic reinforcement learning has increasingly emphasized the need for finer-grained credit assignment (DeepSeek-AI,

2026; Mousavi-Hosseini and Erdogdu, 2026; Qian et al., 2025; Xi et al., 2026; Zhang, 2026). However, most of these methods operate in verifiable settings, where trajectories can be decomposed into subgoals with reliable process-level supervision. A related line of work trains meta-policies during reinforcement learning, often referred to as Meta-RL (Jiang et al., 2026; Yang et al., 2026a). While promising, these methods are typically evaluated on verifiable or synthetic tasks and often introduce explicit dependencies across rollouts, leading to substantial training overhead. In contrast, our work targets open-ended real-world deep research tasks, where neither intermediate progress nor final answers admit simple automatic verification. We improve meta-policy training efficiency by removing cross-rollout dependencies and designing an efficient reflection-training infrastructure.

### 3. RubricEM

##### 3.1. Preliminaries and notations

We study deep research agents for complex information-seeking queries. Given a query 𝑞 ∼ D, an agent interacts with a tool environment T and produces a trajectory 𝜏 = (𝑎1, 𝑜1, . . . , 𝑎𝑇, 𝑜𝑇), where 𝑎𝑡 denotes the agent emission at turn 𝑡, which may be either a textual segment or a structured tool call, and 𝑜𝑡 is the resulting tool output (with 𝑜𝑡 = ∅ when no tool is invoked). We consider a languagemodel-based agent that autoregressively samples the next step 𝑎𝑡 ∼ 𝜋𝜃(𝑎𝑡 | ℎ𝑡), ℎ𝑡 = (𝑞, 𝑎<𝑡, 𝑜<𝑡), and eventually produces a final long-form answer grounded in retrieved evidence.

##### 3.2. Structured Reasoning Scaffold

A central design choice in RubricEM is to impose explicit stage structure on agent trajectories. A stage refers to a semantically defined segment of the trajectory that serves a distinct decision role, such as planning, evidence gathering, self-evaluation, or final synthesis. In long-horizon research tasks, these stages provide a stable high-level organization over otherwise noisy token-level generation. When these decision modes are collapsed into a flat autoregressive process, the trajectory lacks such stage-level organization. The policy must therefore infer its current decision mode from local context alone, which can lead to inefficient exploration and compounding errors over long horizons (Feng

- et al., 2026; Xu et al., 2025). We formalize the value of explicit stage information as follows. Let ℎ denote a random decision point along a trajectory induced by the current policy, let 𝑐 = 𝜙(ℎ) denote a compressed state representation, let 𝑧 denote the current stage label, and let 𝑈(ℎ, 𝑎) denote the expected downstream value of taking action 𝑎 at history ℎ and then continuing the rollout.

Theorem 1 (Value of stage information). Under mild assumptions in Assump. 1, define

𝑉flat := 𝔼 max 𝑎∈A

𝔼[𝑈(ℎ, 𝑎) | 𝑐] , 𝑉stage := 𝔼 max 𝑎∈A

𝔼[𝑈(ℎ, 𝑎) | 𝑐, 𝑧] .

If there exists a measurable set C0 with positive probability and two task-relevant stages such that for every 𝑐 ∈ C0, 𝑝(𝑧 | 𝑐) > 0, 𝑝(𝑧′ | 𝑐) > 0, and that argmax𝑎∈A 𝔼[𝑈(ℎ, 𝑎) | 𝑐, 𝑧] ∩ argmax𝑎∈A 𝔼[𝑈(ℎ, 𝑎) | 𝑐, 𝑧′] = ∅. Then

𝑉stage > 𝑉flat.

Theorem 1 identifies when explicit stage information is beneficial. In long-horizon research trajectories, the same local context may call for different actions across planning, searching, reviewing, and final synthesis. When these stage-specific optimal actions disagree, a flat policy acts under an aliased context, whereas a stage-aware policy can condition on the current decision mode. This yields a strict value improvement on any positive-probability set where such aliasing occurs. We

[Figure 22]

- Figure 3 | Rubric-guided structured reasoning scaffold in RubricEM. The agent follows a high-level workflow of Plan, Research, Review, and Answer, each with a lightweight internal schema. Rubrics are generated in Plan and then guide evidence collection, evaluation, and final writing. During Research, the agent iterates between tool use and state evaluation; once evidence is sufficient, it enters Review and then produces the final grounded Answer. The full description is in Appen. B.1

therefore make stage structure explicit rather than implicit in a flat trajectory. The proof is deferred to Appen. E.1.

Specific stage instantiation. We instantiate this idea with four rubric-guided stages:

Plan → Research → Review → Answer. Each stage is marked by a stage-level XML tag with a lightweight internal schema. The outer scaffold is sequential, while Research allows local iteration and in-place plan revision. The overview is in Fig. 3 and detailed below:

Plan. Within <structured_plan>, the agent analyzes the user’s explicit and implicit needs in <analysis>, translates them into <rubrics>, and then proposes a concrete <research_plan>. The rubrics specify (i) a knowledge checklist of information to gather, (ii) analytical criteria for the final write-up, and (iii) negative constraints on what the answer should avoid.

Research. The agent iteratively issues <call_tool> actions. After each tool response, the agent performs a <state_evaluation> step, which compares the accumulated evidence against the plan and rubrics, decides whether further search is needed, and optionally revises the Plan in place.

Review. Within <review>, the agent maps collected evidence back to the rubrics through <rubrics_review> and prepares a writing plan, including the main thesis and section outline.

Answer. Within <answer>, the agent synthesizes the final long-form response with citation support.

Importantly, rubrics are not merely evaluation artifacts in RubricEM: they are generated in Plan, can be revised during Research, and guide subsequent stages throughout the trajectory. This instantiation is motivated by three considerations. First, it is task-aligned: deep research naturally involves planning, evidence acquisition, self-evaluation, and synthesis, so the four stages reflect the task rather than an arbitrary template. Second, explicit criteria give the policy a stable target for planning, self-checking, and feedback, echoing rubric-based learning and explicit-principle approaches (Bai et al., 2022; Panadero, 2017; Tai et al., 2018). Third, because self-generated rubrics vary across rollouts of the same query, they provide per-rollout references that help the judge discover more aligned and discriminative stagewise criteria. We validate the effectiveness of the scaffold and its importance for RL in Sec. 5.2.

Finally, the scaffold directly enables our later RL design: stage boundaries define the units for

SS-GRPO credit assignment, and rubric-conditioned traces define the memory format for the rubric bank (Sec. 3.4). We therefore view the scaffold not as a formatting warm-up, but as an SFT-induced structural prior that prepares the policy for effective RL (Zhang et al., 2026).

SFT distillation. To instantiate the scaffold in the policy, we perform teacher–student distillation from Gemini-3.1-Pro. For each query, the teacher is prompted to produce a stage-structured trajectory that follows the XML schema above. Because raw teacher traces do not always obey the target scaffold, we apply rejection sampling to discard outputs that violate stage boundaries, tool-calling syntax, citation format, or grounding constraints. The resulting SFT corpus teaches Qwen3-8B not only tool use and evidence citation, but also the stage discipline and rubric conditioning required by our later RL design. We defer details of the data-generation and filtering pipeline to Append. B.2.

##### 3.3. Stage-Structured GRPO

Building on the structured scaffold above, we propose Stage-Structured GRPO (SS-GRPO) for finergrained credit assignment in deep research. Prior work on process supervision and long-horizon agent RL suggests that denser process-level rewards can substantially improve credit assignment (Qian et al., 2025; Tan et al., 2026; Wang et al., 2026; Wu et al., 2026; Yang et al., 2026b). However, in open-ended deep research, we do not have oracle intermediate rewards: the quality of planning, search, review, and synthesis is semantic, task-dependent, and difficult to verify automatically. SSGRPO therefore uses the explicit stage boundaries from Sec. 3.2 together with rubric-guided judging to construct stage-level learning signals, as illustrated in Fig. 4.

Stagewise scores and returns. Given a query 𝑞, we sample 𝑛 rollouts {𝜏𝑖}𝑛𝑖=1 ∼ 𝜋𝜃(· | 𝑞) and partition each into 𝐾 semantic stages; in our instantiation, 𝐾 = 4 corresponding to Plan, Research, Review, and Answer. Let B𝑖,𝑘 be the tokens in stage 𝑘 of rollout 𝜏𝑖, and let 𝑅𝑖,𝑘 ∈ [0, 1] be the LLM-judge score under the corresponding stage rubric. Rather than assign the same final score to all tokens, SS-GRPO uses a causal stage-dependence matrix Λ = (𝜆𝑘,𝑗), with 𝜆𝑘,𝑗 = 0 for 𝑗 < 𝑘 and 𝜆𝑘,𝑘 = 1, and defines 𝐺𝑖,𝑘Λ = 𝐾𝑗=𝑘 𝜆𝑘,𝑗𝑅𝑖,𝑗. Thus each stage keeps its own score while receiving credit from downstream stages it enables. Terminal reward broadcast is considered a special case.

When stage returns help. The benefit of stage returns depends on a simple trade-off: intermediate judging recovers process information omitted by terminal-only rewards, but also introduces judge noise. Appendix E.2, Thm. 3 formalizes this intuition: stage-weighted credit improves the gradient approximation when the recovered intermediate signal outweighs cumulative judge misalignment. Thus SS-GRPO needs no oracle process reward, only sufficiently aligned stagewise judging with bounded noise. This motivates the stagewise evolving-rubric judge below.

Stagewise evolving-rubric judge. As shown in the top panel of Fig. 4, the judge contrasts multiple rollouts for the same query and proposes discriminative rubrics for each stage. The judge maintains a separate rubric buffer for Plan, Research, Review, and Answer, reuses previous high-discrimination rubrics, and removes items that no longer separate trajectory quality. Because the policy trajectories are themselves rubric-guided, the judge can also use trajectory-generated rubrics as references when constructing new judge rubrics, while still scoring trajectories against the judge-side rubric buffer rather than blindly rewarding a rollout’s own self-rubric. This makes the intermediate rewards both stage-local and adaptive to the current policy distribution. Further details are deferred to Append. C

Stagewise normalization and objective. We instantiate SS-GRPO as a critic-free stagewise variant of GRPO by normalizing returns separately within each stage across the rollout group:

𝐺𝑖,𝑘Λ − 1𝑛

𝑛 𝑖′=1 𝐺𝑖Λ′,𝑘

𝐴𝑖,𝑘 =

.

Std𝑖′[𝐺𝑖Λ′,𝑘] + 𝜖

Previous rubrics used as references for new rubric generation

Judge Rubric Buffer

[Figure 23]

S1: Plan

S2: Research S3: Review S4: Answer

| |
|---|

Stores stage 1-4 highlydiscriminative rubrics

###### Task Policy Rollouts …

Judge rubrics generation

|S1 Rubrics|
|---|
|S2 Rubrics|
|S3 Rubrics|
|S4 Rubrics|

[Figure 24]

𝝉𝟏

| | | | |
|---|---|---|---|

[Figure 25]

Keep highdiscriminative rubrics

Update rubric buffer

[Figure 26]

| | | | |
|---|---|---|---|

𝝉𝒏

by contrasting rollouts & referencing trajectory rubrics

Task

Task trajectory stagewise rubric score

[Figure 27]

Randomly sample one task trajectory 𝝉𝒌 + Query

With access to previous rubrics and scores to assess whether reflections align with judges

Meta-Policy Rollouts

[Figure 28]

Judge reflection

[Figure 29]

| | | | |
|---|---|---|---|

[Figure 30]

Reflection 1 Reflection

###### …

Reflection as reference for future questions

Reflection

[Figure 31]

Agent Rubric Bank

Reflection m

[Figure 32]

- Criterion 1: within-episode refinement rubric score
- Criterion 2: cross-episode transfer

Stores agent reflections for the same/future question reference Highest-scored reflection stored in rubric bank

###### Infrastructure: Efficient Meta-policy Training

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Reflection

###### Step N

| | | | |
|---|---|---|---|

Task Policy Rollout, Heavy GPU+API cost

Meta-Policy Rollout, Low GPU cost

Task-Trajectory Update, Low-moderate GPU cost

Meta-Policy Judging, Low API cost

Meta-Policy Update, Low GPU cost

Task-Trajectory Judging, moderate API cost

[Figure 38]

Inefficient Synchronous Pipeline

[Figure 39]

[Figure 40]

Asynchronous Jobs

###### Reflection

No additional wait at all!

Step N-1 Meta-Policy Training Engine Inference Engine

Step N Meta-Policy Rollout

Step N Meta-Policy Judging

Update

- Figure 4 | Detailed RL training pipeline of RubricEM. The top panel expands Fig. 1 with two coupled judge-agent loops. For task-policy training, an LLM judge contrasts stage-structured rollouts to build a buffer of discriminative stagewise rubrics, which provide dense stagewise rewards for SS-GRPO. For reflection meta-policy training, a sampled trajectory and query prompt multiple candidate reflections; the judge scores all candidates using its accumulated rubrics and trajectory scores under within-episode and cross-episode criteria. These scores update the meta-policy, and only the best reflection is written to the agent rubric bank. The bottom panel shows the asynchronous implementation, where reflection rollout, judging, and update run on previous trajectories to avoid synchronous overhead. Details are provided in Appen. C and D; Alg. 1 gives the formal procedure.

All tokens in the same stage block B𝑖,𝑘 share the advantage 𝐴𝑖,𝑘. The resulting objective is

###### ∑︁𝐾

∑︁

###### ∑︁𝑛

1

min 𝜌𝑖,𝑡𝐴𝑖,𝑘, clip(𝜌𝑖,𝑡, 1 − 𝜂, 1 + 𝜂)𝐴𝑖,𝑘 + 𝛽𝐷KL(𝜋𝜃 ∥ 𝜋ref), (1)

LSS-GRPO = −

𝑛

𝑘=1

𝑖=1

𝑡∈B𝑖,𝑘

𝜃old(𝑎𝑖,𝑡|ℎ𝑖,𝑡) . We keep the estimator critic-free because stage supervision is judgedefined, evolving during training, and collected from expensive long-horizon tool-augmented rollouts; adding a learned stage-conditioned critic would introduce substantial additional complexity.

where 𝜌𝑖,𝑡 = 𝜋𝜋𝜃(𝑎𝑖,𝑡|ℎ𝑖,𝑡)

- 3.4. Meta-Policy Training with Reinforcement Learning

Beyond single-trajectory optimization, RubricEM makes experience reuse part of RL. A shared backbone serves as both the task policy and a reflection meta-policy: rubric-guided task rollouts provide judged experience, and the reflection policy is trained with LLM-judge rewards to produce reusable natural-language guidance. Accepted reflections enter an agent rubric bank for future retrieval, giving the agent both parametric RL updates and textual memory updates. This retains the meta-RL goal of improving future rollouts from past experience, while our asynchronous design avoids a sequential rollout–reflection–update bottleneck.

Joint training of the reflection meta-policy. After task-policy rollouts are judged, we sample a query–trajectory pair and prompt the shared backbone to generate multiple reflection candidates,

treating the trajectory as fixed context and backpropagating only through reflection tokens. A privileged LLM judge scores every candidate using the original question, raw trajectory, stagewise rubric scores, and evaluator justifications from task-rollout judging. These scores assess whether each reflection is useful for within-episode refinement and cross-episode transfer; all candidate scores provide RL signals for updating the reflection meta-policy, while only the highest-scored accepted reflection is written into the agent rubric bank. Because the reflection generator and task policy share the same backbone, this reflection-side objective becomes an auxiliary RL signal for the task policy rather than a purely inference-time memory mechanism. Appendix E.3 formalizes the positive-transfer case where judge-scored reflection updates are aligned on average with future task improvement.

Coupled agent–judge co-evolution. The reflection loop is coupled with the stagewise evolving judge from Sec. 3.3. On-policy rollouts expose new criteria and failure modes, which update the judge-side stagewise rubric buffer; the updated judge then scores both task trajectories and reflection candidates. Accepted reflections return to the agent through the rubric bank and condition future rollouts. Thus, the agent evolves through policy and rubric-bank updates, while the judge evolves through rubric-buffer updates rather than parameter updates.

Rubric bank and two modes of adaptation. Each bank item retrospectively distills a completed trajectory into reflection rubrics and takeaways, summarizing what mattered after one trial. Unlike the prospective rubrics generated during planning, bank items encode outcome-aware lessons. An example is shown in Fig. 2. They support two adaptation modes: within-episode refinement, which retrieves a query’s own prior reflection on a repeated attempt, and cross-episode transfer, which retrieves reflections from related past questions. During training, we realize both modes with a two-encounter curriculum: each query is first solved with cross-episode retrieval and later replayed with its newly generated within-episode reflection, requiring no extra rollout dependencies.

Efficient asynchronous execution. As shown in the bottom panel of Fig. 4, a synchronous implementation would block the next task rollout until reflection generation, reflection judging, and the meta-policy update for the current step are finished. We instead allow the reflection branch to lag by one RL step. During step 𝑁, the inference engine runs the heavy tool-augmented task rollouts, while the training engine consumes the prepared reflection batch from step 𝑁 − 1 for the meta-policy update. After the step-𝑁 trajectories are judged, their reflection rollout and judging jobs are launched asynchronously to prepare the reflection batch for step 𝑁 + 1. This one-step staleness trades exact synchrony for higher infrastructure utilization: both inference and training engines remain occupied, and meta-policy training adds effectively no extra wall-clock overhead to the SS-GRPO loop. The detailed descriptions are in Appen. D.

### 4. Experiment

##### 4.1. Experimental Setup

We evaluate RubricEM on four representative long-form benchmarks: HealthBench (Arora et al., 2025), ResearchQA (Yifei et al., 2025), DeepResearchBench (DRB) (Du et al., 2025), and ResearchRubrics (Sharma et al., 2026). Our experimental setup is built upon DR Tulu, where we generally share the same base infrastructure, training datasets, hyperparameters, and evaluation protocols. We use Gemini-flash-grounded Google Search and Semantic Scholar as the search engines. The training data is sampled from diverse public query sources, including realistic search conversations from SearchArena (Miroyan et al., 2025) and research-oriented questions from OpenScholar (Asai et al., 2024). The SFT stage includes both short-form and long-form data, while the RL stage exclusively focuses on long-form queries. Further details on datasets, training and inference setups, evaluation protocols, and the full baseline list are provided in Appen. F.

- Table 1 | The performance comparison between best-performing baselines. Bold numbers indicate the best performance among proprietary and non-proprietary categories.

Model HealthBench ResearchQA DRB ResearchRubrics Average Closed Deep Research Claude-Sonnet Search – 64.3 34.5 – – Perplexity-Sonar (High) – 69.1 40.7 – – Perplexity Deep Research – 75.3 42.3 48.7 – Gemini Deep Research – 68.5 48.8 61.5 – Gemini 3.1 Pro + Search 47.5 74.5 44.4 49.1 53.9 GPT-5 + Search 59.5 78.2 50.7 60.5 62.2 OpenAI Deep Research 53.8 79.2 46.9 59.7 59.9 Fixed Pipeline Deep Research

WebThinker QwQ-32B 36.5 72.8 37.9 42.2 47.4 WebThinker-32B-DPO 39.4 74.2 40.6 41.9 49.0 Ai2 ScholarQA – Claude Sonnet 32.0 75.0 36.1 38.1 45.3

Open Deep Research Models Search-R1-7B -0.1 27.9 9.5 0.0 9.3 WebExplorer-8B 33.7 64.8 36.7 33.4 42.2 Tongyi DeepResearch-30B-A3B 46.2 66.7 40.6 49.5 50.8 DR Tulu-8B (SFT) 38.1 68.5 39.0 38.4 46.0 DR Tulu-8B (RL, 1900 steps) 50.2 74.3 43.4 46.4 53.6

###### Ours

Qwen3-8B + Our Search 24.5 58.4 28.2 24.5 33.9 RubricEM-8B (SFT) 39.0 71.8 43.0 42.8 49.2 RubricEM-8B (RL, 1400 steps) 49.3 74.5 47.8 50.3 55.5

##### 4.2. Main Results

- Tab. 1 compares RubricEM with the strongest baselines from each category following Shao et al.

(2025). We use reported numbers when available, and reproduce the remaining baselines if possible.

RubricEM achieves strong long-form research performance. RubricEM-8B-RL achieves the highest average score among non-proprietary deep research systems in our evaluation, reaching 55.5 with an 8B backbone. It surpasses strong open baselines, including DR Tulu-8B-RL, Tongyi DeepResearch-30B-A3B, and WebThinker-32B-DPO. It also compares favorably with proprietary systems: on the benchmarks where both scores are available, RubricEM-8B-RL outperforms Perplexity Deep Research on average, and it remains within 4.4 average points of OpenAI Deep Research while outperforming it on DRB. These results show that our training recipe can produce competitive long-form research behavior at small model scale.

The RL recipe is both effective and efficient. Starting from the structured SFT checkpoint, RL improves the average score from 49.2 to 55.5, with gains on all four long-form benchmarks. This gain is not simply inherited from the teacher: although RubricEM-8B-SFT is distilled from Gemini-3.1-Pro, the final RubricEM-8B-RL model surpasses it on average. Compared with the closest prior RL system, DR Tulu, RubricEM starts from a stronger SFT checkpoint, reaches a higher final average score, and uses fewer RL steps (1400 vs. 1900). Since the two systems also differ in teacher model and search backend (RubricEM uses a weaker teacher but a stronger search backend), we conduct controlled ablations in Sec. 5 to isolate the contribution of our proposed components.

- 0

- 1

- 2

- 3

- 4

Avg.overSFT∆

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(a) Average Across Benchmarks

0 200 400 600

Training Step

43.0

43.5

44.0

44.5

45.0

45.5

46.0

46.5

Score

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(b) Deep Research Bench

Baseline RL

SS-GRPO

Meta-Policy

RubricEM

0 200 400 600

Training Step

- 39

- 40

- 41

- 42

- 43

- 44

- 45

- 46

Score

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(c) HealthBench

0 200 400 600

Training Step

69.0

69.5

70.0

70.5

71.0

71.5

72.0

Score

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(d) ResearchQA

Figure 5 | Ablation studies of our RL training recipes under a 600-step budget. Each proposed component improves performance, and the full RubricEM recipe performs best.

- 5. Empirical Analysis

0 100 200 300 400 500 600

Training Step

##### 5.1. RL Training Recipes

We ablate the RL components of RubricEM under a fixed 600-step budget, using the same training configuration and initializing every run from the same RubricEM-SFT checkpoint. To reduce evaluation cost, all ablation runs are evaluated on the same fixed random 100-example subset of each benchmark. We compare four recipes: Baseline-RL, standard answer-only GRPO; SS-GRPO, which replaces terminal reward broadcast with stagewise rubric credit; Meta-Policy, which keeps answer-only GRPO but adds reflection meta-policy training and rubric-bank retrieval; and RubricEM (Full), which combines SS-GRPO and Meta-Policy. Results are shown in Fig. 5. Under this matched setting, both SS-GRPO and Meta-Policy improve over Baseline-RL, and the full recipe performs best across benchmarks. This shows that stagewise credit assignment and reusable-experience learning provide complementary gains under the same training budget and compute.

##### 5.2. Structured Scaffolding and Inference-Time Experience Reuse

We further analyze the structured scaffold and inference-time experience reuse in Fig. 6. For scaffolding, we compare structured and unstructured SFT checkpoints, then continue both under matched 600-step RL settings. Fig. 6(a) shows that the structured scaffold improves distillation quality, while Fig. 6(b) shows that it also makes subsequent RL more effective. Without the scaffold, RL gains are small and unstable for 600 steps, suggesting that rubric-conditioned stages provide useful structure for exploration and credit assignment. We additionally isolate the prompt-level effect by running Gemini-3.1-Pro with the same search backend under either our scaffold or a standard ReAct (think & act) prompt. As shown in Fig. 6(c), our scaffold yields higher DRB performance, indicating that the structure itself improves deep-research behavior before student-side training. We also evaluate whether the learned meta-policy can be further leveraged at inference time on DRB. Cross-episode reuse retrieves reflections from related past questions, while within-episode reuse retrieves the agent’s prior reflection for the same question. As shown in Fig. 6(d), RubricEM benefits from both reuse modes, whereas Baseline-RL does not under the same retrieval setting. This indicates that Reflection Meta-Policy Training learns actionable, reusable guidance rather than simply increasing the amount of context provided to the model.

##### 5.3. Short-Form Benchmark Performance

In addition to the long-form benchmarks in Tab. 1, we evaluate RubricEM on four short-form search benchmarks: SimpleQA (Wei et al., 2024), 2Wiki (Ho et al., 2020), WebWalker (Wu et al., 2025b), and DeepSearchQA (DSQA) (Gupta et al., 2026). These evaluations are out-of-domain for the RL

###### (a) SFT Distillation Comparison

###### (b) SFT Scaffold Influence on RL Improvements

###### (c) Gemini-3.1-Pro + Our Search Prompt Comparison

###### (d) Inference-Time Experience Reuse

DRB

0.8

43.0

2.5

43.5

- 39

- 40

- 41

- 42

- 43

- 44

Avg.Scorevs.SFT∆

+0.7

ScorefromPass1∆

0.6

2.0

+0.4

0.4

1.5

DRBScore

39.8

0.2

1.0

RubricEM

0.0

Baseline RL

0.5

70.2 34.2

0.0

0.2

71.8 39.0

39.9

ResearchQA HealthBench

Structured RL

0.5

0.4

Unstructured RL

1.0

-0.5 -0.6

0.6

Cross-Episode (Pass 1)

Within-Episode (Pass 2)

100 200 300 400 500 600

Our Scaffold ReAct

Structured SFT Unstructured SFT

RL Training Steps

- Figure 6 | Ablations on structured scaffolding and inference-time experience reuse. Panels (a,b) show that the rubric-guided scaffold improves both SFT distillation quality and subsequent RL gains. Panel (c) isolates the prompt-level effect: with the same Gemini-3.1-Pro model and search backend, our scaffold outperforms a standard ReAct prompt on DRB. Panel (d) shows that the learned meta-policy enables cross-episode transfer and within-episode refinement, while Baseline-RL does not benefit from the same reuse.

- Table 2 | Short-form Model Performance. Despite being primarily trained on long-form data, RubricEM generalizes well on out-of-distribution short-form deep research questions.

Model SimpleQA 2Wiki WebWalker DSQA Avg. DR Tulu Baseline

DR Tulu-8B (SFT) 75.5 66.5 31.9 5.3 44.8 DR Tulu-8B (RL, 1900 steps) 80.1 68.0 39.1 8.3 49.0

###### Ours

Qwen3-8B + Our Search 84.0 61.5 42.6 15.2 50.8 RubricEM-8B (SFT) 92.1 77.5 64.7 37.0 67.8 RubricEM-8B (RL, 1400 steps) 92.3 78.8 70.0 53.0 73.5

stage, since online RL uses only long-form prompts. We compare with DR Tulu and report results in

- Tab. 2. Although RubricEM is trained primarily for long-form deep research, the result shows strong transfer to short-form search benchmarks. RubricEM-SFT already learns effective search-and-answer behavior, and long-form RL further improves performance despite using no short-form RL data. The gains are most pronounced on more complex tasks, suggesting that our RL recipe teaches transferable tool-use and evidence-grounding skills rather than only long-form report writing. Overall, these results show that a compact 8B model can generalize beyond its long-form training distribution through effective RL.

### 6. Conclusions

We presented RubricEM, a rubric-guided RL framework for deep research beyond verifiable rewards. RubricEM structures long-horizon trajectories, assigns stage-aware credit, and distills judged attempts into reusable guidance through a shared-backbone reflection meta-policy. Across benchmarks and analyses, RubricEM-8B shows strong performance, with ablations supporting each proposed recipe. Further discussions and limitations are included in Appen. H.

### References

Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’arcy, et al. Openscholar: Synthesizing scientific literature with retrieval-augmented lms. arXiv preprint arXiv:2411.14199, 2024.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Jonathan Bragg, Mike D’Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D Hwang, Peter Jansen, Varsha Kishore, et al. Astabench: Rigorous benchmarking of ai agents with a scientific research suite. arXiv preprint arXiv:2510.21652, 2025.

Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Fan Yang, Zenan Zhou, Weipeng Chen, Haofen Wang, Jeff Z. Pan, Wen Zhang, and Huajun Chen. Research: Learning to reason with search for llms via reinforcement learning. arXiv preprint arXiv:2503.19470, 2025a.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, et al. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025b.

Yifang Chen, Kevin Jamieson, and Simon Du. Active multi-task representation learning. In International Conference on Machine Learning, pages 3271–3298. PMLR, 2022.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence. https: //huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf,

2026. Technical report.

Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum likelihood from incomplete data via the em algorithm. Journal of the royal statistical society: series B (methodological), 39(1):1–22, 1977.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: A comprehensive benchmark for deep research agents. arXiv preprint arXiv:2506.11763, 2025.

Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Razvan Pascanu, and Balaji Lakshminarayanan. Adapting auxiliary losses using gradient similarity, 2019. URL https://openreview. net/forum?id=r1gl7hC5Km.

Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl2: Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.

Yuchen Fan, Kaiyan Zhang, Heng Zhou, Yuxin Zuo, Yanxu Chen, Yu Fu, Xinwei Long, Xuekai Zhu, Che Jiang, Yuchen Zhang, Li Kang, Gang Chen, Cheng Huang, Zhizhou He, Bingning Wang, Lei Bai, Ning Ding, and Bowen Zhou. Ssrl: Self-search reinforcement learning. arXiv preprint arXiv:2508.10874, 2025.

Yenchia Feng, Chirag Sharma, and Karime Maamari. Environment maps: Structured environmental representations for long-horizon agents. arXiv preprint arXiv:2603.23610, 2026.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous rl. arXiv preprint arXiv:2508.07976, 2025.

Google. Build with gemini deep research. https://blog.google/innovation-and-ai/ technology/developers-tools/deep-research-agent-gemini-api/, 2025.

Google DeepMind. Gemini deep research. https://blog.google/products/gemini/ google-gemini-deep-research/, 2025.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

Nikita Gupta, Riju Chatterjee, Lukas Haas, Connie Tao, Andrew Wang, Chang Liu, Hidekazu Oiwa, Elena Gribovskaya, Jan Ackermann, John Blitzer, et al. Deepsearchqa: Bridging the comprehensiveness gap for deep research agents. arXiv preprint arXiv:2601.20975, 2026.

Rujun Han, Yanfei Chen, Zoey CuiZhu, Lesly Miculicich, Guan Sun, Yuanjun Bi, Weiming Wen, Hui Wan, Chunfeng Wen, Solène Maître, et al. Deep researcher with test-time diffusion. arXiv preprint arXiv:2507.16075, 2025.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.

Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, and Jiawei Han. Deepretrieval: Hacking real search engines and retrievers with large language models via reinforcement learning. arXiv preprint arXiv:2503.00223, 2025.

Yulun Jiang, Liangze Jiang, Damien Teney, Michael Moor, and Maria Brbic. Meta-RL induces exploration in language agents. In The Fourteenth International Conference on Learning Representations,

#### 2026. URL https://openreview.net/forum?id=4GiBscHW1k.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Jiazheng Li, Yawei Wang, Qiaojing Yan, Yijun Tian, Zhichao Xu, Huan Song, Panpan Xu, and Lin Lee Cheong. Salt: Step-level advantage assignment for long-horizon agents via trajectory graph. In Findings of the Association for Computational Linguistics: EACL 2026, pages 4709–4725, 2026.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025a.

Wenjun Li, Zhi Chen, Jingru Lin, Hannan Cao, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Dexun Li, Chen Zhang, et al. Reinforcement learning foundations for deep research systems: A survey. arXiv preprint arXiv:2509.06733, 2025b.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability. arXiv preprint arXiv:2504.21776, 2025c.

Zhiwei Li, Yong Hu, and Wenqing Wang. Encouraging good processes without the need for good answers: Reinforcement learning for llm agent planning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1654–1666, 2025d.

Zijian Li, Xin Guan, Bo Zhang, Shen Huang, Houquan Zhou, Shaopeng Lai, Ming Yan, Yong Jiang, Pengjun Xie, Fei Huang, et al. Webweaver: Structuring web-scale evidence with dynamic outlines for open-ended deep research. arXiv preprint arXiv:2509.13312, 2025e.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= v8L0pN6EOi.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems, 2021. URL https: //openreview.net/forum?id=_61Qh8tULj_.

Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, Jiayuan Song, Zhengmao Zhu, Wenhu Chen, Pengyu Zhao, and Junxian He. Webexplorer: Explore and evolve for training long-horizon web agents. arXiv preprint arXiv:2509.06501, 2025.

Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Jianbin Jiao, and Junge Zhang. Agentic reinforcement learning with implicit step rewards. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=ooROvpmxMV.

Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K Qiu, and Yuqing Yang. Agent lightning: Train any ai agents with reinforcement learning. arXiv preprint arXiv:2508.03680, 2025.

Andreas Maurer, Massimiliano Pontil, and Bernardino Romera-Paredes. The benefit of multitask representation learning. Journal of Machine Learning Research, 17:2853–2884, 2016. URL https: //www.jmlr.org/papers/v17/15-242.html.

Jianbiao Mei, Tao Hu, Daocheng Fu, Licheng Wen, Xuemeng Yang, Rong Wu, Pinlong Cai, Xinyu Cai, Xing Gao, Yu Yang, Chengjun Xie, Botian Shi, Yong Liu, and Yu Qiao. O2-searcher: A searching-based agent model for open-domain open-ended question answering. arXiv preprint arXiv:2505.16582, 2025.

Mihran Miroyan, Tsung-Han Wu, Logan King, Tianle Li, Jiayi Pan, Xinyan Hu, Wei-Lin Chiang, Anastasios N Angelopoulos, Trevor Darrell, Narges Norouzi, et al. Search arena: Analyzing searchaugmented llms. arXiv preprint arXiv:2506.05334, 2025.

Moonshot AI. Kimi-researcher: End-to-end rl training for emerging agentic capabilities. https: //moonshotai.github.io/Kimi-Researcher/, 2025.

Alireza Mousavi-Hosseini and Murat A Erdogdu. Post-training with policy gradients: Optimality and the base model barrier. arXiv preprint arXiv:2603.06957, 2026.

Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, and Shafiq Joty. Sfr-deepresearch: Towards effective reinforcement learning for autonomously reasoning single agents. arXiv preprint arXiv:2509.06283, 2025.

Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, and Aaron Courville. Asynchronous rlhf: Faster and more efficient off-policy rl for language models. arXiv preprint arXiv:2410.18252, 2024.

#### OpenAI. Introducing deep research. https://openai.com/index/ introducing-deep-research/, 2025.

Ernesto Panadero. A review of self-regulated learning: Six models and four directions for research. Frontiers in psychology, 8:422, 2017.

Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, and Mingyi Hong. Hiper: Hierarchical reinforcement learning with explicit credit assignment for large language model agents. arXiv preprint arXiv:2602.16165, 2026.

#### Perplexity AI. Sonar deep research. https://docs.perplexity.ai/docs/sonar/models/ sonar-deep-research, 2025.

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. arXiv preprint arXiv:2504.13958, 2025.

Zile Qiao, Guoxin Chen, Xuanzhong Chen, Donglei Yu, Wenbiao Yin, Xinyu Wang, Zhen Zhang, Baixuan Li, Huifeng Yin, Kuan Li, Rui Min, Minpeng Liao, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webresearcher: Unleashing unbounded reasoning capability in long-horizon agents. arXiv preprint arXiv:2509.13309, 2025.

Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient off-policy meta-reinforcement learning via probabilistic context variables. In International conference on machine learning, pages 5331–5340. PMLR, 2019.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G Finlayson, David Sontag, et al. Dr tulu: Reinforcement learning with evolving rubrics for deep research. arXiv preprint arXiv:2511.19399, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, and Bing Liu. Researchrubrics: A benchmark of prompts and rubrics for evaluating deep research agents. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=ErnvfmSX0P.

Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu, Run-Ze Fan, Bowen Jin, Yixuan Weng, Minjun Zhu, et al. Deep research: A systematic survey. arXiv preprint arXiv:2512.02038, 2025.

Amanpreet Singh, Joseph Chee Chang, Dany Haddad, Aakanksha Naik, Jena D Hwang, Rodney Kinney, Daniel S Weld, Doug Downey, and Sergey Feldman. Ai2 scholar qa: Organized literature synthesis with attribution. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 513–523, 2025.

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. arXiv preprint arXiv:2503.05592, 2025.

Trevor Standley, Amir R. Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In Proceedings of the 37th International Conference on Machine Learning, 2020. URL https://proceedings.mlr.press/v119/ standley20a.html.

Joanna Tai, Rola Ajjawi, David Boud, Phillip Dawson, and Ernesto Panadero. Developing evaluative judgement: enabling students to make decisions about the quality of work. Higher education, 76

(3):467–481, 2018.

Hui-Ze Tan, Xiao-Wen Yang, Hao Chen, Jie-Jing Shao, Yi Wen, Yuteng Shen, Weihong Luo, Xiku Du, Lan-Zhe Guo, and Yu-Feng Li. Hindsight credit assignment for long-horizon llm agents. arXiv preprint arXiv:2603.08754, 2026.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.

Nilesh Tripuraneni, Michael I. Jordan, and Chi Jin. On the theory of transfer learning: The importance of task diversity. In Advances in Neural Information Processing Systems, 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 59587bffec1c7846f3e34230141556ae-Paper.pdf.

Yi Wan, Jiuqi Wang, Liam Li, Jinsong Liu, Ruihao Zhu, and Zheqing Zhu. Pokeeresearch: Effective deep research via reinforcement learning from ai feedback and robust reasoning scaffold. arXiv preprint arXiv:2510.15862, 2025.

Jane X Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, and Matt Botvinick. Learning to reinforcement learn. arXiv preprint arXiv:1611.05763, 2016.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024.

Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, and Edward Grefenstette. A subgoal-driven framework for improving long-horizon llm agents. arXiv preprint arXiv:2603.19685, 2026.

Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, and Yichao Wu. Stepsearch: Igniting llms search ability via step-wise proximal policy optimization. arXiv preprint arXiv:2505.15107, 2025.

Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=F1vEjWK-lH_.

Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webdancer: Towards autonomous information seeking agency. arXiv preprint arXiv:2505.22648, 2025a.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al. Webwalker: Benchmarking llms in web traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10290–10305, 2025b.

Sen Wu, Hongyang R. Zhang, and Christopher Ré. Understanding and improving information transfer in multi-task learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SylzhkBtDB.

Xixi Wu, Qianguo Sun, Ruiyang Zhang, Chao Song, Junlong Wu, Yiyan Qi, and Hong Cheng. Demystifying reinforcement learning for long-horizon tool-using agents: A comprehensive recipe. arXiv preprint arXiv:2603.21972, 2026.

Zhiheng Xi, Chenyang Liao, Guanyu Li, Zhihao Zhang, Wenxiang Chen, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, et al. Agentprm: Process reward models for llm agents via step-wise promise and progress. In Proceedings of the ACM Web Conference 2026, pages 4184–4195, 2026.

Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, et al. Metaclaw: Just talk–an agent that meta-learns and evolves in the wild. arXiv preprint arXiv:2603.17187, 2026.

Baixuan Xu, Tianshi Zheng, Zhaowei Wang, Hong Ting Tsang, Weiqi Wang, Tianqing Fang, and Yangqiu Song. The cognitive bandwidth bottleneck: Shifting long-horizon agent from planning with actions to planning with schemas. arXiv preprint arXiv:2510.07091, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Lu Yang, Zelai Xu, Minyang Xie, Jiaxuan Gao, Zhao Shok, Yu Wang, and Yi Wu. Mage: Metareinforcement learning for language agents toward strategic exploration and exploitation. arXiv preprint arXiv:2603.03680, 2026a.

Matthew Y. R. Yang, Hao Bai, Ian Wu, Gene Yang, Amrith Setlur, and Aviral Kumar. InT: Self-proposed interventions enable credit assignment in LLM reasoning. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=YAhTj2VgBw.

Li S Yifei, Allen Chang, Chaitanya Malaviya, and Mark Yatskar. Researchqa: Evaluating scholarly question answering at scale across 75 fields with survey-mined questions and rubrics. arXiv preprint arXiv:2509.00496, 2025.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 3fe78a8acf5fda99de95303940a2420c-Paper.pdf.

Chenchen Zhang. From reasoning to agentic: Credit assignment in reinforcement learning for large language models. arXiv preprint arXiv:2604.09459, 2026.

Ding-Chu Zhang, Yida Zhao, Jialong Wu, Liwen Zhang, Baixuan Li, Wenbiao Yin, Yong Jiang, Yu-Feng Li, Kewei Tu, Pengjun Xie, et al. Evolvesearch: An iterative self-evolving search agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, page 13123–13136, 2025a.

Dylan Zhang, Yufeng Xu, Haojin Wang, Qingzhi Chen, and Hao Peng. Good sft optimizes for sft, better sft prepares for reinforcement learning. arXiv preprint arXiv:2602.01058, 2026.

Weizhi Zhang, Yangning Li, Yuanchen Bei, Junyu Luo, Guancheng Wan, Liangwei Yang, Chenxuan Xie, Yuyao Yang, Wei-Chieh Huang, Chunyu Miao, et al. From web search towards agentic deep research: Incentivizing search with reasoning agents. arXiv preprint arXiv:2506.18959, 2025b.

Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, and Limin Liu. R-search: Empowering llm reasoning with search via multi-reward reinforcement learning. arXiv preprint arXiv:2506.04185, 2025.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410, 2024.

Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 414–431, 2025.

### Contents

- A Additional Related Works 20
- B Details on Structured Scaffolds 21

- B.1 Scaffold Description and System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.2 SFT Data Generation Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- C Details on Stagewise Evolving Rubric Evaluation 31

- C.1 Rubric Buffer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- C.2 Adaptive Rubric Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- C.3 Stagewise Scoring . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.4 Buffer Management and Implementation . . . . . . . . . . . . . . . . . . . . . . . . . 34

- D Asynchronous Reflection Pipeline and Windowed Curriculum 35

- D.1 Training Pipeline Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- D.2 One-Step Deferred Reflection Training . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.3 Windowed Curriculum . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.4 Trajectory Sampling and Candidate Generation . . . . . . . . . . . . . . . . . . . . . 36
- D.5 Bank Persistence and Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- D.6 Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- E Theoretical Analysis 41

- E.1 Value of Stage Information . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- E.2 Judge-Aligned Stage-Weighted Credit Assignment . . . . . . . . . . . . . . . . . . . . 48
- E.3 Judge-Gated Co-Evolution of Policy and Rubric Bank . . . . . . . . . . . . . . . . . . 51

- F Experiment Details 57

- F.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
- F.2 Evaluation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- F.3 Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- F.4 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61

- G Algorithm 61
- H Limitations and Discussions 62

### A. Additional Related Works

The Post-training recipes of deep research agents. Deep research agents are increasingly framed as post-trained, tool-interacting policies rather than merely prompt-engineered retrieval pipelines (Li et al., 2025b; Shi et al., 2025; Zhang et al., 2025b). Closed-source systems demonstrate strong long-horizon research capabilities, but their training recipes remain proprietary or only described at a high level (Google, 2025; Moonshot AI, 2025; OpenAI, 2025; Perplexity AI, 2025). Most open training work instead studies search-augmented reasoning as a proxy for deep research, using short-answer or otherwise verifiable tasks whose rewards come from answer matching, retrieval quality, or rule-based process signals (Chen et al., 2025a; Fan et al., 2025; Jiang et al., 2025; Jin et al., 2025; Mei et al., 2025; Song et al., 2025; Wang et al., 2025; Zhang et al., 2025a; Zhao et al., 2025). These methods teach models when and how to retrieve, but leave open how RL should be structured for unverifiable long-form reports.

Recent work has begun to move toward long-horizon deep research agents, focusing on real-web or scalable search training (Gao et al., 2025; Li et al., 2025a; Wu et al., 2025a; Zheng et al., 2025), workflow and report-state design (Han et al., 2025; Li et al., 2025c; Qiao et al., 2025), and large-scale agentic SFT/RL or RLAIF recipes built largely around verifiable or ground-truth-anchored answer supervision (Liu et al., 2025; Nguyen et al., 2025; Tongyi DeepResearch Team et al., 2025; Wan et al., 2025). These works mainly improve the data, environment, workflow, or scale of deep research agents. Our focus is orthogonal in emphasis: rather than scaling agentic pipelines under verifiable answer supervision, we study the RL algorithmic structure needed for open-ended long-form research trajectories.

The closest work to ours is DR Tulu, which introduces RLER and establishes the first open end-toend recipe for training long-form deep research agents beyond verifiable rewards (Shao et al., 2025). We build on this foundation but target the remaining challenge of making RL more effective and training-efficient in this setting. While DR Tulu shows that extended RL training is important for long-form deep research, RubricEM improves RL efficiency through a structure–assign–evolve recipe: rubric-guided stages structure trajectories, stage-level credit assignment provides denser learning signals, and a jointly trained reflection meta-policy turns judged experience into reusable guidance for policy evolution. Empirically, this yields larger gains with fewer RL training steps.

Credit assignment in LLM agentic reinforcement learning. Credit assignment is a central challenge in agentic RL because sparse outcome rewards make it unclear which intermediate planning, tool-use, or reasoning decisions should be reinforced. Prior work addresses this issue through process supervision for mathematical reasoning (Lightman et al., 2024; Wang et al., 2024), tool- or turn-level reward shaping for interactive agents (Li et al., 2025d; Qian et al., 2025), and more general agentic credit-assignment mechanisms such as process reward models, implicit step rewards, trajectory-graph advantage assignment, and hierarchical transition decomposition (Li et al., 2026; Liu et al., 2026; Luo et al., 2025; Peng et al., 2026; Xi et al., 2026). These methods show that dense or process-aware supervision is crucial for long-horizon agent training, but they are typically designed for domains with verifiable intermediate correctness, structured environment states, tool-specific signals, or learned critics. RubricEM studies a different regime: open-ended long-form deep research beyond verifiable rewards, where intermediate progress is semantic rather than objectively checkable. We therefore assign credit at the level of rubric-guided stages, using stage-specific judge scores and causal stage-dependent returns to provide denser supervision without requiring step labels or a learned critic.

Meta-RL with language models. Meta-reinforcement learning studies agents that use past experience to adapt future behavior, with classic formulations including recurrent fast-adaptation policies such as RL2 and Learning to Reinforcement Learn, gradient-based adaptation such as MAML, and latent-context inference methods such as PEARL (Duan et al., 2016; Finn et al., 2017; Rakelly et al., 2019; Wang et al., 2016). For LLM agents, this perspective is natural because trajectories contain rich textual artifacts (plans, tool calls, observations, critiques, and failures) that can be distilled into reusable guidance. Recent language-agent Meta-RL methods train meta-policies to improve exploration and exploration–exploitation across repeated tasks (Jiang et al., 2026; Yang et al., 2026a), while MetaClaw studies continual agent evolution through failure-driven skill synthesis and a reusable skill library (Xia et al., 2026). However, these works are typically evaluated in verifiable, synthetic, or task-completion settings, and often rely on explicit support–query or cross-rollout dependencies that are costly for long-horizon web research. RubricEM targets open-ended long-form research beyond verifiable rewards: we train a shared-backbone reflection meta-policy on judged trajectories as fixed context, distill rubric-grounded reflections into a reusable rubric bank, and run this reflection branch asynchronously without imposing a sequential cross-rollout bottleneck on task-policy RL.

### B. Details on Structured Scaffolds

This section describes the rubric-guided structured scaffold in detail (Section 3.2), presents the full agent system prompt, and documents the SFT data generation pipeline.

##### B.1. Scaffold Description and System Prompt

The structured reasoning scaffold decomposes each agent trajectory into four semantically distinct stages, each marked by XML tags and governed by specific behavioral requirements. We describe each stage in detail below, followed by the full system prompt.

##### B.1.1. Stage 1: Planning (Plan)

The planning stage is the foundation of the entire trajectory. Upon receiving a user query, the agent must:

- 1. Exploratory thinking (<think>): The agent begins with unstructured brainstorming in a computational workspace. This block is used to identify initial questions, obvious roadblocks, missing variables, and to assess the multi-dimensional complexity of the query (retrieval difficulty, reasoning load, intellectual depth, and formatting demands). No structured XML tags are used inside this block.
- 2. Structured plan (<structured_plan>): The agent then produces a visible, structured planning document containing exactly three sub-components:

- • <deep_analysis>: Deconstructs the user’s query into explicit needs (what they directly ask for), implicit needs and gaps (hidden constraints, missing variables, potential roadblocks), and a complexity assessment that determines how much effort the agent should invest.
- • <rubric>: The agent acts as an expert grader and creates a self-evaluation checklist for the eventual answer. This rubric contains three categories: (i) a knowledge checklist specifying the exact facts, definitions, comparisons, or data points required; (ii) analytical and synthesis criteria describing the intellectual connections the answer must achieve (optional for simple queries); and (iii) negative constraints listing what the answer must explicitly avoid. Crucially, these rubrics are formulated before any search is conducted, as prospective target objectives.

- • <research_plan>: A logical roadmap to satisfy the rubric. Simple queries receive a linear one- or two-step plan, while complex queries receive a conditional, look-ahead strategy with explicit routing logic (e.g., “If X confirms Y, investigate Z; if X is inconclusive, fallback to W”).

- 3. First tool call (<call_tool>): Immediately after the plan, the agent executes the first step of its research plan. The agent is strictly forbidden from producing an <answer> in the first turn—it must always search first.

A key design principle is adaptive cognitive effort: the depth of planning scales with query complexity. For a simple factual lookup, the analysis and rubric are brief; for a complex multi-faceted research question, the agent deploys its full planning machinery with detailed rubric criteria and multi-step conditional plans.

##### B.1.2. Stage 2: Research (Research)

The research stage is an iterative loop of evidence gathering and evaluation. After each tool output, the agent:

- 1. Evaluation thinking (<think>): Digests new evidence in an unstructured inner monologue, noting conflicts, necessary pivots, or writing hurdles.
- 2. State evaluation (<state_evaluation>): Produces a visible evaluation of the current state of evidence, comparing accumulated findings against the rubric and research plan. Based on this evaluation, the agent chooses one of two paths:

- • Path A—Continue research: If more information is needed, the agent either (a) issues another <call_tool> directly if the current plan remains valid, or (b) outputs an updated <structured_plan> with revised analysis, rubric, and/or research plan before the next tool call. This dynamic plan revision enables adaptive multi-hop reasoning when initial assumptions are invalidated.
- • Path B—Proceed to review: If evidence is sufficient to satisfy the rubric, the agent transitions to the review stage.

The research stage may iterate multiple times. The outer scaffold is sequential (Plan → Research

→ Review → Answer), but within Research, the agent can freely iterate between tool use and state evaluation, and can revise its plan in place.

##### B.1.3. Stage 3: Review (Review)

Before producing the final answer, the agent must perform a structured self-evaluation. This is not optional—the scaffold enforces it as a mandatory step. The review block (<review>) contains:

- • <rubric_review>: The agent maps its retrieved evidence back to the specific criteria in its Phase 1 rubric (plus any dynamic updates made during research). This includes: (i) knowledge verification—explicitly listing which checklist items have been satisfied with retrieved evidence; and (ii) synthesis and constraints restatement—re-articulating the analytical criteria and negative constraints in the context of the actual evidence gathered, transforming abstract rubric targets into concrete, specific directives for the writing phase.
- • <writing_plan>: An architectural outline for the final answer, scaled to query complexity. For simple queries, this is a brief statement of the verified facts to output. For complex long-form queries, it includes: a unified core thesis, a value proposition (why this answer is exceptional), a narrative architecture with section-by-section outline, and a citation mapping that specifies how and where verified facts will be woven into specific sections.

This mandatory review stage serves two functions: it forces the agent to verify rubric satisfaction before writing (preventing premature or incomplete answers), and it produces a writing plan that structures the final synthesis rather than letting the agent write in a stream-of-consciousness fashion.

##### B.1.4. Stage 4: Answer (Answer)

The final stage produces the long-form response within <answer>...</answer> tags. The answer must follow the writing plan established in the review, satisfy all rubric criteria verified in the rubric review, and ground nontrivial claims with inline citations using <cite id="SNIPPET_ID">claim text</cite> format. The depth, tone, and formatting are dictated by the response format (exact match, short-form, or long-form) and the complexity level assessed during planning.

###### B.1.5. Cross-Stage Design Principles Several design principles span all four stages:

- • Rubric as central anchor. The rubric is not merely an evaluation artifact applied post-hoc—it is generated at the start and actively conditions every subsequent decision: what to search for, how to evaluate evidence, and what the final answer must include or avoid. This makes the rubric the shared interface across the entire trajectory.
- • One action per turn. Each agent generation must end with exactly one of </call_tool> or </answer>. The agent must stop generating immediately after the closing tag. This prevents the agent from hallucinating tool outputs or bypassing the research phase, and enables the tool execution environment to interleave real search results.
- • Substance over syntax. The response format instructions (exact/short/long) dictate the surfacelevel shape of the answer, but the self-generated rubric dictates the substance. The agent is instructed to enforce its rubric before drafting, preventing format instructions from overriding deep analytical criteria.

##### B.1.6. Full System Prompt

We present the complete system prompt below. The workflow example section is abbreviated for space.

Full Agent System Prompt

# Role: Elite Autonomous Research & Synthesis Agent

You are an elite agent designed to analyze complex queries, autonomously perform rubric-guided plans, execute iterative research, and synthesize highly rigorous, evidence-backed answers. You operate in a continuous loop: rubric-guided planning, searching, evaluating, and answering. You DO NOT answer the user’s question directly until you have sufficient evidence, and you DO NOT simulate tool outputs.

## CORE PRINCIPLE: ADAPTIVE COGNITIVE EFFORT Your level of planning rigor, research depth, and synthesis MUST dynamically scale with the true, multi-dimensional complexity of the user’s query and the evolving search landscape.

- - Complexity is Multi-Dimensional: It encompasses retrieval

- difficulty, reasoning load, intellectual depth, and formatting demands.
- - Scale Your Effort: Do not over-engineer simple tasks. Conversely, for highly complex tasks, deploy your full intellectual machinery.

## EXPECTED RESPONSE FORMATS & SYNTAX RULES Map your drafting strategy to the requested format appended to the prompt. ALL final outputs MUST be enclosed within <answer>...</answer>:

- 1. Exact Match: Format strictly as \boxed{exact answer}.
- 2. Short-Form: Write a single, cohesive paragraph with <cite id="...">...</cite> support.
- 3. Long-Form: Write a comprehensive, markdown-structured response. Synthesize sources into a cohesive narrative. Ground all nontrivial claims with <cite id="...">...</cite>.

CRITICAL: formatting instructions appended to the user’s prompt dictate the shape of your answer, but your dynamically updated <rubric> dictates the substance.

---

- ## PHASE 1: PLANNING & INITIATION

- STEP 0: Exploratory Thinking Begin with a <think> block. Use this as a computational workspace for brief, unstructured brainstorming. Identify initial questions, obvious roadblocks, and missing variables.
- STEP 1: The Structured Plan Output a single <structured_plan> block containing exactly three sections, building upon each other logically:

- 1. <deep_analysis>: Discover the True Intent.

- - Complexity Assessment: Evaluate the multi-dimensional complexity (retrieval, reasoning, insight, formulation).
- - Explicit Needs: What they are directly asking for.
- - Implicit Needs & Gaps: Hidden constraints, missing variables, or potential roadblocks.

- 2. <rubric>: Define the Strict Grading Criteria. Act as an expert grader creating a rigorous checklist. Draw upon the <deep_analysis> as a foundation. DO NOT focus on formatting; focus on required content, intellectual depth, and logical constraints.

- - Knowledge Checklist: The exact facts, definitions, comparisons, or data points required.
- - Analytical & Synthesis Criteria (Optional, for complex queries): What intellectual connections must the response achieve?
- - Negative Constraints (Pitfalls): What must the response explicitly AVOID?

- 3. <research_plan>: Formulate the Strategy. Create a logical roadmap to satisfy the rubric. Simple queries need a linear step or two; complex queries require a conditional, look-ahead strategy.

- STEP 2: The First Tool Call Immediately after closing </structured_plan>, execute ONLY the first step of your research plan. Under NO circumstances should you output an <answer> in this first turn.

---

- ## PHASE 2: SEARCH & SYNTHESIS

- STEP 0: Evaluation Thinking Begin with a <think> block for raw inner monologue to digest new evidence.
- STEP 1: State Evaluation & Action Output a <state_evaluation> block to analyze the tool output, then choose exactly ONE path:

- PATH A: Continue Research (More Information Needed)

- - Direct Search: If the current plan and rubric are still valid, output your next <call_tool>.
- - Update & Search: If new information invalidates initial assumptions, output an updated <structured_plan> with revised <deep_analysis>, <rubric>, and/or <research_plan>, then output your next <call_tool>.

- PATH B: Ready to Answer (Evidence Sufficient)

- 1. The Review: Output a <review> block containing:

- - <rubric_review>: Systematically map retrieved evidence back to the Phase 1 <rubric>. Verify knowledge checklist items. Re-articulate synthesis criteria and negative constraints in the context of retrieved evidence.
- - <writing_plan>: Outline the final answer’s architecture. For complex queries: define a unified core thesis, value proposition, narrative architecture, and citation mapping.

- 2. The Final Answer: Output the response within <answer>...</answer> tags.

--## Available Tools

- - google_search: Powered by a grounded AI reasoning engine. Use direct, highly specific search queries. Input: <call_tool name="google_search">query</call_tool>
- - snippet_search: Focused retrieval from scientific papers. Input: <call_tool name="snippet_search">query</call_tool> Optional parameters: limit, year, fieldsOfStudy.

## Tool Output Format

- - Results are appended in <tool_output>...</tool_output> tags.
- - For google_search: summary text with grounding snippets.
- - For snippet_search: <snippet id="ID">content</snippet>.

## WORKFLOW EXAMPLE [... A complete worked example demonstrating the full scaffold on a sample query, showing Phase 1 (think, structured_plan, first call_tool), Phase 2 iterations (state_evaluation, continued search, dynamic plan updates), and the review and answer synthesis. Omitted for brevity. ...]

## CRITICAL CONSTRAINTS

- - Mandatory Starting Tag: Output MUST begin with <think>.
- - Mandatory First Search: First turn MUST end with </call_tool>. Never answer without searching.
- - One Action Per Turn: End with </call_tool> OR </answer>. STOP generating immediately after. Do NOT simulate <tool_output> yourself.
- - Mandatory Citations: Wrap claims in <cite id="S_123">claim text</cite>. Never use empty tags. Multiple sources: <cite id="S_1, S_2">claim</cite>.
- - Substance over Syntax: The <rubric> dictates substance; formatting instructions dictate only surface-level shape.

Additional instructions. Depending on the expected response format, one of three task-specific instructions is appended to the user’s query:

- • Exact answer: “Search iteratively to find the precise answer. Format: <answer>\boxed{exact answer}</answer>.”
- • Short form: “Search iteratively to gather evidence from multiple credible sources. Synthesize into a short paragraph within <answer>...</answer> tags.”
- • Long form: “Search iteratively to gather evidence from multiple credible sources. Synthesize into a comprehensive, evidence-backed long-form response within <answer>...</answer> tags.”

##### B.2. SFT Data Generation Process

To instill the rubric-guided scaffold into the base Qwen3-8B model, we perform supervised finetuning using teacher-generated trajectories. The data generation process consists of a multi-round, tool-augmented pipeline that produces complete stage-structured trajectories from a teacher model.

##### B.2.1. Teacher Model and Prompt Adaptation

We use Gemini-3.1-Pro as the teacher model. The training queries are drawn from the same dataset used by Dr. Tulu (Shao et al., 2025), which contains ∼13K diverse, open-ended research questions spanning multiple domains. Each query is paired with a response format label (exact_answer, short_form, or long_form) that determines the additional instruction appended to the prompt.

A key challenge is that the SFT data generation prompt must be adapted for Gemini, which differs from the target Qwen3-8B model in several important ways.

Reasoning traces and <scratchpad>. Our scaffold includes unstructured <think> blocks at the start of each turn to preserve the base Qwen3-8B policy’s post-training reasoning behavior. However, directly prompting Gemini to produce <think> tags is blocked by the API, and internal thinking traces are inaccessible due to anti-distillation policies. We use <scratchpad> as a substitute reasoning tag during SFT data generation; these are converted to <think> in post-processing. While these traces do not capture Gemini’s true internal chain-of-thought, they provide visible planning rationale and evidence assessment that teaches the student model the habit of explicit reasoning before acting, without disrupting its native <think> patterns.

Stagewise prompt separation. We separate the system prompt into a first-round variant (containing only Phase 1 instructions) and a later-rounds variant (containing only Phase 2 instructions), since each round is a separate Gemini API call. This separation serves two purposes: it keeps each prompt focused on the current stage’s requirements (reducing prompt length and improving adherence), and it ensures the model invests heavily in planning at step 1 (generating the full <structured_plan> with analysis, rubric, and research plan) rather than skipping ahead to search or answering directly. Each variant explicitly constrains the turn-ending tag: the first round must end with </call_tool>, and later rounds must end with either </call_tool> or </answer>.

We present the core of each generation prompt below (omitting tool documentation and workflow examples, which mirror the final agent prompt).

SFT Generation Prompt: First Round (Phase 1 Only)

# Role: Elite Research Planning & Initiation Agent You are an elite Research Planning Agent. Your purpose is to analyze complex queries, generate a rigorous, visible research plan, and initiate the very first search query to kick off the research process. You DO NOT answer the user’s question directly, and you DO NOT simulate tool outputs. ## CORE PRINCIPLE: ADAPTIVE COGNITIVE EFFORT [... same as final prompt ...] ## Process

- STEP 0: Exploratory Scratchpad Begin with a <scratchpad> block. This is your mandatory computational workspace. Use this space for a brief, unstructured exploration of the user’s query --- identify initial hurdles and missing variables before moving to structured output. Do not output XML tags inside scratchpad.
- STEP 1: The Structured Plan Output a single <structured_plan> block containing exactly three sections, building upon each other logically:

- 1. <deep_analysis>: Discover the True Intent.

- - Complexity Assessment: Evaluate the multi-dimensional complexity (retrieval, reasoning, insight, formulation).
- - Explicit Needs: What they are directly asking for (including structural/formatting instructions).

- - Implicit Needs & Gaps: Hidden constraints, missing variables, or potential roadblocks.

- 2. <rubric>: Define the Strict Grading Criteria. Act as an expert grader creating a rigorous checklist. DO NOT focus on formatting; focus on content, depth, and constraints.

- - Knowledge Checklist: Exact facts, definitions, data points required (e.g., "The response explicitly defines the von Neumann bottleneck").
- - Analytical & Synthesis Criteria (Optional, for complex queries): What intellectual connections must the response achieve?
- - Negative Constraints (Pitfalls): What must the response explicitly AVOID? (e.g., "The response avoids presenting industry blogs as academic consensus").

- 3. <research_plan>: Formulate the Strategy. Create a logical roadmap to satisfy the rubric. Simple queries: linear steps. Complex queries: conditional, look-ahead strategy (e.g., "Step 1: Find X. Step 2: If X confirms Y, investigate Z; if inconclusive, fallback to W").

- STEP 2: The First Tool Call Immediately after closing </structured_plan>, execute ONLY the first step of your <research_plan>. Once you output </call_tool>, you must STOP generating text. ## CRITICAL CONSTRAINTS

- - Output MUST begin with <scratchpad>, then <structured_plan>.
- - Execute EXACTLY ONE tool call.
- - DO NOT simulate <tool_output>.
- - DO NOT attempt to write the final <answer>. [... tool documentation and workflow examples omitted ...]

SFT Generation Prompt: Later Rounds (Phase 2 Only)

# Role: Elite Iterative Research & Synthesis Agent You are in the active research phase. You have already generated an initial Rubric and Research Plan. Your goal is to evaluate the latest retrieved information, dynamically adjust your strategy if necessary, and either continue searching or generate the final answer. ## CONTEXT: THE PHASE 1 FOUNDATION Prior to your current phase, an initial <structured_plan> was generated. This plan is your foundational blueprint consisting of <deep_analysis>, <rubric>, and <research_plan>. You must constantly evaluate incoming evidence against this baseline. ## Process & Decision Tree

- STEP 0: Evaluation Scratchpad

- Begin with a <scratchpad> block. This is your mandatory computational workspace for raw, unstructured inner monologue to digest newly retrieved evidence. If data contradicts assumptions, briefly identify the conflict and note your pivot. If evidence is sufficient for a complex query, pinpoint writing hurdles before resolving them in <review>.
- STEP 1: State Evaluation & Action Output a <state_evaluation> block to formally analyze the latest tool output, then choose exactly ONE path:

- PATH A: Continue Research (More Information Needed)

- - Direct Search: If the current plan and rubric are still valid, output your next <call_tool> immediately.
- - Update & Search: If new information invalidates initial assumptions, triggers a fallback, or reveals a deeper multi-hop requirement:

- 1. Output an updated <structured_plan> with revised <deep_analysis>, <rubric>, and/or <research_plan>. (Only include sections that require updates.)
- 2. Then output your next <call_tool>.

- PATH B: Ready to Answer (Evidence Sufficient)

- 1. The Review: Output a <review> block containing:

- - <rubric_review>: Systematically map retrieved evidence back to the Phase 1 <rubric> (plus any dynamic updates).

- * Knowledge Verification: List the critical facts and data points that satisfy the Knowledge Checklist.
- * Synthesis & Constraints Restatement: Re-articulate Analytical Criteria and Negative Constraints in the context of retrieved evidence. Transform abstract rubric targets into concrete directives for drafting.

- - <writing_plan>: Outline the final answer architecture, scaling effort to query complexity:

- * Low-Complexity: State verified facts directly.
- * High-Complexity: Define (1) unified core thesis,

(2) value proposition, (3) narrative architecture with section-by-section outline, (4) citation mapping specifying how verified facts integrate into sections.

- 2. The Final Answer: Output within <answer>...</answer> tags. Depth and formatting MUST match the writing plan. Follow the Knowledge Checklist and honor all constraints from the <rubric_review>.

## CRITICAL CONSTRAINTS

- - Output MUST begin with <scratchpad>, then <state_evaluation>.
- - One Action Per Turn: end with </call_tool> OR </answer>. STOP generating immediately after the closing tag.
- - Mandatory Citations: wrap claims in <cite id="S_123">claim text</cite>. Never use empty tags.
- - Substance over Syntax: formatting instructions dictate shape; the <rubric> dictates substance. Enforce the rubric before drafting.

[... tool documentation and workflow examples omitted ...]

The key difference from the unified final agent prompt is the strict separation of responsibilities: the first-round prompt forbids answering and requires a full planning phase, while the later-rounds prompt assumes the plan already exists and focuses on evidence evaluation and synthesis. This separation ensures the teacher invests in thorough planning rather than short-circuiting to an answer.

Common failure modes. Despite these adaptations, Gemini-3.1-Pro frequently violates the scaffold constraints. Common failures include answering directly from internal knowledge without searching, producing output that does not end with the required closing tag, omitting required structural elements such as <rubric> or <state_evaluation>, and generating variant tool names that do not match the expected schema. These violations necessitate aggressive rejection sampling (described below).

##### B.2.2. Multi-Round Generation Pipeline

Because the scaffold requires iterative tool interaction, we cannot generate complete trajectories in a single model call. Instead, we use a multi-round pipeline that alternates between model generation and real tool execution:

- 1. Round 1 (Initiation): The teacher receives the first-round system prompt and user query. It generates Phase 1 output: <scratchpad>, <structured_plan> (with <deep_analysis>, <rubric>, <research_plan>), and the first <call_tool>. Generation stops at the closing </call_tool> tag.
- 2. Tool routing and execution: Each <call_tool> is parsed to extract the tool name and query. The pipeline routes calls to one of two backends:

- • google_search: Submitted as Vertex AI batch prediction jobs using Gemini with Google Search grounding enabled, which returns AI-synthesized summaries with grounding snippets.
- • snippet_search: Executed asynchronously against the Semantic Scholar API, which returns paper excerpts matching the query.

Both backends run in parallel to maximize throughput. Responses where the tool fails to call a valid tool (e.g., answering from internal knowledge) are separated into an error stream for potential reprocessing.

- 3. Rounds 2–𝑁 (Research iteration): The tool output is appended to the conversation history, and the teacher receives the later-rounds system prompt. It generates the next Phase 2 turn: <scratchpad>, <state_evaluation>, and either another <call_tool> or the <review> and <answer>. This loop continues until the teacher produces a closing </answer> tag or a maximum of 10 rounds is reached.
- 4. Completion detection: A trajectory is considered complete when the teacher’s output ends with </answer>. Incomplete trajectories that exhaust the maximum rounds are discarded. Because Gemini often fails to produce a well-formed closing tag, a substantial fraction of trajectories (∼15–25%) are discarded at this stage.

##### B.2.3. Quality Filtering and Rejection Sampling

Raw teacher trajectories undergo rejection sampling and several filtering steps. This is critical because Gemini-3.1-Pro, despite being a strong model, frequently violates the scaffold’s structural constraints when generating multi-round tool-augmented trajectories.

Rejection criteria. The following trajectories are rejected:

- • Missing </answer> tag (hard reject): The most common failure mode. The teacher generates extensive content but never produces the closing </answer> tag, either because it runs out of tokens, enters an endless research loop, or simply stops mid-sentence. These trajectories are discarded entirely.
- • No valid tool call in non-final rounds: Each non-final round must end with a valid </call_tool> tag containing a parseable tool name and query. Trajectories where the model skips tool use and attempts to answer from internal knowledge (e.g., producing a long response without any <call_tool>) are rejected. This is the second most common failure mode, particularly on simple factual queries where Gemini “knows” the answer and refuses to search.
- • Missing structural elements: Trajectories must contain valid<structured_plan>(with<deep_analysis> and <rubric>) in the first round, and <state_evaluation> in subsequent rounds. Those with malformed XML or missing required sections are discarded.
- • Consecutive tool errors: Trajectories with two or more consecutive rounds where the tool backend returned an error (e.g., search blocked by safety filters, API rate limits, “no grounding data available”) are discarded. Single error rounds followed by successful recovery are retained, as these teach the model error-resilient behavior.

Post-processing transformations. Accepted trajectories undergo the following conversions:

- • Reasoning tag conversion: All <scratchpad>...</scratchpad> tags from Gemini’s output are converted to <think>...</think> for compatibility with Qwen3’s chat template, which uses <think> as the native reasoning tag.
- • Tool name normalization: Gemini occasionally produces variant tool names (e.g., google_web_search, Scholar_Search) or malformed attribute strings. These are normalized to the canonicalgoogle_search and snippet_search names.

- B.2.4. Training Data Format Each filtered trajectory is converted into a single-turn ChatML conversation with three messages:

- • System: The full agent system prompt (the unified Qwen3 version from Sec. B.1, not the Geminiadapted generation prompt).
- • User: The original query with the format-specific additional instruction appended.
- • Assistant: The complete multi-round trajectory, with tool outputs wrapped in special span-masking markers. Only the model-generated tokens (thinking, planning, tool calls, state evaluation, review, and answer) receive gradient during training; tool output tokens are masked so the model does not learn to memorize search results.

Finally, this process yields around 11k SFT samples, approximately 2k fewer than DR Tulu due to the repeated errors described above, which are filtered out.

### C. Details on Stagewise Evolving Rubric Evaluation

This section provides implementation details for the stagewise evolving rubric design described in Section 3.3. The judge operates in two phases: adaptive rubric generation proposes new rubrics by comparing trajectories for the same question, and stagewise scoring evaluates each trajectory against the current rubric set. Both phases use an LLM judge (Gemini-3-Flash) with structured JSON output.

##### C.1. Rubric Buffer

For each training question, we maintain a rubric buffer with two types of rubrics: persistent rubrics (static, ground-truth rubrics from the training data, never removed) and active rubrics (adaptive rubrics proposed online by the judge, organized by stage and subject to per-stage capacity caps). The persistent rubrics are adapted from Shao et al. (2025) and only used for the final answer stage. At each training step, for each unique question in the batch, the judge (i) generates new adaptive rubrics by comparing the rollout group, (ii) adds them to the buffer, (iii) scores all trajectories against the combined rubric set, and (iv) removes low-discrimination rubrics that did not meaningfully separate trajectory quality.

##### C.2. Adaptive Rubric Generation

The rubric generation prompt instructs the judge to identify the most discriminative, stage-local criteria that explain why some trajectories are better than others. Several design principles considered:

- • Discriminative and specific: Each rubric must actually separate stronger from weaker trajectories for the same question. Descriptions must reference concrete aspects of the question so a separate

- scorer can unambiguously judge them. Vague rubrics like “good research quality” are prohibited.

- • Stage-local and non-redundant: Rubrics target each stage’s specific responsibility. The judge avoids creating positive and negative mirror versions of the same criterion, and skips criteria already covered by existing rubrics.
- • Anti-hack: The judge must not create rubrics that merely check XML tag existence, reward formal obedience to a weak plan, overfit to one trajectory’s wording, or confuse self-consistency with true quality.

A key feature enabled by our rubric-guided scaffold is that the judge can use the agent’s own <rubric> blocks from different trajectories as references when proposing adaptive rubrics—strong rubrics across trajectories can reveal important task dimensions and failure modes. However, a trajectory’s own rubric is never the scoring standard: the judge is instructed to never reward following a weak self-rubric or penalize evidence-based improvement beyond the original plan.

Rather than prescribing fixed templates, the prompt provides stage-specific guidance as starting points. For each stage, a central question frames the evaluation: “Did the agent understand the problem?” (Plan), “Did it search effectively and adapt?” (Research), “Did it honestly audit readiness?” (Review), and “Is the final answer high-quality?” (Answer). Common useful dimensions are suggested (e.g., query specificity, evidence-based pivoting for Research), but the judge is encouraged to discover what matters for each specific question.

Each rubric item contains a title, description, and weight (1=minor, 2=important, 3=critical), organized into positive and negative rubrics per stage. Target counts are 1–4 rubrics per stage for stages 1–3 and 2–5 for stage 4, treated as targets rather than quotas. The full rubric generation prompt is shown below:

Stagewise Adaptive Rubric Generation Prompt (Core)

You are an expert evaluator generating stagewise comparative rubrics for multiple full trajectories produced by the SAME research agent on the SAME question.

## Goal Identify the most discriminative, stage-local criteria that

explain why some trajectories are better than others. Evaluate real process quality and answer quality — not XML compliance, verbosity, or shallow self-consistency.

## How to Read the Trajectories Each trajectory is a structured full execution trace:

- - Stage 1 — Plan: <structured_plan> with <deep_analysis>, <rubric>, <research_plan>
- - Stage 2 — Research: iterative <state_evaluation>, tool calls, and possible plan/rubric updates
- - Stage 3 — Review: <review> with <rubric_review> and <writing_plan>
- - Stage 4 — Answer: final <answer>

Do not reward a stage just because the relevant tags exist. Judge whether that stage actually performs its intended function well.

## How to Use Agent-Generated Rubrics A trajectory’s own Stage 1 <rubric> is never the scoring standard. Use it only as a reference signal:

- - it shows what the agent recognized as important,
- - it may surface useful task dimensions,
- - it may expose shallow planning if it misses critical needs. But: never reward following a weak self-rubric, never let a narrow self-rubric lower the real standard, never penalize evidence-based improvement beyond the original plan.

## Stage-by-Stage Guidance For each stage, the most discriminative criteria often emerge from the specific question and trajectories themselves. The dimensions below are common starting points:

- Stage 1 — PLAN: task understanding depth, identification of hidden constraints, rubric specificity, plan proportionality.
- Stage 2 — RESEARCH: query specificity, evidence-based pivoting, triangulation, stopping criteria.
- Stage 3 — REVIEW: honest gap assessment, evidence-tostructure mapping, drafting plan quality.
- Stage 4 — ANSWER: readability, instruction following, insightfulness, comprehensiveness, correctness, evidence grounding. Evaluated independently from earlier trajectory.

## Anti-Hack Rules Do NOT create rubrics that: only check tag existence, reward formal obedience to a bad plan, overfit to one trajectory’s wording, confuse self-consistency with true quality, punish evidence-based improvement, or reward unnecessary overhead.

## Selection Rules

- - Fewer, sharper rubrics over many generic ones.
- - Skip criteria where all trajectories perform similarly.
- - Never create positive and negative mirrors of the same idea.
- - Conservative negative rubrics: target clear failure modes.

## Output Format Return rubrics by stage (stage_1 through stage_4), each with positive_rubrics and negative_rubrics. Each item: title, description, weight (1-3).

##### C.3. Stagewise Scoring

After rubric generation, each trajectory is scored independently against the combined rubric set (persistent + active). Positive rubrics use a 0/1/2 scale (absent / partial / full), and negative rubrics use the same scale but inverted during aggregation: a score of 0 (no flaw) contributes positively. Each evaluation includes a brief justification grounded in specific trajectory evidence. For each stage 𝑘, the per-stage score 𝑅𝑖,𝑘 ∈ [0, 1] is a weighted average of rubric scores within that stage, with negative

- scores inverted before aggregation. Stagewise Scoring Prompt

You are an expert evaluator. You will receive an AI research agent’s full trajectory and a set of rubrics grouped by stage. Score each rubric strictly based on the trajectory evidence.

## Trajectory Structure

- - Stage 1 (Plan): start through end of </structured_plan>
- - Stage 2 (Research): after </structured_plan> through all tool-call loops
- - Stage 3 (Review): last <think> before <review> through </review>
- - Stage 4 (Answer): <answer> through end of text

## Scoring Rules [POSITIVE] rubrics:

2 = FULLY exhibits this quality 1 = PARTIALLY exhibits it

- 0 = Completely ABSENT or fails

[NEGATIVE] rubrics: 2 = Fully EXHIBITS the flaw

- 1 = PARTIALLY exhibits the flaw 0 = Successfully AVOIDS the flaw

## Instructions

- - Score each rubric independently.
- - Use the rubric description as the sole scoring criterion.
- - Provide brief justification grounded in specific trajectory evidence (quote or reference concrete content).
- - Return the exact rubric index with each score.

##### C.4. Buffer Management and Implementation

Buffer dynamics. Each stage maintains a capacity cap on active rubrics (3, 2, 2, 3 for stages 1–4 in our experiments). When a stage exceeds its cap after scoring, rubrics with the lowest discrimination (measured by score variance across the rollout group) are removed. Persistent rubrics are never removed. When a question reappears in a later training step, new adaptive rubrics from the improved

policy are added, potentially replacing stale ones that no longer discriminate. This ensures the judge’s criteria co-evolve with the policy, consistent with the evolving rubric paradigm of Shao et al. (2025), extended here from answer-only to stagewise evaluation.

Structured JSON output. Both rubric generation and scoring use Gemini’s structured output mode (response_mime_type="application/json") with Pydantic-derived JSON schemas. This guarantees schema compliance (rubrics organized by stage with title/description/weight, scores with indices and justifications) without fragile post-hoc text parsing.

Batch parallelism. At each training step, the batch contains 32 unique questions × 8 rollouts = 256 trajectories. All 32 rubric generation calls are launched concurrently via asyncio.gather, followed by all 256 scoring calls in parallel. Each call includes exponential-backoff retry logic (up to 5 retries). Failed calls are treated as “no new rubrics” (generation) or fall back to terminal answer score only (scoring).

Efficiency. Rubric generation requires only 32 calls per step (one per question, not per trajectory). Scoring requires 256 calls but each is lightweight (∼8–15 rubrics per trajectory). The combined rubric generation + scoring phase takes approximately 5 minutes per step, overlapping with the training engine’s gradient computation. The judge model (Gemini-3-Flash) is chosen for cost and latency; the structured output constraint and detailed prompt compensate for the smaller model’s limitations.

### D. Asynchronous Reflection Pipeline and Windowed Curriculum

This appendix provides the implementation details for the meta-policy training pipeline and the windowed retrieval curriculum described in Section 3.4.

##### D.1. Training Pipeline Architecture

Each RL training step in our agentic loop involves several sequential phases: rollout generation (multiturn tool-augmented trajectories via vLLM), stagewise judge scoring (SS-GRPO reward computation), and policy gradient update. Reflection generation (prompting the backbone to produce rubricgrounded reflections from sampled trajectories) introduces an additional compute requirement that we overlap with the main loop.

Our pipeline uses three concurrent threads:

- 1. Main thread (training engine): Performs gradient updates on the policy. Alternates between Phase A (meta-policy update on deferred reflections from the previous step) and Phase B (task-policy SS-GRPO update on the current step’s rollouts).
- 2. Inference thread (vLLM): Generates multi-turn tool-augmented rollouts. During Phase A, the inference engine begins generating the next batch of rollouts while the training engine trains on deferred reflections—fully overlapping inference and reflection training.
- 3. Data preparation thread: Runs judge scoring (async Gemini API calls), launches reflection generation and reflection judging as concurrent background tasks, and prepares packed training batches. Reward scoring results are returned immediately for task-policy training; reflection scoring continues asynchronously.

##### D.2. One-Step Deferred Reflection Training

A synchronous implementation would block the next task rollout until reflection generation, judging, and the meta-policy update are finished. We instead defer reflection training by one RL step, as illustrated in the bottom panel of Fig. 4.

At step 𝑁:

- 1. The inference engine generates task rollouts for step 𝑁.
- 2. The data preparation thread scores step 𝑁 rollouts (SS-GRPO rewards) and simultaneously launches background tasks for: (a) reflection rollout generation via vLLM, producing 𝑛 candidates per sampled trajectory, and (b) judge scoring of the reflection candidates.
- 3. Reward results are returned immediately; the training engine performs the SS-GRPO task-policy update on step 𝑁.
- 4. Meanwhile, reflection scoring completes in the background. Accepted reflections are inserted into the rubric bank, and the scored reflection samples are placed in a deferred buffer.
- 5. At the start of step 𝑁+1, during Phase A, the training engine trains on the deferred reflection buffer from step 𝑁 while the inference engine generates step 𝑁+1 rollouts.

This one-step staleness trades exact synchrony for higher infrastructure utilization: both inference and training engines remain continuously occupied, and meta-policy training adds effectively no extra wall-clock overhead to the SS-GRPO loop.

##### D.3. Windowed Curriculum

Since reflection training is deferred by one step and bank insertion happens asynchronously, a naïve curriculum that immediately revisits a query after its first encounter may attempt within-episode retrieval before the corresponding reflection has been generated and accepted. To avoid this conflict, we introduce a windowed curriculum with window size 𝐾 that provides sufficient temporal separation between a query’s first and second encounters.

Training alternates between two phases within each window of 2𝐾 steps:

- 1. New-query phase (steps 1, . . . , 𝐾 within the window): The dataloader samples 𝐾 fresh query batches {𝐵1, 𝐵2, . . . , 𝐵𝐾}. For each batch, the bank performs cross-episode retrieval: semantically similar items from past (different) queries are injected as few-shot exemplars. At the end of each step, reflection generation is launched asynchronously in the background.
- 2. Repeat phase (steps 𝐾+1, . . . , 2𝐾 within the window): The same batches are replayed in order: 𝐵1, 𝐵2, . . . , 𝐵𝐾. For each batch, the bank performs within-episode retrieval: the exact reflection generated during the new-query phase is retrieved and injected as direct self-guidance.

The 𝐾-step gap between a query’s first encounter and its repeat guarantees that the deferred reflection pipeline (generation, judging, bank insertion, and at least one deferred training step) has fully completed before within-episode retrieval is attempted. In our experiments, we use 𝐾=3, providing a comfortable margin over the one-step deferral.

##### D.4. Trajectory Sampling and Candidate Generation

For each query in the new-query phase, we randomly sample one trajectory from the rollout group for reflection generation. The sampling is uniform random over trajectory indices, without length bias or score-based selection, mirroring inference conditions where the agent does not have access to

###### Step in window Phase Data Retrieval mode

- 1 New 𝐵1 (fresh) Cross-episode (similar)

- 2 New 𝐵2 (fresh) Cross-episode (similar)

- 3 New 𝐵3 (fresh) Cross-episode (similar)

- 4 Repeat 𝐵1 (replay) Within-episode (exact)

- 5 Repeat 𝐵2 (replay) Within-episode (exact)

- 6 Repeat 𝐵3 (replay) Within-episode (exact)

Figure 7 | Windowed curriculum with 𝐾=3. Each window of 2𝐾=6 steps alternates between new queries with cross-episode retrieval and repeated queries with within-episode exact retrieval. The 3-step gap ensures the deferred reflection pipeline has fully completed before the repeat.

privileged scoring information. From the sampled trajectory, we generate 𝑛=8 reflection candidates via vLLM parallel sampling (temperature 0.7, single prompt, 𝑛 completions). Each candidate is independently scored by the judge, and only the highest-scored candidate with valid output format is accepted into the bank. Each query’s reflection generation is independent, so a failure for one query does not affect others.

##### D.5. Bank Persistence and Retrieval

The rubric bank maintains an in-memory FAISS index over query embeddings (computed via Qwen3Embedding-0.6B on CPU). For cross-episode retrieval, the query embedding is compared against stored embeddings via inner-product similarity, and the top-𝑘 items (default: 𝑘=2) are returned. For within-episode retrieval, items are matched by exact question hash (SHA-256). When a query already has a bank item and the same query is encountered in a later training step, the new reflection overwrites the old one, keeping the bank synchronized with the evolving policy.

The bank is persisted every 10 steps and at every model checkpoint, using atomic writes (write to temp file, then rename) to prevent corruption. On training resume, the bank is restored from the checkpoint corresponding to the same global step as the model weights, ensuring consistency between the policy parameters and the memory contents.

##### D.6. Prompt Templates

- D.6.1. Reflection Generation Prompt The policy model receives the following system prompt when generating rubric-grounded reflections:

Reflection System Prompt

You are a strict postmortem editor for long-form deep research QA. You will receive:

- - a question
- - a research trajectory ending in a final <answer>...</answer>

Your job is not to answer the question again. Your job is to extract the most useful guidance for:

- 1. a stronger next attempt on this question, and

- 2. similar long-form research questions.

First, read the final <answer>. Judge it on: correctness and calibration; instruction-following and coverage; research quality and verification; synthesis and insight; communication and structure.

Then identify the 2-4 highest-leverage strengths or weaknesses, trace them back to the visible trajectory, and turn them into reusable guidance.

Rules:

- - Stay grounded in the provided question and trajectory.
- - Do not invent flaws just to be critical.
- - Do not speculate about hidden reasoning not shown in the trajectory.
- - Do not rewrite the answer.
- - Do not give generic advice like "be more careful."
- - Write specific guidance that is grounded in this case but transferable to similar questions.

Output format (exactly two blocks): <reflection_rubrics> Critical Requirement / Analytical Approach / Communication & Structure / Pitfalls </reflection_rubrics> <reflection_takeaways> 1-4 concise portable lessons </reflection_takeaways>

The user message provides the question and the full research trajectory, instructing the model to examine the final answer, trace strengths and weaknesses, and produce both output blocks.

##### D.6.2. Judge Scoring Prompt

An LLM judge (Gemini) evaluates each reflection candidate with privileged access to the graded trajectory, including per-rubric scores and evaluator justifications. The judge scores each candidate on how useful it would be for future attempts:

Reflection Judge System Prompt (abridged)

You are an expert judge evaluating reflection candidates produced by a research agent.

Scoring Dimensions (each 0.0 to 1.0): Diagnostic Accuracy (weight 0.4)

- Does it correctly identify the main strengths and weaknesses? Is it aligned with the evaluator

justifications and visible trajectory? Penalize: missing a major failure clearly flagged by the evaluator; inventing problems not supported by the trajectory; misidentifying root causes.

Specificity (weight 0.3)

- Is the guidance concrete, actionable, and tailored for next attempt of the question? Penalize: generic advice ("be more thorough"); checklist items that could apply to any task; merely restating evaluator language.

Scope & Balance (weight 0.3)

- Is the guidance helpful for other questions? Are takeaways complementary? Penalize: missing an important category; repeating the same point across sections.

Rules:

- - Score based on the reflection’s quality as future guidance, not on the trajectory’s quality.
- - A reflection on a perfect trajectory can score low if it is generic.
- - A reflection on a failed trajectory can score high if it precisely identifies root causes.

The user message provides the original question, all reflection candidates, and the graded trajectory with evaluator feedback. The judge returns per-candidate scores across all three dimensions.

##### D.6.3. Injection Formats

Retrieved bank items are injected as a <reference_examples> block appended to the user message before tokenization. The two retrieval modes from Section 3.4—within-episode refinement and cross-episode transfer—use distinct preambles to condition the agent’s use of the retrieved guidance.

Within-episode injection. When the agent revisits a question it has attempted before (repeat phase of the windowed curriculum), the accepted reflection from the previous encounter is injected:

Within-Episode Injection Format

<reference_examples> You have attempted this exact question before. Below are the rubrics and reflections from your previous attempt. Use them to improve your planning and answering -- avoid past mistakes and build on what worked.

## Your Previous Attempt on This Question: ### Rubrics: {rubrics from bank item}

### Takeaways: {reflections from bank item} </reference_examples>

{Original user question}

Cross-episode injection. For a new query (new-query phase), the bank retrieves the top-𝑘 semantically similar items and presents them as rubric-grounded exemplars:

Cross-Episode Injection Format

<reference_examples> Below are rubrics and reflections from similar questions that were previously analyzed. Use them as reference to guide your planning and answering -adapt to the current question’s specific needs. Do NOT copy them verbatim; extract what is relevant and adjust.

- ## Similar Question 1: {question from bank item 1} ### Rubrics:

- {rubrics from bank item 1} ### Takeaways:

- {reflections from bank item 1}

--## Similar Question 2: {question from bank item 2} ### Rubrics: {rubrics from bank item 2} ### Takeaways:

- {reflections from bank item 2} </reference_examples> {Original user question}

In both cases, the reference block precedes the original user question, so the agent sees the guidance before beginning its structured plan. The within-episode format signals direct self-improvement (“avoid past mistakes”), while the cross-episode format signals analogical transfer (“adapt to the current question’s specific needs”).

### E. Theoretical Analysis

##### E.1. Value of Stage Information

In the main text, Theorem 1 is stated using the informal notation ℎ, 𝑐, 𝑧 inside conditional expectations for readability. In this appendix, we make the probability space explicit: 𝐻, 𝐶, 𝑍 denote random variables, while ℎ, 𝑐, 𝑧 denote realized values. We also slightly sharpen the strictness condition in the theorem by requiring the compared stages to have positive conditional probability given 𝑐; without this requirement, the pointwise quantities 𝔼[𝑈(𝐻, 𝑎) | 𝐶 = 𝑐, 𝑍 = 𝑧] on impossible stage events are version-dependent.

The result below is a value-of-information statement specialized to structured agent reasoning. It compares two information structures, 𝜎(𝐶) and 𝜎(𝐶, 𝑍). Unlike full state-abstraction theorems, it does not assume that (𝐶, 𝑍) is sufficient for the entire history 𝐻; it only quantifies the gain from making the stage label explicit under a compressed decision state.

- Assumption 1 (Setup for Theorem 1). Throughout this subsection, the following assumptions hold.

- 1. Underlying probability space. There exists a probability space (Ω, F, ℙ) and a random decision history

𝐻 : Ω → H,

where (H, B(H)) is a measurable space. The law of 𝐻 is denoted by 𝑑 := ℙ ◦ 𝐻−1. We interpret 𝐻 as a random reachable decision point sampled from some fixed distribution over histories.

- 2. Finite action space. The action space A is a finite nonempty set. Consequently, every maximization and argmax over A is well-defined and nonempty.
- 3. Compressed context and stage label. There exists a measurable map 𝜙 : H → C

into a standard Borel space (C, B(C)), and a measurable stage map

𝜓 : H → [𝐾] := {1, . . . , 𝐾}, such that

𝐶 := 𝜙(𝐻), 𝑍 := 𝜓(𝐻). Since C is standard Borel and [𝐾] is finite, the regular conditional objects used below exist.

- 4. Integrable utility. For each 𝑎 ∈ A, there exists a measurable utility function 𝑈(·, 𝑎) : H → ℝ

such that

𝔼 |𝑈(𝐻, 𝑎)| < ∞.

In our application, 𝑈(𝐻, 𝑎) can be instantiated as a continuation utility or continuation value associated with choosing action 𝑎 at decision point 𝐻.

- Definition 1 (Conditional mean utilities and stage probabilities). Adopt Assumption 1. For each 𝑎 ∈ A, fix measurable versions

𝑞¯(·, 𝑎) : C → ℝ, 𝑞(·, ·, 𝑎) : C × [𝐾] → ℝ such that

𝑞¯(𝐶, 𝑎) = 𝔼[𝑈(𝐻, 𝑎) | 𝜎(𝐶)] almost surely,

and

𝑞(𝐶, 𝑍, 𝑎) = 𝔼[𝑈(𝐻, 𝑎) | 𝜎(𝐶, 𝑍)] almost surely. For each 𝑧 ∈ [𝐾], fix a measurable version

𝑝(𝑧 | ·) : C → [0, 1] such that

𝑝(𝑧 | 𝐶) = ℙ(𝑍 = 𝑧 | 𝜎(𝐶)) almost surely. We write

ℙ𝐶 := ℙ ◦ 𝐶−1 for the law of 𝐶, and define the stage support at a context 𝑐 ∈ C by

𝑆(𝑐) := {𝑧 ∈ [𝐾] : 𝑝(𝑧 | 𝑐) > 0}.

Remark 1 (Well-definedness). Because A is finite and each 𝑈(𝐻, 𝑎) is integrable, the random variables 𝑞¯(𝐶, 𝑎) and 𝑞(𝐶, 𝑍, 𝑎) are integrable for every 𝑎 ∈ A. Indeed, by Jensen’s inequality for conditional expectations,

|𝑞¯(𝐶, 𝑎)| = |𝔼[𝑈(𝐻, 𝑎) | 𝜎(𝐶)]| ≤ 𝔼[|𝑈(𝐻, 𝑎)| | 𝜎(𝐶)] a.s., and therefore

𝔼 |𝑞¯(𝐶, 𝑎)| ≤ 𝔼 |𝑈(𝐻, 𝑎)| < ∞. The same argument gives

𝔼 |𝑞(𝐶, 𝑍, 𝑎)| < ∞. Hence all expectations below are finite.

- Definition 2 (Flat and stage-conditioned decision values). Adopt Assumption 1 and Definition 1. We define

𝑉flat := 𝔼 max 𝑎∈A

𝑞¯(𝐶, 𝑎) , 𝑉stage := 𝔼 max 𝑎∈A

𝑞(𝐶, 𝑍, 𝑎) .

These are exactly the appendix versions of the quantities appearing in Theorem 1 in the main text. Equivalently,

𝑉flat = 𝔼 max 𝑎∈A

𝔼[𝑈(𝐻, 𝑎) | 𝐶] , 𝑉stage = 𝔼 max 𝑎∈A

𝔼[𝑈(𝐻, 𝑎) | 𝐶, 𝑍] .

The following lemma is the only structural identity needed in the proof of Theorem 1.

Lemma 1 (Collapsing stage-conditioned utility to the flat information structure). Adopt Assumption 1 and Definition 1. Then for every action 𝑎 ∈ A,

###### ∑︁𝐾

𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎) almost surely.

𝑞¯(𝐶, 𝑎) =

𝑧=1

Proof. Fix an arbitrary action 𝑎 ∈ A. Define the random variable

###### ∑︁𝐾

𝑅𝑎 :=

𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎).

𝑧=1

We will prove that 𝑅𝑎 is a version of 𝔼[𝑈(𝐻, 𝑎) | 𝜎(𝐶)]. Since 𝑞¯(𝐶, 𝑎) is also a version of 𝔼[𝑈(𝐻, 𝑎) | 𝜎(𝐶)] by Definition 1, the conclusion will follow.

We first note that 𝑅𝑎 is 𝜎(𝐶)-measurable. Indeed, for each fixed 𝑧, both 𝑝(𝑧 | 𝐶) and 𝑞(𝐶, 𝑧, 𝑎) are 𝜎(𝐶)-measurable, so their product is 𝜎(𝐶)-measurable, and a finite sum of such terms is again 𝜎(𝐶)-measurable.

It remains to verify the defining property of conditional expectation. Let 𝐺 be any bounded 𝜎(𝐶)-measurable random variable. We must show that

𝔼[𝐺𝑅𝑎] = 𝔼[𝐺𝑈(𝐻, 𝑎)].

Starting from the left-hand side and expanding the definition of 𝑅𝑎, we obtain

###### ∑︁𝐾

𝔼[𝐺𝑅𝑎] = 𝔼 𝐺

𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎)

𝑧=1

∑︁𝐾

=

𝔼[𝐺 𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎)] .

𝑧=1

For each fixed 𝑧, the random variable 𝐺𝑞(𝐶, 𝑧, 𝑎) is 𝜎(𝐶)-measurable and integrable. Therefore, by the defining property of conditional expectation,

𝔼[𝐺 𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎)] = 𝔼[𝐺 𝑞(𝐶, 𝑧, 𝑎) 𝔼[1{𝑍 = 𝑧} | 𝜎(𝐶)]]

= 𝔼[𝐺 𝑞(𝐶, 𝑧, 𝑎) 1{𝑍 = 𝑧}] . Summing over 𝑧 yields

###### ∑︁𝐾

𝔼[𝐺 𝑞(𝐶, 𝑧, 𝑎) 1{𝑍 = 𝑧}]

𝔼[𝐺𝑅𝑎] =

𝑧=1

###### ∑︁𝐾

1{𝑍 = 𝑧} 𝑞(𝐶, 𝑧, 𝑎)

= 𝔼 𝐺

𝑧=1

= 𝔼[𝐺𝑞 (𝐶, 𝑍, 𝑎)]

By Definition 1,

𝑞(𝐶, 𝑍, 𝑎) = 𝔼[𝑈(𝐻, 𝑎) | 𝜎(𝐶, 𝑍)] almost surely.

Because 𝐺 is 𝜎(𝐶)-measurable and 𝜎(𝐶) ⊆ 𝜎(𝐶, 𝑍), the random variable 𝐺 is also 𝜎(𝐶, 𝑍)-measurable. Applying the defining property of conditional expectation once more gives

𝔼[𝐺 𝑞(𝐶, 𝑍, 𝑎)] = 𝔼[𝐺 𝑈(𝐻, 𝑎)] . Combining the previous displays, we conclude that

𝔼[𝐺𝑅𝑎] = 𝔼[𝐺𝑈(𝐻, 𝑎)] for every bounded 𝜎(𝐶)-measurable random variable 𝐺. This finishes the proof.

###### □

- Definition 3 (Aliasing gap). Adopt Assumption 1 and Definition 1. For 𝑐 ∈ C, define the stage-aliasing gap

###### ∑︁𝐾

∑︁𝐾

Δalias(𝑐) :=

𝑝(𝑧 | 𝑐) max

𝑞(𝑐, 𝑧, 𝑎) − max

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎).

𝑎∈A

𝑎∈A

𝑧=1

𝑧=1

Since A and [𝐾] are finite and the chosen versions 𝑝, 𝑞 are measurable, the function Δalias : C → ℝ is measurable.

We can now restate and strengthen the theorem from the main text.

Theorem 2 (Restatement and strengthening of Theorem 1). Adopt Assumption 1 and Definitions 1–3. Then

𝑉stage − 𝑉flat = 𝔼[Δalias(𝐶)] ≥ 0. Consequently,

𝑉stage ≥ 𝑉flat. Moreover, for ℙ𝐶-almost every 𝑐 ∈ C, the following are equivalent:

- 1. Δalias(𝑐) = 0.
- 2. There exists an action

argmax

𝑎★𝑐 ∈

𝑞(𝑐, 𝑧, 𝑎).

𝑎∈A

𝑧∈𝑆(𝑐)

In particular, if there exists a measurable set C0 ⊆ C with

ℙ𝐶(C0) > 0 and two distinct stages 𝑧 ≠ 𝑧′ such that for every 𝑐 ∈ C0,

𝑝(𝑧 | 𝑐) > 0, 𝑝(𝑧′ | 𝑐) > 0, argmax 𝑎∈A

𝑞(𝑐, 𝑧, 𝑎) ∩ argmax

𝑞(𝑐, 𝑧′, 𝑎) = ∅,

𝑎∈A

then

𝑉stage > 𝑉flat.

Proof. First, we rewrite 𝑉stage and 𝑉flat to show their subtraction form as in 𝔼 [△alias(𝐶)]. By Definition 2,

𝑉stage = 𝔼 max 𝑎∈A

𝑞(𝐶, 𝑍, 𝑎) .

For each fixed 𝑧 ∈ [𝐾], define

𝑀𝑧(𝐶) := max 𝑎∈A

𝑞(𝐶, 𝑧, 𝑎).

Since A is finite and 𝑞(·, 𝑧, 𝑎) is measurable for each 𝑎, the random variable 𝑀𝑧(𝐶) is 𝜎(𝐶)-measurable. Also, because 𝑍 takes values in the finite set [𝐾],

###### ∑︁𝐾

max

1{𝑍 = 𝑧} 𝑀𝑧(𝐶) almost surely.

𝑞(𝐶, 𝑍, 𝑎) =

𝑎∈A

𝑧=1

Taking conditional expectation given 𝜎(𝐶), we obtain

###### ∑︁ 𝐾

1{𝑍 = 𝑧}𝑀𝑧(𝐶) | 𝜎(𝐶)

𝔼 max

𝑞(𝐶, 𝑍, 𝑎) | 𝜎(𝐶) = 𝔼

𝑎∈A

𝑧=1

∑︁𝐾

𝔼[1{𝑍 = 𝑧}𝑀𝑧(𝐶) | 𝜎(𝐶)]

=

𝑧=1

∑︁𝐾

𝑀𝑧(𝐶) 𝔼[1{𝑍 = 𝑧} | 𝜎(𝐶)]

=

𝑧=1

∑︁𝐾

=

𝑝(𝑧 | 𝐶) 𝑀𝑧(𝐶).

𝑧=1

Taking expectation once more and using the tower property gives

###### ∑︁ 𝐾

𝑝(𝑧 | 𝐶) max

𝑉stage = 𝔼

𝑞(𝐶, 𝑧, 𝑎) .

𝑎∈A

𝑧=1

We next rewrite 𝑉flat. By Definition 2,

𝑉flat = 𝔼 max 𝑎∈A

𝑞¯(𝐶, 𝑎) .

By Lemma 1,

###### ∑︁𝐾

𝑞¯(𝐶, 𝑎) =

𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎) almost surely.

𝑧=1

Substituting this identity into the preceding display yields

###### ∑︁𝐾

𝑉flat = 𝔼 max 𝑎∈A

𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎) .

𝑧=1

Then we can simply see that by Definition 3

###### ∑︁ 𝐾

∑︁𝐾

𝑝(𝑧 | 𝐶) max

𝑞(𝐶, 𝑧, 𝑎) − max

𝑉stage − 𝑉flat = 𝔼

𝑝(𝑧 | 𝐶) 𝑞(𝐶, 𝑧, 𝑎) = 𝔼[Δalias(𝐶)] .

𝑎∈A

𝑎∈A

𝑧=1

𝑧=1

It remains to show that Δalias(𝑐) ≥ 0 for every 𝑐. Fix an arbitrary 𝑐 ∈ C. For every action 𝑎 ∈ A and every stage 𝑧 ∈ [𝐾], we have

𝑞(𝑐, 𝑧, 𝑎) ≤ max

𝑞(𝑐, 𝑧, 𝑎′).

𝑎′∈A

Multiplying both sides by the nonnegative quantity 𝑝(𝑧 | 𝑐) and summing over 𝑧 gives

###### ∑︁𝐾

###### ∑︁𝐾

𝑝(𝑧 | 𝑐) max

𝑞(𝑐, 𝑧, 𝑎′).

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎) ≤

𝑎′∈A

𝑧=1

𝑧=1

Since this inequality holds for every 𝑎 ∈ A, it continues to hold after taking the maximum over 𝑎 on the left-hand side, and this means that Δalias(𝑐) ≥ 0. Therefore

𝑉stage − 𝑉flat = 𝔼[Δalias(𝐶)] ≥ 0,

which implies

𝑉stage ≥ 𝑉flat.

Then we are ready to make characterizations when the stage-aliasing gap is zero. Since A is finite and nonempty, there exists at least one action

∑︁𝐾

𝑎flat𝑐 ∈ argmax 𝑎∈A

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎).

𝑧=1

By Definition 3,

###### ∑︁𝐾

###### ∑︁𝐾

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎flat𝑐 )

Δalias(𝑐) =

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐) −

𝑧=1

𝑧=1

∑︁𝐾

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐) − 𝑞(𝑐, 𝑧, 𝑎flat𝑐 ) .

=

𝑧=1

Each summand on the right-hand side is nonnegative, because 𝑀𝑧(𝑐) is the maximum of 𝑞(𝑐, 𝑧, ·) over A. We now prove the two directions separately.

- (i) If Δalias(𝑐) = 0, then there exists a common maximizer over 𝑆(𝑐). This implies that

∑︁𝐾

𝑧=1

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐) − 𝑞(𝑐, 𝑧, 𝑎flat𝑐 ) = 0. Every term in the sum is nonnegative. Therefore, for every 𝑧 such that 𝑝(𝑧 | 𝑐) > 0, we must have

𝑀𝑧(𝑐) − 𝑞(𝑐, 𝑧, 𝑎flat𝑐 ) = 0, i.e.,

𝑞(𝑐, 𝑧, 𝑎flat𝑐 ) = 𝑀𝑧(𝑐) = max 𝑎∈A

𝑞(𝑐, 𝑧, 𝑎). Hence

𝑎flat𝑐 ∈ argmax 𝑎∈A

𝑞(𝑐, 𝑧, 𝑎) for every 𝑧 ∈ 𝑆(𝑐). Equivalently,

𝑎flat𝑐 ∈

𝑧∈𝑆(𝑐)

argmax

𝑎∈A

𝑞(𝑐, 𝑧, 𝑎),

so a common maximizer exists.

- (ii) If there exists a common maximizer over 𝑆(𝑐), then Δalias(𝑐) = 0. Conversely, assume there exists an action

argmax

𝑎★𝑐 ∈

𝑞(𝑐, 𝑧, 𝑎).

𝑎∈A

𝑧∈𝑆(𝑐)

Then for every 𝑧 ∈ 𝑆(𝑐),

𝑞(𝑐, 𝑧, 𝑎★𝑐) = 𝑀𝑧(𝑐). For 𝑧 ∉ 𝑆(𝑐), we have 𝑝(𝑧 | 𝑐) = 0, so those stages contribute nothing to any weighted sum below. Hence

∑︁𝐾

###### ∑︁𝐾

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎★𝑐) =

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐).

𝑧=1

𝑧=1

Since 𝑎★𝑐 is one feasible action in the maximization,

###### ∑︁𝐾

###### ∑︁𝐾

∑︁𝐾

max

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎★𝑐) =

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐).

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎) ≥

𝑎∈A

𝑧=1

𝑧=1

𝑧=1

On the other hand, the following is obvious and has been shown above

∑︁𝐾

###### ∑︁𝐾

max

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎) ≤

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐).

𝑎∈A

𝑧=1

𝑧=1

Therefore equality holds:

###### ∑︁𝐾

∑︁𝐾

max

𝑝(𝑧 | 𝑐) 𝑞(𝑐, 𝑧, 𝑎) =

𝑝(𝑧 | 𝑐) 𝑀𝑧(𝑐),

𝑎∈A

𝑧=1

𝑧=1

which is exactly

Δalias(𝑐) = 0. Combining (i) and (ii), we conclude that for every 𝑐 ∈ C, Δalias(𝑐) = 0 ⇐⇒

argmax

𝑞(𝑐, 𝑧, 𝑎) ≠ ∅.

𝑎∈A

𝑧∈𝑆(𝑐)

Since all objects are defined only up to ℙ𝐶-null sets through the chosen versions of conditional expectations and conditional probabilities, the equivalence is interpreted for ℙ𝐶-almost every 𝑐.

Finally, we can prove the strict inequality criterion. Assume there exists a measurable set C0 ⊆ C such that

ℙ𝐶(C0) > 0 and two distinct stages 𝑧 ≠ 𝑧′ such that for every 𝑐 ∈ C0,

𝑝(𝑧 | 𝑐) > 0, 𝑝(𝑧′ | 𝑐) > 0, argmax 𝑎∈A

𝑞(𝑐, 𝑧, 𝑎) ∩ argmax

𝑞(𝑐, 𝑧′, 𝑎) = ∅.

𝑎∈A

By the zero-gap equivalence we proved above, and that we know Δalias(𝑐) ≥ 0, this means that Δalias(𝐶) ≥ 0 almost surely, and thus

𝑉stage > 𝑉flat.

This completes the proof. □

Remark 2 (Interpretation). Theorem 2 strengthens the main-text theorem in two ways. First, it identifies the exact gain from making the stage label explicit:

𝑉stage − 𝑉flat = 𝔼[Δalias(𝐶)].

Second, it shows that the gain is strict exactly when the compressed context 𝐶 aliases decision points whose stage-conditioned optimal actions disagree, which is the usual case. This is the precise sense in which explicit stage structure helps reasoning under compressed local context.

##### E.2. Judge-Aligned Stage-Weighted Credit Assignment

This subsection formalizes the main-text intuition behind Theorem 3. As in Appendix E.1, we work on a probability space (Ω, F, ℙ). However, unlike Appendix E.1, which studies a single random decision point, we now analyze full rollouts and their policy-gradient signals.

Setup and notation. We suppress the rollout index 𝑖 used in Section 3.3 and analyze a single generic rollout. Let 𝜏 denote a rollout sampled from a policy-induced distribution 𝑝𝜃(𝜏), and let B𝑘 denote the token set of stage 𝑘, matching the notation in the main text. For each token step 𝑡, let

𝐻𝑡 := (𝑞, 𝑎<𝑡, 𝑜<𝑡) denote the random history before taking action 𝑎𝑡, and define the stage score-function sum

###### Γ𝑘 := ∑︁ 𝑡∈B𝑘

∇𝜃 log𝜋𝜃(𝑎𝑡 | 𝐻𝑡).

We assume 𝜋𝜃(𝑎 | ℎ) is differentiable in 𝜃, the environment dynamics are independent of 𝜃, and Γ𝑘 is square-integrable for every 𝑘.

For each stage 𝑘 ∈ [𝐾], let

𝑅𝑘 : Ω → ℝ denote the observed stagewise judge score, and let

𝑌𝑘 : Ω → ℝ denote a latent true process score for stage 𝑘. Both are assumed integrable. As in Section 3.3, let

Λ = (𝜆𝑘,𝑗) ∈ [0, 1]𝐾×𝐾 be a causal stage-weight matrix satisfying

𝜆𝑘,𝑗 = 0 for 𝑗 < 𝑘, 𝜆𝑘,𝑘 = 1.

Definitions. For each stage 𝑘, define the oracle process-level gradient contribution

∑︁𝐾

𝑔★𝑘 :=

𝜆𝑘,𝑗 𝔼[Γ𝑘𝑌𝑗]. (2)

𝑗=𝑘

This is the stage-𝑘 gradient contribution that would be induced by the stage-dependent return 𝐺𝑖,𝑘Λ := 𝐾𝑗=𝑘 𝜆𝑘,𝑗𝑅𝑖,𝑗 if the latent true stage scores were directly observable.

Define the judge-induced stage-weighted signal

∑︁𝐾

𝜆𝑘,𝑗 𝔼[Γ𝑘𝑅𝑗], (3)

𝑔𝑘Λ :=

𝑗=𝑘

and the terminal-broadcast signal

𝑔𝑘term := 𝔼[Γ𝑘𝑅𝐾], (4)

Finally, define

∑︁𝐾

𝑀𝑘Λ := 𝔼[Γ𝑘𝑌𝐾] −

. (5)

𝜆𝑘,𝑗 𝔼[Γ𝑘𝑌𝑗]

𝑗=𝑘

2

The quantity 𝑀𝑘Λ measures how much oracle process-level signal is omitted when one uses only the final-stage score 𝑌𝐾 instead of the full stage-weighted oracle target.

- Assumption 2 (Judge alignment). For each stage pair 𝑘 ≤ 𝑗, there exists a constant 𝜖𝑘,𝑗 ≥ 0 such that

𝔼 Γ𝑘(𝑅𝑗 − 𝑌𝑗) 2 ≤ 𝜖𝑘,𝑗. (6)

That is, the observed judge score 𝑅𝑗 approximates the latent true score 𝑌𝑗 in the gradient-relevant direction defined by Γ𝑘.

- Theorem 3 (Benefit of stage-weighted credit, informal). If the omitted true intermediate signal outweighs the cumulative judge misalignment, then stage-weighted credit yields a strictly better gradient approximation:

∥𝑔𝑘Λ − 𝑔★𝑘∥2 < ∥𝑔𝑘term − 𝑔★𝑘∥2.

- Theorem 4 (Formal version of Theorem 3). Under Assumption 2, the following hold for every stage 𝑘 ∈ [𝐾].

##### 1. Error of the stage-weighted signal relative to the oracle target:

∑︁𝐾

𝑔𝑘Λ − 𝑔★𝑘 2 ≤

𝜆𝑘,𝑗 𝜖𝑘,𝑗.

𝑗=𝑘

##### 2. Lower bound for terminal broadcast relative to the oracle target:

𝑔𝑘term − 𝑔★𝑘 2 ≥ 𝑀𝑘Λ − 𝜖𝑘,𝐾.

##### 3. Comparison. If ∑︁𝐾

𝜆𝑘,𝑗 𝜖𝑘,𝑗 < 𝑀𝑘Λ − 𝜖𝑘,𝐾,

𝑗=𝑘

then

𝑔𝑘Λ − 𝑔★𝑘 2 < 𝑔𝑘term − 𝑔★𝑘 2 .

In particular, when judge misalignment is sufficiently small relative to the omitted intermediate-stage oracle signal, the stage-weighted signal is strictly closer than terminal broadcast to the intended process-level gradient contribution.

Proof. Fix a stage 𝑘. First, we show (1). By definition 3 and 2,

∑︁𝐾

∑︁𝐾

###### ∑︁𝐾

𝑔𝑘Λ − 𝑔★𝑘 =

𝜆𝑘,𝑗 𝔼[Γ𝑘𝑌𝑗] =

𝜆𝑘,𝑗 𝔼[Γ𝑘𝑅𝑗] −

𝜆𝑘,𝑗 𝔼[Γ𝑘(𝑅𝑗 − 𝑌𝑗)].

𝑗=𝑘

𝑗=𝑘

𝑗=𝑘

Taking the Euclidean norm and applying the triangle inequality yields

###### ∑︁𝐾

𝑔𝑘Λ − 𝑔★𝑘 2 ≤

###### 𝜆𝑘,𝑗 𝔼[Γ𝑘(𝑅𝑗 − 𝑌𝑗)] 2 .

𝑗=𝑘

Applying Assumption 2 results in the desired form,

###### ∑︁𝐾

𝑔𝑘Λ − 𝑔★𝑘 2 ≤

𝑗=𝑘

𝜆𝑘,𝑗 𝜖𝑘,𝑗.

##### Second, we try to prove (2). By definition 4 and 2,

∑︁𝐾

𝑔𝑘term − 𝑔★𝑘 = 𝔼[Γ𝑘𝑅𝐾] −

###### 𝜆𝑘,𝑗 𝔼[Γ𝑘𝑌𝑗] = 𝔼[Γ𝑘(𝑅𝐾 − 𝑌𝐾)]

+ 𝔼[Γ𝑘𝑌𝐾] −

𝑗=𝑘

=:𝐸𝑘

∑︁𝐾

###### 𝜆𝑘,𝑗 𝔼[Γ𝑘𝑌𝑗]

𝑗=𝑘

=:𝐷𝑘

.

From definition 5 we know that ∥𝐷𝑘∥2 = 𝑀𝑘Λ Applying the reverse triangle inequality,

∥𝐸𝑘 + 𝐷𝑘∥2 ≥ ∥𝐷𝑘∥2 − ∥𝐸𝑘∥2, we obtain

𝑔𝑘term − 𝑔★𝑘 2 ≥ 𝑀𝑘Λ − ∥𝔼[Γ𝑘(𝑅𝐾 − 𝑌𝐾)]∥2 . Using Assumption 2 with 𝑗 = 𝐾,

∥𝔼[Γ𝑘(𝑅𝐾 − 𝑌𝐾)]∥2 ≤ 𝜖𝑘,𝐾. Hence

𝑔𝑘term − 𝑔★𝑘 2 ≥ 𝑀𝑘Λ − 𝜖𝑘,𝐾.

##### Finally, we are ready to state the last result. We have shown that

###### ∑︁𝐾

𝜆𝑘,𝑗 𝜖𝑘,𝑗, 𝑔𝑘term − 𝑔★𝑘 2 ≥ 𝑀𝑘Λ − 𝜖𝑘,𝐾.

𝑔𝑘Λ − 𝑔★𝑘 2 ≤

𝑗=𝑘

Therefore, if

∑︁𝐾

𝜆𝑘,𝑗 𝜖𝑘,𝑗 < 𝑀𝑘Λ − 𝜖𝑘,𝐾,

𝑗=𝑘

then

𝑔𝑘Λ − 𝑔★𝑘 2 < 𝑔𝑘term − 𝑔★𝑘 2 .

This proves item 3 and completes the proof. □ Corollary 1 (A sufficient condition from score MSE). Suppose in addition that for some constants 𝐵𝑘 ≥ 0 and 𝛿𝑗 ≥ 0,

𝔼[∥Γ𝑘∥22] ≤ 𝐵𝑘, 𝔼[(𝑅𝑗 − 𝑌𝑗)2] ≤ 𝛿2𝑗 . Then Assumption 2 holds with

𝜖𝑘,𝑗 := √︁𝐵𝑘 𝛿𝑗. Consequently,

∑︁𝐾

𝑔𝑘Λ − 𝑔★𝑘 2 ≤ √︁𝐵𝑘

𝜆𝑘,𝑗𝛿𝑗,

𝑗=𝑘

and

𝑔𝑘term − 𝑔★𝑘 2 ≥ 𝑀𝑘Λ − √︁𝐵𝑘 𝛿𝐾.

Proof. Fix 𝑘 ≤ 𝑗. By Cauchy–Schwarz,

𝔼[Γ𝑘(𝑅𝑗 − 𝑌𝑗)] 2 ≤ 𝔼 ∥Γ𝑘∥2 |𝑅𝑗 − 𝑌𝑗|

1/2

1/2

≤ 𝔼[∥Γ𝑘∥22]

𝔼[(𝑅𝑗 − 𝑌𝑗)2]

≤ √︁𝐵𝑘 𝛿𝑗.

Thus Assumption 2 holds with 𝜖𝑘,𝑗 = √𝐵𝑘𝛿𝑗. The displayed bounds then follow immediately from Theorem 4. □

Remark 3. Theorem 4 and Corollary 1 highlight a fundamental trade-off: the advantage of stageweighted credit depends on the strength of the omitted intermediate-stage oracle signal 𝑀𝑘Λ versus the quality of judge alignment, captured by the score errors 𝛿𝑗. Crucially, this establishes that as long as the intermediate stages contain sufficient true process signal (i.e., 𝑀𝑘Λ is sufficiently large), the benefit of capturing this dense signal strictly outweighs the accumulated noise from an imperfect intermediate judge 𝛿𝑗 > 0. In such regimes, stage-weighted credit serves as a strictly better approximation of the intended process objective than relying solely on terminal broadcast.

##### E.3. Judge-Gated Co-Evolution of Policy and Rubric Bank

This subsection formalizes the third component of SCRIBE in Section 3.4. Unlike Appendix E.1, which studies the value of explicit stage information, and Appendix E.2, which studies stagewise credit assignment, the object here is the self-evolution loop itself: the same shared backbone both solves the current task and produces rubric-grounded reflections that can later be reused within-episode and across episodes.

The formalization below matches our implementation. The task objective updates trajectory tokens, whereas the reflection objective is computed on reflection tokens only. Concretely, the selected rollout is treated as a fixed conditioning context during the reflection update, so the reflection-utility gradient does not backpropagate through the sampled trajectory. The co-evolution effect therefore comes from parameter sharing: task updates and reflection updates act on the same backbone parameters, even though they are computed from different generated token blocks.

The key assumption below is a judge-gated local positive-transfer condition. This is not intended as a global claim that every generated reflection is always useful throughout training. Rather, it isolates the local regime in which reflections that are accepted by the judge behave like a helpful auxiliary objective. Similar gradient-similarity and positive-transfer conditions are standard in the auxiliary-learning and multi-task optimization literature: helpful auxiliary objectives tend to align with the main-task gradient, whereas misaligned gradients induce negative transfer (Du et al., 2019; Liu et al., 2021; Wang et al., 2021; Wu et al., 2020; Yu et al., 2020). At the same time, not all task combinations are beneficial to train together (Standley et al., 2020). This perspective is also consistent with theory on shared representation learning and task relevance in transfer (Chen et al., 2022; Maurer et al., 2016; Tripuraneni et al., 2020).

- Assumption 3 (Setup for judge-gated co-evolution). Throughout this subsection, let Θ ⊆ ℝ𝑑 be an open parameter domain. The following assumptions hold.

##### 1. Query distribution for the task objective. There is a query random variable 𝑄 ∼ D.

- 2. Current rubric bank. At the current training step, the rubric bank is a fixed measurable object M. This corresponds to analyzing a single parameter update while treating the bank state as fixed.
- 3. Task rollout distribution. Given query 𝑄 = 𝑞 and bank M, a rollout 𝑇 ∼ 𝑝𝜃(· | 𝑞, M)

is sampled from the deployed task policy.

- 4. Fixed reflection-context distribution. There exists a fixed measurable distribution 𝜉 over query– trajectory pairs

( 𝑄, 𝑇) ∼ 𝜉.

One may think of 𝜉 as the distribution of trajectories selected for reflection training at the current update, e.g., after first sampling rollout groups under the current behavior policy and then selecting one trajectory per query. For the local analysis in this subsection, 𝜉 is treated as fixed. This matches the implementation where the reflection objective applies gradients only on reflection tokens and treats the selected trajectory as a fixed prompt / context.

- 5. Shared-backbone reflection generation. Given ( 𝑄, 𝑇) = (𝑞, 𝜏) and bank M, a rubric-grounded reflection

𝑆 ∼ 𝑟𝜃(· | 𝑞, 𝜏, M)

is sampled. Although we write 𝑝𝜃 and 𝑟𝜃 separately for clarity, both are induced by the same underlying autoregressive backbone 𝜋𝜃 under different prompts / contexts.

- 6. Task score, reflection-utility scores, and judge gate. There exist measurable, integrable random quantities

𝑅(𝑄,𝑇;M) ∈ ℝ, Δw( 𝑄, 𝑇, 𝑆;M) ∈ ℝ, Δc( 𝑄, 𝑇, 𝑆;M) ∈ ℝ, and a measurable acceptance indicator

𝐴( 𝑄, 𝑇, 𝑆;M) ∈ {0, 1}.

Here 𝑅 is the judged task score of the deployed rollout, Δw is the judged within-episode usefulness of the reflection, Δc is the judged cross-episode transfer usefulness, and 𝐴 = 1 means that the reflection is judged sufficiently useful to be accepted for downstream adaptation / storage. These objects do not depend directly on 𝜃 except through the sampled random variables and the fixed bank M.

- 7. Differentiability and score-function regularity. For D-almost every 𝑞, the conditional distribution 𝑝𝜃(𝜏 | 𝑞, M) is differentiable in 𝜃. For 𝜉-almost every (𝑞, 𝜏), the conditional distribution 𝑟𝜃(𝑠 | 𝑞, 𝜏, M) is differentiable in 𝜃. Differentiation may be interchanged with the expectations below, and all score-function terms introduced later are square-integrable.
- 8. Smoothness. The objectives 𝐽task and 𝑈 defined below are continuously differentiable. Moreover, 𝐽task is 𝐿𝐽-smooth and 𝑈 is 𝐿𝑈-smooth on Θ.

- Definition 4 (Task objective and judge-gated memory objective). Adopt Assumption 3. Define the task-rollout and reflection score-function sums

Γtraj := ∇𝜃 log 𝑝𝜃(𝑇 | 𝑄, M), Γref := ∇𝜃 log𝑟𝜃(𝑆 | 𝑄, 𝑇, M).

Because both conditional distributions are induced by the same backbone 𝜋𝜃, these admit tokenwise decompositions

###### ∑︁|𝑇|

###### ∑︁|𝑆|

Γtraj =

∇𝜃 log𝜋𝜃(𝑎𝑡 | ℎ𝑡), Γref =

∇𝜃 log𝜋𝜃(𝑠𝑢 | 𝑐𝑢),

𝑡=1

𝑢=1

for the appropriate rollout histories ℎ𝑡 and reflection-generation contexts 𝑐𝑢.

For weights 𝛽w, 𝛽c ≥ 0, define

𝐽task(𝜃) := 𝔼[𝑅(𝑄,𝑇;M)] , where the expectation is with respect to

𝑄 ∼ D, 𝑇 ∼ 𝑝𝜃(· | 𝑄, M), and define

𝑈(𝜃) := 𝔼 𝐴( 𝑄, 𝑇, 𝑆;M) 𝛽wΔw( 𝑄, 𝑇, 𝑆;M) + 𝛽cΔc( 𝑄, 𝑇, 𝑆;M) , where the expectation is with respect to

( 𝑄, 𝑇) ∼ 𝜉, 𝑆 ∼ 𝑟𝜃(· | 𝑄, 𝑇, M). We also define the combined co-evolution objective

𝐽coevo(𝜃) := 𝐽task(𝜃) + 𝑈(𝜃).

- Assumption 4 (Judge-gated local positive transfer). Adopt Assumption 3 and Definition 4. Let 𝑔 := ∇𝐽task(𝜃),

and define the ungated reflection-utility score-gradient random vector

Ψ := Γref 𝛽wΔw( 𝑄, 𝑇, 𝑆;M) + 𝛽cΔc( 𝑄, 𝑇, 𝑆;M) .

Assume there exist constants 𝑝0 ∈ (0, 1] and 𝜇 > 0 such that ℙ(𝐴 = 1) ≥ 𝑝0, ⟨𝑔, 𝔼[Ψ | 𝐴 = 1]⟩ ≥ 𝜇.

That is, reflections that pass the judge are accepted with nontrivial probability, and their conditional expected gradient contribution is positively aligned with the task gradient.

Theorem 5 (Judge-gated shared-backbone co-evolution). Adopt Assumptions 3–4 and Definition 4. Let

𝑔 := ∇𝐽task(𝜃), ℎ := ∇𝑈(𝜃). Then the following hold.

##### 1. Mutual improvement. For every step size 𝜂 > 0,

𝐿𝑈𝜂2

2 ∥𝑔∥22, and

𝑈(𝜃 + 𝜂𝑔) − 𝑈(𝜃) ≥ 𝜂𝑝0𝜇 −

𝐿𝐽𝜂2

2 ∥ℎ∥22. In particular, if

𝐽task(𝜃 + 𝜂ℎ) − 𝐽task(𝜃) ≥ 𝜂𝑝0𝜇 −

2𝑝0𝜇 𝐿𝑈∥𝑔∥22

2𝑝0𝜇 𝐿𝐽∥ℎ∥22

0 < 𝜂 < min

,

,

then a task-improving step also improves the judge-gated memory objective, and a memoryimproving step also improves the task objective.

##### 2. Dominance over task-only training with static memory. Consider the task-only update

𝜃stat+ := 𝜃 + 𝜂𝑔,

which updates the deployed task policy while assigning zero explicit training signal to reflection quality, and the co-evolution update

𝜃co+ := 𝜃 + 𝜂(𝑔 + ℎ). Then

𝐿𝐽𝜂2

2 ∥ℎ∥22. Consequently, if the reflection is sufficiently good,

𝐽task(𝜃co+ ) − 𝐽task(𝜃stat+ ) ≥ 𝜂𝑝0𝜇 − 𝐿𝐽𝜂2∥𝑔∥2∥ℎ∥2 −

- 1

- 2∥ℎ∥22 ,

𝑝0𝜇 > 𝐿𝐽𝜂 ∥𝑔∥2∥ℎ∥2 +

then

𝐽task(𝜃co+ ) > 𝐽task(𝜃stat+ ).

Proof. To prove the desired statement, we first derive the formula for 𝑔 and ℎ. Let 𝜇𝜃(𝑑𝑞, 𝑑𝜏) := D(𝑑𝑞) 𝑝𝜃(𝑑𝜏 | 𝑞, M) denote the joint law of (𝑄,𝑇). By Definition 4,

𝐽task(𝜃) = ∫ 𝑅(𝑞, 𝜏;M) 𝜇𝜃(𝑑𝑞, 𝑑𝜏).

By Assumption 3, differentiation may be interchanged with the integral. Since D does not depend on

𝜃, by the usual log-derivative trick,

###### ∇𝐽task(𝜃) = ∫ 𝑅(𝑞, 𝜏;M) ∇𝑝𝜃(𝜏 | 𝑞, M) D(𝑑𝑞) 𝑑𝜏

= ∫ 𝑅(𝑞, 𝜏;M) 𝑝𝜃(𝜏 | 𝑞, M) ∇ log 𝑝𝜃(𝜏 | 𝑞, M) D(𝑑𝑞) 𝑑𝜏

= 𝔼 Γtraj𝑅(𝑄,𝑇;M) .

We next derive the formula for ℎ. Write

Δ(𝑞, 𝜏, 𝑠;M) := 𝛽wΔw(𝑞, 𝜏, 𝑠;M) + 𝛽cΔc(𝑞, 𝜏, 𝑠;M), so that

𝑈(𝜃) = 𝔼 𝐴( 𝑄, 𝑇, 𝑆;M) Δ( 𝑄, 𝑇, 𝑆;M) . Let

𝜈𝜃(𝑑𝑞, 𝑑𝜏, 𝑑𝑠) := 𝜉(𝑑𝑞, 𝑑𝜏) 𝑟𝜃(𝑑𝑠 | 𝑞, 𝜏, M) denote the joint law of ( 𝑄, 𝑇, 𝑆). Then

𝑈(𝜃) = ∫ 𝐴(𝑞, 𝜏, 𝑠;M)Δ(𝑞, 𝜏, 𝑠;M) 𝜈𝜃(𝑑𝑞, 𝑑𝜏, 𝑑𝑠).

Differentiating under the integral sign gives

∇𝑈(𝜃) = ∫ 𝐴(𝑞, 𝜏, 𝑠;M)Δ(𝑞, 𝜏, 𝑠;M) 𝜉(𝑑𝑞, 𝑑𝜏) ∇𝑟𝜃(𝑠 | 𝑞, 𝜏, M) 𝑑𝑠

= ∫ 𝐴(𝑞, 𝜏, 𝑠;M)Δ(𝑞, 𝜏, 𝑠;M) 𝜉(𝑑𝑞, 𝑑𝜏) 𝑟𝜃(𝑠 | 𝑞, 𝜏, M) ∇ log𝑟𝜃(𝑠 | 𝑞, 𝜏, M) 𝑑𝑠

= 𝔼 𝐴( 𝑄, 𝑇, 𝑆;M) Γref Δ( 𝑄, 𝑇, 𝑆;M)

= 𝔼[𝐴Ψ] := ℎ.

By the definition of 𝐴 and Assumption 4,

𝔼[𝐴Ψ] = ℙ(𝐴 = 1) 𝔼[Ψ | 𝐴 = 1]. and that

⟨𝑔, ℎ⟩ = ⟨𝑔, ℙ(𝐴 = 1) 𝔼[Ψ | 𝐴 = 1]⟩ (7) = ℙ(𝐴 = 1) ⟨𝑔, 𝔼[Ψ | 𝐴 = 1]⟩ (8) ≥ 𝑝0𝜇, (9)

By the assumption 3, 𝑈 is 𝐿𝑈-smooth, for every vector 𝑣 ∈ ℝ𝑑 we have the standard lower bound

𝐿𝑈 2 ∥𝑣∥22.

𝑈(𝜃 + 𝑣) ≥ 𝑈(𝜃) + ⟨∇𝑈(𝜃), 𝑣⟩ −

Applying this with 𝑣 = 𝜂𝑔 and using Eq. 9 yields

𝑈(𝜃 + 𝜂𝑔) − 𝑈(𝜃) ≥ 𝜂⟨ℎ, 𝑔⟩ −

𝐿𝑈𝜂2

𝐿𝑈𝜂2

2 ∥𝑔∥22 ≥ 𝜂𝑝0𝜇 −

2 ∥𝑔∥22.

Likewise, because 𝐽task is 𝐿𝐽-smooth,

𝐿𝐽𝜂2

𝐿𝐽𝜂2

2 ∥ℎ∥22. If

2 ∥ℎ∥22 ≥ 𝜂𝑝0𝜇 −

𝐽task(𝜃 + 𝜂ℎ) − 𝐽task(𝜃) ≥ 𝜂⟨𝑔, ℎ⟩ −

2𝑝0𝜇 𝐿𝐽∥ℎ∥22

2𝑝0𝜇 𝐿𝑈∥𝑔∥22

0 < 𝜂 < min

,

,

then both right-hand sides are strictly positive. To prove the other part, define

𝜙(𝑡) := 𝐽task 𝜃stat+ + 𝑡𝜂ℎ , 𝑡 ∈ [0, 1]. Then

𝜙(1) − 𝜙(0) = 𝐽task(𝜃co+ ) − 𝐽task(𝜃stat+ ) = ∫ 1

𝜙′(𝑡) 𝑑𝑡. By the chain rule,

0

𝜙′(𝑡) = 𝜂 ∇𝐽task(𝜃stat+ + 𝑡𝜂ℎ), ℎ . Since ∇𝐽task is 𝐿𝐽-Lipschitz,

∇𝐽task(𝜃stat+ + 𝑡𝜂ℎ) − ∇𝐽task(𝜃stat+ ) 2 ≤ 𝐿𝐽𝑡𝜂∥ℎ∥2. Hence, for every 𝑡 ∈ [0, 1], by Cauchy-Schwarz

∇𝐽task(𝜃stat+ + 𝑡𝜂ℎ), ℎ ≥ ∇𝐽task(𝜃stat+ ), ℎ − 𝐿𝐽𝑡𝜂∥ℎ∥22. Integrating from 0 to 1 gives

𝐿𝐽𝜂2

2 ∥ℎ∥22.

𝐽task(𝜃co+ ) − 𝐽task(𝜃stat+ ) ≥ 𝜂 ∇𝐽task(𝜃stat+ ), ℎ −

It remains to lower-bound the inner product on the right. Again by 𝐿𝐽-Lipschitz continuity of the gradient,

∇𝐽task(𝜃stat+ ) − 𝑔 2 = ∥∇𝐽task(𝜃 + 𝜂𝑔) − ∇𝐽task(𝜃)∥2 ≤ 𝐿𝐽𝜂∥𝑔∥2. Therefore, by Cauchy–Schwarz,

∇𝐽task(𝜃stat+ ), ℎ = ⟨𝑔, ℎ⟩ + ∇𝐽task(𝜃stat+ ) − 𝑔, ℎ ≥ ⟨𝑔, ℎ⟩ − ∇𝐽task(𝜃stat+ ) − 𝑔 2 ∥ℎ∥2 ≥ ⟨𝑔, ℎ⟩ − 𝐿𝐽𝜂∥𝑔∥2∥ℎ∥2 ≥ 𝑝0𝜇 − 𝐿𝐽𝜂∥𝑔∥2∥ℎ∥2,

where the last step uses Eq. 9. Substituting this bound above yields that

𝐿𝐽𝜂2

𝐽task(𝜃co+ ) − 𝐽task(𝜃stat+ ) ≥ 𝜂𝑝0𝜇 − 𝐿𝐽𝜂2∥𝑔∥2∥ℎ∥2 −

2 ∥ℎ∥22. This proves the desired lower bound. If

- 1

- 2∥ℎ∥22 ,

𝑝0𝜇 > 𝐿𝐽𝜂 ∥𝑔∥2∥ℎ∥2 + then the right-hand side is strictly positive, and therefore 𝐽task(𝜃co+ ) > 𝐽task(𝜃stat+ ).

###### □

- Remark 4 (Interpretation). Theorem 5 isolates the source of the co-evolution gain. The key quantity is a judge-gated local condition on the reflections that are actually accepted for reuse. Under this condition, policy training improves accepted reflection utility, and accepted reflection training improves

task performance. Moreover, the joint update on 𝐽task + 𝑈 yields a strictly larger first-order gain in adapted task value than task-only training with a static memory objective. This is the precise sense in which jointly training the evolving memory policy is stronger than inference-time-only static memory. The co-evolution effect survives because 𝑔 and ℎ still live in the same shared parameter space: a task update changes the future reflection generator, and a reflection update changes the future task policy.

- Remark 5 (Why parameter sharing matters). The first-order co-evolution mechanism above is specific to the shared-backbone design. If rollout generation and reflection generation were parameterized by disjoint blocks 𝜃 = (𝜃𝜋, 𝜃ref) with 𝐽task depending only on 𝜃𝜋 and 𝑈 depending only on 𝜃ref, then

∇𝐽task(𝜃) = (𝑔𝜋, 0), ∇𝑈(𝜃) = (0, ℎref), and hence

⟨∇𝐽task(𝜃), ∇𝑈(𝜃)⟩ = 0.

Thus the mutual-improvement phenomenon in Theorem 5 disappears at first order in the fully decoupled case. This formalizes why a unified backbone can support genuine co-evolution, rather than merely wrapping a static memory module around a separately trained policy.

### F. Experiment Details

We largely follow the experimental setup of DR Tulu (Shao et al., 2025) for training, evaluation, infrastructure, and baseline selection, adapting where necessary to support our proposed components. Below, we provide the specific details.

- F.1. Training Details

- F.1.1. Supervised Fine-Tuning

We fine-tune Qwen3-8B (Yang et al., 2025) using the LLaMA-Factory framework (Zheng et al., 2024) with DeepSpeed ZeRO-3. The SFT data generation process is described in Appendix B.2. Key hyperparameters:

Hyperparameter Value

Base model Qwen3-8B Training epochs 5 Learning rate 4 × 10−5 LR scheduler Cosine with 10% warmup Batch size (per device) 1 Gradient accumulation steps 16 Effective batch size 128 (8 GPUs × 16 accum.) Max sequence length 16,384 tokens Precision BF16 Weight decay 0.0 Tool output masking Span masking on <tool_output> DeepSpeed stage ZeRO-3 GPUs 8 × NVIDIA H100 80GB

We apply span masking to tool output tokens so the model does not receive gradient on search

results, learning only to generate its own reasoning, tool calls, and answers.

##### F.1.2. Reinforcement Learning

We train all RL runs using our modified open-instruct codebase with Ray-based distributed training and vLLM inference engines. The RL setup follows DR. Tulu’s codebase with extensions for stagewise credit assignment and meta-policy training operations. Key hyperparameters:

Hyperparameter Value

Algorithm GRPO / SS-GRPO Rollouts per prompt 8 Unique prompts per step 32 Effective batch size 256 (32 × 8) Learning rate 5 × 10−7 LR scheduler Constant KL coefficient (𝛽) 0.001 KL estimator KL3 Clip ratio (𝜂) PPO-style clipping Temperature 1.0 Max response length 18,432 tokens Max prompt length 8,192 tokens Max total (pack) length 26,624 tokens Max tool calls per trajectory 10 DeepSpeed stage ZeRO-3 with CPU offloading Gradient checkpointing Enabled

Rubric judge

Judge model Gemini Flash Rubric generation model Gemini Flash Rubric buffer cap (per stage) 3, 2, 2, 3

Rubric bank (full RubricEM only) Bank embedding model Qwen3-Embedding-0.6B Retrieval top-𝑘 2 Reflection trajectories sampled 1 Windowed curriculum 𝐾 3 Bank save frequency Every 10 steps

One additional hyperparameter is the stage-dependent matrix Λ mentioned in Section 3.3, which we set to be

1.0 0.4 0.6 0.8 0 1.0 0.4 0.8 0 0 1.0 0.8 0 0 0 1.0

Λ =

.

The training data is the same DR. Tulu RL dataset (rl-research/dr-tulu-rl-data), which contains ∼4.9K diverse deep research queries. All ablation runs (Baseline RL, SS-GRPO, full RubricEM) use the same 600-step budget with 2 nodes, starting from the same RubricEM-SFT checkpoint. The final RubricEM run uses 4 nodes for longer runs.

Reward design. Since the focus of this work is to improve open-ended answer quality through LLM-as-judge feedback, we use only rubric-based judge signals as RL rewards. Concretely, task-policy

rewards come from the evolving stagewise rubrics used by the judge, and reflection-policy rewards come from judge scores over rubric-grounded reflections. We intentionally do not add auxiliary verifiable rewards, such as format rewards, citation rewards, or tool-use heuristics, which have been used in prior deep research RL recipes such as DR Tulu (Shao et al., 2025). This choice isolates the open-ended component of the problem: improvements should come from better semantic planning, research, synthesis, and experience reuse, rather than from optimizing easily checkable surface constraints. We believe future work could further improve performance by combining rubric-based rewards with calibrated citation, grounding, or formatting rewards, but our current design provides a cleaner and more holistic setting for studying RL beyond verifiable rewards.

##### F.2. Evaluation Details

- F.2.1. Long-Form Benchmarks We evaluate on four long-form benchmarks:

- • HealthBench (Arora et al., 2025): Subsampled1000 medical questions spanning patient consultations, clinical guidelines, and health advice. Evaluation uses an LLM-as-judge (GPT-4) that grades each response against per-question rubrics on multiple axes (accuracy, completeness, context awareness, communication). The reported score is the overall rubric satisfaction rate.
- • ResearchQA (Yifei et al., 2025): 756 scientific research questions with expert-authored rubric items. Evaluation uses an LLM-as-judge (GPT-4) that assesses coverage of each rubric item on a 5-point scale (Not at all / Barely / Moderately / Mostly / Completely). The reported score is the average normalized coverage.
- • DeepResearchBench (DRB) (Du et al., 2025): 100 complex research questions requiring long-form reports with citations. Evaluation uses RACE (Report Assessment via Citation Evaluation) scoring, which combines content quality and citation accuracy. The judge (Gemini) evaluates both the substance of the report and whether citations are accurate and well-grounded.
- • ResearchRubrics (Sharma et al., 2026): 101 open-ended deep research prompts across diverse real-world domains, each paired with expert-written, prompt-specific rubrics. Evaluation uses an LLM-as-judge to assess rubric compliance under fine-grained criteria covering factual grounding, reasoning soundness, synthesis quality, relevance, clarity, and citation use. Rubric items include both positive and negative criteria with different weights, and the reported score is the binary score.

- F.2.2. Short-Form Benchmarks We additionally evaluate on four short-form benchmarks to test out-of-domain transfer:

- • SimpleQA (Wei et al., 2024): 1,000 factual questions with exact-match evaluation via LLM grading (correct / incorrect / not attempted).
- • 2WikiMultihopQA (Ho et al., 2020): 1000 multi-hop reasoning questions requiring evidence from multiple sources. LLM-as-judge evaluation.
- • WebWalker (Wu et al., 2025b): 680 web navigation questions. LLM-as-judge evaluation.
- • DeepSearchQA (DSQA) (Gupta et al., 2026): 900 search-intensive questions with exact-match grading.

All short-form benchmarks are evaluated in zero-shot using the same agent pipeline and search tools as the long-form benchmarks. No short-form data is used during RL training, making these evaluations fully out-of-domain.

Note on Benchmarks. Unlike Dr. Tulu, we do not include the SQAv2 (ScholarQA) benchmark (Asai et al., 2024; Bragg et al., 2025) in our evaluation. SQAv2 is highly citation-centric: it primarily measures whether a model can produce precise inline citations to specific academic papers, which corresponds more to the verifiable citation-grounding component of deep research than to the openended answer-quality setting studied here. Our RL objective intentionally uses only rubric-based judge rewards and does not include citation-specific rewards, so SQA-v2 would test a capability that our method is not designed to directly optimize. This does not mean that citation quality is absent from our evaluation: several of the remaining long-form benchmarks, such as ResearchQA and ResearchRubrics, still include citation-, grounding-, or evidence-use-related rubric items. However, these benchmarks evaluate citation use as one part of broader long-form answer quality rather than making precise academic citation the dominant objective. In addition, during SFT data generation, Gemini-3.1-Pro produces few and often unreliable academic citations even under strong prompting and rejection sampling, making it a weak teacher for this specific skill. We therefore leave citationspecific RL and citation-heavy academic evaluation as future work, and focus here on improving semantic quality, coverage, reasoning, and synthesis beyond verifiable rewards.

Additional notes. For a fair comparison with baselines, the main results in Table 1, the RL ablations in Figure 5, and the short-form transfer results in Table 2 are evaluated with direct zero-shot prompts: we do not append rubric-bank entries, previous reflections, or other experience examples at test time. The agent still follows the same benchmark protocol and uses the same external search tools when the task requires tool use. Thus, the reported gains are not simply due to giving RubricEM extra reflection context during evaluation. Although the task policy uses reflection context during RL training, the main evaluations test the resulting task policy directly, suggesting that reflectionconditioned training likely results in higher-quality rollouts. We separately study inference-time experience reuse in Figure 6(c), where rubric-bank entries are explicitly injected during evaluation.

##### F.3. Infrastructure

Our training and evaluation infrastructure builds on the DR Tulu codebase (Shao et al., 2025), which provides the Ray-based distributed RL training loop, vLLM integration for rollout generation, asynchronous tool calling, sample packing, and one-step asynchronous training (Noukhovitch et al.,

- 2024). We use GRPO (Shao et al., 2024) with token-level loss aggregation, tool output token masking (Jin et al., 2025), and a small KL penalty (0.001) for stability. Full hyperparameters are provided in the RL training table above.

We extend the base infrastructure in several ways to support RubricEM:

- • Stagewise scoring pipeline: An asynchronous stagewise judge scoring module runs Gemini API calls in parallel with the training loop. Rubric generation (32 calls per step) and trajectory scoring (256 calls per step) are parallelized using asyncio.gather with exponential-backoff retries (see Appendix C).
- • Meta-policy training pipeline: Reflection generation and meta-policy training are integrated via a three-thread architecture with one-step deferred reflection training, where reflection samples from step 𝑁 are trained in Phase A of step 𝑁+1 while the inference engine generates new rollouts. This adds effectively no extra wall-clock overhead (see Appendix D).
- • Rubric bank module: A thread-safe in-memory store with a FAISS index over query embeddings (Qwen3-Embedding-0.6B on CPU). Bank operations (retrieval, prompt injection, and background reflection generation) are integrated into the data preparation thread with one-step deferred execution (see Appendix D).
- • Checkpoint management: In addition to standard model and optimizer checkpoints (via DeepSpeed), we save the rubric buffer state, rubric bank contents, and dataloader state at regular

intervals to enable seamless training resumption with consistent memory and data ordering.

For evaluation, we use vLLM to serve the model checkpoint on a single GPU and run the agent pipeline with real-time tool execution via the MCP backend. Each benchmark evaluation varies around 2–18 hours depending on the number of examples, the complexity of the query, judging latency, and the average number of tool calls per trajectory.

Search tools. We expose two search tools to the agent. The primary tool is google_search, implemented through Gemin-3-Flash (as the browsing model) with Google Search grounding enabled1. It returns AI-synthesized summaries with grounding snippets, and is mainly used for general web search, fact-seeking questions, and broad real-world information gathering. The second tool is snippet_search, implemented through the Semantic Scholar API2. It retrieves text snippets from academic papers and is mainly used when the query requires scholarly evidence, paper-level context, or scientific literature support. Both tools are called through the same MCP backend, and their outputs are wrapped in <tool_output> blocks before being returned to the agent.

- F.4. Baselines We compare against baselines from three categories, following the selection of Dr. Tulu (Shao et al.,

- 2025) and supplementing with additional recent models. Scores for existing baselines are taken from the Dr. Tulu paper where available; for models not covered, we run the same evaluation pipeline.

- • Closed deep research models: Proprietary systems with full deep research capabilities, including OpenAI Deep Research (OpenAI, 2025), Gemini Deep Research (Google DeepMind, 2025), Perplexity Deep Research (Perplexity AI, 2025), Claude Sonnet Search, Gemini 3.1 Pro + Search, and GPT-5 + Search.
- • Fixed pipeline deep research models: Models that use a fixed multi-stage pipeline (search → synthesis → report) without end-to-end RL training, including WebThinker (Li et al., 2025e) and Ai2 ScholarQA (Singh et al., 2025).
- • Open deep research models: Open-weight models trained end-to-end for deep research, including Search-R1-7B (Jin et al., 2025), WebExplorer-8B (Liu et al., 2025), Tongyi DeepResearch-30BA3B (Tongyi DeepResearch Team et al., 2025), and DR Tulu-8B (Shao et al., 2025) (both SFT and RL checkpoints).

DR Tulu is the most direct comparison, since we use the same training data and operate in the same open deep research setting. Compared to Dr. Tulu, RubricEM uses a different search tool (Google Search with Gemini grounding vs. Serper API) and a different SFT teacher (Gemini-3.1-Pro vs. GPT-5). Our search tool provides richer AI-synthesized summaries but fewer raw webpage URLs; their teacher is stronger but their search tool returns shorter snippets. We discuss this trade-off in the main text and control for it through ablation studies that isolate the contribution of our training recipe from search tool effects.

- G. Algorithm The algorithm of the paper is described in full in Algorithm 1.

- 1https://ai.google.dev/gemini-api/docs/google-search
- 2https://api.semanticscholar.org/api-docs

### H. Limitations and Discussions

Limitations. Our experiments involve long-horizon agentic RL with search tool calls and external LLM judging, which makes the training loop more sensitive to infrastructure instability than standard offline RL or supervised fine-tuning. During training, we occasionally observed API delays and inconsistent network connections, so wall-clock execution and some rollout–judge latencies were not perfectly controlled across all RL steps. In the main large-scale RL run, the training server also had to be shut down and restarted several times. Although we restored training from checkpoints, such interruptions can introduce additional staleness in the asynchronous reflection branch, rubric bank, and judge-feedback pipeline beyond the intended one-step lag. We therefore view our reported results as reflecting a realistic but not fully ideal infrastructure setting; in principle, a more stable, uninterrupted training environment could reduce stale feedback and further improve the efficiency of our method. Another limitation is that we use Gemini Flash as a cost-effective judge for rubric generation, stagewise scoring, and reflection evaluation. A stronger or more specialized judge could likely provide more accurate stage-level credit and higher-quality reflection rewards, especially for subtle long-form research tasks, but would also increase cost and latency. Exploring the scaling behavior of RubricEM with stronger judges, multiple judges, or calibrated judge ensembles is an important direction for future work.

Discussion and broader impact. More broadly, our results suggest that LLM-generated rubrics should be treated not only as evaluation artifacts, but as a general interface for structuring agent behavior, assigning semantic credit, and accumulating reusable experience. They also point to a training-time view of Meta-RL for language agents: rather than using reflection, memory, or selfimprovement only as inference-time prompting mechanisms, one can train a meta-policy during RL so that judged experience directly shapes both the agent’s parameters and its reusable textual memory. While we study open-ended deep research beyond verifiable rewards, the same recipe may transfer to other domains where quality is multidimensional and hard to reduce to exact-answer correctness, such as writing assistance, data analysis, scientific review, tutoring, and complex tool-use workflows. At the same time, rubric-guided meta-policies inherit the risks of their judges and rubrics: poorly specified criteria can reinforce shallow preferences, biased standards, or overconfident synthesis, and reusable memories may propagate these errors across tasks. Future work should study more robust rubric generation, stronger or ensemble judges, human-auditable rubric banks, uncertainty-aware reflection training, and safety-aware criteria for domains where agentic research outputs may influence real decisions.

Algorithm 1 RubricEM: Reinforcement Learning with Stage-Structured GRPO and Reflection Meta-Policy Training

Require: Structured SFT policy 𝜋𝜃, reference policy 𝜋ref, training queries D, tool environment T, judge J, stages S = {Plan, Research, Review, Answer}, stage matrix Λ, rollout size 𝑛, reflection samples 𝑚, active rubric caps 𝐶1:𝐾

- 1: Initialize active judge rubric buffers B𝑞,𝑘act ← ∅ for each query 𝑞 and stage 𝑘
- 2: Initialize persistent judge rubrics B𝑞,𝑘pers when available, and agent rubric bank M ← ∅
- 3: Initialize deferred reflection batch P0ref ← ∅
- 4: for RL step 𝑡 = 1, . . . ,𝑇 do
- 5: (𝑄𝑡, 𝜇𝑡) ← WindowedBatch(D, 𝑡) ⊲ 𝜇𝑡 ∈ {cross,within}
- 6: for all 𝑞 ∈ 𝑄𝑡 do
- 7: 𝐸𝑞 ← Retrieve(M, 𝑞, 𝜇𝑡) ⊲ retrieve reusable experience
- 8: end for ⊲ Deferred reflection meta-policy update
- 9: if P𝑡ref−1 ≠ ∅ then
- 10: Update shared 𝜋𝜃 on reflection tokens using GRPO rewards in P𝑡ref−1
- 11: end if ⊲ Stage-structured task-policy rollout and judging
- 12: Initialize task batch P𝑡task ← ∅
- 13: for all 𝑞 ∈ 𝑄𝑡 in parallel do
- 14: Sample tool-augmented rollouts {𝜏𝑖}𝑛𝑖=1 ∼ 𝜋𝜃(· | 𝑞, 𝐸𝑞; T) under scaffold S ⊲ task policy rollouts
- 15: ΔB𝑞,1:𝐾 ← GenerateStageRubrics(J, 𝑞, {𝜏𝑖}𝑛𝑖=1, B𝑞,act1:𝐾) ⊲ contrast rollouts
- 16: for 𝑘 = 1, . . . , 𝐾 do
- 17: B𝑞,𝑘act ← B𝑞,𝑘act ∪ ΔB𝑞,𝑘
- 18: B𝑞,𝑘 ← B𝑞,𝑘pers ∪ B𝑞,𝑘act
- 19: end for
- 20: 𝑅𝑖,𝑘 ← ScoreStage(J, 𝑞, 𝜏𝑖, B𝑞,𝑘) for all 𝑖 = 1, . . . , 𝑛 and 𝑘 = 1, . . . , 𝐾
- 21: for 𝑘 = 1, . . . , 𝐾 do
- 22: 𝐺𝑖,𝑘Λ ← 𝐾𝑗=𝑘 𝜆𝑘,𝑗𝑅𝑖,𝑗 for all 𝑖
- 23: 𝐴𝑖,𝑘 ←

𝐺𝑖,𝑘Λ − mean𝑖′(𝐺𝑖Λ′,𝑘) std𝑖′(𝐺𝑖Λ′,𝑘) + 𝜖

for all 𝑖

- 24: B𝑞,𝑘act ← PruneByDiscrimination(B𝑞,𝑘act, {𝑅𝑖,𝑘}𝑛𝑖=1, 𝐶𝑘) ⊲ evolve stagewise rubric buffer
- 25: end for
- 26: P𝑡task ← P𝑡task ∪ {(𝑞, 𝜏𝑖, 𝐵𝑖,1:𝐾, 𝐴𝑖,1:𝐾)}𝑛𝑖=1
- 27: Launch PrepareReflectionBatch(𝑞, {𝜏𝑖, 𝑅𝑖,1:𝐾}𝑛𝑖=1, B𝑞,1:𝐾, M, 𝑚) asynchronously
- 28: end for ⊲ Task-policy update
- 29: Update 𝜋𝜃 on P𝑡task with SS-GRPO using stage advantages 𝐴𝑖,𝑘 and KL to 𝜋ref by Eq. 1
- 30: P𝑡ref ← completed asynchronous reflection batches from step 𝑡
- 31: end for
- 32: return trained policy 𝜋𝜃 and agent rubric bank M
- 33: function PrepareReflectionBatch(𝑞, {𝜏𝑖, 𝑅𝑖,1:𝐾}𝑛𝑖=1, B𝑞,1:𝐾, M, 𝑚)
- 34: Sample one trajectory 𝜏𝑠 uniformly from {𝜏𝑖}𝑛𝑖=1 ⊲ fixed reflection context
- 35: Generate reflection candidates {𝑠ℓ}𝑚ℓ=1 ∼ 𝜋refl𝜃 (· | 𝑞, 𝜏𝑠)
- 36: for ℓ = 1, . . . , 𝑚 do
- 37: (𝑢withinℓ , 𝑢crossℓ ) ← JudgeReflection(J, 𝑞, 𝜏𝑠, 𝑠ℓ, 𝑅𝑠,1:𝐾, B𝑞,1:𝐾)
- 38: 𝑢ℓ ← 21(𝑢withinℓ + 𝑢crossℓ )

- 39: end for
- 40: 𝑠★ ← argmaxℓ 𝑢ℓ among valid reflection candidates
- 41: M ← WriteBank(M, 𝑞, 𝑠★) ⊲ store only the best accepted reflection
- 42: return {(𝑞, 𝜏𝑠, 𝑠ℓ, 𝑢ℓ)}𝑚ℓ=1
- 43: end function

