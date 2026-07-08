# arXiv:2605.31433v1[cs.CL]29May2026

## SCOPE: Self-Play via Co-Evolving Policies for Open-Ended Tasks

#### Wai-Chung Kwan1 Aryo Pradipta Gema1 Joshua Ong Jun Leang2 Pasquale Minervini1,3 1University of Edinburgh 2Imperial College London 3Miniml.AI wkwan@ed.ac.uk

[Figure 1]

[Figure 2]

[Figure 3]

Code Models Logs

### Abstract

Self-play can train language models without external supervision. However, existing methods require rule-checkable answers, leaving open-ended tasks dependent on curated prompts or frontier-model judges. We introduce SCOPE, a data-free selfplay framework for open-ended tasks that co-evolves two policies: a Challenger that generates document-grounded tasks, and a Solver that answers them through multi-turn retrieval. A frozen copy of the initial model serves as the self-judge, which writes task-specific rubrics from the source document and grades Solver responses against them. Across three 7–8B instruction-tuned models (Qwen2.5, Qwen3, OLMo-3), SCOPE improves open-ended performance by up to +10.4 points on eight benchmarks and matches or exceeds GRPOdata trained on ∼9K curated prompts. Although trained only on open-ended tasks, SCOPE also improves held-out short-form QA by up to +13.8 points on seven held-out benchmarks, surpassing GRPOdata on all three models. Ablations show that co-evolving the Challenger is necessary to keep tasks near the Solver’s frontier, that gains arise from improvements in both retrieval and synthesis with the relative contribution varying by task, and that rubric generation quality is the bottleneck for self-judging.

### 1 Introduction

Reinforcement learning drives recent advances in language model capabilities [Shao et al., 2024], yet training for open-ended tasks such as deep research, complex QA, and writing still relies on humancurated prompts, reference answers, or frontier-model judges, tethering performance to human-level supervision [Gunjal et al., 2025, Shao et al., 2025, Goel et al., 2025]. Existing approaches face two limits: human-curated supervision caps learning at human performance [Silver et al., 2017, Hughes et al., 2024], while stronger-model judges require access to a more capable model.

Self-play offers a path beyond both ceilings: models have achieved superhuman performance in games by learning entirely from self-generated experience [Silver et al., 2016, 2017, Brown and Sandholm, 2019, OpenAI et al., 2019, Vinyals et al., 2019, Baker et al., 2020, Meta FAIR et al., 2022]. Recent methods apply self-play to language models, co-evolving task-generating and task-solving policies without human-curated data [Liu et al., 2025a, Zhao et al., 2025, Huang et al., 2025a, Yue et al., 2026, Kwan et al., 2025]. Yet this line of work shares a fundamental dependency: tasks must have answers verifiable via string match [Yue et al., 2026], numerical equality [Huang et al., 2025a, Liu et al., 2025a, Kwan et al., 2025], or code execution [Zhao et al., 2025] to compute the reward signal that drives policy improvement.

Open-ended tasks admit many valid responses and lack a unique correct answer, placing them beyond the reach of rule-based verification. Rubric-based evaluation offers an alternative, decomposing

Preprint.

###### Method Open-Ended Data-Free Reward

SPICE [Liu et al., 2025a] × ✓ Rule match Dr. Zero [Yue et al., 2026] × ✓ Rule match R-Zero [Huang et al., 2025a] × ✓ Rule match Abs. Zero [Zhao et al., 2025] × ✓ Code executor OpenSIR [Kwan et al., 2025] × ✓ Rule match

RaR [Gunjal et al., 2025] ✓ × Rubric (frontier LLM) DR Tulu [Shao et al., 2025] ✓ × Rubric (frontier LLM) RPG [Goel et al., 2025] ✓ × Rubric (self-judge) SCOPE ✓ ✓ Rubric (self-judge)

- Table 1: SCOPE is the first to extend data-free self-play to open-ended tasks. Prior self-play requires verifiable answers; rubric-based RL handles open-ended tasks but depends on curated prompts or frontier judges.

response quality into task-specific criteria that a language model can grade. RaR [Gunjal et al., 2025], DR Tulu [Shao et al., 2025], and RPG [Goel et al., 2025] show that rubric-based rewards can support open-ended RL; however, their use of curated prompts or frontier-model judges leaves unresolved the external-supervision bottleneck that motivates data-free self-play.

We introduce SCOPE (Self-play via Co-evolving Policies for Open-Ended tasks), the first framework to extend data-free self-play to open-ended tasks (Table 1). Figure 1 provides an overview of our method. SCOPE co-evolves two policies from the same base model: a Challenger trained to generate open-ended, document-grounded tasks near the Solver’s capability frontier, and a Solver trained to answer them through multi-turn retrieval. A frozen copy of the base model serves as a fixed Judge, writing task-specific evaluation rubrics from the source document and scoring Solver responses against them. At each iteration, the Challenger is rewarded for proposing moderately challenging document-grounded tasks for the current Solver. The Solver is then rewarded for searching effectively and satisfying the Judge’s rubrics on those tasks. This creates the information asymmetry needed for sustained self-play [Liu et al., 2026]: the Challenger and Judge condition on documents the Solver never sees, so it must recover what it needs through retrieval.

We evaluate SCOPE on three 7–8B instruction-tuned models: Qwen2.5-7B [Qwen Team, 2025], Qwen3-8B [Yang et al., 2025], and OLMo-3-7B [Team OLMo, 2025]. Across eight open-ended benchmarks, SCOPE yields substantial improvements, with gains of up to +10.4 points on Qwen2.5-

7B (24.4→34.8; Section 4.2). Despite using no curated prompts, it matches or exceeds GRPOdata trained on ∼9K curated prompts. This holds even on Qwen3-8B, the strongest base model tested and

the setting with the least remaining headroom for improvement (SCOPE/GRPOdata: 43.1 vs. 41.5).

SCOPE also generalises more broadly than curated-prompt training. Although trained exclusively on open-ended tasks, it surpasses GRPOdata on short-form QA across all three model families, with gains of up to +13.8. The contrast is largest on creative writing, a task without factual grounding, where SCOPE surpasses GRPOdata on all three backbones by up to +7.4, while GRPOdata falls below the base model on two of them. We find that co-evolution is essential: a frozen Challenger fails to sustain improvement beyond the first iteration. The quality gates and cosine length penalty each prevent a distinct reward-hacking failure. We further show that the quality of rubric generation, not grading capacity, is the bottleneck for self-judging (Section 5). Performance gains arise from improvements in both retrieval and synthesis, and the relative importance of each depends on the task. Together, these results show that data-free self-play produces transferable gains on open-ended tasks, extending self-improvement beyond the verifiable-answer domains to which it was previously confined.

### 2 Background

We assume access to an unlabelled document corpus D and a pretrained language model π. Given a task q, πθ generates G rollouts o1:G, each performing multi-turn retrieval over D to gather an evidence set d of supporting documents; an answer ai is parsed from each rollout oi. A judge J evaluates each rollout against task-specific rubrics G = {(ck, wk)}Kk=1q , where each ck is a natural-language criterion and wk ≥ 0 is its importance weight [Gunjal et al., 2025, Shao et al., 2025, Goel et al., 2025]. For each criterion, the judge produces a binary verdict bk ∈ {0,1}, collected into the grade

1 2

Creates Task

Writes Rubric

###### Rubric

Task

[Figure 4]

[Figure 5]

Judge J

Challenger Ct

3

Solves Task

Multi-turn retrieval

[Figure 6]

[Figure 7]

Response

Solver St

[Figure 8]

Judge J

4

Grades Responses

###### GRPO Update

[Figure 9]

GRPO Update Format

Rubric

Search

###### Format <think>

Difficulty

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

<think>

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

<search>

<search>

[Figure 19]

[Figure 20]

2/3=0.67

<answer>

<task>

5

Updates Challenger/Solver

Figure 1: Overview of SCOPE. From the same base model, SCOPE trains a Challenger to generate document-grounded open-ended tasks from corpus documents d, and a Solver to answer them through multi-turn retrieval. A fixed Judge derives task-specific rubrics from d and grades Solver responses. Each iteration alternates between (1) Challenger training against the current Solver and (2) Solver training on filtered Challenger-generated tasks with rubric-based rewards (details in Section 3.1).

vector b(o, q, G) ∈ {0,1}K

q. The rubric reward aggregates these verdicts as a weighted average:

Kq

1

wk bk. (1)

g(o, G) =

Kq k=1 wk

k=1

The overall reward combines multiple components R(o) = j λj rj(o), where λj ≥ 0 are weights. We optimise πθ with group relative policy optimisation (GRPO) [Shao et al., 2024], which estimates advantages from within-group reward statistics without a learned value function. For each prompt x, GRPO samples G rollouts o1:G from πθ and computes group-relative advantages:

R(oi) − mean(R(o1:G)) std(R(o1:G))

. (2)

Ai =

The policy is updated by maximising a clipped surrogate objective with KL regularisation against a reference policy πref:

J (θ) = E

G

1 G

min ρi Ai, clip(ρi, 1−ϵ, 1+ϵ)Ai − β DKL πθ ∥πref , (3)

i=1

where ρi = πθ(oi | x)/πθ

(oi | x) is the importance-sampling ratio, ϵ is the clipping range, and β controls KL regularisation strength.

old

### 3 SCOPE

SCOPE organises self-play around three roles, all initialised from the same base model M0 (Figure 1):

- • Challenger Ct (evolving): generates document-grounded tasks via multi-turn retrieval, trained to produce tasks at the Solver’s capability frontier;
- • Solver St (evolving): answers tasks via multi-turn retrieval-augmented generation, trained to maximise rubric scores;
- • Judge J = M0 (fixed): generates task-specific evaluation rubrics from the source document, scores Solver responses against them, and applies quality gates to Challenger-generated tasks.

We denote the policies of Ct and St as πC and πS, respectively; only these two are updated via GRPO [Shao et al., 2024], while J remains fixed at M0 throughout training. We describe how training iterates between the two evolving roles and justify why this process sustains self-improvement in Section 3.2.

#### 3.1 Self-Play Training Loop

Each iteration of the self-play loop (Algorithm 1) consists of two stages: Challenger training and Solver training.

- Stage 1: Train Challenger. For a source document d ∈ D and a specified task type, the Challenger produces a multi-turn rollout oC. During this rollout, it retrieves additional information (evidence set d) and generates a candidate task q (extracted from oC) designed to require multi-turn retrieval to answer. The Judge J first applies a binary quality gate QG(q,d) ∈ {0,1}, which equals 1 if all checks pass: (1) entity identifiability—the task references unambiguous entities that the Solver can locate via retrieval; (2) source relevance—the task is grounded in the provided document. We find that without these gates, the Challenger degenerates to producing generic tasks unrelated to the source document to exploit the reward. For each task that passes, J generates rubrics G. The previous Solver

St−1 then generates N multi-turn rollouts oS1:N, which J grades against the rubrics. We compute the rubric score g(oS,G) as in Equation (1), then average the N grades to obtain the group mean g¯. The Challenger reward combines format compliance with task difficulty:

RC(oC,d) = λfmt · rfmt(oC) + λdiff · [QG(q,d)] · f diff(¯g; τ), (4)

where rfmt(·) is a role-specific format reward—rfmt(oC) checks the Challenger’s think/search/task structure and rfmt(oS) checks the Solver’s think/tool/answer structure. The difficulty reward fdiff targets tasks at the Solver’s capability frontier by comparing g¯ to a target difficulty τ ∈ (0,1); it peaks where feedback variance g¯(1−g¯) is maximised:

fdiff(¯g; τ) = max 0, 1 −

|g¯ − τ| min(τ, 1−τ)

, (5)

with τ = 12, so the reward peaks when the Solver achieves g¯ = 12 and vanishes at the extremes. The Challenger is optimised with GRPO (Equation (3)) to maximise E[RC], using πref = M0 as the reference policy.

- Stage 2: Train Solver. The trained Challenger Ct first generates candidate tasks over D. Each candidate is evaluated: the Judge applies quality gates QG(q,d), generates rubrics G, the previous

Solver St−1 produces rollouts, and the Judge grades them. We discard tasks that fail the quality gates or whose mean rubric score g¯ falls outside a difficulty window [ℓ,u] (ℓ = 0.2, u = 0.8), retaining

only moderate-difficulty tasks. We then optimise the Solver St with GRPO (Equation (3)) on these filtered tasks, again using πref = M0. The Solver reward combines a length-controlled rubric score with format and search rewards:

RS(oS,q,d) = λacc · L(a) · g(oS,G) + λfmt · rfmt(oS) + λsearch · rsearch(oS), (6)

where a is the answer extracted from rollout oS, g(oS,G) ∈ [0,1] is the rubric reward, rfmt(oS) is the format reward, and rsearch(oS) rewards search tool usage. Without length control, the Solver produces increasingly long responses because the rubric judge tends to score longer outputs more favourably [Gunjal et al., 2025, Huang et al., 2025b, Sharma et al., 2025]. L(a) counteracts length inflation with a cosine penalty:

 

1 if |a| ≤ ℓsoft ℓmin + 1−ℓ

(7)

2 1 + cos π · |a|−ℓ

ℓhard−ℓsoft if ℓsoft < |a| < ℓhard ℓmin if |a| ≥ ℓhard

L(a) =

soft

min



where |a| is the answer length in tokens, ℓsoft and ℓhard are soft and hard token limits, and ℓmin is a floor value. Intuitively, L applies no penalty below ℓsoft, then smoothly decays the score multiplier to a near-zero floor at ℓhard, discouraging the model from inflating answers beyond the soft limit without introducing a sharp reward discontinuity.

#### 3.2 Sustainable Self-Improvement

We justify SCOPE by showing it sustains self-improvement across iterations. Following Liu et al. [2026], sustained self-play requires exposing learnable information to the Solver—not merely generating more data. Let d ∼ pD denote a source document, q ∼ Ct(· | d) the task generated by the Challenger, and G ∼ J(· | q,d) the evaluation rubric produced by the Judge. SCOPE creates learnable information through document grounding: the Challenger and Judge observe d when constructing tasks and rubrics, while the Solver sees only q and must close the resulting gap through retrieval. Two conditions determine whether the pipeline exposes learnable information. First, task grounding: the task must carry information about the source document that the Solver cannot answer from parametric knowledge alone. We quantify this as the task–document mutual information:

Ct(q | d) pt(q)

> 0, with pt(q) = Ed∼p

[Ct(q | d)]. (8)

I(q; d) = Ed∼p

log

D, q∼Ct(·|d)

D

Second, a hidden-rubric gap: rubrics must encode document-specific criteria beyond what the task reveals, so that even after observing the task, the Solver’s evaluation depends on information it does not yet possess. We quantify this as the conditional mutual information between the rubric and the source document given the task:

I(G; d | q) = H(G | q) − H(G | q,d) > 0. (9)

This quantity directly measures the learnable information in the pipeline: the rubric content that depends on the document beyond what the task alone reveals. Task grounding ensures there is information the Solver lacks; the hidden-rubric gap ensures that precisely this information determines the learning signal. As the Solver improves, it closes the hidden-rubric gap on current tasks and the learning signal saturates. The Challenger must therefore co-evolve to propose tasks where the gap remains open, sustaining the self-improvement loop.

### 4 Experiments

#### 4.1 Setup

Training Details. We train and evaluate three instruction-tuned models (Qwen2.5-7BInstruct [Qwen Team, 2025], Qwen3-8B [Yang et al., 2025], and OLMo-3-7B-Instruct [Team OLMo, 2025]). Self-play is most relevant for post-trained models because they have already absorbed humancurated data and require self-generated signal to improve further [Kwan et al., 2025]. We train each model with GRPO [Shao et al., 2024] for 3 iterations of 20 steps per role per iteration. We sample training documents from English Wikipedia. The Challenger generates four task types: long-form QA, summarisation, planning, and writing. We use uniform rubric weights (wk = 1 in Equation (1)); Table 9 lists reward weights and length-penalty thresholds. We report the last checkpoint of each iteration.

Baseline. We compare against GRPOdata, which trains each base model with GRPO on DR Tulu’s ∼9K curated prompts for open-ended tasks [Shao et al., 2025] (5K from SearchArena/OpenScholar,

- 4K rubric-annotated prompts from RaR [Gunjal et al., 2025]). GRPOdata uses the same base-model Judge as SCOPE; the only difference is its reliance on externally curated prompts with frontier-model rubrics, versus SCOPE’s fully self-generated tasks and rubrics. This comparison tests whether data-free self-play can match training with curated data.

Benchmarks. We evaluate on eight open-ended benchmarks: deep research (DRB-RACE [Du et al., 2025], ResearchRubrics [Sharma et al., 2025]), scholarly QA (ResearchQA [Yifei et al., 2025], SQA-CS-V2 [Bragg et al., 2025]), planning (ResearchPlanGen [Goel et al., 2025]), user assistance (HealthBench [Arora et al., 2025], WildBench [Lin et al., 2025]), and creative writing (Arena-Hard-CW [Li et al., 2025a]).

To assess whether these gains generalise beyond open-ended tasks, we additionally evaluate on seven short-form QA benchmarks: general QA (NQ [Kwiatkowski et al., 2019], TriviaQA [Joshi et al., 2017], PopQA [Mallen et al., 2023]) and multi-hop QA (HotpotQA [Yang et al., 2018], 2WikiMultiHopQA [Ho et al., 2020], MuSiQue [Trivedi et al., 2022], Bamboogle [Press et al.,

- 2023]). Following [Shao et al., 2025], we subsample benchmarks exceeding 1,000 examples to 1,000.

Deep Research Scholarly QA Planning User Assist. Creative

Model DRB Rubrics ResQA SQAv2 ResPlan HealthB. WildB. AH-CW Avg. Qwen2.5-7B 33.2 13.1 32.0 25.8 32.4 16.1 33.8 8.5 24.4

GRPOdata 52.1 24.4 46.4 33.3 43.1 20.7 35.4 11.9 33.4

- SCOPE iter1 47.5 19.8 46.0 31.0 40.5 17.2 34.6 10.6 30.9
- SCOPE iter2 50.0 24.3 48.5 31.0 43.6 19.4 35.7 11.5 33.0
- SCOPE iter3 53.2 27.3 50.5 32.8 45.4 20.1 35.0 13.9 34.8

Qwen3-8B 49.2 24.2 40.0 37.5 53.9 25.9 45.7 24.9 37.7 GRPOdata 56.8 33.1 51.2 40.4 57.0 25.1 47.0 21.0 41.5

- SCOPE iter1 51.3 25.8 47.8 38.5 55.3 26.2 46.2 26.5 39.7
- SCOPE iter2 55.5 30.1 51.6 39.8 56.8 26.8 46.9 27.6 41.9
- SCOPE iter3 57.6 31.3 53.7 41.8 57.7 28.1 46.1 28.4 43.1

OLMo-3-7B 40.3 19.0 35.9 33.0 51.1 16.2 35.5 14.8 30.7 GRPOdata 50.7 32.8 52.3 37.8 58.8 23.9 42.7 12.8 39.0

- SCOPE iter1 45.4 27.2 44.1 34.8 57.7 19.5 41.4 16.8 35.8
- SCOPE iter2 47.7 31.3 48.3 38.0 58.8 19.9 41.2 18.0 37.9
- SCOPE iter3 47.9 31.6 51.1 37.3 59.5 21.2 40.9 18.7 38.5

- Table 2: Open-ended benchmark results. Average scores improve monotonically across iterations for all three models. Though fully data-free, SCOPE is comparable to GRPOdata overall while improving creative writing where GRPOdata regresses. SCOPE gains +5.4 to +10.4 points by iter-3, with the largest improvements on Deep Research and Scholarly QA. Best score in bold.

General QA Multi-Hop QA

Model NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle Avg. Qwen2.5-7B 56.4 71.6 44.9 49.0 36.0 17.9 45.6 45.9

GRPOdata 68.1 77.0 50.8 63.7 53.6 28.8 52.0 56.3

- SCOPE iter1 64.2 73.7 47.2 57.8 45.8 26.6 51.2 52.4
- SCOPE iter2 67.2 76.8 50.2 65.7 53.1 31.3 50.4 56.4
- SCOPE iter3 74.4 81.0 53.5 65.6 52.3 32.6 58.4 59.7

Qwen3-8B 62.8 77.2 48.7 56.8 52.3 24.4 56.0 54.0 GRPOdata 69.0 79.0 52.5 71.5 57.6 32.7 62.3 60.7

- SCOPE iter1 68.4 78.9 49.3 60.9 55.6 29.7 64.0 58.1
- SCOPE iter2 70.5 81.1 53.3 62.8 55.5 34.0 59.2 59.5
- SCOPE iter3 71.9 80.9 54.5 65.8 60.5 34.4 64.8 61.8

OLMo-3-7B 53.1 61.3 45.4 41.7 32.4 15.7 30.4 40.0 GRPOdata 73.0 64.0 52.0 60.0 32.0 21.0 38.0 48.6

- SCOPE iter1 57.6 67.4 46.0 46.2 39.5 16.7 27.2 42.9
- SCOPE iter2 63.3 71.3 48.9 53.3 44.7 20.9 33.6 48.0
- SCOPE iter3 67.3 71.8 49.2 55.0 43.4 23.1 34.4 49.2

- Table 3: Short-form QA results. SCOPE trains exclusively on open-ended tasks, yet improves

short-form QA by +7.8 to +13.8 points and surpasses GRPOdata across models. This suggests SCOPE generalises more broadly than curated-prompt training.

Evaluation Details. We score open-ended benchmarks with gpt-5.4-mini at T=1.0 and reasoning_effort=high; for short-form QA we use gpt-5-mini at T=1.0. Following prior work [Shao et al., 2025, Li et al., 2025b, Wei et al., 2024], we use LLM-as-judge rather than exact match for short-form QA. Each task allows up to 5 search turns within a 24,576-token context window and 16,384-token answer budget; each search call retrieves up to 3 documents truncated to 500 tokens combined. Qwen3-8B decodes at T=1.0 to avoid repetition loops; Qwen2.5-7B and OLMo-3-7B decode greedily.

#### 4.2 Main Results

Largest gains on research-intensive tasks. All three models see substantial gains that improve monotonically across iterations, inversely related to base model capacity, as Table 2 and Figure 5 show: +10.4 for Qwen2.5-7B (24.4 → 34.8), +7.8 for OLMo-3-7B (30.7 → 38.5), and +5.4 for Qwen3-8B (37.7 → 43.1). SCOPE’s improvements are most pronounced on research-intensive

tasks, with Deep Research and Scholarly QA gaining +11.1 points averaged across models, followed by planning (+8.4), creative writing (+4.3), and user assistance (+3.0)—consistent with SCOPE’s training focus on document-grounded retrieval and synthesis.

SCOPE matches curated-data training. Without curated prompts or frontier-model rubrics, SCOPE matches GRPOdata on average scores across open-ended benchmarks (SCOPE/GRPOdata): 34.8/33.4 on Qwen2.5, 43.1/41.5 on Qwen3, and 38.5/39.0 on OLMo-3. SCOPE improves all models across task types, while GRPOdata regresses on creative writing for Qwen3 (−3.9) and OLMo-3 (−2.0). Because creative writing differs from the document-grounded training tasks, these gains suggest that corpus-grounded self-play builds synthesis skills that transfer across task types (Section 5). SCOPE is comparable even on deep research, the setting most aligned with GRPOdata’s training data.

Short-form QA transfer. SCOPE trains exclusively on open-ended tasks, yet its short-form QA averages reach 59.7 on Qwen2.5, 49.2 on OLMo-3, and 61.8 on Qwen3—gains of 7.8–13.8 points over the respective base models, as Table 3 shows. Across three models, SCOPE yields comparable gains on general QA (+9.2) and multi-hop QA (+11.0), with a modest advantage on the latter, suggesting that training on open-ended tasks builds retrieval and reasoning skills that generalise to short-form QA. On all three backbones, SCOPE surpasses GRPOdata on the short-form average by as much as 3.4 points. This suggests SCOPE yields broader capability gains than training on curated prompts.

Reward design ablation. To confirm that the quality gates and cosine length penalty each prevent a distinct failure mode, we ablate them individually on Qwen3-8B, as shown in Figure 2. All three conditions start from comparable iter-1 performance and track closely through iter-2, but the ablated variants diverge sharply at iter-3: No-QG drops to 37.4 and No-LP collapses to 32.8—both below the 37.7 base model—while SCOPE continues improving to 43.1. The two ablations expose distinct failure modes. Without quality gates, the Challenger drifts towards generic tasks that ignore the source document but still elicit moderate Solver rubric scores, maximising the difficulty reward while degrading the training signal as tasks converge to a narrow distribution. Without the length penalty, the Solver exploits the rubric judge’s tendency to score longer outputs more favourably; on Qwen3-8B, this escalates to compressing thinking tokens to allocate more of the fixed context window to the visible response, eventually collapsing training entirely.

SCOPE No-QG No-LP

48

43.1

44

Avg.Acc.

40

37.4

36

32

32.8

Iter 1 Iter 2 Iter 3

Figure 2: Reward design ablation on Qwen3-8B. No-QG removes the Challenger’s quality gates; No-LP removes the Solver’s cosine length penalty. Reward hacking compounds over iterations and both mitigations are necessary for sustained improvement.

1

### 5 Analysis

We study SCOPE on Qwen3-8B, the strongest base model in our experiments, to understand what drives its improvement and where it plateaus. Since SCOPE depends on all three roles working in concert, we examine a single question from three angles: (1) whether the Challenger must co-evolve with the Solver, (2) what the Solver learns, and (3) what makes the Self-Judge effective.

Co-evolution is necessary. We ask whether the Challenger must co-evolve with the Solver, or whether a fixed Challenger suffices. We compare SCOPE (Challenger and Solver both evolve across three iterations) against a variant that fixes the Challenger at its iter-1 checkpoint while the Solver continues training on fresh rollouts from this static Challenger (“No Co-Evolution” in Figure 3). Both conditions share the iter-1 checkpoint (step 20).

Figure 3 reveals a widening gap between the two conditions. From iter-1 to iter-3, SCOPE gains +3.4 points on average across open-ended benchmarks (39.7 → 43.1), while the frozen Challenger condition gains only +0.8 (39.7 → 40.5). The gap widens over iterations—2.0 points at iter-2, 2.6 at iter-3—because without co-evolution the Challenger’s tasks no longer challenge the improving Solver, diminishing the training signal. Training-time rubric scores in Figure 3 (right) show the same

Downstream Performance

Training Task Difficulty

0.8

SCOPE

SCOPE

Avg.RubricScore

0.71

44

No Co-Evolution

No Co-Evolution

43.1

0.7

0.64

Easier

Avg.Acc.

41.9

42

+2.6

0.6

0.53

+2.0

Harder

0.5

40.5

40

0.50 0.48

39.9

39.7

0.4

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

Figure 3: The Challenger must co-evolve with the1 Solver. (a) On downstream benchmarks, SCOPE improves by +3.4 points over three iterations, whereas a frozen Challenger limits gains to +0.8. (b) Co-evolution keeps average rubric score near the 0.5 sweet spot, while a frozen Challenger drifts to trivially easy tasks (rubric score 0.71 by iter3), weakening the training signal.

Search Answer NQ HotpotQA ResearchQA HealthBench Iter-1 Iter-1 68.4 60.9 47.8 26.2 Iter-1 Iter-3 71.1(+2.7) 61.8(+0.9) 51.3(+3.5) 27.5(+1.3) Iter-3 Iter-1 69.0(+0.6) 64.5(+3.6) 50.0(+2.2) 26.6(+0.4)

- Table 4: SCOPE improves retrieval and synthesis. Starting from the Iter-1 Solver, each row replaces one component with its Iter-3 counterpart. Replacing only the answer generator (Iter-1 search fixed) measures synthesis: turning fixed evidence into the final answer. Replacing only the search trajectory (Iter-1 generator fixed) measures retrieval: finding task-relevant evidence. Gains appear across all benchmarks, with retrieval larger on multi-hop tasks and synthesis larger on singlehop and knowledge-mismatched tasks.

pattern: under co-evolution, mean scores remain near the optimal difficulty of 0.5 across iterations (0.53→0.50→0.48), indicating tasks stay at the Solver’s frontier; without Challenger evolution, scores rise to 0.71 by iter-3, reflecting tasks the Solver has outgrown. Sustained self-play improvement on open-ended tasks therefore requires both policies to co-evolve.

Both retrieval and synthesis improve. As Tables 2 and 3 show, SCOPE’s gains span both open-ended and short-form tasks, raising the question of whether training improves the Solver’s ability to retrieve relevant information or to consolidate it into a coherent answer. We disentangle these components for Qwen3-8B via a controlled replay experiment.1 Starting from the iter-1 Solver as baseline, we swap in one component at a time: the iter-3 answer generator or the iter-3 search trajectory.

4B

8B

32B

| |
|---|

| |
|---|

| |
|---|

- 38

- 39

- 40

- 41

- 42

- 43

- 44

42.2 42.4

Avg.Acc.

41.9 41.9

41.5

39.1

Grading Rubric

- Table 4 shows that SCOPE improves both retrieval and synthesis across all four benchmarks, with the dominant source of gain tracking each task’s bottleneck. When retrieval is the limiting factor—as in multi-hop HotpotQA, which requires chaining queries across bridging entities—retrieval contributes the larger gain (+3.6 vs. +0.9 from synthesis). Conversely, when relevant evidence is accessible through simple queries or the knowledge source is less aligned with the task, synthesis contributes more:

Figure 4: Self-Judge1 effectiveness depends more on rubrics than grading. We vary the model size of either the grader or rubric generator while holding the other fixed at 8B. Average open-ended benchmark performance is stable across grader sizes, but drops sharply with a 4B rubric generator.

+2.7 vs. +0.6 on single-hop NQ, and +1.3 vs. +0.4 on HealthBench. ResearchQA, which demands both multi-step retrieval and long-form integration, shows sizable gains from both (+2.2 and +3.5). These gains confirm that SCOPE improves both retrieval and synthesis across task types.

1We compare iter-1 and iter-3 rather than base and iter-3 because iter-1 is the earliest checkpoint that has learned the multi-turn search format; the comparison would otherwise confound format learning with genuine improvements in retrieval and synthesis.

Rubric quality matters more than grading. SCOPE uses a frozen copy of the base model as Judge, requiring no external supervision. To understand what makes this viable, we separately scale the Judge’s two roles (rubric generation and grading) from 4B to 32B, holding the other fixed at 8B (Figure 4). All conditions branch from a shared iter-1 checkpoint and run iter-2 Solver training only, after the Solver has acquired the multi-turn search format; this isolates judge quality from format learning. Varying only the grader changes the open-ended benchmark average by just 0.7 points (41.5–42.2), whereas a 4B rubric generator drops the average by 2.8 points. Scaling rubric generation to 32B adds only +0.5 over 8B, suggesting that rubric specificity, not grading capacity, is the binding factor.

To explain this gap, we examine the generated rubrics (Table 10 in Section F shows a representative case). We find that 4B rubrics often omit document-specific facts, producing criteria that any on-topic response can satisfy. In contrast, 8B and 32B rubrics tend to ground their criteria in concrete source details—dates, dollar amounts, and named entities—and produce largely overlapping requirements; the few cases where 32B differs tend to be analytical, such as testing the reasoning behind an event rather than just its occurrence.

### 6 Related Work

Data-free self-play. Self-play achieved superhuman performance in various games [Silver et al., 2016, 2017, Vinyals et al., 2019, Brown and Sandholm, 2019, OpenAI et al., 2019, Meta FAIR et al., 2022], demonstrating that agents can discover complex strategies without human data [Baker et al., 2020]. Recent work extends self-play to LLM post-training: Absolute Zero [Zhao et al., 2025] and R-Zero [Huang et al., 2025a] co-evolve task generation and solving via GRPO but reward code execution or math answer correctness; SPICE [Liu et al., 2025a] and Dr. Zero [Yue et al., 2026] ground self-play in an unlabelled corpus but still require verifiable short answers; EVA [Ye et al.,

- 2024] and OpenSIR [Kwan et al., 2025] broaden coverage but rely on frontier reward models or remain in verifiable domains. All prior data-free self-play depends on deterministic correctness or external reward models; SCOPE preserves co-evolution while replacing answer correctness with rubric-based self-judging.

Rubric rewards for open-ended tasks. Rubric-based RL replaces scalar rewards with task-specific evaluation criteria. Prompt-based approaches generate rubrics via off-the-shelf models [Gunjal et al., 2025, Jia et al., 2026] or update them on-policy with search-grounded knowledge [Shao et al.,

- 2025], while Goel et al. [2025] extract goals and rubrics from conference papers via frontier models. Complementary work trains reward models to reason through rubric criteria [Chen et al., 2025], co-trains rubric generators and policies with correctness labels [Sheng et al., 2026], learns rubric generators or judges from preference data [Liu et al., 2025c, Xu et al., 2026, Lv et al., 2026], or formalises rubric rewards as weighted scores with online criterion elicitation [Rezaei et al., 2025]. SCOPE is the first rubric-based self-play method that requires no curated prompts, no external labels, and no frontier supervision.

### 7 Conclusion

We presented SCOPE, which extends data-free self-play to open-ended tasks through documentgrounded rubric decomposition: a frozen self-judge writes task-specific criteria from source documents, providing the reward signal that prior methods could only obtain from verifiable answers. Across three 7–8B model families, we improve open-ended performance by +5.4 to +10.4 points and match or exceed GRPOdata trained on ∼9K curated prompts, without any curated data or frontiermodel supervision. These gains transfer beyond the training domain to short-form QA and creative writing, despite training solely on information-dense open-ended tasks. Our analysis shows that co-evolution is necessary, as a frozen Challenger fails to propose tasks that offer learning value. SCOPE strengthens both retrieval and synthesis, with the relative contribution varying by task. We further find that rubric generation quality, not grading capacity, is the bottleneck for self-judging: once criteria are specific and document-grounded, scaling either component yields diminishing returns. Together, these findings suggest that SCOPE represents an important first step towards scalable, data-free self-improvement on open-ended tasks.

### References

Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Qui nonero Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, and Karan Singhal. HealthBench: Evaluating large language models towards improved human health. CoRR, abs/2505.08775, 2025.

Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula. In Proceedings of the International Conference on Learning Representations (ICLR), 2020.

Rajendra Bhatia and Chandler Davis. A better bound on the variance. The American Mathematical Monthly, 107(4):353–357, 2000. doi: 10.1080/00029890.2000.12005203. URL https://doi. org/10.1080/00029890.2000.12005203.

Jonathan Bragg, Mike D’Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D. Hwang, Peter Jansen, Varsha Kishore, Bodhisattwa Prasad Majumder, Aakanksha Naik, Sigal Rahamimov, Kyle Richardson, Amanpreet Singh, Harshit Surana, Aryeh Tiktinsky, Rosni Vasu, Guy Wiener, Peter Clark, Doug Downey, Yoav Goldberg, Ashish Sabharwal, and Daniel S. Weld. AstaBench: Rigorous benchmarking of AI agents with a scientific research suite. CoRR, abs/2510.21652, 2025.

Noam Brown and Tuomas Sandholm. Superhuman AI for multiplayer poker. Science, 365(6456): 885–890, 2019.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, and Heng Ji. RM-R1: Reward modeling as reasoning. CoRR, abs/2505.02387, 2025.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. DeepResearch Bench: A comprehensive benchmark for deep research agents. CoRR, abs/2506.11763, 2025.

Shashwat Goel, Rishi Hazra, Dulhan Jayalath, Timon Willi, Parag Jain, William F. Shen, Ilias Leontiadis, Francesco Barbieri, Yoram Bachrach, Jonas Geiping, and Chenxi Whitehouse. Training AI co-scientists using rubric rewards. CoRR, abs/2512.23707, 2025.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. CoRR, abs/2507.17746, 2025.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing A multi-hop QA dataset for comprehensive evaluation of reasoning steps. In International Conference on Computational Linguistics, pages 6609–6625, 2020.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-evolving reasoning LLM from zero data. CoRR, abs/2508.05004, 2025a.

Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, Xijun Gu, Peiyi Tu, Jiaxin Liu, Wenyu Chen, Yuzhuo Fu, Zhiting Fan, Yanmei Gu, Yuanyuan Wang, Zhengkai Yang, Jianguo Li, and Junbo Zhao. Reinforcement learning with rubric anchors. CoRR, abs/2508.12790, 2025b.

Edward Hughes, Michael D. Dennis, Jack Parker-Holder, Feryal M. P. Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, and Tim Rocktäschel. Position: Open-endedness is essential for artificial superhuman intelligence. In International Conference on Machine Learning, pages 20597–20616, 2024.

Ruipeng Jia, Junyi Yang, Yuxin Wu, Yongbo Gai, Siyuan Tao, Mengyu Zhou, Jianhe Lin, Xiaoxi Jiang, and Guanjun Jiang. Open rubric system: Scaling reinforcement learning with pairwise adaptive rubric. CoRR, abs/2602.14069, 2026.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Annual Meeting of the Association for Computational Linguistics, pages 1601–1611, 2017.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6769–6781, 2020.

Wai-Chung Kwan, Joshua Ong Jun Leang, Pavlos Vougiouklis, Jeff Z. Pan, Marco Valentino, and Pasquale Minervini. OpenSIR: Open-ended self-improving reasoner. CoRR, abs/2511.00602, 2025.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc V. Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-Hard and BenchBuilder pipeline. In International Conference on Machine Learning, 2025a.

Xiao Li et al. WebThinker: Empowering large reasoning models with deep research capability. arXiv preprint arXiv:2504.21776, 2025b.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. WildBench: Benchmarking LLMs with challenging tasks from real users in the wild. In International Conference on Learning Representations, 2025.

Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, and Jason Weston. SPICE: Self-play in corpus environments improves reasoning. CoRR, abs/2510.24684, 2025a.

Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, Jiayuan Song, Zhengmao Zhu, Wenhu Chen, Pengyu Zhao, and Junxian He. Webexplorer: Explore and evolve for training long-horizon web agents, 2025b. URL https://arxiv.org/

##### abs/2509.06501.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. OpenRubrics: Towards scalable synthetic rubric generation for reward modeling and LLM alignment. CoRR,

###### abs/2510.07743, 2025c.

Wei Liu, Siya Qi, Yali Du, and Yulan He. Self-play only evolves when self-synthetic pipeline ensures learnable information gain. CoRR, abs/2603.02218, 2026.

Changze Lv, Jie Zhou, Wentao Zhao, Jingwen Xu, Zisu Huang, Muzhao Tian, Shihan Dou, Tao Gui, Le Tian, Xiao Zhou, Xiaoqing Zheng, Xuanjing Huang, and Jie Zhou. Learning query-specific rubrics from human preferences for DeepResearch report generation. CoRR, abs/2602.03619, 2026.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Annual Meeting of the Association for Computational Linguistics, pages 9802–9822, 2023.

Meta FAIR, Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, Athul Paul Jacob, Mojtaba Komeili, Karthik Konath, Minae Kwon, Adam Lerer, Mike Lewis, Alexander H. Miller, Sasha Mitts, Adithya Renduchintala, Stephen Roller, Dirk Rowe, Weiyan Shi, Joe Suber, Da Ju, Hugh Zhang, and Markus Zijlstra. Human-level play in the game of Diplomacy by combining language models with strategic reasoning. Science, 378(6624):1067–1074, 2022.

OpenAI, Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, Jonas Schneider, Nikolas Tezak, Jerry Tworek, Peter Welinder, Lilian Weng, Qiming Yuan, Wojciech Zaremba, and Lei Zhang. Dota 2 with large scale deep reinforcement learning. CoRR, abs/1912.06680, 2019.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP, 2023.

Qwen Team. Qwen2.5 technical report. CoRR, abs/2412.15115, 2025. MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Bing Liu, Yunzhong

He, and Afra Feyza Akyürek. Online rubrics elicitation from pairwise comparisons. CoRR, abs/2510.07284, 2025.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, and Pang Wei Koh. DR Tulu: Reinforcement learning with evolving rubrics for deep research. CoRR, abs/2511.19399, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300.

Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, and Bing Liu. ResearchRubrics: A benchmark of prompts and rubrics for evaluating deep research agents. CoRR, abs/2511.07685, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Leheng Sheng, Wenchang Ma, Ruixin Hong, Xiang Wang, An Zhang, and Tat-Seng Chua. Reinforcing chain-of-thought reasoning with self-evolving rubrics. CoRR, abs/2602.10885, 2026.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy P. Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy P. Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of Go without human knowledge. Nature, 550(7676):354–359, 2017.

Team OLMo. OLMo 3. CoRR, abs/2512.13961, 2025. Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop

questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.

Oriol Vinyals, Igor Babuschkin, Wojciech M. Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H. Choi, Richard Powell, Timo Ewalds, Petko Georgiev, Junhyuk Oh, Dan Horgan, Manuel Kroiss, Ivo Danihelka, Aja Huang, Laurent Sifre, Trevor Cai, John P. Agapiou, Max Jaderberg, Alexander S. Vezhnevets, Rémi Leblond, Tobias Pohlen, Valentin Dalibard, David Budden, Yury Sulsky, James Molloy, Tom L. Paine, Caglar Gulcehre, Ziyu Wang, Tobias Pfaff, Yuhuai Wu, Roman Ring, Dani Yogatama, Dario Wünsch, Katrina McKinney, Oliver Smith, Tom Schaul, Timothy Lillicrap, Koray Kavukcuoglu, Demis Hassabis, Chris Apps, and David Silver. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature, 575(7782): 350–354, 2019.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. CoRR, abs/2212.03533, 2022.

Jason Wei et al. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.

Ran Xu, Tianci Liu, Zihan Dong, Tony Yu, Ilgee Hong, Carl Yang, Linjun Zhang, Tao Zhao, and Haoyu Wang. Alternating reinforcement learning for rubric-based reward modeling in nonverifiable LLM post-training. CoRR, abs/2602.01511, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. CoRR, abs/2505.09388, 2025.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing, pages 2369– 2380, 2018.

Ziyu Ye, Rishabh Agarwal, Tianqi Liu, Rishabh Joshi, Sarmishta Velury, Quoc V. Le, Qijun Tan, and Yuan Liu. Scalable reinforcement post-training beyond static human prompts: Evolving alignment via asymmetric self-play. CoRR, abs/2411.00062, 2024.

Li S. Yifei, Allen Chang, Chaitanya Malaviya, and Mark Yatskar. ResearchQA: Evaluating scholarly question answering at scale across 75 fields with survey-mined questions and rubrics. CoRR, abs/2509.00496, 2025.

Zhenrui Yue, Kartikeya Upasani, Xianjun Yang, Suyu Ge, Shaoliang Nie, Yuning Mao, Zhe Liu, and Dong Wang. Dr. zero: Self-evolving search agents without training data. arXiv preprint arXiv:2601.07055, 2026.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. CoRR, abs/2505.03335, 2025.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. PyTorch FSDP: Experiences on scaling fully sharded data parallel. Proceedings of the VLDB Endowment, 16(12):3848–3860, 2023. doi: 10.14778/3611540.3611569.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id= VqkAKQibpq.

### A Limitations

The multi-stage pipeline (Challenger training, rubric generation, and task filtering) requires more compute than single-stage GRPO on curated prompts. This overhead limited our experiments to 7–8B models; whether the gains hold at larger scales (e.g., 32B) remains an open question. Self-play targets the post-data regime where curated supervision has been exhausted, so trading additional compute for continued improvement may be justified.

Broader impacts. SCOPE may reduce dependence on human-curated prompts and frontier-model judges, making self-improvement more accessible to open-weight models and research groups that lack proprietary supervision pipelines. A risk specific to data-free self-play is that the Challenger may generate tasks involving sensitive, biased, or inappropriate content; for example, tasks that synthesise medical misinformation or reproduce stereotypes from the source corpus. Because the Solver trains on these tasks, harmful content can propagate across iterations and entrench undesirable behaviours. The quality gates (Section 3.2) mitigate this in part by filtering ill-formed or source-irrelevant tasks, but they do not screen for harmful content explicitly. Future deployments should add content-safety filtering at the task-generation stage and monitor the training distribution for harmful topic drift.

### B Additional Results

#### B.1 Training Dynamics

Qwen3-8B

OLMo-3-7B

Qwen2.5-7B

38

40

44

36

43.1

39.0

34.8

34

41.9

33.0

37.9

Avg.Acc.

38.5

42

38

33.4

32

41.5

30.9

32.5

39.7

37.2

35.8

40

30

36

40.0

SCOPE

28

38.7

35.3

28.4

38

GRPOdata

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

1

#### Figure 5: Average open-ended benchmark score across three training iterations. SCOPE matches

or exceeds GRPOdata at nearly every iteration, with only OLMo-3-7B at iteration 3 reversing this pattern. Both methods show diminishing gains, and weaker base models benefit most from training.

Performance over iterations. Figure 5 tracks average open-ended benchmark scores for SCOPE and GRPOdata across three iterations. Despite using no curated prompts, SCOPE matches or exceeds GRPOdata at every iteration across all three model families, with the sole exception of OLMo-3-7B at iteration 3 where GRPOdata edges ahead by 0.5 points. Per-iteration gains diminish for both methods, and weaker base models benefit most (+10.4 for Qwen2.5-7B vs. +5.4 for Qwen3-8B).

Qwen3-8B

OLMo-3-7B

Qwen2.5-7B

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

1.00

1.25

1.20

1.00

0.80

1.00

Reward

0.75

0.60

0.80

0.50

0.40

0.60

Total

0.25

Rubric

0.40

0.20

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

1

###### Figure 6: Training-time reward across Solver iterations (total: solid; rubric: dashed). Rubric reward dips at iteration boundaries as the co-evolved Challenger generates harder tasks, then recovers as the Solver adapts. Total reward rises more steadily, driven by stable format and search components.

Reward evolution. Figure 6 shows both total and rubric reward during Solver training; both increase steadily across all three models, confirming that GRPO training consistently improves the Solver. Within this overall trend, rubric reward dips at iteration boundaries for Qwen3-8B and OLMo-3-7B: once the Solver has improved on the current tasks, the co-evolved Challenger responds with harder tasks calibrated to elicit an average Solver score near τ=0.5. Qwen2.5-7B does not exhibit these dips because its rubric reward remains below 0.5 throughout, so the difficulty increase between iterations is less pronounced. The total reward climbs more steadily because format and search rewards are less affected by task difficulty.

Search behaviour. Figure 7 reveals a shared drop-then-recovery pattern: all three Solvers reduce search calls early in iteration 1, then increase usage through iterations 2–3, converging near three

Qwen3-8B

OLMo-3-7B

Qwen2.5-7B

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

3.5

Searches/Rollout

3.0

3.0

3.0

2.5

2.5

2.0

2.0

2.0

1.0

1.5

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

Iter 1 Iter 2 Iter 3

1

- Figure 7: Mean valid search calls per rollout during Solver training. All three models reduce search usage early in iteration 1, then recover and converge near three searches per rollout by iteration 3, matching the search reward saturation point.

searches per rollout—the search reward saturation point. We inspected the rollouts to understand this pattern. Early in training, the Solver relies on parametric knowledge and issues few searches, producing the initial dip even as rubric reward rises; in later iterations, it learns that retrieval improves answer quality and increases its search usage, despite the small search bonus. OLMo-3-7B shows the sharpest contraction (∼3.5 to ∼1.9) before steadily climbing back.

#### B.2 Extended Training Iterations

Deep Research Scholarly QA Planning User Assist. Creative

Iteration DRB Rubrics ResQA SQAv2 ResPlan HealthB. WildB. AH-CW Avg. Base model 49.2 24.2 40.0 37.5 53.9 25.9 45.7 24.9 37.7

- SCOPE iter 1 51.3 25.8 47.8 38.5 55.3 26.2 46.2 26.5 39.7
- SCOPE iter 2 55.5 30.1 51.6 39.8 56.8 26.8 46.9 27.6 41.9
- SCOPE iter 3 57.6 31.3 53.7 41.8 57.7 28.1 46.1 28.4 43.1
- SCOPE iter 4 58.6 32.1 55.0 42.5 58.4 27.8 46.9 29.5 43.9
- SCOPE iter 5 59.1 32.8 55.7 43.0 58.5 28.6 47.0 30.5 44.4
- SCOPE iter 6 59.6 33.4 56.8 43.4 58.8 28.8 47.3 30.3 44.8

Table 5: Extended training beyond three iterations on Qwen3-8B with identical hyperparameters. Average performance rises monotonically from 37.7 to 44.8, with diminishing per-iteration gains (+2.0 to +0.4). Iterations 1–3 account for +5.4 of the total +7.1 point gain. Best per benchmark in bold.

A natural question is whether SCOPE continues to improve beyond three iterations or eventually saturates. We extend training to six iterations on Qwen3-8B, the strongest model in our comparison, keeping all hyperparameters identical. As Table 5 shows, per-iteration gains shrink steadily but remain positive through iteration 6, and three iterations yield the bulk of improvement (+5.4 of +7.1). The sustained gains through iteration 6 demonstrate that Challenger co-evolution enables sustainable self-improvement.

#### B.3 Leave-One-Out Domain Ablation

We train four leave-one-out variants of SCOPE on Qwen3-8B, each excluding one Challenger task domain (long-form QA, summarisation, planning, or writing) while keeping all other settings fixed, to isolate each domain’s contribution to open-ended performance.

Iteration-level trends. Every leave-one-out variant improves monotonically across iterations (Figure 8), confirming that SCOPE training is robust to the composition of the task mixture. However, the full four-domain mixture improves fastest: its margin over the strongest leave-one-out variant

Base model

w/o Planning

w/o Summarisation

SCOPE (full)

w/o Writing

w/o Long-form QA

43.0

- 37

- 38

- 39

- 40

- 41

- 42

- 43

41.5

41.4 41.2

Avg.Acc.

40.0

Iter 1 Iter 2 Iter 3

- Figure 8: Leave-one-out domain ablation across1 iterations on Qwen3-8B, averaged over openended benchmarks. Dashed grey line: base model. All ablations improve monotonically, but the full four-domain mixture improves fastest; its margin over the best leave-one-out variant widens from

- 0.24 (iter 1) to 0.99 (iter 2) to 1.55 (iter 3), indicating that cross-domain transfer compounds with training.

compounds from 0.24 at iteration 1 to 0.99 at iteration 2 and 1.55 at iteration 3. This widening gap suggests that each domain cultivates distinct capabilities whose benefits compound when combined.

Deep Research Scholarly QA Planning User Assist. Creative

Config DRB Rubrics ResQA SQAv2 ResPlan HealthB. WildB. AH-CW Avg. SCOPE 57.6 31.3 53.7 41.8 57.7 28.1 46.1 28.4 43.1

w/o Planning 55.3 29.4 51.5 38.5 54.8 30.9 45.1 25.5 41.4 w/o Writing 55.7 29.2 54.7 39.0 58.1 30.2 42.4 20.3 41.2 w/o Summarisation 54.9 29.9 51.4 38.3 58.1 28.9 44.2 26.3 41.5 w/o Long-form QA 53.9 27.4 49.2 34.8 57.0 30.5 43.2 24.3 40.0

- Table 6: Per-benchmark iter-3 results for leave-one-out domain ablation on Qwen3-8B. Longform QA is the most broadly impactful domain (−3.0 avg, worst on 4/8 benchmarks); writing and planning show concentrated effects on closely related benchmarks (Arena-Hard −8.1, ResearchPlanGen −2.9). Best per benchmark in bold.

Per-benchmark breakdown. Table 6 shows a clear hierarchy at iteration 3. Long-form QA is the most foundational: removing it causes the largest average drop (−3.0 points, roughly twice that of any other single-domain ablation) and the worst scores on four of eight benchmarks (SQA-CS-V2 −7.0, ResearchQA −4.5, ResearchRubrics −3.9, DRB-RACE −3.7), all centred on retrieval, reasoning, and synthesis. The remaining three domains cluster in average impact (−1.5 to −1.9) yet specialise distinctly: removing writing produces the steepest drops on creative and user-facing benchmarks (Arena-Hard −8.1, WildBench −3.7), removing planning primarily degrades structured reasoning (ResearchPlanGen −2.9), and removing summarisation spreads a diffuse deficit across benchmarks without dominating any single one. Each domain thus provides a distinct training signal; their combination compounds rather than plateaus.

#### B.4 Reference Comparison with Related Methods

Compared methods. Several recent methods also train language models for open-ended tasks with search capabilities [Shao et al., 2025, Liu et al., 2025b, Li et al., 2025b]. We consider only methods with publicly released checkpoints, enabling re-evaluation under our protocol. Critically, all of them rely on curated or externally sourced prompts (3K–12K), whereas SCOPE generates its entire training curriculum from a raw corpus. They also differ from SCOPE along additional axes: DR Tulu and WebExplorer require SFT warmup on frontier-model trajectories and frontier-model judges, and all three provide richer tool access (browsing, web crawling) than SCOPE’s search-only

SCOPE (ours) DR Tulu WebExplorer WebThinker-R1 Base Model

Qwen3-8B Qwen3-8B Qwen3-8B QwQ-32B

Task Type

Open-ended Deep research Short-form QA Short-form QA # Seed Prompts

0 ∼9K ∼12K ∼3K

Rubric Source

Self-generated GPT-4.1 — —

Judge Self-judge GPT-4.1-mini — Training Recipe

SFT → GRPO (1.9K steps)

SFT → GRPO (∼380 steps)

DPO (2 iters) External Model Dependency

GRPO (60 steps)

- GPT-4.1

GPT-4.1-mini

- GPT-5

DeepSeek-V3 Gemini 2.5 Flash

None

Qwen2.5-72B-Instruct

- Table 7: Training setup comparison with related methods. SCOPE is the only method that self-generates its entire training curriculum with no frontier model dependencies. WebExplorer and WebThinker train on verifiable QA tasks, where rubric-based evaluation is unnecessary (—).

Deep Research Scholarly QA Planning User Assist. Creative

Method DRB Rubrics ResQA SQAv2 ResPlan HealthB. WildB. AH-CW Avg. SCOPE (ours) 57.6 31.3 53.7 41.8 57.7 28.1 46.1 28.4 43.1 DR Tulu 70.1 40.0 67.5 41.0 68.4 25.3 36.0 4.4 44.1 WebExplorer 55.6 29.2 55.4 39.0 50.3 26.4 41.7 14.6 39.0 WebThinker-R1 40.2 9.5 47.0 37.8 23.0 12.5 36.5 5.7 26.5

- Table 8: Per-benchmark comparison with related methods on open-ended benchmarks. SCOPE is the most balanced, achieving the best score on 4/8 benchmarks (SQAv2, HealthBench, WildBench, Arena-Hard). DR Tulu leads on deep research tasks but scores below the base model on Arena-Hard (4.4) and WildBench (36.0). Best per benchmark in bold.

setup. Because these overlapping confounds prevent controlled comparison, we report results as a reference evaluation rather than including them in the main results.

Evaluation protocol. To ensure a uniform evaluation, we replace each method’s native tools (browsing, paper search, web crawling) with SCOPE’s single retrieval endpoint and grade all outputs with the same LLM judge, while preserving each method’s original prompt format and tool-calling conventions. Table 7 summarises the training differences; Table 8 reports per-benchmark scores.

Results. Despite using no curated prompts and no frontier model supervision, SCOPE achieves 43.1 on average, within 1.0 point of DR Tulu (44.1) and ahead of WebExplorer (39.0) and WebThinker-R1 (26.5), while scoring highest on four of eight benchmarks. The gap is concentrated on researchheavy benchmarks, where DR Tulu’s curated prompts and frontier-model rubrics closely match the evaluation distribution. On user-facing and creative tasks, SCOPE leads despite requiring no external data or frontier-model supervision.

### C Detailed Training Algorithm

Algorithm 1 gives the complete pseudocode of the SCOPE training loop, including rollout collection, quality gating, rubric generation, and difficulty filtering.

Algorithm 1 SCOPE training loop. Require: Corpus D, base model M0, iterations T, batch size B, difficulty window [ℓ,u]

- 1: Challenger C0 ← M0, Solver S0 ← M0, Judge J ← M0
- 2: for t = 1,...,T do

- — Stage 1: Train Challenger —

3: for b ← 1 to B do 4: Sample document d ∼ D 5: Generate task (oC, q) ← ROLLOUT(Ct−1, d) 6: if q passes quality gates QG(q,d) then ▷ Estimate task difficulty 7: Generate rubric G ← GENRUBRIC(q,d) 8: Collect N solver rollouts oS1:N ← ROLLOUT(St−1, q, N) 9: Grade rollouts g¯ ← GRADE(oS1:N, G)

10: Compute reward RC ▷ Equation (4) 11: else

12: RC ← rfmt(oC) ▷ Format reward only 13: end if 14: end for 15: Update Ct−1 → Ct via GRPO

- — Stage 2: Train Solver —

- 16: T ← ∅ ▷ Collect difficulty-filtered tasks
- 17: for each document d ∈ SAMPLE(D) do
- 18: Generate task (_,q) ← ROLLOUT(Ct, d)
- 19: Generate rubric G ← GENRUBRIC(q,d)
- 20: Grade N solver rollouts: g¯ ← GRADE(St−1, q, G, N)
- 21: if QG(q,d) ∧ ℓ ≤ g¯ ≤ u then
- 22: T ← T ∪ {(q, G)}
- 23: end if
- 24: end for
- 25: for b ← 1 to B do
- 26: Sample task (q,G) ∼ T
- 27: Generate response oS ← ROLLOUT(St−1, q)
- 28: Compute reward RS ▷ Equation (6)
- 29: end for
- 30: Update St−1 → St via GRPO
- 31: end for
- 32: return CT, ST

### D Theoretical Justification of the Difficulty Reward

The difficulty reward fdiff(¯g; τ) (Equation (5)) peaks at mean rubric score g¯ = τ = 12, targeting the regime of maximum feedback variance. Under GRPO (Equation (2)), group-relative advantages Ai =

(Ri − mean(R1:G))/std(R1:G) drive policy updates. With population standard deviation, i A2i = G exactly, so the advantage magnitude is constant; the mechanism by which extreme difficulty

degrades training is the breakdown of the empirical standard deviation σˆG in the denominator. For rubric scores g ∈ [0,1] with population mean µq = E[g | q], the Bhatia–Davis inequality [Bhatia and Davis, 2000] gives

Var(g | q) ≤ µq(1 − µq) ≤ 14, (10)

with equality for binary outcomes and the right bound tight at µq = 12. As µq → 0 or 1, Var(g | q) → 0, so σˆG → 0 and Ai becomes either undefined or dominated by sampling noise. The difficulty

reward fdiff and the variance bound µq(1−µq) share the same unique maximiser at µq = 12 and are both monotone decreasing in |µq − 12|, so fdiff ranks tasks by feedback variance in the same order as the bound. The difficulty filter g¯ ∈ [ℓ,u] with ℓ = 0.2, u = 0.8 complements the reward by lower-bounding µq(1−µq) ≥ ℓ(1−ℓ) = 0.16, preventing the variance collapse that stalls training at the extremes.

### E Training Hyperparameters and Configuration

All three models are trained with the same optimiser and schedule on 6 NVIDIA H100 GPUs. Qwen38B and OLMo-3-7B use their native tool-calling syntax, while Qwen2.5-7B employs a custom XML tag format (Section H). Training uses the verl framework [Sheng et al., 2024] with FSDP [Zhao et al., 2023] (parameter and optimiser offloading, gradient checkpointing) and SGLang [Zheng et al., 2024] for rollout inference. Table 9 lists the full hyperparameter configuration.

Table 9: Training hyperparameters shared across all three models.

Category Hyperparameter Value

Algorithm GRPO Learning rate 1 × 10−6 LR warmup ratio 0.03 Gradient clip 1.0 KL coefficient (β) 1 × 10−3 Steps per iteration (per role) 20 Iterations 3

Trainer

Batch size†,‡ 64 / 256 Group size (G) 8 Max prompt length† 1024 / 2048 tokens Max response length 8192 tokens Rollout prompt length 2048 tokens Max model length 16,384 tokens Engine SGLang

Rollout

Rubric weight (λacc) 1.0 Format weight (λfmt) 0.5 Search weight (λsearch) 0.1 Search reward max turns 3

Solver Reward

Length soft limit (ℓsoft) 1024 tokens Length hard limit (ℓhard) 2048 tokens Length penalty floor (ℓmin) 0.05

Format weight (λfmt) 0.5 Difficulty weight (λdiff) 1.0 Difficulty target (τ) 0.5 Solver rollouts per task (K) 8

Challenger Reward

Corpus 2018 English Wikipedia [Karpukhin et al., 2020] Encoding model E5-base-v2 [Wang et al., 2022] Documents per query 3 Max tool response length 500 tokens

Retrieval

Rubric generation temperature 0.0 Rubric generation max tokens 8192 Min rubrics per task 3 Grading temperature 0.6 Grading max tokens 16,384

Self-Judge

† Challenger / Solver. All other values are shared. ‡ The number of unique prompts per gradient update; each prompt generates G rollouts.

Qwen3-8B: per-turn datums. Qwen3-8B natively produces <think> blocks before each assistant response, including in multi-turn tool-use conversations. At inference time, prior thinking is not visible to the model on subsequent turns; however, naively constructing training inputs from full trajectories retains all prior thinking blocks, creating a train–inference mismatch. We split each multiturn trajectory into per-turn training datums: each datum has thinking trimmed from all previous assistant turns while preserved for the current turn being trained on. This ensures the training distribution matches the inference distribution at every turn.

### F Reward Function Details

This section describes the implementation of each reward component, covering the computational pipeline from raw rollout output to final scalar reward.

- F.1 Challenger Format Reward

Each Challenger rollout is decoded into a sequence of assistant turns. Three components are scored and averaged to produce the format reward rfmt ∈ [0,1]:

- 1. Think reward: the fraction of assistant turns containing a <think>...</think> block. Each assistant turn in the decoded conversation is matched against the regex <think>.*?</think>; the think reward equals think_count/num_assistant_turns.
- 2. Tool reward: the number of assistant turns containing a valid <search>...</search> tag (in the search_r1 format), divided by the per-item expected search turn count (sampled from the “4:3:2” stratified distribution over 1/2/3 turns). A search tag is “valid” if it is well-formed XML with non-empty query content. The tool reward is clamped to [0,1].
- 3. Structure reward: 1.0 if the final assistant turn contains a well-formed <task> XML element with the required sub-elements (<question> at minimum), 0.0 otherwise.

The format score is rfmt = (think + tool + structure)/3. Rollouts with rfmt = 0 are excluded from difficulty estimation and receive zero total reward.

- F.2 Solver Format and Search Rewards

Format reward. The full multi-turn conversation (prompt + response) is decoded and segmented into assistant turns. Three components are scored:

- 1. Think reward: the fraction of assistant turns containing <think>...</think> tags. Every assistant turn is expected to open with a reasoning block; turns without one reduce the think reward proportionally.
- 2. Tool reward: the ratio of valid <search> calls in non-final assistant turns to the number of nonfinal turns, i.e., min(valid_search/(num_turns − 1), 1.0). A search call is “valid” if it matches the regex <search>.*?</search> with non-empty content.
- 3. Answer reward: 1.0 if the final response contains a valid <answer>...</answer> block (verified by attempting extraction of the answer text between tags), 0.0 otherwise.

The format score is F(a) = (think + tool + answer)/3. Search reward. The search reward is separate from the tool component of the format score and provides an additive bonus for using retrieval:

valid_search_calls 3

S(a) = min

, 1.0 ,

where “valid_search_calls” counts all well-formed <search> tags across all assistant turns (including the final turn, unlike the tool component of the format score which only considers non-final turns). The denominator of 3 means the reward saturates once the Solver has performed at least 3 retrieval steps, incentivizing multi-hop search without rewarding excessive retrieval beyond the saturation point.

Rubric quality across model scales. Table 10 illustrates how rubric specificity varies with model size on a representative summarisation task.

### G Task Generation Pipeline

Challenger prompt creation. Each iteration generates 2,000–3,000 challenger training prompts by sampling documents from the corpus and pairing them with task types. The number of required search turns per prompt is allocated via a stratified ratio (“4:3:2” for 1/2/3 turns), ensuring diverse retrieval depths. Prompts are shuffled with a per-iteration seed to ensure non-overlapping data across iterations.

Task: Summarise events behind David Tuckman’s arena project and basketball franchise in Bellevue, WA. Qwen3-4B

- 1. Identifies key events and motivations of Tuckman’s initiative.
- 2. Accurately summarises financial details and project goals.

- 1. Synthesises essential information without omitting the $150 million project and 2001 proposal.
- 2. Connects the arena project to the team’s role as anchor tenant.
- 3. Correctly attributes the $125,000 franchise purchase to Tuckman in 2003.

Qwen3-8B

- 1. Identifies key events including the 2001 arena proposal and 2003 CBA franchise purchase.
- 2. Incorporates $150 million arena cost and $125,000 acquisition fee.
- 3. Explains motivations such as securing long-term tenants for financing and establishing an anchor tenant.

Qwen3-32B

- Table 10: Rubrics from Qwen3-4B, 8B, and 32B on a summarisation task. Red: generic prompt restatements; blue: source-grounded facts; green: analytical requirements. 4B rubrics omit concrete facts, so generic responses can satisfy them. 8B and 32B both require specific details from the source document and produce largely overlapping criteria; here, 32B also adds deeper analytical requirements.

Task creation and filtering. After Challenger training (Stage 1), the trained Challenger generates candidate tasks over fresh corpus samples. Each candidate task undergoes:

- 1. Quality gating: the Judge evaluates entity identifiability, source relevance.
- 2. Rubric generation: the Judge produces 3–5 task-specific rubrics from the task and source document.
- 3. Difficulty estimation: Nfilter = 4 solver rollouts are generated and graded; the mean rubric score g¯ is computed.
- 4. Difficulty filtering: tasks with g¯ ∈/ [0.2,0.8] are discarded.

The pipeline targets 5,120 filtered tasks per iteration (256 batch × 20 steps), generating candidate tasks in batches of 200–500 until the target is met. At most one task is retained per source document (max_tasks_per_prompt=1) to ensure diversity.

Note that the number of solver rollouts differs between stages: Stage 1 Challenger training uses Ntrain = 8 rollouts for difficulty estimation (higher variance reduction during gradient computation), while Stage 2 filtering uses Nfilter = 4 rollouts (sufficient for the coarse accept/reject decision).

H Prompt Templates

This section provides the exact prompt templates used during training. The Challenger and Solver prompts adapt the search-action syntax to each model’s native tool-calling convention; all other elements—task emission (<task>), final answers (<answer>), and all Judge prompts—use the same XML format across all three models. Table 11 summarises the model-specific search syntax; the full templates below show the Qwen2.5-7B variant.

Model Search action Qwen2.5-7B <search>query</search> Qwen3-8B <tool_call>{"name":"search", "arguments":{...}}</tool_call> OLMo-3-7B <function_calls>search(query="...")</function_calls>

- Table 11: Model-specific search-action syntax in Challenger and Solver prompts. Only the search call differs; task emission (<task>) and answer submission (<answer>) use XML tags for all models.

Quality gate prompts. Two quality gate prompts validate Challenger-generated tasks (Figures 13 and 14). All gates output binary scores (0 or 1) using the <think>...<score> format.

Task type descriptions. The Challenger selects from five task types during training (creative writing is defined but excluded):

Generate a challenging long-form task using the source document and retrieved information. ## Source Document {source_document} ## Task Type You must generate a task of type: **{task_type}** {task_description} ## Instructions You will use exactly **{num_search_turns}** search turns. Each search must be about ONE specific entity that appears by name in the PREVIOUS search results — not the source document topic. When told your searches are complete, produce the final task. The final task MUST require unique facts from ALL search turns — if any search turn’s results could be removed without affecting the answer, the task is invalid. ## Search Turn Format <think>Analysis of search results... What new entity or fact to follow next...</think> <search>your search query</search> ## Final Task Format (when told searches are complete) <think>Your final reasoning here... Task Prompt Verification:

- - [ ] Each search turn contributed a unique essential fact not in the source document
- - [ ] The task requires ALL of these facts to answer fully
- - [ ] Entities are identifiable (full names + generic qualifier if needed; acronyms expanded)
- - [ ] No answer leakage (no specific facts/dates/stats/secondary names)
- - [ ] Requires retrieval (cannot be answered from the prompt alone)
- - [ ] Specifies output format
- - [ ] English only </think> <task>The task prompt for the solver. Must withhold searchable information.</task>

- Figure 9: Challenger prompt template (Qwen2.5-7B, custom XML format). The Challenger receives a source document and task type, performs multi-turn search, then produces a task. Model-specific search-action variants are listed in Table 11. Template variables shown in {braces}.

Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search up to {max_search_turns} times, so plan your searches carefully. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.{budget_instructions} Question: {question}

- Figure 10: Solver prompt template (Qwen2.5-7B, custom XML format). The Solver reasons in <think> blocks, searches via <search> tags, and produces a final answer in <answer> tags. Modelspecific search-action variants are listed in Table 11.

You are an expert grader evaluating a user’s response to a task prompt against a SINGLE rubric. Be strict: only award credit when the rubric is completely and unambiguously satisfied. <prompt>{prompt}</prompt> <response>{response}</response> <rubric>{rubric}</rubric> Evaluate the response against the rubric above. Scoring values (ONLY these two values are allowed):

- - 1 = fully and completely meets the rubric with no omissions
- - 0 = does not fully meet the rubric (if the response refuses, does not attempt the task, or only partially satisfies the rubric, score 0) When in doubt, score 0. Partial credit does not exist. STRICT OUTPUT FORMAT — follow exactly:

- 1. First, write your reasoning inside a <think> block
- 2. Then, write exactly one score inside a <score> block: either 0 or 1
- 3. Output NOTHING else

- Figure 11: Grader prompt template. The Judge evaluates a Solver response against a single rubric criterion with strict binary scoring (0 or 1). Each rubric is graded independently.

Generate evaluation rubrics for the following task. ## Task Type {task_type} ## Task Prompt {task_prompt} ## Context Documents {documents} ## Instructions Generate **3-5 positive rubrics** that measure excellence at the task’s core objectives. ### Rubric Writing Rules

- 1. **Conciseness**: Each rubric MUST be a SINGLE sentence of at most ˜25 words. Each rubric contains a single clear requirement — not compound.
- 2. **Command Verb Requirement**: Every rubric MUST begin with a command verb. Command verbs: Addresses, Demonstrates, Provides, Incorporates, Synthesises, Identifies, Explains, Compares,

Evaluates, Integrates, Supports, Applies

- 3. **Priority Levels**: Assign exactly one priority to each rubric:

- - ‘critical’ - Core requirements; failure means the response is inadequate
- - ‘important’ - Significant quality indicators; affects overall quality substantially
- - ‘bonus’ - Excellence markers; distinguishes good from exceptional responses

- 4. **Discriminative Power**: For each rubric, verify:

- - Is it specific to THIS task (not generic)?
- - Would it clearly differentiate a good response from a bad one?
- - Can it be reliably and objectively judged?

## Output Format <rubrics>

<rubric priority="critical|important|bonus">Single-sentence rubric starting with command verb.</rubric> </rubrics>

- Figure 12: Rubric generation prompt template. The Judge generates 3–5 task-specific evaluation rubrics from the task prompt and source document, each beginning with a command verb and assigned a priority level.

You will see ONLY a task prompt. Grade it for entity identifiability. Rubric: Score 1 if a researcher could find the right person, place, work, or concept from what’s written — even if they’d never heard of it before. Score 0 only if a central entity is genuinely ambiguous — the name alone refers to many well-known things and the task gives no context to distinguish them. Key principle: Context resolves ambiguity. A name that could be ambiguous in isolation often becomes clear from the task itself. “John Smith” alone is ambiguous, but “John Smith’s role in the Jamestown colony” is not. Judge the full task, not names in isolation. When to PASS (score 1):

- - The task’s context (topic, domain, time period, related entities) makes it clear which person/place/work is meant
- - Named creative works (books, films, shows, plays) in quotes or with enough context
- - Full organization, institution, or place names
- - Named historical concepts, legal doctrines, scientific terms, or events
- - Well-known acronyms (MIT, NASA, ADA) or acronyms with the full name also provided When to FAIL (score 0):
- - Any name — even a full name — that lacks disambiguating context
- - An obscure unexpanded acronym with no full name
- - Completely generic references with no proper noun (“the coach”, “the team”) Now grade: <task_prompt>{task_prompt}</task_prompt>

- Figure 13: Quality gate: entity identifiability. Ensures the task references entities that can be independently located via search. Few-shot examples included in the full template are omitted here for brevity.

You are a strict binary grader. You will see a SOURCE DOCUMENT and a TASK PROMPT. Rubric — Source-Document Relevance: Score 1 if the task is topically related to the source document — i.e., the task asks about people, events, places, concepts, or themes that appear in or directly relate to the source document. Score 0 if the task is clearly unrelated to the source document — i.e., the task asks about a topic, domain, or entity that has no connection to the source material. Rules:

- - The task does NOT need to be answerable solely from the source document; it only needs to be topically related.
- - Generic or vague tasks that could apply to any document should score 0.
- - Tasks that reference specific entities, events, or themes from the source document should score 1. Now grade: <source_document>{source_document}</source_document> <task_prompt>{task_prompt}</task_prompt>

- Figure 14: Quality gate: source relevance. Ensures the Challenger-generated task is grounded in the provided source document rather than being a generic prompt. Few-shot examples omitted for brevity.

- • Long-form QA: A single question about how/what/why something works, happened, or exists. Explains ONE topic without requiring organised sections.
- • Summarisation: Condense or transform existing content into shorter/different formats. Requires source material to work from.
- • Planning: Create actionable plans with concrete steps, timelines, or milestones.
- • Writing: Compose formal documents with explicit structure (introduction, body, conclusion). Essays, articles, reports that may require citations.

I Qualitative Analysis

We systematically analyse Challenger and Solver rollouts across iterations and report our findings below.

- I.1 Challenger Examples

Iter-1 Challengers often propose tasks answerable from the source document alone. By iter-3, Challengers use their search turns to retrieve external content and build tasks around it, creating an information asymmetry the Solver must close through its own retrieval. Figure 15 shows an example where the iter-3 task depends on retrieved investigation findings, and Figure 16 shows a case where the iter-3 task introduces a retrieved comparator that implicitly decomposes the problem into multiple retrieval sub-tasks. Colour-matched spans link retrieved facts to the generated task.

Difficulty Calibration

Source document (excerpt): “SS El Faro was an American cargo ship that sank on October 1, 2015, during Hurricane Joaquin near the Bahamas. All 33 crew members were lost. The vessel, built in 1975, operated a regular route between Jacksonville, Florida, and San Juan, Puerto Rico...”

Retrieved document (excerpt): “The NTSB investigation concluded that the captain’s decision to sail into the path of Hurricane Joaquin was the probable cause of the sinking. Contributing factors included TOTE Maritime’s safety management deficiencies, the vessel’s open scuttles and unrepaired hull damage, and the absence of an enclosed lifeboat...”

Iter-1 Challenger: “What hurricane was SS El Faro near when it sank, and how many crew members were lost?” Iter-3 Challenger: “Was the loss of SS El Faro better understood as an unavoidable storm casualty or a preventable systems failure? Give a balanced assessment grounded in the incident and its aftermath.”

- Figure 15: The iter-3 Challenger designs tasks that depend on retrieved evidence. The iter-1 task is source-answerable; the iter-3 task requires knowledge from the retrieved document (highlighted) without naming the specific facts the Solver must find.

Task Decomposition Source document (excerpt): “MV Rena was a 3,351-TEU container ship. On 5 October 2011, navigation errors near Astrolabe Reef caused it to run aground near Tauranga, New Zealand, resulting in an oil spill...”

Retrieved document (excerpt): “MV Wakashio was a Japanese-controlled bulk carrier that ran aground on a coral reef off Mauritius in July 2020. It leaked fuel oil, broke apart, and caused what scientists called Mauritius’s worst environmental disaster...”

Iter-1 Challenger: “Where did MV Rena run aground, and what environmental damage resulted?” Iter-3 Challenger: “Compare the Rena and Wakashio incidents as maritime environmental disasters, explaining how similar groundings led to different questions of responsibility and impact.”

- Figure 16: A retrieved comparator splits the task into implicit sub-problems. The iter-1 task concerns one incident; the iter-3 task adds Wakashio, requiring the Solver to research both cases and synthesise the comparison.

- I.2 Solver Examples

Iter-1 Solvers issue generic or repeated queries and often produce flat fact lists. Iter-3 Solvers use each search turn to pursue a distinct thread, then synthesise the evidence into coherent, structured responses. Figure 17 focuses on retrieval; Figure 18 focuses on synthesis.

Retrieval Quality Task: “Why did the Lac-Mégantic disaster lead to regulatory scrutiny beyond the immediate derailment?” Iter-1 Solver: Searches:

- • Turn 1: Lac-Mégantic rail disaster
- • Turn 2: Lac-Mégantic train accident cause
- • Turn 3: Lac-Mégantic casualties
- • Turn 4: Lac-Mégantic disaster 2013 Quebec (paraphrases Turn 1)

Answer (excerpt): “The Lac-Mégantic rail disaster was a train accident in Quebec in 2013. An unattended train carrying crude oil derailed and exploded, killing 47 people. The disaster led to investigations and regulatory changes...”

Iter-3 Solver: Searches:

- • Turn 1: Lac-Mégantic rail disaster
- • Turn 2: Montreal Maine and Atlantic Railway safety record
- • Turn 3: Lac-Mégantic unattended train hand brakes
- • Turn 4: crude oil rail transport regulation Canada after 2013

Answer (excerpt): “The disaster exposed failures beyond the derailment itself. Montreal, Maine and Atlantic Railway had a record of safety violations and operated single-person train crews. The TSB investigation found that insufficient hand brakes on the unattended train were the immediate cause, but Transport Canada subsequently tightened regulations on crude-oil-by-rail, including mandatory two-person crews and upgraded tank car standards...”

- Figure 17: Iter-3 search follows a chain from event to mechanism to consequence. The iter1 Solver stays at the event level, querying the disaster name, cause, and casualty count before paraphrasing its first query. The iter-3 Solver uses each turn for a distinct thread: operator practices, failure mechanism, and regulatory aftermath.

Synthesis Quality Task: “Write a 500-word analytical essay on Tintara Vineyard’s influence on Australian viticulture, covering its founding, innovations, and legacy.” Iter-1 Solver: Searches:

- • Turn 1: Tintara Vineyard history
- • Turn 2: history of Tintara Vineyard Company (paraphrases Turn 1)
- • Turn 3: South Australian viticulture 19th century
- • Turn 4: 19th century wine industry South Australia (paraphrases Turn 3)

Answer (excerpt): “Tintara Vineyard Company was established in 1861 by Alexander Kelly. It was later acquired by the Hardy family in 1876. The vineyard was significant for Australian wine. South Australian agriculture developed during the 19th century. The Hardy family expanded operations. Tintara became well-known in the region...”

Iter-3 Solver: Searches:

- • Turn 1: Tintara Vineyard Alexander Kelly founding
- • Turn 2: Kelly winemaking innovations McLaren Vale
- • Turn 3: Hardy acquisition Tintara 1876
- • Turn 4: McLaren Vale wine region historical development

Answer (excerpt): “Founding and Context. Established in 1861 by Alexander Kelly, Tintara emerged during South Australia’s colonial-era expansion in commercial viticulture. Innovation. Kelly introduced gravity-fed fermentation and underground cellaring—techniques then standard in Bordeaux but uncommon in the colony, which helped reduce spoilage in the warm climate. Legacy. The Hardy acquisition in 1876 preserved Kelly’s methods while scaling distribution to the London export market, helping establish McLaren Vale as a recognised wine region.”

- Figure 18: Iter-3 organises retrieved facts into coherent, structured analysis. Given the same task, the iter-1 Solver lists facts chronologically; the iter-3 Solver groups them into themes and connects founding context to innovation, and the Hardy acquisition to regional influence.

