### M-A-P

## OProver: A Unified Framework for Agentic Formal Theorem Proving

###### M-A-P

# arXiv:2605.17283v1[cs.CL]17May2026

multimodal-art-projection/OProver huggingface.co/m-a-p/oprover

#### Abstract

Recent progress in formal theorem proving has benefited from large-scale proof generation and verifier-aware training, but agentic proving is rarely integrated into prover training, appearing only at inference time. We present OProver, a unified framework for agentic formal theorem proving in Lean 4, in which failed proof attempts are iteratively revised using retrieved compilerverified proofs and Lean compiler feedback. OProver is trained through continued pretraining followed by iterative post-training: each iteration runs agentic proving, indexes newly verified proofs into OProofs and the retrieval memory, uses repair trajectories as SFT data, and uses unresolved hard cases for RL. OProofs is built from public Lean resources, large-scale proof synthesis, and agentic proving traces, containing 1.77M Lean statements, 6.86M compilerverified proofs, and serialized trajectories with retrieved context, failed attempts, feedback, and repairs. Across five benchmarks, OProver-32B attains the best Pass@32 on MiniF2F (93.3%), ProverBench (58.2%), and PutnamBench (11.3%), and ranks second on MathOlympiad (22.8%) and ProofNet (33.2%)—more top placements than any prior open-weight whole-proof prover.

DeepSeek-Prover-V2-7B DeepSeek-Prover-V2-671B

Kimina-Prover-8B Kimina-Prover-72B

Goedel-Prover-V2-8B Goedel-Prover-V2-32B Goedel-Prover-V2-32B (Self-Correction)

LongCat-Flash-Prover (Whole-Proof) LongCat-Flash-Prover (Whole-Proof w/ TIR)

OProver-8B OProver-32B

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

| |
|---|

###### MiniF2F-Test

###### PutnamBench

MathOlympiadBench

95

30

12

93.3%

11.3%

27.5%

91.8%

10.4%

90.4%

90.2%

10

25

90

9.0%

88.1%

22.8%

8.6%

21.7%

8

20.3%

###### Performance(%)

20

84.6%

6.7%

84.4%

85

84.0%

16.9%

82.4%

6

16.7%

4.9%

4.8%

15

13.9%

80

3.9% 3.8%

13.1%

4

78.3%

3.3%

11.1%

11.1%

75.6%

10

2

1.4%

8.6%

75

0

Figure 1. Pass@32 performance on MiniF2F-Test, PutnamBench, and MathOlympiadBench. OProver consistently outperforms prior open-source whole-proof provers across all three benchmarks; the only exception is LongCat-Flash-Prover with TIR on MathOlympiadBench.

##### 1. Introduction

Formal theorem proving in systems such as Lean 4 [de Moura and Ullrich, 2021] provides a rigorous setting for machine reasoning, where every step of a proof is mechanically verified by a small, trusted kernel. This makes them a natural foundation for reliable mathematical reasoning and verified software, but also imposes a high bar: a proof is accepted only if it is fully formal and type-correct. Recent prover systems built on Lean have substantially improved performance on challenging benchmarks such as MiniF2F [Zheng et al., 2021] and PutnamBench [Tsoukalas

- et al., 2024], yet absolute success rates on harder benchmarks remain low, and most systems still rely primarily on single-pass or best-of-𝑁 whole-proof generation. Retrieval and compiler feedback, when used at all, are typically applied as test-time heuristics rather than as a proving policy that the prover is trained to use, leaving an important source of supervision largely unexploited.

A small but growing line of work has begun to incorporate retrieval, compiler feedback, and iterative repair into formal theorem proving [Yang et al., 2023, Jiang et al., 2022, Wu et al., 2025]. However, these capabilities are typically introduced as inference-time augmentations on top of a fixed prover, rather than as a learned policy that the model is trained to use. As a result, a prover trained mainly on finalized proofs sees compiler feedback and retrieved evidence only at deployment, in distributions it was never optimized for. Closing this train–inference mismatch requires training the prover to perform retrieval-grounded, feedback-conditioned refinement as part of its policy, not as a separate inference-time procedure.

Training such a policy in turn requires data that existing formal corpora do not provide. Public formal theorem proving corpora and proof-synthesis datasets [Wu et al., 2022, Ying et al., 2024, Peng et al., 2025] focus on the end products of proving: large collections of formal statements and final compiler-verified proofs. In general, they omit failed attempts, retrieved context, and compiler diagnostics that drive proof repair in practice. Autoformalization and proof synthesis have greatly expanded the number of available statements and verified proofs, but they do not by themselves record the multi-round interaction histories needed to learn agentic self-correction. What is missing is therefore not more verified proofs, but corpora that explicitly preserve how proofs are constructed, fail, and get repaired.

In this paper, we present OProver, a unified framework for agentic formal theorem proving in Lean 4. Rather than treating retrieval, compiler feedback, and iterative repair as separate inference-time modules built on top of a fixed prover, OProver unifies them with training and data construction into a single proving framework. At inference time, OProver treats proving as a multi-round refinement loop, in which each failed proof attempt is revised using retrieved compiler-verified proofs and Lean 4 compiler feedback. At training time, the same retrieval and feedback signals shape the prover’s policy: training proceeds in two phases, continued pretraining on Lean code and mathematics, followed by iterative post-training in which agentic proving, supervised fine-tuning, and reinforcement learning alternate. Newly verified proofs and repair traces produced by the current prover are recirculated into OProofs and the retrieval memory, so that the data, the training procedure, and the proving policy co-evolve within a single framework.

We pair OProver with OProofs, a large-scale corpus for agentic formal theorem proving that supports both pretraining and the co-evolution loop above. Unlike prior Lean datasets, which mostly preserve only final compiler-verified proofs, OProofs additionally records the trajectories of proof construction, failure, feedback, and repair. It contains 1.77M Lean statements and 6.86M compiler-verified proofs, paired with serialized proving trajectories that capture

retrieved context, failed attempts, compiler feedback, and subsequent repairs. We build OProofs from three complementary sources: public Lean resources, large-scale proof synthesis with compiler verification, and traces from OProver’s own agentic proving. The first two sources provide an initial corpus that supports both pretraining and post-training, while the third grows continuously as OProver improves: newly verified proofs are indexed into the retrieval memory and repair trajectories are added to subsequent training rounds.

In summary, our contributions are:

- 1. A unified framework for agentic formal theorem proving. We propose OProver, which treats proving as a retrieval-grounded, feedback-conditioned refinement loop and trains the prover end-to-end to use these signals as part of its policy, rather than as test-time heuristics.
- 2. A large-scale corpus for agentic formal theorem proving. We construct OProofs, containing

- 1.77M Lean statements, 6.86M compiler-verified proofs, and serialized proving trajectories that record retrieved context, failed attempts, compiler feedback, and subsequent repairssupervision that prior Lean corpora do not provide.

- 3. A co-evolution pipeline between prover and corpus. We develop an iterative post-training pipeline in which the current prover produces new verified proofs and repair trajectories: verified proofs are indexed into the retrieval memory, repair trajectories become SFT data, and the hardest unresolved cases provide RL signal for the next round.
- 4. State-of-the-art performance among open-weight whole-proof provers. OProver-32B attains three best and two second-best results across five formal theorem-proving benchmarks, reaching Pass@32 of 93.3 on MiniF2F, 58.2 on ProverBench, and 11.3 on PutnamBench.

- 2. Methodology

Figure 2 illustrates the three components of our framework. The OProofs Construction pipeline (top-left) builds a Lean-specific corpus from public Lean resources and raw informal sources, through deduplication, filtering, autoformalization, agentic proving, and Lean 4 verification (§2.2). The OProver Agentic Proving pipeline (top-right) performs theorem proving as a multiround interaction: given a target theorem, the prover policy queries a retrieval memory for top-𝑘 compiler-verified proofs, produces a proof attempt, and is verified by the Lean 4 compiler; on failure, compiler feedback is returned to the policy and the proof is revised in the next round, while successful proofs are added back into the retrieval memory (§2.1). The OProver Agentic Training pipeline (bottom) trains OProver in two stages: a one-time continued pretraining on OProofs yields a domain-adapted base model OProver-Base, followed by an iterative posttraining loop in which the current prover performs agentic rollouts and is updated by SFT on round-level repair examples and RL on harder unresolved cases; verified proofs and repair trajectories from each iteration are folded back into OProofs and the retrieval memory, so that the corpus and the prover co-evolve across post-training iterations (§2.3).

###### 2.1. The OProver Framework

OProver consists of three interacting components: a proving policy 𝜋, a retrieval memory M of compiler-verified proofs, and the Lean 4 compiler V as the verification environment. The policy is trained to use the other two components through the agentic proving formulation below.

###### OProofs Construction OProver Agentic Proving

query

Public Lean Resources

Raw Informal Sources

Retrieval Memory

Target theorem

top-k

deduplicate

filter

Prover policy

Retrieved proof

informal math statements

Lean statements

revisenextround

Proof attempt

autoformalize

verify

agentic prove

add to memory

formal lean statements

Lean 4 compiler

agentic prove

failure success

Compiler feedback

Verified proof

OProofs

###### OProver Agentic Training

Update Oproofs

Agentic Proving SFT+RL

CPT

Updated OProver

Current OProver

rollouts

OProofs OProver-base

Next iteration

- Figure 2. Overview of OProver. The framework has three components: OProofs Construction (top-left), which builds a Lean-specific corpus from public Lean resources and autoformalized statements; OProver Agentic Proving (top-right), which performs multi-round refinement under retrieval and Lean 4 compiler feedback; and OProver Agentic Training (bottom), where a one-time CPT yields OProver-Base, followed by an iterative post-training loop in which agentic

proving, SFT, and RL produce OProver𝑡+1 from OProver𝑡, while verified proofs are folded back into OProofs.

Agentic proving formulation. We formulate theorem proving as a bounded multi-round refinement process. At round 𝑡, the policy conditions on a state

𝑋𝑡 = (𝑠, R𝑡, 𝑝𝑡−1, 𝑓𝑡−1),

where 𝑠 is the target theorem statement, R𝑡 is the retrieved proof context, 𝑝𝑡−1 is the previous proof attempt, and 𝑓𝑡−1 is the corresponding compiler feedback. The policy then produces a revised proof attempt

𝑝𝑡 ∼ 𝜋(· | 𝑋𝑡).

At the initial round, 𝑝0 and 𝑓0 are empty. The interaction terminates when V verifies a proof attempt or when a predefined round budget 𝑇 is exhausted.

Compact interaction state. A key modeling choice in OProver is that the policy conditions only on the most recent proof attempt and its compiler feedback, rather than on the full interaction history (𝑝0, 𝑓0, . . . , 𝑝𝑡−1, 𝑓𝑡−1). This keeps the interaction state compact and preserves the local correction signal most relevant for proof repair. Crucially, the same state formulation is used throughout rollout collection, supervised fine-tuning, and reinforcement learning, so that the training-time interface exactly matches the proving-time interaction (§2.3).

Retrieved compiler-verified proofs. At each round, OProver retrieves top-𝑘 compiler-verified proofs from the retrieval memory M by semantic similarity to the target statement 𝑠, using a sentence-embedding model trained on Lean theorem–proof pairs (details in §2.2). M is constructed from OProofs and is continuously expanded with newly verified proofs during iterative post-training, so the retrieval context becomes richer as training progresses. The retrieved proofs provide reusable lemma usage patterns, tactic structures, and proof strategies from related formal contexts, and are inserted into the policy input alongside 𝑠, 𝑝𝑡−1, and 𝑓𝑡−1.

Compiler feedback as a correction signal. When a proof attempt 𝑝𝑡−1 fails Lean verification, V returns textual diagnostics describing the failure—syntax errors, type mismatches, unknown identifiers, tactic failures, or unsolved goals. We pass these diagnostics directly to the policy as 𝑓𝑡−1, without projecting them into a hand-designed error taxonomy. Preserving the raw textual form lets the policy condition on fine-grained information that would be lost under categorical encoding, and is essential for the targeted revisions that distinguish multi-round refinement from repeated single-shot regeneration.

###### 2.2. OProofs: A Unified Training Corpus

OProofs is a Lean-specific corpus that serves two roles in the OProver framework: it provides training data across CPT, SFT, and RL, and it supplies the compiler-verified proofs that populate the retrieval memory M used at proving time (§2.1). In total, OProofs contains ∼1.5M Lean statements and ∼5.0M compiler-verified proofs, together with serialized proving trajectories that record retrieved context, failed attempts, compiler feedback, and subsequent repairs.

Construction pipeline. Figure 2 (top-left) shows the two-branch construction pipeline, both ending in Lean 4 verification.

- Branch 1: public Lean resources. We collect open-source Lean statements from {NuminaMathLEAN, Lean-Workbook, Leanabell-FormalStmt, Goedel-Pset, ...} and deduplicate at the statement level by exact and near-duplicate matching of theorem signatures. We then run agentic proving with open-source provers on each unique statement and retain only Lean-verified proofs.
- Branch 2: raw informal sources. We mine informal mathematical statements from Common Crawl and GitHub code repositories using a FastText classifier trained on math-tagged web pages. The filtered statements are autoformalized into Lean 4 statements by Criticlean Peng et al. [2025], and the same agentic proving process as in Branch 1 is used to obtain Lean-verified proofs.

Round-level repair data. Although OProver interacts with Lean over multiple rounds, training does not consume the full interaction history as a single long sequence. Instead, each rollout is decomposed into round-level repair examples of the form (𝑠, R𝑡, 𝑝𝑡−1, 𝑓𝑡−1) → 𝑝𝑡, matching the state formulation in §2.1 exactly. This provides process-level supervision across multi-round rollouts while keeping the training interface identical to the proving-time interaction. The round-level repair set is used by SFT in §2.3.

An evolving corpus. OProofs is not a static dataset. As OProver improves, newly verified proofs and repair trajectories are continuously added back, expanding both the training pool

Algorithm 1 Iterative co-evolution of OProver and OProofs Require: CPT-initialized policy 𝜋0 (OProver-Base); initial corpus D0; theorem pool Q; Lean

verifier V; number of co-evolution iterations 𝐾

- 1: M0 ← Index(D0)
- 2: for 𝑘 = 0,1, . . . , 𝐾 − 1 do
- 3: Run multi-round agentic proving rollouts of 𝜋𝑘 on Q, retrieving from M𝑘 and verifying with V
- 4: Collect newly verified theorem–proof pairs P𝑘+
- 5: Extract round-level repair examples B𝑘 (each of form (𝑠, R𝑡, 𝑝𝑡−1, 𝑓𝑡−1) → 𝑝𝑡, where 𝑡 is the within-rollout round index)
- 6: Select hard cases H𝑘 as groups with non-trivial success rate
- 7: D𝑘+1 ← D𝑘 ∪ P𝑘+; M𝑘+1 ← Index(D𝑘+1)
- 8: 𝜋𝑘+1 ← SFT(𝜋𝑘; B𝑘) followed by GSPO( · ; H𝑘, M𝑘+1)
- 9: end for
- 10: return 𝜋𝐾, D𝐾

and the retrieval memory M (details in §2.3).

###### 2.3. Training OProver with OProofs

Training proceeds in two phases: a one-time continued pretraining on OProofs produces a domain-adapted base model OProver-Base, after which iterative post-training alternates agentic proving rollouts with SFT and RL, expanding OProofs as the prover improves (Algorithm 1).

Continued pretraining. We perform a one-time continued pretraining on a 65B-token mixture designed to strengthen formal reasoning, code, and long-context mathematical reasoning. The mixture contains approximately 30% Lean-related formal data drawn from OProofs (§2.2), 20% code data from OpenCoder [Huang et al., 2025], 40% mathematical data from Nemotron-Math4-Plus [Mahabadi et al., 2025], and 10% long-chain-of-thought data from ProLong-64K [Gao

- et al., 2025]. After CPT, we obtain OProver-Base, which serves as the starting point of iterative post-training.

Supervised fine-tuning through agentic proving. Starting from OProver-Base, we perform agentic proving on theorems that the current model does not yet solve reliably. Each rollout consumes retrieved proof context, prior proof attempts, and compiler feedback, exactly matching the proving-time interaction defined in §2.1. From these rollouts, we extract round-level repair examples (𝑠, R𝑡, 𝑝𝑡−1, 𝑓𝑡−1) → 𝑝𝑡 and use them as SFT data, computing cross-entropy loss only on the target proof attempt 𝑝𝑡. Newly verified proofs are simultaneously added to OProofs and indexed into the retrieval memory M for subsequent rounds.

Reinforcement learning. After SFT, we further improve the policy with Group Sequence Policy Optimization (GSPO) [Zheng et al., 2025], with group-relative advantage normalization. For each theorem, we sample 𝑛 multi-round agentic proving rollouts (each containing up to 𝑅 refinement rounds), and assign a per-round reward

𝑟𝑡 =

0.8 + 0.2 · 1[format correct] if V(𝑝𝑡) = verified, 0.0 otherwise.

Metric Value Unique Lean statements 1.77M Compiler-verified proofs 6.86M

Geometry 6.8%

Other 6.9%

Number Theory 13.0%

with retrieval context 4.33M with compiler feedback 869K

Algebra 59.5%

Analysis 13.8%

Agentic proving trajectories 1.07M

with multi-round repair 164K Round-level repair examples 280K Lean corpus tokens 51.8B

(a) Corpus overview.

(b) Mathematical domain.

27.1% (471K)

Elementary

48.9% (848K)

High School

19.2% (333K)

Undergraduate

4.8% (84K)

Graduate+

0% 20% 40% 60%

Share of unique statements

(c) Difficulty level.

- Figure 3. OProofs Lean 4 corpus statistics. (a) Headline counts of statements, proofs, retrieval/feedback signals, agentic trajectories, and training tokens. (b) Mathematical domain distribution. (c) Difficulty level distribution.

That is, format quality is rewarded only when the round is verified by Lean. Although rewards are assigned at the round level, all 𝑛 × 𝑅 rounds for the same theorem are pooled into a single group, and advantages are computed by group-relative normalization across this pooled set. This pooled normalization allows the policy to learn from contrasts both across independent attempts and across successive refinement rounds within a single rollout. The state 𝑋𝑡 and retrieval memory used during RL are identical to those at proving time (§2.1), so the policy is optimized end-to-end on the same interface it is deployed under. Specific values of 𝑛, 𝑅, and other hyperparameters are reported in §A.4.

Co-evolution between OProver and OProofs. Each post-training iteration follows the loop in Algorithm 1. Let D𝑘, M𝑘, B𝑘, and H𝑘 denote OProofs, the retrieval memory, the SFT repair set, and the RL hard-case set at iteration 𝑘. The current policy 𝜋𝑘 runs agentic proving on a theorem pool Q, contributing newly verified proofs P𝑘+ to D𝑘+1, round-level repair examples to B𝑘, and groups with non-trivial success rate to H𝑘. Indexing D𝑘+1 yields the next-iteration retrieval memory M𝑘+1. As D𝑘 grows across iterations, both retrieval context and supervision signal become richer, jointly driving 𝜋𝑘 toward stronger proving capability.

##### 3. Data Statistics

###### 3.1. Corpus Overview

OProofs is a large-scale Lean 4 corpus constructed from both public formal resources and modelgenerated data. Public resources contribute formal statements and theorem–proof pairs, while our proof synthesis and agentic proving pipelines further expand the corpus with additional compiler-verified proofs and serialized proving trajectories. These trajectories are further decomposed into round-level repair examples for training. As a result, OProofs supports not only compiler-verified proof supervision, but also retrieval-grounded and feedback-conditioned agentic theorem proving.

Figure 3(a) summarizes the current Lean 4 component of OProofs. The present version contains approximately 1.77M unique Lean statements paired with 6.86M compiler-verified proofs, together with 1.07M agentic proving trajectories. Beyond raw verified proofs, OProofs explicitly preserves the supervision signals required for retrieval-grounded and feedback-

- Table 1. Comparison of OProofs with representative open Lean resources.

Formal stmt.

Verified proof

NL rationale / CoT

Compiler feedback

Retrieved proof

Multi-round repair

Resource

NuminaMath-LEAN ✓ ✗ ✗ ✗ ✗ ✗ Leanabell-SFT ✓ ✓ ✓ ✗ ✗ ✗ Lean-Workbook ✓ ✗ ✗ ✗ ✗ ✗ Leanabell-FormalStmt ✓ ✗ ✗ ✗ ✗ ✗ Workbook-proofs ✓ ✓ ✗ ✗ ✗ ✗ Goedel-Pset ✓ ✗ ✗ ✗ ✗ ✗ FineLeanCorpus ✓ ✗ ✗ ✗ ✗ ✗ FormalMATH-All ✓ ✗ ✗ ✓ ✗ ✗ Nemotron-Math-Proofs ✓ ✓ ✓ ✓ ✗ ✗

OProofs ✓ ✓ ✓ ✓ ✓ ✓

conditioned proof revision: 4.33M proofs carry retrieved proof context, 869k carry non-trivial compiler feedback from prior failed attempts, and 280k round-level repair examples are derived from the 164k trajectories that required at least one repair step. The frozen Lean corpus snapshot used in our current training pipeline comprises 51.8B tokens (prompt and chain-of-thought combined, tokenized with the Qwen3 tokenizer).

The corpus spans a broad spectrum of mathematics (Figure 3(b)): Algebra (59.5%), Analysis (13.8%), Number Theory (13.0%), and Geometry (6.8%) dominate, with the remaining 6.9% grouped as “Other” (covering Combinatorics, Logic, Probability, Topology, and Computation). Difficulty levels (Figure 3(c)) range from Elementary arithmetic (27.1%) through High School (48.9%) and Undergraduate (19.2%) to Graduate-level problems (4.8%); domain and difficulty are inferred per statement by an LLM classifier (see Appendix C.5 for the prompt and validation protocol). During iterative training, the underlying OProofs corpus continues to expand through newly verified proofs and repair data collected from the evolving prover. We emphasize that the statistics reported in Figure 3 cover only the Lean 4 theorem-proving component of OProofs, which is also the component we primarily analyze and release in this work. Auxiliary code, mathematical reasoning, and long-context reasoning corpora used during continued pretraining are excluded from these counts.

Table 1 compares OProofs with representative open Lean resources along supervision dimensions relevant to agentic formal theorem proving, including formal statements, compiler-verified proofs, natural-language rationales, compiler feedback, retrieved proof context, and multi-round repair signals. Among existing public resources, Nemotron-Math-Proofs is one of the closest to OProofs, as it already provides Lean 4 formal statements, compiler-verified proofs, naturallanguage reasoning traces, and compiler feedback. However, it does not explicitly release retrieved proof context or multi-round repair supervision. More broadly, most public resources expose only a subset of the signals needed for agentic formal theorem proving. OProofs is designed to close this gap by jointly providing retrieved proof context, compiler feedback, and repair supervision within a unified corpus for retrieval-grounded, feedback-conditioned proof revision.

##### 4. Experiments

Benchmarks. We evaluate OProver on five Lean 4 theorem-proving benchmarks of varying difficulty and mathematical scope: MiniF2F [Zheng et al., 2021] (244 high-school olympiad problems), MathOlympiadBench [Lin et al., 2025b] (360 recent competition problems), ProofNet [Azer-

bayev et al., 2023] (186 undergraduate-level theorems from textbook formalizations), ProverBench [Ren et al., 2025] (325 problems spanning olympiad and undergraduate mathematics), and PutnamBench [Tsoukalas et al., 2024] (672 Putnam competition problems, the hardest in the suite). This is also the standard evaluation suite used by recent open-source provers including Goedel-Prover-V2 [Lin et al., 2025b] and LongCat-Flash-Prover [Wang et al., 2026], allowing direct comparison.

Baselines. We compare OProver against open-weight reasoning models (DeepSeek-V3.2 [Liu et al., 2025] and Kimi-K2.5 [Team et al., 2026]) and open-weight whole-proof formal theorem provers, including Kimina-Prover [Wang et al., 2025], DeepSeek-Prover-V2 [Ren et al., 2025], Leanabell-Prover-V2 [Ji et al., 2025], Goedel-Prover-V2 [Lin et al., 2025b], and LongCat-FlashProver [Wang et al., 2026]. Our primary comparisons focus on Goedel-Prover-V2 (closest model scale at 32B) and LongCat-Flash-Prover (state-of-the-art open-weight whole-proof prover at 560B MoE / 27B active). For baseline results not re-evaluated by us, we mark the numbers with †.

Evaluation protocol. We report Pass@𝑘 as the primary metric, computed using the standard unbiased estimator

𝑛−𝑚 𝑘 𝑛 𝑘

Pass@𝑘 = 1 −

,

where 𝑛 is the number of independent samples per statement and 𝑚 is the number of successful samples; specific values of 𝑛 and 𝑘 are reported in each table or caption. Unless otherwise noted, we report Pass@32 with 𝑛 = 64. OProver is evaluated under the agentic proving setup defined in §2.1, where one sample is a complete multi-round rollout (up to 𝑅 refinement rounds), and a sample is counted as successful if any of its attempts is verified by Lean 4. Baseline models are evaluated under their own protocols (whole-proof sampling, best-of-N with verifier filtering, or whole-proof with TIR) to match each method’s intended deployment. Section 4.2 provides a budget-controlled comparison that accounts for OProver’s multi-round compute.

###### 4.1. Main Results

- Table 2 compares OProver with representative open-weight reasoning models and open-weight prover models on five theorem-proving benchmarks. OProver-32B attains the best Pass@32 on three benchmarks (MiniF2F 93.3, ProverBench 58.2, PutnamBench 11.3) and the second-best on the remaining two (MathOlympiad 22.8, ProofNet 33.2), behind only LongCat-Flash-Prover with whole-proof TIR. This gives OProver-32B more top placements (three best, two second-best) than any other model in the table. Even OProver-8B outperforms Goedel-Prover-V2-32B on all five benchmarks despite having four times fewer parameters.

These results are notable because OProver-32B is a 32B dense model, whereas its closest competitor, LongCat-Flash-Prover, is a 560B Mixture-of-Experts model with roughly 27B active parameters per token—over 17× more total parameters. OProver-32B also outperforms the 671B DeepSeek-Prover-V2 on all five benchmarks, and improves over Goedel-Prover-V2-32B by 6.3 to 11.2 points per benchmark. Taken together, these comparisons indicate that the gains are not a consequence of model scale, but are driven by the combination of retrieval, multi-turn compiler feedback, and iterative test-time refinement.

The gains span benchmarks of varying difficulty: from competition-style problems on MiniF2F, through textbook-level formalizations on ProverBench, to the substantially harder

- Table 2. Theorem-proving performance (Pass@32, %) of open-weight reasoning models and prover models across multiple benchmarks. Best results are in bold and second-best results are underlined. † marks scores not re-evaluated by us under our protocol.

MathOlympiad (Pass@32)

PutnamBench (Pass@32) Open-Weights Reasoning Models

MiniF2F (Pass@32)

ProofNet (Pass@32)

ProverBench (Pass@32)

Model #Params

DeepSeek-V3.2 671B 14.7† 77.9† 20.4† 42.8† 5.8† Kimi-K2.5 1T 7.5† 76.6† 19.9† 44.3† 1.2†

Open-Weights Prover Models Kimina-Prover-8B 8B 7.6 77.2 13.0 40.5 2.3 Kimina-Prover-72B 72B 11.5 83.4 16.3 45.0 4.0 DeepSeek-Prover-V2-7B 7B 7.9 73.9 18.1 45.7 1.5 DeepSeek-Prover-V2-671B 671B 13.9† 82.4† 30.5† 52.9† 3.3† Leanabell-Prover-V2-KM 8B – 68.4† 13.4† 39.8† – Leanabell-Prover-V2-DS 7B – 76.6† 23.7† 47.8† – Goedel-Prover-V2-8B 8B 9.9 79.4 17.3 48.6 0.7

w/ self-correction 8B – 86.7† – – – Goedel-Prover-V2-32B 32B 16.0 85.8 22.0 51.0 5.0

w/ self-correction 32B 20.3† 90.4† – – 8.6† LongCat-Flash-Prover 560B 16.9† 84.4† 19.9† 49.9† 4.9†

whole-proof w/ TIR 560B 27.5† 90.2† 36.1† 57.9† 10.4†

Ours OProver-8B 8B 21.7 91.8 31.9 56.0 9.0 OProver-32B 32B 22.8 93.3 33.2 58.2 11.3

Putnam-style problems on PutnamBench, where absolute success rates remain below 12% for all methods. Achieving the best results across this entire spectrum suggests that OProver’s improvements reflect a broadly effective proving strategy rather than specialization on any single benchmark family.

###### 4.2. Test-Time Scaling

We next study whether OProver can effectively benefit from additional test-time compute. We define

BestPass(𝐵) = max

Pass(𝑅, 𝑘),

𝑅·𝑘=𝐵

where 𝑅 is the number of refinement rounds, 𝑘 is the number of samples per round, and 𝐵 is the total test-time budget. This metric measures the best success rate achievable under a fixed total budget, regardless of how the budget is allocated between iterative refinement and independent sampling.

Figure 4 shows that both OProver-8B and OProver-32B improve consistently as the total budget increases on all five benchmarks. For example, when increasing the budget from 𝐵 = 8 to 𝐵 = 256, OProver-32B improves from 87.5 to 92.8 on MiniF2F, from 15.5 to 22.0 on MathOlympiad, from 25.6 to 32.8 on ProofNet, from 51.3 to 56.9 on ProverBench, and from 6.4 to 11.3 on PutnamBench. Similar gains are observed for OProver-8B. These results indicate that test-time scaling is robust across both model sizes and benchmark types.

At the same time, the gains show clear diminishing returns, with the improvement per budget doubling shrinking steadily as 𝐵 grows. On ProofNet with OProver-32B, for instance, doubling 𝐵 from 8 to 16 yields a 2.1-point gain (25.6 → 27.7), but the same doubling from 128 to 256 adds only 0.9 points (31.9 → 32.8). The same pattern is visible on every benchmark and for

miniF2F

MathOlympiad

ProofNet

ProverBench

PutnamBench

12

22

55

30

90

10

20

25

BestPass(B)(%)

BestPass(B)(%)

BestPass(B)(%)

BestPass(B)(%)

BestPass(B)(%)

50

18

85

8

16

20

45

14

6

80

15

12

40

4

10

75

10

Oprover-8B

Oprover-8B

Oprover-8B

Oprover-8B

Oprover-8B

35

Oprover-32B

Oprover-32B

Oprover-32B

Oprover-32B

Oprover-32B

8

2

5

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

log2 B

log2 B

log2 B

log2 B

log2 B

- Figure 4. Test-time scaling of OProver under a fixed total budget 𝐵. For each benchmark, we report BestPass(𝐵) (defined in §4.2), the best success rate achievable under budget 𝐵 across all (𝑅, 𝑘) allocations. Both OProver-8B and OProver-32B improve consistently with larger budgets, while the gains exhibit clear diminishing returns.

both model sizes. The rate of diminishing returns is also benchmark-dependent: on MiniF2F and ProverBench, where baseline success rates are already high, BestPass approaches saturation more quickly, whereas on PutnamBench the gains remain comparatively steady throughout the budget range, consistent with the much lower baseline leaving more room for additional compute to help.

To further study how test-time compute should be allocated, we fix the total budget 𝐵 and plot Pass(𝑅, 𝐵/𝑅) as a function of the refinement depth 𝑅, which isolates the trade-off between deeper multi-turn interaction and wider independent sampling under the same total compute budget.

Figure 5 reveals that the optimal allocation between refinement depth 𝑅 and sampling width 𝑘 = 𝐵/𝑅 is governed by the per-chain success probability of the benchmark. Increasing 𝑅 lets each chain exploit Lean feedback to repair its own proof, while increasing 𝑘 spreads the budget across more independent attempts; under a fixed budget these two forces compete.

On MiniF2F, MathOlympiad, ProofNet, and ProverBench, single-chain success probabilities are already substantial, so the marginal gain from additional refinement rounds remains positive throughout. Performance improves monotonically with 𝑅, and the best configuration under any medium-to-large budget is consistently attained at 𝑅 = 16 for both model sizes. 1 − (1 − 𝑝)𝑘 describes the probability that at least one of 𝑘 independent chains succeeds. When 𝑝 is small—as on PutnamBench, where per-chain success stays in the 5–11% range—this probability is highly sensitive to 𝑘: halving 𝑘 to double 𝑅 costs more in lost exploration than it gains in deeper refinement. Empirically, the optimum settles at 𝑅 = 8 rather than 𝑅 = 16 for every 𝐵 ≥ 16 on both model sizes; at 𝐵 = 256, OProver-32B reaches 11.3% at 𝑅 = 8 but drops to 10.7% at 𝑅 = 16. The same effect appears on easier benchmarks under small budgets: when 𝐵 = 8, pushing 𝑅 to 16 forces 𝑘 = 1, eliminating all parallel exploration, and the optimum shifts to 𝑅 = 4–8.

In summary, the optimal allocation is benchmark-dependent. On easier-to-moderate benchmarks, 𝑅 = 16 is consistently optimal once 𝐵 ≥ 16. On the hardest benchmark, the optimum saturates at 𝑅 = 8—the point where additional refinement depth no longer compensates for the reduced sampling width.

###### 4.3. Effectiveness of Iterative Post-Training

B=8 B=16 B=32 B=64 B=128 B=256 Oprover-8B Oprover-32B

miniF2F

MathOlympiad

ProofNet

ProverBench

PutnamBench

- 85
- 86
- 87
- 88
- 89
- 90
- 91
- 92
- 93

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

22

- 5
- 6
- 7
- 8
- 9
- 10
- 11

32

56

30

20

Pass(R,B/R)(%)

54

28

26

18

52

24

50

16

22

20

48

14

18

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

0 1 2 3 4

log2 R

log2 R

log2 R

log2 R

log2 R

- Figure 5. Performance under fixed total budget 𝐵 as a function of refinement depth 𝑅. Each curve plots Pass(𝑅, 𝐵/𝑅), isolating the trade-off between deeper interaction and wider sampling. On most benchmarks performance keeps improving as 𝑅 grows up to 𝑅 = 16, while on the hardest benchmark (PutnamBench) the optimum settles around 𝑅 = 8, indicating that very narrow sampling can hurt when single-chain success probability is low.

- Table 3. Ablation on multi-turn compiler feedback and retrieval augmentation. “-FB” removes multi-turn compiler feedback and keeps only the first interaction round (retrieval retained). “-FB, -RAG” additionally removes retrieved proofs from the prompt. All numbers are Pass@32 (%).

Model Variant MathOlympiad MiniF2F ProofNet ProverBench PutnamBench

OProver-8B Full 21.7 91.8 31.9 56.0 9.0 OProver-8B -FB 15.8 87.7 22.4 50.6 5.7 OProver-8B -FB, -RAG 12.4 86.2 22.4 49.9 4.1

OProver-32B Full 22.8 93.3 33.2 58.2 11.3 OProver-32B -FB 16.5 88.4 25.8 52.0 7.0 OProver-32B -FB, -RAG 14.8 87.9 24.7 51.1 5.9

To examine whether the iterative co-evolution loop between OProver and OProofs (§2.3) yields measurable improvements at each step, we evaluate Pass@32 on MiniF2F-Test for every intermediate checkpoint. Figure 6 shows that both model sizes improve monotonically across iterations. OProver-8B rises from 79.5 (Base, after continued pretraining only) to 86.2, 87.0, and 91.8 after Rounds 1, 2, and 3, for a total gain of 12.3 points. OProver-32B rises from 84.7 to 88.1 and 93.3 after Rounds 1 and 2, for a total gain of 8.6 points. These gains confirm that recirculating verified proofs and repair trajectories from the current prover into OProofs (and into the retrieval memory) produces a stronger prover in the subsequent iteration, rather than saturating after the first post-training round.

93.3

94

| |
|---|

92

MiniF2F-TestPass@32(%)

91.8

90

88.1

88

87.0

86

84.7

86.2

| |
|---|

84

82

80

OProver-8B

79.5

78

OProver-32B

Base Round 1 Round 2 Round 3

Post-training Iteration

Figure 6. MiniF2F-Test Pass@32 across post-training iterations. Both OProver-8B and OProver-32B improve monotonically.

- 4.4. Ablation Studies

We ablate two core components of OProver: multi-turn compiler feedback and retrieval augmentation. Specifically, w/o feedback keeps only the first interaction round, i.e., the prover generates 32 candidates in round 1 without subsequent refinement (retrieval retained), while w/o feedback, w/o RAG additionally removes retrieved proofs from the prompt.

Table 3 shows that both components contribute to OProver’s performance, but with markedly different magnitudes. Removing compiler feedback causes the largest and most consistent degradation across benchmarks and model sizes, confirming that iterative refinement is the primary driver of OProver’s gains. For OProver-8B, Pass@32 drops from 21.7 to 15.8 on MathOlympiad, from 91.8 to 87.7 on MiniF2F, from 31.9 to 22.4 on ProofNet, from 56.0 to 50.6 on ProverBench, and from 9.0 to 5.7 on PutnamBench. For OProver-32B, the corresponding drops are from 22.8 to 16.5, from 93.3 to 88.4, from 33.2 to 25.8, from 58.2 to 52.0, and from 11.3 to 7.0.

Further removing RAG leads to a smaller but consistent degradation, indicating that retrieval provides complementary benefits beyond feedback alone. For OProver-32B, the additional drops range from 0.5 points on MiniF2F (88.4 → 87.9) to 1.7 points on MathOlympiad (16.5 → 14.8), with intermediate drops on ProofNet (1.1), ProverBench (0.9), and PutnamBench (1.1). While feedback mainly helps repair partially correct proofs over multiple rounds, retrieval improves proof grounding by exposing relevant premises and lemmas before refinement begins. The combined results therefore suggest that OProver’s gains come from the synergy between retrieval-augmented grounding and compiler-guided iterative refinement, rather than from single-round sampling alone, with feedback contributing the dominant share.

- 5. Related Work

- 5.1. Proof Generation and Search

Neural theorem proving with language models has developed along two closely related directions: proof generation and guided search. Early work such as GPT-f [Polu and Sutskever, 2020] and Proof Artifact Co-Training (PACT) [Han et al., 2021] showed that formal proofs can be modeled as generated sequences and that proof artifacts provide useful supervision for neural provers. In parallel, search-based approaches such as HyperTree Proof Search (HTPS) [Lample

- et al., 2022] demonstrated that neural guidance can be integrated with symbolic exploration to improve proof discovery.

Within Lean theorem proving, a major line of work studies tactic-level interaction and state-space search. LeanDojo and ReProver [Yang et al., 2023] introduced a retrieval-augmented interactive environment and highlighted premise selection as a core challenge for large-library theorem proving. Subsequent systems strengthened this paradigm with learned critics, planners, and feedback-aware search. InternLM2.5-StepProver [Wu et al., 2025] scales expert iteration with critic-guided proving, BFS-Prover [Xin et al., 2025] revisits best-first search with preference optimization from compiler feedback, Bourbaki [Zimmer et al., 2025] formulates theorem proving as a goal-conditioned decision process with structured subgoal search, and DeepSeekProver-V1.5 [Xin et al., 2024] combines proof-assistant feedback with search-time exploration through an MCTS-style procedure.

A second line increasingly treats theorem proving as whole-proof generation. DeepSeekProver-V1.5 [Xin et al., 2024] demonstrated that scaling synthetic theorem–proof data, combined with proof-assistant feedback, substantially improves end-to-end Lean proof generation. More recent systems extend this paradigm with stronger base models, longer-horizon reasoning,

subgoal decomposition, and reinforcement learning, including Goedel-Prover [Lin et al., 2025a], DeepSeek-Prover-V2 [Ren et al., 2025], Kimina-Prover [Wang et al., 2025], and Seed-Prover [Chen et al., 2025]. Together, these results establish whole-proof generation as a competitive paradigm for modern formal theorem proving.

###### 5.2. Verifier-Guided Refinement and Agentic Proving

A growing body of work argues that formal theorem proving should not be treated as a singlepass generation problem, since proof assistants provide dense and semantically meaningful feedback on failed attempts. Baldur [First et al., 2023] is an influential early example in this direction: it combines whole-proof generation with a dedicated repair model, showing that unsuccessful proofs can be revised rather than discarded. In Lean, Lean-STaR [Lin et al., 2024] interleaves informal reasoning with formal proving steps, while DeepSeek-Prover-V1.5 [Xin et al., 2024] incorporates proof-assistant feedback into both reinforcement learning and search.

Recent work has made this interaction pattern increasingly explicit and increasingly agentic. Leanabell-Prover-V2 [Ji et al., 2025] studies verifier-integrated reasoning with reinforcement learning, StepFun-Prover [Shang et al., 2025] trains tool-integrated reasoning models that refine proofs using environment feedback, and Prover Agent [Baba et al., 2025] coordinates informal reasoning, formal proving, and auxiliary lemma generation within an agent-style architecture. Gupta et al. [2025] further show that multi-turn proof repair can be learned directly from interactive trajectories rather than being introduced only at inference time. More recent systems strengthen this trend: Seed-Prover [Chen et al., 2025] combines Lean feedback with proved lemmas and self-summarization to support iterative whole-proof refinement, and LongCatFlash-Prover [Wang et al., 2026] advances agentic tool-integrated reinforcement learning for formal reasoning. Collectively, these works show that verifier feedback is not merely a binary correctness signal, but a rich supervision source for iterative and long-horizon proof refinement.

###### 5.3. Formal Data Synthesis and Process Supervision

Recent progress in formal theorem proving has also depended critically on advances in data construction. Because high-quality formal theorem–proof pairs remain scarce relative to the scale required for modern model training, many recent systems rely on synthetic proof generation, large-scale formalization, and iterative data expansion. Lean Workbook [Ying et al., 2024] contributes a large-scale collection of Lean problems formalized from natural-language mathematics, while TheoremForge [Tao and Xu, 2026] studies how agentic workflows can be used to synthesize formal reasoning data under limited annotation budgets. Large-scale prover pipelines such as DeepSeek-Prover-V2 [Ren et al., 2025], Goedel-Prover [Lin et al., 2025a], and Goedel-Prover-V2 [Lin et al., 2025b] further demonstrate that synthetic data construction is now central to scaling formal reasoning systems.

A related line of work investigates how supervision can be enriched beyond final verified proofs. Draft, Sketch, and Prove [Jiang et al., 2022] shows that informal proof sketches can guide formal proof generation through intermediate reasoning structure, and ProofNet [Azerbayev

- et al., 2023] establishes a benchmark connecting undergraduate-level mathematical statements, informal proofs, and formal verification. More recent efforts further incorporate critic signals and tool feedback into the supervision interface. CriticLean [Peng et al., 2025] introduces criticguided training for semantic fidelity in Lean formalization, Autoformalizer with Tool Feedback [Guo et al., 2025] uses compiler feedback and consistency signals to iteratively improve generated formal statements, and STP [Dong and Ma, 2025] moves toward self-improving theorem proving

through iterative cycles of conjecturing and proving.

Our work is situated at the intersection of these directions. OProofs extends formal proof corpora toward trajectory-level supervision by preserving multi-turn interaction histories, including failed attempts, retrieved proof context, compiler feedback, and subsequent repairs. The OProver post-training loop further couples data construction with policy improvement by adding newly verified proofs to OProofs and indexing them into retrieval memory, while using repair examples and harder unresolved cases for supervised and reinforcement learning. In this sense, our work unifies formal data synthesis and process supervision within a single training framework for agentic formal theorem proving. Compared to prior self-improving systems such as STP [Dong and Ma, 2025] and Seed-Prover [Chen et al., 2025], which iterate on statement–proof pairs, OProver explicitly preserves and learns from serialized multi-turn proving trajectories, enabling retrieval-grounded and feedback-conditioned proof revision as a learned policy rather than an inference-time wrapper.

##### 6. Conclusion

We present OProver, a unified framework for agentic formal theorem proving in Lean 4. OProver treats proving as a multi-round refinement loop in which a trained policy revises failed proof attempts using retrieved compiler-verified proofs and Lean 4 compiler feedback, unifying retrieval, feedback, and iterative repair within a single proving policy. To support this framework, we construct OProofs, a large-scale Lean 4 corpus containing approximately 1.77M statements,

- 6.86M compiler-verified proofs, and serialized agentic proving trajectories that record retrieval contexts, failed attempts, compiler feedback, and subsequent repairs. OProofs provides Leanspecific supervision across continued pretraining, supervised fine-tuning, and reinforcement learning, and serves as the memory indexed for retrieval at proving time. Using OProofs, we train OProver-8B and OProver-32B through continued pretraining followed by iterative post-training with SFT and RL, where verified proofs and repair trajectories produced by the current prover are folded back into the corpus for subsequent iterations. Across five Lean 4 benchmarks, OProver-32B attains the best Pass@32 on MiniF2F, ProverBench, and PutnamBench, and the second-best on MathOlympiad and ProofNet, achieving more top placements than any prior open-weight whole-proof prover.

##### 7. Contributions and Acknowledgments

Multimodal Art Projection (M-A-P) is a non-profit open-source AI research community, run by donations. The community members are working on research topics in a wide range of spectrum, including but not limited to the pre-training paradigm of foundation models, largescale data collection and processing, and the derived applications on coding, reasoning, and music generation.

###### Core Contributors

- • David Ma, M-A-P • Kaijing Ma, M-A-P

Contributors

- • Shawn Guo, M-A-P
- • Enduo Zhao, M-A-P, Monolith
- • Yunfeng Shi, M-A-P
- • Jiajun Shi, M-A-P, Beihang University
- • Jiameng Huang, M-A-P, Peking Univer-
- sity
- • Bingrui Li, M-A-P, Peking University
- • Zhenzhu Yang, M-A-P
- • Zhaoxiang Zhang, CASIA
- Corresponding Authors
- • Gavin Cheung, M-A-P
- • Jiaheng Liu, M-A-P, Nanjing University

• Zili Wang, M-A-P

##### References

Z. Azerbayev, B. Piotrowski, H. Schoelkopf, E. W. Ayers, D. Radev, and J. Avigad. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. arXiv preprint arXiv:2302.12433, 2023.

- K. Baba, C. Liu, S. Kurita, and A. Sannai. Prover agent: An agent-based framework for formal mathematical proofs. arXiv preprint arXiv:2506.19923, 2025.
- L. Chen, J. Gu, L. Huang, W. Huang, Z. Jiang, A. Jie, X. Jin, X. Jin, C. Li, K. Ma, et al. Seed-prover: Deep and broad reasoning for automated theorem proving. arXiv preprint arXiv:2507.23726, 2025.

L. de Moura and S. Ullrich. The lean 4 theorem prover and programming language. In A. Platzer and G. Sutcliffe, editors, Automated Deduction – CADE 28, pages 625–635, Cham, 2021. Springer International Publishing.

K. Dong and T. Ma. Stp: Self-play llm theorem provers with iterative conjecturing and proving. arXiv preprint arXiv:2502.00212, 2025.

E. First, M. N. Rabe, T. Ringer, and Y. Brun. Baldur: Whole-proof generation and repair with large language models. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 1229–1241, 2023.

T. Gao, A. Wettig, H. Yen, and D. Chen. How to train long-context language models (effectively). In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7376–7399, 2025.

- Q. Guo, J. Wang, J. Zhang, D. Kong, X. Huang, X. Xi, W. Wang, J. Wang, X. Cai, S. Zhang, et al. Autoformalizer with tool feedback. arXiv preprint arXiv:2510.06857, 2025.

K. Gupta, J. Limperg, and U. Ghai. Tool-assisted multi-turn theorem proving with llms. In The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025.

- J. M. Han, J. Rute, Y. Wu, E. W. Ayers, and S. Polu. Proof artifact co-training for theorem proving with language models. arXiv preprint arXiv:2102.06203, 2021.

S. Huang, T. Cheng, J. K. Liu, W. Xu, J. Hao, L. Song, Y. Xu, J. Yang, J. Liu, C. Zhang, et al. Opencoder: The open cookbook for top-tier code large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 33167–33193, 2025.

- X. Ji, Y. Liu, Q. Wang, J. Zhang, Y. Yue, R. Shi, C. Sun, F. Zhang, G. Zhou, and K. Gai. Leanabellprover-v2: Verifier-integrated reasoning for formal theorem proving via reinforcement learning. arXiv preprint arXiv:2507.08649, 2025.

- A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamnik, T. Lacroix, Y. Wu, and G. Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. arXiv preprint arXiv:2210.12283, 2022.

- G. Lample, T. Lacroix, M.-A. Lachaux, A. Rodriguez, A. Hayat, T. Lavril, G. Ebner, and X. Martinet. Hypertree proof search for neural theorem proving. Advances in neural information processing systems, 35:26337–26349, 2022.

- H. Lin, Z. Sun, S. Welleck, and Y. Yang. Lean-star: Learning to interleave thinking and proving. arXiv preprint arXiv:2407.10040, 2024.

- Y. Lin, S. Tang, B. Lyu, J. Wu, H. Lin, K. Yang, J. Li, M. Xia, D. Chen, S. Arora, et al. Goedel-prover: A frontier model for open-source automated theorem proving. arXiv preprint arXiv:2502.07640, 2025a.
- Y. Lin, S. Tang, B. Lyu, Z. Yang, J.-H. Chung, H. Zhao, L. Jiang, Y. Geng, J. Ge, J. Sun, et al. Goedelprover-v2: Scaling formal theorem proving with scaffolded data synthesis and self-correction. arXiv preprint arXiv:2508.03613, 2025b.

- A. Liu, A. Mei, B. Lin, B. Xue, B. Wang, B. Xu, B. Wu, B. Zhang, C. Lin, C. Dong, et al. Deepseekv3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

- R. K. Mahabadi, S. Satheesh, S. Prabhumoye, M. Patwary, M. Shoeybi, and B. Catanzaro. Nemotron-cc-math: A 133 billion-token-scale high quality math pretraining dataset. arXiv preprint arXiv:2508.15096, 2025.

Z. Peng, Y. Yao, K. Ma, S. Guo, Y. Li, Y. Zhang, C. Zhang, Y. Zhang, Z. Yu, L. Li, et al. Criticlean: Critic-guided reinforcement learning for mathematical formalization. arXiv preprint arXiv:2507.06181, 2025.

- S. Polu and I. Sutskever. Generative language modeling for automated theorem proving. arXiv preprint arXiv:2009.03393, 2020.

Z. Ren, Z. Shao, J. Song, H. Xin, H. Wang, W. Zhao, L. Zhang, Z. Fu, Q. Zhu, D. Yang, et al. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025.

S. Shang, R. Wan, Y. Peng, Y. Wu, X.-h. Chen, J. Yan, and X. Zhang. Stepfun-prover preview: Let’s think and verify step by step. arXiv preprint arXiv:2507.20199, 2025.

Y. Tao and H. Xu. Theoremforge: Scaling up formal data synthesis with low-budget agentic workflow. arXiv preprint arXiv:2601.17332, 2026.

- K. Team, T. Bai, Y. Bai, Y. Bao, S. Cai, Y. Cao, Y. Charles, H. Che, C. Chen, G. Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

- G. Tsoukalas, J. Lee, J. Jennings, J. Xin, M. Ding, M. Jennings, A. Thakur, and S. Chaudhuri. Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. CoRR, abs/2407.11214, 2024. doi: 10.48550/ARXIV.2407.11214. URL https://doi.org/10

.48550/arXiv.2407.11214.

- H. Wang, M. Unsal, X. Lin, M. Baksys, J. Liu, M. D. Santos, F. Sung, M. Vinyes, Z. Ying, Z. Zhu, et al. Kimina-prover preview: Towards large formal reasoning models with reinforcement learning. arXiv preprint arXiv:2504.11354, 2025.

- J. Wang, J. Zhang, Q. Guo, L. Guo, R. Li, C. Zhang, C. Peng, C. Wang, D. Zhao, J. Shi, et al. Longcat-flash-prover: Advancing native formal reasoning via agentic tool-integrated reinforcement learning. arXiv preprint arXiv:2603.21065, 2026.

- Y. Wu, A. Q. Jiang, W. Li, M. Rabe, C. Staats, M. Jamnik, and C. Szegedy. Autoformalization with large language models. Advances in neural information processing systems, 35:32353–32368, 2022.

- Z. Wu, S. Huang, Z. Zhou, H. Ying, Z. Yuan, W. Zhang, D. Lin, and K. Chen. Internlm2. 5stepprover: Advancing automated theorem proving via critic-guided search. In 2nd AI for Math Workshop@ ICML 2025, 2025.

H. Xin, Z. Ren, J. Song, Z. Shao, W. Zhao, H. Wang, B. Liu, L. Zhang, X. Lu, Q. Du, et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152, 2024.

R. Xin, C. Xi, J. Yang, F. Chen, H. Wu, X. Xiao, Y. Sun, S. Zheng, and M. Ding. Bfs-prover: Scalable best-first tree search for llm-based automatic theorem proving. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32588–32599, 2025.

- K. Yang, A. Swope, A. Gu, R. Chalamala, P. Song, S. Yu, S. Godil, R. J. Prenger, and A. Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models. Advances in Neural Information Processing Systems, 36:21573–21612, 2023.

H. Ying, Z. Wu, Y. Geng, J. Wang, D. Lin, and K. Chen. Lean workbook: A large-scale lean problem set formalized from natural language math problems, 2024.

C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

K. Zheng, J. M. Han, and S. Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021.

- M. Zimmer, X. Ji, R. Tutunov, A. Bordg, J. Wang, and H. B. Ammar. Bourbaki: Self-generated and goal-conditioned mdps for theorem proving. arXiv preprint arXiv:2507.02726, 2025.

- Table 4. Retriever comparison on 1,000 queries with 9,000 candidates. Relevance is assessed by GPT-4o pairwise judgment over top-5 retrieved contexts.

Retriever Sim. spread Judge pref.

Goedel-Prover-V2-8B [0.85,0.99] 333 / 1000 Qwen3-8B-Embedding [0.20,0.99] 667 / 1000

##### A. Additional Method Details

This appendix provides additional details on retriever selection, repair instance construction, data recirculation, training and evaluation hyperparameters, and pretraining data recall.

###### A.1. Retriever Selection

- Section 2.1 describes how OProver retrieves top-𝑘 compiler-verified proofs from a memory M at each proving round. This appendix details the retriever model selection.

We compared two candidate retrievers: Qwen3-8B-Embedding and Goedel-Prover-V2-8B. The comparison was conducted on a held-out retrieval set of 1,000 query statements and 9,000 candidate retrieval documents.

Similarity score distribution. Qwen3-8B-Embedding produced similarity scores spanning approximately [0.20,0.99], whereas Goedel-Prover-V2-8B produced scores concentrated in a narrow high-similarity range [0.85,0.99]. The broader spread of Qwen3-8B-Embedding provides a more discriminative ranking signal: when score variance is small across candidates, top-𝑘 selection becomes effectively random, whereas a broader spread aligns top-𝑘 selection with semantic relevance.

Pairwise LLM-as-a-judge evaluation. We further evaluated retrieval quality using GPT-4o in a pairwise LLM-as-a-judge setup. For each query, GPT-4o compared the top-5 retrieved theorem contexts returned by the two retrievers. Over 1,000 query-level comparisons, Qwen38B-Embedding was preferred in 667 cases (66.7%), while Goedel-Prover-V2-8B was preferred in 333 cases (Table 4). Based on these results, OProver uses Qwen3-8B-Embedding as the default retriever.

A.2. Repair Instance Construction Details

- Section 2.2 defines a round-level repair instance as (𝑠, R, 𝑝−, 𝑓−, 𝑝+). We describe here how such instances are extracted from raw proving trajectories.

Extraction. Given an agentic proving trajectory of 𝑇 rounds (𝑝1, 𝑓1, 𝑝2, 𝑓2, . . . , 𝑝𝑇), each transition (𝑝𝑡−1, 𝑓𝑡−1) → 𝑝𝑡 for 𝑡 ≥ 2 becomes one repair instance, paired with the retrieval context R𝑡 used at round 𝑡. A trajectory of 𝑇 rounds therefore yields up to 𝑇 − 1 repair instances.

Filtering. We discard instances in which (i) 𝑝𝑡−1 is empty or syntactically malformed before verification, (ii) 𝑓𝑡−1 exceeds 8,000 tokens (these are dominated by repeated boilerplate errors),

or (iii) 𝑝𝑡 differs from 𝑝𝑡−1 by fewer than 3 tokens (near-no-op revisions). The retained repair instances form the primary supervision substrate for feedback-aware post-training.

Deduplication. Repair instances are deduplicated by exact match on (𝑝−, 𝑓−, 𝑝+) to avoid multiple trajectories producing identical local transitions.

###### A.3. Data Recirculation Details

- Section 2.3 describes how each post-training iteration recirculates the prover’s rollouts into OProofs, the retrieval memory, the SFT repair set, and the RL hard-case set. We provide additional implementation details below.

Routing rules. After each iteration, each rollout is routed according to its outcome and the success rate of its theorem’s group (the 𝑛 rollouts sampled for the same theorem):

- • Rollouts that yield a Lean-verified proof contribute their final proof to P𝑡+, which is appended to OProofs D𝑡+1 and re-indexed into the retrieval memory M𝑡+1.
- • Repair instances extracted from successful rollouts are added to the SFT set B𝑡.
- • Theorems whose group success rate is in (0,1)—that is, not solved by all rollouts and not failed by all rollouts—are retained as hard cases H𝑡 for the next round of RL.

The routing follows DAPO-style group filtering, which empirically reduces gradient variance by excluding both fully-solved and fully-failed groups.

###### A.4. Training and Evaluation Hyperparameters

Continued pretraining. OProver-Base is obtained by continued pretraining on the 65B-token mixture described in Section 2.3, using AdamW with peak learning rate 5×10−5, cosine schedule with 3% warmup, global batch size 512, and sequence length 8192.

Supervised fine-tuning. OProver-8B and OProver-32B are fine-tuned with global sequence length 40,960 and global batch size 64. Learning rates are 2 × 10−5 for OProver-8B and 5 × 10−5 for OProver-32B. Cross-entropy loss is computed only on the target proof attempt 𝑝𝑡.

Reinforcement learning. We adopt GSPO with learning rate 2 × 10−6, training batch size 256, PPO mini-batch size 256, and 𝑛 = 8 rollouts per theorem. The maximum prompt length is 14,000 tokens. The maximum response length is 24,000 tokens for OProver-8B and 20,000 tokens for OProver-32B. The maximum number of refinement rounds during RL is 4 for OProver-8B and 2 for OProver-32B. Rollout decoding uses temperature 1.0 and top-𝑝 0.999. The per-round verification reward is 𝑟𝑡 = 0.8+0.2·1[format correct] when V(𝑝𝑡) =verified, and 𝑟𝑡 = 0 otherwise. Advantages are computed by group-relative normalization pooled across the 𝑛 × 𝑅 rounds for the same theorem. Lean verification during RL uses proof reconstruction with fully explicit proofs, a per-request timeout of 120s, and a heartbeat cap of 2M.

Evaluation. At evaluation time, we use the same multi-turn proving interface as RL training. Generation uses temperature 1.0, top-𝑝 0.999, and a maximum response length of 32,000 tokens.

Proof verification uses the Kimina Lean server (Lean 4.15.0) with proof reconstruction and fully explicit proofs, a per-request timeout of 240s, and a heartbeat cap of 4M.

###### A.5. Pretraining Data Recall

We recall mathematics-adjacent examples from the pretraining corpus through a two-stage pipeline.

- Stage 1: iterative fastText recall. We initialize a seed set with verified proofs and repair instances from OProofs, and train a fastText classifier to distinguish seed documents from background pretraining documents. The trained model is applied to the pretraining corpus to retrieve a high-recall candidate pool. High-scoring candidates are merged back into the seed set, after which the classifier is retrained for another retrieval round. This train–retrieve–expand cycle is repeated to broaden coverage beyond exact Lean syntax matches.
- Stage 2: embedding-based precision filter. After the final fastText iteration, both seed examples and recalled candidates are encoded with a small embedding model trained on the same domain data. Candidates whose nearest-seed cosine similarity falls below a threshold 𝜏 are discarded. The retained documents form the recalled pretraining subset, which is mixed into the CPT data described in Section 2.3.

##### B. Prompt Templates and Feedback Format

This appendix specifies the prompt template and the feedback serialization used by OProver during both training and inference, following the state formulation 𝑋𝑡 = (𝑠, R𝑡, 𝑝𝑡−1, 𝑓𝑡−1) from Section 2.1.

###### B.1. Prompt Template

We use a single prompt template across all rounds. At round 𝑡, the model conditions on the target Lean 4 statement, the top-𝑘 retrieved references, and the immediately preceding proof attempt with its Lean feedback. When 𝑡 = 1, the previous-attempt and feedback fields are left empty.

###### Unified Prompt Template at Round 𝑡

**Current Task:** Complete the following Lean 4 code: {FORMAL_STATEMENT}

Before producing the Lean 4 proof, first provide a concise proof plan summarizing the intended strategy, key lemmas, and proof structure. If a previous attempt and Lean feedback are provided, revise the proof accordingly.

Reference theorems and proofs:

- {RETRIEVED_PROOF_1}
- {RETRIEVED_PROOF_2}

Previous Failed Attempt: {PREVIOUS_PROOF_ATTEMPT}

Lean Feedback: {LEAN_ERROR_MESSAGES}

###### B.2. Feedback Serialization

Lean feedback is serialized as the raw plain-text diagnostic returned by the Lean 4 compiler for the immediately preceding proof attempt, with no projection into a fixed error taxonomy. This preserves identifier names, expected/actual types, and goal contexts that targeted revision often depends on. An example diagnostic, after stripping ANSI control codes, looks like:

###### Example Lean Feedback

type mismatch h0 has type Nat.succ ?m.123 but is expected to have type Real

tactic ’linarith’ failed to find a contradiction case h x y : Real

- h0 : x <= y
- h1 : y <= 0 turnstile False

###### B.3. Common Feedback Categories

Although feedback is passed in its raw textual form, the underlying diagnostics fall into a few recurring categories, which we summarize in Table 5 together with the typical repair actions we observe across multi-round refinement.

In raw OProver logs, a non-verifier failure mode no_lean_code_found also appears, reflecting generation or proof-extraction failure rather than a Lean compiler diagnostic. We exclude it from Table 5 since it does not represent native verifier feedback.

##### C. Effectiveness of Agentic Reasoning

This appendix provides qualitative evidence for the effectiveness of the agentic refinement loop in OProver. In our setting, the policy does not generate a proof in a single shot. Instead, it iteratively refines candidate proofs by conditioning on retrieved formal references and Lean verifier feedback. The examples below illustrate how these two sources of information support grounded repair behavior across rounds. For space, all proof excerpts and error messages are abridged but preserve the essential failure mode and the corresponding repair.

- Table 5. Common Lean feedback categories observed in OProver’s proving trajectories, with the typical repair actions taken in the subsequent round.

Feedback type Typical repair action

Unsolved goals / goal mismatch

Introduce missing intermediate lemmas or case splits, restate the target in a more convenient form, or align the local proof state before continuing.

Tactic or automation failure Replace brittle one-shot automation with explicit derivation steps, reorder tactics, or simplify the context before invoking automation such as linarith, omega, or simp.

Type or term mismatch Rewrite ill-typed expressions, add explicit coercions or binders, and ensure intermediate terms match the expected type.

Unknown identifier / namespace / notation error

Qualify names explicitly, open the required namespace, fix notation, or replace unavailable lemmas with in-scope equivalents.

Incomplete induction or case split

Add missing branches, make parity or sign cases explicit, and construct the required witnesses before closing the proof.

Timeout / heartbeat exhaustion Shorten proof scripts, reduce expensive search, and replace heavy automation with more structured local reasoning.

###### C.1. Roles of Retrieval and Compiler Feedback

Retrieved formal references are most useful in three recurring ways. First, they help ground lemma names, rewrite rules, and tactic idioms that are easy to miss in one-shot generation. Second, they provide short structural hints about how to decompose a goal into subgoals that better match Lean’s expected proof state. Third, they expose proof patterns that help the model avoid common formulation errors, such as missing coercions, inappropriate tactic ordering, or mismatched algebraic normal forms. In practice, these signals are lightweight, but they often suffice to redirect a stalled proof attempt toward a more verifiable local proof structure.

Lean feedback provides a complementary signal: it identifies concrete verifier-side failures that can be mapped to local edits in the next round. Rather than projecting compiler outputs into a hand-designed taxonomy during inference, OProver conditions directly on the original textual diagnostics. This preserves fine-grained information about the actual source of failure—including type mismatches, missing identifiers, unresolved goals, or tactic failures. Common categories of such feedback and the corresponding repair actions are listed in Table 5 of Appendix B.

Presentation note. For compatibility with the pdflatex toolchain, the proof excerpts below normalize Lean Unicode notation into an ASCII display form while preserving the logical structure of each repair trace.

###### C.2. Case Study A: Putnam 1963 A3

This example illustrates a simple but representative local repair. The first-round attempt introduces an ill-typed contradiction block; the second-round repair removes the offending fragment and restores a compilable scaffold for subsequent refinement.

###### Round 1 Proof Attempt

import Mathlib set_option maxHeartbeats 2000000 open Nat Set Topology Filter theorem putnam_1963_a3 (P : Nat -> (Real -> Real) -> (Real -> Real)) (hP : P 0 = id /\ forall i y, P (i + 1) y = P i (fun x => x * deriv y x - i * y x)) (n : Nat) (hn : 0 < n) (f y : Real -> Real) (hf : ContinuousOn f (Ici 1)) (hy : ContDiffOn Real n y (Ici 1)) (hy1 : ContDiffAt Real n y 1) : (forall i, i < n -> deriv^[i] y 1 = 0) /\ (Ici 1).EqOn (P n y) f <-> forall x, x >= 1 -> y x = integral ... := by

- have h1 : False := by
- have h2 : forall (alpha : Nat -> Type) (a : alpha 0), (forall i : Nat, alpha (i + 1)) -> True := by

intro alpha a h trivial

- have h3 := h2 (fun n => Nat) 0
- have h4 := h3 (by decide) -- ill-typed use of decide trivial exfalso exact h1

###### Round 1 Lean Feedback

Lean errors: application type mismatch @decide ((i : Nat) -> (fun n => Nat) (i + 1)) argument has type Type but is expected to have type Prop

tactic ’assumption’ failed Goal: False

Reasoning summary. The failure is purely local: the proof attempt uses decide on a typevalued term rather than a proposition. The feedback directly identifies the mismatch, allowing the next round to remove the ill-typed block rather than continuing to elaborate an invalid contradiction.

###### Round 2 Revised Proof

import Mathlib set_option maxHeartbeats 2000000 open Nat Set Topology Filter theorem putnam_1963_a3 (P : Nat -> (Real -> Real) -> (Real -> Real))

(hP : P 0 = id /\ forall i y, P (i + 1) y = P i (fun x => x * deriv y x - i * y x)) (n : Nat) (hn : 0 < n) (f y : Real -> Real) (hf : ContinuousOn f (Ici 1)) (hy : ContDiffOn Real n y (Ici 1)) (hy1 : ContDiffAt Real n y 1) : (forall i, i < n -> deriv^[i] y 1 = 0) /\ (Ici 1).EqOn (P n y) f <-> forall x, x >= 1 -> y x = integral ... := by

-- The invalid contradiction block is removed.

-- A compilable scaffold is restored for subsequent refinement. sorry

Repair summary. The repair removes the ill-typed decide construction and restores type correctness. Although this example does not yet complete the theorem, it shows that verifier feedback can reliably eliminate a local blocking error and recover a viable proof scaffold for future rounds.

###### C.3. Case Study B: Putnam 1986 A1

This example shows a more substantive repair. The first-round attempt reaches the right highlevel objective but fails to make the algebraic bounds explicit enough for linarith. The second-round revision introduces the missing interval constraints on 𝑦2 and then performs a bounded case split.

###### Round 1 Proof Attempt

theorem putnam_1986_a1 (S : Set Real) (f : Real -> Real) (hS : S = {x : Real | x ^ 4 + 36 <= 13 * x ^ 2}) (hf : f = fun x => x ^ 3 - 3 * x) : IsGreatest {f x | x in S} 18 := by have h_main : IsGreatest {f x | x in S} 18 := by

- have h1 : forall (x : Real), x in S -> f x <= 18 := by intro x hx
- have h2 : x ^ 4 + 36 <= 13 * x ^ 2 := by rw [hS] at hx exact hx
- have h3 : x <= 3 := by by_contra h
- have h4 : x > 3 := by linarith
- have h5 : x ^ 2 > 9 := by nlinarith
- have h6 : x ^ 4 + 36 > 13 * x ^ 2 := by nlinarith

linarith

- have h4 : f x <= 18 := by rw [hf]
- have h5 : x ^ 3 - 3 * x <= 18 := by nlinarith

linarith -- automation does not close the goal exact h4 sorry exact h_main

linarith failed to find a contradiction

Context: x : Real hx : x in S

- h2 : x ^ 4 + 36 <= 13 * x ^ 2
- h3 : x <= 3 h5 : x ^ 3 - 3 * x <= 18 Goal: False

Reasoning summary. The proof attempt is directionally correct, but the arithmetic state is too implicit for linarith to finish. The feedback suggests that the missing step is not a different theorem, but a more explicit decomposition of the polynomial constraint into usable bounds.

Round 2 Revised Proof

theorem putnam_1986_a1 (S : Set Real) (f : Real -> Real) (hS : S = {x : Real | x ^ 4 + 36 <= 13 * x ^ 2}) (hf : f = fun x => x ^ 3 - 3 * x) : IsGreatest {f x | x in S} 18 := by

- have h1 : forall (y : Real), y in S -> f y <= 18 := by intro y hy
- have h2 : y ^ 4 + 36 <= 13 * y ^ 2 := by rw [hS] at hy exact hy
- have h3 : 4 <= y ^ 2 /\ y ^ 2 <= 9 := by

-- derive explicit bounds from the polynomial constraint constructor · by_contra h nlinarith · by_contra h nlinarith

- have h4 : y >= 2 \/ y <= -2 := by by_cases hy2 : y >= 2 · exact Or.inl hy2 · have : y <= -2 := by nlinarith [h3.1]

exact Or.inr this cases h4 with | inl hypos =>

have hyub : y <= 3 := by nlinarith [h3.2] rw [hf] nlinarith

| inr hyneg => have hylb : y >= -3 := by nlinarith [h3.2] rw [hf] nlinarith

sorry

Repair summary. The revised proof makes the hidden algebraic structure explicit by deriving

- 4 ≤ 𝑦2 ≤ 9, then splitting into the two sign-consistent cases. This produces subgoals that match the strengths of arithmetic automation and allows the upper-bound argument to go through.

###### C.4. Case Study C: Putnam 1992 A1

This example illustrates a more complex multi-error repair. The first-round attempt mixes invalid integer destructuring, incomplete induction branches, inconsistent local bindings, and incorrect uses of congr_fun. The next round repairs these structural failures by restructuring the argument around explicit tuple destructuring, injectivity of 𝑓, and systematic parity and sign case analysis over the integers; three induction branches remain as sorry placeholders for subsequent rounds.

###### Round 1 Proof Attempt

theorem putnam_1992_a1 (f : Int -> Int) : (f = fun n => 1 - n) <-> (forall n : Int, f (f n) = n) /\ (forall n : Int, f (f (n + 2) + 2) = n) /\ (f 0 = 1) := by have h_formula : (forall n : Int, f (f n) = n) -> (forall n : Int, f (f (n + 2) + 2) = n) -> (f 0 = 1) -> (forall n : Int, f n = 1 - n) := by intro hf1 hf2 h0 have h_rec : forall n : Int, f (n + 2) = f n - 2 := by sorry have h_even : forall k : Int, f (2 * k) = 1 - 2 * k := by intro k induction k using Int.induction_on with | hz =>

norm_num [h0]

| hp k ih => have h := h_rec (2 * k) omega

-- missing negative branch have h_odd : forall k : Int, f (2 * k + 1) = -2 * k := by intro k induction k using Int.induction_on with | hz =>

norm_num [h0]

| hp k ih => have h := h_rec (2 * k + 1) omega

-- missing negative branch have hpar : n % 2 = 0 \/ n % 2 = 1 := by omega cases hpar with | inl h =>

obtain (k, hk) := by use n / 2 omega

have h := h_even k rw [hk] at h omega

| inr h =>

obtain (k, hk) := by use (n - 1) / 2 omega

have h := h_odd k rw [hk] at h omega

invalid match-expression, type of pattern variable contains metavariables type mismatch in function equality application type mismatch at congr_fun alternative ’hn’ has not been provided unknown identifier

Reasoning summary. Unlike the previous examples, this proof fails for multiple interacting reasons. The feedback indicates that the problem is structural rather than purely algebraic: the induction over integers is incomplete, some local names are inconsistent, and the proof attempts to use equalities at the wrong type. The next round therefore needs to reorganize the proof, not merely patch one line.

###### Round 2 Revised Proof

theorem putnam_1992_a1 (f : Int -> Int) : (f = fun n => 1 - n) <-> (forall n : Int, f (f n) = n) /\ (forall n : Int, f (f (n + 2) + 2) = n) /\ (f 0 = 1) := by constructor · intro h have h1 : f = fun n : Int => 1 - n := h refine (?_, ?_, ?_) · intro n

- have hn1 := congr_fun h1 n
- have hn2 := congr_fun h1 (f n) simp [hn1, hn2] <;> ring_nf <;> linarith · intro n

- have hn1 := congr_fun h1 (n + 2)
- have hn2 := congr_fun h1 (f (n + 2))
- have hn3 := congr_fun h1 (f (n + 2) + 2) simp [hn1, hn2, hn3] at * <;> ring_nf at * <;> linarith · have h0 := congr_fun h1 0

- simpa using h0 · rintro (hf1, hf2, h0) have h1 : f 1 = 0 := by have h01 : f (f 0) = 0 := hf1 0 rw [h0] at h01
- simpa using h01 have hrec : forall n : Int, f (n + 2) = f n - 2 := by intro n

intro a b hab

- have ha := hf1 a
- have hb := hf1 b rw [hab] at ha simpa using ha.trans hb.symm

apply hinj calc

f (f (n + 2) + 2) = n := hf2 n

_ = f (f n) := by simpa [hf1 n] have hsum : forall n : Int, f n + n = 1 := by intro n have h_even : forall k : Int, f (2 * k) + 2 * k = 1 := by

intro k by_cases hk : k >= 0 · obtain (m, hm) : exists m : Nat, (m : Int) = k := by

use k.toNat simp [Int.toNat_of_nonneg hk]

rw [<- hm]

-- positive branch proved by induction on m using hrec sorry

· obtain (m, hm) : exists m : Nat, -((m : Int)) = k := by use (-k).toNat have : 0 <= -k := by linarith simp [Int.toNat_of_nonneg this]

rw [<- hm]

-- negative branch proved by induction on m using hrec sorry

have h_odd : forall k : Int, f (2 * k + 1) + (2 * k + 1) = 1 := by intro k

-- analogous parity argument, initialized from h1 sorry

have hpar : n % 2 = 0 \/ n % 2 = 1 := by omega cases hpar with | inl h =>

obtain (k, hk) : exists k : Int, n = 2 * k := by use n / 2 omega

rw [hk] simpa [two_mul] using h_even k

| inr h =>

obtain (k, hk) : exists k : Int, n = 2 * k + 1 := by use (n - 1) / 2 omega

rw [hk] simpa [two_mul, add_assoc] using h_odd k

funext n have hn : f n + n = 1 := hsum n linarith

Repair summary. The revised proof closes the structural failures of the first round—invalid integer destructuring, incomplete induction handling, and congr_fun misuse—by using explicit tuple destructuring for both directions, deriving the recurrence through injectivity of 𝑓, and dispatching the integer domain by parity together with separate nonnegative and negative branches. Three induction sub-branches remain as sorry for subsequent rounds. The key improvement here is not a single local fix, but a reorganization of the proof into a structure on which later refinement can build.

Overall takeaway. The three case studies span a range of repair difficulties. In Case A, a single Lean diagnostic eliminates an ill-typed fragment and restores a compilable scaffold. In Case B, the diagnostic signals not a wrong theorem but a missing decomposition, leading to a substantive but local rewrite that surfaces the algebraic bounds required for the next tactic to close. In Case C, multiple interacting failures prompt a structural reorganization of the proof, replacing the unstable scaffold with one that subsequent rounds can refine toward completion.

Together, these traces show that the feedback-conditioned refinement loop scales gracefully from one-line local edits to multi-step structural revisions, matching the granularity of the repair to the failure mode reported by Lean.

###### C.5. Statement Labeling Prompt

In addition to the prover-side template, we use a separate offline classification prompt to annotate each Lean 4 statement with a mathematical domain (one of 10 categories) and a difficulty level (one of 4 tiers), as reported in Figure 3(b,c). The full prompt is shown below.

###### Lean 4 Statement Domain & Difficulty Classification Prompt

**Instruction: Lean 4 Statement Classification (Mathematical Domain & Difficulty Level)**

You are an expert mathematician and Lean 4 specialist. You will receive a Lean 4 formal statement. Your task is to classify it along two dimensions:

- 1. **Mathematical Domain** (one of 10 categories)
- 2. **Difficulty Level** (one of 4 levels, by mathematical sophistication required, NOT by Lean proof length)

---

- **DIMENSION 1: MATHEMATICAL DOMAIN**

Choose **exactly one** primary domain. If the statement spans multiple domains, pick the most central one.

- - **Algebra**: abstract algebra, groups/rings/fields, linear algebra, polynomials, algebraic identities, algebraic inequalities
- - **NumberTheory**: divisibility, congruences, primes, Diophantine equations, modular arithmetic, analytic number theory
- - **Analysis**: real/complex/functional analysis, calculus, limits, continuity, series, (non-probabilistic) measure theory, analytic inequalities
- - **Topology**: general topology, algebraic topology, differential topology, topological spaces, continuity in topological settings
- - **Geometry**: planar/solid/analytic geometry, differential geometry, geometric inequalities, classical geometry problems
- - **Combinatorics**: combinatorial identities, graph theory, counting, discrete mathematics, Ramsey-style problems
- - **ProbabilityStatistics**: probability theory, statistics, probabilistic inequalities, stochastic processes
- - **LogicFoundations**: logic, set theory, type theory, model theory, foundational results, definability
- - **Computation**: algorithms, complexity, formal languages, information theory, CS-flavor mathematics
- - **Other**: anything that does not fit the above, including trivially true statements, malformed statements, or non-mathematical content.
- ---

- **DIMENSION 2: DIFFICULTY LEVEL**

Choose **exactly one** level based on the mathematical sophistication required to prove the statement. Do NOT base difficulty on the length or syntactic complexity of the Lean proof.

- - **Elementary**: middle-school level or below; basic arithmetic, simple

- algebraic manipulations, elementary number facts, basic geometric facts.
- - **HighSchool**: high-school level math; standard algebra/trigonometry/ precalculus, basic combinatorics, standard introductory contests (AMC, AIME, Chinese gaokao).
- - **Undergraduate**: standard undergraduate mathematics; linear algebra, single/multi-variable calculus, introductory abstract algebra, introductory real analysis, introductory topology.
- - **GraduatePlus**: graduate-level or research mathematics; advanced topology, functional analysis, algebraic geometry, advanced number theory. Also includes olympiad-level problems (e.g., IMO, USAMO, Putnam) that require nontrivial mathematical insight beyond standard curriculum.

- *Note 1: When in doubt between two adjacent levels, prefer the lower level unless the statement clearly requires the higher one.*
- *Note 2: For statements that are trivially true (e.g., ‘True‘, ‘0 = 0‘), malformed, or otherwise not meaningful mathematical claims, classify the domain as ‘Other‘ and the difficulty as ‘Elementary‘.*

---

- **STRICT OUTPUT REQUIREMENTS:**

- 1. Output **ONLY** a single JSON object on a single line.
- 2. Do **NOT** include any explanation, markdown formatting (no ‘‘‘json fences), or extra text outside the JSON.
- 3. All three fields are required.

**Output format (single JSON object):** {"domain": "<one of the 10 domains above>", "difficulty": "<one of the 4 levels

above>", "rationale": "<one short sentence, max 25 words, explaining the classification>"}

---

**EXAMPLES**

- **Example 1 (Elementary, Algebra):** Statement: theorem ex (a b : Nat) : a + b = b + a := by ring Output: {"domain": "Algebra", "difficulty": "Elementary", "rationale": "Basic

commutativity of natural number addition, taught at elementary level."}

- **Example 2 (HighSchool, Combinatorics):** Statement: theorem ex (n : Nat) (hn : 5 <= n) : Nat.choose n 2 = n * (n - 1) / 2 := by sorry Output: {"domain": "Combinatorics", "difficulty": "HighSchool", "rationale":

"Standard binomial-coefficient identity, accessible at high-school combinatorics level."}

- **Example 3 (Undergraduate, Topology):** Statement: theorem ex {X : Type*} [TopologicalSpace X] [CompactSpace X] [T2Space X] :

NormalSpace X := by sorry Output: {"domain": "Topology", "difficulty": "Undergraduate", "rationale": "Standard

introductory topology result that compact Hausdorff spaces are normal."}

- **Example 4 (GraduatePlus, Analysis):**

Statement: theorem ex (f : Real -> Real) (hf : Continuous f) (h_nn : forall x, 0 <= f x) (h_zero : Real.intervalIntegral f 0 1 = 0) : forall x in Set.Icc 0 1, f x = 0 := by

sorry

Output: {"domain": "Analysis", "difficulty": "GraduatePlus", "rationale": "Standard measure-theoretic result that a non-negative continuous function with zero integral

vanishes on the domain."}

---

**[Input Content Starts]** {FORMAL_STATEMENT}

**[Input Content Ends]**

**Output (JSON only, single line):**

We run this prompt once per unique Lean 4 statement; 96.4% of responses parse to a wellformed JSON object with valid domain and difficulty values, yielding the 1.73M statement-level annotations summarized in Figure 3(b,c).

