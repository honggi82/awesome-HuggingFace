## Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty

Jeonghye Kim1,2*, Xufang Luo1†, Minbeom Kim3, Sangmook Lee3, Dongsheng Li1, Yuqing Yang1

1Microsoft Research 2KAIST, 3Seoul National University

# arXiv:2603.15500v2[cs.AI]26May2026

### Abstract

LLMs often exhibit Aha moments such as selfcorrection after tokens like "Wait," yet the underlying mechanism remains unclear. Standard LLMs collapse mainly through silent divergence, where trajectories drift from the correct answer yet remain locally coherent, so no explicit error triggers reactive self-correction. We introduce an information-theoretic framework that separates reasoning into procedural advancement and epistemic verbalization, the token-level externalization of uncertainty, and prove that sporadic verbalization restores convergence toward the correct answer even without explicit error triggers. Empirically, a minimal doubt cue recovers failed trajectories, and small-scale SFT suffices to instill or suppress this capability, suggesting that strong reasoning hinges less on an extraordinary inner mechanism than on the linguistic habit of externalizing uncertainty. Our framework recasts reasoning as strategic information allocation under uncertainty, offering a new lens for understanding and advancing LLM reasoning.

### 1 Introduction

Recent large language models (LLMs) often exhibit so-called Aha moments during reasoning, with behaviors such as self-correction or reflection appearing after tokens like "Wait" (Guo et al., 2025; Yang et al., 2025d). These phenomena are frequently cited as key mechanisms underlying effective reasoning, yet there remains little consensus on what computational or informational role they actually play (d’Aliberti and Ribeiro, 2026; Liu et al., 2025; Tsui, 2025). Prior work tends to group Aha moments, reflection, self-correction, and the emergence of specific tokens as a single class of phenomena, making it difficult to disentangle their underlying mechanisms.

*Work done during Jeonghye’s internship at Microsoft Research. † Corresponding author. Our analysis code can be found at link.

[Figure 1]

Figure 1: Two pathways for self-correction. The Procedural channel reacts only when an explicit violation triggers an alarm, whereas the Epistemic channel continuously monitors and prompts self-checking even without any visible contradiction.

In parallel, recent studies have examined reasoning from an information-theoretic perspective (Ton et al., 2025; Liang, 2025), reinterpreting Chain-ofThought (CoT) (Wei et al., 2022a) as a process of information accumulation toward the correct answer. While offering valuable insights, these approaches largely assume procedural, step-by-step execution and do not fully account for the selfcorrective behaviors of modern reasoning models, particularly recovery after entering an incorrect trajectory. Once execution enters an erroneous path, reasoning may remain locally coherent yet globally incorrect, with the underlying error unrecognized.

To investigate this gap, we analyze the reasoning behavior of standard LLMs and large reasoning models (LRMs). LLMs rely almost exclusively on reactive correction, revising their trajectory only when an explicit contradiction or failed check surfaces. When errors remain latent, no trigger arises and the trace drifts into reasoning collapse, the dominant failure mode. LRMs additionally exhibit proactive correction, questioning prior steps without any overt error and surfacing hidden mistakes

that reactive mechanisms cannot reach (Figure 1). Although such proactive signals are imprecise and often second-guess already-correct chains, in this regime even a noisy signal is more useful than a precise one that never fires.

Building on this observation, we identify an additional informational axis distinct from procedural information. Our key idea is epistemic uncertainty verbalization, the explicit externalization at the token level of a model’s internal uncertainty about its reasoning state. Since autoregressive generation conditions only on preceding tokens, latent uncertainty is informationally inert, whereas verbalized uncertainty becomes actionable and supports downstream control actions such as self-correction. From this perspective, commonly discussed tokens such as "Wait" are surface manifestations of epistemic verbalization rather than the mechanism itself, and reasoning can be reframed as strategic information allocation under uncertainty: the joint determination of how to advance the procedure and how much epistemic verbalization to allocate at each step, given the model’s current uncertainty.

We empirically validate that epistemic verbalization carries substantive information for reasoning. On failed trajectories, injecting a minimal doubt cue such as "Wait, is that correct?", without identifying what went wrong, recovers around 15% of incorrect rollouts, showing that the externalization of uncertainty itself supplies actionable signal. We further show that epistemic verbalization is a rapidly learnable capability. In line with the findings of Ye et al. (2025), as few as 800 training samples suffice to either instill or suppress this behavior, indicating that externalizing internal uncertainty requires no extraordinary ability but rather reflects a linguistic habit. Our new taxonomy and framing of LLM reasoning offers a unified lens for interpreting Aha moments, self-correction, and post-training phenomena, and points toward principled directions for designing and controlling reasoning models.

### 2 Related Works

Understanding Aha moments. Recent studies have cast doubt on Aha-like phenomena in LLM reasoning. d’Aliberti and Ribeiro (2026) show that markers such as “Wait” emerge from high-entropy prediction states and correlate only weakly with performance gains. Liu et al. (2025) find that apparent self-reflection often degenerates into repetition rather than producing constructive revisions.

A related line of evidence shows that LLMs correct externally provided errors yet fail to fix the same errors in their own outputs (Tsui, 2025; Huang et al., 2024; Tyen et al., 2024; Kamoi et al., 2024). These studies document when Aha tokens fail to predict correction, but leave open why: the marker itself may be unreliable, the model may lack a corrective policy, or both. The underlying mechanism remains unidentified.

Theoretical understanding of reasoning. A parallel line of work analyzes reasoning trajectories at the aggregate level. Yang et al. (2025c) show that reasoning can both fix and introduce errors by decoupling knowledge-based responses from reasoning-based corrections. Gan et al. (2025) recast CoT as optimization over reasoning states and identify trade-offs between noise reduction and generalization. Information-theoretic analyses show that CoT preserves task-relevant information and tightens error bounds (Ton et al., 2025; Liang, 2025). Qian et al. (2025) observe that effective reasoning concentrates information gain in a few critical steps, often marked by tokens like “Wait.” These accounts describe how information flows across a trajectory but treat every token as a procedural step, leaving unexplained how a model recovers after drifting onto an incorrect path without outside help.

We close both gaps with a unified informationtheoretic framework of reasoning at the token level. First, we identify epistemic verbalization, the externalization of uncertainty about one’s own reasoning, as the informational source that lets a model regain traction after a procedural misstep, and prove that even sporadic occurrences suffice to restore convergence. Second, we separate this informational role from the control actions, such as selfcorrection, that act on it. Together, these results reconcile the mixed empirical findings on Aha moments and supply the token-level grounding missing from aggregate information-theoretic accounts.

### 3 A Self-Conditioning Framework for LLM Reasoning

Our analysis mainly focuses on the closed-world inference setting, where an LLM operates without external observations at inference time. Unlike toolaugmented agents that reduce uncertainty through environmental interaction, a closed-world LLM is constrained to a fixed parameterization θ and an initial input x, so all progress toward correct inference

must come from internal belief transformation. We formalize this as self-Bayesian reasoning, where CoT (Wei et al., 2022b) acts as self-conditioning that reshapes the model’s belief over the target variable Y without new external observations. Appendix B extends this to the open-world setting.

- 3.1 Reasoning as Self-Conditioning Given an input x ∈ X, an LLM parameterized by

θ induces a conditional distribution Pθ(· | x) over tokens from vocabulary V, and autoregressively generates a sequence a1,...,aT until terminal step T. We define the reasoning state at step t as

s0 := x, st := (x,a1,...,at) for t ≥ 1, (1)

so that each token is drawn as at ∼ Pθ(· | st−1). Each state st induces a predictive distribution Pθ(Y | st) over the target variable Y ∈ Y, and we take the objective of reasoning to be the minimization of H(Y | sT), where H(·) denotes Shannon entropy. Reasoning thus proceeds by self-conditioning: each generated token refines the model’s belief over Y without introducing external evidence, and the information gain of a step is

∆t := H(Y | st−1) − H(Y | st). (2)

Self-generated intermediate steps serve as the model’s sole source of evidence, making their informational quality critical for reasoning success.

- 3.2 Procedural Reasoning and Its Collapse

A dominant class of self-generated evidence in LLM reasoning consists of procedural information: explicit step-by-step computations, symbolic manipulations, variable instantiations, and executions of learned subroutines (Lai et al., 2024; Feng et al., 2025; Oh et al., 2025; Ton et al., 2025). Formally, let 0 = t0 < t1 < ··· < tK = T partition a reasoning trace into sub-tasks with task-level state Uk := (x,a1,...,atk). Procedural reasoning is then a sequence Uk = Λθ(Uk−1,τk), where Λθ is an autoregressive execution operator implementing sub-task τk.

Reasoning collapse in standard LLMs. We examine how this regime fails in practice by analyzing reasoning traces from the Qwen2.5 and Qwen3-Base series (Qwen2.5-7B, Qwen2.5-32BInstruct, Qwen2.5-Math-7B (Yang et al., 2024), Qwen3-1.7B/8B/14B-Base (Yang et al., 2025a)) on AIME24/25, AMC23, and MATH500. These models exhibit strong step-by-step procedural reasoning

ability, achieve competitive performance on these benchmarks, and serve as standard baselines in recent math reasoning research (Ye et al., 2025; Li et al., 2025; Wang et al., 2025; Yang et al., 2025d). Using GPT-5 as an automated judge on 4,800 generated samples (600 problems × 8 samples), we quantify the prevalence of five recurring modes of reasoning collapse across the traces (Figure 2): incoherence, hallucination spirals, repetition, topic drift, and degenerate loops. Collapse is pervasive, occurring in 50–83% of incorrect responses and rising sharply with problem difficulty (Figure 2a). See Appendix D.1 for details.

[Figure 2]

2.5-7B

- 2.5-32B-Inst 3-1.7B-Base

3-8B-Base 3-14B-Base

50% 66% 80% 78%

- 55% 57% 73% 65%
- 56% 68% 83% 80% 49% 57% 78% 70% 54% 59% 78% 66%

| |36%<br><br>46% 37% 39% 42%<br><br>29%<br><br>45%<br><br>27%<br><br>33% 34%<br><br>22% 30% 23% 21%<br><br>11%| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

2.5-7B 2.5-32B 3-1.7B 3-8B 3-14B

(b) Collapse Type by Model

0

25

50

75

100

CollapseProportion(%)

Incoherence

Hallucination Spiral

Repetition

Topic Drift

Infinite Loop

[Figure 3]

40

60

80

100

CollapseRate(%)

Figure 2: Reasoning collapse analysis across models and benchmarks. (a) Collapse rate among incorrect responses per model and dataset. (b) Breakdown of collapse types per model. Model names omit the “Qwen” prefix (e.g., 2.5-7B refers to Qwen2.5-7B).

In each mode, the surface structure of step-bystep execution is preserved while Pθ(Y | st) drifts from the correct Y , often without overt error. This silent divergence is the dominant failure pattern, consistent with prior observations that procedural reasoning is brittle when a sub-task is unidentifiable (Ton et al., 2025) or silently misinstantiated through an early misjudgment.

- 3.3 How Models Escape (or Fail to Escape) Collapse

MATHAMC23AIME24AIME25

(a) Collapse Rate by Model & Dataset

We next ask how models escape, or fail to escape, these regimes. For each self-correction event in the traces above, together with traces from Large Reasoning Models (LRMs: DeepSeek-R1-DistillQwen-7B/32B (Guo et al., 2025), Qwen3-8B/14B (Yang et al., 2025a)), we distinguish two modes:

- • Reactive correction. The reasoning chain contains an explicit error, such as a contradiction, a failed check, or an invalid derivation, that the model recognizes and revises.
- • Proactive correction. No overt error has surfaced, yet the model questions a prior step and verifies or backtracks on internal doubt alone.

Examples of both types of modes are provided in Appendix I. In LLMs, self-correction occurs in

Type Model Reactive Proactive

n (%) n (%)

Qwen3-1.7B-Base† 12 (100.0) 0 (0.0) Qwen3-8B-Base† 19 (86.4) 3 (13.6) Qwen3-14B-Base† 23 (92.0) 2 (8.0) Qwen2.5-7B† 11 (100.0) 0 (0.0) Qwen2.5-Math-7B† 35 (79.5) 9 (20.5) Qwen2.5-32B-Instruct† 8 (80.0) 2 (20.0)

LLM

Qwen3-8B 576 (76.3) 179 (23.7) Qwen3-14B 484 (77.8) 138 (22.2) DeepSeek-R1-Distill-7B 475 (65.0) 256 (35.0) DeepSeek-R1-Distill-32B 471 (69.3) 209 (30.7)

LRM

Table 1: Reactive and proactive self-correction counts across models. For each generation, a GPT-5-based filter flags traces with a wrong path → correction → correct answer pattern; n is the number of such traces per model. Proactive correction is largely absent in LLMs and emerges consistently in LRMs. †Small sample (N < 50). Details can be found in Appendix D.2.

at most 35 of 4,800 generations (under 1%) and is overwhelmingly reactive (Table 1). Reactive correction is the dominant escape mechanism, but it only works when errors surface as overt local mistakes. The collapse modes of Figure 2 produce no such signal, so reactive correction has nothing to trigger on, and the trace silently diverges, ultimately leading to collapse.

#### 3.4 Proactive Correction in LRMs

LRMs exhibit a markedly different pattern. As shown in Table 1, proactive corrections account for 22–35% of all self-corrections. These corrections arise without any overt error: the model pauses, questions a prior step, and either verifies or backtracks based on internal doubt alone, as shown in Appendix I. Figure 3 situates this capability in the broader model landscape. Standard LLMs cluster along the bottom of the plot regardless of their direct fluency, whereas reasoning-tuned models occupy a distinct upper-right region where higher direct fluency is paired with a qualitatively new capacity for proactive correction.

Precision of proactive signals. Since proactive correction is preventive, the model may flag and recheck something that turns out to be fine. In other words, prevalence does not imply accuracy. Table 2 reports the precision of proactive signals, defined as the fraction of suspicion-triggered corrections issued on a trajectory that was in fact wrong. Across LRMs, precision averages only 24.4%: roughly three out of four proactive verbalizations second-

ProactiveCorrectionRate(%)

Base DeepSeek-R1-7B

- 0

- 1

- 2

- 3

- 4

- 5

Reasoning

DeepSeek-R1-32B

Qwen3-8B (inst)

Qwen3-14B (inst)

Qwen3-8B-Base

Qwen3-1.7B-Base gemma-3-4b-it

Mistral-7B-v0.3 Base

Qwen3-14B-Base Qwen2.5-32B

Qwen2.5-Math-7B Qwen2.5-7B

Llama-3.1-8B Base

0 20 40 60 80

Direct Fluency (%)

Figure 3: Capability map across model families. The x-axis reports direct fluency, the share of responses answered correctly without invoking any self-correction, computed as (correct − SC)/N. The y-axis reports the proactive correction rate, the share of responses containing a suspicion-based self-correction, i.e., a correction triggered without explicit evidence of error.

guess an already-correct chain. DeepSeek-R1 distillations are better calibrated (37.3%, 23.9%) than the Qwen3 series (20.5%, 15.8%), and precision does not improve with scale.

Model Precision (%)

DeepSeek-R1-Distill-Qwen-7B 37.3 DeepSeek-R1-Distill-Qwen-32B 23.9 Qwen3-8B 20.5 Qwen3-14B 15.8

Table 2: Precision of proactive self-correction signals across LRMs. Precision is the fraction of suspicionbased corrections issued on genuinely incorrect trajectories, as judged by GPT-5 (See Appendix D.2 for details). The remainder second-guess already-correct chains.

However, low precision does not undermine the role of proactive correction. The silent divergences in Figure 2 never produce an overt error, so reactive correction has no trigger to act on. In this regime, even a noisy extra signal is more useful than a precise reactive one that never fires. Proactive correction is what brings these hidden errors to the surface and lets the model recover, allowing strong reasoners to maintain trajectory quality where reactive correction cannot reach. We now term this mechanism epistemic verbalization.

### 4 Epistemic Verbalization

Definition. Epistemic verbalization is the linguistic channel through which the model explicitly externalizes uncertainty about its own trajectory, turning otherwise inert internal assessments into conditionable tokens that subsequent reasoning can act on. Reasoning traces of strong models contain utterances that do not advance procedural execu-

tion but instead express judgments of uncertainty or doubt about intermediate steps (Table 3).

# Example

- 1 Wait, is that correct?
- 2 Wait, 2023 is 7 multiplied by 17 squared, right?
- 3 Wait, maybe f(n) is related to the Möbius function but scaled differently.
- 4 Wait, perhaps I can write it as (f ∗ (n/d))(n) = 1, but that doesn’t seem helpful.

Table 3: Examples of epistemic verbalizations cooccurring with “Wait” in reasoning traces from DeepSeek-R1-Distill-Qwen-1.5B/7B/32B.

Let Zt denote a latent epistemic variable at step t, representing the model’s internal assessment of its trajectory’s reliability. As a latent quantity, Zt is informationally inert: although I(Y ;Zt | st−1) may be strictly positive, this information remains trapped inside the model’s internal states. Since autoregressive decoding conditions on the realized token sequence st−1, not on internal states, Zt cannot influence subsequent generation. Epistemic verbalization closes this gap by emitting tokens vt from which Zt is recoverable. Once vt ∈ st, we have

H(Y | st−1,vt) ≤ H(Y | st−1), (3)

with strict inequality whenever vt carries nontrivial information about Zt. Epistemic verbalization is thus the mechanism by which latent assessments become causally active in reasoning.

- 4.1 Injecting Epistemic Verbalization Recovers Failed Trajectories.

We empirically validate that epistemic verbalization supplies actionable information beyond procedural reasoning through an intervention study on failed reasoning trajectories.

We collect all incorrect rollouts from Qwen38B/14B-Base, which predominantly exhibit procedural reasoning (Section 3.2), on three benchmarks (AIME24, AMC23, MATH500) with 8 rollouts per problem. Since every rollout is incorrect, it must diverge from a valid path at some unknown point. We therefore sweep candidate truncation points: for each rollout, we truncate at relative position α ∈ {0.2,0.5,0.8,0.9} and resume generation from that point, optionally inserting a short epistemic verbalization.

We compare five conditions (Table 4). The NONE baseline performs no injection. The other

Condition Phrase NONE (no injection) HMM-DOUBT “Hmm, I’m not sure this is right.” HMM-DOUBTREVISIT

“Hmm, I’m not sure this is right. Let me reconsider.”

WAIT-DOUBT “Wait, is that correct?” WAIT-REVISIT “Wait, let me double-check.”

Table 4: Injection conditions. Four epistemic phrases span two phrase types (HMM, WAIT) and vary in whether they express doubt alone, signal an intent to revisit, or both. NONE controls for the effect of mere re-sampling from the truncated prefix.

four are epistemic phrases that vary along two axes. The first axis is phrase type (HMM vs. WAIT), and the second is function. DOUBT expresses uncertainty alone, REVISIT signals an intent to reexamine the reasoning, and DOUBT-REVISIT combines the two. Because it is unknown whether the truncation point occurs before or after the divergence point, the NONE condition isolates the effect of re-sampling alone. For each condition we draw n = 4 continuations and report the recovery rate: the fraction of originally incorrect rollouts whose continuation reaches the correct answer.

Qwen3-8B-Base AIME24

Qwen3-8B-Base AMC23

Qwen3-8B-Base MATH500

None (resample only)

"Hmm, let me reconsider"

"Wait, let me double-check"

"Hmm, not sure "

"Wait, is that correct?"

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.20

RecoveryRate

RecoveryRate

RecoveryRate

0.3

0.06

0.15

| |
|---|

0.04

0.2

| |
|---|

0.10

| |
|---|

0.02

| |
|---|

0.1

0.05

| |
|---|

| |
|---|

0.00

| |
|---|

| |
|---|

0.2 0.5 0.8 0.9

0.2 0.5 0.8 0.9

0.2 0.5 0.8 0.9

Truncation Fraction ( )

Truncation Fraction ( )

Truncation Fraction ( )

Qwen3-14B-Base AIME24

Qwen3-14B-Base AMC23

Qwen3-14B-Base MATH500

None (resample only)

"Hmm, let me reconsider"

"Wait, let me double-check"

"Hmm, not sure "

"Wait, is that correct?"

0.3

RecoveryRate

RecoveryRate

RecoveryRate

0.06

0.20

| |
|---|

| |
|---|

0.2

0.15

0.04

| |
|---|

0.10

0.02

| |
|---|

| |
|---|

0.1

| |
|---|

0.05

| |
|---|

0.00

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

0.2 0.5 0.8 0.9

0.2 0.5 0.8 0.9

0.2 0.5 0.8 0.9

Truncation Fraction ( )

Truncation Fraction ( )

Truncation Fraction ( )

Figure 4: Recovery rate vs. truncation fraction α across injection conditions and benchmarks, for Qwen3-8BBase (top) and Qwen3-14B-Base (bottom).

Figure 4 reports recovery rates across conditions, truncation positions, and benchmarks. Across all settings, simply expressing uncertainty, without pinpointing what went wrong, acts as an informative signal that drives recovery. Notably, even DOUBT, which conveys only uncertainty without any explicit cue to re-examine the reasoning (e.g., "Hmm, I’m not sure ..."), suffices for the model to detect its own errors and recover the correct answer. Recovery rates decline as α increases, but

the baseline decays fastest, dropping to near zero at α = 0.9, so the gap between injected cues and the baseline widens with α. Differences among phrasings are small, though "Wait"-style cues slightly outperform "Hmm"-style ones. Overall recovery tracks benchmark difficulty (MATH500 > AMC23 > AIME24), and the same pattern holds for both Qwen3-8B-Base and Qwen3-14B-Base, indicating robustness to model scale.

#### 4.2 Epistemic Tokens as Measurement Proxies

The linguistic realizations of uncertainty expression can be numerous and diverse, making full semantic analysis infeasible at scale. Still, surface tokens provide a useful observational handle on where verbalization is likely occurring.

To identify which surface tokens reliably cooccur with epistemic verbalization, we analyzed reasoning traces from four models (Qwen3-8/14B and DeepSeek-R1-Distill-Qwen-7B/32B) on math benchmarks and measured how frequently each candidate token appeared using GPT-5 as a judge. Details are provided in Appendix D.4. The resulting frequencies were: "wait" (73.0%), "maybe" (32.9%), "actually" (12.4%), "check" (10.5%), "hmm" (8.3%), "perhaps" (8.2%), "might" (6.6%), "seems" (3.3%), and "alternatively" (1.2%).

We therefore adopt these nine terms as epistemic tokens: "wait", "hmm", "perhaps", "maybe", "actually", "alternatively", "seems", "might", and "check" as practical indicators of regions where epistemic verbalization likely occurs.

5 A Unified Framework: Reasoning as Strategic Information Allocation

We now formalize the analysis of Sections 3–4 within a unified information-theoretic framework: reasoning under uncertainty operates along two informational axes, procedural advancement and epistemic verbalization, on which control actions such as self-correction subsequently act.

#### 5.1 Epistemic Verbalization Enables Continued Information Acquisition

Procedural reasoning operates by proceeding along the correct path or revising an incorrect one when explicit errors are detected. Epistemic verbalization, by contrast, exposes information about the reliability of the current trajectory regardless of whether any error has surfaced, thereby altering the model’s effective belief state. We formalize this

distinction by specifying when procedural reasoning alone becomes insufficient and showing that even sporadic epistemic verbalization suffices to restore convergence.

- Assumption 5.1 (Procedural Information is Trigger-Conditional). Suppose the procedural trajectory enters a diverged execution path at sub-task index k∗. For k > k∗, let Ek ∈ {0,1} indicate whether sub-task τk surfaces an overt error available to procedural revision, and let pE := P(Ek = 1 | Uk−1). There exists δproc > 0 such that

I(Y ;Uk | Uk−1) ≥ δproc only if Ek = 1.

Procedural information about Y thus accumulates only when overt errors surface along the diverged path. In the silent-divergence regime characterized in Section 3.2, the surface structure of stepby-step execution is preserved even as Pθ(Y | Uk) drifts from the correct Y , so pE ≈ 0 and procedural information acquisition stalls in expectation.

- Assumption 5.2 (Epistemic Informativeness). There exist δ > 0 and a threshold η > 0 such that, at every step t where an epistemic token is generated with H(Y | S˜t−1) > η,

H(Y | S˜t−1) − H(Y | S˜t) ≥ δ,

independently of whether an overt error has surfaced at the current sub-task.

Intuitively, whenever uncertainty about Y is still substantial, generating an epistemic token yields a non-trivial reduction in that uncertainty, regardless of whether the procedural channel has flagged an error. The key property is thus trigger-independence: epistemic verbalization can introduce information gain even when pE → 0. This is consistent with the intervention study in Section 4.1, where injecting a doubt cue alone, without pinpointing any error, meaningfully enabled the recovery of failed trajectories. In addition, we empirically demonstrate in Appendix F that epistemic verbalization introduces information by analyzing the mutual information between epistemic verbalization and Y .

Proposition 5.3 (Sporadic Epistemic Verbalization Suffices for Convergence). Under Assumption 5.2, if an epistemic token is generated with at least some non-zero probability ρ whenever H(Y | S˜t) > η, then E[H(Y | S˜t)] → 0 as t → ∞, regardless of the procedural trigger probability pE.

The formal statement and proof are deferred to Appendix A. This explains the empirical pattern

in Section 3. LLMs rely on reactive correction and thus fail in silent-divergence regimes where no procedural trigger arises, whereas LRMs sustain a non-trivial rate of epistemic verbalization that keeps belief contracting in the same regimes.

#### 5.2 Self-Correction as a Control Action

Building on the distinction between procedural and epistemic information, we now separate information from control. Self-correction is a control action that fires whenever an error surfaces in the reasoning chain. Epistemic verbalization does not itself perform correction. It is a conditionable informational mechanism that renders the model’s uncertainty at the token level, enabling the policy to control the reasoning trajectory accordingly.

Taken together, these results characterize reasoning as strategic information allocation under uncertainty: a process in which an LLM balances procedural and epistemic information acquisition and then performs appropriate control actions based on that information.

- 6 Experiments

#### 6.1 Test-Time Control of Epistemic Tokens

Following the previous finding that injecting epistemic verbalization meaningfully recovers failed trajectories (Section 4.1), we conversely manipulate the epistemic tokens defined in Section 4.2 at test time to further analyze their impact on reasoning performance. Specifically, we use DeepSeekR1-Distill-Qwen-14B/32B (Guo et al., 2025) and compare standard inference against inference in which epistemic tokens are suppressed.

AIME24 (DeepSeek-32B)

AIME24 (DeepSeek-14B)

AIME25 (DeepSeek-32B)

AIME25 (DeepSeek-14B)

80

60

Accuracy(%)

|68.33%| | | | |
|---|---|---|---|---|
| | |60.42%| | |
| | | | | |
|B|efore|o wit|After<br><br>f h| |

80

60 55.00%

50.42%

72.08%

70

50

45.83%

47.92%

70

50

64.58%

60

40

60

40

50

Before After

Before After

Before After

- Figure 5: Comparison avg@8 between baseline inference and inference epistemic tokens suppressed on DeepSeek-R1-Distill-Qwen-14B/32B.

As shown in Figure 5, masking epistemic tokens results in performance drops around 10%. While suppression substantially disrupts uncertainty expression, performance does not collapse entirely. Analysis of the reasoning traces (Appendix H.1) shows that the models sometimes route around the masked tokens by adopting alternative forms of epistemic verbalization.

#### 6.2 Suppressing Epistemic Verbalization via Distillation

To rule out such circumvention and isolate the effect of epistemic verbalization more cleanly, we design a controlled self-distillation SFT experiment. For each base model, we fine-tune it on 800 of its own reasoning traces generated with epistemic verbalization suppressed. Specifically, we append the instruction "Proceed directly without expressing uncertainty or doubt within the reasoning steps." to the prompt, collect 800 problems the model answers correctly under this prompt, and use the resulting traces as SFT data. This yields a fine-tuned variant whose training traces differ from the base model’s natural reasoning only in the absence of epistemic verbalization.

| |Base SFT (no EV)<br><br>|
|---|---|
|Qwen2.5-7B Qwen3-8B-Base Qwen3-14B-Base DeepSeek-R1-Distill-7B DeepSeek-R1-Distill-32B|13.3 6.7 16.7 3.3<br><br>16.7 10.0 50.0 30.0 80.0 43.3<br><br>|

Table 5: AIME24 pass@1 across base models and SFT variants trained on traces with epistemic verbalization (EV) suppressed.

As shown in Table 5, training on traces with epistemic verbalization suppressed consistently degrades performance, cutting accuracy by more than half in some cases despite the fact that all training traces contain correct answers. To better understand this degradation, we revisit the per-model capability map in Figure 3 that plots direct fluency against proactive correction rate. There, LRMs exhibit direct fluency comparable to that of LLMs but substantially higher proactive correction rates. Interestingly, suppressing epistemic verbalization through fine-tuning lowers their proactive correction rate, projecting the LRMs onto the LLM region of the map and bringing both their reasoning performance and behavior in line with those of LLMs. This further indicates that epistemic verbalization is not mere verbosity or stylistic decoration, but a substantive axis of reasoning capability.

#### 6.3 Impact of Epistemic Verbalization on Reasoning Distillation

We further examine the effects of distillation on a public dataset rich in epistemic verbalizations across several models, and find that epistemic verbalization is rapidly distilled and translates into

quick performance gains when certain conditions are met, where the degree of distributional alignment on epistemic tokens plays a critical role. We use the LIMO-v2 dataset (Ye et al., 2025), which was gathered from DeepSeek R1, DeepSeek-R1Distill-Qwen-32B (Guo et al., 2025), and QwQ32B (Yang et al., 2025a), and, as shown in Figure 6, contains a substantial number of epistemic tokens. Like our SFT setup in Section 6.2, LIMO consists of only 800 examples, making it far too small to instill mathematical knowledge and instead serving primarily to reshape the model’s linguistic habits.

Epistemic Token Frequency in the LIMO Dataset (Per Sample)

77.1

AvgCount

48.4 46.8

50

0

wait hmm perhaps maybe actuallyalternatively seems might check

- Figure 6: Per-sample epistemic token counts in LIMO. "Wait" alone appears 77 times per response on average.
- Figure 7 shows the AIME24 pass@1 scores of

various models trained on the same LIMO dataset. For Qwen2.5-7B and Qwen3-8B/14B-Base, performance rapidly improves by up to 2.6x with only 800 samples, indicating that epistemic verbalization is not a special model capability but rather a linguistic habit that is easily distilled. In contrast, other models exhibit substantial performance degradation despite being trained on the same dataset with similar initial accuracy and model size.

Qwen2.5-Math-7B

Qwen2.5-7B

DeepSeek-Math-7B -Instruct

100

100

pass@1

pass@1

Accuracy(%)

Accuracy(%)

pass@32

pass@32

80

80

100

Accuracy(%)

pass@1

56.7%

60

60

75

pass@32

53.3%

40

40

50

36.7% 26.7%

16.7%

20

20

25

13.3%

16.7%

3.3%

13.3%

0.0%

0.0%

0

0

0

Base LIMO

Base LIMO

Base LIMO

Qwen3-1.7B-Base

Qwen3-8B-Base

Qwen3-14B-Base

100

100

100

pass@1

pass@1

pass@1

86.7%

Accuracy(%)

Accuracy(%)

Accuracy(%)

80.0%

pass@32

pass@32

pass@32

80

80

80

60.0% 60.0%

60

60

60

46.7% 40.0%

30.0%

40

40

40

30.0%

16.7%

20

20

20

6.7%

16.7%

3.3%

0

0

0

Base LIMO

Base LIMO

Base LIMO

Figure 7: Comparison of AIME24 pass@1 scores between base models and models trained with SFT on the LIMO-v2 dataset.

To explain this discrepancy, we analyze distributional alignment between the student models and the dataset via token-level log probabilities, focusing on how students evaluate frequent epistemic tokens such as "Wait" and "Alternatively." The contrast between successful and failed distillation is clear: as shown in Figure 8, successful models exhibit well-aligned distributions in which epistemic

[Figure 4]

Figure 8: Distributions of token-level log-probability and entropy for all tokens versus the subset of "Wait" and "Alternative" tokens. Vertical blue and red lines indicate the dataset-level averages for "Wait" and "Alternative" tokens, respectively.

tokens fall within the model’s support, whereas poorly performing models show large gaps that place these tokens outside the support, hindering the adoption of the dataset’s epistemic verbalization and control. Notably, in well-performing models, epistemic tokens remain low-probability and high-entropy relative to other tokens.

Together, the dispersion in distillation effectiveness from a high-reasoning model can be attributed to whether the base model’s pre-existing characteristics are sufficiently "warmed up" to follow the high-reasoning model’s epistemic verbalization. When the base model can readily absorb this ability, performance improves rapidly with only a small dataset, regardless of model size.

### 7 Conclusion

In this work, we study effective reasoning in LLMs from an information-theoretic perspective, foregrounding the role of epistemic verbalization under uncertainty. We show that externalizing uncertainty enables continued information acquisition and selfcorrection, and reframe reasoning as a strategic allocation of information between procedural and epistemic axes. Empirically, we demonstrate that a minimal doubt cue suffices to recover failed trajectories, and that as few as 800 training samples can instill or suppress epistemic verbalization, suggesting it functions as a learnable linguistic habit rather than an intrinsic model capability. By distinguishing these two axes and their associated control actions, our framework clarifies otherwise elusive phenomena such as Aha moments and provides a general lens on contemporary LLM reasoning. We hope this perspective informs future theoretical work and inspires new approaches to modeling and controlling reasoning under uncertainty.

### Limitations

Our empirical analysis is mainly conducted on mathematical reasoning benchmarks, where correctness is objectively verifiable. Extending the analysis to other closed-world reasoning tasks and developing the formal world-Bayesian extension (Appendix B) for open-world settings such as toolaugmented or interactive agents are left for future work. As discussed in Appendix B, external observations in such settings can independently surface latent errors, potentially reducing the model’s reliance on epistemic verbalization. Empirically characterizing this trade-off is an interesting direction for future investigation.

Moreover, the nine epistemic tokens in Sections 4.2 and 6.1 serve as practical proxies and do not cover the full range of uncertainty expressions. While our core results do not rely on this specific set, a more thorough linguistic analysis of these proxies may be warranted. Finally, the classification of reactive and proactive correction, as well as the reasoning collapse analysis, depends on GPT-5 as an automated judge, introducing potential annotation noise.

### Acknowledgments

We sincerely thank Dohyung Kim and Sojeong Rhee for valuable discussions. This work was supported by Microsoft Research.

### References

Liv G d’Aliberti and Manoel Horta Ribeiro. 2026. The illusion of insight in reasoning models. arXiv preprint arXiv:2601.00514.

Shengyu Feng, Xiang Kong, Shuang Ma, Aonan Zhang, Dong Yin, Chong Wang, Ruoming Pang, and Yiming Yang. 2025. Step-by-step reasoning for math problems via twisted sequential monte carlo. In The Thirteenth International Conference on Learning Representations.

Zeyu Gan, Hao Yi, and Yong Liu. 2025. Cotspace: A theoretical framework for internal slowthinking via reinforcement learning. arXiv preprint arXiv:2509.04027.

Arthur Gretton, Kenji Fukumizu, Choon Teo, Le Song, Bernhard Schölkopf, and Alex Smola. 2007. A kernel statistical test of independence. Advances in neural information processing systems, 20.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.

Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training LLMs to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling.

Ryo Kamoi, Sarkar Snigdha Sarathi Das, Renze Lou, Jihyun Janice Ahn, Yilun Zhao, Xiaoxin Lu, Nan Zhang, Yusen Zhang, Haoran Ranran Zhang, Sujeeth Reddy Vummanthala, Salika Dave, Shaobo Qin, Arman Cohan, Wenpeng Yin, and Rui Zhang. 2024. Evaluating LLMs at detecting errors in LLM responses. In First Conference on Language Modeling.

Jeonghye Kim, Sojeong Rhee, Minbeom Kim, Dohyung Kim, Sangmook Lee, Youngchul Sung, and Kyomin Jung. 2025. Reflact: World-grounded decision making in llm agents via goal-state reflection. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 33421–33453.

Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. 2024. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025. Limr: Less is more for rl scaling. arXiv preprint arXiv:2502.11886.

Jessica E Liang. 2025. Chain-of-thought reasoning for math: Theoretical foundation and applications. In 2nd AI for Math Workshop@ ICML 2025.

Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. 2025. There may not be aha moment in r1-zero-like training — a pilot study. https://oatllm.notion.site/oat-zero. Notion Blog.

Haipeng Luo, Huawen Feng, Qingfeng Sun, Can Xu, Kai Zheng, Yufei Wang, Tao Yang, Han Hu, Yansong Tang, and Di Wang. 2025. Agentmath: Empowering mathematical reasoning for large language models via tool-augmented agent. arXiv preprint arXiv:2512.20745.

Minhae Oh, Jeonghye Kim, Nakyung Lee, Donggeon Seo, Taeuk Kim, and Jungwoo Lee. 2025. Raise: Enhancing scientific reasoning in llms via step-bystep retrieval. arXiv preprint arXiv:2506.08625.

Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. 2025. Demystifying reasoning dynamics with mutual information: Thinking tokens are information peaks in LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551.

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and JiRong Wen. 2025. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. arXiv preprint arXiv:2503.05592.

Jean-Francois Ton, Muhammad Faaiz Taufiq, and Yang Liu. 2025. Understanding chain-of-thought in LLMs through information theory. In Forty-second International Conference on Machine Learning.

Ken Tsui. 2025. Self-correction bench: Revealing and addressing the self-correction blind spot in llms. arXiv preprint arXiv:2507.02778.

Gladys Tyen, Hassan Mansoor, Victor C˘arbune, Yuanzhu Peter Chen, and Tony Mak. 2024. Llms cannot find reasoning errors, but can correct them given the error location. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13894–13908.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. 2025. Beyond the 80/20 rule: Highentropy minority tokens drive effective reinforcement learning for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022a. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022b. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, and 1 others. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. 2025b. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895.

Mutian Yang, Jiandong Gao, and Ji Wu. 2025c. Decoupling knowledge and reasoning in llms: An exploration using cognitive dual-system theory. arXiv preprint arXiv:2507.18178.

Shu Yang, Junchao Wu, Xin Chen, Yunze Xiao, Xinyi Yang, Derek F Wong, and Di Wang. 2025d. Understanding aha moments: from external observations to internal mechanisms. arXiv preprint arXiv:2504.02956.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. 2025. LIMO: Less is more for reasoning. In Second Conference on Language Modeling.

Xixian Yong, Xiao Zhou, Yingying Zhang, Jinlin Li, Yefeng Zheng, and Xian Wu. 2025. Think or not? exploring thinking efficiency in large reasoning models via an information-theoretic lens. arXiv preprint arXiv:2505.18237.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics.

### A Proof of Proposition 5.3

We first restate Proposition 5.3 in a formal form that makes the dependence on (ρ,δ,η) explicit, and then provide a proof.

Proposition A.1. Let Ht := H(Y | S˜t) and τη := inf{t ≥ 0 : Ht ≤ η}. Suppose Assumption 5.2 holds, and that at every step t with Ht−1 > η, an epistemic token is generated with probability at least ρ ∈ (0,1] (conditional on S˜t−1). Then

E[H0] − η ρδ

E[τη] ≤

,

regardless of the procedural trigger probability pE. If such pairs (ρ(η),δ(η)) exist for every η > 0, then E[Ht] → 0 as t → ∞.

Proof. Let Vt ∈ {0,1} indicate whether an epistemic token is generated at step t, and let ∆t := Ht−1 − Ht ≥ 0. By Assumption 5.2 and the lower bound P(Vt = 1 | S˜t−1) ≥ ρ, on the event {Ht−1 > η} we have

E[∆t | S˜t−1] ≥ δ · P(Vt = 1 | S˜t−1) ≥ ρδ,

which depends only on (ρ,δ) and not on pE. Telescoping up to τη and taking expectations,

E[H0] − η ≥ E[H0] − E[Hτη] ≥ ρδ · E[τη], which rearranges to the stated bound. If (ρ(η),δ(η)) exist for every η > 0, then E[τη] < ∞ for all η > 0, so E[Ht] → 0.

| |
|---|

### B World-Bayesian Reasoning with External Observations

We now extend the framework beyond the closedworld setting to reasoning processes that incorporate external observations. This includes embodied agents, tool-augmented language models, and interactive systems that acquire information from their environment during inference (Yao et al., 2022; Schick et al., 2023; Jin et al., 2025; Song et al., 2025; Kim et al., 2025; Luo et al., 2025).

In contrast to the closed-world, self-Bayesian setting, we consider world-Bayesian reasoning, in which the model observes additional environmental signals that are statistically informative about the

- target variable. As we show below, the availability of external observations fundamentally changes the role of epistemic verbalization: it is no longer the sole mechanism for escaping silent divergence, which in turn clarifies why epistemic verbalization is so critical in the closed-world regime studied in the main text. Setup. Let x denote the initial input and Y the
- target variable. At each reasoning step t, the agent may take an action at and receive an external observation ot ∈ O, where

ot ∼ P(ot | Y,at,st−1). The reasoning state now evolves as

st := (st−1,at,ot), s0 := x. Unlike internally generated tokens, observations ot constitute exogenous random variables that may introduce new information about Y . Each state st induces a predictive distribution Pθ(Y | st), which may now be interpreted as an approximate Bayesian posterior updated through both selfconditioning and external evidence.

Information gain from external observation. The information gain associated with an external observation is given by

IGext(st) = H(Y | st−1) − H(Y | st−1,ot)

= I(Y ; ot | st−1).

This quantity is strictly non-negative and may be positive even when procedural self-conditioning yields no additional information. In particular, a single informative tool call or environmental observation can resolve an ambiguity that would otherwise require extended internal deliberation.

Reduced dependence on epistemic verbalization. In the closed-world setting, the central challenge is that silent divergence produces no overt error, leaving epistemic verbalization as the only mechanism through which the model can regain traction (Proposition 5.3). The world-Bayesian setting relaxes this constraint. External observations can directly surface errors that remain latent under purely procedural reasoning. A tool call that returns a contradictory result, or an environmental signal that conflicts with the model’s current trajectory, provides an exogenous trigger for correction, one that does not depend on the model’s ability to verbalize its own uncertainty. In this regime, the burden of escaping silent divergence is shared between the epistemic channel and the external observation channel, reducing the model’s reliance on the former.

Residual role of epistemic verbalization. Despite its diminished necessity, epistemic verbalization retains a residual role in the world-Bayesian setting: it serves as an internal monitoring signal that can trigger the decision to seek external information. Recognizing that one’s current trajectory is uncertain is a prerequisite for deciding to query a tool, request clarification, or perform an experiment. In this sense, epistemic verbalization shifts from being the primary mechanism of uncertainty reduction to a secondary role as a dispatch signal that routes the policy toward external actions when internal reasoning is insufficient.

Relationship to the main framework. The closed-world framework arises as a special case when O = ∅. In this regime, all uncertainty reduction must come from internal belief transformation, making epistemic verbalization the sole escape route from silent divergence. The contrast with

the world-Bayesian setting highlights why epistemic verbalization plays such a disproportionate role in the closed-world regime: these LLMs lack external corrective signals entirely. Conversely, in tool-augmented or interactive systems, the performance gap between models with and without epistemic verbalization may narrow as the external observation channel becomes richer. Empirically characterizing this trade-off is an interesting direction for future work.

### C Limits of Internal Uncertainty in Procedural Reasoning.

A natural response to the limitations of procedural reasoning is to leverage model uncertainty as a corrective signal. Token-level entropy,

H(At | st−1) = −Ea′∼Pθ(·|st−1) log Pθ(a′ | st−1) .

(4) has been widely studied as a proxy for reasoning uncertainty (Yong et al., 2025; Yang et al., 2025b; d’Aliberti and Ribeiro, 2026). While token-level entropy can serve as a useful proxy for the model’s internal uncertainty, it captures local confidence over the next token rather than uncertainty about the target variable Y , and this gap limits its effectiveness as a corrective signal.

[Figure 5]

[Figure 6]

Figure 9: Token-level entropy over reasoning steps for Qwen2.5-Math-7B and Qwen3-14B-Base on AIME24 decreases similarly in both correct and incorrect solutions, suggesting that entropy alone does not reliably reflect uncertainty toward the correct answer.

Figure 9 illustrates a case where token-level entropy fails to distinguish correct from incorrect reasoning: the model becomes locally more confident at similar rates regardless of whether its trajectory is productive, suggesting that entropy alone may not suffice as a corrective signal.

### D Details of the Analysis

#### D.1 Reasoning Collapse Analysis

This appendix details the procedure behind the reasoning-collapse statistics in Section 3.2.

Pipeline. For each of the eight models analyzed in the main text, we sample N=8 traces per problem across AIME24/25, AMC23, and MATH500

(4,800 responses per model). Incorrect responses are passed to an LLM judge (GPT-5-2025-08-07) using the prompt below, which returns a JSON object indicating whether the trace exhibits collapse, its dominant type, the onset sentence, and a brief justification. The collapse rate is the fraction of incorrect responses flagged as collapsed.

Per-model statistics. Table 6 reports aggregate results on the Qwen family. Collapse accounts for 54–62% of all incorrect responses (58.4% overall). Scaling lowers the total error rate but leaves the share of collapse among remaining errors roughly unchanged, indicating that larger models reduce surface-level mistakes faster than silent divergence.

Model #Inc. Err. Rate #Col. Col./Inc.

Qwen2.5-7B 2,568 53.5% 1,452 56.5% Qwen2.5-32B 1,202 25.0% 717 59.7% Qwen3-1.7B-Base 2,486 51.8% 1,534 61.7% Qwen3-8B-Base 1,987 41.4% 1,091 54.9% Qwen3-14B-Base 1,401 29.2% 839 59.9%

Total 9,644 40.2% 5,633 58.4%

Table 6: Reasoning collapse statistics. Each model is evaluated on 600 problems with 8 sampled responses. "Inc." = incorrect responses, "Col." = collapsed responses.

Judge prompt. We used GPT-5 with the prompt below to detect reasoning collapse.

Prompt for Reasoning Collapse Detection [System Prompt]

You are an expert at analyzing mathematical reasoning traces. Your task is to detect reasoning collapse patterns in failed solution attempts.

###### [User Prompt]

Analyze the following INCORRECT math solution and determine if it exhibits a reasoning collapse pattern. Reasoning collapse includes ANY of the following:

- 1. REPETITION: Repeating the same sentences, phrases, or steps in a loop.
- 2. TOPIC DRIFT: Suddenly solving a completely different or made-up problem.
- 3. INCOHERENCE: Statements become logically disconnected or nonsensical mid-solution.
- 4. INFINITE LOOP: Getting stuck cycling through the same dead-end approach repeatedly without progress.
- 5. HALLUCINATION SPIRAL: Inventing fictitious theorems, formulas, or numeric facts and building on them.

Respond with a JSON object only: {

"has_collapse": true/false, "confidence": "high"/"medium"/"low", "collapse_type": "repetition"/"

topic_drift"/"incoherence"

/"infinite_loop"/" hallucination_spiral "/null,

"collapse_onset": "the sentence or phrase where collapse begins", "reasoning": "1-2 sentence explanation

of your judgment" }

Question: {question} Solution: {solution} Correct Answer: {gold_answer}

- D.2 Reactive and Proactive Correction Analysis

Next, we provide details of the reactive and proactive correction classification in Section 3.3.

Stage 1: Self-correction detection. To categorize all self-correction cases observed per model, we first apply the prompt to all correct-answer responses.

Prompt for Self-Correction Pattern Detection

[System Prompt] You are an expert at analyzing mathematical reasoning traces. Your task is to determine whether a solution demonstrates a specific self-correction pattern.

[User Prompt] Analyze the following math solution and determine if it exhibits the self-correction pattern. The self-correction pattern requires ALL of the following:

- 1. The solver initially pursues a WRONG approach, method, or makes a significant mistake.
- 2. The solver RECOGNIZES the error.
- 3. The solver ABANDONS the wrong path and takes a different approach.
- 4. The solver arrives at the CORRECT final answer. Respond with a JSON object only:

{

"has_self_correction": true/false, "confidence": "high"/"medium"/"low", "wrong_path_description": "brief

description of the initial wrong approach",

"correction_trigger": "the exact phrase/ sentence where the solver recognizes the error",

"reasoning": "1-2 sentence explanation

of your judgment" }

Question: {question} Solution: {solution}

Correct Answer: {gold_answer}

Stage 2: Trigger classification (reactive vs. proactive). We then pass all self-correction cases identified above to the prompt below to classify each correction as reactive (evidence-based) or proactive (suspicion-based).

Prompt for Self-Correction Trigger Classification [System Prompt] You are an expert at analyzing mathematical reasoning traces.

[User Prompt] A language model was solving a math problem and performed self-correction. Classify what triggered the self-correction into one of two types:

evidence_based: The model encountered a clear, explicit signal that something was wrong BEFORE deciding to correct. Examples:

- • A mathematical result that is impossible (negative angle, sin > 1, division by zero).
- • A result that explicitly violates a stated condition (e.g., x = 6 but problem says x > 7).
- • A logical contradiction derived from the work (e.g., 6 = 3, empty solution set, negative count).
- • An exhausted search that found nothing (tried all cases, none worked).

suspicion_based: The model decided to re-examine WITHOUT a clear signal of error. Examples:

- • Vague unease (“let me reconsider”, “I’m not sure”, “let me double-check”).
- • The model just tried a different approach without identifying what was wrong.
- • Re-examination triggered by aesthetic/intuitive concern rather than a concrete contradiction.

Respond with a JSON object only: {

"trigger_type": "evidence_based" or " suspicion_based", "confidence": "high" or "medium" or " low",

"evidence": "the specific contradiction/ signal that triggered correction ( if evidence_based), or null",

"reasoning": "1 sentence explanation" }

D.3 Analysis of the Precision of Proactive Signals

To analyze the precision of proactive signals in Section 3.4, we additionally sample 80 responses per model (320 total across the four LRMs) that contain at least one proactive-pattern verbalization, and verify via GPT-5 whether the model was actually on a wrong path at that point using the prompt below.

Prompt for Wrong-Path Verification at Doubt Point [System Prompt] You are an expert at analyzing LLM mathematical reasoning traces.

[User Prompt] A language model was solving a math problem. At the marked point [DOUBT], it expressed uncertainty or suspicion without an explicit error signal.

Was the model actually on a WRONG reasoning path just before [DOUBT]? Or was it on a CORRECT path expressing unnecessary doubt?

Question: {question} Gold answer: {gold_answer}

###### Reasoning excerpt (ends at [DOUBT]):

""" {excerpt} [DOUBT: "{trigger}"] """

Respond with JSON only: {

"was_on_wrong_path": true | false, "confidence": "high" | "medium" | "low",

"reasoning": "<one sentence>" }

#### D.4 Identifying Epistemic Tokens

To identify epistemic tokens in Section 4.2, we apply the characterization prompt below to all proactive (suspicion-based) self-correction cases identified in Appendix D.2 for the four LRMs that exhibit meaningful proactive correction rates. GPT-5 is asked to (i) distill the minimal trigger tokens, (ii) assign a free-form verbalization label, and (iii) describe the epistemic function expressed.

Prompt for Suspicion Verbalization Characterization [System Prompt] You are an expert at analyzing LLM reasoning traces. Your task is to characterize the type of verbalization a language model uses when it expresses uncertainty or doubt without any clear evidence of error — no mathematical contradiction, no violated constraint, just vague unease.

[User Prompt] A language model was solving a math problem. At some point it expressed uncertainty or suspicion — without any concrete evidence that something was wrong — using the following phrase. Your tasks:

- 1. Extract the minimal “trigger tokens” — the 1–5 word core phrase that most directly expresses the uncertainty (e.g. “wait, maybe”, “hmm”, “i’m not sure”).
- 2. Give this verbalization a short, descriptive label

of your own choosing (2–4 words, snake_case). Do NOT use any predefined list — describe what you actually observe.

3. In one sentence, describe what internal state the

model is expressing through this verbalization. Question context: {question} Prior reasoning path: {wrong_path} Verbalization phrase: “{trigger}” Surrounding context: {reasoning} Respond with JSON only: {

"trigger_tokens": "<1-5 word core phrase, lowercased>", "verbalization_label": "<your 2-4 word descriptive label in snake_case>",

"epistemic_function": "<one sentence describing what the model is expressing>",

"confidence": "high" | "medium" | "low" }

### E Details of the Experimental Setup

Training hyperparameters. For all distillation experiments, all models are fine-tuned using LLaMA-Factory (Zheng et al., 2024) under the default LIMO configuration (Ye et al., 2025). We utilize four B200 GPUs for training.

Table 7: AIME24 pass@1 performance across various base models with varying temperature and top-p configurations.

|Temperature (Top-P)|0.0 0.7 (1.0)<br><br>0.7 (0.8)|
|---|---|
|DeepSeek-R1-Distill-32B DeepSeek-R1-Distill-14B DeepSeek-R1-Distill-1.5B<br><br>|50.0 70.0 80.0 56.67 56.67 70.0 20.0 30.0 30.0|
|DeepSeek-Math-7B-Instruct<br><br>|3.33 0.0 0.0|
|Qwen2.5-Math-7B Qwen2.5-7B|16.67 10.0 6.67 13.33 6.67 3.33<br><br>|
|Qwen3-1.7B-Base Qwen3-4B-Base Qwen3-8B-Base Qwen3-14B-Base|0.0 0.0 6.67 6.67 13.33 6.67 13.33 16.67 10.0 16.67 10.0 13.33<br><br>|

Selection of decoding hyperparameters. We choose the decoding temperature mainly as 0.0 for calculating pass@1 and 0.7 for calculating pass@k or acc@16. However, there are some concerns, especially for recent models that exhibit more explicit reasoning behavior, such as DeepSeek or Qwen3, that a temperature of 0.0 may lead to performance degradation. Since we could not find detailed guidance for the Qwen3-base model, we additionally conduct an ablation study with varying temper-

atures to verify this effect when computing the pass@1 score.

As shown in Table 7, a temperature of 0.0 is harmful for the DeepSeek-Distill models, while it is beneficial for the Qwen2.5 models. For the Qwen3Base models, the effect varies across settings. In particular, for the DeepSeek-Distill models, setting Top-P to 0.8 provides a substantial performance gain. Following the results in Table 7, we report the pass@1 performance of the base model using the larger value between temperatures 0.0 and 0.7, while the LIMO pass@1 performance is always reported with the temperature fixed at 0.0.

### F Epistemic Verbalization Produces Information Gain

Qian et al. (2025) observe that, in LRM reasoning, most steps carry little mutual information (MI) with the correct answer, while a small number of steps exhibit sharp MI increases (“MI peaks”) that frequently coincide with thinking tokens such as “Wait”. In this section, we extend their analysis to examine whether epistemic verbalization indeed induces information gain.

Setup. Building on the official implementation of Qian et al. (2025) (https://github.com/ChnQ/ MI-Peaks), we measure the statistical dependence between the hidden representation at each token position and the ground-truth answer representation. Specifically, for each token position t in the model-generated reasoning trajectory, we extract the last-layer hidden state ht and compute its mutual information with the final-token representation of the ground-truth answer, hGT, using an HSICbased estimator (Gretton et al., 2007). This enables token-level tracking of when and how answerrelevant information emerges in the model’s internal representations during reasoning.

Results. To examine whether epistemic verbalization drives information gain, we compare two models from the same family (Qwen3-8B-Base) on the same problem: one that recovers the correct answer through self-correction, and one that fails to recover. Since epistemic verbalization and selfcorrection are rarely observed in Qwen3-Base, we additionally include Qwen3-8B-SFT, fine-tuned from the same base on reasoning traces rich in epistemic verbalization (Ye et al., 2025) (as shown in Section 6.3), to elicit such behavior at inference time. On AIME24 #7, both models initially commit

to an incorrect trajectory, but only Qwen3-8B-SFT recovers via self-correction, while Qwen3-Base remains on the incorrect path (trajectories in Appendix H.2).

[Figure 7]

- Figure 10: On AIME24 #7, both models initially fail, but only Qwen3-8B-SFT sustains information gain through epistemic verbalization and recovers the correct answer.

As shown in Figure 10, Qwen3-8B-Base’s MI collapses to near zero shortly after divergence, whereas Qwen3-8B-SFT sustains elevated MI while producing evaluative expressions such as “Wait, let me check.” A closer inspection of highMI regions reveals that MI tends to rise at thinking tokens, but not always. Instead, elevated MI appears in utterances that perform epistemic verbalization, and thinking tokens occurring without such verbalization (e.g., "Hmm" in the left panel of Figure 11) do not trigger an MI increase.

[Figure 8]

- Figure 11: Token-level MI analysis: high MI aligns with evaluative behavior, not with the thinking tokens themselves.

Aligning with our framework and the preceding discussion, these tokens are surface manifestations of a deeper mechanism. They externalize uncertainty, rendering the model’s epistemic state conditionable and reusable during inference.

### G More Relationship Between Uncertainty and Epistemic Verbalization

We further investigate whether the uncertainty expressed during reasoning truly reflects the model’s underlying uncertainty. Leveraging the observation that more challenging problems tend to elicit greater uncertainty, we analyze how uncertainty is verbalized in the outputs of strong reasoning

models that exhibit Aha-moment or self-reflective behaviors.

DeepSeek-R1-Distill-Qwen. We first study DeepSeek-R1-Distill-Qwen models ranging from 1.5B to 14B parameters, comparing their performance and response length.

Acc@16

Len@16

AIME24 AIME25 MATH AMC

|| |
|---|
<br><br>5<br><br>10<br><br>15<br><br>Length(k)|| |
|---|
<br><br>|
|---|---|
| || |
|---|
<br><br>|
| | |
| | |

80

Length(k)

Score(%)

60

40

| |
|---|

20

1.5B 7B 14B

1.5B 7B 14B

Model Size

Model Size

- Figure 12: Acc@16 (avg. score) and Len@16 (avg. response length) of DeepSeek-Distill 1.5B–14B on math benchmarks.

As shown in Figure 12, more difficult problems (AIME24/25) elicit longer responses than easier ones (AMC23/MATH500). Increasing model size is associated with higher scores and shorter responses.

We further examined how frequently DeepSeekR1-Distill-Qwen-{1.5B, 7B, 14B} generate epistemic tokens during solution generation. As shown in Figure 12, smaller models use these tokens more frequently: relative to the 14B model, the 1.5B model produces 75% more occurrences of "Wait" and 235% more of "Perhaps" in AIME24. This suggests that when reasoning models face problems beyond their capacity, they verbalize epistemic uncertainty more often. We also observe that epistemic tokens appear more frequently on harder benchmarks (AIME24/25) and less frequently on easier ones (AMC23/MATH500), indicating that these tokens indeed reflect the uncertainty the model experiences during reasoning.

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

0

25

50

75

100

125

150

CountperProblem

+75%

+234%

Epistemic Token Frequency by Model Size

Model Size

14B

7B

1.5B

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

0

25

50

75

100

125

150

CountperProblem

+63%

+141%

Epistemic Token Frequency by Model Size

Model Size 14B

| |
|---|

7B

1.5B

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

0

20

40

60

80

CountperProblem

+229%

+332%

Epistemic Token Frequency by Model Size

Model Size 14B

| |
|---|

7B 1.5B

| |
|---|

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

0

10

20

30

40

CountperProblem

+169%

+280%

Epistemic Token Frequency by Model Size

Model Size 14B 7B 1.5B

| |
|---|

| |
|---|

| |
|---|

- Figure 13: Token occurrence counts for DeepSeek-R1Distill-Qwen-{1.5B, 7B, 14B} models on the AIME24, AIME25, AMC23, and MATH500 benchmarks.

Qwen3-Base LIMO. Additionally, we investigated whether distilling a base model using a teacher model that heavily incorporates epistemic verbalization leads to increased generation of epistemic verbalizations depending on model size and problem difficulty. We observe that, in these distilled models as well, smaller models tend to generate more epistemic tokens as their uncertainty about a given problem increases.

Interestingly, unlike the analysis on DeepSeekR1-Distill-Qwen models, where higher uncertainty substantially increased the use of the perhaps token, these models do not show a marked increase in perhaps. Instead, they primarily rely on the "Wait" token. As shown in Figure 6, this may be because the LIMO dataset contains relatively few instances of perhaps. This suggests that patterns of uncertainty verbalization can shift depending on the training data distribution.

###### AIME24

###### AIME25

250

400

Model Size

Model Size

CountperProblem

CountperProblem

14B 8B 4B

14B 8B 4B

200

300

| |
|---|

| |
|---|

150

200

| |
|---|

100

100

50

0

0

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

###### AMC

###### MATH500

200

Model Size 14B 8B 4B

Model Size 14B 8B 4B

CountperProblem

CountperProblem

60

| |
|---|

| |
|---|

150

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|

100

20

50

0

0

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

waitmaybeactuallycheckhmmperhapsmightseemsalternatively

Figure 14: Token occurrence counts for Qwen3-{4B, 8B, 14B}-Base models on the AIME24, AIME25, AMC23, and MATH500 benchmarks.

### H Quantitative Analysis

#### H.1 Bypass Pathways under Epistemic Token Suppression

In Section 6.1, we suppress nine epistemic tokens (wait, hmm, perhaps, maybe, actually, alternatively, seems, might, check) via logit bias −100 and observe performance drops of around 10%. While suppression substantially disrupts uncertainty expression, performance does not collapse entirely. Analysis of the reasoning traces reveals that the models sometimes route around the masked tokens by substituting equivalents outside the banned vocabulary, such as hold on, no, Let me, or a paragraph break. Table 8 lists the observed substitutions. In these cases, the underlying doubt-andverify pattern is preserved, and only its lexical surface is rerouted.

Baseline surface form

Suppression substitute

Wait, let me check... But hold on, let me... Hmm, No, that’s not quite right / Hold

on, no Maybe / Perhaps It’s possible that... / Suppose... Alternatively Another way to see this Actually I realize / On closer look Let me check Let me verify

- Table 8: Bypass patterns observed when the epistemic tokens are suppressed during generation.

#### H.2 Comparison Between Base and LIMO Distillation

Qwen3-8B-Base AIME24 #7 Let’s solve the problem step by step. Given:

- • Let B be the set of rectangular boxes with surface area 54 and volume 23.
- • Let r be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of B.
- • We need to find r2 in the form pq, where p and q are relatively prime positive integers, and then compute p + q.

- Step 1: Define the Variables Let the dimensions of the rectangular box be a, b, and c.
- Step 2: Express the Given Conditions

- 1. Surface Area: 2(ab + bc + ca) = 54 =⇒ ab + bc + ca = 27
- 2. Volume: abc = 23

......

- Step 5: Use the AM-GM Inequality To minimize a2 + b2 + c2, we can use the AM-GM inequality. The minimum occurs when a = b = c. Let a = b = c. Then: 3a2 = 27 =⇒ a2 = 9 =⇒ a = 3 a3 = 23 =⇒ 33 = 27 ̸= 23 This is not possible, so the minimum does not occur when a = b = c.

- Step 6: Use the Method of Lagrange Multipliers To find the minimum of a2 + b2 + c2 subject to the constraints ab+bc+ca = 27 and abc = 23, we can use the method of Lagrange multipliers. However, this method is complex for this problem. Instead, we can use the following approach.
- Step 7: Express a2+b2+c2 in Terms of ab+bc+ca and abc We know that: a2 +b2 +c2 = (a+b+c)2 −2(ab+ bc + ca) But we don’t have a + b + c. Instead, we can use the identity: a2+b2+c2 = (a+b+c)2−2(ab+bc+ca) However, without a + b + c, we need another approach.

- Step 8: Use the Identity for a2 + b2 + c2 We can use the identity: a2 + b2 + c2 = (a + b + c)2 − 2(ab + bc + ca) But we don’t have a + b + c. Instead, we can use the identity: a2 + b2 + c2 = (a + b + c)2 − 2(ab + bc + ca) However, without a + b + c, we need another approach.

......

- Step 244: Use the Identity for a2 + b2 + c2 We can use the identity: a2 + b2 + c2 = (a + b + c)2 − 2(ab + bc + ca) But we don’t have a + b + c. Instead, we can use the identity: a2+b2+c2 = (a+b+c)2−2(ab+bc+ca) However, without a + b + c, we need another approach.

- Step 245: Use the Identity for a2 + b2 + c2 We can use the identity: a2 + b2 + c2 = (a + b + c)2 − 2(ab + bc + ca) But we don’t have a + b + c. Instead, we can use the identity: a2+b2+c2 = (a+b+c)2−2(ab+bc+ca)

...

LIMO Qwen3-8B-Base AIME24 #7

Okay, so I need to find the smallest sphere that can contain any rectangular box from the set B, which has surface area 54 and volume 23. Then, I have to find r2 where r is the radius of that sphere, express it

as a fraction pq in lowest terms, and compute p + q.

First, let me recall that the smallest sphere that can contain a rectangular box is the one whose diameter is equal to the space diagonal of the box. The space diagonal of a rectangular box with dimensions a, b, and c is √a2 + b2 + c2. Therefore, the diameter of the sphere is equal to this space diagonal, so the radius r would be half of that, right?

So, r = 12√a2 + b2 + c2. Therefore, r2 would be a2+b2+c2

4 .

Therefore, to find the minimal r2, I need to minimize a2 + b2 + c2 over all possible boxes with surface area 54 and volume 23. So the problem reduces to minimizing a2+b2+c2 given that ab+bc+ca = 27 (since surface area is 2(ab + bc + ca) = 54) and abc = 23 (since volume is abc = 23).

Hmm, so I have to minimize a2 + b2 + c2 given ab + bc + ca = 27 and abc = 23.

I remember that in optimization problems with constraints, Lagrange multipliers might be helpful, but since these are symmetric constraints, maybe I can use some inequalities here. Alternatively, since it’s about rectangular boxes, maybe the minimal space diagonal occurs when the box is a cube? Wait, but maybe not necessarily a cube, but perhaps when the dimensions are in some proportion?

√4+92

Solving quadratic equation: a = −2±

2 = −2±

2 = −1 ± 2√6.

√96

√6

2 = −2±4

Since a is a length, it must be positive. So, possible solutions are a = 2, or a = −1 + 2√6 ≈ 3.899, or a = −1 − 2√6 which is negative, so discard that.

So, if a = 2, then c = 23/4 = 5.75. Check if this satisfies the first equation: a2 + 2ac = 4 + 2 · 2 · 5.75 = 4 + 23 = 27. Yes, that works. So, in this case, the dimensions are a = b = 2, c = 5.75.

......

Now, let’s compute a2+b2+c2 = 4+4+(23/4)2 = 8 + 529/16 = (128 + 529)/16 = 657/16 ≈ 41.0625.

Alternatively, if we take the other root a = −1 + 2√6 ≈ 3.899, then c = 23/a2 ≈ 1.513. Then, let’s compute a2 +b2 +c2 ≈ 15.204+15.204+2.289 ≈ 32.697. Hmm, which is larger than 657/16... wait, 32.697 is smaller than 41.0625.

Wait, wait a second! I think I made a critical mistake here.

The problem asks for the smallest sphere that can contain each of the rectangular boxes that are elements of B. Since B is the set of all boxes with surface area 54 and volume 23, the minimal sphere must work for all such boxes, so we need the maximum possible space diagonal among all such boxes, then the radius is half that. Therefore, actually, we need to maximize the space diagonal, not minimize it.

Wait, so my earlier approach was incorrect. I thought we needed to minimize the space diagonal, but actually, since the sphere must contain all boxes, the radius must be at least half the maximum space diagonal of any box in B. Therefore, the minimal such sphere’s radius r is determined by the maximum of a2 + b2 + c2.

Using (a+b+c)2 = a2+b2+c2+2(ab+bc+ca) = (a2 + b2 + c2) + 54. So, to maximize a2 + b2 + c2, we must maximize a + b + c.

From my symmetric cases analysis: Case 1 (a = 2): a+b+c = 2+2+5.75 = 9.75. Case 2 (a ≈ 3.899): a + b + c ≈ 3.899 + 3.899 + 1.513 = 9.311.

The maximum occurs at a = 2, b = 2, c = 5.75. Max a2 + b2 + c2 = 657/16.

Therefore, the minimal radius r of the sphere that can contain any box in B satisfies:

r2 = 41 · max(a2 + b2 + c2) = 14 · 65716 = 65764 .

Checking if 65764 is in lowest terms. 657 = 32 · 73 and 64 = 26. No common factors. p = 657, q = 64. p + q = 657 + 64 = 721.

### I Examples of Reactive and Proactive Correction

Table 9 lists representative examples of the two correction modes discussed in Sections 3.3 and 3.4, pairing each model’s preceding computation with the verbalization that triggers correction. Reactive corrections, observed in both LLMs and LRMs, are triggered by an explicit failure in the preceding computation (e.g., a violated constraint, a contradiction between derivations, or a hallucinated runtime error). Proactive corrections, observed only in LRMs, arise without any such surfaced error.

### J Use of AI Assistants

We used Claude and ChatGPT to polish and refine text drafted by the authors, to create Figure 1, and to improve the design of plots. As described in the main text, LLMs including GPT-5 were used as automated judges and evaluation tools as part of the experimental pipeline. No AI assistants were used to generate research ideas or design experiments.

Model Preceding computation Trigger verbalization LLM — reactive correction only Qwen2.5-32B-Instruct Diagonal midpoint equation yields x = 6,

“However, we are given that x > 7. Therefore, we need to re-evaluate the vertices.”

which explicitly violates the given constraint x > 7.

Qwen2.5-7B Substitution gives y = 25z z=5/2 = 1, violating the condition y > 1.

“We made an error in the interpretation of the equations.”

Qwen2.5-Math-7B solve(...) raises TypeError: list indices must be integers or slices, not Symbol.

“It seems there was an issue with how the solution was being accessed.”

LRM — reactive correction

2025 − 4225 286

= −2200 286

DeepSeek-R1-DistillQwen-32B

< 0 “Wait, negative length? That can’t be.”

AP =

√2 9 ; coordinate

DeepSeek-R1-DistillQwen-32B

Cayley–Menger: V = 20

“That contradicts the earlier volume ... So, clearly, something is wrong.”

method: V = 1603

Qwen3-14B Numerical evaluation shows f(2) ≈ 4.16, f(5) ≈ 4.83, f(10) ≈ 4.61: non-monotone, contradicting the earlier claim of monotonic increase.

“Wait, so it increases up to some point and then decreases? That contradicts my previous thought.”

LRM — proactive correction

DeepSeek-R1-DistillQwen-32B

Serially recalling symmedian length formulas. (each is immediately replaced by another with no computation performed.)

“Wait, perhaps I should look for a different approach.”

“Wait, maybe I need to think again.”

Qwen3-14B Game-tree analysis of n = 2 is proceeding correctly. (no error has surfaced).

DeepSeek-R1-DistillQwen-32B

Correctly identifying n = 2 as a losing position for the moving player. (no contradiction present).

“Wait, maybe I’m getting confused.”

- Table 9: Representative examples of reactive and proactive correction in LLMs and LRMs. LLMs exhibit predominantly reactive correction, triggered by explicit computational failures or contradiction. LRMs additionally exhibit proactive correction, where doubt arises without any surfaced error.

