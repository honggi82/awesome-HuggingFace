# arXiv:2605.13301v1[cs.AI]13May2026

## Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling

Yafu Li1,2∗ , Runzhe Zhan1∗, Haoran Zhang1,4∗, Shunkai Zhang1,5∗, Yizhuo Li1∗, Zhilin Wang1, Jiacheng Chen2, Futing Wang1, Xuyang Hu1, Yuchen Fan1, Bangjie Xu3, Yucheng Su3, Xinmiao Han3, Chenxi Li1, Haodi Lei1, Yufeng Zhao1, Zejin Lin3, Qianjia Cheng1, Tong Zhu1, Xiaoye Qu1, Ganqu Cui1, Peng Ye1 , Yun Luo1 , Zhouchen Lin5, Yu Qiao1, Bowen Zhou1,3 , Ning Ding3,1 , Yu Cheng2,1

1Shanghai AI Laboratory 2The Chinese University of Hong Kong 3Tsinghua University 4Shanghai Jiao Tong University 5Peking University

ABSTRACT

Recent progress in reasoning models has substantially advanced long-horizon mathematical and scientific problem solving, with several systems now reaching gold-medal-level performance on International Mathematical Olympiad (IMO) and International Physics Olympiad (IPhO) problems. In this paper, we introduce a simple and unified recipe for converting a post-trained reasoning backbone into a rigorous olympiad-level solver. The recipe first uses a reverse-perplexity curriculum for SFT to instill rigorous proof-search and self-checking behaviors, then scales these behaviors through a two-stage RL pipeline that progresses from RL with verifiable rewards to more delicate proof-level RL, and finally boosts solving performance with test-time scaling. Applying this recipe, we train a 30BA3B backbone with SFT on around 340K sub-8K-token trajectories followed by 200 RL steps. The resulting model, SU-01, supports stable reasoning on difficult problems with trajectories exceeding 100K tokens, while achieving gold-medallevel performance on mathematical and physical olympiad competitions, including IMO 2025/USAMO 2026 and IPhO 2024/2025. It also demonstrates strong generalization of scientific reasoning to domains beyond mathematics and physics.

Project Page Code Models

GPT-5.5-High

80.7

80.5

DeepSeekMath-V2 (Heavy)

Gemini-3.1-Pro

72.6

SU-01

70.2

60.7

Gemini-2.5-DeepThink

SU-01

57.6

DeepSeek-V3.2-Speciale

56.0

Nemotron-Cascade-2

52.9

SFT + Coarse RL

51.0

Gemini-2.5-Pro

47.2

GPT-5

39.5

Gemini-2.5-Pro

36.4

SFT

36.2

GLM-4.7-Flash

33.8

Gemma-4-31B

31.4

Qwen3.6-35B-A3B

23.1

P1-30B-A3B

20.0

0 10 20 30 40 50 60 70 80

IMO-ProofBench (%)

Figure 1: Overall comparison on IMO-ProofBench. ⋆ denotes results reported in the original paper, and † denotes results with test-time scaling. Blue bars trace the evolution of our pipeline from the 30B-A3B backbone through rigorous SFT (Section 2), coarse RL, refined RL (Section 3), and test-time scaling (Section 4), culminating in gold-medal-level olympiad reasoning (Section 5).

∗ Core contributors. Yafu Li is the project lead.

Corresponding authors. Contact: yafuly@gmail.com and chengyu@cse.cuhk.edu.hk.

#### 1 Introduction

Olympiad competitions provide one of the clearest stress tests for long-horizon reasoning. Unlike many standard benchmarks, these problems require a model to search over many possible solution paths, control assumptions precisely, verify intermediate claims, and present a final argument that can survive strict grading across mathematical and scientific settings. Recent systems have made rapid progress in this direction: AlphaGeometry combined neural guidance with symbolic search for olympiad geometry (Trinh et al., 2024), while AlphaProof, AlphaGeometry 2, and Gemini Deep Think reached silver- or gold-medal standards on International Mathematical Olympiad problems with larger search and verification budgets (Google DeepMind, 2024; 2025). At the same time, general reasoning models have improved through chain-of-thought prompting, math-specialized post-training, and reinforcement learning with verifiable rewards (Wei et al., 2022; Shao et al., 2024; Yang et al., 2024; Guo et al., 2025; Yan et al., 2025; Zhan et al., 2025), while scientific olympiad benchmarks test transfer to modeling, derivation, and competition-style justification (He et al., 2024; Chen et al., 2025; Luo et al., 2026).

A central question is therefore whether a reasoning backbone can be pushed to olympiad-level performance with a compact, domain-unified recipe that applies the same reasoning-centric pipeline across mathematical and scientific problems. Using a 30B-A3B model, we build a modular pipeline: SFT reshapes reasoning behavior, RL scales solving capability, and TTS allocates additional inference compute to the hardest proof-search problems. Together, these stages align behavior shaping, reward design, experience replay, and self-verification into a compact recipe for rigorous mathematical and scientific reasoning. The desgin follows a specializable-generalist view: rather than building a narrow olympiad solver, we specialize a broadly capable post-trained model toward expert-level proof reasoning while preserving transfer across scientific domains.

The first stage aims to instill a more disciplined proof-search pattern. Starting from a post-trained model that is already competitive on scientific reasoning tasks, we curate long-form solution, selfverification, and self-refinement trajectories from mathematical, scientific, coding, and instructionfollowing sources. After filtering, the SFT mixture contains 338K trajectories with responses shorter than 8K tokens. SFT on this rigorous proof data instills reasoning behaviors centered on proof search, self-checking, and repair. We then order the examples by reverse perplexity so that each pass starts with trajectories most mismatched to the initial policy before consolidating on more familiar examples. This curriculum helps preserve and recover the capability of the post-trained model with its reasoning behavior reshaped.

The second stage scales this behavior through two levels of RL. Coarse RL uses verifiable prompts and efficient outcome checking to scale the reasoning behaviors introduced by SFT under reliable binary rewards, following the broader RLVR paradigm for efficient reasoning improvement (Guo et al., 2025; Shao et al., 2024). Refined RL then shifts the target from answer correctness to proof quality. It combines a proof-level generative reward model for scoring complete proofs, selfrefinement prompts for training critique-and-repair behavior, and experience replay for preserving rare successful trajectories on hard problems. Finally, we apply test-time scaling through a selfverification-and-refinement loop to elevate the trained model to olympiad-level reasoning (Huang & Yang, 2025).

On answer-verifiable benchmarks, the resulting model, SU-01, nearly matches the strongest similarsize baseline, Qwen3.6-35B-A3B, across AnswerBench, AMO-Bench, AIME 2025/2026, and FrontierScience-Olympiad. On proof-oriented evaluation, SU-01 reaches 57.6% on IMO-ProofBench with direct generation and 70.2% with TTS, substantially outperforming similar-size models and approaching competitive commercial systems such as Gemini 3.1 Pro Thinking. Beyond solving competition problems, SU-01 obtains the best similar-size overall score on FrontierScience-Research, suggesting that the recipe generalizes scientific reasoning toward research-style problems beyond olympiad benchmarks.

On official competition problems, SU-01 shows strong end-to-end reasoning beyond benchmark-style evaluation. Direct SU-01 already exceeds the IPhO gold lines for both 2024 and 2025, and clears the bronze-medal lines on IMO 2025 and USAMO 2026. With test-time scaling, it reaches 35 points on both mathematical olympiads, meeting the IMO 2025 gold line and exceeding the USAMO 2026 gold line by 10 points. Notably, on USAMO 2026, this matches the highest reported human total among 340 competitors, indicating that the overall recipe can elicit top-level human-like olympiad

###### 2 Coarse RL

###### 3 Refined RL

###### 1 SFT

###### 4 TTS

Refined Reinforcement Learning

Scale Test-Time Reasoning

Expand Reasoning Mechanisms

Boost Reasoning Capabilities

Instill Rigorous Reasoning

###### Model-Agnostic Test-time Scaling Verification-and-Refinement Loop

Supervised Fine-Tuning Reverse-Perplexity Curriculum

Coarse Reinforcement Learning Verifiable Rewards

Refined Reinforcement Learning Generative Rewards

###### Policy Rollouts

- 1 Generation

- 2 Self-Refinement

2 Iterative Self-Improvement Pipeline

###### Post-trained Reasoning Model (Initialized from P1-30B-A3B)

✗

✓

Iterative

Refined

✗ ✓

Initial

Process

[Figure 1]

Solution

Solution

Initial

Problem Policy

Problem Policy Rollouts

Solution

Long-CoT SFT

###### Generative Reward

Data Curation Data Curriculum

RL-Optimized Policy (Stronger Verification/Refinement Ability)

Outcome-Based Reward Model

Unfamiliar

High-PPL Examples

DeepSeek-Math V2

ProofsearchPatterns

Math STEM

∫x dx

Review Feedback

✓ CoT Validity

| |
|---|

Initial Solution

Refined

Solution

Refine

✓ Proof Quality

###### Bug Report

Code IF

Model-Based

Final Answer

Formal Verifier /

Critical Errors Major Justification Gaps

3 Experience Replay

...

Judge

Correctness

Symbolic Check

✓ Anti-Hacking

Replay

Minor Justification Gaps

Self-Verify Self-Correct

Familiar

Low-PPL Examples

GSPO

Store

###### Accept

###### Reject

Experience Buffer Policy

Post-Trained Reasoning Model (Stronger Reasoning Style)

Pass Verification

1

Major Issues

1

5 Consecutive Times

Persist for 10 Steps

Policy Update

Goal: Scale test-time compute via iterative

Goal: Use more fine-grained feedback to encourage proof rigor and self-refinement.

Goal: Improve search, coverage, and direct

Goal: Install the rigorous reasoning pattern that the rest of the pipeline will later scale.

refinement to enhance answer quality.

solving performance on hard tasks.

Coarse RL

Coarse RL

|Model / Policy Input / Problem Output / Answer Experience Buffer Iterative Process Optimization<br><br>|
|---|

Figure 2: Overview of the SU-01 training and inference pipeline. The recipe first reshapes the backbone with rigorous long-form SFT, then scales the resulting behavior through coarse and refined RL, and finally applies test-time verification and refinement for olympiad-level problem solving.

reasoning from a compact 30B-A3B model. The TTS traces further show how this capability emerges at inference time: SU-01 can sustain reasoning trajectories beyond 100K tokens, condition on its own drafts and error analyses, and repeatedly verify and repair candidate proofs. Overall, these results support a specializable-generalist view of compact reasoning models: with the right training and inference recipe, a broadly capable backbone can be driven toward expert-level proof reasoning while retaining meaningful scientific transfer.

#### 2 Instilling Rigorous Reasoning via SFT

The first stage of the SU-01 pipeline uses supervised fine-tuning to reshape the model’s reasoning behavior. We choose P1-30B-A3B (Chen et al., 2025) as the initial model because it already shows competitive performance in scientific reasoning, including both mathematics and physics. Despite its strong results on verifiable tasks, we observe that its solutions are not always organized around rigorous proof-search patterns. The purpose of SFT is therefore to reshape its reasoning behavior toward more explicit, disciplined, and proof-oriented long-form reasoning while preserving as much of its existing capability as possible.

We empirically find that applying SFT to a post-trained backbone is more efficient than training the same reasoning behavior from a base model. A post-trained model already contains useful instructionfollowing behavior, problem-solving ability, and broad scientific competence. Starting from that checkpoint allows SFT to focus on changing the reasoning pattern rather than rebuilding these capabilities from scratch. In this framing, SFT specializes the generalist backbone toward rigorous proof-search behavior while preserving its broad scientific competence, providing a stronger starting policy for subsequent RL to scale. The launch configuration and optimization hyperparameters for this stage are summarized in Section C.

##### 2.1 SFT Data Curation

We curate SFT prompts from a broad mixture of mathematical, scientific, instruction-following, and coding sources. The mathematical subset includes problems from Evan Chen’s olympiad materials1, the Shuzhimi Forum2, AoPS (Art of Problem Solving)3, online mathematical competition training books4, and DeepMath problems with difficulty at least 6 (He et al., 2025). For scientific reasoning, we include prompts from NaturalReasoning (Yuan et al., 2025). To improve the generalization of the SFT model beyond narrow olympiad-style mathematics, we also include chat prompts from

1Evan Chen’s olympiad materials: https://web.evanchen.cc/. 2The Shuzhimi Forum is an online Chinese mathematical problem-solving community. 3AoPS: https://artofproblemsolving.com/. 4The book subset is curated from publicly available online mathematical competition training materials.

Nemotron-Instruction-Following-Chat-v15 and coding prompts from Eurus-2-RL-Data (Cui et al., 2025a) and OpenCodeReasoning-26; the latter extends the OpenCodeReasoning data-distillation line for competitive coding (Ahmad et al., 2025).

Before generation, we first filter contaminated problems from the prompt pool. For each remaining prompt, we use DeepSeek-V3.2-Speciale (DeepSeek-AI, 2025a) to generate high-quality longform reasoning trajectories. We then filter noisy generations and remove trajectories longer than 8,192 tokens. This filtering step keeps the supervised signal focused on rigorous and usable reasoning traces, while avoiding extremely long outputs that are more likely to introduce truncation or unstable optimization.

###### Direct Generation 54.3%

Math 71.8K 21.2% STEM 62.9K 18.6% Code 30.2K 8.9% IF 18.8K 5.6%

### 338K

trajectories

###### Self-improvement 45.7%

Self-Verify 89.5K 26.4% Self-Refine 65.2K 19.3%

Figure 3: Composition of the SFT data after filtering. Math, STEM, Code, and IF form the direct-generation group; Self-Verify and SelfRefine form the self-improvement group.

In addition to direct solution trajectories, we further equip the model with self-verification and selfrefinement behaviors. For the mathematical subset, we ask DeepSeek-V3.2-Speciale to generate verification traces for the generated solutions, followed by refinement traces that address issues identified during verification. These examples expose the model to the behaviors that are especially important for olympiad-level reasoning: checking whether a proof is actually justified and improving an argument when a flaw is found. Finally, we obtain a filtered SFT mixture of 338K trajectories, as shown in Figure 3.

##### 2.2 Reverse-Perplexity Curriculum for SFT

Long-CoT SFT on a post-trained reasoning model is a delicate optimization problem. The model already contains a strong instruction-following and reasoning policy, so SFT is not simply adding a new capability to an empty backbone; it is modifying an existing policy while trying to preserve its original competence. If the supervised signal is too narrow or the training is stopped too early, performance can degrade substantially even when the model starts to imitate more explicit long-form reasoning. This tension is consistent with the long-CoT degradation phenomenon studied by Luo et al. (2025): a post-trained model often needs sufficient data scale and enough SFT epochs to absorb the new reasoning style without overwriting the useful competence installed by previous post-training stages.

In our setting, recovery depends strongly on both training duration and the length behavior of the resulting model (Ren et al., 2026). For trajectories capped at 8,192 tokens, we empirically find that four epochs are usually sufficient to recover most of the model capability after the initial behavioral shift, provided that the data mixture and learning rate are well controlled. We also treat validation truncation rate as an operational indicator of SFT sufficiency. A post-trained model that has not been sufficiently adapted to rigorous long-CoT supervision often exhibits shallow reasoning behaviors: it circles around local heuristics, repeats intermediate claims, and continues reasoning without making decisive progress. These repetitive and endless-reasoning patterns naturally increase truncation. In practice, we find that a truncation rate below 5% is a useful sign that the model has largely adapted to the target reasoning style.

To make long-CoT SFT more stable, we use a reverse-perplexity training curriculum. Let D = {(xi,yi)}Ni=1 be the SFT set, where xi is the prompt and yi = (yi,1,...,yi,T

) is the teacher trajectory. Given the initial policy π0, we score each example by its length-normalized perplexity, PPL(xi,yi) = exp −T1

i

Ti t=1 log π0(yi,t | xi,yi,<t) . Instead of presenting examples in random order or in ascending perplexity, we sort the data in descending perplexity and train from high-PPL examples to low-PPL examples within each epoch. This order repeatedly starts each pass from teacher trajectories that are most mismatched with the current policy, using unfamiliar proof-search patterns for behavioral adaptation before consolidating on more familiar examples. We discuss the empirical effect of this ordering in Section 6.3.

i

5Nemotron-Instruction-Following-Chat-v1 Hugging Face dataset card: link. 6OpenCodeReasoning-2 Hugging Face dataset card: link.

#### 3 Boosting Reasoning Capability with RL

Once the model has acquired a stronger long-form reasoning pattern, reinforcement learning provides the scalable feedback mechanism for turning this pattern into stronger expert behavior. We split this stage into two levels. Coarse RL converts the SFT reasoning pattern into stronger answer-seeking behavior under reliable, mostly verifiable reward signals, improving search, coverage, and direct solving performance on hard tasks. Refined RL then specializes the policy toward complete, auditable proof construction, using more fine-grained feedback to encourage proof rigor and self-refinement. The shared RL launch configuration and stage-specific hyperparameters are summarized in Section D.

##### 3.1 RL Data Curation

RL training uses a separate prompt pool from SFT, curated to support both answer-verifiable optimization and proof-quality refinement. The physics subset is derived from olympiad-level physics data associated with P1 (Chen et al., 2025). The mathematical subset follows the same source families as our SFT data, including AoPS, online competition training books, Evan Chen’s olympiad materials, and the Shuzhimi Forum. We refer readers to §2.1 for source attribution. We additionally include OPC7, a human-evaluated corpus of advanced mathematical proofs (Dekoninck et al., 2025), to increase coverage of proof-oriented prompts.

We split the resulting RL pool into a verifiable set and a non-verifiable set. The verifiable set contains prompts whose final answers or structured outputs can be checked reliably, while the non-verifiable set includes proof-oriented or open-ended reasoning prompts that require softer judgment, e.g., generative reward. Before training, we first deduplicate and decontaminate the prompt pool. We then apply rejection sampling to remove examples that are already too easy or too hard for the current policy, and further filter noisy prompts that are poorly formatted or otherwise unreliable. The final RL pool contains 8,967 verifiable prompts and 16,287 non-verifiable prompts.

##### 3.2 Coarse RL

Coarse RL trains the SFT model on the 8,967 verifiable prompts described above. We formulate this stage as reinforcement learning with verifiable rewards (RLVR; Lambert et al. 2024; Guo et al. 2025), using Group Sequence Policy Optimization (GSPO; Zheng et al. 2025). GSPO is better aligned with outcome-reward training than token-level GRPO because both reward assignment and policy clipping operate at the complete-response level. For each prompt q ∈ Dver (the verifiable prompt set), the rollout policy πθ

samples a group of K candidate solutions Gq = {oi}Ki=1. The verifier converts each final answer into a binary outcome reward r(q,o) = 1 if the extracted final answer is verified as correct, and r(q,o) = 0 otherwise. The group-relative advantage is computed from the within-prompt reward baseline. We use the unnormalized form without group standard-deviation

old

= K1 Kj=1 r(q,oj). The key GSPO quantity is the length-normalized sequence-level importance ratio si(θ) = exp{|o1

normalization, Ai = r(q,oi) − µG

, where µG

q

q

|oi| t=1 log ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t)}. The policy is updated with the clipped sequence-level surrogate

i|

JGSPO(θ) = Eq,{o

i}

K

1 K

min si(θ) Ai,clip(si(θ),1 − ϵ,1 + ϵ) Ai . (1)

i=1

These definitions are also the interface used by the experience replay variants in the subsequent subsection: replayed trajectories can reuse the same reward, advantage, and sequence-ratio notation while changing the source policy in the denominator of si(θ). Following the routing-replay motivation in GSPO (Zheng et al., 2025), we freeze the MoE router during RL so replayed trajectories are evaluated under stable expert-routing decisions, which reduces replay-induced instability.

The reward system is intentionally layered to keep high-precision automatic checks before more expensive model-based judgments. We first extract the final answer and apply canonicalized text matching. Unresolved cases are then checked by Math-Verify8, a rule-based mathematical-expression

7OPC dataset card: link. 8Math-Verify repository: link.

evaluation pipeline for LLM outputs. Samples that still fail these rule-based checks are sent to gptoss-120b9 (OpenAI, 2025) for generative verification. This ordering makes the reward conservative by default, while still recovering correct solutions whose final answers are equivalent but difficult to normalize with rule-based parsers alone.

##### 3.3 Refined RL

After coarse RL has established strong search behavior, refined RL shifts the optimization target from answer correctness to proof quality. The central issue is that many olympiad solutions can reach a correct final answer while still containing hidden gaps, unjustified transformations, or incomplete case analysis. Refined RL therefore uses a stronger process-level reward and adds two memory mechanisms: self-refinement, which turns recent failures into repair tasks, and experience replay, which preserves rare successful proofs long enough for the policy to learn from them.

Generative proof reward. We use DeepSeekMath-V2 as a generative reward model for refined RL (DeepSeek-AI, 2025b), except for physics prompts. For every rollout from both the verifiable and non-verifiable subsets, the reward model reads the problem and the complete solution or proof, then outputs a binary score rproof(q,o) ∈ {0,1}. Unlike the coarse verifier in §3.2, this score is not restricted to checking whether the final answer matches a reference answer. It evaluates whether the full reasoning path is mathematically valid, sufficiently rigorous, and complete. This makes the reward more aligned with the final goal of olympiad reasoning, but also more expensive and more vulnerable to judge artifacts. We therefore apply anti-hack preprocessing before sending a response to the reward model: malformed generations with leaked chat-template tokens, unbalanced thinking delimiters, or severe repetition are replaced by a safe fallback answer. This prevents the policy from receiving reward by exploiting formatting or verifier-input pathologies rather than by improving the proof. The reward-model serving configuration is summarized in Section E.

Self-refinement. Self-refinement exposes the policy to the same repair pattern that we use at test time: propose a solution, inspect it, locate gaps, and produce a corrected proof. After each rollout, responses are grouped by query. If a query group has average proof reward below a threshold τref = 0.5, failed responses from that group are converted into refinement prompts. Each prompt contains the original problem, the previous incorrect solution, and an instruction to critique the argument, fix proof errors, fill missing justifications, and output a complete final solution. These prompts are stored in a self-refinement buffer and mixed into subsequent batches with target ratio ηref = 0.2. Normal samples displaced by refinement queries are returned to a buffer, so refinement does not silently discard fresh training data. We also do not recursively enqueue failed refinement attempts, which avoids spending repeated updates on examples that remain outside the current policy’s learnable region.

Experience replay. On difficult proof problems, the policy may occasionally discover a valid solution trajectory even though it usually fails on the same query. Immediately discarding such a trajectory wastes a high-value training signal. Following ExGRPO (Zhan et al., 2025), we keep a replay buffer E keyed by query, but our implementation is simpler: it uses the same GSPO-style update and does not apply the policy-shaping transform introduced in ExGRPO. After each rollout, a query is admitted to the replay buffer only when it is hard but solvable, operationalized as 0 < n+(q) < 2, where n+(q) is the number of successful trajectories in the current group. In answer-only RLVR, such a unique success can be a lucky hit: a trajectory may end with the correct final answer while still containing brittle or invalid reasoning(Zhan et al., 2025). In our refined RL setting, however, success is assigned by the DeepSeekMath-V2 proof reward, which inspects the full solution rather than only the final answer. This does not eliminate reward-model noise, but it makes a rare successful rollout substantially more likely to encode a reusable proof pattern and therefore a safer replay target. Stored trajectories are deduplicated, and a query is retired once fresh on-policy rollouts solve it often enough, with threshold n+(q) ≥ 4.

Replay is mixed with fresh proof-reward training rather than run as a separate mode. A replay rollout injects one stored successful trajectory for the selected query, and the replay ratio is controlled by ρ = 0.25 over the non-refinement portion of the batch. When multiple successful trajectories are

9gpt-oss-120b model card: link.

stored for a query, we select the lowest-entropy one, o∗ = arg mino∈E(q) H(o;πθ), using rollout-side top-k log probabilities as an efficient entropy estimate, following the trajectory-selection principle in ExGRPO (Zhan et al., 2025). The resulting refined objective is

[JGSPO(q∗,{o∗} ∪ Gq∗;θ,πθ

Jrefined(θ) = (1 − ρ)EB

)] + ρEB

[JGSPO(q,Gq;θ,πθ

)], (2)

exp

src

fresh

old

where πθ

for fresh rollouts. This replay design is targeted rather than exhaustive: it stores rare valid proofs, prefers the most stable stored trajectory, replays it at a controlled ratio, and removes it once the current policy can reliably reproduce the behavior on-policy.

for the replayed trajectory and πθ

= πθ

= πθ

src

past

src

old

#### 4 Achieving Gold-Medal-Level Reasoning via Test-time Scaling

Even with a strong reasoning policy, the hardest problems often require substantial search and revision at inference time. This is not merely a matter of sampling more answers. IMO-style tasks demand complete and rigorous proofs, and a solution with the right final conclusion can still fail if it contains a hidden gap or a logical fallacy. Recent work on IMO 2025 makes this point explicit: strong frontier models already contain significant mathematical capability, but their single-pass outputs and even best-of-many selection can remain far below the level obtained by a structured verification-and-refinement pipeline (Huang & Yang, 2025).

The need for test-time scaling is also tied to reasoning budget. A single generation has a finite context and thinking budget, while an olympiad proof may require several rounds of exploration, lemma checking, counterexample search, and exposition repair. A model can spend most of its budget discovering a promising approach and still fail to fully close the proof. Breaking inference into repeated solve–verify–refine stages effectively allocates additional computation to the same problem while keeping each step focused and auditable. This extra budget is useful only when the model can remain coherent across repeated drafts, critiques, and repairs. After the full training pipeline, SU-01 is able to use this budget productively on difficult problems, sustaining coherent reasoning trajectories longer than 100K tokens during inference.

Our TTS procedure follows the verification-and-refinement paradigm of Huang & Yang (2025) as a self-verification and refinement loop. The model first produces an initial solution under a solver prompt that prioritizes proof rigor rather than merely reaching a final answer. It then enters refinement, where it revisits the draft, repairs weak steps, and tries to turn a promising argument into a complete proof. The refined candidate is next checked through a verification prompt: the model inspects the full solution and writes a structured bug report, identifying issues such as critical errors, unjustified claims, or missing cases. A verdict step interprets this report and decides whether the candidate should be accepted, rejected, or sent back for another refinement round. This loop is repeated until the solution consistently passes self-verification or the refinement budget is exhausted. Multiple independent runs can be executed in parallel or serial, and accepted candidates are selected only after the proof is stable under repeated verification. The corresponding inference setting is summarized in Section E.

#### 5 Experimental Results

The experimental section is organized around three complementary evaluation views: answerverifiable reasoning tasks, non-verifiable or proof-oriented tasks, and official olympiad competition problems.

##### 5.1 Benchmarks

We organize evaluation into three benchmark families. The first family contains answer-verifiable reasoning tasks, where correctness can be checked by a final answer or a high-confidence automatic verifier. It includes AMO-Bench (An et al., 2025), AIME 2025 and AIME 202610, AnswerBench from the IMO-Bench evaluation suite (Luong et al., 2025), and FrontierScience-Olympiad, the Olympiad

10AIME official competition page: https://maa.org/math-competitions/aime.

Table 1: Performance on answer-verifiable reasoning tasks. Results for AnswerBench, AMO-Bench, AIME 25/26, and FrontierScience-Olympiad are averaged over 4, 8, 8, and 4 runs, respectively. FrontierScience-Olympiad abbreviates the Olympiad subset of FrontierScience. Avg. is the mean of AnswerBench, AMO-Bench, AIME 2025, AIME 2026, and FrontierScience-Olympiad. Within each comparison block, bold marks the best score and underline marks the second best.

FrontierScience-Olympiad

Model AnswerBench AMO-Bench AIME 25/26

Avg. Physics Chemistry Biology Overall

P1-30B-A3B 69.3% 41.3% 90.4% / 89.6% 57.5% 57.5% 27.5% 54.5% 69.0% GLM-4.7-Flash 73.8% 53.8% 91.3% / 88.3% 54.5% 60.0% 17.5% 53.0% 72.0% Nemotron-Cascade-2 80.5% 40.8% 94.2% / 90.0% 56.0% 56.3% 30.0% 53.5% 71.8% Qwen3.6-35B-A3B 78.0% 58.8% 92.5% / 92.9% 65.5% 74.4% 25.0% 65.0% 77.4% Gemma-4-31B 74.0% 39.3% 88.8% / 91.3% 69.0% 61.9% 17.5% 61.0% 70.9% SU-01 77.5% 59.8% 94.6% / 93.3% 62.5% 69.4% 25.0% 61.5% 77.3%

subset of FrontierScience (Wang et al., 2026). These benchmarks mainly test whether the model can produce correct final answers under single-pass or fixed-budget inference.

The second family contains non-verifiable or proof-oriented tasks. We include ProofBench from IMO-Bench (Luong et al., 2025), which emphasizes proof quality rather than only final-answer matching, and FrontierScience-Research, the research subset of FrontierScience (Wang et al., 2026). These tasks are used to probe whether training improves rigorous reasoning and scientific problem solving beyond answer-checkable settings.

The third family contains official olympiad competition problems, including IMO 202511, USAMO 202612 and IPhO (2024, 2025)13. Detailed grading and verifier settings are summarized in Section G.

##### 5.2 Verifiable Problems

As shown in Table 1, SU-01 reaches a 77.3% average score across AnswerBench, AMO-Bench, AIME 2025, AIME 2026, and FrontierScience-Olympiad, nearly matching the strongest similarsize baseline, Qwen3.6-35B-A3B (77.4%). Importantly, SU-01 achieves this level of performance with a simpler unified post-training recipe and substantially lower training cost, highlighting the efficiency of our approach. The mathematical benchmarks show where this improvement is most pronounced. SU-01 achieves the best similar-size results on AMO-Bench (59.8%) and AIME 2025/2026 (94.6%/93.3%), which are closer to competition-style problem solving than routine answer extraction. On AnswerBench, SU-01 remains competitive at 77.5%, close to Qwen3.635B-A3B (78.0%) and behind only Nemotron-Cascade-2 (80.5%). FrontierScience-Olympiad tests whether this behavior transfers beyond pure mathematics. Although the RL stages use only math and physics signals, SU-01 reaches 61.5% overall and shows strong transfer to untrained STEM domains, including 69.4% on Chemistry and 25.0% on Biology. This cross-domain transfer supports the specializable-generalist framing: the model is specialized through math and physics reasoning signals, yet the resulting capability does not collapse into a narrow contest solver.

##### 5.3 Non-verifiable Problems

Non-verifiable benchmarks test whether the training recipe improves the quality of full reasoning traces, rather than only optimizing final-answer rewards. On IMO-ProofBench, Table 3 shows that SU-01 reaches 57.6% overall in direct generation, already the strongest result among similar-size models. Test-time scaling further raises the score to 70.2%, including 91.0% on the basic split and 49.5% on the advanced split, bringing a 30B-A3B model close to much larger frontier systems such

- as Gemini 3.1 Pro. This improvement indicates that self-verification and refinement are especially useful when correctness depends on the complete proof, not merely on producing the right final answer.

11International Mathematical Olympiad official archive: https://www.imo-official.org/. 12USA Mathematical Olympiad: https://maa.org/math-competitions/usamo. 13International Physics Olympiad official site: https://www.ipho-new.org/.

###### Table 3: Performance on non-verifiable benchmarks. FrontierScience-Research refers to the research subset of FrontierScience. For SU-01, x/y reports scores without and with TTS on ProofBench.

IMO-ProofBench FrontierScience-Research

Model

Basic Advanced Overall Physics Chemistry Biology Overall Larger models

Gemini 3.1 Pro Thinking 95.2% 50.0% 72.6% 0.0% 30.0% 10.0% 13.3% GPT-5.5-High 96.7% 64.8% 80.7% 25.0% 40.0% 45.0% 36.7% DeepSeek-V3.2-Speciale 77.6% 34.3% 56.0% 10.0% 20.0% 15.0% 15.0%

Similar-size models

P1-30B-A3B 33.8% 6.2% 20.0% 0.0% 10.0% 0.0% 3.3% GLM-4.7-Flash 51.0% 16.7% 33.8% 0.0% 0.0% 0.0% 0.0% Nemotron-Cascade-2 77.1% 28.6% 52.9% 5.0% 5.0% 20.0% 10.0% Qwen3.6-35B-A3B 39.1% 7.1% 23.1% 0.0% 5.0% 10.0% 5.0% Gemma-4-31B 46.7% 16.2% 31.4% 0.0% 10.0% 5.0% 5.0% SU-01 77.1%/91.0% 38.1%/49.5% 57.6%/70.2% 10.0% 10.0% 15.0% 11.7%

FrontierScience-Research is a substantially harder research-oriented subset of FrontierScience, covering physics, chemistry, and biology problems that require scientific modeling and multi-step reasoning beyond standard contest formats. Absolute scores remain low even for frontier systems, but SU01 obtains the best similar-size overall score at 11.7%. It also leads the similar-size block on Physics, ties for the best Chemistry score, and ranks second on Biology, despite our RL stages using only mathematics and physics signals. This cross-domain pattern suggests that the recipe learns a more general scientific reasoning behavior rather than only specializing to the training domains, providing early evidence of transferable research-level reasoning in a compact model.

Table 2: Performance on physics olympiad problems, averaged over 8 runs. Gold lines for IPhO 2024/2025 are 20.8/19.7 points. For SU-01, x/y reports scores without and with test-time scaling.

Model IPhO 2024 IPhO 2025

Larger models

Gemini 3.1 Pro Thinking 25.9 25.1 GPT-5.5-High 25.8 23.2 DeepSeek-V3.2-Speciale 25.1 21.9

Similar-size models

P1-30B-A3B 23.1 17.7 GLM-4.7-Flash 22.2 19.5 Nemotron-Cascade-2 21.2 16.7 Qwen3.6-35B-A3B 24.3 19.9 Gemma-4-31B 24.4 20.3 SU-01 23.5/25.3 20.3/21.7

5.4 Olympiad Competition Problems

Even without TTS, SU-01 averages 23.5 points on IPhO 2024 and 20.3 points on IPhO 2025, exceeding the corresponding gold lines of 20.8 and 19.7 points, as shown in Table 2. TTS further raises the scores to 25.3 and 21.7 points, making SU-01 the strongest similar-size model in both years among models with available scores.

###### Table 4 reports the final competitionstyle mathematics results. In direct generation, SU-01 reaches 21 points on IMO 2025 and 15 points on USAMO 2026, already clearing the bronze-medal lines for both competitions. The direct model obtains full credit on IMO 2025 P2 and USAMO 2026 P1/P4, nearcomplete solutions on IMO 2025 P4/P5,

Table 4: Performance on mathematical olympiad competition problems. Medal lines for IMO 2025 are 35/28/19 points, and medal lines for USAMO 2026 are 25/18/11 points. ⋆ indicates that TTS results are evaluated by human experts, while direct generation results are evaluated automatically (Section G).

IMO 2025 Model P1 P2 P3 P4 P5 P6 Total

SU-01 1 7 1 6 6 0 21 SU-01 w/ TTS 7⋆ 7⋆ 7⋆ 7⋆ 7⋆ 0⋆ 35⋆

[Figure 2]

USAMO 2026 Model P1 P2 P3 P4 P5 P6 Total

SU-01 7 0 0 7 0 1 15 SU-01 w/ TTS 7⋆ 0⋆ 7⋆ 7⋆ 7⋆ 7⋆ 35⋆

[Figure 3]

and still fails several harder problems in a single pass. This indicates that the base model has acquired substantial olympiad reasoning ability, but still benefits from additional search and self-correction on the most brittle proof attempts.

With test-time scaling, SU-01 reaches 35 points on both IMO 2025 and USAMO 2026, meeting the IMO gold line exactly and exceeding the USAMO gold line by 10 points. TTS upgrades five IMO

- 2025 problems to full credit while P6 remains unsolved, and recovers full-credit solutions on five of six USAMO 2026 problems while P2 remains unresolved. The USAMO 2026 score summary reports 340 competitors, a median score of 6, a top-12 cutoff of 26, and a maximum score of 3514. SU-01 therefore matches the highest reported human total on this contest, while still exposing a concrete failure mode on P2. This result suggests that our overall recipe can elicit top-level human-like olympiad reasoning from a compact 30B-A3B model.

Case study. We include the corresponding model-generated solutions and expert verdicts in Section H. Across the twelve IMO 2025 and USAMO 2026 problems, the model gives full-credit solutions to ten problems, with failures on IMO 2025 P6 and USAMO 2026 P2. Its main strength is translating olympiad problems into formal frameworks: coordinates or complex numbers for geometry, modular classifications for number theory, recurrences for functional equations, and automata-based dynamic programming for digit problems. A particularly striking example is USAMO 2026 P3: rather than following the standard synthetic geometry route, the model elegantly uses complex numbers to unify the unit circle, equilateral-triangle rotations, chord relations, and tangent conditions within a single algebraic framework. This yields an ingenious analytic reformulation of a configuration that olympiad solvers would typically approach through angle chasing and carefully chosen auxiliary constructions. IMO 2025 P2 shows a complementary strength: the model reduces a configuration involving two intersecting circles, an orthocenter, and a tangency claim to coordinate and distance computations. Other strong examples include the carry-state dynamic programming approach for USAMO P4 and the number-theoretic proof using totients, congruences, Vieta jumping, and Fibonacci structure in USAMO P6. However, the failures show a clear limitation: the model can miss subtle structural constraints, as in the invalid column-permutation reduction in IMO P6, or leave gaps in delicate global strategy arguments, as in USAMO P2. Overall, the model performs well when a problem admits a rigid formal representation, but is less reliable when the core challenge is preserving combinatorial structure or proving a finely tuned process invariant.

#### 6 Analysis and Discussion

ProofBench-Basic ProofBench-Advanced

AnswerBench

100

|77.2<br><br>(+17.4)<br><br>77.5 (+0.3) 91.0<br><br>|
|---|
|69.2 59.8 (-9.4) 76.7<br><br>(+19.1)<br><br>77.1<br><br>(+0.5)<br><br>(+13.8)|
|57.6<br><br>(+23.8)<br><br>38.1<br><br>(+12.9)<br><br>49.5<br><br>(+11.4)|
|33.8 14.8<br><br>(+8.6)<br><br>25.2<br><br>(+10.5)<br><br>|
|6.2<br><br>|

##### 6.1 Progressive Rigorous Reasoning

80

- Figure 4 separates two kinds of reasoning progress. AnswerBench measures whether the model can recover a correct final answer under verifiable evaluation, whereas ProofBench grades the total solution and therefore exposes gaps in rigor, justification, and proof completion. The starting P1-30BA3B model is already strong on AnswerBench, but its ProofBench scores are much lower, especially on the Advanced split, indicating that answerseeking ability alone does not imply olympiadstyle proof reliability.

60

Score

40

20

0

P1-30B-A3B SFT SFT + Coarse RL

SU-01 SU-01 w/ TTS

Training Stage

Figure 4: Progressive reasoning performance across training stages.

The staged trend matches the intended role of each method component. SFT lowers AnswerBench from 69.2 to 59.8, but raises ProofBench-Basic from 33.8 to 57.6 and ProofBench-Advanced from 6.2 to 14.8. This is consistent with behavior shaping: the model is moved away from short answer recovery and toward longer proof-search, self-checking, and refinement patterns. Coarse RL then uses verifiable rewards to recover and improve direct solving ability, lifting AnswerBench to 77.2 while also improving ProofBench-Basic

14https://web.evanchen.cc/exams/posted-usamo-statistics.pdf

to 76.7 and ProofBench-Advanced to 25.2. This suggests that RLVR scales the rigorous reasoning behavior introduced by SFT into stronger problem-solving capability.

The final SU-01 model keeps AnswerBench essentially saturated at 77.5 and ProofBench-Basic nearly unchanged at 77.1, but improves ProofBench-Advanced from 25.2 to 38.1. The gain is therefore concentrated on harder non-verifiable proof problems, matching the role of Refined RL: proof-level generative rewards provide supervision beyond final-answer correctness, self-refinement prompts train the model to critique and repair its own solutions, and experience replay keeps rare successful hard-problem trajectories available long enough for the policy to learn more robust proof construction. Test-time scaling then further lifts the proof-oriented evaluation, reaching 91.0 on ProofBench-Basic and 49.5 on ProofBench-Advanced, showing that the trained self-verification and refinement behavior remains useful when additional inference compute is spent on checking and repairing candidate proofs.

##### 6.2 Characterizing Inference Scaling

We further inspect the TTS traces from USAMO 2026 to understand where inference compute is spent during difficult proof search. The key distinction is not only between short and long responses, but between qualitatively different reasoning contexts: initial generation starts from the problem statement, whereas refinement must condition on an existing solution together with verifier feedback or a bug report and then produce a revised proof.

- Figure 5 shows a clear allocation of computation across TTS actions. Initial solution generation is the longest stage, with a median length of 106K tokens, reflecting broad proof search and candidate construction. Refinement remains length-intensive, with a median of 83K tokens and a heavier upper tail, consistent with substantial proof repair. Verification is shorter but still substantive, with a median of 28.7K tokens, reflecting its role in auditing complete arguments for hidden gaps. Verdict parsing is lightweight, with a median of only 404 tokens.

Initial solution

Refinement

Verification

Verdict

0.1 1 10 100 160

Generation Length (thousand tokens, log scale)

Figure 5: Generation-length distribution of actions in the TTS pipeline on USAMO 2026.

The long refinement traces indicate that the model can reason over complicated conditioning contexts rather than merely produce long first-pass solutions: given a candidate proof and a structured critique, it often sustains another long reasoning trajectory to localize the flaw, preserve useful parts of the argument, and synthesize a corrected proof. The pattern therefore suggests that the training recipe enables the 30B model to generalize beyond direct solution generation, sustaining complex reasoning beyond 100K tokens and repeatedly verifying and refining its own solutions toward stronger candidates.

P1-30B Random

##### 6.3 Reverse-Perplexity Ordering

PPL asc. PPL desc.

|69.3|
|---|
|39.5<br><br>55.8<br><br>41.3 40.0|
|24.3<br><br>31.0<br><br>|
|15.0|

We compare the effect of different SFT data orderings on validation performance, as shown in Figure 6. The results come from validation experiments whose SFT training data differ from the final training mixture used for SU-01. Data ordering has a large effect on both score recovery and generation stability. Random ordering substantially under-recovers the P1-30B baseline, reaching 39.5 on AnswerBench and 31.0 on AMO-Bench, with truncation rates of 7.3% and 8.0%, respectively. Descending-PPL ordering recovers much more of the original capability, reaching 55.8 on AnswerBench and 40.0 on AMO-Bench, while reducing truncation to 0.3% and

60

Score

40

20

0

AnswerBench AMO-Bench

Figure 6: Validation results for SFT data ordering.

- 0.0%. The low-PPL-first setting is the weakest curriculum,

degrading to 24.3 on AnswerBench and 15.0 on AMO-Bench. These results suggest that descendingPPL ordering helps preserve the capability of the post-trained model while reshaping its reasoning behavior, and also prevents training from falling into a superficial long-generation regime with high truncation rates.

##### 6.4 Cost Analysis

SU-01 uses a compact and transparent post-training setup. Starting from a 30B-A3B backbone, we train on 338K SFT trajectories shorter than 8K tokens for four epochs with batch size 128. The RL stage uses 25K prompts for 200 steps, with batch size 128, 8 rollouts per prompt, and a 160K-token maximum response length. For reference, DeepSeek-V3.2 (DeepSeek-AI, 2025a) reports continued pre-training with 1,000 indexer warm-up steps over 2.1B tokens and 15,000 sparse-training steps over 943.7B tokens, followed by post-training with specialist distillation and mixed RL over thousands of continued-RL steps; its high-compute Speciale variant further trains on reasoning data with DeepSeekMath-V2-style proof rewards. Nemotron-Cascade 2 (Yang et al., 2026), a same-size 30B-A3B reference model, reports a substantially broader SFT mixture. Summing the disclosed category counts gives roughly 26.6M SFT samples across math, proof, code, science, long-context, chat, instruction-following, tool-use, and software-engineering data, trained with packed 256K-token sequences for 33K steps. Its post-training then continues with a multi-stage Cascade RL pipeline covering IF-RL, multi-domain RL, on-policy distillation, RLHF, long-context RL, CodeRL, and SWE RL. These comparisons highlight the main design point of SU-01: a simple and unified recipe can elicit strong olympiad-level reasoning from a compact 30B-A3B model while preserving scientific transfer.

#### 7 Related Work

Post-training for Large Reasoning Models. Post-training has become the main mechanism for turning strong pretrained language models into reliable reasoning systems. Early self-improvement work showed that models can bootstrap rationales from their own successful attempts (Zelikman et al., 2022), while open post-training recipes combine instruction tuning, preference optimization, and reinforcement learning to improve general assistant behavior (Lambert et al., 2024). For mathematical and long-CoT reasoning, DeepSeekMath introduced large-scale mathematical pretraining together with GRPO (Shao et al., 2024), Qwen2.5-Math emphasized math-specific self-improvement and tool-augmented data construction (Yang et al., 2024), DeepSeek-R1 demonstrated that large-scale RL can induce long reasoning traces and self-correction behaviors (Guo et al., 2025), and Kimi k1.5 highlighted long-context RL, curriculum design, and sampling-based inference (Kimi Team, 2025). Recent work further studies how to stabilize and reuse learning signal: off-policy guidance and experience replay improve sample reuse for reasoning policies (Yan et al., 2025; Zhan et al., 2025), entropy analyses explain exploration, entropy collapse, and selective entropy regularization in RLVR (Cui et al., 2025b; Jiang et al., 2025), and GSPO optimizes at the sequence level for MoE reasoning models (Zheng et al., 2025). Contemporary technical reports such as MiniMax-M2.5, Kimi-K2.5, and GLM-5 also show the growing importance of agentic post-training, efficient reasoning, long-context execution, and tool-oriented RL (MiniMax AI, 2026; Moonshot AI, 2026; Z.AI, 2026b).

Toward Olympiad-Level Reasoning. Olympiad reasoning pushes beyond benchmark math accuracy because solutions must be complete, rigorous, and robust to hidden gaps. One line of work addresses this challenge with specialized symbolic or neuro-symbolic systems: AlphaGeometry solves geometry problems by combining a neural language model with symbolic deduction (Trinh et al., 2024), and AlphaProof/AlphaGeometry 2 reached silver-medal-level IMO performance through formal reasoning and search (Google DeepMind, 2024). More recent frontier systems have moved toward broader natural-language reasoning and test-time search, including Gemini Deep Think’s gold-medal-level IMO result (Google DeepMind, 2025). In parallel, model-agnostic verification-andrefinement pipelines show that repeated generation, critique, repair, and acceptance decisions can substantially improve proof quality without relying on a single-pass answer (Huang & Yang, 2025), and DeepSeekMath-V2 studies self-verifiable mathematical reasoning as a training and inference target (DeepSeek-AI, 2025b). Nemotron-Cascade 2 provides another recent example of a compact MoE reasoning model approaching frontier mathematical and olympiad performance through cascade RL and multi-domain on-policy distillation (Yang et al., 2026). Our contribution is a simple and

unified recipe that enables a 30B-A3B model to develop rigorous proof behavior through post-training and reach olympiad-level performance with self-verification, refinement, and test-time scaling.

#### 8 Conclusion

This report presents a simple and unified recipe for turning a compact post-trained reasoning model into a stronger mathematical and scientific reasoner. Starting from a broadly capable 30B-A3B backbone, SU-01 combines reverse-perplexity curriculum SFT, efficient coarse RL with outcome verification, refined RL with proof-level rewards, self-refinement and experience replay, and test-time scaling through self-verification and refinement. Together, these stages decompose rigorous reasoning improvement into behavior shaping, scalable reward feedback, proof-level specialization, and inference-time repair. The resulting model reaches gold-medal-level performance on mathematical and physical olympiad competitions, sustains reasoning trajectories beyond 100K tokens during inference, and shows transfer to scientific domains beyond the main math and physics training signals. In summary, SU-01 supports a specializable-generalist view of compact reasoning models: with the right training and inference recipe, a broadly capable backbone can be driven toward expert-level proof reasoning while retaining meaningful scientific transfer.

#### Acknowledgments

This work was supported by the Shanghai Artificial Intelligence Laboratory. We thank the authors and maintainers of prior open research and infrastructure that made this work possible. In particular, we are grateful to DeepSeek for open-sourcing strong reasoning policies and generative reward models, which provided an important reference point for our work (DeepSeek-AI, 2025a;b). IMOBench, AMO-Bench, and FrontierScience helped guide the overall system optimization by offering challenging mathematical and scientific reasoning benchmarks and evaluation protocols (Luong et al., 2025; An et al., 2025; Wang et al., 2026). We also thank prior data efforts that supported our SFT and RL data curation, including DeepMath, NaturalReasoning, Eurus, OpenCodeReasoning, P1, and OPC (He et al., 2025; Yuan et al., 2025; Cui et al., 2025a; Ahmad et al., 2025; Chen et al., 2025; Dekoninck et al., 2025), as well as the many public problem sources and communities that cannot all be listed here. We further acknowledge the broader open-source infrastructure ecosystem, including slime for training and SGLang for efficient inference and serving (THUDM, 2026; SGLang Team, 2026).

#### Bibliography

Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding. arXiv preprint, abs/2504.01943, 2025.

Shengnan An, Xunliang Cai, Xuezhi Cao, Xiaoyu Li, Yehao Lin, Junlin Liu, Xinxuan Lv, Dan Ma, Xuanlin Wang, Ziwen Wang, and Shuang Zhou. AMO-Bench: Large language models still struggle in high school math competitions. arXiv preprint, abs/2510.26768, 2025.

Jiacheng Chen, Qianjia Cheng, Fangchen Yu, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Yun Luo, Yufeng Zhao, Futing Wang, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Wenxauan Zeng, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai, Yu Cheng, Ning Ding, Bowen Zhou, Peng Ye, and Ganqu Cui. P1: Mastering physics olympiads with reinforcement learning. arXiv preprint, abs/2511.13612, 2025.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025a.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint, abs/2505.22617, 2025b.

DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models. arXiv preprint, abs/2512.02556, 2025a.

DeepSeek-AI. DeepSeekMath-V2: Towards self-verifiable mathematical reasoning. arXiv preprint, abs/2511.22570, 2025b.

Jasper Dekoninck, Ivo Petrov, Kristian Minchev, Mislav Balunovic, Martin Vechev, Miroslav Marinov, Maria Drencheva, Lyuba Konova, Milen Shumanov, Kaloyan Tsvetkov, et al. The open proof corpus: A large-scale study of llm-generated mathematical proofs. arXiv preprint, abs/2506.21621, 2025.

Google. Gemma 4 31B model card, 2026. URL https://huggingface.co/google/ gemma-4-31B-it.

Google DeepMind. AI achieves silver-medal standard solving international mathematical olympiad problems, 2024. URL https://deepmind.google/blog/ ai-solves-imo-problems-at-silver-medal-level/.

Google DeepMind. Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the international mathematical olympiad, 2025. URL https://deepmind.google/ blog/gemini-deep-think-imo-2025/.

Google DeepMind. Gemini 3.1 Pro model card, 2026. URL https://deepmind.google/ models/model-cards/gemini-3-1-pro.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint, abs/2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint, abs/2504.11456, 2025.

Yichen Huang and Lin F. Yang. Winning gold at IMO 2025 with a model-agnostic verification-andrefinement pipeline. arXiv preprint, abs/2507.15855, 2025.

Yuxian Jiang, Yafu Li, Guanxu Chen, Dongrui Liu, Yu Cheng, and Jing Shao. Rethinking entropy regularization in large reasoning models. arXiv preprint, abs/2509.25133, 2025.

Kimi Team. Kimi k1.5: Scaling reinforcement learning with LLMs. arXiv preprint, abs/2501.12599, 2025.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint, abs/2411.15124, 2024.

Renjie Luo, Jiaxi Li, Chen Huang, and Wei Lu. Through the valley: Path to effective long CoT training for small language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 4972–4992. Association for Computational Linguistics, 2025.

Yun Luo, Futing Wang, Qianjia Cheng, Fangchen Yu, Haodi Lei, Jianhao Yan, Chenxi Li, Jiacheng Chen, Yufeng Zhao, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Wenxuan Zeng, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai, Yu Cheng, Ning Ding, Bowen Zhou, Peng Ye, and Ganqu Cui. P1-VL: Bridging visual perception and scientific reasoning in physics olympiads. arXiv preprint, abs/2602.09443, 2026.

Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuk Jung. Towards robust mathematical reasoning. arXiv preprint, abs/2511.01846, 2025.

MiniMax AI. MiniMax-M2.5 model repository, 2026. URL https://huggingface.co/ MiniMaxAI/MiniMax-M2.5.

Moonshot AI. Kimi-K2.5 model card, 2026. URL https://huggingface.co/ moonshotai/Kimi-K2.5.

OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/ 2508.10925.

OpenAI. GPT-5.5 system card, 2026. URL https://openai.com/index/ gpt-5-5-system-card/.

Qwen. Qwen3.6-35B-A3B model card, 2026. URL https://huggingface.co/Qwen/ Qwen3.6-35B-A3B.

Qihan Ren, Peng Wang, Ruikun Cai, Shuai Shao, Dadi Guo, Yuejin Xie, Yafu Li, Quanshi Zhang, Xia Hu, Jing Shao, and Dongrui Liu. Rethinking generalization in reasoning SFT: A conditional analysis on optimization, data, and model capability. arXiv preprint, abs/2604.06628, 2026.

SGLang Team. SGLang: Efficient execution of structured language model programs, 2026. URL https://github.com/sgl-project/sglang.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint, abs/2402.03300, 2024.

THUDM. SLiMe: An SGLang-Native post-training framework for reasoning models, 2026. URL https://github.com/THUDM/slime.

Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong. Solving olympiad geometry with-

out human demonstrations. Nature, 625(7995):476–482, 2024. doi: 10.1038/s41586-023-06747-5. vLLM Team. vLLM: A high-throughput and memory-efficient inference and serving engine for llms,

2026. URL https://github.com/vllm-project/vllm.

Miles Wang, Robi Lin, Kat Hu, Joy Jiao, Neil Chowdhury, Ethan Chang, and Tejal Patwardhan. FrontierScience: Evaluating AI’s ability to perform expert-level scientific tasks. arXiv preprint, abs/2601.21165, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems, NeurIPS, 2022.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint, abs/2504.14945, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint, abs/2409.12122, 2024.

Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nemotron-Cascade 2: Post-training LLMs with cascade RL and multi-domain on-policy distillation. arXiv preprint, abs/2603.19220, 2026.

Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E. Weston, and Xian Li. Naturalreasoning: Reasoning in the wild with 2.8m challenging questions. arXiv preprint, abs/2502.13124, 2025.

Z.AI. GLM-4.7-Flash overview, 2026a. URL https://docs.z.ai/guides/llm/glm-4.7. Z.AI. GLM-5 overview, 2026b. URL https://docs.z.ai/guides/llm/glm-5. Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping reasoning with

reasoning. arXiv preprint, abs/2203.14465, 2022. Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, and Yu Cheng. ExGRPO: Learning to reason from experience. arXiv preprint, abs/2510.02245, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint, abs/2507.18071, 2025.

#### Appendix

Contents

- A Implementation and Evaluation Details 17
- B Problem-Solving Prompt 17
- C SFT Training Details 18
- D RL Training Details 18
- E Inference and Reward-Model Serving Details 18
- F Compared Models 19
- G Evaluation Details 19
- H Model Solutions for IMO 2025 and USAMO 2026 20

- H.1 IMO 2025 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- H.2 USAMO 2026 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

#### A Implementation and Evaluation Details

This appendix collects the implementation details needed to interpret and reproduce the reported training, inference, reward-model serving, and evaluation procedures. We summarize the effective modeling, optimization, decoding, serving, and grading settings extracted from the corresponding launch scripts and evaluation protocols, while omitting infrastructure-only commands that do not affect the training objective, data semantics, inference behavior, reward definition, or evaluation criteria.

#### B Problem-Solving Prompt

Unless otherwise specified, all RL training and inference stages use the following fixed problemsolving prompt for SU-01:

##### Problem-Solving Prompt

Please solve the following olympiad problem. Show your complete reasoning and proof.

- 1. Please use LaTeX format to represent the variables and formulas used in the solution process and results.
- 2. If the problem asks you to find specific values, please put the final answer(s) in \boxed{}.
- 3. If the problem requires a proof, present a clear and rigorous argument.

#### C SFT Training Details

The SFT stage is implemented with slime (THUDM, 2026), trained on 8 GPUs, initializes from P130B, and optimizes the reverse-perplexity-ordered mixture described in Section 2.1; rollout shuffling is disabled so that the curriculum order is preserved throughout training. We train for four epochs with batch size 128, using Adam with learning rate 1 × 10−5, cosine decay to a minimum learning rate of 1 × 10−6, warmup fraction 0.1, weight decay 0.1, and momentum parameters β1 = 0.9 and β2 = 0.95. This configuration treats SFT as a controlled behavioral adaptation step: the objective is not to maximize benchmark performance directly, but to expose the post-trained backbone to a stable sequence of rigorous long-form trajectories while preserving as much of its existing mathematical and scientific competence as possible.

#### D RL Training Details

The RL stage is implemented with slime (THUDM, 2026), trained on 64 GPUs, continues from the SFT checkpoint, and trains with a GSPO objective over a balanced mixture of verifiable prompts, proof-reward prompts, self-refinement prompts, and replayed experience. In total, the model is trained for 200 RL steps, consisting of 96 coarse-RL steps followed by 104 refined-RL steps. Each rollout uses prompt batch size 128, 8 samples per prompt, maximum response length 160k tokens, temperature 1.0, and 4 policy-update steps per rollout. The data pipeline applies dynamic sampling with a non-zero standard-deviation reward requirement, partial rollout, and oversampling batch size 160; the oversampling and partial-rollout mechanism increases the candidate prompt pool under filtering while recycling completed partial generations, and replay filtering prevents repeatedly training on queries that have become too easy for the current policy. The policy objective uses the GSPO advantage estimator with KL coefficient 0, entropy coefficient 0, symmetric clip range 10−3, and trajectory importance sampling enabled.

The self-refinement mechanism uses a refinement prompt template that includes the original problem and the previous failed answer. The self-refinement ratio is set to 20%, which denotes the fraction of training queries sampled from refinement data, and the group reward threshold selects failed or partially failed rollout groups with an average reward below 0.5 for refinement. Experience replay largely follows the ExGRPO setting without reward shaping and is re-implemented in the slime training framework. We use an experience admission threshold of online rollout correctness below 25%, one replayed trajectory per replay query, and entropy-based selection estimated from SGLang top-16 log probabilities. A query is admitted to the experience pool if only one sampled rollout succeeds, making the successful trajectory a hard positive example. Experiences are retired once online rollout correctness reaches 50%, indicating improved model performance on the corresponding query. Optimization is performed using Adam with a constant learning rate of 1 × 10−6, weight decay of 0.1, and momentum parameters β1 = 0.9 and β2 = 0.98.

#### E Inference and Reward-Model Serving Details

For test-time scaling, inference is served with SGLang (SGLang Team, 2026) by default; the Nemotron-Cascade-2 and DeepSeek-V3.2-Speciale models are served with vLLM (vLLM Team,

- 2026) 15. We follow the same model-agnostic verification-and-refinement setting as Huang & Yang

(2025).

Concretely, inference is organized as an iterative solve–verify–refine procedure: the solver first produces a candidate proof, the candidate is inspected by a verifier that returns a structured critique or bug report, and the solver then revises the proof conditioned on this feedback until the candidate is accepted or the refinement budget is exhausted. Unless otherwise specified, each generation uses a maximum length of 160,000 tokens, temperature 1.0, and top-p 0.95. We use the following exceptions for model- or benchmark-specific constraints. For Nemotron model, we set top-p to

- 1.0. The maximum generation length is set to 131K tokens for AIME25/26, AMOBench, IPhO, and FrontierScience-Olympic, and to 256K tokens for IMO-AnswerBench, IMO-ProofBench, and FrontierScience-Research subset. For API-limited models, we set the maximum decoding completion

15Nemotron-Cascade-2:vLLM-0.17.2rc1.dev148+g47b7af0d8.cu128 ; DeepSeek-V3.2-Speciale: vLLMv0.20-CUDA12.9.

length to the largest value allowed by the API request constraints: 128,000 tokens for GPT-5.5 API and 65,535 tokens for Gemini 3.1-Pro.

The TTS loop uses explicit stopping rules following the verification-and-refinement protocol. Within a single run, MAX VERIFICATION TRUE ROUNDS=5 means that a candidate is accepted once it passes verification for five consecutive rounds, while MAX VERIFICATION FALSE ROUNDS=10 terminates the run early after ten consecutive failed verification rounds. Each run allows at most MAX EXPLORATION ROUNDS=30 solve–verify–refine cycles, corresponding to the repeated exploration and correction loop in the TTS pipeline. For each problem, we set MAX RUNS=10, so at most ten independent runs are launched. This setting is intentionally aligned with the self-refinement behavior used during refined RL, so the model is evaluated under the same type of proof-repair workflow that it sees during training.

For proof-reward evaluation during refined RL, the DeepSeekMath-V2 reward server is deployed on 32 GPUs with maximum context length 32k and data-parallel degree 64. Speculative decoding is enabled during server-side inference using an MTP-based configuration. We adopt a 3-step speculative decoding scheme with a single top-ranked EAGLE draft candidate and 4 draft tokens per step. This setting improves verifier efficiency while maintaining the evaluation correctness required by the refined-RL training.

#### F Compared Models

The comparison tables use a mixture of public technical reports, official model cards, and official benchmark reports. For the answer-verifiable comparisons in Table 1, the larger-model group contains DeepSeek-V3.2 (DeepSeek-AI, 2025a), GPT-5.5 (OpenAI, 2026), and Gemini 3.1 Pro Thinking (Google DeepMind, 2026). The similar-size group contains GLM-4.7-Flash (Z.AI, 2026a), Nemotron-Cascade-2-30B-A3B (Yang et al., 2026), Qwen3.6-35B-A3B (Qwen, 2026), and Gemma4-31B (Google, 2026), together with our SU-01. These same active comparison families are reused in the non-verifiable benchmark table in Table 3, with missing benchmark entries marked by –. GPT-5.5-High is treated as the high-reasoning variant of GPT-5.5, Gemini 3.1 Pro Thinking is drawn from the Gemini 3.1 Pro family, and DeepSeek V3.2 uses the same DeepSeek-V3.2 source as above. When no paper-style technical report is available, we cite the corresponding official model card, repository, or product page.

#### G Evaluation Details

For answer-verifiable tasks, we use a layered automatic grading pipeline. We first apply rule-based verification, including canonicalized answer matching and symbolic or expression-level checks with Math-Verify16. If a response is not resolved by these checks, we send it to gpt-oss-120b for generative verification (OpenAI, 2025). For FrontierScience-O, we use the same verifiable grading pipeline, but adopt the dedicated prompt described in the original FrontierScience paper (Wang et al., 2026).

For non-verifiable and proof-oriented tasks, we follow each benchmark’s official evaluation protocol. ProofBench is evaluated following the IMO-Bench framework (Luong et al., 2025): solutions are graded with the four-level score set {0,1,6,7}, and we use Gemini-2.5-Pro as the grading model to align with the official reported setting. FrontierScience-R is evaluated with the official FrontierScience procedure using GPT-5-high as the judge (Wang et al., 2026).

For IPhO, we follow the evaluation paradigms used in P1 and P1-VL (Chen et al., 2025; Luo et al., 2026), which combine physics-specific grading prompts with competition-style scoring criteria. For IMO and USAMO, we construct a ProofBench-style automatic grading framework from annotated reference solutions and grading guidelines prepared by expert annotators who have won gold medals in mathematical olympiads, such as IMO or CMO. For the final verdict, three gold-medal experts independently assign scores from {0,1,6,7}, and we report the worst of the three scores as a conservative final score.

16Math-Verify repository: https://github.com/huggingface/Math-Verify.

#### H Model Solutions for IMO 2025 and USAMO 2026

This section lists the model-generated solutions used for the human-expert scoring reported in Table 4. Each solution is followed by its final expert verdict. For USAMO 2026 Problem 2, we show only an excerpt because the original response is substantially longer than the other solutions.

##### H.1 IMO 2025 IMO 2025 Problem 1.

###### IMO 2025 Problem 1

A line in the plane is called sunny if it is not parallel to any of the x-axis, the y-axis, or the line x + y = 0. Let n ≥ 3 be a given integer. Determine all nonnegative integers k such that there exist n distinct lines in the plane satisfying both of the following:

- • for all positive integers a and b with a + b ≤ n + 1, the point (a, b) lies on at least one of the lines; and
- • exactly k of the n lines are sunny.

Solution Remark. 7/7. Definition. For an integer n ≥ 3 let

Pn = {(a, b) ∈ N2 | a ≥ 1, b ≥ 1, a + b ≤ n + 1}.

We are asked for which nonnegative integers k there exist n distinct lines in the plane such that

▷ every point of Pn lies on at least one of the lines, and

▷ exactly k of the lines are sunny (i.e. not parallel to the x-axis, the y-axis, or the line x + y = 0). We will prove that for every n ≥ 3 the only possible values of k are 0, 1, 3.

1. Constructions for k = 0, 1, 3 We give explicit families of n distinct lines that satisfy the two conditions.

- k = 0 Take the n horizontal lines

y = 1, y = 2, . . . , y = n.

Each point (a, b) ∈ Pn lies on the line y = b. No line is sunny (all are parallel to the x-axis). Hence k = 0 is attainable.

- k = 1 Take the n − 1 horizontal lines

y = 1, y = 2, . . . , y = n − 1. These cover all points with b ≤ n − 1. The only point not covered is (1, n) (since a + n ≤ n + 1 forces a = 1). Add one extra line, for instance

L : y = x + (n − 1). L has slope 1, so it is sunny. It passes through (1, n). The total number of lines is (n − 1) + 1 = n, and exactly one line is sunny. Thus k = 1 is attainable.

k = 3 If n = 3:Use the three sunny lines

L1 : y = x, L2 : y = −2x + 5, L3 : y = −12x + 52.

Check: L1 contains (1, 1) and (2, 2); L2 contains (1, 3) and (2, 1); L3 contains (1, 2) and (3, 1). The six points of P3 are all covered. All three lines are sunny, so k = 3.

If n > 3:First take the n − 3 horizontal lines

y = 1, y = 2, . . . , y = n − 3. These cover all points with b ≤ n − 3. The remaining points are those with b = n − 2, n − 1, n; there are exactly 3 + 2 + 1 = 6 such points. Translate the situation by (0, −(n − 3)): the map

(x, y)  −→ (x, y − (n − 3))

sends these six points bijectively onto the set P3 (because a + b ≤ n + 1 becomes a + (b′ + (n − 3)) ≤ n + 1 ⇐⇒ a + b′ ≤ 4). Therefore, if we take the three sunny lines that cover P3 in the translated coordinates and then translate them back, we obtain three sunny lines in the original plane:

- S1 : y = x + (n − 3),
- S2 : y = −2x + (n + 2),
- S3 : y = −21x + 2n2−1.

Together with the n − 3 horizontal lines we have n distinct lines, exactly three of which are sunny. Hence k = 3 is attainable for every n ≥ 3.

###### 2. Upper bound on points per sunny line

- Lemma 1. For any sunny line L, we have |L ∩ Pn| ≤

n + 1 2

.

Proof. If L contains no integer point, then obviously |L ∩ Pn| ≤ 1 ≤ ⌊(n + 1)/2⌋ (since n ≥ 3). So assume L contains at least one integer point. Since L is sunny, it is not vertical, so we can write its equation as y = mx+d. ▷ If m is irrational, then L can contain at most one integer point (two integer points would give a rational slope). Hence the bound holds. Thus we may assume m is rational.

Write m = p/q in lowest terms, q > 0, gcd(p, q) = 1. Choose an integer point (x0, y0) on L. Then all integer points on L are

(x, y) = (x0 + qt, y0 + pt), t ∈ Z. Let the set of t for which the point lies in Pn be a consecutive integer interval tmin, . . . , tmax; let N be the number of such t. Then tmax − tmin = N − 1. For any such point we have the constraints

1 ≤ x ≤ n, 1 ≤ y ≤ n, 2 ≤ x + y ≤ n + 1. In particular,

q (N − 1) = |xmax − xmin| ≤ n − 1, (A) |p| (N − 1) = |ymax − ymin| ≤ n − 1, (B)

|p + q| (N − 1) = |(x + y)max − (x + y)min| ≤ n − 1. (C) Set M = max{q, |p|, |p + q|}. Then (A)-(C) give

N − 1 ≤

n − 1 M

.

Now, because L is sunny, we have p ̸= 0 (otherwise horizontal) and p + q ̸= 0 (otherwise slope −1). We claim that M ≥ 2. Suppose M = 1. Then q = 1 (since q > 0), |p| ≤ 1, |p + q| ≤ 1. As p ̸= 0, either p = 1 or p = −1.

▷ If p = 1, then |p + q| = |1 + 1| = 2 > 1, contradiction.

▷ If p = −1, then p + q = 0, which is forbidden. Hence M ≥ 2. Consequently,

N − 1 ≤

n − 1 2

=⇒ N ≤

n + 1 2

. Since N is an integer, N ≤ n+12 . □

3. No covering by all sunny lines for n ≥ 4

- Lemma 2. For n ≥ 4 there is no set of n distinct sunny lines that covers Pn. Proof. Even n: Write n = 2m. By Lemma 1 each sunny line contains at most m points. Therefore the total number of incidences (counting each point as many times as lines through it) is at most n · m = 2m2. But

n(n + 1) 2

= m(2m + 1) = 2m2 + m > 2m2 (m ≥ 1),

|Pn| =

so it is impossible to cover all points. Odd n: Write n = 2m + 1 with m ≥ 2 (i.e. n ≥ 5). Lemma 1 gives each sunny line at most m + 1 points. Hence

n

|Li ∩ Pn| ≤ n(m + 1) = (2m + 1)(m + 1). But

i=1

(2m + 1)(2m + 2) 2

|Pn| =

= (2m + 1)(m + 1).

Thus the inequality must be an equality. In particular, each line contains exactly m + 1 points, and the sum of the sizes equals |Pn|. Since every point is covered at least once, equality forces each point to be covered exactly once, and therefore the n lines are pairwise disjoint.

We now classify all sunny lines that can contain exactly m + 1 points in Pn.

- Lemma 3. Let n = 2m + 1 ≥ 3. If a sunny line L satisfies |L ∩ Pn| = m + 1, then L is one of the three lines

LA : y = x, LB : y = −2x + (n + 2), LC : y = −12x + n+22 .

Proof. Write L in reduced form y = pqx + d with q > 0, gcd(p, q) = 1, p ̸= 0, p + q ̸= 0. Let N = m + 1. From (A)-(C) we have (with N − 1 = m and n − 1 = 2m)

q m ≤ 2m, |p| m ≤ 2m, |p + q| m ≤ 2m, hence

q ≤ 2, |p| ≤ 2, |p + q| ≤ 2. We examine the possibilities.

- Case q = 1. Then |p| ≤ 2 and |p + 1| ≤ 2. Since p ̸= 0 and p + 1 ̸= 0 (i.e. p ̸= −1), the admissible integers are p = 1 and p = −2.
- Case q = 2. Then gcd(p, 2) = 1, so p is odd. |p| ≤ 2 forces p = ±1. |p + 2| ≤ 2 eliminates p = 1 because |3| > 2; thus only p = −1 remains. Now we determine the intercept d so that exactly m + 1 integer points of Pn lie on the line. Subcase (1, 1): L : y = x + d. Let the integer points be (a + i, a + i + d) for i = 0, 1, . . . , m (since x increases by q = 1). The x-coordinates range from a to a + m and must satisfy 1 ≤ a + i ≤ 2m + 1. Hence 1 ≤ a ≤ m + 1. The constraints from the bounds on y and on x + y yield:

 

a + d ≥ 1, a + m + d ≤ 2m + 1, 2a + 2m + d ≤ 2m + 2.



The third inequality gives d ≤ 2 − 2a; the second gives d ≤ m + 1 − a; the first gives d ≥ 1 − a. For consistency we need 1 − a ≤ 2 − 2a, i.e. a ≤ 1. Thus a = 1. Then the bounds become 0 ≤ d ≤ min{m, 0} = 0, so d = 0. Hence L : y = x.

- Subcase (1, -2): L : y = −2x + d. Again the points are (a + i, −2(a + i) + d) with i = 0, . . . , m and 1 ≤ a ≤ m + 1. The tightest constraints come from y ≥ 1 (at i = m), y ≤ 2m + 1 (at i = 0), and x + y ≤ 2m + 2 (at i = 0):

−2(a + m) + d ≥ 1 =⇒ d ≥ 1 + 2a + 2m,

−2a + d ≤ 2m + 1 =⇒ d ≤ 2m + 1 + 2a, −a + d ≤ 2m + 2 =⇒ d ≤ 2m + 2 + a. The first two give

1 + 2a + 2m ≤ d ≤ 2m + 1 + 2a. Hence d must equal 2m + 1 + 2a (the only integer satisfying both). Substituting into the third inequality yields

2m + 1 + 2a ≤ 2m + 2 + a =⇒ a ≤ 1.

Since a ≥ 1, we have a = 1. Then d = 2m + 1 + 2 = 2m + 3 = n + 2. So L : y = −2x + (n + 2).

###### Subcase (2, -1): L : y = −12x + d.

The x-coordinates differ by q = 2. Let the smallest be a. Then the points are (a + 2i, −12(a + 2i) + d) for i = 0, . . . , m. To stay within the bounds 1 ≤ x ≤ 2m + 1 we need

a ≥ 1 and a + 2m ≤ 2m + 1 =⇒ a ≤ 1, so a = 1. Thus the x-coordinates are 1, 3, 5, . . . , 2m + 1. For the y-coordinates to be integers, d must have the form d = d′ + 12 with d′ ∈ Z. Then

yi = −21(2i + 1) + d = −i − 12 + d′ + 12 = d′ − i. Now apply the constraints:

▷ yi ≥ 1 for all i: the smallest y occurs at i = m (since slope negative), so d′ − m ≥ 1 ⇒ d′ ≥ m + 1.

▷ yi ≤ 2m + 1 for all i: the largest y occurs at i = 0, giving d′ ≤ 2m + 1.

▷ xi + yi = (2i + 1) + (d′ − i) = d′ + i + 1 ≤ 2m + 2: the most restrictive is at i = m, giving d′ + m + 1 ≤ 2m + 2 ⇒ d′ ≤ m + 1.

Thus d′ ≥ m + 1 and d′ ≤ m + 1, so d′ = m + 1. Then d = m + 1 + 12 = 2m2+3 = n+22 . Hence L : y = −12x + n+22 .

We have exhausted all possibilities; therefore any sunny line with exactly m + 1 points must be one of the three listed. □

Returning to the proof of Lemma 2 for odd n ≥ 5: we have n distinct sunny lines, each must contain exactly m + 1 points, hence each must be one of LA, LB, LC. But there are only three distinct such lines, while n ≥ 5 gives a contradiction. Hence no covering exists for odd n ≥ 5 either. (For n = 3 the three lines are exactly these three, and that case is covered by the construction.) This completes the proof of Lemma 2. □

###### 4. Induction on n

We now prove by strong induction that for every n ≥ 3 the number k of sunny lines in any covering of Pn by n distinct lines can only be 0, 1, 3. Base case n = 3. Constructions in 1 show that k = 0, 1, 3 are attainable. We must show that k = 2 is impossible. Assume three lines cover P3 with exactly two sunny. Let the sunny lines be L1, L2 and the third line L3 be non-sunny. Lemma 1 implies that each sunny line contains at most two points of P3. The only pairs of points in P3 that are collinear with a sunny line (i.e. with slope different from 0, undefined, or −1) are exactly the three pairs:

{(1, 1), (2, 2)} (slope 1), {(1, 3), (2, 1)} (slope − 2), {(1, 2), (3, 1)} (slope − 1/2). Hence any sunny line that contains two points must be one of the three lines

ℓ1 : y = x, ℓ2 : y = −2x + 5, ℓ3 : y = −12x + 52. Now consider the possibilities.

If both L1 and L2 contain two points and are disjoint. Then they must be two of the three lines ℓ1, ℓ2, ℓ3. Those three lines are pairwise disjoint, so they cover four points. The two remaining points lie on the third ℓ, which is sunny. The third line L3 is non-sunny, so it cannot cover both of those points (the unique line through them is sunny). Contradiction.

If at least one of L1, L2 contains at most one point. Then the two sunny lines together cover at most three points. Consequently, the non-sunny line L3 must cover at least three points. The only lines that contain three points of P3 are the three ”boundary” lines:

y = 1, x = 1, x + y = 4.

(One verifies that any other line contains at most two points.) So L3 is one of these three.

▷ L3 = y = 1: covers (1, 1), (2, 1), (3, 1). The uncovered points are (1, 2), (1, 3), (2, 2). Each of these three points lies on a different one of ℓ1, ℓ2, ℓ3 (namely (2, 2) ∈ ℓ1, (1, 3) ∈ ℓ2, (1, 2) ∈ ℓ3). Two sunny lines can cover at most two of them (each ℓi contains exactly one). Hence impossible.

▷ L3 = x = 1: covers (1, 1), (1, 2), (1, 3). Uncovered: (2, 1), (2, 2), (3, 1). Again these are one per ℓi; two sunny lines cannot cover all three. Contradiction.

▷ L3 = x+y = 4: covers (1, 3), (2, 2), (3, 1). Uncovered: (1, 1), (1, 2), (2, 1), again one per ℓi. Contradiction. Thus k = 2 is impossible, so the only possible values for n = 3 are 0, 1, 3. Inductive step. Assume the statement holds for all m with 3 ≤ m < n, where n ≥ 4. Consider any covering

of Pn by n distinct lines. Define three special lines:

###### R : y = 1, C : x = 1, D : x + y = n + 1.

Note that R, C, D are not sunny (horizontal, vertical, slope −1 respectively). Claim. At least one of R, C, D belongs to the covering. Proof of claim. Suppose none of them is present. Then:

▷ The line y = 1 contains n distinct points (a, 1). To cover a point (a, 1), the line must intersect y = 1 at that point. A horizontal line different from y = 1 does not intersect y = 1. Hence every line that covers a point on y = 1 must be non-horizontal. Moreover, a non-horizontal line meets y = 1 in at most one point. Therefore we need at least n non-horizontal lines; since there are exactly n lines, all lines are non-horizontal, and each covers exactly one point on y = 1.

▷ The line x = 1 contains n points (1, b). Since C is absent, a vertical line different from x = 1 does not contain any point with x = 1. Hence every line covering a point on x = 1 must be non-vertical. Since all lines are already non-horizontal, they must all be non-vertical to cover those points (a vertical line would miss x = 1 unless it is x = 1 itself). Thus all lines are non-vertical.

▷ The line x + y = n + 1 contains n points. Since D is absent, a line with slope −1 other than D is parallel to it and does not intersect it. Therefore every line covering a point on D must have slope different from −1. Hence all lines have slope ̸= −1.

Therefore every line is non-horizontal, non-vertical, and slope ̸= −1; i.e., every line is sunny. But Lemma 2 states that for n ≥ 4 there is no covering of Pn by n sunny lines. Contradiction. Hence the claim holds. □ Now we examine which of R, C, D is present. (If more than one is present, we may choose any one; the argument works in each case.)

If R is present: Remove R from the set. The remaining n − 1 lines still cover all points of Pn with y ≥ 2 (because R only covered points with y = 1). Apply the translation ϕR : (x, y)  → (x, y − 1). This map sends the set {(x, y) ∈ Pn | y ≥ 2} bijectively onto Pn−1. Moreover, ϕR sends each line to a line (translations preserve lines). The resulting family consists of n − 1 distinct lines (translation preserves distinctness) and covers Pn−1 (each point of Pn−1 corresponds to a point with y ≥ 2 that was covered). Since R is not sunny, the number of sunny lines among the n − 1 lines remains k.

Thus we have a covering of Pn−1 by n − 1 distinct lines with exactly k sunny lines. By the induction hypothesis (applied to m = n − 1), we must have k ∈ {0, 1, 3}.

- If C is present: Remove C and use ϕC : (x, y)  → (x − 1, y). It maps {(x, y) ∈ Pn | x ≥ 2} onto Pn−1. C is non-sunny, so k unchanged. Induction gives k ∈ {0, 1, 3}.
- If D is present: Remove D. The remaining n − 1 lines cover all points of Pn that are not on D. Observe that Pn−1 ⊆ Pn and Pn−1 ∩ D = ∅ (since points in Pn−1 satisfy a + b ≤ n, while D requires a + b = n + 1. Hence they cover Pn−1. D is not sunny, so k stays the same. Induction yields k ∈ {0, 1, 3}. In every possible case we conclude k ∈ {0, 1, 3}, which completes the inductive step. By strong induction, the statement holds for all integers n ≥ 3.

###### 5. Conclusion

We have shown that for every n ≥ 3 the only possible numbers of sunny lines in a covering of Pn by n distinct lines are 0, 1, and 3. Explicit constructions demonstrate that each of these values is indeed attainable.

##### IMO 2025 Problem 2.

|k ∈ {0, 1, 3} for all n ≥ 3.|
|---|

###### IMO 2025 Problem 2

Let Ω and Γ be circles with centres M and N, respectively, such that the radius of Ω is less than the radius of Γ. Suppose Ω and Γ intersect at two distinct points A and B. Line MN intersects Ω at C and Γ at D, so that C, M, N, D lie on MN in that order. Let P be the circumcenter of triangle ACD. Line AP meets Ω again at E ̸= A and meets Γ again at F ̸= A. Let H be the orthocenter of triangle PMN.

Prove that the line through H parallel to AP is tangent to the circumcircle of triangle BEF.

Solution Remark. 7/7.

We will prove the statement using analytic geometry. The configuration is invariant under similarity transformations, so we may fix convenient coordinates.

- 1. Coordinate setup Place M and N on the x-axis. Without loss of generality, set MN = 1 (scale the figure appropriately). Let

###### M = (0, 0), N = (1, 0).

Denote the radius of Ω (with center M) by r and the radius of Γ (with center N) by R, with 0 < r < R. Since the circles intersect in two distinct points, we have

###### R − r < 1 < R + r.

- 2. Intersection points A and B The equations of the circles are

x2 + y2 = r2, (Ω) (x − 1)2 + y2 = R2, (Γ).

Subtracting gives 2x − 1 = R2 − r2, so

x = a =

1 + r2 − R2 2

.

Let h > 0 be such that a2 + h2 = r2. The two intersections are symmetric about the x-axis; we take

A = (a, h), B = (a, −h).

- 3. Points C and D on line MN The line MN is the x-axis. Intersection of Ω with the x-axis: x2 = r2 ⇒ x = ±r. Because the order on the line is C, M, N, D, point C must lie left of M and D right of N. Hence

C = (−r, 0), D = (1 + R, 0).

- 4. Circumcenter P of △ACD Points C and D lie on the x-axis, so the perpendicular bisector of CD is the vertical line through the midpoint of CD:

midpoint = −r + 1 + R

2

, 0 =

1 + R − r 2

, 0 . Thus

xP =

1 + R − r 2

. To find yP, use PA = PC:

(xP − a)2 + (yP − h)2 = (xP + r)2 + yP2 . Expanding and using a2 + h2 = r2 yields

−2hyP = 2(r + a)xP =⇒ yP = −

(r + a)xP h

. Hence

P =

1 + R − r 2

, −

(r + a)xP h

.

- 5. Simplifying notation with auxiliary parameters Introduce

p = R + r, q = R − r, S = 1 + R + r = 1 + p. Then

p − q 2

- 1 − pq

- 2

p + q 2

- 1 + q

- 2

. From h2 = r2 − a2 we obtain

r =

, R =

, a =

, xP =

(p2 − 1)(1 − q2) 4

h2 =

. Also

S(1 − q) 2

r + a =

.

- 6. Vector −→AP Set ⃗v = −→AP = (v1, v2).

v1 = xP − a =

- 1 + q

- 2 −

- 1 − pq

- 2

=

q + pq 2

=

qS 2

. For v2:

v2 = yP − h = −

(r + a)xP h − h = −

(r + a)xP + h2 h

. Since h2 = (r − a)(r + a),

(r + a)xP + h2 = (r + a)(xP + r − a). Compute xP + r − a:

xP + r − a =

- 1 + q

- 2

+ r −

- 1 − pq

- 2

=

q + 2r + pq 2

. But 2r = p − q, so

q + 2r + pq = q + (p − q) + pq = p + pq = p(1 + q). Thus

xP + r − a =

p(1 + q) 2

. Now (r + a) =

S(1 − q) 2

, so

(r + a)(xP + r − a) =

S(1 − q) 2 ·

p(1 + q) 2

=

pS(1 − q2) 4

. Therefore

v2 = −

1 h ·

pS(1 − q2) 4

= −

pS(1 − q2) 4h

.

- 7. Points E and F (second intersections of AP with the circles) Parameterize line AP as A + λ⃗v. Substitute into Ω: |A + λ⃗v|2 = r2. Because |A|2 = r2, we get

2λ(⃗v · A) + λ2|⃗v|2 = 0. Hence λ = 0 (point A) or

2⃗v · A |⃗v|2

λE = −

.

Similarly, for Γ substitute into |(A + λ⃗v) − N|2 = R2 (since N = (1, 0)). |A − N|2 = R2, so

2⃗v · (A − N) |⃗v|2

λF = −

. Now compute the needed dot products.

pS(1 − q2) 4h

pS(1 − q2) 4

qS 2

aqS 2 −

S 4

2aq − p(1 − q2) . But 2aq = q(1 − pq) = q − pq2, so

⃗v · A = av1 + hv2 = a ·

+ h −

=

=

2aq − p(1 − q2) = q − pq2 − p + pq2 = q − p. Thus

S(p − q) 4

S · 2r 4

S 4

rS 2

⃗v · A =

(q − p) = −

= −

= −

. Next,

rS 2 −

⃗v · (A − N) = ⃗v · A − v1 = −

S(r + q) 2

qS 2

RS 2

= −

= −

,

because r + q = R. Consequently,

2 −rS2 |⃗v|2

2 −RS2 |⃗v|2

rS |⃗v|2

RS |⃗v|2

λE = −

, λF = −

. Hence

=

=

rS |⃗v|2

RS |⃗v|2

E = A +

⃗v, F = A +

⃗v.

- 8. Orthocenter H of △PMN Vertices: M = (0, 0), N = (1, 0), P = (xP, yP). Side MN is horizontal, so the altitude from P is the vertical line x = xP.

Altitude from M is the line through M perpendicular to PN. Vector −−→PN = (1 − xP, −yP). A vector perpendicular to PN is (yP, 1 − xP) (since (1 − xP)yP + (−yP)(1 − xP) = 0). Parametric equation: t (yP, 1 − xP).

The orthocenter H is the intersection of x = xP with this altitude. So set

tyP = xP =⇒ t =

xP yP

(yP ̸= 0). Then

H = xP, t(1 − xP) = xP,

xP(1 − xP) yP

.

Now substitute yP = −

(r + a)xP h

:

xP(1 − xP) yP

=

xP(1 − xP) −(r+ah)xP

= −

h(1 − xP) r + a

.

Compute 1 − xP = 1 −

- 1 + q

- 2

=

- 1 − q

- 2

. And r + a =

S(1 − q) 2

. Therefore

−

h(1 − xP) r + a

= −

h · 1−2q

S(1−q) 2

= −

h S

. Thus

H =

- 1 + q

- 2

, −

h S

.

- 9. Circumcenter O′ of △BEF We will determine O′ = (x, y) that is equidistant from B, E, F. Because E and F lie on line AP, the segment EF is collinear with ⃗v. The perpendicular bisector of EF consists of points X satisfying ⃗v · (X − MEF) = 0, where MEF is the midpoint of EF. The midpoint:

E + F 2

λE + λF 2

MEF =

= A +

⃗v.

(r + R)S |⃗v|2

pS |⃗v|2

,

Since λE + λF =

=

pS 2|⃗v|2

⃗v. Hence the condition ⃗v · X = ⃗v · MEF gives

MEF = A +

pS 2

⃗v · O′ = ⃗v · A +

. But ⃗v · A = −

rS 2

, so

(p − r)S 2

rS 2

pS 2

RS 2

⃗v · O′ = −

. (1) Next impose |O′ − B|2 = |O′ − E|2. Write U = O′ − A. Then

+

=

=

O′ − B = U − w, with w = B − A = (0, −2h), and

O′ − E = U − λE⃗v. Then

|U − w|2 = |U − λE⃗v|2. Expand:

|U|2 − 2U · w + |w|2 = |U|2 − 2λE⃗v · U + λ2E|⃗v|2. Cancel |U|2:

−2U · w + |w|2 = −2λE⃗v · U + λ2E|⃗v|2. (2) Now U · w = (x − a) · 0 + (y − h)(−2h) = −2h(y − h). Thus

−2U · w + |w|2 = −2[−2h(y − h)] + 4h2 = 4h(y − h) + 4h2 = 4hy. So (2) becomes

4hy = −2λE⃗v · U + λ2E|⃗v|2. (3) We have

(R + r)S 2

rS 2

pS 2

RS 2 − −

⃗v · U = ⃗v · (O′ − A) = ⃗v · O′ − ⃗v · A =

=

=

.

rS |⃗v|2

Recall λE =

. Substitute into (3):

2

rpS2 |⃗v|2

r2S2 |⃗v|2

rS2 |⃗v|2

rpS2 |⃗v|2

r2S2 |⃗v|2

rRS2 |⃗v|2

rS |⃗v|2

rS |⃗v|2

pS 2

|⃗v|2 = −

(since p−r = R). Thus

4hy = −2

·

(r−p) = −

= −

+

+

=

+

rRS2 |⃗v|2

rRS2 4h|⃗v|2

, so y = −

. (4)

4hy = −

###### 10. Computation of |⃗v|2 We have

pS(1 − q2) 4h

qS 2

, v2 = −

. Hence

v1 =

q2S2 4

v12 =

,

p2S2(1 − q2)2 16h2

v22 =

.

(p2 − 1)(1 − q2) 4

Using h2 =

,

p2S2(1 − q2) 4(p2 − 1)

v22 =

. Therefore

p2(1 − q2) p2 − 1

q2(p2 − 1) + p2(1 − q2) p2 − 1

S2 4

S2 4 ·

|⃗v|2 =

q2 +

. The numerator simplifies:

=

q2(p2 − 1) + p2(1 − q2) = p2 − q2. Thus

S2 4 ·

p2 − q2 p2 − 1

|⃗v|2 =

.

- Now p2 − q2 = (R + r)2 − (R − r)2 = 4Rr. And p2 − 1 = (p − 1)(p + 1) = (R + r − 1)(R + r + 1) = (R + r − 1)S. Hence

|⃗v|2 =

S2 4 ·

4Rr (R + r − 1)S

=

SRr R + r − 1

. (5)

11. Simplify y from (4) Insert (5) into (4):

y = −

rRS2 4h ·

R + r − 1 SRr

= −

S(R + r − 1) 4h

. (6)

12. Determine x from equation (1) Equation (1): ⃗v · O′ = v1x + v2y =

RS 2

. We know v1 =

qS 2

, and from (6) we have y. Compute v2y.

v2 = −

pS(1 − q2) 4h

, y = −

S(R + r − 1) 4h

. Thus

v2y =

pS2(1 − q2)(R + r − 1) 16h2

.

Recall 4h2 = (R + r − 1)S(1 − q2). Then 16h2 = 4(R + r − 1)S(1 − q2).

v2y =

pS2(1 − q2)(R + r − 1) 4(R + r − 1)S(1 − q2)

=

pS 4

. Now (1) becomes

qS 2

x +

pS 4

=

RS 2

. Divide by S ̸= 0:

q 2

x +

p 4

=

R 2

. Multiply by 4:

2qx + p = 2R.

- Now q = R − r, p = R + r, so

2(R − r)x + (R + r) = 2R =⇒ 2(R − r)x = 2R − (R + r) = R − r. Since R ̸= r (strict inequality), we obtain

R − r 2(R − r)

- 1

- 2

x =

. Therefore the circumcenter of △BEF is

=

O′ =

S(R + r − 1) 4h

- 1

- 2

, −

. (7)

- 13. Distance from O′ to the line through H parallel to ⃗v Let ℓ be the line through H with direction ⃗v. The distance from a point to a line with direction ⃗v through H is

d = |(O′ − H) × ⃗v| |⃗v|

,

where the cross product (in the plane) is taken as the scalar ∆x · v2 − ∆y · v1.

Coordinates:

- 1 + q

- 2

h S

, −

H =

. Compute differences:

- 1

- 2 −

- 1 + q

- 2

q 2

1 2 − xH =

= −

.

∆x =

S(R + r − 1) 4h − −

S(R + r − 1) 4h

h S

h S

∆y = yO′ − yH = −

= −

. Write as a single fraction:

+

∆y = −S2(R + r − 1) + 4h2 4hS

. But 4h2 = (R + r − 1)S(1 − q2). Hence

(R + r − 1)S (1 − q2) − S 4hS

(R + r − 1) (1 − q2) − S 4h

(R + r − 1)S(1 − q2) − S2(R + r − 1) 4hS

. Now S = 1 + p, so

∆y =

=

=

(1 − q2) − S = 1 − q2 − 1 − p = −p − q2. Thus

(R + r − 1)(p + q2) 4h

. (8) Now compute the cross product:

∆y = −

∆ × ⃗v = ∆x · v2 − ∆y · v1. ∆x · v2 = −

pS(1 − q2) 4h

qpS(1 − q2) 8h

q 2 −

=

.

(R + r − 1)(p + q2) 4h

qS(R + r − 1)(p + q2) 8h

qS 2

∆y · v1 = −

= −

. Therefore

qS(R + r − 1)(p + q2) 8h

qpS(1 − q2) 8h − −

qS 8h

p(1 − q2) + (R + r − 1)(p + q2) . But R + r − 1 = p − 1. So the bracket becomes

∆ × ⃗v =

=

p(1 − q2) + (p − 1)(p + q2). Expand:

p(1 − q2) = p − pq2, (p − 1)(p + q2) = p(p − 1) + (p − 1)q2 = p2 − p + pq2 − q2. Sum:

(p − pq2) + (p2 − p + pq2 − q2) = p2 − q2. Thus

- Since q = R − r,

∆ × ⃗v =

qS 8h

qS 8h · 4Rr =

qSRr 2h

(p2 − q2) =

.

(R − r)SRr 2h

. (9)

∆ × ⃗v =

###### 14. Squared distance from O′ to ℓ

2

(R − r)SRr 2h

(R − r)2S2R2r2 4h2 ·

1 |⃗v|2

d2 =

=

.

|⃗v|2

SRr R + r − 1

Substitute |⃗v|2 =

:

(R − r)2S2R2r2 4h2 ·

(R − r)2SRr(R + r − 1) 4h2

R + r − 1 SRr

d2 =

. (10) Now use 4h2 = (R + r − 1)S(1 − q2):

=

(R − r)2SRr(R + r − 1) (R + r − 1)S(1 − q2)

(R − r)2Rr 1 − (R − r)2

d2 =

. (11)

=

- 15. Radius of the circumcircle of △BEF Compute Rc = |O′ − B|, with B = (a, −h) and O′ as in (7).

∆x′ =

- 1

- 2 − a =

- 1

- 2 −

- 1 − pq

- 2

=

pq 2

.

∆y′ = yO′ + h = −

S(R + r − 1) 4h

+ h =

4h2 − S(R + r − 1) 4h

. Again, 4h2 = (R + r − 1)S(1 − q2),

∆y′ =

(R + r − 1)S(1 − q2) − S(R + r − 1) 4h

=

S(R + r − 1)(1 − q2 − 1) 4h

= −

S(R + r − 1)q2 4h

. Thus

Rc2 =

pq 2

2

+ −

S(R + r − 1)q2 4h

2

=

p2q2 4

+

S2(R + r − 1)2q4 16h2

. Substitute 16h2 = 4(R + r − 1)S(1 − q2):

Rc2 =

p2q2 4

+

S2(R + r − 1)2q4 4(R + r − 1)S(1 − q2)

=

p2q2 4

+

S(R + r − 1)q4 4(1 − q2)

. Write as

Rc2 =

q2 4

p2 +

S(R + r − 1)q2 1 − q2

. Now S(R + r − 1) = (p + 1)(p − 1) = p2 − 1. Therefore

Rc2 =

q2 4 ·

p2(1 − q2) + (p2 − 1)q2 1 − q2

=

q2 4 ·

p2 − p2q2 + p2q2 − q2 1 − q2

=

q2 4 ·

p2 − q2 1 − q2

.

- Since p2 − q2 = 4Rr and q2 = (R − r)2,

Rc2 =

(R − r)2 4 ·

4Rr 1 − (R − r)2

=

(R − r)2Rr 1 − (R − r)2

. (12)

- 16. Conclusion Comparing (11) and (12) we obtain

d2 = Rc2,

so d = Rc (all quantities are nonnegative). Hence the distance from the circumcenter O′ of △BEF to the line ℓ (through H parallel to AP) equals the radius of its circumcircle. Therefore ℓ is tangent to the circumcircle of △BEF.

##### IMO 2025 Problem 3.

###### IMO 2025 Problem 3

A function f : N → N is said to be bonza if

f(a) divides ba − f(b)f(a)

for all positive integers a and b. Determine the smallest real constant c such that f(n) ≤ cn for all bonza functions f and all positive integers n.

Solution Remark. 7/7.

- 1. Preliminary lemmas

- Lemma 1. f(1) = 1. Proof. Taking a = b = 1 gives f(1) | 11 − f(1)f(1) = 1 − f(1)f(1). Since f(1) | f(1)f(1), we have f(1) | 1. Hence f(1) = 1.
- Lemma 2. For every positive integer a, f(a) | aa. Proof. Put b = a in the definition: f(a) | aa − f(a)f(a). Because f(a) | f(a)f(a), it follows that f(a) | aa. Corollary. If a prime p divides f(a), then p divides a. Proof. From f(a) | aa, any prime divisor of f(a) must divide aa, hence a.

- 2. Determination of f(2), f(3) and f(5) From Lemma 2 we have:

- f(2) | 22 = 4 =⇒ f(2) ∈ {1, 2, 4},
- f(3) | 33 = 27 =⇒ f(3) ∈ {1, 3, 9, 27}.

f(5) | 55 = 3125 =⇒ f(5) = 5δ with 0 ≤ δ ≤ 5. Now use the mixed conditions (a, b) = (2, 3) and (3, 2):

- (2, 3) : f(2) | 32 − f(3)f(2) = 9 − f(3)f(2),
- (3, 2) : f(3) | 23 − f(2)f(3) = 8 − f(2)f(3).

Let x = f(2), y = f(3). We test all x ∈ {1, 2, 4}, y ∈ {1, 3, 9, 27} (note all y are odd).

- ▷ x = 1: (2, 3) automatic. (3, 2) gives y | 8 − 1y = 7. Only y = 1 divides 7. So (1, 1) works.
- ▷ x = 2:

- (2, 3): 2 | 9 − y2. Since y is odd, y2 is odd, 9 − odd is even - holds for all odd y.
- (3, 2): y | 8 − 2y.

▷ y = 1: 1 | 6 - true.

▷ y = 3: 3 | 8 − 8 = 0 - true.

▷ y = 9: 9 | 8 − 512 = −504 - true (−504/9 = −56).

▷ y = 27: compute 227 mod 27. 29 ≡ 512 ≡ 26 (mod 27), then 227 = (29)3 ≡ 263 ≡ (−1)3 = −1 ≡ 26 (mod 27). So 8 − 26 = −18 ̸≡ 0 (mod 27) - false. Thus for x = 2 we have (2, 1), (2, 3), (2, 9).

- ▷ x = 4: (2, 3): 4 | 9 − y4. For odd y, y4 ≡ 1 (mod 4), so 9 − 1 = 8 divisible by 4 - holds for all odd y. (3, 2): y | 8 − 4y.
- ▷ y = 1: 1 | 8 − 4 = 4 - true.

▷ y = 3: 3 | 8 − 64 = −56 - false.

▷ y = 9: 43 ≡ 64 ≡ 1 (mod 9), so 49 ≡ 1 (mod 9), then 8 − 1 = 7 ̸≡ 0 (mod 9) - false.

▷ y = 27: 43 ≡ 64 ≡ 10 (mod 27), so 49 ≡ 103 = 1000 ≡ 1 (mod 27), thus 427 ≡ 1 (mod 27), 8 − 1 = 7 ̸≡ 0 (mod 27) - false.

Hence only (4, 1) works. Therefore after (2, 3) and (3, 2) the possible pairs are

(1, 1), (2, 1), (2, 3), (2, 9), (4, 1). (1) Now incorporate f(5) = 5δ. For each candidate pair we must satisfy

- (2, 5) : x | 25 − 5δx,
- (3, 5) : y | 125 − 5δy,

- (5, 2) : 5δ | 32 − x5

δ

,

- (5, 3) : 5δ | 243 − y5

δ

. We test δ = 0, 1, 2, 3, 4, 5 for each pair.

- 1. Pair (x, y) = (1, 1).

▷ (2, 5) automatic. ▷ (3, 5) automatic.

- ▷ (5, 2): 5δ | 32 − 15δ = 31. So 5δ | 31 forces δ = 0.
- ▷ (5, 3): 5δ | 243 − 15δ = 242. Again only δ = 0 works.

⇒ (f(2), f(3), f(5)) = (1, 1, 1).

- 2. Pair (x, y) = (2, 1).

- ▷ (2, 5): 2 | 25 − 52δ - difference even, automatic.
- ▷ (3, 5): 1 divides everything.

▷ (5, 3): 5δ | 243 − 1 = 242. Since 5 ∤ 242, only δ = 0.

- ▷ (5, 2) with δ = 0: 1 | 32 − 2 = 30 - true.

⇒ (2, 1, 1).

3. Pair (x, y) = (4, 1).

- ▷ (2, 5): 4 | 25 − 52δ. Mod 4: 25 ≡ 1, 52δ ≡ 1, difference divisible by 4 - automatic.
- ▷ (3, 5): automatic.

- ▷ (5, 3): 5δ | 242 - forces δ = 0.

- 4. Pair (x, y) = (2, 3).

- ▷ (2, 5): automatic (even difference).
- ▷ (3, 5): 3 | 125 − 53δ. ▷ (5, 2): 5δ | 32 − 25δ. ▷ (5, 3): 5δ | 243 − 35δ.

Test δ:

- ▷ δ = 0: f(5) = 1. Then 3 | 125 − 1 = 124 - false.
- ▷ δ = 1: f(5) = 5.

▷ 3 | 125 − 53 = 0 - true.

▷ 5 | 32 − 25 = 0 - true.

▷ 5 | 243 − 35 = 0 - true. So δ = 1 works.

- ▷ δ = 2: f(5) = 25. 3 | 125 − 56. Mod 3: 125 ≡ 2, 56 ≡ (53)2 ≡ 22 = 4 ≡ 1, so difference 2 − 1 = 1 false.
- ▷ δ = 3: f(5) = 125.

▷ 3 | 125 − 59: 59 ≡ 53 ≡ 125 ≡ 2 (mod 3), difference 0 - true.

▷ Check (5, 2): 125 | 32 − 2125. Compute 2125 mod 125: φ(125) = 100, so 2125 ≡ 225 (mod 125). 25 = 32, 210 ≡ 24, 220 ≡ 242 = 576 ≡ 576 − 5 · 125 = 76, then 225 = 220 · 25 ≡ 76 · 32 = 2432 ≡ 2432 − 19 · 125 = 57. So 32 − 57 = −25 not divisible by 125 - false.

- ▷ δ = 4: f(5) = 625. 3 | 125 − 512. 512 ≡ (56)2 ≡ 12 = 1 (mod 3), 125 ≡ 2, difference 1 - false.
- ▷ δ = 5: f(5) = 3125.

▷ 3 | 125 − 515: 515 ≡ 53 ≡ 125 ≡ 2 (mod 3), difference 0 - true.

▷ (5, 2): 3125 | 32 − 23125. Mod 125: 3125 ≡ 25 (mod 100), so 23125 ≡ 225 ≡ 57 (mod 125) as before, 32 − 57 = −25 not divisible by 125 - false. Thus only δ = 1 works, giving (2, 3, 5).

- 5. Pair (x, y) = (2, 9). Test δ:

▷ (5, 2) with δ = 0: 1 | 32 − 4 = 28 - true.

⇒ (4, 1, 1).

- ▷ δ = 0: 3 | 125 − 1 = 124 - false.
- ▷ δ = 1: 5 | 125 − 59. 59 mod 9: 56 ≡ 1 (mod 9) (since φ(9) = 6), so 59 ≡ 53 = 125 ≡ 8 (mod 9). 125 ≡ 8, difference 0 - true. (5, 2): 5 | 32 − 25 = 0 - true. (5, 3): 5 | 243 − 95. 9 ≡ 4 (mod 5), 45 = 1024 ≡ 4 (mod 5). 243 ≡ 3 (mod 5), so 3 − 4 = −1 ̸≡ 0 (mod 5) - false.

- ▷ δ = 2: 25 | 125 − 518. 518 ≡ (56)3 ≡ 13 = 1 (mod 9), 125 ≡ 8, difference 7 - false.
- ▷ δ = 3: 125 | 125 − 527. 527 ≡ 53 ≡ 8 (mod 9), difference 0 - true. (5, 2): 125 | 32 − 2125. As before, 2125 ≡ 57 (mod 125), 32 − 57 = −25 - false.
- ▷ δ = 4: 625 | 125 − 536. 36 ≡ 0 (mod 6), so 536 ≡ 1 (mod 9), difference 7 - false.
- ▷ δ = 5: 3125 | 125 − 545. 45 ≡ 3 (mod 6), 545 ≡ 8 (mod 9), difference 0 - true. (5, 2): 3125 | 32 − 23125. Mod 125 fails as before. Hence no δ satisfies all conditions. Conclusion of classification. The only admissible triples (f(2), f(3), f(5)) are

(1, 1, 1), (2, 1, 1), (4, 1, 1), (2, 3, 5). We will analyze two families:

- ▷ Family A: those with f(3) = 1, i.e. the first three triples.
- ▷ Family B: the triple (2, 3, 5).

###### 3. Analysis of Family A (f(3) = 1)

- Lemma 3. If f(3) = 1, then f(q) = 1 for every odd prime q. Proof. From (q, q) we have f(q) | qq, so f(q) = qk for some 0 ≤ k ≤ q. Consider (q, 3):

f(q) | 3q − f(3)f(q) = 3q − 1f(q) = 3q − 1. If k ≥ 1, then q | f(q) | 3q − 1. By Fermat’s little theorem, 3q ≡ 3 (mod q), so q | 3 − 1 = 2. Hence q = 2, contradicting that q is odd. Therefore k = 0 and f(q) = 1.

- Lemma 4. In Family A, for every positive integer n, f(n) is a power of 2. Proof. Suppose an odd prime r divides f(n). By Lemma 2, r | n. Now use (n, r):

f(n) | rn − f(r)f(n). Since r is odd, Lemma 3 gives f(r) = 1. Hence

f(n) | rn − 1.

In particular, r | rn − 1, i.e. r | −1, contradiction. Thus no odd prime divides f(n); i.e. f(n) = 2e(n) for some e(n) ≥ 0.

- Lemma 5. For any n, 2e(n) | 3n − 1. Proof. Take (n, 3):

f(n) | 3n − f(3)f(n) = 3n − 1f(n) = 3n − 1. Since f(n) = 2e(n), we have 2e(n) | 3n − 1.

- Lemma 6. For n ∈ N,

Proof. Odd n: Write

v2(3n − 1) =

1 if n is odd, v2(n) + 2 if n is even.

3n − 1 = (3 − 1)(3n−1 + 3n−2 + · · · + 1) = 2 · (odd sum), because there are n terms, each odd, and n odd ⇒ the sum is odd. Hence v2 = 1. Even n: Write n = 2αm with m odd, α ≥ 1. We prove by induction on α that v2(32αm − 1) = α + 2. Base α = 1: n = 2m. Then

32m − 1 = (3m − 1)(3m + 1). We know v2(3m − 1) = 1 (since m odd). For 3m + 1, note 3m ≡ 3 (mod 8) when m is odd, so 3m + 1 ≡ 4 (mod 8), thus v2(3m + 1) = 2. Hence

v2(32m − 1) = 1 + 2 = 3 = 1 + 2 = α + 2.

Inductive step: Assume true for a given α ≥ 1. Write n′ = 2α+1m = 2 · (2αm). Let k = 2αm. Then

′

3n

− 1 = (3k − 1)(3k + 1).

By hypothesis, v2(3k − 1) = α + 2. Since k is even (because α ≥ 1), 3k ≡ 1 (mod 4), so 3k + 1 ≡ 2 (mod 4), giving v2(3k + 1) = 1.

Thus

v2(3n

This completes the induction. From Lemmas 5 and 6 we obtain

′

− 1) = (α + 2) + 1 = α + 3 = (α + 1) + 2.

Consequently

e(n) ≤

1 if n is odd, v2(n) + 2 if n is even.

f(n) = 2e(n) ≤

2 if n odd, 2v2(n)+2 if n even.

For even n, 2v2(n)+2 = 4 · 2v2(n) ≤ 4n because 2v2(n) | n implies 2v2(n) ≤ n. For odd n, 2 ≤ 4n (since n ≥ 1).

- Hence in Family A we have

|f(n) ≤ 4n for all n ∈ N|
|---|

.

- 4. Analysis of Family B (f(2) = 2, f(3) = 3, f(5) = 5)

- Proposition 7. For every prime p, f(p) = p. Proof. We proceed by strong induction on p. The base cases p = 2, 3, 5 are given. Assume p > 5 and that for every prime q < p we already know f(q) = q. From (p, p) we have f(p) | pp, so f(p) = pk for some integer k with 0 ≤ k ≤ p. We show that k must be 1.

- Step 1 - Eliminate k = 0 and k = 2. Take any prime q < p. Consider (q, p):

f(q) = q | pq − f(p)q = pq − pkq. Factor out pq:

q | pq(1 − p(k−1)q). Since q ̸= p, q ∤ pq. Hence

q | 1 − p(k−1)q.

By Fermat’s little theorem, pq ≡ p (mod q), and more generally p(k−1)q ≡ pk−1 (mod q) (raising to the qth power is the identity modulo q). Thus

1 − pk−1 ≡ 0 (mod q) =⇒ pk−1 ≡ 1 (mod q). (2) ▷ If k = 0, then the original condition (q, p) gives q | pq − 1. Using FLT, pq ≡ p (mod q), so p ≡ 1 (mod q).

▷ If k = 2, (2) gives p1 ≡ 1 (mod q), i.e. p ≡ 1 (mod q). Thus for both k = 0 and k = 2 we have

p ≡ 1 (mod q) for every prime q < p. (3) Now by Bertrand’s postulate (for any integer m > 1 there exists a prime between m and 2m), for p > 3 there exists a prime q with p2 < q < p. From (3), q | p − 1. But q > p/2 and q ≤ p − 1 (since q < p). The only positive multiple of q that is at most p − 1 is q itself (because 2q > p). Hence p − 1 = q, so p = q + 1, which is even - contradicting that p > 2 is odd. Therefore k cannot be 0 or 2.

- Step 2 - Eliminate k ≥ 3. Consider the pair (p, p − 1):

k

f(p) = pk | (p − 1)p − f(p − 1)p

. Let r = f(p − 1).

###### ▷ Modulo p:

(p − 1)p ≡ (−1)p = −1 (mod p),

and by Fermat’s little theorem, rpk ≡ r (mod p) (since rp ≡ r (mod p), and then by induction rpk ≡ r (mod p).

Since pk | D := (p − 1)p − rpk, in particular p | D. Reducing modulo p gives

−1 ≡ r (mod p), i.e. r ≡ −1 (mod p). (4) Write r = −1 + pt for some integer t.

▷ Expansion of (p − 1)p: By the binomial theorem,

p

p i

pi(−1)p−i.

(p − 1)p =

i=0

The term with i = 0 is (−1)p = −1. The term with i = 1 is p1 p1(−1)p−1 = p · p · 1 = p2. For i ≥ 2, pi is divisible by p, and together with pi yields a factor pi+1 ≥ p3. Thus we can write

(p − 1)p = −1 + p2A, where A is an integer and, because the i = 1 term contributes exactly p2 and the higher terms are multiples of p3, we have A ≡ 1 (mod p) (in particular p ∤ A).

▷ Expansion of rpk: Write r = −1 + pt. Then

pk

pk j

k

(−1)p

rp

=

j=0

Since pk is odd, (−1)pk = −1.

- ▷ j = 0: term = −1.
- ▷ j = 1: p1k (−1)pk−1pt = pk · 1 · pt = pk+1t.

▷ For j ≥ 2: we show that each term is divisible by pk+2. Indeed, for j ≥ 2,

k−j(pt)j.

pk j

= k − vp(j), hence

vp

pk j

pj = (k − vp(j)) + j = k + j − vp(j) ≥ k + 2,

vp

because j ≥ 2 and if p ∤ j then j ≥ 2 and vp(j) = 0; if p | j then j ≥ p and vp(j) ≥ 1, so j − vp(j) ≥ j − (j − 1) = 1? Wait, need to be precise: For any j with 2 ≤ j ≤ pk − 1, we have

pk j

= k − vp(j) (a standard property for prime powers). Then

vp

vp

pk j

pj = k − vp(j) + j = k + j − vp(j).

- Since j ≥ 2 and vp(j) ≤ logp j < j, but we need at least k + 2. The minimal value occurs when j = 2 and p ∤ 2 (since p ≥ 3), then vp(j) = 0, giving k + 2. For j = p we have vp(j) = 1, then k + p − 1 ≥ k + 2 because p ≥ 3. So indeed each term for j ≥ 2 is divisible by pk+2. Therefore

k

= −1 + pk+1t + pk+2S for some integer S.

rp

###### ▷ Form the difference:

k

= (−1 + p2A) − (−1 + pk+1t + pk+2S) = p2A − pk+1t − pk+2S. Factor p2:

D = (p − 1)p − rp

D = p2 A − pk−1t − pkS . (5)

- Since k ≥ 3, we have k − 1 ≥ 2, so pk−1t and pkS are both multiples of p2. Moreover,

A − pk−1t − pkS ≡ A (mod p),

and p ∤ A. Hence the expression in parentheses is not divisible by p, and we obtain

vp(D) = 2. However, we require vp(D) ≥ k ≥ 3 (because pk | D). This contradiction shows that k ≥ 3 is impossible. Since k cannot be 0, 2, or ≥ 3, the only remaining possibility is k = 1. Thus f(p) = p.

- Proposition 8. In Family B, f(n) = n for all positive integers n. Proof. Fix any n and any prime p. Using the bonza condition with a = p, b = n:

f(p) = p | np − f(n)p. By Fermat’s little theorem,

np ≡ n (mod p), f(n)p ≡ f(n) (mod p). Therefore

p | n − f(n).

This holds for every prime p. The only integer divisible by all primes is 0, so n − f(n) = 0, i.e. f(n) = n.

- Hence in Family B we have f(n) = n ≤ 4n for all n.

###### 5. Upper bound summary We have shown:

- ▷ In Family A, f(n) ≤ 4n for all n.
- ▷ In Family B, f(n) = n ≤ 4n. Therefore for every bonza function f and every n ∈ N,

|f(n) ≤ 4n|
|---|

.

###### 6. Sharpness: a bonza function attaining 4n Define a function h : N → N by

h(1) = 1, and for n ≥ 2:

 

1 if n is odd, 4 if n = 2, 2v2(n)+2 if n is even and n > 2.

h(n) =



We verify that h is bonza, i.e.,

###### h(a) | ba − h(b)h(a) for all a, b ∈ N.

Verification We consider cases based on a.

###### ▷ Case 1: a is odd.

- Then h(a) = 1 (by definition: for odd n > 1, h(n) = 1; and a = 1 gives 1). Since 1 divides any integer, the condition holds trivially.

###### ▷ Case 2: a = 2. Here h(2) = 4. We must show

4 | b2 − h(b)4 for all b. ▷ If b is odd: b2 is odd, h(b) = 1, so h(b)4 = 1. Then b2 − 1 is divisible by 4 because for an odd b, b2 ≡ 1 (mod 8), hence certainly by 4.

▷ If b = 2: b2 = 4, h(b) = 4, h(b)4 = 44 = 256. Then 4 − 256 = −252, and −252 is divisible by 4.

▷ If b is even and b > 2: write b = 2βc with β ≥ 1, c odd. Then h(b) = 2β+2 (since b > 2 even). Then h(b)4 = 24(β+2) = 24β+8, which is divisible by 4. Moreover, b2 is divisible by 4 (since b is even). Thus b2 − h(b)4 is a difference of two multiples of 4, hence itself a multiple of 4.

Therefore the condition holds for a = 2.

- ▷ Case 3: a is even and a > 2. Let α = v2(a) (so α ≥ 1; note that if a is even and > 2, then either α ≥ 2 or α = 1 with a = 2m, m odd ≥ 3).

- Then h(a) = 2α+2. We need to prove

2α+2 | ba − h(b)2

α+2

. We split into subcases according to b.

- ▷ Subcase 3a: b is odd.

Then h(b) = 1. So we need 2α+2 | ba − 1. Since a is even, write a = 2αm with m odd. Claim: For any odd integer b and any integer s ≥ 1,

b2

s

≡ 1 (mod 2s+2). Proof of claim. For s = 1, any odd b satisfies b2 ≡ 1 (mod 8), i.e. modulo 23. Assume the claim holds for some s ≥ 1. Then

b2

s

= 1 + 2s+2K for some integer K. Squaring gives

b2

s+1

= (1 + 2s+2K)2 = 1 + 2s+3K + 22s+4K2 ≡ 1 (mod 2s+3), because the extra term is divisible by 2s+3. This proves the claim. Applying the claim with s = α yields

b2

α

≡ 1 (mod 2α+2). Then

ba = b2

α m ≡ 1m = 1 (mod 2α+2), and therefore ba − 1 is divisible by 2α+2.

- ▷ Subcase 3b: b = 1. Then ba = 1, h(b) = h(1) = 1, so h(b)2α+2 = 1, and 1 − 1 = 0 is divisible by any integer.
- ▷ Subcase 3c: b = 2. Here h(b) = h(2) = 4. So we need

2α+2 | 2a − 42

α+2

= 2a − (22)2

α+2

= 2a − 22·2

α+2

= 2a − 22

α+3

. Let D = 2a − 22α+3. The 2-adic valuation of a difference of two powers of two is

v2(D) =

 



a if a < 2α+3, 2α+3 if a > 2α+3, higher if a = 2α+3 (then D = 0).

We claim v2(D) ≥ α + 2. Lemma 9. If a is even and a > 2, then a ≥ α + 2 where α = v2(a). Proof. Write a = 2α · m with m odd.

▷ If α = 1, then a = 2m with m odd. Since a > 2, we have m ≥ 3, so a ≥ 6. Meanwhile α + 2 = 3. So a ≥ 3.

▷ If α ≥ 2, then a ≥ 2α. It remains to show 2α ≥ α + 2 for α ≥ 2. This is true for α = 2 (4 ≥ 4), and if true for α, then 2α+1 = 2 · 2α ≥ 2(α + 2) = 2α + 4 ≥ (α + 1) + 2 for α ≥ −1, which holds. Hence a ≥ α + 2. Now we have two possibilities:

▷ If a ≤ 2α+3, then v2(D) = a (if a < 2α+3) or D = 0 (if a = 2α+3). In either case, since a ≥ α + 2, we get v2(D) ≥ α + 2.

▷ If a > 2α+3, then v2(D) = 2α+3. Since α ≥ 1, 2α+3 ≥ 24 = 16 for α = 1, and α + 2 = 3; clearly 2α+3 ≥ α + 2 (e.g. for α = 1, 16 ≥ 3; for larger α it’s even larger). Thus v2(D) ≥ α + 2. Therefore 2α+2 | D.

- ▷ Subcase 3d: b is even and b > 2.

Write b = 2βc with β = v2(b) ≥ 1 and c odd. Since b > 2, we have β ≥ 1; note that if b = 2 we already handled.

- Then h(b) = 2β+2 (by definition, because b is even and b > 2). We need to show 2α+2 divides

α+2

α+2

ba − 2β+2 2

= ba − 2(β+2)2

. Observe that the second term is a power of two, and its exponent is (β + 2)2α+2 ≥ 2α+2 (since β ≥ 1 gives β + 2 ≥ 3). Hence it is certainly divisible by 2α+2. For the first term,

ba = (2βc)a = 2βaca.

Since ca is odd, the 2-adic valuation of ba is βa. By Lemma 9, a ≥ α + 2. Therefore

βa ≥ a ≥ α + 2.

Thus 2α+2 | ba. Consequently both ba and the second term are multiples of 2α+2, so their difference is also a multiple of 2α+2. All subcases are exhausted, and in each the required divisibility holds. Thus h is bonza. Now evaluate h(n) for powers of two:

▷ h(2) = 4 = 2 · 2.

▷ For any t ≥ 2, n = 2t is even and > 2, so

t)+2 = 2t+2 = 4 · 2t = 4n. In particular, h(4) = 16 = 4 · 4. Hence

h(2t) = 2v2(2

h(n) n

= 4 for all n = 2t (t ≥ 2). If c < 4, then for n = 4 (or any 2t with t ≥ 2) we have

h(4) = 16 > c · 4,

so c cannot satisfy f(n) ≤ cn for all bonza functions. Therefore no constant smaller than 4 works.

- 7. Conclusion We have proved that 4 is an upper bound for f(n)/n and that this bound is attainable. Consequently, the smallest real constant c with the property that f(n) ≤ cn for every bonza function f and every n ∈ N is

|4|
|---|

.

##### IMO 2025 Problem 4.

###### IMO 2025 Problem 4

An infinite sequence a1, a2, ... consists of positive integers has each of which has at least three proper divisors. Suppose that for each n ≥ 1, an+1 is the sum of the three largest proper divisors of an. Determine all possible values of a1.

Solution Remark. 7/7.

Let a1, a2, . . . be an infinite sequence of positive integers such that each an has at least three proper divisors, and for each n ≥ 1

an+1 = sum of the three largest proper divisors of an. Define the function f by f(N) = sum of the three largest proper divisors of N.

- 1. A useful formula for f(N) Let the positive divisors of N be

1 = d1 < d2 < · · · < dk = N, where k ≥ 4 because N has at least three proper divisors. The three largest proper divisors are dk−3, dk−2, dk−1. Using the divisor pairing di · dk+1−i = N, we get

N d2

N d3

N d4

. Thus

dk−1 =

, dk−2 =

, dk−3 =

N d2

N d3

f(N) =

+

N d4

+

= N

1 d2

1 d3

1 d4

+

+

. (1)

- 2. Fixed points A fixed point satisfies f(N) = N. From (1) this is equivalent to

1 d2

+

1 d3

+

1 d4

= 1. (2)

Let a = d2, b = d3, c = d4 (so 2 ≤ a < b < c). Solve 1/a + 1/b + 1/c = 1.

- ▷ If a = 2, then 1/b + 1/c = 1/2. Multiplying gives 2(b + c) = bc or (b − 2)(c − 2) = 4. With b < c, the only solution is b − 2 = 1, c − 2 = 4, i.e., (a, b, c) = (2, 3, 6).
- ▷ If a = 3, then 1/b + 1/c = 2/3. But the maximum for b ≥ 4, c ≥ 5 is 1/4 + 1/5 = 9/20 < 2/3, so no solution.

▷ If a ≥ 4, the sum is at most 1/4 + 1/5 + 1/6 = 37/60 < 1. Hence the unique triple is (2, 3, 6). Therefore a fixed point must have

d2 = 2, d3 = 3, d4 = 6. Interpretation:

- ▷ 2 | N (so N even),
- ▷ 3 | N,

▷ There is no divisor between 3 and 6; i.e., 4 ∤ N and 5 ∤ N.

▷ Because 2 is the smallest divisor > 1, we must have ν2(N) = 1 (otherwise 4 | N would be a divisor < 6). Thus the set F of all fixed points is

|F = N ∈ N ν2(N) = 1, 3 | N, 5 ∤ N .|
|---|

- 3. Special case: 12 | N If 12 | N, then 2, 3, 4 | N and these are the three smallest proper divisors (since 4 is the smallest possible after 2, 3). Then

f(N) =

N 2

+

N 3

+

N 4

=

13 12

N. (3)

- 4. Lemma on odd numbers

- Lemma 1. Let X be an odd positive integer with at least three proper divisors. Then f(X) < X and f(X) is odd. Proof. All divisors of an odd number are odd. The three smallest divisors greater than 1 are at least 3, 5, 7. Therefore

1 d2

+

1 d3

+

1 d4 ≤

1 3

+

1 5

+

1 7

=

71 105

< 1,

- so f(X) < X. Moreover, each quotient X/di is odd (odd divided by odd). The sum of three odd numbers is odd, hence f(X) is odd. An immediate corollary: In an infinite orbit, no term can be odd, because starting from an odd term the sequence would be strictly decreasing and infinite - impossible. Hence every term of an infinite sequence is even.

5. Lemma for even numbers not divisible by 12

- Lemma 2. Let Y be an even integer with at least three proper divisors and 12 ∤ Y . If the orbit of Y is infinite, then Y ∈ F. Proof. Since 12 ∤ Y and Y is even, we have either 4 ∤ Y or 3 ∤ Y (or both). Consider two cases.

###### Case 1: 3 | Y

Because 12 ∤ Y , we must have 4 ∤ Y ; thus ν2(Y ) = 1. Write Y = 2M with M odd. Since 3 | Y and gcd(2, 3) = 1, we get 3 | M.

The three smallest proper divisors are d2 = 2 and d3 = 3. The fourth divisor d4 depends on 5.

- ▷ Subcase 1a: 5 | M. Then 5 | Y and 5 < 6, so d4 = 5. Then

f(Y ) =

Y 2

+

Y 3

+

Y 5

=

31 30

Y > Y.

Moreover, Y is divisible by 30 (it contains factors 2, 3, 5), so Y = 30Z with Z odd. Then f(Y ) = 31Z, which is odd. By Lemma 1, the orbit from an odd number is strictly decreasing and finite - contradiction to infinite orbit. Hence this subcase cannot occur.

- ▷ Subcase 1b: 5 ∤ M. Then no divisor equals 4 or 5; the next divisor after 2, 3 is 6 = 2 · 3. Hence d4 = 6, and

Y 2

Y 3

Y 6

f(Y ) =

+

+

= Y.

Thus Y is a fixed point, i.e., Y ∈ F. Therefore, if the orbit is infinite, we must be in Subcase 1b, so Y ∈ F.

- Case 2: 3 ∤ Y Then Y is even but not divisible by 3. For any such Y we have

1 d2

1 d3

1 d4 ≤

- 1

- 2

1 4

1 5

- 19

- 20

+

+

+

+

=

< 1,

- so f(Y ) < Y . If the orbit never contained a term divisible by 3, then we would have an infinite strictly decreasing sequence - impossible. Hence there exists a smallest index m ≥ 2 with 3 | am. Then am−1 is not divisible by 3, and am = f(am−1) < am−1. Now examine am−1.

- ▷ Subcase 2a: am−1 is odd. By Lemma 1, am = f(am−1) is odd and less than am−1. Since am is odd and divisible by 3, Lemma 1 again implies that from am onward the sequence is strictly decreasing and odd - finite, contradiction.
- ▷ Subcase 2b: am−1 is even. Write Y = am−1 = 2M, with M odd. Since 3 ∤ Y , we have 3 ∤ M. We will show that am is odd, reducing to Subcase 2a and giving a contradiction. Let r = Y mod 3 (r = 1 or 2). For any divisor d of Y (which is coprime to 3), we have d ≡ 1 or 2 (mod 3) and d−1 ≡ d (mod 3) (because 1 · 1 ≡ 1, 2 · 2 ≡ 1 (mod 3)). Hence

Y d ≡ r · d (mod 3).

Thus

###### f(Y ) ≡ r d2 + d3 + d4 (mod 3).

- Since r ̸= 0, the condition 3 | f(Y ) is equivalent to

###### 2 + d3 + d4 ≡ 0 (mod 3). (4)

Now, d2 = 2. Determine d3, d4. First, 4 cannot divide Y . If 4 | Y , then d3 = 4, and (4) gives 2 + 4 + d4 ≡ 0 ⇒ d4 ≡ 0 (mod 3), impossible because 3 ∤ Y . So ν2(Y ) = 1: Y = 2M with M odd, and 4 ∤ Y . Thus the smallest divisor greater than 2 is an odd number; call it p. Since 3 ∤ Y , p ̸= 3, so p ≥ 5. The next divisor d4 is the smallest divisor larger than p. Since 4 ∤ Y , the next even divisor would be 2p. So either

▷ d4 = q (an odd divisor, p < q < 2p) if such an odd divisor exists, or

▷ d4 = 2p (if there is no odd divisor between p and 2p). Check the two possibilities against (4) with d3 = p:

2 + p + d4 ≡ 0 (mod 3). (5) ▷ If d4 = 2p, then 2 + p + 2p = 2 + 3p ≡ 2 (mod 3), not 0. Hence d4 cannot be 2p. Therefore there exists an odd divisor q with p < q < 2p and d4 = q. Then (5) becomes 2 + p + q ≡ 0 (mod 3). Since p, q ̸≡ 0 (mod 3), the only way is

p ≡ q ≡ 2 (mod 3). Now compute f(Y ):

Y p

Y q

2M p

2M q

Y 2

+

+

= M +

+

.

f(Y ) =

Because p, q divide M (they are odd divisors of Y = 2M), the fractions are integers. Now

▷ M is odd,

###### ▷ 2Mp = 2 · Mp is even,

▷ 2Mq is even. Thus f(Y ) = odd + even + even = odd. Hence am = f(Y ) is odd. But 3 | am (by definition of m). So am is an odd multiple of 3. By Lemma 1, the orbit from am is strictly decreasing and odd - finite, contradiction. Both subcases lead to contradiction. Therefore Case 2 cannot occur. The only possibility is Case 1, and specifically Subcase 1b, which forces Y ∈ F.

- 6. Necessity - form of a1 in an infinite orbit Assume the sequence is infinite. Let N = a1. Define

t = max{k ≥ 0 | 12k | N}.

Write N = 12tR with 12 ∤ R. We claim that for i = 1, . . . , t, the term ai is divisible by 12. Proof by induction: a1 = 12tR is divisible by 12 (if t ≥ 1). Suppose ai is divisible by 12. Since 12 | ai, we have by (3) that ai+1 = 1312ai. If we write ai = 12s · S with 12 ∤ S and s ≥ 1, then ai+1 = 12s−1(13S). Since 12 ∤ S and 13 is coprime to 12, we have 12 ∤ 13S. Thus ai+1 is divisible by 12s−1. Starting with s = t at i = 1, after i steps the exponent of 12 is t − i + 1. Hence as long as i ≤ t, the exponent is at least 1, i.e., 12 | ai. Thus we can apply (3) exactly t times:

at+1 =

13 12

t

a1 =

13t 12t · 12tR = 13tR.

Now 12 ∤ R and gcd(13, 12) = 1, so 12 ∤ 13tR; i.e., 12 ∤ at+1.

The orbit of at+1 is also infinite (tail of an infinite sequence). By the corollary of Lemma 1, at+1 cannot be odd; hence at+1 is even. Also each term has at least three proper divisors. Therefore at+1 satisfies the hypotheses of Lemma 2 (even, not divisible by 12, infinite orbit). Lemma 2 then yields

at+1 ∈ F.

Recall at+1 = 13tR. Since 13t is coprime to 30, the properties defining F must already hold for R:

▷ ν2(at+1) = ν2(R) = 1,

▷ 3 | at+1 and 3 ∤ 13t ⇒ 3 | R, ▷ 5 ∤ at+1 and 5 ∤ 13t ⇒ 5 ∤ R. Thus R ∈ F. We have shown

|a1 = 12t · K with t ≥ 0, K ∈ F|
|---|

.

- 7. Sufficiency - every such number works We prove by induction on t that if a1 = 12tK with K ∈ F, then the sequence is infinite.

▷ Base case t = 0: a1 = K ∈ F. By definition of a fixed point, f(K) = K. Hence the sequence is constant: an = K for all n. Since K has at least three proper divisors (as shown when characterizing F), the sequence is infinite.

▷ Inductive step: Assume the statement holds for all starting values with parameter t − 1 (where t ≥ 1). Let a1 = 12tK with K ∈ F. Because 12 | a1, we can use (3):

a2 = f(a1) =

13 12

a1 =

13 12 · 12tK = 12t−1(13K).

Now, because K ∈ F, we have ν2(K) = 1, 3 | K, 5 ∤ K. Multiplying by 13 (coprime to 30) preserves these properties, so 13K ∈ F. Thus

a2 = 12t−1L with L = 13K ∈ F. By the induction hypothesis, the sequence starting from a2 is infinite. Consequently, the whole sequence from a1 is infinite. This completes the induction, establishing sufficiency.

- 8. Final characterization

Combining necessity and sufficiency, the possible values of a1 are exactly those positive integers that can be written as

a1 = 12t · K (t ≥ 0, K ∈ F), where

F = K ∈ N ν2(K) = 1, 3 | K, 5 ∤ K . Equivalently, in terms of prime exponents:

ν2(a1) is odd, ν3(a1) > 12ν2(a1), 5 ∤ a1. More explicitly, writing

ν2(a1) 2

, ℓ = ν3(a1) − (t + 1) ≥ 0, gcd(R, 30) = 1, we have

t =

a1 = 22t+1 · 3t+1+ℓ · R.

|a1 = 12t · K with t ≥ 0 and K satisfying 2 ∥ K, 3 | K, 5 ∤ K.|
|---|

|a1 = 22t+1 · 3t+1+ℓ · R, t, ℓ ≥ 0, gcd(R, 30) = 1.|
|---|

This completely solves the problem.

##### IMO 2025 Problem 5.

###### IMO 2025 Problem 5

Alice and Bazza are playing the inekoalaty game, a two-player game whose rules depend on a positive real number λ which is known to both players. On the nth turn of the game (starting with n = 1) the following happens:

- • If n is odd, Alice chooses a nonnegative real number xn such that x1 + x2 + · · · + xn ≤ λn.
- • If n is even, Bazza chooses a nonnegative real number xn such that x21 + x22 + · · · + x2n ≤ n.

If a player cannot choose a suitable xn, the game ends and the other player wins. If the game goes on forever, neither player wins. All chosen numbers are known to both players. Determine all values of λ for which Alice has a winning strategy and all those for which Bazza has a winning strategy.

Solution Remark. 7/7.

We begin by setting notation and a couple of useful lemmas. For each n ≥ 1, let

n

n

x2i. The rules are:

Sn =

xi, Qn =

i=1

i=1

▷ On odd turns n, Alice chooses xn ≥ 0 with Sn ≤ λn.

▷ On even turns n, Bazza chooses xn ≥ 0 with Qn ≤ n. If a player cannot make a legal move, the game ends and the other wins. If it continues forever, the result is a draw (no winner).

###### Two elementary lemmas

- Lemma 1. After any even turn n, we have Sn ≤ n and Qn ≤ n. Proof. The condition for even n is Qn ≤ n. By the Cauchy-Schwarz inequality,

Sn2 ≤ n · Qn ≤ n · n = n2, hence Sn ≤ n.

- Lemma 2. Suppose that on all odd turns up to 2M Alice plays 0 (so the only nonzero numbers are Bazza’s choices on even turns). Then after turn 2M we have

√

S2M ≤ M

2.

Proof. Write yi = x2i for i = 1, . . . , M. Then

M

M

yi2. By Cauchy-Schwarz,

S2M =

yi, Q2M =

i=1

i=1

S22M ≤ M · Q2M. Lemma 1 with n = 2M gives Q2M ≤ 2M. Therefore

√

2. □

S22M ≤ M · 2M = 2M2 =⇒ S2M ≤ M

1 √2

- 1. The case λ >

###### - Alice wins

Alice will force a win in a finite number of moves. Choose an integer M large enough so that

√

√

2. (1) (Such M exists because λ > 1/√2 implies 2λ −

λ(2M + 1) − M

2 >

√2 > 0, so the left-hand side tends to +∞ as M → ∞.) Her strategy is:

▷ On turns 1, 3, 5, . . . , 2M − 1 (all odd turns before 2M + 1) she plays xn = 0.

▷ On turn 2M + 1 she plays x2M+1 = λ(2M + 1) − S2M. We must check that the zeros are legal and that the move on turn 2M + 1 is well-defined.

###### Legality of the zeros

After each even turn 2k (1 ≤ k ≤ M), by Lemma 2 we have S2k ≤ k√2. For the odd turn 2k + 1, Alice wants to choose 0; this is allowed iff

S2k + 0 ≤ λ(2k + 1).

k√2 2k + 1 increases with k (its limit is √2/2 = 1/√2). Because λ > 1/√2, we have λ >

k√2 2k + 1

Since S2k ≤ k√2, it suffices to show k√2 < λ(2k + 1). Rewrite as λ >

. The function k  →

k√2 2k + 1

for every k; hence

k√2 < λ(2k + 1). Thus each 0 is legal. The decisive move After turn 2M, Lemma 2 gives S2M ≤ M√2. Define

- From (1) and the bound on S2M,

a = x2M+1 = λ(2M + 1) − S2M.

√

√

a ≥ λ(2M + 1) − M

2 > 0, so a is nonnegative and satisfies S2M + a = λ(2M + 1), hence the sum constraint on turn 2M + 1 is met. Now we analyze the sum of squares after this move. Let S = S2M. Then

2 >

Q2M+1 = Q2M + a2.

By Cauchy-Schwarz applied to the M numbers x2, x4, . . . , x2M we have

S2 M

S2 ≤ M · Q2M =⇒ Q2M ≥

. Therefore

S2 M

S2 M

+ λ(2M + 1) − S 2. Define the function

+ a2 =

Q2M+1 ≥

f(S) =

We need a lower bound for f(S). Monotonicity of f. Compute

√

S2 M

+ λ(2M + 1) − S 2, 0 ≤ S ≤ M

2.

2S M − 2 λ(2M + 1) − S = 2 S 1 +

1 M − λ(2M + 1) , f′′(S) =

2 M

f′(S) =

+ 2 > 0. Thus f′ is strictly increasing. The critical point f′(S) = 0 gives

M · λ(2M + 1) M + 1

S0 =

.

Because λ > 1/√2, we have 2λ > √2, and for sufficiently large M (which we may assume)

√

2. (Simply check that S0 > M√2 ⇐⇒ λ(2M + 1) > √2(M + 1), which holds for large M since the left grows like 2λM and the right like √2M.) Since f′ is increasing and S0 > M√2, we have f′(S) < f′(S0) = 0 for all S ≤ M√2. Hence f is strictly decreasing on [0, M√2]. Consequently its minimum on the interval is attained at S = M√2:

S0 > M

(M√2)2 M

√

√

√

2 2. Thus

2 2 = 2M + λ(2M + 1) − M

+ λ(2M + 1) − M

min

2) =

f(S) = f(M

0≤S≤M√2

√

2 2. But condition (1) says λ(2M + 1) − M√2 > √2, so

Q2M+1 ≥ 2M + λ(2M + 1) − M

√

2 2 > 2. Therefore

λ(2M + 1) − M

###### Q2M+1 > 2M + 2.

Now it is Bazza’s turn (turn 2M + 2). He must choose x2M+2 ≥ 0 such that

Q2M+2 = Q2M+1 + x22M+2 ≤ 2M + 2. Even if he plays x2M+2 = 0, we obtain Q2M+2 = Q2M+1 > 2M + 2, which violates the constraint. Hence Bazza has no legal move, and Alice wins.

1 √2

- 2. The case λ <

###### - Bazza wins

Bazza will use the following strategy on every even turn n:

xn = n − Qn−1 . In words, he takes the largest possible number that still keeps the sum of squares at most n. We will prove by induction that after each of his moves (i.e., after turn 2m) we have

√

(i) Q2m = 2m, (ii) S2m ≥ m

2. (2)

###### Base case m = 1 (turn 2). After turn 1, Alice has chosen some a1 ≥ 0 with a1 ≤ λ. Then

x2 = 2 − a21 (which is real because a21 ≤ λ2 < 1/2 < 2). We get

Q2 = a21 + (2 − a21) = 2. For the sum,

S2 = a1 + 2 − a21. Squaring gives

S22 = a21 + (2 − a21) + 2a1 2 − a21 = 2 + 2a1 2 − a21 ≥ 2, so S2 ≥

√2. Thus (2) holds for m = 1.

Inductive step. Assume (2) is true for m − 1, i.e., after turn 2(m − 1) we have

√

Q2(m−1) = 2(m − 1), S2(m−1) ≥ (m − 1)

2. Turn 2m − 1 (odd): Alice chooses a ≥ 0 with

S2(m−1) + a ≤ λ(2m − 1). (3) Turn 2m (even): Bazza computes

x2m = 2m − Q2m−1. First we verify that the square root is defined, i.e., Q2m−1 ≤ 2m. We have

Q2m−1 = Q2(m−1) + a2 = 2(m − 1) + a2. Hence we need a2 ≤ 2, or a ≤

√2. To see that this always holds, use (3) and the inductive lower bound on S2(m−1):

√

2. (4) We claim that the right-hand side of (4) is at most √2. Indeed,

a ≤ λ(2m − 1) − S2(m−1) ≤ λ(2m − 1) − (m − 1)

√

√

√

λ(2m − 1) − (m − 1)

2 ≤

2 ⇐⇒ λ(2m − 1) ≤ m

2. Because λ < 1/√2, we have 2λ < √2. Then

√

√

λ(2m − 1) = 2λm − λ ≤

2 m − λ <

2 m,

###### √2 m, we get λ(2m − 1) ≤

so the inequality holds (the left side is 2λm − λ, the right side is √2 m; since 2λm ≤

√2 m. More formally:

√

√

2)m ≤ λ. The left-hand side is ≤ 0 (because 2λ −

λ(2m − 1) ≤

2 m ⇐⇒ (2λ −

√2 < 0), while the right-hand side is positive, so the inequality is true for all m ≥ 1. Therefore a ≤

√2, so a2 ≤ 2, and the square root is defined. Now compute:

x2m = 2m − 2(m − 1) + a2 = 2 − a2. Then

Q2m = Q2m−1 + x22m = 2(m − 1) + a2 + (2 − a2) = 2m. For the sum:

S2m = S2(m−1) + a + 2 − a2. Observe that for any a with 0 ≤ a ≤

√2,

a + 2 − a2 2 = 2 + 2a 2 − a2 ≥ 2, hence a + √2 − a2 ≥

√2. Using the inductive bound S2(m−1) ≥ (m − 1)√2, we obtain

√

√

√

S2m ≥ (m − 1)

2. This completes the induction.

2 +

2 = m

###### Alice loses

- From (2) we have S2m ≥ m√2 for every m. Since λ < 1/√2, we have √2−2λ > 0. Choose m large enough so that

√

2 > λ(2m + 1). (5) (Such m exists because m(√2 − 2λ) > λ eventually holds.) Consider turn 2m + 1. After turn 2m,

m

√

S2m ≥ m

2 > λ(2m + 1) by (5). Therefore, even if Alice tries to play x2m+1 = 0, we have

S2m+1 = S2m > λ(2m + 1),

which violates the sum constraint. Consequently she has no legal move, and Bazza wins.

1 √2

- 3. The case λ =

###### - Draw

We show that neither player has a winning strategy by exhibiting a strategy for each that prevents the opponent from winning.

Bazza prevents Alice from winning Bazza uses the same maximal strategy as in case 2: on each even turn n, set xn = √n − Qn−1. We check that the induction in case 2 still works when λ = 1/√2. For the inductive step, from (3) and the lower bound S2(m−1) ≥ (m − 1)√2 we obtain

√

a ≤ λ(2m − 1) − (m − 1)

2. With λ = 1/√2,

√

√

2m − 1 − 2(m − 1) √2

2m − 1 √2 − (m − 1)

1 √2

λ(2m − 1) − (m − 1)

2 =

2 =

=

.

Thus a ≤ 1/√2 < √2, so a2 ≤ 1/2, and the square root √2 − a2 is well-defined. The rest of the induction is unchanged, yielding

√

2 for all m. (6) Now consider any odd turn 2m + 1. Alice must choose a′ ≥ 0 such that

Q2m = 2m, S2m ≥ m

2m + 1 √2

S2m + a′ ≤ λ(2m + 1) =

.

From (6), S2m ≥ m√2 =

2m √2

. Therefore

2m + 1 √2 − S2m ≤

2m + 1 √2 −

2m √2

1 √2

a′ ≤

=

.

- 1

- 2

Consequently a′2 ≤

, and

- 1

- 2

Q2m+1 = Q2m + a′2 = 2m + a′2 ≤ 2m +

< 2m + 2.

Thus after Alice’s move, we have Q2m+1 < 2m + 2. Bazza can then reply with x2m+2 = 0 (or any number keeping Q2m+2 ≤ 2m + 2). In particular, Bazza always has a legal move, so Alice cannot force a win.

Alice prevents Bazza from winning Alice adopts the strategy: on every odd turn she plays 0. Then the only nonzero numbers are Bazza’s even-turn choices. By Lemma 2, after turn 2m we have

√

S2m ≤ m

2. The sum constraint on turn 2m + 1 is

2m + 1 √2

S2m+1 = S2m + x2m+1 ≤ λ(2m + 1) =

. Since

√

2m √2

2m + 1 √2

S2m ≤ m

2 =

<

,

the choice x2m+1 = 0 satisfies the inequality. Hence Alice never loses on her turn. Moreover, with this strategy Bazza cannot force a win, because Alice always has a legal move (namely 0). Therefore Bazza does not possess a winning strategy. Since both players have strategies that thwart a forced win, the game is a draw; it can also be continued indefinitely (e.g., with Bazza always choosing 0 as well). So λ = 1/√2 leads to no winner.

Conclusion We have determined the outcome for all positive λ:

|Alice has a winning strategy if and only if λ ><br><br>1 √2<br><br>. Bazza has a winning strategy if and only if λ <<br><br>1 √2<br><br>. When λ =<br><br>1 √2<br><br>, the game is a draw (no winner).|
|---|

##### IMO 2025 Problem 6.

###### IMO 2025 Problem 6

Consider a 2025 × 2025 grid of unit squares. Matilda wishes to place on the grid some rectangular tiles, possibly of different sizes, such that each side of every tile lies on a grid line and every unit square is covered by at most one tile.

Determine the minimum number of tiles Matilda needs to place so that each row and each column of the grid has exactly one unit square that is not covered by any tile.

Solution Remark. 0/7.

Let n = 2025. The grid consists of n2 unit squares. We must choose a set of squares to remain uncovered such that each row and each column contains exactly one uncovered square; equivalently, the uncovered squares form a permutation matrix. The remaining n2 − n squares must be covered by non-overlapping axis-aligned rectangles (tiles) whose sides lie on the grid lines. The goal is to minimize the number of tiles.

- 1. Reduction to the diagonal configuration Let the uncovered square in row i be at column ci; then c1, c2, . . . , cn is a permutation of {1, . . . , n}. Consider the column permutation σ defined by σ(ci) = i. Applying σ to the column indices (i.e., renaming column j to σ(j)) transforms the uncovered squares into the diagonal positions (i, i). The tiling is transformed by moving each tile together with the columns; this does not change the number of tiles, and the covering properties remain valid because the grid structure is preserved (only the labels of columns are changed). Therefore, the minimum number of tiles over all configurations is the same as the minimum over configurations where the uncovered squares are exactly the main diagonal {(i, i) | 1 ≤ i ≤ n}. Consequently, we may assume without loss of generality that the diagonal squares are the only uncovered ones.
- 2. A construction using 2n − 2 tiles We now exhibit a tiling of the board with the diagonal uncovered, using exactly 2n − 2 rectangles.

▷ Lower triangle (i > j): For each column j = 1, 2, . . . , n − 1, place a vertical tile that covers every cell in column j whose row index is greater than j. Formally,

Rj(L) = {(i, j) | j + 1 ≤ i ≤ n}.

These n − 1 rectangles are pairwise disjoint (they lie in different columns) and cover all squares with i > j.

▷ Upper triangle (i < j): For each row i = 1, 2, . . . , n − 1, place a horizontal tile that covers every cell in row i whose column index is greater than i:

Ri(U) = {(i, j) | i + 1 ≤ j ≤ n}. These n − 1 rectangles are also pairwise disjoint (they occupy distinct rows) and cover all squares with i < j. The two families are disjoint because a square with i > j belongs to some Rj(L) (column j) and a square with i < j belongs to some Ri(U) (row i); no square can satisfy both conditions. The diagonal squares (i, i) are not covered by any tile. Thus we have a valid covering of all off-diagonal squares using

k = (n − 1) + (n − 1) = 2n − 2 tiles.

- 3. Lower bound: at least 2n − 2 tiles Now take any tiling T of the board that leaves exactly the diagonal squares uncovered. Partition the off-diagonal squares into two sets:

###### L = {(i, j) | i > j}, U = {(i, j) | i < j}.

- Lemma 1. Every tile T ∈ T is entirely contained either in L or in U. Proof. Suppose a tile T contains a square from L and a square from U. Represent T as the Cartesian product of an interval of rows and an interval of columns:

T = {(i, j) | a ≤ i ≤ b, c ≤ j ≤ d}, with 1 ≤ a ≤ b ≤ n, 1 ≤ c ≤ d ≤ n. If the intervals [a, b] and [c, d] intersect, then there exists an integer r ∈ [a, b] ∩ [c, d], and the square (r, r) belongs to T, contradicting the fact that diagonal squares are uncovered. Hence [a, b] and [c, d] are disjoint.

▷ If b < c, then for any (i, j) ∈ T we have i ≤ b < c ≤ j, so i < j; thus T ⊆ U, contradicting the presence of an Lsquare.

▷ If d < a, then j ≤ d < a ≤ i, so i > j, giving T ⊆ L, contradiction. Therefore a tile cannot contain squares from both L and U; it must lie wholly in one of them.

- Lemma 2. In the lower triangle L, consider the n − 1 cells

DL = {(i, i − 1) | i = 2, 3, . . . , n}. No tile that is a subset of L can contain two distinct cells from DL. Proof. Let T ⊆ L be a tile, so T = [a, b] × [c, d]. Because T ⊆ L, we have i > j for every (i, j) ∈ T. In particular, the cell (a, d) (the topmost row and rightmost column of the tile) satisfies a > d. Assume, for contradiction, that T contains two distinct cells (i, i − 1) and (j, j − 1) with i < j. From (i, i − 1) ∈ T we obtain

a ≤ i ≤ b, c ≤ i − 1 ≤ d. Since a > d and d ≥ i − 1, we have a > i − 1, hence a ≥ i. Combined with a ≤ i, we get a = i. Now a = i and a > d imply i > d, i.e., d < i. But d ≥ i − 1 from the containment, so d = i − 1. Now apply the same reasoning to (j, j − 1) ∈ T. We obtain a = j and d = j − 1. But we already have a = i, so i = j, contradicting i < j. Hence T cannot contain two cells of DL.

- Lemma 3. In the upper triangle U, consider the n − 1 cells

DU = {(i, i + 1) | i = 1, 2, . . . , n − 1}. No tile that is a subset of U can contain two distinct cells from DU. Proof. Let T ⊆ U be a tile, T = [a, b] × [c, d]. Since T ⊆ U, we have i < j for every (i, j) ∈ T. In particular, the cell (b, c) (the bottommost row and leftmost column) satisfies b < c. Suppose T contains (i, i + 1) and (j, j + 1) with i < j. From (i, i + 1) ∈ T we have

a ≤ i ≤ b, c ≤ i + 1 ≤ d. Because c > b and b ≥ i, we get c > i, thus c ≥ i + 1. But c ≤ i + 1 from the containment, so c = i + 1. Moreover, c > b ≥ i and c = i + 1 imply b < i + 1, i.e., b ≤ i. Combined with i ≤ b, we obtain b = i. Now from (j, j + 1) ∈ T we similarly deduce c = j + 1 and b = j. Since c = i + 1, we have i + 1 = j + 1, so i = j, contradiction. Hence T cannot contain two cells of DU.

Lower bound argument. The sets DL and DU each contain n − 1 squares, and every square must be covered. By Lemma 1, any tile that covers a square from L is entirely contained in L; similarly, any tile covering a square from U is entirely in U.

▷ DL must be covered by tiles that lie in L. Lemma 2 tells us that each such tile can cover at most one element of DL. Therefore at least |DL| = n − 1 tiles are required that belong to L.

▷ DU must be covered by tiles that lie in U, and Lemma 3 says each such tile can cover at most one element of DU, so at least n − 1 tiles are needed that belong to U. A tile cannot be in both L and U because L and U are disjoint. Hence the total number of tiles is at least

(n − 1) + (n − 1) = 2n − 2.

- 4. Minimum number of tiles We have shown:

▷ There exists a tiling with 2n − 2 tiles (Construction).

▷ Any tiling uses at least 2n − 2 tiles (Lower bound). Therefore the minimum possible number of tiles is exactly 2n − 2. Substituting n = 2025 gives

2 × 2025 − 2 = 4050 − 2 = 4048.

|4048|
|---|

##### H.2 USAMO 2026 USAMO 2026 Problem 1.

###### USAMO 2026 Problem 1

Let n be an integer greater than 1. For which real numbers x is

n

⌊kx⌋ k

⌊nx⌋ −

k=1

maximal, and what is the maximal value that this expression can take? Note: ⌊z⌋ denotes the greatest integer less than or equal to z.

Solution Remark. 7/7.

- Step 1. Reduction to the fractional part. Let n > 1 be an integer. For any real x, write x = ⌊x⌋+{x} = m+θ, where m ∈ Z and θ ∈ [0, 1). For each k ≥ 1,

⌊kx⌋ = ⌊km + kθ⌋ = km + ⌊kθ⌋, because km is an integer. Hence the expression becomes

E(x) = ⌊nx⌋ −

n

k=1

⌊kx⌋ k

= nm + ⌊nθ⌋ −

n

k=1

km + ⌊kθ⌋ k

= ⌊nθ⌋ −

n

k=1

⌊kθ⌋ k

. Thus E(x) depends only on θ; we define

F(θ) = ⌊nθ⌋ −

n

k=1

⌊kθ⌋ k

, θ ∈ [0, 1).

- Step 2. Value on the interval 1 − n1 , 1 . Assume θ ∈ 1 − n1 , 1 . Then θ ≥ n−n1 and θ < 1, so for every k = 1, . . . , n,

kθ ≥ k ·

n − 1 n

= k −

k n ≥ k − 1,

and kθ < k. Consequently ⌊kθ⌋ = k − 1 for all k. In particular ⌊nθ⌋ = n − 1. Therefore

F(θ) = (n − 1) −

n

k=1

k − 1 k

. Now

n

k=1

k − 1 k

=

n

k=1

1 −

1 k

= n − Hn,

where Hn = 1 + 12 + · · · + n1 (the n-th harmonic number). Hence

F(θ) = (n − 1) − (n − Hn) = Hn − 1.

So on the whole interval [1 − n1 , 1) the expression is constant and equals Hn − 1.

- Step 3. Upper bound: first decomposition. To prove that Hn − 1 is the maximum, we show F(θ) ≤ Hn − 1 for all θ ∈ [0, 1). Let θ be arbitrary and write

###### N = ⌊nθ⌋ ∈ {0, 1, . . . , n − 1}, β = nθ − N ∈ [0, 1). Set t = β/n ∈ [0, 1/n). Then

N n

+ t. For k = 1, . . . , n,

θ =

kN n

+ kt. Write the division of kN by n:

kθ =

kN n

kN n

rk n

=

,

+

where rk is the remainder, i.e. rk = kN mod n with 0 ≤ rk < n. Then

Define

kθ =

kN n

integer

+

rk n

+ kt

in [0,2)

###### .

Then

δk =

1 if rnk + kt ≥ 1, 0 otherwise.

kN n

⌊kθ⌋ =

+ δk. Substituting into F(θ) gives

n

kN

n + δk k

F(θ) = N −

k=1

n

n

kN n

δk k

= N −

−

.

k

k=1

k=1

A(N)

Since each δk/k ≥ 0, we obtain the important inequality

n

kN n

F(θ) ≤ A(N), where A(N) = N −

. (1)

k

k=1

Thus it suffices to prove A(N) ≤ Hn − 1 for all N = 0, 1, . . . , n − 1.

###### Step 4. Simplifying A(N). Using kNn = kNn−rk , we compute

n

kN − rk nk

A(N) = N −

k=1

n

N n −

rk nk

= N −

k=1

n

n

N n

1 n

rk k

= N −

1 +

k=1

k=1

n

n

rk k

rk k

1 n

1 n

. (2)

= N − N +

=

k=1

k=1

###### Step 5. Introducing the greatest common divisor. Let d = gcd(N, n). Write

n = d n1, N = d N1,

with gcd(N1, n1) = 1. Because rk is the remainder of kN modulo n, we have rk = d · sk, where

sk = (kN1) mod n1, 0 ≤ sk < n1.

Moreover, as k runs from 1 to n, the values sk take each integer from 0 to n1 − 1 exactly d times (this follows from the fact that k  → kN1 (mod n1) is a bijection over each block of length n1). Therefore

n

n

n

n

1 n

1 n

d n

1 n1

rk k

d sk k

sk k

sk k

. (3)

=

=

=

k=1

k=1

k=1

k=1

###### Now decompose k as k = in1 + j with i = 0, 1, . . . , d − 1 and j = 1, 2, . . . , n1. Then sk = sj because sin1+j = ((in1 + j)N1) mod n1 = (jN1) mod n1 = sj. Hence

n1

d−1

n

sk k

1 in1 + j

sj

. Define the weights

=

i=0

j=1

k=1

d−1

1 in1 + j

wj =

, j = 1, . . . , n1.

i=0

Observe that wj is strictly decreasing in j (since larger j makes denominators larger). Then

n1

1 n1

sjwj. (4)

A(N) =

j=1

- Step 6. Applying the rearrangement inequality. The set {s1, . . . , sn1} is a permutation of {0, 1, . . . , n1 − 1} (this is true for the indices j = 1, . . . , n1, because jN1 mod n1 runs through all residues exactly once). The weights wj are decreasing. By the rearrangement inequality, the sum sjwj is maximized when the sj are also arranged in decreasing order. The decreasing order of the numbers 0, 1, . . . , n1 −1 is s1 = n1 −1, s2 = n1 −2,

..., sn1 = 0. Hence for any permutation we have

n1

j=1

sjwj ≤

n1

j=1

(n1 − j)wj. (5) Consequently,

A(N) ≤

1 n1

n1

j=1

(n1 − j)wj. (6)

- Step 7. Relating to the harmonic number. Compute the right-hand side:

n1

j=1

(n1 − j)wj =

n1

j=1

(n1 − j)

d−1

i=0

1 in1 + j

=

d−1

i=0

n1

j=1

n1 − j in1 + j

=: S.

Now the harmonic number Hn can be written as

Hn =

d−1

i=0

n1

j=1

1 in1 + j

. Thus

n1Hn =

d−1

i=0

n1

j=1

n1 in1 + j

. Consider the difference

n1Hn − n1 − S =

d−1

i=0

n1

j=1

n1 in1 + j −

n1 − j in1 + j − n1 =

d−1

i=0

n1

j=1

j

in1 + j − n1. Define

T =

d−1

i=0

n1

j=1

j in1 + j

. (7)

Then S ≤ n1Hn − n1 is equivalent to T ≥ n1.

- Step 8. Proof that T ≥ n1. Rewrite each term:

j in1 + j

in1 in1 + j

= 1 −

. Hence

n1

d−1

T =

i=0

n1

d−1

=

i=0

in1 in1 + j

1 −

j=1

n1

d−1

in1 in1 + j

1 −

j=1

i=0

j=1

d−1

1 in1 + j

= dn1 − n1

. Let

- i

n1

- j=1

i=0

n1

d−1

1 in1 + j

.

Q =

i

i=0

j=1

The term with i = 0 is 0. For i ≥ 1, note that in1 + j > in1 for all j ≥ 1, so

n1

n1

1 in1 + j

1 in1

n1 in1

1 i

. Multiplying by i (positive) gives

<

=

=

j=1

j=1

1 in1 + j

< 1. Summing these inequalities for i = 1, 2, . . . , d − 1 yields

- i

n1

- j=1

d−1

1 = d − 1. (8) If d = 1, then Q = 0 (the sum over i is empty). Then

Q <

i=1

T = dn1 − n1Q = n1 (since d = 1). If d ≥ 2, we have Q < d − 1, hence

T = dn1 − n1Q > dn1 − n1(d − 1) = n1. Thus in all cases T ≥ n1, with equality exactly when d = 1.

- Step 9. Final bound for A(N). From the equivalence established in Step 7, T ≥ n1 implies S ≤ n1Hn−n1. Using (6) we obtain

A(N) ≤

1 n1

S ≤

1 n1

(n1Hn − n1) = Hn − 1. Hence A(N) ≤ Hn − 1 for every N.

- Step 10. Equality conditions. We have the chain

F(θ) ≤ A(N) ≤ Hn − 1. To have F(θ) = Hn − 1, both inequalities must be equalities.

###### ▷ Equality in F(θ) ≤ A(N) requires nk=1 δk/k = 0, i.e., δk = 0 for all k. This means rnk + kt < 1 for every k.

▷ Equality in A(N) ≤ Hn − 1 requires that the upper bound we derived be attained. From the derivation, this happens exactly when d = 1 (i.e., gcd(N, n) = 1) and the rearrangement inequality (5) is sharp. Since wj is strictly decreasing, the maximum of sjwj is achieved when the sj are also decreasing. For d = 1, the set {s1, . . . , sn} (since n1 = n) is a permutation of {0, 1, . . . , n − 1}. The decreasing order corresponds to sj = n − j for j = 1, . . . , n (with sn = 0). This is equivalent to (jN) mod n = n − j for all j = 1, . . . , n. This congruence simplifies to jN ≡ −j (mod n), i.e., N ≡ −1 (mod n). Because 0 ≤ N ≤ n − 1, this forces N = n − 1.

Now check that for N = n−1 the condition δk = 0 indeed holds. Write N = n−1, then d = gcd(n−1, n) = 1. For k = 1, . . . , n, rk = (k(n − 1)) mod n = n − k (for k < n) and rn = 0. Then

n − k n

rk n

+ kt (or 0 + nt for k = n). Since t < 1/n, we have n−k

+ kt =

n + kt < n−nk + k · n1 = 1, and for k = n, nt < 1. Hence δk = 0 for all k. Therefore F(θ) = A(N) = Hn − 1. Conversely, if θ ∈ 1 − n1 , 1 , then N = n − 1 and t = θ − (n − 1)/n ∈ [0, 1/n). As just verified, this yields F(θ) = Hn − 1. Thus the maximum value Hn−1 is attained precisely when the fractional part of x lies in the interval [1−1/n, 1).

- Step 11. Conclusion. The maximal value of the expression is Hn − 1, where Hn = 1 + 12 + · · · + n1 , and this maximum occurs exactly for those real numbers x whose fractional part belongs to [1 − 1/n, 1).

|Hn − 1, where Hn = 1 +<br><br>1<br><br>2<br><br><br>+ · · · +<br><br>1 n<br><br>, and the maximum is attained iff {x} ∈ 1 −<br><br>1 n<br><br>, 1|
|---|

##### USAMO 2026 Problem 2.

###### USAMO 2026 Problem 2

Annie is playing a game where she starts with a row of positive integers, written on a blackboard, each of which is a power of 2. On each turn, she can erase two adjacent numbers and replace them with a power of 2 that is greater than either of the erased numbers. This shortens the row of numbers, and she continues to take turns until only one number remains. Annie wins the game if the final remaining number is less than 4 times the sum of the original numbers. Is it always possible for Annie to win, regardless of the starting row of numbers?

Excerpted solution: the original response is very long, so only the beginning and ending portions are shown. Solution

###### Remark. 0/7.

We will prove that Annie can always win, i.e., for any initial row of powers of two, there exists a sequence of moves that ends with a single number F satisfying F < 4S, where S is the sum of the original numbers. Let S be the sum of the numbers, and let k = ⌊log2 S⌋, so that

2k ≤ S < 2k+1. The proof proceeds by strong induction on the number of terms n. We will show the following statement: Claim. For any list of n powers of two, there exists a sequence of moves such that the final number F satisfies F ≤ 4 · 2k. Once the Claim is established, we obtain Annie’s victory:

▷ If S is not a power of two, then 2k < S, hence 4 · 2k < 4S and therefore F < 4S.

▷ If S is a power of two, the same inductive proof (when analysed more carefully) actually yields the sharper bound F ≤ 2S, which is certainly < 4S for S > 0. (We will comment on this improvement at the end.) Thus the main task is to prove the Claim by induction on n.

Base case n = 1 There is only the number itself, so F = S. Since S ≤ 2k+1 ≤ 4 · 2k, the Claim holds.

Inductive step Assume the Claim holds for every list of size less than n. Consider a list of size n. Let its sum be S and let k = ⌊log2 S⌋.

###### Case 1: There exists an adjacent equal pair

Choose such a pair, say ai = ai+1 = a. Replace them by 2a (the smallest allowed power of two greater than a). The sum does not change, so k remains the same. The new list has size n − 1, so by the induction hypothesis it can be reduced to a final number ≤ 4 · 2k. Hence the same holds for the original list.

###### Case 2: No two adjacent numbers are equal

In this case we choose an adjacent pair (ai, ai+1) for which max(ai, ai+1) is as small as possible; denote the two numbers by x and y with x < y. Replace them by 2y (again the smallest possible power of two greater than y). Let the new sum be

###### S′ = S + (y − x).

Set k′ = ⌊log2 S′⌋. We now consider two subcases.

- Subcase 2a: k′ = k Then S′ < 2k+1, so the floor has not increased. The new list has size n − 1; by the induction hypothesis it can be reduced to a final number ≤ 4 · 2k′ = 4 · 2k. So the original list also admits such a strategy.
- Subcase 2b: k′ = k + 1 In this situation 2k+1 ≤ S′ < 2k+2. First note that y ≤ 2k; otherwise y ≥ 2k+1 would force S ≥ y ≥ 2k+1, contradicting S < 2k+1. Consequently 2y ≤ 2k+1. Now examine the list after the merge; it has size n − 1 and sum S′. Although its floor is k + 1, we can still use the induction hypothesis - not directly on this list (which would give a bound of 4 · 2k+1 = 8 · 2k), but we can instead apply the induction hypothesis to the rest of the list - i.e., the numbers that were not part of the merged pair. However, after the merge the list is contiguous, so the ”rest” is not isolated. Instead we perform the following two step reduction:

- 1. Reduce the part of the list that does not contain the new number 2y (i.e., all other numbers) to a single number T. Because the rest consists of several (possibly one) contiguous blocks, we can reduce each block independently to a single number using the induction hypothesis (the blocks have size < n). By doing so we obtain a list consisting of 2y and the numbers T1, T2, . . . (one per block). Each Tj satisfies Tj ≤ 4 · 2k because the sum of the corresponding block is < 2k+1 (since the total sum S < 2k+1 and we have removed at least the two numbers x, y with y ≥ 2, so the block sum is at most 2k+1 − 3? Actually a precise bound is not needed; we only need that every such Tj is ≤ 4 · 2k. Indeed, for any block, its sum is less than 2k+1, so by the induction hypothesis (applied to that block) we obtain a number ≤ 4 · 2k (its floor is at most k, so the bound is 4 · 2k). After this reduction, the list contains 2y and several numbers Tj, each ≤ 4 · 2k. Now we can merge these numbers one by one. A crucial observation is that 2y ≤ 2k+1 ≤ 4 · 2k for k ≥ 0. Hence all numbers present are at most 4 · 2k. When we merge two numbers each at most 4 · 2k, the result is at most 2 · 4 · 2k = 8 · 2k, which is still too large. To keep the bound at 4 · 2k, we must merge in a way that the maximum never exceeds 2k+1. Notice that 2y ≤ 2k+1. If we can ensure that every Tj is also ≤ 2k+1, then merging any two of them yields at most 2k+2 = 4 · 2k. Therefore the problem reduces to showing that each block can be reduced to a number ≤ 2k+1 rather than the coarser 4 · 2k. But can we guarantee ≤ 2k+1? Since the sum of a block is < 2k+1, a direct application of the induction hypothesis would only give ≤ 4 · 2k. However, we can prove a slightly stronger lemma that will serve our purpose: Lemma. If a list has sum T with 2L < T < 2L+1, then there exists a strategy with final number ≤ 2L+1 provided that T ≤ 3 · 2L−1 − 1. (This lemma will be proved by induction on n in a moment.) In our situation, each block sum is at most S − x − y < 2k+1. Moreover, because we are in Subcase 2b, the original list has no adjacent equal and we chose the pair with smallest maximum. A detailed analysis (which we will carry out for the lemma) shows that the sum of each block is actually ≤ 3 · 2k−1 − 1. Hence the Lemma applies, giving Tj ≤ 2k+1.
- 2. With all numbers ≤ 2k+1, merging any two yields at most 2k+2 = 4 · 2k. By continuing this process we eventually obtain a single number ≤ 4 · 2k. Thus, if the Lemma holds, the Claim is established in Subcase 2b as well.

Proof of the Lemma We now prove the Lemma stated above. Lemma. Let L ≥ 2 be an integer. For any list of powers of two whose sum T satisfies

2L < T ≤ 3 · 2L−1 − 1, there exists a sequence of moves that ends with a number ≤ 2L+1. Proof. We use strong induction on the number of terms n.

▷ Base n = 2. The two numbers are powers of two, say a ≤ b. Since T = a + b > 2L, we must have b ≥ 2L. If b > 2L, then b ≥ 2L+1, implying T ≥ 2L+1, which contradicts T ≤ 3 · 2L−1 − 1 < 2L+1 (for L ≥ 2). Hence b = 2L. Then a = T − 2L ≤ 2L−1 − 1. The only legal move is to replace (a, 2L) with a power of two greater than 2L; the smallest such is 2L+1. By choosing that, we obtain a final number exactly 2L+1, which certainly is ≤ 2L+1.

▷ Inductive step. Assume the Lemma holds for all lists of size smaller than n. Consider a list of size n with sum T in the prescribed interval. If there is an adjacent equal pair, merge it. The sum stays the same, so T remains in the interval, and the size becomes n − 1. By the induction hypothesis we get a final number ≤ 2L+1. If there is no adjacent equal pair, let (x, y) be the adjacent pair with the smallest maximum; x < y. Merge them to 2y and obtain a new list of size n − 1 with sum T′ = T + (y − x).

We need to show that T′ still lies in an interval that allows us to apply the induction hypothesis (possibly with a different L) and still obtain the bound 2L+1.

First, bound y. Because the list has no adjacent equal numbers and its sum is less than 2L+1, the maximum element cannot exceed 2L−1 if we want to stay within the interval after the merge? Actually we can have

- y = 2L, but then the structure of the list is very restricted. We analyse the two possibilities:

###### ▷ y ≤ 2L−1. Then 2y ≤ 2L. Also, y − x ≤ 2L−1 − 1. Consequently

T′ ≤ T + (2L−1 − 1) ≤ (3 · 2L−1 − 1) + (2L−1 − 1) = 4 · 2L−1 − 2 = 2L+1 − 2. Moreover, T′ > T > 2L. Hence T′ lies in (2L, 2L+1). If T′ ≤ 3 · 2L−1 − 1, then T′ is still in the original interval for L, and the induction hypothesis (applied to the list of size n − 1) gives F ≤ 2L+1. If T′ > 3 · 2L−1 − 1, then T′ is in (3 · 2L−1, 2L+1). In this range, we claim that T′ actually belongs to the interval for L + 1, i.e., 2L+1 < T′ ≤ 3 · 2L − 1? But T′ < 2L+1, so it cannot be in (2L+1, . . . ]. However, observe that T′ ≥ 3 · 2L−1 + 1 (since it exceeds the upper bound). But then T′ ≥ 3 · 2L−1 + 1. For L ≥ 2,

- 3 · 2L−1 + 1 may be less than 2L+1, but we need a bound of 2L+1. Since we are only required to obtain some final number ≤ 2L+1, we can simply note that T′ < 2L+1 and therefore the floor of T′ is at most L. If the floor is L (i.e., T′ < 2L+1), then we can apply the induction hypothesis for the same L? The induction hypothesis requires the sum to be > 2L. It is, but we also need the upper bound 3 · 2L−1 − 1 to use the Lemma? Actually the Lemma we are proving is exactly: for any list with sum in (2L, 3 · 2L−1 − 1] we can achieve ≤ 2L+1. If T′ falls outside this interval (i.e., T′ > 3 · 2L−1 − 1), the Lemma does not directly apply. However, we can then apply the induction hypothesis for a different statement: for any list with sum < 2L+1 we can achieve ≤ 2L+1. But is that universally true? No, as the example [8, 4, 2] with L = 3 (sum 14) shows that the minimum final is 32, which is > 16 = 2L+1. So we cannot claim that for every list with sum < 2L+1 we can get ≤ 2L+1. Hence we must be more careful.

▷ y = 2L. Then, because there are no adjacent equal numbers and the sum is at most 3·2L−1 −1 < 2L+1, the list can contain at most one copy of 2L (two would sum to at least 2L+1). Moreover, to avoid any adjacent pair with maximum < 2L, every other number must be adjacent only to 2L (otherwise a pair of smaller numbers would have maximum < 2L). This forces the list to be of the form

[ a1, a2, . . . , ap, 2L, b1, b2, . . . , bq ]

where all ai and bj are powers of two less than 2L, and no two of the ai (or two of the bj) are adjacent; in particular, p ≤ 1 and q ≤ 1. Thus the list has at most three elements. For size 2, the list is [2L, c] or [c, 2L] with

- c < 2L. That case is already covered by the base (size 2). For size 3, the list is [a, 2L, b] with a, b powers of two, a, b < 2L, and a + b ≤ 2L−1 − 1 (since the total sum is at most 3 · 2L−1 − 1). Now, the pair with smallest maximum is either (a, 2L) or (2L, b), both have maximum 2L. Choose one, say (a, 2L). Merging gives 2L+1 and the list becomes [2L+1, b]. Its sum is 2L+1 + b. Since b ≤ 2L−1 − 1, we have

2L+1 < 2L+1 + b ≤ 2L+1 + 2L−1 − 1 = 5 · 2L−1 − 1. For L ≥ 2, 5 · 2L−1 − 1 is still less than 6 · 2L−1 − 1 = 3 · 2L − 1, so the sum lies in (2L+1, 3 · 2L − 1]. Now, by the induction hypothesis (the Lemma for the larger index L + 1), we can reduce this list to a number ≤ 2(L+1)+1 = 2L+2. But we need ≤ 2L+1. However, we are not forced to apply the Lemma for L + 1; we can directly finish the game: from [2L+1, b], the only move is to merge the two numbers (they are different because b < 2L < 2L+1). The smallest possible result is 2L+2 (since we must replace them with a power of two greater than max(2L+1, b) = 2L+1, so the smallest is 2L+2. That would give F = 2L+2, which is larger than 2L+1. But we are free to choose a larger number if we wish; however, to achieve the bound ≤ 2L+1 we would need to obtain exactly 2L+1, which is impossible because the numbers are different and the rule requires a number greater than both, so the result must be at least 2L+2. This suggests that the Lemma might be false for lists of size 3 of this form? Let’s test with concrete numbers: L = 3, so 2L = 8. Choose a = 2, b = 2? But a + b ≤ 2L−1 − 1 = 3, so a = 2, b = 1 gives sum 8 + 2 + 1 = 11, which is within the interval (8, 11]. According to the Lemma, there should be a strategy yielding F ≤ 2L+1 = 16. But earlier we saw that from [2, 8, 1] the minimum final is 32, not 16. Let’s verify carefully:

List: [2, 8, 1]. No adjacent equal. The pair with smallest maximum: pairs are (2, 8) max=8, (8, 1) max=8. Suppose we merge (2, 8) to 16 (smallest possible). Resulting list: [16, 1]. Now merge (16, 1): the smallest power of two greater than 16 is 32. So final 32. If we merge (8, 1) first, we get [2, 16] then merge to 32. So indeed the minimum is 32. Therefore the Lemma, as stated, is false for L = 3 with the list [2, 8, 1]. Indeed, 2L = 8, 3 · 2L−1 − 1 = 11, and T = 11 is allowed. But we cannot achieve ≤ 16; the best is 32.

Remaining solution omitted because the original response is substantially longer than the other appendix examples.

##### USAMO 2026 Problem 3.

###### USAMO 2026 Problem 3

Let ABC be an acute scalene triangle with no angle equal to 60◦. Let ω be the circumcircle of ABC. Let ∆B be the equilateral triangle with three vertices on ω, one of which is B. Let ℓB be the line through the two vertices of ∆B other than B. Let ∆C and ℓC be defined analogously. Let Y be the intersection of AC and ℓB, and let Z be the intersection of AB and ℓC.

Let N be the midpoint of minor arc BC on ω. Let R be the triangle formed by ℓB, ℓC, and the tangent to ω through N. Prove that the circumcircle of AY Z and the incircle of R are tangent.

Solution Remark. 7/7.

We are given an acute scalene triangle ABC with no angle equal to 60◦. Its circumcircle is ω. For vertex B we construct the equilateral triangle ∆B inscribed in ω (one vertex is B); let ℓB be the line through the two other vertices. Similarly, ∆C and ℓC. Define

Y = AC ∩ ℓB, Z = AB ∩ ℓC.

Let N be the midpoint of the minor arc BC (the arc not containing A). Let R be the triangle bounded by ℓB, ℓC and the tangent to ω at N. We must prove that the circumcircle of △AY Z and the incircle of △R are tangent.

- 1. Complex numbers on the unit circle Place the circumcircle ω as the unit circle in the complex plane. Denote the points by

A = a, B = b, C = c, |a| = |b| = |c| = 1. The triangle is acute, scalene and none of its angles is 60◦.

- Lemma 1 (Line through two points on the unit circle). For distinct u, v on the unit circle, the line uv is given by

z + uv z¯ = u + v. Proof. A point z is collinear with u and v iff z−u

v−u is real. Taking conjugates and using u¯ = 1/u, v¯ = 1/v gives the stated equation. □

- 2. Equations of the relevant lines

Let ζ = e2πi/3. The equilateral triangle ∆B inscribed in ω with vertex B has the other two vertices Bζ and Bζ2. The side opposite B is ℓB, the line through Bζ and Bζ2. Applying Lemma 1 with u = bζ, v = bζ2 yields uv = b2 and u + v = b(ζ + ζ2) = −b. Hence

- ℓB : z + b2z¯ = −b. (1) Analogously,
- ℓC : z + c2z¯ = −c. (2) The point N is the midpoint of the minor arc BC (not containing A). On the unit circle the midpoint of an arc has the property that its square equals the product of the endpoints:

N2 = bc. (3) The tangent to ω at N is given by

z + N2z¯ = 2N. Using (3) we obtain

tangent at N : z + bc z¯ = 2N. (4)

- 3. Intersection points Y and Z The line AC has equation (by Lemma 1)

AC : z + ac z¯ = a + c. (5) Intersect AC with ℓB. Subtract (5) from (1):

(b2 − ac) z¯ = −b − (a + c) = −(a + b + c). Set

S = a + b + c, D1 = ac − b2. Then z¯ = S/D1. Substitute into (1) to find the coordinate y:

bD1 + b2S D1

b2S D1

y = −b − b2y¯ = −b −

= −

. Compute the numerator:

bD1 + b2S = b(ac − b2) + b2(a + b + c) = abc − b3 + ab2 + b3 + b2c = b(ac + ab + bc) = bT, where

T = ab + bc + ca. Thus

bT D1

S D1

. (6)

y = −

, y¯ =

By symmetry, intersecting AB (equation z + ab z¯ = a + b) with ℓC gives

cT D2

z = −

, z¯ =

S D2

, D2 = ab − c2. (7)

- 4. A convenient rotation The configuration is invariant under rotations of the circle. We choose the rotation so that N = 1. Then (3) yields bc = 1, hence c = ¯b. Write

b = eiα, c = e−iα, where α = ∠A (by the inscribed angle theorem, the central angle subtended by BC is 2α). Since the triangle is acute, α ∈ (0◦, 90◦) and, by hypothesis, α ̸= 60◦. Let

k = cos α. The remaining vertex is A = a = eiθ. Because N = 1 lies on the minor arc BC not containing A, the argument θ satisfies θ ∈ (α, 2π − α) (or its symmetric equivalent). Now compute the quantities that appear in (6), (7) in this coordinate system:

- S = a + b + c = a + eiα + e−iα = a + 2k,
- T = ab + bc + ca = aeiα + 1 + ae−iα = 1 + 2ka, Q = b2 + bc + c2 = ei2α + 1 + e−i2α = 2 cos 2α + 1 = 4k2 − 1,

D1 = ac − b2 = ae−iα − ei2α, D2 = ab − c2 = aeiα − e−i2α,

∆ = D1D2 = (ae−iα − ei2α)(aeiα − e−i2α) = a2 − 2a cos 3α + 1.

- 5. A key algebraic identity

- Lemma 2. With the above notation,

(a + 2k)2 = Q T + ∆. (8) Proof. Expand Q T + ∆:

Q T = (4k2 − 1)(1 + 2ka) = (4k2 − 1) + 2k(4k2 − 1)a,

∆ = a2 − 2a cos 3α + 1. Recall cos 3α = 4k3 − 3k. Then

Q T + ∆ = a2 + 2k(4k2 − 1) − 2 cos 3α a + (4k2 − 1) + 1

= a2 + 8k3 − 2k − (8k3 − 6k) a + 4k2

= a2 + 4ka + 4k2 = (a + 2k)2. □

###### 6. The circumcenter O of △AY Z We claim that the point

aQT ∆

(9) is the circumcenter of △AY Z. Let us verify that it is equidistant from A, Y and Z. Distance to A

O = −

a(QT + ∆) ∆

QT ∆

aQT ∆ − a = −a

O − a = −

+ 1 = −

. By Lemma 2, QT + ∆ = (a + 2k)2, so

a(a + 2k)2 ∆

. (10) Hence

O − a = −

|O − a| = |a + 2k|2 |∆|

. (11)

- Distance to Y Using (6), y¯ = S/D1 and y = −bT/D1. Compute

O − y = −

aQT ∆

+

bT D1

= T −

aQ ∆

+

b D1

. Since ∆ = D1D2,

−

aQ ∆

+

b D1

= −aQ D1D2

+

bD2 D1D2

= −aQ + bD2 ∆

. Thus

O − y = T −aQ + bD2 ∆

. (12) Now compute bD2 − aQ:

bD2 = eiα(aeiα − e−i2α) = aei2α − e−iα,

aQ = a(ei2α + 1 + e−i2α) = aei2α + a + ae−i2α, ∴ bD2 − aQ = −e−iα − a − ae−i2α.

On the other hand,

−e−iαT = −e−iα(1+2ka) = −e−iα−2kae−iα = −e−iα−a(eiα+e−iα)e−iα = −e−iα−a(1+e−i2α) = −e−iα−a−ae−i2α. Therefore

bD2 − aQ = −e−iαT, and consequently

|bD2 − aQ| = |T|. From (12) we obtain

|O − y| = |T|2 |∆|

. (13)

- Distance to Z A completely symmetric computation (interchanging b with c, D1 with D2) gives

Equality of the distances Now note that

|O − z| = |T|2 |∆|

. (14)

|T|2 = (1 + 2ka)(1 + 2ka¯) = 1 + 2k(a + a¯) + 4k2 = 1 + 2kt + 4k2, t = a + a¯ = 2 cos θ. But also

|a + 2k|2 = (a + 2k)(¯a + 2k) = 1 + 2k(a + a¯) + 4k2 = |T|2. Thus

, |O − y| = |T|2 |∆|

= |a + 2k|2 |∆|

, |O − z| = |a + 2k|2 |∆|

|O − a| = |a + 2k|2 |∆|

. Hence O is indeed the circumcenter of △AY Z, and the circumradius is

R = |a + 2k|2 |∆|

. (15)

- 7. Cartesian description of triangle R Now we work in the rotated coordinate system where N = 1, b = eiα, c = e−iα. Write a complex number

z = x + iy.

- Equation of ℓB From (1): z + b2z¯ = −b. Substituting b = cos α + i sin α, b2 = cos 2α + i sin 2α, we separate real and imaginary parts:

x(1 + cos 2α) + y sin 2α = − cos α, x sin 2α + y(1 − cos 2α) = − sin α.

Using the identities

1 + cos 2α = 2 cos2 α, sin 2α = 2 sin α cos α, 1 − cos 2α = 2 sin2 α, and dividing the first equation by 2 cos α (since cos α > 0) and the second by 2 sin α (since sin α > 0), both reduce to

x cos α + y sin α = −

- 1

- 2

. (16)

- Equation of ℓC Similarly, for ℓC given by z + c2z¯ = −c with c = cos α − i sin α, c2 = cos 2α − i sin 2α. Separating real and imaginary parts yields

x(1 + cos 2α) − y sin 2α = − cos α, −x sin 2α + y(1 − cos 2α) = sin α.

Again, dividing appropriately we obtain

x cos α − y sin α = −

- 1

- 2

. (17)

Equation of the tangent at N = 1 Since N = 1 lies on the unit circle, its tangent is the line z + z¯ = 2, i.e.,

x = 1. (18)

Thus R is the triangle bounded by the three lines (16), (17) and (18).

- 8. Vertices and interior of R The three vertices are the pairwise intersections of the lines.

- ▷ P = ℓB ∩ ℓC: solving (16) and (17) gives

- 1

- 2 cos α

2x cos α = −1 =⇒ x = −

, y = 0.

- ▷ Q′ = ℓC ∩ tangent x = 1: plug x = 1 into (17):

cos α − y sin α = −

1 2

=⇒ y =

cos α + 12 sin α

.

- ▷ R′ = ℓB ∩ tangent x = 1: plug x = 1 into (16):

cos α + 12 sin α

- 1

- 2

cos α + y sin α = −

=⇒ y = −

. Hence the triangle R has vertices

cos α + 12 sin α

cos α + 21 sin α

- 1

- 2 cos α

, 0 , Q′ 1,

, R′ 1, −

P −

.

To determine which side of each line constitutes the interior, we test the vertex opposite that line.

- ▷ For line ℓB, the opposite vertex is Q′. Compute the left-hand side of (16) at Q′:

x cos α + y sin α = 1 · cos α +

cos α + 21 sin α

sin α = cos α + cos α +

- 1

- 2

= 2 cos α +

1 2

> −

1 2

. Therefore the interior of R satisfies

x cos α + y sin α > −

- 1

- 2

. (19a)

- ▷ For line ℓC, opposite vertex is R′. At R′:

cos α + 12 sin α

1 2

- 1

- 2

- 1

- 2

x cos α − y sin α = 1 · cos α − −

> −

, so the interior satisfies

sin α = cos α + cos α +

= 2 cos α +

1 2

. (19b)

x cos α − y sin α > −

▷ For the tangent x = 1, opposite vertex is P with xP = −2cos1 α < 1, thus the interior satisfies x < 1. (19c)

Therefore R = (x, y) : x cos α + y sin α > −21, x cos α − y sin α > −12, x < 1 .

###### 9. Incenter I and inradius r of R

The triangle is symmetric with respect to the x-axis (the lines ℓB and ℓC are symmetric, the tangent is vertical). Hence the incenter lies on the x-axis: I = (p, 0). For a point (x, 0) inside R (so satisfying the inequalities), the distances to the three lines are:

- ▷ To ℓB: the line is x cos α+y sin α+ 12 = 0; distance =

x cos α + 12 cos2 α + sin2 α

= x cos α+ 12 (since the interior gives x cos α + 12 > 0).

- ▷ To ℓC: similarly, distance = x cos α + 12 (because x cos α − 0 + 12 = x cos α + 21 > 0).

▷ To the tangent x = 1: distance = 1 − x (since x < 1). Setting these equal gives

- 1

- 2

- 1

- 2

1 2(1 + cos α)

= 1 − x =⇒ x(cos α + 1) =

=⇒ x =

. Thus the incenter is

x cos α +

1 2(1 + k)

1 2(1 + cos α)

. (20) The common distance is the inradius:

I =

, 0 = (p, 0), p =

1 2(1 + k)

r = 1 − p = 1 −

=

2(1 + k) − 1 2(1 + k)

=

1 + 2k 2(1 + k)

. (21)

###### 10. Distance OI We have the circumcenter

aQT ∆

O = −

, and the incenter

1 2(1 + cos α)

1 D0

, where we denote

I = p =

=

D0 = 2(1 + cos α) = 2(1 + k). Then

aQT ∆ − p = −

aQT + p∆ ∆

O − I = −

.

Set U = aQT + p∆; then OI = |U|/|∆|. Now substitute the expressions for aQT, p and ∆. Write

aQT = a(4k2 − 1)(1 + 2ka) = (4k2 − 1)a + 2k(4k2 − 1)a2.

Also p = 1/D0 and ∆ = a2 − 2a cos 3α + 1. Thus

U = 2k(4k2 − 1) + p a2 + (4k2 − 1) − 2p cos 3α a + p. (22) Compute the coefficients with denominator D0 = 2(1 + k).

###### ▷ Coefficient of a2:

1 D0

A2 = 2k(4k2 − 1) +

. Write as a single fraction:

(2k(4k2 − 1))D0 + 1 D0

A2 =

.

Compute (2k(4k2 − 1))D0 = (8k3 − 2k) · 2(1 + k) = 16k4 + 16k3 − 4k2 − 4k. Adding 1 gives

16k4 + 16k3 − 4k2 − 4k + 1. Notice that

X = 4k2 + 2k − 1, then

X2 = (4k2 + 2k − 1)2 = 16k4 + 16k3 − 4k2 − 4k + 1. Hence

###### ▷ Coefficient of a:

X2 D0

. (23)

A2 =

2 cos 3α D0

A1 = (4k2 − 1) − 2p cos 3α = (4k2 − 1) −

. Write as

(4k2 − 1)D0 − 2 cos 3α D0

A1 =

.

Compute (4k2−1)D0 = (4k2−1)·2(1+k) = 8k3+8k2−2k−2. And 2 cos 3α = 2(4k3−3k) = 8k3−6k. Subtract:

(8k3 + 8k2 − 2k − 2) − (8k3 − 6k) = 8k2 + 4k − 2 = 2(4k2 + 2k − 1) = 2X. Therefore

1 D0

▷ Constant term: p =

. Substituting (23) and (24) into (22) yields

2X D0

. (24)

A1 =

(Xa + 1)2 D0

X2 D0

2X D0

1 D0

a2 +

. (25) Consequently,

U =

a +

=

= |Xa + 1|2 D0 |∆|

OI = |U| |∆|

. (26)

- 11. Relating OI to R and r Recall the circumradius of △AY Z:

R = |a + 2k|2 |∆|

. (27) The inradius of R is

1 + 2k D0

. (28) Now compute

r =

|Xa + 1|2 D0

= |a + 2k|2 + L, where

L = |Xa + 1|2

D0 − |a + 2k|2. From (26),

OI = |a + 2k|2 + L |∆|

L |∆|

. (29)

= R +

Thus if we can show L = ±r |∆|, we will have OI = R ± r, which implies tangency. Let t = a + a¯ = 2 cos θ. Then

|Xa + 1|2 = (Xa + 1)(Xa¯ + 1) = X2 + Xt + 1. Also

|a + 2k|2 = (a + 2k)(¯a + 2k) = 1 + 2kt + 4k2. Hence

X2 + Xt + 1

D0 − (1 + 2kt + 4k2). Multiply by D0:

L =

D0L = X2 + Xt + 1 − D0(1 + 2kt + 4k2). Now D0(1 + 2kt + 4k2) = 2(1 + k) + 4k(1 + k)t + 8k2(1 + k). Thus

D0L = X2 + 1 − 2(1 + k) − 8k2(1 + k)

###### + (X − 4k(1 + k))

###### t.

constant

coefficient of t

Compute the coefficient of t:

X − 4k(1 + k) = (4k2 + 2k − 1) − (4k2 + 4k) = −2k − 1 = −(2k + 1). For the constant term, substitute X2 = 16k4 + 16k3 − 4k2 − 4k + 1:

X2 + 1 − 2(1 + k) − 8k2(1 + k) = (16k4 + 16k3 − 4k2 − 4k + 1) + 1 − 2 − 2k − 8k2 − 8k3

= 16k4 + (16k3 − 8k3) + (−4k2 − 8k2) + (−4k − 2k) + (1 + 1 − 2)

= 16k4 + 8k3 − 12k2 − 6k. Factor this expression:

16k4 + 8k3 − 12k2 − 6k = 2k(8k3 + 4k2 − 6k − 3). Now note that 8k3 + 4k2 − 6k − 3 = (4k2 − 3)(2k + 1), because

(4k2 − 3)(2k + 1) = 8k3 + 4k2 − 6k − 3. Thus

constant = 2k(4k2 − 3)(2k + 1). But 4k2 − 3 = cosk3α (since cos 3α = 4k3 − 3k = k(4k2 − 3)). Hence

cos 3α

constant = 2k ·

k · (2k + 1) = 2(2k + 1) cos 3α. Therefore

D0L = 2(2k + 1) cos 3α + (−(2k + 1))t = (2k + 1)(2 cos 3α − t). (30) Now recall that r = (2k + 1)/D0. So

(2k + 1) = rD0. Substitute into (30):

D0L = rD0(2 cos 3α − t) =⇒ L = r(2 cos 3α − t). (31)

Connection with ∆ Compute ∆ in terms of t:

∆ = a2 − 2a cos 3α + 1 = a(a + a¯ − 2 cos 3α) = a(t − 2 cos 3α). Since |a| = 1, we have

|∆| = |t − 2 cos 3α| = |2 cos 3α − t|. Hence

2 cos 3α − t = ε |∆|, where ε = ±1 depending on the sign of t − 2 cos 3α. Consequently,

Final step Insert (32) into (29):

L = ε r |∆|. (32)

ε r |∆| |∆|

= R + εr = R ± r.

OI = R +

Thus the distance between the centers O and I equals either the sum or the absolute difference of the radii R and r. Therefore the circumcircle of △AY Z and the incircle of △R are tangent. □

###### 12. Remarks on non-vanishing denominators

▷ D1 = 0 would imply ac = b2. In our rotated frame, this gives a = ei3α, which forces AB = BC (since then |a − b| = |ei3α − eiα| = 2 sin α and |b − c| = |eiα − e−iα| = 2 sin α), contradicting the scalene hypothesis. Hence D1 ̸= 0; similarly D2 ̸= 0, so ∆ ̸= 0.

###### ▷ a + 2k = 0 would give a = −2k real; with |a| = 1 we get 2k = 1 i.e. cos α = 1/2, α = 60◦, which is excluded. Thus a + 2k ̸= 0, ensuring R > 0.

▷ cos α > 0 because α is acute. All algebraic manipulations are therefore justified, and the proof is complete.

##### USAMO 2026 Problem 4.

###### USAMO 2026 Problem 4

A positive integer n is called solitary if, for any nonnegative integers a and b such that a + b = n, either a or b contains the digit “1”. Determine, with proof, the number of solitary integers less than 102026.

Solution Remark. 7/7.

- 1. Equivalent formulation Let

S = {x ∈ N0 | the decimal representation of x contains no digit 1}. (Note that 0 ∈ S because its representation ”0” has no digit 1.)

If n = a + b with a, b ∈ S, then the pair (a, b) shows that n is not solitary (both a and b lack a digit 1). Conversely, if for every representation n = a + b at least one of a, b contains a digit 1, then certainly no representation with both in S exists. Hence

n is solitary ⇐⇒ n ∈/ S + S, where S + S = {x + y | x, y ∈ S}. We need to count the positive integers n < 102026 that are not in S + S. Set

N = 2026. We will first count how many numbers 0 ≤ n < 10N belong to S + S; then subtract to get the number of positive solitary integers.

- 2. Digit-wise analysis and carries Every integer m with 0 ≤ m < 10N can be written uniquely as

m =

N−1

i=0

di10i,

where each digit di is in {0, 1, . . . , 9}. To have a uniform treatment, we pad the representation with leading zeros so that every such m uses exactly N digits.

- If a, b ∈ S, then each of their digits belongs to

D = {0, 2, 3, 4, 5, 6, 7, 8, 9} (all digits except 1).

Write a = ai10i, b = bi10i with ai, bi ∈ D. The addition a + b = n proceeds digit by digit with carries c0, c1, . . . , cN:

ai + bi + ci = ni + 10 ci+1, c0 = 0, where each ci is either 0 or 1 (since the maximum sum is 9 + 9 + 1 = 19). Because n < 10N, the final carry must be cN = 0.

- 3. Possible sums of two digits from D Define

Σ = {ai + bi | ai, bi ∈ D}.

Since the only digit missing from D is 1, the sums that can be obtained are

Σ = {0} ∪ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}.

(Indeed, 1 cannot be expressed as a sum of two digits from D; all other integers from 0 to 18 can.)

- 4. Transition sets T(s, d) For a fixed carryin s ∈ {0, 1} and a target digit d ∈ {0, . . . , 9}, let

T(s, d) = t ∈ {0, 1} ∃ai, bi ∈ D with ai + bi + s = d + 10t . Using Σ, we can compute T(s, d).

- Case s = 0 Here the total sum before splitting is simply s0 with s0 ∈ Σ.

▷ If s0 ≤ 9, we may take t = 0 and the digit is s0.

▷ If s0 ≥ 10, we may take t = 1 and the digit is s0 − 10. Thus:

|d<br><br>|T(0, d)|
|---|---|
|0<br><br>1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br><br>|{0, 1} (totals 0 and 10) {1} (total 11)<br><br>{0, 1} (totals 2 and 12)<br>{0, 1} (totals 3 and 13)<br>{0, 1} (totals 4 and 14)<br>{0, 1} (totals 5 and 15)<br>{0, 1} (totals 6 and 16)<br>{0, 1} (totals 7 and 17)<br>{0, 1} (totals 8 and 18) {0} (total 9)<br>|

In words:

▷ d = 9 gives only t = 0. ▷ d = 1 gives only t = 1.

- ▷ The other eight digits (0, 2, 3, 4, 5, 6, 7, 8) allow both t = 0 and t = 1.

Case s = 1 Now the total sum is s0 + 1 with s0 ∈ Σ. Hence the attainable totals are

Σ + 1 = {1} ∪ {3, 4, 5, . . . , 19},

i.e., all integers from 1 to 19 except 2. Proceeding analogously:

|d<br><br>|T(1, d)|
|---|---|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>|{1} (total 19)<br><br>{0, 1} (totals 1 and 11)<br><br>{1} (total 12)<br><br><br>{0, 1} (totals 3 and 13)<br><br>{0, 1} (totals 4 and 14)<br><br>{0, 1} (totals 5 and 15)<br><br>{0, 1} (totals 6 and 16)<br><br>{0, 1} (totals 7 and 17)<br><br>{0, 1} (totals 8 and 18)<br><br>{0, 1} (totals 9 and 19)<br>|

Thus:

- ▷ d = 0 and d = 2 give only t = 1.

- ▷ The other eight digits (1, 3, 4, 5, 6, 7, 8, 9) allow both t = 0 and t = 1.

- 5. State of the possible carries For a given n with digits n0, n1, . . . , nN−1 (least significant first), consider all sequences of carries c0, c1, . . . , cN that can arise from some choices of ai, bi ∈ D. Define

Ri = { ci | there exist aj, bj ∈ D (j < i) such that the carry after processing digits 0, . . . , i − 1 is ci }. By definition, R0 = {0}. For i ≥ 0,

T(s, ni). (1)

Ri+1 =

s∈Ri

Each T(s, d) is a subset of {0, 1}, and a simple check shows it is never empty. Hence every Ri is a non-empty subset of {0, 1}. Consequently, Ri can only be one of three types:

▷ Ai: Ri = {0} (only zero carry possible),

▷ Bi: Ri = {1} (only one carry possible), ▷ Ci: Ri = {0, 1} (both carries possible). Let

Ai = #{ n ∈ [0, 10i) | Ri = {0} }, Bi = #{ n ∈ [0, 10i) | Ri = {1} }, Ci = #{ n ∈ [0, 10i) | Ri = {0, 1} }.

- 6. Transition counts We now determine, given the current state Ri, for which next digits d = ni we obtain each possible next state Ri+1.

- ▷ If Ri = {0}: then Ri+1 = T(0, d). From the table for s = 0:

▷ d = 9 gives {0}; ▷ d = 1 gives {1};

- ▷ the other eight digits (0, 2, 3, 4, 5, 6, 7, 8) give {0, 1}. Hence:

- ▷ 1 digit yields state {0},
- ▷ 1 digit yields state {1},

▷ 8 digits yield state {0, 1}.

▷ If Ri = {1}: Ri+1 = T(1, d). From the table for s = 1:

▷ d = 0 and d = 2 give {1};

- ▷ the other eight digits (1, 3, 4, 5, 6, 7, 8, 9) give {0, 1}. Hence:
- ▷ 2 digits yield state {1},

▷ 8 digits yield state {0, 1}.

▷ If Ri = {0, 1}: Ri+1 = T(0, d) ∪ T(1, d). Checking each digit individually shows that this union is always {0, 1}. Indeed:

- ▷ d = 0: T(0, 0) = {0, 1}, T(1, 0) = {1} → union = {0, 1};
- ▷ d = 1: T(0, 1) = {1}, T(1, 1) = {0, 1} → union = {0, 1};
- ▷ d = 2: T(0, 2) = {0, 1}, T(1, 2) = {1} → union = {0, 1};

▷ d = 3, . . . , 8: both sets contain 0 and 1;

▷ d = 9: T(0, 9) = {0}, T(1, 9) = {0, 1} → union = {0, 1}. Therefore, all 10 digits keep the state as {0, 1}. Summarising the transitions in a matrix:

|from\to<br><br>|{0} {1} {0, 1}|
|---|---|
|{0}<br>{1}<br><br><br>{0, 1}|1 1 8 0 2 8 0 0 10<br><br>|

- 7. Recurrence relations A number with i + 1 digits (i.e., an integer in [0, 10i+1)) is obtained by taking a number m in [0, 10i) (which

yields a certain state) and appending a new most significant digit d (which becomes ni). The number of ways to reach each state for length i + 1 is therefore:

Ai+1 = 1 · Ai + 0 · Bi + 0 · Ci = Ai, Bi+1 = 1 · Ai + 2 · Bi + 0 · Ci = Ai + 2Bi, Ci+1 = 8 · Ai + 8 · Bi + 10 · Ci.

(2)

Initial conditions: For i = 0 we have processed no digits. The only possible ”number” is 0, and the only possible carry is c0 = 0. Hence

A0 = 1, B0 = 0, C0 = 0.

- 8. Solving the recurrences

- 8.1. Ai From Ai+1 = Ai and A0 = 1, we immediately obtain

|Ai = 1 for all i ≥ 0|
|---|

.

- 8.2. Bi Substituting Ai = 1 into the recurrence for Bi:

Bi+1 = 1 + 2Bi, B0 = 0. We claim

|Bi = 2i − 1 for all i ≥ 0|
|---|

. Proof by induction.

###### ▷ i = 0: 20 − 1 = 0, true.

▷ Assume Bi = 2i − 1. Then

###### Bi+1 = 1 + 2(2i − 1) = 1 + 2i+1 − 2 = 2i+1 − 1, which completes the induction.

- 8.3. Ci Now substitute Ai = 1 and Bi = 2i − 1 into the recurrence for Ci+1:

Ci+1 = 8 · 1 + 8 · (2i − 1) + 10 Ci = 8 + 8 · 2i − 8 + 10 Ci

= 8 · 2i + 10 Ci. With C0 = 0. We claim

|Ci = 10i − 2i for all i ≥ 0|
|---|

. Proof by induction.

###### ▷ i = 0: 100 − 20 = 1 − 1 = 0, true.

▷ Assume Ci = 10i − 2i. Then

Ci+1 = 8 · 2i + 10(10i − 2i)

= 8 · 2i + 10i+1 − 10 · 2i

= 10i+1 − 2 · 2i = 10i+1 − 2i+1,

which establishes the claim.

- 9. Representable numbers After processing all N digits, a number n ∈ [0, 10N) can be written as a + b with a, b ∈ S iff there exists a carry sequence with cN = 0. In terms of the sets, this is equivalent to 0 ∈ RN. The possible values of RN are {0}, {1}, or {0, 1}. Among these, 0 ∈ RN exactly when RN = {0} or RN = {0, 1}. Therefore

|S + S ∩ [0, 10N)| = AN + CN. Plugging in the formulas:

|S + S ∩ [0, 10N)| = 1 + (10N − 2N) = 10N − 2N + 1. This count includes n = 0 (since 0 = 0 + 0 and 0 ∈ S).

- 10. Positive solitary numbers We are interested in positive integers n < 10N that are solitary, i.e., n > 0 and n ∈/ S + S. Total positive integers less than 10N are

10N − 1.

The number of positive integers that belong to S + S (hence are not solitary) is

|(S + S) \ {0}| = (10N − 2N + 1) − 1 = 10N − 2N. Thus the number of positive solitary integers is

(10N − 1) − (10N − 2N) = 2N − 1.

- 11. Final answer For N = 2026, we obtain

|22026 − 1|
|---|

.

- 12. Small-case verification (optional)

- ▷ N = 1: numbers 1, . . . , 9. The formula gives 21 − 1 = 1 solitary number. Indeed, n = 1 is solitary because every pair (a, b) with a + b = 1 has either a = 0, b = 1 or a = 1, b = 0; in both cases the number 1 contains a digit 1. All other n (e.g., 2) have a representation with both numbers lacking digit 1 (e.g., 2 = 2 + 0), so they are not solitary.

- ▷ N = 2: numbers 1, . . . , 99. The formula gives 22 − 1 = 3 solitary numbers. One can check that they are 1, 19, 21. This matches the known pattern. The solution is complete.

##### USAMO 2026 Problem 5.

###### USAMO 2026 Problem 5

Let ABC be a triangle. Points D, E, and F lie on sides BC, CA, and AB, respectively, such that

###### ∠AFE = ∠BDF = ∠CED.

Let OA, OB, and OC be the circumcenters of triangles AFE, BDF, and CED, respectively. Let M, N, and O be the circumcenters of triangles ABC, DEF, and OAOBOC, respectively. Prove that OM = ON.

Solution Remark. 7/7.

We are given triangle ABC with points D ∈ BC, E ∈ CA, F ∈ AB such that

∠AFE = ∠BDF = ∠CED =: θ. We shall use complex numbers. Without loss of generality we may place the triangle in the complex plane and denote by a, b, c the complex numbers corresponding to A, B, C.

- 1. Angle Chase Denote the angles of △ABC by α = ∠BAC, β = ∠ABC, γ = ∠ACB. At vertex F

- Points A, F, B are collinear, so ∠AFB = 180◦. The three angles around F on the side of AB that contains the triangle are ∠AFE, ∠EFD, ∠DFB and they sum to 180◦:

∠AFE + ∠EFD + ∠DFB = 180◦. (1)

In △BDF we have ∠BDF = θ and ∠DBF = β (since D lies on BC and F on AB). Hence

∠BFD = 180◦ − θ − β,

so ∠DFB = 180◦ − θ − β. Inserting ∠AFE = θ into (1) gives

θ + ∠EFD + (180◦ − θ − β) = 180◦ =⇒ ∠EFD = β. Thus

∠DFE = β. (2)

- At vertex D

Points B, D, C are collinear, so ∠BDC = 180◦. The angles at D are ∠BDF, ∠FDE, ∠EDC and they sum to 180◦:

∠BDF + ∠FDE + ∠EDC = 180◦. (3)

Given ∠BDF = θ. In △CED, ∠CED = θ and ∠ECD = γ, so

∠CDE = 180◦ − θ − γ,

i.e., ∠EDC = 180◦ − θ − γ. Substituting into (3):

θ + ∠FDE + (180◦ − θ − γ) = 180◦ =⇒ ∠FDE = γ. Thus

∠EDF = γ. (4)

- At vertex E

- Points C, E, A are collinear, so ∠CEA = 180◦. The angles at E are ∠CED, ∠DEF, ∠FEA and they sum to 180◦:

∠CED + ∠DEF + ∠FEA = 180◦. (5)

Given ∠CED = θ. In △AFE, ∠AFE = θ and ∠FAE = α, so

i.e., ∠FEA = 180◦ − θ − α. Substituting into (5):

∠AEF = 180◦ − θ − α,

θ + ∠DEF + (180◦ − θ − α) = 180◦ =⇒ ∠DEF = α. Thus

∠DEF = α. (6) From (2), (4), (6) we obtain

###### ∠DFE = β, ∠EDF = γ, ∠DEF = α.

Therefore △DEF has angles α, β, γ; it is similar to △ABC. The vertex correspondence is

###### E ←→ A, F ←→ B, D ←→ C. (7)

- 2. Complex Representation of the Similarity The similarity (7) may be either orientation-preserving (direct) or orientation-reversing (opposite). The statement OM = ON involves only distances, which are invariant under reflection. Hence we may reflect the whole configuration if necessary and assume that the similarity is direct. Consequently there exist a non-zero complex number k and a complex number t such that

e = ka + t, f = kb + t, d = kc + t. (8)

(If k = 0, then e = f = t, making △AFE degenerate, contrary to the existence of its circumcenter OA.)

- 3. Eliminating the Translation Assume for contradiction that k = 1. Then (8) becomes

- e = a + t, f = b + t, d = c + t. Because F lies on AB, the points A, F, B are collinear; hence
- f − a b − a

=

b + t − a b − a

= 1 +

t

b − a ∈ R. Similarly, D lies on BC, so

d − b c − b

=

c + t − b c − b

= 1 +

t c − b ∈ R,

thus b−ta ∈ R and c−tb ∈ R. The vectors b − a and c − b are not parallel (they are sides of a non-degenerate triangle), so the only complex number that is a real multiple of both is 0. Hence t = 0, which gives e = a, f = b, d = c. Then △AFE degenerates to the segment AF, contradicting the existence of OA. Therefore k ̸= 1. Consider the fixed point of the spiral similarity z  → kz + t (when k ̸= 1):

X =

t 1 − k

.

Translate the plane so that X becomes the origin. (Translation is an isometry, so all distances and circumcenters are preserved up to the same translation; we keep the same letters for the new coordinates.) After this translation we have

e = ka, f = kb, d = kc. (9) (We also note that a, b, c ̸= 0; otherwise, e.g., a = 0 would imply A = X, and then e = k · 0 = 0, so A and E coincide, making △AFE degenerate - impossible. Hence a, b, c are non-zero.)

- 4. The Origin Lies on the Three Circles

We now show that the origin 0 belongs to the circumcircles of △AFE, △BDF, and △CED. Lemma (Cross Ratio and Concyclicity). Four distinct points z1, z2, z3, z4 in the complex plane lie on a common circle or line if and only if their cross ratio

is a real number. Proof. The M¨obius transformation

- (z1 − z3)(z2 − z4)

- (z1 − z4)(z2 − z3)

(z1, z2; z3, z4) =

(z − z1)(z2 − z3) (z − z3)(z2 − z1)

T(z) =

maps z1, z2, z3 to 0, 1, ∞ respectively. Under T, the circle/line through z1, z2, z3 is mapped to the real line. Hence z4 lies on that circle/line exactly when T(z4) is real, and T(z4) equals the cross ratio (z1, z2; z3, z4).

Circle (AFE)

- Take z1 = a, z2 = f, z3 = e, z4 = 0. Using (9):

a − e = a − ka = a(1 − k), f − 0 = kb, a − 0 = a, f − e = kb − ka = k(b − a). Thus

(a, f; e, 0) =

(a − e)(f − 0) (a − 0)(f − e)

=

a(1 − k) · kb a · k(b − a)

=

(1 − k)b b − a

. (10)

Because A, F, B are collinear, the ratio

f − a b − a

is real. Compute

f − a b − a

=

kb − a b − a

. Now

(1 − k)b b − a

= 1 −

kb − a b − a

,

which is therefore real. The points a, f, e are not collinear (otherwise △AFE would be degenerate). By the lemma, 0 lies on the circle through a, f, e, i.e.,

0 ∈ (AFE). (11)

Circle (BDF)

- Take z1 = b, z2 = d, z3 = f, z4 = 0. Using (9):

b − d = b − kc, f − 0 = kb, b − 0 = b, f − d = kb − kc = k(b − c). Hence

(b, d; f, 0) =

(b − d)(f − 0) (b − 0)(f − d)

=

(b − kc) · kb b · k(b − c)

=

b − kc b − c

. (12)

- Collinearity of B, D, C gives

d − b c − b ∈ R:

kc − b

c − b ∈ R. Observe that

b − kc b − c

= −(kc − b) −(c − b)

=

kc − b c − b

,

so the cross ratio is real. Since b, d, f are non-collinear (triangle BDF is non-degenerate), the lemma yields

0 ∈ (BDF). (13)

Circle (CED)

- Take z1 = c, z2 = e, z3 = d, z4 = 0. Using (9):

c − e = c − ka, d − 0 = kc, c − 0 = c, d − e = kc − ka = k(c − a). Thus

(c − e)(d − 0) (c − 0)(d − e)

(c − ka) · kc c · k(c − a)

c − ka c − a

. (14)

(c, e; d, 0) =

=

=

e − c a − c ∈ R:

- Collinearity of C, E, A gives

ka − c

a − c ∈ R. Since

= −(ka − c) −(a − c)

c − ka c − a

ka − c a − c

, the cross ratio is real. Non-collinearity of c, e, d gives

=

###### 0 ∈ (CED). (15)

- 5. k is Not Real Suppose, for contradiction, that k ∈ R. Then from (9) we have e = ka, so points 0, a, e are collinear (they lie on the line through the origin and a). However, by (11) all three points 0, a, e lie on the circle (AFE). A non-degenerate circle cannot contain three distinct collinear points. This contradiction forces

k ∈/ R. (16)

(Recall k ̸= 0, 1; therefore 0, 1, k are distinct and not collinear.)

- 6. Circumcenters OA, OB, OC Since 0 lies on (AFE), the circle through a, f, e coincides with the circle through 0, a, e (because three noncollinear points determine a circle, and 0, a, e are distinct and, because k ∈/ R, they are not collinear). Hence OA is the circumcenter of △(0, a, e). Now consider the triangle with vertices 0, 1, k. As k ∈/ R and k ̸= 0, 1, these points are distinct and non-collinear. Let µ be its circumcenter; thus

|µ − 0| = |µ − 1| = |µ − k|. (17)

We claim that OA = aµ. Indeed, the map z  → az (a spiral similarity about 0) sends

0  → 0, 1  → a, k  → ka = e. Under a similarity, the image of the circumcenter of a triangle is the circumcenter of the image triangle. Therefore aµ is the circumcenter of △(0, a, e), i.e., OA = aµ. One may also verify directly: The distances from aµ to 0, a, e are

|aµ − 0| = |a| |µ|, |aµ − a| = |a| |µ − 1|, |aµ − e| = |a| |µ − k|, and by (17) these three numbers are equal, so aµ is equidistant from 0, a, e. Because 0, a, e are not collinear, aµ is the unique point with that property, hence it is the circumcenter. Analogously, applying the same reasoning to the other two circles, we obtain

OB = bµ, OC = cµ. (18)

- 7. Circumcenters M, N, O Let M be the circumcenter of △ABC, so

|M − a| = |M − b| = |M − c| =: R. (19)

Circumcenter O of △OAOBOC The vertices are OA = aµ, OB = bµ, OC = cµ. Consider the point µM:

- |µM − OA| = |µM − aµ| = |µ| |M − a| = |µ|R,
- |µM − OB| = |µM − bµ| = |µ| |M − b| = |µ|R,

- |µM − OC| = |µM − cµ| = |µ| |M − c| = |µ|R.

Thus µM is equidistant from OA, OB, OC. Since △OAOBOC is non-degenerate (it is the image of △ABC under the non-zero similarity z  → µz), µM is the circumcenter:

###### O = µM. (20)

Circumcenter N of △DEF The vertices are D = kc, E = ka, F = kb. Consider the point kM:

- |kM − D| = |kM − kc| = |k| |M − c| = |k|R,
- |kM − E| = |kM − ka| = |k| |M − a| = |k|R,
- |kM − F| = |kM − kb| = |k| |M − b| = |k|R.

Hence kM is equidistant from D, E, F, and because △DEF is non-degenerate (it is similar to △ABC and, under the given configuration, non-degenerate), we have

N = kM. (21)

(If M = 0, then O = 0, N = 0, and the equalities hold trivially.)

- 8. Proof of OM = ON Using (20) and (21),

- OM = |O − M| = |µM − M| = |M| |µ − 1|,
- ON = |O − N| = |µM − kM| = |M| |µ − k|.

Now µ is the circumcenter of △(0, 1, k), so by (17)

|µ − 1| = |µ − k|. Consequently,

OM = |M| |µ − 1| = |M| |µ − k| = ON.

- 9. Conclusion We have shown that under the given conditions the distances OM and ON are equal.

##### USAMO 2026 Problem 6.

|OM = ON|
|---|

.

###### USAMO 2026 Problem 6

Let a and b be positive integers such that φ(ab + 1) divides a2 + b2 + 1. Prove that a and b are Fibonacci numbers.

Solution Remark. 7/7.

Let a and b be positive integers such that

φ(ab + 1) | a2 + b2 + 1. Set N = ab + 1 and define the positive integer

a2 + b2 + 1 φ(N)

k =

. We will prove that both a and b must be Fibonacci numbers.

- 1. Preliminary bounds Since N > 1, we have φ(N) ≤ N − 1 = ab (with equality exactly when N is prime). Hence

a2 + b2 + 1 = k φ(N) ≤ k ab. (1) By the quadratic-mean-arithmetic-mean inequality, a2 + b2 ≥ 2ab, so

a2 + b2 + 1 ≥ 2ab + 1. (2) Combining (1) and (2) yields

1 ab

2ab + 1 ≤ k ab =⇒ k ≥ 2 +

.

Because k is an integer, if ab > 1 we obtain k ≥ 3. If ab = 1 then necessarily a = b = 1, N = 2, φ(2) = 1, and then k = 3. Thus in every case

|k ≥ 3|
|---|

. (3)

- 2. Parity and the even case Lemma 2.1. If N is even and N > 2, then the divisibility condition cannot hold. Proof. For N > 2, φ(N) is even (a standard fact: φ(n) is even for every n > 2). Because N even implies ab = N − 1 odd, both a and b are odd. Then a2 + b2 + 1 is odd (odd+odd+1). An even number cannot divide an odd number. The only even possibility is N = 2, which gives a = b = 1 and indeed φ(2) = 1 divides 3. Consequently, the only even N that can occur is N = 2, producing (a, b) = (1, 1) - both Fibonacci numbers. From now on we assume

N is odd and N > 2. Then ab = N − 1 is even, so at least one of a, b is even. If both were even, a2 + b2 + 1 would be odd, while for odd N > 2 we still have φ(N) even. Hence exactly one of a, b is even and the other is odd. In particular,

a2 + b2 + 1 ≡ 0 + 1 + 1 ≡ 2 (mod 4). (4)

- 3. Case I: N is prime Assume N = ab + 1 is prime. Then φ(N) = N − 1 = ab, and (1) becomes

a2 + b2 + 1 = k ab. (5) We first determine k.

- 3.1. The value of k Without loss of generality, order the variables so that a ≤ b. Write (5) as a quadratic in b:

- If b is a root, the other root is

###### b2 − (ka) b + (a2 + 1) = 0.

- a2 + 1

- b

b′ =

= ka − b,

which is a positive integer. Claim. If a ≥ 2, then b′ < b.

###### Proof. Suppose b′ ≥ b. Then a2b+1 ≥ b, i.e. a2 + 1 ≥ b2. Because b ≥ a, we have a2 ≤ b2 ≤ a2 + 1. Hence either b2 = a2 or b2 = a2 + 1.

▷ If b2 = a2, then b = a. Substituting b = a into (5) gives 2a2 + 1 = ka2 ⇒ k = 2 + a12 , which is not an integer for a ≥ 2. Contradiction.

▷ If b2 = a2 + 1, then a2 + 1 is a perfect square. For a ≥ 2, this is impossible because between a2 and (a + 1)2 = a2 + 2a + 1 there is no other square.

Thus whenever a ≥ 2, we can replace (a, b) by (a, b′) (which is also a solution of (5)) with a + b′ < a + b. Repeating this descent (the sum a + b strictly decreases each step) we eventually reach a solution with a = 1 (the minimal sum cannot have a ≥ 2).

Now consider a = 1. Equation (5) becomes

1 + b2 + 1 = k · 1 · b =⇒ b2 − kb + 2 = 0. (6) For integer b, the discriminant ∆ = k2 − 8 must be a perfect square, say d2. Then

###### k2 − d2 = 8 =⇒ (k − d)(k + d) = 8.

Both factors are positive integers of the same parity. The factor pairs of 8 are (1, 8) and (2, 4).

- ▷ (1, 8) gives k − d = 1, k + d = 8 ⇒ 2k = 9 ⇒ k = 9/2, not integer.
- ▷ (2, 4) gives k − d = 2, k + d = 4 ⇒ 2k = 6 ⇒ k = 3, d = 1. Hence k = 3 is forced. With k = 3, equation (6) is b2 −3b+2 = 0, whose roots are b = 1 and b = 2. Thus the only solutions with a = 1 are (1, 1) and (1, 2) (and by symmetry, (2, 1) if we had not assumed a ≤ b). Since the reduction process preserves k, every solution of (5) must have k = 3. Therefore, when N is prime,

a2 + b2 + 1 = 3ab ⇐⇒ a2 − 3ab + b2 = −1. (7)

- 3.2. Solving a2 − 3ab + b2 = −1 Equation (7) is symmetric; we may assume a ≤ b. Define a sequence (An)n≥0 by

A0 = 1, A1 = 1, An+2 = 3An+1 − An (n ≥ 0). One verifies by induction that (An, An+1) satisfies (7): For n = 0, 12 − 3 · 1 · 1 + 12 = −1. Assuming it holds for n, using the recurrence one checks that it also holds for n + 1. (The computation is straightforward and can be filled in similarly to the proof in Lemma 4.1 of the original draft.)

Now let (a, b) be any positive integer solution of (7) with a ≤ b. View (7) as a quadratic in b:

b2 − 3a b + (a2 + 1) = 0. Its two roots are b and

- a2 + 1

- b

b′ =

= 3a − b,

which is an integer. A similar argument as in 3.1 shows that if a ≥ 2, then b′ < b. Consequently, by infinite descent on a + b, we can reduce any solution to one with a = 1. The only solutions with a = 1 are (1, 1) and (1, 2) (and their symmetric versions). Therefore every solution can be obtained from these minimal ones by the inverse transformation: if (x, y) is a solution with x ≤ y, then (y, 3y − x) is also a solution. Starting from (1, 1) and repeatedly applying this transformation yields the increasing sequence

(1, 1), (1, 2), (2, 5), (5, 13), (13, 34), . . . which exactly corresponds to the pairs (An, An+1) for n ≥ 0. It remains to identify these An with Fibonacci numbers. Define the Fibonacci numbers by

F1 = 1, F2 = 1, Fn+2 = Fn+1 + Fn. We claim that for n ≥ 1,

An = F2n−1. Proof by induction.

▷ Base n = 1: A1 = 1 = F1.

▷ n = 2: A2 = 3A1 − A0 = 3 · 1 − 1 = 2 = F3.

▷ Inductive step: Assume An = F2n−1 and An+1 = F2n+1. Then

An+2 = 3An+1 − An = 3F2n+1 − F2n−1. Using the identities

F2n+1 − F2n−1 = F2n, and F2n+1 + F2n = F2n+2, we compute

3F2n+1 − F2n−1 = 2F2n+1 + (F2n+1 − F2n−1)

= 2F2n+1 + F2n

= F2n+1 + (F2n+1 + F2n)

= F2n+1 + F2n+2

= F2n+3. Thus An+2 = F2n+3 = F2(n+2)−1, completing the induction. Hence all solutions of (7) are pairs of Fibonacci numbers (specifically, consecutive odd-index Fibonacci numbers). This completes Case I.

###### 4. Case II: N is composite and odd (N > 2)

Now N = ab + 1 is composite, odd, and N > 2. From (4) we have

###### a2 + b2 + 1 ≡ 2 (mod 4).

- 4.1. N cannot have two distinct odd prime factors Suppose N has at least two distinct odd prime divisors p and q. Then φ(N) is divisible by (p − 1)(q − 1). Since p and q are odd, both p − 1 and q − 1 are even, hence 4 | (p − 1)(q − 1) | φ(N). Thus 4 | φ(N). Since φ(N) | a2 + b2 + 1, we would have 4 | a2 + b2 + 1, contradicting (4). Therefore N cannot possess two distinct odd prime factors. It follows that N must be a prime power:

N = pe, e ≥ 2 (since N is composite).

- 4.2. The prime p cannot be ≡ 1 (mod 4) If p ≡ 1 (mod 4), then p − 1 is divisible by 4, so 4 | φ(N). Again this forces 4 | a2 + b2 + 1, which is impossible by (4). Hence

p ≡ 3 (mod 4). (8)

- 4.3. Analysis modulo 3 We now examine the residue of p modulo 3.

- Subcase p ≡ 1 (mod 3). Then 3 | (p−1), so 3 | φ(N). Moreover, p ≡ 1 (mod 3) implies pe ≡ 1 (mod 3), so ab = pe − 1 ≡ 0 (mod 3). Thus 3 | ab. Now evaluate S = a2 + b2 + 1 modulo 3. Since 3 | ab, there are two possibilities:

▷ 3 | a and 3 | b: then a2 ≡ 0, b2 ≡ 0, so S ≡ 1 (mod 3).

▷ Exactly one of a, b is divisible by 3: then S ≡ 0 + 1 + 1 = 2 (mod 3). In neither case is S divisible by 3. However, φ(N) | S and 3 | φ(N) would force 3 | S. Contradiction. Therefore p ̸≡ 1 (mod 3).

- Subcase p ≡ 2 (mod 3). Then p − 1 ≡ 1 (mod 3), so 3 ∤ (p − 1). But because e ≥ 2, φ(N) = pe−1(p − 1) contains the factor pe−1, so in particular p | φ(N). Consequently,

p | S. (9) Also, from ab = pe − 1 we have

ab ≡ −1 (mod p). (10) In particular, p does not divide a or b (otherwise (10) would give 0 ≡ −1 (mod p), impossible). Hence a and b are invertible modulo p. Write b ≡ −a−1 (mod p). Substituting into (9):

a2 + (−a−1)2 + 1 ≡ 0 (mod p), i.e.,

a2 + a−2 + 1 ≡ 0 (mod p). Multiply by a2 (invertible modulo p):

a4 + a2 + 1 ≡ 0 (mod p). (11) Set y = a2 (mod p); then (11) becomes

y2 + y + 1 ≡ 0 (mod p). (12) Equation (12) has a solution in Fp iff −3 is a quadratic residue modulo p. Indeed, multiplying (12) by 4 gives (2y + 1)2 ≡ −3 (mod p). We need the following standard lemma: Lemma. For an odd prime p ̸= 3, −3 is a quadratic residue modulo p if and only if p ≡ 1 (mod 3).

Proof. The multiplicative group F×p is cyclic of order p − 1. The equation y2 + y + 1 = 0 is equivalent to y3 = 1 with y ̸= 1 (since (y − 1)(y2 + y + 1) = y3 − 1. Hence a solution exists iff the group contains an element of order 3, i.e., iff 3 | p − 1, which is p ≡ 1 (mod 3). Conversely, if p ≡ 1 (mod 3), then 3 | p − 1, so such an element exists.

Since p ≡ 2 (mod 3), we have p − 1 not divisible by 3, so there is no element of order 3 in F×p ; therefore (12) has no solution modulo p. This contradicts the existence of a. Thus p ̸≡ 2 (mod 3). The only remaining possibility for the odd prime p is p = 3.

- 4.4. The exponent e Now N = 3e with e ≥ 2. Then

φ(N) = 3e−1 · 2. If e ≥ 3: Then 3e−1 is at least 32 = 9, so 9 | φ(N). Since φ(N) | S, we must have 9 | S. But we will show that S ≡ 3 (mod 9), which is not divisible by 9, a contradiction. Let us verify this congruence. Because e ≥ 3, N = 3e is divisible by 27, in particular N ≡ 0 (mod 9). Hence

ab = N − 1 ≡ −1 (mod 9). (13) Moreover, N ≡ 0 (mod 3) gives ab ≡ −1 ≡ 2 (mod 3). Hence 3 ∤ a and 3 ∤ b (otherwise ab ≡ 0 (mod 3)). Thus a and b are each ≡ 1 or 2 (mod 3), and their product is 2 (mod 3); the only possibilities are {a, b} ≡ {1, 2} (mod 3). Consequently,

a + b ≡ 1 + 2 ≡ 0 (mod 3). (14) Now compute

S = a2 + b2 + 1 = (a + b)2 − 2ab + 1. From (14), a + b = 3t for some integer t, so (a + b)2 = 9t2 ≡ 0 (mod 9). Using (13), ab ≡ −1 (mod 9), so −2ab ≡ −2(−1) = 2 (mod 9). Thus

S ≡ 0 + 2 + 1 = 3 (mod 9). Thus 9 ∤ S, contradicting 9 | φ(N) | S. Therefore e cannot be ≥ 3. The only remaining possibility is e = 2, i.e., N = 9. Then φ(9) = 6 and ab = 8. The positive factor pairs of 8 are

(1, 8), (2, 4), (4, 2), (8, 1). Compute S for each:

- ▷ (1, 8): S = 1 + 64 + 1 = 66, and 66/6 = 11 ∈ Z.
- ▷ (2, 4): S = 4 + 16 + 1 = 21, 21/6 = 3.5 ∈/ Z. Thus only (1, 8) and (8, 1) satisfy the divisibility condition. Both 1 and 8 are Fibonacci numbers (1 = F1, 8 = F6).

- 5. Conclusion We have examined all possible values of N = ab + 1:

▷ Even case: N = 2 gives (a, b) = (1, 1) - Fibonacci numbers.

▷ Prime case: N prime leads to the equation a2 − 3ab + b2 = −1, whose complete set of positive integer solutions consists of pairs of Fibonacci numbers (as shown in 3.2).

▷ Composite case: N composite forces N = 9 and yields (a, b) = (1, 8) or (8, 1), again Fibonacci numbers.

Hence, under the hypothesis φ(ab + 1) | a2 + b2 + 1, the integers a and b are necessarily Fibonacci numbers. Remark. The Fibonacci numbers are defined by F1 = F2 = 1 and Fn+2 = Fn+1 + Fn. The solutions from the prime case are precisely (F2n−1, F2n+1) (and symmetric) for n ≥ 1, together with the degenerate pair (1, 1) which can be seen as (F1, F2) or (F2, F1). The pair (1, 8) from the composite case also fits the Fibonacci sequence (F1 = 1, F6 = 8). This completes the proof.

