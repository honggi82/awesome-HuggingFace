## World Models Meet Language Models: On the Complementarity of Concrete and Abstract Reasoning

Yucheng Zhou1, Wei Tao2, Yiwen Guo3*, Jianbing Shen1* 1University of Macau, 2LIGHTSPEED, 3Independent Researcher yucheng.zhou@connect.um.edu.mo

# arXiv:2606.03603v1[cs.CV]2Jun2026

### Abstract

World models and multimodal large language models (MLLMs) provide complementary capabilities for predicting future outcomes from static visual observations. World models can generate concrete visual rollouts of possible futures, while MLLMs can reason abstractly over questions, goals, and rules. However, generated rollouts are stochastic and may be visually plausible but task-incorrect, making it necessary to determine when visual simulation is useful, whether a rollout is credible, and how it should influence the final answer. We formulate this problem as controlled concrete reasoning, where a model learns to invoke, verify, and integrate visual future simulation alongside abstract reasoning. To study this setting, we construct two humanverified benchmarks, VRQABench for controllable spatial lookahead and OpenWorldQA for open-domain physical prediction, and propose Privileged-Future On-Policy Self-Distillation (PF-OPSD). During training, PF-OPSD uses ground-truth future videos and answers only as teacher-side privileged context to evaluate onpolicy concrete-reasoning trajectories, while the deployable student never observes true futures at test time. Experimental results show that PF-OPSD outperforms baseline by 10.6% and 10.9% on VRQABench and OpenWorldQA, respectively, while increasing robustness to noisy or conflicting rollouts. Our code and dataset are available at https://github.com/ yczhou001/PF-OPSD.

### 1 Introduction

Future-oriented visual reasoning asks a model to answer questions about outcomes that are not yet visible in a static observation. A single image or anchor frame may reveal objects, contacts, spatial constraints, or a puzzle state, but the answer often depends on extrapolating short-horizon dynamics

* Corresponding authors

[Figure 1]

Abstract Reasoning

No Abstract Reasoning

No Concrete Reasoning via visual simulation

Concrete Reasoning via visual simulation

Figure 1: Abstract reasoning organizes goals, rules, and questions in language, while concrete reasoning uses world-model rollouts to make possible futures visually explicit; reliable agents must coordinate both capabilities.

or plans. Recent MLLMs can organize goals, rules, and alternatives in language (Yin et al., 2024; Yoon et al., 2025; Bai et al., 2025), while video world models can make possible futures visually explicit through generated rollouts (Wan et al., 2025; Team, 2025; Yuan et al., 2026; Yue et al., 2025; Zhang et al., 2025). This makes future prediction a natural testbed for combining abstract language-level reasoning with concrete visual simulation.

Figure 1 shows this conceptual motivation: abstract language-level reasoning organizes goals, rules, and questions, while concrete rollout-based reasoning makes possible futures explicit. However, simply attaching a world model to an MLLM does not make this coordination reliable. Generated rollouts are noisy reasoning traces rather than precise oracles: they may miss task-critical interactions, drift from the initial geometry, or produce futures that are visually plausible but answerincorrect (Qian et al., 2026; Yue et al., 2025; Zhang et al., 2025). The key challenge is therefore not whether world models can generate futures, but whether an MLLM can control when those futures should influence reasoning.

A preliminary empirical study makes this challenge concrete. Under optional tool use, models often continue to rely on abstract reasoning even

when simulation would help, which we call Simulation Inertia. Under forced simulation, models may accept misleading rollouts without sufficient scrutiny, leading to the Forced-Simulation Paradox. These failures suggest that world-model assistance requires arbitration between abstract priors and rollout-based evidence, rather than unconditional generation or unconditional trust.

We formulate this problem as controlled concrete reasoning: given an initial observation and a future-oriented question, the MLLM must learn when to invoke a world model, how to verify the resulting rollout, and how much to rely on it when predicting the final answer. The external world model supplies candidate visual futures, while the MLLM remains responsible for simulation selection, rollout verification, rollout reliance, and answer prediction.

To evaluate this setting, we construct two complementary human-verified benchmarks. VRQABench targets controllable spatial lookahead in maze, irregular-maze, and Sokoban-style puzzle environments (Yang et al., 2025); OpenWorldQA targets open-domain physical prediction from pre-event anchor frames in real-world videos. Together, they test future prediction from static initial conditions across structured spatial environments and natural physical scenes.

We further propose Privileged-Future On-Policy Self-Distillation (PF-OPSD), a training framework tailored to these simulation-control decisions. During training, a privileged evaluator observes the ground-truth future video and answer only as teacher-side context, scores the utility of the student’s on-policy concrete-reasoning trajectories, and distills advantage-weighted targets back into a deployable student. At test time, the student has no access to true futures and must decide for itself when to simulate, verify, rely, or fall back to abstract reasoning.

Our contributions are summarized as follows:

- • We frame future outcome prediction as controlled concrete reasoning, where an MLLM learns when to invoke world-model rollouts, how to verify them, and how much to rely on them alongside abstract reasoning.
- • We identify Simulation Inertia and ForcedSimulation Paradox as two failure modes that reveal limits of naive world-model attachment.
- • We construct two human-verified evaluation settings, VRQABench and OpenWorldQA, for control-

lable spatial lookahead and open-domain physical future prediction from initial observations.

• We propose PF-OPSD, which uses privileged future context during training to distill concretereasoning decisions into an MLLM, yielding 10.6% and 10.9% performance gains over baseline on two benchmarks.

### 2 Related Work

LLMs support structured inference and tool use through chain-of-thought, self-consistency, program-aided reasoning, and agentic tool invocation (Wei et al., 2022; Wang et al., 2023; Kojima et al., 2022; Chen et al., 2023; Yao et al., 2023; Schick et al., 2023). MLLMs extend these abilities to image-text reasoning (Yin et al., 2024; Yoon et al., 2025; Bai et al., 2025), while visual and physical reasoning benchmarks study compositional VQA, intuitive physics, and outcome prediction (Johnson et al., 2025; Hudson and Manning, 2019; Bakhtin et al., 2019; Riochet et al., 2018). Recent video models can serve as world models for future rollouts (Tong et al., 2022; Drozdov et al., 2024; Wiedemer et al., 2025; Wan et al., 2025; Team, 2025; Yue et al., 2025; Zhang et al., 2025), but such rollouts may hallucinate or drift from task-relevant geometry (Luo et al., 2025; Cao et al., 2026; Qian et al., 2026). PF-OPSD treats these rollouts as noisy concrete-reasoning traces and instantiates distillation, privileged information, and on-policy learning (Hinton et al., 2015; Vapnik and Vashist, 2009; Schulman et al., 2017) for simulation-control decisions. The full related work is provided in Appendix C.

### 3 Problem Definition and Benchmarks

#### 3.1 Controlled Concrete Reasoning

We define world-model-assisted future prediction as a controlled concrete-reasoning problem. Given a current image or pre-event anchor frame o and a question q, the agent predicts a future-outcome answer y while optionally using a generative world model W. The world model provides concrete reasoning through candidate future rollouts: when invoked with an agent-written prompt, it returns a rollout vˆ rather than an answer. The ground-truth future video v∗ is used only as privileged training information and is unavailable at test time.

The policy produces a concrete-reasoning trajectory instead of consuming vˆ as a fixed input. It first

(a) Simulation Inertia

(b) Forced-Simulation Paradox

100

45.0

98%

Base

| |
|---|

+Helios

96%

| |
|---|

42.5

95

39.8

No-call(%)

40.0

38.6

Acc.(%)

90

37.5

35.0

33.2

85

32.6

32.5

80

30.0

Gemini 3-Flash

Qwen3.5 9B

VRQA OWQA

- Figure 2: Preliminary empirical observations showing simulation inertia and the forced-simulation paradox. Panel (a) reports the no-call probability under optional world-model use, and Panel (b) compares accuracy before and after forced Helios simulation. decides whether simulation is needed,

dsim ∼ πθ(· | o,q), (1)

and, if dsim = 1, writes a simulation prompt and receives a rollout:

psim ∼ πθ(· | o,q,dsim),vˆ ∼ W(· | o,psim). (2)

The same policy then either verifies and relies on the rollout, or falls back to abstract reasoning when no simulation is used:

zver,zrel,y∼πθ(·|o,q,dsim,psim,vˆ), dsim=1 zrel,y∼πθ(·|o,q,dsim), dsim=0 (3)

Here, zver is the rollout-verification decision and zrel is the rollout-reliance or abstract-reasoning fallback process. The objective is to improve future prediction while avoiding negative transfer from erroneous simulations and unnecessary reliance on weak rollouts.

#### 3.2 Why Naive Integration Fails

We conduct a preliminary diagnostic on VRQABench using Gemini-3-Flash (Pichai et al., 2025) as the language agent and Helios (Yuan et al., 2026) as the video world model. As shown in Figure 2, we compare optional world-model use with forced simulation to isolate two limitations of naive worldmodel attachment. First, optional tool use leads to Simulation Inertia: even when prompts encourage simulation for complex spatial reasoning, the agent often relies on abstract reasoning and does not call the world model. Second, forced simulation leads to a Forced-Simulation Paradox: providing a generated rollout for every query does not necessarily improve accuracy and may even hurt performance when the rollout is visually plausible but task-incorrect. These observations show that the core problem is not merely access to more future videos, but learning when to request, trust, discount, or reject noisy rollouts.

Benchmark Type Train Test Coverage

VRQABench Puzzles 4,000 636 5 spatial categories OpenWorldQA Real-World 3,904 500 12 physical categories

Table 1: Overview of the two benchmark distributions. All examples are four-choice questions from an initial image or anchor frame; category definitions and split details are provided in Appendix D, and construction prompts are provided in Appendix F.

3.3 Benchmark Suite for Future Prediction from Static Observations

We introduce two four-choice benchmarks for the same input regime: a model observes only an initial state and must answer a question about a later outcome. VRQABench isolates rule-governed spatial lookahead in puzzle environments, while OpenWorldQA tests open-domain physical prediction from natural videos. Future frames are never part of the benchmark input; generated rollouts, when used, are optional concrete-reasoning inputs controlled by the agent.

VRQABench: Controllable Spatial Lookahead from Initial Puzzle Images. VRQABench is built from VR-Bench (Yang et al., 2025) by turning maze, irregular-maze, and Sokoban states into multiple-choice future-prediction questions. For each puzzle, we first derive the target statistic from the underlying state with deterministic solvers: shortest-path search and geometric path analysis for maze variants, and Sokoban search for boxpushing tasks. We then use language models only for surface realization, writing the question, producing plausible distractors, and filtering item quality, so that labels remain programmatically grounded. After automatic filtering, human annotators verify every retained item for visual consistency, option plausibility, and answer validity; items that fail this final check are removed. The full construction prompt is provided in Appendix F.

The resulting benchmark contains 4,636 humanverified questions, with 4,000 training examples and 636 evaluation examples. Its categories cover turn counting, turn direction, Sokoban pushes, direction counts, and push-direction counts; detailed category definitions and split distributions are provided in Appendix D. For world-model-assisted experiments, VRQABench uses the VR-Bench-finetuned Helios world model (Yuan et al., 2026) only as an external rollout generator; the evaluated model still receives the initial image, question, options, and any invoked rollout.

OpenWorldQA: Predicting Real-World Physical Futures from Anchor Frames. OpenWorldQA uses short real-world videos but exposes only a preoutcome anchor frame to the evaluated model. We construct it with a five-stage agentic pipeline. A scene-analysis stage selects anchor frames that contain enough initial-condition cues without revealing the outcome; a question-design stage writes oneto-three-step physical prediction questions; a distractor stage creates locally plausible alternatives; a small-model probe removes items that are too easy; and a reviewer verifies answer correctness, anchor validity, distractor plausibility, visual consistency, and category alignment. Each surviving item then undergoes human verification, where annotators check whether the anchor frame supports prediction, the future outcome is unambiguous, and the answer/options are physically valid; bad samples are filtered out. Detailed prompts for these stages are included in Appendix F.

The resulting benchmark contains 4,404 humanverified four-choice questions, with 3,904 training examples and a 500-question balanced test set. It spans 12 physical-reasoning categories and six question forms, including order, count, first contact, intermediate state, failure, and counterfactual questions; Appendix D gives the full category taxonomy and split distributions. Table 1 summarizes the benchmark distributions.

### 4 Controlled Concrete Reasoning with PF-OPSD

- Figure 3 illustrates the overall PF-OPSD pipeline. The central difficulty is not access to the world model W, but controlling how uncertain rollouts are used for concrete reasoning. Given an input x = (o,q,O), we define the induced trajectory distribution as follows:

Nsim(τ)

W(ˆv(i)|o,p(simi) ). (4)

pθ(τ |x;W)=

πθ(at | ht)·

i=1

t∈T (τ)

Here, τ contains the control actions and final answer, ht is the history before action at, and W only samples candidate futures. In our experiments, W is instantiated by Helios (Yuan et al., 2026): VRQABench uses the VR-Bench-fine-tuned Helios model, while OpenWorldQA uses the general Helios model.

PF-OPSD trains this deployable student policy with asymmetric information. During training,

a privileged evaluator E+ observes the groundtruth future v∗ and answer y∗ as teacher-side context, and uses them to score candidate concretereasoning trajectories generated by the student. During inference, E+, v∗, and y∗ are removed; the student receives only x and optional rollouts from W. The learning problem is

J(θ)= E(x,y∗,v∗)Eτ∼pθ(·|x;W) R+(τ;y∗,v∗) , θ∗= arg max

J(θ), (y∗,v∗) ∈/ xtest. (5)

θ

Thus, privileged futures and answers are used only to construct training targets for the student’s own trajectories, not as test-time inputs.

#### 4.1 Policy Trajectory and Action Space

Given an observation o and a question q, the student policy emits a concrete-reasoning trajectory rather than consuming a rollout as a fixed input. Without simulation, the trajectory is

##### τno = (dsim,zrel,y). (6)

When simulation is invoked, the trajectory contains a bounded sequence of simulation attempts:

τsim=(dsim,A1:m,zrel,y),Ai=(p(simi) ,vˆ(i),zver(i)) (7) where 1 ≤ m ≤ B and B = 3. Each attempt writes a prompt p(simi) , receives vˆ(i) ∼ W(· | o,p(simi) ), and predicts zver(i) ∈ {accept,reject,uncertain}. Let a denote accept. The number of attempts is determined by

m=τstop=min{i ≤ B:zver(i) = a ∨ i = B}. (8)

The full probability of this trajectory follows the opening product form, with one policy factor for each control action and one world-model factor for each queried rollout.

The action space exposes five control decisions:

- • Simulation decision. dsim ∈ {yes,no} decides whether concrete reasoning from a rollout is likely to be useful.
- • Simulation query. p(simi) specifies the taskrelevant objects, paths, contacts, or event changes for attempt i.
- • Rollout verification. zver(i) judges whether the rollout is consistent, plausible, and relevant to the question.
- • Rollout reliance. zrel states how an accepted, uncertain, or rejected rollout should be used, discounted, or overridden.

III. Rollout Verification & Reliance

II. Future Rollout Generation

[Figure 2]

Input Observation + Question

I. Simulation Decision

Prompt: 𝑝

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Rollout Verification

Is abstract reasoning sufficient?

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

World Model 𝑊

Accept

[Figure 16]

[Figure 17]

Yes

Rollout

Max retries reached

[Figure 18]

[Figure 19]

Reliance Rollout: 𝑣

[Figure 20]

Simulation Prompt Generation

[Figure 21]

Retry

[Figure 22]

Abstract Reasoning

Output Final Answer: 𝑦

Figure 3: Inference-time controlled concrete reasoning. The student MLLM first decides whether abstract reasoning is sufficient. If simulation is needed, it writes a simulation prompt psim, queries the world model W for a candidate rollout vˆ, verifies the rollout, and decides how much to rely on it alongside abstract reasoning. Rejected rollouts trigger prompt retry up to three simulation attempts; once this per-example cap is reached, the policy proceeds with its rollout-reliance state before predicting y.

#### • Answer prediction. y ∈ {A,B,C,D} is the final choice.

completed as τa. The privileged evaluator E+ then receives (ct,hst,a,τa,y∗,v∗) and assigns a teacherside score. In our implementation, E+ is instantiated by Qwen3.6-27B, a native multimodal model that observes the generated rollout and the groundtruth future video only during training. Concretely, E+ uses y∗ to judge final-answer correctness and uses v∗ to judge whether accepted rollouts are consistent with the actual future and useful for the question:

This trajectory makes the failure points explicit: the model can underuse useful simulation, accept misleading rollouts, or over-rely on weak rollouts.

#### 4.2 Two-Stage Training

PF-OPSD separates format learning from utility calibration. Stage 1 teaches the student to produce valid concrete-reasoning trajectories. Stage 2 calibrates these trajectories with a teacher-side evaluator that sees the privileged future only during training.

R+(τa) = 1[ˆy = y∗] − λsimNsim (9)

− λFA1[false accept]−λFR1[false reject]. These labels are evaluator-derived training signals, not human oracle annotations; their operational definitions are provided in Appendix Table 4. The evaluator also provides a teacher-view preference Pt+(a) = π+(a | c+t ,hst) under c+t = ct∪{y∗,v∗}. We set Q+t (a) = R+(τa) and compute A+t (a) = Q+t (a) − Vt+, where

- Stage 1: protocol SFT. We initialize the student with protocol supervision from a Gemini-3.1-Pro

+ Agent workflow (Pichai et al., 2025). For each training example, the workflow observes the image or anchor frame, question, options, ground-truth answer, and training-time future video as teacher context, and generates a structured trajectory over dsim, psim, zver, zrel, and y. We keep only trajectories that pass rule checks, answer-consistency checks, and reviewer-style filtering. This stage fixes the output protocol; the workflow is not used during PF-OPSD self-distillation or test-time inference.

- Stage 2: privileged-future on-policy selfdistillation. The second stage calibrates decisions produced by the current student. For each training example, the student first generates a

Pt+(a)Q+t (a), t ∈ Cdisc, K−1 Kk=1 Q+t (ak), t ∈ Ctext.

Vt+= a∈Ct

(10)

This on-policy design asks which alternatives would improve the student’s own behavior, rather than imitating a fixed privileged trace. Appendix B gives the procedural form.

#### 4.3 Advantage-Weighted Distillation

Objective The privileged advantages define student-view targets. For a discrete node t, we form

base trajectory τs under the student-view context c = (o,q,options), which excludes the groundtruth future v∗ and answer y∗. At each decision node t with prefix hst, we build a candidate set Ct. Discrete nodes use the valid action set, including dsim, zver(i), and y. Text nodes, including p(simi) and zrel, use K samples from the current policy.

qt⋆(a)=Zt−1π+(a | c+t ,hst)exp(A+t (a)/τA),(11) Zt=

π+(a′ | c+t ,hst)exp(A+t (a′)/τA).

a′∈Ct

The student minimizes

DKL sg[qt⋆(·)]∥πθ(· | ct,hst) , (12)

Ldisc=

Each candidate action a ∈ Ct is forced at node t, after which the remaining trajectory is greedily

t∈Cdisc

where ct excludes y∗ and v∗. For text nodes, we convert the privileged advantages into normalized candidate weights and optimize a weighted loglikelihood:

exp(A+t (ak)/τA) jexp(A+t (aj)/τA)

, (13)

wt,k=

K

sg[wt,k]log πθ(ak |ct,hst). (14)

Ltext=−

t∈Ctext

k=1

Combining the discrete-node KL objective and the text-node weighted likelihood gives the advantagedistillation loss:

LadvPF-OPSD = Ldisc + Ltext. (15)

The full training objective further includes protocol SFT and a simulation-call penalty:

L = LSFT + LadvPF-OPSD + λcallE[Nsim]. (16)

Unlike outcome-level RL such as GRPO (Shao et al., 2024), PF-OPSD assigns credit to intermediate concrete-reasoning decisions, such as whether to simulate, whether to reject a hallucinated rollout, and whether to rely on abstract reasoning.

#### 4.4 Inference

At test time, both v∗ and the protocol-generation workflow are removed. Deployment follows a learned simulation-control policy with only the student-view state si = (x,A1:i−1) and a hard per-example retry cap:

log pθ(τ | x;W),B = 3. (17)

τˆ= arg max

τ: Nsim(τ)≤B

The first action gates simulation:

πθ(d | x). (18)

dsim= arg max

d∈{yes,no}

If dsim = no, the policy answers from abstract reasoning. Otherwise, each attempt samples or

decodes p(simi) , queries W for vˆ(i), and predicts zver(i). The retry gate is

gi= 1[zver(i) ̸= accept]1[i < B], (19)

so rejected or uncertain rollouts trigger another prompt only before the per-example cap is reached. After stopping at m, the model forms a reliance state

πθ(y|rm)(20)

rm=ϕθ(x,A1:m,zrel),yˆ=arg max

y

This formulation makes inference a closed-loop policy over simulation selection, rollout verification, retry, rollout reliance, and answering. The world model remains a fallible concrete-reasoning source: rejected rollouts are not discarded mechanically, but can be discounted in zrel when no accepted rollout is available.

### 5 Experiments

#### 5.1 Experimental Setup

We evaluate PF-OPSD on the two future-prediction benchmarks introduced in Section 3: VRQABench and OpenWorldQA. Unless otherwise stated, the student is Qwen3.5-9B (Qwen Team, 2026a); the external world model is Helios (Yuan et al., 2026), using the VR-Bench-fine-tuned Helios model for VRQABench and the general Helios model for OpenWorldQA; and protocol trajectories are generated offline by a privileged Gemini-3.1-Pro + Agent teacher (Pichai et al., 2025). We train with Stage-1 protocol SFT followed by Stage-2 PFOPSD self-distillation, and evaluate against zeroshot/no-simulation MLLMs, Qwen-based training baselines, and our workflow-agent prompting baseline using accuracy, simulation-decision quality, and rollout-verification metrics. Detailed splits, backbone/world-model choices, optimization hyperparameters, inference protocol, baseline definitions, and metric implementations are provided in Appendix A; dataset category taxonomies and distributions are in Appendix D.

#### 5.2 Main Results

Table 2 summarizes the benchmark results. PFOPSD achieves the best accuracy on both benchmarks (72.4% on VRQABench and 70.5% on OpenWorldQA), improving over SFT and SFT + GRPO. The SFT baseline is already a learnedcontroller baseline because it is trained on structured trajectories containing simulation decisions, simulation prompts, rollout verification, rollout reliance, and final answers. The prompted Workflow Agent baseline, which has prompt-level access to Helios but no PF-OPSD training, performs worse than the image-only supervised model, indicating that world-model access alone is insufficient without learned simulation selection and rollout verification. The gains are consistent across the two benchmark designs rather than being driven by a single dataset: relative to SFT, PF-OPSD improves by 10.6 points on VRQABench and 10.9 points on

Model / Method VRQABench OpenWorldQA Zero-shot / no-simulation baselines

Gemini-3-Flash (Pichai et al., 2025) 45.9% 48.2% GPT-5.4 (OpenAI, 2026) 43.2% 53.4% HY3 (Tencent HY, 2026) 38.2% 35.0% Qwen3.6-27B (Qwen Team, 2026b) 33.0% 41.4% Qwen3.5-9B (Qwen Team, 2026a) 33.2% 39.8%

- Qwen2.5-VL-7B (Bai et al., 2025) 32.7% 14.2%

Workflow-agent baselines

Workflow Agent (Qwen3.5-9B) 32.6% 38.6% Workflow Agent (Gemini-3-Flash) 47.2% 49.5%

- Qwen3.5-9B training baselines SFT (Qwen Team, 2026a) 61.8% 59.6% GRPO (Shao et al., 2024) 63.5% 61.2% PF-OPSD 72.4% 70.5%

- Table 2: Overall accuracy on the two benchmarks. Full percategory results are reported in Table 8 and Table 9 in the appendix.

Variant VRQA OWQA Call Rate Calls

PF-OPSD 72.4% 70.5% 42.5% 0.45 w/o Simulation decision 68.5% 67.2% 100% 1.00 w/o Rollout verification 65.2% 64.8% 42.5% 0.45 w/o Rollout reliance 67.8% 66.5% 42.5% 0.45 w/o Advantage weighting 66.4% 65.2% 45.2% 0.48 answer-only distillation 64.5% 63.8% 35.4% 0.38 SFT 61.8% 59.6% 23.0% 0.23

- Table 3: Ablation results for training mechanisms and concrete-reasoning nodes. Call Rate denotes the percentage of examples that trigger world-model simulation, and Calls denotes the average number of simulation calls per example.

OpenWorldQA. This suggests that the learned controller is not merely increasing world-model usage, but learning when rollouts are likely to be useful and when they should be rejected or discounted.

#### 5.3 Ablation Studies

Table 3 isolates the contribution of each PF-OPSD component. SFT removes Stage-2 self-distillation but keeps the same structured reasoning-chain supervision, so it serves as a learned-controller cold start. Forcing simulation for all samples exposes the model to unnecessary noisy rollouts. Removing rollout verification or reliance weakens the model’s ability to assess and balance concrete evidence, and removing advantage weighting reduces training to uncalibrated teacher-view imitation. Answer-only distillation further shows that supervising the final answer alone is insufficient. Overall, the gains come from utility-calibrated supervision over intermediate concrete-reasoning actions, especially rollout verification and advantage weighting, rather than from simply adding generated videos.

#### 5.4 Cross-World-Model Diagnostic

We test cross-world-model transfer by replacing the default Helios generator with Wan 2.2 at inference time while keeping the student policy and

(a) VRQABench

(b) OpenWorldQA

75

75

Helios Wan 2.2

73.6

| |
|---|

72.4

71.8

Accuracy(%)

70.5

70.0

70

70

68.7

68.5

67.2

65

65

Always Sim.

PF-OPSD

Always Sim.

PF-OPSD

- Figure 4: Cross-world-model diagnostic. Replacing Helios with Wan 2.2 improves both the always-simulate variant and PF-OPSD, while PF-OPSD remains stronger than always simulating with the stronger rollout generator.

VRQA overall

OWQA overall

OWQA hard

0

20

40

60

80

100

Callrate(%)

45.2

38.4

65.5

(a) Selective Calls

Always

VRQA overall

OWQA overall

OWQA hard

20

40

60

80

100

Score(%)

80.4

77.3

87.3

42.5

48.2

28.5

(b) Decision Quality

| |
|---|

Call F1 Reduced

| |
|---|

- Figure 5: Simulation decision quality. PF-OPSD keeps the overall call rate well below always-simulate while increasing calls on OWQA hard samples and maintaining high agreement with privileged evaluator-derived simulation-help labels. “Reduced” denotes the fraction of unnecessary simulation calls avoided relative to an always-simulate policy, not the complement of Call Rate.

prompts unchanged. Figure 4 compares PF-OPSD with an Always Simulate (No Gate) variant, which preserves the trained policy but forces one worldmodel query for every example. Wan 2.2 gives slightly higher accuracy but is substantially slower, so we use it only as a diagnostic rollout generator. Its gains confirm that stronger world models can provide better concrete evidence, but the alwayssimulate variant with Wan 2.2 still trails PF-OPSD with Helios, showing that world-model quality does not replace learned simulation control.

#### 5.5 Simulation Decision Analysis

- Figure 5 evaluates simulation selection using call rate, agreement with privileged evaluator-derived “simulation-helps” labels, and avoided calls relative to always simulating; detailed subset results are provided in Appendix Table 5. PF-OPSD invokes the world model selectively, calling more often for dynamically or spatially demanding samples and avoiding calls when static cues are sufficient.
- Figure 6 further breaks down the realized number of simulation attempts. The aggregate behavior matches Table 3: 42.5% of examples invoke simulation, with 0.45 calls per example on average. Multi-attempt cases correspond to harder examples where earlier rollouts are rejected or uncertain, resulting in lower accuracy and higher misjudgment.

(a) Realized Depth

(b) Performance

12

Call rate

57.5

60

80

50

Accuracy(%)

Misjudge(%)

40.5

75

Share(%)

8

40

30

70

4

20

65

VRQA Acc.

10

OWQA Acc.

60

Misjudge

1.5 0.5

0

0

0 1 2 3

0 1 2 3

Attempts

Attempts

- Figure 6: Realized simulation depth and performance under PF-OPSD. Attempts denotes the actual number of worldmodel calls selected by the policy for an example, not an externally imposed global constraint. Rows with one to three attempts in panel (a) sum to the learned call rate; panel (b) shows that repeated retries correspond to harder examples with lower accuracy and higher misjudgment.

| |on|ly|
|---|---|---|
| | | |
| | | |

90

100

92.5

Rollout Verification Diagnostics

| |
|---|

Accept rate False accept

| |
|---|

False reject

High quality

Low quality

Corrupted Image conflict

Missing interaction

Physical inconsistency

0

10

20

18.4

5.2

12.5

15.8

14.2

4.6

14.8

7.5

11.6

13.5

10.8

7.5

12.4

4.2

8.5

10.2

9.4

Rate(%)

- Figure 7: Rollout-verification behavior under controlled rollout-quality conditions. Accept rate measures how often PF-OPSD trusts a generated rollout, while false accept and false reject measure erroneous acceptance of misleading rollouts and erroneous rejection of useful rollouts, respectively. The broken y-axis is used to improve the visibility of low-frequency errors, with exact values annotated on each bar.

#### 5.6 Rollout Verification Analysis

Figure 7 summarizes whether PF-OPSD decides to trust generated rollouts under controlled rolloutquality conditions. The model accepts high-quality rollouts at a high rate, while sharply reducing acceptance for corrupted, conflicting, incomplete, or physically inconsistent rollouts. The full diagnostic table, including verification precision, recall, false acceptance, false rejection, and final accuracy, is provided in Appendix Table 6. Figure 8 compresses these controlled conditions into a rolloutquality severity diagnostic. As rollout quality degrades from verified useful futures to incomplete, wrong-but-plausible, and strongly corrupted rollouts, PF-OPSD reduces its acceptance rate from 92.5% to 5.2%. Final accuracy degrades gracefully rather than collapsing, suggesting that the policy can discount unreliable concrete evidence and fall back on abstract reasoning.

#### 5.7 Rollout Reliance and Conflict Analysis

Figure 9 summarizes representative diagnostic subsets in which abstract reasoning and rollouts are conflicting or complementary. It visualizes whether

(a) Verification

(b) Final Accuracy

100

| |Accept<br><br>False accept<br><br>False reject| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

88.5

90

75

Accuracy(%)

80

Rate(%)

50

70

65.2 64.5

25

62.8

60

0

S0 Useful

S1 Low-q

S2 Wrong

S3 Corrupt

S0 Useful

S1 Low-q

S2 Wrong

S3 Corrupt

- Figure 8: Rollout-quality severity diagnostic. PF-OPSD sharply reduces rollout acceptance as generated futures become less useful or more conflicting, while maintaining noncollapsed final accuracy under misleading rollouts.

|82.5|[Figure 23]<br><br>8.4|9.1|
|---|---|---|
|12.5|78.4|9.1|
|8.5|85.2|6.3|
|25.4|5.2|69.4|

Follow Roll.

Reject Roll. Fuse

Abstract wrong rollout correct

Abstract correct rollout misleading

Rollout hallucination

Static + rollout complementary

[Figure 24]

0

30

60

90

Share(%)

- Figure 9: Representative conflict-resolution behavior. PFOPSD follows a rollout when it corrects abstract reasoning, rejects misleading or hallucinated rollouts, and combines rollouts with static cues when the two sources are complementary.

the model follows the rollout, rejects it, or combines it with static cues; the full subset-level table, including final accuracy and recovery, is provided in Appendix Table 7. The goal of PF-OPSD is not to always trust simulation. Rather, the model should use rollouts when they provide corrective concrete reasoning, reject them when they are misleading, and combine them with static cues when the two sources are complementary. The results show that the model recovers many errors when abstract reasoning is wrong but the rollout is correct, and that it often rejects hallucinated or misleading rollouts when static cues are stronger.

### 6 Conclusion

We studied multimodal future outcome prediction from current images or pre-event anchor frames and framed world-model assistance as controlled concrete reasoning rather than naive world-model attachment. To evaluate this setting, we introduced human-verified VRQABench and OpenWorldQA, covering controllable spatial planning and open-domain physical prediction. We further proposed PF-OPSD, which uses training-time future videos and answers as privileged teacherside context to calibrate simulation, rollout verification, rollout reliance, and answer decisions. Experiments show improved accuracy and stronger robustness to noisy or conflicting rollouts.

### Limitations

This work focuses on image-conditioned future prediction with optional concrete reasoning from generative world models. Its conclusions are most directly applicable to settings where the world model can produce futures that are at least partially relevant to the given scene and question. When generated rollouts are poorly aligned with the input, PF-OPSD is intended to reduce unnecessary reliance on them rather than to replace improvements in world-model quality.

Our experiments cover two complementary benchmarks and a concrete world-model interface, but they do not exhaust all forms of physical reasoning or all possible world-model designs. More specialized domains, longer temporal horizons, and interactive environments may require additional benchmark coverage and task-specific prompting strategies.

Finally, PF-OPSD uses privileged future videos only during training to estimate action utility. This keeps the deployed policy free from ground-truth future inputs, while making the approach depend on the availability and alignment of training-time privileged signals. Extending the framework to weaker or more implicit forms of future supervision is a promising direction for future work.

### References

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael G. Rabbat, Yann LeCun, and Nicolas Ballas. 2023. Selfsupervised learning from images with a jointembedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 15619–15629. IEEE.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. arXiv Preprint, abs/2511.21631.

Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross B. Girshick. 2019. PHYRE: A new benchmark for physical reasoning. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 5083– 5094.

Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, Nicolas Pinto, and Joseph P. Turian. 2020. Experience grounds language. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 8718–8735. Association for Computational Linguistics.

Yilin Cao, Yufeng Zhong, Zhixiong Zeng, Liming Zheng, Jing Huang, Haibo Qiu, Peng Shi, Wenji Mao, and Wan Guanglu. 2026. Mobiledreamer: Generative sketch world model for gui agent. arXiv preprint arXiv:2601.04035.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Trans. Mach. Learn. Res., 2023.

Katrina Drozdov, Ravid Shwartz-Ziv, and Yann LeCun. 2024. Video representation learning with jointembedding predictive architectures. arXiv Preprint, abs/2412.10925.

Gaël Gendron, Qiming Bao, Michael Witbrock, and Gillian Dobbie. 2024. Large language models are not strong abstract reasoners. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pages 6270–6278. ijcai.org.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

- Drew A. Hudson and Christopher D. Manning. 2018. Compositional attention networks for machine reasoning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.
- Drew A. Hudson and Christopher D. Manning. 2019. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 6700–6709. Computer Vision Foundation / IEEE.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Fei-Fei Li, C. Lawrence Zitnick, and Ross B. Girshick. 2025. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning (version 1). https://doi.org/10.5281/ zenodo.14936905.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Dezhao Luo, Bohan Tang, Kang Li, Georgios Papoudakis, Jifei Song, Shaogang Gong, Jianye Hao, Jun Wang, and Kun Shao. 2025. Vimo: A generative visual gui world model for app agents. arXiv preprint arXiv:2504.13936.

OpenAI. 2026. Introducing gpt-5.4. https://openai.com/zh-Hans-CN/index/ introducing-gpt-5-4/. Accessed: 2026-05-25.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, and 7 others. 2024. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. 2018. Film: Visual reasoning with a general conditioning layer. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 3942– 3951. AAAI Press.

Sundar Pichai, Demis Hassabis, and Koray Kavukcuoglu. 2025. A new era of intelligence with gemini 3.

Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Dilek Hakkani-Tür, Gokhan Tur, Yunzhu Li, and 1 others. 2026. Current agents fail to leverage world model as tool for foresight. arXiv preprint arXiv:2601.03905.

- Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.
- Qwen Team. 2026b. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. 2018. Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv Preprint, abs/1803.07616.

Adam Santoro, David Raposo, David G. T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter W. Battaglia, and Tim Lillicrap. 2017. A simple neural network module for relational reasoning. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 4967–4976.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves

to use tools. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Wei Tao, Yucheng Zhou, Yanlin Wang, Wenqiang Zhang, Hongyu Zhang, and Yu Cheng. 2024. Magis: Llm-based multi-agent framework for github issue resolution. Advances in Neural Information Processing Systems, 37:51963–51993.

Tencent Hunyuan Foundation Model Team. 2025. Hunyuanvideo 1.5 technical report. arXiv Preprint, abs/2511.18870.

Tencent HY. 2026. Hy3 preview. https://hy. tencent.com/research/hy3. Accessed: 2026-0525.

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. 2022. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pretraining. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Vladimir Vapnik and Akshay Vashist. 2009. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544–557.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, and 43 others. 2025. Wan: Open and advanced largescale video generative models. arXiv preprint arXiv:2503.20314.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35:

Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. 2025. Video models are zero-shot learners and reasoners. arXiv Preprint, abs/2509.20328.

Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Chen, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, and Chenglin Wu. 2025. Reasoning via video: The first evaluation of video models’ reasoning abilities through mazesolving tasks. arXiv Preprint, abs/2511.15065.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2024. A survey on multimodal large language models. National Science Review, 11(12):nwae403.

Heeji Yoon, Jaewoo Jung, Junwan Kim, Hyungyu Choi, Heeseong Shin, Sangbeom Lim, Honggyu An, Chaehyun Kim, Jisang Han, Donghyun Kim, Chanho Eom, Sunghwan Hong, and Seungryong Kim. 2025. Visual representation alignment for multimodal large language models. arXiv Preprint, abs/2509.07979.

Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. 2026. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379.

Jingtong Yue, Ziqi Huang, Zhaoxi Chen, Xintao Wang, Pengfei Wan, and Ziwei Liu. 2025. Simulating the visual world with artificial intelligence: A roadmap. arXiv Preprint, abs/2511.08585.

Xiaoyuan Zhang, Chengdong Ma, Yizhe Huang, Weidong Huang, Siyuan Qi, Song-Chun Zhu, Xue Feng, and Yaodong Yang. 2025. World models should prioritize the unification of physical and social dynamics. arXiv Preprint, abs/2510.21219.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv preprint arXiv:2601.18734.

Yucheng Zhou, Xiubo Geng, Tao Shen, Chongyang Tao, Guodong Long, Jian-Guang Lou, and Jianbing Shen. 2023. Thread of thought unraveling chaotic contexts. arXiv preprint arXiv:2311.08734.

Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. 2024. Visual in-context learning for large

vision-language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 15890–15902.

### A Detailed Experimental Setup

Datasets and splits. All training and evaluation use the splits introduced in Section 3: VRQABench provides 4,000 training and 636 evaluation puzzles over 5 spatial categories, and OpenWorldQA provides 3,904 training and 500 evaluation anchorframe questions over 12 physical categories. Both datasets are filtered by automatic reviewers and then manually verified item by item; retained examples must have a valid initial observation, a unique answer, plausible distractors, and no visible leakage of the future outcome. Stage-1 protocol SFT and Stage-2 PF-OPSD self-distillation are run on the union of the two training splits; all reported numbers are computed on the held-out evaluation splits, which are never seen during training. Detailed category taxonomies and per-split distributions are given in Appendix D.

Backbone, world model, and protocol teacher. Unless otherwise stated, the student MLLM is Qwen3.5-9B (Qwen Team, 2026a); the external world model is Helios (Yuan et al., 2026). We use the VR-Bench-fine-tuned Helios model for VRQABench, and the general Helios model for OpenWorldQA; in both cases, Helios is queried only when the policy emits dsim = 1. Stage-1 protocol trajectories are produced by a Gemini-3.1-Pro + Agent workflow (Pichai et al., 2025) that sees the privileged future video as teacher-side context; this workflow is used only for offline data generation and is not invoked during PF-OPSD training or at test time.

Training. Following the compact post-training configuration reported in OPSD (Zhao et al., 2026), we tune only LoRA adapters on the student MLLM, with rank r = 64, scaling factor α = 128, and target modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj; the visual encoder and external world model are frozen. Both stages use AdamW with learning rate 5 × 10−6, effective batch size 32, bfloat16 precision, gradient checkpointing, and FlashAttention-2. Stage-1 SFT optimizes LSFT on filtered protocol trajectories for one epoch, with maximum sequence length 16k. Stage-2 PF-OPSD runs for 100 on-policy update steps and optimizes L = LSFT + LadvPF-OPSD + λcallE[Nsim]. Unless otherwise stated, we use (λsim,λFA,λFR) = (0.05,0.50,0.25), λcall = 0.02, advantage temperature τA = 0.5, and K = 4 candidate samples at text nodes; text candidates are

sampled with temperature 1.1 and forced rollouts are greedily completed. The same optimizer and adapter settings are used for all trainable Qwenbased baselines, changing only the training objective.

Inference. At test time the student receives only the image (or anchor frame), question, and options; the privileged future v∗, answer y∗, and protocol teacher are removed. The policy decodes a structured trajectory over (dsim,psim,zver,zrel,y) with a hard per-example simulation cap B = 3, following the inference objective in Section 4. Decoding is deterministic (greedy) for discrete control nodes and uses a low temperature for text nodes.

Baselines. We compare against three families covering the natural alternatives to controlled integration: (i) Zero-shot / no-simulation MLLMs, including closed-source reference models such as Gemini-3-Flash, GPT-5.4, and HY3, as well as Qwen-series MLLMs that answer directly from the image and question; (ii) Qwen-based training baselines, including SFT and SFT+GRPO (Shao et al., 2024) on the same backbone and data; and (iii) Workflow-agent baselines, where either Qwen3.5-9B or Gemini-3-Flash is given promptlevel access to Helios rollouts but receives no PFOPSD training. SFT is trained on the same structured trajectory format as PF-OPSD, but without on-policy privileged utility calibration.

Metrics. We report (1) overall and per-category multiple-choice accuracy on both benchmarks; (2) average simulation calls per sample Nsim and call rate; (3) decision quality via call precision/recall/F1 against privileged evaluator-derived “simulationhelps” labels; and (4) rollout verification precision/recall together with false-accept and falsereject rates under controlled rollout-quality conditions. Unless noted, all numbers are computed on the fixed evaluation splits.

Privileged evaluator-derived labels. Table 4 summarizes the operational labels produced by the privileged evaluator. These labels are derived by comparing the generated rollout with the groundtruth future using Qwen3.6-27B; they are not human oracle annotations.

A.1 OpenWorldQA Macro and Resource-Grouped Results

Because the OpenWorldQA training split is intentionally collected from natural videos and is not

Label Operational definition

Useful rollout The evaluator judges that the generated rollout is consistent with v∗ on the taskrelevant future outcome and informative for answering q.

False accept The student accepts a rollout that the evaluator marks as inconsistent with v∗ or not useful for answering q.

False reject The student rejects a rollout that the evaluator marks as consistent with v∗ and useful for answering q.

Simulation helps

A diagnostic label indicating that rollout evidence can correct or materially support the final answer.

- Table 4: Operational definitions of privileged evaluatorderived labels.

category-balanced, we additionally report macroaveraged and resource-grouped results. Macro averaging gives equal weight to the 12 physical categories. We also group categories by the number of training examples: low-resource categories have fewer than 100 training examples (C1, C3, C5, C7, C10, C12), medium-resource categories have 100– 800 examples (C6, C8, C11), and high-resource categories have more than 800 examples (C2, C4, C9). The test split remains approximately balanced, with 41–42 questions per category.

PF-OPSD improves over SFT in all resource groups, including categories with fewer than 100 training examples. This suggests that the gains are not only a consequence of repeated motifs in the largest training categories. The high-resource group remains the hardest because it contains spatial relation, support stability, and tool-use questions with visually subtle outcome differences; nevertheless, PF-OPSD gives the largest absolute gain there, indicating that simulation control is especially helpful when static cues and generated rollouts are easy to confuse.

#### A.2 Retry Depth Diagnostic

Table 11 reports the same realized simulation-depth statistics as Figure 6. The number of attempts is endogenous: examples with two or three attempts are those for which earlier rollouts are rejected or uncertain. Their lower accuracy should therefore be interpreted primarily as evidence that the policy routes harder examples to deeper retry paths, not as evidence that retry itself causes errors.

### B PF-OPSD Algorithm

Algorithm 1 summarizes the training procedure used to construct the advantage-weighted targets in the main text.

Algorithm 1 Privileged-Future On-Policy Self-Distillation (PF-OPSD)

Require: Training sample (x, y∗, v∗) with x = (o, q, O), world model W, student policy πθ, privileged evaluator E+, simulation cap B = 3 Ensure: Updated student policy πθ

- 1: for each training step do
- 2: Generate a student-view trajectory τs ∼ pθ(· | x; W) with v∗ hidden
- 3: for each decision node t visited in τs do
- 4: if t ∈ Cdisc then
- 5: Set Ct to the valid discrete action set
- 6: else
- 7: Sample Ct = {a1, . . . , aK} from πθ(· | ct, hst)
- 8: end if
- 9: for each candidate a ∈ Ct do
- 10: Force a at node t and greedily complete τa under the same student-view rollout protocol
- 11: Score R+(τa; y∗, v∗) with E+ using teacherside context (y∗, v∗)
- 12: Set Q+t (a) = R+(τa; y∗, v∗)
- 13: end for
- 14: Compute Vt+ and A+t (a) = Q+t (a) − Vt+
- 15: Construct discrete targets qt⋆ or text weights wt,k
- 16: end for
- 17: Update πθ with Ldisc + Ltext
- 18: end for

### C Related Work

C.1 Multimodal and Tool-Augmented Reasoning

Large language models have shown strong language-based reasoning abilities through structured prompting and intermediate inference, including chain-of-thought prompting, self-consistency, and zero-shot reasoning (Wei et al., 2022; Wang et al., 2023; Kojima et al., 2022; Zhou et al., 2023; Tao et al., 2024). Subsequent work extends this reasoning process with external computation and interaction: program-aided reasoning separates symbolic reasoning from executable calculation (Chen et al., 2023), while ReAct and Toolformer demonstrate that language models can interleave reasoning with tool invocation and observations (Yao et al., 2023; Schick et al., 2023). These lines of work establish the importance of external resources, but most tools return relatively explicit textual, symbolic, or numerical outputs.

Multimodal large language models further connect language reasoning with visual perception, enabling image-text dialogue, visual question answering, and complex multimodal instruction following

Dataset / Subset # Call Rate Precision Recall F1 Acc. Called Acc. Not Called Reduced

VRQABench overall 636 45.2% 82.5% 78.4% 80.4% 78.5% 67.2% 42.5% VRQABench maze 275 38.6% 84.2% 81.5% 82.8% 76.4% 65.8% 52.4% VRQABench irregular 171 45.2% 86.5% 85.2% 85.8% 75.2% 62.4% 48.5% VRQABench sokoban 190 55.4% 75.4% 68.5% 71.8% 85.6% 72.5% 35.2% OpenWorldQA overall 500 38.4% 79.6% 75.2% 77.3% 76.8% 66.5% 48.2% OWQA hard 84 65.5% 88.2% 86.4% 87.3% 72.4% 55.6% 28.5%

- Table 5: Detailed simulation decision quality. The model preferentially invokes simulation for dynamically or spatially demanding samples.

Rollout Condition # Accept Rate Precision Recall False Accept False Reject Acc.

High-quality rollout 300 92.5% 95.4% 92.5% 4.6% 7.5% 88.5% Low-quality rollout 150 18.4% 85.2% 22.5% 14.8% 12.4% 65.2% Corrupted rollout 100 5.2% 92.5% 8.4% 7.5% 4.2% 62.8% Conflicts with image cues 120 12.5% 88.4% 15.2% 11.6% 8.5% 64.5% Misses key interaction 140 15.8% 86.5% 18.4% 13.5% 10.2% 63.4% Physical inconsistency 130 14.2% 89.2% 16.5% 10.8% 9.4% 64.8%

- Table 6: Full rollout-verification diagnostics under controlled rollout-quality conditions. Accept Rate measures how often PF-OPSD trusts the generated rollout; Precision and Recall evaluate verification decisions against privileged evaluator-derived rollout usefulness labels; False Accept and False Reject measure erroneous acceptance of misleading rollouts and erroneous rejection of useful rollouts, respectively.

(Yin et al., 2024; Yoon et al., 2025; Bai et al., 2025; Zhou et al., 2024). Nevertheless, future-oriented reasoning remains difficult because the required future state may not be visible in the input image. The model must infer motion tendencies, spatial constraints, contact changes, and causal consequences from a static state. This exposes the gap between language-centric abstraction and grounded dynamic prediction, echoing broader observations that language-only or weakly grounded models can lack stable experiential grounding (Bisk et al., 2020; Gendron et al., 2024).

C.2 Visual Reasoning, Physical Prediction, and Benchmarks

Visual reasoning has long studied how models combine perception with compositional structure and physical regularities. Diagnostic benchmarks such as CLEVR and GQA evaluate compositional visual question answering (Johnson et al., 2025; Hudson and Manning, 2019), while physical reasoning benchmarks such as PHYRE and IntPhys focus on intuitive physics and outcome prediction (Bakhtin et al., 2019; Riochet et al., 2018). These benchmarks inspired architectures for relational and compositional reasoning, including relational networks, FiLM, and compositional attention (Santoro et al., 2017; Perez et al., 2018; Hudson and Manning, 2018). More recent self-supervised visual representations also capture useful structural regularities from large-scale image data (Oquab et al., 2024;

Assran et al., 2023).

Despite this progress, many existing visual reasoning tasks either emphasize static recognition or evaluate perception from observed videos. Our setting is different: the agent receives only a current image or pre-event anchor frame and must answer a question about a future outcome. This makes the task closer to foresight for embodied and interactive agents. VRQABench builds on VR-Bench (Yang et al., 2025) to test controllable spatial lookahead in maze and Sokoban-style environments, while OpenWorldQA complements it with open-domain physical prediction from real-world anchor frames. Together, they focus on whether a model can infer future states from static initial conditions rather than merely recognize completed events.

C.3 Generative World Models as Concrete-Reasoning Sources

Video representation learning and video generation have expanded visual reasoning from static perception to temporal dynamics. Predictive video representation methods and masked video modeling learn temporal regularities from video data (Tong et al., 2022; Drozdov et al., 2024), and recent studies suggest that video models can exhibit zero-shot physical and spatial reasoning through frame-byframe generation (Wiedemer et al., 2025). At the same time, large-scale generative video models are increasingly viewed as world models that can synthesize plausible future rollouts conditioned on an

Conflict Subset # Follow Roll. Reject Roll. Fuse Final Acc. Recovery

Abstract wrong, rollout correct 150 82.5% 8.4% 9.1% 85.2% 78.5% Abstract correct, rollout misleading 120 12.5% 78.4% 9.1% 81.5% 75.2% Both uncertain 100 45.2% 15.8% 39.0% 62.4% 55.8% Rollout hallucination 130 8.5% 85.2% 6.3% 76.8% 72.4% Static cues insufficient 160 75.4% 12.5% 12.1% 78.5% 74.2% Static cues and rollout complementary 140 25.4% 5.2% 69.4% 88.2% 85.6%

- Table 7: Full conflict analysis between abstract reasoning and rollouts. Follow Roll., Reject Roll., and Fuse measure how PF-OPSD arbitrates between abstract reasoning and rollout-based concrete reasoning; Final Acc. and Recovery report the resulting task accuracy and error-recovery rate.

Model / Method Overall C1 C2 C3 C4 C5 Zero-shot / no-simulation baselines

Gemini-3-Flash (Pichai et al., 2025) 45.9% 43.5% 41.2% 56.5% 46.3% 37.2% GPT-5.4 (OpenAI, 2026) 43.2% 48.4% 46.1% 38.8% 31.6% 51.2% HY3 (Tencent HY, 2026) 38.2% 38.7% 47.9% 29.3% 30.5% 46.5% Qwen3.6-27B (Qwen Team, 2026b) 33.0% 34.4% 31.5% 36.7% 27.4% 32.6% Qwen3.5-9B (Qwen Team, 2026a) 33.2% 41.4% 44.2% 25.9% 11.6% 27.9%

- Qwen2.5-VL-7B (Bai et al., 2025) 32.7% 29.0% 44.8% 21.1% 34.7% 37.2%

Workflow-agent baselines

Workflow Agent (Qwen3.5-9B) 32.6% 40.8% 43.6% 25.2% 11.0% 27.3% Workflow Agent (Gemini-3-Flash) 47.2% 44.8% 42.5% 57.8% 47.6% 38.5%

- Qwen3.5-9B training baselines SFT (Qwen Team, 2026a) 61.8% 59.7% 58.2% 70.1% 66.3% 46.5% GRPO (Shao et al., 2024) 63.5% 61.2% 60.5% 71.8% 67.4% 48.2% PF-OPSD 72.4% 70.5% 69.8% 78.4% 76.2% 61.5%

- Table 8: Detailed VRQABench evaluation results. Category abbreviations correspond to turn count, turn direction, Sokoban push, direction count, and push-direction count.

initial image and text instruction (Wan et al., 2025; Team, 2025; Yue et al., 2025; Zhang et al., 2025).

Using such models as concrete-reasoning sources is attractive for future prediction because generated rollouts may expose object motion, path evolution, contact changes, and temporal consequences that are absent from the static input. Recent agent work has explored world models in structured digital environments, such as GUI transition simulation for app agents (Luo et al., 2025; Cao et al., 2026). However, world models are not precise oracles. They may hallucinate visually plausible but task-incorrect futures, alter important geometry, ignore small causal factors, or generate rollouts with uncertain task utility. Recent evidence further suggests that current agents often fail to leverage world models effectively for foresight (Qian et al., 2026). Therefore, world-model assistance should be treated as noisy concrete reasoning rather than unconditional truth.

#### C.4 Learning to Control Concrete Reasoning

The above observations motivate a shift from simple world-model attachment to controlled concrete reasoning. A generative world model differs

from conventional tools because it returns highdimensional, continuous, and potentially misleading visual rollouts. An MLLM must therefore decide whether simulation is necessary, formulate an appropriate query, verify whether the generated rollout is credible, and resolve conflicts between abstract reasoning and rollout-based concrete reasoning. This makes world-model-assisted reasoning a problem of simulation selection, rollout verification, and rollout reliance rather than only tool invocation.

Our PF-OPSD framework is related to knowledge distillation, learning with privileged information, and on-policy learning (Hinton et al., 2015; Vapnik and Vashist, 2009; Schulman et al., 2017). Standard distillation transfers teacher supervision into a deployable student, while privilegedinformation settings use training-only signals that are unavailable at test time. PF-OPSD should be viewed as an instantiation of these ideas for worldmodel-assisted future QA rather than as a new general-purpose distillation principle. The specific contribution is to make the privileged signal act on simulation-control nodes: whether to call a world model, whether to accept or reject a rollout, how to

Model / Method Overall C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 Zero-shot / no-simulation baselines

Gemini-3-Flash (Pichai et al., 2025) 48.2 61.0 35.7 61.0 35.7 65.9 40.5 71.4 50.0 21.4 53.7 40.5 42.9 GPT-5.4 (OpenAI, 2026) 53.4 58.5 19.0 65.9 50.0 68.3 57.1 71.4 47.6 28.6 65.9 54.8 54.8 HY3 (Tencent HY, 2026) 35.0 39.0 16.7 48.8 33.3 43.9 31.0 35.7 42.9 28.6 48.8 33.3 19.0 Qwen3.6-27B (Qwen Team, 2026b) 41.4 68.3 0.0 65.9 21.4 53.7 47.6 59.5 38.1 0.0 34.1 45.2 64.3 Qwen3.5-9B (Qwen Team, 2026a) 39.8 68.3 0.0 68.3 2.4 36.6 54.8 57.1 35.7 0.0 31.7 52.4 71.4 Qwen2.5-VL-7B (Bai et al., 2025) 14.2 12.2 0.0 36.6 2.4 48.8 2.4 23.8 11.9 0.0 19.5 4.8 9.5

Workflow-agent baselines

Workflow Agent (Qwen3.5-9B) 38.6 66.8 0.0 66.5 1.8 35.2 52.4 55.8 34.2 0.0 30.5 50.8 69.8 Workflow Agent (Gemini-3-Flash) 49.5 62.3 37.0 62.3 37.0 67.2 41.8 72.7 51.3 22.7 55.0 41.8 44.2

Qwen3.5-9B training baselines

SFT (Qwen Team, 2026a) 59.6 75.6 31.0 78.0 40.5 61.0 69.0 71.4 61.9 19.0 68.3 57.1 83.3 GRPO (Shao et al., 2024) 61.2 76.8 33.5 79.2 42.8 62.5 70.4 72.8 63.5 21.4 69.5 58.8 84.5 PF-OPSD 70.5 81.7 47.8 84.9 55.7 71.8 77.9 80.5 72.8 37.9 78.7 67.8 89.5

Table 9: Detailed OpenWorldQA test results. Values are percentages; category names follow the dataset taxonomy.

Model / Method Macro Avg. Low-resource Medium-resource High-resource

SFT (Qwen Team, 2026a) 59.7 72.9 62.7 30.2 GRPO (Shao et al., 2024) 61.3 74.2 64.2 32.6 PF-OPSD 70.6 81.2 72.8 47.1

PF-OPSD − SFT +10.9 +8.3 +10.1 +16.9

- Table 10: Macro-averaged and resource-grouped OpenWorldQA results. Values are percentages. Resource groups are defined by the number of training examples per category, while evaluation remains balanced across categories.

Attempts Share VRQA Acc. OWQA Acc. Misjudge

- 0 57.5% 67.2% 66.5% –
- 1 40.5% 80.1% 76.4% 4.0%
- 2 1.5% 70.8% 68.2% 7.5%
- 3 0.5% 62.4% 60.5% 11.0%

- Table 11: Realized retry-depth diagnostic under PF-OPSD. Attempt depth is selected by the policy and is therefore correlated with sample difficulty.

rely on it, and which final answer to produce. In our case, ground-truth future videos provide privileged context for calibrating whether rollouts and intermediate concrete-reasoning decisions are useful. Unlike offline imitation of fixed demonstrations, the policy first samples its own trajectories under test-time inputs, and the privileged future is used only to produce soft targets for decision nodes visited by the current policy. This on-policy design aligns training with the states the deployable agent actually encounters, while keeping true futures unavailable during inference.

### D Dataset Details

This section provides the full category definitions and split distributions for the two benchmarks summarized in the main paper.

#### D.1 Benchmark Verification Protocol

Both benchmarks are manually verified after automatic filtering. Human verification is used as a final quality gate rather than as a source of model input. Retained examples must have a valid initial observation, a unique answer, plausible distractors, and no leakage of the future outcome. Ambiguous, visually inconsistent, or option-invalid items are removed.

Dataset Human verification checks Removed case types VRQABench Visual consistency; unique

Invalid question text; implausible option; ambiguous puzzle state

solver-derived answer; plausible options

OpenWorldQA Anchor validity; answer uniqueness; no future leakage; option plausibility

Ambiguous outcome; leaked anchor; visually inconsistent question; invalid option

Table 12: Minimal verification protocol for the two benchmark suites.

#### D.2 VRQABench Category Taxonomy and Distribution

VRQABench is built from three VR-Bench task families: 2D maze navigation, irregular-maze path tracing, and Sokoban box pushing. Each example is a four-choice question generated from an initial puzzle image. Table 13 defines the five retained spatial question categories, Table 14 reports their train/evaluation distributions, and Table 15 summarizes the task-family distribution.

#### D.3 OpenWorldQA Category Taxonomy and Distribution

OpenWorldQA contains four-choice questions from real-world short videos. Each question uses an anchor frame before the outcome is visible and asks

Category Applicable tasks Required reasoning

C1_turn_count maze, irregular maze Count total turns along the optimal path. C2_turn_direction maze, irregular maze Count left or right turns along the optimal path. C3_sokoban_push Sokoban Infer minimum push count or the first push direction. C4_direction_count maze Count optimal-path steps in a specified direction. C5_push_dir_count Sokoban Count pushes in a specified direction in the optimal solution.

Table 13: VRQABench category definitions.

Category Train Eval Total C1_turn_count 1,194 186 1,380 C2_turn_direction 1,033 165 1,198 C3_sokoban_push 928 147 1,075 C4_direction_count 637 95 732 C5_push_dir_count 208 43 251 Total 4,000 636 4,636

Table 14: VRQABench category distribution across train and evaluation splits.

Task type Train Eval Total maze 1,756 275 2,031 irregular maze 1,108 171 1,279 Sokoban 1,136 190 1,326 Total 4,000 636 4,636

Table 15: VRQABench task-family distribution.

the model to predict a future physical event. Table 16 defines the 12 physical reasoning categories, Table 17 reports the train/test category distribution, and Table 18 reports the question-type distribution.

### E Workflow-Agent Prompt Template

For the workflow-agent baselines, we use a fixed prompt that exposes Helios as an optional worldmodel tool while leaving all simulation decisions to the base MLLM. The agent receives only the initial image or anchor frame, the question, and the answer options; it never receives the groundtruth future or answer. A Helios query returns a 100-frame rollout, and the agent may issue at most three simulation queries per example.

Workflow-Agent System Prompt

You are a multimodal future-prediction agent equipped with an optional video

world-model tool. Your goal is to answer four-choice questions about what will happen after the current image or anchor frame.

\textbf{Inputs.} You are given an initial observation, a question, and four

answer options. The future video and the correct answer are hidden from you.

\textbf{Capabilities.} \begin{enumerate}[leftmargin=*, itemsep=1

pt, topsep=1pt] \item \textbf{Abstract reasoning.}

You can reason from the static visual evidence, question, options, spatial constraints, physical rules, and

task context.

\item \textbf{Concrete simulation.} When the future outcome depends on non-trivial motion, contact, path following, object interaction, or physical dynamics, you may call the world model to generate a candidate future rollout.

\end{enumerate} \textbf{Protocol.} \begin{itemize}[leftmargin=*, itemsep=1pt,

topsep=1pt] \item First decide whether the answer

can be inferred reliably from the static observation alone.

\item If simulation is useful, output

\texttt{<SIMULATE>} followed by a concise, task-specific simulation prompt describing the relevant objects, motion, path, contact, or physical event to roll out. The simulator will return a 100-frame generated video.

\item After receiving a rollout, act as a gatekeeper before using it. If the rollout is visually consistent with the input, physically plausible,

and relevant to the question, output \texttt{<ACC>} and incorporate it

into your reasoning. If it changes important geometry, misses the queried interaction, hallucinates objects, looks glitchy, or conflicts with physical constraints, output \ texttt{<REJ>} and either retry with a

better prompt or fall back to abstract reasoning.

\item You may use at most three simulation attempts. Rejected or uncertain rollouts should be discounted rather than treated as ground truth.

\item Finish with exactly one choice in the format \texttt{Final Answer: [ A/B/C/D]}.

Category Name Description C1_fit_clearance fit clearance Infer the intermediate pose or orientation needed for a rigid object to pass

through limited clearance. C2_spatial spatial relation Predict final position or region after a physical event. C3_containment containment Reason about overflow, filling, liquid distribution, or whether a container

can hold the material. C4_support support stability Predict stability, tipping direction, collapse, or balance. C5_friction friction Predict sliding distance, slipping versus gripping, or stopping position. C6_inertia inertia Predict trajectory, direction, or distance after force application or removal. C7_fluidity fluidity Predict flow path, spill threshold, splash pattern, or absorption. C8_deformability deformability Predict stretching, tearing, wrinkling, breaking points, or elastic versus

permanent deformation. C9_tool_use tool use Predict the physical result of a tool acting on an object.

- C10_chain_reaction chain reaction Predict indirect or downstream effects beyond the first contact point.
- C11_process_race process race Decide which of two concurrent physical processes completes first.
- C12_multi_body multi-body motion Reason about momentum transfer, competing forces, or spatial conflicts among multiple bodies.

Table 16: OpenWorldQA category definitions.

Category Train Train % Test Test % Total C1_fit_clearance 34 45.3 41 54.7 75 C2_spatial 1,062 96.2 42 3.8 1,104 C3_containment 31 43.1 41 56.9 72 C4_support 843 95.3 42 4.7 885 C5_friction 63 60.6 41 39.4 104 C6_inertia 168 80.0 42 20.0 210 C7_fluidity 87 67.4 42 32.6 129 C8_deformability 443 91.3 42 8.7 485 C9_tool_use 853 95.3 42 4.7 895

- C10_chain_reaction 33 44.6 41 55.4 74
- C11_process_race 200 82.6 42 17.4 242
- C12_multi_body 87 67.4 42 32.6 129 Total 3,904 88.6 500 11.4 4,404

Table 17: OpenWorldQA category distribution across train and test splits. The test split is approximately balanced across categories, with 41–42 questions per category.

Question type Count Share [ORDER] 1,458 33.1 [FIRST-CONTACT] 1,213 27.5 [INTERMEDIATE] 1,093 24.8 [COUNT] 254 5.8 [COUNTERFACTUAL] 208 4.7 [FAILURE] 178 4.0 Total 4,404 100.0

Table 18: OpenWorldQA question-type distribution.

\end{itemize}

### F Dataset Construction Prompts

For reproducibility, we include the datasetconstruction prompt materials used to build the two benchmarks. Each dataset begins with a short prose overview for context, followed by the actual stage or agent prompts shown verbatim; only non-

ASCII decorative symbols such as box-drawing characters, arrows, and warning/check icons are normalized to ASCII for LaTeX compatibility. After these automatic stages, every retained item in both datasets is manually verified by human annotators, and samples with ambiguous answers, invalid anchors, implausible options, or visual inconsistencies are removed.

F.1 Full VRQABench Dataset-Construction Prompt

The VRQABench v2 pipeline replaces the original VLM-based solution tracing with a programmatic solver, eliminating hallucination in the answer generation stage. VLM calls are limited to writing question text and reviewing question quality. The overall construction flow is: state.json → programmatic solver → build_qa_items() → QuestionWriter → SmallModelProbe → Reviewer → human verification → output/reviewed/. The

actual stage prompts and specifications are shown below.

- VRQABench Step 1: Programmatic Solver

- ## Stage 1 - Programmatic Solver

**Role.** Reads `state.json` (groundtruth game state bundled with each puzzle) and computes the exact solution via search or geometry. No VLM involved.

**Solver by task type.**

| Task | Algorithm | Key outputs | |------|-----------|-------------| | maze | BFS on grid (`grid.data`, walls

= 1) | `steps`, `total_turns`, ` left_turns`, `right_turns` |

| irregular\_maze | Geometric angle analysis on `solution_path` waypoints

(turn threshold 30 degrees) | ` direction_changes`, `left_turns`, ` right_turns` |

| sokoban | Push-level BFS over `( player_pos, box_positions)` states with inner reachability check | ` push_sequence`, `total_pushes`, ` first_push_direction` |

**Turn counting rules.**

- - maze: a turn is any step where the movement direction differs from the previous step; a U-turn (180 degrees)

counts as two turns.

- - irregular\_maze: a junction is a turn when the angle between incoming and outgoing vectors exceeds 30 degrees; straight-through junctions are ignored.

**Direction conventions.**

- - maze / sokoban: `up`, `down`, `left`, ` right` (image coordinate frame, yaxis down).
- - irregular\_maze: turns are classified as left or right using the cross product of consecutive direction vectors (image coordinate frame: cross > 0 -> right).

VRQABench Step 2: QA Item Builder

- ## Stage 2 - build_qa_items()

**Role.** Programmatically generates answer values and distractor options for each applicable question category

. No VLM. The `question` field is left empty at this stage.

**Question categories.**

| Category | Question asked | Applicable tasks |

|---------|------------|--------------| | C1\_turn\_count | How many times does

the path change direction in the optimal solution? | maze, irregular\ _maze |

| C2\_turn\_direction | How many times does the path turn left (or right) in

the optimal solution? | maze, irregular\_maze |

| C3\_sokoban\_push | (a) What is the minimum number of pushes to solve the

puzzle? (b) In which direction is the first push? | sokoban |

| C4\_direction\_count | How many steps

does the path take going \<direction \> in the optimal solution? | maze |

| C5\_push\_dir\_count | How many times is the box pushed \<direction\> in the optimal solution? | sokoban |

For C4 and C5, the direction asked is the

**most-frequent direction** in the solution (to avoid trivially-zero answers).

**Questions generated per sample.**

| Task | Questions | |------|-----------| | maze | up to 3 (C1 + C2 + C4) | | irregular\_maze | up to 2 (C1 + C2) | | sokoban | up to 3 (C3-count + C3-

direction + C5) |

**Distractor design.**

- - *Counting questions*: distractors are sampled from integers within +/-4 of the correct value (all >= 0); correct

answer placed at a randomly chosen option letter.

- - *Direction questions*: all four cardinal directions appear exactly once across the four options; correct

answer placed at a random letter.

VRQABench Step 3: QuestionWriter Prompt

## Stage 3 - QuestionWriter (VLM)

**Role.** Receives the puzzle image and the category/direction metadata. Writes ONE natural-language question sentence describing the puzzle visually. Does **not** generate or modify the answer.

**Model.** gpt-5.5 (MODEL_LARGE).

**Output format.** ```json {"question": "Starting from the ball in

the upper-right corner, following the

shortest path to the green square at the bottom-left, how many times does the path change direction?"}

```

**Writing constraints.**

- - Mention start and goal positions using visible landmarks (not grid coordinates).
- - Use the exact question phrasing for the category.
- - Do **not** include any numbers, directions, or hints toward the correct answer.
- - One sentence is ideal, two at most.

- VRQABench Step 4: SmallModelProbe Filter

## Pre-Reviewer Filter - SmallModelProbe Before entering the Reviewer stage, each

candidate question is subjected to a difficulty probe using a smaller model (GPT-5.4-nano):

- 1. The question is presented **twice**, each time with the option order independently shuffled.
- 2. If the probe answers correctly on ** both** attempts, the question is classified as `too_easy` and discarded without reaching the Reviewer.

This filter removes questions that can be

answered correctly without genuine spatial reasoning.

- VRQABench Step 5: Reviewer Prompt

## Stage 4 - Reviewer (VLM)

**Role.** Receives the puzzle image and the QA list. Evaluates **question text quality only** - the answer key is trusted as correct ( programmatically verified) and is not

re-examined.

**Model.** gpt-5.5 (MODEL_LARGE).

**Verification criteria.** | Criterion | Pass condition | |-----------|---------------| | Question text validity | The question

text accurately describes features visible in the puzzle image (start/ goal positions, task type language). |

| Distractor plausibility | Each wrong option is plausible; no option is

obviously impossible given the visible puzzle layout. |

| Difficulty | The answer requires genuine path-tracing reasoning; it cannot be read off the image trivially. |

**Acceptance rule.** A question is accepted if and only if all three criteria pass and the overall quality

score is >= 7 out of 10.

VRQABench Step 6: Final Data Format

## Final Data Format Each accepted question is saved as an

individual JSON file (`VRB_{category} _{NNNN}.json`):

```json {

"category": "C1_turn_count", "question": "Starting from the ball in

the upper-right corner, following the shortest path to the green square at the bottom-left, how many times does the path change direction?",

"options": {"A": "8", "B": "11", "C":

"9", "D": "10"}, "answer": "D", "correct_value": "10", "source_sample_id": "maze_1_hard_0097_0

", "task_type": "maze", "input_image": "<absolute path to

puzzle image>", "video_path": "<absolute path to solution video>",

"_solution": { "total_turns": 10, " left_turns": 5, "right_turns": 5, "..." : "..." },

"_review": { "decision": "accept", " score": 9, "..." : "..." },

"_pipeline_metadata": { " pipeline_version": "v2", "..." : "..." }

} ```

F.2 Full OpenWorldQA Dataset-Construction Prompts

The OpenWorldQA dataset is constructed with a fivestage multi-agent pipeline: SceneAnalyst, QuestionDesigner, DistractorForge, SmallModelProbe, and Reviewer. The pipeline starts from a video frame sequence, produces a structured scene report and anchor frame, designs question skeletons, fills plausible distractors, filters questions with two small-model probes, and applies a five-dimensional reviewer. The accepted items are then manually

verified by human annotators before being written to the train/test splits. The actual agent prompts are shown below.

OpenWorldQA Step 1: SceneAnalyst Prompt

## Agent 1: SceneAnalyst

**Role:** Receives the full frame sequence, produces a structured physical scene report, and selects the best anchor frame.

--``` You are SceneAnalyst - the first agent in

a multi-stage physical reasoning QA pipeline.

You receive a sequence of frames extracted from a short video clip (up to 12 frames, ~0.5s apart, covering the first 10 seconds of the video). Your job is twofold:

- 1. Produce a structured physical scene report that later agents will use to design hard questions.
- 2. Select the best anchor frame - the single frame that best represents the

moment just BEFORE

the main physical event reaches its outcome (i.e., initial-condition frame).

============================== ============================== Analyse ALL provided frames carefully and

- PART 1 - SCENE REPORT

fill every section below.

[OBJECTS] List every physically relevant object

visible. For each object state:

- name, approximate size (small/medium/ large relative to scene), material or texture cues,

deformability (rigid / flexible / liquid), and current state (static / moving / being held).

[AGENTS] Describe the people present: number, body

position (standing/sitting/crouching ),

which hand(s) are active, and what they appear to be doing.

[FORCES & MOTION] Identify any visible forces or motions

across the frame sequence:

- direction of movement (toward / away / left / right / downward / rotating)

- - estimated speed (slow / moderate / fast)
- - contact points between objects or between person and object
- - any visible momentum, spin, tilt, or instability

[PHYSICAL STATE AT EACH FRAME] For each frame (frame_001 ... frame_N),

write one sentence describing the key physical state

change compared to the previous frame. Focus on: position change, contact change, deformation,

fluid level change, or stability change. [CANDIDATE EVENTS] List up to 3 distinct physical events

that are either in progress or clearly about to happen.

For each event describe:

- - what is happening physically
- - what the uncertain outcome is (i.e., what a model must PREDICT, not just observe)
- - which frames bracket the event (e.g., "frame_003 to frame_007")

[SUITABLE CATEGORIES] From the list below, identify which

categories are NATURALLY AND CLEARLY present in this clip.

Only include a category if there is genuine visual evidence for it - do not guess.

C1_fit_clearance - predicting whether an object fits through/into a space given size cues C2_spatial - predicting final spatial position/region after a physical event

- C3_containment - overflow, underfill, fit vs. no-fit, liquid distribution
- C4_support - stability, tipping direction, collapse, balance
- C5_friction - sliding distance, slip vs. grip, stopping point
- C6_inertia - trajectory, momentum after release, spin, rebound direction
- C7_fluidity - flow path, overflow threshold, splash pattern, absorption
- C8_deformability - stretch, tear, crumple, snap, permanent vs. elastic deformation
- C9_tool_use - physical outcome of tool acting on target (not human intent)
- C10_chain_reaction - indirect / downstream effect beyond the direct contact
- C11_process_race - which of two concurrent physical processes completes first

C12_multi_body - momentum transfer, force contest, or spatial conflict between bodies

============================== ============================== Select the single best anchor frame using

- PART 2 - ANCHOR FRAME SELECTION

this criterion: The anchor must show the INITIAL CONDITIONS of the main physical event

- enough context for a reasoning model to predict the outcome - but must NOT yet reveal the

outcome itself. Rules:

- - Prefer the frame just BEFORE the critical transition (e.g., object about to be released,

liquid about to reach the rim, object tipping but not yet fallen).

- - Reject frames where the answer is already visible (object already fallen, already spilled).
- - Reject the very first frame if it appears to be a cover/title/static shot with no action.
- - If the video has no clear physical event, still pick the most informative frame.

============================== OUTPUT FORMAT

============================== Return a single JSON object with the

following structure (ALL text in English):

{

"objects": [

{"name": "...", "size": "...", " material": "...", "deformability": "...", "state": "..."}

], "agents": "...", "forces_and_motion": "...", "frame_states": {

- "frame_001": "...",
- "frame_002": "...",

...

}, "candidate_events": [

{

"description": "...", "uncertain_outcome": "...", "frame_range": "frame_XXX to

frame_YYY" }

], "suitable_categories": ["

C1_fit_clearance", "C4_support",

...], "anchor_frame": "frame_XXX",

"anchor_rationale": "One sentence explaining why this frame is the best anchor."

} Output the JSON and stop. Do not add

commentary outside the JSON. ```

OpenWorldQA Step 2: QuestionDesigner Prompt

## Agent 2: QuestionDesigner

**Role:** Receives the scene report (text

only), designs 6 question skeletons without answer options - those are filled in by DistractorForge.

--``` You are QuestionDesigner - the second

agent in a multi-stage physical reasoning QA pipeline.

You receive a structured scene report produced by SceneAnalyst (text only no images).

Your job is to design 6 candidate question SKELETONS that will become hard physical-reasoning

multiple-choice questions. You do NOT write the answer options yet - that is DistractorForge's job.

============================== CORE PRINCIPLE

============================== A question is only hard if the answer

REQUIRES mentally simulating 1-3 future steps.

Self-check before finalising each question: "If I did NOT run a short mental

simulation of what happens next, could I still answer this correctly

more than 60% of the time?" If YES -> discard this question and try a different type.

============================== QUESTION TYPES - you MUST use only these

six

============================== For each question, assign exactly one

type tag: [ORDER] Which of two specific

events happens FIRST from the current state?

[COUNT] How many times will X change / switch / contact before the action ends?

[FIRST-CONTACT] What does the subject first make contact with next?

[INTERMEDIATE] What transient/ transitional state does the system enter BEFORE the final outcome?

[FAILURE] If uncorrected, which specific step or component fails first?

[COUNTERFACTUAL] If the current posture / force / direction is maintained unchanged,

which branch does the future trajectory approach?

============================== STRICTLY FORBIDDEN question patterns

============================== Do NOT generate questions of these forms -

they rely on high scene-prior and do not require

future simulation: [REJECT] "What will the person do next

?" [REJECT] "Will X succeed or fail?" [REJECT] "What is the person trying to

do?" [REJECT] "Will the object be picked up / opened / poured / placed?"

[REJECT] Any question whose correct answer follows directly from gravity alone

("the object will fall") without requiring specific scene details.

============================== CATEGORY ALIGNMENT

============================== The scene report includes "

suitable_categories". Assign each question to one of those categories.

Try to cover at least 3 different

categories across your 6 skeletons. Category definitions for reference:

C1_fit_clearance - fit/gap prediction

based on visible size relationships C2_spatial - final resting

position or region after motion C3_containment - overflow, underfill, fit vs. no-fit

C4_support - tipping direction, collapse, stability C5_friction - sliding endpoint, slip vs. grip

- C6_inertia - trajectory / direction / distance after force applied or removed
- C7_fluidity - flow path, overflow threshold, splash, absorption
- C8_deformability - stretch/tear/snap/ buckle point

C9_tool_use - physical result of tool on target (not human intent)

- C10_chain_reaction - indirect downstream effect beyond direct contact
- C11_process_race - which concurrent process completes first
- C12_multi_body - momentum transfer, force contest between bodies

============================== ANCHOR AWARENESS

============================== The scene report specifies an anchor

frame. Every question MUST be answerable from that frame

alone (given physical reasoning). Do NOT design questions that require information only visible

AFTER the anchor frame - the test model will only see the anchor.

[WARNING] NEVER reference frame numbers in question text (e.g., do NOT write "From frame_005",

"At frame_008", "As seen in frame_003 ", etc.). The question must describe the scene state in plain language only. The anchor frame identity is handled by the pipeline - not by the question text.

============================== OUTPUT FORMAT

============================== Return a single JSON object (ALL text in

English): {

"question_skeletons": [ {

"id": 1, "type": "[ORDER] / [COUNT] / [FIRST-

CONTACT] / [INTERMEDIATE] / [FAILURE]

/ [COUNTERFACTUAL]", "category": "C_X_name", "question_text": "Full question

sentence referencing specific objects and scene details.",

"key_physical_principle": "One sentence: what physical knowledge is needed to answer this?",

"true_answer_direction": "Brief description of what the correct answer should describe

(NOT the exact wording - DistractorForge will write options).",

"why_hard": "One sentence explaining why this cannot be answered without simulation.", "anchor_frame": "frame_XXX" },

... (exactly 6 skeletons) ]

} Output the JSON and stop. ```

OpenWorldQA Step 3: DistractorForge Prompt

## Agent 3: DistractorForge

**Role:** Receives the scene report and question skeletons, generates complete four-option QA items for each skeleton, and selects the best 3

out of 6.

--``` You are DistractorForge - the third agent

in a multi-stage physical reasoning QA pipeline.

You receive:

- 1. The structured scene report from SceneAnalyst (text only).
- 2. The 6 question skeletons from QuestionDesigner (text only).

Your job is to turn each skeleton into a complete multiple-choice QA item with 4 options

(A/B/C/D) where ALL four options are physically plausible in the scene context.

============================== DISTRACTOR QUALITY - this is your primary

responsibility

============================== Every wrong option (distractor) MUST

satisfy ALL three conditions:

- 1. PHYSICALLY POSSIBLE in the real world under some conditions.
- 2. LOCALLY PLAUSIBLE in THIS specific scene - a viewer unfamiliar with the exact physics

could reasonably believe it might happen given the visible setup.

- 3. CLEARLY WRONG when the anchor frame is examined carefully using correct physical reasoning.

ABSOLUTELY FORBIDDEN distractor types ( instant reject): [REJECT] Objects floating, levitating, or defying gravity [REJECT] Objects freezing in midair without cause [REJECT] Objects passing through solid surfaces [REJECT] Motion reversing spontaneously without an applied force

[REJECT] Any outcome that violates basic Newtonian physics

GOOD distractor strategies - make options differ in:

- - Degree : "slides 10cm past the edge" vs "stops just before the edge"

vs "stops 5cm short"

- - Direction : "tips left" vs "tips right" vs "slides forward off the surface"
- - Timing : "overflows immediately" vs "overflows after 2 more pours" vs "never reaches the rim"
- - Part : "snaps at the midpoint" vs "snaps near the grip end" vs " bends permanently but holds"
- - Sequence : "A happens before B" vs " B happens before A" vs "they happen simultaneously"

The four options should be difficult to distinguish without careful examination of the anchor.

A careless observer should find at least 2 options tempting.

============================== QUESTION-ONLY SELF-CHECK

============================== Before finalising each QA item, mentally cover the scene report and ask:

"If I only read the question text (no image), could I guess the correct answer

more than 60% of the time based on world knowledge alone?"

If YES -> the question is too high-prior.

Rewrite the question to reference specific

scene details that anchor it (e.g., " given the visible tilt angle of ~30 degrees",

"given the surface appears to be smooth tile", "given the cup is already ~80% full").

============================== QUESTION TEXT RULES

============================== [WARNING] NEVER include frame numbers in

any question or option text.

Do NOT write "From frame_005", "At frame_008", "As shown in frame_003", etc.

Describe the scene state in plain language only (e.g., "As the person begins lowering

toward the seat with their right hand near their face").

The anchor frame identity is managed by the pipeline - not by the question text.

[WARNING] NEVER reference events or states that are only visible AFTER the anchor frame.

The question and all options must be answerable from the anchor frame state alone.

Do NOT use phrases like "later seen", "as subsequently shown", "which then becomes visible", etc.

============================== ANSWER KEY CORRECTNESS

============================== The correct answer must be:

- - Uniquely supported by the physical evidence described in the scene report.
- - Not directly visible in the anchor frame (requires prediction, not recognition).
- - Consistent with the " true_answer_direction" from the skeleton.

Write a clear explanation (2-4 sentences)

citing specific scene details ( object material,

visible angle, fill level, speed, contact

surface, etc.) that support the correct answer.

Write a one-sentence justification for each distractor explaining why it is plausible

but wrong given the specific scene.

============================== SELECTION - choose the best 3 out of 6

============================== After completing all 6 QA items, select

the best 3 to pass to the next stage. Prioritise:

- 1. Questions that cover different categories (diversity).
- 2. Questions with the strongest distractor quality (all 3 wrong options are tempting).
- 3. Questions that most clearly require future-process simulation to answer.

Mark the 3 selected items with "selected":

true and the other 3 with "selected ": false.

============================== OUTPUT FORMAT

============================== Return a single JSON object with key "

qa_drafts" containing exactly 6 items

(ALL text in English): {

"qa_drafts": [ {

"id": 1,

"selected": true, "category": "C_X_name", "category_desc": "Brief English

description",

"question_type": "[ORDER] / [COUNT] / etc.",

"question": "Full question text referencing specific scene details.", "options": {

- "A": "...",
- "B": "...",
- "C": "...",
- "D": "..."

}, "answer": "A", "explanation": "2-4 sentences

citing specific visual evidence for the correct answer.",

"distractor_justification": {

- "B": "Plausible because X, but

wrong because Y.",

- "C": "Plausible because X, but

wrong because Y.",

- "D": "Plausible because X, but

wrong because Y." }, "difficulty_estimate": "Hard", "anchor_frame": "frame_XXX", "video_path": "", "sample_id": ""

},

... (6 total, 3 with selected=true) ]

} Output the JSON and stop. ```

OpenWorldQA Step 4: Small-model Difficulty Probe

## Agent 4: SmallModelProbe

**Role:** Filters questions by difficulty

using a small model (gpt-5-nano). A question is discarded if the small model answers it correctly on both independent attempts; otherwise it passes. The prompt is intentionally minimal.

--``` Answer the physical reasoning question

below using the image provided. Choose the single best answer. Reply with

JSON only: {"answer": "A"} The answer must be one of: A, B, C, or D. ```

> **Note:** The question text is appended

after this prompt in the following format:

###### > ```

> QUESTION: <question text> > > OPTIONS:

- > A: <option A>
- > B: <option B>
- > C: <option C>
- > D: <option D> > ``` > Each question is tested twice

independently with options shuffled randomly to avoid position bias. A question passes only if the small model answers incorrectly on at least

one attempt.

OpenWorldQA Step 5: Reviewer Prompt

## Agent 5: Reviewer

**Role:** Final quality gate. Receives the anchor frame + context frames + probe-surviving QA items, evaluates each item across five dimensions ( answer correctness, anchor validity, distractor plausibility, visual consistency, category alignment), and

accepts only items scoring >= 7.

--``` You are Reviewer - the final quality-gate

agent in a multi-stage physical reasoning QA pipeline.

============================== IMAGE LAYOUT - READ THIS FIRST ============================== You receive images in two groups, clearly labelled in the INPUT section: [ANCHOR] - exactly 1 image.

This is the ONLY image the test model sees at evaluation time.

When you evaluate

anchor_validity, evaluate THIS image. [CONTEXT] - 0 or more images that

follow.

These are provided for your reference ONLY to verify answer

correctness.

They are NOT available to the test model at evaluation time.

The INPUT section at the bottom of this

prompt lists each image in order. Image 1 in your visual input = the anchor.

Images 2+ = context frames.

============================== PIPELINE CONTEXT:

- The SmallModelProbe has already rejected questions that gpt-5-nano answered correctly

on two independent attempts (with shuffled option order).

- - Questions reaching you have PASSED the difficulty gate.
- - Do NOT re-evaluate difficulty. Focus on correctness and plausibility only.

============================== YOUR REVIEW TASKS (evaluate each QA

independently)

==============================

- 1. ANSWER CORRECTNESS Using ALL frames (anchor + context),

verify that the claimed correct answer is physically

accurate. Consider: physical laws, object properties, visible forces, trajectory.

Reject if the answer key is wrong or ambiguous.

- 2. ANCHOR VALIDITY Look at Image 1 (the anchor frame,

filename in INPUT section).

Does it show sufficient initialcondition cues for a reasoning model to predict the outcome?

Does it NOT already reveal the answer

directly? Reject if:

- (a) Image 1 is a blank/cover shot with no relevant physical content, OR
- (b) The outcome is already fully visible in Image 1 (static leakage).

- 3. DISTRACTOR PHYSICAL PLAUSIBILITY For each wrong option independently:

- - Is it physically possible in the real world under some conditions? ( MUST be yes)
- - Could a viewer reasonably think it might happen in this specific scene? (MUST be yes)

Reject immediately if ANY distractor:

- - Defies gravity or levitates without cause
- - Freezes in midair
- - Passes through solid objects
- - Reverses motion spontaneously
- - Is obviously nonsensical in the scene context

- 4. VISUAL CONSISTENCY Does the question text accurately

describe what is shown in the frames?

Reject if the question references objects or actions not visible anywhere in the clip.

- 5. CATEGORY ALIGNMENT Does this QA item genuinely test the

assigned category's type of reasoning ?

(Not a blocking reject - note and suggest correction if misaligned.)

==============================

APPROVE if ALL of the following hold:

============================== [PASS] Answer key is correct and unambiguous

[PASS] Image 1 (anchor) provides sufficient initial-condition cues without revealing the answer

[PASS] Every distractor is physically plausible and locally tempting in this scene

[PASS] No distractor violates basic physics [PASS] Question text matches visible content [PASS] Score >= 7

REJECT if ANY of the following hold: [REJECT] Answer key is wrong or ambiguous

[REJECT] Image 1 (anchor) is a cover/ blank shot OR already reveals the outcome

[REJECT] Any distractor defies physics or is obviously nonsensical [REJECT] Question references objects or

actions not visible in the clip [REJECT] Score < 7

============================== OUTPUT FORMAT

============================== Return a single JSON object with key "

review_list" containing one entry per QA item (same order as input). ALL text in

English. {

"review_list": [ {

"decision": "accept", "score": 8, "reasoning": {

"answer_correctness": {" pass": true, "note": "..."},

"anchor_validity": {" pass": true, "note": "..."},

"distractor_plausibility": {" pass": true, "note": "Evaluate each wrong option individually."},

"visual_consistency": {" pass": true, "note": "..."},

"category_alignment": {"

pass": true, "note": "..."} }, "suggestions": ""

},

... ]

} Output the JSON and stop. ```

