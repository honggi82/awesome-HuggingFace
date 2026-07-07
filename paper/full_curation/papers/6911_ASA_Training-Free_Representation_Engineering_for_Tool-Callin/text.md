# arXiv:2602.04935v3[cs.SE]9Jun2026

## ASA: Backbone-Training-Free Representation Engineering for Tool-Calling Agents

### Youjin Wang1* Run Zhou1* Yingjie Ma1* Rong Fu2 Jiani Liang1 Shuaishuai Cao3 Min Huang4 Tao Fang5† Liangming Pan6†

1Renmin University of China 2University of Macau 3Central South University 4Jiangxi Normal University 5Macau Millennium College 6Peking University wangyoujin@ruc.edu.cn taylefang@gmail.com liangmingpan@pku.edu.cn

### Abstract

Robust tool calling in LLM agents requires executable schema compliance, yet schema shifts often cause missing calls or parserinvalid outputs. We study this problem by decomposing tool-use representations into a shared boundary component and domain-local residual components. Our analysis reveals that tool-use evidence can predict boundary entry but does not guarantee executable schema realization across format, tool-name, argument, and sequence constraints. Based on this decomposition, we propose Activation Steering Adapter (ASA), a backbone-training-free inference-time controller that gates shared boundary steering with domain-local residual control. On BFCL, ASA improves Qwen38B across multi-turn, single-turn non-live, and live settings, including a 32.50% to 38.75% gain on multi-turn Prompt-mode. On NESTFUL, it reduces missing-tool failures from 59.14% to 6.72% while improving ﬁrst-call and sequence accuracy. Error analysis further shows that ASA mainly improves boundary entry and trajectory continuation, while ﬁnegrained schema realization and nested argument binding remain the main bottlenecks.

### 1 Introduction

Large pretrained models increasingly act as agents that retrieve information, call APIs, execute code, and interact with external environments, where a missed, wrong, or malformed tool call can directly invalidate the answer. For example, a model may recognize that an API call is needed, yet answer in natural language, choose an invalid function, or omit a nested argument. This problem is especially difﬁcult in domain-speciﬁc tool calling, where tool names, schemas, argument meanings, and calling conventions vary across domains. Existing benchmarks show that models degrade un-

*Equal Contribution †Co-corresponding Authors

der strict multi-step, parallel, nested, and longcontext tool-calling settings (Schick et al., 2023; Qin et al., 2024; Patil et al., 2024; Li et al., 2023; Patil et al., 2025; Wu et al., 2024a). Our promptonly ceiling study further shows that stronger prompting still leaves a strong proprietary baseline degraded on BFCL parallel variants and OOD multi-level API settings (see Appendix A).

Existing adaptation strategies operate at the input interface or in parameter space: prompting, demonstrations, tool-document retrieval, schema constraints, and structured decoding are easy to deploy but sensitive to prompt wording, context length, and interface changes (Yao et al., 2022; Qin et al., 2024; Schick et al., 2023; Patil et al., 2024; Dong et al., 2025). Supervised tool-use tuning, LoRA-style adapters, quantized variants, and reward-based tool learning can improve indomain invocation and format compliance, but require retraining and regression testing as tool sets evolve (Hu et al., 2022; Tang et al., 2023; Prottasha et al., 2025; Lu et al., 2025). Representationlevel studies further show that tool necessity and tool-calling decisions can be decoded from pregeneration hidden states and sometimes shifted by activation steering (Sun et al., 2026; Esakkiraja et al., 2026). These ﬁndings leave open a stricter execution question: does readable tool-use evidence guarantee executable tool calls?

We show that readable tool-use evidence mainly predicts whether the model crosses the ﬁrst boundary from direct-answer generation into a parserrecognizable tool-call format (boundary entry). However, boundary entry is not equivalent to strict execution: a generated output must still satisfy format constraints, use a valid tool name, provide valid arguments, and, in multi-step or nested settings, preserve the required call sequence. We refer to this mismatchhidden states contain tooluse evidence, but decoding fails to realize an executable callas the Intent–Execution Gap. This

gap suggests two coupled problems: crossing the direct-answer/tool-call boundary and realizing the correct schema after entry.

Motivated by this view, we propose Activation Steering Adapter (ASA), a backbonetraining-free inference-time controller for schemaconstrained tool calling. ASA uses a signed gate to open the tool boundary, suppress spurious boundary crossing, or abstain under uncertainty, and applies domain-local residual steering when reliable local estimates are available.

Our main contributions are as follows.

- 1. We formulate schema-constrained tool calling as a decomposed hidden-state control problem, distinguishing boundary entry from post-entry schema realization.
- 2. We propose ASA, a backbone-training-free inference controller that combines shared boundary steering, local residual steering, and a probe-guided abstention gate.
- 3. We evaluate ASA on NESTFUL and BFCL, showing that it reduces missing-tool failures and improves executable accuracy. Beyond aggregate scores, we provide an error-level analysis showing that ASA mainly corrects boundary-entry and trajectory-continuation failures, while ﬁne-grained schema realization and nested argument binding remain the dominant residual bottlenecks.

### 2 Related Work

#### 2.1 Adapting LLMs for Tool Calling

LLM agents extend language models from text generation to external action. ReAct elicits tool use by interleaving reasoning and acting, while Toolformer shows that models can learn API use from self-supervised data (Yao et al.,

- 2022; Schick et al., 2023). ToolLLM and Gorilla scale tool use to large API collections and API-grounded invocation (Qin et al., 2024; Patil et al., 2024). More recent function-calling models, including Functionary, NexusRaven, xLAM, and Phi-4-mini, further improve API invocation through tool-oriented instruction tuning or model specialization (MeetKai, 2023; Srinivasan et al.,
- 2023; Zhang et al., 2025; Abouelenin et al., 2025). Despite this progress, strict benchmarks such as BFCL and NESTFUL show that models still fail in multi-step, irrelevant-function, and nested-call

settings (Patil et al., 2025; Basu et al., 2025). ASA targets this stricter setting: rather than only eliciting tool use, it calibrates the hidden-state boundary between direct answering and parser-valid tool execution.

Function-calling reliability can also be improved through adaptation or constrained decoding. ToolAlpaca represents supervised tool-use instruction tuning, ToolACE constructs high-quality function-calling data through automatic agentic synthesis and veriﬁcation, and ToolRL studies reward design for reinforcement-learning-based tool use (Tang et al., 2023; Liu et al.; Qian et al., 2026). Hammer studies robustness when irrelevant functions are present (Lin et al., 2024). Parameterefﬁcient methods such as LoRA, QLoRA, and ReFT reduce adaptation cost but still require ﬁtting additional parameters (Hu et al., 2022; Dettmers et al., 2023; Wu et al., 2024b). Grammaror schema-constrained decoding can reduce malformed outputs after a constrained format has been selected (Dong et al., 2025; Geng et al., 2025). In contrast, ASA keeps the backbone frozen and intervenes before decoding commits to either a direct-answer or tool-call trajectory.

2.2 Representation Control and Hidden-State Tool Diagnostics

Representation engineering and activation steering show that model behavior can be analyzed or shifted through intermediate activations without changing prompts or backbone weights (Zou et al., 2023; Turner et al., 2023). Recent work extends this idea to conditional behavior control, agent policies, and reasoning-related states (Lee et al., 2025; Rahn et al., 2024; Venhoff et al., 2025). ASA follows this inference-time control paradigm, but applies it to parser-deﬁned tool execution rather than general behavioral steering.

Recent tool-use diagnostics show that tool decisions can be decoded from pre-generation hidden states. When2Tool probes tool necessity before generation, and ToolReadable shows that tool identity can be linearly readable and steerable in ﬁxed-menu settings (Sun et al., 2026; Wu et al., 2026). These works motivate our controlled Probe&Preﬁll-style and Tool-identity steering baselines, which test whether ASA’s gains can be explained by probe-triggered preﬁlling or ungated mean-difference steering alone. ASA asks a stricter question: whether readable tool-use evi-

dence guarantees boundary entry and correct tool execution under strict evaluation; we show it not.

### 3 Problem Setup

- 3.1 Parser-deﬁned tool execution Let x ∈ X denote an input instruction and y = (y1,...,yT) the generated output of an autoregressive model fθ. Each example has a binary oracle label y⋆(x) ∈ {0,1}, where y⋆(x) = 1 indicates that a tool call is required and y⋆(x) = 0 indicates that a direct answer is sufﬁcient.

A deterministic parser maps the generated string to a trigger event

Tm(x) ∈ {0,1}, (1)

where Tm(x) = 1 iff method m achieves boundary entry (i.e., enters tool mode). A triggered output may still be invalid, so we deﬁne schema realization under strict execution as

Sm(x) ∈ {0,1}, (2)

where Sm(x) = 1 iff the output achieves boundary entry and passes all evaluator-required post-entry checks, including format validity, tool-name validity, argument validity, and, when applicable, tasklevel or multi-turn execution correctness. The objective is to improve Sm(x) on Tool-Necessary inputs while keeping Tm(x) = 0 on Non-Tool inputs, without modifying θ.

The central question is whether pre-generation tool-use evidence, if readable from hidden states, is sufﬁcient for parser-valid execution.

#### 3.2 Diagnostic quantities

Let hL(x) ∈ RD be the Pre-LN residual-stream representation at layer L for the ﬁnal non-padding prompt token. For later use, we deﬁne standardized representations

h˜L(x) = Std(hL(x))

hL(x) − µtrain σtrain

=

,

(3)

where µtrain and σtrain are estimated on the training split.

Each input belongs to a domain d(x) ∈ Ddom. In this work, a domain denotes a tool-schema family rather than a topical category: examples in the same domain share tool names, argument semantics, and executable output conventions. A

Table 1: Representation-side diagnostics on Qwen3-8B. Tool-use evidence is linearly readable from mid-layer representations, and the corresponding direction shifts trigger-token evidence relative to random controls.

Diagnostic Setting Main quantity Result Linear probe hL(x) AUC ∼ 0.84 Shufﬂe control shufﬂed labels AUC 0.50 [0.43, 0.57] Positive direction scale = 1.00 ∆Logit / ∆Prob +0.84 / +7.1% Negative direction scale = 1.00 ∆Logit / ∆Prob −0.67 / −5.6% Random direction scale = 1.00 ∆Logit / ∆Prob +0.10 / +0.8%

lightweight predictor outputs dˆ, and the corresponding domain-speciﬁc probe estimates tooluse evidence:

p(x) = qψ(dˆ)(hL(x)) ∈ [0,1]. (4) Unless otherwise stated, p(x) denotes the probe score used both for diagnostic ranking and for ASA gating; it is trained and selected using only non-test splits. For diagnostics, we use the ToolNecessary subset:

X+ = {x | y⋆(x) = 1}, (5)

and for a fraction ρ, Bρ ⊆ X+ denotes the top-ρ examples ranked by p(x). These quantities are used in Section 4 to compare readable evidence, parservisible boundary entry, and strict executable success.

### 4 Evidence for the Intent–Execution Gap

This section empirically validates the Intent– Execution Gap. We ﬁrst show that tool-use evidence is readable before decoding, and then show that readable evidence does not guarantee parservalid execution. All experiments in this section use the diagnostic conﬁguration provided in Appendix B.

4.1 Tool-use evidence is readable before decoding

We ﬁrst verify that tool-use evidence is available before decoding. A linear probe on mid-layer residual representations separates Tool-Necessary from Non-Tool inputs with an AUC of approximately 0.84, while shufﬂed labels remain near chance. Moreover, steering along the corresponding direction shifts trigger-token evidence in the expected direction, whereas a norm-matched random direction has a much smaller effect.

This conﬁrms that the frozen backbone contains a readable and causally relevant tool-use signal. The stricter question is whether this signal is sufﬁcient for executable success.

- Table 2: Probe–Trigger dissociation on Qwen3-8B baseline. High tool-use evidence predicts boundary entry much better than schema realization under strict execution.

Probe bucket #samples Trigger Recall Strict Success No-trigger Rate

top_10pct 16 1.0000 0.2500 0.0000 top_25pct 40 0.9750 0.2000 0.0250 top_50pct 80 0.9750 0.2000 0.0250 all_positive 98 0.9694 0.2449 0.0306

- 4.2 Readable evidence does not guarantee strict execution

Readable tool-use evidence is not equivalent to executable success. We rank held-out Tool-Necessary test examples by the frozenrepresentation probe score p(x), where the probe is trained only on non-test splits. The all_positive row contains the 98 test examples with p(x) > 0.5; top-ρ rows are diagnostic buckets rather than separate evaluation splits. If tool-use evidence were sufﬁcient for execution, high-probe buckets should have high strict success. Table 2 shows the opposite pattern.

In the top 10% probe bucket, the baseline enters tool mode on all examples, but only 25.0% pass strict evaluator. Across top 25% and top 50% buckets, trigger recall remains 97.5%, whereas strict success stays at only 20.0%. Thus, the model often crosses the tool boundary but fails to produce a parser-valid executable call. This is the empirical signature of the Intent–Execution Gap: tooluse evidence is often present, but decoding does not reliably realize it as strict executable behavior.

#### 4.3 Implication for control

These results suggest that tool-calling control should not be treated as unconditional ampliﬁcation of generic toolness. A useful controller must decide when to open the tool boundary, when to suppress risky boundary crossing, and when to add schema-local bias after the boundary is reachable. As a supporting diagnostic, cross-domain tool-use directions are only partially aligned; we report the full cosine matrix in Appendix D. We therefore instantiate this view in ASA, which combines a shared boundary direction, domain-local residual directions, and a signed abstention gate.

### 5 Proposed Method

Section 4 shows that readable tool-use evidence does not guarantee parser-valid execution. ASA addresses this gap with a backbone-training-free

controller over a frozen model: a shared boundary direction calibrates tool-mode entry, domain-local residual directions support schema-speciﬁc realization, and a signed gate decides when to open, suppress, or abstain from intervention.

The following subsections deﬁne one shared gated update rule. We use this rule in two inference instantiations: a one-shot pre-decoding variant for diagnostics and a state-cascade variant for multi-step and nested tool-call trajectories.

#### 5.1 Boundary and domain-local directions

Using a calibration split disjoint from evaluation, ASA estimates a shared boundary direction at layer L. Let hL(x) ∈ RD denote the ﬁnal-token residual representation. We deﬁne

] − E

[

hL(x) | y⋆(x) = 1

vb = E

(6)

[

]

hL(x) | y⋆(x) = 0

.

This direction approximates the shared displacement needed for boundary entry (entering tool mode) from chat or non-tool behavior.

For each tool-schema domain d, ASA also estimates a raw domain direction:

] − E

[

hL(x) | y⋆(x) = 1, d(x) = d

vd = E

(7)

[

]

hL(x) | y⋆(x) = 0, d(x) = d

.

Because vd may still contain the same generic toolness component as vb, directly adding vb and vd can redundantly amplify tool-mode evidence. We therefore remove the shared boundary component from vd and keep the residual:

vd⊤vb ∥vb∥2

vb. (8)

vs(d) = vd −

Here vs(d) is the domain-local residual direction. It is an operational linear estimate of schema-local variation, not a claim of perfect semantic factorization. Before use, we unit-normalize the boundary and residual directions and write them as vˆb and vˆs(d), respectively.

#### 5.2 Signed abstention gate

At inference time, ASA computes three scalar signals from the current representation. A lightweight domain predictor gives the predicted tool-schema domain and its conﬁdence:

##### (d,sˆ dom(x)) = R(h˜L(x)). (9)

[Figure 1]

Figure 1: ASA overview. The framework contains three stages: (I) calibration asset preparation; (II) gated controller construction from frozen-backbone representations, including shared boundary and domain-local directions; and (III) inference-time intervention through ASA-Preﬁll or ASA-StateCascade with strict executable evaluation.

Here R is a lightweight domain predictor, dˆ is the predicted tool-schema domain, and sdom(x) ∈ [0,1] is its conﬁdence.

Given dˆ, a domain-speciﬁc probe estimates tooluse evidence:

p(x) = σ(wd⊤ˆhL(x) + bdˆ). (10) This logistic form instantiates the domain-speciﬁc probe qψ(dˆ) introduced in Section 3.2.

ASA also computes a boundary-readiness score

##### rb(x) = calib(vˆb⊤h˜L(x)), (11)

where calib(·) is a validation-ﬁtted scalar calibration map. The gate converts these signals into two intervention actions. Here rb(x) is a scalar readiness score, while ab(x) is a discrete boundary action. The boundary action ab(x) ∈ {−1,0,+1} opens the tool boundary, suppresses spurious boundary crossing, or abstains:

ab(x) =

 

+1, p(x) > τp and rb(x) < τb, −1, p(x) < 1 − τp and rb(x) > γ, 0, otherwise.



(12)

The schema-local action as(x) ∈ {0,1} activates only when the model is already boundaryready and the domain estimate is reliable:

##### as(x) = 1{p(x) > τp, rb(x) ≥ τb, sdom(x) > τs}.

(13)

Here τp,τb,γ,τs are validation-selected thresholds. ASA then applies the gated hidden-state update:

##### h′L(x) = hL(x)+ηb ab(x)ˆvb+ηs as(x)ˆvs(dˆ), (14)

where ηb and ηs are validation-selected update strengths. If ab(x) = as(x) = 0, ASA abstains and leaves the hidden state unchanged. Intuitively, p(x), rb(x), and sdom(x) correspond to tool necessity, boundary readiness, and domain reliability. Thus ab = +1 rescues missed boundary entry, ab = −1 suppresses spurious or premature entry, and as = 1 applies schema-local steering only after the boundary is reachable. This prevents ASA from acting as unconditional tool-use ampliﬁcation. For ASA-Preﬁll, Eq. (14) is applied once before decoding; for ASA-StateCascade, the same rule is re-evaluated and applied at selected generation states with validation-selected layer weights and thresholds as needed.

#### 5.3 ASA-Preﬁll inference

ASA-Preﬁll is implemented as a single forward hook at layer L. During pre-ﬁll, the hook reads the ﬁnal-token residual state, predicts the domain, computes tool-use evidence and boundary readiness, and applies Eq. (14) only when the signed gate activates. After this one-time update, autoregressive decoding proceeds normally. Disabling the hook exactly recovers the frozen backbone.

#### 5.4 ASA-StateCascade inference

ASA-StateCascade is the trajectory-level instantiation of the same gated update rule in practice. It is motivated by multi-turn and nested tool-calling settings, where correctness is not determined by a single pre-decoding decision: the model must enter tool mode when needed, continue incomplete tool-call trajectories, avoid premature stopping, and suppress calls on irrelevant turns when appropriate. At selected generation states, the controller re-computes state features, evaluates the signed gate, and applies a hidden-state correction when the current trajectory is predicted to require rescue, suppression, or abstention. These statelevel actions instantiate the same shared–local correction principle in Eq. (14) over a sequence of generation states, without updating backbone parameters. The implementation details, including active layers, state features, thresholds, and norm caps, are reported in Appendices E and F.

- 6 Experiments

We evaluate ASA under strict parser-deﬁned toolcalling interfaces using Qwen3-8B (Yang et al., 2025) with greedy decoding throughout our experiments. Unless otherwise stated, NESTFUL and BFCL multi-turn results use ASAStateCascade, as both settings evaluate trajectorylevel executable behavior rather than a single pregeneration tool-entry decision; ASA-Preﬁll is retained as a minimal one-shot variant for diagnostics and ablations.

#### 6.1 Experimental Setup

We evaluate ASA on NESTFUL (Basu et al., 2025) and BFCL (Patil et al., 2025) using Qwen3-8B with greedy decoding. Steering vectors, probes, thresholds, and hyperparameters are selected without access to the held-out evaluation split. All reported numeric metrics are percentages.

NESTFUL NESTFUL evaluates executable nested function calling, where later calls may depend on earlier call outputs. We report ﬁrst-call accuracy, full-sequence accuracy, and missingtool rate. First-call accuracy checks the ﬁrst emitted tool call; sequence accuracy requires the entire trajectory to be correct; missing-tool rate measures failures to call a required tool.

BFCL BFCL provides external validation under our ﬁxed local Qwen3-8B prompt pipeline, covering multi-turn Prompt-mode, non-live single-turn, and live single-turn settings. We report multi-turn success overall and by category (Base, missingfunction, missing-parameter, long-context). For single-turn settings, we report executable-subset AST accuracy and argument-level F1. The Train column indicates whether a method requires backbone training or adaptation.

Implementation, diagnostic, and efﬁciency details are provided in Appendices I, B, and K.1. 6.2 Main Results

Results on NESTFUL Table 3 shows that ASA substantially improves executable nested tool calling over the Qwen3-8B baseline. ASA raises overall ﬁrst-call accuracy from 24.46% to 41.94% and sequence accuracy from 16.94% to 25.00%, while reducing missing-tool failures from 59.14% to 6.72%. The two closest local controls clarify the source of the gain: Probe&Preﬁll improves boundary entry through probe-triggered preﬁx control, and Tool-identity steering reduces missing-tool failures through ungated mean-difference steering, but neither matches ASA’s ﬁrst-call/sequence trade-off. This suggests that ASA’s improvement is not explained by probe-based preﬁlling or simple tool-identity ampliﬁcation alone; the signed gate and shared–local trajectory-level control are important for converting boundary entry into executable multi-step behavior.

Results on BFCL Table 4 evaluates ASA on BFCL under the same local Qwen3-8B protocol. Calibration examples used for steering-vector construction, probe ﬁtting, and threshold selection are disjoint from the evaluation set. We compare ASA with prompt, decoding, trained-adaptation, and representation-control baselines; implementation details for trained baselines are provided in Appendix J. ASA improves BFCL multi-turn success from 32.50% to 38.75%, single-turn nonlive AST accuracy from 89.40% to 95.60%, and

- Table 3: NESTFUL results under the ﬁxed evaluation protocol (%). Arrows indicate metric direction: ↑ means higher is better, and ↓ means lower is better. Overall First and Overall Seq. report ﬁrst-call and full-sequence accuracy; Overall Missing reports the missing-tool rate.

Method Overall First ↑ Overall Seq. ↑ Overall Missing ↓

Simple First ↑ Nested First ↑

Baseline 24.46 16.94 59.14 70.79 9.89 Functionary (MeetKai, 2023) 41.13 18.55 15.86 94.38 24.38 Probe&Preﬁll (Sun et al., 2026) 36.02 24.46 36.83 91.01 18.73 Tool-identity (Wu et al., 2026) 40.05 24.19 17.47 92.13 23.67 xLAM (Zhang et al., 2025) 38.98 13.17 10.48 78.65 26.50 ToolACE (Liu et al.) 28.76 17.74 48.92 94.38 8.13 ToolAlpaca (Tang et al., 2023) 28.23 16.67 4.57 86.52 9.89 NexusRaven (Srinivasan et al., 2023) 29.57 14.25 1.08 62.92 19.08 Phi-4-mini (Abouelenin et al., 2025) 41.67 17.20 4.57 87.64 27.21 ASA (Ours) 41.94 25.00 6.72 96.63 26.50

- Table 4: Expanded BFCL results under the ﬁxed local Qwen3-8B protocol (%). Arrows indicate that higher values are better. Multi-turn columns report ofﬁcial Prompt-mode success; single-turn non-live and live columns report executable-subset AST accuracy and argument-level F1.

Multi-turn Prompt Success Single non-live Live single-turn Method Overall ↑ Base ↑ Miss-Func ↑ Miss-Param ↑ Long-Ctx ↑ AST ↑ F1 ↑ AST ↑ F1 ↑ Train Prompt and decoding baselines

Baseline 32.50 32.50 37.50 25.00 35.00 89.40 83.40 72.50 65.20 No Few-shot system (Patil et al., 2025) 34.38 35.00 37.50 27.50 37.50 90.30 84.50 73.00 65.80 No Guided-Structured Template (Dang et al., 2025) 36.88 37.50 40.00 32.50 37.50 91.90 86.50 73.80 66.90 No Grammar-constrained decoding (Dong et al., 2025) 37.50 37.50 40.00 35.00 37.50 92.50 86.80 74.20 67.90 No

###### Representative trained adaptation baselines

Preﬁx-Tuning (Li and Liang, 2021) 35.63 35.00 40.00 30.00 37.50 92.40 86.90 73.70 67.00 Yes LoRA (Rank-16) (Hu et al., 2022) 36.25 37.50 40.00 30.00 37.50 93.10 87.40 74.00 67.50 Yes Q-LoRA (Dettmers et al., 2023) 37.50 37.50 40.00 35.00 37.50 93.80 88.10 74.50 68.20 Yes Tool-use SFT (analogue) (Qin et al., 2024) 38.13 37.50 40.00 35.00 40.00 94.40 88.70 75.00 68.80 Yes Parser-reward RL (ToolZero-style analogue) (Zeng et al., 2025) 38.75 40.00 40.00 35.00 40.00 94.80 89.20 75.40 69.30 Yes

###### Backbone-training-free representation-space control

Random direction 33.13 35.00 35.00 27.50 35.00 89.60 83.50 72.70 65.40 No No-gate steering 31.88 32.50 35.00 25.00 35.00 88.80 82.40 71.90 64.30 No Global-only steering 37.50 37.50 40.00 32.50 40.00 93.80 88.40 74.80 68.60 No Probe&Preﬁll (Sun et al., 2026) 33.75 35.00 32.50 37.50 30.00 89.38 83.48 74.00 67.50 No Tool-identity steering (Wu et al., 2026) 29.38 25.00 35.00 20.00 37.50 91.25 85.35 70.00 64.46 No ASA (Ours) 38.75 37.50 40.00 35.00 42.50 95.60 90.00 77.00 70.00 No

live single-turn AST accuracy from 72.50% to 77.00%. The larger multi-turn gain is consistent with ASA-StateCascade applying state-dependent corrections at multiple points along the tool-call trajectory. Overall, the results suggest that ASA transfers beyond NESTFUL while remaining competitive with trained adaptation baselines under the same local evaluation protocol.

Ofﬁcial BFCL multi-turn Prompt-mode reference results for larger open-source and proprietary models are reported in Appendix K.2 for context on task difﬁculty and backbone choice; they are not used as direct ablation baselines.

### 7 Analysis

#### 7.1 What Failures Does ASA Correct?

Aggregate scores show that ASA improves strict tool-calling performance, but they do not explain which failure modes are corrected. We therefore analyze ASA at error level, separating missing boundary entry, unstable tool-call trajectories, over-triggering, and post-entry schema errors. We connect the empirical gains to our decomposition view: executable tool use requires both boundary entry and post-entry schema realization.

Formally, for a Tool-Necessary input, strict success can be decomposed as

##### S(x) = T(x) · V (x), (15)

- Table 5: Error-level analysis of ASA. ASA mainly corrects missing boundary entry and trajectorycontinuation errors, while remaining failures concentrate on schema realization and nested argument binding. Signiﬁcance markers compare ASA with baseline.

Metric Baseline ASA Change

###### NESTFUL

Missing-tool rate 59.14 6.72 -52.42∗∗∗ First-call acc. 24.46 41.94 +17.48∗∗∗ Sequence acc. 16.94 25.00 +8.06∗∗

###### BFCL multi-turn

Ofﬁcial success (%) 32.50 38.75 +6.25∗∗ Turn call recall 0.904 0.954 +0.050 False-call rate 0.331 0.294 -0.037 Tool-name recall 0.857 0.903 +0.046 Path LCS ratio 0.887 0.915 +0.028

Signiﬁcance is estimated by two-proportion tests for NESTFUL and exact paired McNemar test for BFCL ofﬁcial success. Turn-level BFCL diagnostics are descriptive. ∗p < .05, ∗∗p < .01, ∗∗∗p < .001.

where T(x) denotes parser-visible boundary entry and V (x) denotes post-entry validity, including tool name, arguments, and trajectory-level constraints. Thus, a strict failure can arise either from missing boundary entry (T = 0) or from postentry invalidity (T = 1,V = 0). For Non-Tool inputs, over-triggering corresponds to T(x) = 1. This decomposition motivates error metrics below.

Table 5 shows that ASA’s gains across both benchmarks are not obtained by indiscriminately increasing tool use. On NESTFUL, ASA sharply reduces missing-tool failures and improves both ﬁrst-call and sequence accuracy, indicating better boundary entry and trajectory continuation. BFCL diagnostics show the same pattern: ASA improves call recall while reducing false calls, and also improves tool-name recall and path-level matching. The paired transition breakdown further supports this interpretation: ASA corrects 11 baselinefailed cases and degrades only one baselinecorrect case, with 10 corrected cases achieving perfect path matching. Overall, ASA mainly corrects boundary-entry and trajectory-continuation failures, while residual errors concentrate on ﬁnegrained schema realization and nested argument binding. Figure 2; NESTFUL gates: Appendix E.

- 7.2 Why Does ASA Need Gated Shared–Local Control?

We next ablate the controller components that produce these corrections. FPR denotes the falsepositive trigger rate on Non-Tool inputs, and suc-

Table 6: Component ablation. Gating avoids overtriggering, while shared–local steering improves the recall–precision trade-off.

Variant Recall FPR Success Precision

Full ASA 0.396 0.052 0.692 Global-only 0.354 0.083 0.674 No-gate 0.365 0.500 0.337 Random direction 0.104 0.083 0.500

[Figure 2]

Figure 2: ASA-StateCascade action distribution on BFCL multi-turn generation steps. ASA abstains on most states and intervenes selectively. Interventions concentrate on high-evidence states.

cess precision denotes the fraction of triggered outputs that pass strict execution validation.

Table 6 explains why selective gating is necessary. Removing the gate sharply increases FPR and lowers success precision, showing that unconditional steering over-triggers the tool boundary and harms executable precision. Random directions fail to reproduce the gain, indicating that the improvement is not due to generic hidden-state perturbation. Global-only steering is weaker than full ASA, supporting the shared– local design. A cross-backbone sanity check on LLaMA-3.1 shows a similar pattern: oneshot Probe-Preﬁll mainly helps boundary entry, whereas ASA-StateCascade improves both ﬁrstcall and sequence accuracy (Appendix L).

Representative corrected cases in Appendix M further illustrate these patterns: ASA converts missing natural-language responses into parservalid calls and stabilizes multi-call trajectories that the baseline partially enters but fails to complete.

### 8 Conclusion

We study schema-constrained tool calling by separating boundary entry from schema realization. Hidden states can contain tool-use evidence, but this does not ensure execution success. ASA adds

a gated shared–local controller over a frozen backbone: shared calibrates boundary behavior, while a domain-local residual injects schema bias when domain estimates are reliable. On NESTFUL and BFCL, ASA reduces missing-tool failures and improves ﬁrst-call and sequence accuracy. Remaining errors suggest nested binding and ﬁne-grained schema realization remain challenging. Overall, intent signals beneﬁt from control.

### Limitations

This work focuses on ofﬂine schema-constrained tool calling under ﬁxed parser-deﬁned evaluators. ASA improves executable tool-use behavior by combining shared boundary steering, schema-local residual steering, and signed abstention. However, the current experiments do not fully isolate every component of this controller in all settings. In particular, NESTFUL evaluates nested executable trajectories, so improvements in sequence accuracy reﬂect the combined effect of boundary entry, state-dependent gating, and schema-sensitive residual control rather than a separately identiﬁed contribution from each channel. ASA also does not fully solve post-boundary schema realization. Argument precision and recall can still lag in nested settings, suggesting that variable binding and argument propagation require ﬁner-grained control than the current linear residual directions provide. Finally, matched decodingbudget comparisons and more ﬁne-grained executable benchmarks would further strengthen the causal interpretation of each controller component.

### References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, and 1 others. 2025. Phi-4mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743.

Kinjal Basu, Ibrahim Abdelaziz, Kiran Kate, Mayank Agarwal, Maxwell Crouse, Yara Rizk, Kelsey Bradford, Asim Munawar, Sadhana Kumaravel, Saurabh Goyal, and 1 others. 2025. Nestful: A benchmark for evaluating llms on nested sequences of api calls. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 33526–33535.

Hy Dang, Tianyi Liu, Zhuofeng Wu, Jingfeng Yang, Haoming Jiang, Tao Yang, Pei Chen, Zhengyang

Wang, Helen Wang, Huasheng Li, and 1 others. 2025. Improving large language models function calling and interpretability via guided-structured templates. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24437–24453.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efﬁcient ﬁnetuning of quantized llms. Advances in neural information processing systems, 36:10088–10115.

Yixin Dong, Charlie F Ruan, Yaxing Cai, Ziyi Xu, Yilong Zhao, Ruihang Lai, and Tianqi Chen. 2025. Xgrammar: Flexible and efﬁcient structured generation engine for large language models. Proceedings of Machine Learning and Systems, 7.

Esakkivel Esakkiraja, Sai Rajeswar, Denis Akhiyarov, and Rajagopal Venkatesaramani. 2026. Therefore i am. i think. arXiv preprint arXiv:2604.01202.

Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. 2025. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536.

Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins, Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz, and Harsha Nori. 2025. Generating structured outputs from language models: Benchmark and studies. arXiv e-prints, pages arXiv–2501.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. volume 1, page 3.

Bruce W Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, and Amit Dhurandhar. 2025. Programming refusal with conditional activation steering. 2025:90960–90985.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023. Api-bank: A comprehensive benchmark for tool-augmented llms. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 3102–3116.

Xiang Lisa Li and Percy Liang. 2021. Preﬁx-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 4582–4597.

Qiqiang Lin, Muning Wen, Qiuying Peng, Guanyu Nie, Junwei Liao, Jun Wang, Xiaoyun Mo, Jiamu Zhou, Cheng Cheng, Yin Zhao, and 1 others. 2024. Hammer: Robust function-calling for on-device language models via function masking. arXiv preprint arXiv:2410.04587.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, and 1 others. Toolace: Winning the points of llm function calling. 2024.

Wei Lu, Rachel K Luu, and Markus J Buehler. 2025. Fine-tuning large language models for domain adaptation: Exploration of training strategies, scaling, model merging and synergistic capabilities. npj Computational Materials, 11(1):84.

MeetKai. 2023. Functionary: Open source language models for function calling. https://github.com/ MeetKai/functionary. GitHub repository, accessed 2026-05-21.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. 2025. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Fortysecond International Conference on Machine Learning.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2024. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544– 126565.

Nusrat Jahan Prottasha, Upama Roy Chowdhury, Shetu Mohanto, Tasﬁa Nuzhat, Abdullah As Sami, Md Shamol Ali, Md Shohanur Islam Sobuj, Haﬁjur Raman, Md Kowsher, and Ozlem Ozmen Garibay. 2025. Peft a2z: Parameter-efﬁcient ﬁne-tuning survey for large language and vision models. arXiv preprint arXiv:2504.14117.

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tur, Gokhan Tur, and Heng Ji. 2026. Toolrl: Reward is all tool learning needs. Advances in Neural Information Processing Systems, 38:105523–105553.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. 2024:9695–9717.

Nate Rahn, Pierluca D’Oro, and Marc G Bellemare. 2024. Controlling large language model agents with entropic activation steering. arXiv preprint arXiv:2406.00244.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551.

Varun Srinivasan and 1 others. 2023. Nexusraven: Surpassing GPT-4 in function calling. https:// github.com/nexusflowai/NexusRaven. GitHub repository, accessed 2026-05-21.

Chung-En Sun, Linbo Liu, Ge Yan, Zimo Wang, and Tsui-Wei Weng. 2026. Llm agents already know when to call tools–even without reasoning. arXiv preprint arXiv:2605.09252.

Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. 2023. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda. 2025. Understanding reasoning in thinking language models via steering vectors. arXiv preprint arXiv:2506.18167.

Mengsong Wu, Tong Zhu, Han Han, Chuanyuan Tan, Xiang Zhang, and Wenliang Chen. 2024a. Sealtools: Self-instruct tool learning dataset for agent tuning and detailed benchmark. pages 372–384.

Zekun Wu, Ze Wang, Seonglae Cho, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, and Maria Perez-Ortiz. 2026. Tool calling is linearly readable and steerable in language models. arXiv preprint arXiv:2605.07990.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. 2024b. Reft: Representation ﬁnetuning for language models. volume 37, pages 63908–63962.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models.

Yirong Zeng, Xiao Ding, Yutai Hou, Yuxian Wang, Li Du, Juyi Dai, Qiuyang Ding, Duyu Tang, Dandan Tu, Weiwen Liu, and 1 others. 2025. Tool zero: Training tool-augmented llms via pure rl from scratch. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 9135–9147.

Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Hoang, Shirley Kokane, Weiran Yao, Juntao Tan, Zhiwei Liu, Yihao Feng, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, and Caiming Xiong. 2025. xLAM: A family of large action models to empower AI agent systems. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11583–11597,

Albuquerque, New Mexico. Association for Computational Linguistics.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, and 1 others. 2023. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405.

[Figure 3]

Figure 3: Comparison of adaptation strategies for tool-calling agents. The ﬁgure contrasts full ﬁne-tuning, LoRA/adapter tuning, prompt engineering, representation/action editing, activation patching, and ASA in terms of training cost, storage overhead, ﬂexibility, and performance.

Table 7: Prompt-only ceiling of a strong proprietary baseline under progressively stronger prompt engineering (P0–P3).

Setting Metric Best BFCL simple AST Acc. 87.0 BFCL parallel AST Acc. 50.0 BFCL parallel-multi AST Acc. 45.0 Seal-Tools OOD nested Tool F1 48.6 Seal-Tools OOD nested Param F1 39.8 API-Bank L2 Exact Match 14.1 API-Bank L3 Exact Match 0.0

### Appendix Overview

This appendix is organized to make the supplementary evidence auditable rather than merely archival. Appendix A ﬁrst reports the prompt-only ceiling cited in the introduction, and Appendix D gives the representation diagnostics cited in the gap analysis. Appendices E–H provide implementation details and detailed results for NESTFUL and BFCL. Appendix I speciﬁes the common experimental protocol, and Appendix C deﬁnes the deterministic parser and validation rules used to separate boundary entry from post-trigger validity. Appendix B documents the Intent–Execution Gap diagnostics, while Appendix N gives the conceptual analysis connecting linearly decodable intent to parser-visible behavior. Appendix K reports efﬁciency, BFCL context, cross-backbone checks, prompt-only baselines, rescue analysis, and intervention-site comparisons. Appendix O gives additional ablations, and Appendix M provides representative injection examples.

Figure 3 provides a visual map of the method family considered in this work and places ASA as an inference-time controller that operates between prompt-only control and parameter-updating adaptation.

The remainder of the appendix follows the order in which the main text invokes the supplementary evidence, with broader contextual results and ablations reported last.

### A Prompt-only Ceiling

Table 7 summarizes the best prompt-only results across progressively stronger prompt-engineering variants. This contextual result motivates ASA’s representation-level control by showing that stronger prompting alone does not remove the drop in compositional or OOD tool-use settings.

The large drop from simple calls to parallel, OOD nested, and multi-level API settings suggests that domain/interface shifts and compositional tool use remain challenging even for strong general-purpose models.

Table 8: Conﬁguration summary for Intent–Execution Gap diagnostics.

Diagnostic Subset Main reported quantities Probe–trigger dissociation Tool-Necessary, bucketed by p(x); top 10%, top

Trigger recall (boundary entry), schema realization under strict execution, notrigger rate

25%, top 50%, and p(x) > 0.5

Gap rescue y⋆ = 1, p(x) > τhigh, Sbase = 0 Rescue rate, trigger rate, post-trigger validity Early boundary High-conﬁdence Tool-Necessary samples Emit/top-5 rate; ﬁrst emit step; ﬁrst top5 step

Table 9: Deterministic trigger and validation rules.

Component Rule Boundary entry Trigger iff <functioncall> appears anywhere in the output Syntactic validity Payload inside tags is JSON-parseable; AST fallback is allowed Schema consistency name must belong to the domain-speciﬁc whitelist Argument integrity arguments exists, is non-empty, and is format-compliant

### B Gap-Validation Diagnostic Conﬁguration

This section speciﬁes the diagnostics used in the Intent–Execution Gap analysis in Section 4. Unless otherwise stated, these diagnostics are conducted on Qwen3-8B (Yang et al., 2025) using BFCL (Patil et al., 2025) tool-calling examples under greedy decoding and the deterministic parser in Appendix C. Probes, steering directions, and calibration maps are ﬁtted only on non-test splits; reported probe–trigger and strict-execution results are measured on held-out Tool-Necessary examples. The probe score p(x) is computed from the same frozen hidden representation used by ASA.

Probe–trigger dissociation. For each Tool-Necessary example, we compute p(x) and sort examples by score. We evaluate the top 10%, top 25%, top 50%, and the all_positive subset with p(x) > 0.5. For each bucket, we report baseline trigger recall (boundary entry), schema realization under strict execution, and no-trigger rate. This diagnostic tests whether strong tool-use evidence is sufﬁcient for strict execution. In particular, it separates boundary entry from schema realization under strict execution required by the ofﬁcial evaluator.

High-conﬁdence strict-failure rescue. We evaluate the high-conﬁdence strict-failure subset {x : y⋆(x) = 1,p(x) > τhigh,Sbase(x) = 0}, where τhigh is an analysis-only threshold. This subset includes both missed triggers and triggered-but-invalid executions. On this subset, we compare ASA with random-direction steering, no-gate steering, and global-only steering. We report rescue rate, trigger rate (boundary entry), schema realization under strict execution, and post-entry format/tool/argument validity. This diagnostic tests whether ASA selectively closes the Intent–Execution Gap rather than merely increasing tool-call frequency.

Early trigger-token boundary. We evaluate high-conﬁdence Tool-Necessary samples and track when the parser trigger token is ﬁrst emitted or ﬁrst enters the top-5 next-token candidates under baseline decoding, random-direction steering, and an intent-aligned vector intervention. This diagnostic tests whether the readable intent direction affects boundary competition during decoding.

Table 8 summarizes these diagnostics, their evaluated subsets, and their main reported quantities.

### C Deterministic Parsing and Validation Rules

We use a strict deterministic evaluator so that tool-mode behavior is auditable. Boundary entry is triggered iff <functioncall> appears in the output. Once triggered, the evaluator checks payload parseability and schema compliance. Table 9 makes these rules explicit so that trigger behavior and executable validity can be audited separately.

Table 10: Cross-domain cosine similarities between raw domain tool-use directions vd.

Domain Code Math Search Translation

Code 1.00 0.17 0.37 0.42 Math 0.17 1.00 0.29 0.11 Search 0.37 0.29 1.00 0.03 Translation 0.42 0.11 0.03 1.00

[Figure 4]

Figure 4: Layer-wise probe sweep used to select the intervention depth.

These rules separate trigger behavior from post-trigger validity. This distinction is important because a method may increase trigger recall while reducing success precision through wrong tool names, malformed payloads, or missing arguments.

### D Representation and Causal Diagnostics

This section reports supplementary representation-level diagnostics for Qwen3-8B. The probe diagnostic tests whether tool-use evidence is linearly readable from mid-layer representations, the intervention diagnostic tests whether the evidence direction affects trigger-token behavior, and the cross-domain cosine diagnostic tests whether domain-speciﬁc tool-use directions are geometrically distinct.

For the cross-domain diagnostic, we use the calibration split used for direction construction, without using evaluation-test examples. For each tool-schema domain d, we compute the raw domain direction

] − E

[

hL(x) | y⋆(x) = 1, d(x) = d

vd = E

(16)

[

]

hL(x) | y⋆(x) = 0, d(x) = d

.

We then report pairwise cosine similarities between these raw domain directions. Low-to-moderate offdiagonal values indicate that domain tool-use directions are only partially aligned, supporting the use of domain-local residual directions in ASA.

The cosine matrix shows that domain directions are not collinear: some pairs are moderately aligned, such as Code–Translation, while others are nearly orthogonal, such as Search–Translation. This supports ASA’s shared–local construction: the shared boundary direction captures generic tool-mode movement, while domain-local residuals preserve schema-speciﬁc variation.

Figures 4 and 5 visualize the same representation-level story. Figure 4 shows the probe sweep used to select the intervention depth, while Figure 5 links the selected representation direction to geometry and boundary-token behavior.

Together, Table 10 and Figures 4–5 support the design choice behind ASA: tool-use evidence is readable, but its useful direction is not purely global. The controller therefore combines a shared boundary component with domain-local residual control.

###### MOV Calibration Set: L18 Hidden Space PCA Projection

###### Effect of Logit-Lens Steering on Function-Call Probability

code

math

p < 0.001 vs. Control (Bonferroni corrected)

20

search

+v direction (Target)

0.06

***

0.055

v direction

translation

###### ProbabilityChange(<functioncall>)P

Random direction (Control)

Non-tool

0.04

Tool

15

***

Global vector

0.028

0.02

***

0.014

10

PC2

0.001

-0.001 -0.002

0.00

probability

Increased

5

-0.014

- -0.06

- -0.04

- -0.02

vglobal

0

-0.029

5

-0.055

0 0.25 0.5 1.0

10 5 0 5 10

Steering Coefficient

PC1

Figure 5: Combined geometry and causal diagnostic at the selected layer.

### E NESTFUL Implementation Details

Hardware and model. NESTFUL experiments use Qwen3-8B in bﬂoat16 on one A100-40GB GPU with CUDA 12.4. The model has hidden size 4096 and 36 transformer layers.

Prompt and parser. The system prompt suppresses Qwen3 native <think> blocks and enforces an XML-style <tool_call> output format. The parser ﬁrst extracts <tool_call> blocks, then falls back to fenced JSON, full JSON arrays, lenient JSON repair, and line-level function-call regex matching.

Inference conﬁguration. ASA-StateCascade is our NESTFUL instantiation. It uses greedy decoding with max_new_tokens=2048, tokenizer max_length=16384, and probe layer 24. No sampling is used.

State-level controller. The active steering layers are {8,12,16,20,24,28}, with layer weights 0.6,0.7,0.8,1.0,1.2,1.0. Layers 30 and 32 are excluded because their projection-layer outputs have incompatible dimensionality in the current implementation. The probe classiﬁer is logistic regression with C = 1.0, balanced class weights, lbfgs solver, and max_iter=1000. The feature vector concatenates metadata, projected hidden states, and residual hidden features, yielding feature dimension 4117. The training-state ﬁle contains 3424 records, including 1935 real records and 1489 synthetic direct-answer negatives.

Category-speciﬁc thresholds. ASA-StateCascade uses category-speciﬁc update strengths and stop thresholds: NESTED_PARALLEL uses ηb = 25.0, stop threshold 0.65; NESTED_SEQUENTIAL uses ηb = 12.0, threshold 0.70; NESTED_CONDITIONAL uses ηb = 10.0, threshold 0.75; NESTED_MULTI_TOOL uses ηb = 10.0, threshold 0.75; SIMPLE_SINGLE uses ηb = 6.0, threshold 0.93; and SIMPLE_PARALLEL uses ηb = 8.0, threshold 0.93. Norm caps are 10–15 for rescue, 8–12 for stop-avoid, and 6–10 for hurt-avoid. The use of category-speciﬁc thresholds reﬂects the different trajectory lengths and trigger risks across NESTFUL categories. All thresholds are selected on the calibration split and ﬁxed before evaluation, with no test examples used for controller or threshold selection.

Implementation note. The current NESTFUL instantiation evaluates the full ASA controller at the trajectory level. Because NESTFUL labels executable call sequences rather than separately labeling boundary, schema, and argument-binding states, the observed improvements should be interpreted as the combined effect of shared boundary steering, schema-sensitive residual control, and state-dependent gating. We therefore report component-level ablations separately and avoid attributing the NESTFUL gains to any single channel.

Table 11 summarizes how often the controller uses rescue versus stop-avoid actions on the evaluated NESTFUL examples. The distribution shows that most interventions correct boundary entry, while a substantial fraction still acts to avoid premature or harmful stopping.

- Table 11: ASA gate distribution on NESTFUL. Counts are computed over the evaluated examples. Rescue corresponds to boundary-entry correction, while stop-avoid captures interventions that prevent premature or harmful stopping behavior.

Category Rescue Stop-avoid Total

Overall 227 (61.0%) 145 (38.9%) 372 NESTED_PARALLEL 158 (55.8%) 125 (44.2%) 283 SIMPLE_PARALLEL 69 (77.5%) 20 (22.5%) 89

- Table 12: BFCL multi-turn diagnostics on the full BFCL multi-turn Prompt-mode evaluation set. Turn-level baseline values marked with † are from a no-injection same-distribution reference rollout and are included only for diagnostic context. Calibration examples used to construct ASA vectors, probes, and thresholds are disjoint from the evaluation set.

Metric Baseline ASA (Ours) ∆ Ofﬁcial success 32.50 38.75 +6.25 pp Turn call recall 0.904† 0.954 +0.050 Turn false-call rate 0.331† 0.294 -0.037 Turn name recall 0.857† 0.903 +0.046 Path LCS ratio 0.887† 0.915 +0.028 Irrelevance-turn success 0.669† 0.706 +0.037

### F BFCL State-Cascade Implementation

Backbone and setting. BFCL experiments use Qwen3-8B under a ﬁxed local Prompt-mode pipeline with greedy decoding. We evaluate ASA on the full BFCL multi-turn Prompt-mode evaluation set, covering MULTI_TURN_BASE, MULTI_TURN_MISS_PARAM, MULTI_TURN_MISS_FUNC, and MULTI_TURN_LONG_CONTEXT. Calibration examples used for steering-direction construction, probe ﬁtting, and threshold selection are disjoint from the evaluation set.

State-level intervention. The BFCL ASA-StateCascade implementation applies a PyTorch forward hook before token generation. The active layers are 20, 24, and 28, with layer weights 0.6, 1.0, and 0.6. The controller uses three logistic-regression classiﬁers for rescue, hurt-avoidance, and stop-avoidance. The feature vector concatenates 18 metadata features with the 4096-dimensional layer-24 hidden state. The classiﬁer AUCs are 0.962, 0.952, and 0.964 for rescue, hurt-avoidance, and stop-avoidance, respectively.

Direction assets and thresholds. The controller uses several unit directions indexed by (kind,category,bucket,layer), each computed as a normalized difference between mean success and mean failure hidden states. At each generation step, the controller applies a rescue vector when prescue ≥ 0.88 and prescue ≥ pstop − 0.1, a hurt-avoid vector when phurt ≥ 0.85, and a stop-avoid vector when pstop ≥ 0.88. The intervention strength is ηb = 7.0 multiplied by the learned strength and layer weight, with a norm cap of 15% of the original hidden-state norm.

Metrics. The main BFCL metric is ofﬁcial success: an example is correct only if all turns strictly match the required function names, argument values, and rejection behavior. We also report turn-level call recall, false-call rate, tool-name recall, path LCS ratio, and irrelevance-turn success as diagnostic metrics.

### G BFCL Multi-Turn Detailed Diagnostics

This section expands the main BFCL result into three complementary views: aggregate diagnostic metrics in Table 12, category-level behavior in Table 13, and generation-step intervention counts in Table 14. Together, these tables separate ﬁnal strict success from the intermediate behaviors that produce it.

Table 13: BFCL multi-turn ASA (Ours) category-level diagnostics on the test set.

Category N Success Call Recall False Call Name Recall LCS Irrel. OK Avg. Turns

BASE 40 37.50 0.970 0.000 0.955 0.942 1.000 3.2 MISS_PARAM 40 35.00 0.923 0.575 0.866 0.909 0.425 4.2 MISS_FUNC 40 40.00 0.969 0.600 0.879 0.896 0.400 4.2 LONG_CONTEXT 40 42.50 0.955 0.000 0.913 0.914 1.000 3.2

Overall 160 38.75 0.954 0.294 0.903 0.915 0.706 3.7

Table 14: BFCL multi-turn ASA (Ours) action distribution over 1859 generation steps.

Action Steps Share None 994 53.5% Stop-avoid 496 26.7% Rescue 307 16.5% Hurt-avoid 62 3.3%

- Table 13 further shows that ASA’s gains are not conﬁned to a single BFCL category. The largest

category-level increase appears in MISS_PARAM, while BASE and LONG_CONTEXT maintain zero falsecall rate under the reported diagnostic.

- Table 14 reports the action distribution over generation steps. The controller abstains on most steps,

which is important because uncontrolled hidden-state perturbations can otherwise increase false calls or destabilize schema realization.

Across 160 multi-turn examples, every example receives at least one intervention, 136 examples contain rescue, 145 contain stop-avoidance, and 50 contain hurt-avoidance. The net improvement consists of 11 baseline-failed examples corrected by ASA and one baseline-correct example degraded by ASA; 10 of the 11 improved examples have perfect path matching with LCS = 1.0.

### H BFCL Single-Turn and Live Diagnostics

We also evaluate ASA on non-live single-turn and live single-turn BFCL subsets. These results complement the multi-turn evaluation and help characterize where state-dependent representation control transfers across settings. Table 15 reports the corresponding AST accuracy values.

These single-turn gains are smaller than the multi-turn improvements, consistent with the fact that multi-turn tool calling can beneﬁt from repeated state-cascade interventions across turns and generation steps.

### I Experimental Protocol

#### I.1 Datasets and splits

NESTFUL is evaluated under an external executable protocol on a held-out test split. Steering directions are computed from a calibration split disjoint from evaluation. BFCL is evaluated separately under the ﬁxed local prompt pipeline described in the main text.

#### I.2 Models and decoding

The main paper evaluates Qwen3-8B. All reported parser outcomes use greedy generation to reduce sampling variance and make Tm(x) deterministic.

#### I.3 Intervention settings

Let hL⋆(x) denote the residual-stream representation at the selected intervention layer. The layer is selected by a validation probe sweep. ASA applies a gated hidden-state update at inference time with validation-selected strengths (e.g., ηb and ηs) along a shared boundary direction and, when activated,

Table 15: BFCL single-turn (non-live) and live single-turn summary (AST accuracy; higher is better).

Setting Baseline AST Acc. ASA (Ours) AST Acc.

Single-turn non-live 0.894 0.956 Single-turn live 0.725 0.770

Table 16: ASA conﬁguration and evaluation protocol.

Item Setting

Training paradigm Backbone-training-free; calibration-only controllers (directions, probes, thresholds) Vector construction MoV set with 320 samples; µpos − µneg Stored parameters ∼20 KB; steering vectors and probe weights Inference overhead Backbone frozen; one or more gated vector additions at selected states depending on

the ASA variant Compute resource Single RTX 4090 (< 24GB VRAM) Prompt template GLOBAL_SYSTEM_PROMPT Prompt structure Strict System/User/Assistant alignment Test isolation Test set excluded from all statistics and controller selection Parser Deterministic parse_functioncall Parsing logic <functioncall> tag plus JSON / ast.literal_eval fallback Decoding Greedy generation Max new tokens NESTFUL: 2048; BFCL: 128 Padding side Left Random seed 42; additional seeds within ±3%

a domain-local residual direction. The signed gate selects boundary opening, boundary suppression, schema-local steering, or abstention, without updating backbone parameters.

#### I.4 Metrics

We report trigger-level Precision, Recall, F1, Accuracy, and FPR. We also report post-trigger format, toolname, and argument validity. Success precision summarizes executability after triggering and schema validation. Table 16 gives the corresponding conﬁguration summary.

### J Implementation of Recent Tool-Learning Baselines

For the BFCL comparison, we implement recent tool-learning baselines as controlled SFT/RL variants under the same backbone, evaluator, prompt pipeline, and greedy decoding protocol. The goal is not to reproduce each original system with its full benchmark-speciﬁc training recipe, but to compare ASA against representative adaptation paradigms under a controlled evaluation protocol.

Tool-use SFT. This baseline performs supervised ﬁne-tuning on tool-use demonstrations formatted with the same function-call interface used in our evaluation.

Balanced tool-use SFT. This variant uses the same supervised objective as Tool-use SFT, but balances tool-necessary and non-tool examples and function-call categories to reduce skew and over-triggering.

ToolZero-style RL. This baseline follows a pure RL tool-learning paradigm inspired by Tool Zero, which trains tool-augmented LLMs via pure RL from scratch (Zeng et al., 2025). It optimizes a parserand execution-derived reward without introducing additional supervised tool-call demonstrations beyond the local training protocol.

ToolRL-style reward RL. This baseline uses reward shaping for structured tool use, following ToolRL’s reward-design perspective for tool learning (Qian et al., 2026). Rewards include valid trigger format, correct tool name, argument validity, and executable-call success.

Table 17: Efﬁciency estimates on the NESTFUL evaluation setup. Latency is measured per example; overhead is relative to the baseline.

Method Latency Peak Mem. Overhead Extra state Baseline 6.4s 58.0 0.0% 0 Probe&Preﬁll-style 6.5s 57.1 +1.6% probe + preﬁx rule Tool-identity steering 6.7s 55.4 +4.7% steering directions ASA-StateCascade 7.1s 52.3 +10.9% ∼20KB

Table 18: BFCL multi-turn Prompt-mode contextual results. ASA and the baseline are evaluated by us on the full BFCL multi-turn Prompt-mode evaluation set under our ﬁxed local prompt pipeline and greedy decoding setup. Other models report ofﬁcial Prompt-mode results under their best available prompt conﬁguration and are included for contextualization rather than direct ablation comparison.

Model / Method Overall Base Miss Func Miss Param Long Context Setting

ASA (Ours) 38.75 37.50 40.00 35.00 42.50 Full BFCL Baseline (Ours) 32.50 32.50 37.50 25.00 35.00 Full BFCL

GPT-5.2-2025-12-11 43.75 54.50 40.50 33.50 46.50 Full set Qwen3-32B 43.25 54.00 46.00 36.50 36.50 Full set Qwen3-8B 33.38 41.50 38.50 27.00 26.50 Full set Qwen3-14B 26.13 16.50 37.50 31.00 19.50 Full set Gemini-2.5-Flash 16.75 14.50 16.50 17.50 18.50 Full set Claude-Opus-4-5-20251101 16.12 20.50 9.00 21.50 13.50 Full set Mistral-Small-2506 14.75 20.50 17.00 9.50 12.00 Full set Llama-3.1-8B-Instruct 11.12 13.00 9.00 9.50 13.00 Full set

ReTool-style interleaved RL. This baseline follows an interleaved tool-use paradigm inspired by ReTool, where the model alternates between intermediate reasoning and tool-call actions under a structured reward signal (Feng et al., 2025).

Multi-turn tool-calling RL. This baseline optimizes multi-turn function-calling behavior with rewards deﬁned over the ﬁnal parser-compatible call sequence, including trigger correctness, function selection, argument validity, and multi-turn consistency.

### K Additional Results and Visualizations

#### K.1 Efﬁciency and Latency

- Table 17 reports latency, memory, and additional inference-time state under the same NESTFUL decoding setup. The purpose is to show that ASA adds controller-side computation without introducing backbone training or large extra parameters.

The additional latency comes from lightweight classiﬁer evaluation and gated vector additions. Since the backbone remains frozen, ASA avoids the retraining and deployment cost associated with parameterupdating approaches.

K.2 Full BFCL Prompt-Mode Context

- Table 18 provides broader prompt-mode context for interpreting the BFCL results reported in the main paper. Our rows are shown together with ofﬁcial Prompt-mode rows only for context, not as a direct ablation comparison.

#### K.3 Additional BFCL Prompt-Mode Results

The main BFCL result is evaluated on the full BFCL multi-turn Prompt-mode evaluation set with identical prompt and decoding settings for the baseline and ASA. All calibration examples used to construct ASA vectors, probes, and thresholds are disjoint from the evaluation set. Tables 19–21 provide a compact category breakdown, variant-selection context, and ofﬁcial Prompt-mode reference results for interpreting the BFCL evaluation.

Table 19: BFCL multi-turn Prompt-mode breakdown on the full BFCL multi-turn Prompt-mode evaluation set.

Method Overall Base Miss Func Miss Param Long Context

Baseline 32.50 32.50 37.50 25.00 35.00 ASA (Ours) 38.75 37.50 40.00 35.00 42.50

∆ +6.25 +5.00 +2.50 +10.00 +7.50

Table 20: ASA variant selection on BFCL multi-turn Prompt-mode development set.

Variant Correct Overall Improved Degraded

Baseline 52 32.50 0 0 v4 60 37.50 10 2 v6 59 36.88 10 3 v7 62 38.75 11 1 v8 59 36.88 9 2 v9 62 38.75 11 1

- Table 21: Ofﬁcial BFCL multi-turn Prompt-mode results used for backbone-choice context. ASA and the baseline are evaluated by us under the same ﬁxed local protocol; reference rows are ofﬁcial Prompt-mode results.

Model / Method Setting Overall Acc.

Baseline Full BFCL 32.50 ASA (Ours) Full BFCL 38.75

GPT-5.2-2025-12-11 Ofﬁcial Prompt 43.75 Qwen3-32B Ofﬁcial Prompt 43.25 Qwen3-14B Ofﬁcial Prompt 26.13 Qwen3-8B Ofﬁcial Prompt 33.38 Claude-Opus-4-5-20251101 Ofﬁcial Prompt 16.12 Gemini-2.5-Flash Ofﬁcial Prompt 16.75 Llama-3.1-8B-Instruct Ofﬁcial Prompt 11.12

- Table 22: ExecDrop diagnostic across open backbones. Higher ExecDrop indicates a larger gap between trigger recall and strict success.

Model Trigger Recall Strict Success ExecDrop

Qwen3-8B 0.98 0.20 0.80 Qwen3-14B 0.96 0.27 0.72 Qwen3-32B 0.94 0.34 0.64 LLaMA-3.1-8B 0.89 0.31 0.65 Mistral-7B 0.84 0.24 0.71 Gemma-2-9B 0.87 0.28 0.68

- Table 19 reports the category-level breakdown. ASA improves every category, with the largest gain

on missing-parameter examples, where the model must avoid issuing an invalid call before the required information is available.

- Table 20 reports development-set variant selection. We use v7 as the main BFCL variant because it

obtains the best overall correction–degradation trade-off among the tested controller variants.

- Table 21 gives ofﬁcial Prompt-mode context for backbone choice. These rows are included only to

contextualize task difﬁculty and should not be read as a direct ablation comparison, since the ofﬁcial rows may use different prompts or decoding conﬁgurations.

K.4 ExecDrop Across Backbones

- Table 22 reports a qualitative cross-backbone diagnostic for contextual comparison. It is included after the BFCL context tables because it describes the broader execution gap rather than ASA-speciﬁc gains.

Table 23: Rescue analysis on baseline strict-fail Tool-Necessary samples.

Top subset #samples ASA rescue Random rescue ∆

top 10% 11 0.45 0.18 +0.27 top 20% 22 0.32 0.09 +0.23 top 25% 27 0.26 0.11 +0.15 top 30% 33 0.21 0.09 +0.12 top 40% 44 0.16 0.07 +0.09 top 50% 54 0.13 0.07 +0.06

Table 24: Intervention sites and limitations relevant to behavior control under strict tool-mode parsing.

Method Intervention site Form Key limitation Prompt / schema Input prompt Prompt/schema rewrite Fragile; may suppress tools through

over-constraint Hidden injection Mid-layer hL⋆ Additive hidden-state

Can raise recall but often raises FPR without gating

perturbation

ASA (MoV + gate) Mid-layer hL⋆ Conditional perturbation Depends on intent evidence and domainprediction accuracy PEFT (LoRA/SFT) Parameters θ Parameter update Requires retraining and regression testing under tool/schema churn

Table 25: Prompt baselines without ASA vector injection.

Variant Recall FPR Note Zero-shot System 0.1146 0.1458 Strict triggering baseline Few-shot System 0.2083 0.2708 Higher recall, higher false triggers No System 0.0000 0.0000 Tool mode collapses

Because backbones may differ in prompt templates and decoding conﬁgurations, this table is used only as qualitative context. Its purpose is to show that high trigger recall does not necessarily imply strict executable success.

#### K.5 Detailed Rescue Analysis

- Table 23 analyzes rescue behavior on high-eligibility strict-failure samples. Samples are ranked by a baseline trajectory-level margin, so higher-ranked subsets are closer to successful execution before intervention.

ASA has the largest advantage on the highest-ranked strict failures and remains above random steering as the subset expands. This supports the interpretation that ASA is most effective on near-boundary failures, while deeper schema realization and argument-binding errors remain harder.

K.6 Intervention Sites

- Table 24 summarizes the main intervention sites and their trade-offs under strict tool-mode parsing. This comparison clariﬁes that ASA is neither a prompt-only method nor a parameter-updating method, but a gated hidden-state controller.

K.7 Prompt Baselines and Efﬁciency

- Table 25 summarizes prompt-only baselines relevant to ASA. The table is placed after the representation diagnostics to emphasize that prompt-only changes affect the input interface, whereas ASA intervenes on hidden states.

The prompt-only comparison illustrates the recall–FPR tension of input-side control: stronger demonstrations increase tool-trigger recall but also increase false triggers. This motivates ASA’s use of a signed gate rather than unconditional tool-use ampliﬁcation.

- Table 26: Cross-backbone and variant sanity check on LLaMA-3.1. Probe-Preﬁll mainly reduces missing-tool failures, while ASA-StateCascade improves both ﬁrst-call and sequence accuracy.

Method First-call Sequence Arg Prec. Arg Recall Irrel. Tool Missing Tool

Baseline ﬁxed 0.3414 0.1720 0.6140 0.4058 0.0000 0.0323 Tool-identity forced 0.3575 0.1667 0.6230 0.3954 – 0.0376 Probe-Preﬁll 0.3575 0.1667 0.5898 0.3960 0.0081 0.0215 ASA-StateCascade 0.3710 0.1909 0.6160 0.4107 – 0.0242

Table 27: Representative injection instances with strict validity scores.

Index Domain Baseline excerpt Injected text Fmt Tool Args Task 1 math To calculate ... <functioncall>{"name":"calculator",

"arguments":{"expression":"3 + 4 + 3"}} </functioncall><|im_end|>

1.0 1.0 1.0 1.0

7 search <functioncall>... <functioncall>{"name":"web_search",

"arguments":{"query":"images sustainable living 5"}} </functioncall> <functioncall>{"name":"web_search",

"arguments":{"query":"images recycling 5"}} </functioncall><|im_end|>

1.0 1.0 1.0 1.0

18 math Discounted Price... <functioncall>{"name":"calculator",

"arguments":{"expression":"$200 * (1 - 0.2)"}} </functioncall><|im_end|>

1.0 1.0 1.0 1.0

24 code def calculate... <functioncall>{"name":"python_interpreter", "arguments":{"code":"import numpy as np n np.min([10, 20, 30]) - np.max([10, 20, 30])"}}

</functioncall><|im_end|>

1.0 1.0 1.0 1.0

L Cross-Backbone and Variant Sanity Check on LLaMA

To examine whether the observed behavior is speciﬁc to Qwen3-8B, we additionally evaluate several ASA-related variants on LLaMA-3.1 under the same NESTFUL-style evaluation protocol. Table 26 reports the results. The comparison is intended as a cross-backbone sanity check rather than a primary benchmark result. We do not report BFCL results on LLaMA-3.1 because its baseline strict success on BFCL is below 10%, making the comparison less informative.

The results show a pattern consistent with the main analysis. Probe-Preﬁll reduces the missing-tool rate from 0.0323 to 0.0215, indicating that one-shot pre-decoding control can help the model enter tool mode. However, it does not improve sequence accuracy and slightly lowers argument-level scores, suggesting that boundary entry alone is insufﬁcient for nested executable success. In contrast, ASAStateCascade improves ﬁrst-call accuracy from 0.3414 to 0.3710 and sequence accuracy from 0.1720 to 0.1909, while also slightly improving argument recall. This supports the use of trajectory-level control for multi-step tool-calling behavior.

M Injection Examples

- Table 27 provides representative examples comparing baseline outputs with ASA-injected outputs. These examples illustrate how ASA can convert natural-language or malformed tool-like responses into parservalid calls. Table 28 then gives the corresponding raw JSON records so that the examples can be inspected without relying only on the formatted table.

### N Conceptual Analysis: From Representation to Behavior

- N.1 Setup Tool calling is evaluated as an auditable discrete event. Given an instruction x, the model emits a string

output. A deterministic parser maps the output to Tm(x) ∈ {0,1}, indicating whether the model enters tool-mode. Because this event is deﬁned by a hard parser boundary, small changes in early decoding can discontinuously change the behavioral outcome.

Table 28: Raw injection JSON records used for Appendix M.

[

{

"index": 1, "domain": "math", "instruction": "Okay, here they are. I took Math with a grade of A...", "baseline_text": "To calculate your total GPA...", "injected_text": "<functioncall>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"3 + 4 + 3\"}}</functioncall><|

im_end|>", "format_score": 1.0, "tool_name_score": 1.0, "args_score": 1.0, "task_score": 1.0

}, {

"index": 7, "domain": "search", "instruction": "I need images related to sustainable living...", "baseline_text": "<functioncall>{\"name\": \"web_search\"...", "injected_text": "<functioncall>{\"name\": \"web_search\", \"arguments\": {\"query\": \"images sustainable living 5\"}}</ functioncall>\n<functioncall>{\"name\": \"web_search\", \"arguments\": {\"query\": \"images recycling 5\"}}</ functioncall><|im_end|>",

"format_score": 1.0, "tool_name_score": 1.0, "args_score": 1.0, "task_score": 1.0

}, {

"index": 18, "domain": "math", "instruction": "I saw a dress that I liked...", "baseline_text": "To calculate the discounted price...", "injected_text": "<functioncall>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"$200 * (1 - 0.2)\"}}</

functioncall><|im_end|>", "format_score": 1.0, "tool_name_score": 1.0, "args_score": 1.0, "task_score": 1.0

}, {

"index": 24, "domain": "code", "instruction": "I have an array of numerical entities...", "baseline_text": "To calculate the range...", "injected_text": "<functioncall>{\"name\": \"python_interpreter\", \"arguments\": {\"code\": \"import numpy as np\\nnp.min

([10, 20, 30]) - np.max([10, 20, 30])\"}}</functioncall><|im_end|>", "format_score": 1.0, "tool_name_score": 1.0, "args_score": 1.0, "task_score": 1.0

} ]

Prior work shows that tool-calling decisions can be decoded from pre-generation activations (Esakkiraja et al., 2026). In our setting, a linear probe on mid-layer residual-stream representations also separates Tool-Necessary from Non-Tool inputs with high AUC. However, a high-AUC intent signal does not imply valid execution: the model must emit the trigger pattern and produce schema-compliant JSON under decoding constraints.

#### N.2 Why decodable intent may fail to produce execution

Discrete boundary and early-token competition. Tool entry often depends on early generation decisions that determine whether the model commits to <functioncall>. A representation may encode tool intent, while the output still remains in natural language if the trigger token does not win the early token competition. This motivates interventions that affect the hidden state before decoding commits to a response mode.

Strict interfaces amplify surface errors. A model can produce tool-like text while still failing deployment validators. Invalid JSON, an incorrect tool name, missing arguments, or an incorrect multi-turn execution path can all invalidate the call. Therefore, high trigger recall is not sufﬁcient: a useful controller must improve the conversion from boundary entry to strict execution success, rather than merely making the model call tools more often.

- Table 29: Ablation study on the ALL domain. Precision, Recall, F1, Accuracy, and FPR are computed for toolmode triggering under the strict protocol.

Ablation Mode Strategy Precision Recall F1 Accuracy FPR

full baseline 0.440 0.115 0.182 0.484 0.146 full steer 0.872 0.396 0.504 0.651 0.052 no_domain_predictor baseline 0.440 0.115 0.182 0.484 0.146 no_domain_predictor steer 0.971 0.354 0.519 0.672 0.010 global_only baseline 0.440 0.115 0.182 0.484 0.146 global_only steer 0.826 0.354 0.535 0.656 0.083 domain_only baseline 0.440 0.115 0.182 0.484 0.146 domain_only steer 0.364 0.125 0.186 0.453 0.219 no_gate baseline 0.440 0.115 0.182 0.484 0.146 no_gate steer 0.422 0.365 0.391 0.432 0.500 random baseline 0.440 0.115 0.182 0.484 0.146 random steer 0.556 0.104 0.175 0.510 0.083 mismatch baseline 0.440 0.115 0.182 0.484 0.146 mismatch steer 0.788 0.427 0.554 0.656 0.115

- Table 30: Post-trigger validity under the steer strategy. Component accuracies are computed conditional on triggering.

Ablation Mode Call Count Format Acc Tool Name Acc Args Acc Success Recall Success Precision

full 39 0.949 0.744 0.949 0.281 0.692 no_domain_predictor 35 0.943 0.800 0.943 0.281 0.771 global_only 46 0.978 0.696 0.978 0.323 0.674 domain_only 33 0.970 0.758 0.970 0.115 0.333 no_gate 83 0.976 0.723 0.976 0.292 0.337 random 18 1.000 0.944 1.000 0.094 0.500 mismatch 52 0.962 0.635 0.962 0.333 0.615

Prompt-level control is brittle under protocol variation. Prompt instructions and demonstrations act through the input interface. They can increase tool use, but may also overgeneralize tool patterns to NonTool inputs. This yields a poor precision–FPR trade-off under strict parsers and motivates representationlevel conditional control.

#### N.3 Implication for ASA

These failure modes motivate a controller that is both selective and domain-aware. Probe-guided gating limits intervention to high-conﬁdence inputs, while domain-conditioned directions reduce cross-domain schema interference. ASA therefore treats activation steering as execution-boundary calibration rather than unconditional behavior ampliﬁcation.

### O Additional Ablations

Tables 29 and 30 report supplementary ablation results for analyzing the effect of different ASA components. The ﬁrst table focuses on trigger-level classiﬁcation behavior, while the second table conditions on triggered calls and evaluates post-trigger validity.

Taken together, Tables 29 and 30 isolate the role of selective gating and domain conditioning. Compared to global_only, the full ASA setup yields a better trigger trade-off (higher recall with lower FPR), and it also improves success precision. Removing the gate substantially increases FPR, while random directions fail to reproduce the gains. Domain mismatch primarily affects tool-name validity, indicating that direction selection is important for schema consistency.

