# arXiv:2602.06570v1[cs.CL]6Feb2026

## Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making

Baichuan-M3 Team

Abstract

We introduce Baichuan-M3, a medical-enhanced large language model engineered to shift the paradigm from passive question-answering to active, clinical-grade decision support. Addressing the limitations of existing systems in open-ended consultations, Baichuan-M3 utilizes a specialized training pipeline to model the systematic workflow of a physician. Key capabilities include: (i) proactive information acquisition to resolve ambiguity; (ii) long-horizon reasoning that unifies scattered evidence into coherent diagnoses; and (iii) adaptive hallucination suppression to ensure factual reliability. Empirical evaluations demonstrate that Baichuan-M3 achieves state-of-the-art results on HealthBench, the newly introduced HealthBench-Hallu and ScanBench, significantly outperforming GPT-5.2 in clinical inquiry, advisory and safety. The models are publicly available at https://huggingface.co/collections/baichuan-inc/baichuan-m3.

### 1 Introduction

Large language models (LLMs) are advancing rapidly [1–4], driving broader adoption in healthcare [5–9]. As a result, expectations are shifting from one-off question answering to end-to-end clinical decision support [10, 11]. This trend is reflected in leading systems such as OpenAI’s GPT5.2 [12], ChatGPT Health [13], and Claude in Healthcare [14]. Yet key limitations persist: despite better scores on static, well-specified benchmarks, models often fail to stay evidence-grounded and uncertainty-aware in open-ended clinical interactions, where missing information and longhorizon decisions make hallucinations harder to control [15]. Our goal is therefore to move beyond reliable QA toward decision-support partners that can operate safely in practice.

To bridge the gap between passive retrieval and active clinical support, recent studies have been

evaluated along two partially disconnected paradigms: (i) factuality-focused, single-turn benchmarks and (ii) process-oriented, multi-turn consultation simulations. Benchmarks such as HealthBench [16] and Med-HALT [17] measure factual consistency and common error modes, including hallucinations. They often show that performance drops on harder cases that require complex, multi-constraint clinical reasoning, and that models may produce ungrounded claims. In parallel, growing attention has been paid to Interactive History Taking (IHT) in OSCE-style (Objective Structured Clinical Examination) settings, where models are assessed on their ability to elicit missing information and follow clinical workflows. Systems such as Google’s AMIE [18, 19] demonstrate strong communication quality in simulated encounters under these rubrics.

However, a critical gap remains in unifying these two dimensions. Existing approaches often treat conversational interaction and clinical reasoning as orthogonal objectives, rather than components of a single coherent system. Knowledge-centric models exhibit “inquiry inertia,” lacking the agency to elicit missing evidence, while interaction-focused models can sacrifice diagnostic depth, prioritizing fluency over principled differential reasoning.

Integrating interaction and reasoning is challenging due to three technical bottlenecks. First, heterogeneous training environments across diverse clinical tasks hinder stable multi-task fusion. Second, long-horizon interactions amplify the credit-assignment problem in reinforcement learning(RL)[20–22]: whensupervisionisdominatedbyterminaloutcomes, modelsstruggletoidentify which conversational turns were causally responsible for diagnostic success. Third, efforts to increase reasoning depth often encounter reward saturation, where learning signals diminish near performance plateaus—sometimes accompanied by increased hallucination as models attempt to satisfy complex logical constraints [23, 24].

To address these challenges, we introduce Baichuan-M3, a next-generation medical LLM designed tounifyclinicalinquirywithreliabledecision-making. Baichuan-M3emphasizesthreecorecompetencies aligned with real clinical workflows: (i) proactive information acquisition, (ii) construction of coherent reasoning trajectories, and (iii) adaptive hallucination suppression.

Methodologically, we propose a three-stage training framework consisting of Task-Specific Reinforcement Learning (TaskRL), Offline Policy Distillation, and Multi-Teacher Online Policy Distillation (MOPD). This hierarchical design decouples the optimization of individual competencies before integrating them into a single policy. To improve long-horizon consultation performance, we introduce Segmented Pipeline Reinforcement Learning, which decomposes complex tasks into stages with separate reward signals. We further strengthen diagnostic reasoning via Dynamic Rubric Evolution and targeted RL objectives for hallucination suppression.

Baichuan-M3 achieves state-of-the-art performance on three authoritative benchmarks: (1) 44.4 on HealthBench-Hard, outperforming GPT-5.2; (2) top performance across all three dimensions of ScanBench (our OSCE-like benchmark)—Clinical Inquiry (74.9), Laboratory Testing (72.1), and Diagnosis (74.4)—exceeding both GPT-5.2-High and expert baselines; and (3) superior factual reliability in tool-free hallucination assessments. In summary, our contributions are three-fold:

- • Bridging Inquiry and Reasoning: We present Baichuan-M3, designed to transform LLMs from passive information retrievers into robust decision-support partners. By modeling the full clinical decision-making process, our system demonstrates the agency to actively elicit missing data while maintaining rigorous diagnostic logic, addressing the critical limitation of "hallucination via assumption" in open-ended scenarios.
- • Clinical-Process-AlignedOptimization: Weintroduceatrainingparadigmthatmirrorsprofessional medical training, utilizing Segmented Pipeline RL to align model behavior with distinct stages of clinical consultation (inquiry, lab testing, diagnosis). This approach, combined with dynamic rubric evolution, ensures that the model learns to prioritize evidence-based reasoning over mere conversational fluency or forced logical fitting.
- • Superior Empirical Results: Extensive experiments demonstrate Baichuan-M3’s leadership in both factual reliability and interactive procedural rigor. It sets new state-of-the-art records on HealthBench-Hard and our proposed ScanBench, showing significant gains in hallucination suppression and diagnostic accuracy compared to leading proprietary models like GPT-5.2 and expert baselines.

### 2 Training Infrastructure

In this section, we describe the training infrastructure of Baichuan-M3, including a stable patient simulation environment, a verification system that integrates rubric-based and fact-aware evaluation, and a progressive multi-stage training pipeline. Together, these components provide reliable interaction signals and scalable optimization support for long-horizon medical training.

#### 2.1 Patient Simulator

In our previous work [6], we introduced a patient simulator for doctor–patient interactions. During deployment, we found that the simulator became unstable when modeling proactive patients. Specifically, such behaviors disrupt the consultation flow, leading to simulations difficult to re-

produce. To balance the generalization benefits of stochasticity with the stability required for long-horizon training, we adopt a passive-personality patient simulator and implement two complementary script-driven modes, referred to as Passive Interaction Mode and Interruption-Injected Mode.

Passive Interaction Mode (75% sampling probability): This mode provides only the patient profile, inquiry rubrics, and behavioral constraints rubrics, without any predefined dialogue history. It simulates the opening phase of an initial consultation. Starting from an empty interaction state, this mode evaluates the physician agent’s ability to proactively elicit relevant information and form an initial diagnostic hypothesis under high uncertainty.

Interruption-Injected Mode (25% sampling probability): This mode augments the basic patient information with a predefined dialogue snippet to simulate a mid-consultation state. The snippet ends with a patient-initiated question (often anxiety-driven about severity or treatment), modeling a common interruption during inquiry. However, exposing this snippet to the patient simulator may cause it to mimic the snippet’s speaking style and deviate from our passive-response protocol, introducing instability. We therefore use an asymmetric visibility mechanism: the snippet is visible only to the physician agent and hidden from the patient simulator. The physician answers the question and continues the consultation, while the simulator passively receives and responds to subsequent queries.

To reduce distribution mismatch between the simulated environment and real-world online consultations, we further introduce fine-grained probabilistic configurations in Interruption-Injected Mode. Specifically, end-of-turn questioning accounts for 50% of cases, where the patient question appears only at the end of the snippet, testing the model’s ability to handle abrupt interruptions. The remaining 50% corresponds to mid-turn questioning, which injects questions within an ongoing turn to simulate multi-turn. This mixed configuration ensures robust diagnostic performance underinteractionnoise, includingfrequentinterruptions, challenges, andanxiety-drivenfollow-up questions.

#### 2.2 Verify System

To guarantee the reliability and clinical safety of the generated responses, we construct a comprehensive Verify System that serves as the primary source of reward signals for Reinforcement Learning. Unlike general-domain chat models where fluency and helpfulness are often sufficient,

medical agents face the dual challenge of strictly adhering to diagnostic protocols while ensuring absolute factual precision. A monolithic reward model often struggles to disentangle these orthogonal dimensions—potentially encouraging fluent hallucinations over rigid accuracy. To address this, our system decouples the evaluation process into two parallel streams: a Rubric Verifier that assesses the structural quality and adherence to clinical guidelines through fine-grained criteria, and a Fact Verifier that rigorously checks the biological and medical validity of atomic claims against external authoritative sources. This hybrid approach ensures the model is optimized for both professional compliance and factual groundedness.

##### 2.2.1 Rubric Verifier

The M3 verification stack follows the rubric-based verifier paradigm [25, 26] introduced in M2 [6]. Insteadoftreatingmedicalresponsequalityasasinglemonolithicpreferencesignal, wedecompose each interaction into a set of independently decidable rubric clauses. An LLM-based judge [27] evaluates each clause item by item, and the resulting decisions are aggregated into a scalar reward for policy optimization.

Given a sample x and a set of rubrics R = {ri}Ni=1, each rubric ri is assigned a signed weight wi ∈ [−10,10], where wi > 0 denotes rewards and wi < 0 denotes penalties. An LLM judge outputs a binary decision ai ∈ {0,1} indicating whether ri is satisfied. The task reward is obtained by min-max normalization:

N i=1 wiai − i:wi<0 wi

(1)

Rtask =

N i=1 |wi|

This normalization decouples the reward scale from both the number of rubrics and the magnitude of their weights into [0,1]. As a result, reward distributions remain comparable across different samples and rubric configurations, while integer weights can be used to directly encode relative clause importance in an interpretable manner.

To increase the efficiency in the RL pipeline, reward evaluation is scheduled asynchronously with the rollout process: once the policy emits a response, we immediately launch the corresponding rubric-judging jobs, overlapping judge compute with other unconflicted RL actions (e.g., computing πref/πold log-probabilities) to accelerate the training. In addition, we adopt a prefix-affinity prompt design for the judge: system constraints, output schema, and dialog context share the same template prefix, while the suffixes are substituted with the rubric clause. This processing performs micro-batching to maximize KV-cache [28] reuse, substantially reducing the time cost.

##### 2.2.2 Fact Verifier

Baichuan-M2 [6] mainly relies on rubric-based rewards to shape medical reasoning. As performance saturates on common criteria, the model tends to chase marginal gains by adding rarer clinical details, which increases hallucination risk; we therefore optimize for both completeness and reliability. To make the objective of a low hallucination rate quantitative and optimizable, we build upon existing long-form factuality evaluation methodologies [29, 30] to construct a FactAware Verification Pipeline. The pipeline adopts a two-stage architecture: Firstly, we decompose long-form responses into fine-grained, independently verifiable atomic claims. Secondly, we employ a search-augmented verification agent to validate each claim against authoritative sources. This module runs in parallel with the Rubric Verifier as a service, minimizing latency.

Atomic Claim Extraction Model. The foundation of the pipeline is transforming unstructured model outputs into discrete, verifiable units, referred to as atomic claims. These claims must be self-contained and verifiable even when detached from their original context. To meet this requirement, we apply the following rules:

- 1. Atomicity and Coreference Resolution. We decompose complex compound sentences into single-fact units and resolve pronouns to guarantee semantic independence.
- 2. Noise and Distractor Filtering. We discard units that lack sufficient context to form a factual statement. Crucially, in multiple-choice scenarios, we explicitly exclude the recitation of incorrect options (distractors) to prevent them from being misclassified as hallucinations.
- 3. Deduplication with Order Preservation. Semantically redundant assertions are eliminated while maintaining the original logical sequence of the reasoning chain.

Intheevaluationphase, weinitiallyuseGPT-5[31]asahigh-qualityextractor; however, itsinference latency is prohibitive for online RL. We therefore distill an efficient 8B extraction model from GPT5, trained on datasets spanning multiple medical subtasks, and use it as the final extractor, with detailed fidelity evaluation provided in Appendix A.2.1.

Search-Augmented Verifier. After extraction, atomic claims are fed into a search-augmented validation agent. This agent performs iterative searches over authoritative medical sources such as clinical guidelines and autonomously decides whether the collected evidence is sufficient to support a verdict. Each claim is finally assigned one of three labels: Supported, Refuted or, Uncertain. Compared with methods that rely only on parametric model knowledge or static

knowledge bases, this dynamic search-based mechanism can stay aligned with the latest clinical evidence and better handle the continuous evolution of medical knowledge.

Two-Level Caching System for Acceleration. Introducing fine-grained fact verification into the online RL loop raises significant computational challenges. A single RL iteration may generate thousandsofatomicclaims, andperformingreal-timeexternalsearchforeachclaimisunacceptable in both cost and latency.

To accelerate the training, we design a two-level claim caching system based on the key observation that, for the same clinical query, multiple sampled responses from the model differ in wording but share a high proportion of underlying medical claims.

- • Level-1 cache (exact match). We use Redis to cache verification results for identical claim strings, enabling millisecond-level lookups and result reuse.
- • Level-2 cache (semantic match). We store embeddings of historical claims in a vector database and apply ANN retrieval to find semantically equivalent claims and reuse their verification results.

As the cache pool grows, the overall hit rate increases from below 40% in the early stage to around 80%. This reduces external search requests by approximately 85%, making the impact of fact verification on training is negligible. Semantic caching inevitably introduces some systematic bias. For example, claims with subtle dosage differences may be incorrectly treated as equivalent. We address this issue using the signal denoising mechanism described in Section 3.2.2.

#### 2.3 Multi-Task Training Pipeline

To mitigate the trade-off issue in multi-task learning [32, 33] and reduce development complexity, the training pipeline of Baichuan-M3 is decomposed into three progressive stages: Capability Learning, Distribution Fusion, and Policy Unification. By isolating the acquisition of expert capabilities from the student models, this design achieves a stable fusion process and significantly improves the overall performance. An illustration for the overall pipeline is summarized in Fig. 1.

- Stage 1: Task RL. In the initial stage, our objective is to construct a diverse and high-quality set of expert teachers. Starting from a shared initialization, we deploy independent RL pipelines tailored to specific capability domains. Specifically, to address the diverse requirements from

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### Baichuan-M3 Technical Report

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Stage 2: Offline Policy Distillation

Stage 1: Task RL Stage 3: MOPD

[Figure 17]

Teacher Rollout

RL Pipeline

Token 1

Token 2

Final Output

(Reward)

| | |
|---|---|
| | |

Clinical

Expert

|Reverse KL|
|---|

[Figure 18]

R

Teacher Rollout

Student Rollout

RL Pipeline

Token 1 Token 2

Final Output

(Reward)

Student Rollout

Reverse KL

Healthcare

Expert

Student Model

Init Model

Final Model

Clip-Forward KL

Rewards

MOPD Update

Teacher Dist. 𝜋

Yes

RL Pipeline

Gradient Update

[Figure 19]

[Figure 20]

(Reward)

Env

Condition: 𝜋 < 𝜋

General

Student Dist. 𝜋

Expert

Rewards

Suppress (Mask)

No

Figure 1: An illustration for the three stage pipeline.

medical applications, we explicitly train specialized experts for Clinical Inquiry and Healthcare Consultation, ensuring the model captures the nuances of real-world diagnostic dialogue and patient-centric health advice. In addition, we also train a generalist expert focused on fundamental capabilities, including Instruction Following and General Reasoning.

Inspired by [34], the core philosophy here is differentiation rather than unification. By allowing different models to explore fully under their respective task-specific rewards, we obtain a set of domain-specialized teachers with strong, distinct inductive biases. This divide-and-conquer strategy following [34] effectively isolates gradient interference across tasks, avoiding the optimization conflicts typical in the early stages of multi-task mixture training, thereby providing high-confidence behavioral guidance for subsequent fusion.

- Stage 2: Offline Policy Distillation. The second stage focuses on “compressing” the capabilities of multiple teachers into a single student model via offline distillation. We freeze all teacher models and perform rollouts within their respective domains to construct an offline trajectory dataset, D. The student model then learns from this data in an off-policy manner.

To adapt to the reality of single-sample offline data and ensure numerical stability during fusion, we introduce a standard KL divergence in favor of a restricted distillation objective based on ClipForward-KL. For each sample (s,a) ∈ D and its corresponding teacher policy πt, we define the loss function as follows:

###### Lclip-FKL(θ) = E(s,a)∼D [I(log πθ(a|s) < log πt(a|s)) · (−log πθ(a|s))] (2)

where I(·) denotes the indicator function and πθ(·) and πt(·) denotes the token-wise probability of student and teacher model, respectively. This design applies a one-sided update that only enforces non-inferiority on the teacher’s empirical support, rather than overfitting to the full conditional distribution. This avoids probability over-amplification in the single-sample regime and implicitly preserves entropy outside the datasupport. It can mitigate the mode collapse and leave exploration space for the subsequent mode-seeking optimization in Stage 3.

Leveraging the mode-covering property of Forward KL, this stage enables the student model to broadly cover high-probability regions of different expert distributions. This facilitates the stable inheritance of behavioral patterns and eliminates the instability brought by cold-start associated with direct multi-objective RL.

- Stage 3: Multi-Teacher On-Policy Distillation (MOPD) [35]. In the third stage, the student model re-enters the online interaction environment, performing rollouts across mixed domain distributions. At this point, the model is constrained simultaneously by ground-truth task rewards and multi-teacher priors. Unlike the distribution imitation in the second stage, this stage employs reverse KL regularization [36].

Leveraging the mode-seeking nature of Reverse KL, the student is driven to select the optimal mode when comprised with conflicting advice from multiple teachers, rather than passively averaging them. Guided by real reward signals, the student transitions from an “imitator” to a “decisionmaker,” achieving deep unification and de-noising capability at the policy level.

The Evolution of Teacher Capability. Our proposed framework is not a static pipeline and supports cyclic iterative refinement. The unified model obtained after MOPD can serve as a new initialization for the Stage 1 to achieve domain-specific enhancement, followed by another round of distillation. This flexibility allows for continuous improvement of the model’s capabilities with low marginal cost.

### 3 Task-specific Training Methods

Clinical practice encompasses a diverse spectrum of cognitive activities, ranging from the active elicitation of symptom history to the rigorous provision of evidence-based advice. A monolithic training approach often fails to balance the distinct requirements of these scenarios—specifically, the deductive logic required for diagnosis versus the strict factual adherence required for advisory services. To address this, we adopt task-specific optimization strategies tailored to the unique modalities of medical interaction. In this section, we present our specialized methodologies for two core capabilities: Deep Clinical Consultation, where we employ a segmented pipeline and the Step-Penalized Advantage with Relative baseline algorithm to master the multi-turn diagnostic trajectory; and Credible Healthcare Advisory, which utilizes dynamic rubric evolution and FactAware Reinforcement Learning to ensure the safety and verifiability of medical information.

#### 3.1 Deep Clinical Consultation

As medical AIevolves, the demand for "instantdiagnosis based on symptom input" has surged [37]. However, clinical medicine transcends simple knowledge retrieval; it is a discipline grounded in rigorous evidence and deductive logic. Because identical symptoms may originate from diverse etiologies and patient-specific risk thresholds vary, reliable clinical decisions must rely on granular data, systematic risk assessment, and traceable reasoning [38, 39].

We introduce the Deep Clinical Consultation framework, which re-imagines the consultation as a clinical-grade, structured, and auditable process of information production. Our objective is to collect pivotal clinical data within brief interactions while maintaining stringent safety standards. To achieve this, we propose a training framework (see Fig. 2) comprising a Segmented Pipeline RL

Task Scheduling

Policy Learning

Global Steps

|Update|Step-wise Advantage|
|---|---|
| | |

|Policy Model| |
|---|---|
| | |

Task Slot Task Slot Task Slot Task Slot

Inquiry (T1)

Inquiry (T2)

Inquiry (T3)

Inquiry (T4)

|Step Verifier| |
|---|---|
| | |

|Step Penalty| |
|---|---|
| | |

Context Reuse

For Each Step

DDX (T1)

DDX (T2)

DDX (T3)

...

|SPAR| |
|---|---|
| | |

|Trajectory| |
|---|---|
|Tra|jecto|

Rollout

T

Context Reuse

Lab Test (T1)

Lab Test (T2)

For Total ry

Context Reuse

Diagnose (T1)

|Global Verifier| |Global Reward| |
|---|---|---|---|
| | | | |

###### Figure 2: Segmented Pipeline RL (left) and Policy Learning Algorithm (right).

architecture and the Step-Penalized Advantage with Relative baseline (SPAR) algorithm.

##### 3.1.1 Segmented Pipeline RL

We formulate the consultation as a K-stage generation process with K = 4. Let [K] = {1,...,K} index the stages, and let S = {Inq,DDX,Lab,Diag} denote the set of stage types. For a patient case i at stage k, let x(ki) denote the current input context, which encapsulates the entire trajectory history plus the specific instruction for the current stage (e.g., "Based on the above, suggest lab tests"). The policy πθ generates a response segment yk(i):

yk(i) ∼ πθ(·|x(ki)) (3) As shown in Fig. 2, we employ an Asynchronous Multi-Task Pipeline in which, at global step t, multiple task slots run in parallel on different pipeline stages (and potentially different patient cases); the union of these stage-specific tasks forms the training batch Bt.

Multi-Task Training Objectives. The optimization targets the quality of the generated segment yk relative to the stage-specific goals. For a batch of active contexts Bt = {x(ki)

}Ni=1, the joint objective function is:

i

1 |Bt|

LRL yk(i),Gk(i) (4)

J (θ) =

x(ki)∈Bt

where Gk is the reward function tailored to stage k. This formulation unifies diverse reasoning capabilities (gathering vs. deducing) into a single generative model.

Context Reuse via Gated Transition. The pipeline relies on the auto-regressive accumulation of context. The input for the next stage, x(ki+1) , is constructed by appending the generated response yk(i) and the next stage’s instruction pk+1 to the current context:

x(ki+1) = [x(ki),yk(i),pk+1] (5) However, open-ended generation carries the risk of error propagation. To enforce the "garbagein, garbage-out" principle, we implement a Quality-Gated Transition. The training pool for the subsequent stage, Dk+1, is populated via a filtration process:

 

Dk+1 ∪ {[x(ki),yk(i),pk+1]}, if Vk(i) ≥ τ Dk+1, otherwise (Discard)

(6)

Dk+1 ←



where Vk(i) is the score produced by the stage-k quality verifier and τ is the acceptance threshold. This ensures that only trajectories with clinically valid logical chains are extended, effectively

pruning error paths from the training curriculum.

##### 3.1.2 Policy Learning Algorithm: SPAR

Conventional RL approaches, such as GRPO [20, 40] with global rewards, exhibit significant limitations in long-horizon medical interviewing. These include reward hacking (inflating recall via redundant questions), logic fragmentation (disjointed clinical transitions), and ineffective credit assignment. In long-context dialogues, trajectory-level rewards fail to isolate local errors [41], often inducing training instability by penalizing valid reasoning chains alongside specific flaws.

To address these challenges, we propose SPAR (Step-Penalized Advantage with Relative baseline), which introduces fine-grained step-wise penalties and a decoupled advantage estimation mechanism to induce an adaptive curriculum-learning effect.

Reward Formulation. SPAR employs a hierarchical reward structure. For a generated response y consisting of L logical interaction steps [z1,...,zL], we define each step zj as one complete dialogue exchange (i.e., a user turn followed by the assistant turn). We first compute a Global Reward Rglobal by evaluating the complete trajectory against a set of pre-defined Clinical Rubrics (e.g., diagnostic accuracy, evidence completeness).

Simultaneously, a step verifier performs real-time validation for each interaction step zj. Let Vj denote the set of violation types triggered by step zj (e.g., redundancy, safety risk). We define the step-wise validity factor γj ∈ (0,1] as:

 

1, if Vj = ∅ min

(7)

γj =

(λv), otherwise



v∈Vj

where λv ∈ (0,1) is the penalty coefficient for violation type v. This formulation enforces a minimum validity principle: if a step commits multiple errors, only the most severe penalty (smallest λ) is applied. The effective return for the step is then modulated as Rj = γj · Rglobal.

Step-wiseAdvantageEstimation. ThecoreinnovationofSPARliesinitsadvantagecomputation, which decouples local penalties from the group baseline. Unlike traditional approaches that normalize rewards using the penalized distribution, SPAR computes the advantage Aˆj for step zj by comparing the step-penalized return against an unpenalized group average:

γjRglobal − µraw σraw + ϵ

(8)

Aˆj =

where µraw and σraw represent the mean and standard deviation of the raw global rewards (Rglobal) within the sampled group, excluding any step penalties. Here, a sampled group refers to multiple rollouts generated under the same prompt (i.e., the same consultation context), which enables relative comparison among candidates.

We then apply a GSPO-style policy update [42], using the step-wise advantages Aˆj to weight the likelihood-ratioobjective, sothatpenaltiesareattributedtothespecificinteractionstepsresponsible for violations. Here, a sampled group refers to multiple rollouts generated under the same prompt (i.e., the same consultation context), which enables relative comparison among candidates.

Implicit Curriculum Mechanism. This advantage design facilitates an implicit curriculum [43] by naturally scheduling optimization priorities based on error severity:

- • Phase 1: Correction of Critical Errors. For severe violations (e.g., repetition), we assign a rigorous penalty (e.g., λ ≈ 0.1). This forces the effective return γjRglobal significantly below the group baseline µraw, resulting in a large negative advantage (Aˆj ≪ 0). This dominant gradient signal compels the model to prioritize rectifying fundamental usability flaws in the early training phase.
- • Phase 2: Refinement of Nuance. For subtle imperfections (e.g., rigid phrasing), a milder penalty (e.g., λ ≈ 0.9) is applied. Initially, this small deviation is masked by the high variance (σraw) of global rewards. However, as the policy stabilizes and σraw decreases, these finegrained signals begin to dictate the gradient direction, guiding the model toward stylistic perfection.

By isolating the impact of specific step-level behaviors, SPAR enables precise credit assignment, distinguishing local flaws from overall diagnostic success.

#### 3.2 Credible Healthcare Advisory

This section details our training methodology for credible healthcare advisory, focusing on the synthesis of clinical credibility and interactive utility. To overcome the limitations of static feedback, we first introduce a dynamic rubric evolution framework designed to mitigate reward hacking and ensure that the model pursues genuine reasoning rather than superficial patterns. Furthermore, we present a Fact-Aware Reinforcement Learning strategy that enhances the model’s factual reasoning capability. By moving beyond naive penalty mechanisms, this approach effectively suppresses unfaithful hallucinations while circumventing induced conservatism, thereby preserving

the model’s ability to provide detailed and informative medical counsel.

##### 3.2.1 Dynamic Rubric Evolution

In our previous work [6], we proposed a rubric-based RL approach to enhance clinical reasoning capabilities. However, we observed that this method is highly susceptible to “reward hacking.” The model tends to pursue superficial “high-score performance” rather than genuine reasoning, manifesting as passive verbosity, defensive template usage, or hallucinated details. The root cause lies in the fact that prior rubric formulation relied solely on the input question. Once the question is fixed, the rubric becomes static, creating a predictable structure that the model can easily exploit.

To address this issue, we introduce a significant upgrade to the fundamental quality of the feedback signalintheM3model, comparedtoitspredecessor, M2. Inspiredby[44, 45], weproposeaHumanAI Collaborative Dynamic Evolution mechanism, which crucially incorporates the model’s response into the rubric synthesis process. Specifically, we categorize constraints into two distinct classes:

- • Core Rubric Set: Synthesized solely based on the question, this set guides the general direction of model optimization and ensures fundamental safety.
- • Dynamic Rubric Set: Synthesized dynamically based on both the question and the model’s historical responses. This set aims to constrain non-compliant behaviors and specific vulnerabilities discovered during the training process.

The construction and maintenance of this dynamic set rely on two key modules: Quality Control and Admission/Exit Rules.

Quality Control To ensure the efficacy and clinical relevance of the dynamically generated rubrics, we implement a “Mine-Verify-Inject” closed-loop workflow. First, we identify HighConfidence Samples—responses that achieve high scores under the current system but harbor latent defects. Next, the Rubric Mining Agent analyzes these samples to identify adversarial patterns and draft candidate constraints. Finally, Human Experts intervene to validate these candidates. Instead of authoring rules from scratch, experts review the agent’s output, assessing its boundary determinism and compliance with meta-principles (e.g., Safety > Empiricism). This ensures that the rubric incentivizes reasoning rather than shifting the locus of reward hacking.

Admission and Exit Rules To prevent rule explosion and ensure the reward signal remains potent, the dynamic rubric set follows a “Problem-Driven” lifecycle:

- • Admission: A candidate rubric is not admitted merely for its validity; it is activated only when it targets a statistically significant failure mode (i.e., a high violation rate in model responses). This ensures feedback remains focused on active behavioral deficits.
- • Exit: Once a constraint is consistently satisfied over multiple training epochs (violation rate

→ 0), it is automatically retired from the dynamic set. This “pruning” mechanism prevents reward signal dilution, ensuring the optimization focus remains strictly on emerging and unresolved issues without being washed out by redundant positive rewards.

##### 3.2.2 Fact-Aware Reinforcement Learning

In the paradigm of Reinforcement Learning with Verification Rewards (RLVR), naive hallucination suppression strategies typically convert verification signals directly into scalar penalties [29, 46]. The standard objective form is defined as

R = Rtask + α · Rhallu (9) Here, Rhallu utilizes a count-based hallucination rate metric (−Nhallu/Ntotal). While this objective aims to preserve core medical reasoning capability (Rtask) while reducing hallucinations via a penalty term, it is vulnerable to two major forms of reward hacking in long-form generation:

- • Redundancy-Induced Dilution: The model inflates the denominator Ntotal by producing many factually correct but low-value statements, thereby reducing the measured hallucination rate without correcting the core errors.
- • Penalty-Induced Conservatism: Strict penalties can encourage overly conservative strategies (e.g., shortening outputs to avoid penalties), which undermines exploration and the acquisition of complex reasoning behaviors.

To address these issues, we propose a joint optimization framework comprising Structured Signal Denoising and Dynamic Multi-Objective Aggregation, as shown in Fig. 3.

Structured Signal Denoising To establish a reward signal robust to redundancy, we reformulate the verification of atomic claims as a weighted evaluation based on semantic density. This ensures that hallucinations in core diagnostic statements incur significantly higher penalties than those in marginal content.

Rollout Claim Decomposition,

Reward Aggregation

Online Fact Verification

De-noising & Weighting

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Claim1 Claim2

- Claim A

with Importance

- Claim B

with Importance

- Claim C

Medical Query

TRUE

Hallucination Reward

Cache System

[Figure 33]

[Figure 34]

[Figure 35]

Claim3 Claim4

Write Search

UNCERTAIN

Policy Model

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

FALSE

Claim5

Task Reward

with Importance

LLM WebSearch WebFetch

Verification Outcomes

Response Claim Cluster

Select Claim

Multi-Turn Search

Policy Update RL Optimizer (e.g. GSPO)

Figure 3: Fact-Aware Reinforcement Learning Algorithm.

We first map the response sequences {sj} and extracted claims {ci} into a vector space via a semantic encoder E(·). To prevent the manipulation of metrics through synonymous paraphrasing, we apply semantic clustering with a cosine similarity threshold. By selecting a representative claim c∗k for each cluster Ck, we transition the evaluation metric from lexical frequency to semantic unit density. We then quantify the information contribution of each claim using a Saliency Weight, w(c∗k), defined as its maximum semantic correlation across the response sentences:

,Esj) (10) The factuality reward Rfact is computed as a weighted penalty term:

w(c∗k) = max

cos(Ec∗

k

1≤j≤M

K k=1 w(c∗k) · I(c∗k)

(11)

Rfact = −

K k=1 w(c∗

k) + ϵ

where I(c∗k) is the verification penalty indicator defined as:

 

1 if c∗k ∈ {Refuted,Uncertain} 0 otherwise

(12)

I(c∗k) =



Mechanically, the weighted denominator neutralizes the inflation of claim counts (anti-dilution), while the saliency-dependent numerator ensures that penalties are concentrated on core errors rather than marginal text, thereby preserving reasoning utility.

Dynamic Multi-Objective Aggregation To balance medical reasoning reinforcement with hallucination suppression, we implement a dynamic aggregation mechanism. We introduce a softgating coefficient, λ(Rtask), which modulates the penalty intensity based on the task reward achieved by the on-policy generated response.

Given the Rtask associated with a specific response, we compute the dynamic coefficient λ(Rtask)

strictly as a function of the task reward using a shaped Sigmoid function:

Rtask − µ ∆

(13)

λ(Rtask) = σ κ ·

where σ(·) is the standard sigmoid function, centered at µ = (τmin + τmax)/2 and scaled by ∆ = τmax − τmin (with steepness κ = 10).

The thresholds are calibrated via posterior analysis of the task reward distribution. We specifically set τmin = 0.75 and τmax = 0.95 to delimit the critical interval reflecting effective medical reasoning. This setting aligns the penalty strength with the model’s demonstrated capability:

- • Protection Zone (Rtask < τmin): λ(Rtask) → 0. Penalties are suppressed to prioritize capability optimization, shielding the acquisition of fundamental reasoning skills from interference.
- • Transition Zone (τmin ≤ Rtask ≤ τmax): λ(Rtask) increases non-linearly. In this phase, the system progressively introduces factual constraints.
- • Constraint Zone (Rtask > τmax): λ(Rtask) → 1. Full penalties are enforced to maximize the suppression of hallucinations once the model demonstrates sufficient reasoning competence.

The final total reward R aggregates the task utility with the gated factuality penalty:

R = Rtask + λ(Rtask) · Rfact (14) This formulation implements an implicit curriculum that secures reasoning competence before imposing rigorous safety constraints. We empirically validate this behavior through comparative ablation studies in Appendix A.2.2.

### 4 Evaluation

To comprehensively evaluate the clinical utility and safety of Baichuan-M3, we conduct rigorous experiments across two complementary dimensions: dynamic clinical workflow simulation (ScanBench) and broad-spectrum medical reasoning (HealthBench [16]). We benchmark Baichuan-M3 against a diverse set of competitive baselines to ensure a robust assessment. These include state-of-the-art general-purpose LLMs known for superior reasoning capabilities (e.g., GPT-5.2-High[12], Deepseek-V3.2-Thinking[47], Qwen3-235B-thinking-2507[48]), representative medical-specific models (e.g., AntAngelMed[49]), and our previous generation model (Baichuan-M2 [6]). Crucially, to benchmark the model against real-world professional standards, we explicitly introduce a Human Baseline composed of attending physicians from Grade-A tertiary hospitals, each with a minimum of 5 years of clinical experience. This comparative analysis aims to verify the model’s advancements in complex decision-making, specialized knowledge application,

and hallucination suppression relative to generalist giants, domain specialists, and human experts.

#### 4.1 ScanBench

Unlike traditional static question-answering benchmarks, ScanBench simulates the authentic clinical workflow of "Inquiry → Lab Testing → Diagnosis." The dataset, slated for open-source release, transforms diverse clinical cases into an observable and quantifiable decision path divided into three progressive stages.

##### 4.1.1 Dataset Composition and Statistics

ScanBench is constructed to simulate the authentic clinical environment with high fidelity across three dimensions: case diversity, inquiry granularity, and examination complexity.

Clinical Case Diversity As shown in Table 1, ScanBench contains 303 cases from 12 departments. It covers both common conditions (e.g., General Practice) and long-tail specialties (e.g., Rheumatology, Hematology).

Table 1: Distribution of Clinical Samples by Department.

Department Count Department Count

General Practice 111 Nephrology 15 Surgery 50 Cardiology 14 Gynecology 26 Respiratory Medicine 14 Neurology 25 Endocrinology 7 Gastroenterology 18 Rheumatology 6 Hematology 15 Geriatrics 2

Inquiry Granularity and Rigor To quantify the inquiry process, we annotated 8,857 checklist items as ground truth.

- • Information Density: Each case contains 29.23 items on average (range: 20–35), requiring sustained multi-turn context tracking.
- • Logical Distribution: The checklist mirrors clinical diagnostic logic: History of Present Illness dominates (55.8%), followed by Past Medical History (19.6%) and Personal/Social

- History (14.6%), with Obstetric/Gynecological (5.4%) and Family History (4.7%) included when relevant.
- • Criticality Weighting: We distinguish between essential safety points and general information: 51.3% are labeled Level 2 (Critical) for diagnosis/risk exclusion, and the remaining 48.7% are Level 1 (Supplementary).

Comprehensive Examination Action Space In the auxiliary examination phase, the model operates within a unified action space of 38 distinct categories, replicating the resource management complexity of real hospitals. This includes:

- • Routine & Biochemical: Blood/Urine/Stool Routine, Liver/Kidney Function, Electrolytes, CRP, Glucose Metabolism (including OGTT/HbA1c), etc.
- • Imaging & Functional: CT, MRI, Ultrasound, X-ray, ECG, EEG, and Pulmonary Function Tests.
- • Pathology & Specialized: Tumor Markers, Viral Markers, Autoantibodies, Hormones, Bone Marrow Biopsy, and Endoscopy.

This extensive candidate set requires the model to precisely select necessary tests while avoiding resource waste.

##### 4.1.2 Workflow and Evaluation Methodology

The evaluation pipeline starts with Station 1: Inquiry, where the model conducts multi-turn interactions with a Standardized Patient. To assess process quality, we introduce the SCAN framework, which decomposes consultation performance into four dimensions: Safety Stratification, Information Clarification, Associative Questioning, and Normative Output. We use GPT-4.1 [1] to verify coverage of OSCE-derived key clinical points, while excluding non-diagnostic (or easily engineered) content that is repeatedly mentioned in each consultation (e.g., age/sex restatements or templated self-introductions).

Given the elicited evidence, Station 2: Lab Testing evaluates both resource efficiency and interpretative accuracy. The model proposes a differential diagnosis and selects laboratory or imaging tests from a unified candidate pool, where candidate tests are categorized into essential and optional groups based on their contributions to clinical diagnosis and decision-making. Performance is evaluated using a weighted F1 score, in which recall for essential tests is assigned a higher weight, thereby reflecting their greater importance in the diagnostic workflow while maintaining

a balanced assessment of overall precision and coverage.

TheworkflowconcludeswithStation3: Diagnosis, wherethemodelintegratesallpriorinformation toinferafinaldiagnosis. WeadoptahierarchicalmatchingcriterionbasedontheICD-10taxonomy, rewarding correct leaf-node matches while penalizing off-branch predictions.

##### 4.1.3 Performance Analysis

As illustrated in Fig. 4, Baichuan-M3 exhibits a comprehensive advantage, ranking first across all three stations. Most notably, in the challenging Clinical Inquiry phase, it achieves a score of 74.9, surpassing the second-best model (GPT-5.2-High) by 12.4 points and the human baseline by over 20 points. This dominance extends to laboratory testing (72.1) and final diagnosis (74.4), suggesting that Baichuan-M3 possesses robust, end-to-end medical reasoning capabilities rather than merely excelling in isolated tasks.

[Figure 46]

Figure 4: Overall performance comparison on ScanBench.

Further decomposition of the inquiry capability via the SCAN framework (Fig. 5) reveals that Baichuan-M3 is the sole model to demonstrate dominant leadership across all four dimensions, consistently outperforming both SOTA LLMs and human experts.

Specifically, in Safety Stratification, Baichuan-M3 achieves a remarkable score of 75.8, creating a substantial gap over the runner-up Qwen3-235B (48.3) and nearly doubling the human benchmark (40.1). This indicates superior sensitivity to “red flag” symptoms and critical risks. In terms of Association & Inquiry, the model scores 72.6, significantly surpassing GPT-5.2-High (54.5). This highlights its sophisticated grasp of differential diagnosis, enabling it to proactively uncover hidden clinical clues beyond the user’s initial description. Furthermore, Baichuan-M3 excels in Clarity Matters (84.5) and Normative Protocol (59.9), ensuring that the collected information is both

granular and structurally standardized. By integrating these strengths, Baichuan-M3 successfully creates a closed loop of precise inquiry and secure decision-making that exceeds human-level standardization.

###### LLM Inquiry Capability Evaluation (SCAN Framework)

###### Safety Stratification

###### Clarity Matters

75.8

84.5

###### Baichuan-M3

###### Baichuan-M3

48.3

81.4

Qwen3-235B-Thinking

GPT-5.2-High

43.2

74.6

Human

GPT-5.2-High

40.1

68.1

Human

AntAngelMed

39.9

67.5

Baichuan-M2

Baichuan-M2

36.4

66.3

AntAngelMed

Deepseek-V3.2-Thinking

33.9

66

Deepseek-V3.2-Thinking

Qwen3-235B-Thinking

0 20 40 60 80 100

0 20 40 60 80 100

###### Association & Inquiry

Normative Protocol

72.6

59.9

###### Baichuan-M3

###### Baichuan-M3

54.5

43.2

GPT-5.2-High

GPT-5.2-High

47.8

41.4

Human

Qwen3-235B-Thinking

40.8

39.4

Baichuan-M2

Deepseek-V3.2-Thinking

37.4

38.8

AntAngelMed

Qwen3-235B-Thinking

34.4

37.2

Human

Deepseek-V3.2-Thinking

28.6

35.3

Baichuan-M2

AntAngelMed

0 20 40 60 80 100

0 20 40 60 80 100

Figure 5: Detailed breakdown of Inquiry Capabilities.

##### 4.1.4 Dynamic Consultation Efficiency

Figure 6 plots model scores against the number of dialogue turns. To reduce noise from very rare long conversations, we drop turn bins that contain fewer than 10% of the total samples. This makes the trend across turns easier to interpret.

Convergence in Basics, Divergence in Reasoning Figure 6b shows that most models quickly reach a similar level on symptom clarification (around 0.7–0.8). The gap becomes clear in Association & Inquiry (Fig. 6c): Baichuan-M3 keeps improving as the dialogue goes on, while generalist models (e.g., Deepseek and Qwen) fall behind — resulting in nearly a twofold advantage at longer horizons.

Risk Sensitivity and Contextual Adherence For Safety Stratification (Fig. 6a), Baichuan-M3 is more responsive to risk signals, and its score rises to about 0.7 as evidence accumulates. By contrast, GPT-5.2-High stays around 0.5 even with longer dialogues, suggesting weaker acuterisk recognition. For Normative Protocol (Fig. 6d), performance tends to improve in later turns, indicating that keeping track of context helps models follow structured clinical workflows.

###### (a) Safety Stratification

###### (b) Clarity Matters

| |Triage| |Inpatient|Inquiry| | | |
|---|---|---|---|---|---|---|---|
| |Gene|ral Inquiry| | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |Triage Gene|ral Inquiry|Inpatient|Inquiry| | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.8

0.8

0.7

0.6

AverageScore

AverageScore

0.6

0.5

0.4

0.4

0.3

0.2

0.2

0.1

0.0

0.0

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

Number of Turns

Number of Turns

###### (c) Association & Inquiry

(d) Normative Protocol

| |Triage| |Inpatient|Inquiry| | | |
|---|---|---|---|---|---|---|---|
| |Gene|ral Inquiry| | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |Triage Gene|ral Inquiry|Inpat|ient Inquiry| | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.8

0.6

0.7

0.5

0.6

AverageScore

AverageScore

0.5

0.4

0.4

0.3

0.3

0.2

0.2

0.1

0.1

0.0

0.0

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

Number of Turns

Number of Turns

- Figure 6: Evolution of model performance across dialogue turns.

In summary, while general LLMs suffice for basic information gathering, specialized training is indispensable for the complex reasoning and safety assurance required in clinical settings.

#### 4.2 HealthBench

HealthBench serves as a rigorous benchmark to evaluate the clinical reasoning capabilities and safety boundaries of LLMs. By assessing performance across varying difficulty levels and dimensions, it provides a comprehensive view of a model’s utility in real-world medical scenarios.

##### 4.2.1 HealthBench-Main

As illustrated in Fig. 7, Baichuan-M3 establishes a new state-of-the-art (SOTA) standard across all key metrics. On the comprehensive HealthBench Total, Baichuan-M3 achieves a score of 65.1, surpassing the runner-up GPT-5.2-High (63.3) by a clear margin. Notably, in the challenging HealthBench Hard subset, Baichuan-M3 further extends its lead with a score of 44.4, significantly outperforming strong competitors such as GPT-5.2-High (42.0) and AntAngelMed (39.6). Furthermore, Baichuan-M3 demonstrates exceptional reliability by securing the lowest Hallucination Rate of 3.5%, indicating a robust balance between deep clinical reasoning and factual accuracy compared to other models.

[Figure 50]

- Figure 7: Overall performance comparison on HealthBench.

In terms of fine-grained capabilities compared with M2, M3 exhibits broad gains across most evaluation dimensions, resulting in a more robust and well-balanced medical service profile (see Fig. 8). The improvements are especially pronounced in context seeking and context awareness. We attribute these gains to transfer from enhanced Deep Clinical Consultation training: M3 more proactively elicits missing history and risk factors while recognizing contextual constraints that shape clinical decisions. As a result, it produces more complete and reliable treatment recommendations and triage assessments. Overall, these results suggest that M3 generalizes the “inquiry–clarification–decision” interaction paradigm learned in consultation tasks to a broader set of healthcare scenarios.

context seeking

communication quality

0.637

complex responses

communication

0.623 0.619

completeness

0.711

0.699

0.558

0.686

0.449

0.672

0.414

0.593

instruction following

0.598

0.476

0.746

0.534

0.754

health data tasks

emergency referrals

0.480

0.571

0.562

0.635

0.649

0.642

0.678

context awareness

0.689

global health hedging

accuracy

M2 M3

Figure 8: HealthBench fine-grained comparison: M2 vs M3 (themes & axes).

##### 4.2.2 HealthBench-Hallu

While benchmarks such as HealthBench have established standards for evaluating medical LLMs, ensuring clinical safety remains paramount for healthcare AI deployment. Medical hallucinations, which involve the generation of superficially plausible yet factually incorrect information, represent a significant concern acknowledged by regulatory authorities and the research community [17, 50, 51]. This issue is particularly pronounced in complex medical reasoning tasks, where model outputs involve extended passages with high information density, encompassing specialized knowledge such as precise pharmaceutical dosing and rare clinical presentations. The combination of extended generation lengths and linguistic coherence creates conditions wherein fine-grained factual errors may evade detection, thereby presenting substantial clinical risks. We therefore introduce HealthBench-Hallu, a fine-grained evaluation framework designed to assess the factual integrity of model responses generated across HealthBench tasks. By decomposing these outputs into discrete atomic claims and validating them against external evidence, this metric provides a rigorous quantification of medical hallucinations.

Metric Design HealthBench-Hallu focuses on healthcare factual misconceptions embedded within the model’s generated content, which primarily manifest as:

- • Erroneous or misapplied medical knowledge: Stating incorrect clinical facts or applying correct knowledge to inappropriate contexts.
- • Fabricated or distorted medical evidence: Inventing non-existent references, falsifying clin-

ical data, or fabricating causal relationships.

HealthBench-HalluEvaluation HealthBench-HallureusestheFact-AwareVerificationPipeline(see

Section 2.2.2), but runs it in a high-precision (rather than high-throughput) configuration: we upgrade the claim extractor to GPT-5 to improve coverage of fine-grained atomic claims, and enforce real-time multi-turn search verification instead of relying on cached evidence. Based on above evidence labels generated by this verification pipeline, we calculate the Weighted Hallucination Rate (H):

N i=1 wi

H =

|Total Claims|

(15)

The weights wi are assigned based on factual risk stratification: Refuted (1.0) represents severe factual fallacies; Uncertain (0.5) represents statements with insufficient evidence or ambiguity risks; and Supported (0.0) denotes correct statements. This metric directly reflects the credibility boundaries of the model when addressing complex medical problems.

Experimental Results Table 2 demonstrates that the integration of Fact-Aware RL enables Baichuan-M3-235B to effectively suppress hallucinations while preserving medical reasoning capabilities. The model maintains competitive task performance (HealthBench Score: 65.1) comparable to the variant without reinforcement learning (66.2), while substantially reducing both refuted response and uncertainty rates by approximately 50%. These results indicate that the adaptive weighting strategy effectively mitigates the typical trade-off between safety constraints and model utility, avoiding the reasoning degradation commonly induced by overly conservative approaches.

Table 2: Trade-off between hallucination suppression and capability

Model HealthBench Score Refuted Rate Uncertain Rate

GPT-5.2-High 63.3 2.37% 2.78% Baichuan-M2-32B 60.1 5.73% 5.43% M3 - w/o Fact Aware RL 66.2 4.68% 3.64% Baichuan-M3-235B 65.1 2.45% 2.07%

Hallucination Analysis through Knowledge Probes To elucidate the underlying mechanism by which Fact-Aware RL reduces hallucination rates, we employ knowledge probes to analyze the

alignment between the model’s Internal Cognition (parametric truthfulness) and its External Output (generated claims), as illustrated in Fig. 9. This analysis allows us to decouple the sources of error by examining whether the model’s output faithfully reflects its internal parameters or deviates due to generation instability.

Knowledge Consistency on Supported Claims (Facts)

Knowledge Consistency on Refuted Claims (Hallucinations)

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | |10.1%| | | | |88.3%| | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | |11.2%| | | | |87.1%| | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | |13.5%| | | | |84.2%| | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | |15.2%| | | | |83.5%| | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |26.4%| | |28.| | |7%| | |44.9%| | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| |30.4%| | | | | |29.9%| | |39.| |7%| |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| |35.4%| | | | | |28.0%| | | |36|.6%| |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| |9.2%| |32.6%| | | | | | |58.2%| | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

Baichuan-M3

Baichuan-M3

M3 w/o Fact Aware RL

M3 w/o Fact Aware RL

Baichuan-M2

Baichuan-M2

GPT-5.2

GPT-5.2

0 20 40 60 80 100

0 20 40 60 80 100

Percentage of Claims (%)

Percentage of Claims (%)

Contradictory (Knowledge contradicts Claim)

Inconsistent (Knowledge partially matches)

Consistent (Knowledge supports Claim)

| |
|---|

| |
|---|

| |
|---|

Figure 9: Knowledge boundary alignment analysis. We categorize the alignment between internal parameters and generated claims into three states: Consistent (output aligns with internal cognition), Inconsistent (partial mismatch), and Contradictory (output opposes internal cognition). Notably, in the context of hallucinations (Refuted Claims), a Consistent state indicates an “honest error” derived from incorrect internal knowledge.

The results reveal a distinct bifurcation in the model’s alignment dynamics induced by Fact-Aware RL. For factually correct outputs (Supported Claims), the consistency between internal cognition and external output remains robust at 88.3%, indicating that the model’s factual truths are firmly anchored in its internal parameters. More critically, for erroneous outputs (Refuted Claims), the consistency rate significantly increases to 44.9%. This upward shift signifies a marked reduction in “unfaithful hallucinations”—instances where the model possesses correct internal cognition but generates contradictory false information. The data suggests that the remaining hallucinations are predominantly honest errors, where the external output faithfully reflects a misconception inherent in the model’s parameters.

Ultimately, these findings imply that the primary efficacy of Fact-Aware RL lies not in injecting new knowledge, but in regulating the model’s generation strategy to strictly converge within its authentic knowledge boundaries.

### 5 Inference Optimization

To facilitate the accessibility and efficient deployment of the Baichuan-M3 model in healthcare applications, two inference optimized strategies are implemented. First, to accelerate text generation, the Gated Eagle-3 speculative decoding approach was introduced, leading to a substantial improvement in inference throughput. Second, advanced quantization methods were applied to significantly reduce the memory requirements of the model without compromising accuracy. These inference optimizations lower the practical barriers to deployment and support broader applications of advanced medical large language models.

#### 5.1 Speculative Decoding

Speculative decoding is an inference-time acceleration technique for autoregressive generation. It uses a lightweight draft model to propose multiple candidate tokens, which are then verified in parallel by the target model. The algorithm accepts the longest verified prefix and discards the remaining candidates, allowing the target model to commit multiple tokens per verification step and thereby increasing decoding throughput.

In the Eagle-3 [52] framework, the draft model is additionally conditioned on hidden states from the target model to leverage richer semantic information. However, the pronounced capacity gap between the target and draft models can induce a representation mismatch: high-dimensional and information-dense hidden states may overwhelm the lightweight draft model, reducing its ability to effectively exploit the auxiliary signal. This often leads to lower candidate acceptance rates and consequently limits the achievable speedup.

To mitigate this issue, we incorporate a Gated-Attention [53] module into the Eagle-3 draft model (called Gated Eagle-3; see Fig. 10), providing a dynamic and learnable mechanism to regulate the injected information. Concretely, the attention output is routed through a gating unit and modulated via element-wise multiplication with a gate vector that is dynamically generated from the current-layer input. This design enables fine-grained, dimension-wise control of information flow, emphasizing salient features while suppressing redundant or noisy components.

As detailed in Appendix A.4, Gated Eagle-3 achieves an average acceptance length improvement of 0.31 and improves average throughput by 12% over the Eagle-3 base. These results indicate that selectively regulating target-model information translates more reliably into practical decoding acceleration, providing useful insights for future improvements.

Head

Gated MLP Attention

Target Model

o proj

Head

Norm

Element-wise multiply

σ

GQA

×N

gate

q k v

Gated Attention

Decoder Layer

q proj

k proj v proj Gate proj

Norm

Embedding

Hidden state Embedding

Gated Eagle-3 Draft Decoder Layer

Figure 10: An illustration of Gated Eagle-3 draft model.

#### 5.2 Quantization

To reduce GPU memory consumption for Baichuan-M3 in resource-constrained settings, we apply INT4 weight quantization. However, the sparsely activated nature of Mixture-of-Experts models poses challenges for standard quantization calibration [54]: calibration corpora typically activate only a subset of experts, resulting in biased calibration. Consequently, frequently activated experts are quantized accurately, whereas rarely activated experts incur larger quantization errors, which can lead to unstable and unpredictable accuracy degradation at inference time.

To address this issue, we propose a self-generated calibration scheme that promotes uniform expert coverage. Specifically, we construct a multi-domain prompt set and use the BF16 model to generate high-quality responses for calibration. This approach offers two key benefits: (i) the diverse prompts activate nearly all experts, providing sufficient samples for each expert and mitigating calibration bias; and (ii) the self-generated responses encourage the quantized model to match the output distribution of the BF16 model, thereby reducing distributional discrepancies. The training of quantization parameters was based on AutoRound [55] framework, and the quantization format adheres to the GPTQ [56] standard.

Empirically, the resulting INT4-quantized M3 model achieves near-lossless performance relative

to its BF16 counterpart on mainstream benchmarks. These results validate the effectiveness of the proposed MoE-specific quantization calibration strategy and provide practical guidance for deploying large-scale sparse models.

### 6 Conclusion

We introduce Baichuan-M3, a medical-enhanced large language model that unifies clinical inquiry with reliable medical decision-making. By explicitly modeling the clinical workflow, Baichuan-M3 enables proactive information acquisition, coherent multi-step reasoning, and effective hallucination suppression. Through a three-stage training paradigm combining task-specific reinforcement learning and multi-teacher distillation, the model achieves strong performance on both factual and process-oriented benchmarks, including HealthBench, HealthBench-Hallu and the OSCEstyle ScanBench. These results demonstrate that workflow-aligned optimization is an effective approach for advancing clinical-grade medical LLMs.

### 7 Limitation and Future Work

Baichuan-M3 is currently limited to episodic, text-based clinical scenarios and does not fully capture longitudinal disease management, multimodal clinical signals, or ultra-long-horizon reasoning across patient trajectories. While hallucination control is substantially improved, rare high-risk errors and limited explicit grounding in evidence-based sources remain open challenges. Future work will focus on extending the model toward full-pathway clinical reasoning with multimodal inputs, long-context optimization, and tighter integration of evidence retrieval, safety constraints, and environment-based reinforcement learning.

### 8 Contribution

Contributors are presented in alphabetical order according to their first names. An asterisk (*) denotes those who are no longer part of the team.

#### Core Contributors

Chengfeng Dou, Fan Yang, Fei Li, Jiyuan Jia, Qiang Ju, Shuai Wang, Tianpeng Li, Xiangrong Zeng*, Yĳie Zhou

#### Contributors

Hongda Zhang, Jinyang Tai, Linzhuang Sun, Peidong Guo, Yichuan Mo

#### Experts and Advisors

Xiaochuan Wang, Hengfu Cui, Zhishou Zhang

### References

- [1] OpenAI. GPT-4 Technical Report. 2023. arXiv: 2303.08774. url: https://arxiv.org/abs/2303.08774.
- [2] Aaron Jaech, Adam Kalai, Adam Lerer, et al. “OpenAI o1 System Card”. In: CoRR abs/2412.16720 (2024). doi: 10.48550/ARXIV.2412.16720. arXiv: 2412.16720. url: https://doi.org/10.48550/arXiv.2412.16720.
- [3] Aiyuan Yang, Bin Xiao, Bingning Wang, et al. “Baichuan 2: Open Large-scale Language Models”. In: CoRR abs/2309.10305 (2023). doi: 10.48550/ARXIV.2309.10305. arXiv: 2309.10305. url: https://doi.org/10.48550/arXiv.2309.10305.
- [4] Yadong Li, Jun Liu, Tao Zhang, et al. “Baichuan-Omni-1.5 Technical Report”. In: CoRR abs/2501.15368 (2025). doi: 10.48550/ARXIV.2501.15368. arXiv: 2501.15368. url: https://doi.org/10.48550/arXiv.2501.15368.
- [5] Bingning Wang, Haizhou Zhao, Huozhi Zhou, et al. “Baichuan-M1: Pushing the Medical Capability of Large Language Models”. In: CoRR abs/2502.12671 (2025). doi: 10.48550/ARXIV.2502.12671. arXiv: 2502.12671. url: https://doi.org/10.48550/arXiv.2502.12671.
- [6] Chengfeng Dou, Chong Liu, Fan Yang, et al. “Baichuan-M2: Scaling Medical Capability with Large Verifier System”. In: CoRR abs/2509.02208 (2025). doi: 10.48550/ARXIV.2509.02208. arXiv: 2509.02208. url: https://doi.org/10.48550/arXiv.2509.02208.
- [7] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, et al. “MedGemma Technical Report”. In: CoRR abs/2507.05201 (2025). doi: 10.48550/ARXIV.2507.05201. arXiv: 2507.05201. url: https://doi.org/10.48550/arXiv.2507.05201.
- [8] LASA Team, Weiwen Xu, Hou Pong Chan, et al. “Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning”. In: CoRR abs/2506.07044 (2025). doi: 10.48550/ARXIV.2506.07044. arXiv: 2506.07044. url: https://doi.org/10.48550/arXiv.2506.07044.

- [9] Songtao Jiang, Yuan Wang, Sibo Song, et al. “Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding”. In: CoRR abs/2510.08668 (2025). doi: 10.48550/ARXIV.2510.08668. arXiv: 2510.08668. url: https://doi.org/10.48550/arXiv.2510.08668.
- [10] Binbin Li, Tianxin Meng, Xiaoming Shi, Jie Zhai, and Tong Ruan. “MedDM: LLM-executable clinical guidance tree for clinical decision-making”. In: CoRR abs/2312.02441 (2023). doi: 10.48550/ARXIV.2312.02441. arXiv: 2312.02441. url: https://doi.org/10.48550/arXiv.2312.02441.
- [11] Shengxin Hong, Liang Xiao, Xin Zhang, and Jianxia Chen. “ArgMed-Agents: Explainable Clinical Decision Reasoning with LLM Disscusion via Argumentation Schemes”. In: IEEE International Conference on Bioinformatics and Biomedicine, BIBM 2024, Lisbon, Portugal, December 3-6, 2024. Ed. by Mario Cannataro, Huiru Jane Zheng, Lin Gao, et al. IEEE, 2024, pp. 5486–5493. doi: 10.1109/BIBM62325.2024.10822109. url: https://doi.org/10.1109/BIBM62325.2024.10822109.
- [12] OpenAI. Introducing GPT-5.2. https://openai.com/index/introducing-gpt-5-2/. Accessed: 2026-01-29. Dec. 2025.
- [13] OpenAI. Introducing ChatGPT Health. Jan. 2026. url: https://openai.com/index/introducing-chatgpt-health.
- [14] Anthropic. Advancing Claude in healthcare and the life sciences. Jan. 2026. url: https://www.anthropic.com/news/healthcare-life-sciences.
- [15] Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Munan Ning, and Li Yuan. “LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples”. In: CoRR abs/2310.01469 (2023). doi: 10.48550/ARXIV.2310.01469. arXiv: 2310.01469. url: https://doi.org/10.48550/arXiv.2310.01469.
- [16] Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, et al. “HealthBench: Evaluating Large Language Models Towards Improved Human Health”. In: CoRR abs/2505.08775 (2025). doi: 10.48550/ARXIV.2505.08775. arXiv: 2505.08775. url: https://doi.org/10.48550/arXiv.2505.08775.
- [17] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. “Med-HALT: Medical Domain Hallucination Test for Large Language Models”. In: Proceedings of the 27th Conference on Computational Natural Language Learning, CoNLL 2023, Singapore, December 6-7, 2023. Ed. by Jing Jiang, David Reitter, and Shumin Deng. Association for Computational Linguistics, 2023, pp. 314–334. doi: 10.18653/V1/2023.CONLL-1.21. url: https://doi.org/10.18653/v1/2023.conll-1.21.
- [18] Tao Tu, Anil Palepu, Mike Schaekermann, et al. “Towards Conversational Diagnostic AI”. In: CoRR abs/2401.05654 (2024). doi: 10.48550/ARXIV.2401.05654. arXiv: 2401.05654. url: https://doi.org/10.48550/arXiv.2401.05654.
- [19] Elahe Vedadi, David G. T. Barrett, Natalie Harris, et al. “Towards physician-centered oversight of conversational diagnostic AI”. In: CoRR abs/2507.15743 (2025). doi: 10.48550/ARXIV.2507.15743. arXiv: 2507.15743. url: https://doi.org/10.48550/arXiv.2507.15743.
- [20] Daya Guo, Dejian Yang, Haowei Zhang, et al. “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning”. In: Nat. 645.8081 (2025), pp. 633–638. doi: 10.1038/S41586-025-09422-Z. url: https://doi.org/10.1038/s41586-025-09422-z.
- [21] Mingan Lin, Fan Yang, Yanjun Shen, et al. “Baichuan Alignment Technical Report”. In: CoRR abs/2410.14940

(2024). doi: 10.48550/ARXIV.2410.14940. arXiv: 2410.14940. url: https://doi.org/10.48550/arXiv.2410.14940.

- [22] LiChun Cao and ZhiMin. “An Overview of Deep Reinforcement Learning”. In: Proceedings of the 2019 4th International Conference on Automation, Control and Robotics Engineering, CACRE 2019, Shenzhen, China, July 19-21,

2019. ACM, 2019, 17:1–17:9. doi: 10.1145/3351917.3351989. url: https://doi.org/10.1145/3351917.3351989.

- [23] Zĳun Yao, Yantao Liu, Yanxu Chen, et al. “Are Reasoning Models More Prone to Hallucination?” In: CoRR abs/2505.23646 (2025). doi: 10.48550/ARXIV.2505.23646. arXiv: 2505.23646. url: https://doi.org/10.48550/arXiv.2505.23646.
- [24] Chenlong Yin, Zeyang Sha, Shiwen Cui, and Changhua Meng. “The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination”. In: CoRR abs/2510.22977 (2025). doi: 10.48550/ARXIV.2510.22977. arXiv: 2510.22977. url: https://doi.org/10.48550/arXiv.2510.22977.
- [25] Zenan Huang, Yihong Zhuang, Guoshan Lu, et al. “Reinforcement Learning with Rubric Anchors”. In: CoRR abs/2508.12790 (2025). doi: 10.48550/ARXIV.2508.12790. arXiv: 2508.12790. url: https://doi.org/10.48550/arXiv.2508.12790.
- [26] Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Bing Liu, and Sean Hendryx. “Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains”. In: CoRR abs/2507.17746 (2025). doi: 10.48550/ARXIV.2507.17746. arXiv: 2507.17746. url: https://doi.org/10.48550/arXiv.2507.17746.
- [27] Jiawei Gu, Xuhui Jiang, Zhichao Shi, et al. “A Survey on LLM-as-a-Judge”. In: CoRR abs/2411.15594 (2024). doi: 10.48550/ARXIV.2411.15594. arXiv: 2411.15594. url: https://doi.org/10.48550/arXiv.2411.15594.
- [28] Haoyang Li, Yiming Li, Anxin Tian, et al. “A Survey on Large Language Model Acceleration based on KV Cache Management”. In: Trans. Mach. Learn. Res. 2025 (2025). url: https://openreview.net/forum?id=z3JZzu9EA3.
- [29] Sewon Min, Kalpesh Krishna, Xinxi Lyu, et al. “FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation”. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023. Ed. by Houda Bouamor, Juan Pino, and Kalika Bali. Association for Computational Linguistics, 2023, pp. 12076–12100. doi: 10.18653/V1/2023.EMNLP-MAIN.741. url: https://doi.org/10.18653/v1/2023.emnlp-main.741.
- [30] Jerry Wei, Chengrun Yang, Xinying Song, et al. “Long-form factuality in large language models”. In: Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024. Ed. by Amir Globersons, Lester Mackey, Danielle Belgrave, et al. 2024. url: http://papers.nips.cc/paper_files/paper/2024/hash/937ae0e83eb08d2cb8627fe1def8c751-AbstractConference.html.
- [31] Aaditya Singh, Adam Fry, Adam Perelman, et al. “Openai gpt-5 system card”. In: CoRR (2025). arXiv: 2601.03267.
- [32] Sachin Ravi, Sebastian Musslick, Maia Hamin, Theodore L. Willke, and Jonathan D. Cohen. “Navigating the Trade-Off between Multi-Task Learning and Learning to Multitask in Deep Neural Networks”. In: CoRR abs/2007.10527 (2020). arXiv: 2007.10527. url: https://arxiv.org/abs/2007.10527.
- [33] Yichuan Mo and Shilin Wang. “Multi-Task Learning Improves Synthetic Speech Detection”. In: IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, Virtual and Singapore, 23-27 May

2022. IEEE, 2022, pp. 6392–6396. doi: 10.1109/ICASSP43922.2022.9746059. url: https://doi.org/10.1109/ICASSP43922.2022.9746059.

- [34] Yisen Wang, Yichuan Mo, Hongjun Wang, Junyi Li, and Zhouchen Lin. “Generalist++: A Meta-learning Framework for Mitigating Trade-off in Adversarial Training”. In: CoRR abs/2510.13361 (2025). doi: 10.48550/ARXIV.2510.13361. arXiv: 2510.13361. url: https://doi.org/10.48550/arXiv.2510.13361.
- [35] LLM-Core Xiaomi. MiMo-V2-Flash Technical Report. 2025. url: https://github.com/XiaomiMiMo/MiMo-V2-Flash/paper.pdf.
- [36] Kevin Lu and Thinking Machines Lab. “On-Policy Distillation”. In: Thinking Machines Lab: Connectionism (2025). https://thinkingmachines.ai/blog/on-policy-distillation. doi: 10.64434/tml.20251026.
- [37] Jie Liu, Wenxuan Wang, Zizhan Ma, et al. Medchain: Bridging the Gap Between LLM Agents and Clinical Practice with Interactive Sequence. 2025. arXiv: 2412.01605 [cs.CL]. url: https://arxiv.org/abs/2412.01605.
- [38] Kangenbei Liao, Qianlong Liu, Zhongyu Wei, et al. “Task-oriented Dialogue System for Automatic Disease Diagnosis via Hierarchical Reinforcement Learning”. In: CoRR abs/2004.14254 (2020). arXiv: 2004.14254. url: https://arxiv.org/abs/2004.14254.
- [39] Chengfeng Dou, Ying Zhang, Zhi Jin, et al. “Integrating Physician Diagnostic Logic into Large Language Models: Preference Learning from Process Feedback”. In: Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024. Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Vol. ACL 2024. Findings of ACL. Association for Computational Linguistics, 2024, pp. 2453–2473. doi: 10.18653/V1/2024.FINDINGS-ACL.144. url: https://doi.org/10.18653/v1/2024.findings-acl.144.
- [40] Shihui Yang, Chengfeng Dou, Peidong Guo, et al. “DCPO: Dynamic Clipping Policy Optimization”. In: CoRR abs/2509.02333 (2025). doi: 10.48550/ARXIV.2509.02333. arXiv: 2509.02333. url: https://doi.org/10.48550/arXiv.2509.02333.
- [41] Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, et al. “Let’s Verify and Reinforce Image Generation Step by Step”. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025. Computer Vision Foundation / IEEE, 2025, pp. 28662–28672. doi: 10.1109/CVPR52734.2025.02669. url: https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Lets_Verify_and_Reinforce_Image_ Generation_Step_by_Step_CVPR_2025_paper.html.
- [42] Chujie Zheng, Shixuan Liu, Mingze Li, et al. “Group Sequence Policy Optimization”. In: CoRR abs/2507.18071

(2025). doi: 10.48550/ARXIV.2507.18071. arXiv: 2507.18071. url: https://doi.org/10.48550/arXiv.2507.18071.

- [43] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. “Curriculum learning”. In: Proceedings of the 26th annual international conference on machine learning. 2009, pp. 41–48.
- [44] Rulin Shao, Akari Asai, Shannon Zejiang Shen, et al. “DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research”. In: CoRR abs/2511.19399 (2025). doi: 10.48550/ARXIV.2511.19399. arXiv: 2511.19399. url: https://doi.org/10.48550/arXiv.2511.19399.
- [45] Junkai Zhang, Zihao Wang, Lin Gui, et al. “Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training”. In: CoRR abs/2509.21500 (2025). doi: 10.48550/ARXIV.2509.21500. arXiv: 2509.21500. url: https://doi.org/10.48550/arXiv.2509.21500.
- [46] Xilun Chen, Ilia Kulikov, Vincent-Pierre Berges, et al. “Learning to Reason for Factuality”. In: CoRR abs/2508.05618 (2025). doi: 10.48550/ARXIV.2508.05618. arXiv: 2508.05618. url: https://doi.org/10.48550/arXiv.2508.05618.

- [47] DeepSeek-AI and collaborators. “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models”. In: arXiv preprint arXiv:2512.02556 (2025). DeepSeek-V3.2 thinking-enhanced reasoning model technical report; accessed 2026-01-29. url: https://arxiv.org/abs/2512.02556.
- [48] An Yang, Anfeng Li, Baosong Yang, et al. Qwen3 Technical Report. 2025. doi: 10.48550/ARXIV.2505.09388. arXiv: 2505.09388. url: https://doi.org/10.48550/arXiv.2505.09388.
- [49] AntAngelMed Team / Ant Group Health AI. AntAngelMed: Open-Source Medical Language Model. https://github.com/MedAIBase/AntAngelMed. Open-source medical reasoning model with MoE architecture and HealthBench-leading performance; accessed 2026-01-29. 2026.
- [50] Yubin Kim, Hyewon Jeong, Shan Chen, et al. “Medical Hallucinations in Foundation Models and Their Impact on Healthcare”. In: CoRR abs/2503.05777 (2025). doi: 10.48550/ARXIV.2503.05777. arXiv: 2503.05777. url: https://doi.org/10.48550/arXiv.2503.05777.
- [51] Itay Manes, Naama Ronn, David Cohen, Ran Ilan Ber, Zehavi Horowitz-Kugler, and Gabriel Stanovsky. “K-QA: A Real-World Medical Q&A Benchmark”. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing, BioNLP@ACL 2024, Bangkok, Thailand, August 16, 2024. Ed. by Dina Demner-Fushman, Sophia Ananiadou, Makoto Miwa, Kirk Roberts, and Junichi Tsujii. Association for Computational Linguistics, 2024, pp. 277–294. doi: 10.18653/V1/2024.BIONLP-1.22. url: https://doi.org/10.18653/v1/2024.bionlp-1.22.
- [52] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. “EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test”. In: CoRR abs/2503.01840 (2025). doi: 10.48550/ARXIV.2503.01840. arXiv: 2503.01840. url: https://doi.org/10.48550/arXiv.2503.01840.
- [53] Zihan Qiu, Zekun Wang, Bo Zheng, et al. “Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free”. In: CoRR abs/2505.06708 (2025). doi: 10.48550/ARXIV.2505.06708. arXiv: 2505.06708. url: https://doi.org/10.48550/arXiv.2505.06708.
- [54] Zihao Zheng, Xiuping Cui, Size Zheng, et al. “MoQa: Rethinking MoE Quantization with Multi-stage Data-model Distribution Awareness”. In: CoRR abs/2503.21135 (2025). doi: 10.48550/ARXIV.2503.21135. arXiv: 2503.21135. url: https://doi.org/10.48550/arXiv.2503.21135.
- [55] Wenhua Cheng, Weiwei Zhang, Haihao Shen, et al. “Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs”. In: Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024. Ed. by Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen. Vol. EMNLP

2024. Findings of ACL. Association for Computational Linguistics, 2024, pp. 11332–11350. doi: 10.18653/V1/2024.FINDINGS-EMNLP.662. url: https://doi.org/10.18653/v1/2024.findings-emnlp.662.

- [56] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers”. In: CoRR abs/2210.17323 (2022). doi: 10.48550/ARXIV.2210.17323. arXiv: 2210.17323. url: https://doi.org/10.48550/arXiv.2210.17323.
- [57] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. “Training Verifiers to Solve Math Word Problems”. In: CoRR abs/2110.14168 (2021). arXiv: 2110.14168. url: https://arxiv.org/abs/2110.14168.
- [58] Mark Chen, Jerry Tworek, Heewoo Jun, et al. “Evaluating Large Language Models Trained on Code”. In: CoRR abs/2107.03374 (2021). arXiv: 2107.03374. url: https://arxiv.org/abs/2107.03374.

- [59] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”. In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023. Ed. by Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine. 2023. url: http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-AbstractDatasets_and_Benchmarks.html.
- [60] Shenggui Li, Yikai Zhu, Chao Wang, et al. SpecForge: Train speculative decoding models effortlessly. https://github.com/sgl-project/specforge. 2025.
- [61] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. “SGLang: Efficient Execution of Structured Language Model Programs”. In: Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024. Ed. by Amir Globersons, Lester Mackey, Danielle Belgrave, et al. 2024. url: http://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-AbstractConference.html.

### A Appendix

This appendix presents supplementary experimental results and detailed analyses to further validate our proposed methods. Specifically, we provide ablation studies on the SPAR algorithm and the Clip-Forward-KL objective, along with extended evaluations of the Fact-Aware RL framework and the Gated Eegle-3 speculative decoding mechanism. For comprehensive details regarding the specific prompts utilized for training and evaluation, please refer to the associated GitHub repository.

#### A.1 Ablation Study of SPAR

0.99

1.0

25

0.95

0.90

21.7

20.9

20.4

NormalizedScore(0-1)

0.8

20

0.76

0.69

###### AverageTurns

0.67

0.65

0.63

0.6

15

0.55

0.51

0.47

10.4

0.4

10

0.28

0.2

5

0.0

0

Repeat Score Logical Score Rubrics Score Avg Turns

baseline grpo global penalty baichuan-m3

- Figure 11: Performance comparison between SPAR and baseline models in multi-turn medical consultation tasks.

To systematically evaluate the efficacy of the SPAR algorithm in multi-turn medical consultation tasks, we conducted ablation studies using the following experimental configurations:

- 1. Backbone: We use the Baichuan-M3 base model (before RL) as the foundational architecture.
- 2. GRPO (Global Reward): Employs the GRPO algorithm with global rewards based solely on final consultation outcomes, excluding intermediate feedback.
- 3. Global Penalty: Extends the GRPO baseline by incorporating a global repetition penalty to mitigate conversational redundancy.
- 4. SPAR (Baichuan-M3): Implements the proposed step-level reward calculation algorithm to provide granular guidance.

Metrics: The evaluation framework comprises four metrics normalized to the [0,1] interval: Repeat

Score (measuring non-redundancy), Logical Score, Rubrics Score, and Average Turns (Avg Turns). Specifically, the first two metrics are derived by employing GPT-5 to evaluate dialogue logs on a three-point scale (0,1,2), with the resulting scores subsequently normalized.

Results: As shown in Fig. 11, training with the GRPO algorithm using only global rewards yields a steady improvement in Rubrics Score, but it also increases redundant inquiries (as reflected by the decline in Repeat Score). Although adding a global repetition penalty effectively reduces redundancy, it causes a sharp drop in Logical Score, indicating severe logical fragmentation. These results suggest that coarse-grained global penalties can substantially disrupt the model’s natural reasoning flow. In contrast, SPAR achieves a better balance: it markedly reduces repetition while preserving logical coherence. This dual optimization allows the model to extract a higher density of critical medical information within a limited number of consultation turns.

#### A.2 Fact-Aware RL

##### A.2.1 Evaluation of Distilled Claim Extraction Models

While offline claim extraction relies on GPT-5 as the reference extractor, deploying such a large teacher model for online reinforcement learning is computationally prohibitive. To address this, we fine-tune smaller models via Supervised Fine-Tuning (SFT) to serve as efficient claim extractors during online RL. We evaluate extraction fidelity using three metrics relative to GPT-5:

- • Recall: Evaluates the coverage of the SFT model relative to the reference baseline (GPT-5).
- • SFT Exclusivity Rate: The proportion of claims identified by the SFT model that are absent in the GPT-5 reference, reflecting potential over-generation or discovery of novel insights.
- • GPT Exclusivity Rate: The proportion of claims identified by GPT-5 that remain uncaptured by the SFT model, indicating information loss.

Table 3: Comparative experiments on claim extraction models across different scales.

Model Recall (%) Our Exclusive Rate (%) GPT Exclusive Rate (%) Claim Count Qwen3-8B 30.45 31.02 69.55 8.50 Qwen3-32B 37.68 33.29 62.32 12.81 SFT-A3B-30B 69.69 46.89 30.31 47.19 SFT-8B 72.80 45.60 27.20 46.66 SFT-32B 73.00 47.15 27.01 46.78

As shown in Table 3, SFT substantially improves extraction performance, with the 8B model achieving 72.80% recall compared to 30.45% for the untuned baseline. While the 32B variant provides marginal performance gains, the significantly higher deployment cost is not justified for the online RL pipeline. We adopt the SFT-8B model as the claim extractor, balancing extraction fidelity with practical deployment constraints.

##### A.2.2 Ablation Study on Reward Components

We investigate the impact of different reward shaping strategies by comparing our method against two control groups starting from an intermediate SFT checkpoint: a w/o Fact Aware RL model optimizing only task rewards, and a Baseline utilizing static hallucination penalties in Eq. (9).

HealthBench Score

Hallucination Rate

0.68

w/o Fact Aware RL

0.08

Baseline

Denoise & Reweight

0.07

0.67

0.06

0.66

0.05

0.65

0.04

w/o Fact Aware RL

0.64

Baseline

0.03

Denoise & Reweight

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Steps

Steps

- Figure 12: Training dynamics of different optimization strategies. The plots contrast the impact of each objective on medical reasoning capability (Left) and factual reliability (Right).

Figure 12 highlights distinct behavioral patterns driven by these objectives. The w/o Fact Aware RL model (Grey) achieves the highest HealthBench score (∼0.68) but suffers from significant hallucination drift (increasing to 0.08). This observation suggests that an unconstrained optimization objective tends to bias the model toward generating expansive content, which may inadvertently compromise factual stability. The Baseline (Blue) corrects this drift but overcompensates; its aggressive suppression of hallucinations results in a severe degradation of reasoning capability, characteristic of penalty-induced conservatism.

Our Denoise & Reweight strategy in Eq. (14) effectively decouples safety alignment from capability loss. It achieves a hallucination reduction rate comparable to the Baseline (dropping to ∼0.035) while maintaining reasoning scores (∼0.665) close to the unconstrained model’s starting point.

This stability verifies the efficacy of our dual-protection design, where marginal noise filtering and competence-based gating work in tandem. Consequently, the approach successfully guides the model toward factual correctness while preserving its core medical utility.

#### A.3 Ablation Study on Clip-Forward-KL for Offline Expert Fusion

We analyze the role of Clip-Forward-KL in offline expert fusion. The student model is initialized from a medical inquiry expert and incorporates a healthcare expert through offline distillation. Both inquiry and healthcare data are distilled using either Clip-Forward-KL or standard Forward-KL, under identical initialization, datasets, and training configurations.

Table 4 reports results on ScanBench, HealthBench, and HealthBench-Hard. ScanBench primarily reflects the preservation of medical inquiry capability, while HealthBench and HealthBench-Hard assess the effectiveness of integrating broader healthcare expertise.

Table 4: Ablation on Clip-Forward-KL for offline expert fusion. Clip-Forward-KL improves HealthBench and HealthBench-Hard while maintaining comparable ScanBench performance under identical training configurations.

Method ScanBench HealthBench HealthBench-Hard

Forward-KL 73.7 58.6 33.2 Clip-Forward-KL 73.5 61.1 38.5

Clip-Forward-KL preserves inquiry performance while substantially improving healthcare benchmarks, indicating more effective fusion of new domain expertise. This suggests that standard Forward-KL tends to over-amplify probabilities on sparse offline samples, which interferes with the integration of new healthcare capabilities rather than the retention of existing inquiry ability. By enforcing a one-sided lower-bound constraint on teacher-supported actions, Clip-Forward-KL enables a more conservative fusion and yields an initialization for on-policy optimization.

#### A.4 Ablation Study on Gated Eagle-3 Speculative Decoding.

To ensure a rigorous and fair comparison, we evaluated the original Eagle-3 draft model against our proposed Gated Eagle-3 using an identical dataset of 35,000 training samples and the same training protocol. Both models were assessed under uniform inference configurations across a suite of representative benchmarks, including GSM8K [57], HumanEval [58], MT-Bench [59],

and HealthBench. All experiments were conducted on NVIDIA H20 GPUs, with model training performed using the SpecForge [60] framework and inference executed via SGLang [61]. As shown inTable5, GatedEagle-3achievesanaverageacceptancelengthimprovementof0.31overtheEagle-

- 3 base. Moreover, throughput comparisons conducted under a parallelism of 8 are presented in Table 6.

- Table 5: Comparison of average acceptance length between Eagle-3 Base and Gated Eagle-3. Method GSM8K HumanEval MT-Bench HealthBench

Eagle-3 Base 3.59 3.58 2.97 2.41 Gated Eagle-3 3.84 4.03 3.23 2.70

∆ (Gain) +0.25 +0.45 +0.26 +0.29

- Table 6: Comparison of throughput (tokens/s) between Eagle-3 Base and Gated Eagle-3. Method GSM8K HumanEval MT-Bench HealthBench

Eagle-3 Base 376.98 576.19 451.39 356.94 Gated Eagle-3 431.26 646.17 490.12 400.52

∆ (Gain) +14.40% +12.15% +8.58% +12.21%

