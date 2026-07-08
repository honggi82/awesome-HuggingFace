# arXiv:2511.22134v1[cs.CV]27Nov2025

#### DualVLA: Building a Generalizable Embodied Agent via Partial Decoupling of Reasoning and Action

Zhen Fang1∗ Zhuoyang Liu2∗ Jiaming Liu2† Hao Chen3 Yu Zeng1 Shiting Huang1 Zehui Chen1† Lin Chen1 Shanghang Zhang2 Feng Zhao1

1MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China 2 State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University 3 CUHK

∗ Equal Contribution. † Project Lead. Corresponding Authors.

[Figure 1]

Figure 1. DUALVLA first constructs a sparse, information-dense embodied reasoning dataset by combining video event prediction with kinematic cues, mitigating the negative impact of redundant reasoning on action generation. It then adopts a dual-teacher strategy: an action teacher offering fine-grained supervision for manipulation, and a reasoning teacher maintaining general reasoning capability. Together, these components enable DUALVLA to achieve strong performance in both simulation and real-world robotic evaluations.

###### Abstract

To build a generalizable Vision–Language–Action (VLA) model with reasoning capability, a common approach is to first train a specialist VLA on robot demonstrations to acquire reliable manipulation skills, and then introduce mixed annotated-robot data with multimodal data to restore general reasoning. However, we observe that the resulting reasoning VLA exhibits degraded action performance compared to the specialist VLA before fine-tuning. We define

this phenomenon as action degeneration. To tackle this issue, we propose DUALVLA, which improves action performance through carefully designed post-training while preserving the reasoning ability. We first propose a dual-layer data pruning method to remove redundant embodied reasoning and alleviate its adverse guidance on action learning. To further enhance the model’s action generation capabilities, we adopt a dual-teacher adaptive distillation strategy that assigns different supervision signals to different data domains and maintains its reasoning ability. To fill the

evaluation gap of generalist VLAs, we introduce VLA Score, which decouples VLA capabilities into reasoning, intention, action, and alignment, enabling a more fine-grained evaluation. Experiments show that DUALVLA achieves an average success rate of 61.0 in SimplerEnv and an average score of 65.4 across eight competitive multimodal benchmarks, demonstrating a stronger balance between action execution and multimodal understanding. Project Website.

###### 1. Introduction

An embodied agent is an AI system that perceives, reasons, and acts within complex environments, grounding intelligence in real-world interaction. Vision–Language–Action(VLA) models represent a key step toward such agents by leveraging the rich priors of Vision–Language Models (VLMs) and large-scale robotic datasets [42, 44, 57] to map observations and instructions to control signals. Recent advances [2, 23, 27, 35] show strong manipulation accuracy, generalization, and longhorizon capability, highlighting the promise of VLAs for embodied intelligence.

Previous specialist VLAs [23, 44, 48, 74] fine-tune a VLM on robotic datasets and thus achieve strong manipulation performance, though their multimodal reasoning ability remains limited. To enrich the reasoning capacity of VLAs, which helps models leverage VLM priors during manipulation and interpret complex scenes. Recent works augment robot trajectories with reasoning annotations and mix them with multimodal corpora during finetuning [7, 9, 52, 64, 67, 76]. This paradigm provides a practical route toward building reasoning VLAs. Building on this direction, we observe that enhancing a specialist VLA with additional reasoning may reduce its manipulation performance, a phenomenon we refer to as action degeneration. This suggests that reasoning and action rely on shared internal representations, and that reasoning-oriented supervision can inadvertently reshape the model’s visuomotor behavior. Importantly, this decline arises even when the training data becomes larger and more diverse. Such a deviation from the expected trend predicted by the scaling law [20] highlights a fundamental challenge: increasing data volume alone is insufficient unless the supervision signals for reasoning and action are properly balanced.

To address this issue, we propose DUALVLA, a reasoning VLA equipped with a dedicated post-training framework that enhances action performance while preserving multimodal reasoning capabilities. Our central insight is that action degradation arises from two factors: the misleading influence of repetitive, low-entropy embodied reasoning in the training data, and the lack of differentiated, fine-grained supervision for simultaneously learning reasoning and action. Guided by this insight, we first observe

that redundant embodied reasoning can bias the optimization process and deteriorate action execution. To mitigate this effect, we introduce a dual-layer data pruning strategy that leverages both embodiment cues and scene-level event changes to remove unnecessary reasoning segments while retaining action-critical content. To explicitly strengthen manipulation ability, we further utilize the specialist VLA as a natural provider of high-quality action supervision. Building on mix training, we develop a dual-teacher adaptive distillation strategy that assigns distinct soft-label supervision to robot data and multimodal reasoning data, enabling the model to learn both capabilities under balanced and fine-grained guidance. Finally, we note that existing VLAs evaluations do not disentangle reasoning from action and therefore cannot faithfully reflect the performance trade-offs observed in reasoning VLAs. To address this limitation, we introduce VLA Score, the first evaluation pipeline tailored for reasoning-capable VLAs. By incorporating a strong VLM as an evaluator, VLA Score provides comprehensive assessment along four dimensions—action, reasoning, intention, and reasoning–action alignment—and brings the MLLM-as-a-Judge paradigm into the evaluation of VLA systems.

In summary, our contributions are three-fold:

- • We propose DUALVLA, a reasoning VLA with a posttraining framework that decouples reasoning and action learning at the data and loss levels. Through dual-layer reasoning pruning and dual-teacher adaptive distillation, DUALVLA effectively preserves multimodal reasoning while improving manipulation ability.
- • We introduce VLA Score, the first evaluation framework tailored for reasoning VLAs. By leveraging a strong VLM as an assessor, VLA Score provides fine-grained evaluation across reasoning, intention, action, and reasoning–action alignment, offering deeper insight into model behavior and failure modes.
- • We conduct extensive experiments across simulation and real-world tasks, showing that DUALVLA consistently mitigates action degeneration and improves overall VLA performance. VLA Score further reveals important bottlenecks in current VLA development and validates the strengths of our approach.

###### 2. Related Work

Vision-Language-Action(VLA) Models. Leveraging the rich visual priors and zero-shot generalization capabilities of VLMs, VLAs fine-tune VLMs on robotic manipulation datasets [22, 44, 57] to produce action signals, emerging as a scalable and promising direction in embodied intelligence [23, 49, 75, 78]. OpenVLA [23] exemplifies this paradigm by finetuning Prismatic-7B [21] and mapping discrete tokens to action bins, while subsequent studies further explore architectural designs and action model-

- Table 1. Comparison of base model, data, reasoning capability, and action capability from VLMs to specialist VLAs and finally to reasoning VLAs. SVLAs denotes specialist VLAs and RVLAs denotes reasoning VLAs.

###### Base Model Data Reason Act

VLMs LLMs Corpora ↑ SVLAs VLMs Robot Data ↓ ↑ RVLAs SVLAs Both ↑ ↓

ing [2, 4, 25, 29, 35, 38, 39, 60]. However, such specialist VLAs typically sacrifice general multimodal and reasoning capabilities, motivating the development of reasoning VLAs. By distilling reasoning traces from advanced VLMs, recent works construct high-quality embodied CoT data for supervised finetuning [52, 56, 67]; for example, Emma-X [52] employs Gemini [54] to generate embodied CoT annotations for finetuning OpenVLA [23]. Incorporating multimodal corpora further enhances general understanding and reasoning [12, 24, 64]. Nonetheless, compared with pre-finetuned specialist VLAs, these reasoning VLAs consistently exhibit degraded action performance, indicating catastrophic forgetting not only in the VLM-toVLA transition but also in the specialist-to-generalist adaptation—ultimately hindering the development of truly generalizable embodied agents.

MLLM as a judge. In natural language processing, the LLM-as-a-Judge paradigm [73] leverages the strong priors and zero-shot generalization of large language models to evaluate model outputs through contextual prompts [11], achieving notable success in mathematical reasoning [51, 71], function calling [13, 47], agentic tasks [77], and other domains [8]. With the rise of R1-style models, assessing a model’s generated reasoning has become increasingly important [19, 43]. This motivates extending the paradigm to multimodal settings, which are inherently more complex and difficult to evaluate. Recent advances in VLMs [46, 58, 59, 68, 69, 72] and early explorations of MLLM-as-a-Judge [3] demonstrate its feasibility, further supported by systems like Llava-critic [62], which provide both scores and rationales. MLLM-based evaluation has also shown robustness in tasks such as spatial reasoning [18] and controllable image generation [6, 14, 15]. In contrast, existing VLA evaluations rely heavily on task success rates [28, 33, 42], overlooking reasoning quality and failing to capture beneficial trial-and-error behaviors. To address this gap, we introduce the MLLM-as-a-Judge paradigm into VLA evaluation. As VLMs continue to advance, we believe they will play an increasingly important role in evaluating VLA systems.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Figure 2. VLMs possess strong reasoning ability but lack action skills. Specialist VLAs achieve strong action capability but lose general reasoning. Reasoning VLAs partially recover reasoning through additional supervision, yet their action performance drops, illustrating the action degeneration problem. Our goal is to build a model that excels at both reasoning and action simultaneously.

###### 3. Method

In this section, we begin with the preliminaries of VLAs in Sec. 3.1. Then we introduce DUALVLA, which combines dual-layer pruning in Sec. 3.2.1 and dual-teacher distillation in Sec. 3.2.2 to balance reasoning and action learning. Finally, we present VLA Score in Sec. 3.3, a fine-grained evaluation framework for assessing both capabilities.

- 3.1. Preliminaries Given a robotic dataset [57] of expert trajectories

D = {τi}Ni=1, τi = {(ot,it,at)}T

i

t=1, (1)

where ot is the visual observation, it the instruction, and at ∈ Rd the action. A specialist VLA πθ(a | o,i) is trained via the autoregressive action loss:

Ls(θ) = −

τi t k

log πθ(at,k | ot,it,at,1:k−1), (2)

where at,1:k−1 are previously generated tokens.

For reasoning VLAs, each trajectory is augmented with reasoning (ot,it,rt,at). Multimodal samples share the unified form (o,i,r,∅), and the policy πθ(r,a | o,i) jointly predicts reasoning and actions using:

Lg(θ) = −

τi t k

log πθ(yt,k | ot,it,yt,1:k−1), (3)

where yt denotes the joint reasoning–action sequence.

- 3.2. DUALVLA

A general embodied agent should exhibit both strong multimodal reasoning and precise action prediction. The emergence of Action Degeneration directly contradicts this goal. Our analyses indicate that this degradation arises from two

factors: (1) redundant, low-entropy embodied reasoning, which dominates the training sequence and interferes with visuomotor learning, and (2) the absence of fine-grained, discriminative supervision needed to separate reasoning learning from action learning. These observations highlight the need to decouple the two processes and provide differentiated supervision.

To address Action Degeneration, we propose DUALVLA. Motivated by the negative impact of overfitted embodied reasoning, we introduce a two-stage pruning strategy that removes redundant reasoning while preserving action-relevant segments. To directly enhance action capability, we further adopt a dual-teacher framework: an action teacher offers fine-grained supervision for robot data, while a multimodal teacher maintains general reasoning ability. Together, the two teachers provide balanced, task-specific guidance that jointly improves action execution and multimodal understanding.

###### 3.2.1. Dual-Layers Data Pruning

Embodied reasoning is often treated as an implicit curriculum that enables the model to learn high-level reasoning before acquiring fine-grained action skills [7]. However, the reasoning tokens in embodied datasets are typically lowentropy, repetitive, and weakly coupled to the underlying visuomotor dynamics. This occurs because embodied scenarios (e.g., kitchen tasks) have limited object and action diversity (e.g., grasp, move, place), causing nearly identical reasoning text to be repeated across long temporal spans. During joint training, these abundant and homogeneous reasoning tokens dominate the loss and lead the model to overfit the linguistic reasoning patterns rather than learning the essential visuomotor relationships for manipulation. As a result, the action-relevant gradients are diluted or overridden, ultimately degrading the model’s action execution capability. To mitigate this undesirable supervision bias, we selectively prune redundant reasoning and retain only segments that are truly action-critical. Using advanced VLMs to annotate such segments is costly and difficult to scale [32], while relying solely on robotic kinematics ignores scene semantics and suffers from robustness issues [16]. Our duallayer pruning addresses both challenges by jointly considering scene event changes and action dynamics.

Inspired by perception–action coupling theory [10, 61], which posits that cognitive effort is allocated only when action demands and environmental changes jointly require additional reasoning, we retain only reasoning contents with high reasoning value by video event boundary detection and kinematic key-frame selection. For each frame, we assign a reasoning scene label and a reasoning action label. Keyframes are identified from two perspectives: video event changes and robot motion changes. Frames for which both labels are set to 1 are considered essential for reasoning. For video event boundary detec-

tion, we manually annotate a small set of trajectories with reasoning scene labels, indicating whether each step requires reasoning. Using high-quality annotated data, we retrained DDM-Net [53], which is designed for generic event boundary detection. Based on this well-pretrained detection net as the reasoning trigger, we obtain reasoning scene labels. For kinematic key-frame selection, we focus on moments where abrupt velocity changes or gripper state transitions occur. Let the end-effector pose be T(t) = [x(t),y(t),z(t),θx(t),θy(t),θz(t)]⊤ and the gripper state be G(t) ∈ {0,1}. The reasoning action label of the keyframe at time tk will be set to 1 if:

N

1 N

T ¨(ti)

T ¨(tk)

>

(4)

2

2

i=1

∨ lim

G(tk − ϵ) ̸= lim

G(tk + ϵ)

ϵ→0+

ϵ→0+

This approach is parameter-free compared with NoTVLA [16]. We mark frames whose scene and action labels are both 1 as keyframes, retain their reasoning contents, and mask all others. By jointly using sceneboundary detection and action-change cues, we prune redundant low-entropy reasoning and build a sparse, information-dense dataset. This selective retention naturally forms a curriculum: the model first consolidates action grounding, then learns reasoning only when necessary, preventing early overfitting and mitigating Action Degeneration.

###### 3.2.2. Dual-Teacher Adaptive Distillation

While Sec. 3.2.1 focuses on suppressing the negative effect of embodied-reasoning overfitting on action behavior, an orthogonal direction is to ask: can we directly improve the model’s action capability rather than merely preventing degradation? A key observation is that action degeneration fundamentally results from misaligned supervision signals: during mixed-modality finetuning, action tokens receive weak or noisy gradients, while reasoning tokens dominate the objective, leading the model to favor linguistic correctness over precise motor control. Inspired by the proverb ‘Turn inward and examine yourself when you encounter difficulties in life’, we revisit the source model from which the reasoning VLA is finetuned. Since specialist VLAs inherently possess strong action execution capabilities, their output distributions naturally provide fine-grained, actionaligned supervision that the reasoning VLA can benefit from. Thus, we introduce an action distillation loss that uses the specialist VLA as an action teacher to provide smooth, structured supervision over robot-centric samples:

Laction KD = T2DKL πθ

(a | o,i) πθ(a | o,i,r) . (5)

a

where T > 0 is the temperature parameter used to soften the probability distributions and πθ

denotes the action teacher.

a

Reasoning Action Alignment Intension

[Figure 2]

[Figure 3]

###### Input

[Figure 4]

Result

Knowledge Base

Task

[Figure 5]

[Figure 6]

[Figure 7]

Reasoning Score: 7/10

Tasks

Success: True/False

Task : Pick Coke Can

Put Open

Pick

[Figure 8]

[Figure 9]

Action Score: 4/10

Retrive

Push

Move Stack

[Figure 10]

Intension Score: 8/10

[Figure 11]

[Figure 12]

[Figure 13]

Trajectory

[Figure 14]

VLA Score

Cases

Pick

[Figure 15]

[Figure 16]

Retrive

Alignment Score: 3/10

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

##### …

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

I should evaluate the robot’s trajectory from four perspectives.

Reasoning

[Optional]

[Figure 26]

[Figure 27]

- • Intention: …
- • Alignment: …

- • Reasoning: …
- • Action: …

PLAN: The can is on the

table and I should close the gripper to secure the grasp.

Reasoning Score

Few-Shot

| | | |
|---|---|---|
| |Score Reasoning| |

Figure 3. Overview of VLA Score evaluation pipeline. Given the policy trajectory, task description, and optional reasoning as input, VLA Score first performs dual retrieval to fetch task-relevant textual examples and visually similar trajectories from a curated knowledge base. The retrieved samples serve as few-shot context for the VLM judge, which evaluates the trajectory along four dimensions: Reasoning, Action, Intention, and Alignment. These scores are then combined with the simulation outcome to produce the final VLA Score.

However, directly applying action distillation throughout training would cause the model to inherit the specialist VLA’s limited multimodal understanding ability. This reveals a second supervision-signal mismatch: while robot data require action-aligned supervision, multimodal reasoning data require reasoning-aligned supervision. To preserve multimodal competence, we therefore designate the finetuning initialization, which is already strong in general reasoning, as a reasoning teacher and introduce an analogous distillation objective:

alleviating action decay and leading to a more generalizable embodied agent.

VLM Score

3.3. VLA Score

At present, VLAs evaluation primarily relies on the success rate. However, this sparse and coarse-grained metric cannot capture critical aspects of VLAs performance, such as action smoothness, reasoning correctness, and the degree to which actions adhere to the inferred plan. To address this gap, we propose VLA Score, the first metric specifically designed for fine-grained VLAs evaluation. It is worth noting that VLA Score remains compatible with VLAs models lacking explicit reasoning capabilities and can be seamlessly adapted to various simulation environments. Fig. 3 shows the pipeline of VLA Score. Given the trajectory of policy πθ, task description, reasoning content (if available), and simulation success indicator, we prompt GPT-4o [17] to perform evaluations from four perspectives as follows:

Lreason KD = T2DKL πθ

(r | o,i) πθ(r | o,i) . (6) where πθ

r

denotes the reasoning teacher. Applying both teachers to all samples would be computationally expensive and exacerbate conflicts between heterogeneous supervision signals. Instead, the mixed-modality training regime naturally reveals which teacher to use: robot data receive action-aligned supervision, and multimodal reasoning data receive reasoning-aligned supervision. The final loss is:

r

- • Reasoning Score R: Measures the correctness, logical consistency, and usefulness of the reasoning process in guiding the agent toward successful task completion.
- • Action Score A: Measures the coherence and smoothness of the action sequence.
- • Intention Score I: Determines whether the model’s actions contribute constructively to solving the task.
- • Reason–Act Alignment Score RA: Measures how well the action sequences align with reasoning content.

Ltotal = LVLA + λLKD (7)

where Lkd = Laction KD, if robot data

Lreason KD, if multimodal data

LVLA corresponds to the VLA training loss using hardlabel cross-entropy and λ is the auxiliary weight and set to 0.15. By constructing this isomorphic dual-teacher adaptive distillation strategy and assigning supervision signals that match the nature of each data modality, the model learns both reasoning and action under fine-grained, aligned guidance. Compared to hard labels, soft supervision reduces gradient conflicts and restores the appropriate balance between linguistic reasoning and low-level control, thereby

Combined with the trajectory’s simulation result B, we calculate the overall VLAs Score using the following formula:

 

R + A · I

2 · RA · B, πθ ∈ RVLA A · I · RA · B, πθ ∈/ RVLA

VLA Score =

(8)



- Table 2. Comparison of manipulation success rates between DUALVLA and specialist & generalist baselines in SimplerEnv. Google Robot and WidowX Robot denote two embodiments in SimplerEnv. VM refers to visual matching and VA refers to variance aggregation. † denotes models without released checkpoints, results are taken from their papers.

Google Robot WidowX Robot

Methods

Avg Drawer Pick Can Move Near Put

Put Carrot

Stack

Spoon

Blocks VM VA VM VA VM VA

- RT-1-X [44] 59.7 29.4 56.7 49.0 31.7 32.3 0.0 4.2 0.0 26.8
- RT-2-X [44] 25.0 35.5 78.7 82.3 77.9 79.2 - - - Octo-Base [55] 22.7 1.1 17.0 0.6 4.2 3.1 15.8 12.5 0.0 7.0 RoboVLMs [34] 43.5 10.6 77.3 75.6 61.7 60.0 45.8 20.8 4.2 38.8 TraceVLA [74] 63.1 61.6 45.0 64.3 63.8 60.6 12.5 16.6 16.6 38.6 OpenVLA [23] 63.0 28.8 18.0 60.8 56.3 67.7 4.2 0.0 0.0 27.2 SpatialVLA [48] 57.4 41.8 86.0 88.0 77.9 72.7 16.7 25.0 29.2 45.9 InstructVLA-E [64] 47.2 60.6 87.7 76.0 68.3 77.3 45.8 20.8 20.5 56.0

Magma [63] 9.7 5.8 46.0 46.4 60.0 82.0 45.8 33.3 8.3 30.5 ECoT [67] 0.0 0.0 0.0 0.0 0.2 0.2 4.2 4.2 0.0 1.0 Emma-X [52] 18.3 20.5 2.3 5.3 3.3 7.3 0.0 0.0 0.0 6.3 ThinkACT† [12] 50.0 47.6 92.0 84.0 72.4 63.8 58.3 37.5 8.7 57.1 InstructVLA-G [64] 55.0 55.2 77.2 90.8 54.1 70.0 33.3 29.2 12.5 53.0

DUALVLA 63.9 64.0 93.3 82.7 60.9 75.3 50.0 50.0 8.3 61.0

1, if success 0, if fall

where B =

To improve evaluation accuracy and generalization, we adopt a retrieval-enhanced judge. We first collect 100 diverse trajectories and obtain preliminary scores using the previous pipeline. Human experts refine these results to form a VLAs knowledge base, which the judge retrieves during evaluation. For each input trajectory, we apply a dual-retrieval mechanism: task retrieval encodes the textual description and retrieves the most relevant task from the knowledge base, and scene retrieval encodes the image frames to fetch the most similar annotated trajectory as contextual reference. This retrieved context improves the VLM’s evaluation accuracy. We use text-embeddingada-002 1 as the text embedding encoder and CLIP ViTB/32 [45] as the image embedding encoder, both of which are commonly adopted in Retrieval-Augmented Generation (RAG) pipelines.

###### 4. Experiments 4.1. Main Results

Implementation details. We first evaluate on SimplerEnv [28], a widely adopted benchmark in the robotics community. By configuring visual matching and variance aggregation, SimplerEnv enables a more reliable assessment. We reconstruct the 650k VLA-IT [64] dataset following the procedure described in Sec 3.2.1 and subsequently fine-tune InstructVLA-G [64] on the recon-

1https://platform.openai.com/docs/models/text-embedding-ada-002

structed dataset. We designate InstructVLA-E as the action teacher and InstructVLA-G as the multimodal reasoning teacher. The learning rate is fixed at 2e-5. DUALVLA addresses Action Degeneration that emerges when transitioning from specialist VLAs to reasoning VLAs. To prove it, both types of VLAs are included in the comparison.:

- (1) specialist VLAs, including RT-1-X and RT-2-X from OXE [44], Octo [55], RoboVLMs [34], SpatialVLA [48], TraceVLA [74], InstructVLA-E [64] and OpenVLA [23].
- (2) reasoning VLAs, including ECoT [67], Emma-X [52], Magma [63], ThinkACT [12], and InstructVLA-G [64].

Quantitative Results. In Tab. 2, DUALVLA achieves an average success rate of 61.0 on the SimplerEnv benchmark. DUALVLA improves upon the baseline InstructVLA-G by 8.0 average success rate. This indicates that DUALVLA successfully resolved Action Degeneration. Compared with other prior methods, DUALVLA improves the mean success rate by 5.0 over the top-performing specialist VLA InstructVLA-E method and by 3.9 over the report result of top-performing reasoning VLA ThinkACT. Moreover, we observe an emergent teacher surpassing phenomenon, where DUALVLA achieves higher scores than the action teacher. And our pruning strategy yields about a 20% inference speedup over the baseline. Experimental results demonstrate that the model consistently improves performance across multiple tasks and settings, indicating that our carefully designed data pruning and dual-teacher adaptive distillation strategy effectively resolves the action decay phenomenon and further outperforms specialist VLAs.

[Figure 28]

Figure 4. Visualization of the two real-world task progress.

- Table 3. Quantitative results for real-world experiments. We report the success rate of moving the 3 objects separately for the long-horizon case. Bold indicates the highest score across all the models.

Method

Move Handover

Average O1 O2 O3 O1 O2 O3

InstructVLA [64] 60 60 50 60 40 40 45 Ours 90 80 70 80 60 50 60

- 4.2. Real-world Results

Real-world Setup. To systematically evaluate our approach, we designed two real-world dual-arm tasks on the Galaxea R1-lite robot. For the dual-arm setting, three RealSense 455 camera are used to get image observations, one on the head and two on the left and right wrist, respectively. The model takes the images of three views as image observation, and outputs a 14-DoF vector as the dual-arm action.

Self-collected Data. We designe two complex tasks:(1) Move Objects, (2) Handover Objects. Both tasks require the model to move three objects from right to left and follow the order in the language instruction. For each task, we collected 50 high-quality demonstration trajectories. Fig. 4 shows the progress of the two tasks.

Quantitative Results. We test 10 rollouts for each task and use the average success rate as the quantitative result. Tab. 3 shows that DUALVLA significantly improves manipulation performance, raising the average success rate from 45% to 60.0% in real-world tasks. The gains in both Move and Handover tasks demonstrate more reliable and coordinated action generation in real robotic settings.

###### 4.3. VLA Score Results

Can VLM evaluate VLA? Before analyzing VLA Score, a natural question should be clarified: can GPT-4o evaluate VLA capabilities? VLA Score evaluation can be formu-

Table 4. Comparison between DUALVLA and specialist VLAs on VLA Score. Sim. denotes the average success rate on SimplerEnv

###### Methods A I Sim. VLA Score

RoboVLMs-2B [34] 62.0 71.8 38.8 31.2 TraceVLA-7B [74] 53.3 62.4 38.6 23.2 OpenVLA-7B [23] 45.3 57.7 27.2 14.2 SpatialVLA-3B [48] 61.9 70.5 45.9 40.1 InstructVLA-E [64] 63.1 73.2 56.0 45.3

lated as an extension of video understanding. It involves recognizing objects, inferring required actions, determining whether the action and reasoning are consistent, and assessing whether the robot arm moves toward the target. Extensive prior work [1, 26, 30, 31, 70] has demonstrated that VLMs possess strong video understanding abilities.

Quantitative Results. In Tab. 4 and Tab. 5, DUALVLA achieves the highest score in VLA Score in reasoning VLAs. We observe that for specialist VLAs, the intention score is consistently slightly higher than the action score, indicating that the main bottleneck in their development lies in achieving smoother, more robust, and more efficient action modeling. For reasoning VLAs, the reasoning score is significantly higher than both the action and alignment scores. From analyzing failure cases, we find that the low action score primarily results from the model’s inability to execute effective actions throughout the trajectory. Even when it correctly reasons how to do, it often struggles to approach or manipulate the target. DUALVLA inherits the reasoning ability of the reasoning teacher while acquiring the teacher model’s more refined and smooth actions, which cross-entropy alone cannot achieve. We posit that this combination is a key factor behind DUALVLA’s effectiveness.

###### 4.4. Ablation Study

We still employ SimplerEnv for robotic manipulation evaluation, and assess multimodal tasks on MMMU [66],

- Table 5. Comparison between DUALVLA and reasoning VLAs.

Methods R A I RA Sim. VLA Score

ECoT [67] 47.2 27.9 39.9 29.6 0.8 0.0 Emma-X [52] 54.1 39.2 37.2 29.2 6.3 2.2 Magma [63] 66.0 61.0 71.2 36.0 30.5 19.5 Base [64] 72.3 48.6 68.0 43.3 53.0 35.1

Ours 74.0 64.8 74.7 51.0 61.0 42.9

- Table 6. Ablation of two strategies introduced in DUALVLA.

Setting GR WR MM Average

Base 67.1 25.0 65.5 52.5 Base(ft) 66.1 33.3 65.1 54.8 w.o. Pruning 72.4 29.2 65.0 55.6 w.o. Distillation 70.4 28.1 65.2 54.5 Ours 73.4 31.1 65.4 56.6

Table 7. Ablation for different purning methods. Setting GR WR MM Average

Ours 73.4 31.1 65.4 56.6 Random 63.6 23.1 64.6 50.4 w.o. Action 72.0 30.3 64.8 55.7 w.o. Scene 66.8 28.3 61.3 52.1

MM-Vet [65], MMStar [5], OCRBench [37], MMB [36], TextVQA [50], InfoVQA [41] and DocVQA [40] for all ablation studies. MM denotes the average of these multimodal understanding and QA benchmarks. GR denotes Google Robot set and WR denotes WidowX Robot. Owing to the limited default number of trials in WidowX (only 24 per task), we repeat every ablation experiment five times in the main paper, yielding 120 trajectories per task. Following a coarse-to-fine principle, we first perform macro-level ablations to evaluate the overall contributions of data pruning and distillation. As shown in Tab. 6, Base denotes the baseline InstructVLA-G, and Base(ft) denotes fine-tuning without data pruning and distillation strategies. Plain fine-tuning results in only marginal gains, showing diminishing returns. Teacher distillation significantly improves action execution across tasks while preserving multimodal capability. Data distillation further boosts performance by reducing the adverse influence of excessive embodied reasoning.

Pruning and Distillation. Tab. 7 shows that two-layer data pruning achieves better performance than single-layer pruning and random dropout. Scene pruning contributes the majority of the performance gain, while action pruning provides an additional complementary improvement. Moreover, the double-layer pruning strategy consistently outperforms proportional random dropout, showing that its effectiveness stems from the deliberate pruning design rather than randomness like ECoT-Lite [7]. For dual-teacher

Table 8. Ablation for distillation strategies. a t denotes the action teacher and r t denotes the reasoning teacher.

Setting GR WR MM Average

Ours 73.4 31.1 65.4 56.6 w.o. a t 67.8 26.1 65.2 53.0 w.o. r t 74.3 34.7 30.7 46.6

adaptive distillation, we primarily investigate whether the model can preserve its multimodal general capabilities while improving action performance. As shown in Tab 8, the dual-teacher adaptive design enables the model to simultaneously strengthen action generation capability and preserve general multimodal understanding. Although using only the action teacher yields a slight improvement in action performance, it comes at the cost of a substantial drop in multimodal capability, highlighting that mere action-specific supervision cannot maintain generalization. Fig. 5 shows that distillation from the reasoning teacher effectively preserves multimodal reasoning capabilities, with no significant difference compared to the reasoning teacher or the base VLM. This suggests that the current bottleneck in advancing reasoning VLAs lies in their action proficiency. Full distillation from a single teacher for all domain data reduces to a plain Kullback-Leibler (KL) Divergence loss, offering no performance gain.

Figure 5. Ablation for distillation.

Figure 6. Ablation for base models.

OpenVLA Variant To assess the extrapolation ability of DUALVLA, we extend our framework to OpenVLA [23] and ECoT [67], yielding OpenVLA-Dual. OpenVLA-Dual surpasses the action teacher (OpenVLA) in action performance and outperforms the reasoning teacher (ECoT) in multimodal generalization, confirming the practicality and broad applicability of our method. Although the action gains are modest, we attribute this to the inherent constraints of OpenVLA’s discrete-token action representation, which limits the ceiling of motion optimization.

###### 5. Conclusion, Limitations and Future Work

In this paper, we propose DUALVLA, which mitigates action degeneration through dual-layer pruning and dual-

teacher adaptive distillation, achieving stronger action execution while preserving reasoning. A limitation is the reliance on two teachers; although both are pretrained and introduce no extra training cost, this dependency still adds structural complexity. Another limitation is the increased forward passes required during distillation; although forward computation is not the main bottleneck in VLA training, it still adds overhead, and our preliminary attempt at attention-level distillation shows that similar benefits can be retained with reduced computation. Future work will focus on simplifying the distillation pipeline, reducing the reliance on teacher models, and extending the framework to broader embodied settings.

###### References

- [1] Anurag Arnab, Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid. Temporal chain of thought: Longvideo understanding by thinking in frames. arXiv preprint arXiv:2507.02001, 2025. 7
- [2] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom,

Karol Hausman, Brian Ichter, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2, 3

- [3] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge with vision-language benchmark. In Fortyfirst International Conference on Machine Learning, 2024. 3
- [4] Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, et al. Fast-in-slow: A dual-system foundation model unifying fast manipulation within slow reasoning. arXiv preprint arXiv:2506.01953, 2025. 3
- [5] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024. 8
- [6] Lan Chen, Qi Mao, Yuchao Gu, and Mike Zheng Shou. Edit transfer: Learning image editing via vision in-context relations. arXiv preprint arXiv:2503.13327, 2025. 3
- [7] William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, and Sergey Levine. Training strategies for efficient embodied reasoning. arXiv preprint arXiv:2505.08243, 2025. 2, 4, 8
- [8] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187, 2024. 3
- [9] Zhekai Duan, Yuan Zhang, Shikai Geng, Gaowen Liu, Joschka Boedecker, and Chris Xiaoxuan Lu. Fast ecot: Efficient embodied chain-of-thought via thoughts reuse. arXiv preprint arXiv:2506.07639, 2025. 2

- [10] Damian Farrow and Bruce Abernethy. Do expertise and the degree of perception—action coupling affect natural anticipatory performance? Perception, 32(9):1127–1139, 2003. 4
- [11] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024. 3
- [12] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, YuChiang Frank Wang, and Fu-En Yang. Thinkact: Visionlanguage-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815, 2025. 3, 6
- [13] Shiting Huang, Zhen Fang, Zehui Chen, Siyu Yuan, Junjie Ye, Yu Zeng, Lin Chen, Qi Mao, and Feng Zhao. Critictool: Evaluating self-critique capabilities of large language models in tool-calling error scenarios. arXiv preprint arXiv:2506.13977, 2025. 3
- [14] Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, et al. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv:2509.06945, 2025. 3
- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 3
- [16] Zheng Huang, Mingyu Liu, Xiaoyi Lin, Muzhi Zhu, Canyu Zhao, Zongze Du, Xiaoman Li, Yiduo Jia, Hao Zhong, Hao Chen, et al. Notvla: Narrowing of dense action trajectories for generalizable robot manipulation. arXiv preprint arXiv:2510.03895, 2025. 4
- [17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5
- [18] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135,

2025. 3

- [19] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025. 3
- [20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 2

- [21] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning, 2024. 2

- [22] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset,

2025. 2

- [23] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 3, 6, 7, 8
- [24] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025. 3
- [25] Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, et al. Cronusvla: Transferring latent motion across time for multi-frame prediction in manipulation. arXiv preprint arXiv:2506.19816, 2025. 3
- [26] Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. Intentqa: Context-aware video intent reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11963–11974, 2023. 7
- [27] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational visionlanguage-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650,

2024. 2

- [28] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea

- Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating realworld robot manipulation policies in simulation. In 8th Annual Conference on Robot Learning, 2024. 3, 6
- [29] Xiaoqi Li, Jingyun Xu, Mingxu Zhang, Jiaming Liu, Yan Shen, Iaroslav Ponomarenko, Jiahui Xu, Liang Heng, Siyuan Huang, Shanghang Zhang, and Hao Dong. Object-centric prompt-driven vision-language-action model for robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27638–27648, 2025. 3
- [30] Yanshu Li. Advancing multimodal in-context learning in large vision-language models with task-aware demonstrations. arXiv preprint arXiv:2503.04839, 2025. 7
- [31] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding visionlanguage model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025. 7
- [32] Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Junming Zhao, and Yang Gao. Onetwovla: A unified visionlanguage-action model with adaptive reasoning. arXiv preprint arXiv:2505.11917, 2025. 4
- [33] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 3
- [34] Huaping Liu, Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, and Hanbo Zhang. Towards generalist robot policies: What matters in building vision-language-action models. 2025. 6, 7
- [35] Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631, 2025. 2, 3
- [36] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 8
- [37] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 8
- [38] Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Xiangyang Xue, and Yanwei Fu. Trivla: A unified triple-system-based unified vision-language-action model for general robot control. arXiv preprint arXiv:2507.01424, 2025. 3
- [39] Zhuoyang Liu, Jiaming Liu, Jiadong Xu, Nuowei Han, Chenyang Gu, Hao Chen, Kaichen Zhou, Renrui Zhang, Kai Chin Hsieh, Kun Wu, et al. Mla: A multisensory language-action model for multimodal understanding and forecasting in robotic manipulation. arXiv preprint arXiv:2509.26642, 2025. 3
- [40] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceed-

- ings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 8
- [41] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 8
- [42] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3): 7327–7334, 2022. 2, 3
- [43] Minh-Vuong Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Phung, Yuan-Fang Li, Thuy-Trang Vu, and Gholamreza Haffari. Direct evaluation of chain-of-thought in multihop reasoning with knowledge graphs. arXiv preprint arXiv:2402.11199, 2024. 3
- [44] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 2, 6
- [45] Taesung Park, Alexei A Efros, Richard Zhang, and JunYan Zhu. Contrastive learning for unpaired image-to-image translation. In European conference on computer vision, pages 319–345. Springer, 2020. 6
- [46] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and Feng Zhao. Vcr-bench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 3
- [47] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789,

2023. 3

- [48] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025. 2, 6, 7
- [49] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844,

2025. 2

- [50] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019. 8
- [51] Andreas Stephan, Dawei Zhu, Matthias Aßenmacher, Xiaoyu Shen, and Benjamin Roth. From calculation to adjudication: Examining llm judges on mathematical reasoning tasks. arXiv preprint arXiv:2409.04168, 2024. 3

- [52] Qi Sun, Pengfei Hong, Tej Deep Pala, Vernon Toh, U Tan, Deepanway Ghosal, Soujanya Poria, et al. Emma-x: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning. arXiv preprint arXiv:2412.11974, 2024. 2, 3, 6, 8
- [53] Jiaqi Tang, Zhaoyang Liu, Chen Qian, Wayne Wu, and Limin Wang. Progressive attention on multi-level dense difference maps for generic event boundary detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3355–3364, 2022. 4
- [54] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 3
- [55] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 6
- [56] Tuan Van Vo, Tan Quang Nguyen, Khang Minh Nguyen, Duy Ho Minh Nguyen, and Minh Nhat Vu. Refinevla: Reasoning-aware teacher-guided transfer fine-tuning. arXiv preprint arXiv:2505.19080, 2025. 3
- [57] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023. 2, 3
- [58] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. ViDoRAG: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9124–9145, Suzhou, China, 2025. Association for Computational Linguistics. 3
- [59] Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. Vrag-rl: Empower vision-perception-based rag for visually rich information understanding via iterative reasoning with reinforcement learning. arXiv preprint arXiv:2505.22019, 2025. 3
- [60] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. arXiv preprint arXiv:2509.09372, 2025. 3
- [61] William H Warren Jr. The perception-action coupling. In Sensory-Motor Organizations and Development in Infancy and Early Childhood: Proceedings of the NATO Advanced Research Workshop on Sensory-Motor Organizations and Development in Infancy and Early Childhood Chateu de Rosey, France, pages 23–37. Springer, 1990. 4
- [62] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llavacritic: Learning to evaluate multimodal models. In Proceed-

- ings of the Computer Vision and Pattern Recognition Conference, pages 13618–13628, 2025. 3
- [63] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14203–14214, 2025. 6, 8
- [64] Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation. arXiv preprint arXiv:2507.17520, 2025. 2, 3, 6, 7, 8
- [65] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 8
- [66] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 7
- [67] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 2, 3, 6, 8
- [68] Yu Zeng, Wenxuan Huang, Shiting Huang, Xikun Bao, Yukun Qi, Yiming Zhao, Qiuchen Wang, Lin Chen, Zehui Chen, Huaian Chen, et al. Agentic jigsaw interaction learning for enhancing visual perception and reasoning in visionlanguage models. arXiv preprint arXiv:2510.01304, 2025. 3
- [69] Yu Zeng, Yukun Qi, Yiming Zhao, Xikun Bao, Lin Chen, Zehui Chen, Shiting Huang, Jie Zhao, and Feng Zhao. Enhancing large vision-language models with ultra-detailed image caption generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26703–26729, Suzhou, China, 2025. Association for Computational Linguistics. 3
- [70] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyusong Lee. Omagent: A multi-modal agent framework for complex video understanding with task divide-and-conquer. arXiv preprint arXiv:2406.16620, 2024. 7
- [71] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301, 2025. 3
- [72] Yiming Zhao, Yu Zeng, Yukun Qi, YaoYang Liu, Lin Chen, Zehui Chen, Xikun Bao, Jie Zhao, and Feng Zhao. V2pbench: Evaluating video-language understanding with visual prompts for better human-model interaction. arXiv preprint arXiv:2503.17736, 2025. 3
- [73] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan

- Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023. 3
- [74] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 2, 6, 7
- [75] Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, et al. A survey on visionlanguage-action models: An action tokenization perspective. arXiv preprint arXiv:2507.01925, 2025. 2
- [76] Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. arXiv preprint arXiv:2502.14420, 2025. 2
- [77] Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. Agent-as-a-judge: Evaluate agents with agents. arXiv preprint arXiv:2410.10934, 2024. 3
- [78] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 2

- A. VLA Score The prompts are shown in Fig. 15 and 16.
- B. Additional Qualitative Analyses

Data Pruning. Fig. 7 visualizes the embodied reasoning embeddings from robot datasets. We observe that embodied reasoning in robotic datasets is highly redundant and densely clustered. As illustrated in Fig. 8, although the visual scenes continually change across an action sequence, the associated reasoning remains nearly identical. For example, when a robot approaches an object, multiple consecutive frames correspond to the same reasoning, such as “Move Near”, despite perceptible motion in the visual input. This redundancy arises from the nature of embodied tasks: trajectories often contain many low-level movement steps but only a small number of distinct semantic intentions. As a result, different frames map to nearly identical reasoning statements, creating a low-entropy and tightly clustered distribution. Such concentrated reasoning offers limited additional supervisory value yet dominates the training signal, thereby biasing the model toward overfitting trivial embodied reasoning rather than improving manipulation. To mitigate the adverse effect of redundant embodied reasoning, we selectively prune repetitive samples rather than uniformly preserving the full corpus. Consider a training objective that combines action labels and reasoning supervision:

L = E ∗ (x,y) ∼ D ℓact(x,y) + ℓreason(x,y) (9)

When embodied trajectories contain highly repetitive states, the distribution D becomes skewed toward low-information reasoning segments, causing the expectation to overweigh ℓreason at these redundant points. In this scenario, minimizing L encourages the model to focus on predictable, lowentropy textual reasoning patterns instead of learning informative action–state relations. Consequently, the training bias suppresses the action objective ℓact, degrading the model’s motion capability. Pruning concentrated, repetitive reasoning samples rebalances D, enlarging the contribution of diverse state–action pairs and restoring meaningful supervision. Thus, pruning is not merely data reduction, but a necessary step to adjust the effective training distribution and preserve action learning.

Why does DUALVLA work? Training a reasoning VLA naturally involves optimizing two competing objectives, namely accurate action execution and robust multimodal reasoning. Let the corresponding goals be denoted as Lact, Lreason. Since their gradients usually point to different directions in parameter space, learning should not be viewed as minimizing a single scalar objective, but as

VLA-IT

ECoT

Emma-X

Bunny-2m

- 0

- 1

- 2

- 3

- 4

1

2

3

4

4

4

3

3

2

2

1

1

0

0

1

1

2

2

3

3

4

4

Figure 7. Embedding visualization of embodied reasoning and multimodal data. Robotic reasoning samples form dense, lowentropy clusters, whereas multimodal data remain more dispersed, indicating higher semantic diversity.

converging to a Pareto optimal solution, where no capability can be further improved without degrading the other. In practice, na¨ıvely mixing robot data with embodied reasoning drives optimization toward reasoning-dominant directions, because redundant and low-entropy reasoning traces generate disproportionately strong update signals. As a result, the model tends to converge toward the interior of the Pareto set, corresponding to suboptimal action performance despite moderate reasoning improvements. To mitigate this imbalance, our method suppresses over-represented reasoning signals through pruning while enforcing structured gradient alignment via dual-teacher supervision. Pruning attenuates the dominance of ∇Lreason, caused by repeated contextual traces, whereas expert-guided distillation maintains balanced gradients that avoid weakening ∇Lact. Together, these mechanisms steer optimization toward the Paretoefficient frontier, yielding a VLA that enhances manipulation performance without sacrificing general multimodal capability. This geometric perspective explains why pruning or distillation alone is insufficient, while their combination effectively leads to a more desirable Pareto-efficient balance between action and reasoning.

###### C. Case Study

###### C.1. Additional Visualization

Simulation. Fig. 9 and Fig. 10 show successful execution examples of DUALVLA on SimplerEnv.

Real-Task. Fig. 11 and Fig. 12 show successful execution examples of DUALVLA on real-world tasks.

“Move Right” “Move Down” “Finish”

[Figure 29]

[Figure 30]

[Figure 31]

“Move Near” “Move Near” “Move Near”

[Figure 32]

[Figure 33]

[Figure 34]

Figure 8. Example of redundant embodied reasoning across consecutive robot frames. Despite scene and motion changes, multiple frames share identical reasoning (e.g., “Move Near”), revealing redundancy that motivates pruning.

###### C.2. Additional Failure Case Analysis

The failure cases of DUALVLA, illustrated in Fig. 13 and Fig. 14, are primarily caused by insufficiently detailed action modeling, which can lead to imprecise or unstable execution, and the lack of historical context, which sometimes results in suboptimal or inconsistent decision-making. Additionally, a portion of failures arises from evaluation bias in the simulation environment, where tasks that are not perfectly completed are still marked as unsuccessful. These observations underscore the importance of fine-grained assessment, as provided by the proposed VLA Score.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Task: Put spoon on the towel.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Task: Put carrot on plate.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Task: Stack the green block on the yellow block.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Task: Close the bottom drawer.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Task: Open the middle drawer.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Task: Pick Coke Can.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Task: Pick Coke Can.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Task: Move orange near Pepsi can.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Task: Move 7up can near apple.

- Figure 9. Visual examples of SimplerEnv Google robot tasks driven by DUALVLA.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Task: Put spoon on the towel.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Task: Put carrot on plate.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Task: Stack the green block on the yellow block.

- Figure 10. Visual examples of SimplerEnv WidowX robot tasks driven by DUALVLA.

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Task: Close the bottom drawer.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Task: Open the middle drawer.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

## Task1-success1

Task: Pick Coke Can.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Task: pick the three objects from the right plate to the left plate, follow the order of pen, toy, banana.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Task: Pick Coke Can.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Task: Move orange near Pepsi can.

Task: pick the three objects from the right plate to the left plate, follow the order of toy, pen, banana.

## Task1-success2

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Figure 11. The successful cases of DUALVLA in real-world tasks.

Task: Move 7up can near apple.

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

###### Task: pick the three objects by the right arm and handover to the left arm, then put them

on the left plate, follow the order of toy, pen, banana.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Task: pick the three objects by the right arm and handover to the left arm, then put them on the left plate, follow the order of toy, pen, banana.

Figure 12. The successful cases of DUALVLA in real-world tasks.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Task: Stack the green block on the yellow block.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Task: Move 7up can near apple.

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Task: Open the bottom drawer.

Figure 13. The failure cases of DUALVLA in SimplerEnv.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

### Task1-fail （夹爪关闭高度不对）

Task: pick the three objects from the right plate to the left plate, follow the order of pen,

banana, toy.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

### Task2-fail （handover错位）

Task: pick the three objects by the right arm and handover to the left arm, then put them on

the left plate, follow the order of banana, toy, pen.

Figure 14. The failure cases of DUALVLA in real-world tasks.

VLA Score for Reasoning VLAs

###### System Prompt

Character Introduction You are an engineer proficient in evaluating robotic operations. Your task is to perform a fine-grained, rigorous evaluation of a robot’s behavior. Input Information You will receive the following four components:

- 1. Task Description — natural language description of the robot’s assigned task.
- 2. Motion Trajectory — a sequence of images representing the robot’s movements over time.
- 3. Reasoning Content — the robot’s moment-by-moment reasoning corresponding to the trajectory. Your Task Evaluate the robot’s performance along four dimensions, each scored on a 0–10 scale:

|Dimension<br><br>|Description|
|---|---|
|Reasoning Score<br><br>|Measures the correctness, logical consistency, and usefulness of the reasoning in guiding the task.|
|Action Score|Measures the coherence, precision, and efficiency of the action sequence in achieving the task.|
|Intention Score<br><br>|Evaluates whether the reasoning and actions constructively contribute toward solving the task.|
|Reason–Act Alignment Score|Measures the consistency between reasoning and corresponding actions, ensuring logical and behavioral alignment.|

Scoring Guideline: 10 = excellent; 5 = acceptable; 0 = poor or missing evidence. Evaluation Requirements

- • Be objective, thorough, and specific — do not overlook details or fabricate facts.
- • If the task or data is ambiguous, acknowledge uncertainty explicitly.
- • Mention any critical failure modes (e.g., collisions, unsafe motions, too slow).
- • What you are seeing is the complete process of a robotic arm performing a task. A task must be fully executed to be considered successful; if opening a drawer, it must be fully opened, and if closing a drawer, it must be fully closed.
- • Please provide a rigorous evaluation. Example Output Format Output must be strictly formatted in JSON and include your internal reasoning under "Thought".

{

"Thought": string, // Briefly describe your thought process and reasoning steps to evaluate the robot’s performance. "Result": {

"Action": int, "Intention": int, "Reasoning": int, "Alignment": int, "Success": bool [optional]

} }

###### User:

Task: Pick Coke Can; Reasoning: [Reasoning Contents]; Trajectory:[Images]

Figure 15. An example prompt of VLA Score for reasoning VLAs.

VLA Score for Specialist VLAs

###### System Prompt

Character Introduction You are an engineer proficient in evaluating robotic operations. Your task is to perform a fine-grained, rigorous evaluation of a robot’s behavior. Input Information You will receive the following four components:

- 1. Task Description — natural language description of the robot’s assigned task.
- 2. Motion Trajectory — a sequence of images representing the robot’s movements over time. Your Task Evaluate the robot’s performance along four dimensions, each scored on a 0–10 scale:

|Dimension<br><br>|Description|
|---|---|
|Action Score<br><br>|Measures the coherence, precision, and efficiency of the action sequence in achieving the task.|
|Intention Score|Evaluates whether the reasoning and actions constructively contribute toward solving the task.|

Scoring Guideline: 10 = excellent; 5 = acceptable; 0 = poor or missing evidence. Evaluation Requirements

- • Be objective, thorough, and specific — do not overlook details or fabricate facts.
- • If the task or data is ambiguous, acknowledge uncertainty explicitly.
- • Mention any critical failure modes (e.g., collisions, unsafe motions, too slow).
- • What you are seeing is the complete process of a robotic arm performing a task. A task must be fully executed to be considered successful; if opening a drawer, it must be fully opened, and if closing a drawer, it must be fully closed.
- • Please provide a rigorous evaluation. Examples Output Format Output must be strictly formatted in JSON and include your internal reasoning under "Thought".

{

"Thought": string, // Briefly describe your thought process and reasoning steps to evaluate the robot’s performance. "Result": {

"Action": int, "Intention": int, "Success": bool [optional]

} }

###### User:

Task: Pick Coke Can; Trajectory:[Images]

Figure 16. An example prompt of VLA Score for specialist VLAs.

