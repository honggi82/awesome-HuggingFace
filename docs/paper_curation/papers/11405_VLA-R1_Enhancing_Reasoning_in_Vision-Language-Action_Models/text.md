## VLA-R1: Enhancing Reasoning in Vision-Language-Action Models

#### Angen Ye12∗ Zeyu Zhang1∗ Boyuan Wang12 Xiaofeng Wang13 Dapeng Zhang2 Zheng Zhu1† 1GigaAI 2CASIA 3Tsinghua University

∗Equal contribution. †Corresponding author: zhengzhu@ieee.org.

# arXiv:2510.01623v1[cs.CV]2Oct2025

Abstract—Vision-Language-Action (VLA) models aim to unify perception, language understanding, and action generation, offering strong cross-task and crossscene generalization with broad impact on embodied AI. However, current VLA models often lack explicit stepby-step reasoning, instead emitting final actions without considering affordance constraints or geometric relations. Their post-training pipelines also rarely reinforce reasoning quality, relying primarily on supervised fine-tuning with weak reward design. To address these challenges, we present VLA-R1, a reasoning-enhanced VLA that integrates Reinforcement Learning from Verifiable Rewards (RLVR) with Group Relative Policy Optimization (GRPO) to systematically optimize both reasoning and execution. Specifically, we design an RLVR-based post-training strategy with verifiable rewards for region alignment, trajectory consistency, and output formatting, thereby strengthening reasoning robustness and execution accuracy. Moreover, we develop VLA-CoT-13K, a high-quality dataset that provides chain-of-thought supervision explicitly aligned with affordance and trajectory annotations. Furthermore, extensive evaluations on in-domain, out-ofdomain, simulation, and real-robot platforms demonstrate that VLA-R1 achieves superior generalization and realworld performance compared to prior VLA methods. We plan to release the model, code, and dataset following the publication of this work. Code: https://github. com/GigaAI-research/VLA-R1. Website: https:// gigaai-research.github.io/VLA-R1.

I. INTRODUCTION

Vision–Language–Action (VLA) models unify perception, language, and action. They first learn openvocabulary semantics and cross-modal alignment from internet-scale image–text pretraining. These semantics are then grounded into the action space through multitask manipulation data. This enables analogical transfer to unseen objects and compositional generalization to novel commands. Compared with modular pipelines [6], [41] or state-driven policies [9], VLAs show stronger cross-task and cross-scene generalization [35], [17], [18], [26], [16], [37]. Representative works include VoxPoser [14] for zero-shot trajectory planning, and ManipLVM-R1 [35] and RoboBrain [17] for integrating affordance perception and pose estimation. Meanwhile, Reinforcement Learning from Verifiable Rewards (RLVR) enhances reasoning and generalization in vision–language models. Vision-R1 [15] matches larger

[Figure 1]

Fig. 1. VLA-R1: pipeline from instruction to execution, with benchmark comparisons against baselines.

models through cold-start data and progressive training; LMM-R1 [30] employs a two-stage regimen from textual reasoning to multimodal tasks; and VLM-R1 [33] applies R1-style reinforcement to visual grounding, boosting open-vocabulary detection.

However, existing VLA models present two significant challenges. First, they often lack step-by-step reasoning: models tend to emit final actions directly

without explicit inference over affordance constraints, geometric relations, or container selection. This limitation leads to instruction-disambiguation failures under color similarity, duplicate instances, or multiple candidate receptacles. Second, post-training rarely provides systematic reinforcement of reasoning. Current method relies on supervised fine-tuning (SFT) with little reward optimization targeted at reasoning quality and execution efficacy. Even when Reinforcement Learning (RL) is used, reward design is typically single-objective and struggles to jointly optimize region alignment and trajectory consistency, degrading performance on out-ofdistribution data and in the real world.

To address these challenges, we propose VLA-R1, a post-training-enhanced VLA model capable of stepby-step reasoning. Unlike prior approaches, VLA-R1 simultaneously emphasizes data-level Chain-of-Thought (CoT) supervision and optimization-level reward alignment, bridging the gap between reasoning and execution. This enables the model to not only provide answers but also explain them, making it robust to challenges like color similarity, repeated instances, and multiple receptacle choices during reasoning.

To further enhance the model’s reasoning capabilities, we introduce an RLVR-based post-training strategy at the optimization layer. Specifically, we employ Group Relative Policy Optimization (GRPO) [32] with three verifiable rewards: an affordance reward based on Generalized Intersection over Union (GIoU) [31] to provide informative gradients for non-overlapping predicted and ground truth affordance regions, speeding up learning; a distance-based reward using the improved Fréchet distance to ensure reasonable trajectory curvature and segment length; and an output-format reward to enforce well-formed reasoning and action specifications. These optimizations enable VLA-R1 to generate accurate affordance regions and well-formed execution trajectories, enhancing decision-making.

Moreover, many existing datasets, although large in scale, fail to fully support complex reasoning tasks due to the lack of detailed explanations and reasoning processes in their annotations. To address this, we develop the VLA-CoT data engine, which generates the highquality VLA-CoT-13K dataset, making reasoning steps explicit. The engine aligns CoT with affordance and trajectory annotations, encouraging the model to learn task-consistent reasoning and enabling it to acquire basic reasoning capabilities during the SFT phase.

Finally, we conduct comprehensive evaluations of VLA-R1 across in-domain, out-of-domain, simulation, and real-robot settings. Empirically, VLA-R1 achieves an IoU of 36.51 on the in-domain affordance benchmark, a 17.78% improvement over the baseline; on the in-domain trajectory benchmark it attains an Average

TABLE I COMPARISON OF DATASETS ON AFFORDANCE, TRAJECTORY, REASONING, SCENES, AND ROBOTS. ✓ INDICATES THE DATASET INCLUDES THAT ANNOTATION; ✗ INDICATES IT DOES NOT. “#SCENES” COUNTS DISTINCT ENVIRONMENTS. “24+” FOR VLA-IT MEANS AT LEAST 24 KNOWN ENVIRONMENTS (FROM BRIDGEDATA V2) WITH ADDITIONAL RT-1 SITES NOT CONSOLIDATED. IF FOLLOWING THE OFFICIAL UMD RELEASE, SET “#SCENES” TO 3.

Dataset #Aff #Traj #Reasoning #Scenes #Robot UMD ✓ ✗ ✗ 4 VAIT ✗ ✓ ✗ — 13 VLA-IT ✗ ✓ ✓ 24+ 2 ShareRobot ✓ ✓ ✗ 102 12 VLA-CoT-13K ✓ ✓ ✓ 102 12

distance of 91.74 (lower is better), reducing the baseline by 17.25%. It also delivers state-of-the-art (SOTA) performance in the out-of-domain setting. On physical hardware, VLA-R1 reaches 62.5% success for affordance perception and 75% for trajectory execution. These results demonstrate the method’s effectiveness under controlled conditions and its robustness and practicality across domains and real-world scenarios.

Contributions in our paper can be summarized in the following three folds:

- • We propose VLA-R1, a VLA foundation model that VLA foundation model that introduces an RLVR optimization scheme with carefully designed rewards (region alignment, trajectory consistency, and output formatting), augmented by GRPO, to systematically strengthen reasoning and execution robustness while reducing reliance on manual annotation.
- • We introduce the VLA-CoT data engine, which produces high-quality VLA-CoT-13K aligned with affordance and trajectory labels and incorporates verifiable rewards, explicitly remedying the lack of step-wise reasoning in existing VLA models.
- • We comprehensively evaluate VLA-R1 on indomain and out-of-domain datasets, in simulation, and on real-robot platforms, empirically verifying its effectiveness and cross-domain generalization.

II. RELATED WORK A. VLA Models

Early manipulation research often relied on statebased reinforcement learning [8], [1], but these methods struggled with high-dimensional visual inputs. More recently, vision-centric approaches have become dominant, harnessing the reasoning capabilities of large language models (LLMs) to improve generalization [5],

[Figure 2]

Your reasoning should follow these four steps:

 Task parsing — What kind of physical interaction does the instruction imply?

When the task type is

 Scene understanding — What object is involved? What visual cues (e.g. folds, corners, graspable edges) support affordance identification?

- 1. Task parsing
- 2. Scene understanding
- 3. Affordance region localization
- 4. Feasibility check

[36], [39], [20], [24]. VoxPoser [14] uses visionlanguage models to generate 3D value maps, enabling zero-shot trajectory planning. RoboFlamingo [21] finetunes on manipulation datasets to perform languageconditioned tasks, while ManipLLM [20] incorporates chain-of-thought reasoning to integrate object understanding, affordance perception, and pose prediction into an interpretable framework. Building on this line, OpenVLA [18] and RoboMamba [24] leverage finegrained CoT data and supervised fine-tuning for further performance gains [34]. Other works, such as Embodied-Reasoner [42], Cosmos-Reason1 [2], and RoboBrain [17], focus on long-horizon reasoning, interpretability, and logical consistency in manipulation tasks. Despite progress, most approaches still depend on large-scale annotated datasets. In contrast, ManipLVMR1 [35] reduces reliance on supervision by combining small amounts of labeled data with RLVR-based selfimprovement, yielding robust generalization under limited supervision.

[Figure 3]

[Figure 4]

### Qwen2.5-VL-72B

 Affordance localization — Where is the predicted actionable region? Is it consistent with the instruction?

Affordance Trajectory

 Feasibility check — Can the robot perform the intended interaction at this region?

Datasets

Task parsing Task parsing

Scene understanding

Object and gripper localization

Your reasoning should follow these four steps:

[Figure 5]

 Task parsing — What is the intended manipulation behavior?

Affordance region localization

Scene analysis

 Object and gripper localization — Where is the target object? Where is the robot arm? Describe their spatial relationship.

Feasibility check

Trajectory logic analysis

When the task type is trajectory

CoT-Datasets

......

 Scene analysis — Are there any obstacles or constraints that affect motion planning?

- Fig. 2. CoT Data Engine. After ingesting multimodal data, the system parses tasks based on type (e.g., affordance or trajectory), performs scene understanding and localization, validates feasibility, and generates structured CoT traces for training.

and generalization. While these works highlight RLVR’s potential, their scope remains limited to non-embodied domains. To bridge this gap, ManipLVM-R1 [35], adapts RLVR to robotic manipulation, enhancing both reasoning and action execution.

III. METHOD

- A. Overview

The overall architecture of VLA-R1 is shown in Fig. 3. Given an input image and a natural language instruction, VLA-R1 encodes multimodal information through a vision-language backbone and then produces low-level control signals via an action decoder. Specifically, the vision branch processes raw images through a visual encoder that projects features into a shared embedding space. In parallel, the language branch tokenizes and embeds the task instruction. Both modalities are fused in the multimodal decoder, which jointly reasons over visual cues, textual context, and temporal history to generate a structured output consisting of a reasoning segment and an action prediction. The reasoning trace makes intermediate steps explicit, while the action output is represented in a discrete token space. Finally, the action de-tokenizer maps the predicted tokens into continuous 7D robot actions (∆x, ∆θ, and ∆Grip), which can be directly executed on the robot arm. This design allows VLA-R1 to bridge high-level task descriptions with grounded low-level control, while maintaining interpretability through explicit reasoning traces.

- B. Data Synthesis

 Trajectory logic — Why is this path chosen? Is it straight, curved, smooth? How many points, what spacing?

Trajectory logic

Trajectory

- 1. Task parsing
- 2. Object and gripper localization
- 3. Scene analysis
- 4. Trajectory logic

- B. RLHF for VLMs

Taskparsing

Large vision-language models (LVLMs) have demonstrated remarkable reasoning capabilities across diverse visual tasks [23], [22], [19], [3], [38], [11], [13], [40], [25]. Recently, reinforcement learning with verifiable rewards (RLVR) has emerged as a promising way to enhance their reasoning abilities [12], [7]. For example, Vision-R1 [15] leverages a cold-start math dataset and Progressive Thinking Suppression Training to achieve results comparable to much larger models without relying on human annotations. LMM-R1 [30] adopts a two-stage framework, first refining reasoning on textual data and then extending to multimodal and agent-based reasoning tasks. Similarly, VLM-R1 [33] applies an R1style reinforcement learning approach to visual grounding, improving open-vocabulary detection and generalization. While these works highlight RLVR’s potential, their scope remains limited to non-embodied domains. To bridge this gap, ManipLVM-R1 [35], adapts RLVR to robotic manipulation, enhancing both reasoning and action execution. LVLMs have demonstrated remarkable reasoning capabilities across diverse visual tasks [23], [22], [19], [3], [38]. Recently, RLVR has emerged as a promising way to enhance their reasoning abilities. For example, Vision-R1 [15] leverages a cold-start math dataset and Progressive Thinking Suppression Training to achieve results comparable to much larger models without relying on human annotations. LMM-R1 [30] adopts a two-stage framework, first refining reasoning on textual data and then extending to multimodal and agent-based reasoning tasks. Similarly, VLM-R1 [33] applies an R1-style reinforcement learning approach to visual grounding, improving open-vocabulary detection

To further strengthen the reasoning ability of our model, we construct a CoT dataset using Qwen2.5-VL72B. As shown in Table I and Figure 2, we employ the model to automatically generate intermediate reasoning steps for both affordance and trajectory tasks. In total, 13K CoT annotations are produced, which serve as high-quality supervision to bridge perception and

[Figure 6]

- Fig. 3. Overall architecture of VLA-R1. Training has two stages: Stage 1 uses SFT with CoT supervision to learn reasoning over images and instructions; Stage 2 refines reasoning and actions via RL with verifiable rewards (GRPO). During inference, a control stack converts outputs into joint-level robot commands.

action. These CoT data not only enrich the semantic interpretability of the training corpus but also provide explicit step-by-step guidance, enabling the model to learn task-consistent reasoning patterns.

- C. Supervised Fine-Tuning

We perform supervised fine tuning on our synthetic high quality VLA-CoT-13K dataset, which presents step by step think chains paired with grounded visual evidence and action targets. Compared with naive question and answer instruction tuning, chain of thought provides intermediate supervision signals that encourage explicit decomposition, stronger visual grounding, and stable credit assignment across time. This produces policies that reason before acting, which improves sample efficiency and prepares the model for subsequent post training under verifiable rewards. In practice we supervise both the structured <think> segment and the final <output> or action segment, which regularizes reasoning style, reduces hallucination, and yields more reliable action decoding under long horizon inputs.

We initialize our foundation model with Qwen2.5VL-3B [4]. The vision pathway is a redesigned Vision Transformer with window attention and 2D RoPE that supports native input resolution and dynamic frame rate sampling for videos. Visual tokens are softly compressed by an MLP merger before being fed into the language decoder. The text side adopts the Qwen2.5 tokenizer with a large vocabulary and the standard Qwen2.5 decoder stack. On top of the multimodal decoder we

attach an action decoder that we implement to map hidden states to control outputs for downstream tasks. This initialization provides a strong balance of accuracy and efficiency for long temporal contexts.

D. Reinforcement Learning

After SFT, we further optimize VLA-R1 through RL, as shown in Fig. 3. We adopt the GRPO algorithm, recently proposed by DeepSeek [10], [32] as a scalable variant of RLHF. We extend this approach to multimodal action reasoning, allowing the model to benefit from structured verifiable rewards while maintaining training stability. For input q, GRPO samples {o1,...,on} from πold, scores each with a reward function to get rg. Normalize via intra-group mean r¯ and std σr: Aˆg = (rg −r¯)/σr. For process supervision, step-wise rewards are normalized similarly, with token-wise advantages accumulated and shared across outputs. For the k-th token of the g-th output, the new/old policy probability ratio is:

πθ(og,k | q,og,<k) πold(og,k | q,og,<k)

. (1) GRPO’s objective:

rg,k(θ) =

|og|

n

1 |og|

min rg,k(θ)Aˆg,k,

LGRPO(θ) = −

g=1

k=1

clip rg,k(θ), 1 − ε, 1 + ε A ˆg,k

− β DKL πθ ∥ πref .

(2)

[Figure 7]

- Fig. 4. Case Analysis: The figure illustrates VLA-R1’s reasoning process and outcomes for both affordance and trajectory tasks. VLA-R1 parses the action requirements, infers relevant objects and spatial relations, and outputs the corresponding bounding boxes or waypoint sequences. The affordance form and trajectory form are fixed prompt templates that instruct the model to produce outputs in a specified format.

where clip(·) bounds the ratio to [1 − ε,1 + ε], and the last term is a KL penalty to avoid excessive policy drift.

Fréchet Trajectory Reward. The primary reward measures alignment using Angle-Length Augmented Fréchet distance (ALAF). Unlike pointwise Euclidean losses, ALAF respects the temporal ordering of the curves and augments it with local geometry. We represent each trajectory as a sequence of triples T = {pi,ti,ℓi}, where pi is the 2D waypoint (normalized image coordinates), ti is the unit motion direction at pi (forward/backward difference at endpoints and a normalized blend of adjacent segment directions for interior vertices), and ℓi is the local segment length (distance to the next waypoint; for the last vertex, to the previous one). ALAF combines the positional discrete Fréchet term with an angle penalty between unit tangents and a scale penalty based on the log ratio of neighboring segment lengths, weighted by λθ and λr; see Eq. (3).

DALHF(T,Tˆ ∗) = min

∥pˆi − p∗j∥2 position

max

Φ

(i,j)∈Φ

∗ j

+ λθ arccos t ˆit

##### +λr log(ℓˆi/ℓ∗j)

∥tˆi∥ ∥t∗j ∥ angle

length ratio

,

(3)

where Φ denotes all order-preserving couplings between the sequences. Tˆ = {pˆi,tˆi,ℓˆi} denotes the groundtruth trajectory. T∗ = {p∗i ,t∗i ,ℓ∗i }denotes the predicted trajectory. The trajectory reward is defined as

##### Rtraj = 1 − D˜ALAF(T,Tˆ ∗). (4)

Here, D˜ALAF denotes the ALAF distance normalized to [0,1]; larger Rtraj indicates better alignment.

GIoU Affordance Reward. For spatial grounding, we introduce a GIoU [31] reward between predicted and ground-truth bounding boxes. While IoU only considers the overlapping region, GIoU additionally accounts for

[Figure 8]

Fig. 5. Visualization of evaluation in real-world scenarios.

the smallest enclosing box, penalizing misaligned predictions even when boxes do not overlap. This improves spatial robustness, especially in cluttered environments where partial overlaps are common:

RGIoU = GIoU(ˆb,b∗). (5)

Format Reward. Finally, we enforce structural correctness with a format reward. The model must output responses that follow the required structure (<think>...</think> reasoning segment followed by a <output>...</output> action segment). The format reward is binary:

Rformat =

1 if the output adheres to format, 0 otherwise.

(6)

This encourages interpretable reasoning traces and prevents degenerate outputs during post-training.

IV. EXPERIMENT

To rigorously evaluate the effectiveness and generalization capacity of the proposed approach, we conduct experiments across 4 settings: in-domain datasets, outof-domain datasets, simulation environments, and realrobot platforms. We compare with strong baselines and ablate each component to show its impact.

- A. Dataset and Metrics

1) In domain datasets: All baseline models and our proposed VLA-R1 are trained on the ShareRobot dataset[17], a large-scale corpus purpose-built to advance affordance perception and trajectory prediction. ShareRobot is curated from 23 selected datasets within Open X-Embodiment[29], spanning 12 robotic embodiments, 102 manipulation scenarios, and hundreds of primitive actions; it undergoes multiple rounds of human

auditing to ensure high image resolution, successful task execution, accurate annotations, and complete, clean trajectory traces. The corpus comprises over one million planning question–answer pairs, 6,522 images with affordance annotations, and 6,870 images with trajectory annotations. In our experiments, we restrict training to the affordance- and trajectory-annotated image subsets and, on this basis, synthesize CoT rationales for model training.

- 2) Out of domain datasets: To assess generalization,

we conduct out-of-domain (OOD) evaluations. For affordance perception, we adopt a subset of the UMD Part Affordance dataset[27] as the OOD benchmark. UMD spans 105 tools commonly encountered in gardening, kitchen, and workshop contexts. We construct our OOD test set by randomly sampling 1,200 examples from four affordance categories—grasp, cut, pound, and scoop. For trajectory prediction, we evaluate on VAIT, the validation split of LLARVA’s pretraining corpus[28]. As VAIT originates from the highly diverse Open X-Embodiment collection, we select 500 samples and manually rectify trajectories exhibiting excessive deviation to ensure a fair and meaningful evaluation.

- 3) Metric Setting: For affordance perception, we

adopt Intersection over Union (IoU) as the principal metric. IoU quantifies spatial localization fidelity by measuring the overlap between predicted and groundtruth regions; higher values indicate more accurate detection and alignment. For trajectory prediction, we evaluate the concordance between predicted and groundtruth trajectories. Following prior work[17], [35], a trajectory is represented as an ordered set of 2D waypoints normalized to the interval [0, 1000). Similarity is assessed using three complementary metrics: Discrete Fréchet Distance (DFD), capturing global shape and

[Figure 9]

Fig. 6. Visualization of simulation.

temporal alignment; Hausdorff Distance (HD), measuring the maximum pointwise deviation; and Root Mean Square Error (RMSE), quantifying the average pointwise error. Together, these metrics furnish a comprehensive assessment across global shape, worst-case discrepancy, and average error, characterizing both the accuracy and consistency of trajectory prediction.

In both real-world and simulated evaluations, we report Success Rate (SR) as the task-level metric, defined as the ratio of successful executions to total trials. Success is determined as follows: Affordance tasks: a trial is deemed successful if an object is present in the scene, the predicted bounding box correctly localizes the target object, and the system successfully grasps it; if no object is present, the model should emit no bounding box, which is likewise counted as success. Trajectory tasks: a trial is deemed successful if the executed trajectory terminates within the designated goal location (or region) and the target object is reliably transported to that endpoint.

- B. Experiment on Benchmark

Implementation Details. To ensure a fair comparison, we curate a contemporary suite of baselines. Specifically, we evaluate Gemma-3-12B-it, Gemma-3-27B-it, Phi-4multimodal-Instruct, and the Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-32B-Instruct. All open-source models are assessed under few-shot prompting to furnish a minimal perception prior. To validate the effectiveness of our training framework, we further include supervised fine-tuning baselines—InternVL2-2B , LLaVA-1.6-7B, RoboBrain-7B , and NORA-3B—as well as an RL posttrained model, ManipLVM-R1-3B.

Experiment Results. As shown in the table III, opensource multimodal instruction-following models perform poorly on the in-domain dataset: despite large parameter counts, IoU remains below 10, and trajectory errors (DFD, HD, RMSE) are uniformly high. This indicates that generic models alone are inadequate for the precision demands of embodied tasks. Supervised finetuning (SFT) yields clear gains—e.g., RoboBrain-7B and NORA-3B attain higher IoU and lower trajectory errors than open-source baselines—yet their IoU typically remains in the 5–25 range. By contrast, VLA-

TABLE II SIMULATION EVALUATION ON DIFFERENT ROBOT PLATFORMS. Model Task Piper UR5 avg VLA-R1

affordance 60% 50% 55% trajectory 80% 60% 70%

affordance 50% 30% 40% trajectory 10% 0% 5%

NORA

R1-3B achieves the best results across all metrics: IoU = 36.51, DFD = 106.2, HD = 97.9, and RMSE = 71.12. Relative to the strong baseline ManipLVM-R1, IoU improves by 17.78%, and the overall trajectory error is reduced by 17.25%, attesting to the effectiveness of our training paradigm.From the OOD results, despite substantial distribution shifts, VLA-R1-3B remains superior on trajectory prediction: IoU increases to 33.96, while DFD, HD, and RMSE decrease to 114.3, 98.43, and 68.97, respectively—surpassing the strongest baseline, ManipLVM-R1-3B, and demonstrating strong cross-domain generalization and robustness. For the two types of tasks, the analysis process of VLA-R1 can be seen in Fig 4.

C. Experiment on Simulation

Implementation Details. To assess performance under controlled yet stochastic conditions, we conducted additional experiments in a simulated tabletop environment. Using the RoboTwin simulator, we instantiated a single randomized tabletop clutter generator that dynamically varies object categories, colors, poses/positions, and the table color throughout each trial. To evaluate the crossrobot generality of VLA-R1, we tested two robotic embodiments—Piper and UR5. Each embodiment was evaluated over ten independent trials with randomized initialization.

Experiment Results. Because our training data are drawn entirely from real-world settings, the simulated environment exhibits greater variability; nevertheless, as shown in Table II anf Fig. 6, VLA-R1 attains strong performance on both tasks. For affordance perception, VLA-R1 achieves 6/10 successes on Piper and 5/10 on UR5 (average SR = 55%). For trajectory execution, performance improves to 8/10 on Piper and 6/10 on UR5 (average SR = 70%), indicating that once a reliable grasp is established, the trajectory policy remains highly stable in simulation. By contrast, the NORA baseline performs notably worse under the same conditions: on the affordance task, SR drops to 50% (Piper) and 30% (UR5); on the trajectory task, it records 1/10 on Piper and 0/10 on UR5. Overall, these results confirm that VLA-R1 preserves robust cross-embodiment stability and superior generalization, even under heightened environmental variation.

TABLE III IN-DOMAIN AND OUT-OF-DOMAIN PERFORMANCE COMPARISON.

Method In-Domain Out of Domain

IoU ↑ DFD ↓ HD ↓ RMSE ↓ Avg ↓ IoU ↑ DFD ↓ HD ↓ RMSE ↓ Avg ↓ Open-source Models

Phi-4-multimodal-Instruct 0.58 243.92 224.73 189.27 228.21 2.17 240.18 235.44 202.69 226.10 Gemma-3-12b-it 1.18 206.72 190.64 154.96 184.10 4.65 204.94 209.88 175.42 193.75 Gemma-3-27b-it 1.32 257.42 230.29 184.47 224.09 8.20 232.86 268.03 209.25 250.67 Qwen2.5-VL-3B-Instruct 6.15 208.02 179.12 144.14 175.37 23.96 211.80 205.00 140.49 250.67 Qwen2.5-VL-32B-Instruct 7.40 125.54 113.00 85.05 107.86 25.14 182.73 176.51 133.17 164.14

###### Supervised Fine-Tuning

LLaVA-1.6-7B 3.98 184.40 178.00 133.28 165.23 5.90 170.88 167.10 160.79 166.25 InternVL2-2B 6.74 250.20 239.34 194.74 228.09 15.25 165.98 167.84 145.64 157.50 RoboBrain-7B 11.79 156.10 136.52 106.71 133.11 22.00 220.94 214.14 173.02 202.70 NORA-3B 23.48 139.65 126.76 92.97 120.45 22.44 154.81 129.84 95.65 126.77

Supervised Fine-Tuning + Reinforcement learning ManipLVM-R1-3B 31.00 134.18 111.14 87.28 110.87 28.00 146.82 140.52 108.64 131.99 VLA-R1-3B 36.51 106.20 97.90 71.12 91.74 33.96 114.30 98.43 68.97 93.90

- D. Experiment on the Real World

Implementation Details. To comprehensively assess real-world performance, we design four canonical scenarios on a tabletop platform. We instantiate: S1: Bowl picking, containing bowls of multiple colors placed in diverse locations; the model must grasp the userspecified color and, for trajectory tasks, place it precisely into a designated frame/basket of a given color.

- S2: Fruit picking, featuring repeated instances of the same fruit; the model must disambiguate and grasp the specified item and, for trajectory tasks, place it into the basket or onto the plate indicated by the instruction.
- S3: Kitchen scenario, comprising an open microwave, plates, and food props, where the model must contend with visual occlusion from the door and the spatial constraints of the cavity. S4: Mixed scenario, in which bowls, produce, baskets, and plates co-occur, requiring grasp-and-place under multi-category, multi-attribute distractors. Each scenario is evaluated over ten independent trials; we randomize initial object placements and poses and shuffle scenario order to mitigate potential ordering effects.

Experiment Results. As shown in Table IV and Figure 5, VLA-R1 achieves an average Success Rate (SR) of 62.5%˜ across the four scenarios for affordance perception. By contrast, trajectory prediction attains a higher SR of 75%. The NORA-3B baseline records 35%˜ on affordance perception and 47.5% on trajectory prediction. We observe that distractors such as color similarity and positional variation materially affect the model’s decisions, constituting the primary sources of error. Nevertheless, even under heavy clutter, VLA-R1’s pre-

TABLE IV REAL-WORLD EXPERIMENTS Model Task S1 S2 S3 S4 avg VLA-R1

affordance 80% 60% 70% 60% 62.5% trajectory 60% 80% 80% 80% 75%

affordance 40% 30% 30% 40% 35% trajectory 40% 50% 30% 70% 47.5%

NORA

dictions typically concentrate near the target rather than diverging arbitrarily, indicating a degree of tolerance and self-correction in perception and trajectory generation; when the target cannot be fully locked, the model still preserves reasonable local spatial consistency. Overall, these results validate the method’s practical viability in real settings and its ability to maintain stability under attribute similarity and visual clutter.

E. Ablation Study

To rigorously assess the impact of Chain-of-Thought (CoT) reasoning and Reinforcement Learning (RL) on performance, we conduct an ablation study with three configurations: (1) without CoT and RL (w/o CoT & RL); (2) CoT only; and (3) CoT+RL. All models are trained under identical hyperparameters to ensure a fair comparison.

From the table V, using CoT alone—relative to the configuration without CoT or RL—raises IoU from 23.74 to 28.37 and reduces the average distance metric from 128.38 to 124.6. The improvement is more pronounced for IoU, indicating that CoT confers a degree of attribute disambiguation and thus benefits affordancecentric tasks. When combined with RL, the model

TABLE V ABLATION STUDY ON THE EFFECT OF COT REASONING AND RL. HIGHER IOU IS BETTER, LOWER DFD/HD/RMSE/AVG ARE BETTER.

Method IoU ↑ DFD ↓ HD ↓ RMSE ↓ Avg ↓ w/o CoT & RL 23.74 149.38 135.72 100.04 128.38 CoT only 28.37 145.51 131.26 97.03 124.60 CoT + RL 36.51 106.20 97.90 71.12 91.74

achieves substantial gains across all metrics, underscoring the complementarity of CoT and RLVR in trajectory prediction: CoT provides structured task decomposition and reasoning, while RLVR leverages reward signals to refine execution policies, producing significant end-toend performance improvements.

V. LIMITATION AND FUTURE WORK

While VLA-R1 demonstrates strong performance across benchmarks, simulation, and real-robot settings, a key limitation is that it has not yet been developed or validated on other types of robotic platforms such as bimanual robot arms and quadruped robot dogs. Extending VLA-R1 to these embodiments represents an important direction for future work, enabling broader applicability and testing its generalization in more diverse real-world scenarios.

VI. CONCLUSION

In this work, we introduced VLA-R1, a reasoningenhanced Vision–Language–Action model that integrates chain-of-thought supervision with reinforcement learning from verifiable rewards. By designing the VLACoT-13K dataset and incorporating an RLVR-based post-training strategy, VLA-R1 explicitly strengthens both step-by-step reasoning and execution robustness. Comprehensive experiments across in-domain, out-ofdomain, simulation, and real-robot platforms demonstrate that VLA-R1 achieves state-of-the-art performance and superior generalization. We believe this work provides a promising step toward bridging the gap between reasoning quality and action execution in embodied AI.

REFERENCES

- [1] M. Andrychowicz, B. Baker, M. Chociej, et al., “Learning dexterous in-hand manipulation,” International Journal of Robotics Research, vol. 39, no. 1, pp. 3–20, 2020.
- [2] A. Azzolini, H. Brandon, P. Chattopadhyay, et al., “Cosmosreason1: From physical common sense to embodied reasoning,” arXiv preprint arXiv:2503.15558, 2025.
- [3] J. Bai, S. Bai, Y. Chu, et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.
- [4] S. Bai, K. Chen, X. Liu, J. Wang, et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.
- [5] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, et al., “Rt2: Vision-language-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023.

- [6] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Pérez, “Integrated task and motion planning,” Annual review of control, robotics, and autonomous systems, vol. 4, no. 1, pp. 265–293, 2021.
- [7] J. Ge, T. Cheng, B. Wu, Z. Zhang, S. Huang, J. Bishop, G. Shepherd, M. Fang, L. Chen, and Y. Zhao, “Vasevqa: Multimodal agent and benchmark for ancient greek pottery,” arXiv preprint arXiv:2509.17191, 2025.
- [8] Y. Geng, B. An, H. Geng, et al., “Rlafford: End-to-end affordance learning for robotic manipulation,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 5880–5886.
- [9] S. Gu, E. Holly, T. Lillicrap, and S. Levine, “Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates,” in 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017, pp. 3389–3396.
- [10] D. Guo, D. Yang, H. Zhang, et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [11] T. Huang, Z. Zhang, et al., “3d coca: Contrastive learners are 3d captioners,” arXiv preprint arXiv:2504.09518, 2025.
- [12] T. Huang, Z. Zhang, and H. Tang, “3d-r1: Enhancing reasoning in 3d vlms for unified scene understanding,” arXiv preprint arXiv:2507.23478, 2025.
- [13] T. Huang, Z. Zhang, R. Zhang, and Y. Zhao, “Dc-scene: Datacentric learning for 3d scene understanding,” arXiv preprint arXiv:2505.15232, 2025.
- [14] W. Huang, C. Wang, R. Zhang, Y. Li, et al., “Voxposer: Composable 3d value maps for robotic manipulation with language models,” in Proceedings of the Conference on Robot Learning (CoRL), ser. Proceedings of Machine Learning Research, vol. 229, 2023, pp. 540–562.
- [15] W. Huang, B. Jia, Z. Zhai, S. Cao, et al., “Vision-r1: Incentivizing reasoning capability in multimodal large language models,” arXiv preprint arXiv:2503.06749, 2025.
- [16] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al., “π0. 5: a vision-language-action model with open-world generalization, 2025,” arXiv preprint arXiv:2504.16054, 2025.
- [17] Y. Ji, H. Tan, J. Shi, X. Hao, et al., “Robobrain: A unified brain model for robotic manipulation from abstract to concrete,” arXiv preprint arXiv:2502.21257, 2025.
- [18] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.
- [19] B. Li, Y. Zhang, D. Guo, et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.
- [20] X. Li, M. Zhang, Y. Geng, et al., “Manipllm: Embodied multimodal large language model for object-centric robotic manipulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18061–18070.
- [21] X. Li, M. Liu, H. Zhang, Yu, et al., “Vision-language foundation models as effective robot imitators,” arXiv preprint arXiv:2311.01378, 2023.
- [22] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 26296– 26306.
- [23] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2024.
- [24] J. Liu, M. Liu, Z. Wang, Lee, et al., “Robomamba: Multimodal state space model for efficient robot reasoning and manipulation,” arXiv preprint arXiv:2406.04339, 2024.
- [25] Q. Liu, T. Huang, Z. Zhang, and H. Tang, “Nav-r1: Reasoning and navigation in embodied scenes,” arXiv preprint arXiv:2509.10884, 2025.
- [26] S. Liu, L. Wu, B. Li, et al., “Rdt-1b: a diffusion foundation model

- for bimanual manipulation,” arXiv preprint arXiv:2410.07864, 2024.
- [27] A. Myers, C. L. Teo, C. Fermüller, and Y. Aloimonos, “Affordance detection of tool parts from geometric features,” in 2015 IEEE international conference on robotics and automation (ICRA). IEEE, 2015, pp. 1374–1381.
- [28] D. Niu, Y. Sharma, G. Biamby, J. Quenum, et al., “Llarva: Vision-action instruction tuning enhances robot learning,” arXiv preprint arXiv:2406.11815, 2024.
- [29] A. O’Neill, A. Rehman, A. Maddukuri, et al., “Open XEmbodiment: Robotic learning datasets and RT-X models,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6892–6903.
- [30] Y. Peng, G. Zhang, M. Zhang, et al., “Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rulebased rl,” arXiv preprint arXiv:2503.07536, 2025.
- [31] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese, “Generalized intersection over union: A metric and a loss for bounding box regression,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 658–666.
- [32] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.
- [33] H. Shen, P. Liu, J. Li, et al., “Vlm-r1: A stable and generalizable r1-style large vision-language model,” arXiv preprint arXiv:2504.07615, 2025.
- [34] Z. Song, G. Ouyang, M. Fang, et al., “Hazards in daily life? enabling robots to proactively detect and resolve anomalies,” arXiv preprint arXiv:2411.00781, 2024.
- [35] Z. Song, G. Ouyang, M. Li, et al., “Maniplvm-r1: Reinforcement learning for reasoning in embodied manipulation with large vision-language models,” arXiv preprint arXiv:2505.16517, 2025.
- [36] W. Wan, H. Geng, Liu, et al., “Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 3891–3902.
- [37] B. Wang, X. Meng, X. Wang, Z. Zhu, A. Ye, Y. Wang, Z. Yang, C. Ni, G. Huang, and X. Wang, “Embodiedreamer: Advancing real2sim2real transfer for policy training via embodied world modeling,” arXiv preprint arXiv:2507.05198, 2025.
- [38] P. Wang, S. Bai, S. Tan, et al., “Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.
- [39] Q. Wang, H. Zhang, C. Deng, Y. You, et al., “Sparsedff: Sparseview feature distillation for one-shot dexterous manipulation,” arXiv preprint arXiv:2310.16838, 2023.
- [40] Z. Wu, Y. Wang, Y. Wen, Z. Zhang, B. Wu, and H. Tang, “Stereoadapter: Adapting stereo depth estimation to underwater scenes,” arXiv preprint arXiv:2509.16415, 2025.
- [41] A. Ye, Y. Song, J. Su, and D. Zhang, “Non-invasive spatial registration using customized dental bracket and improved genetic algorithms,” in 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE). IEEE, 2024, pp. 3250–3255.
- [42] W. Zhang, M. Wang, G. Liu, et al., “Embodied-reasoner: Synergizing visual search, reasoning, and action for embodied interactive tasks,” arXiv preprint arXiv:2503.21696, 2025.

[Figure 10]

Prompt: Pick up the yellow bowl.

[Figure 11]

Answer: <think>The instruction "pick up the yellow bowl" suggests a pick-and-place operation where the robot must identify and grasp a yellow bowl. This involves precise positioning and handling due to the bowl's distinctness. In the provided image, there is a yellow bowl located on the table surface. The bowl has a smooth surface and a distinct color and shape. The bowl's handle is visible, which is a common affordance for picking up. The predicted affordance region seems to be centered over the yellow bowl's handle. This placement makes sense because the handle is a logical affordance for picking up the bowl. The size of the region indicates it covers enough of the handle to provide enough surface area for the robot's grippers to securely grasp the handle. Considering the spatial arrangement, the robot arm would have sufficient clearance to move into the designated region and execute the pickup. The robot then would need to lift the bowl off the table, ensuring that it stays securely held during the lift-off. The robot’s workspace must be sufficient to accommodate both the pick-up and the movement.</think> so, the answer is <output>{"x": 0.563749, "y": 0.638217, "width": 0.094696, "height": 0.123233}</output>

###### {“x”:354“y”:330“w”:41“h”:39}

###### Side View

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Prompt: Move the yellow bowl to the white basket.

Head Camera View

[Figure 19]

Answer: <think>The taskSiderequiresView moving the yellow bowl to the white basket, implying that the robot needs to grasp the yellow bowl and execute movement. The yellow bowl is in the center of the frame, alongside which the white basket is visible. The robot arm starts from the bottom right of the frame, indicating its current position relative to the bowl. The gripper is currently gripping the bowl and must move it to the basket. The scene contains several objects such as a bowl, a basket, and other items. However, none appear to obstruct the path directly between the bowl and the basket. There may still be minor considerations for avoiding contact with surrounding objects during movement. The trajectory consists of multiple waypoints guiding the bowl smoothly from its current position upward and slightly to the left. This approach ensures controlled and incremental movement while avoiding abrupt changes in direction. The waypoints seem evenly spaced, further emphasizing precision throughout the approach.</think> so, the answer is <output>[[0.715645, 0.684211], [0.593633, 0.504111], [0.463881, 0.436894], [0.356734, 0.386189]]</output>

###### [[458, 328], [380, 242], [297, 210], [228, 185]]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Prompt: Pick up the yellow bowl

[Figure 27]

Answer:<think>The instruction "pick up the yellow bowl" suggests a pick-and-place operation where the robot must identify and grasp a yellow bowl. This involves precise positioning and handling due to the bowl's distinctness. In the provided image, there is a yellow bowl located on the table surface. The bowl has a smooth surface and a distinct color, which makes it identifiable among other bowls. The bowl’s handle or grip point is a logical affordance for picking up. The predicted affordance region seems to be centered over the yellow bowl. This placement makes sense because the grippers would need to interact directly with the bowl by gripping its handle or grip point. The size of the region indicates it covers enough of the bowl's handle or grip point to provide enough surface area for a firm hold while avoiding interference with surrounding space. Considering the spatial arrangement, the robot arm could feasibly move into the designated region and execute the pick-up. The robot then would need to lift the bowl off the table, ensuring that it stays securely held during the lift-off. The robot’s workspace must be sufficient to accommodate both the pick-up and the movement.</think> so, the answer is <output>{"x": 0.693497, "y": 0.515815, "width": 0.047462, "height": 0.053402}</output>

###### {"x": 444, "y": 248, "w": 30, "h": 26}

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Prompt: Move the yellow bowl to the white basket.

[Figure 35]

Answer: <think>The task requires moving the yellow bowl to the white basket, implying that the robot needs to grasp the yellow bowl and execute movement. The yellow bowl is located on the table surface, alongside other bowls and baskets. The robot arm starts from the top left corner of the frame, indicating its current position relative to the bowl. The gripper is currently positioned over the bowl but must move it to the basket. The scene contains several objects such as a bowl, a basket, and a garlic press bottle. However, none appear to obstruct the path directly between the arm and the bowl. There may still be minor considerations for avoiding contact with surrounding items during movement. The trajectory consists of multiple waypoints guiding the arm smoothly from its current position down and slightly to the right. This approach ensures controlled and incremental movement while avoiding abrupt changes in direction. The waypoints seem evenly spaced, further emphasizing precision throughout the approach.</think> so, the answer is <output>[[0.883966, 0.556454], [0.733085, 0.436189], [0.572835, 0.325734], [0.450518, 0.255637]]</output>

###### [[566, 267], [469, 209], [367, 156], [288, 123]]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

{ from: “human”: value: “pick ketchup from table and place on oven.” }, { from: “gpt” value: “<think>The instruction \"pick ketchup from table and place on oven\" involves identifying and grasping a bottle of ketchup before moving it to a different location. This necessitates recognizing the object's shape, size, and orientation to ensure successful picking.\n\nIn the scene, various items are scattered across a white tabletop. Among them is a red bottle resembling typical ketchup packaging. The bottle has distinct features like its cylindrical body, cap, and label which provide strong visual cues for recognition. Notably, one end of the bottle seems slightly narrower where a cap would be located, indicating a potential grasping point.\n\nThe predicted region corresponds to the section of the image where the ketchup bottle is placed. Given the dimensions provided, the region encapsulates part of the bottle’s body, likely around its midsection – a logical spot for a robot arm to grip securely while avoiding slippage due to smooth surfaces or weight distribution.\n\nConsidering the setup, the robot must be able to reach into this space using its grippers effectively. With sufficient clearance above and around the bottle, the robot could maneuver into position easily. The absence of obstructions suggests the operation would be feasible within the constraints presented by the environment.</think> so, the answer is <output>{\"x\": 0.555652, \"y\": 0.605505, \"width\": 0.072702, \"height\": 0.079511}</output>” }

[Figure 42]

Original image

[Figure 43]

Visualization

{ from: “human”: value: “reach for the pan.” }, { from: “gpt” value: “<think>The task involves reaching for the pan located on the wooden countertop. This requires moving the robot arm towards the pan while avoiding other objects like the sponge and fork.\n\nThe pan is situated at the center of the wooden countertop, easily identifiable by its metallic color and handles. The robot arm starts off-screen but is likely approaching the countertop from one direction. There may not be immediate visual confirmation of the arm's starting position based solely on the image provided.\n\nThe scene contains several items besides the pan: a yellow sponge to the left, a red fork-like tool to the right, and some hanging black objects in the background. These objects do not seem to obstruct the path directly toward the pan; however, care must be taken to avoid knocking them over during the movement.\n\nGiven the absence of significant obstacles around the pan, a straightforward approach would involve selecting a linear path aimed directly at the pan. The trajectory waypoints suggest a smooth curve if necessary to navigate past nearby items without colliding. Four points spaced evenly across this route ensure gradual and controlled motion until the gripper reaches the desired location near the pan handle.</think> so, the answer is <output>[[0.45, 0.002353], [0.48, 0.129412], [0.547059, 0.378824], [0.555882, 0.611765]]</output>” }

[Figure 44]

Original image

[Figure 45]

Visualization

{ from: “human” value: “open the left door.” },

[Figure 46]

{ from: “gpt” value: “<think>The instruction \"open the left door\" suggests a rotational movement around a hinge axis, requiring engagement with a handle mechanism on the left-hand side of the door structure.\n\nIn the provided scene, we observe a pair of doors with distinct handles—one on each door. The left door features a metallic handle that protrudes outward, signaling its function as a grip for opening. The polished surface reflects light differently compared to the wood, aiding in identifying the handle's location. Additionally, there is evidence of a slight shadow beneath the handle, indicating its three-dimensional nature and suitability for grasping.\n\nThe predicted actionable region centers on the handle of the left door. This placement is logical because manipulating the handle directly influences the door's state. The dimensions of the region match the size of the handle, suggesting precise targeting necessary for effective manipulation.\n\nGiven the position and orientation of the predicted region, a robot could feasibly apply force along the handle’s length to initiate rotation. Assuming standard door mechanics, the surrounding space allows adequate room for the robot's grippers to interact without interference from adjacent structures like the right door or the frame.</think> so, the answer is <output>{\"x\": 0.466102, \"y\": 0.622034, \"width\": 0.101695, \"height\": 0.061017}</output>” }

Original image

[Figure 47]

Visualization

{ from: “human” value: “pick the cube.” },

[Figure 48]

{ from: “gpt” value: “<think>The instruction \"pick the cube\" indicates a single-object manipulation task requiring precise control to grasp and lift a small, potentially cubic item from its current position on the surface.\n\nObserving the image reveals multiple objects placed atop a dark-colored shelf-like structure. Among them, there's a white cube that stands out due to its distinct geometry and color contrast against the darker background. This object has sharp edges and flat faces typical of a cube, which aids in identifying it as the target for picking.\n\nThe affordance region seems centered over one face of the white cube, suggesting it is within reach and provides a stable surface for the robot’s grippers to interact with. Given the size parameters provided, the region adequately covers part of the cube while avoiding overlap with neighboring items like the discshaped object nearby.\n\nConsidering the location and orientation of the predicted region relative to the cube, the robot could feasibly execute the pick operation. With appropriate pre-grasp adjustments, the robot arm would be able to securely enclose the cube using its grippers and then lift it off the shelf surface without disturbing adjacent objects.</think> so, the answer is <output>{\"x\": 0.564231, \"y\": 0.827692, \"width\": 0.039808, \"height\": 0.083077}</output>” }

Original image

[Figure 49]

Visualization

