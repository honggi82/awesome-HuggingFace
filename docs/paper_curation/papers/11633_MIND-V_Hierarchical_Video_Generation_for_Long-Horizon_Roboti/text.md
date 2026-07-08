## MMIND-VIND-V: Hierarchical World Model for Long-Horizon Robotic Manipulation with RL-based Physical Alignment

Ruicheng Zhang1 Mingyang Zhang1 Jun Zhou1 Xiaofan Liu3 Zunnan Xu1,2* Zhizhou Zhong4 Puxin Yan4 Haocheng Luo1 Xiu Li1†

1Tsinghua University 2X Square Robot 3Sun Yat-sen University 4HKUST

# arXiv:2512.06628v3[cs.RO]9Jun2026

#### Abstract

Scalable embodied intelligence is constrained by the scarcity of diverse, long-horizon robotic manipulation data. Existing video world models are limited to short clips of simple actions and frequently rely on manually defined trajectories. We introduce MMIND-VIND-V, a cognitive hierarchical world model for synthesizing physically plausible, logically coherent long-horizon manipulation videos. Inspired by cognitive science, MMIND-VIND-V bridges high-level reasoning with pixel-level synthesis via three components: a Semantic Reasoning Hub (SRH) that leverages a pretrained vision-language model for task planning; a Behavioral Semantic Bridge (BSB) that converts abstract instructions into domain-invariant representations; and a Motor Video Generator (MVG) for conditional video rendering. To enforce physical plausibility, we introduce a GRPO-based RL posttraining stage guided by a Physical Foresight Coherence (PFC) reward, which employs VJEPA2 as a physics referee to penalize implausible dynamics in latent space. Experiments demonstrate state-of-the-art performance in long-horizon simulation and significant utility for downstream policy learning, establishing a scalable, fully autonomous framework for embodied data synthesis.

#### 1 Introduction

Scalable robot learning within Embodied AI (Black et al., 2024) is critically bottlenecked by the scarcity of high-quality, diverse, and long-horizon robotic manipulation data (Fu et al., 2025). Video world models (Zhu et al., 2025a; Chi et al., 2025; Agarwal et al., 2025) (VWM) that can simulate physical interactions and predict future outcomes offer a promising solution by potentially synthesizing an infinite stream of robotic operation videos.

However, generating high-quality, long-horizon robotic manipulation videos that faithfully follow

*Project Lead. †Corresponding author.

[Figure 1]

Figure 1: Architectural comparison of MMIND-VIND-V (c) against existing paradigms for long-horizon robotic manipulation modeling.

commands poses three core challenges: (1) LongHorizon Coherence: Maintaining causal consistency and logical flow across interconnected subtasks, where a single error can compromise the entire operation (Ji et al., 2025). (2) Semantic-toPixel Generation: Accurately translating abstract semantic commands into spatiotemporal pixellevel interactions, placing stringent demands on semantic understanding and instruction-following fidelity (Ma et al., 2025). (3) Physical Plausibility: Ensuring adherence to fundamental physical laws governing collision dynamics, object permanence, and interaction forces (Meng et al., 2025). Existing methods address these challenges only partially. Directly fine-tuning video foundation models (Kong et al., 2024) for long-horizon robotic modeling suffers from imprecise instruction following and visual degradation, as these models struggle to bridge the gap from abstract commands to pixel-level execution (Fig. 1(a)). Trajectory-controlled generative models (Wu et al., 2024), while offering enhanced controllability, sacrifice the autonomy and scalability required for large-scale data generation, functioning more as renderers than as intelligent world simulators (Fig. 1(b)).

Inspired by the hierarchical theory of human motor control (Grillner, 1985), we draw an analogy to how the brain executes complex tasks: highlevel cognitive centers such as the cerebral cortex handle intent understanding and abstract planning, while low-level motor systems such as the cerebellum translate these plans into precise motor com-

mands, with specialized neural pathways bridging the two (Merel et al., 2019).

Embracing this paradigm, we introduce MMIND-VIND-V, a cognitive hierarchical world model for simulating physically plausible, logically coherent long-horizon robotic manipulation. MMIND-VIND-V is built upon three core components (Fig. 1(c)): (1) a Semantic Reasoning Hub (SRH) that performs high-level task planning via a pretrained Vision-Language Model (VLM); (2) a Behavioral Semantic Bridge (BSB) that translates abstract plans into structured, domain-invariant representations; and (3) a Motor Video Generator (MVG) that synthesizes physically realistic manipulation videos conditioned on the BSB. By decomposing cognition and execution into explicit hierarchical stages, MMIND-VIND-V effectively bridges high-level reasoning with pixel-level synthesis.

To align the MVG with physical laws, we introduce a GRPO-based (Liu et al., 2025; Zhang et al., 2026b) post-training stage (Zhang et al., 2025c) guided by a Physical Foresight Coherence (PFC) reward, which employs a pretrained world model as a physics referee to quantify the dynamic coherence of generated videos in latent space. We further propose Staged Visual Future Rollouts, a test-time optimization strategy that decomposes long-horizon planning into locally optimal decisions. At each subtask transition, MMIND-VIND-V simulates multiple candidate futures and selects the most coherent continuation via a propose-verify-refine loop, suppressing error propagation and semantic drift.

Our main contributions are as follows:

- • MMIND-VIND-V is, to the best of our knowledge, the first hierarchical video world model for longhorizon robotic manipulation, bridging highlevel task planning and pixel-level synthesis through a three-tier architecture comprising the SRH, BSB, and MVG.
- • We present Staged Visual Future Rollouts, a test-time optimization strategy that decomposes long-horizon generation into locally optimal decisions via a propose-verify-refine loop at each subtask transition, mitigating error accumulation and improving generation robustness.
- • We propose a GRPO post-training stage guided by a Physical Foresight Coherence (PFC) reward, which employs a pretrained world model as a physics referee to score generated dynamics in latent space, steering the MVG toward physically plausible outputs.

• Extensive experiments demonstrate state-ofthe-art performance in long-horizon manipulation video synthesis and significant utility for downstream policy learning, establishing a scalable and autonomous paradigm for robotic world modeling.

#### 2 Related Work

- 2.1 Video World Models for Embodied AI Scalable embodied intelligence critically depends on large-scale data, yet collecting realworld robotic demonstrations is costly and laborintensive. Video World Models (VWMs) (Zhang et al., 2026a), which predict future states from current observations, have emerged as an efficient alternative for synthesizing photorealistic training data. UniPi (Du et al., 2023) and AVDC (Ko

- et al., 2023) frame robotic planning as a text-tovideo generation problem, while WoW (Chi et al., 2025), DreamDojo (Gao et al., 2026), and RoboDreamer (Zhou et al., 2024) learn latent physical dynamics from large-scale video interaction data to achieve compositional generalization. Despite strong semantic understanding, these methods lack fine-grained control over manipulation execution (Ma et al., 2025), which can cause logical failures and physical inconsistencies in long-horizon settings. A complementary line of work, including IRASim (Zhu et al., 2025a), Cosmos (Agarwal et al., 2025), and RoboMaster (Fu et al., 2025), employs explicit trajectory guidance for more precise action modeling, but requires dense manual annotations such as motion paths and object masks, limiting scalability and autonomy. MMIND-VIND-V addresses both limitations through a hierarchical architecture that autonomously decomposes highlevel commands into executable generator instructions, enabling long-horizon, high-fidelity manipulation video synthesis without additional manual supervision.

2.2 Controllable Video Generation

Recent advances in diffusion-based generation have intensified interest in controllable methods that faithfully translate user intent into visual content. Existing approaches span a spectrum of conditioning modalities, ranging from high-level semantic signals such as text prompts (Wan et al., 2025) to low-level structural inputs including masks (Wu

- et al., 2024), trajectories (Zhang et al., 2025d), sketches (Ma et al., 2024), and pose estimates (Xu
- et al., 2025). This spectrum reflects a fundamental

###### (c) Motor Video

###### (a) Semantic Reasoning Hub (SRH)

###### (d) Staged Visual Future Rollouts

###### Generator (MVG)

[Figure 2]

[Figure 3]

K sampled videos Comprehensive

[Figure 4]

[Figure 5]

“I should put the spoon and

“Clear the desktop.”

[Figure 6]

Evaluation

[Figure 7]

[Figure 8]

towel in the metal pot. ”

Encoder

Decoder

[Figure 9]

|Vision<br><br>Language Model (VLM)<br><br>Gemini<br><br>2.5-Pro<br><br>[Figure 10]|
|---|

DiT

[Figure 11]

VLM

[Figure 12]

[Figure 13]

|Sub-tasks<br><br>Sub Task 1: {pick, spoon, metal pot}<br>Sub Task 2: {pick, towel, metal pot}<br>Sub Task 3: ······<br>|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

###### BSB

Sub Task Ni i~(1,K)

SceneImage

Task Completion Judge

[Figure 19]

[Figure 20]

· · · · · ·

C1 = 1, (0 or 1) Successfully grabbed the

Scene Image

[Figure 21]

Video

spoon andput it into thepot.

Physical Plausibility Judge

[Figure 22]

[Figure 23]

[Figure 24]

###### C2 = 0.5, (0—1)

(b) Behavioral

The spoon is slightly deformed and blurry.

[Figure 25]

[Figure 26]

Semantic Bridge (BSB)

###### Affordance Grounding

Trajectory Planning

[Figure 27]

[Figure 28]

[Figure 29]

Visual Quality Judge

Object Representations

Affordance R1 VLM

[Figure 30]

- 1
- 2
- 3

[Figure 31]

C3 = 0.7, (0—1)

Successes.

[Figure 32]

[Figure 33]

[Figure 34]

Not at the center ofthe pot.

The image quality is basically clear without artifacts.

[Figure 35]

[Figure 36]

Decomposed

[Figure 37]

[Figure 38]

Collaborative Trajectories

[Figure 39]

[Figure 40]

C = C1 + C2 + C3 Execute the next

[Figure 41]

Phase Transition Points

[Figure 42]

subtask or replan

[Figure 43]

[Figure 44]

Gripping point

Object Mask

[Figure 45]

Plan – Check – Adjust

[Figure 46]

[Figure 47]

- Figure 2: Overview of MMIND-VIND-V: a cognitive hierarchical world model for long-horizon robotic manipulation.

trade-off: high-level conditions offer intuitive guidance but are prone to semantic drift and fidelity degradation in long-horizon, multi-stage tasks, whereas low-level conditions afford precise spatiotemporal control but impose a heavy annotation burden. MMIND-VIND-V resolves this dilemma through a hierarchical VWM framework that bridges highlevel reasoning and pixel-level synthesis. The Semantic Reasoning Hub (SRH) interprets abstract user intent, while the Behavioral Semantic Bridge (BSB) automatically converts it into structured, multi-dimensional representations for the generator, achieving high semantic fidelity and precise control in long-horizon world modeling without auxiliary manual annotations.

#### 3 Method

- 3.1 Framework As illustrated in Fig. 2, MMIND-VIND-V implements a top-down pipeline from high-level cognition to concrete visual representation. First, the Semantic Reasoning Hub (SRH) decomposes a long-horizon task into atomic subtasks based on initial observations and user instructions. For each subtask, the SRH then employs vision modules for affordance localization and trajectory planning to construct a structured, domain-invariant intermediate representation, termed the Behavioral Semantic Bridge (BSB). This representation guides the Motor Video Generator (MVG) in synthesizing a photorealistic video sequence. A closed-loop feedback mechanism via Staged Visual Future Rollouts returns the generated results to the SRH for evaluation and potential re-planning. A detailed description of each component follows. Semantic Reasoning Hub (SRH)

As the cognitive core of MMIND-VIND-V, the SRH

translates abstract semantics into actionable geometric signals by synergizing two components: a pretrained Vision-Language Model (VLM, e.g., Gemini-2.5-Pro (Comanici et al., 2025)) for longhorizon planning and semantic reasoning, and an affordance-based visual localizer (e.g., AffordanceR1 (Wang et al., 2025)) for grounding plans with physical common sense.

Given an initial observation I0 and a longhorizon instruction L (e.g., “clean the desktop”), the VLM analyzes the scene and decomposes L into an ordered sequence of atomic subtasks, each defined by a tuple SubTaski = {ActionTypei,Objecti,Destinationi} specifying the action primitive, manipulation target, and goal location. This structured decomposition provides a symbolic foundation for precise downstream control. For each subtask, the affordance localizer identifies the object segmentation mask Mobj and predicts functional interaction points Pobj (e.g., the handle of a cup). Based on this affordance information, the VLM plans a physically plausible trajectory, parameterized as a smooth curve and discretized into frame-aligned waypoints.

###### Behavioral Semantic Bridge (BSB)

The BSB is a structured, domain-invariant intermediate representation that translates symbolic outputs from the SRH into an actionable conditioning format for the MVG. It comprises three elements:

- • Object Representation: Segmentation masks for the manipulated object (Mobj) and robot arm (Mrob), encoded via a VAE and injected at designated spatiotemporal locations during generation to maintain consistent object identity.
- • Decomposed Collaborative Trajectory: The trajectory is partitioned into three phases:

MotionInjection

MotionInjection

###### ······

###### (b) Motion Embedding

###### (a) Latent Diffusion Transformer

Instruction Prompt

T5

Spatial Convolution

Conv2d Rearrange Conv1d Norm

DiTBlock

[Figure 48]

DiTBlock

DiTBlock

###### ···

|ℎ|
|---|

###### ℎ + ℎ ···

###### ···

Decoder

+ +

Encoder

Temporal Convolution

Sence

|𝐺|
|---|

|𝐺|
|---|

|𝐺|
|---|

Image

tContac

Layer

[Figure 49]

Normalization

Motion Embedding

Video

×

ℎ: DiT Output

[Figure 50]

Motion Guidance

𝐺

Initial Latents

###### (c) Spatiotemporal Guidance Tensor

Latent Space

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Encoder

[Figure 55]

Object Mask

[Figure 56]

: Object Representation

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

: Decomposed Collaborative Trajectory

[Figure 63]

[Figure 64]

[Figure 65]

Spatiotemporal Injection

[Figure 66]

[Figure 67]

| |
|---|

[Figure 68]

[Figure 69]

: Phase Transition Points

Trajectory

Pre-interaction Iteraction Post-interaction BSB Guidance

- Figure 3: Architecture of the Motor Video Generator (MVG). Serving as the simulation engine, the MVG translates abstract intents (BSB) into concrete visual futures. The process initiates with encoding the BSB’s semantic representation into the (c) Spatiotemporal Guidance Tensor, which embeds the visual features of the active agent along its planned trajectory. This guidance is then processed by the (b) Motion Embedding module to produce refined motion signals (G). These signal are finally injected into the (a) Latent Diffusion Transformer, conditioning the denoising process to ensure the synthesized video strictly adheres to the intended physical dynamics.

pre-interaction (Tpre, arm approach), interaction (Tinteract, object manipulation), and postinteraction (Tpost, arm retraction), clearly defining the active agent and objective at each stage.

intermediate hidden state h via additive fusion:

hnew = h + norm(G) · G, (1)

where norm(·) denotes Group Normalization to stabilize training. This continuous injection of kinematic constraints compels the model to adhere to the specified trajectory throughout the denoising process, yielding a final video that is both spatiotemporally precise and visually coherent.

• Phase Transition Points: A triplet of frame indices (Fpre,Finteract,Fpost) allocating a specific duration to each phase, ensuring natural motion dynamics and appropriate emphasis on the core physical interaction.

By decoupling task logic from visual appearance, the BSB achieves domain invariance, enhancing generalization to novel environments and tasks.

###### 3.2 Staged Visual Future Rollouts

Long-horizon video generation is susceptible to error accumulation, where minor deviations in early subtasks cascade into overall task failure (Zhang et al., 2025b). To mitigate this, we reframe inference as a search process over the world model’s latent space, introducing Staged Visual Future Rollouts: a test-time optimization strategy that emulates System-2 thinking (Zhang et al., 2025a) by deliberating over candidate futures at critical decision points. As illustrated in Fig. 2(d), the strategy operates as a propose-verify-refine loop at each subtask transition. Rather than committing to a single plan, the SRH proposes K semantically diverse candidate plans (i.e., BSB variants). The MVG rolls out each plan in parallel into short-horizon video clips {V1,...,VK}. The VLM then acts as a critic, evaluating each rollout on task success, physical plausibility, and visual quality.

###### Motor Video Generator (MVG)

As depicted in Fig. 3, the MVG functions as a learnable physics engine. Built upon a Diffusion Transformer (DiT) backbone (Peebles and Xie,

- 2023), it is trained to synthesize high-fidelity video sequences that strictly adhere to the kinematic constraints defined by the BSB. To enforce this control, the MVG first encodes the BSB’s object representation into a spatiotemporal guidance tensor of size (T ×C×H×W). This tensor dynamically embeds the visual features of the active agent (arm or manipulated object) onto its planned path across the time dimension. A motion embedding module integrates this guidance into the DiT backbone during denoising process. The module employs spatiotemporal convolutions to encode the guidance tensor into a feature representation G. Within each DiT block, this representation is fused with the video’s

If the top-scoring rollout Vtop meets a predefined success threshold, it is selected and the process advances to the next subtask. Otherwise, the

[Figure 70]

- Figure 4: (a) Illustration of the SRH Planning. (b) Physical Foresight Coherence (PFC) Reward. The PFC leverages a frozen V-JEPA2 world model as a “Physics Referee” to predict the latent evolution of future states. The reward quantifies the alignment between the generated video’s dynamics and V-JEPA2’s internal physical laws, computed via cosine similarity in the predictive latent space.

VLM provides structured textual feedback identifying the failure modes (e.g., “end position error”), which instructs the SRH to re-plan and propose a refined set of masks and trajectories. This iterative propose-verify-refine cycle transforms MMIND-VIND-V from a feed-forward predictor into a self-correcting agent, improving robustness and success rate in long-horizon task simulation.

- 3.3 MVG Training: From SFT to RL To establish the MVG as a physically grounded simulator, we adopt a two-stage training paradigm: Supervised Fine-Tuning (SFT) to adapt a pretrained video model to the robotics domain, followed by a GRPO post-training that aligns the model’s dynamics with physical laws and aesthetic standards.

- Stage 1: Supervised Fine-Tuning (SFT) SFT trains the MVG to learn the mapping from

BSB representations to coherent pixel-level execution, minimizing the standard BSB-conditioned denoising objective:

LSFT(θ) = Eϵ∼N(0,I), t ∥ϵ − ϵθ(xt,t,BSB)∥2 ,

(2) where xt is the noised video at timestep t. This stage yields a reference policy πref for subsequent alignment. Notably, training on short subtask videos suffices, as the hierarchical framework enables generalization to arbitrary long-horizon tasks.

- Stage 2: Physical Alignment via GRPO PostTraining

SFT alone cannot guarantee physical plausibility or aesthetic quality, objectives ill-suited to conventional loss functions (Liu et al., 2025). We therefore introduce an RL alignment stage that models the denoising process as a Markov Decision Process and employs Group Relative Policy Optimization

(GRPO) (Liu et al., 2025). Optimization is guided by a composite reward:

###### R(x0) = wp·Rphysics(x0)+wa·Raesthetic(x0), (3)

with wp = 0.2 and wa = 1; ablation details are provided in Sec. 4.4.

Physical Foresight Coherence (PFC) Reward (Rphysics) We leverage V-JEPA2 (Assran et al., 2025), pretrained via self-supervision on largescale video data and fine-tuned on robotics datasets, as a physics referee. Its learned world dynamics enable accurate latent-space prediction of future states. For each generated video, a sliding window computes the local physical plausibility score si as the cosine similarity between V-JEPA2’s latent prediction and the actual future (Fig. 4):

###### si = simcos Pv(Ev(x(contexti) )), Ev(x(targeti) ) , (4)

where Ev and Pv denote V-JEPA2’s visual encoder and predictor, respectively. To concentrate optimization on the most severe physical violations, a softmax-weighted aggregation assigns higher weights to windows with larger errors (1 − si):

Nw

exp((1 − si)/τ)

Nw j=1 exp((1 − sj)/τ) · si, (5)

Rphysics(x0) =

i=1

where temperature τ controls focus sharpness: lower values concentrate the reward on the worstoffending window. The PFC reward thus reframes physical evaluation as targeted optimization of dynamic causal consistency, improving the physical plausibility of generated actions (Yuan et al., 2025).

Aesthetic Reward (Raesthetic) The aesthetic reward is provided by a VLM. The VLM assesses each video for clarity, artifacts, and realism, assigning a discrete integer score (e.g., 1-5).

GRPO Optimization GRPO is an efficient, valuefree policy optimization algorithm. At each optimization step, we sample a group of G videos {xi0}Gi=1 from the current policy πθ. The advantage Aˆi for each sample is computed by normalizing its reward relative to the group statistics: Aˆi =

R(xi0) − mean({R(xj0)}Gj=1) /std({R(xj0)}Gj=1), where mean(·) and std(·) denote the group mean and standard deviation, respectively. This grouprelative normalization provides a stable advantage estimation without requiring a separate critic network. The policy is then updated by maximizing the standard GRPO objective function:

JGRPO(θ) = E

G

1 G

i=1

min ri(θ)Aˆi,

clip(ri(θ), 1 − ϵ, 1 + ϵ)Aˆi − βDKL(πθ∥πref) .

(6)

i 0)

where ri(θ) = πθ(x

πref(xi0) is the importance sampling ratio, ϵ is a clipping hyperparameter that ensures conservative policy updates, and the KLdivergence term regularizes the policy towards the SFT policy πref to mitigate reward hacking and maintain generation quality. This optimization process progressively aligns the MVG towards higher physical fidelity and aesthetic quality while maintaining its adherence to kinematic conditioning.

#### 4 Experiments

###### 4.1 Experiment Settings

Architecture and Training The SRH employs Gemini-2.5 Pro (Comanici et al., 2025) as the core Vision-Language Model and Affordance-R1 (Wang et al., 2025) as the visual localizer. The MVG is initialized from the pretrained CogVideoX-5B (Yang

- et al., 2024) architecture. Experiments are conducted on the Bridge V2 (Walke et al., 2023) dataset following the preprocessing protocol of (Fu
- et al., 2025), with a resolution of 480 × 640 pixels and 37 frames per subtask. Training proceeds in two stages: supervised fine-tuning (SFT) for 30,000 steps using AdamW with a learning rate of 2 × 10−5, followed by GRPO post-training for 1,500 iterations at 5×10−5. At inference time, generating a 111-frame long-horizon video spanning three subtasks requires approximately 50 GB of VRAM. Owing to its autoregressive subtask-based design, MMIND-VIND-V supports arbitrarily long task sequences with only linear growth in computational cost. All experiments are run on four NVIDIA H200 GPUs, with further implementation details provided in the Appendix.

Evaluation Protocol and Metrics Evaluation is performed on a test set of 108 samples, comprising scenes from the Bridge V2 test set (Walke et al., 2023) and unseen web-sourced scenes. Given that different task horizons demand different criteria, we adopt a bifurcated evaluation protocol. For short-horizon tasks, we assess visual quality using V-Bench (Zheng et al., 2025). For long-horizon tasks, we additionally conduct a user study and report two metrics: (1) physical plausibility, quantified by our Physical Foresight Coherence (PFC)

score (Section 3.3); and (2) task success rate, defined as the average success rate across all subtasks of the full long-horizon task.

Baselines and Comparative Setup For shorthorizon tasks, we benchmark MMIND-VIND-V against two categories of baselines: autonomous world models (DreamDojo (Gao et al., 2026), RoboDreamer (Zhou et al., 2024), WoW (Chi et al., 2025), CogVideoX-5B (Yang et al., 2024), HunyuanVideo (Kong et al., 2024)) and trajectoryguided methods (IRASim (Zhu et al., 2025a), Cosmos (Agarwal et al., 2025), MotionCtrl (Wang et al., 2024), DragAnything (Wu et al., 2024), Tora (Zhang et al., 2025e)). Video foundation models (CogVideoX-5B and HunyuanVideo) are finetuned on our dataset for fair comparison. Notably, trajectory-guided baselines receive privileged inputs at inference time, such as manual trajectories, masks, or anchor points, which are unavailable to our trajectory-free approach, underscoring MMIND-VIND-V’s ability to infer complex dynamics from high-level intent alone. For long-horizon tasks, which require complex planning and reasoning without explicit guidance, comparisons are restricted to the autonomous world model baselines. Each long-horizon task comprises 2 to 4 subtasks, designed to stress-test long-horizon planning and generation.

4.2 Qualitative and Quantitative Comparison As shown in Fig. 5 and Tables 1–2, MMIND-VIND-V consistently outperforms all baselines across both short- and long-horizon tasks. On long-horizon benchmarks, MMIND-VIND-V surpasses the second-best method by 9.0%, 76.7%, and 172.2% on PFC Score, Task Success Rate, and User Preference, respectively. Baseline models exhibit critical failure modes (Fig. 5(a)), including logical hallucinations, physical implausibility (e.g., spontaneous object disappearance), and inaccurate semantic grounding (e.g., incorrect object manipulation), resulting in poor long-horizon coherence (Table 2).

MMIND-VIND-V overcomes these limitations through its cognitive hierarchical design. The SRH and BSB collaborate to decompose user instructions into an explicit, executable plan, mitigating the semantic drift prevalent in end-to-end approaches. The MVG, aligned via the PFC reward, synthesizes videos that adhere to both physical laws and command requirements. Staged Visual Future Rollouts further suppress error accumulation, enabling coherent and physically plausible long-horizon ma-

[Figure 71]

Task Prompt: With the black robotic arm in the picture, first grasp the yellow-handled spoon and put it into the metal pot. Afterwards, take the blue cloth and place it over the spoon.

[Figure 72]

Results

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Task failed: The robotic arm pours green liquid into the metal pot.

Wan2.2

- -14B

WoW

- -14B

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Partial completion: The grasped spoon vanish unexpectedly during the process of moving.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Task successful: The spoon is correctly placed in the metal pot and covered with the cloth.

###### MIND-V

[Figure 91]

Task Prompt: With the black robotic arm in the picture, first grab the carrot on the table and put it into the yellowish-green plate, and then put the strawberry into the pot.

[Figure 92]

Results

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Task failed: No action is performed, and a plate appear unexpectedly.

Hunyuan Video

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Task failed: The wrong object is grasped, and an unexpected object appear.

WoW -14B

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Task successful: The carrot and the strawberry are correctly grasped and Frames placed.

MIND-V

###### Figure 5: Qualitative comparison of long-horizon robotic manipulation video generation.

[Figure 111]

instruction, MMIND-VIND-V generates a successful visual rollout o, against which the VLA’s predicted rollout oˆ is supervised:

Task

[Figure 112]

[Figure 113]

[Figure 114]

VLA MIND-V

[Figure 115]

[Figure 116]

+

Lpose

###### =

Actions

[Figure 117]

Policy Optimization

L = λpix∥oˆ− o∥1 + λpercdLPIPS(ˆo,o) + λILLpose,

Trajectory

[Figure 118]

[Figure 119]

(7) where ∥oˆ − o∥1 and dLPIPS(ˆo,o) denote the L1 pixel reconstruction loss and LPIPS perceptual loss (Zhang et al., 2018), respectively.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Loss

[Figure 125]

[Figure 126]

Prediction Execution Frames Frames 𝑜

···

|𝑜ො|
|---|

[Figure 127]

[Figure 128]

Trainable Frozen

Figure 6: Policy learning with MMIND-VIND-V.

Discussion As shown in Table 3, MMIND-VIND-V guidance raises the mean success rate from 27.7% (Base) to 43.5%, substantially exceeding the 33.4% achieved by IL continuous training. This result highlights a key advantage of pixel-space world models: operating in the same visual space as the VLA’s pre-training data, MMIND-VIND-V’s imagined rollouts directly engage the model’s visual priors, effectively bridging generative world simulation and actionable policy learning (Zhu et al., 2025b).

nipulation sequences.

###### 4.3 World Model for Policy Learning

Trajectory

[Figure 129]

Pixel-level Loss

[Figure 130]

|[Figure 131]<br><br>𝐿1|
|---|

To validate MMIND-VIND-V as a training ground for embodied agents, we conduct a generation-to-policy experiment on the MimicGen simulation benchmark (Mandlekar et al., 2023) (Fig. 6), evaluating three fine-grained manipulation tasks (Coffee, StackThree, Square) with 128 demonstration trajectories per task.

=

[Figure 132]

[Figure 133]

+

[Figure 134]

Perceptual Loss

[Figure 135]

|[Figure 136]<br><br>𝐿LPI𝑃𝑆|
|---|

=

[Figure 137]

###### 4.4 Ablation Study

Setup We first fine-tune OpenVLA-OFT (Kim et al., 2025) via imitation learning (IL) on 300 expert trajectories per task to obtain a base policy. The IL baseline continues training this model under end-effector pose supervision (Lpose) until convergence. Our approach augments the pose loss with a visual goal derived from MMIND-VIND-V: given a task

Component Contribution Analysis To evaluate the contribution of each core component, we compare the full MMIND-VIND-V model against four ablated variants on long-horizon simulation: (a) w/o GRPO: trained solely with SFT, without GRPO post-training stage; (b) Re Affordance w/

###### Table 1: Visual quality evaluation on short-horizon and long-horizon tasks.

Aesthetic Quality ↑

Imaging Quality↑

Temporal Flicker ↑

Motion Smoothness↑

Subject Consistency↑

Bg.

Method

Consistency↑ Short-horizon Tasks

MotionCtrl (Wang et al., 2024) 0.491 0.665 0.977 0.972 0.915 0.942 IRASim (Zhu et al., 2025a) 0.504 0.676 0.979 0.986 0.929 0.957

Cosmos (Agarwal et al., 2025) 0.519 0.680 0.980 0.988 0.932 0.958 DragAnything (Wu et al., 2024) 0.500 0.679 0.980 0.983 0.935 0.957 Tora (Zhang et al., 2025e) 0.509 0.670 0.981 0.984 0.922 0.961 RoboMaster (Fu et al., 2025) 0.502 0.688 0.982 0.981 0.937 0.950 Robodreamer (Zhou et al., 2024) 0.511 0.680 0.977 0.976 0.930 0.945 WoW-1-DiT-7B (Chi et al., 2025) 0.522 0.682 0.982 0.985 0.933 0.960 MMIND-VIND-V (Ours) 0.526 0.684 0.986 0.991 0.940 0.963

###### Long-horizon Tasks

Robodreamer (Zhou et al., 2024) 0.464 0.628 0.910 0.918 0.839 0.885 WoW-1-DiT-7B (Chi et al., 2025) 0.476 0.635 0.922 0.929 0.851 0.894 WoW-1-Wan-14B (Chi et al., 2025) 0.498 0.652 0.935 0.950 0.874 0.906

Dreamdojo (Gao et al., 2026) 0.504 0.660 0.944 0.948 0.883 0.909 HunyuanVideo (Kong et al., 2024) 0.487 0.643 0.928 0.952 0.862 0.900 CogVideoX-5B (Yang et al., 2024) 0.484 0.640 0.926 0.947 0.860 0.895 MMIND-VIND-V (Ours) 0.512 0.658 0.955 0.953 0.896 0.924

- Table 2: Comprehensive evaluation of long-horizon tasks. Higher is better; best results are bolded.

Method

PFC Score ↑

Task Success Rate

User Pref. (%) ↑

Dreamdojo (Gao et al., 2026) 0.424 0.333 18 Robodreamer (Zhou et al., 2024) 0.418 0.275 7 WoW-1-DiT-7B (Chi et al., 2025) 0.423 0.322 11 WoW-1-Wan-14B (Chi et al., 2025) 0.420 0.347 14 CogVideoX-5B (Yang et al., 2024) 0.406 0.081 0 HunyuanVideo (Kong et al., 2024) 0.411 0.098 1

MMIND-VIND-V (Ours) 0.462 0.613 49

- Table 3: Task success rates (%) under different training paradigms in the MimicGen simulation benchmark.

Methods Coffee StackThree Square Mean

Base policy 32.6 30.2 20.2 27.7 IL 37.4 36.7 26.1 33.4 MMIND-VIND-V + IL 51.7 48.3 30.4 43.5

YOLO+SAM2: replacing the affordance localizer (Wang et al., 2025) in BSB with a YOLOWorld (Cheng et al., 2024) + SAM2 (Ravi et al.,

- 2024) pipeline; (c) w/o Staged Rollouts: disabling the test-time staged optimization mechanism; (d) Re Gemini w/ Qwen3-VL: substituting Gemini 2.5 Pro in SRH with Qwen3-VL (Bai et al., 2025) to test VLM dependence.

As shown in Table 4, the full model outperforms all variants, confirming the necessity of each component. Removing GRPO yields a clear decline in PFC Score, underscoring the effectiveness of RL-based alignment for physical plausibility. Replacing the affordance module causes a substantial drop in subtask success rate, highlighting its critical role in functional grounding within the BSB. Disabling staged rollouts leads to pronounced degradation, demonstrating that the propose-verify-refine mechanism is essential for mitigating error accu-

Table 4: Ablation study on long-horizon simulation tasks. Visual quality (Aesthetic and Imaging Quality) and functional correctness (PFC Score and subtask Average Success Rate) are reported. Best results are bolded.

Model Variant Visual Quality Functional Correctness

Aesthetic ↑ Imaging ↑ PFC Score ↑ subtask Avg. ↑

- (a) w/o GRPO 0.491 0.675 0.429 0.582
- (b) Re Affordance w/ YOLO 0.498 0.680 0.445 0.455
- (c) w/o Staged Rollouts 0.482 0.671 0.438 0.327
- (d) Re Gemini w/ Qwen3-VL 0.500 0.679 0.452 0.567 MMIND-VIND-V (Full) 0.504 0.684 0.462 0.613

Table 5: Sensitivity analysis of reward weights.

wa wp Aesthetic ↑ Imaging ↑ PFC Score ↑ subtask Avg. ↑

- 0.5 0.2 0.498 0.651 0.449 0.612

- 1.0 0.2 0.504 0.658 0.445 0.613

- 2.0 0.2 0.508 0.657 0.434 0.562

mulation over extended horizons. Substituting the VLM results in only a marginal performance drop, indicating that MMIND-VIND-V benefits from but does not depend on a specific VLM.

Reward Sensitivity Analysis These parameters wa and wp in Eq. 3 serve to normalize reward scales to the same order of magnitude. The sensitivity analysis (Table 5) on long-horizon tasks shows robust performance across a reasonable range and the rationality of our parameter selection.

#### 5 Conclusion

We presented MMIND-VIND-V, a cognitive hierarchical video world model for simulating the physical dynamics of long-horizon robotic manipulation. Inspired by human cognitive pathways, the model integrates a Semantic Reasoning Hub, a Behavioral Semantic Bridge, and a Motor Video Generator. This decoupled architecture separates high-level semantic reasoning from low-level pixel synthesis, ensuring long-horizon coherence, semantic ground-

ing, and physical plausibility. To strictly enforce physical laws and prevent error accumulation, we introduced a GRPO post-training phase guided by a Physical Foresight Coherence reward alongside a test-time optimization strategy utilizing staged visual rollouts. Extensive evaluations demonstrate that MMIND-VIND-V achieves state-of-the-art video synthesis and establishes a highly effective simulation environment for training embodied AI policies.

#### 6 Limitations

Despite generating physically compliant and semantically consistent manipulation videos, MMIND-VIND-V exhibits two primary limitations. First, the Staged Visual Future Rollouts mechanism incurs additional inference overhead, as the proposeverify-refine cycle generates multiple candidate videos per subtask transition. Nonetheless, this fully automated process offsets the cost by eliminating the extensive random sampling and manual cherry-picking required by baseline methods. Second, the framework is sensitive to upstream SRH inaccuracies, where affordance localization failures (Wang et al., 2025) can propagate downstream. To mitigate this, we introduce a VLM-based fallback mechanism that infers target coordinates and destinations via semantic and spatial reasoning. These coordinates are then refined and applied as point prompts for a segmentation model (Ravi et al., 2024) to extract precise masks.

K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter,

- S. Jakubczak, T. Jones, L. Ke, S. Levine, A. LiBell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi,

and 5 others. 2024. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164.

- T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan.

2024. Yolo-world: Real-time open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1234–1245. IEEE.

- X. Chi, P. Jia, C. K. Fan, X. Ju, W. Mi, Z. Qin, K. Zhang,

- W. Tian, K. Ge, H. Li, and 1 others. 2025. Wow: Towards a world omniscient world model through embodied interaction. arXiv preprint arXiv:2509.22642.

G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, L. Marris, and S. Petulla. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. B. Tenenbaum, D. Schuurmans, and P. Abbeel. 2023. Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 9156–9172. Curran Associates, Inc.

- X. Fu, X. Wang, X. Liu, J. Bai, R. Xu, P. Wan, D. Zhang, and D. Lin. 2025. Learning video generation for robotic manipulation with collaborative trajectory control. arXiv preprint arXiv:2506.01943.

S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W. C. Tseng, Y. Dong, K. Mo, C. H. Lin, Q. Ma, S. Nah, L. Magne, J. Xiang, Y. Xie, R. Zheng, D. Niu, Y. L. Tan, K. R. Zentner, and 11 others. 2026. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949.

S. Grillner. 1985. Neurobiological bases of rhythmic motor acts in vertebrates. Science, 228(4696):143– 149.

- Y. Ji, H. Tan, J. Shi, X. Hao, Y. Zhang, H. Zhang, P. Wang, M. Zhao, Y. Mu, P. An, and 1 others. 2025. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1724–1734. IEEE.

#### References

N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, and 1 others. 2025. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575.

M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes,

- M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, and 10 others. 2025. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985.

J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.

S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

M. J. Kim, C. Finn, and P. Liang. 2025. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645.

D. P. Kingma and M. Welling. 2022. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

P. C. Ko, J. Mao, Y. Du, S. H. Sun, and J. B. Tenenbaum. 2023. Learning to act from actionless videos through dense correspondences. arXiv preprint arXiv:2310.08576.

W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, and 1 others. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang. 2025. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470.

Y. Ma, K. Feng, Z. Hu, X. Wang, Y. Wang, M. Zheng,

- X. He, C. Zhu, H. Liu, Y. He, and 1 others. 2025. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869.
- Y. Ma, H. Liu, H. Wang, H. Pan, Y. He, J. Yuan, A. Zeng, C. Cai, H. Y. Shum, W. Liu, and 1 others. 2024. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12. ACM.

- A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. 2023. Mimicgen: A data generation system for scalable robot learning using human demonstrations. In 7th Annual Conference on Robot Learning (CoRL). PMLR.

- S. Meng, Y. Luo, and P. Liu. 2025. Grounding creativity in physics: a brief survey of physical priors in aigc. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI). IJCAI. Article No. 1176.

J. Merel, M. Botvinick, and G. Wayne. 2019. Hierarchical motor control in mammals and machines. Nature Communications, 10(1):5489.

W. Peebles and S. Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205. IEEE.

N. Ravi, V. Gabeur, Y. T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson,

- E. Mintun, J. Pan, K. V. Alwala, N. Carion, C. Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer.

2024. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714.

- T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang.

2024. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159.

H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, 2023. Bridgedata v2: A dataset for robot learning

- A. He, V. Myers, K. Fang, C. Finn, and S. Levine.

at scale. In Conference on Robot Learning (CoRL). PMLR.

Team Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang,

- J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao,
- K. Yan, and 43 others. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

H. Wang, S. Wang, Y. Zhong, Z. Yang, J. Wang, Z. Cui, J. Yuan, Y. Han, M. Liu, and Y. Ma. 2025. Affordance-r1: Reinforcement learning for generalizable affordance reasoning in multimodal large language model. arXiv preprint arXiv:2508.06206.

Z. Wang, Z. Yuan, X. Wang, Y. Li, T. Chen, M. Xia, P. Luo, and Y. Shan. 2024. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11. ACM.

W. Wu, Z. Li, Y. Gu, R. Zhao, Y. He, D. J. Zhang, M. Z. Shou, Y. Li, T. Gao, and D. Zhang. 2024. Draganything: Motion control for anything using entity representation. In European Conference on Computer Vision (ECCV), pages 331–348. Springer.

Z. Xu, Z. Yu, Z. Zhou, J. Zhou, X. Jin, F. T. Hong, X. Ji, J. Zhu, C. Cai, S. Tang, and 1 others. 2025. Hunyuanportrait: Implicit condition control for enhanced portrait animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15909–15919. IEEE.

Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, and 1 others. 2024. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

- J. Yuan, X. Zhang, F. Friedrich, N. Beltran-Velez, M. Hall, R. Askari-Hemmat, X. Han, N. Ballas, M. Drozdzal, and A. Romero-Soriano. 2025. Improving the physics of video generation with vjepa-2 reward signal. arXiv preprint arXiv:2510.21840.

D. Zhang, Z. Li, M. Zhang, J. Zhang, Z. Liu, Y. Yao, H. Xu, J. Zheng, P. Wang, X. Chen, and 1 others. 2025a. From system 1 to system 2: A survey of reasoning large language models. IEEE Transactions on Pattern Analysis and Machine Intelligence.

- K. Zhang, R. Xu, P. Ren, J. Lin, H. Wu, L. Lin, and X. Liang. 2025b. Robridge: A hierarchical architecture bridging cognition and execution for general robotic manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14590–14601. IEEE.

R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595. IEEE.

R. Zhang, Y. Sun, Z. Zhang, J. Li, X. Liu, H. F. Au, H. Guo, and P. Yan. 2025c. Marl-mambacontour: Unleashing multi-agent deep reinforcement learning for active contour optimization in medical image segmentation. In Proceedings of the 33rd ACM International Conference on Multimedia (MM), pages 7815–7824. ACM.

- R. Zhang, J. Zhou, Z. Xu, Z. Liu, J. Huang, M. Zhang,

- Y. Sun, and X. Li. 2025d. Zero-shot 3d-aware trajectory-guided image-to-video generation via testtime training. arXiv preprint arXiv:2509.06723.

Ruicheng Zhang, Guangyu Chen, Zunnan Xu, Zihao Liu, Zhizhou Zhong, Mingyang Zhang, Jun Zhou, and Xiu Li. 2026a. Robostereo: Dual-tower 4d embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639.

Ruicheng Zhang, Kaixi Cong, Jun Zhou, Zhizhou Zhong, Zunnan Xu, Shuiyang Mao, Wei Liu, and Xiu Li. 2026b. Kvpo: Ode-native grpo for autoregressive video alignment via kv semantic exploration. arXiv preprint arXiv:2605.14278.

- Z. Zhang, J. Liao, M. Li, Z. Dai, B. Qiu, S. Zhu, L. Qin, and W. Wang. 2025e. Tora: Trajectory-oriented diffusion transformer for video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2063–

2073. IEEE.

- D. Zheng, Z. Huang, H. Liu, K. Zou, Y. He, F. Zhang, Y. Zhang, J. He, W. S. Zheng, Y. Qiao, and Z. Liu.

2025. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755.

- S. Zhou, Y. Du, J. Chen, Y. Li, D. Y. Yeung, and C. Gan. 2024. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377.

- F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong. 2025a. Irasim: A fine-grained world model for robot manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9834–9844. IEEE.

F. Zhu, Z. Yan, Z. Hong, Q. Shou, X. Ma, and S. Guo. 2025b. Wmpo: World model-based policy optimization for vision-language-action models. arXiv preprint arXiv:2511.09515.

#### A Analysis of Computational Cost and Hyperparameters

This section provides a detailed analysis of the computational cost and key hyperparameters of the MMIND-VIND-V framework. We first examine the world model’s scalability with respect to task length and then present an ablation study on the number of staged visual future rollouts (K) utilized in our testtime optimization strategy.

###### A.1 Scalability with Task Length

To validate MMIND-VIND-V’s efficiency as a world simulator in long-horizon tasks, we measure the simulation time and peak VRAM usage across tasks ranging from one to three sub-tasks. The key metrics are summarized in Table 6 and visualized in Figure 7.

Our findings highlight two critical properties of the proposed architecture. First, the total simulation time scales linearly with the number of subtasks. The average time per sub-task remains constant at approximately 60 seconds 1, demonstrating that our framework’s computational time scales predictably with task length. Second, and crucially, the peak VRAM usage remains constant regardless of the task length. As shown by the consistently sized circles in Figure 7, the peak VRAM remains constant at approximately 70 GB. This constant memory footprint is a direct benefit of our hierarchical and autoregressive design, where memory is allocated for a single sub-task and subsequently reused, enabling highly scalable visual world simulation for long task sequences.

Furthermore, Table 6 details the internal resource distribution. The Motor Video Generator (MVG) constitutes the primary computational bottleneck, consuming the majority of both execution time (approximately 65-70%) and VRAM (approximately 86%). In contrast, the Semantic Reasoning Hub (SRH) planning stage introduces only a minor and stable computational overhead.

###### A.2 Analysis of the Number of Rollout Samples (K)

The Staged Visual Future Rollouts mechanism is governed by a key hyperparameter, K, which defines the number of candidate future trajectories

1This is a reference value using the Gemini-2.5 API (Comanici et al., 2025). The inference speed of the SRH is influenced by multiple factors, including VLM API response time and network latency. Reported times represent pure inference duration, excluding network transmission latency.

Figure 7: Scalability of MMIND-VIND-V. Total generation time (Y-axis) scales linearly with the number of subtasks (X-axis). Circle size represents peak VRAM, which remains constant, demonstrating the memory efficiency of our approach.

simulated at each sub-task transition. While a larger K increases the probability of identifying a physically plausible future, it also incurs greater computational cost. To analyze this trade-off, we conduct an ablation study by varying K from 1 to 5.

The results, visualized in Figure 8, illustrate the relationship between simulation fidelity and computational cost. As shown in the performance radar chart (Figure 8, left), a significant performance uplift is observed as K increases from 1 to 3, demonstrating the effectiveness of the rollout mechanism in filtering out physically incoherent futures. For instance, the Task Success Rate increases from 35.2% at K=1 to 61.3% at K=3. However, this trend exhibits sharply diminishing returns, with only marginal gains when increasing K from 3 to 5. In contrast, the computational cost scales unfavorably with larger K as shown in the bar chart (Figure 8, right). For example, Peak VRAM consumption nearly doubles from 70.1 GB at K=3 to 122.0 GB at K=5. This analysis confirms that K=3 strikes an optimal balance between simulation accuracy and computational efficiency. Therefore, we adopt K=3 as the default setting for all experiments.

#### B Dataset Construction

The Supervised Fine-Tuning (SFT) stage of our training paradigm requires a large-scale dataset of robotic manipulation videos annotated with our structured Behavioral Semantic Bridge (BSB) representation. To this end, we developed an automated pipeline to generate ground-truth BSB annotations from the raw Bridge V2 dataset (Walke

- Table 6: Computational cost as a function of task length. We report total time, average time per sub-task, peak VRAM usage, and the percentage distribution of time and VRAM across the SRH planning and MVG simulation stages.

No. of Total Avg. Time per Peak VRAM Time Dist. (%) VRAM Dist. (%) Sub-tasks Time (s) Sub-task (s) (GB) Plan (SRH) Gen (MVG) Plan (SRH) Gen (MVG)

- 1 60.24 30.14 70.12 36.5% 63.5% 14.1% 85.9%

- 2 123.02 30.60 70.12 34.3% 65.7% 14.4% 85.6%

- 3 181.55 30.85 70.12 32.4% 67.6% 14.0% 86.0%

[Figure 138]

| |
|---|

- Figure 8: Analysis of the trade-off for the number of visual future rollouts (K). (Left) The performance radar chart shows that the overall performance area expands significantly up to K=3 but exhibits diminishing returns thereafter. (Right) The cost chart shows that both time and Peak VRAM increase steadily with K, with memory cost escalating significantly. K=3 (highlighted) is chosen as the optimal balance.

- Table 7: Ablation study on the number of visual future rollouts (K). We evaluate the impact of varying K on functional correctness, visual quality, and computational cost. The setting K=3 (highlighted row) achieves the best balance between performance and efficiency.

Cost Performance Time (s) ↓

K

Peak VRAM (GB) ↓

Task Success Rate (%) ↑

PFC Score ↑

Aesthetic Quality ↑

Imaging Quality ↑

Motion Smoothness ↑

Subject Consistency ↑

- 1 144.5 31.8 35.2 0.405 0.471 0.660 0.931 0.865

- 2 167.1 50.5 51.7 0.428 0.492 0.675 0.946 0.884

- 3 181.6 70.1 61.3 0.445 0.504 0.684 0.953 0.896

- 4 199.7 94.1 62.1 0.447 0.506 0.685 0.954 0.897

- 5 223.4 122.0 62.5 0.450 0.507 0.686 0.954 0.898

- et al., 2023) following the data processing protocol established in (Fu et al., 2025), as illustrated in Figure 9. This pipeline comprises two primary stages: Object Representation Generation and Trajectory Decomposition.

###### B.1 Object Representation Generation

This stage derives the Object Representation (segmentation masks) for both the manipulated object (Mobj) and the robot arm (Mrob) by visually ground-

ing the natural language instructions. For each subtask video and its corresponding instruction (e.g., “pick up the red block”), the extraction process proceeds as follows:

1. Object Identification: A pre-trained VisionLanguage Model (VLM), such as Qwen-VL 2.5 (Bai et al., 2023), extracts the noun phrase corresponding to the object of manipulation (e.g., “red block”) from the instruction.

### Dataset Construction

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

###### Prompt: Place

[Figure 144]

- 2

- 3

the glue stick from the desk into the bowl.

##### Sub-task

(3) Phase Transition Points

###### Video

1

Behavioral Semantic Bridge (BSB)

(Frame1, Frame2)

VLM

Target Object: glue stick

Pre-interaction (arm approaches the object) Interaction (object is manipulated) Post-interaction (arm retracts)

1

(2) Decomposed Collaborative Trajectory

(1) Object Representation

[Figure 145]

2

Ground SAM2

3

Object and Robot Mask

[Figure 146]

{ (x1, y1), (x2, y2), ··· (xn, yn) }

Frames

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Trajectory Tracking

(Mask Center)

- Figure 9: Overview of our automated BSB annotation pipeline. A VLM first extracts the target object from the language prompt, which is then used by Grounded SAM2 (Ren et al., 2024) to generate the (1) Object Representation (masks). Concurrently, trajectory tracking is performed on the object and gripper masks. The trajectory is partitioned based on the object’s motion to produce the (2) Decomposed Collaborative Trajectory and (3) Phase Transition Points. These components collectively form the structured BSB annotation used for SFT.

- 2. Language-Grounded Segmentation: The extracted noun phrase serves as a text prompt for Grounded SAM2 (Ren et al., 2024), an open-vocabulary segmentation model, which generates the pixel-wise segmentation mask for the target object (Mobj) in the initial frame.
- 3. Robot Arm Mask: Because the manipulator’s visual appearance remains consistent across the dataset, we apply a pre-defined template mask to obtain the robot arm segmentation (Mrob).

robot gripper throughout the video sequence. The center point of these masks forms the raw trajectory data. The Decomposed Collaborative Trajectory is then segmented based on the object’s motion:

- 1. Pre-interaction Phase (Tpre): This phase is defined as the sequence of frames from the start of the sub-task until the target object begins to move.
- 2. Interaction Phase (Tinteract): This phase covers the frames during which the target object is in motion.
- 3. Post-interaction Phase (Tpost): This phase begins once the target object comes to rest again and continues until the end of the subtask.

We reserve a randomly sampled 10% of the data for testing and use the remainder for training.

- B.2 Trajectory Decomposition and Phase Segmentation

This stage tracks and partitions the trajectories of the robot and the object into three meaningful phases: pre-interaction, interaction, and postinteraction. The process is based on the motion state of the manipulated object.

The spatiotemporal trajectories of the robot arm and the manipulated object are determined by the paths of their respective mask centroids during these phases. The Phase Transition Points (Fpre,Finteract,Fpost) are defined by the start and end points of the object’s motion. To ensure fidelity of world dynamics, failure cases from the automated pipeline, such as incorrect grounding or

We first employ a video object tracking model, in this case SAM2 (Ravi et al., 2024), to track the segmentation masks of both the target object and the

[Figure 168]

First, the deposition of blue grapes onto a chartreuse plate, followed by the placement of a spoon within a metallic pot. Frames Sub-task 1: Position the blue grapes onto the chartreuse plate. Sub-task 2: lace the spoon within the metallic pot.

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Sub-task1Sub-task2

WoW-14B

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Frames

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Sub-task1Sub-task2

MIND-V

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

- Figure 10: Qualitative comparison on a complex long-horizon task. The model is instructed to first place blue grapes onto a chartreuse plate, and then place a spoon into a metallic pot. (Top) The baseline model, WoW-14B (Chi et al., 2025), exhibits a catastrophic failure in long-horizon reasoning. In Sub-task 1, the grapes levitate without being touched, a clear physical violation. In Sub-task 2, it demonstrates severe semantic grounding error by incorrectly interacting with the plate instead of the instructed spoon, resulting in a complete breakdown of logical coherence. (Bottom) In stark contrast, MMIND-VIND-V successfully executes the full sequence, correctly completing both sub-tasks as instructed. This result validates the efficacy of our hierarchical architecture; the SRH’s explicit planning and the BSB’s structured guidance prevent the semantic drift and error accumulation that plague the baseline, ensuring robust execution of multi-step instructions.

trajectory errors, are flagged for manual correction by human annotators.

#### C Additional Visual Results

This section presents a comprehensive qualitative evaluation comparing MMIND-VIND-V against state-ofthe-art baselines. We analyze performance across three distinct regimes, including multi-stage longhorizon manipulation tasks shown in Figure 10, atomic short-horizon interactions illustrated in Figure 11, and generalization to complex out-ofdistribution (OOD) scenarios with diverse action primitives as depicted in Figure 12. These visualizations substantiate our quantitative findings, showcasing MMIND-VIND-V’s superiority as a physically grounded and logically coherent world simulator.

Long-horizon Tasks. As illustrated in Figure 10, baseline models exhibit a clear breakdown in longhorizon tasks. They not only violate physical common sense within a individual sub-task but also fail to maintain causal consistency across the temporal sequence. For instance, baseline attempts to interact with an object that spontaneously disappeared

in a previous step highlight a profound failure in tracking latent world states. In contrast, MMIND-VIND-V maintains robust logical coherence throughout the sequence. This success stems from our hierarchical design: the Semantic Reasoning Hub (SRH) explicitly decomposes complex instructions into structured sub-tasks, while the Staged Visual Future Rollouts mechanism actively filters physically implausible transitions, effectively preventing longterm semantic drift.

Short-horizon Tasks. The short-horizon examples in Fig. 11 further emphasize the physical fidelity of our approach. Even when baselines correctly identify target objects, they frequently hallucinate physically impossible interactions, such as objects levitating without contact or penetrating solid surfaces. Our GRPO-based physical alignment successfully mitigates these dynamic violations. Furthermore, MMIND-VIND-V excels at interpreting abstract commands (e.g., “clean the floor”) by autonomously inferring the correct latent action sequence (e.g., “grasping a cloth, then wiping”). This demonstrates the SRH’s cognitive capacity to

[Figure 195]

Put the mushrooms into the metal pot.

Frames

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

MIND-VWan-14BMIND-V Hunyuan

Video

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Clean the floor.

Frames

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

- Figure 11: Qualitative comparison on short-horizon tasks. This figure illustrates performance on two distinct single-step instructions. (Top) For “Put the mushrooms into the metal pot”, the baseline (HunyuanVideo (Kong

- et al., 2024)) exhibits physical implausibility, with the mushroom clipping through the pot’s rim. MMIND-VIND-V, in contrast, simulates a physically plausible interaction. (Bottom) For the more abstract instruction “Clean the floor”, the baseline (Wan-14B (Wan et al., 2025)) fails to take any action, demonstrating a lack of semantic grounding. MMIND-VIND-V correctly interprets the instruction, grasps the cloth, and performs a wiping motion, showcasing its superior planning and reasoning capabilities.

translate high-level intent into executable, physically grounded trajectories—a key differentiator from monolithic models that struggle with action grounding.

MVG’s simulation engine creates a highly capable embodied world model.

#### D Network Architecture Details

Complex OOD Scenarios and Diverse Interactive Actions. Fig. 12 demonstrates the robust generalization capabilities that are the hallmark of a scalable world model. Panel (a) illustrates the model’s precision in visually diverse OOD environments. Whether distinguishing a specific object within severe clutter or operating in a stylized scene, MMIND-VIND-V accurately executes the intent. Crucially, the background remains strictly static despite high-frequency textures. This stability is achieved because the Behavioral Semantic Bridge (BSB) decouples semantic intent from visual appearance, providing structural, domaininvariant guidance to the diffusion process. Panel (b) highlights MMIND-VIND-V’s ability to model complex physical state transitions. The model effortlessly simulates interactions with articulated objects (e.g., cabinets) and deformable materials (e.g., cloth). Executing these actions requires deep kinematic and affordance understanding, proving that integrating the SRH’s cognitive reasoning with the

This section details the architectural design and data flow of the Motor Video Generator (MVG). Built upon the CogVideoX-5B (Yang et al., 2024) foundation, the MVG employs a 3D Variational Autoencoder (VAE) and a Diffusion Transformer (DiT) (Peebles and Xie, 2023) backbone. Our primary modification is the conditioning mechanism, which injects explicit spatiotemporal guidance from the Behavioral Semantic Bridge (BSB) to achieve precise kinematic control. The key module specifications and tensor transformations are summarized in Table 8 and elaborated below.

Latent Space Definition. The MVG operates within a compressed latent space parameterized by a pre-trained 3D-VAE (Kingma and Welling, 2022). As detailed in Table 8, the VAE encoder performs spatiotemporal compression on an input video of shape [B,3,T,H,W], producing a compact latent representation of shape [B,16,T/4, H/8,W/8]. The entire forward and reverse diffusion processes are executed within this latent space.

(a) Interaction in Complex Scenarios

Frames

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Grab the bread on the table.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Pick up the

dumplings

from the plate.

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

Pick up the peaches from the plate.

Frames (b) A Variety of Interactive Actions

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Open the

cabinet door.

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Close the microwave door.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Fold the tablecloth.

- Figure 12: Generalization to complex scenarios and diverse manipulation skills. Panel (a) demonstrates the robust generalization ability of MMIND-VIND-V in out-of-distribution (OOD) scenarios. The world model accurately isolates and manipulates targets in cluttered environments (e.g., grabbing bread from a full table or picking a dumpling), as well as in stylistically distinct scenes (e.g., picking peaches in an artistic setting). The domaininvariant BSB representation ensures precise control without disturbing the background fidelity. Panel (b) highlights the simulation of diverse interactive dynamics where the model leverages affordance-aware reasoning to execute physics-compliant interactions. This includes manipulating articulated objects like opening a cabinet or closing a microwave, as well as handling deformable materials such as folding a tablecloth.

Subsequently, the VAE decoder maps the final denoised latent back to the pixel space to render the simulated video.

Diffusion Transformer Backbone. The core of the MVG simulation engine is a 30-block DiT with a hidden dimension of 1920. The latent video is first partitioned into non-overlapping 2×2×2 spatiotemporal patches, which are linearly embedded into a sequence of tokens. This token sequence is augmented with 3D sinusoidal positional encoding and diffusion timestep embeddings prior to being processed by the transformer blocks.

BSB Conditioning Mechanism. To enforce

action adherence, the conditioning mechanism integrates BSB guidance into the DiT backbone via a Guidance Embedding module. The structured semantic information from the BSB is first rasterized into a dense Spatiotemporal Guidance Tensor, which is then processed by a sequence of spatiotemporal convolutions, specifically a 3 × 3 spatial convolution followed by a 1D temporal convolution, yielding a 1920-dimensional feature representation G. Maintaining identical spatiotemporal resolution to the latent video, G is injected into the DiT backbone via additive fusion exclusively within evennumbered transformer blocks (e.g., blocks 0, 2, 4,

- Table 8: Detailed architecture and data flow of the Motor Video Generator (MVG). The table shows the transformation of tensor shapes from the input video and BSB guidance through each major component. Notations: B=Batch Size, C=Channels, T=Temporal Length, H=Height, W=Width, L=Sequence Length, D=Embedding Dimension.

Component Module Input Shape Output Shape Key Hyperparameters 3D VAE

Encoder [B, 3, T, H, W] [B, 16, T/4, H/8, W/8] Latent Channels: 16 Decoder [B, 16, T/4, H/8, W/8] [B, 3, T, H, W] Symmetrical to Encoder

Guidance Tensor

BSB [B, 128, T/4, H/8, W/8] Encodes BSB masks & trajectories into

Guidance Embedding

a dense tensor. Spatial Conv [B, 128, T/4, H/8, W/8] [B, 480, T/4, H/8, W/8] Kernel: 3x3, Stride: 1 Temporal Conv [B, 480, T/4, H/8, W/8] [B, 1920, T/4, H/8, W/8] Kernel: 3 (1D), Stride: 1

Patch Embedding [B, 16, T/4, H/8, W/8] [L, D] Patch Size: 2x2x2, Hidden Dim (D): 1920

DiT Backbone (30 Blocks)

Positional Encoding

[L, D] [L, D] Type: 3D Sinusoidal

Timestep Embedding

Scalar t [1, 512] -

Transformer Blocks

[L, D] [L, D] Layers: 30, Attention Heads: 30, Head Dim: 64

Scheduler DDIM Scalar t Noise Schedule Timesteps: 50, Schedule: Linear

...), leaving odd-numbered blocks unconditioned. This alternating fusion strategy interleaves explicit kinematic control from the BSB with the model’s internal generative priors, steering the denoising trajectory to adhere to the planned trajectory.

#### E Key GRPO Hyperparameters

We initialize GRPO from the SFT checkpoint and train for 1,500 iterations with AdamW using a learning rate of 5 × 10−5, β1 = 0.9, β2 = 0.999, and weight decay 1 × 10−4. For each prompt, the policy samples a group of G = 8 candidate videos to compute group-relative advantages, avoiding an additional value network while preserving stable credit assignment. We adopt a clipping threshold of ϵ = 0.2 in the GRPO objective and set the KL regularization coefficient to β = 0.01, which constrains overly large denoising-policy updates and keeps the updated policy close to the SFT reference model. The reward combines aesthetic and physical terms with wa = 1.0 and wp = 0.2, normalizing the two reward scales while prioritizing visual fidelity and retaining sufficient pressure toward physically coherent dynamics. For the PFC reward, we set the softmax temperature to τ = 0.1 so that optimization emphasizes the least plausible temporal windows without collapsing the reward to a single frame segment. Unless otherwise specified, GRPO uses 50 DDIM denoising steps and

37-frame subtask videos at 480 × 640 resolution, matching the SFT and evaluation settings.

#### F User Study Protocol

To complement automatic metrics, we conduct a user study to evaluate the perceptual quality and task-level correctness of long-horizon manipulation videos. The study includes 30 volunteer participants with backgrounds in computer vision, robotics, or related engineering fields. All participants were recruited on a voluntary basis, and no monetary compensation or other payment was provided. Each participant is shown a randomized subset of long-horizon tasks from the 108sample evaluation set. For each task, we present the original instruction, the initial scene observation, and anonymized videos generated by all compared methods. Method names are hidden, and the display order is randomly shuffled for every question to avoid positional and brand bias.

Participants are asked to judge each video according to three criteria: (1) whether the video follows the language instruction and completes the intended sub-tasks in the correct order; (2) whether the object interactions are physically plausible, including contact, collision, object permanence, and motion continuity; and (3) whether the video is visually coherent, with limited artifacts, flickering, or identity drift. For each task, participants select the

best overall video under these criteria. We aggregate the selections across all valid responses and report the resulting percentage as the User Preference score in Table 2. To ensure response reliability, all videos are shown at the same resolution and playback speed, participants are allowed to replay clips before making a decision, and incomplete responses are excluded from aggregation.

